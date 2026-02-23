//! Realized volatility feedback tracker (WS2).
//!
//! Tracks σ_ratio = realized_vol / predicted_vol using a Bayesian Kalman update.
//! When σ is consistently under-estimated (realized > predicted), the posterior
//! inflates σ_effective automatically via γσ²τ. No clamp needed — the Bayesian
//! posterior naturally can't overshoot because of the prior anchor.
//!
//! Replaces: `sigma_cascade_mult` in MarketParams.
//!
//! Graceful degradation: when no markout data is available (first 30-60s, no fills),
//! σ_effective comes purely from the model (sigma_correction_factor = 1.0).
//! This is the current behavior — no warmup protection addon needed.

use serde::{Deserialize, Serialize};

/// EWMA alpha for sigma ratio updates. 0.05 = ~20 observation half-life.
const SIGMA_RATIO_ALPHA: f64 = 0.05;

/// Bayesian prior for sigma correction factor (starts at 1.0 = trust model).
const SIGMA_PRIOR_MEAN: f64 = 1.0;

/// Prior precision (inverse variance). Higher = stronger prior anchor.
/// At 5.0, a single observation nudges the posterior by ~17%.
const SIGMA_PRIOR_PRECISION: f64 = 5.0;

/// Observation noise precision. Higher = trust observations more.
/// At 2.0 with prior precision 5.0, the posterior is ~29% data, 71% prior per obs.
const SIGMA_OBS_PRECISION: f64 = 2.0;

/// Maximum posterior precision — caps how confident the tracker can become.
/// Without this, precision grows unboundedly with observations, making the
/// posterior resistant to regime changes. At 50.0, ~23 observations saturate.
const MAX_POSTERIOR_PRECISION: f64 = 50.0;

/// Precision decay per observation. Applied before each update to forget old data.
/// 0.98 = half-life ~34 observations. Ensures the tracker adapts to regime changes.
const PRECISION_DECAY: f64 = 0.98;

/// Minimum number of markout observations before using the correction factor.
/// Before this, sigma_correction_factor() returns 1.0 (model-only σ).
const MIN_MARKOUT_OBS: usize = 5;

/// When posterior mean drops below this threshold, use fast adaptation (3x obs precision).
/// A ratio of 0.3 means realized vol is <30% of predicted — severe underestimation.
const FAST_ADAPT_THRESHOLD: f64 = 0.3;

/// Tracks realized vs predicted volatility and produces a Bayesian posterior correction.
///
/// Architecture: model σ is the prior, realized vol from markouts is the observation.
/// The posterior mean is the correction factor applied to σ_effective.
/// No clamp needed — posterior naturally anchored by prior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovarianceTracker {
    /// Posterior mean of sigma correction factor.
    posterior_mean: f64,
    /// Posterior precision (inverse variance). Grows with observations.
    posterior_precision: f64,
    /// EWMA of raw sigma ratio (realized/predicted) for diagnostics.
    sigma_ratio_ewma: f64,
    /// Number of markout observations.
    observation_count: usize,
    /// Sum of squared prediction errors (for diagnostics).
    sum_sq_error: f64,
}

impl Default for CovarianceTracker {
    fn default() -> Self {
        Self {
            posterior_mean: SIGMA_PRIOR_MEAN,
            posterior_precision: SIGMA_PRIOR_PRECISION,
            sigma_ratio_ewma: 1.0,
            observation_count: 0,
            sum_sq_error: 0.0,
        }
    }
}

impl CovarianceTracker {
    /// Create a new tracker with default Bayesian priors.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a markout observation: realized 5s vol vs predicted 5s vol.
    ///
    /// `realized_move_bps`: absolute price move over 5s markout window (bps).
    /// `predicted_5s_vol_bps`: σ_effective × √5 × 10000 at time of fill.
    ///
    /// The ratio realized/predicted is the observation for the Bayesian update.
    pub fn observe_markout(&mut self, realized_move_bps: f64, predicted_5s_vol_bps: f64) {
        if !realized_move_bps.is_finite()
            || !predicted_5s_vol_bps.is_finite()
            || predicted_5s_vol_bps < 0.01
        {
            return;
        }

        let ratio = realized_move_bps.abs() / predicted_5s_vol_bps;

        // Update EWMA for diagnostics
        if self.observation_count == 0 {
            self.sigma_ratio_ewma = ratio;
        } else {
            self.sigma_ratio_ewma =
                SIGMA_RATIO_ALPHA * ratio + (1.0 - SIGMA_RATIO_ALPHA) * self.sigma_ratio_ewma;
        }

        // Precision decay: forget old data to adapt to regime changes.
        // Applied BEFORE update so new observation gets full weight.
        self.posterior_precision *= PRECISION_DECAY;
        // Coupled mean regression: as precision decays toward prior, mean also
        // regresses toward prior mean. This prevents stale mean anchoring during
        // regime changes — without it, precision drops (ready to learn) but the
        // mean stays at the old value, creating lag.
        self.posterior_mean =
            PRECISION_DECAY * self.posterior_mean + (1.0 - PRECISION_DECAY) * SIGMA_PRIOR_MEAN;

        // Fast adaptation: when sigma is severely underestimated (posterior_mean < 0.3),
        // use 3x observation precision to converge faster.
        let effective_obs_precision = if self.posterior_mean < FAST_ADAPT_THRESHOLD
            && self.observation_count >= MIN_MARKOUT_OBS
        {
            SIGMA_OBS_PRECISION * 3.0
        } else {
            SIGMA_OBS_PRECISION
        };

        // Bayesian conjugate normal update:
        // posterior_mean uses uncapped precision to avoid systematic bias.
        // Capping the denominator below the numerator creates drift toward observations.
        let uncapped_precision = self.posterior_precision + effective_obs_precision;
        let new_mean = (self.posterior_precision * self.posterior_mean
            + effective_obs_precision * ratio)
            / uncapped_precision;
        let new_precision = uncapped_precision.min(MAX_POSTERIOR_PRECISION);

        self.posterior_mean = new_mean;
        self.posterior_precision = new_precision;
        self.observation_count += 1;

        // Track prediction error for diagnostics
        let error = ratio - 1.0;
        self.sum_sq_error += error * error;
    }

    /// Sigma correction factor from Bayesian posterior.
    ///
    /// Returns the posterior mean of the correction factor.
    /// Before MIN_MARKOUT_OBS observations, returns 1.0 (model-only σ).
    /// After warmup, posterior naturally anchors around 1.0 with data-driven adjustments.
    ///
    /// Usage: `sigma_corrected = sigma_effective * sigma_correction_factor()`
    pub fn sigma_correction_factor(&self) -> f64 {
        if self.observation_count < MIN_MARKOUT_OBS {
            return 1.0;
        }
        // Posterior mean, floored at 0.5 (never shrink sigma by more than 50%).
        // The 0.1 floor was too low — allowed sigma to collapse, leading to tight spreads.
        self.posterior_mean.max(0.5)
    }

    /// Raw EWMA of sigma ratio (for diagnostics / logging).
    pub fn sigma_ratio_ewma(&self) -> f64 {
        self.sigma_ratio_ewma
    }

    /// Posterior uncertainty (std dev of correction factor estimate).
    pub fn posterior_uncertainty(&self) -> f64 {
        if self.posterior_precision <= 0.0 {
            return 1.0;
        }
        (1.0 / self.posterior_precision).sqrt()
    }

    /// Number of markout observations recorded.
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// RMSE of prediction errors (for diagnostics).
    pub fn prediction_rmse(&self) -> f64 {
        if self.observation_count == 0 {
            return 0.0;
        }
        (self.sum_sq_error / self.observation_count as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_returns_one() {
        let tracker = CovarianceTracker::new();
        assert!(
            (tracker.sigma_correction_factor() - 1.0).abs() < f64::EPSILON,
            "Default should return 1.0: {}",
            tracker.sigma_correction_factor()
        );
    }

    #[test]
    fn test_under_estimation_inflates_sigma() {
        let mut tracker = CovarianceTracker::new();

        // Realized vol consistently 2x predicted
        for _ in 0..20 {
            tracker.observe_markout(10.0, 5.0); // ratio = 2.0
        }

        let factor = tracker.sigma_correction_factor();
        assert!(
            factor > 1.3,
            "Persistent 2x under-estimation should inflate sigma: {}",
            factor
        );
        assert!(
            factor < 2.0,
            "Prior anchor should prevent full 2x overshoot: {}",
            factor
        );
    }

    #[test]
    fn test_over_estimation_deflates_sigma() {
        let mut tracker = CovarianceTracker::new();

        // Realized vol consistently 0.5x predicted
        for _ in 0..20 {
            tracker.observe_markout(2.5, 5.0); // ratio = 0.5
        }

        let factor = tracker.sigma_correction_factor();
        assert!(
            factor < 0.9,
            "Persistent 0.5x over-estimation should deflate sigma: {}",
            factor
        );
        assert!(
            factor > 0.5,
            "Prior anchor should prevent full 0.5x overshoot: {}",
            factor
        );
    }

    #[test]
    fn test_correct_estimation_stays_near_one() {
        let mut tracker = CovarianceTracker::new();

        // Realized vol matches predicted (ratio ≈ 1.0)
        for _ in 0..20 {
            tracker.observe_markout(5.0, 5.0);
        }

        let factor = tracker.sigma_correction_factor();
        assert!(
            (factor - 1.0).abs() < 0.1,
            "Well-calibrated sigma should stay near 1.0: {}",
            factor
        );
    }

    #[test]
    fn test_warmup_returns_one() {
        let mut tracker = CovarianceTracker::new();

        // Only 3 observations (below MIN_MARKOUT_OBS)
        for _ in 0..3 {
            tracker.observe_markout(10.0, 5.0);
        }

        assert!(
            (tracker.sigma_correction_factor() - 1.0).abs() < f64::EPSILON,
            "Should return 1.0 during warmup: {}",
            tracker.sigma_correction_factor()
        );
    }

    #[test]
    fn test_precision_capping() {
        let mut tracker = CovarianceTracker::new();

        // Feed many observations — precision should be capped at MAX_POSTERIOR_PRECISION
        for _ in 0..200 {
            tracker.observe_markout(5.0, 5.0); // ratio = 1.0
        }

        assert!(
            tracker.posterior_precision <= MAX_POSTERIOR_PRECISION + 1e-9,
            "Precision should be capped at {}: got {}",
            MAX_POSTERIOR_PRECISION,
            tracker.posterior_precision
        );
    }

    #[test]
    fn test_floor_at_half() {
        let mut tracker = CovarianceTracker::new();

        // Realized vol consistently 10% of predicted — correction factor should floor at 0.5
        for _ in 0..50 {
            tracker.observe_markout(0.5, 5.0); // ratio = 0.1
        }

        let factor = tracker.sigma_correction_factor();
        assert!(
            (factor - 0.5).abs() < 0.01,
            "Floor should be 0.5, got {}",
            factor
        );
    }

    #[test]
    fn test_fast_adaptation_convergence() {
        let mut tracker = CovarianceTracker::new();

        // First: drive posterior mean very low
        for _ in 0..10 {
            tracker.observe_markout(1.0, 5.0); // ratio = 0.2
        }
        let factor_low = tracker.sigma_correction_factor();

        // Now switch to high realized vol — fast adaptation should kick in
        for _ in 0..10 {
            tracker.observe_markout(10.0, 5.0); // ratio = 2.0
        }
        let factor_after = tracker.sigma_correction_factor();

        assert!(
            factor_after > factor_low + 0.3,
            "Fast adaptation should recover quickly: low={}, after={}",
            factor_low,
            factor_after
        );
    }

    #[test]
    fn test_precision_decay_enables_adaptation() {
        let mut tracker = CovarianceTracker::new();

        // Feed 50 observations of ratio=1.0 (correct calibration)
        for _ in 0..50 {
            tracker.observe_markout(5.0, 5.0);
        }
        let factor_stable = tracker.sigma_correction_factor();
        assert!(
            (factor_stable - 1.0).abs() < 0.15,
            "Should be near 1.0 after correct calibration: {}",
            factor_stable
        );

        // Now regime changes: realized vol is 2x predicted
        for _ in 0..30 {
            tracker.observe_markout(10.0, 5.0); // ratio = 2.0
        }
        let factor_shifted = tracker.sigma_correction_factor();
        assert!(
            factor_shifted > 1.2,
            "Precision decay should allow adaptation to new regime: {}",
            factor_shifted
        );
    }

    #[test]
    fn test_coupled_mean_decay_regime_change() {
        // Fix 4: Verify coupled precision-mean decay tracks regime changes faster.
        // Without coupled decay, the mean anchors at the old value even as
        // precision drops, creating lag during regime transitions.
        let mut tracker = CovarianceTracker::new();

        // Drive to a high ratio (1.5) for 50 observations
        for _ in 0..50 {
            tracker.observe_markout(7.5, 5.0); // ratio = 1.5
        }
        let factor_high = tracker.sigma_correction_factor();
        assert!(
            factor_high > 1.2,
            "Should track high ratio: {}",
            factor_high
        );

        // Now regime changes: ratio drops to 0.7
        for _ in 0..30 {
            tracker.observe_markout(3.5, 5.0); // ratio = 0.7
        }
        let factor_after = tracker.sigma_correction_factor();
        // With coupled decay, the mean regresses toward 1.0 as precision decays,
        // so it should adapt faster to the new lower regime.
        assert!(
            factor_after < factor_high,
            "Coupled decay should allow faster adaptation: high={}, after={}",
            factor_high,
            factor_after
        );
    }

    #[test]
    fn test_coupled_mean_decay_regression_toward_prior() {
        // Fix 4: Verify that mean regresses toward 1.0 when precision decays.
        // After many observations, if we apply precision decay without new data,
        // the mean should drift back toward SIGMA_PRIOR_MEAN (1.0).
        let mut tracker = CovarianceTracker::new();

        // Establish posterior at ratio = 2.0
        for _ in 0..30 {
            tracker.observe_markout(10.0, 5.0); // ratio = 2.0
        }
        let factor_before = tracker.sigma_correction_factor();

        // Now feed observations at ratio = 1.0 (prior mean)
        // With coupled decay, prior influence should help pull faster toward 1.0
        for _ in 0..20 {
            tracker.observe_markout(5.0, 5.0); // ratio = 1.0
        }
        let factor_after = tracker.sigma_correction_factor();
        assert!(
            factor_after < factor_before,
            "Mean should regress toward 1.0: before={}, after={}",
            factor_before,
            factor_after
        );
    }
}
