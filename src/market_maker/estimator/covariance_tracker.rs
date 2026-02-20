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

/// Minimum number of markout observations before using the correction factor.
/// Before this, sigma_correction_factor() returns 1.0 (model-only σ).
const MIN_MARKOUT_OBS: usize = 5;

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

        // Bayesian conjugate normal update:
        // posterior_precision = prior_precision + obs_precision
        // posterior_mean = (prior_precision * prior_mean + obs_precision * obs) / posterior_precision
        let new_precision = self.posterior_precision + SIGMA_OBS_PRECISION;
        let new_mean = (self.posterior_precision * self.posterior_mean
            + SIGMA_OBS_PRECISION * ratio)
            / new_precision;

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
        // Posterior mean, bounded by physical constraints (never negative)
        self.posterior_mean.max(0.1)
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
}
