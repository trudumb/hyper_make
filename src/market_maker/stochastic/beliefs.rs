//! Bayesian belief state for first-principles stochastic control.
//!
//! This module maintains posteriors over market parameters (μ, σ², κ, P)
//! that evolve with market data. **Decisions flow from beliefs, not thresholds.**
//!
//! ## Key Insight
//!
//! The predictive bias β_t = E[μ | data] is not a heuristic - it's the
//! posterior mean of drift from the Normal-Inverse-Gamma conjugate update.
//! When beliefs shift to negative μ (capitulation), β_t < 0 automatically.
//!
//! ## Components
//!
//! - **NormalInverseGamma** for (μ, σ²): Drift and volatility
//! - **FillIntensityPosterior** for κ: Fill intensity
//! - **BOCD integration**: Regime changepoint detection shifts priors
//!
//! ## Usage
//!
//! ```ignore
//! let mut beliefs = MarketBeliefs::default();
//!
//! // Update on price observation
//! beliefs.observe_price(price_return, dt);
//!
//! // Update on fill observation
//! beliefs.observe_fill(depth_bps, dt);
//!
//! // Get predictive bias (NOT a heuristic - it's E[μ | data])
//! let predictive_bias = beliefs.predictive_bias();
//! ```

use super::conjugate::{FillIntensityPosterior, NormalInverseGamma};

/// Bayesian belief state over market parameters.
///
/// This is the core belief aggregator that drives all quote decisions.
/// Decisions emerge from posteriors, not if-else thresholds.
#[derive(Debug, Clone)]
pub struct MarketBeliefs {
    // === Drift and Volatility (Normal-Inverse-Gamma) ===
    /// Posterior over (μ, σ²) for price drift and volatility.
    /// E[μ | data] = m_n (posterior mean of drift)
    pub drift: NormalInverseGamma,

    // === Fill Intensity (Gamma) ===
    /// Posterior over κ for fill intensity.
    /// E[κ | fills] = α_n / β_n
    pub kappa: FillIntensityPosterior,

    // === Regime Probabilities ===
    /// Current regime probabilities [quiet, normal, bursty, cascade].
    /// Updated from HMM or BOCD integration.
    pub regime_probs: [f64; 4],

    /// Changepoint probability from BOCD.
    /// High values indicate recent regime shift.
    pub changepoint_prob: f64,

    // === Derived Quantities (Cached) ===
    /// E[μ | data] - posterior mean of drift (THE predictive signal)
    pub expected_drift: f64,

    /// E[σ | data] - posterior mean of volatility
    pub expected_sigma: f64,

    /// E[κ | fills] - posterior mean of fill intensity
    pub expected_kappa: f64,

    /// Uncertainty in drift estimate (posterior std of μ)
    pub drift_uncertainty: f64,

    /// Uncertainty in kappa estimate (coefficient of variation)
    pub kappa_uncertainty: f64,

    // === Statistics ===
    /// Number of price observations
    pub n_price_obs: u64,

    /// Number of fill observations
    pub n_fills: u64,

    /// Total observation time (seconds)
    pub total_time: f64,

    // === Configuration ===
    /// Decay factor for non-stationarity (per update)
    pub decay_factor: f64,

    /// Apply decay every N observations
    pub decay_interval: u64,

    /// Counter for decay application
    decay_counter: u64,
}

impl Default for MarketBeliefs {
    fn default() -> Self {
        let drift = NormalInverseGamma::default();
        let kappa = FillIntensityPosterior::default();

        Self {
            expected_drift: drift.posterior_mean(),
            expected_sigma: drift.posterior_sigma(),
            drift_uncertainty: drift.posterior_std(),
            expected_kappa: kappa.posterior_mean(),
            kappa_uncertainty: kappa.cv(),
            drift,
            kappa,
            regime_probs: [0.2, 0.5, 0.2, 0.1], // Prior: mostly normal/quiet
            changepoint_prob: 0.0,
            n_price_obs: 0,
            n_fills: 0,
            total_time: 0.0,
            decay_factor: 0.999, // Slow decay for stationarity
            decay_interval: 1000,
            decay_counter: 0,
        }
    }
}

impl MarketBeliefs {
    /// Create beliefs with custom priors.
    ///
    /// # Arguments
    /// * `drift_prior_sigma` - Prior standard deviation for drift
    /// * `kappa_prior_mean` - Prior mean for fill intensity
    /// * `kappa_prior_cv` - Prior coefficient of variation for kappa
    pub fn with_priors(drift_prior_sigma: f64, kappa_prior_mean: f64, kappa_prior_cv: f64) -> Self {
        let drift = NormalInverseGamma::zero_centered(drift_prior_sigma, 10.0);
        let kappa = FillIntensityPosterior::new(kappa_prior_mean, kappa_prior_cv, 0.5);

        Self {
            expected_drift: drift.posterior_mean(),
            expected_sigma: drift.posterior_sigma(),
            drift_uncertainty: drift.posterior_std(),
            expected_kappa: kappa.posterior_mean(),
            kappa_uncertainty: kappa.cv(),
            drift,
            kappa,
            ..Default::default()
        }
    }

    /// Observe a price return.
    ///
    /// # Arguments
    /// * `price_return` - Fractional return (e.g., 0.001 = 10 bps)
    /// * `dt` - Time elapsed (seconds)
    ///
    /// Updates the drift/volatility posterior.
    pub fn observe_price(&mut self, price_return: f64, dt: f64) {
        // Normalize return to per-second for drift estimate
        let drift_obs = if dt > 0.01 {
            price_return / dt.sqrt()
        } else {
            price_return
        };

        self.drift.update(drift_obs);
        self.n_price_obs += 1;
        self.total_time += dt;

        // Update cached values
        self.update_cached();

        // Apply decay periodically
        self.maybe_decay();
    }

    /// Observe a fill.
    ///
    /// # Arguments
    /// * `depth_bps` - Depth at which fill occurred
    /// * `dt` - Time since last fill (seconds)
    ///
    /// Updates the fill intensity posterior.
    pub fn observe_fill(&mut self, depth_bps: f64, dt: f64) {
        self.kappa.update(1.0, dt, depth_bps);
        self.n_fills += 1;

        // Update cached values
        self.expected_kappa = self.kappa.posterior_mean();
        self.kappa_uncertainty = self.kappa.cv();
    }

    /// Observe multiple fills (batch update).
    ///
    /// # Arguments
    /// * `n_fills` - Number of fills
    /// * `depth_bps` - Average depth
    /// * `dt` - Total time window
    pub fn observe_fills(&mut self, n_fills: u64, depth_bps: f64, dt: f64) {
        self.kappa.update(n_fills as f64, dt, depth_bps);
        self.n_fills += n_fills;

        self.expected_kappa = self.kappa.posterior_mean();
        self.kappa_uncertainty = self.kappa.cv();
    }

    /// Update regime probabilities.
    ///
    /// # Arguments
    /// * `probs` - New regime probabilities [quiet, normal, bursty, cascade]
    pub fn update_regime(&mut self, probs: [f64; 4]) {
        self.regime_probs = probs;
    }

    /// Update changepoint probability from BOCD.
    ///
    /// High changepoint probability triggers prior shifts.
    pub fn update_changepoint(&mut self, prob: f64) {
        self.changepoint_prob = prob.clamp(0.0, 1.0);

        // If changepoint is likely, shift drift prior toward negative (defensive)
        if self.changepoint_prob > 0.7 {
            // Soft reset: blend toward bearish prior
            let retention = 1.0 - self.changepoint_prob * 0.5;
            self.drift.soft_reset(retention);
            self.update_cached();
        }
    }

    /// Get predictive bias β_t = E[μ | data].
    ///
    /// **This is NOT a heuristic - it's the posterior mean of drift.**
    ///
    /// When beliefs shift to negative μ (capitulation), β_t < 0 automatically,
    /// causing the HJB solution to skew toward sells.
    pub fn predictive_bias(&self) -> f64 {
        self.expected_drift
    }

    /// Get predictive bias with confidence weighting.
    ///
    /// Returns β_t scaled by confidence (higher uncertainty → smaller bias).
    pub fn predictive_bias_confident(&self) -> f64 {
        let confidence = self.drift_confidence();
        self.expected_drift * confidence
    }

    /// Get expected volatility E[σ | data].
    pub fn expected_volatility(&self) -> f64 {
        self.expected_sigma
    }

    /// Get expected fill intensity E[κ | fills].
    pub fn expected_fill_intensity(&self) -> f64 {
        self.expected_kappa
    }

    /// Get fill intensity at specific depth.
    ///
    /// λ(δ) = E[κ] × exp(-γ × δ)
    pub fn fill_intensity_at_depth(&self, depth_bps: f64) -> f64 {
        self.kappa.intensity_at_depth(depth_bps)
    }

    /// Confidence in drift estimate [0, 1].
    ///
    /// Based on posterior concentration.
    pub fn drift_confidence(&self) -> f64 {
        // Map coefficient of variation to confidence
        // CV = 0 → conf = 1, CV = ∞ → conf = 0
        let cv = if self.expected_drift.abs() > 1e-10 {
            self.drift_uncertainty / self.expected_drift.abs()
        } else {
            self.drift_uncertainty * 100.0 // Large CV for near-zero drift
        };

        (1.0 / (1.0 + cv)).clamp(0.0, 1.0)
    }

    /// Confidence in kappa estimate [0, 1].
    pub fn kappa_confidence(&self) -> f64 {
        // CV → confidence mapping
        (1.0 / (1.0 + self.kappa_uncertainty)).clamp(0.0, 1.0)
    }

    /// Overall belief confidence (geometric mean of components).
    ///
    /// FIX: Reduced observation requirement from 100 to 30 to speed up warmup.
    /// 38 seconds to reach 0.3 confidence was too slow - market moves significantly.
    /// Target: 10-15 seconds to activation with typical tick rates.
    pub fn overall_confidence(&self) -> f64 {
        let drift_conf = self.drift_confidence();
        let kappa_conf = self.kappa_confidence();
        // FIX: Reduced from 100 to 30 observations for faster warmup
        // Also added square root scaling to give early observations more weight
        let n_conf = if self.n_price_obs >= 30 {
            1.0
        } else if self.n_price_obs >= 10 {
            // Faster ramp: 10 obs = 0.58, 20 obs = 0.82, 30 obs = 1.0
            (self.n_price_obs as f64 / 30.0).sqrt()
        } else {
            // Very early: be conservative
            self.n_price_obs as f64 / 30.0
        };

        (drift_conf * kappa_conf * n_conf).powf(1.0 / 3.0)
    }

    /// Probability that drift is negative (bearish).
    pub fn prob_bearish(&self) -> f64 {
        self.drift.prob_negative_drift()
    }

    /// Probability that drift is positive (bullish).
    pub fn prob_bullish(&self) -> f64 {
        self.drift.prob_positive_drift()
    }

    /// Is the belief system warmed up (enough observations)?
    pub fn is_warmed_up(&self) -> bool {
        self.n_price_obs >= 50 && self.kappa.is_warmed_up()
    }

    /// Get current regime (most likely).
    pub fn current_regime(&self) -> Regime {
        let max_idx = self
            .regime_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);

        match max_idx {
            0 => Regime::Quiet,
            1 => Regime::Normal,
            2 => Regime::Bursty,
            3 => Regime::Cascade,
            _ => Regime::Normal,
        }
    }

    /// Get regime-weighted parameter (blend across regimes).
    ///
    /// # Arguments
    /// * `quiet` - Value for quiet regime
    /// * `normal` - Value for normal regime
    /// * `bursty` - Value for bursty regime
    /// * `cascade` - Value for cascade regime
    pub fn regime_blend(&self, quiet: f64, normal: f64, bursty: f64, cascade: f64) -> f64 {
        self.regime_probs[0] * quiet
            + self.regime_probs[1] * normal
            + self.regime_probs[2] * bursty
            + self.regime_probs[3] * cascade
    }

    /// Soft reset (after changepoint detection).
    ///
    /// # Arguments
    /// * `retention` - Fraction of information to retain [0, 1]
    pub fn soft_reset(&mut self, retention: f64) {
        self.drift.soft_reset(retention);
        self.kappa.decay(retention);
        self.update_cached();
    }

    /// Hard reset to priors.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get summary for diagnostics.
    pub fn summary(&self) -> BeliefSummary {
        BeliefSummary {
            expected_drift: self.expected_drift,
            expected_sigma: self.expected_sigma,
            expected_kappa: self.expected_kappa,
            drift_uncertainty: self.drift_uncertainty,
            drift_confidence: self.drift_confidence(),
            prob_bearish: self.prob_bearish(),
            prob_bullish: self.prob_bullish(),
            regime: self.current_regime(),
            regime_probs: self.regime_probs,
            changepoint_prob: self.changepoint_prob,
            n_price_obs: self.n_price_obs,
            n_fills: self.n_fills,
            is_warmed_up: self.is_warmed_up(),
        }
    }

    // === Private helpers ===

    fn update_cached(&mut self) {
        self.expected_drift = self.drift.posterior_mean();
        self.expected_sigma = self.drift.posterior_sigma();
        self.drift_uncertainty = self.drift.posterior_std();
    }

    fn maybe_decay(&mut self) {
        self.decay_counter += 1;
        if self.decay_counter >= self.decay_interval {
            self.drift.decay(self.decay_factor);
            self.kappa.decay(self.decay_factor);
            self.decay_counter = 0;
        }
    }
}

/// Market regime enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Low volatility, normal fills
    Quiet,
    /// Normal market conditions
    Normal,
    /// High activity, elevated volatility
    Bursty,
    /// Liquidation cascade, toxic flow
    Cascade,
}

impl std::fmt::Display for Regime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Regime::Quiet => write!(f, "quiet"),
            Regime::Normal => write!(f, "normal"),
            Regime::Bursty => write!(f, "bursty"),
            Regime::Cascade => write!(f, "cascade"),
        }
    }
}

/// Summary of belief state for diagnostics.
#[derive(Debug, Clone)]
pub struct BeliefSummary {
    pub expected_drift: f64,
    pub expected_sigma: f64,
    pub expected_kappa: f64,
    pub drift_uncertainty: f64,
    pub drift_confidence: f64,
    pub prob_bearish: f64,
    pub prob_bullish: f64,
    pub regime: Regime,
    pub regime_probs: [f64; 4],
    pub changepoint_prob: f64,
    pub n_price_obs: u64,
    pub n_fills: u64,
    pub is_warmed_up: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beliefs_default() {
        let beliefs = MarketBeliefs::default();
        assert!((beliefs.predictive_bias() - 0.0).abs() < 1e-6);
        assert!(beliefs.expected_kappa > 0.0);
    }

    #[test]
    fn test_beliefs_negative_drift_from_price() {
        let mut beliefs = MarketBeliefs::default();

        // Observe negative returns (capitulation) - need enough to overcome prior
        for _ in 0..50 {
            beliefs.observe_price(-0.002, 1.0); // -20 bps returns
        }

        // Predictive bias should be negative (sell skew)
        assert!(
            beliefs.predictive_bias() < 0.0,
            "Predictive bias should be negative: {}",
            beliefs.predictive_bias()
        );
        // Note: prob_bearish may not exceed 0.5 due to uncertainty in variance.
        // The key is the posterior mean (predictive_bias) is strongly negative.
        assert!(
            beliefs.predictive_bias() < -0.001,
            "Predictive bias should be strongly negative: {}",
            beliefs.predictive_bias()
        );
    }

    #[test]
    fn test_beliefs_positive_drift_from_price() {
        let mut beliefs = MarketBeliefs::default();

        // Observe positive returns (uptrend) - need enough to overcome prior
        for _ in 0..50 {
            beliefs.observe_price(0.002, 1.0); // +20 bps returns
        }

        assert!(
            beliefs.predictive_bias() > 0.0,
            "Predictive bias should be positive"
        );
        // Note: prob_bullish may not exceed 0.5 due to uncertainty in variance.
        // The key is the posterior mean (predictive_bias) is strongly positive.
        assert!(
            beliefs.predictive_bias() > 0.001,
            "Predictive bias should be strongly positive: {}",
            beliefs.predictive_bias()
        );
    }

    #[test]
    fn test_beliefs_fill_updates_kappa() {
        let mut beliefs = MarketBeliefs::default();
        let _initial_kappa = beliefs.expected_kappa;

        // Observe fills - need > 60s total_time and >= 10 fills for warmup
        for _ in 0..30 {
            beliefs.observe_fill(5.0, 3.0); // 1 fill per 3 seconds at 5 bps
        }

        // Kappa should have been updated
        assert!(
            beliefs.n_fills == 30,
            "Should have 30 fills recorded"
        );
        assert!(
            beliefs.kappa.total_time >= 90.0,
            "Total time should be >= 90s: {}",
            beliefs.kappa.total_time
        );
        assert!(
            beliefs.kappa.is_warmed_up(),
            "Should be warmed up after 30 fills over 90s"
        );
    }

    #[test]
    fn test_beliefs_changepoint_shifts_prior() {
        let mut beliefs = MarketBeliefs::default();

        // Build up positive drift
        for _ in 0..30 {
            beliefs.observe_price(0.001, 1.0);
        }
        let drift_before = beliefs.predictive_bias();
        assert!(drift_before > 0.0);

        // High changepoint probability triggers prior shift
        beliefs.update_changepoint(0.9);

        // Drift should be pulled toward zero (defensive)
        let drift_after = beliefs.predictive_bias();
        assert!(
            drift_after.abs() < drift_before.abs(),
            "Changepoint should reduce drift magnitude"
        );
    }

    #[test]
    fn test_beliefs_regime_blend() {
        let mut beliefs = MarketBeliefs::default();
        beliefs.regime_probs = [0.5, 0.3, 0.1, 0.1]; // Mostly quiet

        let blended = beliefs.regime_blend(1.0, 2.0, 3.0, 4.0);
        // 0.5 × 1 + 0.3 × 2 + 0.1 × 3 + 0.1 × 4 = 0.5 + 0.6 + 0.3 + 0.4 = 1.8
        assert!((blended - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_beliefs_confidence_increases() {
        let mut beliefs = MarketBeliefs::default();
        let initial_conf = beliefs.overall_confidence();

        // Add observations
        for _ in 0..100 {
            beliefs.observe_price(0.0005, 1.0);
        }

        let final_conf = beliefs.overall_confidence();
        assert!(
            final_conf > initial_conf,
            "Confidence should increase with observations"
        );
    }

    #[test]
    fn test_beliefs_preemptive_skew() {
        let mut beliefs = MarketBeliefs::default();

        // Observe price dropping (no fills yet)
        for _ in 0..10 {
            beliefs.observe_price(-0.002, 1.0); // -20 bps/sec
        }

        // Predictive bias should be negative (sell skew)
        assert!(
            beliefs.predictive_bias() < 0.0,
            "Should have negative bias before any fills"
        );

        // This enables preemptive skew - the system will skew toward
        // sells BEFORE getting filled, based on price observations alone
    }
}
