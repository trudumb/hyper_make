//! Threshold-Dependent Kappa Estimation (TAR Model).
//!
//! First-Principles Grounding:
//!
//! Standard market making assumes linear mean-reversion (constant κ).
//! But crypto markets exhibit **threshold behavior**:
//! - Small deviations: Mean reversion dominates (high κ)
//! - Large deviations: Momentum dominates (κ → 0)
//!
//! This is mathematically modeled as a Threshold AutoRegressive (TAR) process:
//! ```text
//! κ_eff(δ) = κ_base × exp(-decay × max(0, |δ| - threshold))
//! ```
//!
//! ## Intuition
//!
//! When price deviates slightly from fair value, arbitrageurs push it back.
//! But when price moves significantly (news, liquidations), arbitrageurs
//! become market followers, not market makers. They don't fight the trend.
//!
//! ## Implications for Market Making
//!
//! - In mean-reversion regime: Tight spreads, high fill rate expected
//! - In momentum regime: Wide spreads, expect adverse selection

/// Configuration for threshold-dependent kappa.
#[derive(Debug, Clone)]
pub struct ThresholdKappaConfig {
    /// Base kappa for small deviations (mean-reversion strength)
    pub kappa_base: f64,
    /// Deviation threshold (in bps) where momentum begins to dominate
    pub threshold_bps: f64,
    /// Decay rate beyond threshold (how fast kappa decays)
    pub decay_rate: f64,
    /// EMA alpha for tracking recent returns
    pub ema_alpha: f64,
    /// Lookback window for volatility scaling (ms)
    pub lookback_ms: u64,
}

impl Default for ThresholdKappaConfig {
    fn default() -> Self {
        Self {
            kappa_base: 1000.0,  // Typical HIP-3 kappa
            threshold_bps: 15.0, // Start momentum decay at 15 bps
            decay_rate: 0.3,     // Moderate decay
            ema_alpha: 0.1,      // 10% weight on new return
            lookback_ms: 60_000, // 1 minute lookback
        }
    }
}

/// Kappa regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdKappaRegime {
    /// Strong mean reversion (high kappa, tight spreads safe)
    StrongMeanReversion,
    /// Weak mean reversion (moderate kappa)
    WeakMeanReversion,
    /// Weak momentum (low kappa, widen spreads)
    WeakMomentum,
    /// Strong momentum (very low kappa, danger zone)
    StrongMomentum,
}

impl ThresholdKappaRegime {
    /// Get recommended spread multiplier for this regime.
    pub fn spread_multiplier(&self) -> f64 {
        match self {
            ThresholdKappaRegime::StrongMeanReversion => 1.0,
            ThresholdKappaRegime::WeakMeanReversion => 1.2,
            ThresholdKappaRegime::WeakMomentum => 1.5,
            ThresholdKappaRegime::StrongMomentum => 2.0,
        }
    }

    /// Get recommended fill expectation discount.
    pub fn fill_discount(&self) -> f64 {
        match self {
            ThresholdKappaRegime::StrongMeanReversion => 1.0, // Full fill expectation
            ThresholdKappaRegime::WeakMeanReversion => 0.8,
            ThresholdKappaRegime::WeakMomentum => 0.5,
            ThresholdKappaRegime::StrongMomentum => 0.2, // Don't expect fills
        }
    }
}

/// Threshold-dependent kappa estimator.
///
/// First-Principles: κ is not constant. It depends on how far price
/// has deviated from recent equilibrium.
#[derive(Debug, Clone)]
pub struct ThresholdKappa {
    config: ThresholdKappaConfig,
    /// EMA of recent returns (in bps)
    return_ema: f64,
    /// EMA of absolute returns (for volatility scaling)
    abs_return_ema: f64,
    /// Count of observations
    observations: usize,
    /// Last computed effective kappa
    kappa_effective: f64,
    /// Last computed regime
    regime: ThresholdKappaRegime,
}

impl Default for ThresholdKappa {
    fn default() -> Self {
        Self::new(ThresholdKappaConfig::default())
    }
}

impl ThresholdKappa {
    /// Create a new threshold kappa estimator.
    pub fn new(config: ThresholdKappaConfig) -> Self {
        Self {
            kappa_effective: config.kappa_base,
            config,
            return_ema: 0.0,
            abs_return_ema: 10.0, // Start with reasonable default
            observations: 0,
            regime: ThresholdKappaRegime::StrongMeanReversion,
        }
    }

    /// Update with new return observation.
    ///
    /// # Arguments
    /// * `return_bps` - Return since last observation in basis points
    pub fn update(&mut self, return_bps: f64) {
        self.observations += 1;

        // Update return EMA (tracks directional deviation)
        self.return_ema =
            self.config.ema_alpha * return_bps + (1.0 - self.config.ema_alpha) * self.return_ema;

        // Update absolute return EMA (tracks volatility scale)
        self.abs_return_ema = self.config.ema_alpha * return_bps.abs()
            + (1.0 - self.config.ema_alpha) * self.abs_return_ema;

        // Recompute effective kappa and regime
        self.kappa_effective = self.compute_kappa();
        self.regime = self.compute_regime();
    }

    /// Compute effective kappa based on current deviation.
    fn compute_kappa(&self) -> f64 {
        let deviation = self.return_ema.abs();

        // Volatility-scaled threshold
        // Higher volatility → higher threshold (momentum needs bigger moves to trigger)
        let vol_scale = (self.abs_return_ema / 10.0).clamp(0.5, 2.0);
        let effective_threshold = self.config.threshold_bps * vol_scale;

        if deviation < effective_threshold {
            // Mean-reversion regime: full kappa
            self.config.kappa_base
        } else {
            // Momentum regime: kappa decays exponentially
            let excess = deviation - effective_threshold;
            self.config.kappa_base * (-self.config.decay_rate * excess / 100.0).exp()
        }
    }

    /// Compute current regime classification.
    fn compute_regime(&self) -> ThresholdKappaRegime {
        let deviation = self.return_ema.abs();
        let vol_scale = (self.abs_return_ema / 10.0).clamp(0.5, 2.0);
        let t = self.config.threshold_bps * vol_scale;

        if deviation < t * 0.5 {
            ThresholdKappaRegime::StrongMeanReversion
        } else if deviation < t {
            ThresholdKappaRegime::WeakMeanReversion
        } else if deviation < t * 2.0 {
            ThresholdKappaRegime::WeakMomentum
        } else {
            ThresholdKappaRegime::StrongMomentum
        }
    }

    /// Get current effective kappa.
    pub fn kappa(&self) -> f64 {
        self.kappa_effective
    }

    /// Get kappa at a specific deviation level (for simulation).
    pub fn kappa_at_deviation(&self, deviation_bps: f64) -> f64 {
        let vol_scale = (self.abs_return_ema / 10.0).clamp(0.5, 2.0);
        let effective_threshold = self.config.threshold_bps * vol_scale;

        if deviation_bps.abs() < effective_threshold {
            self.config.kappa_base
        } else {
            let excess = deviation_bps.abs() - effective_threshold;
            self.config.kappa_base * (-self.config.decay_rate * excess / 100.0).exp()
        }
    }

    /// Get current regime.
    pub fn regime(&self) -> ThresholdKappaRegime {
        self.regime
    }

    /// Get kappa ratio (effective / base).
    /// Values < 1.0 indicate momentum regime.
    pub fn kappa_ratio(&self) -> f64 {
        self.kappa_effective / self.config.kappa_base
    }

    /// Get current deviation (EMA of returns) in bps.
    pub fn deviation_bps(&self) -> f64 {
        self.return_ema
    }

    /// Get current volatility estimate (EMA of |returns|) in bps.
    pub fn volatility_bps(&self) -> f64 {
        self.abs_return_ema
    }

    /// Get base kappa (max kappa in mean-reversion regime).
    pub fn base_kappa(&self) -> f64 {
        self.config.kappa_base
    }

    /// Set base kappa (for integration with other kappa estimators).
    pub fn set_base_kappa(&mut self, kappa: f64) {
        self.config.kappa_base = kappa;
        self.kappa_effective = self.compute_kappa();
    }

    /// Check if estimator is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.observations >= 10
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.return_ema = 0.0;
        self.abs_return_ema = 10.0;
        self.observations = 0;
        self.kappa_effective = self.config.kappa_base;
        self.regime = ThresholdKappaRegime::StrongMeanReversion;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ThresholdKappaConfig::default();
        assert!((config.kappa_base - 1000.0).abs() < 1e-10);
        assert!((config.threshold_bps - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_initial_state() {
        let tk = ThresholdKappa::default();
        assert_eq!(tk.regime(), ThresholdKappaRegime::StrongMeanReversion);
        assert!((tk.kappa() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_reversion_regime() {
        let mut tk = ThresholdKappa::default();

        // Small returns should stay in mean-reversion
        for _ in 0..20 {
            tk.update(2.0); // Small positive return
        }

        assert!(matches!(
            tk.regime(),
            ThresholdKappaRegime::StrongMeanReversion | ThresholdKappaRegime::WeakMeanReversion
        ));
        assert!(tk.kappa() > 800.0); // Kappa should remain high
    }

    #[test]
    fn test_momentum_regime() {
        let mut tk = ThresholdKappa::new(ThresholdKappaConfig {
            kappa_base: 1000.0,
            threshold_bps: 10.0, // Lower threshold for easier trigger
            decay_rate: 2.0,     // Strong decay for test
            ema_alpha: 0.3,      // Faster adaptation
            ..Default::default()
        });

        // Large returns should trigger momentum regime
        for _ in 0..20 {
            tk.update(100.0); // Very large positive return
        }

        // Should be in momentum regime
        assert!(matches!(
            tk.regime(),
            ThresholdKappaRegime::WeakMomentum | ThresholdKappaRegime::StrongMomentum
        ));
        // Kappa should be lower than base (decayed)
        assert!(
            tk.kappa() < tk.base_kappa(),
            "kappa {} should be less than base {}",
            tk.kappa(),
            tk.base_kappa()
        );
    }

    #[test]
    fn test_kappa_monotonic_decay() {
        let tk = ThresholdKappa::default();

        // Kappa should decrease as deviation increases beyond threshold
        let kappa_5 = tk.kappa_at_deviation(5.0);
        let kappa_20 = tk.kappa_at_deviation(20.0);
        let kappa_50 = tk.kappa_at_deviation(50.0);

        assert!(kappa_5 >= kappa_20);
        assert!(kappa_20 >= kappa_50);
    }

    #[test]
    fn test_regime_spread_multiplier() {
        assert!(
            (ThresholdKappaRegime::StrongMeanReversion.spread_multiplier() - 1.0).abs() < 1e-10
        );
        assert!(ThresholdKappaRegime::StrongMomentum.spread_multiplier() > 1.5);
    }

    #[test]
    fn test_warmup() {
        let mut tk = ThresholdKappa::default();
        assert!(!tk.is_warmed_up());

        for _ in 0..10 {
            tk.update(1.0);
        }

        assert!(tk.is_warmed_up());
    }

    #[test]
    fn test_set_base_kappa() {
        let mut tk = ThresholdKappa::default();
        tk.set_base_kappa(2000.0);

        assert!((tk.base_kappa() - 2000.0).abs() < 1e-10);
    }
}
