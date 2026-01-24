//! Signal Standardizer for Shrinkage Gamma
//!
//! Standardizes raw signals to mean 0, variance 1 using Welford's online algorithm.
//! This ensures that learned weights are comparable across signals with different scales.

/// Online signal standardizer using Welford's algorithm.
///
/// Standardizes signals to approximately N(0, 1) using running statistics.
/// This is essential for the shrinkage gamma model where weights are
/// comparable across signals.
#[derive(Debug, Clone)]
pub struct SignalStandardizer {
    /// Running mean
    mean: f64,

    /// Running M2 (sum of squared deviations)
    m2: f64,

    /// Observation count
    n: usize,

    /// Minimum observations before standardization is valid
    min_observations: usize,

    /// Default mean to use during warmup
    default_mean: f64,

    /// Default std to use during warmup
    default_std: f64,
}

impl SignalStandardizer {
    /// Create a new signal standardizer.
    ///
    /// # Arguments
    /// * `default_mean` - Default mean during warmup
    /// * `default_std` - Default std during warmup
    /// * `min_observations` - Minimum obs before using learned statistics
    pub fn new(default_mean: f64, default_std: f64, min_observations: usize) -> Self {
        Self {
            mean: default_mean,
            m2: 0.0,
            n: 0,
            min_observations,
            default_mean,
            default_std,
        }
    }

    /// Create with typical defaults for a [0, 1] bounded signal.
    /// Uses 5 observations for warmup (down from 20) for faster adaptation.
    pub fn for_bounded_signal() -> Self {
        Self::new(0.5, 0.25, 5)
    }

    /// Create with typical defaults for a ratio signal (e.g., vol_ratio).
    /// Uses 5 observations for warmup (down from 20) for faster adaptation.
    pub fn for_ratio_signal() -> Self {
        Self::new(1.0, 0.5, 5)
    }

    /// Create with typical defaults for a [-1, 1] bounded signal.
    /// Uses 5 observations for warmup (down from 20) for faster adaptation.
    pub fn for_symmetric_signal() -> Self {
        Self::new(0.0, 0.5, 5)
    }

    /// Update statistics and return standardized value.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    pub fn standardize(&mut self, raw: f64) -> f64 {
        // Update running statistics using Welford's algorithm
        self.n += 1;
        let delta = raw - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = raw - self.mean;
        self.m2 += delta * delta2;

        // Compute standard deviation
        let std = self.std();

        // Standardize: z = (x - μ) / σ
        (raw - self.mean) / std.max(1e-9)
    }

    /// Standardize without updating statistics (peek).
    pub fn standardize_peek(&self, raw: f64) -> f64 {
        let std = self.std();
        (raw - self.mean) / std.max(1e-9)
    }

    /// Get current mean.
    pub fn mean(&self) -> f64 {
        if self.n < self.min_observations {
            self.default_mean
        } else {
            self.mean
        }
    }

    /// Get current standard deviation.
    pub fn std(&self) -> f64 {
        // Use default if not enough observations OR less than 2 for variance calc
        if self.n < self.min_observations || self.n < 2 {
            self.default_std
        } else {
            (self.m2 / (self.n - 1) as f64).sqrt().max(1e-9)
        }
    }

    /// Get current variance.
    pub fn variance(&self) -> f64 {
        self.std().powi(2)
    }

    /// Check if standardizer has enough data.
    pub fn is_warmed_up(&self) -> bool {
        self.n >= self.min_observations
    }

    /// Get observation count.
    pub fn observation_count(&self) -> usize {
        self.n
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.mean = self.default_mean;
        self.m2 = 0.0;
        self.n = 0;
    }
}

/// Collection of signal standardizers for all gamma signals.
#[derive(Debug, Clone)]
pub(super) struct SignalStandardizers {
    // Base signals
    pub vol_ratio: SignalStandardizer,
    pub jump_ratio: SignalStandardizer,
    pub inventory: SignalStandardizer,
    pub hawkes: SignalStandardizer,
    pub spread_regime: SignalStandardizer,
    pub cascade: SignalStandardizer,

    // Interaction terms (products of base signals)
    pub vol_x_momentum: SignalStandardizer,
    pub regime_x_inventory: SignalStandardizer,
    pub jump_x_flow: SignalStandardizer,

    // Additional base signals needed for interactions (reserved for future use)
    #[allow(dead_code)]
    pub momentum_abs: SignalStandardizer,
    #[allow(dead_code)]
    pub flow_abs: SignalStandardizer,
}

impl Default for SignalStandardizers {
    fn default() -> Self {
        Self {
            // Vol ratio: typically 0.5 - 2.0, centered at 1.0
            // Reduced warmup to 5 observations for faster adaptation
            vol_ratio: SignalStandardizer::new(1.0, 0.5, 5),

            // Jump ratio: typically 1.0 - 5.0, centered at 1.5
            // Reduced warmup to 5 observations for faster adaptation
            jump_ratio: SignalStandardizer::new(1.5, 1.0, 5),

            // Inventory utilization: 0.0 - 1.0
            inventory: SignalStandardizer::for_bounded_signal(),

            // Hawkes percentile: 0.0 - 1.0
            hawkes: SignalStandardizer::for_bounded_signal(),

            // Spread regime: -1 to 1 (encoded)
            spread_regime: SignalStandardizer::for_symmetric_signal(),

            // Cascade severity: 0.0 - 1.0
            cascade: SignalStandardizer::for_bounded_signal(),

            // === Interaction terms ===
            // These are products of standardized signals, so they're ~N(0,1) products
            // For z1 ~ N(0,1) and z2 ~ N(0,1), E[z1*z2] = 0, Var[z1*z2] ≈ 1
            vol_x_momentum: SignalStandardizer::new(0.0, 1.0, 5),
            regime_x_inventory: SignalStandardizer::new(0.0, 1.0, 5),
            jump_x_flow: SignalStandardizer::new(0.0, 1.0, 5),

            // Additional base signals for interactions
            // Momentum absolute value: 0.0 - 20 bps typically
            momentum_abs: SignalStandardizer::new(5.0, 5.0, 5),
            // Flow imbalance absolute: 0.0 - 1.0
            flow_abs: SignalStandardizer::for_bounded_signal(),
        }
    }
}

impl SignalStandardizers {
    /// Standardize a signal by name.
    pub(super) fn standardize(&mut self, signal: &super::config::GammaSignal, raw: f64) -> f64 {
        match signal {
            super::config::GammaSignal::VolatilityRatio => self.vol_ratio.standardize(raw),
            super::config::GammaSignal::JumpRatio => self.jump_ratio.standardize(raw),
            super::config::GammaSignal::InventoryUtilization => self.inventory.standardize(raw),
            super::config::GammaSignal::HawkesIntensity => self.hawkes.standardize(raw),
            super::config::GammaSignal::SpreadRegime => self.spread_regime.standardize(raw),
            super::config::GammaSignal::CascadeSeverity => self.cascade.standardize(raw),
            // Interaction terms - raw is already the product
            super::config::GammaSignal::VolatilityXMomentum => self.vol_x_momentum.standardize(raw),
            super::config::GammaSignal::RegimeXInventory => {
                self.regime_x_inventory.standardize(raw)
            }
            super::config::GammaSignal::JumpXFlow => self.jump_x_flow.standardize(raw),
        }
    }

    /// Standardize a signal without updating (peek).
    pub(super) fn standardize_peek(&self, signal: &super::config::GammaSignal, raw: f64) -> f64 {
        match signal {
            super::config::GammaSignal::VolatilityRatio => self.vol_ratio.standardize_peek(raw),
            super::config::GammaSignal::JumpRatio => self.jump_ratio.standardize_peek(raw),
            super::config::GammaSignal::InventoryUtilization => {
                self.inventory.standardize_peek(raw)
            }
            super::config::GammaSignal::HawkesIntensity => self.hawkes.standardize_peek(raw),
            super::config::GammaSignal::SpreadRegime => self.spread_regime.standardize_peek(raw),
            super::config::GammaSignal::CascadeSeverity => self.cascade.standardize_peek(raw),
            super::config::GammaSignal::VolatilityXMomentum => {
                self.vol_x_momentum.standardize_peek(raw)
            }
            super::config::GammaSignal::RegimeXInventory => {
                self.regime_x_inventory.standardize_peek(raw)
            }
            super::config::GammaSignal::JumpXFlow => self.jump_x_flow.standardize_peek(raw),
        }
    }

    /// Standardize momentum_abs (helper for interaction terms).
    #[allow(dead_code)]
    pub(super) fn standardize_momentum_abs(&mut self, raw: f64) -> f64 {
        self.momentum_abs.standardize(raw)
    }

    /// Standardize flow_abs (helper for interaction terms).
    #[allow(dead_code)]
    pub(super) fn standardize_flow_abs(&mut self, raw: f64) -> f64 {
        self.flow_abs.standardize(raw)
    }

    /// Check if all standardizers are warmed up.
    pub(super) fn all_warmed_up(&self) -> bool {
        self.vol_ratio.is_warmed_up()
            && self.jump_ratio.is_warmed_up()
            && self.inventory.is_warmed_up()
            && self.hawkes.is_warmed_up()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standardization_mean_zero() {
        let mut s = SignalStandardizer::new(0.0, 1.0, 5);

        // Feed in values centered at 10
        for i in 0..100 {
            let raw = 10.0 + (i as f64 % 5.0) - 2.0; // 8, 9, 10, 11, 12
            s.standardize(raw);
        }

        // Mean should be close to 10
        assert!(
            (s.mean() - 10.0).abs() < 0.1,
            "Mean should be ~10, got {}",
            s.mean()
        );

        // Standardized mean should be ~0
        let z = s.standardize_peek(s.mean());
        assert!(z.abs() < 0.1, "Standardized mean should be ~0, got {}", z);
    }

    #[test]
    fn test_standardization_unit_variance() {
        let mut s = SignalStandardizer::new(0.0, 1.0, 5);

        // Feed in standard normal samples (approximated)
        let samples = [
            -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.8, 0.3, -0.2, 0.7, -1.2, 0.9, -0.6, 0.4,
        ];

        for &x in &samples {
            s.standardize(x);
        }

        // Std should be close to empirical std of samples
        let std = s.std();
        assert!(
            std > 0.5 && std < 1.5,
            "Std should be reasonable, got {}",
            std
        );
    }

    #[test]
    fn test_warmup_uses_defaults() {
        let s = SignalStandardizer::new(5.0, 2.0, 20);

        assert_eq!(s.mean(), 5.0);
        assert_eq!(s.std(), 2.0);
        assert!(!s.is_warmed_up());
    }

    #[test]
    fn test_warmup_transition() {
        let mut s = SignalStandardizer::new(0.0, 1.0, 10);

        for i in 0..9 {
            s.standardize(i as f64);
        }
        assert!(!s.is_warmed_up());

        s.standardize(9.0);
        assert!(s.is_warmed_up());

        // Now should use learned statistics
        assert!((s.mean() - 4.5).abs() < 0.1);
    }
}
