//! Entropy distribution parameter set.

/// Entropy-based stochastic order distribution parameters.
///
/// This system completely replaces the concentration fallback mechanism
/// with a principled, diversity-preserving approach using information-theoretic
/// entropy constraints.
///
/// Key features:
/// - **Minimum entropy floor**: Distribution NEVER drops below H_min
/// - **Softmax temperature control**: Smooth transitions instead of hard cutoffs
/// - **Thompson sampling**: Stochastic allocation prevents predictability
/// - **Dirichlet smoothing**: Prior regularization prevents zero allocations
#[derive(Debug, Clone)]
pub struct EntropyDistributionParams {
    /// Minimum entropy floor (bits).
    /// H_min = 1.5 → at least exp(1.5) ≈ 4.5 effective levels always active.
    pub min_entropy: f64,

    /// Base temperature for softmax.
    /// Higher = more uniform distribution, lower = more concentrated.
    pub base_temperature: f64,

    /// Minimum allocation floor per level (prevents zero allocations).
    pub min_allocation_floor: f64,

    /// Number of Thompson samples for stochastic allocation.
    pub thompson_samples: usize,

    /// Market toxicity (RV/BV ratio) for temperature scaling.
    pub toxicity: f64,

    /// Volatility ratio vs baseline for temperature scaling.
    pub volatility_ratio: f64,

    /// Cascade severity for temperature scaling.
    pub cascade_severity: f64,
}

impl Default for EntropyDistributionParams {
    fn default() -> Self {
        Self {
            min_entropy: 1.5,
            base_temperature: 1.0,
            min_allocation_floor: 0.02,
            thompson_samples: 5,
            toxicity: 1.0,
            volatility_ratio: 1.0,
            cascade_severity: 0.0,
        }
    }
}

impl EntropyDistributionParams {
    /// Convert to MarketRegime for the entropy distributor.
    pub fn to_market_regime(&self) -> crate::market_maker::quoting::MarketRegime {
        crate::market_maker::quoting::MarketRegime {
            toxicity: self.toxicity,
            volatility_ratio: self.volatility_ratio,
            cascade_severity: self.cascade_severity,
            book_imbalance: 0.0, // Not used directly in entropy
        }
    }

    /// Convert to EntropyDistributionConfig for the distributor.
    pub fn to_config(&self) -> crate::market_maker::quoting::EntropyDistributionConfig {
        crate::market_maker::quoting::EntropyDistributionConfig {
            min_entropy: self.min_entropy,
            base_temperature: self.base_temperature,
            min_allocation_floor: self.min_allocation_floor,
            thompson_samples: self.thompson_samples,
            ..Default::default()
        }
    }
}
