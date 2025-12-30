//! Mock estimator for testing.

use super::{MarketEstimator, VolatilityRegime};

/// Default mock estimator for testing.
///
/// Returns neutral/safe values for all parameters.
#[derive(Debug, Clone, Default)]
pub struct MockEstimator {
    pub sigma: f64,
    pub kappa: f64,
    pub is_warmed: bool,
}

impl MockEstimator {
    /// Create a mock estimator with default safe values.
    pub fn new() -> Self {
        Self {
            sigma: 0.0002, // 2 bps/sec
            kappa: 500.0,  // Moderate fill rate
            is_warmed: true,
        }
    }

    /// Create with custom sigma and kappa.
    pub fn with_params(sigma: f64, kappa: f64) -> Self {
        Self {
            sigma,
            kappa,
            is_warmed: true,
        }
    }
}

impl MarketEstimator for MockEstimator {
    fn sigma_clean(&self) -> f64 {
        self.sigma
    }
    fn sigma_total(&self) -> f64 {
        self.sigma
    }
    fn sigma_effective(&self) -> f64 {
        self.sigma
    }
    fn volatility_regime(&self) -> VolatilityRegime {
        VolatilityRegime::Normal
    }
    fn kappa(&self) -> f64 {
        self.kappa
    }
    fn kappa_bid(&self) -> f64 {
        self.kappa
    }
    fn kappa_ask(&self) -> f64 {
        self.kappa
    }
    fn arrival_intensity(&self) -> f64 {
        1.0
    }
    fn liquidity_gamma_multiplier(&self) -> f64 {
        1.0
    }
    fn is_toxic_regime(&self) -> bool {
        false
    }
    fn jump_ratio(&self) -> f64 {
        1.0
    }
    fn momentum_bps(&self) -> f64 {
        0.0
    }
    fn flow_imbalance(&self) -> f64 {
        0.0
    }
    fn falling_knife_score(&self) -> f64 {
        0.0
    }
    fn rising_knife_score(&self) -> f64 {
        0.0
    }
    fn book_imbalance(&self) -> f64 {
        0.0
    }
    fn microprice(&self) -> f64 {
        0.0
    }
    fn beta_book(&self) -> f64 {
        0.0
    }
    fn beta_flow(&self) -> f64 {
        0.0
    }
    fn lambda_jump(&self) -> f64 {
        0.01
    }
    fn mu_jump(&self) -> f64 {
        0.0
    }
    fn sigma_jump(&self) -> f64 {
        0.001
    }
    fn kappa_vol(&self) -> f64 {
        0.5
    }
    fn theta_vol_sigma(&self) -> f64 {
        self.sigma
    }
    fn xi_vol(&self) -> f64 {
        0.1
    }
    fn rho_price_vol(&self) -> f64 {
        -0.5
    }
    fn is_warmed_up(&self) -> bool {
        self.is_warmed
    }
    fn sigma_confidence(&self) -> f64 {
        if self.is_warmed {
            1.0
        } else {
            0.0
        }
    }
}
