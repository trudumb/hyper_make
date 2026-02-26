//! Volatility parameter set.

use crate::market_maker::estimator::VolatilityRegime;

/// Volatility metrics from the estimator.
#[derive(Debug, Clone, Default)]
pub struct VolatilityParams {
    /// Clean volatility (σ_clean) - √BV, robust to jumps.
    /// Use for base spread calculation (continuous risk).
    pub sigma: f64,

    /// Total volatility (σ_total) - √RV, includes jumps.
    /// Captures full price variance including discontinuities.
    pub sigma_total: f64,

    /// Effective volatility (σ_effective) - blended clean/total.
    /// Reacts to jump regime; use for inventory skew.
    pub sigma_effective: f64,

    /// Volatility regime (Low/Normal/High/Extreme).
    pub regime: VolatilityRegime,

    // Stochastic volatility model parameters (Gap 2)
    /// Vol mean-reversion speed κ_vol.
    pub kappa_vol: f64,
    /// Long-run volatility θ_vol.
    pub theta_vol: f64,
    /// Vol-of-vol ξ.
    pub xi_vol: f64,
    /// Price-vol correlation ρ (leverage effect).
    pub rho_price_vol: f64,
}

impl VolatilityParams {
    /// Create with basic volatility metrics.
    pub fn new(sigma: f64, sigma_total: f64, sigma_effective: f64) -> Self {
        Self {
            sigma,
            sigma_total,
            sigma_effective,
            regime: VolatilityRegime::Normal,
            kappa_vol: 0.5,
            theta_vol: 0.0001_f64.powi(2),
            xi_vol: 0.1,
            rho_price_vol: -0.3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_params_new() {
        let params = VolatilityParams::new(0.001, 0.002, 0.0015);
        assert!((params.sigma - 0.001).abs() < f64::EPSILON);
        assert!((params.sigma_total - 0.002).abs() < f64::EPSILON);
        assert!((params.sigma_effective - 0.0015).abs() < f64::EPSILON);
    }
}
