//! Dynamic risk configuration types.

/// Configuration for first-principles dynamic risk limits.
///
/// All parameters are derived from mathematical principles - no arbitrary clamps.
/// Position limits adapt to account equity and volatility via Bayesian regularization.
/// Skew adjustments respond to order flow using exponential modifiers.
#[derive(Debug, Clone)]
pub struct DynamicRiskConfig {
    /// Fraction of capital to risk in a num_sigmas move.
    /// Derived from Kelly criterion: risk_fraction ≈ edge / variance
    /// At 0.5, a 5-sigma move leaves 50% of capital intact.
    pub risk_fraction: f64,

    /// Confidence level in standard deviations.
    /// 5.0 = 99.99997% confidence (5-sigma)
    pub num_sigmas: f64,

    /// Prior volatility when estimator has low confidence.
    /// Use historical baseline (e.g., 0.0002 = 2bp/sec for BTC)
    pub sigma_prior: f64,

    /// Flow sensitivity β for skew adjustment.
    /// exp(-β × alignment) is the modifier.
    /// β = 0.5 → ±39% adjustment at perfect alignment
    /// β = 1.0 → ±63% adjustment at perfect alignment
    pub flow_sensitivity: f64,

    /// Maximum leverage from exchange (queried from asset metadata).
    /// Caps position_value to account_value × max_leverage.
    /// This is the hard constraint - volatility can only reduce, never exceed.
    pub max_leverage: f64,
}

impl Default for DynamicRiskConfig {
    fn default() -> Self {
        Self {
            risk_fraction: 0.5,
            num_sigmas: 5.0,
            sigma_prior: 0.0002, // 2bp/sec baseline
            flow_sensitivity: 0.5,
            max_leverage: 20.0, // Conservative default, should be queried from exchange
        }
    }
}

impl DynamicRiskConfig {
    /// Create a new dynamic risk config with custom risk fraction.
    pub fn with_risk_fraction(mut self, risk_fraction: f64) -> Self {
        self.risk_fraction = risk_fraction;
        self
    }

    /// Create a new dynamic risk config with custom sigma prior.
    pub fn with_sigma_prior(mut self, sigma_prior: f64) -> Self {
        self.sigma_prior = sigma_prior;
        self
    }

    /// Create a new dynamic risk config with custom flow sensitivity.
    pub fn with_flow_sensitivity(mut self, flow_sensitivity: f64) -> Self {
        self.flow_sensitivity = flow_sensitivity;
        self
    }

    /// Create a new dynamic risk config with custom max leverage.
    pub fn with_max_leverage(mut self, max_leverage: f64) -> Self {
        self.max_leverage = max_leverage;
        self
    }

    /// Validate invariants for dynamic risk parameters.
    ///
    /// All parameters are derived from Kelly criterion and Bayesian principles —
    /// invalid values cause nonsensical risk limits.
    pub fn validate(&self) -> Result<(), String> {
        if self.risk_fraction <= 0.0 || self.risk_fraction > 1.0 {
            return Err(format!(
                "risk_fraction must be in (0.0, 1.0], got {}",
                self.risk_fraction
            ));
        }
        if self.num_sigmas <= 0.0 {
            return Err(format!(
                "num_sigmas must be > 0.0, got {}",
                self.num_sigmas
            ));
        }
        if self.sigma_prior <= 0.0 {
            return Err(format!(
                "sigma_prior must be > 0.0, got {}",
                self.sigma_prior
            ));
        }
        if self.max_leverage <= 0.0 {
            return Err(format!(
                "max_leverage must be > 0.0, got {}",
                self.max_leverage
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_config_validate_accepts_defaults() {
        let cfg = DynamicRiskConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_risk_config_validate_rejects_zero_risk_fraction() {
        let cfg = DynamicRiskConfig::default().with_risk_fraction(0.0);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("risk_fraction"), "error should mention risk_fraction: {err}");
    }

    #[test]
    fn test_risk_config_validate_rejects_excessive_risk_fraction() {
        let cfg = DynamicRiskConfig::default().with_risk_fraction(1.5);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("risk_fraction"), "error should mention risk_fraction: {err}");
    }

    #[test]
    fn test_risk_config_validate_rejects_zero_max_leverage() {
        let cfg = DynamicRiskConfig::default().with_max_leverage(0.0);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("max_leverage"), "error should mention max_leverage: {err}");
    }
}
