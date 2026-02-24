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
            return Err(format!("num_sigmas must be > 0.0, got {}", self.num_sigmas));
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

/// Configuration for fill cascade detection and mitigation.
///
/// Uses a size-weighted Hawkes process (one per side) to detect fill clustering.
/// Intensity ratio λ/μ drives graduated response: burst → widen → suppress.
/// Large fills excite more than small fills via `size_scale` normalization.
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    // === Hawkes process parameters ===
    /// Baseline fill intensity μ (fills/sec/side). ~5 fills/min = 0.08.
    pub baseline_intensity: f64,
    /// Self-excitation α. Must be < β for stationarity.
    pub alpha: f64,
    /// Decay rate β (per second). β=0.1 → ~10s effective memory.
    pub beta: f64,
    /// Size normalization. Contribution per fill = α × (size / size_scale).
    pub size_scale: f64,

    // === Intensity ratio thresholds ===
    /// λ/μ ratio to trigger widen (spread addon).
    pub widen_intensity_ratio: f64,
    /// λ/μ ratio to trigger suppress (reduce-only).
    pub suppress_intensity_ratio: f64,
    /// λ/μ ratio for burst detection (sigma boost). Lower than widen.
    pub burst_intensity_ratio: f64,

    // === Response parameters ===
    /// Spread additive widening (bps) when in widen mode.
    pub widen_addon_bps: f64,
    /// Spread additive widening (bps) when in suppress mode.
    pub suppress_addon_bps: f64,
    /// Cooldown duration in seconds for widen mode.
    pub widen_cooldown_secs: u64,
    /// Cooldown duration in seconds for suppress mode.
    pub suppress_cooldown_secs: u64,
    /// Duration in seconds for sigma boost after burst detected.
    pub burst_sigma_boost_secs: u64,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            baseline_intensity: 0.08,
            alpha: 0.05,
            beta: 0.1,
            size_scale: 20.0,
            widen_intensity_ratio: 3.0,
            suppress_intensity_ratio: 5.0,
            burst_intensity_ratio: 2.0,
            widen_addon_bps: 10.0,
            suppress_addon_bps: 20.0,
            widen_cooldown_secs: 15,
            suppress_cooldown_secs: 30,
            burst_sigma_boost_secs: 30,
        }
    }
}

impl CascadeConfig {
    /// Validate cascade configuration invariants.
    pub fn validate(&self) -> Result<(), String> {
        if self.baseline_intensity <= 0.0 {
            return Err(format!(
                "baseline_intensity must be > 0.0, got {}",
                self.baseline_intensity
            ));
        }
        if self.alpha <= 0.0 {
            return Err(format!("alpha must be > 0.0, got {}", self.alpha));
        }
        if self.beta <= 0.0 {
            return Err(format!("beta must be > 0.0, got {}", self.beta));
        }
        if self.alpha >= self.beta {
            return Err(format!(
                "alpha ({}) must be < beta ({}) for stationarity",
                self.alpha, self.beta
            ));
        }
        if self.size_scale <= 0.0 {
            return Err(format!("size_scale must be > 0.0, got {}", self.size_scale));
        }
        if self.suppress_intensity_ratio <= self.widen_intensity_ratio {
            return Err(format!(
                "suppress_intensity_ratio ({}) must be > widen_intensity_ratio ({})",
                self.suppress_intensity_ratio, self.widen_intensity_ratio
            ));
        }
        if self.burst_intensity_ratio <= 1.0 {
            return Err(format!(
                "burst_intensity_ratio must be > 1.0, got {}",
                self.burst_intensity_ratio
            ));
        }
        if self.widen_addon_bps < 0.0 {
            return Err(format!(
                "widen_addon_bps must be >= 0.0, got {}",
                self.widen_addon_bps
            ));
        }
        if self.suppress_addon_bps < 0.0 {
            return Err(format!(
                "suppress_addon_bps must be >= 0.0, got {}",
                self.suppress_addon_bps
            ));
        }
        Ok(())
    }
}

/// Configuration for asymmetric quote staleness defense.
///
/// Between quote cycles (5-6s), price may move, making quotes on one side
/// stale (too generous). This adds per-side spread based on the displacement.
///
/// Formula: addon_bps = max_addon_bps × (1 - exp(-|move_bps| / decay_bps))
/// Applied to the stale side only (bids when market drops, asks when rises).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StalenessConfig {
    /// Whether staleness defense is enabled.
    pub enabled: bool,
    /// Characteristic decay scale (bps). Higher = slower saturation.
    /// At |move| = decay_bps, addon ≈ 63% of max.
    pub decay_bps: f64,
    /// Maximum additive widening from staleness (bps).
    pub max_addon_bps: f64,
    /// Minimum price move to trigger (bps). Below this, no widening.
    pub min_move_bps: f64,
}

impl Default for StalenessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            decay_bps: 3.0,
            max_addon_bps: 10.0,
            min_move_bps: 0.5,
        }
    }
}

/// Configuration for inventory governor zone thresholds.
///
/// Controls when the governor transitions between zones and how
/// aggressively it widens the position-increasing side.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GovernorConfig {
    /// Position ratio threshold to enter Yellow zone [0.0, 1.0].
    /// Below this is Green (full two-sided quoting).
    pub yellow_threshold: f64,
    /// Position ratio threshold to enter Red zone [yellow, 1.0].
    /// Red = reduce-only.
    pub red_threshold: f64,
    /// Maximum additive spread widening (bps) at the top of Yellow zone.
    /// Linear ramp from 0.0 at yellow_threshold to this at red_threshold.
    pub yellow_max_addon_bps: f64,
    /// Additive spread widening (bps) in Red zone (reduce-only).
    pub red_addon_bps: f64,
    /// Additive spread widening (bps) in Kill zone (at/above max position).
    pub kill_addon_bps: f64,
    /// Additive spread TIGHTENING (bps) for the reducing side in Kill zone.
    /// Applied as negative addon to attract fills and escape overexposed state.
    /// E.g., 25.0 → reducing side quotes tightened by 25 bps in Kill zone.
    #[serde(default = "default_kill_reducing_addon")]
    pub kill_reducing_addon_bps: f64,
    /// Additive spread TIGHTENING (bps) for the reducing side in Red zone.
    /// Milder than Kill zone but still attracts reducing fills.
    #[serde(default = "default_red_reducing_addon")]
    pub red_reducing_addon_bps: f64,
}

fn default_kill_reducing_addon() -> f64 {
    25.0
}

fn default_red_reducing_addon() -> f64 {
    15.0
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            yellow_threshold: 0.50,
            red_threshold: 0.80,
            yellow_max_addon_bps: 10.0,
            red_addon_bps: 15.0,
            kill_addon_bps: 25.0,
            kill_reducing_addon_bps: 25.0,
            red_reducing_addon_bps: 15.0,
        }
    }
}

/// Configuration for directional flow toxicity defense.
///
/// Widens the vulnerable side when informed flow is detected on one side.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlowToxicityConfig {
    /// Whether flow toxicity defense is enabled.
    pub enabled: bool,
    /// Toxicity threshold to start widening [0.0, 1.0].
    /// Below this, no addon.
    pub threshold: f64,
    /// Maximum additive widening from flow toxicity (bps).
    pub max_addon_bps: f64,
}

impl Default for FlowToxicityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.3,
            max_addon_bps: 8.0,
        }
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
        assert!(
            err.contains("risk_fraction"),
            "error should mention risk_fraction: {err}"
        );
    }

    #[test]
    fn test_risk_config_validate_rejects_excessive_risk_fraction() {
        let cfg = DynamicRiskConfig::default().with_risk_fraction(1.5);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("risk_fraction"),
            "error should mention risk_fraction: {err}"
        );
    }

    #[test]
    fn test_risk_config_validate_rejects_zero_max_leverage() {
        let cfg = DynamicRiskConfig::default().with_max_leverage(0.0);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("max_leverage"),
            "error should mention max_leverage: {err}"
        );
    }

    #[test]
    fn test_cascade_config_validate_accepts_defaults() {
        let cfg = CascadeConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_cascade_config_validate_rejects_bad_ratio_ordering() {
        let mut cfg = CascadeConfig::default();
        cfg.suppress_intensity_ratio = cfg.widen_intensity_ratio; // equal -> invalid
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_cascade_config_validate_rejects_alpha_ge_beta() {
        let mut cfg = CascadeConfig::default();
        cfg.alpha = cfg.beta; // non-stationary
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_cascade_config_validate_rejects_zero_baseline() {
        let mut cfg = CascadeConfig::default();
        cfg.baseline_intensity = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_cascade_config_validate_rejects_low_burst_ratio() {
        let mut cfg = CascadeConfig::default();
        cfg.burst_intensity_ratio = 1.0; // must be > 1.0
        assert!(cfg.validate().is_err());
    }
}
