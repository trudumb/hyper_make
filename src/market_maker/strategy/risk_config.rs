//! Risk configuration for dynamic gamma scaling.

/// Configuration for dynamic risk aversion scaling.
///
/// All parameters are explicit for future online optimization.
/// γ_effective = γ_base × vol_scalar × toxicity_scalar × inventory_scalar
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RiskConfig {
    /// Base risk aversion (γ_base) - personality in normal conditions
    /// Typical values: 0.1 (aggressive) to 1.0 (conservative)
    pub gamma_base: f64,

    /// Baseline volatility for scaling (per-second σ)
    /// When σ_effective > this, γ scales up
    pub sigma_baseline: f64,

    /// Weight for volatility scaling [0.0, 1.0]
    /// 0.0 = ignore volatility, 1.0 = full scaling
    pub volatility_weight: f64,

    /// Maximum volatility multiplier
    /// Caps how much high volatility can increase γ
    pub max_volatility_multiplier: f64,

    /// Toxicity threshold (jump_ratio above this triggers scaling)
    pub toxicity_threshold: f64,

    /// How much toxicity increases γ per unit of jump_ratio above 1.0
    pub toxicity_sensitivity: f64,

    /// Inventory utilization threshold for γ scaling [0.0, 1.0]
    /// Below this, no inventory scaling
    pub inventory_threshold: f64,

    /// How aggressively γ increases near position limits
    /// Uses quadratic scaling: 1 + sensitivity × (utilization - threshold)²
    pub inventory_sensitivity: f64,

    /// Minimum γ floor
    pub gamma_min: f64,

    /// Maximum γ ceiling
    pub gamma_max: f64,

    /// Minimum spread floor (as fraction, e.g., 0.00015 = 1.5 bps)
    /// Should be >= maker_fee_rate to ensure profitability at minimum spread
    pub min_spread_floor: f64,

    /// Maximum holding time cap (seconds)
    /// Prevents skew explosion in dead markets
    pub max_holding_time: f64,

    /// Flow sensitivity β for inventory skew adjustment.
    /// Controls how strongly flow alignment dampens/amplifies skew.
    /// exp(-β × alignment) is the modifier:
    ///   - β = 0.5 → ±39% adjustment at perfect alignment
    ///   - β = 1.0 → ±63% adjustment at perfect alignment
    ///
    /// Derived from information theory (exponential link function).
    pub flow_sensitivity: f64,

    /// Maker fee rate as fraction of notional.
    /// This is added to the GLFT half-spread to ensure profitability.
    /// The HJB equation with fees: dW = (δ - f_maker) × dN
    /// Therefore optimal spread: δ* = δ_GLFT + f_maker
    ///
    /// Hyperliquid maker fee: 0.00015 (1.5 bps)
    pub maker_fee_rate: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            gamma_base: 0.3,
            sigma_baseline: 0.0002, // 20bp per-second
            volatility_weight: 0.5,
            max_volatility_multiplier: 3.0,
            toxicity_threshold: 1.5,
            toxicity_sensitivity: 0.3,
            inventory_threshold: 0.5,
            inventory_sensitivity: 2.0,
            gamma_min: 0.05,
            gamma_max: 5.0,
            min_spread_floor: 0.00015, // 1.5 bps - matches maker fee for guaranteed profitability
            max_holding_time: 120.0,   // 2 minutes
            flow_sensitivity: 0.5,     // exp(-0.5) ≈ 0.61 at perfect alignment
            maker_fee_rate: 0.00015,   // 1.5 bps Hyperliquid maker fee
        }
    }
}
