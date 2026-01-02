//! Risk configuration for dynamic gamma scaling.

use chrono::{Timelike, Utc};

/// Configuration for dynamic risk aversion scaling.
///
/// All parameters are explicit for future online optimization.
/// γ_effective = γ_base × vol_scalar × toxicity_scalar × inventory_scalar × time_of_day_scalar
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

    // ==================== Time-of-Day Risk Scaling ====================
    // FIRST PRINCIPLES: Adverse selection varies by time of day.
    // Trade history analysis (Dec 2025) showed toxic hours with -13 to -15 bps edge:
    //   - 06:00-08:00 UTC (London open): High informed flow
    //   - 14:00-15:00 UTC (US afternoon): Institutional activity
    // During these hours, wider spreads are needed to remain profitable.
    /// Enable time-of-day gamma scaling.
    /// When true, γ is multiplied during toxic hours.
    pub enable_time_of_day_scaling: bool,

    /// Gamma multiplier during toxic hours (06-08, 14-15 UTC).
    /// Higher multiplier → wider spreads during informed flow periods.
    /// Recommended: 1.5-2.5 based on trade history adverse selection.
    pub toxic_hour_gamma_multiplier: f64,

    /// Toxic hours configuration (UTC).
    /// Default: 6, 7, 14 (London open and US afternoon).
    /// These hours showed -13 to -15 bps adverse selection in trade data.
    #[serde(default = "default_toxic_hours")]
    pub toxic_hours: Vec<u32>,
}

fn default_toxic_hours() -> Vec<u32> {
    vec![6, 7, 14] // London open (06-08) and US afternoon (14-15)
}

impl RiskConfig {
    /// Get time-of-day gamma multiplier based on current UTC hour.
    ///
    /// Returns toxic_hour_gamma_multiplier during toxic hours, 1.0 otherwise.
    pub fn time_of_day_multiplier(&self) -> f64 {
        if !self.enable_time_of_day_scaling {
            return 1.0;
        }

        let current_hour = Utc::now().hour();
        if self.toxic_hours.contains(&current_hour) {
            self.toxic_hour_gamma_multiplier
        } else {
            1.0
        }
    }

    /// Get time-of-day gamma multiplier for a specific hour (for testing).
    pub fn time_of_day_multiplier_for_hour(&self, hour: u32) -> f64 {
        if !self.enable_time_of_day_scaling {
            return 1.0;
        }

        if self.toxic_hours.contains(&hour) {
            self.toxic_hour_gamma_multiplier
        } else {
            1.0
        }
    }
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
            // FIRST PRINCIPLES: min_spread_floor = fees + AS + buffer + toxic_margin
            // Trade history (Dec 2025):
            //   - Fees: 1.5 bps
            //   - Average AS: 0.5 bps (but 11.6 bps on large trades)
            //   - Need 11.67 bps for break-even
            // Setting to 8 bps = fees (1.5) + AS (0.5) + buffer (6) for base profitability
            min_spread_floor: 0.0008, // 8 bps (raised from 5 bps)
            max_holding_time: 120.0,  // 2 minutes
            flow_sensitivity: 0.5,    // exp(-0.5) ≈ 0.61 at perfect alignment
            maker_fee_rate: 0.00015,  // 1.5 bps Hyperliquid maker fee
            // Time-of-day scaling: ENABLED by default for profitability
            // Trade history showed -13 to -15 bps edge during toxic hours
            enable_time_of_day_scaling: true,
            toxic_hour_gamma_multiplier: 2.0, // Double γ during toxic hours → ~2× wider spreads
            toxic_hours: default_toxic_hours(),
        }
    }
}
