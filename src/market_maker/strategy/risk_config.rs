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

    // ==================== Book Depth Risk Scaling ====================
    // FIRST PRINCIPLES: Thin order books mean higher execution risk.
    // When we can't exit positions easily, we should be more risk averse.
    // This flows through γ in the GLFT formula: δ = (1/γ) × ln(1 + γ/κ)
    // Higher γ → wider spreads, but computed through the principled formula.
    /// Enable book depth gamma scaling.
    /// When true, γ increases as near-touch book depth decreases.
    pub enable_book_depth_scaling: bool,

    /// Book depth threshold (USD) below which gamma scaling activates.
    /// Orders within 5 bps of mid are counted.
    pub book_depth_threshold_usd: f64,

    /// Maximum gamma multiplier when book depth approaches zero.
    /// 1.5 means γ can increase by up to 50% for very thin books.
    pub max_book_depth_gamma_mult: f64,

    // ==================== Warmup Uncertainty Scaling ====================
    // FIRST PRINCIPLES: During parameter estimation warmup, we have uncertainty.
    // Higher uncertainty → more risk aversion → higher γ.
    // This replaces the arbitrary warmup spread multiplier.
    /// Enable warmup gamma scaling.
    /// When true, γ is scaled up during adaptive warmup period.
    pub enable_warmup_gamma_scaling: bool,

    /// Maximum warmup gamma multiplier at 0% warmup progress.
    /// Decays linearly to 1.0 as warmup completes.
    /// 1.1 means 10% higher γ during early warmup.
    pub max_warmup_gamma_mult: f64,
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

    /// Get book depth gamma multiplier.
    ///
    /// FIRST PRINCIPLES: Thin order books → harder to exit → higher risk → higher γ
    /// Multiplier scales linearly from 1.0 (at threshold) to max_book_depth_gamma_mult (at zero depth).
    ///
    /// Formula: mult = 1 + (max_mult - 1) × (1 - depth/threshold)  when depth < threshold
    ///          mult = 1.0                                         when depth >= threshold
    pub fn book_depth_multiplier(&self, near_touch_depth_usd: f64) -> f64 {
        if !self.enable_book_depth_scaling || near_touch_depth_usd >= self.book_depth_threshold_usd
        {
            return 1.0;
        }

        // Linearly scale from 1.0 to max_mult as depth decreases
        let depth_ratio = (near_touch_depth_usd / self.book_depth_threshold_usd).max(0.0);
        let additional = (self.max_book_depth_gamma_mult - 1.0) * (1.0 - depth_ratio);
        1.0 + additional
    }

    /// Get warmup uncertainty gamma multiplier.
    ///
    /// FIRST PRINCIPLES: During warmup, parameter estimates have high variance.
    /// Higher uncertainty → more risk aversion → higher γ.
    ///
    /// Formula: mult = 1 + (max_mult - 1) × (1 - warmup_progress)
    /// At 0% warmup: mult = max_warmup_gamma_mult (e.g., 1.1)
    /// At 100% warmup: mult = 1.0
    pub fn warmup_multiplier(&self, warmup_progress: f64) -> f64 {
        if !self.enable_warmup_gamma_scaling {
            return 1.0;
        }

        let progress = warmup_progress.clamp(0.0, 1.0);
        let additional = (self.max_warmup_gamma_mult - 1.0) * (1.0 - progress);
        1.0 + additional
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
            // Book depth gamma scaling: ENABLED by default
            // FIRST PRINCIPLES: Thin book → harder to exit → more risk averse
            enable_book_depth_scaling: true,
            book_depth_threshold_usd: 50_000.0, // $50k threshold for scaling
            max_book_depth_gamma_mult: 1.5,     // Up to 50% more risk averse for thin books
            // Warmup gamma scaling: ENABLED by default
            // FIRST PRINCIPLES: During warmup, parameter uncertainty → more risk averse
            enable_warmup_gamma_scaling: true,
            max_warmup_gamma_mult: 1.1, // 10% higher γ at start of warmup
        }
    }
}

impl RiskConfig {
    /// Config for HIP-3 DEX with 15-25 bps target spreads.
    ///
    /// Key changes from default:
    /// - Lower gamma_base (0.15 vs 0.3) for tighter spreads
    /// - Lower spread floor (6 bps vs 8 bps)
    /// - Disable time-of-day scaling (HIP-3 has different trading patterns)
    /// - Disable book depth scaling (always thin, causes unnecessary widening)
    /// - Reduced warmup penalty
    ///
    /// GLFT math: With κ=1500 and γ=0.15, δ* ≈ 1/κ ≈ 6.7 bps per side
    /// Total spread: 2 × 6.7 + 3 bps fees = ~16.4 bps (within 15-25 target)
    pub fn hip3() -> Self {
        Self {
            gamma_base: 0.15,                    // Aggressive (vs 0.3 default)
            gamma_min: 0.08,                     // Allow tight quotes
            gamma_max: 2.0,                      // Lower ceiling
            sigma_baseline: 0.00015,             // 15 bps/sec (lower vol expectation)
            volatility_weight: 0.3,              // Less vol scaling
            max_volatility_multiplier: 2.0,      // Lower cap
            toxicity_threshold: 2.0,             // Higher tolerance (less informed flow on HIP-3)
            toxicity_sensitivity: 0.2,           // Less sensitive
            inventory_threshold: 0.4,            // Higher inventory tolerance
            inventory_sensitivity: 1.5,          // Less aggressive scaling
            min_spread_floor: 0.0006,            // 6 bps floor (vs 8 bps)
            max_holding_time: 300.0,             // 5 minutes (slower markets)
            flow_sensitivity: 0.3,               // Less flow adjustment
            maker_fee_rate: 0.00015,             // Same fee
            // DISABLE time-of-day scaling for HIP-3 (different patterns)
            enable_time_of_day_scaling: false,
            toxic_hour_gamma_multiplier: 1.0,
            toxic_hours: vec![],
            // DISABLE book depth scaling (always thin, would cause perpetual widening)
            enable_book_depth_scaling: false,
            book_depth_threshold_usd: 0.0,
            max_book_depth_gamma_mult: 1.0,
            // Reduced warmup penalty
            enable_warmup_gamma_scaling: true,
            max_warmup_gamma_mult: 1.05,         // Minimal warmup penalty (5% vs 10%)
        }
    }
}
