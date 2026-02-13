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

    /// Minimum spread floor (as fraction, e.g., 0.00015 = 1.5 bps).
    /// Physical minimum: must be >= maker_fee_rate to avoid guaranteed loss.
    /// Regime risk routes through gamma_multiplier, not this floor.
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

    /// Funding skew sensitivity for perpetual funding cost integration.
    ///
    /// Theory: Funding creates continuous position cost. Skew should reflect
    /// expected funding cost over holding period.
    ///
    /// Formula: funding_skew = (funding_rate / 3600) × time_horizon × sensitivity
    ///
    /// Values:
    /// - 1.0 = full economic impact (funding cost fully reflected in skew)
    /// - 0.5 = conservative (funding cost partially reflected)
    /// - 0.0 = disabled (no funding-based skew)
    ///
    /// Typical impact at extreme funding (10 bps/hour):
    /// - sensitivity=1.0: ~0.17 bps skew over 60s horizon
    pub funding_skew_sensitivity: f64,

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
    ///
    /// DERIVATION (Dec 2025 trade analysis, 2,038 trades):
    /// - Toxic hours (6,7,14 UTC): avg adverse selection = -14 bps
    /// - Maker fee: 1.5 bps
    /// - Normal spread floor: 8 bps
    /// - Required to break even: toxic_mult = (|toxic_edge| + fees) / normal_spread
    ///   = (14 + 1.5) / 8 = 1.94 ≈ 2.0
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
    ///
    /// DERIVATION (Exit slippage model):
    /// - Reference depth: $50k (can exit typical $10k position at near-touch)
    /// - At $10k depth: only 20% executable at touch, 80% walks book
    /// - AS decay model: AS(δ) = 3 × exp(-δ/8) bps
    /// - Expected slippage increase when depth drops 5×: ~50%
    /// - gamma_adj = 1 + slippage_increase = 1.5
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

    // ==================== Log-Odds AS Integration ====================
    // FIRST PRINCIPLES: The theoretically correct way to integrate adverse
    // selection probability into the spread is via the log-odds ratio, not
    // multiplicative scaling. From Bayesian market maker theory:
    //   as_adjustment = (1/γ) × ln(p_informed / p_noise) (when p_informed > p_noise)
    // This gives a convex, principled mapping from toxicity to spread widening.
    /// Enable log-odds additive AS integration.
    /// When true, pre-fill toxicity is converted to an additive spread
    /// adjustment via the log-odds ratio. When false, uses the legacy
    /// multiplicative spread multiplier path.
    #[serde(default = "default_true")]
    pub use_log_odds_as: bool,

    /// Maximum AS adjustment in basis points when using log-odds mode.
    /// Caps the adjustment to prevent blow-up when p_informed -> 1.0.
    /// Default 15 bps: at typical gamma=0.15 this caps around p_informed=0.85.
    #[serde(default = "default_max_as_adjustment_bps")]
    pub max_as_adjustment_bps: f64,

    /// Enable monopolist LP pricing for illiquid tokens.
    /// When true and competitor_count < 1.5, applies an additive markup
    /// based on estimated taker price elasticity.
    /// Default: false (opt-in for illiquid tokens only).
    #[serde(default)]
    pub use_monopolist_pricing: bool,

    /// Maximum monopolist markup in basis points.
    /// Caps the elasticity-derived markup to prevent excessive spreads.
    /// Default: 5.0 bps.
    #[serde(default = "default_monopolist_markup_cap_bps")]
    pub monopolist_markup_cap_bps: f64,

    /// Minimum observations for taker elasticity estimation.
    /// Below this count, the estimator returns a conservative default.
    /// Default: 50.
    #[serde(default = "default_min_observations_for_elasticity")]
    pub min_observations_for_elasticity: usize,
}

fn default_monopolist_markup_cap_bps() -> f64 {
    5.0
}

fn default_min_observations_for_elasticity() -> usize {
    50
}

fn default_toxic_hours() -> Vec<u32> {
    vec![6, 7, 14] // London open (06-08) and US afternoon (14-15)
}

fn default_true() -> bool {
    true
}

fn default_max_as_adjustment_bps() -> f64 {
    15.0
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

    /// Get warmup uncertainty gamma multiplier (legacy: progress-based).
    ///
    /// FIRST PRINCIPLES: During warmup, parameter estimates have high variance.
    /// Higher uncertainty → more risk aversion → higher γ.
    ///
    /// Formula: mult = 1 + (max_mult - 1) × (1 - warmup_progress)
    /// At 0% warmup: mult = max_warmup_gamma_mult (e.g., 1.1)
    /// At 100% warmup: mult = 1.0
    ///
    /// NOTE: Prefer `warmup_multiplier_from_ci()` when posterior data is available.
    pub fn warmup_multiplier(&self, warmup_progress: f64) -> f64 {
        if !self.enable_warmup_gamma_scaling {
            return 1.0;
        }

        let progress = warmup_progress.clamp(0.0, 1.0);
        let additional = (self.max_warmup_gamma_mult - 1.0) * (1.0 - progress);
        1.0 + additional
    }

    /// Warmup multiplier derived from Bayesian posterior uncertainty.
    ///
    /// DERIVATION (Information-theoretic):
    /// - Wider credible interval → more parameter uncertainty → higher risk
    /// - CI width normalized by mean gives dimensionless uncertainty measure
    /// - Linear scaling: gamma_mult = 1 + ci_width / scaling_factor
    /// - scaling_factor=10 chosen so CI width of 1.0 → mult of 1.1
    ///
    /// At warmup start: ci_width ≈ 1.0 → mult ≈ 1.1
    /// After convergence: ci_width ≈ 0.3 → mult ≈ 1.03
    ///
    /// This is preferred over `warmup_multiplier()` as it directly reflects
    /// the actual posterior uncertainty rather than a time-based proxy.
    pub fn warmup_multiplier_from_ci(&self, kappa_ci_width: f64) -> f64 {
        if !self.enable_warmup_gamma_scaling {
            return 1.0;
        }

        // Scale CI width to multiplier
        // CI width of 1.0 → mult of 1.1 (matches legacy max_warmup_gamma_mult)
        // CI width of 0.0 → mult of 1.0 (no uncertainty penalty)
        let scaling_factor = 10.0;
        (1.0 + kappa_ci_width / scaling_factor).clamp(1.0, self.max_warmup_gamma_mult)
    }

    /// Derive optimal gamma from GLFT first principles.
    ///
    /// DERIVATION: From the GLFT formula δ* = (1/γ) × ln(1 + γ/κ)
    ///
    /// For a target spread δ_target:
    /// γ = κ × (exp(δ × κ) - 1) / exp(δ × κ)
    ///
    /// Adjusted for volatility: higher vol → higher risk aversion
    /// Adjusted for time horizon: longer horizon → more time to offset losses
    ///
    /// # Arguments
    /// * `target_half_spread_bps` - Desired half-spread in bps
    /// * `kappa` - Fill intensity (fills per unit spread)
    /// * `sigma` - Current volatility (per-second)
    /// * `time_horizon` - Expected holding time (seconds)
    ///
    /// # Returns
    /// Optimal gamma for achieving target spread, clamped to [gamma_min, gamma_max]
    pub fn derive_gamma_from_glft(
        &self,
        target_half_spread_bps: f64,
        kappa: f64,
        sigma: f64,
        time_horizon: f64,
    ) -> f64 {
        let target_spread = target_half_spread_bps / 10_000.0;

        if kappa <= 0.0 || target_spread <= 0.0 {
            return self.gamma_base; // Fallback to default
        }

        let exp_term = (target_spread * kappa).exp();
        let gamma_raw = kappa * (exp_term - 1.0) / exp_term;

        // Volatility adjustment: higher vol → higher risk aversion
        let vol_adjustment = 1.0 + (sigma / self.sigma_baseline - 1.0).max(0.0) * self.volatility_weight;

        // Time horizon adjustment: longer horizon → more time to offset losses
        let time_adjustment = (time_horizon / 60.0).sqrt().clamp(0.5, 2.0);

        (gamma_raw * vol_adjustment / time_adjustment).clamp(self.gamma_min, self.gamma_max)
    }

    /// Derive spread floor from first principles.
    ///
    /// DERIVATION: Minimum profitable spread must cover:
    /// 1. Maker fee (1.5 bps on Hyperliquid)
    /// 2. Expected adverse selection
    /// 3. Execution slippage from latency
    ///
    /// δ_min = f_maker + E[AS] + σ × √(τ_update) + buffer
    ///
    /// # Arguments
    /// * `expected_as_bps` - Expected adverse selection in bps
    /// * `sigma` - Current volatility (per-second)
    /// * `update_latency_ms` - Quote update latency in milliseconds
    ///
    /// # Returns
    /// Minimum spread floor in fraction (e.g., 0.0005 for 5 bps)
    pub fn derive_spread_floor(
        &self,
        expected_as_bps: f64,
        sigma: f64,
        update_latency_ms: f64,
    ) -> f64 {
        let maker_fee_bps = self.maker_fee_rate * 10_000.0;
        let latency_s = update_latency_ms / 1000.0;

        // Slippage from latency: price can move σ×√τ during quote update cycle
        let latency_slippage_bps = sigma * latency_s.sqrt() * 10_000.0;

        // Total spread floor (with 10% buffer)
        let floor_bps = (maker_fee_bps + expected_as_bps.abs() + latency_slippage_bps) * 1.1;

        // Convert to fraction and clamp
        let floor = floor_bps / 10_000.0;
        floor.clamp(self.min_spread_floor, 0.01) // Max 100 bps
    }
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            // MAINNET OPTIMIZED: Lower gamma for liquid markets with deep books
            gamma_base: 0.15, // Reduced from 0.3 - liquid markets need less risk aversion
            sigma_baseline: 0.0002, // 2 bps per-√second (0.02% instantaneous volatility)
                                    // Equivalent to ~32 bps daily vol in 252-day year
                                    // Or ~12 bps per minute for short horizons
            volatility_weight: 0.5,
            // MAINNET OPTIMIZED: Cap vol scaling for liquid markets
            max_volatility_multiplier: 2.0, // Reduced from 3.0 - less extreme scaling
            toxicity_threshold: 1.5,
            toxicity_sensitivity: 0.3,
            // MAINNET OPTIMIZED: Higher inventory tolerance for liquid markets
            inventory_threshold: 0.5, // Increased from 0.3 - easier to exit on liquid books
            inventory_sensitivity: 3.0, // Increased from 2.0 - stronger inventory effect on gamma
            gamma_min: 0.05,
            gamma_max: 5.0,
            // FIRST PRINCIPLES: min_spread_floor = maker fee (physical minimum).
            // Regime risk is now routed through gamma_multiplier, not floor.
            // Floor only prevents quoting below fee (guaranteed loss).
            min_spread_floor: 0.00015, // 1.5 bps = Hyperliquid maker fee
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
            // Funding skew sensitivity: 1.0 = full economic impact
            // At extreme funding (10 bps/hour), produces ~0.17 bps skew over 60s horizon
            funding_skew_sensitivity: 1.0,
            // Log-odds AS: ENABLED by default (theoretically correct integration)
            // Converts pre-fill toxicity probability to additive spread via log-odds ratio
            use_log_odds_as: true,
            max_as_adjustment_bps: 15.0, // Cap at 15 bps to prevent blow-up
            // Monopolist pricing: DISABLED by default (opt-in for illiquid tokens)
            use_monopolist_pricing: false,
            monopolist_markup_cap_bps: 5.0,
            min_observations_for_elasticity: 50,
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
            gamma_base: 0.15,               // Aggressive (vs 0.3 default)
            gamma_min: 0.08,                // Allow tight quotes
            gamma_max: 2.0,                 // Lower ceiling
            sigma_baseline: 0.00015,        // 15 bps/sec (lower vol expectation)
            volatility_weight: 0.3,         // Less vol scaling
            max_volatility_multiplier: 2.0, // Lower cap
            toxicity_threshold: 2.0,        // Higher tolerance (less informed flow on HIP-3)
            toxicity_sensitivity: 0.2,      // Less sensitive
            inventory_threshold: 0.4,       // Higher inventory tolerance
            inventory_sensitivity: 1.5,     // Less aggressive scaling
            min_spread_floor: 0.0006,       // 6 bps floor (vs 8 bps)
            max_holding_time: 300.0,        // 5 minutes (slower markets)
            flow_sensitivity: 0.3,          // Less flow adjustment
            maker_fee_rate: 0.00015,        // Same fee
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
            max_warmup_gamma_mult: 1.05, // Minimal warmup penalty (5% vs 10%)
            // Funding skew sensitivity: same as default
            funding_skew_sensitivity: 1.0,
            // Log-odds AS: same as default
            use_log_odds_as: true,
            max_as_adjustment_bps: 15.0,
            // Monopolist pricing: ENABLED for HIP-3 (typically sole LP)
            use_monopolist_pricing: true,
            monopolist_markup_cap_bps: 5.0,
            min_observations_for_elasticity: 50,
        }
    }
}
