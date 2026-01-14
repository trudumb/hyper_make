//! Stochastic module integration configuration.

/// Method for calculating Kelly time horizon for first-passage fill probability.
///
/// The Kelly-Stochastic optimizer needs a time horizon τ to compute fill probabilities.
/// This is semantically different from the GLFT inventory time horizon T = 1/λ.
///
/// For first-passage probability P(fill) = 2Φ(-δ/(σ√τ)) to be meaningful:
/// - τ must be long enough for price to diffuse to quote depth
/// - τ = (δ/σ)² gives P(fill at δ) ≈ 15.9%
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum KellyTimeHorizonMethod {
    /// Fixed time horizon in seconds.
    /// Use when you want predictable behavior regardless of volatility.
    Fixed,
    /// Diffusion-based: τ = (δ_char / σ)² clamped to [τ_min, τ_max].
    /// Automatically scales with volatility to maintain meaningful fill probabilities.
    /// This is the recommended default for first-principles correctness.
    #[default]
    DiffusionBased,
    /// Use arrival intensity (1/λ) - legacy behavior.
    /// WARNING: This typically produces τ ~milliseconds, causing P(fill) ≈ 0.
    ArrivalIntensity,
}

/// Configuration for stochastic module integration.
///
/// Controls feature flags for first-principles stochastic components:
/// - HJB optimal inventory skew
/// - Kalman filter for price denoising
/// - Constrained variational ladder optimization
/// - Depth-dependent adverse selection calibration
/// - Kelly-Stochastic optimal allocation
#[derive(Debug, Clone)]
pub struct StochasticConfig {
    /// Use HJB optimal_skew instead of heuristic inventory_skew_with_flow.
    /// When true: skew = γσ²qT + terminal_penalty × q × urgency + funding_bias
    /// When false: skew = inventory_ratio × γ × σ² × T × flow_modifier (existing)
    pub use_hjb_skew: bool,

    /// Use Kalman-filtered price for microprice base and spread widening.
    /// Adds uncertainty-based spread widening: half_spread += γ × σ_kalman × √T
    pub use_kalman_filter: bool,

    /// Feed fills to DepthDecayAS for calibration.
    /// Records fill depth and realized AS for exponential decay model fitting.
    /// Safe to enable - passive data collection only.
    pub calibrate_depth_as: bool,

    /// Use calibrated DepthDecayAS model in ladder generation.
    /// When true and calibrated: AS(δ) = AS₀ × exp(-δ/δ_char)
    /// When false: uses config-based flat AS adjustment
    pub use_calibrated_as: bool,

    /// Kalman filter process noise Q (price variance per tick).
    /// Higher Q = trust observations more (reactive).
    /// Typical: 1e-8 (1 bp² per tick)
    pub kalman_q: f64,

    /// Kalman filter observation noise R (bid-ask bounce variance).
    /// Higher R = trust model more (smooth).
    /// Typical: 2.5e-9 (0.5 bp²)
    pub kalman_r: f64,

    /// HJB controller session duration (seconds).
    /// For 24/7 markets, use daily session (86400) or shorter sub-sessions.
    pub hjb_session_duration: f64,

    /// HJB terminal inventory penalty ($/unit²).
    /// Higher = more aggressive position reduction near session end.
    pub hjb_terminal_penalty: f64,

    /// HJB funding rate EWMA half-life (seconds).
    pub hjb_funding_half_life: f64,

    // ==================== Kelly-Stochastic Parameters ====================
    /// Informed trader probability at the touch (0.0-1.0).
    /// Estimated from fill data or set conservatively.
    /// Default: 0.15 (15% of trades at touch are informed)
    pub kelly_alpha_touch: f64,

    /// Characteristic depth for alpha decay in bps.
    /// α(δ) = α_touch × exp(-δ/alpha_decay_bps)
    /// Default: 10 bps
    pub kelly_alpha_decay_bps: f64,

    /// Kelly fraction (0.25 = quarter Kelly, recommended 0.25-0.5).
    /// Lower values are more conservative.
    /// Default: 0.25
    pub kelly_fraction: f64,

    // ==================== Kelly Time Horizon Parameters ====================
    /// Method for calculating Kelly time horizon.
    /// Default: DiffusionBased (scales τ with volatility for correct fill probabilities)
    pub kelly_time_horizon_method: KellyTimeHorizonMethod,

    /// Fixed tau value in seconds (used when method = Fixed).
    /// Default: 60.0 (1 minute)
    pub kelly_tau_fixed: f64,

    /// Minimum tau in seconds (clamp floor for DiffusionBased).
    /// Prevents τ from becoming too short in high-vol regimes.
    /// Default: 10.0 (10 seconds)
    pub kelly_tau_min: f64,

    /// Maximum tau in seconds (clamp ceiling for DiffusionBased/ArrivalIntensity).
    /// Prevents τ from becoming too long in low-vol regimes.
    /// Default: 600.0 (10 minutes)
    pub kelly_tau_max: f64,

    /// Characteristic depth in bps for diffusion-based tau calculation.
    /// τ = (kelly_char_depth_bps / σ)² gives P(fill at δ_char) ≈ 15.9%.
    /// Should be set to mid-ladder depth for balanced allocation.
    /// Default: 25.0 bps
    pub kelly_char_depth_bps: f64,

    // ==================== Stochastic Constraints (First Principles) ====================
    /// Enable latency-aware spread floor: δ_min = σ × √(2×τ_update) + fee
    /// When enabled, spread floor dynamically scales with volatility and quote latency.
    /// Default: true
    pub use_latency_spread_floor: bool,

    /// Expected quote update latency in milliseconds.
    /// Used in latency-aware spread floor: δ_min = σ × √(2×τ_update)
    /// Lower latency allows tighter spreads.
    /// Default: 50.0 ms
    pub quote_update_latency_ms: f64,

    /// Enable book depth threshold for tight quoting.
    /// When enabled, tight spreads require sufficient book depth to ensure fills.
    /// Default: true
    pub use_book_depth_constraint: bool,

    /// Minimum book depth (USD) within 5 bps of mid to allow tight quoting.
    /// Below this, spreads widen to protect against thin-book slippage.
    /// Default: $50,000
    pub min_book_depth_usd: f64,

    /// Book depth (USD) required for tightest spreads (3-5 bps).
    /// Interpolate between min and tight thresholds for spread adjustment.
    /// Default: $200,000
    pub tight_spread_book_depth_usd: f64,

    /// Enable conditional tight quoting logic.
    /// When true, enforces all prerequisites for tight spreads:
    /// - Calm volatility regime
    /// - Low toxicity (< toxicity_threshold)
    /// - Sufficient book depth
    /// - Low inventory utilization
    ///
    /// Default: true
    pub use_conditional_tight_quoting: bool,

    /// Maximum inventory utilization (fraction of max_position) for tight quoting.
    /// Above this, spreads widen to reduce inventory risk.
    /// Default: 0.3 (30%)
    pub tight_quoting_max_inventory: f64,

    /// Maximum toxicity (predicted alpha) for tight quoting.
    /// Above this, spreads widen to protect against informed flow.
    /// Default: 0.1 (10%)
    pub tight_quoting_max_toxicity: f64,

    /// Hours (UTC) to exclude from tight quoting due to high volatility.
    /// Common: US open (14:30 UTC), EU open (7-8 UTC), Asia session (0-2 UTC).
    /// Default: [7, 14] (EU open, US open)
    pub tight_quoting_excluded_hours: Vec<u8>,

    // ==================== Adaptive Bayesian System ====================
    /// Enable adaptive Bayesian spread calculator.
    /// When true, uses learned spread floor, blended kappa, and shrinkage gamma.
    /// This replaces static parameters with online-learned values.
    /// Default: false (conservative - enable after testing)
    pub use_adaptive_spreads: bool,

    /// Baseline volatility for adaptive gamma scaling (per-second σ).
    /// This is duplicated from RiskConfig for module isolation.
    /// Default: 0.0002 (20 bps per second)
    pub sigma_baseline: f64,

    // ==================== Entropy-Based Distribution ====================
    // Entropy-based stochastic order distribution is always enabled.
    // Key features:
    // - Minimum entropy floor ensures at least N effective levels always active
    // - Softmax temperature controls distribution spread
    // - Thompson sampling adds controlled randomness
    // - Dirichlet smoothing prevents zero allocations

    /// Minimum entropy floor (bits).
    /// H_min = 1.5 → at least exp(1.5) ≈ 4.5 effective levels always active.
    /// H_min = 2.0 → at least exp(2.0) ≈ 7.4 effective levels always active.
    /// Default: 1.5
    pub entropy_min_entropy: f64,

    /// Base temperature for softmax distribution.
    /// Higher = more uniform distribution, lower = more concentrated.
    /// T = 1.0: Standard softmax
    /// T = 2.0: Twice as uniform
    /// T = 0.5: Twice as concentrated
    /// Default: 1.0
    pub entropy_base_temperature: f64,

    /// Minimum allocation floor per level (prevents zero allocations).
    /// floor = 0.02 → each level gets at least 2% of total capacity.
    /// Default: 0.02
    pub entropy_min_allocation_floor: f64,

    /// Number of Thompson samples for stochastic allocation.
    /// Higher = more stable but less explorative.
    /// n = 1: Pure Thompson (most random)
    /// n = 5: Averaged Thompson (moderate)
    /// n = 20: Quasi-deterministic
    /// Default: 5
    pub entropy_thompson_samples: usize,

    // ==================== Calibration Fill Rate Controller ====================
    /// Enable calibration-aware fill rate targeting.
    /// During warmup, reduces gamma to attract fills for parameter calibration.
    /// Automatically phases out as calibration completes.
    ///
    /// Default: true
    pub enable_calibration_fill_rate: bool,

    /// Target fill rate per hour (across all levels).
    /// The controller adjusts gamma to achieve this fill rate during warmup.
    /// 10 fills/hour ≈ 2 fills/level for a 5-level ladder.
    ///
    /// Default: 10.0
    pub target_fill_rate_per_hour: f64,

    /// Minimum gamma multiplier during fill-hungry mode.
    /// 0.3 = allow up to 70% gamma reduction (tighter quotes).
    /// Lower values = more aggressive fill seeking.
    ///
    /// Default: 0.3
    pub min_fill_hungry_gamma: f64,

    // ==================== Microprice EMA Smoothing ====================
    /// EMA smoothing factor for microprice output (0.0-1.0).
    /// Higher = more weight to new observations (more reactive).
    /// 0.2 = 5-update half-life, 0.1 = 10-update half-life, 0.0 = disabled
    ///
    /// Default: 0.2
    pub microprice_ema_alpha: f64,

    /// Minimum change in bps to update microprice EMA (noise filter).
    /// Changes smaller than this are ignored to reduce quote volatility.
    ///
    /// Default: 2.0
    pub microprice_ema_min_change_bps: f64,
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self {
            // Feature flags
            use_hjb_skew: true,
            use_kalman_filter: true,

            // Calibration flags - ON by default (passive, safe)
            calibrate_depth_as: true,
            use_calibrated_as: true,

            // Kalman parameters
            kalman_q: 1e-8,   // 1 bp² per tick
            kalman_r: 2.5e-9, // 0.5 bp² observation noise

            // HJB parameters
            hjb_session_duration: 86400.0, // 24 hour session
            hjb_terminal_penalty: 0.0005,  // 0.05% per unit²
            hjb_funding_half_life: 3600.0, // 1 hour

            // Kelly-Stochastic parameters
            // CALIBRATED from trade_history.csv (2,038 trades, Dec 2025):
            // - 42.5% win rate suggests ~25% informed flow at touch
            // - Large trades (>$2k) had -11.58 bps edge, 14.3% win rate
            // - Conservatively set alpha higher than measured AS
            kelly_alpha_touch: 0.25,     // 25% informed at touch (was 0.15)
            kelly_alpha_decay_bps: 15.0, // Slower decay for wider protection (was 10)
            kelly_fraction: 0.20,        // Conservative 1/5 Kelly (was 0.25)

            // Kelly time horizon parameters
            kelly_time_horizon_method: KellyTimeHorizonMethod::DiffusionBased,
            kelly_tau_fixed: 60.0,      // 1 minute (for Fixed method)
            kelly_tau_min: 10.0,        // 10 seconds floor
            kelly_tau_max: 600.0,       // 10 minutes ceiling
            kelly_char_depth_bps: 25.0, // Mid-ladder characteristic depth

            // Stochastic Constraints (First Principles)
            use_latency_spread_floor: true,
            quote_update_latency_ms: 50.0, // 50ms expected round-trip
            use_book_depth_constraint: true,
            min_book_depth_usd: 20_000.0, // $20k minimum (reduced from $50k)
            tight_spread_book_depth_usd: 100_000.0, // $100k for tightest spreads (reduced from $200k)
            use_conditional_tight_quoting: true,
            tight_quoting_max_inventory: 0.3, // 30% of max position
            tight_quoting_max_toxicity: 0.1,  // 10% predicted alpha
            tight_quoting_excluded_hours: vec![7, 14], // EU open, US open

            // Adaptive Bayesian System
            // ENABLED by default - uses learned parameters instead of static ones:
            // - Learned spread floor from Bayesian AS estimation
            // - Blended kappa from book depth + own fills
            // - Log-additive shrinkage gamma (prevents multiplicative explosion)
            // - Fill rate targeting with spread ceiling
            use_adaptive_spreads: true,
            sigma_baseline: 0.0002, // 20 bps per second (matches RiskConfig)

            // Entropy-Based Distribution (always enabled)
            entropy_min_entropy: 1.5,      // At least ~4.5 effective levels
            entropy_base_temperature: 1.0, // Standard softmax
            entropy_min_allocation_floor: 0.02, // 2% minimum per level
            entropy_thompson_samples: 5,   // Moderate stochasticity

            // Calibration Fill Rate Controller
            // ENABLED by default - ensures fills for parameter calibration
            enable_calibration_fill_rate: true,
            target_fill_rate_per_hour: 10.0, // 10 fills/hour = ~2/level for 5 levels
            min_fill_hungry_gamma: 0.3,      // Max 70% gamma reduction

            // Microprice EMA Smoothing
            // ENABLED by default - reduces quote volatility from microprice noise
            microprice_ema_alpha: 0.2,          // 5-update half-life
            microprice_ema_min_change_bps: 2.0, // 2 bps noise filter
        }
    }
}

impl StochasticConfig {
    /// Create config with all stochastic features enabled.
    pub fn all_enabled() -> Self {
        Self {
            use_hjb_skew: true,
            use_kalman_filter: true,
            calibrate_depth_as: true,
            use_calibrated_as: true,
            ..Default::default()
        }
    }

    /// Create config with only passive calibration (no quote changes).
    pub fn passive_only() -> Self {
        Self {
            use_hjb_skew: false,
            use_kalman_filter: false,
            calibrate_depth_as: true,
            use_calibrated_as: false,
            ..Default::default()
        }
    }

    /// Builder: enable HJB skew.
    pub fn with_hjb_skew(mut self) -> Self {
        self.use_hjb_skew = true;
        self
    }

    /// Builder: enable Kalman filter.
    pub fn with_kalman_filter(mut self) -> Self {
        self.use_kalman_filter = true;
        self
    }

    /// Builder: set Kelly alpha at touch.
    pub fn with_kelly_alpha_touch(mut self, alpha: f64) -> Self {
        self.kelly_alpha_touch = alpha;
        self
    }

    /// Builder: set Kelly fraction.
    pub fn with_kelly_fraction(mut self, fraction: f64) -> Self {
        self.kelly_fraction = fraction;
        self
    }

    /// Builder: set Kalman Q (process noise).
    pub fn with_kalman_q(mut self, q: f64) -> Self {
        self.kalman_q = q;
        self
    }

    /// Builder: set Kalman R (observation noise).
    pub fn with_kalman_r(mut self, r: f64) -> Self {
        self.kalman_r = r;
        self
    }

    /// Builder: set Kelly time horizon method.
    pub fn with_kelly_time_horizon_method(mut self, method: KellyTimeHorizonMethod) -> Self {
        self.kelly_time_horizon_method = method;
        self
    }

    /// Builder: set Kelly characteristic depth for diffusion-based tau.
    pub fn with_kelly_char_depth_bps(mut self, depth: f64) -> Self {
        self.kelly_char_depth_bps = depth;
        self
    }

    /// Builder: set Kelly tau min (floor for diffusion-based).
    pub fn with_kelly_tau_min(mut self, tau_min: f64) -> Self {
        self.kelly_tau_min = tau_min;
        self
    }

    /// Builder: set Kelly tau max (ceiling).
    pub fn with_kelly_tau_max(mut self, tau_max: f64) -> Self {
        self.kelly_tau_max = tau_max;
        self
    }

    /// Builder: set fixed Kelly tau (for Fixed method).
    pub fn with_kelly_tau_fixed(mut self, tau: f64) -> Self {
        self.kelly_tau_fixed = tau;
        self
    }
}
