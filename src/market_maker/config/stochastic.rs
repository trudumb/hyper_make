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

    // ==================== Calibrated Risk Model (Log-Additive Gamma) ====================
    /// Feature flag: use log-additive calibrated risk model instead of multiplicative.
    /// When enabled, gamma is computed as: γ = exp(log_gamma_base + Σ βᵢ × xᵢ)
    /// This prevents multiplicative explosion that can occur with 11+ scalars.
    /// Default: false (conservative - enable after shadow mode validation)
    pub use_calibrated_risk_model: bool,

    /// Blend factor for gradual rollout (0=old multiplicative, 1=new log-additive).
    /// Use values in [0.0, 1.0] to gradually transition between models.
    /// Default: 0.0 (start with old model)
    pub risk_model_blend: f64,

    /// Baseline kappa for risk feature normalization.
    /// Used to compute excess_intensity = (κ_activity - κ_baseline) / κ_baseline.
    /// Default: 2500.0 (reasonable prior for liquid markets)
    pub kappa_baseline: f64,

    /// Baseline book depth for risk feature normalization (USD).
    /// Used to compute depth_depletion = 1 - depth/depth_baseline.
    /// Default: 100,000 ($100k baseline)
    pub book_depth_baseline_usd: f64,

    /// Minimum samples required before using calibrated coefficients.
    /// During warmup, conservative defaults (50% higher betas) are used.
    /// Default: 100
    pub min_calibration_samples: usize,

    /// Hours after which calibration is considered stale.
    /// When stale, model blends toward conservative defaults.
    /// Default: 4.0
    pub calibration_staleness_hours: f64,

    // ==================== Kelly Criterion Sizing ====================
    /// Feature flag: use Kelly criterion for position sizing.
    /// When enabled, size = kelly_fraction × bankroll × f*, where f* = (pb - q) / b.
    /// Default: false (conservative - enable after tracker warmup)
    pub use_kelly_sizing: bool,

    /// Minimum P(edge > 0) required to take a position.
    /// Below this threshold, Kelly returns zero size.
    /// Default: 0.55 (55% confidence required)
    pub kelly_min_p_win: f64,

    /// Minimum expected edge (bps) required to take a position.
    /// Below this threshold, Kelly returns zero size.
    /// Default: 2.0 (2 bps minimum edge)
    pub kelly_min_edge_bps: f64,

    /// Maximum position as fraction of bankroll.
    /// Caps the Kelly fraction to limit concentration risk.
    /// Default: 0.5 (50% max)
    pub kelly_max_position_fraction: f64,

    /// EWMA decay factor for win/loss tracking.
    /// Higher values = slower adaptation (0.99 ≈ 100 trade half-life).
    /// Default: 0.99
    pub kelly_tracker_decay: f64,

    /// Minimum trades before using Kelly (warmup period).
    /// During warmup, uses 50% of position limit instead.
    /// Default: 20
    pub kelly_min_warmup_trades: usize,

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

    // ==================== Proactive Position Management ====================
    // Phase 1: Time-Based Position Ramp
    /// Enable time-based position ramping.
    /// When true, max position starts at initial_fraction and ramps up over time.
    /// Default: true
    pub enable_position_ramp: bool,

    /// Time to reach full position capacity (seconds).
    /// Default: 1800 (30 minutes)
    pub ramp_duration_secs: f64,

    /// Starting fraction of max position (0.0 - 1.0).
    /// Default: 0.10 (10%)
    pub ramp_initial_fraction: f64,

    /// Ramp curve type: "linear", "sqrt", "log".
    /// sqrt is recommended: fast start, slow finish.
    /// Default: "sqrt"
    pub ramp_curve: String,

    // Phase 2: Confidence-Gated Sizing
    /// Enable confidence-gated sizing.
    /// When true, quote size scales with model confidence.
    /// Default: true
    pub enable_confidence_sizing: bool,

    /// Minimum size fraction when confidence is zero.
    /// 0.3 = always quote at least 30% size.
    /// Default: 0.3
    pub confidence_min_size_fraction: f64,

    // Phase 3: Proactive Directional Skew
    /// Enable proactive directional skew based on momentum predictions.
    /// When true, skews quotes to BUILD position with momentum (before getting filled).
    /// Default: true
    pub enable_proactive_skew: bool,

    /// Sensitivity for proactive skew (bps per unit momentum×confidence).
    /// Higher = more aggressive position building with momentum.
    /// Default: 2.0
    pub proactive_skew_sensitivity: f64,

    /// Minimum momentum confidence to apply proactive skew.
    /// Below this, no proactive skew is applied.
    /// Default: 0.6
    pub proactive_min_momentum_confidence: f64,

    /// Minimum momentum strength (bps) to apply proactive skew.
    /// Filters out noise from weak momentum signals.
    /// Default: 5.0
    pub proactive_min_momentum_bps: f64,

    // Phase 4: Performance-Gated Capacity
    /// Enable performance-based capacity gating.
    /// When true, max position scales based on realized P&L.
    /// Default: true
    pub enable_performance_gating: bool,

    /// Loss reduction multiplier for performance gating.
    /// 2.0 = capacity_reduction = 2 × loss_ratio.
    /// Default: 2.0
    pub performance_loss_reduction_mult: f64,

    /// Minimum capacity fraction (floor) for performance gating.
    /// 0.3 = never go below 30% of max position.
    /// Default: 0.3
    pub performance_min_capacity_fraction: f64,

    // ==================== Quote Gate (Directional Edge Gating) ====================
    /// Enable Quote Gate for directional edge-based quoting decisions.
    /// When enabled, the system will NOT quote when there's no directional edge.
    /// This prevents whipsaw losses from random fills.
    ///
    /// Default: true
    pub enable_quote_gate: bool,

    /// Minimum |flow_imbalance| to have directional edge.
    /// Below this threshold, we consider ourselves "edgeless" and may not quote.
    ///
    /// Default: 0.25
    pub quote_gate_min_edge_signal: f64,

    /// Minimum momentum confidence to trust the edge signal.
    /// Only required when signal is weak (below strong_signal_threshold).
    ///
    /// Default: 0.45 (below baseline 0.50)
    pub quote_gate_min_edge_confidence: f64,

    /// Signal strength that bypasses confidence requirement.
    /// If |flow_imbalance| >= this, we trust it regardless of confidence.
    ///
    /// Default: 0.50 (strong signal = trust it)
    pub quote_gate_strong_signal_threshold: f64,

    /// Minimum position (as fraction of max) to trigger one-sided quoting.
    /// Below this, position is considered "flat".
    ///
    /// Default: 0.05 (5%)
    pub quote_gate_position_threshold: f64,

    /// Maximum position (as fraction of max) before ONLY reducing.
    /// Above this, we become very defensive and only quote to reduce position.
    ///
    /// Default: 0.7 (70%)
    pub quote_gate_max_position_before_reduce_only: f64,

    /// Enable cascade protection in Quote Gate (pull all quotes during cascade).
    ///
    /// Default: true
    pub quote_gate_cascade_protection: bool,

    /// Cascade threshold (cascade_size_factor below this = cascade).
    ///
    /// Default: 0.3 (70% cascade severity)
    pub quote_gate_cascade_threshold: f64,

    /// Quote both sides when flat, even without strong edge signal.
    /// When true (market-making mode): quotes both sides for spread capture.
    /// When false (API budget mode): only quotes when directional edge detected.
    ///
    /// Default: false (API budget conservation - prevents excessive API churn)
    /// Use --quote-flat-without-edge to enable market-making mode.
    pub quote_gate_flat_without_edge: bool,

    // ==================== Calibrated Quote Gate (IR-Based Thresholds) ====================
    /// Enable calibrated quote gate with IR-based thresholds.
    /// When enabled, replaces arbitrary thresholds (0.15, 0.45, etc.) with:
    /// - Edge signal: IR > 1.0 means signal adds value
    /// - Position threshold: Derived from P&L data
    /// - Reduce-only threshold: Regime-specific from P&L data
    /// - Cascade detection: Uses changepoint probability
    ///
    /// Default: false (enable after collecting calibration data)
    pub enable_calibrated_quote_gate: bool,

    // ==================== Predictive Bias Extension (A-S Model) ====================
    /// Sensitivity for predictive bias β_t calculation.
    /// β_t = -sensitivity × prob_excess × σ
    /// Higher values = more aggressive skew when changepoint is detected.
    /// Default: 2.0 (expect 2σ move on confirmed changepoint)
    ///
    /// DERIVATION: From changepoint analysis - E[move | CP] / σ ≈ 2
    /// This means when a changepoint is confirmed, expect a 2σ move.
    /// The prior Normal(2.0, 1.0²) reflects uncertainty in this estimate.
    pub predictive_bias_sensitivity: f64,

    /// Minimum changepoint probability to activate predictive bias.
    /// Below this threshold, no predictive bias is applied.
    /// Default: 0.3
    ///
    /// DERIVATION: From ROC optimization on changepoint detection.
    /// Cost asymmetry: missing a real changepoint (FN) is 3× worse than
    /// false alarm (FP), so threshold is set at 0.3 instead of 0.5.
    pub predictive_bias_threshold: f64,

    // ==================== Learned Parameters Integration ====================
    /// Enable use of Bayesian learned parameters for magic number replacement.
    /// When true, parameters like alpha_touch, spread_floor, etc. are learned
    /// from data with Bayesian regularization instead of using static defaults.
    ///
    /// Default: true (enables statistical grounding)
    pub use_learned_parameters: bool,

    /// Minimum observations before trusting learned parameters over priors.
    /// Below this threshold, parameters shrink heavily toward their priors.
    ///
    /// DERIVATION: Power analysis for IR CI width < 0.2 requires ~96 samples.
    /// We use 100 as a round number.
    ///
    /// Default: 100
    pub learned_param_min_observations: usize,

    /// Maximum coefficient of variation for a parameter to be considered calibrated.
    /// Higher CV means more uncertainty in the estimate.
    ///
    /// Default: 0.5 (50% CV is acceptable for most parameters)
    pub learned_param_max_cv: f64,

    /// Hours before learned parameters are considered stale and need recalibration.
    ///
    /// DERIVATION: Signal decay analysis shows predictive power halves every ~4 hours
    /// in typical market conditions.
    ///
    /// Default: 4.0
    pub learned_param_staleness_hours: f64,
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
            // CHANGED: Increased defaults to enforce better distribution with small capital
            entropy_min_entropy: 2.2,      // At least ~9.0 effective levels to keep most of 10 active
            entropy_base_temperature: 1.5, // More uniform → fewer levels filtered by min notional
            entropy_min_allocation_floor: 0.04, // 4% minimum per level (was 2%)
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

            // Proactive Position Management
            // Phase 1: Time-Based Position Ramp
            enable_position_ramp: true,
            ramp_duration_secs: 1800.0,      // 30 minutes
            ramp_initial_fraction: 0.10,     // Start at 10%
            ramp_curve: "sqrt".to_string(),  // Fast start, slow finish

            // Phase 2: Confidence-Gated Sizing
            enable_confidence_sizing: true,
            confidence_min_size_fraction: 0.3, // Always at least 30%

            // Phase 3: Proactive Directional Skew
            enable_proactive_skew: true,
            proactive_skew_sensitivity: 2.0,         // bps per unit momentum×confidence
            proactive_min_momentum_confidence: 0.6,  // 60% confidence threshold
            proactive_min_momentum_bps: 5.0,         // 5 bps minimum momentum

            // Phase 4: Performance-Gated Capacity
            enable_performance_gating: true,
            performance_loss_reduction_mult: 2.0,     // 2x loss ratio
            performance_min_capacity_fraction: 0.3,  // Never below 30%

            // Quote Gate (Directional Edge Gating)
            // ENABLED by default - optimized for API budget conservation
            enable_quote_gate: true,
            quote_gate_min_edge_signal: 0.15,               // Lower threshold for MM (was 0.25)
            quote_gate_min_edge_confidence: 0.45,           // Below baseline (was 0.55)
            quote_gate_strong_signal_threshold: 0.50,       // Strong signal bypasses confidence
            quote_gate_position_threshold: 0.05,            // 5% of max = "flat"
            quote_gate_max_position_before_reduce_only: 0.7, // 70% of max = reduce only
            quote_gate_cascade_protection: true,
            quote_gate_cascade_threshold: 0.3,              // 70% cascade severity
            // MARKET MAKING MODE: Quote both sides even without edge signal
            // Market makers profit from spread capture, not direction
            quote_gate_flat_without_edge: true,             // Enable market-making mode

            // Calibrated Quote Gate (IR-Based Thresholds)
            // ENABLED: Uses IR > 1.0 instead of arbitrary 0.15 threshold
            enable_calibrated_quote_gate: true,

            // Predictive Bias Extension (A-S Model)
            // Sensitivity and threshold for predictive skew from changepoint detection
            predictive_bias_sensitivity: 2.0, // 2σ expected move on confirmed changepoint
            predictive_bias_threshold: 0.3,   // Activate when cp_prob > 30%

            // Calibrated Risk Model (Log-Additive Gamma)
            // ENABLED: Use log-additive gamma to prevent multiplicative explosion
            // The old system could compound 11 scalars to 45x; this bounds via exp(sum)
            use_calibrated_risk_model: true,
            risk_model_blend: 1.0,              // Full cutover to new model
            kappa_baseline: 2500.0,             // Prior for liquid markets
            book_depth_baseline_usd: 100_000.0, // $100k baseline
            min_calibration_samples: 100,
            calibration_staleness_hours: 4.0,

            // Kelly Criterion Sizing
            // DISABLED by default - enable after tracker warmup
            use_kelly_sizing: false,
            kelly_min_p_win: 0.55,
            kelly_min_edge_bps: 2.0,
            kelly_max_position_fraction: 0.5,
            kelly_tracker_decay: 0.99,
            kelly_min_warmup_trades: 20,

            // Learned Parameters Integration
            // ENABLED: Replace magic numbers with Bayesian-learned values
            use_learned_parameters: true,
            learned_param_min_observations: 100, // Power analysis: N for IR CI width < 0.2
            learned_param_max_cv: 0.5,           // 50% CV acceptable
            learned_param_staleness_hours: 4.0,  // Signal decay half-life
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

    /// Create a QuoteGateConfig from this StochasticConfig.
    pub fn quote_gate_config(&self) -> crate::market_maker::control::QuoteGateConfig {
        crate::market_maker::control::QuoteGateConfig {
            enabled: self.enable_quote_gate,
            min_edge_signal: self.quote_gate_min_edge_signal,
            min_edge_confidence: self.quote_gate_min_edge_confidence,
            strong_signal_threshold: self.quote_gate_strong_signal_threshold,
            position_threshold: self.quote_gate_position_threshold,
            max_position_before_reduce_only: self.quote_gate_max_position_before_reduce_only,
            cascade_protection: self.quote_gate_cascade_protection,
            cascade_threshold: self.quote_gate_cascade_threshold,
            quote_flat_without_edge: self.quote_gate_flat_without_edge,
            use_bayesian_warmup: true, // Enable Bayesian IR warmup by default
            min_ir_outcomes_for_trust: 25, // Default: require meaningful IR data
            probe_config: crate::market_maker::control::ProbeConfig::default(),
            bootstrap_config: crate::market_maker::control::BayesianBootstrapConfig::default(),
            // Default to ThinDex for conservative changepoint thresholds
            market_regime: crate::market_maker::control::MarketRegime::ThinDex,
            // Pre-fill AS toxicity gating - enabled by default
            toxicity_gate_threshold: 0.75, // 75% toxicity = skip quoting
            enable_toxicity_gate: true,
            // Quota shadow pricing - use defaults
            quota_shadow: crate::market_maker::control::quote_gate::QuotaShadowConfig::default(),
        }
    }
}
