//! Market parameters for quoting strategies.

use crate::market_maker::adverse_selection::DepthDecayAS;
use crate::market_maker::estimator::VolatilityRegime;
use crate::market_maker::process_models::SpreadRegime;

use super::params;

/// Parameters estimated from live market data.
///
/// For the infinite-horizon GLFT model with regime detection and directional protection:
/// - Dual-sigma: sigma_clean (BV-based) for spreads, sigma_effective (blended) for skew
/// - kappa: order book depth decay constant (from weighted L2 book regression)
/// - arrival_intensity: volume ticks per second
/// - Regime detection: jump_ratio > 1.5 = toxic
/// - Directional flow: momentum_bps, flow_imbalance, falling/rising knife scores
#[derive(Debug, Clone)]
pub struct MarketParams {
    // === Volatility (all per-second, NOT annualized) ===
    /// Clean volatility (σ_clean) - √BV, robust to jumps
    /// Use for base spread calculation (continuous risk)
    pub sigma: f64,

    /// Total volatility (σ_total) - √RV, includes jumps
    /// Captures full price variance including discontinuities
    pub sigma_total: f64,

    /// Effective volatility (σ_effective) - blended clean/total
    /// Reacts to jump regime; use for inventory skew
    pub sigma_effective: f64,

    /// Leverage-adjusted effective volatility
    /// Wider during down moves when ρ < 0 (leverage effect)
    /// Use for asymmetric inventory skew in falling markets
    pub sigma_leverage_adjusted: f64,

    // === Order Book ===
    /// Estimated order book depth decay (κ) - from weighted L2 book regression
    pub kappa: f64,

    /// Directional kappa for bid side (our buy fills).
    /// Theory: When informed flow is selling, κ_bid < κ_ask → wider bid spread.
    /// Use for asymmetric GLFT: δ_bid = (1/γ) × ln(1 + γ/κ_bid)
    pub kappa_bid: f64,

    /// Directional kappa for ask side (our sell fills).
    /// Theory: When informed flow is buying, κ_ask < κ_bid → wider ask spread.
    /// Use for asymmetric GLFT: δ_ask = (1/γ) × ln(1 + γ/κ_ask)
    pub kappa_ask: f64,

    /// Whether fill distance distribution is heavy-tailed (CV > 1.2).
    /// Heavy-tailed distributions mean occasional large fills are more likely.
    /// When true, kappa values should be treated more conservatively.
    pub is_heavy_tailed: bool,

    /// Coefficient of variation of fill distances (CV = σ/μ).
    /// CV = 1.0 for exponential, CV > 1.2 indicates heavy tail (power-law like).
    pub kappa_cv: f64,

    // === V3: Robust Kappa Orchestrator ===
    /// Robust kappa from V3 orchestrator (outlier-resistant).
    /// Blends book-structure κ, Student-t robust κ, and own-fill κ.
    /// Use this instead of `kappa` for illiquid/HIP-3 markets.
    pub kappa_robust: f64,

    /// Whether to use kappa_robust instead of kappa for spread calculation.
    /// When true, ladder_strat uses kappa_robust for GLFT formula.
    pub use_kappa_robust: bool,

    /// Number of outliers detected by robust estimator.
    /// High count indicates market has heavy-tailed trade distances.
    pub kappa_outlier_count: u64,

    // === V2: Uncertainty Quantification (Bayesian Estimator) ===
    /// Kappa posterior standard deviation (√Var[κ|data]).
    /// Use for spread uncertainty: δσ² ≈ (∂δ/∂κ)² × σ²_κ
    pub kappa_uncertainty: f64,

    /// 95% credible interval lower bound for kappa.
    /// κ_95_lower = posterior_alpha / posterior_beta quantile(0.025)
    pub kappa_95_lower: f64,

    /// 95% credible interval upper bound for kappa.
    /// κ_95_upper = posterior_alpha / posterior_beta quantile(0.975)
    pub kappa_95_upper: f64,

    /// Soft toxicity score [0, 1] from mixture model.
    /// Rolling average of P(jump) - smoother than binary is_toxic_regime.
    /// 0.0 = pure diffusion, 0.5+ = significant jump component
    pub toxicity_score: f64,

    /// (κ, σ) correlation coefficient [-1, 1].
    /// Positive correlation means kappa and sigma move together.
    /// Used for joint uncertainty propagation in spread calculations.
    pub param_correlation: f64,

    /// Adverse selection factor φ(AS) ∈ [0.5, 1.0].
    /// Multiplied with kappa to get effective fill rate.
    /// Lower values mean more informed flow → lower effective kappa.
    pub as_factor: f64,

    /// Order arrival intensity (A) - volume ticks per second
    pub arrival_intensity: f64,

    // === Regime Detection ===
    /// Whether market is in toxic (jump) regime: RV/BV > 1.5
    pub is_toxic_regime: bool,

    /// RV/BV jump ratio: ≈1.0 = normal diffusion, >1.5 = toxic
    pub jump_ratio: f64,

    // === Directional Flow ===
    /// Signed momentum over 500ms window (in bps)
    /// Negative = market falling, Positive = market rising
    pub momentum_bps: f64,

    /// Order flow imbalance [-1, 1]
    /// Negative = sell pressure, Positive = buy pressure
    pub flow_imbalance: f64,

    /// Falling knife score [0, 3]
    /// > 0.5 = some downward momentum, > 1.0 = severe (protect bids!)
    pub falling_knife_score: f64,

    /// Rising knife score [0, 3]
    /// > 0.5 = some upward momentum, > 1.0 = severe (protect asks!)
    pub rising_knife_score: f64,

    // === L2 Book Structure ===
    /// L2 book imbalance [-1, 1]
    /// Positive = more bids (buying pressure), Negative = more asks (selling pressure)
    /// Use for directional quote skew adjustment
    pub book_imbalance: f64,

    /// Liquidity-based gamma multiplier [1.0, 2.0]
    /// Returns values greater than 1.0 when near-touch liquidity is below average (thin book).
    /// Scales gamma up for wider spreads in thin conditions.
    pub liquidity_gamma_mult: f64,

    // === Microprice (Data-Driven Fair Price) ===
    /// Microprice - fair price incorporating signal predictions.
    /// Quote around this instead of raw mid.
    pub microprice: f64,

    /// Actual market mid price (from AllMids exchange data).
    /// Used for safety checks to prevent crossing the spread.
    /// This is the raw exchange mid, NOT adjusted by microprice model.
    pub market_mid: f64,

    /// β_book coefficient (return prediction per unit book imbalance).
    /// For diagnostics: expected range ~0.001-0.01 (1-10 bps per unit signal).
    pub beta_book: f64,

    /// β_flow coefficient (return prediction per unit flow imbalance).
    /// For diagnostics: expected range ~0.001-0.01 (1-10 bps per unit signal).
    pub beta_flow: f64,

    // === Tier 1: Adverse Selection ===
    /// Spread adjustment from AS estimator (as fraction of mid price).
    /// Add to half-spread to compensate for informed flow.
    pub as_spread_adjustment: f64,

    /// Predicted alpha: P(next trade is informed) in [0, 1].
    /// Use for diagnostics and future alpha-aware sizing.
    pub predicted_alpha: f64,

    /// Is AS estimator warmed up with enough fills?
    pub as_warmed_up: bool,

    /// Depth-dependent AS model (calibrated from fills).
    /// First-principles: AS(δ) = AS₀ × exp(-δ/δ_char)
    /// Used by ladder for depth-aware spread capture.
    pub depth_decay_as: Option<DepthDecayAS>,

    // === Tier 1: Liquidation Cascade ===
    /// Tail risk multiplier for gamma [1.0, 5.0].
    /// Multiply gamma by this during cascade conditions.
    pub tail_risk_multiplier: f64,

    /// Should pull all quotes due to extreme cascade?
    pub should_pull_quotes: bool,

    /// Size reduction factor [0, 1] for graceful degradation.
    /// 1.0 = full size, 0.0 = no quotes (cascade severe).
    pub cascade_size_factor: f64,

    // === Tier 2: Hawkes Order Flow ===
    /// Hawkes buy intensity (λ_buy) - self-exciting arrival rate
    pub hawkes_buy_intensity: f64,

    /// Hawkes sell intensity (λ_sell) - self-exciting arrival rate
    pub hawkes_sell_intensity: f64,

    /// Hawkes flow imbalance [-1, 1] - normalized buy/sell intensity difference
    pub hawkes_imbalance: f64,

    /// Hawkes activity percentile [0, 1] - where current intensity sits in history
    pub hawkes_activity_percentile: f64,

    // === Tier 2: Funding Rate ===
    /// Current funding rate (annualized)
    pub funding_rate: f64,

    /// Predicted funding cost for holding period
    pub predicted_funding_cost: f64,

    // === Tier 2: Spread Process ===
    /// Fair spread from vol-adjusted model
    pub fair_spread: f64,

    /// Spread percentile [0, 1] - where current spread sits in history
    pub spread_percentile: f64,

    /// Spread regime (Tight/Normal/Wide)
    pub spread_regime: SpreadRegime,

    /// Observed market spread in basis points (best_ask - best_bid) / mid × 10000
    /// Used to cap GLFT optimal spread to stay competitive
    pub market_spread_bps: f64,

    // === Volatility Regime ===
    /// Volatility regime (Low/Normal/High/Extreme)
    pub volatility_regime: VolatilityRegime,

    // === First Principles Extensions (Gaps 1-10) ===

    // Jump-Diffusion (Gap 1)
    /// Jump intensity λ (expected jumps per second)
    pub lambda_jump: f64,
    /// Mean jump size μ_j
    pub mu_jump: f64,
    /// Jump size standard deviation σ_j
    pub sigma_jump: f64,

    // Stochastic Volatility (Gap 2)
    /// Vol mean-reversion speed κ_vol
    pub kappa_vol: f64,
    /// Long-run volatility θ_vol
    pub theta_vol: f64,
    /// Vol-of-vol ξ
    pub xi_vol: f64,
    /// Price-vol correlation ρ (leverage effect)
    pub rho_price_vol: f64,

    // Queue Model (Gap 3)
    /// Calibrated volume at touch rate (units/sec)
    pub calibrated_volume_rate: f64,
    /// Calibrated cancel rate (fraction/sec)
    pub calibrated_cancel_rate: f64,

    // Funding Enhancement (Gap 6)
    /// Mark-index premium
    pub premium: f64,
    /// Premium alpha signal
    pub premium_alpha: f64,

    // Momentum Protection (Gap 10)
    /// Bid protection factor (>= 1.0 when market falling)
    pub bid_protection_factor: f64,
    /// Ask protection factor (>= 1.0 when market rising)
    pub ask_protection_factor: f64,
    /// Probability momentum continues
    pub p_momentum_continue: f64,

    // === Stochastic Module Integration (HJB Controller) ===
    /// Whether to use HJB optimal skew instead of heuristic inventory_skew_with_flow
    pub use_hjb_skew: bool,
    /// HJB optimal inventory skew (from Avellaneda-Stoikov HJB solution)
    /// Formula: γσ²qT + terminal_penalty × q × urgency + funding_bias
    pub hjb_optimal_skew: f64,
    /// HJB gamma multiplier (for logging/diagnostics)
    pub hjb_gamma_multiplier: f64,
    /// HJB inventory target (optimal q* for current session state)
    pub hjb_inventory_target: f64,
    /// Whether HJB controller is in terminal zone (near session end)
    pub hjb_is_terminal_zone: bool,

    // === Drift-Adjusted Skew (First Principles Extension) ===
    /// Whether to use drift-adjusted skew from momentum signals.
    /// When true, HJB skew incorporates predicted price drift.
    pub use_drift_adjusted_skew: bool,
    /// Drift urgency component of HJB skew (from momentum-position opposition).
    /// Positive when position needs urgent reduction (opposed to momentum).
    pub hjb_drift_urgency: f64,
    /// Variance multiplier from directional risk.
    /// > 1.0 when position opposes momentum (increased risk).
    pub directional_variance_mult: f64,
    /// Whether position opposes momentum (short + rising, long + falling).
    pub position_opposes_momentum: bool,
    /// Urgency score [0, 5] combining all urgency factors.
    /// 0 = no urgency, 5 = maximum urgency (should cover/liquidate ASAP).
    pub urgency_score: f64,

    // === Stochastic Module Integration (Kalman Filter) ===
    /// Whether to use Kalman filter spread widening
    pub use_kalman_filter: bool,
    /// Kalman-filtered fair price (denoised mid)
    pub kalman_fair_price: f64,
    /// Kalman filter uncertainty (posterior std dev)
    pub kalman_uncertainty: f64,
    /// Kalman-based spread widening: γ × σ_kalman × √T
    pub kalman_spread_widening: f64,
    /// Whether Kalman filter is warmed up
    pub kalman_warmed_up: bool,

    // === Stochastic Module Integration (Constrained Optimizer) ===
    /// Whether to use constrained ladder optimizer
    pub use_constrained_optimizer: bool,
    /// Available margin for order placement (USD)
    pub margin_available: f64,
    /// Current leverage ratio
    pub leverage: f64,

    // === Stochastic Module Integration (Kelly-Stochastic Allocation) ===
    /// Whether to use Kelly-Stochastic allocation (requires use_constrained_optimizer=true)
    pub use_kelly_stochastic: bool,
    /// Informed trader probability at the touch (0.0-1.0)
    pub kelly_alpha_touch: f64,
    /// Characteristic depth for alpha decay in bps
    pub kelly_alpha_decay_bps: f64,
    /// Kelly fraction (0.25 = quarter Kelly)
    pub kelly_fraction: f64,
    /// Kelly-specific time horizon for first-passage probability (seconds).
    /// Semantically different from GLFT inventory time horizon T = 1/λ.
    /// Computed from KellyTimeHorizonMethod in config.
    pub kelly_time_horizon: f64,

    // === Exchange Position Limits (Pre-flight Rejection Prevention) ===
    /// Whether exchange limits have been fetched and are valid
    pub exchange_limits_valid: bool,
    /// Effective bid limit: min(local_max, exchange_available_buy)
    /// Use this to cap bid ladder sizes
    pub exchange_effective_bid_limit: f64,
    /// Effective ask limit: min(local_max, exchange_available_sell)
    /// Use this to cap ask ladder sizes
    pub exchange_effective_ask_limit: f64,
    /// Age of exchange limits in milliseconds (for staleness warnings)
    pub exchange_limits_age_ms: u64,

    // === Pending Exposure (Resting Order State) ===
    /// Total remaining size on resting buy orders (would increase long if filled)
    pub pending_bid_exposure: f64,
    /// Total remaining size on resting sell orders (would increase short if filled)
    pub pending_ask_exposure: f64,

    // === Dynamic Position Limits (First Principles) ===
    /// Dynamic max position SIZE derived from first-principles VALUE limit.
    /// Formula: dynamic_max_position = max_position_value / mid_price
    /// where max_position_value = min(leverage_limit, volatility_limit)
    /// - leverage_limit = account_value × max_leverage
    /// - volatility_limit = (equity × risk_fraction) / (num_sigmas × σ × √T)
    ///
    /// This replaces the arbitrary static max_position with an equity/volatility-derived limit.
    /// Use this for all sizing calculations instead of config.max_position.
    pub dynamic_max_position: f64,

    /// Whether the dynamic limit is valid (has been calculated from margin state).
    /// If false, strategies should fall back to config.max_position.
    pub dynamic_limit_valid: bool,

    /// Margin-based quoting capacity (SIZE, not value).
    /// Formula: margin_available × leverage / microprice
    /// This is the HARD solvency constraint - what the account can actually support.
    /// Used for ladder allocation instead of user's arbitrary max_position.
    /// The Kelly optimizer should allocate up to this limit.
    pub margin_quoting_capacity: f64,

    // ==================== Stochastic Constraints (First Principles) ====================
    /// Asset tick size in basis points.
    /// Spread floor must be >= tick_size_bps (can't quote finer than tick).
    /// Set from asset metadata (e.g., BTC tick = 0.1 → 10 bps at $100k).
    pub tick_size_bps: f64,

    /// Latency-aware spread floor: δ_min = σ × √(2×τ_update)
    /// Dynamically computed from current volatility and expected quote latency.
    /// In fractional terms (multiply by 10000 for bps).
    pub latency_spread_floor: f64,

    /// Near-touch book depth (USD) within 5 bps of mid.
    /// Used for book depth constraint on tight quoting.
    pub near_touch_depth_usd: f64,

    /// Whether tight quoting is currently allowed based on all constraints.
    /// Combines: regime, toxicity, book depth, inventory, time-of-day.
    pub tight_quoting_allowed: bool,

    /// Reason tight quoting is blocked (if not allowed).
    /// None if tight quoting is allowed.
    pub tight_quoting_block_reason: Option<String>,

    /// Stochastic spread floor multiplier [1.0, 2.0+].
    /// Combines all constraints into a single spread widening factor.
    /// 1.0 = no widening (all constraints satisfied)
    /// > 1.0 = widen spreads proportionally
    pub stochastic_spread_multiplier: f64,

    // ==================== Entropy-Based Distribution ====================
    /// Whether to use entropy-based stochastic order distribution.
    /// This completely replaces the concentration fallback with diversity-preserving allocation.
    pub use_entropy_distribution: bool,

    /// Minimum entropy floor (bits). Distribution NEVER drops below this.
    /// H_min = 1.5 → at least exp(1.5) ≈ 4.5 effective levels always active.
    pub entropy_min_entropy: f64,

    /// Base temperature for softmax. Higher = more uniform distribution.
    pub entropy_base_temperature: f64,

    /// Minimum allocation floor per level (prevents zero allocations).
    pub entropy_min_allocation_floor: f64,

    /// Number of Thompson samples for stochastic allocation.
    pub entropy_thompson_samples: usize,

    // ==================== Adaptive Bayesian System ====================
    /// Whether the adaptive Bayesian system is enabled.
    pub use_adaptive_spreads: bool,

    /// Adaptive learned spread floor from Bayesian AS estimation.
    /// Replaces static min_spread_floor when enabled.
    /// Formula: floor = fees + max(0, μ_AS) + k × σ_AS
    pub adaptive_spread_floor: f64,

    /// Adaptive kappa from blended book/own-fill estimation.
    /// More accurate than book-only kappa especially in thin markets.
    pub adaptive_kappa: f64,

    /// Adaptive gamma from shrinkage estimation.
    /// Log-additive to prevent multiplicative explosion.
    pub adaptive_gamma: f64,

    /// Spread ceiling from fill rate controller.
    /// Ensures minimum fill rate to maintain market presence.
    pub adaptive_spread_ceiling: f64,

    /// Dynamic kappa floor from Bayesian confidence + credible intervals.
    /// Replaces hardcoded --kappa-floor CLI argument with principled model-driven bounds.
    /// Formula: blend(prior, ci_95_lower) based on confidence level.
    /// None during warmup or when CLI override is active.
    pub dynamic_kappa_floor: Option<f64>,

    /// Dynamic spread ceiling from fill rate controller + market spread p80.
    /// Replaces hardcoded --max-spread-bps CLI argument with model-driven bounds.
    /// Formula: max(fill_controller_ceiling, market_spread_p80)
    /// None during warmup or when CLI override is active.
    pub dynamic_spread_ceiling_bps: Option<f64>,

    /// Whether dynamic bounds are enabled (no CLI overrides).
    /// When true, use dynamic_kappa_floor and dynamic_spread_ceiling_bps.
    /// When false, static config values are in effect.
    pub use_dynamic_bounds: bool,

    /// Whether adaptive system is warmed up with enough data.
    /// NOTE: For deciding to USE adaptive values, check `adaptive_can_estimate` instead.
    /// This field indicates full calibration (20+ fills).
    pub adaptive_warmed_up: bool,

    /// Whether adaptive system can provide reasonable estimates.
    /// TRUE immediately - uses Bayesian priors that give sensible starting values.
    /// Use this to decide whether to use adaptive values (not adaptive_warmed_up).
    pub adaptive_can_estimate: bool,

    /// Warmup progress (0.0 to 1.0).
    /// Used for uncertainty scaling during warmup.
    pub adaptive_warmup_progress: f64,

    /// Uncertainty factor for warmup period.
    /// Multiply spreads by this factor (> 1.0 during warmup, 1.0 when warmed up).
    pub adaptive_uncertainty_factor: f64,

    // ==================== Calibration Fill Rate Controller ====================
    /// Gamma multiplier from calibration controller [0.3, 1.0].
    /// Lower values = tighter quotes to attract fills during warmup.
    /// 1.0 = calibration complete, normal GLFT behavior.
    pub calibration_gamma_mult: f64,

    /// Calibration progress [0.0, 1.0].
    /// Based on AS fills + kappa confidence.
    pub calibration_progress: f64,

    /// Whether calibration is complete.
    /// When true, calibration_gamma_mult = 1.0.
    pub calibration_complete: bool,

    // ==================== Model-Derived Sizing (GLFT First Principles) ====================
    /// Account value in USD (total equity).
    /// Used for GLFT-derived target liquidity: Q_soft = √(2×loss_budget/(γ×σ²))
    pub account_value: f64,

    /// Measured WebSocket latency in milliseconds (from ping/pong).
    /// Used for latency penalty: penalty = max(0.3, 1 - 0.5×√(τ/30ms))
    /// NOT hardcoded - comes from live WS measurements.
    pub measured_latency_ms: f64,

    /// Estimated fill rate (fills per second).
    /// Used for GLFT time horizon T = 1/fill_rate.
    /// Derived from observed fills, NOT assumed.
    pub estimated_fill_rate: f64,

    /// Model-derived target liquidity (SIZE, not value).
    /// Formula: min(Q_hard, Q_soft × latency_penalty).max(exchange_min)
    /// Where:
    /// - Q_hard = account_value × leverage × 0.5 / mark_price (margin solvency)
    /// - Q_soft = √(2 × acceptable_loss / (γ × σ²)) (GLFT risk budget)
    /// - latency_penalty = max(0.3, 1 - 0.5×√(τ/30ms)) (MEASURED τ)
    /// This replaces hardcoded target_liquidity when > 0.
    pub derived_target_liquidity: f64,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            sigma: 0.0001,                   // 0.01% per-second volatility (clean)
            sigma_total: 0.0001,             // Same initially
            sigma_effective: 0.0001,         // Same initially
            sigma_leverage_adjusted: 0.0001, // Same initially (no leverage effect)
            kappa: 100.0,                    // Moderate depth decay
            kappa_bid: 100.0,                // Same as kappa initially
            kappa_ask: 100.0,                // Same as kappa initially
            is_heavy_tailed: false,          // Assume exponential tails
            kappa_cv: 1.0,                   // CV=1 for exponential
            // V3: Robust Kappa Orchestrator
            kappa_robust: 2000.0, // Start at prior (will be updated from orchestrator)
            use_kappa_robust: true, // Default ON - use robust kappa for spreads
            kappa_outlier_count: 0, // No outliers detected yet
            // V2: Uncertainty Quantification
            kappa_uncertainty: 0.0,    // Will be computed from posterior
            kappa_95_lower: 100.0,     // Conservative lower bound
            kappa_95_upper: 100.0,     // Conservative upper bound
            toxicity_score: 0.0,       // No toxicity initially
            param_correlation: 0.0,    // No correlation initially
            as_factor: 1.0,            // No AS adjustment initially
            arrival_intensity: 0.5,    // 0.5 volume ticks per second
            is_toxic_regime: false,    // Default: not toxic
            jump_ratio: 1.0,           // Default: normal diffusion
            momentum_bps: 0.0,         // Default: no momentum
            flow_imbalance: 0.0,       // Default: balanced flow
            falling_knife_score: 0.0,  // Default: no falling knife
            rising_knife_score: 0.0,   // Default: no rising knife
            book_imbalance: 0.0,       // Default: balanced book
            liquidity_gamma_mult: 1.0, // Default: normal liquidity
            microprice: 0.0,           // Will be set from estimator
            market_mid: 0.0,           // Will be set from latest AllMids
            beta_book: 0.0,            // Will be learned from data
            beta_flow: 0.0,            // Will be learned from data
            // Tier 1: Adverse Selection
            as_spread_adjustment: 0.0, // No adjustment until warmed up
            predicted_alpha: 0.0,      // Default: no informed flow detected
            as_warmed_up: false,       // Starts not warmed up
            depth_decay_as: None,      // No calibrated model initially
            // Tier 1: Liquidation Cascade
            tail_risk_multiplier: 1.0, // Default: no tail risk scaling
            should_pull_quotes: false, // Default: don't pull quotes
            cascade_size_factor: 1.0,  // Default: full size
            // Tier 2: Hawkes Order Flow
            hawkes_buy_intensity: 0.0,
            hawkes_sell_intensity: 0.0,
            hawkes_imbalance: 0.0,
            hawkes_activity_percentile: 0.5,
            // Tier 2: Funding Rate
            funding_rate: 0.0,
            predicted_funding_cost: 0.0,
            // Tier 2: Spread Process
            fair_spread: 0.0,
            spread_percentile: 0.5,
            spread_regime: SpreadRegime::Normal,
            market_spread_bps: 0.0, // Will be computed from L2 book
            // Volatility Regime
            volatility_regime: VolatilityRegime::Normal,
            // First Principles Extensions
            lambda_jump: 0.0,
            mu_jump: 0.0,
            sigma_jump: 0.0001,
            kappa_vol: 0.5,
            theta_vol: 0.0001_f64.powi(2), // Default variance
            xi_vol: 0.1,
            rho_price_vol: -0.3,
            calibrated_volume_rate: 1.0,
            calibrated_cancel_rate: 0.2,
            premium: 0.0,
            premium_alpha: 0.0,
            bid_protection_factor: 1.0,
            ask_protection_factor: 1.0,
            p_momentum_continue: 0.5,
            // HJB Controller (stochastic integration)
            use_hjb_skew: false,         // Default OFF for safety
            hjb_optimal_skew: 0.0,       // Will be computed from HJB controller
            hjb_gamma_multiplier: 1.0,   // No multiplier by default
            hjb_inventory_target: 0.0,   // Zero inventory target
            hjb_is_terminal_zone: false, // Not in terminal zone
            // Drift-Adjusted Skew (first-principles momentum integration)
            use_drift_adjusted_skew: true, // ON by default for first-principles trading
            hjb_drift_urgency: 0.0,        // No urgency initially
            directional_variance_mult: 1.0, // No variance adjustment initially
            position_opposes_momentum: false, // No opposition initially
            urgency_score: 0.0,            // No urgency initially
            // Kalman Filter (stochastic integration)
            use_kalman_filter: false,    // Default OFF for safety
            kalman_fair_price: 0.0,      // Will be computed from Kalman filter
            kalman_uncertainty: 0.0,     // Will be computed from Kalman filter
            kalman_spread_widening: 0.0, // Will be computed from Kalman filter
            kalman_warmed_up: false,     // Not warmed up initially
            // Constrained Optimizer (stochastic integration)
            use_constrained_optimizer: true, // Enable for entropy-based allocation
            margin_available: 0.0,           // Will be fetched from margin sizer
            leverage: 1.0,                   // Default 1x leverage
            // Kelly-Stochastic Allocation (stochastic integration)
            use_kelly_stochastic: false, // Default OFF for safety
            kelly_alpha_touch: 0.15,     // 15% informed at touch
            kelly_alpha_decay_bps: 10.0, // Alpha decays with 10bp characteristic
            kelly_fraction: 0.25,        // Quarter Kelly (conservative)
            kelly_time_horizon: 60.0,    // Default 60s (will be computed from config method)
            // Exchange Position Limits
            exchange_limits_valid: false,           // Not fetched yet
            exchange_effective_bid_limit: f64::MAX, // No constraint until fetched
            exchange_effective_ask_limit: f64::MAX, // No constraint until fetched
            exchange_limits_age_ms: u64::MAX,       // Never fetched
            // Pending Exposure
            pending_bid_exposure: 0.0, // No resting orders initially
            pending_ask_exposure: 0.0, // No resting orders initially
            // Dynamic Position Limits
            dynamic_max_position: 0.0,    // Will be set from kill switch
            dynamic_limit_valid: false,   // Not valid until margin state refreshed
            margin_quoting_capacity: 0.0, // Will be computed from margin_available
            // Stochastic Constraints
            tick_size_bps: 10.0,          // Default 10 bps tick
            latency_spread_floor: 0.0003, // 3 bps default floor
            near_touch_depth_usd: 0.0,    // No depth data initially
            tight_quoting_allowed: false, // Conservative default
            tight_quoting_block_reason: Some("Warmup".to_string()),
            stochastic_spread_multiplier: 1.0, // No widening initially
            // Entropy-Based Distribution (FIRST PRINCIPLES)
            use_entropy_distribution: true, // Entropy-based allocation (replaces concentration fallback)
            entropy_min_entropy: 1.5,       // At least ~4.5 effective levels
            entropy_base_temperature: 1.0,  // Standard softmax
            entropy_min_allocation_floor: 0.02, // 2% minimum per level
            entropy_thompson_samples: 5,    // Moderate stochasticity
            // Adaptive Bayesian System
            use_adaptive_spreads: false,       // Default OFF for safety
            adaptive_spread_floor: 0.0008,     // 8 bps fallback floor
            adaptive_kappa: 100.0,             // Moderate kappa fallback
            adaptive_gamma: 0.3,               // Base gamma fallback
            adaptive_spread_ceiling: f64::MAX, // No ceiling by default
            // Dynamic bounds (model-driven, replaces hardcoded CLI values)
            dynamic_kappa_floor: None, // Computed at runtime from Bayesian CI
            dynamic_spread_ceiling_bps: None, // Computed at runtime from fill rate + market p80
            use_dynamic_bounds: true,  // Default ON - use model-driven bounds
            adaptive_warmed_up: false, // Not warmed up initially
            adaptive_can_estimate: true, // Can estimate immediately via priors
            adaptive_warmup_progress: 0.0, // Start at 0% progress
            adaptive_uncertainty_factor: 1.2, // Start with 20% wider spreads
            // Calibration Fill Rate Controller
            calibration_gamma_mult: 0.3, // Start fill-hungry (70% gamma reduction)
            calibration_progress: 0.0,   // Start at 0% calibration
            calibration_complete: false, // Not calibrated yet
            // Model-Derived Sizing (GLFT First Principles)
            account_value: 0.0,              // Will be set from margin state
            measured_latency_ms: 50.0,       // Default 50ms until measured
            estimated_fill_rate: 0.001,      // Conservative 0.001/sec until observed
            derived_target_liquidity: 0.0,   // Will be computed from GLFT formula
        }
    }
}

impl MarketParams {
    /// Get net pending exposure (positive = net long exposure from resting orders).
    pub fn net_pending_exposure(&self) -> f64 {
        self.pending_bid_exposure - self.pending_ask_exposure
    }

    /// Get effective max_position for sizing calculations (QUOTING CAPACITY).
    ///
    /// Priority order for quoting capacity:
    /// 1. dynamic_max_position (volatility-adjusted from kill switch)
    /// 2. margin_quoting_capacity (pure margin-based: margin × leverage / price)
    /// 3. static_fallback (last resort during early warmup)
    ///
    /// IMPORTANT: This is for QUOTING allocation, not reduce-only triggers.
    /// The Kelly optimizer should be able to allocate across full margin capacity.
    /// User's --max-position config is used ONLY for reduce-only filter.
    ///
    /// Priority order (MARGIN is the HARD solvency constraint):
    /// 1. margin_quoting_capacity (margin × leverage / price) - can't exceed available margin
    /// 2. dynamic_max_position (value-based from kill switch) - risk management layer
    /// 3. static_fallback (user's --max-position) - reduce-only trigger only
    pub fn effective_max_position(&self, static_fallback: f64) -> f64 {
        const EPSILON: f64 = 1e-9;

        // PRIORITY 1: Margin-based quoting capacity (HARD solvency constraint)
        // This is the true limit from available margin - we can't exceed this
        if self.margin_quoting_capacity > EPSILON {
            // If dynamic_max_position is also valid, take the minimum (both are valid constraints)
            if self.dynamic_limit_valid && self.dynamic_max_position > EPSILON {
                return self.margin_quoting_capacity.min(self.dynamic_max_position);
            }
            return self.margin_quoting_capacity;
        }

        // PRIORITY 2: Dynamic volatility-adjusted limit from kill switch
        // Used when margin data isn't available yet
        if self.dynamic_limit_valid && self.dynamic_max_position > EPSILON {
            return self.dynamic_max_position;
        }

        // Last resort: static fallback (only during early warmup)
        static_fallback
    }

    /// Get quoting capacity (margin-based, for ladder allocation).
    ///
    /// Returns the margin-based capacity for quoting, ignoring user's max_position.
    /// This is the HARD solvency constraint from available margin.
    pub fn quoting_capacity(&self) -> f64 {
        const EPSILON: f64 = 1e-9;

        // Prefer margin-based capacity when available
        let result = if self.margin_quoting_capacity > EPSILON {
            self.margin_quoting_capacity
        } else if self.dynamic_limit_valid && self.dynamic_max_position > EPSILON {
            self.dynamic_max_position
        } else {
            // Compute on-the-fly if fields not populated
            if self.microprice > EPSILON && self.leverage > 0.0 {
                (self.margin_available * self.leverage / self.microprice).max(0.0)
            } else {
                0.0
            }
        };

        // Debug: trace which branch was taken
        tracing::trace!(
            margin_quoting_capacity = %format!("{:.6}", self.margin_quoting_capacity),
            dynamic_limit_valid = self.dynamic_limit_valid,
            dynamic_max_position = %format!("{:.6}", self.dynamic_max_position),
            margin_available = %format!("{:.2}", self.margin_available),
            leverage = %format!("{:.1}", self.leverage),
            result = %format!("{:.6}", result),
            "quoting_capacity() returning"
        );

        result
    }
}

impl MarketParams {
    /// Extract volatility parameters as a focused struct.
    pub fn volatility(&self) -> params::VolatilityParams {
        params::VolatilityParams {
            sigma: self.sigma,
            sigma_total: self.sigma_total,
            sigma_effective: self.sigma_effective,
            regime: self.volatility_regime,
            kappa_vol: self.kappa_vol,
            theta_vol: self.theta_vol,
            xi_vol: self.xi_vol,
            rho_price_vol: self.rho_price_vol,
        }
    }

    /// Extract liquidity parameters as a focused struct.
    pub fn liquidity(&self) -> params::LiquidityParams {
        params::LiquidityParams {
            kappa: self.kappa,
            kappa_bid: self.kappa_bid,
            kappa_ask: self.kappa_ask,
            is_heavy_tailed: self.is_heavy_tailed,
            kappa_cv: self.kappa_cv,
            arrival_intensity: self.arrival_intensity,
            liquidity_gamma_mult: self.liquidity_gamma_mult,
            calibrated_volume_rate: self.calibrated_volume_rate,
            calibrated_cancel_rate: self.calibrated_cancel_rate,
        }
    }

    /// Extract flow parameters as a focused struct.
    pub fn flow(&self) -> params::FlowParams {
        params::FlowParams {
            flow_imbalance: self.flow_imbalance,
            book_imbalance: self.book_imbalance,
            momentum_bps: self.momentum_bps,
            falling_knife_score: self.falling_knife_score,
            rising_knife_score: self.rising_knife_score,
            hawkes_buy_intensity: self.hawkes_buy_intensity,
            hawkes_sell_intensity: self.hawkes_sell_intensity,
            hawkes_imbalance: self.hawkes_imbalance,
            hawkes_activity_percentile: self.hawkes_activity_percentile,
            bid_protection_factor: self.bid_protection_factor,
            ask_protection_factor: self.ask_protection_factor,
            p_momentum_continue: self.p_momentum_continue,
        }
    }

    /// Extract regime parameters as a focused struct.
    pub fn regime(&self) -> params::RegimeParams {
        params::RegimeParams {
            is_toxic_regime: self.is_toxic_regime,
            jump_ratio: self.jump_ratio,
            spread_regime: self.spread_regime,
            spread_percentile: self.spread_percentile,
            fair_spread: self.fair_spread,
            lambda_jump: self.lambda_jump,
            mu_jump: self.mu_jump,
            sigma_jump: self.sigma_jump,
        }
    }

    /// Extract fair price parameters as a focused struct.
    pub fn fair_price(&self) -> params::FairPriceParams {
        params::FairPriceParams {
            microprice: self.microprice,
            beta_book: self.beta_book,
            beta_flow: self.beta_flow,
            use_kalman_filter: self.use_kalman_filter,
            kalman_fair_price: self.kalman_fair_price,
            kalman_uncertainty: self.kalman_uncertainty,
            kalman_spread_widening: self.kalman_spread_widening,
            kalman_warmed_up: self.kalman_warmed_up,
        }
    }

    /// Extract adverse selection parameters as a focused struct.
    pub fn adverse_selection(&self) -> params::AdverseSelectionParams {
        params::AdverseSelectionParams {
            as_spread_adjustment: self.as_spread_adjustment,
            predicted_alpha: self.predicted_alpha,
            as_warmed_up: self.as_warmed_up,
            depth_decay_as: self.depth_decay_as.clone(),
        }
    }

    /// Extract cascade parameters as a focused struct.
    pub fn cascade(&self) -> params::CascadeParams {
        params::CascadeParams {
            tail_risk_multiplier: self.tail_risk_multiplier,
            should_pull_quotes: self.should_pull_quotes,
            cascade_size_factor: self.cascade_size_factor,
        }
    }

    /// Extract funding parameters as a focused struct.
    pub fn funding(&self) -> params::FundingParams {
        params::FundingParams {
            funding_rate: self.funding_rate,
            predicted_funding_cost: self.predicted_funding_cost,
            premium: self.premium,
            premium_alpha: self.premium_alpha,
        }
    }

    /// Extract HJB controller parameters as a focused struct.
    pub fn hjb(&self) -> params::HJBParams {
        params::HJBParams {
            use_hjb_skew: self.use_hjb_skew,
            hjb_optimal_skew: self.hjb_optimal_skew,
            hjb_gamma_multiplier: self.hjb_gamma_multiplier,
            hjb_inventory_target: self.hjb_inventory_target,
            hjb_is_terminal_zone: self.hjb_is_terminal_zone,
        }
    }

    /// Extract margin constraint parameters as a focused struct.
    pub fn margin_constraints(&self) -> params::MarginConstraintParams {
        params::MarginConstraintParams {
            use_constrained_optimizer: self.use_constrained_optimizer,
            margin_available: self.margin_available,
            leverage: self.leverage,
        }
    }

    /// Extract Kelly-Stochastic parameters as a focused struct.
    pub fn kelly_stochastic(&self) -> params::KellyStochasticConfigParams {
        params::KellyStochasticConfigParams {
            use_kelly_stochastic: self.use_kelly_stochastic,
            alpha_touch: self.kelly_alpha_touch,
            alpha_decay_bps: self.kelly_alpha_decay_bps,
            kelly_fraction: self.kelly_fraction,
            time_horizon: self.kelly_time_horizon,
        }
    }

    /// Extract exchange position limit parameters as a focused struct.
    pub fn exchange_limits(&self) -> params::ExchangeLimitsParams {
        params::ExchangeLimitsParams {
            valid: self.exchange_limits_valid,
            effective_bid_limit: self.exchange_effective_bid_limit,
            effective_ask_limit: self.exchange_effective_ask_limit,
            age_ms: self.exchange_limits_age_ms,
        }
    }

    /// Extract stochastic constraint parameters as a focused struct.
    pub fn stochastic_constraints(&self) -> params::StochasticConstraintParams {
        params::StochasticConstraintParams {
            tick_size_bps: self.tick_size_bps,
            latency_spread_floor: self.latency_spread_floor,
            near_touch_depth_usd: self.near_touch_depth_usd,
            tight_quoting_allowed: self.tight_quoting_allowed,
            stochastic_spread_multiplier: self.stochastic_spread_multiplier,
        }
    }

    /// Extract entropy distribution parameters as a focused struct.
    pub fn entropy_distribution(&self) -> params::EntropyDistributionParams {
        params::EntropyDistributionParams {
            use_entropy_distribution: self.use_entropy_distribution,
            min_entropy: self.entropy_min_entropy,
            base_temperature: self.entropy_base_temperature,
            min_allocation_floor: self.entropy_min_allocation_floor,
            thompson_samples: self.entropy_thompson_samples,
            // Derive toxicity from jump_ratio for temperature scaling
            toxicity: self.jump_ratio,
            volatility_ratio: self.sigma_effective / 0.0001, // Normalized to baseline
            cascade_severity: if self.should_pull_quotes {
                1.0
            } else {
                (self.tail_risk_multiplier - 1.0) / 4.0
            },
        }
    }

    /// Compute stochastic constraints and update fields.
    ///
    /// This evaluates all first-principles constraints and updates:
    /// - `latency_spread_floor`: σ × √(2×τ_update)
    /// - `tight_quoting_allowed`: Combined constraint check
    /// - `tight_quoting_block_reason`: Why blocked (if any)
    /// - `stochastic_spread_multiplier`: Combined widening factor
    ///
    /// # Arguments
    /// - `config`: Stochastic configuration with constraint parameters
    /// - `position`: Current position (signed)
    /// - `max_position`: Maximum allowed position
    /// - `current_hour_utc`: Current hour in UTC (0-23)
    pub fn compute_stochastic_constraints(
        &mut self,
        config: &crate::market_maker::StochasticConfig,
        position: f64,
        max_position: f64,
        current_hour_utc: u8,
    ) {
        // === 1. Latency-Aware Spread Floor ===
        // δ_min = σ × √(2×τ) where τ is MEASURED round-trip latency in seconds
        // Use measured_latency_ms (from WS ping) if available, fall back to config
        if config.use_latency_spread_floor {
            // Prefer MEASURED latency over hardcoded config value
            let tau_sec = if self.measured_latency_ms > 0.0 {
                self.measured_latency_ms / 1000.0
            } else {
                config.quote_update_latency_ms / 1000.0
            };
            self.latency_spread_floor = self.sigma * (2.0 * tau_sec).sqrt();
        }

        // === 2. Evaluate Tight Quoting Conditions ===
        let mut can_quote_tight = true;
        let mut block_reason: Option<String> = None;

        if config.use_conditional_tight_quoting {
            // Condition 1: Volatility regime must be Low or Normal
            if self.volatility_regime != VolatilityRegime::Low
                && self.volatility_regime != VolatilityRegime::Normal
            {
                can_quote_tight = false;
                block_reason = Some(format!("Vol regime {:?}", self.volatility_regime));
            }

            // Condition 2: Toxicity (predicted alpha) must be low
            if can_quote_tight && self.predicted_alpha > config.tight_quoting_max_toxicity {
                can_quote_tight = false;
                block_reason = Some(format!(
                    "Toxicity {:.1}% > {:.1}%",
                    self.predicted_alpha * 100.0,
                    config.tight_quoting_max_toxicity * 100.0
                ));
            }

            // Condition 3: Inventory utilization must be low
            let inventory_util = if max_position > 0.0 {
                (position / max_position).abs()
            } else {
                0.0
            };
            if can_quote_tight && inventory_util > config.tight_quoting_max_inventory {
                can_quote_tight = false;
                block_reason = Some(format!(
                    "Inventory {:.0}% > {:.0}%",
                    inventory_util * 100.0,
                    config.tight_quoting_max_inventory * 100.0
                ));
            }

            // Condition 4: Not in excluded hours
            if can_quote_tight
                && config
                    .tight_quoting_excluded_hours
                    .contains(&current_hour_utc)
            {
                can_quote_tight = false;
                block_reason = Some(format!("Excluded hour {} UTC", current_hour_utc));
            }

            // Condition 5: Book depth must be sufficient (if enabled)
            if config.use_book_depth_constraint
                && can_quote_tight
                && self.near_touch_depth_usd < config.min_book_depth_usd
            {
                can_quote_tight = false;
                block_reason = Some(format!(
                    "Thin book ${:.0}k < ${:.0}k",
                    self.near_touch_depth_usd / 1000.0,
                    config.min_book_depth_usd / 1000.0
                ));
            }
        }

        self.tight_quoting_allowed = can_quote_tight;
        self.tight_quoting_block_reason = block_reason;

        // === 3. Stochastic Spread Multiplier - DEPRECATED ===
        // FIRST PRINCIPLES REFACTOR: Arbitrary spread multipliers bypass the GLFT model.
        // All risk factors now flow through gamma (risk aversion) scaling:
        //   - Book depth → gamma scaling via RiskConfig.book_depth_multiplier()
        //   - Toxicity → gamma scaling via RiskConfig.toxicity_sensitivity
        //   - Warmup uncertainty → gamma scaling via RiskConfig.warmup_multiplier()
        //
        // This ensures spreads are computed through the principled formula:
        //   δ = (1/γ) × ln(1 + γ/κ)
        //
        // Setting to 1.0 (no-op) - the field is kept for API compatibility.
        self.stochastic_spread_multiplier = 1.0;
    }

    /// Get the effective minimum spread floor in fractional terms.
    ///
    /// Returns the maximum of:
    /// - Static min_spread_floor from RiskConfig
    /// - Tick size (can't quote finer than tick)
    /// - Latency-based floor (σ × √(2×τ_update))
    pub fn effective_spread_floor(&self, risk_config_floor: f64) -> f64 {
        let tick_floor = self.tick_size_bps / 10_000.0; // Convert bps to fraction
        risk_config_floor
            .max(tick_floor)
            .max(self.latency_spread_floor)
    }

    /// Compute model-derived target liquidity from GLFT first principles.
    ///
    /// ALL inputs are from MEASURED data or exchange metadata - NO assumptions:
    /// - `gamma`: Risk aversion (user preference)
    /// - `sigma`: Measured volatility (from bipower variation)
    /// - `measured_latency_ms`: Measured WS ping latency
    /// - `estimated_fill_rate`: Observed fills per second
    /// - `account_value`: From margin state
    /// - `leverage`: From exchange metadata
    /// - `microprice`: Current fair price
    /// - `num_levels`: Ladder configuration
    /// - `min_notional`: Exchange minimum ($10)
    ///
    /// Formula:
    /// - Q_hard = account_value × leverage × 0.5 / mark_price (margin solvency)
    /// - Q_soft = √(2 × acceptable_loss / (γ × σ²)) (GLFT risk budget, 2% of account)
    /// - latency_penalty = max(0.3, 1 - 0.5 × √(τ_measured / 30ms))
    /// - derived = min(Q_hard, Q_soft × latency_penalty).max(exchange_min)
    pub fn compute_derived_target_liquidity(
        &mut self,
        gamma: f64,
        num_levels: usize,
        min_notional: f64,
    ) {
        // Sanity check inputs
        if self.account_value <= 0.0 || self.leverage <= 0.0 || self.microprice <= 0.0 {
            self.derived_target_liquidity = 0.0;
            return;
        }

        // === Hard Constraint: Margin Solvency ===
        // Can only quote what the account can actually support
        // Use 50% safety factor to leave buffer for adverse price moves
        let q_hard = (self.account_value * self.leverage * 0.5) / self.microprice;

        // === Soft Constraint: GLFT Risk Budget ===
        // Acceptable loss = 2% of account value (conservative)
        // Q_soft = √(2 × acceptable_loss / (γ × σ²)) / 2
        // The /2 at the end is because we size each side independently
        let acceptable_loss = self.account_value * 0.02;
        let sigma_sq = self.sigma.powi(2).max(1e-12); // Floor to prevent div by zero
        let gamma_eff = gamma.max(0.01); // Floor gamma to prevent explosion

        // Fill rate adjustment: higher fill rate allows larger inventory bounds
        // T = 1/fill_rate, so Q_soft scales with sqrt(fill_rate)
        let fill_rate_factor = self.estimated_fill_rate.sqrt().max(0.01);

        let q_soft = ((2.0 * acceptable_loss) / (gamma_eff * sigma_sq)).sqrt()
            * fill_rate_factor / 2.0;

        // === Latency Penalty: Reduce exposure when slow ===
        // At high latency, price moves σ√τ during round-trip
        // This erodes edge, so size down proportionally
        // penalty = max(0.3, 1 - 0.5 × √(τ_measured / 30ms))
        let tau_ms = self.measured_latency_ms.max(10.0); // Floor at 10ms
        let tau_baseline_ms = 30.0; // 30ms baseline (low latency)
        let latency_penalty = (1.0 - 0.5 * (tau_ms / tau_baseline_ms).sqrt()).max(0.3);

        let q_soft_adjusted = q_soft * latency_penalty;

        // === Per-Level Constraint ===
        // Don't size so large that individual levels are too big
        let num_levels_f = (num_levels as f64).max(1.0);
        let per_level_cap = q_hard / num_levels_f;

        // === Exchange Minimum ===
        // Must be at least min_notional in USD, with 50% buffer
        let min_viable = (min_notional * 1.5) / self.microprice;

        // === Final Derived Liquidity ===
        // Take minimum of all constraints, but ensure at least exchange minimum
        self.derived_target_liquidity = q_hard
            .min(q_soft_adjusted)
            .min(per_level_cap)
            .max(min_viable);

        tracing::debug!(
            q_hard = %format!("{:.6}", q_hard),
            q_soft = %format!("{:.6}", q_soft),
            q_soft_adjusted = %format!("{:.6}", q_soft_adjusted),
            latency_penalty = %format!("{:.2}", latency_penalty),
            per_level_cap = %format!("{:.6}", per_level_cap),
            min_viable = %format!("{:.6}", min_viable),
            derived = %format!("{:.6}", self.derived_target_liquidity),
            account_value = %format!("{:.2}", self.account_value),
            gamma = %format!("{:.3}", gamma_eff),
            sigma = %format!("{:.6}", self.sigma),
            latency_ms = %format!("{:.1}", tau_ms),
            fill_rate = %format!("{:.4}", self.estimated_fill_rate),
            "GLFT-derived target liquidity (all inputs from measured data)"
        );
    }
}
