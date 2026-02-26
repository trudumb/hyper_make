//! Market parameters for quoting strategies.

use crate::market_maker::adverse_selection::DepthDecayAS;
use crate::market_maker::config::auto_derive::CapitalTier;
use crate::market_maker::config::{CapacityBudget, CapitalAwarePolicy};
use crate::market_maker::estimator::{MarketEstimator, VolatilityRegime};
use crate::market_maker::process_models::SpreadRegime;

use super::params;
use super::regime_state::ControllerObjective;

/// Additive spread composition replacing multiplicative spread factor chains.
///
/// Total half-spread = sum of all components, floored at fee_bps.
/// Each component is individually bounded to prevent any single factor from dominating.
///
/// This replaces the old pattern of `base × bandit × widening × quota` which caused
/// death spirals (e.g., 3.34x quota × 2x bandit = 6.7x blowup).
#[derive(Debug, Clone)]
pub struct SpreadComposition {
    /// Core GLFT half-spread: (1/gamma) * ln(1 + gamma/kappa) + vol_comp + fee
    /// This is the market-microstructure-optimal spread before any risk adjustments.
    /// Units: basis points.
    pub glft_half_spread_bps: f64,
    /// Additive risk premium from regime detection, Hawkes excitation, toxicity, and staleness.
    /// Built from individual additive components, NOT a multiplicative factor.
    /// Units: basis points.
    pub risk_premium_bps: f64,
    /// Quota shadow spread: pressure from API rate limit constraints.
    /// When quota headroom is low, widens spreads to reduce cancel/replace frequency.
    /// Units: basis points. Capped at 50 bps.
    pub quota_addon_bps: f64,
    /// Warmup uncertainty premium: extra spread during early session when parameters
    /// are estimated from priors rather than data.
    /// Units: basis points. Decays to zero as warmup completes.
    pub warmup_addon_bps: f64,
    /// Maker fee on Hyperliquid (always 1.5 bps).
    /// Included in GLFT computation already, tracked here for decomposition visibility.
    /// Units: basis points.
    pub fee_bps: f64,
}

impl Default for SpreadComposition {
    fn default() -> Self {
        Self {
            glft_half_spread_bps: 0.0,
            risk_premium_bps: 0.0,
            quota_addon_bps: 0.0,
            warmup_addon_bps: 0.0,
            fee_bps: 1.5,
        }
    }
}

impl SpreadComposition {
    /// Total half-spread in basis points, floored at fee_bps.
    ///
    /// The floor ensures we never quote tighter than the maker fee.
    pub fn total_half_spread_bps(&self) -> f64 {
        (self.glft_half_spread_bps
            + self.risk_premium_bps
            + self.quota_addon_bps
            + self.warmup_addon_bps)
            .max(self.fee_bps)
    }

    /// Total Bid half-spread in basis points including asymmetric addons.
    pub fn bid_half_spread_bps(&self) -> f64 {
        self.total_half_spread_bps().max(self.fee_bps)
    }

    /// Total Ask half-spread in basis points including asymmetric addons.
    pub fn ask_half_spread_bps(&self) -> f64 {
        self.total_half_spread_bps().max(self.fee_bps)
    }

    /// Diagnostic breakdown string for logging.
    pub fn breakdown_string(&self) -> String {
        format!(
            "glft={:.1} risk={:.1} quota={:.1} warmup={:.1} fee={:.1} total={:.1}",
            self.glft_half_spread_bps,
            self.risk_premium_bps,
            self.quota_addon_bps,
            self.warmup_addon_bps,
            self.fee_bps,
            self.total_half_spread_bps()
        )
    }
}

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

    // === WS2: Dual-Timescale Inventory Control ===
    /// EWMA of reducing-fill holding durations (seconds).
    /// Measured quantity: how long inventory is held before a reducing fill.
    /// Default: 60s. Updated by FillProcessor on each reducing fill.
    pub tau_inventory_s: f64,

    /// Variance of τ_inventory (seconds²).
    /// Used by PPIP timing_uncertainty term.
    pub tau_variance_s2: f64,

    /// BMA posterior variance of σ². Feeds PPIP ambiguity aversion.
    /// 0.0 when BMA not yet warmed up.
    pub sigma_sq_variance_bma: f64,

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

    /// Kappa credible interval width (normalized by mean).
    /// ci_width = (ci_95_upper - ci_95_lower) / kappa_mean
    /// Used for uncertainty-based warmup gamma scaling.
    /// Range: ~0.3 (converged) to ~1.0+ (high uncertainty)
    pub kappa_ci_width: f64,

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

    /// Lead-lag signal from cross-exchange analysis (in bps).
    /// Binance → Hyperliquid: when Binance moves, Hyperliquid follows.
    /// Positive = predicted upward move, Negative = predicted downward move.
    /// Confidence-weighted: signal * (mutual_info / 0.1).min(1.0)
    /// NOTE: This is the CAPPED skew signal. For reservation mid drift, use `drift_signal_bps`.
    pub lead_lag_signal_bps: f64,

    /// Uncapped drift signal for reservation mid (bps, positive = bullish).
    /// Separated from skew: drift shifts reservation mid, skew shifts bid/ask asymmetry.
    /// Only bounded by ±95% of GLFT half-spread clamp in the reservation mid calculation.
    pub drift_signal_bps: f64,

    /// Lead-lag signal confidence [0, 1].
    /// Based on mutual information between signal and target.
    /// > 0.02 bits is considered informative.
    pub lead_lag_confidence: f64,

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
    /// Per-side AS spread adjustment for bids (from buy-side realized AS).
    pub as_spread_adjustment_bid: f64,
    /// Per-side AS spread adjustment for asks (from sell-side realized AS).
    pub as_spread_adjustment_ask: f64,

    /// Predicted alpha: P(next trade is informed) in [0, 1].
    /// Use for diagnostics and future alpha-aware sizing.
    pub predicted_alpha: f64,

    /// Is AS estimator warmed up with enough fills?
    pub as_warmed_up: bool,

    /// Depth-dependent AS model (calibrated from fills).
    /// First-principles: AS(δ) = AS₀ × exp(-δ/δ_char)
    /// Used by ladder for depth-aware spread capture.
    pub depth_decay_as: Option<DepthDecayAS>,

    /// Conditional adverse selection: E[AS | fill] posterior mean (bps).
    /// Learned from Bayesian update on realized fill AS, NOT a magic number.
    /// This captures that fills cluster around toxic moments, so
    /// E[AS | fill] ≠ E[AS] unconditional. None during warmup.
    pub conditional_as_posterior_mean_bps: Option<f64>,

    // === Tier 1: Pre-Fill AS Classifier (Phase 3) ===
    /// Pre-fill toxicity score for bid side [0, 1].
    /// Predicts how toxic a bid fill would be BEFORE it happens.
    /// 0 = safe, 1 = highly toxic. Used for proactive spread widening.
    pub pre_fill_toxicity_bid: f64,

    /// Pre-fill toxicity score for ask side [0, 1].
    /// Predicts how toxic an ask fill would be BEFORE it happens.
    pub pre_fill_toxicity_ask: f64,

    // === Tier 1: Liquidation Cascade ===
    /// Tail risk intensity [0, 1].
    /// 0 = no tail risk, 1 = extreme tail risk (liquidation cascades).
    /// Distinct from cascade_intensity: captures depth-of-crisis severity.
    pub tail_risk_intensity: f64,

    /// Should pull all quotes due to extreme cascade?
    pub should_pull_quotes: bool,

    /// Cascade intensity [0, 1] for graceful degradation.
    /// 0.0 = calm market, 1.0 = full cascade severity.
    /// Fed into CalibratedRiskModel beta_cascade.
    pub cascade_intensity: f64,

    // === Tier 2: Hawkes Order Flow ===
    /// Hawkes buy intensity (λ_buy) - self-exciting arrival rate
    pub hawkes_buy_intensity: f64,

    /// Hawkes sell intensity (λ_sell) - self-exciting arrival rate
    pub hawkes_sell_intensity: f64,

    /// Hawkes flow imbalance [-1, 1] - normalized buy/sell intensity difference
    pub hawkes_imbalance: f64,

    /// Hawkes activity percentile [0, 1] - where current intensity sits in history
    pub hawkes_activity_percentile: f64,

    // --- Hawkes Excitation Prediction (Phase 7: Bayesian Fusion) ---
    /// Probability of cluster/cascade in next τ seconds.
    /// High values indicate elevated risk of self-exciting price cascades.
    /// Source: HawkesExcitationPredictor.p_cluster(τ)
    pub hawkes_p_cluster: f64,

    /// Edge penalty multiplier [0.5, 1.0] for quoting decisions.
    /// Lower values = more conservative quoting during high excitation.
    /// Formula: 1.0 - (1.0 - min_penalty) × √p_cluster
    pub hawkes_excitation_penalty: f64,

    /// Whether we're in a high excitation state.
    /// True when: branching_ratio > 0.7 OR intensity_percentile > 0.8
    pub hawkes_is_high_excitation: bool,

    /// Current branching ratio n = α/β from GMM calibration.
    /// Critical threshold: n → 1 indicates near-critical regime.
    /// Range: [0, 1) for stationary processes.
    pub hawkes_branching_ratio: f64,

    /// Spread widening factor for defensive quoting [1.0, 3.0].
    /// Multiplied with GLFT optimal spread during high excitation.
    /// Formula: 1 + 0.3×(intensity_ratio - 1) + 0.5×n²
    pub hawkes_spread_widening: f64,

    /// Expected time to next cluster event (seconds).
    /// Lower values indicate imminent cluster risk.
    pub hawkes_expected_cluster_time: f64,

    /// Excess intensity ratio: λ_current / λ_baseline.
    /// Values > 1 indicate elevated activity above normal.
    pub hawkes_excess_intensity_ratio: f64,

    // --- Phase 8: RL Policy Recommendations ---
    /// RL recommended spread delta (bps). Positive = widen.
    pub rl_spread_delta_bps: f64,
    /// RL recommended bid skew (bps).
    pub rl_bid_skew_bps: f64,
    /// RL recommended ask skew (bps).
    pub rl_ask_skew_bps: f64,
    /// RL policy confidence [0, 1].
    pub rl_confidence: f64,
    /// Whether RL is exploring (vs exploiting).
    pub rl_is_exploration: bool,
    /// Expected Q-value from RL agent.
    pub rl_expected_q: f64,
    /// Whether RL action was applied to quoting (vs observation-only).
    pub rl_action_applied: bool,

    // --- Contextual Bandit SpreadOptimizer ---
    /// Bandit-selected spread additive adjustment (bps).
    /// Applied after GLFT base spread: final_spread = base + bandit_additive_bps.
    pub bandit_spread_additive_bps: f64,
    /// Whether the bandit selection was exploration (Thompson) vs exploitation (greedy).
    pub bandit_is_exploration: bool,

    // --- Phase 8: Competitor Model ---
    /// Competitor snipe probability [0, 1].
    pub competitor_snipe_prob: f64,
    /// Competitor-driven spread widening factor.
    pub competitor_spread_factor: f64,
    /// Estimated number of active competitor MMs.
    pub competitor_count: f64,
    /// Estimated market share [0, 1] (own volume / total volume).
    /// Used for monopolist LP pricing: higher market share → more pricing power.
    pub market_share: f64,

    // === Phase 9: Rate Limit Death Spiral Prevention ===
    /// Rate limit headroom as fraction [0, 1].
    /// 1.0 = full budget available, 0.0 = exhausted.
    /// Used for quota-aware ladder density and shadow pricing.
    pub rate_limit_headroom_pct: f64,

    /// Continuous shadow spread from quota pressure (bps).
    /// Additive to GLFT optimal spread. Smoothly increases as headroom drops.
    /// Formula: lambda_shadow / headroom.max(0.01), capped at max_shadow_spread_bps.
    pub quota_shadow_spread_bps: f64,

    // === Tier 2: Funding Rate ===
    /// Current funding rate (annualized)
    pub funding_rate: f64,

    /// Predicted funding cost for holding period
    pub predicted_funding_cost: f64,

    /// Time to next funding settlement in seconds.
    /// Used by prediction logging for conditional calibration analysis.
    pub time_to_funding_settlement_s: f64,

    /// Current open interest (contracts or USD depending on exchange).
    /// Used for concentration risk checks and cascade detection.
    pub open_interest: f64,

    /// Open interest change over last 1 minute (fraction).
    /// Negative values signal potential liquidation cascades.
    pub oi_change_1m: f64,

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
    /// Probability momentum continues
    pub p_momentum_continue: f64,

    // === Stochastic Module Integration (HJB Controller) ===
    /// Whether to use HJB optimal skew instead of heuristic inventory_skew_with_flow
    pub use_hjb_skew: bool,
    /// HJB optimal inventory skew (from Avellaneda-Stoikov HJB solution)
    /// Formula: γσ²qT + terminal_penalty × q × urgency + funding_bias
    pub hjb_optimal_skew: f64,
    /// HJB optimal inventory target (optimal q* for current session state)
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
    /// Available margin for order placement (USD)
    pub margin_available: f64,
    /// Current leverage ratio
    pub leverage: f64,

    // === Stochastic Module Integration (Kelly-Stochastic Allocation) ===
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

    // === Exposure Budget (Worst-Case Aggregate) ===
    /// Available capacity for new buy orders, accounting for ALL resting + in-flight bids.
    /// available_bid_budget = max_position - (position + resting_bids + inflight_bids)
    pub available_bid_budget: f64,
    /// Available capacity for new sell orders, accounting for ALL resting + in-flight asks.
    /// available_ask_budget = max_position - (-position + resting_asks + inflight_asks)
    pub available_ask_budget: f64,

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

    // NOTE: stochastic_spread_multiplier has been REMOVED.
    // All uncertainty is now handled through gamma scaling (kappa_ci_width flows
    // through uncertainty_scalar). The GLFT formula naturally widens spreads
    // when gamma increases due to uncertainty.

    // ==================== Entropy-Based Distribution (always enabled) ====================
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
    /// Calibration progress [0.0, 1.0].
    /// Based on AS fills + kappa confidence.
    pub calibration_progress: f64,

    /// Whether calibration is complete.
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
    ///
    /// Where:
    /// - Q_hard = account_value × leverage × 0.5 / mark_price (margin solvency)
    /// - Q_soft = √(2 × acceptable_loss / (γ × σ²)) (GLFT risk budget)
    /// - latency_penalty = max(0.3, 1 - 0.5×√(τ/30ms)) (MEASURED τ)
    ///
    /// This replaces hardcoded target_liquidity when > 0.
    pub derived_target_liquidity: f64,

    // ==================== New Latent State Estimators (Phases 2-7) ====================

    // --- Particle Filter Volatility (Phase 2) ---
    /// Volatility from particle filter (bps per sqrt(second)).
    /// More accurate than bipower variation, especially with regime switching.
    pub sigma_particle: f64,

    /// Regime probabilities [P(Low), P(Normal), P(High), P(Extreme)].
    /// From particle filter ensemble.
    pub regime_probs: [f64; 4],

    // --- Informed Flow Model (Phase 3) ---
    /// Probability current trade is from informed trader.
    /// From online EM mixture model.
    pub p_informed: f64,

    /// Probability current trade is noise (uninformed).
    pub p_noise: f64,

    /// Probability current trade is forced (liquidation/rebalance).
    pub p_forced: f64,

    /// Confidence in flow decomposition.
    pub flow_decomp_confidence: f64,

    // --- Fill Rate Model (Phase 4) ---
    /// Expected fill rate at standard depth (8 bps).
    pub fill_rate_8bps: f64,

    /// Optimal depth to achieve 50% fill rate.
    pub optimal_depth_50pct: f64,

    // --- Adverse Selection Decomposition (Phase 5) ---
    /// Permanent adverse selection (information component) in bps.
    pub as_permanent_bps: f64,

    /// Temporary adverse selection (market impact) in bps.
    pub as_temporary_bps: f64,

    /// Timing adverse selection (inventory pressure) in bps.
    pub as_timing_bps: f64,

    /// Total adverse selection (sum of components) in bps.
    pub total_as_bps: f64,

    // --- Edge Surface (Phase 6) ---
    /// Current expected edge in basis points.
    /// Accounts for spread, AS, and fees.
    pub current_edge_bps: f64,

    /// Whether we should quote based on edge surface analysis.
    /// True when expected edge > 0 with sufficient confidence.
    pub should_quote_edge: bool,

    // --- Joint Dynamics (Phase 7) ---
    /// Whether in toxic state based on joint parameter correlations.
    /// True when AS and informed flow are highly correlated with volatility.
    pub is_toxic_joint: bool,

    /// Sigma-kappa correlation (negative = widening spreads during vol).
    pub sigma_kappa_correlation: f64,

    // ==================== L2 Decision Engine Outputs (A-S Framework) ====================
    /// Reservation price shift from L2 decision engine (price units).
    /// From A-S: δ_μ = μ / (γσ²) where μ is expected edge.
    /// Positive = shift quotes UP (aggressive asks), Negative = shift DOWN (aggressive bids).
    /// Add this to microprice when calculating bid/ask prices.
    pub l2_reservation_shift: f64,

    // NOTE: l2_spread_multiplier has been REMOVED.
    // The L2 Bayesian uncertainty premium now flows through gamma scaling
    // via kappa_ci_width → uncertainty_scalar. This eliminates the arbitrary
    // spread multiplier and routes all uncertainty through the GLFT formula.

    // ==================== Proactive Position Management (Small Fish) ====================
    /// Whether proactive directional skew is enabled.
    /// When enabled, skews quotes to BUILD position with momentum (opposite of inventory skew).
    pub enable_proactive_skew: bool,

    /// Sensitivity for proactive skew (bps per unit momentum×confidence).
    /// Higher = more aggressive position building with momentum.
    pub proactive_skew_sensitivity: f64,

    /// Minimum momentum confidence to apply proactive skew.
    /// Below this threshold, proactive skew is 0.
    pub proactive_min_momentum_confidence: f64,

    /// Minimum momentum magnitude (bps) to apply proactive skew.
    /// Below this threshold, proactive skew is 0.
    pub proactive_min_momentum_bps: f64,

    // ==================== Regime-Conditioned Kappa ====================
    /// Effective kappa from regime-conditioned estimator (blended across regimes).
    /// Per-regime priors: Low=3000, Normal=2000, High=1000, Extreme=500.
    /// Used for regime-aware kappa blending in ladder strategy.
    /// None = regime kappa not available (feature disabled or not warmed up).
    pub regime_kappa: Option<f64>,

    /// Current dominant volatility regime index from regime kappa estimator.
    /// 0=Low, 1=Normal, 2=High, 3=Extreme.
    pub regime_kappa_current_regime: usize,

    /// Regime-conditioned expected adverse selection cost (bps).
    pub regime_as_expected_bps: f64,
    /// Regime-conditioned additive risk premium (bps).
    pub regime_risk_premium_bps: f64,
    /// Regime-conditioned skew gain multiplier.
    pub regime_skew_gain: f64,
    /// Controller objective for current regime.
    pub controller_objective: ControllerObjective,
    /// Max position fraction for current regime (regime tightening).
    pub max_position_fraction: f64,
    /// Total risk premium (bps): regime risk premium + hawkes addon + toxicity addon + staleness addon.
    /// Used as input to solve_min_gamma() for self-consistent spread floor.
    pub total_risk_premium_bps: f64,
    /// Regime gamma multiplier — routes regime risk through gamma instead of floor.
    /// Continuously blended from regime probabilities via
    /// `RegimeState::blended_gamma_multiplier_from_probs()` or
    /// `RegimeState::effective_gamma_multiplier()`.
    /// Discrete values per regime: Calm=1.0, Normal=1.2, Volatile=2.0, Extreme=3.0.
    /// Blended example: probs [0.2, 0.3, 0.3, 0.2] -> 1.76.
    pub regime_gamma_multiplier: f64,

    // ==================== Cost-Basis-Aware Quoting ====================
    /// Average entry price for current position. None if flat.
    /// From PnLTracker::avg_entry_price.
    pub avg_entry_price: Option<f64>,
    /// Breakeven price including fees. Entry + fees for longs, entry - fees for shorts.
    /// Selling below this (long) or buying above this (short) guarantees a loss.
    pub breakeven_price: f64,
    /// Unrealized PnL in basis points relative to entry.
    /// Positive = in profit, negative = underwater.
    pub unrealized_pnl_bps: f64,

    // ==================== Bayesian Gamma Components (Alpha Plan) ====================
    /// Trend confidence [0, 1] from directional signals.
    /// High confidence → can quote tighter spreads (lower gamma).
    /// Source: momentum persistence, book imbalance consistency, etc.
    pub trend_confidence: f64,

    /// Bootstrap calibration confidence [0, 1] - P(calibrated) from Bayesian bootstrap.
    /// Low confidence → need fills → quote tighter (lower gamma).
    /// High confidence → calibrated → normal GLFT behavior.
    pub bootstrap_confidence: f64,

    /// Adverse selection uncertainty (std dev of posterior).
    /// High uncertainty → quote wider (higher gamma) for safety.
    /// Source: RegimeAwareBayesianAdverse.variance().sqrt()
    pub adverse_uncertainty: f64,

    /// Current adverse selection regime (0=Calm, 1=Normal, 2=Volatile).
    /// Volatile regime → wider spreads (higher gamma).
    pub adverse_regime: usize,

    // ==================== Predictive Bias (A-S Extension) ====================
    /// Changepoint probability from BOCD detector [0, 1].
    /// High values indicate imminent regime change.
    /// Used to compute predictive bias β_t for preemptive quote skewing.
    pub changepoint_prob: f64,

    // ==================== First-Principles Stochastic Control ====================
    /// Belief-derived predictive bias β_t = E[μ | data].
    /// This is the posterior mean of drift from Normal-Inverse-Gamma update.
    /// NOT a heuristic - it's the mathematically correct signal.
    /// Positive = bullish drift, Negative = bearish drift.
    pub belief_predictive_bias: f64,

    /// Belief-derived expected volatility E[σ | data].
    /// From the NIG posterior over (μ, σ²).
    pub belief_expected_sigma: f64,

    /// Belief-derived expected fill intensity E[κ | fills].
    /// From Gamma posterior over fill intensity.
    pub belief_expected_kappa: f64,

    /// Confidence in belief system [0, 1].
    /// Low values → blend toward legacy behavior.
    pub belief_confidence: f64,

    /// Whether to use belief-derived parameters instead of heuristics.
    /// Feature flag for gradual rollout.
    pub use_belief_system: bool,

    // ==================== Position Direction Confidence ====================
    /// Position direction confidence [0, 1].
    /// High confidence (>0.5) = position likely from informed flow.
    /// Based on: belief alignment, fill alignment, time held without adverse move.
    /// Used by CalibratedRiskModel to modulate gamma via beta_confidence (NEGATIVE).
    /// High confidence → lower gamma → more two-sided quoting.
    pub position_direction_confidence: f64,

    /// Time in seconds since last adverse price move against position.
    /// Longer time without adverse move → higher confidence in position.
    /// Updated when price moves against position direction.
    pub time_since_adverse_move: f64,

    // Used to transform inventory_ratio for GLFT skew calculation.
    /// P(continuation) from Beta-Binomial posterior.
    /// Higher values indicate the position direction is likely to continue.
    /// Range: [0, 1], prior mean depends on regime.
    pub continuation_p: f64,

    /// Confidence in continuation estimate [0, 1].
    /// Based on variance reduction from uniform prior.
    /// Higher with more fill observations.
    pub continuation_confidence: f64, // ==================== Bayesian Learned Parameters (Phase 6) ====================
    /// Whether using Bayesian learned parameters instead of static config values.
    /// When true, learned_kappa, learned_alpha_touch, etc. are being used.
    pub use_learned_parameters: bool,

    /// Learned kappa (fill intensity) from Bayesian parameter learner.
    /// Used in GLFT calculations when calibrated and enabled.
    /// Falls back to adaptive_kappa or estimator kappa when not calibrated.
    pub learned_kappa: f64,

    /// Learned alpha_touch (informed probability at touch) from Bayesian parameter learner.
    /// Used in Kelly sizing when calibrated and enabled.
    /// Falls back to config kelly_alpha_touch when not calibrated.
    pub learned_alpha_touch: f64,

    /// Learned spread floor in bps from Bayesian parameter learner.
    /// Used for minimum spread enforcement when calibrated and enabled.
    /// Falls back to config min_spread_floor when not calibrated.
    pub learned_spread_floor_bps: f64,

    /// Whether learned parameters are calibrated (Tier 1 ready).
    /// True when alpha_touch has 50+ obs with CV < 0.5, etc.
    pub learned_params_calibrated: bool,

    // === Cached BBO from L2 Book ===
    /// Cached best bid from L2 book (for bounding inventory skew offset).
    /// This is the actual exchange BBO, NOT derived from market_mid ± spread.
    pub cached_best_bid: f64,
    /// Cached best ask from L2 book (for bounding inventory skew offset).
    pub cached_best_ask: f64,

    // === Drift Rate (Kalman / GLFT Drift Integration) ===
    /// Kalman posterior drift rate per second (fractional, not bps).
    /// Positive = price rising. Used by GLFT half_spread_with_drift
    /// for asymmetric bid/ask spread (classical μ·T term).
    /// UNCLAMPED: no ±3 bps cap. Kalman posterior variance naturally bounds.
    pub drift_rate_per_sec: f64,

    /// Raw (unshrunk) drift rate for PPIP, bypasses James-Stein shrinkage.
    /// Fractional per second, same units as drift_rate_per_sec.
    pub drift_rate_per_sec_raw: f64,

    /// Kalman posterior drift uncertainty √P (bps).
    /// Lower = more confident. Used for logging and E[PnL] confidence.
    pub drift_uncertainty_bps: f64,

    // === Phase 3B: Direction Hysteresis ===
    /// Bid-side gamma multiplier from hysteresis (>1.0 = wider bids).
    /// Penalizes re-accumulation of previous direction after zero-crossing.
    pub hysteresis_bid_gamma_mult: f64,
    /// Ask-side gamma multiplier from hysteresis (>1.0 = wider asks).
    pub hysteresis_ask_gamma_mult: f64,

    /// Enable per-level E[PnL] filter (replaces binary quote gate).
    /// When true, levels with E[PnL] ≤ 0 are dropped instead of using
    /// NoQuote/OnlyBids/OnlyAsks binary decisions.
    /// Default false during dual-run validation, true after confirmation.
    pub use_epnl_filter: bool,

    /// Funding carry cost in bps for E[PnL] computation.
    /// Positive when we're on the paying side of funding.
    pub funding_carry_bps: f64,

    /// Short-term vs long-term realized vol ratio (σ_short / σ_long).
    /// \> 1.0 = volatility expansion → widen spreads.
    /// Phase 8: cold tier feature.
    pub vol_term_structure_ratio: f64,

    /// Kurtosis-based AS ratio scaling (≥ 1.0).
    /// Fat tails (excess kurtosis > 0) → scale up adverse selection estimate.
    /// Phase 8: cold tier feature.
    pub as_ratio_adjustment: f64,

    /// Current drawdown as fraction of account value [0.0, 1.0+].
    /// Used by GLFT effective_gamma for continuous risk aversion scaling.
    /// Higher drawdown → higher gamma → wider spreads.
    pub current_drawdown_frac: f64,

    // === Capital Tier ===
    /// Capital tier from CapitalProfile — used for logging, metrics, and warmup
    /// bootstrapping only. Does NOT drive code path selection in ladder generation.
    pub capital_tier: CapitalTier,

    /// Capacity budget computed at the top of each quote cycle.
    /// Contains min_viable_depth_bps, viable_levels_per_side, and tier info.
    /// Used by ladder and QueueValue for depth-aware decisions.
    pub capacity_budget: Option<CapacityBudget>,

    /// Capital-aware policy derived from `CapitalTier`. Encodes ALL tier-dependent
    /// behavior (ladder levels, reconcile mode, trust caps, warmup, quota) in one
    /// place, threaded through the entire pipeline.
    pub capital_policy: CapitalAwarePolicy,

    // === Kill Switch Headroom (for dynamic margin utilization) ===
    /// Kill switch headroom [0, 1]. Fraction of drawdown budget remaining.
    /// 1.0 = no drawdown, 0.0 = at kill switch threshold.
    /// Used by tick grid for dynamic margin utilization scaling.
    pub kill_switch_headroom: f64,

    // === CalibrationCoordinator (Bootstrap from Book) ===
    /// Kappa from CalibrationCoordinator (L2-derived, fill-refined).
    /// Conservative during warmup (0.5x factor), converges to fill-based kappa.
    /// Used in kappa priority chain: Robust > Adaptive > Coordinator > Legacy
    pub coordinator_kappa: f64,
    /// Uncertainty premium from CalibrationCoordinator (bps).
    /// Added to spreads during Cold/Warming phase, decays as fills accumulate.
    pub coordinator_uncertainty_premium_bps: f64,
    /// Whether coordinator kappa is available (seeded from L2 profile).
    pub use_coordinator_kappa: bool,

    // === Fix 2: Inviolable Spread Floor ===
    /// Dynamic AS floor in bps from AdverseSelectionEstimator.
    /// Added to physical floor: fee + latency + AS → solve_min_gamma().
    /// Seeded from checkpoint when no live markout data yet.
    pub as_floor_bps: f64,

    /// AS posterior variance in bps² for uncertainty premium computation.
    /// floor = E[AS] + k × √Var[AS] where k = as_uncertainty_premium_k.
    pub as_floor_variance_bps2: f64,

    /// Profile-level spread floor in bps (static safety bound from SpreadProfile).
    /// max()'d with adaptive floor — can never make spreads tighter.
    pub profile_spread_floor_bps: f64,

    // === Fix 4: Proactive AS — Hawkes σ_conditional ===
    // REMOVED: sigma_cascade_mult — WS2 CovarianceTracker handles realized vol feedback.
    // The Bayesian posterior on σ automatically inflates when realized vol > predicted.

    // === Fix 3: Ghost liquidity detection ===
    /// Ghost liquidity gamma multiplier [1.0, 5.0].
    /// When book kappa >> robust kappa (>3x), book shows standing orders
    /// that don't represent real fill intensity. Inflate γ to widen spread.
    pub ghost_liquidity_gamma_mult: f64,

    // ==================== Governor Asymmetric Spread Widening ====================
    /// Per-side governor spread addon for bids (bps) [0.0, 25.0].
    /// When long, bids increase position → widen. When short, stays 0.0.
    /// Source: InventoryGovernor.assess().increasing_side_addon_bps
    pub governor_bid_addon_bps: f64,
    /// Per-side governor spread addon for asks (bps) [0.0, 25.0].
    /// When short, asks increase position → widen. When long, stays 0.0.
    pub governor_ask_addon_bps: f64,

    // ==================== Funding Carry Per-Side ====================
    /// Funding carry cost for bid side (bps). Positive when longs pay funding.
    /// At 0.01%/hr funding: ~0.003 bps (negligible). At 1%/hr: ~0.3 bps (meaningful).
    pub funding_carry_bid_bps: f64,
    /// Funding carry cost for ask side (bps). Positive when shorts pay funding.
    pub funding_carry_ask_bps: f64,

    // ==================== Options-Theoretic Volatility Floor ====================
    /// Dynamic volatility-aware spread floor (bps).
    /// Formula: safety_mult × sigma × sqrt(tau) × sqrt(2/pi).
    /// Adapts to volatility — tight in calm, wide in stress.
    pub option_floor_bps: f64,

    // ==================== Self-Impact Estimator ====================
    /// Additive spread widening from self-impact (bps).
    /// Formula: coefficient × (our_fraction)^2.
    /// At 40% book dominance: ~0.8 bps. At 80%: ~3.2 bps.
    pub self_impact_addon_bps: f64,

    // REMOVED: staleness_addon_bid/ask_bps — WS5 latency-aware mid handles price displacement.
    // REMOVED: flow_toxicity_addon_bid/ask_bps — β_toxicity in CalibratedRiskModel handles
    //   informed flow through γ. Higher toxicity → higher γ → wider spreads automatically.

    // ==================== Bayesian Fair Value Model ====================
    /// Cascade z-score from Bayesian fair value posterior drift rate.
    /// High values (> 2.5) indicate a sweep cascade detected via posterior shift velocity.
    pub fv_cascade_score: f64,
    /// Fair value model confidence [0, 1] based on fill count and posterior tightness.
    /// Used for blending posterior mean into microprice.
    pub fv_confidence: f64,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            sigma: 0.0001,                   // 0.01% per-second volatility (clean)
            sigma_total: 0.0001,             // Same initially
            sigma_effective: 0.0001,         // Same initially
            sigma_leverage_adjusted: 0.0001, // Same initially (no leverage effect)
            // WS2: Dual-timescale inventory control
            tau_inventory_s: 60.0,      // Default 60s holding period
            tau_variance_s2: 900.0,     // Default 30s std dev
            sigma_sq_variance_bma: 0.0, // BMA not warmed up
            kappa: 100.0,               // Moderate depth decay
            kappa_bid: 100.0,           // Same as kappa initially
            kappa_ask: 100.0,           // Same as kappa initially
            is_heavy_tailed: false,     // Assume exponential tails
            kappa_cv: 1.0,              // CV=1 for exponential
            // V3: Robust Kappa Orchestrator
            kappa_robust: 2000.0, // Start at prior (will be updated from orchestrator)
            use_kappa_robust: true, // Default ON - use robust kappa for spreads
            kappa_outlier_count: 0, // No outliers detected yet
            // V2: Uncertainty Quantification
            kappa_uncertainty: 0.0,   // Will be computed from posterior
            kappa_95_lower: 100.0,    // Conservative lower bound
            kappa_95_upper: 100.0,    // Conservative upper bound
            kappa_ci_width: 1.0,      // High uncertainty initially (CI width / mean)
            toxicity_score: 0.0,      // No toxicity initially
            param_correlation: 0.0,   // No correlation initially
            as_factor: 1.0,           // No AS adjustment initially
            arrival_intensity: 0.5,   // 0.5 volume ticks per second
            is_toxic_regime: false,   // Default: not toxic
            jump_ratio: 1.0,          // Default: normal diffusion
            momentum_bps: 0.0,        // Default: no momentum
            flow_imbalance: 0.0,      // Default: balanced flow
            lead_lag_signal_bps: 0.0, // Default: no cross-exchange signal
            drift_signal_bps: 0.0,    // Default: no drift signal
            lead_lag_confidence: 0.0, // Default: no confidence
            falling_knife_score: 0.0, // Default: no falling knife
            rising_knife_score: 0.0,  // Default: no rising knife
            book_imbalance: 0.0,      // Default: balanced book
            microprice: 0.0,          // Will be set from estimator
            market_mid: 0.0,          // Will be set from latest AllMids
            beta_book: 0.0,           // Will be learned from data
            beta_flow: 0.0,           // Will be learned from data
            // Tier 1: Adverse Selection
            as_spread_adjustment: 0.0, // No adjustment until warmed up
            as_spread_adjustment_bid: 0.0,
            as_spread_adjustment_ask: 0.0,
            predicted_alpha: 0.0, // Default: no informed flow detected
            as_warmed_up: false,  // Starts not warmed up
            depth_decay_as: None, // No calibrated model initially
            conditional_as_posterior_mean_bps: None, // Learned from fill AS data, not magic number
            // Tier 1: Pre-Fill AS Classifier (Phase 3)
            pre_fill_toxicity_bid: 0.0, // No toxicity initially
            pre_fill_toxicity_ask: 0.0, // No toxicity initially
            // Tier 1: Liquidation Cascade
            tail_risk_intensity: 0.0, // Default to 0 intensity (calm)
            should_pull_quotes: false,
            cascade_intensity: 0.0, // Default to 0 intensity (calm): full size
            // Tier 2: Hawkes Order Flow
            hawkes_buy_intensity: 0.0,
            hawkes_sell_intensity: 0.0,
            hawkes_imbalance: 0.0,
            hawkes_activity_percentile: 0.5,
            // Hawkes Excitation Prediction (Phase 7: Bayesian Fusion)
            hawkes_p_cluster: 0.0,            // No cluster risk initially
            hawkes_excitation_penalty: 1.0,   // No penalty (full edge)
            hawkes_is_high_excitation: false, // Not excited initially
            hawkes_branching_ratio: 0.3,      // Moderate default (from prior)
            hawkes_spread_widening: 1.0,      // No widening initially
            hawkes_expected_cluster_time: f64::INFINITY, // No imminent cluster
            hawkes_excess_intensity_ratio: 1.0, // At baseline
            // Phase 8: RL Policy Recommendations
            rl_spread_delta_bps: 0.0,
            rl_bid_skew_bps: 0.0,
            rl_ask_skew_bps: 0.0,
            rl_confidence: 0.0,
            rl_is_exploration: false,
            rl_expected_q: 0.0,
            rl_action_applied: false,
            bandit_spread_additive_bps: 0.0,
            bandit_is_exploration: false,
            // Phase 8: Competitor Model
            competitor_snipe_prob: 0.1,    // 10% baseline
            competitor_spread_factor: 1.0, // No adjustment
            competitor_count: 3.0,         // Assume 3 competitors
            market_share: 0.0,             // Unknown until observed
            // Phase 9: Rate Limit Death Spiral Prevention
            rate_limit_headroom_pct: 1.0, // Full budget available
            quota_shadow_spread_bps: 0.0, // No shadow spread at full headroom
            // Tier 2: Funding Rate
            funding_rate: 0.0,
            predicted_funding_cost: 0.0,
            time_to_funding_settlement_s: 28800.0, // 8 hours in seconds (unknown)
            open_interest: 0.0,                    // Will be set from exchange data
            oi_change_1m: 0.0,                     // Will be computed from OI history
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
            p_momentum_continue: 0.5,
            // HJB Controller (stochastic integration)
            use_hjb_skew: false,         // Default OFF for safety
            hjb_optimal_skew: 0.0,       // Will be computed
            hjb_inventory_target: 0.0,   // Zero inventory target
            hjb_is_terminal_zone: false, // Not in terminal zone
            // Drift-Adjusted Skew (first-principles momentum integration)
            use_drift_adjusted_skew: true, // ON by default for first-principles trading
            hjb_drift_urgency: 0.0,        // No urgency initially
            position_opposes_momentum: false, // No opposition initially
            urgency_score: 0.0,            // No urgency initially
            // Kalman Filter (stochastic integration)
            use_kalman_filter: false,    // Default OFF for safety
            kalman_fair_price: 0.0,      // Will be computed from Kalman filter
            kalman_uncertainty: 0.0,     // Will be computed from Kalman filter
            kalman_spread_widening: 0.0, // Will be computed from Kalman filter
            kalman_warmed_up: false,     // Not warmed up initially
            // Constrained Optimizer (stochastic integration)
            margin_available: 0.0, // Will be fetched from margin sizer
            leverage: 1.0,         // Default 1x leverage
            // Kelly-Stochastic Allocation (stochastic integration)
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
            pending_bid_exposure: 0.0,      // No resting orders initially
            pending_ask_exposure: 0.0,      // No resting orders initially
            available_bid_budget: f64::MAX, // Unconstrained until ExposureBudget wired
            available_ask_budget: f64::MAX,
            // Dynamic Position Limits
            dynamic_max_position: 0.0,    // Will be set from kill switch
            dynamic_limit_valid: false,   // Not valid until margin state refreshed
            margin_quoting_capacity: 0.0, // Will be computed from margin_available
            // Stochastic Constraints
            tick_size_bps: 0.5, // Conservative default; overridden by compute_tick_size_bps()
            latency_spread_floor: 0.0003, // 3 bps default floor
            near_touch_depth_usd: 0.0, // No depth data initially
            tight_quoting_allowed: false, // Conservative default
            tight_quoting_block_reason: Some("Warmup".to_string()),
            // NOTE: stochastic_spread_multiplier removed - uncertainty flows through gamma
            // Entropy-Based Distribution (always enabled)
            entropy_min_entropy: 1.5,      // At least ~4.5 effective levels
            entropy_base_temperature: 1.0, // Standard softmax
            entropy_min_allocation_floor: 0.02, // 2% minimum per level
            entropy_thompson_samples: 5,   // Moderate stochasticity
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
            calibration_progress: 0.0,   // Start at 0% calibration
            calibration_complete: false, // Not calibrated yet
            // Model-Derived Sizing (GLFT First Principles)
            account_value: 0.0,            // Will be set from margin state
            measured_latency_ms: 50.0,     // Default 50ms until measured
            estimated_fill_rate: 0.001,    // Conservative 0.001/sec until observed
            derived_target_liquidity: 0.0, // Will be computed from GLFT formula
            // === New Latent State Estimators (Phases 2-7) ===
            // Particle Filter Volatility (Phase 2)
            sigma_particle: 0.0, // Will be set from particle filter (bps/sqrt(s))
            regime_probs: [0.1, 0.7, 0.15, 0.05], // Default: mostly Normal regime
            // Informed Flow Model (Phase 3)
            p_informed: 0.05,            // 5% informed baseline
            p_noise: 0.90,               // 90% noise
            p_forced: 0.05,              // 5% forced
            flow_decomp_confidence: 0.0, // Not confident until warmed up
            // Fill Rate Model (Phase 4)
            fill_rate_8bps: 0.5,       // 50% fill rate at 8bps (prior)
            optimal_depth_50pct: 10.0, // 10 bps for 50% fill rate (prior)
            // Adverse Selection Decomposition (Phase 5)
            as_permanent_bps: 0.0, // No permanent AS until measured
            as_temporary_bps: 0.0, // No temporary AS until measured
            as_timing_bps: 0.0,    // No timing AS until measured
            total_as_bps: 0.0,     // Total starts at 0
            // Edge Surface (Phase 6)
            current_edge_bps: 0.0,   // No edge until measured
            should_quote_edge: true, // Default: safe to quote
            // Joint Dynamics (Phase 7)
            is_toxic_joint: false,         // Not toxic initially
            sigma_kappa_correlation: -0.3, // Typical negative correlation
            // L2 Decision Engine Outputs (A-S Framework)
            l2_reservation_shift: 0.0, // No shift initially (neutral)
            // NOTE: l2_spread_multiplier removed - uncertainty flows through gamma
            // Proactive Position Management (Small Fish)
            enable_proactive_skew: false,    // Off by default, opt-in
            proactive_skew_sensitivity: 2.0, // 2 bps per unit momentum×confidence
            proactive_min_momentum_confidence: 0.6, // Need 60% confidence
            proactive_min_momentum_bps: 5.0, // Need 5 bps minimum momentum
            // Regime-Conditioned Kappa
            regime_kappa: None, // Not available until regime estimator warmed up
            regime_kappa_current_regime: 1, // Normal regime default
            // Regime-conditioned objective and parameters
            regime_as_expected_bps: 1.0,
            regime_risk_premium_bps: 1.0,
            regime_skew_gain: 1.0,
            controller_objective: ControllerObjective::MeanRevert,
            max_position_fraction: 0.8,
            total_risk_premium_bps: 1.0,  // Default: 1 bps risk premium
            regime_gamma_multiplier: 1.0, // Default: Normal regime
            avg_entry_price: None,
            breakeven_price: 0.0,
            unrealized_pnl_bps: 0.0,
            // Component-Level Spread Addons
            // Bayesian Gamma Components (Alpha Plan)
            trend_confidence: 0.5,     // 50% confidence initially (uncertain)
            bootstrap_confidence: 0.0, // Not calibrated initially
            adverse_uncertainty: 0.1,  // Moderate uncertainty (10% std)
            adverse_regime: 1,         // Normal regime initially
            // Predictive Bias (A-S Extension)
            changepoint_prob: 0.0, // No changepoint detected initially
            // First-Principles Stochastic Control
            belief_predictive_bias: 0.0, // No drift bias until beliefs updated
            belief_expected_sigma: 0.0001, // Default sigma
            belief_expected_kappa: 200.0, // Conservative default for thin DEX
            belief_confidence: 0.0,      // Not confident until warmed up
            use_belief_system: true,     // Use first-principles belief system
            // Position Direction Confidence
            position_direction_confidence: 0.5, // Neutral confidence initially
            time_since_adverse_move: 0.0,       // No history initially
            // Position Continuation Model
            continuation_p: 0.5,          // Neutral prior
            continuation_confidence: 0.0, // No confidence until fills observed
            // Bayesian Learned Parameters (Phase 6)
            use_learned_parameters: false,    // Off until calibrated
            learned_kappa: 2000.0,            // Prior mean
            learned_alpha_touch: 0.25,        // Prior mean (25% informed)
            learned_spread_floor_bps: 5.0,    // Prior mean (5 bps)
            learned_params_calibrated: false, // Not calibrated initially
            // Cached BBO (0.0 = not yet received from L2 book)
            cached_best_bid: 0.0,
            cached_best_ask: 0.0,
            // Drift rate (Kalman)
            drift_rate_per_sec: 0.0,
            drift_rate_per_sec_raw: 0.0,
            drift_uncertainty_bps: 0.0,
            hysteresis_bid_gamma_mult: 1.0,
            hysteresis_ask_gamma_mult: 1.0,
            current_drawdown_frac: 0.0,
            // Capital tier
            capital_tier: CapitalTier::Large,
            // Capacity budget — computed per cycle in quote_engine
            capacity_budget: None,
            // Capital-aware policy — derived from tier, flows through entire pipeline
            capital_policy: CapitalAwarePolicy::default(),
            // Kill switch headroom
            kill_switch_headroom: 1.0, // Full headroom initially
            // CalibrationCoordinator — inactive until seeded from L2 profile
            coordinator_kappa: 0.0,
            coordinator_uncertainty_premium_bps: 0.0,
            use_coordinator_kappa: false,

            // Fix 2: AS floor
            as_floor_bps: 0.0,
            as_floor_variance_bps2: 0.0,
            profile_spread_floor_bps: 0.0,

            // Fix 3: Ghost liquidity
            ghost_liquidity_gamma_mult: 1.0,

            // Governor Asymmetric Spread Widening (additive bps)
            governor_bid_addon_bps: 0.0,
            governor_ask_addon_bps: 0.0,

            // Funding Carry Per-Side
            funding_carry_bid_bps: 0.0,
            funding_carry_ask_bps: 0.0,

            // Options-Theoretic Volatility Floor
            option_floor_bps: 0.0,

            // Self-Impact Estimator
            self_impact_addon_bps: 0.0,

            // Unified Adverse Selection Framework (Phase 4)
            use_epnl_filter: true,
            funding_carry_bps: 0.0,

            // Phase 8: Warm/Cold tier features
            vol_term_structure_ratio: 1.0,
            as_ratio_adjustment: 1.0,

            // Bayesian Fair Value Model
            fv_cascade_score: 0.0,
            fv_confidence: 0.0,
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
            tail_risk_intensity: self.tail_risk_intensity,
            should_pull_quotes: self.should_pull_quotes,
            cascade_intensity: self.cascade_intensity,
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
            hjb_inventory_target: self.hjb_inventory_target,
            hjb_is_terminal_zone: self.hjb_is_terminal_zone,
        }
    }

    /// Extract margin constraint parameters as a focused struct.
    pub fn margin_constraints(&self) -> params::MarginConstraintParams {
        params::MarginConstraintParams {
            margin_available: self.margin_available,
            leverage: self.leverage,
        }
    }

    /// Extract Kelly-Stochastic parameters as a focused struct.
    pub fn kelly_stochastic(&self) -> params::KellyStochasticConfigParams {
        params::KellyStochasticConfigParams {
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
            // NOTE: stochastic_spread_multiplier removed - uncertainty flows through gamma
        }
    }

    /// Extract entropy distribution parameters as a focused struct.
    pub fn entropy_distribution(&self) -> params::EntropyDistributionParams {
        params::EntropyDistributionParams {
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
                (self.tail_risk_intensity - 1.0) / 4.0
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
            // Condition 1: Use soft HMM probability instead of hard regime check
            // Block tight quoting if P(High) + P(Extreme) > 0.5 (elevated regime)
            let p_elevated = self.regime_probs[2] + self.regime_probs[3];
            if p_elevated > 0.5 {
                can_quote_tight = false;
                block_reason = Some(format!("Elevated regime P(H+E)={:.0}%", p_elevated * 100.0));
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
                block_reason = Some(format!("Excluded hour {current_hour_utc} UTC"));
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
        // NOTE: stochastic_spread_multiplier has been removed entirely.
        // All uncertainty now flows through gamma via uncertainty_scalar.

        // === 4. Proactive Position Management (Small Fish Strategy) ===
        // Copy proactive skew config values from StochasticConfig to MarketParams
        // so GLFT strategy has access to them without needing the full config.
        self.enable_proactive_skew = config.enable_proactive_skew;
        self.proactive_skew_sensitivity = config.proactive_skew_sensitivity;
        self.proactive_min_momentum_confidence = config.proactive_min_momentum_confidence;
        self.proactive_min_momentum_bps = config.proactive_min_momentum_bps;
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
        config_target: f64,
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

        let q_soft =
            ((2.0 * acceptable_loss) / (gamma_eff * sigma_sq)).sqrt() * fill_rate_factor / 2.0;

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

        // === Fix 6: Geometric Blend ===
        // Position sizes have multiplicative risk effects → geometric mean is the
        // appropriate interpolator. More conservative when config and derived disagree.
        // α = warmup_progress [0.01, 0.99]: starts near config, converges to derived.
        let margin_capped = config_target.max(0.01).min(q_hard);
        let derived_constrained = q_soft_adjusted.min(per_level_cap).max(0.01);
        let alpha = self.adaptive_warmup_progress.clamp(0.01, 0.99);
        self.derived_target_liquidity = (margin_capped.ln() * (1.0 - alpha)
            + derived_constrained.ln() * alpha)
            .exp()
            .max(min_viable);

        tracing::debug!(
            q_hard = %format!("{:.6}", q_hard),
            q_soft = %format!("{:.6}", q_soft),
            q_soft_adjusted = %format!("{:.6}", q_soft_adjusted),
            config_target = %format!("{:.6}", config_target),
            alpha = %format!("{:.2}", alpha),
            latency_penalty = %format!("{:.2}", latency_penalty),
            per_level_cap = %format!("{:.6}", per_level_cap),
            min_viable = %format!("{:.6}", min_viable),
            derived = %format!("{:.6}", self.derived_target_liquidity),
            account_value = %format!("{:.2}", self.account_value),
            gamma = %format!("{:.3}", gamma_eff),
            sigma = %format!("{:.6}", self.sigma),
            latency_ms = %format!("{:.1}", tau_ms),
            fill_rate = %format!("{:.4}", self.estimated_fill_rate),
            "Fix 6: geometric blend target liquidity"
        );
    }
}

impl MarketParams {
    /// Build minimal MarketParams from just the estimator.
    ///
    /// This is useful for fill processing where we need basic market state
    /// but don't have access to all the full parameter sources.
    pub fn from_estimator(
        estimator: &crate::market_maker::estimator::ParameterEstimator,
        mid: f64,
        _position: f64,
        max_position: f64,
    ) -> Self {
        let mut params = Self {
            // Volatility
            sigma: estimator.sigma_clean(),
            sigma_total: estimator.sigma_total(),
            sigma_effective: estimator.sigma_effective(),
            volatility_regime: estimator.volatility_regime(),

            // Kappa
            kappa: estimator.kappa(),
            kappa_bid: estimator.kappa_bid(),
            kappa_ask: estimator.kappa_ask(),

            // Microprice - use directly, EMA smoothing in microprice.rs handles outliers
            microprice: estimator.microprice(),
            market_mid: mid,

            // Flow
            flow_imbalance: estimator.flow_imbalance(),
            book_imbalance: estimator.book_imbalance(),
            momentum_bps: estimator.momentum_bps(),

            // Toxicity / Regime
            toxicity_score: estimator.soft_toxicity_score(),
            jump_ratio: estimator.jump_ratio(),
            p_informed: estimator.p_informed(),
            is_toxic_regime: estimator.is_toxic_regime(),

            // Position
            dynamic_max_position: max_position,

            ..Default::default()
        };

        // Set position-related derivations
        params.dynamic_limit_valid = true;

        params
    }
}

impl MarketParams {
    /// Compute confidence that current position is from informed flow.
    ///
    /// Returns [0, 1] where:
    ///   0.0 = no confidence (position likely adverse)
    ///   0.5 = neutral (flat position or uncertain)
    ///   1.0 = high confidence (position from informed fills, beliefs aligned)
    ///
    /// This confidence is used by CalibratedRiskModel.beta_confidence (NEGATIVE)
    /// to REDUCE gamma when we're confident, enabling more two-sided quoting.
    ///
    /// # Arguments
    /// * `position` - Current position (signed, positive = long)
    /// * `max_position` - Maximum allowed position for normalization
    ///
    /// # Components
    /// 1. **Belief alignment**: Do beliefs support our position direction?
    ///    - belief_predictive_bias aligns with position sign → high confidence
    /// 2. **Time stability**: How long held without adverse move?
    ///    - Longer hold → higher confidence (max at 60s)
    /// 3. **Flow alignment**: Is recent flow supporting our position?
    ///    - flow_imbalance aligns with position → higher confidence
    pub fn compute_position_direction_confidence(&self, position: f64, max_position: f64) -> f64 {
        const EPSILON: f64 = 1e-9;

        // Flat position → neutral confidence
        if max_position < EPSILON || (position.abs() / max_position) < 0.01 {
            return 0.5;
        }

        let position_sign = position.signum();

        // === Factor 1: Belief alignment ===
        // Do beliefs (predictive_bias from particle filter) support our position?
        // If we're long and drift is positive, or short and drift is negative
        let belief_alignment = if self.belief_confidence > 0.1 {
            let bias = self.belief_predictive_bias;
            let aligned =
                (position_sign > 0.0 && bias > 0.0) || (position_sign < 0.0 && bias < 0.0);
            if aligned {
                // Confidence-weighted alignment: stronger beliefs = more confidence
                // bias magnitude is typically small (e.g., 0.0001), so we normalize
                let bias_strength = (bias.abs() * 10000.0).min(1.0); // in bps, cap at 100bps
                self.belief_confidence * bias_strength
            } else {
                0.0
            }
        } else {
            0.0
        };

        // === Factor 2: Time stability ===
        // How long have we held this position without adverse price move?
        // Longer hold without adverse move → more confident
        let time_factor = (self.time_since_adverse_move / 60.0).min(1.0); // Max out at 1 min

        // === Factor 3: Flow alignment ===
        // Is recent flow (order flow imbalance) supporting our position?
        // Positive flow_imbalance = buying pressure → good for long
        // Negative flow_imbalance = selling pressure → good for short
        let flow_alignment = {
            let flow_aligned = (position_sign > 0.0 && self.flow_imbalance > 0.0)
                || (position_sign < 0.0 && self.flow_imbalance < 0.0);
            if flow_aligned {
                self.flow_imbalance.abs().min(1.0)
            } else {
                0.0
            }
        };

        // === Combine factors ===
        // Weighted combination with belief alignment being most important
        // since it's the most forward-looking signal
        let raw_confidence = 0.5 * belief_alignment + 0.25 * time_factor + 0.25 * flow_alignment;

        // Scale from [0, 1] raw to [0.2, 0.9] output range
        // This prevents extreme confidence values
        let scaled = 0.2 + raw_confidence * 0.7;

        scaled.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// WS7: Estimator Diagnostics
// ---------------------------------------------------------------------------

/// Unified diagnostics for all three estimators (drift, covariance, risk aversion)
/// plus execution quality metrics.
///
/// Logged every 10 quote cycles or when any estimator is in unusual state.
/// Replaces the scattered per-component logging with a single, structured record.
#[derive(Debug, Clone)]
pub struct EstimatorDiagnostics {
    // --- Drift health ---
    /// Current drift estimate in bps/sec.
    pub drift_mean_bps: f64,
    /// Posterior variance of drift estimate.
    pub drift_variance: f64,
    /// Fill-quote autocorrelation [0,1] — how much fills echo our own quotes.
    pub fill_quote_autocorrelation: f64,

    // --- Covariance health ---
    /// Model-predicted σ effective (per-sec fraction).
    pub sigma_effective_bps: f64,
    /// σ correction factor from realized vol tracker (target: 1.0).
    pub sigma_correction_factor: f64,

    // --- Risk aversion health ---
    /// Current log(γ) from calibrated risk model.
    pub log_gamma: f64,
    /// Tracking error between predicted and breakeven γ.
    pub gamma_tracking_error: f64,
    /// Signed bias (positive = over-estimating risk).
    pub gamma_signed_bias: f64,

    // --- Execution quality ---
    /// Anticipated mid adjustment in bps.
    pub anticipated_mid_shift_bps: f64,
    /// Execution latency EWMA in milliseconds.
    pub latency_ewma_ms: f64,

    // --- Outcome ---
    /// Current half-spread in bps (from GLFT).
    pub spread_bps: f64,
    /// Current skew in bps (from GLFT).
    pub skew_bps: f64,

    // --- Vol Sampling Bias (Q18) ---
    /// Ratio of sigma during fills vs non-fills. > 1.0 = upward vol bias.
    pub vol_sampling_bias_ratio: f64,
    /// Fraction of intervals that had fills (dead-time analysis).
    pub fillable_fraction: f64,
}

impl EstimatorDiagnostics {
    /// Check if any estimator is in an unusual state warranting immediate logging.
    pub fn is_unusual(&self) -> bool {
        self.sigma_correction_factor > 1.5
            || self.sigma_correction_factor < 0.5
            || self.gamma_tracking_error > 0.5
            || self.fill_quote_autocorrelation > 0.7
            || self.vol_sampling_bias_ratio > 1.5
    }
}
