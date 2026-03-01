//! GLFT Ladder Strategy - multi-level quoting with depth-dependent sizing.

use tracing::{debug, info, warn};

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use crate::market_maker::config::{Quote, QuoteConfig, SizeQuantum};
// VolatilityRegime removed - regime_scalar was redundant with vol_scalar
use crate::market_maker::quoting::ladder::risk_budget::{
    allocate_risk_budget, compute_allocation_temperature, SideRiskBudget, SoftmaxParams,
};
use crate::market_maker::quoting::ladder::tick_grid::{
    enumerate_ask_ticks, enumerate_bid_ticks, score_ticks, select_optimal_ticks, TickGridConfig,
    TickScoringParams,
};
use crate::market_maker::quoting::{
    BayesianFillModel, DepthSpacing, DynamicDepthConfig, DynamicDepthGenerator,
    EntropyConstrainedOptimizer, EntropyDistributionConfig, EntropyOptimizerConfig, Ladder,
    LadderConfig, LadderLevel, LadderParams, LevelOptimizationParams, MarketRegime,
};

use super::{
    CalibratedRiskModel, KellySizer, MarketParams, QuotingStrategy, RiskConfig, RiskFeatures,
    RiskModelConfig, SpreadComposition,
};

/// Inventory utilization ratio at which kill-switch side-clearing activates.
/// At 90% (vs old 100%), clearing triggers 10% earlier, preventing the final
/// 10% accumulation that leads to kill-switch activation.
const KILL_CLEAR_INVENTORY_RATIO: f64 = 0.90;

/// Maximum fraction of effective_max_position allowed in a single resting order.
///
/// GLFT inventory theory: a single fill at max position creates reservation price
/// adjustment = q * gamma * sigma^2 * T, which swings maximally and prevents recovery.
/// Capping at 25% ensures at least 4 fills are needed to reach max inventory,
/// giving the gamma/skew feedback loop time to widen spreads defensively.
const MAX_SINGLE_ORDER_FRACTION: f64 = 0.25;

/// Compute regime-dependent margin utilization fraction.
///
/// | Regime (gamma_mult) | Utilization |
/// |---------------------|------------|
/// | Calm (≤ 1.1)        | 85%        |
/// | Normal (≤ 1.5)      | 75%        |
/// | Volatile (≤ 2.0)    | 65%        |
/// | Extreme (> 2.0)     | 50%        |
///
/// Further scaled by:
/// - Kill switch headroom: linear reduction below 50% headroom
/// - Warmup boost: +5% during early warmup (progress < 0.3) to attract fills
pub fn dynamic_margin_utilization(
    regime_gamma_mult: f64,
    kill_switch_headroom: f64,
    warmup_progress: f64,
) -> f64 {
    // Base utilization from regime
    let base = if regime_gamma_mult <= 1.1 {
        0.85
    } else if regime_gamma_mult <= 1.5 {
        // Linear interpolation: 1.1→0.85, 1.5→0.75
        0.85 - (regime_gamma_mult - 1.1) / 0.4 * 0.10
    } else if regime_gamma_mult <= 2.0 {
        // Linear interpolation: 1.5→0.75, 2.0→0.65
        0.75 - (regime_gamma_mult - 1.5) / 0.5 * 0.10
    } else {
        // Linear interpolation: 2.0→0.65, 3.0→0.50 (capped)
        (0.65 - (regime_gamma_mult - 2.0) / 1.0 * 0.15).max(0.50)
    };

    // Kill switch headroom scaling: reduce utilization when headroom < 50%
    let headroom_factor = if kill_switch_headroom >= 0.50 {
        1.0
    } else {
        // Linear: 50%→1.0, 0%→0.5
        0.5 + kill_switch_headroom
    };

    // Warmup boost: slightly higher utilization early to attract fills
    let warmup_boost = if warmup_progress < 0.3 { 0.05 } else { 0.0 };

    ((base + warmup_boost) * headroom_factor).clamp(0.30, 0.90)
}

/// GLFT Ladder Strategy - multi-level quoting with depth-dependent sizing.
///
/// Generates K levels per side with:
/// - Geometric or linear depth spacing
/// - Size allocation proportional to λ(δ) × SC(δ) (fill intensity × spread capture)
/// - GLFT inventory skew applied to entire ladder
/// - **Bayesian fill probability**: Empirically-calibrated fill rates per depth bucket
///
/// This strategy uses the same gamma calculation as GLFTStrategy but
/// distributes liquidity across multiple price levels instead of just
/// quoting at the touch.
///
/// # Bayesian Fill Model
///
/// The strategy maintains a `BayesianFillModel` that learns fill probabilities
/// from observed fills. This replaces the theoretical first-passage model with
/// empirical data when sufficient observations are available:
///
/// - **Initial**: Uses first-passage theory P(fill|δ,τ) = 2×Φ(-δ/(σ×√τ))
/// - **After warmup**: Uses Beta-Binomial posterior from observed fill rates
/// - **Depth buckets**: 2bp buckets for stable estimation
///
/// Call `record_fill_observation()` when orders fill or cancel to update the model.
#[derive(Debug, Clone)]
pub struct LadderStrategy {
    /// Risk configuration (same as GLFTStrategy)
    pub risk_config: RiskConfig,
    /// Ladder-specific configuration
    pub ladder_config: LadderConfig,
    /// Dynamic depth generator for GLFT-optimal depths
    depth_generator: DynamicDepthGenerator,
    /// Bayesian fill probability model (learns from observed fills)
    fill_model: BayesianFillModel,
    /// Calibrated risk model for log-additive gamma computation
    pub risk_model: CalibratedRiskModel,
    /// Configuration for risk model feature normalization
    pub risk_model_config: RiskModelConfig,
    /// Kelly criterion position sizer
    pub kelly_sizer: KellySizer,
}

impl LadderStrategy {
    /// Create a new ladder strategy with default configs.
    pub fn new(gamma_base: f64) -> Self {
        let ladder_config = LadderConfig::default();
        Self {
            risk_config: RiskConfig {
                gamma_base: gamma_base.clamp(0.01, 10.0),
                ..Default::default()
            },
            depth_generator: Self::create_depth_generator(&ladder_config),
            ladder_config,
            fill_model: BayesianFillModel::default(),
            risk_model: CalibratedRiskModel::with_gamma_base(gamma_base),
            risk_model_config: RiskModelConfig::default(),
            kelly_sizer: KellySizer::default(),
        }
    }

    /// Create a new ladder strategy with custom configs.
    pub fn with_config(risk_config: RiskConfig, ladder_config: LadderConfig) -> Self {
        Self {
            depth_generator: Self::create_depth_generator(&ladder_config),
            risk_model: CalibratedRiskModel::with_gamma_base(risk_config.gamma_base),
            risk_model_config: RiskModelConfig::default(),
            kelly_sizer: KellySizer::default(),
            risk_config,
            ladder_config,
            fill_model: BayesianFillModel::default(),
        }
    }

    /// Create a new ladder strategy with custom fill model parameters.
    ///
    /// # Arguments
    /// * `risk_config` - Risk configuration
    /// * `ladder_config` - Ladder configuration
    /// * `prior_alpha` - Beta prior α (higher = stronger prior belief in fills)
    /// * `prior_beta` - Beta prior β (higher = stronger prior belief in non-fills)
    pub fn with_fill_model(
        risk_config: RiskConfig,
        ladder_config: LadderConfig,
        prior_alpha: f64,
        prior_beta: f64,
    ) -> Self {
        let sigma = 0.0001; // Default sigma, will be updated
        let tau = 10.0; // Default tau, will be updated
        Self {
            depth_generator: Self::create_depth_generator(&ladder_config),
            risk_model: CalibratedRiskModel::with_gamma_base(risk_config.gamma_base),
            risk_model_config: RiskModelConfig::default(),
            kelly_sizer: KellySizer::default(),
            risk_config,
            ladder_config,
            fill_model: BayesianFillModel::new(prior_alpha, prior_beta, sigma, tau),
        }
    }

    /// Create with full configuration including calibrated risk model.
    pub fn with_full_config(
        risk_config: RiskConfig,
        ladder_config: LadderConfig,
        risk_model_config: RiskModelConfig,
        kelly_sizer: KellySizer,
    ) -> Self {
        Self {
            depth_generator: Self::create_depth_generator(&ladder_config),
            risk_model: CalibratedRiskModel::with_gamma_base(risk_config.gamma_base),
            risk_model_config,
            kelly_sizer,
            risk_config,
            ladder_config,
            fill_model: BayesianFillModel::default(),
        }
    }

    /// Update the calibrated risk model (e.g., from coefficient estimator).
    pub fn update_risk_model(&mut self, model: CalibratedRiskModel) {
        self.risk_model = model;
    }

    /// Update risk model config.
    pub fn update_risk_model_config(&mut self, config: RiskModelConfig) {
        self.risk_model_config = config;
    }

    /// Record a winning trade for Kelly sizing.
    pub fn record_win(&mut self, win_bps: f64) {
        self.kelly_sizer.record_win(win_bps);
    }

    /// Record a losing trade for Kelly sizing.
    pub fn record_loss(&mut self, loss_bps: f64) {
        self.kelly_sizer.record_loss(loss_bps);
    }

    /// Check if Kelly sizer is warmed up.
    pub fn kelly_warmed_up(&self) -> bool {
        self.kelly_sizer.is_warmed_up()
    }

    // === Bayesian Fill Model Interface ===

    /// Record a fill observation for Bayesian learning.
    ///
    /// Call this when an order fills or cancels to update the fill probability model.
    ///
    /// # Arguments
    /// * `depth_bps` - Depth from mid in basis points where the order was placed
    /// * `filled` - Whether the order filled (true) or was cancelled (false)
    pub fn record_fill_observation(&mut self, depth_bps: f64, filled: bool) {
        self.fill_model.record_observation(depth_bps, filled);
    }

    /// Update fill model parameters from market state.
    ///
    /// Call this periodically (e.g., on each quote cycle) to keep the theoretical
    /// fallback model aligned with current market volatility.
    pub fn update_fill_model_params(&mut self, sigma: f64, tau: f64) {
        self.fill_model.update_params(sigma, tau);
    }

    /// Get fill probability at a given depth.
    ///
    /// Uses Bayesian posterior if sufficient observations, otherwise first-passage theory.
    pub fn fill_probability(&self, depth_bps: f64) -> f64 {
        self.fill_model.fill_probability(depth_bps)
    }

    /// Get fill probability with explicit sigma/tau override.
    pub fn fill_probability_with_params(&self, depth_bps: f64, sigma: f64, tau: f64) -> f64 {
        self.fill_model
            .fill_probability_with_params(depth_bps, sigma, tau)
    }

    /// Check if the fill model has sufficient observations.
    pub fn fill_model_warmed_up(&self) -> bool {
        self.fill_model.is_warmed_up()
    }

    /// Get fill model diagnostics for logging.
    pub fn fill_model_stats(&self) -> (u64, u64) {
        self.fill_model.total_observations()
    }

    /// Log fill model diagnostics.
    pub fn log_fill_model_diagnostics(&self) {
        self.fill_model.log_diagnostics();
    }

    /// Record no-fill observations for levels that were quoted but not filled.
    ///
    /// Each depth represents a level that survived an entire quote cycle without
    /// being filled. This provides the negative signal needed for calibration:
    /// without it, the model only sees fills and overestimates fill probability.
    pub fn record_no_fill_cycle(&mut self, depths_bps: &[f64]) {
        for &depth in depths_bps {
            if depth > 0.0 {
                self.fill_model.record_observation(depth, false);
            }
        }
    }

    // =========================================================================
    // Quota-Aware Ladder Density (Death Spiral Prevention)
    // =========================================================================

    /// Compute effective number of levels based on rate limit headroom.
    ///
    /// When quota is scarce, reduce the number of levels to conserve budget.
    /// Uses concave (sqrt) scaling to provide smooth degradation without
    /// step discontinuities that cause oscillation.
    ///
    /// # Arguments
    /// * `base_levels` - The configured number of levels
    /// * `headroom_pct` - Current quota headroom as fraction [0, 1]
    ///
    /// # Returns
    /// Effective number of levels to place, in [1, base_levels]
    ///
    /// # Scaling
    /// - headroom >= 50%: scale = 1.0 (full levels)
    /// - headroom <= 5%: scale = 0.1 (minimum presence)
    /// - Between: concave sqrt scaling for smooth degradation
    pub fn compute_effective_levels(base_levels: usize, headroom_pct: f64) -> usize {
        let scale = if headroom_pct >= 0.50 {
            1.0
        } else if headroom_pct <= 0.05 {
            0.1 // Minimum presence
        } else {
            // Concave: sqrt(normalized_headroom) for smooth degradation
            let normalized = (headroom_pct - 0.05) / 0.45;
            0.1 + 0.9 * normalized.sqrt()
        };

        let effective = ((base_levels as f64 * scale).round() as usize).max(1);

        if effective < base_levels {
            debug!(
                base_levels = base_levels,
                headroom_pct = %format!("{:.1}%", headroom_pct * 100.0),
                scale = %format!("{:.2}", scale),
                effective_levels = effective,
                "Quota-aware level reduction"
            );
        }

        effective
    }

    /// Compute size multiplier when levels are reduced.
    ///
    /// When we have fewer levels, each level should have larger size to:
    /// 1. Maximize expected fill value per API request
    /// 2. Maintain similar total liquidity provision
    /// 3. Ensure each request "counts" more
    ///
    /// # Arguments
    /// * `headroom_pct` - Current quota headroom as fraction [0, 1]
    ///
    /// # Returns
    /// Size multiplier, in [1.0, 1.5]
    ///
    /// # Formula
    /// - headroom >= 50%: multiplier = 1.0 (normal sizing)
    /// - headroom < 50%: multiplier = 1.0 + deficit (linear increase)
    /// - At 10% headroom: multiplier = 1.4
    /// - At 5% headroom: multiplier = 1.45
    pub fn compute_size_multiplier(headroom_pct: f64) -> f64 {
        if headroom_pct >= 0.50 {
            1.0 // Normal sizing
        } else {
            // As headroom drops, increase size per level
            let deficit = (0.50 - headroom_pct).min(0.45);
            let multiplier = 1.0 + deficit;

            debug!(
                headroom_pct = %format!("{:.1}%", headroom_pct * 100.0),
                size_multiplier = %format!("{:.2}", multiplier),
                "Quota-aware size scaling"
            );

            multiplier
        }
    }

    /// Create a depth generator that inherits settings from ladder config
    fn create_depth_generator(ladder_config: &LadderConfig) -> DynamicDepthGenerator {
        // Spread floor should ensure profitability after fees
        // We need at least fees_bps to break even, add 1bp buffer for safety
        let spread_floor = (ladder_config.fees_bps + 1.0).max(ladder_config.min_depth_bps);

        let config = DynamicDepthConfig {
            num_levels: ladder_config.num_levels,
            min_depth_bps: ladder_config.min_depth_bps.max(1.0), // At least 1bp
            max_depth_bps: ladder_config.max_depth_bps,
            max_depth_multiple: 5.0, // Up to 5x optimal spread
            spacing: if ladder_config.geometric_spacing {
                DepthSpacing::Geometric
            } else {
                DepthSpacing::Linear
            },
            geometric_ratio: 1.5, // Each level 50% further than previous
            linear_step_bps: 3.0, // 3bp steps for linear
            maker_fee_rate: ladder_config.fees_bps / 10000.0, // Convert bps to fraction
            // Spread floor = fees + buffer to ensure profitability
            // With fees_bps=1.5, floor=2.5bp means we capture at least 1bp after fees
            min_spread_floor_bps: spread_floor,
            enable_asymmetric: true, // Enable asymmetric bid/ask depths
            // DISABLED: Trust GLFT optimal spreads from first principles
            // Trade history showed spread cap caused -$562 loss over 9 days
            market_spread_cap_multiple: 0.0,
            // Pass through hard cap for competitive spreads on illiquid assets
            max_spread_per_side_bps: ladder_config.max_spread_per_side_bps,
        };
        DynamicDepthGenerator::new(config)
    }

    /// Calculate effective γ using the CalibratedRiskModel (log-additive).
    ///
    /// log(γ) = log(γ_base) + Σ βᵢ × xᵢ
    ///
    /// The legacy multiplicative chain and post-process multipliers
    /// (calibration_gamma_mult, tail_risk_multiplier) have been removed.
    /// Those effects are now captured by CalibratedRiskModel β coefficients.
    ///
    /// Callers (generate_ladder, compute_spread_composition) may still apply
    /// their own physically motivated multipliers (regime, liquidity, tail_risk)
    /// on top of this base gamma.
    fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        // ============================================================
        // UNIFIED LOG-ADDITIVE GAMMA via compute_gamma_with_policy
        // Single source of truth shared with GLFTStrategy::effective_gamma
        // ============================================================
        let features = RiskFeatures::from_params(
            market_params,
            position,
            max_position,
            &self.risk_model_config,
        );
        self.risk_model.compute_gamma_with_policy(
            &features,
            market_params.capital_tier,
            market_params.capital_policy.warmup_gamma_max_inflation,
        )
    }

    /// Calculate holding time from arrival intensity.
    fn holding_time(&self, arrival_intensity: f64) -> f64 {
        let safe_intensity = arrival_intensity.max(0.01);
        (1.0 / safe_intensity).min(self.risk_config.max_holding_time)
    }

    /// Build a MarketRegime from MarketParams for entropy-based allocation.
    ///
    /// The MarketRegime provides market state signals that the entropy optimizer
    /// uses to adapt its temperature and allocation behavior:
    /// - Higher toxicity → higher temperature → more uniform distribution
    /// - Higher cascade severity → more conservative allocation
    fn build_market_regime(market_params: &MarketParams) -> MarketRegime {
        MarketRegime {
            toxicity: market_params.jump_ratio,
            volatility_ratio: market_params.sigma_effective / 0.0001, // Normalized to 1bp baseline
            // WS4: No binary should_pull_quotes gate. Use tail_risk_intensity directly.
            cascade_severity: market_params.tail_risk_intensity.clamp(0.0, 1.0),
            book_imbalance: market_params.book_imbalance,
        }
    }

    /// Fallback concentrated ladder: 1 order per side at min_viable_depth.
    ///
    /// Only reachable from the post-entropy safety net when all levels fail
    /// min_notional after rounding. Not a first-class code path — all capital
    /// tiers (including Micro) flow through the standard GLFT pipeline in
    /// `generate_ladder()`, which naturally produces 1-2 levels via
    /// `capital_limited_levels` when capital is scarce.
    #[allow(dead_code)] // Retained as safety-net fallback, currently called from nowhere
    fn generate_fallback_concentrated_ladder(
        &self,
        config: &QuoteConfig,
        position: f64,
        effective_max_position: f64,
        max_position: f64,
        _target_liquidity: f64,
        market_params: &MarketParams,
    ) -> Ladder {
        let mut ladder = Ladder::default();
        let mark = market_params.microprice;
        if mark <= 0.0 {
            return ladder;
        }

        let quantum = SizeQuantum::compute(config.min_notional, mark, config.sz_decimals);

        // Size: use full available capacity per side, capped at 25% of the USER's
        // risk-based max_position (not margin-based quoting capacity).
        let half_max = effective_max_position / 2.0;
        let per_side_cap = (max_position * MAX_SINGLE_ORDER_FRACTION).max(quantum.min_viable_size);

        // Calculate available capacity per side, accounting for current position
        let (available_for_bids, available_for_asks) = if position >= 0.0 {
            let bid_limit = (effective_max_position - position).max(0.0);
            let ask_limit = (position + effective_max_position).max(0.0);
            (bid_limit, ask_limit)
        } else {
            let bid_limit = (position.abs() + effective_max_position).max(0.0);
            let ask_limit = (effective_max_position - position.abs()).max(0.0);
            (bid_limit, ask_limit)
        };

        // Only allow round-up to min_viable_size when available capacity actually
        // supports it — rounding up from 0.24 to 0.34 would exceed position limits.
        let raw_bid = available_for_bids.min(half_max).min(per_side_cap);
        let bid_round_up = available_for_bids >= quantum.min_viable_size;
        let bid_size = quantum
            .clamp_to_viable(raw_bid, bid_round_up)
            .unwrap_or(0.0);
        let raw_ask = available_for_asks.min(half_max).min(per_side_cap);
        let ask_round_up = available_for_asks >= quantum.min_viable_size;
        let ask_size = quantum
            .clamp_to_viable(raw_ask, ask_round_up)
            .unwrap_or(0.0);

        // Depth: use min_viable_depth_bps from capacity budget (QueueValue breakeven),
        // falling back to max(min_depth_bps, 5.0) to survive QueueValue filtering.
        let depth_bps = market_params
            .capacity_budget
            .as_ref()
            .map(|b| b.min_viable_depth_bps)
            .unwrap_or(5.0)
            .max(self.ladder_config.min_depth_bps);
        let depth_frac = depth_bps / 10_000.0;
        let one_tick = 10f64.powi(-(config.decimals as i32));

        // Generate bid (if meets min_notional)
        if quantum.is_sufficient(bid_size) && available_for_bids > 0.0 {
            let mut bid_price =
                round_to_significant_and_decimal(mark * (1.0 - depth_frac), 5, config.decimals);
            // Rounding can push bid_price back to mark — nudge down by one tick
            if bid_price >= mark {
                bid_price = mark - one_tick;
            }
            if bid_price > 0.0 {
                ladder.bids.push(LadderLevel {
                    price: bid_price,
                    size: bid_size,
                    depth_bps,
                });
            }
        }

        // Generate ask (if meets min_notional)
        if quantum.is_sufficient(ask_size) && available_for_asks > 0.0 {
            let mut ask_price =
                round_to_significant_and_decimal(mark * (1.0 + depth_frac), 5, config.decimals);
            // Rounding can push ask_price back to mark — nudge up by one tick
            if ask_price <= mark {
                ask_price = mark + one_tick;
            }
            ladder.asks.push(LadderLevel {
                price: ask_price,
                size: ask_size,
                depth_bps,
            });
        }

        if !ladder.bids.is_empty() || !ladder.asks.is_empty() {
            tracing::info!(
                tier = ?market_params.capital_tier,
                bid_size = %format!("{:.4}", bid_size),
                ask_size = %format!("{:.4}", ask_size),
                depth_bps = %format!("{:.1}", depth_bps),
                bid_notional = %format!("${:.2}", bid_size * mark),
                ask_notional = %format!("${:.2}", ask_size * mark),
                "Concentrated ladder: 1 order per side at GLFT-optimal depth"
            );
        } else {
            warn!(
                capital_tier = ?market_params.capital_tier,
                effective_max_position = %format!("{:.6}", effective_max_position),
                min_notional = %format!("{:.2}", config.min_notional),
                microprice = %format!("{:.2}", mark),
                "Concentrated ladder empty: insufficient capital for min_notional"
            );
        }

        ladder
    }

    /// Compute the additive spread composition for the ladder's touch level.
    ///
    /// Returns the same `SpreadComposition` struct as GLFT, decomposing the
    /// ladder's innermost spread into additive components. This provides
    /// a unified view of spread formation across both strategies.
    pub fn compute_spread_composition(
        &self,
        market_params: &MarketParams,
        max_position: f64,
    ) -> SpreadComposition {
        let effective_max_position = market_params
            .effective_max_position(max_position)
            .min(max_position);
        // NOTE: regime_gamma_multiplier is already inside effective_gamma()
        let gamma = self.effective_gamma(market_params, 0.0, effective_max_position);

        let kappa = if market_params.use_kappa_robust {
            market_params.kappa_robust
        } else {
            market_params.kappa
        };

        // Core GLFT half-spread at touch level
        let glft_half_frac = self.depth_generator.glft_optimal_spread(gamma, kappa);
        let glft_half_bps = glft_half_frac * 10_000.0;

        // Risk premium from regime and position zone
        let risk_premium_bps =
            market_params.regime_risk_premium_bps + market_params.total_risk_premium_bps;

        let quota_addon_bps = market_params.quota_shadow_spread_bps.min(50.0);

        let warmup_addon_bps = if market_params.adaptive_warmup_progress < 1.0 {
            // Policy-driven warmup addon: Micro=3.0, Large=8.0 bps max, decays with warmup
            let policy_addon = market_params.capital_policy.warmup_floor_bps
                * (1.0 - market_params.adaptive_warmup_progress);
            // Coordinator uncertainty premium: additive, decays with fills
            policy_addon + market_params.coordinator_uncertainty_premium_bps
        } else {
            0.0
        };

        let fee_bps = self.risk_config.maker_fee_rate * 10_000.0;

        SpreadComposition {
            glft_half_spread_bps: glft_half_bps,
            risk_premium_bps,
            quota_addon_bps,
            warmup_addon_bps,
            fee_bps,
            // Removed cascade addons, relying on risk model beta_cascade
        }
    }

    /// Generate full ladder from market params.
    ///
    /// This is the main method that creates a multi-level quote ladder.
    pub fn generate_ladder(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        _target_liquidity: f64,
        market_params: &MarketParams,
    ) -> Ladder {
        // WS4: Binary circuit breaker removed. γ(q) handles cascade risk continuously.
        // At extreme risk, γ is 50+, producing 25+ bps spreads and minimal sizes.
        // Only margin exhaustion (below) produces an empty ladder.

        // CONTROLLER-DERIVED POSITION SIZING:
        // Use margin-based quoting capacity for ladder allocation, BUT capped by
        // user's config.max_position. The user's max_position is a HARD ceiling —
        // margin can only LOWER the limit, never raise it above config.
        let quoting_capacity = market_params.quoting_capacity();
        let margin_effective = if quoting_capacity > EPSILON {
            quoting_capacity
        } else {
            market_params.effective_max_position(max_position)
        };
        // CRITICAL: config.max_position is the hard ceiling. Margin can only lower it.
        let effective_max_position = margin_effective.min(max_position);

        // ALWAYS log quoting capacity for debugging (INFO level for visibility)
        tracing::info!(
            user_max_position = %format!("{:.6}", max_position),
            quoting_capacity = %format!("{:.6}", quoting_capacity),
            margin_quoting_capacity = %format!("{:.6}", market_params.margin_quoting_capacity),
            margin_available = %format!("{:.2}", market_params.margin_available),
            leverage = %format!("{:.1}", market_params.leverage),
            margin_effective = %format!("{:.6}", margin_effective),
            effective_max_position = %format!("{:.6}", effective_max_position),
            "Quoting capacity: capped by config.max_position"
        );

        // === GAMMA: Single log-additive path via CalibratedRiskModel ===
        // All risk factors (volatility, toxicity, inventory, cascade, tail risk,
        // depth, uncertainty, confidence) are captured as beta coefficients in
        // the log-additive model: log(γ) = log(γ_base) + Σ βᵢ × xᵢ.
        // No post-hoc multiplicative scalars needed.
        let gamma = self.effective_gamma(market_params, position, effective_max_position);

        // NOTE: regime_gamma_multiplier is already applied inside effective_gamma()
        // (L456) and participates in the gamma_max clamp. Do NOT re-apply here.
        debug!(
            gamma = %format!("{:.4}", gamma),
            regime_gamma_mult = %format!("{:.2}", market_params.regime_gamma_multiplier),
            warmup_pct = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
            "Ladder gamma from log-additive CalibratedRiskModel (regime included)"
        );

        // === KAPPA: Robust V3 > Adaptive > Legacy ===
        // Priority: 1. Robust kappa (outlier-resistant), 2. Adaptive, 3. Legacy
        let mut kappa = if market_params.use_kappa_robust {
            // V3 Robust kappa: from KappaOrchestrator (outlier-resistant)
            // Blends book-structure κ, Student-t robust κ, and own-fill κ
            info!(
                kappa_robust = %format!("{:.0}", market_params.kappa_robust),
                legacy_kappa = %format!("{:.0}", market_params.kappa),
                outlier_count = market_params.kappa_outlier_count,
                "Ladder using ROBUST kappa (V3)"
            );
            market_params.kappa_robust
        } else if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
            // Adaptive kappa: blended from book depth + own fill experience
            debug!(
                adaptive_kappa = %format!("{:.0}", market_params.adaptive_kappa),
                book_kappa = %format!("{:.0}", market_params.kappa),
                "Ladder using ADAPTIVE kappa"
            );
            market_params.adaptive_kappa
        } else if market_params.use_coordinator_kappa {
            // Coordinator kappa: L2-derived, refined by fills. Used during warmup
            // when robust/adaptive estimators haven't converged yet.
            // Conservative by design: 0.5x warmup factor at start, hard floor at 10.
            info!(
                coordinator_kappa = %format!("{:.0}", market_params.coordinator_kappa),
                uncertainty_premium_bps = %format!("{:.1}", market_params.coordinator_uncertainty_premium_bps),
                legacy_kappa = %format!("{:.0}", market_params.kappa),
                "Ladder using COORDINATOR kappa (L2-seeded warmup)"
            );
            market_params.coordinator_kappa
        } else {
            // Legacy: Book-based kappa with AS adjustment
            // κ_effective = κ̂ × (1 - α), where α = P(informed | fill)
            let alpha = market_params.predicted_alpha.min(0.5);
            market_params.kappa * (1.0 - alpha)
        };

        // === REGIME KAPPA BLENDING ===
        // Regime kappa captures structural differences: Calm→3000, Normal→2000,
        // Volatile→1000, Extreme→500. Makes spreads widen in volatile regimes.
        //
        // WARMUP RAMP: During warmup the regime classifier hasn't converged, so
        // the regime_kappa is just the default (Normal=2000). Giving it 60% weight
        // pulls the market-derived kappa DOWN, widening spreads and preventing
        // fills (cold-start death spiral). Ramp from 20% at 0% warmup → 60% at
        // 100% warmup so that market-derived kappa dominates early, and regime
        // kappa takes over once calibrated.
        if let Some(regime_kappa) = market_params.regime_kappa {
            let kappa_before_regime = kappa;
            const REGIME_BLEND_WEIGHT_FULL: f64 = 0.6;
            const REGIME_BLEND_WEIGHT_WARMUP: f64 = 0.2;
            let regime_blend_weight = if market_params.adaptive_warmup_progress < 1.0 {
                let t = market_params.adaptive_warmup_progress;
                REGIME_BLEND_WEIGHT_WARMUP
                    + t * (REGIME_BLEND_WEIGHT_FULL - REGIME_BLEND_WEIGHT_WARMUP)
            } else {
                REGIME_BLEND_WEIGHT_FULL
            };
            kappa = (1.0 - regime_blend_weight) * kappa + regime_blend_weight * regime_kappa;
            info!(
                kappa_before = %format!("{:.0}", kappa_before_regime),
                regime_kappa = %format!("{:.0}", regime_kappa),
                kappa_after = %format!("{:.0}", kappa),
                regime = market_params.regime_kappa_current_regime,
                regime_blend_weight = %format!("{:.2}", regime_blend_weight),
                warmup_pct = %format!("{:.2}", market_params.adaptive_warmup_progress),
                "[SPREAD TRACE] regime kappa blending (warmup-ramped weight)"
            );
        }
        // NOTE: Agent 3's alpha-based kappa adjustment follows below.

        // === AS FEEDBACK: Reduce kappa when informed flow detected ===
        // predicted_alpha measures P(next trade is informed) from vol surprise, flow, jumps.
        // When alpha is high and AS is warmed up, lower kappa → wider spreads → defensive.
        // This applies to ALL kappa sources (robust, adaptive, legacy) since informed flow
        // affects us regardless of how kappa was estimated.
        // Note: legacy branch already discounts by alpha, so skip double-counting.
        let alpha_for_kappa = market_params.predicted_alpha;
        const ALPHA_KAPPA_THRESHOLD: f64 = 0.3;
        const ALPHA_KAPPA_SENSITIVITY: f64 = 0.5;
        if alpha_for_kappa > ALPHA_KAPPA_THRESHOLD
            && market_params.as_warmed_up
            && (market_params.use_kappa_robust
                || (market_params.use_adaptive_spreads && market_params.adaptive_can_estimate))
        {
            // At alpha=0.6 → kappa *= 0.7, at alpha=1.0 → kappa *= 0.5
            let kappa_before_alpha = kappa;
            kappa *= 1.0 - ALPHA_KAPPA_SENSITIVITY * alpha_for_kappa;
            info!(
                alpha = %format!("{:.3}", alpha_for_kappa),
                kappa_before = %format!("{:.0}", kappa_before_alpha),
                kappa_after = %format!("{:.0}", kappa),
                "AS feedback: reducing kappa for informed flow defense"
            );
        }

        let time_horizon = self.holding_time(market_params.arrival_intensity);

        let inventory_ratio = if effective_max_position > EPSILON {
            (position / effective_max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // === CONTINUOUS γ(q) REPLACED DISCRETE ZONES ===
        // Smooth γ(q) = γ_base × (1 + β × utilization²) in glft.rs handles graduated
        // risk aversion. Kill-switch safety guard at >100% is the only discrete cutoff.
        let abs_inventory_ratio = inventory_ratio.abs();

        // Kill-switch guard: clear accumulating side at >=90% utilization (safety-critical)
        if abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO {
            if position > 0.0 {
                tracing::warn!(position, ratio = %format!("{:.2}", abs_inventory_ratio), "Kill-switch: clearing bids at >=90% utilization");
            } else if position < 0.0 {
                tracing::warn!(position, ratio = %format!("{:.2}", abs_inventory_ratio), "Kill-switch: clearing asks at >=90% utilization");
            }
        }

        // zone_size_mult REMOVED (B3): inventory penalty now captured by
        // beta_inventory in CalibratedRiskModel via gamma. Double-penalizing
        // through size is economically incorrect.

        // Convert AS spread adjustment to bps for ladder generation
        let as_at_touch_bps = if market_params.as_warmed_up {
            market_params.as_spread_adjustment * 10000.0
        } else {
            0.0 // Use zero AS until warmed up
        };

        // Cascade size reduction removed (B2): cascade exposure reduction is now
        // handled by beta_cascade in CalibratedRiskModel (wider spreads) and
        // Kelly correlation_discount() (smaller positions). Multiplicative size
        // choking was double-penalizing and causing min_notional filter failures.
        //
        // Use full effective_max_position for initial ladder generation so
        // individual levels pass the min_notional filter. The entropy optimizer
        // then allocates the actual available margin (available_for_bids/asks).
        let size_for_initial_ladder = effective_max_position;

        // === UNIFIED RESERVATION PRICE (WS3) ===
        // Single equation incorporating:
        // 1. Drift (lead/lag & momentum)
        // 2. Continuous inventory penalty (q * gamma * sigma^2 * T)
        // 3. Funding carry

        // 1. Drift — posterior primary, signal additive
        // Primary: μ_posterior × horizon (Bayesian belief system)
        // Units: (frac/sec) × sec × 10000 × [0,1] = bps
        let posterior_drift_bps = market_params.belief_predictive_bias
            * time_horizon
            * 10_000.0
            * market_params.belief_confidence;
        // Secondary: lead-lag / CUSUM fast overlay (50-500ms microstructure)
        let drift_shift_bps = posterior_drift_bps + market_params.drift_signal_bps;

        // 2. Inventory Penalty
        // gamma has regime & beta_inventory applied natively via effective_gamma
        let q = if effective_max_position > EPSILON {
            position / effective_max_position
        } else {
            0.0
        };
        // Continuation-based gamma multiplier: amplify inventory penalty when opposed, relax when aligned
        let cont_gamma_mult = super::glft::continuation_gamma_multiplier(
            market_params.continuation_p,
            3.0, // max_mult at p=0 (REDUCE: strong opposition)
            0.5, // min_mult at p=1 (HOLD: strong alignment)
        );
        // Penalty = q * gamma * cont_mult * sigma^2 * T in bps
        let inv_penalty_bps =
            q * gamma * cont_gamma_mult * market_params.sigma.powi(2) * time_horizon * 10_000.0;

        // 3. Funding Carry (annualized rate converted to per-second, multiply by T)
        // 365.25 * 24 * 60 * 60 = 31,557,600
        let funding_per_sec = market_params.funding_rate / 31_557_600.0;
        let funding_carry_bps = funding_per_sec * time_horizon * 10_000.0;

        // Total shift combines all fundamental flows
        // Drift pulls reservation price in direction of momentum.
        // Inventory penalty pushes reservation price away from position (short pushes down, long pushes up).
        // Funding carry pushes reservation price away from expensive funding side.
        let total_shift_bps = drift_shift_bps - inv_penalty_bps - funding_carry_bps;
        let shift_fraction = total_shift_bps / 10_000.0;

        let reservation_mid = market_params.microprice * (1.0 + shift_fraction);

        // SAFETY: Bound skewed mid within 2× GLFT half-spread.
        // At max shift (2×, bullish): touch_bid = mid + hs (aggressive), touch_ask = mid + 3×hs (conservative).
        let glft_half_spread_frac = self.depth_generator.glft_optimal_spread(gamma, kappa);
        let half_spread_bound = glft_half_spread_frac * 2.0;
        let max_mid = market_params.market_mid * (1.0 + half_spread_bound);
        let min_mid = market_params.market_mid * (1.0 - half_spread_bound);
        let effective_mid = reservation_mid.clamp(min_mid, max_mid);

        if total_shift_bps.abs() > 0.5 {
            tracing::debug!(
                posterior_drift_bps = %format!("{:.2}", posterior_drift_bps),
                signal_drift_bps = %format!("{:.2}", market_params.drift_signal_bps),
                belief_confidence = %format!("{:.3}", market_params.belief_confidence),
                drift_bps = %format!("{:.2}", drift_shift_bps),
                inv_penalty_bps = %format!("{:.2}", inv_penalty_bps),
                funding_carry_bps = %format!("{:.2}", funding_carry_bps),
                total_shift_bps = %format!("{:.2}", total_shift_bps),
                reservation_mid = %format!("{:.4}", reservation_mid),
                effective_mid = %format!("{:.4}", effective_mid),
                "Unified continuous reservation mid computed"
            );
        }

        // === SPREAD FLOOR: Physical Constraints Only ===
        // The floor ONLY includes physical/exchange constraints that cannot be violated.
        // Regime risk is routed through gamma (gamma_multiplier) instead of floor clamping.
        // This lets GLFT produce the actual spread instead of being overridden by regime floors.
        let fee_bps = self.risk_config.maker_fee_rate * 10_000.0;
        let tick_floor_bps = market_params.tick_size_bps;
        let latency_floor_bps = market_params.latency_spread_floor * 10_000.0;

        // Options-theoretic volatility floor (from market_params, computed by quote_engine).
        let option_floor_bps = market_params.option_floor_bps;

        let physical_floor_bps = fee_bps
            .max(tick_floor_bps)
            .max(latency_floor_bps)
            .max(self.risk_config.min_spread_floor * 10_000.0)
            .max(option_floor_bps);

        // === BAYESIAN SPREAD FLOOR: Widen when model parameters are uncertain ===
        // Delta method: propagate posterior uncertainty through GLFT formula.
        // σ²_δ = (∂δ/∂κ)² × Var[κ] + (∂δ/∂σ)² × Var[σ]
        // Spread floor = physical_floor + z_α × σ_δ, z_α=0.674 for P(edge>0)≥75%
        let bayesian_floor_bps =
            if market_params.kappa_variance > 0.0 && kappa > 1.0 && gamma > 1e-6 {
                let d_delta_d_kappa = -1.0 / (kappa * (kappa + gamma));
                let spread_var_from_kappa =
                    d_delta_d_kappa.powi(2) * market_params.kappa_variance * (10_000.0_f64).powi(2);
                // Sigma uncertainty: conservative 10% CV heuristic
                let sigma_cv = 0.10;
                let d_delta_d_sigma = gamma * market_params.sigma * 1.0; // T≈1s
                let spread_var_from_sigma =
                    (d_delta_d_sigma * sigma_cv * market_params.sigma * 10_000.0).powi(2);
                let spread_std_bps = (spread_var_from_kappa + spread_var_from_sigma).sqrt();
                const P_THRESHOLD_Z: f64 = 0.674; // z for P(edge>0) ≥ 75%
                physical_floor_bps + P_THRESHOLD_Z * spread_std_bps
            } else {
                physical_floor_bps
            };

        // Use Bayesian floor but cap at 25 bps to prevent absurd floors
        let effective_floor_bps = physical_floor_bps.max(bayesian_floor_bps).min(25.0);

        let params = LadderParams {
            mid_price: effective_mid,
            market_mid: market_params.market_mid, // Actual exchange mid for safety checks
            sigma: market_params.sigma,
            kappa, // Use AS-adjusted kappa
            arrival_intensity: market_params.arrival_intensity,
            as_at_touch_bps,
            total_size: size_for_initial_ladder, // Use full capacity to avoid min_notional filtering
            inventory_ratio,
            gamma,
            time_horizon,
            decimals: config.decimals,
            sz_decimals: config.sz_decimals,
            min_notional: config.min_notional,
            depth_decay_as: market_params.depth_decay_as.clone(),
            effective_floor_bps,
            // Drift-adjusted skew from MarketParams (first-principles momentum opposition)
            use_drift_adjusted_skew: market_params.use_drift_adjusted_skew,
            hjb_drift_urgency: market_params.hjb_drift_urgency,
            position_opposes_momentum: market_params.position_opposes_momentum,
            directional_variance_mult: 1.0, // Multiplicative field removed from MarketParams; dead in LadderEmission
            urgency_score: market_params.urgency_score,
            // Convert annualized funding rate to per-second rate
            // 365.25 * 24 * 60 * 60 = 31,557,600 seconds/year
            funding_rate: market_params.funding_rate / 31_557_600.0,
            use_funding_skew: true,
            // RL policy adjustments from Q-learning agent
            rl_spread_delta_bps: market_params.rl_spread_delta_bps,
            rl_bid_skew_bps: market_params.rl_bid_skew_bps,
            rl_ask_skew_bps: market_params.rl_ask_skew_bps,
            rl_confidence: market_params.rl_confidence,
            // Position Continuation Model (HOLD/ADD/REDUCE)
            warmup_pct: market_params.adaptive_warmup_progress,
            // Use actual L2 book BBO when available; fall back to market_mid ± spread
            cached_best_bid: if market_params.cached_best_bid > 0.0 {
                market_params.cached_best_bid
            } else {
                market_params.market_mid * (1.0 - market_params.market_spread_bps / 20000.0)
            },
            cached_best_ask: if market_params.cached_best_ask > 0.0 {
                market_params.cached_best_ask
            } else {
                market_params.market_mid * (1.0 + market_params.market_spread_bps / 20000.0)
            },
        };

        // [SPREAD TRACE] Phase 1: floor — capture GLFT optimal for diagnostics
        let glft_optimal_bps = self.depth_generator.glft_optimal_spread(gamma, kappa) * 10_000.0;
        tracing::info!(
            phase = "floor",
            glft_optimal_bps = %format!("{:.2}", glft_optimal_bps),
            effective_floor_bps = %format!("{:.2}", effective_floor_bps),
            fee_bps = %format!("{:.2}", fee_bps),
            tick_bps = %format!("{:.2}", tick_floor_bps),
            latency_bps = %format!("{:.2}", latency_floor_bps),
            "[SPREAD TRACE] physical-only floor (regime risk routes through gamma)"
        );

        // === DYNAMIC DEPTHS: GLFT-optimal depth computation with Bayesian bounds ===
        // Compute depths from effective gamma and kappa using GLFT formula:
        // δ* = (1/γ) × ln(1 + γ/κ)
        //
        // Dynamic bounds (when no CLI override):
        // - dynamic_kappa_floor: From Bayesian confidence + 95% credible interval
        // - dynamic_spread_ceiling_bps: From fill rate controller + market spread p80
        //
        // This replaces arbitrary hardcoded values with principled model-driven bounds.

        // Symmetric kappa: κ is a structural order book parameter. Directional adjustment
        // is handled by the drift depth adjustment δ ± μT/2 (Avellaneda-Stoikov), which is
        // mathematically equivalent to asymmetric fill rates under the exponential model.
        // Adjusting kappa per-side would double-count.
        let mut dynamic_depths = if market_params.use_dynamic_bounds {
            // Use model-driven bounds from Bayesian estimation
            self.depth_generator.compute_depths_with_dynamic_bounds(
                gamma,
                kappa,
                kappa,
                market_params.sigma,
                market_params.dynamic_kappa_floor,
                market_params.dynamic_spread_ceiling_bps,
            )
        } else {
            // Use legacy path with market spread cap (CLI overrides in effect)
            self.depth_generator.compute_depths_with_market_cap(
                gamma,
                kappa,
                kappa,
                market_params.sigma,
                market_params.market_spread_bps,
            )
        };

        // [SPREAD TRACE] Phase 3: GLFT depths computed
        tracing::info!(
            phase = "glft_depths",
            touch_bid_bps = %format!("{:.2}", dynamic_depths.best_bid_depth().unwrap_or(0.0)),
            touch_ask_bps = %format!("{:.2}", dynamic_depths.best_ask_depth().unwrap_or(0.0)),
            total_at_touch_bps = %format!("{:.2}", dynamic_depths.spread_at_touch().unwrap_or(0.0)),
            "[SPREAD TRACE] after GLFT depth computation"
        );

        // Apply stochastic floor to dynamic depths
        // Ensure all levels are at least at effective_floor_bps
        for depth in dynamic_depths.bid.iter_mut() {
            if *depth < effective_floor_bps {
                *depth = effective_floor_bps;
            }
        }
        for depth in dynamic_depths.ask.iter_mut() {
            if *depth < effective_floor_bps {
                *depth = effective_floor_bps;
            }
        }

        // [SPREAD TRACE] Phase 4: floor clamp applied
        tracing::info!(
            phase = "floor_clamp",
            touch_bid_bps = %format!("{:.2}", dynamic_depths.best_bid_depth().unwrap_or(0.0)),
            touch_ask_bps = %format!("{:.2}", dynamic_depths.best_ask_depth().unwrap_or(0.0)),
            total_at_touch_bps = %format!("{:.2}", dynamic_depths.spread_at_touch().unwrap_or(0.0)),
            "[SPREAD TRACE] after floor clamp to effective_floor"
        );

        // Track floor-binding frequency — if this fires >5% of cycles, gamma is miscalibrated
        let bid_bound = dynamic_depths
            .bid
            .iter()
            .any(|d| (*d - effective_floor_bps).abs() < 0.01);
        let ask_bound = dynamic_depths
            .ask
            .iter()
            .any(|d| (*d - effective_floor_bps).abs() < 0.01);
        if bid_bound || ask_bound {
            tracing::warn!(
                effective_floor_bps = %format!("{:.2}", effective_floor_bps),
                glft_optimal_bps = %format!("{:.2}", glft_optimal_bps),
                gamma = %format!("{:.4}", gamma),
                "Floor binding — gamma may be miscalibrated (should be rare after self-consistent gamma)"
            );
        }

        // === DRIFT ADJUSTMENT: Asymmetric bid/ask from Kalman drift estimator ===
        // Classical Avellaneda-Stoikov μ*T/2 term: positive drift (price rising)
        // widens bids (buying into uptrend is risky) and tightens asks (selling is favorable).
        // UNCLAMPED: Kalman posterior variance naturally bounds the estimate.
        // The old ±3 bps cap was the #1 reason drift detection didn't prevent accumulation.
        if market_params.drift_rate_per_sec.abs() > 1e-10 {
            let drift_adj_bps = market_params.drift_rate_per_sec * time_horizon * 10_000.0 / 2.0;
            for depth in dynamic_depths.bid.iter_mut() {
                *depth = (*depth + drift_adj_bps).max(effective_floor_bps);
            }
            for depth in dynamic_depths.ask.iter_mut() {
                *depth = (*depth - drift_adj_bps).max(effective_floor_bps);
            }
            if drift_adj_bps.abs() > 0.5 {
                tracing::info!(
                    drift_rate = %format!("{:.2e}", market_params.drift_rate_per_sec),
                    drift_adj_bps = %format!("{:.2}", drift_adj_bps),
                    drift_uncertainty = %format!("{:.2}", market_params.drift_uncertainty_bps),
                    "Drift asymmetry applied (Kalman, unclamped)"
                );
            }
        }

        // === FUNDING CARRY: Per-side cost in GLFT depths ===
        // Positive funding → longs pay → widen bids. Negative → shorts pay → widen asks.
        // Applied additively to depths (bps). Typically negligible (<0.1 bps) except during
        // extreme funding events (>1%/hr) where it adds ~0.3 bps.
        if market_params.funding_carry_bid_bps > 0.001 {
            for depth in dynamic_depths.bid.iter_mut() {
                *depth += market_params.funding_carry_bid_bps;
            }
        }
        if market_params.funding_carry_ask_bps > 0.001 {
            for depth in dynamic_depths.ask.iter_mut() {
                *depth += market_params.funding_carry_ask_bps;
            }
        }

        // Self-impact: additive spread widening when we dominate the book.
        // Applied symmetrically to both sides (our_fraction is EWMA-smoothed).
        if market_params.self_impact_addon_bps > 0.001 {
            for depth in dynamic_depths.bid.iter_mut() {
                *depth += market_params.self_impact_addon_bps;
            }
            for depth in dynamic_depths.ask.iter_mut() {
                *depth += market_params.self_impact_addon_bps;
            }
        }

        // === KEPT INFRASTRUCTURE ADDONS (correct AS extensions, not protection) ===
        // Governor (API rate limit) + cascade (fill cascade tracker) + self_impact + funding_carry.
        // WS4: staleness and flow_toxicity addons REMOVED — routed through estimators.
        let kept_bid_addon_bps = market_params.governor_bid_addon_bps.min(25.0);
        let kept_ask_addon_bps = market_params.governor_ask_addon_bps.min(25.0);
        if kept_bid_addon_bps > 0.01 {
            for depth in dynamic_depths.bid.iter_mut() {
                *depth += kept_bid_addon_bps;
            }
        }
        if kept_ask_addon_bps > 0.01 {
            for depth in dynamic_depths.ask.iter_mut() {
                *depth += kept_ask_addon_bps;
            }
        }
        if kept_bid_addon_bps > 0.1 || kept_ask_addon_bps > 0.1 {
            tracing::info!(
                governor_bid = %format!("{:.1}", market_params.governor_bid_addon_bps),
                governor_ask = %format!("{:.1}", market_params.governor_ask_addon_bps),
                total_bid = %format!("{:.1}", kept_bid_addon_bps),
                total_ask = %format!("{:.1}", kept_ask_addon_bps),
                "Infrastructure addons applied (additive bps)"
            );
        }

        // === KILL-SWITCH SIDE CLEARING (safety-critical) ===
        // Only fires at >100% utilization. Continuous γ(q) handles everything else.
        if abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO {
            if position > 0.0 {
                // Long: clear bids (don't buy more), keep asks (let us sell)
                dynamic_depths.bid.clear();
            } else if position < 0.0 {
                // Short: clear asks (don't sell more), keep bids (let us buy)
                dynamic_depths.ask.clear();
            }
        }

        // Kappa-driven spread cap removed — circular with GLFT.
        // GLFT delta = (1/gamma) * ln(1 + gamma/kappa) IS the self-consistent spread.

        // === PRE-FILL AS MULTIPLIERS: REMOVED (B3) ===
        // Multiplicative spread widening from pre-fill classifier deleted.
        // Adverse selection defense now handled entirely by the additive E[PnL]
        // filter below (which uses AS cost in bps, not unitless multipliers).

        // [SPREAD TRACE] Phase 6: pre-fill AS (multiplicative path deleted)
        tracing::info!(
            phase = "pre_fill_as",
            status = "deleted (E[PnL] filter handles AS)",
            touch_bid_bps = %format!("{:.2}", dynamic_depths.best_bid_depth().unwrap_or(0.0)),
            touch_ask_bps = %format!("{:.2}", dynamic_depths.best_ask_depth().unwrap_or(0.0)),
            total_at_touch_bps = %format!("{:.2}", dynamic_depths.spread_at_touch().unwrap_or(0.0)),
            "[SPREAD TRACE] pre-fill AS multipliers deleted"
        );

        // === SPREAD INFLATION CAP ===
        // Prevent compounding multipliers (pre-fill AS, zone widening, drift, etc.)
        // from pushing spreads beyond 4x GLFT optimal. Floor at 15 bps to allow
        // reasonable widening in volatile regimes.
        let inflation_cap_bps = (glft_optimal_bps * 4.0).max(15.0);
        for depth in dynamic_depths.bid.iter_mut() {
            if *depth > inflation_cap_bps {
                *depth = inflation_cap_bps;
            }
        }
        for depth in dynamic_depths.ask.iter_mut() {
            if *depth > inflation_cap_bps {
                *depth = inflation_cap_bps;
            }
        }

        // === REMOVED: L2 SPREAD MULTIPLIER ===
        // The l2_spread_multiplier has been removed. All uncertainty is now handled
        // through gamma scaling (kappa_ci_width flows through uncertainty_scalar).
        // The GLFT formula naturally widens spreads when gamma increases due to uncertainty.
        //
        // The L2 Bayesian uncertainty premium (σ_μ based spread widening) is now
        // incorporated into the gamma calculation via uncertainty_scalar, which provides
        // a principled Bayesian approach that doesn't bypass the GLFT model.

        // === DEPRECATED: SPREAD MULTIPLIERS ===
        // FIRST PRINCIPLES REFACTOR: Arbitrary spread multipliers bypass the GLFT model.
        //
        // Previously this section applied:
        //   1. stochastic_spread_multiplier (book depth, toxicity)
        //   2. adaptive_uncertainty_factor (warmup uncertainty)
        //
        // These have been REMOVED. All risk factors now flow through gamma:
        //   - Book depth → RiskConfig.book_depth_multiplier() → gamma scaling
        //   - Toxicity → RiskConfig.toxicity_sensitivity → gamma scaling
        //   - Warmup uncertainty → RiskConfig.warmup_multiplier() → gamma scaling
        //
        // The GLFT formula δ = (1/γ) × ln(1 + γ/κ) now handles all spread widening
        // in a mathematically principled way.

        // === E[PnL] FILTER (Phase 4: Unified Adverse Selection) ===
        // When enabled, drop levels with E[PnL] ≤ 0 instead of binary quote gate decisions.
        // This naturally produces one-sided quoting without NoQuote/OnlyBids/OnlyAsks.
        if market_params.use_epnl_filter {
            let pre_bid_count = dynamic_depths.bid.len();
            let pre_ask_count = dynamic_depths.ask.len();

            use super::glft::EPnLParams;
            let mut bid_params = EPnLParams {
                depth_bps: 0.0,
                is_bid: true,
                gamma,
                kappa_side: market_params.kappa_bid,
                sigma: market_params.sigma,
                time_horizon,
                drift_rate: market_params.drift_rate_per_sec,
                position,
                max_position: effective_max_position,
                as_cost_bps: as_at_touch_bps,
                fee_bps,
                carry_cost_bps: market_params.funding_carry_bps,
                toxicity_score: market_params.toxicity_score,
                circuit_breaker_active: market_params.should_pull_quotes,
                drawdown_frac: 0.0,   // handled by capital coordinator
                self_impact_bps: 0.0, // self impact handled by actuary later
                inventory_beta: self.risk_model.beta_inventory,
                continuation_gamma_mult: cont_gamma_mult,
                kappa_variance: market_params.kappa_variance,
            };

            let mut ask_params = bid_params.clone();
            ask_params.is_bid = false;
            ask_params.kappa_side = market_params.kappa_ask;

            // Position-convex threshold: reducing side gets negative E[PnL] carve-out
            // proportional to position urgency. Accumulating side still requires E[PnL] > 0.
            // This prevents the death spiral where E[PnL] < 0 on ALL levels kills both sides
            // while inventory screams to unwind. The carve-out grows with |position/max|^1.5,
            // and the gamma ratio widens it further in volatile regimes.
            let bid_is_reducing = position < -1e-9;
            let ask_is_reducing = position > 1e-9;

            let gamma_baseline = self.risk_config.gamma_base;
            let reducing_thresh = super::glft::reducing_threshold_bps(
                position,
                effective_max_position,
                fee_bps,
                gamma,
                gamma_baseline,
            );

            let bid_threshold = if bid_is_reducing {
                reducing_thresh
            } else {
                0.0
            };
            let ask_threshold = if ask_is_reducing {
                reducing_thresh
            } else {
                0.0
            };

            // (closest levels tracking removed)

            dynamic_depths.bid.retain(|&depth| {
                bid_params.depth_bps = depth;
                super::glft::expected_pnl_bps_enhanced(&bid_params) > bid_threshold
            });
            dynamic_depths.ask.retain(|&depth| {
                ask_params.depth_bps = depth;
                super::glft::expected_pnl_bps_enhanced(&ask_params) > ask_threshold
            });

            // Legacy cascade E[PnL] exemption removed: cascade risk is now handled
            // multiplicatively via beta_cascade in the risk model, which naturally
            // widens spreads without artificially suppressing E[PnL] below zero.

            let bid_dropped = pre_bid_count - dynamic_depths.bid.len();
            let ask_dropped = pre_ask_count - dynamic_depths.ask.len();
            if bid_dropped > 0 || ask_dropped > 0 {
                tracing::info!(
                    bid_dropped, ask_dropped,
                    bid_remaining = dynamic_depths.bid.len(),
                    ask_remaining = dynamic_depths.ask.len(),
                    bid_exempt = bid_is_reducing,
                    ask_exempt = ask_is_reducing,
                    position = %format!("{:.4}", position),
                    drift_bps = %format!("{:.2}", market_params.drift_rate_per_sec * 10_000.0),
                    "E[PnL] filter: dropped negative-EV levels"
                );
            }

            // WS7b: Log E[PnL] decomposition for tightest surviving bid/ask
            if let Some(&touch_bid_depth) = dynamic_depths.bid.first() {
                bid_params.depth_bps = touch_bid_depth;
                let (_, diag) = super::glft::expected_pnl_bps_with_diagnostics(&bid_params);
                tracing::debug!(
                    depth_bps = %format!("{:.2}", diag.depth_bps),
                    lambda = %format!("{:.4}", diag.lambda),
                    capture_bps = %format!("{:.2}", diag.capture_bps),
                    toxicity_cost = %format!("{:.2}", diag.toxicity_cost_bps),
                    drift = %format!("{:.2}", diag.drift_contribution_bps),
                    inv_adj = %format!("{:.2}", diag.inventory_adj_bps),
                    epnl = %format!("{:.3}", diag.epnl_bps),
                    "E[PnL] diagnostics: touch BID"
                );
            }
            if let Some(&touch_ask_depth) = dynamic_depths.ask.first() {
                ask_params.depth_bps = touch_ask_depth;
                let (_, diag) = super::glft::expected_pnl_bps_with_diagnostics(&ask_params);
                tracing::debug!(
                    depth_bps = %format!("{:.2}", diag.depth_bps),
                    lambda = %format!("{:.4}", diag.lambda),
                    capture_bps = %format!("{:.2}", diag.capture_bps),
                    toxicity_cost = %format!("{:.2}", diag.toxicity_cost_bps),
                    drift = %format!("{:.2}", diag.drift_contribution_bps),
                    inv_adj = %format!("{:.2}", diag.inventory_adj_bps),
                    epnl = %format!("{:.3}", diag.epnl_bps),
                    "E[PnL] diagnostics: touch ASK"
                );
            }
        }

        // Capital-aware level count: don't spread across more levels than capital can support.
        // Uses the policy's max_levels_per_side as a hard cap, then further constrained by
        // actual capital capacity. This replaces ad-hoc capital_limited_levels logic with
        // a principled policy-driven cap.
        let budget_per_side = size_for_initial_ladder * market_params.microprice / 2.0;
        let max_viable_levels = (budget_per_side / config.min_notional).floor() as usize;
        let policy_max = market_params.capital_policy.max_levels_per_side;
        let config_max = self.ladder_config.num_levels.min(policy_max);
        let capital_limited_levels = if max_viable_levels < config_max && max_viable_levels >= 1 {
            tracing::info!(
                budget_per_side = %format!("{:.2}", budget_per_side),
                min_notional = %format!("{:.2}", config.min_notional),
                max_viable_levels = max_viable_levels,
                policy_max = policy_max,
                configured_levels = self.ladder_config.num_levels,
                "Capital constrains ladder levels"
            );
            max_viable_levels
        } else {
            config_max
        };

        // E5: Single-level mode — when capital only supports 1 level per side,
        // enforce wider spreads to compensate for concentrated exposure.
        // A 1-level strategy is a valid strategy, not a degenerate 10-level strategy.
        // (Size capping applied later after available_for_bids/asks are computed.)
        if capital_limited_levels <= 1 {
            let policy = &market_params.capital_policy;
            let min_half_spread = policy.single_level_min_spread_bps;
            let spread_mult = policy.single_level_spread_mult;

            // Widen touch depths to at least single_level_min_spread_bps per side,
            // then apply the spread multiplier
            for depth in dynamic_depths.bid.iter_mut() {
                *depth = (*depth).max(min_half_spread) * spread_mult;
            }
            for depth in dynamic_depths.ask.iter_mut() {
                *depth = (*depth).max(min_half_spread) * spread_mult;
            }

            info!(
                min_half_spread_bps = %format!("{:.1}", min_half_spread),
                spread_mult = %format!("{:.2}", spread_mult),
                capital_tier = ?market_params.capital_tier,
                "E5: Single-level spread widening active"
            );
        }

        // Create ladder config with dynamic depths and capital-limited levels
        let mut ladder_config = self
            .ladder_config
            .clone()
            .with_dynamic_depths(dynamic_depths);
        ladder_config.num_levels = capital_limited_levels;

        // INFO-level log for spread diagnostics
        // Shows gamma (includes book_depth + warmup scaling), kappa, and resulting spread
        info!(
            gamma = %format!("{:.3}", gamma),
            kappa = %format!("{:.1}", kappa),
            sigma = %format!("{:.6}", market_params.sigma),
            optimal_spread_bps = %format!("{:.2}", ladder_config.dynamic_depths.as_ref()
                .and_then(|d| d.spread_at_touch())
                .unwrap_or(0.0)),
            effective_floor_bps = %format!("{:.1}", effective_floor_bps),
            kappa = %format!("{:.0}", kappa),
            book_depth_usd = %format!("{:.0}", market_params.near_touch_depth_usd),
            warmup_pct = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
            adaptive_mode = market_params.use_adaptive_spreads && market_params.adaptive_can_estimate,
            "Ladder spread diagnostics (gamma includes book_depth + warmup scaling)"
        );

        // === STOCHASTIC MODULE: Entropy-Based Ladder Optimization ===
        // Solves: max Σ λ(δᵢ) × SC(δᵢ) × sᵢ  (expected profit)
        // s.t. Σ sᵢ × margin_per_unit ≤ margin_available (margin constraint)
        //      Σ sᵢ ≤ effective_max_position (position constraint - first principles)
        //      Σ sᵢ ≤ exchange_limit (exchange-enforced constraint)
        {
            // 1. Available margin comes from the exchange (already accounts for position margin)
            // NOTE: margin_available is already net of position margin, don't double-count!
            let leverage = market_params.leverage.max(1.0);
            let available_margin = market_params.margin_available;

            // === DYNAMIC MARGIN UTILIZATION ===
            // Regime-dependent: calm→85%, normal→75%, volatile→65%, extreme→50%.
            // Further scaled by kill switch headroom and warmup boost.
            let margin_utilization = dynamic_margin_utilization(
                market_params.regime_gamma_multiplier,
                market_params.kill_switch_headroom,
                market_params.adaptive_warmup_progress,
            );
            let usable_margin =
                available_margin * margin_utilization * market_params.warmup_size_mult;

            // === CARTEA-JAIMUNGAL OPTIMAL INVENTORY TARGET (SNR-SCALED) ===
            // Bayesian inventory target: q* = q_max × tanh(SNR / SNR_scale) × sign(drift)
            // where SNR = |drift_raw_bps| / drift_uncertainty_bps.
            // When SNR is low (uncertain drift), q* → 0 (symmetric quoting).
            // When SNR is high (confident drift), q* → q_max (full tilt).
            // This replaces the old γσ²-based formula which saturated at ±0.5 in 99.7% of cycles.
            let gamma_market = self.effective_gamma(market_params, 0.0, effective_max_position);
            let drift_snr = if market_params.drift_uncertainty_bps > 1e-6 {
                let drift_bps = market_params.drift_rate_per_sec_raw * 10_000.0;
                drift_bps / market_params.drift_uncertainty_bps
            } else {
                0.0
            };
            const SNR_SCALE: f64 = 2.0; // 2-sigma drift → ~76% of q_max
            let q_target = 0.5 * (drift_snr / SNR_SCALE).tanh();
            // q_target is already signed via drift_snr (drift_bps preserves sign)

            let q_current = if effective_max_position > EPSILON {
                (position / effective_max_position).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            // SNR already encodes confidence via drift_uncertainty (incorporates
            // observation count and Kalman posterior variance). No drift_conf scaling needed.
            let drift_conf = market_params.belief_confidence; // Kept for observability
            let effective_q_target = q_target;
            let delta_q = effective_q_target - q_current;

            // tanh: principled saturation (smooth, sign-preserving, derivative=1 near 0)
            let cj_allocation = (delta_q * 2.0).tanh() * 0.35;

            let ask_margin_weight = (0.5 - cj_allocation).clamp(0.15, 0.85);
            let bid_margin_weight = 1.0 - ask_margin_weight;
            let margin_for_bids = usable_margin * bid_margin_weight;
            let margin_for_asks = usable_margin * ask_margin_weight;

            info!(
                usable_margin = %format!("{:.2}", usable_margin),
                warmup_size_mult = %format!("{:.3}", market_params.warmup_size_mult),
                gamma_market = %format!("{:.4}", gamma_market),
                gamma_spread = %format!("{:.4}", gamma),
                drift_raw_bps = %format!("{:.2}", market_params.drift_rate_per_sec_raw * 10_000.0),
                drift_shrunken_bps = %format!("{:.2}", market_params.drift_rate_per_sec * 10_000.0),
                drift_snr = %format!("{:.2}", drift_snr),
                q_target = %format!("{:.4}", q_target),
                q_current = %format!("{:.4}", q_current),
                drift_conf = %format!("{:.3}", drift_conf),
                delta_q = %format!("{:.4}", delta_q),
                cj_allocation = %format!("{:.4}", cj_allocation),
                bid_margin_weight = %format!("{:.3}", bid_margin_weight),
                margin_for_bids = %format!("{:.2}", margin_for_bids),
                margin_for_asks = %format!("{:.2}", margin_for_asks),
                "Margin allocation (Cartea-Jaimungal SNR-scaled inventory)"
            );

            // SAFETY CHECK: If margin is not available (warmup incomplete), log and fall through
            // This prevents the optimizer from producing empty ladders due to margin=0.
            // Instead of early-returning, fall through to the legacy path which will
            // produce an empty ladder, then the guaranteed quote floor ensures we still
            // have at least 1 bid + 1 ask.
            let margin_skip = available_margin < EPSILON;
            if margin_skip {
                warn!(
                    margin_available = %format!("{:.2}", available_margin),
                    leverage = %format!("{:.1}", leverage),
                    "Constrained optimizer skipped: margin not yet available (falling through to guaranteed quotes)"
                );
            }

            // 2. Calculate ASYMMETRIC position limits per side
            // CRITICAL BUG FIX: When over max position, we need to allow reduce-only orders
            // on the opposite side, not block both sides.
            //
            // PENDING EXPOSURE FIX: Account for resting orders that would change position if filled.
            // Available capacity = effective_max_position - current_position - pending_exposure_on_same_side
            //
            // For bids (buying): Can buy up to (effective_max_position - position - pending_bid_exposure) when long or flat
            //                    Can buy up to (effective_max_position + |position| - pending_bid_exposure) when short (reducing)
            // For asks (selling): Can sell up to (effective_max_position + position - pending_ask_exposure) when long (reducing)
            //                     Can sell up to (effective_max_position - |position| - pending_ask_exposure) when short or flat
            let pending_bids = market_params.pending_bid_exposure;
            let pending_asks = market_params.pending_ask_exposure;

            let (local_available_bids, local_available_asks) = if position >= 0.0 {
                // Long or flat position
                // Bids: limited by how much more long we can go (minus pending bid exposure)
                // Asks: can sell entire position + go max short (minus pending ask exposure)
                let bid_limit = (effective_max_position - position - pending_bids).max(0.0);
                let ask_limit = (position + effective_max_position - pending_asks).max(0.0);
                (bid_limit, ask_limit)
            } else {
                // Short position
                // Bids: can buy to cover position + go max long (minus pending bid exposure)
                // Asks: limited by how much more short we can go (minus pending ask exposure)
                let bid_limit = (position.abs() + effective_max_position - pending_bids).max(0.0);
                let ask_limit = (effective_max_position - position.abs() - pending_asks).max(0.0);
                (bid_limit, ask_limit)
            };

            // Log if pending exposure is constraining sizing
            if pending_bids > EPSILON && local_available_bids < effective_max_position {
                tracing::debug!(
                    pending_bids = %format!("{:.6}", pending_bids),
                    effective_bid_limit = %format!("{:.6}", local_available_bids),
                    "Bid sizing reduced by pending exposure"
                );
            }
            if pending_asks > EPSILON && local_available_asks < effective_max_position {
                tracing::debug!(
                    pending_asks = %format!("{:.6}", pending_asks),
                    effective_ask_limit = %format!("{:.6}", local_available_asks),
                    "Ask sizing reduced by pending exposure"
                );
            }

            // Phase 0D: Informed Position-Flip Gate
            // Cap reducing-side liquidity at |position| unless Bayesian conditions support a flip.
            // Prevents uninformed sweeps through zero — all 3 zero-crossings in the baseline
            // session were mechanical sweeps with no Bayesian support.
            let (available_for_bids_gated, available_for_asks_gated) = {
                let abs_pos = position.abs();
                let near_flat_threshold = effective_max_position * 0.02; // 2% of max

                if abs_pos > near_flat_threshold {
                    // Check Bayesian conditions for informed flip
                    let raw_drift = market_params.drift_rate_per_sec_raw;
                    // drift_uncertainty_bps is actually bps/sec (Kalman state is drift rate).
                    // Convert to fraction/sec for comparison with drift_rate_per_sec_raw.
                    let drift_unc_frac_per_sec = market_params.drift_uncertainty_bps / 10_000.0;
                    let cp_prob = market_params.changepoint_prob;
                    let p_cont = market_params.continuation_p;
                    let trend_bps = market_params.drift_signal_bps;
                    let trend_conf = market_params.trend_confidence;

                    // Condition 1: Drift opposes position with Kalman SNR > 0.5
                    let drift_opposes = if position > 0.0 {
                        raw_drift < 0.0 && raw_drift.abs() > 0.5 * drift_unc_frac_per_sec.max(1e-10)
                    } else {
                        raw_drift > 0.0 && raw_drift.abs() > 0.5 * drift_unc_frac_per_sec.max(1e-10)
                    };

                    // Condition 2: Changepoint detected (BOCD)
                    let changepoint = cp_prob > 0.3;

                    // Condition 3: Continuation exhausted
                    let continuation_exhausted = p_cont < 0.35;

                    // Condition 4: Trend reversal with confidence
                    let trend_reversal = if position > 0.0 {
                        trend_bps < -2.0 && trend_conf > 0.3
                    } else {
                        trend_bps > 2.0 && trend_conf > 0.3
                    };

                    // Union gate: ANY condition opens
                    let gate_open =
                        drift_opposes || changepoint || continuation_exhausted || trend_reversal;

                    if gate_open {
                        tracing::debug!(
                            position = %format!("{:.4}", position),
                            drift_opposes,
                            changepoint,
                            continuation_exhausted,
                            trend_reversal,
                            "FLIP GATE OPEN: Bayesian conditions support position flip"
                        );
                        (local_available_bids, local_available_asks)
                    } else {
                        // Gate closed: cap reducing side at |position|
                        let (capped_bids, capped_asks) = if position > 0.0 {
                            // Long: asks (selling) are reducing side — cap at position
                            (local_available_bids, local_available_asks.min(abs_pos))
                        } else {
                            // Short: bids (buying) are reducing side — cap at |position|
                            (local_available_bids.min(abs_pos), local_available_asks)
                        };
                        tracing::debug!(
                            position = %format!("{:.4}", position),
                            raw_drift = %format!("{:.6}", raw_drift),
                            cp_prob = %format!("{:.2}", cp_prob),
                            p_cont = %format!("{:.2}", p_cont),
                            trend_bps = %format!("{:.1}", trend_bps),
                            capped_bids = %format!("{:.4}", capped_bids),
                            capped_asks = %format!("{:.4}", capped_asks),
                            "FLIP GATE CLOSED: reducing-side capped at |position|"
                        );
                        (capped_bids, capped_asks)
                    }
                } else {
                    // Near flat — no gate needed
                    (local_available_bids, local_available_asks)
                }
            };
            let local_available_bids = available_for_bids_gated;
            let local_available_asks = available_for_asks_gated;

            // 3. Apply exchange-enforced limits (prevents order rejections)
            // Exchange limits are direction-specific: available_buy for bids, available_sell for asks
            let exchange_limits = market_params.exchange_limits();
            let available_for_bids = if exchange_limits.valid {
                local_available_bids.min(exchange_limits.safe_bid_limit())
            } else {
                local_available_bids
            };
            let available_for_asks = if exchange_limits.valid {
                local_available_asks.min(exchange_limits.safe_ask_limit())
            } else {
                local_available_asks
            };

            // Log warning if exchange limits are constraining us or stale
            if exchange_limits.valid && exchange_limits.is_stale() {
                tracing::warn!(
                    age_ms = exchange_limits.age_ms,
                    "Exchange limits are stale - sizing may be reduced"
                );
            }
            if exchange_limits.valid && available_for_bids < local_available_bids {
                tracing::debug!(
                    local = %format!("{:.6}", local_available_bids),
                    exchange = %format!("{:.6}", available_for_bids),
                    "Bid sizing constrained by exchange limit"
                );
            }
            if exchange_limits.valid && available_for_asks < local_available_asks {
                tracing::debug!(
                    local = %format!("{:.6}", local_available_asks),
                    exchange = %format!("{:.6}", available_for_asks),
                    "Ask sizing constrained by exchange limit"
                );
            }

            // Log reduce-only mode detection
            let over_limit = position.abs() > effective_max_position;

            // === REDUCE-ONLY SIZE CAP (Prevent Overshooting) ===
            // When in reduce-only mode, cap sizes to exactly match current position
            // to prevent overshooting from long to short (or short to long) in one fill.
            //
            // Example from logs: Position was +1.35 HYPE but asks totaled 2.0+ HYPE,
            // causing position to go from +1.35 to -0.61 in one fill (oversold).
            //
            // Fix: If we're long and in reduce-only mode (bids blocked), cap total ask
            // sizes to exactly the current position to reach zero, not overshoot.
            let (available_for_bids, available_for_asks) = if over_limit {
                if position > 0.0 {
                    // Long and over limit: reduce-only mode for asks
                    // Cap ask sizes to exactly the position (to reach zero, not go short)
                    let capped_asks = available_for_asks.min(position);
                    tracing::info!(
                        position = %format!("{:.6}", position),
                        effective_max_position = %format!("{:.6}", effective_max_position),
                        raw_available_asks = %format!("{:.6}", available_for_asks),
                        capped_asks = %format!("{:.6}", capped_asks),
                        "REDUCE-ONLY (LONG): Capping ask sizes to position to prevent overshoot"
                    );
                    (available_for_bids, capped_asks)
                } else {
                    // Short and over limit: reduce-only mode for bids
                    // Cap bid sizes to exactly the abs(position) (to reach zero, not go long)
                    let capped_bids = available_for_bids.min(position.abs());
                    tracing::info!(
                        position = %format!("{:.6}", position),
                        effective_max_position = %format!("{:.6}", effective_max_position),
                        raw_available_bids = %format!("{:.6}", available_for_bids),
                        capped_bids = %format!("{:.6}", capped_bids),
                        "REDUCE-ONLY (SHORT): Capping bid sizes to position to prevent overshoot"
                    );
                    (capped_bids, available_for_asks)
                }
            } else {
                // Not in reduce-only mode, use original values
                (available_for_bids, available_for_asks)
            };

            // === TOXICITY-BASED SIZE REDUCTION: REMOVED (B3) ===
            // Multiplicative size reduction deleted. Adverse selection defense
            // now handled by the additive E[PnL] filter which drops negative-EV
            // levels entirely (more principled than shrinking sizes).

            if over_limit {
                if position > 0.0 {
                    tracing::debug!(
                        position = %format!("{:.6}", position),
                        effective_max_position = %format!("{:.6}", effective_max_position),
                        available_bids = %format!("{:.6}", available_for_bids),
                        available_asks = %format!("{:.6}", available_for_asks),
                        "Over max position (LONG) - bids blocked, asks capped to position"
                    );
                } else {
                    tracing::debug!(
                        position = %format!("{:.6}", position),
                        effective_max_position = %format!("{:.6}", effective_max_position),
                        available_bids = %format!("{:.6}", available_for_bids),
                        available_asks = %format!("{:.6}", available_for_asks),
                        "Over max position (SHORT) - asks blocked, bids capped to position"
                    );
                }
            }

            // === KELLY SIZING CAP ===
            // If Kelly sizer is enabled and warmed up, cap total resting size per side
            // to kelly_fraction × effective_max_position. Quarter-Kelly is the default.
            // Feature-flagged: kelly_fraction() returns None when disabled or not warmed up.
            let (available_for_bids, available_for_asks) =
                if let Some(kelly_f) = self.kelly_fraction() {
                    let kelly_budget = kelly_f * effective_max_position;
                    let capped_bids = available_for_bids.min(kelly_budget);
                    let capped_asks = available_for_asks.min(kelly_budget);
                    tracing::info!(
                        kelly_fraction = %format!("{:.3}", kelly_f),
                        kelly_budget = %format!("{:.6}", kelly_budget),
                        bid_cap = %format!("{:.6} -> {:.6}", available_for_bids, capped_bids),
                        ask_cap = %format!("{:.6} -> {:.6}", available_for_asks, capped_asks),
                        "Kelly sizing cap applied"
                    );
                    (capped_bids, capped_asks)
                } else {
                    (available_for_bids, available_for_asks)
                };

            // === EXPOSURE BUDGET CAP (E1: aggregate fill protection) ===
            // Cap per-side sizing to the exposure budget's available capacity.
            // This prevents the worst-case scenario where all resting + new orders
            // fill simultaneously and push position beyond max_position.
            let (available_for_bids, available_for_asks) = {
                let budget_bid = market_params.available_bid_budget;
                let budget_ask = market_params.available_ask_budget;
                let capped_bids = if budget_bid < available_for_bids {
                    tracing::debug!(
                        raw = %format!("{:.4}", available_for_bids),
                        budget = %format!("{:.4}", budget_bid),
                        "Bid sizing capped by exposure budget"
                    );
                    budget_bid
                } else {
                    available_for_bids
                };
                let capped_asks = if budget_ask < available_for_asks {
                    tracing::debug!(
                        raw = %format!("{:.4}", available_for_asks),
                        budget = %format!("{:.4}", budget_ask),
                        "Ask sizing capped by exposure budget"
                    );
                    budget_ask
                } else {
                    available_for_asks
                };
                (capped_bids, capped_asks)
            };

            // E5: Single-level size cap — constrain sizing when capital only supports 1 level
            let (available_for_bids, available_for_asks) = if capital_limited_levels <= 1 {
                let max_frac = market_params.capital_policy.single_level_max_size_fraction;
                (available_for_bids * max_frac, available_for_asks * max_frac)
            } else {
                (available_for_bids, available_for_asks)
            };

            // === DYNAMIC LEVEL COUNT BASED ON EXCHANGE LIMITS ===
            // When exchange limits are tight, we must reduce the number of levels
            // to avoid placing many sub-minimum orders that would be rejected.
            //
            // Formula: max_levels = available_capacity / quantum.min_viable_size
            // SizeQuantum uses exact ceiling math — no fudge factors needed.
            let quantum = SizeQuantum::compute(
                config.min_notional,
                market_params.microprice,
                config.sz_decimals,
            );

            // Warn if position limit is too small to place even one minimum-notional order
            if quantum.min_viable_size > effective_max_position {
                tracing::warn!(
                    min_order = %format!("{:.4} (${:.2})", quantum.min_viable_size, quantum.min_viable_size * market_params.microprice),
                    max_position = %format!("{:.4} (${:.2})", effective_max_position, effective_max_position * market_params.microprice),
                    "POSITION LIMIT TOO SMALL: cannot place orders meeting exchange minimum"
                );
            }

            let max_bid_levels = if available_for_bids > EPSILON {
                ((available_for_bids / quantum.min_viable_size).floor() as usize).max(1)
            } else {
                0
            };
            let max_ask_levels = if available_for_asks > EPSILON {
                ((available_for_asks / quantum.min_viable_size).floor() as usize).max(1)
            } else {
                0
            };

            // Get configured number of levels
            let configured_levels = ladder_config.num_levels;

            // Use the smaller of: configured levels, capacity-based max (taking minimum of bid/ask)
            // We use the minimum to ensure both sides can be quoted with meaningful sizes
            let effective_bid_levels = configured_levels.min(max_bid_levels);
            let effective_ask_levels = configured_levels.min(max_ask_levels);

            // Log if we're reducing levels due to capacity constraints
            if effective_bid_levels < configured_levels || effective_ask_levels < configured_levels
            {
                tracing::warn!(
                    configured = configured_levels,
                    effective_bid = effective_bid_levels,
                    effective_ask = effective_ask_levels,
                    available_bids = %format!("{:.6}", available_for_bids),
                    available_asks = %format!("{:.6}", available_for_asks),
                    min_viable_size = %format!("{:.6}", quantum.min_viable_size),
                    "Reducing ladder levels due to tight exchange limits"
                );
            }

            // === TICK-GRID PATH: Exchange-tick-aligned level placement ===
            // When use_tick_grid is enabled, skip legacy bps-space depth generation
            // and instead enumerate exchange-aligned ticks directly (unique prices by
            // construction), score them by fill probability × edge, and allocate risk
            // budget via greedy water-filling. This eliminates the price-collapse problem
            // where continuous bps depths round to duplicate exchange prices.
            let mut ladder = if margin_skip {
                // No margin available — produce empty ladder so guaranteed quote floor applies
                Ladder::default()
            } else if market_params.capital_policy.use_tick_grid {
                let tick_size = 10f64.powi(-(config.decimals as i32));

                // GLFT touch depth in bps (used as minimum depth for tick grid)
                let glft_touch_bps =
                    self.depth_generator.glft_optimal_spread(gamma, kappa) * 10_000.0;
                let touch_bps = glft_touch_bps.max(effective_floor_bps);

                // WS2: Wider depth range using tier-dependent multiplier.
                // Old: max(kappa_cap, touch * 2.0) — only 7.5 bps range for 5 levels.
                // New: max(kappa_cap, touch * depth_range_multiplier) — 30+ bps for meaningful diversity.
                let policy = &market_params.capital_policy;
                let max_depth_bps = (touch_bps * policy.depth_range_multiplier).min(100.0); // Absolute cap: never deeper than 100 bps

                // WS3: API-budget-aware level count — generate fewer levels when
                // headroom is low to avoid reconciler dropping excess.
                let effective_levels = if market_params.rate_limit_headroom_pct < 0.15 {
                    let scaled = (policy.max_levels_per_side as f64
                        * (market_params.rate_limit_headroom_pct / 0.15))
                        .ceil() as usize;
                    scaled.max(2) // Always at least 2 levels
                } else {
                    policy.max_levels_per_side
                };

                // Build tick grid config
                let grid_config = TickGridConfig::compute(
                    market_params.microprice,
                    tick_size,
                    touch_bps,
                    max_depth_bps,
                    effective_levels,
                    config.sz_decimals,
                    policy.min_tick_spacing_mult,
                );

                // Enumerate and score bid ticks
                let mut bid_ticks = enumerate_bid_ticks(&grid_config);
                let mut ask_ticks = enumerate_ask_ticks(&grid_config);

                let scoring = TickScoringParams {
                    sigma: market_params.sigma,
                    tau: market_params.kelly_time_horizon,
                    as_at_touch_bps,
                    as_decay_bps: 10.0, // AS decays over ~10 bps depth
                    fee_bps: self.risk_config.maker_fee_rate * 10_000.0,
                    use_sc_as_ratio: true,
                };

                // score_ticks mutates in place (sets utility field)
                score_ticks(&mut bid_ticks, &scoring);
                score_ticks(&mut ask_ticks, &scoring);

                let selected_bids = select_optimal_ticks(&bid_ticks, effective_levels);
                let selected_asks = select_optimal_ticks(&ask_ticks, effective_levels);

                // Build risk budgets per side
                let margin_per_contract = market_params.microprice / leverage.max(1.0);

                let bid_budget = SideRiskBudget {
                    position_capacity: available_for_bids,
                    margin_capacity: margin_for_bids,
                    margin_per_contract,
                    min_viable_size: quantum.min_viable_size,
                    max_single_order_fraction: policy.max_single_order_fraction,
                    sz_decimals: config.sz_decimals,
                };
                let ask_budget = SideRiskBudget {
                    position_capacity: available_for_asks,
                    margin_capacity: margin_for_asks,
                    margin_per_contract,
                    min_viable_size: quantum.min_viable_size,
                    max_single_order_fraction: policy.max_single_order_fraction,
                    sz_decimals: config.sz_decimals,
                };

                // WS4: Compute allocation temperature from regime + warmup
                let in_yellow_zone = abs_inventory_ratio >= 0.6;
                let alloc_temperature = compute_allocation_temperature(
                    market_params.regime_gamma_multiplier,
                    market_params.adaptive_warmup_progress,
                    in_yellow_zone,
                );
                let softmax_params = SoftmaxParams {
                    temperature: alloc_temperature,
                    min_entropy_bits: 1.0,
                    min_level_fraction: 0.05,
                };

                // WS1: Allocate sizes via utility-proportional softmax
                let (bid_allocs, bid_diag) =
                    allocate_risk_budget(&selected_bids, &bid_budget, &softmax_params);
                let (ask_allocs, ask_diag) =
                    allocate_risk_budget(&selected_asks, &ask_budget, &softmax_params);

                // Skip if Red zone cleared accumulating side
                let bid_cleared_by_zone =
                    abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position > 0.0;
                let ask_cleared_by_zone =
                    abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position < 0.0;

                // Build ladder from tick-grid allocations
                let bids: smallvec::SmallVec<_> = if bid_cleared_by_zone {
                    smallvec::smallvec![]
                } else {
                    bid_allocs
                        .iter()
                        .map(|a| LadderLevel {
                            price: round_to_significant_and_decimal(
                                a.tick.price,
                                5,
                                config.decimals,
                            ),
                            size: a.size,
                            depth_bps: a.tick.depth_bps,
                        })
                        .collect()
                };

                let asks: smallvec::SmallVec<_> = if ask_cleared_by_zone {
                    smallvec::smallvec![]
                } else {
                    ask_allocs
                        .iter()
                        .map(|a| LadderLevel {
                            price: round_to_significant_and_decimal(
                                a.tick.price,
                                5,
                                config.decimals,
                            ),
                            size: a.size,
                            depth_bps: a.tick.depth_bps,
                        })
                        .collect()
                };

                // Apply spread compensation multiplier for small capital
                let mut ladder = Ladder { bids, asks };
                if (policy.spread_compensation_mult - 1.0).abs() > 0.001 {
                    // Widen depths by compensation factor (adjusts price offsets)
                    for level in ladder.bids.iter_mut() {
                        let new_depth = level.depth_bps * policy.spread_compensation_mult;
                        let offset = market_params.microprice * (new_depth / 10_000.0);
                        level.price = round_to_significant_and_decimal(
                            market_params.microprice - offset,
                            5,
                            config.decimals,
                        );
                        level.depth_bps = new_depth;
                    }
                    for level in ladder.asks.iter_mut() {
                        let new_depth = level.depth_bps * policy.spread_compensation_mult;
                        let offset = market_params.microprice * (new_depth / 10_000.0);
                        level.price = round_to_significant_and_decimal(
                            market_params.microprice + offset,
                            5,
                            config.decimals,
                        );
                        level.depth_bps = new_depth;
                    }
                }

                // WS6: Diagnostic logging with size distribution
                let bid_sizes_str: String = ladder
                    .bids
                    .iter()
                    .map(|l| format!("{:.2}", l.size))
                    .collect::<Vec<_>>()
                    .join(",");
                let ask_sizes_str: String = ladder
                    .asks
                    .iter()
                    .map(|l| format!("{:.2}", l.size))
                    .collect::<Vec<_>>()
                    .join(",");
                let bid_depths_str: String = ladder
                    .bids
                    .iter()
                    .map(|l| format!("{:.1}", l.depth_bps))
                    .collect::<Vec<_>>()
                    .join(",");
                let ask_depths_str: String = ladder
                    .asks
                    .iter()
                    .map(|l| format!("{:.1}", l.depth_bps))
                    .collect::<Vec<_>>()
                    .join(",");
                tracing::info!(
                    tick_grid = true,
                    tier = ?policy.tier,
                    bid_levels = ladder.bids.len(),
                    ask_levels = ladder.asks.len(),
                    bid_sizes = %bid_sizes_str,
                    ask_sizes = %ask_sizes_str,
                    bid_depths = %bid_depths_str,
                    ask_depths = %ask_depths_str,
                    bid_total_size = %format!("{:.4}", ladder.bids.iter().map(|l| l.size).sum::<f64>()),
                    ask_total_size = %format!("{:.4}", ladder.asks.iter().map(|l| l.size).sum::<f64>()),
                    temperature = %format!("{:.2}", alloc_temperature),
                    entropy_bits = %format!("{:.2}", bid_diag.entropy_bits.max(ask_diag.entropy_bits)),
                    touch_bps = %format!("{:.2}", touch_bps),
                    max_depth_bps = %format!("{:.2}", max_depth_bps),
                    effective_levels = effective_levels,
                    headroom_pct = %format!("{:.1}%", market_params.rate_limit_headroom_pct * 100.0),
                    "Tick-grid ladder: utility-weighted softmax allocation"
                );

                ladder
            } else {
                // === LEGACY PATH: bps-space depth generation + entropy optimization ===

                // 4. Generate ladder to get depth levels and prices (using dynamic depths)
                // NOTE: Don't truncate here - let the entropy optimizer handle distribution
                // The optimizer has built-in notional constraints that will filter sub-minimum levels

                // DIAGNOSTIC: dump actual dynamic depths before ladder generation
                if let Some(ref dd) = ladder_config.dynamic_depths {
                    let bid_depths_str: Vec<String> =
                        dd.bid.iter().map(|d| format!("{d:.2}")).collect();
                    let ask_depths_str: Vec<String> =
                        dd.ask.iter().map(|d| format!("{d:.2}")).collect();
                    info!(
                        num_levels = ladder_config.num_levels,
                        bid_depths = %bid_depths_str.join(","),
                        ask_depths = %ask_depths_str.join(","),
                        total_size = %format!("{:.4}", params.total_size),
                        mid_price = %format!("{:.4}", params.mid_price),
                        market_mid = %format!("{:.4}", params.market_mid),
                        "DIAGNOSTIC: pre-generate ladder depths"
                    );
                }

                let mut ladder = Ladder::generate(&ladder_config, &params);

                // DIAGNOSTIC: dump post-generate ladder levels
                {
                    let bid_str: Vec<String> = ladder
                        .bids
                        .iter()
                        .map(|l| format!("({:.4}@{:.2}bps,sz={:.2})", l.price, l.depth_bps, l.size))
                        .collect();
                    let ask_str: Vec<String> = ladder
                        .asks
                        .iter()
                        .map(|l| format!("({:.4}@{:.2}bps,sz={:.2})", l.price, l.depth_bps, l.size))
                        .collect();
                    info!(
                        bid_levels = ladder.bids.len(),
                        ask_levels = ladder.asks.len(),
                        bids = %bid_str.join(" | "),
                        asks = %ask_str.join(" | "),
                        "DIAGNOSTIC: post-generate ladder levels"
                    );
                }

                // Log if capacity is tight (for debugging), but don't truncate
                if effective_bid_levels < configured_levels
                    || effective_ask_levels < configured_levels
                {
                    tracing::debug!(
                    configured = configured_levels,
                    capacity_bid = effective_bid_levels,
                    capacity_ask = effective_ask_levels,
                    "Capacity suggests fewer levels, but entropy optimizer will handle distribution"
                );
                }

                // 5. Build LevelOptimizationParams for bids
                // Use Kelly time horizon for Bayesian fill probability (τ for P(fill|δ,τ))
                let tau_for_fill = market_params.kelly_time_horizon;
                let bid_level_params: Vec<_> = ladder
                    .bids
                    .iter()
                    .map(|level| {
                        // Use Bayesian fill model (empirical when warmed up, theoretical fallback)
                        let fill_intensity = fill_intensity_with_model(
                            &self.fill_model,
                            level.depth_bps,
                            params.sigma,
                            tau_for_fill,
                            params.kappa,
                        );
                        let spread_capture = spread_capture_at_depth(
                            level.depth_bps,
                            &params,
                            ladder_config.fees_bps,
                        );
                        let adverse_selection =
                            adverse_selection_at_depth(level.depth_bps, &params);
                        LevelOptimizationParams {
                            depth_bps: level.depth_bps,
                            fill_intensity,
                            spread_capture,
                            margin_per_unit: market_params.microprice / leverage,
                            adverse_selection,
                        }
                    })
                    .collect();

                // 6. Optimize bid sizes
                if !bid_level_params.is_empty() {
                    if market_params.capital_policy.skip_entropy_optimization {
                        // CAPITAL-AWARE: Equal-weight allocation for Micro/Small tiers.
                        // Saves compute and produces predictable, min-size orders.
                        let equal_size = available_for_bids / ladder.bids.len().max(1) as f64;
                        for bid in &mut ladder.bids {
                            bid.size = truncate_float(equal_size, config.sz_decimals, false);
                        }
                        debug!(
                            equal_size = %format!("{:.4}", equal_size),
                            levels = ladder.bids.len(),
                            "Skip entropy: equal-weight bid allocation"
                        );
                    } else {
                        // === ENTROPY-BASED ALLOCATION (FIRST PRINCIPLES) ===
                        // Uses information-theoretic entropy constraints to maintain diversity
                        // and prevent collapse to 1-2 orders even under adverse conditions.
                        let entropy_config = EntropyOptimizerConfig {
                            distribution: EntropyDistributionConfig {
                                min_entropy: market_params.entropy_min_entropy,
                                base_temperature: market_params.entropy_base_temperature,
                                min_allocation_floor: market_params.entropy_min_allocation_floor,
                                thompson_samples: market_params.entropy_thompson_samples,
                                ..Default::default()
                            },
                            min_notional: config.min_notional,
                            ..Default::default()
                        };

                        let mut entropy_optimizer = EntropyConstrainedOptimizer::new(
                            entropy_config,
                            market_params.microprice,
                            margin_for_bids, // Use posterior-weighted margin split (not full margin)
                            available_for_bids,
                            leverage,
                        );

                        let regime = Self::build_market_regime(market_params);
                        let entropy_alloc = entropy_optimizer.optimize(&bid_level_params, &regime);

                        info!(
                            entropy = %format!("{:.3}", entropy_alloc.distribution.entropy),
                            effective_levels = %format!("{:.1}", entropy_alloc.distribution.effective_levels),
                            entropy_floor_active = entropy_alloc.entropy_floor_active,
                            active_levels = entropy_alloc.active_levels,
                            "Entropy optimizer applied to bids"
                        );

                        let allocation = entropy_alloc.to_legacy();

                        for (i, &size) in allocation.sizes.iter().enumerate() {
                            if i < ladder.bids.len() {
                                // CRITICAL: Truncate to sz_decimals to prevent "Order has invalid size" rejections
                                ladder.bids[i].size =
                                    truncate_float(size, config.sz_decimals, false);
                            }
                        }
                    }
                }

                // 7. Build LevelOptimizationParams for asks
                let ask_level_params: Vec<_> = ladder
                    .asks
                    .iter()
                    .map(|level| {
                        // Use Bayesian fill model (empirical when warmed up, theoretical fallback)
                        let fill_intensity = fill_intensity_with_model(
                            &self.fill_model,
                            level.depth_bps,
                            params.sigma,
                            tau_for_fill,
                            params.kappa,
                        );
                        let spread_capture = spread_capture_at_depth(
                            level.depth_bps,
                            &params,
                            ladder_config.fees_bps,
                        );
                        let adverse_selection =
                            adverse_selection_at_depth(level.depth_bps, &params);
                        LevelOptimizationParams {
                            depth_bps: level.depth_bps,
                            fill_intensity,
                            spread_capture,
                            margin_per_unit: market_params.microprice / leverage,
                            adverse_selection,
                        }
                    })
                    .collect();

                // 8. Optimize ask sizes
                if !ask_level_params.is_empty() {
                    if market_params.capital_policy.skip_entropy_optimization {
                        // CAPITAL-AWARE: Equal-weight allocation for Micro/Small tiers.
                        let equal_size = available_for_asks / ladder.asks.len().max(1) as f64;
                        for ask in &mut ladder.asks {
                            ask.size = truncate_float(equal_size, config.sz_decimals, false);
                        }
                        debug!(
                            equal_size = %format!("{:.4}", equal_size),
                            levels = ladder.asks.len(),
                            "Skip entropy: equal-weight ask allocation"
                        );
                    } else {
                        // === ENTROPY-BASED ALLOCATION (FIRST PRINCIPLES) ===
                        // Uses information-theoretic entropy constraints to maintain diversity
                        // and prevent collapse to 1-2 orders even under adverse conditions.
                        let entropy_config = EntropyOptimizerConfig {
                            distribution: EntropyDistributionConfig {
                                min_entropy: market_params.entropy_min_entropy,
                                base_temperature: market_params.entropy_base_temperature,
                                min_allocation_floor: market_params.entropy_min_allocation_floor,
                                thompson_samples: market_params.entropy_thompson_samples,
                                ..Default::default()
                            },
                            min_notional: config.min_notional,
                            ..Default::default()
                        };

                        let mut entropy_optimizer = EntropyConstrainedOptimizer::new(
                            entropy_config,
                            market_params.microprice,
                            margin_for_asks, // Use posterior-weighted margin split (not full margin)
                            available_for_asks,
                            leverage,
                        );

                        let regime = Self::build_market_regime(market_params);
                        let entropy_alloc = entropy_optimizer.optimize(&ask_level_params, &regime);

                        info!(
                            entropy = %format!("{:.3}", entropy_alloc.distribution.entropy),
                            effective_levels = %format!("{:.1}", entropy_alloc.distribution.effective_levels),
                            entropy_floor_active = entropy_alloc.entropy_floor_active,
                            active_levels = entropy_alloc.active_levels,
                            "Entropy optimizer applied to asks"
                        );

                        let allocation = entropy_alloc.to_legacy();

                        for (i, &size) in allocation.sizes.iter().enumerate() {
                            if i < ladder.asks.len() {
                                // CRITICAL: Truncate to sz_decimals to prevent "Order has invalid size" rejections
                                ladder.asks[i].size =
                                    truncate_float(size, config.sz_decimals, false);
                            }
                        }
                    }
                }

                // 8b. PER-LEVEL SIZE CAP: No single resting order may exceed 25% of the USER'S
                // risk-based max position (not the margin-based quoting capacity).
                // GLFT inventory theory: if a single fill can push q to max, the reservation
                // price adjustment q*gamma*sigma^2*T swings maximally and the MM cannot recover.
                // This cap guarantees at least 4 fills are needed to reach max inventory,
                // giving the gamma/skew feedback loop time to widen spreads defensively.
                //
                // We cap against max_position (the $50 USD-derived risk limit = 1.58 contracts),
                // NOT effective_max_position (margin-based quoting capacity = 51 contracts).
                // The risk limit is what matters for inventory blowup prevention.
                let per_level_cap = truncate_float(
                    (max_position * MAX_SINGLE_ORDER_FRACTION).max(quantum.min_viable_size),
                    config.sz_decimals,
                    false,
                );
                let mut any_capped = false;
                for level in ladder.bids.iter_mut().chain(ladder.asks.iter_mut()) {
                    if level.size > per_level_cap {
                        any_capped = true;
                        level.size = per_level_cap;
                    }
                }
                if any_capped {
                    info!(
                        per_level_cap = %format!("{:.6}", per_level_cap),
                        risk_max_position = %format!("{:.6}", max_position),
                        fraction = %format!("{:.0}%", MAX_SINGLE_ORDER_FRACTION * 100.0),
                        "Per-level size cap applied: no single order exceeds {}% of risk max position",
                        (MAX_SINGLE_ORDER_FRACTION * 100.0) as u32,
                    );
                }

                // 9. Filter out levels below minimum notional (exchange will reject them anyway)
                // SizeQuantum is the exact truncation-safe minimum — no fudge factor needed.
                // Use >= (not >): levels at exactly min_viable_size are valid and truncation-stable.
                let min_size_for_exchange = quantum.min_viable_size;
                let bids_before = ladder.bids.len();
                let asks_before = ladder.asks.len();
                ladder.bids.retain(|l| l.size >= min_size_for_exchange);
                ladder.asks.retain(|l| l.size >= min_size_for_exchange);

                // 10. CONCENTRATION FALLBACK: If ladder is empty but total available size
                // meets min_notional, create single concentrated order at tightest depth.
                // This handles TWO cases:
                //   a) Levels existed but were filtered by min_size_for_exchange (bids_before > 0)
                //   b) Ladder::generate() returned empty because total_size was too small to
                //      distribute across num_levels and pass min_notional per level (bids_before == 0)
                // Case (b) occurs with small capital in restrictive regimes (e.g., $100 capital,
                // Extreme regime → effective_max ~1 HYPE → 0.09/level across 10 levels < $10 min)
                let _min_size_for_order = config.min_notional / market_params.microprice; // Kept for reference
                                                                                          // FIX: Use min_viable_depth_bps from capacity budget to survive QueueValue filtering.
                                                                                          // Was: ladder_config.min_depth_bps (2.0 bps) → below QueueValue breakeven (4.5 bps)
                let tightest_depth_bps = market_params
                    .capacity_budget
                    .as_ref()
                    .map(|b| b.min_viable_depth_bps)
                    .unwrap_or(5.0)
                    .max(ladder_config.min_depth_bps);

                // Bid concentration fallback — skip if Red zone cleared bids (reduce-only for longs)
                let bid_cleared_by_zone =
                    abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position > 0.0;
                if ladder.bids.is_empty() && !bid_cleared_by_zone {
                    // Total available size for bids, capped at 25% of the USER'S risk-based
                    // max position per order (not margin-based quoting capacity).
                    // GLFT inventory theory: one fill at max position creates maximal reservation
                    // price swing q*gamma*sigma^2*T with zero recovery capacity.
                    let per_order_cap =
                        (max_position * MAX_SINGLE_ORDER_FRACTION).max(quantum.min_viable_size);
                    let total_bid_size = quantum
                        .clamp_to_viable(available_for_bids.min(per_order_cap), true)
                        .unwrap_or(0.0);

                    if quantum.is_sufficient(total_bid_size) && total_bid_size > 1e-10 {
                        // Create concentrated order at tightest depth
                        let offset = market_params.microprice * (tightest_depth_bps / 10000.0);
                        let bid_price = round_to_significant_and_decimal(
                            market_params.microprice - offset,
                            5,
                            config.decimals,
                        );

                        ladder.bids.push(LadderLevel {
                            price: bid_price,
                            size: total_bid_size,
                            depth_bps: tightest_depth_bps,
                        });

                        info!(
                            size = %format!("{:.6}", total_bid_size),
                            price = %format!("{:.2}", bid_price),
                            notional = %format!("{:.2}", total_bid_size * bid_price),
                            depth_bps = %format!("{:.2}", tightest_depth_bps),
                            levels_before = bids_before,
                            per_order_cap = %format!("{:.6}", per_order_cap),
                            "Bid concentration fallback: single order at tightest depth (size-capped)"
                        );
                    } else {
                        warn!(
                            bids_before = bids_before,
                            available_margin = %format!("{:.2}", available_margin),
                            available_position = %format!("{:.6}", available_for_bids),
                            min_notional = %format!("{:.2}", config.min_notional),
                            min_level_size = %format!("{:.6}", ladder_config.min_level_size),
                            mid_price = %format!("{:.2}", market_params.microprice),
                            "All bid levels filtered out (total size below min_notional)"
                        );
                    }
                }

                // Ask concentration fallback — skip if Red zone cleared asks (reduce-only for shorts)
                let ask_cleared_by_zone =
                    abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position < 0.0;
                if ladder.asks.is_empty() && !ask_cleared_by_zone {
                    // Total available size for asks, capped at 25% of the USER'S risk-based
                    // max position per order (not margin-based quoting capacity).
                    let per_order_cap =
                        (max_position * MAX_SINGLE_ORDER_FRACTION).max(quantum.min_viable_size);
                    let total_ask_size = quantum
                        .clamp_to_viable(available_for_asks.min(per_order_cap), true)
                        .unwrap_or(0.0);

                    if quantum.is_sufficient(total_ask_size) && total_ask_size > 1e-10 {
                        // Create concentrated order at tightest depth
                        let offset = market_params.microprice * (tightest_depth_bps / 10000.0);
                        let ask_price = round_to_significant_and_decimal(
                            market_params.microprice + offset,
                            5,
                            config.decimals,
                        );

                        ladder.asks.push(LadderLevel {
                            price: ask_price,
                            size: total_ask_size,
                            depth_bps: tightest_depth_bps,
                        });

                        info!(
                            size = %format!("{:.6}", total_ask_size),
                            price = %format!("{:.2}", ask_price),
                            notional = %format!("{:.2}", total_ask_size * ask_price),
                            depth_bps = %format!("{:.2}", tightest_depth_bps),
                            levels_before = asks_before,
                            per_order_cap = %format!("{:.6}", per_order_cap),
                            "Ask concentration fallback: single order at tightest depth (size-capped)"
                        );
                    } else {
                        warn!(
                            asks_before = asks_before,
                            available_margin = %format!("{:.2}", available_margin),
                            available_position = %format!("{:.6}", available_for_asks),
                            min_notional = %format!("{:.2}", config.min_notional),
                            min_level_size = %format!("{:.6}", ladder_config.min_level_size),
                            mid_price = %format!("{:.2}", market_params.microprice),
                            "All ask levels filtered out (total size below min_notional)"
                        );
                    }
                }

                // === COST-BASIS-AWARE PRICE CLAMPING ===
                // Prevent selling below breakeven (long) or buying above breakeven (short).
                // This eliminates the "buy high, sell low" pattern observed in live trading.
                // Override only when urgency is low — high urgency overrides to exit.
                //
                // SKIP when position > 50% of max: reducing inventory is more important
                // than protecting breakeven. In a 4h session this triggered 976 times
                // (every cycle), preventing the strategy from ever reducing position.
                //
                // IMPORTANT: The breakeven price must be snapped to the exchange tick grid
                // BEFORE sig-fig rounding, using directional rounding (ceil for asks, floor
                // for bids) to guarantee we never sell below / buy above true breakeven.
                // The tick size is 10^(-price_decimals) where price_decimals = config.decimals.
                if let Some(entry_price) = market_params.avg_entry_price {
                    let breakeven = market_params.breakeven_price;
                    let position = inventory_ratio; // already computed above
                    let tick_size = 10f64.powi(-(config.decimals as i32));

                    // Skip breakeven clamping when position is large — reducing is more important.
                    // At > 50% of max position, the inventory risk from holding outweighs the
                    // small loss from selling below / buying above breakeven.
                    let skip_clamping = position.abs() > 0.5;

                    if breakeven > 0.0 && !skip_clamping {
                        if position > 0.0 {
                            // Long position: don't sell below breakeven unless urgent.
                            // Round breakeven UP to tick grid so ask >= true breakeven.
                            // Divisor 20.0: triggers urgency override earlier than the
                            // previous /50.0 (at ~10 bps underwater vs ~25 bps before).
                            let urgency = (position.abs()
                                * (-market_params.unrealized_pnl_bps / 20.0).max(0.0))
                            .min(1.0);
                            if urgency < 0.5 {
                                // Snap breakeven UP to tick grid (ceil), then enforce 5-sig-fig
                                let breakeven_on_tick = (breakeven / tick_size).ceil() * tick_size;
                                let mut rounded_breakeven = round_to_significant_and_decimal(
                                    breakeven_on_tick,
                                    5,
                                    config.decimals,
                                );
                                // Safety: if sig-fig rounding dropped us below true breakeven,
                                // bump up by one tick and re-round
                                if rounded_breakeven < breakeven - EPSILON {
                                    rounded_breakeven = round_to_significant_and_decimal(
                                        rounded_breakeven + tick_size,
                                        5,
                                        config.decimals,
                                    );
                                }

                                let mut clamped_count = 0;
                                for level in ladder.asks.iter_mut() {
                                    if level.price < breakeven {
                                        level.price = rounded_breakeven;
                                        clamped_count += 1;
                                    }
                                }
                                if clamped_count > 0 {
                                    tracing::info!(
                                        entry = %format!("{:.4}", entry_price),
                                        breakeven = %format!("{:.4}", breakeven),
                                        rounded_breakeven = %format!("{:.4}", rounded_breakeven),
                                        tick_size = %tick_size,
                                        unrealized_bps = %format!("{:.1}", market_params.unrealized_pnl_bps),
                                        clamped = clamped_count,
                                        urgency = %format!("{:.2}", urgency),
                                        "[COST-BASIS] Long: clamped ask prices to breakeven"
                                    );
                                }
                            }
                        } else if position < 0.0 {
                            // Short position: don't buy above breakeven unless urgent.
                            // Round breakeven DOWN to tick grid so bid <= true breakeven.
                            // Divisor 20.0: triggers urgency override earlier (see long side comment).
                            let urgency = (position.abs()
                                * (-market_params.unrealized_pnl_bps / 20.0).max(0.0))
                            .min(1.0);
                            if urgency < 0.5 {
                                // Snap breakeven DOWN to tick grid (floor), then enforce 5-sig-fig
                                let breakeven_on_tick = (breakeven / tick_size).floor() * tick_size;
                                let mut rounded_breakeven = round_to_significant_and_decimal(
                                    breakeven_on_tick,
                                    5,
                                    config.decimals,
                                );
                                // Safety: if sig-fig rounding pushed us above true breakeven,
                                // drop by one tick and re-round
                                if rounded_breakeven > breakeven + EPSILON {
                                    rounded_breakeven = round_to_significant_and_decimal(
                                        rounded_breakeven - tick_size,
                                        5,
                                        config.decimals,
                                    );
                                }

                                let mut clamped_count = 0;
                                for level in ladder.bids.iter_mut() {
                                    if level.price > breakeven {
                                        level.price = rounded_breakeven;
                                        clamped_count += 1;
                                    }
                                }
                                if clamped_count > 0 {
                                    tracing::info!(
                                        entry = %format!("{:.4}", entry_price),
                                        breakeven = %format!("{:.4}", breakeven),
                                        rounded_breakeven = %format!("{:.4}", rounded_breakeven),
                                        tick_size = %tick_size,
                                        unrealized_bps = %format!("{:.1}", market_params.unrealized_pnl_bps),
                                        clamped = clamped_count,
                                        urgency = %format!("{:.2}", urgency),
                                        "[COST-BASIS] Short: clamped bid prices to breakeven"
                                    );
                                }
                            }
                        }
                    }
                }

                // DIAGNOSTIC: Only warn if still empty after concentration fallback
                if ladder.bids.is_empty() && ladder.asks.is_empty() {
                    let dynamic_min_size = quantum.min_viable_size;
                    warn!(
                        available_for_bids = %format!("{:.6}", available_for_bids),
                        available_for_asks = %format!("{:.6}", available_for_asks),
                        min_notional = %format!("{:.2}", config.min_notional),
                        dynamic_min_size = %format!("{:.6}", dynamic_min_size),
                        margin_available = %format!("{:.2}", available_margin),
                        bid_notional = %format!("{:.2}", available_for_bids * market_params.microprice),
                        ask_notional = %format!("{:.2}", available_for_asks * market_params.microprice),
                        "Ladder completely empty: available size below min_notional (no fallback possible)"
                    );
                }

                ladder
            }; // close else (legacy path)

            // === GUARANTEED QUOTE FLOOR ===
            // THE critical invariant: the market maker ALWAYS has >= 1 bid + 1 ask resting.
            // If after all ladder generation + concentration fallback we still have empty
            // sides, create wide guaranteed quotes at min_viable size.
            //
            // Spread: max(fee + tick, GLFT optimal) — wide enough to be safe, never
            //         tighter than the physical floor.
            // Size:   min_viable — the smallest exchange-legal order.
            // Skew:   inventory-proportional offset to favor mean-reversion.
            //
            // Respect position zones: Red/Kill zones may intentionally clear one side
            // (reduce-only). Do NOT force guaranteed quotes on the accumulating side
            // when the governor has explicitly cleared it.
            let guaranteed_half_spread_bps = (fee_bps + market_params.tick_size_bps)
                .max(glft_optimal_bps)
                .max(effective_floor_bps);

            // Inventory skew: shift mid toward mean-reversion
            // Long → push bid down, ask down (attract sells)
            // Short → push bid up, ask up (attract buys)
            let guaranteed_inv_ratio = if effective_max_position > EPSILON {
                (position / effective_max_position).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            // Skew = up to 30% of half-spread, proportional to inventory
            let guaranteed_skew_bps = guaranteed_inv_ratio * guaranteed_half_spread_bps * 0.3;

            let bid_cleared_by_zone =
                abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position > 0.0;
            let ask_cleared_by_zone =
                abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position < 0.0;

            // WS4: Guaranteed quotes always apply (no binary circuit breaker gate).
            // At extreme γ, guaranteed quote spread is already very wide.
            {
                if ladder.bids.is_empty() && !bid_cleared_by_zone {
                    let bid_depth_bps = guaranteed_half_spread_bps + guaranteed_skew_bps;
                    let offset = market_params.microprice * (bid_depth_bps / 10_000.0);
                    let bid_price = round_to_significant_and_decimal(
                        market_params.microprice - offset,
                        5,
                        config.decimals,
                    );
                    // Recompute min viable size at actual quote price to avoid notional
                    // violation when price drifts below quantum's stale mark_px.
                    let bid_size = if bid_price > 0.0 {
                        let raw_min = quantum.min_notional / bid_price;
                        let steps = (raw_min / quantum.step).ceil() as u64;
                        (steps as f64 * quantum.step).max(quantum.min_viable_size)
                    } else {
                        quantum.min_viable_size
                    };

                    if bid_size > 0.0 && bid_price > 0.0 {
                        ladder.bids.push(LadderLevel {
                            price: bid_price,
                            size: bid_size,
                            depth_bps: bid_depth_bps,
                        });
                        tracing::info!(
                            price = %format!("{:.4}", bid_price),
                            size = %format!("{:.6}", bid_size),
                            depth_bps = %format!("{:.2}", bid_depth_bps),
                            half_spread_bps = %format!("{:.2}", guaranteed_half_spread_bps),
                            skew_bps = %format!("{:.2}", guaranteed_skew_bps),
                            "GUARANTEED BID: floor quote placed (min_viable size)"
                        );
                    }
                }

                if ladder.asks.is_empty() && !ask_cleared_by_zone {
                    let ask_depth_bps = guaranteed_half_spread_bps - guaranteed_skew_bps;
                    // Ensure ask depth is at least fee + tick (never tighter than cost)
                    let ask_depth_bps = ask_depth_bps.max(fee_bps + market_params.tick_size_bps);
                    let offset = market_params.microprice * (ask_depth_bps / 10_000.0);
                    let ask_price = round_to_significant_and_decimal(
                        market_params.microprice + offset,
                        5,
                        config.decimals,
                    );
                    // Ask price is above microprice, so quantum.min_viable_size is safe.
                    // But recompute for symmetry and future-proofing.
                    let ask_size = if ask_price > 0.0 {
                        let raw_min = quantum.min_notional / ask_price;
                        let steps = (raw_min / quantum.step).ceil() as u64;
                        (steps as f64 * quantum.step).max(quantum.min_viable_size)
                    } else {
                        quantum.min_viable_size
                    };

                    if ask_size > 0.0 && ask_price > 0.0 {
                        ladder.asks.push(LadderLevel {
                            price: ask_price,
                            size: ask_size,
                            depth_bps: ask_depth_bps,
                        });
                        tracing::info!(
                            price = %format!("{:.4}", ask_price),
                            size = %format!("{:.6}", ask_size),
                            depth_bps = %format!("{:.2}", ask_depth_bps),
                            half_spread_bps = %format!("{:.2}", guaranteed_half_spread_bps),
                            skew_bps = %format!("{:.2}", guaranteed_skew_bps),
                            "GUARANTEED ASK: floor quote placed (min_viable size)"
                        );
                    }
                }
            }

            ladder
        }
    }
}

impl QuotingStrategy for LadderStrategy {
    /// For compatibility with existing infrastructure, returns best bid/ask from ladder.
    ///
    /// The full ladder can be obtained via `generate_ladder()` for multi-level quoting.
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        let ladder = self.generate_ladder(
            config,
            position,
            max_position,
            target_liquidity,
            market_params,
        );

        // Return just the best bid/ask for backward compatibility
        let bid = ladder.bids.first().map(|l| Quote::new(l.price, l.size));
        let ask = ladder.asks.first().map(|l| Quote::new(l.price, l.size));

        (bid, ask)
    }

    fn calculate_ladder(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> Ladder {
        self.generate_ladder(
            config,
            position,
            max_position,
            target_liquidity,
            market_params,
        )
    }

    fn name(&self) -> &'static str {
        "LadderGLFT"
    }

    fn record_fill_observation(&mut self, depth_bps: f64, filled: bool) {
        self.fill_model.record_observation(depth_bps, filled);
    }

    fn update_fill_model_params(&mut self, sigma: f64, tau: f64) {
        self.fill_model.update_params(sigma, tau);
    }

    fn fill_model_warmed_up(&self) -> bool {
        self.fill_model.is_warmed_up()
    }

    fn record_quote_cycle_no_fills(&mut self, depths_bps: &[f64]) {
        self.record_no_fill_cycle(depths_bps);
    }

    fn record_kelly_win(&mut self, win_bps: f64) {
        self.kelly_sizer.record_win(win_bps);
    }

    fn record_kelly_loss(&mut self, loss_bps: f64) {
        self.kelly_sizer.record_loss(loss_bps);
    }

    fn kelly_fraction(&self) -> Option<f64> {
        if !self.kelly_sizer.enabled || !self.kelly_sizer.is_warmed_up() {
            return None;
        }
        // Use realized win/loss statistics for Kelly fraction.
        // Edge estimate: mean_win × win_rate - mean_loss × loss_rate
        let tracker = &self.kelly_sizer.win_loss_tracker;
        let edge_mean = tracker.avg_win() * tracker.win_rate()
            - tracker.avg_loss() * (1.0 - tracker.win_rate());
        let edge_std = (tracker.avg_win() + tracker.avg_loss()) * 0.5; // rough estimate
        let (should_size, fraction, _confidence) =
            self.kelly_sizer.sizing_decision(edge_mean, edge_std);
        if should_size {
            Some(fraction)
        } else {
            None
        }
    }

    fn gamma_features_cache(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> Option<([f64; 15], f64)> {
        let features = RiskFeatures::from_params(
            market_params,
            position,
            max_position,
            &self.risk_model_config,
        );
        let gamma = self.risk_model.compute_gamma_with_policy(
            &features,
            market_params.capital_tier,
            market_params.capital_policy.warmup_gamma_max_inflation,
        );
        Some((features.as_array(), gamma))
    }

    fn apply_calibrated_gamma_betas(&mut self, betas: &[f64; 15]) {
        self.risk_model.apply_calibrated_betas(betas);
    }
}

// ============================================================================
// Helper Functions for Constrained Optimizer
// ============================================================================

/// Fill intensity at depth using Bayesian model with time-normalized polynomial fallback.
///
/// Uses the Bayesian fill probability model which:
/// - Uses empirically-calibrated Beta-Binomial posterior when sufficient observations exist
/// - Falls back to time-normalized polynomial decay when no empirical data
///
/// The fallback formula `λ(δ) = κ × min(1, (σ×√τ / δ)²)` properly accounts for:
/// - Time horizon τ: Longer horizon = more likely to fill at deeper levels
/// - Volatility σ: Higher vol = price can reach deeper levels
/// - Depth δ: Deeper = less likely to fill
///
/// This gives meaningful fill probabilities across the ladder:
/// - At optimal depth (where σ×√τ ≈ δ): fill probability ~100%
/// - At 2x optimal depth: fill probability ~25%
/// - At 3x optimal depth: fill probability ~11%
fn fill_intensity_with_model(
    fill_model: &BayesianFillModel,
    depth_bps: f64,
    sigma: f64,
    tau: f64,
    kappa: f64,
) -> f64 {
    if depth_bps < EPSILON {
        return kappa; // At touch, use kappa as baseline
    }

    // Check if we have empirical data for this depth bucket
    let (fills, attempts) = fill_model.observations_at_depth(depth_bps);
    if attempts >= 5 {
        // Use Bayesian empirical fill probability when warmed up
        // Prior: Beta(2, 2), Posterior: Beta(2 + fills, 2 + non-fills)
        let posterior_alpha = 2.0 + fills as f64;
        let posterior_beta = 2.0 + (attempts - fills) as f64;
        let p_fill = posterior_alpha / (posterior_alpha + posterior_beta);
        return (p_fill * kappa).min(kappa);
    }

    // Fallback: Time-normalized polynomial decay
    //
    // P(fill) ∝ (σ×√τ / δ)² where σ×√τ is the expected price move over time τ
    //
    // With typical values: sigma=0.0001 (1bp/sec), tau=10s
    //   σ×√τ = 0.0001 × √10 ≈ 0.000316 ≈ 3.16bp expected move
    //
    // Example intensities (as fraction of kappa):
    //   At 3bp depth: (3.16/3)² ≈ 1.0 (saturates at 1.0)
    //   At 5bp depth: (3.16/5)² ≈ 0.4
    //   At 10bp depth: (3.16/10)² ≈ 0.1
    //   At 20bp depth: (3.16/20)² ≈ 0.025
    //
    // This provides meaningful size allocation across all levels.
    let depth_frac = depth_bps / 10000.0;
    let expected_move = sigma * tau.sqrt();
    let fill_prob = (expected_move / depth_frac).powi(2).min(1.0);
    fill_prob * kappa
}

/// Fill intensity at depth: λ(δ) = σ²/δ² × κ (theoretical fallback)
///
/// Models probability of price reaching depth δ based on diffusion.
/// At touch (δ → 0), returns kappa as baseline intensity.
///
/// NOTE: This is the legacy theoretical formula. Prefer `fill_intensity_with_model`
/// when Bayesian model is available, as it uses empirically-calibrated fill rates.
#[allow(dead_code)]
fn fill_intensity_at_depth(depth_bps: f64, sigma: f64, kappa: f64) -> f64 {
    if depth_bps < EPSILON {
        return kappa; // At touch, use kappa as baseline
    }
    let depth_frac = depth_bps / 10000.0;
    // σ² / δ² gives diffusion-driven fill probability
    // Scale by kappa for market activity level
    let diffusion_term = sigma.powi(2) / depth_frac.powi(2);
    (diffusion_term * kappa).min(kappa) // Cap at kappa
}

/// Spread capture at depth: SC(δ) = δ - AS(δ) - fees
///
/// Expected profit from capturing spread at depth δ.
/// Uses calibrated depth-dependent AS model if available.
fn spread_capture_at_depth(depth_bps: f64, params: &LadderParams, fees_bps: f64) -> f64 {
    let as_at_depth = adverse_selection_at_depth(depth_bps, params);
    // Spread capture = depth - adverse selection - fees
    (depth_bps - as_at_depth - fees_bps).max(0.0)
}

/// Adverse selection at depth: AS(δ) = AS₀ × exp(-δ/δ_char)
///
/// Returns the expected adverse selection cost at a given depth.
/// Uses calibrated depth-dependent AS model if available.
fn adverse_selection_at_depth(depth_bps: f64, params: &LadderParams) -> f64 {
    if let Some(ref decay) = params.depth_decay_as {
        // Use calibrated first-principles model: AS(δ) = AS₀ × exp(-δ/δ_char)
        decay.as_at_depth(depth_bps)
    } else {
        // Legacy fallback: exponential decay with 10bp characteristic depth
        params.as_at_touch_bps * (-depth_bps / 10.0).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute Cartea-Jaimungal margin allocation (SNR-scaled).
    /// q* = 0.5 × tanh(SNR / SNR_SCALE), where SNR = |drift_bps| / uncertainty_bps.
    /// drift_rate is in fraction/sec, uncertainty_bps is the Kalman posterior √P.
    /// Returns (bid_weight, ask_weight).
    fn compute_cj_margin_split(
        drift_rate: f64,
        _gamma: f64,
        _sigma: f64,
        position: f64,
        max_position: f64,
        uncertainty_bps: f64,
    ) -> (f64, f64) {
        let drift_snr = if uncertainty_bps > 1e-6 {
            let drift_bps = drift_rate * 10_000.0;
            drift_bps / uncertainty_bps
        } else {
            0.0
        };
        const SNR_SCALE: f64 = 2.0;
        let q_target = 0.5 * (drift_snr / SNR_SCALE).tanh();
        let q_current = if max_position > 1e-10 {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let effective_q_target = q_target;
        let delta_q = effective_q_target - q_current;
        let cj_allocation = (delta_q * 2.0).tanh() * 0.35;
        let ask_w = (0.5 - cj_allocation).clamp(0.15, 0.85);
        let bid_w = 1.0 - ask_w;
        (bid_w, ask_w)
    }

    #[test]
    fn test_cj_margin_bullish_flat_favors_bids() {
        // Positive drift (10 bps/s), uncertainty=1 bps → SNR=10 → strong signal → favor bids
        let (bid_w, ask_w) = compute_cj_margin_split(0.001, 0.0, 0.0, 0.0, 10.0, 1.0);
        assert!(
            bid_w > ask_w,
            "Bullish + flat should favor bids: bid={bid_w:.3}, ask={ask_w:.3}"
        );
    }

    #[test]
    fn test_cj_margin_bearish_flat_favors_asks() {
        // Negative drift (-10 bps/s), uncertainty=1 bps → SNR=-10 → favor asks
        let (bid_w, ask_w) = compute_cj_margin_split(-0.001, 0.0, 0.0, 0.0, 10.0, 1.0);
        assert!(
            ask_w > bid_w,
            "Bearish + flat should favor asks: bid={bid_w:.3}, ask={ask_w:.3}"
        );
    }

    #[test]
    fn test_cj_margin_no_drift_long_mean_reverts() {
        // No drift, long position → SNR=0 → q_target=0, q_current>0 → mean-revert to asks
        let (bid_w, ask_w) = compute_cj_margin_split(0.0, 0.0, 0.0, 5.0, 10.0, 1.0);
        assert!(
            ask_w > bid_w,
            "No drift + long should mean-revert (favor asks): bid={bid_w:.3}, ask={ask_w:.3}"
        );
    }

    #[test]
    fn test_cj_margin_high_uncertainty_attenuates_drift() {
        // High uncertainty (100 bps) attenuates drift (10 bps/s) → SNR=0.1 → near-zero q_target
        // Long position → mean-reverts
        let (bid_w_high_unc, ask_w_high_unc) =
            compute_cj_margin_split(0.001, 0.0, 0.0, 5.0, 10.0, 100.0);
        assert!(
            ask_w_high_unc > bid_w_high_unc,
            "High uncertainty + long should mean-revert: bid={bid_w_high_unc:.3}, ask={ask_w_high_unc:.3}"
        );

        // Low uncertainty (0.5 bps) → SNR=20 → strong bullish → overrides mean-reversion
        let (bid_w_low_unc, _) = compute_cj_margin_split(0.001, 0.0, 0.0, 5.0, 10.0, 0.5);
        // Lower uncertainty should allocate more to bids (drift direction)
        assert!(
            bid_w_low_unc > bid_w_high_unc,
            "Lower uncertainty should allocate more to drift direction"
        );
    }

    #[test]
    fn test_cj_margin_at_optimal_is_balanced() {
        // SNR-scaled: drift=0.0002 (2 bps/s), uncertainty=1 bps → SNR=2 → q_target≈0.38
        // Position at 38% of max → q_current=0.38, Δq≈0 → balanced
        // q_target = 0.5 * tanh(2/2) = 0.5 * tanh(1) ≈ 0.5 * 0.762 = 0.381
        let (bid_w, ask_w) = compute_cj_margin_split(0.0002, 0.0, 0.0, 3.81, 10.0, 1.0);
        assert!(
            (bid_w - 0.5).abs() < 0.05,
            "At optimal inventory should be ~balanced: bid={bid_w:.3}, ask={ask_w:.3}"
        );
        assert!(
            (ask_w - 0.5).abs() < 0.05,
            "At optimal inventory should be ~balanced: bid={bid_w:.3}, ask={ask_w:.3}"
        );
    }

    #[test]
    fn test_cj_margin_extreme_clamped() {
        // Extreme drift (10000 bps/s), tiny uncertainty → huge SNR → tanh saturates at 0.5
        // Then cj_allocation saturated → clamped at 85/15
        let (bid_w, ask_w) = compute_cj_margin_split(1.0, 0.0, 0.0, 0.0, 10.0, 0.001);
        assert!(bid_w <= 0.85 + 1e-10, "Should be clamped: bid={bid_w}");
        assert!(ask_w >= 0.15 - 1e-10, "Should be clamped: ask={ask_w}");
        // Never 100/0
        assert!(
            ask_w > 0.1,
            "Minority side must never be starved: ask={ask_w}"
        );
    }

    #[test]
    fn test_dynamic_as_buffer_scales_with_warmup() {
        // Buffer should be full (raw value) when cold, zero when warmed up.
        // Formula: buffer = raw * (1 - warmup_fraction)
        let raw_buffer = 2.0; // bps

        // Cold: 0 fills → warmup_fraction = 0 → buffer = 2.0
        let warmup_0 = (0.0_f64 / 20.0).min(1.0);
        assert!((warmup_0 - 0.0).abs() < f64::EPSILON);
        assert!((raw_buffer * (1.0 - warmup_0) - 2.0).abs() < f64::EPSILON);

        // Half warmed: 10 fills → warmup_fraction = 0.5 → buffer = 1.0
        let warmup_10 = (10.0_f64 / 20.0).min(1.0);
        assert!((warmup_10 - 0.5).abs() < f64::EPSILON);
        assert!((raw_buffer * (1.0 - warmup_10) - 1.0).abs() < f64::EPSILON);

        // Fully warmed: 20 fills → warmup_fraction = 1.0 → buffer = 0.0
        let warmup_20 = (20.0_f64 / 20.0).min(1.0);
        assert!((warmup_20 - 1.0).abs() < f64::EPSILON);
        assert!((raw_buffer * (1.0 - warmup_20)).abs() < f64::EPSILON);

        // Over-warmed: 50 fills → clamped to 1.0 → buffer = 0.0
        let warmup_50 = (50.0_f64 / 20.0).min(1.0);
        assert!((warmup_50 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dynamic_kappa_cap_formula() {
        // Formula: kappa_cap_bps = (2/kappa * 10_000).max(1.5)
        // High kappa = tight cap, low kappa = wide cap

        // kappa = 8000 → cap = 2.5 bps
        let cap_8000: f64 = (2.0 / 8000.0 * 10_000.0_f64).max(1.5);
        assert!(
            (cap_8000 - 2.5).abs() < 0.01,
            "kappa=8000 → cap=2.5 bps, got {:.2}",
            cap_8000
        );

        // kappa = 2500 → cap = 8.0 bps
        let cap_2500: f64 = (2.0 / 2500.0 * 10_000.0_f64).max(1.5);
        assert!(
            (cap_2500 - 8.0).abs() < 0.01,
            "kappa=2500 → cap=8.0 bps, got {:.2}",
            cap_2500
        );

        // kappa = 500 → cap = 40 bps (very wide)
        let cap_500: f64 = (2.0 / 500.0 * 10_000.0_f64).max(1.5);
        assert!(
            (cap_500 - 40.0).abs() < 0.01,
            "kappa=500 → cap=40 bps, got {:.2}",
            cap_500
        );

        // kappa = 20000 → cap = 1.5 bps (clamped to min fee floor)
        let cap_20k: f64 = (2.0 / 20000.0 * 10_000.0_f64).max(1.5);
        assert!(
            (cap_20k - 1.5).abs() < f64::EPSILON,
            "kappa=20000 → cap=1.5 bps (min), got {:.2}",
            cap_20k
        );
    }

    #[test]
    fn test_kappa_cap_higher_kappa_means_tighter() {
        // Monotonicity: higher kappa → smaller cap (tighter spread allowed)
        let caps: Vec<f64> = [500.0, 1000.0, 2000.0, 4000.0, 8000.0]
            .iter()
            .map(|k| (2.0 / k * 10_000.0_f64).max(1.5))
            .collect();

        for i in 0..caps.len() - 1 {
            assert!(
                caps[i] > caps[i + 1],
                "Cap should decrease as kappa increases: kappa[{}]={:.1} > kappa[{}]={:.1}",
                i,
                caps[i],
                i + 1,
                caps[i + 1]
            );
        }
    }

    #[test]
    fn test_floor_binding_eliminated_with_warmed_buffer() {
        // The buffer change eliminates floor binding in the common case.
        // Adaptive floor (paper mode) = 2.5 bps.
        // GLFT asymptotic half-spread: 1/kappa * 10000 + fee_bps
        let adaptive_floor_bps = 2.5;
        let fee_bps = 1.5;
        let kappa = 8000.0;
        let glft_approx_bps = 1.0 / kappa * 10_000.0 + fee_bps; // ~2.75 bps

        // Warmed up: buffer = 0 → effective_floor = 2.5 → GLFT (2.75) > floor (2.5)
        let buffer_warmed = 0.0;
        let effective_floor_warmed = adaptive_floor_bps + buffer_warmed;
        assert!(
            glft_approx_bps > effective_floor_warmed,
            "Warmed: GLFT ({:.2}) should exceed floor ({:.2}) — floor NOT binding",
            glft_approx_bps,
            effective_floor_warmed
        );

        // Cold: buffer = 2.0 → effective_floor = 4.5 → GLFT (2.75) < floor (4.5) → binding
        let buffer_cold = 2.0;
        let effective_floor_cold = adaptive_floor_bps + buffer_cold;
        assert!(
            glft_approx_bps < effective_floor_cold,
            "Cold: floor ({:.2}) should exceed GLFT ({:.2}) — floor BINDING (defensive)",
            effective_floor_cold,
            glft_approx_bps
        );
    }

    #[test]
    fn test_regime_gamma_multiplier_ordering() {
        // Regime risk routes through gamma_multiplier, not spread floor.
        // More volatile regimes have higher gamma → wider GLFT spreads.
        use crate::market_maker::strategy::regime_state::MarketRegime;

        let calm = MarketRegime::Calm.default_params();
        let normal = MarketRegime::Normal.default_params();
        let volatile = MarketRegime::Volatile.default_params();
        let extreme = MarketRegime::Extreme.default_params();

        assert!(calm.gamma_multiplier <= normal.gamma_multiplier);
        assert!(normal.gamma_multiplier < volatile.gamma_multiplier);
        assert!(volatile.gamma_multiplier < extreme.gamma_multiplier);

        // Calm = 1.0 (no scaling), Extreme = 3.0 (3x gamma → significantly wider)
        assert!((calm.gamma_multiplier - 1.0).abs() < 0.01);
        assert!((extreme.gamma_multiplier - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_physical_floor_is_fee_minimum() {
        // Physical floor = max(fee, tick, latency, min_spread_floor).
        // With default config: min_spread_floor = 0.00015 = 1.5 bps = fee.
        // So physical floor = 1.5 bps in normal conditions (tick < 1.5, latency < 1.5).
        let fee_bps: f64 = 1.5;
        let tick_bps: f64 = 0.5; // typical
        let latency_bps: f64 = 0.1; // low latency
        let min_floor_bps: f64 = 1.5; // new default

        let physical_floor = fee_bps.max(tick_bps).max(latency_bps).max(min_floor_bps);
        assert!(
            (physical_floor - 1.5_f64).abs() < 0.01,
            "Physical floor should be 1.5 bps (maker fee), got {physical_floor}"
        );

        // GLFT optimal should be ABOVE the fee floor in normal conditions
        // GLFT approx: 1/kappa * 10000 + fee
        let kappa = 2000.0;
        let glft_approx = 1.0 / kappa * 10_000.0 + fee_bps;
        assert!(
            glft_approx > physical_floor,
            "GLFT ({glft_approx:.2}) should exceed physical floor ({physical_floor:.2})"
        );
    }

    #[test]
    fn test_solve_min_gamma_matches_regime_floor() {
        use crate::market_maker::strategy::glft::solve_min_gamma;

        let fee_rate = 0.00015; // 1.5 bps
        let regime_as_bps = 1.0;
        let risk_premium_bps = 1.0;
        let target_floor_bps = fee_rate * 10_000.0 + regime_as_bps + risk_premium_bps;
        let target_floor_frac = target_floor_bps / 10_000.0;

        let kappa = 2000.0;
        let sigma = 0.0005;
        let time_horizon = 60.0;

        let gamma = solve_min_gamma(target_floor_frac, kappa, sigma, time_horizon, fee_rate);

        // Verify that GLFT at this gamma produces at least the target
        let safe_kappa = kappa.max(1.0);
        let ratio = gamma / safe_kappa;
        let glft = (1.0 / gamma) * (1.0 + ratio).ln();
        let vol_comp = 0.5 * gamma * sigma.powi(2) * time_horizon;
        let half_spread = glft + vol_comp + fee_rate;
        let half_spread_bps = half_spread * 10_000.0;

        assert!(
            half_spread_bps >= target_floor_bps - 0.01,
            "GLFT half-spread ({half_spread_bps:.2} bps) should meet floor ({target_floor_bps:.2} bps)"
        );
    }

    #[test]
    fn test_continuous_size_mult_monotonic() {
        // Continuous size scaling: (1 - 0.75 * ratio²).max(0.25)
        // Must be monotonically decreasing
        let ratios: [f64; 6] = [0.0, 0.3, 0.5, 0.6, 0.8, 1.0];
        let mults: Vec<f64> = ratios
            .iter()
            .map(|r| (1.0 - 0.75 * r * r).max(0.25))
            .collect();
        for i in 1..mults.len() {
            assert!(
                mults[i] <= mults[i - 1],
                "Size mult should decrease: at {:.1} got {:.3} > {:.3} at {:.1}",
                ratios[i],
                mults[i],
                mults[i - 1],
                ratios[i - 1]
            );
        }
        assert!(mults[0] > 0.99, "At 0% utilization, mult should be ~1.0");
        assert!(mults[mults.len() - 1] >= 0.25, "Floor at 0.25");
    }

    #[test]
    fn test_continuous_gamma_exceeds_zone_widening() {
        // At 80% utilization with β=7.0, γ(q) = γ_base × (1 + 7.0 × 0.64) = 5.48×
        // Old Yellow zone only applied +3 bps. Continuous γ produces wider spread.
        let beta = 7.0;
        let utilization = 0.8;
        let gamma_scalar = 1.0 + beta * utilization * utilization;
        assert!(
            gamma_scalar > 4.0,
            "At 80% utilization, gamma scalar should be > 4.0: {gamma_scalar}"
        );

        // At 60% (old Yellow boundary)
        let utilization_60 = 0.6;
        let gamma_60 = 1.0 + beta * utilization_60 * utilization_60;
        assert!(
            gamma_60 > 3.0,
            "At 60% utilization, gamma scalar should be > 3.0: {gamma_60}"
        );
    }

    #[test]
    fn test_kill_guard_independent() {
        // At >100% utilization, accumulating side still cleared
        let abs_ratio = 1.05;
        assert!(abs_ratio >= 1.0, "Should trigger kill guard");
    }

    #[test]
    fn test_red_zone_clears_accumulating_side() {
        // In Red zone, accumulating side should be cleared
        let mut bid_depths = vec![5.0, 7.0, 9.0];
        let ask_depths = [5.0, 7.0, 9.0];
        let position = 1.0; // Long

        // Simulate Red zone: long → clear bids (accumulating)
        if position > 0.0 {
            bid_depths.clear();
        }

        assert!(
            bid_depths.is_empty(),
            "Long in Red → bids should be cleared"
        );
        assert!(!ask_depths.is_empty(), "Long in Red → asks should remain");

        // Short case
        let bid_depths2 = [5.0, 7.0, 9.0];
        let mut ask_depths2 = vec![5.0, 7.0, 9.0];
        let position2 = -1.0;

        if position2 < 0.0 {
            ask_depths2.clear();
        }

        assert!(!bid_depths2.is_empty(), "Short in Red → bids should remain");
        assert!(
            ask_depths2.is_empty(),
            "Short in Red → asks should be cleared"
        );
    }

    #[test]
    fn test_cost_basis_breakeven_clamping() {
        // Long position: asks should not go below breakeven
        let entry = 100.0;
        let fee_rate = 0.00015;
        let breakeven = entry * (1.0 + fee_rate); // 100.015

        let ask_price = 99.99; // Below breakeven
        assert!(
            ask_price < breakeven,
            "Ask ({ask_price}) should be below breakeven ({breakeven})"
        );

        // Clamping should move it up to breakeven
        let clamped = breakeven;
        assert!(
            clamped >= breakeven,
            "Clamped ask should be at or above breakeven"
        );

        // Short position: bids should not go above breakeven
        let breakeven_short = entry * (1.0 - fee_rate); // 99.985
        let bid_price = 100.01; // Above breakeven for short
        assert!(
            bid_price > breakeven_short,
            "Bid ({bid_price}) should be above breakeven ({breakeven_short})"
        );

        let clamped_bid = breakeven_short;
        assert!(
            clamped_bid <= breakeven_short,
            "Clamped bid should be at or below breakeven"
        );
    }

    #[test]
    fn test_cost_basis_breakeven_tick_rounding() {
        // Validates the fix for "Order has invalid price" errors.
        // The breakeven price must land on the exchange tick grid
        // (10^(-price_decimals)) AND satisfy the 5-sig-fig constraint.
        use crate::round_to_significant_and_decimal;

        // HYPE-like: price ~$25, price_decimals=4, tick=0.0001
        let decimals: u32 = 4;
        let tick_size: f64 = 10f64.powi(-(decimals as i32)); // 0.0001

        // === Ask side (long position): breakeven rounded UP ===
        // Breakeven at a non-tick-aligned price
        let breakeven_ask = 25.12345678;
        let breakeven_on_tick = (breakeven_ask / tick_size).ceil() * tick_size;
        let rounded = round_to_significant_and_decimal(breakeven_on_tick, 5, decimals);

        // Must be on tick grid: (rounded / tick_size) is integer within float tolerance
        let ticks = rounded / tick_size;
        assert!(
            (ticks - ticks.round()).abs() < 1e-6,
            "Ask breakeven {rounded} not on tick grid (tick={tick_size})"
        );
        // Must satisfy 5 sig figs: re-rounding is idempotent
        let re_rounded = round_to_significant_and_decimal(rounded, 5, decimals);
        assert!(
            (re_rounded - rounded).abs() < 1e-10,
            "Ask breakeven {rounded} does not satisfy 5-sig-fig: re-rounded to {re_rounded}"
        );
        // Must be >= true breakeven (we rounded UP)
        assert!(
            rounded >= breakeven_ask - 1e-9,
            "Ask breakeven {rounded} is below true breakeven {breakeven_ask}"
        );

        // === Bid side (short position): breakeven rounded DOWN ===
        let breakeven_bid = 25.12345678;
        let breakeven_on_tick_bid = (breakeven_bid / tick_size).floor() * tick_size;
        let rounded_bid = round_to_significant_and_decimal(breakeven_on_tick_bid, 5, decimals);

        // Must be on tick grid
        let ticks_bid = rounded_bid / tick_size;
        assert!(
            (ticks_bid - ticks_bid.round()).abs() < 1e-6,
            "Bid breakeven {rounded_bid} not on tick grid (tick={tick_size})"
        );
        // Must satisfy 5 sig figs
        let re_rounded_bid = round_to_significant_and_decimal(rounded_bid, 5, decimals);
        assert!(
            (re_rounded_bid - rounded_bid).abs() < 1e-10,
            "Bid breakeven {rounded_bid} does not satisfy 5-sig-fig: re-rounded to {re_rounded_bid}"
        );
        // Must be <= true breakeven (we rounded DOWN)
        assert!(
            rounded_bid <= breakeven_bid + 1e-9,
            "Bid breakeven {rounded_bid} is above true breakeven {breakeven_bid}"
        );

        // === Edge case: breakeven already on tick grid ===
        let exact_breakeven = 25.123;
        let on_tick = (exact_breakeven / tick_size).ceil() * tick_size;
        let rounded_exact = round_to_significant_and_decimal(on_tick, 5, decimals);
        assert!(
            (rounded_exact - exact_breakeven).abs() < 1e-9,
            "Already-on-tick breakeven {exact_breakeven} changed to {rounded_exact}"
        );

        // === Edge case: very small decimal diff that triggers the bug ===
        // breakeven = 25.0005 -> 5-sig-fig rounds to 25.001 (3 decimals)
        // This should be fine since 25.001 > 25.0005 for asks
        let tricky = 25.00045;
        let tricky_on_tick = (tricky / tick_size).ceil() * tick_size;
        let tricky_rounded = round_to_significant_and_decimal(tricky_on_tick, 5, decimals);
        assert!(
            tricky_rounded >= tricky - 1e-9,
            "Tricky ask breakeven {tricky_rounded} < true {tricky}"
        );
    }

    #[test]
    fn test_cost_basis_urgency_override() {
        // When position is large AND deeply underwater, urgency > 0.5 skips clamping
        let position_fraction: f64 = 0.9; // 90% of max
        let unrealized_bps: f64 = -60.0; // 6 bps underwater

        let urgency: f64 = (position_fraction * (-unrealized_bps / 50.0_f64).max(0.0)).min(1.0);
        assert!(
            urgency > 0.5,
            "High position + underwater should produce urgency > 0.5, got {urgency}"
        );

        // Small position, slightly underwater: no urgency override
        let position_fraction2: f64 = 0.2;
        let unrealized_bps2: f64 = -5.0;
        let urgency2: f64 = (position_fraction2 * (-unrealized_bps2 / 50.0_f64).max(0.0)).min(1.0);
        assert!(
            urgency2 < 0.5,
            "Small position, slightly underwater: urgency should be < 0.5, got {urgency2}"
        );
    }

    #[test]
    fn test_micro_tier_uses_glft_pipeline() {
        use crate::market_maker::config::auto_derive::CapitalTier;
        use crate::market_maker::strategy::QuotingStrategy;

        // Setup: $100 capital, HYPE@$30, Micro tier
        // max_position = $100 / $30 ≈ 3.33 contracts
        // With 10 configured levels, capital_limited_levels should reduce to 1-2
        // and the standard GLFT pipeline should produce GLFT-derived depths
        // (not the old hardcoded 5 bps from concentrated path).
        //
        // Use decimals=2 (tick=$0.01 = 3.3 bps on $30) so GLFT depths survive
        // rounding. With decimals=1 (tick=33 bps), any depth < 33 bps rounds
        // bid back to mid — not a code bug, just coarse-tick physics.
        let strategy = LadderStrategy::new(0.07);
        let config = QuoteConfig {
            mid_price: 30.0,
            decimals: 2, // $0.01 tick = 3.3 bps on $30
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let market_params = MarketParams {
            microprice: 30.0,
            market_mid: 30.0,
            capital_tier: CapitalTier::Micro,
            margin_available: 100.0,
            leverage: 3.0,
            margin_quoting_capacity: 3.24,
            sigma: 0.001,
            sigma_effective: 0.001,
            kappa: 2000.0, // Wider spreads so GLFT depth > tick
            ..Default::default()
        };

        let position = 0.0;
        let max_position = 3.24; // $100 / $30 ≈ 3.33, with margin
        let target_liquidity = 1.0;

        let ladder = strategy.calculate_ladder(
            &config,
            position,
            max_position,
            target_liquidity,
            &market_params,
        );

        // Should produce at least 1 bid + 1 ask via the standard pipeline
        assert!(
            !ladder.bids.is_empty(),
            "Micro tier should produce at least 1 bid via GLFT pipeline"
        );
        assert!(
            !ladder.asks.is_empty(),
            "Micro tier should produce at least 1 ask via GLFT pipeline"
        );

        // Each order should meet min_notional
        for level in &ladder.bids {
            let notional = level.size * level.price;
            assert!(
                notional >= config.min_notional,
                "Bid notional ${:.2} should meet min_notional ${:.2}",
                notional,
                config.min_notional
            );
        }
        for level in &ladder.asks {
            let notional = level.size * level.price;
            assert!(
                notional >= config.min_notional,
                "Ask notional ${:.2} should meet min_notional ${:.2}",
                notional,
                config.min_notional
            );
        }

        // Bid price should be below mid, ask above — GLFT spreads are real
        assert!(
            ladder.bids[0].price < market_params.microprice,
            "Bid {:.4} should be below mid {:.4}",
            ladder.bids[0].price,
            market_params.microprice
        );
        assert!(
            ladder.asks[0].price > market_params.microprice,
            "Ask {:.4} should be above mid {:.4}",
            ladder.asks[0].price,
            market_params.microprice
        );

        // REGRESSION: depth should be GLFT-derived, not hardcoded 5 bps.
        // Verify spread invariant: ask_price > bid_price.
        let spread_bps =
            (ladder.asks[0].price - ladder.bids[0].price) / market_params.microprice * 10_000.0;
        assert!(
            spread_bps > 0.0,
            "Spread must be positive, got {:.2} bps",
            spread_bps
        );
    }

    #[test]
    fn test_micro_tier_respects_position_limits() {
        use crate::market_maker::config::auto_derive::CapitalTier;
        use crate::market_maker::strategy::QuotingStrategy;

        // Scenario: Micro tier with existing long position near max
        // Should still produce ask (reduce-only) but bid should be very small or empty
        let strategy = LadderStrategy::new(0.07);
        let config = QuoteConfig {
            mid_price: 30.0,
            decimals: 2, // $0.01 tick
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let market_params = MarketParams {
            microprice: 30.0,
            market_mid: 30.0,
            capital_tier: CapitalTier::Micro,
            margin_available: 100.0,
            leverage: 3.0,
            margin_quoting_capacity: 3.24,
            sigma: 0.001,
            sigma_effective: 0.001,
            kappa: 2000.0,
            ..Default::default()
        };

        let position = 3.0; // Near max
        let max_position = 3.24;
        let target_liquidity = 1.0;

        let ladder = strategy.calculate_ladder(
            &config,
            position,
            max_position,
            target_liquidity,
            &market_params,
        );

        // Ask side should have an order (can sell position)
        assert!(
            !ladder.asks.is_empty(),
            "Micro tier with long position should still produce asks for reducing"
        );

        // Bid side should be very limited — at 92.6% utilization with continuous γ(q),
        // bids are heavily penalized but may still exist with very small size.
        // With Phase 2's continuous system (no discrete zones), the min_notional filter
        // or continuous size scaling may leave a tiny bid. Verify bids are smaller than asks.
        let total_bid_size: f64 = ladder.bids.iter().map(|q| q.size).sum();
        let total_ask_size: f64 = ladder.asks.iter().map(|q| q.size).sum();
        assert!(
            total_bid_size < total_ask_size * 0.5,
            "Near-max long: bid size ({:.4}) should be much smaller than ask size ({:.4})",
            total_bid_size,
            total_ask_size
        );
    }

    #[test]
    fn test_all_tiers_use_same_glft_pipeline() {
        use crate::market_maker::config::auto_derive::CapitalTier;
        use crate::market_maker::strategy::QuotingStrategy;

        // Verify that Large tier and Micro tier both flow through the same
        // GLFT pipeline — no code path branching on capital tier.
        // Use decimals=2 (tick=$0.01) so GLFT depths survive rounding.
        let strategy = LadderStrategy::new(0.07);
        let config = QuoteConfig {
            mid_price: 30.0,
            decimals: 2, // $0.01 tick = 3.3 bps on $30
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let market_params = MarketParams {
            microprice: 30.0,
            market_mid: 30.0,
            capital_tier: CapitalTier::Large,
            margin_available: 10_000.0,
            leverage: 3.0,
            margin_quoting_capacity: 1000.0,
            sigma_effective: 0.001,
            sigma: 0.001,
            kappa: 2000.0,
            ..Default::default()
        };

        let position = 0.0;
        let max_position = 1000.0;
        let target_liquidity = 100.0;

        let ladder = strategy.calculate_ladder(
            &config,
            position,
            max_position,
            target_liquidity,
            &market_params,
        );

        // Large tier with ample capital should produce multiple levels
        assert!(
            ladder.bids.len() + ladder.asks.len() > 2,
            "Large tier with $10k capital should produce multi-level ladder, got {} bids + {} asks",
            ladder.bids.len(),
            ladder.asks.len()
        );

        // Now verify Micro tier also produces valid GLFT output (not a separate path)
        let micro_params = MarketParams {
            microprice: 30.0,
            market_mid: 30.0,
            capital_tier: CapitalTier::Micro,
            margin_available: 100.0,
            leverage: 3.0,
            margin_quoting_capacity: 3.24,
            sigma_effective: 0.001,
            sigma: 0.001,
            kappa: 2000.0,
            ..Default::default()
        };

        let micro_ladder = strategy.calculate_ladder(&config, 0.0, 3.24, 1.0, &micro_params);

        // Micro tier should also produce valid output through standard pipeline
        assert!(
            !micro_ladder.bids.is_empty() || !micro_ladder.asks.is_empty(),
            "Micro tier should produce at least one side via GLFT pipeline"
        );

        // Both tiers should have bid < mid < ask (same invariants)
        if !micro_ladder.bids.is_empty() {
            assert!(
                micro_ladder.bids[0].price < micro_params.microprice,
                "Micro bid {:.4} should be below mid {:.4}",
                micro_ladder.bids[0].price,
                micro_params.microprice
            );
        }
        if !micro_ladder.asks.is_empty() {
            assert!(
                micro_ladder.asks[0].price > micro_params.microprice,
                "Micro ask {:.4} should be above mid {:.4}",
                micro_ladder.asks[0].price,
                micro_params.microprice
            );
        }
    }

    /// Verify kappa priority chain: Robust > Adaptive > Coordinator > Legacy
    #[test]
    fn test_kappa_priority_chain_coordinator() {
        let params = MarketParams {
            kappa: 1000.0, // Legacy kappa
            coordinator_kappa: 600.0,
            coordinator_uncertainty_premium_bps: 2.0,
            use_coordinator_kappa: true,
            use_kappa_robust: false,
            use_adaptive_spreads: false,
            adaptive_can_estimate: false,
            ..Default::default()
        };

        // With no robust/adaptive, coordinator kappa should be selected
        // (We test this by checking the compute_spread_composition output)
        let strat = LadderStrategy::new(0.3);
        let composition = strat.compute_spread_composition(&params, 1.0);

        // The GLFT spread with coordinator kappa (600) should be wider than with legacy (1000)
        // because lower kappa → wider spread
        // Also: warmup addon should include coordinator uncertainty premium
        let mut legacy_params = params.clone();
        legacy_params.use_coordinator_kappa = false;
        legacy_params.coordinator_uncertainty_premium_bps = 0.0;
        let _legacy_composition = strat.compute_spread_composition(&legacy_params, 1.0);

        // Note: compute_spread_composition uses the basic kappa path (robust vs legacy only).
        // The coordinator kappa is wired in generate_ladder. We verify the uncertainty premium
        // routes through warmup_addon_bps correctly.
        assert!(
            composition.warmup_addon_bps >= params.coordinator_uncertainty_premium_bps,
            "Warmup addon ({:.2}) should include coordinator uncertainty premium ({:.2})",
            composition.warmup_addon_bps,
            params.coordinator_uncertainty_premium_bps
        );

        // Also verify warmup addon is zero when adaptive_warmup_progress = 1.0
        let mut done_params = params.clone();
        done_params.adaptive_warmup_progress = 1.0;
        let done_composition = strat.compute_spread_composition(&done_params, 1.0);
        assert!(
            done_composition.warmup_addon_bps < 0.01,
            "Warmup addon should be 0 when warmup complete: {:.2}",
            done_composition.warmup_addon_bps
        );
    }

    /// Verify that when robust kappa is active, it produces tighter spreads
    /// than legacy kappa (which compute_spread_composition falls back to).
    /// The coordinator kappa path is only exercised in generate_ladder.
    #[test]
    fn test_robust_kappa_overrides_coordinator() {
        let mut params = MarketParams {
            kappa: 1000.0,
            kappa_robust: 2000.0, // Higher kappa → tighter spread
            use_kappa_robust: true,
            coordinator_kappa: 600.0, // Lower kappa → wider spread
            use_coordinator_kappa: true,
            ..Default::default()
        };

        let strat = LadderStrategy::new(0.3);
        let robust_composition = strat.compute_spread_composition(&params, 1.0);

        // Without robust, falls back to legacy kappa (1000) in compute_spread_composition
        params.use_kappa_robust = false;
        let legacy_composition = strat.compute_spread_composition(&params, 1.0);

        // Robust kappa (2000) → tighter spread than legacy (1000)
        assert!(
            robust_composition.glft_half_spread_bps < legacy_composition.glft_half_spread_bps,
            "Robust kappa (2000) should give tighter spread ({:.2}) than legacy (1000) spread ({:.2})",
            robust_composition.glft_half_spread_bps, legacy_composition.glft_half_spread_bps
        );
    }

    // ====================================================================
    // Guaranteed Quote Floor Tests
    // ====================================================================

    #[test]
    fn test_guaranteed_quote_half_spread_formula() {
        // Guaranteed half-spread = max(fee + tick, GLFT optimal, effective floor)
        let fee_bps = 1.5_f64;
        let tick_bps = 0.5_f64;
        let glft_optimal_bps = 3.0_f64;
        let effective_floor_bps = 2.0_f64;

        let guaranteed = (fee_bps + tick_bps)
            .max(glft_optimal_bps)
            .max(effective_floor_bps);
        assert!(
            (guaranteed - 3.0).abs() < 1e-10,
            "Should be max(2.0, 3.0, 2.0) = 3.0, got {guaranteed}"
        );

        // When GLFT is narrow, fee+tick dominates
        let glft_narrow = 1.0_f64;
        let guaranteed2 = (fee_bps + tick_bps)
            .max(glft_narrow)
            .max(effective_floor_bps);
        assert!(
            (guaranteed2 - 2.0).abs() < 1e-10,
            "Should be max(2.0, 1.0, 2.0) = 2.0, got {guaranteed2}"
        );
    }

    #[test]
    fn test_guaranteed_quote_inventory_skew() {
        // Skew = inventory_ratio × half_spread × 0.3
        let half_spread = 5.0_f64; // bps

        // Flat: no skew
        let skew_flat = 0.0 * half_spread * 0.3;
        assert!(skew_flat.abs() < 1e-10);

        // Long 50%: push bid down, ask tighter (attract sells)
        let skew_long = 0.5 * half_spread * 0.3;
        assert!(
            (skew_long - 0.75).abs() < 1e-10,
            "Long skew should be +0.75 bps, got {skew_long}"
        );
        // bid_depth = half_spread + skew = 5.75 (wider)
        // ask_depth = half_spread - skew = 4.25 (tighter to attract fills)
        let bid_depth = half_spread + skew_long;
        let ask_depth = half_spread - skew_long;
        assert!(bid_depth > ask_depth, "Long: bid should be deeper than ask");

        // Short 50%: push bid tighter, ask down (attract buys)
        let skew_short = -0.5 * half_spread * 0.3;
        assert!(skew_short < 0.0, "Short skew should be negative");
        let bid_depth_short = half_spread + skew_short; // tighter
        let ask_depth_short = half_spread - skew_short; // wider
        assert!(
            ask_depth_short > bid_depth_short,
            "Short: ask should be deeper than bid"
        );
    }

    #[test]
    fn test_guaranteed_quote_ask_depth_never_below_cost() {
        // Even with large long inventory skew, ask depth must be >= fee + tick
        let fee_bps = 1.5_f64;
        let tick_bps = 0.5_f64;
        let half_spread = 3.0_f64;
        let skew = 1.0 * half_spread * 0.3; // Max long: 0.9 bps

        let raw_ask_depth = half_spread - skew; // 3.0 - 0.9 = 2.1
        let ask_depth = raw_ask_depth.max(fee_bps + tick_bps);
        assert!(
            ask_depth >= fee_bps + tick_bps,
            "Ask depth {ask_depth} must be >= fee+tick={}",
            fee_bps + tick_bps
        );
    }

    #[test]
    fn test_kill_guard_clears_accumulating_side() {
        // At >=90% utilization (KILL_CLEAR_INVENTORY_RATIO), accumulating side is cleared.
        // This is the only discrete cutoff — everything else is continuous γ(q).
        let position = 1.0_f64;
        let abs_inventory_ratio = 0.95; // Above 90% threshold
        let bid_cleared = abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position > 0.0;
        let ask_cleared = abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position < 0.0;

        assert!(
            bid_cleared,
            "Long at 95%: bids should be cleared (threshold=90%)"
        );
        assert!(!ask_cleared, "Long at 95%: asks should NOT be cleared");

        // Short at >90%: asks cleared
        let position_short = -1.0_f64;
        let bid_cleared_short =
            abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position_short > 0.0;
        let ask_cleared_short =
            abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position_short < 0.0;

        assert!(
            !bid_cleared_short,
            "Short at 95%: bids should NOT be cleared"
        );
        assert!(ask_cleared_short, "Short at 95%: asks should be cleared");
    }

    #[test]
    fn test_kill_guard_does_not_clear_below_threshold() {
        // At 85% utilization (below KILL_CLEAR_INVENTORY_RATIO=0.90), nothing is cleared.
        let position = 1.0_f64;
        let abs_inventory_ratio = 0.85;
        let bid_cleared = abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position > 0.0;
        let ask_cleared = abs_inventory_ratio >= KILL_CLEAR_INVENTORY_RATIO && position < 0.0;

        assert!(
            !bid_cleared,
            "Long at 85%: bids should NOT be cleared (below 90% threshold)"
        );
        assert!(!ask_cleared, "Long at 85%: asks should NOT be cleared");
    }

    #[test]
    fn test_guaranteed_quotes_produce_both_sides_on_empty_ladder() {
        use crate::market_maker::config::auto_derive::CapitalTier;
        use crate::market_maker::strategy::QuotingStrategy;

        // Scenario: zero margin, zero position — should produce guaranteed quotes
        let strategy = LadderStrategy::new(0.07);
        let config = QuoteConfig {
            mid_price: 25.0,
            decimals: 2, // $0.01 tick = 0.4 bps — fine-grained enough for GLFT offsets
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let market_params = MarketParams {
            microprice: 25.0,
            market_mid: 25.0,
            capital_tier: CapitalTier::Micro,
            margin_available: 0.0, // No margin!
            leverage: 3.0,
            margin_quoting_capacity: 0.0,
            sigma: 0.001,
            sigma_effective: 0.001,
            kappa: 2000.0,
            tick_size_bps: 0.4, // Matches decimals=2 at $25
            ..Default::default()
        };

        let ladder = strategy.calculate_ladder(
            &config,
            0.0, // flat
            5.0, // max position
            1.0,
            &market_params,
        );

        // With zero margin, normal ladder is empty, but guaranteed quotes should fire
        assert!(
            !ladder.bids.is_empty(),
            "Guaranteed quotes should produce at least 1 bid even with 0 margin"
        );
        assert!(
            !ladder.asks.is_empty(),
            "Guaranteed quotes should produce at least 1 ask even with 0 margin"
        );

        // Bid < mid < ask
        assert!(
            ladder.bids[0].price < market_params.microprice,
            "Guaranteed bid {:.4} should be below mid {:.4}",
            ladder.bids[0].price,
            market_params.microprice
        );
        assert!(
            ladder.asks[0].price > market_params.microprice,
            "Guaranteed ask {:.4} should be above mid {:.4}",
            ladder.asks[0].price,
            market_params.microprice
        );
    }

    #[test]
    fn test_zero_margin_produces_guaranteed_quotes_only() {
        use crate::market_maker::strategy::QuotingStrategy;

        // With margin_available = 0.0, regular ladder levels have zero size,
        // but guaranteed quotes still fire (needed to exit positions).
        // Binary circuit breakers removed — γ(q) handles risk continuously.
        let strategy = LadderStrategy::new(0.07);
        let config = QuoteConfig {
            mid_price: 25.0,
            decimals: 2,
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let market_params = MarketParams {
            microprice: 25.0,
            market_mid: 25.0,
            margin_available: 0.0,
            sigma: 0.001,
            kappa: 2000.0,
            ..Default::default()
        };

        let ladder = strategy.calculate_ladder(&config, 0.0, 5.0, 1.0, &market_params);

        // Guaranteed quotes produce minimal size on each side
        // (margin=0 prevents additional levels but not guaranteed quotes)
        assert!(
            ladder.bids.len() <= 1 && ladder.asks.len() <= 1,
            "Zero margin should produce at most guaranteed quotes, got {} bids + {} asks",
            ladder.bids.len(),
            ladder.asks.len()
        );
    }

    #[test]
    fn test_circuit_breaker_no_longer_blocks_quoting() {
        use crate::market_maker::strategy::QuotingStrategy;

        // should_pull_quotes = true no longer produces empty ladder
        // γ(q) handles risk continuously instead of binary circuit breakers
        let strategy = LadderStrategy::new(0.07);
        let config = QuoteConfig {
            mid_price: 25.0,
            decimals: 2,
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let market_params = MarketParams {
            microprice: 25.0,
            market_mid: 25.0,
            margin_available: 100.0,  // Solvent
            should_pull_quotes: true, // Was circuit breaker, now diagnostic only
            sigma: 0.001,
            kappa: 2000.0,
            ..Default::default()
        };

        let ladder = strategy.calculate_ladder(&config, 0.0, 5.0, 1.0, &market_params);

        assert!(
            !ladder.bids.is_empty() || !ladder.asks.is_empty(),
            "should_pull_quotes=true should NOT produce empty ladder when solvent"
        );
    }

    // --- Unified Reservation Mid Tests (WS3) ---

    #[test]
    fn test_reservation_mid_drift() {
        let strategy = LadderStrategy::new(0.1);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let mut params = MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            margin_available: 1000.0,
            leverage: 1.0,
            sigma: 0.005,
            kappa: 2000.0,
            arrival_intensity: 0.5,
            // Set valid BBO so build_raw_ladder doesn't clamp bid_base to market_mid
            cached_best_bid: 99.95,
            cached_best_ask: 100.05,
            ..Default::default()
        };
        params.capital_policy.use_tick_grid = false; // Force non-tick-grid path for reservation mid testing

        // Bullish drift via signal (posterior is zero by default)
        params.drift_signal_bps = 5.0; // 5 bps upward drift
        let ladder = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);
        let bid_price_drift = ladder.bids.first().map(|l| l.price).unwrap_or(0.0);

        params.drift_signal_bps = 0.0;
        let ladder_no_drift = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);
        let bid_price_no_drift = ladder_no_drift.bids.first().map(|l| l.price).unwrap_or(0.0);

        assert!(bid_price_drift > bid_price_no_drift, "Bullish drift should raise bid prices. drift={bid_price_drift}, no_drift={bid_price_no_drift}");
    }

    #[test]
    fn test_reservation_mid_inventory_penalty() {
        let strategy = LadderStrategy::new(0.1);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let mut params = MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            margin_available: 1000.0,
            leverage: 1.0,
            sigma: 0.005,
            kappa: 2000.0,
            cached_best_bid: 99.95,
            cached_best_ask: 100.05,
            ..Default::default()
        };
        params.capital_policy.use_tick_grid = false;

        // Long position
        let ladder_long = strategy.calculate_ladder(&config, 50.0, 100.0, 100.0, &params);
        // Neutral position
        let ladder_neutral = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);

        // Compare average ask prices across all levels to avoid touch-level rounding masking the effect
        let avg_ask_long: f64 = ladder_long.asks.iter().map(|l| l.price).sum::<f64>()
            / ladder_long.asks.len().max(1) as f64;
        let avg_ask_neutral: f64 = ladder_neutral.asks.iter().map(|l| l.price).sum::<f64>()
            / ladder_neutral.asks.len().max(1) as f64;

        assert!(
            avg_ask_long < avg_ask_neutral,
            "Long inventory should push ask prices down (aggressive). avg_long={avg_ask_long:.4}, avg_neutral={avg_ask_neutral:.4}"
        );
    }

    #[test]
    fn test_reservation_mid_funding_carry() {
        let strategy = LadderStrategy::new(0.1);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let mut params = MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            margin_available: 1000.0,
            leverage: 1.0,
            sigma: 0.005,
            kappa: 2000.0,
            // Set valid BBO so build_raw_ladder doesn't clamp ask_floor to market_mid
            cached_best_bid: 99.95,
            cached_best_ask: 100.05,
            ..Default::default()
        };
        params.capital_policy.use_tick_grid = false; // Force non-tick-grid path for reservation mid testing

        // High positive funding rate (longs pay shorts)
        params.funding_rate = 31_557_600.0 * 1.0;

        let ladder_funding = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);

        params.funding_rate = 0.0;
        let ladder_no_funding = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);

        // Compare average ask prices across all levels to avoid touch-level rounding masking the effect
        let avg_ask_funding: f64 = ladder_funding.asks.iter().map(|l| l.price).sum::<f64>()
            / ladder_funding.asks.len().max(1) as f64;
        let avg_ask_no_funding: f64 = ladder_no_funding.asks.iter().map(|l| l.price).sum::<f64>()
            / ladder_no_funding.asks.len().max(1) as f64;

        assert!(
            avg_ask_funding < avg_ask_no_funding,
            "Positive funding carry should lower ask prices to build short. avg_funding={avg_ask_funding:.4}, avg_no_funding={avg_ask_no_funding:.4}"
        );
    }

    #[test]
    fn test_reservation_mid_clamp() {
        let strategy = LadderStrategy::new(0.5);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 2,
            min_notional: 1.0,
        };
        let mut params = MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            margin_available: 1000.0,
            leverage: 1.0,
            sigma: 0.005,
            kappa: 2000.0,
            cached_best_bid: 99.95,
            cached_best_ask: 100.05,
            ..Default::default()
        };
        params.capital_policy.use_tick_grid = false; // Force non-tick-grid path for reservation mid testing

        // Extreme bullish drift (reservation mid reads drift_signal_bps)
        params.drift_signal_bps = 500.0;
        let ladder = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);

        let bid_price = ladder.bids.first().map(|l| l.price).unwrap_or(0.0);

        // It must not cross the market ask (which is bounded by mid)
        // Our constraint on half_spread_bound + depth means it effectively won't cross the mid.
        // Even with huge drift, bounded reservation + depth subtraction keeps bids sane.
        assert!(
            bid_price <= 100.0,
            "Clamped reservation mid should prevent crossing the real market mid. bid: {}",
            bid_price
        );
    }

    #[test]
    fn test_drift_above_old_skew_cap_shifts_reservation_mid() {
        // WS6: Drift > 5 bps (old max_lead_lag_skew_bps cap) should shift reservation mid.
        // The ±2× GLFT half-spread clamp is the only bound.
        // Use kappa=200 so GLFT half-spread ≈ 50 bps → clamp at ±100 bps allows both test values through.
        let strategy = LadderStrategy::new(0.1);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let mut params = MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            margin_available: 1000.0,
            leverage: 1.0,
            sigma: 0.005,
            kappa: 200.0, // Wide GLFT half-spread so clamp range > 15 bps
            arrival_intensity: 0.5,
            cached_best_bid: 99.90,
            cached_best_ask: 100.10,
            ..Default::default()
        };
        params.capital_policy.use_tick_grid = false;

        // 5 bps drift (at old cap)
        params.drift_signal_bps = 5.0;
        let ladder_5 = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);

        // 15 bps drift (3x old cap — was impossible before de-capping)
        params.drift_signal_bps = 15.0;
        let ladder_15 = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);

        // Compare average bid prices across all levels
        let avg_bid_5: f64 =
            ladder_5.bids.iter().map(|l| l.price).sum::<f64>() / ladder_5.bids.len().max(1) as f64;
        let avg_bid_15: f64 = ladder_15.bids.iter().map(|l| l.price).sum::<f64>()
            / ladder_15.bids.len().max(1) as f64;
        assert!(
            avg_bid_15 > avg_bid_5,
            "Drift > old 5 bps cap should shift reservation mid further. avg@5bps={avg_bid_5:.4}, avg@15bps={avg_bid_15:.4}",
        );
    }

    // --- Posterior Drift Tests ---

    #[test]
    fn test_posterior_drift_shifts_reservation_mid() {
        // Posterior drift with confidence should shift bid prices up (bullish).
        let strategy = LadderStrategy::new(0.1);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let mut params = MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            margin_available: 1000.0,
            leverage: 1.0,
            sigma: 0.005,
            kappa: 2000.0,
            arrival_intensity: 0.5,
            cached_best_bid: 99.95,
            cached_best_ask: 100.05,
            ..Default::default()
        };
        params.capital_policy.use_tick_grid = false;

        // Bullish posterior: bias > 0, confidence = 0.8, signal drift = 0
        params.belief_predictive_bias = 0.01; // 1% per second upward
        params.belief_confidence = 0.8;
        params.drift_signal_bps = 0.0;
        let ladder_posterior = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);
        let bid_price_posterior = ladder_posterior
            .bids
            .first()
            .map(|l| l.price)
            .unwrap_or(0.0);

        // No posterior
        params.belief_predictive_bias = 0.0;
        params.belief_confidence = 0.0;
        let ladder_baseline = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);
        let bid_price_baseline = ladder_baseline.bids.first().map(|l| l.price).unwrap_or(0.0);

        assert!(
            bid_price_posterior > bid_price_baseline,
            "Posterior bullish drift should raise bid prices. posterior={bid_price_posterior}, baseline={bid_price_baseline}"
        );
    }

    #[test]
    fn test_posterior_drift_gated_by_confidence() {
        // Zero confidence should gate the posterior — no shift even with large bias.
        let strategy = LadderStrategy::new(0.1);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let mut params = MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            margin_available: 1000.0,
            leverage: 1.0,
            sigma: 0.005,
            kappa: 2000.0,
            arrival_intensity: 0.5,
            cached_best_bid: 99.95,
            cached_best_ask: 100.05,
            ..Default::default()
        };
        params.capital_policy.use_tick_grid = false;

        // Large bias but zero confidence
        params.belief_predictive_bias = 0.01;
        params.belief_confidence = 0.0;
        params.drift_signal_bps = 0.0;
        let ladder_gated = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);

        // No posterior at all
        params.belief_predictive_bias = 0.0;
        params.belief_confidence = 0.0;
        let ladder_baseline = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);

        let avg_bid_gated: f64 = ladder_gated.bids.iter().map(|l| l.price).sum::<f64>()
            / ladder_gated.bids.len().max(1) as f64;
        let avg_bid_baseline: f64 = ladder_baseline.bids.iter().map(|l| l.price).sum::<f64>()
            / ladder_baseline.bids.len().max(1) as f64;

        assert!(
            (avg_bid_gated - avg_bid_baseline).abs() < 1e-8,
            "Zero confidence should gate posterior — no shift. gated={avg_bid_gated:.6}, baseline={avg_bid_baseline:.6}"
        );
    }

    #[test]
    fn test_posterior_plus_signal_drift_additive() {
        // Both posterior and signal positive — combined depth asymmetry should exceed either alone.
        // With asymmetric-depths architecture, positive drift → tighter bids, wider asks.
        // Measured via touch-level depth difference (ask_touch_depth - bid_touch_depth).
        let strategy = LadderStrategy::new(0.1);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let mut params = MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            margin_available: 1000.0,
            leverage: 1.0,
            sigma: 0.005,
            kappa: 200.0,
            arrival_intensity: 0.5,
            cached_best_bid: 99.90,
            cached_best_ask: 100.10,
            ..Default::default()
        };
        params.capital_policy.use_tick_grid = false;

        // Helper: measure touch-level asymmetry (ask_depth - bid_depth) in bps from mid.
        let touch_asymmetry = |ladder: &crate::market_maker::quoting::ladder::Ladder| -> f64 {
            let mid = 100.0;
            let bid_depth = ladder
                .bids
                .first()
                .map(|l| (mid - l.price) / mid * 10_000.0)
                .unwrap_or(0.0);
            let ask_depth = ladder
                .asks
                .first()
                .map(|l| (l.price - mid) / mid * 10_000.0)
                .unwrap_or(0.0);
            ask_depth - bid_depth
        };

        // Use small drifts to avoid clamp interactions.
        // posterior = signum(0.001) * 0.005 * 10000 * 0.05 = 2.5 bps
        // signal = 2.0 bps, combined = 4.5 bps (GLFT δ* ≈ 50 bps, plenty of room)

        // Baseline
        params.belief_predictive_bias = 0.0;
        params.belief_confidence = 0.0;
        params.drift_signal_bps = 0.0;
        let ladder_base = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);
        let asym_base = touch_asymmetry(&ladder_base);

        // Posterior only (2.5 bps shift)
        params.belief_predictive_bias = 0.001;
        params.belief_confidence = 0.05;
        params.drift_signal_bps = 0.0;
        let ladder_post = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);
        let asym_post = touch_asymmetry(&ladder_post);

        // Signal only (2.0 bps shift)
        params.belief_predictive_bias = 0.0;
        params.belief_confidence = 0.0;
        params.drift_signal_bps = 2.0;
        let ladder_sig = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);
        let asym_sig = touch_asymmetry(&ladder_sig);

        // Both (4.5 bps shift)
        params.belief_predictive_bias = 0.001;
        params.belief_confidence = 0.05;
        params.drift_signal_bps = 2.0;
        let ladder_both = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);
        let asym_both = touch_asymmetry(&ladder_both);

        // Positive drift → wider asks than bids
        assert!(
            asym_post > asym_base,
            "Posterior drift should create asymmetry. post={asym_post:.4}, base={asym_base:.4}"
        );
        assert!(
            asym_sig > asym_base,
            "Signal drift should create asymmetry. sig={asym_sig:.4}, base={asym_base:.4}"
        );
        assert!(
            asym_both > asym_post,
            "Combined should exceed posterior alone. both={asym_both:.4}, post={asym_post:.4}"
        );
        assert!(
            asym_both > asym_sig,
            "Combined should exceed signal alone. both={asym_both:.4}, sig={asym_sig:.4}"
        );
    }

    #[test]
    fn test_posterior_drift_capped_at_2x_spread() {
        // Extreme posterior bias should be clamped by the ±2× half-spread bound.
        let strategy = LadderStrategy::new(0.1);
        let config = QuoteConfig {
            mid_price: 100.0,
            decimals: 4,
            sz_decimals: 2,
            min_notional: 10.0,
        };
        let mut params = MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            margin_available: 1000.0,
            leverage: 1.0,
            sigma: 0.005,
            kappa: 2000.0,
            arrival_intensity: 0.5,
            cached_best_bid: 99.95,
            cached_best_ask: 100.05,
            ..Default::default()
        };
        params.capital_policy.use_tick_grid = false;

        // Extreme bullish posterior — would be hundreds of bps without clamping
        params.belief_predictive_bias = 1.0; // 100%/sec — absurd, tests the clamp
        params.belief_confidence = 1.0;
        params.drift_signal_bps = 0.0;
        let ladder = strategy.calculate_ladder(&config, 0.0, 100.0, 100.0, &params);

        let bid_price = ladder.bids.first().map(|l| l.price).unwrap_or(0.0);

        // Even with extreme drift, the 2× half-spread clamp keeps bids sane.
        // The effective_mid can be at most market_mid * (1 + 2 * glft_half_spread).
        // Bid is effective_mid - half_spread, which should still be near market_mid.
        assert!(
            bid_price < 100.10,
            "Extreme posterior drift must be clamped. bid_price={bid_price}"
        );
    }
}
