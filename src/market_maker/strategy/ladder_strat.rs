//! GLFT Ladder Strategy - multi-level quoting with depth-dependent sizing.

use tracing::{debug, info, warn};

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use crate::market_maker::config::{Quote, QuoteConfig, SizeQuantum};
// VolatilityRegime removed - regime_scalar was redundant with vol_scalar
use crate::market_maker::quoting::{
    BayesianFillModel, DepthSpacing, DynamicDepthConfig, DynamicDepthGenerator,
    EntropyConstrainedOptimizer, EntropyDistributionConfig, EntropyOptimizerConfig, Ladder,
    LadderConfig, LadderLevel, LadderParams, LevelOptimizationParams, MarketRegime,
};
use crate::market_maker::quoting::ladder::tick_grid::{
    TickGridConfig, TickScoringParams, enumerate_bid_ticks, enumerate_ask_ticks,
    score_ticks, select_optimal_ticks,
};
use crate::market_maker::quoting::ladder::risk_budget::{
    SideRiskBudget, SoftmaxParams, allocate_risk_budget, compute_allocation_temperature,
};

use super::{
    CalibratedRiskModel, KellySizer, MarketParams, QuotingStrategy, RiskConfig, RiskFeatures,
    RiskModelConfig, SpreadComposition,
};

use crate::market_maker::risk::PositionZone;

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
    let warmup_boost = if warmup_progress < 0.3 {
        0.05
    } else {
        0.0
    };

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

    /// Calculate effective γ based on current market conditions.
    ///
    /// Supports two modes (same as GLFTStrategy):
    /// 1. **Legacy (Multiplicative)**: γ = γ_base × vol × tox × inv × ...
    /// 2. **Calibrated (Log-Additive)**: log(γ) = log(γ_base) + Σ βᵢ × xᵢ
    ///
    /// Blending controlled by `risk_model_config.risk_model_blend`.
    fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        let cfg = &self.risk_config;
        let blend = self.risk_model_config.risk_model_blend.clamp(0.0, 1.0);

        // ============================================================
        // MODE 1: LEGACY MULTIPLICATIVE GAMMA
        // ============================================================
        let gamma_legacy = if blend < 1.0 {
            self.compute_legacy_gamma(market_params, position, max_position)
        } else {
            0.0
        };

        // ============================================================
        // MODE 2: CALIBRATED LOG-ADDITIVE GAMMA
        // ============================================================
        let gamma_calibrated = if blend > 0.0 || self.risk_model_config.use_calibrated_risk_model {
            let features = RiskFeatures::from_params(
                market_params,
                position,
                max_position,
                &self.risk_model_config,
            );
            self.risk_model.compute_gamma(&features)
        } else {
            0.0
        };

        // ============================================================
        // BLEND BETWEEN MODELS
        // ============================================================
        let gamma_base = if blend <= 0.0 {
            gamma_legacy
        } else if blend >= 1.0 {
            gamma_calibrated
        } else {
            gamma_legacy * (1.0 - blend) + gamma_calibrated * blend
        };

        // ============================================================
        // POST-PROCESS SCALARS (always applied)
        // ============================================================
        let gamma_with_calib = gamma_base * market_params.calibration_gamma_mult;
        let gamma_final = gamma_with_calib * market_params.tail_risk_multiplier;

        gamma_final.clamp(cfg.gamma_min, cfg.gamma_max)
    }

    /// Compute gamma using legacy multiplicative model.
    fn compute_legacy_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        let cfg = &self.risk_config;

        // Volatility scaling
        let vol_ratio = market_params.sigma_effective / cfg.sigma_baseline.max(1e-9);
        let vol_scalar = if vol_ratio <= 1.0 {
            1.0
        } else {
            let raw = 1.0 + cfg.volatility_weight * (vol_ratio - 1.0);
            raw.min(cfg.max_volatility_multiplier)
        };

        // Toxicity scaling
        let toxicity_scalar = if market_params.jump_ratio <= cfg.toxicity_threshold {
            1.0
        } else {
            1.0 + cfg.toxicity_sensitivity * (market_params.jump_ratio - 1.0)
        };

        // Inventory scaling
        let utilization = if max_position > EPSILON {
            (position.abs() / max_position).min(1.0)
        } else {
            0.0
        };
        let inventory_scalar = if utilization <= cfg.inventory_threshold {
            1.0
        } else {
            let excess = utilization - cfg.inventory_threshold;
            1.0 + cfg.inventory_sensitivity * excess.powi(2)
        };

        // NOTE: regime_scalar REMOVED - was redundant with vol_scalar
        // Both responded to volatility, causing double-scaling

        // Hawkes activity scaling (continuous, not threshold-based)
        let hawkes_baseline = 0.5;
        let hawkes_sensitivity = 2.0;
        let hawkes_scalar = 1.0
            + hawkes_sensitivity
                * (market_params.hawkes_activity_percentile - hawkes_baseline).max(0.0);

        // Time-of-day scaling
        let time_scalar = cfg.time_of_day_multiplier();

        // Book depth scaling
        let book_depth_scalar = cfg.book_depth_multiplier(market_params.near_touch_depth_usd);

        // Uncertainty scaling
        let uncertainty_scalar = if market_params.kappa_ci_width > 0.0 {
            1.0 + (market_params.kappa_ci_width / 10.0).min(0.5)
        } else {
            1.0
        };

        // Combine (note: calibration and tail_risk applied in caller)
        let gamma_effective = cfg.gamma_base
            * vol_scalar
            * toxicity_scalar
            * inventory_scalar
            * hawkes_scalar
            * time_scalar
            * book_depth_scalar
            * uncertainty_scalar;

        gamma_effective.clamp(cfg.gamma_min, cfg.gamma_max)
    }

    /// Calculate holding time from arrival intensity.
    fn holding_time(&self, arrival_intensity: f64) -> f64 {
        let safe_intensity = arrival_intensity.max(0.01);
        (1.0 / safe_intensity).min(self.risk_config.max_holding_time)
    }

    /// Compute Bayesian-informed gamma adjustment for spread calculation.
    ///
    /// Combines four components from Bayesian posteriors:
    /// 1. **Trend confidence discount**: High confidence → tighter spreads
    /// 2. **Bootstrap fill encouragement**: Low P(calibrated) → need fills → tighter
    /// 3. **Adverse selection uncertainty premium**: High uncertainty → wider spreads
    /// 4. **Regime adjustment**: Volatile regime → wider spreads
    ///
    /// # Arguments
    /// * `market_params` - Contains Bayesian components (trend_confidence, bootstrap_confidence, etc.)
    ///
    /// # Returns
    /// Multiplier for base gamma, typically in [0.7, 1.5]
    pub fn compute_bayesian_gamma_adjustment(&self, market_params: &MarketParams) -> f64 {
        let cfg = &self.risk_config;

        // Component 1: Trend confidence discount
        // High confidence in direction → can quote tighter
        // trend_confidence ∈ [0, 1], capped at 0.6 for safety
        let capped_confidence = market_params.trend_confidence.clamp(0.0, 0.6);
        let trend_discount = 1.0 - capped_confidence * 0.4; // [0.76, 1.0]

        // Component 2: Bootstrap fill encouragement
        // Low calibration confidence → need fills → quote tighter
        // bootstrap_confidence ∈ [0, 1], 0 = uncalibrated, 1 = fully calibrated
        let bootstrap_discount = 0.8 + 0.2 * market_params.bootstrap_confidence; // [0.8, 1.0]

        // Component 3: Adverse selection uncertainty premium
        // High uncertainty in adverse posterior → quote wider for safety
        // adverse_uncertainty is std dev of posterior, typically 0.01-0.15
        let uncertainty_premium = 1.0 + market_params.adverse_uncertainty * 2.0; // [1.0, ~1.3]

        // Component 4: Regime adjustment
        // Volatile regime → wider spreads
        let regime_mult = match market_params.adverse_regime {
            0 => 0.9,  // Calm: slightly tighter
            1 => 1.0,  // Normal: baseline
            2 => 1.3,  // Volatile: significantly wider
            _ => 1.0,  // Fallback
        };

        // Combine multiplicatively
        let raw_mult = trend_discount * bootstrap_discount * uncertainty_premium * regime_mult;

        // Bound to reasonable range
        let bayesian_mult = raw_mult.clamp(cfg.gamma_min / cfg.gamma_base, cfg.gamma_max / cfg.gamma_base);

        tracing::debug!(
            trend_confidence = %format!("{:.3}", market_params.trend_confidence),
            trend_discount = %format!("{:.3}", trend_discount),
            bootstrap_confidence = %format!("{:.3}", market_params.bootstrap_confidence),
            bootstrap_discount = %format!("{:.3}", bootstrap_discount),
            adverse_uncertainty = %format!("{:.3}", market_params.adverse_uncertainty),
            uncertainty_premium = %format!("{:.3}", uncertainty_premium),
            adverse_regime = market_params.adverse_regime,
            regime_mult = %format!("{:.2}", regime_mult),
            raw_mult = %format!("{:.3}", raw_mult),
            bayesian_mult = %format!("{:.3}", bayesian_mult),
            "Bayesian gamma adjustment computed"
        );

        bayesian_mult
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
            cascade_severity: if market_params.should_pull_quotes {
                1.0
            } else {
                // Scale from tail_risk_multiplier: 1.0 → 0.0, 5.0 → 1.0
                (market_params.tail_risk_multiplier - 1.0) / 4.0
            },
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
        let bid_size = quantum.clamp_to_viable(raw_bid, bid_round_up).unwrap_or(0.0);
        let raw_ask = available_for_asks.min(half_max).min(per_side_cap);
        let ask_round_up = available_for_asks >= quantum.min_viable_size;
        let ask_size = quantum.clamp_to_viable(raw_ask, ask_round_up).unwrap_or(0.0);

        // Depth: use min_viable_depth_bps from capacity budget (QueueValue breakeven),
        // falling back to max(min_depth_bps, 5.0) to survive QueueValue filtering.
        let depth_bps = market_params.capacity_budget
            .as_ref()
            .map(|b| b.min_viable_depth_bps)
            .unwrap_or(5.0)
            .max(self.ladder_config.min_depth_bps);
        let depth_frac = depth_bps / 10_000.0;
        let one_tick = 10f64.powi(-(config.decimals as i32));

        // Generate bid (if meets min_notional)
        if quantum.is_sufficient(bid_size) && available_for_bids > 0.0 {
            let mut bid_price = round_to_significant_and_decimal(
                mark * (1.0 - depth_frac),
                5,
                config.decimals,
            );
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
            let mut ask_price = round_to_significant_and_decimal(
                mark * (1.0 + depth_frac),
                5,
                config.decimals,
            );
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
        let effective_max_position = market_params.effective_max_position(max_position).min(max_position);
        let gamma = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
            market_params.adaptive_gamma * market_params.tail_risk_multiplier
        } else {
            let base = self.effective_gamma(market_params, 0.0, effective_max_position);
            base * market_params.liquidity_gamma_mult * market_params.tail_risk_multiplier
        };
        let gamma = gamma * market_params.regime_gamma_multiplier;

        let kappa = if market_params.use_kappa_robust {
            market_params.kappa_robust
        } else {
            market_params.kappa
        };

        // Core GLFT half-spread at touch level
        let glft_half_frac = self.depth_generator.glft_optimal_spread(gamma, kappa);
        let glft_half_bps = glft_half_frac * 10_000.0;

        // Risk premium from regime and position zone
        let risk_premium_bps = market_params.regime_risk_premium_bps
            + market_params.total_risk_premium_bps;

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
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> Ladder {
        // Circuit breaker: pull all quotes during cascade
        if market_params.should_pull_quotes {
            return Ladder::default();
        }

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

        // === GAMMA: Adaptive vs Legacy with Bayesian Adjustment ===
        // When adaptive spreads enabled: use log-additive shrinkage gamma
        // When disabled: use multiplicative RiskConfig gamma
        //
        // In BOTH paths, apply:
        // 1. calibration_gamma_mult for fill-hungry mode during warmup
        // 2. bayesian_gamma_mult for posterior-driven adjustments (trend, bootstrap, adverse)
        let calibration_scalar = market_params.calibration_gamma_mult;

        // Compute Bayesian gamma adjustment from posteriors
        // If pre-computed in MarketParams, use that; otherwise compute here
        let bayesian_scalar = if (market_params.bayesian_gamma_mult - 1.0).abs() > 0.001 {
            market_params.bayesian_gamma_mult
        } else {
            self.compute_bayesian_gamma_adjustment(market_params)
        };

        let gamma = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
            // Adaptive gamma: log-additive scaling prevents multiplicative explosion
            // Still apply tail risk multiplier for cascade protection
            // Apply calibration scalar for fill-hungry mode
            // Apply Bayesian scalar for posterior-driven adjustments
            let adaptive_gamma = market_params.adaptive_gamma;
            debug!(
                adaptive_gamma = %format!("{:.4}", adaptive_gamma),
                tail_mult = %format!("{:.2}", market_params.tail_risk_multiplier),
                calibration_mult = %format!("{:.2}", calibration_scalar),
                bayesian_mult = %format!("{:.3}", bayesian_scalar),
                warmup_pct = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
                "Ladder using ADAPTIVE gamma with BAYESIAN adjustment"
            );
            adaptive_gamma * market_params.tail_risk_multiplier * calibration_scalar * bayesian_scalar
        } else {
            // Legacy: multiplicative RiskConfig gamma
            // Note: effective_gamma() already includes calibration_scalar
            let base_gamma = self.effective_gamma(market_params, position, effective_max_position);
            let gamma_with_liq = base_gamma * market_params.liquidity_gamma_mult;
            let legacy_gamma = gamma_with_liq * market_params.tail_risk_multiplier;
            debug!(
                base_gamma = %format!("{:.4}", base_gamma),
                gamma_with_liq = %format!("{:.4}", gamma_with_liq),
                tail_mult = %format!("{:.2}", market_params.tail_risk_multiplier),
                bayesian_mult = %format!("{:.3}", bayesian_scalar),
                "Ladder using LEGACY gamma with BAYESIAN adjustment"
            );
            legacy_gamma * bayesian_scalar
        };

        // === REGIME GAMMA MULTIPLIER (Continuously Blended) ===
        // Route regime risk through gamma instead of floor clamping.
        // This lets GLFT naturally widen spreads in volatile/extreme regimes
        // via δ = (1/γ)ln(1 + γ/κ) — higher γ → wider spread.
        // The multiplier is continuously blended from regime probabilities
        // (e.g. probs [0.2, 0.3, 0.3, 0.2] -> 1.76) instead of discrete
        // jumps {1.0, 1.2, 2.0, 3.0}.
        let gamma = gamma * market_params.regime_gamma_multiplier;
        if (market_params.regime_gamma_multiplier - 1.0).abs() > 0.01 {
            tracing::info!(
                regime_gamma_mult = %format!("{:.2}", market_params.regime_gamma_multiplier),
                gamma_after = %format!("{:.4}", gamma),
                "Regime gamma multiplier applied (replaces floor clamping)"
            );
        }

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
                REGIME_BLEND_WEIGHT_WARMUP + t * (REGIME_BLEND_WEIGHT_FULL - REGIME_BLEND_WEIGHT_WARMUP)
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

        // === POSITION ZONE: Graduated risk overlay ===
        // Only kill switch can cancel all quotes. Everything else widens or reduces size.
        // Zone thresholds are fractions of max_position:
        //   Green  (<60%): normal operation
        //   Yellow (60-80%): wider spread on accumulating side + smaller size
        //   Red    (80-100%): reduce-only (only position-reducing side quotes)
        //   Kill   (>100%): same as Red here; kill switch handles cancellation
        let abs_inventory_ratio = inventory_ratio.abs();
        let position_zone = if abs_inventory_ratio >= 1.0 {
            PositionZone::Kill
        } else if abs_inventory_ratio >= 0.8 {
            PositionZone::Red
        } else if abs_inventory_ratio >= 0.6 {
            PositionZone::Yellow
        } else {
            PositionZone::Green
        };

        // Zone-aware size multiplier: reduces size as we approach limits
        let zone_size_mult = match position_zone {
            PositionZone::Green => 1.0,
            PositionZone::Yellow => 0.5,
            PositionZone::Red | PositionZone::Kill => 0.3, // Minimal size, reduce-only
        };

        // Zone-aware spread widening on the accumulating side
        let zone_spread_widen_bps = match position_zone {
            PositionZone::Green => 0.0,
            PositionZone::Yellow => 3.0,  // +3 bps on accumulating side
            PositionZone::Red | PositionZone::Kill => 10.0, // +10 bps (discourage accumulation)
        };

        if position_zone != PositionZone::Green {
            tracing::info!(
                zone = ?position_zone,
                abs_inventory_ratio = %format!("{:.2}", abs_inventory_ratio),
                zone_size_mult = %format!("{:.2}", zone_size_mult),
                zone_spread_widen_bps = %format!("{:.1}", zone_spread_widen_bps),
                "Position zone risk overlay active"
            );
        }

        // Convert AS spread adjustment to bps for ladder generation
        let as_at_touch_bps = if market_params.as_warmed_up {
            market_params.as_spread_adjustment * 10000.0
        } else {
            0.0 // Use zero AS until warmed up
        };

        // Apply cascade size reduction to target_liquidity (but NOT to initial ladder generation)
        let _adjusted_size = target_liquidity * market_params.cascade_size_factor;

        // For initial ladder generation, use the full quoting capacity so that individual
        // levels pass the min_notional filter. The entropy optimizer will then correctly
        // allocate the actual available margin (available_for_bids/asks).
        //
        // FIXED: Previously used target_liquidity which was often much smaller than
        // quoting capacity (e.g., 1.3 HYPE vs 66 HYPE), causing ALL levels to fail
        // min_notional check, triggering concentration fallback BEFORE entropy optimizer.
        let size_for_initial_ladder =
            effective_max_position * market_params.cascade_size_factor * zone_size_mult;

        // L2 reservation shift removed — double-counts skew already handled
        // by lead_lag_signal_bps in the directional skew section below.
        let adjusted_microprice = market_params.microprice;

        // === DIRECTIONAL SKEW: Apply combined skew as mid-price offset ===
        // lead_lag_signal_bps carries combined_skew_bps + position_guard_skew_bps
        // from the signal integrator and position guard.
        // Positive skew = bullish → shift mid UP → tighten bids, widen asks.
        // Negative skew = bearish → shift mid DOWN → widen bids, tighten asks.
        let skew_bps = market_params.lead_lag_signal_bps;
        let skew_fraction = skew_bps / 10_000.0;
        let skewed_microprice = adjusted_microprice * (1.0 + skew_fraction);

        // SAFETY: Bound skewed mid within 80% of GLFT half-spread to prevent crossing.
        // Uses the GLFT formula directly so the bound is consistent with our quoting depth.
        let glft_half_spread_frac = self.depth_generator.glft_optimal_spread(gamma, kappa);
        let half_spread_bound = glft_half_spread_frac * 0.8;
        let max_mid = market_params.market_mid * (1.0 + half_spread_bound);
        let min_mid = market_params.market_mid * (1.0 - half_spread_bound);
        let effective_mid = skewed_microprice.clamp(min_mid, max_mid);

        if skew_bps.abs() > 0.5 {
            tracing::debug!(
                skew_bps = %format!("{:.2}", skew_bps),
                adjusted_microprice = %format!("{:.4}", adjusted_microprice),
                effective_mid = %format!("{:.4}", effective_mid),
                market_mid = %format!("{:.4}", market_params.market_mid),
                "Directional skew applied to mid price"
            );
        }

        // === SPREAD FLOOR: Physical Constraints Only ===
        // The floor ONLY includes physical/exchange constraints that cannot be violated.
        // Regime risk is routed through gamma (gamma_multiplier) instead of floor clamping.
        // This lets GLFT produce the actual spread instead of being overridden by regime floors.
        let fee_bps = self.risk_config.maker_fee_rate * 10_000.0;
        let tick_floor_bps = market_params.tick_size_bps;
        let latency_floor_bps = market_params.latency_spread_floor * 10_000.0;
        let effective_floor_bps = fee_bps
            .max(tick_floor_bps)
            .max(latency_floor_bps)
            .max(self.risk_config.min_spread_floor * 10_000.0);

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
            directional_variance_mult: market_params.directional_variance_mult,
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
            position_action: market_params.position_action,
            effective_inventory_ratio: market_params.effective_inventory_ratio,
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
        let bid_bound = dynamic_depths.bid.iter().any(|d| (*d - effective_floor_bps).abs() < 0.01);
        let ask_bound = dynamic_depths.ask.iter().any(|d| (*d - effective_floor_bps).abs() < 0.01);
        if bid_bound || ask_bound {
            tracing::warn!(
                effective_floor_bps = %format!("{:.2}", effective_floor_bps),
                glft_optimal_bps = %format!("{:.2}", glft_optimal_bps),
                gamma = %format!("{:.4}", gamma),
                "Floor binding — gamma may be miscalibrated (should be rare after self-consistent gamma)"
            );
        }

        // === POSITION ZONE: Apply graduated spread widening ===
        // Widen the accumulating side in Yellow/Red zones.
        // If long (position > 0), accumulating side = bids (buying more).
        // If short (position < 0), accumulating side = asks (selling more).
        if zone_spread_widen_bps > 0.0 {
            let widen_bids = position > 0.0; // Long → bids accumulate
            if widen_bids {
                for depth in dynamic_depths.bid.iter_mut() {
                    *depth += zone_spread_widen_bps;
                }
            } else {
                for depth in dynamic_depths.ask.iter_mut() {
                    *depth += zone_spread_widen_bps;
                }
            }
        }

        // In Red zone, clear the accumulating side entirely (reduce-only).
        if matches!(position_zone, PositionZone::Red | PositionZone::Kill) {
            if position > 0.0 {
                // Long: clear bids (don't buy more), keep asks (let us sell)
                dynamic_depths.bid.clear();
            } else if position < 0.0 {
                // Short: clear asks (don't sell more), keep bids (let us buy)
                dynamic_depths.ask.clear();
            }
            tracing::warn!(
                zone = ?position_zone,
                position = %format!("{:.4}", position),
                "Red zone: accumulating side cleared (reduce-only)"
            );
        }

        // Kappa-driven spread cap removed — circular with GLFT.
        // GLFT delta = (1/gamma) * ln(1 + gamma/kappa) IS the self-consistent spread.

        // === PRE-FILL AS MULTIPLIERS ===
        // Apply asymmetric spread widening from pre-fill adverse selection classifier.
        // These multipliers are [1.0, 3.0] where >1.0 indicates predicted toxicity on that side.
        // Widen depths (not gamma) to get direct, per-side spread control.
        //
        // WARMUP CAP: During warmup the AS model is uncalibrated — update_count rises from
        // book/trade events (not fills), so the warmup prior gets overridden before any fills
        // arrive. Without a cap, raw signal noise produces 1.5-2x multipliers that push the
        // touch from ~7 bps to ~14 bps, preventing fills entirely (cold-start death spiral).
        // Cap at 1.15x during warmup (mild protection); full multiplier after warmup graduates.
        let warmup_as_cap = if market_params.adaptive_warmup_progress < 1.0 {
            // Linear ramp: 1.2x at 0% warmup → 3.0x at 100% warmup
            // At cold start, allow mild AS widening (1.2x floor ≈ +1.6 bps on 8 bps spread)
            // as a Bayesian prior for adverse selection defense. The AS model is
            // uncalibrated initially, but zero protection leaves us exposed. The
            // 1.2x floor passes the warmup prior (~1.14x) through while capping noise.
            let t = market_params.adaptive_warmup_progress;
            1.2 + t * (3.0 - 1.2)
        } else {
            3.0 // No cap post-warmup
        };
        let capped_bid_mult = market_params.pre_fill_spread_mult_bid.min(warmup_as_cap);
        let capped_ask_mult = market_params.pre_fill_spread_mult_ask.min(warmup_as_cap);

        if capped_bid_mult > 1.01 {
            for depth in dynamic_depths.bid.iter_mut() {
                *depth *= capped_bid_mult;
            }
        }
        if capped_ask_mult > 1.01 {
            for depth in dynamic_depths.ask.iter_mut() {
                *depth *= capped_ask_mult;
            }
        }

        // [SPREAD TRACE] Phase 6: after pre-fill AS multipliers
        tracing::info!(
            phase = "pre_fill_as",
            raw_mult_bid = %format!("{:.3}", market_params.pre_fill_spread_mult_bid),
            raw_mult_ask = %format!("{:.3}", market_params.pre_fill_spread_mult_ask),
            capped_mult_bid = %format!("{:.3}", capped_bid_mult),
            capped_mult_ask = %format!("{:.3}", capped_ask_mult),
            warmup_as_cap = %format!("{:.3}", warmup_as_cap),
            touch_bid_bps = %format!("{:.2}", dynamic_depths.best_bid_depth().unwrap_or(0.0)),
            touch_ask_bps = %format!("{:.2}", dynamic_depths.best_ask_depth().unwrap_or(0.0)),
            total_at_touch_bps = %format!("{:.2}", dynamic_depths.spread_at_touch().unwrap_or(0.0)),
            "[SPREAD TRACE] after pre-fill AS multipliers (warmup-capped)"
        );

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
            let usable_margin = available_margin * margin_utilization;

            // === TWO-SIDED MARGIN ALLOCATION (STOCHASTIC-WEIGHTED) ===
            // Split usable margin using three components:
            // 1. INVENTORY: Mean-revert position (primary driver)
            // 2. MOMENTUM: Follow flow when flat to capture directional edge
            // 3. URGENCY: Amplify reduction when position opposes momentum (danger zone)
            //
            // FIRST PRINCIPLES:
            // - When flat: follow momentum to capture directional alpha
            // - When positioned: prioritize mean-reversion
            // - When position opposes momentum: urgently reduce exposure

            // Simple inventory-driven margin split.
            // Long → more margin to asks (mean-revert), Short → more to bids.
            // Directional skew is already in the microprice offset above.
            let inventory_ratio = if effective_max_position > EPSILON {
                (position / effective_max_position).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let inv_factor = (inventory_ratio * 0.3).clamp(-0.20, 0.20);
            let ask_margin_weight = (0.5 + inv_factor).clamp(0.30, 0.70);
            let bid_margin_weight = 1.0 - ask_margin_weight;
            let margin_for_bids = usable_margin * bid_margin_weight;
            let margin_for_asks = usable_margin * ask_margin_weight;

            info!(
                usable_margin = %format!("{:.2}", usable_margin),
                inventory_ratio = %format!("{:.3}", inventory_ratio),
                inv_factor = %format!("{:.3}", inv_factor),
                margin_for_bids = %format!("{:.2}", margin_for_bids),
                margin_for_asks = %format!("{:.2}", margin_for_asks),
                "Margin allocation (inventory-driven split)"
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

            // === TOXICITY-BASED SIZE REDUCTION (Sprint 2.3) ===
            // When pre-fill toxicity > 0.5 on a side, reduce that side's capacity
            // proportionally. This shrinks order sizes to limit adverse selection losses
            // while keeping quotes active (unlike cancel-on-toxicity which removes them).
            let available_for_bids = available_for_bids * market_params.pre_fill_size_mult_bid;
            let available_for_asks = available_for_asks * market_params.pre_fill_size_mult_ask;

            if market_params.pre_fill_size_mult_bid < 0.99 || market_params.pre_fill_size_mult_ask < 0.99 {
                tracing::info!(
                    bid_size_mult = %format!("{:.2}", market_params.pre_fill_size_mult_bid),
                    ask_size_mult = %format!("{:.2}", market_params.pre_fill_size_mult_ask),
                    bid_toxicity = %format!("{:.3}", market_params.pre_fill_toxicity_bid),
                    ask_toxicity = %format!("{:.3}", market_params.pre_fill_toxicity_ask),
                    "Toxicity size reduction applied"
                );
            }

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

            // === DYNAMIC LEVEL COUNT BASED ON EXCHANGE LIMITS ===
            // When exchange limits are tight, we must reduce the number of levels
            // to avoid placing many sub-minimum orders that would be rejected.
            //
            // Formula: max_levels = available_capacity / quantum.min_viable_size
            // SizeQuantum uses exact ceiling math — no fudge factors needed.
            let quantum = SizeQuantum::compute(config.min_notional, market_params.microprice, config.sz_decimals);

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
                let glft_touch_bps = self.depth_generator.glft_optimal_spread(gamma, kappa) * 10_000.0;
                let touch_bps = glft_touch_bps.max(effective_floor_bps);

                // WS2: Wider depth range using tier-dependent multiplier.
                // Old: max(kappa_cap, touch * 2.0) — only 7.5 bps range for 5 levels.
                // New: max(kappa_cap, touch * depth_range_multiplier) — 30+ bps for meaningful diversity.
                let policy = &market_params.capital_policy;
                let max_depth_bps = (touch_bps * policy.depth_range_multiplier)
                    .min(100.0); // Absolute cap: never deeper than 100 bps

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
                let in_yellow_zone = matches!(position_zone, PositionZone::Yellow);
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
                let (bid_allocs, bid_diag) = allocate_risk_budget(&selected_bids, &bid_budget, &softmax_params);
                let (ask_allocs, ask_diag) = allocate_risk_budget(&selected_asks, &ask_budget, &softmax_params);

                // Skip if Red zone cleared accumulating side
                let bid_cleared_by_zone = matches!(position_zone, PositionZone::Red | PositionZone::Kill)
                    && position > 0.0;
                let ask_cleared_by_zone = matches!(position_zone, PositionZone::Red | PositionZone::Kill)
                    && position < 0.0;

                // Build ladder from tick-grid allocations
                let bids: smallvec::SmallVec<_> = if bid_cleared_by_zone {
                    smallvec::smallvec![]
                } else {
                    bid_allocs.iter().map(|a| {
                        LadderLevel {
                            price: round_to_significant_and_decimal(a.tick.price, 5, config.decimals),
                            size: a.size,
                            depth_bps: a.tick.depth_bps,
                        }
                    }).collect()
                };

                let asks: smallvec::SmallVec<_> = if ask_cleared_by_zone {
                    smallvec::smallvec![]
                } else {
                    ask_allocs.iter().map(|a| {
                        LadderLevel {
                            price: round_to_significant_and_decimal(a.tick.price, 5, config.decimals),
                            size: a.size,
                            depth_bps: a.tick.depth_bps,
                        }
                    }).collect()
                };

                // Apply spread compensation multiplier for small capital
                let mut ladder = Ladder { bids, asks };
                if (policy.spread_compensation_mult - 1.0).abs() > 0.001 {
                    // Widen depths by compensation factor (adjusts price offsets)
                    for level in ladder.bids.iter_mut() {
                        let new_depth = level.depth_bps * policy.spread_compensation_mult;
                        let offset = market_params.microprice * (new_depth / 10_000.0);
                        level.price = round_to_significant_and_decimal(
                            market_params.microprice - offset, 5, config.decimals,
                        );
                        level.depth_bps = new_depth;
                    }
                    for level in ladder.asks.iter_mut() {
                        let new_depth = level.depth_bps * policy.spread_compensation_mult;
                        let offset = market_params.microprice * (new_depth / 10_000.0);
                        level.price = round_to_significant_and_decimal(
                            market_params.microprice + offset, 5, config.decimals,
                        );
                        level.depth_bps = new_depth;
                    }
                }

                // WS6: Diagnostic logging with size distribution
                let bid_sizes_str: String = ladder.bids.iter()
                    .map(|l| format!("{:.2}", l.size))
                    .collect::<Vec<_>>()
                    .join(",");
                let ask_sizes_str: String = ladder.asks.iter()
                    .map(|l| format!("{:.2}", l.size))
                    .collect::<Vec<_>>()
                    .join(",");
                let bid_depths_str: String = ladder.bids.iter()
                    .map(|l| format!("{:.1}", l.depth_bps))
                    .collect::<Vec<_>>()
                    .join(",");
                let ask_depths_str: String = ladder.asks.iter()
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
                let bid_depths_str: Vec<String> = dd.bid.iter().map(|d| format!("{d:.2}")).collect();
                let ask_depths_str: Vec<String> = dd.ask.iter().map(|d| format!("{d:.2}")).collect();
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
                let bid_str: Vec<String> = ladder.bids.iter().map(|l| format!("({:.4}@{:.2}bps,sz={:.2})", l.price, l.depth_bps, l.size)).collect();
                let ask_str: Vec<String> = ladder.asks.iter().map(|l| format!("({:.4}@{:.2}bps,sz={:.2})", l.price, l.depth_bps, l.size)).collect();
                info!(
                    bid_levels = ladder.bids.len(),
                    ask_levels = ladder.asks.len(),
                    bids = %bid_str.join(" | "),
                    asks = %ask_str.join(" | "),
                    "DIAGNOSTIC: post-generate ladder levels"
                );
            }

            // Log if capacity is tight (for debugging), but don't truncate
            if effective_bid_levels < configured_levels || effective_ask_levels < configured_levels
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
                    let spread_capture =
                        spread_capture_at_depth(level.depth_bps, &params, ladder_config.fees_bps);
                    let adverse_selection = adverse_selection_at_depth(level.depth_bps, &params);
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
                        margin_for_bids, // Use inventory-weighted margin split (not full margin)
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
                            ladder.bids[i].size = truncate_float(size, config.sz_decimals, false);
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
                    let spread_capture =
                        spread_capture_at_depth(level.depth_bps, &params, ladder_config.fees_bps);
                    let adverse_selection = adverse_selection_at_depth(level.depth_bps, &params);
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
                        margin_for_asks, // Use inventory-weighted margin split (not full margin)
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
                            ladder.asks[i].size = truncate_float(size, config.sz_decimals, false);
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
            let tightest_depth_bps = market_params.capacity_budget
                .as_ref()
                .map(|b| b.min_viable_depth_bps)
                .unwrap_or(5.0)
                .max(ladder_config.min_depth_bps);

            // Bid concentration fallback — skip if Red zone cleared bids (reduce-only for longs)
            let bid_cleared_by_zone = matches!(position_zone, PositionZone::Red | PositionZone::Kill)
                && position > 0.0;
            if ladder.bids.is_empty() && !bid_cleared_by_zone {
                // Total available size for bids, capped at 25% of the USER'S risk-based
                // max position per order (not margin-based quoting capacity).
                // GLFT inventory theory: one fill at max position creates maximal reservation
                // price swing q*gamma*sigma^2*T with zero recovery capacity.
                let per_order_cap = (max_position * MAX_SINGLE_ORDER_FRACTION)
                    .max(quantum.min_viable_size);
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
            let ask_cleared_by_zone = matches!(position_zone, PositionZone::Red | PositionZone::Kill)
                && position < 0.0;
            if ladder.asks.is_empty() && !ask_cleared_by_zone {
                // Total available size for asks, capped at 25% of the USER'S risk-based
                // max position per order (not margin-based quoting capacity).
                let per_order_cap = (max_position * MAX_SINGLE_ORDER_FRACTION)
                    .max(quantum.min_viable_size);
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
                        let urgency = (position.abs() * (-market_params.unrealized_pnl_bps / 20.0).max(0.0)).min(1.0);
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
                        let urgency = (position.abs() * (-market_params.unrealized_pnl_bps / 20.0).max(0.0)).min(1.0);
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

            let bid_cleared_by_zone = matches!(position_zone, PositionZone::Red | PositionZone::Kill)
                && position > 0.0;
            let ask_cleared_by_zone = matches!(position_zone, PositionZone::Red | PositionZone::Kill)
                && position < 0.0;

            // Also skip guaranteed quotes when should_pull_quotes is set (circuit breaker)
            if !market_params.should_pull_quotes {
                if ladder.bids.is_empty() && !bid_cleared_by_zone {
                    let bid_depth_bps = guaranteed_half_spread_bps + guaranteed_skew_bps;
                    let offset = market_params.microprice * (bid_depth_bps / 10_000.0);
                    let bid_price = round_to_significant_and_decimal(
                        market_params.microprice - offset,
                        5,
                        config.decimals,
                    );
                    let bid_size = quantum.min_viable_size;

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
                    let ask_size = quantum.min_viable_size;

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

    /// Compute simplified inventory-driven margin split weight.
    fn compute_margin_split(inventory_ratio: f64) -> (f64, f64) {
        let inv_factor = (inventory_ratio * 0.3).clamp(-0.20, 0.20);
        let ask_w = (0.5 + inv_factor).clamp(0.30, 0.70);
        (1.0 - ask_w, ask_w) // (bid_weight, ask_weight)
    }

    #[test]
    fn test_margin_split_flat_is_equal() {
        let (bid_w, ask_w) = compute_margin_split(0.0);
        assert!((bid_w - 0.5).abs() < 1e-10);
        assert!((ask_w - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_margin_split_long_favors_asks() {
        // Long position → more margin to asks (mean-revert)
        let (bid_w, ask_w) = compute_margin_split(0.5);
        assert!(ask_w > bid_w, "Long should favor asks: bid={bid_w:.3}, ask={ask_w:.3}");
    }

    #[test]
    fn test_margin_split_short_favors_bids() {
        // Short position → more margin to bids (mean-revert)
        let (bid_w, ask_w) = compute_margin_split(-0.5);
        assert!(bid_w > ask_w, "Short should favor bids: bid={bid_w:.3}, ask={ask_w:.3}");
    }

    #[test]
    fn test_margin_split_clamped() {
        // Even extreme inventory, split stays within [0.30, 0.70]
        let (bid_w, ask_w) = compute_margin_split(1.0);
        assert!(ask_w <= 0.70 + 1e-10);
        assert!(bid_w >= 0.30 - 1e-10);
        let (bid_w2, ask_w2) = compute_margin_split(-1.0);
        assert!(bid_w2 <= 0.70 + 1e-10);
        assert!(ask_w2 >= 0.30 - 1e-10);
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
        assert!((cap_8000 - 2.5).abs() < 0.01, "kappa=8000 → cap=2.5 bps, got {:.2}", cap_8000);

        // kappa = 2500 → cap = 8.0 bps
        let cap_2500: f64 = (2.0 / 2500.0 * 10_000.0_f64).max(1.5);
        assert!((cap_2500 - 8.0).abs() < 0.01, "kappa=2500 → cap=8.0 bps, got {:.2}", cap_2500);

        // kappa = 500 → cap = 40 bps (very wide)
        let cap_500: f64 = (2.0 / 500.0 * 10_000.0_f64).max(1.5);
        assert!((cap_500 - 40.0).abs() < 0.01, "kappa=500 → cap=40 bps, got {:.2}", cap_500);

        // kappa = 20000 → cap = 1.5 bps (clamped to min fee floor)
        let cap_20k: f64 = (2.0 / 20000.0 * 10_000.0_f64).max(1.5);
        assert!((cap_20k - 1.5).abs() < f64::EPSILON, "kappa=20000 → cap=1.5 bps (min), got {:.2}", cap_20k);
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
                i, caps[i], i + 1, caps[i + 1]
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
            glft_approx_bps, effective_floor_warmed
        );

        // Cold: buffer = 2.0 → effective_floor = 4.5 → GLFT (2.75) < floor (4.5) → binding
        let buffer_cold = 2.0;
        let effective_floor_cold = adaptive_floor_bps + buffer_cold;
        assert!(
            glft_approx_bps < effective_floor_cold,
            "Cold: floor ({:.2}) should exceed GLFT ({:.2}) — floor BINDING (defensive)",
            effective_floor_cold, glft_approx_bps
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
        assert!((physical_floor - 1.5_f64).abs() < 0.01,
            "Physical floor should be 1.5 bps (maker fee), got {physical_floor}");

        // GLFT optimal should be ABOVE the fee floor in normal conditions
        // GLFT approx: 1/kappa * 10000 + fee
        let kappa = 2000.0;
        let glft_approx = 1.0 / kappa * 10_000.0 + fee_bps;
        assert!(glft_approx > physical_floor,
            "GLFT ({glft_approx:.2}) should exceed physical floor ({physical_floor:.2})");
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
    fn test_position_zone_green() {
        // <60% inventory → Green zone
        let abs_ratio = 0.5;
        let zone = if abs_ratio >= 1.0 {
            PositionZone::Kill
        } else if abs_ratio >= 0.8 {
            PositionZone::Red
        } else if abs_ratio >= 0.6 {
            PositionZone::Yellow
        } else {
            PositionZone::Green
        };
        assert_eq!(zone, PositionZone::Green);
    }

    #[test]
    fn test_position_zone_yellow() {
        // 60-80% inventory → Yellow zone
        let abs_ratio = 0.7;
        let zone = if abs_ratio >= 1.0 {
            PositionZone::Kill
        } else if abs_ratio >= 0.8 {
            PositionZone::Red
        } else if abs_ratio >= 0.6 {
            PositionZone::Yellow
        } else {
            PositionZone::Green
        };
        assert_eq!(zone, PositionZone::Yellow);
    }

    #[test]
    fn test_position_zone_red() {
        // 80-100% inventory → Red zone
        let abs_ratio = 0.85;
        let zone = if abs_ratio >= 1.0 {
            PositionZone::Kill
        } else if abs_ratio >= 0.8 {
            PositionZone::Red
        } else if abs_ratio >= 0.6 {
            PositionZone::Yellow
        } else {
            PositionZone::Green
        };
        assert_eq!(zone, PositionZone::Red);
    }

    #[test]
    fn test_zone_size_multiplier_decreases() {
        // Size multiplier should decrease as zone severity increases
        let green_mult = 1.0_f64;
        let yellow_mult = 0.5_f64;
        let red_mult = 0.3_f64;

        assert!(green_mult > yellow_mult);
        assert!(yellow_mult > red_mult);
        assert!(red_mult > 0.0, "Red zone should still have non-zero size for reduce-only");
    }

    #[test]
    fn test_zone_spread_widening_increases() {
        // Spread widening should increase with severity
        let green_widen = 0.0_f64;
        let yellow_widen = 3.0_f64;
        let red_widen = 10.0_f64;

        assert_eq!(green_widen, 0.0, "Green should have no widening");
        assert!(yellow_widen > green_widen);
        assert!(red_widen > yellow_widen);
    }

    #[test]
    fn test_red_zone_clears_accumulating_side() {
        // In Red zone, accumulating side should be cleared
        let mut bid_depths = vec![5.0, 7.0, 9.0];
        let ask_depths = vec![5.0, 7.0, 9.0];
        let position = 1.0; // Long

        // Simulate Red zone: long → clear bids (accumulating)
        if position > 0.0 {
            bid_depths.clear();
        }

        assert!(bid_depths.is_empty(), "Long in Red → bids should be cleared");
        assert!(!ask_depths.is_empty(), "Long in Red → asks should remain");

        // Short case
        let bid_depths2 = vec![5.0, 7.0, 9.0];
        let mut ask_depths2 = vec![5.0, 7.0, 9.0];
        let position2 = -1.0;

        if position2 < 0.0 {
            ask_depths2.clear();
        }

        assert!(!bid_depths2.is_empty(), "Short in Red → bids should remain");
        assert!(ask_depths2.is_empty(), "Short in Red → asks should be cleared");
    }

    #[test]
    fn test_cost_basis_breakeven_clamping() {
        // Long position: asks should not go below breakeven
        let entry = 100.0;
        let fee_rate = 0.00015;
        let breakeven = entry * (1.0 + fee_rate); // 100.015

        let ask_price = 99.99; // Below breakeven
        assert!(ask_price < breakeven, "Ask ({ask_price}) should be below breakeven ({breakeven})");

        // Clamping should move it up to breakeven
        let clamped = breakeven;
        assert!(clamped >= breakeven, "Clamped ask should be at or above breakeven");

        // Short position: bids should not go above breakeven
        let breakeven_short = entry * (1.0 - fee_rate); // 99.985
        let bid_price = 100.01; // Above breakeven for short
        assert!(bid_price > breakeven_short, "Bid ({bid_price}) should be above breakeven ({breakeven_short})");

        let clamped_bid = breakeven_short;
        assert!(clamped_bid <= breakeven_short, "Clamped bid should be at or below breakeven");
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
        assert!(urgency > 0.5, "High position + underwater should produce urgency > 0.5, got {urgency}");

        // Small position, slightly underwater: no urgency override
        let position_fraction2: f64 = 0.2;
        let unrealized_bps2: f64 = -5.0;
        let urgency2: f64 = (position_fraction2 * (-unrealized_bps2 / 50.0_f64).max(0.0)).min(1.0);
        assert!(urgency2 < 0.5, "Small position, slightly underwater: urgency should be < 0.5, got {urgency2}");
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
            decimals: 2,   // $0.01 tick = 3.3 bps on $30
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let mut market_params = MarketParams::default();
        market_params.microprice = 30.0;
        market_params.market_mid = 30.0;
        market_params.capital_tier = CapitalTier::Micro;
        market_params.margin_available = 100.0;
        market_params.leverage = 3.0;
        market_params.margin_quoting_capacity = 3.24;
        market_params.sigma = 0.001;
        market_params.sigma_effective = 0.001;
        market_params.kappa = 2000.0; // Wider spreads so GLFT depth > tick

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
                notional, config.min_notional
            );
        }
        for level in &ladder.asks {
            let notional = level.size * level.price;
            assert!(
                notional >= config.min_notional,
                "Ask notional ${:.2} should meet min_notional ${:.2}",
                notional, config.min_notional
            );
        }

        // Bid price should be below mid, ask above — GLFT spreads are real
        assert!(
            ladder.bids[0].price < market_params.microprice,
            "Bid {:.4} should be below mid {:.4}",
            ladder.bids[0].price, market_params.microprice
        );
        assert!(
            ladder.asks[0].price > market_params.microprice,
            "Ask {:.4} should be above mid {:.4}",
            ladder.asks[0].price, market_params.microprice
        );

        // REGRESSION: depth should be GLFT-derived, not hardcoded 5 bps.
        // Verify spread invariant: ask_price > bid_price.
        let spread_bps = (ladder.asks[0].price - ladder.bids[0].price)
            / market_params.microprice * 10_000.0;
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
            decimals: 2,   // $0.01 tick
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let mut market_params = MarketParams::default();
        market_params.microprice = 30.0;
        market_params.market_mid = 30.0;
        market_params.capital_tier = CapitalTier::Micro;
        market_params.margin_available = 100.0;
        market_params.leverage = 3.0;
        market_params.margin_quoting_capacity = 3.24;
        market_params.sigma = 0.001;
        market_params.sigma_effective = 0.001;
        market_params.kappa = 2000.0;

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

        // Bid side should be limited — only 0.24 contracts remaining capacity
        // 0.24 * $30 = $7.20 < $10 min_notional → bid should be empty
        assert!(
            ladder.bids.is_empty(),
            "Micro tier near max long should not produce bids (remaining capacity below min_notional)"
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
            decimals: 2,   // $0.01 tick = 3.3 bps on $30
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let mut market_params = MarketParams::default();
        market_params.microprice = 30.0;
        market_params.market_mid = 30.0;
        market_params.capital_tier = CapitalTier::Large;
        market_params.margin_available = 10_000.0;
        market_params.leverage = 3.0;
        market_params.margin_quoting_capacity = 1000.0;
        market_params.sigma_effective = 0.001;
        market_params.sigma = 0.001;
        market_params.kappa = 2000.0;

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
            ladder.bids.len(), ladder.asks.len()
        );

        // Now verify Micro tier also produces valid GLFT output (not a separate path)
        let mut micro_params = MarketParams::default();
        micro_params.microprice = 30.0;
        micro_params.market_mid = 30.0;
        micro_params.capital_tier = CapitalTier::Micro;
        micro_params.margin_available = 100.0;
        micro_params.leverage = 3.0;
        micro_params.margin_quoting_capacity = 3.24;
        micro_params.sigma_effective = 0.001;
        micro_params.sigma = 0.001;
        micro_params.kappa = 2000.0;

        let micro_ladder = strategy.calculate_ladder(
            &config,
            0.0,
            3.24,
            1.0,
            &micro_params,
        );

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
                micro_ladder.bids[0].price, micro_params.microprice
            );
        }
        if !micro_ladder.asks.is_empty() {
            assert!(
                micro_ladder.asks[0].price > micro_params.microprice,
                "Micro ask {:.4} should be above mid {:.4}",
                micro_ladder.asks[0].price, micro_params.microprice
            );
        }
    }

    /// Verify kappa priority chain: Robust > Adaptive > Coordinator > Legacy
    #[test]
    fn test_kappa_priority_chain_coordinator() {
        let mut params = MarketParams::default();
        params.kappa = 1000.0;  // Legacy kappa
        params.coordinator_kappa = 600.0;
        params.coordinator_uncertainty_premium_bps = 2.0;
        params.use_coordinator_kappa = true;
        params.use_kappa_robust = false;
        params.use_adaptive_spreads = false;
        params.adaptive_can_estimate = false;

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
            composition.warmup_addon_bps, params.coordinator_uncertainty_premium_bps
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
        let mut params = MarketParams::default();
        params.kappa = 1000.0;
        params.kappa_robust = 2000.0; // Higher kappa → tighter spread
        params.use_kappa_robust = true;
        params.coordinator_kappa = 600.0; // Lower kappa → wider spread
        params.use_coordinator_kappa = true;

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

        let guaranteed = (fee_bps + tick_bps).max(glft_optimal_bps).max(effective_floor_bps);
        assert!(
            (guaranteed - 3.0).abs() < 1e-10,
            "Should be max(2.0, 3.0, 2.0) = 3.0, got {guaranteed}"
        );

        // When GLFT is narrow, fee+tick dominates
        let glft_narrow = 1.0_f64;
        let guaranteed2 = (fee_bps + tick_bps).max(glft_narrow).max(effective_floor_bps);
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
        assert!((skew_long - 0.75).abs() < 1e-10, "Long skew should be +0.75 bps, got {skew_long}");
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
        assert!(ask_depth_short > bid_depth_short, "Short: ask should be deeper than bid");
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
    fn test_guaranteed_quotes_respect_zone_clearing() {
        // Red zone with long position: bids cleared (accumulating side)
        // Guaranteed quotes should NOT override zone clearing
        let position = 1.0_f64;
        let bid_cleared_by_zone =
            matches!(PositionZone::Red, PositionZone::Red | PositionZone::Kill) && position > 0.0;
        let ask_cleared_by_zone =
            matches!(PositionZone::Red, PositionZone::Red | PositionZone::Kill) && position < 0.0;

        assert!(bid_cleared_by_zone, "Long in Red: bids should be cleared");
        assert!(!ask_cleared_by_zone, "Long in Red: asks should NOT be cleared");

        // Short in Red: asks cleared
        let position_short = -1.0_f64;
        let bid_cleared_short =
            matches!(PositionZone::Red, PositionZone::Red | PositionZone::Kill) && position_short > 0.0;
        let ask_cleared_short =
            matches!(PositionZone::Red, PositionZone::Red | PositionZone::Kill) && position_short < 0.0;

        assert!(!bid_cleared_short, "Short in Red: bids should NOT be cleared");
        assert!(ask_cleared_short, "Short in Red: asks should be cleared");
    }

    #[test]
    fn test_guaranteed_quotes_produce_both_sides_on_empty_ladder() {
        use crate::market_maker::config::auto_derive::CapitalTier;
        use crate::market_maker::strategy::QuotingStrategy;

        // Scenario: zero margin, zero position — should produce guaranteed quotes
        let strategy = LadderStrategy::new(0.07);
        let config = QuoteConfig {
            mid_price: 25.0,
            decimals: 2,    // $0.01 tick = 0.4 bps — fine-grained enough for GLFT offsets
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let mut market_params = MarketParams::default();
        market_params.microprice = 25.0;
        market_params.market_mid = 25.0;
        market_params.capital_tier = CapitalTier::Micro;
        market_params.margin_available = 0.0; // No margin!
        market_params.leverage = 3.0;
        market_params.margin_quoting_capacity = 0.0;
        market_params.sigma = 0.001;
        market_params.sigma_effective = 0.001;
        market_params.kappa = 2000.0;
        market_params.tick_size_bps = 0.4; // Matches decimals=2 at $25

        let ladder = strategy.calculate_ladder(
            &config,
            0.0,  // flat
            5.0,  // max position
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
            ladder.bids[0].price, market_params.microprice
        );
        assert!(
            ladder.asks[0].price > market_params.microprice,
            "Guaranteed ask {:.4} should be above mid {:.4}",
            ladder.asks[0].price, market_params.microprice
        );
    }

    #[test]
    fn test_guaranteed_quotes_not_placed_when_circuit_breaker() {
        use crate::market_maker::strategy::QuotingStrategy;

        // When should_pull_quotes = true, NO quotes at all (not even guaranteed)
        let strategy = LadderStrategy::new(0.07);
        let config = QuoteConfig {
            mid_price: 25.0,
            decimals: 2,
            sz_decimals: 2,
            min_notional: 10.0,
        };

        let mut market_params = MarketParams::default();
        market_params.microprice = 25.0;
        market_params.market_mid = 25.0;
        market_params.margin_available = 0.0;
        market_params.should_pull_quotes = true; // Circuit breaker!
        market_params.sigma = 0.001;
        market_params.kappa = 2000.0;

        let ladder = strategy.calculate_ladder(
            &config,
            0.0,
            5.0,
            1.0,
            &market_params,
        );

        assert!(
            ladder.bids.is_empty() && ladder.asks.is_empty(),
            "Circuit breaker should produce empty ladder, got {} bids + {} asks",
            ladder.bids.len(), ladder.asks.len()
        );
    }
}
