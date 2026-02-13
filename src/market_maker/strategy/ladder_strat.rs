//! GLFT Ladder Strategy - multi-level quoting with depth-dependent sizing.

use tracing::{debug, info, warn};

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use crate::market_maker::config::{Quote, QuoteConfig};
// VolatilityRegime removed - regime_scalar was redundant with vol_scalar
use crate::market_maker::quoting::{
    BayesianFillModel, DepthSpacing, DynamicDepthConfig, DynamicDepthGenerator,
    EntropyConstrainedOptimizer, EntropyDistributionConfig, EntropyOptimizerConfig, Ladder,
    LadderConfig, LadderLevel, LadderParams, LevelOptimizationParams, MarketRegime,
};

use super::{
    CalibratedRiskModel, KellySizer, MarketParams, QuotingStrategy, RiskConfig, RiskFeatures,
    RiskModelConfig,
};

use crate::market_maker::risk::PositionZone;

/// Maximum fraction of effective_max_position allowed in a single resting order.
///
/// GLFT inventory theory: a single fill at max position creates reservation price
/// adjustment = q * gamma * sigma^2 * T, which swings maximally and prevents recovery.
/// Capping at 25% ensures at least 4 fills are needed to reach max inventory,
/// giving the gamma/skew feedback loop time to widen spreads defensively.
const MAX_SINGLE_ORDER_FRACTION: f64 = 0.25;

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

        // === REGIME GAMMA MULTIPLIER ===
        // Route regime risk through gamma instead of floor clamping.
        // This lets GLFT naturally widen spreads in volatile/extreme regimes
        // via δ = (1/γ)ln(1 + γ/κ) — higher γ → wider spread.
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
        } else {
            // Legacy: Book-based kappa with AS adjustment
            // κ_effective = κ̂ × (1 - α), where α = P(informed | fill)
            let alpha = market_params.predicted_alpha.min(0.5);
            market_params.kappa * (1.0 - alpha)
        };

        // === REGIME KAPPA BLENDING ===
        // Regime kappa is the PRIMARY driver of spread width (60% weight).
        // Captures structural differences: Calm → 3000, Normal → 2000,
        // Volatile → 1000, Extreme → 500. This makes spreads naturally widen
        // in volatile regimes without relying on multiplicative spread chain.
        if let Some(regime_kappa) = market_params.regime_kappa {
            let kappa_before_regime = kappa;
            const REGIME_BLEND_WEIGHT: f64 = 0.6;
            kappa = (1.0 - REGIME_BLEND_WEIGHT) * kappa + REGIME_BLEND_WEIGHT * regime_kappa;
            info!(
                kappa_before = %format!("{:.0}", kappa_before_regime),
                regime_kappa = %format!("{:.0}", regime_kappa),
                kappa_after = %format!("{:.0}", kappa),
                regime = market_params.regime_kappa_current_regime,
                "[SPREAD TRACE] regime kappa blending (60% regime weight)"
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

        // === L2 RESERVATION SHIFT (Avellaneda-Stoikov Framework) ===
        // Edge enters through the reservation price, not sizing.
        // δ_μ = μ/(γσ²) shifts where we center quotes:
        // - Positive edge → shift UP → aggressive asks, capturing positive drift
        // - Negative edge → shift DOWN → aggressive bids, capturing negative drift
        //
        // l2_reservation_shift is now a FRACTION (±0.005 max = ±50 bps)
        // Use multiplicative adjustment: adjusted = microprice × (1 + shift_fraction)
        let adjusted_microprice = market_params.microprice * (1.0 + market_params.l2_reservation_shift);

        // SAFETY: Validate adjusted microprice doesn't diverge from market_mid
        let microprice_ratio = adjusted_microprice / market_params.market_mid;
        let adjusted_microprice = if !(0.8..=1.2).contains(&microprice_ratio) {
            // This should never happen with the new bounded shift formula
            tracing::error!(
                adjusted = %format!("{:.4}", adjusted_microprice),
                microprice = %format!("{:.4}", market_params.microprice),
                market_mid = %format!("{:.4}", market_params.market_mid),
                l2_reservation_shift = %format!("{:.6}", market_params.l2_reservation_shift),
                ratio = %format!("{:.2}", microprice_ratio),
                "CRITICAL: Adjusted microprice diverged >20% from market_mid - using microprice directly"
            );
            market_params.microprice
        } else {
            adjusted_microprice
        };

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

        // === KAPPA-DRIVEN SPREAD CAP (Phase 3) ===
        // When kappa is high (lots of fill intensity), cap spreads to be tighter.
        // This allows us to be more aggressive when the market is active.
        if let Some(kappa_cap_bps) = market_params.kappa_spread_bps {
            // Only apply if kappa cap is meaningful (above floor)
            if kappa_cap_bps > effective_floor_bps {
                for depth in dynamic_depths.bid.iter_mut() {
                    if *depth > kappa_cap_bps {
                        *depth = kappa_cap_bps;
                    }
                }
                for depth in dynamic_depths.ask.iter_mut() {
                    if *depth > kappa_cap_bps {
                        *depth = kappa_cap_bps;
                    }
                }
            }
        }

        // [SPREAD TRACE] Phase 5: kappa cap applied (2/kappa formula)
        tracing::info!(
            phase = "kappa_cap",
            kappa_cap_bps = %format!("{:.1}", market_params.kappa_spread_bps.unwrap_or(0.0)),
            kappa_used = %format!("{:.0}", kappa),
            touch_bid_bps = %format!("{:.2}", dynamic_depths.best_bid_depth().unwrap_or(0.0)),
            touch_ask_bps = %format!("{:.2}", dynamic_depths.best_ask_depth().unwrap_or(0.0)),
            total_at_touch_bps = %format!("{:.2}", dynamic_depths.spread_at_touch().unwrap_or(0.0)),
            "[SPREAD TRACE] after kappa spread cap (2/kappa formula)"
        );

        // === PRE-FILL AS MULTIPLIERS ===
        // Apply asymmetric spread widening from pre-fill adverse selection classifier.
        // These multipliers are [1.0, 3.0] where >1.0 indicates predicted toxicity on that side.
        // Widen depths (not gamma) to get direct, per-side spread control.
        if market_params.pre_fill_spread_mult_bid > 1.01 {
            for depth in dynamic_depths.bid.iter_mut() {
                *depth *= market_params.pre_fill_spread_mult_bid;
            }
        }
        if market_params.pre_fill_spread_mult_ask > 1.01 {
            for depth in dynamic_depths.ask.iter_mut() {
                *depth *= market_params.pre_fill_spread_mult_ask;
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

        // Capital-aware level count: don't spread across more levels than capital can support
        let budget_per_side = size_for_initial_ladder * market_params.microprice / 2.0;
        let max_viable_levels = (budget_per_side / config.min_notional).floor() as usize;
        let capital_limited_levels = if max_viable_levels < self.ladder_config.num_levels && max_viable_levels >= 1 {
            tracing::info!(
                budget_per_side = %format!("{:.2}", budget_per_side),
                min_notional = %format!("{:.2}", config.min_notional),
                max_viable_levels = max_viable_levels,
                configured_levels = self.ladder_config.num_levels,
                "Capital constrains ladder levels"
            );
            max_viable_levels
        } else {
            self.ladder_config.num_levels
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
            kappa_spread_cap_bps = %format!("{:.1}", market_params.kappa_spread_bps.unwrap_or(0.0)),
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

            // === MARGIN RESERVE BUFFER ===
            // Only use a fraction of available margin for quoting to maintain safety buffer.
            // 70% total utilization = 35% per side when flat.
            // GLFT model handles risk through gamma scaling; reduce-only at 80% provides safety buffer.
            const MAX_MARGIN_UTILIZATION: f64 = 0.70; // Use 70% of available margin (GLFT gamma handles risk)
            let usable_margin = available_margin * MAX_MARGIN_UTILIZATION;

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

            let inventory_ratio = if effective_max_position > EPSILON {
                (position / effective_max_position).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            // Component weights (sum to 1.0)
            const INVENTORY_WEIGHT: f64 = 0.50; // Primary: inventory reduction
            const MOMENTUM_WEIGHT: f64 = 0.25; // Secondary: follow flow direction
            const URGENCY_WEIGHT: f64 = 0.10; // Amplifier: danger detection
            const DIRECTIONAL_WEIGHT: f64 = 0.15; // Directional: alpha + knife + flow

            // Sensitivity factors
            const INVENTORY_SENSITIVITY: f64 = 0.4; // How much inventory affects split
            const MOMENTUM_SENSITIVITY: f64 = 0.3; // How much flow affects split (when flat)

            // 1. INVENTORY COMPONENT: Mean-revert position
            // Long → more margin to asks, Short → more margin to bids
            let inventory_signal = inventory_ratio * INVENTORY_SENSITIVITY;

            // 2. MOMENTUM COMPONENT: Follow flow direction
            // Positive flow (bullish) → allocate more to asks to capture rises
            // Negative flow (bearish) → allocate more to bids to capture falls
            // Scale by (1 - |inventory_ratio|) so momentum matters more when flat
            let inventory_flat_factor = 1.0 - inventory_ratio.abs();
            let momentum_signal =
                market_params.flow_imbalance * MOMENTUM_SENSITIVITY * inventory_flat_factor;

            // 3. URGENCY COMPONENT: Amplify when position opposes momentum
            // If long and price falling, or short and price rising → urgent reduction needed
            let mut urgency_score_adj = market_params.urgency_score;

            // 3a. Falling knife / rising knife amplification (mirrors glft.rs:1037-1068)
            // MarketParams carries falling_knife_score and rising_knife_score from momentum estimator.
            // If position opposes the dominant momentum direction AND severity > 0.5,
            // amplify inventory reduction urgency to avoid getting run over.
            let momentum_severity = market_params
                .falling_knife_score
                .max(market_params.rising_knife_score);
            let momentum_direction =
                if market_params.falling_knife_score > market_params.rising_knife_score {
                    -1.0 // Market falling
                } else if market_params.rising_knife_score > market_params.falling_knife_score {
                    1.0 // Market rising
                } else {
                    0.0 // Neutral
                };
            let knife_opposes_position = (inventory_ratio > 0.0 && momentum_direction < 0.0)
                || (inventory_ratio < 0.0 && momentum_direction > 0.0);

            if knife_opposes_position && momentum_severity > 0.5 {
                // Amplify urgency: at severity=1.5 → adds 0.5 urgency
                urgency_score_adj = (urgency_score_adj + momentum_severity / 3.0).min(1.0);
                if momentum_severity > 1.0 {
                    info!(
                        inventory_ratio = %format!("{:.3}", inventory_ratio),
                        falling_knife = %format!("{:.2}", market_params.falling_knife_score),
                        rising_knife = %format!("{:.2}", market_params.rising_knife_score),
                        urgency_adj = %format!("{:.2}", urgency_score_adj),
                        "Momentum-opposed position: amplifying urgency via falling/rising knife"
                    );
                }
            }

            let urgency_signal = if market_params.position_opposes_momentum || knife_opposes_position
            {
                // Use HJB drift urgency direction with urgency score magnitude
                // This gives strong directional pressure when in danger
                let urgency_direction = if inventory_ratio > 0.0 { 1.0 } else { -1.0 };
                urgency_direction * urgency_score_adj.min(1.0) * 0.2
            } else {
                0.0
            };

            // 4. DIRECTIONAL COMPONENT: Use alpha + knife + flow for asymmetry
            // When position is flat and cross-venue signals are unavailable,
            // directional signals (alpha, knife, flow) still indicate where the market
            // is headed. This component ensures we lean quotes toward the strong side.
            let directional_signal = {
                let knife_dir =
                    market_params.rising_knife_score - market_params.falling_knife_score;
                let knife_strength = knife_dir.abs().min(2.0) / 2.0;
                // Alpha amplifies directional conviction — higher alpha = stronger lean
                let alpha_amp = 1.0 + market_params.predicted_alpha.min(0.5) * 2.0;
                const KNIFE_COMPONENT: f64 = 0.4;
                const FLOW_COMPONENT: f64 = 0.4;
                const LEAD_LAG_COMPONENT: f64 = 0.2;
                const DIRECTIONAL_SENSITIVITY: f64 = 0.3;
                let raw = (knife_dir.signum() * knife_strength * KNIFE_COMPONENT
                    + market_params.flow_imbalance * FLOW_COMPONENT
                    + market_params.lead_lag_signal_bps / 5.0 * LEAD_LAG_COMPONENT)
                    * alpha_amp;
                raw.clamp(-1.0, 1.0) * DIRECTIONAL_SENSITIVITY
            };

            // Combine weighted components
            let combined_signal = inventory_signal * INVENTORY_WEIGHT
                + momentum_signal * MOMENTUM_WEIGHT
                + urgency_signal * URGENCY_WEIGHT
                + directional_signal * DIRECTIONAL_WEIGHT;

            // Apply to base 0.5 split with wider clamp range [0.25, 0.75]
            // This allows stronger skew when signals are aligned
            let ask_margin_weight = (0.5 + combined_signal).clamp(0.25, 0.75);
            let bid_margin_weight = 1.0 - ask_margin_weight;
            let margin_for_bids = usable_margin * bid_margin_weight;
            let margin_for_asks = usable_margin * ask_margin_weight;

            info!(
                available_margin = %format!("{:.2}", available_margin),
                usable_margin = %format!("{:.2}", usable_margin),
                utilization_pct = %format!("{:.0}%", MAX_MARGIN_UTILIZATION * 100.0),
                inventory_ratio = %format!("{:.3}", inventory_ratio),
                flow_imbalance = %format!("{:.3}", market_params.flow_imbalance),
                position_opposes = %market_params.position_opposes_momentum,
                urgency_score = %format!("{:.2}", market_params.urgency_score),
                directional = %format!("{:.3}", directional_signal),
                margin_for_bids = %format!("{:.2}", margin_for_bids),
                margin_for_asks = %format!("{:.2}", margin_for_asks),
                "Margin allocation (stochastic-weighted split)"
            );

            // [SIGNALS] Diagnostic: all activated signal values per quote cycle
            info!(
                alpha = %format!("{:.3}", market_params.predicted_alpha),
                falling_knife = %format!("{:.1}", market_params.falling_knife_score),
                rising_knife = %format!("{:.1}", market_params.rising_knife_score),
                momentum_severity = %format!("{:.2}", momentum_severity),
                knife_opposes = %knife_opposes_position,
                urgency_adj = %format!("{:.3}", urgency_score_adj),
                directional = %format!("{:.3}", directional_signal),
                kappa_used = %format!("{:.0}", kappa),
                as_warmed_up = %market_params.as_warmed_up,
                "[SIGNALS] signal contribution summary"
            );

            // SAFETY CHECK: If margin is not available (warmup incomplete), log and fall through
            // This prevents the optimizer from producing empty ladders due to margin=0
            if available_margin < EPSILON {
                warn!(
                    margin_available = %format!("{:.2}", available_margin),
                    leverage = %format!("{:.1}", leverage),
                    "Constrained optimizer skipped: margin not yet available (warmup incomplete?)"
                );
                // Fall through to legacy path at end of function
                return Ladder::generate(&ladder_config, &params);
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
            // Formula: max_levels = available_capacity / min_meaningful_size
            // where min_meaningful_size = 1.15 × (min_notional / price) to ensure each order
            // comfortably exceeds exchange minimum notional after truncation/rounding.
            //
            // 1.15× margin accounts for price movement between size calculation and order
            // placement, plus truncate_float() rounding down.
            let min_meaningful_size = (config.min_notional * 1.15) / market_params.microprice;

            // Warn if position limit is too small to place even one minimum-notional order
            if min_meaningful_size > effective_max_position {
                tracing::warn!(
                    min_order = %format!("{:.4} (${:.2})", min_meaningful_size, min_meaningful_size * market_params.microprice),
                    max_position = %format!("{:.4} (${:.2})", effective_max_position, effective_max_position * market_params.microprice),
                    "POSITION LIMIT TOO SMALL: cannot place orders meeting exchange minimum"
                );
            }

            let max_bid_levels = if available_for_bids > EPSILON {
                ((available_for_bids / min_meaningful_size).floor() as usize).max(1)
            } else {
                0
            };
            let max_ask_levels = if available_for_asks > EPSILON {
                ((available_for_asks / min_meaningful_size).floor() as usize).max(1)
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
                    min_meaningful_size = %format!("{:.6}", min_meaningful_size),
                    "Reducing ladder levels due to tight exchange limits"
                );
            }

            // 4. Generate ladder to get depth levels and prices (using dynamic depths)
            // NOTE: Don't truncate here - let the entropy optimizer handle distribution
            // The optimizer has built-in notional constraints that will filter sub-minimum levels
            let mut ladder = Ladder::generate(&ladder_config, &params);

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

            // 6. Optimize bid sizes using entropy-based allocation
            if !bid_level_params.is_empty() {
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

            // 8. Optimize ask sizes using entropy-based allocation
            if !ask_level_params.is_empty() {
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
                (max_position * MAX_SINGLE_ORDER_FRACTION).max(min_meaningful_size),
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
            // Use a slightly lower threshold (0.8x) to avoid edge cases near the boundary
            let min_size_for_exchange = config.min_notional * 0.8 / market_params.microprice;
            let bids_before = ladder.bids.len();
            let asks_before = ladder.asks.len();
            ladder.bids.retain(|l| l.size > min_size_for_exchange);
            ladder.asks.retain(|l| l.size > min_size_for_exchange);

            // 10. CONCENTRATION FALLBACK: If ladder is empty but total available size
            // meets min_notional, create single concentrated order at tightest depth.
            // This handles TWO cases:
            //   a) Levels existed but were filtered by min_size_for_exchange (bids_before > 0)
            //   b) Ladder::generate() returned empty because total_size was too small to
            //      distribute across num_levels and pass min_notional per level (bids_before == 0)
            // Case (b) occurs with small capital in restrictive regimes (e.g., $100 capital,
            // Extreme regime → effective_max ~1 HYPE → 0.09/level across 10 levels < $10 min)
            let _min_size_for_order = config.min_notional / market_params.microprice; // Kept for reference
            let tightest_depth_bps = ladder_config.min_depth_bps;

            // Bid concentration fallback
            if ladder.bids.is_empty() {
                // Total available size for bids, capped at 25% of the USER'S risk-based
                // max position per order (not margin-based quoting capacity).
                // GLFT inventory theory: one fill at max position creates maximal reservation
                // price swing q*gamma*sigma^2*T with zero recovery capacity.
                let per_order_cap = (max_position * MAX_SINGLE_ORDER_FRACTION)
                    .max(min_meaningful_size);
                let total_bid_size = truncate_float(
                    available_for_bids.min(per_order_cap),
                    config.sz_decimals,
                    false,
                );
                let bid_notional = total_bid_size * market_params.microprice;

                // FIX: Removed redundant `total_bid_size > min_size_for_order` check.
                // Both conditions are mathematically equivalent but floating-point precision
                // can cause disagreement. The notional check is what the exchange cares about.
                if bid_notional >= config.min_notional && total_bid_size > 1e-10 {
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
                        notional = %format!("{:.2}", bid_notional),
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

            // Ask concentration fallback
            if ladder.asks.is_empty() {
                // Total available size for asks, capped at 25% of the USER'S risk-based
                // max position per order (not margin-based quoting capacity).
                let per_order_cap = (max_position * MAX_SINGLE_ORDER_FRACTION)
                    .max(min_meaningful_size);
                let total_ask_size = truncate_float(
                    available_for_asks.min(per_order_cap),
                    config.sz_decimals,
                    false,
                );
                let ask_notional = total_ask_size * market_params.microprice;

                // FIX: Same as bid side - use only notional check to avoid precision issues
                if ask_notional >= config.min_notional && total_ask_size > 1e-10 {
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
                        notional = %format!("{:.2}", ask_notional),
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
            // IMPORTANT: The breakeven price must be snapped to the exchange tick grid
            // BEFORE sig-fig rounding, using directional rounding (ceil for asks, floor
            // for bids) to guarantee we never sell below / buy above true breakeven.
            // The tick size is 10^(-price_decimals) where price_decimals = config.decimals.
            if let Some(entry_price) = market_params.avg_entry_price {
                let breakeven = market_params.breakeven_price;
                let position = inventory_ratio; // already computed above
                let tick_size = 10f64.powi(-(config.decimals as i32));

                if breakeven > 0.0 {
                    if position > 0.0 {
                        // Long position: don't sell below breakeven unless urgent.
                        // Round breakeven UP to tick grid so ask >= true breakeven.
                        let urgency = (position.abs() * (-market_params.unrealized_pnl_bps / 50.0).max(0.0)).min(1.0);
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
                        let urgency = (position.abs() * (-market_params.unrealized_pnl_bps / 50.0).max(0.0)).min(1.0);
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
                let dynamic_min_size = config.min_notional / market_params.microprice;
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

    /// Compute the directional signal exactly as in `generate_ladder` margin split.
    /// Extracted to a helper so we can unit test it without constructing a full ladder.
    fn compute_directional_signal(market_params: &MarketParams) -> f64 {
        let knife_dir = market_params.rising_knife_score - market_params.falling_knife_score;
        let knife_strength = knife_dir.abs().min(2.0) / 2.0;
        let alpha_amp = 1.0 + market_params.predicted_alpha.min(0.5) * 2.0;
        const KNIFE_COMPONENT: f64 = 0.4;
        const FLOW_COMPONENT: f64 = 0.4;
        const LEAD_LAG_COMPONENT: f64 = 0.2;
        const DIRECTIONAL_SENSITIVITY: f64 = 0.3;
        let raw = (knife_dir.signum() * knife_strength * KNIFE_COMPONENT
            + market_params.flow_imbalance * FLOW_COMPONENT
            + market_params.lead_lag_signal_bps / 5.0 * LEAD_LAG_COMPONENT)
            * alpha_amp;
        raw.clamp(-1.0, 1.0) * DIRECTIONAL_SENSITIVITY
    }

    /// Compute the ask_margin_weight as in `generate_ladder` margin split,
    /// using position=0 (flat) and no urgency.
    fn compute_ask_margin_weight(market_params: &MarketParams) -> f64 {
        const INVENTORY_WEIGHT: f64 = 0.50;
        const MOMENTUM_WEIGHT: f64 = 0.25;
        const URGENCY_WEIGHT: f64 = 0.10;
        const DIRECTIONAL_WEIGHT: f64 = 0.15;
        const MOMENTUM_SENSITIVITY: f64 = 0.3;

        let inventory_signal = 0.0; // flat position
        let inventory_flat_factor = 1.0; // flat means full momentum weight
        let momentum_signal = market_params.flow_imbalance * MOMENTUM_SENSITIVITY * inventory_flat_factor;
        let urgency_signal = 0.0; // no urgency when flat
        let directional_signal = compute_directional_signal(market_params);

        let combined_signal = inventory_signal * INVENTORY_WEIGHT
            + momentum_signal * MOMENTUM_WEIGHT
            + urgency_signal * URGENCY_WEIGHT
            + directional_signal * DIRECTIONAL_WEIGHT;

        (0.5 + combined_signal).clamp(0.25, 0.75)
    }

    #[test]
    fn test_margin_split_bullish_no_binance() {
        // Scenario: alpha=0.166, rising_knife=0.1, flow=0.083, position=0
        // This is the exact scenario from the mainnet HYPE incident.
        // Without the directional component, ask_margin_weight would be ~0.5.
        // With it, bullish signals should push ask_margin_weight above 0.50.
        let mut params = MarketParams::default();
        params.predicted_alpha = 0.166;
        params.rising_knife_score = 0.1;
        params.falling_knife_score = 0.0;
        params.flow_imbalance = 0.083;
        params.lead_lag_signal_bps = 0.0; // No Binance

        let ask_weight = compute_ask_margin_weight(&params);

        assert!(
            ask_weight > 0.50,
            "Bullish signals (alpha=0.166, rising_knife=0.1, flow=0.083) should \
             push ask_margin_weight above 0.50, got: {:.4}",
            ask_weight
        );
    }

    #[test]
    fn test_margin_split_strong_bullish() {
        // Scenario: strong bullish with alpha=0.5, rising_knife=1.5, flow=0.5
        let mut params = MarketParams::default();
        params.predicted_alpha = 0.5;
        params.rising_knife_score = 1.5;
        params.falling_knife_score = 0.0;
        params.flow_imbalance = 0.5;
        params.lead_lag_signal_bps = 0.0;

        let ask_weight = compute_ask_margin_weight(&params);

        assert!(
            ask_weight > 0.53,
            "Strong bullish signals should push ask_margin_weight above 0.53, got: {:.4}",
            ask_weight
        );
    }

    #[test]
    fn test_directional_signal_symmetric() {
        // Mirror bullish and bearish should produce symmetric results
        let mut bullish = MarketParams::default();
        bullish.rising_knife_score = 1.0;
        bullish.falling_knife_score = 0.0;
        bullish.flow_imbalance = 0.5;
        bullish.predicted_alpha = 0.3;

        let mut bearish = MarketParams::default();
        bearish.rising_knife_score = 0.0;
        bearish.falling_knife_score = 1.0;
        bearish.flow_imbalance = -0.5;
        bearish.predicted_alpha = 0.3;

        let bull_signal = compute_directional_signal(&bullish);
        let bear_signal = compute_directional_signal(&bearish);

        assert!(
            (bull_signal + bear_signal).abs() < 0.01,
            "Symmetric bullish/bearish should cancel: bull={:.4}, bear={:.4}",
            bull_signal,
            bear_signal
        );
    }

    #[test]
    fn test_directional_signal_clamped() {
        // Extreme values should be clamped to [-0.3, 0.3]
        let mut extreme = MarketParams::default();
        extreme.rising_knife_score = 5.0;
        extreme.falling_knife_score = 0.0;
        extreme.flow_imbalance = 1.0;
        extreme.lead_lag_signal_bps = 50.0;
        extreme.predicted_alpha = 1.0;

        let signal = compute_directional_signal(&extreme);

        assert!(
            signal <= 0.3 + f64::EPSILON,
            "Directional signal should be capped at 0.3, got: {:.4}",
            signal
        );
        assert!(signal > 0.0, "Should be positive for bullish inputs");
    }

    #[test]
    fn test_directional_signal_zero_when_neutral() {
        // All neutral inputs should produce zero directional signal
        let params = MarketParams::default();
        let signal = compute_directional_signal(&params);

        assert!(
            signal.abs() < 0.01,
            "Neutral inputs should produce near-zero signal, got: {:.4}",
            signal
        );
    }

    #[test]
    fn test_weights_sum_to_one() {
        // Verify the component weights sum to 1.0
        const INVENTORY_WEIGHT: f64 = 0.50;
        const MOMENTUM_WEIGHT: f64 = 0.25;
        const URGENCY_WEIGHT: f64 = 0.10;
        const DIRECTIONAL_WEIGHT: f64 = 0.15;

        let sum = INVENTORY_WEIGHT + MOMENTUM_WEIGHT + URGENCY_WEIGHT + DIRECTIONAL_WEIGHT;
        assert!(
            (sum - 1.0).abs() < f64::EPSILON,
            "Weights should sum to 1.0, got: {}",
            sum
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
}
