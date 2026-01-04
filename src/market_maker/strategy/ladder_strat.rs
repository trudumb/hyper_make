//! GLFT Ladder Strategy - multi-level quoting with depth-dependent sizing.

use tracing::{debug, info, warn};

use crate::{truncate_float, EPSILON};

use crate::market_maker::config::{Quote, QuoteConfig};
use crate::market_maker::estimator::VolatilityRegime;
#[allow(deprecated)] // ConstrainedLadderOptimizer is deprecated but kept for legacy path
use crate::market_maker::quoting::{
    BayesianFillModel, ConstrainedLadderOptimizer, DepthSpacing, DynamicDepthConfig,
    DynamicDepthGenerator, EntropyConstrainedOptimizer, EntropyDistributionConfig,
    EntropyOptimizerConfig, KellyStochasticParams, Ladder, LadderConfig, LadderParams,
    LevelOptimizationParams, MarketRegime,
};

use super::{MarketParams, QuotingStrategy, RiskConfig};

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
        }
    }

    /// Create a new ladder strategy with custom configs.
    pub fn with_config(risk_config: RiskConfig, ladder_config: LadderConfig) -> Self {
        Self {
            depth_generator: Self::create_depth_generator(&ladder_config),
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
            risk_config,
            ladder_config,
            fill_model: BayesianFillModel::new(prior_alpha, prior_beta, sigma, tau),
        }
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
            // With fees_bps=3.5, floor=4.5bp means we capture at least 1bp after fees
            min_spread_floor_bps: spread_floor,
            enable_asymmetric: true, // Enable asymmetric bid/ask depths
            // DISABLED: Trust GLFT optimal spreads from first principles
            // Trade history showed spread cap caused -$562 loss over 9 days
            market_spread_cap_multiple: 0.0,
        };
        DynamicDepthGenerator::new(config)
    }

    /// Calculate effective γ based on current market conditions.
    /// (Same logic as GLFTStrategy for consistency)
    fn effective_gamma(
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

        // Volatility regime scaling
        let regime_scalar = match market_params.volatility_regime {
            VolatilityRegime::Low => 0.8,
            VolatilityRegime::Normal => 1.0,
            VolatilityRegime::High => 1.5,
            VolatilityRegime::Extreme => 2.5,
        };

        // Hawkes activity scaling
        let hawkes_scalar = if market_params.hawkes_activity_percentile > 0.9 {
            1.5
        } else if market_params.hawkes_activity_percentile > 0.8 {
            1.2
        } else {
            1.0
        };

        // Time-of-day scaling (toxic hours have higher adverse selection)
        // Trade history showed -13 to -15 bps edge during 06-08, 14-15 UTC
        let time_scalar = cfg.time_of_day_multiplier();

        // Book depth scaling (thin books → harder to exit → higher risk)
        // FIRST PRINCIPLES: This replaces the arbitrary stochastic_spread_multiplier
        let book_depth_scalar = cfg.book_depth_multiplier(market_params.near_touch_depth_usd);

        // Warmup uncertainty scaling (parameter uncertainty → more conservative)
        // FIRST PRINCIPLES: This replaces the arbitrary adaptive_uncertainty_factor
        let warmup_scalar = cfg.warmup_multiplier(market_params.adaptive_warmup_progress);

        // Calibration fill-hungry scaling (reduce gamma to attract fills during warmup)
        // FIRST PRINCIPLES: calibration_gamma_mult ∈ [0.3, 1.0]
        // Lower values = tighter quotes = more fills for calibration
        let calibration_scalar = market_params.calibration_gamma_mult;

        let gamma_effective = cfg.gamma_base
            * vol_scalar
            * toxicity_scalar
            * inventory_scalar
            * regime_scalar
            * hawkes_scalar
            * time_scalar
            * book_depth_scalar
            * warmup_scalar
            * calibration_scalar;
        gamma_effective.clamp(cfg.gamma_min, cfg.gamma_max)
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
        // Use margin-based quoting capacity for ladder allocation.
        // The user's max_position is ONLY used for reduce-only filter, not quoting capacity.
        // This allows the Kelly optimizer to allocate across full margin capacity.
        let quoting_capacity = market_params.quoting_capacity();
        let effective_max_position = if quoting_capacity > EPSILON {
            quoting_capacity
        } else {
            market_params.effective_max_position(max_position)
        };

        // ALWAYS log quoting capacity for debugging (INFO level for visibility)
        tracing::info!(
            user_max_position = %format!("{:.6}", max_position),
            quoting_capacity = %format!("{:.6}", quoting_capacity),
            margin_quoting_capacity = %format!("{:.6}", market_params.margin_quoting_capacity),
            margin_available = %format!("{:.2}", market_params.margin_available),
            leverage = %format!("{:.1}", market_params.leverage),
            effective_max_position = %format!("{:.6}", effective_max_position),
            "Quoting capacity: user max_position is for reduce-only only"
        );

        // === GAMMA: Adaptive vs Legacy ===
        // When adaptive spreads enabled: use log-additive shrinkage gamma
        // When disabled: use multiplicative RiskConfig gamma
        //
        // In BOTH paths, apply calibration_gamma_mult for fill-hungry mode during warmup.
        // calibration_gamma_mult ∈ [0.3, 1.0]: reduces gamma to tighten quotes for calibration fills.
        let calibration_scalar = market_params.calibration_gamma_mult;

        let gamma = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
            // Adaptive gamma: log-additive scaling prevents multiplicative explosion
            // Still apply tail risk multiplier for cascade protection
            // Apply calibration scalar for fill-hungry mode
            let adaptive_gamma = market_params.adaptive_gamma;
            debug!(
                adaptive_gamma = %format!("{:.4}", adaptive_gamma),
                tail_mult = %format!("{:.2}", market_params.tail_risk_multiplier),
                calibration_mult = %format!("{:.2}", calibration_scalar),
                warmup_pct = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
                "Ladder using ADAPTIVE gamma"
            );
            adaptive_gamma * market_params.tail_risk_multiplier * calibration_scalar
        } else {
            // Legacy: multiplicative RiskConfig gamma
            // Note: effective_gamma() already includes calibration_scalar
            let base_gamma = self.effective_gamma(market_params, position, effective_max_position);
            let gamma_with_liq = base_gamma * market_params.liquidity_gamma_mult;
            gamma_with_liq * market_params.tail_risk_multiplier
        };

        // === KAPPA: Adaptive vs Legacy ===
        // When adaptive spreads enabled: use blended book/own-fill kappa
        // When disabled: use book-only kappa with AS adjustment
        let kappa = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
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

        let time_horizon = self.holding_time(market_params.arrival_intensity);

        let inventory_ratio = if effective_max_position > EPSILON {
            (position / effective_max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Convert AS spread adjustment to bps for ladder generation
        let as_at_touch_bps = if market_params.as_warmed_up {
            market_params.as_spread_adjustment * 10000.0
        } else {
            0.0 // Use zero AS until warmed up
        };

        // Apply cascade size reduction
        let adjusted_size = target_liquidity * market_params.cascade_size_factor;

        let params = LadderParams {
            mid_price: market_params.microprice,
            sigma: market_params.sigma,
            kappa, // Use AS-adjusted kappa
            arrival_intensity: market_params.arrival_intensity,
            as_at_touch_bps,
            total_size: adjusted_size,
            inventory_ratio,
            gamma,
            time_horizon,
            decimals: config.decimals,
            sz_decimals: config.sz_decimals,
            min_notional: config.min_notional,
            depth_decay_as: market_params.depth_decay_as.clone(),
        };

        // === SPREAD FLOOR: Adaptive vs Static ===
        // When adaptive spreads enabled: use learned floor from Bayesian AS estimation
        // When disabled: use static RiskConfig floor + tick/latency constraints
        let effective_floor_frac = if market_params.use_adaptive_spreads
            && market_params.adaptive_can_estimate
        {
            // Adaptive floor: learned from actual fill AS + fees + safety buffer
            // During warmup, the prior-based floor is already conservative (fees + 3bps + 1.5σ)
            debug!(
                adaptive_floor_bps = %format!("{:.2}", market_params.adaptive_spread_floor * 10000.0),
                static_floor_bps = %format!("{:.2}", self.risk_config.min_spread_floor * 10000.0),
                warmup_pct = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
                "Ladder using ADAPTIVE spread floor"
            );
            market_params.adaptive_spread_floor
        } else {
            // Legacy: Static floor from RiskConfig + tick/latency constraints
            market_params.effective_spread_floor(self.risk_config.min_spread_floor)
        };
        let effective_floor_bps = effective_floor_frac * 10_000.0;

        // === DYNAMIC DEPTHS: GLFT-optimal depth computation ===
        // Compute depths from effective gamma and kappa using GLFT formula:
        // δ* = (1/γ) × ln(1 + γ/κ)
        //
        // With market spread cap: GLFT optimal is capped at 5× observed market spread
        // to ensure competitive quotes even when GLFT parameters suggest wider spreads.
        //
        // For asymmetric depths, we could use separate bid/ask kappa estimates.
        // Currently using symmetric kappa (same for both sides).
        let mut dynamic_depths = self.depth_generator.compute_depths_with_market_cap(
            gamma,
            kappa,
            kappa,
            market_params.sigma,
            market_params.market_spread_bps,
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

        // Create ladder config with dynamic depths
        let ladder_config = self
            .ladder_config
            .clone()
            .with_dynamic_depths(dynamic_depths);

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
            book_depth_usd = %format!("{:.0}", market_params.near_touch_depth_usd),
            warmup_pct = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
            adaptive_mode = market_params.use_adaptive_spreads && market_params.adaptive_can_estimate,
            "Ladder spread diagnostics (gamma includes book_depth + warmup scaling)"
        );

        // === STOCHASTIC MODULE: Constrained Ladder Optimization ===
        // Solves: max Σ λ(δᵢ) × SC(δᵢ) × sᵢ  (expected profit)
        // s.t. Σ sᵢ × margin_per_unit ≤ margin_available (margin constraint)
        //      Σ sᵢ ≤ effective_max_position (position constraint - first principles)
        //      Σ sᵢ ≤ exchange_limit (exchange-enforced constraint)
        tracing::debug!(
            use_constrained_optimizer = market_params.use_constrained_optimizer,
            "Checking constrained optimizer path"
        );
        if market_params.use_constrained_optimizer {
            // 1. Available margin comes from the exchange (already accounts for position margin)
            // NOTE: margin_available is already net of position margin, don't double-count!
            let leverage = market_params.leverage.max(1.0);
            let available_margin = market_params.margin_available;

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
            if over_limit {
                if position > 0.0 {
                    tracing::debug!(
                        position = %format!("{:.6}", position),
                        effective_max_position = %format!("{:.6}", effective_max_position),
                        available_bids = %format!("{:.6}", available_for_bids),
                        available_asks = %format!("{:.6}", available_for_asks),
                        "Over max position (LONG) - bids blocked, asks allowed for reducing"
                    );
                } else {
                    tracing::debug!(
                        position = %format!("{:.6}", position),
                        effective_max_position = %format!("{:.6}", effective_max_position),
                        available_bids = %format!("{:.6}", available_for_bids),
                        available_asks = %format!("{:.6}", available_for_asks),
                        "Over max position (SHORT) - asks blocked, bids allowed for reducing"
                    );
                }
            }

            // 4. Generate ladder to get depth levels and prices (using dynamic depths)
            let mut ladder = Ladder::generate(&ladder_config, &params);

            // 5. Create separate optimizers for bids and asks with their respective limits
            // These are only used when use_entropy_distribution=false (legacy path)
            #[allow(deprecated)]
            let optimizer_bids = ConstrainedLadderOptimizer::new(
                available_margin,
                available_for_bids,
                ladder_config.min_level_size,
                config.min_notional,
                market_params.microprice,
                leverage,
            );
            #[allow(deprecated)]
            let optimizer_asks = ConstrainedLadderOptimizer::new(
                available_margin,
                available_for_asks,
                ladder_config.min_level_size,
                config.min_notional,
                market_params.microprice,
                leverage,
            );

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

            // 6. Optimize bid sizes (entropy-based > Kelly-Stochastic > proportional MV)
            if !bid_level_params.is_empty() {
                let allocation = if market_params.use_entropy_distribution {
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
                        available_margin,
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

                    entropy_alloc.to_legacy()
                } else if market_params.use_kelly_stochastic {
                    // === KELLY-STOCHASTIC ALLOCATION (LEGACY) ===
                    // Apply volatility regime adjustment to Kelly fraction
                    // High vol → more conservative (lower Kelly)
                    // Low vol → more aggressive (higher Kelly)
                    let regime_multiplier =
                        market_params.volatility_regime.kelly_fraction_multiplier();
                    let dynamic_kelly =
                        (market_params.kelly_fraction * regime_multiplier).clamp(0.05, 0.75);

                    let kelly_params = KellyStochasticParams {
                        sigma: market_params.sigma,
                        time_horizon: market_params.kelly_time_horizon,
                        alpha_touch: market_params.kelly_alpha_touch,
                        alpha_decay_bps: market_params.kelly_alpha_decay_bps,
                        kelly_fraction: dynamic_kelly,
                    };

                    debug!(
                        kelly_fraction = %format!("{:.3}", dynamic_kelly),
                        regime = ?market_params.volatility_regime,
                        "Kelly-stochastic optimizer applied to bids (legacy)"
                    );

                    optimizer_bids.optimize_kelly_stochastic(&bid_level_params, &kelly_params)
                } else {
                    // === PROPORTIONAL MV ALLOCATION (LEGACY) ===
                    debug!("Proportional MV optimizer applied to bids (legacy)");
                    optimizer_bids.optimize(&bid_level_params)
                };

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

            // 8. Optimize ask sizes (entropy-based > Kelly-Stochastic > proportional MV)
            if !ask_level_params.is_empty() {
                let allocation = if market_params.use_entropy_distribution {
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
                        available_margin,
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

                    entropy_alloc.to_legacy()
                } else if market_params.use_kelly_stochastic {
                    // === KELLY-STOCHASTIC ALLOCATION (LEGACY) ===
                    // Apply volatility regime adjustment to Kelly fraction
                    // High vol → more conservative (lower Kelly)
                    // Low vol → more aggressive (higher Kelly)
                    let regime_multiplier =
                        market_params.volatility_regime.kelly_fraction_multiplier();
                    let dynamic_kelly =
                        (market_params.kelly_fraction * regime_multiplier).clamp(0.05, 0.75);

                    let kelly_params = KellyStochasticParams {
                        sigma: market_params.sigma,
                        time_horizon: market_params.kelly_time_horizon,
                        alpha_touch: market_params.kelly_alpha_touch,
                        alpha_decay_bps: market_params.kelly_alpha_decay_bps,
                        kelly_fraction: dynamic_kelly,
                    };

                    debug!(
                        kelly_fraction = %format!("{:.3}", dynamic_kelly),
                        regime = ?market_params.volatility_regime,
                        "Kelly-stochastic optimizer applied to asks (legacy)"
                    );

                    optimizer_asks.optimize_kelly_stochastic(&ask_level_params, &kelly_params)
                } else {
                    // === PROPORTIONAL MV ALLOCATION (LEGACY) ===
                    debug!("Proportional MV optimizer applied to asks (legacy)");
                    optimizer_asks.optimize(&ask_level_params)
                };

                for (i, &size) in allocation.sizes.iter().enumerate() {
                    if i < ladder.asks.len() {
                        // CRITICAL: Truncate to sz_decimals to prevent "Order has invalid size" rejections
                        ladder.asks[i].size = truncate_float(size, config.sz_decimals, false);
                    }
                }
            }

            // 9. Filter out zero-size levels
            let bids_before = ladder.bids.len();
            let asks_before = ladder.asks.len();
            ladder.bids.retain(|l| l.size > EPSILON);
            ladder.asks.retain(|l| l.size > EPSILON);

            // Diagnostic: warn if all levels were filtered out
            if bids_before > 0 && ladder.bids.is_empty() {
                warn!(
                    bids_before = bids_before,
                    available_margin = %format!("{:.2}", available_margin),
                    available_position = %format!("{:.6}", available_for_bids),
                    min_notional = %format!("{:.2}", config.min_notional),
                    min_level_size = %format!("{:.6}", ladder_config.min_level_size),
                    mid_price = %format!("{:.2}", market_params.microprice),
                    "All bid levels filtered out (sizes too small)"
                );
            }
            if asks_before > 0 && ladder.asks.is_empty() {
                warn!(
                    asks_before = asks_before,
                    available_margin = %format!("{:.2}", available_margin),
                    available_position = %format!("{:.6}", available_for_asks),
                    min_notional = %format!("{:.2}", config.min_notional),
                    min_level_size = %format!("{:.6}", ladder_config.min_level_size),
                    mid_price = %format!("{:.2}", market_params.microprice),
                    "All ask levels filtered out (sizes too small)"
                );
            }

            // DIAGNOSTIC: Detailed warning when ladder is completely empty
            // Helps identify margin, min_notional, or capacity constraints
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
                    "Ladder completely empty: all levels below min_notional or margin constraint"
                );
            }

            ladder
        } else {
            // Fallback path without constrained optimization (still uses dynamic depths)
            Ladder::generate(&ladder_config, &params)
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
