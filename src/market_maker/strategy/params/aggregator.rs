//! Parameter aggregation from multiple sources.

use crate::market_maker::adaptive::AdaptiveSpreadCalculator;
use crate::market_maker::adverse_selection::{AdverseSelectionEstimator, DepthDecayAS};
use crate::market_maker::config::{KellyTimeHorizonMethod, StochasticConfig};
use crate::market_maker::estimator::{MarketEstimator, ParameterEstimator};
use crate::market_maker::infra::MarginAwareSizer;
use crate::market_maker::process_models::{
    DriftAdjustedSkew, FundingRateEstimator, HJBInventoryController, HawkesOrderFlowEstimator,
    LiquidationCascadeDetector, SpreadProcessEstimator,
};
use crate::market_maker::belief::BeliefSnapshot;
use crate::market_maker::stochastic::StochasticControlBuilder;

use super::super::MarketParams;

/// Sources for parameter aggregation.
///
/// Bundles references to all modules that provide market parameters.
/// This replaces the 95-line parameter gathering in update_quotes.
pub struct ParameterSources<'a> {
    // Core estimator
    pub estimator: &'a ParameterEstimator,

    // Tier 1: Adverse selection
    pub adverse_selection: &'a AdverseSelectionEstimator,
    pub depth_decay_as: &'a DepthDecayAS,
    pub liquidation_detector: &'a LiquidationCascadeDetector,

    // Tier 2: Process models
    pub hawkes: &'a HawkesOrderFlowEstimator,
    pub funding: &'a FundingRateEstimator,
    pub spread_tracker: &'a SpreadProcessEstimator,

    // Stochastic modules
    pub hjb_controller: &'a HJBInventoryController,
    pub margin_sizer: &'a MarginAwareSizer,
    pub stochastic_config: &'a StochasticConfig,

    // Drift-adjusted skew (pre-computed from HJB controller + momentum)
    pub drift_adjusted_skew: DriftAdjustedSkew,

    // Adaptive Bayesian system
    pub adaptive_spreads: &'a AdaptiveSpreadCalculator,

    // Context
    pub position: f64,
    pub max_position: f64,
    pub latest_mid: f64,
    pub risk_aversion: f64,

    // Exchange position limits
    pub exchange_limits_valid: bool,
    pub exchange_effective_bid_limit: f64,
    pub exchange_effective_ask_limit: f64,
    pub exchange_limits_age_ms: u64,

    // Pending exposure from resting orders
    pub pending_bid_exposure: f64,
    pub pending_ask_exposure: f64,

    // Dynamic position limits (first principles)
    /// Dynamic max position VALUE from kill switch (USD).
    /// Derived from: min(leverage_limit, volatility_limit)
    pub dynamic_max_position_value: f64,
    /// Whether dynamic limit has been calculated from valid margin state.
    pub dynamic_limit_valid: bool,

    // Stochastic constraints (first principles)
    /// Asset tick size in basis points (from asset metadata).
    pub tick_size_bps: f64,
    /// Near-touch book depth (USD) within 5 bps of mid.
    pub near_touch_depth_usd: f64,

    // Calibration fill rate controller
    /// Gamma multiplier from calibration controller [0.3, 1.0].
    pub calibration_gamma_mult: f64,
    /// Calibration progress [0.0, 1.0].
    pub calibration_progress: f64,
    /// Whether calibration is complete.
    pub calibration_complete: bool,

    // Dynamic bounds (model-driven, replaces hardcoded CLI values)
    /// Whether to use dynamic kappa floor (true if no CLI override).
    pub use_dynamic_kappa_floor: bool,
    /// Whether to use dynamic spread ceiling (true if no CLI override).
    pub use_dynamic_spread_ceiling: bool,

    // First-principles belief system (DEPRECATED - Phase 7)
    /// DEPRECATED: Use `beliefs` field instead. This field is retained for
    /// fallback when beliefs is None, but beliefs_builder is no longer updated.
    /// Reference to beliefs_builder for Bayesian posterior-derived values.
    pub beliefs_builder: &'a StochasticControlBuilder,

    // === Centralized Belief State (Phase 5) ===
    /// Centralized belief snapshot for unified parameter access.
    /// When Some, these values are preferred over scattered sources.
    pub beliefs: Option<&'a BeliefSnapshot>,
}

/// Calculate Kelly time horizon based on config method.
///
/// For first-passage fill probability P(fill) = 2Φ(-δ/(σ√τ)) to be meaningful,
/// τ must be long enough for price to diffuse to quote depth.
///
/// # Methods
/// - **Fixed**: Use config-specified fixed tau
/// - **DiffusionBased**: τ = (δ_char / σ)² gives P(δ_char) ≈ 15.9%
/// - **ArrivalIntensity**: τ = 1/λ (legacy, typically too short)
fn calculate_kelly_time_horizon(
    config: &StochasticConfig,
    sigma: f64,
    arrival_intensity: f64,
) -> f64 {
    match config.kelly_time_horizon_method {
        KellyTimeHorizonMethod::Fixed => config.kelly_tau_fixed,
        KellyTimeHorizonMethod::DiffusionBased => {
            // τ = (δ_char / σ)² gives P(fill at δ_char) ≈ 15.9%
            // This scales with volatility: low vol → longer tau, high vol → shorter tau
            let depth_frac = config.kelly_char_depth_bps / 10000.0;
            let safe_sigma = sigma.max(1e-9);
            let tau = (depth_frac / safe_sigma).powi(2);
            tau.clamp(config.kelly_tau_min, config.kelly_tau_max)
        }
        KellyTimeHorizonMethod::ArrivalIntensity => {
            // Legacy behavior: τ = 1/λ
            // WARNING: This typically produces τ ~milliseconds, causing P(fill) ≈ 0
            (1.0 / arrival_intensity.max(0.01)).min(config.kelly_tau_max)
        }
    }
}

/// Aggregates parameters from multiple sources into MarketParams.
///
/// This struct provides a single method to build MarketParams from all
/// the various estimators and modules, replacing scattered parameter
/// gathering code.
pub struct ParameterAggregator;

impl ParameterAggregator {
    /// Build MarketParams from all sources.
    ///
    /// This replaces the 95-line parameter gathering block in update_quotes.
    pub fn build(sources: &ParameterSources) -> MarketParams {
        let est = sources.estimator;

        // Calculate derived values
        let arrival_intensity = est.arrival_intensity();
        let holding_time = 1.0 / arrival_intensity.max(0.01);

        MarketParams {
            // === Volatility (dual-sigma architecture) ===
            sigma: est.sigma_clean(),
            sigma_total: est.sigma_total(),
            sigma_effective: est.sigma_effective(),
            sigma_leverage_adjusted: est.sigma_leverage_adjusted(),
            volatility_regime: est.volatility_regime(),

            // === Order book / Liquidity ===
            kappa: est.kappa(),
            kappa_bid: est.kappa_bid(),
            kappa_ask: est.kappa_ask(),
            is_heavy_tailed: est.is_heavy_tailed(),
            kappa_cv: est.kappa_cv(),
            // V2: Uncertainty quantification (wired from V2 components)
            kappa_uncertainty: est.hierarchical_kappa_std(),
            kappa_95_lower: est.hierarchical_kappa_ci_95().0,
            kappa_95_upper: est.hierarchical_kappa_ci_95().1,
            // Compute normalized CI width for uncertainty-based warmup scaling
            // ci_width = (upper - lower) / mean, high uncertainty → wide CI → higher warmup gamma
            kappa_ci_width: {
                let (ci_lower, ci_upper) = est.hierarchical_kappa_ci_95();
                let kappa_mean = est.kappa();
                if ci_upper > ci_lower && kappa_mean > 0.0 {
                    (ci_upper - ci_lower) / kappa_mean
                } else {
                    1.0 // Default high uncertainty
                }
            },

            // V3: Robust Kappa Orchestrator (outlier-resistant)
            kappa_robust: est.kappa_robust(),
            use_kappa_robust: true, // Always enabled - uses confidence-weighted blending
            kappa_outlier_count: est.kappa_outlier_count(),

            toxicity_score: est.soft_toxicity_score(),
            param_correlation: est.kappa_sigma_correlation(),
            as_factor: est.hierarchical_as_factor(),
            arrival_intensity,
            liquidity_gamma_mult: est.liquidity_gamma_multiplier(),

            // === Regime detection ===
            is_toxic_regime: est.is_toxic_regime(),
            jump_ratio: est.jump_ratio(),

            // === Directional flow ===
            momentum_bps: est.momentum_bps(),
            flow_imbalance: est.flow_imbalance(),
            // Lead-lag signal (wired in quote_engine.rs when lag model is warmed up)
            lead_lag_signal_bps: 0.0,
            lead_lag_confidence: 0.0,
            falling_knife_score: est.falling_knife_score(),
            rising_knife_score: est.rising_knife_score(),

            // === L2 book structure ===
            book_imbalance: est.book_imbalance(),

            // === Microprice: data-driven fair price ===
            microprice: est.microprice(),
            market_mid: sources.latest_mid, // Raw exchange mid for safety checks
            beta_book: est.beta_book(),
            beta_flow: est.beta_flow(),

            // === Tier 1: Adverse Selection ===
            as_spread_adjustment: sources.adverse_selection.spread_adjustment(),
            predicted_alpha: sources.adverse_selection.predicted_alpha(),
            as_warmed_up: sources.adverse_selection.is_warmed_up(),
            depth_decay_as: Some(sources.depth_decay_as.clone()),

            // === Tier 1: Liquidation Cascade ===
            tail_risk_multiplier: sources.liquidation_detector.tail_risk_multiplier(),
            should_pull_quotes: sources.liquidation_detector.should_pull_quotes(),
            cascade_size_factor: sources.liquidation_detector.size_reduction_factor(),

            // === Tier 2: Hawkes Order Flow ===
            hawkes_buy_intensity: sources.hawkes.lambda_buy(),
            hawkes_sell_intensity: sources.hawkes.lambda_sell(),
            hawkes_imbalance: sources.hawkes.flow_imbalance(),
            hawkes_activity_percentile: sources.hawkes.intensity_percentile(),
            // Hawkes Excitation Prediction (Phase 7: Bayesian Fusion)
            // These will be populated from HawkesExcitationPredictor in orchestrator
            hawkes_p_cluster: 0.0,               // Default until predictor integrated
            hawkes_excitation_penalty: 1.0,      // Default full edge
            hawkes_is_high_excitation: false,    // Default not excited
            hawkes_branching_ratio: 0.3,         // Default moderate
            hawkes_spread_widening: 1.0,         // Default no widening
            hawkes_expected_cluster_time: f64::INFINITY, // Default no imminent
            hawkes_excess_intensity_ratio: 1.0,  // Default at baseline

            // Phase 8: RL Policy Recommendations
            // These will be populated from QLearningAgent in orchestrator
            rl_spread_delta_bps: 0.0,            // Default no adjustment
            rl_bid_skew_bps: 0.0,
            rl_ask_skew_bps: 0.0,
            rl_confidence: 0.0,                  // Default no confidence
            rl_is_exploration: false,
            rl_expected_q: 0.0,

            // Phase 8: Competitor Model
            // These will be populated from CompetitorModel in orchestrator
            competitor_snipe_prob: 0.1,          // Default 10% baseline
            competitor_spread_factor: 1.0,       // Default no adjustment
            competitor_count: 3.0,               // Default 3 competitors

            // === Tier 2: Funding Rate ===
            funding_rate: sources.funding.current_rate(),
            predicted_funding_cost: sources.funding.funding_cost(
                sources.position,
                sources.latest_mid,
                3600.0, // 1 hour holding period
            ),
            premium: sources.funding.current_premium(),
            premium_alpha: sources.funding.premium_alpha(),

            // === Tier 2: Spread Process ===
            fair_spread: sources.spread_tracker.fair_spread(),
            spread_percentile: sources.spread_tracker.spread_percentile(),
            spread_regime: sources.spread_tracker.spread_regime(),
            // Market spread from observed best bid/ask
            market_spread_bps: sources.spread_tracker.current_spread_bps(),

            // === First Principles Extensions (Gaps 1-10) ===
            // Jump-Diffusion (Gap 1)
            lambda_jump: est.lambda_jump(),
            mu_jump: est.mu_jump(),
            sigma_jump: est.sigma_jump(),

            // Stochastic Volatility (Gap 2)
            kappa_vol: est.kappa_vol(),
            theta_vol: est.theta_vol_sigma().powi(2),
            xi_vol: est.xi_vol(),
            rho_price_vol: est.rho_price_vol(),

            // Queue Model (Gap 3) - defaults until CalibratedQueueModel integrated
            calibrated_volume_rate: 1.0,
            calibrated_cancel_rate: 0.2,

            // Momentum Protection (Gap 10) - defaults until MomentumModel integrated
            bid_protection_factor: 1.0,
            ask_protection_factor: 1.0,
            p_momentum_continue: 0.5,

            // === Stochastic Module: HJB Controller ===
            use_hjb_skew: sources.stochastic_config.use_hjb_skew,
            hjb_optimal_skew: sources
                .hjb_controller
                .optimal_skew(sources.position, sources.max_position),
            hjb_gamma_multiplier: sources.hjb_controller.gamma_multiplier(),
            hjb_inventory_target: sources.hjb_controller.optimal_inventory_target(),
            hjb_is_terminal_zone: sources.hjb_controller.is_terminal_zone(),

            // === Drift-Adjusted Skew (First Principles Extension) ===
            // Compute drift-adjusted skew if momentum data available
            use_drift_adjusted_skew: sources.stochastic_config.use_hjb_skew, // Enabled when HJB is enabled
            hjb_drift_urgency: sources.drift_adjusted_skew.drift_urgency,
            directional_variance_mult: sources.drift_adjusted_skew.variance_multiplier,
            position_opposes_momentum: sources.drift_adjusted_skew.is_opposed,
            urgency_score: sources.drift_adjusted_skew.urgency_score,

            // === Stochastic Module: Kalman Filter ===
            use_kalman_filter: sources.stochastic_config.use_kalman_filter,
            kalman_fair_price: est.kalman_fair_price(),
            kalman_uncertainty: est.kalman_uncertainty(),
            kalman_spread_widening: est.kalman_spread_widening(sources.risk_aversion, holding_time),
            kalman_warmed_up: est.kalman_warmed_up(),

            // === Stochastic Module: Constrained Optimizer ===
            margin_available: sources.margin_sizer.state().available_margin,
            // NOTE: Use max_leverage (allowed leverage) not current_leverage (how levered we are)
            // This determines how much margin is needed per unit of position
            leverage: sources.margin_sizer.summary().max_leverage,

            // === Stochastic Module: Kelly-Stochastic Allocation ===
            kelly_alpha_touch: sources.stochastic_config.kelly_alpha_touch,
            kelly_alpha_decay_bps: sources.stochastic_config.kelly_alpha_decay_bps,
            kelly_fraction: sources.stochastic_config.kelly_fraction,
            kelly_time_horizon: calculate_kelly_time_horizon(
                sources.stochastic_config,
                est.sigma_clean(),
                arrival_intensity,
            ),

            // === Exchange Position Limits ===
            // These are populated from infra.exchange_limits in the MarketMaker
            // Default to "not valid" - will be set explicitly when exchange limits are fetched
            exchange_limits_valid: sources.exchange_limits_valid,
            exchange_effective_bid_limit: sources.exchange_effective_bid_limit,
            exchange_effective_ask_limit: sources.exchange_effective_ask_limit,
            exchange_limits_age_ms: sources.exchange_limits_age_ms,

            // === Pending Exposure ===
            // Resting orders that would change position if filled
            pending_bid_exposure: sources.pending_bid_exposure,
            pending_ask_exposure: sources.pending_ask_exposure,

            // === Dynamic Position Limits (First Principles) ===
            // Convert VALUE limit to SIZE limit: max_position = max_value / price
            // Get margin-derived values for fallback
            dynamic_max_position: {
                let margin_available = sources.margin_sizer.state().available_margin;
                let leverage = sources.margin_sizer.summary().max_leverage;

                if sources.dynamic_limit_valid && sources.latest_mid > 0.0 {
                    sources.dynamic_max_position_value / sources.latest_mid
                } else if margin_available > 0.0 && leverage > 0.0 && sources.latest_mid > 0.0 {
                    // CONTROLLER-DERIVED: Use margin-based capacity, not user's arbitrary limit
                    // This ensures Kelly optimizer can allocate across full margin capacity
                    (margin_available * leverage / sources.latest_mid).max(0.0)
                } else {
                    sources.max_position // Last resort: static limit during early warmup
                }
            },
            dynamic_limit_valid: sources.dynamic_limit_valid,

            // Margin-based quoting capacity (pure solvency constraint)
            // This is the HARD limit from available margin, ignoring user's max_position
            margin_quoting_capacity: {
                let margin_available = sources.margin_sizer.state().available_margin;
                let leverage = sources.margin_sizer.summary().max_leverage;

                let capacity =
                    if margin_available > 0.0 && leverage > 0.0 && sources.latest_mid > 0.0 {
                        (margin_available * leverage / sources.latest_mid).max(0.0)
                    } else {
                        0.0 // Will be computed dynamically via quoting_capacity()
                    };

                // Debug: trace margin_quoting_capacity computation
                tracing::debug!(
                    margin_available = %format!("{:.2}", margin_available),
                    leverage = %format!("{:.1}", leverage),
                    mid = %format!("{:.4}", sources.latest_mid),
                    computed_capacity = %format!("{:.6}", capacity),
                    "margin_quoting_capacity computed"
                );

                capacity
            },

            // === Stochastic Constraints (First Principles) ===
            // These are computed dynamically via compute_stochastic_constraints()
            tick_size_bps: sources.tick_size_bps,
            latency_spread_floor: 0.0, // Computed dynamically
            near_touch_depth_usd: sources.near_touch_depth_usd,
            tight_quoting_allowed: false, // Conservative default, computed by caller
            tight_quoting_block_reason: Some("Warmup".to_string()),
            // NOTE: stochastic_spread_multiplier removed - uncertainty flows through gamma

            // === Adaptive Bayesian System ===
            use_adaptive_spreads: sources.stochastic_config.use_adaptive_spreads,
            adaptive_spread_floor: sources.adaptive_spreads.spread_floor(),
            adaptive_kappa: sources.adaptive_spreads.kappa(sources.estimator.kappa()),
            adaptive_gamma: sources.adaptive_spreads.gamma(
                sources.risk_aversion,
                sources.estimator.sigma_effective()
                    / sources.stochastic_config.sigma_baseline.max(1e-9),
                sources.estimator.jump_ratio(),
                sources
                    .adaptive_spreads
                    .inventory_utilization(sources.position, sources.max_position),
                sources.hawkes.intensity_percentile(),
                sources.liquidation_detector.cascade_severity(),
            ),
            adaptive_spread_ceiling: sources.adaptive_spreads.spread_ceiling(),

            // === Dynamic Bounds (Model-Driven, Replaces Hardcoded CLI Values) ===
            // These are computed from Bayesian models to replace arbitrary --kappa-floor and --max-spread-bps
            dynamic_kappa_floor: if sources.use_dynamic_kappa_floor {
                Some(sources.estimator.dynamic_kappa_floor())
            } else {
                None // CLI override active, static floor in use
            },
            dynamic_spread_ceiling_bps: if sources.use_dynamic_spread_ceiling {
                // Get market p80 from spread tracker
                let market_p80_bps = sources.spread_tracker.spread_p80_bps();
                // Combine with fill controller ceiling
                sources
                    .adaptive_spreads
                    .dynamic_spread_ceiling(market_p80_bps)
            } else {
                None // CLI override active, static ceiling in use
            },
            use_dynamic_bounds: sources.use_dynamic_kappa_floor
                || sources.use_dynamic_spread_ceiling,

            adaptive_warmed_up: sources.adaptive_spreads.is_warmed_up(),
            adaptive_can_estimate: sources.adaptive_spreads.can_provide_estimates(),
            adaptive_warmup_progress: sources.adaptive_spreads.warmup_progress(),
            adaptive_uncertainty_factor: sources.adaptive_spreads.warmup_uncertainty_factor(),

            // === Entropy-Based Distribution (always enabled) ===
            entropy_min_entropy: sources.stochastic_config.entropy_min_entropy,
            entropy_base_temperature: sources.stochastic_config.entropy_base_temperature,
            entropy_min_allocation_floor: sources.stochastic_config.entropy_min_allocation_floor,
            entropy_thompson_samples: sources.stochastic_config.entropy_thompson_samples,

            // === Calibration Fill Rate Controller ===
            calibration_gamma_mult: sources.calibration_gamma_mult,
            calibration_progress: sources.calibration_progress,
            calibration_complete: sources.calibration_complete,

            // === Model-Derived Sizing (GLFT First Principles) ===
            // These are set from margin state and WS latency measurements
            account_value: sources.margin_sizer.state().account_value,
            measured_latency_ms: 0.0, // Will be set from WS manager in mod.rs
            estimated_fill_rate: sources
                .adaptive_spreads
                .fill_rate_controller()
                .observed_fill_rate()
                .max(0.001), // Floor at 0.001/sec
            derived_target_liquidity: 0.0, // Computed by compute_derived_target_liquidity()

            // === New Latent State Estimators (Phases 2-7) ===
            // Particle Filter Volatility (Phase 2)
            // Phase 5: Use centralized beliefs when available
            sigma_particle: if let Some(beliefs) = sources.beliefs {
                beliefs.drift_vol.expected_sigma
            } else {
                est.sigma_particle_filter()
            },
            regime_probs: if let Some(beliefs) = sources.beliefs {
                beliefs.regime.probs
            } else {
                est.regime_probabilities()
            },

            // Informed Flow Model (Phase 3)
            p_informed: est.p_informed(),
            p_noise: est.p_noise(),
            p_forced: est.p_forced(),
            flow_decomp_confidence: est.flow_decomposition_confidence(),

            // Fill Rate Model (Phase 4)
            fill_rate_8bps: est.fill_rate_at_depth(8.0),
            optimal_depth_50pct: est.optimal_depth_for_fill_rate(0.5),

            // Adverse Selection Decomposition (Phase 5)
            as_permanent_bps: est.as_permanent_bps(),
            as_temporary_bps: est.as_temporary_bps(),
            as_timing_bps: est.as_timing_bps(),
            total_as_bps: est.total_as_bps(),

            // Edge Surface (Phase 6)
            current_edge_bps: est.current_edge_bps(),
            should_quote_edge: est.should_quote_edge(),

            // Joint Dynamics (Phase 7)
            is_toxic_joint: est.is_toxic_joint(),
            sigma_kappa_correlation: est.sigma_kappa_correlation(),

            // === L2 Decision Engine Outputs (A-S Framework) ===
            // Set by QuoteEngine after decision_engine.should_quote() call
            // Default to neutral values; actual values populated in quote_engine.rs
            l2_reservation_shift: 0.0, // No shift initially
            // NOTE: l2_spread_multiplier removed - uncertainty flows through gamma

            // === Proactive Position Management (Small Fish Strategy) ===
            // These are populated from stochastic config for GLFT access
            enable_proactive_skew: sources.stochastic_config.enable_proactive_skew,
            proactive_skew_sensitivity: sources.stochastic_config.proactive_skew_sensitivity,
            proactive_min_momentum_confidence: sources
                .stochastic_config
                .proactive_min_momentum_confidence,
            proactive_min_momentum_bps: sources.stochastic_config.proactive_min_momentum_bps,

            // Kappa-Driven Spread (Phase 3) - computed separately in quote_engine
            kappa_spread_bps: None,

            // === Bayesian Gamma Components (Alpha Plan) ===
            // These are populated from QuoteGate and TheoreticalEdgeEstimator
            // Default to neutral values; actual values populated in quote_engine.rs
            // Phase 5: Use centralized beliefs when available
            trend_confidence: if let Some(beliefs) = sources.beliefs {
                beliefs.continuation.signal_summary.trend_confidence
            } else {
                0.5 // 50% initially (uncertain)
            },
            bootstrap_confidence: 0.0,    // Not calibrated initially
            adverse_uncertainty: 0.1,     // Moderate uncertainty
            adverse_regime: 1,            // Normal regime
            bayesian_gamma_mult: 1.0,     // No adjustment until computed

            // === Phase 9: Rate Limit Death Spiral Prevention ===
            rate_limit_headroom_pct: 1.0, // Full budget available by default, updated by caller

            // === Predictive Bias (A-S Extension) ===
            // Phase 5: Use centralized beliefs when available
            changepoint_prob: if let Some(beliefs) = sources.beliefs {
                beliefs.changepoint.prob_5
            } else {
                0.0 // Populated from stochastic controller in quote_engine
            },
            spread_widening_mult: 1.0,    // Populated from QuoteGate when pending confirmation

            // === First-Principles Stochastic Control (Belief System) ===
            // Phase 7: Centralized beliefs are the single source of truth.
            // Fallbacks to beliefs_builder are DEPRECATED and only for safety.
            belief_predictive_bias: if let Some(beliefs) = sources.beliefs {
                beliefs.drift_vol.expected_drift
            } else {
                // DEPRECATED: beliefs_builder is no longer updated
                sources.beliefs_builder.predictive_bias()
            },
            belief_expected_sigma: {
                // Phase 7: Centralized beliefs are primary
                if let Some(beliefs) = sources.beliefs {
                    if beliefs.is_warmed_up() && beliefs.drift_vol.expected_sigma > 0.0 {
                        beliefs.drift_vol.expected_sigma
                    } else {
                        est.sigma_clean()
                    }
                } else {
                    // DEPRECATED fallback: beliefs_builder is no longer updated
                    let belief_sigma = sources.beliefs_builder.expected_sigma();
                    if sources.beliefs_builder.is_warmed_up() && belief_sigma > 0.0 {
                        belief_sigma
                    } else {
                        est.sigma_clean()
                    }
                }
            },
            belief_expected_kappa: {
                // Phase 7: Centralized beliefs are primary
                if let Some(beliefs) = sources.beliefs {
                    if beliefs.is_warmed_up() && beliefs.kappa.kappa_effective > 0.0 {
                        beliefs.kappa.kappa_effective
                    } else {
                        est.kappa()
                    }
                } else {
                    // DEPRECATED fallback: beliefs_builder is no longer updated
                    let belief_kappa = sources.beliefs_builder.expected_kappa();
                    if sources.beliefs_builder.is_warmed_up() && belief_kappa > 0.0 {
                        belief_kappa
                    } else {
                        est.kappa()
                    }
                }
            },
            belief_confidence: if let Some(beliefs) = sources.beliefs {
                beliefs.overall_confidence()
            } else {
                // DEPRECATED fallback: beliefs_builder is no longer updated
                sources.beliefs_builder.beliefs().overall_confidence()
            },
            use_belief_system: sources.beliefs_builder.config().enable_belief_system,
            // Position direction confidence - will be computed in RiskFeatures::from_params
            // using the fields above (belief_predictive_bias, belief_confidence, flow_imbalance)
            position_direction_confidence: 0.5, // Default neutral, computed at use site
            time_since_adverse_move: 0.0,       // Tracked externally in orchestrator state

            // === Position Continuation Model (HOLD/ADD/REDUCE) ===
            // Defaults populated here; actual values computed in quote_engine.rs
            // from PositionDecisionEngine.decide() based on fills and regime
            // Phase 5: Use centralized beliefs when available
            position_action: super::super::PositionAction::default(),
            continuation_p: if let Some(beliefs) = sources.beliefs {
                beliefs.continuation.p_fused
            } else {
                0.5 // Default 50% continuation (uninformed prior)
            },
            continuation_confidence: if let Some(beliefs) = sources.beliefs {
                beliefs.continuation.confidence_fused
            } else {
                0.0 // Default no confidence (will be computed)
            },
            effective_inventory_ratio: 0.0, // Default no transformation (computed from position_action)
        }
    }
}
