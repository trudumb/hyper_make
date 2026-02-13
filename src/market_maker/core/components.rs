//! Component bundles for MarketMaker.
//!
//! Groups related modules into logical bundles for cleaner organization.

use crate::market_maker::{
    adaptive::{AdaptiveBayesianConfig, AdaptiveSpreadCalculator},
    adverse_selection::{
        AdverseSelectionConfig, AdverseSelectionEstimator, DepthDecayAS, EnhancedASClassifier,
        PreFillASClassifier,
    },
    analytics::{EdgeTracker, MarketToxicityComposite, MarketToxicityConfig},
    config::{ImpulseControlConfig, MetricsRecorder},
    control::{
        CalibratedEdgeConfig, CalibratedEdgeSignal, PositionPnLConfig, PositionPnLTracker,
        QuoteGate, StochasticController, StochasticControllerConfig,
        TheoreticalEdgeEstimator,
    },
    stochastic::{StochasticControlBuilder, StochasticControlConfig},
    strategy::{PositionDecisionConfig, PositionDecisionEngine, SignalIntegrator, SignalIntegratorConfig,
               regime_state::RegimeState},
    estimator::{
        CalibrationController, CalibrationControllerConfig, RegimeHMM,
        EnhancedFlowConfig, EnhancedFlowEstimator,
        LiquidityEvaporationConfig, LiquidityEvaporationDetector,
        VpinConfig, VpinEstimator,
        CumulativeOFI, CumulativeOFIConfig,
        TradeSizeDistribution, TradeSizeDistributionConfig,
        BOCPDKappaConfig, BOCPDKappaPredictor,
        ThresholdKappa, ThresholdKappaConfig,
        TradeFlowTracker,
    },
    calibration::{LearnedParameters, SignalDecayTracker},
    quoting::{KappaSpreadConfig, KappaSpreadController},
    simulation::{QuickMCConfig, QuickMCSimulator},
    execution::{FillTracker, OrderLifecycleTracker},
    fills::{FillProcessor, FillSignalStore},
    infra::{
        ConnectionHealthMonitor, ConnectionSupervisor, DataQualityConfig, DataQualityMonitor,
        ExchangePositionLimits, ExecutionBudget, MarginAwareSizer, MarginConfig, OrphanTracker,
        OrphanTrackerConfig, PositionReconciler, ProactiveRateLimitConfig,
        ProactiveRateLimitTracker, PrometheusMetrics, ReconciliationConfig, RecoveryConfig,
        RecoveryManager, RejectionRateLimitConfig, RejectionRateLimiter, SupervisorConfig,
    },
    learning::AdaptiveEnsemble,
    monitoring::{AlertConfig, Alerter, DashboardState},
    process_models::{
        FundingConfig, FundingRateEstimator, HJBConfig, HJBInventoryController, HawkesConfig,
        HawkesOrderFlowEstimator, LiquidationCascadeDetector, LiquidationConfig, SpreadConfig,
        SpreadProcessEstimator,
    },
    risk::{
        CircuitBreakerConfig, CircuitBreakerMonitor, DrawdownConfig, DrawdownTracker, KillSwitch,
        KillSwitchConfig, PositionGuard, RiskAggregator, RiskChecker, RiskLimits,
    },
    tracking::{
        ImpulseFilter, ModelCalibrationOrchestrator, PnLConfig, PnLTracker, QueueConfig,
        QueuePositionTracker,
    },
    DynamicRiskConfig, StochasticConfig,
};

/// Tier 1 components: Production resilience modules.
///
/// These modules are critical for production trading:
/// - Adverse selection measurement
/// - Pre-fill toxicity prediction
/// - Queue position tracking
/// - Liquidation cascade detection
/// - Circuit breaker monitoring
pub struct Tier1Components {
    /// Adverse selection estimator (post-fill measurement)
    pub adverse_selection: AdverseSelectionEstimator,
    /// Depth-dependent AS model
    pub depth_decay_as: DepthDecayAS,
    /// Pre-fill AS classifier (toxicity prediction BEFORE fills)
    pub pre_fill_classifier: PreFillASClassifier,
    /// Enhanced AS classifier with microstructure features
    /// Uses z-score normalized features and online learning
    pub enhanced_classifier: EnhancedASClassifier,
    /// Queue position tracker
    pub queue_tracker: QueuePositionTracker,
    /// Liquidation cascade detector
    pub liquidation_detector: LiquidationCascadeDetector,
    /// Circuit breaker for market condition monitoring
    pub circuit_breaker: CircuitBreakerMonitor,
}

impl Tier1Components {
    /// Create Tier 1 components from configs.
    pub fn new(
        as_config: AdverseSelectionConfig,
        queue_config: QueueConfig,
        liquidation_config: LiquidationConfig,
    ) -> Self {
        Self::with_circuit_breaker(
            as_config,
            queue_config,
            liquidation_config,
            CircuitBreakerConfig::default(),
        )
    }

    /// Create Tier 1 components with custom circuit breaker config.
    pub fn with_circuit_breaker(
        as_config: AdverseSelectionConfig,
        queue_config: QueueConfig,
        liquidation_config: LiquidationConfig,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> Self {
        Self {
            adverse_selection: AdverseSelectionEstimator::new(as_config),
            depth_decay_as: DepthDecayAS::default(),
            pre_fill_classifier: PreFillASClassifier::default(),
            enhanced_classifier: EnhancedASClassifier::default_config(),
            queue_tracker: QueuePositionTracker::new(queue_config),
            liquidation_detector: LiquidationCascadeDetector::new(liquidation_config),
            circuit_breaker: CircuitBreakerMonitor::new(circuit_breaker_config),
        }
    }
}

/// Tier 2 components: Process models.
///
/// These modules provide market process estimation:
/// - Hawkes order flow
/// - Funding rate
/// - Spread dynamics
/// - P&L tracking
pub struct Tier2Components {
    /// Hawkes order flow estimator
    pub hawkes: HawkesOrderFlowEstimator,
    /// Funding rate estimator
    pub funding: FundingRateEstimator,
    /// Spread process estimator
    pub spread_tracker: SpreadProcessEstimator,
    /// P&L tracker
    pub pnl_tracker: PnLTracker,
    /// Edge validation tracker (predicted vs realized edge)
    pub edge_tracker: EdgeTracker,
    /// Proactive market toxicity composite scorer
    pub toxicity: MarketToxicityComposite,
}

impl Tier2Components {
    /// Create Tier 2 components from configs.
    pub fn new(
        hawkes_config: HawkesConfig,
        funding_config: FundingConfig,
        spread_config: SpreadConfig,
        pnl_config: PnLConfig,
    ) -> Self {
        Self {
            hawkes: HawkesOrderFlowEstimator::new(hawkes_config),
            funding: FundingRateEstimator::new(funding_config),
            spread_tracker: SpreadProcessEstimator::new(spread_config),
            pnl_tracker: PnLTracker::new(pnl_config),
            edge_tracker: EdgeTracker::new(),
            toxicity: MarketToxicityComposite::new(MarketToxicityConfig::default()),
        }
    }
}

/// Safety components: Risk management and kill switch.
pub struct SafetyComponents {
    /// Kill switch
    pub kill_switch: KillSwitch,
    /// Risk aggregator
    pub risk_aggregator: RiskAggregator,
    /// Fill processor
    pub fill_processor: FillProcessor,
    /// Position and order limit checker
    pub risk_checker: RiskChecker,
    /// Equity drawdown tracker
    pub drawdown_tracker: DrawdownTracker,
    /// Signal diagnostic store for fill analysis
    pub signal_store: FillSignalStore,
    /// Pre-order position guard (hard entry gate)
    pub position_guard: PositionGuard,
}

impl SafetyComponents {
    /// Create safety components from config.
    pub fn new(
        kill_switch_config: KillSwitchConfig,
        risk_aggregator: RiskAggregator,
        max_position: f64,
        gamma: f64,
    ) -> Self {
        Self::with_risk_limits(
            kill_switch_config,
            risk_aggregator,
            RiskLimits::default(),
            DrawdownConfig::default(),
            max_position,
            gamma,
        )
    }

    /// Create safety components with custom risk limits and drawdown config.
    pub fn with_risk_limits(
        kill_switch_config: KillSwitchConfig,
        risk_aggregator: RiskAggregator,
        risk_limits: RiskLimits,
        drawdown_config: DrawdownConfig,
        max_position: f64,
        gamma: f64,
    ) -> Self {
        Self {
            kill_switch: KillSwitch::new(kill_switch_config),
            risk_aggregator,
            fill_processor: FillProcessor::new(),
            risk_checker: RiskChecker::new(risk_limits),
            // Initial equity of 10,000 is a placeholder; updated on first margin refresh
            drawdown_tracker: DrawdownTracker::new(drawdown_config, 10_000.0),
            signal_store: FillSignalStore::new(),
            position_guard: PositionGuard::new(max_position, gamma),
        }
    }
}

/// Infrastructure components: Monitoring and execution infrastructure.
pub struct InfraComponents {
    /// Margin-aware sizer
    pub margin_sizer: MarginAwareSizer,
    /// Exchange-enforced position limits (from active_asset_data API)
    pub exchange_limits: ExchangePositionLimits,
    /// Prometheus metrics
    pub prometheus: PrometheusMetrics,
    /// Connection health monitor (low-level tracking)
    pub connection_health: ConnectionHealthMonitor,
    /// Connection supervisor (high-level proactive monitoring)
    pub connection_supervisor: ConnectionSupervisor,
    /// Data quality monitor
    pub data_quality: DataQualityMonitor,
    /// Optional metrics recorder
    pub metrics: MetricsRecorder,
    /// Last margin refresh time
    pub last_margin_refresh: std::time::Instant,
    /// Recovery manager for stuck reduce-only mode (Phase 3)
    pub recovery_manager: RecoveryManager,
    /// Position reconciler for drift detection (Phase 4)
    pub reconciler: PositionReconciler,
    /// Rejection-aware rate limiter (Phase 5)
    pub rate_limiter: RejectionRateLimiter,
    /// Proactive rate limit tracker (Phase 6)
    /// Tracks API usage to avoid hitting Hyperliquid limits
    pub proactive_rate_tracker: ProactiveRateLimitTracker,
    /// Orphan order tracker (Phase 7)
    /// Prevents false orphan detection during order lifecycle
    pub orphan_tracker: OrphanTracker,
    /// Execution budget for statistical impulse control (Phase 8)
    /// Token-based budget for gating API calls
    pub execution_budget: ExecutionBudget,
    /// Impulse filter for Δλ-based update gating (Phase 8)
    /// Only updates orders when fill probability improvement exceeds threshold
    pub impulse_filter: ImpulseFilter,
    /// Whether impulse control is enabled
    pub impulse_control_enabled: bool,
    /// Cached exchange rate limit (from userRateLimit API)
    pub cached_rate_limit: Option<CachedRateLimit>,
    /// Monitoring alerts
    pub alerter: Alerter,
    /// Dashboard state for display
    pub dashboard: DashboardState,
    /// Execution quality tracker
    pub fill_tracker: FillTracker,
    /// Order lifecycle tracker
    pub order_lifecycle: OrderLifecycleTracker,
    /// Pending fill outcomes for 5-second adverse selection markout.
    /// Fills are pushed here; on each mid update, expired entries are drained
    /// and fed to the pre-fill classifier and model gating.
    pub pending_fill_outcomes: std::collections::VecDeque<crate::market_maker::fills::PendingFillOutcome>,
    /// Whether the emergency pull is currently active (for hysteresis).
    /// Once triggered, stays active until momentum drops below the off-ramp threshold.
    pub emergency_pull_active: bool,
}

/// Cached exchange rate limit with timestamp.
#[derive(Debug, Clone)]
pub struct CachedRateLimit {
    /// Requests used (from exchange)
    pub n_requests_used: u64,
    /// Requests cap (from exchange)
    pub n_requests_cap: u64,
    /// Requests surplus
    pub n_requests_surplus: u64,
    /// When this was last fetched
    pub fetched_at: std::time::Instant,
}

impl CachedRateLimit {
    /// Create from UserRateLimitResponse.
    pub fn from_response(resp: &crate::info::response_structs::UserRateLimitResponse) -> Self {
        Self {
            n_requests_used: resp.n_requests_used,
            n_requests_cap: resp.n_requests_cap,
            n_requests_surplus: resp.n_requests_surplus,
            fetched_at: std::time::Instant::now(),
        }
    }

    /// Calculate headroom as a fraction (0.0 to 1.0).
    pub fn headroom_pct(&self) -> f64 {
        if self.n_requests_cap == 0 {
            return 0.0;
        }
        (self.n_requests_cap.saturating_sub(self.n_requests_used)) as f64
            / self.n_requests_cap as f64
    }

    /// Check if cache is stale (older than given duration).
    pub fn is_stale(&self, max_age: std::time::Duration) -> bool {
        self.fetched_at.elapsed() > max_age
    }
}

impl InfraComponents {
    /// Create infrastructure components from configs.
    pub fn new(
        margin_config: MarginConfig,
        data_quality_config: DataQualityConfig,
        metrics: MetricsRecorder,
        recovery_config: RecoveryConfig,
        reconciliation_config: ReconciliationConfig,
        rate_limit_config: RejectionRateLimitConfig,
        proactive_rate_config: ProactiveRateLimitConfig,
    ) -> Self {
        Self::with_supervisor_config(
            margin_config,
            data_quality_config,
            metrics,
            recovery_config,
            reconciliation_config,
            rate_limit_config,
            proactive_rate_config,
            SupervisorConfig::default(),
        )
    }

    /// Create infrastructure components with custom supervisor config.
    #[allow(clippy::too_many_arguments)]
    pub fn with_supervisor_config(
        margin_config: MarginConfig,
        data_quality_config: DataQualityConfig,
        metrics: MetricsRecorder,
        recovery_config: RecoveryConfig,
        reconciliation_config: ReconciliationConfig,
        rate_limit_config: RejectionRateLimitConfig,
        proactive_rate_config: ProactiveRateLimitConfig,
        supervisor_config: SupervisorConfig,
    ) -> Self {
        Self::with_orphan_config(
            margin_config,
            data_quality_config,
            metrics,
            recovery_config,
            reconciliation_config,
            rate_limit_config,
            proactive_rate_config,
            supervisor_config,
            OrphanTrackerConfig::default(),
        )
    }

    /// Create infrastructure components with all custom configs.
    #[allow(clippy::too_many_arguments)]
    pub fn with_orphan_config(
        margin_config: MarginConfig,
        data_quality_config: DataQualityConfig,
        metrics: MetricsRecorder,
        recovery_config: RecoveryConfig,
        reconciliation_config: ReconciliationConfig,
        rate_limit_config: RejectionRateLimitConfig,
        proactive_rate_config: ProactiveRateLimitConfig,
        supervisor_config: SupervisorConfig,
        orphan_config: OrphanTrackerConfig,
    ) -> Self {
        Self::with_impulse_config(
            margin_config,
            data_quality_config,
            metrics,
            recovery_config,
            reconciliation_config,
            rate_limit_config,
            proactive_rate_config,
            supervisor_config,
            orphan_config,
            ImpulseControlConfig::default(),
        )
    }

    /// Create infrastructure components with impulse control config.
    #[allow(clippy::too_many_arguments)]
    pub fn with_impulse_config(
        margin_config: MarginConfig,
        data_quality_config: DataQualityConfig,
        metrics: MetricsRecorder,
        recovery_config: RecoveryConfig,
        reconciliation_config: ReconciliationConfig,
        rate_limit_config: RejectionRateLimitConfig,
        proactive_rate_config: ProactiveRateLimitConfig,
        supervisor_config: SupervisorConfig,
        orphan_config: OrphanTrackerConfig,
        impulse_config: ImpulseControlConfig,
    ) -> Self {
        let connection_supervisor = ConnectionSupervisor::with_config(supervisor_config);
        Self {
            margin_sizer: MarginAwareSizer::new(margin_config),
            exchange_limits: ExchangePositionLimits::new(),
            prometheus: PrometheusMetrics::new(),
            // Use the health monitor from the supervisor for consistency
            connection_health: connection_supervisor.health_monitor(),
            connection_supervisor,
            data_quality: DataQualityMonitor::new(data_quality_config),
            metrics,
            last_margin_refresh: std::time::Instant::now(),
            recovery_manager: RecoveryManager::with_config(recovery_config),
            reconciler: PositionReconciler::with_config(reconciliation_config),
            rate_limiter: RejectionRateLimiter::with_config(rate_limit_config),
            proactive_rate_tracker: ProactiveRateLimitTracker::with_config(proactive_rate_config),
            orphan_tracker: OrphanTracker::with_config(orphan_config),
            execution_budget: impulse_config.create_budget(),
            impulse_filter: ImpulseFilter::new(impulse_config.filter.clone()),
            impulse_control_enabled: impulse_config.enabled,
            cached_rate_limit: None,
            // Phase 2/3 monitoring components with defaults
            alerter: Alerter::new(AlertConfig::default(), 1000),
            dashboard: DashboardState::default(),
            fill_tracker: FillTracker::new(1000),
            order_lifecycle: OrderLifecycleTracker::new(1000),
            pending_fill_outcomes: std::collections::VecDeque::with_capacity(64),
            emergency_pull_active: false,
        }
    }
}

/// Stochastic module components: First-principles risk and HJB control.
#[derive(Debug)]
pub struct StochasticComponents {
    /// HJB inventory controller
    pub hjb_controller: HJBInventoryController,
    /// Stochastic module configuration
    pub stochastic_config: StochasticConfig,
    /// Dynamic risk configuration
    pub dynamic_risk_config: DynamicRiskConfig,
    /// Adaptive Bayesian spread calculator
    pub adaptive_spreads: AdaptiveSpreadCalculator,
    /// Calibration-aware fill rate controller
    pub calibration_controller: CalibrationController,
    /// Layer 3: Stochastic controller (POMDP-based sequential decisions)
    pub controller: StochasticController,
    /// HMM-based regime detection for soft regime probabilities
    pub regime_hmm: RegimeHMM,
    /// Model calibration orchestrator
    pub model_calibration: ModelCalibrationOrchestrator,
    /// Adaptive model ensemble
    pub ensemble: AdaptiveEnsemble,

    // === Proactive Position Management (Phase 1-4) ===
    /// Time-based position ramp: limits max position based on session time
    pub position_ramp: super::super::SessionPositionRamp,
    /// Performance-gated capacity: adjusts max position based on P&L
    pub performance_gating: super::super::PerformanceGatedCapacity,

    // === Quote Gate (Directional Edge Gating) ===
    /// Quote gate: decides WHETHER to quote based on directional edge.
    /// Prevents whipsaw losses from random fills when no edge exists.
    pub quote_gate: QuoteGate,

    // === Calibrated Thresholds (IR-Based) ===
    /// Calibrated edge signal tracker.
    /// Tracks whether flow_imbalance predicts price direction using IR > 1.0.
    pub calibrated_edge: CalibratedEdgeSignal,

    /// Position P&L tracker.
    /// Derives position thresholds from actual P&L data.
    pub position_pnl: PositionPnLTracker,

    /// Theoretical edge estimator.
    /// Uses market microstructure priors when IR not calibrated.
    pub theoretical_edge: TheoreticalEdgeEstimator,

    // === Enhanced Flow and MC Simulation (Phase 1-3) ===
    /// Enhanced flow estimator: multi-feature composite flow signal.
    /// Provides varied confidence values for better IR calibration.
    pub enhanced_flow: EnhancedFlowEstimator,

    /// Quick MC simulator: fast Monte Carlo EV estimation.
    /// Used for proactive quoting decisions when IR not calibrated.
    pub mc_simulator: QuickMCSimulator,

    /// Kappa-driven spread controller: dynamic spread adjustment.
    /// Tightens spreads when fill intensity is high.
    pub kappa_spread: KappaSpreadController,

    // === Phase 8: RL and Competitor Modeling ===
    /// Q-Learning agent for adaptive quoting policy (DEPRECATED — replaced by spread_bandit).
    /// Retained for checkpoint backward compatibility.
    pub rl_agent: crate::market_maker::learning::QLearningAgent,

    /// Competitor model for rival MM inference.
    /// Tracks snipe probability and queue competition.
    pub competitor_model: crate::market_maker::learning::CompetitorModel,

    // === Contextual Bandit SpreadOptimizer (replaces RL MDP) ===
    /// Contextual bandit for spread multiplier selection.
    /// 81 contexts × 8 arms, Thompson Sampling with exponential forgetting.
    pub spread_bandit: crate::market_maker::learning::SpreadBandit,

    /// EWMA baseline tracker for counterfactual reward centering.
    /// Subtracts fee drag (~-1.5 bps) so bandit learns meaningful reward differences.
    pub baseline_tracker: crate::market_maker::learning::BaselineTracker,

    // === First-Principles Stochastic Control (DEPRECATED) ===
    /// DEPRECATED (Phase 7): Use `CentralBeliefState` instead.
    ///
    /// This field is no longer updated - price observations now flow to
    /// `central_beliefs` in the MarketMaker struct. Retained for backward
    /// compatibility but will be removed in a future version.
    ///
    /// Previously: Bayesian belief system for first-principles quoting.
    /// Now: Use `market_maker.central_beliefs().snapshot()` for beliefs.
    pub beliefs_builder: StochasticControlBuilder,

    // === Position Continuation Model ===
    /// Position decision engine: HOLD/ADD/REDUCE based on Bayesian continuation.
    /// Transforms inventory_ratio based on P(continuation | fills, regime).
    /// - HOLD: inventory_ratio = 0 (no skew, symmetric quotes)
    /// - ADD: inventory_ratio < 0 (reverse skew, tighter on position-building side)
    /// - REDUCE: inventory_ratio > 0 (normal skew, tighter on position-reducing side)
    pub position_decision: PositionDecisionEngine,

    // === Microstructure Signals (Phase 1: Alpha-Generating Architecture) ===
    /// VPIN estimator: Volume-Synchronized Probability of Informed Trading.
    /// Provides toxicity signal [0, 1] based on volume-classified buckets.
    pub vpin: VpinEstimator,

    /// Liquidity evaporation detector: detects rapid depth drops.
    /// Provides evaporation score [0, 1] for pre-cascade detection.
    pub liquidity_evaporation: LiquidityEvaporationDetector,

    // === Phase 1A Refinements: Toxic Volume Detection ===
    /// Cumulative OFI with decay: distinguishes temporary flickers from sustained shifts.
    /// Raw OFI is noisy - COFI accumulates with decay to filter noise.
    pub cofi: CumulativeOFI,

    /// Trade size distribution tracker: detects anomalous trade sizes.
    /// 3σ jump in median trade size during rising VPIN = accelerated toxicity.
    pub trade_size_dist: TradeSizeDistribution,

    // === V2 Refinements: Statistical Improvements ===
    /// BOCPD predictor for detecting feature→κ relationship breaks.
    /// Uses Bayesian Online Change Point Detection to track when regression
    /// coefficients change, indicating the need to fall back to priors.
    pub bocpd_kappa: BOCPDKappaPredictor,

    /// Signal decay tracker for latency-adjusted calibration.
    /// Tracks signal value decay over time and computes latency-adjusted IR.
    /// Helps identify when signals are stale by the time we act.
    pub signal_decay: SignalDecayTracker,

    /// Cached BOCPD features for update after fill.
    /// Stored during quote generation, used to update BOCPD when fill occurs.
    pub bocpd_kappa_features: Option<[f64; 4]>,

    // === First-Principles Gap 2: Threshold-Dependent Kappa (TAR Model) ===
    /// Threshold kappa: mean-reversion vs momentum regime detection.
    /// Implements TAR model where κ decays when price deviates beyond threshold.
    /// Used to widen spreads during momentum regimes (large moves).
    pub threshold_kappa: ThresholdKappa,

    // === Bayesian Learned Parameters (Magic Number Elimination) ===
    /// Learned parameters: Bayesian-regularized replacements for magic numbers.
    ///
    /// Every parameter has:
    /// 1. A prior based on domain knowledge (the old magic number)
    /// 2. Online learning from observed data
    /// 3. Uncertainty quantification (credible intervals)
    ///
    /// Parameters shrink toward their priors when data is scarce, preventing
    /// overfitting. As fills accumulate, estimates converge toward MLE.
    ///
    /// Categories:
    /// - Tier 1 (P&L Critical): alpha_touch, gamma_base, spread_floor
    /// - Tier 2 (Risk): max_daily_loss, max_drawdown, cascade_threshold
    /// - Tier 3 (Calibration): kappa, hawkes params, decay rates
    /// - Tier 4 (Microstructure): kalman noise, momentum normalizer
    pub learned_params: LearnedParameters,

    // === Cross-Exchange Signal Integration ===
    /// Signal integrator: combines lead-lag, informed flow, regime kappa signals.
    /// Receives Binance prices via channel, computes optimal skew for quote engine.
    pub signal_integrator: SignalIntegrator,

    // === HL-Native Trade Flow Tracking ===
    /// EWMA-based trade flow tracker: directional volume imbalance at multiple horizons.
    /// Replaces hardcoded zeros in FlowFeatureVec for imbalance_30s, avg_buy_size, etc.
    pub trade_flow_tracker: TradeFlowTracker,

    // === Regime State Machine (Phase 2 Redesign) ===
    /// Single source of truth for all regime-dependent parameters.
    /// Updated early in each quote cycle from HMM probs, BOCPD, and kappa estimator.
    /// All downstream consumers read `regime_state.params` for regime-conditioned values.
    pub regime_state: RegimeState,

    // === Hawkes Excitation Prediction (Phase 1: Feature Engineering) ===
    /// Hawkes excitation predictor: detects trade clustering for reactive spread widening.
    /// Uses calibrated branching ratio and intensity percentiles from HawkesOrderFlowEstimator.
    pub hawkes_predictor: crate::market_maker::process_models::hawkes::HawkesExcitationPredictor,
}

impl StochasticComponents {
    /// Create stochastic components from configs.
    pub fn new(
        hjb_config: HJBConfig,
        stochastic_config: StochasticConfig,
        dynamic_risk_config: DynamicRiskConfig,
    ) -> Self {
        Self::with_adaptive_config(
            hjb_config,
            stochastic_config,
            dynamic_risk_config,
            AdaptiveBayesianConfig::default(),
            StochasticControllerConfig::default(),
        )
    }

    /// Create stochastic components with custom adaptive config.
    pub fn with_adaptive_config(
        hjb_config: HJBConfig,
        stochastic_config: StochasticConfig,
        dynamic_risk_config: DynamicRiskConfig,
        adaptive_config: AdaptiveBayesianConfig,
        controller_config: StochasticControllerConfig,
    ) -> Self {
        // Create calibration controller from stochastic config
        let calibration_config = CalibrationControllerConfig {
            enabled: stochastic_config.enable_calibration_fill_rate,
            target_fill_rate_per_hour: stochastic_config.target_fill_rate_per_hour,
            min_gamma_mult: stochastic_config.min_fill_hungry_gamma,
            ..CalibrationControllerConfig::default()
        };

        // Create position ramp from stochastic config
        use super::super::{RampCurve, SessionPositionRamp, PerformanceGatedCapacity};
        let ramp_curve = match stochastic_config.ramp_curve.as_str() {
            "linear" => RampCurve::Linear,
            "log" => RampCurve::Log,
            _ => RampCurve::Sqrt, // Default to sqrt
        };
        let mut position_ramp = SessionPositionRamp::new(
            stochastic_config.ramp_duration_secs,
            stochastic_config.ramp_initial_fraction,
            ramp_curve,
        );
        // Start session immediately on construction
        if stochastic_config.enable_position_ramp {
            position_ramp.start_session();
        }

        // Create performance gating (will be configured with max_position later)
        let performance_gating = if stochastic_config.enable_performance_gating {
            PerformanceGatedCapacity::new(
                0.0, // Will be updated with actual max_position
                0.0, // Will be updated with mid price
                stochastic_config.performance_loss_reduction_mult,
                stochastic_config.performance_min_capacity_fraction,
            )
        } else {
            PerformanceGatedCapacity::disabled(0.0)
        };

        // Create quote gate from stochastic config
        let quote_gate = QuoteGate::new(stochastic_config.quote_gate_config());

        Self {
            hjb_controller: HJBInventoryController::new(hjb_config),
            stochastic_config,
            dynamic_risk_config,
            adaptive_spreads: AdaptiveSpreadCalculator::new(adaptive_config),
            calibration_controller: CalibrationController::new(calibration_config),
            controller: StochasticController::new(controller_config),
            regime_hmm: RegimeHMM::new(),
            model_calibration: ModelCalibrationOrchestrator::default(),
            ensemble: {
                let mut ens = AdaptiveEnsemble::default();
                ens.register_model("GLFT");
                ens.register_model("SpreadBandit");
                ens
            },
            position_ramp,
            performance_gating,
            quote_gate,
            // Calibrated thresholds (IR-based)
            calibrated_edge: CalibratedEdgeSignal::new(CalibratedEdgeConfig::default()),
            position_pnl: PositionPnLTracker::new(PositionPnLConfig::default()),
            // Theoretical edge for fallback when IR not calibrated
            theoretical_edge: TheoreticalEdgeEstimator::new(),
            // Enhanced flow and MC simulation (Phase 1-3)
            enhanced_flow: EnhancedFlowEstimator::new(EnhancedFlowConfig::default()),
            mc_simulator: QuickMCSimulator::new(QuickMCConfig::default()),
            kappa_spread: KappaSpreadController::new(KappaSpreadConfig::default()),
            // Phase 8: RL and Competitor Modeling
            rl_agent: crate::market_maker::learning::QLearningAgent::default(),
            competitor_model: crate::market_maker::learning::CompetitorModel::default(),
            // Contextual Bandit SpreadOptimizer (replaces RL MDP)
            spread_bandit: crate::market_maker::learning::SpreadBandit::default(),
            baseline_tracker: crate::market_maker::learning::BaselineTracker::default(),
            // First-Principles Stochastic Control
            beliefs_builder: StochasticControlBuilder::new(StochasticControlConfig::default()),
            // Position Continuation Model (HOLD/ADD/REDUCE)
            position_decision: PositionDecisionEngine::new(PositionDecisionConfig::default()),
            // Microstructure Signals (Phase 1: Alpha-Generating Architecture)
            vpin: VpinEstimator::new(VpinConfig::default()),
            liquidity_evaporation: LiquidityEvaporationDetector::new(LiquidityEvaporationConfig::default()),
            // Phase 1A Refinements: Toxic Volume Detection
            cofi: CumulativeOFI::new(CumulativeOFIConfig::default()),
            trade_size_dist: TradeSizeDistribution::new(TradeSizeDistributionConfig::default()),
            // V2 Refinements: Statistical Improvements
            bocpd_kappa: BOCPDKappaPredictor::new(BOCPDKappaConfig::default()),
            signal_decay: SignalDecayTracker::new(),
            bocpd_kappa_features: None,
            // First-Principles Gap 2: Threshold-Dependent Kappa (TAR Model)
            threshold_kappa: ThresholdKappa::new(ThresholdKappaConfig::default()),
            // Bayesian Learned Parameters (Magic Number Elimination)
            learned_params: LearnedParameters::default(),
            // Cross-Exchange Signal Integration (Binance → Hyperliquid lead-lag)
            signal_integrator: SignalIntegrator::new(SignalIntegratorConfig::default()),
            // HL-Native Trade Flow Tracking
            trade_flow_tracker: TradeFlowTracker::new(),
            // Regime state machine (Phase 2 Redesign)
            regime_state: RegimeState::new(),
            // Hawkes excitation predictor for spread widening
            hawkes_predictor: crate::market_maker::process_models::hawkes::HawkesExcitationPredictor::default(),
        }
    }

    /// Synchronize volatility regime across all components.
    ///
    /// Call this periodically (e.g., each quote cycle) with the current regime
    /// from `BeliefState.most_likely_regime()` or `RegimeHMM`.
    ///
    /// This ensures `TheoreticalEdgeEstimator` learns regime-specific alpha.
    pub fn sync_regime(&mut self, regime: usize) {
        self.theoretical_edge.set_regime(regime);
    }
    
    /// Handle changepoint detection.
    ///
    /// Call this when `StochasticController.changepoint.should_reset_beliefs()` is true.
    /// Decays alpha posteriors toward prior to "forget" old regime data.
    ///
    /// # Arguments
    /// * `retention` - Fraction of posterior to keep (0.3 recommended)
    /// * `all_regimes` - If true, decay all regimes; if false, only current
    pub fn on_changepoint(&mut self, retention: f64, all_regimes: bool) {
        self.theoretical_edge.decay_alpha(retention, all_regimes);
        tracing::info!(
            retention = %format!("{:.1}%", retention * 100.0),
            all_regimes = all_regimes,
            "Changepoint detected - decayed Bayesian alpha"
        );
    }


    // ==================== Learned Parameters Methods ====================

    /// Update alpha_touch from a fill event.
    ///
    /// Call this after each fill with information about whether the fill
    /// was "informed" (adverse move > 5 bps within 1 second).
    ///
    /// # Arguments
    /// * `is_informed` - True if fill was followed by adverse price move
    pub fn update_alpha_touch(&mut self, is_informed: bool) {
        if is_informed {
            self.learned_params.alpha_touch.observe_beta(1, 0);
        } else {
            self.learned_params.alpha_touch.observe_beta(0, 1);
        }
        self.learned_params.total_fills_observed += 1;
    }

    /// Update kappa from observed fill rate.
    ///
    /// Call this periodically with fill count and observation time.
    ///
    /// # Arguments
    /// * `fills` - Number of fills observed
    /// * `exposure_seconds` - Time period in seconds
    /// * `avg_spread_bps` - Average spread during observation
    pub fn update_kappa_from_fills(&mut self, fills: usize, exposure_seconds: f64, avg_spread_bps: f64) {
        if exposure_seconds > 0.0 && avg_spread_bps > 0.0 {
            let fill_rate = fills as f64 / exposure_seconds;
            let kappa_obs = fill_rate / (avg_spread_bps / 10_000.0);
            if kappa_obs > 100.0 && kappa_obs < 100_000.0 {
                // Observe as Poisson count
                self.learned_params.kappa.observe_gamma_poisson(fills, exposure_seconds * (avg_spread_bps / 10_000.0));
            }
        }
    }

    /// Get the learned alpha_touch estimate (Bayesian posterior mean).
    ///
    /// With few fills, this shrinks toward prior (0.25).
    /// With many fills, this converges to observed rate.
    pub fn learned_alpha_touch(&self) -> f64 {
        self.learned_params.alpha_touch.estimate()
    }

    /// Get the learned kappa estimate.
    pub fn learned_kappa(&self) -> f64 {
        self.learned_params.kappa.estimate()
    }

    /// Get the learned spread floor in bps.
    pub fn learned_spread_floor_bps(&self) -> f64 {
        self.learned_params.spread_floor_bps.estimate()
    }

    /// Check if learned parameters have enough data to be trusted.
    ///
    /// Returns true if Tier 1 parameters (P&L critical) are calibrated.
    pub fn learned_params_calibrated(&self) -> bool {
        self.learned_params.calibration_status().tier1_ready
    }

    /// Get a summary of learned parameter estimates for logging.
    pub fn learned_params_summary(&self) -> Vec<(&str, f64, f64, usize)> {
        self.learned_params.summary()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier1_construction() {
        let tier1 = Tier1Components::new(
            AdverseSelectionConfig::default(),
            QueueConfig::default(),
            LiquidationConfig::default(),
        );
        assert!(!tier1.adverse_selection.is_warmed_up());
    }

    #[test]
    fn test_tier1_with_circuit_breaker() {
        let config = CircuitBreakerConfig::default();
        let tier1 = Tier1Components::with_circuit_breaker(
            AdverseSelectionConfig::default(),
            QueueConfig::default(),
            LiquidationConfig::default(),
            config,
        );
        // Circuit breaker should have no triggered breakers initially
        assert!(tier1.circuit_breaker.triggered_breakers().is_empty());
    }

    #[test]
    fn test_tier2_construction() {
        let tier2 = Tier2Components::new(
            HawkesConfig::default(),
            FundingConfig::default(),
            SpreadConfig::default(),
            PnLConfig::default(),
        );
        assert!((tier2.pnl_tracker.summary(50000.0).total_pnl).abs() < f64::EPSILON);
        assert_eq!(tier2.edge_tracker.edge_count(), 0);
    }

    #[test]
    fn test_safety_components_construction() {
        let safety = SafetyComponents::new(KillSwitchConfig::default(), RiskAggregator::new(), 1.0, 0.1);
        // Kill switch should not be triggered initially
        assert!(!safety.kill_switch.is_triggered());
        // Risk checker should be created with defaults
        assert!(!safety.risk_checker.check_order_size(0.5).is_breach());
    }

    #[test]
    fn test_safety_components_with_risk_limits() {
        use crate::market_maker::risk::RiskCheckResult;
        let limits = RiskLimits::default().with_max_order_size(0.1);
        let safety = SafetyComponents::with_risk_limits(
            KillSwitchConfig::default(),
            RiskAggregator::new(),
            limits,
            DrawdownConfig::default(),
            1.0,  // max_position
            0.15, // gamma
        );
        // Risk checker should enforce the custom limit
        assert_eq!(
            safety.risk_checker.check_order_size(0.05),
            RiskCheckResult::Ok
        );
        assert!(safety.risk_checker.check_order_size(0.2).is_hard_breach());
    }

    #[test]
    fn test_stochastic_components_construction() {
        let stochastic = StochasticComponents::new(
            HJBConfig::default(),
            StochasticConfig::default(),
            DynamicRiskConfig::default(),
        );
        // Regime HMM should start with Normal as most likely
        let belief = stochastic.regime_hmm.regime_probabilities();
        assert!(belief[1] > 0.5); // Normal regime index is 1
                                  // Ensemble should start empty
        let summary = stochastic.ensemble.summary();
        assert_eq!(summary.total_models, 2);
    }
}
