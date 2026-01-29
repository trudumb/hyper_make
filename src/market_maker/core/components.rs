//! Component bundles for MarketMaker.
//!
//! Groups related modules into logical bundles for cleaner organization.

use crate::market_maker::{
    adaptive::{AdaptiveBayesianConfig, AdaptiveSpreadCalculator},
    adverse_selection::{AdverseSelectionConfig, AdverseSelectionEstimator, DepthDecayAS},
    config::{ImpulseControlConfig, MetricsRecorder},
    control::{
        CalibratedEdgeConfig, CalibratedEdgeSignal, PositionPnLConfig, PositionPnLTracker,
        QuoteGate, StochasticController, StochasticControllerConfig,
        TheoreticalEdgeEstimator,
    },
    estimator::{
        CalibrationController, CalibrationControllerConfig, RegimeHMM,
        EnhancedFlowConfig, EnhancedFlowEstimator,
    },
    quoting::{KappaSpreadConfig, KappaSpreadController},
    simulation::{QuickMCConfig, QuickMCSimulator},
    execution::{FillTracker, OrderLifecycleTracker},
    fills::FillProcessor,
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
        KillSwitchConfig, RiskAggregator, RiskChecker, RiskLimits,
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
/// - Queue position tracking
/// - Liquidation cascade detection
/// - Circuit breaker monitoring
pub struct Tier1Components {
    /// Adverse selection estimator
    pub adverse_selection: AdverseSelectionEstimator,
    /// Depth-dependent AS model
    pub depth_decay_as: DepthDecayAS,
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
}

impl SafetyComponents {
    /// Create safety components from config.
    pub fn new(kill_switch_config: KillSwitchConfig, risk_aggregator: RiskAggregator) -> Self {
        Self::with_risk_limits(
            kill_switch_config,
            risk_aggregator,
            RiskLimits::default(),
            DrawdownConfig::default(),
        )
    }

    /// Create safety components with custom risk limits and drawdown config.
    pub fn with_risk_limits(
        kill_switch_config: KillSwitchConfig,
        risk_aggregator: RiskAggregator,
        risk_limits: RiskLimits,
        drawdown_config: DrawdownConfig,
    ) -> Self {
        Self {
            kill_switch: KillSwitch::new(kill_switch_config),
            risk_aggregator,
            fill_processor: FillProcessor::new(),
            risk_checker: RiskChecker::new(risk_limits),
            // Initial equity of 10,000 is a placeholder; updated on first margin refresh
            drawdown_tracker: DrawdownTracker::new(drawdown_config, 10_000.0),
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
            ensemble: AdaptiveEnsemble::default(),
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
    }

    #[test]
    fn test_safety_components_construction() {
        let safety = SafetyComponents::new(KillSwitchConfig::default(), RiskAggregator::new());
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
        assert_eq!(summary.total_models, 0);
    }
}
