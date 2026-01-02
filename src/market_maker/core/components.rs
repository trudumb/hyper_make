//! Component bundles for MarketMaker.
//!
//! Groups related modules into logical bundles for cleaner organization.

use crate::market_maker::{
    adverse_selection::{AdverseSelectionConfig, AdverseSelectionEstimator, DepthDecayAS},
    config::MetricsRecorder,
    fills::FillProcessor,
    infra::{
        ConnectionHealthMonitor, ConnectionSupervisor, DataQualityConfig, DataQualityMonitor,
        ExchangePositionLimits, MarginAwareSizer, MarginConfig, PositionReconciler,
        ProactiveRateLimitConfig, ProactiveRateLimitTracker, PrometheusMetrics,
        ReconciliationConfig, RecoveryConfig, RecoveryManager, RejectionRateLimitConfig,
        RejectionRateLimiter, SupervisorConfig,
    },
    process_models::{
        FundingConfig, FundingRateEstimator, HJBConfig, HJBInventoryController, HawkesConfig,
        HawkesOrderFlowEstimator, LiquidationCascadeDetector, LiquidationConfig, SpreadConfig,
        SpreadProcessEstimator,
    },
    risk::{KillSwitch, KillSwitchConfig, RiskAggregator},
    tracking::{PnLConfig, PnLTracker, QueueConfig, QueuePositionTracker},
    DynamicRiskConfig, StochasticConfig,
};

/// Tier 1 components: Production resilience modules.
///
/// These modules are critical for production trading:
/// - Adverse selection measurement
/// - Queue position tracking
/// - Liquidation cascade detection
pub struct Tier1Components {
    /// Adverse selection estimator
    pub adverse_selection: AdverseSelectionEstimator,
    /// Depth-dependent AS model
    pub depth_decay_as: DepthDecayAS,
    /// Queue position tracker
    pub queue_tracker: QueuePositionTracker,
    /// Liquidation cascade detector
    pub liquidation_detector: LiquidationCascadeDetector,
}

impl Tier1Components {
    /// Create Tier 1 components from configs.
    pub fn new(
        as_config: AdverseSelectionConfig,
        queue_config: QueueConfig,
        liquidation_config: LiquidationConfig,
    ) -> Self {
        Self {
            adverse_selection: AdverseSelectionEstimator::new(as_config),
            depth_decay_as: DepthDecayAS::default(),
            queue_tracker: QueuePositionTracker::new(queue_config),
            liquidation_detector: LiquidationCascadeDetector::new(liquidation_config),
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
}

impl SafetyComponents {
    /// Create safety components from config.
    pub fn new(kill_switch_config: KillSwitchConfig, risk_aggregator: RiskAggregator) -> Self {
        Self {
            kill_switch: KillSwitch::new(kill_switch_config),
            risk_aggregator,
            fill_processor: FillProcessor::new(),
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
        }
    }
}

/// Stochastic module components: First-principles risk and HJB control.
#[derive(Debug, Clone)]
pub struct StochasticComponents {
    /// HJB inventory controller
    pub hjb_controller: HJBInventoryController,
    /// Stochastic module configuration
    pub stochastic_config: StochasticConfig,
    /// Dynamic risk configuration
    pub dynamic_risk_config: DynamicRiskConfig,
}

impl StochasticComponents {
    /// Create stochastic components from configs.
    pub fn new(
        hjb_config: HJBConfig,
        stochastic_config: StochasticConfig,
        dynamic_risk_config: DynamicRiskConfig,
    ) -> Self {
        Self {
            hjb_controller: HJBInventoryController::new(hjb_config),
            stochastic_config,
            dynamic_risk_config,
        }
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
    fn test_tier2_construction() {
        let tier2 = Tier2Components::new(
            HawkesConfig::default(),
            FundingConfig::default(),
            SpreadConfig::default(),
            PnLConfig::default(),
        );
        assert!((tier2.pnl_tracker.summary(50000.0).total_pnl).abs() < f64::EPSILON);
    }
}
