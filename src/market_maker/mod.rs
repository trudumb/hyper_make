//! Modular market maker implementation.
//!
//! This module provides a market making system with pluggable components:
//! - **Strategy**: Determines quote prices and sizes (symmetric, inventory-aware, etc.)
//! - **OrderManager**: Tracks resting orders
//! - **PositionTracker**: Tracks position and deduplicates fills
//! - **Executor**: Handles order placement and cancellation

pub mod adaptive;
mod adverse_selection;
pub mod calibration;
mod config;
pub mod core;
mod estimator;
pub mod events;
pub mod execution;
pub mod fills;
pub mod infra;
pub mod messages;
pub mod multi;
pub mod process_models;
pub mod quoting;
pub mod risk;
pub mod safety;
mod strategy;
pub mod tracking;

pub mod control;
pub mod edge;
pub mod latent;
pub mod learning;
pub mod monitoring;
pub mod simulation;

mod orchestrator;

#[cfg(test)]
mod tests;

pub use adverse_selection::*;
pub use calibration::{
    BrierScoreTracker, ConditionalCalibration, InformationRatioTracker, PredictionLog,
    PredictionRecord, PredictionType,
};
pub use config::*;
pub use edge::{
    ABMetrics, ABTest, ABTestConfig, ABTestManager, ABVariant, EdgeSignalKind, HealthStatus,
    SignalHealth, SignalHealthMonitor, SignalHealthSummary,
};
pub use monitoring::{
    Alert, AlertConfig, AlertHandler, AlertSeverity, AlertType, Alerter,
    DashboardState as MonitoringDashboardState, LoggingAlertHandler, PositionSide,
};
pub use estimator::*;
pub use execution::{
    CancelAnalysis, FillMetrics, FillRecord, FillStatistics, FillTracker, OrderEvent,
    OrderLifecycle, OrderLifecycleTracker, OrderState as ExecutionOrderState,
    Side as ExecutionSide,
};
pub use infra::*;
pub use process_models::*;
pub use quoting::*;
pub use risk::*;
pub use strategy::*;
pub use tracking::*;

use alloy::primitives::Address;
use tracing::{error, info, warn};

use crate::InfoClient;

/// Market maker orchestrator.
///
/// Coordinates strategy, order management, position tracking, and execution.
/// Includes live parameter estimation for GLFT strategy (σ from trades, κ from L2 book).
/// Tier 1 modules provide adverse selection measurement, queue tracking, and cascade detection.
///
/// # Component Organization
///
/// Fields are organized into logical bundles:
/// - **Core**: config, strategy, executor, orders, position, estimator
/// - **Tier1**: adverse selection, queue tracking, liquidation detection
/// - **Tier2**: hawkes, funding, spread, P&L
/// - **Safety**: kill switch, risk aggregator, fill processor
/// - **Infra**: margin, prometheus, connection health, data quality
/// - **Stochastic**: HJB controller, stochastic config, dynamic risk
pub struct MarketMaker<S: QuotingStrategy, E: OrderExecutor> {
    // === Core Fields ===
    /// Configuration
    config: MarketMakerConfig,
    /// Quoting strategy
    strategy: S,
    /// Order executor
    executor: E,
    /// Order state manager
    orders: OrderManager,
    /// WebSocket-based order state manager for improved state tracking
    ws_state: WsOrderStateManager,
    /// Position tracker
    position: PositionTracker,
    /// Info client for subscriptions
    info_client: InfoClient,
    /// User address
    user_address: Address,
    /// Latest mid price
    latest_mid: f64,
    /// Parameter estimator for σ and κ
    estimator: ParameterEstimator,
    /// Last logged warmup progress (to avoid spam)
    last_warmup_log: usize,
    /// Last time warmup blocked quoting was logged (Instant for time-based throttling)
    last_warmup_block_log: Option<std::time::Instant>,

    // === Component Bundles ===
    /// Tier 1: Production resilience (AS, queue, liquidation)
    tier1: core::Tier1Components,
    /// Tier 2: Process models (hawkes, funding, spread, P&L)
    tier2: core::Tier2Components,
    /// Safety: Kill switch, risk aggregation, fill processing
    safety: core::SafetyComponents,
    /// Infrastructure: Margin, metrics, health, data quality
    infra: core::InfraComponents,
    /// Stochastic: HJB control, dynamic risk
    stochastic: core::StochasticComponents,

    // === First-Principles Position Limit ===
    /// Effective max position SIZE (contracts), updated each quote cycle.
    ///
    /// Derived from first principles:
    /// - During warmup: Uses config.max_position as fallback
    /// - After warmup: Uses dynamic_max_position_value / mid_price
    ///
    /// Formula: max_position = min(leverage_limit, volatility_limit) / mid_price
    /// where:
    ///   leverage_limit = account_value × max_leverage
    ///   volatility_limit = (equity × risk_fraction) / (num_sigmas × σ × √T)
    ///
    /// This is THE source of truth for all position limit checks.
    /// NEVER use config.max_position directly for runtime decisions.
    effective_max_position: f64,

    /// Effective target liquidity SIZE (contracts), updated each quote cycle.
    ///
    /// Derived from first principles:
    /// - min_viable = MIN_ORDER_NOTIONAL / mid_price (exchange requirement)
    /// - effective = max(config.target_liquidity, min_viable).min(effective_max_position)
    ///
    /// This ensures orders are viable for ANY asset on ANY DEX, regardless of price.
    /// NEVER use config.target_liquidity directly for runtime quoting.
    effective_target_liquidity: f64,

    // === WebSocket State Sync (Phase 2) ===
    /// Latest OpenOrders snapshot received via WebSocket.
    /// Used by safety_sync to avoid REST API race conditions.
    last_ws_open_orders_snapshot: Option<std::collections::HashSet<u64>>,
    /// Timestamp of the last WS snapshot.
    last_ws_snapshot_time: Option<std::time::Instant>,
    /// Timestamp of last ActiveAssetData WS message (for exchange limits).
    last_active_asset_data_time: Option<std::time::Instant>,
    /// Timestamp of last WebData2 WS message (for user/margin state).
    last_web_data2_time: Option<std::time::Instant>,

    /// Local cache of spot balances (coin -> balance).
    /// Initialized via REST, updated via WebSocket ledger updates.
    spot_balance_cache: std::collections::HashMap<String, f64>,

    /// Tracks whether we're in high risk state to avoid log spam.
    /// Only logs transitions (normal → high risk, high risk → normal).
    last_high_risk_state: bool,

    // === Closed-Loop Learning ===
    /// Learning module for model confidence tracking and ensemble predictions
    learning: learning::LearningModule,

    // === Session Tracking ===
    /// Session start time for controller terminal condition handling
    session_start_time: std::time::Instant,
}

impl<S: QuotingStrategy, E: OrderExecutor> MarketMaker<S, E> {
    /// Create a new market maker.
    ///
    /// # Parameters
    /// - `config`: Market maker configuration (asset, position limits, etc.)
    /// - `strategy`: Quoting strategy (GLFT, Symmetric, etc.)
    /// - `executor`: Order executor (Hyperliquid or mock)
    /// - `info_client`: WebSocket client for market data
    /// - `user_address`: Trader's Ethereum address
    /// - `initial_position`: Starting position (from exchange sync)
    /// - `metrics`: Optional metrics recorder
    /// - `estimator_config`: Configuration for parameter estimation
    /// - `as_config`: Adverse selection estimator configuration
    /// - `queue_config`: Queue position tracker configuration
    /// - `liquidation_config`: Liquidation cascade detector configuration
    /// - `kill_switch_config`: Kill switch configuration for emergency shutdown
    /// - `hawkes_config`: Hawkes order flow estimator configuration
    /// - `funding_config`: Funding rate estimator configuration
    /// - `spread_config`: Spread process estimator configuration
    /// - `pnl_config`: P&L tracker configuration
    /// - `margin_config`: Margin-aware sizer configuration
    /// - `data_quality_config`: Data quality monitor configuration
    /// - `recovery_config`: Recovery manager configuration (Phase 3)
    /// - `reconciliation_config`: Position reconciler configuration (Phase 4)
    /// - `rate_limit_config`: Rejection rate limiter configuration (Phase 5)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: MarketMakerConfig,
        strategy: S,
        executor: E,
        info_client: InfoClient,
        user_address: Address,
        initial_position: f64,
        metrics: MetricsRecorder,
        estimator_config: EstimatorConfig,
        as_config: AdverseSelectionConfig,
        queue_config: QueueConfig,
        liquidation_config: LiquidationConfig,
        kill_switch_config: KillSwitchConfig,
        hawkes_config: HawkesConfig,
        funding_config: FundingConfig,
        spread_config: SpreadConfig,
        pnl_config: PnLConfig,
        margin_config: MarginConfig,
        data_quality_config: infra::DataQualityConfig,
        recovery_config: infra::RecoveryConfig,
        reconciliation_config: infra::ReconciliationConfig,
        rate_limit_config: infra::RejectionRateLimitConfig,
    ) -> Self {
        // Capture config values before moving config
        let stochastic_config = config.stochastic.clone();
        let hjb_config = HJBConfig {
            session_duration_secs: stochastic_config.hjb_session_duration,
            terminal_penalty: stochastic_config.hjb_terminal_penalty,
            gamma_base: config.risk_aversion,
            funding_ewma_half_life: stochastic_config.hjb_funding_half_life,
            ..HJBConfig::default()
        };

        // Build component bundles
        let tier1 = core::Tier1Components::new(as_config, queue_config, liquidation_config);
        let tier2 =
            core::Tier2Components::new(hawkes_config, funding_config, spread_config, pnl_config);
        let safety = core::SafetyComponents::new(
            kill_switch_config.clone(),
            Self::build_risk_aggregator(&kill_switch_config),
        );
        let infra = core::InfraComponents::new(
            margin_config,
            data_quality_config,
            metrics,
            recovery_config,
            reconciliation_config,
            rate_limit_config,
            infra::ProactiveRateLimitConfig::default(),
        );
        let stochastic = core::StochasticComponents::new(
            hjb_config,
            stochastic_config,
            DynamicRiskConfig::default(),
        );

        // Initialize effective_max_position to static fallback (used during warmup)
        let effective_max_position = config.max_position;

        // Initialize effective_target_liquidity to static fallback (used during warmup)
        // Will be updated with first-principles min_viable floor once we have price data
        let effective_target_liquidity = config.target_liquidity;

        Self {
            config,
            strategy,
            executor,
            orders: OrderManager::new(),
            ws_state: WsOrderStateManager::new(),
            position: PositionTracker::new(initial_position),
            info_client,
            user_address,
            latest_mid: -1.0,
            estimator: ParameterEstimator::new(estimator_config),
            last_warmup_log: 0,
            last_warmup_block_log: None,
            tier1,
            tier2,
            safety,
            infra,
            stochastic,
            effective_max_position,
            effective_target_liquidity,
            last_ws_open_orders_snapshot: None,
            last_ws_snapshot_time: None,
            last_active_asset_data_time: None,
            last_web_data2_time: None,
            spot_balance_cache: std::collections::HashMap::new(),
            last_high_risk_state: false,
            learning: learning::LearningModule::default(),
            session_start_time: std::time::Instant::now(),
        }
    }

    /// Set the dynamic risk configuration.
    pub fn with_dynamic_risk_config(mut self, config: DynamicRiskConfig) -> Self {
        self.stochastic.dynamic_risk_config = config;
        self
    }

    /// Configure microprice EMA smoothing from StochasticConfig.
    pub fn with_microprice_ema(mut self, alpha: f64, min_change_bps: f64) -> Self {
        self.estimator.set_microprice_ema(alpha, min_change_bps);
        self
    }

    // =========================================================================
    // Accessor Methods
    // =========================================================================

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.position.position()
    }

    /// Get latest mid price.
    pub fn latest_mid(&self) -> f64 {
        self.latest_mid
    }

    /// Get kill switch reference for monitoring.
    pub fn kill_switch(&self) -> &KillSwitch {
        &self.safety.kill_switch
    }

    /// Get prometheus metrics for external monitoring.
    pub fn prometheus(&self) -> &PrometheusMetrics {
        &self.infra.prometheus
    }

    /// Get asset name for metrics labeling.
    pub fn asset(&self) -> &str {
        &self.config.asset
    }

    /// Get count of tracked orders.
    pub fn orders_count(&self) -> usize {
        self.orders.order_ids().len()
    }

    // =========================================================================
    // Risk Methods
    // =========================================================================

    /// Build a unified RiskState snapshot from current MarketMaker state.
    ///
    /// This provides a single point-in-time view of all risk-relevant data
    /// for use with the RiskAggregator system.
    pub fn build_risk_state(&self) -> risk::RiskState {
        let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);
        let data_age = self.infra.connection_health.time_since_last_data();
        let last_data_time = std::time::Instant::now()
            .checked_sub(data_age)
            .unwrap_or_else(std::time::Instant::now);

        let position = self.position.position();

        // HARMONIZED: Use effective_max_position for risk state calculations
        risk::RiskState::new(
            pnl_summary.total_pnl,
            pnl_summary.total_pnl.max(0.0), // Peak PnL tracking simplified
            position,
            self.effective_max_position, // First-principles limit
            self.latest_mid,
            self.safety.kill_switch.max_position_value(),
            self.infra.margin_sizer.state().account_value,
            self.estimator.sigma_clean(),
            self.tier1.liquidation_detector.cascade_severity(),
            last_data_time,
        )
        .with_pnl_breakdown(pnl_summary.realized_pnl, pnl_summary.unrealized_pnl)
        .with_volatility(
            self.estimator.sigma_clean(),
            self.estimator.sigma_confidence(),
            self.estimator.jump_ratio(),
        )
        .with_cascade(
            self.tier1.liquidation_detector.cascade_severity(),
            self.tier1.liquidation_detector.tail_risk_multiplier(),
            self.tier1.liquidation_detector.should_pull_quotes(),
        )
        .with_adverse_selection(
            self.tier1.adverse_selection.realized_as_bps(),
            self.tier1.adverse_selection.predicted_alpha(),
        )
        .with_pending_exposure(
            self.orders.pending_exposure().0,
            self.orders.pending_exposure().1,
        )
        .with_connection_state(
            self.infra.connection_health.state()
                == crate::market_maker::infra::ConnectionState::Reconnecting,
            self.infra.connection_health.current_attempt() as u32,
            self.infra.connection_health.state()
                == crate::market_maker::infra::ConnectionState::Failed,
        )
    }

    /// Get current session time as fraction [0, 1].
    ///
    /// Returns 0.0 at session start, 1.0 at session end.
    /// Used by the stochastic controller for terminal condition handling.
    pub fn session_time_fraction(&self) -> f64 {
        // Use elapsed time since session start
        let elapsed_secs = self.session_start_time.elapsed().as_secs_f64();

        // Assume 24-hour session (can be configured later)
        const SESSION_DURATION_SECS: f64 = 24.0 * 60.0 * 60.0;

        (elapsed_secs / SESSION_DURATION_SECS).clamp(0.0, 1.0)
    }

    /// Check kill switch conditions and update state.
    /// Called after each message and periodically.
    fn check_kill_switch(&mut self) {
        // Calculate data age from connection health
        let data_age = self.infra.connection_health.time_since_last_data();
        let last_data_time = std::time::Instant::now()
            .checked_sub(data_age)
            .unwrap_or_else(std::time::Instant::now);

        // Build current state for kill switch check using actual P&L
        let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);
        let margin_state = self.infra.margin_sizer.state();
        let state = KillSwitchState {
            daily_pnl: pnl_summary.total_pnl, // Using total as daily for now
            peak_pnl: pnl_summary.total_pnl,  // Simplified (actual peak needs tracking)
            position: self.position.position(),
            mid_price: self.latest_mid,
            last_data_time,
            rate_limit_errors: 0, // TODO: Track rate limit errors
            cascade_severity: self.tier1.liquidation_detector.cascade_severity(),
            // Capital-efficient: margin data for position runaway check
            account_value: margin_state.account_value,
            leverage: self.infra.margin_sizer.max_leverage(),
        };

        // Update position in kill switch (for value calculation)
        self.safety
            .kill_switch
            .update_position(self.position.position(), self.latest_mid);

        // Update cascade severity
        self.safety
            .kill_switch
            .update_cascade_severity(self.tier1.liquidation_detector.cascade_severity());

        // Warn if data is stale
        if self.infra.connection_health.is_data_stale() {
            warn!(
                data_age_secs = %format!("{:.1}", data_age.as_secs_f64()),
                "WebSocket data is stale - connection may be unhealthy"
            );
        }

        // Check all conditions
        if let Some(reason) = self.safety.kill_switch.check(&state) {
            warn!("Kill switch condition detected: {}", reason);
        }

        // Evaluate using RiskAggregator for unified risk assessment
        let risk_state = self.build_risk_state();
        let aggregated = self.safety.risk_aggregator.evaluate(&risk_state);

        if aggregated.should_kill() {
            error!(
                reasons = ?aggregated.kill_reasons,
                "RiskAggregator kill condition triggered"
            );
        } else {
            // Use state-transition logging to reduce spam
            let is_high_risk = aggregated.max_severity >= risk::RiskSeverity::High;
            if is_high_risk && !self.last_high_risk_state {
                // Transition: normal → high risk
                warn!(
                    summary = %aggregated.summary(),
                    "High risk detected by RiskAggregator"
                );
            } else if !is_high_risk && self.last_high_risk_state {
                // Transition: high risk → normal
                info!("Risk returned to normal");
            }
            self.last_high_risk_state = is_high_risk;
        }
    }

    /// Build RiskAggregator with monitors from kill switch configuration.
    fn build_risk_aggregator(config: &KillSwitchConfig) -> risk::RiskAggregator {
        use risk::monitors::*;

        // Default cascade thresholds (pull at 0.8, kill at 0.95)
        const DEFAULT_CASCADE_PULL: f64 = 0.8;
        const DEFAULT_CASCADE_KILL: f64 = 0.95;

        risk::RiskAggregator::new()
            .with_monitor(Box::new(LossMonitor::new(config.max_daily_loss)))
            .with_monitor(Box::new(DrawdownMonitor::new(config.max_drawdown / 100.0)))
            .with_monitor(Box::new(PositionMonitor::new()))
            .with_monitor(Box::new(DataStalenessMonitor::new(
                config.stale_data_threshold,
            )))
            .with_monitor(Box::new(CascadeMonitor::new(
                DEFAULT_CASCADE_PULL,
                DEFAULT_CASCADE_KILL,
            )))
            .with_monitor(Box::new(RateLimitMonitor::new(
                config.max_rate_limit_errors,
            )))
    }

    /// Evaluate risk using the unified RiskAggregator.
    ///
    /// Returns an aggregated risk assessment from all monitors.
    pub fn evaluate_risk(&self) -> risk::AggregatedRisk {
        let state = self.build_risk_state();
        self.safety.risk_aggregator.evaluate(&state)
    }
}
