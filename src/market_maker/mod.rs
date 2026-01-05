//! Modular market maker implementation.
//!
//! This module provides a market making system with pluggable components:
//! - **Strategy**: Determines quote prices and sizes (symmetric, inventory-aware, etc.)
//! - **OrderManager**: Tracks resting orders
//! - **PositionTracker**: Tracks position and deduplicates fills
//! - **Executor**: Handles order placement and cancellation

pub mod adaptive;
mod adverse_selection;
mod config;
pub mod core;
mod estimator;
pub mod events;
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

pub use adverse_selection::*;
pub use config::*;
pub use estimator::*;
pub use infra::*;
pub use process_models::*;
pub use quoting::*;
pub use risk::*;
pub use strategy::*;
pub use tracking::*;

use tracking::ws_order_state::{WsFillEvent, WsOrderUpdateEvent};

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use alloy::primitives::Address;
use tokio::sync::mpsc::unbounded_channel;
use tracing::{debug, error, info, warn};

use chrono::Timelike;

use crate::helpers::truncate_float;
use crate::prelude::Result;
use crate::{InfoClient, Message, Subscription, EPSILON};

/// Minimum order notional value in USD (Hyperliquid requirement)
const MIN_ORDER_NOTIONAL: f64 = 10.0;

/// Result of recovery check - indicates what action was taken.
struct RecoveryAction {
    /// Whether to skip normal quoting this cycle
    skip_normal_quoting: bool,
}

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

    /// Sync open orders from the exchange.
    /// Call this after creating the market maker to recover state.
    ///
    /// This method performs critical startup safety checks:
    /// 1. Syncs position from exchange (authoritative source of truth)
    /// 2. Cancels ALL existing orders to prevent untracked fills
    /// 3. Refreshes margin and exchange limit state
    pub async fn sync_open_orders(&mut self) -> Result<()> {
        // === CRITICAL: Sync position from exchange first ===
        // Exchange position is authoritative - detect any drift from local state
        if let Err(e) = self.sync_position_from_exchange().await {
            warn!("Failed to sync position from exchange: {e}");
        }

        // === CRITICAL: Cancel ALL existing orders before starting ===
        // This prevents untracked fills from orders placed in previous sessions
        if let Err(e) = self.cancel_all_orders_on_startup().await {
            error!("Failed to cancel existing orders on startup: {e}");
            // Continue anyway - orders will be tracked if we find them below
        }

        // === Refresh margin state on startup ===
        if let Err(e) = self.refresh_margin_state().await {
            warn!("Failed to refresh margin state on startup: {e}");
        }

        // === Refresh exchange position limits on startup ===
        if let Err(e) = self.refresh_exchange_limits().await {
            warn!("Failed to refresh exchange limits on startup: {e}");
        } else {
            info!(
                "Exchange limits initialized: bid={:.6}, ask={:.6}",
                self.infra.exchange_limits.effective_bid_limit(),
                self.infra.exchange_limits.effective_ask_limit()
            );
        }

        // Log impulse control status
        info!(
            impulse_control_enabled = %self.infra.impulse_control_enabled,
            budget_tokens = %self.infra.execution_budget.available(),
            improvement_threshold = %self.infra.impulse_filter.config().improvement_threshold,
            queue_lock_threshold = %self.infra.impulse_filter.config().queue_lock_threshold,
            "Impulse control status"
        );

        // Track any remaining orders (should be empty after cancel-all, but be safe)
        // Use DEX-aware open_orders for HIP-3 support
        let open_orders = self
            .info_client
            .open_orders_for_dex(self.user_address, self.config.dex.as_deref())
            .await?;

        for order in open_orders.iter().filter(|o| o.coin == *self.config.asset) {
            let sz: f64 = order.sz.parse().unwrap_or(0.0);
            let px: f64 = order.limit_px.parse().unwrap_or(0.0);
            let side = if order.side == "B" {
                Side::Buy
            } else {
                Side::Sell
            };

            warn!(
                "[MarketMaker] Unexpected order still resting after cancel-all: {} oid={} sz={} px={}",
                if side == Side::Buy { "BUY" } else { "SELL" },
                order.oid,
                sz,
                px
            );

            self.orders
                .add_order(TrackedOrder::new(order.oid, side, px, sz));
        }

        Ok(())
    }

    /// Sync position from exchange - the exchange is the authoritative source.
    async fn sync_position_from_exchange(&mut self) -> Result<()> {
        let user_state = self
            .info_client
            .user_state_for_dex(self.user_address, self.config.dex.as_deref())
            .await?;

        // Find position for our asset
        let exchange_position = user_state
            .asset_positions
            .iter()
            .find(|p| p.position.coin == *self.config.asset)
            .map(|p| p.position.szi.parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);

        let local_position = self.position.position();
        let drift = (exchange_position - local_position).abs();

        if drift > crate::EPSILON {
            warn!(
                "Position drift detected: local={:.6}, exchange={:.6}, drift={:.6}",
                local_position, exchange_position, drift
            );
            // Update local position to match exchange (authoritative)
            self.position.set_position(exchange_position);
            info!("Position synced from exchange: {:.6}", exchange_position);
        } else {
            info!(
                "Position verified: {:.6} (exchange matches local)",
                exchange_position
            );
        }

        // Initialize metrics with current position
        // This ensures metrics show correct position from startup (not 0)
        self.infra
            .prometheus
            .update_position(self.position.position(), self.effective_max_position);

        // Check if existing position exceeds configured limit - will enter reduce-only mode
        // Note: effective_max_position may be recalculated higher once price data arrives
        // (margin-based calculation vs static config.max_position fallback)
        let position_abs = self.position.position().abs();
        if position_abs > self.effective_max_position {
            warn!(
                "Existing position {:.6} exceeds max {:.6} - will enter reduce-only mode after warmup",
                position_abs, self.effective_max_position
            );
            info!(
                "Reduce-only mode will only place orders that reduce position until within limits"
            );
        }

        Ok(())
    }

    /// Cancel ALL orders for our asset on startup.
    /// Uses retry loop to ensure all orders are cancelled.
    async fn cancel_all_orders_on_startup(&mut self) -> Result<()> {
        const MAX_RETRIES: usize = 5;
        const RETRY_DELAY_MS: u64 = 500;

        for attempt in 0..MAX_RETRIES {
            // Use DEX-aware open_orders for HIP-3 support
            let open_orders = self
                .info_client
                .open_orders_for_dex(self.user_address, self.config.dex.as_deref())
                .await?;
            let our_orders: Vec<_> = open_orders
                .iter()
                .filter(|o| o.coin == *self.config.asset)
                .collect();

            if our_orders.is_empty() {
                if attempt > 0 {
                    info!("All orders cancelled after {} attempts", attempt);
                }
                return Ok(());
            }

            info!(
                "Cancelling {} existing orders on startup (attempt {})",
                our_orders.len(),
                attempt + 1
            );

            // RATE LIMIT OPTIMIZATION: Use bulk cancel instead of individual cancels
            let oids: Vec<u64> = our_orders.iter().map(|o| o.oid).collect();
            let results = self
                .executor
                .cancel_bulk_orders(&self.config.asset, oids.clone())
                .await;

            for (oid, result) in oids.iter().zip(results.iter()) {
                match result {
                    CancelResult::Cancelled => {
                        debug!("Startup cancel: oid={} cancelled", oid);
                    }
                    CancelResult::AlreadyCancelled => {
                        debug!("Startup cancel: oid={} already cancelled", oid);
                    }
                    CancelResult::AlreadyFilled => {
                        warn!(
                            "Startup cancel: oid={} was already filled - position may need sync",
                            oid
                        );
                    }
                    CancelResult::Failed => {
                        warn!("Startup cancel: oid={} failed, will retry", oid);
                    }
                }
            }

            // Wait before checking again
            tokio::time::sleep(Duration::from_millis(RETRY_DELAY_MS)).await;
        }

        // After max retries, check what's left
        // Use DEX-aware open_orders for HIP-3 support
        let remaining = self
            .info_client
            .open_orders_for_dex(self.user_address, self.config.dex.as_deref())
            .await?
            .iter()
            .filter(|o| o.coin == *self.config.asset)
            .count();

        if remaining > 0 {
            error!(
                "Failed to cancel all orders after {} attempts, {} remaining",
                MAX_RETRIES, remaining
            );
        }

        Ok(())
    }

    /// Start the market maker event loop.
    pub async fn start(&mut self) -> Result<()> {
        // Channel uses Arc<Message> for zero-copy dispatch from WsManager
        let (sender, mut receiver) = unbounded_channel::<Arc<Message>>();

        // Subscribe to UserFills for fill detection
        self.info_client
            .subscribe(
                Subscription::UserFills {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to AllMids for price updates (DEX-aware for HIP-3)
        self.info_client
            .subscribe(
                Subscription::AllMids {
                    dex: self.config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to Trades for volatility and arrival intensity estimation (DEX-aware)
        self.info_client
            .subscribe(
                Subscription::Trades {
                    coin: self.config.asset.to_string(),
                    dex: self.config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to L2Book for kappa (order book depth decay) estimation (DEX-aware)
        self.info_client
            .subscribe(
                Subscription::L2Book {
                    coin: self.config.asset.to_string(),
                    dex: self.config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to OrderUpdates for order state tracking via WsOrderStateManager
        self.info_client
            .subscribe(
                Subscription::OrderUpdates {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        drop(sender); // Explicitly drop the sender since we're done with subscriptions

        info!(
            "Market maker started for {} with {} strategy",
            self.config.asset,
            self.strategy.name()
        );
        info!("Warming up parameter estimator...");

        // === Start HJB session (stochastic module integration) ===
        self.stochastic.hjb_controller.start_session();
        debug!("HJB inventory controller session started");

        // Safety sync interval (60 seconds) - fallback to catch any state divergence
        let mut sync_interval = tokio::time::interval(Duration::from_secs(60));
        // Skip the immediate first tick
        sync_interval.tick().await;

        // === ROBUST SIGNAL HANDLING ===
        // Problem: tokio::select! only checks signals at the START of each iteration.
        // When message processing is slow (HTTP calls to exchange), Ctrl+C won't be
        // handled until the current message handler completes.
        //
        // Solution: Use a shared atomic flag set by a dedicated signal handler task.
        // The main loop checks this flag frequently (after each message + periodically).
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_flag_clone = shutdown_flag.clone();

        // Spawn dedicated signal handler task
        tokio::spawn(async move {
            #[cfg(unix)]
            {
                use tokio::signal::unix::{signal, SignalKind};
                let mut sigterm =
                    signal(SignalKind::terminate()).expect("Failed to register SIGTERM handler");

                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        info!("Shutdown signal received (SIGINT/Ctrl+C)");
                    }
                    _ = sigterm.recv() => {
                        info!("Shutdown signal received (SIGTERM)");
                    }
                }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
                info!("Shutdown signal received (SIGINT/Ctrl+C)");
            }

            shutdown_flag_clone.store(true, Ordering::SeqCst);
        });

        loop {
            // === Shutdown Check (checked EVERY iteration) ===
            if shutdown_flag.load(Ordering::SeqCst) {
                info!("Shutdown flag detected, initiating graceful shutdown...");
                break;
            }

            // === Kill Switch Check (before processing any message) ===
            if self.safety.kill_switch.is_triggered() {
                let reasons = self.safety.kill_switch.trigger_reasons();
                error!(
                    "KILL SWITCH TRIGGERED: {:?}",
                    reasons.iter().map(|r| r.to_string()).collect::<Vec<_>>()
                );
                break;
            }

            // Process messages - the shutdown check happens at loop start
            // and after each message is processed
            tokio::select! {
                // Message processing
                message = receiver.recv() => {
                    match message {
                        Some(arc_msg) => {
                            // Zero-copy unwrap: Arc::try_unwrap succeeds when we're the only owner
                            // (which is typical since WsManager sends to each subscriber separately).
                            // Falls back to clone only if Arc is still shared.
                            let msg = Arc::try_unwrap(arc_msg)
                                .unwrap_or_else(|arc| (*arc).clone());

                            if let Err(e) = self.handle_message(msg).await {
                                error!("Error handling message: {e}");
                            }

                            // Check kill switch after each message
                            self.check_kill_switch();

                            // Phase 4: Event-driven reconciliation check
                            // The reconciler may have been triggered by order rejection,
                            // unmatched fill, or large position change during message handling
                            if let Some(trigger) = self.infra.reconciler.should_sync() {
                                debug!(
                                    trigger = ?trigger,
                                    "Event-driven reconciliation triggered"
                                );
                                if let Err(e) = self.safety_sync().await {
                                    warn!("Event-driven sync failed: {e}");
                                }
                            }
                        }
                        None => {
                            warn!("Message channel closed, stopping market maker");
                            break;
                        }
                    }
                }

                // Periodic safety sync
                _ = sync_interval.tick() => {
                    if let Err(e) = self.safety_sync().await {
                        warn!("Safety sync failed: {e}");
                    }

                    // === Update Prometheus metrics ===
                    let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);
                    // HARMONIZED: Use effective_max_position for accurate utilization metrics
                    self.infra.prometheus.update_position(
                        self.position.position(),
                        self.effective_max_position, // First-principles limit
                    );
                    self.infra.prometheus.update_pnl(
                        pnl_summary.total_pnl,      // daily_pnl (total for now)
                        pnl_summary.total_pnl,      // peak_pnl (simplified)
                        pnl_summary.realized_pnl,
                        pnl_summary.unrealized_pnl,
                    );
                    self.infra.prometheus.update_market(
                        self.latest_mid,
                        self.tier2.spread_tracker.current_spread_bps(),
                        self.estimator.sigma_clean(),
                        self.estimator.jump_ratio(),
                        self.estimator.kappa(),
                    );
                    self.infra.prometheus.update_risk(
                        self.safety.kill_switch.is_triggered(),
                        self.tier1.liquidation_detector.cascade_severity(),
                        self.tier1.adverse_selection.realized_as_bps(),
                        self.tier1.liquidation_detector.tail_risk_multiplier(),
                    );
                    // V2 Bayesian estimator metrics
                    self.infra.prometheus.update_v2_estimator(
                        self.estimator.hierarchical_kappa_std(),
                        self.estimator.hierarchical_kappa_ci_95().0,
                        self.estimator.hierarchical_kappa_ci_95().1,
                        self.estimator.soft_toxicity_score(),
                        self.estimator.kappa_sigma_correlation(),
                        self.estimator.hierarchical_as_factor(),
                    );
                    // Robust kappa diagnostic metrics
                    self.infra.prometheus.update_robust_kappa(
                        self.estimator.robust_kappa_ess(),
                        self.estimator.kappa_outlier_count(),
                        self.estimator.robust_kappa_nu(),
                        self.estimator.robust_kappa_obs_count(),
                    );
                    let (bid_exp, ask_exp) = self.orders.pending_exposure();
                    self.infra.prometheus.update_pending_exposure(
                        bid_exp,
                        ask_exp,
                        self.position.position(),
                    );

                    // === Update connection health metrics ===
                    let connected = self.infra.connection_health.state() == ConnectionState::Healthy;
                    self.infra.prometheus.set_websocket_connected(connected);
                    self.infra.prometheus.set_last_trade_age_ms(
                        self.infra.connection_health.time_since_last_data().as_millis() as u64
                    );
                    // L2 book uses the same connection health tracker
                    self.infra.prometheus.set_last_book_age_ms(
                        self.infra.connection_health.time_since_last_data().as_millis() as u64
                    );

                    // Update supervisor statistics for Prometheus
                    let supervisor_stats = self.infra.connection_supervisor.stats();
                    self.infra.prometheus.update_supervisor_stats(
                        supervisor_stats.consecutive_stale_count,
                        supervisor_stats.reconnect_signal_count,
                    );

                    // Update calibration fill rate controller metrics
                    self.infra.prometheus.update_calibration(
                        self.stochastic.calibration_controller.gamma_multiplier(),
                        self.stochastic.calibration_controller.calibration_progress(),
                        self.stochastic.calibration_controller.fill_count(),
                        self.stochastic.calibration_controller.is_calibrated(),
                    );

                    // Check if supervisor recommends reconnection
                    if self.infra.connection_supervisor.is_reconnect_recommended() {
                        warn!(
                            time_since_market_data_secs = supervisor_stats.time_since_market_data.as_secs_f64(),
                            connection_state = %supervisor_stats.connection_state,
                            "Connection supervisor recommends reconnection"
                        );
                    }

                    // === Update P&L inventory snapshot for carry calculation ===
                    if self.latest_mid > 0.0 {
                        self.tier2.pnl_tracker.record_inventory_snapshot(self.latest_mid);
                    }

                    // Log Prometheus output (for scraping or debugging)
                    debug!(
                        prometheus_output = %self.infra.prometheus.to_prometheus_text(&self.config.asset, &self.config.collateral.symbol),
                        "Prometheus metrics snapshot"
                    );

                    // Log kill switch status periodically
                    let summary = self.safety.kill_switch.summary();
                    debug!(
                        daily_pnl = %format!("${:.2}", summary.daily_pnl),
                        drawdown = %format!("{:.1}%", summary.drawdown_pct),
                        position_value = %format!("${:.2}", summary.position_value),
                        data_age = %format!("{:.1}s", summary.data_age_secs),
                        cascade_severity = %format!("{:.2}", summary.cascade_severity),
                        "Kill switch status"
                    );
                }
            }
        }

        // Graceful shutdown with timeout to ensure completion even under load
        // We give ourselves 5 seconds to cancel orders before forced exit
        info!("Initiating graceful shutdown (5 second timeout)...");
        match tokio::time::timeout(Duration::from_secs(5), self.shutdown()).await {
            Ok(Ok(())) => {
                info!("Graceful shutdown completed successfully");
            }
            Ok(Err(e)) => {
                error!("Graceful shutdown encountered error: {e}");
            }
            Err(_) => {
                error!(
                    "Graceful shutdown timed out after 5 seconds - orders may remain on exchange"
                );
            }
        }

        info!("Market maker stopped.");
        Ok(())
    }

    /// Handle a message from subscriptions.
    /// Main message dispatcher - routes to focused handlers.
    async fn handle_message(&mut self, message: Message) -> Result<()> {
        match message {
            Message::AllMids(all_mids) => self.handle_all_mids(all_mids).await,
            Message::Trades(trades) => self.handle_trades(trades),
            Message::UserFills(user_fills) => self.handle_user_fills(user_fills).await,
            Message::L2Book(l2_book) => self.handle_l2_book(l2_book),
            Message::OrderUpdates(order_updates) => self.handle_order_updates(order_updates),
            _ => Ok(()),
        }
    }

    /// Handle AllMids message - updates mid price and triggers quote refresh.
    async fn handle_all_mids(&mut self, all_mids: crate::ws::message_types::AllMids) -> Result<()> {
        // HARMONIZED: Use effective_max_position for position utilization calculations
        let ctx = messages::MessageContext::new(
            self.config.asset.clone(),
            self.latest_mid,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            self.estimator.is_warmed_up(),
            Arc::from(self.config.collateral.symbol.as_str()),
        );

        let mut state = messages::AllMidsState {
            estimator: &mut self.estimator,
            connection_health: &mut self.infra.connection_health,
            connection_supervisor: &self.infra.connection_supervisor,
            adverse_selection: &mut self.tier1.adverse_selection,
            depth_decay_as: &mut self.tier1.depth_decay_as,
            liquidation_detector: &mut self.tier1.liquidation_detector,
            hjb_controller: &mut self.stochastic.hjb_controller,
            stochastic_config: &self.stochastic.stochastic_config,
            latest_mid: &mut self.latest_mid,
        };

        let result = messages::process_all_mids(&all_mids, &ctx, &mut state)?;

        if result.is_some() {
            self.update_quotes().await
        } else {
            Ok(())
        }
    }

    /// Handle Trades message - updates volatility and flow estimates.
    fn handle_trades(&mut self, trades: crate::ws::message_types::Trades) -> Result<()> {
        // HARMONIZED: Use effective_max_position for position utilization calculations
        let ctx = messages::MessageContext::new(
            self.config.asset.clone(),
            self.latest_mid,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            self.estimator.is_warmed_up(),
            Arc::from(self.config.collateral.symbol.as_str()),
        );

        let mut state = messages::TradesState {
            estimator: &mut self.estimator,
            hawkes: &mut self.tier2.hawkes,
            data_quality: &mut self.infra.data_quality,
            prometheus: &mut self.infra.prometheus,
            last_warmup_log: &mut self.last_warmup_log,
        };

        let _result = messages::process_trades(&trades, &ctx, &mut state)?;
        Ok(())
    }

    /// Handle UserFills message - processes fills through FillProcessor.
    async fn handle_user_fills(
        &mut self,
        user_fills: crate::ws::message_types::UserFills,
    ) -> Result<()> {
        // Skip snapshot fills - these are historical fills from previous sessions.
        // Position is already loaded from exchange at startup, so processing these
        // would cause "untracked order filled" warnings for orders we didn't place.
        if user_fills.data.is_snapshot.unwrap_or(false) {
            debug!(
                fills = user_fills.data.fills.len(),
                "Skipping UserFills snapshot (historical fills from previous sessions)"
            );
            return Ok(());
        }

        // HARMONIZED: Use effective_max_position for position utilization calculations
        let ctx = messages::MessageContext::new(
            self.config.asset.clone(),
            self.latest_mid,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            self.estimator.is_warmed_up(),
            Arc::from(self.config.collateral.symbol.as_str()),
        );

        // Create fill state for the processor
        // HARMONIZED: Use effective_max_position for position threshold warnings
        let mut fill_state = fills::FillState {
            position: &mut self.position,
            orders: &mut self.orders,
            adverse_selection: &mut self.tier1.adverse_selection,
            depth_decay_as: &mut self.tier1.depth_decay_as,
            queue_tracker: &mut self.tier1.queue_tracker,
            estimator: &mut self.estimator,
            pnl_tracker: &mut self.tier2.pnl_tracker,
            prometheus: &mut self.infra.prometheus,
            metrics: &self.infra.metrics,
            latest_mid: self.latest_mid,
            asset: &self.config.asset,
            max_position: self.effective_max_position, // First-principles limit
            calibrate_depth_as: self.stochastic.stochastic_config.calibrate_depth_as,
        };

        let result = messages::process_user_fills(
            &user_fills,
            &ctx,
            &mut self.safety.fill_processor,
            &mut fill_state,
        )?;

        // Process fills through WsOrderStateManager for additional state tracking
        // This provides secondary deduplication and state consistency
        for fill in &user_fills.data.fills {
            if fill.coin != *self.config.asset {
                continue;
            }

            let fill_event = WsFillEvent {
                oid: fill.oid,
                tid: fill.tid,
                size: fill.sz.parse().unwrap_or(0.0),
                price: fill.px.parse().unwrap_or(0.0),
                is_buy: fill.side == "B" || fill.side.to_lowercase() == "buy",
                coin: fill.coin.clone(),
                cloid: fill.cloid.clone(),
                timestamp: fill.time,
            };

            // Note: ws_state.handle_fill would update position, but position is already
            // updated by the main fill processor. We call it with a no-op tracker.
            // The main benefit is order state tracking and dedup validation.
            if let Some(ws_order) = self.ws_state.get_order(fill.oid) {
                // Just record the fill for state tracking (already processed above)
                if !ws_order.fill_tids.contains(&fill.tid) {
                    if let Some(order) = self.ws_state.get_order_mut(fill.oid) {
                        order.record_fill_with_price(
                            fill_event.tid,
                            fill_event.size,
                            fill_event.price,
                        );
                    }
                }
            }
        }

        // Phase 4: Trigger reconciliation for unmatched fills
        if result.unmatched_fills > 0 {
            for _ in 0..result.unmatched_fills {
                self.infra.reconciler.on_unmatched_fill(0, 0.0); // OID/size unknown at this point
            }
        }

        // Post-processing: cleanup completed orders
        messages::cleanup_orders(&mut self.orders, &mut self.tier1.queue_tracker);

        // Record fill observations to the strategy for Bayesian learning
        // These observations update the fill probability model
        for obs in &result.fill_observations {
            self.strategy
                .record_fill_observation(obs.depth_bps, obs.filled);
        }
        if !result.fill_observations.is_empty() {
            debug!(
                observations = result.fill_observations.len(),
                "Recorded fill observations for Bayesian learning"
            );
        }

        // Update adaptive Bayesian spread calculator with fill data
        // This updates: learned floor (AS), blended kappa (fill rate), shrinkage gamma (PnL)
        // ALWAYS feed fills to adaptive system for learning, even if not using adaptive quotes.
        // This ensures the system can learn from actual market behavior and be ready when enabled.
        {
            for fill in &user_fills.data.fills {
                if fill.coin != *self.config.asset {
                    continue;
                }

                let fill_price: f64 = fill.px.parse().unwrap_or(0.0);
                let is_buy = fill.side == "B" || fill.side.to_lowercase() == "buy";

                // Compute realized adverse selection: (mid_after - fill_price) × direction
                // direction = +1 for buy (we bought, if mid moved up we gained),
                // direction = -1 for sell (we sold, if mid moved down we gained)
                // Positive AS means we lost (price moved against us after fill)
                let direction = if is_buy { 1.0 } else { -1.0 };
                let as_realized = (self.latest_mid - fill_price) * direction / fill_price;

                // Compute depth of this fill (distance from mid when placed)
                // For now use adverse selection as proxy until we track order placement mid
                let depth_from_mid = (fill_price - self.latest_mid).abs() / self.latest_mid;

                // PnL for this fill: simplified as -AS (negative AS = profit)
                let fill_pnl = -as_realized;

                // Update all adaptive components via simplified fill handler
                self.stochastic.adaptive_spreads.on_fill_simple(
                    as_realized,
                    depth_from_mid,
                    fill_pnl,
                    self.estimator.kappa(),
                );

                // Record fill for calibration controller
                // This updates fill rate tracking for fill-hungry mode
                self.stochastic.calibration_controller.record_fill();
            }
        }

        // Record fill volume for rate limit budget calculation
        if result.total_volume_usd > 0.0 {
            self.infra
                .proactive_rate_tracker
                .record_fill_volume(result.total_volume_usd);

            // Earn execution budget tokens from fills (impulse control)
            if self.infra.impulse_control_enabled {
                self.infra.execution_budget.on_fill(result.total_volume_usd);
                debug!(
                    volume_usd = %format!("{:.2}", result.total_volume_usd),
                    budget_available = %format!("{:.1}", self.infra.execution_budget.available()),
                    "Earned execution budget tokens from fill"
                );
            }

            debug!(
                volume_usd = %format!("{:.2}", result.total_volume_usd),
                "Recorded fill volume for rate limit budget"
            );
        }

        // Margin refresh on fills
        const MARGIN_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);
        if self.infra.last_margin_refresh.elapsed() > MARGIN_REFRESH_INTERVAL {
            if let Err(e) = self.refresh_margin_state().await {
                warn!(error = %e, "Failed to refresh margin after fill");
            }
            self.infra.last_margin_refresh = std::time::Instant::now();
        }

        if result.should_update_quotes {
            self.update_quotes().await
        } else {
            Ok(())
        }
    }

    /// Handle L2Book message - updates order book metrics.
    fn handle_l2_book(&mut self, l2_book: crate::ws::message_types::L2Book) -> Result<()> {
        // HARMONIZED: Use effective_max_position for position utilization calculations
        let ctx = messages::MessageContext::new(
            self.config.asset.clone(),
            self.latest_mid,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            self.estimator.is_warmed_up(),
            Arc::from(self.config.collateral.symbol.as_str()),
        );

        let mut state = messages::L2BookState {
            estimator: &mut self.estimator,
            queue_tracker: &mut self.tier1.queue_tracker,
            spread_tracker: &mut self.tier2.spread_tracker,
            data_quality: &mut self.infra.data_quality,
            prometheus: &mut self.infra.prometheus,
        };

        let result = messages::process_l2_book(&l2_book, &ctx, &mut state)?;

        // Feed L2 data to adaptive spread calculator's blended kappa estimator
        // This enables book-based kappa estimation to work alongside own-fill kappa
        if result.is_valid && self.latest_mid > 0.0 {
            if let Some((bids, asks)) = Self::parse_l2_for_adaptive(&l2_book.data.levels) {
                self.stochastic
                    .adaptive_spreads
                    .on_l2_update(&bids, &asks, self.latest_mid);
            }
        }

        Ok(())
    }

    /// Parse L2 book levels into (bids, asks) tuples for adaptive spread calculator.
    /// Returns None if the levels are invalid or insufficient.
    #[allow(clippy::type_complexity)]
    fn parse_l2_for_adaptive(
        levels: &[Vec<crate::types::OrderBookLevel>],
    ) -> Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        if levels.len() < 2 {
            return None;
        }

        let bids: Vec<(f64, f64)> = levels[0]
            .iter()
            .filter_map(|level| {
                let px: f64 = level.px.parse().ok()?;
                let sz: f64 = level.sz.parse().ok()?;
                Some((px, sz))
            })
            .collect();

        let asks: Vec<(f64, f64)> = levels[1]
            .iter()
            .filter_map(|level| {
                let px: f64 = level.px.parse().ok()?;
                let sz: f64 = level.sz.parse().ok()?;
                Some((px, sz))
            })
            .collect();

        Some((bids, asks))
    }

    /// Handle OrderUpdates message - processes order state changes via WsOrderStateManager.
    fn handle_order_updates(
        &mut self,
        order_updates: crate::ws::message_types::OrderUpdates,
    ) -> Result<()> {
        for update in &order_updates.data {
            // Filter to our asset
            if update.order.coin != *self.config.asset {
                continue;
            }

            // Convert to WsOrderUpdateEvent format
            let event = WsOrderUpdateEvent {
                oid: update.order.oid,
                cloid: update.order.cloid.clone(),
                status: update.status.clone(),
                size: update.order.sz.parse().unwrap_or(0.0),
                orig_size: update.order.orig_sz.parse().unwrap_or(0.0),
                price: update.order.limit_px.parse().unwrap_or(0.0),
                coin: update.order.coin.clone(),
                is_buy: update.order.side == "B" || update.order.side.to_lowercase() == "buy",
                status_timestamp: update.status_timestamp,
            };

            // Process through WsOrderStateManager
            self.ws_state.handle_order_update(&event);

            // CRITICAL: Also sync state to OrderManager to keep both tracking systems in sync.
            // The safety_sync uses OrderManager for state comparison, so it must know about
            // filled/cancelled orders to avoid false stale/orphan detection.
            // Note: Using set_state() is appropriate here since we're receiving status from WS,
            // not initiating state transitions internally.
            #[allow(deprecated)]
            match event.status.as_str() {
                "filled" => {
                    if self.orders.set_state(event.oid, OrderState::Filled) {
                        info!(
                            oid = event.oid,
                            status = %event.status,
                            "Order state update: filled (synced to OrderManager)"
                        );
                    } else {
                        // Order not in OrderManager - could be from previous session
                        debug!(
                            oid = event.oid,
                            status = %event.status,
                            "Order filled but not tracked in OrderManager"
                        );
                    }
                }
                "canceled" => {
                    if self.orders.set_state(event.oid, OrderState::Cancelled) {
                        debug!(
                            oid = event.oid,
                            status = %event.status,
                            "Order state update: canceled (synced to OrderManager)"
                        );
                    } else {
                        debug!(
                            oid = event.oid,
                            status = %event.status,
                            "Order canceled but not tracked in OrderManager"
                        );
                    }
                }
                "open" => {
                    // Order is resting - no state change needed unless partially filled
                    // Size changes are handled by ws_state
                    debug!(
                        oid = event.oid,
                        status = %event.status,
                        "Order state update: open"
                    );
                }
                _ => {
                    debug!(
                        oid = event.oid,
                        status = %event.status,
                        "Order state update (unknown status)"
                    );
                }
            }
        }

        // Periodic cleanup of terminal orders in ws_state
        let removed = self.ws_state.cleanup();
        if !removed.is_empty() {
            debug!(
                count = removed.len(),
                "Cleaned up terminal orders from ws_state"
            );
        }

        Ok(())
    }

    /// Update quotes based on current market state.
    #[tracing::instrument(name = "quote_cycle", skip_all, fields(asset = %self.config.asset))]
    async fn update_quotes(&mut self) -> Result<()> {
        // Don't place orders until estimator is warmed up
        if !self.estimator.is_warmed_up() {
            // Log warmup status every 10 seconds to help diagnose why orders aren't placed
            let should_log = match self.last_warmup_block_log {
                None => true,
                Some(last) => last.elapsed() >= std::time::Duration::from_secs(10),
            };
            if should_log {
                let (vol_ticks, min_vol, trade_obs, min_trades) = self.estimator.warmup_progress();
                warn!(
                    volume_ticks = vol_ticks,
                    volume_ticks_required = min_vol,
                    trade_observations = trade_obs,
                    trade_observations_required = min_trades,
                    "Warmup incomplete - no orders placed (waiting for market data)"
                );
                self.last_warmup_block_log = Some(std::time::Instant::now());
            }
            return Ok(());
        }

        // Update calibration controller with current calibration status
        // This uses AS fills measured and kappa confidence to track calibration progress
        // and adjust fill-hungry gamma multiplier accordingly
        let as_fills_measured = self.tier1.adverse_selection.fills_measured() as u64;
        let kappa_confidence = self.estimator.kappa_confidence();
        self.stochastic
            .calibration_controller
            .update_calibration_status(as_fills_measured, kappa_confidence);

        // HIP-3: OI cap pre-flight check (fast path for unlimited)
        // This is on the hot path, so we use pre-computed values from runtime config
        let current_position_notional = self.position.position().abs() * self.latest_mid;
        let remaining_oi_capacity = self
            .config
            .runtime
            .remaining_oi_capacity(current_position_notional);

        if remaining_oi_capacity < MIN_ORDER_NOTIONAL {
            warn!(
                oi_cap_usd = %self.config.runtime.oi_cap_usd,
                current_notional = %format!("{:.2}", current_position_notional),
                "HIP-3 OI cap reached, skipping quotes"
            );
            return Ok(());
        }

        // Phase 6: Rate limit throttling - respect minimum requote interval
        if !self.infra.proactive_rate_tracker.can_requote() {
            debug!("Skipping requote: minimum interval not elapsed");
            return Ok(());
        }

        // Check for rate limit warnings
        if self.infra.proactive_rate_tracker.ip_rate_warning() {
            warn!("IP rate limit warning: approaching 80% of budget");
        }
        if self.infra.proactive_rate_tracker.address_budget_low() {
            warn!("Address rate limit warning: budget below 1000 requests");
        }

        // Mark that we're doing a requote
        self.infra.proactive_rate_tracker.mark_requote();

        // Phase 3: Check recovery state and handle IOC recovery if needed
        if let Some(action) = self.check_and_handle_recovery().await? {
            if action.skip_normal_quoting {
                return Ok(());
            }
        }

        let quote_config = QuoteConfig {
            mid_price: self.latest_mid,
            decimals: self.config.decimals,
            sz_decimals: self.config.sz_decimals,
            min_notional: MIN_ORDER_NOTIONAL,
        };

        // Build market params from econometric estimates via ParameterAggregator
        let exchange_limits = &self.infra.exchange_limits;
        // Get pending exposure from resting orders (prevents position breach from multiple fills)
        let (pending_bid_exposure, pending_ask_exposure) = self.orders.pending_exposure();

        // Get dynamic position VALUE limit from kill switch (first-principles derived)
        let dynamic_max_position_value = self.safety.kill_switch.max_position_value();
        // Margin state is valid if we've refreshed margin at least once
        let margin_state = self.infra.margin_sizer.state();
        let dynamic_limit_valid = margin_state.account_value > 0.0;

        // CRITICAL: Pre-compute effective_max_position and update exchange limits BEFORE building sources
        // This ensures sources.exchange_effective_bid/ask_limit use the margin-based capacity
        let margin_quoting_capacity = if margin_state.available_margin > 0.0
            && self.infra.margin_sizer.summary().max_leverage > 0.0
            && self.latest_mid > 0.0
        {
            (margin_state.available_margin * self.infra.margin_sizer.summary().max_leverage
                / self.latest_mid)
                .max(0.0)
        } else {
            0.0
        };

        // Compute effective_max_position using same priority as MarketParams::effective_max_position
        let dynamic_max_position = if dynamic_limit_valid && self.latest_mid > 0.0 {
            dynamic_max_position_value / self.latest_mid
        } else {
            0.0
        };

        let pre_effective_max_position = {
            const EPSILON: f64 = 1e-9;
            if margin_quoting_capacity > EPSILON {
                if dynamic_limit_valid && dynamic_max_position > EPSILON {
                    margin_quoting_capacity.min(dynamic_max_position)
                } else {
                    margin_quoting_capacity
                }
            } else if dynamic_limit_valid && dynamic_max_position > EPSILON {
                dynamic_max_position
            } else {
                self.config.max_position // Fallback to config during warmup
            }
        };

        // Update exchange limits with margin-based capacity BEFORE building sources
        self.infra
            .exchange_limits
            .update_local_max(pre_effective_max_position);

        // Pre-compute drift-adjusted skew from HJB controller + momentum signals
        let momentum_bps = self.estimator.momentum_bps();
        let p_continuation = self.estimator.momentum_continuation_probability();
        let position = self.position.position();

        // Update momentum EWMA signals for smoothing (reduces whipsawing)
        self.stochastic.hjb_controller.update_momentum_signals(
            momentum_bps,
            p_continuation,
            position,
            self.config.max_position,
        );

        let drift_adjusted_skew = self.stochastic.hjb_controller.optimal_skew_with_drift(
            position,
            self.config.max_position,
            momentum_bps,
            p_continuation,
        );

        // Log momentum diagnostics for drift-adjusted skew debugging
        // Only log when: (a) position opposes momentum, or (b) momentum exceeds threshold
        // Note: Short-term momentum may differ from medium-term trend (bounces within trends)
        if drift_adjusted_skew.is_opposed || momentum_bps.abs() > 10.0 {
            let ewma_warmed = self.stochastic.hjb_controller.is_drift_warmed_up();
            let smoothed_drift = self.stochastic.hjb_controller.smoothed_drift();
            debug!(
                momentum_bps = %format!("{:.2}", momentum_bps),
                p_continuation = %format!("{:.3}", p_continuation),
                position = %format!("{:.2}", position),
                is_opposed = drift_adjusted_skew.is_opposed,
                drift_urgency_bps = %format!("{:.2}", drift_adjusted_skew.drift_urgency * 10000.0),
                variance_mult = %format!("{:.3}", drift_adjusted_skew.variance_multiplier),
                urgency_score = %format!("{:.1}", drift_adjusted_skew.urgency_score),
                ewma_warmed = ewma_warmed,
                smoothed_drift = %format!("{:.6}", smoothed_drift),
                "Momentum-position alignment"
            );
        }

        let sources = ParameterSources {
            estimator: &self.estimator,
            adverse_selection: &self.tier1.adverse_selection,
            depth_decay_as: &self.tier1.depth_decay_as,
            liquidation_detector: &self.tier1.liquidation_detector,
            hawkes: &self.tier2.hawkes,
            funding: &self.tier2.funding,
            spread_tracker: &self.tier2.spread_tracker,
            hjb_controller: &self.stochastic.hjb_controller,
            margin_sizer: &self.infra.margin_sizer,
            stochastic_config: &self.stochastic.stochastic_config,
            drift_adjusted_skew,
            adaptive_spreads: &self.stochastic.adaptive_spreads,
            position: self.position.position(),
            max_position: self.config.max_position,
            latest_mid: self.latest_mid,
            risk_aversion: self.config.risk_aversion,
            // Exchange position limits
            exchange_limits_valid: exchange_limits.is_initialized(),
            exchange_effective_bid_limit: exchange_limits.effective_bid_limit(),
            exchange_effective_ask_limit: exchange_limits.effective_ask_limit(),
            exchange_limits_age_ms: exchange_limits.age_ms(),
            // Pending exposure from resting orders
            pending_bid_exposure,
            pending_ask_exposure,
            // Dynamic position limits (first principles)
            dynamic_max_position_value,
            dynamic_limit_valid,
            // Stochastic constraints (first principles)
            tick_size_bps: 10.0, // TODO: Get from asset metadata
            near_touch_depth_usd: self.estimator.near_touch_depth_usd(),
            // Calibration fill rate controller
            calibration_gamma_mult: self.stochastic.calibration_controller.gamma_multiplier(),
            calibration_progress: self
                .stochastic
                .calibration_controller
                .calibration_progress(),
            calibration_complete: self.stochastic.calibration_controller.is_calibrated(),
            // Dynamic bounds (model-driven, replaces hardcoded CLI values)
            // true when no CLI override is active (kappa_floor/max_spread_ceiling_bps = None)
            use_dynamic_kappa_floor: !self.estimator.has_static_kappa_floor(),
            use_dynamic_spread_ceiling: !self.estimator.has_static_max_spread_ceiling(),
        };
        let mut market_params = ParameterAggregator::build(&sources);

        // Compute stochastic constraints (latency floor, tight quoting conditions)
        let current_hour_utc = chrono::Utc::now().hour() as u8;
        market_params.compute_stochastic_constraints(
            &self.stochastic.stochastic_config,
            self.position.position(),
            self.config.max_position,
            current_hour_utc,
        );

        // CRITICAL: Update cached effective max_position from first principles
        // This is THE source of truth for all position limit checks
        let new_effective = market_params.effective_max_position(self.config.max_position);
        if (new_effective - self.effective_max_position).abs() > 0.001 {
            debug!(
                old = %format!("{:.6}", self.effective_max_position),
                new = %format!("{:.6}", new_effective),
                dynamic_valid = market_params.dynamic_limit_valid,
                "Effective max position updated from first principles"
            );
        }
        self.effective_max_position = new_effective;

        // Note: exchange limits were already updated BEFORE building sources (line ~1210)
        // using pre_effective_max_position which should equal new_effective

        // CRITICAL: Update cached effective target_liquidity from first principles
        // min_viable = MIN_ORDER_NOTIONAL / price ensures orders pass exchange minimum
        // This is THE source of truth for all sizing calculations
        //
        // Account for size truncation: sz_decimals truncation can lose up to 10^(-sz_decimals)
        // Example: sz_decimals=2 → truncation can lose 0.01 → add buffer of 1.5× tick
        // Formula: min_viable = min_notional / price + 1.5 × 10^(-sz_decimals)
        let truncation_buffer = 1.5 * (10.0_f64).powi(-(self.config.sz_decimals as i32));
        let min_viable_liquidity =
            (MIN_ORDER_NOTIONAL / market_params.microprice) + truncation_buffer;
        let new_effective_liquidity = self
            .config
            .target_liquidity
            .max(min_viable_liquidity) // Ensure orders pass min_notional after truncation
            .min(self.effective_max_position); // Can't exceed position limit

        if (new_effective_liquidity - self.effective_target_liquidity).abs() > 0.001 {
            info!(
                old = %format!("{:.6}", self.effective_target_liquidity),
                new = %format!("{:.6}", new_effective_liquidity),
                config_target = %format!("{:.6}", self.config.target_liquidity),
                min_viable = %format!("{:.6}", min_viable_liquidity),
                truncation_buffer = %format!("{:.6}", truncation_buffer),
                max_position = %format!("{:.6}", self.effective_max_position),
                "Effective target liquidity updated (first-principles with truncation buffer)"
            );
        }
        self.effective_target_liquidity = new_effective_liquidity;

        // Log adaptive system status if enabled
        if self.stochastic.stochastic_config.use_adaptive_spreads {
            let adaptive = &self.stochastic.adaptive_spreads;
            debug!(
                can_estimate = market_params.adaptive_can_estimate,
                fully_warmed_up = market_params.adaptive_warmed_up,
                warmup_progress = %format!("{:.0}%", market_params.adaptive_warmup_progress * 100.0),
                uncertainty_factor = %format!("{:.3}", market_params.adaptive_uncertainty_factor),
                adaptive_floor_bps = %format!("{:.2}", market_params.adaptive_spread_floor * 10000.0),
                adaptive_kappa = %format!("{:.0}", market_params.adaptive_kappa),
                adaptive_gamma = %format!("{:.3}", market_params.adaptive_gamma),
                adaptive_ceiling_bps = %format!("{:.2}", market_params.adaptive_spread_ceiling * 10000.0),
                fill_rate = %format!("{:.4}", adaptive.fill_rate_controller().observed_fill_rate()),
                "Adaptive Bayesian spreads (using immediately via priors)"
            );

            // Call on_no_fill to nudge toward tighter spreads when quoting without fills
            // This is a soft decay that reduces spread over time when not getting filled
            self.stochastic.adaptive_spreads.on_no_fill_simple();
        }

        debug!(
            mid = self.latest_mid,
            microprice = %format!("{:.4}", market_params.microprice),
            position = self.position.position(),
            static_max_pos = self.config.max_position,
            dynamic_max_pos = %format!("{:.6}", market_params.dynamic_max_position),
            effective_max_pos = %format!("{:.6}", self.effective_max_position),
            dynamic_valid = market_params.dynamic_limit_valid,
            config_target_liq = self.config.target_liquidity,
            effective_target_liq = %format!("{:.6}", self.effective_target_liquidity),
            sigma_clean = %format!("{:.6}", market_params.sigma),
            sigma_effective = %format!("{:.6}", market_params.sigma_effective),
            kappa = %format!("{:.2}", market_params.kappa),
            jump_ratio = %format!("{:.2}", market_params.jump_ratio),
            is_toxic = market_params.is_toxic_regime,
            beta_book = %format!("{:.6}", market_params.beta_book),
            beta_flow = %format!("{:.6}", market_params.beta_flow),
            book_imbalance = %format!("{:.2}", market_params.book_imbalance),
            liq_gamma_mult = %format!("{:.2}", market_params.liquidity_gamma_mult),
            "Quote inputs with microprice"
        );

        // Update fill model params for Bayesian fill probability
        // Uses Kelly time horizon (τ) which is computed in ParameterAggregator
        self.strategy
            .update_fill_model_params(market_params.sigma, market_params.kelly_time_horizon);

        // Try multi-level ladder quoting first
        // HARMONIZED: Use effective_target_liquidity (first-principles derived)
        let ladder = self.strategy.calculate_ladder(
            &quote_config,
            self.position.position(),
            self.effective_max_position,     // First-principles limit
            self.effective_target_liquidity, // First-principles viable size
            &market_params,
        );

        if !ladder.bids.is_empty() || !ladder.asks.is_empty() {
            // Multi-level ladder mode
            let mut bid_quotes: Vec<Quote> = ladder
                .bids
                .iter()
                .map(|l| Quote::new(l.price, l.size))
                .collect();
            let mut ask_quotes: Vec<Quote> = ladder
                .asks
                .iter()
                .map(|l| Quote::new(l.price, l.size))
                .collect();

            // Reduce-only mode: when over max position, position value, OR margin utilization
            // Phase 3: Use exchange-aware reduce-only that checks exchange limits and signals escalation
            // CAPITAL-EFFICIENT: Use margin utilization as primary trigger (80% threshold)
            let margin_state = self.infra.margin_sizer.state();
            let reduce_only_config = quoting::ReduceOnlyConfig {
                position: self.position.position(),
                max_position: self.effective_max_position,
                mid_price: self.latest_mid,
                max_position_value: self.safety.kill_switch.max_position_value(),
                asset: self.config.asset.to_string(),
                margin_used: margin_state.margin_used,
                account_value: margin_state.account_value,
                // Dynamic reduce-only based on liquidation proximity
                liquidation_price: margin_state.liquidation_price,
                liquidation_buffer_ratio: margin_state.liquidation_buffer_ratio(),
                liquidation_trigger_threshold: quoting::DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD,
            };
            let reduce_only_result = quoting::QuoteFilter::apply_reduce_only_with_exchange_limits(
                &mut bid_quotes,
                &mut ask_quotes,
                &reduce_only_config,
                &self.infra.exchange_limits,
            );

            // If escalation is needed, the recovery manager should be notified
            // (This happens automatically via rate limiter when orders get rejected)
            if reduce_only_result.needs_escalation {
                debug!("Reduce-only mode activated with potential escalation");
            }

            debug!(
                bid_levels = bid_quotes.len(),
                ask_levels = ask_quotes.len(),
                best_bid = ?bid_quotes.first().map(|q| (q.price, q.size)),
                best_ask = ?ask_quotes.first().map(|q| (q.price, q.size)),
                "Calculated ladder quotes"
            );

            // DIAGNOSTIC: Warn when ladder is completely empty after processing
            // This helps diagnose min_notional and capacity issues at INFO level
            if bid_quotes.is_empty() && ask_quotes.is_empty() {
                let pos = self.position.position();
                let max_pos = self.effective_max_position;
                let mid = self.latest_mid;
                warn!(
                    position = %format!("{:.6}", pos),
                    max_position = %format!("{:.6}", max_pos),
                    mid_price = %format!("{:.2}", mid),
                    bid_capacity_notional = %format!("{:.2}", (max_pos - pos).max(0.0) * mid),
                    ask_capacity_notional = %format!("{:.2}", (max_pos + pos).max(0.0) * mid),
                    min_notional = %format!("{:.2}", MIN_ORDER_NOTIONAL),
                    reduce_only_was_filtered = reduce_only_result.was_filtered,
                    "No orders to place: ladder empty after filtering (check min_notional vs capacity)"
                );
                return Ok(());
            }

            // Reconcile ladder quotes
            if self.config.smart_reconcile {
                // Smart reconciliation with ORDER MODIFY for queue preservation
                self.reconcile_ladder_smart(bid_quotes, ask_quotes).await?;
            } else {
                // Legacy all-or-nothing reconciliation
                self.reconcile_ladder_side(Side::Buy, bid_quotes).await?;
                self.reconcile_ladder_side(Side::Sell, ask_quotes).await?;
            }
        } else {
            // Fallback to single-quote mode for non-ladder strategies
            // HARMONIZED: Use effective values (first-principles derived)
            let (mut bid, mut ask) = self.strategy.calculate_quotes(
                &quote_config,
                self.position.position(),
                self.effective_max_position,     // First-principles limit
                self.effective_target_liquidity, // First-principles viable size
                &market_params,
            );

            // Reduce-only mode: when over max position, position value, OR margin utilization
            // CAPITAL-EFFICIENT: Use margin utilization as primary trigger (80% threshold)
            let margin_state = self.infra.margin_sizer.state();
            let reduce_only_config = quoting::ReduceOnlyConfig {
                position: self.position.position(),
                max_position: self.effective_max_position,
                mid_price: self.latest_mid,
                max_position_value: self.safety.kill_switch.max_position_value(),
                asset: self.config.asset.to_string(),
                margin_used: margin_state.margin_used,
                account_value: margin_state.account_value,
                // Dynamic reduce-only based on liquidation proximity
                liquidation_price: margin_state.liquidation_price,
                liquidation_buffer_ratio: margin_state.liquidation_buffer_ratio(),
                liquidation_trigger_threshold: quoting::DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD,
            };
            quoting::QuoteFilter::apply_reduce_only_single(&mut bid, &mut ask, &reduce_only_config);

            debug!(
                bid = ?bid.as_ref().map(|q| (q.price, q.size)),
                ask = ?ask.as_ref().map(|q| (q.price, q.size)),
                "Calculated quotes"
            );

            // Handle bid side
            self.reconcile_side(Side::Buy, bid).await?;

            // Handle ask side
            self.reconcile_side(Side::Sell, ask).await?;
        }

        Ok(())
    }

    /// Initiate bulk cancel for multiple orders and update order states appropriately.
    ///
    /// RATE LIMIT OPTIMIZATION: Uses single bulk cancel API call instead of individual cancels.
    /// Per Hyperliquid docs: batched requests are 1 weight for IP limits (vs n for individual).
    async fn initiate_bulk_cancel(&mut self, oids: Vec<u64>) {
        if oids.is_empty() {
            return;
        }

        // Collect order prices for Bayesian learning before cancel
        let order_prices: Vec<Option<f64>> = oids
            .iter()
            .map(|&oid| self.orders.get_order(oid).map(|o| o.price))
            .collect();

        // Mark all as CancelPending, filter out those not in cancellable state
        let cancellable_oids: Vec<u64> = oids
            .iter()
            .filter(|&&oid| self.orders.initiate_cancel(oid))
            .copied()
            .collect();

        if cancellable_oids.is_empty() {
            debug!("No orders in cancellable state for bulk cancel");
            return;
        }

        // Debug level here since executor.rs logs at INFO level
        debug!("Bulk cancelling {} orders", cancellable_oids.len());

        let num_cancels = cancellable_oids.len() as u32;
        let results = self
            .executor
            .cancel_bulk_orders(&self.config.asset, cancellable_oids.clone())
            .await;

        // Record API call for rate limit tracking
        // Bulk cancel: 1 IP weight, n address requests
        self.infra
            .proactive_rate_tracker
            .record_call(1, num_cancels);

        // Process results and update order states
        for (i, (oid, result)) in cancellable_oids.iter().zip(results.iter()).enumerate() {
            let order_price = order_prices.get(i).and_then(|p| *p);

            match result {
                CancelResult::Cancelled | CancelResult::AlreadyCancelled => {
                    self.orders.on_cancel_confirmed(*oid);
                    // Record cancel observation for Bayesian learning
                    if let Some(price) = order_price {
                        if self.latest_mid > 0.0 {
                            let depth_bps =
                                ((price - self.latest_mid).abs() / self.latest_mid) * 10_000.0;
                            self.strategy.record_fill_observation(depth_bps, false);
                        }
                    }
                }
                CancelResult::AlreadyFilled => {
                    self.orders.on_cancel_already_filled(*oid);
                    info!("Order {} was already filled when cancelled", oid);
                }
                CancelResult::Failed => {
                    self.orders.on_cancel_failed(*oid);
                    warn!("Bulk cancel failed for oid={}, reverted to active", oid);
                }
            }
        }
    }

    /// Initiate cancel and update order state appropriately.
    /// Does NOT remove order from tracking - that happens via cleanup cycle.
    async fn initiate_and_track_cancel(&mut self, oid: u64) {
        // Get order price before cancel for Bayesian learning
        let order_price = self.orders.get_order(oid).map(|o| o.price);

        // Mark as CancelPending before sending request
        if !self.orders.initiate_cancel(oid) {
            debug!("Order {} not in cancellable state", oid);
            return;
        }

        let cancel_result = self.cancel_with_retry(&self.config.asset, oid, 3).await;

        match cancel_result {
            CancelResult::Cancelled | CancelResult::AlreadyCancelled => {
                // Cancel confirmed - start fill window (DO NOT REMOVE YET)
                self.orders.on_cancel_confirmed(oid);
                info!("Cancel confirmed for oid={}, waiting for fill window", oid);

                // Record cancel observation for Bayesian fill probability learning
                // This teaches the model that orders at this depth did NOT fill
                if let Some(price) = order_price {
                    if self.latest_mid > 0.0 {
                        let depth_bps =
                            ((price - self.latest_mid).abs() / self.latest_mid) * 10_000.0;
                        self.strategy.record_fill_observation(depth_bps, false);
                        debug!(
                            oid = oid,
                            depth_bps = %format!("{:.2}", depth_bps),
                            "Recorded cancel observation for Bayesian learning"
                        );
                    }
                }
            }
            CancelResult::AlreadyFilled => {
                // Order was filled - mark appropriately
                self.orders.on_cancel_already_filled(oid);
                info!("Order {} was already filled when cancelled", oid);
                // Note: Fill observation is recorded via FillProcessor, not here
            }
            CancelResult::Failed => {
                // Cancel failed - revert state
                self.orders.on_cancel_failed(oid);
                warn!("Cancel failed for oid={}, reverted to active", oid);
            }
        }
    }

    /// Reconcile orders on one side.
    /// Cancels ALL orders on a side (not just first) with retry logic.
    async fn reconcile_side(&mut self, side: Side, new_quote: Option<Quote>) -> Result<()> {
        // Get ALL active orders on this side (excludes those being cancelled)
        let current_orders: Vec<u64> = self
            .orders
            .get_all_by_side(side)
            .iter()
            .map(|o| o.oid)
            .collect();

        // Check for orders in cancel process (for logging/debugging)
        let pending_cancels: Vec<u64> = self
            .orders
            .get_all_by_side_including_pending(side)
            .iter()
            .filter(|o| {
                matches!(
                    o.state,
                    OrderState::CancelPending | OrderState::CancelConfirmed
                )
            })
            .map(|o| o.oid)
            .collect();

        if !pending_cancels.is_empty() {
            debug!(
                side = %side_str(side),
                pending_cancel_count = pending_cancels.len(),
                "Orders still in cancel window on this side"
            );
        }

        debug!(
            side = %side_str(side),
            current_count = current_orders.len(),
            current_oids = ?current_orders,
            new = ?new_quote.as_ref().map(|q| (q.price, q.size)),
            "Reconciling"
        );

        match (current_orders.is_empty(), new_quote) {
            // Have orders, no quote -> cancel all
            (false, None) => {
                for oid in current_orders {
                    self.initiate_and_track_cancel(oid).await;
                }
            }
            // Have orders, have quote -> check if needs update, cancel all if so
            (false, Some(quote)) => {
                let needs_update = self
                    .orders
                    .needs_update(side, &quote, self.config.max_bps_diff);

                if needs_update {
                    // Cancel all orders on this side
                    for oid in current_orders {
                        self.initiate_and_track_cancel(oid).await;
                    }
                    // Place new order
                    self.place_new_order(side, &quote).await?;
                }
            }
            // No orders, have quote -> place new
            (true, Some(quote)) => {
                self.place_new_order(side, &quote).await?;
            }
            // No orders, no quote -> nothing to do
            (true, None) => {}
        }

        Ok(())
    }

    /// Reconcile multi-level ladder orders on one side.
    ///
    /// Manages multiple orders per side for ladder quoting strategy.
    /// Uses bulk order placement for efficiency (single API call for all levels).
    async fn reconcile_ladder_side(&mut self, side: Side, new_quotes: Vec<Quote>) -> Result<()> {
        let is_buy = side == Side::Buy;

        // Phase 5: Check rate limiter before placing orders
        if self.infra.rate_limiter.should_skip(is_buy) {
            if let Some(remaining) = self.infra.rate_limiter.remaining_backoff(is_buy) {
                debug!(
                    side = %side_str(side),
                    remaining_secs = %format!("{:.1}", remaining.as_secs_f64()),
                    "Skipping ladder reconciliation due to rate limit backoff"
                );
            }
            return Ok(());
        }

        // Get all active orders on this side (excludes those being cancelled)
        let current_orders: Vec<u64> = self
            .orders
            .get_all_by_side(side)
            .iter()
            .map(|o| o.oid)
            .collect();

        debug!(
            side = %side_str(side),
            current_count = current_orders.len(),
            new_levels = new_quotes.len(),
            "Reconciling ladder"
        );

        // If no new quotes, cancel all existing orders
        if new_quotes.is_empty() {
            // RATE LIMIT OPTIMIZATION: Use bulk cancel instead of individual cancels
            self.initiate_bulk_cancel(current_orders).await;
            return Ok(());
        }

        // Check if ladder needs update
        let needs_update = self.ladder_needs_update(side, &new_quotes);

        if needs_update || current_orders.is_empty() {
            // RATE LIMIT OPTIMIZATION: Cancel all existing orders in single API call
            self.initiate_bulk_cancel(current_orders.clone()).await;

            // Build order specs for bulk placement WITH CLOIDs for deterministic tracking (Phase 1)
            let is_buy = side == Side::Buy;
            let mut order_specs: Vec<OrderSpec> = Vec::new();

            // Track cumulative exposure for exchange limit checking (Phase 2 Fix)
            let mut cumulative_size = 0.0;

            for quote in &new_quotes {
                if quote.size <= 0.0 {
                    continue;
                }

                // Apply margin check to each level
                let sizing_result = self.infra.margin_sizer.adjust_size(
                    quote.size,
                    quote.price,
                    self.position.position(),
                    is_buy,
                );

                if sizing_result.adjusted_size <= 0.0 {
                    debug!(
                        side = %side_str(side),
                        price = quote.price,
                        requested_size = quote.size,
                        reason = ?sizing_result.constraint_reason,
                        "Ladder level blocked by margin constraints"
                    );
                    continue;
                }

                // Truncate size to sz_decimals to ensure valid order size
                let mut truncated_size =
                    truncate_float(sizing_result.adjusted_size, self.config.sz_decimals, false);
                if truncated_size <= 0.0 {
                    continue;
                }

                // Phase 2 Fix: Pre-flight exchange limit check
                // Check if this order (plus all orders we've already queued) would exceed limits
                let total_exposure = cumulative_size + truncated_size;
                let (safe_size, was_clamped, reason) = self
                    .infra
                    .exchange_limits
                    .calculate_safe_order_size(total_exposure, is_buy, self.position.position());

                if safe_size <= cumulative_size {
                    // No additional capacity - skip this level
                    if let Some(reason) = reason {
                        debug!(
                            side = %side_str(side),
                            price = quote.price,
                            requested_size = truncated_size,
                            reason = %reason,
                            "Ladder level blocked by exchange limits"
                        );
                    }
                    continue;
                }

                // How much can we actually add?
                let available_for_this_level = safe_size - cumulative_size;
                if was_clamped && available_for_this_level < truncated_size {
                    // Clamp this order size to what's available
                    truncated_size =
                        truncate_float(available_for_this_level, self.config.sz_decimals, false);
                    if truncated_size <= 0.0 {
                        continue;
                    }
                    debug!(
                        side = %side_str(side),
                        price = quote.price,
                        original_size = sizing_result.adjusted_size,
                        clamped_size = truncated_size,
                        reason = ?reason,
                        "Ladder level size clamped by exchange limits"
                    );
                }

                cumulative_size += truncated_size;

                // Generate CLOID for deterministic fill tracking (Phase 1 Fix)
                let cloid = uuid::Uuid::new_v4().to_string();
                order_specs.push(OrderSpec::with_cloid(
                    quote.price,
                    truncated_size,
                    is_buy,
                    cloid,
                ));
            }

            if order_specs.is_empty() {
                debug!(
                    side = %side_str(side),
                    "No ladder levels to place (all blocked by margin)"
                );
                return Ok(());
            }

            debug!(
                side = %side_str(side),
                levels = order_specs.len(),
                "Placing ladder levels via bulk order with CLOIDs"
            );

            // Pre-register orders as pending with CLOIDs BEFORE the API call (Phase 1 Fix).
            // CLOID lookup is deterministic - eliminates timing race between REST and WebSocket.
            let mut cloids_for_tracker: Vec<String> = Vec::with_capacity(order_specs.len());
            for spec in &order_specs {
                if let Some(ref cloid) = spec.cloid {
                    self.orders
                        .add_pending_with_cloid(side, spec.price, spec.size, cloid.clone());
                    cloids_for_tracker.push(cloid.clone());
                } else {
                    // Fallback (shouldn't happen with new code)
                    self.orders.add_pending(side, spec.price, spec.size);
                }
            }

            // Phase 7: Register expected CLOIDs with orphan tracker BEFORE API call.
            // This protects these orders from being cancelled as orphans if safety_sync
            // runs before finalization completes.
            if !cloids_for_tracker.is_empty() {
                self.infra
                    .orphan_tracker
                    .register_expected_cloids(&cloids_for_tracker);
            }

            // Place all orders in a single API call
            let num_orders = order_specs.len() as u32;
            let results = self
                .executor
                .place_bulk_orders(&self.config.asset, order_specs.clone())
                .await;

            // Record API call for rate limit tracking
            // Bulk order: 1 IP weight, n address requests
            self.infra.proactive_rate_tracker.record_call(1, num_orders);

            // Finalize pending orders with real OIDs (using CLOID for deterministic matching)
            for (i, result) in results.iter().enumerate() {
                let spec = &order_specs[i];

                // Phase 5: Record success/rejection for rate limiting
                if result.oid > 0 {
                    // Order placed successfully - reset rejection counter
                    self.infra.rate_limiter.record_success(is_buy);
                    self.infra.recovery_manager.record_success();
                } else if let Some(ref err) = result.error {
                    // Order rejected - record for rate limiting and reconciliation
                    if let Some(backoff) = self.infra.rate_limiter.record_rejection(is_buy, err) {
                        warn!(
                            side = %side_str(side),
                            backoff_secs = %format!("{:.1}", backoff.as_secs_f64()),
                            error = %err,
                            "Rate limiter entering backoff due to rejection"
                        );
                    }
                    // Phase 4: Trigger reconciliation for position-related rejections
                    self.infra.reconciler.on_order_rejection(err);
                    // Phase 3: Record rejection for recovery state machine
                    // This may transition to IocRecovery if consecutive rejections exceed threshold
                    if self.infra.recovery_manager.record_rejection(is_buy, err) {
                        warn!(
                            side = %side_str(side),
                            "Recovery manager escalating to IOC recovery mode"
                        );
                    }
                    // Phase 7: Mark CLOID as failed in orphan tracker
                    if let Some(ref cloid) = spec.cloid {
                        self.infra.orphan_tracker.mark_failed(cloid);
                    }
                }

                if result.oid > 0 {
                    let cloid = spec.cloid.as_ref().or(result.cloid.as_ref());

                    // Phase 7: Record OID for CLOID in orphan tracker.
                    // This associates the OID with the expected CLOID for protection.
                    if let Some(c) = cloid {
                        self.infra
                            .orphan_tracker
                            .record_oid_for_cloid(c, result.oid);
                    }

                    if result.filled {
                        // CRITICAL FIX: Order filled immediately - update position NOW
                        // Don't wait for WebSocket which may be delayed by seconds!
                        //
                        // Previous bug: We removed from pending but didn't update position,
                        // causing position drift until WebSocket arrived (100-8000ms delay).
                        // This allowed placing duplicate orders and position runaway.

                        let filled_size = result.resting_size; // 0 if fully filled
                        let actual_fill = if filled_size < EPSILON {
                            spec.size // Fully filled
                        } else {
                            spec.size - filled_size // Partial immediate fill
                        };

                        // Update position immediately from API response
                        let is_buy = side == Side::Buy;
                        self.position.process_fill(actual_fill, is_buy);

                        // Track order as FilledImmediately for WebSocket deduplication
                        // When WebSocket fill arrives, it will find this order and deduplicate
                        let mut tracked = if let Some(c) = cloid {
                            TrackedOrder::with_cloid(
                                result.oid,
                                c.clone(),
                                side,
                                spec.price,
                                spec.size,
                            )
                        } else {
                            TrackedOrder::new(result.oid, side, spec.price, spec.size)
                        };
                        tracked.filled = actual_fill;
                        tracked.transition_to(OrderState::FilledImmediately);
                        self.orders.add_order(tracked);

                        // Remove from pending using CLOID (Phase 1 Fix)
                        if let Some(c) = cloid {
                            self.orders.remove_pending_by_cloid(c);
                            // Phase 7: Mark as finalized - order now tracked locally
                            self.infra.orphan_tracker.mark_finalized(c, result.oid);
                        } else {
                            self.orders.remove_pending(side, spec.price);
                        }

                        // Register immediate fill amount for WS deduplication.
                        // The FillProcessor tracks the AMOUNT filled immediately, so when
                        // WS fills arrive, it can properly dedup by decrementing from the
                        // tracked amount. This works for both full and partial immediate fills.
                        self.safety
                            .fill_processor
                            .pre_register_immediate_fill(result.oid, actual_fill);

                        // If partially filled, still need to track the resting portion
                        if filled_size > EPSILON {
                            // Partial immediate fill - update order to PartialFilled so it's
                            // included in reconciliation for the resting portion.
                            //
                            // IMPORTANT: Can't use FilledImmediately for partial fills because:
                            // - FilledImmediately is not in is_active() (Resting | PartialFilled)
                            // - Order wouldn't be included in get_all_by_side()
                            // - The resting portion would become orphaned!
                            //
                            // For partial fills, we need PartialFilled state so reconciliation
                            // sees the resting portion and can modify/cancel it as needed.
                            if let Some(order) = self.orders.get_order_mut(result.oid) {
                                order.transition_to(OrderState::PartialFilled);
                                // Update size to reflect what's actually resting
                                order.size = filled_size;
                                debug!(
                                    oid = result.oid,
                                    filled = actual_fill,
                                    resting_size = filled_size,
                                    "Partial immediate fill - tracking resting portion as PartialFilled"
                                );
                            }
                        }

                        info!(
                            oid = result.oid,
                            cloid = ?cloid,
                            price = spec.price,
                            size = spec.size,
                            actual_fill = actual_fill,
                            resting = filled_size,
                            position = self.position.position(),
                            "Order filled immediately - position updated"
                        );

                        continue;
                    }

                    // Move from pending to tracked with real OID using CLOID (Phase 1 Fix)
                    if let Some(c) = cloid {
                        self.orders
                            .finalize_pending_by_cloid(c, result.oid, result.resting_size);
                        // Phase 7: Mark as finalized - order now fully tracked locally
                        self.infra.orphan_tracker.mark_finalized(c, result.oid);
                    } else {
                        self.orders.finalize_pending(
                            side,
                            spec.price,
                            result.oid,
                            result.resting_size,
                        );
                    }

                    // Initialize queue tracking for this order
                    // depth_ahead is 0.0 initially; will be updated on L2 book updates
                    self.tier1.queue_tracker.order_placed(
                        result.oid,
                        spec.price,
                        result.resting_size,
                        0.0, // depth_ahead estimated; will be refined by L2 updates
                        side == Side::Buy,
                    );
                }
            }
        }

        Ok(())
    }

    /// Check if the ladder needs to be updated.
    ///
    /// Returns true if:
    /// - Number of levels has changed
    /// - Any level's price has moved more than max_bps_diff
    /// - Any level's size has changed by more than 10%
    fn ladder_needs_update(&self, side: Side, new_quotes: &[Quote]) -> bool {
        let current = self.orders.get_all_by_side(side);

        // Different number of levels
        if current.len() != new_quotes.len() {
            return true;
        }

        // Sort current orders by price (bids: descending, asks: ascending)
        let mut sorted_current: Vec<_> = current.iter().collect();
        if side == Side::Buy {
            sorted_current.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
        } else {
            sorted_current.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
        }

        // Check each level for meaningful price/size changes
        for (order, quote) in sorted_current.iter().zip(new_quotes.iter()) {
            // Price change check
            let price_diff_bps = ((order.price - quote.price) / order.price).abs() * 10000.0;
            if price_diff_bps > self.config.max_bps_diff as f64 {
                return true;
            }

            // Size change check (10% threshold)
            if order.size > 0.0 {
                let size_diff_pct = ((order.size - quote.size) / order.size).abs();
                if size_diff_pct > 0.1 {
                    return true;
                }
            }
        }

        false
    }

    /// Smart ladder reconciliation with ORDER MODIFY for queue preservation.
    ///
    /// Uses differential updates (SKIP/MODIFY/CANCEL+PLACE) to preserve queue position
    /// when possible. Both sides are processed in parallel for reduced latency.
    ///
    /// Decision logic per level:
    /// - SKIP: Order unchanged (price ≤1 bps, size ≤5%)
    /// - MODIFY: Small change (price ≤10 bps, size ≤50%) - preserves queue
    /// - CANCEL+PLACE: Large change - fresh queue position
    async fn reconcile_ladder_smart(
        &mut self,
        bid_quotes: Vec<Quote>,
        ask_quotes: Vec<Quote>,
    ) -> Result<()> {
        use crate::market_maker::quoting::LadderLevel;
        use crate::market_maker::tracking::{
            reconcile_side_smart, reconcile_side_smart_with_impulse,
        };

        let reconcile_config = &self.config.reconcile;

        // Convert Quote to LadderLevel for reconciliation
        let bid_levels: Vec<LadderLevel> = bid_quotes
            .iter()
            .map(|q| LadderLevel {
                price: q.price,
                size: q.size,
                depth_bps: 0.0, // Not used for reconciliation
            })
            .collect();

        let ask_levels: Vec<LadderLevel> = ask_quotes
            .iter()
            .map(|q| LadderLevel {
                price: q.price,
                size: q.size,
                depth_bps: 0.0,
            })
            .collect();

        // Log quote positions relative to market mid for diagnosing fill issues
        // Shows where our quotes are placed vs the exchange mid price
        if !bid_levels.is_empty() && !ask_levels.is_empty() && self.latest_mid > 0.0 {
            let our_best_bid = bid_levels.first().map(|l| l.price).unwrap_or(0.0);
            let our_best_ask = ask_levels.first().map(|l| l.price).unwrap_or(0.0);
            let bid_distance_bps = (self.latest_mid - our_best_bid) / self.latest_mid * 10000.0;
            let ask_distance_bps = (our_best_ask - self.latest_mid) / self.latest_mid * 10000.0;
            let total_spread_bps = bid_distance_bps + ask_distance_bps;

            info!(
                market_mid = %format!("{:.4}", self.latest_mid),
                our_bid = %format!("{:.4}", our_best_bid),
                our_ask = %format!("{:.4}", our_best_ask),
                bid_from_mid_bps = %format!("{:.1}", bid_distance_bps),
                ask_from_mid_bps = %format!("{:.1}", ask_distance_bps),
                total_spread_bps = %format!("{:.1}", total_spread_bps),
                "Quote positions vs market"
            );
        }

        // Get current orders for each side
        let current_bids: Vec<&TrackedOrder> = self.orders.get_all_by_side(Side::Buy);
        let current_asks: Vec<&TrackedOrder> = self.orders.get_all_by_side(Side::Sell);

        // Impulse Control: Check execution budget before proceeding
        let impulse_enabled =
            self.infra.impulse_control_enabled && reconcile_config.use_impulse_filter;
        if impulse_enabled {
            // Estimate potential actions: at most (current + target) per side
            let max_potential_actions =
                current_bids.len() + bid_levels.len() + current_asks.len() + ask_levels.len();

            if !self
                .infra
                .execution_budget
                .can_afford_bulk(max_potential_actions)
            {
                debug!(
                    budget = %format!("{:.1}", self.infra.execution_budget.available()),
                    potential_actions = max_potential_actions,
                    "Impulse control: skipping reconciliation due to insufficient execution budget"
                );
                return Ok(());
            }
        }

        // Generate actions for each side using smart reconciliation
        // When impulse control is enabled, use the impulse-aware version with Δλ filtering
        let (mut bid_actions, bid_stats, mut ask_actions, ask_stats) = if impulse_enabled {
            // Get mid price for P(fill) estimation
            let mid_price = Some(self.latest_mid);

            // Use impulse-aware reconciliation with queue tracker and impulse filter
            let (bid_acts, bid_st) = reconcile_side_smart_with_impulse(
                &current_bids,
                &bid_levels,
                Side::Buy,
                reconcile_config,
                Some(&self.tier1.queue_tracker),
                Some(&mut self.infra.impulse_filter),
                mid_price,
            );

            let (ask_acts, ask_st) = reconcile_side_smart_with_impulse(
                &current_asks,
                &ask_levels,
                Side::Sell,
                reconcile_config,
                Some(&self.tier1.queue_tracker),
                Some(&mut self.infra.impulse_filter),
                mid_price,
            );

            // Log impulse filter statistics
            if bid_st.impulse_filtered_count > 0
                || ask_st.impulse_filtered_count > 0
                || bid_st.queue_locked_count > 0
                || ask_st.queue_locked_count > 0
            {
                debug!(
                    bid_impulse_filtered = bid_st.impulse_filtered_count,
                    ask_impulse_filtered = ask_st.impulse_filtered_count,
                    bid_queue_locked = bid_st.queue_locked_count,
                    ask_queue_locked = ask_st.queue_locked_count,
                    "Impulse control: filtered low-value updates"
                );
            }

            (bid_acts, Some(bid_st), ask_acts, Some(ask_st))
        } else {
            // Standard reconciliation without impulse filtering
            let bid_acts =
                reconcile_side_smart(&current_bids, &bid_levels, Side::Buy, reconcile_config);
            let ask_acts =
                reconcile_side_smart(&current_asks, &ask_levels, Side::Sell, reconcile_config);
            (bid_acts, None, ask_acts, None)
        };

        // Track impulse stats for metrics
        let _impulse_stats = (bid_stats, ask_stats);

        // Queue-aware enhancement: Check if any "skipped" orders should be refreshed
        // based on queue position analysis
        if reconcile_config.use_queue_aware {
            // Find orders that were skipped (in current but not in any action)
            let bid_action_oids: std::collections::HashSet<u64> = bid_actions
                .iter()
                .filter_map(|a| match a {
                    LadderAction::Cancel { oid } | LadderAction::Modify { oid, .. } => Some(*oid),
                    _ => None,
                })
                .collect();
            let ask_action_oids: std::collections::HashSet<u64> = ask_actions
                .iter()
                .filter_map(|a| match a {
                    LadderAction::Cancel { oid } | LadderAction::Modify { oid, .. } => Some(*oid),
                    _ => None,
                })
                .collect();

            // Check skipped bids for queue refresh
            for order in &current_bids {
                if !bid_action_oids.contains(&order.oid) {
                    // This order was skipped - check if queue tracker says refresh
                    if self
                        .tier1
                        .queue_tracker
                        .should_refresh(order.oid, reconcile_config.queue_horizon_seconds)
                    {
                        debug!(
                            oid = order.oid,
                            price = %order.price,
                            "Queue-aware refresh: forcing bid order refresh due to low fill probability"
                        );
                        // Add cancel action for this skipped order
                        bid_actions.push(LadderAction::Cancel { oid: order.oid });
                        // Find matching target level to place
                        for level in &bid_levels {
                            if (level.price - order.price).abs() < crate::EPSILON * order.price {
                                bid_actions.push(LadderAction::Place {
                                    side: Side::Buy,
                                    price: level.price,
                                    size: level.size,
                                });
                                break;
                            }
                        }
                    }
                }
            }

            // Check skipped asks for queue refresh
            for order in &current_asks {
                if !ask_action_oids.contains(&order.oid)
                    && self
                        .tier1
                        .queue_tracker
                        .should_refresh(order.oid, reconcile_config.queue_horizon_seconds)
                {
                    debug!(
                        oid = order.oid,
                        price = %order.price,
                        "Queue-aware refresh: forcing ask order refresh due to low fill probability"
                    );
                    ask_actions.push(LadderAction::Cancel { oid: order.oid });
                    for level in &ask_levels {
                        if (level.price - order.price).abs() < crate::EPSILON * order.price {
                            ask_actions.push(LadderAction::Place {
                                side: Side::Sell,
                                price: level.price,
                                size: level.size,
                            });
                            break;
                        }
                    }
                }
            }
        }

        // Partition actions by type for batching
        let (bid_cancels, bid_modifies, bid_places) =
            partition_ladder_actions(&bid_actions, Side::Buy);
        let (ask_cancels, ask_modifies, ask_places) =
            partition_ladder_actions(&ask_actions, Side::Sell);

        // Track action counts for impulse control budget spending
        let cancel_count = bid_cancels.len() + ask_cancels.len();
        let modify_count = bid_modifies.len() + ask_modifies.len();

        // Log action summary (skip count only considers existing orders minus those modified/cancelled)
        let _total_skips = (current_bids.len() + current_asks.len()).saturating_sub(
            bid_cancels.len() + bid_modifies.len() + ask_cancels.len() + ask_modifies.len(),
        );
        if !bid_actions.is_empty() || !ask_actions.is_empty() {
            debug!(
                bid_skip = current_bids
                    .len()
                    .saturating_sub(bid_cancels.len() + bid_modifies.len()),
                bid_modify = bid_modifies.len(),
                bid_cancel = bid_cancels.len(),
                bid_place = bid_places.len(),
                ask_skip = current_asks
                    .len()
                    .saturating_sub(ask_cancels.len() + ask_modifies.len()),
                ask_modify = ask_modifies.len(),
                ask_cancel = ask_cancels.len(),
                ask_place = ask_places.len(),
                "Smart reconciliation actions"
            );
        }

        // Execute cancels first (bulk cancel is efficient)
        let all_cancels: Vec<u64> = bid_cancels.into_iter().chain(ask_cancels).collect();
        if !all_cancels.is_empty() {
            self.initiate_bulk_cancel(all_cancels).await;
        }

        // Execute modifies (preserves queue position)
        // RATE LIMIT FIX: Check modify debounce before executing
        let all_modifies: Vec<ModifySpec> = bid_modifies.into_iter().chain(ask_modifies).collect();
        if !all_modifies.is_empty() {
            // Check if we're rate limited or modify interval hasn't passed
            if self.infra.proactive_rate_tracker.is_rate_limited() {
                debug!(
                    remaining_backoff = ?self.infra.proactive_rate_tracker.remaining_backoff(),
                    "Skipping modify: rate limited (429 backoff active)"
                );
                // Convert modifies to cancel+place instead (will be handled in next cycle)
                // Don't execute modifies during backoff
            } else if !self.infra.proactive_rate_tracker.can_modify() {
                debug!("Skipping modify: minimum modify interval not elapsed (prevents OID churn)");
                // Skip this cycle, will retry next time
            } else {
                // Mark that we're doing a modify
                self.infra.proactive_rate_tracker.mark_modify();

                let num_modifies = all_modifies.len() as u32;
                let modify_results = self
                    .executor
                    .modify_bulk_orders(&self.config.asset, all_modifies.clone())
                    .await;

                // Record API call for rate tracking
                self.infra
                    .proactive_rate_tracker
                    .record_call(1, num_modifies);

                // Track OID remaps for summary logging
                let mut oid_remap_count = 0u32;
                let mut modify_success_count = 0u32;

                // Process modify results
                for (i, result) in modify_results.iter().enumerate() {
                    let spec = &all_modifies[i];
                    if result.success {
                        // CRITICAL FIX: Hyperliquid modify can return NEW OID.
                        // The exchange may assign a new order ID on price change.
                        // We must re-key our tracking to use the new OID.
                        let effective_oid = if result.oid > 0 && result.oid != spec.oid {
                            // OID changed - re-key the order in tracking
                            if self.orders.replace_oid(spec.oid, result.oid) {
                                // Log at DEBUG to reduce verbosity - summary logged at end of cycle
                                debug!(old_oid = spec.oid, new_oid = result.oid, "OID remapped");
                                oid_remap_count += 1;
                            }
                            result.oid
                        } else {
                            spec.oid
                        };

                        // Update tracking with new price/size using the effective OID
                        self.orders
                            .on_modify_success(effective_oid, spec.new_price, spec.new_size);
                        // Record successful modify in metrics
                        self.infra.prometheus.record_order_modified();
                        modify_success_count += 1;
                    } else {
                        // Modify failed - fallback to cancel+place
                        warn!(
                            oid = spec.oid,
                            error = ?result.error,
                            "Modify failed, falling back to cancel+place"
                        );
                        // Record fallback in metrics
                        self.infra.prometheus.record_modify_fallback();

                        // Check if the error indicates order is already filled or canceled.
                        // CRITICAL FIX: Check "canceled" FIRST because the error message
                        // "Cannot modify canceled or filled order" contains BOTH words.
                        // We need to handle canceled orders (remove from tracking) differently
                        // from filled orders (wait for WS fill notification).
                        let error_msg = result.error.as_deref().unwrap_or("");
                        let is_already_canceled = error_msg.contains("canceled");
                        let is_already_filled =
                            !is_already_canceled && error_msg.contains("filled");

                        if is_already_canceled {
                            // Order was already canceled - remove from tracking
                            debug!(
                            oid = spec.oid,
                            "Modify failed because order already canceled - removing from tracking"
                        );
                            self.orders.remove_order(spec.oid);
                            continue;
                        }

                        if is_already_filled {
                            // Order was filled - mark as FilledImmediately so it's excluded
                            // from future reconciliation until WS fill arrives
                            debug!(
                                oid = spec.oid,
                                "Modify failed because order already filled - marking for WS fill"
                            );
                            // Transition to FilledImmediately - this removes it from is_active()
                            // so get_all_by_side won't include it in future modify attempts
                            if let Some(order) = self.orders.get_order_mut(spec.oid) {
                                order.transition_to(OrderState::FilledImmediately);
                            }
                            continue;
                        }

                        // Neither canceled nor filled - proceed with cancel+place fallback
                        // (This handles other errors like "Order not found", rate limits, etc.)

                        // Initiate cancel in OrderManager BEFORE sending to exchange
                        // This properly transitions state and prevents stale tracking
                        self.orders.initiate_cancel(spec.oid);

                        // Cancel the order on exchange
                        let cancel_result = self
                            .executor
                            .cancel_order(&self.config.asset, spec.oid)
                            .await;

                        // Update OrderManager based on cancel result
                        match cancel_result {
                            CancelResult::Cancelled | CancelResult::AlreadyCancelled => {
                                self.orders.on_cancel_confirmed(spec.oid);
                            }
                            CancelResult::AlreadyFilled => {
                                // Order was filled during our cancel attempt
                                // Mark it appropriately and wait for WS fill
                                self.orders.on_cancel_already_filled(spec.oid);
                                debug!(
                                    oid = spec.oid,
                                    "Cancel in modify fallback found order already filled"
                                );
                                continue; // Don't place replacement order
                            }
                            CancelResult::Failed => {
                                // Cancel failed - order may still be active
                                // Revert state and don't place replacement to avoid duplicates
                                self.orders.on_cancel_failed(spec.oid);
                                warn!(
                                    oid = spec.oid,
                                    "Cancel failed in modify fallback - not placing replacement"
                                );
                                continue;
                            }
                        }

                        // Place new order at target price/size with proper tracking
                        let side = if spec.is_buy { Side::Buy } else { Side::Sell };
                        let cloid = uuid::Uuid::new_v4().to_string();

                        // Pre-register as pending BEFORE API call (consistent with bulk order flow)
                        self.orders.add_pending_with_cloid(
                            side,
                            spec.new_price,
                            spec.new_size,
                            cloid.clone(),
                        );

                        // Phase 7: Register expected CLOID with orphan tracker BEFORE API call
                        self.infra
                            .orphan_tracker
                            .register_expected_cloids(std::slice::from_ref(&cloid));

                        // Pass the pre-registered CLOID to place_order for deterministic matching.
                        // This ensures the pending order is finalized with the correct OID.
                        let place_result = self
                            .executor
                            .place_order(
                                &self.config.asset,
                                spec.new_price,
                                spec.new_size,
                                spec.is_buy,
                                Some(cloid.clone()),
                            )
                            .await;

                        // Finalize tracking based on placement result
                        if place_result.oid > 0 {
                            // Phase 7: Record OID for CLOID in orphan tracker
                            self.infra
                                .orphan_tracker
                                .record_oid_for_cloid(&cloid, place_result.oid);

                            if place_result.filled {
                                // Filled immediately - calculate actual fill vs resting
                                self.orders.remove_pending_by_cloid(&cloid);
                                // Phase 7: Mark as finalized
                                self.infra
                                    .orphan_tracker
                                    .mark_finalized(&cloid, place_result.oid);

                                let resting_size = place_result.resting_size;
                                let actual_fill = if resting_size < EPSILON {
                                    spec.new_size // Fully filled
                                } else {
                                    spec.new_size - resting_size // Partial immediate fill
                                };

                                self.position.process_fill(actual_fill, spec.is_buy);

                                // Register immediate fill amount for WS deduplication
                                self.safety
                                    .fill_processor
                                    .pre_register_immediate_fill(place_result.oid, actual_fill);

                                let mut tracked = TrackedOrder::with_cloid(
                                    place_result.oid,
                                    cloid.clone(),
                                    side,
                                    spec.new_price,
                                    spec.new_size,
                                );
                                tracked.filled = actual_fill;

                                // Determine correct state based on fill completeness
                                if resting_size > EPSILON {
                                    // Partial fill - use PartialFilled so order is included in reconciliation
                                    tracked.transition_to(OrderState::PartialFilled);
                                    tracked.size = resting_size; // Track resting portion
                                    info!(
                                        oid = place_result.oid,
                                        price = spec.new_price,
                                        filled = actual_fill,
                                        resting = resting_size,
                                        "Modify fallback order partially filled"
                                    );
                                } else {
                                    // Fully filled - use FilledImmediately for WS dedup
                                    tracked.transition_to(OrderState::FilledImmediately);
                                    info!(
                                        oid = place_result.oid,
                                        price = spec.new_price,
                                        size = spec.new_size,
                                        "Modify fallback order filled immediately"
                                    );
                                }

                                self.orders.add_order(tracked);
                            } else {
                                // Resting - finalize pending to tracked
                                self.orders.finalize_pending_by_cloid(
                                    &cloid,
                                    place_result.oid,
                                    place_result.resting_size,
                                );
                                // Phase 7: Mark as finalized
                                self.infra
                                    .orphan_tracker
                                    .mark_finalized(&cloid, place_result.oid);
                                debug!(
                                    oid = place_result.oid,
                                    price = spec.new_price,
                                    size = spec.new_size,
                                    "Modify fallback order placed and tracked"
                                );
                            }
                        } else {
                            // Order placement failed - remove from pending
                            self.orders.remove_pending_by_cloid(&cloid);
                            // Phase 7: Mark CLOID as failed
                            self.infra.orphan_tracker.mark_failed(&cloid);
                            warn!(
                                price = spec.new_price,
                                size = spec.new_size,
                                error = ?place_result.error,
                                "Modify fallback order placement failed"
                            );
                        }
                    }
                }

                // Log consolidated quote cycle summary with spread info
                if modify_success_count > 0 {
                    // Get best quotes from tracked orders for spread calculation
                    let (best_bid, best_ask, bid_levels, ask_levels) =
                        self.orders.get_quote_summary();
                    let mid = if best_bid > 0.0 && best_ask > 0.0 {
                        (best_bid + best_ask) / 2.0
                    } else {
                        0.0
                    };
                    let spread_bps = if mid > 0.0 {
                        (best_ask - best_bid) / mid * 10000.0
                    } else {
                        0.0
                    };

                    info!(
                        modified = modify_success_count,
                        oid_remaps = oid_remap_count,
                        best_bid = %format!("{:.2}", best_bid),
                        best_ask = %format!("{:.2}", best_ask),
                        spread_bps = %format!("{:.2}", spread_bps),
                        bid_levels = bid_levels,
                        ask_levels = ask_levels,
                        position = %format!("{:.6}", self.position.position()),
                        "Quote cycle complete"
                    );
                }
            } // Close the else block for rate limit check
        }

        // Execute places (new orders)
        let bid_place_count = bid_places.len();
        let ask_place_count = ask_places.len();
        if !bid_places.is_empty() {
            self.place_bulk_ladder_orders(Side::Buy, bid_places).await?;
        }
        if !ask_places.is_empty() {
            self.place_bulk_ladder_orders(Side::Sell, ask_places)
                .await?;
        }

        // Impulse Control: Spend execution budget tokens for actions executed
        if impulse_enabled {
            let total_actions = cancel_count + modify_count + bid_place_count + ask_place_count;
            if total_actions > 0 {
                self.infra.execution_budget.spend(total_actions);
                debug!(
                    actions = total_actions,
                    budget_remaining = %format!("{:.1}", self.infra.execution_budget.available()),
                    "Impulse control: spent execution budget tokens"
                );
            }
        }

        Ok(())
    }

    /// Helper to place bulk ladder orders with proper tracking and margin checks.
    async fn place_bulk_ladder_orders(
        &mut self,
        side: Side,
        orders: Vec<(f64, f64)>, // (price, size) tuples
    ) -> Result<()> {
        let is_buy = side == Side::Buy;

        let mut order_specs: Vec<OrderSpec> = Vec::new();
        for (price, size) in orders {
            if size <= 0.0 {
                continue;
            }

            // Apply margin check
            let sizing_result =
                self.infra
                    .margin_sizer
                    .adjust_size(size, price, self.position.position(), is_buy);

            if sizing_result.adjusted_size <= 0.0 {
                continue;
            }

            let truncated_size =
                truncate_float(sizing_result.adjusted_size, self.config.sz_decimals, false);
            if truncated_size <= 0.0 {
                continue;
            }

            // Post-truncation notional check - truncation can push size below $10 minimum
            let notional = truncated_size * price;
            if notional < MIN_ORDER_NOTIONAL {
                tracing::debug!(
                    size = %format!("{:.6}", truncated_size),
                    price = %format!("{:.2}", price),
                    notional = %format!("{:.2}", notional),
                    min = MIN_ORDER_NOTIONAL,
                    "Skipping order: post-truncation notional below minimum"
                );
                continue;
            }

            let cloid = uuid::Uuid::new_v4().to_string();
            order_specs.push(OrderSpec::with_cloid(price, truncated_size, is_buy, cloid));
        }

        if order_specs.is_empty() {
            return Ok(());
        }

        // Pre-register as pending
        let mut cloids_for_tracker: Vec<String> = Vec::with_capacity(order_specs.len());
        for spec in &order_specs {
            if let Some(ref cloid) = spec.cloid {
                self.orders
                    .add_pending_with_cloid(side, spec.price, spec.size, cloid.clone());
                cloids_for_tracker.push(cloid.clone());
            }
        }

        // Phase 7: Register expected CLOIDs with orphan tracker BEFORE API call
        if !cloids_for_tracker.is_empty() {
            self.infra
                .orphan_tracker
                .register_expected_cloids(&cloids_for_tracker);
        }

        // Place orders
        let num_orders = order_specs.len() as u32;
        let results = self
            .executor
            .place_bulk_orders(&self.config.asset, order_specs.clone())
            .await;

        // Record API call
        self.infra.proactive_rate_tracker.record_call(1, num_orders);

        // Finalize pending orders with proper state handling
        for (i, result) in results.iter().enumerate() {
            let spec = &order_specs[i];

            // Handle success vs rejection for rate limiting and recovery
            if result.oid > 0 {
                self.infra.rate_limiter.record_success(is_buy);
                self.infra.recovery_manager.record_success();
            } else if let Some(ref err) = result.error {
                // Record rejection for rate limiting
                if let Some(backoff) = self.infra.rate_limiter.record_rejection(is_buy, err) {
                    warn!(
                        side = %if is_buy { "buy" } else { "sell" },
                        backoff_secs = %format!("{:.1}", backoff.as_secs_f64()),
                        error = %err,
                        "Rate limiter entering backoff due to rejection"
                    );
                }
                // Trigger reconciliation for position-related rejections
                self.infra.reconciler.on_order_rejection(err);
                // Record for recovery state machine
                if self.infra.recovery_manager.record_rejection(is_buy, err) {
                    warn!(
                        side = %if is_buy { "buy" } else { "sell" },
                        "Recovery manager escalating to IOC recovery mode"
                    );
                }
                // Remove from pending on rejection
                if let Some(ref cloid) = spec.cloid {
                    self.orders.remove_pending_by_cloid(cloid);
                    // Phase 7: Mark CLOID as failed
                    self.infra.orphan_tracker.mark_failed(cloid);
                }
                continue;
            }

            if result.oid > 0 {
                let cloid = spec.cloid.as_ref().or(result.cloid.as_ref());

                // Phase 7: Record OID for CLOID in orphan tracker
                if let Some(c) = cloid {
                    self.infra
                        .orphan_tracker
                        .record_oid_for_cloid(c, result.oid);
                }

                if result.filled {
                    // Order filled immediately - update position NOW
                    let filled_size = result.resting_size;
                    let actual_fill = if filled_size < EPSILON {
                        spec.size // Fully filled
                    } else {
                        spec.size - filled_size // Partial immediate fill
                    };

                    // Update position immediately from API response
                    self.position.process_fill(actual_fill, is_buy);

                    // Track order as FilledImmediately for WebSocket deduplication
                    let mut tracked = if let Some(c) = cloid {
                        TrackedOrder::with_cloid(result.oid, c.clone(), side, spec.price, spec.size)
                    } else {
                        TrackedOrder::new(result.oid, side, spec.price, spec.size)
                    };
                    tracked.filled = actual_fill;
                    tracked.transition_to(OrderState::FilledImmediately);
                    self.orders.add_order(tracked.clone());
                    // Also add to WsOrderStateManager
                    self.ws_state.add_order(tracked);

                    // Remove from pending using CLOID
                    if let Some(c) = cloid {
                        self.orders.remove_pending_by_cloid(c);
                        // Phase 7: Mark as finalized
                        self.infra.orphan_tracker.mark_finalized(c, result.oid);
                    } else {
                        self.orders.remove_pending(side, spec.price);
                    }

                    // Register immediate fill for WS deduplication
                    self.safety
                        .fill_processor
                        .pre_register_immediate_fill(result.oid, actual_fill);

                    // Handle partial immediate fills
                    if filled_size > EPSILON {
                        if let Some(order) = self.orders.get_order_mut(result.oid) {
                            order.transition_to(OrderState::PartialFilled);
                            order.size = filled_size;
                            debug!(
                                oid = result.oid,
                                filled = actual_fill,
                                resting_size = filled_size,
                                "Bulk order partial immediate fill - tracking resting portion"
                            );
                        }
                    }

                    info!(
                        oid = result.oid,
                        cloid = ?cloid,
                        price = spec.price,
                        size = spec.size,
                        actual_fill = actual_fill,
                        resting = filled_size,
                        position = self.position.position(),
                        "Bulk order filled immediately - position updated"
                    );

                    continue;
                }

                // Move from pending to tracked atomically using finalize_pending_by_cloid
                if let Some(c) = cloid {
                    self.orders
                        .finalize_pending_by_cloid(c, result.oid, result.resting_size);
                    // Phase 7: Mark as finalized
                    self.infra.orphan_tracker.mark_finalized(c, result.oid);
                } else {
                    self.orders
                        .finalize_pending(side, spec.price, result.oid, result.resting_size);
                }

                // Also add to WsOrderStateManager for improved state tracking
                let tracked = if let Some(c) = cloid {
                    TrackedOrder::with_cloid(
                        result.oid,
                        c.clone(),
                        side,
                        spec.price,
                        result.resting_size,
                    )
                } else {
                    TrackedOrder::new(result.oid, side, spec.price, result.resting_size)
                };
                self.ws_state.add_order(tracked);

                // Initialize queue tracking for this order
                self.tier1.queue_tracker.order_placed(
                    result.oid,
                    spec.price,
                    result.resting_size,
                    0.0, // depth_ahead will be refined by L2 updates
                    is_buy,
                );
            }
        }

        Ok(())
    }

    /// Phase 3: Check recovery state and handle IOC recovery if needed.
    ///
    /// This method implements the recovery state machine for stuck reduce-only mode:
    /// - Normal: No action needed
    /// - ReduceOnlyStuck: Detected via consecutive rejections, handled by record_rejection()
    /// - IocRecovery: Attempts aggressive IOC orders to reduce position
    /// - Cooldown: Waiting before resuming normal quoting
    async fn check_and_handle_recovery(&mut self) -> Result<Option<RecoveryAction>> {
        use infra::RecoveryState;

        match self.infra.recovery_manager.state().clone() {
            RecoveryState::Normal => {
                // Normal state - recovery manager handles transitions via record_rejection()
                // which is called from the order placement results loop
                Ok(None)
            }
            RecoveryState::ReduceOnlyStuck {
                is_buy_side,
                consecutive_rejections,
                ..
            } => {
                // Already in stuck state - waiting for rejections to hit threshold
                // The record_rejection() method will transition to IocRecovery when threshold is hit
                debug!(
                    is_buy_side = is_buy_side,
                    consecutive_rejections = consecutive_rejections,
                    threshold = self.infra.recovery_manager.config().rejection_threshold,
                    "In reduce-only-stuck state, waiting for threshold"
                );
                Ok(None) // Continue normal quoting until we hit threshold
            }
            RecoveryState::IocRecovery {
                is_buy_side,
                attempts,
                ..
            } => {
                // Check if we should send an IOC
                if let Some((ioc_is_buy, slippage_bps)) =
                    self.infra.recovery_manager.should_send_ioc()
                {
                    // Attempt IOC order to reduce position
                    let position = self.position.position();
                    let min_size = self.infra.recovery_manager.config().min_ioc_size;
                    let reduce_size = position.abs().max(min_size);

                    if reduce_size < min_size {
                        // Position already near zero - recovery successful
                        info!("Position reduced to near zero, recovery complete");
                        self.infra.recovery_manager.record_success();
                        return Ok(Some(RecoveryAction {
                            skip_normal_quoting: false,
                        }));
                    }

                    info!(
                        position = position,
                        reduce_size = reduce_size,
                        is_buy = ioc_is_buy,
                        slippage_bps = slippage_bps,
                        attempt = attempts + 1,
                        "Attempting IOC recovery order"
                    );

                    // Record that we're sending IOC (updates last_attempt time)
                    self.infra.recovery_manager.record_ioc_sent();

                    let result = self
                        .executor
                        .place_ioc_reduce_order(
                            &self.config.asset,
                            reduce_size,
                            ioc_is_buy,
                            slippage_bps,
                            self.latest_mid,
                        )
                        .await;

                    if result.oid > 0 && result.filled {
                        // IOC filled - update position and record fill
                        let fill_size = result.resting_size;
                        self.position.process_fill(fill_size, ioc_is_buy);
                        self.infra.recovery_manager.record_ioc_fill(fill_size);
                        info!(
                            oid = result.oid,
                            filled_size = fill_size,
                            new_position = self.position.position(),
                            "IOC recovery order filled"
                        );
                        self.infra.recovery_manager.record_success();
                    } else {
                        // IOC failed or not filled
                        warn!(
                            oid = result.oid,
                            filled = result.filled,
                            "IOC recovery order not filled"
                        );
                    }
                } else {
                    // Either cooling down between attempts or max attempts reached
                    // The record_rejection() method handles exhaustion
                    debug!(
                        is_buy_side = is_buy_side,
                        attempts = attempts,
                        "IOC recovery: waiting for cooldown or exhausted"
                    );
                }

                Ok(Some(RecoveryAction {
                    skip_normal_quoting: true,
                }))
            }
            RecoveryState::Cooldown { until, reason, .. } => {
                // In cooldown - check if we should exit
                if std::time::Instant::now() >= until {
                    debug!(?reason, "Cooldown complete, resuming normal quoting");
                    self.infra.recovery_manager.reset();
                    Ok(None)
                } else {
                    let remaining = until.saturating_duration_since(std::time::Instant::now());
                    debug!(
                        remaining_secs = %format!("{:.1}", remaining.as_secs_f64()),
                        ?reason,
                        "In cooldown, skipping quote cycle"
                    );
                    Ok(Some(RecoveryAction {
                        skip_normal_quoting: true,
                    }))
                }
            }
        }
    }

    /// Place a new order.
    async fn place_new_order(&mut self, side: Side, quote: &Quote) -> Result<()> {
        let is_buy = side == Side::Buy;

        // === Margin Pre-Flight Check ===
        // Verify order meets margin requirements before placing
        let sizing_result = self.infra.margin_sizer.adjust_size(
            quote.size,
            quote.price,
            self.position.position(),
            is_buy,
        );

        // Skip order if margin check reduces size to zero or margin constrained
        if sizing_result.adjusted_size <= 0.0 {
            debug!(
                side = %side_str(side),
                requested_size = quote.size,
                adjusted_size = sizing_result.adjusted_size,
                reason = ?sizing_result.constraint_reason,
                "Order blocked by margin constraints"
            );
            return Ok(());
        }

        // Use adjusted size from margin check, truncated to sz_decimals
        let adjusted_size =
            truncate_float(sizing_result.adjusted_size, self.config.sz_decimals, false);
        if adjusted_size <= 0.0 {
            debug!(
                side = %side_str(side),
                "Adjusted size truncated to zero, skipping order"
            );
            return Ok(());
        }

        // Post-truncation notional check - truncation can push size below $10 minimum
        let notional = adjusted_size * quote.price;
        if notional < MIN_ORDER_NOTIONAL {
            debug!(
                side = %side_str(side),
                size = %format!("{:.6}", adjusted_size),
                price = %format!("{:.2}", quote.price),
                notional = %format!("{:.2}", notional),
                min = MIN_ORDER_NOTIONAL,
                "Skipping order: post-truncation notional below minimum"
            );
            return Ok(());
        }

        let result = self
            .executor
            .place_order(&self.config.asset, quote.price, adjusted_size, is_buy, None)
            .await;

        if result.oid != 0 {
            info!(
                "{} {} {} resting at {} (margin-adjusted from {})",
                side_str(side),
                result.resting_size,
                self.config.asset,
                quote.price,
                quote.size
            );

            // Add to primary order manager
            self.orders.add_order(TrackedOrder::new(
                result.oid,
                side,
                quote.price,
                result.resting_size,
            ));

            // Also add to WsOrderStateManager for improved state tracking
            self.ws_state.add_order(TrackedOrder::new(
                result.oid,
                side,
                quote.price,
                result.resting_size,
            ));

            // === Tier 1: Register with queue tracker ===
            // Estimate depth ahead conservatively (2x our target liquidity)
            let depth_ahead = self.config.target_liquidity * 2.0;
            self.tier1.queue_tracker.order_placed(
                result.oid,
                quote.price,
                result.resting_size,
                depth_ahead,
                is_buy,
            );
        }

        Ok(())
    }

    /// Cancel an order with retry logic.
    ///
    /// Returns `CancelResult` indicating what happened:
    /// - `Cancelled`: Order was cancelled successfully
    /// - `AlreadyCancelled`: Order was already cancelled
    /// - `AlreadyFilled`: Order was filled before cancel (DO NOT remove from tracking!)
    /// - `Failed`: Cancel failed after all retries
    async fn cancel_with_retry(&self, asset: &str, oid: u64, max_attempts: u32) -> CancelResult {
        for attempt in 0..max_attempts {
            let result = self.executor.cancel_order(asset, oid).await;

            // If order is gone (any reason), stop retrying
            if result.order_is_gone() {
                return result;
            }

            // Only retry on Failed
            if attempt < max_attempts - 1 {
                let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                debug!(
                    "Cancel failed for oid={}, retrying in {:?} (attempt {}/{})",
                    oid,
                    delay,
                    attempt + 1,
                    max_attempts
                );
                tokio::time::sleep(delay).await;
            }
        }
        CancelResult::Failed
    }

    /// Graceful shutdown - cancel all resting orders and log final state.
    ///
    /// Ensures:
    /// 1. All resting orders are cancelled with retry logic
    /// 2. Final position and P&L state is logged
    /// 3. Kill switch status is reported
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("=== GRACEFUL SHUTDOWN INITIATED ===");

        // Log final state before cancelling
        let final_position = self.position.position();
        let final_mid = self.latest_mid;
        let position_value = final_position.abs() * final_mid;
        let kill_switch_summary = self.safety.kill_switch.summary();

        info!(
            position = %format!("{:.6}", final_position),
            mid_price = %format!("{:.2}", final_mid),
            position_value = %format!("${:.2}", position_value),
            "Final position state"
        );

        // Log kill switch status
        if kill_switch_summary.triggered {
            warn!(
                reasons = ?kill_switch_summary.reasons,
                "Kill switch was triggered during session"
            );
        } else {
            info!(
                daily_pnl = %format!("${:.2}", kill_switch_summary.daily_pnl),
                max_drawdown = %format!("{:.1}%", kill_switch_summary.drawdown_pct),
                "Kill switch was NOT triggered"
            );
        }

        // Log Tier 1 module summaries
        let as_summary = self.tier1.adverse_selection.summary();
        info!(
            fills_measured = as_summary.fills_measured,
            realized_as_bps = %format!("{:.4}", as_summary.realized_as_bps),
            spread_adjustment_bps = %format!("{:.4}", as_summary.spread_adjustment_bps),
            "Adverse selection summary"
        );

        let liq_summary = self.tier1.liquidation_detector.summary();
        info!(
            total_liquidations = liq_summary.total_liquidations,
            cascade_severity = %format!("{:.2}", liq_summary.cascade_severity),
            "Liquidation cascade summary"
        );

        // Cancel all resting orders with confirmation
        let oids: Vec<u64> = self.orders.order_ids();
        let total_orders = oids.len();

        if total_orders == 0 {
            info!("No resting orders to cancel");
        } else {
            info!("Bulk cancelling {} resting orders...", total_orders);
            self.initiate_bulk_cancel(oids).await;
            info!(cancelled = total_orders, "Bulk cancel completed");
        }

        info!("=== GRACEFUL SHUTDOWN COMPLETE ===");
        Ok(())
    }

    /// Safety sync - periodically verify local state matches exchange.
    /// This is a fallback mechanism; if working correctly, should find no discrepancies.
    #[tracing::instrument(name = "safety_sync", skip_all, fields(asset = %self.config.asset))]
    async fn safety_sync(&mut self) -> Result<()> {
        debug!("Running safety sync...");

        // === Step 1: Order cleanup ===
        let cleaned = self.orders.cleanup();
        for oid in cleaned {
            self.tier1.queue_tracker.order_removed(oid);
        }

        // === Step 2: Stale pending cleanup ===
        let stale_pending = self
            .orders
            .cleanup_stale_pending(std::time::Duration::from_secs(5));
        if stale_pending > 0 {
            warn!(
                "[SafetySync] Cleaned up {} stale pending orders",
                stale_pending
            );
        }

        // === Step 3: Stuck cancel check ===
        let stuck = self.orders.check_stuck_cancels();
        if !stuck.is_empty() {
            warn!("[SafetySync] Stuck cancel orders detected: {:?}", stuck);
        }

        // === Step 4: Margin refresh ===
        if let Err(e) = self.refresh_margin_state().await {
            warn!("Failed to refresh margin state: {e}");
        }
        self.infra.last_margin_refresh = std::time::Instant::now();

        // === Step 4b: Exchange position limits refresh ===
        if let Err(e) = self.refresh_exchange_limits().await {
            warn!("Failed to refresh exchange limits: {e}");
        }

        // === Step 5: Exchange reconciliation ===
        // Use DEX-aware open_orders for HIP-3 support
        let exchange_orders = self
            .info_client
            .open_orders_for_dex(self.user_address, self.config.dex.as_deref())
            .await?;
        let exchange_oids: HashSet<u64> = exchange_orders
            .iter()
            .filter(|o| o.coin == *self.config.asset)
            .map(|o| o.oid)
            .collect();
        let local_oids: HashSet<u64> = self.orders.order_ids().into_iter().collect();

        // Phase 7: Clean up expired entries in orphan tracker
        self.infra.orphan_tracker.cleanup();

        // Cancel orphan orders (on exchange but not tracked locally)
        // Phase 7: Use orphan tracker with grace period instead of immediate cancellation.
        // This prevents false positives during the window between API response and finalization.
        let candidate_orphans = safety::SafetyAuditor::find_orphans(&exchange_oids, &local_oids);
        let (aged_orphans, new_orphan_count) = self
            .infra
            .orphan_tracker
            .filter_aged_orphans(&candidate_orphans);

        if new_orphan_count > 0 {
            debug!(
                new_orphans = new_orphan_count,
                total_candidates = candidate_orphans.len(),
                aged = aged_orphans.len(),
                "[SafetySync] New potential orphans detected - starting grace period"
            );
        }

        // PERFORMANCE FIX: Use bulk cancel instead of sequential cancellation.
        // With 16+ orphans, sequential cancellation takes 8+ seconds (500ms each).
        // Bulk cancel is a single API call regardless of count.
        if !aged_orphans.is_empty() {
            info!(
                count = aged_orphans.len(),
                "[SafetySync] Bulk cancelling {} orphan orders that aged past grace period",
                aged_orphans.len()
            );

            // Log each OID for debugging
            for &oid in &aged_orphans {
                warn!(
                    "[SafetySync] Orphan order aged past grace period: oid={} - queued for bulk cancel",
                    oid
                );
            }

            // Bulk cancel for efficiency (single API call)
            let cancel_results = self
                .executor
                .cancel_bulk_orders(&self.config.asset, aged_orphans.clone())
                .await;

            // Process results and clear from orphan tracking
            for (oid, result) in aged_orphans.iter().zip(cancel_results.iter()) {
                safety::SafetyAuditor::log_orphan_cancellation(*oid, result.order_is_gone());
                self.infra.orphan_tracker.clear_orphan(*oid);
            }
        }

        // Remove stale local orders (tracked but not on exchange)
        let is_in_cancel_window = |oid: u64| {
            self.orders
                .get_order(oid)
                .map(|o| {
                    matches!(
                        o.state,
                        OrderState::CancelPending | OrderState::CancelConfirmed
                    )
                })
                .unwrap_or(false)
        };
        let stale_local = safety::SafetyAuditor::find_stale_local(
            &exchange_oids,
            &local_oids,
            is_in_cancel_window,
        );
        for oid in stale_local {
            safety::SafetyAuditor::log_stale_removal(oid);
            self.orders.remove_order(oid);
            self.tier1.queue_tracker.order_removed(oid);
        }

        // Log sync status
        let active_local_oids: HashSet<u64> = self
            .orders
            .order_ids()
            .into_iter()
            .filter(|oid| {
                self.orders
                    .get_order(*oid)
                    .map(|o| o.is_active())
                    .unwrap_or(false)
            })
            .collect();
        let is_synced = exchange_oids == active_local_oids;
        safety::SafetyAuditor::log_sync_status(
            exchange_oids.len(),
            active_local_oids.len(),
            is_synced,
        );

        // === Step 5b: Rate limit status logging ===
        // Only log at INFO level if there are warnings, otherwise DEBUG to reduce noise
        let rate_metrics = self.infra.proactive_rate_tracker.get_metrics();
        if rate_metrics.ip_warning || rate_metrics.address_warning {
            warn!(
                ip_weight_used = rate_metrics.ip_weight_used_per_minute,
                ip_weight_limit = rate_metrics.ip_weight_limit,
                address_requests = rate_metrics.address_requests_used,
                address_budget_remaining = rate_metrics.address_budget_remaining,
                volume_traded = %format!("{:.2}", rate_metrics.usd_volume_traded),
                ip_warning = rate_metrics.ip_warning,
                address_warning = rate_metrics.address_warning,
                "[SafetySync] Rate limit WARNING"
            );
        } else {
            debug!(
                ip_weight_used = rate_metrics.ip_weight_used_per_minute,
                address_requests = rate_metrics.address_requests_used,
                "[SafetySync] Rate limit OK"
            );
        }

        // === Step 6: Dynamic limit update ===
        let account_value = self.infra.margin_sizer.state().account_value;
        let sigma = self.estimator.sigma_clean();
        let sigma_confidence = self.estimator.sigma_confidence();
        let time_horizon = (1.0 / self.estimator.arrival_intensity()).min(120.0);

        if account_value > 0.0 {
            let new_limit = calculate_dynamic_max_position_value(
                account_value,
                sigma,
                time_horizon,
                sigma_confidence,
                &self.stochastic.dynamic_risk_config,
            );
            self.safety.kill_switch.update_dynamic_limit(new_limit);
            safety::SafetyAuditor::log_dynamic_limit(
                new_limit,
                account_value,
                sigma,
                sigma_confidence,
                time_horizon,
            );
        }

        // === Step 7: Reduce-only status ===
        // HARMONIZED: Use effective_max_position for reduce-only check
        let position = self.position.position();
        let position_value = position.abs() * self.latest_mid;
        let max_position_value = self.safety.kill_switch.max_position_value();

        let (reduce_only, reason) = safety::SafetyAuditor::check_reduce_only(
            position,
            position_value,
            self.effective_max_position, // First-principles limit
            max_position_value,
        );
        safety::SafetyAuditor::log_reduce_only_status(reduce_only, reason.as_deref());

        // Phase 4: Record sync completed for reconciler
        self.infra.reconciler.record_sync_completed();

        Ok(())
    }

    /// Refresh margin state from exchange.
    ///
    /// Fetches user account state and updates the MarginAwareSizer with current values.
    /// For HIP-3 DEXs, collateral is in spot balance. For validator perps, in clearinghouse.
    /// Also extracts liquidation price for dynamic reduce-only mode.
    async fn refresh_margin_state(&mut self) -> Result<()> {
        // For HIP-3 DEXs, collateral is in spot balance (USDE, USDH, etc.)
        // For validator perps, collateral is in perps clearinghouse (USDC)
        let (account_value, margin_used, total_notional, liquidation_price) =
            if self.config.dex.is_some() {
                // HIP-3: Get account value from spot balance
                let balances = self
                    .info_client
                    .user_token_balances(self.user_address)
                    .await?;
                let account_value = self
                    .config
                    .collateral
                    .available_balance_from_spot(&balances.balances)
                    .unwrap_or(0.0);

                // Get margin used from DEX-specific clearinghouse
                let user_state = self
                    .info_client
                    .user_state_for_dex(self.user_address, self.config.dex.as_deref())
                    .await?;
                let margin_used: f64 = user_state
                    .margin_summary
                    .total_margin_used
                    .parse()
                    .unwrap_or(0.0);
                let total_notional: f64 = user_state
                    .margin_summary
                    .total_ntl_pos
                    .parse()
                    .unwrap_or(0.0);

                // Extract liquidation price for our asset from the position data
                let liquidation_price = user_state
                    .asset_positions
                    .iter()
                    .find(|p| p.position.coin == *self.config.asset)
                    .and_then(|p| p.position.liquidation_px.as_ref())
                    .and_then(|px| px.parse::<f64>().ok());

                (
                    account_value,
                    margin_used,
                    total_notional,
                    liquidation_price,
                )
            } else {
                // Validator perps: Get all values from perps clearinghouse
                let user_state = self.info_client.user_state(self.user_address).await?;
                let account_value: f64 = user_state
                    .cross_margin_summary
                    .account_value
                    .parse()
                    .unwrap_or(0.0);
                let margin_used: f64 = user_state
                    .cross_margin_summary
                    .total_margin_used
                    .parse()
                    .unwrap_or(0.0);
                let total_notional: f64 = user_state
                    .cross_margin_summary
                    .total_ntl_pos
                    .parse()
                    .unwrap_or(0.0);

                // Extract liquidation price for our asset from the position data
                let liquidation_price = user_state
                    .asset_positions
                    .iter()
                    .find(|p| p.position.coin == *self.config.asset)
                    .and_then(|p| p.position.liquidation_px.as_ref())
                    .and_then(|px| px.parse::<f64>().ok());

                (
                    account_value,
                    margin_used,
                    total_notional,
                    liquidation_price,
                )
            };

        // Get current position and price for liquidation proximity calculation
        let position_size = self.position.position();
        let mark_price = self.latest_mid;

        // Update the margin sizer with liquidation proximity data
        self.infra.margin_sizer.update_state_with_liquidation(
            account_value,
            margin_used,
            total_notional,
            liquidation_price,
            mark_price,
            position_size,
        );

        // Log liquidation proximity for monitoring
        let margin_state = self.infra.margin_sizer.state();
        if let Some(liq_px) = liquidation_price {
            let distance_pct = if mark_price > 0.0 {
                ((mark_price - liq_px).abs() / mark_price) * 100.0
            } else {
                0.0
            };
            debug!(
                account_value = %format!("{:.2}", account_value),
                margin_used = %format!("{:.2}", margin_used),
                liquidation_px = %format!("{:.4}", liq_px),
                mark_price = %format!("{:.4}", mark_price),
                distance_pct = %format!("{:.2}%", distance_pct),
                buffer_ratio = ?margin_state.liquidation_buffer_ratio().map(|r| format!("{:.2}", r)),
                "Margin state refreshed with liquidation proximity"
            );
        } else {
            debug!(
                account_value = %format!("{:.2}", account_value),
                margin_used = %format!("{:.2}", margin_used),
                total_notional = %format!("{:.2}", total_notional),
                available = %format!("{:.2}", account_value - margin_used),
                "Margin state refreshed (no position for liquidation tracking)"
            );
        }

        Ok(())
    }

    /// Refresh exchange position limits from API.
    ///
    /// Fetches `active_asset_data` to get exchange-enforced position limits.
    /// This prevents order rejections due to position limit violations.
    async fn refresh_exchange_limits(&mut self) -> Result<()> {
        let asset_data = self
            .info_client
            .active_asset_data(self.user_address, self.config.asset.to_string())
            .await?;

        // Update exchange limits with effective_max_position (margin-based) for proper capacity
        // Fall back to config.max_position if effective hasn't been computed yet (early startup)
        let local_max = if self.effective_max_position > 0.0 {
            self.effective_max_position
        } else {
            self.config.max_position
        };
        self.infra
            .exchange_limits
            .update_from_response(&asset_data, local_max);

        // Log summary for diagnostics
        let summary = self.infra.exchange_limits.summary();
        debug!(
            max_long = %format!("{:.6}", summary.max_long),
            max_short = %format!("{:.6}", summary.max_short),
            available_buy = %format!("{:.6}", summary.available_buy),
            available_sell = %format!("{:.6}", summary.available_sell),
            effective_bid = %format!("{:.6}", summary.effective_bid_limit),
            effective_ask = %format!("{:.6}", summary.effective_ask_limit),
            "Exchange position limits refreshed"
        );

        // Update Prometheus metrics
        self.infra.prometheus.update_exchange_limits(
            summary.max_long,
            summary.max_short,
            summary.available_buy,
            summary.available_sell,
            summary.effective_bid_limit,
            summary.effective_ask_limit,
            summary.age_ms,
            true, // valid
        );

        // Log capacity info for monitoring.
        //
        // IMPORTANT: `available_to_trade` from Hyperliquid represents capacity to
        // INCREASE exposure (open new positions), NOT capacity to close positions.
        // Closing a position (e.g., selling a long) RELEASES margin rather than
        // consuming it, so these limits don't apply to position closing.
        //
        // For market making, we care about:
        // - Long position: available_buy = capacity to add more long
        // - Short position: available_sell = capacity to add more short
        //
        // We only warn when the capacity to increase position on the side we're
        // already exposed to is low, which could limit our ability to scale up.
        let position = self.position.position();

        // Warn if capacity to increase existing exposure is very low
        // (less than 10% of max position)
        let low_capacity_threshold = self.config.max_position * 0.1;

        if position > 0.0 && summary.available_buy < low_capacity_threshold {
            debug!(
                available_buy = %format!("{:.6}", summary.available_buy),
                position = %format!("{:.6}", position),
                threshold = %format!("{:.6}", low_capacity_threshold),
                "Limited capacity to increase long exposure"
            );
        }
        if position < 0.0 && summary.available_sell < low_capacity_threshold {
            debug!(
                available_sell = %format!("{:.6}", summary.available_sell),
                position = %format!("{:.6}", position),
                threshold = %format!("{:.6}", low_capacity_threshold),
                "Limited capacity to increase short exposure"
            );
        }

        Ok(())
    }

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

    /// Check kill switch conditions and update state.
    /// Called after each message and periodically.
    fn check_kill_switch(&self) {
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
        } else if aggregated.max_severity >= risk::RiskSeverity::High {
            warn!(
                summary = %aggregated.summary(),
                "High risk detected by RiskAggregator"
            );
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

fn side_str(side: Side) -> &'static str {
    match side {
        Side::Buy => "BUY",
        Side::Sell => "SELL",
    }
}

/// Partition LadderActions into separate collections by action type.
///
/// Returns (cancels, modifies, places) where:
/// - cancels: Vec<u64> of order IDs to cancel
/// - modifies: Vec<ModifySpec> of orders to modify
/// - places: Vec<(f64, f64)> of (price, size) for new orders
fn partition_ladder_actions(
    actions: &[tracking::LadderAction],
    side: Side,
) -> (Vec<u64>, Vec<ModifySpec>, Vec<(f64, f64)>) {
    use tracking::LadderAction;

    let mut cancels = Vec::new();
    let mut modifies = Vec::new();
    let mut places = Vec::new();
    let is_buy = side == Side::Buy;

    for action in actions {
        match action {
            LadderAction::Cancel { oid } => {
                cancels.push(*oid);
            }
            LadderAction::Modify {
                oid,
                new_price,
                new_size,
                ..
            } => {
                modifies.push(ModifySpec {
                    oid: *oid,
                    new_price: *new_price,
                    new_size: *new_size,
                    is_buy,
                });
            }
            LadderAction::Place { price, size, .. } => {
                places.push((*price, *size));
            }
        }
    }

    (cancels, modifies, places)
}
