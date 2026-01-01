//! Modular market maker implementation.
//!
//! This module provides a market making system with pluggable components:
//! - **Strategy**: Determines quote prices and sizes (symmetric, inventory-aware, etc.)
//! - **OrderManager**: Tracks resting orders
//! - **PositionTracker**: Tracks position and deduplicates fills
//! - **Executor**: Handles order placement and cancellation

mod adverse_selection;
mod config;
pub mod core;
mod estimator;
pub mod events;
pub mod fills;
pub mod infra;
pub mod messages;
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

use std::collections::HashSet;
use std::time::Duration;

use alloy::primitives::Address;
use tokio::sync::mpsc::unbounded_channel;
use tracing::{debug, error, info, warn};

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

        Self {
            config,
            strategy,
            executor,
            orders: OrderManager::new(),
            position: PositionTracker::new(initial_position),
            info_client,
            user_address,
            latest_mid: -1.0,
            estimator: ParameterEstimator::new(estimator_config),
            last_warmup_log: 0,
            tier1,
            tier2,
            safety,
            infra,
            stochastic,
            effective_max_position,
        }
    }

    /// Set the dynamic risk configuration.
    pub fn with_dynamic_risk_config(mut self, config: DynamicRiskConfig) -> Self {
        self.stochastic.dynamic_risk_config = config;
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

        // Track any remaining orders (should be empty after cancel-all, but be safe)
        let open_orders = self.info_client.open_orders(self.user_address).await?;

        for order in open_orders.iter().filter(|o| o.coin == self.config.asset) {
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
        let user_state = self.info_client.user_state(self.user_address).await?;

        // Find position for our asset
        let exchange_position = user_state
            .asset_positions
            .iter()
            .find(|p| p.position.coin == self.config.asset)
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

        Ok(())
    }

    /// Cancel ALL orders for our asset on startup.
    /// Uses retry loop to ensure all orders are cancelled.
    async fn cancel_all_orders_on_startup(&mut self) -> Result<()> {
        const MAX_RETRIES: usize = 5;
        const RETRY_DELAY_MS: u64 = 500;

        for attempt in 0..MAX_RETRIES {
            let open_orders = self.info_client.open_orders(self.user_address).await?;
            let our_orders: Vec<_> = open_orders
                .iter()
                .filter(|o| o.coin == self.config.asset)
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
        let remaining = self
            .info_client
            .open_orders(self.user_address)
            .await?
            .iter()
            .filter(|o| o.coin == self.config.asset)
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
        let (sender, mut receiver) = unbounded_channel();

        // Subscribe to UserFills for fill detection
        self.info_client
            .subscribe(
                Subscription::UserFills {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to AllMids for price updates
        self.info_client
            .subscribe(Subscription::AllMids, sender.clone())
            .await?;

        // Subscribe to Trades for volatility and arrival intensity estimation
        self.info_client
            .subscribe(
                Subscription::Trades {
                    coin: self.config.asset.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to L2Book for kappa (order book depth decay) estimation
        self.info_client
            .subscribe(
                Subscription::L2Book {
                    coin: self.config.asset.clone(),
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

        // Create shutdown signal listener ONCE before the loop (not inside select!)
        // This ensures the signal is captured even if pressed during message processing
        let mut shutdown_signal = std::pin::pin!(tokio::signal::ctrl_c());

        loop {
            // === Kill Switch Check (before processing any message) ===
            if self.safety.kill_switch.is_triggered() {
                let reasons = self.safety.kill_switch.trigger_reasons();
                error!(
                    "KILL SWITCH TRIGGERED: {:?}",
                    reasons.iter().map(|r| r.to_string()).collect::<Vec<_>>()
                );
                break;
            }

            tokio::select! {
                message = receiver.recv() => {
                    match message {
                        Some(msg) => {
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

                    // === Update P&L inventory snapshot for carry calculation ===
                    if self.latest_mid > 0.0 {
                        self.tier2.pnl_tracker.record_inventory_snapshot(self.latest_mid);
                    }

                    // Log Prometheus output (for scraping or debugging)
                    debug!(
                        prometheus_output = %self.infra.prometheus.to_prometheus_text(&self.config.asset),
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
                _ = &mut shutdown_signal => {
                    info!("Shutdown signal received");
                    break;
                }
            }
        }

        // Graceful shutdown
        self.shutdown().await?;

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
        );

        let mut state = messages::AllMidsState {
            estimator: &mut self.estimator,
            connection_health: &mut self.infra.connection_health,
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
        // HARMONIZED: Use effective_max_position for position utilization calculations
        let ctx = messages::MessageContext::new(
            self.config.asset.clone(),
            self.latest_mid,
            self.position.position(),
            self.effective_max_position, // First-principles limit
            self.estimator.is_warmed_up(),
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
            self.strategy.record_fill_observation(obs.depth_bps, obs.filled);
        }
        if !result.fill_observations.is_empty() {
            debug!(
                observations = result.fill_observations.len(),
                "Recorded fill observations for Bayesian learning"
            );
        }

        // Record fill volume for rate limit budget calculation
        if result.total_volume_usd > 0.0 {
            self.infra
                .proactive_rate_tracker
                .record_fill_volume(result.total_volume_usd);
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
        );

        let mut state = messages::L2BookState {
            estimator: &mut self.estimator,
            queue_tracker: &mut self.tier1.queue_tracker,
            spread_tracker: &mut self.tier2.spread_tracker,
            data_quality: &mut self.infra.data_quality,
            prometheus: &mut self.infra.prometheus,
        };

        let _result = messages::process_l2_book(&l2_book, &ctx, &mut state)?;
        Ok(())
    }

    /// Update quotes based on current market state.
    #[tracing::instrument(name = "quote_cycle", skip_all, fields(asset = %self.config.asset))]
    async fn update_quotes(&mut self) -> Result<()> {
        // Don't place orders until estimator is warmed up
        if !self.estimator.is_warmed_up() {
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
        let dynamic_limit_valid = self.infra.margin_sizer.state().account_value > 0.0;

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
        };
        let market_params = ParameterAggregator::build(&sources);

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

        debug!(
            mid = self.latest_mid,
            microprice = %format!("{:.4}", market_params.microprice),
            position = self.position.position(),
            static_max_pos = self.config.max_position,
            dynamic_max_pos = %format!("{:.6}", market_params.dynamic_max_position),
            effective_max_pos = %format!("{:.6}", self.effective_max_position),
            dynamic_valid = market_params.dynamic_limit_valid,
            target_liq = self.config.target_liquidity,
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
        self.strategy.update_fill_model_params(
            market_params.sigma,
            market_params.kelly_time_horizon,
        );

        // Try multi-level ladder quoting first
        let ladder = self.strategy.calculate_ladder(
            &quote_config,
            self.position.position(),
            self.config.max_position,
            self.config.target_liquidity,
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

            // Reduce-only mode: when over max position OR position value, only allow quotes that reduce position
            // Phase 3: Use exchange-aware reduce-only that checks exchange limits and signals escalation
            // HARMONIZED: Use effective_max_position (first-principles derived) instead of static config
            let reduce_only_config = quoting::ReduceOnlyConfig {
                position: self.position.position(),
                max_position: self.effective_max_position, // First-principles limit
                mid_price: self.latest_mid,
                max_position_value: self.safety.kill_switch.max_position_value(),
                asset: self.config.asset.clone(),
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
                debug!(
                    "Reduce-only mode activated with potential escalation"
                );
            }

            debug!(
                bid_levels = bid_quotes.len(),
                ask_levels = ask_quotes.len(),
                best_bid = ?bid_quotes.first().map(|q| (q.price, q.size)),
                best_ask = ?ask_quotes.first().map(|q| (q.price, q.size)),
                "Calculated ladder quotes"
            );

            // Reconcile ladder quotes
            self.reconcile_ladder_side(Side::Buy, bid_quotes).await?;
            self.reconcile_ladder_side(Side::Sell, ask_quotes).await?;
        } else {
            // Fallback to single-quote mode for non-ladder strategies
            let (mut bid, mut ask) = self.strategy.calculate_quotes(
                &quote_config,
                self.position.position(),
                self.config.max_position,
                self.config.target_liquidity,
                &market_params,
            );

            // Reduce-only mode: when over max position OR position value, only allow quotes that reduce position
            // HARMONIZED: Use effective_max_position (first-principles derived) instead of static config
            let reduce_only_config = quoting::ReduceOnlyConfig {
                position: self.position.position(),
                max_position: self.effective_max_position, // First-principles limit
                mid_price: self.latest_mid,
                max_position_value: self.safety.kill_switch.max_position_value(),
                asset: self.config.asset.clone(),
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

        info!("Bulk cancelling {} orders", cancellable_oids.len());

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
                        let depth_bps = ((price - self.latest_mid).abs() / self.latest_mid) * 10_000.0;
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
            for spec in &order_specs {
                if let Some(ref cloid) = spec.cloid {
                    self.orders
                        .add_pending_with_cloid(side, spec.price, spec.size, cloid.clone());
                } else {
                    // Fallback (shouldn't happen with new code)
                    self.orders.add_pending(side, spec.price, spec.size);
                }
            }

            // Place all orders in a single API call
            let num_orders = order_specs.len() as u32;
            let results = self
                .executor
                .place_bulk_orders(&self.config.asset, order_specs.clone())
                .await;

            // Record API call for rate limit tracking
            // Bulk order: 1 IP weight, n address requests
            self.infra
                .proactive_rate_tracker
                .record_call(1, num_orders);

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
                }

                if result.oid > 0 {
                    let cloid = spec.cloid.as_ref().or(result.cloid.as_ref());

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
                        } else {
                            self.orders.remove_pending(side, spec.price);
                        }

                        // Record fill to safety components for analytics
                        self.safety
                            .fill_processor
                            .pre_register_immediate_fill(result.oid);

                        info!(
                            oid = result.oid,
                            cloid = ?cloid,
                            price = spec.price,
                            size = spec.size,
                            actual_fill = actual_fill,
                            position = self.position.position(),
                            "Order filled immediately - position updated, tracking for WS dedup"
                        );

                        // If partially filled, still need to track the resting portion
                        if filled_size > EPSILON {
                            // Partial fill - some size still resting
                            debug!(
                                oid = result.oid,
                                resting_size = filled_size,
                                "Partial immediate fill, {} still resting",
                                filled_size
                            );
                            // Order already added above with FilledImmediately state
                            // It will transition to PartialFilled when WS confirms
                        }

                        continue;
                    }

                    // Move from pending to tracked with real OID using CLOID (Phase 1 Fix)
                    if let Some(c) = cloid {
                        self.orders
                            .finalize_pending_by_cloid(c, result.oid, result.resting_size);
                    } else {
                        self.orders
                            .finalize_pending(side, spec.price, result.oid, result.resting_size);
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
            RecoveryState::ReduceOnlyStuck { is_buy_side, consecutive_rejections, .. } => {
                // Already in stuck state - waiting for rejections to hit threshold
                // The record_rejection() method will transition to IocRecovery when threshold is hit
                debug!(
                    is_buy_side = is_buy_side,
                    consecutive_rejections = consecutive_rejections,
                    threshold = self.infra.recovery_manager.config().rejection_threshold,
                    "In reduce-only-stuck state, waiting for threshold"
                );
                Ok(None)  // Continue normal quoting until we hit threshold
            }
            RecoveryState::IocRecovery { is_buy_side, attempts, .. } => {
                // Check if we should send an IOC
                if let Some((ioc_is_buy, slippage_bps)) = self.infra.recovery_manager.should_send_ioc() {
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
        let result = self
            .executor
            .place_order(&self.config.asset, quote.price, adjusted_size, is_buy)
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

            self.orders.add_order(TrackedOrder::new(
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
            info!("Cancelling {} resting orders...", total_orders);

            let mut cancelled = 0;
            let mut failed = 0;

            for oid in oids {
                let cancel_result = self.cancel_with_retry(&self.config.asset, oid, 3).await;
                if cancel_result.order_is_gone() {
                    // Order is gone (cancelled, already cancelled, or already filled)
                    if cancel_result == CancelResult::AlreadyFilled {
                        info!("Order already filled during shutdown: oid={}", oid);
                    } else {
                        info!("Cancelled order: oid={}", oid);
                    }
                    cancelled += 1;
                } else {
                    warn!("Failed to cancel order after retries: oid={}", oid);
                    failed += 1;
                }
            }

            if failed > 0 {
                warn!(
                    cancelled = cancelled,
                    failed = failed,
                    "Some orders failed to cancel - may need manual cleanup"
                );
            } else {
                info!(cancelled = cancelled, "All orders cancelled successfully");
            }
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
        let exchange_orders = self.info_client.open_orders(self.user_address).await?;
        let exchange_oids: HashSet<u64> = exchange_orders
            .iter()
            .filter(|o| o.coin == self.config.asset)
            .map(|o| o.oid)
            .collect();
        let local_oids: HashSet<u64> = self.orders.order_ids().into_iter().collect();

        // Cancel orphan orders (on exchange but not tracked locally)
        let orphans = safety::SafetyAuditor::find_orphans(&exchange_oids, &local_oids);
        for oid in orphans {
            warn!(
                "[SafetySync] Orphan order detected: oid={} - cancelling",
                oid
            );
            let cancel_result = self.executor.cancel_order(&self.config.asset, oid).await;
            safety::SafetyAuditor::log_orphan_cancellation(oid, cancel_result.order_is_gone());
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
        let rate_metrics = self.infra.proactive_rate_tracker.get_metrics();
        info!(
            ip_weight_used = rate_metrics.ip_weight_used_per_minute,
            ip_weight_limit = rate_metrics.ip_weight_limit,
            address_requests = rate_metrics.address_requests_used,
            address_budget_remaining = rate_metrics.address_budget_remaining,
            volume_traded = %format!("{:.2}", rate_metrics.usd_volume_traded),
            ip_warning = rate_metrics.ip_warning,
            address_warning = rate_metrics.address_warning,
            "[SafetySync] Rate limit status"
        );

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
    async fn refresh_margin_state(&mut self) -> Result<()> {
        let user_state = self.info_client.user_state(self.user_address).await?;

        // Parse margin values from the response
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

        // Update the margin sizer with fresh values
        self.infra
            .margin_sizer
            .update_state(account_value, margin_used, total_notional);

        debug!(
            account_value = %format!("{:.2}", account_value),
            margin_used = %format!("{:.2}", margin_used),
            total_notional = %format!("{:.2}", total_notional),
            available = %format!("{:.2}", account_value - margin_used),
            "Margin state refreshed"
        );

        Ok(())
    }

    /// Refresh exchange position limits from API.
    ///
    /// Fetches `active_asset_data` to get exchange-enforced position limits.
    /// This prevents order rejections due to position limit violations.
    async fn refresh_exchange_limits(&mut self) -> Result<()> {
        let asset_data = self
            .info_client
            .active_asset_data(self.user_address, self.config.asset.clone())
            .await?;

        // Update exchange limits with local max_position for effective limit calculation
        self.infra
            .exchange_limits
            .update_from_response(&asset_data, self.config.max_position);

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

        // Warn if available capacity is low
        // HARMONIZED: Use effective_max_position for capacity threshold
        let position = self.position.position();
        if position > 0.0 && summary.available_sell < self.effective_max_position * 0.1 {
            warn!(
                available_sell = %format!("{:.6}", summary.available_sell),
                position = %format!("{:.6}", position),
                effective_max = %format!("{:.6}", self.effective_max_position),
                "Low sell capacity - near exchange position limit"
            );
        }
        if position < 0.0 && summary.available_buy < self.effective_max_position * 0.1 {
            warn!(
                available_buy = %format!("{:.6}", summary.available_buy),
                position = %format!("{:.6}", position),
                effective_max = %format!("{:.6}", self.effective_max_position),
                "Low buy capacity - near exchange position limit"
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
        let state = KillSwitchState {
            daily_pnl: pnl_summary.total_pnl, // Using total as daily for now
            peak_pnl: pnl_summary.total_pnl,  // Simplified (actual peak needs tracking)
            position: self.position.position(),
            mid_price: self.latest_mid,
            last_data_time,
            rate_limit_errors: 0, // TODO: Track rate limit errors
            cascade_severity: self.tier1.liquidation_detector.cascade_severity(),
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
