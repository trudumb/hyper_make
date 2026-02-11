//! Main event loop and startup synchronization for the market maker.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc::unbounded_channel;
use tracing::{debug, error, info, warn};

use crate::prelude::Result;
use crate::{Message, Subscription};

use super::super::{
    CancelResult, ConnectionState, MarketMaker, TradingEnvironment, QuotingStrategy, Side, TrackedOrder,
};

impl<S: QuotingStrategy, Env: TradingEnvironment> MarketMaker<S, Env> {
    /// Sync open orders from exchange and initialize state.
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
        // Check if WebSocket data is fresh - skip REST if so
        if let Some(last_ws_time) = self.last_web_data2_time {
            let ws_age = last_ws_time.elapsed();
            if ws_age < std::time::Duration::from_secs(30) {
                debug!(
                    ws_age_secs = ws_age.as_secs(),
                    "[PositionSync] Skipping REST - WebSocket data is fresh"
                );
                return Ok(());
            } else {
                warn!(
                    ws_age_secs = ws_age.as_secs(),
                    "[PositionSync] WebSocket data is stale, falling back to REST"
                );
            }
        }

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

        // Record initial position for kill switch runaway exemption
        self.safety
            .kill_switch
            .set_initial_position(self.position.position());

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
                .environment
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

        // Subscribe to OpenOrders for initial state sync and periodic snapshots
        self.info_client
            .subscribe(
                Subscription::OpenOrders {
                    user: self.user_address,
                    dex: self.config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to ActiveAssetData for real-time exchange limit updates
        self.info_client
            .subscribe(
                Subscription::ActiveAssetData {
                    user: self.user_address,
                    coin: self.config.asset.to_string(),
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to WebData2 for real-time margin/position updates (Phase 2)
        self.info_client
            .subscribe(
                Subscription::WebData2 {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        // Subscribe to UserNonFundingLedgerUpdates for spot balance deltas (Phase 3)
        self.info_client
            .subscribe(
                Subscription::UserNonFundingLedgerUpdates {
                    user: self.user_address,
                },
                sender.clone(),
            )
            .await?;

        drop(sender); // Explicitly drop the sender since we're done with subscriptions

        // Phase 3 Init: Populate spot balance cache via REST once
        match self
            .info_client
            .user_token_balances(self.user_address)
            .await
        {
            Ok(balances) => {
                for balance in balances.balances {
                    let total = balance.total.parse::<f64>().unwrap_or(0.0);
                    self.spot_balance_cache.insert(balance.coin, total);
                }
                info!(
                    "Initialized spot balance cache with {} assets",
                    self.spot_balance_cache.len()
                );
            }
            Err(e) => {
                error!("Failed to fetch initial spot balances: {}. Spot margin tracking may be incorrect until next update.", e);
            }
        }

        info!(
            "Market maker started for {} with {} strategy",
            self.config.asset,
            self.strategy.name()
        );
        info!("Warming up parameter estimator...");

        // === Start HJB session (stochastic module integration) ===
        self.stochastic.hjb_controller.start_session();
        debug!("HJB inventory controller session started");

        // Safety sync interval (5 seconds) - catch orphan orders and state divergence more quickly
        // FIX: Reduced from 15s to 5s to reduce state sync lag
        // 15s interval was causing SafetySync snapshot to be 1-5s stale
        let mut sync_interval = tokio::time::interval(Duration::from_secs(5));
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
                let reason_strs: Vec<String> = reasons.iter().map(|r| r.to_string()).collect();
                error!("KILL SWITCH TRIGGERED: {:?}", reason_strs);

                // Write post-mortem dump for incident analysis
                let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);
                let trigger_str = reason_strs.first().cloned().unwrap_or_default();
                let mut dump = crate::market_maker::monitoring::PostMortemDump::new(
                    trigger_str,
                    pnl_summary.total_pnl,
                    self.safety.kill_switch.config().max_daily_loss,
                );
                dump.position = self.position.position();
                dump.daily_pnl = pnl_summary.total_pnl;
                dump.realized_pnl = pnl_summary.realized_pnl;
                dump.unrealized_pnl = pnl_summary.unrealized_pnl;
                dump.mid_price = self.latest_mid;
                dump.cascade_severity = self.tier1.liquidation_detector.cascade_severity();
                dump.risk_state = crate::market_maker::monitoring::RiskSnapshot {
                    drawdown_pct: if pnl_summary.peak_pnl > 0.0 {
                        (pnl_summary.peak_pnl - pnl_summary.total_pnl) / pnl_summary.peak_pnl * 100.0
                    } else {
                        0.0
                    },
                    margin_utilization_pct: 0.0,
                    position_utilization_pct: if self.effective_max_position > 0.0 {
                        self.position.position().abs() / self.effective_max_position * 100.0
                    } else {
                        0.0
                    },
                    rate_limit_errors: self.safety.kill_switch.state().rate_limit_errors,
                    kill_switch_reasons: reason_strs,
                };

                let postmortem_dir = std::path::Path::new("logs/postmortem");
                match dump.write_to_dir(postmortem_dir) {
                    Ok(path) => error!("Post-mortem dump written to {path:?}"),
                    Err(e) => error!("Failed to write post-mortem dump: {e}"),
                }

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

                            // === Phase 3: Event-Driven Quote Updates (Churn Reduction) ===
                            // Check if accumulated events should trigger a quote update.
                            // This replaces the previous timed polling approach with
                            // event-driven updates that only reconcile when meaningful changes occur.
                            if let Err(e) = self.check_event_accumulator().await {
                                warn!("Event accumulator quote update failed: {e}");
                            }
                        }
                        None => {
                            warn!("Message channel closed, stopping market maker");
                            break;
                        }
                    }
                }

                // Binance feed (optional) - price updates and trades
                // When enabled, updates SignalIntegrator for cross-exchange skew
                // and cross-venue flow analysis
                Some(update) = async {
                    match self.binance_receiver.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    self.handle_binance_update(update);
                }

                // Periodic safety sync
                _ = sync_interval.tick() => {
                    if let Err(e) = self.safety_sync().await {
                        warn!("Safety sync failed: {e}");
                    }

                    // === Update Prometheus metrics ===
                    let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);

                    // Update trend tracker with unrealized P&L for underwater detection
                    self.estimator.update_trend_pnl(pnl_summary.unrealized_pnl);

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

                    // Update learned parameters metrics for Prometheus
                    let learned_status = self.stochastic.learned_params.calibration_status();
                    self.infra.prometheus.update_learned_params(
                        self.stochastic.learned_params.alpha_touch.estimate(),
                        self.stochastic.learned_params.kappa.estimate(),
                        self.stochastic.learned_params.spread_floor_bps.estimate(),
                        self.stochastic.learned_params.total_fills_observed,
                        learned_status.tier1_ready,
                    );

                    // === Periodic Component Update (Phase 0: Calibration Logging) ===
                    // Updates model calibration metrics and logs Brier scores / IR
                    self.periodic_component_update();

                    // === Periodic Checkpoint Save (every 5 minutes) ===
                    if self.last_checkpoint_save.elapsed() >= Duration::from_secs(300) {
                        if let Some(ref manager) = self.checkpoint_manager {
                            let mut bundle = self.assemble_checkpoint_bundle();
                            // Stamp readiness assessment on every periodic save
                            let gate = crate::market_maker::calibration::gate::CalibrationGate::new(
                                crate::market_maker::calibration::gate::CalibrationGateConfig::default(),
                            );
                            bundle.readiness = gate.assess(&bundle);
                            if let Err(e) = manager.save_all(&bundle) {
                                warn!("Checkpoint save failed: {e}");
                            }
                            self.last_checkpoint_save = std::time::Instant::now();
                        }
                    }

                    // Check if supervisor recommends reconnection and act on it
                    if self.infra.connection_supervisor.is_reconnect_recommended() {
                        let attempt = self.infra.connection_health.current_attempt() + 1;
                        warn!(
                            time_since_market_data_secs = supervisor_stats.time_since_market_data.as_secs_f64(),
                            connection_state = %supervisor_stats.connection_state,
                            reconnection_attempt = attempt,
                            "Connection supervisor recommends reconnection - initiating"
                        );

                        // Mark that we're starting reconnection (increments attempt counter)
                        self.infra.connection_supervisor.record_reconnection_start();

                        // Force reconnect the WebSocket
                        if let Err(e) = self.info_client.reconnect().await {
                            warn!(
                                error = %e,
                                attempt = attempt,
                                "Failed to initiate WebSocket reconnection"
                            );
                            // Track the failure - this will eventually set connection_failed=true
                            // after max_consecutive_failures is reached
                            if !self.infra.connection_supervisor.record_reconnection_failed() {
                                error!(
                                    "Reconnection permanently failed after max retries - kill switch will trigger"
                                );
                            }
                        } else {
                            info!(
                                attempt = attempt,
                                "WebSocket reconnection initiated - waiting for data to resume"
                            );
                            // Note: We do NOT call reconnection_success() here because the
                            // reconnect is async. Success is detected when record_market_data()
                            // is called and sees we were in Reconnecting state.
                        }

                        // Clear the recommendation (the supervisor will set it again if still stale)
                        self.infra.connection_supervisor.clear_reconnect_recommendation();
                    }

                    // === Update P&L inventory snapshot for carry calculation ===
                    if self.latest_mid > 0.0 {
                        self.tier2.pnl_tracker.record_inventory_snapshot(self.latest_mid);
                    }

                    // === Update Learned Parameters from Adverse Selection ===
                    // Transfer informed/uninformed fill classifications from AS estimator
                    // to the Bayesian learned parameters for alpha_touch estimation.
                    if self.stochastic.stochastic_config.use_learned_parameters {
                        let (informed, uninformed) = self.tier1.adverse_selection.take_informed_counts();
                        if informed > 0 || uninformed > 0 {
                            // Batch update the Bayesian alpha_touch parameter
                            self.stochastic.learned_params.alpha_touch.observe_beta(informed, uninformed);

                            // Log the update for monitoring
                            let alpha = self.stochastic.learned_params.alpha_touch.estimate();
                            let n = self.stochastic.learned_params.alpha_touch.n_observations;
                            debug!(
                                informed = informed,
                                uninformed = uninformed,
                                total_obs = n,
                                alpha_touch = %format!("{:.4}", alpha),
                                "Updated learned alpha_touch from AS classifier"
                            );
                        }

                        // === Update Learned Kappa from Fill Rate ===
                        // Every sync interval, update kappa using observed fills and spreads.
                        // Kappa = fill_rate / spread, so we need: fills, time, average spread.
                        let total_fills = self.tier2.pnl_tracker.fill_count();
                        let session_duration_s = self.session_start_time.elapsed().as_secs_f64();
                        let avg_spread_bps = self.tier2.spread_tracker.current_spread_bps();

                        // Only update if we have meaningful data (at least 10 fills, 60 seconds)
                        if total_fills >= 10 && session_duration_s > 60.0 && avg_spread_bps > 0.0 {
                            // kappa = (fills / time) / spread
                            // We pass raw values and let the method handle the Bayesian update
                            self.stochastic.update_kappa_from_fills(
                                total_fills,
                                session_duration_s,
                                avg_spread_bps,
                            );

                            // Log every 100 fills
                            if total_fills.is_multiple_of(100) {
                                let kappa = self.stochastic.learned_params.kappa.estimate();
                                let kappa_n = self.stochastic.learned_params.kappa.n_observations;
                                debug!(
                                    total_fills = total_fills,
                                    duration_s = session_duration_s,
                                    avg_spread_bps = %format!("{:.2}", avg_spread_bps),
                                    learned_kappa = %format!("{:.0}", kappa),
                                    kappa_obs = kappa_n,
                                    "Updated learned kappa from fill rate"
                                );
                            }
                        }

                        // Periodic INFO-level summary of learned parameters (every 100 fills)
                        let total_obs = self.stochastic.learned_params.total_fills_observed;
                        if total_obs > 0 && total_obs.is_multiple_of(100) {
                            let status = self.stochastic.learned_params.calibration_status();
                            let alpha = self.stochastic.learned_params.alpha_touch.estimate();
                            let alpha_cv = self.stochastic.learned_params.alpha_touch.cv();
                            let kappa = self.stochastic.learned_params.kappa.estimate();
                            let kappa_cv = self.stochastic.learned_params.kappa.cv();
                            let spread_floor = self.stochastic.learned_params.spread_floor_bps.estimate();

                            info!(
                                total_observations = total_obs,
                                alpha_touch = %format!("{:.4}", alpha),
                                alpha_cv = %format!("{:.2}", alpha_cv),
                                kappa = %format!("{:.0}", kappa),
                                kappa_cv = %format!("{:.2}", kappa_cv),
                                spread_floor_bps = %format!("{:.2}", spread_floor),
                                tier1_calibrated = status.tier1_ready,
                                warmup_complete = status.warmup_complete,
                                "Learned parameters summary"
                            );
                        }
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

    /// Run the market maker using the observation-based event loop.
    ///
    /// This is the unified event loop that works with any [`TradingEnvironment`].
    /// Both paper and live environments produce [`Observation`] values; the core
    /// processes them identically through [`handle_observation()`].
    ///
    /// For paper trading, fills are synthesized by the environment's stream.
    /// For live trading, fills come from the exchange's UserFills WS channel.
    pub async fn run(&mut self) -> Result<()> {
        use futures_util::StreamExt;
        use tracing::trace;

        // Environment startup sync (live: cancel orders + sync position; paper: no-op).
        self.environment.sync_state().await?;

        // Create the observation stream from the environment.
        let mut obs_stream = self.environment.observation_stream().await?;

        info!(
            asset = %self.config.asset,
            strategy = %self.strategy.name(),
            is_live = self.environment.is_live(),
            "Market maker started (observation loop)"
        );
        info!("Warming up parameter estimator...");

        // Start HJB session.
        self.stochastic.hjb_controller.start_session();
        debug!("HJB inventory controller session started");

        // Safety sync interval.
        let mut sync_interval = tokio::time::interval(Duration::from_secs(5));
        sync_interval.tick().await;

        // Shutdown signal handling.
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_flag_clone = shutdown_flag.clone();

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
            // Shutdown check.
            if shutdown_flag.load(Ordering::SeqCst) {
                info!("Shutdown flag detected, initiating graceful shutdown...");
                break;
            }

            // Kill switch check.
            if self.safety.kill_switch.is_triggered() {
                let reasons = self.safety.kill_switch.trigger_reasons();
                let reason_strs: Vec<String> = reasons.iter().map(|r| r.to_string()).collect();
                error!("KILL SWITCH TRIGGERED: {:?}", reason_strs);

                // Write post-mortem dump.
                let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);
                let trigger_str = reason_strs.first().cloned().unwrap_or_default();
                let mut dump = crate::market_maker::monitoring::PostMortemDump::new(
                    trigger_str,
                    pnl_summary.total_pnl,
                    self.safety.kill_switch.config().max_daily_loss,
                );
                dump.position = self.position.position();
                dump.daily_pnl = pnl_summary.total_pnl;
                dump.realized_pnl = pnl_summary.realized_pnl;
                dump.unrealized_pnl = pnl_summary.unrealized_pnl;
                dump.mid_price = self.latest_mid;
                dump.cascade_severity = self.tier1.liquidation_detector.cascade_severity();
                dump.risk_state = crate::market_maker::monitoring::RiskSnapshot {
                    drawdown_pct: if pnl_summary.peak_pnl > 0.0 {
                        (pnl_summary.peak_pnl - pnl_summary.total_pnl) / pnl_summary.peak_pnl * 100.0
                    } else {
                        0.0
                    },
                    margin_utilization_pct: 0.0,
                    position_utilization_pct: if self.effective_max_position > 0.0 {
                        self.position.position().abs() / self.effective_max_position * 100.0
                    } else {
                        0.0
                    },
                    rate_limit_errors: self.safety.kill_switch.state().rate_limit_errors,
                    kill_switch_reasons: reason_strs,
                };

                let postmortem_dir = std::path::Path::new("logs/postmortem");
                match dump.write_to_dir(postmortem_dir) {
                    Ok(path) => error!("Post-mortem dump written to {path:?}"),
                    Err(e) => error!("Failed to write post-mortem dump: {e}"),
                }

                break;
            }

            tokio::select! {
                // Observation processing — the unified dispatch.
                obs = obs_stream.next() => {
                    match obs {
                        Some(observation) => {
                            trace!(obs = observation.label(), "Processing observation");
                            if let Err(e) = self.handle_observation(observation).await {
                                error!("Error handling observation: {e}");
                            }

                            // Check kill switch after each observation.
                            self.check_kill_switch();

                            // Event-driven reconciliation check (live only).
                            // In paper mode, safety_sync queries the exchange REST API
                            // which would overwrite synthetic margin with real $0 balance.
                            if self.environment.is_live() {
                                if let Some(trigger) = self.infra.reconciler.should_sync() {
                                    debug!(trigger = ?trigger, "Event-driven reconciliation triggered");
                                    if let Err(e) = self.safety_sync().await {
                                        warn!("Event-driven sync failed: {e}");
                                    }
                                }
                            }

                            // Event-driven quote updates.
                            if let Err(e) = self.check_event_accumulator().await {
                                warn!("Event accumulator quote update failed: {e}");
                            }
                        }
                        None => {
                            warn!("Observation stream closed, stopping market maker");
                            break;
                        }
                    }
                }

                // Binance feed (optional).
                Some(update) = async {
                    match self.binance_receiver.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    self.handle_binance_update(update);
                }

                // Periodic safety sync + metrics + checkpoint.
                _ = sync_interval.tick() => {
                    // Safety sync for live environments only.
                    if self.environment.is_live() {
                        if let Err(e) = self.safety_sync().await {
                            warn!("Safety sync failed: {e}");
                        }
                    }

                    // Prometheus metrics update.
                    let pnl_summary = self.tier2.pnl_tracker.summary(self.latest_mid);
                    self.estimator.update_trend_pnl(pnl_summary.unrealized_pnl);
                    self.infra.prometheus.update_position(
                        self.position.position(),
                        self.effective_max_position,
                    );
                    self.infra.prometheus.update_pnl(
                        pnl_summary.total_pnl,
                        pnl_summary.total_pnl,
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
                    self.infra.prometheus.update_v2_estimator(
                        self.estimator.hierarchical_kappa_std(),
                        self.estimator.hierarchical_kappa_ci_95().0,
                        self.estimator.hierarchical_kappa_ci_95().1,
                        self.estimator.soft_toxicity_score(),
                        self.estimator.kappa_sigma_correlation(),
                        self.estimator.hierarchical_as_factor(),
                    );
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
                    self.infra.prometheus.update_calibration(
                        self.stochastic.calibration_controller.gamma_multiplier(),
                        self.stochastic.calibration_controller.calibration_progress(),
                        self.stochastic.calibration_controller.fill_count(),
                        self.stochastic.calibration_controller.is_calibrated(),
                    );
                    let learned_status = self.stochastic.learned_params.calibration_status();
                    self.infra.prometheus.update_learned_params(
                        self.stochastic.learned_params.alpha_touch.estimate(),
                        self.stochastic.learned_params.kappa.estimate(),
                        self.stochastic.learned_params.spread_floor_bps.estimate(),
                        self.stochastic.learned_params.total_fills_observed,
                        learned_status.tier1_ready,
                    );

                    self.periodic_component_update();

                    // Periodic checkpoint save (every 5 minutes).
                    if self.last_checkpoint_save.elapsed() >= Duration::from_secs(300) {
                        if let Some(ref manager) = self.checkpoint_manager {
                            let bundle = self.assemble_checkpoint_bundle();
                            if let Err(e) = manager.save_all(&bundle) {
                                warn!("Checkpoint save failed: {e}");
                            }
                            self.last_checkpoint_save = std::time::Instant::now();
                        }
                    }

                    // P&L inventory snapshot for carry calculation.
                    if self.latest_mid > 0.0 {
                        self.tier2.pnl_tracker.record_inventory_snapshot(self.latest_mid);
                    }

                    // Update learned parameters from AS classifier.
                    if self.stochastic.stochastic_config.use_learned_parameters {
                        let (informed, uninformed) = self.tier1.adverse_selection.take_informed_counts();
                        if informed > 0 || uninformed > 0 {
                            self.stochastic.learned_params.alpha_touch.observe_beta(informed, uninformed);
                        }

                        let total_fills = self.tier2.pnl_tracker.fill_count();
                        let session_duration_s = self.session_start_time.elapsed().as_secs_f64();
                        let avg_spread_bps = self.tier2.spread_tracker.current_spread_bps();
                        if total_fills >= 10 && session_duration_s > 60.0 && avg_spread_bps > 0.0 {
                            self.stochastic.update_kappa_from_fills(
                                total_fills,
                                session_duration_s,
                                avg_spread_bps,
                            );
                        }
                    }

                    // Kill switch status log.
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

        // Graceful shutdown.
        info!("Initiating graceful shutdown (5 second timeout)...");
        match tokio::time::timeout(Duration::from_secs(5), self.shutdown()).await {
            Ok(Ok(())) => info!("Graceful shutdown completed successfully"),
            Ok(Err(e)) => error!("Graceful shutdown encountered error: {e}"),
            Err(_) => error!("Graceful shutdown timed out after 5 seconds"),
        }

        info!("Market maker stopped.");
        Ok(())
    }
}
