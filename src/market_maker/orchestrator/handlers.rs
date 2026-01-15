//! WebSocket message handlers for the market maker.

use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::prelude::Result;
use crate::Message;

use super::super::{
    fills, messages, tracking::ws_order_state::WsFillEvent,
    tracking::ws_order_state::WsOrderUpdateEvent, MarketMaker, OrderExecutor, OrderState,
    QuotingStrategy, Side, TrackedOrder,
};

impl<S: QuotingStrategy, E: OrderExecutor> MarketMaker<S, E> {
    /// Handle a message from subscriptions.
    /// Main message dispatcher - routes to focused handlers.
    pub(crate) async fn handle_message(&mut self, message: Message) -> Result<()> {
        match message {
            Message::AllMids(all_mids) => self.handle_all_mids(all_mids).await,
            Message::Trades(trades) => self.handle_trades(trades),
            Message::UserFills(user_fills) => self.handle_user_fills(user_fills).await,
            Message::L2Book(l2_book) => self.handle_l2_book(l2_book),
            Message::OrderUpdates(order_updates) => self.handle_order_updates(order_updates),
            Message::OpenOrders(open_orders) => self.handle_open_orders(open_orders),
            Message::ActiveAssetData(active_asset_data) => {
                self.handle_active_asset_data(active_asset_data)
            }
            Message::WebData2(web_data2) => self.handle_web_data2(web_data2),
            Message::UserNonFundingLedgerUpdates(update) => self.handle_ledger_update(update),
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
            // Update learning module with current mid for prediction scoring
            self.learning.update_mid(self.latest_mid);

            // Record price for dashboard time series visualization
            self.infra.prometheus.record_price_for_dashboard(self.latest_mid);

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
            connection_supervisor: &self.infra.connection_supervisor,
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
            learning: &mut self.learning,
            stochastic_controller: &mut self.stochastic.controller,
            fee_bps: self.config.fee_bps,
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
            connection_supervisor: &self.infra.connection_supervisor,
        };

        let result = messages::process_l2_book(&l2_book, &ctx, &mut state)?;

        // Feed L2 data to adaptive spread calculator's blended kappa estimator
        // This enables book-based kappa estimation to work alongside own-fill kappa
        if result.is_valid && self.latest_mid > 0.0 {
            if let Some((bids, asks)) = Self::parse_l2_for_adaptive(&l2_book.data.levels) {
                self.stochastic
                    .adaptive_spreads
                    .on_l2_update(&bids, &asks, self.latest_mid);

                // Record book snapshot for dashboard time series visualization
                self.infra
                    .prometheus
                    .record_book_for_dashboard(&bids, &asks);
            }
        }

        Ok(())
    }

    /// Parse L2 book levels into (bids, asks) tuples for adaptive spread calculator.
    /// Returns None if the levels are invalid or insufficient.
    #[allow(clippy::type_complexity)]
    pub(super) fn parse_l2_for_adaptive(
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

    /// Handle OpenOrders message - syncs WS order snapshot to local state.
    ///
    /// This subscription provides a reliable snapshot of all open orders directly from
    /// the WebSocket stream, avoiding the race conditions inherent in REST-based polling.
    /// The snapshot is used for:
    /// 1. Initial state synchronization on startup
    /// 2. Periodic state reconciliation (replaces REST-based safety_sync)
    fn handle_open_orders(
        &mut self,
        open_orders: crate::ws::message_types::OpenOrders,
    ) -> Result<()> {
        // Filter to our asset
        let our_orders: Vec<_> = open_orders
            .data
            .orders
            .iter()
            .filter(|o| o.coin == *self.config.asset)
            .collect();

        let ws_open_oids: std::collections::HashSet<u64> =
            our_orders.iter().map(|o| o.oid).collect();

        // Store the latest WS snapshot for use by safety_sync
        self.last_ws_open_orders_snapshot = Some(ws_open_oids.clone());
        self.last_ws_snapshot_time = Some(std::time::Instant::now());

        // === Full Synchronization: ws_state should mirror the snapshot ===
        // Step 1: Get current ws_state OIDs
        let current_ws_oids = self.ws_state.open_order_ids();

        // Step 2: Identify orders to REMOVE (in ws_state but NOT in snapshot)
        // IMPORTANT: Apply grace period for fresh orders to avoid race condition
        // between WS POST placement and openOrders snapshot delivery
        const GRACE_PERIOD_SECS: u64 = 2;
        let grace_period = std::time::Duration::from_secs(GRACE_PERIOD_SECS);

        let (stale_oids, skipped_fresh): (Vec<u64>, Vec<u64>) = current_ws_oids
            .iter()
            .filter(|oid| !ws_open_oids.contains(oid))
            .partition(|oid| {
                // Only mark as stale if order is older than grace period
                self.ws_state
                    .get_order(**oid)
                    .map(|o| o.placed_at.elapsed() > grace_period)
                    .unwrap_or(true) // If order not found, treat as stale
            });

        // Log fresh orders that were skipped
        if !skipped_fresh.is_empty() {
            debug!(
                count = skipped_fresh.len(),
                grace_period_secs = GRACE_PERIOD_SECS,
                "[OpenOrders] Skipping removal of fresh orders (age < grace period)"
            );
        }

        // Step 3: Identify orders to ADD (in snapshot but NOT in ws_state)
        let new_oids: Vec<_> = our_orders
            .iter()
            .filter(|o| !current_ws_oids.contains(&o.oid))
            .collect();

        // Log sync activity
        if !stale_oids.is_empty() || !new_oids.is_empty() {
            info!(
                snapshot_count = our_orders.len(),
                ws_state_count = current_ws_oids.len(),
                removed = stale_oids.len(),
                added = new_oids.len(),
                "[OpenOrders] Synchronizing ws_state with exchange snapshot"
            );
        } else {
            debug!(
                orders = our_orders.len(),
                "Received OpenOrders snapshot (ws_state already in sync)"
            );
        }

        // Step 4: Remove stale orders from ws_state
        let mut removed_count = 0usize;
        for oid in &stale_oids {
            if self.ws_state.remove_order(*oid).is_some() {
                removed_count += 1;
                debug!(
                    oid = oid,
                    "[OpenOrders] Removed order from ws_state (not in exchange snapshot)"
                );
            }
            // Also clear from orphan tracker if present
            self.infra.orphan_tracker.clear_orphan(*oid);
        }
        // Log summary at INFO level if any were removed
        if removed_count > 0 {
            info!(
                removed_count = removed_count,
                "[OpenOrders] Cleaned up stale orders from ws_state"
            );
        }

        // Step 5: Add new orders to ws_state
        for order in new_oids {
            let side = if order.side == "B" || order.side.to_lowercase() == "buy" {
                Side::Buy
            } else {
                Side::Sell
            };
            let tracked = TrackedOrder::with_cloid(
                order.oid,
                order.cloid.clone().unwrap_or_default(),
                side,
                order.limit_px.parse().unwrap_or(0.0),
                order.sz.parse().unwrap_or(0.0),
            );
            self.ws_state.add_order(tracked);
            debug!(
                oid = order.oid,
                "Added order from OpenOrders snapshot to ws_state"
            );
        }

        Ok(())
    }

    /// Handle ActiveAssetData message - real-time exchange limit updates.
    ///
    /// This replaces the periodic REST polling in `refresh_exchange_limits`.
    /// Updates position limits (max long/short) and available margin in real-time.
    fn handle_active_asset_data(
        &mut self,
        active_asset_data: crate::ws::message_types::ActiveAssetData,
    ) -> Result<()> {
        // Filter to our asset (should already match since we subscribed per-coin)
        if active_asset_data.data.coin != *self.config.asset {
            return Ok(());
        }

        // Update exchange limits using WebSocket data
        // Note: WS message lacks mark_px, so we use our tracked latest_mid
        self.infra.exchange_limits.update_from_ws(
            &active_asset_data.data,
            self.latest_mid,
            self.effective_max_position,
        );

        // Update staleness timestamp
        self.last_active_asset_data_time = Some(std::time::Instant::now());

        debug!(
            coin = %active_asset_data.data.coin,
            "Exchange limits updated from WebSocket"
        );

        Ok(())
    }

    /// Handle WebData2 message - real-time margin and position updates.
    ///
    /// This replaces the periodic REST polling in `refresh_margin_state` and `sync_position_from_exchange`.
    fn handle_web_data2(&mut self, web_data2: crate::ws::message_types::WebData2) -> Result<()> {
        // Note: web_data2.data does not contain 'user', so we assume it matches our subscription.

        let state = &web_data2.data.clearinghouse_state;

        // 1. Parse margin summary
        let account_value = state
            .margin_summary
            .account_value
            .parse::<f64>()
            .unwrap_or(0.0);
        let margin_used = state
            .margin_summary
            .total_margin_used
            .parse::<f64>()
            .unwrap_or(0.0);
        let total_notional = state
            .margin_summary
            .total_ntl_pos
            .parse::<f64>()
            .unwrap_or(0.0);

        // 2. Find position and liquidation price for our asset
        let (exchange_position, liquidation_price) = state
            .asset_positions
            .iter()
            .find(|p| p.position.coin == *self.config.asset)
            .map(|p| {
                (
                    p.position.szi.parse::<f64>().unwrap_or(0.0),
                    p.position
                        .liquidation_px
                        .as_ref()
                        .and_then(|px| px.parse::<f64>().ok()),
                )
            })
            .unwrap_or((0.0, None));

        // Update staleness timestamp
        self.last_web_data2_time = Some(std::time::Instant::now());

        // 3. Update local position tracking if relevant
        let local_position = self.position.position();
        let drift = (exchange_position - local_position).abs();

        if drift > crate::EPSILON {
            self.position.set_position(exchange_position);
            debug!(
                local = local_position,
                exchange = exchange_position,
                "Position synced from WebData2"
            );
        }

        // 4. Update MarginSizer state
        // IMPORTANT: For HIP-3 DEXs, account_value comes from spot balances (Phase 3).
        // WebData2's clearinghouseState reports the USDC clearinghouse (=0 for HIP-3 users).
        // Only update margin for validator perps (non-HIP-3).
        if self.config.dex.is_none() {
            self.infra.margin_sizer.update_state_with_liquidation(
                account_value,
                margin_used,
                total_notional,
                liquidation_price,
                self.latest_mid,
                self.position.position(),
            );

            // Log occasionally
            debug!(
                account_value = account_value,
                margin_used = margin_used,
                liquidation_price = ?liquidation_price,
                "Margin state updated from WebData2"
            );
        }

        Ok(())
    }

    /// Handle ledger updates (spot balance deltas).
    fn handle_ledger_update(
        &mut self,
        update: crate::ws::message_types::UserNonFundingLedgerUpdates,
    ) -> Result<()> {
        let data = update.data;
        if data.user != self.user_address {
            return Ok(());
        }

        use crate::types::LedgerUpdate;

        for item in data.non_funding_ledger_updates {
            match item.delta {
                LedgerUpdate::Deposit(d) => {
                    let amount = d.usdc.parse::<f64>().unwrap_or(0.0);
                    let entry = self
                        .spot_balance_cache
                        .entry("USDC".to_string())
                        .or_insert(0.0);
                    *entry += amount;
                    debug!(
                        coin = "USDC",
                        amount = amount,
                        total = *entry,
                        "Deposit processed"
                    );
                }
                LedgerUpdate::Withdraw(w) => {
                    let amount = w.usdc.parse::<f64>().unwrap_or(0.0);
                    let entry = self
                        .spot_balance_cache
                        .entry("USDC".to_string())
                        .or_insert(0.0);
                    *entry -= amount;
                    debug!(
                        coin = "USDC",
                        amount = amount,
                        total = *entry,
                        "Withdraw processed"
                    );
                }
                LedgerUpdate::SpotTransfer(t) => {
                    let token = t.token.clone();
                    let amount = t.amount.parse::<f64>().unwrap_or(0.0);

                    if t.destination == self.user_address {
                        let entry = self.spot_balance_cache.entry(token.clone()).or_insert(0.0);
                        *entry += amount;
                    } else {
                        let entry = self.spot_balance_cache.entry(token.clone()).or_insert(0.0);
                        *entry -= amount;
                    }
                    debug!(coin = token, amount = amount, "SpotTransfer processed");
                }
                _ => {}
            }
        }
        Ok(())
    }
}
