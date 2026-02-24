//! Recovery logic, safety sync, and state refresh.
//!
//! Contains graceful shutdown, periodic safety sync operations, and margin/limit refresh.

use std::collections::HashSet;

use tracing::{debug, info, warn};

use crate::prelude::Result;

use super::super::{
    calculate_dynamic_max_position_value, safety, CancelResult, MarketMaker, QuotingStrategy,
    TradingEnvironment,
};

impl<S: QuotingStrategy, Env: TradingEnvironment> MarketMaker<S, Env> {
    /// Graceful shutdown - cancel all orders and log final state.
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("=== GRACEFUL SHUTDOWN INITIATED ===");

        // Save final checkpoint before cancelling orders
        if let Some(ref manager) = self.checkpoint_manager {
            let bundle = self.assemble_checkpoint_bundle();
            match manager.save_all(&bundle) {
                Ok(()) => info!("Final checkpoint saved successfully"),
                Err(e) => warn!("Final checkpoint save failed: {e}"),
            }
        }

        // Flush analytics before shutdown
        self.live_analytics.flush();

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
        // IMPORTANT: Fetch from exchange API (not local tracking) to catch orphans
        let exchange_oids: Vec<u64> = if let Some(dex) = &self.config.dex {
            // HIP-3 DEX: use dex-specific API
            match self
                .info_client
                .open_orders_for_dex(self.user_address, Some(dex.as_str()))
                .await
            {
                Ok(orders) => orders
                    .iter()
                    .filter(|o| o.coin == *self.config.asset)
                    .map(|o| o.oid)
                    .collect(),
                Err(e) => {
                    warn!(error = %e, "Failed to fetch orders from exchange at shutdown, using local tracking");
                    self.orders.order_ids()
                }
            }
        } else {
            // Validator perps: use standard API
            match self.info_client.open_orders(self.user_address).await {
                Ok(orders) => orders
                    .iter()
                    .filter(|o| o.coin == *self.config.asset)
                    .map(|o| o.oid)
                    .collect(),
                Err(e) => {
                    warn!(error = %e, "Failed to fetch orders from exchange at shutdown, using local tracking");
                    self.orders.order_ids()
                }
            }
        };

        let local_oids: Vec<u64> = self.orders.order_ids();
        let total_orders = exchange_oids.len();

        if exchange_oids.len() != local_oids.len() {
            warn!(
                exchange = exchange_oids.len(),
                local = local_oids.len(),
                "Order count mismatch detected at shutdown"
            );
            // Don't panic/assert, just proceed to sync logic via cancel-all below
        }

        if total_orders == 0 {
            info!("No resting orders to cancel");
        } else {
            info!(
                "Bulk cancelling {} resting orders (from exchange)...",
                total_orders
            );
            // IMPORTANT: Call executor DIRECTLY, not initiate_bulk_cancel.
            // initiate_bulk_cancel filters by OrderManager state, which skips orders
            // that aren't tracked locally (orphans, race conditions, previous sessions).
            // At shutdown, we want to cancel ALL orders from the exchange unconditionally.
            let results = self
                .environment
                .cancel_bulk_orders(&self.config.asset, exchange_oids.clone())
                .await;

            // Count successes
            let cancelled_count = results
                .iter()
                .filter(|r| {
                    matches!(
                        r,
                        CancelResult::Cancelled
                            | CancelResult::AlreadyCancelled
                            | CancelResult::AlreadyFilled
                    )
                })
                .count();
            info!(
                total = total_orders,
                cancelled = cancelled_count,
                "Bulk cancel completed"
            );
        }

        info!("=== GRACEFUL SHUTDOWN COMPLETE ===");
        Ok(())
    }

    /// Safety sync - periodically verify local state matches exchange.
    /// This is a fallback mechanism; if working correctly, should find no discrepancies.
    #[tracing::instrument(name = "safety_sync", skip_all, fields(asset = %self.config.asset))]
    pub(crate) async fn safety_sync(&mut self) -> Result<()> {
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
        // TRUST ORDERMANAGER: OrderManager is the source of truth for orders WE placed.
        // The OpenOrders snapshot lags behind order placements by 500ms-2s, so we should
        // NEVER use it to remove orders from local tracking. We trust that if we placed
        // an order and got an OID back, it exists on the exchange.
        //
        // The ONLY valid use of OpenOrders snapshot is to detect ORPHANS: orders that
        // exist on the exchange but we didn't place (from previous sessions, race conditions,
        // or other sources).

        let local_oids: HashSet<u64> = self.orders.active_order_ids();

        // Check if we have a recent snapshot for orphan detection
        const MAX_SNAPSHOT_AGE: std::time::Duration = std::time::Duration::from_secs(10);
        let snapshot_age = self
            .last_ws_snapshot_time
            .map(|t| t.elapsed())
            .unwrap_or(std::time::Duration::MAX);
        let snapshot_is_fresh = snapshot_age < MAX_SNAPSHOT_AGE;

        // Get exchange state from snapshot (for orphan detection only)
        let exchange_oids = self
            .last_ws_open_orders_snapshot
            .clone()
            .unwrap_or_default();

        // For diagnostics - lagging_count is EXPECTED due to snapshot latency
        let orphan_count = exchange_oids.difference(&local_oids).count();
        let lagging_count = local_oids.difference(&exchange_oids).count();

        debug!(
            local_count = local_oids.len(),
            snapshot_count = exchange_oids.len(),
            snapshot_age_ms = snapshot_age.as_millis(),
            orphans_in_snapshot = orphan_count,
            orders_not_in_snapshot = lagging_count,
            "[SafetySync] Order state (OrderManager is source of truth)"
        );

        // Phase 7: Clean up expired entries in orphan tracker
        self.infra.orphan_tracker.cleanup();

        // Skip orphan cancellation if snapshot is stale or empty
        if !snapshot_is_fresh || exchange_oids.is_empty() {
            debug!(
                snapshot_age_ms = snapshot_age.as_millis(),
                snapshot_empty = exchange_oids.is_empty(),
                "[SafetySync] Skipping orphan detection (snapshot stale or empty)"
            );
            return Ok(());
        }

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
                    "[SafetySync] Orphan order aged past grace period: oid={} - initiating WS verification",
                    oid
                );
            }

            // OPTIMIZATION: Verify orphans via WebSocket Request before cancelling.
            // This prevents "consuming api requests on submissions that will fail" (Ghost orders).
            // By fetching a fresh snapshot via WS (Info Request = Unlimited), we can distinguish:
            // 1. Real Orphans (In Snapshot) -> Send CANCEL (Consumes Action Limit)
            // 2. Ghost Orders (Not in Snapshot) -> Remove Locally (No Cost)
            let mut confirmed_orphans = Vec::new();
            let mut ghost_orphans = Vec::new();

            // OPTIMIZATION: Verify orphans via WebSocket Request before cancelling.
            // This prevents "consuming api requests on submissions that will fail" (Ghost orders).
            // By fetching a fresh snapshot via WS (Info Request = Unlimited), we can distinguish:
            // 1. Real Orphans (In Snapshot) -> Send CANCEL (Consumes Action Limit)
            // 2. Ghost Orders (Not in Snapshot) -> Remove Locally (No Cost)

            // Step 1: Build the request payload
            use crate::info::info_client::InfoRequest;
            let payload_result = serde_json::to_value(InfoRequest::OpenOrders {
                user: self.user_address,
                dex: None,
            });

            let verification_result: Option<
                Vec<crate::info::response_structs::OpenOrdersResponse>,
            > = match payload_result {
                Ok(payload) => {
                    debug!("[SafetySync] Sending WS POST info request for OpenOrders verification");

                    // Step 2: Send via WS POST with faster timeout
                    // FIX: Reduced from 5s to 1s to reduce state sync lag
                    // On timeout, SafetySync will skip verification (conservative fallback)
                    match self
                        .info_client
                        .ws_post_info(payload, std::time::Duration::from_secs(1))
                        .await
                    {
                        Ok(response_data) => {
                            debug!("[SafetySync] WS POST response received, parsing...");

                            // Step 3: Extract payload from response
                            match response_data.response {
                                crate::ws::message_types::WsPostResponsePayload::Info {
                                    payload,
                                } => {
                                    // Step 4: Deserialize to OpenOrdersResponse
                                    match serde_json::from_value::<
                                        Vec<crate::info::response_structs::OpenOrdersResponse>,
                                    >(payload.clone())
                                    {
                                        Ok(orders) => {
                                            info!(
                                                count = orders.len(),
                                                "[SafetySync] WS verification successful, got {} orders from exchange",
                                                orders.len()
                                            );
                                            Some(orders)
                                        }
                                        Err(e) => {
                                            warn!(
                                                error = %e,
                                                payload = %payload,
                                                "[SafetySync] Failed to deserialize OpenOrders from WS POST response"
                                            );
                                            None
                                        }
                                    }
                                }
                                crate::ws::message_types::WsPostResponsePayload::Error {
                                    payload,
                                } => {
                                    warn!(
                                        error = %payload,
                                        "[SafetySync] WS POST returned error response"
                                    );
                                    None
                                }
                                crate::ws::message_types::WsPostResponsePayload::Action {
                                    ..
                                } => {
                                    warn!("[SafetySync] WS POST returned unexpected Action response for Info request");
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            warn!(
                                error = %e,
                                "[SafetySync] WS POST info request failed"
                            );
                            None
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        "[SafetySync] Failed to serialize InfoRequest payload"
                    );
                    None
                }
            };

            if let Some(open_orders) = verification_result {
                // Convert to Set for fast lookup
                let exchange_oids_set: std::collections::HashSet<u64> =
                    open_orders.iter().map(|o| o.oid).collect();

                for &oid in &aged_orphans {
                    if exchange_oids_set.contains(&oid) {
                        confirmed_orphans.push(oid);
                    } else {
                        ghost_orphans.push(oid);
                    }
                }

                if !ghost_orphans.is_empty() {
                    info!(
                        count = ghost_orphans.len(),
                        "[SafetySync] Detected {} ghost orders (present in WS state but gone on Exchange). Removing locally without Cancel.",
                        ghost_orphans.len()
                    );

                    for &oid in &ghost_orphans {
                        // Remove from tracker
                        self.infra.orphan_tracker.clear_orphan(oid);

                        // Force remove from WS State (Fix synchronization)
                        if let Some(order) = self.ws_state.remove_order(oid) {
                            self.safety.fill_processor.record_cancelled_order(
                                oid,
                                order.side,
                                order.price,
                                order.size,
                            );
                            warn!(oid = oid, "[SafetySync] Ghost order removed from ws_state (Verified gone via WS Snapshot)");
                        }
                    }
                }
            } else {
                // specific warning if verification failed
                warn!("[SafetySync] WS Verification failed (timeout/error). Proceeding with bulk cancel for ALL candidates to be safe.");
                confirmed_orphans = aged_orphans.clone();
            }

            // Only cancel confirmed orphans
            if !confirmed_orphans.is_empty() {
                info!(
                    count = confirmed_orphans.len(),
                    "[SafetySync] Bulk cancelling {} CONFIRMED orphan orders",
                    confirmed_orphans.len()
                );

                // Bulk cancel for efficiency (single API call)
                let cancel_results = self
                    .environment
                    .cancel_bulk_orders(&self.config.asset, confirmed_orphans.clone())
                    .await;

                // Process results and clear from orphan tracking
                for (oid, result) in confirmed_orphans.iter().zip(cancel_results.iter()) {
                    safety::SafetyAuditor::log_orphan_cancellation(*oid, result.order_is_gone());
                    self.infra.orphan_tracker.clear_orphan(*oid);

                    // CRITICAL FIX: If the order is confirmed gone (Cancelled OR AlreadyFilled),
                    // we MUST remove it from ws_state.
                    if result.order_is_gone() {
                        if let Some(order) = self.ws_state.remove_order(*oid) {
                            self.safety.fill_processor.record_cancelled_order(
                                *oid,
                                order.side,
                                order.price,
                                order.size,
                            );
                            warn!(
                                oid = oid,
                                "Forced removal from ws_state to correct synchronization (cancel confirmed/filled)"
                            );
                        }
                    }
                }
            } else if !ghost_orphans.is_empty() {
                info!("[SafetySync] All candidates were ghost orders. No Cancel Actions sent.");
            }
        }

        // DISABLED: Stale local order removal.
        //
        // This was the ROOT CAUSE of the state mismatch bug. The logic was:
        // "If order is in local but not in snapshot → it's stale → remove it"
        //
        // But the OpenOrders snapshot LAGS order placements by 1-5 seconds!
        // So this was DELETING valid orders that simply hadn't appeared in the snapshot yet.
        //
        // TRUST ORDERMANAGER: If we placed an order and got an OID back, it exists.
        // The snapshot not having it is expected latency, not a problem.
        //
        // The ONLY time to remove from OrderManager is:
        // 1. When we get a fill confirmation (order filled)
        // 2. When we get a cancel confirmation (order cancelled)
        // 3. Never based on snapshot absence
        //
        // Keeping this code commented for reference:
        // let stale_local = safety::SafetyAuditor::find_stale_local(...);
        // for oid in stale_local { self.orders.remove_order(oid); }

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
    pub(crate) async fn refresh_margin_state(&mut self) -> Result<()> {
        // Phase 2 Optimization:
        // If we are on standard perps (not HIP-3 DEX) and have fresh WebData2,
        // we can skip the REST call entirely because handle_web_data2 updates the state.
        if self.config.dex.is_none() {
            if let Some(last_ws_time) = self.last_web_data2_time {
                if last_ws_time.elapsed() < std::time::Duration::from_secs(10) {
                    debug!("[MarginSync] Skipping REST - WebData2 is fresh");
                    return Ok(());
                }
            }
        }

        // For HIP-3 DEXs, collateral is in spot balance (USDE, USDH, etc.)
        // For validator perps, collateral is in perps clearinghouse (USDC)
        let (account_value, margin_used, total_notional, liquidation_price) =
            if self.config.dex.is_some() {
                // HIP-3: Get account value from spot balance (total amount)
                // Use local cache if available, otherwise fall back to REST
                let account_value = if !self.spot_balance_cache.is_empty() {
                    // Cache stores total balance (not available), which is what we want
                    self.spot_balance_cache
                        .get(&self.config.collateral.symbol)
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    // Fallback to REST if cache is empty
                    let balances = self
                        .info_client
                        .user_token_balances(self.user_address)
                        .await?;
                    self.config
                        .collateral
                        .balance_from_spot(&balances.balances)
                        .map(|(total, _hold)| total)
                        .unwrap_or(0.0)
                };

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
                let cross_value: f64 = user_state
                    .cross_margin_summary
                    .account_value
                    .parse()
                    .unwrap_or(0.0);
                let margin_value: f64 = user_state
                    .margin_summary
                    .account_value
                    .parse()
                    .unwrap_or(0.0);

                // Unified margin: perps clearinghouse + spot stablecoin balances
                let perps_value = cross_value.max(margin_value);
                let spot_collateral: f64 = ["USDC", "USDT", "USDE"]
                    .iter()
                    .filter_map(|coin| self.spot_balance_cache.get(*coin).copied())
                    .sum();
                let account_value = perps_value + spot_collateral;

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
                buffer_ratio = ?margin_state.liquidation_buffer_ratio().map(|r| format!("{r:.2}")),
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
    ///
    /// NOTE: This is now a FALLBACK. If WebSocket `ActiveAssetData` is fresh (<30s),
    /// we skip the REST call entirely.
    pub(crate) async fn refresh_exchange_limits(&mut self) -> Result<()> {
        // Check if WebSocket data is fresh - skip REST if so
        if let Some(last_ws_time) = self.last_active_asset_data_time {
            let ws_age = last_ws_time.elapsed();
            if ws_age < std::time::Duration::from_secs(30) {
                debug!(
                    ws_age_secs = ws_age.as_secs(),
                    "[ExchangeLimits] Skipping REST - WebSocket data is fresh"
                );
                return Ok(());
            } else {
                warn!(
                    ws_age_secs = ws_age.as_secs(),
                    "[ExchangeLimits] WebSocket data is stale, falling back to REST"
                );
            }
        }

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
}
