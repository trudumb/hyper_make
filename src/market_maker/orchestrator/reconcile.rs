//! Order reconciliation and ladder management.
//!
//! Handles matching current orders to target quotes and generating the
//! appropriate cancel/modify/place actions.

use tracing::{debug, info, warn};

use crate::helpers::truncate_float;
use crate::prelude::Result;

use crate::EPSILON;

use super::super::{
    tracking, CancelResult, MarketMaker, ModifySpec, OrderExecutor, OrderSpec, OrderState, Quote,
    QuotingStrategy, Side, TrackedOrder,
};
use super::{partition_ladder_actions, side_str};

/// Minimum order notional value in USD (Hyperliquid requirement)
pub(crate) const MIN_ORDER_NOTIONAL: f64 = 10.0;

impl<S: QuotingStrategy, E: OrderExecutor> MarketMaker<S, E> {
    /// Reconcile a single order per side (legacy single-order mode).
    pub(crate) async fn reconcile_side(
        &mut self,
        side: Side,
        new_quote: Option<Quote>,
    ) -> Result<()> {
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
    pub(crate) async fn reconcile_ladder_side(
        &mut self,
        side: Side,
        new_quotes: Vec<Quote>,
    ) -> Result<()> {
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
                    true, // post_only (ALO)
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

            // Track batch rejection metrics
            // Problem: Previously, each order rejection individually incremented the counter,
            // so a batch of 25 rejections caused exponential backoff to 120s immediately.
            // Fix: Count rejections per batch and record once after the loop.
            let mut rejection_count: u32 = 0;
            let mut first_rejection_error: Option<String> = None;
            let mut had_success = false;
            let mut recovery_escalated = false;

            // Finalize pending orders with real OIDs (using CLOID for deterministic matching)
            for (i, result) in results.iter().enumerate() {
                let spec = &order_specs[i];

                // Phase 5: Record success/rejection for rate limiting
                if result.oid > 0 {
                    // Order placed successfully - reset rejection counter
                    had_success = true;
                    self.infra.recovery_manager.record_success();
                } else if let Some(ref err) = result.error {
                    // Count rejections for batch processing (rate limiter)
                    rejection_count += 1;
                    if first_rejection_error.is_none() {
                        first_rejection_error = Some(err.clone());
                    }
                    // Phase 4: Trigger reconciliation for position-related rejections
                    self.infra.reconciler.on_order_rejection(err);
                    // Phase 3: Record rejection for recovery state machine
                    // This may transition to IocRecovery if consecutive rejections exceed threshold
                    if self.infra.recovery_manager.record_rejection(is_buy, err) {
                        recovery_escalated = true;
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

            // After loop: Record batch rejection for rate limiting (once per batch, not per order)
            // This prevents 25 rejections from immediately hitting 120s backoff.
            if rejection_count > 0 {
                if let Some(ref err) = first_rejection_error {
                    if let Some(backoff) =
                        self.infra
                            .rate_limiter
                            .record_batch_rejection(is_buy, err, rejection_count)
                    {
                        warn!(
                            side = %side_str(side),
                            backoff_secs = %format!("{:.1}", backoff.as_secs_f64()),
                            rejection_count = rejection_count,
                            error = %err,
                            "Rate limiter entering backoff due to batch rejection"
                        );
                    }
                }
                if recovery_escalated {
                    warn!(
                        side = %side_str(side),
                        "Recovery manager escalating to IOC recovery mode"
                    );
                }
            }
            // Record success for rate limiter (resets counter) - only if we had any success
            if had_success {
                self.infra.rate_limiter.record_success(is_buy);
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
    pub(crate) fn ladder_needs_update(&self, side: Side, new_quotes: &[Quote]) -> bool {
        let current = self.orders.get_all_by_side(side);

        // Different number of levels
        if current.len() != new_quotes.len() {
            return true;
        }

        // Sort current orders by price (bids: descending, asks: ascending)
        // Filter out any orders with NaN prices (defensive - should never happen)
        let mut sorted_current: Vec<_> = current
            .iter()
            .filter(|order| {
                if order.price.is_nan() {
                    tracing::error!(oid = ?order.oid, "Order has NaN price, excluding from reconciliation");
                    false
                } else {
                    true
                }
            })
            .collect();
        if side == Side::Buy {
            // Use unwrap_or(Equal) to handle potential NaN comparisons safely
            sorted_current.sort_by(|a, b| {
                b.price
                    .partial_cmp(&a.price)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            sorted_current.sort_by(|a, b| {
                a.price
                    .partial_cmp(&b.price)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
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
    pub(crate) async fn reconcile_ladder_smart(
        &mut self,
        bid_quotes: Vec<Quote>,
        ask_quotes: Vec<Quote>,
    ) -> Result<()> {
        use crate::market_maker::quoting::LadderLevel;
        use crate::market_maker::tracking::{
            priority_based_matching, DynamicReconcileConfig, LadderAction, ReconcileStats,
        };

        let reconcile_config = &self.config.reconcile;

        // === UNCONDITIONAL RATE LIMIT CHECK ===
        // Refresh rate limit cache and throttle if headroom is low.
        // This runs regardless of impulse_enabled to prevent hitting rate limits.
        {
            use crate::market_maker::core::CachedRateLimit;
            use std::time::Duration;

            // Refresh rate limit cache if stale (every 60 seconds)
            let cache_max_age = Duration::from_secs(60);
            let should_refresh = self
                .infra
                .cached_rate_limit
                .as_ref()
                .is_none_or(|c| c.is_stale(cache_max_age));

            if should_refresh {
                if let Ok(response) = self.info_client.user_rate_limit(self.user_address).await {
                    self.infra.cached_rate_limit = Some(CachedRateLimit::from_response(&response));
                    debug!(
                        used = response.n_requests_used,
                        cap = response.n_requests_cap,
                        headroom_pct = %format!("{:.1}%", response.headroom_pct() * 100.0),
                        "Rate limit status"
                    );
                }
            }

            // Throttle if headroom < 10% (more aggressive than impulse control's 5%)
            // This gives buffer before we hit the hard 5% limit
            // Uses proactive_rate_tracker's minimum interval check
            if let Some(ref cache) = self.infra.cached_rate_limit {
                let headroom = cache.headroom_pct();
                if headroom < 0.10 && !self.infra.proactive_rate_tracker.can_modify() {
                    // Rate limit is tight AND we're within the minimum modify interval
                    // Skip this cycle to conserve rate limit budget
                    debug!(
                        headroom_pct = %format!("{:.1}%", headroom * 100.0),
                        "Rate limit headroom low (<10%) - throttling reconciliation"
                    );
                    return Ok(());
                }
            }
        }

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

        // DIAGNOSTIC: Log order counts to help debug reconciliation issues
        // Compare local tracking (OrderManager) vs WebSocket state (exchange truth)
        // If ws_* differs from local_*, there's a state synchronization issue
        let ws_bids = self.ws_state.get_orders_by_side(Side::Buy);
        let ws_asks = self.ws_state.get_orders_by_side(Side::Sell);

        // Format order details as "price@size" for compact logging
        let format_orders = |orders: &[&TrackedOrder]| -> String {
            let mut sorted: Vec<_> = orders.iter().collect();
            sorted.sort_by(|a, b| {
                b.price
                    .partial_cmp(&a.price)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted
                .iter()
                .map(|o| format!("{:.4}@{:.4}", o.price, o.size))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let format_levels = |levels: &[LadderLevel]| -> String {
            levels
                .iter()
                .map(|l| format!("{:.4}@{:.4}", l.price, l.size))
                .collect::<Vec<_>>()
                .join(", ")
        };

        info!(
            local_bids = current_bids.len(),
            local_asks = current_asks.len(),
            ws_bids = ws_bids.len(),
            ws_asks = ws_asks.len(),
            target_bid_levels = bid_levels.len(),
            target_ask_levels = ask_levels.len(),
            "[Reconcile] Order counts (local vs exchange)"
        );

        // Log detailed order levels for debugging price/size discrepancies
        // Note: Use 'ladder' instead of 'target' to avoid collision with tracing's target field
        info!(
            local = %format_orders(&current_bids),
            ws = %format_orders(&ws_bids),
            ladder = %format_levels(&bid_levels),
            "[Reconcile] BID levels (price@size)"
        );
        info!(
            local = %format_orders(&current_asks),
            ws = %format_orders(&ws_asks),
            ladder = %format_levels(&ask_levels),
            "[Reconcile] ASK levels (price@size)"
        );

        // === EMPTY LADDER RECOVERY ===
        // When local tracking is completely empty but we have target quotes,
        // this is a CRITICAL recovery situation. Bypass drift/impulse checks
        // and immediately place all target orders to restore quoting.
        let ladder_empty = current_bids.is_empty() && current_asks.is_empty();
        let has_targets = !bid_levels.is_empty() || !ask_levels.is_empty();

        if ladder_empty && has_targets {
            warn!(
                target_bids = bid_levels.len(),
                target_asks = ask_levels.len(),
                "EMPTY LADDER DETECTED - forcing immediate replenishment (bypassing all checks)"
            );

            // Delegate to place_bulk_ladder_orders for proper CLOID tracking
            // This prevents orphan orders by using the pending registration flow
            let bid_places: Vec<(f64, f64)> =
                bid_levels.iter().map(|l| (l.price, l.size)).collect();
            let ask_places: Vec<(f64, f64)> =
                ask_levels.iter().map(|l| (l.price, l.size)).collect();

            let mut placed_count = 0usize;
            if !bid_places.is_empty() {
                placed_count += bid_places.len();
                self.place_bulk_ladder_orders(Side::Buy, bid_places).await?;
            }
            if !ask_places.is_empty() {
                placed_count += ask_places.len();
                self.place_bulk_ladder_orders(Side::Sell, ask_places)
                    .await?;
            }

            info!(
                placed = placed_count,
                "[Reconcile] Empty ladder recovery complete"
            );

            // Skip normal reconciliation since we just replenished
            return Ok(());
        }

        // === DRIFT DETECTION: Force full reconciliation on large price drift ===
        // If existing orders are too far from targets, cancel all and place new
        // This catches cases where orders become "orphaned" from reconciliation
        const MAX_ACCEPTABLE_DRIFT_BPS: f64 = 100.0;

        let bid_drift = if !current_bids.is_empty() && !bid_levels.is_empty() {
            // Calculate max drift between current bids and target bid levels
            current_bids
                .iter()
                .filter_map(|order| {
                    bid_levels
                        .iter()
                        .map(|level| crate::bps_diff(order.price, level.price) as f64)
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0)
        } else {
            0.0
        };

        let ask_drift = if !current_asks.is_empty() && !ask_levels.is_empty() {
            // Calculate max drift between current asks and target ask levels
            current_asks
                .iter()
                .filter_map(|order| {
                    ask_levels
                        .iter()
                        .map(|level| crate::bps_diff(order.price, level.price) as f64)
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0)
        } else {
            0.0
        };

        let max_drift = bid_drift.max(ask_drift);
        if max_drift > MAX_ACCEPTABLE_DRIFT_BPS {
            warn!(
                bid_drift_bps = %format!("{:.1}", bid_drift),
                ask_drift_bps = %format!("{:.1}", ask_drift),
                threshold_bps = MAX_ACCEPTABLE_DRIFT_BPS,
                "[Reconcile] Large price drift detected! Forcing full cancel + replace"
            );

            // Cancel all current orders
            let all_oids: Vec<u64> = current_bids
                .iter()
                .chain(current_asks.iter())
                .map(|o| o.oid)
                .collect();

            if !all_oids.is_empty() {
                self.initiate_bulk_cancel(all_oids.clone()).await;
                info!(
                    cancelled = all_oids.len(),
                    "[Reconcile] Cancelled all stale orders due to drift"
                );
            }

            // Delegate to place_bulk_ladder_orders for proper CLOID tracking
            // This prevents orphan orders by using the pending registration flow
            let bid_places: Vec<(f64, f64)> =
                bid_levels.iter().map(|l| (l.price, l.size)).collect();
            let ask_places: Vec<(f64, f64)> =
                ask_levels.iter().map(|l| (l.price, l.size)).collect();

            if !bid_places.is_empty() {
                self.place_bulk_ladder_orders(Side::Buy, bid_places).await?;
            }
            if !ask_places.is_empty() {
                self.place_bulk_ladder_orders(Side::Sell, ask_places)
                    .await?;
            }

            info!(
                bid_targets = bid_levels.len(),
                ask_targets = ask_levels.len(),
                "[Reconcile] Placed new orders after drift correction"
            );

            // Skip normal reconciliation since we did a full replace
            return Ok(());
        }

        // Impulse Control: Check exchange rate limits before proceeding
        // Uses actual Hyperliquid API instead of arbitrary token bucket
        let impulse_enabled =
            self.infra.impulse_control_enabled && reconcile_config.use_impulse_filter;
        if impulse_enabled {
            use crate::market_maker::core::CachedRateLimit;
            use std::time::Duration;

            // Refresh rate limit cache if stale (every 60 seconds)
            let cache_max_age = Duration::from_secs(60);
            let should_refresh = self
                .infra
                .cached_rate_limit
                .as_ref()
                .is_none_or(|c| c.is_stale(cache_max_age));

            if should_refresh {
                match self.info_client.user_rate_limit(self.user_address).await {
                    Ok(response) => {
                        self.infra.cached_rate_limit =
                            Some(CachedRateLimit::from_response(&response));
                        debug!(
                            used = response.n_requests_used,
                            cap = response.n_requests_cap,
                            headroom_pct = %format!("{:.1}%", response.headroom_pct() * 100.0),
                            "Refreshed exchange rate limit cache"
                        );
                    }
                    Err(e) => {
                        debug!(
                            error = %e,
                            "Failed to refresh rate limit cache - proceeding without limit check"
                        );
                    }
                }
            }

            // Only throttle if headroom < 5% (approaching rate limit)
            if let Some(ref cache) = self.infra.cached_rate_limit {
                let headroom = cache.headroom_pct();
                if headroom < 0.05 {
                    warn!(
                        headroom_pct = %format!("{:.1}%", headroom * 100.0),
                        used = cache.n_requests_used,
                        cap = cache.n_requests_cap,
                        "Exchange rate limit low (<5% headroom) - throttling reconciliation"
                    );
                    return Ok(());
                }
            }
        }

        // Generate actions for each side using priority-based matching (default)
        // Fallback paths kept for legacy compatibility but not used

        // DIAGNOSTIC: Checkpoint to verify we reach this point
        info!(
            local_bids = current_bids.len(),
            local_asks = current_asks.len(),
            target_bids = bid_levels.len(),
            target_asks = ask_levels.len(),
            bid_drift_bps = %format!("{:.1}", bid_drift),
            ask_drift_bps = %format!("{:.1}", ask_drift),
            impulse_enabled = impulse_enabled,
            "[Reconcile] Checkpoint: passed drift and impulse checks"
        );

        // === PRIORITY-BASED MATCHING (DEFAULT) ===
        // Uses stochastic optimal spread to derive matching thresholds
        // Ensures best bid/ask levels are always covered first

        // Get sigma from queue tracker (updated from book data)
        let sigma = self.tier1.queue_tracker.sigma();

        // Use conservative gamma/kappa until we have market params cached
        // TODO: Cache actual gamma/kappa from ladder strategy in per-cycle state
        let gamma = 0.1; // Conservative default
        let kappa = 1500.0; // Typical for liquid perps

        let dynamic_config = DynamicReconcileConfig::from_market_params(
            gamma,
            kappa,
            sigma,
            reconcile_config.queue_horizon_seconds,
        );

        info!(
            sigma = %format!("{:.6}", sigma),
            optimal_spread_bps = %format!("{:.2}", dynamic_config.optimal_spread_bps),
            best_tolerance_bps = %format!("{:.2}", dynamic_config.best_level_tolerance_bps),
            outer_tolerance_bps = %format!("{:.2}", dynamic_config.outer_level_tolerance_bps),
            "[Reconcile] Priority-based matching with dynamic thresholds"
        );

        let bid_acts = priority_based_matching(
            &current_bids,
            &bid_levels,
            Side::Buy,
            &dynamic_config,
            Some(&self.tier1.queue_tracker),
        );

        let ask_acts = priority_based_matching(
            &current_asks,
            &ask_levels,
            Side::Sell,
            &dynamic_config,
            Some(&self.tier1.queue_tracker),
        );

        let (mut bid_actions, bid_stats, mut ask_actions, ask_stats): (
            Vec<LadderAction>,
            Option<ReconcileStats>,
            Vec<LadderAction>,
            Option<ReconcileStats>,
        ) = (bid_acts, None, ask_acts, None);

        // Track impulse stats for metrics (priority_based_matching doesn't produce stats)
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

        // DIAGNOSTIC: Always log reconciliation results at INFO level to debug order placement issues
        // This helps identify if Place actions are generated but silently dropped
        info!(
            raw_bid_actions = bid_actions.len(),
            raw_ask_actions = ask_actions.len(),
            bid_cancel = bid_cancels.len(),
            bid_modify = bid_modifies.len(),
            bid_place = bid_places.len(),
            ask_cancel = ask_cancels.len(),
            ask_modify = ask_modifies.len(),
            ask_place = ask_places.len(),
            "[Reconcile] Actions generated and partitioned"
        );

        // Execute cancels first (bulk cancel is efficient)
        let all_cancels: Vec<u64> = bid_cancels.into_iter().chain(ask_cancels).collect();
        if !all_cancels.is_empty() {
            self.initiate_bulk_cancel(all_cancels.clone()).await;
            // Sync ws_state by removing cancelled orders
            for oid in &all_cancels {
                self.ws_state.remove_order(*oid);
            }
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
                            } else {
                                // ORPHAN FIX: replace_oid failed (old order not found in tracking,
                                // e.g., was already cleaned up). But the modify SUCCEEDED on exchange,
                                // so we have a new order with new OID that we're not tracking!
                                // Create a new tracked order to prevent orphan.
                                warn!(
                                    old_oid = spec.oid,
                                    new_oid = result.oid,
                                    "replace_oid failed but modify succeeded - creating new tracked order to prevent orphan"
                                );
                                let side = if spec.is_buy {
                                    tracking::Side::Buy
                                } else {
                                    tracking::Side::Sell
                                };
                                let new_order = tracking::TrackedOrder::new(
                                    result.oid,
                                    side,
                                    spec.new_price,
                                    spec.new_size,
                                );
                                self.orders.add_order(new_order);
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
                                true, // post_only
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
    pub(crate) async fn place_bulk_ladder_orders(
        &mut self,
        side: Side,
        orders: Vec<(f64, f64)>, // (price, size) tuples
    ) -> Result<()> {
        let is_buy = side == Side::Buy;
        let side_str = if is_buy { "bid" } else { "ask" };
        let input_count = orders.len();

        // FIX: Get total available capacity from exchange limits UPFRONT
        // This is the key fix - we track remaining capacity directly instead of
        // using calculate_safe_order_size which doesn't handle cumulative properly
        //
        // CRITICAL: Also check is_initialized() - if not initialized, the available_buy/sell
        // values will be f64::MAX, which would pass all orders through without any filtering.
        let limits_initialized = self.infra.exchange_limits.is_initialized();
        let total_available = if !limits_initialized {
            // Not initialized - be conservative and block all orders
            // This prevents mass rejections when we don't know actual limits
            0.0
        } else if is_buy {
            self.infra.exchange_limits.available_buy()
        } else {
            self.infra.exchange_limits.available_sell()
        };

        info!(
            side = %side_str,
            input_orders = input_count,
            total_available = %format!("{:.6}", total_available),
            limits_initialized = limits_initialized,
            "[PlaceBulk] Entering place_bulk_ladder_orders"
        );

        // EARLY EXIT: No capacity at all OR limits not initialized
        if total_available < EPSILON {
            warn!(
                side = %side_str,
                available = %format!("{:.6}", total_available),
                requested_orders = input_count,
                limits_initialized = limits_initialized,
                "[PlaceBulk] Skipping ALL orders: no exchange capacity or limits not initialized"
            );
            return Ok(());
        }

        let mut order_specs: Vec<OrderSpec> = Vec::new();
        let mut cumulative_size = 0.0; // Track cumulative exposure for exchange limits
        let mut orders_blocked_by_capacity = 0usize;

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

            let mut truncated_size =
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

            // FIX: Exchange limits pre-flight check using REMAINING capacity
            // Compare directly against remaining capacity instead of calling
            // calculate_safe_order_size which doesn't understand cumulative tracking
            let remaining_capacity = total_available - cumulative_size;

            if remaining_capacity < EPSILON {
                // No more capacity - skip all remaining orders
                orders_blocked_by_capacity += 1;
                debug!(
                    side = %side_str,
                    price = price,
                    requested_size = truncated_size,
                    cumulative = cumulative_size,
                    total_available = total_available,
                    "[PlaceBulk] Order blocked: no remaining capacity"
                );
                continue;
            }

            // Clamp to remaining capacity if needed
            if truncated_size > remaining_capacity {
                let old_size = truncated_size;
                truncated_size = truncate_float(remaining_capacity, self.config.sz_decimals, false);
                if truncated_size <= 0.0 {
                    orders_blocked_by_capacity += 1;
                    continue;
                }
                // Re-check notional after clamping
                let clamped_notional = truncated_size * price;
                if clamped_notional < MIN_ORDER_NOTIONAL {
                    orders_blocked_by_capacity += 1;
                    continue;
                }
                debug!(
                    side = %side_str,
                    price = price,
                    original_size = old_size,
                    clamped_size = truncated_size,
                    remaining_capacity = remaining_capacity,
                    "[PlaceBulk] Order size clamped to remaining capacity"
                );
            }

            cumulative_size += truncated_size;

            let cloid = uuid::Uuid::new_v4().to_string();
            order_specs.push(OrderSpec::with_cloid(
                price,
                truncated_size,
                is_buy,
                cloid,
                true, // post_only (ALO)
            ));
        }

        // DIAGNOSTIC: Log how many orders passed filtering
        info!(
            side = %side_str,
            input_orders = input_count,
            orders_passed_filter = order_specs.len(),
            orders_filtered_out = input_count - order_specs.len(),
            orders_blocked_by_capacity = orders_blocked_by_capacity,
            cumulative_size = %format!("{:.6}", cumulative_size),
            total_available = %format!("{:.6}", total_available),
            "[PlaceBulk] Order filtering complete"
        );

        if order_specs.is_empty() {
            warn!(
                side = %side_str,
                input_orders = input_count,
                "[PlaceBulk] All orders filtered out - NOT PLACING ANY ORDERS"
            );
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

        // Track batch rejection metrics (same fix as place_ladder_for_side)
        let mut rejection_count: u32 = 0;
        let mut first_rejection_error: Option<String> = None;
        let mut had_success = false;
        let mut recovery_escalated = false;

        // Finalize pending orders with proper state handling
        for (i, result) in results.iter().enumerate() {
            let spec = &order_specs[i];

            // Handle success vs rejection for rate limiting and recovery
            if result.oid > 0 {
                had_success = true;
                self.infra.recovery_manager.record_success();
            } else if let Some(ref err) = result.error {
                // Count rejections for batch processing (rate limiter)
                rejection_count += 1;
                if first_rejection_error.is_none() {
                    first_rejection_error = Some(err.clone());
                }
                // Trigger reconciliation for position-related rejections
                self.infra.reconciler.on_order_rejection(err);
                // Record for recovery state machine
                if self.infra.recovery_manager.record_rejection(is_buy, err) {
                    recovery_escalated = true;
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

        // After loop: Record batch rejection for rate limiting (once per batch, not per order)
        if rejection_count > 0 {
            if let Some(ref err) = first_rejection_error {
                if let Some(backoff) =
                    self.infra
                        .rate_limiter
                        .record_batch_rejection(is_buy, err, rejection_count)
                {
                    warn!(
                        side = %if is_buy { "buy" } else { "sell" },
                        backoff_secs = %format!("{:.1}", backoff.as_secs_f64()),
                        rejection_count = rejection_count,
                        error = %err,
                        "Rate limiter entering backoff due to batch rejection"
                    );
                }
            }
            if recovery_escalated {
                warn!(
                    side = %if is_buy { "buy" } else { "sell" },
                    "Recovery manager escalating to IOC recovery mode"
                );
            }
        }
        // Record success for rate limiter (resets counter)
        if had_success {
            self.infra.rate_limiter.record_success(is_buy);
        }

        Ok(())
    }
}
