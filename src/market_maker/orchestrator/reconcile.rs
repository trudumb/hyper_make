//! Order reconciliation and ladder management.
//!
//! Handles matching current orders to target quotes and generating the
//! appropriate cancel/modify/place actions.

use tracing::{debug, info, warn};

use crate::helpers::truncate_float;
use crate::prelude::Result;

use crate::EPSILON;

use super::super::{
    tracking, CancelResult, MarketMaker, ModifySpec, TradingEnvironment, OrderSpec, OrderState, Quote,
    QuotingStrategy, Side, TrackedOrder,
};
use super::super::infra::RejectionErrorType;
use super::{partition_ladder_actions, side_str};

/// Minimum order notional value in USD (Hyperliquid requirement)
pub(crate) const MIN_ORDER_NOTIONAL: f64 = 10.0;

/// Compute adaptive quote latch threshold based on quoted spread and API quota headroom.
///
impl<S: QuotingStrategy, Env: TradingEnvironment> MarketMaker<S, Env> {
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
    /// Superseded by `reconcile_unified()` — kept for fallback/reference.
    #[allow(dead_code)]
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
            // Phase 5: Also make calibration predictions (fill + AS) for each order.
            let mut cloids_for_tracker: Vec<String> = Vec::with_capacity(order_specs.len());
            let regime_str = format!("{:?}", self.estimator.volatility_regime());
            let mid_price = self.latest_mid;

            for spec in &order_specs {
                // Calculate depth from mid in bps (needed for fill prediction)
                let depth_bps = if mid_price > 0.0 {
                    ((spec.price - mid_price).abs() / mid_price) * 10000.0
                } else {
                    0.0
                };

                // Make calibration predictions for this order
                let (fill_prob, fill_pred_id) = self
                    .stochastic
                    .model_calibration
                    .fill_model
                    .predict_with_regime(depth_bps, &regime_str);
                let (_as_alpha, as_pred_id) = self
                    .stochastic
                    .model_calibration
                    .as_model
                    .predict_with_regime(&regime_str);

                if let Some(ref cloid) = spec.cloid {
                    // Use calibration-aware registration
                    self.orders.add_pending_with_calibration(
                        side,
                        spec.price,
                        spec.size,
                        cloid.clone(),
                        Some(fill_pred_id),
                        Some(as_pred_id),
                        Some(depth_bps),
                        mid_price,
                    );
                    cloids_for_tracker.push(cloid.clone());

                    debug!(
                        cloid = %cloid,
                        side = ?side,
                        price = spec.price,
                        fill_prob = %format!("{:.3}", fill_prob),
                        fill_pred_id = fill_pred_id,
                        as_pred_id = as_pred_id,
                        depth_bps = %format!("{:.2}", depth_bps),
                        "Registered pending order with calibration predictions"
                    );
                } else {
                    // Fallback (shouldn't happen with new code)
                    self.orders.add_pending(side, spec.price, spec.size, mid_price);
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
                .environment
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
                                self.latest_mid,
                            )
                        } else {
                            TrackedOrder::new(result.oid, side, spec.price, spec.size, self.latest_mid)
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

                    // Initialize queue tracking with L2-derived depth estimate.
                    // Depth = total size ahead of us at our price level from the cached L2 book.
                    let is_buy = side == Side::Buy;
                    let depth_ahead = self.tier1.queue_tracker.estimate_depth_at_price(spec.price, is_buy);
                    self.tier1.queue_tracker.order_placed(
                        result.oid,
                        spec.price,
                        result.resting_size,
                        depth_ahead,
                        is_buy,
                    );
                }
            }

            // After loop: Record batch rejection for rate limiting (once per batch, not per order)
            // This prevents 25 rejections from immediately hitting 120s backoff.
            if rejection_count > 0 {
                if let Some(ref err) = first_rejection_error {
                    // Check if this is a rate limit error - triggers kill switch counter
                    let error_type = RejectionErrorType::classify(err);
                    if error_type.is_rate_limit() {
                        self.safety.kill_switch.record_rate_limit_error();
                        warn!(
                            side = %side_str(side),
                            error = %err,
                            "Rate limit error detected - kill switch counter incremented"
                        );
                    }

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
    ///
    /// Superseded by economic scoring in `reconcile_unified()`.
    #[allow(dead_code)]
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

        // Compute adaptive latch threshold from quoted half-spread
        let quoted_half_spread_bps = if !new_quotes.is_empty() && self.latest_mid > 0.0 {
            let innermost_price = new_quotes[0].price;
            ((innermost_price - self.latest_mid).abs() / self.latest_mid * 10_000.0).max(1.0)
        } else {
            3.0 // Conservative fallback
        };
        let headroom_pct = self
            .infra
            .cached_rate_limit
            .as_ref()
            .map(|c| c.headroom_pct())
            .unwrap_or(1.0);
        // Inline latch computation (spread-proportional with quota widening).
        let latch_bps = {
            let base = quoted_half_spread_bps * 0.3;
            if headroom_pct < 0.30 {
                (base * 2.0).clamp(3.0, 15.0)
            } else {
                base.clamp(1.0, 10.0)
            }
        };

        // Check each level for meaningful price/size changes
        for (order, quote) in sorted_current.iter().zip(new_quotes.iter()) {
            // Price change check
            let price_diff_bps = ((order.price - quote.price) / order.price).abs() * 10000.0;

            // QUOTE LATCHING: Skip tiny changes to preserve queue position
            // Adaptive threshold proportional to quoted spread
            if price_diff_bps <= latch_bps {
                // Size change within latch window - also check size tolerance
                if order.size > 0.0 {
                    let size_diff_pct = ((order.size - quote.size) / order.size).abs();
                    if size_diff_pct <= 0.10 {
                        // Both price and size within latch threshold - preserve this order
                        continue;
                    }
                } else {
                    continue; // Price within threshold, skip
                }
            }

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
    ///
    /// Superseded by `reconcile_unified()` — kept for fallback/reference.
    #[allow(dead_code)]
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

            // Only hard-block at 1% headroom (truly exhausted).
            // Previous 10% threshold caused 94% zero-action cycles. The budget
            // allocator now handles quota conservation; reconciler should always
            // attempt to maintain market presence.
            if let Some(ref cache) = self.infra.cached_rate_limit {
                let headroom = cache.headroom_pct();
                let stale = cache.is_stale(std::time::Duration::from_secs(60));

                if headroom < 0.01 && !stale {
                    debug!(
                        headroom_pct = %format!("{:.1}%", headroom * 100.0),
                        "Rate limit headroom critically low (<1%) - hard block"
                    );
                    return Ok(());
                }
            }
        }

        // Convert Quote to LadderLevel for reconciliation
        let mut bid_levels: Vec<LadderLevel> = bid_quotes
            .iter()
            .map(|q| LadderLevel {
                price: q.price,
                size: q.size,
                depth_bps: 0.0, // Not used for reconciliation
            })
            .collect();

        let mut ask_levels: Vec<LadderLevel> = ask_quotes
            .iter()
            .map(|q| LadderLevel {
                price: q.price,
                size: q.size,
                depth_bps: 0.0,
            })
            .collect();

        // === PRE-PLACEMENT BBO CROSSING VALIDATION ===
        // From microstructure theory: GLFT quotes are derived from mid price at time T.
        // If the actual BBO shifts between quote calculation and placement, a post-only
        // bid can cross the exchange ask (or vice versa), causing rejection. The surviving
        // side then creates one-sided directional exposure — the opposite of market making.
        //
        // Defense: Validate ALL quotes against the LATEST cached BBO. If any quote would
        // cross, skip the ENTIRE quote cycle (both sides) to prevent asymmetric exposure.
        {
            let book_age = self.last_l2_update_time.elapsed();
            let bbo_valid = self.cached_best_bid > 0.0 && self.cached_best_ask > 0.0;

            // Maximum acceptable L2 book age before we refuse to quote
            const MAX_BOOK_AGE_SECS: u64 = 5;
            // Staleness buffer: widen validation margin when book is aging (2+ seconds)
            const STALENESS_BUFFER_SECS: u64 = 2;

            if bbo_valid {
                // Book staleness gate: if L2 data is too old, skip entire quote cycle
                if book_age.as_secs() >= MAX_BOOK_AGE_SECS {
                    warn!(
                        book_age_ms = book_age.as_millis() as u64,
                        max_age_ms = MAX_BOOK_AGE_SECS * 1000,
                        "Skipping quote cycle: L2 book data too stale for safe order placement"
                    );
                    return Ok(());
                }

                // Calculate staleness buffer: 1 tick per second of book age beyond threshold.
                // Use actual exchange tick size from config.decimals (10^-decimals).
                // Old proxy (mid * 0.0001) was 31x too large for HYPE at $31 with tick=0.0001.
                let actual_tick = 10f64.powi(-(self.config.decimals as i32));
                let safety_margin = actual_tick * 2.0; // 2 ticks
                let staleness_ticks = if book_age.as_secs() >= STALENESS_BUFFER_SECS {
                    (book_age.as_secs() - STALENESS_BUFFER_SECS + 1) as f64
                } else {
                    0.0
                };
                let staleness_buffer = staleness_ticks * actual_tick;

                // Validate: bid must be strictly below exchange best ask (with buffer)
                let effective_ask_limit = self.cached_best_ask - safety_margin - staleness_buffer;
                // Validate: ask must be strictly above exchange best bid (with buffer)
                let effective_bid_limit = self.cached_best_bid + safety_margin + staleness_buffer;

                // Filter crossing levels instead of skipping the entire cycle.
                // Previously, ANY crossing level caused the whole cycle to be skipped,
                // which with tight spreads (2-3 bps) meant 0 fills for minutes.
                // Now we remove only the offending levels and continue with the rest.
                let bid_count_before = bid_levels.len();
                let ask_count_before = ask_levels.len();

                bid_levels.retain(|l| l.price < effective_ask_limit);
                ask_levels.retain(|l| l.price > effective_bid_limit);

                let bids_filtered = bid_count_before - bid_levels.len();
                let asks_filtered = ask_count_before - ask_levels.len();

                if bids_filtered > 0 || asks_filtered > 0 {
                    warn!(
                        bids_filtered = bids_filtered,
                        asks_filtered = asks_filtered,
                        bids_remaining = bid_levels.len(),
                        asks_remaining = ask_levels.len(),
                        cached_best_bid = %format!("{:.6}", self.cached_best_bid),
                        cached_best_ask = %format!("{:.6}", self.cached_best_ask),
                        effective_bid_limit = %format!("{:.6}", effective_bid_limit),
                        effective_ask_limit = %format!("{:.6}", effective_ask_limit),
                        book_age_ms = book_age.as_millis() as u64,
                        "BBO crossing: filtered crossing levels instead of skipping cycle"
                    );
                    self.infra.prometheus.record_bbo_crossing_skip();
                }

                // Only skip if BOTH sides are fully empty after filtering
                if bid_levels.is_empty() && ask_levels.is_empty() {
                    warn!(
                        cached_best_bid = %format!("{:.6}", self.cached_best_bid),
                        cached_best_ask = %format!("{:.6}", self.cached_best_ask),
                        latest_mid = %format!("{:.6}", self.latest_mid),
                        book_age_ms = book_age.as_millis() as u64,
                        "Skipping quote cycle: ALL levels filtered by BBO crossing"
                    );
                    return Ok(());
                }
            }
        }

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

        // === EMPTY LADDER RECOVERY (QUOTA-AWARE) ===
        // When local tracking is completely empty but we have target quotes,
        // this is a CRITICAL recovery situation. However, we MUST respect rate
        // limits to avoid death spiral: empty → aggressive recovery → burn quota
        // → fail → empty → repeat.
        let ladder_empty = current_bids.is_empty() && current_asks.is_empty();
        let has_targets = !bid_levels.is_empty() || !ask_levels.is_empty();

        if ladder_empty && has_targets {
            // Empty ladder cooldown: prevent churn from repeated recovery attempts
            const EMPTY_LADDER_COOLDOWN_SECS: u64 = 2;
            if let Some(last_recovery) = self.last_empty_ladder_recovery {
                if last_recovery.elapsed().as_secs() < EMPTY_LADDER_COOLDOWN_SECS {
                    debug!(
                        cooldown_remaining_ms = (EMPTY_LADDER_COOLDOWN_SECS * 1000).saturating_sub(last_recovery.elapsed().as_millis() as u64),
                        "EMPTY LADDER recovery in cooldown - skipping to reduce churn"
                    );
                    return Ok(());
                }
            }

            // QUOTA-AWARE CHECK: Get current headroom before deciding recovery strategy
            let headroom = self
                .infra
                .cached_rate_limit
                .as_ref()
                .map(|c| c.headroom_pct())
                .unwrap_or(1.0);

            // Check if we're in cumulative quota backoff
            let in_backoff = self.infra.proactive_rate_tracker.is_cumulative_backoff();

            if in_backoff {
                // CRITICAL: Do NOT burn quota while in backoff - let fills accumulate
                let remaining = self
                    .infra
                    .proactive_rate_tracker
                    .remaining_cumulative_backoff();
                warn!(
                    headroom_pct = %format!("{:.1}%", headroom * 100.0),
                    backoff_remaining_secs = %format!("{:.1}", remaining.map(|d| d.as_secs_f64()).unwrap_or(0.0)),
                    "EMPTY LADDER but in quota backoff - SKIPPING recovery to break death spiral"
                );
                return Ok(());
            }

            if headroom < 0.01 {
                // CRITICAL: Quota truly exhausted (<1%) - hard block
                // Continuous shadow pricing handles 1-10% naturally, so only block
                // at the absolute minimum. This prevents the death spiral without
                // the cliff effect of the old 5% threshold.
                let backoff = self
                    .infra
                    .proactive_rate_tracker
                    .record_cumulative_exhaustion();
                warn!(
                    headroom_pct = %format!("{:.1}%", headroom * 100.0),
                    backoff_secs = %format!("{:.1}", backoff.as_secs_f64()),
                    "EMPTY LADDER but quota truly exhausted (<1%) - entering conservation mode"
                );
                return Ok(());
            }

            // Prepare levels for placement
            let mut bid_places: Vec<(f64, f64)> =
                bid_levels.iter().map(|l| (l.price, l.size)).collect();
            let mut ask_places: Vec<(f64, f64)> =
                ask_levels.iter().map(|l| (l.price, l.size)).collect();

            // CONTINUOUS LADDER DENSITY: Smoothly scale levels based on headroom.
            // Replaces hard tier cliff (was: <20% → 2 levels, >=20% → full).
            // Uses sqrt scaling: at 25% headroom → ~half levels, at 4% → ~1/5th.
            let max_target_levels = bid_places.len().max(ask_places.len());
            let policy = &self.capital_policy;
            let allowed_levels = self
                .stochastic
                .quote_gate
                .continuous_ladder_levels(
                    max_target_levels,
                    headroom,
                    policy.quota_density_scaling,
                    Some(policy.quota_min_headroom_for_full),
                );
            bid_places.truncate(allowed_levels);
            ask_places.truncate(allowed_levels);
            warn!(
                headroom_pct = %format!("{:.1}%", headroom * 100.0),
                max_target_levels = max_target_levels,
                allowed_levels = allowed_levels,
                bid_levels = bid_places.len(),
                ask_levels = ask_places.len(),
                "EMPTY LADDER recovery - continuous ladder density applied"
            );

            // Delegate to place_bulk_ladder_orders for proper CLOID tracking
            // This prevents orphan orders by using the pending registration flow
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
            self.last_empty_ladder_recovery = Some(std::time::Instant::now());

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

            // Only hard-throttle if headroom < 1% AND cache is fresh.
            // Previously was 5%, but continuous shadow pricing + ladder density
            // now handles 1-10% headroom smoothly without cliff effects.
            // If cache is stale (>60s), proceed anyway to refresh it - this prevents deadlock
            // where no requests → stale cache → perpetual throttle
            if let Some(ref cache) = self.infra.cached_rate_limit {
                let headroom = cache.headroom_pct();
                let stale = cache.is_stale(std::time::Duration::from_secs(60));

                if headroom < 0.01 {
                    if stale {
                        info!(
                            headroom_pct = %format!("{:.1}%", headroom * 100.0),
                            cache_age_secs = cache.fetched_at.elapsed().as_secs(),
                            used = cache.n_requests_used,
                            cap = cache.n_requests_cap,
                            "Rate limit cache stale (>60s) - proceeding to refresh and break potential deadlock"
                        );
                        // Don't return - proceed to make a request which will refresh the cache
                    } else {
                        warn!(
                            headroom_pct = %format!("{:.1}%", headroom * 100.0),
                            used = cache.n_requests_used,
                            cap = cache.n_requests_cap,
                            "Exchange rate limit truly exhausted (<1%) - hard throttling reconciliation"
                        );
                        return Ok(());
                    }
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

        // === CONTINUOUS LADDER DENSITY ===
        // Smoothly scale ladder levels based on rate limit headroom.
        // At high headroom: full levels. At low headroom: fewer levels.
        // This replaces discrete tier cliffs with smooth sqrt scaling.
        {
            let headroom = self
                .infra
                .cached_rate_limit
                .as_ref()
                .map(|c| c.headroom_pct())
                .unwrap_or(1.0);
            let max_target = bid_levels.len().max(ask_levels.len());
            let policy = &self.capital_policy;
            let allowed = self
                .stochastic
                .quote_gate
                .continuous_ladder_levels(
                    max_target,
                    headroom,
                    policy.quota_density_scaling,
                    Some(policy.quota_min_headroom_for_full),
                );
            if allowed < max_target {
                bid_levels.truncate(allowed);
                ask_levels.truncate(allowed);
                debug!(
                    headroom_pct = %format!("{:.1}%", headroom * 100.0),
                    max_target = max_target,
                    allowed = allowed,
                    "Continuous ladder density: reduced levels for quota conservation"
                );
            }
        }

        // === PRIORITY-BASED MATCHING (DEFAULT) ===
        // Uses stochastic optimal spread to derive matching thresholds
        // Ensures best bid/ask levels are always covered first

        // Get sigma from queue tracker (updated from book data)
        let sigma = self.tier1.queue_tracker.sigma();

        // Use actual gamma/kappa from cached market params when available.
        // Falls back to conservative defaults during early warmup before first quote cycle.
        // IMPORTANT: gamma must include regime_gamma_multiplier to match ladder strategy.
        // Without it, the reconciler uses base gamma (e.g. 0.3) while the ladder uses
        // regime-adjusted gamma (e.g. 0.22), causing tolerance mismatches.
        let (gamma, kappa) = self
            .cached_market_params
            .as_ref()
            .map(|mp| {
                // Prefer adaptive (learned) values, fall back to base estimator values
                let g_base = if mp.adaptive_gamma > 0.0 {
                    mp.adaptive_gamma
                } else {
                    mp.calibration_gamma_mult.max(0.01)
                };
                // Apply regime multiplier to match ladder strategy's gamma
                let g = g_base * mp.regime_gamma_multiplier;
                let k = if mp.use_kappa_robust && mp.kappa_robust > 0.0 {
                    mp.kappa_robust
                } else if mp.adaptive_kappa > 0.0 {
                    mp.adaptive_kappa
                } else if mp.regime_kappa.unwrap_or(0.0) > 0.0 {
                    mp.regime_kappa.unwrap_or(0.0)
                } else {
                    mp.kappa.max(1.0)
                };
                (g, k)
            })
            .unwrap_or((0.1, 1500.0));

        // Tick size in bps for minimum threshold floor
        let tick_bps = self
            .cached_market_params
            .as_ref()
            .map(|mp| mp.tick_size_bps)
            .unwrap_or(0.5);

        // API headroom for adaptive latch widening
        let headroom_pct = self
            .infra
            .cached_rate_limit
            .as_ref()
            .map(|c| c.headroom_pct())
            .unwrap_or(1.0);

        let dynamic_config = DynamicReconcileConfig::from_market_params_with_context(
            gamma,
            kappa,
            sigma,
            reconcile_config.queue_horizon_seconds,
            tick_bps,
            headroom_pct,
        );

        info!(
            gamma = %format!("{:.4}", gamma),
            kappa = %format!("{:.0}", kappa),
            sigma = %format!("{:.6}", sigma),
            optimal_spread_bps = %format!("{:.2}", dynamic_config.optimal_spread_bps),
            best_tolerance_bps = %format!("{:.2}", dynamic_config.best_level_tolerance_bps),
            outer_tolerance_bps = %format!("{:.2}", dynamic_config.outer_level_tolerance_bps),
            latch_bps = %format!("{:.2}", dynamic_config.latch_threshold_bps),
            "[Reconcile] Priority-based matching with dynamic thresholds"
        );

        // Build price grid for churn reduction (only when enabled)
        let price_grid = if self.price_grid_config.enabled && self.latest_mid > 0.0 {
            let tick_price = self.latest_mid * tick_bps / 10_000.0;
            // sigma is per-second fraction; convert to 1-minute bps
            let sigma_bps_1m = sigma * 10_000.0 * 60.0_f64.sqrt();
            Some(crate::market_maker::quoting::PriceGrid::for_current_state(
                self.latest_mid,
                tick_price,
                sigma_bps_1m,
                &self.price_grid_config,
                headroom_pct,
            ))
        } else {
            None
        };

        let (bid_acts, bid_latched) = priority_based_matching(
            &current_bids,
            &bid_levels,
            Side::Buy,
            &dynamic_config,
            Some(&self.tier1.queue_tracker),
            self.config.sz_decimals,
            price_grid.as_ref(),
        );

        let (ask_acts, ask_latched) = priority_based_matching(
            &current_asks,
            &ask_levels,
            Side::Sell,
            &dynamic_config,
            Some(&self.tier1.queue_tracker),
            self.config.sz_decimals,
            price_grid.as_ref(),
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

            // Check skipped bids for queue refresh (but never override deliberate latches)
            for order in &current_bids {
                if !bid_action_oids.contains(&order.oid) && !bid_latched.contains(&order.oid) {
                    // This order was skipped and NOT latched - check if queue tracker says refresh
                    if self
                        .tier1
                        .queue_tracker
                        .should_refresh(order.oid, reconcile_config.queue_horizon_seconds)
                    {
                        debug!(
                            oid = order.oid,
                            price = %order.price,
                            "Queue-aware: forcing bid refresh (genuinely stale)"
                        );
                        // Add cancel action for this skipped order
                        bid_actions.push(LadderAction::Cancel { oid: order.oid });
                        // Find closest target level to place (using bps tolerance, not EPSILON)
                        // The target price may differ from the current order price by several bps
                        // after ladder recalculation, so EPSILON matching is too tight.
                        if let Some(closest) = bid_levels
                            .iter()
                            .min_by_key(|l| crate::bps_diff(l.price, order.price))
                        {
                            if crate::bps_diff(closest.price, order.price) as f64
                                <= dynamic_config.best_level_tolerance_bps
                            {
                                bid_actions.push(LadderAction::Place {
                                    side: Side::Buy,
                                    price: closest.price,
                                    size: closest.size,
                                });
                            }
                        }
                    }
                }
            }

            // Check skipped asks for queue refresh (but never override deliberate latches)
            for order in &current_asks {
                if !ask_action_oids.contains(&order.oid)
                    && !ask_latched.contains(&order.oid)
                    && self
                        .tier1
                        .queue_tracker
                        .should_refresh(order.oid, reconcile_config.queue_horizon_seconds)
                {
                    debug!(
                        oid = order.oid,
                        price = %order.price,
                        "Queue-aware: forcing ask refresh (genuinely stale)"
                    );
                    ask_actions.push(LadderAction::Cancel { oid: order.oid });
                    // Find closest target level to place (using bps tolerance)
                    if let Some(closest) = ask_levels
                        .iter()
                        .min_by_key(|l| crate::bps_diff(l.price, order.price))
                    {
                        if crate::bps_diff(closest.price, order.price) as f64
                            <= dynamic_config.best_level_tolerance_bps
                        {
                            ask_actions.push(LadderAction::Place {
                                side: Side::Sell,
                                price: closest.price,
                                size: closest.size,
                            });
                        }
                    }
                }
            }
        }

        // === L5: Budget-Aware Filtering ===
        // Filter low-value operations when API budget is tight.
        // Emergency actions (cancels during cascade) always pass.
        let budget_suppressed;
        {
            use crate::market_maker::infra::{BudgetPacer, OperationPriority};

            let classify_action = |action: &LadderAction, idx: usize| -> OperationPriority {
                match action {
                    LadderAction::Cancel { .. } => {
                        BudgetPacer::priority_for_cancel(false, false)
                    }
                    LadderAction::Modify { .. } => {
                        // Modifies preserve queue position — always at least Normal priority.
                        OperationPriority::Normal
                    }
                    LadderAction::Place { .. } => {
                        BudgetPacer::priority_for_placement(idx == 0)
                    }
                }
            };

            let budget_pre_bid = bid_actions.len();
            let budget_pre_ask = ask_actions.len();
            bid_actions = bid_actions
                .into_iter()
                .enumerate()
                .filter(|(i, a)| self.infra.budget_pacer.should_spend(classify_action(a, *i)))
                .map(|(_, a)| a)
                .collect();
            ask_actions = ask_actions
                .into_iter()
                .enumerate()
                .filter(|(i, a)| self.infra.budget_pacer.should_spend(classify_action(a, *i)))
                .map(|(_, a)| a)
                .collect();
            budget_suppressed = (budget_pre_bid + budget_pre_ask) - (bid_actions.len() + ask_actions.len());
            if budget_suppressed > 0 {
                info!(
                    budget_suppressed,
                    remaining_budget = self.infra.budget_pacer.remaining_budget(),
                    "L5: Budget pacer suppressed low-value actions"
                );
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
            // Sync ws_state by removing cancelled orders, recording tombstones for fill attribution
            for oid in &all_cancels {
                if let Some(order) = self.ws_state.remove_order(*oid) {
                    self.safety.fill_processor.record_cancelled_order(
                        *oid, order.side, order.price, order.size,
                    );
                }
            }
        }

        // Execute modifies (preserves queue position)
        // RATE LIMIT FIX: Check modify debounce before executing
        // Filter modify specs that would cross the BBO (defense-in-depth)
        let actual_tick = 10f64.powi(-(self.config.decimals as i32));
        let safety_margin = actual_tick * 2.0; // 2 ticks
        let bbo_valid = self.cached_best_bid > 0.0 && self.cached_best_ask > 0.0;
        let all_modifies: Vec<ModifySpec> = bid_modifies
            .into_iter()
            .chain(ask_modifies)
            .filter(|spec| {
                if !bbo_valid {
                    return true; // Can't validate without BBO
                }
                if spec.is_buy && spec.new_price >= self.cached_best_ask - safety_margin {
                    warn!(
                        oid = spec.oid,
                        new_price = %format!("{:.6}", spec.new_price),
                        exchange_ask = %format!("{:.6}", self.cached_best_ask),
                        "Filtering modify: bid would cross exchange ask"
                    );
                    return false;
                }
                if !spec.is_buy && spec.new_price <= self.cached_best_bid + safety_margin {
                    warn!(
                        oid = spec.oid,
                        new_price = %format!("{:.6}", spec.new_price),
                        exchange_bid = %format!("{:.6}", self.cached_best_bid),
                        "Filtering modify: ask would cross exchange bid"
                    );
                    return false;
                }
                true
            })
            .collect();
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
                info!(
                    modify_count = all_modifies.len(),
                    "Modify debounced: latching {} orders (preserving resting state and queue position)",
                    all_modifies.len()
                );
                // Latch: keep resting orders untouched until next modify window.
                // Do NOT fall through to cancel+place — that's 2x API cost and loses queue position.
            } else {
                // Mark that we're doing a modify
                self.infra.proactive_rate_tracker.mark_modify();

                let num_modifies = all_modifies.len() as u32;
                let modify_results = self
                    .environment
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
                                    self.latest_mid,
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
                            .environment
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

                        // Calculate depth and make calibration predictions
                        let mid_price = self.latest_mid;
                        let depth_bps = if mid_price > 0.0 {
                            ((spec.new_price - mid_price).abs() / mid_price) * 10000.0
                        } else {
                            0.0
                        };
                        let regime_str = format!("{:?}", self.estimator.volatility_regime());
                        let (_, fill_pred_id) = self
                            .stochastic
                            .model_calibration
                            .fill_model
                            .predict_with_regime(depth_bps, &regime_str);
                        let (_, as_pred_id) = self
                            .stochastic
                            .model_calibration
                            .as_model
                            .predict_with_regime(&regime_str);

                        // Pre-register as pending BEFORE API call with calibration predictions
                        self.orders.add_pending_with_calibration(
                            side,
                            spec.new_price,
                            spec.new_size,
                            cloid.clone(),
                            Some(fill_pred_id),
                            Some(as_pred_id),
                            Some(depth_bps),
                            mid_price,
                        );

                        // Phase 7: Register expected CLOID with orphan tracker BEFORE API call
                        self.infra
                            .orphan_tracker
                            .register_expected_cloids(std::slice::from_ref(&cloid));

                        // Truncate size to exchange precision before placement
                        let safe_size = truncate_float(spec.new_size, self.config.sz_decimals, false);
                        if safe_size <= 0.0 {
                            warn!(
                                original_size = %format!("{:.6}", spec.new_size),
                                sz_decimals = self.config.sz_decimals,
                                "Modify fallback: size truncated to zero, skipping placement"
                            );
                            self.orders.remove_pending_by_cloid(&cloid);
                            self.infra.orphan_tracker.mark_failed(&cloid);
                            continue;
                        }

                        // Pass the pre-registered CLOID to place_order for deterministic matching.
                        // This ensures the pending order is finalized with the correct OID.
                        let place_result = self
                            .environment
                            .place_order(
                                &self.config.asset,
                                spec.new_price,
                                safe_size,
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
                                    self.latest_mid,
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
        // NOTE: True parallel execution (tokio::join!) requires refactoring place_bulk_ladder_orders
        // to separate order building from mutable state updates. Currently sequential ~100ms each.
        // TODO: Refactor to enable parallel bid/ask placement for additional ~100ms latency reduction.
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

        // === L5b: Record API spend in BudgetPacer ===
        let total_api_calls =
            (cancel_count + modify_count + bid_place_count + ask_place_count) as u32;
        if total_api_calls > 0 {
            self.infra.budget_pacer.record_spend(total_api_calls);
        }

        // === L6: Record churn metrics ===
        {
            use crate::market_maker::analytics::churn_tracker::CycleSummary;
            let latched = _total_skips as u32;
            self.churn_tracker.record_cycle(CycleSummary {
                placed: (bid_place_count + ask_place_count) as u32,
                cancelled: cancel_count as u32,
                filled: 0, // fills tracked separately via fill processor
                modified: modify_count as u32,
                latched,
                grid_preserved: 0, // TODO: track grid-snapped preservations
                budget_suppressed: budget_suppressed as u32,
            });
        }

        Ok(())
    }

    /// Unified reconciliation using economic scoring and budget-constrained allocation.
    ///
    /// Replaces both `reconcile_ladder_side()` (batch) and the priority_based_matching path
    /// in `reconcile_ladder_smart()` with a single pipeline:
    ///
    /// ```text
    /// targets + orders → SCORE (EV-based) → ALLOCATE (greedy knapsack) → EXECUTE
    /// ```
    ///
    /// All safety checks (rate limits, BBO crossing, drift detection, empty ladder recovery)
    /// are preserved from `reconcile_ladder_smart()`. The scoring and allocation phases
    /// replace `priority_based_matching()` + `BudgetPacer` with principled economic decisions.
    pub(crate) async fn reconcile_unified(
        &mut self,
        bid_quotes: Vec<Quote>,
        ask_quotes: Vec<Quote>,
    ) -> Result<()> {
        use crate::market_maker::quoting::LadderLevel;
        use crate::market_maker::tracking::{
            score_reconcile_actions, DynamicReconcileConfig,
        };
        use super::budget_allocator::{allocate, ApiBudget};

        let reconcile_config = &self.config.reconcile;

        // === UNCONDITIONAL RATE LIMIT CHECK ===
        // (Identical to reconcile_ladder_smart — preserves all safety behavior)
        {
            use crate::market_maker::core::CachedRateLimit;
            use std::time::Duration;

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

            if let Some(ref cache) = self.infra.cached_rate_limit {
                let headroom = cache.headroom_pct();
                let stale = cache.is_stale(std::time::Duration::from_secs(60));

                if headroom < 0.01 && !stale {
                    debug!(
                        headroom_pct = %format!("{:.1}%", headroom * 100.0),
                        "Rate limit headroom critically low (<1%) - hard block"
                    );
                    return Ok(());
                }
            }
        }

        // Convert Quote to LadderLevel
        let mut bid_levels: Vec<LadderLevel> = bid_quotes
            .iter()
            .map(|q| LadderLevel {
                price: q.price,
                size: q.size,
                depth_bps: 0.0,
            })
            .collect();

        let mut ask_levels: Vec<LadderLevel> = ask_quotes
            .iter()
            .map(|q| LadderLevel {
                price: q.price,
                size: q.size,
                depth_bps: 0.0,
            })
            .collect();

        // === PRE-PLACEMENT BBO CROSSING VALIDATION ===
        // (Identical to reconcile_ladder_smart)
        {
            let book_age = self.last_l2_update_time.elapsed();
            let bbo_valid = self.cached_best_bid > 0.0 && self.cached_best_ask > 0.0;

            const MAX_BOOK_AGE_SECS: u64 = 5;
            const STALENESS_BUFFER_SECS: u64 = 2;

            if bbo_valid {
                if book_age.as_secs() >= MAX_BOOK_AGE_SECS {
                    warn!(
                        book_age_ms = book_age.as_millis() as u64,
                        max_age_ms = MAX_BOOK_AGE_SECS * 1000,
                        "Skipping quote cycle: L2 book data too stale for safe order placement"
                    );
                    return Ok(());
                }

                let actual_tick = 10f64.powi(-(self.config.decimals as i32));
                let safety_margin = actual_tick * 2.0;
                let staleness_ticks = if book_age.as_secs() >= STALENESS_BUFFER_SECS {
                    (book_age.as_secs() - STALENESS_BUFFER_SECS + 1) as f64
                } else {
                    0.0
                };
                let staleness_buffer = staleness_ticks * actual_tick;

                let effective_ask_limit = self.cached_best_ask - safety_margin - staleness_buffer;
                let effective_bid_limit = self.cached_best_bid + safety_margin + staleness_buffer;

                let bid_count_before = bid_levels.len();
                let ask_count_before = ask_levels.len();

                bid_levels.retain(|l| l.price < effective_ask_limit);
                ask_levels.retain(|l| l.price > effective_bid_limit);

                let bids_filtered = bid_count_before - bid_levels.len();
                let asks_filtered = ask_count_before - ask_levels.len();

                if bids_filtered > 0 || asks_filtered > 0 {
                    warn!(
                        bids_filtered = bids_filtered,
                        asks_filtered = asks_filtered,
                        bids_remaining = bid_levels.len(),
                        asks_remaining = ask_levels.len(),
                        "BBO crossing: filtered crossing levels instead of skipping cycle"
                    );
                    self.infra.prometheus.record_bbo_crossing_skip();
                }

                if bid_levels.is_empty() && ask_levels.is_empty() {
                    warn!(
                        "Skipping quote cycle: ALL levels filtered by BBO crossing"
                    );
                    return Ok(());
                }
            }
        }

        // Get current order counts (drop references immediately for borrow safety)
        let num_current_bids = self.orders.get_all_by_side(Side::Buy).len();
        let num_current_asks = self.orders.get_all_by_side(Side::Sell).len();

        info!(
            local_bids = num_current_bids,
            local_asks = num_current_asks,
            target_bid_levels = bid_levels.len(),
            target_ask_levels = ask_levels.len(),
            "[Unified] Order counts"
        );

        // === EMPTY LADDER RECOVERY ===
        // (Identical to reconcile_ladder_smart)
        let ladder_empty = num_current_bids == 0 && num_current_asks == 0;
        let has_targets = !bid_levels.is_empty() || !ask_levels.is_empty();

        if ladder_empty && has_targets {
            const EMPTY_LADDER_COOLDOWN_SECS: u64 = 2;
            if let Some(last_recovery) = self.last_empty_ladder_recovery {
                if last_recovery.elapsed().as_secs() < EMPTY_LADDER_COOLDOWN_SECS {
                    debug!("EMPTY LADDER recovery in cooldown - skipping");
                    return Ok(());
                }
            }

            let headroom = self
                .infra
                .cached_rate_limit
                .as_ref()
                .map(|c| c.headroom_pct())
                .unwrap_or(1.0);

            let in_backoff = self.infra.proactive_rate_tracker.is_cumulative_backoff();
            if in_backoff {
                warn!(
                    headroom_pct = %format!("{:.1}%", headroom * 100.0),
                    "EMPTY LADDER but in quota backoff - SKIPPING recovery"
                );
                return Ok(());
            }

            if headroom < 0.01 {
                let backoff = self
                    .infra
                    .proactive_rate_tracker
                    .record_cumulative_exhaustion();
                warn!(
                    headroom_pct = %format!("{:.1}%", headroom * 100.0),
                    backoff_secs = %format!("{:.1}", backoff.as_secs_f64()),
                    "EMPTY LADDER but quota exhausted (<1%) - conservation mode"
                );
                return Ok(());
            }

            let mut bid_places: Vec<(f64, f64)> =
                bid_levels.iter().map(|l| (l.price, l.size)).collect();
            let mut ask_places: Vec<(f64, f64)> =
                ask_levels.iter().map(|l| (l.price, l.size)).collect();

            let max_target_levels = bid_places.len().max(ask_places.len());
            let policy = &self.capital_policy;
            let allowed_levels = self
                .stochastic
                .quote_gate
                .continuous_ladder_levels(
                    max_target_levels,
                    headroom,
                    policy.quota_density_scaling,
                    Some(policy.quota_min_headroom_for_full),
                );
            bid_places.truncate(allowed_levels);
            ask_places.truncate(allowed_levels);

            if !bid_places.is_empty() {
                self.place_bulk_ladder_orders(Side::Buy, bid_places).await?;
            }
            if !ask_places.is_empty() {
                self.place_bulk_ladder_orders(Side::Sell, ask_places).await?;
            }

            self.last_empty_ladder_recovery = Some(std::time::Instant::now());
            return Ok(());
        }

        // === DRIFT DETECTION ===
        // (Identical to reconcile_ladder_smart — borrow scoped to avoid aliasing)
        const MAX_ACCEPTABLE_DRIFT_BPS: f64 = 100.0;

        let (bid_drift, ask_drift) = {
            let cb: Vec<&TrackedOrder> = self.orders.get_all_by_side(Side::Buy);
            let ca: Vec<&TrackedOrder> = self.orders.get_all_by_side(Side::Sell);

            let bd = if !cb.is_empty() && !bid_levels.is_empty() {
                cb.iter()
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

            let ad = if !ca.is_empty() && !ask_levels.is_empty() {
                ca.iter()
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

            (bd, ad)
        }; // borrow of self.orders dropped here

        let max_drift = bid_drift.max(ask_drift);
        if max_drift > MAX_ACCEPTABLE_DRIFT_BPS {
            warn!(
                bid_drift_bps = %format!("{:.1}", bid_drift),
                ask_drift_bps = %format!("{:.1}", ask_drift),
                "[Unified] Large drift - forcing full cancel + replace"
            );

            let all_oids: Vec<u64> = {
                let cb: Vec<&TrackedOrder> = self.orders.get_all_by_side(Side::Buy);
                let ca: Vec<&TrackedOrder> = self.orders.get_all_by_side(Side::Sell);
                cb.iter().chain(ca.iter()).map(|o| o.oid).collect()
            };

            if !all_oids.is_empty() {
                self.initiate_bulk_cancel(all_oids).await;
            }

            let bid_places: Vec<(f64, f64)> =
                bid_levels.iter().map(|l| (l.price, l.size)).collect();
            let ask_places: Vec<(f64, f64)> =
                ask_levels.iter().map(|l| (l.price, l.size)).collect();

            if !bid_places.is_empty() {
                self.place_bulk_ladder_orders(Side::Buy, bid_places).await?;
            }
            if !ask_places.is_empty() {
                self.place_bulk_ladder_orders(Side::Sell, ask_places).await?;
            }

            return Ok(());
        }

        // === IMPULSE CONTROL ===
        // (Identical to reconcile_ladder_smart)
        let impulse_enabled =
            self.infra.impulse_control_enabled && reconcile_config.use_impulse_filter;
        if impulse_enabled {
            use crate::market_maker::core::CachedRateLimit;
            use std::time::Duration;

            let cache_max_age = Duration::from_secs(60);
            let should_refresh = self
                .infra
                .cached_rate_limit
                .as_ref()
                .is_none_or(|c| c.is_stale(cache_max_age));

            if should_refresh {
                if let Ok(response) = self.info_client.user_rate_limit(self.user_address).await {
                    self.infra.cached_rate_limit =
                        Some(CachedRateLimit::from_response(&response));
                }
            }

            if let Some(ref cache) = self.infra.cached_rate_limit {
                let headroom = cache.headroom_pct();
                let stale = cache.is_stale(std::time::Duration::from_secs(60));

                if headroom < 0.01 && !stale {
                    warn!(
                        headroom_pct = %format!("{:.1}%", headroom * 100.0),
                        "Rate limit exhausted (<1%) - hard throttling"
                    );
                    return Ok(());
                }
            }
        }

        // === CONTINUOUS LADDER DENSITY ===
        {
            let headroom = self
                .infra
                .cached_rate_limit
                .as_ref()
                .map(|c| c.headroom_pct())
                .unwrap_or(1.0);
            let max_target = bid_levels.len().max(ask_levels.len());
            let policy = &self.capital_policy;
            let allowed = self
                .stochastic
                .quote_gate
                .continuous_ladder_levels(
                    max_target,
                    headroom,
                    policy.quota_density_scaling,
                    Some(policy.quota_min_headroom_for_full),
                );
            if allowed < max_target {
                bid_levels.truncate(allowed);
                ask_levels.truncate(allowed);
                debug!(
                    headroom_pct = %format!("{:.1}%", headroom * 100.0),
                    allowed = allowed,
                    "Continuous ladder density: reduced levels"
                );
            }
        }

        // =====================================================================
        // === ECONOMIC SCORING + BUDGET ALLOCATION (replaces priority_based_matching) ===
        // =====================================================================

        let sigma = self.tier1.queue_tracker.sigma();

        let (gamma, kappa) = self
            .cached_market_params
            .as_ref()
            .map(|mp| {
                let g_base = if mp.adaptive_gamma > 0.0 {
                    mp.adaptive_gamma
                } else {
                    mp.calibration_gamma_mult.max(0.01)
                };
                let g = g_base * mp.regime_gamma_multiplier;
                let k = if mp.use_kappa_robust && mp.kappa_robust > 0.0 {
                    mp.kappa_robust
                } else if mp.adaptive_kappa > 0.0 {
                    mp.adaptive_kappa
                } else if mp.regime_kappa.unwrap_or(0.0) > 0.0 {
                    mp.regime_kappa.unwrap_or(0.0)
                } else {
                    mp.kappa.max(1.0)
                };
                (g, k)
            })
            .unwrap_or((0.1, 1500.0));

        let tick_bps = self
            .cached_market_params
            .as_ref()
            .map(|mp| mp.tick_size_bps)
            .unwrap_or(0.5);

        let headroom_pct = self
            .infra
            .cached_rate_limit
            .as_ref()
            .map(|c| c.headroom_pct())
            .unwrap_or(1.0);

        let dynamic_config = DynamicReconcileConfig::from_market_params_with_context(
            gamma,
            kappa,
            sigma,
            reconcile_config.queue_horizon_seconds,
            tick_bps,
            headroom_pct,
        );

        // Toxicity regime for queue value scoring
        let ofi_1s = self.stochastic.trade_flow_tracker.imbalance_at_1s();
        let ofi_5s = self.stochastic.trade_flow_tracker.imbalance_at_5s();
        let toxicity = self
            .tier1
            .pre_fill_classifier
            .toxicity_regime(ofi_1s, ofi_5s);

        info!(
            gamma = %format!("{:.4}", gamma),
            kappa = %format!("{:.0}", kappa),
            sigma = %format!("{:.6}", sigma),
            optimal_spread_bps = %format!("{:.2}", dynamic_config.optimal_spread_bps),
            headroom_pct = %format!("{:.1}%", headroom_pct * 100.0),
            toxicity = ?toxicity,
            "[Unified] Economic scoring with dynamic thresholds"
        );

        // Score all bid + ask updates (scoped borrow of self.orders)
        let (mut bid_scored, mut ask_scored) = {
            let current_bids: Vec<&TrackedOrder> = self.orders.get_all_by_side(Side::Buy);
            let current_asks: Vec<&TrackedOrder> = self.orders.get_all_by_side(Side::Sell);

            let bs = score_reconcile_actions(
                &current_bids,
                &bid_levels,
                Side::Buy,
                &dynamic_config,
                Some(&self.tier1.queue_tracker),
                &self.queue_value_heuristic,
                toxicity,
                self.latest_mid,
                self.config.sz_decimals,
            );

            let as_ = score_reconcile_actions(
                &current_asks,
                &ask_levels,
                Side::Sell,
                &dynamic_config,
                Some(&self.tier1.queue_tracker),
                &self.queue_value_heuristic,
                toxicity,
                self.latest_mid,
                self.config.sz_decimals,
            );

            (bs, as_)
        }; // borrow of self.orders dropped here

        // Compute API budget from headroom
        let rate_cap = self
            .infra
            .cached_rate_limit
            .as_ref()
            .map(|c| c.n_requests_cap as u32)
            .unwrap_or(1200);
        // Estimate cycle interval from last cycle or use conservative default.
        // The exact value only affects budget sizing (how many calls per cycle).
        let cycle_interval_s = 5.0;
        let budget = ApiBudget::from_headroom(headroom_pct, rate_cap, cycle_interval_s);

        // Merge both sides and allocate within budget
        let mut all_scored: Vec<_> = bid_scored.drain(..).chain(ask_scored.drain(..)).collect();

        // Priority-filtered execution: when headroom is tight, drop low-value actions
        // to preserve quota for high-value ones (new placements, large price moves).
        // Low-value = small price moves (<5 bps) or size-only modifications with value < 1 bps.
        if headroom_pct < 0.20 {
            let pre_filter = all_scored.len();
            all_scored.retain(|s| {
                use crate::market_maker::tracking::ActionType;
                // Always keep: latches (0 API cost), stale cancels (cleanup), new placements
                match s.action {
                    ActionType::Latch => true,
                    ActionType::StaleCancel | ActionType::NewPlace => true,
                    // Size-only mods: keep only if value > 1 bps
                    ActionType::ModifySize => s.value_bps > 1.0,
                    // Price moves & cancel+place: keep only if value > 0.5 bps
                    ActionType::ModifyPrice | ActionType::CancelPlace => s.value_bps > 0.5,
                }
            });
            let filtered = pre_filter - all_scored.len();
            if filtered > 0 {
                info!(
                    filtered_count = filtered,
                    remaining = all_scored.len(),
                    headroom_pct = %format!("{:.1}%", headroom_pct * 100.0),
                    "[Unified] Priority-filtered low-value actions under quota pressure"
                );
            }
        }

        let allocation = allocate(&mut all_scored, &budget, self.config.sz_decimals);

        info!(
            calls_used = allocation.calls_used,
            calls_budget = allocation.calls_budget,
            latched = allocation.latched_count,
            budget_exhausted = allocation.budget_exhausted,
            total_value_bps = %format!("{:.2}", allocation.total_value_bps),
            suppressed = allocation.suppressed_count,
            actions = allocation.actions.len(),
            "[Unified] Budget allocation result"
        );

        // Partition actions by type and side for execution
        let (bid_cancels, bid_modifies, bid_places) =
            partition_ladder_actions(&allocation.actions, Side::Buy);
        let (ask_cancels, ask_modifies, ask_places) =
            partition_ladder_actions(&allocation.actions, Side::Sell);

        let cancel_count = bid_cancels.len() + ask_cancels.len();
        let modify_count = bid_modifies.len() + ask_modifies.len();

        info!(
            bid_cancel = bid_cancels.len(),
            bid_modify = bid_modifies.len(),
            bid_place = bid_places.len(),
            ask_cancel = ask_cancels.len(),
            ask_modify = ask_modifies.len(),
            ask_place = ask_places.len(),
            "[Unified] Actions partitioned"
        );

        // === EXECUTE CANCELS ===
        let all_cancels: Vec<u64> = bid_cancels.into_iter().chain(ask_cancels).collect();
        if !all_cancels.is_empty() {
            self.initiate_bulk_cancel(all_cancels.clone()).await;
            for oid in &all_cancels {
                if let Some(order) = self.ws_state.remove_order(*oid) {
                    self.safety.fill_processor.record_cancelled_order(
                        *oid, order.side, order.price, order.size,
                    );
                }
            }
        }

        // === EXECUTE MODIFIES ===
        let actual_tick = 10f64.powi(-(self.config.decimals as i32));
        let safety_margin = actual_tick * 2.0;
        let bbo_valid = self.cached_best_bid > 0.0 && self.cached_best_ask > 0.0;
        let all_modifies: Vec<ModifySpec> = bid_modifies
            .into_iter()
            .chain(ask_modifies)
            .filter(|spec| {
                if !bbo_valid {
                    return true;
                }
                if spec.is_buy && spec.new_price >= self.cached_best_ask - safety_margin {
                    warn!(oid = spec.oid, "Filtering modify: bid would cross exchange ask");
                    return false;
                }
                if !spec.is_buy && spec.new_price <= self.cached_best_bid + safety_margin {
                    warn!(oid = spec.oid, "Filtering modify: ask would cross exchange bid");
                    return false;
                }
                true
            })
            .collect();

        if !all_modifies.is_empty() {
            if self.infra.proactive_rate_tracker.is_rate_limited() {
                debug!("Skipping modify: rate limited (429 backoff active)");
            } else if !self.infra.proactive_rate_tracker.can_modify() {
                info!(
                    modify_count = all_modifies.len(),
                    "Modify debounced: latching orders"
                );
            } else {
                self.infra.proactive_rate_tracker.mark_modify();

                let num_modifies = all_modifies.len() as u32;
                let modify_results = self
                    .environment
                    .modify_bulk_orders(&self.config.asset, all_modifies.clone())
                    .await;

                self.infra
                    .proactive_rate_tracker
                    .record_call(1, num_modifies);

                let mut oid_remap_count = 0u32;
                let mut modify_success_count = 0u32;

                for (i, result) in modify_results.iter().enumerate() {
                    let spec = &all_modifies[i];
                    if result.success {
                        let effective_oid = if result.oid > 0 && result.oid != spec.oid {
                            if self.orders.replace_oid(spec.oid, result.oid) {
                                oid_remap_count += 1;
                            } else {
                                let side = if spec.is_buy { tracking::Side::Buy } else { tracking::Side::Sell };
                                let new_order = tracking::TrackedOrder::new(
                                    result.oid, side, spec.new_price, spec.new_size, self.latest_mid,
                                );
                                self.orders.add_order(new_order);
                                oid_remap_count += 1;
                            }
                            result.oid
                        } else {
                            spec.oid
                        };

                        self.orders.on_modify_success(effective_oid, spec.new_price, spec.new_size);
                        self.infra.prometheus.record_order_modified();
                        modify_success_count += 1;
                    } else {
                        warn!(
                            oid = spec.oid,
                            error = ?result.error,
                            "Modify failed, falling back to cancel+place"
                        );
                        self.infra.prometheus.record_modify_fallback();

                        let error_msg = result.error.as_deref().unwrap_or("");
                        let is_already_canceled = error_msg.contains("canceled");
                        let is_already_filled = !is_already_canceled && error_msg.contains("filled");

                        if is_already_canceled {
                            self.orders.remove_order(spec.oid);
                            continue;
                        }
                        if is_already_filled {
                            if let Some(order) = self.orders.get_order_mut(spec.oid) {
                                order.transition_to(OrderState::FilledImmediately);
                            }
                            continue;
                        }

                        // Cancel + place fallback
                        self.orders.initiate_cancel(spec.oid);
                        let cancel_result = self
                            .environment
                            .cancel_order(&self.config.asset, spec.oid)
                            .await;

                        match cancel_result {
                            CancelResult::Cancelled | CancelResult::AlreadyCancelled => {
                                self.orders.on_cancel_confirmed(spec.oid);
                            }
                            CancelResult::AlreadyFilled => {
                                self.orders.on_cancel_already_filled(spec.oid);
                                continue;
                            }
                            CancelResult::Failed => {
                                self.orders.on_cancel_failed(spec.oid);
                                continue;
                            }
                        }

                        let side = if spec.is_buy { Side::Buy } else { Side::Sell };
                        let cloid = uuid::Uuid::new_v4().to_string();
                        let mid_price = self.latest_mid;
                        let depth_bps = if mid_price > 0.0 {
                            ((spec.new_price - mid_price).abs() / mid_price) * 10000.0
                        } else {
                            0.0
                        };
                        let regime_str = format!("{:?}", self.estimator.volatility_regime());
                        let (_, fill_pred_id) = self
                            .stochastic
                            .model_calibration
                            .fill_model
                            .predict_with_regime(depth_bps, &regime_str);
                        let (_, as_pred_id) = self
                            .stochastic
                            .model_calibration
                            .as_model
                            .predict_with_regime(&regime_str);

                        self.orders.add_pending_with_calibration(
                            side, spec.new_price, spec.new_size,
                            cloid.clone(), Some(fill_pred_id), Some(as_pred_id),
                            Some(depth_bps), mid_price,
                        );
                        self.infra.orphan_tracker.register_expected_cloids(std::slice::from_ref(&cloid));

                        let safe_size = truncate_float(spec.new_size, self.config.sz_decimals, false);
                        if safe_size <= 0.0 {
                            self.orders.remove_pending_by_cloid(&cloid);
                            self.infra.orphan_tracker.mark_failed(&cloid);
                            continue;
                        }

                        let place_result = self
                            .environment
                            .place_order(
                                &self.config.asset, spec.new_price, safe_size,
                                spec.is_buy, Some(cloid.clone()), true,
                            )
                            .await;

                        if place_result.oid > 0 {
                            self.infra.orphan_tracker.record_oid_for_cloid(&cloid, place_result.oid);

                            if place_result.filled {
                                self.orders.remove_pending_by_cloid(&cloid);
                                self.infra.orphan_tracker.mark_finalized(&cloid, place_result.oid);

                                let resting_size = place_result.resting_size;
                                let actual_fill = if resting_size < crate::EPSILON {
                                    spec.new_size
                                } else {
                                    spec.new_size - resting_size
                                };

                                self.position.process_fill(actual_fill, spec.is_buy);
                                self.safety.fill_processor.pre_register_immediate_fill(place_result.oid, actual_fill);

                                let mut tracked = TrackedOrder::with_cloid(
                                    place_result.oid, cloid, side,
                                    spec.new_price, spec.new_size, self.latest_mid,
                                );
                                tracked.filled = actual_fill;

                                if resting_size > crate::EPSILON {
                                    tracked.transition_to(OrderState::PartialFilled);
                                    tracked.size = resting_size;
                                } else {
                                    tracked.transition_to(OrderState::FilledImmediately);
                                }
                                self.orders.add_order(tracked);
                            } else {
                                self.orders.finalize_pending_by_cloid(
                                    &cloid, place_result.oid, place_result.resting_size,
                                );
                                self.infra.orphan_tracker.mark_finalized(&cloid, place_result.oid);
                            }
                        } else {
                            self.orders.remove_pending_by_cloid(&cloid);
                            self.infra.orphan_tracker.mark_failed(&cloid);
                        }
                    }
                }

                if modify_success_count > 0 {
                    let (best_bid, best_ask, bid_levels_count, ask_levels_count) =
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
                        spread_bps = %format!("{:.2}", spread_bps),
                        bid_levels = bid_levels_count,
                        ask_levels = ask_levels_count,
                        position = %format!("{:.6}", self.position.position()),
                        "Quote cycle complete"
                    );
                }
            }
        }

        // === EXECUTE PLACES ===
        let bid_place_count = bid_places.len();
        let ask_place_count = ask_places.len();
        if !bid_places.is_empty() {
            self.place_bulk_ladder_orders(Side::Buy, bid_places).await?;
        }
        if !ask_places.is_empty() {
            self.place_bulk_ladder_orders(Side::Sell, ask_places).await?;
        }

        // === IMPULSE CONTROL BUDGET ===
        if impulse_enabled {
            let total_actions = cancel_count + modify_count + bid_place_count + ask_place_count;
            if total_actions > 0 {
                self.infra.execution_budget.spend(total_actions);
            }
        }

        // === RECORD API SPEND ===
        let total_api_calls =
            (cancel_count + modify_count + bid_place_count + ask_place_count) as u32;
        if total_api_calls > 0 {
            self.infra.budget_pacer.record_spend(total_api_calls);
        }

        // === CHURN METRICS ===
        {
            use crate::market_maker::analytics::churn_tracker::CycleSummary;
            let total_existing = num_current_bids + num_current_asks;
            let latched = total_existing.saturating_sub(cancel_count + modify_count) as u32;
            self.churn_tracker.record_cycle(CycleSummary {
                placed: (bid_place_count + ask_place_count) as u32,
                cancelled: cancel_count as u32,
                filled: 0,
                modified: modify_count as u32,
                latched,
                grid_preserved: 0,
                budget_suppressed: allocation.suppressed_count,
            });
        }

        // === RECORD OUTCOME DECISIONS ===
        // Record non-latch decisions in the outcome tracker for fill rate learning.
        {
            use crate::market_maker::tracking::{
                ActionType, ReconcileActionType, ReconcileDecision,
            };

            for update in &all_scored {
                // Only track actions that actually interact with the exchange
                // (modifies, cancel+places, new placements). Latches are no-ops.
                let action_type = match update.action {
                    ActionType::Latch => continue,
                    ActionType::ModifySize => ReconcileActionType::ModifySize,
                    ActionType::ModifyPrice => ReconcileActionType::ModifyPrice,
                    ActionType::CancelPlace => ReconcileActionType::CancelPlace,
                    ActionType::NewPlace => ReconcileActionType::NewPlace,
                    ActionType::StaleCancel => ReconcileActionType::StaleCancel,
                };

                if let Some(oid) = update.oid {
                    let price_drift_bps = if self.latest_mid > crate::EPSILON {
                        (update.target_price - update.current_price).abs()
                            / self.latest_mid
                            * 10_000.0
                    } else {
                        0.0
                    };

                    self.reconcile_outcome_tracker.record_decision(
                        ReconcileDecision {
                            oid,
                            action: action_type,
                            predicted_p_fill: update.p_fill_new,
                            predicted_queue_value_bps: update.value_bps,
                            price_drift_bps,
                            decided_at: std::time::Instant::now(),
                        },
                    );
                }
            }
        }

        // === ADAPTIVE CYCLE TIMING ===
        // Set dynamic fallback interval based on volatility, latch threshold, and headroom.
        {
            use super::event_accumulator::compute_next_cycle_time;
            let next_interval = compute_next_cycle_time(
                sigma,
                dynamic_config.latch_threshold_bps,
                headroom_pct,
            );
            self.event_accumulator.set_dynamic_fallback(next_interval);
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

        // G4: Debug paranoia gate — catch invalid prices before they reach the exchange.
        // Zero-cost in release builds. Catches validation bugs during development.
        #[cfg(debug_assertions)]
        for (price, _size) in &orders {
            debug_assert!(
                self.exchange_rules.is_valid_price(*price),
                "[PlaceBulk] Invalid price {price} on {side_str} side — \
                 round_to={}, rules={:?}",
                self.exchange_rules.round_price(*price),
                self.exchange_rules,
            );
        }

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

        // === POSITION LIMIT / REDUCE-ONLY ENFORCEMENT (defense-in-depth) ===
        // Same checks as place_new_order — prevents position-increasing orders
        // from bypassing reduce-only mode via the bulk placement path.
        {
            let current_pos = self.position.position();
            // When user explicitly specified max_position, enforce as hard ceiling.
            // Otherwise, let dynamic margin-based effective_max_position be the sole limit.
            let max_pos = if self.config.max_position_user_specified {
                self.effective_max_position.min(self.config.max_position)
            } else {
                self.effective_max_position
            };
            let would_increase = (is_buy && current_pos >= 0.0) || (!is_buy && current_pos <= 0.0);

            // Hard limit: reject entire batch if position already at/above max
            if would_increase && current_pos.abs() >= max_pos {
                warn!(
                    side = %side_str,
                    position = %format!("{:.4}", current_pos),
                    max_position = %format!("{:.4}", max_pos),
                    "[PlaceBulk] HARD LIMIT: blocking position-increasing batch"
                );
                return Ok(());
            }

            // Reduce-only: reject if position at/above 95% of effective max
            let reduce_only_threshold = max_pos * 0.95;
            if would_increase && current_pos.abs() >= reduce_only_threshold {
                warn!(
                    side = %side_str,
                    position = %format!("{:.4}", current_pos),
                    threshold = %format!("{:.4}", reduce_only_threshold),
                    "[PlaceBulk] Reduce-only: blocking position-increasing batch"
                );
                return Ok(());
            }
        }

        let mut order_specs: Vec<OrderSpec> = Vec::new();
        let mut cumulative_size = 0.0; // Track cumulative exposure for exchange limits
        let mut orders_blocked_by_capacity = 0usize;
        let mut orders_blocked_by_governor = 0usize;
        let mut orders_zero_size = 0usize;
        let mut orders_post_truncation_zero = 0usize;
        let mut orders_margin_zero = 0usize;
        let mut orders_below_notional = 0usize;
        let mut orders_bbo_crossing = 0usize;

        for (price, size) in orders {
            if size <= 0.0 {
                orders_zero_size += 1;
                continue;
            }

            // === InventoryGovernor: Hard ceiling check (defense-in-depth) ===
            if self.inventory_governor.would_exceed(self.position.position(), size, is_buy)
                && !self.inventory_governor.is_reducing(self.position.position(), is_buy)
            {
                orders_blocked_by_governor += 1;
                if orders_blocked_by_governor == 1 {
                    warn!(
                        side = %side_str,
                        position = %format!("{:.4}", self.position.position()),
                        order_size = %format!("{:.4}", size),
                        max_position = %format!("{:.4}", self.inventory_governor.max_position()),
                        "[PlaceBulk] InventoryGovernor: blocking position-increasing order"
                    );
                }
                continue;
            }

            // Defense-in-depth BBO crossing check per order.
            // The reconcile-level check catches most cases, but this catches
            // any order that slips through (e.g., from modify fallback paths).
            {
                let bbo_valid = self.cached_best_bid > 0.0 && self.cached_best_ask > 0.0;
                if bbo_valid {
                    let actual_tick = 10f64.powi(-(self.config.decimals as i32));
                    let safety_margin = actual_tick * 2.0; // 2 ticks
                    if is_buy && price >= self.cached_best_ask - safety_margin {
                        warn!(
                            bid_price = %format!("{:.6}", price),
                            exchange_ask = %format!("{:.6}", self.cached_best_ask),
                            "Filtering bid order: would cross exchange best ask"
                        );
                        orders_bbo_crossing += 1;
                        continue;
                    }
                    if !is_buy && price <= self.cached_best_bid + safety_margin {
                        warn!(
                            ask_price = %format!("{:.6}", price),
                            exchange_bid = %format!("{:.6}", self.cached_best_bid),
                            "Filtering ask order: would cross exchange best bid"
                        );
                        orders_bbo_crossing += 1;
                        continue;
                    }
                }
            }

            // Apply margin check
            let sizing_result =
                self.infra
                    .margin_sizer
                    .adjust_size(size, price, self.position.position(), is_buy);

            if sizing_result.adjusted_size <= 0.0 {
                orders_margin_zero += 1;
                continue;
            }

            // Smart round-up: if pre-truncation notional exceeds minimum, round UP to avoid
            // losing orders that are just barely above the $10 threshold
            let pre_notional = sizing_result.adjusted_size * price;
            let should_round_up = pre_notional >= MIN_ORDER_NOTIONAL;
            let mut truncated_size =
                truncate_float(sizing_result.adjusted_size, self.config.sz_decimals, should_round_up);
            if truncated_size <= 0.0 {
                orders_post_truncation_zero += 1;
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
                orders_below_notional += 1;
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

        // DIAGNOSTIC: Structured summary log with per-reason filter counts
        info!(
            side = %side_str,
            input_orders = input_count,
            orders_passed_filter = order_specs.len(),
            orders_filtered_out = input_count - order_specs.len(),
            zero_size = orders_zero_size,
            margin_zero = orders_margin_zero,
            post_truncation_zero = orders_post_truncation_zero,
            below_notional = orders_below_notional,
            bbo_crossing = orders_bbo_crossing,
            blocked_by_capacity = orders_blocked_by_capacity,
            blocked_by_governor = orders_blocked_by_governor,
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

        // Pre-register as pending with calibration predictions
        let mut cloids_for_tracker: Vec<String> = Vec::with_capacity(order_specs.len());
        let regime_str = format!("{:?}", self.estimator.volatility_regime());
        let mid_price = self.latest_mid;

        for spec in &order_specs {
            if let Some(ref cloid) = spec.cloid {
                // Calculate depth and make calibration predictions
                let depth_bps = if mid_price > 0.0 {
                    ((spec.price - mid_price).abs() / mid_price) * 10000.0
                } else {
                    0.0
                };
                let (_, fill_pred_id) = self
                    .stochastic
                    .model_calibration
                    .fill_model
                    .predict_with_regime(depth_bps, &regime_str);
                let (_, as_pred_id) = self
                    .stochastic
                    .model_calibration
                    .as_model
                    .predict_with_regime(&regime_str);

                self.orders.add_pending_with_calibration(
                    side,
                    spec.price,
                    spec.size,
                    cloid.clone(),
                    Some(fill_pred_id),
                    Some(as_pred_id),
                    Some(depth_bps),
                    mid_price,
                );
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
            .environment
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
                        TrackedOrder::with_cloid(result.oid, c.clone(), side, spec.price, spec.size, self.latest_mid)
                    } else {
                        TrackedOrder::new(result.oid, side, spec.price, spec.size, self.latest_mid)
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
                        self.latest_mid,
                    )
                } else {
                    TrackedOrder::new(result.oid, side, spec.price, result.resting_size, self.latest_mid)
                };
                self.ws_state.add_order(tracked);

                // Initialize queue tracking with L2-derived depth estimate.
                let depth_ahead = self.tier1.queue_tracker.estimate_depth_at_price(spec.price, is_buy);
                self.tier1.queue_tracker.order_placed(
                    result.oid,
                    spec.price,
                    result.resting_size,
                    depth_ahead,
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
