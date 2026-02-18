//! Order operations - placement, cancellation, and tracking.

use std::time::Duration;

use tracing::{debug, info, warn};

use crate::helpers::truncate_float;
use crate::prelude::Result;

use super::super::{
    infra, CancelResult, MarketMaker, TradingEnvironment, Quote, QuotingStrategy, Side, TrackedOrder,
};
use super::side_str;
use super::RecoveryAction;

/// Minimum order notional value in USD (Hyperliquid requirement)
const MIN_ORDER_NOTIONAL: f64 = 10.0;

impl<S: QuotingStrategy, Env: TradingEnvironment> MarketMaker<S, Env> {
    /// Initiate bulk cancel for multiple orders and update order states appropriately.
    ///
    /// RATE LIMIT OPTIMIZATION: Uses single bulk cancel API call instead of individual cancels.
    /// Per Hyperliquid docs: batched requests are 1 weight for IP limits (vs n for individual).
    pub(crate) async fn initiate_bulk_cancel(&mut self, oids: Vec<u64>) {
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

        // Record cancel requests for cancel-race AS tracking (Sprint 6.3)
        let cancel_ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        for &oid in &cancellable_oids {
            self.cancel_race_tracker.record_cancel_request(oid, cancel_ts_ms);
        }

        // Debug level here since executor.rs logs at INFO level
        debug!("Bulk cancelling {} orders", cancellable_oids.len());

        let num_cancels = cancellable_oids.len() as u32;
        let results = self
            .environment
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
    pub(crate) async fn initiate_and_track_cancel(&mut self, oid: u64) {
        // Get order price before cancel for Bayesian learning
        let order_price = self.orders.get_order(oid).map(|o| o.price);

        // Record cancel request for cancel-race AS tracking (Sprint 6.3 fix F2)
        // Must happen before cancel is sent so we can detect fills-after-cancel
        let cancel_ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.cancel_race_tracker.record_cancel_request(oid, cancel_ts_ms);

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

    /// Place a new order.
    pub(crate) async fn place_new_order(&mut self, side: Side, quote: &Quote) -> Result<()> {
        let is_buy = side == Side::Buy;

        // === BBO Crossing Guard (defense-in-depth) ===
        // Prevent post-only rejections that cause one-sided exposure
        {
            let bbo_valid = self.cached_best_bid > 0.0 && self.cached_best_ask > 0.0;
            if bbo_valid && self.latest_mid > 0.0 {
                let tick_proxy = self.latest_mid * 0.0001; // 1 bps
                if is_buy && quote.price >= self.cached_best_ask - tick_proxy {
                    warn!(
                        side = %side_str(side),
                        bid_price = %format!("{:.6}", quote.price),
                        exchange_ask = %format!("{:.6}", self.cached_best_ask),
                        "Blocking order: bid would cross exchange best ask"
                    );
                    return Ok(());
                }
                if !is_buy && quote.price <= self.cached_best_bid + tick_proxy {
                    warn!(
                        side = %side_str(side),
                        ask_price = %format!("{:.6}", quote.price),
                        exchange_bid = %format!("{:.6}", self.cached_best_bid),
                        "Blocking order: ask would cross exchange best bid"
                    );
                    return Ok(());
                }
            }
        }

        // === InventoryGovernor: Hard ceiling check (defense-in-depth) ===
        if self.inventory_governor.would_exceed(self.position.position(), quote.size, is_buy)
            && !self.inventory_governor.is_reducing(self.position.position(), is_buy)
        {
            warn!(
                side = %side_str(side),
                position = %format!("{:.4}", self.position.position()),
                order_size = %format!("{:.4}", quote.size),
                max_position = %format!("{:.4}", self.inventory_governor.max_position()),
                "InventoryGovernor: blocking order that would exceed config.max_position"
            );
            return Ok(());
        }

        // === P1: HARD Position Limit Enforcement ===
        // Absolute hard limit — reject any order that would increase position beyond max.
        // When user explicitly specified max_position, enforce as hard ceiling.
        // Otherwise, let dynamic margin-based effective_max_position be the sole limit.
        {
            let current_pos = self.position.position();
            let max_pos = if self.config.max_position_user_specified {
                self.effective_max_position.min(self.config.max_position)
            } else {
                self.effective_max_position
            };
            let would_increase = (is_buy && current_pos >= 0.0) || (!is_buy && current_pos <= 0.0);
            if would_increase && current_pos.abs() >= max_pos {
                warn!(
                    side = %side_str(side),
                    position = %format!("{:.4}", current_pos),
                    max_position = %format!("{:.4}", max_pos),
                    effective_max = %format!("{:.4}", self.effective_max_position),
                    config_max = %format!("{:.4}", self.config.max_position),
                    "HARD LIMIT: rejecting order that would increase position beyond max"
                );
                return Ok(());
            }
        }

        // === P1: Reduce-Only Enforcement ===
        // When position exceeds a fraction of max, reject any position-increasing orders.
        // Uses 95% of the effective limit as soft limit.
        {
            let current_pos = self.position.position();
            let hard_max = if self.config.max_position_user_specified {
                self.effective_max_position.min(self.config.max_position)
            } else {
                self.effective_max_position
            };
            let reduce_only_threshold = hard_max * 0.95;
            if current_pos.abs() >= reduce_only_threshold {
                let would_increase = (is_buy && current_pos >= 0.0) || (!is_buy && current_pos <= 0.0);
                if would_increase {
                    debug!(
                        side = %side_str(side),
                        position = %format!("{:.4}", current_pos),
                        threshold = %format!("{:.4}", reduce_only_threshold),
                        "Reduce-only: rejecting position-increasing order (position at 95%+ of limit)"
                    );
                    return Ok(());
                }
            }
        }

        // === Position Guard Hard Entry Gate ===
        // Reject orders that would push worst-case position beyond hard limit
        let entry_check = self.safety.position_guard.check_order_entry(
            self.position.position(),
            quote.size,
            side,
        );
        if !entry_check.is_allowed() {
            if let super::super::risk::OrderEntryCheck::Rejected { reason, .. } = &entry_check {
                warn!(
                    side = %side_str(side),
                    size = quote.size,
                    position = self.position.position(),
                    "{reason}"
                );
            }
            return Ok(());
        }

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
            .environment
            .place_order(
                &self.config.asset,
                quote.price,
                adjusted_size,
                is_buy,
                None,
                true, // post_only
            )
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
                self.latest_mid,
            ));

            // Also add to WsOrderStateManager for improved state tracking
            self.ws_state.add_order(TrackedOrder::new(
                result.oid,
                side,
                quote.price,
                result.resting_size,
                self.latest_mid,
            ));

            // === Tier 1: Register with queue tracker ===
            // Use L2-derived depth estimate when available, fall back to conservative heuristic.
            let depth_ahead = self.tier1.queue_tracker.estimate_depth_at_price(quote.price, is_buy);
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
    pub(crate) async fn cancel_with_retry(
        &self,
        asset: &str,
        oid: u64,
        max_attempts: u32,
    ) -> CancelResult {
        for attempt in 0..max_attempts {
            let result = self.environment.cancel_order(asset, oid).await;

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

    /// Check recovery state and handle IOC recovery if needed.
    ///
    /// Recovery states:
    /// - Normal: No recovery needed
    /// - ReduceOnlyStuck: Detected via consecutive rejections, handled by record_rejection()
    /// - IocRecovery: Attempts aggressive IOC orders to reduce position
    /// - Cooldown: Waiting before resuming normal quoting
    pub(crate) async fn check_and_handle_recovery(&mut self) -> Result<Option<RecoveryAction>> {
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
                        .environment
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
}
