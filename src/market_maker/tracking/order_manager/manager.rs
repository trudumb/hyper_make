//! Order manager implementation.
//!
//! The `OrderManager` struct tracks all resting orders with proper lifecycle handling.

use std::collections::HashMap;
use std::time::Duration;

use tracing::{debug, warn};

use crate::{bps_diff, EPSILON};

use super::reconcile::reconcile_side;
use super::types::{
    price_to_key, LadderAction, OrderManagerConfig, OrderState, PendingOrder, Side, TrackedOrder,
};
use crate::market_maker::config::Quote;
use crate::market_maker::infra::capacity::ORDER_MANAGER_CAPACITY;
use crate::market_maker::quoting::Ladder;

/// Manages tracked orders for the market maker with proper lifecycle handling.
///
/// Key invariants:
/// 1. Orders are never removed immediately after cancel - they wait for fill window
/// 2. Position updates happen regardless of tracking (position is always correct)
/// 3. Cleanup is centralized through the `cleanup()` method
/// 4. Pending orders bridge the race between placement and fill notification
///
/// # CLOID Tracking (Phase 1 Fix)
///
/// The primary lookup for pending orders is now by CLOID (Client Order ID).
/// CLOIDs are UUIDs generated before order placement, making fill matching deterministic.
/// This eliminates the race condition where fills arrived before REST returned the OID.
///
/// Lookup priority:
/// 1. CLOID (primary) - always available, deterministic
/// 2. Price-based (fallback) - for edge cases where CLOID is missing
#[derive(Debug)]
pub struct OrderManager {
    /// Orders indexed by order ID
    orders: HashMap<u64, TrackedOrder>,
    /// Pending orders awaiting OID assignment, indexed by (side, price_key).
    /// Used to handle immediate fills that arrive before OID is known.
    pending: HashMap<(Side, u64), PendingOrder>,
    /// Pending orders indexed by CLOID (primary lookup).
    /// CLOID is generated before placement and returned in fill notifications.
    pending_by_cloid: HashMap<String, PendingOrder>,
    /// Configuration
    config: OrderManagerConfig,
}

impl Default for OrderManager {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderManager {
    /// Create a new order manager with default configuration.
    ///
    /// Pre-allocates HashMaps to avoid heap reallocations in hot paths.
    pub fn new() -> Self {
        Self {
            orders: HashMap::with_capacity(ORDER_MANAGER_CAPACITY),
            pending: HashMap::with_capacity(ORDER_MANAGER_CAPACITY),
            pending_by_cloid: HashMap::with_capacity(ORDER_MANAGER_CAPACITY),
            config: OrderManagerConfig::default(),
        }
    }

    /// Create a new order manager with custom configuration.
    ///
    /// Pre-allocates HashMaps to avoid heap reallocations in hot paths.
    pub fn with_config(config: OrderManagerConfig) -> Self {
        Self {
            orders: HashMap::with_capacity(ORDER_MANAGER_CAPACITY),
            pending: HashMap::with_capacity(ORDER_MANAGER_CAPACITY),
            pending_by_cloid: HashMap::with_capacity(ORDER_MANAGER_CAPACITY),
            config,
        }
    }

    /// Add an order to track.
    pub fn add_order(&mut self, order: TrackedOrder) {
        self.orders.insert(order.oid, order);
    }

    /// Remove an order by ID (use cleanup() for normal removal).
    pub fn remove_order(&mut self, oid: u64) -> Option<TrackedOrder> {
        self.orders.remove(&oid)
    }

    /// Get an order by ID.
    pub fn get_order(&self, oid: u64) -> Option<&TrackedOrder> {
        self.orders.get(&oid)
    }

    /// Get a mutable order by ID.
    pub fn get_order_mut(&mut self, oid: u64) -> Option<&mut TrackedOrder> {
        self.orders.get_mut(&oid)
    }

    /// Update an order after successful modify.
    ///
    /// Preserves tracking data (placed_at, fill history, state) while updating price/size.
    /// This is critical for queue position preservation - the order maintains its place
    /// in the book rather than being re-queued at the back.
    ///
    /// # Returns
    /// `true` if the order was found and updated, `false` otherwise.
    pub fn on_modify_success(&mut self, oid: u64, new_price: f64, new_size: f64) -> bool {
        if let Some(order) = self.orders.get_mut(&oid) {
            debug!(
                oid = oid,
                old_price = order.price,
                new_price = new_price,
                old_size = order.size,
                new_size = new_size,
                "Order modified, tracking updated (queue position preserved)"
            );
            order.price = new_price;
            order.size = new_size;
            true
        } else {
            false
        }
    }

    /// Replace an order's OID after modify (exchange may assign new OID).
    ///
    /// Hyperliquid's modify API can return a NEW order ID. This method re-keys
    /// the order in our tracking HashMap while preserving all tracking data.
    ///
    /// # Arguments
    /// * `old_oid` - The original order ID we tracked
    /// * `new_oid` - The new order ID from exchange modify response
    ///
    /// # Returns
    /// `true` if the order was found and re-keyed, `false` otherwise.
    pub fn replace_oid(&mut self, old_oid: u64, new_oid: u64) -> bool {
        if old_oid == new_oid {
            return true; // No change needed
        }

        if let Some(mut order) = self.orders.remove(&old_oid) {
            debug!(
                old_oid = old_oid,
                new_oid = new_oid,
                side = ?order.side,
                price = order.price,
                "Replacing order OID after modify (exchange assigned new OID)"
            );
            order.oid = new_oid;
            self.orders.insert(new_oid, order);
            true
        } else {
            warn!(
                old_oid = old_oid,
                new_oid = new_oid,
                "Cannot replace OID - old order not found in tracking"
            );
            false
        }
    }

    /// Get the first active order on a given side (for single-order-per-side strategies).
    /// Only returns orders that are actively quoting (Resting or PartialFilled).
    pub fn get_by_side(&self, side: Side) -> Option<&TrackedOrder> {
        self.orders
            .values()
            .find(|o| o.side == side && !o.is_filled() && o.is_active())
    }

    /// Get all active orders on a given side.
    /// Only returns orders that are actively quoting (Resting or PartialFilled).
    pub fn get_all_by_side(&self, side: Side) -> Vec<&TrackedOrder> {
        self.orders
            .values()
            .filter(|o| o.side == side && !o.is_filled() && o.is_active())
            .collect()
    }

    /// Get all non-terminal orders on a side, including those being cancelled.
    /// Use this when you need to know about pending cancels.
    pub fn get_all_by_side_including_pending(&self, side: Side) -> Vec<&TrackedOrder> {
        self.orders
            .values()
            .filter(|o| o.side == side && !o.is_terminal())
            .collect()
    }

    // === Pending Order Management ===
    // These methods handle the race condition between order placement and fill notification.
    // When a bulk order is placed, we register it as "pending" by CLOID (primary) and
    // (side, price) (fallback) before the API call returns.
    //
    // CLOID-FIRST LOOKUP (Phase 1 Fix):
    // - CLOID is a UUID generated BEFORE placement
    // - Fill notifications include the CLOID
    // - Lookup is deterministic - no timing race condition
    // - Price-based lookup is kept as fallback for edge cases

    /// Register a pending order before placing it on the exchange (legacy, without CLOID).
    ///
    /// Call this BEFORE the bulk order API call. When the API returns with OIDs,
    /// call `finalize_pending()` to convert to a tracked order.
    ///
    /// NOTE: Prefer `add_pending_with_cloid()` for deterministic fill tracking.
    pub fn add_pending(&mut self, side: Side, price: f64, size: f64, mid: f64) {
        let key = (side, price_to_key(price));
        self.pending
            .insert(key, PendingOrder::new(side, price, size, mid));
    }

    /// Register a pending order with CLOID before placing it on the exchange.
    ///
    /// This is the preferred method - CLOID provides deterministic fill matching.
    /// Call this BEFORE the bulk order API call. When the API returns with OIDs,
    /// call `finalize_pending_by_cloid()` to convert to a tracked order.
    ///
    /// # Arguments
    /// - `side`: Buy or Sell
    /// - `price`: Limit price
    /// - `size`: Order size
    /// - `cloid`: Client Order ID (UUID string, generated before placement)
    pub fn add_pending_with_cloid(
        &mut self,
        side: Side,
        price: f64,
        size: f64,
        cloid: String,
        mid: f64,
    ) {
        // Store in CLOID map (primary lookup)
        let pending = PendingOrder::with_cloid(side, price, size, cloid.clone(), mid);
        self.pending_by_cloid.insert(cloid, pending.clone());

        // Also store by price (fallback lookup)
        let key = (side, price_to_key(price));
        self.pending.insert(key, pending);
    }

    /// Register a pending order with full calibration tracking.
    ///
    /// Use this method when making predictions at quote time. The prediction IDs
    /// will be preserved through to the TrackedOrder and used when recording
    /// calibration outcomes on fill.
    ///
    /// # Arguments
    /// - `side`: Buy or Sell
    /// - `price`: Limit price
    /// - `size`: Order size
    /// - `cloid`: Client Order ID (UUID string, generated before placement)
    /// - `fill_prediction_id`: Prediction ID from CalibratedFillModel::predict()
    /// - `as_prediction_id`: Prediction ID from CalibratedASModel::predict()
    /// - `depth_bps`: Depth from mid in basis points (for calibration context)
    /// - `mid`: Mid price at placement time (for edge measurement)
    #[allow(clippy::too_many_arguments)]
    pub fn add_pending_with_calibration(
        &mut self,
        side: Side,
        price: f64,
        size: f64,
        cloid: String,
        fill_prediction_id: Option<u64>,
        as_prediction_id: Option<u64>,
        depth_bps: Option<f64>,
        mid: f64,
    ) {
        // Use the with_calibration constructor to include all tracking data
        let pending = PendingOrder::with_calibration(
            side,
            price,
            size,
            cloid.clone(),
            fill_prediction_id,
            as_prediction_id,
            depth_bps,
            mid,
        );
        self.pending_by_cloid.insert(cloid, pending.clone());

        // Also store by price (fallback lookup)
        let key = (side, price_to_key(price));
        self.pending.insert(key, pending);
    }

    /// Finalize a pending order by assigning it a real OID (legacy, price-based).
    ///
    /// Call this when the bulk order API returns with the assigned OID.
    /// Moves the order from pending to tracked.
    /// Returns the pending order if found, None if not found (shouldn't happen).
    pub fn finalize_pending(&mut self, side: Side, price: f64, oid: u64, resting_size: f64) {
        let key = (side, price_to_key(price));
        if let Some(pending) = self.pending.remove(&key) {
            // Also remove from CLOID map if it was stored there
            if let Some(ref cloid) = pending.cloid {
                self.pending_by_cloid.remove(cloid);
            }
            // Create tracked order from pending - preserves prediction IDs for calibration
            let mut order = TrackedOrder::from_pending(oid, &pending);
            order.size = resting_size; // Use resting size from exchange
            self.add_order(order);
        }
        // Note: If pending not found, order may have been cleaned up or never registered.
        // This is fine - the order will be tracked when we see it in responses.
    }

    /// Finalize a pending order by CLOID (preferred method).
    ///
    /// Call this when the bulk order API returns with the assigned OID.
    /// Moves the order from pending to tracked.
    ///
    /// # Returns
    /// `true` if the pending order was found and finalized, `false` otherwise.
    pub fn finalize_pending_by_cloid(&mut self, cloid: &str, oid: u64, resting_size: f64) -> bool {
        if let Some(pending) = self.pending_by_cloid.remove(cloid) {
            // Also remove from price-based map
            let key = (pending.side, price_to_key(pending.price));
            self.pending.remove(&key);

            // Create tracked order from pending - preserves prediction IDs for calibration
            let mut order = TrackedOrder::from_pending(oid, &pending);
            order.size = resting_size; // Use resting size from exchange
            self.add_order(order);
            true
        } else {
            false
        }
    }

    /// Get a pending order by CLOID (primary lookup).
    ///
    /// This is the preferred lookup method - CLOID is deterministic.
    pub fn get_pending_by_cloid(&self, cloid: &str) -> Option<&PendingOrder> {
        self.pending_by_cloid.get(cloid)
    }

    /// Get a pending order by side and price (fallback lookup).
    ///
    /// Used when a fill arrives but the order isn't tracked by OID yet.
    /// Returns the placement price so we can feed the kappa estimator.
    ///
    /// NOTE: Prefer `get_pending_by_cloid()` when CLOID is available in the fill.
    pub fn get_pending(&self, side: Side, price: f64) -> Option<&PendingOrder> {
        let key = (side, price_to_key(price));
        self.pending.get(&key)
    }

    /// Remove a pending order by CLOID.
    ///
    /// Used when an order fills immediately and we don't want to track it.
    pub fn remove_pending_by_cloid(&mut self, cloid: &str) -> Option<PendingOrder> {
        if let Some(pending) = self.pending_by_cloid.remove(cloid) {
            // Also remove from price-based map
            let key = (pending.side, price_to_key(pending.price));
            self.pending.remove(&key);
            Some(pending)
        } else {
            None
        }
    }

    /// Remove a pending order by side and price (legacy).
    ///
    /// Used when an order fills immediately and we don't want to track it.
    pub fn remove_pending(&mut self, side: Side, price: f64) -> Option<PendingOrder> {
        let key = (side, price_to_key(price));
        if let Some(pending) = self.pending.remove(&key) {
            // Also remove from CLOID map if present
            if let Some(ref cloid) = pending.cloid {
                self.pending_by_cloid.remove(cloid);
            }
            Some(pending)
        } else {
            None
        }
    }

    /// Remove stale pending orders that have been waiting too long.
    ///
    /// Pending orders should be finalized within milliseconds. If they're still
    /// pending after max_age, something went wrong (e.g., API error, disconnection).
    /// Returns the number of stale orders removed.
    pub fn cleanup_stale_pending(&mut self, max_age: Duration) -> usize {
        // Clean both maps
        let before_price = self.pending.len();
        self.pending.retain(|_, p| p.placed_at.elapsed() < max_age);

        let before_cloid = self.pending_by_cloid.len();
        self.pending_by_cloid
            .retain(|_, p| p.placed_at.elapsed() < max_age);

        // Return max removed (they should be in sync, but just in case)
        (before_price - self.pending.len()).max(before_cloid - self.pending_by_cloid.len())
    }

    /// Get the number of pending orders (for debugging/metrics).
    pub fn pending_count(&self) -> usize {
        // Return CLOID count as it's the primary tracking
        self.pending_by_cloid.len().max(self.pending.len())
    }

    /// Get the number of pending orders by CLOID (for debugging/metrics).
    pub fn pending_by_cloid_count(&self) -> usize {
        self.pending_by_cloid.len()
    }

    /// Set the state of an order (legacy method - prefer transition methods).
    /// Returns true if the order was found and updated.
    #[deprecated(note = "Use initiate_cancel, on_cancel_confirmed, etc. instead")]
    pub fn set_state(&mut self, oid: u64, state: OrderState) -> bool {
        if let Some(order) = self.orders.get_mut(&oid) {
            order.transition_to(state);
            true
        } else {
            false
        }
    }

    /// Initiate a cancel on an order. Marks it as CancelPending.
    /// Returns true if the order was found and transitioned.
    pub fn initiate_cancel(&mut self, oid: u64) -> bool {
        if let Some(order) = self.orders.get_mut(&oid) {
            if order.is_active() {
                order.transition_to(OrderState::CancelPending);
                debug!(oid = oid, "Order transitioned to CancelPending");
                return true;
            }
        }
        false
    }

    /// Handle cancel confirmation from exchange.
    /// Transitions to CancelConfirmed to start fill window.
    pub fn on_cancel_confirmed(&mut self, oid: u64) {
        if let Some(order) = self.orders.get_mut(&oid) {
            if order.state == OrderState::CancelPending {
                order.transition_to(OrderState::CancelConfirmed);
                debug!(
                    oid = oid,
                    "Order transitioned to CancelConfirmed, starting fill window"
                );
            }
        }
    }

    /// Handle "already filled" response from cancel attempt.
    /// Transitions to FilledDuringCancel.
    pub fn on_cancel_already_filled(&mut self, oid: u64) {
        if let Some(order) = self.orders.get_mut(&oid) {
            order.transition_to(OrderState::FilledDuringCancel);
            debug!(oid = oid, "Order marked as FilledDuringCancel");
        }
    }

    /// Handle cancel failure - revert to previous active state.
    pub fn on_cancel_failed(&mut self, oid: u64) {
        if let Some(order) = self.orders.get_mut(&oid) {
            if order.state == OrderState::CancelPending {
                // Revert based on fill status
                if order.filled > EPSILON {
                    order.transition_to(OrderState::PartialFilled);
                    debug!(oid = oid, "Cancel failed, reverted to PartialFilled");
                } else {
                    order.transition_to(OrderState::Resting);
                    debug!(oid = oid, "Cancel failed, reverted to Resting");
                }
            }
        }
    }

    /// Process a fill for an order.
    /// Returns (order_found, is_new_fill, is_order_complete).
    pub fn process_fill(&mut self, oid: u64, tid: u64, amount: f64) -> (bool, bool, bool) {
        if let Some(order) = self.orders.get_mut(&oid) {
            // Check if we already processed this tid for this order
            if !order.record_fill(tid, amount) {
                return (true, false, false); // Duplicate fill
            }

            let is_complete = order.is_filled();

            // Update state based on current state
            match order.state {
                OrderState::Resting => {
                    if is_complete {
                        order.transition_to(OrderState::Filled);
                    } else {
                        order.transition_to(OrderState::PartialFilled);
                    }
                }
                OrderState::PartialFilled => {
                    if is_complete {
                        order.transition_to(OrderState::Filled);
                    }
                    // else stay in PartialFilled
                }
                OrderState::CancelPending | OrderState::CancelConfirmed => {
                    // Fill arrived during cancel window - this is expected!
                    order.transition_to(OrderState::FilledDuringCancel);
                    debug!(
                        oid = oid,
                        tid = tid,
                        amount = amount,
                        "Fill arrived during cancel window"
                    );
                }
                _ => {
                    // Unexpected state - still process
                    debug!(
                        oid = oid,
                        state = ?order.state,
                        "Fill arrived in unexpected state"
                    );
                }
            }

            (true, true, is_complete)
        } else {
            (false, false, false) // Order not found
        }
    }

    /// Update an order with a fill amount (legacy method).
    /// Returns `true` if the order was found and updated.
    #[deprecated(note = "Use process_fill() instead for proper state management")]
    pub fn update_fill(&mut self, oid: u64, filled_amount: f64) -> bool {
        if let Some(order) = self.orders.get_mut(&oid) {
            order.filled += filled_amount;
            true
        } else {
            false
        }
    }

    /// Check if an order on the given side needs to be updated based on a new quote.
    pub fn needs_update(&self, side: Side, new_quote: &Quote, max_bps_diff: u16) -> bool {
        match self.get_by_side(side) {
            Some(order) => {
                // Check if size changed significantly
                let size_changed = (new_quote.size - order.remaining()).abs() > EPSILON;
                // Check if price deviated too much
                let price_deviated = bps_diff(new_quote.price, order.price) > max_bps_diff;
                size_changed || price_deviated
            }
            None => {
                // No order on this side, need to place one if quote has size
                new_quote.size > EPSILON
            }
        }
    }

    /// Run cleanup cycle - remove orders that are safe to remove.
    /// Returns list of removed order IDs for external cleanup (QueueTracker, etc.)
    pub fn cleanup(&mut self) -> Vec<u64> {
        let fill_window = self.config.fill_window_duration;
        let mut to_remove = Vec::new();

        for (&oid, order) in &mut self.orders {
            let should_remove = match order.state {
                // Terminal states - always safe to remove
                OrderState::Filled | OrderState::FilledDuringCancel => {
                    debug!(
                        oid = oid,
                        state = ?order.state,
                        filled = order.filled,
                        "Cleanup: removing filled order"
                    );
                    true
                }
                // Cancelled is terminal
                OrderState::Cancelled => {
                    debug!(oid = oid, "Cleanup: removing cancelled order");
                    true
                }
                // CancelConfirmed - check if fill window expired
                OrderState::CancelConfirmed => {
                    if order.fill_window_expired(fill_window) {
                        // Transition to Cancelled before removal
                        order.transition_to(OrderState::Cancelled);
                        debug!(
                            oid = oid,
                            window_ms = fill_window.as_millis(),
                            "Cleanup: fill window expired, removing cancelled order"
                        );
                        true
                    } else {
                        let elapsed = order.state_changed_at.elapsed();
                        debug!(
                            oid = oid,
                            elapsed_ms = elapsed.as_millis(),
                            window_ms = fill_window.as_millis(),
                            "Cleanup: order in fill window, not removing yet"
                        );
                        false
                    }
                }
                // Active states - don't remove
                _ => false,
            };

            if should_remove {
                to_remove.push(oid);
            }
        }

        for oid in &to_remove {
            self.orders.remove(oid);
        }

        to_remove
    }

    /// Remove fully filled orders (legacy method - prefer cleanup()).
    #[deprecated(note = "Use cleanup() for proper lifecycle management")]
    pub fn cleanup_filled(&mut self) {
        self.orders.retain(|_, order| !order.is_filled());
    }

    /// Get all order IDs.
    pub fn order_ids(&self) -> Vec<u64> {
        self.orders.keys().copied().collect()
    }

    /// Get all ACTIVE order IDs (Resting or PartialFilled only).
    ///
    /// This matches the semantics of `WsOrderStateManager::open_order_ids()` for proper
    /// comparison in SafetySync. Orders in cancel/filled states are excluded.
    pub fn active_order_ids(&self) -> std::collections::HashSet<u64> {
        self.orders
            .values()
            .filter(|o| o.is_active())
            .map(|o| o.oid)
            .collect()
    }

    /// Check if there are any orders.
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Get the number of orders.
    pub fn len(&self) -> usize {
        self.orders.len()
    }

    /// Check for stuck orders (cancel timeout exceeded).
    pub fn check_stuck_cancels(&self) -> Vec<u64> {
        let timeout = self.config.cancel_timeout;
        self.orders
            .values()
            .filter_map(|o| {
                if o.state == OrderState::CancelPending && o.state_changed_at.elapsed() > timeout {
                    Some(o.oid)
                } else {
                    None
                }
            })
            .collect()
    }

    // === Pending Exposure Calculation ===
    // These methods calculate the potential position change if all resting orders execute.
    // Critical for risk management - knowing worst-case position exposure.

    /// Calculate pending exposure for each side.
    ///
    /// Returns (bid_exposure, ask_exposure) where:
    /// - bid_exposure: Total remaining size on buy orders (would increase position if filled)
    /// - ask_exposure: Total remaining size on sell orders (would decrease position if filled)
    pub fn pending_exposure(&self) -> (f64, f64) {
        let bid_exposure: f64 = self
            .get_all_by_side(Side::Buy)
            .iter()
            .map(|o| o.remaining())
            .sum();
        let ask_exposure: f64 = self
            .get_all_by_side(Side::Sell)
            .iter()
            .map(|o| o.remaining())
            .sum();
        (bid_exposure, ask_exposure)
    }

    /// Get the count of active orders on each side.
    ///
    /// Returns (bid_count, ask_count) where each is the number of active resting orders.
    pub fn order_counts(&self) -> (usize, usize) {
        let bid_count = self.get_all_by_side(Side::Buy).len();
        let ask_count = self.get_all_by_side(Side::Sell).len();
        (bid_count, ask_count)
    }

    /// Calculate net pending inventory change.
    ///
    /// Positive = net long exposure if all orders fill
    /// Negative = net short exposure if all orders fill
    ///
    /// Formula: bid_exposure - ask_exposure
    pub fn net_pending_change(&self) -> f64 {
        let (bids, asks) = self.pending_exposure();
        bids - asks
    }

    /// Calculate worst-case position if all orders on one side fill.
    ///
    /// Given current position, returns (min_position, max_position):
    /// - max_position: current + all bid exposure (all buys fill, no sells)
    /// - min_position: current - all ask exposure (all sells fill, no buys)
    pub fn worst_case_positions(&self, current_position: f64) -> (f64, f64) {
        let (bid_exposure, ask_exposure) = self.pending_exposure();
        let max_position = current_position + bid_exposure;
        let min_position = current_position - ask_exposure;
        (min_position, max_position)
    }

    /// Get a summary of current quotes for logging.
    ///
    /// Returns (best_bid, best_ask, bid_levels, ask_levels).
    /// Prices are 0.0 if no active orders exist on that side.
    pub fn get_quote_summary(&self) -> (f64, f64, usize, usize) {
        let bids = self.get_all_by_side(Side::Buy);
        let asks = self.get_all_by_side(Side::Sell);

        let best_bid = bids.iter().map(|o| o.price).fold(0.0_f64, f64::max);
        let best_ask = asks.iter().map(|o| o.price).fold(f64::MAX, f64::min);
        let best_ask = if best_ask == f64::MAX { 0.0 } else { best_ask };

        (best_bid, best_ask, bids.len(), asks.len())
    }

    /// Get total resting notional value (for risk monitoring).
    ///
    /// Returns (bid_notional, ask_notional) in USD.
    pub fn resting_notional(&self) -> (f64, f64) {
        let bid_notional: f64 = self
            .get_all_by_side(Side::Buy)
            .iter()
            .map(|o| o.remaining() * o.price)
            .sum();
        let ask_notional: f64 = self
            .get_all_by_side(Side::Sell)
            .iter()
            .map(|o| o.remaining() * o.price)
            .sum();
        (bid_notional, ask_notional)
    }

    /// Compute actions to reconcile current orders with target ladder.
    ///
    /// Returns a list of `LadderAction`s to execute:
    /// - `Cancel` for orders that don't match any target level
    /// - `Place` for target levels that don't have matching orders
    pub fn reconcile_ladder(&self, ladder: &Ladder, max_bps_diff: u16) -> Vec<LadderAction> {
        let mut actions = Vec::new();

        // Get current orders by side
        let current_bids = self.get_all_by_side(Side::Buy);
        let current_asks = self.get_all_by_side(Side::Sell);

        // Reconcile bids
        actions.extend(reconcile_side(
            &current_bids,
            &ladder.bids,
            Side::Buy,
            max_bps_diff,
        ));

        // Reconcile asks
        actions.extend(reconcile_side(
            &current_asks,
            &ladder.asks,
            Side::Sell,
            max_bps_diff,
        ));

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_oid() {
        let mut manager = OrderManager::new();

        // Add an order with old OID
        let order = TrackedOrder::new(100, Side::Buy, 50000.0, 0.01, 0.0);
        manager.add_order(order);

        assert!(manager.get_order(100).is_some());
        assert!(manager.get_order(200).is_none());

        // Replace OID 100 -> 200
        assert!(manager.replace_oid(100, 200));

        // Old OID should be gone, new OID should exist
        assert!(manager.get_order(100).is_none());
        assert!(manager.get_order(200).is_some());

        // Check order data preserved
        let order = manager.get_order(200).unwrap();
        assert_eq!(order.oid, 200);
        assert_eq!(order.side, Side::Buy);
        assert!((order.price - 50000.0).abs() < f64::EPSILON);
        assert!((order.size - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_replace_oid_same() {
        let mut manager = OrderManager::new();
        let order = TrackedOrder::new(100, Side::Buy, 50000.0, 0.01, 0.0);
        manager.add_order(order);

        // Replacing with same OID should succeed (no-op)
        assert!(manager.replace_oid(100, 100));
        assert!(manager.get_order(100).is_some());
    }

    #[test]
    fn test_replace_oid_not_found() {
        let mut manager = OrderManager::new();

        // Replacing non-existent OID should fail
        assert!(!manager.replace_oid(999, 1000));
    }
}
