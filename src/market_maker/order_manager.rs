//! Order state management for tracking resting orders.
//!
//! This module implements a robust order lifecycle state machine that handles
//! the race condition between cancel requests and fill notifications. The key
//! insight is that fills can arrive via WebSocket *after* a cancel has been
//! confirmed by the exchange, so we must maintain a "fill window" after cancel
//! confirmation before removing orders from tracking.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use tracing::debug;

use crate::{bps_diff, EPSILON};

use super::config::Quote;
use super::ladder::{Ladder, LadderLevel};

/// Side of an order (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// Convert from Hyperliquid side string ("B" or "A"/"S").
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "B" => Some(Side::Buy),
            "A" | "S" => Some(Side::Sell),
            _ => None,
        }
    }
}

/// Order lifecycle state.
///
/// State transitions:
/// ```text
/// Resting ─────────────────────────────────────────► Filled
///    │                                                  ▲
///    │ (partial fill)                                   │
///    ▼                                                  │
/// PartialFilled ───────────────────────────────────────┘
///    │                                                  ▲
///    │ (cancel requested)                               │
///    ▼                                                  │
/// CancelPending ──────────────────────────────────────► FilledDuringCancel
///    │                                                  ▲
///    │ (cancel confirmed)                               │
///    ▼                                                  │
/// CancelConfirmed ─────────────────────────────────────┘
///    │
///    │ (fill window expired, no late fills)
///    ▼
/// Cancelled
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    /// Order is resting on the exchange, no fills yet
    Resting,
    /// Order has been partially filled but still has remaining size
    PartialFilled,
    /// Cancel request has been sent, awaiting confirmation.
    /// Fills can still arrive during this window!
    CancelPending,
    /// Cancel has been confirmed by exchange, waiting for fill window to expire.
    /// This is the critical state that prevents premature removal - late fills
    /// from WebSocket can still arrive during this window.
    CancelConfirmed,
    /// Fill arrived during cancel process (CancelPending or CancelConfirmed).
    /// Terminal state - safe to remove immediately.
    FilledDuringCancel,
    /// Fully filled. Terminal state - safe to remove immediately.
    Filled,
    /// Cancel confirmed and fill window has expired with no late fills.
    /// Terminal state - safe to remove immediately.
    Cancelled,
}

/// Action to take for ladder reconciliation.
#[derive(Debug, Clone)]
pub enum LadderAction {
    /// Place a new order at the specified price and size
    Place { side: Side, price: f64, size: f64 },
    /// Cancel an existing order by ID
    Cancel { oid: u64 },
}

/// A tracked order with its current state and lifecycle metadata.
#[derive(Debug, Clone)]
pub struct TrackedOrder {
    /// Order ID from the exchange
    pub oid: u64,
    /// Side of the order
    pub side: Side,
    /// Limit price
    pub price: f64,
    /// Original size
    pub size: f64,
    /// Amount filled so far
    pub filled: f64,
    /// Order lifecycle state
    pub state: OrderState,
    /// When the order was placed (for age tracking)
    pub placed_at: Instant,
    /// When the order entered its current state (for fill window timing)
    pub state_changed_at: Instant,
    /// Trade IDs of fills processed for this order (for dedup at order level)
    pub fill_tids: Vec<u64>,
}

impl TrackedOrder {
    /// Create a new tracked order in Resting state.
    pub fn new(oid: u64, side: Side, price: f64, size: f64) -> Self {
        let now = Instant::now();
        Self {
            oid,
            side,
            price,
            size,
            filled: 0.0,
            state: OrderState::Resting,
            placed_at: now,
            state_changed_at: now,
            fill_tids: Vec::new(),
        }
    }

    /// Get remaining size (unfilled).
    pub fn remaining(&self) -> f64 {
        (self.size - self.filled).max(0.0)
    }

    /// Check if the order is fully filled.
    pub fn is_filled(&self) -> bool {
        self.remaining() <= EPSILON
    }

    /// Transition to a new state, recording the timestamp.
    pub fn transition_to(&mut self, new_state: OrderState) {
        self.state = new_state;
        self.state_changed_at = Instant::now();
    }

    /// Record a fill on this order.
    /// Returns true if this is a new fill, false if duplicate (already processed).
    pub fn record_fill(&mut self, tid: u64, amount: f64) -> bool {
        if self.fill_tids.contains(&tid) {
            return false; // Duplicate fill
        }
        self.fill_tids.push(tid);
        self.filled += amount;
        true
    }

    /// Check if the fill window has expired (for CancelConfirmed state).
    pub fn fill_window_expired(&self, window: Duration) -> bool {
        if self.state == OrderState::CancelConfirmed {
            self.state_changed_at.elapsed() >= window
        } else {
            false
        }
    }

    /// Check if order is in a terminal state (ready for cleanup).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            OrderState::Filled | OrderState::Cancelled | OrderState::FilledDuringCancel
        )
    }

    /// Check if order can receive fills (not yet fully cancelled).
    pub fn can_receive_fills(&self) -> bool {
        matches!(
            self.state,
            OrderState::Resting
                | OrderState::PartialFilled
                | OrderState::CancelPending
                | OrderState::CancelConfirmed
        )
    }

    /// Check if order is actively quoting (not in cancel process).
    pub fn is_active(&self) -> bool {
        matches!(self.state, OrderState::Resting | OrderState::PartialFilled)
    }
}

/// Configuration for order state management.
#[derive(Debug, Clone)]
pub struct OrderManagerConfig {
    /// Time to wait after cancel confirmation for potential late fills.
    /// Hyperliquid typically processes within 2 seconds, so 5s is safe.
    pub fill_window_duration: Duration,
    /// Maximum time an order can be in CancelPending before considered stuck.
    pub cancel_timeout: Duration,
}

impl Default for OrderManagerConfig {
    fn default() -> Self {
        Self {
            fill_window_duration: Duration::from_secs(5),
            cancel_timeout: Duration::from_secs(30),
        }
    }
}

/// Manages tracked orders for the market maker with proper lifecycle handling.
///
/// Key invariants:
/// 1. Orders are never removed immediately after cancel - they wait for fill window
/// 2. Position updates happen regardless of tracking (position is always correct)
/// 3. Cleanup is centralized through the `cleanup()` method
#[derive(Debug)]
pub struct OrderManager {
    /// Orders indexed by order ID
    orders: HashMap<u64, TrackedOrder>,
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
    pub fn new() -> Self {
        Self {
            orders: HashMap::new(),
            config: OrderManagerConfig::default(),
        }
    }

    /// Create a new order manager with custom configuration.
    pub fn with_config(config: OrderManagerConfig) -> Self {
        Self {
            orders: HashMap::new(),
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
                debug!(oid = oid, "Order transitioned to CancelConfirmed, starting fill window");
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
                if o.state == OrderState::CancelPending
                    && o.state_changed_at.elapsed() > timeout
                {
                    Some(o.oid)
                } else {
                    None
                }
            })
            .collect()
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

/// Reconcile a single side: match current orders to target levels.
fn reconcile_side(
    current: &[&TrackedOrder],
    target: &[LadderLevel],
    side: Side,
    max_bps_diff: u16,
) -> Vec<LadderAction> {
    use crate::bps_diff;

    let mut actions = Vec::new();
    let mut matched_levels: HashSet<usize> = HashSet::new();

    // Match current orders to target levels
    for order in current {
        let mut found_match = false;
        for (i, level) in target.iter().enumerate() {
            if matched_levels.contains(&i) {
                continue;
            }
            // Check if order matches level (within tolerance)
            let price_diff = bps_diff(order.price, level.price);
            if price_diff <= max_bps_diff {
                matched_levels.insert(i);
                found_match = true;
                break;
            }
        }
        if !found_match {
            // Order doesn't match any target level - cancel it
            actions.push(LadderAction::Cancel { oid: order.oid });
        }
    }

    // Place orders for unmatched target levels
    for (i, level) in target.iter().enumerate() {
        if !matched_levels.contains(&i) && level.size > EPSILON {
            actions.push(LadderAction::Place {
                side,
                price: level.price,
                size: level.size,
            });
        }
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_order() {
        let mut mgr = OrderManager::new();
        let order = TrackedOrder::new(123, Side::Buy, 100.0, 1.0);
        mgr.add_order(order);

        let retrieved = mgr.get_order(123).unwrap();
        assert_eq!(retrieved.oid, 123);
        assert_eq!(retrieved.side, Side::Buy);
    }

    #[test]
    fn test_get_by_side() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));
        mgr.add_order(TrackedOrder::new(2, Side::Sell, 101.0, 0.5));

        let buy = mgr.get_by_side(Side::Buy).unwrap();
        assert_eq!(buy.oid, 1);

        let sell = mgr.get_by_side(Side::Sell).unwrap();
        assert_eq!(sell.oid, 2);
    }

    #[test]
    fn test_process_fill() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));

        // First fill - should work and transition to PartialFilled
        let (found, is_new, complete) = mgr.process_fill(1, 101, 0.5);
        assert!(found && is_new && !complete);

        let order = mgr.get_order(1).unwrap();
        assert!((order.filled - 0.5).abs() < f64::EPSILON);
        assert!((order.remaining() - 0.5).abs() < f64::EPSILON);
        assert_eq!(order.state, OrderState::PartialFilled);
    }

    #[test]
    fn test_is_filled() {
        let mut order = TrackedOrder::new(1, Side::Buy, 100.0, 1.0);
        assert!(!order.is_filled());

        order.filled = 1.0;
        assert!(order.is_filled());
    }

    #[test]
    fn test_cancel_then_fill_race() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));

        // Initiate cancel
        assert!(mgr.initiate_cancel(1));
        assert_eq!(
            mgr.get_order(1).unwrap().state,
            OrderState::CancelPending
        );

        // Fill arrives during cancel
        let (found, is_new, complete) = mgr.process_fill(1, 101, 1.0);
        assert!(found && is_new && complete);
        assert_eq!(
            mgr.get_order(1).unwrap().state,
            OrderState::FilledDuringCancel
        );

        // Cleanup should remove it
        let removed = mgr.cleanup();
        assert_eq!(removed, vec![1]);
        assert!(mgr.get_order(1).is_none());
    }

    #[test]
    fn test_cancel_confirmed_fill_window() {
        let config = OrderManagerConfig {
            fill_window_duration: Duration::from_millis(50),
            cancel_timeout: Duration::from_secs(30),
        };
        let mut mgr = OrderManager::with_config(config);

        // Place and initiate cancel
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));
        mgr.initiate_cancel(1);
        mgr.on_cancel_confirmed(1);

        // Immediately after confirm - should NOT be cleaned up
        let removed = mgr.cleanup();
        assert!(removed.is_empty());
        assert!(mgr.get_order(1).is_some());

        // Wait for fill window to expire
        std::thread::sleep(Duration::from_millis(60));

        // Now should be cleaned up
        let removed = mgr.cleanup();
        assert_eq!(removed, vec![1]);
        assert!(mgr.get_order(1).is_none());
    }

    #[test]
    fn test_late_fill_after_cancel_confirmed() {
        let config = OrderManagerConfig {
            fill_window_duration: Duration::from_millis(100),
            cancel_timeout: Duration::from_secs(30),
        };
        let mut mgr = OrderManager::with_config(config);

        // Place, initiate cancel, confirm
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));
        mgr.initiate_cancel(1);
        mgr.on_cancel_confirmed(1);

        // Fill arrives within window
        let (found, is_new, _) = mgr.process_fill(1, 101, 0.5);
        assert!(found && is_new);
        assert_eq!(
            mgr.get_order(1).unwrap().state,
            OrderState::FilledDuringCancel
        );

        // Should be cleaned up immediately (no need to wait)
        let removed = mgr.cleanup();
        assert_eq!(removed, vec![1]);
    }

    #[test]
    fn test_duplicate_fill_rejected() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 2.0));

        // First fill
        let (_, is_new1, _) = mgr.process_fill(1, 101, 1.0);
        assert!(is_new1);
        assert!((mgr.get_order(1).unwrap().filled - 1.0).abs() < f64::EPSILON);

        // Duplicate fill (same tid)
        let (_, is_new2, _) = mgr.process_fill(1, 101, 1.0);
        assert!(!is_new2);
        // Filled amount unchanged
        assert!((mgr.get_order(1).unwrap().filled - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_active_excludes_cancelling() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));
        mgr.add_order(TrackedOrder::new(2, Side::Buy, 99.0, 1.0));

        // Cancel one
        mgr.initiate_cancel(1);

        // get_all_by_side should only return order 2 (active orders)
        let active = mgr.get_all_by_side(Side::Buy);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].oid, 2);

        // get_all_by_side_including_pending should return both
        let all = mgr.get_all_by_side_including_pending(Side::Buy);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_cancel_failure_reverts_state() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));

        // Initiate cancel
        mgr.initiate_cancel(1);
        assert_eq!(mgr.get_order(1).unwrap().state, OrderState::CancelPending);

        // Cancel failed - should revert to Resting
        mgr.on_cancel_failed(1);
        assert_eq!(mgr.get_order(1).unwrap().state, OrderState::Resting);

        // Order should be visible to get_by_side again
        assert!(mgr.get_by_side(Side::Buy).is_some());
    }

    #[test]
    fn test_reconcile_ladder_empty_to_ladder() {
        let mgr = OrderManager::new();
        let ladder = Ladder {
            bids: vec![
                LadderLevel {
                    price: 99.0,
                    size: 1.0,
                    depth_bps: 10.0,
                },
                LadderLevel {
                    price: 98.0,
                    size: 0.5,
                    depth_bps: 20.0,
                },
            ],
            asks: vec![LadderLevel {
                price: 101.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
        };

        let actions = mgr.reconcile_ladder(&ladder, 5);

        // Should place 3 orders (2 bids, 1 ask)
        assert_eq!(actions.len(), 3);
        let place_count = actions
            .iter()
            .filter(|a| matches!(a, LadderAction::Place { .. }))
            .count();
        assert_eq!(place_count, 3);
    }

    #[test]
    fn test_reconcile_ladder_orders_match() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 99.0, 1.0));
        mgr.add_order(TrackedOrder::new(2, Side::Sell, 101.0, 1.0));

        let ladder = Ladder {
            bids: vec![LadderLevel {
                price: 99.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
            asks: vec![LadderLevel {
                price: 101.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
        };

        let actions = mgr.reconcile_ladder(&ladder, 5);

        // Orders match target levels - no actions needed
        assert!(actions.is_empty());
    }

    #[test]
    fn test_reconcile_ladder_cancel_stale() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 95.0, 1.0)); // Too far from target
        mgr.add_order(TrackedOrder::new(2, Side::Sell, 105.0, 1.0)); // Too far from target

        let ladder = Ladder {
            bids: vec![LadderLevel {
                price: 99.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
            asks: vec![LadderLevel {
                price: 101.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
        };

        let actions = mgr.reconcile_ladder(&ladder, 5); // 5 bps tolerance

        // Should cancel 2 stale orders and place 2 new ones
        assert_eq!(actions.len(), 4);
        let cancel_count = actions
            .iter()
            .filter(|a| matches!(a, LadderAction::Cancel { .. }))
            .count();
        let place_count = actions
            .iter()
            .filter(|a| matches!(a, LadderAction::Place { .. }))
            .count();
        assert_eq!(cancel_count, 2);
        assert_eq!(place_count, 2);
    }

    #[test]
    fn test_reconcile_ladder_partial_match() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 99.0, 1.0)); // Matches first level
        // No ask order

        let ladder = Ladder {
            bids: vec![
                LadderLevel {
                    price: 99.0,
                    size: 1.0,
                    depth_bps: 10.0,
                },
                LadderLevel {
                    price: 98.0,
                    size: 0.5,
                    depth_bps: 20.0,
                },
            ],
            asks: vec![LadderLevel {
                price: 101.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
        };

        let actions = mgr.reconcile_ladder(&ladder, 5);

        // Should place 1 bid (98.0) and 1 ask (101.0)
        assert_eq!(actions.len(), 2);
        let place_count = actions
            .iter()
            .filter(|a| matches!(a, LadderAction::Place { .. }))
            .count();
        assert_eq!(place_count, 2);
    }
}
