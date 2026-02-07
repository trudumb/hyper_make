//! Order lifecycle tracking for execution quality analysis.
//!
//! This module provides order lifecycle tracking for analyzing order execution quality:
//! - Order state transitions (Placed -> PartiallyFilled -> Filled/Cancelled)
//! - Time to terminal state
//! - Cancel analysis (before/after fill, time to cancel)
//! - Thread-safe order tracking
//!
//! # Usage
//!
//! ```ignore
//! use std::sync::Arc;
//!
//! let tracker = OrderLifecycleTracker::new(1000);
//!
//! // Create an order
//! tracker.create_order(CreateOrderParams {
//!     order_id: 12345,
//!     client_order_id: "cloid-uuid".to_string(),
//!     symbol: "BTC".to_string(),
//!     side: Side::Bid,
//!     price: 50000.0,
//!     size: 0.01,
//!     timestamp_ms: current_time_ms(),
//! });
//!
//! // Update order state
//! tracker.update_order(12345, OrderEvent {
//!     timestamp: current_time_ms() + 100,
//!     state: OrderState::PartiallyFilled,
//!     filled_size: 0.005,
//!     remaining_size: 0.005,
//!     reason: None,
//! });
//!
//! // Get cancel analysis
//! let analysis = tracker.cancel_analysis();
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

// Re-export Side from fill_tracker for convenience (crate-visible only)
pub(crate) use super::fill_tracker::Side;

/// Order lifecycle state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderState {
    /// Order has been placed but not yet confirmed
    Placed,
    /// Order has been partially filled
    PartiallyFilled,
    /// Order has been fully filled
    Filled,
    /// Order has been cancelled
    Cancelled,
    /// Order has expired
    Expired,
    /// Order was rejected by the exchange
    Rejected,
}

impl OrderState {
    /// Check if this is a terminal state (no more transitions possible).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderState::Filled | OrderState::Cancelled | OrderState::Expired | OrderState::Rejected
        )
    }

    /// Get a human-readable name for the state.
    pub fn as_str(&self) -> &'static str {
        match self {
            OrderState::Placed => "placed",
            OrderState::PartiallyFilled => "partially_filled",
            OrderState::Filled => "filled",
            OrderState::Cancelled => "cancelled",
            OrderState::Expired => "expired",
            OrderState::Rejected => "rejected",
        }
    }
}

/// An event in the order lifecycle.
#[derive(Debug, Clone)]
pub struct OrderEvent {
    /// Timestamp of the event in milliseconds since epoch
    pub timestamp: u64,
    /// New state after the event
    pub state: OrderState,
    /// Size filled so far (cumulative)
    pub filled_size: f64,
    /// Size remaining after this event
    pub remaining_size: f64,
    /// Optional reason for the state change (e.g., rejection reason)
    pub reason: Option<String>,
}

impl OrderEvent {
    /// Create a new order event.
    pub fn new(timestamp: u64, state: OrderState, filled_size: f64, remaining_size: f64) -> Self {
        Self {
            timestamp,
            state,
            filled_size,
            remaining_size,
            reason: None,
        }
    }

    /// Create a new order event with a reason.
    pub fn with_reason(
        timestamp: u64,
        state: OrderState,
        filled_size: f64,
        remaining_size: f64,
        reason: String,
    ) -> Self {
        Self {
            timestamp,
            state,
            filled_size,
            remaining_size,
            reason: Some(reason),
        }
    }
}

/// Complete lifecycle of a single order.
#[derive(Debug, Clone)]
pub struct OrderLifecycle {
    /// Order ID from the exchange
    pub order_id: u64,
    /// Client order ID (for correlation)
    pub client_order_id: String,
    /// Trading symbol
    pub symbol: String,
    /// Order side
    pub side: Side,
    /// Limit price
    pub price: f64,
    /// Original order size
    pub original_size: f64,
    /// When the order was created
    pub created_at: u64,
    /// List of events in this order's lifecycle
    pub events: Vec<OrderEvent>,
    /// Current state of the order
    pub current_state: OrderState,
    /// Total amount filled
    pub total_filled: f64,
}

impl OrderLifecycle {
    /// Create a new order lifecycle.
    pub fn new(
        order_id: u64,
        client_order_id: String,
        symbol: String,
        side: Side,
        price: f64,
        size: f64,
        timestamp: u64,
    ) -> Self {
        let initial_event = OrderEvent::new(timestamp, OrderState::Placed, 0.0, size);

        Self {
            order_id,
            client_order_id,
            symbol,
            side,
            price,
            original_size: size,
            created_at: timestamp,
            events: vec![initial_event],
            current_state: OrderState::Placed,
            total_filled: 0.0,
        }
    }

    /// Add an event to the order lifecycle.
    ///
    /// Updates the current state and total filled based on the event.
    pub fn add_event(&mut self, event: OrderEvent) {
        self.current_state = event.state;
        self.total_filled = event.filled_size;
        self.events.push(event);
    }

    /// Check if the order is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.current_state.is_terminal()
    }

    /// Get the fill ratio (filled / original size).
    pub fn fill_ratio(&self) -> f64 {
        if self.original_size > 0.0 {
            self.total_filled / self.original_size
        } else {
            0.0
        }
    }

    /// Get the time from placed to terminal state.
    ///
    /// Returns None if the order is not yet in a terminal state.
    pub fn time_to_terminal(&self) -> Option<u64> {
        if !self.is_terminal() {
            return None;
        }

        self.events.last().map(|last| {
            last.timestamp.saturating_sub(self.created_at)
        })
    }

    /// Get the remaining size.
    pub fn remaining(&self) -> f64 {
        (self.original_size - self.total_filled).max(0.0)
    }

    /// Check if this order was cancelled before any fill occurred.
    pub fn was_cancelled_before_fill(&self) -> bool {
        self.current_state == OrderState::Cancelled && self.total_filled == 0.0
    }

    /// Check if this order was cancelled after a partial fill.
    pub fn was_cancelled_after_partial(&self) -> bool {
        self.current_state == OrderState::Cancelled && self.total_filled > 0.0
    }

    /// Get the time from placed to first fill.
    ///
    /// Returns None if no fill has occurred yet.
    pub fn time_to_first_fill(&self) -> Option<u64> {
        for event in &self.events {
            if event.filled_size > 0.0 {
                return Some(event.timestamp.saturating_sub(self.created_at));
            }
        }
        None
    }

    /// Get the last event timestamp.
    pub fn last_event_time(&self) -> u64 {
        self.events
            .last()
            .map(|e| e.timestamp)
            .unwrap_or(self.created_at)
    }

    /// Get the order age (time since creation, based on last event).
    pub fn age(&self) -> u64 {
        self.last_event_time().saturating_sub(self.created_at)
    }
}

/// Analysis of cancel patterns.
#[derive(Debug, Clone, Default)]
pub struct CancelAnalysis {
    /// Total number of cancelled orders
    pub total_cancels: usize,
    /// Orders cancelled before any fill occurred
    pub cancel_before_any_fill: usize,
    /// Orders cancelled after partial fill
    pub cancel_after_partial: usize,
    /// Average time from placement to cancel in milliseconds
    pub avg_time_to_cancel_ms: f64,
}

impl CancelAnalysis {
    /// Get the ratio of cancels that happened before any fill.
    pub fn cancel_before_fill_ratio(&self) -> f64 {
        if self.total_cancels > 0 {
            self.cancel_before_any_fill as f64 / self.total_cancels as f64
        } else {
            0.0
        }
    }

    /// Get the ratio of cancels that happened after a partial fill.
    pub fn cancel_after_partial_ratio(&self) -> f64 {
        if self.total_cancels > 0 {
            self.cancel_after_partial as f64 / self.total_cancels as f64
        } else {
            0.0
        }
    }
}

/// Parameters for creating a new tracked order.
pub struct CreateOrderParams {
    pub order_id: u64,
    pub client_order_id: String,
    pub symbol: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub timestamp_ms: u64,
}

/// Thread-safe order lifecycle tracker.
///
/// Tracks active and completed orders with their full lifecycle history.
/// Uses RwLock for thread-safety, making it suitable for concurrent access.
pub struct OrderLifecycleTracker {
    /// Active orders (not yet in terminal state)
    orders: Arc<RwLock<HashMap<u64, OrderLifecycle>>>,
    /// Maximum number of completed orders to keep in history
    max_history: usize,
    /// Completed orders (in terminal state)
    completed_orders: Arc<RwLock<VecDeque<OrderLifecycle>>>,
}

impl OrderLifecycleTracker {
    /// Create a new order lifecycle tracker.
    ///
    /// # Arguments
    ///
    /// * `max_history` - Maximum number of completed orders to keep in history
    pub fn new(max_history: usize) -> Self {
        Self {
            orders: Arc::new(RwLock::new(HashMap::new())),
            max_history,
            completed_orders: Arc::new(RwLock::new(VecDeque::with_capacity(max_history))),
        }
    }

    /// Create a new order and start tracking its lifecycle.
    ///
    /// # Arguments
    ///
    /// * `params` - Order creation parameters
    pub fn create_order(&self, params: CreateOrderParams) {
        let lifecycle = OrderLifecycle::new(
            params.order_id,
            params.client_order_id,
            params.symbol,
            params.side,
            params.price,
            params.size,
            params.timestamp_ms,
        );

        let mut orders = self.orders.write().unwrap();
        orders.insert(params.order_id, lifecycle);
    }

    /// Update an order with a new event.
    ///
    /// If the event transitions the order to a terminal state, it will be
    /// moved to the completed orders history.
    ///
    /// # Arguments
    ///
    /// * `order_id` - The order to update
    /// * `event` - The new event
    ///
    /// # Returns
    ///
    /// true if the order was found and updated, false otherwise
    pub fn update_order(&self, order_id: u64, event: OrderEvent) -> bool {
        let mut orders = self.orders.write().unwrap();

        if let Some(lifecycle) = orders.get_mut(&order_id) {
            lifecycle.add_event(event);

            // If terminal, move to completed
            if lifecycle.is_terminal() {
                let completed = lifecycle.clone();
                orders.remove(&order_id);
                drop(orders); // Release lock before acquiring another

                self.add_to_completed(completed);
            }
            true
        } else {
            false
        }
    }

    /// Add a completed order to history.
    fn add_to_completed(&self, lifecycle: OrderLifecycle) {
        let mut completed = self.completed_orders.write().unwrap();
        completed.push_back(lifecycle);

        // Maintain max history
        while completed.len() > self.max_history {
            completed.pop_front();
        }
    }

    /// Get an order by ID.
    ///
    /// Searches both active and completed orders.
    pub fn get_order(&self, order_id: u64) -> Option<OrderLifecycle> {
        // Try active orders first
        {
            let orders = self.orders.read().unwrap();
            if let Some(lifecycle) = orders.get(&order_id) {
                return Some(lifecycle.clone());
            }
        }

        // Try completed orders
        {
            let completed = self.completed_orders.read().unwrap();
            for lifecycle in completed.iter() {
                if lifecycle.order_id == order_id {
                    return Some(lifecycle.clone());
                }
            }
        }

        None
    }

    /// Get all active (non-terminal) orders.
    pub fn active_orders(&self) -> Vec<OrderLifecycle> {
        let orders = self.orders.read().unwrap();
        orders.values().cloned().collect()
    }

    /// Get the N most recent completed orders.
    ///
    /// Returns orders in reverse chronological order (most recent first).
    pub fn completed_orders(&self, n: usize) -> Vec<OrderLifecycle> {
        let completed = self.completed_orders.read().unwrap();
        completed.iter().rev().take(n).cloned().collect()
    }

    /// Analyze cancel patterns from completed orders.
    pub fn cancel_analysis(&self) -> CancelAnalysis {
        let completed = self.completed_orders.read().unwrap();

        let cancelled: Vec<_> = completed
            .iter()
            .filter(|o| o.current_state == OrderState::Cancelled)
            .collect();

        if cancelled.is_empty() {
            return CancelAnalysis::default();
        }

        let total_cancels = cancelled.len();
        let cancel_before_any_fill = cancelled.iter().filter(|o| o.total_filled == 0.0).count();
        let cancel_after_partial = cancelled.iter().filter(|o| o.total_filled > 0.0).count();

        let total_time_to_cancel: u64 = cancelled.iter().filter_map(|o| o.time_to_terminal()).sum();

        let orders_with_time = cancelled
            .iter()
            .filter_map(|o| o.time_to_terminal())
            .count();
        let avg_time_to_cancel_ms = if orders_with_time > 0 {
            total_time_to_cancel as f64 / orders_with_time as f64
        } else {
            0.0
        };

        CancelAnalysis {
            total_cancels,
            cancel_before_any_fill,
            cancel_after_partial,
            avg_time_to_cancel_ms,
        }
    }

    /// Get the number of active orders.
    pub fn active_count(&self) -> usize {
        let orders = self.orders.read().unwrap();
        orders.len()
    }

    /// Get the number of completed orders in history.
    pub fn completed_count(&self) -> usize {
        let completed = self.completed_orders.read().unwrap();
        completed.len()
    }

    /// Get active orders by side.
    pub fn active_orders_by_side(&self, side: Side) -> Vec<OrderLifecycle> {
        let orders = self.orders.read().unwrap();
        orders
            .values()
            .filter(|o| o.side == side)
            .cloned()
            .collect()
    }

    /// Get active orders by symbol.
    pub fn active_orders_by_symbol(&self, symbol: &str) -> Vec<OrderLifecycle> {
        let orders = self.orders.read().unwrap();
        orders
            .values()
            .filter(|o| o.symbol == symbol)
            .cloned()
            .collect()
    }

    /// Remove an active order without moving to completed history.
    ///
    /// Useful for cleaning up orders that were never actually placed.
    pub fn remove_order(&self, order_id: u64) -> Option<OrderLifecycle> {
        let mut orders = self.orders.write().unwrap();
        orders.remove(&order_id)
    }

    /// Clear all active orders.
    pub fn clear_active(&self) {
        let mut orders = self.orders.write().unwrap();
        orders.clear();
    }

    /// Clear completed orders history.
    pub fn clear_completed(&self) {
        let mut completed = self.completed_orders.write().unwrap();
        completed.clear();
    }

    /// Get fill statistics from completed orders.
    pub fn fill_statistics(&self) -> FillStatistics {
        let completed = self.completed_orders.read().unwrap();

        if completed.is_empty() {
            return FillStatistics::default();
        }

        let filled_orders: Vec<_> = completed
            .iter()
            .filter(|o| o.current_state == OrderState::Filled)
            .collect();

        let partial_cancelled: Vec<_> = completed
            .iter()
            .filter(|o| o.was_cancelled_after_partial())
            .collect();

        let total_orders = completed.len();
        let fully_filled = filled_orders.len();
        let partially_filled_cancelled = partial_cancelled.len();

        // Calculate average fill ratio
        let total_fill_ratio: f64 = completed.iter().map(|o| o.fill_ratio()).sum();
        let avg_fill_ratio = total_fill_ratio / total_orders as f64;

        // Calculate average time to fill for filled orders
        let fill_times: Vec<u64> = filled_orders
            .iter()
            .filter_map(|o| o.time_to_terminal())
            .collect();

        let avg_time_to_fill_ms = if !fill_times.is_empty() {
            fill_times.iter().sum::<u64>() as f64 / fill_times.len() as f64
        } else {
            0.0
        };

        FillStatistics {
            total_orders,
            fully_filled,
            partially_filled_cancelled,
            avg_fill_ratio,
            avg_time_to_fill_ms,
        }
    }
}

/// Statistics about fill quality from completed orders.
#[derive(Debug, Clone, Default)]
pub struct FillStatistics {
    /// Total orders in the history
    pub total_orders: usize,
    /// Orders that were fully filled
    pub fully_filled: usize,
    /// Orders that were partially filled then cancelled
    pub partially_filled_cancelled: usize,
    /// Average fill ratio across all orders
    pub avg_fill_ratio: f64,
    /// Average time to fully fill in milliseconds
    pub avg_time_to_fill_ms: f64,
}

impl FillStatistics {
    /// Get the full fill rate (fully_filled / total_orders).
    pub fn full_fill_rate(&self) -> f64 {
        if self.total_orders > 0 {
            self.fully_filled as f64 / self.total_orders as f64
        } else {
            0.0
        }
    }
}

// Ensure OrderLifecycleTracker is Send + Sync
static_assertions::assert_impl_all!(OrderLifecycleTracker: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_state_is_terminal() {
        assert!(!OrderState::Placed.is_terminal());
        assert!(!OrderState::PartiallyFilled.is_terminal());
        assert!(OrderState::Filled.is_terminal());
        assert!(OrderState::Cancelled.is_terminal());
        assert!(OrderState::Expired.is_terminal());
        assert!(OrderState::Rejected.is_terminal());
    }

    #[test]
    fn test_order_state_as_str() {
        assert_eq!(OrderState::Placed.as_str(), "placed");
        assert_eq!(OrderState::Filled.as_str(), "filled");
        assert_eq!(OrderState::Cancelled.as_str(), "cancelled");
    }

    #[test]
    fn test_order_event_new() {
        let event = OrderEvent::new(1000, OrderState::PartiallyFilled, 0.5, 0.5);
        assert_eq!(event.timestamp, 1000);
        assert_eq!(event.state, OrderState::PartiallyFilled);
        assert!((event.filled_size - 0.5).abs() < f64::EPSILON);
        assert!(event.reason.is_none());
    }

    #[test]
    fn test_order_event_with_reason() {
        let event = OrderEvent::with_reason(
            1000,
            OrderState::Rejected,
            0.0,
            1.0,
            "insufficient funds".to_string(),
        );
        assert_eq!(event.reason, Some("insufficient funds".to_string()));
    }

    #[test]
    fn test_order_lifecycle_new() {
        let lifecycle = OrderLifecycle::new(
            123,
            "cloid-1".to_string(),
            "BTC".to_string(),
            Side::Bid,
            50000.0,
            0.01,
            1000,
        );

        assert_eq!(lifecycle.order_id, 123);
        assert_eq!(lifecycle.client_order_id, "cloid-1");
        assert_eq!(lifecycle.symbol, "BTC");
        assert_eq!(lifecycle.side, Side::Bid);
        assert!((lifecycle.price - 50000.0).abs() < f64::EPSILON);
        assert!((lifecycle.original_size - 0.01).abs() < f64::EPSILON);
        assert_eq!(lifecycle.created_at, 1000);
        assert_eq!(lifecycle.current_state, OrderState::Placed);
        assert_eq!(lifecycle.events.len(), 1);
        assert!((lifecycle.total_filled - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_order_lifecycle_add_event() {
        let mut lifecycle = OrderLifecycle::new(
            123,
            "cloid-1".to_string(),
            "BTC".to_string(),
            Side::Bid,
            50000.0,
            0.01,
            1000,
        );

        lifecycle.add_event(OrderEvent::new(
            1100,
            OrderState::PartiallyFilled,
            0.005,
            0.005,
        ));

        assert_eq!(lifecycle.current_state, OrderState::PartiallyFilled);
        assert!((lifecycle.total_filled - 0.005).abs() < f64::EPSILON);
        assert_eq!(lifecycle.events.len(), 2);
    }

    #[test]
    fn test_order_lifecycle_is_terminal() {
        let mut lifecycle = OrderLifecycle::new(
            123,
            "cloid-1".to_string(),
            "BTC".to_string(),
            Side::Bid,
            50000.0,
            0.01,
            1000,
        );

        assert!(!lifecycle.is_terminal());

        lifecycle.add_event(OrderEvent::new(1100, OrderState::Filled, 0.01, 0.0));
        assert!(lifecycle.is_terminal());
    }

    #[test]
    fn test_order_lifecycle_fill_ratio() {
        let mut lifecycle = OrderLifecycle::new(
            123,
            "cloid-1".to_string(),
            "BTC".to_string(),
            Side::Bid,
            50000.0,
            0.01,
            1000,
        );

        assert!((lifecycle.fill_ratio() - 0.0).abs() < f64::EPSILON);

        lifecycle.add_event(OrderEvent::new(
            1100,
            OrderState::PartiallyFilled,
            0.005,
            0.005,
        ));
        assert!((lifecycle.fill_ratio() - 0.5).abs() < f64::EPSILON);

        lifecycle.add_event(OrderEvent::new(1200, OrderState::Filled, 0.01, 0.0));
        assert!((lifecycle.fill_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_order_lifecycle_time_to_terminal() {
        let mut lifecycle = OrderLifecycle::new(
            123,
            "cloid-1".to_string(),
            "BTC".to_string(),
            Side::Bid,
            50000.0,
            0.01,
            1000,
        );

        // Not terminal yet
        assert!(lifecycle.time_to_terminal().is_none());

        // Add terminal event
        lifecycle.add_event(OrderEvent::new(1500, OrderState::Filled, 0.01, 0.0));
        assert_eq!(lifecycle.time_to_terminal(), Some(500));
    }

    #[test]
    fn test_order_lifecycle_was_cancelled_before_fill() {
        let mut lifecycle = OrderLifecycle::new(
            123,
            "cloid-1".to_string(),
            "BTC".to_string(),
            Side::Bid,
            50000.0,
            0.01,
            1000,
        );

        lifecycle.add_event(OrderEvent::new(1100, OrderState::Cancelled, 0.0, 0.01));

        assert!(lifecycle.was_cancelled_before_fill());
        assert!(!lifecycle.was_cancelled_after_partial());
    }

    #[test]
    fn test_order_lifecycle_was_cancelled_after_partial() {
        let mut lifecycle = OrderLifecycle::new(
            123,
            "cloid-1".to_string(),
            "BTC".to_string(),
            Side::Bid,
            50000.0,
            0.01,
            1000,
        );

        lifecycle.add_event(OrderEvent::new(
            1100,
            OrderState::PartiallyFilled,
            0.005,
            0.005,
        ));
        lifecycle.add_event(OrderEvent::new(1200, OrderState::Cancelled, 0.005, 0.005));

        assert!(!lifecycle.was_cancelled_before_fill());
        assert!(lifecycle.was_cancelled_after_partial());
    }

    #[test]
    fn test_order_lifecycle_time_to_first_fill() {
        let mut lifecycle = OrderLifecycle::new(
            123,
            "cloid-1".to_string(),
            "BTC".to_string(),
            Side::Bid,
            50000.0,
            0.01,
            1000,
        );

        // No fills yet
        assert!(lifecycle.time_to_first_fill().is_none());

        // First fill at 1100
        lifecycle.add_event(OrderEvent::new(
            1100,
            OrderState::PartiallyFilled,
            0.005,
            0.005,
        ));
        assert_eq!(lifecycle.time_to_first_fill(), Some(100));

        // Second fill at 1200 shouldn't change time to first fill
        lifecycle.add_event(OrderEvent::new(1200, OrderState::Filled, 0.01, 0.0));
        assert_eq!(lifecycle.time_to_first_fill(), Some(100));
    }

    #[test]
    fn test_order_lifecycle_tracker_new() {
        let tracker = OrderLifecycleTracker::new(100);
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.completed_count(), 0);
    }

    #[test]
    fn test_order_lifecycle_tracker_create_order() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(CreateOrderParams {
            order_id: 123,
            client_order_id: "cloid-1".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 50000.0,
            size: 0.01,
            timestamp_ms: 1000,
        });

        assert_eq!(tracker.active_count(), 1);

        let order = tracker.get_order(123).unwrap();
        assert_eq!(order.order_id, 123);
        assert_eq!(order.current_state, OrderState::Placed);
    }

    #[test]
    fn test_order_lifecycle_tracker_update_order() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(CreateOrderParams {
            order_id: 123,
            client_order_id: "cloid-1".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 50000.0,
            size: 0.01,
            timestamp_ms: 1000,
        });

        // Partial fill
        assert!(tracker.update_order(
            123,
            OrderEvent::new(1100, OrderState::PartiallyFilled, 0.005, 0.005)
        ));

        let order = tracker.get_order(123).unwrap();
        assert_eq!(order.current_state, OrderState::PartiallyFilled);
        assert_eq!(tracker.active_count(), 1);

        // Full fill (terminal)
        assert!(tracker.update_order(123, OrderEvent::new(1200, OrderState::Filled, 0.01, 0.0)));

        // Should be moved to completed
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.completed_count(), 1);

        // Should still be findable via get_order
        let order = tracker.get_order(123).unwrap();
        assert_eq!(order.current_state, OrderState::Filled);
    }

    #[test]
    fn test_order_lifecycle_tracker_update_nonexistent() {
        let tracker = OrderLifecycleTracker::new(100);

        assert!(!tracker.update_order(999, OrderEvent::new(1100, OrderState::Filled, 0.01, 0.0)));
    }

    #[test]
    fn test_order_lifecycle_tracker_active_orders() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(CreateOrderParams {
            order_id: 1,
            client_order_id: "cloid-1".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 50000.0,
            size: 0.01,
            timestamp_ms: 1000,
        });
        tracker.create_order(CreateOrderParams {
            order_id: 2,
            client_order_id: "cloid-2".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Ask,
            price: 50100.0,
            size: 0.01,
            timestamp_ms: 1000,
        });

        let active = tracker.active_orders();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_order_lifecycle_tracker_completed_orders() {
        let tracker = OrderLifecycleTracker::new(100);

        for i in 0..5 {
            tracker.create_order(CreateOrderParams {
                order_id: i,
                client_order_id: format!("cloid-{}", i),
                symbol: "BTC".to_string(),
                side: Side::Bid,
                price: 50000.0,
                size: 0.01,
                timestamp_ms: 1000 + i,
            });
            tracker.update_order(i, OrderEvent::new(1100 + i, OrderState::Filled, 0.01, 0.0));
        }

        let completed = tracker.completed_orders(3);
        assert_eq!(completed.len(), 3);
        // Should be in reverse order (most recent first)
        assert_eq!(completed[0].order_id, 4);
        assert_eq!(completed[1].order_id, 3);
        assert_eq!(completed[2].order_id, 2);
    }

    #[test]
    fn test_order_lifecycle_tracker_max_history() {
        let tracker = OrderLifecycleTracker::new(3); // Only keep 3

        for i in 0..5 {
            tracker.create_order(CreateOrderParams {
                order_id: i,
                client_order_id: format!("cloid-{}", i),
                symbol: "BTC".to_string(),
                side: Side::Bid,
                price: 50000.0,
                size: 0.01,
                timestamp_ms: 1000,
            });
            tracker.update_order(i, OrderEvent::new(1100, OrderState::Filled, 0.01, 0.0));
        }

        assert_eq!(tracker.completed_count(), 3);

        // Only orders 2, 3, 4 should be in history (0, 1 were evicted)
        assert!(tracker.get_order(0).is_none());
        assert!(tracker.get_order(1).is_none());
        assert!(tracker.get_order(2).is_some());
        assert!(tracker.get_order(3).is_some());
        assert!(tracker.get_order(4).is_some());
    }

    #[test]
    fn test_order_lifecycle_tracker_cancel_analysis() {
        let tracker = OrderLifecycleTracker::new(100);

        // 2 cancelled before fill
        for i in 0..2 {
            tracker.create_order(CreateOrderParams {
                order_id: i,
                client_order_id: format!("cloid-{}", i),
                symbol: "BTC".to_string(),
                side: Side::Bid,
                price: 50000.0,
                size: 0.01,
                timestamp_ms: 1000,
            });
            tracker.update_order(i, OrderEvent::new(1100, OrderState::Cancelled, 0.0, 0.01));
        }

        // 1 cancelled after partial
        tracker.create_order(CreateOrderParams {
            order_id: 2,
            client_order_id: "cloid-2".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 50000.0,
            size: 0.01,
            timestamp_ms: 1000,
        });
        tracker.update_order(
            2,
            OrderEvent::new(1050, OrderState::PartiallyFilled, 0.005, 0.005),
        );
        tracker.update_order(
            2,
            OrderEvent::new(1200, OrderState::Cancelled, 0.005, 0.005),
        );

        // 1 filled (not cancelled)
        tracker.create_order(CreateOrderParams {
            order_id: 3,
            client_order_id: "cloid-3".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 50000.0,
            size: 0.01,
            timestamp_ms: 1000,
        });
        tracker.update_order(3, OrderEvent::new(1100, OrderState::Filled, 0.01, 0.0));

        let analysis = tracker.cancel_analysis();
        assert_eq!(analysis.total_cancels, 3);
        assert_eq!(analysis.cancel_before_any_fill, 2);
        assert_eq!(analysis.cancel_after_partial, 1);
        assert!(analysis.avg_time_to_cancel_ms > 0.0);
    }

    #[test]
    fn test_order_lifecycle_tracker_active_orders_by_side() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(CreateOrderParams {
            order_id: 1,
            client_order_id: "cloid-1".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 50000.0,
            size: 0.01,
            timestamp_ms: 1000,
        });
        tracker.create_order(CreateOrderParams {
            order_id: 2,
            client_order_id: "cloid-2".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Ask,
            price: 50100.0,
            size: 0.01,
            timestamp_ms: 1000,
        });
        tracker.create_order(CreateOrderParams {
            order_id: 3,
            client_order_id: "cloid-3".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 49900.0,
            size: 0.01,
            timestamp_ms: 1000,
        });

        let bids = tracker.active_orders_by_side(Side::Bid);
        let asks = tracker.active_orders_by_side(Side::Ask);

        assert_eq!(bids.len(), 2);
        assert_eq!(asks.len(), 1);
    }

    #[test]
    fn test_order_lifecycle_tracker_active_orders_by_symbol() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(CreateOrderParams {
            order_id: 1,
            client_order_id: "cloid-1".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 50000.0,
            size: 0.01,
            timestamp_ms: 1000,
        });
        tracker.create_order(CreateOrderParams {
            order_id: 2,
            client_order_id: "cloid-2".to_string(),
            symbol: "ETH".to_string(),
            side: Side::Bid,
            price: 3000.0,
            size: 0.1,
            timestamp_ms: 1000,
        });

        let btc_orders = tracker.active_orders_by_symbol("BTC");
        let eth_orders = tracker.active_orders_by_symbol("ETH");

        assert_eq!(btc_orders.len(), 1);
        assert_eq!(eth_orders.len(), 1);
    }

    #[test]
    fn test_order_lifecycle_tracker_remove_order() {
        let tracker = OrderLifecycleTracker::new(100);

        tracker.create_order(CreateOrderParams {
            order_id: 1,
            client_order_id: "cloid-1".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 50000.0,
            size: 0.01,
            timestamp_ms: 1000,
        });

        let removed = tracker.remove_order(1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().order_id, 1);
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.completed_count(), 0); // Not moved to completed
    }

    #[test]
    fn test_order_lifecycle_tracker_clear_active() {
        let tracker = OrderLifecycleTracker::new(100);

        for i in 0..3 {
            tracker.create_order(CreateOrderParams {
                order_id: i,
                client_order_id: format!("cloid-{}", i),
                symbol: "BTC".to_string(),
                side: Side::Bid,
                price: 50000.0,
                size: 0.01,
                timestamp_ms: 1000,
            });
        }

        assert_eq!(tracker.active_count(), 3);
        tracker.clear_active();
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn test_order_lifecycle_tracker_clear_completed() {
        let tracker = OrderLifecycleTracker::new(100);

        for i in 0..3 {
            tracker.create_order(CreateOrderParams {
                order_id: i,
                client_order_id: format!("cloid-{}", i),
                symbol: "BTC".to_string(),
                side: Side::Bid,
                price: 50000.0,
                size: 0.01,
                timestamp_ms: 1000,
            });
            tracker.update_order(i, OrderEvent::new(1100, OrderState::Filled, 0.01, 0.0));
        }

        assert_eq!(tracker.completed_count(), 3);
        tracker.clear_completed();
        assert_eq!(tracker.completed_count(), 0);
    }

    #[test]
    fn test_order_lifecycle_tracker_fill_statistics() {
        let tracker = OrderLifecycleTracker::new(100);

        // 2 fully filled
        for i in 0..2 {
            tracker.create_order(CreateOrderParams {
                order_id: i,
                client_order_id: format!("cloid-{}", i),
                symbol: "BTC".to_string(),
                side: Side::Bid,
                price: 50000.0,
                size: 0.01,
                timestamp_ms: 1000,
            });
            tracker.update_order(i, OrderEvent::new(1100, OrderState::Filled, 0.01, 0.0));
        }

        // 1 partially filled then cancelled
        tracker.create_order(CreateOrderParams {
            order_id: 2,
            client_order_id: "cloid-2".to_string(),
            symbol: "BTC".to_string(),
            side: Side::Bid,
            price: 50000.0,
            size: 0.01,
            timestamp_ms: 1000,
        });
        tracker.update_order(
            2,
            OrderEvent::new(1050, OrderState::PartiallyFilled, 0.005, 0.005),
        );
        tracker.update_order(
            2,
            OrderEvent::new(1100, OrderState::Cancelled, 0.005, 0.005),
        );

        let stats = tracker.fill_statistics();
        assert_eq!(stats.total_orders, 3);
        assert_eq!(stats.fully_filled, 2);
        assert_eq!(stats.partially_filled_cancelled, 1);
        assert!(stats.avg_fill_ratio > 0.0);
        assert!(stats.avg_time_to_fill_ms > 0.0);
    }

    #[test]
    fn test_cancel_analysis_ratios() {
        let analysis = CancelAnalysis {
            total_cancels: 10,
            cancel_before_any_fill: 6,
            cancel_after_partial: 4,
            avg_time_to_cancel_ms: 100.0,
        };

        assert!((analysis.cancel_before_fill_ratio() - 0.6).abs() < f64::EPSILON);
        assert!((analysis.cancel_after_partial_ratio() - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fill_statistics_full_fill_rate() {
        let stats = FillStatistics {
            total_orders: 10,
            fully_filled: 8,
            partially_filled_cancelled: 2,
            avg_fill_ratio: 0.9,
            avg_time_to_fill_ms: 100.0,
        };

        assert!((stats.full_fill_rate() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let tracker = Arc::new(OrderLifecycleTracker::new(100));

        let handles: Vec<_> = (0..4)
            .map(|thread_id| {
                let tracker = Arc::clone(&tracker);
                thread::spawn(move || {
                    for i in 0..25 {
                        let order_id = thread_id * 100 + i;
                        tracker.create_order(CreateOrderParams {
                            order_id,
                            client_order_id: format!("cloid-{}", order_id),
                            symbol: "BTC".to_string(),
                            side: Side::Bid,
                            price: 50000.0,
                            size: 0.01,
                            timestamp_ms: 1000,
                        });
                        tracker.update_order(
                            order_id,
                            OrderEvent::new(1100, OrderState::Filled, 0.01, 0.0),
                        );
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All orders should be completed
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.completed_count(), 100);
    }
}
