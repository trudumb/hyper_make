//! Order state management for tracking resting orders.

use std::collections::HashMap;

use crate::{bps_diff, EPSILON};

use super::config::Quote;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    /// Order is resting on the exchange
    Resting,
    /// Cancel request has been sent, awaiting confirmation
    Cancelling,
}

/// A tracked order with its current state.
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
}

impl TrackedOrder {
    /// Create a new tracked order in Resting state.
    pub fn new(oid: u64, side: Side, price: f64, size: f64) -> Self {
        Self {
            oid,
            side,
            price,
            size,
            filled: 0.0,
            state: OrderState::Resting,
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
}

/// Manages tracked orders for the market maker.
#[derive(Debug, Default)]
pub struct OrderManager {
    /// Orders indexed by order ID
    orders: HashMap<u64, TrackedOrder>,
}

impl OrderManager {
    /// Create a new order manager.
    pub fn new() -> Self {
        Self {
            orders: HashMap::new(),
        }
    }

    /// Add an order to track.
    pub fn add_order(&mut self, order: TrackedOrder) {
        self.orders.insert(order.oid, order);
    }

    /// Remove an order by ID.
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

    /// Get the first resting order on a given side (for single-order-per-side strategies).
    /// Only returns orders in Resting state (not Cancelling).
    pub fn get_by_side(&self, side: Side) -> Option<&TrackedOrder> {
        self.orders
            .values()
            .find(|o| o.side == side && !o.is_filled() && o.state == OrderState::Resting)
    }

    /// Get all resting orders on a given side.
    /// Only returns orders in Resting state (not Cancelling).
    pub fn get_all_by_side(&self, side: Side) -> Vec<&TrackedOrder> {
        self.orders
            .values()
            .filter(|o| o.side == side && !o.is_filled() && o.state == OrderState::Resting)
            .collect()
    }

    /// Set the state of an order.
    /// Returns true if the order was found and updated.
    pub fn set_state(&mut self, oid: u64, state: OrderState) -> bool {
        if let Some(order) = self.orders.get_mut(&oid) {
            order.state = state;
            true
        } else {
            false
        }
    }

    /// Update an order with a fill amount.
    /// Returns `true` if the order was found and updated.
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

    /// Remove fully filled orders.
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
    fn test_update_fill() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));

        assert!(mgr.update_fill(1, 0.5));
        let order = mgr.get_order(1).unwrap();
        assert!((order.filled - 0.5).abs() < f64::EPSILON);
        assert!((order.remaining() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_filled() {
        let mut order = TrackedOrder::new(1, Side::Buy, 100.0, 1.0);
        assert!(!order.is_filled());

        order.filled = 1.0;
        assert!(order.is_filled());
    }
}
