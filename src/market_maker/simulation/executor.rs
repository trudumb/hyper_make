//! Simulation Executor
//!
//! A mock implementation of OrderExecutor that logs all order operations
//! without actually submitting them to the exchange. Used for paper trading
//! and backtesting.

use crate::market_maker::{
    tracking::ws_order_state::WsFillEvent, CancelResult, ModifyResult, ModifySpec, OrderExecutor,
    OrderResult, OrderSpec,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

/// Order state in the simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedOrder {
    /// Order ID
    pub oid: u64,
    /// Client Order ID
    pub cloid: String,
    /// Asset
    pub asset: String,
    /// Price
    pub price: f64,
    /// Remaining size
    pub size: f64,
    /// Original size
    pub original_size: f64,
    /// Is buy order
    pub is_buy: bool,
    /// Creation timestamp (ns)
    pub created_at_ns: u64,
    /// Last modified timestamp (ns)
    pub modified_at_ns: u64,
    /// Is post-only
    pub post_only: bool,
    /// Order status
    pub status: SimulatedOrderStatus,
}

/// Status of a simulated order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulatedOrderStatus {
    /// Order is resting in the book
    Resting,
    /// Order has been cancelled
    Cancelled,
    /// Order has been fully filled
    Filled,
    /// Order was rejected (post-only crossed)
    Rejected,
}

/// Statistics about simulation execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulationStats {
    /// Total orders placed
    pub orders_placed: u64,
    /// Total orders cancelled
    pub orders_cancelled: u64,
    /// Total orders modified
    pub orders_modified: u64,
    /// Total orders rejected (post-only crossed)
    pub orders_rejected: u64,
    /// Total simulated fills
    pub simulated_fills: u64,
    /// Total notional placed (USD)
    pub total_notional_placed: f64,
    /// Total size placed
    pub total_size_placed: f64,
    /// Bid orders placed
    pub bid_orders: u64,
    /// Ask orders placed
    pub ask_orders: u64,
}

/// Simulation executor that logs but doesn't submit orders
pub struct SimulationExecutor {
    /// Next order ID
    next_oid: Arc<RwLock<u64>>,
    /// Active simulated orders by OID
    orders: Arc<RwLock<HashMap<u64, SimulatedOrder>>>,
    /// Order ID lookup by CLOID
    cloid_to_oid: Arc<RwLock<HashMap<String, u64>>>,
    /// Simulation statistics
    stats: Arc<RwLock<SimulationStats>>,
    /// Current market mid price (for post-only validation)
    current_mid: Arc<RwLock<f64>>,
    /// Order log for analysis
    order_log: Arc<RwLock<Vec<OrderLogEntry>>>,
    /// Whether to log verbose details
    verbose: bool,
}

/// Entry in the order log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLogEntry {
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Event type
    pub event: OrderEvent,
    /// Order details
    pub order: Option<SimulatedOrder>,
    /// Additional context
    pub context: String,
}

/// Order event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderEvent {
    Place,
    Cancel,
    Modify,
    Fill,
    Reject,
}

impl SimulationExecutor {
    /// Create a new simulation executor
    pub fn new(verbose: bool) -> Self {
        Self {
            next_oid: Arc::new(RwLock::new(1)),
            orders: Arc::new(RwLock::new(HashMap::new())),
            cloid_to_oid: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SimulationStats::default())),
            current_mid: Arc::new(RwLock::new(0.0)),
            order_log: Arc::new(RwLock::new(Vec::new())),
            verbose,
        }
    }

    /// Update the current market mid price
    pub fn update_mid(&self, mid: f64) {
        *self.current_mid.write().unwrap() = mid;
    }

    /// Get simulation statistics
    pub fn get_stats(&self) -> SimulationStats {
        self.stats.read().unwrap().clone()
    }

    /// Get all active orders
    pub fn get_active_orders(&self) -> Vec<SimulatedOrder> {
        self.orders
            .read()
            .unwrap()
            .values()
            .filter(|o| o.status == SimulatedOrderStatus::Resting)
            .cloned()
            .collect()
    }

    /// Get order log
    pub fn get_order_log(&self) -> Vec<OrderLogEntry> {
        self.order_log.read().unwrap().clone()
    }

    /// Simulate a fill on an order (called by FillSimulator)
    pub fn simulate_fill(&self, oid: u64, fill_size: f64, fill_price: f64) -> bool {
        let mut orders = self.orders.write().unwrap();
        if let Some(order) = orders.get_mut(&oid) {
            if order.status != SimulatedOrderStatus::Resting {
                return false;
            }

            let actual_fill = fill_size.min(order.size);
            order.size -= actual_fill;

            if order.size <= 0.0 {
                order.status = SimulatedOrderStatus::Filled;
            }

            // Log the fill
            self.log_event(
                OrderEvent::Fill,
                Some(order.clone()),
                format!(
                    "Filled {} @ {} (remaining: {})",
                    actual_fill, fill_price, order.size
                ),
            );

            let mut stats = self.stats.write().unwrap();
            stats.simulated_fills += 1;

            true
        } else {
            false
        }
    }

    /// Simulate a fill and return a WsFillEvent for injection into handlers.
    ///
    /// This is the primary method for paper trading integration. It:
    /// 1. Updates the simulated order state
    /// 2. Returns a WsFillEvent that can be passed to MarketMaker fill handlers
    ///
    /// Returns None if the order doesn't exist or isn't resting.
    pub fn apply_fill(&self, oid: u64, fill_size: f64, fill_price: f64) -> Option<WsFillEvent> {
        let mut orders = self.orders.write().unwrap();
        let order = orders.get_mut(&oid)?;

        if order.status != SimulatedOrderStatus::Resting {
            return None;
        }

        let actual_fill = fill_size.min(order.size);
        order.size -= actual_fill;

        if order.size <= 0.0 {
            order.status = SimulatedOrderStatus::Filled;
        }

        // Generate unique trade ID
        let tid = self.next_order_id(); // Reuse OID generator for TID

        // Log the fill
        self.log_event(
            OrderEvent::Fill,
            Some(order.clone()),
            format!(
                "Filled {} @ {} (remaining: {}) tid={}",
                actual_fill, fill_price, order.size, tid
            ),
        );

        let mut stats = self.stats.write().unwrap();
        stats.simulated_fills += 1;

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Some(WsFillEvent {
            oid,
            tid,
            size: actual_fill,
            price: fill_price,
            is_buy: order.is_buy,
            coin: order.asset.clone(),
            cloid: Some(order.cloid.clone()),
            timestamp: now_ms,
        })
    }

    /// Get an order by OID (for fill simulation logic)
    pub fn get_order(&self, oid: u64) -> Option<SimulatedOrder> {
        self.orders.read().unwrap().get(&oid).cloned()
    }

    /// Check if a post-only order would cross the spread
    fn would_cross_spread(&self, price: f64, is_buy: bool) -> bool {
        let mid = *self.current_mid.read().unwrap();
        if mid <= 0.0 {
            return false; // No mid price set, allow order
        }

        if is_buy {
            price >= mid // Buy above mid would cross
        } else {
            price <= mid // Sell below mid would cross
        }
    }

    /// Generate next order ID
    fn next_order_id(&self) -> u64 {
        let mut oid = self.next_oid.write().unwrap();
        let id = *oid;
        *oid += 1;
        id
    }

    /// Log an order event
    fn log_event(&self, event: OrderEvent, order: Option<SimulatedOrder>, context: String) {
        let entry = OrderLogEntry {
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            event,
            order,
            context,
        };

        if self.verbose {
            debug!(?entry, "Simulation order event");
        }

        self.order_log.write().unwrap().push(entry);
    }

    /// Get current timestamp in nanoseconds
    fn now_ns(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

#[async_trait]
impl OrderExecutor for SimulationExecutor {
    async fn place_order(
        &self,
        asset: &str,
        price: f64,
        size: f64,
        is_buy: bool,
        cloid: Option<String>,
        post_only: bool,
    ) -> OrderResult {
        let cloid_str = cloid.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // Check if post-only would cross
        if post_only && self.would_cross_spread(price, is_buy) {
            let mut stats = self.stats.write().unwrap();
            stats.orders_rejected += 1;

            self.log_event(
                OrderEvent::Reject,
                None,
                format!(
                    "Post-only {} order @ {} would cross spread",
                    if is_buy { "BUY" } else { "SELL" },
                    price
                ),
            );

            return OrderResult {
                oid: 0,
                resting_size: 0.0,
                filled: false,
                cloid: Some(cloid_str),
                error: Some("Post-only order would cross spread".to_string()),
            };
        }

        let oid = self.next_order_id();
        let now = self.now_ns();

        let order = SimulatedOrder {
            oid,
            cloid: cloid_str.clone(),
            asset: asset.to_string(),
            price,
            size,
            original_size: size,
            is_buy,
            created_at_ns: now,
            modified_at_ns: now,
            post_only,
            status: SimulatedOrderStatus::Resting,
        };

        // Store the order
        {
            let mut orders = self.orders.write().unwrap();
            orders.insert(oid, order.clone());
        }
        {
            let mut cloid_map = self.cloid_to_oid.write().unwrap();
            cloid_map.insert(cloid_str.clone(), oid);
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.orders_placed += 1;
            stats.total_notional_placed += price * size;
            stats.total_size_placed += size;
            if is_buy {
                stats.bid_orders += 1;
            } else {
                stats.ask_orders += 1;
            }
        }

        self.log_event(
            OrderEvent::Place,
            Some(order),
            format!(
                "{} {} {} @ {}",
                if is_buy { "BUY" } else { "SELL" },
                size,
                asset,
                price
            ),
        );

        info!(
            oid,
            asset,
            side = if is_buy { "BUY" } else { "SELL" },
            price,
            size,
            "[SIM] Order placed"
        );

        OrderResult {
            oid,
            resting_size: size,
            filled: false,
            cloid: Some(cloid_str),
            error: None,
        }
    }

    async fn place_bulk_orders(&self, asset: &str, orders: Vec<OrderSpec>) -> Vec<OrderResult> {
        let mut results = Vec::with_capacity(orders.len());

        for spec in orders {
            let result = self
                .place_order(
                    asset,
                    spec.price,
                    spec.size,
                    spec.is_buy,
                    spec.cloid,
                    spec.post_only,
                )
                .await;
            results.push(result);
        }

        results
    }

    async fn place_ioc_reduce_order(
        &self,
        asset: &str,
        size: f64,
        is_buy: bool,
        slippage_bps: u32,
        mid_price: f64,
    ) -> OrderResult {
        // IOC orders in simulation immediately "fill" or fail
        let slippage_mult = 1.0 + (slippage_bps as f64 / 10000.0);
        let fill_price = if is_buy {
            mid_price * slippage_mult
        } else {
            mid_price / slippage_mult
        };

        info!(
            asset,
            side = if is_buy { "BUY" } else { "SELL" },
            size,
            fill_price,
            "[SIM] IOC reduce order (simulated immediate fill)"
        );

        let mut stats = self.stats.write().unwrap();
        stats.simulated_fills += 1;

        OrderResult {
            oid: self.next_order_id(),
            resting_size: 0.0,
            filled: true,
            cloid: Some(uuid::Uuid::new_v4().to_string()),
            error: None,
        }
    }

    async fn cancel_order(&self, _asset: &str, oid: u64) -> CancelResult {
        let mut orders = self.orders.write().unwrap();

        if let Some(order) = orders.get_mut(&oid) {
            match order.status {
                SimulatedOrderStatus::Resting => {
                    order.status = SimulatedOrderStatus::Cancelled;
                    let mut stats = self.stats.write().unwrap();
                    stats.orders_cancelled += 1;

                    self.log_event(
                        OrderEvent::Cancel,
                        Some(order.clone()),
                        format!("Order {oid} cancelled"),
                    );

                    info!(oid, "[SIM] Order cancelled");
                    CancelResult::Cancelled
                }
                SimulatedOrderStatus::Cancelled => CancelResult::AlreadyCancelled,
                SimulatedOrderStatus::Filled => CancelResult::AlreadyFilled,
                SimulatedOrderStatus::Rejected => CancelResult::AlreadyCancelled,
            }
        } else {
            CancelResult::Failed
        }
    }

    async fn cancel_bulk_orders(&self, asset: &str, oids: Vec<u64>) -> Vec<CancelResult> {
        let mut results = Vec::with_capacity(oids.len());

        for oid in oids {
            let result = self.cancel_order(asset, oid).await;
            results.push(result);
        }

        results
    }

    async fn modify_order(
        &self,
        _asset: &str,
        oid: u64,
        new_price: f64,
        new_size: f64,
        is_buy: bool,
        post_only: bool,
    ) -> ModifyResult {
        // Check if would cross
        if post_only && self.would_cross_spread(new_price, is_buy) {
            return ModifyResult {
                oid: 0,
                resting_size: 0.0,
                success: false,
                error: Some("Modified order would cross spread".to_string()),
            };
        }

        let mut orders = self.orders.write().unwrap();

        if let Some(order) = orders.get_mut(&oid) {
            if order.status != SimulatedOrderStatus::Resting {
                return ModifyResult {
                    oid,
                    resting_size: 0.0,
                    success: false,
                    error: Some("Order not resting".to_string()),
                };
            }

            order.price = new_price;
            order.size = new_size;
            order.modified_at_ns = self.now_ns();

            let mut stats = self.stats.write().unwrap();
            stats.orders_modified += 1;

            self.log_event(
                OrderEvent::Modify,
                Some(order.clone()),
                format!("Order {oid} modified to {new_size} @ {new_price}"),
            );

            info!(oid, new_price, new_size, "[SIM] Order modified");

            ModifyResult {
                oid,
                resting_size: new_size,
                success: true,
                error: None,
            }
        } else {
            ModifyResult {
                oid,
                resting_size: 0.0,
                success: false,
                error: Some("Order not found".to_string()),
            }
        }
    }

    async fn modify_bulk_orders(
        &self,
        asset: &str,
        modifies: Vec<ModifySpec>,
    ) -> Vec<ModifyResult> {
        let mut results = Vec::with_capacity(modifies.len());

        for spec in modifies {
            let result = self
                .modify_order(
                    asset,
                    spec.oid,
                    spec.new_price,
                    spec.new_size,
                    spec.is_buy,
                    spec.post_only,
                )
                .await;
            results.push(result);
        }

        results
    }
}

impl Default for SimulationExecutor {
    fn default() -> Self {
        Self::new(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_place_order() {
        let executor = SimulationExecutor::new(true);
        executor.update_mid(100.0);

        let result = executor
            .place_order("BTC", 99.0, 1.0, true, None, true)
            .await;

        assert!(result.oid > 0);
        assert_eq!(result.resting_size, 1.0);
        assert!(!result.filled);

        let stats = executor.get_stats();
        assert_eq!(stats.orders_placed, 1);
        assert_eq!(stats.bid_orders, 1);
    }

    #[tokio::test]
    async fn test_post_only_rejection() {
        let executor = SimulationExecutor::new(true);
        executor.update_mid(100.0);

        // Buy above mid should be rejected
        let result = executor
            .place_order("BTC", 101.0, 1.0, true, None, true)
            .await;

        assert_eq!(result.oid, 0);
        assert!(result.error.is_some());

        let stats = executor.get_stats();
        assert_eq!(stats.orders_rejected, 1);
    }

    #[tokio::test]
    async fn test_cancel_order() {
        let executor = SimulationExecutor::new(true);
        executor.update_mid(100.0);

        let result = executor
            .place_order("BTC", 99.0, 1.0, true, None, true)
            .await;
        let oid = result.oid;

        let cancel_result = executor.cancel_order("BTC", oid).await;
        assert_eq!(cancel_result, CancelResult::Cancelled);

        // Cancel again should return AlreadyCancelled
        let cancel_result2 = executor.cancel_order("BTC", oid).await;
        assert_eq!(cancel_result2, CancelResult::AlreadyCancelled);
    }

    #[tokio::test]
    async fn test_simulate_fill() {
        let executor = SimulationExecutor::new(true);
        executor.update_mid(100.0);

        let result = executor
            .place_order("BTC", 99.0, 1.0, true, None, true)
            .await;
        let oid = result.oid;

        // Simulate partial fill
        let filled = executor.simulate_fill(oid, 0.5, 99.0);
        assert!(filled);

        let orders = executor.get_active_orders();
        let order = orders.iter().find(|o| o.oid == oid).unwrap();
        assert_eq!(order.size, 0.5);

        // Simulate remaining fill
        let filled = executor.simulate_fill(oid, 0.5, 99.0);
        assert!(filled);

        // Order should no longer be active
        let orders = executor.get_active_orders();
        assert!(orders.iter().find(|o| o.oid == oid).is_none());
    }
}
