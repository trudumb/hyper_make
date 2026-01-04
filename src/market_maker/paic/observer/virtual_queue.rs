//! Virtual Queue Tracker - Priority Index (π) Estimation.
//!
//! Since Hyperliquid doesn't expose exact queue position via API, we estimate it
//! by tracking cumulative volume traded at each price level since order placement.
//!
//! # Queue Position Model
//!
//! ```text
//! π_t = cumulative_volume_traded_at_level / (initial_depth_ahead + our_size)
//! ```
//!
//! Where:
//! - `cumulative_volume_traded_at_level`: Volume executed at our price since placement
//! - `initial_depth_ahead`: Depth ahead of us when we placed (from L2 snapshot)
//! - `our_size`: Our order size
//!
//! π ranges from 0.0 (front of queue) to 1.0 (back of queue).
//!
//! # Priority Classification
//!
//! - High Priority (π < 0.3): We're near the front, our order is valuable
//! - Medium Priority (0.3 ≤ π < 0.7): Middle of queue
//! - Low Priority (π ≥ 0.7): Near the back, can freely update
//!
//! # Volume Decay
//!
//! Volume observations decay exponentially to handle stale data:
//! ```text
//! effective_volume(t) = volume × e^(-λ × Δt)
//! ```

use std::collections::HashMap;
use std::time::Instant;

use super::super::config::VirtualQueueConfig;

/// Priority classification for an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorityClass {
    /// Near front of queue (π < 0.3)
    High,
    /// Middle of queue (0.3 ≤ π < 0.7)
    Medium,
    /// Near back of queue (π ≥ 0.7)
    Low,
}

impl PriorityClass {
    /// Create from priority index.
    pub fn from_pi(pi: f64) -> Self {
        if pi < 0.3 {
            Self::High
        } else if pi < 0.7 {
            Self::Medium
        } else {
            Self::Low
        }
    }

    /// Check if this is high priority.
    pub fn is_high(&self) -> bool {
        matches!(self, Self::High)
    }

    /// Get priority premium multiplier.
    ///
    /// High priority orders have more value (premium closer to 1.0).
    pub fn priority_premium(&self) -> f64 {
        match self {
            Self::High => 0.8,   // 80% of spread as premium
            Self::Medium => 0.4, // 40% of spread
            Self::Low => 0.1,    // 10% of spread
        }
    }
}

/// Per-order priority tracking state.
#[derive(Debug, Clone)]
pub struct OrderPriority {
    /// Order ID
    pub oid: u64,
    /// Price level
    pub price: f64,
    /// Our order size
    pub size: f64,
    /// Is this a bid order?
    pub is_bid: bool,
    /// Initial depth ahead when placed (from L2 snapshot)
    pub initial_depth_ahead: f64,
    /// Cumulative volume traded at this level since placement
    pub cumulative_volume_traded: f64,
    /// Time of order placement
    pub placed_at: Instant,
    /// Last update time
    pub last_update: Instant,
    /// Smoothed priority index [0, 1]
    pub pi: f64,
    /// Priority class
    pub priority_class: PriorityClass,
}

impl OrderPriority {
    /// Create new priority tracker for an order.
    pub fn new(oid: u64, price: f64, size: f64, depth_ahead: f64, is_bid: bool) -> Self {
        let now = Instant::now();
        // Initial pi is 1.0 (back of queue) since no volume has traded yet
        let pi = 1.0;
        Self {
            oid,
            price,
            size,
            is_bid,
            initial_depth_ahead: depth_ahead,
            cumulative_volume_traded: 0.0,
            placed_at: now,
            last_update: now,
            pi,
            priority_class: PriorityClass::from_pi(pi),
        }
    }

    /// Update with volume traded at this level.
    ///
    /// Returns the new priority index.
    pub fn record_volume(&mut self, volume: f64, ewma_alpha: f64) -> f64 {
        self.cumulative_volume_traded += volume;
        self.last_update = Instant::now();

        // Calculate raw priority index
        let total_queue = self.initial_depth_ahead + self.size;
        if total_queue <= 0.0 {
            return self.pi;
        }

        // Raw π = fraction of queue that has traded
        // If cumulative_volume > initial_depth, we should be near front
        let raw_pi = if self.cumulative_volume_traded >= self.initial_depth_ahead {
            // We've advanced to the front
            0.0
        } else {
            // Remaining depth ahead / total queue
            let remaining_ahead = self.initial_depth_ahead - self.cumulative_volume_traded;
            (remaining_ahead / total_queue).clamp(0.0, 1.0)
        };

        // EWMA smoothing to prevent jitter
        self.pi = ewma_alpha * raw_pi + (1.0 - ewma_alpha) * self.pi;
        self.priority_class = PriorityClass::from_pi(self.pi);

        self.pi
    }

    /// Apply decay to cumulative volume (for stale orders).
    pub fn apply_decay(&mut self, decay_factor: f64) {
        self.cumulative_volume_traded *= decay_factor;
        // Recalculate pi after decay
        let total_queue = self.initial_depth_ahead + self.size;
        if total_queue > 0.0 {
            let remaining_ahead = (self.initial_depth_ahead - self.cumulative_volume_traded).max(0.0);
            self.pi = (remaining_ahead / total_queue).clamp(0.0, 1.0);
            self.priority_class = PriorityClass::from_pi(self.pi);
        }
    }

    /// Get age in seconds.
    pub fn age_secs(&self) -> f64 {
        self.placed_at.elapsed().as_secs_f64()
    }

    /// Get time since last update in seconds.
    pub fn time_since_update_secs(&self) -> f64 {
        self.last_update.elapsed().as_secs_f64()
    }

    /// Estimate the "option value" of this queue position.
    ///
    /// High priority orders have more value because they're more likely
    /// to fill passively before price moves away.
    ///
    /// Value = (1 - π) × spread × size
    pub fn option_value(&self, spread: f64) -> f64 {
        (1.0 - self.pi) * spread * self.size
    }
}

/// Virtual Queue Tracker - manages priority tracking for all orders.
///
/// This tracker estimates queue position (π) for each order by observing
/// trade volume at each price level.
#[derive(Debug)]
pub struct VirtualQueueTracker {
    /// Configuration
    config: VirtualQueueConfig,

    /// Priority state per order ID
    orders: HashMap<u64, OrderPriority>,

    /// Volume per price level in current window: (price_key, volume, last_update)
    /// Price key is price * 1e8 as i64 for hash stability
    level_volumes: HashMap<i64, (f64, Instant)>,

    /// Decay rate (derived from half-life)
    decay_rate: f64,

    /// Current best bid/ask for reference
    best_bid: Option<f64>,
    best_ask: Option<f64>,
}

impl VirtualQueueTracker {
    /// Create a new virtual queue tracker.
    pub fn new(config: VirtualQueueConfig) -> Self {
        let decay_rate = 2.0_f64.ln() / config.volume_decay_half_life_secs;
        Self {
            config,
            orders: HashMap::with_capacity(32),
            level_volumes: HashMap::with_capacity(64),
            decay_rate,
            best_bid: None,
            best_ask: None,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(VirtualQueueConfig::default())
    }

    /// Convert price to hash key.
    #[inline]
    fn price_to_key(price: f64) -> i64 {
        (price * 1e8) as i64
    }

    /// Register a new order for priority tracking.
    pub fn order_placed(
        &mut self,
        oid: u64,
        price: f64,
        size: f64,
        depth_ahead: f64,
        is_bid: bool,
    ) {
        let depth = depth_ahead.max(self.config.min_queue_position);
        let priority = OrderPriority::new(oid, price, size, depth, is_bid);
        self.orders.insert(oid, priority);
    }

    /// Remove an order from tracking.
    pub fn order_removed(&mut self, oid: u64) {
        self.orders.remove(&oid);
    }

    /// Update order after partial fill.
    pub fn order_partially_filled(&mut self, oid: u64, filled_amount: f64) {
        if let Some(order) = self.orders.get_mut(&oid) {
            order.size = (order.size - filled_amount).max(0.0);
            // Partial fill means we advanced in queue
            order.cumulative_volume_traded += filled_amount;
            order.record_volume(0.0, self.config.priority_ewma_alpha);
        }
    }

    /// Update with best bid/ask.
    pub fn update_bbo(&mut self, best_bid: f64, best_ask: f64) {
        self.best_bid = Some(best_bid);
        self.best_ask = Some(best_ask);
    }

    /// Process a trade at a price level.
    ///
    /// This updates the cumulative volume for all orders at this level.
    pub fn on_trade(&mut self, price: f64, volume: f64, is_buy: bool) {
        let now = Instant::now();
        let price_key = Self::price_to_key(price);

        // Update level volume
        self.level_volumes
            .entry(price_key)
            .and_modify(|(v, t)| {
                *v += volume;
                *t = now;
            })
            .or_insert((volume, now));

        // Update all orders at this price level
        let alpha = self.config.priority_ewma_alpha;
        for order in self.orders.values_mut() {
            // Check if trade is at our order's level
            let at_level = (order.price - price).abs() < 1e-10;
            if !at_level {
                continue;
            }

            // Trade must be on same side (buys execute bids, sells execute asks)
            let same_side = (is_buy && order.is_bid) || (!is_buy && !order.is_bid);
            if !same_side {
                continue;
            }

            // Update priority
            order.record_volume(volume, alpha);
        }
    }

    /// Apply decay to all orders based on time elapsed.
    ///
    /// Call this periodically (e.g., every second) to handle stale orders.
    pub fn apply_decay(&mut self) {
        let now = Instant::now();

        for order in self.orders.values_mut() {
            let dt = order.time_since_update_secs();
            if dt > 0.1 {
                // Only decay if no recent updates
                let decay_factor = (-self.decay_rate * dt).exp();
                order.apply_decay(decay_factor);
            }
        }

        // Clean up old level volumes
        let window_secs = self.config.volume_window_ms as f64 / 1000.0;
        self.level_volumes.retain(|_, (_, last)| {
            now.duration_since(*last).as_secs_f64() < window_secs
        });
    }

    /// Get priority index for an order.
    pub fn get_priority(&self, oid: u64) -> Option<f64> {
        self.orders.get(&oid).map(|o| o.pi)
    }

    /// Get priority class for an order.
    pub fn get_priority_class(&self, oid: u64) -> Option<PriorityClass> {
        self.orders.get(&oid).map(|o| o.priority_class)
    }

    /// Get full priority state for an order.
    pub fn get_order_priority(&self, oid: u64) -> Option<&OrderPriority> {
        self.orders.get(&oid)
    }

    /// Get option value for an order.
    pub fn get_option_value(&self, oid: u64, spread: f64) -> Option<f64> {
        self.orders.get(&oid).map(|o| o.option_value(spread))
    }

    /// Calculate the "priority premium" - the value of keeping this queue position.
    ///
    /// ```text
    /// priority_premium = (1 - π) × spread × multiplier
    /// ```
    ///
    /// This represents the expected value we'd lose by canceling and re-placing.
    pub fn priority_premium(&self, oid: u64, spread: f64, multiplier: f64) -> Option<f64> {
        self.orders.get(&oid).map(|order| {
            let priority_value = 1.0 - order.pi;
            priority_value * spread * multiplier
        })
    }

    /// Determine if an order should hold its position (high priority + small drift).
    ///
    /// Returns true if the order is high priority and the drift doesn't exceed
    /// the priority premium.
    pub fn should_hold(&self, oid: u64, drift_bps: f64, spread_bps: f64) -> bool {
        if let Some(order) = self.orders.get(&oid) {
            if order.priority_class.is_high() {
                // Calculate priority premium
                let premium = (1.0 - order.pi) * spread_bps * 0.8;
                return drift_bps < premium;
            }
        }
        false
    }

    /// Get summary of all tracked orders.
    pub fn summary(&self) -> VirtualQueueSummary {
        let mut high_priority = Vec::new();
        let mut medium_priority = Vec::new();
        let mut low_priority = Vec::new();

        for order in self.orders.values() {
            let entry = OrderPrioritySummary {
                oid: order.oid,
                price: order.price,
                pi: order.pi,
                class: order.priority_class,
                age_secs: order.age_secs(),
                cumulative_volume: order.cumulative_volume_traded,
            };

            match order.priority_class {
                PriorityClass::High => high_priority.push(entry),
                PriorityClass::Medium => medium_priority.push(entry),
                PriorityClass::Low => low_priority.push(entry),
            }
        }

        VirtualQueueSummary {
            total_orders: self.orders.len(),
            high_priority,
            medium_priority,
            low_priority,
            best_bid: self.best_bid,
            best_ask: self.best_ask,
        }
    }
}

/// Summary of a single order's priority.
#[derive(Debug, Clone)]
pub struct OrderPrioritySummary {
    pub oid: u64,
    pub price: f64,
    pub pi: f64,
    pub class: PriorityClass,
    pub age_secs: f64,
    pub cumulative_volume: f64,
}

/// Summary of virtual queue state.
#[derive(Debug, Clone)]
pub struct VirtualQueueSummary {
    pub total_orders: usize,
    pub high_priority: Vec<OrderPrioritySummary>,
    pub medium_priority: Vec<OrderPrioritySummary>,
    pub low_priority: Vec<OrderPrioritySummary>,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_class_from_pi() {
        assert_eq!(PriorityClass::from_pi(0.0), PriorityClass::High);
        assert_eq!(PriorityClass::from_pi(0.29), PriorityClass::High);
        assert_eq!(PriorityClass::from_pi(0.3), PriorityClass::Medium);
        assert_eq!(PriorityClass::from_pi(0.5), PriorityClass::Medium);
        assert_eq!(PriorityClass::from_pi(0.7), PriorityClass::Low);
        assert_eq!(PriorityClass::from_pi(1.0), PriorityClass::Low);
    }

    #[test]
    fn test_order_priority_new() {
        let order = OrderPriority::new(1, 100.0, 1.0, 10.0, true);
        assert_eq!(order.oid, 1);
        assert_eq!(order.price, 100.0);
        assert_eq!(order.size, 1.0);
        assert_eq!(order.initial_depth_ahead, 10.0);
        assert!(order.is_bid);
        // Initially at back of queue
        assert!((order.pi - 1.0).abs() < 0.01);
        assert_eq!(order.priority_class, PriorityClass::Low);
    }

    #[test]
    fn test_order_priority_record_volume() {
        let mut order = OrderPriority::new(1, 100.0, 1.0, 10.0, true);

        // Use high alpha for instant update in tests
        let alpha = 1.0; // No smoothing

        // After 5 units trade, we should have advanced
        order.record_volume(5.0, alpha);
        assert!(order.pi < 1.0, "Expected pi < 1.0, got {}", order.pi);

        // After 10 units trade (our full depth ahead), we should be at front
        order.record_volume(5.0, alpha);
        assert!(order.pi < 0.1, "Expected pi < 0.1 (at front), got {}", order.pi);
    }

    #[test]
    fn test_virtual_queue_tracker_basic() {
        // Use config with fast priority updates
        let config = VirtualQueueConfig {
            priority_ewma_alpha: 1.0, // No smoothing for tests
            ..Default::default()
        };
        let mut tracker = VirtualQueueTracker::new(config);

        // Place an order
        tracker.order_placed(1, 100.0, 1.0, 10.0, true);
        assert!(tracker.get_priority(1).is_some());
        assert_eq!(tracker.get_priority_class(1), Some(PriorityClass::Low));

        // Simulate trades at the level
        tracker.on_trade(100.0, 5.0, true);
        let pi = tracker.get_priority(1).unwrap();
        assert!(pi < 1.0, "Expected pi < 1.0 after 5 units, got {}", pi);

        // More trades (total 10 = initial_depth_ahead)
        tracker.on_trade(100.0, 5.0, true);
        let pi = tracker.get_priority(1).unwrap();
        assert!(pi < 0.1, "Expected pi < 0.1 at front, got {}", pi);

        // Remove order
        tracker.order_removed(1);
        assert!(tracker.get_priority(1).is_none());
    }

    #[test]
    fn test_should_hold() {
        // Use config with fast priority updates
        let config = VirtualQueueConfig {
            priority_ewma_alpha: 1.0, // No smoothing for tests
            ..Default::default()
        };
        let mut tracker = VirtualQueueTracker::new(config);

        // Place order and advance to front
        tracker.order_placed(1, 100.0, 1.0, 10.0, true);
        tracker.on_trade(100.0, 10.0, true); // At front

        // At front, priority premium = (1-0) * 10 * 0.8 = 8 bps
        // Should hold for small drift (2 < 8)
        assert!(tracker.should_hold(1, 2.0, 10.0), "Should hold: 2 bps drift < 8 bps premium");

        // Should not hold for large drift (15 > 8)
        assert!(!tracker.should_hold(1, 15.0, 10.0), "Should not hold: 15 bps drift > 8 bps premium");
    }

    #[test]
    fn test_option_value() {
        // Use config with fast priority updates
        let config = VirtualQueueConfig {
            priority_ewma_alpha: 1.0, // No smoothing for tests
            ..Default::default()
        };
        let mut tracker = VirtualQueueTracker::new(config);

        // Order at back of queue (pi = 1.0)
        tracker.order_placed(1, 100.0, 1.0, 10.0, true);
        let value_back = tracker.get_option_value(1, 0.001).unwrap();

        // Advance to front
        tracker.on_trade(100.0, 10.0, true);
        let value_front = tracker.get_option_value(1, 0.001).unwrap();

        // Front of queue should have higher option value
        assert!(value_front > value_back, "Expected front value {} > back value {}", value_front, value_back);
    }

    #[test]
    fn test_summary() {
        // Use config with fast priority updates
        let config = VirtualQueueConfig {
            priority_ewma_alpha: 1.0, // No smoothing for tests
            ..Default::default()
        };
        let mut tracker = VirtualQueueTracker::new(config);

        // Place orders at different priorities
        tracker.order_placed(1, 100.0, 1.0, 10.0, true);
        tracker.order_placed(2, 101.0, 1.0, 10.0, false);

        // Advance order 1 to high priority
        tracker.on_trade(100.0, 10.0, true);

        let summary = tracker.summary();
        assert_eq!(summary.total_orders, 2);
        // Order 1 should now be high priority (pi ≈ 0)
        assert!(!summary.high_priority.is_empty(), "Expected order 1 in high priority, but high_priority is empty");
        // Order 2 should still be low priority
        assert!(!summary.low_priority.is_empty(), "Expected order 2 in low priority, but low_priority is empty");
    }
}
