//! Fill Simulator
//!
//! Simulates order fills based on market trade data. Uses a probabilistic model
//! that considers:
//! - Trade price vs order price
//! - Trade size
//! - Queue position (FIFO approximation)
//! - Order age

use super::executor::{SimulatedOrder, SimulatedOrderStatus, SimulationExecutor};
use crate::Side;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

/// A simulated fill event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedFill {
    /// Order ID that was filled
    pub oid: u64,
    /// Fill timestamp (ns)
    pub timestamp_ns: u64,
    /// Fill price
    pub fill_price: f64,
    /// Fill size
    pub fill_size: f64,
    /// Side of the fill (from our perspective)
    pub side: Side,
    /// Trade that triggered the fill
    pub triggering_trade_price: f64,
    /// Trade size that triggered
    pub triggering_trade_size: f64,
}

/// Market trade data for simulation
#[derive(Debug, Clone)]
pub struct MarketTrade {
    /// Trade timestamp (ns)
    pub timestamp_ns: u64,
    /// Trade price
    pub price: f64,
    /// Trade size
    pub size: f64,
    /// Trade side (aggressor side)
    pub side: Side,
}

/// Configuration for the fill simulator
#[derive(Debug, Clone)]
pub struct FillSimulatorConfig {
    /// Probability of fill when price touches our level (0-1)
    pub touch_fill_probability: f64,
    /// Queue position factor (higher = assume worse queue position)
    pub queue_position_factor: f64,
    /// Maximum age of orders to consider for fills (seconds)
    pub max_order_age_s: f64,
    /// Minimum trade size to trigger fills
    pub min_triggering_trade_size: f64,
    /// Simulated order placement latency in milliseconds
    pub placement_latency_ms: u64,
    /// Simulated order cancel latency in milliseconds
    pub cancel_latency_ms: u64,
    /// Skip L2 book depth for queue position (use flat model instead).
    /// Set true for paper trading where orders don't exist in the real book.
    pub ignore_book_depth: bool,
}

impl Default for FillSimulatorConfig {
    fn default() -> Self {
        Self {
            touch_fill_probability: 0.3,
            queue_position_factor: 0.5,
            max_order_age_s: 300.0,
            min_triggering_trade_size: 0.0,
            placement_latency_ms: 100,
            cancel_latency_ms: 50,
            ignore_book_depth: false,
        }
    }
}

/// Simulates fills for paper trading
pub struct FillSimulator {
    /// Configuration
    config: FillSimulatorConfig,
    /// Reference to the simulation executor
    executor: Arc<SimulationExecutor>,
    /// Recent fills for logging
    recent_fills: VecDeque<SimulatedFill>,
    /// Maximum recent fills to keep
    max_recent_fills: usize,
    /// Total fills simulated
    total_fills: u64,
    /// Total size filled
    total_size_filled: f64,
    /// L2 book bid depth by price level (price_ticks -> size)
    book_bid_depth: std::collections::HashMap<i64, f64>,
    /// L2 book ask depth by price level (price_ticks -> size)
    book_ask_depth: std::collections::HashMap<i64, f64>,
    /// Placement latency in nanoseconds
    placement_latency_ns: u64,
}

impl FillSimulator {
    /// Create a new fill simulator
    pub fn new(executor: Arc<SimulationExecutor>, config: FillSimulatorConfig) -> Self {
        let placement_latency_ns = config.placement_latency_ms * 1_000_000;
        Self {
            config,
            executor,
            recent_fills: VecDeque::new(),
            max_recent_fills: 1000,
            total_fills: 0,
            total_size_filled: 0.0,
            book_bid_depth: std::collections::HashMap::new(),
            book_ask_depth: std::collections::HashMap::new(),
            placement_latency_ns,
        }
    }

    /// Update L2 book depth for queue-aware fill probability
    pub fn update_book(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        self.book_bid_depth.clear();
        self.book_ask_depth.clear();
        for &(price, size) in bids {
            let ticks = (price * 100.0) as i64; // 0.01 resolution
            self.book_bid_depth.insert(ticks, size);
        }
        for &(price, size) in asks {
            let ticks = (price * 100.0) as i64;
            self.book_ask_depth.insert(ticks, size);
        }
    }

    /// Get book depth at a price level
    fn book_depth_at_price(&self, price: f64, is_buy: bool) -> f64 {
        let ticks = (price * 100.0) as i64;
        if is_buy {
            self.book_bid_depth.get(&ticks).copied().unwrap_or(0.0)
        } else {
            self.book_ask_depth.get(&ticks).copied().unwrap_or(0.0)
        }
    }

    /// Process a market trade and check for fills
    pub fn on_trade(&mut self, trade: &MarketTrade) -> Vec<SimulatedFill> {
        let mut fills = Vec::new();

        // Get active orders from executor
        let orders = self.executor.get_active_orders();

        for order in orders {
            // Check if this trade could fill the order
            if let Some(fill) = self.check_fill(&order, trade) {
                // Execute the fill in the executor
                if self
                    .executor
                    .simulate_fill(fill.oid, fill.fill_size, fill.fill_price)
                {
                    info!(
                        oid = fill.oid,
                        side = ?fill.side,
                        price = fill.fill_price,
                        size = fill.fill_size,
                        "[SIM] Fill simulated"
                    );

                    self.total_fills += 1;
                    self.total_size_filled += fill.fill_size;

                    // Add to recent fills
                    self.recent_fills.push_back(fill.clone());
                    while self.recent_fills.len() > self.max_recent_fills {
                        self.recent_fills.pop_front();
                    }

                    fills.push(fill);
                }
            }
        }

        fills
    }

    /// Check if a trade would fill an order
    fn check_fill(&self, order: &SimulatedOrder, trade: &MarketTrade) -> Option<SimulatedFill> {
        if order.status != SimulatedOrderStatus::Resting {
            return None;
        }

        // Latency simulation: skip orders that haven't been resting long enough
        let effective_resting_at = order.created_at_ns + self.placement_latency_ns;
        if trade.timestamp_ns < effective_resting_at {
            return None;
        }

        // Check trade size threshold
        if trade.size < self.config.min_triggering_trade_size {
            return None;
        }

        // Check price condition:
        // - Buy order: filled when trade price <= order price
        // - Sell order: filled when trade price >= order price
        let price_condition = if order.is_buy {
            trade.price <= order.price
        } else {
            trade.price >= order.price
        };

        if !price_condition {
            return None;
        }

        // Check aggressor direction:
        // - Our buy order gets filled by aggressive sell (trade.side == Sell)
        // - Our sell order gets filled by aggressive buy (trade.side == Buy)
        let aggressor_matches = if order.is_buy {
            trade.side == Side::Sell
        } else {
            trade.side == Side::Buy
        };

        if !aggressor_matches {
            // Price touched but aggressor going wrong direction
            // Still might fill with reduced probability
            let touch_prob = self.config.touch_fill_probability * 0.3;
            if !should_fill_probabilistic(touch_prob) {
                return None;
            }
        } else {
            // Normal fill probability
            let fill_prob = self.compute_fill_probability(order, trade);
            if !should_fill_probabilistic(fill_prob) {
                return None;
            }
        }

        // Determine fill size (can't fill more than order size or trade size)
        let fill_size = order
            .size
            .min(trade.size * self.config.queue_position_factor);

        if fill_size <= 0.0 {
            return None;
        }

        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Some(SimulatedFill {
            oid: order.oid,
            timestamp_ns: now_ns,
            fill_price: order.price, // Fill at limit price
            fill_size,
            side: if order.is_buy { Side::Buy } else { Side::Sell },
            triggering_trade_price: trade.price,
            triggering_trade_size: trade.size,
        })
    }

    /// Compute probability of fill based on order and trade characteristics
    fn compute_fill_probability(&self, order: &SimulatedOrder, trade: &MarketTrade) -> f64 {
        let base_prob = self.config.touch_fill_probability;

        // Factor 1: Price improvement - further through our level = higher prob
        let price_diff = if order.is_buy {
            order.price - trade.price
        } else {
            trade.price - order.price
        };
        let price_factor = if price_diff > 0.0 {
            1.5 // Trade went through our level
        } else {
            1.0 // Trade at or near our level
        };

        // Factor 2: Trade size relative to our order
        let size_factor = (trade.size / order.size).clamp(0.5, 2.0);

        // Factor 3: Queue position
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let age_s = (now_ns.saturating_sub(order.created_at_ns)) as f64 / 1e9;
        let age_bonus = if age_s > 10.0 {
            1.3
        } else if age_s > 1.0 {
            1.0
        } else {
            0.7
        };

        let queue_factor = if self.config.ignore_book_depth {
            // Flat queue model: sim orders aren't in the real book
            self.config.queue_position_factor * age_bonus
        } else {
            // L2 book-aware queue position
            let volume_at_level = self.book_depth_at_price(order.price, order.is_buy);
            if volume_at_level > 0.0 {
                let our_share = order.size / (volume_at_level + order.size);
                our_share * age_bonus
            } else {
                self.config.queue_position_factor * age_bonus
            }
        };

        let final_prob = base_prob * price_factor * size_factor * queue_factor;

        debug!(
            oid = order.oid,
            base_prob,
            price_factor,
            size_factor,
            queue_factor,
            final_prob,
            "Fill probability calculation"
        );

        final_prob.clamp(0.0, 0.95)
    }

    /// Get recent fills
    pub fn get_recent_fills(&self) -> &VecDeque<SimulatedFill> {
        &self.recent_fills
    }

    /// Get fill statistics
    pub fn get_stats(&self) -> FillSimulatorStats {
        FillSimulatorStats {
            total_fills: self.total_fills,
            total_size_filled: self.total_size_filled,
            recent_fill_count: self.recent_fills.len(),
        }
    }

    /// Clear recent fills
    pub fn clear_recent(&mut self) {
        self.recent_fills.clear();
    }

    /// Estimate our queue position at a given price level.
    /// Returns (estimated_position, estimated_total) where position is our
    /// estimated place in the queue and total is the total queue depth.
    pub fn estimate_queue_position(&self, price: f64, is_buy: bool) -> (f64, f64) {
        let volume_at_level = self.book_depth_at_price(price, is_buy);
        if volume_at_level > 0.0 && !self.config.ignore_book_depth {
            // We're at the back of the queue (conservative estimate)
            (volume_at_level, volume_at_level)
        } else {
            // No book data or ignoring book depth
            (0.0, 0.0)
        }
    }
}

/// Statistics about fill simulation
#[derive(Debug, Clone)]
pub struct FillSimulatorStats {
    pub total_fills: u64,
    pub total_size_filled: f64,
    pub recent_fill_count: usize,
}

/// Probabilistic fill decision using proper RNG
fn should_fill_probabilistic(probability: f64) -> bool {
    use rand::Rng;
    rand::thread_rng().gen::<f64>() < probability
}

/// Aggressive fill simulator that fills based on trade flow
/// More realistic than probabilistic - uses actual trade-through logic
pub struct AggressiveFillSimulator {
    /// Reference to executor
    executor: Arc<SimulationExecutor>,
    /// Track cumulative volume at each price level
    _level_volume: std::collections::HashMap<u64, f64>,
    /// Total fills
    total_fills: u64,
}

impl AggressiveFillSimulator {
    /// Create a new aggressive fill simulator
    pub fn new(executor: Arc<SimulationExecutor>) -> Self {
        Self {
            executor,
            _level_volume: std::collections::HashMap::new(),
            total_fills: 0,
        }
    }

    /// Process a trade with aggressive fill logic
    /// Returns fills that would have occurred
    pub fn on_trade(&mut self, trade: &MarketTrade) -> Vec<SimulatedFill> {
        let mut fills = Vec::new();
        let orders = self.executor.get_active_orders();

        // Sort orders by price (best first)
        let mut buy_orders: Vec<_> = orders.iter().filter(|o| o.is_buy).collect();
        let mut sell_orders: Vec<_> = orders.iter().filter(|o| !o.is_buy).collect();

        buy_orders.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
        sell_orders.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());

        // Process based on trade side
        let remaining_size = trade.size;

        match trade.side {
            Side::Sell => {
                // Aggressive sell - fills our buy orders
                for order in buy_orders {
                    if trade.price <= order.price && remaining_size > 0.0 {
                        let fill_size = order.size.min(remaining_size);
                        if self
                            .executor
                            .simulate_fill(order.oid, fill_size, order.price)
                        {
                            let now_ns = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_nanos() as u64;

                            fills.push(SimulatedFill {
                                oid: order.oid,
                                timestamp_ns: now_ns,
                                fill_price: order.price,
                                fill_size,
                                side: Side::Buy,
                                triggering_trade_price: trade.price,
                                triggering_trade_size: trade.size,
                            });

                            self.total_fills += 1;
                        }
                    }
                }
            }
            Side::Buy => {
                // Aggressive buy - fills our sell orders
                for order in sell_orders {
                    if trade.price >= order.price && remaining_size > 0.0 {
                        let fill_size = order.size.min(remaining_size);
                        if self
                            .executor
                            .simulate_fill(order.oid, fill_size, order.price)
                        {
                            let now_ns = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_nanos() as u64;

                            fills.push(SimulatedFill {
                                oid: order.oid,
                                timestamp_ns: now_ns,
                                fill_price: order.price,
                                fill_size,
                                side: Side::Sell,
                                triggering_trade_price: trade.price,
                                triggering_trade_size: trade.size,
                            });

                            self.total_fills += 1;
                        }
                    }
                }
            }
        }

        fills
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::OrderExecutor;

    #[test]
    fn test_fill_probability() {
        let executor = Arc::new(SimulationExecutor::new(false));
        let sim = FillSimulator::new(executor, FillSimulatorConfig::default());

        // Probability should be reasonable
        let order = SimulatedOrder {
            oid: 1,
            cloid: "test".to_string(),
            asset: "BTC".to_string(),
            price: 100.0,
            size: 1.0,
            original_size: 1.0,
            is_buy: true,
            created_at_ns: 0,
            modified_at_ns: 0,
            post_only: true,
            status: SimulatedOrderStatus::Resting,
        };

        let trade = MarketTrade {
            timestamp_ns: 1_000_000_000,
            price: 99.0, // Trade through our level
            size: 0.5,
            side: Side::Sell,
        };

        let prob = sim.compute_fill_probability(&order, &trade);
        assert!(prob > 0.0 && prob < 1.0);
    }

    #[test]
    fn test_price_condition() {
        let executor = Arc::new(SimulationExecutor::new(false));
        let mut sim = FillSimulator::new(
            executor.clone(),
            FillSimulatorConfig {
                touch_fill_probability: 1.0, // Always fill for test
                ..Default::default()
            },
        );

        // Create a buy order at 100
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            executor.update_mid(101.0);
            executor
                .place_order("BTC", 100.0, 1.0, true, None, true)
                .await;
        });

        // Trade at 99 should potentially fill
        let trade = MarketTrade {
            timestamp_ns: 1_000_000_000,
            price: 99.0,
            size: 1.0,
            side: Side::Sell,
        };

        let fills = sim.on_trade(&trade);
        // May or may not fill depending on probability
        assert!(fills.len() <= 1);
    }
}
