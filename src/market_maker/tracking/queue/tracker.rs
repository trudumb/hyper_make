//! Queue position tracker implementation.

use std::collections::HashMap;
use std::time::Instant;

use tracing::debug;

use super::config::QueueConfig;
use super::normal_cdf;
use super::position::{OrderQueuePosition, QueuePositionSummary, QueueSummary};
use crate::market_maker::infra::capacity::QUEUE_TRACKER_CAPACITY;

/// Tracks queue positions for all active orders.
#[derive(Debug)]
pub struct QueuePositionTracker {
    config: QueueConfig,

    /// Queue positions by order ID
    positions: HashMap<u64, OrderQueuePosition>,

    /// Cached best bid/ask for quick reference
    best_bid: Option<f64>,
    best_ask: Option<f64>,

    /// Current volatility (sigma per second) for P(touch) calculation
    sigma: f64,

    /// Cached L2 bid levels (price, size) sorted best-to-worst for depth estimation.
    /// Updated on each L2 book snapshot.
    cached_l2_bids: Vec<(f64, f64)>,

    /// Cached L2 ask levels (price, size) sorted best-to-worst for depth estimation.
    cached_l2_asks: Vec<(f64, f64)>,
}

impl QueuePositionTracker {
    /// Create a new queue position tracker.
    ///
    /// Pre-allocates HashMap to avoid heap reallocations in hot paths.
    pub fn new(config: QueueConfig) -> Self {
        Self {
            config,
            positions: HashMap::with_capacity(QUEUE_TRACKER_CAPACITY),
            best_bid: None,
            best_ask: None,
            sigma: 0.0001, // Default sigma
            cached_l2_bids: Vec::new(),
            cached_l2_asks: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(QueueConfig::default())
    }

    /// Register a new order with initial queue position.
    ///
    /// `depth_ahead` should be the total size at this price level
    /// that was already there when we placed our order.
    pub fn order_placed(
        &mut self,
        oid: u64,
        price: f64,
        size: f64,
        depth_ahead: f64,
        is_bid: bool,
    ) {
        let depth = depth_ahead.max(self.config.min_queue_position);
        let position = OrderQueuePosition::new(oid, price, size, depth, is_bid);

        debug!(
            oid = oid,
            price = price,
            size = size,
            depth_ahead = depth,
            is_bid = is_bid,
            "Queue: Order placed"
        );

        self.positions.insert(oid, position);
    }

    /// Remove an order from tracking (cancelled or filled).
    pub fn order_removed(&mut self, oid: u64) {
        if self.positions.remove(&oid).is_some() {
            debug!(oid = oid, "Queue: Order removed");
        }
    }

    /// Update order after a partial fill - reduce tracked size.
    ///
    /// This is called when an order is partially filled but still has remaining size.
    /// We reduce our tracked size and assume we moved up in the queue.
    pub fn order_partially_filled(&mut self, oid: u64, filled_amount: f64) {
        if let Some(position) = self.positions.get_mut(&oid) {
            position.size = (position.size - filled_amount).max(0.0);
            // Assume we moved up in the queue by the filled amount
            position.depth_ahead =
                (position.depth_ahead - filled_amount).max(self.config.min_queue_position);
            debug!(
                oid = oid,
                new_size = %format!("{:.6}", position.size),
                new_depth = %format!("{:.6}", position.depth_ahead),
                "Queue: Partial fill processed"
            );
        }
    }

    /// Get the current volatility (sigma per second).
    #[inline]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Update tracker with new L2 book data.
    ///
    /// This decays queue positions and updates best bid/ask.
    pub fn update_from_book(&mut self, best_bid: f64, best_ask: f64, sigma: f64) {
        self.best_bid = Some(best_bid);
        self.best_ask = Some(best_ask);
        self.sigma = sigma.max(1e-10);

        let now = Instant::now();
        let decay_rate = self.config.cancel_decay_rate;

        for position in self.positions.values_mut() {
            let dt = position.time_since_update();

            // Decay queue position (exponential decay from cancellations)
            // depth_ahead(t) = depth_ahead(0) × e^(-decay_rate × t)
            let decay_factor = (-decay_rate * dt).exp();
            position.depth_ahead =
                (position.depth_ahead * decay_factor).max(self.config.min_queue_position);

            position.last_update = now;
        }
    }

    /// Update depth estimates from full L2 book snapshot.
    ///
    /// For each tracked order, recalculates depth from the current L2 state.
    /// Uses `min(current_depth, book_depth)` to preserve queue decay —
    /// the book can only tell us an upper bound on our queue position, since
    /// we may have advanced through fills and cancellations ahead of us.
    ///
    /// Also caches the L2 levels for use in `estimate_depth_at_price`.
    pub fn update_depth_from_book(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        // Cache the L2 levels for depth-at-placement estimation
        self.cached_l2_bids.clear();
        self.cached_l2_bids.extend_from_slice(bids);
        self.cached_l2_asks.clear();
        self.cached_l2_asks.extend_from_slice(asks);

        if self.positions.is_empty() {
            return;
        }

        let min_depth = self.config.min_queue_position;

        for position in self.positions.values_mut() {
            let book_depth = if position.is_bid {
                // For bids: depth = sum of sizes at prices >= our price (ahead in queue)
                Self::compute_depth_from_levels(bids, position.price, true)
            } else {
                // For asks: depth = sum of sizes at prices <= our price (ahead in queue)
                Self::compute_depth_from_levels(asks, position.price, false)
            };

            // Use min(current, book) to preserve queue advancement from decay/trades.
            // The book gives us an upper bound; our tracked depth may be lower
            // because we've observed fills and cancellations draining the queue.
            position.depth_ahead = position.depth_ahead.min(book_depth).max(min_depth);
        }
    }

    /// Update queue positions when a market trade executes.
    ///
    /// Decrements `depth_ahead` for orders at the same price on the same side.
    /// This is a convenience wrapper over `trades_at_level`.
    pub fn on_market_trade(&mut self, price: f64, size: f64, is_buy: bool) {
        // A buy trade lifts asks (ask-side orders get filled).
        // A sell trade hits bids (bid-side orders get filled).
        // So the *opposite* side's queue drains.
        let is_bid_side = !is_buy;
        self.trades_at_level(price, size, is_bid_side);
    }

    /// Estimate the depth ahead of a hypothetical order at `price`.
    ///
    /// Uses the cached L2 book levels from the most recent `update_depth_from_book` call.
    /// Returns the configured `default_queue_position` if no L2 data is cached.
    pub fn estimate_depth_at_price(&self, price: f64, is_bid: bool) -> f64 {
        let levels = if is_bid {
            &self.cached_l2_bids
        } else {
            &self.cached_l2_asks
        };

        if levels.is_empty() {
            return self.config.default_queue_position;
        }

        Self::compute_depth_from_levels(levels, price, is_bid).max(self.config.min_queue_position)
    }

    /// Compute depth ahead from L2 levels for a given price and side.
    ///
    /// For bids: sums sizes at prices strictly greater than `price` (those are ahead in queue).
    /// For asks: sums sizes at prices strictly less than `price` (those are ahead in queue).
    /// Orders at the *same* price are partially ahead (we assume we're at the back of our level).
    fn compute_depth_from_levels(levels: &[(f64, f64)], price: f64, is_bid: bool) -> f64 {
        let mut depth = 0.0;
        for &(level_price, level_size) in levels {
            if is_bid {
                // Bids sorted high-to-low. Prices > ours are better bids (ahead of us).
                // At our price, that size is also ahead (we join at the back).
                if level_price > price + 1e-10 {
                    depth += level_size;
                } else if (level_price - price).abs() < 1e-10 {
                    // At our price level: all existing size is ahead of us
                    depth += level_size;
                }
            } else {
                // Asks sorted low-to-high. Prices < ours are better asks (ahead of us).
                // At our price, that size is also ahead.
                // Asks sorted low-to-high: prices below ours or at our level are ahead.
                if level_price < price + 1e-10 {
                    depth += level_size;
                }
            }
        }
        depth
    }

    /// Update queue positions when trades execute at a price level.
    ///
    /// `executed_volume` is the total volume that executed at this level.
    /// This reduces depth_ahead for orders at or better than this price.
    pub fn trades_at_level(&mut self, price: f64, executed_volume: f64, is_bid_side: bool) {
        for position in self.positions.values_mut() {
            // Only affects orders on the same side at this price
            if position.is_bid != is_bid_side {
                continue;
            }

            let at_level = (position.price - price).abs() < 1e-10;
            if at_level {
                // Volume executed ahead of us in queue
                position.depth_ahead =
                    (position.depth_ahead - executed_volume).max(self.config.min_queue_position);
            }
        }
    }

    /// Get queue position for an order.
    pub fn get_position(&self, oid: u64) -> Option<&OrderQueuePosition> {
        self.positions.get(&oid)
    }

    /// Calculate probability that price will touch our order's level.
    ///
    /// Uses reflection principle: P(touch δ in time T) = 2Φ(-δ/(σ√T))
    /// where Φ is the standard normal CDF.
    pub fn probability_touch(&self, oid: u64, horizon_seconds: f64) -> Option<f64> {
        let position = self.positions.get(&oid)?;

        // Distance from current best to our order
        let delta = if position.is_bid {
            // For bids, we need price to come DOWN to us
            self.best_bid? - position.price
        } else {
            // For asks, we need price to come UP to us
            position.price - self.best_ask?
        };

        if delta <= 0.0 {
            // We're at or better than best, definitely will be touched
            return Some(1.0);
        }

        // Reflection principle: P(touch) = 2Φ(-δ/(σ√T))
        let sigma_sqrt_t = self.sigma * horizon_seconds.sqrt();
        let z = -delta / sigma_sqrt_t;
        let p_touch = 2.0 * normal_cdf(z);

        Some(p_touch.min(1.0))
    }

    /// Calculate probability of execution given we're touched.
    ///
    /// P(execute|touch) = exp(-queue_position / expected_volume)
    pub fn probability_execute_given_touch(&self, oid: u64, horizon_seconds: f64) -> Option<f64> {
        let position = self.positions.get(&oid)?;

        // Expected volume at touch over horizon
        let expected_volume = self.config.expected_volume_per_second * horizon_seconds;

        if expected_volume <= 0.0 {
            return Some(0.0);
        }

        // P(execute|touch) = exp(-depth_ahead / expected_volume)
        let p_exec = (-position.depth_ahead / expected_volume).exp();

        Some(p_exec.min(1.0))
    }

    /// Calculate total fill probability.
    ///
    /// P(fill) = P(touch) × P(execute|touch)
    pub fn fill_probability(&self, oid: u64, horizon_seconds: f64) -> Option<f64> {
        let p_touch = self.probability_touch(oid, horizon_seconds)?;
        let p_exec = self.probability_execute_given_touch(oid, horizon_seconds)?;

        Some(p_touch * p_exec)
    }

    /// Check if an order should be refreshed (replaced with new quote).
    ///
    /// Uses EV-based comparison: only refresh if EV improvement > cost.
    ///
    /// Returns true if:
    /// - Order is old enough (beyond min_order_age_for_refresh)
    /// - Fill probability is below refresh_threshold OR
    /// - EV of refreshed position > EV of current position + refresh_cost_bps
    pub fn should_refresh(&self, oid: u64, horizon_seconds: f64) -> bool {
        let position = match self.positions.get(&oid) {
            Some(p) => p,
            None => return false,
        };

        // Check minimum age - protect young orders
        if position.age_seconds() < self.config.min_order_age_for_refresh {
            return false;
        }

        // Calculate current fill probability
        let p_fill_current = self.fill_probability(oid, horizon_seconds).unwrap_or(0.0);

        // Fast path: if fill probability is very low, refresh is likely better
        if p_fill_current < self.config.refresh_threshold {
            return true;
        }

        // EV-based comparison for borderline cases
        // Current EV = P(fill) * spread_capture
        let ev_current = p_fill_current * self.config.spread_capture_bps;

        // Estimate refreshed position EV
        // When we refresh, we go to back of queue - estimate P(fill) for fresh order
        let p_touch = self.probability_touch(oid, horizon_seconds).unwrap_or(0.0);
        let expected_volume = self.config.expected_volume_per_second * horizon_seconds;
        let estimated_depth = self.config.expected_volume_per_second * 0.5; // ~0.5s of volume

        let p_exec_refreshed = if expected_volume > 0.0 {
            (-estimated_depth / expected_volume).exp()
        } else {
            0.0
        };
        let p_fill_refreshed = p_touch * p_exec_refreshed;
        let ev_refreshed = p_fill_refreshed * self.config.spread_capture_bps;

        // Only refresh if improvement exceeds cost
        // This prevents low-value refreshes that burn API budget
        let ev_improvement = ev_refreshed - ev_current;
        ev_improvement > self.config.refresh_cost_bps
    }

    /// Determine if an order should be preserved despite being off-target.
    ///
    /// Compares expected value of keeping the order (with queue advantage)
    /// vs refreshing to the target price (fresh queue).
    ///
    /// Returns (should_preserve, reason) where:
    /// - should_preserve: true if keeping is better than refreshing
    /// - reason: explanation for logging
    ///
    /// # Arguments
    /// * `oid` - Order ID to evaluate
    /// * `target_price` - The optimal price we'd like to be at
    /// * `horizon_seconds` - Time horizon for fill probability
    /// * `spread_capture_bps` - Expected profit if filled (basis points)
    pub fn should_preserve_order(
        &self,
        oid: u64,
        _target_price: f64,
        horizon_seconds: f64,
        spread_capture_bps: f64,
    ) -> (bool, &'static str) {
        let position = match self.positions.get(&oid) {
            Some(p) => p,
            None => return (false, "order_not_tracked"),
        };

        // Calculate current order's expected value
        let p_fill_current = self.fill_probability(oid, horizon_seconds).unwrap_or(0.0);
        let ev_keep = p_fill_current * spread_capture_bps;

        // Estimate queue position at target price (assume back of queue)
        // Use current best bid/ask depth as estimate
        let estimated_depth_at_target = if position.is_bid {
            // For bids: if moving to a better (higher) price, we're at back
            // Estimate based on typical touch depth
            self.config.expected_volume_per_second * 0.5 // ~0.5 second of volume
        } else {
            self.config.expected_volume_per_second * 0.5
        };

        // Calculate P(fill) at new position (fresh queue)
        let p_touch = self.probability_touch(oid, horizon_seconds).unwrap_or(0.0);
        let expected_volume = self.config.expected_volume_per_second * horizon_seconds;
        let p_exec_new = if expected_volume > 0.0 {
            (-estimated_depth_at_target / expected_volume).exp()
        } else {
            0.0
        };
        let p_fill_new = p_touch * p_exec_new;
        let ev_refresh = p_fill_new * spread_capture_bps;

        // Preserve if current EV > refresh EV by meaningful margin (10%)
        if ev_keep > ev_refresh * 1.1 {
            (true, "queue_value_exceeds_refresh")
        } else if p_fill_current > 0.5 {
            // High fill probability - preserve even if EV is similar
            (true, "high_fill_probability")
        } else if position.age_seconds() < 0.5 {
            // Very young order - give it time to build queue value
            (true, "order_too_young")
        } else {
            (false, "refresh_has_better_ev")
        }
    }

    /// Calculate the expected value of keeping an order vs refreshing.
    ///
    /// Returns (keep_value, refresh_value) where:
    /// - keep_value: Expected value of keeping current order
    /// - refresh_value: Expected value of cancelling and re-placing
    ///
    /// The values consider:
    /// - Fill probability
    /// - Expected profit per fill
    /// - Cost of refreshing (losing queue position)
    pub fn refresh_value(
        &self,
        oid: u64,
        horizon_seconds: f64,
        expected_profit_per_fill: f64,
        new_queue_position: f64,
    ) -> Option<(f64, f64)> {
        let _current = self.positions.get(&oid)?;

        // Value of keeping = P(fill|current) × profit
        let p_fill_current = self.fill_probability(oid, horizon_seconds)?;
        let keep_value = p_fill_current * expected_profit_per_fill;

        // Value of refreshing = P(fill|new_queue) × profit
        // Create temporary position for calculation
        let expected_volume = self.config.expected_volume_per_second * horizon_seconds;
        let p_touch = self.probability_touch(oid, horizon_seconds)?;

        let p_exec_new = if expected_volume > 0.0 {
            (-new_queue_position / expected_volume).exp()
        } else {
            0.0
        };

        let p_fill_new = p_touch * p_exec_new;
        let refresh_value = p_fill_new * expected_profit_per_fill;

        Some((keep_value, refresh_value))
    }

    /// Get diagnostic summary.
    pub fn summary(&self) -> QueueSummary {
        let mut bid_positions = Vec::new();
        let mut ask_positions = Vec::new();

        for pos in self.positions.values() {
            let entry = QueuePositionSummary {
                oid: pos.oid,
                price: pos.price,
                depth_ahead: pos.depth_ahead,
                age_seconds: pos.age_seconds(),
            };
            if pos.is_bid {
                bid_positions.push(entry);
            } else {
                ask_positions.push(entry);
            }
        }

        QueueSummary {
            total_orders: self.positions.len(),
            bid_positions,
            ask_positions,
            best_bid: self.best_bid,
            best_ask: self.best_ask,
        }
    }

    // === HJB Queue Value Integration (Phase 2: Churn Reduction) ===

    /// Compute the HJB queue value for an order.
    ///
    /// This uses the queue value formula from the HJB extension:
    /// v(q) = (s/2) × exp(-α×q) - β×q
    ///
    /// # Arguments
    /// * `oid` - Order ID to evaluate
    /// * `half_spread_bps` - Half-spread in basis points
    /// * `alpha` - Queue value decay rate (default 0.1)
    /// * `beta` - Linear cost (default 0.02)
    pub fn order_queue_value(
        &self,
        oid: u64,
        half_spread_bps: f64,
        alpha: f64,
        beta: f64,
    ) -> Option<f64> {
        let position = self.positions.get(&oid)?;

        // Queue depth ahead (in units)
        let q = position.depth_ahead;

        // Queue value formula: (s/2) × exp(-α×q) - β×q
        let exp_term = (half_spread_bps / 2.0) * (-alpha * q).exp();
        let lin_term = beta * q;

        Some((exp_term - lin_term).max(0.0))
    }

    /// Check if an order should be preserved based on HJB queue value.
    ///
    /// # Arguments
    /// * `oid` - Order ID to evaluate
    /// * `half_spread_bps` - Half-spread in basis points
    /// * `alpha` - Queue value decay rate
    /// * `beta` - Linear cost
    /// * `modify_cost_bps` - Threshold cost of modifying (default 3 bps)
    pub fn should_preserve_by_hjb_value(
        &self,
        oid: u64,
        half_spread_bps: f64,
        alpha: f64,
        beta: f64,
        modify_cost_bps: f64,
    ) -> bool {
        self.order_queue_value(oid, half_spread_bps, alpha, beta)
            .map(|v| v >= modify_cost_bps)
            .unwrap_or(false)
    }

    /// Compute total queue value across all orders.
    ///
    /// # Arguments
    /// * `half_spread_bps` - Half-spread in basis points
    /// * `alpha` - Queue value decay rate
    /// * `beta` - Linear cost
    pub fn total_queue_value(&self, half_spread_bps: f64, alpha: f64, beta: f64) -> f64 {
        self.positions
            .keys()
            .filter_map(|&oid| self.order_queue_value(oid, half_spread_bps, alpha, beta))
            .sum()
    }

    /// Get queue depth for an order.
    ///
    /// Returns None if order is not tracked.
    pub fn queue_depth(&self, oid: u64) -> Option<f64> {
        self.positions.get(&oid).map(|p| p.depth_ahead)
    }

    /// Evaluate all orders for queue value preservation.
    ///
    /// Returns a list of (oid, queue_value, should_preserve) tuples.
    pub fn evaluate_all_queue_values(
        &self,
        half_spread_bps: f64,
        alpha: f64,
        beta: f64,
        modify_cost_bps: f64,
    ) -> Vec<(u64, f64, bool)> {
        self.positions
            .keys()
            .filter_map(|&oid| {
                let value = self.order_queue_value(oid, half_spread_bps, alpha, beta)?;
                let preserve = value >= modify_cost_bps;
                Some((oid, value, preserve))
            })
            .collect()
    }
}
