//! Queue position tracking for fill probability estimation.
//!
//! Tracks where our orders sit in the order book queue and estimates
//! the probability of getting filled. This enables smarter quote
//! replacement decisions.
//!
//! Key components:
//! - Queue position tracking: Estimate depth ahead of each order
//! - Decay modeling: Queue shrinks from cancels and executions
//! - Fill probability: P(fill) = P(touch) × P(execute|touch)
//! - Refresh decisions: When to replace degraded queue positions

use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for queue position tracking.
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Decay rate for queue position from cancels (fraction per second)
    /// Typical: 0.1-0.3 (10-30% of queue cancels per second)
    pub cancel_decay_rate: f64,

    /// Expected volume at touch per second (in asset units)
    /// Used for execution probability calculation
    pub expected_volume_per_second: f64,

    /// Minimum queue position (floor to prevent division issues)
    pub min_queue_position: f64,

    /// Default queue position when we can't estimate (conservative)
    pub default_queue_position: f64,

    /// Fill probability threshold below which we should consider refreshing
    pub refresh_threshold: f64,

    /// Minimum time before considering a refresh (seconds)
    /// Prevents thrashing on fast markets
    pub min_order_age_for_refresh: f64,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            cancel_decay_rate: 0.2,          // 20% decay per second
            expected_volume_per_second: 1.0, // 1 unit per second at touch
            min_queue_position: 0.01,
            default_queue_position: 10.0,
            refresh_threshold: 0.1,         // Refresh if P(fill) < 10%
            min_order_age_for_refresh: 0.5, // Wait 500ms before refresh
        }
    }
}

// ============================================================================
// Calibrated Queue Model (First Principles Gap 3)
// ============================================================================

use std::collections::VecDeque;

/// Calibrated queue model for fill probability estimation.
///
/// Estimates queue parameters from market data using MLE:
/// - volume_at_touch_rate: Calibrated from trade tape (volume executed at best)
/// - cancel_rate: Calibrated from book dynamics (depth reduction not from fills)
///
/// Key formulas:
/// - P(touch) = 2Φ(-δ/(σ√T))  (Black-Scholes barrier touching probability)
/// - P(execute|touch) = exp(-queue_ahead / (volume_rate × horizon))
#[derive(Debug, Clone)]
pub struct CalibratedQueueModel {
    /// Calibrated volume at touch per second (from trade tape)
    volume_at_touch_rate: f64,
    /// Calibrated cancel rate (fraction of queue cancelled per second)
    cancel_rate: f64,
    /// History of touch volumes: (timestamp_ms, volume)
    touch_volume_history: VecDeque<(u64, f64)>,
    /// History of cancel observations: (timestamp_ms, queue_delta_not_from_fills)
    cancel_history: VecDeque<(u64, f64)>,
    /// History window (ms)
    window_ms: u64,
    /// Minimum observations for calibration
    min_observations: usize,
    /// EWMA alpha for online updates
    alpha: f64,
    /// Last recorded best bid/ask depth
    last_best_bid_depth: f64,
    last_best_ask_depth: f64,
    last_update_ms: u64,
}

impl CalibratedQueueModel {
    /// Create a new calibrated queue model.
    pub fn new(window_ms: u64, min_observations: usize) -> Self {
        Self {
            volume_at_touch_rate: 1.0, // Conservative default
            cancel_rate: 0.2,          // 20% per second default
            touch_volume_history: VecDeque::with_capacity(500),
            cancel_history: VecDeque::with_capacity(500),
            window_ms,
            min_observations,
            alpha: 0.05,
            last_best_bid_depth: 0.0,
            last_best_ask_depth: 0.0,
            last_update_ms: 0,
        }
    }

    /// Create with default parameters.
    pub fn default_config() -> Self {
        Self::new(300_000, 50) // 5 minute window, 50 min observations
    }

    /// Record a trade at the touch level.
    ///
    /// This is used to calibrate volume_at_touch_rate.
    /// Only call for trades that execute at best bid or ask.
    pub fn on_touch_trade(&mut self, timestamp_ms: u64, size: f64) {
        self.touch_volume_history.push_back((timestamp_ms, size));
        self.expire_old_entries(timestamp_ms);
        self.calibrate_volume_rate(timestamp_ms);
    }

    /// Record book depth update.
    ///
    /// Used to calibrate cancel_rate by observing queue shrinkage
    /// that is not explained by fills.
    pub fn on_book_update(
        &mut self,
        timestamp_ms: u64,
        best_bid_depth: f64,
        best_ask_depth: f64,
        fill_volume_since_last: f64,
    ) {
        if self.last_update_ms == 0 {
            // First update, just store
            self.last_best_bid_depth = best_bid_depth;
            self.last_best_ask_depth = best_ask_depth;
            self.last_update_ms = timestamp_ms;
            return;
        }

        // Calculate queue changes not explained by fills
        let bid_delta = self.last_best_bid_depth - best_bid_depth;
        let ask_delta = self.last_best_ask_depth - best_ask_depth;

        // Positive delta after removing fills = cancellations
        let cancel_bid = (bid_delta - fill_volume_since_last / 2.0).max(0.0);
        let cancel_ask = (ask_delta - fill_volume_since_last / 2.0).max(0.0);

        if cancel_bid > 0.0 || cancel_ask > 0.0 {
            self.cancel_history
                .push_back((timestamp_ms, cancel_bid + cancel_ask));
        }

        self.last_best_bid_depth = best_bid_depth;
        self.last_best_ask_depth = best_ask_depth;
        self.last_update_ms = timestamp_ms;

        self.expire_old_entries(timestamp_ms);
        self.calibrate_cancel_rate(timestamp_ms);
    }

    /// Calibrate volume rate from touch trade history.
    fn calibrate_volume_rate(&mut self, timestamp_ms: u64) {
        if self.touch_volume_history.len() < self.min_observations {
            return;
        }

        // Total volume and time span
        let total_volume: f64 = self.touch_volume_history.iter().map(|(_, v)| v).sum();
        let oldest = self
            .touch_volume_history
            .front()
            .map(|(t, _)| *t)
            .unwrap_or(timestamp_ms);
        let span_secs = (timestamp_ms.saturating_sub(oldest)) as f64 / 1000.0;

        if span_secs > 0.0 {
            let new_rate = total_volume / span_secs;
            // EWMA update
            self.volume_at_touch_rate =
                self.alpha * new_rate + (1.0 - self.alpha) * self.volume_at_touch_rate;
        }
    }

    /// Calibrate cancel rate from cancel history.
    fn calibrate_cancel_rate(&mut self, timestamp_ms: u64) {
        if self.cancel_history.len() < self.min_observations {
            return;
        }

        // Average cancel volume per second
        let total_cancels: f64 = self.cancel_history.iter().map(|(_, v)| v).sum();
        let oldest = self
            .cancel_history
            .front()
            .map(|(t, _)| *t)
            .unwrap_or(timestamp_ms);
        let span_secs = (timestamp_ms.saturating_sub(oldest)) as f64 / 1000.0;

        if span_secs > 0.0 && self.last_best_bid_depth + self.last_best_ask_depth > 0.0 {
            let avg_depth = (self.last_best_bid_depth + self.last_best_ask_depth) / 2.0;
            // Cancel rate = fraction of queue cancelled per second
            let new_rate = (total_cancels / span_secs) / avg_depth.max(1e-9);
            // EWMA update, clamp to reasonable range
            self.cancel_rate =
                (self.alpha * new_rate + (1.0 - self.alpha) * self.cancel_rate).clamp(0.05, 0.8);
        }
    }

    /// Expire old entries from history.
    fn expire_old_entries(&mut self, timestamp_ms: u64) {
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);

        while self
            .touch_volume_history
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.touch_volume_history.pop_front();
        }

        while self
            .cancel_history
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.cancel_history.pop_front();
        }
    }

    /// Probability that price will touch a level δ away within horizon.
    ///
    /// Uses Black-Scholes barrier probability: P(touch) = 2Φ(-δ/(σ√T))
    ///
    /// # Arguments
    /// - `delta`: Distance from best (in price units, already converted to fraction)
    /// - `sigma`: Volatility per second
    /// - `horizon_secs`: Time horizon in seconds
    pub fn p_touch(&self, delta: f64, sigma: f64, horizon_secs: f64) -> f64 {
        if delta <= 0.0 {
            return 1.0; // Already at or through level
        }

        let sigma_sqrt_t = sigma * horizon_secs.sqrt();
        if sigma_sqrt_t < 1e-12 {
            return 0.0;
        }

        let z = -delta / sigma_sqrt_t;
        (2.0 * normal_cdf(z)).min(1.0)
    }

    /// Probability of execution given price touches our level.
    ///
    /// P(execute|touch) = exp(-queue_ahead / expected_volume_at_touch)
    ///
    /// # Arguments
    /// - `queue_ahead`: Depth ahead of us in the queue
    /// - `horizon_secs`: Time horizon in seconds
    pub fn p_execute_given_touch(&self, queue_ahead: f64, horizon_secs: f64) -> f64 {
        let expected_volume = self.volume_at_touch_rate * horizon_secs;
        if expected_volume <= 0.0 {
            return 0.0;
        }

        (-queue_ahead / expected_volume).exp().min(1.0)
    }

    /// Combined fill probability.
    ///
    /// P(fill) = P(touch) × P(execute|touch)
    pub fn p_fill(&self, delta: f64, sigma: f64, queue_ahead: f64, horizon_secs: f64) -> f64 {
        let p_t = self.p_touch(delta, sigma, horizon_secs);
        let p_e = self.p_execute_given_touch(queue_ahead, horizon_secs);
        p_t * p_e
    }

    /// Get expected queue decay over time (for position estimation).
    ///
    /// Returns the expected queue position after `dt` seconds.
    pub fn expected_queue_decay(&self, current_queue: f64, dt_secs: f64) -> f64 {
        current_queue * (-self.cancel_rate * dt_secs).exp()
    }

    // === Getters ===

    /// Get calibrated volume at touch rate (units per second).
    pub fn volume_at_touch_rate(&self) -> f64 {
        self.volume_at_touch_rate
    }

    /// Get calibrated cancel rate (fraction per second).
    pub fn cancel_rate(&self) -> f64 {
        self.cancel_rate
    }

    /// Check if model is calibrated (has enough data).
    pub fn is_calibrated(&self) -> bool {
        self.touch_volume_history.len() >= self.min_observations
    }

    /// Get number of touch volume observations.
    pub fn touch_observation_count(&self) -> usize {
        self.touch_volume_history.len()
    }
}

// ============================================================================
// Queue Position State
// ============================================================================

/// Queue position state for a single order.
#[derive(Debug, Clone)]
pub struct OrderQueuePosition {
    /// Order ID
    pub oid: u64,
    /// Price level
    pub price: f64,
    /// Our order size
    pub size: f64,
    /// Estimated depth ahead of us in queue
    pub depth_ahead: f64,
    /// Time when order was placed
    pub placed_at: Instant,
    /// Time of last queue update
    pub last_update: Instant,
    /// True if this is a bid (buy order)
    pub is_bid: bool,
}

impl OrderQueuePosition {
    /// Create a new queue position for an order.
    pub fn new(oid: u64, price: f64, size: f64, depth_ahead: f64, is_bid: bool) -> Self {
        let now = Instant::now();
        Self {
            oid,
            price,
            size,
            depth_ahead,
            placed_at: now,
            last_update: now,
            is_bid,
        }
    }

    /// Get time since order was placed (seconds).
    pub fn age_seconds(&self) -> f64 {
        self.placed_at.elapsed().as_secs_f64()
    }

    /// Get time since last update (seconds).
    pub fn time_since_update(&self) -> f64 {
        self.last_update.elapsed().as_secs_f64()
    }
}

// ============================================================================
// Queue Position Tracker
// ============================================================================

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
}

impl QueuePositionTracker {
    /// Create a new queue position tracker.
    pub fn new(config: QueueConfig) -> Self {
        Self {
            config,
            positions: HashMap::new(),
            best_bid: None,
            best_ask: None,
            sigma: 0.0001, // Default sigma
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
    /// Returns true if:
    /// - Order is old enough (beyond min_order_age_for_refresh)
    /// - Fill probability is below refresh_threshold
    pub fn should_refresh(&self, oid: u64, horizon_seconds: f64) -> bool {
        let position = match self.positions.get(&oid) {
            Some(p) => p,
            None => return false,
        };

        // Check minimum age
        if position.age_seconds() < self.config.min_order_age_for_refresh {
            return false;
        }

        // Check fill probability
        let p_fill = self.fill_probability(oid, horizon_seconds).unwrap_or(0.0);
        p_fill < self.config.refresh_threshold
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
}

/// Summary of a single queue position.
#[derive(Debug, Clone)]
pub struct QueuePositionSummary {
    pub oid: u64,
    pub price: f64,
    pub depth_ahead: f64,
    pub age_seconds: f64,
}

/// Summary of all queue positions.
#[derive(Debug, Clone)]
pub struct QueueSummary {
    pub total_orders: usize,
    pub bid_positions: Vec<QueuePositionSummary>,
    pub ask_positions: Vec<QueuePositionSummary>,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Standard normal CDF approximation.
/// Uses Abramowitz and Stegun approximation (error < 7.5e-8).
fn normal_cdf(x: f64) -> f64 {
    // For large negative x, return small probability
    if x < -8.0 {
        return 0.0;
    }
    // For large positive x, return probability close to 1
    if x > 8.0 {
        return 1.0;
    }

    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();

    let t = 1.0 / (1.0 + p * x_abs);
    let y =
        1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker() -> QueuePositionTracker {
        QueuePositionTracker::new(QueueConfig {
            cancel_decay_rate: 0.2,
            expected_volume_per_second: 10.0,
            min_queue_position: 0.01,
            default_queue_position: 5.0,
            refresh_threshold: 0.1,
            min_order_age_for_refresh: 0.1, // 100ms for fast tests
        })
    }

    #[test]
    fn test_normal_cdf() {
        // Standard normal CDF values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }

    #[test]
    fn test_order_placement() {
        let mut tracker = make_tracker();

        tracker.order_placed(1, 100.0, 1.0, 5.0, true);
        assert!(tracker.get_position(1).is_some());

        let pos = tracker.get_position(1).unwrap();
        assert_eq!(pos.oid, 1);
        assert_eq!(pos.price, 100.0);
        assert_eq!(pos.depth_ahead, 5.0);
        assert!(pos.is_bid);
    }

    #[test]
    fn test_order_removal() {
        let mut tracker = make_tracker();

        tracker.order_placed(1, 100.0, 1.0, 5.0, true);
        assert!(tracker.get_position(1).is_some());

        tracker.order_removed(1);
        assert!(tracker.get_position(1).is_none());
    }

    #[test]
    fn test_probability_touch_at_best() {
        let mut tracker = make_tracker();

        // Place order at best bid
        tracker.order_placed(1, 100.0, 1.0, 5.0, true);
        tracker.update_from_book(100.0, 101.0, 0.001);

        // At best, should have high probability of being touched
        let p_touch = tracker.probability_touch(1, 60.0).unwrap();
        assert!(p_touch > 0.9);
    }

    #[test]
    fn test_probability_touch_away_from_best() {
        let mut tracker = make_tracker();

        // Place bid order 1% away from best
        tracker.order_placed(1, 99.0, 1.0, 5.0, true);
        tracker.update_from_book(100.0, 101.0, 0.001);

        // Away from best, lower probability
        let p_touch = tracker.probability_touch(1, 60.0).unwrap();
        assert!(p_touch < 0.5);
    }

    #[test]
    fn test_fill_probability() {
        let mut tracker = make_tracker();

        tracker.order_placed(1, 100.0, 1.0, 5.0, true);
        tracker.update_from_book(100.0, 101.0, 0.001);

        let p_fill = tracker.fill_probability(1, 60.0).unwrap();
        assert!(p_fill > 0.0 && p_fill <= 1.0);
    }

    #[test]
    fn test_queue_decay() {
        let mut tracker = make_tracker();

        tracker.order_placed(1, 100.0, 1.0, 10.0, true);

        // Wait and update
        std::thread::sleep(std::time::Duration::from_millis(100));
        tracker.update_from_book(100.0, 101.0, 0.001);

        let pos = tracker.get_position(1).unwrap();
        // Depth should have decayed
        assert!(pos.depth_ahead < 10.0);
    }

    #[test]
    fn test_trades_reduce_queue() {
        let mut tracker = make_tracker();

        tracker.order_placed(1, 100.0, 1.0, 10.0, true);
        tracker.update_from_book(100.0, 101.0, 0.001);

        // Execute 5 units at our price level
        tracker.trades_at_level(100.0, 5.0, true);

        let pos = tracker.get_position(1).unwrap();
        // Depth ahead should be reduced
        assert!(pos.depth_ahead < 10.0);
    }

    #[test]
    fn test_should_refresh() {
        let mut tracker = QueuePositionTracker::new(QueueConfig {
            cancel_decay_rate: 0.2,
            expected_volume_per_second: 0.1, // Low volume = low fill prob
            min_queue_position: 0.01,
            default_queue_position: 5.0,
            refresh_threshold: 0.5,          // High threshold
            min_order_age_for_refresh: 0.05, // 50ms
        });

        // Place order with large queue ahead
        tracker.order_placed(1, 99.0, 1.0, 100.0, true);
        tracker.update_from_book(100.0, 101.0, 0.0001);

        // Wait for minimum age
        std::thread::sleep(std::time::Duration::from_millis(60));

        // Should recommend refresh (low fill prob + old enough)
        assert!(tracker.should_refresh(1, 60.0));
    }

    #[test]
    fn test_refresh_value() {
        let mut tracker = make_tracker();

        tracker.order_placed(1, 100.0, 1.0, 50.0, true);
        tracker.update_from_book(100.0, 101.0, 0.001);

        let (keep, refresh) = tracker
            .refresh_value(1, 60.0, 1.0, 1.0) // new queue pos = 1
            .unwrap();

        // Refreshing with smaller queue should have higher value
        assert!(refresh > keep);
    }
}
