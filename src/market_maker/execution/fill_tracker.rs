//! Fill quality tracking and metrics computation.
//!
//! This module provides execution quality tracking for market making fills:
//! - Fill rate monitoring (fills / quotes placed)
//! - Adverse selection rate (% of fills where price moved against within 1s)
//! - Queue position tracking
//! - Latency metrics (p50, p99)
//!
//! # Usage
//!
//! ```ignore
//! let mut tracker = FillTracker::new(1000);
//!
//! // Record quotes placed
//! tracker.record_quote();
//!
//! // Record fills when they occur
//! tracker.record_fill(FillRecord {
//!     timestamp: current_time_ms(),
//!     side: Side::Bid,
//!     fill_price: 50000.0,
//!     fill_size: 0.01,
//!     queue_position: 5,
//!     latency_ms: 15.0,
//!     price_after_1s: None,  // Updated later
//! });
//!
//! // Update price after 1s for adverse selection calculation
//! tracker.update_price_after(fill_timestamp, current_price);
//!
//! // Get metrics
//! let metrics = tracker.metrics();
//! ```

use std::collections::VecDeque;

/// Side of a fill (Bid = we bought, Ask = we sold).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    /// Convert from boolean is_buy
    pub fn from_is_buy(is_buy: bool) -> Self {
        if is_buy {
            Side::Bid
        } else {
            Side::Ask
        }
    }

    /// Convert to boolean is_buy
    pub fn is_buy(&self) -> bool {
        matches!(self, Side::Bid)
    }
}

/// Aggregated fill quality metrics.
#[derive(Debug, Clone, Default)]
pub struct FillMetrics {
    /// Fills / quotes placed (0.0 to 1.0)
    pub fill_rate: f64,
    /// % of fills where price moved against within 1s
    pub adverse_selection_rate: f64,
    /// Average queue position at fill time
    pub queue_position_mean: f64,
    /// 50th percentile latency in milliseconds
    pub latency_p50_ms: f64,
    /// 99th percentile latency in milliseconds
    pub latency_p99_ms: f64,
    /// Total number of fills in the window
    pub total_fills: usize,
    /// Total number of quotes placed in the window
    pub total_quotes: usize,
}

impl FillMetrics {
    /// Check if we have enough data for meaningful metrics.
    pub fn has_sufficient_data(&self, min_fills: usize) -> bool {
        self.total_fills >= min_fills
    }

    /// Get the fill rate as a percentage.
    pub fn fill_rate_pct(&self) -> f64 {
        self.fill_rate * 100.0
    }

    /// Get the adverse selection rate as a percentage.
    pub fn adverse_selection_rate_pct(&self) -> f64 {
        self.adverse_selection_rate * 100.0
    }
}

/// A single fill record with execution quality data.
#[derive(Debug, Clone)]
pub struct FillRecord {
    /// Timestamp in milliseconds since epoch
    pub timestamp: u64,
    /// Side of the fill
    pub side: Side,
    /// Price at which we were filled
    pub fill_price: f64,
    /// Size of the fill
    pub fill_size: f64,
    /// Queue position at fill time (0 = front of queue)
    pub queue_position: usize,
    /// Latency from order placement to fill in milliseconds
    pub latency_ms: f64,
    /// Price 1 second after fill (for adverse selection calculation)
    /// None until we have the data
    pub price_after_1s: Option<f64>,
}

impl FillRecord {
    /// Create a new fill record.
    pub fn new(
        timestamp: u64,
        side: Side,
        fill_price: f64,
        fill_size: f64,
        queue_position: usize,
        latency_ms: f64,
    ) -> Self {
        Self {
            timestamp,
            side,
            fill_price,
            fill_size,
            queue_position,
            latency_ms,
            price_after_1s: None,
        }
    }

    /// Check if this fill was adversely selected.
    ///
    /// Returns None if we don't have the price after 1s yet.
    /// Returns Some(true) if price moved against us:
    /// - For bids (we bought): price dropped
    /// - For asks (we sold): price rose
    pub fn is_adversely_selected(&self) -> Option<bool> {
        self.price_after_1s.map(|price_after| {
            let price_move = price_after - self.fill_price;
            match self.side {
                Side::Bid => price_move < 0.0, // We bought, price dropped
                Side::Ask => price_move > 0.0, // We sold, price rose
            }
        })
    }

    /// Get the price movement in basis points after 1s.
    ///
    /// Returns None if we don't have the price after 1s yet.
    /// Positive = price went up, Negative = price went down.
    pub fn price_move_bps(&self) -> Option<f64> {
        self.price_after_1s.map(|price_after| {
            if self.fill_price > 0.0 {
                ((price_after - self.fill_price) / self.fill_price) * 10_000.0
            } else {
                0.0
            }
        })
    }

    /// Get the adverse selection cost in basis points.
    ///
    /// Returns None if we don't have the price after 1s yet.
    /// Positive = we lost money (adverse selection), Negative = we made money.
    pub fn adverse_selection_bps(&self) -> Option<f64> {
        self.price_move_bps().map(|bps| {
            match self.side {
                Side::Bid => -bps, // We bought, so price drop is loss
                Side::Ask => bps,  // We sold, so price rise is loss
            }
        })
    }
}

/// Tracks fill quality metrics over a rolling window.
pub struct FillTracker {
    /// Maximum number of fills to keep in the window
    window_size: usize,
    /// Recent fills in the window
    fills: VecDeque<FillRecord>,
    /// Number of quotes placed (for fill rate calculation)
    quotes_placed: usize,
    /// Latencies for percentile calculation (separate from fills for efficiency)
    latencies: VecDeque<f64>,
}

impl FillTracker {
    /// Create a new fill tracker with the specified window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Maximum number of fills to keep in the rolling window
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            fills: VecDeque::with_capacity(window_size),
            quotes_placed: 0,
            latencies: VecDeque::with_capacity(window_size),
        }
    }

    /// Record that a quote was placed.
    ///
    /// This is used to calculate the fill rate (fills / quotes).
    pub fn record_quote(&mut self) {
        self.quotes_placed = self.quotes_placed.saturating_add(1);
    }

    /// Record multiple quotes placed at once.
    pub fn record_quotes(&mut self, count: usize) {
        self.quotes_placed = self.quotes_placed.saturating_add(count);
    }

    /// Record a fill.
    ///
    /// Maintains the rolling window by removing old fills when at capacity.
    pub fn record_fill(&mut self, record: FillRecord) {
        // Track latency
        self.latencies.push_back(record.latency_ms);
        if self.latencies.len() > self.window_size {
            self.latencies.pop_front();
        }

        // Track fill
        self.fills.push_back(record);
        if self.fills.len() > self.window_size {
            self.fills.pop_front();
        }
    }

    /// Update the price 1 second after a fill occurred.
    ///
    /// This is used to calculate adverse selection. Finds the fill by timestamp
    /// and updates its `price_after_1s` field.
    ///
    /// # Arguments
    ///
    /// * `fill_timestamp` - The timestamp of the fill to update
    /// * `price` - The price 1 second after the fill
    ///
    /// # Returns
    ///
    /// true if the fill was found and updated, false otherwise
    pub fn update_price_after(&mut self, fill_timestamp: u64, price: f64) -> bool {
        for fill in self.fills.iter_mut().rev() {
            if fill.timestamp == fill_timestamp && fill.price_after_1s.is_none() {
                fill.price_after_1s = Some(price);
                return true;
            }
        }
        false
    }

    /// Compute aggregated fill metrics.
    pub fn metrics(&self) -> FillMetrics {
        if self.fills.is_empty() {
            return FillMetrics {
                total_quotes: self.quotes_placed,
                ..Default::default()
            };
        }

        // Fill rate
        let fill_rate = if self.quotes_placed > 0 {
            self.fills.len() as f64 / self.quotes_placed as f64
        } else {
            0.0
        };

        // Adverse selection rate
        let (as_count, as_total) = self.fills.iter().fold((0usize, 0usize), |acc, fill| {
            match fill.is_adversely_selected() {
                Some(true) => (acc.0 + 1, acc.1 + 1),
                Some(false) => (acc.0, acc.1 + 1),
                None => acc, // Skip fills without price_after_1s
            }
        });
        let adverse_selection_rate = if as_total > 0 {
            as_count as f64 / as_total as f64
        } else {
            0.0
        };

        // Queue position mean
        let queue_position_sum: usize = self.fills.iter().map(|f| f.queue_position).sum();
        let queue_position_mean = queue_position_sum as f64 / self.fills.len() as f64;

        // Latency percentiles
        let (latency_p50_ms, latency_p99_ms) = if !self.latencies.is_empty() {
            let mut sorted: Vec<f64> = self.latencies.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            (percentile(&sorted, 0.50), percentile(&sorted, 0.99))
        } else {
            (0.0, 0.0)
        };

        FillMetrics {
            fill_rate,
            adverse_selection_rate,
            queue_position_mean,
            latency_p50_ms,
            latency_p99_ms,
            total_fills: self.fills.len(),
            total_quotes: self.quotes_placed,
        }
    }

    /// Check if the current fill rate is within normal bounds.
    ///
    /// # Arguments
    ///
    /// * `baseline` - Expected baseline fill rate (e.g., historical average)
    /// * `threshold` - Acceptable deviation from baseline (e.g., 0.2 for 20%)
    ///
    /// # Returns
    ///
    /// true if fill rate is within [baseline * (1 - threshold), baseline * (1 + threshold)]
    pub fn is_fill_rate_normal(&self, baseline: f64, threshold: f64) -> bool {
        let metrics = self.metrics();
        if metrics.total_fills < 10 {
            // Not enough data
            return true;
        }
        let lower = baseline * (1.0 - threshold);
        let upper = baseline * (1.0 + threshold);
        metrics.fill_rate >= lower && metrics.fill_rate <= upper
    }

    /// Get the most recent N fills.
    ///
    /// Returns fills in reverse chronological order (most recent first).
    pub fn recent_fills(&self, n: usize) -> Vec<&FillRecord> {
        self.fills.iter().rev().take(n).collect()
    }

    /// Get all fills in the window.
    pub fn all_fills(&self) -> impl Iterator<Item = &FillRecord> {
        self.fills.iter()
    }

    /// Get the number of fills in the window.
    pub fn fill_count(&self) -> usize {
        self.fills.len()
    }

    /// Reset the tracker, clearing all data.
    pub fn reset(&mut self) {
        self.fills.clear();
        self.latencies.clear();
        self.quotes_placed = 0;
    }

    /// Get fills by side.
    pub fn fills_by_side(&self, side: Side) -> impl Iterator<Item = &FillRecord> {
        self.fills.iter().filter(move |f| f.side == side)
    }

    /// Compute metrics for a specific side only.
    pub fn metrics_by_side(&self, side: Side) -> FillMetrics {
        let side_fills: Vec<_> = self.fills.iter().filter(|f| f.side == side).collect();

        if side_fills.is_empty() {
            return FillMetrics::default();
        }

        // Adverse selection rate for this side
        let (as_count, as_total) =
            side_fills.iter().fold((0usize, 0usize), |acc, fill| {
                match fill.is_adversely_selected() {
                    Some(true) => (acc.0 + 1, acc.1 + 1),
                    Some(false) => (acc.0, acc.1 + 1),
                    None => acc,
                }
            });
        let adverse_selection_rate = if as_total > 0 {
            as_count as f64 / as_total as f64
        } else {
            0.0
        };

        // Queue position mean
        let queue_position_sum: usize = side_fills.iter().map(|f| f.queue_position).sum();
        let queue_position_mean = queue_position_sum as f64 / side_fills.len() as f64;

        // Latency percentiles for this side
        let mut latencies: Vec<f64> = side_fills.iter().map(|f| f.latency_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (latency_p50_ms, latency_p99_ms) = if !latencies.is_empty() {
            (percentile(&latencies, 0.50), percentile(&latencies, 0.99))
        } else {
            (0.0, 0.0)
        };

        FillMetrics {
            fill_rate: 0.0, // Cannot compute per-side fill rate without per-side quote count
            adverse_selection_rate,
            queue_position_mean,
            latency_p50_ms,
            latency_p99_ms,
            total_fills: side_fills.len(),
            total_quotes: 0, // Not tracked per-side
        }
    }

    /// Get average adverse selection cost in basis points.
    ///
    /// Returns None if no fills have price_after_1s data.
    pub fn avg_adverse_selection_bps(&self) -> Option<f64> {
        let as_values: Vec<f64> = self
            .fills
            .iter()
            .filter_map(|f| f.adverse_selection_bps())
            .collect();

        if as_values.is_empty() {
            None
        } else {
            Some(as_values.iter().sum::<f64>() / as_values.len() as f64)
        }
    }
}

/// Calculate a percentile from a sorted slice.
///
/// Uses linear interpolation between closest ranks.
///
/// # Arguments
///
/// * `sorted` - A sorted slice of values (ascending order)
/// * `p` - The percentile to calculate (0.0 to 1.0)
///
/// # Returns
///
/// The value at the specified percentile
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let p = p.clamp(0.0, 1.0);
    let n = sorted.len();

    // Use the "nearest rank" method with linear interpolation
    let rank = p * (n - 1) as f64;
    let lower_idx = rank.floor() as usize;
    let upper_idx = (lower_idx + 1).min(n - 1);
    let fraction = rank - lower_idx as f64;

    sorted[lower_idx] * (1.0 - fraction) + sorted[upper_idx] * fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_side_conversion() {
        assert_eq!(Side::from_is_buy(true), Side::Bid);
        assert_eq!(Side::from_is_buy(false), Side::Ask);
        assert!(Side::Bid.is_buy());
        assert!(!Side::Ask.is_buy());
    }

    #[test]
    fn test_fill_record_adverse_selection_bid() {
        // We bought (bid), price dropped = adversely selected
        let mut fill = FillRecord::new(1000, Side::Bid, 50000.0, 0.01, 5, 10.0);
        assert!(fill.is_adversely_selected().is_none());

        fill.price_after_1s = Some(49990.0); // Price dropped
        assert_eq!(fill.is_adversely_selected(), Some(true));

        fill.price_after_1s = Some(50010.0); // Price rose
        assert_eq!(fill.is_adversely_selected(), Some(false));
    }

    #[test]
    fn test_fill_record_adverse_selection_ask() {
        // We sold (ask), price rose = adversely selected
        let mut fill = FillRecord::new(1000, Side::Ask, 50000.0, 0.01, 5, 10.0);

        fill.price_after_1s = Some(50010.0); // Price rose
        assert_eq!(fill.is_adversely_selected(), Some(true));

        fill.price_after_1s = Some(49990.0); // Price dropped
        assert_eq!(fill.is_adversely_selected(), Some(false));
    }

    #[test]
    fn test_fill_record_price_move_bps() {
        let mut fill = FillRecord::new(1000, Side::Bid, 50000.0, 0.01, 5, 10.0);
        fill.price_after_1s = Some(50050.0); // +50 / 50000 = +10 bps

        let bps = fill.price_move_bps().unwrap();
        assert!((bps - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_record_adverse_selection_bps_bid() {
        let mut fill = FillRecord::new(1000, Side::Bid, 50000.0, 0.01, 5, 10.0);
        fill.price_after_1s = Some(49950.0); // -50 / 50000 = -10 bps move

        // For bid, price drop = loss, so AS cost = -(-10) = +10 bps
        let as_bps = fill.adverse_selection_bps().unwrap();
        assert!((as_bps - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_record_adverse_selection_bps_ask() {
        let mut fill = FillRecord::new(1000, Side::Ask, 50000.0, 0.01, 5, 10.0);
        fill.price_after_1s = Some(50050.0); // +50 / 50000 = +10 bps move

        // For ask, price rise = loss, so AS cost = +10 bps
        let as_bps = fill.adverse_selection_bps().unwrap();
        assert!((as_bps - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_tracker_new() {
        let tracker = FillTracker::new(100);
        assert_eq!(tracker.window_size, 100);
        assert_eq!(tracker.fill_count(), 0);
    }

    #[test]
    fn test_fill_tracker_record_quote() {
        let mut tracker = FillTracker::new(100);
        tracker.record_quote();
        tracker.record_quote();
        tracker.record_quotes(3);

        let metrics = tracker.metrics();
        assert_eq!(metrics.total_quotes, 5);
    }

    #[test]
    fn test_fill_tracker_record_fill() {
        let mut tracker = FillTracker::new(100);

        let fill = FillRecord::new(1000, Side::Bid, 50000.0, 0.01, 5, 15.0);
        tracker.record_fill(fill);

        assert_eq!(tracker.fill_count(), 1);
    }

    #[test]
    fn test_fill_tracker_window_limit() {
        let mut tracker = FillTracker::new(3);

        for i in 0..5 {
            let fill = FillRecord::new(i as u64, Side::Bid, 50000.0, 0.01, 5, 10.0);
            tracker.record_fill(fill);
        }

        // Should only have 3 fills (window size)
        assert_eq!(tracker.fill_count(), 3);

        // Most recent fills should be timestamps 2, 3, 4
        let recent: Vec<_> = tracker.recent_fills(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].timestamp, 4);
        assert_eq!(recent[1].timestamp, 3);
        assert_eq!(recent[2].timestamp, 2);
    }

    #[test]
    fn test_fill_tracker_update_price_after() {
        let mut tracker = FillTracker::new(100);

        let fill = FillRecord::new(1000, Side::Bid, 50000.0, 0.01, 5, 15.0);
        tracker.record_fill(fill);

        // Update should succeed
        assert!(tracker.update_price_after(1000, 50010.0));

        // Updating again should fail (already has price)
        assert!(!tracker.update_price_after(1000, 50020.0));

        // Wrong timestamp should fail
        assert!(!tracker.update_price_after(999, 50010.0));
    }

    #[test]
    fn test_fill_tracker_metrics_fill_rate() {
        let mut tracker = FillTracker::new(100);

        // 10 quotes, 5 fills = 50% fill rate
        for _ in 0..10 {
            tracker.record_quote();
        }
        for i in 0..5 {
            let fill = FillRecord::new(i, Side::Bid, 50000.0, 0.01, 5, 10.0);
            tracker.record_fill(fill);
        }

        let metrics = tracker.metrics();
        assert!((metrics.fill_rate - 0.5).abs() < 0.01);
        assert_eq!(metrics.total_fills, 5);
        assert_eq!(metrics.total_quotes, 10);
    }

    #[test]
    fn test_fill_tracker_metrics_adverse_selection_rate() {
        let mut tracker = FillTracker::new(100);

        // 4 fills: 2 adversely selected, 2 not
        for i in 0..4 {
            let mut fill = FillRecord::new(i, Side::Bid, 50000.0, 0.01, 5, 10.0);
            if i % 2 == 0 {
                fill.price_after_1s = Some(49990.0); // Adversely selected (price dropped)
            } else {
                fill.price_after_1s = Some(50010.0); // Not adversely selected
            }
            tracker.record_fill(fill);
        }

        let metrics = tracker.metrics();
        assert!((metrics.adverse_selection_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fill_tracker_metrics_queue_position() {
        let mut tracker = FillTracker::new(100);

        // Queue positions: 2, 4, 6 -> mean = 4
        for (i, qp) in [2, 4, 6].iter().enumerate() {
            let fill = FillRecord::new(i as u64, Side::Bid, 50000.0, 0.01, *qp, 10.0);
            tracker.record_fill(fill);
        }

        let metrics = tracker.metrics();
        assert!((metrics.queue_position_mean - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_tracker_metrics_latency_percentiles() {
        let mut tracker = FillTracker::new(100);

        // Latencies: 10, 20, 30, ..., 100
        for i in 1..=10 {
            let fill = FillRecord::new(i, Side::Bid, 50000.0, 0.01, 5, (i * 10) as f64);
            tracker.record_fill(fill);
        }

        let metrics = tracker.metrics();
        // p50 should be around 50-60ms
        assert!(metrics.latency_p50_ms >= 50.0 && metrics.latency_p50_ms <= 60.0);
        // p99 should be close to 100ms
        assert!(metrics.latency_p99_ms >= 95.0);
    }

    #[test]
    fn test_fill_tracker_is_fill_rate_normal() {
        let mut tracker = FillTracker::new(100);

        // 100 quotes, 50 fills = 50% fill rate
        for _ in 0..100 {
            tracker.record_quote();
        }
        for i in 0..50 {
            let fill = FillRecord::new(i, Side::Bid, 50000.0, 0.01, 5, 10.0);
            tracker.record_fill(fill);
        }

        // 50% fill rate with baseline 50% and 20% threshold should be normal
        assert!(tracker.is_fill_rate_normal(0.5, 0.2));

        // 50% fill rate with baseline 80% and 20% threshold should NOT be normal
        assert!(!tracker.is_fill_rate_normal(0.8, 0.2));
    }

    #[test]
    fn test_fill_tracker_recent_fills() {
        let mut tracker = FillTracker::new(100);

        for i in 0..5 {
            let fill = FillRecord::new(i, Side::Bid, 50000.0, 0.01, 5, 10.0);
            tracker.record_fill(fill);
        }

        let recent = tracker.recent_fills(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].timestamp, 4); // Most recent first
        assert_eq!(recent[1].timestamp, 3);
        assert_eq!(recent[2].timestamp, 2);
    }

    #[test]
    fn test_fill_tracker_reset() {
        let mut tracker = FillTracker::new(100);

        tracker.record_quote();
        let fill = FillRecord::new(1, Side::Bid, 50000.0, 0.01, 5, 10.0);
        tracker.record_fill(fill);

        assert_eq!(tracker.fill_count(), 1);
        assert_eq!(tracker.metrics().total_quotes, 1);

        tracker.reset();

        assert_eq!(tracker.fill_count(), 0);
        assert_eq!(tracker.metrics().total_quotes, 0);
    }

    #[test]
    fn test_fill_tracker_fills_by_side() {
        let mut tracker = FillTracker::new(100);

        tracker.record_fill(FillRecord::new(1, Side::Bid, 50000.0, 0.01, 5, 10.0));
        tracker.record_fill(FillRecord::new(2, Side::Ask, 50000.0, 0.01, 5, 10.0));
        tracker.record_fill(FillRecord::new(3, Side::Bid, 50000.0, 0.01, 5, 10.0));

        let bids: Vec<_> = tracker.fills_by_side(Side::Bid).collect();
        let asks: Vec<_> = tracker.fills_by_side(Side::Ask).collect();

        assert_eq!(bids.len(), 2);
        assert_eq!(asks.len(), 1);
    }

    #[test]
    fn test_fill_tracker_metrics_by_side() {
        let mut tracker = FillTracker::new(100);

        // Add 2 bids with queue positions 2 and 4 (mean = 3)
        tracker.record_fill(FillRecord::new(1, Side::Bid, 50000.0, 0.01, 2, 10.0));
        tracker.record_fill(FillRecord::new(2, Side::Bid, 50000.0, 0.01, 4, 10.0));

        // Add 1 ask with queue position 10
        tracker.record_fill(FillRecord::new(3, Side::Ask, 50000.0, 0.01, 10, 20.0));

        let bid_metrics = tracker.metrics_by_side(Side::Bid);
        assert_eq!(bid_metrics.total_fills, 2);
        assert!((bid_metrics.queue_position_mean - 3.0).abs() < 0.01);

        let ask_metrics = tracker.metrics_by_side(Side::Ask);
        assert_eq!(ask_metrics.total_fills, 1);
        assert!((ask_metrics.queue_position_mean - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_tracker_avg_adverse_selection_bps() {
        let mut tracker = FillTracker::new(100);

        // Bid: bought at 50000, price dropped to 49950 -> AS = +10 bps loss
        let mut fill1 = FillRecord::new(1, Side::Bid, 50000.0, 0.01, 5, 10.0);
        fill1.price_after_1s = Some(49950.0);
        tracker.record_fill(fill1);

        // Ask: sold at 50000, price rose to 50100 -> AS = +20 bps loss
        let mut fill2 = FillRecord::new(2, Side::Ask, 50000.0, 0.01, 5, 10.0);
        fill2.price_after_1s = Some(50100.0);
        tracker.record_fill(fill2);

        // Average AS = (10 + 20) / 2 = 15 bps
        let avg_as = tracker.avg_adverse_selection_bps().unwrap();
        assert!((avg_as - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_percentile_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((percentile(&values, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile(&values, 0.5) - 3.0).abs() < 0.01);
        assert!((percentile(&values, 1.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_interpolation() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        // p25 should be between 10 and 20
        let p25 = percentile(&values, 0.25);
        assert!((10.0..=30.0).contains(&p25));

        // p75 should be between 30 and 50
        let p75 = percentile(&values, 0.75);
        assert!((30.0..=50.0).contains(&p75));
    }

    #[test]
    fn test_percentile_empty() {
        let values: Vec<f64> = vec![];
        assert!((percentile(&values, 0.5) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_single_element() {
        let values = vec![42.0];
        assert!((percentile(&values, 0.0) - 42.0).abs() < 0.01);
        assert!((percentile(&values, 0.5) - 42.0).abs() < 0.01);
        assert!((percentile(&values, 1.0) - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_metrics_has_sufficient_data() {
        let metrics = FillMetrics {
            total_fills: 5,
            ..Default::default()
        };

        assert!(!metrics.has_sufficient_data(10));
        assert!(metrics.has_sufficient_data(5));
        assert!(metrics.has_sufficient_data(3));
    }

    #[test]
    fn test_fill_metrics_percentages() {
        let metrics = FillMetrics {
            fill_rate: 0.5,
            adverse_selection_rate: 0.25,
            ..Default::default()
        };

        assert!((metrics.fill_rate_pct() - 50.0).abs() < 0.01);
        assert!((metrics.adverse_selection_rate_pct() - 25.0).abs() < 0.01);
    }
}
