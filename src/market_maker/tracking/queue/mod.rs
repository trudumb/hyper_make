//! Queue position tracking for fill probability estimation.
//!
//! Tracks where our orders sit in the order book queue and estimates
//! the probability of getting filled. This enables smarter quote
//! replacement decisions.
//!
//! # Module Structure
//!
//! - `config`: Queue tracking configuration
//! - `model`: Calibrated queue model for fill probability
//! - `position`: Queue position state types
//! - `tracker`: Queue position tracker implementation
//!
//! # Key Components
//!
//! - Queue position tracking: Estimate depth ahead of each order
//! - Decay modeling: Queue shrinks from cancels and executions
//! - Fill probability: P(fill) = P(touch) × P(execute|touch)
//! - Refresh decisions: When to replace degraded queue positions

mod comparator;
mod config;
mod model;
mod position;
mod tracker;

pub use comparator::{
    QueueKeepReason, QueueValueComparator, QueueValueConfig, QueueValueDecision, QueueValueStats,
};
pub use config::QueueConfig;
pub use model::CalibratedQueueModel;
pub use position::{OrderQueuePosition, QueuePositionSummary, QueueSummary};
pub use tracker::QueuePositionTracker;

/// Standard normal CDF approximation.
/// Uses Abramowitz and Stegun approximation (error < 7.5e-8).
pub(crate) fn normal_cdf(x: f64) -> f64 {
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
            refresh_cost_bps: 5.0,          // EV cost of refresh
            spread_capture_bps: 8.0,        // Expected spread capture
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
            refresh_cost_bps: 5.0,           // EV cost of refresh
            spread_capture_bps: 8.0,         // Expected spread capture
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

    // === Tests for L2 depth estimation (Phase 1: Queue Observability) ===

    #[test]
    fn test_update_depth_from_book_caps_at_book_depth() {
        let mut tracker = make_tracker();

        // Place a bid with large initial depth
        tracker.order_placed(1, 99.5, 1.0, 100.0, true);

        // Book shows only 8.0 units ahead at our price
        let bids = vec![(100.0, 5.0), (99.5, 3.0)];
        let asks = vec![(100.5, 4.0)];
        tracker.update_depth_from_book(&bids, &asks);

        let pos = tracker.get_position(1).unwrap();
        // depth should be min(100.0, book_depth) = book_depth
        // book_depth at 99.5 bid = 5.0 (at 100.0, ahead) + 3.0 (at 99.5, same level) = 8.0
        assert!(
            (pos.depth_ahead - 8.0).abs() < 0.01,
            "depth should be capped at book depth 8.0, got {}",
            pos.depth_ahead
        );
    }

    #[test]
    fn test_update_depth_from_book_preserves_lower_depth() {
        let mut tracker = make_tracker();

        // Place bid with small depth (already advanced through fills)
        tracker.order_placed(1, 99.5, 1.0, 2.0, true);

        // Book shows 8.0 units ahead, but we've already advanced past some
        let bids = vec![(100.0, 5.0), (99.5, 3.0)];
        let asks = vec![(100.5, 4.0)];
        tracker.update_depth_from_book(&bids, &asks);

        let pos = tracker.get_position(1).unwrap();
        // depth should be min(2.0, 8.0) = 2.0 (preserve queue advancement)
        assert!(
            (pos.depth_ahead - 2.0).abs() < 0.01,
            "depth should preserve lower tracked value 2.0, got {}",
            pos.depth_ahead
        );
    }

    #[test]
    fn test_update_depth_from_book_ask_side() {
        let mut tracker = make_tracker();

        // Place an ask at 100.5 with large initial depth
        tracker.order_placed(1, 100.5, 1.0, 50.0, false);

        // Book shows 4.0 ahead at ask side
        let bids = vec![(100.0, 5.0)];
        let asks = vec![(100.0, 2.0), (100.5, 2.0)];
        tracker.update_depth_from_book(&bids, &asks);

        let pos = tracker.get_position(1).unwrap();
        // For asks: prices < 100.5 are ahead (2.0 at 100.0) + same level (2.0 at 100.5) = 4.0
        assert!(
            (pos.depth_ahead - 4.0).abs() < 0.01,
            "ask depth should be 4.0, got {}",
            pos.depth_ahead
        );
    }

    #[test]
    fn test_on_market_trade_decrements_depth() {
        let mut tracker = make_tracker();

        // Place a bid at 100.0
        tracker.order_placed(1, 100.0, 1.0, 10.0, true);
        // Place an ask at 101.0
        tracker.order_placed(2, 101.0, 1.0, 8.0, false);

        // A sell trade at 100.0 hits bids — bid-side queue drains
        tracker.on_market_trade(100.0, 3.0, false);

        let bid_pos = tracker.get_position(1).unwrap();
        assert!(
            (bid_pos.depth_ahead - 7.0).abs() < 0.01,
            "bid depth should decrease by 3.0 to 7.0, got {}",
            bid_pos.depth_ahead
        );

        // Ask should be unaffected
        let ask_pos = tracker.get_position(2).unwrap();
        assert!(
            (ask_pos.depth_ahead - 8.0).abs() < 0.01,
            "ask depth should be unchanged at 8.0, got {}",
            ask_pos.depth_ahead
        );
    }

    #[test]
    fn test_on_market_trade_buy_decrements_asks() {
        let mut tracker = make_tracker();

        // Place an ask at 101.0
        tracker.order_placed(1, 101.0, 1.0, 10.0, false);

        // A buy trade at 101.0 lifts asks — ask-side queue drains
        tracker.on_market_trade(101.0, 4.0, true);

        let pos = tracker.get_position(1).unwrap();
        assert!(
            (pos.depth_ahead - 6.0).abs() < 0.01,
            "ask depth should decrease by 4.0 to 6.0, got {}",
            pos.depth_ahead
        );
    }

    #[test]
    fn test_depth_never_negative() {
        let mut tracker = make_tracker();

        tracker.order_placed(1, 100.0, 1.0, 2.0, true);

        // A sell at 100.0 hits bids — trade volume far exceeds depth ahead
        tracker.on_market_trade(100.0, 100.0, false);

        let pos = tracker.get_position(1).unwrap();
        assert!(
            pos.depth_ahead >= 0.0,
            "depth should never be negative, got {}",
            pos.depth_ahead
        );
        // Should be clamped at min_queue_position (0.01)
        assert!(
            (pos.depth_ahead - 0.01).abs() < 0.001,
            "depth should be clamped at min_queue_position 0.01, got {}",
            pos.depth_ahead
        );
    }

    #[test]
    fn test_estimate_depth_at_price_with_empty_book() {
        let tracker = make_tracker();

        // No L2 data cached — should return default
        let depth = tracker.estimate_depth_at_price(100.0, true);
        assert!(
            (depth - 5.0).abs() < 0.01,
            "should return default_queue_position (5.0) with no book, got {}",
            depth
        );
    }

    #[test]
    fn test_estimate_depth_at_price_bid() {
        let mut tracker = make_tracker();

        // Cache L2 levels
        let bids = vec![(100.0, 5.0), (99.5, 3.0), (99.0, 2.0)];
        let asks = vec![(100.5, 4.0)];
        tracker.update_depth_from_book(&bids, &asks);

        // Depth at best bid (100.0): all size at 100.0 = 5.0 (we join at back)
        let depth_best = tracker.estimate_depth_at_price(100.0, true);
        assert!(
            (depth_best - 5.0).abs() < 0.01,
            "depth at best bid should be 5.0, got {}",
            depth_best
        );

        // Depth at 99.5: size at 100.0 (5.0 ahead) + at 99.5 (3.0 same level) = 8.0
        let depth_mid = tracker.estimate_depth_at_price(99.5, true);
        assert!(
            (depth_mid - 8.0).abs() < 0.01,
            "depth at 99.5 bid should be 8.0, got {}",
            depth_mid
        );
    }

    #[test]
    fn test_estimate_depth_at_price_ask() {
        let mut tracker = make_tracker();

        let bids = vec![(100.0, 5.0)];
        let asks = vec![(100.5, 4.0), (101.0, 3.0), (101.5, 2.0)];
        tracker.update_depth_from_book(&bids, &asks);

        // Depth at best ask (100.5): all size at 100.5 = 4.0
        let depth_best = tracker.estimate_depth_at_price(100.5, false);
        assert!(
            (depth_best - 4.0).abs() < 0.01,
            "depth at best ask should be 4.0, got {}",
            depth_best
        );

        // Depth at 101.0: size at 100.5 (4.0 ahead) + at 101.0 (3.0 same level) = 7.0
        let depth_away = tracker.estimate_depth_at_price(101.0, false);
        assert!(
            (depth_away - 7.0).abs() < 0.01,
            "depth at 101.0 ask should be 7.0, got {}",
            depth_away
        );
    }
}
