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

mod config;
mod model;
mod position;
mod tracker;

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
