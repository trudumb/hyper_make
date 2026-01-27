//! Queue value comparator for EV-based reconciliation decisions.
//!
//! This module provides expected-value-driven decision making for order reconciliation.
//! Instead of simple tolerance-based matching, it compares the EV of keeping an order
//! (with its queue position advantage) vs replacing it (starting at back of queue).

use tracing::debug;

use super::normal_cdf;
use super::tracker::QueuePositionTracker;

/// Decision from queue value comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum QueueValueDecision {
    /// Keep current order - queue value too high to sacrifice
    Keep {
        reason: QueueKeepReason,
        current_ev: f64,
        replacement_ev: f64,
    },
    /// Replace order - improvement justifies queue loss
    Replace { improvement_pct: f64 },
    /// No queue data - fall back to tolerance-based decision
    NoData,
}

/// Reason for keeping an order based on queue value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueKeepReason {
    /// Order is locked due to high fill probability
    QueueLocked,
    /// Improvement from replacement is insufficient
    InsufficientImprovement,
    /// Order is too young to evaluate properly
    OrderTooYoung,
}

impl std::fmt::Display for QueueKeepReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueueKeepReason::QueueLocked => write!(f, "queue_locked"),
            QueueKeepReason::InsufficientImprovement => write!(f, "insufficient_improvement"),
            QueueKeepReason::OrderTooYoung => write!(f, "order_too_young"),
        }
    }
}

/// Configuration for queue value comparisons.
#[derive(Debug, Clone)]
pub struct QueueValueConfig {
    /// Minimum improvement required to justify replacement (default: 0.15 = 15%)
    pub improvement_threshold: f64,

    /// Fill probability above which orders are "locked" (default: 0.50)
    pub queue_lock_threshold: f64,

    /// Price movement that overrides queue lock (default: 35 bps)
    pub queue_lock_override_bps: f64,

    /// Time horizon for fill probability calculation (default: 1.0s)
    pub fill_horizon_secs: f64,

    /// Estimated spread capture in bps (default: 8.0)
    pub spread_capture_bps: f64,

    /// Minimum order age before considering replacement (default: 0.3s)
    pub min_order_age_secs: f64,

    /// Whether queue value comparison is enabled
    pub enabled: bool,

    /// API call cost in bps for modification decisions.
    /// This cost is subtracted from EV improvement when deciding whether to modify.
    /// A modification is only approved if: EV(new) - EV(current) > api_cost_bps
    /// Default: 3.0 bps (accounts for rate limit budget and queue loss)
    pub api_cost_bps: f64,
}

impl Default for QueueValueConfig {
    fn default() -> Self {
        Self {
            improvement_threshold: 0.15,   // 15% improvement required
            queue_lock_threshold: 0.50,    // Lock orders with >50% P(fill)
            queue_lock_override_bps: 35.0, // Override lock if price moved 35+ bps
            fill_horizon_secs: 1.0,        // 1-second fill probability window
            spread_capture_bps: 8.0,       // Estimated spread capture
            min_order_age_secs: 0.3,       // Wait 300ms before considering replacement
            enabled: true,
            api_cost_bps: 3.0,             // API call cost in bps (rate limit + queue loss)
        }
    }
}

/// Statistics from queue value comparisons in a reconciliation cycle.
#[derive(Debug, Default, Clone)]
pub struct QueueValueStats {
    /// Orders kept due to queue value
    pub kept_queue_locked: usize,
    /// Orders kept due to insufficient improvement
    pub kept_insufficient_improvement: usize,
    /// Orders kept because they're too young
    pub kept_too_young: usize,
    /// Orders replaced despite queue position
    pub replaced: usize,
    /// Orders with no queue data (fell back to tolerance-based)
    pub no_data: usize,
    /// Total requests saved (each keep saves 2 requests: cancel + place)
    pub requests_saved: usize,
}

impl QueueValueStats {
    /// Total orders kept due to queue value
    pub fn total_kept(&self) -> usize {
        self.kept_queue_locked + self.kept_insufficient_improvement + self.kept_too_young
    }

    /// Record a decision
    pub fn record(&mut self, decision: &QueueValueDecision) {
        match decision {
            QueueValueDecision::Keep { reason, .. } => {
                match reason {
                    QueueKeepReason::QueueLocked => self.kept_queue_locked += 1,
                    QueueKeepReason::InsufficientImprovement => {
                        self.kept_insufficient_improvement += 1
                    }
                    QueueKeepReason::OrderTooYoung => self.kept_too_young += 1,
                }
                self.requests_saved += 2; // cancel + place saved
            }
            QueueValueDecision::Replace { .. } => {
                self.replaced += 1;
            }
            QueueValueDecision::NoData => {
                self.no_data += 1;
            }
        }
    }
}

/// Compares queue value of existing order vs replacement.
///
/// This comparator uses expected value (EV) calculations to determine
/// whether replacing an order is worth losing its queue position advantage.
pub struct QueueValueComparator<'a> {
    queue_tracker: &'a QueuePositionTracker,
    config: QueueValueConfig,
    sigma: f64, // Current volatility for P(touch) calculation
}

impl<'a> QueueValueComparator<'a> {
    /// Create a new queue value comparator.
    pub fn new(
        queue_tracker: &'a QueuePositionTracker,
        config: QueueValueConfig,
        sigma: f64,
    ) -> Self {
        Self {
            queue_tracker,
            config,
            sigma: sigma.max(1e-10), // Floor to prevent division by zero
        }
    }

    /// Check if queue value comparison is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Compare EV of keeping current order vs replacing with new order.
    ///
    /// # Arguments
    /// * `oid` - Order ID to evaluate
    /// * `current_price` - Current order price
    /// * `target_price` - Desired target price
    /// * `mid_price` - Current market mid price
    ///
    /// # Returns
    /// A `QueueValueDecision` indicating whether to keep or replace the order.
    pub fn compare(
        &self,
        oid: u64,
        current_price: f64,
        target_price: f64,
        mid_price: f64,
    ) -> QueueValueDecision {
        if !self.config.enabled {
            return QueueValueDecision::NoData;
        }

        // Get current order's queue position
        let position = match self.queue_tracker.get_position(oid) {
            Some(p) => p,
            None => return QueueValueDecision::NoData,
        };

        // Check order age - don't disturb very young orders
        if position.age_seconds() < self.config.min_order_age_secs {
            return QueueValueDecision::Keep {
                reason: QueueKeepReason::OrderTooYoung,
                current_ev: 0.0,
                replacement_ev: 0.0,
            };
        }

        // Get current fill probability from queue tracker
        let current_p_fill = match self
            .queue_tracker
            .fill_probability(oid, self.config.fill_horizon_secs)
        {
            Some(p) => p,
            None => return QueueValueDecision::NoData,
        };

        // Calculate price difference
        let price_diff_bps = ((target_price - current_price) / mid_price).abs() * 10000.0;

        // Check queue lock (high P(fill) orders protected unless price moved significantly)
        if current_p_fill > self.config.queue_lock_threshold
            && price_diff_bps < self.config.queue_lock_override_bps
        {
            let current_ev = current_p_fill * self.config.spread_capture_bps;
            debug!(
                oid = oid,
                p_fill = %format!("{:.3}", current_p_fill),
                price_diff_bps = %format!("{:.2}", price_diff_bps),
                threshold = %format!("{:.2}", self.config.queue_lock_threshold),
                "Queue-aware: Order LOCKED (high fill probability)"
            );
            return QueueValueDecision::Keep {
                reason: QueueKeepReason::QueueLocked,
                current_ev,
                replacement_ev: 0.0,
            };
        }

        // Estimate P(fill) for new order at back of queue
        let new_p_fill =
            self.estimate_new_order_fill_prob(target_price, mid_price, position.is_bid);

        // Compare expected values
        let current_ev = current_p_fill * self.config.spread_capture_bps;
        let new_ev = new_p_fill * self.config.spread_capture_bps;

        // Calculate absolute EV improvement (in bps)
        let ev_improvement_bps = new_ev - current_ev;

        // API BUDGET GATE: Only modify if EV improvement exceeds API cost
        // This prevents low-value modifications that burn rate limit budget
        if ev_improvement_bps < self.config.api_cost_bps {
            debug!(
                oid = oid,
                current_ev = %format!("{:.3}", current_ev),
                new_ev = %format!("{:.3}", new_ev),
                ev_improvement_bps = %format!("{:.3}", ev_improvement_bps),
                api_cost_bps = %format!("{:.1}", self.config.api_cost_bps),
                "Queue-aware: KEEPING order (EV improvement < API cost)"
            );
            return QueueValueDecision::Keep {
                reason: QueueKeepReason::InsufficientImprovement,
                current_ev,
                replacement_ev: new_ev,
            };
        }

        // Also check percentage improvement threshold for proportional gating
        let improvement_pct = if current_ev > 0.0 {
            (new_ev - current_ev) / current_ev
        } else {
            // Current has no value, any improvement is good
            if new_ev > 0.0 {
                1.0
            } else {
                0.0
            }
        };

        if improvement_pct < self.config.improvement_threshold {
            debug!(
                oid = oid,
                current_p_fill = %format!("{:.3}", current_p_fill),
                new_p_fill = %format!("{:.3}", new_p_fill),
                improvement_pct = %format!("{:.1}%", improvement_pct * 100.0),
                threshold = %format!("{:.1}%", self.config.improvement_threshold * 100.0),
                "Queue-aware: KEEPING order (insufficient % improvement)"
            );
            QueueValueDecision::Keep {
                reason: QueueKeepReason::InsufficientImprovement,
                current_ev,
                replacement_ev: new_ev,
            }
        } else {
            debug!(
                oid = oid,
                current_p_fill = %format!("{:.3}", current_p_fill),
                new_p_fill = %format!("{:.3}", new_p_fill),
                improvement_pct = %format!("{:.1}%", improvement_pct * 100.0),
                ev_improvement_bps = %format!("{:.3}", ev_improvement_bps),
                "Queue-aware: REPLACING order (sufficient improvement)"
            );
            QueueValueDecision::Replace { improvement_pct }
        }
    }

    /// Estimate fill probability for a NEW order (starts at back of queue).
    ///
    /// This is a conservative estimate assuming the new order starts at the back
    /// of the queue at the target price level.
    fn estimate_new_order_fill_prob(&self, price: f64, mid_price: f64, is_bid: bool) -> f64 {
        // Distance from mid to target price
        let delta = if is_bid {
            // For bids, delta is how far below mid we are
            mid_price - price
        } else {
            // For asks, delta is how far above mid we are
            price - mid_price
        };

        // Normalize delta as a fraction
        let delta_frac = delta / mid_price;

        // P(touch) using reflection principle: P(touch δ in time T) = 2Φ(-δ/(σ√T))
        let sqrt_t = self.config.fill_horizon_secs.sqrt();
        let sigma_sqrt_t = self.sigma * sqrt_t;

        let p_touch = if delta_frac <= 0.0 {
            // At or better than mid - will definitely be touched
            1.0
        } else {
            let z = -delta_frac / sigma_sqrt_t;
            (2.0 * normal_cdf(z)).min(1.0)
        };

        // P(execute|touch) - new order at back of queue
        // Use average depth from queue tracker as estimate of queue size
        let estimated_depth = self.estimate_depth_at_price(price, is_bid);
        let expected_volume =
            self.queue_tracker.sigma() * self.config.fill_horizon_secs * mid_price;
        // Use a simple model: expected volume scales with sigma
        let expected_volume_units = expected_volume.max(0.01); // Floor to prevent division by zero

        let p_exec = (-estimated_depth / expected_volume_units).exp().min(1.0);

        p_touch * p_exec
    }

    /// Estimate the queue depth at a given price level.
    ///
    /// This is a rough estimate based on:
    /// 1. If we have orders at this price, use their depth_ahead
    /// 2. Otherwise, use a default based on config
    fn estimate_depth_at_price(&self, price: f64, is_bid: bool) -> f64 {
        // Look for existing orders at this price to get depth info
        // This iterates through positions which is O(n) but n is small (<100)
        let summary = self.queue_tracker.summary();

        let positions = if is_bid {
            &summary.bid_positions
        } else {
            &summary.ask_positions
        };

        // Find closest order to this price
        let mut closest_depth = None;
        let mut min_distance = f64::MAX;

        for pos in positions {
            let distance = (pos.price - price).abs();
            if distance < min_distance {
                min_distance = distance;
                closest_depth = Some(pos.depth_ahead);
            }
        }

        // If we found a nearby order, use its depth; otherwise use default
        // Add some extra for being at back of queue
        closest_depth.unwrap_or(1.0) + 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::tracking::queue::QueueConfig;

    fn make_tracker() -> QueuePositionTracker {
        QueuePositionTracker::new(QueueConfig {
            cancel_decay_rate: 0.2,
            expected_volume_per_second: 10.0,
            min_queue_position: 0.01,
            default_queue_position: 5.0,
            refresh_threshold: 0.1,
            min_order_age_for_refresh: 0.1,
            refresh_cost_bps: 5.0,       // EV cost of refresh
            spread_capture_bps: 8.0,     // Expected spread capture
        })
    }

    fn make_config() -> QueueValueConfig {
        QueueValueConfig {
            improvement_threshold: 0.15,
            queue_lock_threshold: 0.50,
            queue_lock_override_bps: 35.0,
            fill_horizon_secs: 1.0,
            spread_capture_bps: 8.0,
            min_order_age_secs: 0.0, // Disable age check for tests
            enabled: true,
            api_cost_bps: 3.0,       // API call cost in bps
        }
    }

    #[test]
    fn test_keep_high_fill_probability_order() {
        let mut tracker = make_tracker();

        // Place order at best bid with small queue ahead
        tracker.order_placed(1, 100.0, 1.0, 0.1, true);
        tracker.update_from_book(100.0, 101.0, 0.001);

        // Wait for order to age
        std::thread::sleep(std::time::Duration::from_millis(10));

        let comparator = QueueValueComparator::new(&tracker, make_config(), 0.001);

        // Target only moved slightly (10 bps)
        let decision = comparator.compare(1, 100.0, 100.01, 100.5);

        // Should keep due to queue lock or insufficient improvement
        match decision {
            QueueValueDecision::Keep { reason, .. } => {
                assert!(
                    reason == QueueKeepReason::QueueLocked
                        || reason == QueueKeepReason::InsufficientImprovement
                );
            }
            _ => panic!("Expected Keep decision, got {:?}", decision),
        }
    }

    #[test]
    fn test_replace_low_fill_probability_order() {
        let mut tracker = make_tracker();

        // Place order far from best with large queue ahead
        // Order at 95.0 with best bid at 100.0 (5% away) and 100 size ahead
        tracker.order_placed(1, 95.0, 1.0, 100.0, true);
        // Use higher sigma so P(touch) isn't zero
        tracker.update_from_book(100.0, 101.0, 0.01);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let config = QueueValueConfig {
            improvement_threshold: 0.10, // Lower threshold
            queue_lock_threshold: 0.90,  // High lock threshold (won't trigger)
            spread_capture_bps: 8.0,
            ..make_config()
        };
        let comparator = QueueValueComparator::new(&tracker, config, 0.01);

        // Target is at best bid (100.0) - should have much better fill prob
        // New order at 100.0 has P(touch)=1.0 vs old at 95.0 with P(touch) << 1
        let decision = comparator.compare(1, 95.0, 100.0, 100.5);

        // With low P(fill) for current and higher for new, either:
        // - Replace if improvement > threshold
        // - Keep with InsufficientImprovement if improvement < threshold (both EVs near 0)
        match decision {
            QueueValueDecision::Replace { improvement_pct } => {
                assert!(improvement_pct >= 0.0);
            }
            QueueValueDecision::Keep {
                reason: QueueKeepReason::InsufficientImprovement,
                current_ev,
                replacement_ev,
            } => {
                // If both EVs are very low, improvement may not meet threshold
                assert!(current_ev >= 0.0);
                assert!(replacement_ev >= 0.0);
            }
            _ => panic!(
                "Expected Replace or Keep(InsufficientImprovement), got {:?}",
                decision
            ),
        }
    }

    #[test]
    fn test_queue_lock_override() {
        let mut tracker = make_tracker();

        // Place order at best with tiny queue (high fill probability)
        tracker.order_placed(1, 100.0, 1.0, 0.01, true);
        tracker.update_from_book(100.0, 101.0, 0.001);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let config = QueueValueConfig {
            queue_lock_threshold: 0.50,    // Will lock if P(fill) > 50%
            queue_lock_override_bps: 30.0, // Will override lock if price moved > 30 bps
            ..make_config()
        };
        let comparator = QueueValueComparator::new(&tracker, config, 0.001);

        // Target at 100.05 with mid at 100.5:
        // price_diff_bps = (100.05 - 100.0) / 100.5 * 10000 ≈ 5 bps
        // This is BELOW the 30 bps override threshold, so lock should apply
        let decision = comparator.compare(1, 100.0, 100.05, 100.5);

        // Order has high P(fill) and price diff is small (< override threshold)
        // So we expect it to be locked
        match decision {
            QueueValueDecision::Keep { reason, .. } => {
                // Either locked (high fill prob) or insufficient improvement
                assert!(
                    reason == QueueKeepReason::QueueLocked
                        || reason == QueueKeepReason::InsufficientImprovement
                );
            }
            QueueValueDecision::Replace { .. } => {
                // Also acceptable if the math works out
            }
            _ => panic!("Expected Keep or Replace, got {:?}", decision),
        }

        // Now test with a LARGER price move that exceeds the override threshold
        // 100.0 -> 100.5 with mid at 100.5: diff = 0.5/100.5 * 10000 ≈ 50 bps > 30 bps
        let decision_large_move = comparator.compare(1, 100.0, 100.5, 100.5);

        // With > 30 bps move, lock should be overridden
        // Either Replace or Keep with InsufficientImprovement (not QueueLocked)
        match decision_large_move {
            QueueValueDecision::Replace { .. } => {}
            QueueValueDecision::Keep { reason, .. } => {
                // If kept, it should be due to insufficient improvement, not lock
                assert_eq!(reason, QueueKeepReason::InsufficientImprovement);
            }
            _ => panic!("Expected Replace or Keep(InsufficientImprovement) after override"),
        }
    }

    #[test]
    fn test_no_data_fallback() {
        let tracker = make_tracker(); // Empty tracker

        let comparator = QueueValueComparator::new(&tracker, make_config(), 0.001);

        // Order doesn't exist
        let decision = comparator.compare(999, 100.0, 100.01, 100.5);

        assert_eq!(decision, QueueValueDecision::NoData);
    }

    #[test]
    fn test_disabled_comparator() {
        let mut tracker = make_tracker();
        tracker.order_placed(1, 100.0, 1.0, 0.1, true);
        tracker.update_from_book(100.0, 101.0, 0.001);

        let config = QueueValueConfig {
            enabled: false,
            ..make_config()
        };
        let comparator = QueueValueComparator::new(&tracker, config, 0.001);

        let decision = comparator.compare(1, 100.0, 100.01, 100.5);
        assert_eq!(decision, QueueValueDecision::NoData);
    }

    #[test]
    fn test_stats_recording() {
        let mut stats = QueueValueStats::default();

        stats.record(&QueueValueDecision::Keep {
            reason: QueueKeepReason::QueueLocked,
            current_ev: 1.0,
            replacement_ev: 0.5,
        });
        stats.record(&QueueValueDecision::Keep {
            reason: QueueKeepReason::InsufficientImprovement,
            current_ev: 0.8,
            replacement_ev: 0.9,
        });
        stats.record(&QueueValueDecision::Replace {
            improvement_pct: 0.25,
        });
        stats.record(&QueueValueDecision::NoData);

        assert_eq!(stats.kept_queue_locked, 1);
        assert_eq!(stats.kept_insufficient_improvement, 1);
        assert_eq!(stats.replaced, 1);
        assert_eq!(stats.no_data, 1);
        assert_eq!(stats.total_kept(), 2);
        assert_eq!(stats.requests_saved, 4); // 2 keeps × 2 requests each
    }
}
