//! Impulse filter for statistical order update control.
//!
//! Implements Δλ (delta-lambda) filtering to reduce API churn:
//! - Only update orders when improvement in fill probability exceeds threshold
//! - Lock high-P(fill) orders to protect queue position
//! - Override lock when price has moved significantly
//!
//! The core formula: `Δλ = (λ_new - λ_current) / λ_current`
//! where λ is the fill probability (lambda).

use serde::{Deserialize, Serialize};

use crate::market_maker::tracking::queue::QueuePositionTracker;

/// Configuration for the impulse filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseFilterConfig {
    /// Minimum relative improvement in fill probability to justify an update.
    /// E.g., 0.10 means 10% improvement required.
    pub improvement_threshold: f64,

    /// Time horizon in seconds for fill probability calculation.
    pub fill_horizon_seconds: f64,

    /// P(fill) threshold above which orders are "locked" (protected from updates).
    /// E.g., 0.30 means orders with >30% fill probability are locked.
    pub queue_lock_threshold: f64,

    /// Price movement in bps that overrides queue lock.
    /// If price moved more than this, update even if order is locked.
    pub queue_lock_override_bps: f64,
}

impl Default for ImpulseFilterConfig {
    fn default() -> Self {
        Self {
            improvement_threshold: 0.10, // 10% Δλ required (user preference: responsive)
            fill_horizon_seconds: 1.0,   // 1-second fill horizon
            queue_lock_threshold: 0.30,  // Lock orders with >30% P(fill)
            queue_lock_override_bps: 25.0, // Override lock if price moved >25bps
        }
    }
}

/// Decision from the impulse filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpulseDecision {
    /// Update the order (Δλ exceeds threshold or other criteria met).
    Update,
    /// Skip the update (Δλ too small, not worth the API cost).
    Skip,
    /// Order is locked (high P(fill), don't disturb queue position).
    Locked,
}

/// Statistics from impulse filter evaluation.
#[derive(Debug, Clone, Default)]
pub struct ImpulseFilterStats {
    /// Number of orders evaluated.
    pub evaluated_count: usize,
    /// Number of updates allowed.
    pub update_count: usize,
    /// Number of updates skipped (Δλ too small).
    pub skip_count: usize,
    /// Number of orders locked (high P(fill)).
    pub locked_count: usize,
}

/// Impulse filter for gating order updates based on fill probability improvement.
#[derive(Debug)]
pub struct ImpulseFilter {
    config: ImpulseFilterConfig,
    stats: ImpulseFilterStats,
}

impl ImpulseFilter {
    /// Create a new impulse filter with the given configuration.
    pub fn new(config: ImpulseFilterConfig) -> Self {
        Self {
            config,
            stats: ImpulseFilterStats::default(),
        }
    }

    /// Evaluate whether an order update is worthwhile.
    ///
    /// # Arguments
    /// * `queue_tracker` - Queue position tracker for P(fill) calculation
    /// * `oid` - Order ID of the existing order
    /// * `current_price` - Current order price
    /// * `new_price` - Proposed new price
    /// * `price_diff_bps` - Absolute price difference in basis points
    /// * `mid_price` - Current mid price (for new order P(fill) estimation)
    /// * `is_bid` - Whether this is a bid order
    ///
    /// # Returns
    /// `ImpulseDecision` indicating whether to Update, Skip, or consider the order Locked.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &mut self,
        queue_tracker: &QueuePositionTracker,
        oid: u64,
        _current_price: f64,
        new_price: f64,
        price_diff_bps: f64,
        mid_price: f64,
        is_bid: bool,
    ) -> ImpulseDecision {
        self.stats.evaluated_count += 1;

        let horizon = self.config.fill_horizon_seconds;

        // Step 1: Get current P(fill) from queue tracker
        let p_fill_current = match queue_tracker.fill_probability(oid, horizon) {
            Some(p) => p,
            None => {
                // Order not in queue tracker - allow update (new order)
                self.stats.update_count += 1;
                return ImpulseDecision::Update;
            }
        };

        // Step 2: Check queue lock condition
        if p_fill_current > self.config.queue_lock_threshold {
            // Order has high P(fill), check if override applies
            if price_diff_bps < self.config.queue_lock_override_bps {
                // Price hasn't moved enough to override lock
                self.stats.locked_count += 1;
                return ImpulseDecision::Locked;
            }
            // Price moved significantly, override lock and allow update
        }

        // Step 3: Estimate P(fill) for new order at back of queue
        let p_fill_new = self.estimate_new_fill_probability(
            queue_tracker,
            new_price,
            mid_price,
            is_bid,
            horizon,
        );

        // Step 4: Calculate Δλ = (λ_new - λ_current) / λ_current
        let delta_lambda = if p_fill_current > 1e-10 {
            (p_fill_new - p_fill_current) / p_fill_current
        } else {
            // If current P(fill) is essentially zero, any improvement is significant
            if p_fill_new > 1e-10 {
                1.0 // Treat as 100% improvement
            } else {
                0.0 // Both zero, no improvement
            }
        };

        // Step 5: Decision based on Δλ threshold
        if delta_lambda > self.config.improvement_threshold {
            self.stats.update_count += 1;
            ImpulseDecision::Update
        } else {
            self.stats.skip_count += 1;
            ImpulseDecision::Skip
        }
    }

    /// Estimate fill probability for a new order at the given price.
    ///
    /// Since a new order starts at the back of the queue, we estimate based on:
    /// - Distance from mid price (P(touch))
    /// - Queue position = 100% (back of queue)
    fn estimate_new_fill_probability(
        &self,
        queue_tracker: &QueuePositionTracker,
        new_price: f64,
        mid_price: f64,
        is_bid: bool,
        horizon_seconds: f64,
    ) -> f64 {
        // Use the queue tracker's methods to estimate P(touch)
        // A new order at the back of queue has:
        // - P(touch) based on distance from mid
        // - P(execute|touch) ≈ 0 for back of queue (very conservative)

        // Conservative estimate: new order has lower priority
        // P(fill) ≈ P(touch) × 0.1 (10% of queue executes)
        // This is a simplified model - in practice we'd want more sophisticated estimation

        let distance_from_mid = if is_bid {
            mid_price - new_price
        } else {
            new_price - mid_price
        };

        // If new price is on the wrong side of mid, P(fill) = 0
        if distance_from_mid < 0.0 {
            return 0.0;
        }

        // Get sigma from queue tracker for P(touch) calculation
        // P(touch) ≈ 2 * N(-d / (σ * √T)) where d is distance
        // Simplified: use exponential decay based on distance/sigma

        let sigma = queue_tracker.sigma();
        if sigma < 1e-10 {
            return 0.0;
        }

        let volatility_adjusted_distance = distance_from_mid / (sigma * horizon_seconds.sqrt());

        // Approximate P(touch) using error function approximation
        // P(touch) ≈ 2 * Φ(-d/σ√T)
        let p_touch = Self::approximate_p_touch(volatility_adjusted_distance);

        // New order at back of queue: assume 10% execution probability if touched
        // This is conservative - actual value depends on queue depth
        let p_execute_given_touch = 0.10;

        (p_touch * p_execute_given_touch).min(1.0)
    }

    /// Approximate P(touch) using fast approximation.
    ///
    /// Uses the complementary error function approximation:
    /// Φ(-x) ≈ exp(-x²/2) / (x * √(2π)) for x > 0
    fn approximate_p_touch(normalized_distance: f64) -> f64 {
        if normalized_distance <= 0.0 {
            return 1.0; // At or beyond mid, P(touch) = 1
        }

        if normalized_distance > 5.0 {
            return 0.0; // Very far from mid, P(touch) ≈ 0
        }

        // Fast approximation using exponential
        // 2 * Φ(-x) ≈ erfc(x/√2) ≈ exp(-x²/2) for moderate x
        let x_squared = normalized_distance * normalized_distance;
        (2.0 * (-x_squared / 2.0).exp()).min(1.0)
    }

    /// Get current filter statistics.
    pub fn stats(&self) -> &ImpulseFilterStats {
        &self.stats
    }

    /// Reset statistics (for new quote cycle).
    pub fn reset_stats(&mut self) {
        self.stats = ImpulseFilterStats::default();
    }

    /// Get the configuration.
    pub fn config(&self) -> &ImpulseFilterConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_filter() -> ImpulseFilter {
        ImpulseFilter::new(ImpulseFilterConfig::default())
    }

    #[test]
    fn test_config_defaults() {
        let config = ImpulseFilterConfig::default();
        assert!((config.improvement_threshold - 0.10).abs() < f64::EPSILON);
        assert!((config.fill_horizon_seconds - 1.0).abs() < f64::EPSILON);
        assert!((config.queue_lock_threshold - 0.30).abs() < f64::EPSILON);
        assert!((config.queue_lock_override_bps - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_impulse_decision_eq() {
        assert_eq!(ImpulseDecision::Update, ImpulseDecision::Update);
        assert_eq!(ImpulseDecision::Skip, ImpulseDecision::Skip);
        assert_eq!(ImpulseDecision::Locked, ImpulseDecision::Locked);
        assert_ne!(ImpulseDecision::Update, ImpulseDecision::Skip);
    }

    #[test]
    fn test_p_touch_approximation() {
        // At mid (distance = 0), P(touch) = 1
        assert!((ImpulseFilter::approximate_p_touch(0.0) - 1.0).abs() < 0.01);

        // Far from mid, P(touch) → 0
        assert!(ImpulseFilter::approximate_p_touch(5.0) < 0.01);

        // Moderate distance (use 2.0 to ensure p < 1.0)
        // For x=2.0: exp(-2.0) ≈ 0.135, so 2 * 0.135 = 0.27
        let p = ImpulseFilter::approximate_p_touch(2.0);
        assert!(p > 0.0 && p < 1.0, "p = {} should be in (0, 1)", p);

        // Small distance gets clamped to 1.0
        let p_small = ImpulseFilter::approximate_p_touch(0.5);
        assert!(p_small > 0.0 && p_small <= 1.0);
    }

    #[test]
    fn test_stats_tracking() {
        let filter = default_filter();
        assert_eq!(filter.stats().evaluated_count, 0);

        // After evaluation, stats should be updated
        // (We can't test evaluate() directly without QueuePositionTracker mock)
    }

    #[test]
    fn test_reset_stats() {
        let mut filter = default_filter();
        filter.stats.evaluated_count = 10;
        filter.stats.update_count = 5;
        filter.stats.skip_count = 3;
        filter.stats.locked_count = 2;

        filter.reset_stats();

        assert_eq!(filter.stats().evaluated_count, 0);
        assert_eq!(filter.stats().update_count, 0);
        assert_eq!(filter.stats().skip_count, 0);
        assert_eq!(filter.stats().locked_count, 0);
    }
}
