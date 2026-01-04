//! Priority value calculation for impulse control.
//!
//! Calculates the "virtual value" of a resting limit order based on its
//! queue priority. This treats queue position as an asset with real economic value.

use super::super::observer::PriorityClass;

/// Calculator for priority-based order valuation.
#[derive(Debug, Clone)]
pub struct PriorityValueCalculator {
    /// Multiplier for priority premium (0-1)
    pub priority_premium_multiplier: f64,

    /// High priority threshold (π < this = high)
    pub high_priority_threshold: f64,

    /// Medium priority threshold (π < this = medium)
    pub medium_priority_threshold: f64,
}

impl Default for PriorityValueCalculator {
    fn default() -> Self {
        Self {
            priority_premium_multiplier: 0.8,
            high_priority_threshold: 0.3,
            medium_priority_threshold: 0.7,
        }
    }
}

impl PriorityValueCalculator {
    /// Calculate the value of holding this queue position.
    ///
    /// ```text
    /// V_hold = (1 - π) × spread × size × multiplier
    /// ```
    ///
    /// This represents the expected "free" profit from a passive fill
    /// that would be lost if we cancel and re-place.
    pub fn hold_value(&self, pi: f64, spread: f64, size: f64) -> f64 {
        let priority_value = 1.0 - pi;
        priority_value * spread * size * self.priority_premium_multiplier
    }

    /// Calculate the value of moving the order (resetting queue position).
    ///
    /// ```text
    /// V_move = |drift| × size - move_cost
    /// ```
    ///
    /// Move is worthwhile if drift × size exceeds the hold value.
    pub fn move_value(&self, drift_abs: f64, size: f64, move_cost: f64) -> f64 {
        (drift_abs * size - move_cost).max(0.0)
    }

    /// Calculate the priority premium in basis points.
    ///
    /// ```text
    /// premium_bps = (1 - π) × spread_bps × multiplier
    /// ```
    pub fn priority_premium_bps(&self, pi: f64, spread_bps: f64) -> f64 {
        let priority_value = 1.0 - pi;
        priority_value * spread_bps * self.priority_premium_multiplier
    }

    /// Calculate the dynamic modify threshold.
    ///
    /// ```text
    /// threshold = base_threshold + priority_premium
    /// ```
    ///
    /// High priority orders have higher thresholds (more drift tolerance).
    pub fn modify_threshold_bps(
        &self,
        pi: f64,
        spread_bps: f64,
        base_threshold_bps: f64,
    ) -> f64 {
        let premium = self.priority_premium_bps(pi, spread_bps);
        base_threshold_bps + premium
    }

    /// Classify priority based on π.
    pub fn classify_priority(&self, pi: f64) -> PriorityClass {
        if pi < self.high_priority_threshold {
            PriorityClass::High
        } else if pi < self.medium_priority_threshold {
            PriorityClass::Medium
        } else {
            PriorityClass::Low
        }
    }

    /// Should we hold based on priority and drift?
    ///
    /// Hold if: priority_premium > drift
    pub fn should_hold(&self, pi: f64, spread_bps: f64, drift_bps: f64) -> bool {
        let premium = self.priority_premium_bps(pi, spread_bps);
        premium > drift_bps
    }

    /// Calculate "stickiness" - how resistant to changes this order should be.
    ///
    /// Stickiness = 1.0 for orders at front of queue, 0.0 for orders at back.
    pub fn stickiness(&self, pi: f64) -> f64 {
        (1.0 - pi).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hold_value() {
        let calc = PriorityValueCalculator::default();

        // Front of queue (π = 0)
        let value_front = calc.hold_value(0.0, 0.001, 1.0);
        // Back of queue (π = 1)
        let value_back = calc.hold_value(1.0, 0.001, 1.0);

        assert!(value_front > value_back);
        assert!((value_back - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_priority_premium_bps() {
        let calc = PriorityValueCalculator::default();

        // Front of queue
        let premium_front = calc.priority_premium_bps(0.0, 10.0);
        // Back of queue
        let premium_back = calc.priority_premium_bps(1.0, 10.0);

        assert!((premium_front - 8.0).abs() < 0.01); // 0.8 * 10 = 8
        assert!((premium_back - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_modify_threshold() {
        let calc = PriorityValueCalculator::default();
        let base = 2.0; // 2 bps base threshold

        // Front of queue should have higher threshold
        let threshold_front = calc.modify_threshold_bps(0.0, 10.0, base);
        // Back of queue should have lower threshold
        let threshold_back = calc.modify_threshold_bps(1.0, 10.0, base);

        assert!(threshold_front > threshold_back);
        assert!((threshold_back - base).abs() < 0.01);
    }

    #[test]
    fn test_should_hold() {
        let calc = PriorityValueCalculator::default();

        // High priority, small drift: should hold
        assert!(calc.should_hold(0.1, 10.0, 2.0));

        // Low priority, small drift: should not hold
        assert!(!calc.should_hold(0.9, 10.0, 2.0));

        // High priority, large drift: should not hold
        assert!(!calc.should_hold(0.1, 10.0, 15.0));
    }

    #[test]
    fn test_classify_priority() {
        let calc = PriorityValueCalculator::default();

        assert_eq!(calc.classify_priority(0.1), PriorityClass::High);
        assert_eq!(calc.classify_priority(0.5), PriorityClass::Medium);
        assert_eq!(calc.classify_priority(0.8), PriorityClass::Low);
    }
}
