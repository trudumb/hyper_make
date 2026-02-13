//! Inventory Governor — enforces config.max_position as a HARD ceiling.
//!
//! This is the FIRST check in every quote cycle. Position safety is structural.
//! `config.max_position` is NEVER overridden by margin-derived limits.
//!
//! # Position Zones
//!
//! ```text
//! |--- Green (< 50%) ---|--- Yellow (50-80%) ---|--- Red (80-100%) ---|--- Kill (> 100%) ---|
//! | full two-sided       | bias toward reducing  | reduce-only         | cancel all          |
//! ```

use serde::{Deserialize, Serialize};

/// Position zone classification for the inventory governor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionZone {
    /// < 50% of max — full two-sided quoting
    Green,
    /// 50-80% — bias toward reducing, wider on increasing side
    Yellow,
    /// > 80% — reduce-only both sides
    Red,
    /// > 100% — cancel all, kill switch territory
    Kill,
}

/// Assessment result from the inventory governor.
#[derive(Debug, Clone)]
pub struct PositionAssessment {
    /// Classified zone
    pub zone: PositionZone,
    /// Maximum new exposure allowed on position-increasing side (0.0 in Red/Kill)
    pub max_new_exposure: f64,
    /// Current position ratio abs(position) / max_position
    pub position_ratio: f64,
    /// Current position (signed)
    pub position: f64,
    /// Is the position long? (for determining which side is reducing)
    pub is_long: bool,
    /// Spread multiplier for position-increasing side in Yellow zone
    pub increasing_side_spread_mult: f64,
}

/// Inventory governor — enforces config.max_position as HARD ceiling.
///
/// This is the FIRST check in every quote cycle. Position safety is structural.
/// `config.max_position` is NEVER overridden by margin-derived limits.
#[derive(Debug, Clone)]
pub struct InventoryGovernor {
    /// Hard position ceiling from user config
    max_position: f64,
}

impl InventoryGovernor {
    /// Create a new inventory governor with the given max position.
    ///
    /// # Panics
    /// Panics if `max_position <= 0.0`.
    pub fn new(max_position: f64) -> Self {
        assert!(
            max_position > 0.0,
            "InventoryGovernor: max_position must be > 0.0, got {max_position}"
        );
        Self { max_position }
    }

    /// Returns the hard position ceiling.
    pub fn max_position(&self) -> f64 {
        self.max_position
    }

    /// Classify current position into a zone and compute constraints.
    pub fn assess(&self, position: f64) -> PositionAssessment {
        let abs_pos = position.abs();
        let ratio = abs_pos / self.max_position;
        let is_long = position > 0.0;
        let remaining = (self.max_position - abs_pos).max(0.0);

        let (zone, max_new_exposure, spread_mult) = if ratio >= 1.0 {
            // Kill zone: position at or above max
            (PositionZone::Kill, 0.0, 3.0)
        } else if ratio >= 0.8 {
            // Red zone: reduce-only
            (PositionZone::Red, 0.0, 2.0)
        } else if ratio >= 0.5 {
            // Yellow zone: bias toward reducing, cap new exposure
            let capped_exposure = remaining * 0.5;
            // Linear ramp: 1.0 at ratio=0.5, 2.0 at ratio=0.8
            let mult = 1.0 + (ratio - 0.5) * (1.0 / 0.3);
            (PositionZone::Yellow, capped_exposure, mult)
        } else {
            // Green zone: full two-sided
            (PositionZone::Green, remaining, 1.0)
        };

        PositionAssessment {
            zone,
            max_new_exposure,
            position_ratio: ratio,
            position,
            is_long,
            increasing_side_spread_mult: spread_mult,
        }
    }

    /// Check if placing an order of `order_size` would push position past max.
    ///
    /// Returns `true` if the resulting |position| would exceed max_position.
    pub fn would_exceed(
        &self,
        current_position: f64,
        order_size: f64,
        is_buy: bool,
    ) -> bool {
        let new_position = if is_buy {
            current_position + order_size
        } else {
            current_position - order_size
        };
        new_position.abs() > self.max_position
    }

    /// Returns `true` if the order reduces |position|.
    pub fn is_reducing(&self, position: f64, is_buy: bool) -> bool {
        if position > 0.0 {
            // Long position — selling reduces
            !is_buy
        } else if position < 0.0 {
            // Short position — buying reduces
            is_buy
        } else {
            // Flat — nothing reduces
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_zone() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(0.0);
        assert_eq!(a.zone, PositionZone::Green);
        assert!((a.max_new_exposure - 10.0).abs() < 1e-9);
        assert!((a.position_ratio).abs() < 1e-9);
        assert!((a.increasing_side_spread_mult - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_green_zone_partial() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(3.0);
        assert_eq!(a.zone, PositionZone::Green);
        assert!((a.max_new_exposure - 7.0).abs() < 1e-9);
        assert!((a.position_ratio - 0.3).abs() < 1e-9);
        assert!(a.is_long);
    }

    #[test]
    fn test_yellow_zone_lower_bound() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(5.0);
        assert_eq!(a.zone, PositionZone::Yellow);
        // remaining = 5.0, capped = 2.5
        assert!((a.max_new_exposure - 2.5).abs() < 1e-9);
        // spread_mult at ratio=0.5 should be 1.0
        assert!((a.increasing_side_spread_mult - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_yellow_zone_upper_bound() {
        let gov = InventoryGovernor::new(10.0);
        // ratio = 0.79 — still yellow
        let a = gov.assess(7.9);
        assert_eq!(a.zone, PositionZone::Yellow);
        // remaining = 2.1, capped = 1.05
        assert!((a.max_new_exposure - 1.05).abs() < 1e-9);
        // spread_mult should be close to 2.0
        let expected_mult = 1.0 + (0.79 - 0.5) * (1.0 / 0.3);
        assert!((a.increasing_side_spread_mult - expected_mult).abs() < 0.01);
    }

    #[test]
    fn test_red_zone() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(8.5);
        assert_eq!(a.zone, PositionZone::Red);
        assert!((a.max_new_exposure).abs() < 1e-9);
        assert!((a.increasing_side_spread_mult - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_kill_zone() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(10.5);
        assert_eq!(a.zone, PositionZone::Kill);
        assert!((a.max_new_exposure).abs() < 1e-9);
        assert!((a.increasing_side_spread_mult - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_kill_zone_exact_boundary() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(10.0);
        assert_eq!(a.zone, PositionZone::Kill);
    }

    #[test]
    fn test_red_zone_exact_boundary() {
        let gov = InventoryGovernor::new(10.0);
        let a = gov.assess(8.0);
        assert_eq!(a.zone, PositionZone::Red);
    }

    #[test]
    fn test_negative_position_zones() {
        let gov = InventoryGovernor::new(10.0);

        // Short position, green zone
        let a = gov.assess(-3.0);
        assert_eq!(a.zone, PositionZone::Green);
        assert!(!a.is_long);
        assert!((a.max_new_exposure - 7.0).abs() < 1e-9);

        // Short position, yellow zone
        let a = gov.assess(-6.0);
        assert_eq!(a.zone, PositionZone::Yellow);
        assert!(!a.is_long);

        // Short position, red zone
        let a = gov.assess(-9.0);
        assert_eq!(a.zone, PositionZone::Red);

        // Short position, kill zone
        let a = gov.assess(-11.0);
        assert_eq!(a.zone, PositionZone::Kill);
    }

    #[test]
    fn test_would_exceed_long() {
        let gov = InventoryGovernor::new(10.0);
        // Current position 8.0, buying 3.0 -> 11.0 > 10.0
        assert!(gov.would_exceed(8.0, 3.0, true));
        // Current position 8.0, buying 1.0 -> 9.0 <= 10.0
        assert!(!gov.would_exceed(8.0, 1.0, true));
        // Current position 8.0, selling 3.0 -> 5.0 <= 10.0
        assert!(!gov.would_exceed(8.0, 3.0, false));
    }

    #[test]
    fn test_would_exceed_short() {
        let gov = InventoryGovernor::new(10.0);
        // Current position -8.0, selling 3.0 -> -11.0, abs = 11.0 > 10.0
        assert!(gov.would_exceed(-8.0, 3.0, false));
        // Current position -8.0, selling 1.0 -> -9.0, abs = 9.0 <= 10.0
        assert!(!gov.would_exceed(-8.0, 1.0, false));
        // Current position -8.0, buying 3.0 -> -5.0, abs = 5.0 <= 10.0
        assert!(!gov.would_exceed(-8.0, 3.0, true));
    }

    #[test]
    fn test_would_exceed_zero_position() {
        let gov = InventoryGovernor::new(10.0);
        // From flat, buying 11 would exceed
        assert!(gov.would_exceed(0.0, 11.0, true));
        // From flat, buying 10 would NOT exceed (10.0 == 10.0, not >)
        assert!(!gov.would_exceed(0.0, 10.0, true));
        // From flat, selling 11 would exceed
        assert!(gov.would_exceed(0.0, 11.0, false));
    }

    #[test]
    fn test_would_exceed_reducing_can_flip() {
        let gov = InventoryGovernor::new(10.0);
        // Long 8.0, selling 100 -> -92 -> abs 92 > 10 — exceeds by flipping past max
        assert!(gov.would_exceed(8.0, 100.0, false));
        // Long 8.0, selling 5.0 -> 3.0, abs = 3.0 <= 10.0 — safe reduce
        assert!(!gov.would_exceed(8.0, 5.0, false));
        // Long 8.0, selling 18.0 -> -10.0, abs = 10.0 <= 10.0 — exact boundary ok
        assert!(!gov.would_exceed(8.0, 18.0, false));
        // Short -8.0, buying 100 -> 92 > 10 — exceeds by flipping
        assert!(gov.would_exceed(-8.0, 100.0, true));
        // Short -8.0, buying 5.0 -> -3.0, abs = 3.0 <= 10.0
        assert!(!gov.would_exceed(-8.0, 5.0, true));
    }

    #[test]
    fn test_is_reducing_long() {
        let gov = InventoryGovernor::new(10.0);
        assert!(gov.is_reducing(5.0, false)); // selling reduces long
        assert!(!gov.is_reducing(5.0, true)); // buying increases long
    }

    #[test]
    fn test_is_reducing_short() {
        let gov = InventoryGovernor::new(10.0);
        assert!(gov.is_reducing(-5.0, true)); // buying reduces short
        assert!(!gov.is_reducing(-5.0, false)); // selling increases short
    }

    #[test]
    fn test_is_reducing_flat() {
        let gov = InventoryGovernor::new(10.0);
        assert!(!gov.is_reducing(0.0, true)); // flat, nothing reduces
        assert!(!gov.is_reducing(0.0, false));
    }

    #[test]
    #[should_panic(expected = "max_position must be > 0.0")]
    fn test_new_zero_panics() {
        InventoryGovernor::new(0.0);
    }

    #[test]
    #[should_panic(expected = "max_position must be > 0.0")]
    fn test_new_negative_panics() {
        InventoryGovernor::new(-1.0);
    }

    #[test]
    fn test_spread_mult_linear_ramp() {
        let gov = InventoryGovernor::new(100.0);
        // At ratio=0.5: mult should be 1.0
        let a = gov.assess(50.0);
        assert!((a.increasing_side_spread_mult - 1.0).abs() < 0.01);

        // At ratio=0.65: mult should be 1.5
        let a = gov.assess(65.0);
        assert!((a.increasing_side_spread_mult - 1.5).abs() < 0.01);

        // Just below 0.8: mult should approach 2.0
        let a = gov.assess(79.9);
        assert!((a.increasing_side_spread_mult - 2.0).abs() < 0.05);
    }
}
