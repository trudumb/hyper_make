//! Position tracking consumer.
//!
//! Updates position based on fills. This is the highest priority consumer
//! as position must be correct before any other calculations.

use crate::market_maker::fills::{FillConsumer, FillEvent};

/// Consumer that updates position tracking.
///
/// This wraps the position value and provides atomic updates.
/// Position is tracked as a simple f64 since deduplication is handled
/// by the FillPipeline.
pub struct PositionConsumer {
    /// Current position (positive = long, negative = short)
    position: f64,
    /// Total fills processed
    fill_count: usize,
}

impl PositionConsumer {
    /// Create a new position consumer with initial position.
    pub fn new(initial_position: f64) -> Self {
        Self {
            position: initial_position,
            fill_count: 0,
        }
    }

    /// Get the current position.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Set the position directly (for sync/initialization).
    pub fn set_position(&mut self, position: f64) {
        self.position = position;
    }

    /// Get the number of fills processed.
    pub fn fill_count(&self) -> usize {
        self.fill_count
    }
}

impl Default for PositionConsumer {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl FillConsumer for PositionConsumer {
    fn on_fill(&mut self, fill: &FillEvent) -> Option<String> {
        // Update position based on fill direction
        self.position += fill.signed_size();
        self.fill_count += 1;
        None
    }

    fn name(&self) -> &'static str {
        "Position"
    }

    /// Position tracking is highest priority (must happen first)
    fn priority(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fill(tid: u64, size: f64, is_buy: bool) -> FillEvent {
        FillEvent::new(
            tid,
            100,
            size,
            50000.0,
            is_buy,
            50000.0,
            None,
            "BTC".to_string(),
        )
    }

    #[test]
    fn test_buy_increases_position() {
        let mut consumer = PositionConsumer::new(0.0);
        consumer.on_fill(&make_fill(1, 1.0, true));
        assert!((consumer.position() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sell_decreases_position() {
        let mut consumer = PositionConsumer::new(1.0);
        consumer.on_fill(&make_fill(1, 0.5, false));
        assert!((consumer.position() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fill_count() {
        let mut consumer = PositionConsumer::new(0.0);
        consumer.on_fill(&make_fill(1, 1.0, true));
        consumer.on_fill(&make_fill(2, 0.5, false));
        assert_eq!(consumer.fill_count(), 2);
    }

    #[test]
    fn test_priority_is_zero() {
        let consumer = PositionConsumer::new(0.0);
        assert_eq!(consumer.priority(), 0);
    }
}
