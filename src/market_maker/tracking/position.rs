//! Position tracking.
//!
//! Simple position state management. Fill deduplication is handled
//! centrally by `FillDeduplicator` in the fills module.

/// Tracks current position state.
#[derive(Debug)]
pub struct PositionTracker {
    /// Current position (positive = long, negative = short)
    position: f64,
    /// Total fills processed
    fill_count: usize,
}

impl PositionTracker {
    /// Create a new position tracker with an initial position.
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

    /// Set the position directly (used for initialization/sync).
    pub fn set_position(&mut self, position: f64) {
        self.position = position;
    }

    /// Process a fill and update position.
    ///
    /// Note: Caller is responsible for deduplication via FillDeduplicator.
    pub fn process_fill(&mut self, amount: f64, is_buy: bool) {
        if is_buy {
            self.position += amount;
        } else {
            self.position -= amount;
        }
        self.fill_count += 1;
    }

    /// Get the number of fills processed.
    pub fn fill_count(&self) -> usize {
        self.fill_count
    }
}

impl Default for PositionTracker {
    fn default() -> Self {
        Self::new(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_position_tracker() {
        let tracker = PositionTracker::new(1.5);
        assert!((tracker.position() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_process_buy_fill() {
        let mut tracker = PositionTracker::new(0.0);
        tracker.process_fill(0.5, true);
        assert!((tracker.position() - 0.5).abs() < f64::EPSILON);
        assert_eq!(tracker.fill_count(), 1);
    }

    #[test]
    fn test_process_sell_fill() {
        let mut tracker = PositionTracker::new(1.0);
        tracker.process_fill(0.3, false);
        assert!((tracker.position() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multiple_fills() {
        let mut tracker = PositionTracker::new(0.0);
        tracker.process_fill(1.0, true);
        tracker.process_fill(0.5, false);
        tracker.process_fill(0.2, true);
        assert!((tracker.position() - 0.7).abs() < f64::EPSILON);
        assert_eq!(tracker.fill_count(), 3);
    }
}
