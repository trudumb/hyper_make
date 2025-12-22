//! Position tracking with fill deduplication.

use std::collections::HashSet;

/// Tracks current position and deduplicates fills by trade ID.
#[derive(Debug)]
pub struct PositionTracker {
    /// Current position (positive = long, negative = short)
    position: f64,
    /// Set of processed trade IDs to avoid double-counting
    processed_tids: HashSet<u64>,
}

impl PositionTracker {
    /// Create a new position tracker with an initial position.
    pub fn new(initial_position: f64) -> Self {
        Self {
            position: initial_position,
            processed_tids: HashSet::new(),
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
    /// Returns `true` if this is a new fill, `false` if duplicate.
    pub fn process_fill(&mut self, tid: u64, amount: f64, is_buy: bool) -> bool {
        if self.processed_tids.contains(&tid) {
            return false; // Duplicate fill
        }
        self.processed_tids.insert(tid);

        if is_buy {
            self.position += amount;
        } else {
            self.position -= amount;
        }
        true
    }

    /// Check if a trade ID has already been processed.
    pub fn is_processed(&self, tid: u64) -> bool {
        self.processed_tids.contains(&tid)
    }

    /// Get the number of processed fills.
    pub fn processed_count(&self) -> usize {
        self.processed_tids.len()
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
        let result = tracker.process_fill(1, 0.5, true);
        assert!(result);
        assert!((tracker.position() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_process_sell_fill() {
        let mut tracker = PositionTracker::new(1.0);
        let result = tracker.process_fill(1, 0.3, false);
        assert!(result);
        assert!((tracker.position() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_duplicate_fill_rejected() {
        let mut tracker = PositionTracker::new(0.0);
        assert!(tracker.process_fill(1, 0.5, true));
        assert!(!tracker.process_fill(1, 0.5, true)); // Duplicate
        assert!((tracker.position() - 0.5).abs() < f64::EPSILON); // Position unchanged
    }
}
