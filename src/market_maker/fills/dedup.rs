//! Fill deduplication.
//!
//! Provides a single source of truth for fill deduplication,
//! replacing multiple HashSets scattered across modules.

use std::collections::{HashSet, VecDeque};

/// Single source of fill deduplication.
///
/// Uses a fixed-size HashSet with FIFO eviction to track processed trade IDs.
/// This replaces multiple deduplication mechanisms in PositionTracker,
/// AdverseSelectionEstimator, and DepthDecayAS.
#[derive(Debug)]
pub struct FillDeduplicator {
    /// Set of processed trade IDs
    processed_tids: HashSet<u64>,
    /// Queue for FIFO eviction when at capacity
    eviction_queue: VecDeque<u64>,
    /// Maximum number of TIDs to track
    max_tracked: usize,
}

impl FillDeduplicator {
    /// Create a new deduplicator with default capacity (10,000 fills).
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    /// Create a new deduplicator with specified capacity.
    pub fn with_capacity(max: usize) -> Self {
        Self {
            processed_tids: HashSet::with_capacity(max),
            eviction_queue: VecDeque::with_capacity(max),
            max_tracked: max,
        }
    }

    /// Check if a trade ID has already been processed.
    pub fn is_duplicate(&self, tid: u64) -> bool {
        self.processed_tids.contains(&tid)
    }

    /// Check if a trade ID is new (not yet processed).
    pub fn is_new(&self, tid: u64) -> bool {
        !self.is_duplicate(tid)
    }

    /// Mark a trade ID as processed.
    ///
    /// Returns `true` if this was a new TID, `false` if already processed.
    pub fn mark_processed(&mut self, tid: u64) -> bool {
        if self.processed_tids.contains(&tid) {
            return false;
        }

        // Evict oldest if at capacity
        if self.processed_tids.len() >= self.max_tracked {
            if let Some(old_tid) = self.eviction_queue.pop_front() {
                self.processed_tids.remove(&old_tid);
            }
        }

        self.processed_tids.insert(tid);
        self.eviction_queue.push_back(tid);
        true
    }

    /// Check and mark in one operation.
    ///
    /// Returns `true` if this was a new fill (and marks it as processed),
    /// `false` if it was a duplicate.
    pub fn check_and_mark(&mut self, tid: u64) -> bool {
        self.mark_processed(tid)
    }

    /// Get the number of tracked TIDs.
    pub fn len(&self) -> usize {
        self.processed_tids.len()
    }

    /// Check if no TIDs are being tracked.
    pub fn is_empty(&self) -> bool {
        self.processed_tids.is_empty()
    }

    /// Get the maximum capacity.
    pub fn capacity(&self) -> usize {
        self.max_tracked
    }

    /// Clear all tracked TIDs.
    pub fn clear(&mut self) {
        self.processed_tids.clear();
        self.eviction_queue.clear();
    }
}

impl Default for FillDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_fill() {
        let mut dedup = FillDeduplicator::new();
        assert!(dedup.is_new(1));
        assert!(dedup.mark_processed(1));
        assert!(dedup.is_duplicate(1));
        assert!(!dedup.is_new(1));
    }

    #[test]
    fn test_duplicate_rejected() {
        let mut dedup = FillDeduplicator::new();
        assert!(dedup.mark_processed(1));
        assert!(!dedup.mark_processed(1)); // Duplicate
        assert_eq!(dedup.len(), 1);
    }

    #[test]
    fn test_check_and_mark() {
        let mut dedup = FillDeduplicator::new();
        assert!(dedup.check_and_mark(1));  // New
        assert!(!dedup.check_and_mark(1)); // Duplicate
    }

    #[test]
    fn test_fifo_eviction() {
        let mut dedup = FillDeduplicator::with_capacity(3);

        // Fill to capacity
        assert!(dedup.mark_processed(1));
        assert!(dedup.mark_processed(2));
        assert!(dedup.mark_processed(3));
        assert_eq!(dedup.len(), 3);

        // Adding 4 should evict 1
        assert!(dedup.mark_processed(4));
        assert_eq!(dedup.len(), 3);
        assert!(dedup.is_new(1)); // 1 was evicted, so it's "new" again
        assert!(dedup.is_duplicate(2));
        assert!(dedup.is_duplicate(3));
        assert!(dedup.is_duplicate(4));
    }

    #[test]
    fn test_clear() {
        let mut dedup = FillDeduplicator::new();
        dedup.mark_processed(1);
        dedup.mark_processed(2);
        assert_eq!(dedup.len(), 2);

        dedup.clear();
        assert!(dedup.is_empty());
        assert!(dedup.is_new(1));
        assert!(dedup.is_new(2));
    }
}
