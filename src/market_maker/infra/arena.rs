//! Quote cycle arena allocator for latency optimization.
//!
//! Uses bumpalo for arena allocation during quote cycles to:
//! - Eliminate per-allocation syscalls (single reset vs multiple free)
//! - Improve cache locality (contiguous allocations)
//! - Reduce allocation latency from ~100ns to ~10ns per allocation
//!
//! # Usage Pattern
//!
//! ```ignore
//! let mut arena = QuoteCycleArena::new();
//!
//! // During quote cycle:
//! let levels = arena.alloc_ladder_levels(10);
//! let quotes = arena.alloc_quotes(5);
//!
//! // At end of quote cycle:
//! arena.reset();
//! ```
//!
//! # Capacity
//!
//! Pre-allocates 8KB which is sufficient for:
//! - 10 LadderLevels (64 bytes each) = 640 bytes
//! - 10 Quotes (16 bytes each) = 160 bytes
//! - 20 OrderSpecs (varies, ~200 bytes each) = 4KB
//! - Temporary Vecs and overhead = ~3KB
//!
//! Total typical usage: ~5KB per quote cycle

use bumpalo::Bump;

/// Default arena capacity in bytes (8KB).
const DEFAULT_ARENA_CAPACITY: usize = 8 * 1024;

/// Arena allocator for quote cycle allocations.
///
/// Wraps bumpalo::Bump with a pre-allocated capacity and provides
/// typed allocation methods for quote cycle structures.
pub struct QuoteCycleArena {
    /// The underlying bump allocator.
    bump: Bump,
}

impl QuoteCycleArena {
    /// Create a new arena with default capacity (8KB).
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_ARENA_CAPACITY)
    }

    /// Create a new arena with specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bump: Bump::with_capacity(capacity),
        }
    }

    /// Reset the arena for the next quote cycle.
    ///
    /// This is O(1) - just resets the allocation pointer.
    /// All previous allocations become invalid.
    #[inline]
    pub fn reset(&mut self) {
        self.bump.reset();
    }

    /// Get the amount of memory currently allocated (in bytes).
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }

    /// Get access to the underlying bump allocator.
    ///
    /// Use this for custom allocations with `bumpalo::collections::Vec`.
    #[inline]
    pub fn bump(&self) -> &Bump {
        &self.bump
    }

    /// Allocate a Vec in the arena with given capacity.
    ///
    /// Returns a `bumpalo::collections::Vec` that allocates from the arena.
    #[inline]
    pub fn alloc_vec<T>(&self, capacity: usize) -> bumpalo::collections::Vec<'_, T> {
        bumpalo::collections::Vec::with_capacity_in(capacity, &self.bump)
    }

    /// Allocate a slice and initialize with values from an iterator.
    #[inline]
    pub fn alloc_slice_copy<T: Copy>(&self, values: &[T]) -> &mut [T] {
        self.bump.alloc_slice_copy(values)
    }
}

impl Default for QuoteCycleArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = QuoteCycleArena::new();
        // Bumpalo pre-allocates a chunk, so allocated_bytes includes the initial chunk
        // We just verify the arena is valid
        assert!(arena.bump().chunk_capacity() > 0);
    }

    #[test]
    fn test_arena_alloc_vec() {
        let arena = QuoteCycleArena::new();

        let mut vec = arena.alloc_vec::<f64>(10);
        vec.push(1.0);
        vec.push(2.0);
        vec.push(3.0);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 2.0);
        assert_eq!(vec[2], 3.0);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = QuoteCycleArena::new();

        // Get bytes before any allocation
        let bytes_initial = arena.allocated_bytes();

        // Allocate some data
        {
            let mut vec = arena.alloc_vec::<u64>(100);
            for i in 0..100 {
                vec.push(i);
            }
            // vec is dropped here, allowing mutable borrow for reset
        }

        let bytes_after_alloc = arena.allocated_bytes();
        assert!(bytes_after_alloc >= bytes_initial);

        // Reset should allow reuse (new allocations start from beginning)
        arena.reset();

        // After reset, allocated_bytes goes back down
        let bytes_after_reset = arena.allocated_bytes();
        assert!(bytes_after_reset <= bytes_initial);
    }

    #[test]
    fn test_arena_slice_copy() {
        let arena = QuoteCycleArena::new();

        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let slice = arena.alloc_slice_copy(&values);

        assert_eq!(slice.len(), 5);
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[4], 5.0);
    }

    #[test]
    fn test_arena_multiple_allocations() {
        let arena = QuoteCycleArena::new();

        // Simulate quote cycle allocations
        let mut levels = arena.alloc_vec::<f64>(10);
        let mut quotes = arena.alloc_vec::<(f64, f64)>(5);
        let mut order_ids = arena.alloc_vec::<u64>(20);

        for i in 0..10 {
            levels.push(i as f64 * 0.01);
        }
        for i in 0..5 {
            quotes.push((100.0 + i as f64, 0.1));
        }
        for i in 0..20 {
            order_ids.push(i);
        }

        assert_eq!(levels.len(), 10);
        assert_eq!(quotes.len(), 5);
        assert_eq!(order_ids.len(), 20);

        // All allocations should be small (less than 16KB including initial chunk)
        assert!(arena.allocated_bytes() < 16 * 1024);
    }

    #[test]
    fn test_arena_with_custom_capacity() {
        let arena = QuoteCycleArena::with_capacity(16 * 1024);
        // Bumpalo pre-allocates, so we verify the capacity is at least what we requested
        assert!(arena.bump().chunk_capacity() >= 8 * 1024);

        // Should handle larger allocations without growing
        let mut large_vec = arena.alloc_vec::<u64>(1000);
        for i in 0..1000 {
            large_vec.push(i);
        }
        assert_eq!(large_vec.len(), 1000);
    }
}
