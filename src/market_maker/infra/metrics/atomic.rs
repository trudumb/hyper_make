//! Atomic f64 wrapper for thread-safe metrics.
//!
//! Uses AtomicU64 internally since AtomicF64 is not available in std.

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic f64 wrapper using AtomicU64.
pub(super) struct AtomicF64(AtomicU64);

impl AtomicF64 {
    pub(super) fn new(val: f64) -> Self {
        Self(AtomicU64::new(val.to_bits()))
    }

    pub(super) fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub(super) fn store(&self, val: f64) {
        self.0.store(val.to_bits(), Ordering::Relaxed);
    }

    pub(super) fn fetch_add(&self, val: f64) -> f64 {
        loop {
            let current = self.0.load(Ordering::Relaxed);
            let current_f64 = f64::from_bits(current);
            let new = (current_f64 + val).to_bits();
            if self
                .0
                .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return current_f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_f64() {
        let af = AtomicF64::new(1.5);
        assert_eq!(af.load(), 1.5);

        af.store(2.5);
        assert_eq!(af.load(), 2.5);

        let old = af.fetch_add(1.0);
        assert_eq!(old, 2.5);
        assert_eq!(af.load(), 3.5);
    }
}
