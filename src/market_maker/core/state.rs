//! Core state bundle for MarketMaker.
//!
//! Groups frequently-accessed state fields.

use crate::market_maker::order_manager::OrderManager;
use crate::market_maker::position::PositionTracker;

/// Core trading state (position, orders, mid price).
///
/// This struct bundles the most frequently accessed mutable state.
#[derive(Debug)]
pub struct CoreState {
    /// Current position
    pub position: PositionTracker,
    /// Order manager
    pub orders: OrderManager,
    /// Latest mid price
    pub latest_mid: f64,
    /// Last warmup log progress (for throttling)
    pub last_warmup_log: usize,
}

impl CoreState {
    /// Create new core state.
    pub fn new(initial_position: f64) -> Self {
        Self {
            position: PositionTracker::new(initial_position),
            orders: OrderManager::new(),
            latest_mid: -1.0,
            last_warmup_log: 0,
        }
    }

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.position.position()
    }

    /// Check if mid price has been set.
    pub fn has_mid(&self) -> bool {
        self.latest_mid > 0.0
    }
}

impl Default for CoreState {
    fn default() -> Self {
        Self::new(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_state_new() {
        let state = CoreState::new(1.5);
        assert!((state.position() - 1.5).abs() < f64::EPSILON);
        assert!(!state.has_mid());
    }

    #[test]
    fn test_has_mid() {
        let mut state = CoreState::new(0.0);
        assert!(!state.has_mid());

        state.latest_mid = 50000.0;
        assert!(state.has_mid());
    }
}
