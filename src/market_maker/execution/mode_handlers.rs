//! Per-mode quote generation handlers.
//!
//! Each execution mode maps to a specific quoting behavior:
//! - `handle_flat()`: Cancel all resting orders
//! - `handle_maker()`: Generate GLFT ladder, mask disabled sides
//! - `handle_inventory_reduce()`: Aggressive limit at touch, cancel increasing side

use super::state_machine::ExecutionMode;
use crate::market_maker::quoting::Ladder;

/// Apply execution mode to a computed ladder.
///
/// This is a post-processing step: the GLFT ladder is computed normally,
/// then this function masks/modifies it based on the execution mode.
///
/// Returns `None` if mode is `Flat` (cancel all orders).
pub fn apply_mode_to_ladder(mode: &ExecutionMode, ladder: Ladder) -> Option<Ladder> {
    match mode {
        ExecutionMode::Flat => None,

        ExecutionMode::Maker { bid, ask } => {
            let mut filtered = ladder;
            if !bid {
                filtered.bids.clear();
            }
            if !ask {
                filtered.asks.clear();
            }
            Some(filtered)
        }

        ExecutionMode::InventoryReduce { urgency } => {
            let mut filtered = ladder;
            // Keep only the closest level on each side, tightened by urgency
            if !filtered.bids.is_empty() {
                filtered.bids.truncate(1);
                // Higher urgency → smaller size reduction (we WANT fills on reducing side)
                if let Some(level) = filtered.bids.first_mut() {
                    level.size *= 0.5 + 0.5 * urgency;
                }
            }
            if !filtered.asks.is_empty() {
                filtered.asks.truncate(1);
                if let Some(level) = filtered.asks.first_mut() {
                    level.size *= 0.5 + 0.5 * urgency;
                }
            }
            Some(filtered)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::quoting::{Ladder, LadderLevel};
    use smallvec::smallvec;

    fn sample_ladder() -> Ladder {
        Ladder {
            bids: smallvec![
                LadderLevel {
                    price: 100.0,
                    size: 1.0,
                    depth_bps: 5.0,
                },
                LadderLevel {
                    price: 99.9,
                    size: 0.5,
                    depth_bps: 10.0,
                },
            ],
            asks: smallvec![
                LadderLevel {
                    price: 100.1,
                    size: 1.0,
                    depth_bps: 5.0,
                },
                LadderLevel {
                    price: 100.2,
                    size: 0.5,
                    depth_bps: 10.0,
                },
            ],
        }
    }

    #[test]
    fn test_flat_returns_none() {
        let result = apply_mode_to_ladder(&ExecutionMode::Flat, sample_ladder());
        assert!(result.is_none());
    }

    #[test]
    fn test_maker_both_sides() {
        let mode = ExecutionMode::Maker {
            bid: true,
            ask: true,
        };
        let result = apply_mode_to_ladder(&mode, sample_ladder()).unwrap();
        assert_eq!(result.bids.len(), 2);
        assert_eq!(result.asks.len(), 2);
    }

    #[test]
    fn test_maker_bid_only() {
        let mode = ExecutionMode::Maker {
            bid: true,
            ask: false,
        };
        let result = apply_mode_to_ladder(&mode, sample_ladder()).unwrap();
        assert_eq!(result.bids.len(), 2);
        assert!(result.asks.is_empty());
    }

    #[test]
    fn test_maker_ask_only() {
        let mode = ExecutionMode::Maker {
            bid: false,
            ask: true,
        };
        let result = apply_mode_to_ladder(&mode, sample_ladder()).unwrap();
        assert!(result.bids.is_empty());
        assert_eq!(result.asks.len(), 2);
    }

    #[test]
    fn test_inventory_reduce_truncates_to_one_level() {
        let mode = ExecutionMode::InventoryReduce { urgency: 1.0 };
        let result = apply_mode_to_ladder(&mode, sample_ladder()).unwrap();
        assert_eq!(result.bids.len(), 1);
        assert_eq!(result.asks.len(), 1);
    }

    #[test]
    fn test_inventory_reduce_high_urgency_keeps_size() {
        let mode = ExecutionMode::InventoryReduce { urgency: 1.0 };
        let result = apply_mode_to_ladder(&mode, sample_ladder()).unwrap();
        // urgency 1.0: size * (0.5 + 0.5 * 1.0) = size * 1.0
        assert!((result.bids[0].size - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inventory_reduce_low_urgency_reduces_size() {
        let mode = ExecutionMode::InventoryReduce { urgency: 0.0 };
        let result = apply_mode_to_ladder(&mode, sample_ladder()).unwrap();
        // urgency 0.0: size * (0.5 + 0.5 * 0.0) = size * 0.5
        assert!((result.bids[0].size - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_empty_ladder_maker() {
        let mode = ExecutionMode::Maker {
            bid: true,
            ask: true,
        };
        let empty = Ladder::default();
        let result = apply_mode_to_ladder(&mode, empty).unwrap();
        assert!(result.bids.is_empty());
        assert!(result.asks.is_empty());
    }
}
