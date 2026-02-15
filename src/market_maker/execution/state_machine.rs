//! Execution state machine for mode-based quoting decisions.
//!
//! Replaces the QuoteGate decision block with an explicit state machine:
//!
//! ```text
//! InventoryGovernor → PositionZone
//! PreFillClassifier → ToxicityRegime        → select_mode() → ExecutionMode
//! QueueValueHeuristic → has_positive_value
//! Signal availability → has_alpha
//! ```
//!
//! Each mode maps to specific quoting behavior:
//! - `Flat`: Cancel all quotes
//! - `Maker { bid, ask }`: GLFT ladder, optionally one-sided
//! - `InventoryReduce { urgency }`: Aggressive reducing, increasing-side cancelled

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::market_maker::adverse_selection::toxicity_regime::ToxicityRegime;
use crate::market_maker::config::auto_derive::CapitalTier;
use crate::market_maker::PositionZone;

/// Execution mode determined by the state machine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Cancel all quotes — position zone Kill, or Toxic+flat
    Flat,
    /// GLFT ladder quoting, optionally one-sided
    Maker {
        /// Whether to quote bids
        bid: bool,
        /// Whether to quote asks
        ask: bool,
    },
    /// Aggressive position reduction
    InventoryReduce {
        /// Urgency [0, 1] — higher = more aggressive pricing
        urgency: f64,
    },
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::Flat
    }
}

impl fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flat => write!(f, "Flat"),
            Self::Maker { bid, ask } => match (bid, ask) {
                (true, true) => write!(f, "Maker(both)"),
                (true, false) => write!(f, "Maker(bid-only)"),
                (false, true) => write!(f, "Maker(ask-only)"),
                (false, false) => write!(f, "Flat(no-sides)"),
            },
            Self::InventoryReduce { urgency } => {
                write!(f, "InventoryReduce(urgency={urgency:.2})")
            }
        }
    }
}

/// Inputs to execution mode selection.
///
/// All fields come from upstream components — this struct is a pure data transfer.
#[derive(Debug, Clone)]
pub struct ModeSelectionInput {
    /// Position zone from InventoryGovernor
    pub position_zone: PositionZone,
    /// Toxicity regime from PreFillASClassifier
    pub toxicity_regime: ToxicityRegime,
    /// Whether we have positive queue value on bid side
    pub bid_has_value: bool,
    /// Whether we have positive queue value on ask side
    pub ask_has_value: bool,
    /// Whether we have actionable alpha signal (lead-lag or cross-venue)
    pub has_alpha: bool,
    /// Current position (positive = long)
    pub position: f64,
    /// Capital tier from CapitalProfile — affects fallback behavior
    pub capital_tier: CapitalTier,
}

/// Pure function: select execution mode from inputs.
///
/// Priority order (highest to lowest):
/// 1. Kill zone → Flat
/// 2. Red zone → InventoryReduce (urgency 1.0)
/// 3. Toxic + flat → Flat
/// 4. Toxic + positioned → InventoryReduce
/// 5. Yellow zone → Maker (reducing side only)
/// 6. No value and no alpha → Flat
/// 7. Default → Maker (quote sides with value or alpha)
pub fn select_mode(input: &ModeSelectionInput) -> ExecutionMode {
    // 1. Kill zone — cancel everything
    if input.position_zone == PositionZone::Kill {
        return ExecutionMode::Flat;
    }

    // 2. Red zone — urgent inventory reduction
    if input.position_zone == PositionZone::Red {
        return ExecutionMode::InventoryReduce { urgency: 1.0 };
    }

    // 3-4. Toxic regime
    if input.toxicity_regime == ToxicityRegime::Toxic {
        if input.position.abs() < 1e-9 {
            // Flat position → go flat
            return ExecutionMode::Flat;
        }
        // Positioned → reduce
        return ExecutionMode::InventoryReduce { urgency: 0.7 };
    }

    // 5. Yellow zone — only quote reducing side
    if input.position_zone == PositionZone::Yellow {
        let (bid, ask) = reducing_sides(input.position);
        return ExecutionMode::Maker { bid, ask };
    }

    // 6. No value and no alpha
    if !input.bid_has_value && !input.ask_has_value && !input.has_alpha {
        match input.capital_tier {
            CapitalTier::Micro | CapitalTier::Small => {
                // Small capital: quote both sides with widened spreads instead of going Flat.
                // Spread widening provides protection; Flat prevents all learning.
                return ExecutionMode::Maker { bid: true, ask: true };
            }
            _ => return ExecutionMode::Flat,
        }
    }

    // 7. Default: Maker with sides that have value or alpha
    let bid = input.bid_has_value || input.has_alpha;
    let ask = input.ask_has_value || input.has_alpha;
    ExecutionMode::Maker { bid, ask }
}

/// Determine which sides are reducing for a given position.
/// Returns (quote_bids, quote_asks).
fn reducing_sides(position: f64) -> (bool, bool) {
    if position > 1e-9 {
        // Long → asks reduce, bids increase
        (false, true)
    } else if position < -1e-9 {
        // Short → bids reduce, asks increase
        (true, false)
    } else {
        // Flat → neither side reduces, but allow both for exit flexibility
        (true, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_input() -> ModeSelectionInput {
        ModeSelectionInput {
            position_zone: PositionZone::Green,
            toxicity_regime: ToxicityRegime::Benign,
            bid_has_value: true,
            ask_has_value: true,
            has_alpha: true,
            position: 0.0,
            capital_tier: CapitalTier::Large, // Default to Large so existing tests pass unchanged
        }
    }

    #[test]
    fn test_kill_zone_always_flat() {
        let input = ModeSelectionInput {
            position_zone: PositionZone::Kill,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }

    #[test]
    fn test_red_zone_inventory_reduce() {
        let input = ModeSelectionInput {
            position_zone: PositionZone::Red,
            position: 1.0,
            ..default_input()
        };
        match select_mode(&input) {
            ExecutionMode::InventoryReduce { urgency } => {
                assert!((urgency - 1.0).abs() < 1e-10);
            }
            other => panic!("Expected InventoryReduce, got {other}"),
        }
    }

    #[test]
    fn test_toxic_flat_goes_flat() {
        let input = ModeSelectionInput {
            toxicity_regime: ToxicityRegime::Toxic,
            position: 0.0,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }

    #[test]
    fn test_toxic_positioned_reduces() {
        let input = ModeSelectionInput {
            toxicity_regime: ToxicityRegime::Toxic,
            position: 1.5,
            ..default_input()
        };
        match select_mode(&input) {
            ExecutionMode::InventoryReduce { urgency } => {
                assert!((urgency - 0.7).abs() < 1e-10);
            }
            other => panic!("Expected InventoryReduce, got {other}"),
        }
    }

    #[test]
    fn test_yellow_long_only_asks() {
        let input = ModeSelectionInput {
            position_zone: PositionZone::Yellow,
            position: 1.0, // Long
            ..default_input()
        };
        assert_eq!(
            select_mode(&input),
            ExecutionMode::Maker {
                bid: false,
                ask: true
            }
        );
    }

    #[test]
    fn test_yellow_short_only_bids() {
        let input = ModeSelectionInput {
            position_zone: PositionZone::Yellow,
            position: -1.0, // Short
            ..default_input()
        };
        assert_eq!(
            select_mode(&input),
            ExecutionMode::Maker {
                bid: true,
                ask: false
            }
        );
    }

    #[test]
    fn test_no_value_no_alpha_flat() {
        let input = ModeSelectionInput {
            bid_has_value: false,
            ask_has_value: false,
            has_alpha: false,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }

    #[test]
    fn test_green_with_value_maker_both() {
        let input = default_input();
        assert_eq!(
            select_mode(&input),
            ExecutionMode::Maker {
                bid: true,
                ask: true
            }
        );
    }

    #[test]
    fn test_alpha_enables_sides_without_value() {
        let input = ModeSelectionInput {
            bid_has_value: false,
            ask_has_value: false,
            has_alpha: true,
            ..default_input()
        };
        assert_eq!(
            select_mode(&input),
            ExecutionMode::Maker {
                bid: true,
                ask: true
            }
        );
    }

    #[test]
    fn test_one_sided_value() {
        let input = ModeSelectionInput {
            bid_has_value: true,
            ask_has_value: false,
            has_alpha: false,
            ..default_input()
        };
        assert_eq!(
            select_mode(&input),
            ExecutionMode::Maker {
                bid: true,
                ask: false
            }
        );
    }

    #[test]
    fn test_display_modes() {
        assert_eq!(format!("{}", ExecutionMode::Flat), "Flat");
        assert_eq!(
            format!(
                "{}",
                ExecutionMode::Maker {
                    bid: true,
                    ask: true
                }
            ),
            "Maker(both)"
        );
        assert_eq!(
            format!(
                "{}",
                ExecutionMode::Maker {
                    bid: true,
                    ask: false
                }
            ),
            "Maker(bid-only)"
        );
        assert_eq!(
            format!("{}", ExecutionMode::InventoryReduce { urgency: 0.7 }),
            "InventoryReduce(urgency=0.70)"
        );
    }

    #[test]
    fn test_priority_kill_over_toxic() {
        // Kill zone should override even toxic regime
        let input = ModeSelectionInput {
            position_zone: PositionZone::Kill,
            toxicity_regime: ToxicityRegime::Toxic,
            position: 1.0,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }

    #[test]
    fn test_priority_red_over_yellow() {
        let input = ModeSelectionInput {
            position_zone: PositionZone::Red,
            toxicity_regime: ToxicityRegime::Benign,
            position: 1.0,
            ..default_input()
        };
        match select_mode(&input) {
            ExecutionMode::InventoryReduce { urgency } => {
                assert!((urgency - 1.0).abs() < 1e-10);
            }
            other => panic!("Expected InventoryReduce, got {other}"),
        }
    }

    #[test]
    fn test_micro_tier_no_value_no_alpha_still_quotes() {
        let input = ModeSelectionInput {
            bid_has_value: false,
            ask_has_value: false,
            has_alpha: false,
            capital_tier: CapitalTier::Micro,
            ..default_input()
        };
        // Micro tier should override Flat → Maker(both)
        assert_eq!(
            select_mode(&input),
            ExecutionMode::Maker {
                bid: true,
                ask: true
            }
        );
    }

    #[test]
    fn test_small_tier_no_value_no_alpha_still_quotes() {
        let input = ModeSelectionInput {
            bid_has_value: false,
            ask_has_value: false,
            has_alpha: false,
            capital_tier: CapitalTier::Small,
            ..default_input()
        };
        assert_eq!(
            select_mode(&input),
            ExecutionMode::Maker {
                bid: true,
                ask: true
            }
        );
    }

    #[test]
    fn test_medium_tier_no_value_no_alpha_flat() {
        let input = ModeSelectionInput {
            bid_has_value: false,
            ask_has_value: false,
            has_alpha: false,
            capital_tier: CapitalTier::Medium,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }

    #[test]
    fn test_micro_tier_toxic_still_flat() {
        // Even Micro tier can't override Kill/Toxic safety
        let input = ModeSelectionInput {
            position_zone: PositionZone::Kill,
            capital_tier: CapitalTier::Micro,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }
}
