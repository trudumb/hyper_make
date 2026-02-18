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
    /// Whether the system is still in warmup (fill_samples < min_warmup_trades).
    /// During warmup, never go Flat just because signals aren't calibrated —
    /// the system NEEDS fills to calibrate, which requires quoting.
    pub is_warmup: bool,
}

/// Pure function: select execution mode from inputs.
///
/// Only Kill zone is a hard cutoff. Everything else lets GLFT's continuous
/// γ·q·σ²·(T-t) penalty handle inventory asymmetry:
/// - Red zone → wider spreads via elevated γ (from regime_gamma_multiplier)
/// - Yellow zone → γ·q penalty already biases against accumulation
/// - Toxic regime → routes through σ_conditional (Hawkes) and AS floor
/// - No value/alpha → GLFT still produces valid spreads from γ/κ/σ
pub fn select_mode(input: &ModeSelectionInput) -> ExecutionMode {
    // Kill zone: HJB constraint boundary — the ONLY hard cutoff.
    // All other inventory management flows through GLFT's continuous
    // γ·q·σ²·(T-t) penalty, which widens the accumulating side proportionally.
    if input.position_zone == PositionZone::Kill {
        return ExecutionMode::Flat;
    }
    // Everything else: GLFT γ·q penalty handles all asymmetry continuously.
    // Red zone → wider spreads via elevated γ (from regime_gamma_multiplier).
    // Yellow zone → γ·q penalty already biases against accumulation.
    // Toxic regime → routes through σ_conditional (Hawkes) and AS floor.
    ExecutionMode::Maker { bid: true, ask: true }
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
            is_warmup: false, // Default: calibrated (not warming up)
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
    fn test_red_zone_maker_both() {
        // Red zone routes through elevated γ, not InventoryReduce
        let input = ModeSelectionInput {
            position_zone: PositionZone::Red,
            position: 1.0,
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
    fn test_toxic_flat_maker_both() {
        // Toxic regime routes through σ_conditional and AS floor, not Flat
        let input = ModeSelectionInput {
            toxicity_regime: ToxicityRegime::Toxic,
            position: 0.0,
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
    fn test_toxic_positioned_maker_both() {
        // Toxic + positioned routes through σ_conditional and AS floor
        let input = ModeSelectionInput {
            toxicity_regime: ToxicityRegime::Toxic,
            position: 1.5,
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
    fn test_yellow_long_maker_both() {
        // Yellow zone: γ·q penalty handles asymmetry continuously
        let input = ModeSelectionInput {
            position_zone: PositionZone::Yellow,
            position: 1.0, // Long
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
    fn test_yellow_short_maker_both() {
        // Yellow zone: γ·q penalty handles asymmetry continuously
        let input = ModeSelectionInput {
            position_zone: PositionZone::Yellow,
            position: -1.0, // Short
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
    fn test_no_value_no_alpha_maker_both() {
        // GLFT produces valid spreads from γ/κ/σ even without alpha signals
        let input = ModeSelectionInput {
            bid_has_value: false,
            ask_has_value: false,
            has_alpha: false,
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
    fn test_one_sided_value_maker_both() {
        // GLFT quotes both sides; value/alpha no longer gates individual sides
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
                ask: true
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
    fn test_priority_red_over_yellow_maker_both() {
        // Red zone routes through elevated γ, same as all non-Kill zones
        let input = ModeSelectionInput {
            position_zone: PositionZone::Red,
            toxicity_regime: ToxicityRegime::Benign,
            position: 1.0,
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
    fn test_medium_tier_no_value_no_alpha_maker_both() {
        // Medium tier: GLFT produces valid spreads regardless of value/alpha
        let input = ModeSelectionInput {
            bid_has_value: false,
            ask_has_value: false,
            has_alpha: false,
            capital_tier: CapitalTier::Medium,
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
    fn test_micro_tier_toxic_still_flat() {
        // Even Micro tier can't override Kill/Toxic safety
        let input = ModeSelectionInput {
            position_zone: PositionZone::Kill,
            capital_tier: CapitalTier::Micro,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }

    // === Warmup tests: system MUST quote during warmup to break death spiral ===

    #[test]
    fn test_warmup_no_value_no_alpha_still_quotes() {
        // THE cold-start death spiral fix: during warmup, never go Flat
        // even for medium+ capital with no detected edge.
        let input = ModeSelectionInput {
            bid_has_value: false,
            ask_has_value: false,
            has_alpha: false,
            capital_tier: CapitalTier::Medium,
            is_warmup: true,
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
    fn test_warmup_large_tier_still_quotes() {
        let input = ModeSelectionInput {
            bid_has_value: false,
            ask_has_value: false,
            has_alpha: false,
            capital_tier: CapitalTier::Large,
            is_warmup: true,
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
    fn test_warmup_toxic_flat_still_quotes() {
        // During warmup, "Toxic" regime may be noise from uncalibrated classifier.
        // Don't go Flat — quote with wide spreads instead.
        let input = ModeSelectionInput {
            toxicity_regime: ToxicityRegime::Toxic,
            position: 0.0,
            is_warmup: true,
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
    fn test_warmup_kill_zone_still_flat() {
        // Kill zone overrides warmup — safety is absolute
        let input = ModeSelectionInput {
            position_zone: PositionZone::Kill,
            is_warmup: true,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }

    #[test]
    fn test_warmup_red_zone_maker_both() {
        // Red zone + warmup: GLFT γ·q penalty handles inventory continuously
        let input = ModeSelectionInput {
            position_zone: PositionZone::Red,
            position: 1.0,
            is_warmup: true,
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
    fn test_warmup_toxic_positioned_maker_both() {
        // Warmup + toxic + positioned: GLFT handles via σ_conditional and AS floor
        let input = ModeSelectionInput {
            toxicity_regime: ToxicityRegime::Toxic,
            position: 1.5,
            is_warmup: true,
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
    fn test_only_kill_goes_flat() {
        // Verify ALL non-Kill zones return Maker{both}
        for zone in [PositionZone::Green, PositionZone::Yellow, PositionZone::Red] {
            let input = ModeSelectionInput {
                position_zone: zone,
                ..default_input()
            };
            assert_eq!(
                select_mode(&input),
                ExecutionMode::Maker {
                    bid: true,
                    ask: true
                },
                "Non-Kill zone {:?} should return Maker(both)",
                zone
            );
        }
    }
}
