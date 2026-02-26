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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum ExecutionMode {
    /// Cancel all quotes — position zone Kill, or Toxic+flat
    #[default]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    // --- Cascade & Hawkes fields (A2: continuous GLFT-based quoting) ---
    /// Cascade size factor [0, 1]. 0 = full cascade (OI collapsing), 1 = calm.
    /// Derived from OI-drop and liquidation cascade tracker.
    #[serde(default = "default_cascade_size_factor")]
    pub cascade_size_factor: f64,
    /// Threshold below which `cascade_size_factor` triggers Flat.
    /// Default 0.3 — below 30% of normal depth, pull quotes entirely.
    #[serde(default = "default_cascade_threshold")]
    pub cascade_threshold: f64,
    /// Hawkes process probability of being in a fill cluster [0, 1].
    /// High values indicate self-exciting fill arrival — likely adverse.
    #[serde(default)]
    pub hawkes_p_cluster: f64,
    /// Hawkes process branching ratio [0, 1+].
    /// \> 1.0 = supercritical (fills beget more fills), combined with
    /// high `p_cluster` indicates toxic self-exciting flow.
    #[serde(default)]
    pub hawkes_branching_ratio: f64,
    /// Net directional flow pressure. Positive = buy pressure, negative = sell pressure.
    /// Used to detect when flow opposes our position (adverse for inventory).
    #[serde(default)]
    pub flow_direction: f64,
    /// Position fraction threshold for inventory-reduce mode.
    /// When `|position| / max_position > reduce_only_threshold` in Red zone
    /// with opposing flow, switch to InventoryReduce.
    #[serde(default = "default_reduce_only_threshold")]
    pub reduce_only_threshold: f64,
    /// Maximum allowed position (absolute). Used to compute position fraction
    /// for inventory-reduce decisions.
    #[serde(default = "default_max_position")]
    pub max_position: f64,
}

fn default_cascade_size_factor() -> f64 {
    1.0
}
fn default_cascade_threshold() -> f64 {
    0.3
}
fn default_reduce_only_threshold() -> f64 {
    0.7
}
fn default_max_position() -> f64 {
    f64::MAX
}

/// Pure function: select execution mode from inputs.
///
/// Priority-ordered checks (higher = checked first):
/// 1. Kill zone → Flat (HJB constraint boundary)
/// 2. Cascade circuit breaker → Flat (OI collapse, not a pricing issue)
/// 3. Hawkes critical clustering → Flat (self-exciting toxic flow)
/// 4. Red zone + high position + opposing flow → InventoryReduce
/// 5. Everything else → Maker{both} (GLFT handles via γ·q·σ²·(T-t))
///
/// Cascade and Hawkes are circuit breakers, not pricing adjustments — when
/// the market structure is broken (OI collapsing, fills self-exciting), no
/// spread is wide enough to compensate. Pull quotes entirely.
///
/// InventoryReduce triggers only when all three conditions align: deep Red
/// zone, position fraction above threshold, AND flow actively opposing our
/// position. This is the "being run over" scenario where passive GLFT
/// inventory penalty is insufficient.
pub fn select_mode(input: &ModeSelectionInput) -> ExecutionMode {
    // 1. Kill zone: HJB constraint boundary — absolute safety cutoff.
    if input.position_zone == PositionZone::Kill {
        return ExecutionMode::Flat;
    }

    // 2. Cascade circuit breaker: OI collapsing means the orderbook is
    // structurally broken. No spread compensates for a liquidation cascade.
    // Use >= so that exactly-at-threshold does NOT trigger Flat.
    if input.cascade_size_factor < input.cascade_threshold {
        return ExecutionMode::Flat;
    }

    // 3. Hawkes critical clustering: self-exciting fill arrival where each
    // fill triggers more fills. p_cluster > 0.8 AND branching_ratio > 0.85
    // indicates supercritical regime — pull quotes before getting run over.
    if input.hawkes_p_cluster > 0.8 && input.hawkes_branching_ratio > 0.85 {
        return ExecutionMode::Flat;
    }

    // 4. Red zone + high position fraction + opposing flow → InventoryReduce.
    // This is the "being run over" scenario: we're deep in Red zone,
    // position is large, and market flow is actively pushing against us.
    // GLFT's passive γ·q penalty is insufficient — we need aggressive reduction.
    if input.position_zone == PositionZone::Red && input.max_position > 0.0 {
        let position_fraction = input.position.abs() / input.max_position;
        if position_fraction > input.reduce_only_threshold {
            // Check if flow opposes our position:
            // Long (position > 0) + sell pressure (flow < 0) → opposing
            // Short (position < 0) + buy pressure (flow > 0) → opposing
            let flow_opposes = input.position * input.flow_direction < 0.0;
            if flow_opposes {
                let urgency = position_fraction.min(1.0);
                return ExecutionMode::InventoryReduce { urgency };
            }
        }
    }

    // 5. Everything else: GLFT γ·q penalty handles all asymmetry continuously.
    // Red zone → wider spreads via elevated γ (from regime_gamma_multiplier).
    // Yellow zone → γ·q penalty already biases against accumulation.
    // Toxic regime → routes through σ_conditional (Hawkes) and AS floor.
    ExecutionMode::Maker {
        bid: true,
        ask: true,
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
            is_warmup: false,                 // Default: calibrated (not warming up)
            // Cascade & Hawkes defaults: non-triggering values
            cascade_size_factor: default_cascade_size_factor(),
            cascade_threshold: default_cascade_threshold(),
            hawkes_p_cluster: 0.0,
            hawkes_branching_ratio: 0.0,
            flow_direction: 0.0,
            reduce_only_threshold: default_reduce_only_threshold(),
            max_position: default_max_position(),
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
        // Verify ALL non-Kill zones return Maker{both} with default (non-triggering) inputs
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

    // === Cascade & Hawkes circuit breaker tests (A2) ===

    #[test]
    fn test_cascade_triggers_flat() {
        // OI collapsing below 30% of normal → pull all quotes.
        // cascade_size_factor=0.2 < cascade_threshold=0.3 → Flat
        let input = ModeSelectionInput {
            cascade_size_factor: 0.2,
            cascade_threshold: 0.3,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }

    #[test]
    fn test_hawkes_critical_triggers_flat() {
        // Self-exciting fill cluster: p_cluster=0.85 > 0.8 AND
        // branching_ratio=0.90 > 0.85 → supercritical, pull quotes.
        let input = ModeSelectionInput {
            hawkes_p_cluster: 0.85,
            hawkes_branching_ratio: 0.90,
            ..default_input()
        };
        assert_eq!(select_mode(&input), ExecutionMode::Flat);
    }

    #[test]
    fn test_red_zone_opposing_flow_triggers_reduce() {
        // Red zone + position at 80% of max + opposing flow → InventoryReduce.
        // Long position (8.0) with sell pressure (flow=-1.0) is adverse.
        let input = ModeSelectionInput {
            position_zone: PositionZone::Red,
            position: 8.0, // 80% of max_position=10.0
            max_position: 10.0,
            reduce_only_threshold: 0.7,
            flow_direction: -1.0, // Sell pressure opposes long position
            ..default_input()
        };
        match select_mode(&input) {
            ExecutionMode::InventoryReduce { urgency } => {
                // urgency = position_fraction clamped to [0,1] = 0.8
                assert!(
                    (urgency - 0.8).abs() < 1e-10,
                    "Expected urgency ~0.8, got {urgency}"
                );
            }
            other => panic!("Expected InventoryReduce, got {other}"),
        }
    }

    #[test]
    fn test_red_zone_aligned_flow_stays_maker() {
        // Red zone + position at 80% of max, but flow is ALIGNED (not opposing).
        // Long position (8.0) with buy pressure (flow=+1.0) supports us — stay Maker.
        let input = ModeSelectionInput {
            position_zone: PositionZone::Red,
            position: 8.0, // 80% of max_position=10.0
            max_position: 10.0,
            reduce_only_threshold: 0.7,
            flow_direction: 1.0, // Buy pressure ALIGNS with long position
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
    fn test_boundary_values() {
        // cascade_size_factor EXACTLY at threshold → should NOT trigger Flat.
        // The condition is strict less-than: `< cascade_threshold`.
        let input = ModeSelectionInput {
            cascade_size_factor: 0.3, // Exactly at default threshold
            cascade_threshold: 0.3,
            ..default_input()
        };
        assert_eq!(
            select_mode(&input),
            ExecutionMode::Maker {
                bid: true,
                ask: true
            },
            "cascade_size_factor exactly at threshold should NOT trigger Flat"
        );

        // Also verify just below threshold DOES trigger
        let input_below = ModeSelectionInput {
            cascade_size_factor: 0.29999,
            cascade_threshold: 0.3,
            ..default_input()
        };
        assert_eq!(
            select_mode(&input_below),
            ExecutionMode::Flat,
            "cascade_size_factor just below threshold should trigger Flat"
        );
    }
}
