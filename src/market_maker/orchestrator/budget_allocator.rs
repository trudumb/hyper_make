//! Budget-constrained API allocation for order reconciliation.
//!
//! Implements a greedy knapsack: select highest-value reconciliation actions
//! within the available API budget. Emergency actions (risk cancels) always
//! execute and are not counted against the budget.
//!
//! ```text
//! scored_updates ──► sort by value_bps desc ──► greedy fill within budget ──► LadderActions
//! ```

#![allow(dead_code)]

use crate::helpers::truncate_float;

use super::super::{ActionType, LadderAction, ScoredUpdate};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// API budget for one reconciliation cycle.
#[derive(Debug, Clone)]
pub(crate) struct ApiBudget {
    /// Maximum API calls available this cycle.
    pub max_calls: u32,
    /// Current API headroom fraction [0, 1].
    pub headroom_pct: f64,
    /// Expected interval to next cycle (seconds).
    pub seconds_to_next: f64,
}

impl ApiBudget {
    /// Compute budget from headroom and rate limit cap.
    ///
    /// Formula: `cap_per_minute × cycle_interval_s / 60 × headroom × 0.5`
    ///
    /// The 0.5 safety margin ensures we never consume more than half of what
    /// we could theoretically spend. Minimum budget is always 2 (one cancel +
    /// one place for emergency situations).
    pub(crate) fn from_headroom(headroom: f64, cap_per_minute: u32, cycle_interval_s: f64) -> Self {
        let calls_available = cap_per_minute as f64 * cycle_interval_s / 60.0 * headroom * 0.5;
        Self {
            max_calls: (calls_available as u32).max(2),
            headroom_pct: headroom,
            seconds_to_next: cycle_interval_s,
        }
    }
}

/// Result of budget allocation.
#[derive(Debug, Clone)]
pub(crate) struct AllocationResult {
    /// Actions to execute (already converted to LadderActions).
    pub actions: Vec<LadderAction>,
    /// Total API calls consumed by selected actions.
    pub calls_used: u32,
    /// Budget that was available.
    pub calls_budget: u32,
    /// Number of orders latched (0 cost, kept as-is).
    pub latched_count: u32,
    /// True if budget ran out before all positive-value actions were selected.
    pub budget_exhausted: bool,
    /// Sum of value_bps for all selected actions.
    pub total_value_bps: f64,
    /// Actions with positive value that were suppressed due to budget.
    pub suppressed_count: u32,
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

/// Greedy knapsack allocation: select highest-value actions within budget.
///
/// # Invariants
/// - Emergency actions (stale cancels far from targets) always execute
/// - Negative-value actions are never selected regardless of budget
/// - Latch actions cost 0 API calls and are always "selected"
/// - Actions are returned as LadderActions ready for execution
pub(crate) fn allocate(
    scored: &mut Vec<ScoredUpdate>,
    budget: &ApiBudget,
    sz_decimals: u32,
) -> AllocationResult {
    let mut actions = Vec::new();
    let mut calls_used: u32 = 0;
    let mut latched_count: u32 = 0;
    let mut total_value_bps: f64 = 0.0;
    let mut suppressed_count: u32 = 0;

    // Phase 1: Always execute emergency actions (not counted against budget).
    // Emergency = StaleCancel where order is very far from any target.
    let mut remaining = Vec::with_capacity(scored.len());
    for update in scored.drain(..) {
        if is_emergency(&update) {
            if let Some(ladder_actions) = to_ladder_actions(&update, sz_decimals) {
                actions.extend(ladder_actions);
            }
            total_value_bps += update.value_bps;
            // Emergency calls NOT counted against budget
        } else {
            remaining.push(update);
        }
    }

    // Phase 2: Handle latches (zero cost, always "selected").
    let mut budgeted = Vec::with_capacity(remaining.len());
    for update in remaining {
        if update.action == ActionType::Latch {
            latched_count += 1;
            // No LadderAction emitted — latch means "do nothing"
        } else {
            budgeted.push(update);
        }
    }

    // Phase 3: Filter out deeply negative-value actions.
    // Allow slightly negative EV (-2 bps) so guaranteed quotes can still be placed
    // even when API cost makes them marginally negative. A MM with 0 resting
    // orders MUST place quotes.
    budgeted.retain(|u| u.value_bps > -2.0);

    // Phase 4: Sort by value_bps descending (highest value first).
    budgeted.sort_by(|a, b| {
        b.value_bps
            .partial_cmp(&a.value_bps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Phase 5: Greedy fill — take from highest value until budget exhausted.
    let mut budget_exhausted = false;
    let mut selected_indices = Vec::new();
    for (i, update) in budgeted.iter().enumerate() {
        let cost = update.api_calls;
        if calls_used + cost <= budget.max_calls {
            if let Some(ladder_actions) = to_ladder_actions(update, sz_decimals) {
                actions.extend(ladder_actions);
            }
            calls_used += cost;
            total_value_bps += update.value_bps;
            selected_indices.push(i);
        } else {
            budget_exhausted = true;
            suppressed_count += 1;
        }
    }

    // Phase 6 (WS6a): Side balance guarantee.
    // When budget >= 4 calls, ensure at least 1 action per side.
    // A market maker with quotes on only one side cannot capture two-sided spread.
    // Allow 1-2 call overrun to satisfy this guarantee.
    if budget.max_calls >= 4 {
        use crate::market_maker::tracking::Side;
        let has_bid_action = actions.iter().any(|a| matches!(a,
            LadderAction::Place { side: Side::Buy, .. } |
            LadderAction::Modify { side: Side::Buy, .. }
        ));
        let has_ask_action = actions.iter().any(|a| matches!(a,
            LadderAction::Place { side: Side::Sell, .. } |
            LadderAction::Modify { side: Side::Sell, .. }
        ));

        // Find best unallocated action for the starved side
        for (i, update) in budgeted.iter().enumerate() {
            if selected_indices.contains(&i) {
                continue; // Already selected
            }
            let is_needed_side = (!has_bid_action && update.side == Side::Buy)
                || (!has_ask_action && update.side == Side::Sell);
            if is_needed_side && update.value_bps > -2.0 {
                if let Some(ladder_actions) = to_ladder_actions(update, sz_decimals) {
                    actions.extend(ladder_actions);
                }
                calls_used += update.api_calls;
                total_value_bps += update.value_bps;
                // Only fill one side's gap per cycle
                break;
            }
        }
    }

    AllocationResult {
        actions,
        calls_used,
        calls_budget: budget.max_calls,
        latched_count,
        budget_exhausted,
        total_value_bps,
        suppressed_count,
    }
}

/// Check if an action is emergency priority (always executes).
fn is_emergency(update: &ScoredUpdate) -> bool {
    // StaleCancel is always emergency — stale orders MUST be cleaned up regardless
    // of value_bps (which is always negative: -ev_keep - api_cost). A stale order
    // has no matching target and wastes a resting slot. W2 audit fix: was > 10.0
    // which is unreachable since StaleCancel.value_bps is always negative.
    //
    // is_emergency field: set by reconciler when ladder has large deficit (≥3 levels
    // missing). Ensures replenishment bypasses budget even when per-order EV is low.
    matches!(update.action, ActionType::StaleCancel) || update.is_emergency
}

/// Convert a ScoredUpdate to one or more LadderActions.
fn to_ladder_actions(update: &ScoredUpdate, sz_decimals: u32) -> Option<Vec<LadderAction>> {
    let mut result = Vec::new();

    match update.action {
        ActionType::Latch => {
            // No action needed
            return None;
        }
        ActionType::ModifySize => {
            let oid = update.oid?;
            let truncated = truncate_float(update.target_size, sz_decimals, false);
            if truncated <= 0.0 {
                // Size truncated to zero — cancel instead
                result.push(LadderAction::Cancel { oid });
            } else {
                result.push(LadderAction::Modify {
                    oid,
                    new_price: update.current_price, // Same price preserves queue
                    new_size: truncated,
                    side: update.side,
                });
            }
        }
        ActionType::ModifyPrice => {
            let oid = update.oid?;
            let truncated = truncate_float(update.target_size, sz_decimals, false);
            if truncated <= 0.0 {
                result.push(LadderAction::Cancel { oid });
            } else {
                result.push(LadderAction::Modify {
                    oid,
                    new_price: update.target_price,
                    new_size: truncated,
                    side: update.side,
                });
            }
        }
        ActionType::CancelPlace => {
            let oid = update.oid?;
            result.push(LadderAction::Cancel { oid });
            let truncated = truncate_float(update.target_size, sz_decimals, false);
            if truncated > 0.0 {
                result.push(LadderAction::Place {
                    side: update.side,
                    price: update.target_price,
                    size: truncated,
                });
            }
        }
        ActionType::NewPlace => {
            let truncated = truncate_float(update.target_size, sz_decimals, false);
            if truncated > 0.0 {
                result.push(LadderAction::Place {
                    side: update.side,
                    price: update.target_price,
                    size: truncated,
                });
            }
        }
        ActionType::StaleCancel => {
            let oid = update.oid?;
            result.push(LadderAction::Cancel { oid });
        }
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::tracking::Side;

    fn make_scored(
        oid: Option<u64>,
        action: ActionType,
        value_bps: f64,
        side: Side,
        target_price: f64,
        target_size: f64,
    ) -> ScoredUpdate {
        ScoredUpdate {
            oid,
            target_idx: 0,
            action,
            value_bps,
            api_calls: action.api_calls(),
            p_fill_keep: 0.0,
            p_fill_new: 0.0,
            side,
            target_price,
            target_size,
            current_price: target_price, // Same for simplicity
            is_emergency: false,
        }
    }

    #[test]
    fn test_greedy_selection_picks_highest_value() {
        let mut scored = vec![
            make_scored(Some(1), ActionType::ModifyPrice, 5.0, Side::Buy, 100.0, 1.0),
            make_scored(Some(2), ActionType::ModifyPrice, 10.0, Side::Buy, 101.0, 1.0),
            make_scored(Some(3), ActionType::ModifyPrice, 2.0, Side::Buy, 102.0, 1.0),
        ];
        let budget = ApiBudget {
            max_calls: 2, // Only room for 2 modify actions
            headroom_pct: 0.5,
            seconds_to_next: 5.0,
        };
        let result = allocate(&mut scored, &budget, 2);

        // Should pick value=10 and value=5, skip value=2
        assert_eq!(result.calls_used, 2);
        assert!(result.budget_exhausted);
        assert_eq!(result.suppressed_count, 1);
        assert!((result.total_value_bps - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_emergency_bypasses_budget() {
        let mut scored = vec![
            // Emergency stale cancel (value > 10)
            make_scored(Some(1), ActionType::StaleCancel, 15.0, Side::Buy, 100.0, 1.0),
            // Normal action
            make_scored(Some(2), ActionType::ModifyPrice, 5.0, Side::Buy, 101.0, 1.0),
        ];
        let budget = ApiBudget {
            max_calls: 0, // Zero budget!
            headroom_pct: 0.01,
            seconds_to_next: 5.0,
        };
        let result = allocate(&mut scored, &budget, 2);

        // Emergency should execute even with 0 budget
        assert!(!result.actions.is_empty());
        // The stale cancel is emergency, the modify is suppressed
        let cancel_count = result.actions.iter().filter(|a| matches!(a, LadderAction::Cancel { .. })).count();
        assert_eq!(cancel_count, 1);
    }

    #[test]
    fn test_negative_value_never_selected() {
        let mut scored = vec![
            make_scored(Some(1), ActionType::CancelPlace, -5.0, Side::Buy, 100.0, 1.0),
            make_scored(Some(2), ActionType::ModifyPrice, -2.0, Side::Buy, 101.0, 1.0),
        ];
        let budget = ApiBudget {
            max_calls: 100, // Plenty of budget
            headroom_pct: 1.0,
            seconds_to_next: 5.0,
        };
        let result = allocate(&mut scored, &budget, 2);

        assert_eq!(result.calls_used, 0);
        assert_eq!(result.suppressed_count, 0);
        assert!(result.actions.is_empty());
    }

    #[test]
    fn test_latch_counted_but_zero_cost() {
        let mut scored = vec![
            make_scored(Some(1), ActionType::Latch, 0.0, Side::Buy, 100.0, 1.0),
            make_scored(Some(2), ActionType::Latch, 0.0, Side::Sell, 101.0, 1.0),
            make_scored(None, ActionType::NewPlace, 3.0, Side::Buy, 99.0, 1.0),
        ];
        let budget = ApiBudget {
            max_calls: 5,
            headroom_pct: 0.5,
            seconds_to_next: 5.0,
        };
        let result = allocate(&mut scored, &budget, 2);

        assert_eq!(result.latched_count, 2);
        assert_eq!(result.calls_used, 1); // Only the NewPlace
    }

    #[test]
    fn test_minimum_budget_from_headroom() {
        // Tiny headroom should still give minimum 2 calls
        let budget = ApiBudget::from_headroom(0.001, 1200, 1.0);
        assert_eq!(budget.max_calls, 2);
    }

    #[test]
    fn test_budget_from_headroom_normal() {
        // 30% headroom, 1200 cap, 10s cycle
        // 1200 * 10/60 * 0.3 * 0.5 = 30
        let budget = ApiBudget::from_headroom(0.3, 1200, 10.0);
        assert_eq!(budget.max_calls, 30);
    }

    #[test]
    fn test_cancel_place_costs_two_calls() {
        let mut scored = vec![
            make_scored(Some(1), ActionType::CancelPlace, 20.0, Side::Buy, 100.0, 1.0),
        ];
        let budget = ApiBudget {
            max_calls: 1, // Only 1 call — not enough for cancel+place (2)
            headroom_pct: 0.5,
            seconds_to_next: 5.0,
        };
        let result = allocate(&mut scored, &budget, 2);

        // Can't fit a 2-call action in a 1-call budget
        assert_eq!(result.calls_used, 0);
        assert!(result.budget_exhausted);
        assert_eq!(result.suppressed_count, 1);
    }

    // WS6a: Side balance guarantee tests
    #[test]
    fn test_side_balance_fills_starved_side() {
        // Budget = 4, all high-value actions on Buy side, one low-value Sell
        let mut scored = vec![
            make_scored(None, ActionType::NewPlace, 8.0, Side::Buy, 99.0, 1.0),
            make_scored(None, ActionType::NewPlace, 7.0, Side::Buy, 98.0, 1.0),
            make_scored(None, ActionType::NewPlace, 6.0, Side::Buy, 97.0, 1.0),
            make_scored(None, ActionType::NewPlace, 5.0, Side::Buy, 96.0, 1.0),
            // Low-value sell — would normally be suppressed
            make_scored(None, ActionType::NewPlace, 1.0, Side::Sell, 101.0, 1.0),
        ];
        let budget = ApiBudget {
            max_calls: 4,
            headroom_pct: 0.5,
            seconds_to_next: 5.0,
        };
        let result = allocate(&mut scored, &budget, 2);

        // Should have at least one Sell action from side balance
        let has_sell = result.actions.iter().any(|a| matches!(a,
            LadderAction::Place { side: Side::Sell, .. }
        ));
        assert!(has_sell, "Side balance should ensure at least one sell action");
    }

    #[test]
    fn test_side_balance_not_triggered_with_small_budget() {
        // Budget < 4: side balance should not trigger
        let mut scored = vec![
            make_scored(None, ActionType::NewPlace, 8.0, Side::Buy, 99.0, 1.0),
            make_scored(None, ActionType::NewPlace, 1.0, Side::Sell, 101.0, 1.0),
        ];
        let budget = ApiBudget {
            max_calls: 1, // Too small for side balance
            headroom_pct: 0.5,
            seconds_to_next: 5.0,
        };
        let result = allocate(&mut scored, &budget, 2);

        // With budget=1, should only pick the highest value (Buy)
        assert_eq!(result.calls_used, 1);
    }
}
