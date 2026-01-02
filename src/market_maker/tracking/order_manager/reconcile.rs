//! Ladder reconciliation logic.
//!
//! Implements diffing between current orders and target ladder to generate
//! minimal cancel/place/modify actions for improved spread capturing.

use std::collections::HashSet;

use crate::{bps_diff, EPSILON};

use super::types::{LadderAction, Side, TrackedOrder};
use crate::market_maker::quoting::LadderLevel;

/// Configuration for smart reconciliation thresholds.
///
/// Controls when to SKIP (unchanged), MODIFY (preserve queue), or CANCEL+PLACE.
#[derive(Debug, Clone)]
pub struct ReconcileConfig {
    /// Max price difference in bps to consider using MODIFY (larger = cancel+place)
    pub max_modify_price_bps: u16,
    /// Max size change percentage to consider using MODIFY (larger = cancel+place)
    pub max_modify_size_pct: f64,
    /// Price tolerance in bps for considering an order unchanged (SKIP)
    pub skip_price_tolerance_bps: u16,
    /// Size tolerance as fraction for considering an order unchanged (SKIP)
    pub skip_size_tolerance_pct: f64,
}

impl Default for ReconcileConfig {
    fn default() -> Self {
        Self {
            max_modify_price_bps: 10,      // Modify if price ≤ 10 bps change
            max_modify_size_pct: 0.50,     // Modify if size ≤ 50% change
            skip_price_tolerance_bps: 1,   // Skip if price ≤ 1 bps (unchanged)
            skip_size_tolerance_pct: 0.05, // Skip if size ≤ 5% (unchanged)
        }
    }
}

/// Result of matching an order to a target level.
#[derive(Debug)]
struct MatchResult<'a> {
    order: &'a TrackedOrder,
    target_index: usize,
    price_diff_bps: f64,
    size_diff_pct: f64,
}

/// Smart reconciliation with SKIP/MODIFY/CANCEL+PLACE decisions.
///
/// Decision logic per level:
/// 1. Order matches target within skip tolerance -> SKIP (no action, preserve queue)
/// 2. Order matches target within modify threshold -> MODIFY (preserve queue)
/// 3. Order exists but price/size moved too far -> CANCEL + PLACE
/// 4. No order at target level -> PLACE
/// 5. Order exists with no matching target -> CANCEL
pub fn reconcile_side_smart(
    current: &[&TrackedOrder],
    target: &[LadderLevel],
    side: Side,
    config: &ReconcileConfig,
) -> Vec<LadderAction> {
    let mut actions = Vec::new();
    let mut matched_targets: HashSet<usize> = HashSet::new();
    let mut matched_orders: HashSet<u64> = HashSet::new();

    // Phase 1: Find best matches between current orders and target levels
    let matches = find_best_matches(current, target, &matched_orders);

    for m in matches {
        matched_targets.insert(m.target_index);
        matched_orders.insert(m.order.oid);

        let target_level = &target[m.target_index];

        // Decision: SKIP, MODIFY, or CANCEL+PLACE
        if m.price_diff_bps <= config.skip_price_tolerance_bps as f64
            && m.size_diff_pct <= config.skip_size_tolerance_pct
        {
            // SKIP - order is close enough, preserve queue position
            continue;
        } else if m.price_diff_bps <= config.max_modify_price_bps as f64
            && m.size_diff_pct <= config.max_modify_size_pct
        {
            // MODIFY - small change, preserve queue position
            actions.push(LadderAction::Modify {
                oid: m.order.oid,
                new_price: target_level.price,
                new_size: target_level.size,
                side,
            });
        } else {
            // CANCEL + PLACE - too large a change, fresh queue
            actions.push(LadderAction::Cancel { oid: m.order.oid });
            if target_level.size > EPSILON {
                actions.push(LadderAction::Place {
                    side,
                    price: target_level.price,
                    size: target_level.size,
                });
            }
        }
    }

    // Phase 2: Cancel any unmatched orders
    for order in current {
        if !matched_orders.contains(&order.oid) {
            actions.push(LadderAction::Cancel { oid: order.oid });
        }
    }

    // Phase 3: Place orders for unmatched target levels
    for (i, level) in target.iter().enumerate() {
        if !matched_targets.contains(&i) && level.size > EPSILON {
            actions.push(LadderAction::Place {
                side,
                price: level.price,
                size: level.size,
            });
        }
    }

    actions
}

/// Find best matching order for each target level.
fn find_best_matches<'a>(
    orders: &[&'a TrackedOrder],
    targets: &[LadderLevel],
    already_matched: &HashSet<u64>,
) -> Vec<MatchResult<'a>> {
    let mut results = Vec::new();
    let mut used_orders: HashSet<u64> = already_matched.clone();
    let mut used_targets: HashSet<usize> = HashSet::new();

    // For each target, find the closest unmatched order
    for (target_idx, target) in targets.iter().enumerate() {
        let mut best_match: Option<MatchResult> = None;
        let mut best_distance = f64::MAX;

        for order in orders.iter() {
            if used_orders.contains(&order.oid) {
                continue;
            }

            let price_diff_bps = bps_diff(order.price, target.price) as f64;
            let size_diff_pct = if order.remaining() > EPSILON {
                ((order.remaining() - target.size).abs() / order.remaining()).min(1.0)
            } else {
                1.0
            };

            // Combined distance metric: prioritize price, then size
            let distance = price_diff_bps + size_diff_pct * 10.0;

            // Only consider matches within 100 bps (anything further is not a match)
            if price_diff_bps <= 100.0 && distance < best_distance {
                best_distance = distance;
                best_match = Some(MatchResult {
                    order,
                    target_index: target_idx,
                    price_diff_bps,
                    size_diff_pct,
                });
            }
        }

        if let Some(m) = best_match {
            used_orders.insert(m.order.oid);
            used_targets.insert(target_idx);
            results.push(m);
        }
    }

    results
}

/// Reconcile a single side: match current orders to target levels.
pub(crate) fn reconcile_side(
    current: &[&TrackedOrder],
    target: &[LadderLevel],
    side: Side,
    max_bps_diff: u16,
) -> Vec<LadderAction> {
    let mut actions = Vec::new();
    let mut matched_levels: HashSet<usize> = HashSet::new();

    // Match current orders to target levels
    for order in current {
        let mut found_match = false;
        for (i, level) in target.iter().enumerate() {
            if matched_levels.contains(&i) {
                continue;
            }
            // Check if order matches level (within tolerance)
            let price_diff = bps_diff(order.price, level.price);
            if price_diff <= max_bps_diff {
                matched_levels.insert(i);
                found_match = true;
                break;
            }
        }
        if !found_match {
            // Order doesn't match any target level - cancel it
            actions.push(LadderAction::Cancel { oid: order.oid });
        }
    }

    // Place orders for unmatched target levels
    for (i, level) in target.iter().enumerate() {
        if !matched_levels.contains(&i) && level.size > EPSILON {
            actions.push(LadderAction::Place {
                side,
                price: level.price,
                size: level.size,
            });
        }
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_order(oid: u64, side: Side, price: f64) -> TrackedOrder {
        TrackedOrder::new(oid, side, price, 1.0)
    }

    #[test]
    fn test_reconcile_empty_current_to_target() {
        let current: Vec<&TrackedOrder> = vec![];
        let target = vec![
            LadderLevel {
                price: 99.0,
                size: 1.0,
                depth_bps: 10.0,
            },
            LadderLevel {
                price: 98.0,
                size: 0.5,
                depth_bps: 20.0,
            },
        ];

        let actions = reconcile_side(&current, &target, Side::Buy, 5);

        assert_eq!(actions.len(), 2);
        assert!(actions
            .iter()
            .all(|a| matches!(a, LadderAction::Place { .. })));
    }

    #[test]
    fn test_reconcile_matching_orders() {
        let order1 = make_order(1, Side::Buy, 99.0);
        let order2 = make_order(2, Side::Buy, 98.0);
        let current: Vec<&TrackedOrder> = vec![&order1, &order2];

        let target = vec![
            LadderLevel {
                price: 99.0,
                size: 1.0,
                depth_bps: 10.0,
            },
            LadderLevel {
                price: 98.0,
                size: 0.5,
                depth_bps: 20.0,
            },
        ];

        let actions = reconcile_side(&current, &target, Side::Buy, 5);

        // Orders match - no actions needed
        assert!(actions.is_empty());
    }

    #[test]
    fn test_reconcile_stale_orders() {
        let order1 = make_order(1, Side::Buy, 95.0); // Too far
        let current: Vec<&TrackedOrder> = vec![&order1];

        let target = vec![LadderLevel {
            price: 99.0,
            size: 1.0,
            depth_bps: 10.0,
        }];

        let actions = reconcile_side(&current, &target, Side::Buy, 5);

        // Should cancel old and place new
        assert_eq!(actions.len(), 2);
        assert!(actions
            .iter()
            .any(|a| matches!(a, LadderAction::Cancel { oid: 1 })));
        assert!(actions
            .iter()
            .any(|a| matches!(a, LadderAction::Place { .. })));
    }
}
