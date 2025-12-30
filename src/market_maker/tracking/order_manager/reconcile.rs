//! Ladder reconciliation logic.
//!
//! Implements diffing between current orders and target ladder to generate
//! minimal cancel/place actions.

use std::collections::HashSet;

use crate::{bps_diff, EPSILON};

use super::types::{LadderAction, Side, TrackedOrder};
use crate::market_maker::quoting::LadderLevel;

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
        assert!(actions.iter().all(|a| matches!(a, LadderAction::Place { .. })));
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
        assert!(actions.iter().any(|a| matches!(a, LadderAction::Cancel { oid: 1 })));
        assert!(actions.iter().any(|a| matches!(a, LadderAction::Place { .. })));
    }
}
