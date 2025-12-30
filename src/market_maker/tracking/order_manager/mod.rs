//! Order state management for tracking resting orders.
//!
//! This module implements a robust order lifecycle state machine that handles
//! the race condition between cancel requests and fill notifications. The key
//! insight is that fills can arrive via WebSocket *after* a cancel has been
//! confirmed by the exchange, so we must maintain a "fill window" after cancel
//! confirmation before removing orders from tracking.
//!
//! # Module Structure
//!
//! - `types`: Core types (`Side`, `OrderState`, `TrackedOrder`, `PendingOrder`)
//! - `manager`: `OrderManager` implementation
//! - `reconcile`: Ladder reconciliation logic

mod manager;
mod reconcile;
mod types;

pub use manager::OrderManager;
pub use types::{
    LadderAction, OrderManagerConfig, OrderState, PendingOrder, Side, TrackedOrder,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::quoting::{Ladder, LadderLevel};
    use std::time::Duration;

    #[test]
    fn test_add_and_get_order() {
        let mut mgr = OrderManager::new();
        let order = TrackedOrder::new(123, Side::Buy, 100.0, 1.0);
        mgr.add_order(order);

        let retrieved = mgr.get_order(123).unwrap();
        assert_eq!(retrieved.oid, 123);
        assert_eq!(retrieved.side, Side::Buy);
    }

    #[test]
    fn test_get_by_side() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));
        mgr.add_order(TrackedOrder::new(2, Side::Sell, 101.0, 0.5));

        let buy = mgr.get_by_side(Side::Buy).unwrap();
        assert_eq!(buy.oid, 1);

        let sell = mgr.get_by_side(Side::Sell).unwrap();
        assert_eq!(sell.oid, 2);
    }

    #[test]
    fn test_process_fill() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));

        // First fill - should work and transition to PartialFilled
        let (found, is_new, complete) = mgr.process_fill(1, 101, 0.5);
        assert!(found && is_new && !complete);

        let order = mgr.get_order(1).unwrap();
        assert!((order.filled - 0.5).abs() < f64::EPSILON);
        assert!((order.remaining() - 0.5).abs() < f64::EPSILON);
        assert_eq!(order.state, OrderState::PartialFilled);
    }

    #[test]
    fn test_is_filled() {
        let mut order = TrackedOrder::new(1, Side::Buy, 100.0, 1.0);
        assert!(!order.is_filled());

        order.filled = 1.0;
        assert!(order.is_filled());
    }

    #[test]
    fn test_cancel_then_fill_race() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));

        // Initiate cancel
        assert!(mgr.initiate_cancel(1));
        assert_eq!(mgr.get_order(1).unwrap().state, OrderState::CancelPending);

        // Fill arrives during cancel
        let (found, is_new, complete) = mgr.process_fill(1, 101, 1.0);
        assert!(found && is_new && complete);
        assert_eq!(
            mgr.get_order(1).unwrap().state,
            OrderState::FilledDuringCancel
        );

        // Cleanup should remove it
        let removed = mgr.cleanup();
        assert_eq!(removed, vec![1]);
        assert!(mgr.get_order(1).is_none());
    }

    #[test]
    fn test_cancel_confirmed_fill_window() {
        let config = OrderManagerConfig {
            fill_window_duration: Duration::from_millis(50),
            cancel_timeout: Duration::from_secs(30),
        };
        let mut mgr = OrderManager::with_config(config);

        // Place and initiate cancel
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));
        mgr.initiate_cancel(1);
        mgr.on_cancel_confirmed(1);

        // Immediately after confirm - should NOT be cleaned up
        let removed = mgr.cleanup();
        assert!(removed.is_empty());
        assert!(mgr.get_order(1).is_some());

        // Wait for fill window to expire
        std::thread::sleep(Duration::from_millis(60));

        // Now should be cleaned up
        let removed = mgr.cleanup();
        assert_eq!(removed, vec![1]);
        assert!(mgr.get_order(1).is_none());
    }

    #[test]
    fn test_late_fill_after_cancel_confirmed() {
        let config = OrderManagerConfig {
            fill_window_duration: Duration::from_millis(100),
            cancel_timeout: Duration::from_secs(30),
        };
        let mut mgr = OrderManager::with_config(config);

        // Place, initiate cancel, confirm
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));
        mgr.initiate_cancel(1);
        mgr.on_cancel_confirmed(1);

        // Fill arrives within window
        let (found, is_new, _) = mgr.process_fill(1, 101, 0.5);
        assert!(found && is_new);
        assert_eq!(
            mgr.get_order(1).unwrap().state,
            OrderState::FilledDuringCancel
        );

        // Should be cleaned up immediately (no need to wait)
        let removed = mgr.cleanup();
        assert_eq!(removed, vec![1]);
    }

    #[test]
    fn test_duplicate_fill_rejected() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 2.0));

        // First fill
        let (_, is_new1, _) = mgr.process_fill(1, 101, 1.0);
        assert!(is_new1);
        assert!((mgr.get_order(1).unwrap().filled - 1.0).abs() < f64::EPSILON);

        // Duplicate fill (same tid)
        let (_, is_new2, _) = mgr.process_fill(1, 101, 1.0);
        assert!(!is_new2);
        // Filled amount unchanged
        assert!((mgr.get_order(1).unwrap().filled - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_active_excludes_cancelling() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));
        mgr.add_order(TrackedOrder::new(2, Side::Buy, 99.0, 1.0));

        // Cancel one
        mgr.initiate_cancel(1);

        // get_all_by_side should only return order 2 (active orders)
        let active = mgr.get_all_by_side(Side::Buy);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].oid, 2);

        // get_all_by_side_including_pending should return both
        let all = mgr.get_all_by_side_including_pending(Side::Buy);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_cancel_failure_reverts_state() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 100.0, 1.0));

        // Initiate cancel
        mgr.initiate_cancel(1);
        assert_eq!(mgr.get_order(1).unwrap().state, OrderState::CancelPending);

        // Cancel failed - should revert to Resting
        mgr.on_cancel_failed(1);
        assert_eq!(mgr.get_order(1).unwrap().state, OrderState::Resting);

        // Order should be visible to get_by_side again
        assert!(mgr.get_by_side(Side::Buy).is_some());
    }

    #[test]
    fn test_reconcile_ladder_empty_to_ladder() {
        let mgr = OrderManager::new();
        let ladder = Ladder {
            bids: vec![
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
            ],
            asks: vec![LadderLevel {
                price: 101.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
        };

        let actions = mgr.reconcile_ladder(&ladder, 5);

        // Should place 3 orders (2 bids, 1 ask)
        assert_eq!(actions.len(), 3);
        let place_count = actions
            .iter()
            .filter(|a| matches!(a, LadderAction::Place { .. }))
            .count();
        assert_eq!(place_count, 3);
    }

    #[test]
    fn test_reconcile_ladder_orders_match() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 99.0, 1.0));
        mgr.add_order(TrackedOrder::new(2, Side::Sell, 101.0, 1.0));

        let ladder = Ladder {
            bids: vec![LadderLevel {
                price: 99.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
            asks: vec![LadderLevel {
                price: 101.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
        };

        let actions = mgr.reconcile_ladder(&ladder, 5);

        // Orders match target levels - no actions needed
        assert!(actions.is_empty());
    }

    #[test]
    fn test_reconcile_ladder_cancel_stale() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 95.0, 1.0)); // Too far from target
        mgr.add_order(TrackedOrder::new(2, Side::Sell, 105.0, 1.0)); // Too far from target

        let ladder = Ladder {
            bids: vec![LadderLevel {
                price: 99.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
            asks: vec![LadderLevel {
                price: 101.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
        };

        let actions = mgr.reconcile_ladder(&ladder, 5); // 5 bps tolerance

        // Should cancel 2 stale orders and place 2 new ones
        assert_eq!(actions.len(), 4);
        let cancel_count = actions
            .iter()
            .filter(|a| matches!(a, LadderAction::Cancel { .. }))
            .count();
        let place_count = actions
            .iter()
            .filter(|a| matches!(a, LadderAction::Place { .. }))
            .count();
        assert_eq!(cancel_count, 2);
        assert_eq!(place_count, 2);
    }

    #[test]
    fn test_reconcile_ladder_partial_match() {
        let mut mgr = OrderManager::new();
        mgr.add_order(TrackedOrder::new(1, Side::Buy, 99.0, 1.0)); // Matches first level
                                                                   // No ask order

        let ladder = Ladder {
            bids: vec![
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
            ],
            asks: vec![LadderLevel {
                price: 101.0,
                size: 1.0,
                depth_bps: 10.0,
            }],
        };

        let actions = mgr.reconcile_ladder(&ladder, 5);

        // Should place 1 bid (98.0) and 1 ask (101.0)
        assert_eq!(actions.len(), 2);
        let place_count = actions
            .iter()
            .filter(|a| matches!(a, LadderAction::Place { .. }))
            .count();
        assert_eq!(place_count, 2);
    }
}
