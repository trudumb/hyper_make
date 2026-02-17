//! Market maker orchestrator - event loop, message handling, and order management.
//!
//! This module contains the runtime logic for the market maker, split into logical submodules:
//! - `event_loop`: Main event loop and startup synchronization
//! - `handlers`: WebSocket message handlers
//! - `quote_engine`: Quote generation and update logic
//! - `reconcile`: Order reconciliation and ladder management
//! - `order_ops`: Order placement, cancellation, and tracking
//! - `recovery`: Recovery logic, safety sync, and state refresh
//! - `event_accumulator`: Event-driven quote update triggering (Phase 3: Churn Reduction)

pub(crate) mod budget_allocator;
pub(crate) mod diagnosis;
mod event_loop;
pub(crate) mod event_accumulator;
mod handlers;
mod order_ops;
mod quote_engine;
pub(crate) mod reconcile;
mod recovery;

#[cfg(test)]
mod tests;

pub(crate) use event_accumulator::EventAccumulator;

use super::*;

/// Result of recovery check - indicates what action was taken.
pub(crate) struct RecoveryAction {
    /// Whether to skip normal quoting this cycle
    pub skip_normal_quoting: bool,
}

/// Helper to convert Side to string for logging.
pub(crate) fn side_str(side: Side) -> &'static str {
    match side {
        Side::Buy => "BUY",
        Side::Sell => "SELL",
    }
}

/// Partition LadderActions into separate collections by action type.
///
/// Returns (cancels, modifies, places) where:
/// - cancels: Vec<u64> of order IDs to cancel
/// - modifies: Vec<ModifySpec> of orders to modify
/// - places: Vec<(f64, f64)> of (price, size) for new orders
pub(super) fn partition_ladder_actions(
    actions: &[tracking::LadderAction],
    side: Side,
) -> (Vec<u64>, Vec<ModifySpec>, Vec<(f64, f64)>) {
    use tracking::LadderAction;

    let mut cancels = Vec::new();
    let mut modifies = Vec::new();
    let mut places = Vec::new();
    let is_buy = side == Side::Buy;

    for action in actions {
        match action {
            LadderAction::Cancel { oid } => {
                cancels.push(*oid);
            }
            LadderAction::Modify {
                oid,
                new_price,
                new_size,
                ..
            } => {
                modifies.push(ModifySpec {
                    oid: *oid,
                    new_price: *new_price,
                    new_size: *new_size,
                    is_buy,
                    post_only: true, // ALO on modify
                });
            }
            LadderAction::Place { price, size, .. } => {
                places.push((*price, *size));
            }
        }
    }

    (cancels, modifies, places)
}
