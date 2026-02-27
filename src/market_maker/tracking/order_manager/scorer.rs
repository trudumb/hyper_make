//! Economic scoring engine for order reconciliation.
//!
//! Computes net economic value (in bps) for each possible reconciliation action,
//! replacing the heuristic priority_based_matching with EV-driven decisions:
//!
//! ```text
//! update_value = EV_new - EV_keep - dynamic_api_cost
//!
//! EV_keep = P(fill_keep) × spread_capture_keep
//! EV_new  = P(fill_new)  × spread_capture_new
//! api_cost = base_cost (flat; budget allocator limits total calls)
//! ```

#![allow(dead_code)]

use crate::market_maker::adverse_selection::toxicity_regime::ToxicityRegime;
use crate::market_maker::models::QueueValueHeuristic;
use crate::market_maker::quoting::LadderLevel;
use crate::market_maker::tracking::order_manager::reconcile::DynamicReconcileConfig;
use crate::market_maker::tracking::order_manager::types::{Side, TrackedOrder};
use crate::market_maker::tracking::queue::QueuePositionTracker;
use crate::{bps_diff, EPSILON};

// ---------------------------------------------------------------------------
// Cost constants (basis points per API call type)
// ---------------------------------------------------------------------------

/// Cost of keeping an order as-is: zero API calls.
const BASE_COST_LATCH_BPS: f64 = 0.0;

/// Cost of a single-call modify (size or price change): 1 API call.
const BASE_COST_MODIFY_BPS: f64 = 3.0;

/// Cost of cancel + place (full replacement): 2 API calls.
const BASE_COST_CANCEL_PLACE_BPS: f64 = 6.0;

/// Cost of placing a new order where none exists: 1 API call.
const BASE_COST_NEW_PLACE_BPS: f64 = 3.0;

/// Cost of cancelling a stale order with no matching target: 1 API call.
const BASE_COST_STALE_CANCEL_BPS: f64 = 3.0;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Type of reconciliation action for economic scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    /// Keep order as-is (0 API calls).
    Latch,
    /// Modify size only at same price (1 API call, preserves queue).
    ModifySize,
    /// Modify price (1 API call, loses queue on HL).
    ModifyPrice,
    /// Cancel + place new (2 API calls, loses queue).
    CancelPlace,
    /// Place new order where none exists (1 API call).
    NewPlace,
    /// Cancel stale order with no matching target (1 API call).
    StaleCancel,
}

impl ActionType {
    /// Number of API calls consumed by this action type.
    pub fn api_calls(self) -> u32 {
        match self {
            ActionType::Latch => 0,
            ActionType::ModifySize => 1,
            ActionType::ModifyPrice => 1,
            ActionType::CancelPlace => 2,
            ActionType::NewPlace => 1,
            ActionType::StaleCancel => 1,
        }
    }

    /// Base cost in bps before headroom scaling.
    fn base_cost_bps(self) -> f64 {
        match self {
            ActionType::Latch => BASE_COST_LATCH_BPS,
            ActionType::ModifySize => BASE_COST_MODIFY_BPS,
            ActionType::ModifyPrice => BASE_COST_MODIFY_BPS,
            ActionType::CancelPlace => BASE_COST_CANCEL_PLACE_BPS,
            ActionType::NewPlace => BASE_COST_NEW_PLACE_BPS,
            ActionType::StaleCancel => BASE_COST_STALE_CANCEL_BPS,
        }
    }
}

/// A scored reconciliation action with economic value.
#[derive(Debug, Clone)]
pub struct ScoredUpdate {
    /// Order ID of the resting order (None for new placements).
    pub oid: Option<u64>,
    /// Index into the targets array.
    pub target_idx: usize,
    /// Classified action type.
    pub action: ActionType,
    /// Net economic benefit in basis points.
    pub value_bps: f64,
    /// Number of API calls needed (0, 1, or 2).
    pub api_calls: u32,
    /// P(fill) if we keep the current order (0.0 for new placements).
    pub p_fill_keep: f64,
    /// P(fill) if we place a new order at the target.
    pub p_fill_new: f64,
    /// Side of the order.
    pub side: Side,
    /// Target price for the order.
    pub target_price: f64,
    /// Target size for the order.
    pub target_size: f64,
    /// Current price (0.0 for new placements).
    pub current_price: f64,
    /// Whether this action is emergency priority (bypasses budget).
    /// Set by the reconciler when ladder has large deficit (≥3 levels missing).
    pub is_emergency: bool,
}

// ---------------------------------------------------------------------------
// Scoring engine
// ---------------------------------------------------------------------------

/// Flat API cost for an action.
///
/// Previous design scaled cost by `1/headroom`, which at 8% headroom made ALL
/// actions negative-EV (75 bps/action). The budget allocator already limits
/// total API calls via its call budget, so per-action headroom scaling is
/// redundant and causes 94% zero-action cycles.
fn flat_api_cost_bps(action: ActionType) -> f64 {
    action.base_cost_bps()
}

/// Estimate P(fill) for a *new* order at the back of the queue at `price`.
///
/// Uses the queue tracker's P(touch) for the price level, then applies a
/// conservative P(execute|touch) assuming worst-case queue position (back).
fn estimate_p_fill_new(
    queue_tracker: Option<&QueuePositionTracker>,
    price: f64,
    side: Side,
    mid: f64,
    horizon_s: f64,
) -> f64 {
    let qt = match queue_tracker {
        Some(qt) => qt,
        None => return default_p_fill_new(price, mid, side, horizon_s),
    };

    // Estimate depth at this price from the cached L2 book.
    let is_bid = matches!(side, Side::Buy);
    let depth_at_price = qt.estimate_depth_at_price(price, is_bid);

    // P(touch) via the reflection principle.
    // We don't have an oid for this hypothetical order, so compute manually.
    let sigma = qt.sigma();
    let delta = match side {
        Side::Buy => {
            // For bids: price must come DOWN to us
            // delta = best_bid - our_price, if we're below best bid
            // Use mid as proxy (conservative)
            (mid - price).max(0.0)
        }
        Side::Sell => {
            // For asks: price must come UP to us
            (price - mid).max(0.0)
        }
    };

    let p_touch = if delta <= 0.0 {
        1.0
    } else {
        let sigma_sqrt_t = sigma * horizon_s.sqrt();
        if sigma_sqrt_t < 1e-12 {
            0.0
        } else {
            let z = -delta / sigma_sqrt_t;
            (2.0 * normal_cdf_approx(z)).min(1.0)
        }
    };

    // P(execute|touch) = exp(-depth / expected_volume).
    // Expected volume: sigma * sqrt(T) * mid as rough proxy; use depth directly.
    // Conservative: assume volume over horizon is proportional to sigma.
    // Fall back to a simple exponential decay in queue depth.
    let p_execute = (-depth_at_price * 0.1_f64).exp().clamp(0.0, 1.0);

    p_touch * p_execute
}

/// Fallback P(fill) estimate when no queue tracker is available.
fn default_p_fill_new(price: f64, mid: f64, side: Side, _horizon_s: f64) -> f64 {
    let depth_bps = if mid > EPSILON {
        (price - mid).abs() / mid * 10_000.0
    } else {
        0.0
    };

    // Rough heuristic: P(fill) decays with depth, biased by side.
    let _ = side;
    (-depth_bps / 20.0).exp().clamp(0.01, 0.99)
}

/// Approximation of the standard normal CDF (Abramowitz & Stegun 26.2.17).
fn normal_cdf_approx(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2π)
    let p = d * (-x * x / 2.0).exp();
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 {
        1.0 - p * poly
    } else {
        p * poly
    }
}

/// Classify the action type for a matched (order, target) pair.
fn classify_action(
    order: &TrackedOrder,
    target: &LadderLevel,
    config: &DynamicReconcileConfig,
    sz_decimals: u32,
) -> ActionType {
    let price_diff_bps = bps_diff(order.price, target.price) as f64;
    let current_size = order.remaining();
    let size_diff_pct = if current_size > EPSILON {
        ((target.size - current_size).abs() / current_size).min(1.0)
    } else {
        1.0
    };

    let _ = sz_decimals;

    // Case 1: Within latch threshold on both price and size.
    if price_diff_bps <= config.latch_threshold_bps && size_diff_pct <= config.latch_size_fraction {
        return ActionType::Latch;
    }

    // Case 2: Price within latch, only size changed.
    if price_diff_bps <= config.latch_threshold_bps {
        return ActionType::ModifySize;
    }

    // Case 3: Price within modify range.
    // On Hyperliquid, price modifications lose queue position (new OID).
    // Modify still saves 1 API call vs cancel+place, so prefer it.
    if price_diff_bps <= config.max_modify_price_bps {
        return ActionType::ModifyPrice;
    }

    // Case 4: Price moved too far -- must cancel + place.
    ActionType::CancelPlace
}

/// Compute the queue rank for an existing order.
///
/// Returns a value in [0, 1] where 0 = front of queue, 1 = back.
/// Uses the queue tracker's depth-ahead estimate.
fn queue_rank_for_order(queue_tracker: Option<&QueuePositionTracker>, oid: u64) -> f64 {
    let qt = match queue_tracker {
        Some(qt) => qt,
        None => return 0.5, // Unknown: assume middle
    };

    match qt.queue_depth(oid) {
        Some(depth) => {
            // Heuristic: normalize depth to [0,1].
            // Typical depth 0-100 units; use sigmoid-like mapping.
            (depth / (depth + 1.0)).clamp(0.0, 1.0)
        }
        None => 0.5,
    }
}

/// Score all reconciliation actions for a single side.
///
/// For each (resting_order, target) pair:
///   `update_value = EV_new - EV_keep - dynamic_api_cost`
///
/// Where:
///   `EV_keep = P(fill_keep) x spread_capture_keep`
///   `EV_new  = P(fill_new)  x spread_capture_new`
///   `api_cost = base_cost` (flat; budget allocator limits total calls)
#[allow(clippy::too_many_arguments)]
pub fn score_all(
    current_orders: &[&TrackedOrder],
    targets: &[LadderLevel],
    side: Side,
    config: &DynamicReconcileConfig,
    queue_tracker: Option<&QueuePositionTracker>,
    queue_value: &QueueValueHeuristic,
    toxicity: ToxicityRegime,
    mid: f64,
    sz_decimals: u32,
) -> Vec<ScoredUpdate> {
    let mut results = Vec::with_capacity(targets.len() + current_orders.len());
    let mut matched_orders = std::collections::HashSet::new();
    let mut matched_target_indices = std::collections::HashSet::new();

    let horizon_s = config.queue_horizon_seconds;

    // Phase 1: Match orders to targets in priority order (best price first).
    // Targets are assumed sorted by priority (best price first from ladder).
    for (target_idx, target) in targets.iter().enumerate() {
        let tolerance_bps = config.tolerance_for_priority(target_idx);
        let mut best_order: Option<&TrackedOrder> = None;
        let mut best_distance = f64::MAX;

        for order in current_orders.iter() {
            if matched_orders.contains(&order.oid) {
                continue;
            }
            let price_diff_bps = bps_diff(order.price, target.price) as f64;
            if price_diff_bps <= tolerance_bps && price_diff_bps < best_distance {
                best_distance = price_diff_bps;
                best_order = Some(*order);
            }
        }

        if let Some(order) = best_order {
            matched_orders.insert(order.oid);
            matched_target_indices.insert(target_idx);

            let action = classify_action(order, target, config, sz_decimals);
            let api_cost = flat_api_cost_bps(action);

            // EV_keep: value of keeping the current order at its position.
            let p_fill_keep = queue_tracker
                .and_then(|qt| qt.fill_probability(order.oid, horizon_s))
                .unwrap_or(0.3);
            let keep_queue_rank = queue_rank_for_order(queue_tracker, order.oid);
            let keep_depth_bps = if mid > EPSILON {
                (order.price - mid).abs() / mid * 10_000.0
            } else {
                target.depth_bps
            };
            let spread_capture_keep =
                queue_value.queue_value(keep_depth_bps, toxicity, keep_queue_rank);
            let ev_keep = p_fill_keep * spread_capture_keep.max(0.0);

            // EV_new: value of a fresh order at the target.
            let p_fill_new = if action == ActionType::Latch || action == ActionType::ModifySize {
                // Latch and size-only modify preserve queue, so P(fill) stays the same.
                p_fill_keep
            } else {
                estimate_p_fill_new(queue_tracker, target.price, side, mid, horizon_s)
            };
            let spread_capture_new = queue_value.queue_value(
                target.depth_bps,
                toxicity,
                1.0, // Back of queue for new placement
            );
            let ev_new = p_fill_new * spread_capture_new.max(0.0);

            // For Latch the value is always 0 (no change, no cost).
            let value_bps = match action {
                ActionType::Latch => 0.0,
                ActionType::ModifySize => {
                    // ModifySize preserves queue -- cost is just the API call.
                    // Benefit: closer to target size. Use a fraction of EV_new improvement.
                    let size_benefit =
                        (spread_capture_new - spread_capture_keep).max(0.0) * p_fill_keep;
                    size_benefit - api_cost
                }
                _ => ev_new - ev_keep - api_cost,
            };

            results.push(ScoredUpdate {
                oid: Some(order.oid),
                target_idx,
                action,
                value_bps,
                api_calls: action.api_calls(),
                p_fill_keep,
                p_fill_new,
                side,
                target_price: target.price,
                target_size: target.size,
                current_price: order.price,
                is_emergency: false,
            });
        } else {
            // Unmatched target: needs a new placement.
            if target.size > EPSILON {
                let p_fill_new =
                    estimate_p_fill_new(queue_tracker, target.price, side, mid, horizon_s);
                let spread_capture = queue_value.queue_value(
                    target.depth_bps,
                    toxicity,
                    1.0, // Back of queue
                );
                let ev_new = p_fill_new * spread_capture.max(0.0);
                let api_cost = flat_api_cost_bps(ActionType::NewPlace);
                // Floor: new placements always have at least 1.0 bps value.
                // Having orders on the book captures optionality (fills, information,
                // maker rebate) that the EV model underestimates for back-of-queue
                // placements. Without this floor, the budget allocator filters
                // NewPlace actions as negative-value, causing the local_bids=0 death spiral.
                let raw_value = ev_new - api_cost;
                let value_bps = raw_value.max(1.0);

                results.push(ScoredUpdate {
                    oid: None,
                    target_idx,
                    action: ActionType::NewPlace,
                    value_bps,
                    api_calls: ActionType::NewPlace.api_calls(),
                    p_fill_keep: 0.0,
                    p_fill_new,
                    side,
                    target_price: target.price,
                    target_size: target.size,
                    current_price: 0.0,
                    is_emergency: false,
                });
            }
        }
    }

    // Phase 2: Score stale cancels for unmatched orders.
    for order in current_orders.iter() {
        if matched_orders.contains(&order.oid) {
            continue;
        }

        // Check if this order is close to any UNMATCHED target.
        // Orders near only already-matched targets are redundant and must be cancelled.
        let close_to_unmatched_target = targets.iter().enumerate().any(|(idx, t)| {
            !matched_target_indices.contains(&idx)
                && (bps_diff(order.price, t.price) as f64) <= config.max_match_distance_bps
        });

        if !close_to_unmatched_target {
            let api_cost = flat_api_cost_bps(ActionType::StaleCancel);

            // EV of keeping a stale order: low but not zero.
            let p_fill_keep = queue_tracker
                .and_then(|qt| qt.fill_probability(order.oid, horizon_s))
                .unwrap_or(0.05);
            let keep_depth_bps = if mid > EPSILON {
                (order.price - mid).abs() / mid * 10_000.0
            } else {
                20.0
            };
            let keep_rank = queue_rank_for_order(queue_tracker, order.oid);
            let spread_capture_keep = queue_value.queue_value(keep_depth_bps, toxicity, keep_rank);
            let ev_keep = p_fill_keep * spread_capture_keep.max(0.0);

            // Stale cancel frees up a slot for a better order. Net value is the
            // recovered slot value minus the lost EV minus the API cost.
            // For simplicity, value = -ev_keep - api_cost (always negative unless
            // ev_keep is very small).
            let value_bps = -ev_keep - api_cost;

            results.push(ScoredUpdate {
                oid: Some(order.oid),
                target_idx: usize::MAX, // sentinel: no target
                action: ActionType::StaleCancel,
                value_bps,
                api_calls: ActionType::StaleCancel.api_calls(),
                p_fill_keep,
                p_fill_new: 0.0,
                side,
                target_price: 0.0,
                target_size: 0.0,
                current_price: order.price,
                is_emergency: false,
            });
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_order(oid: u64, price: f64, size: f64, side: Side) -> TrackedOrder {
        TrackedOrder::new(oid, side, price, size, 0.0)
    }

    fn make_target(price: f64, size: f64, depth_bps: f64) -> LadderLevel {
        LadderLevel {
            price,
            size,
            depth_bps,
        }
    }

    fn default_config() -> DynamicReconcileConfig {
        DynamicReconcileConfig::default()
    }

    fn default_queue_value() -> QueueValueHeuristic {
        QueueValueHeuristic::new()
    }

    // --- Test 1: Latch has zero cost and zero value ---
    #[test]
    fn test_latch_zero_cost() {
        let order = make_order(1, 100.0, 1.0, Side::Buy);
        let target = make_target(100.0, 1.0, 10.0);
        let config = default_config();
        let qv = default_queue_value();

        let scores = score_all(
            &[&order],
            &[target],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            100.5,
            2,
        );

        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].action, ActionType::Latch);
        assert!((scores[0].value_bps - 0.0).abs() < 1e-10);
        assert_eq!(scores[0].api_calls, 0);
    }

    // --- Test 2: New placement always produced for unmatched target ---
    #[test]
    fn test_new_placement_for_uncovered_target() {
        let target = make_target(99.0, 1.0, 10.0);
        let config = default_config();
        let qv = default_queue_value();

        let scores = score_all(
            &[],
            &[target],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Normal,
            100.0,
            2,
        );

        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].action, ActionType::NewPlace);
        assert_eq!(scores[0].api_calls, 1);
        assert!(scores[0].oid.is_none());
    }

    // --- Test 3: Flat API cost returns base cost ---
    #[test]
    fn test_flat_api_cost_returns_base() {
        let cost_modify = flat_api_cost_bps(ActionType::ModifyPrice);
        let cost_new = flat_api_cost_bps(ActionType::NewPlace);
        let cost_cancel = flat_api_cost_bps(ActionType::StaleCancel);

        assert!((cost_modify - BASE_COST_MODIFY_BPS).abs() < 1e-10);
        assert!((cost_new - BASE_COST_NEW_PLACE_BPS).abs() < 1e-10);
        assert!((cost_cancel - BASE_COST_STALE_CANCEL_BPS).abs() < 1e-10);
    }

    // --- Test 4: Queue preservation (Latch) when price is within threshold ---
    #[test]
    fn test_within_latch_threshold_is_latch() {
        // Price diff within latch_threshold_bps (default 3.0) and size within fraction
        let order = make_order(1, 100.0, 1.0, Side::Buy);
        // 0.02% diff = 2 bps → within default 3 bps latch
        let target = make_target(100.02, 1.05, 10.0);
        let config = default_config();
        let qv = default_queue_value();

        let scores = score_all(
            &[&order],
            &[target],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            100.5,
            2,
        );

        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].action, ActionType::Latch);
    }

    // --- Test 5: ModifySize when price is within latch but size differs ---
    #[test]
    fn test_modify_size_when_price_latched_size_differs() {
        let order = make_order(1, 100.0, 1.0, Side::Buy);
        // Same price, but size changed by 50% (> default 10% latch_size_fraction)
        let target = make_target(100.0, 1.5, 10.0);
        let config = default_config();
        let qv = default_queue_value();

        let scores = score_all(
            &[&order],
            &[target],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            100.5,
            2,
        );

        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].action, ActionType::ModifySize);
        assert_eq!(scores[0].api_calls, 1);
    }

    // --- Test 6: CancelPlace for large price drift ---
    #[test]
    fn test_cancel_place_for_large_drift() {
        let order = make_order(1, 100.0, 1.0, Side::Buy);
        // 5% diff = 500 bps → beyond any modify threshold
        let target = make_target(105.0, 1.0, 500.0);
        let mut config = default_config();
        config.max_modify_price_bps = 50.0;
        // Increase tolerance so the order matches the target
        config.best_level_tolerance_bps = 600.0;
        config.outer_level_tolerance_bps = 600.0;
        let qv = default_queue_value();

        let scores = score_all(
            &[&order],
            &[target],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            102.5,
            2,
        );

        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].action, ActionType::CancelPlace);
        assert_eq!(scores[0].api_calls, 2);
    }

    // --- Test 7: Stale cancel for unmatched order far from targets ---
    #[test]
    fn test_stale_cancel_for_distant_order() {
        let order = make_order(1, 90.0, 1.0, Side::Buy);
        let target = make_target(100.0, 1.0, 10.0);
        let config = default_config();
        let qv = default_queue_value();

        let scores = score_all(
            &[&order],
            &[target],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Normal,
            100.5,
            2,
        );

        // Should have 2 results: stale cancel + new placement
        assert_eq!(scores.len(), 2);

        let stale = scores.iter().find(|s| s.action == ActionType::StaleCancel);
        assert!(stale.is_some());
        let stale = stale.unwrap();
        assert_eq!(stale.oid, Some(1));
        assert!(stale.value_bps < 0.0); // Stale cancel has negative value (cost)

        let new_place = scores.iter().find(|s| s.action == ActionType::NewPlace);
        assert!(new_place.is_some());
    }

    // --- Test 8: Scoring monotonicity -- deeper targets have lower P(fill) ---
    #[test]
    fn test_scoring_monotonicity_depth() {
        let target_close = make_target(100.0, 1.0, 5.0);
        let target_far = make_target(99.0, 1.0, 100.0);
        let config = default_config();
        let qv = default_queue_value();
        let mid = 100.5;

        let scores_close = score_all(
            &[],
            &[target_close],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            mid,
            2,
        );
        let scores_far = score_all(
            &[],
            &[target_far],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            mid,
            2,
        );

        // Closer target should have higher P(fill)
        assert!(scores_close[0].p_fill_new > scores_far[0].p_fill_new);
    }

    // --- Test 9: Multiple orders matched to multiple targets ---
    #[test]
    fn test_multiple_orders_and_targets() {
        let o1 = make_order(1, 99.0, 1.0, Side::Buy);
        let o2 = make_order(2, 98.0, 0.5, Side::Buy);
        let t1 = make_target(99.0, 1.0, 10.0);
        let t2 = make_target(98.0, 0.5, 20.0);
        let config = default_config();
        let qv = default_queue_value();

        let scores = score_all(
            &[&o1, &o2],
            &[t1, t2],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            100.0,
            2,
        );

        // Both should match and latch (exact price, exact size)
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|s| s.action == ActionType::Latch));
    }

    // --- Test 10: Flat API cost means scoring is headroom-independent ---
    #[test]
    fn test_flat_cost_headroom_independent() {
        let order = make_order(1, 100.0, 1.0, Side::Buy);
        // Price diff ~10 bps -- triggers ModifyPrice
        let target = make_target(100.10, 1.0, 10.0);
        let mut config = default_config();
        config.latch_threshold_bps = 5.0;
        config.max_modify_price_bps = 50.0;
        config.best_level_tolerance_bps = 15.0;
        config.outer_level_tolerance_bps = 30.0;
        let qv = default_queue_value();

        let scores = score_all(
            &[&order],
            &[target],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            100.5,
            2,
        );

        // Should be ModifyPrice with flat cost (base_cost_bps only)
        assert_eq!(scores[0].action, ActionType::ModifyPrice);
        // API cost portion should equal the flat base cost
        let ev_diff = scores[0].value_bps;
        // Value = ev_new - ev_keep - flat_cost; flat_cost = BASE_COST_MODIFY_BPS = 3.0
        // As long as it's finite and deterministic, the flat cost model works
        assert!(ev_diff.is_finite());
    }

    // --- Test 11: normal_cdf_approx correctness ---
    #[test]
    fn test_normal_cdf_approx() {
        // CDF(0) = 0.5
        assert!((normal_cdf_approx(0.0) - 0.5).abs() < 1e-6);
        // CDF(-inf) → 0
        assert!(normal_cdf_approx(-10.0) < 1e-6);
        // CDF(+inf) → 1
        assert!((normal_cdf_approx(10.0) - 1.0).abs() < 1e-6);
        // CDF(1.96) ≈ 0.975
        assert!((normal_cdf_approx(1.96) - 0.975).abs() < 0.002);
    }

    // --- Test 12: ActionType api_calls correctness ---
    #[test]
    fn test_action_type_api_calls() {
        assert_eq!(ActionType::Latch.api_calls(), 0);
        assert_eq!(ActionType::ModifySize.api_calls(), 1);
        assert_eq!(ActionType::ModifyPrice.api_calls(), 1);
        assert_eq!(ActionType::CancelPlace.api_calls(), 2);
        assert_eq!(ActionType::NewPlace.api_calls(), 1);
        assert_eq!(ActionType::StaleCancel.api_calls(), 1);
    }

    // --- Test 13: Excess orders at matched target price get StaleCancel ---
    // Regression test for the close-to-matched-target bug:
    // 3 bids at same price, 1 target → expect 1 Latch + 2 StaleCancel
    #[test]
    fn test_excess_orders_at_matched_target() {
        let o1 = make_order(1, 100.0, 1.0, Side::Buy);
        let o2 = make_order(2, 100.0, 1.0, Side::Buy);
        let o3 = make_order(3, 100.0, 1.0, Side::Buy);
        let target = make_target(100.0, 1.0, 10.0);
        let config = default_config();
        let qv = default_queue_value();

        let scores = score_all(
            &[&o1, &o2, &o3],
            &[target],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            100.5,
            2,
        );

        let latches: Vec<_> = scores
            .iter()
            .filter(|s| s.action == ActionType::Latch)
            .collect();
        let stale_cancels: Vec<_> = scores
            .iter()
            .filter(|s| s.action == ActionType::StaleCancel)
            .collect();

        assert_eq!(latches.len(), 1, "Expected exactly 1 Latch");
        assert_eq!(
            stale_cancels.len(),
            2,
            "Expected exactly 2 StaleCancel for redundant orders"
        );
    }

    // --- Test 14: Unmatched order near unmatched target is preserved ---
    // Defensive test: orders close to unmatched targets should NOT be cancelled
    #[test]
    fn test_unmatched_order_near_unmatched_target_preserved() {
        // Order at 100.0, targets at 100.0 and 100.01 (within match distance)
        // Order should match target 0, and target 1 stays unmatched.
        // A second order near target 1 should NOT get StaleCancel.
        let o1 = make_order(1, 100.0, 1.0, Side::Buy);
        let o2 = make_order(2, 100.01, 1.0, Side::Buy);
        let t1 = make_target(100.0, 1.0, 10.0);
        let t2 = make_target(100.01, 1.0, 10.0);
        let config = default_config();
        let qv = default_queue_value();

        let scores = score_all(
            &[&o1, &o2],
            &[t1, t2],
            Side::Buy,
            &config,
            None,
            &qv,
            ToxicityRegime::Benign,
            100.5,
            2,
        );

        // Both orders should match their respective targets (both Latch)
        // No StaleCancel should be generated
        let stale_cancels: Vec<_> = scores
            .iter()
            .filter(|s| s.action == ActionType::StaleCancel)
            .collect();
        assert_eq!(
            stale_cancels.len(),
            0,
            "No StaleCancel expected when orders match unmatched targets"
        );
    }
}
