//! Ladder reconciliation logic.
//!
//! Implements diffing between current orders and target ladder to generate
//! minimal cancel/place/modify actions for improved spread capturing.

use std::collections::HashSet;

use crate::{bps_diff, EPSILON};

use super::impulse_filter::{ImpulseDecision, ImpulseFilter};
use super::types::{LadderAction, Side, TrackedOrder};
use crate::market_maker::quoting::LadderLevel;
use crate::market_maker::tracking::queue::QueuePositionTracker;

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
    /// Enable queue-aware reconciliation that uses QueuePositionTracker
    /// to make refresh decisions based on expected fill value
    pub use_queue_aware: bool,
    /// Time horizon in seconds for queue fill probability calculation
    /// Orders with low P(fill) within this horizon may be refreshed
    pub queue_horizon_seconds: f64,
    /// Enable impulse filtering (Δλ-based update gating).
    /// When true, only updates orders when fill probability improvement exceeds threshold.
    pub use_impulse_filter: bool,
}

impl Default for ReconcileConfig {
    fn default() -> Self {
        Self {
            // NOTE: On Hyperliquid, price modifications always reset queue position (new OID).
            // Only SIZE-only modifications preserve queue. Therefore, these tolerances
            // primarily affect API call frequency, not queue preservation.
            max_modify_price_bps: 50, // Modify if price ≤ 50 bps change (was 10)
            max_modify_size_pct: 0.50, // Modify if size ≤ 50% change
            skip_price_tolerance_bps: 10, // Skip if price ≤ 10 bps (was 1 - reduces churn ~50%)
            skip_size_tolerance_pct: 0.05, // Skip if size ≤ 5% (unchanged)
            use_queue_aware: false,   // Disabled by default until validated
            queue_horizon_seconds: 1.0, // 1-second fill horizon
            use_impulse_filter: false, // Disabled by default until validated
        }
    }
}

/// Dynamic reconciliation config with thresholds derived from stochastic optimal spread.
///
/// Unlike static `ReconcileConfig`, this is computed fresh each quote cycle from
/// market parameters (gamma, kappa, sigma). This ensures matching thresholds
/// adapt to current market conditions.
#[derive(Debug, Clone)]
pub struct DynamicReconcileConfig {
    /// Tight tolerance for best level (must be accurate) - optimal_spread / 4
    pub best_level_tolerance_bps: f64,
    /// Looser tolerance for outer levels - optimal_spread / 2
    pub outer_level_tolerance_bps: f64,
    /// Maximum delta (bps) to consider any match - beyond this, force cancel+place
    pub max_match_distance_bps: f64,
    /// Fill probability threshold for queue preservation
    /// If P(fill) > this, preserve order even if price is slightly off
    pub queue_value_threshold: f64,
    /// Time horizon for fill probability calculation
    pub queue_horizon_seconds: f64,
    /// The stochastic optimal spread (half-spread per side, in bps)
    /// Computed as: δ* = (1/γ) × ln(1 + γ/κ) × 10000
    pub optimal_spread_bps: f64,
    /// Enable priority-based matching (ensures best levels are covered first)
    pub use_priority_matching: bool,
}

impl Default for DynamicReconcileConfig {
    fn default() -> Self {
        Self {
            best_level_tolerance_bps: 5.0,   // Tight for best level
            outer_level_tolerance_bps: 15.0, // Looser for outer levels
            max_match_distance_bps: 30.0,    // Beyond this, no match
            queue_value_threshold: 0.3,      // 30% P(fill) = valuable queue
            queue_horizon_seconds: 1.0,
            optimal_spread_bps: 20.0,        // Default ~20 bps
            use_priority_matching: true,
        }
    }
}

impl DynamicReconcileConfig {
    /// Create from market parameters (call each quote cycle).
    ///
    /// Derives matching thresholds from the stochastic optimal spread formula:
    /// δ* = (1/γ) × ln(1 + γ/κ)
    ///
    /// This ensures thresholds adapt to current market conditions:
    /// - Low gamma (calm market) → tight thresholds
    /// - High gamma (volatile/risky) → looser thresholds
    pub fn from_market_params(
        gamma: f64,
        kappa: f64,
        _sigma: f64,
        queue_horizon: f64,
    ) -> Self {
        // Clamp inputs to avoid division by zero
        let gamma_safe = gamma.max(0.001);
        let kappa_safe = kappa.max(1.0);

        // GLFT optimal spread: δ* = (1/γ) × ln(1 + γ/κ)
        let optimal_spread_frac = (1.0 / gamma_safe) * (1.0 + gamma_safe / kappa_safe).ln();
        let optimal_spread_bps = optimal_spread_frac * 10_000.0;

        // Derive thresholds from optimal spread
        // Best level: tight (1/4 of optimal spread)
        let best_level_tolerance_bps = (optimal_spread_bps / 4.0).clamp(2.0, 10.0);
        // Outer levels: looser (1/2 of optimal spread)
        let outer_level_tolerance_bps = (optimal_spread_bps / 2.0).clamp(5.0, 25.0);
        // Maximum match distance: 1.5x optimal spread
        let max_match_distance_bps = (optimal_spread_bps * 1.5).clamp(15.0, 50.0);

        Self {
            best_level_tolerance_bps,
            outer_level_tolerance_bps,
            max_match_distance_bps,
            queue_value_threshold: 0.3,
            queue_horizon_seconds: queue_horizon,
            optimal_spread_bps,
            use_priority_matching: true,
        }
    }

    /// Get tolerance for a given priority level (0 = best, higher = further out)
    #[inline]
    pub fn tolerance_for_priority(&self, priority: usize) -> f64 {
        if priority == 0 {
            self.best_level_tolerance_bps
        } else {
            self.outer_level_tolerance_bps
        }
    }
}

/// Statistics from reconciliation.
#[derive(Debug, Clone, Default)]
pub struct ReconcileStats {
    /// Number of orders skipped (within tolerance).
    pub skipped_count: usize,
    /// Number of orders modified (preserve queue).
    pub modified_count: usize,
    /// Number of orders cancelled.
    pub cancelled_count: usize,
    /// Number of new orders placed.
    pub placed_count: usize,
    /// Number of updates blocked by impulse filter (Δλ too small).
    pub impulse_filtered_count: usize,
    /// Number of orders locked by impulse filter (high P(fill)).
    pub queue_locked_count: usize,
}

impl ReconcileStats {
    /// Total number of API actions (non-skip).
    pub fn total_actions(&self) -> usize {
        self.modified_count + self.cancelled_count + self.placed_count
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
///
/// When impulse filtering is enabled (via config + queue_tracker + impulse_filter):
/// - Orders with high P(fill) are "locked" and protected from updates
/// - Updates only proceed if Δλ (fill probability improvement) exceeds threshold
pub fn reconcile_side_smart(
    current: &[&TrackedOrder],
    target: &[LadderLevel],
    side: Side,
    config: &ReconcileConfig,
) -> Vec<LadderAction> {
    let (actions, _stats) = reconcile_side_smart_with_impulse(
        current, target, side, config, None, // No queue tracker
        None, // No impulse filter
        None, // No mid price
    );
    actions
}

/// Smart reconciliation with optional impulse filtering.
///
/// Extended version that accepts queue tracker and impulse filter for
/// Δλ-based update gating. Returns both actions and reconciliation statistics.
///
/// # Arguments
/// * `current` - Current orders on this side
/// * `target` - Target ladder levels
/// * `side` - Buy or Sell side
/// * `config` - Reconciliation configuration
/// * `queue_tracker` - Optional queue position tracker for P(fill) calculation
/// * `impulse_filter` - Optional impulse filter for Δλ gating
/// * `mid_price` - Optional mid price for new order P(fill) estimation
pub fn reconcile_side_smart_with_impulse(
    current: &[&TrackedOrder],
    target: &[LadderLevel],
    side: Side,
    config: &ReconcileConfig,
    queue_tracker: Option<&QueuePositionTracker>,
    mut impulse_filter: Option<&mut ImpulseFilter>,
    mid_price: Option<f64>,
) -> (Vec<LadderAction>, ReconcileStats) {
    let mut actions = Vec::new();
    let mut stats = ReconcileStats::default();
    let mut matched_targets: HashSet<usize> = HashSet::new();
    let mut matched_orders: HashSet<u64> = HashSet::new();

    // Determine if impulse filtering is active
    let impulse_active = config.use_impulse_filter
        && queue_tracker.is_some()
        && impulse_filter.is_some()
        && mid_price.is_some();

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
            stats.skipped_count += 1;
            continue;
        }

        // Apply impulse filter if active
        if impulse_active {
            let qt = queue_tracker.unwrap();
            let filter = impulse_filter.as_mut().unwrap();
            let mid = mid_price.unwrap();

            let decision = filter.evaluate(
                qt,
                m.order.oid,
                m.order.price,
                target_level.price,
                m.price_diff_bps,
                mid,
                side == Side::Buy,
            );

            match decision {
                ImpulseDecision::Skip => {
                    stats.impulse_filtered_count += 1;
                    stats.skipped_count += 1;
                    continue; // Δλ too small, skip update
                }
                ImpulseDecision::Locked => {
                    stats.queue_locked_count += 1;
                    stats.skipped_count += 1;
                    continue; // High P(fill), don't disturb
                }
                ImpulseDecision::Update => {
                    // Proceed with normal logic
                }
            }
        }

        // Standard MODIFY vs CANCEL+PLACE decision
        if m.price_diff_bps <= config.max_modify_price_bps as f64
            && m.size_diff_pct <= config.max_modify_size_pct
        {
            // MODIFY - small change, preserve queue position
            actions.push(LadderAction::Modify {
                oid: m.order.oid,
                new_price: target_level.price,
                new_size: target_level.size,
                side,
            });
            stats.modified_count += 1;
        } else {
            // CANCEL + PLACE - too large a change, fresh queue
            actions.push(LadderAction::Cancel { oid: m.order.oid });
            stats.cancelled_count += 1;
            if target_level.size > EPSILON {
                actions.push(LadderAction::Place {
                    side,
                    price: target_level.price,
                    size: target_level.size,
                });
                stats.placed_count += 1;
            }
        }
    }

    // Phase 2: Cancel any unmatched orders
    for order in current {
        if !matched_orders.contains(&order.oid) {
            actions.push(LadderAction::Cancel { oid: order.oid });
            stats.cancelled_count += 1;
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
            stats.placed_count += 1;
        }
    }

    (actions, stats)
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

/// Priority-based matching: ensures best levels are covered first.
///
/// Unlike `find_best_matches`, this function:
/// 1. Processes targets in priority order (best price first)
/// 2. Uses dynamic thresholds from stochastic optimal spread
/// 3. For unmatched targets, generates PLACE actions (critical coverage gaps)
/// 4. For orders not matching any target, generates CANCEL actions
///
/// This prevents the bug where stale far orders "steal" matches from near targets.
pub fn priority_based_matching(
    current: &[&TrackedOrder],
    targets: &[LadderLevel],
    side: Side,
    config: &DynamicReconcileConfig,
    queue_tracker: Option<&QueuePositionTracker>,
) -> Vec<LadderAction> {
    use tracing::debug;
    
    let mut actions = Vec::new();
    let mut matched_orders: HashSet<u64> = HashSet::new();
    let mut matched_targets: HashSet<usize> = HashSet::new();

    // Phase 1: Match orders to targets in PRIORITY ORDER (best price first)
    // For bids: highest price = best = priority 0
    // For asks: lowest price = best = priority 0
    // Targets are assumed to be in priority order from ladder generation
    for (priority, target) in targets.iter().enumerate() {
        let tolerance_bps = config.tolerance_for_priority(priority);
        let mut best_order: Option<&TrackedOrder> = None;
        let mut best_distance = f64::MAX;

        // Find the best unmatched order within tolerance
        for order in current.iter() {
            if matched_orders.contains(&order.oid) {
                continue;
            }

            let price_diff_bps = bps_diff(order.price, target.price) as f64;

            // Only consider orders within tolerance for this priority level
            if price_diff_bps <= tolerance_bps && price_diff_bps < best_distance {
                best_distance = price_diff_bps;
                best_order = Some(*order);
            }
        }

        if let Some(order) = best_order {
            // Order found within tolerance - check if we should preserve it
            matched_orders.insert(order.oid);
            matched_targets.insert(priority);

            // Check queue value using EV comparison if tracker available
            // Spread capture estimate: use target depth as proxy for spread capture
            let spread_capture_bps = target.depth_bps.max(config.optimal_spread_bps / 2.0);
            let (should_preserve, reason) = if let Some(qt) = queue_tracker {
                qt.should_preserve_order(
                    order.oid,
                    target.price,
                    config.queue_horizon_seconds,
                    spread_capture_bps,
                )
            } else {
                (false, "no_tracker")
            };

            if should_preserve {
                // Queue position is valuable - skip this order (preserve it)
                debug!(
                    oid = order.oid,
                    price = order.price,
                    target_price = target.price,
                    reason = reason,
                    "Priority matching: preserving order due to queue value"
                );
            } else {
                // Check if we need to modify (price or size difference)
                let size_diff_pct = if order.remaining() > EPSILON {
                    ((order.remaining() - target.size).abs() / order.remaining()).min(1.0)
                } else {
                    1.0
                };

                // If within very tight tolerance, skip entirely
                if best_distance <= 2.0 && size_diff_pct <= 0.05 {
                    // Perfect match - no action needed
                    continue;
                }

                // Otherwise, cancel and place new (MODIFY resets queue on HL anyway)
                if best_distance > config.best_level_tolerance_bps || size_diff_pct > 0.10 {
                    actions.push(LadderAction::Cancel { oid: order.oid });
                    if target.size > EPSILON {
                        actions.push(LadderAction::Place {
                            side,
                            price: target.price,
                            size: target.size,
                        });
                    }
                }
            }
        } else {
            // No order within tolerance - this is a CRITICAL COVERAGE GAP
            // Must place new order at this target price
            if target.size > EPSILON {
                debug!(
                    priority = priority,
                    target_price = target.price,
                    tolerance_bps = tolerance_bps,
                    "Priority matching: placing order for uncovered target"
                );
                actions.push(LadderAction::Place {
                    side,
                    price: target.price,
                    size: target.size,
                });
            }
        }
    }

    // Phase 2: Cancel any orders not matched to any target (stale orders)
    for order in current.iter() {
        if !matched_orders.contains(&order.oid) {
            // Check if this order is close to ANY target before cancelling
            let close_to_any_target = targets.iter().any(|t| {
                bps_diff(order.price, t.price) as f64 <= config.max_match_distance_bps
            });

            if !close_to_any_target {
                debug!(
                    oid = order.oid,
                    price = order.price,
                    "Priority matching: cancelling stale order (not close to any target)"
                );
                actions.push(LadderAction::Cancel { oid: order.oid });
            }
        }
    }

    actions
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
