//! Ladder reconciliation logic.
//!
//! Implements diffing between current orders and target ladder to generate
//! minimal cancel/place/modify actions for improved spread capturing.

use std::collections::HashSet;

use crate::{bps_diff, EPSILON};

use super::impulse_filter::{ImpulseDecision, ImpulseFilter};
use super::types::{LadderAction, Side, TrackedOrder};
use crate::market_maker::quoting::LadderLevel;
use crate::market_maker::tracking::queue::{
    QueueKeepReason, QueuePositionTracker, QueueValueComparator, QueueValueConfig,
    QueueValueDecision, QueueValueStats,
};

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
    /// DEPRECATED: Use queue_value_config instead for EV-based decisions.
    pub use_impulse_filter: bool,
    /// Enable queue value comparison (EV-based update gating).
    /// When true, orders are kept if their queue position EV exceeds replacement EV.
    /// This subsumes and replaces impulse filtering with a more principled approach.
    pub use_queue_value_comparison: bool,
    /// Configuration for queue value EV comparison.
    pub queue_value_config: QueueValueConfig,
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
            use_queue_aware: true,    // Enabled by default (verified)
            queue_horizon_seconds: 1.0, // 1-second fill horizon
            use_impulse_filter: false, // Disabled - superseded by queue value comparison
            use_queue_value_comparison: true, // Enabled by default - EV-based decisions
            queue_value_config: QueueValueConfig::default(),
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
    /// Maximum price modification allowed (dynamically computed from vol)
    pub max_modify_price_bps: f64,
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
            optimal_spread_bps: 20.0,   // Default ~20 bps
            max_modify_price_bps: 50.0, // Default 50 bps
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
    pub fn from_market_params(gamma: f64, kappa: f64, sigma: f64, queue_horizon: f64) -> Self {
        // Clamp inputs to avoid division by zero
        let gamma_safe = gamma.max(0.001);
        let kappa_safe = kappa.max(1.0);

        // GLFT optimal spread: δ* = (1/γ) × ln(1 + γ/κ)
        let optimal_spread_frac = (1.0 / gamma_safe) * (1.0 + gamma_safe / kappa_safe).ln();
        let optimal_spread_bps = optimal_spread_frac * 10_000.0;

        // Derive thresholds from optimal spread
        // Volatility-driven max modify: 2 standard deviations over queue horizon
        // 2 * sigma * sqrt(T) converted to bps
        let vol_bps = sigma * queue_horizon.sqrt() * 10_000.0;
        let max_modify_price_bps = (2.0 * vol_bps).max(optimal_spread_bps).clamp(10.0, 100.0);

        // Derive matching tolerances from volatility/modify capability
        // We must be able to match orders up to max_modify to utilize it
        let best_level_tolerance_bps = (max_modify_price_bps / 2.0).clamp(2.0, 50.0);
        let outer_level_tolerance_bps = (max_modify_price_bps).clamp(5.0, 100.0);

        // Maximum match distance: 1.5x optimal spread
        let max_match_distance_bps = (optimal_spread_bps * 1.5).clamp(15.0, 50.0);

        Self {
            best_level_tolerance_bps,
            outer_level_tolerance_bps,
            max_match_distance_bps,
            queue_value_threshold: 0.3,
            queue_horizon_seconds: queue_horizon,
            optimal_spread_bps,
            max_modify_price_bps,
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
    /// Number of orders kept due to queue value comparison (EV-based).
    pub queue_value_kept_count: usize,
    /// Number of orders replaced after queue value comparison approved.
    pub queue_value_replaced_count: usize,
    /// Number of orders with no queue data (fallback to tolerance).
    pub queue_value_no_data_count: usize,
    /// Aggregate queue value statistics for this reconciliation cycle.
    pub queue_value_stats: Option<QueueValueStats>,
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

/// Smart reconciliation with optional impulse filtering and queue value comparison.
///
/// Extended version that accepts queue tracker and impulse filter for
/// Δλ-based update gating, plus EV-based queue value comparison.
/// Returns both actions and reconciliation statistics.
///
/// # Decision Hierarchy
/// 1. Queue value comparison (if enabled): Keep orders where EV(current) > EV(new)
/// 2. Impulse filter (legacy, if enabled): Δλ-based gating
/// 3. Tolerance-based: SKIP/MODIFY/CANCEL+PLACE based on price/size thresholds
///
/// # Arguments
/// * `current` - Current orders on this side
/// * `target` - Target ladder levels
/// * `side` - Buy or Sell side
/// * `config` - Reconciliation configuration
/// * `queue_tracker` - Optional queue position tracker for P(fill) calculation
/// * `impulse_filter` - Optional impulse filter for Δλ gating (legacy)
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
    use tracing::debug;

    let mut actions = Vec::new();
    let mut stats = ReconcileStats::default();
    let mut matched_targets: HashSet<usize> = HashSet::new();
    let mut matched_orders: HashSet<u64> = HashSet::new();
    let mut queue_stats = QueueValueStats::default();

    // Determine if queue value comparison is active
    let queue_value_active = config.use_queue_value_comparison
        && config.queue_value_config.enabled
        && queue_tracker.is_some()
        && mid_price.is_some();

    // Determine if impulse filtering is active (legacy fallback)
    let impulse_active = config.use_impulse_filter
        && !queue_value_active // Queue value takes precedence
        && queue_tracker.is_some()
        && impulse_filter.is_some()
        && mid_price.is_some();

    // Create queue value comparator if active
    let comparator = if queue_value_active {
        let qt = queue_tracker.unwrap();
        let sigma = qt.sigma();
        Some(QueueValueComparator::new(
            qt,
            config.queue_value_config.clone(),
            sigma,
        ))
    } else {
        None
    };

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

        // Apply queue value comparison if active (takes precedence)
        if let Some(ref comp) = comparator {
            let mid = mid_price.unwrap();
            let decision = comp.compare(m.order.oid, m.order.price, target_level.price, mid);

            match decision {
                QueueValueDecision::Keep {
                    reason,
                    current_ev,
                    replacement_ev,
                } => {
                    debug!(
                        oid = m.order.oid,
                        reason = ?reason,
                        current_ev = %format!("{:.4}", current_ev),
                        replacement_ev = %format!("{:.4}", replacement_ev),
                        price_diff_bps = %format!("{:.2}", m.price_diff_bps),
                        "Queue value: KEEP order (EV comparison)"
                    );
                    // Update stats based on reason
                    match reason {
                        QueueKeepReason::QueueLocked => {
                            queue_stats.kept_queue_locked += 1;
                        }
                        QueueKeepReason::InsufficientImprovement => {
                            queue_stats.kept_insufficient_improvement += 1;
                        }
                        QueueKeepReason::OrderTooYoung => {
                            queue_stats.kept_too_young += 1;
                        }
                    }
                    stats.queue_value_kept_count += 1;
                    stats.skipped_count += 1;
                    continue; // Keep order - queue value too high to sacrifice
                }
                QueueValueDecision::Replace { improvement_pct } => {
                    debug!(
                        oid = m.order.oid,
                        improvement_pct = %format!("{:.1}%", improvement_pct * 100.0),
                        "Queue value: REPLACE order (sufficient improvement)"
                    );
                    queue_stats.replaced += 1;
                    stats.queue_value_replaced_count += 1;
                    // Fall through to normal logic
                }
                QueueValueDecision::NoData => {
                    queue_stats.no_data += 1;
                    stats.queue_value_no_data_count += 1;
                    // Fall through to tolerance-based decision
                }
            }
        }

        // Apply impulse filter if active (legacy fallback)
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

    // Record queue value stats if comparator was used
    if comparator.is_some() {
        // Calculate requests saved: each kept order saves 2 requests (cancel + place)
        let total_kept = queue_stats.kept_queue_locked
            + queue_stats.kept_insufficient_improvement
            + queue_stats.kept_too_young;
        queue_stats.requests_saved = total_kept * 2;
        stats.queue_value_stats = Some(queue_stats);
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
                // Evaluate what action is needed based on FIFO queue rules:
                // - Size reduction: MODIFY preserves queue position ✓
                // - Price change: Loses queue (goes to back of new level)
                // - Size increase: Loses queue (treated as new order)

                let price_diff_bps = best_distance;
                let current_size = order.remaining();
                let target_size = target.size;
                let size_change = target_size - current_size;
                let size_diff_pct = if current_size > EPSILON {
                    (size_change.abs() / current_size).min(1.0)
                } else {
                    1.0
                };

                // Case 1: Perfect match - no action needed
                if price_diff_bps <= 2.0 && size_diff_pct <= 0.05 {
                    continue;
                }

                // Case 2: Price is good, only SIZE REDUCTION needed
                // → Use MODIFY to preserve queue position (FIFO advantage!)
                if price_diff_bps <= config.max_modify_price_bps
                    && size_change < 0.0
                    && size_diff_pct > 0.05
                {
                    debug!(
                        oid = order.oid,
                        current_size = current_size,
                        target_size = target_size,
                        max_modify_bps = config.max_modify_price_bps,
                        "Priority matching: MODIFY (size down) to preserve queue"
                    );
                    actions.push(LadderAction::Modify {
                        oid: order.oid,
                        new_price: order.price, // Keep same price
                        new_size: target_size,  // Reduce size
                        side,
                    });
                    continue;
                }

                // Case 3: Price change required OR size increase needed
                // → Must CANCEL+PLACE (loses queue, but necessary)
                if price_diff_bps > config.best_level_tolerance_bps
                    || size_change > current_size * 0.10
                {
                    debug!(
                        oid = order.oid,
                        price_diff_bps = price_diff_bps,
                        size_change_pct = %format!("{:.1}%", size_change / current_size * 100.0),
                        reason = if price_diff_bps > config.best_level_tolerance_bps { "price_change" } else { "size_increase" },
                        "Priority matching: CANCEL+PLACE (queue lost)"
                    );
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
            let close_to_any_target = targets
                .iter()
                .any(|t| bps_diff(order.price, t.price) as f64 <= config.max_match_distance_bps);

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
