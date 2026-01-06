//! Safety auditor - extracts safety sync responsibilities into focused methods.
//!
//! The auditor provides utility methods for:
//! - Order cleanup (expired fill windows)
//! - Stale pending order detection
//! - Stuck cancel detection
//! - Exchange state reconciliation
//! - Dynamic limit updates
//! - Reduce-only mode status reporting

use std::collections::HashSet;
use tracing::{debug, warn};

/// Result of a safety audit run.
#[derive(Debug, Clone)]
pub struct AuditResult {
    /// Number of orders cleaned up (expired fill windows)
    pub orders_cleaned: usize,
    /// Number of stale pending orders removed
    pub stale_pending_removed: usize,
    /// Number of stuck cancel orders detected
    pub stuck_cancels: usize,
    /// Number of orphan orders found on exchange (not in local tracking)
    pub orphan_orders: usize,
    /// Number of stale local orders removed (not on exchange)
    pub stale_local_removed: usize,
    /// Whether local and exchange state are in sync
    pub is_synced: bool,
    /// Whether reduce-only mode is active
    pub reduce_only_active: bool,
    /// Reason for reduce-only mode (if active)
    pub reduce_only_reason: Option<String>,
}

impl Default for AuditResult {
    fn default() -> Self {
        Self {
            orders_cleaned: 0,
            stale_pending_removed: 0,
            stuck_cancels: 0,
            orphan_orders: 0,
            stale_local_removed: 0,
            is_synced: true, // Default to synced (no issues)
            reduce_only_active: false,
            reduce_only_reason: None,
        }
    }
}

impl AuditResult {
    /// Create a new audit result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any issues were found.
    pub fn has_issues(&self) -> bool {
        self.stuck_cancels > 0
            || self.orphan_orders > 0
            || self.stale_local_removed > 0
            || !self.is_synced
    }
}

/// Safety auditor - provides utility methods for safety sync.
///
/// This struct holds no state - it provides pure functions that operate
/// on the data passed to them, making them easily testable.
pub struct SafetyAuditor;

impl SafetyAuditor {
    /// Analyze reduce-only status.
    ///
    /// Returns (is_active, reason) tuple.
    pub fn check_reduce_only(
        position: f64,
        position_value: f64,
        max_position: f64,
        max_position_value: f64,
    ) -> (bool, Option<String>) {
        let over_position_limit = position.abs() > max_position;
        let over_value_limit = position_value > max_position_value;

        if !over_position_limit && !over_value_limit {
            return (false, None);
        }

        let direction = if position > 0.0 { "long" } else { "short" };
        let reason = if over_value_limit && over_position_limit {
            format!(
                "{}: position={:.4} > {:.4} AND value ${:.2} > ${:.2}",
                direction,
                position.abs(),
                max_position,
                position_value,
                max_position_value
            )
        } else if over_value_limit {
            format!(
                "{}: value ${:.2} > ${:.2} limit",
                direction, position_value, max_position_value
            )
        } else {
            format!(
                "{}: position={:.4} > {:.4} limit",
                direction,
                position.abs(),
                max_position
            )
        };

        (true, Some(reason))
    }

    /// Find orphan orders (on exchange but not in local tracking).
    pub fn find_orphans(exchange_oids: &HashSet<u64>, local_oids: &HashSet<u64>) -> Vec<u64> {
        exchange_oids.difference(local_oids).copied().collect()
    }

    /// Find stale local orders (in local tracking but not on exchange).
    ///
    /// Returns OIDs of orders that should be removed from local tracking.
    /// Excludes orders in cancel window (CancelPending/CancelConfirmed).
    /// exclude_young: filter returns true if order should be KEPT (e.g. it is young)
    pub fn find_stale_local<F, G>(
        exchange_oids: &HashSet<u64>,
        local_oids: &HashSet<u64>,
        is_in_cancel_window: F,
        is_young: G,
    ) -> Vec<u64>
    where
        F: Fn(u64) -> bool,
        G: Fn(u64) -> bool,
    {
        local_oids
            .difference(exchange_oids)
            .filter(|oid| !is_in_cancel_window(**oid))
            .filter(|oid| !is_young(**oid)) // Logic: if young, don't remove (keep it)
            .copied()
            .collect()
    }

    /// Log orphan orders being cancelled.
    pub fn log_orphan_cancellation(oid: u64, success: bool) {
        if success {
            debug!("[SafetySync] Cancelled orphan order: oid={}", oid);
        } else {
            warn!("[SafetySync] Failed to cancel orphan order: oid={}", oid);
        }
    }

    /// Log stale local order removal.
    pub fn log_stale_removal(oid: u64) {
        warn!(
            "[SafetySync] Stale order in tracking (not on exchange): oid={} - removing",
            oid
        );
    }

    /// Log sync status.
    pub fn log_sync_status(exchange_count: usize, local_active_count: usize, is_synced: bool) {
        if is_synced {
            debug!(
                "[SafetySync] State in sync: {} active orders (exchange matches local)",
                local_active_count
            );
        } else {
            warn!(
                "[SafetySync] State mismatch: exchange={}, local_active={}",
                exchange_count, local_active_count
            );
        }
    }

    /// Log dynamic limit update.
    pub fn log_dynamic_limit(
        new_limit: f64,
        account_value: f64,
        sigma: f64,
        sigma_confidence: f64,
        time_horizon: f64,
    ) {
        debug!(
            "[SafetySync] Dynamic limit: ${:.2} (equity=${:.2}, σ={:.6}, conf={:.2}, T={:.1}s)",
            new_limit, account_value, sigma, sigma_confidence, time_horizon
        );
    }

    /// Log reduce-only mode status.
    pub fn log_reduce_only_status(is_active: bool, reason: Option<&str>) {
        if is_active {
            if let Some(r) = reason {
                warn!("[SafetySync] REDUCE-ONLY MODE ACTIVE - {}", r);
            } else {
                warn!("[SafetySync] REDUCE-ONLY MODE ACTIVE");
            }
        }
    }

    /// Analyze pending exposure risk.
    ///
    /// Returns a warning message if worst-case positions would breach limits.
    pub fn check_pending_exposure_risk(
        position: f64,
        pending_bid_exposure: f64,
        pending_ask_exposure: f64,
        max_position: f64,
    ) -> Option<String> {
        let worst_case_long = position + pending_bid_exposure;
        let worst_case_short = position - pending_ask_exposure;

        let exceeds_long = worst_case_long > max_position;
        let exceeds_short = worst_case_short < -max_position;

        if exceeds_long && exceeds_short {
            Some(format!(
                "Worst-case positions would breach limits on BOTH sides: \
                 long={:.4} short={:.4} (limit=±{:.4})",
                worst_case_long,
                worst_case_short.abs(),
                max_position
            ))
        } else if exceeds_long {
            Some(format!(
                "Worst-case LONG position {:.4} would exceed limit {:.4} \
                 (pending_bid={:.4})",
                worst_case_long, max_position, pending_bid_exposure
            ))
        } else if exceeds_short {
            Some(format!(
                "Worst-case SHORT position {:.4} would exceed limit {:.4} \
                 (pending_ask={:.4})",
                worst_case_short.abs(),
                max_position,
                pending_ask_exposure
            ))
        } else {
            None
        }
    }

    /// Log pending exposure risk warning.
    pub fn log_pending_exposure_warning(warning: &str) {
        warn!("[SafetySync] PENDING EXPOSURE WARNING: {}", warning);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_only_under_limits() {
        let (is_active, reason) = SafetyAuditor::check_reduce_only(
            0.5,     // position
            5000.0,  // position_value
            1.0,     // max_position
            10000.0, // max_position_value
        );
        assert!(!is_active);
        assert!(reason.is_none());
    }

    #[test]
    fn test_reduce_only_over_position() {
        let (is_active, reason) = SafetyAuditor::check_reduce_only(
            1.5,     // position > max
            7500.0,  // position_value
            1.0,     // max_position
            10000.0, // max_position_value
        );
        assert!(is_active);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("position"));
    }

    #[test]
    fn test_reduce_only_over_value() {
        let (is_active, reason) = SafetyAuditor::check_reduce_only(
            0.5,     // position
            15000.0, // position_value > max
            1.0,     // max_position
            10000.0, // max_position_value
        );
        assert!(is_active);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("value"));
    }

    #[test]
    fn test_reduce_only_both_limits() {
        let (is_active, reason) = SafetyAuditor::check_reduce_only(
            -2.0,    // position > max (short)
            20000.0, // position_value > max
            1.0,     // max_position
            10000.0, // max_position_value
        );
        assert!(is_active);
        let reason_str = reason.unwrap();
        assert!(reason_str.contains("short"));
        assert!(reason_str.contains("AND"));
    }

    #[test]
    fn test_find_orphans() {
        let exchange: HashSet<u64> = [1, 2, 3, 4].into_iter().collect();
        let local: HashSet<u64> = [2, 3].into_iter().collect();

        let orphans = SafetyAuditor::find_orphans(&exchange, &local);
        assert_eq!(orphans.len(), 2);
        assert!(orphans.contains(&1) || orphans.contains(&4));
    }

    #[test]
    fn test_find_stale_local() {
        let exchange: HashSet<u64> = [1, 2].into_iter().collect();
        let local: HashSet<u64> = [1, 2, 3, 4, 5].into_iter().collect();

        // Simulate order 3 being in cancel window
        let is_cancel = |oid: u64| oid == 3;
        // Simulate order 5 being young
        let is_young = |oid: u64| oid == 5;

        let stale = SafetyAuditor::find_stale_local(&exchange, &local, is_cancel, is_young);
        assert_eq!(stale.len(), 1);
        assert!(stale.contains(&4)); // 3 excluded (cancel), 5 excluded (young)
    }

    #[test]
    fn test_audit_result_has_issues() {
        let mut result = AuditResult::new();
        assert!(!result.has_issues());

        result.stuck_cancels = 1;
        assert!(result.has_issues());
    }

    // === Pending Exposure Tests ===

    #[test]
    fn test_pending_exposure_within_limits() {
        let warning = SafetyAuditor::check_pending_exposure_risk(
            0.5, // position
            0.3, // pending_bid
            0.2, // pending_ask
            1.0, // max_position
        );
        assert!(warning.is_none()); // 0.5 + 0.3 = 0.8 < 1.0
    }

    #[test]
    fn test_pending_exposure_exceeds_long() {
        let warning = SafetyAuditor::check_pending_exposure_risk(
            0.5, // position
            0.7, // pending_bid - would exceed
            0.2, // pending_ask
            1.0, // max_position
        );
        assert!(warning.is_some());
        let msg = warning.unwrap();
        assert!(msg.contains("LONG"));
        assert!(msg.contains("1.2")); // 0.5 + 0.7
    }

    #[test]
    fn test_pending_exposure_exceeds_short() {
        let warning = SafetyAuditor::check_pending_exposure_risk(
            -0.5, // position (short)
            0.2,  // pending_bid
            0.7,  // pending_ask - would exceed
            1.0,  // max_position
        );
        assert!(warning.is_some());
        let msg = warning.unwrap();
        assert!(msg.contains("SHORT"));
        assert!(msg.contains("1.2")); // |-0.5 - 0.7| = 1.2
    }

    #[test]
    fn test_pending_exposure_exceeds_both() {
        let warning = SafetyAuditor::check_pending_exposure_risk(
            0.0, // neutral position
            1.5, // pending_bid - would exceed long
            1.5, // pending_ask - would exceed short
            1.0, // max_position
        );
        assert!(warning.is_some());
        let msg = warning.unwrap();
        assert!(msg.contains("BOTH"));
    }
}
