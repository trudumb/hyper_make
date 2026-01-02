//! Orphan order tracker - prevents false orphan detection during order lifecycle.
//!
//! The market maker has a race condition where orders placed via REST API may not
//! be immediately reflected in local tracking when safety_sync runs. This tracker
//! solves two problems:
//!
//! 1. **Expected CLOIDs**: Orders that are in-flight (API called but OID not yet assigned
//!    to local tracking) should not be treated as orphans.
//!
//! 2. **Orphan Grace Period**: Even if an order appears orphaned, we wait a configurable
//!    grace period before cancelling it, allowing the finalization to complete.
//!
//! ## Order Lifecycle
//!
//! ```text
//! T=0ms   add_pending_with_cloid(cloid) → Expected CLOID registered
//! T=1ms   API call to exchange
//! T=500ms API returns with OID → record_oid_for_cloid(cloid, oid)
//! T=501ms finalize_pending_by_cloid() → mark_finalized(oid)
//! T=10s   safety_sync() → exclude expected OIDs, check grace periods
//! ```

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, trace};

/// Configuration for orphan tracking behavior.
#[derive(Debug, Clone)]
pub struct OrphanTrackerConfig {
    /// How long to wait before considering an order truly orphaned.
    /// Orders seen on exchange but not in local tracking must be seen
    /// for this duration before they're cancelled.
    /// Default: 5 seconds
    pub orphan_grace_period: Duration,

    /// How long to keep expected CLOIDs before expiring them.
    /// If an API call takes longer than this, we assume it failed.
    /// Default: 30 seconds
    pub expected_cloid_ttl: Duration,

    /// How long to keep finalized OIDs in the protection set.
    /// This prevents recently-finalized orders from being detected as orphans.
    /// Default: 10 seconds
    pub finalized_protection_period: Duration,
}

impl Default for OrphanTrackerConfig {
    fn default() -> Self {
        Self {
            orphan_grace_period: Duration::from_secs(5),
            expected_cloid_ttl: Duration::from_secs(30),
            finalized_protection_period: Duration::from_secs(10),
        }
    }
}

/// Tracks in-flight orders and orphan grace periods.
#[derive(Debug)]
pub struct OrphanTracker {
    config: OrphanTrackerConfig,

    /// CLOIDs for orders that have been submitted to API but not yet finalized.
    /// Key: CLOID, Value: (submitted_at, maybe OID if API returned)
    expected_cloids: HashMap<String, (Instant, Option<u64>)>,

    /// Recently finalized OIDs - protected from orphan detection.
    /// Key: OID, Value: finalized_at
    recently_finalized: HashMap<u64, Instant>,

    /// First time each potential orphan OID was observed.
    /// Only after grace_period do we consider it truly orphaned.
    /// Key: OID, Value: first_seen_at
    orphan_first_seen: HashMap<u64, Instant>,
}

impl OrphanTracker {
    /// Create a new tracker with default config.
    pub fn new() -> Self {
        Self::with_config(OrphanTrackerConfig::default())
    }

    /// Create a new tracker with custom config.
    pub fn with_config(config: OrphanTrackerConfig) -> Self {
        Self {
            config,
            expected_cloids: HashMap::new(),
            recently_finalized: HashMap::new(),
            orphan_first_seen: HashMap::new(),
        }
    }

    // =========================================================================
    // Order Placement Flow
    // =========================================================================

    /// Register CLOIDs for orders about to be placed.
    ///
    /// Call this BEFORE the bulk order API call, for all CLOIDs being submitted.
    /// These CLOIDs will be protected from orphan detection until they're finalized
    /// or expire.
    pub fn register_expected_cloids(&mut self, cloids: &[String]) {
        let now = Instant::now();
        for cloid in cloids {
            self.expected_cloids.insert(cloid.clone(), (now, None));
        }
        trace!(
            count = cloids.len(),
            "Registered expected CLOIDs for order placement"
        );
    }

    /// Record that an API call returned with an OID for a CLOID.
    ///
    /// Call this when the bulk order API returns with OIDs.
    /// The OID will be protected until `mark_finalized()` is called.
    pub fn record_oid_for_cloid(&mut self, cloid: &str, oid: u64) {
        if let Some(entry) = self.expected_cloids.get_mut(cloid) {
            entry.1 = Some(oid);
            trace!(cloid = %cloid, oid = oid, "Recorded OID for expected CLOID");
        } else {
            // CLOID not found - might have expired or was never registered
            debug!(
                cloid = %cloid,
                oid = oid,
                "OID received for unknown CLOID - protecting as recently finalized"
            );
            // Still protect it briefly
            self.recently_finalized.insert(oid, Instant::now());
        }
    }

    /// Mark an order as fully finalized (moved from pending to tracked).
    ///
    /// Call this after `finalize_pending_by_cloid()` succeeds.
    /// Moves the OID from expected to recently_finalized for continued protection.
    pub fn mark_finalized(&mut self, cloid: &str, oid: u64) {
        // Remove from expected
        self.expected_cloids.remove(cloid);

        // Add to recently finalized for continued protection
        self.recently_finalized.insert(oid, Instant::now());

        // Clear from orphan tracking if present
        self.orphan_first_seen.remove(&oid);

        trace!(
            cloid = %cloid,
            oid = oid,
            "Order finalized - moved to recently finalized protection"
        );
    }

    /// Mark a CLOID as failed (order placement failed, no OID assigned).
    ///
    /// Call this if the API returns an error for a CLOID.
    pub fn mark_failed(&mut self, cloid: &str) {
        self.expected_cloids.remove(cloid);
        trace!(cloid = %cloid, "Order placement failed - removed from expected");
    }

    // =========================================================================
    // Orphan Detection (called during safety_sync)
    // =========================================================================

    /// Get OIDs that should be protected from orphan detection.
    ///
    /// Returns a set of OIDs that are either:
    /// - In-flight (CLOID expected, OID assigned but not finalized)
    /// - Recently finalized (within protection period)
    pub fn protected_oids(&self) -> HashSet<u64> {
        let now = Instant::now();
        let mut protected = HashSet::new();

        // Add OIDs from expected CLOIDs
        for (_, maybe_oid) in self.expected_cloids.values() {
            if let Some(oid) = maybe_oid {
                protected.insert(*oid);
            }
        }

        // Add recently finalized OIDs still within protection period
        for (oid, finalized_at) in &self.recently_finalized {
            if now.duration_since(*finalized_at) < self.config.finalized_protection_period {
                protected.insert(*oid);
            }
        }

        protected
    }

    /// Filter orphan candidates to only those that have aged past grace period.
    ///
    /// Takes a list of potential orphans (on exchange but not in local tracking)
    /// and returns only those that:
    /// 1. Are not protected (not in expected or recently finalized)
    /// 2. Have been observed as orphans for longer than grace_period
    ///
    /// Returns (aged_orphans, new_orphans_count) where new_orphans_count is
    /// how many new potential orphans were added to tracking this cycle.
    pub fn filter_aged_orphans(&mut self, candidate_orphans: &[u64]) -> (Vec<u64>, usize) {
        let now = Instant::now();
        let protected = self.protected_oids();

        let mut aged = Vec::new();
        let mut new_count = 0;

        for &oid in candidate_orphans {
            // Skip if protected
            if protected.contains(&oid) {
                trace!(oid = oid, "Orphan candidate is protected - skipping");
                continue;
            }

            // Track first-seen time
            let first_seen = self.orphan_first_seen.entry(oid).or_insert_with(|| {
                new_count += 1;
                now
            });

            // Check if aged past grace period
            if now.duration_since(*first_seen) >= self.config.orphan_grace_period {
                aged.push(oid);
            } else {
                let remaining = self.config.orphan_grace_period - now.duration_since(*first_seen);
                trace!(
                    oid = oid,
                    remaining_ms = remaining.as_millis(),
                    "Orphan candidate in grace period"
                );
            }
        }

        (aged, new_count)
    }

    /// Clear an orphan from tracking (after it's been cancelled or found to not be orphaned).
    pub fn clear_orphan(&mut self, oid: u64) {
        self.orphan_first_seen.remove(&oid);
    }

    // =========================================================================
    // Maintenance
    // =========================================================================

    /// Clean up expired entries.
    ///
    /// Call this periodically (e.g., during safety_sync) to prevent memory growth.
    /// Returns the number of entries cleaned up.
    pub fn cleanup(&mut self) -> usize {
        let now = Instant::now();
        let mut cleaned = 0;

        // Clean up expired expected CLOIDs
        let expired_cloids: Vec<String> = self
            .expected_cloids
            .iter()
            .filter(|(_, (submitted_at, _))| {
                now.duration_since(*submitted_at) > self.config.expected_cloid_ttl
            })
            .map(|(cloid, _)| cloid.clone())
            .collect();

        for cloid in expired_cloids {
            self.expected_cloids.remove(&cloid);
            cleaned += 1;
        }

        // Clean up expired recently finalized
        let expired_finalized: Vec<u64> = self
            .recently_finalized
            .iter()
            .filter(|(_, finalized_at)| {
                now.duration_since(**finalized_at) > self.config.finalized_protection_period
            })
            .map(|(oid, _)| *oid)
            .collect();

        for oid in expired_finalized {
            self.recently_finalized.remove(&oid);
            cleaned += 1;
        }

        // Clean up very old orphan first-seen entries
        // (orphans that weren't cancelled for some reason)
        let stale_orphan_threshold = self.config.orphan_grace_period * 10;
        let stale_orphans: Vec<u64> = self
            .orphan_first_seen
            .iter()
            .filter(|(_, first_seen)| now.duration_since(**first_seen) > stale_orphan_threshold)
            .map(|(oid, _)| *oid)
            .collect();

        for oid in stale_orphans {
            self.orphan_first_seen.remove(&oid);
            cleaned += 1;
        }

        if cleaned > 0 {
            debug!(
                cleaned = cleaned,
                expected_cloids = self.expected_cloids.len(),
                recently_finalized = self.recently_finalized.len(),
                orphan_tracking = self.orphan_first_seen.len(),
                "Orphan tracker cleanup complete"
            );
        }

        cleaned
    }

    /// Get current tracking statistics.
    pub fn stats(&self) -> OrphanTrackerStats {
        OrphanTrackerStats {
            expected_cloids: self.expected_cloids.len(),
            expected_with_oids: self
                .expected_cloids
                .values()
                .filter(|(_, oid)| oid.is_some())
                .count(),
            recently_finalized: self.recently_finalized.len(),
            orphans_in_grace: self.orphan_first_seen.len(),
        }
    }
}

impl Default for OrphanTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about orphan tracker state.
#[derive(Debug, Clone, Default)]
pub struct OrphanTrackerStats {
    /// Number of CLOIDs expected (in-flight)
    pub expected_cloids: usize,
    /// Number of expected CLOIDs that have received OIDs
    pub expected_with_oids: usize,
    /// Number of recently finalized OIDs (protected)
    pub recently_finalized: usize,
    /// Number of potential orphans in grace period
    pub orphans_in_grace: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_cloid_protection() {
        let mut tracker = OrphanTracker::new();

        // Register expected CLOID
        tracker.register_expected_cloids(&["cloid-1".to_string()]);

        // Record OID for it
        tracker.record_oid_for_cloid("cloid-1", 12345);

        // Protected OIDs should include this OID
        let protected = tracker.protected_oids();
        assert!(protected.contains(&12345));

        // If we see 12345 as orphan, it should be filtered out
        let (aged, new) = tracker.filter_aged_orphans(&[12345, 99999]);
        assert!(aged.is_empty()); // Neither is aged yet
        assert_eq!(new, 1); // Only 99999 is new (12345 is protected)
    }

    #[test]
    fn test_finalization_moves_to_recently_finalized() {
        let mut tracker = OrphanTracker::new();

        tracker.register_expected_cloids(&["cloid-1".to_string()]);
        tracker.record_oid_for_cloid("cloid-1", 12345);
        tracker.mark_finalized("cloid-1", 12345);

        // Should no longer be in expected_cloids
        assert!(!tracker.expected_cloids.contains_key("cloid-1"));

        // Should be in recently_finalized
        assert!(tracker.recently_finalized.contains_key(&12345));

        // Should still be protected
        let protected = tracker.protected_oids();
        assert!(protected.contains(&12345));
    }

    #[test]
    fn test_orphan_grace_period() {
        let config = OrphanTrackerConfig {
            orphan_grace_period: Duration::from_millis(100),
            ..Default::default()
        };
        let mut tracker = OrphanTracker::with_config(config);

        // First observation - should not be aged
        let (aged, new) = tracker.filter_aged_orphans(&[12345]);
        assert!(aged.is_empty());
        assert_eq!(new, 1);

        // Immediately after - still not aged
        let (aged, new) = tracker.filter_aged_orphans(&[12345]);
        assert!(aged.is_empty());
        assert_eq!(new, 0); // Not new anymore

        // Wait for grace period
        std::thread::sleep(Duration::from_millis(150));

        // Now should be aged
        let (aged, _) = tracker.filter_aged_orphans(&[12345]);
        assert_eq!(aged, vec![12345]);
    }

    #[test]
    fn test_cleanup_removes_expired() {
        let config = OrphanTrackerConfig {
            expected_cloid_ttl: Duration::from_millis(50),
            finalized_protection_period: Duration::from_millis(50),
            orphan_grace_period: Duration::from_millis(10),
        };
        let mut tracker = OrphanTracker::with_config(config);

        tracker.register_expected_cloids(&["cloid-1".to_string()]);
        tracker.mark_finalized("cloid-2", 99999);
        tracker.filter_aged_orphans(&[88888]); // Add to orphan tracking

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(100));

        let cleaned = tracker.cleanup();
        assert!(cleaned > 0);

        // Check everything is cleaned up
        assert!(tracker.expected_cloids.is_empty());
        assert!(tracker.recently_finalized.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut tracker = OrphanTracker::new();

        tracker.register_expected_cloids(&["a".to_string(), "b".to_string()]);
        tracker.record_oid_for_cloid("a", 1);
        tracker.mark_finalized("b", 2);
        tracker.filter_aged_orphans(&[3, 4]);

        let stats = tracker.stats();
        assert_eq!(stats.expected_cloids, 1); // "a" still expected
        assert_eq!(stats.expected_with_oids, 1); // "a" has OID
        assert_eq!(stats.recently_finalized, 1); // "b" finalized
        assert_eq!(stats.orphans_in_grace, 2); // 3 and 4 being tracked
    }
}
