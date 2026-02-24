//! Synchronization Health Tracker.
//!
//! Provides non-blocking health assessment of exchange state synchronization.
//! Instead of blocking on sync checks, the quote engine can query sync health
//! and adjust behavior accordingly (widen spreads, reduce size, or go passive).
//!
//! # Problem Solved
//!
//! The existing SafetySync runs blocking checks that dominated 93.5% of execution time
//! in thin DEX environments. This prevented the market maker from quoting actively.
//!
//! # Solution
//!
//! Track sync health as a continuous metric [0.0, 1.0] rather than binary pass/fail:
//! - **Healthy (> 0.8)**: Normal quoting with full size
//! - **Degraded (0.5-0.8)**: Wider spreads, reduced size (conservative mode)
//! - **Desync (< 0.5)**: Passive mode (reduce-only, no new orders)
//!
//! # Metrics Tracked
//!
//! 1. **Order count match**: local_count / exchange_count ratio
//! 2. **Snapshot freshness**: How old is the last exchange snapshot
//! 3. **Recent sync success**: Rolling success rate of sync operations
//! 4. **Orphan ratio**: Orphan orders / total orders
//! 5. **Pending age**: Average age of pending orders

use std::collections::VecDeque;
use std::time::Instant;
use tracing::debug;

/// Synchronization health level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncHealthLevel {
    /// Healthy: local and exchange state are in sync
    /// Normal quoting with full size
    Healthy,
    /// Degraded: some discrepancies but recoverable
    /// Use wider spreads and reduced size
    Degraded,
    /// Desync: significant state mismatch
    /// Passive mode - reduce-only, no new orders
    Desync,
}

impl SyncHealthLevel {
    /// Get spread multiplier for this health level.
    pub fn spread_multiplier(&self) -> f64 {
        match self {
            SyncHealthLevel::Healthy => 1.0,
            SyncHealthLevel::Degraded => 1.5,
            SyncHealthLevel::Desync => 2.0, // Extra wide if still quoting
        }
    }

    /// Get size multiplier for this health level.
    pub fn size_multiplier(&self) -> f64 {
        match self {
            SyncHealthLevel::Healthy => 1.0,
            SyncHealthLevel::Degraded => 0.5,
            SyncHealthLevel::Desync => 0.0, // No new orders
        }
    }

    /// Check if new quotes should be placed.
    pub fn can_place_new_orders(&self) -> bool {
        !matches!(self, SyncHealthLevel::Desync)
    }

    /// Check if only reduce-only orders are allowed.
    pub fn reduce_only(&self) -> bool {
        matches!(self, SyncHealthLevel::Desync)
    }
}

/// Configuration for sync health tracker.
#[derive(Debug, Clone)]
pub struct SyncHealthConfig {
    /// Maximum snapshot age before health degrades (ms)
    pub max_snapshot_age_ms: u64,
    /// Critical snapshot age (ms)
    pub critical_snapshot_age_ms: u64,
    /// Maximum orphan ratio before health degrades
    pub max_orphan_ratio: f64,
    /// Number of recent syncs to track for success rate
    pub sync_history_size: usize,
    /// Threshold for healthy status
    pub healthy_threshold: f64,
    /// Threshold for degraded status (below this = desync)
    pub degraded_threshold: f64,
}

impl Default for SyncHealthConfig {
    fn default() -> Self {
        Self {
            max_snapshot_age_ms: 5000,       // 5 seconds
            critical_snapshot_age_ms: 10000, // 10 seconds
            max_orphan_ratio: 0.2,           // 20% orphans
            sync_history_size: 20,
            healthy_threshold: 0.8,
            degraded_threshold: 0.5,
        }
    }
}

/// Sync operation result for tracking.
#[derive(Debug, Clone, Copy)]
struct SyncResult {
    #[allow(dead_code)] // Reserved for time-weighted health calculations
    timestamp: Instant,
    success: bool,
    local_count: usize,
    exchange_count: usize,
    orphan_count: usize,
}

/// Non-blocking synchronization health tracker.
#[derive(Debug)]
pub struct SyncHealthTracker {
    config: SyncHealthConfig,
    /// Recent sync results for success rate calculation
    sync_history: VecDeque<SyncResult>,
    /// Last known snapshot time
    last_snapshot_time: Option<Instant>,
    /// Current health score [0.0, 1.0]
    health_score: f64,
    /// Current health level
    health_level: SyncHealthLevel,
    /// Total syncs performed
    total_syncs: u64,
    /// Last health update time
    last_update: Instant,
}

impl SyncHealthTracker {
    /// Create a new sync health tracker.
    pub fn new(config: SyncHealthConfig) -> Self {
        Self {
            sync_history: VecDeque::with_capacity(config.sync_history_size),
            config,
            last_snapshot_time: None,
            health_score: 1.0, // Start healthy
            health_level: SyncHealthLevel::Healthy,
            total_syncs: 0,
            last_update: Instant::now(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SyncHealthConfig::default())
    }

    /// Record a sync operation result.
    pub fn record_sync(
        &mut self,
        local_count: usize,
        exchange_count: usize,
        orphan_count: usize,
        success: bool,
    ) {
        let result = SyncResult {
            timestamp: Instant::now(),
            success,
            local_count,
            exchange_count,
            orphan_count,
        };

        // Add to history
        if self.sync_history.len() >= self.config.sync_history_size {
            self.sync_history.pop_front();
        }
        self.sync_history.push_back(result);
        self.total_syncs += 1;

        // Update health
        self.update_health();
    }

    /// Record a snapshot received from exchange.
    pub fn record_snapshot(&mut self) {
        self.last_snapshot_time = Some(Instant::now());
        self.update_health();
    }

    /// Get current health score [0.0, 1.0].
    pub fn health_score(&self) -> f64 {
        self.health_score
    }

    /// Get current health level.
    pub fn health_level(&self) -> SyncHealthLevel {
        self.health_level
    }

    /// Check if quoting is allowed (not in desync state).
    pub fn can_quote(&self) -> bool {
        self.health_level.can_place_new_orders()
    }

    /// Get spread multiplier based on health.
    pub fn spread_multiplier(&self) -> f64 {
        self.health_level.spread_multiplier()
    }

    /// Get size multiplier based on health.
    pub fn size_multiplier(&self) -> f64 {
        self.health_level.size_multiplier()
    }

    /// Update health score based on tracked metrics.
    fn update_health(&mut self) {
        let mut score = 1.0;

        // 1. Snapshot freshness (40% of score)
        let snapshot_score = self.compute_snapshot_freshness_score();
        score *= 0.6 + 0.4 * snapshot_score;

        // 2. Recent sync success rate (30% of score)
        let success_score = self.compute_success_rate_score();
        score *= 0.7 + 0.3 * success_score;

        // 3. Order count match (20% of score)
        let match_score = self.compute_order_match_score();
        score *= 0.8 + 0.2 * match_score;

        // 4. Orphan ratio (10% of score)
        let orphan_score = self.compute_orphan_score();
        score *= 0.9 + 0.1 * orphan_score;

        self.health_score = score.clamp(0.0, 1.0);

        // Determine level based on score
        self.health_level = if self.health_score >= self.config.healthy_threshold {
            SyncHealthLevel::Healthy
        } else if self.health_score >= self.config.degraded_threshold {
            SyncHealthLevel::Degraded
        } else {
            SyncHealthLevel::Desync
        };

        self.last_update = Instant::now();
    }

    /// Compute snapshot freshness score [0.0, 1.0].
    fn compute_snapshot_freshness_score(&self) -> f64 {
        match self.last_snapshot_time {
            None => 0.5, // No snapshot yet - moderate score
            Some(t) => {
                let age_ms = t.elapsed().as_millis() as u64;
                if age_ms <= self.config.max_snapshot_age_ms {
                    1.0
                } else if age_ms >= self.config.critical_snapshot_age_ms {
                    0.0
                } else {
                    // Linear interpolation
                    let range =
                        self.config.critical_snapshot_age_ms - self.config.max_snapshot_age_ms;
                    let excess = age_ms - self.config.max_snapshot_age_ms;
                    1.0 - (excess as f64 / range as f64)
                }
            }
        }
    }

    /// Compute recent sync success rate [0.0, 1.0].
    fn compute_success_rate_score(&self) -> f64 {
        if self.sync_history.is_empty() {
            return 0.8; // No history - assume mostly healthy
        }

        let success_count = self.sync_history.iter().filter(|r| r.success).count();
        success_count as f64 / self.sync_history.len() as f64
    }

    /// Compute order count match score [0.0, 1.0].
    fn compute_order_match_score(&self) -> f64 {
        if self.sync_history.is_empty() {
            return 1.0;
        }

        // Use most recent sync
        let recent = self.sync_history.back().unwrap();

        if recent.exchange_count == 0 && recent.local_count == 0 {
            return 1.0; // Both empty is perfect match
        }

        if recent.exchange_count == 0 || recent.local_count == 0 {
            return 0.5; // One empty, one not - degraded
        }

        // Compute ratio (smaller / larger)
        if recent.local_count <= recent.exchange_count {
            recent.local_count as f64 / recent.exchange_count as f64
        } else {
            recent.exchange_count as f64 / recent.local_count as f64
        }
    }

    /// Compute orphan ratio score [0.0, 1.0].
    fn compute_orphan_score(&self) -> f64 {
        if self.sync_history.is_empty() {
            return 1.0;
        }

        let recent = self.sync_history.back().unwrap();
        let total = recent.exchange_count.max(1);
        let orphan_ratio = recent.orphan_count as f64 / total as f64;

        if orphan_ratio <= 0.0 {
            1.0
        } else if orphan_ratio >= self.config.max_orphan_ratio {
            0.0
        } else {
            1.0 - (orphan_ratio / self.config.max_orphan_ratio)
        }
    }

    /// Get summary for diagnostics.
    pub fn summary(&self) -> SyncHealthSummary {
        SyncHealthSummary {
            health_score: self.health_score,
            health_level: self.health_level,
            snapshot_age_ms: self
                .last_snapshot_time
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(u64::MAX),
            success_rate: self.compute_success_rate_score(),
            total_syncs: self.total_syncs,
            spread_multiplier: self.spread_multiplier(),
            size_multiplier: self.size_multiplier(),
            can_quote: self.can_quote(),
        }
    }

    /// Log current health state.
    pub fn log_state(&self) {
        let summary = self.summary();
        debug!(
            health_score = %format!("{:.2}", summary.health_score),
            health_level = ?summary.health_level,
            snapshot_age_ms = summary.snapshot_age_ms,
            success_rate = %format!("{:.2}", summary.success_rate),
            spread_mult = %format!("{:.2}", summary.spread_multiplier),
            size_mult = %format!("{:.2}", summary.size_multiplier),
            can_quote = summary.can_quote,
            "[SyncHealth] State"
        );
    }
}

impl Default for SyncHealthTracker {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Summary of sync health state.
#[derive(Debug, Clone)]
pub struct SyncHealthSummary {
    pub health_score: f64,
    pub health_level: SyncHealthLevel,
    pub snapshot_age_ms: u64,
    pub success_rate: f64,
    pub total_syncs: u64,
    pub spread_multiplier: f64,
    pub size_multiplier: f64,
    pub can_quote: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let tracker = SyncHealthTracker::default_config();
        assert!(tracker.health_score() >= 0.8);
        assert_eq!(tracker.health_level(), SyncHealthLevel::Healthy);
        assert!(tracker.can_quote());
    }

    #[test]
    fn test_record_sync_success() {
        let mut tracker = SyncHealthTracker::default_config();

        // Record a fresh snapshot (required for high health score)
        tracker.record_snapshot();

        // Record successful syncs
        for _ in 0..5 {
            tracker.record_sync(10, 10, 0, true);
        }

        assert!(tracker.health_score() >= 0.9);
        assert_eq!(tracker.health_level(), SyncHealthLevel::Healthy);
    }

    #[test]
    fn test_degraded_on_failures() {
        let mut tracker = SyncHealthTracker::default_config();

        // Mix of success and failure
        for i in 0..10 {
            tracker.record_sync(10, 10, 0, i % 3 != 0);
        }

        // Some failures should degrade health
        assert!(tracker.health_score() < 1.0);
    }

    #[test]
    fn test_orphan_impact() {
        let mut tracker = SyncHealthTracker::default_config();

        // Sync with high orphan count
        tracker.record_sync(5, 10, 5, true);

        // Orphans should reduce health
        assert!(tracker.health_score() < 0.95);
    }

    #[test]
    fn test_snapshot_freshness() {
        let mut tracker = SyncHealthTracker::default_config();

        // Fresh snapshot
        tracker.record_snapshot();
        assert!(tracker.compute_snapshot_freshness_score() > 0.99);

        // Without snapshot
        let tracker2 = SyncHealthTracker::default_config();
        assert!(tracker2.compute_snapshot_freshness_score() < 1.0);
    }

    #[test]
    fn test_health_levels() {
        let mut tracker = SyncHealthTracker::new(SyncHealthConfig {
            healthy_threshold: 0.8,
            degraded_threshold: 0.5,
            ..Default::default()
        });

        // Force healthy
        for _ in 0..10 {
            tracker.record_sync(10, 10, 0, true);
            tracker.record_snapshot();
        }
        assert_eq!(tracker.health_level(), SyncHealthLevel::Healthy);

        // Force degraded through failures
        for _ in 0..15 {
            tracker.record_sync(5, 10, 2, false);
        }
        // Health should be reduced
        assert!(tracker.health_score() < 0.9);
    }

    #[test]
    fn test_multipliers() {
        assert_eq!(SyncHealthLevel::Healthy.spread_multiplier(), 1.0);
        assert_eq!(SyncHealthLevel::Degraded.spread_multiplier(), 1.5);
        assert_eq!(SyncHealthLevel::Desync.spread_multiplier(), 2.0);

        assert_eq!(SyncHealthLevel::Healthy.size_multiplier(), 1.0);
        assert_eq!(SyncHealthLevel::Degraded.size_multiplier(), 0.5);
        assert_eq!(SyncHealthLevel::Desync.size_multiplier(), 0.0);
    }
}
