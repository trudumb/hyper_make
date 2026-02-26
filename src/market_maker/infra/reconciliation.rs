//! Position Reconciliation Module (Phase 4 Fix)
//!
//! Event-driven position sync with faster background interval.
//! Detects and corrects position drift between local tracking and exchange state.
//!
//! # Triggers
//!
//! Reconciliation is triggered by:
//! 1. Background timer (every 10s vs old 60s)
//! 2. Order rejection (position-related errors)
//! 3. Unmatched fill (fill for untracked order)
//! 4. Large position change (>5% of max_position)
//!
//! # Design
//!
//! - Event-driven: syncs immediately when anomalies detected
//! - Low overhead: timer-based sync is lightweight
//! - Safe: only updates local state to match exchange

use std::time::{Duration, Instant};

/// Trigger types for reconciliation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconciliationTrigger {
    /// Background timer fired
    Timer,
    /// Order was rejected by exchange
    OrderRejection,
    /// Fill received for untracked order
    UnmatchedFill,
    /// Position changed significantly
    LargePositionChange,
    /// Manual/startup sync request
    Manual,
}

/// Configuration for position reconciliation.
#[derive(Debug, Clone)]
pub struct ReconciliationConfig {
    /// Background sync interval (default: 10s)
    pub background_interval: Duration,
    /// Position change threshold (% of max_position) to trigger sync
    pub drift_threshold_pct: f64,
    /// Minimum time between event-triggered syncs (prevent spam)
    pub min_sync_interval: Duration,
    /// Whether to auto-sync on order rejection
    pub sync_on_rejection: bool,
    /// Whether to auto-sync on unmatched fill
    pub sync_on_unmatched_fill: bool,
}

impl Default for ReconciliationConfig {
    fn default() -> Self {
        Self {
            background_interval: Duration::from_secs(10),
            drift_threshold_pct: 5.0,
            min_sync_interval: Duration::from_secs(2),
            sync_on_rejection: true,
            sync_on_unmatched_fill: true,
        }
    }
}

/// Manages position reconciliation scheduling.
#[derive(Debug)]
pub struct PositionReconciler {
    /// When the last sync occurred
    last_sync: Instant,
    /// Number of syncs performed this session
    sync_count: u64,
    /// Number of drift events detected
    drift_events: u64,
    /// Configuration
    config: ReconciliationConfig,
    /// Pending sync request (trigger type if any)
    pending_trigger: Option<ReconciliationTrigger>,
}

impl Default for PositionReconciler {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionReconciler {
    /// Create a new position reconciler with default configuration.
    pub fn new() -> Self {
        Self {
            last_sync: Instant::now(),
            sync_count: 0,
            drift_events: 0,
            config: ReconciliationConfig::default(),
            pending_trigger: None,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: ReconciliationConfig) -> Self {
        Self {
            last_sync: Instant::now(),
            sync_count: 0,
            drift_events: 0,
            config,
            pending_trigger: None,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &ReconciliationConfig {
        &self.config
    }

    /// Check if reconciliation should run now.
    ///
    /// Call this in the main event loop. Returns the trigger if sync is needed.
    pub fn should_sync(&mut self) -> Option<ReconciliationTrigger> {
        // Check pending event trigger first
        if let Some(trigger) = self.pending_trigger.take() {
            // Respect minimum interval
            if self.last_sync.elapsed() >= self.config.min_sync_interval {
                return Some(trigger);
            } else {
                // Re-queue for later
                self.pending_trigger = Some(trigger);
            }
        }

        // Check background timer
        if self.last_sync.elapsed() >= self.config.background_interval {
            return Some(ReconciliationTrigger::Timer);
        }

        None
    }

    /// Record that a sync was completed.
    pub fn record_sync_completed(&mut self) {
        self.last_sync = Instant::now();
        self.sync_count += 1;
        self.pending_trigger = None;
    }

    /// Trigger sync due to order rejection.
    ///
    /// # Arguments
    /// - `error`: The rejection error message
    ///
    /// # Returns
    /// `true` if a sync was triggered
    pub fn on_order_rejection(&mut self, error: &str) -> bool {
        if !self.config.sync_on_rejection {
            return false;
        }

        // Only trigger for position-related rejections
        if error.contains("position") || error.contains("exceed") || error.contains("leverage") {
            tracing::debug!(
                error = %error,
                "Triggering reconciliation due to order rejection"
            );
            self.pending_trigger = Some(ReconciliationTrigger::OrderRejection);
            true
        } else {
            false
        }
    }

    /// Trigger sync due to unmatched fill.
    ///
    /// Called when a fill is received for an untracked order.
    pub fn on_unmatched_fill(&mut self, oid: u64, size: f64) {
        if !self.config.sync_on_unmatched_fill {
            return;
        }

        // oid=0 with size=0 is a dedup artifact, not a real unmatched fill
        // Log at debug level to avoid spam
        if oid == 0 && size.abs() < 1e-9 {
            tracing::debug!("Skipping reconciliation for oid=0 size=0 (dedup artifact)");
            return;
        }

        tracing::warn!(
            oid = oid,
            size = %format!("{:.6}", size),
            "Triggering reconciliation due to unmatched fill"
        );
        self.drift_events += 1;
        self.pending_trigger = Some(ReconciliationTrigger::UnmatchedFill);
    }

    /// Trigger sync due to large position change.
    ///
    /// # Arguments
    /// - `delta`: Position change
    /// - `max_position`: Maximum allowed position
    ///
    /// # Returns
    /// `true` if the change is large enough to trigger sync
    pub fn on_position_change(&mut self, delta: f64, max_position: f64) -> bool {
        let threshold = max_position * self.config.drift_threshold_pct / 100.0;
        if delta.abs() > threshold {
            tracing::debug!(
                delta = %format!("{:.6}", delta),
                threshold = %format!("{:.6}", threshold),
                "Triggering reconciliation due to large position change"
            );
            self.pending_trigger = Some(ReconciliationTrigger::LargePositionChange);
            true
        } else {
            false
        }
    }

    /// Request manual sync.
    pub fn request_sync(&mut self) {
        self.pending_trigger = Some(ReconciliationTrigger::Manual);
    }

    /// Get time since last sync.
    pub fn time_since_sync(&self) -> Duration {
        self.last_sync.elapsed()
    }

    /// Get reconciliation metrics.
    pub fn get_metrics(&self) -> ReconciliationMetrics {
        ReconciliationMetrics {
            sync_count: self.sync_count,
            drift_events: self.drift_events,
            last_sync_age_ms: self.last_sync.elapsed().as_millis() as u64,
            pending_sync: self.pending_trigger.is_some(),
        }
    }
}

/// Metrics for reconciliation observability.
#[derive(Debug, Clone)]
pub struct ReconciliationMetrics {
    /// Total syncs performed this session
    pub sync_count: u64,
    /// Number of drift events detected
    pub drift_events: u64,
    /// Milliseconds since last sync
    pub last_sync_age_ms: u64,
    /// Whether a sync is pending
    pub pending_sync: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_background_timer() {
        let config = ReconciliationConfig {
            background_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let mut reconciler = PositionReconciler::with_config(config);

        // Initially shouldn't need sync (just created)
        assert!(reconciler.should_sync().is_none());

        // Wait for timer
        std::thread::sleep(Duration::from_millis(15));

        // Now should trigger
        let trigger = reconciler.should_sync();
        assert!(matches!(trigger, Some(ReconciliationTrigger::Timer)));
    }

    #[test]
    fn test_rejection_trigger() {
        let mut reconciler = PositionReconciler::new();

        // Non-position error doesn't trigger
        assert!(!reconciler.on_order_rejection("insufficient margin"));

        // Position error triggers
        assert!(reconciler.on_order_rejection("exceeds maximum position"));
        assert!(reconciler.pending_trigger.is_some());
    }

    #[test]
    fn test_unmatched_fill_trigger() {
        let mut reconciler = PositionReconciler::new();

        reconciler.on_unmatched_fill(12345, 0.5);
        assert!(reconciler.pending_trigger.is_some());
        assert_eq!(reconciler.drift_events, 1);
    }

    #[test]
    fn test_large_position_change() {
        let mut reconciler = PositionReconciler::new();

        // Small change (< 5% of 1.0)
        assert!(!reconciler.on_position_change(0.01, 1.0));
        assert!(reconciler.pending_trigger.is_none());

        // Large change (> 5% of 1.0)
        assert!(reconciler.on_position_change(0.1, 1.0));
        assert!(reconciler.pending_trigger.is_some());
    }

    #[test]
    fn test_sync_completed() {
        let mut reconciler = PositionReconciler::new();
        reconciler.request_sync();

        assert!(reconciler.pending_trigger.is_some());

        reconciler.record_sync_completed();

        assert!(reconciler.pending_trigger.is_none());
        assert_eq!(reconciler.sync_count, 1);
    }
}
