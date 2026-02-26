//! Cancel-Race Adverse Selection Tracker
//!
//! Fills received during cancel attempts are systematically more toxic because
//! an informed trader beat our cancel. This module tracks the excess adverse
//! selection cost of race fills vs normal fills, producing an additive spread
//! floor component.
//!
//! Theory: When we send a cancel, the order is still live for `cancel_latency_ms`.
//! If a fill arrives in that window, an informed counterparty saw the signal
//! that triggered our cancel and raced to fill us before it went through.
//! These fills carry higher AS cost on average.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Configuration for cancel-race AS tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRaceConfig {
    /// Window after cancel request in which fills are classified as race fills (ms).
    /// Default: 500ms (typical cancel roundtrip on Hyperliquid).
    #[serde(default = "default_cancel_latency_window_ms")]
    pub cancel_latency_window_ms: u64,

    /// EWMA decay factor for race/normal AS tracking.
    /// Higher = faster adaptation. Default: 0.1 (~10-fill half-life).
    #[serde(default = "default_ewma_alpha")]
    pub ewma_alpha: f64,

    /// Minimum race fills before excess AS is considered significant.
    /// Below this count, excess_race_as_bps returns 0.
    #[serde(default = "default_min_race_fills")]
    pub min_race_fills: u64,

    /// Maximum excess AS in bps that can be added to spread floor.
    /// Prevents extreme outliers from blowing up the spread.
    #[serde(default = "default_max_excess_as_bps")]
    pub max_excess_as_bps: f64,
}

fn default_cancel_latency_window_ms() -> u64 {
    500
}
fn default_ewma_alpha() -> f64 {
    0.1
}
fn default_min_race_fills() -> u64 {
    10
}
fn default_max_excess_as_bps() -> f64 {
    5.0
}

impl Default for CancelRaceConfig {
    fn default() -> Self {
        Self {
            cancel_latency_window_ms: default_cancel_latency_window_ms(),
            ewma_alpha: default_ewma_alpha(),
            min_race_fills: default_min_race_fills(),
            max_excess_as_bps: default_max_excess_as_bps(),
        }
    }
}

/// Tracks adverse selection separately for cancel-race fills vs normal fills.
///
/// Usage:
/// 1. Call `record_cancel_request(oid)` when a cancel is sent
/// 2. Call `record_fill(oid, as_bps, timestamp_ms)` when a fill arrives
/// 3. Read `excess_race_as_bps()` for the additive spread floor component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRaceTracker {
    /// EWMA of AS (bps) for fills during cancel race windows
    race_fill_as_bps: f64,
    /// EWMA of AS (bps) for normal fills
    normal_fill_as_bps: f64,
    /// Total race fills observed
    race_fill_count: u64,
    /// Total normal fills observed
    normal_fill_count: u64,
    /// Pending cancel requests: oid -> cancel_sent_timestamp_ms
    #[serde(skip)]
    pending_cancels: HashMap<u64, u64>,
    /// Configuration
    #[serde(default)]
    config: CancelRaceConfig,

    // --- Drift-conditioned tracking ---
    /// EWMA of AS (bps) for race fills that occurred during drift (|momentum| > 5 bps).
    #[serde(default)]
    drift_race_fill_as_bps: f64,
    /// EWMA of AS (bps) for race fills that occurred without drift.
    #[serde(default)]
    no_drift_race_fill_as_bps: f64,
    /// Count of race fills during drift.
    #[serde(default)]
    drift_race_fill_count: u64,
    /// Count of race fills without drift.
    #[serde(default)]
    no_drift_race_fill_count: u64,
    /// Current momentum (bps), updated externally via `update_momentum()`.
    #[serde(default)]
    current_momentum_bps: f64,
}

impl CancelRaceTracker {
    /// Create a new tracker with default configuration.
    pub fn new() -> Self {
        Self::with_config(CancelRaceConfig::default())
    }

    /// Create a new tracker with custom configuration.
    pub fn with_config(config: CancelRaceConfig) -> Self {
        Self {
            race_fill_as_bps: 0.0,
            normal_fill_as_bps: 0.0,
            race_fill_count: 0,
            normal_fill_count: 0,
            pending_cancels: HashMap::new(),
            config,
            drift_race_fill_as_bps: 0.0,
            no_drift_race_fill_as_bps: 0.0,
            drift_race_fill_count: 0,
            no_drift_race_fill_count: 0,
            current_momentum_bps: 0.0,
        }
    }

    /// Record that a cancel request was sent for an order.
    /// `timestamp_ms` is the time the cancel was sent.
    pub fn record_cancel_request(&mut self, oid: u64, timestamp_ms: u64) {
        self.pending_cancels.insert(oid, timestamp_ms);

        // Prune old entries to prevent unbounded growth
        // Remove cancels older than 10x the window (definitely stale)
        let stale_cutoff_ms =
            timestamp_ms.saturating_sub(self.config.cancel_latency_window_ms * 10);
        self.pending_cancels.retain(|_, ts| *ts > stale_cutoff_ms);
    }

    /// Record a fill and classify it as race fill or normal fill.
    ///
    /// `oid`: order ID that was filled
    /// `as_bps`: measured adverse selection in basis points (absolute value)
    /// `timestamp_ms`: fill timestamp in milliseconds
    ///
    /// Returns `true` if this was classified as a race fill.
    pub fn record_fill(&mut self, oid: u64, as_bps: f64, timestamp_ms: u64) -> bool {
        let is_race_fill = self.is_race_fill(oid, timestamp_ms);
        let alpha = self.config.ewma_alpha;
        let as_abs = as_bps.abs();

        if is_race_fill {
            self.race_fill_as_bps = alpha * as_abs + (1.0 - alpha) * self.race_fill_as_bps;
            self.race_fill_count += 1;

            // Drift-conditioned classification: |momentum| > 5 bps threshold
            const DRIFT_THRESHOLD_BPS: f64 = 5.0;
            if self.current_momentum_bps.abs() > DRIFT_THRESHOLD_BPS {
                self.drift_race_fill_as_bps =
                    alpha * as_abs + (1.0 - alpha) * self.drift_race_fill_as_bps;
                self.drift_race_fill_count += 1;
            } else {
                self.no_drift_race_fill_as_bps =
                    alpha * as_abs + (1.0 - alpha) * self.no_drift_race_fill_as_bps;
                self.no_drift_race_fill_count += 1;
            }

            // Remove from pending cancels since it's been consumed
            self.pending_cancels.remove(&oid);

            debug!(
                oid,
                as_bps = %format!("{:.2}", as_abs),
                race_ewma_bps = %format!("{:.2}", self.race_fill_as_bps),
                race_count = self.race_fill_count,
                momentum_bps = %format!("{:.1}", self.current_momentum_bps),
                "Cancel-race fill recorded"
            );
        } else {
            self.normal_fill_as_bps = alpha * as_abs + (1.0 - alpha) * self.normal_fill_as_bps;
            self.normal_fill_count += 1;
        }

        is_race_fill
    }

    /// Check if a fill arrived within the cancel race window for this OID.
    fn is_race_fill(&self, oid: u64, fill_timestamp_ms: u64) -> bool {
        if let Some(&cancel_ts) = self.pending_cancels.get(&oid) {
            let elapsed_ms = fill_timestamp_ms.saturating_sub(cancel_ts);
            elapsed_ms <= self.config.cancel_latency_window_ms
        } else {
            false
        }
    }

    /// Compute the excess adverse selection of race fills over normal fills.
    ///
    /// Returns 0.0 if insufficient data or if race fills are not more toxic.
    /// Capped at `config.max_excess_as_bps`.
    pub fn excess_race_as_bps(&self) -> f64 {
        if self.race_fill_count < self.config.min_race_fills {
            return 0.0;
        }
        (self.race_fill_as_bps - self.normal_fill_as_bps)
            .max(0.0)
            .min(self.config.max_excess_as_bps)
    }

    /// Get the current race fill rate (fraction of all fills that are race fills).
    pub fn race_fill_rate(&self) -> f64 {
        let total = self.race_fill_count + self.normal_fill_count;
        if total == 0 {
            return 0.0;
        }
        self.race_fill_count as f64 / total as f64
    }

    /// Get the EWMA AS for race fills.
    pub fn race_fill_as_ewma_bps(&self) -> f64 {
        self.race_fill_as_bps
    }

    /// Get the EWMA AS for normal fills.
    pub fn normal_fill_as_ewma_bps(&self) -> f64 {
        self.normal_fill_as_bps
    }

    /// Get the total race fill count.
    pub fn race_fill_count(&self) -> u64 {
        self.race_fill_count
    }

    /// Get the total normal fill count.
    pub fn normal_fill_count(&self) -> u64 {
        self.normal_fill_count
    }

    /// Update the current momentum used for drift classification of race fills.
    pub fn update_momentum(&mut self, momentum_bps: f64) {
        self.current_momentum_bps = momentum_bps;
    }

    /// Excess AS of race fills during drift vs race fills without drift (bps, clamped >= 0).
    ///
    /// Returns 0.0 if insufficient drift race fills or no-drift race fills.
    pub fn excess_race_as_in_drift_bps(&self) -> f64 {
        if self.drift_race_fill_count < self.config.min_race_fills
            || self.no_drift_race_fill_count < self.config.min_race_fills
        {
            return 0.0;
        }
        (self.drift_race_fill_as_bps - self.no_drift_race_fill_as_bps).max(0.0)
    }

    /// Ratio of drift race AS / no-drift race AS (clamped >= 1.0).
    ///
    /// A ratio of 2.0 means race fills are 2x more toxic during drift.
    /// Returns 1.0 if insufficient data or no-drift AS is near zero.
    pub fn drift_excess_ratio(&self) -> f64 {
        if self.drift_race_fill_count < self.config.min_race_fills
            || self.no_drift_race_fill_count < self.config.min_race_fills
        {
            return 1.0;
        }
        if self.no_drift_race_fill_as_bps < 0.01 {
            return 1.0;
        }
        (self.drift_race_fill_as_bps / self.no_drift_race_fill_as_bps).max(1.0)
    }

    /// Get a summary for logging.
    pub fn summary(&self) -> CancelRaceSummary {
        CancelRaceSummary {
            race_fill_as_bps: self.race_fill_as_bps,
            normal_fill_as_bps: self.normal_fill_as_bps,
            excess_as_bps: self.excess_race_as_bps(),
            race_fill_count: self.race_fill_count,
            normal_fill_count: self.normal_fill_count,
            race_fill_rate: self.race_fill_rate(),
        }
    }
}

impl Default for CancelRaceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for cancel-race AS tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRaceSummary {
    pub race_fill_as_bps: f64,
    pub normal_fill_as_bps: f64,
    pub excess_as_bps: f64,
    pub race_fill_count: u64,
    pub normal_fill_count: u64,
    pub race_fill_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tracker_returns_zero() {
        let tracker = CancelRaceTracker::new();
        assert_eq!(tracker.excess_race_as_bps(), 0.0);
        assert_eq!(tracker.race_fill_rate(), 0.0);
        assert_eq!(tracker.race_fill_count(), 0);
        assert_eq!(tracker.normal_fill_count(), 0);
    }

    #[test]
    fn test_normal_fill_classification() {
        let mut tracker = CancelRaceTracker::new();

        // Fill without any cancel request → normal fill
        let is_race = tracker.record_fill(1, 5.0, 1000);
        assert!(!is_race);
        assert_eq!(tracker.normal_fill_count(), 1);
        assert_eq!(tracker.race_fill_count(), 0);
    }

    #[test]
    fn test_race_fill_within_window() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            min_race_fills: 1,
            ..Default::default()
        });

        // Send cancel at t=1000
        tracker.record_cancel_request(42, 1000);

        // Fill arrives at t=1200 (200ms later, within 500ms window)
        let is_race = tracker.record_fill(42, 10.0, 1200);
        assert!(is_race);
        assert_eq!(tracker.race_fill_count(), 1);
    }

    #[test]
    fn test_fill_outside_window_is_normal() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            min_race_fills: 1,
            ..Default::default()
        });

        // Send cancel at t=1000
        tracker.record_cancel_request(42, 1000);

        // Fill arrives at t=2000 (1000ms later, outside 500ms window)
        let is_race = tracker.record_fill(42, 10.0, 2000);
        assert!(!is_race);
        assert_eq!(tracker.normal_fill_count(), 1);
        assert_eq!(tracker.race_fill_count(), 0);
    }

    #[test]
    fn test_excess_as_computation() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            ewma_alpha: 1.0, // instant adaptation for deterministic testing
            min_race_fills: 1,
            max_excess_as_bps: 10.0,
        });

        // Record some normal fills with 3 bps AS
        for i in 0..5 {
            tracker.record_fill(100 + i, 3.0, 1000 + i * 100);
        }

        // Record race fills with 8 bps AS
        for i in 0..5 {
            let oid = 200 + i;
            tracker.record_cancel_request(oid, 5000 + i * 100);
            tracker.record_fill(oid, 8.0, 5050 + i * 100); // 50ms after cancel
        }

        // Excess = race(8.0) - normal(3.0) = 5.0 bps
        let excess = tracker.excess_race_as_bps();
        assert!(
            (excess - 5.0).abs() < 0.01,
            "Expected excess ~5.0 bps, got {:.2}",
            excess
        );
    }

    #[test]
    fn test_excess_as_capped() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            ewma_alpha: 1.0,
            min_race_fills: 1,
            max_excess_as_bps: 3.0, // Low cap
        });

        // Normal fills: 2 bps
        tracker.record_fill(1, 2.0, 1000);

        // Race fill: 20 bps (very toxic)
        tracker.record_cancel_request(2, 2000);
        tracker.record_fill(2, 20.0, 2100);

        let excess = tracker.excess_race_as_bps();
        assert!(
            (excess - 3.0).abs() < 0.01,
            "Expected excess capped at 3.0, got {:.2}",
            excess
        );
    }

    #[test]
    fn test_insufficient_data_returns_zero() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            min_race_fills: 10, // Require 10 race fills
            ewma_alpha: 1.0,
            ..Default::default()
        });

        // Only 5 race fills — not enough
        for i in 0..5 {
            tracker.record_cancel_request(i, 1000 + i * 100);
            tracker.record_fill(i, 15.0, 1050 + i * 100);
        }

        assert_eq!(
            tracker.excess_race_as_bps(),
            0.0,
            "Should return 0 with insufficient data"
        );
        assert_eq!(tracker.race_fill_count(), 5);
    }

    #[test]
    fn test_race_fills_not_more_toxic_returns_zero() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            ewma_alpha: 1.0,
            min_race_fills: 1,
            ..Default::default()
        });

        // Normal fills MORE toxic than race fills
        tracker.record_fill(1, 10.0, 1000);

        tracker.record_cancel_request(2, 2000);
        tracker.record_fill(2, 3.0, 2050);

        // Excess should be 0 (race fills less toxic)
        assert_eq!(
            tracker.excess_race_as_bps(),
            0.0,
            "Should return 0 when race fills are less toxic"
        );
    }

    #[test]
    fn test_race_fill_rate() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            ..Default::default()
        });

        // 3 normal fills
        for i in 0..3 {
            tracker.record_fill(i, 5.0, 1000 + i * 100);
        }

        // 2 race fills
        for i in 0..2 {
            let oid = 10 + i;
            tracker.record_cancel_request(oid, 5000 + i * 100);
            tracker.record_fill(oid, 8.0, 5050 + i * 100);
        }

        let rate = tracker.race_fill_rate();
        assert!(
            (rate - 0.4).abs() < 0.01,
            "Expected 2/5 = 0.4 race fill rate, got {:.3}",
            rate
        );
    }

    #[test]
    fn test_stale_cancels_pruned() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            ..Default::default()
        });

        // Record cancel at t=1000
        tracker.record_cancel_request(1, 1000);

        // Much later (t=100000), record another cancel — should prune the old one
        tracker.record_cancel_request(2, 100_000);

        // Old cancel should be pruned (10x window = 5000ms, t=1000 < 100000-5000)
        assert!(
            !tracker.pending_cancels.contains_key(&1),
            "Old cancel should be pruned"
        );
        assert!(tracker.pending_cancels.contains_key(&2));
    }

    #[test]
    fn test_ewma_decay() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            ewma_alpha: 0.5, // 50% decay
            min_race_fills: 1,
            ..Default::default()
        });

        // First race fill: 10 bps
        tracker.record_cancel_request(1, 1000);
        tracker.record_fill(1, 10.0, 1050);
        assert!(
            (tracker.race_fill_as_ewma_bps() - 5.0).abs() < 0.01,
            "First update: 0.5*10 + 0.5*0 = 5.0, got {:.2}",
            tracker.race_fill_as_ewma_bps()
        );

        // Second race fill: 10 bps
        tracker.record_cancel_request(2, 2000);
        tracker.record_fill(2, 10.0, 2050);
        assert!(
            (tracker.race_fill_as_ewma_bps() - 7.5).abs() < 0.01,
            "Second update: 0.5*10 + 0.5*5 = 7.5, got {:.2}",
            tracker.race_fill_as_ewma_bps()
        );
    }

    #[test]
    fn test_summary() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            ewma_alpha: 1.0,
            min_race_fills: 1,
            cancel_latency_window_ms: 500,
            max_excess_as_bps: 10.0,
        });

        tracker.record_fill(1, 3.0, 1000);
        tracker.record_cancel_request(2, 2000);
        tracker.record_fill(2, 8.0, 2050);

        let summary = tracker.summary();
        assert_eq!(summary.race_fill_count, 1);
        assert_eq!(summary.normal_fill_count, 1);
        assert!((summary.race_fill_as_bps - 8.0).abs() < 0.01);
        assert!((summary.normal_fill_as_bps - 3.0).abs() < 0.01);
        assert!((summary.excess_as_bps - 5.0).abs() < 0.01);
        assert!((summary.race_fill_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_drift_race_fills_tracked_higher_as() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            ewma_alpha: 1.0,
            min_race_fills: 1,
            max_excess_as_bps: 20.0,
        });

        // Race fills during drift (|momentum| > 5 bps) — high AS
        tracker.update_momentum(15.0);
        for i in 0..5 {
            let oid = 100 + i;
            tracker.record_cancel_request(oid, 1000 + i * 200);
            tracker.record_fill(oid, 12.0, 1050 + i * 200);
        }

        // Race fills without drift (|momentum| < 5 bps) — low AS
        tracker.update_momentum(2.0);
        for i in 0..5 {
            let oid = 200 + i;
            tracker.record_cancel_request(oid, 5000 + i * 200);
            tracker.record_fill(oid, 4.0, 5050 + i * 200);
        }

        assert_eq!(tracker.drift_race_fill_count, 5);
        assert_eq!(tracker.no_drift_race_fill_count, 5);
        assert!(
            (tracker.drift_race_fill_as_bps - 12.0).abs() < 0.01,
            "Drift race AS should be ~12 bps, got {:.2}",
            tracker.drift_race_fill_as_bps
        );
        assert!(
            (tracker.no_drift_race_fill_as_bps - 4.0).abs() < 0.01,
            "No-drift race AS should be ~4 bps, got {:.2}",
            tracker.no_drift_race_fill_as_bps
        );
    }

    #[test]
    fn test_no_drift_baseline_separate() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            ewma_alpha: 1.0,
            min_race_fills: 1,
            max_excess_as_bps: 20.0,
        });

        // All race fills during no-drift
        tracker.update_momentum(1.0);
        for i in 0..5 {
            let oid = 300 + i;
            tracker.record_cancel_request(oid, 1000 + i * 200);
            tracker.record_fill(oid, 6.0, 1050 + i * 200);
        }

        // Drift fills not populated → excess should be 0
        assert_eq!(tracker.drift_race_fill_count, 0);
        assert_eq!(tracker.no_drift_race_fill_count, 5);
        assert!(
            tracker.excess_race_as_in_drift_bps() == 0.0,
            "Excess should be 0 when no drift fills exist"
        );
        assert!(
            (tracker.drift_excess_ratio() - 1.0).abs() < f64::EPSILON,
            "Ratio should be 1.0 when no drift fills exist"
        );
    }

    #[test]
    fn test_drift_excess_ratio_above_one() {
        let mut tracker = CancelRaceTracker::with_config(CancelRaceConfig {
            cancel_latency_window_ms: 500,
            ewma_alpha: 1.0,
            min_race_fills: 1,
            max_excess_as_bps: 20.0,
        });

        // Drift race fills: 10 bps
        tracker.update_momentum(20.0);
        for i in 0..3 {
            let oid = 400 + i;
            tracker.record_cancel_request(oid, 1000 + i * 200);
            tracker.record_fill(oid, 10.0, 1050 + i * 200);
        }

        // No-drift race fills: 4 bps
        tracker.update_momentum(0.5);
        for i in 0..3 {
            let oid = 500 + i;
            tracker.record_cancel_request(oid, 5000 + i * 200);
            tracker.record_fill(oid, 4.0, 5050 + i * 200);
        }

        let excess = tracker.excess_race_as_in_drift_bps();
        assert!(
            (excess - 6.0).abs() < 0.01,
            "Expected excess ~6.0 bps (10-4), got {:.2}",
            excess
        );

        let ratio = tracker.drift_excess_ratio();
        assert!(
            (ratio - 2.5).abs() < 0.01,
            "Expected ratio ~2.5 (10/4), got {:.2}",
            ratio
        );
    }
}
