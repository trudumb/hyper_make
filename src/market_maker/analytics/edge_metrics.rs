//! Predicted vs realized edge tracking.

use serde::{Deserialize, Serialize};

/// Phase of edge measurement lifecycle.
///
/// Pending: measured at fill time (AS estimated from classifier).
/// Resolved: updated after 5s markout (AS measured from actual mid movement).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgePhase {
    #[default]
    Pending,
    Resolved,
}

/// Per-fill edge measurement comparing predictions to realizations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSnapshot {
    pub timestamp_ns: u64,
    /// Predicted spread in basis points.
    pub predicted_spread_bps: f64,
    /// Realized spread in basis points.
    pub realized_spread_bps: f64,
    /// Predicted adverse selection in basis points.
    pub predicted_as_bps: f64,
    /// Realized adverse selection in basis points.
    pub realized_as_bps: f64,
    /// Trading fee in basis points.
    pub fee_bps: f64,
    /// Predicted edge: predicted_spread - predicted_as - fees.
    pub predicted_edge_bps: f64,
    /// Realized edge: realized_spread - realized_as - fees.
    pub realized_edge_bps: f64,
    /// Gross edge: spread_capture - adverse_selection (pre-fee).
    /// Measures model accuracy without fee drag.
    #[serde(default)]
    pub gross_edge_bps: f64,
    /// Phase of edge measurement: Pending (at fill) or Resolved (after markout).
    #[serde(default)]
    pub phase: EdgePhase,
    /// Mid price when the order was originally placed.
    #[serde(default)]
    pub mid_at_placement: f64,
    /// Markout-based adverse selection in bps (populated after 5s markout).
    #[serde(default)]
    pub markout_as_bps: Option<f64>,
}

/// Tracks edge snapshots and computes aggregate statistics.
///
/// Monitors whether predicted edge matches realized edge and whether
/// the strategy has statistically significant positive edge.
#[derive(Debug, Clone)]
pub struct EdgeTracker {
    snapshots: Vec<EdgeSnapshot>,
}

impl EdgeTracker {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    /// Record an edge measurement.
    ///
    /// INVARIANT: Only `EdgePhase::Resolved` snapshots should be added.
    /// Pending snapshots have placeholder zeroes for realized_edge_bps,
    /// gross_edge_bps, and realized_as_bps, which would corrupt all
    /// aggregate statistics. The caller (check_pending_fill_outcomes in
    /// handlers.rs) enforces this by only creating Resolved snapshots
    /// after the 5-second markout window.
    pub fn add_snapshot(&mut self, snap: EdgeSnapshot) {
        // Defensive guard: silently skip Pending snapshots to prevent
        // tautological edge measurements from corrupting statistics.
        if snap.phase == EdgePhase::Pending {
            return;
        }
        self.snapshots.push(snap);
    }

    /// Mean predicted edge in basis points.
    pub fn mean_predicted_edge(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.snapshots.iter().map(|s| s.predicted_edge_bps).sum();
        sum / self.snapshots.len() as f64
    }

    /// Mean realized edge in basis points (net of fees).
    pub fn mean_realized_edge(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.snapshots.iter().map(|s| s.realized_edge_bps).sum();
        sum / self.snapshots.len() as f64
    }

    /// Mean gross edge in basis points (pre-fee: spread capture minus AS).
    ///
    /// This measures model accuracy without fee drag. A fill at mid with zero AS
    /// has gross_edge=0 (neutral), not -1.5 (alarm from net edge).
    pub fn mean_gross_edge(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.snapshots.iter().map(|s| s.gross_edge_bps).sum();
        sum / self.snapshots.len() as f64
    }

    /// Variance of gross edge in basis points squared.
    ///
    /// High variance with neutral mean indicates uncertain model performance.
    pub fn gross_edge_variance(&self) -> f64 {
        let n = self.snapshots.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean_gross_edge();
        let n_f = n as f64;
        self.snapshots
            .iter()
            .map(|s| (s.gross_edge_bps - mean).powi(2))
            .sum::<f64>()
            / (n_f - 1.0)
    }

    /// Mean prediction error: mean(predicted - realized).
    pub fn edge_prediction_error(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .snapshots
            .iter()
            .map(|s| s.predicted_edge_bps - s.realized_edge_bps)
            .sum();
        sum / self.snapshots.len() as f64
    }

    /// Number of edge snapshots recorded.
    pub fn edge_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Most recent realized edge in basis points, or 0.0 if no fills yet.
    ///
    /// Used by the stochastic controller as a TD reward signal for Quote actions.
    pub fn last_realized_edge_bps(&self) -> f64 {
        self.snapshots
            .last()
            .map(|s| s.realized_edge_bps)
            .unwrap_or(0.0)
    }

    /// Whether mean realized edge is positive at 95% confidence (one-sided t-test).
    ///
    /// Uses t-statistic: `t = mean / (std / sqrt(n))`, rejects if `t > 1.645`.
    pub fn is_edge_positive(&self) -> bool {
        const T_CRITICAL_95: f64 = 1.645;

        let n = self.snapshots.len();
        if n < 3 {
            return false;
        }

        let mean = self.mean_realized_edge();
        let n_f = n as f64;
        let variance = self
            .snapshots
            .iter()
            .map(|s| (s.realized_edge_bps - mean).powi(2))
            .sum::<f64>()
            / (n_f - 1.0);
        let std = variance.sqrt();

        if std < 1e-12 {
            return mean > 0.0;
        }

        let t_stat = mean / (std / n_f.sqrt());
        t_stat > T_CRITICAL_95
    }

    /// Returns an alarm multiplier if mean gross edge is negative.
    ///
    /// Uses gross edge (pre-fee) so that fills at mid with zero AS don't trigger
    /// false alarms from fee drag alone.
    ///
    /// - `None` if fewer than 10 fills or gross edge is non-negative.
    /// - `Some(2.0)` if mean gross edge < -1 bps (strongly negative).
    /// - `Some(1.5)` if mean gross edge < 0 (mildly negative).
    pub fn negative_edge_alarm(&self) -> Option<f64> {
        if self.snapshots.len() < 10 {
            return None;
        }
        let mean = self.mean_gross_edge();
        if mean >= 0.0 {
            return None;
        }
        if mean < -1.0 {
            Some(2.0)
        } else {
            Some(1.5)
        }
    }

    /// Whether trading should be paused due to negative edge.
    ///
    /// Always returns `false` — never stop quoting from edge data alone.
    /// Use `max_defensive_multiplier()` for graduated spread widening instead.
    pub fn should_pause_trading(&self) -> bool {
        false
    }

    /// Smooth defensive spread multiplier based on gross edge.
    ///
    /// Returns a multiplier in `[1.0, 5.0]`:
    /// - 1.0 when `mean_gross_edge >= 0` (no widening needed)
    /// - Linear ramp to 5.0 when `mean_gross_edge <= -3 bps`
    ///
    /// Avoids cliff effects from binary alarm thresholds.
    pub fn max_defensive_multiplier(&self) -> f64 {
        if self.snapshots.len() < 10 {
            return 1.0;
        }
        let gross = self.mean_gross_edge();
        1.0 + 4.0 * (-gross / 3.0).clamp(0.0, 1.0)
    }

    /// Bootstrapped confidence interval for mean realized edge in basis points.
    ///
    /// Returns `(mean_edge_bps, lower_ci, upper_ci)`.
    /// Uses 1000 bootstrap resamples of the realized edge observations.
    /// `confidence` should be in (0, 1), e.g., 0.90 for 90% CI.
    pub fn edge_with_confidence(&self, confidence: f64) -> (f64, f64, f64) {
        let point = self.mean_realized_edge();
        let n = self.snapshots.len();
        if n < 2 {
            return (point, point, point);
        }

        const NUM_RESAMPLES: usize = 1000;
        let mut resampled_means = Vec::with_capacity(NUM_RESAMPLES);
        let mut rng_state: u64 = 0xA1B2_C3D4_E5F6_7890 ^ (n as u64);

        for _ in 0..NUM_RESAMPLES {
            let mut sum = 0.0;
            for _ in 0..n {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let idx = ((rng_state >> 33) as usize) % n;
                sum += self.snapshots[idx].realized_edge_bps;
            }
            resampled_means.push(sum / n as f64);
        }

        resampled_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = (1.0 - confidence) / 2.0;
        let lower_idx = ((alpha * NUM_RESAMPLES as f64) as usize).min(NUM_RESAMPLES - 1);
        let upper_idx = (((1.0 - alpha) * NUM_RESAMPLES as f64) as usize).min(NUM_RESAMPLES - 1);

        (
            point,
            resampled_means[lower_idx],
            resampled_means[upper_idx],
        )
    }

    /// Human-readable edge report.
    pub fn format_report(&self) -> String {
        if self.snapshots.is_empty() {
            return "No edge data".to_string();
        }

        let sig = if self.is_edge_positive() { "YES" } else { "NO" };

        format!(
            "Edge Metrics (n={}):\n  Predicted: {:.2} bps\n  Realized:  {:.2} bps\n  Gross:     {:.2} bps\n  Error:     {:.2} bps\n  Significant at 95%: {}",
            self.edge_count(),
            self.mean_predicted_edge(),
            self.mean_realized_edge(),
            self.mean_gross_edge(),
            self.edge_prediction_error(),
            sig,
        )
    }
}

impl Default for EdgeTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(predicted_bps: f64, realized_bps: f64) -> EdgeSnapshot {
        make_snapshot_with_gross(predicted_bps, realized_bps, realized_bps + 1.5)
    }

    fn make_snapshot_with_gross(
        predicted_bps: f64,
        realized_bps: f64,
        gross_bps: f64,
    ) -> EdgeSnapshot {
        EdgeSnapshot {
            timestamp_ns: 0,
            predicted_spread_bps: predicted_bps + 1.5,
            realized_spread_bps: realized_bps + 1.5,
            predicted_as_bps: 0.0,
            realized_as_bps: 0.0,
            fee_bps: 1.5,
            predicted_edge_bps: predicted_bps,
            realized_edge_bps: realized_bps,
            gross_edge_bps: gross_bps,
            phase: EdgePhase::Resolved,
            mid_at_placement: 0.0,
            markout_as_bps: None,
        }
    }

    #[test]
    fn test_mean_edge() {
        let mut tracker = EdgeTracker::new();
        tracker.add_snapshot(make_snapshot(5.0, 4.0));
        tracker.add_snapshot(make_snapshot(3.0, 2.0));
        tracker.add_snapshot(make_snapshot(7.0, 6.0));

        assert!((tracker.mean_predicted_edge() - 5.0).abs() < 1e-10);
        assert!((tracker.mean_realized_edge() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_error() {
        let mut tracker = EdgeTracker::new();
        // Predicted always 2 bps above realized
        tracker.add_snapshot(make_snapshot(5.0, 3.0));
        tracker.add_snapshot(make_snapshot(4.0, 2.0));

        let error = tracker.edge_prediction_error();
        assert!((error - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_edge_positive() {
        let mut tracker = EdgeTracker::new();

        // Not enough data
        assert!(!tracker.is_edge_positive());

        // Add consistently positive realized edge
        for _ in 0..100 {
            tracker.add_snapshot(make_snapshot(5.0, 4.0));
        }

        // With 100 observations of constant positive edge, should be significant
        assert!(tracker.is_edge_positive());
    }

    #[test]
    fn test_is_edge_positive_negative() {
        let mut tracker = EdgeTracker::new();

        // Add consistently negative realized edge
        for _ in 0..100 {
            tracker.add_snapshot(make_snapshot(5.0, -2.0));
        }

        assert!(!tracker.is_edge_positive());
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = EdgeTracker::new();
        assert_eq!(tracker.mean_predicted_edge(), 0.0);
        assert_eq!(tracker.mean_realized_edge(), 0.0);
        assert_eq!(tracker.edge_prediction_error(), 0.0);
        assert_eq!(tracker.edge_count(), 0);
        assert!(!tracker.is_edge_positive());
    }

    #[test]
    fn test_format_report() {
        let mut tracker = EdgeTracker::new();
        tracker.add_snapshot(make_snapshot(5.0, 3.0));
        tracker.add_snapshot(make_snapshot(4.0, 2.0));
        tracker.add_snapshot(make_snapshot(6.0, 4.0));

        let report = tracker.format_report();
        assert!(report.contains("Edge Metrics"));
        assert!(report.contains("Predicted"));
        assert!(report.contains("Realized"));
    }

    #[test]
    fn test_format_report_empty() {
        let tracker = EdgeTracker::new();
        assert_eq!(tracker.format_report(), "No edge data");
    }

    #[test]
    fn test_negative_edge_alarm_insufficient_data() {
        let mut tracker = EdgeTracker::new();
        // Add fewer than 10 fills with negative edge
        for _ in 0..9 {
            tracker.add_snapshot(make_snapshot(5.0, -3.0));
        }
        assert!(tracker.negative_edge_alarm().is_none());
    }

    #[test]
    fn test_negative_edge_alarm_positive_edge() {
        let mut tracker = EdgeTracker::new();
        for _ in 0..20 {
            tracker.add_snapshot(make_snapshot(5.0, 3.0));
        }
        assert!(tracker.negative_edge_alarm().is_none());
    }

    #[test]
    fn test_negative_edge_alarm_mild_negative() {
        let mut tracker = EdgeTracker::new();
        // Mean gross edge = -0.5 bps (negative but > -1 bps)
        for _ in 0..10 {
            tracker.add_snapshot(make_snapshot_with_gross(5.0, -2.0, -0.5));
        }
        assert_eq!(tracker.negative_edge_alarm(), Some(1.5));
    }

    #[test]
    fn test_negative_edge_alarm_strong_negative() {
        let mut tracker = EdgeTracker::new();
        // Mean gross edge = -3.0 bps (< -1 bps)
        for _ in 0..10 {
            tracker.add_snapshot(make_snapshot_with_gross(5.0, -4.5, -3.0));
        }
        assert_eq!(tracker.negative_edge_alarm(), Some(2.0));
    }

    #[test]
    fn test_should_pause_trading_always_false() {
        let mut tracker = EdgeTracker::new();

        // Empty tracker — never pause
        assert!(!tracker.should_pause_trading());

        // Even with strongly negative edge and many fills — still never pause
        for _ in 0..30 {
            tracker.add_snapshot(make_snapshot_with_gross(5.0, -5.0, -3.5));
        }
        assert!(!tracker.should_pause_trading());

        // Mild negative — also never pause
        let mut mild_tracker = EdgeTracker::new();
        for _ in 0..25 {
            mild_tracker.add_snapshot(make_snapshot(5.0, -1.0));
        }
        assert!(!mild_tracker.should_pause_trading());
    }

    #[test]
    fn test_last_realized_edge_empty() {
        let tracker = EdgeTracker::new();
        assert!(
            (tracker.last_realized_edge_bps() - 0.0).abs() < 1e-10,
            "Empty tracker should return 0.0"
        );
    }

    #[test]
    fn test_last_realized_edge_returns_most_recent() {
        let mut tracker = EdgeTracker::new();
        tracker.add_snapshot(make_snapshot(5.0, 2.0));
        assert!((tracker.last_realized_edge_bps() - 2.0).abs() < 1e-10);

        tracker.add_snapshot(make_snapshot(5.0, -1.5));
        assert!(
            (tracker.last_realized_edge_bps() - (-1.5)).abs() < 1e-10,
            "Should return most recent, not average"
        );
    }

    #[test]
    fn test_edge_with_confidence() {
        let mut tracker = EdgeTracker::new();
        // 50 fills with positive edge and some variance
        for i in 0..50 {
            let realized = 2.0 + (i as f64 % 4.0); // 2, 3, 4, 5, 2, 3, ...
            tracker.add_snapshot(make_snapshot(5.0, realized));
        }
        let (point, lower, upper) = tracker.edge_with_confidence(0.90);
        assert!(point > 0.0, "Expected positive edge, got {point}");
        assert!(lower <= point, "Lower CI {lower} > point {point}");
        assert!(upper >= point, "Upper CI {upper} < point {point}");
        assert!(upper > lower, "CI has zero width: [{lower}, {upper}]");
    }

    #[test]
    fn test_edge_with_confidence_insufficient_data() {
        let mut tracker = EdgeTracker::new();
        tracker.add_snapshot(make_snapshot(5.0, 3.0));
        let (point, lower, upper) = tracker.edge_with_confidence(0.90);
        assert!((point - 3.0).abs() < 1e-10);
        assert_eq!(lower, point);
        assert_eq!(upper, point);
    }

    #[test]
    fn test_mean_gross_edge() {
        let mut tracker = EdgeTracker::new();
        assert_eq!(tracker.mean_gross_edge(), 0.0);

        // gross_edge = realized + fee = 4.0 + 1.5 = 5.5, 2.0 + 1.5 = 3.5, 6.0 + 1.5 = 7.5
        tracker.add_snapshot(make_snapshot(5.0, 4.0)); // gross=5.5
        tracker.add_snapshot(make_snapshot(3.0, 2.0)); // gross=3.5
        tracker.add_snapshot(make_snapshot(7.0, 6.0)); // gross=7.5

        let expected = (5.5 + 3.5 + 7.5) / 3.0;
        assert!((tracker.mean_gross_edge() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gross_edge_separate_from_net() {
        let mut tracker = EdgeTracker::new();
        // Set gross_edge independently from realized_edge
        tracker.add_snapshot(make_snapshot_with_gross(5.0, -2.0, 1.0));
        tracker.add_snapshot(make_snapshot_with_gross(5.0, -1.0, 2.0));

        // Net edge is negative
        assert!(tracker.mean_realized_edge() < 0.0);
        // Gross edge is positive
        assert!(tracker.mean_gross_edge() > 0.0);
    }

    #[test]
    fn test_defensive_multiplier_positive_gross() {
        let mut tracker = EdgeTracker::new();
        // Fewer than 10 fills — returns 1.0
        for _ in 0..5 {
            tracker.add_snapshot(make_snapshot_with_gross(5.0, 3.0, 2.0));
        }
        assert!((tracker.max_defensive_multiplier() - 1.0).abs() < 1e-10);

        // 10+ fills with positive gross edge — returns 1.0
        for _ in 0..5 {
            tracker.add_snapshot(make_snapshot_with_gross(5.0, 3.0, 2.0));
        }
        assert!((tracker.max_defensive_multiplier() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_defensive_multiplier_ramp() {
        // gross = -1.5 bps → mult = 1 + 4*(1.5/3.0) = 1 + 2.0 = 3.0
        let mut tracker = EdgeTracker::new();
        for _ in 0..15 {
            tracker.add_snapshot(make_snapshot_with_gross(5.0, -3.0, -1.5));
        }
        assert!((tracker.max_defensive_multiplier() - 3.0).abs() < 1e-10);

        // gross = -3.0 bps → mult = 1 + 4*(3.0/3.0).clamp = 1 + 4 = 5.0
        let mut tracker2 = EdgeTracker::new();
        for _ in 0..15 {
            tracker2.add_snapshot(make_snapshot_with_gross(5.0, -4.5, -3.0));
        }
        assert!((tracker2.max_defensive_multiplier() - 5.0).abs() < 1e-10);

        // gross = -6.0 bps → clamped to 5.0
        let mut tracker3 = EdgeTracker::new();
        for _ in 0..15 {
            tracker3.add_snapshot(make_snapshot_with_gross(5.0, -7.5, -6.0));
        }
        assert!((tracker3.max_defensive_multiplier() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gross_edge_variance() {
        let mut tracker = EdgeTracker::new();
        // Fewer than 2 — returns 0.0
        assert_eq!(tracker.gross_edge_variance(), 0.0);
        tracker.add_snapshot(make_snapshot_with_gross(5.0, 3.0, 2.0));
        assert_eq!(tracker.gross_edge_variance(), 0.0);

        // Constant gross edge — variance = 0
        let mut const_tracker = EdgeTracker::new();
        for _ in 0..10 {
            const_tracker.add_snapshot(make_snapshot_with_gross(5.0, 3.0, 2.0));
        }
        assert!(const_tracker.gross_edge_variance().abs() < 1e-10);

        // Known variance: values 1.0 and 3.0 alternating → mean=2.0, var=1.0 (sample)
        // Actually with n=2: var = ((1-2)^2 + (3-2)^2) / (2-1) = 2/1 = 2.0
        // With n=10 (5 each): var = (5*1 + 5*1) / 9 = 10/9 ≈ 1.111
        let mut var_tracker = EdgeTracker::new();
        for i in 0..10 {
            let gross = if i % 2 == 0 { 1.0 } else { 3.0 };
            var_tracker.add_snapshot(make_snapshot_with_gross(5.0, 0.0, gross));
        }
        let var = var_tracker.gross_edge_variance();
        assert!((var - 10.0 / 9.0).abs() < 1e-10, "Expected 10/9, got {var}");
    }

    #[test]
    fn test_edge_phase_default_is_pending() {
        let phase = EdgePhase::default();
        assert_eq!(phase, EdgePhase::Pending);
    }

    #[test]
    fn test_edge_snapshot_serde_roundtrip_with_new_fields() {
        let snap = EdgeSnapshot {
            timestamp_ns: 42000,
            predicted_spread_bps: 5.0,
            realized_spread_bps: 4.0,
            predicted_as_bps: 1.0,
            realized_as_bps: 0.5,
            fee_bps: 1.5,
            predicted_edge_bps: 2.5,
            realized_edge_bps: 2.0,
            gross_edge_bps: 3.5,
            phase: EdgePhase::Resolved,
            mid_at_placement: 50_000.0,
            markout_as_bps: Some(0.8),
        };
        let json = serde_json::to_string(&snap).unwrap();
        let deser: EdgeSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.phase, EdgePhase::Resolved);
        assert!((deser.mid_at_placement - 50_000.0).abs() < f64::EPSILON);
        assert_eq!(deser.markout_as_bps, Some(0.8));
    }

    #[test]
    fn test_edge_snapshot_serde_backward_compat() {
        // Old JSON without the new fields — should deserialize with defaults
        let old_json = r#"{
            "timestamp_ns": 1000,
            "predicted_spread_bps": 5.0,
            "realized_spread_bps": 4.0,
            "predicted_as_bps": 1.0,
            "realized_as_bps": 0.5,
            "fee_bps": 1.5,
            "predicted_edge_bps": 2.5,
            "realized_edge_bps": 2.0,
            "gross_edge_bps": 3.5
        }"#;
        let snap: EdgeSnapshot = serde_json::from_str(old_json).unwrap();
        assert_eq!(snap.phase, EdgePhase::Pending);
        assert!((snap.mid_at_placement - 0.0).abs() < f64::EPSILON);
        assert_eq!(snap.markout_as_bps, None);
    }

    #[test]
    fn test_edge_snapshot_pending_vs_resolved() {
        let pending = EdgeSnapshot {
            timestamp_ns: 1000,
            predicted_spread_bps: 5.0,
            realized_spread_bps: 4.0,
            predicted_as_bps: 1.0,
            realized_as_bps: 0.5,
            fee_bps: 1.5,
            predicted_edge_bps: 2.5,
            realized_edge_bps: 2.0,
            gross_edge_bps: 3.5,
            phase: EdgePhase::Pending,
            mid_at_placement: 50_000.0,
            markout_as_bps: None,
        };
        assert_eq!(pending.phase, EdgePhase::Pending);
        assert!(pending.markout_as_bps.is_none());

        // Simulate resolution: update phase and set markout
        let resolved = EdgeSnapshot {
            phase: EdgePhase::Resolved,
            markout_as_bps: Some(1.2),
            ..pending
        };
        assert_eq!(resolved.phase, EdgePhase::Resolved);
        assert_eq!(resolved.markout_as_bps, Some(1.2));
        // Original fields preserved
        assert!((resolved.mid_at_placement - 50_000.0).abs() < f64::EPSILON);
        assert!((resolved.realized_edge_bps - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add_snapshot_rejects_pending() {
        let mut tracker = EdgeTracker::new();

        // Pending snapshot should be silently rejected
        let pending = EdgeSnapshot {
            timestamp_ns: 1000,
            predicted_spread_bps: 5.0,
            realized_spread_bps: 4.0,
            predicted_as_bps: 1.0,
            realized_as_bps: 0.0,
            fee_bps: 1.5,
            predicted_edge_bps: 2.5,
            realized_edge_bps: 0.0,
            gross_edge_bps: 0.0,
            phase: EdgePhase::Pending,
            mid_at_placement: 50_000.0,
            markout_as_bps: None,
        };
        tracker.add_snapshot(pending);
        assert_eq!(
            tracker.edge_count(),
            0,
            "Pending snapshots should not be added"
        );

        // Resolved snapshot should be accepted
        let resolved = EdgeSnapshot {
            timestamp_ns: 6000,
            predicted_spread_bps: 5.0,
            realized_spread_bps: 4.0,
            predicted_as_bps: 1.0,
            realized_as_bps: 0.8,
            fee_bps: 1.5,
            predicted_edge_bps: 2.5,
            realized_edge_bps: 1.7,
            gross_edge_bps: 3.2,
            phase: EdgePhase::Resolved,
            mid_at_placement: 50_000.0,
            markout_as_bps: Some(0.8),
        };
        tracker.add_snapshot(resolved);
        assert_eq!(
            tracker.edge_count(),
            1,
            "Resolved snapshots should be added"
        );
    }
}
