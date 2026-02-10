//! Predicted vs realized edge tracking.

use serde::{Deserialize, Serialize};

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
    pub fn add_snapshot(&mut self, snap: EdgeSnapshot) {
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

    /// Mean realized edge in basis points.
    pub fn mean_realized_edge(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.snapshots.iter().map(|s| s.realized_edge_bps).sum();
        sum / self.snapshots.len() as f64
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

    /// Returns an alarm multiplier if mean realized edge is negative.
    ///
    /// - `None` if fewer than 10 fills or edge is non-negative.
    /// - `Some(2.0)` if mean realized edge < -1 bps (strongly negative).
    /// - `Some(1.5)` if mean realized edge < 0 (mildly negative).
    pub fn negative_edge_alarm(&self) -> Option<f64> {
        if self.snapshots.len() < 10 {
            return None;
        }
        let mean = self.mean_realized_edge();
        if mean >= 0.0 {
            return None;
        }
        if mean < -1.0 {
            Some(2.0)
        } else {
            Some(1.5)
        }
    }

    /// Whether trading should be paused due to consistently negative edge.
    ///
    /// Returns `true` if 20+ fills and mean realized edge < -2 bps.
    pub fn should_pause_trading(&self) -> bool {
        if self.snapshots.len() < 20 {
            return false;
        }
        self.mean_realized_edge() < -2.0
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
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = ((rng_state >> 33) as usize) % n;
                sum += self.snapshots[idx].realized_edge_bps;
            }
            resampled_means.push(sum / n as f64);
        }

        resampled_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = (1.0 - confidence) / 2.0;
        let lower_idx = ((alpha * NUM_RESAMPLES as f64) as usize).min(NUM_RESAMPLES - 1);
        let upper_idx = (((1.0 - alpha) * NUM_RESAMPLES as f64) as usize).min(NUM_RESAMPLES - 1);

        (point, resampled_means[lower_idx], resampled_means[upper_idx])
    }

    /// Human-readable edge report.
    pub fn format_report(&self) -> String {
        if self.snapshots.is_empty() {
            return "No edge data".to_string();
        }

        let sig = if self.is_edge_positive() {
            "YES"
        } else {
            "NO"
        };

        format!(
            "Edge Metrics (n={}):\n  Predicted: {:.2} bps\n  Realized:  {:.2} bps\n  Error:     {:.2} bps\n  Significant at 95%: {}",
            self.edge_count(),
            self.mean_predicted_edge(),
            self.mean_realized_edge(),
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
        EdgeSnapshot {
            timestamp_ns: 0,
            predicted_spread_bps: predicted_bps + 1.5,
            realized_spread_bps: realized_bps + 1.5,
            predicted_as_bps: 0.0,
            realized_as_bps: 0.0,
            fee_bps: 1.5,
            predicted_edge_bps: predicted_bps,
            realized_edge_bps: realized_bps,
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
        // Mean realized edge = -0.5 bps (negative but > -1 bps)
        for _ in 0..10 {
            tracker.add_snapshot(make_snapshot(5.0, -0.5));
        }
        assert_eq!(tracker.negative_edge_alarm(), Some(1.5));
    }

    #[test]
    fn test_negative_edge_alarm_strong_negative() {
        let mut tracker = EdgeTracker::new();
        // Mean realized edge = -3.0 bps (< -1 bps)
        for _ in 0..10 {
            tracker.add_snapshot(make_snapshot(5.0, -3.0));
        }
        assert_eq!(tracker.negative_edge_alarm(), Some(2.0));
    }

    #[test]
    fn test_should_pause_trading() {
        let mut tracker = EdgeTracker::new();

        // Fewer than 20 fills — never pause
        for _ in 0..19 {
            tracker.add_snapshot(make_snapshot(5.0, -5.0));
        }
        assert!(!tracker.should_pause_trading());

        // Add one more to hit 20 fills with mean edge = -5.0 bps (< -2 bps)
        tracker.add_snapshot(make_snapshot(5.0, -5.0));
        assert!(tracker.should_pause_trading());

        // Tracker with mild negative edge (> -2 bps) should NOT pause
        let mut mild_tracker = EdgeTracker::new();
        for _ in 0..25 {
            mild_tracker.add_snapshot(make_snapshot(5.0, -1.0));
        }
        assert!(!mild_tracker.should_pause_trading());
    }

    #[test]
    fn test_last_realized_edge_empty() {
        let tracker = EdgeTracker::new();
        assert!((tracker.last_realized_edge_bps() - 0.0).abs() < 1e-10,
            "Empty tracker should return 0.0");
    }

    #[test]
    fn test_last_realized_edge_returns_most_recent() {
        let mut tracker = EdgeTracker::new();
        tracker.add_snapshot(make_snapshot(5.0, 2.0));
        assert!((tracker.last_realized_edge_bps() - 2.0).abs() < 1e-10);

        tracker.add_snapshot(make_snapshot(5.0, -1.5));
        assert!((tracker.last_realized_edge_bps() - (-1.5)).abs() < 1e-10,
            "Should return most recent, not average");
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
}
