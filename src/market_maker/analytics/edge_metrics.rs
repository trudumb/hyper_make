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
}
