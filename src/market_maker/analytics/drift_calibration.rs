//! Drift calibration tracking for monitoring drift prediction quality.
//!
//! Provides:
//! - **Drift prediction hit rate**: `sign(drift) == sign(5s_price_move)` via EWMA
//! - **Brier score of continuation**: `(predicted - actual)^2` via EWMA
//! - **Fill rate during opposition vs alignment**: separate counters
//! - **Emergency pull frequency**: counts by urgency level (low/medium/high)

use serde::{Deserialize, Serialize};

/// Tracks drift prediction quality for calibration monitoring.
///
/// All running averages use exponential weighting with configurable decay `alpha`.
/// Higher alpha gives more weight to recent observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftCalibrationTracker {
    /// EWMA hit rate: sign(drift) == sign(5s move)
    prediction_hit_rate: f64,
    /// EWMA Brier score for continuation probability
    continuation_brier: f64,
    /// Fills per update cycle when position opposes trend
    fill_rate_opposed: f64,
    /// Fills per update cycle when position aligned with trend
    fill_rate_aligned: f64,
    /// Emergency pull counts by urgency \[low, medium, high\]
    emergency_pull_count: [u64; 3],
    /// Total predictions recorded
    total_predictions: u64,
    /// Total fills in opposed state
    opposed_fills: u64,
    /// Total fills in aligned state
    aligned_fills: u64,
    /// EWMA decay factor
    #[serde(default = "default_alpha")]
    alpha: f64,
}

fn default_alpha() -> f64 {
    0.05
}

impl DriftCalibrationTracker {
    /// Create a new tracker with the given EWMA decay factor.
    ///
    /// `alpha` controls how quickly the running averages respond to new data.
    /// Typical value: 0.05 (roughly 20-observation half-life).
    pub fn new(alpha: f64) -> Self {
        Self {
            prediction_hit_rate: 0.5, // uninformative prior
            continuation_brier: 0.25, // uninformative prior (0.5^2)
            fill_rate_opposed: 0.0,
            fill_rate_aligned: 0.0,
            emergency_pull_count: [0; 3],
            total_predictions: 0,
            opposed_fills: 0,
            aligned_fills: 0,
            alpha,
        }
    }

    /// Record a drift prediction outcome.
    ///
    /// `drift_sign` is the predicted drift direction (positive = up, negative = down).
    /// `actual_5s_move` is the realized 5-second price change.
    /// Updates the EWMA hit rate based on whether the sign matched.
    pub fn record_prediction(&mut self, drift_sign: f64, actual_5s_move: f64) {
        // Skip trivially zero predictions or moves
        if drift_sign == 0.0 || actual_5s_move == 0.0 {
            return;
        }
        let hit = if drift_sign.signum() == actual_5s_move.signum() {
            1.0
        } else {
            0.0
        };
        self.prediction_hit_rate = self.alpha * hit + (1.0 - self.alpha) * self.prediction_hit_rate;
        self.total_predictions += 1;
    }

    /// Record a continuation probability prediction outcome.
    ///
    /// `predicted` is the model's predicted probability of continuation (0.0..1.0).
    /// `actual` is whether continuation actually occurred.
    /// Updates the EWMA Brier score.
    pub fn record_continuation(&mut self, predicted: f64, actual: bool) {
        let actual_f = if actual { 1.0 } else { 0.0 };
        let brier = (predicted - actual_f).powi(2);
        self.continuation_brier = self.alpha * brier + (1.0 - self.alpha) * self.continuation_brier;
    }

    /// Record a fill event, distinguishing position-opposed vs position-aligned fills.
    ///
    /// `is_opposed` is true when the fill occurred while the position was opposing
    /// the estimated drift direction.
    pub fn record_fill(&mut self, is_opposed: bool) {
        if is_opposed {
            self.opposed_fills += 1;
        } else {
            self.aligned_fills += 1;
        }
    }

    /// Record an emergency pull event by urgency level.
    ///
    /// `urgency_level`: 0 = low, 1 = medium, 2 = high.
    /// Out-of-range values are clamped to 2 (high).
    pub fn record_emergency_pull(&mut self, urgency_level: usize) {
        let idx = urgency_level.min(2);
        self.emergency_pull_count[idx] += 1;
    }

    /// Current EWMA prediction hit rate.
    pub fn hit_rate(&self) -> f64 {
        self.prediction_hit_rate
    }

    /// Current EWMA Brier score.
    pub fn brier_score(&self) -> f64 {
        self.continuation_brier
    }

    /// Total opposed fills recorded.
    pub fn opposed_fills(&self) -> u64 {
        self.opposed_fills
    }

    /// Total aligned fills recorded.
    pub fn aligned_fills(&self) -> u64 {
        self.aligned_fills
    }

    /// Emergency pull counts as [low, medium, high].
    pub fn emergency_pull_counts(&self) -> [u64; 3] {
        self.emergency_pull_count
    }

    /// Total predictions recorded.
    pub fn total_predictions(&self) -> u64 {
        self.total_predictions
    }

    /// Returns a JSONL-formatted summary line with all metrics.
    pub fn log_summary(&self) -> String {
        format!(
            "{{\"type\":\"drift_calibration\",\"hit_rate\":{:.4},\"brier\":{:.4},\"opposed_fills\":{},\"aligned_fills\":{},\"pulls\":[{},{},{}],\"total_predictions\":{}}}",
            self.prediction_hit_rate,
            self.continuation_brier,
            self.opposed_fills,
            self.aligned_fills,
            self.emergency_pull_count[0],
            self.emergency_pull_count[1],
            self.emergency_pull_count[2],
            self.total_predictions,
        )
    }
}

impl Default for DriftCalibrationTracker {
    fn default() -> Self {
        Self::new(default_alpha())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opposition_fill_recording() {
        let mut tracker = DriftCalibrationTracker::new(0.05);

        // Record opposed fills
        tracker.record_fill(true);
        tracker.record_fill(true);
        tracker.record_fill(true);

        // Record aligned fills
        tracker.record_fill(false);

        assert_eq!(tracker.opposed_fills(), 3);
        assert_eq!(tracker.aligned_fills(), 1);

        // Record more
        tracker.record_fill(false);
        tracker.record_fill(true);
        assert_eq!(tracker.opposed_fills(), 4);
        assert_eq!(tracker.aligned_fills(), 2);
    }

    #[test]
    fn test_hit_rate_converges() {
        let mut tracker = DriftCalibrationTracker::new(0.1);

        // Feed 200 perfectly correct predictions: drift_sign positive, actual positive
        for _ in 0..200 {
            tracker.record_prediction(1.0, 0.5);
        }

        // After 200 correct predictions with alpha=0.1, EWMA should be very close to 1.0
        assert!(
            tracker.hit_rate() > 0.99,
            "Hit rate should converge to 1.0 for all-correct predictions, got {}",
            tracker.hit_rate()
        );

        // Now feed 200 wrong predictions
        for _ in 0..200 {
            tracker.record_prediction(1.0, -0.5);
        }

        // Should converge toward 0.0
        assert!(
            tracker.hit_rate() < 0.01,
            "Hit rate should converge to 0.0 for all-wrong predictions, got {}",
            tracker.hit_rate()
        );

        assert_eq!(tracker.total_predictions(), 400);
    }

    #[test]
    fn test_log_summary_valid_json() {
        let mut tracker = DriftCalibrationTracker::new(0.05);

        // Populate some data
        tracker.record_prediction(1.0, 0.3);
        tracker.record_prediction(-1.0, -0.2);
        tracker.record_continuation(0.7, true);
        tracker.record_continuation(0.3, false);
        tracker.record_fill(true);
        tracker.record_fill(false);
        tracker.record_fill(false);
        tracker.record_emergency_pull(0);
        tracker.record_emergency_pull(1);
        tracker.record_emergency_pull(2);
        tracker.record_emergency_pull(2);

        let summary = tracker.log_summary();

        // Verify it parses as valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&summary).expect("log_summary must produce valid JSON");

        // Verify required fields exist
        assert_eq!(parsed["type"], "drift_calibration");
        assert!(parsed["hit_rate"].is_number());
        assert!(parsed["brier"].is_number());
        assert_eq!(parsed["opposed_fills"], 1);
        assert_eq!(parsed["aligned_fills"], 2);
        assert_eq!(parsed["pulls"][0], 1);
        assert_eq!(parsed["pulls"][1], 1);
        assert_eq!(parsed["pulls"][2], 2);
        assert_eq!(parsed["total_predictions"], 2);
    }
}
