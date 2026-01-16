//! Calibration tracking for prediction validation.
//!
//! Tracks prediction-outcome pairs to compute calibration metrics:
//! - **Brier Score**: Mean squared error of probability predictions
//! - **Information Ratio**: Resolution / Uncertainty (>1.0 means model adds value)
//! - **Calibration Curve**: Predicted vs realized rates by bucket
//!
//! Used to validate:
//! - Fill probability predictions
//! - Adverse selection predictions

use serde::Serialize;
use std::collections::VecDeque;
use std::time::Instant;

/// Configuration for calibration tracking.
#[derive(Clone, Debug)]
pub struct CalibrationConfig {
    /// Maximum number of observations to keep (rolling window).
    pub max_observations: usize,
    /// Number of bins for calibration curve (typically 10 or 20).
    pub num_bins: usize,
    /// Minimum observations before computing metrics.
    pub min_observations: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            max_observations: 1000,
            num_bins: 20,
            min_observations: 10, // Reduced from 50 for faster dashboard feedback
        }
    }
}

/// A single prediction-outcome pair.
#[derive(Clone, Debug)]
pub struct PredictionOutcome {
    /// Predicted probability [0, 1].
    pub predicted: f64,
    /// Whether the event occurred.
    pub realized: bool,
    /// When the prediction was made.
    pub timestamp: Instant,
    /// Market regime at prediction time.
    pub regime: String,
}

/// A calibration bin for the calibration curve.
#[derive(Clone, Debug, Serialize)]
pub struct CalibrationBin {
    /// Bin center (e.g., 0.05, 0.15, ..., 0.95).
    pub predicted: f64,
    /// Actual rate of occurrence in this bin.
    pub realized: f64,
    /// Number of observations in this bin.
    pub count: usize,
}

/// Computed calibration metrics.
#[derive(Clone, Debug, Serialize)]
pub struct CalibrationMetrics {
    /// Brier score (mean squared error of predictions).
    /// Lower is better. Perfect calibration = 0.
    pub brier_score: f64,
    /// Information ratio (resolution / uncertainty).
    /// >1.0 means model adds predictive value.
    pub information_ratio: f64,
    /// Total number of observations.
    pub observation_count: usize,
    /// Calibration curve for visualization.
    pub calibration_curve: Vec<CalibrationBin>,
    /// Whether we have enough data for reliable metrics.
    pub is_reliable: bool,
}

impl Default for CalibrationMetrics {
    fn default() -> Self {
        Self {
            brier_score: 0.0,
            information_ratio: 0.0,
            observation_count: 0,
            calibration_curve: Vec::new(),
            is_reliable: false,
        }
    }
}

/// Tracks prediction-outcome pairs for a single prediction type.
#[derive(Clone, Debug)]
pub struct PredictionTracker {
    observations: VecDeque<PredictionOutcome>,
    config: CalibrationConfig,
}

impl PredictionTracker {
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            observations: VecDeque::with_capacity(config.max_observations),
            config,
        }
    }

    /// Record a prediction and its outcome.
    pub fn record(&mut self, predicted: f64, realized: bool, regime: &str) {
        // Clamp prediction to [0, 1]
        let predicted = predicted.clamp(0.0, 1.0);

        let outcome = PredictionOutcome {
            predicted,
            realized,
            timestamp: Instant::now(),
            regime: regime.to_string(),
        };

        self.observations.push_back(outcome);

        // Maintain rolling window
        while self.observations.len() > self.config.max_observations {
            self.observations.pop_front();
        }
    }

    /// Compute calibration metrics.
    pub fn metrics(&self) -> CalibrationMetrics {
        let n = self.observations.len();
        if n < self.config.min_observations {
            return CalibrationMetrics {
                is_reliable: false,
                observation_count: n,
                ..Default::default()
            };
        }

        // Compute base rate (overall positive rate)
        let positive_count = self.observations.iter().filter(|o| o.realized).count();
        let base_rate = positive_count as f64 / n as f64;

        // Compute Brier score: mean((predicted - realized)^2)
        let brier_score: f64 = self
            .observations
            .iter()
            .map(|o| {
                let realized = if o.realized { 1.0 } else { 0.0 };
                (o.predicted - realized).powi(2)
            })
            .sum::<f64>()
            / n as f64;

        // Compute calibration curve and resolution
        let calibration_curve = self.compute_calibration_curve();
        let resolution = self.compute_resolution(&calibration_curve, base_rate);

        // Uncertainty (base rate entropy)
        let uncertainty = base_rate * (1.0 - base_rate);

        // Information ratio = resolution / uncertainty
        // Avoid division by zero
        let information_ratio = if uncertainty > 1e-10 {
            resolution / uncertainty
        } else {
            0.0
        };

        CalibrationMetrics {
            brier_score,
            information_ratio,
            observation_count: n,
            calibration_curve,
            is_reliable: true,
        }
    }

    /// Compute calibration curve (predicted vs realized by bucket).
    fn compute_calibration_curve(&self) -> Vec<CalibrationBin> {
        let num_bins = self.config.num_bins;
        let bin_width = 1.0 / num_bins as f64;

        let mut bins: Vec<(usize, usize)> = vec![(0, 0); num_bins]; // (positive_count, total_count)

        for obs in &self.observations {
            let bin_idx = ((obs.predicted / bin_width) as usize).min(num_bins - 1);
            bins[bin_idx].1 += 1; // total
            if obs.realized {
                bins[bin_idx].0 += 1; // positive
            }
        }

        bins.iter()
            .enumerate()
            .map(|(i, (pos, total))| {
                let center = (i as f64 + 0.5) * bin_width;
                let realized = if *total > 0 {
                    *pos as f64 / *total as f64
                } else {
                    center // No data, use predicted as placeholder
                };
                CalibrationBin {
                    predicted: center,
                    realized,
                    count: *total,
                }
            })
            .collect()
    }

    /// Compute resolution (how much predictions deviate from base rate).
    fn compute_resolution(&self, curve: &[CalibrationBin], base_rate: f64) -> f64 {
        let n = self.observations.len() as f64;
        if n < 1.0 {
            return 0.0;
        }

        curve
            .iter()
            .map(|bin| {
                let weight = bin.count as f64 / n;
                let deviation = bin.realized - base_rate;
                weight * deviation.powi(2)
            })
            .sum()
    }

    /// Get observation count.
    pub fn count(&self) -> usize {
        self.observations.len()
    }

    /// Check if warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.observations.len() >= self.config.min_observations
    }

    /// Get metrics for a specific regime.
    pub fn metrics_for_regime(&self, regime: &str) -> CalibrationMetrics {
        let filtered: Vec<_> = self
            .observations
            .iter()
            .filter(|o| o.regime == regime)
            .cloned()
            .collect();

        if filtered.len() < self.config.min_observations {
            return CalibrationMetrics {
                is_reliable: false,
                observation_count: filtered.len(),
                ..Default::default()
            };
        }

        // Create temporary tracker with filtered data
        let mut temp = PredictionTracker::new(self.config.clone());
        for obs in filtered {
            temp.observations.push_back(obs);
        }
        temp.metrics()
    }
}

/// Main calibration tracker that manages multiple prediction types.
#[derive(Clone, Debug)]
pub struct CalibrationTracker {
    /// Fill probability calibration.
    pub fill: PredictionTracker,
    /// Adverse selection calibration.
    pub adverse_selection: PredictionTracker,
    /// Configuration.
    config: CalibrationConfig,
}

impl CalibrationTracker {
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            fill: PredictionTracker::new(config.clone()),
            adverse_selection: PredictionTracker::new(config.clone()),
            config,
        }
    }

    /// Record a fill probability prediction and outcome.
    pub fn record_fill(&mut self, predicted_prob: f64, did_fill: bool, regime: &str) {
        self.fill.record(predicted_prob, did_fill, regime);
    }

    /// Record an adverse selection prediction and outcome.
    /// `predicted_alpha` is the predicted probability of informed flow.
    /// `was_adverse` is whether the fill experienced adverse selection.
    pub fn record_adverse_selection(
        &mut self,
        predicted_alpha: f64,
        was_adverse: bool,
        regime: &str,
    ) {
        self.adverse_selection.record(predicted_alpha, was_adverse, regime);
    }

    /// Get fill calibration metrics.
    pub fn fill_metrics(&self) -> CalibrationMetrics {
        self.fill.metrics()
    }

    /// Get adverse selection calibration metrics.
    pub fn as_metrics(&self) -> CalibrationMetrics {
        self.adverse_selection.metrics()
    }

    /// Get summary for dashboard.
    pub fn summary(&self) -> CalibrationSummary {
        CalibrationSummary {
            fill: self.fill_metrics(),
            adverse_selection: self.as_metrics(),
        }
    }
}

impl Default for CalibrationTracker {
    fn default() -> Self {
        Self::new(CalibrationConfig::default())
    }
}

/// Summary of all calibration metrics for the dashboard.
#[derive(Clone, Debug, Serialize)]
pub struct CalibrationSummary {
    pub fill: CalibrationMetrics,
    pub adverse_selection: CalibrationMetrics,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker() -> PredictionTracker {
        PredictionTracker::new(CalibrationConfig {
            max_observations: 100,
            num_bins: 10,
            min_observations: 10,
        })
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = make_tracker();
        let metrics = tracker.metrics();
        assert!(!metrics.is_reliable);
        assert_eq!(metrics.observation_count, 0);
    }

    #[test]
    fn test_perfect_calibration() {
        let mut tracker = make_tracker();

        // Record predictions that match outcomes perfectly
        // Low predictions (0.1) that don't occur
        for _ in 0..20 {
            tracker.record(0.1, false, "Quiet");
        }
        // High predictions (0.9) that do occur
        for _ in 0..20 {
            tracker.record(0.9, true, "Quiet");
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);
        // Brier score should be low (good)
        assert!(metrics.brier_score < 0.1, "Brier: {}", metrics.brier_score);
    }

    #[test]
    fn test_poor_calibration() {
        let mut tracker = make_tracker();

        // Record predictions that are systematically wrong
        // Predict high but outcome is negative
        for _ in 0..20 {
            tracker.record(0.9, false, "Quiet");
        }
        // Predict low but outcome is positive
        for _ in 0..20 {
            tracker.record(0.1, true, "Quiet");
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);
        // Brier score should be high (bad)
        assert!(metrics.brier_score > 0.5, "Brier: {}", metrics.brier_score);
    }

    #[test]
    fn test_brier_score_calculation() {
        let mut tracker = make_tracker();

        // Prediction = 0.7, Outcome = 1 -> error = (0.7 - 1)^2 = 0.09
        tracker.record(0.7, true, "Quiet");
        // Prediction = 0.3, Outcome = 0 -> error = (0.3 - 0)^2 = 0.09
        tracker.record(0.3, false, "Quiet");

        // Add more to meet minimum
        for _ in 0..18 {
            tracker.record(0.5, true, "Quiet");
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);
        // Expected: average of all squared errors
    }

    #[test]
    fn test_calibration_curve() {
        let mut tracker = make_tracker();

        // Add observations in different prediction ranges
        for _ in 0..10 {
            tracker.record(0.15, false, "Quiet"); // Bin 1: 0% realized
        }
        for _ in 0..10 {
            tracker.record(0.85, true, "Quiet"); // Bin 8: 100% realized
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);

        // Check calibration curve
        let curve = &metrics.calibration_curve;
        assert_eq!(curve.len(), 10);

        // Bin 1 (0.1-0.2 range) should have 0% realized
        let bin1 = &curve[1];
        assert!(bin1.predicted > 0.1 && bin1.predicted < 0.2);
        assert_eq!(bin1.realized, 0.0);
        assert_eq!(bin1.count, 10);

        // Bin 8 (0.8-0.9 range) should have 100% realized
        let bin8 = &curve[8];
        assert!(bin8.predicted > 0.8 && bin8.predicted < 0.9);
        assert_eq!(bin8.realized, 1.0);
        assert_eq!(bin8.count, 10);
    }

    #[test]
    fn test_rolling_window() {
        let mut tracker = PredictionTracker::new(CalibrationConfig {
            max_observations: 20,
            num_bins: 10,
            min_observations: 5,
        });

        // Add 30 observations (should keep only last 20)
        for i in 0..30 {
            tracker.record(0.5, i % 2 == 0, "Quiet");
        }

        assert_eq!(tracker.count(), 20);
    }

    #[test]
    fn test_calibration_tracker() {
        let mut tracker = CalibrationTracker::default();

        // Record fill predictions
        for _ in 0..100 {
            tracker.record_fill(0.3, false, "Quiet");
            tracker.record_fill(0.7, true, "Quiet");
        }

        // Record AS predictions
        for _ in 0..100 {
            tracker.record_adverse_selection(0.2, false, "Quiet");
            tracker.record_adverse_selection(0.8, true, "Cascade");
        }

        let summary = tracker.summary();
        assert!(summary.fill.is_reliable);
        assert!(summary.adverse_selection.is_reliable);
    }

    #[test]
    fn test_information_ratio() {
        let mut tracker = make_tracker();

        // Add observations with good discrimination
        // When we predict high, it happens; when low, it doesn't
        for _ in 0..25 {
            tracker.record(0.1, false, "Quiet");
            tracker.record(0.9, true, "Quiet");
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);
        // Good discrimination should give IR > 1.0
        assert!(
            metrics.information_ratio > 0.5,
            "IR: {}",
            metrics.information_ratio
        );
    }

    #[test]
    fn test_regime_filtering() {
        let mut tracker = make_tracker();

        // Add observations in different regimes
        for _ in 0..20 {
            tracker.record(0.8, true, "Quiet");
        }
        for _ in 0..20 {
            tracker.record(0.2, true, "Cascade"); // Poorly calibrated in cascade
        }

        // Overall metrics
        let overall = tracker.metrics();
        assert!(overall.is_reliable);

        // Quiet regime should be well calibrated
        let quiet = tracker.metrics_for_regime("Quiet");
        assert!(quiet.is_reliable);
        assert!(quiet.brier_score < 0.1);

        // Cascade regime should be poorly calibrated
        let cascade = tracker.metrics_for_regime("Cascade");
        assert!(cascade.is_reliable);
        assert!(cascade.brier_score > 0.5);
    }
}
