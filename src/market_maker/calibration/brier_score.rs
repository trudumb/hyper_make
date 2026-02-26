//! Brier score tracking for probability calibration.
//!
//! The Brier score is a proper scoring rule that measures the accuracy of
//! probabilistic predictions. It is defined as:
//!
//! ```text
//! Brier = mean((predicted_prob - outcome)^2)
//! ```
//!
//! where `outcome` is 1 for true events and 0 for false events.
//!
//! ## Interpretation
//!
//! - **0.0**: Perfect predictions
//! - **0.25**: Maximum score for uninformed predictions (predicting 0.5 always)
//! - **1.0**: Worst possible predictions (always wrong with certainty)
//!
//! A model that predicts the base rate achieves Brier = base_rate * (1 - base_rate).
//! Any model should beat this baseline to be useful.

use std::collections::VecDeque;

/// Rolling Brier score tracker with configurable window size.
///
/// Maintains a sliding window of recent predictions for computing
/// the Brier score. Useful for monitoring model calibration over time
/// and detecting calibration drift.
#[derive(Debug, Clone)]
pub struct BrierScoreTracker {
    /// Maximum number of samples in the window
    window_size: usize,
    /// Individual squared errors for each prediction
    scores: VecDeque<f64>,
    /// Running sum for efficient mean calculation
    running_sum: f64,
}

impl BrierScoreTracker {
    /// Create a new Brier score tracker with the given window size.
    ///
    /// # Arguments
    /// * `window_size` - Maximum number of predictions to track
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            scores: VecDeque::with_capacity(window_size.min(10000)),
            running_sum: 0.0,
        }
    }

    /// Update the tracker with a new prediction and outcome.
    ///
    /// # Arguments
    /// * `predicted` - Predicted probability [0, 1]
    /// * `outcome` - Actual outcome (true/false)
    pub fn update(&mut self, predicted: f64, outcome: bool) {
        let predicted = predicted.clamp(0.0, 1.0);
        let outcome_val = if outcome { 1.0 } else { 0.0 };
        let score = (predicted - outcome_val).powi(2);

        // Evict oldest if at capacity
        if self.scores.len() >= self.window_size {
            if let Some(old) = self.scores.pop_front() {
                self.running_sum -= old;
            }
        }

        self.scores.push_back(score);
        self.running_sum += score;
    }

    /// Get the current Brier score.
    ///
    /// Returns 0.0 if no samples have been recorded.
    pub fn score(&self) -> f64 {
        if self.scores.is_empty() {
            0.0
        } else {
            self.running_sum / self.scores.len() as f64
        }
    }

    /// Get the number of samples in the current window.
    pub fn n_samples(&self) -> usize {
        self.scores.len()
    }

    /// Check if the tracker has enough samples for reliable scoring.
    pub fn is_reliable(&self, min_samples: usize) -> bool {
        self.scores.len() >= min_samples
    }

    /// Get the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Check if the window is full.
    pub fn is_window_full(&self) -> bool {
        self.scores.len() >= self.window_size
    }

    /// Clear all samples and reset the score.
    pub fn clear(&mut self) {
        self.scores.clear();
        self.running_sum = 0.0;
    }

    /// Get the theoretical baseline Brier score for a given base rate.
    ///
    /// This is the score achieved by always predicting the base rate.
    /// A useful model should beat this baseline.
    ///
    /// # Arguments
    /// * `base_rate` - The base rate of positive outcomes [0, 1]
    pub fn baseline_score(base_rate: f64) -> f64 {
        let base_rate = base_rate.clamp(0.0, 1.0);
        base_rate * (1.0 - base_rate)
    }

    /// Calculate the Brier skill score relative to a baseline.
    ///
    /// Returns None if no samples or if baseline is 0.
    ///
    /// # Formula
    /// ```text
    /// BSS = 1 - (Brier / Baseline)
    /// ```
    ///
    /// # Interpretation
    /// - **> 0**: Model is better than baseline
    /// - **= 0**: Model equals baseline
    /// - **< 0**: Model is worse than baseline
    /// - **= 1**: Perfect model
    pub fn skill_score(&self, base_rate: f64) -> Option<f64> {
        if self.scores.is_empty() {
            return None;
        }

        let baseline = Self::baseline_score(base_rate);
        if baseline < 1e-10 {
            return None; // Avoid division by zero
        }

        Some(1.0 - self.score() / baseline)
    }

    /// Get the most recent score (last prediction).
    pub fn last_score(&self) -> Option<f64> {
        self.scores.back().copied()
    }

    /// Calculate the variance of scores in the window.
    ///
    /// Returns None if fewer than 2 samples.
    pub fn score_variance(&self) -> Option<f64> {
        if self.scores.len() < 2 {
            return None;
        }

        let mean = self.score();
        let variance = self.scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / (self.scores.len() - 1) as f64;

        Some(variance)
    }

    /// Get the standard deviation of scores.
    pub fn score_std(&self) -> Option<f64> {
        self.score_variance().map(|v| v.sqrt())
    }
}

// BrierScoreTracker is Send + Sync because it only contains owned data
unsafe impl Send for BrierScoreTracker {}
unsafe impl Sync for BrierScoreTracker {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tracker = BrierScoreTracker::new(100);
        assert_eq!(tracker.window_size(), 100);
        assert_eq!(tracker.n_samples(), 0);
        assert_eq!(tracker.score(), 0.0);
    }

    #[test]
    fn test_new_min_window() {
        // Window size should be at least 1
        let tracker = BrierScoreTracker::new(0);
        assert_eq!(tracker.window_size(), 1);
    }

    #[test]
    fn test_perfect_predictions() {
        let mut tracker = BrierScoreTracker::new(100);

        // Perfect prediction: predicted 1.0, outcome true
        tracker.update(1.0, true);
        assert_eq!(tracker.score(), 0.0);

        // Perfect prediction: predicted 0.0, outcome false
        tracker.update(0.0, false);
        assert_eq!(tracker.score(), 0.0);
    }

    #[test]
    fn test_worst_predictions() {
        let mut tracker = BrierScoreTracker::new(100);

        // Worst prediction: predicted 1.0, outcome false
        tracker.update(1.0, false);
        assert_eq!(tracker.score(), 1.0);

        // Worst prediction: predicted 0.0, outcome true
        tracker.update(0.0, true);
        assert_eq!(tracker.score(), 1.0); // Average of two 1.0 scores
    }

    #[test]
    fn test_intermediate_predictions() {
        let mut tracker = BrierScoreTracker::new(100);

        // Predicted 0.7, outcome true -> (0.7 - 1.0)^2 = 0.09
        tracker.update(0.7, true);
        assert!((tracker.score() - 0.09).abs() < 1e-10);

        // Predicted 0.3, outcome false -> (0.3 - 0.0)^2 = 0.09
        tracker.update(0.3, false);
        assert!((tracker.score() - 0.09).abs() < 1e-10); // Average
    }

    #[test]
    fn test_clamping() {
        let mut tracker = BrierScoreTracker::new(100);

        // Predictions outside [0, 1] should be clamped
        tracker.update(1.5, true); // Clamped to 1.0
        assert_eq!(tracker.score(), 0.0);

        tracker.clear();
        tracker.update(-0.5, false); // Clamped to 0.0
        assert_eq!(tracker.score(), 0.0);
    }

    #[test]
    fn test_window_eviction() {
        let mut tracker = BrierScoreTracker::new(3);

        // Add 3 perfect predictions
        tracker.update(1.0, true);
        tracker.update(1.0, true);
        tracker.update(1.0, true);
        assert_eq!(tracker.n_samples(), 3);
        assert_eq!(tracker.score(), 0.0);

        // Add a bad prediction - evicts first perfect one
        tracker.update(0.0, true); // Score = 1.0
        assert_eq!(tracker.n_samples(), 3);

        // Score should now be (0 + 0 + 1) / 3 = 0.333...
        let expected = 1.0 / 3.0;
        assert!((tracker.score() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_is_reliable() {
        let mut tracker = BrierScoreTracker::new(100);

        assert!(!tracker.is_reliable(10));

        for _ in 0..10 {
            tracker.update(0.5, true);
        }

        assert!(tracker.is_reliable(10));
        assert!(!tracker.is_reliable(11));
    }

    #[test]
    fn test_is_window_full() {
        let mut tracker = BrierScoreTracker::new(3);

        assert!(!tracker.is_window_full());

        tracker.update(0.5, true);
        tracker.update(0.5, true);
        assert!(!tracker.is_window_full());

        tracker.update(0.5, true);
        assert!(tracker.is_window_full());

        tracker.update(0.5, true); // Evicts one
        assert!(tracker.is_window_full());
    }

    #[test]
    fn test_clear() {
        let mut tracker = BrierScoreTracker::new(100);

        tracker.update(0.5, true);
        tracker.update(0.5, false);
        assert_eq!(tracker.n_samples(), 2);

        tracker.clear();
        assert_eq!(tracker.n_samples(), 0);
        assert_eq!(tracker.score(), 0.0);
    }

    #[test]
    fn test_baseline_score() {
        // Base rate 0.5 -> baseline = 0.25
        assert!((BrierScoreTracker::baseline_score(0.5) - 0.25).abs() < 1e-10);

        // Base rate 0.0 -> baseline = 0.0
        assert_eq!(BrierScoreTracker::baseline_score(0.0), 0.0);

        // Base rate 1.0 -> baseline = 0.0
        assert_eq!(BrierScoreTracker::baseline_score(1.0), 0.0);

        // Base rate 0.2 -> baseline = 0.16
        assert!((BrierScoreTracker::baseline_score(0.2) - 0.16).abs() < 1e-10);
    }

    #[test]
    fn test_skill_score_perfect() {
        let mut tracker = BrierScoreTracker::new(100);

        // Perfect predictions -> skill score = 1.0
        tracker.update(1.0, true);
        tracker.update(0.0, false);

        let skill = tracker.skill_score(0.5).unwrap();
        assert!((skill - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_skill_score_baseline() {
        let mut tracker = BrierScoreTracker::new(100);

        // Always predict 0.5 with 50% outcomes -> Brier = 0.25 = baseline
        for _ in 0..50 {
            tracker.update(0.5, true);
        }
        for _ in 0..50 {
            tracker.update(0.5, false);
        }

        let skill = tracker.skill_score(0.5).unwrap();
        assert!(skill.abs() < 1e-10); // Should be very close to 0
    }

    #[test]
    fn test_skill_score_edge_cases() {
        let tracker = BrierScoreTracker::new(100);

        // Empty tracker
        assert!(tracker.skill_score(0.5).is_none());

        // Base rate 0 (undefined baseline)
        let mut tracker2 = BrierScoreTracker::new(100);
        tracker2.update(0.5, true);
        assert!(tracker2.skill_score(0.0).is_none());
    }

    #[test]
    fn test_last_score() {
        let mut tracker = BrierScoreTracker::new(100);

        assert!(tracker.last_score().is_none());

        tracker.update(0.8, true); // (0.8 - 1.0)^2 = 0.04
        assert!((tracker.last_score().unwrap() - 0.04).abs() < 1e-10);

        tracker.update(0.2, false); // (0.2 - 0.0)^2 = 0.04
        assert!((tracker.last_score().unwrap() - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_score_variance() {
        let mut tracker = BrierScoreTracker::new(100);

        // Not enough samples
        assert!(tracker.score_variance().is_none());
        tracker.update(0.5, true);
        assert!(tracker.score_variance().is_none());

        // All same scores -> variance = 0
        tracker.update(0.5, true);
        let var = tracker.score_variance().unwrap();
        assert!(var < 1e-10);

        // Different scores -> positive variance
        tracker.update(1.0, false); // Score = 1.0
        let var = tracker.score_variance().unwrap();
        assert!(var > 0.0);
    }

    #[test]
    fn test_score_std() {
        let mut tracker = BrierScoreTracker::new(100);

        tracker.update(0.5, true);
        tracker.update(0.5, false);

        let std = tracker.score_std();
        assert!(std.is_some());

        // std should be sqrt(variance)
        if let (Some(std_val), Some(var_val)) = (std, tracker.score_variance()) {
            assert!((std_val - var_val.sqrt()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_running_sum_accuracy() {
        let mut tracker = BrierScoreTracker::new(5);

        // Add some predictions
        for i in 0..10 {
            let prob = (i as f64 + 1.0) / 10.0;
            tracker.update(prob, i % 2 == 0);
        }

        // Verify running sum matches manual calculation
        let manual_sum: f64 = tracker.scores.iter().sum();
        assert!((tracker.running_sum - manual_sum).abs() < 1e-10);
    }
}
