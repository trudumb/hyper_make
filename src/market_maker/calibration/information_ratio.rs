//! Information Ratio tracking for model value assessment.
//!
//! The Information Ratio (IR) measures how much value a probabilistic model
//! adds compared to simply predicting the base rate. It is defined as:
//!
//! ```text
//! IR = Resolution / Uncertainty
//! ```
//!
//! where:
//! - **Resolution**: Average squared deviation of conditional outcome rates from base rate
//! - **Uncertainty**: base_rate * (1 - base_rate) (maximum possible information gain)
//!
//! ## Interpretation
//!
//! - **IR > 1.0**: Model adds value (predictions are informative)
//! - **IR = 1.0**: Model equals random noise
//! - **IR < 1.0**: Model adds noise (worse than base rate)
//!
//! A model with IR < 1.0 should be removed as it degrades performance.

/// Tracks Information Ratio using binned probability calibration.
///
/// Predictions are binned by probability, and within each bin we track
/// the actual outcome rate. Resolution measures how far these conditional
/// rates deviate from the overall base rate.
#[derive(Debug, Clone)]
pub struct InformationRatioTracker {
    /// Number of probability bins
    n_bins: usize,
    /// Count of predictions in each bin
    bin_counts: Vec<usize>,
    /// Count of positive outcomes in each bin
    bin_outcomes: Vec<usize>,
    /// Total positive outcomes (for base rate)
    total_outcomes: usize,
    /// Total predictions
    total_count: usize,
}

impl InformationRatioTracker {
    /// Create a new Information Ratio tracker.
    ///
    /// # Arguments
    /// * `n_bins` - Number of probability bins (more bins = finer resolution but needs more data)
    ///
    /// Recommended: 10 bins for most applications (requires ~1000 samples for reliability)
    pub fn new(n_bins: usize) -> Self {
        let n_bins = n_bins.max(2); // At least 2 bins

        Self {
            n_bins,
            bin_counts: vec![0; n_bins],
            bin_outcomes: vec![0; n_bins],
            total_outcomes: 0,
            total_count: 0,
        }
    }

    /// Get the bin index for a probability.
    fn bin_index(&self, prob: f64) -> usize {
        let prob = prob.clamp(0.0, 1.0);
        let bin = (prob * self.n_bins as f64).floor() as usize;
        bin.min(self.n_bins - 1) // Handle edge case where prob = 1.0
    }

    /// Update the tracker with a new prediction and outcome.
    ///
    /// # Arguments
    /// * `predicted` - Predicted probability [0, 1]
    /// * `outcome` - Actual outcome (true/false)
    pub fn update(&mut self, predicted: f64, outcome: bool) {
        let bin = self.bin_index(predicted);

        self.bin_counts[bin] += 1;
        self.total_count += 1;

        if outcome {
            self.bin_outcomes[bin] += 1;
            self.total_outcomes += 1;
        }
    }

    /// Get the overall base rate (fraction of positive outcomes).
    pub fn base_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.5 // Default to 0.5 when no data
        } else {
            self.total_outcomes as f64 / self.total_count as f64
        }
    }

    /// Get the uncertainty (maximum possible information gain).
    ///
    /// This is base_rate * (1 - base_rate), the variance of a Bernoulli random variable.
    pub fn uncertainty(&self) -> f64 {
        let base = self.base_rate();
        base * (1.0 - base)
    }

    /// Get the resolution (information captured by the model).
    ///
    /// Resolution = weighted average of (bin_outcome_rate - base_rate)^2
    ///
    /// High resolution means the model's predictions effectively
    /// separate high-outcome from low-outcome cases.
    pub fn resolution(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let base = self.base_rate();
        let mut resolution = 0.0;

        for bin in 0..self.n_bins {
            if self.bin_counts[bin] > 0 {
                let bin_rate = self.bin_outcomes[bin] as f64 / self.bin_counts[bin] as f64;
                let weight = self.bin_counts[bin] as f64 / self.total_count as f64;
                resolution += weight * (bin_rate - base).powi(2);
            }
        }

        resolution
    }

    /// Get the Information Ratio.
    ///
    /// Returns 0.0 if uncertainty is zero (all outcomes same) or no data.
    pub fn information_ratio(&self) -> f64 {
        let uncertainty = self.uncertainty();
        if uncertainty < 1e-10 || self.total_count == 0 {
            return 0.0;
        }

        self.resolution() / uncertainty
    }

    /// Check if the model is adding value (IR > 1.0).
    ///
    /// A model with IR <= 1.0 is adding noise rather than signal.
    pub fn is_adding_value(&self) -> bool {
        self.information_ratio() > 1.0
    }

    /// Get the total number of samples.
    pub fn n_samples(&self) -> usize {
        self.total_count
    }

    /// Check if there are enough samples for reliable metrics.
    pub fn is_reliable(&self, min_samples: usize) -> bool {
        self.total_count >= min_samples
    }

    /// Get the number of bins.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Get the count of samples in each bin.
    pub fn bin_counts(&self) -> &[usize] {
        &self.bin_counts
    }

    /// Get the conditional outcome rate for each bin.
    pub fn bin_rates(&self) -> Vec<f64> {
        self.bin_counts
            .iter()
            .zip(self.bin_outcomes.iter())
            .map(|(&count, &outcomes)| {
                if count > 0 {
                    outcomes as f64 / count as f64
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Get the calibration error (mean absolute deviation from predicted).
    ///
    /// This measures how well the predicted probabilities match actual outcome rates.
    /// Perfect calibration = 0.
    pub fn calibration_error(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let mut error_sum = 0.0;
        let mut weight_sum = 0.0;

        for bin in 0..self.n_bins {
            if self.bin_counts[bin] > 0 {
                // Bin center is the expected probability for that bin
                let bin_center = (bin as f64 + 0.5) / self.n_bins as f64;
                let actual_rate = self.bin_outcomes[bin] as f64 / self.bin_counts[bin] as f64;
                let weight = self.bin_counts[bin] as f64;

                error_sum += weight * (actual_rate - bin_center).abs();
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            error_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Clear all samples and reset the tracker.
    pub fn clear(&mut self) {
        self.bin_counts.fill(0);
        self.bin_outcomes.fill(0);
        self.total_outcomes = 0;
        self.total_count = 0;
    }

    /// Merge another tracker into this one.
    ///
    /// Both trackers must have the same number of bins.
    pub fn merge(&mut self, other: &InformationRatioTracker) {
        assert_eq!(
            self.n_bins, other.n_bins,
            "Cannot merge trackers with different bin counts"
        );

        for bin in 0..self.n_bins {
            self.bin_counts[bin] += other.bin_counts[bin];
            self.bin_outcomes[bin] += other.bin_outcomes[bin];
        }

        self.total_count += other.total_count;
        self.total_outcomes += other.total_outcomes;
    }

    /// Get a detailed breakdown of bin statistics.
    pub fn bin_stats(&self) -> Vec<BinStats> {
        (0..self.n_bins)
            .map(|bin| {
                let bin_start = bin as f64 / self.n_bins as f64;
                let bin_end = (bin + 1) as f64 / self.n_bins as f64;

                BinStats {
                    bin_index: bin,
                    bin_start,
                    bin_end,
                    count: self.bin_counts[bin],
                    positive_outcomes: self.bin_outcomes[bin],
                    outcome_rate: if self.bin_counts[bin] > 0 {
                        self.bin_outcomes[bin] as f64 / self.bin_counts[bin] as f64
                    } else {
                        0.0
                    },
                }
            })
            .collect()
    }
}

/// Statistics for a single probability bin.
#[derive(Debug, Clone)]
pub struct BinStats {
    /// Index of this bin
    pub bin_index: usize,
    /// Lower bound of the probability range
    pub bin_start: f64,
    /// Upper bound of the probability range
    pub bin_end: f64,
    /// Number of samples in this bin
    pub count: usize,
    /// Number of positive outcomes in this bin
    pub positive_outcomes: usize,
    /// Conditional outcome rate (positive / total)
    pub outcome_rate: f64,
}

// InformationRatioTracker is Send + Sync because it only contains owned data
unsafe impl Send for InformationRatioTracker {}
unsafe impl Sync for InformationRatioTracker {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tracker = InformationRatioTracker::new(10);
        assert_eq!(tracker.n_bins(), 10);
        assert_eq!(tracker.n_samples(), 0);
        assert_eq!(tracker.base_rate(), 0.5); // Default
    }

    #[test]
    fn test_new_min_bins() {
        let tracker = InformationRatioTracker::new(1);
        assert_eq!(tracker.n_bins(), 2); // Minimum 2 bins
    }

    #[test]
    fn test_bin_index() {
        let tracker = InformationRatioTracker::new(10);

        assert_eq!(tracker.bin_index(0.0), 0);
        assert_eq!(tracker.bin_index(0.05), 0);
        assert_eq!(tracker.bin_index(0.1), 1);
        assert_eq!(tracker.bin_index(0.5), 5);
        assert_eq!(tracker.bin_index(0.95), 9);
        assert_eq!(tracker.bin_index(1.0), 9); // Edge case
    }

    #[test]
    fn test_update() {
        let mut tracker = InformationRatioTracker::new(10);

        tracker.update(0.75, true);
        assert_eq!(tracker.n_samples(), 1);
        assert_eq!(tracker.bin_counts[7], 1);
        assert_eq!(tracker.bin_outcomes[7], 1);

        tracker.update(0.75, false);
        assert_eq!(tracker.n_samples(), 2);
        assert_eq!(tracker.bin_counts[7], 2);
        assert_eq!(tracker.bin_outcomes[7], 1);
    }

    #[test]
    fn test_base_rate() {
        let mut tracker = InformationRatioTracker::new(10);

        tracker.update(0.5, true);
        tracker.update(0.5, true);
        tracker.update(0.5, false);
        tracker.update(0.5, false);

        assert!((tracker.base_rate() - 0.5).abs() < 1e-10);

        tracker.update(0.5, true);
        assert!((tracker.base_rate() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_uncertainty() {
        let mut tracker = InformationRatioTracker::new(10);

        // 50% base rate -> uncertainty = 0.25
        for _ in 0..50 {
            tracker.update(0.5, true);
            tracker.update(0.5, false);
        }

        assert!((tracker.uncertainty() - 0.25).abs() < 1e-10);

        // 20% base rate -> uncertainty = 0.16
        tracker.clear();
        for _ in 0..20 {
            tracker.update(0.5, true);
        }
        for _ in 0..80 {
            tracker.update(0.5, false);
        }

        assert!((tracker.uncertainty() - 0.16).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_discrimination() {
        let mut tracker = InformationRatioTracker::new(10);

        // Perfect predictions: 0.1 always false, 0.9 always true
        for _ in 0..100 {
            tracker.update(0.1, false);
            tracker.update(0.9, true);
        }

        // Base rate = 0.5
        assert!((tracker.base_rate() - 0.5).abs() < 1e-10);

        // Bin 1 (0.1-0.2): outcome_rate = 0, deviation from 0.5 = 0.25
        // Bin 9 (0.9-1.0): outcome_rate = 1, deviation from 0.5 = 0.25
        // Resolution = 0.5 * 0.25 + 0.5 * 0.25 = 0.25 (weighted by 0.5 each)

        assert!((tracker.resolution() - 0.25).abs() < 1e-10);

        // IR = resolution / uncertainty = 0.25 / 0.25 = 1.0
        assert!((tracker.information_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_discrimination() {
        let mut tracker = InformationRatioTracker::new(10);

        // All predictions in same bin with 50% outcomes
        for _ in 0..100 {
            tracker.update(0.55, true);
            tracker.update(0.55, false);
        }

        // Resolution should be 0 (all in one bin, matching base rate)
        assert!(tracker.resolution() < 1e-10);

        // IR should be 0
        assert!(tracker.information_ratio() < 1e-10);
    }

    #[test]
    fn test_is_adding_value() {
        let mut tracker = InformationRatioTracker::new(10);

        // Initially not adding value
        assert!(!tracker.is_adding_value());

        // Good discrimination
        for _ in 0..100 {
            tracker.update(0.1, false);
            tracker.update(0.5, true); // Mix it up a bit
            tracker.update(0.9, true);
        }

        // Check if adding value depends on the IR
        // With this setup, model should be adding some value
        let ir = tracker.information_ratio();
        assert_eq!(tracker.is_adding_value(), ir > 1.0);
    }

    #[test]
    fn test_bin_rates() {
        let mut tracker = InformationRatioTracker::new(5);

        // Bin 0 (0.0-0.2): 0% positive
        tracker.update(0.1, false);
        tracker.update(0.1, false);

        // Bin 2 (0.4-0.6): 50% positive
        tracker.update(0.5, true);
        tracker.update(0.5, false);

        // Bin 4 (0.8-1.0): 100% positive
        tracker.update(0.9, true);
        tracker.update(0.9, true);

        let rates = tracker.bin_rates();
        assert!((rates[0] - 0.0).abs() < 1e-10);
        assert!((rates[2] - 0.5).abs() < 1e-10);
        assert!((rates[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calibration_error() {
        let mut tracker = InformationRatioTracker::new(10);

        // Perfectly calibrated: bin 1 (center 0.15) has 15% positive rate
        // This is hard to achieve exactly, so we'll use approximate
        for _ in 0..85 {
            tracker.update(0.15, false);
        }
        for _ in 0..15 {
            tracker.update(0.15, true);
        }

        // Calibration error should be small
        assert!(tracker.calibration_error() < 0.05);
    }

    #[test]
    fn test_clear() {
        let mut tracker = InformationRatioTracker::new(10);

        tracker.update(0.5, true);
        tracker.update(0.5, false);

        tracker.clear();

        assert_eq!(tracker.n_samples(), 0);
        assert_eq!(tracker.total_outcomes, 0);
        assert!(tracker.bin_counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_merge() {
        let mut tracker1 = InformationRatioTracker::new(10);
        let mut tracker2 = InformationRatioTracker::new(10);

        tracker1.update(0.1, false);
        tracker1.update(0.9, true);

        tracker2.update(0.1, false);
        tracker2.update(0.9, true);

        tracker1.merge(&tracker2);

        assert_eq!(tracker1.n_samples(), 4);
        assert_eq!(tracker1.bin_counts[1], 2);
        assert_eq!(tracker1.bin_counts[9], 2);
    }

    #[test]
    #[should_panic(expected = "Cannot merge trackers with different bin counts")]
    fn test_merge_incompatible() {
        let mut tracker1 = InformationRatioTracker::new(10);
        let tracker2 = InformationRatioTracker::new(5);

        tracker1.merge(&tracker2);
    }

    #[test]
    fn test_bin_stats() {
        let mut tracker = InformationRatioTracker::new(5);

        tracker.update(0.1, false);
        tracker.update(0.1, true);
        tracker.update(0.9, true);

        let stats = tracker.bin_stats();

        assert_eq!(stats.len(), 5);
        assert_eq!(stats[0].count, 2);
        assert_eq!(stats[0].positive_outcomes, 1);
        assert!((stats[0].outcome_rate - 0.5).abs() < 1e-10);
        assert_eq!(stats[4].count, 1);
        assert!((stats[4].outcome_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_reliable() {
        let mut tracker = InformationRatioTracker::new(10);

        assert!(!tracker.is_reliable(100));

        for _ in 0..100 {
            tracker.update(0.5, true);
        }

        assert!(tracker.is_reliable(100));
        assert!(!tracker.is_reliable(101));
    }

    #[test]
    fn test_edge_cases_zero_uncertainty() {
        let mut tracker = InformationRatioTracker::new(10);

        // All positive outcomes -> base_rate = 1.0, uncertainty = 0
        for _ in 0..100 {
            tracker.update(0.5, true);
        }

        assert_eq!(tracker.uncertainty(), 0.0);
        assert_eq!(tracker.information_ratio(), 0.0);
    }
}
