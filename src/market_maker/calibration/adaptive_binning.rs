//! Adaptive Binning for Information Ratio Tracking
//!
//! Replaces fixed equal-width bins with quantile-based adaptive bins that
//! distribute samples more evenly, enabling IR calibration with fewer fills.
//!
//! # Problem Solved
//!
//! Fixed bins (10 equal-width from 0.0-1.0) leave most bins empty when:
//! - Flow imbalance clusters in narrow range (e.g., 0.3-0.5)
//! - Low liquidity means few samples overall
//!
//! Result: Single bin contains all samples → Resolution = 0 → IR = 0
//!
//! # Solution
//!
//! Quantile-based dynamic binning:
//! 1. Store recent (probability, outcome) samples
//! 2. Recompute bin edges at data quantiles (10%, 20%, ..., 90%)
//! 3. Each bin gets ~equal samples → non-zero resolution possible
//! 4. Include regularization (epsilon smoothing) for numerical stability
//!
//! # Usage
//!
//! ```ignore
//! let mut binner = AdaptiveBinner::new(10, 500);
//!
//! // Record predictions and outcomes
//! binner.update(0.75, true);  // Predicted 75%, outcome was positive
//! binner.update(0.30, false); // Predicted 30%, outcome was negative
//!
//! // Get IR once enough samples
//! if binner.has_sufficient_data() {
//!     let ir = binner.information_ratio();
//! }
//! ```

use std::collections::VecDeque;

/// Adaptive binner using quantile-based bin edges.
#[derive(Debug, Clone)]
pub struct AdaptiveBinner {
    /// Number of probability bins.
    n_bins: usize,
    /// Dynamic quantile-based bin edges.
    quantile_edges: Vec<f64>,
    /// Count of predictions in each bin.
    bin_counts: Vec<usize>,
    /// Count of positive outcomes in each bin.
    bin_outcomes: Vec<usize>,
    /// Rolling sample buffer: (probability, outcome).
    samples: VecDeque<(f64, bool)>,
    /// Maximum samples to retain.
    max_samples: usize,
    /// Recompute quantiles every N samples.
    recompute_interval: usize,
    /// Samples since last recompute.
    samples_since_recompute: usize,
    /// Minimum samples for reliable IR.
    min_samples: usize,
}

impl AdaptiveBinner {
    /// Create a new adaptive binner.
    ///
    /// # Arguments
    /// * `n_bins` - Number of probability bins (default: 10)
    /// * `max_samples` - Maximum samples to retain in rolling buffer (default: 500)
    pub fn new(n_bins: usize, max_samples: usize) -> Self {
        let n_bins = n_bins.max(3); // At least 3 bins
        Self {
            n_bins,
            quantile_edges: Vec::new(),
            bin_counts: vec![0; n_bins],
            bin_outcomes: vec![0; n_bins],
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            recompute_interval: 20, // Recompute every 20 samples
            samples_since_recompute: 0,
            min_samples: 30, // Need at least 30 samples
        }
    }

    /// Create with custom recompute interval.
    pub fn with_recompute_interval(mut self, interval: usize) -> Self {
        self.recompute_interval = interval.max(5);
        self
    }

    /// Create with custom minimum samples.
    pub fn with_min_samples(mut self, min: usize) -> Self {
        self.min_samples = min.max(10);
        self
    }

    /// Update the binner with a new prediction and outcome.
    ///
    /// # Arguments
    /// * `prob` - Predicted probability [0, 1]
    /// * `outcome` - Actual outcome (true/false)
    pub fn update(&mut self, prob: f64, outcome: bool) {
        let prob = prob.clamp(0.0, 1.0);

        // Store sample
        if self.samples.len() >= self.max_samples {
            // Remove oldest sample from bin counts
            if let Some((old_prob, old_outcome)) = self.samples.pop_front() {
                let old_bin = self.get_bin(old_prob);
                if self.bin_counts[old_bin] > 0 {
                    self.bin_counts[old_bin] -= 1;
                    if old_outcome && self.bin_outcomes[old_bin] > 0 {
                        self.bin_outcomes[old_bin] -= 1;
                    }
                }
            }
        }
        self.samples.push_back((prob, outcome));

        // Recompute quantiles periodically
        self.samples_since_recompute += 1;
        if self.samples_since_recompute >= self.recompute_interval {
            self.recompute_quantiles();
            self.rebuild_bins();
            self.samples_since_recompute = 0;
        } else {
            // Just add to current bin
            let bin = self.get_bin(prob);
            self.bin_counts[bin] += 1;
            if outcome {
                self.bin_outcomes[bin] += 1;
            }
        }
    }

    /// Recompute quantile bin edges from stored samples.
    fn recompute_quantiles(&mut self) {
        if self.samples.len() < self.n_bins {
            return; // Not enough samples for quantiles
        }

        let mut probs: Vec<f64> = self.samples.iter().map(|(p, _)| *p).collect();
        probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        self.quantile_edges.clear();
        for i in 1..self.n_bins {
            let idx = (i * probs.len()) / self.n_bins;
            let idx = idx.min(probs.len() - 1);
            self.quantile_edges.push(probs[idx]);
        }
    }

    /// Rebuild bin counts from stored samples after quantile recompute.
    fn rebuild_bins(&mut self) {
        // Reset bin counts
        self.bin_counts.fill(0);
        self.bin_outcomes.fill(0);

        // Recount all samples
        for (prob, outcome) in self.samples.iter() {
            let bin = self.get_bin(*prob);
            self.bin_counts[bin] += 1;
            if *outcome {
                self.bin_outcomes[bin] += 1;
            }
        }
    }

    /// Get the bin index for a probability.
    fn get_bin(&self, prob: f64) -> usize {
        if self.quantile_edges.is_empty() {
            // Fallback to fixed bins before enough data
            let bin = (prob * self.n_bins as f64).floor() as usize;
            return bin.min(self.n_bins - 1);
        }

        for (i, &edge) in self.quantile_edges.iter().enumerate() {
            if prob < edge {
                return i;
            }
        }
        self.n_bins - 1
    }

    /// Get the overall base rate (fraction of positive outcomes).
    pub fn base_rate(&self) -> f64 {
        let total_outcomes: usize = self.bin_outcomes.iter().sum();
        let total_count: usize = self.bin_counts.iter().sum();
        if total_count > 0 {
            total_outcomes as f64 / total_count as f64
        } else {
            0.5 // Default to 0.5 when no data
        }
    }

    /// Get the uncertainty (maximum possible information gain).
    pub fn uncertainty(&self) -> f64 {
        let base = self.base_rate();
        base * (1.0 - base)
    }

    /// Get the resolution with epsilon regularization.
    ///
    /// Resolution = weighted average of (bin_outcome_rate - base_rate)²
    pub fn resolution(&self) -> f64 {
        let total: usize = self.bin_counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let base = self.base_rate();
        let mut resolution = 0.0;

        for i in 0..self.n_bins {
            if self.bin_counts[i] > 0 {
                let bin_rate = self.bin_outcomes[i] as f64 / self.bin_counts[i] as f64;
                let weight = self.bin_counts[i] as f64 / total as f64;
                resolution += weight * (bin_rate - base).powi(2);
            }
        }

        resolution
    }

    /// Get the Information Ratio with regularization.
    ///
    /// Uses epsilon smoothing to handle edge cases:
    /// - epsilon decreases with sample size
    /// - prevents division by zero or near-zero uncertainty
    pub fn information_ratio(&self) -> f64 {
        let total: usize = self.bin_counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let uncertainty = self.uncertainty();
        if uncertainty < 1e-10 {
            return 0.0;
        }

        // Regularization: epsilon decreases with sample size
        let epsilon = 0.01 / total.max(1) as f64;
        let resolution = self.resolution();

        resolution / (uncertainty + epsilon)
    }

    /// Check if there's sufficient data for reliable IR.
    pub fn has_sufficient_data(&self) -> bool {
        self.total_samples() >= self.min_samples && self.filled_bins() >= 2
    }

    /// Get total sample count.
    pub fn total_samples(&self) -> usize {
        self.bin_counts.iter().sum()
    }

    /// Get number of bins with at least some samples.
    pub fn filled_bins(&self) -> usize {
        self.bin_counts.iter().filter(|&&c| c >= 3).count()
    }

    /// Get number of bins with at least `min_count` samples.
    pub fn bins_with_count(&self, min_count: usize) -> usize {
        self.bin_counts.iter().filter(|&&c| c >= min_count).count()
    }

    /// Get the bin counts (for diagnostics).
    pub fn bin_counts(&self) -> &[usize] {
        &self.bin_counts
    }

    /// Get the bin outcome counts (for diagnostics).
    pub fn bin_outcomes(&self) -> &[usize] {
        &self.bin_outcomes
    }

    /// Get conditional outcome rate for each bin.
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

    /// Get number of bins.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Get minimum samples required.
    pub fn min_samples(&self) -> usize {
        self.min_samples
    }

    /// Check if the model is adding value (IR > 1.0).
    pub fn is_adding_value(&self) -> bool {
        self.has_sufficient_data() && self.information_ratio() > 1.0
    }

    /// Get accuracy (hit rate) as a simple alternative metric.
    ///
    /// When IR cannot calibrate due to bin sparsity, accuracy provides
    /// a fallback measure. Less principled but still useful.
    pub fn accuracy(&self) -> f64 {
        // For a binary classifier with probability predictions:
        // - If prob > 0.5, we predict "positive"
        // - If prob <= 0.5, we predict "negative"
        // Accuracy = fraction of correct predictions
        let mut correct = 0;
        let mut total = 0;

        for (prob, outcome) in self.samples.iter() {
            let predicted_positive = *prob > 0.5;
            if predicted_positive == *outcome {
                correct += 1;
            }
            total += 1;
        }

        if total > 0 {
            correct as f64 / total as f64
        } else {
            0.5
        }
    }

    /// Get accuracy for high-confidence predictions only.
    ///
    /// High confidence = |prob - 0.5| > threshold
    pub fn high_confidence_accuracy(&self, threshold: f64) -> (f64, usize) {
        let mut correct = 0;
        let mut total = 0;

        for (prob, outcome) in self.samples.iter() {
            let confidence = (prob - 0.5).abs();
            if confidence > threshold {
                let predicted_positive = *prob > 0.5;
                if predicted_positive == *outcome {
                    correct += 1;
                }
                total += 1;
            }
        }

        let acc = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.5
        };
        (acc, total)
    }

    /// Clear all data and reset.
    pub fn clear(&mut self) {
        self.quantile_edges.clear();
        self.bin_counts.fill(0);
        self.bin_outcomes.fill(0);
        self.samples.clear();
        self.samples_since_recompute = 0;
    }

    /// Merge another binner's samples into this one.
    pub fn merge(&mut self, other: &AdaptiveBinner) {
        // Add other's samples to our buffer
        for (prob, outcome) in other.samples.iter() {
            self.update(*prob, *outcome);
        }
    }

    /// Alias for total_samples() (compatibility with InformationRatioTracker).
    pub fn n_samples(&self) -> usize {
        self.total_samples()
    }

    /// Bayesian posterior probability that IR > threshold.
    ///
    /// Uses a simplified Normal approximation for the posterior.
    /// The posterior mean is a shrinkage estimator:
    ///   posterior_mean = (prior_df * prior_mean + n * sample_ir) / (prior_df + n)
    ///
    /// The posterior variance is approximated as:
    ///   posterior_var = sample_var / (prior_df + n)
    ///
    /// # Arguments
    /// * `threshold` - IR threshold (typically 1.0)
    /// * `prior_mean` - Prior mean for IR (shrinkage target)
    /// * `prior_df` - Prior degrees of freedom (shrinkage strength)
    pub fn posterior_prob_ir_above(&self, threshold: f64, prior_mean: f64, prior_df: f64) -> f64 {
        let n = self.total_samples() as f64;
        if n < 5.0 {
            // Not enough data, return prior probability
            // Simple heuristic: if prior_mean > threshold, return ~0.6, else ~0.4
            return if prior_mean > threshold { 0.55 } else { 0.45 };
        }

        let sample_ir = self.information_ratio();

        // Posterior mean (shrinkage estimator)
        let posterior_mean = (prior_df * prior_mean + n * sample_ir) / (prior_df + n);

        // Approximate standard error of IR
        // IR has high variance with small samples, decreases as sqrt(n)
        let se = if n > 10.0 {
            0.5 / (n - 5.0).sqrt()
        } else {
            0.5
        };

        // Posterior standard deviation (combines prior and data uncertainty)
        let posterior_sd =
            se * (prior_df / (prior_df + n)).sqrt() + se / (1.0 + n / prior_df).sqrt();

        // Z-score for threshold
        if posterior_sd < 1e-10 {
            return if posterior_mean > threshold { 1.0 } else { 0.0 };
        }
        let z = (threshold - posterior_mean) / posterior_sd;

        // Normal CDF approximation for P(IR > threshold) = 1 - Phi(z)
        // Using simple logistic approximation to CDF
        let phi = 1.0 / (1.0 + (-1.7 * z).exp());
        (1.0 - phi).clamp(0.0, 1.0)
    }

    /// Posterior mean of IR (shrinkage estimator).
    ///
    /// Combines prior and sample IR weighted by effective sample size.
    pub fn posterior_mean_ir(&self, prior_mean: f64, prior_df: f64) -> f64 {
        let n = self.total_samples() as f64;
        if n < 1.0 {
            return prior_mean;
        }
        let sample_ir = self.information_ratio();
        (prior_df * prior_mean + n * sample_ir) / (prior_df + n)
    }

    /// Credible interval for IR.
    ///
    /// Returns (lower, upper) bounds at the specified confidence level.
    /// Uses Normal approximation with shrinkage.
    pub fn ir_credible_interval(
        &self,
        confidence: f64,
        prior_mean: f64,
        prior_df: f64,
    ) -> (f64, f64) {
        let n = self.total_samples() as f64;
        if n < 5.0 {
            // Wide interval with minimal data
            return (0.0, prior_mean * 2.0);
        }

        let posterior_mean = self.posterior_mean_ir(prior_mean, prior_df);

        // Approximate standard error
        let se = if n > 10.0 {
            0.5 / (n - 5.0).sqrt()
        } else {
            0.5
        };
        let posterior_sd =
            se * (prior_df / (prior_df + n)).sqrt() + se / (1.0 + n / prior_df).sqrt();

        // Z-score for confidence level (e.g., 0.95 -> 1.96)
        // Simplified: use logistic approximation inverse
        let alpha = (1.0 - confidence) / 2.0;
        let z = if alpha > 0.0 && alpha < 1.0 {
            (alpha.ln() - (1.0 - alpha).ln()) / -1.7
        } else {
            1.96
        };

        let lower = (posterior_mean - z * posterior_sd).max(0.0);
        let upper = posterior_mean + z * posterior_sd;
        (lower, upper)
    }
}

impl Default for AdaptiveBinner {
    fn default() -> Self {
        Self::new(10, 500)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let binner = AdaptiveBinner::new(10, 500);
        assert_eq!(binner.n_bins(), 10);
        assert_eq!(binner.total_samples(), 0);
        assert_eq!(binner.base_rate(), 0.5); // Default
    }

    #[test]
    fn test_min_bins() {
        let binner = AdaptiveBinner::new(2, 100);
        assert_eq!(binner.n_bins(), 3); // Minimum 3 bins
    }

    #[test]
    fn test_update_and_binning() {
        let mut binner = AdaptiveBinner::new(5, 100);

        // Add samples
        for i in 0..50 {
            let prob = (i as f64 + 0.5) / 50.0;
            let outcome = i % 2 == 0;
            binner.update(prob, outcome);
        }

        assert_eq!(binner.total_samples(), 50);
        assert!(binner.filled_bins() >= 3);
    }

    #[test]
    fn test_adaptive_binning_distributes_samples() {
        let mut binner = AdaptiveBinner::new(5, 100).with_recompute_interval(10);

        // Add clustered samples (all around 0.4)
        for i in 0..50 {
            let prob = 0.35 + 0.1 * (i as f64 / 50.0);
            let outcome = i % 3 != 0;
            binner.update(prob, outcome);
        }

        // With adaptive binning, samples should spread across bins
        let filled = binner.filled_bins();
        assert!(
            filled >= 2,
            "Expected at least 2 filled bins, got {}",
            filled
        );
    }

    #[test]
    fn test_information_ratio_perfect_discrimination() {
        let mut binner = AdaptiveBinner::new(10, 200);

        // Perfect predictions: low prob → negative, high prob → positive
        for _ in 0..50 {
            binner.update(0.1, false);
            binner.update(0.9, true);
        }

        assert!(binner.has_sufficient_data());
        let ir = binner.information_ratio();
        // With perfect discrimination, IR should be high
        assert!(
            ir > 0.5,
            "Expected IR > 0.5 for perfect discrimination, got {}",
            ir
        );
    }

    #[test]
    fn test_information_ratio_no_discrimination() {
        let mut binner = AdaptiveBinner::new(10, 200);

        // All predictions in same range with 50% outcomes
        for _ in 0..50 {
            binner.update(0.55, true);
            binner.update(0.55, false);
        }

        // Resolution should be low (all in one bin)
        let ir = binner.information_ratio();
        assert!(
            ir < 0.5,
            "Expected low IR for no discrimination, got {}",
            ir
        );
    }

    #[test]
    fn test_accuracy() {
        let mut binner = AdaptiveBinner::new(10, 100);

        // Good predictions: high prob when positive, low when negative
        for _ in 0..30 {
            binner.update(0.8, true); // High prob, positive outcome - correct
            binner.update(0.2, false); // Low prob, negative outcome - correct
        }
        for _ in 0..10 {
            binner.update(0.8, false); // Wrong
            binner.update(0.2, true); // Wrong
        }

        let acc = binner.accuracy();
        // 60 correct out of 80 = 75%
        assert!(
            (acc - 0.75).abs() < 0.05,
            "Expected ~75% accuracy, got {}",
            acc
        );
    }

    #[test]
    fn test_high_confidence_accuracy() {
        let mut binner = AdaptiveBinner::new(10, 100);

        // High confidence predictions (far from 0.5)
        for _ in 0..20 {
            binner.update(0.9, true); // High conf, correct
            binner.update(0.1, false); // High conf, correct
        }
        // Low confidence predictions (near 0.5)
        for _ in 0..20 {
            binner.update(0.55, true); // Low conf
            binner.update(0.45, false); // Low conf
        }
        // Some wrong high-conf
        for _ in 0..5 {
            binner.update(0.9, false); // High conf, wrong
        }

        let (acc, count) = binner.high_confidence_accuracy(0.3);
        // High conf: 40 correct, 5 wrong = 40/45 ≈ 0.89
        assert!(
            acc > 0.85,
            "Expected high accuracy for high-conf, got {}",
            acc
        );
        assert_eq!(count, 45);
    }

    #[test]
    fn test_rolling_buffer() {
        let mut binner = AdaptiveBinner::new(5, 50);

        // Add more samples than buffer size
        for i in 0..100 {
            let prob = (i % 10) as f64 / 10.0;
            binner.update(prob, i % 2 == 0);
        }

        // Should only have max_samples
        assert!(binner.samples.len() <= 50);
    }

    #[test]
    fn test_clear() {
        let mut binner = AdaptiveBinner::new(10, 100);

        for _ in 0..50 {
            binner.update(0.5, true);
        }

        binner.clear();

        assert_eq!(binner.total_samples(), 0);
        assert_eq!(binner.samples.len(), 0);
        assert!(binner.quantile_edges.is_empty());
    }

    #[test]
    fn test_bin_rates() {
        let mut binner = AdaptiveBinner::new(5, 100);

        // Fill specific bins with known rates
        for _ in 0..20 {
            binner.update(0.1, false); // Bin 0: 0% rate
        }
        for _ in 0..10 {
            binner.update(0.9, true); // High bin: 100% rate
        }
        for _ in 0..10 {
            binner.update(0.5, true);
            binner.update(0.5, false); // Mid bin: 50% rate
        }

        let rates = binner.bin_rates();
        assert_eq!(rates.len(), 5);
        // First bin should have low rate, last bin should have high rate
        // (exact indices depend on quantile recompute)
    }

    #[test]
    fn test_merge() {
        let mut binner1 = AdaptiveBinner::new(10, 100);
        let mut binner2 = AdaptiveBinner::new(10, 100);

        for _ in 0..25 {
            binner1.update(0.3, false);
        }
        for _ in 0..25 {
            binner2.update(0.7, true);
        }

        binner1.merge(&binner2);

        assert_eq!(binner1.total_samples(), 50);
    }

    #[test]
    fn test_is_adding_value() {
        let mut binner = AdaptiveBinner::new(10, 200);

        // Initially not adding value
        assert!(!binner.is_adding_value());

        // Add discriminative predictions
        for _ in 0..50 {
            binner.update(0.1, false);
            binner.update(0.9, true);
        }

        // Should now check if IR > 1.0
        let adding_value = binner.is_adding_value();
        let ir = binner.information_ratio();
        assert_eq!(adding_value, ir > 1.0);
    }
}
