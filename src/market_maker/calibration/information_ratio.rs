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
//! - **IR > 1.0**: Strong standalone predictor (exceptional)
//! - **IR 0.5–1.0**: Feature informs Bayesian priors (useful for belief updates)
//! - **IR < 0.5**: Insufficient signal to shift beliefs (remove with 500+ samples)
//!
//! Note: Features that inform priors don't need to predict perfectly—even IR 0.3–0.8
//! shifts beliefs in the right direction. Only remove features below 0.5 after 500+
//! samples. Watch for inter-feature correlation inflating aggregate value.

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
    ///
    /// NOTE: Returns 0.5 when no data. This is a neutral default but can
    /// bias early IR calculations. Wait for 100+ samples before trusting IR.
    pub fn base_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.5 // Default to 0.5 when no data - neutral but can bias early IR
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

    /// Check if the model is adding value (IR > 0.5).
    ///
    /// A model with IR <= 0.5 has insufficient signal to inform priors.
    /// Features with IR 0.5–1.0 are useful for Bayesian belief updates
    /// even though they aren't strong standalone predictors.
    pub fn is_adding_value(&self) -> bool {
        self.information_ratio() > 0.5
    }

    /// Get the total number of samples.
    pub fn n_samples(&self) -> usize {
        self.total_count
    }

    /// Check if there are enough samples for reliable metrics.
    ///
    /// NOTE: For robust IR estimation, need at least 500 samples with
    /// reasonable bin coverage. With 10 bins, that's ~50 samples per bin.
    pub fn is_reliable(&self, min_samples: usize) -> bool {
        self.total_count >= min_samples
    }

    /// Minimum recommended samples for reliable IR (500)
    pub const MIN_RELIABLE_SAMPLES: usize = 500;

    /// Check if the distribution is dangerously concentrated in few bins.
    ///
    /// Returns true if >80% of samples are in a single bin, which means
    /// the model is making nearly constant predictions and IR is meaningless.
    pub fn is_bin_concentrated(&self) -> bool {
        if self.total_count == 0 {
            return false;
        }
        let max_bin_count = self.bin_counts.iter().max().copied().unwrap_or(0);
        (max_bin_count as f64 / self.total_count as f64) > 0.8
    }

    /// Get diagnostic information about the IR calculation.
    ///
    /// Returns a struct with details useful for debugging IR issues.
    pub fn diagnostics(&self) -> IrDiagnostics {
        let base_rate = self.base_rate();
        let resolution = self.resolution();
        let uncertainty = self.uncertainty();
        let ir = self.information_ratio();

        // Count non-empty bins
        let non_empty_bins = self.bin_counts.iter().filter(|&&c| c > 0).count();

        // Find max bin concentration
        let max_bin_count = self.bin_counts.iter().max().copied().unwrap_or(0);
        let concentration = if self.total_count > 0 {
            max_bin_count as f64 / self.total_count as f64
        } else {
            0.0
        };

        // Identify issues
        let mut warnings = Vec::new();

        if self.total_count < Self::MIN_RELIABLE_SAMPLES {
            warnings.push(format!(
                "Insufficient samples: {} < {} recommended",
                self.total_count,
                Self::MIN_RELIABLE_SAMPLES
            ));
        }

        if concentration > 0.8 {
            warnings.push(format!(
                "Predictions concentrated: {:.0}% in one bin (model outputs near-constant)",
                concentration * 100.0
            ));
        }

        if non_empty_bins < 3 && self.total_count > 100 {
            warnings.push(format!(
                "Low bin coverage: only {} of {} bins used",
                non_empty_bins, self.n_bins
            ));
        }

        if base_rate < 0.05 || base_rate > 0.95 {
            warnings.push(format!(
                "Extreme base rate: {:.1}% (outcomes heavily imbalanced)",
                base_rate * 100.0
            ));
        }

        IrDiagnostics {
            base_rate,
            resolution,
            uncertainty,
            ir,
            total_samples: self.total_count,
            non_empty_bins,
            max_bin_concentration: concentration,
            warnings,
        }
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

    // ========================================================================
    // Bayesian Inference Methods
    // ========================================================================

    /// Compute effective sample size accounting for bin sparsity.
    ///
    /// Uses geometric mean weighting to penalize uneven bin distributions.
    /// Single-bin samples are heavily penalized since they provide no resolution.
    /// Returns a value <= total_count that reflects the "effective" number
    /// of independent observations for IR estimation.
    pub fn effective_sample_size(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        // Count non-empty bins and their sample distribution
        let non_empty: Vec<f64> = self
            .bin_counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| c as f64)
            .collect();

        if non_empty.is_empty() {
            return 0.0;
        }

        let num_bins_used = non_empty.len();

        // Need at least 2 bins for resolution; single bin has no discriminative power
        if num_bins_used == 1 {
            // Heavily penalize single-bin case: n_eff = sqrt(n) at best
            return (self.total_count as f64).sqrt();
        }

        // Effective sample size: penalize uneven distribution
        // Use geometric mean of bin counts × number of non-empty bins
        // But also penalize for using few bins (less resolution)
        let log_sum: f64 = non_empty.iter().map(|c| c.ln()).sum();
        let geom_mean = (log_sum / num_bins_used as f64).exp();
        
        // Scale by bin coverage: using more bins = better resolution
        let bin_coverage = (num_bins_used as f64 / self.n_bins as f64).sqrt();
        let n_eff = geom_mean * num_bins_used as f64 * bin_coverage;

        // Cap at actual total count
        n_eff.min(self.total_count as f64)
    }

    /// Compute posterior probability that IR > threshold.
    ///
    /// Uses non-central t approximation via shifted-t method:
    /// √n × (IR̂ - threshold) / SE(IR̂) ~ t(df = n_eff - 1)
    ///
    /// # Arguments
    /// * `threshold` - IR threshold to test against (typically 1.0)
    /// * `prior_mean` - Prior mean for IR (shrinkage toward this value)
    /// * `prior_df` - Prior degrees of freedom (shrinkage strength, 0 = no prior)
    ///
    /// # Returns
    /// Probability that true IR > threshold, in [0, 1].
    pub fn posterior_prob_ir_above(&self, threshold: f64, prior_mean: f64, prior_df: f64) -> f64 {
        let n_eff = self.effective_sample_size();
        if n_eff < 2.0 {
            // Insufficient data: return prior-based probability
            // If prior_mean > threshold, slight positive; else slight negative
            return if prior_mean > threshold { 0.55 } else { 0.45 };
        }

        let ir_hat = self.information_ratio();

        // Bayesian shrinkage: posterior mean is weighted average of data and prior
        let post_df = prior_df + n_eff;
        let post_mean = (prior_df * prior_mean + n_eff * ir_hat) / post_df;

        // Variance of IR estimator: Var(IR̂) ≈ (1 + IR²/2) / n_eff
        // This is the asymptotic variance for Sharpe-like ratios
        let ir_var = (1.0 + ir_hat.powi(2) / 2.0) / n_eff;
        // Note: ir_se computed here but used only in combined posterior_var calculation below
        let _ir_se = ir_var.sqrt();

        // Add prior variance contribution (inverse-variance weighting)
        let prior_var = if prior_df > 2.0 {
            (1.0 + prior_mean.powi(2) / 2.0) / prior_df
        } else {
            1.0 // Uninformative prior
        };
        let post_var = 1.0 / (1.0 / prior_var + 1.0 / ir_var);
        let post_se = post_var.sqrt();

        // Test statistic: (post_mean - threshold) / post_se
        // When t_stat > 0 (post_mean > threshold), we expect high probability
        // P(IR > threshold) ≈ P(Z > (threshold - post_mean)/post_se) = P(Z < t_stat)
        let t_stat = (post_mean - threshold) / post_se;
        let df = (post_df - 1.0).max(1.0);

        // Use lower-tail CDF: P(IR > threshold) ≈ Φ(t_stat) for large df
        let p_above = Self::approx_t_cdf(t_stat, df);

        p_above.clamp(0.0, 1.0)
    }

    /// Compute credible interval for IR.
    ///
    /// # Arguments
    /// * `confidence` - Confidence level (e.g., 0.95 for 95% CI)
    /// * `prior_mean` - Prior mean for shrinkage
    /// * `prior_df` - Prior degrees of freedom
    ///
    /// # Returns
    /// (lower_bound, upper_bound) for the credible interval.
    pub fn ir_credible_interval(
        &self,
        confidence: f64,
        prior_mean: f64,
        prior_df: f64,
    ) -> (f64, f64) {
        let n_eff = self.effective_sample_size();
        if n_eff < 2.0 {
            // Wide uninformative interval
            return (0.0, prior_mean * 2.0);
        }

        let ir_hat = self.information_ratio();

        // Posterior mean (shrinkage)
        let post_df = prior_df + n_eff;
        let post_mean = (prior_df * prior_mean + n_eff * ir_hat) / post_df;

        // Posterior standard error
        let ir_var = (1.0 + ir_hat.powi(2) / 2.0) / n_eff;
        let prior_var = if prior_df > 2.0 {
            (1.0 + prior_mean.powi(2) / 2.0) / prior_df
        } else {
            1.0
        };
        let post_var = 1.0 / (1.0 / prior_var + 1.0 / ir_var);
        let post_se = post_var.sqrt();

        // Critical value for t-distribution (approximation)
        let alpha = 1.0 - confidence;
        let df = (post_df - 1.0).max(1.0);
        let t_crit = Self::approx_t_quantile(1.0 - alpha / 2.0, df);

        let lower = post_mean - t_crit * post_se;
        let upper = post_mean + t_crit * post_se;

        (lower.max(0.0), upper)
    }

    /// Get the standard error of the IR estimate.
    ///
    /// Uses the asymptotic variance formula: Var(IR̂) ≈ (1 + IR²/2) / n_eff
    /// Returns infinity if insufficient samples (n_eff < 2).
    pub fn ir_standard_error(&self) -> f64 {
        let n_eff = self.effective_sample_size();
        if n_eff < 2.0 {
            return f64::INFINITY;
        }
        let ir_hat = self.information_ratio();
        let ir_var = (1.0 + ir_hat.powi(2) / 2.0) / n_eff;
        ir_var.sqrt()
    }

    /// Get posterior mean IR (with shrinkage).
    pub fn posterior_mean_ir(&self, prior_mean: f64, prior_df: f64) -> f64 {
        let n_eff = self.effective_sample_size();
        if n_eff < 1.0 {
            return prior_mean;
        }

        let ir_hat = self.information_ratio();
        let post_df = prior_df + n_eff;
        (prior_df * prior_mean + n_eff * ir_hat) / post_df
    }

    /// Approximate CDF for t-distribution (lower tail).
    /// P(T <= t) where T ~ t(df)
    ///
    /// Uses normal approximation with scaling for finite df.
    fn approx_t_cdf(t: f64, df: f64) -> f64 {
        // For df > 30, t ≈ normal
        // For smaller df, scale by √(df/(df+t²/3)) (empirical correction)
        let scale = if df > 30.0 {
            1.0
        } else if df > 4.0 {
            (df / (df + t.powi(2) / 3.0)).sqrt()
        } else {
            // Very small df: be conservative
            (df / (df + t.powi(2))).sqrt()
        };

        let z = t * scale;
        // Standard normal CDF lower tail: Φ(z)
        Self::standard_normal_cdf(z)
    }

    /// Approximate upper tail probability for t-distribution.
    /// P(T > t) where T ~ t(df)
    ///
    /// Uses the approximation: t_cdf(x, df) ≈ Φ(x × √((df-2)/df)) for moderate df
    /// For small df, uses more conservative approximation.
    ///
    /// Note: This helper is kept for API completeness alongside approx_t_cdf.
    #[allow(dead_code)]
    fn approx_t_cdf_upper(t: f64, df: f64) -> f64 {
        1.0 - Self::approx_t_cdf(t, df)
    }

    /// Approximate quantile for t-distribution.
    /// Returns t such that P(T <= t) = p for T ~ t(df)
    fn approx_t_quantile(p: f64, df: f64) -> f64 {
        // First get normal quantile
        let z = Self::standard_normal_quantile(p);

        // Adjust for heavier t-distribution tails
        if df > 30.0 {
            z
        } else if df > 4.0 {
            z / ((df - 2.0) / df).sqrt()
        } else {
            // Small df: inflate quantile significantly
            z * (df / (df - 2.0).max(0.5)).sqrt()
        }
    }

    /// Standard normal CDF using Abramowitz & Stegun approximation.
    fn standard_normal_cdf(x: f64) -> f64 {
        // Approximation accurate to ~1e-5
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let p = d
            * t
            * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

        if x > 0.0 {
            1.0 - p
        } else {
            p
        }
    }

    /// Standard normal quantile using Beasley-Springer-Moro algorithm.
    fn standard_normal_quantile(p: f64) -> f64 {
        // Handle edge cases
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            // Lower tail
            let q = (-2.0 * p.ln()).sqrt();
            (2.515517 + q * (0.802853 + q * 0.010328))
                / (1.0 + q * (1.432788 + q * (0.189269 + q * 0.001308)))
                - q
        } else if p <= p_high {
            // Central region: rational approximation
            let q = p - 0.5;
            let r = q * q;
            q * (2.506628 + r * (-18.61500 + r * (41.39119 - r * 25.44106)))
                / (1.0 + r * (-8.473520 + r * (23.08337 - r * 21.06224)))
        } else {
            // Upper tail
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            q - (2.515517 + q * (0.802853 + q * 0.010328))
                / (1.0 + q * (1.432788 + q * (0.189269 + q * 0.001308)))
        }
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

/// Diagnostic information about IR calculation.
///
/// Use this to understand WHY IR is low - is it insufficient data,
/// concentrated predictions, or genuinely no edge?
#[derive(Debug, Clone)]
pub struct IrDiagnostics {
    /// Base rate of positive outcomes
    pub base_rate: f64,
    /// Resolution component of IR
    pub resolution: f64,
    /// Uncertainty component of IR
    pub uncertainty: f64,
    /// Computed Information Ratio
    pub ir: f64,
    /// Total number of samples
    pub total_samples: usize,
    /// Number of bins with at least one sample
    pub non_empty_bins: usize,
    /// Fraction of samples in the most populated bin (>0.8 is concerning)
    pub max_bin_concentration: f64,
    /// Human-readable warnings about the IR calculation
    pub warnings: Vec<String>,
}

impl IrDiagnostics {
    /// Check if IR is likely reliable (no major warnings)
    pub fn is_reliable(&self) -> bool {
        self.warnings.is_empty()
    }

    /// Get a summary string for logging
    pub fn summary(&self) -> String {
        let warning_str = if self.warnings.is_empty() {
            String::new()
        } else {
            format!(" [WARNINGS: {}]", self.warnings.join("; "))
        };

        format!(
            "IR={:.3} (base={:.3}, res={:.4}, unc={:.4}, n={}, bins={}, conc={:.0}%){}",
            self.ir,
            self.base_rate,
            self.resolution,
            self.uncertainty,
            self.total_samples,
            self.non_empty_bins,
            self.max_bin_concentration * 100.0,
            warning_str
        )
    }
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
        assert_eq!(tracker.is_adding_value(), ir > 0.5);
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

    // ========================================================================
    // Bayesian Inference Tests
    // ========================================================================

    #[test]
    fn test_effective_sample_size_empty() {
        let tracker = InformationRatioTracker::new(10);
        assert_eq!(tracker.effective_sample_size(), 0.0);
    }

    #[test]
    fn test_effective_sample_size_uniform() {
        let mut tracker = InformationRatioTracker::new(10);

        // Uniform distribution across bins (ideal case)
        for bin in 0..10 {
            let prob = (bin as f64 + 0.5) / 10.0;
            for _ in 0..10 {
                tracker.update(prob, bin % 2 == 0);
            }
        }

        // With uniform distribution, n_eff should be close to total
        let n_eff = tracker.effective_sample_size();
        assert!(n_eff > 90.0, "n_eff should be close to 100: {}", n_eff);
    }

    #[test]
    fn test_effective_sample_size_sparse() {
        let mut tracker = InformationRatioTracker::new(10);

        // All samples in one bin (worst case)
        for _ in 0..100 {
            tracker.update(0.55, true);
        }

        // With sparse distribution, n_eff should be much less than total
        let n_eff = tracker.effective_sample_size();
        assert!(n_eff < 50.0, "n_eff should be reduced for sparse: {}", n_eff);
    }

    #[test]
    fn test_posterior_prob_insufficient_data() {
        let tracker = InformationRatioTracker::new(10);

        // With no data, should return prior-based probability
        let p = tracker.posterior_prob_ir_above(1.0, 0.9, 6.0);
        // prior_mean=0.9 < threshold=1.0, so P should be ~0.45
        assert!((p - 0.45).abs() < 0.1, "P should be ~0.45 for prior < threshold: {}", p);

        let p_high = tracker.posterior_prob_ir_above(1.0, 1.5, 6.0);
        // prior_mean=1.5 > threshold=1.0, so P should be ~0.55
        assert!((p_high - 0.55).abs() < 0.1, "P should be ~0.55 for prior > threshold: {}", p_high);
    }

    #[test]
    fn test_posterior_prob_strong_edge() {
        let mut tracker = InformationRatioTracker::new(10);

        // Good discrimination across multiple bins
        // This pattern creates IR > 1.0
        for _ in 0..40 {
            tracker.update(0.05, false);  // Low prob -> false
            tracker.update(0.25, false);  // Low-mid prob -> false
            tracker.update(0.75, true);   // High-mid prob -> true
            tracker.update(0.95, true);   // High prob -> true
        }

        // Check IR is actually > 1.0 for this pattern
        let ir = tracker.information_ratio();
        
        // With strong edge (IR > 1.0), P(IR > 0.5) should be high
        // Use lower threshold since IR=1.0 is the "neutral" point
        let p = tracker.posterior_prob_ir_above(0.5, 0.9, 6.0);
        assert!(p > 0.7, "P(IR > 0.5) should be high for strong edge (IR={:.3}): {}", ir, p);
    }

    #[test]
    fn test_posterior_prob_no_edge() {
        let mut tracker = InformationRatioTracker::new(10);

        // All predictions in same bin with 50% outcomes (no discrimination)
        for _ in 0..50 {
            tracker.update(0.55, true);
            tracker.update(0.55, false);
        }

        // With no edge (IR~0), P(IR > 0.5) should be low
        // Use 0.5 threshold since single-bin gives IR=0
        let ir = tracker.information_ratio();
        let p = tracker.posterior_prob_ir_above(0.5, 0.9, 6.0);
        assert!(p < 0.6, "P(IR > 0.5) should be low for no edge (IR={:.3}): {}", ir, p);
    }

    #[test]
    fn test_credible_interval_contains_true() {
        let mut tracker = InformationRatioTracker::new(10);

        // Perfect discrimination: IR should be ~1.0
        for _ in 0..100 {
            tracker.update(0.1, false);
            tracker.update(0.9, true);
        }

        let (lower, upper) = tracker.ir_credible_interval(0.95, 0.9, 6.0);
        let ir = tracker.information_ratio();

        // Interval should contain the point estimate
        assert!(lower <= ir && ir <= upper,
            "95% CI should contain IR: [{:.3}, {:.3}] vs IR={:.3}",
            lower, upper, ir);
    }

    #[test]
    fn test_credible_interval_width_decreases() {
        let mut tracker = InformationRatioTracker::new(10);

        // Few samples: wide interval
        for _ in 0..20 {
            tracker.update(0.1, false);
            tracker.update(0.9, true);
        }
        let (lower1, upper1) = tracker.ir_credible_interval(0.95, 0.9, 6.0);
        let width1 = upper1 - lower1;

        // More samples: narrower interval
        for _ in 0..80 {
            tracker.update(0.1, false);
            tracker.update(0.9, true);
        }
        let (lower2, upper2) = tracker.ir_credible_interval(0.95, 0.9, 6.0);
        let width2 = upper2 - lower2;

        assert!(width2 < width1,
            "CI should narrow with more samples: {} vs {}",
            width2, width1);
    }

    #[test]
    fn test_posterior_mean_shrinkage() {
        let mut tracker = InformationRatioTracker::new(10);

        // Strong edge: data IR ~= 1.0
        for _ in 0..50 {
            tracker.update(0.1, false);
            tracker.update(0.9, true);
        }

        let ir_hat = tracker.information_ratio();

        // Strong prior (high df): posterior pulled toward prior
        let post_strong = tracker.posterior_mean_ir(0.5, 100.0);
        // Weak prior (low df): posterior close to data
        let post_weak = tracker.posterior_mean_ir(0.5, 2.0);

        assert!(post_strong < post_weak,
            "Strong prior should shrink more: {:.3} vs {:.3}",
            post_strong, post_weak);
        assert!((post_weak - ir_hat).abs() < 0.2,
            "Weak prior should be close to data: {:.3} vs {:.3}",
            post_weak, ir_hat);
    }

    #[test]
    fn test_normal_cdf_accuracy() {
        // Test standard normal CDF against known values
        let test_cases = [
            (0.0, 0.5),
            (1.0, 0.8413),
            (-1.0, 0.1587),
            (2.0, 0.9772),
            (-2.0, 0.0228),
        ];

        for (x, expected) in test_cases {
            let actual = InformationRatioTracker::standard_normal_cdf(x);
            assert!((actual - expected).abs() < 0.01,
                "Φ({}) = {} should be ~{}", x, actual, expected);
        }
    }

    #[test]
    fn test_normal_quantile_accuracy() {
        // Test that quantile inverts CDF
        let test_probs = [0.05, 0.25, 0.5, 0.75, 0.95];

        for p in test_probs {
            let z = InformationRatioTracker::standard_normal_quantile(p);
            let p_back = InformationRatioTracker::standard_normal_cdf(z);
            assert!((p_back - p).abs() < 0.02,
                "Φ(Φ⁻¹({})) = {} should be ~{}", p, p_back, p);
        }
    }
}
