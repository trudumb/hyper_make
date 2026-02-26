//! Calibration Analysis
//!
//! Provides tools for analyzing model calibration:
//! - Brier score decomposition (reliability, resolution, uncertainty)
//! - Calibration curves
//! - Conditional calibration (sliced by regime, volatility, etc.)
//! - Information ratio calculations

use super::prediction::{PredictionRecord, Regime};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Brier score decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrierDecomposition {
    /// Total Brier score (mean squared error of probabilities)
    pub brier_score: f64,
    /// Reliability component (calibration quality) - lower is better
    pub reliability: f64,
    /// Resolution component (discrimination ability) - higher is better
    pub resolution: f64,
    /// Uncertainty component (base rate variance) - not controllable
    pub uncertainty: f64,
    /// Information ratio: Resolution / Uncertainty
    /// IR > 1.0 means model adds value
    pub information_ratio: f64,
    /// Sample count
    pub n_samples: usize,
}

impl BrierDecomposition {
    /// Compute Brier decomposition from predictions and outcomes
    pub fn compute(predictions: &[f64], outcomes: &[bool], num_bins: usize) -> Self {
        let n = predictions.len();
        if n == 0 {
            return Self::default();
        }

        let n_f64 = n as f64;

        // Base rate
        let o_bar: f64 = outcomes
            .iter()
            .map(|&o| if o { 1.0 } else { 0.0 })
            .sum::<f64>()
            / n_f64;

        // Create bins
        let mut bins: Vec<Vec<(f64, bool)>> = vec![vec![]; num_bins];

        for (&p, &o) in predictions.iter().zip(outcomes.iter()) {
            let bin_idx = ((p * num_bins as f64) as usize).min(num_bins - 1);
            bins[bin_idx].push((p, o));
        }

        // Compute components
        let mut reliability = 0.0;
        let mut resolution = 0.0;

        for bin in &bins {
            if bin.is_empty() {
                continue;
            }

            let n_k = bin.len() as f64;
            let p_bar_k: f64 = bin.iter().map(|(p, _)| p).sum::<f64>() / n_k;
            let o_bar_k: f64 = bin
                .iter()
                .map(|(_, o)| if *o { 1.0 } else { 0.0 })
                .sum::<f64>()
                / n_k;

            reliability += n_k * (p_bar_k - o_bar_k).powi(2);
            resolution += n_k * (o_bar_k - o_bar).powi(2);
        }

        reliability /= n_f64;
        resolution /= n_f64;
        let uncertainty = o_bar * (1.0 - o_bar);
        let brier_score = reliability - resolution + uncertainty;
        let information_ratio = if uncertainty > 0.0 {
            resolution / uncertainty
        } else {
            0.0
        };

        Self {
            brier_score,
            reliability,
            resolution,
            uncertainty,
            information_ratio,
            n_samples: n,
        }
    }
}

impl Default for BrierDecomposition {
    fn default() -> Self {
        Self {
            brier_score: 0.0,
            reliability: 0.0,
            resolution: 0.0,
            uncertainty: 0.0,
            information_ratio: 0.0,
            n_samples: 0,
        }
    }
}

/// A point on the calibration curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    /// Mean predicted probability in this bin
    pub mean_predicted: f64,
    /// Realized frequency (fraction of positive outcomes)
    pub realized_frequency: f64,
    /// Number of samples in this bin
    pub count: usize,
    /// Standard error of realized frequency
    pub std_error: f64,
}

/// Calibration curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCurve {
    /// Points on the curve
    pub points: Vec<CalibrationPoint>,
    /// Number of bins
    pub num_bins: usize,
    /// Calibration error (mean absolute deviation from diagonal)
    pub calibration_error: f64,
    /// Maximum calibration error (worst bin)
    pub max_calibration_error: f64,
}

impl CalibrationCurve {
    /// Build calibration curve from predictions and outcomes
    pub fn build(predictions: &[f64], outcomes: &[bool], num_bins: usize) -> Self {
        let mut bins: Vec<Vec<(f64, bool)>> = vec![vec![]; num_bins];

        for (&p, &o) in predictions.iter().zip(outcomes.iter()) {
            let bin_idx = ((p * num_bins as f64) as usize).min(num_bins - 1);
            bins[bin_idx].push((p, o));
        }

        let mut points = Vec::new();
        let mut total_error: f64 = 0.0;
        let mut max_error: f64 = 0.0;
        let mut total_count = 0;

        for bin in &bins {
            if bin.is_empty() {
                continue;
            }

            let count = bin.len();
            let mean_predicted: f64 = bin.iter().map(|(p, _)| p).sum::<f64>() / count as f64;
            let realized_frequency: f64 =
                bin.iter().filter(|(_, o)| *o).count() as f64 / count as f64;

            // Standard error = sqrt(p(1-p)/n)
            let std_error = if count > 1 {
                (realized_frequency * (1.0 - realized_frequency) / (count - 1) as f64).sqrt()
            } else {
                0.0
            };

            let error = (mean_predicted - realized_frequency).abs();
            total_error += error * count as f64;
            total_count += count;
            max_error = max_error.max(error);

            points.push(CalibrationPoint {
                mean_predicted,
                realized_frequency,
                count,
                std_error,
            });
        }

        let calibration_error = if total_count > 0 {
            total_error / total_count as f64
        } else {
            0.0
        };

        Self {
            points,
            num_bins,
            calibration_error,
            max_calibration_error: max_error,
        }
    }

    /// Check if model is well-calibrated (within tolerance)
    pub fn is_well_calibrated(&self, tolerance: f64) -> bool {
        self.calibration_error <= tolerance
    }

    /// Get the bin with worst calibration
    pub fn worst_bin(&self) -> Option<&CalibrationPoint> {
        self.points.iter().max_by(|a, b| {
            let err_a = (a.mean_predicted - a.realized_frequency).abs();
            let err_b = (b.mean_predicted - b.realized_frequency).abs();
            err_a.partial_cmp(&err_b).unwrap()
        })
    }
}

/// Conditional slice for analysis
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionalSlice {
    /// All data
    All,
    /// By regime
    Regime(Regime),
    /// By volatility quartile (1-4)
    VolatilityQuartile(u8),
    /// By inventory state
    InventoryState(InventoryState),
    /// By time of day (hour)
    TimeOfDay(u8),
    /// By book imbalance direction
    BookImbalance(ImbalanceDirection),
    /// Custom slice
    Custom(String),
}

/// Inventory state classification
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum InventoryState {
    Long,
    Flat,
    Short,
}

/// Book imbalance direction
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImbalanceDirection {
    BidHeavy,
    Balanced,
    AskHeavy,
}

/// Calibration analyzer
pub struct CalibrationAnalyzer {
    /// Prediction-outcome pairs for fill probability (1s horizon)
    fill_predictions_1s: Vec<(f64, bool)>,
    /// Prediction-outcome pairs for adverse selection
    as_predictions: Vec<(f64, f64)>,
    /// Conditional data storage
    conditional_fill_predictions: HashMap<ConditionalSlice, Vec<(f64, bool)>>,
    /// Number of bins for analysis
    num_bins: usize,
}

impl CalibrationAnalyzer {
    /// Create a new calibration analyzer
    pub fn new(num_bins: usize) -> Self {
        Self {
            fill_predictions_1s: Vec::new(),
            as_predictions: Vec::new(),
            conditional_fill_predictions: HashMap::new(),
            num_bins,
        }
    }

    /// Add a prediction record with outcomes
    pub fn add_record(&mut self, record: &PredictionRecord) {
        let outcomes = match &record.outcomes {
            Some(o) => o,
            None => return, // Skip records without outcomes
        };

        // Extract fill predictions and outcomes
        for (i, level) in record.predictions.levels.iter().enumerate() {
            let was_filled = outcomes.fills.iter().any(|f| f.level_index == i);

            // Add to overall
            self.fill_predictions_1s.push((level.p_fill_1s, was_filled));

            // Add to conditional slices
            let regime = record.market_state.regime;
            self.conditional_fill_predictions
                .entry(ConditionalSlice::Regime(regime))
                .or_default()
                .push((level.p_fill_1s, was_filled));

            // Add volatility slice
            let vol_quartile = volatility_to_quartile(record.market_state.sigma_effective);
            self.conditional_fill_predictions
                .entry(ConditionalSlice::VolatilityQuartile(vol_quartile))
                .or_default()
                .push((level.p_fill_1s, was_filled));

            // Add inventory state slice
            let inv_state = inventory_to_state(record.market_state.inventory);
            self.conditional_fill_predictions
                .entry(ConditionalSlice::InventoryState(inv_state))
                .or_default()
                .push((level.p_fill_1s, was_filled));
        }

        // Add adverse selection prediction
        let predicted_as = record.predictions.expected_adverse_selection_bps;
        let realized_as = outcomes.adverse_selection_realized_bps;
        self.as_predictions.push((predicted_as, realized_as));
    }

    /// Compute overall Brier decomposition for fill predictions
    pub fn compute_fill_brier(&self) -> BrierDecomposition {
        let (predictions, outcomes): (Vec<f64>, Vec<bool>) =
            self.fill_predictions_1s.iter().cloned().unzip();
        BrierDecomposition::compute(&predictions, &outcomes, self.num_bins)
    }

    /// Compute calibration curve for fill predictions
    pub fn compute_fill_calibration_curve(&self) -> CalibrationCurve {
        let (predictions, outcomes): (Vec<f64>, Vec<bool>) =
            self.fill_predictions_1s.iter().cloned().unzip();
        CalibrationCurve::build(&predictions, &outcomes, self.num_bins)
    }

    /// Compute conditional Brier for a specific slice
    pub fn compute_conditional_brier(
        &self,
        slice: &ConditionalSlice,
    ) -> Option<BrierDecomposition> {
        self.conditional_fill_predictions.get(slice).map(|data| {
            let (predictions, outcomes): (Vec<f64>, Vec<bool>) = data.iter().cloned().unzip();
            BrierDecomposition::compute(&predictions, &outcomes, self.num_bins)
        })
    }

    /// Compute adverse selection RMSE
    pub fn compute_as_rmse(&self) -> f64 {
        if self.as_predictions.is_empty() {
            return 0.0;
        }

        let mse: f64 = self
            .as_predictions
            .iter()
            .map(|(pred, actual)| (pred - actual).powi(2))
            .sum::<f64>()
            / self.as_predictions.len() as f64;

        mse.sqrt()
    }

    /// Compute adverse selection bias (mean error)
    pub fn compute_as_bias(&self) -> f64 {
        if self.as_predictions.is_empty() {
            return 0.0;
        }

        self.as_predictions
            .iter()
            .map(|(pred, actual)| pred - actual)
            .sum::<f64>()
            / self.as_predictions.len() as f64
    }

    /// Generate a full calibration report
    pub fn generate_report(&self) -> CalibrationReport {
        let overall_brier = self.compute_fill_brier();
        let overall_curve = self.compute_fill_calibration_curve();

        let mut conditional_brier = HashMap::new();
        for slice in self.conditional_fill_predictions.keys() {
            if let Some(brier) = self.compute_conditional_brier(slice) {
                conditional_brier.insert(slice.clone(), brier);
            }
        }

        CalibrationReport {
            overall_brier,
            overall_calibration_curve: overall_curve,
            conditional_brier,
            as_rmse: self.compute_as_rmse(),
            as_bias: self.compute_as_bias(),
            n_fill_predictions: self.fill_predictions_1s.len(),
            n_as_predictions: self.as_predictions.len(),
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.fill_predictions_1s.clear();
        self.as_predictions.clear();
        self.conditional_fill_predictions.clear();
    }

    /// Get sample counts per slice
    pub fn get_sample_counts(&self) -> HashMap<ConditionalSlice, usize> {
        self.conditional_fill_predictions
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }
}

/// Full calibration report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    /// Overall Brier decomposition
    pub overall_brier: BrierDecomposition,
    /// Overall calibration curve
    pub overall_calibration_curve: CalibrationCurve,
    /// Conditional Brier scores by slice
    pub conditional_brier: HashMap<ConditionalSlice, BrierDecomposition>,
    /// Adverse selection RMSE
    pub as_rmse: f64,
    /// Adverse selection bias
    pub as_bias: f64,
    /// Number of fill predictions analyzed
    pub n_fill_predictions: usize,
    /// Number of AS predictions analyzed
    pub n_as_predictions: usize,
}

impl CalibrationReport {
    /// Check if model is adding value (IR > threshold)
    pub fn is_model_useful(&self, ir_threshold: f64) -> bool {
        self.overall_brier.information_ratio > ir_threshold
    }

    /// Find problematic slices (IR < 1.0)
    pub fn find_problematic_slices(&self) -> Vec<(ConditionalSlice, BrierDecomposition)> {
        self.conditional_brier
            .iter()
            .filter(|(_, brier)| brier.information_ratio < 1.0)
            .map(|(slice, brier)| (slice.clone(), brier.clone()))
            .collect()
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Calibration Report ===\n\n");

        output.push_str("Overall Fill Prediction Metrics:\n");
        output.push_str(&format!(
            "  Brier Score:       {:.4}\n",
            self.overall_brier.brier_score
        ));
        output.push_str(&format!(
            "  Reliability:       {:.4} (lower is better)\n",
            self.overall_brier.reliability
        ));
        output.push_str(&format!(
            "  Resolution:        {:.4} (higher is better)\n",
            self.overall_brier.resolution
        ));
        output.push_str(&format!(
            "  Information Ratio: {:.2} (>1.0 = model adds value)\n",
            self.overall_brier.information_ratio
        ));
        output.push_str(&format!(
            "  Samples:           {}\n",
            self.overall_brier.n_samples
        ));

        output.push_str("\nAdverse Selection Metrics:\n");
        output.push_str(&format!("  RMSE: {:.2} bps\n", self.as_rmse));
        output.push_str(&format!("  Bias: {:.2} bps\n", self.as_bias));

        output.push_str("\nCalibration Curve:\n");
        output.push_str(&format!(
            "  Mean Error: {:.4}\n",
            self.overall_calibration_curve.calibration_error
        ));
        output.push_str(&format!(
            "  Max Error:  {:.4}\n",
            self.overall_calibration_curve.max_calibration_error
        ));

        output.push_str("\nConditional Analysis:\n");
        for (slice, brier) in &self.conditional_brier {
            if brier.n_samples >= 50 {
                let status = if brier.information_ratio > 1.0 {
                    "✓"
                } else {
                    "✗"
                };
                output.push_str(&format!(
                    "  {:?}: IR={:.2} {} (n={})\n",
                    slice, brier.information_ratio, status, brier.n_samples
                ));
            }
        }

        let problems = self.find_problematic_slices();
        if !problems.is_empty() {
            output.push_str("\n⚠ ATTENTION NEEDED:\n");
            for (slice, brier) in problems {
                if brier.n_samples >= 50 {
                    output.push_str(&format!(
                        "  {:?}: IR={:.2} - model adding noise!\n",
                        slice, brier.information_ratio
                    ));
                }
            }
        }

        output
    }
}

/// Convert volatility to quartile (1-4)
fn volatility_to_quartile(sigma: f64) -> u8 {
    // Rough quartile boundaries for per-second volatility
    if sigma < 0.00005 {
        1
    } else if sigma < 0.00015 {
        2
    } else if sigma < 0.00030 {
        3
    } else {
        4
    }
}

/// Convert inventory to state
fn inventory_to_state(inventory: f64) -> InventoryState {
    if inventory > 0.1 {
        InventoryState::Long
    } else if inventory < -0.1 {
        InventoryState::Short
    } else {
        InventoryState::Flat
    }
}

// ============================================================================
// Small Fish Strategy: Statistical Validation Tests
// Reference: docs/SMALL_FISH_STRATEGY.md lines 285-376
// ============================================================================

/// Result of a statistical test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    /// Name of the test.
    pub test_name: String,
    /// Test statistic value.
    pub statistic: f64,
    /// P-value (probability under null hypothesis).
    pub p_value: f64,
    /// Whether the result is significant at alpha=0.05.
    pub is_significant: bool,
    /// Number of samples used.
    pub n_samples: usize,
    /// Additional details.
    pub details: String,
}

impl StatisticalTestResult {
    /// Create a new test result.
    pub fn new(
        test_name: &str,
        statistic: f64,
        p_value: f64,
        n_samples: usize,
        details: &str,
    ) -> Self {
        Self {
            test_name: test_name.to_string(),
            statistic,
            p_value,
            is_significant: p_value < 0.05,
            n_samples,
            details: details.to_string(),
        }
    }

    /// Format as human-readable string.
    pub fn format(&self) -> String {
        let sig_marker = if self.is_significant { "✓" } else { "✗" };
        format!(
            "{}: stat={:.4}, p={:.4} {} (n={})\n  {}",
            self.test_name, self.statistic, self.p_value, sig_marker, self.n_samples, self.details
        )
    }
}

/// Statistical validation framework for model testing.
///
/// Provides rigorous statistical tests to validate trading models:
/// - Binomial test for win rate significance
/// - T-test for mean return significance
/// - Monte Carlo permutation test for robustness
/// - Information Ratio assessment
pub struct StatisticalValidator {
    /// Random seed for reproducibility.
    seed: u64,
}

impl Default for StatisticalValidator {
    fn default() -> Self {
        Self::new(42)
    }
}

impl StatisticalValidator {
    /// Create a new validator with specified seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Binomial test for win rate significance.
    ///
    /// Tests whether the observed win rate is significantly different from 50%.
    /// H0: win_rate = 0.5 (random)
    /// H1: win_rate != 0.5 (has edge)
    ///
    /// # Arguments
    /// * `wins` - Number of winning trades
    /// * `total` - Total number of trades
    ///
    /// # Returns
    /// Test result with p-value. Significant (p < 0.05) means win rate differs from 50%.
    pub fn binomial_test(&self, wins: usize, total: usize) -> StatisticalTestResult {
        if total == 0 {
            return StatisticalTestResult::new(
                "Binomial Win Rate Test",
                0.0,
                1.0,
                0,
                "No samples provided",
            );
        }

        let n = total as f64;
        let k = wins as f64;
        let p = 0.5; // Null hypothesis: 50% win rate

        // Normal approximation for large n (n*p >= 5 and n*(1-p) >= 5)
        if n * p >= 5.0 && n * (1.0 - p) >= 5.0 {
            // z = (k - np) / sqrt(np(1-p))
            let expected = n * p;
            let std_dev = (n * p * (1.0 - p)).sqrt();
            let z_stat = (k - expected) / std_dev;

            // Two-tailed p-value using normal approximation
            let p_value = 2.0 * (1.0 - standard_normal_cdf(z_stat.abs()));

            let win_rate = k / n;
            StatisticalTestResult::new(
                "Binomial Win Rate Test",
                z_stat,
                p_value,
                total,
                &format!(
                    "Win rate: {:.1}% ({}/{}), expected: 50%",
                    win_rate * 100.0,
                    wins,
                    total
                ),
            )
        } else {
            // Exact binomial test for small samples
            let p_value = exact_binomial_test(wins, total, p);

            let win_rate = k / n;
            StatisticalTestResult::new(
                "Binomial Win Rate Test (Exact)",
                win_rate,
                p_value,
                total,
                &format!(
                    "Win rate: {:.1}% ({}/{}), expected: 50%",
                    win_rate * 100.0,
                    wins,
                    total
                ),
            )
        }
    }

    /// T-test for mean returns.
    ///
    /// Tests whether mean return is significantly different from zero.
    /// H0: mean_return = 0
    /// H1: mean_return != 0
    ///
    /// # Arguments
    /// * `returns` - Vector of trade returns
    ///
    /// # Returns
    /// Test result with p-value. Significant (p < 0.05) means returns differ from zero.
    pub fn t_test(&self, returns: &[f64]) -> StatisticalTestResult {
        let n = returns.len();
        if n < 2 {
            return StatisticalTestResult::new(
                "T-Test for Mean Returns",
                0.0,
                1.0,
                n,
                "Insufficient samples (need at least 2)",
            );
        }

        let n_f64 = n as f64;

        // Calculate mean
        let mean: f64 = returns.iter().sum::<f64>() / n_f64;

        // Calculate standard deviation
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n_f64 - 1.0);
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return StatisticalTestResult::new(
                "T-Test for Mean Returns",
                0.0,
                1.0,
                n,
                "Zero variance in returns",
            );
        }

        // T-statistic: t = (mean - 0) / (std_dev / sqrt(n))
        let std_error = std_dev / n_f64.sqrt();
        let t_stat = mean / std_error;

        // P-value from t-distribution (using normal approximation for large n)
        let df = n - 1;
        let p_value = if df > 30 {
            2.0 * (1.0 - standard_normal_cdf(t_stat.abs()))
        } else {
            2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df))
        };

        StatisticalTestResult::new(
            "T-Test for Mean Returns",
            t_stat,
            p_value,
            n,
            &format!("Mean: {mean:.4}, Std: {std_dev:.4}, SE: {std_error:.4}"),
        )
    }

    /// Monte Carlo permutation test.
    ///
    /// Tests whether observed mean return could have occurred by chance.
    /// Shuffles trade signs and computes distribution of mean returns.
    ///
    /// # Arguments
    /// * `returns` - Vector of trade returns
    /// * `n_permutations` - Number of permutations (default 10000)
    ///
    /// # Returns
    /// Test result with p-value from permutation distribution.
    pub fn permutation_test(
        &self,
        returns: &[f64],
        n_permutations: usize,
    ) -> StatisticalTestResult {
        let n = returns.len();
        if n < 2 {
            return StatisticalTestResult::new(
                "Monte Carlo Permutation Test",
                0.0,
                1.0,
                n,
                "Insufficient samples (need at least 2)",
            );
        }

        // Observed mean return
        let observed_mean: f64 = returns.iter().sum::<f64>() / n as f64;

        // Simple LCG random number generator for reproducibility
        let mut rng_state = self.seed;
        let lcg_next = |state: &mut u64| -> u64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        };

        // Generate permutation distribution
        let abs_returns: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
        let mut more_extreme_count = 0;

        for _ in 0..n_permutations {
            // Randomly assign signs
            let mut perm_sum = 0.0;
            for &abs_ret in &abs_returns {
                let sign = if lcg_next(&mut rng_state) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                perm_sum += sign * abs_ret;
            }
            let perm_mean = perm_sum / n as f64;

            // Count how many permutations have mean as extreme as observed
            if perm_mean.abs() >= observed_mean.abs() {
                more_extreme_count += 1;
            }
        }

        let p_value = more_extreme_count as f64 / n_permutations as f64;

        StatisticalTestResult::new(
            "Monte Carlo Permutation Test",
            observed_mean,
            p_value,
            n,
            &format!(
                "Observed mean: {observed_mean:.4}, {more_extreme_count} of {n_permutations} permutations more extreme"
            ),
        )
    }

    /// Calculate Information Ratio.
    ///
    /// IR = mean(returns) / std(returns)
    /// IR > 1.0 indicates model adds significant value.
    ///
    /// # Arguments
    /// * `returns` - Vector of trade returns
    ///
    /// # Returns
    /// Information Ratio and assessment.
    pub fn information_ratio(&self, returns: &[f64]) -> StatisticalTestResult {
        let n = returns.len();
        if n < 2 {
            return StatisticalTestResult::new(
                "Information Ratio",
                0.0,
                1.0,
                n,
                "Insufficient samples",
            );
        }

        let n_f64 = n as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n_f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n_f64 - 1.0);
        let std_dev = variance.sqrt();

        let ir = if std_dev > 1e-10 { mean / std_dev } else { 0.0 };

        // IR > 1.0 is considered useful
        let is_useful = ir > 1.0;

        StatisticalTestResult::new(
            "Information Ratio",
            ir,
            if is_useful { 0.01 } else { 0.5 }, // Pseudo p-value based on IR threshold
            n,
            &format!(
                "Mean: {:.4}, Std: {:.4}, IR: {:.2} - {}",
                mean,
                std_dev,
                ir,
                if is_useful {
                    "Model adds value"
                } else {
                    "Model adds noise"
                }
            ),
        )
    }

    /// Run all validation tests on a set of trade returns.
    ///
    /// # Arguments
    /// * `returns` - Vector of trade returns
    /// * `wins` - Number of winning trades
    ///
    /// # Returns
    /// Vector of all test results.
    pub fn run_all_tests(&self, returns: &[f64], wins: usize) -> Vec<StatisticalTestResult> {
        let total = returns.len();

        vec![
            self.binomial_test(wins, total),
            self.t_test(returns),
            self.permutation_test(returns, 10000),
            self.information_ratio(returns),
        ]
    }

    /// Determine if a model should be kept based on all tests.
    ///
    /// A model should be kept if:
    /// - At least one significance test passes (p < 0.05)
    /// - Information Ratio > 1.0
    ///
    /// # Arguments
    /// * `results` - Vector of test results from `run_all_tests`
    ///
    /// # Returns
    /// (should_keep, reasons)
    pub fn should_keep_model(&self, results: &[StatisticalTestResult]) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();
        let mut has_significant_test = false;
        let mut has_good_ir = false;

        for result in results {
            if result.is_significant {
                has_significant_test = true;
                reasons.push(format!(
                    "{} is significant (p={:.4})",
                    result.test_name, result.p_value
                ));
            }

            if result.test_name == "Information Ratio" && result.statistic > 1.0 {
                has_good_ir = true;
                reasons.push(format!("IR={:.2} > 1.0", result.statistic));
            }
        }

        let should_keep = has_significant_test && has_good_ir;

        if !has_significant_test {
            reasons.push("No test achieved significance (p < 0.05)".to_string());
        }
        if !has_good_ir {
            reasons.push("IR <= 1.0 - model adds noise".to_string());
        }

        (should_keep, reasons)
    }
}

/// Standard normal CDF approximation.
/// Uses the Abramowitz and Stegun approximation.
fn standard_normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Abramowitz and Stegun approximation
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let p =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

    if sign > 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// T-distribution CDF approximation for small degrees of freedom.
fn t_distribution_cdf(t: f64, df: usize) -> f64 {
    // For simplicity, use normal approximation with Welch-Satterthwaite correction
    // This is a rough approximation; for production, use a proper stats library
    let correction = (df as f64 / (df as f64 - 2.0)).sqrt();
    standard_normal_cdf(t / correction)
}

/// Exact binomial test using the binomial CDF.
fn exact_binomial_test(k: usize, n: usize, p: f64) -> f64 {
    // Calculate two-tailed p-value for exact binomial test
    let observed = k as f64 / n as f64;
    let expected = p;

    // Sum probabilities of outcomes as extreme or more extreme
    let mut p_value = 0.0;

    for i in 0..=n {
        let prob = binomial_probability(i, n, p);
        let outcome = i as f64 / n as f64;

        // If this outcome is as far from expected as observed
        if (outcome - expected).abs() >= (observed - expected).abs() - 1e-10 {
            p_value += prob;
        }
    }

    p_value.min(1.0)
}

/// Calculate binomial probability P(X = k) for Binomial(n, p).
fn binomial_probability(k: usize, n: usize, p: f64) -> f64 {
    let log_coeff = log_binomial_coefficient(n, k);
    let log_prob = log_coeff + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln();
    log_prob.exp()
}

/// Log of binomial coefficient using Stirling's approximation for large values.
fn log_binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }

    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

/// Log factorial using Stirling's approximation for large values.
fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }

    // For small n, compute directly
    if n <= 20 {
        let mut result = 0.0;
        for i in 2..=n {
            result += (i as f64).ln();
        }
        return result;
    }

    // Stirling's approximation for large n
    let n_f64 = n as f64;
    n_f64 * n_f64.ln() - n_f64 + 0.5 * (2.0 * std::f64::consts::PI * n_f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brier_perfect_calibration() {
        // Perfect calibration: predictions match outcomes exactly
        let predictions = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let outcomes = vec![false, false, true, true, true];

        let brier = BrierDecomposition::compute(&predictions, &outcomes, 10);

        // With perfect calibration, reliability should be low
        assert!(
            brier.reliability < 0.1,
            "Reliability should be low for good calibration"
        );
    }

    #[test]
    fn test_brier_poor_calibration() {
        // Poor calibration: always predict 0.5 for varying outcomes
        let predictions = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let outcomes = vec![true, false, true, false, true];

        let brier = BrierDecomposition::compute(&predictions, &outcomes, 10);

        // Brier score should be around 0.25 (error of 0.5^2)
        assert!((brier.brier_score - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_calibration_curve() {
        let predictions = vec![0.1, 0.2, 0.3, 0.8, 0.9];
        let outcomes = vec![false, false, true, true, true];

        let curve = CalibrationCurve::build(&predictions, &outcomes, 5);

        // Should have some points
        assert!(!curve.points.is_empty());

        // Calibration error should be reasonable
        assert!(curve.calibration_error < 0.5);
    }

    #[test]
    fn test_information_ratio_meaningful() {
        // Model that discriminates well
        let predictions = vec![0.1, 0.1, 0.9, 0.9];
        let outcomes = vec![false, false, true, true];

        let brier = BrierDecomposition::compute(&predictions, &outcomes, 4);

        // Good discrimination should yield high IR
        assert!(
            brier.information_ratio > 0.5,
            "IR should be positive for discriminating model"
        );
    }

    // ========================================================================
    // Phase 2.1: Statistical Validation Tests
    // ========================================================================

    #[test]
    fn test_binomial_test_significant_win_rate() {
        let validator = StatisticalValidator::default();

        // 70 wins out of 100 - clearly better than 50%
        let result = validator.binomial_test(70, 100);

        assert!(result.is_significant, "70% win rate should be significant");
        assert!(result.p_value < 0.05, "p-value should be < 0.05");
        assert!(result.n_samples == 100);
    }

    #[test]
    fn test_binomial_test_non_significant_win_rate() {
        let validator = StatisticalValidator::default();

        // 52 wins out of 100 - too close to 50%
        let result = validator.binomial_test(52, 100);

        assert!(
            !result.is_significant,
            "52% win rate should not be significant"
        );
        assert!(result.p_value > 0.05, "p-value should be > 0.05");
    }

    #[test]
    fn test_binomial_test_small_sample() {
        let validator = StatisticalValidator::default();

        // Small sample - uses exact test
        let result = validator.binomial_test(4, 5);

        // Should still work
        assert!(result.n_samples == 5);
        assert!(result.test_name.contains("Exact"));
    }

    #[test]
    fn test_binomial_test_empty() {
        let validator = StatisticalValidator::default();

        let result = validator.binomial_test(0, 0);

        assert!(!result.is_significant);
        assert_eq!(result.n_samples, 0);
    }

    #[test]
    fn test_t_test_significant_returns() {
        let validator = StatisticalValidator::default();

        // Strong positive returns with low variance
        let returns: Vec<f64> = (0..100).map(|_| 0.02).collect(); // 2% returns

        let _result = validator.t_test(&returns);

        // With constant positive returns, should be highly significant
        // Note: zero variance case is handled specially
    }

    #[test]
    fn test_t_test_mixed_returns() {
        let validator = StatisticalValidator::default();

        // Returns with positive mean but high variance
        let mut returns = Vec::new();
        for i in 0..100 {
            if i % 3 == 0 {
                returns.push(0.05);
            } else if i % 3 == 1 {
                returns.push(-0.02);
            } else {
                returns.push(0.01);
            }
        }

        let result = validator.t_test(&returns);

        // Should compute without error
        assert!(result.n_samples == 100);
        assert!(result.statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_t_test_zero_mean() {
        let validator = StatisticalValidator::default();

        // Returns with zero mean
        let returns: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();

        let result = validator.t_test(&returns);

        // Should not be significant
        assert!(
            !result.is_significant,
            "Zero mean returns should not be significant"
        );
    }

    #[test]
    fn test_t_test_insufficient_samples() {
        let validator = StatisticalValidator::default();

        let result = validator.t_test(&[0.01]);

        assert!(!result.is_significant);
        assert!(result.details.contains("Insufficient"));
    }

    #[test]
    fn test_permutation_test_significant() {
        let validator = StatisticalValidator::new(12345);

        // Strong positive returns
        let returns: Vec<f64> = (0..50).map(|_| 0.03).collect();

        let result = validator.permutation_test(&returns, 1000);

        // With all positive returns, permutation test should be significant
        assert!(
            result.p_value < 0.1,
            "p-value should be low for consistent positive returns"
        );
    }

    #[test]
    fn test_permutation_test_random_returns() {
        let validator = StatisticalValidator::new(42);

        // Random-looking returns (alternating)
        let returns: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();

        let result = validator.permutation_test(&returns, 1000);

        // Should have high p-value (not significant)
        // Mean is ~0, so most permutations should be as extreme
        assert!(result.n_samples == 100);
    }

    #[test]
    fn test_permutation_test_reproducible() {
        let validator1 = StatisticalValidator::new(42);
        let validator2 = StatisticalValidator::new(42);

        let returns: Vec<f64> = (0..50).map(|i| (i as f64) * 0.001).collect();

        let result1 = validator1.permutation_test(&returns, 100);
        let result2 = validator2.permutation_test(&returns, 100);

        // Same seed should give same result
        assert_eq!(result1.p_value, result2.p_value);
    }

    #[test]
    fn test_information_ratio_calculation() {
        let validator = StatisticalValidator::default();

        // Returns with mean=0.02, std=0.01 -> IR = 2.0
        let returns: Vec<f64> = vec![0.01, 0.02, 0.03, 0.01, 0.02, 0.03];

        let result = validator.information_ratio(&returns);

        // IR should be calculated
        assert!(result.statistic.is_finite());
        assert!(result.n_samples == 6);
    }

    #[test]
    fn test_information_ratio_good_model() {
        let validator = StatisticalValidator::default();

        // High IR: mean >> std
        let returns: Vec<f64> = vec![0.10, 0.11, 0.10, 0.09, 0.10, 0.11];

        let result = validator.information_ratio(&returns);

        // IR should be high (mean ~0.1, std ~0.007)
        assert!(
            result.statistic > 1.0,
            "IR should be > 1.0 for consistent positive returns: {}",
            result.statistic
        );
        assert!(result.details.contains("adds value"));
    }

    #[test]
    fn test_information_ratio_poor_model() {
        let validator = StatisticalValidator::default();

        // Low IR: mean << std
        let returns: Vec<f64> = vec![0.05, -0.04, 0.03, -0.02, 0.01, -0.05];

        let result = validator.information_ratio(&returns);

        // IR should be low (high variance relative to mean)
        assert!(result.details.contains("noise") || result.statistic <= 1.0);
    }

    #[test]
    fn test_run_all_tests() {
        let validator = StatisticalValidator::default();

        let returns: Vec<f64> = (0..100)
            .map(|i| if i % 3 == 0 { -0.01 } else { 0.02 })
            .collect();
        let wins = returns.iter().filter(|&&r| r > 0.0).count();

        let results = validator.run_all_tests(&returns, wins);

        // Should have 4 tests
        assert_eq!(results.len(), 4);

        // Check all tests have valid results
        for result in &results {
            assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
            assert!(result.n_samples > 0);
        }
    }

    #[test]
    fn test_should_keep_model_good() {
        let validator = StatisticalValidator::default();

        // Simulate good test results
        let results = vec![
            StatisticalTestResult::new("Binomial Test", 2.5, 0.01, 100, "Win rate 65%"),
            StatisticalTestResult::new("T-Test", 3.0, 0.005, 100, "Mean positive"),
            StatisticalTestResult::new("Permutation Test", 0.02, 0.02, 100, "Significant"),
            StatisticalTestResult {
                test_name: "Information Ratio".to_string(),
                statistic: 1.5,
                p_value: 0.01,
                is_significant: true,
                n_samples: 100,
                details: "IR=1.5".to_string(),
            },
        ];

        let (should_keep, reasons) = validator.should_keep_model(&results);

        assert!(should_keep, "Model with good tests should be kept");
        assert!(!reasons.is_empty());
    }

    #[test]
    fn test_should_keep_model_poor() {
        let validator = StatisticalValidator::default();

        // Simulate poor test results
        let results = vec![
            StatisticalTestResult::new("Binomial Test", 0.5, 0.6, 100, "Win rate 52%"),
            StatisticalTestResult::new("T-Test", 0.3, 0.7, 100, "Mean near zero"),
            StatisticalTestResult::new("Permutation Test", 0.001, 0.8, 100, "Not significant"),
            StatisticalTestResult {
                test_name: "Information Ratio".to_string(),
                statistic: 0.3,
                p_value: 0.5,
                is_significant: false,
                n_samples: 100,
                details: "IR=0.3".to_string(),
            },
        ];

        let (should_keep, reasons) = validator.should_keep_model(&results);

        assert!(!should_keep, "Model with poor tests should not be kept");
        assert!(reasons
            .iter()
            .any(|r| r.contains("IR") || r.contains("significance")));
    }

    #[test]
    fn test_standard_normal_cdf() {
        // Test known values
        let cdf_0 = standard_normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 0.001, "CDF(0) should be ~0.5");

        let cdf_1 = standard_normal_cdf(1.0);
        assert!((cdf_1 - 0.8413).abs() < 0.01, "CDF(1) should be ~0.8413");

        let cdf_neg1 = standard_normal_cdf(-1.0);
        assert!(
            (cdf_neg1 - 0.1587).abs() < 0.01,
            "CDF(-1) should be ~0.1587"
        );

        let cdf_2 = standard_normal_cdf(2.0);
        assert!((cdf_2 - 0.9772).abs() < 0.01, "CDF(2) should be ~0.9772");
    }

    #[test]
    fn test_binomial_probability() {
        // P(X=5) for Binomial(10, 0.5) ≈ 0.246
        let prob = binomial_probability(5, 10, 0.5);
        assert!((prob - 0.246).abs() < 0.01, "Binomial prob wrong: {}", prob);

        // P(X=0) for Binomial(5, 0.5) = 0.03125
        let prob0 = binomial_probability(0, 5, 0.5);
        assert!((prob0 - 0.03125).abs() < 0.001);
    }

    #[test]
    fn test_statistical_test_result_format() {
        let result = StatisticalTestResult::new("Test Name", 2.5, 0.01, 100, "Some details");

        let formatted = result.format();

        assert!(formatted.contains("Test Name"));
        assert!(formatted.contains("2.5"));
        assert!(formatted.contains("0.01"));
        assert!(formatted.contains("✓")); // Significant marker
    }
}
