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
        let o_bar: f64 = outcomes.iter().map(|&o| if o { 1.0 } else { 0.0 }).sum::<f64>() / n_f64;

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
            let o_bar_k: f64 =
                bin.iter().map(|(_, o)| if *o { 1.0 } else { 0.0 }).sum::<f64>() / n_k;

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
        self.points
            .iter()
            .max_by(|a, b| {
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
            let was_filled = outcomes
                .fills
                .iter()
                .any(|f| f.level_index == i);

            // Add to overall
            self.fill_predictions_1s.push((level.p_fill_1s, was_filled));

            // Add to conditional slices
            let regime = record.market_state.regime;
            self.conditional_fill_predictions
                .entry(ConditionalSlice::Regime(regime))
                .or_insert_with(Vec::new)
                .push((level.p_fill_1s, was_filled));

            // Add volatility slice
            let vol_quartile = volatility_to_quartile(record.market_state.sigma_effective);
            self.conditional_fill_predictions
                .entry(ConditionalSlice::VolatilityQuartile(vol_quartile))
                .or_insert_with(Vec::new)
                .push((level.p_fill_1s, was_filled));

            // Add inventory state slice
            let inv_state = inventory_to_state(record.market_state.inventory);
            self.conditional_fill_predictions
                .entry(ConditionalSlice::InventoryState(inv_state))
                .or_insert_with(Vec::new)
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
    pub fn compute_conditional_brier(&self, slice: &ConditionalSlice) -> Option<BrierDecomposition> {
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
        for (slice, _) in &self.conditional_fill_predictions {
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
        assert!(brier.reliability < 0.1, "Reliability should be low for good calibration");
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
        assert!(brier.information_ratio > 0.5, "IR should be positive for discriminating model");
    }
}
