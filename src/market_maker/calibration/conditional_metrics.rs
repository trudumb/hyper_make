//! Conditional calibration metrics sliced by regime and prediction type.
//!
//! This module provides tools for analyzing model performance across different
//! market conditions and prediction types. Key features:
//!
//! - Separate IR tracking by regime (calm, volatile, cascade)
//! - Separate IR tracking by prediction type
//! - Identification of weakest model component
//! - Health monitoring across all components

use std::collections::HashMap;

use super::information_ratio::InformationRatioTracker;
use super::prediction_log::PredictionType;
use super::DEFAULT_IR_BINS;

/// Number of market regimes (calm, volatile, cascade).
pub(crate) const N_REGIMES: usize = 3;

/// Regime names for display.
pub(crate) const REGIME_NAMES: [&str; N_REGIMES] = ["calm", "volatile", "cascade"];

/// Get the name of a regime by index.
pub(crate) fn regime_name(regime: usize) -> &'static str {
    REGIME_NAMES.get(regime).unwrap_or(&"unknown")
}

/// Conditional calibration metrics sliced by regime and prediction type.
///
/// This struct maintains separate Information Ratio trackers for each
/// combination of regime and prediction type, enabling fine-grained
/// analysis of where models succeed or fail.
#[derive(Debug)]
pub struct ConditionalCalibration {
    /// IR trackers by regime (0=calm, 1=volatile, 2=cascade)
    by_regime: [InformationRatioTracker; N_REGIMES],
    /// IR trackers by prediction type
    by_prediction_type: HashMap<PredictionType, InformationRatioTracker>,
    /// Number of bins used for all trackers
    n_bins: usize,
}

impl Clone for ConditionalCalibration {
    fn clone(&self) -> Self {
        Self {
            by_regime: self.by_regime.clone(),
            by_prediction_type: self.by_prediction_type.clone(),
            n_bins: self.n_bins,
        }
    }
}

impl ConditionalCalibration {
    /// Create a new conditional calibration tracker.
    ///
    /// # Arguments
    /// * `n_bins` - Number of probability bins for IR calculation
    pub fn new(n_bins: usize) -> Self {
        let n_bins = n_bins.max(2);

        // Initialize regime trackers
        let by_regime = [
            InformationRatioTracker::new(n_bins),
            InformationRatioTracker::new(n_bins),
            InformationRatioTracker::new(n_bins),
        ];

        // Initialize prediction type trackers
        let mut by_prediction_type = HashMap::new();
        for pred_type in PredictionType::all() {
            by_prediction_type.insert(*pred_type, InformationRatioTracker::new(n_bins));
        }

        Self {
            by_regime,
            by_prediction_type,
            n_bins,
        }
    }

    /// Create with default bin count.
    pub fn default_bins() -> Self {
        Self::new(DEFAULT_IR_BINS)
    }

    /// Update with a new prediction and outcome.
    ///
    /// # Arguments
    /// * `pred_type` - Type of prediction
    /// * `regime` - Market regime (0=calm, 1=volatile, 2=cascade)
    /// * `predicted` - Predicted probability [0, 1]
    /// * `outcome` - Actual outcome
    pub fn update(
        &mut self,
        pred_type: PredictionType,
        regime: usize,
        predicted: f64,
        outcome: bool,
    ) {
        // Update regime tracker
        let regime_idx = regime.min(N_REGIMES - 1);
        self.by_regime[regime_idx].update(predicted, outcome);

        // Update prediction type tracker
        if let Some(tracker) = self.by_prediction_type.get_mut(&pred_type) {
            tracker.update(predicted, outcome);
        }
    }

    /// Get the Information Ratio for a specific regime.
    pub fn ir_by_regime(&self, regime: usize) -> f64 {
        if regime < N_REGIMES {
            self.by_regime[regime].information_ratio()
        } else {
            0.0
        }
    }

    /// Get the Information Ratio for a specific prediction type.
    pub fn ir_by_type(&self, pred_type: &PredictionType) -> Option<f64> {
        self.by_prediction_type
            .get(pred_type)
            .map(|t| t.information_ratio())
    }

    /// Get sample count for a specific regime.
    pub fn samples_by_regime(&self, regime: usize) -> usize {
        if regime < N_REGIMES {
            self.by_regime[regime].n_samples()
        } else {
            0
        }
    }

    /// Get sample count for a specific prediction type.
    pub fn samples_by_type(&self, pred_type: &PredictionType) -> usize {
        self.by_prediction_type
            .get(pred_type)
            .map(|t| t.n_samples())
            .unwrap_or(0)
    }

    /// Find the weakest component (lowest IR) with sufficient samples.
    ///
    /// Returns the prediction type and its IR, or None if no components
    /// have sufficient samples.
    ///
    /// # Arguments
    /// * `min_samples` - Minimum samples required for consideration
    pub fn weakest_component(&self, min_samples: usize) -> Option<(PredictionType, f64)> {
        self.by_prediction_type
            .iter()
            .filter(|(_, tracker)| tracker.n_samples() >= min_samples)
            .map(|(pred_type, tracker)| (*pred_type, tracker.information_ratio()))
            .min_by(|(_, ir1), (_, ir2)| ir1.partial_cmp(ir2).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Find the strongest component (highest IR) with sufficient samples.
    pub fn strongest_component(&self, min_samples: usize) -> Option<(PredictionType, f64)> {
        self.by_prediction_type
            .iter()
            .filter(|(_, tracker)| tracker.n_samples() >= min_samples)
            .map(|(pred_type, tracker)| (*pred_type, tracker.information_ratio()))
            .max_by(|(_, ir1), (_, ir2)| ir1.partial_cmp(ir2).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Check if all tracked components are healthy (IR > threshold).
    ///
    /// Only considers components with sufficient samples.
    ///
    /// # Arguments
    /// * `min_ir` - Minimum IR threshold for health
    /// * `min_samples` - Minimum samples for consideration
    pub fn all_healthy(&self, min_ir: f64, min_samples: usize) -> bool {
        // Check all regimes
        for regime_tracker in &self.by_regime {
            if regime_tracker.n_samples() >= min_samples
                && regime_tracker.information_ratio() < min_ir
            {
                return false;
            }
        }

        // Check all prediction types
        for tracker in self.by_prediction_type.values() {
            if tracker.n_samples() >= min_samples && tracker.information_ratio() < min_ir {
                return false;
            }
        }

        true
    }

    /// Get a summary of health status for all components.
    pub fn health_summary(&self, min_ir: f64, min_samples: usize) -> HealthSummary {
        let mut unhealthy_regimes = Vec::new();
        let mut unhealthy_types = Vec::new();
        let mut insufficient_regimes = Vec::new();
        let mut insufficient_types = Vec::new();

        // Check regimes
        for (regime, tracker) in self.by_regime.iter().enumerate() {
            if tracker.n_samples() < min_samples {
                insufficient_regimes.push(regime);
            } else if tracker.information_ratio() < min_ir {
                unhealthy_regimes.push((regime, tracker.information_ratio()));
            }
        }

        // Check prediction types
        for (pred_type, tracker) in &self.by_prediction_type {
            if tracker.n_samples() < min_samples {
                insufficient_types.push(*pred_type);
            } else if tracker.information_ratio() < min_ir {
                unhealthy_types.push((*pred_type, tracker.information_ratio()));
            }
        }

        HealthSummary {
            is_healthy: unhealthy_regimes.is_empty() && unhealthy_types.is_empty(),
            unhealthy_regimes,
            unhealthy_types,
            insufficient_regimes,
            insufficient_types,
        }
    }

    /// Get the average IR across all regimes.
    pub fn average_regime_ir(&self) -> f64 {
        let total: f64 = self.by_regime.iter().map(|t| t.information_ratio()).sum();
        total / N_REGIMES as f64
    }

    /// Get the average IR across all prediction types.
    pub fn average_type_ir(&self) -> f64 {
        if self.by_prediction_type.is_empty() {
            return 0.0;
        }

        let total: f64 = self
            .by_prediction_type
            .values()
            .map(|t| t.information_ratio())
            .sum();
        total / self.by_prediction_type.len() as f64
    }

    /// Clear all trackers.
    pub fn clear(&mut self) {
        for tracker in &mut self.by_regime {
            tracker.clear();
        }
        for tracker in self.by_prediction_type.values_mut() {
            tracker.clear();
        }
    }

    /// Get the total number of samples across all trackers.
    pub fn total_samples(&self) -> usize {
        // Sum samples from prediction types (each sample goes to exactly one type)
        self.by_prediction_type
            .values()
            .map(|t| t.n_samples())
            .sum()
    }

    /// Get all regime IRs as an array.
    pub fn regime_irs(&self) -> [f64; N_REGIMES] {
        [
            self.by_regime[0].information_ratio(),
            self.by_regime[1].information_ratio(),
            self.by_regime[2].information_ratio(),
        ]
    }

    /// Get all prediction type IRs as a map.
    pub fn type_irs(&self) -> HashMap<PredictionType, f64> {
        self.by_prediction_type
            .iter()
            .map(|(k, v)| (*k, v.information_ratio()))
            .collect()
    }

    /// Get the regime tracker directly (for advanced analysis).
    pub fn regime_tracker(&self, regime: usize) -> Option<&InformationRatioTracker> {
        if regime < N_REGIMES {
            Some(&self.by_regime[regime])
        } else {
            None
        }
    }

    /// Get the prediction type tracker directly (for advanced analysis).
    pub fn type_tracker(&self, pred_type: &PredictionType) -> Option<&InformationRatioTracker> {
        self.by_prediction_type.get(pred_type)
    }
}

// ConditionalCalibration is Send + Sync because all its fields are
unsafe impl Send for ConditionalCalibration {}
unsafe impl Sync for ConditionalCalibration {}

/// Summary of health status across all components.
#[derive(Debug, Clone)]
pub struct HealthSummary {
    /// Whether all components with sufficient data are healthy
    pub is_healthy: bool,
    /// Regimes that are unhealthy (regime_index, ir)
    pub unhealthy_regimes: Vec<(usize, f64)>,
    /// Prediction types that are unhealthy (type, ir)
    pub unhealthy_types: Vec<(PredictionType, f64)>,
    /// Regimes with insufficient samples
    pub insufficient_regimes: Vec<usize>,
    /// Prediction types with insufficient samples
    pub insufficient_types: Vec<PredictionType>,
}

impl HealthSummary {
    /// Get the number of unhealthy components.
    pub fn n_unhealthy(&self) -> usize {
        self.unhealthy_regimes.len() + self.unhealthy_types.len()
    }

    /// Get the number of components with insufficient data.
    pub fn n_insufficient(&self) -> usize {
        self.insufficient_regimes.len() + self.insufficient_types.len()
    }

    /// Format a human-readable report.
    pub fn report(&self) -> String {
        let mut lines = Vec::new();

        if self.is_healthy {
            lines.push("All components healthy".to_string());
        } else {
            lines.push("HEALTH WARNING: Some components are underperforming".to_string());

            for (regime, ir) in &self.unhealthy_regimes {
                lines.push(format!(
                    "  - Regime '{}' IR: {:.3} (below threshold)",
                    regime_name(*regime),
                    ir
                ));
            }

            for (pred_type, ir) in &self.unhealthy_types {
                lines.push(format!(
                    "  - Prediction type '{}' IR: {:.3} (below threshold)",
                    pred_type.name(),
                    ir
                ));
            }
        }

        if !self.insufficient_regimes.is_empty() || !self.insufficient_types.is_empty() {
            lines.push("Insufficient samples for:".to_string());

            for regime in &self.insufficient_regimes {
                lines.push(format!("  - Regime '{}'", regime_name(*regime)));
            }

            for pred_type in &self.insufficient_types {
                lines.push(format!("  - Prediction type '{}'", pred_type.name()));
            }
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let cc = ConditionalCalibration::new(10);
        assert_eq!(cc.n_bins, 10);
        assert_eq!(cc.by_regime.len(), 3);
        assert_eq!(cc.by_prediction_type.len(), 4);
    }

    #[test]
    fn test_default_bins() {
        let cc = ConditionalCalibration::default_bins();
        assert_eq!(cc.n_bins, DEFAULT_IR_BINS);
    }

    #[test]
    fn test_update() {
        let mut cc = ConditionalCalibration::new(10);

        cc.update(PredictionType::FillProbability, 0, 0.8, true);

        assert_eq!(cc.samples_by_regime(0), 1);
        assert_eq!(cc.samples_by_regime(1), 0);
        assert_eq!(cc.samples_by_type(&PredictionType::FillProbability), 1);
        assert_eq!(cc.samples_by_type(&PredictionType::AdverseSelection), 0);
    }

    #[test]
    fn test_ir_by_regime() {
        let mut cc = ConditionalCalibration::new(10);

        // Add predictions that separate well in regime 0
        for _ in 0..100 {
            cc.update(PredictionType::FillProbability, 0, 0.1, false);
            cc.update(PredictionType::FillProbability, 0, 0.9, true);
        }

        // Regime 0 should have good IR
        let ir0 = cc.ir_by_regime(0);
        assert!(ir0 > 0.5);

        // Other regimes should have 0 IR (no data)
        assert_eq!(cc.ir_by_regime(1), 0.0);
        assert_eq!(cc.ir_by_regime(2), 0.0);
    }

    #[test]
    fn test_ir_by_type() {
        let mut cc = ConditionalCalibration::new(10);

        // Add predictions for FillProbability
        for _ in 0..100 {
            cc.update(PredictionType::FillProbability, 0, 0.1, false);
            cc.update(PredictionType::FillProbability, 0, 0.9, true);
        }

        // FillProbability should have good IR
        let ir_fill = cc.ir_by_type(&PredictionType::FillProbability);
        assert!(ir_fill.is_some());
        assert!(ir_fill.unwrap() > 0.5);

        // AdverseSelection should have 0 IR (no data)
        let ir_as = cc.ir_by_type(&PredictionType::AdverseSelection);
        assert_eq!(ir_as, Some(0.0));
    }

    #[test]
    fn test_weakest_component() {
        let mut cc = ConditionalCalibration::new(10);

        // FillProbability: good predictions
        for _ in 0..100 {
            cc.update(PredictionType::FillProbability, 0, 0.1, false);
            cc.update(PredictionType::FillProbability, 0, 0.9, true);
        }

        // AdverseSelection: random predictions (poor IR)
        for i in 0..200 {
            cc.update(PredictionType::AdverseSelection, 0, 0.5, i % 2 == 0);
        }

        let weakest = cc.weakest_component(50);
        assert!(weakest.is_some());

        let (weak_type, weak_ir) = weakest.unwrap();
        assert_eq!(weak_type, PredictionType::AdverseSelection);
        assert!(weak_ir < 0.5);
    }

    #[test]
    fn test_strongest_component() {
        let mut cc = ConditionalCalibration::new(10);

        // FillProbability: good predictions
        for _ in 0..100 {
            cc.update(PredictionType::FillProbability, 0, 0.1, false);
            cc.update(PredictionType::FillProbability, 0, 0.9, true);
        }

        // AdverseSelection: random predictions
        for i in 0..200 {
            cc.update(PredictionType::AdverseSelection, 0, 0.5, i % 2 == 0);
        }

        let strongest = cc.strongest_component(50);
        assert!(strongest.is_some());

        let (strong_type, strong_ir) = strongest.unwrap();
        assert_eq!(strong_type, PredictionType::FillProbability);
        assert!(strong_ir > 0.5);
    }

    #[test]
    fn test_all_healthy() {
        let mut cc = ConditionalCalibration::new(10);

        // Add some data (but not enough for min_samples)
        for _ in 0..10 {
            cc.update(PredictionType::FillProbability, 0, 0.5, true);
        }

        // Should be healthy because insufficient samples are ignored
        assert!(cc.all_healthy(1.0, 50));

        // With lower min_samples, it's not healthy (IR close to 0)
        assert!(!cc.all_healthy(1.0, 5));
    }

    #[test]
    fn test_health_summary() {
        let mut cc = ConditionalCalibration::new(10);

        // Add random predictions (poor IR)
        for i in 0..100 {
            cc.update(PredictionType::FillProbability, 0, 0.5, i % 2 == 0);
        }

        let summary = cc.health_summary(1.0, 50);

        assert!(!summary.is_healthy);
        assert!(summary.unhealthy_types.len() > 0);
        assert!(summary.insufficient_types.len() > 0); // Other types have no data
    }

    #[test]
    fn test_average_regime_ir() {
        let mut cc = ConditionalCalibration::new(10);

        // All regimes with some data
        for regime in 0..3 {
            for _ in 0..100 {
                cc.update(PredictionType::FillProbability, regime, 0.1, false);
                cc.update(PredictionType::FillProbability, regime, 0.9, true);
            }
        }

        let avg = cc.average_regime_ir();
        assert!(avg > 0.5);
    }

    #[test]
    fn test_clear() {
        let mut cc = ConditionalCalibration::new(10);

        cc.update(PredictionType::FillProbability, 0, 0.5, true);
        cc.update(PredictionType::AdverseSelection, 1, 0.5, false);

        assert!(cc.total_samples() > 0);

        cc.clear();

        assert_eq!(cc.total_samples(), 0);
        assert_eq!(cc.samples_by_regime(0), 0);
        assert_eq!(cc.samples_by_type(&PredictionType::FillProbability), 0);
    }

    #[test]
    fn test_regime_irs() {
        let mut cc = ConditionalCalibration::new(10);

        for _ in 0..100 {
            cc.update(PredictionType::FillProbability, 0, 0.1, false);
            cc.update(PredictionType::FillProbability, 0, 0.9, true);
        }

        let irs = cc.regime_irs();
        assert!(irs[0] > 0.5);
        assert_eq!(irs[1], 0.0);
        assert_eq!(irs[2], 0.0);
    }

    #[test]
    fn test_type_irs() {
        let mut cc = ConditionalCalibration::new(10);

        for _ in 0..100 {
            cc.update(PredictionType::FillProbability, 0, 0.1, false);
            cc.update(PredictionType::FillProbability, 0, 0.9, true);
        }

        let irs = cc.type_irs();
        assert!(irs.contains_key(&PredictionType::FillProbability));
        assert!(irs[&PredictionType::FillProbability] > 0.5);
    }

    #[test]
    fn test_regime_name() {
        assert_eq!(regime_name(0), "calm");
        assert_eq!(regime_name(1), "volatile");
        assert_eq!(regime_name(2), "cascade");
        assert_eq!(regime_name(3), "unknown");
    }

    #[test]
    fn test_health_summary_report() {
        let mut cc = ConditionalCalibration::new(10);

        for i in 0..100 {
            cc.update(PredictionType::FillProbability, 0, 0.5, i % 2 == 0);
        }

        let summary = cc.health_summary(1.0, 50);
        let report = summary.report();

        assert!(report.contains("HEALTH WARNING") || report.contains("All components healthy"));
    }

    #[test]
    fn test_clone() {
        let mut cc = ConditionalCalibration::new(10);

        cc.update(PredictionType::FillProbability, 0, 0.8, true);

        let cloned = cc.clone();

        assert_eq!(cloned.samples_by_regime(0), 1);
        assert_eq!(cloned.samples_by_type(&PredictionType::FillProbability), 1);
    }
}
