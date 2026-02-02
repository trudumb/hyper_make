//! Calibration infrastructure for market making model validation.
//!
//! This module provides tools for measuring and validating model calibration:
//!
//! - **PredictionLog**: Thread-safe logging of predictions with outcome resolution
//! - **BrierScoreTracker**: Rolling Brier score for probability calibration
//! - **InformationRatioTracker**: Resolution/Uncertainty ratio for model value assessment
//! - **ConditionalCalibration**: Metrics sliced by regime and prediction type
//!
//! ## Design Principles
//!
//! 1. **Measurement Before Modeling**: Never build a model without measuring what you're predicting
//! 2. **Calibration is Ground Truth**: A model is only as good as its calibration metrics
//! 3. **Regime Awareness**: All metrics can be sliced by regime for conditional analysis
//! 4. **Thread Safety**: All structs are Send + Sync for concurrent access

pub mod adaptive_binning;
mod brier_score;
mod coefficient_estimator;
mod conditional_metrics;
mod information_ratio;
pub mod model_gating;
mod prediction_log;

pub use adaptive_binning::AdaptiveBinner;
pub use brier_score::BrierScoreTracker;
pub use coefficient_estimator::{CalibrationSample, CoefficientEstimator, CoefficientEstimatorConfig};
pub use conditional_metrics::ConditionalCalibration;
pub use information_ratio::InformationRatioTracker;
pub use model_gating::{InformedFlowAdjustment, ModelGating, ModelGatingConfig, ModelWeights};
pub use prediction_log::{PredictionLog, PredictionRecord, PredictionType};

/// Minimum samples required for reliable calibration metrics.
pub const MIN_SAMPLES_FOR_CALIBRATION: usize = 100;

/// Default number of bins for information ratio calculation.
pub const DEFAULT_IR_BINS: usize = 10;

/// Threshold for considering a model to be adding value (IR > 1.0).
pub const IR_VALUE_THRESHOLD: f64 = 1.0;

/// Check if enough samples exist for reliable calibration.
pub fn has_sufficient_samples(n_samples: usize) -> bool {
    n_samples >= MIN_SAMPLES_FOR_CALIBRATION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_sufficient_samples() {
        assert!(!has_sufficient_samples(50));
        assert!(!has_sufficient_samples(99));
        assert!(has_sufficient_samples(100));
        assert!(has_sufficient_samples(1000));
    }
}
