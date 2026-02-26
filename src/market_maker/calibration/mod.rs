//! Calibration infrastructure for market making model validation.
//!
//! This module provides tools for measuring and validating model calibration:
//!
//! - **PredictionLog**: Thread-safe logging of predictions with outcome resolution
//! - **BrierScoreTracker**: Rolling Brier score for probability calibration
//! - **InformationRatioTracker**: Resolution/Uncertainty ratio for model value assessment
//! - **ConditionalCalibration**: Metrics sliced by regime and prediction type
//! - **ParameterLearner**: Bayesian parameter learning with regularization
//! - **HistoricalCalibrator**: Batch calibration from historical data
//!
//! ## Design Principles
//!
//! 1. **Measurement Before Modeling**: Never build a model without measuring what you're predicting
//! 2. **Calibration is Ground Truth**: A model is only as good as its calibration metrics
//! 3. **Regime Awareness**: All metrics can be sliced by regime for conditional analysis
//! 4. **Thread Safety**: All structs are Send + Sync for concurrent access
//! 5. **Bayesian Regularization**: All parameters use priors to prevent overfitting

pub mod adaptive_binning;
mod brier_score;
mod coefficient_estimator;
mod conditional_metrics;
pub mod derived_constants;
pub mod gate;
pub mod historical_calibrator;
mod information_ratio;
pub mod meta_calibration;
pub mod model_gating;
pub mod parameter_learner;
mod prediction_log;
pub mod signal_decay;

pub use adaptive_binning::AdaptiveBinner;
pub use brier_score::BrierScoreTracker;
pub use coefficient_estimator::{
    CalibrationSample, CoefficientEstimator, CoefficientEstimatorConfig,
};
pub use conditional_metrics::ConditionalCalibration;
pub use derived_constants::{
    derive_cascade_threshold, derive_confidence_threshold, derive_depth_spacing_ratio,
    derive_ewma_alpha, derive_gamma_from_glft, derive_hazard_rate, derive_ir_based_threshold,
    derive_kalman_noise, derive_max_daily_loss, derive_max_drawdown, derive_momentum_normalizer,
    derive_quote_latch_threshold, derive_reduce_only_threshold, derive_spread_floor,
    derive_toxic_hour_multiplier,
};
pub use gate::{CalibrationGate, CalibrationGateConfig, PriorStatus};
pub use historical_calibrator::{
    CalibrationSummary, FillRecord, HistoricalCalibrator, MarketSnapshot, PowerAnalysis,
    TradeRecord,
};
pub use information_ratio::{
    BinStats, ExponentialIRTracker, InformationRatioTracker, IrDiagnostics,
};
pub use meta_calibration::{MetaCalibrationTracker, ModelCalibrationTracker};
pub use model_gating::{InformedFlowAdjustment, ModelGating, ModelGatingConfig, ModelWeights};
pub use parameter_learner::{
    BayesianParam, CalibrationStatus, FillOutcome, LearnedParameters, PriorFamily,
};
pub use prediction_log::{PredictionLog, PredictionRecord, PredictionType};
pub use signal_decay::{
    LatencyStats, SignalDecayConfig, SignalDecayTracker, SignalEmission, SignalOutcome,
};

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
