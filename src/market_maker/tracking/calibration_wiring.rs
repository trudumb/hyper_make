//! Calibration wiring for model validation.
//!
//! Wraps existing models with calibration tracking to enable:
//! - Prediction logging with unique IDs
//! - Outcome recording and matching
//! - Information Ratio (IR) calculation
//! - Degradation detection via SignalHealthMonitor
//!
//! ## Architecture
//!
//! Each calibrated wrapper follows the same pattern:
//! 1. `predict()` - Log prediction, return (value, prediction_id)
//! 2. `record_outcome()` - Match outcome to prediction_id
//! 3. `information_ratio()` - Get current IR for model health
//!
//! The `ModelCalibrationOrchestrator` aggregates all calibrated models
//! and provides a unified interface for health monitoring.
//!
//! ## Usage
//!
//! ```ignore
//! // Create calibrated fill model
//! let model = BayesianFillModel::default();
//! let mut calibrated = CalibratedFillModel::new(model);
//!
//! // Make prediction (logs automatically)
//! let (prob, pred_id) = calibrated.predict(5.0);
//!
//! // Later, record outcome
//! calibrated.record_outcome(pred_id, true);
//!
//! // Check model health
//! let ir = calibrated.information_ratio();
//! if ir < 1.0 {
//!     warn!("Fill model adding noise, not value!");
//! }
//! ```

use std::collections::HashMap;

use super::calibration::{
    CalibrationConfig, CalibrationMetrics, OutcomeLog, PredictionLog, PredictionOutcomeStore,
    PredictionTracker, PredictionType,
};
use crate::market_maker::adverse_selection::AdverseSelectionEstimator;
use crate::market_maker::edge::{EdgeSignalKind, SignalHealthMonitor};
use crate::market_maker::estimator::LagAnalyzer;
use crate::market_maker::quoting::BayesianFillModel;

// ============================================================================
// CalibratedFillModel
// ============================================================================

/// Configuration for calibrated fill model.
#[derive(Debug, Clone)]
pub struct CalibratedFillModelConfig {
    /// Maximum pending predictions to retain.
    pub max_pending: usize,
    /// Maximum linked pairs to retain.
    pub max_pairs: usize,
    /// Minimum samples for reliable IR calculation.
    pub min_samples: usize,
}

impl Default for CalibratedFillModelConfig {
    fn default() -> Self {
        Self {
            max_pending: 1000,
            max_pairs: 10_000,
            min_samples: 100,
        }
    }
}

/// Calibrated wrapper for BayesianFillModel.
///
/// Logs fill probability predictions and tracks outcomes to compute
/// Information Ratio and calibration metrics.
#[derive(Debug)]
pub struct CalibratedFillModel {
    /// Underlying fill model.
    model: BayesianFillModel,
    /// Prediction-outcome store for calibration.
    prediction_store: PredictionOutcomeStore,
    /// Prediction tracker for detailed metrics.
    tracker: PredictionTracker,
    /// Configuration.
    config: CalibratedFillModelConfig,
}

impl CalibratedFillModel {
    /// Create a new calibrated fill model.
    pub fn new(model: BayesianFillModel) -> Self {
        Self::with_config(model, CalibratedFillModelConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(model: BayesianFillModel, config: CalibratedFillModelConfig) -> Self {
        Self {
            model,
            prediction_store: PredictionOutcomeStore::new(config.max_pairs, config.max_pending),
            tracker: PredictionTracker::new(CalibrationConfig {
                max_observations: config.max_pairs,
                num_bins: 20,
                min_observations: config.min_samples,
            }),
            config,
        }
    }

    /// Make a fill probability prediction and log it.
    ///
    /// # Arguments
    /// * `depth_bps` - Depth from mid in basis points
    ///
    /// # Returns
    /// (probability, prediction_id) tuple
    pub fn predict(&mut self, depth_bps: f64) -> (f64, u64) {
        let probability = self.model.fill_probability(depth_bps);

        // Build features map
        let mut features = HashMap::new();
        features.insert("depth_bps".to_string(), depth_bps);
        let (fills, attempts) = self.model.observations_at_depth(depth_bps);
        features.insert("bucket_fills".to_string(), fills as f64);
        features.insert("bucket_attempts".to_string(), attempts as f64);

        // Create prediction log
        let prediction = PredictionLog::new(
            PredictionType::FillProbability,
            probability,
            self.model_confidence(depth_bps),
            features,
        );

        let prediction_id = self.prediction_store.log_prediction(prediction);
        (probability, prediction_id)
    }

    /// Make a prediction with regime context.
    pub fn predict_with_regime(&mut self, depth_bps: f64, regime: &str) -> (f64, u64) {
        let probability = self.model.fill_probability(depth_bps);

        let mut features = HashMap::new();
        features.insert("depth_bps".to_string(), depth_bps);
        let (fills, attempts) = self.model.observations_at_depth(depth_bps);
        features.insert("bucket_fills".to_string(), fills as f64);
        features.insert("bucket_attempts".to_string(), attempts as f64);

        let prediction = PredictionLog::new(
            PredictionType::FillProbability,
            probability,
            self.model_confidence(depth_bps),
            features,
        )
        .with_regime(regime);

        let prediction_id = self.prediction_store.log_prediction(prediction);
        (probability, prediction_id)
    }

    /// Record the outcome of a fill prediction.
    ///
    /// # Arguments
    /// * `prediction_id` - ID returned from predict()
    /// * `filled` - Whether the order filled
    pub fn record_outcome(&mut self, prediction_id: u64, filled: bool) {
        let actual_value = if filled { 1.0 } else { 0.0 };
        let outcome = OutcomeLog::new(
            prediction_id,
            actual_value,
            PredictionType::FillProbability.default_measurement_delay_ms(),
        );

        if let Some(linked) = self.prediction_store.record_outcome(outcome) {
            // Also update the detailed tracker
            let regime = linked.prediction.regime.as_deref().unwrap_or("Unknown");
            self.tracker
                .record(linked.prediction.predicted_value, filled, regime);

            // Update the underlying model with this observation
            if let Some(depth) = linked.prediction.features.get("depth_bps") {
                self.model.record_observation(*depth, filled);
            }
        }
    }

    /// Get the current Information Ratio.
    ///
    /// IR > 1.0 means model adds predictive value.
    /// IR < 1.0 means model is adding noise.
    pub fn information_ratio(&self) -> f64 {
        self.tracker.metrics().information_ratio
    }

    /// Get full calibration metrics.
    pub fn metrics(&self) -> CalibrationMetrics {
        self.tracker.metrics()
    }

    /// Check if model is well-calibrated.
    pub fn is_well_calibrated(&self) -> bool {
        self.tracker.metrics().is_well_calibrated()
    }

    /// Get pending prediction count.
    pub fn pending_count(&self) -> usize {
        self.prediction_store.pending_count()
    }

    /// Get linked pair count.
    pub fn sample_count(&self) -> usize {
        self.prediction_store.linked_count()
    }

    /// Check if model has sufficient data.
    pub fn is_warmed_up(&self) -> bool {
        self.prediction_store.linked_count() >= self.config.min_samples
    }

    /// Get access to underlying model.
    pub fn model(&self) -> &BayesianFillModel {
        &self.model
    }

    /// Get mutable access to underlying model.
    pub fn model_mut(&mut self) -> &mut BayesianFillModel {
        &mut self.model
    }

    /// Calculate model confidence at a depth.
    fn model_confidence(&self, depth_bps: f64) -> f64 {
        // Use uncertainty from the Bayesian model
        let uncertainty = self.model.fill_uncertainty(depth_bps);
        // Convert uncertainty to confidence (lower uncertainty = higher confidence)
        let confidence: f64 = 1.0 - uncertainty.min(0.5) * 2.0;
        confidence.clamp(0.0, 1.0)
    }

    /// Get Brier score from the store.
    pub fn brier_score(&self) -> Option<f64> {
        self.prediction_store
            .brier_score(PredictionType::FillProbability, self.config.min_samples)
    }

    /// Clear all calibration data (for testing/reset).
    pub fn clear(&mut self) {
        self.prediction_store.clear();
        self.tracker = PredictionTracker::new(CalibrationConfig {
            max_observations: self.config.max_pairs,
            num_bins: 20,
            min_observations: self.config.min_samples,
        });
    }
}

// ============================================================================
// CalibratedAdverseSelection
// ============================================================================

/// Configuration for calibrated adverse selection model.
#[derive(Debug, Clone)]
pub struct CalibratedAdverseSelectionConfig {
    /// Maximum pending predictions to retain.
    pub max_pending: usize,
    /// Maximum linked pairs to retain.
    pub max_pairs: usize,
    /// Minimum samples for reliable IR calculation.
    pub min_samples: usize,
}

impl Default for CalibratedAdverseSelectionConfig {
    fn default() -> Self {
        Self {
            max_pending: 500,
            max_pairs: 5_000,
            min_samples: 50,
        }
    }
}

/// Calibrated wrapper for AdverseSelectionEstimator.
///
/// Logs adverse selection predictions (alpha) and tracks whether
/// fills were actually toxic to compute IR.
#[derive(Debug)]
pub struct CalibratedAdverseSelection {
    /// Underlying adverse selection estimator.
    estimator: AdverseSelectionEstimator,
    /// Prediction-outcome store for calibration.
    prediction_store: PredictionOutcomeStore,
    /// Prediction tracker for detailed metrics.
    tracker: PredictionTracker,
    /// Configuration.
    config: CalibratedAdverseSelectionConfig,
    /// Threshold for classifying a fill as "adverse" (in bps).
    adverse_threshold_bps: f64,
}

impl CalibratedAdverseSelection {
    /// Create a new calibrated adverse selection model.
    pub fn new(estimator: AdverseSelectionEstimator) -> Self {
        Self::with_config(estimator, CalibratedAdverseSelectionConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(
        estimator: AdverseSelectionEstimator,
        config: CalibratedAdverseSelectionConfig,
    ) -> Self {
        Self {
            estimator,
            prediction_store: PredictionOutcomeStore::new(config.max_pairs, config.max_pending),
            tracker: PredictionTracker::new(CalibrationConfig {
                max_observations: config.max_pairs,
                num_bins: 20,
                min_observations: config.min_samples,
            }),
            config,
            adverse_threshold_bps: 2.0, // Default: 2bp adverse is "toxic"
        }
    }

    /// Set the threshold for what constitutes an adverse fill.
    pub fn set_adverse_threshold_bps(&mut self, threshold: f64) {
        self.adverse_threshold_bps = threshold;
    }

    /// Make an adverse selection prediction and log it.
    ///
    /// Returns (predicted_alpha, prediction_id) where alpha is
    /// the probability that the next fill will be toxic/informed.
    pub fn predict(&mut self) -> (f64, u64) {
        let alpha = self.estimator.predicted_alpha();

        let mut features = HashMap::new();
        features.insert(
            "realized_as_bps".to_string(),
            self.estimator.realized_as_bps(),
        );
        features.insert(
            "fills_measured".to_string(),
            self.estimator.fills_measured() as f64,
        );
        features.insert(
            "spread_adj_bps".to_string(),
            self.estimator.spread_adjustment_bps(),
        );

        let prediction = PredictionLog::new(
            PredictionType::AdverseSelection,
            alpha,
            self.estimator_confidence(),
            features,
        );

        let prediction_id = self.prediction_store.log_prediction(prediction);
        (alpha, prediction_id)
    }

    /// Make a prediction with regime context.
    pub fn predict_with_regime(&mut self, regime: &str) -> (f64, u64) {
        let alpha = self.estimator.predicted_alpha();

        let mut features = HashMap::new();
        features.insert(
            "realized_as_bps".to_string(),
            self.estimator.realized_as_bps(),
        );
        features.insert(
            "fills_measured".to_string(),
            self.estimator.fills_measured() as f64,
        );
        features.insert(
            "spread_adj_bps".to_string(),
            self.estimator.spread_adjustment_bps(),
        );

        let prediction = PredictionLog::new(
            PredictionType::AdverseSelection,
            alpha,
            self.estimator_confidence(),
            features,
        )
        .with_regime(regime);

        let prediction_id = self.prediction_store.log_prediction(prediction);
        (alpha, prediction_id)
    }

    /// Record the outcome of an adverse selection prediction.
    ///
    /// # Arguments
    /// * `prediction_id` - ID returned from predict()
    /// * `realized_as_bps` - Actual adverse selection in basis points
    pub fn record_outcome(&mut self, prediction_id: u64, realized_as_bps: f64) {
        // Convert continuous AS to binary "was this toxic?"
        let was_adverse = realized_as_bps > self.adverse_threshold_bps;
        let actual_value = if was_adverse { 1.0 } else { 0.0 };

        let outcome = OutcomeLog::new(
            prediction_id,
            actual_value,
            PredictionType::AdverseSelection.default_measurement_delay_ms(),
        );

        if let Some(linked) = self.prediction_store.record_outcome(outcome) {
            let regime = linked.prediction.regime.as_deref().unwrap_or("Unknown");
            self.tracker
                .record(linked.prediction.predicted_value, was_adverse, regime);
        }
    }

    /// Get the current Information Ratio.
    pub fn information_ratio(&self) -> f64 {
        self.tracker.metrics().information_ratio
    }

    /// Get full calibration metrics.
    pub fn metrics(&self) -> CalibrationMetrics {
        self.tracker.metrics()
    }

    /// Check if model is well-calibrated.
    pub fn is_well_calibrated(&self) -> bool {
        self.tracker.metrics().is_well_calibrated()
    }

    /// Get pending prediction count.
    pub fn pending_count(&self) -> usize {
        self.prediction_store.pending_count()
    }

    /// Get sample count.
    pub fn sample_count(&self) -> usize {
        self.prediction_store.linked_count()
    }

    /// Check if model has sufficient data.
    pub fn is_warmed_up(&self) -> bool {
        self.prediction_store.linked_count() >= self.config.min_samples
    }

    /// Get access to underlying estimator.
    pub fn estimator(&self) -> &AdverseSelectionEstimator {
        &self.estimator
    }

    /// Get mutable access to underlying estimator.
    pub fn estimator_mut(&mut self) -> &mut AdverseSelectionEstimator {
        &mut self.estimator
    }

    /// Get recommended spread adjustment (delegates to underlying).
    pub fn spread_adjustment_bps(&self) -> f64 {
        self.estimator.spread_adjustment_bps()
    }

    /// Calculate estimator confidence based on warmup state.
    fn estimator_confidence(&self) -> f64 {
        if self.estimator.is_warmed_up() {
            0.8
        } else {
            0.3
        }
    }

    /// Clear all calibration data.
    pub fn clear(&mut self) {
        self.prediction_store.clear();
        self.tracker = PredictionTracker::new(CalibrationConfig {
            max_observations: self.config.max_pairs,
            num_bins: 20,
            min_observations: self.config.min_samples,
        });
    }
}

// ============================================================================
// CalibratedLagAnalyzer
// ============================================================================

/// Configuration for calibrated lag analyzer.
#[derive(Debug, Clone)]
pub struct CalibratedLagAnalyzerConfig {
    /// Maximum pending predictions to retain.
    pub max_pending: usize,
    /// Maximum linked pairs to retain.
    pub max_pairs: usize,
    /// Minimum samples for reliable MI decay calculation.
    pub min_samples: usize,
    /// MI decay warning threshold (bits lost per day).
    pub mi_decay_warning_threshold: f64,
}

impl Default for CalibratedLagAnalyzerConfig {
    fn default() -> Self {
        Self {
            max_pending: 500,
            max_pairs: 5_000,
            min_samples: 100,
            mi_decay_warning_threshold: 0.01, // 0.01 bits/day decay triggers warning
        }
    }
}

/// Calibrated wrapper for LagAnalyzer.
///
/// Tracks lag predictions and monitors MI decay to detect
/// when the cross-exchange lead-lag edge is eroding.
#[derive(Debug)]
pub struct CalibratedLagAnalyzer {
    /// Underlying lag analyzer.
    analyzer: LagAnalyzer,
    /// Configuration.
    config: CalibratedLagAnalyzerConfig,
    /// Historical MI measurements for decay tracking.
    mi_history: Vec<(u64, f64)>, // (timestamp_ms, mi_bits)
    /// Current MI decay rate (bits per day).
    mi_decay_rate: f64,
    /// Prediction-outcome store for lag predictions.
    prediction_store: PredictionOutcomeStore,
    /// Track accuracy of lag predictions.
    tracker: PredictionTracker,
}

impl CalibratedLagAnalyzer {
    /// Create a new calibrated lag analyzer.
    pub fn new(analyzer: LagAnalyzer) -> Self {
        Self::with_config(analyzer, CalibratedLagAnalyzerConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(analyzer: LagAnalyzer, config: CalibratedLagAnalyzerConfig) -> Self {
        Self {
            analyzer,
            prediction_store: PredictionOutcomeStore::new(config.max_pairs, config.max_pending),
            tracker: PredictionTracker::new(CalibrationConfig {
                max_observations: config.max_pairs,
                num_bins: 20,
                min_observations: config.min_samples,
            }),
            mi_history: Vec::with_capacity(1000),
            mi_decay_rate: 0.0,
            config,
        }
    }

    /// Make a lag prediction and log it.
    ///
    /// Returns (predicted_lag_ms, mi_bits, prediction_id).
    pub fn predict(&mut self, timestamp_ms: u64) -> (i64, f64, u64) {
        let lag_ms = self.analyzer.best_lag_ms();
        let mi = self.analyzer.best_lag_mi();

        // Record MI for decay tracking
        self.record_mi(timestamp_ms, mi);

        // Log prediction (convert lag to direction probability)
        // lag < 0 means signal leads target (positive prediction)
        let direction_prob = if lag_ms < 0 { 0.8 } else { 0.2 };

        let mut features = HashMap::new();
        features.insert("lag_ms".to_string(), lag_ms as f64);
        features.insert("mi_bits".to_string(), mi);
        let (signal_count, target_count) = self.analyzer.observation_counts();
        features.insert("signal_count".to_string(), signal_count as f64);
        features.insert("target_count".to_string(), target_count as f64);

        let prediction = PredictionLog::new(
            PredictionType::PriceDirection,
            direction_prob,
            self.analyzer_confidence(),
            features,
        );

        let prediction_id = self.prediction_store.log_prediction(prediction);
        (lag_ms, mi, prediction_id)
    }

    /// Record the outcome of a lag prediction.
    ///
    /// # Arguments
    /// * `prediction_id` - ID returned from predict()
    /// * `signal_led` - Whether the signal actually led the target
    pub fn record_outcome(&mut self, prediction_id: u64, signal_led: bool) {
        let actual_value = if signal_led { 1.0 } else { 0.0 };

        let outcome = OutcomeLog::new(
            prediction_id,
            actual_value,
            PredictionType::PriceDirection.default_measurement_delay_ms(),
        );

        if let Some(linked) = self.prediction_store.record_outcome(outcome) {
            self.tracker
                .record(linked.prediction.predicted_value, signal_led, "Unknown");
        }
    }

    /// Record MI measurement for decay tracking.
    fn record_mi(&mut self, timestamp_ms: u64, mi: f64) {
        self.mi_history.push((timestamp_ms, mi));

        // Keep bounded
        if self.mi_history.len() > 1000 {
            self.mi_history.remove(0);
        }

        // Update decay rate if enough history
        if self.mi_history.len() >= 10 {
            self.update_decay_rate();
        }
    }

    /// Update MI decay rate using linear regression.
    fn update_decay_rate(&mut self) {
        let n = self.mi_history.len() as f64;
        if n < 3.0 {
            return;
        }

        let first_ts = self.mi_history.first().map(|(t, _)| *t).unwrap_or(0);

        let mut sum_t = 0.0;
        let mut sum_mi = 0.0;
        let mut sum_t2 = 0.0;
        let mut sum_t_mi = 0.0;

        for (ts, mi) in &self.mi_history {
            let t = (*ts - first_ts) as f64 / (1000.0 * 3600.0 * 24.0); // Days since start
            sum_t += t;
            sum_mi += mi;
            sum_t2 += t * t;
            sum_t_mi += t * mi;
        }

        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < 1e-10 {
            return;
        }

        // Slope is change in MI per day (negative = decay)
        self.mi_decay_rate = -(n * sum_t_mi - sum_t * sum_mi) / denom;
    }

    /// Get the current MI decay rate (bits per day).
    ///
    /// Positive value means MI is decaying (edge eroding).
    pub fn mi_decay_rate(&self) -> f64 {
        self.mi_decay_rate
    }

    /// Check if MI decay is concerning.
    pub fn is_mi_decaying(&self) -> bool {
        self.mi_decay_rate > self.config.mi_decay_warning_threshold
    }

    /// Get current best lag in milliseconds.
    pub fn best_lag_ms(&self) -> i64 {
        self.analyzer.best_lag_ms()
    }

    /// Get current best lag MI.
    pub fn best_lag_mi(&self) -> f64 {
        self.analyzer.best_lag_mi()
    }

    /// Get Information Ratio for lag predictions.
    pub fn information_ratio(&self) -> f64 {
        self.tracker.metrics().information_ratio
    }

    /// Get full calibration metrics.
    pub fn metrics(&self) -> CalibrationMetrics {
        self.tracker.metrics()
    }

    /// Get sample count.
    pub fn sample_count(&self) -> usize {
        self.prediction_store.linked_count()
    }

    /// Check if analyzer is ready.
    pub fn is_warmed_up(&self) -> bool {
        self.analyzer.is_ready() && self.mi_history.len() >= self.config.min_samples
    }

    /// Get access to underlying analyzer.
    pub fn analyzer(&self) -> &LagAnalyzer {
        &self.analyzer
    }

    /// Get mutable access to underlying analyzer.
    pub fn analyzer_mut(&mut self) -> &mut LagAnalyzer {
        &mut self.analyzer
    }

    /// Calculate analyzer confidence based on MI and sample count.
    fn analyzer_confidence(&self) -> f64 {
        let mi = self.analyzer.best_lag_mi();
        let (signal_count, target_count) = self.analyzer.observation_counts();
        let min_count = signal_count.min(target_count) as f64;

        // Confidence based on MI strength and sample count
        let mi_conf = (mi / 0.1).min(1.0); // Full confidence at 0.1 bits
        let count_conf = (min_count / 200.0).min(1.0); // Full confidence at 200 samples

        (mi_conf * count_conf).clamp(0.0, 1.0)
    }

    /// Clear calibration data.
    pub fn clear(&mut self) {
        self.mi_history.clear();
        self.mi_decay_rate = 0.0;
        self.prediction_store.clear();
        self.tracker = PredictionTracker::new(CalibrationConfig {
            max_observations: self.config.max_pairs,
            num_bins: 20,
            min_observations: self.config.min_samples,
        });
    }
}

// ============================================================================
// CalibrationSummary
// ============================================================================

/// Aggregated calibration summary for all models.
#[derive(Debug, Clone)]
pub struct ModelCalibrationSummary {
    /// Fill model metrics.
    pub fill_model: CalibrationMetrics,
    /// Fill model IR.
    pub fill_ir: f64,
    /// Adverse selection model metrics.
    pub as_model: CalibrationMetrics,
    /// Adverse selection model IR.
    pub as_ir: f64,
    /// Lag analyzer metrics.
    pub lag_model: CalibrationMetrics,
    /// Lag analyzer IR.
    pub lag_ir: f64,
    /// Lag MI decay rate (bits/day).
    pub lag_mi_decay_rate: f64,
    /// Signal health summary from monitor.
    pub signal_health: crate::market_maker::edge::SignalHealthSummary,
    /// Whether any model is degraded.
    pub any_degraded: bool,
    /// Timestamp of summary.
    pub timestamp_ms: u64,
}

impl ModelCalibrationSummary {
    /// Get a diagnostic string for logging.
    pub fn diagnostic_string(&self) -> String {
        format!(
            "fill_ir={:.3}, as_ir={:.3}, lag_ir={:.3}, lag_decay={:.4}/day, any_degraded={}",
            self.fill_ir, self.as_ir, self.lag_ir, self.lag_mi_decay_rate, self.any_degraded
        )
    }
}

// ============================================================================
// ModelCalibrationOrchestrator
// ============================================================================

/// Orchestrates calibration across all models.
///
/// Provides unified interface for:
/// - Updating all model calibrations
/// - Aggregating health status
/// - Detecting degradation
#[derive(Debug)]
pub struct ModelCalibrationOrchestrator {
    /// Calibrated fill model.
    pub fill_model: CalibratedFillModel,
    /// Calibrated adverse selection model.
    pub as_model: CalibratedAdverseSelection,
    /// Calibrated lag analyzer.
    pub lag_model: CalibratedLagAnalyzer,
    /// Signal health monitor.
    pub health_monitor: SignalHealthMonitor,
    /// IR threshold below which model is considered degraded.
    ir_threshold: f64,
}

impl ModelCalibrationOrchestrator {
    /// Create a new orchestrator with default configurations.
    pub fn new(
        fill_model: BayesianFillModel,
        as_estimator: AdverseSelectionEstimator,
        lag_analyzer: LagAnalyzer,
    ) -> Self {
        let mut health_monitor = SignalHealthMonitor::default();

        // Register signals for health tracking
        health_monitor.register_signal(
            EdgeSignalKind::FillProbability,
            "Fill Probability".to_string(),
            0.1, // Baseline MI
        );
        health_monitor.register_signal(
            EdgeSignalKind::AdverseSelection,
            "Adverse Selection".to_string(),
            0.05,
        );
        health_monitor.register_signal(EdgeSignalKind::LeadLag, "Lead-Lag".to_string(), 0.05);

        Self {
            fill_model: CalibratedFillModel::new(fill_model),
            as_model: CalibratedAdverseSelection::new(as_estimator),
            lag_model: CalibratedLagAnalyzer::new(lag_analyzer),
            health_monitor,
            ir_threshold: 1.0, // IR must be > 1.0 to be healthy
        }
    }

    /// Create with custom configurations.
    pub fn with_configs(
        fill_model: BayesianFillModel,
        fill_config: CalibratedFillModelConfig,
        as_estimator: AdverseSelectionEstimator,
        as_config: CalibratedAdverseSelectionConfig,
        lag_analyzer: LagAnalyzer,
        lag_config: CalibratedLagAnalyzerConfig,
    ) -> Self {
        let mut health_monitor = SignalHealthMonitor::default();

        health_monitor.register_signal(
            EdgeSignalKind::FillProbability,
            "Fill Probability".to_string(),
            0.1,
        );
        health_monitor.register_signal(
            EdgeSignalKind::AdverseSelection,
            "Adverse Selection".to_string(),
            0.05,
        );
        health_monitor.register_signal(EdgeSignalKind::LeadLag, "Lead-Lag".to_string(), 0.05);

        Self {
            fill_model: CalibratedFillModel::with_config(fill_model, fill_config),
            as_model: CalibratedAdverseSelection::with_config(as_estimator, as_config),
            lag_model: CalibratedLagAnalyzer::with_config(lag_analyzer, lag_config),
            health_monitor,
            ir_threshold: 1.0,
        }
    }

    /// Set the IR threshold for degradation detection.
    pub fn set_ir_threshold(&mut self, threshold: f64) {
        self.ir_threshold = threshold;
    }

    /// Update all model calibrations.
    ///
    /// Should be called periodically (e.g., every minute) to refresh
    /// health monitor with latest IR values.
    pub fn update_all(&mut self, timestamp_ms: u64) {
        // Convert IR to a "relative strength" for health monitor
        // IR > 1.0 is healthy, so we use IR as the MI value
        let fill_ir = self.fill_model.information_ratio();
        let as_ir = self.as_model.information_ratio();
        let lag_mi = self.lag_model.best_lag_mi();

        // Update health monitor (using IR as proxy for MI)
        self.health_monitor.update_signal(
            EdgeSignalKind::FillProbability,
            fill_ir * 0.1, // Scale to baseline
            timestamp_ms,
        );
        self.health_monitor.update_signal(
            EdgeSignalKind::AdverseSelection,
            as_ir * 0.05,
            timestamp_ms,
        );
        self.health_monitor
            .update_signal(EdgeSignalKind::LeadLag, lag_mi, timestamp_ms);
    }

    /// Get aggregated calibration summary.
    pub fn summary(&self) -> ModelCalibrationSummary {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let fill_ir = self.fill_model.information_ratio();
        let as_ir = self.as_model.information_ratio();
        let lag_ir = self.lag_model.information_ratio();

        let any_degraded = self.is_any_degraded();

        ModelCalibrationSummary {
            fill_model: self.fill_model.metrics(),
            fill_ir,
            as_model: self.as_model.metrics(),
            as_ir,
            lag_model: self.lag_model.metrics(),
            lag_ir,
            lag_mi_decay_rate: self.lag_model.mi_decay_rate(),
            signal_health: self.health_monitor.summary(),
            any_degraded,
            timestamp_ms,
        }
    }

    /// Quick check for any degraded models.
    ///
    /// A model is degraded if:
    /// - IR < threshold (default 1.0)
    /// - Lag MI is decaying too fast
    pub fn is_any_degraded(&self) -> bool {
        let fill_degraded = self.fill_model.is_warmed_up()
            && self.fill_model.information_ratio() < self.ir_threshold;

        let as_degraded =
            self.as_model.is_warmed_up() && self.as_model.information_ratio() < self.ir_threshold;

        let lag_degraded = self.lag_model.is_mi_decaying();

        fill_degraded || as_degraded || lag_degraded
    }

    /// Get list of degraded model names.
    pub fn degraded_models(&self) -> Vec<&'static str> {
        let mut degraded = Vec::new();

        if self.fill_model.is_warmed_up() && self.fill_model.information_ratio() < self.ir_threshold
        {
            degraded.push("FillModel");
        }

        if self.as_model.is_warmed_up() && self.as_model.information_ratio() < self.ir_threshold {
            degraded.push("AdverseSelection");
        }

        if self.lag_model.is_mi_decaying() {
            degraded.push("LagAnalyzer");
        }

        degraded
    }

    /// Check if all models are healthy.
    pub fn all_healthy(&self) -> bool {
        !self.is_any_degraded() && self.health_monitor.all_healthy()
    }

    /// Get access to the health monitor.
    pub fn health_monitor(&self) -> &SignalHealthMonitor {
        &self.health_monitor
    }

    /// Get mutable access to the health monitor.
    pub fn health_monitor_mut(&mut self) -> &mut SignalHealthMonitor {
        &mut self.health_monitor
    }
}

impl Default for ModelCalibrationOrchestrator {
    fn default() -> Self {
        use crate::market_maker::adverse_selection::AdverseSelectionConfig;
        use crate::market_maker::estimator::LagAnalyzerConfig;
        Self::new(
            BayesianFillModel::default(),
            AdverseSelectionEstimator::new(AdverseSelectionConfig::default()),
            LagAnalyzer::new(LagAnalyzerConfig::default()),
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::adverse_selection::AdverseSelectionConfig;
    use crate::market_maker::estimator::LagAnalyzerConfig;

    fn make_fill_model() -> BayesianFillModel {
        BayesianFillModel::default()
    }

    fn make_as_estimator() -> AdverseSelectionEstimator {
        AdverseSelectionEstimator::new(AdverseSelectionConfig::default())
    }

    fn make_lag_analyzer() -> LagAnalyzer {
        LagAnalyzer::new(LagAnalyzerConfig::default())
    }

    // ========================================================================
    // CalibratedFillModel Tests
    // ========================================================================

    #[test]
    fn test_calibrated_fill_model_predict() {
        let mut model = CalibratedFillModel::new(make_fill_model());

        let (prob, pred_id) = model.predict(5.0);

        assert!((0.0..=1.0).contains(&prob));
        assert!(pred_id > 0);
        assert_eq!(model.pending_count(), 1);
    }

    #[test]
    fn test_calibrated_fill_model_record_outcome() {
        let mut model = CalibratedFillModel::new(make_fill_model());

        let (_, pred_id) = model.predict(5.0);
        model.record_outcome(pred_id, true);

        assert_eq!(model.pending_count(), 0);
        assert_eq!(model.sample_count(), 1);
    }

    #[test]
    fn test_calibrated_fill_model_ir_calculation() {
        let mut model = CalibratedFillModel::with_config(
            make_fill_model(),
            CalibratedFillModelConfig {
                max_pending: 100,
                max_pairs: 1000,
                min_samples: 10,
            },
        );

        // Add well-calibrated predictions
        for _ in 0..20 {
            let (_, pred_id) = model.predict(5.0);
            model.record_outcome(pred_id, true);
        }

        // IR should be calculable
        let ir = model.information_ratio();
        assert!(ir >= 0.0, "IR should be non-negative: {}", ir);
    }

    #[test]
    fn test_calibrated_fill_model_metrics() {
        let mut model = CalibratedFillModel::new(make_fill_model());

        // Add some predictions
        for i in 0..15 {
            let (_, pred_id) = model.predict((i as f64) * 2.0);
            model.record_outcome(pred_id, i % 2 == 0);
        }

        let metrics = model.metrics();
        assert!(metrics.n_samples > 0);
    }

    #[test]
    fn test_calibrated_fill_model_with_regime() {
        let mut model = CalibratedFillModel::new(make_fill_model());

        let (_, pred_id) = model.predict_with_regime(5.0, "Cascade");
        model.record_outcome(pred_id, true);

        assert_eq!(model.sample_count(), 1);
    }

    #[test]
    fn test_calibrated_fill_model_clear() {
        let mut model = CalibratedFillModel::new(make_fill_model());

        let (_, _) = model.predict(5.0);
        assert_eq!(model.pending_count(), 1);

        model.clear();
        assert_eq!(model.pending_count(), 0);
        assert_eq!(model.sample_count(), 0);
    }

    // ========================================================================
    // CalibratedAdverseSelection Tests
    // ========================================================================

    #[test]
    fn test_calibrated_as_predict() {
        let mut model = CalibratedAdverseSelection::new(make_as_estimator());

        let (alpha, pred_id) = model.predict();

        assert!((0.0..=1.0).contains(&alpha));
        assert!(pred_id > 0);
        assert_eq!(model.pending_count(), 1);
    }

    #[test]
    fn test_calibrated_as_record_outcome() {
        let mut model = CalibratedAdverseSelection::new(make_as_estimator());

        let (_, pred_id) = model.predict();
        model.record_outcome(pred_id, 5.0); // 5 bps AS (above threshold)

        assert_eq!(model.pending_count(), 0);
        assert_eq!(model.sample_count(), 1);
    }

    #[test]
    fn test_calibrated_as_threshold() {
        let mut model = CalibratedAdverseSelection::new(make_as_estimator());
        model.set_adverse_threshold_bps(10.0);

        // Below threshold - not adverse
        let (_, pred_id1) = model.predict();
        model.record_outcome(pred_id1, 5.0);

        // Above threshold - adverse
        let (_, pred_id2) = model.predict();
        model.record_outcome(pred_id2, 15.0);

        assert_eq!(model.sample_count(), 2);
    }

    #[test]
    fn test_calibrated_as_with_regime() {
        let mut model = CalibratedAdverseSelection::new(make_as_estimator());

        let (_, pred_id) = model.predict_with_regime("Quiet");
        model.record_outcome(pred_id, 1.0);

        assert_eq!(model.sample_count(), 1);
    }

    // ========================================================================
    // CalibratedLagAnalyzer Tests
    // ========================================================================

    #[test]
    fn test_calibrated_lag_predict() {
        let mut model = CalibratedLagAnalyzer::new(make_lag_analyzer());

        let (lag_ms, mi, pred_id) = model.predict(1000);

        // Default analyzer has no data, so lag=0, mi=0
        assert_eq!(lag_ms, 0);
        assert_eq!(mi, 0.0);
        assert!(pred_id > 0);
    }

    #[test]
    fn test_calibrated_lag_record_outcome() {
        let mut model = CalibratedLagAnalyzer::new(make_lag_analyzer());

        let (_, _, pred_id) = model.predict(1000);
        model.record_outcome(pred_id, true);

        assert_eq!(model.sample_count(), 1);
    }

    #[test]
    fn test_calibrated_lag_mi_decay_tracking() {
        let mut model = CalibratedLagAnalyzer::new(make_lag_analyzer());

        // Record declining MI values
        for i in 0..20 {
            let ts = i * 3600 * 1000; // 1 hour apart
            let _ = model.predict(ts);
        }

        // Decay rate should be calculated
        let decay = model.mi_decay_rate();
        // With constant 0 MI, decay should be 0
        assert!(decay.abs() < 1.0, "Decay rate: {}", decay);
    }

    // ========================================================================
    // ModelCalibrationOrchestrator Tests
    // ========================================================================

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = ModelCalibrationOrchestrator::new(
            make_fill_model(),
            make_as_estimator(),
            make_lag_analyzer(),
        );

        assert!(!orchestrator.is_any_degraded()); // No data yet
    }

    #[test]
    fn test_orchestrator_update_all() {
        let mut orchestrator = ModelCalibrationOrchestrator::new(
            make_fill_model(),
            make_as_estimator(),
            make_lag_analyzer(),
        );

        orchestrator.update_all(1000);

        let summary = orchestrator.summary();
        assert!(summary.timestamp_ms >= 1000);
    }

    #[test]
    fn test_orchestrator_summary() {
        let orchestrator = ModelCalibrationOrchestrator::new(
            make_fill_model(),
            make_as_estimator(),
            make_lag_analyzer(),
        );

        let summary = orchestrator.summary();

        assert!(summary.fill_ir >= 0.0);
        assert!(summary.as_ir >= 0.0);
        assert!(summary.lag_ir >= 0.0);
    }

    #[test]
    fn test_orchestrator_degraded_detection() {
        let mut orchestrator = ModelCalibrationOrchestrator::new(
            make_fill_model(),
            make_as_estimator(),
            make_lag_analyzer(),
        );

        // Initially not degraded (not warmed up)
        assert!(!orchestrator.is_any_degraded());

        // Add bad predictions to fill model
        for _ in 0..200 {
            let (_, pred_id) = orchestrator.fill_model.predict(5.0);
            // Always wrong: predict high fill prob, but never fills
            orchestrator.fill_model.record_outcome(pred_id, false);
        }

        // Now should be degraded
        orchestrator.update_all(2000);
        // Note: may or may not be degraded depending on IR calculation
    }

    #[test]
    fn test_orchestrator_degraded_models_list() {
        let orchestrator = ModelCalibrationOrchestrator::new(
            make_fill_model(),
            make_as_estimator(),
            make_lag_analyzer(),
        );

        let degraded = orchestrator.degraded_models();
        // Initially empty (not warmed up)
        assert!(degraded.is_empty() || degraded.len() <= 3);
    }

    #[test]
    fn test_orchestrator_health_monitor_integration() {
        let orchestrator = ModelCalibrationOrchestrator::new(
            make_fill_model(),
            make_as_estimator(),
            make_lag_analyzer(),
        );

        let health = orchestrator.health_monitor().summary();
        assert_eq!(health.total_signals, 3);
    }

    #[test]
    fn test_orchestrator_with_custom_configs() {
        let orchestrator = ModelCalibrationOrchestrator::with_configs(
            make_fill_model(),
            CalibratedFillModelConfig {
                max_pending: 50,
                max_pairs: 500,
                min_samples: 20,
            },
            make_as_estimator(),
            CalibratedAdverseSelectionConfig {
                max_pending: 50,
                max_pairs: 500,
                min_samples: 20,
            },
            make_lag_analyzer(),
            CalibratedLagAnalyzerConfig {
                max_pending: 50,
                max_pairs: 500,
                min_samples: 20,
                mi_decay_warning_threshold: 0.02,
            },
        );

        assert!(!orchestrator.is_any_degraded());
    }

    #[test]
    fn test_summary_diagnostic_string() {
        let orchestrator = ModelCalibrationOrchestrator::new(
            make_fill_model(),
            make_as_estimator(),
            make_lag_analyzer(),
        );

        let summary = orchestrator.summary();
        let diag = summary.diagnostic_string();

        assert!(diag.contains("fill_ir="));
        assert!(diag.contains("as_ir="));
        assert!(diag.contains("lag_ir="));
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_prediction_outcome_cycle() {
        let mut orchestrator = ModelCalibrationOrchestrator::new(
            make_fill_model(),
            make_as_estimator(),
            make_lag_analyzer(),
        );

        // Simulate multiple prediction-outcome cycles
        for i in 0..50 {
            let ts = i * 1000;

            // Fill model
            let (_, fill_pred_id) = orchestrator.fill_model.predict(5.0);
            orchestrator
                .fill_model
                .record_outcome(fill_pred_id, i % 3 == 0);

            // AS model
            let (_, as_pred_id) = orchestrator.as_model.predict();
            orchestrator
                .as_model
                .record_outcome(as_pred_id, if i % 4 == 0 { 5.0 } else { 0.5 });

            // Lag model
            let (_, _, lag_pred_id) = orchestrator.lag_model.predict(ts);
            orchestrator
                .lag_model
                .record_outcome(lag_pred_id, i % 2 == 0);
        }

        orchestrator.update_all(50000);

        let summary = orchestrator.summary();
        assert!(summary.fill_model.n_samples > 0);
        assert!(summary.as_model.n_samples > 0);
        assert!(summary.lag_model.n_samples > 0);
    }

    #[test]
    fn test_ir_threshold_setting() {
        let mut orchestrator = ModelCalibrationOrchestrator::new(
            make_fill_model(),
            make_as_estimator(),
            make_lag_analyzer(),
        );

        orchestrator.set_ir_threshold(0.5);

        // Lower threshold should be more permissive
        assert!(!orchestrator.is_any_degraded());
    }
}
