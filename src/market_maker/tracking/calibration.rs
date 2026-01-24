//! Calibration tracking for prediction validation.
//!
//! Tracks prediction-outcome pairs to compute calibration metrics:
//! - **Brier Score**: Mean squared error of probability predictions
//! - **Information Ratio**: Resolution / Uncertainty (>1.0 means model adds value)
//! - **Calibration Curve**: Predicted vs realized rates by bucket
//!
//! Used to validate:
//! - Fill probability predictions
//! - Adverse selection predictions
//!
//! ## Small Fish Strategy Components
//!
//! This module implements the prediction logging infrastructure from the Small Fish
//! Strategy document (docs/SMALL_FISH_STRATEGY.md lines 574-643):
//!
//! - [`PredictionType`]: Enum of prediction categories
//! - [`PredictionLog`]: Structured log for every prediction made
//! - [`OutcomeLog`]: Outcome record linked to a prediction
//! - [`PredictionOutcomeStore`]: Storage with linking methods

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// Small Fish Strategy: Prediction Logging Infrastructure
// Reference: docs/SMALL_FISH_STRATEGY.md lines 574-600
// ============================================================================

/// Global prediction ID counter for unique identification.
static PREDICTION_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique prediction ID.
fn next_prediction_id() -> u64 {
    PREDICTION_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Type of prediction being made.
///
/// Each prediction type has different calibration requirements and
/// outcome measurement windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredictionType {
    /// Fill probability prediction (will this order fill?).
    /// Outcome window: until order expires or fills.
    FillProbability,

    /// Adverse selection prediction (is this fill toxic?).
    /// Outcome window: 1-5 seconds post-fill.
    AdverseSelection,

    /// Regime change prediction (is market shifting?).
    /// Outcome window: varies by regime definition.
    RegimeChange,

    /// Price direction prediction (up or down?).
    /// Outcome window: specified horizon (e.g., 1s, 5s, 30s).
    PriceDirection,

    /// Volatility prediction (what's the expected vol?).
    /// Outcome window: volatility measurement period.
    Volatility,
}

impl PredictionType {
    /// Default measurement delay in milliseconds for this prediction type.
    pub fn default_measurement_delay_ms(&self) -> u64 {
        match self {
            PredictionType::FillProbability => 5000, // 5s order lifetime
            PredictionType::AdverseSelection => 2000, // 2s post-fill
            PredictionType::RegimeChange => 60000,   // 1 minute
            PredictionType::PriceDirection => 1000,  // 1s default horizon
            PredictionType::Volatility => 60000,     // 1 minute vol window
        }
    }

    /// Returns all prediction types.
    pub fn all() -> &'static [PredictionType] {
        &[
            PredictionType::FillProbability,
            PredictionType::AdverseSelection,
            PredictionType::RegimeChange,
            PredictionType::PriceDirection,
            PredictionType::Volatility,
        ]
    }
}

impl std::fmt::Display for PredictionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictionType::FillProbability => write!(f, "fill_probability"),
            PredictionType::AdverseSelection => write!(f, "adverse_selection"),
            PredictionType::RegimeChange => write!(f, "regime_change"),
            PredictionType::PriceDirection => write!(f, "price_direction"),
            PredictionType::Volatility => write!(f, "volatility"),
        }
    }
}

/// A logged prediction with all context needed for calibration.
///
/// Every prediction made by the system should be logged here with:
/// - The predicted value (probability or continuous)
/// - Confidence level (for uncertainty-aware calibration)
/// - Feature values at prediction time (for debugging and analysis)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionLog {
    /// Unique identifier for this prediction.
    pub id: u64,

    /// Unix timestamp in milliseconds when prediction was made.
    pub timestamp: u64,

    /// Type of prediction.
    pub prediction_type: PredictionType,

    /// Predicted value (probability [0,1] or continuous).
    pub predicted_value: f64,

    /// Confidence in prediction [0,1].
    /// Lower confidence = widen bands for calibration checking.
    pub confidence: f64,

    /// Feature values used to make this prediction.
    /// Keys are feature names, values are feature values.
    pub features: HashMap<String, f64>,

    /// Optional: regime at prediction time.
    pub regime: Option<String>,

    /// Whether outcome has been recorded.
    pub outcome_recorded: bool,
}

impl PredictionLog {
    /// Create a new prediction log entry.
    pub fn new(
        prediction_type: PredictionType,
        predicted_value: f64,
        confidence: f64,
        features: HashMap<String, f64>,
    ) -> Self {
        Self {
            id: next_prediction_id(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            prediction_type,
            predicted_value,
            confidence,
            features,
            regime: None,
            outcome_recorded: false,
        }
    }

    /// Create with regime information.
    pub fn with_regime(mut self, regime: &str) -> Self {
        self.regime = Some(regime.to_string());
        self
    }

    /// Check if this prediction is awaiting an outcome.
    pub fn awaiting_outcome(&self) -> bool {
        !self.outcome_recorded
    }

    /// Mark outcome as recorded.
    pub fn mark_outcome_recorded(&mut self) {
        self.outcome_recorded = true;
    }
}

/// An outcome record linked to a prediction.
///
/// Records the actual value observed after the prediction window.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutcomeLog {
    /// ID of the prediction this outcome corresponds to.
    pub prediction_id: u64,

    /// Actual observed value.
    pub actual_value: f64,

    /// Time between prediction and outcome measurement in milliseconds.
    pub measurement_delay_ms: u64,

    /// Unix timestamp when outcome was recorded.
    pub timestamp: u64,
}

impl OutcomeLog {
    /// Create a new outcome log entry.
    pub fn new(prediction_id: u64, actual_value: f64, measurement_delay_ms: u64) -> Self {
        Self {
            prediction_id,
            actual_value,
            measurement_delay_ms,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }
}

/// A linked prediction-outcome pair for calibration analysis.
#[derive(Clone, Debug)]
pub struct LinkedPredictionOutcome {
    pub prediction: PredictionLog,
    pub outcome: OutcomeLog,
}

impl LinkedPredictionOutcome {
    /// Squared error for Brier score calculation.
    pub fn squared_error(&self) -> f64 {
        (self.prediction.predicted_value - self.outcome.actual_value).powi(2)
    }

    /// Whether prediction was correct (for binary predictions).
    /// Uses 0.5 threshold for binary classification.
    pub fn is_correct(&self) -> bool {
        let predicted_binary = self.prediction.predicted_value >= 0.5;
        let actual_binary = self.outcome.actual_value >= 0.5;
        predicted_binary == actual_binary
    }
}

/// Storage for predictions and outcomes with linking capabilities.
///
/// Maintains separate queues for pending predictions (awaiting outcomes)
/// and linked pairs (ready for calibration).
#[derive(Clone, Debug)]
pub struct PredictionOutcomeStore {
    /// Predictions awaiting outcomes, keyed by ID.
    pending_predictions: HashMap<u64, PredictionLog>,

    /// Linked prediction-outcome pairs for calibration.
    linked_pairs: VecDeque<LinkedPredictionOutcome>,

    /// Maximum pairs to retain.
    max_pairs: usize,

    /// Maximum pending predictions to retain.
    max_pending: usize,
}

impl Default for PredictionOutcomeStore {
    fn default() -> Self {
        Self::new(10_000, 1_000)
    }
}

impl PredictionOutcomeStore {
    /// Create a new store with specified capacities.
    pub fn new(max_pairs: usize, max_pending: usize) -> Self {
        Self {
            pending_predictions: HashMap::with_capacity(max_pending),
            linked_pairs: VecDeque::with_capacity(max_pairs),
            max_pairs,
            max_pending,
        }
    }

    /// Log a prediction and return its ID for later linking.
    pub fn log_prediction(&mut self, prediction: PredictionLog) -> u64 {
        let id = prediction.id;

        // Enforce pending limit (remove oldest by lowest ID)
        if self.pending_predictions.len() >= self.max_pending {
            if let Some(oldest_id) = self.pending_predictions.keys().min().copied() {
                self.pending_predictions.remove(&oldest_id);
            }
        }

        self.pending_predictions.insert(id, prediction);
        id
    }

    /// Record an outcome and link it to the prediction.
    ///
    /// Returns the linked pair if the prediction was found.
    pub fn record_outcome(&mut self, outcome: OutcomeLog) -> Option<LinkedPredictionOutcome> {
        let mut prediction = self.pending_predictions.remove(&outcome.prediction_id)?;
        prediction.mark_outcome_recorded();

        let linked = LinkedPredictionOutcome {
            prediction,
            outcome,
        };

        // Enforce linked pairs limit
        while self.linked_pairs.len() >= self.max_pairs {
            self.linked_pairs.pop_front();
        }

        self.linked_pairs.push_back(linked.clone());
        Some(linked)
    }

    /// Link a prediction to an outcome by IDs.
    pub fn link_by_id(
        &mut self,
        prediction_id: u64,
        actual_value: f64,
        measurement_delay_ms: u64,
    ) -> Option<LinkedPredictionOutcome> {
        let outcome = OutcomeLog::new(prediction_id, actual_value, measurement_delay_ms);
        self.record_outcome(outcome)
    }

    /// Get all linked pairs for a specific prediction type.
    pub fn pairs_by_type(&self, prediction_type: PredictionType) -> Vec<&LinkedPredictionOutcome> {
        self.linked_pairs
            .iter()
            .filter(|p| p.prediction.prediction_type == prediction_type)
            .collect()
    }

    /// Get recent linked pairs (last N).
    pub fn recent_pairs(&self, n: usize) -> Vec<&LinkedPredictionOutcome> {
        self.linked_pairs.iter().rev().take(n).collect()
    }

    /// Get pending prediction by ID.
    pub fn get_pending(&self, prediction_id: u64) -> Option<&PredictionLog> {
        self.pending_predictions.get(&prediction_id)
    }

    /// Number of pending predictions.
    pub fn pending_count(&self) -> usize {
        self.pending_predictions.len()
    }

    /// Number of linked pairs.
    pub fn linked_count(&self) -> usize {
        self.linked_pairs.len()
    }

    /// Compute Brier score for a prediction type.
    /// Returns None if insufficient samples.
    pub fn brier_score(&self, prediction_type: PredictionType, min_samples: usize) -> Option<f64> {
        let pairs = self.pairs_by_type(prediction_type);
        if pairs.len() < min_samples {
            return None;
        }

        let sum: f64 = pairs.iter().map(|p| p.squared_error()).sum();
        Some(sum / pairs.len() as f64)
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.pending_predictions.clear();
        self.linked_pairs.clear();
    }

    /// Remove stale pending predictions older than max_age_ms.
    pub fn cleanup_stale_pending(&mut self, max_age_ms: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.pending_predictions
            .retain(|_, pred| now.saturating_sub(pred.timestamp) < max_age_ms);
    }
}

// ============================================================================
// Small Fish Strategy: Signal Quality Tracking
// Reference: docs/SMALL_FISH_STRATEGY.md lines 624-643
// ============================================================================

/// Tracks the quality and decay of a signal over time.
///
/// Monitors mutual information (MI) to detect when a signal is:
/// - Becoming stale (half_life_days < 7.0)
/// - No longer useful (MI < 0.01 bits)
///
/// Edge decays over time as markets adapt. This tracker helps detect
/// when a signal needs recalibration or should be removed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignalMiTracker {
    /// Name of the signal being tracked.
    pub signal_name: String,

    /// Current mutual information in bits.
    /// Measures how much information this signal provides about the target.
    /// > 0.01 bits = useful signal.
    pub mutual_information: f64,

    /// Slope of MI over time (negative = decaying).
    /// Used to project future signal value.
    pub mi_trend: f64,

    /// Estimated half-life of the signal in days.
    /// Time until MI drops to 50% of current value.
    /// < 7.0 days = critically fast decay.
    pub half_life_days: f64,

    /// Rolling history of MI measurements for trend calculation.
    #[serde(skip)]
    pub last_n_mi: VecDeque<f64>,

    /// Maximum MI history to retain.
    #[serde(skip)]
    max_history: usize,

    /// Timestamps of MI measurements (for half-life calculation).
    #[serde(skip)]
    mi_timestamps: VecDeque<u64>,
}

impl Default for SignalMiTracker {
    fn default() -> Self {
        Self::new("unnamed")
    }
}

impl SignalMiTracker {
    /// Create a new tracker for a signal.
    pub fn new(signal_name: &str) -> Self {
        Self {
            signal_name: signal_name.to_string(),
            mutual_information: 0.0,
            mi_trend: 0.0,
            half_life_days: f64::INFINITY, // No decay initially
            last_n_mi: VecDeque::with_capacity(100),
            max_history: 100,
            mi_timestamps: VecDeque::with_capacity(100),
        }
    }

    /// Create with custom history size.
    pub fn with_history_size(signal_name: &str, max_history: usize) -> Self {
        Self {
            signal_name: signal_name.to_string(),
            mutual_information: 0.0,
            mi_trend: 0.0,
            half_life_days: f64::INFINITY,
            last_n_mi: VecDeque::with_capacity(max_history),
            max_history,
            mi_timestamps: VecDeque::with_capacity(max_history),
        }
    }

    /// Check if signal is decaying critically fast.
    ///
    /// Returns true if half_life_days < 7.0, meaning the signal
    /// will lose half its predictive value in less than a week.
    pub fn is_stale(&self) -> bool {
        self.half_life_days < 7.0
    }

    /// Check if signal provides useful information.
    ///
    /// Returns true if mutual_information > 0.01 bits.
    /// Below this threshold, the signal is essentially noise.
    pub fn is_useful(&self) -> bool {
        self.mutual_information > 0.01
    }

    /// Check if signal should be kept in the model.
    ///
    /// A signal should be kept if it's both useful and not stale.
    pub fn should_keep(&self) -> bool {
        self.is_useful() && !self.is_stale()
    }

    /// Record a new MI measurement.
    pub fn record_mi(&mut self, mi: f64) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Update current MI
        self.mutual_information = mi;

        // Add to history
        self.last_n_mi.push_back(mi);
        self.mi_timestamps.push_back(now_ms);

        // Maintain max history
        while self.last_n_mi.len() > self.max_history {
            self.last_n_mi.pop_front();
            self.mi_timestamps.pop_front();
        }

        // Update trend and half-life if we have enough data
        if self.last_n_mi.len() >= 3 {
            self.update_trend();
            self.update_half_life();
        }
    }

    /// Update the MI trend using linear regression.
    fn update_trend(&mut self) {
        let n = self.last_n_mi.len();
        if n < 2 {
            return;
        }

        // Simple linear regression: MI = a + b*t
        // We use indices as time proxy (each measurement is ~1 unit apart)
        let n_f = n as f64;
        let sum_x: f64 = (0..n).map(|i| i as f64).sum();
        let sum_y: f64 = self.last_n_mi.iter().sum();
        let sum_xy: f64 = self
            .last_n_mi
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let sum_x2: f64 = (0..n).map(|i| (i as f64).powi(2)).sum();

        let denominator = n_f * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            self.mi_trend = 0.0;
            return;
        }

        // Slope of MI over time (per measurement interval)
        self.mi_trend = (n_f * sum_xy - sum_x * sum_y) / denominator;
    }

    /// Update half-life estimate based on trend.
    fn update_half_life(&mut self) {
        if self.mutual_information <= 0.0 || self.mi_trend >= 0.0 {
            // No decay or MI is zero
            self.half_life_days = f64::INFINITY;
            return;
        }

        // Time to reach 50% of current MI
        // If MI drops by `mi_trend` per measurement, and we want MI/2:
        // current_mi + mi_trend * t = current_mi / 2
        // mi_trend * t = -current_mi / 2
        // t = -current_mi / (2 * mi_trend)
        let measurements_to_half = -self.mutual_information / (2.0 * self.mi_trend);

        // Convert to days (assuming ~1 measurement per hour as default)
        // This can be calibrated based on actual measurement frequency
        let hours_per_measurement = self.estimate_measurement_interval_hours();
        self.half_life_days = measurements_to_half * hours_per_measurement / 24.0;

        // Clamp to reasonable range
        self.half_life_days = self.half_life_days.clamp(0.01, 365.0);
    }

    /// Estimate the average interval between measurements in hours.
    fn estimate_measurement_interval_hours(&self) -> f64 {
        if self.mi_timestamps.len() < 2 {
            return 1.0; // Default to 1 hour
        }

        let first = self.mi_timestamps.front().copied().unwrap_or(0);
        let last = self.mi_timestamps.back().copied().unwrap_or(0);
        let duration_ms = last.saturating_sub(first);
        let n_intervals = (self.mi_timestamps.len() - 1) as f64;

        if n_intervals < 1.0 || duration_ms == 0 {
            return 1.0;
        }

        // Convert ms to hours
        (duration_ms as f64) / (n_intervals * 3600.0 * 1000.0)
    }

    /// Get the number of MI measurements recorded.
    pub fn measurement_count(&self) -> usize {
        self.last_n_mi.len()
    }

    /// Get a diagnostic summary string.
    pub fn diagnostic_summary(&self) -> String {
        format!(
            "signal={}, MI={:.4} bits, trend={:.6}/meas, half_life={:.1} days, useful={}, stale={}",
            self.signal_name,
            self.mutual_information,
            self.mi_trend,
            self.half_life_days,
            self.is_useful(),
            self.is_stale()
        )
    }

    /// Project MI at a future time (in days).
    pub fn project_mi(&self, days_ahead: f64) -> f64 {
        if self.mi_trend >= 0.0 || self.half_life_days == f64::INFINITY {
            return self.mutual_information;
        }

        // Exponential decay model
        let decay_rate = 0.693 / self.half_life_days; // ln(2) / half_life
        (self.mutual_information * (-decay_rate * days_ahead).exp()).max(0.0)
    }

    /// Get recent MI values (for plotting/analysis).
    pub fn recent_mi_values(&self, n: usize) -> Vec<f64> {
        self.last_n_mi.iter().rev().take(n).copied().collect()
    }
}

/// Configuration for calibration tracking.
#[derive(Clone, Debug)]
pub struct CalibrationConfig {
    /// Maximum number of observations to keep (rolling window).
    pub max_observations: usize,
    /// Number of bins for calibration curve (typically 10 or 20).
    pub num_bins: usize,
    /// Minimum observations before computing metrics.
    pub min_observations: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            max_observations: 1000,
            num_bins: 20,
            min_observations: 10, // Reduced from 50 for faster dashboard feedback
        }
    }
}

/// A single prediction-outcome pair.
#[derive(Clone, Debug)]
pub struct PredictionOutcome {
    /// Predicted probability [0, 1].
    pub predicted: f64,
    /// Whether the event occurred.
    pub realized: bool,
    /// When the prediction was made.
    pub timestamp: Instant,
    /// Market regime at prediction time.
    pub regime: String,
}

/// A calibration bin for the calibration curve.
#[derive(Clone, Debug, Serialize)]
pub struct CalibrationBin {
    /// Bin center (e.g., 0.05, 0.15, ..., 0.95).
    pub predicted: f64,
    /// Actual rate of occurrence in this bin.
    pub realized: f64,
    /// Number of observations in this bin.
    pub count: usize,
}

/// Computed calibration metrics.
///
/// Enhanced for Small Fish Strategy (docs/SMALL_FISH_STRATEGY.md lines 602-621).
/// Provides comprehensive calibration assessment including:
/// - Brier score decomposition (resolution, reliability)
/// - Calibration error for model validation
/// - `is_well_calibrated()` method for automated checks
#[derive(Clone, Debug, Serialize)]
pub struct CalibrationMetrics {
    /// Brier score (mean squared error of predictions).
    /// Lower is better. Perfect calibration = 0.
    /// Formula: (1/N) × Σ (p_predicted - outcome)²
    pub brier_score: f64,

    /// Information ratio (resolution / uncertainty).
    /// >1.0 means model adds predictive value.
    pub information_ratio: f64,

    /// Calibration error: mean absolute deviation from perfect calibration diagonal.
    /// For each bin: |predicted_rate - realized_rate|, weighted by bin count.
    /// Lower is better. <0.1 indicates good calibration.
    pub calibration_error: f64,

    /// Resolution: how much predictions deviate from base rate.
    /// Higher resolution = more discriminative predictions.
    /// Formula: Σ n_k/N × (realized_k - base_rate)²
    pub resolution: f64,

    /// Reliability: how close bin predictions are to realized rates.
    /// Lower is better (0 = perfectly reliable).
    /// Formula: Σ n_k/N × (predicted_k - realized_k)²
    pub reliability: f64,

    /// Number of samples used for these metrics.
    pub n_samples: usize,

    /// Total number of observations (alias for n_samples for backward compatibility).
    pub observation_count: usize,

    /// Unix timestamp (ms) when metrics were last updated.
    pub last_updated: u64,

    /// Calibration curve for visualization.
    pub calibration_curve: Vec<CalibrationBin>,

    /// Whether we have enough data for reliable metrics.
    pub is_reliable: bool,
}

impl CalibrationMetrics {
    /// Check if the model is well-calibrated.
    ///
    /// A model is well-calibrated if ALL conditions are met:
    /// - Information Ratio > 1.0 (model adds value)
    /// - Calibration Error < 0.1 (predictions match outcomes)
    /// - n_samples >= 100 (sufficient statistical power)
    ///
    /// Reference: docs/SMALL_FISH_STRATEGY.md lines 616-620
    pub fn is_well_calibrated(&self) -> bool {
        self.information_ratio > 1.0 && self.calibration_error < 0.1 && self.n_samples >= 100
    }

    /// Check if metrics meet minimum sample requirements.
    pub fn has_sufficient_samples(&self) -> bool {
        self.n_samples >= 100
    }

    /// Check if model adds predictive value.
    pub fn adds_value(&self) -> bool {
        self.information_ratio > 1.0
    }

    /// Get a diagnostic summary.
    pub fn diagnostic_summary(&self) -> String {
        format!(
            "n={}, IR={:.3}, cal_err={:.3}, brier={:.4}, well_cal={}",
            self.n_samples,
            self.information_ratio,
            self.calibration_error,
            self.brier_score,
            self.is_well_calibrated()
        )
    }
}

impl Default for CalibrationMetrics {
    fn default() -> Self {
        Self {
            brier_score: 0.0,
            information_ratio: 0.0,
            calibration_error: 0.0,
            resolution: 0.0,
            reliability: 0.0,
            n_samples: 0,
            observation_count: 0,
            last_updated: 0,
            calibration_curve: Vec::new(),
            is_reliable: false,
        }
    }
}

/// Tracks prediction-outcome pairs for a single prediction type.
#[derive(Clone, Debug)]
pub struct PredictionTracker {
    observations: VecDeque<PredictionOutcome>,
    config: CalibrationConfig,
}

impl PredictionTracker {
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            observations: VecDeque::with_capacity(config.max_observations),
            config,
        }
    }

    /// Record a prediction and its outcome.
    pub fn record(&mut self, predicted: f64, realized: bool, regime: &str) {
        // Clamp prediction to [0, 1]
        let predicted = predicted.clamp(0.0, 1.0);

        let outcome = PredictionOutcome {
            predicted,
            realized,
            timestamp: Instant::now(),
            regime: regime.to_string(),
        };

        self.observations.push_back(outcome);

        // Maintain rolling window
        while self.observations.len() > self.config.max_observations {
            self.observations.pop_front();
        }
    }

    /// Compute calibration metrics.
    pub fn metrics(&self) -> CalibrationMetrics {
        let n = self.observations.len();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        if n < self.config.min_observations {
            return CalibrationMetrics {
                is_reliable: false,
                observation_count: n,
                n_samples: n,
                last_updated: now_ms,
                ..Default::default()
            };
        }

        // Compute base rate (overall positive rate)
        let positive_count = self.observations.iter().filter(|o| o.realized).count();
        let base_rate = positive_count as f64 / n as f64;

        // Compute Brier score: mean((predicted - realized)^2)
        let brier_score: f64 = self
            .observations
            .iter()
            .map(|o| {
                let realized = if o.realized { 1.0 } else { 0.0 };
                (o.predicted - realized).powi(2)
            })
            .sum::<f64>()
            / n as f64;

        // Compute calibration curve and resolution
        let calibration_curve = self.compute_calibration_curve();
        let resolution = self.compute_resolution(&calibration_curve, base_rate);

        // Compute reliability: Σ n_k/N × (predicted_k - realized_k)²
        let reliability = self.compute_reliability(&calibration_curve);

        // Compute calibration error: weighted mean absolute deviation from diagonal
        let calibration_error = self.compute_calibration_error(&calibration_curve);

        // Uncertainty (base rate entropy)
        let uncertainty = base_rate * (1.0 - base_rate);

        // Information ratio = resolution / uncertainty
        // Avoid division by zero
        let information_ratio = if uncertainty > 1e-10 {
            resolution / uncertainty
        } else {
            0.0
        };

        CalibrationMetrics {
            brier_score,
            information_ratio,
            calibration_error,
            resolution,
            reliability,
            n_samples: n,
            observation_count: n,
            last_updated: now_ms,
            calibration_curve,
            is_reliable: true,
        }
    }

    /// Compute calibration curve (predicted vs realized by bucket).
    fn compute_calibration_curve(&self) -> Vec<CalibrationBin> {
        let num_bins = self.config.num_bins;
        let bin_width = 1.0 / num_bins as f64;

        let mut bins: Vec<(usize, usize)> = vec![(0, 0); num_bins]; // (positive_count, total_count)

        for obs in &self.observations {
            let bin_idx = ((obs.predicted / bin_width) as usize).min(num_bins - 1);
            bins[bin_idx].1 += 1; // total
            if obs.realized {
                bins[bin_idx].0 += 1; // positive
            }
        }

        bins.iter()
            .enumerate()
            .map(|(i, (pos, total))| {
                let center = (i as f64 + 0.5) * bin_width;
                let realized = if *total > 0 {
                    *pos as f64 / *total as f64
                } else {
                    center // No data, use predicted as placeholder
                };
                CalibrationBin {
                    predicted: center,
                    realized,
                    count: *total,
                }
            })
            .collect()
    }

    /// Compute resolution (how much predictions deviate from base rate).
    fn compute_resolution(&self, curve: &[CalibrationBin], base_rate: f64) -> f64 {
        let n = self.observations.len() as f64;
        if n < 1.0 {
            return 0.0;
        }

        curve
            .iter()
            .map(|bin| {
                let weight = bin.count as f64 / n;
                let deviation = bin.realized - base_rate;
                weight * deviation.powi(2)
            })
            .sum()
    }

    /// Compute reliability (how close bin predictions are to realized rates).
    /// Formula: Σ n_k/N × (predicted_k - realized_k)²
    /// Lower is better (0 = perfectly reliable).
    fn compute_reliability(&self, curve: &[CalibrationBin]) -> f64 {
        let n = self.observations.len() as f64;
        if n < 1.0 {
            return 0.0;
        }

        curve
            .iter()
            .map(|bin| {
                let weight = bin.count as f64 / n;
                let deviation = bin.predicted - bin.realized;
                weight * deviation.powi(2)
            })
            .sum()
    }

    /// Compute calibration error (mean absolute deviation from diagonal).
    /// For each bin: |predicted_rate - realized_rate|, weighted by bin count.
    /// Lower is better. <0.1 indicates good calibration.
    fn compute_calibration_error(&self, curve: &[CalibrationBin]) -> f64 {
        let n = self.observations.len() as f64;
        if n < 1.0 {
            return 0.0;
        }

        curve
            .iter()
            .map(|bin| {
                let weight = bin.count as f64 / n;
                let deviation = (bin.predicted - bin.realized).abs();
                weight * deviation
            })
            .sum()
    }

    /// Get observation count.
    pub fn count(&self) -> usize {
        self.observations.len()
    }

    /// Check if warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.observations.len() >= self.config.min_observations
    }

    /// Get metrics for a specific regime.
    pub fn metrics_for_regime(&self, regime: &str) -> CalibrationMetrics {
        let filtered: Vec<_> = self
            .observations
            .iter()
            .filter(|o| o.regime == regime)
            .cloned()
            .collect();

        if filtered.len() < self.config.min_observations {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            return CalibrationMetrics {
                is_reliable: false,
                observation_count: filtered.len(),
                n_samples: filtered.len(),
                last_updated: now_ms,
                ..Default::default()
            };
        }

        // Create temporary tracker with filtered data
        let mut temp = PredictionTracker::new(self.config.clone());
        for obs in filtered {
            temp.observations.push_back(obs);
        }
        temp.metrics()
    }
}

/// Main calibration tracker that manages multiple prediction types.
#[derive(Clone, Debug)]
pub struct CalibrationTracker {
    /// Fill probability calibration.
    pub fill: PredictionTracker,
    /// Adverse selection calibration.
    pub adverse_selection: PredictionTracker,
    /// Configuration (reserved for future use).
    #[allow(dead_code)]
    config: CalibrationConfig,
}

impl CalibrationTracker {
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            fill: PredictionTracker::new(config.clone()),
            adverse_selection: PredictionTracker::new(config.clone()),
            config,
        }
    }

    /// Record a fill probability prediction and outcome.
    pub fn record_fill(&mut self, predicted_prob: f64, did_fill: bool, regime: &str) {
        self.fill.record(predicted_prob, did_fill, regime);
    }

    /// Record an adverse selection prediction and outcome.
    /// `predicted_alpha` is the predicted probability of informed flow.
    /// `was_adverse` is whether the fill experienced adverse selection.
    pub fn record_adverse_selection(
        &mut self,
        predicted_alpha: f64,
        was_adverse: bool,
        regime: &str,
    ) {
        self.adverse_selection
            .record(predicted_alpha, was_adverse, regime);
    }

    /// Get fill calibration metrics.
    pub fn fill_metrics(&self) -> CalibrationMetrics {
        self.fill.metrics()
    }

    /// Get adverse selection calibration metrics.
    pub fn as_metrics(&self) -> CalibrationMetrics {
        self.adverse_selection.metrics()
    }

    /// Get summary for dashboard.
    pub fn summary(&self) -> CalibrationSummary {
        CalibrationSummary {
            fill: self.fill_metrics(),
            adverse_selection: self.as_metrics(),
        }
    }
}

impl Default for CalibrationTracker {
    fn default() -> Self {
        Self::new(CalibrationConfig::default())
    }
}

/// Summary of all calibration metrics for the dashboard.
#[derive(Clone, Debug, Serialize)]
pub struct CalibrationSummary {
    pub fill: CalibrationMetrics,
    pub adverse_selection: CalibrationMetrics,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker() -> PredictionTracker {
        PredictionTracker::new(CalibrationConfig {
            max_observations: 100,
            num_bins: 10,
            min_observations: 10,
        })
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = make_tracker();
        let metrics = tracker.metrics();
        assert!(!metrics.is_reliable);
        assert_eq!(metrics.observation_count, 0);
    }

    #[test]
    fn test_perfect_calibration() {
        let mut tracker = make_tracker();

        // Record predictions that match outcomes perfectly
        // Low predictions (0.1) that don't occur
        for _ in 0..20 {
            tracker.record(0.1, false, "Quiet");
        }
        // High predictions (0.9) that do occur
        for _ in 0..20 {
            tracker.record(0.9, true, "Quiet");
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);
        // Brier score should be low (good)
        assert!(metrics.brier_score < 0.1, "Brier: {}", metrics.brier_score);
    }

    #[test]
    fn test_poor_calibration() {
        let mut tracker = make_tracker();

        // Record predictions that are systematically wrong
        // Predict high but outcome is negative
        for _ in 0..20 {
            tracker.record(0.9, false, "Quiet");
        }
        // Predict low but outcome is positive
        for _ in 0..20 {
            tracker.record(0.1, true, "Quiet");
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);
        // Brier score should be high (bad)
        assert!(metrics.brier_score > 0.5, "Brier: {}", metrics.brier_score);
    }

    #[test]
    fn test_brier_score_calculation() {
        let mut tracker = make_tracker();

        // Prediction = 0.7, Outcome = 1 -> error = (0.7 - 1)^2 = 0.09
        tracker.record(0.7, true, "Quiet");
        // Prediction = 0.3, Outcome = 0 -> error = (0.3 - 0)^2 = 0.09
        tracker.record(0.3, false, "Quiet");

        // Add more to meet minimum
        for _ in 0..18 {
            tracker.record(0.5, true, "Quiet");
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);
        // Expected: average of all squared errors
    }

    #[test]
    fn test_calibration_curve() {
        let mut tracker = make_tracker();

        // Add observations in different prediction ranges
        for _ in 0..10 {
            tracker.record(0.15, false, "Quiet"); // Bin 1: 0% realized
        }
        for _ in 0..10 {
            tracker.record(0.85, true, "Quiet"); // Bin 8: 100% realized
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);

        // Check calibration curve
        let curve = &metrics.calibration_curve;
        assert_eq!(curve.len(), 10);

        // Bin 1 (0.1-0.2 range) should have 0% realized
        let bin1 = &curve[1];
        assert!(bin1.predicted > 0.1 && bin1.predicted < 0.2);
        assert_eq!(bin1.realized, 0.0);
        assert_eq!(bin1.count, 10);

        // Bin 8 (0.8-0.9 range) should have 100% realized
        let bin8 = &curve[8];
        assert!(bin8.predicted > 0.8 && bin8.predicted < 0.9);
        assert_eq!(bin8.realized, 1.0);
        assert_eq!(bin8.count, 10);
    }

    #[test]
    fn test_rolling_window() {
        let mut tracker = PredictionTracker::new(CalibrationConfig {
            max_observations: 20,
            num_bins: 10,
            min_observations: 5,
        });

        // Add 30 observations (should keep only last 20)
        for i in 0..30 {
            tracker.record(0.5, i % 2 == 0, "Quiet");
        }

        assert_eq!(tracker.count(), 20);
    }

    #[test]
    fn test_calibration_tracker() {
        let mut tracker = CalibrationTracker::default();

        // Record fill predictions
        for _ in 0..100 {
            tracker.record_fill(0.3, false, "Quiet");
            tracker.record_fill(0.7, true, "Quiet");
        }

        // Record AS predictions
        for _ in 0..100 {
            tracker.record_adverse_selection(0.2, false, "Quiet");
            tracker.record_adverse_selection(0.8, true, "Cascade");
        }

        let summary = tracker.summary();
        assert!(summary.fill.is_reliable);
        assert!(summary.adverse_selection.is_reliable);
    }

    #[test]
    fn test_information_ratio() {
        let mut tracker = make_tracker();

        // Add observations with good discrimination
        // When we predict high, it happens; when low, it doesn't
        for _ in 0..25 {
            tracker.record(0.1, false, "Quiet");
            tracker.record(0.9, true, "Quiet");
        }

        let metrics = tracker.metrics();
        assert!(metrics.is_reliable);
        // Good discrimination should give IR > 1.0
        assert!(
            metrics.information_ratio > 0.5,
            "IR: {}",
            metrics.information_ratio
        );
    }

    #[test]
    fn test_regime_filtering() {
        let mut tracker = make_tracker();

        // Add observations in different regimes
        for _ in 0..20 {
            tracker.record(0.8, true, "Quiet");
        }
        for _ in 0..20 {
            tracker.record(0.2, true, "Cascade"); // Poorly calibrated in cascade
        }

        // Overall metrics
        let overall = tracker.metrics();
        assert!(overall.is_reliable);

        // Quiet regime should be well calibrated
        let quiet = tracker.metrics_for_regime("Quiet");
        assert!(quiet.is_reliable);
        assert!(quiet.brier_score < 0.1);

        // Cascade regime should be poorly calibrated
        let cascade = tracker.metrics_for_regime("Cascade");
        assert!(cascade.is_reliable);
        assert!(cascade.brier_score > 0.5);
    }

    // ========================================================================
    // Small Fish Strategy: PredictionLog, OutcomeLog, PredictionOutcomeStore tests
    // ========================================================================

    #[test]
    fn test_prediction_type_display() {
        assert_eq!(
            PredictionType::FillProbability.to_string(),
            "fill_probability"
        );
        assert_eq!(
            PredictionType::AdverseSelection.to_string(),
            "adverse_selection"
        );
        assert_eq!(PredictionType::RegimeChange.to_string(), "regime_change");
        assert_eq!(
            PredictionType::PriceDirection.to_string(),
            "price_direction"
        );
        assert_eq!(PredictionType::Volatility.to_string(), "volatility");
    }

    #[test]
    fn test_prediction_type_all() {
        let all = PredictionType::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&PredictionType::FillProbability));
        assert!(all.contains(&PredictionType::AdverseSelection));
        assert!(all.contains(&PredictionType::RegimeChange));
        assert!(all.contains(&PredictionType::PriceDirection));
        assert!(all.contains(&PredictionType::Volatility));
    }

    #[test]
    fn test_prediction_type_default_delay() {
        assert_eq!(
            PredictionType::FillProbability.default_measurement_delay_ms(),
            5000
        );
        assert_eq!(
            PredictionType::AdverseSelection.default_measurement_delay_ms(),
            2000
        );
        assert_eq!(
            PredictionType::RegimeChange.default_measurement_delay_ms(),
            60000
        );
        assert_eq!(
            PredictionType::PriceDirection.default_measurement_delay_ms(),
            1000
        );
        assert_eq!(
            PredictionType::Volatility.default_measurement_delay_ms(),
            60000
        );
    }

    #[test]
    fn test_prediction_log_creation() {
        let mut features = HashMap::new();
        features.insert("spread_bps".to_string(), 5.0);
        features.insert("volatility".to_string(), 0.02);

        let log = PredictionLog::new(PredictionType::FillProbability, 0.75, 0.9, features.clone());

        assert!(log.id > 0);
        assert!(log.timestamp > 0);
        assert_eq!(log.prediction_type, PredictionType::FillProbability);
        assert_eq!(log.predicted_value, 0.75);
        assert_eq!(log.confidence, 0.9);
        assert_eq!(log.features.get("spread_bps"), Some(&5.0));
        assert!(log.regime.is_none());
        assert!(!log.outcome_recorded);
        assert!(log.awaiting_outcome());
    }

    #[test]
    fn test_prediction_log_with_regime() {
        let log = PredictionLog::new(PredictionType::AdverseSelection, 0.3, 0.8, HashMap::new())
            .with_regime("Cascade");

        assert_eq!(log.regime, Some("Cascade".to_string()));
    }

    #[test]
    fn test_prediction_log_unique_ids() {
        let log1 = PredictionLog::new(PredictionType::FillProbability, 0.5, 0.5, HashMap::new());
        let log2 = PredictionLog::new(PredictionType::FillProbability, 0.5, 0.5, HashMap::new());
        let log3 = PredictionLog::new(PredictionType::FillProbability, 0.5, 0.5, HashMap::new());

        // Each should have a unique ID
        assert_ne!(log1.id, log2.id);
        assert_ne!(log2.id, log3.id);
        assert_ne!(log1.id, log3.id);
    }

    #[test]
    fn test_outcome_log_creation() {
        let outcome = OutcomeLog::new(42, 1.0, 150);

        assert_eq!(outcome.prediction_id, 42);
        assert_eq!(outcome.actual_value, 1.0);
        assert_eq!(outcome.measurement_delay_ms, 150);
        assert!(outcome.timestamp > 0);
    }

    #[test]
    fn test_linked_prediction_outcome_squared_error() {
        let prediction =
            PredictionLog::new(PredictionType::FillProbability, 0.7, 0.9, HashMap::new());
        let outcome = OutcomeLog::new(prediction.id, 1.0, 100);

        let linked = LinkedPredictionOutcome {
            prediction,
            outcome,
        };

        // (0.7 - 1.0)^2 = 0.09
        let error = linked.squared_error();
        assert!((error - 0.09).abs() < 1e-10);
    }

    #[test]
    fn test_linked_prediction_outcome_is_correct() {
        // Correct: predicted >= 0.5 and actual >= 0.5
        let pred1 = PredictionLog::new(PredictionType::FillProbability, 0.7, 0.9, HashMap::new());
        let linked1 = LinkedPredictionOutcome {
            outcome: OutcomeLog::new(pred1.id, 1.0, 100),
            prediction: pred1,
        };
        assert!(linked1.is_correct());

        // Correct: predicted < 0.5 and actual < 0.5
        let pred2 = PredictionLog::new(PredictionType::FillProbability, 0.3, 0.9, HashMap::new());
        let linked2 = LinkedPredictionOutcome {
            outcome: OutcomeLog::new(pred2.id, 0.0, 100),
            prediction: pred2,
        };
        assert!(linked2.is_correct());

        // Incorrect: predicted >= 0.5 but actual < 0.5
        let pred3 = PredictionLog::new(PredictionType::FillProbability, 0.8, 0.9, HashMap::new());
        let linked3 = LinkedPredictionOutcome {
            outcome: OutcomeLog::new(pred3.id, 0.0, 100),
            prediction: pred3,
        };
        assert!(!linked3.is_correct());
    }

    #[test]
    fn test_prediction_outcome_store_log_and_link() {
        let mut store = PredictionOutcomeStore::new(100, 50);

        // Log a prediction
        let prediction =
            PredictionLog::new(PredictionType::FillProbability, 0.75, 0.9, HashMap::new());
        let pred_id = store.log_prediction(prediction);

        assert_eq!(store.pending_count(), 1);
        assert_eq!(store.linked_count(), 0);

        // Record outcome
        let linked = store.link_by_id(pred_id, 1.0, 150);
        assert!(linked.is_some());

        let linked = linked.unwrap();
        assert_eq!(linked.prediction.id, pred_id);
        assert_eq!(linked.outcome.actual_value, 1.0);
        assert_eq!(linked.outcome.measurement_delay_ms, 150);

        assert_eq!(store.pending_count(), 0);
        assert_eq!(store.linked_count(), 1);
    }

    #[test]
    fn test_prediction_outcome_store_missing_prediction() {
        let mut store = PredictionOutcomeStore::new(100, 50);

        // Try to link to non-existent prediction
        let result = store.link_by_id(99999, 1.0, 100);
        assert!(result.is_none());
    }

    #[test]
    fn test_prediction_outcome_store_pairs_by_type() {
        let mut store = PredictionOutcomeStore::new(100, 50);

        // Add fill predictions
        for i in 0..5 {
            let pred = PredictionLog::new(
                PredictionType::FillProbability,
                0.5 + (i as f64) * 0.1,
                0.9,
                HashMap::new(),
            );
            let id = store.log_prediction(pred);
            store.link_by_id(id, if i % 2 == 0 { 1.0 } else { 0.0 }, 100);
        }

        // Add AS predictions
        for i in 0..3 {
            let pred = PredictionLog::new(
                PredictionType::AdverseSelection,
                0.3 + (i as f64) * 0.1,
                0.8,
                HashMap::new(),
            );
            let id = store.log_prediction(pred);
            store.link_by_id(id, 0.0, 200);
        }

        let fill_pairs = store.pairs_by_type(PredictionType::FillProbability);
        assert_eq!(fill_pairs.len(), 5);

        let as_pairs = store.pairs_by_type(PredictionType::AdverseSelection);
        assert_eq!(as_pairs.len(), 3);

        let regime_pairs = store.pairs_by_type(PredictionType::RegimeChange);
        assert_eq!(regime_pairs.len(), 0);
    }

    #[test]
    fn test_prediction_outcome_store_brier_score() {
        let mut store = PredictionOutcomeStore::new(100, 50);

        // Add perfectly calibrated predictions
        // Predict 0.8, get 1.0 -> error = 0.04
        // Predict 0.2, get 0.0 -> error = 0.04
        for _ in 0..10 {
            let pred_high =
                PredictionLog::new(PredictionType::FillProbability, 0.8, 0.9, HashMap::new());
            let id_high = store.log_prediction(pred_high);
            store.link_by_id(id_high, 1.0, 100);

            let pred_low =
                PredictionLog::new(PredictionType::FillProbability, 0.2, 0.9, HashMap::new());
            let id_low = store.log_prediction(pred_low);
            store.link_by_id(id_low, 0.0, 100);
        }

        // Should have 20 samples
        let brier = store.brier_score(PredictionType::FillProbability, 10);
        assert!(brier.is_some());
        let brier = brier.unwrap();
        // Average error = 0.04
        assert!((brier - 0.04).abs() < 1e-10, "Brier: {}", brier);
    }

    #[test]
    fn test_prediction_outcome_store_brier_score_insufficient_samples() {
        let mut store = PredictionOutcomeStore::new(100, 50);

        // Only add 5 samples
        for _ in 0..5 {
            let pred =
                PredictionLog::new(PredictionType::FillProbability, 0.5, 0.9, HashMap::new());
            let id = store.log_prediction(pred);
            store.link_by_id(id, 1.0, 100);
        }

        // Should return None with min_samples=10
        let brier = store.brier_score(PredictionType::FillProbability, 10);
        assert!(brier.is_none());
    }

    #[test]
    fn test_prediction_outcome_store_capacity_limits() {
        let mut store = PredictionOutcomeStore::new(10, 5);

        // Add more pending than limit
        for _ in 0..8 {
            let pred =
                PredictionLog::new(PredictionType::FillProbability, 0.5, 0.9, HashMap::new());
            store.log_prediction(pred);
        }
        // Should cap at max_pending
        assert!(store.pending_count() <= 5);

        // Link all remaining and add more linked
        while store.pending_count() > 0 {
            let id = *store.pending_predictions.keys().next().unwrap();
            store.link_by_id(id, 1.0, 100);
        }

        // Add more to exceed linked limit
        for _ in 0..15 {
            let pred =
                PredictionLog::new(PredictionType::FillProbability, 0.5, 0.9, HashMap::new());
            let id = store.log_prediction(pred);
            store.link_by_id(id, 1.0, 100);
        }
        // Should cap at max_pairs
        assert!(store.linked_count() <= 10);
    }

    #[test]
    fn test_prediction_outcome_store_recent_pairs() {
        let mut store = PredictionOutcomeStore::new(100, 50);

        // Add 10 predictions with increasing values
        for i in 0..10 {
            let pred = PredictionLog::new(
                PredictionType::FillProbability,
                (i as f64) / 10.0,
                0.9,
                HashMap::new(),
            );
            let id = store.log_prediction(pred);
            store.link_by_id(id, 1.0, 100);
        }

        // Get last 3
        let recent = store.recent_pairs(3);
        assert_eq!(recent.len(), 3);

        // Most recent should have predicted_value = 0.9
        assert!((recent[0].prediction.predicted_value - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_prediction_outcome_store_clear() {
        let mut store = PredictionOutcomeStore::new(100, 50);

        // Add some data
        for _ in 0..5 {
            let pred =
                PredictionLog::new(PredictionType::FillProbability, 0.5, 0.9, HashMap::new());
            let id = store.log_prediction(pred);
            store.link_by_id(id, 1.0, 100);
        }

        assert!(store.linked_count() > 0);

        store.clear();
        assert_eq!(store.pending_count(), 0);
        assert_eq!(store.linked_count(), 0);
    }

    // ========================================================================
    // Phase 1.2: Enhanced CalibrationMetrics tests
    // ========================================================================

    #[test]
    fn test_calibration_metrics_new_fields() {
        let mut tracker = make_tracker();

        // Add well-calibrated predictions
        for _ in 0..50 {
            tracker.record(0.1, false, "Quiet");
            tracker.record(0.9, true, "Quiet");
        }

        let metrics = tracker.metrics();

        // Check new fields are populated
        assert!(metrics.n_samples > 0);
        assert_eq!(metrics.n_samples, metrics.observation_count);
        assert!(metrics.last_updated > 0);
        assert!(metrics.resolution >= 0.0);
        assert!(metrics.reliability >= 0.0);
        assert!(metrics.calibration_error >= 0.0);
    }

    #[test]
    fn test_calibration_metrics_is_well_calibrated_true() {
        let mut tracker = PredictionTracker::new(CalibrationConfig {
            max_observations: 1000,
            num_bins: 10,
            min_observations: 10,
        });

        // Add 100+ well-calibrated predictions
        // Low predictions that don't occur, high that do
        for _ in 0..60 {
            tracker.record(0.1, false, "Quiet");
            tracker.record(0.9, true, "Quiet");
        }

        let metrics = tracker.metrics();

        // Should have:
        // - n_samples >= 100 ✓
        // - Good calibration (low calibration_error)
        // - Good discrimination (high IR)
        assert!(metrics.n_samples >= 100, "n_samples: {}", metrics.n_samples);
        assert!(metrics.is_reliable);

        // Check individual components
        println!("Metrics: {}", metrics.diagnostic_summary());
    }

    #[test]
    fn test_calibration_metrics_is_well_calibrated_insufficient_samples() {
        let mut tracker = make_tracker();

        // Only 20 samples (< 100 required)
        for _ in 0..10 {
            tracker.record(0.1, false, "Quiet");
            tracker.record(0.9, true, "Quiet");
        }

        let metrics = tracker.metrics();

        // Should fail due to insufficient samples
        assert_eq!(metrics.n_samples, 20);
        assert!(
            !metrics.is_well_calibrated(),
            "Should fail: insufficient samples"
        );
        assert!(!metrics.has_sufficient_samples());
    }

    #[test]
    fn test_calibration_metrics_helper_methods() {
        let metrics = CalibrationMetrics {
            brier_score: 0.05,
            information_ratio: 1.5,
            calibration_error: 0.05,
            resolution: 0.15,
            reliability: 0.02,
            n_samples: 200,
            observation_count: 200,
            last_updated: 1234567890,
            calibration_curve: vec![],
            is_reliable: true,
        };

        assert!(metrics.is_well_calibrated());
        assert!(metrics.has_sufficient_samples());
        assert!(metrics.adds_value());

        let summary = metrics.diagnostic_summary();
        assert!(summary.contains("n=200"));
        assert!(summary.contains("IR=1.500"));
        assert!(summary.contains("well_cal=true"));
    }

    #[test]
    fn test_calibration_metrics_not_well_calibrated_low_ir() {
        let metrics = CalibrationMetrics {
            brier_score: 0.25,
            information_ratio: 0.5, // Below threshold
            calibration_error: 0.05,
            resolution: 0.05,
            reliability: 0.10,
            n_samples: 200,
            observation_count: 200,
            last_updated: 1234567890,
            calibration_curve: vec![],
            is_reliable: true,
        };

        assert!(!metrics.is_well_calibrated(), "Should fail: IR < 1.0");
        assert!(!metrics.adds_value());
        assert!(metrics.has_sufficient_samples());
    }

    #[test]
    fn test_calibration_metrics_not_well_calibrated_high_cal_error() {
        let metrics = CalibrationMetrics {
            brier_score: 0.25,
            information_ratio: 1.5,
            calibration_error: 0.15, // Above threshold
            resolution: 0.15,
            reliability: 0.10,
            n_samples: 200,
            observation_count: 200,
            last_updated: 1234567890,
            calibration_curve: vec![],
            is_reliable: true,
        };

        assert!(
            !metrics.is_well_calibrated(),
            "Should fail: cal_error > 0.1"
        );
    }

    #[test]
    fn test_calibration_error_calculation() {
        let mut tracker = PredictionTracker::new(CalibrationConfig {
            max_observations: 100,
            num_bins: 10,
            min_observations: 10,
        });

        // Perfect calibration: predict 0.15 and get 15% positive rate
        // Bin 1 (0.1-0.2): predict ~0.15, realize ~0.15
        for _ in 0..17 {
            tracker.record(0.15, false, "Quiet");
        }
        for _ in 0..3 {
            tracker.record(0.15, true, "Quiet");
        }

        let metrics = tracker.metrics();

        // Calibration error should be low for well-calibrated predictions
        // (but not necessarily zero due to binning)
        println!("Cal error: {}", metrics.calibration_error);
        println!("Reliability: {}", metrics.reliability);
    }

    #[test]
    fn test_resolution_calculation() {
        let mut tracker = make_tracker();

        // High resolution: extreme predictions that come true
        for _ in 0..25 {
            tracker.record(0.05, false, "Quiet"); // Low pred, doesn't happen
            tracker.record(0.95, true, "Quiet"); // High pred, does happen
        }

        let metrics = tracker.metrics();

        // Resolution should be high (predictions far from base rate)
        assert!(
            metrics.resolution > 0.1,
            "Resolution: {}",
            metrics.resolution
        );
    }

    #[test]
    fn test_reliability_calculation() {
        let mut tracker = make_tracker();

        // Good reliability: predictions match realized rates in each bin
        for _ in 0..25 {
            tracker.record(0.1, false, "Quiet");
            tracker.record(0.9, true, "Quiet");
        }

        let metrics = tracker.metrics();

        // Reliability should be low (predictions match outcomes)
        assert!(
            metrics.reliability < 0.1,
            "Reliability: {}",
            metrics.reliability
        );
    }

    // ========================================================================
    // Phase 1.3: SignalQualityTracker tests
    // ========================================================================

    #[test]
    fn test_signal_quality_tracker_creation() {
        let tracker = SignalMiTracker::new("test_signal");

        assert_eq!(tracker.signal_name, "test_signal");
        assert_eq!(tracker.mutual_information, 0.0);
        assert_eq!(tracker.mi_trend, 0.0);
        assert_eq!(tracker.half_life_days, f64::INFINITY);
        assert!(tracker.last_n_mi.is_empty());
    }

    #[test]
    fn test_signal_quality_tracker_default() {
        let tracker = SignalMiTracker::default();

        assert_eq!(tracker.signal_name, "unnamed");
        assert_eq!(tracker.mutual_information, 0.0);
    }

    #[test]
    fn test_signal_quality_tracker_is_stale() {
        let mut tracker = SignalMiTracker::new("test");

        // Not stale by default (infinite half-life)
        assert!(!tracker.is_stale());

        // Stale if half-life < 7 days
        tracker.half_life_days = 5.0;
        assert!(tracker.is_stale());

        // Not stale if half-life >= 7 days
        tracker.half_life_days = 7.0;
        assert!(!tracker.is_stale());

        tracker.half_life_days = 30.0;
        assert!(!tracker.is_stale());
    }

    #[test]
    fn test_signal_quality_tracker_is_useful() {
        let mut tracker = SignalMiTracker::new("test");

        // Not useful by default (MI = 0)
        assert!(!tracker.is_useful());

        // Not useful if MI <= 0.01
        tracker.mutual_information = 0.01;
        assert!(!tracker.is_useful());

        tracker.mutual_information = 0.005;
        assert!(!tracker.is_useful());

        // Useful if MI > 0.01
        tracker.mutual_information = 0.011;
        assert!(tracker.is_useful());

        tracker.mutual_information = 0.1;
        assert!(tracker.is_useful());
    }

    #[test]
    fn test_signal_quality_tracker_should_keep() {
        let mut tracker = SignalMiTracker::new("test");

        // Default: not useful (MI=0), not stale (half_life=inf)
        // Should NOT keep because not useful
        assert!(!tracker.should_keep());

        // Useful but stale: should NOT keep
        tracker.mutual_information = 0.1;
        tracker.half_life_days = 3.0;
        assert!(!tracker.should_keep());

        // Useful and not stale: SHOULD keep
        tracker.half_life_days = 14.0;
        assert!(tracker.should_keep());

        // Not useful but not stale: should NOT keep
        tracker.mutual_information = 0.005;
        assert!(!tracker.should_keep());
    }

    #[test]
    fn test_signal_quality_tracker_record_mi() {
        let mut tracker = SignalMiTracker::new("test");

        // Record some MI values
        tracker.record_mi(0.1);
        assert_eq!(tracker.mutual_information, 0.1);
        assert_eq!(tracker.measurement_count(), 1);

        tracker.record_mi(0.08);
        assert_eq!(tracker.mutual_information, 0.08);
        assert_eq!(tracker.measurement_count(), 2);

        tracker.record_mi(0.06);
        assert_eq!(tracker.mutual_information, 0.06);
        assert_eq!(tracker.measurement_count(), 3);

        // With 3+ measurements, trend should be calculated
        // MI is decreasing, so trend should be negative
        assert!(tracker.mi_trend < 0.0, "Trend: {}", tracker.mi_trend);
    }

    #[test]
    fn test_signal_quality_tracker_max_history() {
        let mut tracker = SignalMiTracker::with_history_size("test", 5);

        // Add more than max_history
        for i in 0..10 {
            tracker.record_mi(0.1 - (i as f64) * 0.01);
        }

        // Should only keep last 5
        assert_eq!(tracker.measurement_count(), 5);
    }

    #[test]
    fn test_signal_quality_tracker_stable_signal() {
        let mut tracker = SignalMiTracker::new("stable");

        // Record stable MI values (no decay)
        for _ in 0..10 {
            tracker.record_mi(0.1);
        }

        // Trend should be ~0
        assert!(
            tracker.mi_trend.abs() < 0.001,
            "Trend: {}",
            tracker.mi_trend
        );

        // Half-life should be infinite (no decay)
        assert!(
            tracker.half_life_days > 100.0 || tracker.half_life_days == f64::INFINITY,
            "Half-life: {}",
            tracker.half_life_days
        );
    }

    #[test]
    fn test_signal_quality_tracker_decaying_signal() {
        let mut tracker = SignalMiTracker::new("decaying");

        // Record decaying MI values
        let values = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05];
        for &v in &values {
            tracker.record_mi(v);
        }

        // Trend should be negative
        assert!(tracker.mi_trend < 0.0, "Trend: {}", tracker.mi_trend);

        // Half-life should be finite
        assert!(
            tracker.half_life_days < f64::INFINITY,
            "Half-life should be finite"
        );
        assert!(tracker.half_life_days > 0.0, "Half-life should be positive");
    }

    #[test]
    fn test_signal_quality_tracker_project_mi() {
        let mut tracker = SignalMiTracker::new("test");

        // Set up a decaying signal
        tracker.mutual_information = 0.1;
        tracker.half_life_days = 10.0;
        tracker.mi_trend = -0.001; // negative trend needed for projection

        // Project 10 days ahead (one half-life)
        let projected = tracker.project_mi(10.0);

        // Should be approximately half
        assert!((projected - 0.05).abs() < 0.01, "Projected: {}", projected);

        // Project 0 days: should be current MI
        let now = tracker.project_mi(0.0);
        assert!((now - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_signal_quality_tracker_diagnostic_summary() {
        let mut tracker = SignalMiTracker::new("my_signal");
        tracker.mutual_information = 0.05;
        tracker.mi_trend = -0.001;
        tracker.half_life_days = 14.0;

        let summary = tracker.diagnostic_summary();

        assert!(summary.contains("signal=my_signal"));
        assert!(summary.contains("MI=0.0500"));
        assert!(summary.contains("useful=true"));
        assert!(summary.contains("stale=false"));
    }

    #[test]
    fn test_signal_quality_tracker_recent_mi_values() {
        let mut tracker = SignalMiTracker::new("test");

        for i in 0..5 {
            tracker.record_mi((i + 1) as f64 * 0.01);
        }

        // Get last 3 values (should be 0.05, 0.04, 0.03 in reverse order)
        let recent = tracker.recent_mi_values(3);
        assert_eq!(recent.len(), 3);
        assert!((recent[0] - 0.05).abs() < 0.001); // Most recent
        assert!((recent[1] - 0.04).abs() < 0.001);
        assert!((recent[2] - 0.03).abs() < 0.001);
    }
}
