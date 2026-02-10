//! Level 1: Model Confidence Tracking
//!
//! Tracks prediction accuracy over time for each model component.
//! Detects when the model is breaking down based on calibration scores.

use std::collections::VecDeque;

use super::types::{CalibrationScore, Health, ModelHealth, RingBuffer};

// =============================================================================
// Edge Bias Tracker (Safety Net for Systematic Edge Miscalibration)
// =============================================================================

/// Tracks edge prediction bias to detect systematic miscalibration.
///
/// Uses EWMA (exponential weighted moving average) as the primary signal
/// for detecting bias, with a VecDeque window retained for std computation
/// and backward compatibility.
///
/// Instead of halting quotes (which creates a cold-start deadlock),
/// this tracker provides a recalibration signal: when bias exceeds 1.5 bps
/// (fee-aligned threshold), `should_recalibrate()` returns Some(bias) so
/// downstream can subtract the bias from future predictions.
///
/// # Example
/// - Predicted: +0.02 bps edge
/// - Realized: -16.78 bps edge
/// - Bias: 0.02 - (-16.78) = 16.8 bps overestimate (dangerous!)
#[derive(Debug, Clone)]
pub struct EdgeBiasTracker {
    /// Recent bias values (predicted - realized) — kept for std computation
    recent_bias: VecDeque<f64>,
    /// Window size for tracking
    window_size: usize,
    /// Minimum samples before recalibration can be triggered
    min_samples_for_halt: usize,
    /// Bias threshold in bps (legacy, kept for backward compat)
    halt_threshold_bps: f64,
    /// EWMA of bias (primary signal for recalibration)
    ewma_bias: f64,
    /// EWMA of squared bias (for variance computation)
    ewma_sq_bias: f64,
    /// Decay factor per observation (0.95 = recent fills count more)
    decay_alpha: f64,
    /// Total number of observations (not windowed)
    n_observations: usize,
}

impl Default for EdgeBiasTracker {
    fn default() -> Self {
        Self::new(50, 20, -2.0)
    }
}

impl EdgeBiasTracker {
    pub fn new(window_size: usize, min_samples_for_halt: usize, halt_threshold_bps: f64) -> Self {
        Self {
            recent_bias: VecDeque::with_capacity(window_size),
            window_size,
            min_samples_for_halt,
            halt_threshold_bps,
            ewma_bias: 0.0,
            ewma_sq_bias: 0.0,
            decay_alpha: 0.95,
            n_observations: 0,
        }
    }

    pub fn record(&mut self, predicted_edge_bps: f64, realized_edge_bps: f64) {
        let bias = predicted_edge_bps - realized_edge_bps;

        // VecDeque window for std computation
        if self.recent_bias.len() >= self.window_size {
            self.recent_bias.pop_front();
        }
        self.recent_bias.push_back(bias);

        // EWMA update — primary signal for recalibration
        if self.n_observations == 0 {
            self.ewma_bias = bias;
            self.ewma_sq_bias = bias * bias;
        } else {
            self.ewma_bias = self.decay_alpha * self.ewma_bias + (1.0 - self.decay_alpha) * bias;
            self.ewma_sq_bias = self.decay_alpha * self.ewma_sq_bias + (1.0 - self.decay_alpha) * bias * bias;
        }
        self.n_observations += 1;
    }

    /// Get the mean bias over the recent window.
    ///
    /// Positive bias = overestimating edge (predicting better than realized)
    /// Negative bias = underestimating edge (predicting worse than realized, actually good!)
    pub fn mean_bias(&self) -> f64 {
        if self.recent_bias.is_empty() {
            return 0.0;
        }
        self.recent_bias.iter().sum::<f64>() / self.recent_bias.len() as f64
    }

    /// Get the standard deviation of bias.
    pub fn bias_std(&self) -> f64 {
        if self.recent_bias.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_bias();
        let variance = self.recent_bias.iter()
            .map(|b| (b - mean).powi(2))
            .sum::<f64>() / (self.recent_bias.len() - 1) as f64;
        variance.sqrt()
    }

    pub fn should_halt_quoting(&self) -> bool {
        // Never halt quoting — halting creates a cold-start deadlock where
        // we can't learn without quoting. Use should_recalibrate() instead
        // to get a bias correction signal for downstream consumers.
        false
    }

    /// Check if the edge model needs recalibration due to persistent bias.
    ///
    /// Returns `Some(ewma_bias)` when:
    /// 1. We have enough observations (>= min_samples_for_halt)
    /// 2. EWMA bias exceeds 1.5 bps (fee-aligned threshold)
    ///
    /// The returned value is the bias to subtract from future predictions.
    /// Returns `None` if not enough data or bias is within tolerance.
    pub fn should_recalibrate(&self) -> Option<f64> {
        if self.n_observations < self.min_samples_for_halt {
            return None;
        }

        // 1.5 bps threshold: aligned with maker fee (1.5 bps on Hyperliquid)
        // If we're overestimating edge by more than the fee, we need to recalibrate
        const RECALIBRATE_THRESHOLD_BPS: f64 = 1.5;

        if self.ewma_bias.abs() > RECALIBRATE_THRESHOLD_BPS {
            Some(self.ewma_bias)
        } else {
            None
        }
    }

    /// Check if the edge bias is in warning territory (close to halt).
    pub fn is_warning(&self) -> bool {
        if self.recent_bias.len() < self.min_samples_for_halt / 2 {
            return false;
        }
        // Warning at 50% of halt threshold
        self.mean_bias() > -self.halt_threshold_bps * 0.5
    }

    /// Get the number of samples tracked.
    pub fn sample_count(&self) -> usize {
        self.recent_bias.len()
    }

    /// Get the total number of observations ever recorded (not windowed).
    pub fn total_observations(&self) -> usize {
        self.n_observations
    }

    /// Get the EWMA bias value.
    pub fn ewma_bias(&self) -> f64 {
        self.ewma_bias
    }

    pub fn summary(&self) -> EdgeBiasSummary {
        EdgeBiasSummary {
            mean_bias_bps: self.mean_bias(),
            bias_std_bps: self.bias_std(),
            sample_count: self.recent_bias.len(),
            should_halt: false,
            is_warning: self.is_warning(),
        }
    }

    pub fn clear(&mut self) {
        self.recent_bias.clear();
        self.ewma_bias = 0.0;
        self.ewma_sq_bias = 0.0;
        self.n_observations = 0;
    }
}

/// Summary of edge bias tracker state.
#[derive(Debug, Clone)]
pub struct EdgeBiasSummary {
    /// Mean bias in bps (positive = overestimating edge)
    pub mean_bias_bps: f64,
    /// Standard deviation of bias
    pub bias_std_bps: f64,
    /// Number of samples in window
    pub sample_count: usize,
    /// Whether halt condition is triggered
    pub should_halt: bool,
    /// Whether in warning territory
    pub is_warning: bool,
}

// =============================================================================
// Aggregate Confidence for Proactive Position Management
// =============================================================================

/// Aggregate confidence score combining all model components.
///
/// Used for confidence-gated sizing: when uncertain, quote smaller sizes.
/// This transforms the system from passive (react to fills) to proactive
/// (assess confidence → size appropriately → quote deliberately).
///
/// # Sizing Policy
/// - confidence ≥ 0.8 → full size (high confidence)
/// - confidence ∈ [0.5, 0.8) → scaled size
/// - confidence < 0.5 → minimum size (30%)
///
/// Key insight: During warmup, models have high uncertainty → smaller quotes.
/// As calibration improves, confidence grows → larger quotes. This naturally
/// implements "earn the right to size up" without explicit rules.
#[derive(Debug, Clone, Default)]
pub struct AggregateConfidence {
    /// Overall confidence score [0.0, 1.0].
    /// Geometric mean of component confidences for multiplicative combination.
    pub score: f64,

    /// Momentum/direction prediction confidence.
    /// High when momentum signal is clear and consistent.
    pub momentum_confidence: f64,

    /// Volatility estimate confidence.
    /// High when vol estimates are stable and well-calibrated.
    pub volatility_confidence: f64,

    /// Fill rate (kappa) estimation confidence.
    /// High when we have enough fill observations.
    pub kappa_confidence: f64,

    /// Adverse selection prediction confidence.
    /// High when AS model is calibrated with sufficient fills.
    pub as_confidence: f64,

    /// Number of observations used to compute confidence.
    pub n_observations: usize,

    /// Whether models are in warmup phase.
    pub is_warmup: bool,
}

impl AggregateConfidence {
    /// Create a new aggregate confidence from component confidences.
    pub fn new(
        momentum_confidence: f64,
        volatility_confidence: f64,
        kappa_confidence: f64,
        as_confidence: f64,
        n_observations: usize,
        is_warmup: bool,
    ) -> Self {
        // Clamp all inputs to [0, 1]
        let mom = momentum_confidence.clamp(0.0, 1.0);
        let vol = volatility_confidence.clamp(0.0, 1.0);
        let kappa = kappa_confidence.clamp(0.0, 1.0);
        let as_conf = as_confidence.clamp(0.0, 1.0);

        // Geometric mean for overall score (multiplicative combination)
        // This means ALL components must be confident for high overall score
        let product = mom * vol * kappa * as_conf;
        let score = if product > 0.0 {
            product.powf(0.25) // 4th root = geometric mean of 4 values
        } else {
            0.0
        };

        Self {
            score,
            momentum_confidence: mom,
            volatility_confidence: vol,
            kappa_confidence: kappa,
            as_confidence: as_conf,
            n_observations,
            is_warmup,
        }
    }

    /// Create a warmup confidence with conservative values.
    pub fn warmup(warmup_progress: f64) -> Self {
        // During warmup, confidence scales with progress
        let progress = warmup_progress.clamp(0.0, 1.0);

        // Start very conservative, ramp up
        let base_confidence = 0.3 + 0.5 * progress; // [0.3, 0.8]

        Self {
            score: base_confidence,
            momentum_confidence: base_confidence,
            volatility_confidence: base_confidence,
            kappa_confidence: progress, // Kappa needs most warmup
            as_confidence: base_confidence,
            n_observations: 0,
            is_warmup: true,
        }
    }

    /// Size multiplier based on confidence [0.3, 1.0].
    ///
    /// Never go below 30% size to maintain market presence.
    /// Maps confidence [0, 1] → size [0.3, 1.0].
    ///
    /// Formula: size_mult = 0.3 + 0.7 × confidence
    pub fn size_multiplier(&self) -> f64 {
        let floor = 0.3; // Minimum 30% size always
        let range = 0.7; // Can scale up by 70%

        floor + range * self.score
    }

    /// Alternative size multiplier with steeper curve.
    ///
    /// Uses quadratic scaling: more aggressive reduction at low confidence.
    /// size_mult = 0.3 + 0.7 × confidence²
    pub fn size_multiplier_steep(&self) -> f64 {
        let floor = 0.3;
        let range = 0.7;

        floor + range * self.score.powi(2)
    }

    /// Get the weakest component confidence.
    pub fn weakest_component(&self) -> (&'static str, f64) {
        let components = [
            ("momentum", self.momentum_confidence),
            ("volatility", self.volatility_confidence),
            ("kappa", self.kappa_confidence),
            ("as", self.as_confidence),
        ];

        components
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }

    /// Check if confidence is sufficient for full-size quoting.
    pub fn is_high_confidence(&self) -> bool {
        self.score >= 0.8
    }

    /// Check if in low-confidence mode (reduced sizing).
    pub fn is_low_confidence(&self) -> bool {
        self.score < 0.5
    }
}

/// Volatility prediction record.
#[derive(Debug, Clone)]
pub struct VolPrediction {
    /// Predicted volatility
    pub predicted_sigma: f64,
    /// Predicted uncertainty (from Kalman or Bayesian posterior)
    pub predicted_uncertainty: f64,
    /// Time horizon for prediction
    pub horizon_ms: u64,
    /// Realized volatility (computed after horizon elapsed)
    pub realized_sigma: f64,
}

/// Fill rate prediction record.
#[derive(Debug, Clone)]
pub struct FillPrediction {
    /// Depth from touch in bps
    pub depth_bps: f64,
    /// Predicted fill probability
    pub predicted_fill_prob: f64,
    /// Time horizon for prediction
    pub horizon_ms: u64,
    /// Whether the order was filled
    pub was_filled: bool,
}

/// Adverse selection prediction record.
#[derive(Debug, Clone)]
pub struct ASPrediction {
    /// Fill price
    pub fill_price: f64,
    /// Predicted adverse selection in bps
    pub predicted_as_bps: f64,
    /// Time horizon for measurement
    pub measurement_horizon_ms: u64,
    /// Realized adverse selection in bps
    pub realized_as_bps: f64,
}

/// Edge prediction record.
#[derive(Debug, Clone)]
pub struct EdgePrediction {
    /// Predicted edge in bps
    pub predicted_edge_bps: f64,
    /// Predicted uncertainty
    pub predicted_uncertainty: f64,
    /// Realized P&L in bps
    pub realized_pnl_bps: f64,
}

/// Tracks prediction accuracy for all model components.
///
/// Key metrics:
/// - `vol_rmse`: Should be < 2× predicted uncertainty
/// - `as_bias`: Positive = underestimating AS (dangerous)
/// - `edge_calibration.error`: Should be < 0.5 bps
/// - `edge_bias_tracker`: Safety net for systematic edge miscalibration
pub struct ModelConfidenceTracker {
    // === Volatility predictions ===
    vol_predictions: RingBuffer<VolPrediction>,
    vol_rmse: f64,
    vol_bias: f64,

    // === Fill rate predictions ===
    fill_predictions: RingBuffer<FillPrediction>,
    fill_calibration: CalibrationScore,

    // === Adverse selection predictions ===
    as_predictions: RingBuffer<ASPrediction>,
    as_rmse: f64,
    as_bias: f64,

    // === Edge predictions ===
    edge_predictions: RingBuffer<EdgePrediction>,
    edge_calibration: CalibrationScore,

    // === Edge Bias Safety Net ===
    /// Tracks edge prediction bias for halt decisions
    edge_bias_tracker: EdgeBiasTracker,

    // === Configuration ===
    /// Maximum AS bias before warning (bps)
    max_as_bias_warning: f64,
    /// Maximum AS bias before degraded (bps)
    max_as_bias_degraded: f64,
}

impl Default for ModelConfidenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelConfidenceTracker {
    /// Create a new confidence tracker.
    pub fn new() -> Self {
        Self {
            vol_predictions: RingBuffer::new(200),
            vol_rmse: 0.0,
            vol_bias: 0.0,
            fill_predictions: RingBuffer::new(500),
            fill_calibration: CalibrationScore::default(),
            as_predictions: RingBuffer::new(500),
            as_rmse: 0.0,
            as_bias: 0.0,
            edge_predictions: RingBuffer::new(1000),
            edge_calibration: CalibrationScore::default(),
            // Edge bias safety net: 50 samples, 20 min for halt, -2 bps threshold
            edge_bias_tracker: EdgeBiasTracker::new(50, 20, -2.0),
            max_as_bias_warning: 0.5,  // 0.5 bps
            max_as_bias_degraded: 1.0, // 1.0 bps
        }
    }

    /// Record a volatility prediction.
    pub fn record_vol_prediction(&mut self, pred: VolPrediction) {
        self.vol_predictions.push(pred);
        self.update_vol_metrics();
    }

    /// Record a fill rate prediction.
    pub fn record_fill_prediction(&mut self, pred: FillPrediction) {
        self.fill_predictions.push(pred);
        self.update_fill_metrics();
    }

    /// Record an adverse selection prediction.
    pub fn record_as_prediction(&mut self, pred: ASPrediction) {
        self.as_predictions.push(pred);
        self.update_as_metrics();
    }

    /// Record an edge prediction (main entry point from trading).
    pub fn record_edge_prediction(
        &mut self,
        predicted_edge_bps: f64,
        predicted_uncertainty: f64,
        realized_pnl_bps: f64,
    ) {
        // Log for scatter plot analysis
        tracing::info!(
            target: "learning::scatter",
            predicted_bps = %format!("{:.2}", predicted_edge_bps),
            realized_bps = %format!("{:.2}", realized_pnl_bps),
            uncertainty = %format!("{:.2}", predicted_uncertainty),
            "[EdgeScatter] predicted vs realized edge"
        );

        let pred = EdgePrediction {
            predicted_edge_bps,
            predicted_uncertainty,
            realized_pnl_bps,
        };
        self.edge_predictions.push(pred);
        self.update_edge_metrics();

        // Update edge bias safety net tracker
        self.edge_bias_tracker.record(predicted_edge_bps, realized_pnl_bps);

        // Log recalibration signal if bias is significant
        if let Some(bias_bps) = self.edge_bias_tracker.should_recalibrate() {
            tracing::warn!(
                ewma_bias_bps = %format!("{:.2}", bias_bps),
                n_observations = self.edge_bias_tracker.total_observations(),
                "Edge bias recalibration needed: EWMA bias exceeds fee threshold"
            );
        } else if self.edge_bias_tracker.is_warning() {
            let bias_summary = self.edge_bias_tracker.summary();
            tracing::warn!(
                mean_bias_bps = %format!("{:.2}", bias_summary.mean_bias_bps),
                sample_count = bias_summary.sample_count,
                "Edge bias warning: Approaching recalibration threshold"
            );
        }
    }

    /// Update volatility metrics.
    fn update_vol_metrics(&mut self) {
        if self.vol_predictions.is_empty() {
            return;
        }

        let n = self.vol_predictions.len() as f64;
        let mut sum_sq_error = 0.0;
        let mut sum_bias = 0.0;

        for pred in self.vol_predictions.iter() {
            let error = pred.predicted_sigma - pred.realized_sigma;
            sum_sq_error += error * error;
            sum_bias += error;
        }

        self.vol_rmse = (sum_sq_error / n).sqrt();
        self.vol_bias = sum_bias / n;
    }

    /// Update fill rate metrics using Brier score.
    fn update_fill_metrics(&mut self) {
        if self.fill_predictions.is_empty() {
            return;
        }

        let n = self.fill_predictions.len() as f64;
        let mut sum_sq_error = 0.0;
        let mut sum_bias = 0.0;
        let mut sum_uncertainty = 0.0;

        for pred in self.fill_predictions.iter() {
            let outcome = if pred.was_filled { 1.0 } else { 0.0 };
            let error = pred.predicted_fill_prob - outcome;
            sum_sq_error += error * error;
            sum_bias += error;
            // For fill predictions, uncertainty is related to probability
            sum_uncertainty += pred.predicted_fill_prob * (1.0 - pred.predicted_fill_prob);
        }

        self.fill_calibration = CalibrationScore {
            error: (sum_sq_error / n).sqrt(),
            bias: sum_bias / n,
            sharpness: sum_uncertainty / n,
            n_observations: self.fill_predictions.len(),
        };
    }

    /// Update adverse selection metrics.
    fn update_as_metrics(&mut self) {
        if self.as_predictions.is_empty() {
            return;
        }

        let n = self.as_predictions.len() as f64;
        let mut sum_sq_error = 0.0;
        let mut sum_bias = 0.0;

        for pred in self.as_predictions.iter() {
            let error = pred.predicted_as_bps - pred.realized_as_bps;
            sum_sq_error += error * error;
            sum_bias += error;
        }

        self.as_rmse = (sum_sq_error / n).sqrt();
        // Negative bias = underestimating AS (we think AS is lower than reality)
        // This is DANGEROUS because we're not widening spreads enough
        self.as_bias = sum_bias / n;
    }

    /// Update edge prediction metrics.
    fn update_edge_metrics(&mut self) {
        if self.edge_predictions.is_empty() {
            return;
        }

        let n = self.edge_predictions.len() as f64;
        let mut sum_sq_error = 0.0;
        let mut sum_bias = 0.0;
        let mut sum_uncertainty = 0.0;

        for pred in self.edge_predictions.iter() {
            let error = pred.predicted_edge_bps - pred.realized_pnl_bps;
            sum_sq_error += error * error;
            sum_bias += error;
            sum_uncertainty += pred.predicted_uncertainty;
        }

        self.edge_calibration = CalibrationScore {
            error: (sum_sq_error / n).sqrt(),
            bias: sum_bias / n,
            sharpness: sum_uncertainty / n,
            n_observations: self.edge_predictions.len(),
        };
    }

    /// Compute calibration score for edge predictions.
    ///
    /// Bins predictions by predicted edge and computes mean realized
    /// in each bin. Perfect calibration: predicted == realized in each bin.
    pub fn edge_calibration_score(&self) -> &CalibrationScore {
        &self.edge_calibration
    }

    /// Assess overall model health.
    pub fn model_health(&self) -> ModelHealth {
        let mut health = ModelHealth::default();

        // === Volatility health ===
        if self.vol_predictions.len() < 20 {
            health.volatility = Health::Good; // Not enough data
        } else {
            // Check if RMSE is within 2x of average predicted uncertainty
            let avg_uncertainty: f64 = self
                .vol_predictions
                .iter()
                .map(|p| p.predicted_uncertainty)
                .sum::<f64>()
                / self.vol_predictions.len() as f64;

            health.volatility = if self.vol_rmse < avg_uncertainty * 2.0 {
                Health::Good
            } else if self.vol_rmse < avg_uncertainty * 3.0 {
                Health::Warning
            } else {
                Health::Degraded
            };
        }

        // === Adverse selection health ===
        if self.as_predictions.len() < 20 {
            health.adverse_selection = Health::Good; // Not enough data
        } else {
            // Negative bias (underestimating AS) is dangerous
            health.adverse_selection = if self.as_bias > -self.max_as_bias_warning {
                Health::Good
            } else if self.as_bias > -self.max_as_bias_degraded {
                Health::Warning
            } else {
                Health::Degraded
            };
        }

        // === Fill rate health ===
        if self.fill_predictions.len() < 50 {
            health.fill_rate = Health::Good; // Not enough data
        } else {
            health.fill_rate = if self.fill_calibration.error < 0.2 {
                Health::Good
            } else if self.fill_calibration.error < 0.3 {
                Health::Warning
            } else {
                Health::Degraded
            };
        }

        // === Edge health (calibration ratio: |mean_bias| / prediction_rmse) ===
        if self.edge_predictions.len() < 50 {
            health.edge = Health::Good; // Not enough data
            health.edge_calibration_ratio = 0.0;
        } else {
            let rmse = self.edge_calibration.error;
            let mean_bias = self.edge_calibration.bias.abs();

            // Guard: tiny RMSE means insufficient data to distinguish bias from noise
            if rmse < 0.1 {
                health.edge = Health::Good;
                health.edge_calibration_ratio = 0.0;
            } else {
                let ratio = mean_bias / rmse;
                health.edge_calibration_ratio = ratio;
                health.edge = if ratio < 0.5 {
                    Health::Good
                } else if ratio < 0.7 {
                    Health::Warning
                } else {
                    Health::Degraded
                };
            }
        }

        // Compute overall
        health.update_overall();
        health
    }

    /// Check if model is degraded.
    pub fn is_model_degraded(&self) -> bool {
        self.model_health().is_degraded()
    }

    // === Getters for metrics ===

    pub fn vol_rmse(&self) -> f64 {
        self.vol_rmse
    }

    pub fn vol_bias(&self) -> f64 {
        self.vol_bias
    }

    pub fn as_rmse(&self) -> f64 {
        self.as_rmse
    }

    pub fn as_bias(&self) -> f64 {
        self.as_bias
    }

    pub fn edge_bias(&self) -> f64 {
        self.edge_calibration.bias
    }

    pub fn edge_rmse(&self) -> f64 {
        self.edge_calibration.error
    }

    pub fn n_edge_observations(&self) -> usize {
        self.edge_predictions.len()
    }

    // === Edge Bias Safety Net ===

    pub fn should_halt_quoting(&self) -> bool {
        // Never halt quoting — use edge_bias_tracker().should_recalibrate() instead
        false
    }

    /// Check if edge bias is in warning territory.
    pub fn edge_bias_warning(&self) -> bool {
        self.edge_bias_tracker.is_warning()
    }

    /// Get edge bias tracker summary for diagnostics.
    pub fn edge_bias_summary(&self) -> EdgeBiasSummary {
        self.edge_bias_tracker.summary()
    }

    /// Get the edge bias tracker for direct access.
    pub fn edge_bias_tracker(&self) -> &EdgeBiasTracker {
        &self.edge_bias_tracker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // AggregateConfidence Tests
    // =============================================================================

    #[test]
    fn test_aggregate_confidence_full() {
        let conf = AggregateConfidence::new(1.0, 1.0, 1.0, 1.0, 100, false);
        assert!((conf.score - 1.0).abs() < 0.01);
        assert!((conf.size_multiplier() - 1.0).abs() < 0.01);
        assert!(conf.is_high_confidence());
        assert!(!conf.is_low_confidence());
    }

    #[test]
    fn test_aggregate_confidence_low() {
        let conf = AggregateConfidence::new(0.3, 0.3, 0.3, 0.3, 10, true);
        assert!(conf.score < 0.5);
        assert!(conf.is_low_confidence());
        // Size multiplier should be close to floor (0.3)
        let size_mult = conf.size_multiplier();
        assert!(size_mult >= 0.3 && size_mult < 0.6);
    }

    #[test]
    fn test_aggregate_confidence_mixed() {
        // One weak component should drag down overall
        let conf = AggregateConfidence::new(0.9, 0.9, 0.2, 0.9, 50, false);
        // Geometric mean: (0.9 * 0.9 * 0.2 * 0.9)^0.25 ≈ 0.59
        assert!(conf.score < 0.7);

        let (name, value) = conf.weakest_component();
        assert_eq!(name, "kappa");
        assert!((value - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_confidence_warmup() {
        // At 0% warmup progress
        let conf_0 = AggregateConfidence::warmup(0.0);
        assert!(conf_0.is_warmup);
        assert!((conf_0.score - 0.3).abs() < 0.1);

        // At 50% warmup progress
        let conf_50 = AggregateConfidence::warmup(0.5);
        assert!(conf_50.score > conf_0.score);

        // At 100% warmup progress
        let conf_100 = AggregateConfidence::warmup(1.0);
        assert!((conf_100.score - 0.8).abs() < 0.1);
    }

    #[test]
    fn test_size_multiplier_range() {
        // Score = 0 → size = 0.3 (floor)
        let conf_low = AggregateConfidence::new(0.0, 0.0, 0.0, 0.0, 0, true);
        assert!((conf_low.size_multiplier() - 0.3).abs() < 0.01);

        // Score = 1 → size = 1.0 (ceiling)
        let conf_high = AggregateConfidence::new(1.0, 1.0, 1.0, 1.0, 100, false);
        assert!((conf_high.size_multiplier() - 1.0).abs() < 0.01);

        // Score = 0.5 → size = 0.65 (midpoint)
        let conf_mid = AggregateConfidence {
            score: 0.5,
            ..Default::default()
        };
        assert!((conf_mid.size_multiplier() - 0.65).abs() < 0.01);
    }

    // =============================================================================
    // ModelConfidenceTracker Tests
    // =============================================================================

    #[test]
    fn test_empty_tracker() {
        let tracker = ModelConfidenceTracker::new();
        let health = tracker.model_health();
        assert_eq!(health.overall, Health::Good);
    }

    #[test]
    fn test_edge_prediction_tracking() {
        let mut tracker = ModelConfidenceTracker::new();

        // Add some well-calibrated predictions
        for i in 0..100 {
            let predicted = 2.0 + (i % 5) as f64 * 0.1;
            let realized = predicted + (i % 3) as f64 * 0.1 - 0.1; // Small noise
            tracker.record_edge_prediction(predicted, 0.5, realized);
        }

        let health = tracker.model_health();
        assert_eq!(health.edge, Health::Good);
    }

    #[test]
    fn test_as_underestimation_warning() {
        let mut tracker = ModelConfidenceTracker::new();

        // Add predictions that systematically underestimate AS
        for _ in 0..50 {
            tracker.record_as_prediction(ASPrediction {
                fill_price: 100.0,
                predicted_as_bps: 1.0, // We predict 1bp
                measurement_horizon_ms: 1000,
                realized_as_bps: 2.0, // But it's actually 2bp
            });
        }

        let health = tracker.model_health();
        // Bias is -1.0 (underestimating by 1bp)
        assert!(tracker.as_bias < -0.5);
        assert_eq!(health.adverse_selection, Health::Degraded);
    }

    #[test]
    fn test_fill_calibration() {
        let mut tracker = ModelConfidenceTracker::new();

        // Add well-calibrated fill predictions with high probability
        // For Brier score RMSE to be < 0.2, we need p(1-p) < 0.04
        // Using p=0.96 gives sqrt(0.96 × 0.04) = sqrt(0.0384) ≈ 0.196 < 0.2
        for i in 0..100 {
            let prob = 0.96;
            let was_filled = (i % 25) < 24; // 96% fill rate (24 out of 25)
            tracker.record_fill_prediction(FillPrediction {
                depth_bps: 5.0,
                predicted_fill_prob: prob,
                horizon_ms: 1000,
                was_filled,
            });
        }

        let health = tracker.model_health();
        // Well-calibrated high-probability predictions should be good
        assert_eq!(health.fill_rate, Health::Good);
    }

    // =============================================================================
    // EdgeBiasTracker Tests
    // =============================================================================

    #[test]
    fn test_edge_bias_tracker_empty() {
        let tracker = EdgeBiasTracker::default();
        assert_eq!(tracker.mean_bias(), 0.0);
        assert!(!tracker.should_halt_quoting());
        assert!(!tracker.is_warning());
    }

    #[test]
    fn test_edge_bias_tracker_no_halt_insufficient_samples() {
        let mut tracker = EdgeBiasTracker::new(50, 20, -2.0);

        // Add only 10 samples with bad bias
        for _ in 0..10 {
            tracker.record(5.0, -10.0); // Predicted +5, realized -10, bias = +15
        }

        // Should not halt because not enough samples
        assert!(!tracker.should_halt_quoting());
        assert_eq!(tracker.sample_count(), 10);
    }

    #[test]
    fn test_edge_bias_tracker_halt_on_overestimation() {
        let mut tracker = EdgeBiasTracker::new(50, 20, -2.0);

        // Add 25 samples with consistent overestimation
        // Predicted +5 bps, realized -10 bps → bias = +15 bps (bad!)
        for _ in 0..25 {
            tracker.record(5.0, -10.0);
        }

        // Mean bias = +15, which exceeds threshold
        assert!(tracker.mean_bias() > 2.0);
        // should_halt_quoting() always returns false now
        assert!(!tracker.should_halt_quoting());
        // But should_recalibrate() returns the EWMA bias (~15.0)
        let recal = tracker.should_recalibrate();
        assert!(recal.is_some());
        assert!(recal.unwrap() > 10.0); // EWMA converges toward 15.0
    }

    #[test]
    fn test_edge_bias_tracker_no_halt_good_calibration() {
        let mut tracker = EdgeBiasTracker::new(50, 20, -2.0);

        // Add 25 samples with good calibration (small bias)
        for i in 0..25 {
            let noise = ((i % 3) as f64 - 1.0) * 0.5; // -0.5, 0, 0.5
            tracker.record(2.0 + noise, 2.0); // Nearly perfect prediction
        }

        // Mean bias should be close to 0
        assert!(tracker.mean_bias().abs() < 1.0);
        assert!(!tracker.should_halt_quoting());
        // No recalibration needed — bias is small
        assert!(tracker.should_recalibrate().is_none());
    }

    #[test]
    fn test_edge_bias_tracker_warning_threshold() {
        let mut tracker = EdgeBiasTracker::new(50, 20, -2.0);

        // Add samples with moderate overestimation (warning but not halt)
        // Bias of ~1.5 bps (above 50% of 2.0 threshold but below 2.0)
        for _ in 0..15 {
            tracker.record(3.0, 1.5); // Bias = 1.5
        }

        // Should be in warning (>= 10 samples, bias > 1.0)
        assert!(tracker.is_warning());
        assert!(!tracker.should_halt_quoting()); // Not enough samples for halt
    }

    #[test]
    fn test_edge_bias_tracker_sliding_window() {
        let mut tracker = EdgeBiasTracker::new(10, 5, -2.0);

        // Fill with bad predictions first
        for _ in 0..10 {
            tracker.record(10.0, 0.0); // Bias = +10
        }
        // should_halt_quoting always false now
        assert!(!tracker.should_halt_quoting());
        // But recalibration should trigger (bias ~10.0)
        assert!(tracker.should_recalibrate().is_some());

        // Now add good predictions to push out bad ones
        for _ in 0..10 {
            tracker.record(1.0, 1.0); // Bias = 0
        }

        // Window should now only have good predictions
        assert!(tracker.mean_bias().abs() < 1.0);
        assert!(!tracker.should_halt_quoting());
        // EWMA decays: after 10 zero-bias observations, EWMA should have decayed
        // significantly. With alpha=0.95, after 10 steps: 10.0 * 0.95^10 ≈ 5.99
        // Still above 1.5 threshold, but much reduced
    }

    #[test]
    fn test_edge_bias_integrated_with_confidence_tracker() {
        let mut tracker = ModelConfidenceTracker::new();

        // Record many fills with consistent overestimation
        for _ in 0..25 {
            tracker.record_edge_prediction(5.0, 1.0, -10.0); // Big overestimate
        }

        // should_halt_quoting() always returns false now
        assert!(!tracker.should_halt_quoting());

        let summary = tracker.edge_bias_summary();
        assert!(summary.mean_bias_bps > 10.0); // +15 bps bias
        assert!(!summary.should_halt); // Always false

        // But recalibration signal should fire
        let recal = tracker.edge_bias_tracker().should_recalibrate();
        assert!(recal.is_some());
        assert!(recal.unwrap() > 10.0);
    }

    #[test]
    fn test_edge_bias_ewma_convergence() {
        let mut tracker = EdgeBiasTracker::new(50, 10, -2.0);

        // Insufficient samples → no recalibration yet
        for _ in 0..5 {
            tracker.record(5.0, 0.0); // Bias = +5.0
        }
        assert!(tracker.should_recalibrate().is_none()); // Only 5 obs, need 10

        // Add more to exceed min_samples_for_halt
        for _ in 0..15 {
            tracker.record(5.0, 0.0); // Bias = +5.0
        }
        // 20 total observations, bias converging toward 5.0
        let recal = tracker.should_recalibrate();
        assert!(recal.is_some());
        let bias = recal.unwrap();
        // EWMA with alpha=0.95: converges toward 5.0
        // After 20 steps of constant bias=5.0:
        // ewma = 5.0 * (1 - 0.95^20) ≈ 5.0 * 0.642 = 3.21 at minimum
        assert!(bias > 3.0, "EWMA bias should converge toward 5.0, got {}", bias);
        assert!(bias < 5.1, "EWMA bias should not exceed input, got {}", bias);

        // Verify EWMA getter works
        assert!((tracker.ewma_bias() - bias).abs() < f64::EPSILON);

        // Verify total_observations counts all, not just window
        assert_eq!(tracker.total_observations(), 20);
    }

    // =============================================================================
    // Edge Calibration Ratio Tests
    // =============================================================================

    #[test]
    fn test_edge_health_low_rmse_guard() {
        let mut tracker = ModelConfidenceTracker::new();

        // Add 60 edge predictions with tiny error (RMSE < 0.1)
        // Predicted and realized are nearly identical → tiny RMSE, but tiny bias too
        for i in 0..60 {
            let predicted = 1.0 + (i % 3) as f64 * 0.01;
            let realized = predicted + 0.001; // Negligible difference
            tracker.record_edge_prediction(predicted, 0.5, realized);
        }

        let health = tracker.model_health();
        // Low RMSE guard should produce Good with ratio = 0.0
        assert_eq!(health.edge, Health::Good);
        assert!((health.edge_calibration_ratio - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_health_calibration_ratio_output() {
        let mut tracker = ModelConfidenceTracker::new();

        // Add 100 predictions with known bias and noise
        // Predicted = 5.0, realized varies with systematic negative bias
        // bias = predicted - realized, so positive bias = overestimation
        for i in 0..100 {
            let noise = ((i % 5) as f64 - 2.0) * 0.5; // -1.0 to 1.0
            let predicted = 5.0;
            let realized = predicted - 3.0 + noise; // systematic -3.0 bps bias
            tracker.record_edge_prediction(predicted, 0.5, realized);
        }

        let health = tracker.model_health();
        // RMSE should be well above 0.1, and |bias| / RMSE should be high (> 0.7)
        // because bias dominates noise
        assert!(health.edge_calibration_ratio > 0.7);
        assert_eq!(health.edge, Health::Degraded);
    }

    #[test]
    fn test_edge_health_calibration_ratio_good_when_noise_dominates() {
        let mut tracker = ModelConfidenceTracker::new();

        // Predictions with large random noise but near-zero mean bias
        // Alternating positive and negative errors cancel out bias
        for i in 0..100 {
            let predicted = 5.0;
            // Alternating +-2.0 errors: mean bias ≈ 0, RMSE ≈ 2.0
            let error = if i % 2 == 0 { 2.0 } else { -2.0 };
            let realized = predicted - error;
            tracker.record_edge_prediction(predicted, 0.5, realized);
        }

        let health = tracker.model_health();
        // |mean_bias| ≈ 0, RMSE ≈ 2.0, ratio ≈ 0 → Good
        assert!(health.edge_calibration_ratio < 0.5);
        assert_eq!(health.edge, Health::Good);
    }
}
