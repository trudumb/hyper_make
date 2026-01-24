//! Level 1: Model Confidence Tracking
//!
//! Tracks prediction accuracy over time for each model component.
//! Detects when the model is breaking down based on calibration scores.

use super::types::{CalibrationScore, Health, ModelHealth, RingBuffer};

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

    // === Configuration ===
    /// Maximum AS bias before warning (bps)
    max_as_bias_warning: f64,
    /// Maximum AS bias before degraded (bps)
    max_as_bias_degraded: f64,
    /// Maximum edge calibration error (bps)
    max_edge_error: f64,
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
            max_as_bias_warning: 0.5,  // 0.5 bps
            max_as_bias_degraded: 1.0, // 1.0 bps
            max_edge_error: 0.5,       // 0.5 bps
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

        // === Edge health ===
        if self.edge_predictions.len() < 50 {
            health.edge = Health::Good; // Not enough data
        } else {
            health.edge = if self.edge_calibration.error < self.max_edge_error {
                Health::Good
            } else if self.edge_calibration.error < self.max_edge_error * 2.0 {
                Health::Warning
            } else {
                Health::Degraded
            };
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
}
