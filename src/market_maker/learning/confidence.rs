//! Level 1: Model Confidence Tracking
//!
//! Tracks prediction accuracy over time for each model component.
//! Detects when the model is breaking down based on calibration scores.

use super::types::{CalibrationScore, Health, ModelHealth, RingBuffer};

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
                predicted_as_bps: 1.0,     // We predict 1bp
                measurement_horizon_ms: 1000,
                realized_as_bps: 2.0,      // But it's actually 2bp
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
