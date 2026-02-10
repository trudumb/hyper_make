//! Interface types between Layer 2 (LearningModule) and Layer 3 (StochasticController).
//!
//! This module defines the data structures that flow from the learning system
//! to the stochastic controller.

use crate::market_maker::learning::{
    CalibrationScore, EnsemblePrediction, ModelHealth, QuoteDecision, WeightedPrediction,
};

/// Complete output from the LearningModule for Layer 3 consumption.
///
/// This bundles everything Layer 2 knows about the current market state
/// and model predictions.
#[derive(Debug, Clone)]
pub struct LearningModuleOutput {
    // === From ModelEnsemble ===
    /// Edge prediction with uncertainty
    pub edge_prediction: GaussianEstimate,

    /// Individual model predictions for belief updates
    pub model_predictions: Vec<ModelPrediction>,

    /// Disagreement between models (epistemic uncertainty)
    pub model_disagreement: f64,

    // === From ModelConfidenceTracker ===
    /// Overall model health
    pub model_health: ModelHealth,

    /// Calibration score (are predictions accurate?)
    pub calibration: CalibrationScore,

    /// Bias in AS estimation (positive = underestimating = dangerous)
    pub as_bias: f64,

    // === From DecisionEngine ===
    /// Myopic decision (what Layer 2 would do without sequential optimization)
    pub myopic_decision: QuoteDecision,

    /// Confidence in positive edge (probability)
    pub p_positive_edge: f64,

    // === Additional state ===
    /// Current position
    pub position: f64,

    /// Maximum allowed position
    pub max_position: f64,

    /// Current drawdown fraction
    pub drawdown: f64,
}

impl Default for LearningModuleOutput {
    fn default() -> Self {
        Self {
            edge_prediction: GaussianEstimate::default(),
            model_predictions: Vec::new(),
            model_disagreement: 0.0,
            model_health: ModelHealth::default(),
            calibration: CalibrationScore::default(),
            as_bias: 0.0,
            myopic_decision: QuoteDecision::NoQuote {
                reason: "default".to_string(),
            },
            p_positive_edge: 0.5,
            position: 0.0,
            max_position: 1.0,
            drawdown: 0.0,
        }
    }
}

impl LearningModuleOutput {
    /// Check if the learning module is trustworthy.
    pub fn is_trustworthy(&self) -> bool {
        !self.model_health.is_degraded()
            && self.calibration.is_calibrated(2.0) // 2 bps error threshold
            && self.as_bias < 1.0 // Not underestimating AS
    }

    /// Get effective edge considering uncertainty.
    pub fn conservative_edge(&self, confidence: f64) -> f64 {
        // Lower bound of confidence interval
        self.edge_prediction.percentile(1.0 - confidence)
    }

    /// Position utilization (how close to max).
    pub fn position_utilization(&self) -> f64 {
        if self.max_position > 0.0 {
            self.position.abs() / self.max_position
        } else {
            0.0
        }
    }
}

/// Gaussian estimate with mean and standard deviation.
///
/// Used for edge predictions and other uncertain quantities.
#[derive(Debug, Clone, Copy)]
pub struct GaussianEstimate {
    /// Mean of the distribution
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
}

impl Default for GaussianEstimate {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
        }
    }
}

impl GaussianEstimate {
    /// Create a new Gaussian estimate.
    pub fn new(mean: f64, std: f64) -> Self {
        Self {
            mean,
            std: std.max(1e-9),
        }
    }

    /// Symmetric confidence interval.
    pub fn confidence_interval(&self, z: f64) -> (f64, f64) {
        (self.mean - z * self.std, self.mean + z * self.std)
    }

    /// Probability that the value is positive.
    pub fn p_positive(&self) -> f64 {
        use super::types::normal_cdf;
        normal_cdf(self.mean / self.std)
    }

    /// Probability that value exceeds threshold.
    pub fn p_greater_than(&self, threshold: f64) -> f64 {
        use super::types::normal_cdf;
        normal_cdf((self.mean - threshold) / self.std)
    }

    /// Percentile of the distribution.
    pub fn percentile(&self, p: f64) -> f64 {
        use super::types::normal_quantile;
        self.mean + self.std * normal_quantile(p)
    }

    /// 95% credible interval.
    pub fn ci95(&self) -> (f64, f64) {
        self.confidence_interval(1.96)
    }

    /// Coefficient of variation (relative uncertainty).
    pub fn cv(&self) -> f64 {
        if self.mean.abs() > 1e-9 {
            self.std / self.mean.abs()
        } else {
            f64::INFINITY
        }
    }
}

impl From<EnsemblePrediction> for GaussianEstimate {
    fn from(pred: EnsemblePrediction) -> Self {
        Self::new(pred.mean, pred.std)
    }
}

/// Single model prediction with metadata.
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    /// Model name
    pub name: String,
    /// Predicted edge (mean)
    pub mean: f64,
    /// Prediction uncertainty
    pub std: f64,
    /// Model weight in ensemble
    pub weight: f64,
    /// Recent performance score [0, 1]
    pub performance: f64,
}

impl From<WeightedPrediction> for ModelPrediction {
    fn from(wp: WeightedPrediction) -> Self {
        Self {
            name: wp.model,
            mean: wp.mean,
            std: wp.std,
            weight: wp.weight,
            performance: 1.0, // Default, should be updated
        }
    }
}

/// Trading state snapshot for controller decisions.
#[derive(Debug, Clone)]
pub struct TradingState {
    /// Current wealth (unrealized + realized P&L)
    pub wealth: f64,
    /// Current position
    pub position: f64,
    /// Current margin used
    pub margin_used: f64,
    /// Session time as fraction [0, 1]
    pub session_time: f64,
    /// Time until next funding (hours)
    pub time_to_funding: f64,
    /// Predicted funding rate (bps)
    pub predicted_funding: f64,
    /// Current drawdown fraction
    pub drawdown: f64,
    /// Is in reduce-only mode
    pub reduce_only: bool,
    /// Rate limit headroom fraction [0, 1]
    pub rate_limit_headroom: f64,
    /// Last realized edge from a fill (bps). Updated after each fill.
    /// Used by the controller as the TD reward signal for Quote actions.
    pub last_realized_edge_bps: f64,
    /// Current market spread (bps). Used by the controller to compute
    /// spread crossing cost for DumpInventory actions.
    pub market_spread_bps: f64,
}

impl Default for TradingState {
    fn default() -> Self {
        Self {
            wealth: 0.0,
            position: 0.0,
            margin_used: 0.0,
            session_time: 0.0,
            time_to_funding: 8.0,
            predicted_funding: 0.0,
            drawdown: 0.0,
            reduce_only: false,
            rate_limit_headroom: 1.0,
            last_realized_edge_bps: 0.0,
            market_spread_bps: 0.0,
        }
    }
}

impl TradingState {
    /// Fraction of session remaining.
    pub fn time_remaining(&self) -> f64 {
        (1.0 - self.session_time).max(0.0)
    }

    /// Whether we're in terminal zone (near session end).
    pub fn is_terminal_zone(&self, threshold: f64) -> bool {
        self.session_time > threshold
    }

    /// Whether funding opportunity exists.
    pub fn has_funding_opportunity(&self, threshold_bps: f64) -> bool {
        self.time_to_funding < 1.0 && self.predicted_funding.abs() > threshold_bps
    }

    /// Margin utilization fraction.
    pub fn margin_utilization(&self) -> f64 {
        // Assume some max margin (this should come from config)
        self.margin_used / 10000.0 // Placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_estimate() {
        let g = GaussianEstimate::new(1.0, 0.5);

        assert!(g.p_positive() > 0.95); // 2 sigma above 0

        let (lo, hi) = g.ci95();
        assert!(lo < 1.0 && hi > 1.0);
    }

    #[test]
    fn test_learning_output_trustworthy() {
        let mut output = LearningModuleOutput::default();
        output.calibration.n_observations = 100;
        output.calibration.error = 1.0;

        assert!(output.is_trustworthy());

        output.as_bias = 2.0; // Underestimating AS
        assert!(!output.is_trustworthy());
    }

    #[test]
    fn test_trading_state() {
        let state = TradingState {
            session_time: 0.97,
            time_to_funding: 0.5,
            predicted_funding: 10.0,
            ..Default::default()
        };

        assert!(state.is_terminal_zone(0.95));
        assert!(state.has_funding_opportunity(5.0));
    }
}
