//! Level 2: Adaptive Model Selection
//!
//! Multiple edge models compete, weighted by recent performance.
//! Uses softmax of recent scores to adaptively select models.

use super::types::{
    EnsemblePrediction, MarketState, RingBuffer, TradingOutcome, WeightedPrediction,
};

/// Trait for edge prediction models.
///
/// Each model takes market state and returns (mean_edge, std_edge) in bps.
pub trait EdgeModel: Send + Sync {
    /// Predict edge in basis points.
    ///
    /// Returns (mean, std) where:
    /// - mean: expected edge capture after AS and fees
    /// - std: uncertainty in the prediction
    fn predict_edge(&self, state: &MarketState) -> (f64, f64);

    /// Model name for logging and attribution.
    fn name(&self) -> &str;
}

/// GLFT-based edge model.
///
/// Uses the Guéant-Lehalle-Fernandez-Tapia framework to predict edge.
/// Edge = half_spread - predicted_AS - fees
pub struct GLFTEdgeModel {
    /// Fee rate in bps
    fee_bps: f64,
    /// Base gamma for risk aversion
    gamma_base: f64,
}

impl Default for GLFTEdgeModel {
    fn default() -> Self {
        Self {
            fee_bps: 1.5,
            gamma_base: 0.3,
        }
    }
}

impl GLFTEdgeModel {
    /// Create a new GLFT model.
    pub fn new(fee_bps: f64, gamma_base: f64) -> Self {
        Self {
            fee_bps,
            gamma_base,
        }
    }

    /// Calculate optimal half-spread using GLFT formula.
    fn half_spread(&self, gamma: f64, kappa: f64) -> f64 {
        // δ* = (1/γ) × ln(1 + γ/κ)
        if kappa < 0.001 {
            return 0.0;
        }
        (1.0 / gamma) * (1.0 + gamma / kappa).ln()
    }
}

impl EdgeModel for GLFTEdgeModel {
    fn predict_edge(&self, state: &MarketState) -> (f64, f64) {
        // Dynamic gamma based on volatility and toxicity
        let vol_scalar = (state.sigma_effective / 0.01).clamp(0.5, 3.0);
        let toxicity_scalar = 1.0 + state.toxicity_score * 2.0;
        let gamma = self.gamma_base * vol_scalar * toxicity_scalar;

        // Calculate optimal spread
        let half_spread_frac = self.half_spread(gamma, state.kappa);
        let half_spread_bps = half_spread_frac * 10000.0;

        // Use actual quoted spread when available (Phase 2: edge at actual spread)
        // Theoretical GLFT spread may be much tighter than actual (due to floor/adjustments)
        let effective_half_spread_bps = state
            .actual_quoted_spread_bps
            .map(|s| s / 2.0)
            .unwrap_or(half_spread_bps);

        // Edge = spread - AS - fees
        let edge_mean = effective_half_spread_bps - state.predicted_as_bps - self.fee_bps;

        // Uncertainty from multiple sources
        // 1. Volatility uncertainty: per-second vol converted to bps, scaled by typical fill duration
        //    For a ~1 second fill window, uncertainty ~ sigma * sqrt(1) * 10000
        //    We use full sigma (not half) to be conservative
        let sigma_bps = state.sigma_effective * 10000.0;
        let sigma_uncertainty = sigma_bps.max(0.5); // Floor at 0.5 bps

        // 2. AS uncertainty: 30% of predicted AS
        let as_uncertainty = state.predicted_as_bps.abs() * 0.3;

        // 3. Kappa uncertainty: higher when kappa is uncertain (low kappa = more uncertainty)
        let kappa_uncertainty = if state.kappa > 100.0 {
            0.5 // Low uncertainty when kappa is high
        } else {
            2.0 / (state.kappa.max(10.0) / 100.0) // Higher uncertainty when kappa is low
        };

        // Combined uncertainty (sum of variances, capped at reasonable bounds)
        let edge_std =
            (sigma_uncertainty.powi(2) + as_uncertainty.powi(2) + kappa_uncertainty.powi(2)).sqrt();

        // Cap at reasonable bounds: min 1 bps, max 50 bps
        // Edge predictions with >50 bps uncertainty are essentially uninformative
        let edge_std_bounded = edge_std.clamp(1.0, 50.0);

        (edge_mean, edge_std_bounded)
    }

    fn name(&self) -> &str {
        "GLFT"
    }
}

/// Empirical edge model based on historical binning.
///
/// Bins market states and uses historical realized edge in each bin.
pub struct EmpiricalEdgeModel {
    /// Historical observations per bin
    bin_observations: Vec<RingBuffer<f64>>,
    /// Number of volatility bins
    #[allow(dead_code)] // Reserved for future multi-bin implementation
    n_vol_bins: usize,
    /// Number of toxicity bins
    n_tox_bins: usize,
}

impl Default for EmpiricalEdgeModel {
    fn default() -> Self {
        let n_vol_bins = 3;
        let n_tox_bins = 3;
        let n_bins = n_vol_bins * n_tox_bins;
        Self {
            bin_observations: (0..n_bins).map(|_| RingBuffer::new(100)).collect(),
            n_vol_bins,
            n_tox_bins,
        }
    }
}

impl EmpiricalEdgeModel {
    /// Get bin index for a state.
    fn bin_index(&self, state: &MarketState) -> usize {
        // Volatility bins: low, medium, high
        let vol_bin = if state.sigma_effective < 0.005 {
            0
        } else if state.sigma_effective < 0.02 {
            1
        } else {
            2
        };

        // Toxicity bins: low, medium, high
        let tox_bin = if state.toxicity_score < 0.1 {
            0
        } else if state.toxicity_score < 0.3 {
            1
        } else {
            2
        };

        vol_bin * self.n_tox_bins + tox_bin
    }

    /// Update with realized edge for learning.
    pub fn update(&mut self, state: &MarketState, realized_edge: f64) {
        let bin = self.bin_index(state);
        if bin < self.bin_observations.len() {
            self.bin_observations[bin].push(realized_edge);
        }
    }
}

impl EdgeModel for EmpiricalEdgeModel {
    fn predict_edge(&self, state: &MarketState) -> (f64, f64) {
        let bin = self.bin_index(state);

        if bin >= self.bin_observations.len() || self.bin_observations[bin].len() < 10 {
            // Not enough data, return conservative estimate
            return (0.0, 5.0);
        }

        let observations = &self.bin_observations[bin];
        let n = observations.len() as f64;

        // Mean of historical edges
        let mean: f64 = observations.iter().sum::<f64>() / n;

        // Std of historical edges
        let variance: f64 = observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        (mean, std.max(0.5))
    }

    fn name(&self) -> &str {
        "Empirical"
    }
}

/// Funding-based edge model.
///
/// Predicts edge based on funding rate mean reversion.
pub struct FundingEdgeModel {
    /// Mean reversion speed
    mean_reversion: f64,
    /// Long-term mean funding rate
    long_term_mean: f64,
}

impl Default for FundingEdgeModel {
    fn default() -> Self {
        Self {
            mean_reversion: 0.1,
            long_term_mean: 0.0,
        }
    }
}

impl EdgeModel for FundingEdgeModel {
    fn predict_edge(&self, state: &MarketState) -> (f64, f64) {
        // Funding edge from mean reversion
        // If funding is positive, shorts get paid → prefer short position
        // If funding is negative, longs get paid → prefer long position
        let funding_deviation = state.funding_rate - self.long_term_mean;

        // Expected mean reversion over holding period
        // Edge comes from funding convergence
        let expected_convergence = funding_deviation * self.mean_reversion;

        // Convert to bps (funding is typically in % per 8h)
        // Small contribution to edge
        let edge_from_funding = expected_convergence * 100.0;

        // High uncertainty in funding predictions
        let std = 3.0;

        (edge_from_funding.clamp(-5.0, 5.0), std)
    }

    fn name(&self) -> &str {
        "Funding"
    }
}

/// Model ensemble that combines multiple edge models.
pub struct ModelEnsemble {
    /// Edge models
    models: Vec<Box<dyn EdgeModel>>,
    /// Model weights (softmax of recent performance)
    weights: Vec<f64>,
    /// Recent scores per model
    model_scores: Vec<RingBuffer<f64>>,
    /// Temperature for softmax (higher = more exploration)
    temperature: f64,
    /// Minimum weight floor
    min_weight: f64,
    /// Total weight updates performed
    update_count: usize,
}

impl Default for ModelEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelEnsemble {
    /// Create a new ensemble with default models.
    pub fn new() -> Self {
        let models: Vec<Box<dyn EdgeModel>> = vec![
            Box::new(GLFTEdgeModel::default()),
            Box::new(EmpiricalEdgeModel::default()),
            Box::new(FundingEdgeModel::default()),
        ];

        let n_models = models.len();

        Self {
            models,
            weights: vec![1.0 / n_models as f64; n_models],
            model_scores: (0..n_models).map(|_| RingBuffer::new(100)).collect(),
            temperature: 1.0,
            min_weight: 0.05, // Each model gets at least 5%
            update_count: 0,
        }
    }

    /// Create ensemble with custom models.
    pub fn with_models(models: Vec<Box<dyn EdgeModel>>) -> Self {
        let n_models = models.len();

        Self {
            models,
            weights: vec![1.0 / n_models as f64; n_models],
            model_scores: (0..n_models).map(|_| RingBuffer::new(100)).collect(),
            temperature: 1.0,
            min_weight: 0.05,
            update_count: 0,
        }
    }

    /// Add a model to the ensemble at runtime.
    ///
    /// Rebalances weights equally across all models (including the new one)
    /// and allocates a fresh score ring buffer for weight adaptation.
    pub fn add_model(&mut self, model: Box<dyn EdgeModel>) {
        self.models.push(model);
        self.model_scores.push(RingBuffer::new(100));
        // Rebalance weights equally
        let n = self.models.len();
        self.weights = vec![1.0 / n as f64; n];
    }

    /// Predict edge using ensemble.
    pub fn predict_edge(&self, state: &MarketState) -> EnsemblePrediction {
        let mut contributions = Vec::with_capacity(self.models.len());
        let mut weighted_mean = 0.0;
        let mut weighted_var = 0.0;

        for (i, model) in self.models.iter().enumerate() {
            let (mean, std) = model.predict_edge(state);
            let weight = self.weights[i];

            contributions.push(WeightedPrediction {
                model: model.name().to_string(),
                mean,
                std,
                weight,
            });

            weighted_mean += mean * weight;
            weighted_var += std.powi(2) * weight;
        }

        // Variance of means (disagreement term from law of total variance)
        let mean_variance: f64 = contributions
            .iter()
            .map(|c| (c.mean - weighted_mean).powi(2) * c.weight)
            .sum();

        // Total variance = weighted variance + variance of means
        let total_std = (weighted_var + mean_variance).sqrt();
        let disagreement = mean_variance.sqrt();

        EnsemblePrediction {
            mean: weighted_mean,
            std: total_std,
            disagreement,
            model_contributions: contributions,
        }
    }

    /// Update model weights based on trading outcome.
    pub fn update_weights(&mut self, outcome: &TradingOutcome) {
        let state = &outcome.prediction.state;
        let realized = outcome.realized_edge_bps;

        // Score each model based on prediction error
        for (i, model) in self.models.iter().enumerate() {
            let (predicted, _) = model.predict_edge(state);
            // Negative squared error as score (higher = better)
            let score = -((predicted - realized).powi(2));
            self.model_scores[i].push(score);
        }

        // Compute softmax of average scores
        let avg_scores: Vec<f64> = self
            .model_scores
            .iter()
            .map(|scores| {
                if scores.is_empty() {
                    0.0
                } else {
                    scores.iter().sum::<f64>() / scores.len() as f64
                }
            })
            .collect();

        // Softmax with temperature
        let max_score = avg_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = avg_scores
            .iter()
            .map(|s| ((s - max_score) / self.temperature).exp())
            .collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        // Update weights with minimum floor
        for (i, exp_score) in exp_scores.iter().enumerate() {
            let raw_weight = exp_score / sum_exp;
            self.weights[i] = raw_weight.max(self.min_weight);
        }

        // Renormalize after floor
        let sum_weights: f64 = self.weights.iter().sum();
        for w in &mut self.weights {
            *w /= sum_weights;
        }

        self.update_count += 1;
    }

    /// Update model weights using realized per-signal Sharpe ratios.
    ///
    /// Signals with positive Sharpe get weight proportional to their Sharpe ratio.
    /// Signals with negative or zero Sharpe get the minimum weight floor.
    /// This is SNR-proportional weighting, equivalent to mutual information under Gaussian.
    pub fn update_weights_from_sharpe(&mut self, signal_weights: &[(String, f64)]) {
        if signal_weights.is_empty() {
            return; // No data — keep current weights
        }

        for (i, model) in self.models.iter().enumerate() {
            let model_name = model.name();
            if let Some((_, weight)) = signal_weights.iter().find(|(name, _)| name == model_name) {
                self.weights[i] = weight.max(self.min_weight);
            } else {
                // Signal not in positive Sharpe set — use minimum weight
                self.weights[i] = self.min_weight;
            }
        }

        // Renormalize
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    /// Get current model weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get model names.
    pub fn model_names(&self) -> Vec<&str> {
        self.models.iter().map(|m| m.name()).collect()
    }

    /// Set temperature for exploration/exploitation.
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.1);
    }

    /// Apply a health penalty to a specific model's weight.
    ///
    /// `penalty_factor`: 0.5 for degraded signals, 0.1 for stale signals.
    /// The weights are renormalized after applying the penalty.
    pub fn apply_health_penalty(&mut self, model_index: usize, penalty_factor: f64) {
        if model_index < self.weights.len() {
            self.weights[model_index] *= penalty_factor;
            // Renormalize
            let sum: f64 = self.weights.iter().sum();
            if sum > 0.0 {
                for w in &mut self.weights {
                    *w /= sum;
                }
            }
        }
    }

    /// Get current model weights (cloned, for checkpoint persistence).
    pub fn current_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Get total weight updates performed.
    pub fn total_updates(&self) -> usize {
        self.update_count
    }

    /// Restore weights from checkpoint data.
    ///
    /// Only applies if the weight vector length matches the number of models.
    pub fn restore_weights(&mut self, weights: &[f64], total_updates: usize) {
        if weights.len() == self.weights.len() {
            self.weights.copy_from_slice(weights);
            self.update_count = total_updates;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glft_model() {
        let model = GLFTEdgeModel::default();
        let state = MarketState::default();

        let (mean, std) = model.predict_edge(&state);
        assert!(std > 0.0);
        // With default kappa=1.0 and reasonable params, should have some edge
        println!("GLFT prediction: mean={mean:.2}bp, std={std:.2}bp");
    }

    #[test]
    fn test_ensemble_prediction() {
        let ensemble = ModelEnsemble::new();
        let state = MarketState::default();

        let pred = ensemble.predict_edge(&state);
        assert_eq!(pred.model_contributions.len(), 3);

        // Weights should sum to 1
        let sum: f64 = pred.model_contributions.iter().map(|c| c.weight).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ensemble_weight_update() {
        let mut ensemble = ModelEnsemble::new();
        let state = MarketState::default();

        // Simulate outcomes where GLFT is consistently better
        for _ in 0..50 {
            let (glft_pred, _) = ensemble.models[0].predict_edge(&state);

            // GLFT prediction is close to reality
            let realized = glft_pred + 0.1;

            let outcome = TradingOutcome {
                prediction: crate::market_maker::learning::types::PendingPrediction {
                    timestamp_ms: 0,
                    fill: crate::market_maker::fills::FillEvent::new(
                        0,
                        0,
                        0.1,
                        100.0,
                        true,
                        100.0,
                        Some(100.5),
                        "BTC".to_string(),
                    ),
                    predicted_edge_bps: glft_pred,
                    predicted_uncertainty: 1.0,
                    state: state.clone(),
                    depth_bps: 5.0,
                    predicted_fill_prob: 0.8,
                },
                realized_as_bps: 1.0,
                realized_edge_bps: realized,
                price_at_horizon: 100.0,
                horizon_elapsed_ms: 1000,
            };

            ensemble.update_weights(&outcome);
        }

        // GLFT should have highest weight now
        let weights = ensemble.weights();
        assert!(weights[0] > weights[1]);
        assert!(weights[0] > weights[2]);
    }

    #[test]
    fn test_glft_edge_uses_actual_spread() {
        let model = GLFTEdgeModel::default();

        // Without actual spread: uses theoretical GLFT spread
        let state_no_actual = MarketState {
            kappa: 3250.0,
            sigma_effective: 0.01,
            predicted_as_bps: 2.0,
            toxicity_score: 0.0,
            actual_quoted_spread_bps: None,
            ..Default::default()
        };
        let (edge_theoretical, _) = model.predict_edge(&state_no_actual);

        // With actual spread at 14.84 bps (7.42 per side): edge should be positive
        let state_actual = MarketState {
            actual_quoted_spread_bps: Some(14.84),
            ..state_no_actual
        };
        let (edge_actual, _) = model.predict_edge(&state_actual);

        // Edge at actual spread should be higher than at theoretical
        assert!(
            edge_actual > edge_theoretical,
            "edge at actual spread ({:.2}) should exceed theoretical ({:.2})",
            edge_actual,
            edge_theoretical
        );

        // Edge at actual spread should be positive (7.42 - 2.0 - 1.5 = 3.92 bps)
        assert!(
            edge_actual > 0.0,
            "edge at actual spread should be positive: {:.2}",
            edge_actual
        );
    }

    #[test]
    fn test_glft_edge_fallback_when_no_actual_spread() {
        let model = GLFTEdgeModel::default();
        let state = MarketState {
            kappa: 3250.0,
            sigma_effective: 0.01,
            predicted_as_bps: 2.0,
            toxicity_score: 0.0,
            actual_quoted_spread_bps: None,
            ..Default::default()
        };
        let (edge, _) = model.predict_edge(&state);
        // Should still work without actual spread (backward compat)
        assert!(edge.is_finite(), "edge should be finite");
    }

    #[test]
    fn test_update_weights_from_sharpe() {
        let mut ensemble = ModelEnsemble::new();
        // Default has 3 models: GLFT, Empirical, Funding

        let signal_weights = vec![
            ("GLFT".to_string(), 0.7),
            ("Empirical".to_string(), 0.3),
            // Funding not present — gets min_weight
        ];

        ensemble.update_weights_from_sharpe(&signal_weights);

        let w = ensemble.weights();
        // GLFT should have highest weight
        assert!(
            w[0] > w[2],
            "GLFT should outweigh Funding: {} vs {}",
            w[0],
            w[2]
        );
        // All weights should be positive
        for &wi in w {
            assert!(wi > 0.0, "All weights should be positive");
        }
        // Weights should sum to 1.0
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Weights should sum to 1.0: {sum}");
    }

    #[test]
    fn test_update_weights_from_sharpe_empty() {
        let mut ensemble = ModelEnsemble::new();
        let original_weights = ensemble.weights().to_vec();

        ensemble.update_weights_from_sharpe(&[]);

        // Should keep original weights when no data
        assert_eq!(ensemble.weights(), &original_weights[..]);
    }
}
