//! Adaptive Model Ensemble with Confidence Weighting
//!
//! This module implements a dynamic model weighting system based on recent calibration
//! performance. Models are weighted using a softmax over their Information Ratios (IR),
//! with temperature control for exploration/exploitation tradeoff.
//!
//! ## Key Features
//!
//! - **Information Ratio-based weighting**: Models with better prediction accuracy get higher weights
//! - **Temperature-controlled softmax**: Higher temperature = more uniform weights (exploration)
//! - **Minimum weight floor**: Prevents complete exclusion of any model
//! - **Decay for non-stationarity**: Old observations fade over time
//! - **Integration with BeliefState**: Updates Bayesian belief model weights
//!
//! ## Information Ratio
//!
//! IR = Resolution / Uncertainty
//! - IR > 1.0: Model adds value (predictions are better than base rate)
//! - IR < 1.0: Model adds noise (should be downweighted or removed)
//! - IR = 1.0: Model is at break-even

use std::collections::HashMap;

use crate::market_maker::control::belief::BeliefState;

/// Performance metrics for a single model in the ensemble.
#[derive(Debug, Clone)]
pub struct ModelPerformance {
    /// Information Ratio: Resolution / Uncertainty (>1.0 means model adds value)
    pub information_ratio: f64,
    /// Brier Score: Mean squared error of probability predictions
    pub brier_score: f64,
    /// Number of predictions evaluated
    pub n_predictions: usize,
    /// Timestamp of last performance update (milliseconds)
    pub last_update_ms: u64,
    /// Computed weight after softmax and floor application
    pub weight: f64,
}

impl Default for ModelPerformance {
    fn default() -> Self {
        Self {
            information_ratio: 1.0, // Neutral prior
            brier_score: 0.25,      // Prior for 50% predictions
            n_predictions: 0,
            last_update_ms: 0,
            weight: 0.0,
        }
    }
}

impl ModelPerformance {
    /// Create a new ModelPerformance with initial values.
    pub fn new(information_ratio: f64, brier_score: f64) -> Self {
        Self {
            information_ratio,
            brier_score,
            n_predictions: 0,
            last_update_ms: 0,
            weight: 0.0,
        }
    }

    /// Check if the model has degraded (IR < 1.0).
    pub fn is_degraded(&self) -> bool {
        self.information_ratio < 1.0
    }

    /// Age of last update in milliseconds (from current timestamp).
    pub fn age_ms(&self, current_ms: u64) -> u64 {
        current_ms.saturating_sub(self.last_update_ms)
    }
}

/// Summary of ensemble state for logging and monitoring.
#[derive(Debug, Clone)]
pub struct EnsembleSummary {
    /// Total number of models in the ensemble
    pub total_models: usize,
    /// Number of models with weight > min_weight
    pub active_models: usize,
    /// Number of models with IR < 1.0
    pub degraded_models: usize,
    /// Name of the best-performing model
    pub best_model_name: Option<String>,
    /// IR of the best-performing model
    pub best_model_ir: f64,
    /// Shannon entropy of weight distribution (bits)
    /// Higher entropy = more uniform distribution
    /// Lower entropy = one model dominates
    pub weight_entropy: f64,
    /// Average IR across all models
    pub average_ir: f64,
    /// Total predictions across all models
    pub total_predictions: usize,
}

impl Default for EnsembleSummary {
    fn default() -> Self {
        Self {
            total_models: 0,
            active_models: 0,
            degraded_models: 0,
            best_model_name: None,
            best_model_ir: 0.0,
            weight_entropy: 0.0,
            average_ir: 1.0,
            total_predictions: 0,
        }
    }
}

/// Adaptive ensemble that dynamically weights models based on recent performance.
///
/// Models are weighted using softmax over Information Ratios with temperature control.
/// This allows for both exploitation (low temperature, best model dominates) and
/// exploration (high temperature, more uniform weights).
#[derive(Debug, Clone)]
pub struct AdaptiveEnsemble {
    /// Model name -> performance metrics
    models: HashMap<String, ModelPerformance>,
    /// Softmax temperature for weight calculation
    /// Higher = more uniform weights (exploration)
    /// Lower = best model dominates (exploitation)
    temperature: f64,
    /// Minimum weight floor (prevents complete exclusion)
    min_weight: f64,
    /// Prior weight for new/cold models (not enough predictions)
    /// Reserved for future use in Bayesian model averaging scenarios.
    #[allow(dead_code)]
    prior_weight: f64,
    /// Decay factor for old observations (0-1, applied per update)
    decay_rate: f64,
    /// Minimum predictions before using model's IR (otherwise use prior)
    min_predictions_for_weight: usize,
}

impl Default for AdaptiveEnsemble {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl AdaptiveEnsemble {
    /// Create a new AdaptiveEnsemble with the specified temperature.
    ///
    /// # Arguments
    /// * `temperature` - Softmax temperature (recommended: 0.5-2.0)
    ///   - Low (0.5): Best model gets most weight
    ///   - Medium (1.0): Balanced weighting
    ///   - High (2.0): More uniform weights
    pub fn new(temperature: f64) -> Self {
        Self {
            models: HashMap::new(),
            temperature: temperature.max(0.01), // Prevent division by zero
            min_weight: 0.05,                   // Each model gets at least 5%
            prior_weight: 0.2,                  // New models start with 20%
            decay_rate: 0.995,                  // 0.5% decay per update
            min_predictions_for_weight: 20,     // Need 20 predictions to use model's IR
        }
    }

    /// Create ensemble with custom parameters.
    pub fn with_params(
        temperature: f64,
        min_weight: f64,
        prior_weight: f64,
        decay_rate: f64,
    ) -> Self {
        Self {
            models: HashMap::new(),
            temperature: temperature.max(0.01),
            min_weight: min_weight.clamp(0.0, 0.5),
            prior_weight: prior_weight.clamp(0.0, 1.0),
            decay_rate: decay_rate.clamp(0.9, 1.0),
            min_predictions_for_weight: 20,
        }
    }

    /// Register a new model in the ensemble.
    ///
    /// New models start with default performance metrics and will be assigned
    /// the prior_weight until they have enough predictions.
    pub fn register_model(&mut self, name: &str) {
        if !self.models.contains_key(name) {
            self.models
                .insert(name.to_string(), ModelPerformance::default());
            self.compute_weights();
        }
    }

    /// Update performance metrics for a model.
    ///
    /// # Arguments
    /// * `name` - Model name
    /// * `ir` - Information Ratio (>1.0 means model adds value)
    /// * `brier` - Brier Score (lower is better)
    /// * `timestamp` - Current timestamp in milliseconds
    pub fn update_performance(&mut self, name: &str, ir: f64, brier: f64, timestamp: u64) {
        let perf = self.models.entry(name.to_string()).or_default();

        // Apply decay to blend old and new observations
        let alpha = 1.0 - self.decay_rate; // Learning rate
        perf.information_ratio = perf.information_ratio * self.decay_rate + ir * alpha;
        perf.brier_score = perf.brier_score * self.decay_rate + brier * alpha;
        perf.n_predictions += 1;
        perf.last_update_ms = timestamp;

        self.compute_weights();
    }

    /// Compute weights for all models using softmax over IR.
    ///
    /// Weight calculation:
    /// 1. Compute softmax: w[i] = exp(IR[i] / temperature) / sum(exp(IR[j] / temperature))
    /// 2. Apply minimum floor using water-filling algorithm:
    ///    - Models below floor get exactly floor
    ///    - Remaining weight is distributed proportionally among other models
    /// This guarantees all models have weight >= min_weight after normalization.
    pub fn compute_weights(&mut self) {
        if self.models.is_empty() {
            return;
        }

        let n = self.models.len();

        // Check if min_weight is feasible (n * min_weight <= 1.0)
        let effective_min_weight = self.min_weight.min(1.0 / n as f64);

        // Collect (name, effective_ir) pairs
        // Use prior_weight's equivalent IR for cold models
        let irs: Vec<(String, f64)> = self
            .models
            .iter()
            .map(|(name, perf)| {
                let effective_ir = if perf.n_predictions >= self.min_predictions_for_weight {
                    perf.information_ratio
                } else {
                    // Cold model: use a neutral IR of 1.0
                    1.0
                };
                (name.clone(), effective_ir)
            })
            .collect();

        // Compute softmax with temperature
        // Use log-sum-exp trick for numerical stability
        let max_ir = irs
            .iter()
            .map(|(_, ir)| *ir)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_scores: Vec<f64> = irs
            .iter()
            .map(|(_, ir)| ((ir - max_ir) / self.temperature).exp())
            .collect();

        let sum_exp: f64 = exp_scores.iter().sum();

        // Compute raw weights from softmax
        let raw_weights: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Water-filling algorithm to apply floor while maintaining normalization
        // 1. Identify models that would be below floor
        // 2. Give them the floor, redistribute remaining weight to others
        let mut final_weights = raw_weights.clone();
        let mut remaining_weight = 1.0;
        let mut fixed_indices: Vec<bool> = vec![false; n];

        // Iterate until stable (at most n iterations)
        for _ in 0..n {
            // Count non-fixed models and their total weight
            let free_indices: Vec<usize> = (0..n).filter(|&i| !fixed_indices[i]).collect();
            if free_indices.is_empty() {
                break;
            }

            let free_raw_sum: f64 = free_indices.iter().map(|&i| raw_weights[i]).sum();

            // Check which free models would be below floor after proportional scaling
            let mut any_fixed = false;
            for &i in &free_indices {
                let scaled_weight = if free_raw_sum > 0.0 {
                    raw_weights[i] / free_raw_sum * remaining_weight
                } else {
                    remaining_weight / free_indices.len() as f64
                };

                if scaled_weight < effective_min_weight {
                    // Fix this model at the floor
                    final_weights[i] = effective_min_weight;
                    fixed_indices[i] = true;
                    remaining_weight -= effective_min_weight;
                    any_fixed = true;
                }
            }

            if !any_fixed {
                // All remaining models are above floor, distribute proportionally
                for &i in &free_indices {
                    let scaled_weight = if free_raw_sum > 0.0 {
                        raw_weights[i] / free_raw_sum * remaining_weight
                    } else {
                        remaining_weight / free_indices.len() as f64
                    };
                    final_weights[i] = scaled_weight;
                }
                break;
            }
        }

        // Update model weights
        for ((name, _), weight) in irs.iter().zip(final_weights.iter()) {
            if let Some(perf) = self.models.get_mut(name) {
                perf.weight = *weight;
            }
        }
    }

    /// Get the weight for a specific model.
    ///
    /// Returns 0.0 if the model is not registered.
    pub fn get_weight(&self, name: &str) -> f64 {
        self.models.get(name).map(|p| p.weight).unwrap_or(0.0)
    }

    /// Compute weighted average of predictions from multiple models.
    ///
    /// # Arguments
    /// * `predictions` - Map of model name -> prediction value
    ///
    /// # Returns
    /// Weighted average of predictions. Models not in the ensemble are ignored.
    pub fn weighted_average(&self, predictions: &HashMap<String, f64>) -> f64 {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for (name, pred) in predictions {
            if let Some(perf) = self.models.get(name) {
                sum += pred * perf.weight;
                weight_sum += perf.weight;
            }
        }

        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            // Fallback to simple average if no weights available
            if predictions.is_empty() {
                0.0
            } else {
                predictions.values().sum::<f64>() / predictions.len() as f64
            }
        }
    }

    /// Check if a specific model is degraded (IR < 1.0).
    ///
    /// Returns false if the model is not registered or has too few predictions.
    pub fn is_model_degraded(&self, name: &str) -> bool {
        self.models
            .get(name)
            .map(|p| p.n_predictions >= self.min_predictions_for_weight && p.is_degraded())
            .unwrap_or(false)
    }

    /// Get the name of the best-performing model.
    ///
    /// Returns None if no models are registered.
    pub fn best_model(&self) -> Option<&str> {
        self.models
            .iter()
            .filter(|(_, p)| p.n_predictions >= self.min_predictions_for_weight)
            .max_by(|(_, a), (_, b)| {
                a.information_ratio
                    .partial_cmp(&b.information_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.as_str())
    }

    /// Get a summary of the ensemble state.
    pub fn summary(&self) -> EnsembleSummary {
        let total_models = self.models.len();

        if total_models == 0 {
            return EnsembleSummary::default();
        }

        let active_models = self
            .models
            .values()
            .filter(|p| p.weight > self.min_weight)
            .count();

        let degraded_models = self
            .models
            .values()
            .filter(|p| p.n_predictions >= self.min_predictions_for_weight && p.is_degraded())
            .count();

        let (best_model_name, best_model_ir) = self
            .models
            .iter()
            .filter(|(_, p)| p.n_predictions >= self.min_predictions_for_weight)
            .max_by(|(_, a), (_, b)| {
                a.information_ratio
                    .partial_cmp(&b.information_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, p)| (Some(name.clone()), p.information_ratio))
            .unwrap_or((None, 0.0));

        // Compute Shannon entropy of weight distribution (in bits)
        // H = -sum(w * log2(w))
        let weight_entropy: f64 = self
            .models
            .values()
            .filter(|p| p.weight > 0.0)
            .map(|p| {
                if p.weight > 0.0 {
                    -p.weight * p.weight.log2()
                } else {
                    0.0
                }
            })
            .sum();

        let qualified_models: Vec<_> = self
            .models
            .values()
            .filter(|p| p.n_predictions >= self.min_predictions_for_weight)
            .collect();

        let average_ir = if qualified_models.is_empty() {
            1.0
        } else {
            qualified_models
                .iter()
                .map(|p| p.information_ratio)
                .sum::<f64>()
                / qualified_models.len() as f64
        };

        let total_predictions: usize = self.models.values().map(|p| p.n_predictions).sum();

        EnsembleSummary {
            total_models,
            active_models,
            degraded_models,
            best_model_name,
            best_model_ir,
            weight_entropy,
            average_ir,
            total_predictions,
        }
    }

    /// Apply ensemble weights to a BeliefState's model_weights field.
    ///
    /// This integrates the adaptive ensemble with the Bayesian belief system.
    pub fn apply_to_belief(&self, belief: &mut BeliefState) {
        // Get model names in a consistent order
        let mut model_names: Vec<&str> = self.models.keys().map(|s| s.as_str()).collect();
        model_names.sort();

        // Resize belief's Dirichlet if needed
        if belief.model_weights.k() != model_names.len() {
            belief.model_weights =
                crate::market_maker::control::types::DirichletPosterior::uniform(model_names.len());
        }

        // Update belief weights based on ensemble weights
        // Use a pseudo-observation approach: weight * scale factor
        let scale = 10.0; // Strength of ensemble influence
        for (i, name) in model_names.iter().enumerate() {
            let weight = self.get_weight(name);
            belief.model_weights.update_weighted(i, weight * scale);
        }
    }

    /// Get performance metrics for a specific model.
    pub fn get_performance(&self, name: &str) -> Option<&ModelPerformance> {
        self.models.get(name)
    }

    /// Get all model names.
    pub fn model_names(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered models.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Set the softmax temperature.
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.01);
        self.compute_weights();
    }

    /// Get the current temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get the minimum weight floor.
    pub fn min_weight(&self) -> f64 {
        self.min_weight
    }

    /// Set the minimum weight floor.
    pub fn set_min_weight(&mut self, min_weight: f64) {
        self.min_weight = min_weight.clamp(0.0, 0.5);
        self.compute_weights();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_ensemble() {
        let ensemble = AdaptiveEnsemble::new(1.0);
        assert_eq!(ensemble.temperature(), 1.0);
        assert_eq!(ensemble.model_count(), 0);
    }

    #[test]
    fn test_register_model() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("GLFT");
        ensemble.register_model("Empirical");

        assert_eq!(ensemble.model_count(), 2);
        assert!(ensemble.model_names().contains(&"GLFT"));
        assert!(ensemble.model_names().contains(&"Empirical"));
    }

    #[test]
    fn test_weight_calculation_uniform() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("A");
        ensemble.register_model("B");
        ensemble.register_model("C");

        // With no updates, all models have IR=1.0 (neutral)
        // Weights should be approximately equal (adjusted for floor)
        let wa = ensemble.get_weight("A");
        let wb = ensemble.get_weight("B");
        let wc = ensemble.get_weight("C");

        // Weights should sum to 1.0
        assert!((wa + wb + wc - 1.0).abs() < 1e-9);

        // With same IR, weights should be approximately equal
        assert!((wa - wb).abs() < 0.01);
        assert!((wb - wc).abs() < 0.01);
    }

    #[test]
    fn test_weight_calculation_with_different_ir() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("Good");
        ensemble.register_model("Bad");

        // Update with enough predictions
        let timestamp = 1000u64;
        for _ in 0..25 {
            ensemble.update_performance("Good", 2.0, 0.1, timestamp);
            ensemble.update_performance("Bad", 0.5, 0.4, timestamp);
        }

        // Good model should have higher weight
        let w_good = ensemble.get_weight("Good");
        let w_bad = ensemble.get_weight("Bad");

        assert!(w_good > w_bad, "Good model should have higher weight");

        // Weights should sum to 1.0
        assert!((w_good + w_bad - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_temperature_effect_low() {
        let mut ensemble = AdaptiveEnsemble::new(0.1); // Very low temperature
        ensemble.register_model("Good");
        ensemble.register_model("Bad");

        for _ in 0..25 {
            ensemble.update_performance("Good", 2.0, 0.1, 1000);
            ensemble.update_performance("Bad", 0.5, 0.4, 1000);
        }

        // With low temperature, good model should dominate
        let w_good = ensemble.get_weight("Good");
        assert!(
            w_good > 0.8,
            "Low temperature should make best model dominate, got {w_good}"
        );
    }

    #[test]
    fn test_temperature_effect_high() {
        let mut ensemble = AdaptiveEnsemble::new(5.0); // High temperature
        ensemble.register_model("Good");
        ensemble.register_model("Bad");

        for _ in 0..25 {
            ensemble.update_performance("Good", 2.0, 0.1, 1000);
            ensemble.update_performance("Bad", 0.5, 0.4, 1000);
        }

        // With high temperature, weights should be more uniform
        let w_good = ensemble.get_weight("Good");
        let w_bad = ensemble.get_weight("Bad");

        // Difference should be smaller than with low temperature
        assert!(
            (w_good - w_bad).abs() < 0.3,
            "High temperature should make weights more uniform"
        );
    }

    #[test]
    fn test_minimum_weight_floor() {
        let mut ensemble = AdaptiveEnsemble::new(0.01); // Very low temperature
        ensemble.register_model("Best");
        ensemble.register_model("Worst");
        ensemble.register_model("Middle");

        for _ in 0..25 {
            ensemble.update_performance("Best", 10.0, 0.01, 1000);
            ensemble.update_performance("Worst", 0.1, 0.9, 1000);
            ensemble.update_performance("Middle", 1.0, 0.25, 1000);
        }

        // Even the worst model should have at least min_weight
        let w_worst = ensemble.get_weight("Worst");
        assert!(
            w_worst >= ensemble.min_weight(),
            "Worst model weight {w_worst} should be >= min_weight {}",
            ensemble.min_weight()
        );
    }

    #[test]
    fn test_model_degradation_detection() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("Good");
        ensemble.register_model("Bad");

        // Not enough predictions yet
        assert!(!ensemble.is_model_degraded("Good"));
        assert!(!ensemble.is_model_degraded("Bad"));

        // Update with enough predictions
        for _ in 0..25 {
            ensemble.update_performance("Good", 1.5, 0.15, 1000);
            ensemble.update_performance("Bad", 0.7, 0.35, 1000);
        }

        assert!(
            !ensemble.is_model_degraded("Good"),
            "IR=1.5 is not degraded"
        );
        assert!(
            ensemble.is_model_degraded("Bad"),
            "IR=0.7 should be degraded"
        );
    }

    #[test]
    fn test_weighted_average() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("A");
        ensemble.register_model("B");

        // Update to get different weights
        for _ in 0..25 {
            ensemble.update_performance("A", 2.0, 0.1, 1000);
            ensemble.update_performance("B", 1.0, 0.25, 1000);
        }

        let mut predictions = HashMap::new();
        predictions.insert("A".to_string(), 10.0);
        predictions.insert("B".to_string(), 5.0);

        let avg = ensemble.weighted_average(&predictions);

        // A has higher weight, so average should be closer to 10 than 5
        let w_a = ensemble.get_weight("A");
        let expected = 10.0 * w_a + 5.0 * (1.0 - w_a);
        assert!(
            (avg - expected).abs() < 0.01,
            "Weighted average should be {expected}, got {avg}"
        );
    }

    #[test]
    fn test_best_model() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("GLFT");
        ensemble.register_model("Empirical");
        ensemble.register_model("Funding");

        // Update performance
        for _ in 0..25 {
            ensemble.update_performance("GLFT", 1.8, 0.12, 1000);
            ensemble.update_performance("Empirical", 1.5, 0.18, 1000);
            ensemble.update_performance("Funding", 0.9, 0.28, 1000);
        }

        assert_eq!(ensemble.best_model(), Some("GLFT"));
    }

    #[test]
    fn test_summary() {
        // Use custom params with faster decay for testing
        let mut ensemble = AdaptiveEnsemble::with_params(1.0, 0.05, 0.2, 0.9);
        ensemble.register_model("Good");
        ensemble.register_model("Bad");
        ensemble.register_model("Cold"); // No updates

        // More iterations with faster decay (0.9) to converge to target IR
        for _ in 0..50 {
            ensemble.update_performance("Good", 1.8, 0.12, 1000);
            ensemble.update_performance("Bad", 0.6, 0.4, 1000);
        }

        let summary = ensemble.summary();

        assert_eq!(summary.total_models, 3);
        assert_eq!(summary.degraded_models, 1); // "Bad" is degraded
        assert_eq!(summary.best_model_name, Some("Good".to_string()));
        // With decay 0.9 and 50 iterations, IR should be close to target
        assert!(
            summary.best_model_ir > 1.5,
            "Best model IR should be > 1.5, got {}",
            summary.best_model_ir
        );
        assert!(summary.weight_entropy > 0.0); // Non-zero entropy
        assert!(summary.total_predictions >= 100);
    }

    #[test]
    fn test_entropy_uniform() {
        let mut ensemble = AdaptiveEnsemble::new(100.0); // Very high temp = uniform
        ensemble.register_model("A");
        ensemble.register_model("B");
        ensemble.register_model("C");
        ensemble.register_model("D");

        // All same IR
        for _ in 0..25 {
            ensemble.update_performance("A", 1.0, 0.25, 1000);
            ensemble.update_performance("B", 1.0, 0.25, 1000);
            ensemble.update_performance("C", 1.0, 0.25, 1000);
            ensemble.update_performance("D", 1.0, 0.25, 1000);
        }

        let summary = ensemble.summary();
        // Max entropy for 4 items = log2(4) = 2 bits
        // With floor, might be slightly different
        assert!(
            summary.weight_entropy > 1.5,
            "Uniform distribution should have high entropy"
        );
    }

    #[test]
    fn test_entropy_concentrated() {
        let mut ensemble = AdaptiveEnsemble::new(0.01); // Very low temp
        ensemble.register_model("Best");
        ensemble.register_model("Worst");

        for _ in 0..25 {
            ensemble.update_performance("Best", 10.0, 0.01, 1000);
            ensemble.update_performance("Worst", 0.1, 0.9, 1000);
        }

        let summary = ensemble.summary();
        // Concentrated distribution should have low entropy
        // But due to floor, not zero
        assert!(
            summary.weight_entropy < 0.8,
            "Concentrated distribution should have low entropy, got {}",
            summary.weight_entropy
        );
    }

    #[test]
    fn test_belief_state_integration() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("GLFT");
        ensemble.register_model("Empirical");
        ensemble.register_model("Funding");

        for _ in 0..25 {
            ensemble.update_performance("GLFT", 2.0, 0.1, 1000);
            ensemble.update_performance("Empirical", 1.0, 0.25, 1000);
            ensemble.update_performance("Funding", 0.5, 0.4, 1000);
        }

        let mut belief = BeliefState::default();
        ensemble.apply_to_belief(&mut belief);

        // Belief should have 3 model weights now
        assert_eq!(belief.model_weights.k(), 3);

        // Best model should have highest weight in belief
        let best_idx = belief.best_model();
        // The indices map to sorted model names: ["Empirical", "Funding", "GLFT"]
        // So GLFT should be index 2
        assert_eq!(best_idx, 2, "GLFT (best) should have index 2 after sorting");
    }

    #[test]
    fn test_decay_rate() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("Test");

        // Initial update with high IR
        for _ in 0..25 {
            ensemble.update_performance("Test", 2.0, 0.1, 1000);
        }
        let ir_initial = ensemble.get_performance("Test").unwrap().information_ratio;

        // Update with low IR - should decay towards new value
        for _ in 0..100 {
            ensemble.update_performance("Test", 0.5, 0.5, 2000);
        }
        let ir_after = ensemble.get_performance("Test").unwrap().information_ratio;

        assert!(
            ir_after < ir_initial,
            "IR should decay towards newer observations"
        );
    }

    #[test]
    fn test_unknown_model_weight() {
        let ensemble = AdaptiveEnsemble::new(1.0);
        assert_eq!(ensemble.get_weight("NonExistent"), 0.0);
    }

    #[test]
    fn test_unknown_model_degradation() {
        let ensemble = AdaptiveEnsemble::new(1.0);
        assert!(!ensemble.is_model_degraded("NonExistent"));
    }

    #[test]
    fn test_empty_weighted_average() {
        let ensemble = AdaptiveEnsemble::new(1.0);
        let predictions: HashMap<String, f64> = HashMap::new();
        assert_eq!(ensemble.weighted_average(&predictions), 0.0);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("A");
        ensemble.register_model("B");
        ensemble.register_model("C");
        ensemble.register_model("D");
        ensemble.register_model("E");

        // Various IR values
        for _ in 0..25 {
            ensemble.update_performance("A", 3.0, 0.05, 1000);
            ensemble.update_performance("B", 1.5, 0.15, 1000);
            ensemble.update_performance("C", 1.0, 0.25, 1000);
            ensemble.update_performance("D", 0.7, 0.35, 1000);
            ensemble.update_performance("E", 0.3, 0.6, 1000);
        }

        let total: f64 = ["A", "B", "C", "D", "E"]
            .iter()
            .map(|n| ensemble.get_weight(n))
            .sum();

        assert!(
            (total - 1.0).abs() < 1e-9,
            "Weights should sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_set_temperature() {
        let mut ensemble = AdaptiveEnsemble::new(1.0);
        ensemble.register_model("A");
        ensemble.register_model("B");

        for _ in 0..25 {
            ensemble.update_performance("A", 2.0, 0.1, 1000);
            ensemble.update_performance("B", 0.5, 0.4, 1000);
        }

        let w_a_before = ensemble.get_weight("A");

        ensemble.set_temperature(0.1);
        let w_a_after = ensemble.get_weight("A");

        // Lower temperature should increase weight of better model
        assert!(
            w_a_after > w_a_before,
            "Lower temperature should increase best model's weight"
        );
    }

    #[test]
    fn test_set_min_weight() {
        let mut ensemble = AdaptiveEnsemble::new(0.01);
        ensemble.set_min_weight(0.1);
        ensemble.register_model("Best");
        ensemble.register_model("Worst");

        for _ in 0..25 {
            ensemble.update_performance("Best", 10.0, 0.01, 1000);
            ensemble.update_performance("Worst", 0.1, 0.9, 1000);
        }

        assert!(
            ensemble.get_weight("Worst") >= 0.1,
            "Weight should respect new min_weight"
        );
    }
}
