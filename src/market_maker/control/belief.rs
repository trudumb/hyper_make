//! Bayesian belief state for the stochastic controller.
//!
//! The belief state maintains posterior distributions over latent market parameters
//! that evolve with observations. This enables proper uncertainty quantification
//! for optimal control under incomplete information.

use super::interface::{GaussianEstimate, LearningModuleOutput, ModelPrediction};
use super::types::{DirichletPosterior, GammaPosterior, NormalGammaPosterior};

/// Complete belief state for the stochastic controller (Layer 3).
///
/// # Purpose
///
/// This struct tracks **model ensemble weights** and **fill rate beliefs** for
/// the adaptive learning system. It is used by `AdaptiveEnsemble` for model
/// selection and weighting.
///
/// # Distinction from CentralBeliefState
///
/// This is **NOT** the same as `belief::CentralBeliefState` which tracks:
/// - Market beliefs (drift, volatility, regime)
/// - Fill intensity (kappa)
/// - Changepoint detection (BOCD)
/// - Continuation probability
///
/// This struct tracks:
/// - Fill rate lambda (Gamma posterior)
/// - Adverse selection distribution
/// - Model ensemble weights (Dirichlet posterior)
/// - Edge estimates by regime
///
/// The two systems are complementary - this one for model selection,
/// `CentralBeliefState` for market state estimation.
#[derive(Debug, Clone)]
pub struct BeliefState {
    // === Fill rate belief ===
    /// Fill rate (λ) posterior: Gamma distribution
    /// Updated with fill counts per time period
    pub lambda: GammaPosterior,

    // === Adverse selection belief ===
    /// Adverse selection (AS) posterior: Normal-Gamma
    /// Updated with realized AS observations
    pub as_dist: NormalGammaPosterior,

    /// AS bias correction from Layer 2
    pub as_bias_correction: f64,

    // === Edge beliefs by regime ===
    /// Edge posteriors for each volatility regime
    /// Index: 0=Low, 1=Normal, 2=High/Extreme
    pub edge_by_regime: [NormalGammaPosterior; 3],

    /// Current regime probabilities
    pub regime_probs: [f64; 3],

    // === Model selection belief ===
    /// Dirichlet posterior over ensemble models
    pub model_weights: DirichletPosterior,

    // === Epistemic uncertainty ===
    /// Disagreement between models (from Layer 2)
    pub epistemic_uncertainty: f64,

    /// Overall uncertainty in edge prediction
    pub total_edge_uncertainty: f64,

    // === Sufficient statistics ===
    /// Number of fills observed
    pub n_fills: u64,

    /// Total observation time
    pub total_time: f64,

    /// Last update timestamp
    pub last_update_ms: u64,
}

impl Default for BeliefState {
    fn default() -> Self {
        Self {
            lambda: GammaPosterior::new(2.0, 1.0), // Prior: ~2 fills/unit time
            as_dist: NormalGammaPosterior::new(2.0, 1.0, 2.0, 1.0), // Prior: ~2 bps AS
            as_bias_correction: 0.0,
            edge_by_regime: [
                NormalGammaPosterior::new(3.0, 1.0, 2.0, 1.0), // Low vol: higher edge prior
                NormalGammaPosterior::new(1.0, 1.0, 2.0, 1.0), // Normal: neutral prior
                NormalGammaPosterior::new(-1.0, 1.0, 2.0, 1.0), // High vol: lower edge prior
            ],
            regime_probs: [0.2, 0.6, 0.2], // Prior: mostly normal
            model_weights: DirichletPosterior::uniform(4), // 4 models in ensemble
            epistemic_uncertainty: 0.5,
            total_edge_uncertainty: 1.0,
            n_fills: 0,
            total_time: 0.0,
            last_update_ms: 0,
        }
    }
}

impl BeliefState {
    /// Create belief state from learning module output.
    pub fn from_learning_output(output: &LearningModuleOutput) -> Self {
        let mut state = Self::default();
        state.update_from_learning(output);
        state
    }

    /// Update beliefs from learning module output.
    ///
    /// This is the main bridge between Layer 2 and Layer 3.
    pub fn update_from_learning(&mut self, output: &LearningModuleOutput) {
        // Update epistemic uncertainty from model disagreement
        self.epistemic_uncertainty = output.model_disagreement;
        self.total_edge_uncertainty = output.edge_prediction.std;

        // Update AS bias correction
        self.as_bias_correction = output.as_bias;

        // Update model weights from ensemble
        self.update_model_weights(&output.model_predictions);

        // Update edge belief (using ensemble prediction as observation)
        // This shrinks our prior towards the ensemble's estimate
        let edge_obs = output.edge_prediction.mean;
        let current_regime = self.most_likely_regime();
        self.edge_by_regime[current_regime].update(edge_obs);
    }

    /// Update model weights from ensemble predictions.
    fn update_model_weights(&mut self, predictions: &[ModelPrediction]) {
        if predictions.is_empty() {
            return;
        }

        // Resize if needed
        if self.model_weights.k() != predictions.len() {
            self.model_weights = DirichletPosterior::uniform(predictions.len());
        }

        // Update based on performance scores
        for (i, pred) in predictions.iter().enumerate() {
            // Weight update proportional to performance
            self.model_weights.update_weighted(i, pred.performance);
        }

        // Apply decay to handle non-stationarity
        self.model_weights.decay(0.995);
    }

    /// Update belief with a fill observation.
    pub fn update_fill(&mut self, realized_as_bps: f64, realized_edge_bps: f64, time_elapsed: f64) {
        // Update fill rate (one fill in time_elapsed)
        self.lambda.update(1.0, time_elapsed);

        // Update AS belief
        self.as_dist.update(realized_as_bps);

        // Update edge belief for current regime
        let regime = self.most_likely_regime();
        self.edge_by_regime[regime].update(realized_edge_bps);

        // Update sufficient statistics
        self.n_fills += 1;
        self.total_time += time_elapsed;
        self.last_update_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Update regime probabilities.
    pub fn update_regime(&mut self, vol_regime: usize, transition_prob: f64) {
        // Simple exponential smoothing towards observed regime
        let alpha = transition_prob.clamp(0.01, 0.5);

        for (i, prob) in self.regime_probs.iter_mut().enumerate() {
            if i == vol_regime {
                *prob = *prob * (1.0 - alpha) + alpha;
            } else {
                *prob *= 1.0 - alpha;
            }
        }

        // Normalize
        let sum: f64 = self.regime_probs.iter().sum();
        if sum > 0.0 {
            for prob in &mut self.regime_probs {
                *prob /= sum;
            }
        }
    }

    /// Get expected fill rate.
    pub fn expected_fill_rate(&self) -> f64 {
        self.lambda.mean()
    }

    /// Get fill rate uncertainty (coefficient of variation).
    pub fn fill_rate_uncertainty(&self) -> f64 {
        self.lambda.cv()
    }

    /// Get expected AS with bias correction.
    pub fn expected_as(&self) -> f64 {
        self.as_dist.mean_of_mean() + self.as_bias_correction
    }

    /// Get AS uncertainty.
    pub fn as_uncertainty(&self) -> f64 {
        self.as_dist.std_of_mean()
    }

    /// Get expected edge for current regime.
    pub fn expected_edge(&self) -> f64 {
        // Regime-weighted expectation
        let mut edge = 0.0;
        for (i, prob) in self.regime_probs.iter().enumerate() {
            edge += prob * self.edge_by_regime[i].mean_of_mean();
        }
        edge
    }

    /// Get edge as Gaussian estimate (for compatibility).
    pub fn edge_estimate(&self) -> GaussianEstimate {
        GaussianEstimate::new(self.expected_edge(), self.total_edge_uncertainty)
    }

    /// Get most likely regime index.
    pub fn most_likely_regime(&self) -> usize {
        self.regime_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1)
    }

    /// Probability of positive edge.
    pub fn p_positive_edge(&self) -> f64 {
        let regime = self.most_likely_regime();
        self.edge_by_regime[regime].prob_mean_greater_than(0.0)
    }

    /// Get model entropy (uncertainty over which model is best).
    pub fn model_entropy(&self) -> f64 {
        self.model_weights.entropy()
    }

    /// Get the best model index.
    pub fn best_model(&self) -> usize {
        self.model_weights.argmax()
    }

    /// Overall confidence score [0, 1].
    pub fn confidence(&self) -> f64 {
        let p_edge = self.p_positive_edge();
        let model_certainty = 1.0 - (self.model_entropy() / (self.model_weights.k() as f64).ln());
        let fill_certainty = 1.0 / (1.0 + self.fill_rate_uncertainty());

        // Geometric mean of certainties
        (p_edge * model_certainty.max(0.0) * fill_certainty).powf(1.0 / 3.0)
    }

    /// Reset to prior state (after changepoint).
    pub fn reset_to_prior(&mut self) {
        *self = Self::default();
    }

    /// Partial reset (keep some information).
    pub fn soft_reset(&mut self, retention: f64) {
        // Interpolate towards prior
        let prior = Self::default();

        // Fill rate: keep some observations
        let retained_fills = self.n_fills as f64 * retention;
        self.lambda = GammaPosterior::new(
            prior.lambda.alpha + retained_fills,
            prior.lambda.beta + self.total_time * retention,
        );

        // AS: blend with prior
        self.as_dist.kappa = prior.as_dist.kappa + self.as_dist.kappa * retention;

        // Edge by regime: blend with priors
        for i in 0..3 {
            self.edge_by_regime[i].kappa =
                prior.edge_by_regime[i].kappa + self.edge_by_regime[i].kappa * retention;
        }

        // Model weights: decay
        self.model_weights.decay(retention);

        // Statistics
        self.n_fills = (self.n_fills as f64 * retention) as u64;
        self.total_time *= retention;
    }
}

/// Belief update result with diagnostics.
#[derive(Debug, Clone)]
pub struct BeliefUpdateResult {
    /// Edge estimate before update
    pub prior_edge: f64,
    /// Edge estimate after update
    pub posterior_edge: f64,
    /// Information gained (KL divergence)
    pub info_gain: f64,
    /// Whether update was significant
    pub significant: bool,
}

impl BeliefState {
    /// Update with observation and return diagnostics.
    pub fn update_with_diagnostics(
        &mut self,
        realized_as_bps: f64,
        realized_edge_bps: f64,
        time_elapsed: f64,
    ) -> BeliefUpdateResult {
        let prior_edge = self.expected_edge();

        self.update_fill(realized_as_bps, realized_edge_bps, time_elapsed);

        let posterior_edge = self.expected_edge();

        // Approximate information gain
        let prior_var = self.as_uncertainty().powi(2);
        let posterior_var = self.as_uncertainty().powi(2);
        let info_gain = if prior_var > 1e-10 {
            0.5 * (prior_var / posterior_var - 1.0 + (posterior_var / prior_var).ln())
        } else {
            0.0
        };

        BeliefUpdateResult {
            prior_edge,
            posterior_edge,
            info_gain,
            significant: info_gain > 0.01,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_state_default() {
        let belief = BeliefState::default();
        assert!(belief.expected_fill_rate() > 0.0);
        assert_eq!(belief.n_fills, 0);
    }

    #[test]
    fn test_fill_update() {
        let mut belief = BeliefState::default();

        // Simulate some fills
        for _ in 0..10 {
            belief.update_fill(1.5, 2.0, 1.0);
        }

        assert_eq!(belief.n_fills, 10);
        // Fill rate should increase (more fills than prior expected)
        assert!(belief.expected_fill_rate() > 1.0);
    }

    #[test]
    fn test_regime_update() {
        let mut belief = BeliefState::default();

        // Observe high volatility regime
        for _ in 0..10 {
            belief.update_regime(2, 0.2);
        }

        assert!(belief.regime_probs[2] > belief.regime_probs[0]);
        assert_eq!(belief.most_likely_regime(), 2);
    }

    #[test]
    fn test_soft_reset() {
        let mut belief = BeliefState::default();

        // Accumulate some observations
        for _ in 0..100 {
            belief.update_fill(2.0, 1.0, 1.0);
        }

        let fills_before = belief.n_fills;
        belief.soft_reset(0.5);

        assert!(belief.n_fills < fills_before);
    }

    #[test]
    fn test_confidence() {
        let belief = BeliefState::default();
        let conf = belief.confidence();

        assert!((0.0..=1.0).contains(&conf));
    }
}
