//! Bayesian Model Averaging for volatility estimation (WS3).
//!
//! Replaces precision-weighted sigma priority chain with proper BMA.
//! Each sigma source gets weight proportional to its marginal likelihood p(data | model_k).
//!
//! Key property: BMA variance includes BETWEEN-model variance, so when sigma sources
//! disagree, the posterior variance is LARGER than any individual — exactly the
//! "dangerous" signal that feeds PPIP's ambiguity aversion (WS2).

// Allow dead code: methods like observe_realized_variance(), sigma_bma(), model_weights()
// are used in tests and will be wired to production as realized variance feedback matures.
#![allow(dead_code)]

/// A single sigma model contributing to the BMA posterior.
#[derive(Debug, Clone)]
pub(crate) struct SigmaModel {
    /// Display name for diagnostics
    pub(crate) name: &'static str,
    /// Current point estimate of σ (fractional/√sec)
    pub(crate) sigma_estimate: f64,
    /// Model's own uncertainty about σ (variance of estimate)
    pub(crate) sigma_variance: f64,

    /// Running log marginal likelihood: Σ log p(y_t | y_{1:t-1}, M_k)
    log_ml: f64,
    /// One-step-ahead predictive mean (σ² units)
    predictive_mean: f64,
    /// One-step-ahead predictive variance (σ⁴ units)
    predictive_var: f64,

    /// EWMA decay for predictive updates
    decay: f64,
}

impl SigmaModel {
    /// Create a new sigma model with given name and prior uncertainty.
    pub(crate) fn new(name: &'static str, prior_sigma_variance: f64) -> Self {
        Self {
            name,
            sigma_estimate: 0.0,
            sigma_variance: prior_sigma_variance,
            log_ml: 0.0,
            predictive_mean: 0.0,
            predictive_var: prior_sigma_variance.max(1e-20),
            decay: 0.95,
        }
    }

    /// Update this model's estimate and predictive distribution.
    pub(crate) fn update_estimate(&mut self, new_sigma: f64) {
        let old = self.sigma_estimate;
        self.sigma_estimate = new_sigma;

        // Update predictive distribution (EWMA of past predictions)
        let sigma_sq = new_sigma * new_sigma;
        self.predictive_mean = self.decay * self.predictive_mean + (1.0 - self.decay) * sigma_sq;

        // Predictive variance from innovation
        let innovation = sigma_sq - self.predictive_mean;
        self.predictive_var =
            self.decay * self.predictive_var + (1.0 - self.decay) * innovation * innovation;
        self.predictive_var = self.predictive_var.max(1e-20);

        // Model's own uncertainty (EWMA of squared changes)
        let change = new_sigma - old;
        self.sigma_variance =
            self.decay * self.sigma_variance + (1.0 - self.decay) * change * change;
        self.sigma_variance = self.sigma_variance.max(1e-20);
    }

    /// Evaluate log predictive density for a realized σ² observation.
    /// Returns log N(y | μ_k, σ²_k) where μ_k and σ²_k are the predictive moments.
    fn log_predictive_density(&self, realized_sigma_sq: f64) -> f64 {
        let z = realized_sigma_sq - self.predictive_mean;
        let var = self.predictive_var.max(1e-20);
        -0.5 * (z * z / var + var.ln() + std::f64::consts::TAU.ln())
    }
}

/// Bayesian Model Averaging over multiple sigma sources.
///
/// Outputs:
/// - `sigma_bma()`: posterior mean σ (weighted average across models)
/// - `sigma_variance_bma()`: posterior variance of σ² (includes between-model variance)
///
/// The between-model variance feeds directly into PPIP's `sigma_sq_variance`
/// (WS2 ambiguity aversion term) — when sigma sources disagree, skew automatically increases.
#[derive(Debug, Clone)]
pub(crate) struct BayesianModelAverager {
    models: Vec<SigmaModel>,
    /// Number of realized variance observations
    n_observations: usize,
    /// Minimum observations before BMA weights are used (before this, equal weights)
    min_obs_for_bma: usize,
}

impl Default for BayesianModelAverager {
    fn default() -> Self {
        Self {
            models: vec![
                SigmaModel::new("clean_bv", 1e-12),
                SigmaModel::new("leverage_adjusted", 1e-12),
                SigmaModel::new("particle_filter", 1e-12),
            ],
            n_observations: 0,
            min_obs_for_bma: 20,
        }
    }
}

impl BayesianModelAverager {
    /// Update model estimates from current sigma values.
    /// Call once per quote cycle with latest sigma readings.
    pub(crate) fn update_estimates(
        &mut self,
        sigma_clean: f64,
        sigma_leverage_adjusted: f64,
        sigma_particle: f64,
    ) {
        if self.models.len() >= 3 {
            self.models[0].update_estimate(sigma_clean);
            self.models[1].update_estimate(sigma_leverage_adjusted);
            if sigma_particle > 0.0 {
                self.models[2].update_estimate(sigma_particle);
            }
        }
    }

    /// Update marginal likelihoods after observing realized variance.
    /// `realized_sigma_sq` should be computed from actual price returns
    /// over the prediction horizon (e.g., 5s realized variance).
    pub(crate) fn observe_realized_variance(&mut self, realized_sigma_sq: f64) {
        if realized_sigma_sq <= 0.0 {
            return;
        }
        for model in &mut self.models {
            if model.sigma_estimate > 0.0 {
                let log_pred = model.log_predictive_density(realized_sigma_sq);
                // Forgetting: decay old evidence to adapt to regime changes
                model.log_ml = 0.995 * model.log_ml + log_pred;
            }
        }
        self.n_observations += 1;
    }

    /// Compute BMA model weights (posterior probabilities).
    fn weights(&self) -> Vec<f64> {
        let active: Vec<usize> = self
            .models
            .iter()
            .enumerate()
            .filter(|(_, m)| m.sigma_estimate > 0.0)
            .map(|(i, _)| i)
            .collect();

        if active.is_empty() {
            return vec![0.0; self.models.len()];
        }

        // Before sufficient observations, use equal weights
        if self.n_observations < self.min_obs_for_bma {
            let mut weights = vec![0.0; self.models.len()];
            let w = 1.0 / active.len() as f64;
            for &i in &active {
                weights[i] = w;
            }
            return weights;
        }

        // Log-sum-exp trick for numerical stability
        let max_ll = active
            .iter()
            .map(|&i| self.models[i].log_ml)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut weights = vec![0.0; self.models.len()];
        let mut w_sum = 0.0;
        for &i in &active {
            let w = (self.models[i].log_ml - max_ll).exp();
            weights[i] = w;
            w_sum += w;
        }

        if w_sum > 0.0 {
            for w in &mut weights {
                *w /= w_sum;
            }
        }

        weights
    }

    /// BMA posterior mean: E[σ | data] = Σ w_k × σ_k
    pub(crate) fn sigma_bma(&self) -> f64 {
        let weights = self.weights();
        let mut bma = 0.0;
        for (model, &w) in self.models.iter().zip(&weights) {
            bma += w * model.sigma_estimate;
        }
        bma
    }

    /// BMA posterior variance of σ² (includes between-model variance).
    ///
    /// Var[σ² | data] = Σ w_k × (σ²_k - σ²_bma)² + Σ w_k × var_k
    ///
    /// This is LARGER than any individual model's variance when models disagree —
    /// capturing model uncertainty. Feeds PPIP's `sigma_sq_variance`.
    pub(crate) fn sigma_variance_bma(&self) -> f64 {
        let weights = self.weights();
        let mean_sq = {
            let mut s = 0.0;
            for (model, &w) in self.models.iter().zip(&weights) {
                s += w * model.sigma_estimate.powi(2);
            }
            s
        };

        let mut var = 0.0;
        for (model, &w) in self.models.iter().zip(&weights) {
            // Between-model variance: how much this model differs from BMA mean
            let diff = model.sigma_estimate.powi(2) - mean_sq;
            var += w * diff * diff;

            // Within-model variance: this model's own uncertainty
            var += w * model.sigma_variance;
        }
        var
    }

    /// Number of realized variance observations processed.
    pub(crate) fn n_observations(&self) -> usize {
        self.n_observations
    }

    /// Get model weights for diagnostics.
    pub(crate) fn model_weights(&self) -> Vec<(&str, f64)> {
        let weights = self.weights();
        self.models
            .iter()
            .zip(weights)
            .map(|(m, w)| (m.name, w))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws3_bma_equal_weights_before_warmup() {
        let mut bma = BayesianModelAverager::default();
        bma.update_estimates(0.001, 0.001, 0.001);

        let weights = bma.weights();
        // All three active, equal weights
        for &w in &weights {
            assert!(
                (w - 1.0 / 3.0).abs() < 0.01,
                "Expected equal weights, got {w}"
            );
        }
    }

    #[test]
    fn test_ws3_bma_sigma_is_weighted_average() {
        let mut bma = BayesianModelAverager::default();
        bma.update_estimates(0.001, 0.002, 0.003);

        let sigma = bma.sigma_bma();
        // Equal weights → simple average
        let expected = (0.001 + 0.002 + 0.003) / 3.0;
        assert!(
            (sigma - expected).abs() < 1e-10,
            "BMA sigma should be average: got {sigma}, expected {expected}"
        );
    }

    #[test]
    fn test_ws3_bma_disagree_increases_variance() {
        // When models agree, variance is low
        let mut bma_agree = BayesianModelAverager::default();
        bma_agree.update_estimates(0.001, 0.001, 0.001);
        let var_agree = bma_agree.sigma_variance_bma();

        // When models disagree, variance is high (between-model component)
        let mut bma_disagree = BayesianModelAverager::default();
        bma_disagree.update_estimates(0.0005, 0.001, 0.002);
        let var_disagree = bma_disagree.sigma_variance_bma();

        assert!(
            var_disagree > var_agree,
            "Disagreement should increase BMA variance: agree={var_agree:.2e}, disagree={var_disagree:.2e}"
        );
    }

    #[test]
    fn test_ws3_bma_best_model_gets_higher_weight() {
        let mut bma = BayesianModelAverager::default();

        // Model 0 (clean_bv) consistently predicts well
        // Model 1 (leverage_adjusted) consistently over-predicts
        // Model 2 (particle) consistently under-predicts
        for _ in 0..50 {
            bma.update_estimates(0.001, 0.002, 0.0005);
            // Realized variance close to model 0's prediction
            bma.observe_realized_variance(0.001_f64.powi(2));
        }

        let weights = bma.model_weights();
        let w_clean = weights[0].1;
        let w_leverage = weights[1].1;

        assert!(
            w_clean > w_leverage,
            "Better-predicting model should have higher weight: clean={w_clean:.3}, leverage={w_leverage:.3}"
        );
    }

    #[test]
    fn test_ws3_bma_inactive_particle_excluded() {
        let mut bma = BayesianModelAverager::default();
        // Particle filter at 0.0 means not yet warmed up
        bma.update_estimates(0.001, 0.002, 0.0);

        let weights = bma.weights();
        // Particle should have 0 weight
        assert!(
            weights[2] < 0.01,
            "Inactive particle filter should have ~0 weight: {:.4}",
            weights[2]
        );
        // Other two should share equally
        assert!(
            (weights[0] - 0.5).abs() < 0.01,
            "Two active models should share: {:.4}",
            weights[0]
        );
    }

    #[test]
    fn test_ws3_bma_variance_feeds_ppip() {
        // Integration test: BMA variance should be usable as sigma_sq_variance in PPIP
        let mut bma = BayesianModelAverager::default();
        bma.update_estimates(0.0005, 0.001, 0.002);

        let var = bma.sigma_variance_bma();
        let sigma_sq_mean = bma.sigma_bma().powi(2);

        // CV² = Var[σ²] / E[σ²]² should be meaningful (> 0.01) when models disagree
        let cv_sq = var / sigma_sq_mean.powi(2).max(1e-30);
        assert!(
            cv_sq > 0.01,
            "CV² should be meaningful when models disagree: {cv_sq:.4}"
        );
    }
}
