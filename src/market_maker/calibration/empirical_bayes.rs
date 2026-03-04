//! Empirical Bayes Prior Framework
//!
//! Central registry of all learnable parameters with principled Bayesian priors.
//! Wraps [`BayesianParam`] with a unified trait and registry for:
//!
//! 1. **Prior elicitation**: Seed from domain knowledge (current "magic numbers")
//! 2. **Online learning**: Conjugate updates from observed data
//! 3. **Credible bounds**: Automatic safety clamps from posterior intervals
//! 4. **Checkpoint persistence**: Full registry state saved/restored
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut registry = ParameterRegistry::new();
//! registry.register(EBParam::normal("beta_volatility", 0.3, 0.01));
//! registry.observe("beta_volatility", 0.35, 1.0);
//! let value = registry.get_value("beta_volatility"); // Shrinkage estimate
//! let (lo, hi) = registry.get_bounds("beta_volatility"); // 95% CI
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::parameter_learner::{BayesianParam, PriorFamily};

/// Default min observations for `is_calibrated` check.
const DEFAULT_MIN_OBS: usize = 20;
/// Default max CV for `is_calibrated` check.
const DEFAULT_MAX_CV: f64 = 0.5;

/// A named Empirical Bayes parameter with metadata for the registry.
///
/// Thin wrapper around [`BayesianParam`] adding:
/// - Category tag for grouped summaries
/// - Hard bounds for absolute safety (independent of posterior)
/// - Description for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EBParam {
    /// The underlying Bayesian parameter
    pub param: BayesianParam,

    /// Category for grouping (e.g., "risk_model", "signal", "ewma")
    #[serde(default)]
    pub category: String,

    /// Human-readable description
    #[serde(default)]
    pub description: String,

    /// Hard lower bound (safety floor, independent of posterior)
    #[serde(default = "default_neg_inf")]
    pub hard_lower: f64,

    /// Hard upper bound (safety ceiling, independent of posterior)
    #[serde(default = "default_pos_inf")]
    pub hard_upper: f64,
}

fn default_neg_inf() -> f64 {
    -1e18
}
fn default_pos_inf() -> f64 {
    1e18
}

impl EBParam {
    /// Create a Normal-family EB parameter for unbounded quantities (betas, weights).
    ///
    /// # Arguments
    /// * `name` - Parameter name (must be unique in registry)
    /// * `prior_mean` - Domain knowledge baseline (the old "magic number")
    /// * `prior_var` - Prior variance (uncertainty² in the baseline)
    ///
    /// Note: `BayesianParam::normal` derives `prior_strength = 1/prior_var`.
    pub fn normal(name: &str, prior_mean: f64, prior_var: f64) -> Self {
        let param = BayesianParam::normal(name, prior_mean, prior_var);
        Self {
            param,
            category: String::new(),
            description: String::new(),
            hard_lower: -1e18,
            hard_upper: 1e18,
        }
    }

    /// Create a Gamma-family EB parameter for positive rates (decay, intensity).
    ///
    /// # Arguments
    /// * `name` - Parameter name
    /// * `prior_mean` - Expected rate (e.g., EWMA alpha, fill rate)
    /// * `prior_strength` - Pseudo-observations
    pub fn gamma(name: &str, prior_mean: f64, prior_strength: f64) -> Self {
        let param = BayesianParam::gamma(name, prior_mean, prior_strength);
        Self {
            param,
            category: String::new(),
            description: String::new(),
            hard_lower: 0.0,
            hard_upper: 1e18,
        }
    }

    /// Create a Beta-family EB parameter for probabilities [0, 1].
    ///
    /// # Arguments
    /// * `name` - Parameter name
    /// * `prior_mean` - Expected probability
    /// * `prior_strength` - Pseudo-observations
    pub fn beta(name: &str, prior_mean: f64, prior_strength: f64) -> Self {
        let param = BayesianParam::beta(name, prior_mean, prior_strength);
        Self {
            param,
            category: String::new(),
            description: String::new(),
            hard_lower: 0.0,
            hard_upper: 1.0,
        }
    }

    /// Create a LogNormal-family EB parameter for multiplicative factors.
    ///
    /// # Arguments
    /// * `name` - Parameter name
    /// * `prior_mean` - Expected value (in original scale, not log)
    /// * `prior_cv` - Prior coefficient of variation (σ/μ)
    pub fn log_normal(name: &str, prior_mean: f64, prior_cv: f64) -> Self {
        let param = BayesianParam::log_normal(name, prior_mean, prior_cv);
        Self {
            param,
            category: String::new(),
            description: String::new(),
            hard_lower: 0.0,
            hard_upper: 1e18,
        }
    }

    /// Create an InverseGamma-family EB parameter for variances.
    ///
    /// # Arguments
    /// * `name` - Parameter name
    /// * `prior_mean` - Expected variance
    /// * `prior_strength` - Pseudo-observations (degrees of freedom)
    pub fn inverse_gamma(name: &str, prior_mean: f64, prior_strength: f64) -> Self {
        let param = BayesianParam::inverse_gamma(name, prior_mean, prior_strength);
        Self {
            param,
            category: String::new(),
            description: String::new(),
            hard_lower: 0.0,
            hard_upper: 1e18,
        }
    }

    /// Set category for grouping.
    pub fn with_category(mut self, category: &str) -> Self {
        self.category = category.to_string();
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Set hard safety bounds (independent of posterior).
    pub fn with_hard_bounds(mut self, lower: f64, upper: f64) -> Self {
        self.hard_lower = lower;
        self.hard_upper = upper;
        self
    }

    /// Get the current estimate, clamped to hard bounds.
    pub fn value(&self) -> f64 {
        self.param
            .estimate()
            .clamp(self.hard_lower, self.hard_upper)
    }

    /// Get 95% credible interval, intersected with hard bounds.
    pub fn bounds(&self) -> (f64, f64) {
        let (lo, hi) = self.param.credible_interval_95();
        (lo.max(self.hard_lower), hi.min(self.hard_upper))
    }

    /// Observe a new value. Dispatches to the correct conjugate update.
    ///
    /// * Normal: `value` is the observed quantity, `obs_var` is its variance
    /// * Gamma: `value` is the observed positive quantity
    /// * Beta: `value > 0.5` → success, else failure
    /// * InverseGamma: `value` is a squared residual
    /// * LogNormal: `value` is the observed positive quantity
    pub fn observe(&mut self, value: f64, obs_var: f64) {
        match self.param.family {
            PriorFamily::Normal => {
                self.param.observe_normal(value, obs_var.max(1e-10));
            }
            PriorFamily::Gamma => {
                // Use Poisson conjugate to pull estimate toward observed value
                let count = value.max(0.0).round() as usize;
                self.param.observe_gamma_poisson(count, 1.0);
            }
            PriorFamily::Beta => {
                if value > 0.5 {
                    self.param.observe_beta(1, 0);
                } else {
                    self.param.observe_beta(0, 1);
                }
            }
            PriorFamily::InverseGamma => {
                self.param.observe_variance(value.max(1e-20));
            }
            PriorFamily::LogNormal => {
                self.param.observe_log_normal(value.max(1e-10));
            }
        }
    }

    /// Observe Beta successes and failures directly.
    pub fn observe_beta_counts(&mut self, successes: usize, failures: usize) {
        self.param.observe_beta(successes, failures);
    }

    /// Observe a Normal value with known observation variance.
    pub fn observe_normal(&mut self, value: f64, obs_var: f64) {
        self.param.observe_normal(value, obs_var.max(1e-10));
    }

    /// Observe a positive value for Gamma-distributed parameter.
    ///
    /// Uses Poisson conjugate update where the observed value is treated
    /// as a count per unit exposure, pulling the estimate toward the value.
    pub fn observe_gamma_value(&mut self, value: f64) {
        // Use Poisson conjugate: shape += count, rate += exposure
        // Treat value as "count per 1 unit of exposure"
        // This makes E[λ] = (prior_shape + value) / (prior_rate + 1) → converges to value
        let count = value.max(0.0).round() as usize;
        self.param.observe_gamma_poisson(count, 1.0);
    }

    /// Observe a squared residual for InverseGamma variance estimation.
    pub fn observe_variance(&mut self, squared_residual: f64) {
        self.param.observe_variance(squared_residual.max(1e-20));
    }

    /// Effective sample size (pseudo-observations + real observations).
    pub fn effective_sample_size(&self) -> f64 {
        self.param.effective_sample_size()
    }

    /// Whether enough data has been observed for the posterior to dominate the prior.
    pub fn is_calibrated(&self) -> bool {
        self.param.is_calibrated(DEFAULT_MIN_OBS, DEFAULT_MAX_CV)
    }

    /// Reset to prior (e.g., after regime change).
    pub fn reset_to_prior(&mut self) {
        self.param.reset_to_prior();
    }

    /// Prior mean for reference.
    pub fn prior_mean(&self) -> f64 {
        self.param.prior_mean
    }

    /// Prior variance (from the prior distribution parameters).
    pub fn prior_variance(&self) -> f64 {
        self.param.variance()
    }
}

/// Central registry of all learnable parameters.
///
/// Provides a single source of truth for parameter values, bounds, and calibration
/// state across the entire system. Parameters are organized by category and
/// support checkpoint persistence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParameterRegistry {
    /// All registered parameters, keyed by name.
    /// BTreeMap for deterministic iteration order.
    params: BTreeMap<String, EBParam>,
}

impl ParameterRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            params: BTreeMap::new(),
        }
    }

    /// Register a new parameter. Overwrites if name already exists.
    pub fn register(&mut self, param: EBParam) {
        self.params.insert(param.param.name.clone(), param);
    }

    /// Observe a new value for a named parameter.
    ///
    /// For Normal params, `obs_var` is observation variance.
    /// For other families, `obs_var` is ignored (use type-specific methods).
    /// No-op if parameter not found (defensive — never crash the trader).
    pub fn observe(&mut self, name: &str, value: f64, obs_var: f64) {
        if let Some(p) = self.params.get_mut(name) {
            p.observe(value, obs_var);
        }
    }

    /// Observe Beta successes/failures for a named parameter.
    pub fn observe_beta(&mut self, name: &str, successes: usize, failures: usize) {
        if let Some(p) = self.params.get_mut(name) {
            p.observe_beta_counts(successes, failures);
        }
    }

    /// Observe a Normal value with known observation variance for a named parameter.
    ///
    /// Only updates Normal-family parameters. No-op for other families or if not found.
    pub fn observe_normal(&mut self, name: &str, value: f64, obs_var: f64) {
        if let Some(p) = self.params.get_mut(name) {
            p.observe_normal(value, obs_var);
        }
    }

    /// Get current estimate for a parameter.
    ///
    /// Returns `None` if not registered. Use `get_value_or` for fallback.
    pub fn get_value(&self, name: &str) -> Option<f64> {
        self.params.get(name).map(|p| p.value())
    }

    /// Get current estimate or a default value.
    pub fn get_value_or(&self, name: &str, default: f64) -> f64 {
        self.get_value(name).unwrap_or(default)
    }

    /// Get 95% credible bounds for a parameter.
    ///
    /// Returns `None` if not registered.
    pub fn get_bounds(&self, name: &str) -> Option<(f64, f64)> {
        self.params.get(name).map(|p| p.bounds())
    }

    /// Get a reference to a parameter by name.
    pub fn get(&self, name: &str) -> Option<&EBParam> {
        self.params.get(name)
    }

    /// Get a mutable reference to a parameter by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut EBParam> {
        self.params.get_mut(name)
    }

    /// Check if a parameter is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.params.contains_key(name)
    }

    /// Number of registered parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Get all parameter names.
    pub fn names(&self) -> Vec<&str> {
        self.params.keys().map(|s| s.as_str()).collect()
    }

    /// Get parameters by category.
    pub fn by_category(&self, category: &str) -> Vec<(&str, &EBParam)> {
        self.params
            .iter()
            .filter(|(_, p)| p.category == category)
            .map(|(name, p)| (name.as_str(), p))
            .collect()
    }

    /// Get a summary of all parameters for diagnostics.
    pub fn summary(&self) -> Vec<ParamSummary> {
        self.params
            .iter()
            .map(|(name, p)| {
                let (lo_95, hi_95) = p.bounds();
                ParamSummary {
                    name: name.clone(),
                    category: p.category.clone(),
                    family: p.param.family,
                    value: p.value(),
                    prior_mean: p.prior_mean(),
                    ci_95_lower: lo_95,
                    ci_95_upper: hi_95,
                    ess: p.effective_sample_size(),
                    is_calibrated: p.is_calibrated(),
                }
            })
            .collect()
    }

    /// Reset all parameters to their priors.
    pub fn reset_all(&mut self) {
        for p in self.params.values_mut() {
            p.reset_to_prior();
        }
    }

    /// Reset parameters in a specific category.
    pub fn reset_category(&mut self, category: &str) {
        for p in self.params.values_mut() {
            if p.category == category {
                p.reset_to_prior();
            }
        }
    }

    /// Merge another registry into this one (for checkpoint restore).
    /// Only updates parameters that exist in both registries.
    pub fn merge_from(&mut self, other: &ParameterRegistry) {
        for (name, other_param) in &other.params {
            if let Some(existing) = self.params.get_mut(name) {
                // Restore posterior state from checkpoint
                existing.param.posterior_param1 = other_param.param.posterior_param1;
                existing.param.posterior_param2 = other_param.param.posterior_param2;
                existing.param.n_observations = other_param.param.n_observations;
            }
        }
    }

    /// Count of calibrated parameters (ESS > prior_strength).
    pub fn n_calibrated(&self) -> usize {
        self.params.values().filter(|p| p.is_calibrated()).count()
    }

    /// Fraction of parameters that are calibrated.
    pub fn calibration_fraction(&self) -> f64 {
        if self.params.is_empty() {
            return 0.0;
        }
        self.n_calibrated() as f64 / self.params.len() as f64
    }
}

/// Summary of a single parameter for diagnostics/logging.
#[derive(Debug, Clone)]
pub struct ParamSummary {
    pub name: String,
    pub category: String,
    pub family: PriorFamily,
    pub value: f64,
    pub prior_mean: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
    pub ess: f64,
    pub is_calibrated: bool,
}

impl std::fmt::Display for ParamSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<30} {:>8.4} (prior={:.4}) [{:.4}, {:.4}] ESS={:.0} {}",
            self.name,
            self.value,
            self.prior_mean,
            self.ci_95_lower,
            self.ci_95_upper,
            self.ess,
            if self.is_calibrated {
                "calibrated"
            } else {
                "prior-dominated"
            }
        )
    }
}

/// Create the default parameter registry with all system parameters.
///
/// This seeds every learnable parameter with its domain-knowledge prior.
/// Prior means are the current "magic numbers" — the system degrades gracefully
/// to current behavior when no data is available.
pub fn create_default_registry() -> ParameterRegistry {
    let mut reg = ParameterRegistry::new();

    // === Risk Model Betas (Phase 2) ===
    // Prior means match CalibratedRiskModel::default() — the current production values.
    // Prior variance set wide enough (std ≈ 0.3-1.0) so data dominates after ~100 observations.
    // Hard bounds provide absolute safety rails; credible intervals provide soft guidance.
    reg.register(
        EBParam::normal("beta_volatility", 1.0, 0.25)
            .with_category("risk_model")
            .with_description("Log-gamma per unit excess volatility")
            .with_hard_bounds(-1.0, 4.0),
    );
    reg.register(
        EBParam::normal("beta_toxicity", 0.5, 0.09)
            .with_category("risk_model")
            .with_description("Log-gamma per unit toxicity score")
            .with_hard_bounds(-0.5, 2.0),
    );
    reg.register(
        EBParam::normal("beta_inventory", 4.0, 1.0)
            .with_category("risk_model")
            .with_description("Log-gamma per unit inventory fraction²")
            .with_hard_bounds(0.5, 10.0),
    );
    reg.register(
        EBParam::normal("beta_hawkes", 0.4, 0.04)
            .with_category("risk_model")
            .with_description("Log-gamma per unit excess Hawkes intensity")
            .with_hard_bounds(-0.5, 2.0),
    );
    reg.register(
        EBParam::normal("beta_book_depth", 0.3, 0.04)
            .with_category("risk_model")
            .with_description("Log-gamma per unit depth depletion")
            .with_hard_bounds(-0.5, 2.0),
    );
    reg.register(
        EBParam::normal("beta_uncertainty", 0.2, 0.04)
            .with_category("risk_model")
            .with_description("Log-gamma per unit model uncertainty")
            .with_hard_bounds(-0.5, 1.5),
    );
    reg.register(
        EBParam::normal("beta_confidence", -0.4, 0.04)
            .with_category("risk_model")
            .with_description("Log-gamma per unit direction confidence (negative = tighter)")
            .with_hard_bounds(-1.5, 0.5),
    );
    reg.register(
        EBParam::normal("beta_cascade", 1.2, 0.16)
            .with_category("risk_model")
            .with_description("Log-gamma per unit cascade intensity")
            .with_hard_bounds(0.0, 4.0),
    );
    reg.register(
        EBParam::normal("beta_tail_risk", 0.7, 0.09)
            .with_category("risk_model")
            .with_description("Log-gamma per unit tail risk")
            .with_hard_bounds(0.0, 3.0),
    );
    reg.register(
        EBParam::normal("beta_drawdown", 1.4, 0.16)
            .with_category("risk_model")
            .with_description("Log-gamma per unit drawdown fraction")
            .with_hard_bounds(0.0, 4.0),
    );
    reg.register(
        EBParam::normal("beta_regime", 1.0, 0.09)
            .with_category("risk_model")
            .with_description("Log-gamma per unit regime risk score")
            .with_hard_bounds(0.0, 3.0),
    );
    reg.register(
        EBParam::normal("beta_ghost", 0.5, 0.04)
            .with_category("risk_model")
            .with_description("Log-gamma per unit ghost depth depletion")
            .with_hard_bounds(0.0, 2.0),
    );
    reg.register(
        EBParam::normal("beta_continuation", -0.5, 0.04)
            .with_category("risk_model")
            .with_description("Log-gamma per unit continuation probability (negative)")
            .with_hard_bounds(-1.5, 0.5),
    );
    reg.register(
        EBParam::normal("beta_edge_uncertainty", 0.5, 0.09)
            .with_category("risk_model")
            .with_description("Log-gamma per unit edge uncertainty")
            .with_hard_bounds(0.0, 3.0),
    );
    reg.register(
        EBParam::normal("beta_calibration", 0.3, 0.04)
            .with_category("risk_model")
            .with_description("Log-gamma per unit calibration error")
            .with_hard_bounds(0.0, 2.0),
    );
    reg.register(
        EBParam::normal("beta_as_ratio", 0.8, 0.09)
            .with_category("risk_model")
            .with_description("Log-gamma per unit AS ratio")
            .with_hard_bounds(0.0, 3.0),
    );

    // === Signal Weights (Phase 3) ===
    reg.register(
        EBParam::gamma("signal_precision_price", 1.0, 10.0)
            .with_category("signal")
            .with_description("Precision (1/var) of price-based directional signal")
            .with_hard_bounds(0.01, 100.0),
    );
    reg.register(
        EBParam::gamma("signal_precision_flow", 0.3, 10.0)
            .with_category("signal")
            .with_description("Precision of flow-based directional signal")
            .with_hard_bounds(0.01, 100.0),
    );
    reg.register(
        EBParam::gamma("signal_precision_fill", 0.3, 10.0)
            .with_category("signal")
            .with_description("Precision of fill-based directional signal")
            .with_hard_bounds(0.01, 100.0),
    );
    reg.register(
        EBParam::gamma("signal_precision_as", 0.5, 10.0)
            .with_category("signal")
            .with_description("Precision of AS-based directional signal")
            .with_hard_bounds(0.01, 100.0),
    );
    reg.register(
        EBParam::gamma("signal_precision_burst", 0.2, 10.0)
            .with_category("signal")
            .with_description("Precision of burst-based directional signal")
            .with_hard_bounds(0.01, 100.0),
    );

    // === EWMA Half-Lives (Phase 4) ===
    reg.register(
        EBParam::gamma("ewma_half_life_1s", 3.0, 15.0)
            .with_category("ewma")
            .with_description("Half-life in seconds for 1s EWMA")
            .with_hard_bounds(1.0, 10.0),
    );
    reg.register(
        EBParam::gamma("ewma_half_life_5s", 14.0, 15.0)
            .with_category("ewma")
            .with_description("Half-life in seconds for 5s EWMA")
            .with_hard_bounds(5.0, 30.0),
    );
    reg.register(
        EBParam::gamma("ewma_half_life_30s", 69.0, 15.0)
            .with_category("ewma")
            .with_description("Half-life in seconds for 30s EWMA")
            .with_hard_bounds(30.0, 150.0),
    );
    reg.register(
        EBParam::gamma("ewma_half_life_5m", 346.0, 15.0)
            .with_category("ewma")
            .with_description("Half-life in seconds for 5m EWMA")
            .with_hard_bounds(120.0, 600.0),
    );

    // === Regime HMM (Phase 5) ===
    reg.register(
        EBParam::gamma("regime_transition_stickiness", 10.0, 5.0)
            .with_category("regime")
            .with_description("Dirichlet concentration for diagonal (sticky) transitions")
            .with_hard_bounds(2.0, 50.0),
    );
    reg.register(
        EBParam::gamma("regime_transition_off_diag", 1.0, 5.0)
            .with_category("regime")
            .with_description("Dirichlet concentration for off-diagonal transitions")
            .with_hard_bounds(0.1, 5.0),
    );

    // === Kalman Noise (Phase 6) ===
    reg.register(
        EBParam::inverse_gamma("kalman_obs_noise_R", 1e-8, 5.0)
            .with_category("kalman")
            .with_description("Kalman observation noise variance")
            .with_hard_bounds(1e-12, 1e-4),
    );
    reg.register(
        EBParam::inverse_gamma("kalman_process_noise_Q", 1e-9, 5.0)
            .with_category("kalman")
            .with_description("Kalman process noise variance")
            .with_hard_bounds(1e-14, 1e-5),
    );

    // === Adverse Selection (Phase 7) ===
    reg.register(
        EBParam::beta("as_informed_fraction", 0.25, 8.0)
            .with_category("adverse_selection")
            .with_description("Fraction of fills from informed traders")
            .with_hard_bounds(0.01, 0.90),
    );
    reg.register(
        EBParam::gamma("as_informed_threshold_bps", 5.0, 10.0)
            .with_category("adverse_selection")
            .with_description("AS threshold for classifying informed fills")
            .with_hard_bounds(1.0, 30.0),
    );
    reg.register(
        EBParam::gamma("fill_intensity_decay_rate", 0.1, 10.0)
            .with_category("adverse_selection")
            .with_description("Exponential decay rate for fill probability with depth")
            .with_hard_bounds(0.01, 1.0),
    );

    // === Inventory Management (Phase 8) ===
    reg.register(
        EBParam::gamma("inventory_skew_scale", 0.3, 10.0)
            .with_category("inventory")
            .with_description("Scale factor for inventory-dependent quote skew")
            .with_hard_bounds(0.05, 1.0),
    );

    // === Cascade Detection (Phase 9) ===
    reg.register(
        EBParam::gamma("cascade_widen_ratio", 3.0, 8.0)
            .with_category("cascade")
            .with_description("Intensity ratio threshold for widening quotes")
            .with_hard_bounds(1.5, 10.0),
    );
    reg.register(
        EBParam::gamma("cascade_suppress_ratio", 5.0, 8.0)
            .with_category("cascade")
            .with_description("Intensity ratio threshold for suppressing quotes")
            .with_hard_bounds(2.0, 20.0),
    );

    // === Volatility (Phase 9) ===
    // Use LogNormal because sigma_baseline (0.0002) is below Gamma's floor of 0.001
    reg.register(
        EBParam::log_normal("sigma_baseline", 0.0002, 0.5) // CV=0.5 → moderate uncertainty
            .with_category("volatility")
            .with_description("Baseline per-second volatility")
            .with_hard_bounds(1e-5, 0.01),
    );

    // === Regime Thresholds (Phase 9) ===
    // Derived from theoretical_edge.rs regime classification.
    // Hawkes stability theory: process stable iff branching ratio n = α/β < 1.
    // Intensity ratio λ(t)/λ_∞ = 1/(1-n). So n=0.5 → ratio=2, n=0.8 → ratio=5.
    reg.register(
        EBParam::normal("regime_cascade_prob_threshold", 0.4, 0.01)
            .with_category("regime")
            .with_description("Cascade probability threshold for entering volatile regime")
            .with_hard_bounds(0.2, 0.8),
    );
    reg.register(
        EBParam::normal("regime_vol_ratio_high", 1.8, 0.09)
            .with_category("regime")
            .with_description("Vol ratio threshold for entering volatile regime")
            .with_hard_bounds(1.2, 3.0),
    );
    reg.register(
        EBParam::normal("regime_vol_ratio_calm", 0.6, 0.01)
            .with_category("regime")
            .with_description("Vol ratio threshold for entering calm regime")
            .with_hard_bounds(0.3, 0.9),
    );
    reg.register(
        EBParam::normal("regime_spread_ratio_calm", 0.7, 0.01)
            .with_category("regime")
            .with_description("Spread ratio threshold for calm regime")
            .with_hard_bounds(0.3, 1.0),
    );
    reg.register(
        EBParam::normal("regime_hawkes_branching_calm", 0.4, 0.01)
            .with_category("regime")
            .with_description("Hawkes branching ratio threshold for calm regime (n < this → calm)")
            .with_hard_bounds(0.1, 0.7),
    );
    // Cascade detection threshold (from quote_engine.rs)
    reg.register(
        EBParam::normal("cascade_detection_threshold", 0.3, 0.01)
            .with_category("cascade")
            .with_description("Cascade intensity threshold for circuit breaker activation")
            .with_hard_bounds(0.05, 0.8),
    );

    // === Pre-Fill Classifier Weights (Phase 7) ===
    // Signal weights for toxicity prediction. Sum need not equal 1.0 because they are
    // combined via sigmoid(Σ w_i × z_i), not a simplex mixture.
    // Priors set from domain knowledge: imbalance/flow are strongest AS predictors (Kyle 1985).
    reg.register(
        EBParam::normal("prefill_weight_imbalance", 0.30, 0.01)
            .with_category("adverse_selection")
            .with_description("Pre-fill classifier: orderbook imbalance weight")
            .with_hard_bounds(0.05, 0.80),
    );
    reg.register(
        EBParam::normal("prefill_weight_flow", 0.25, 0.01)
            .with_category("adverse_selection")
            .with_description("Pre-fill classifier: trade flow momentum weight")
            .with_hard_bounds(0.05, 0.80),
    );
    reg.register(
        EBParam::normal("prefill_weight_regime", 0.25, 0.01)
            .with_category("adverse_selection")
            .with_description("Pre-fill classifier: regime distrust weight")
            .with_hard_bounds(0.05, 0.80),
    );
    reg.register(
        EBParam::normal("prefill_weight_funding", 0.10, 0.005)
            .with_category("adverse_selection")
            .with_description("Pre-fill classifier: funding rate weight")
            .with_hard_bounds(0.01, 0.50),
    );
    reg.register(
        EBParam::normal("prefill_weight_changepoint", 0.10, 0.005)
            .with_category("adverse_selection")
            .with_description("Pre-fill classifier: BOCD changepoint weight")
            .with_hard_bounds(0.01, 0.50),
    );
    reg.register(
        EBParam::normal("prefill_weight_trend", 0.30, 0.01)
            .with_category("adverse_selection")
            .with_description("Pre-fill classifier: trend opposition weight (Kyle 1985)")
            .with_hard_bounds(0.05, 0.80),
    );

    // === Directional Kalman Observation Noise (Phase 3: Signal Weights) ===
    // Base observation noise variance for each signal source in the directional
    // Kalman filter (belief/central.rs). These are PRIOR R values; the online
    // innovation tracker adapts them via adapted_dir_noise().
    //
    // Rationale for priors:
    // - Price: z-score Var≈1.0, so R=1.0 matches signal variance → neutral prior
    // - Fill: binary ±1 with low directional info, Var=1.0 but R=16.0 → heavily discounted
    //   (fills are noisy directional indicators; ~50% informed at best)
    // - AS: magnitude-scaled ∈ [0,2], typical Var≈1.5, R=2.5 → mild discount
    // - Flow: direction ∈ [-1,1], Var≈1.0, R=1.0 → neutral
    // - Burst: rare high-info events, R=0.5 → trusted (2x weight vs price)
    reg.register(
        EBParam::gamma("dir_noise_price", 1.0, 5.0)
            .with_category("signal_weights")
            .with_description("Kalman observation noise for price z-scores (Var≈1.0)")
            .with_hard_bounds(0.1, 10.0),
    );
    reg.register(
        EBParam::gamma("dir_noise_fill", 16.0, 3.0)
            .with_category("signal_weights")
            .with_description("Kalman observation noise for fill side (binary, low info)")
            .with_hard_bounds(1.0, 50.0),
    );
    reg.register(
        EBParam::gamma("dir_noise_as", 2.5, 4.0)
            .with_category("signal_weights")
            .with_description("Kalman observation noise for AS-direction evidence")
            .with_hard_bounds(0.5, 20.0),
    );
    reg.register(
        EBParam::gamma("dir_noise_flow", 1.0, 5.0)
            .with_category("signal_weights")
            .with_description("Kalman observation noise for order flow direction")
            .with_hard_bounds(0.1, 10.0),
    );
    reg.register(
        EBParam::gamma("dir_noise_burst", 0.5, 5.0)
            .with_category("signal_weights")
            .with_description("Kalman observation noise for burst events (rare, high-info)")
            .with_hard_bounds(0.05, 5.0),
    );

    // === Signal Integration Caps (Phase 3) ===
    // Caps on signal contributions from signal_integration.rs.
    // These limit the maximum effect any single signal can have on spreads/skew.
    reg.register(
        EBParam::gamma("max_lead_lag_skew_bps", 15.0, 5.0)
            .with_category("signal_caps")
            .with_description("Maximum skew from Binance lead-lag signal")
            .with_hard_bounds(3.0, 50.0),
    );
    reg.register(
        EBParam::gamma("max_spread_adjustment_bps", 20.0, 5.0)
            .with_category("signal_caps")
            .with_description("Maximum total additive spread adjustment from all signals")
            .with_hard_bounds(5.0, 100.0),
    );
    reg.register(
        EBParam::gamma("flow_urgency_max_bps", 10.0, 5.0)
            .with_category("signal_caps")
            .with_description("Maximum flow urgency skew when imbalance > 0.6")
            .with_hard_bounds(2.0, 50.0),
    );
    reg.register(
        EBParam::gamma("signal_skew_max_bps", 3.0, 5.0)
            .with_category("signal_caps")
            .with_description("Maximum signal-based directional skew from alpha/trend")
            .with_hard_bounds(0.5, 20.0),
    );

    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eb_param_normal() {
        let p = EBParam::normal("test_beta", 0.3, 0.01); // var=0.01 → std=0.1
        assert!(
            (p.value() - 0.3).abs() < 0.01,
            "initial value should be near prior mean: {}",
            p.value()
        );
        assert!(p.prior_variance() > 0.0);
        assert!(!p.is_calibrated());
    }

    #[test]
    fn test_eb_param_gamma() {
        let p = EBParam::gamma("test_rate", 2.0, 10.0);
        assert!(p.value() > 0.0);
        assert!((p.value() - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_eb_param_beta() {
        let p = EBParam::beta("test_prob", 0.25, 8.0);
        assert!(p.value() >= 0.0 && p.value() <= 1.0);
        assert!((p.value() - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_eb_param_log_normal() {
        let p = EBParam::log_normal("test_mult", 1.5, 0.3); // cv=0.3
        assert!(p.value() > 0.0);
    }

    #[test]
    fn test_eb_param_hard_bounds() {
        let mut p = EBParam::normal("bounded", 0.5, 1.0) // wide prior
            .with_hard_bounds(0.0, 1.0);

        // Observe extreme values — hard bounds should clamp
        for _ in 0..100 {
            p.observe_normal(10.0, 0.01);
        }
        assert!(
            p.value() <= 1.0,
            "hard upper bound must hold: {}",
            p.value()
        );

        p.reset_to_prior();
        for _ in 0..100 {
            p.observe_normal(-10.0, 0.01);
        }
        assert!(
            p.value() >= 0.0,
            "hard lower bound must hold: {}",
            p.value()
        );
    }

    #[test]
    fn test_eb_param_observe_convergence() {
        let mut p = EBParam::normal("converge", 0.0, 1.0); // wide prior var

        // Observe many values near 1.0 with low obs variance
        for _ in 0..100 {
            p.observe_normal(1.0, 0.01);
        }
        assert!(
            (p.value() - 1.0).abs() < 0.1,
            "should converge to observed mean: {}",
            p.value()
        );
        assert!(p.is_calibrated());
    }

    #[test]
    fn test_eb_param_credible_bounds() {
        let p = EBParam::normal("ci_test", 0.5, 0.04); // std=0.2
        let (lo, hi) = p.bounds();
        assert!(lo < 0.5);
        assert!(hi > 0.5);
        assert!(lo < hi);
    }

    #[test]
    fn test_registry_basic() {
        let mut reg = ParameterRegistry::new();
        reg.register(EBParam::normal("x", 1.0, 0.25).with_category("test")); // std=0.5
        reg.register(EBParam::beta("p", 0.3, 8.0).with_category("test"));

        assert_eq!(reg.len(), 2);
        assert!(reg.contains("x"));
        assert!(!reg.contains("y"));

        assert!((reg.get_value("x").unwrap() - 1.0).abs() < 0.1);
        assert!(reg.get_value("y").is_none());
        assert!((reg.get_value_or("y", 42.0) - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_registry_observe() {
        let mut reg = ParameterRegistry::new();
        reg.register(EBParam::normal("x", 0.0, 1.0)); // wide prior

        for _ in 0..50 {
            reg.observe("x", 2.0, 0.01); // low obs variance → strong signal
        }

        let val = reg.get_value("x").unwrap();
        assert!(val > 1.0, "should move toward observed mean: {}", val);
    }

    #[test]
    fn test_registry_by_category() {
        let mut reg = ParameterRegistry::new();
        reg.register(EBParam::normal("a", 1.0, 0.01).with_category("cat_a"));
        reg.register(EBParam::normal("b", 2.0, 0.01).with_category("cat_a"));
        reg.register(EBParam::normal("c", 3.0, 0.01).with_category("cat_b"));

        let cat_a = reg.by_category("cat_a");
        assert_eq!(cat_a.len(), 2);
        let cat_b = reg.by_category("cat_b");
        assert_eq!(cat_b.len(), 1);
    }

    #[test]
    fn test_registry_reset_category() {
        let mut reg = ParameterRegistry::new();
        reg.register(EBParam::normal("a", 0.0, 1.0).with_category("cat_a"));
        reg.register(EBParam::normal("b", 0.0, 1.0).with_category("cat_b"));

        // Observe values for both
        for _ in 0..50 {
            reg.observe("a", 5.0, 0.01);
            reg.observe("b", 5.0, 0.01);
        }

        // Reset only cat_a
        reg.reset_category("cat_a");

        let a = reg.get_value("a").unwrap();
        let b = reg.get_value("b").unwrap();
        assert!(a < 1.0, "'a' should be near prior after reset: {}", a);
        assert!(b > 3.0, "'b' should still be near observed: {}", b);
    }

    #[test]
    fn test_registry_checkpoint_roundtrip() {
        let mut reg = ParameterRegistry::new();
        reg.register(EBParam::normal("x", 0.0, 1.0));

        for _ in 0..30 {
            reg.observe("x", 3.0, 0.01);
        }

        // Serialize
        let json = serde_json::to_string(&reg).expect("serialize");

        // Deserialize
        let restored: ParameterRegistry = serde_json::from_str(&json).expect("deserialize");

        let orig_val = reg.get_value("x").unwrap();
        let restored_val = restored.get_value("x").unwrap();
        assert!(
            (orig_val - restored_val).abs() < 1e-10,
            "checkpoint roundtrip: {} vs {}",
            orig_val,
            restored_val
        );
    }

    #[test]
    fn test_registry_merge() {
        let mut base = ParameterRegistry::new();
        base.register(EBParam::normal("x", 0.0, 1.0));
        base.register(EBParam::normal("y", 0.0, 1.0));

        let mut checkpoint = ParameterRegistry::new();
        checkpoint.register(EBParam::normal("x", 0.0, 1.0));
        // Simulate observed data in checkpoint
        for _ in 0..30 {
            checkpoint.observe("x", 5.0, 0.01);
        }

        base.merge_from(&checkpoint);

        let x_val = base.get_value("x").unwrap();
        assert!(
            x_val > 2.0,
            "should have merged checkpoint state: {}",
            x_val
        );

        // y should be unchanged (not in checkpoint)
        let y_val = base.get_value("y").unwrap();
        assert!((y_val - 0.0).abs() < 0.5, "y should be at prior: {}", y_val);
    }

    #[test]
    fn test_default_registry() {
        let reg = create_default_registry();

        // Should have all the expected categories
        assert!(!reg.by_category("risk_model").is_empty());
        assert!(!reg.by_category("signal").is_empty());
        assert!(!reg.by_category("ewma").is_empty());
        assert!(!reg.by_category("regime").is_empty());
        assert!(!reg.by_category("adverse_selection").is_empty());
        assert!(!reg.by_category("cascade").is_empty());
        assert!(!reg.by_category("volatility").is_empty());

        // All values should be within hard bounds
        for p in reg.params.values() {
            let val = p.value();
            assert!(
                val >= p.hard_lower && val <= p.hard_upper,
                "{}: {} not in [{}, {}]",
                p.param.name,
                val,
                p.hard_lower,
                p.hard_upper
            );
        }

        // Check specific priors match CalibratedRiskModel::default()
        let beta_vol = reg.get_value("beta_volatility").unwrap();
        assert!((beta_vol - 1.0).abs() < 0.1, "beta_vol = {}", beta_vol);

        let sigma = reg.get_value("sigma_baseline").unwrap();
        assert!((sigma - 0.0002).abs() < 0.0001, "sigma = {}", sigma);
    }

    #[test]
    fn test_summary_display() {
        let reg = create_default_registry();
        let summaries = reg.summary();
        assert!(!summaries.is_empty());

        // Check Display impl doesn't panic
        for s in &summaries {
            let _display = format!("{}", s);
        }
    }

    #[test]
    fn test_calibration_fraction() {
        let mut reg = ParameterRegistry::new();
        reg.register(EBParam::normal("a", 0.0, 1.0));
        reg.register(EBParam::normal("b", 0.0, 1.0));

        assert_eq!(reg.calibration_fraction(), 0.0);

        // Calibrate 'a' by adding enough observations
        for _ in 0..100 {
            reg.observe("a", 1.0, 0.01);
        }

        assert!(reg.calibration_fraction() > 0.0);
    }

    #[test]
    fn test_beta_observe_counts() {
        let mut reg = ParameterRegistry::new();
        reg.register(EBParam::beta("informed_rate", 0.25, 8.0));

        // Observe 30 out of 100 fills as informed
        reg.observe_beta("informed_rate", 30, 70);

        let val = reg.get_value("informed_rate").unwrap();
        // With prior Beta(2,6) + data Beta(30,70) → posterior Beta(32,76)
        // E = 32/108 ≈ 0.296
        assert!(val > 0.20 && val < 0.40, "informed_rate = {}", val);
    }

    #[test]
    fn test_param_summary_fields() {
        let reg = create_default_registry();
        let summaries = reg.summary();

        for s in &summaries {
            assert!(!s.name.is_empty());
            assert!(s.ess > 0.0);
            assert!(s.ci_95_lower <= s.value);
            assert!(s.ci_95_upper >= s.value);
        }
    }

    #[test]
    fn test_inverse_gamma_param() {
        let mut p = EBParam::inverse_gamma("test_var", 0.01, 10.0);
        assert!(p.value() > 0.0);

        // Observe squared residuals
        for _ in 0..30 {
            p.observe_variance(0.02);
        }
        // Should move toward observed variance
        assert!(p.value() > 0.005, "should adapt: {}", p.value());
    }

    #[test]
    fn test_gamma_observe() {
        let mut p = EBParam::gamma("test_rate", 1.0, 5.0);

        for _ in 0..30 {
            p.observe_gamma_value(3.0);
        }
        // Should move toward 3.0
        assert!(p.value() > 1.5, "should adapt: {}", p.value());
    }
}
