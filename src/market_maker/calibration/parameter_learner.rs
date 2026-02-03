//! Bayesian Parameter Learning Infrastructure
//!
//! Provides online parameter learning with regularization to prevent overfitting.
//! Every magic number is replaced with a `BayesianParam` that:
//! 1. Uses the magic number as a prior mean (regularization)
//! 2. Updates from observed data using conjugate Bayesian updates
//! 3. Shrinks toward the prior when data is scarce
//!
//! # Design Principles
//!
//! 1. **Bayesian Regularization**: Prior prevents overfitting with small samples
//! 2. **Online Learning**: Parameters update incrementally, no batch retraining
//! 3. **Uncertainty Quantification**: All parameters have credible intervals
//! 4. **Graceful Degradation**: Falls back to prior when data is unavailable
//!
//! # Prior Elicitation
//!
//! Each parameter's prior is chosen based on:
//! - **Prior mean**: The previous "magic number" (domain knowledge baseline)
//! - **Prior strength**: How many pseudo-observations (regularization strength)
//! - **Distribution family**: Conjugate to the likelihood for efficient updates
//!
//! # Example: Alpha Touch Learning
//!
//! ```rust,ignore
//! // Prior: Beta(2, 6) → E[α] = 0.25, prior_strength = 8 pseudo-observations
//! let mut alpha_touch = BayesianParam::beta(0.25, 8.0);
//!
//! // After observing fills: 3 informed out of 10 total
//! alpha_touch.observe_beta(3, 7); // 3 successes, 7 failures
//!
//! // Posterior: Beta(5, 13) → E[α] = 5/18 ≈ 0.28
//! // Not 0.30 (raw MLE) because prior regularizes toward 0.25
//! let estimate = alpha_touch.estimate(); // Shrinkage estimate
//! ```

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Conjugate distribution family for the parameter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PriorFamily {
    /// Beta(α, β) for probabilities [0, 1]
    /// Conjugate to Binomial likelihood
    Beta,
    /// Gamma(shape, rate) for positive rates
    /// Conjugate to Poisson/Exponential likelihood
    Gamma,
    /// Normal(μ, σ²) for unbounded parameters
    /// Conjugate to Normal likelihood with known variance
    Normal,
    /// Inverse-Gamma(α, β) for variances
    /// Conjugate to Normal likelihood for variance estimation
    InverseGamma,
    /// Log-Normal(μ, σ²) for positive multiplicative parameters
    LogNormal,
}

/// Bayesian parameter with prior + posterior for regularized learning.
///
/// Implements shrinkage estimation: with few samples, estimate shrinks toward
/// the prior mean. With many samples, estimate approaches MLE.
///
/// # Shrinkage Formula
///
/// ```text
/// θ_posterior = w × θ_MLE + (1-w) × θ_prior
/// where w = n / (n + prior_strength)
/// ```
///
/// - N=0: θ = θ_prior (falls back to domain knowledge)
/// - N=prior_strength: θ = 0.5×θ_MLE + 0.5×θ_prior (balanced)
/// - N→∞: θ → θ_MLE (data dominates)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianParam {
    /// Name for logging and debugging
    pub name: String,

    /// Prior distribution family
    pub family: PriorFamily,

    /// Prior mean (the original "magic number" or domain knowledge)
    pub prior_mean: f64,

    /// Prior strength in pseudo-observations
    /// Higher = more regularization, slower adaptation
    pub prior_strength: f64,

    /// First parameter of posterior distribution
    /// - Beta: α (successes + prior_α)
    /// - Gamma: shape (prior_shape + n)
    /// - Normal: mean (weighted average)
    /// - InverseGamma: α (prior_α + n/2)
    /// - LogNormal: μ (log-space mean)
    pub posterior_param1: f64,

    /// Second parameter of posterior distribution
    /// - Beta: β (failures + prior_β)
    /// - Gamma: rate (prior_rate + sum_x)
    /// - Normal: variance (posterior variance)
    /// - InverseGamma: β (prior_β + sum_sq/2)
    /// - LogNormal: σ² (log-space variance)
    pub posterior_param2: f64,

    /// Number of observations incorporated
    pub n_observations: usize,

    /// Sum of observations (for mean estimation)
    sum_x: f64,

    /// Sum of squared observations (for variance estimation)
    sum_x2: f64,

    /// Last update timestamp (skipped in serialization - not persisted)
    #[serde(skip)]
    pub last_updated: Option<Instant>,
}

impl BayesianParam {
    // ==================== Constructors ====================

    /// Create a Beta-distributed parameter for probabilities in [0, 1].
    ///
    /// # Arguments
    /// * `prior_mean` - Expected probability (e.g., 0.25 for 25% informed)
    /// * `prior_strength` - Pseudo-observations (e.g., 8 → moderate confidence)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Prior: Beta(2, 6) → E[α] = 0.25, variance ≈ 0.02
    /// let alpha = BayesianParam::beta("alpha_touch", 0.25, 8.0);
    /// ```
    pub fn beta(name: impl Into<String>, prior_mean: f64, prior_strength: f64) -> Self {
        let prior_mean = prior_mean.clamp(0.001, 0.999);
        let prior_alpha = prior_mean * prior_strength;
        let prior_beta = (1.0 - prior_mean) * prior_strength;

        Self {
            name: name.into(),
            family: PriorFamily::Beta,
            prior_mean,
            prior_strength,
            posterior_param1: prior_alpha,
            posterior_param2: prior_beta,
            n_observations: 0,
            sum_x: 0.0,
            sum_x2: 0.0,
            last_updated: None,
        }
    }

    /// Create a Gamma-distributed parameter for positive rates.
    ///
    /// # Arguments
    /// * `prior_mean` - Expected rate (e.g., 2000 for kappa)
    /// * `prior_strength` - Pseudo-observations (e.g., 10 → weak confidence)
    ///
    /// # Gamma Parameterization
    /// shape = prior_strength, rate = prior_strength / prior_mean
    /// E[X] = shape/rate = prior_mean
    ///
    /// # Example
    /// ```rust,ignore
    /// // Prior: Gamma(10, 0.005) → E[κ] = 2000, CV ≈ 0.32
    /// let kappa = BayesianParam::gamma("kappa", 2000.0, 10.0);
    /// ```
    pub fn gamma(name: impl Into<String>, prior_mean: f64, prior_strength: f64) -> Self {
        let prior_mean = prior_mean.max(0.001);
        let shape = prior_strength;
        let rate = prior_strength / prior_mean;

        Self {
            name: name.into(),
            family: PriorFamily::Gamma,
            prior_mean,
            prior_strength,
            posterior_param1: shape,
            posterior_param2: rate,
            n_observations: 0,
            sum_x: 0.0,
            sum_x2: 0.0,
            last_updated: None,
        }
    }

    /// Create a Normal-distributed parameter for unbounded values.
    ///
    /// # Arguments
    /// * `prior_mean` - Expected value
    /// * `prior_var` - Prior variance (uncertainty in prior_mean)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Prior: Normal(2.0, 0.5²) → E[β] = 2.0, 95% CI ≈ [1, 3]
    /// let beta = BayesianParam::normal("flow_sensitivity", 2.0, 0.25);
    /// ```
    pub fn normal(name: impl Into<String>, prior_mean: f64, prior_var: f64) -> Self {
        let prior_var = prior_var.max(1e-10);
        let prior_strength = 1.0 / prior_var; // Precision as strength

        Self {
            name: name.into(),
            family: PriorFamily::Normal,
            prior_mean,
            prior_strength,
            posterior_param1: prior_mean,
            posterior_param2: prior_var,
            n_observations: 0,
            sum_x: 0.0,
            sum_x2: 0.0,
            last_updated: None,
        }
    }

    /// Create an Inverse-Gamma parameter for variances.
    ///
    /// # Arguments
    /// * `prior_mean` - Expected variance
    /// * `prior_strength` - Degrees of freedom (pseudo-observations)
    ///
    /// # InverseGamma Parameterization
    /// α = prior_strength / 2, β = prior_mean × α
    /// E[σ²] = β / (α - 1) ≈ prior_mean for large α
    pub fn inverse_gamma(name: impl Into<String>, prior_mean: f64, prior_strength: f64) -> Self {
        let prior_mean = prior_mean.max(1e-10);
        let alpha = (prior_strength / 2.0).max(2.0); // α > 1 for finite mean
        let beta = prior_mean * (alpha - 1.0);

        Self {
            name: name.into(),
            family: PriorFamily::InverseGamma,
            prior_mean,
            prior_strength,
            posterior_param1: alpha,
            posterior_param2: beta,
            n_observations: 0,
            sum_x: 0.0,
            sum_x2: 0.0,
            last_updated: None,
        }
    }

    /// Create a Log-Normal parameter for positive multiplicative values.
    ///
    /// # Arguments
    /// * `prior_mean` - Expected value in natural space
    /// * `prior_cv` - Prior coefficient of variation (CV = σ/μ)
    pub fn log_normal(name: impl Into<String>, prior_mean: f64, prior_cv: f64) -> Self {
        let prior_mean = prior_mean.max(1e-10);
        let prior_cv = prior_cv.max(0.01);

        // Convert to log-space parameters
        // If E[X] = μ and CV = σ/μ, then in log-space:
        // σ² = ln(1 + CV²), μ_log = ln(μ) - σ²/2
        let log_var = (1.0 + prior_cv * prior_cv).ln();
        let log_mean = prior_mean.ln() - log_var / 2.0;

        Self {
            name: name.into(),
            family: PriorFamily::LogNormal,
            prior_mean,
            prior_strength: 1.0 / log_var.sqrt(), // Precision in log-space
            posterior_param1: log_mean,
            posterior_param2: log_var,
            n_observations: 0,
            sum_x: 0.0,
            sum_x2: 0.0,
            last_updated: None,
        }
    }

    // ==================== Observation Methods ====================

    /// Observe a binary outcome for Beta-distributed parameters.
    ///
    /// # Arguments
    /// * `successes` - Number of successes (e.g., informed fills)
    /// * `failures` - Number of failures (e.g., uninformed fills)
    pub fn observe_beta(&mut self, successes: usize, failures: usize) {
        if self.family != PriorFamily::Beta {
            tracing::warn!(
                "observe_beta called on non-Beta parameter {}",
                self.name
            );
            return;
        }

        self.posterior_param1 += successes as f64;
        self.posterior_param2 += failures as f64;
        self.n_observations += successes + failures;
        self.last_updated = Some(Instant::now());
    }

    /// Observe count data for Gamma-distributed parameters (Poisson likelihood).
    ///
    /// # Arguments
    /// * `count` - Observed count (e.g., number of fills)
    /// * `exposure` - Exposure time or space (e.g., seconds observed)
    pub fn observe_gamma_poisson(&mut self, count: usize, exposure: f64) {
        if self.family != PriorFamily::Gamma {
            tracing::warn!(
                "observe_gamma_poisson called on non-Gamma parameter {}",
                self.name
            );
            return;
        }

        // Gamma-Poisson conjugate update:
        // posterior_shape = prior_shape + count
        // posterior_rate = prior_rate + exposure
        self.posterior_param1 += count as f64;
        self.posterior_param2 += exposure;
        self.n_observations += 1;
        self.sum_x += count as f64;
        self.last_updated = Some(Instant::now());
    }

    /// Observe a continuous value for Gamma-distributed parameters (Exponential likelihood).
    ///
    /// # Arguments
    /// * `value` - Observed positive value (e.g., inter-arrival time)
    pub fn observe_gamma_exponential(&mut self, value: f64) {
        if self.family != PriorFamily::Gamma || value <= 0.0 {
            return;
        }

        // Gamma-Exponential conjugate update (for rate parameter):
        // posterior_shape = prior_shape + 1
        // posterior_rate = prior_rate + value
        self.posterior_param1 += 1.0;
        self.posterior_param2 += value;
        self.n_observations += 1;
        self.sum_x += value;
        self.last_updated = Some(Instant::now());
    }

    /// Observe a continuous value for Normal-distributed parameters.
    ///
    /// Assumes known variance (observation_var) for conjugate update.
    ///
    /// # Arguments
    /// * `value` - Observed value
    /// * `observation_var` - Variance of the observation (known)
    pub fn observe_normal(&mut self, value: f64, observation_var: f64) {
        if self.family != PriorFamily::Normal {
            return;
        }

        let observation_var = observation_var.max(1e-10);

        // Normal-Normal conjugate update:
        // posterior_precision = prior_precision + 1/obs_var
        // posterior_mean = (prior_precision × prior_mean + value/obs_var) / posterior_precision
        let prior_precision = 1.0 / self.posterior_param2;
        let obs_precision = 1.0 / observation_var;
        let posterior_precision = prior_precision + obs_precision;
        let posterior_mean =
            (prior_precision * self.posterior_param1 + obs_precision * value) / posterior_precision;

        self.posterior_param1 = posterior_mean;
        self.posterior_param2 = 1.0 / posterior_precision;
        self.n_observations += 1;
        self.sum_x += value;
        self.sum_x2 += value * value;
        self.last_updated = Some(Instant::now());
    }

    /// Observe a squared residual for variance estimation (Inverse-Gamma).
    ///
    /// # Arguments
    /// * `squared_residual` - (x - μ)² for a single observation
    pub fn observe_variance(&mut self, squared_residual: f64) {
        if self.family != PriorFamily::InverseGamma {
            return;
        }

        // InverseGamma-Normal (for variance) conjugate update:
        // posterior_α = prior_α + 0.5
        // posterior_β = prior_β + 0.5 × (x - μ)²
        self.posterior_param1 += 0.5;
        self.posterior_param2 += 0.5 * squared_residual;
        self.n_observations += 1;
        self.sum_x2 += squared_residual;
        self.last_updated = Some(Instant::now());
    }

    /// Observe a positive value for Log-Normal parameters.
    ///
    /// # Arguments
    /// * `value` - Observed positive value
    pub fn observe_log_normal(&mut self, value: f64) {
        if self.family != PriorFamily::LogNormal || value <= 0.0 {
            return;
        }

        let log_value = value.ln();

        // Track log-space statistics
        self.n_observations += 1;
        self.sum_x += log_value;
        self.sum_x2 += log_value * log_value;

        // Update log-space mean using running average
        let n = self.n_observations as f64;
        let sample_mean = self.sum_x / n;

        // Bayesian update: blend prior and sample mean
        let prior_precision = self.prior_strength;
        let sample_precision = n; // Assuming unit variance observations
        let posterior_precision = prior_precision + sample_precision;

        let prior_log_mean = self.prior_mean.ln() - self.posterior_param2 / 2.0;
        self.posterior_param1 =
            (prior_precision * prior_log_mean + sample_precision * sample_mean) / posterior_precision;

        self.last_updated = Some(Instant::now());
    }

    // ==================== Estimation Methods ====================

    /// Get the shrinkage estimate (posterior mean).
    ///
    /// This is the Bayes-optimal point estimate under squared error loss.
    /// With few observations, it shrinks toward the prior mean.
    /// With many observations, it converges to the MLE.
    pub fn estimate(&self) -> f64 {
        match self.family {
            PriorFamily::Beta => {
                // Beta mean: α / (α + β)
                self.posterior_param1 / (self.posterior_param1 + self.posterior_param2)
            }
            PriorFamily::Gamma => {
                // Gamma mean: shape / rate
                self.posterior_param1 / self.posterior_param2.max(1e-10)
            }
            PriorFamily::Normal => {
                // Normal mean: directly stored
                self.posterior_param1
            }
            PriorFamily::InverseGamma => {
                // InverseGamma mean: β / (α - 1) for α > 1
                if self.posterior_param1 > 1.0 {
                    self.posterior_param2 / (self.posterior_param1 - 1.0)
                } else {
                    self.prior_mean
                }
            }
            PriorFamily::LogNormal => {
                // LogNormal mean: exp(μ + σ²/2)
                (self.posterior_param1 + self.posterior_param2 / 2.0).exp()
            }
        }
    }

    /// Get the posterior variance (uncertainty in the estimate).
    pub fn variance(&self) -> f64 {
        match self.family {
            PriorFamily::Beta => {
                // Beta variance: αβ / ((α+β)²(α+β+1))
                let sum = self.posterior_param1 + self.posterior_param2;
                (self.posterior_param1 * self.posterior_param2) / (sum * sum * (sum + 1.0))
            }
            PriorFamily::Gamma => {
                // Gamma variance: shape / rate²
                let rate_sq = self.posterior_param2.powi(2).max(1e-10);
                self.posterior_param1 / rate_sq
            }
            PriorFamily::Normal => {
                // Normal variance: directly stored
                self.posterior_param2
            }
            PriorFamily::InverseGamma => {
                // InverseGamma variance: β² / ((α-1)²(α-2)) for α > 2
                if self.posterior_param1 > 2.0 {
                    let denom =
                        (self.posterior_param1 - 1.0).powi(2) * (self.posterior_param1 - 2.0);
                    self.posterior_param2.powi(2) / denom
                } else {
                    self.prior_mean.powi(2) // Fall back to prior-based variance
                }
            }
            PriorFamily::LogNormal => {
                // LogNormal variance: (exp(σ²) - 1) × exp(2μ + σ²)
                let exp_var = self.posterior_param2.exp();
                (exp_var - 1.0) * (2.0 * self.posterior_param1 + self.posterior_param2).exp()
            }
        }
    }

    /// Get the posterior standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the coefficient of variation (std / mean).
    pub fn cv(&self) -> f64 {
        let mean = self.estimate();
        if mean.abs() < 1e-10 {
            f64::INFINITY
        } else {
            self.std_dev() / mean.abs()
        }
    }

    /// Get the 95% credible interval.
    ///
    /// Returns (lower, upper) bounds containing 95% of the posterior mass.
    /// Uses normal approximation for simplicity.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        let mean = self.estimate();
        let std = self.std_dev();
        let z = 1.96; // 97.5th percentile of standard normal

        match self.family {
            PriorFamily::Beta => {
                // Beta is bounded [0, 1]
                ((mean - z * std).max(0.0), (mean + z * std).min(1.0))
            }
            PriorFamily::Gamma | PriorFamily::InverseGamma | PriorFamily::LogNormal => {
                // These are positive
                ((mean - z * std).max(0.0), mean + z * std)
            }
            PriorFamily::Normal => (mean - z * std, mean + z * std),
        }
    }

    /// Get the effective sample size.
    ///
    /// For Bayesian updates, ESS accounts for prior strength:
    /// ESS = n_observations + prior_strength
    pub fn effective_sample_size(&self) -> f64 {
        self.n_observations as f64 + self.prior_strength
    }

    /// Check if the parameter has enough data to be considered calibrated.
    ///
    /// # Arguments
    /// * `min_observations` - Minimum real observations required
    /// * `max_cv` - Maximum coefficient of variation allowed
    pub fn is_calibrated(&self, min_observations: usize, max_cv: f64) -> bool {
        self.n_observations >= min_observations && self.cv() < max_cv
    }

    /// Get the shrinkage weight toward the prior.
    ///
    /// Returns a value in [0, 1]:
    /// - 1.0 = fully data-driven (MLE)
    /// - 0.0 = fully prior-driven
    pub fn data_weight(&self) -> f64 {
        let n = self.n_observations as f64;
        n / (n + self.prior_strength)
    }

    /// Reset to prior (for regime changes or re-initialization).
    pub fn reset_to_prior(&mut self) {
        match self.family {
            PriorFamily::Beta => {
                let prior_alpha = self.prior_mean * self.prior_strength;
                let prior_beta = (1.0 - self.prior_mean) * self.prior_strength;
                self.posterior_param1 = prior_alpha;
                self.posterior_param2 = prior_beta;
            }
            PriorFamily::Gamma => {
                self.posterior_param1 = self.prior_strength;
                self.posterior_param2 = self.prior_strength / self.prior_mean;
            }
            PriorFamily::Normal => {
                self.posterior_param1 = self.prior_mean;
                self.posterior_param2 = 1.0 / self.prior_strength;
            }
            PriorFamily::InverseGamma => {
                let alpha = (self.prior_strength / 2.0).max(2.0);
                self.posterior_param1 = alpha;
                self.posterior_param2 = self.prior_mean * (alpha - 1.0);
            }
            PriorFamily::LogNormal => {
                let log_var = self.posterior_param2; // Preserve original variance
                self.posterior_param1 = self.prior_mean.ln() - log_var / 2.0;
            }
        }
        self.n_observations = 0;
        self.sum_x = 0.0;
        self.sum_x2 = 0.0;
        self.last_updated = None;
    }
}

/// Parameters learned from data with Bayesian regularization.
///
/// Replaces magic numbers with statistically grounded estimates.
/// Each parameter has:
/// 1. A prior based on domain knowledge (the old magic number)
/// 2. Online learning from observed data
/// 3. Uncertainty quantification
///
/// # Usage
///
/// ```rust,ignore
/// let mut params = LearnedParameters::default();
///
/// // Record fills and update alpha_touch
/// params.alpha_touch.observe_beta(informed_count, uninformed_count);
///
/// // Use the regularized estimate
/// let alpha = params.alpha_touch.estimate();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedParameters {
    // ==================== Tier 1: P&L Critical ====================

    /// Informed trader probability at touch.
    /// Prior: Beta(2, 6) → E[α] = 0.25, based on historical analysis
    pub alpha_touch: BayesianParam,

    /// Base risk aversion parameter γ.
    /// Prior: Gamma(3, 20) → E[γ] = 0.15, from GLFT derivation
    pub gamma_base: BayesianParam,

    /// Minimum spread floor in bps.
    /// Prior: Normal(5, 2²) → E[δ] = 5 bps, covers fees + slippage
    pub spread_floor_bps: BayesianParam,

    /// Proactive skew sensitivity (momentum → skew).
    /// Prior: Normal(2.0, 0.5²) → E[β] = 2.0 bps per unit momentum
    pub proactive_skew_sensitivity: BayesianParam,

    /// Quote gate edge threshold.
    /// Prior: Beta(15, 85) → E = 0.15, from IR analysis
    pub quote_gate_edge_threshold: BayesianParam,

    /// Toxic hour gamma multiplier.
    /// Prior: LogNormal(2.0, 0.3) → E[mult] = 2.0, CV = 0.3
    pub toxic_hour_gamma_mult: BayesianParam,

    /// Predictive bias sensitivity.
    /// Prior: Normal(2.0, 1.0²) → E = 2σ move on changepoint
    pub predictive_bias_sensitivity: BayesianParam,

    // ==================== Tier 2: Risk Management ====================

    /// Maximum daily loss as fraction of account.
    /// Prior: Beta(1, 49) → E = 0.02 (2% Kelly-scaled)
    pub max_daily_loss_fraction: BayesianParam,

    /// Maximum drawdown threshold.
    /// Prior: Beta(1, 19) → E = 0.05 (5% VaR-based)
    pub max_drawdown: BayesianParam,

    /// Cascade detection threshold (OI drop).
    /// Prior: Beta(2, 98) → E = 0.02 (2% OI drop)
    pub cascade_oi_threshold: BayesianParam,

    /// BOCPD hazard rate (1 / expected regime duration).
    /// Prior: Gamma(1, 250) → E = 0.004 (1/250 samples)
    pub bocpd_hazard_rate: BayesianParam,

    /// BOCPD changepoint threshold.
    /// Prior: Beta(7, 3) → E = 0.7
    pub bocpd_threshold: BayesianParam,

    /// Quote latch threshold in bps.
    /// Prior: Normal(2.5, 1.0²) → E = 2.5 bps
    pub quote_latch_threshold_bps: BayesianParam,

    // ==================== Tier 3: Calibration ====================

    /// Fill intensity κ (fills per unit spread).
    /// Prior: Gamma(4, 0.002) → E = 2000
    pub kappa: BayesianParam,

    /// Hawkes process baseline intensity μ.
    /// Prior: Gamma(5, 10) → E = 0.5
    pub hawkes_mu: BayesianParam,

    /// Hawkes process excitation α (α < β for stability).
    /// Prior: Beta(3, 7) → E = 0.3
    pub hawkes_alpha: BayesianParam,

    /// Hawkes process decay β.
    /// Prior: Gamma(1, 10) → E = 0.1
    pub hawkes_beta: BayesianParam,

    /// EWMA decay factor for kappa smoothing.
    /// Prior: Beta(90, 10) → E = 0.9 (from autocorrelation analysis)
    pub kappa_ewma_alpha: BayesianParam,

    /// Kelly tracker decay factor.
    /// Prior: Beta(99, 1) → E = 0.99 (100-trade half-life)
    pub kelly_tracker_decay: BayesianParam,

    /// Regime transition stickiness (diagonal of transition matrix).
    /// Prior: Beta(95, 5) → E = 0.95
    pub regime_sticky_diagonal: BayesianParam,

    // ==================== Tier 4: Microstructure ====================

    /// Kalman filter process noise Q.
    /// Prior: InverseGamma(10, 9e-8) → E = 1e-8
    pub kalman_q: BayesianParam,

    /// Kalman filter observation noise R.
    /// Prior: InverseGamma(10, 2.25e-8) → E = 2.5e-9
    pub kalman_r: BayesianParam,

    /// Momentum normalizer in bps.
    /// Prior: Gamma(4, 0.2) → E = 20 bps
    pub momentum_normalizer_bps: BayesianParam,

    /// Depth spacing ratio for ladder levels.
    /// Prior: LogNormal(1.5, 0.2) → E = 1.5, CV = 0.2
    pub depth_spacing_ratio: BayesianParam,

    /// Fill probability width in bps.
    /// Prior: Normal(2.0, 0.5²) → E = 2 bps
    pub fill_probability_width_bps: BayesianParam,

    /// Microprice decay factor.
    /// Prior: Beta(999, 1) → E = 0.999
    pub microprice_decay: BayesianParam,

    // ==================== Metadata ====================

    /// Last full calibration timestamp (skipped in serialization - not persisted)
    #[serde(skip)]
    pub last_calibration: Option<Instant>,

    /// Total fills observed across all parameters
    pub total_fills_observed: usize,
}

impl Default for LearnedParameters {
    fn default() -> Self {
        Self {
            // Tier 1: P&L Critical
            alpha_touch: BayesianParam::beta("alpha_touch", 0.25, 8.0),
            gamma_base: BayesianParam::gamma("gamma_base", 0.15, 3.0),
            spread_floor_bps: BayesianParam::normal("spread_floor_bps", 5.0, 4.0),
            proactive_skew_sensitivity: BayesianParam::normal("proactive_skew_sensitivity", 2.0, 0.25),
            quote_gate_edge_threshold: BayesianParam::beta("quote_gate_edge_threshold", 0.15, 100.0),
            toxic_hour_gamma_mult: BayesianParam::log_normal("toxic_hour_gamma_mult", 2.0, 0.3),
            predictive_bias_sensitivity: BayesianParam::normal("predictive_bias_sensitivity", 2.0, 1.0),

            // Tier 2: Risk Management
            max_daily_loss_fraction: BayesianParam::beta("max_daily_loss_fraction", 0.02, 50.0),
            max_drawdown: BayesianParam::beta("max_drawdown", 0.05, 20.0),
            cascade_oi_threshold: BayesianParam::beta("cascade_oi_threshold", 0.02, 100.0),
            bocpd_hazard_rate: BayesianParam::gamma("bocpd_hazard_rate", 0.004, 1.0),
            bocpd_threshold: BayesianParam::beta("bocpd_threshold", 0.7, 10.0),
            quote_latch_threshold_bps: BayesianParam::normal("quote_latch_threshold_bps", 2.5, 1.0),

            // Tier 3: Calibration
            kappa: BayesianParam::gamma("kappa", 2000.0, 4.0),
            hawkes_mu: BayesianParam::gamma("hawkes_mu", 0.5, 5.0),
            hawkes_alpha: BayesianParam::beta("hawkes_alpha", 0.3, 10.0),
            hawkes_beta: BayesianParam::gamma("hawkes_beta", 0.1, 1.0),
            kappa_ewma_alpha: BayesianParam::beta("kappa_ewma_alpha", 0.9, 100.0),
            kelly_tracker_decay: BayesianParam::beta("kelly_tracker_decay", 0.99, 100.0),
            regime_sticky_diagonal: BayesianParam::beta("regime_sticky_diagonal", 0.95, 100.0),

            // Tier 4: Microstructure
            kalman_q: BayesianParam::inverse_gamma("kalman_q", 1e-8, 20.0),
            kalman_r: BayesianParam::inverse_gamma("kalman_r", 2.5e-9, 20.0),
            momentum_normalizer_bps: BayesianParam::gamma("momentum_normalizer_bps", 20.0, 4.0),
            depth_spacing_ratio: BayesianParam::log_normal("depth_spacing_ratio", 1.5, 0.2),
            fill_probability_width_bps: BayesianParam::normal("fill_probability_width_bps", 2.0, 0.25),
            microprice_decay: BayesianParam::beta("microprice_decay", 0.999, 1000.0),

            // Metadata
            last_calibration: None,
            total_fills_observed: 0,
        }
    }
}

impl LearnedParameters {
    /// Create parameters optimized for liquid markets (BTC mainnet).
    pub fn for_liquid_market() -> Self {
        let mut params = Self::default();

        // Liquid markets have lower adverse selection
        params.alpha_touch = BayesianParam::beta("alpha_touch", 0.20, 10.0);

        // Higher kappa (more fills per unit spread)
        params.kappa = BayesianParam::gamma("kappa", 3000.0, 5.0);

        // Tighter spread floor
        params.spread_floor_bps = BayesianParam::normal("spread_floor_bps", 4.0, 1.0);

        params
    }

    /// Create parameters optimized for thin DEX markets (HIP-3).
    pub fn for_thin_dex() -> Self {
        let mut params = Self::default();

        // Thin markets have higher adverse selection
        params.alpha_touch = BayesianParam::beta("alpha_touch", 0.30, 6.0);

        // Lower kappa (fewer fills)
        params.kappa = BayesianParam::gamma("kappa", 1000.0, 3.0);

        // Wider spread floor
        params.spread_floor_bps = BayesianParam::normal("spread_floor_bps", 8.0, 4.0);

        // Higher regime stickiness (regimes change slowly)
        params.regime_sticky_diagonal = BayesianParam::beta("regime_sticky_diagonal", 0.98, 100.0);

        params
    }

    /// Reset all parameters to their priors.
    pub fn reset_all(&mut self) {
        self.alpha_touch.reset_to_prior();
        self.gamma_base.reset_to_prior();
        self.spread_floor_bps.reset_to_prior();
        self.proactive_skew_sensitivity.reset_to_prior();
        self.quote_gate_edge_threshold.reset_to_prior();
        self.toxic_hour_gamma_mult.reset_to_prior();
        self.predictive_bias_sensitivity.reset_to_prior();
        self.max_daily_loss_fraction.reset_to_prior();
        self.max_drawdown.reset_to_prior();
        self.cascade_oi_threshold.reset_to_prior();
        self.bocpd_hazard_rate.reset_to_prior();
        self.bocpd_threshold.reset_to_prior();
        self.quote_latch_threshold_bps.reset_to_prior();
        self.kappa.reset_to_prior();
        self.hawkes_mu.reset_to_prior();
        self.hawkes_alpha.reset_to_prior();
        self.hawkes_beta.reset_to_prior();
        self.kappa_ewma_alpha.reset_to_prior();
        self.kelly_tracker_decay.reset_to_prior();
        self.regime_sticky_diagonal.reset_to_prior();
        self.kalman_q.reset_to_prior();
        self.kalman_r.reset_to_prior();
        self.momentum_normalizer_bps.reset_to_prior();
        self.depth_spacing_ratio.reset_to_prior();
        self.fill_probability_width_bps.reset_to_prior();
        self.microprice_decay.reset_to_prior();
        self.last_calibration = None;
        self.total_fills_observed = 0;
    }

    /// Get a summary of all parameter estimates for logging.
    pub fn summary(&self) -> Vec<(&str, f64, f64, usize)> {
        vec![
            ("alpha_touch", self.alpha_touch.estimate(), self.alpha_touch.cv(), self.alpha_touch.n_observations),
            ("gamma_base", self.gamma_base.estimate(), self.gamma_base.cv(), self.gamma_base.n_observations),
            ("spread_floor_bps", self.spread_floor_bps.estimate(), self.spread_floor_bps.cv(), self.spread_floor_bps.n_observations),
            ("kappa", self.kappa.estimate(), self.kappa.cv(), self.kappa.n_observations),
            ("hawkes_mu", self.hawkes_mu.estimate(), self.hawkes_mu.cv(), self.hawkes_mu.n_observations),
            ("hawkes_alpha", self.hawkes_alpha.estimate(), self.hawkes_alpha.cv(), self.hawkes_alpha.n_observations),
        ]
    }

    /// Check overall calibration status.
    pub fn calibration_status(&self) -> CalibrationStatus {
        let tier1_calibrated =
            self.alpha_touch.is_calibrated(50, 0.5) &&
            self.gamma_base.is_calibrated(20, 1.0) &&
            self.kappa.is_calibrated(100, 0.5);

        let tier2_calibrated =
            self.cascade_oi_threshold.is_calibrated(10, 1.0) &&
            self.bocpd_threshold.is_calibrated(10, 0.5);

        CalibrationStatus {
            tier1_ready: tier1_calibrated,
            tier2_ready: tier2_calibrated,
            total_observations: self.total_fills_observed,
            warmup_complete: self.total_fills_observed >= 100,
        }
    }

    // ==================== Persistence ====================

    /// Save learned parameters to a JSON file.
    ///
    /// The file can be loaded later with `load_from_file` to resume calibration
    /// from where it left off, avoiding cold start on restart.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load learned parameters from a JSON file.
    ///
    /// Returns the loaded parameters or an error if the file doesn't exist or is invalid.
    /// Use `load_or_default` for graceful fallback to defaults.
    pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Load learned parameters from file or return default if file doesn't exist.
    ///
    /// This provides graceful cold start behavior:
    /// - If calibration file exists, load and continue from previous state
    /// - If no file exists, start fresh with default priors
    pub fn load_or_default(path: &std::path::Path) -> Self {
        match Self::load_from_file(path) {
            Ok(params) => {
                tracing::info!(
                    path = %path.display(),
                    total_fills = params.total_fills_observed,
                    "Loaded learned parameters from file"
                );
                params
            }
            Err(e) => {
                tracing::info!(
                    path = %path.display(),
                    error = %e,
                    "No existing calibration file, starting with defaults"
                );
                Self::default()
            }
        }
    }

    /// Get the recommended file path for persisting learned parameters.
    ///
    /// Uses the format: `calibration/{asset}_learned_params.json`
    pub fn default_path(asset: &str) -> std::path::PathBuf {
        std::path::PathBuf::from("calibration")
            .join(format!("{}_learned_params.json", asset.to_lowercase()))
    }
}

/// Calibration status summary.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CalibrationStatus {
    /// Tier 1 (P&L critical) parameters are calibrated
    pub tier1_ready: bool,
    /// Tier 2 (risk) parameters are calibrated
    pub tier2_ready: bool,
    /// Total observations across all parameters
    pub total_observations: usize,
    /// Warmup phase complete
    pub warmup_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_prior() {
        let param = BayesianParam::beta("test", 0.25, 8.0);
        assert!((param.estimate() - 0.25).abs() < 0.01);
        assert!(param.n_observations == 0);
    }

    #[test]
    fn test_beta_update() {
        let mut param = BayesianParam::beta("test", 0.25, 8.0);
        param.observe_beta(3, 7); // 30% success rate in data

        // Posterior should be between prior (0.25) and data (0.30)
        let estimate = param.estimate();
        assert!(estimate > 0.25 && estimate < 0.30);
        assert_eq!(param.n_observations, 10);
    }

    #[test]
    fn test_gamma_prior() {
        let param = BayesianParam::gamma("kappa", 2000.0, 10.0);
        assert!((param.estimate() - 2000.0).abs() < 1.0);
    }

    #[test]
    fn test_gamma_poisson_update() {
        let mut param = BayesianParam::gamma("kappa", 2000.0, 10.0);

        // Observe 50 fills over 0.02 time units → rate of 2500
        param.observe_gamma_poisson(50, 0.02);

        // Posterior should be between prior (2000) and data (2500)
        let estimate = param.estimate();
        assert!(estimate > 2000.0 && estimate < 2500.0);
    }

    #[test]
    fn test_normal_prior() {
        let param = BayesianParam::normal("test", 2.0, 0.25);
        assert!((param.estimate() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_shrinkage() {
        let mut param = BayesianParam::beta("test", 0.5, 10.0); // Strong prior at 0.5

        // Observe extreme data: 9/10 successes
        param.observe_beta(9, 1);

        // With strong prior, estimate should shrink toward 0.5
        let estimate = param.estimate();
        assert!(estimate < 0.9); // Not as extreme as raw data
        assert!(estimate > 0.5); // But higher than prior

        // Data weight should be 10/(10+10) = 0.5
        assert!((param.data_weight() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_credible_interval() {
        let param = BayesianParam::beta("test", 0.5, 100.0);
        let (lower, upper) = param.credible_interval_95();

        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
        assert!(lower < 0.5 && upper > 0.5);
    }

    #[test]
    fn test_reset_to_prior() {
        let mut param = BayesianParam::beta("test", 0.25, 8.0);
        param.observe_beta(50, 50);
        assert!(param.n_observations > 0);

        param.reset_to_prior();
        assert_eq!(param.n_observations, 0);
        assert!((param.estimate() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_learned_parameters_default() {
        let params = LearnedParameters::default();

        // Check Tier 1 defaults
        assert!((params.alpha_touch.estimate() - 0.25).abs() < 0.01);
        assert!((params.gamma_base.estimate() - 0.15).abs() < 0.01);
        assert!((params.kappa.estimate() - 2000.0).abs() < 1.0);
    }

    #[test]
    fn test_calibration_status() {
        let params = LearnedParameters::default();
        let status = params.calibration_status();

        // Should not be calibrated without observations
        assert!(!status.tier1_ready);
        assert!(!status.warmup_complete);
    }

    #[test]
    fn test_save_load_roundtrip() {
        use std::fs;
        
        // Create params and add some observations
        let mut params = LearnedParameters::default();
        params.alpha_touch.observe_beta(30, 70); // 30% informed
        params.kappa.observe_gamma_poisson(100, 0.05); // 2000 fills/sec/spread
        params.total_fills_observed = 100;
        
        // Save to temp file
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_learned_params.json");
        
        params.save_to_file(&temp_path).expect("Failed to save");
        
        // Load and verify
        let loaded = LearnedParameters::load_from_file(&temp_path).expect("Failed to load");
        
        // Verify key values persisted
        assert_eq!(loaded.alpha_touch.n_observations, params.alpha_touch.n_observations);
        assert!((loaded.alpha_touch.estimate() - params.alpha_touch.estimate()).abs() < 0.001);
        assert_eq!(loaded.kappa.n_observations, params.kappa.n_observations);
        assert!((loaded.kappa.estimate() - params.kappa.estimate()).abs() < 1.0);
        assert_eq!(loaded.total_fills_observed, 100);
        
        // Clean up
        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_load_or_default_missing_file() {
        let nonexistent_path = std::path::PathBuf::from("/nonexistent/path/params.json");
        let params = LearnedParameters::load_or_default(&nonexistent_path);
        
        // Should return defaults
        assert!((params.alpha_touch.estimate() - 0.25).abs() < 0.01);
        assert_eq!(params.total_fills_observed, 0);
    }

    #[test]
    fn test_default_path() {
        let path = LearnedParameters::default_path("HYPE");
        assert!(path.to_str().unwrap().contains("hype_learned_params.json"));
    }
}
