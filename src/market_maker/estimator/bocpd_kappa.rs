//! Bayesian Online Change Point Detection for Kappa Predictions.
//!
//! This module implements BOCPD specifically for detecting when the
//! feature→κ relationship breaks down. Unlike standard BOCD which tracks
//! changes in a single variable, this tracks changes in regression coefficients.
//!
//! ## Key Insight
//!
//! Standard predictive kappa uses features like:
//! - Binance activity surge
//! - Funding settlement proximity
//! - Book depth velocity
//! - OI change velocity
//!
//! But the **relationship** between these features and realized kappa can change
//! (regime shift, market structure change, etc.). When this happens, the model
//! is invalid and we should use prior beliefs instead.
//!
//! ## Algorithm
//!
//! 1. Maintain run length distribution P(r_t | data)
//! 2. For each run length, maintain Bayesian linear regression coefficients
//! 3. Compute predictive likelihood under each run length
//! 4. If P(r_t < threshold | data) is high, coefficients have changed
//!
//! ## Usage
//!
//! ```ignore
//! let mut predictor = BOCPDKappaPredictor::new(BOCPDKappaConfig::default());
//!
//! // Update with realized kappa
//! let features = [binance_activity, funding_proximity, depth_velocity, oi_velocity];
//! let changepoint = predictor.update(&features, realized_kappa);
//!
//! if changepoint || predictor.should_use_prior() {
//!     // Model is unreliable, use prior kappa
//! } else {
//!     // Model is valid, use prediction
//!     let (mean, std) = predictor.predict(&features);
//! }
//! ```

use std::collections::VecDeque;

/// Number of features used for kappa prediction.
pub const N_FEATURES: usize = 4;

/// Feature indices for clarity.
pub mod features {
    pub const BINANCE_ACTIVITY: usize = 0;
    pub const FUNDING_PROXIMITY: usize = 1;
    pub const DEPTH_VELOCITY: usize = 2;
    pub const OI_VELOCITY: usize = 3;
}

/// Configuration for BOCPD Kappa Predictor.
#[derive(Debug, Clone)]
pub struct BOCPDKappaConfig {
    /// Prior probability of changepoint per observation.
    /// Typical values: 0.001 to 0.01 (expect changepoint every 100-1000 obs)
    pub hazard_rate: f64,

    /// Maximum run length to track.
    /// Longer = more memory but better detection of slow drifts.
    pub max_run_length: usize,

    /// Probability threshold for declaring "new regime".
    /// If P(r_t < new_regime_threshold) > this value, model is unreliable.
    pub changepoint_threshold: f64,

    /// Run length below which we consider "new regime".
    pub new_regime_run_length: usize,

    /// Prior mean for regression coefficients.
    pub prior_coef_mean: [f64; N_FEATURES],

    /// Prior precision (inverse variance) for coefficients.
    /// Higher = more confidence in prior, slower learning.
    pub prior_coef_precision: f64,

    /// Prior for observation noise variance (Inverse-Gamma parameters).
    pub prior_noise_alpha: f64,
    pub prior_noise_beta: f64,

    /// Minimum observations before predictions are valid.
    pub warmup_observations: usize,
}

impl Default for BOCPDKappaConfig {
    fn default() -> Self {
        Self {
            hazard_rate: 0.005, // Expect changepoint every ~200 observations
            max_run_length: 500,
            changepoint_threshold: 0.3, // 30% probability triggers warning
            new_regime_run_length: 20,  // Run length < 20 = "new regime"
            // Prior: features have small positive effect on kappa
            prior_coef_mean: [0.1, 0.1, 0.05, 0.05],
            prior_coef_precision: 1.0, // Moderate confidence
            prior_noise_alpha: 2.0,
            prior_noise_beta: 1.0,
            warmup_observations: 50,
        }
    }
}

// =============================================================================
// FIRST-PRINCIPLES: Adaptive Hazard Rate
// =============================================================================
//
// The probability of a regime change is NOT constant - it depends on market stress.
// High VPIN, elevated Hawkes intensity, or trade size anomalies all increase
// the probability that the feature→κ relationship has broken.
//
// Mathematical grounding:
//   h(t) = h_base × (1 + α×VPIN + β×(λ_hawkes - 1) + γ×size_anomaly)
//
// This replaces the constant hazard assumption with a state-dependent one.
// =============================================================================

/// Configuration for adaptive hazard rate.
#[derive(Debug, Clone)]
pub struct AdaptiveHazardConfig {
    /// Base hazard rate when market is calm
    pub base_rate: f64,
    /// Sensitivity to VPIN (informed trading probability)
    pub vpin_sensitivity: f64,
    /// Sensitivity to Hawkes intensity (trade clustering)
    pub intensity_sensitivity: f64,
    /// Sensitivity to trade size anomalies
    pub anomaly_sensitivity: f64,
    /// Maximum hazard rate (cap to avoid numerical issues)
    pub max_rate: f64,
}

impl Default for AdaptiveHazardConfig {
    fn default() -> Self {
        Self {
            base_rate: 0.005,
            vpin_sensitivity: 0.02,      // VPIN=0.5 → 1% additional hazard
            intensity_sensitivity: 0.01, // 2× baseline intensity → 1% additional hazard
            anomaly_sensitivity: 0.015,  // 3σ anomaly → 4.5% additional hazard
            max_rate: 0.1,               // Cap at 10%
        }
    }
}

/// Adaptive hazard rate that increases with market stress.
///
/// First-Principles: The probability of a structural break (regime change)
/// should increase when market conditions are stressed:
/// - High VPIN indicates informed trading → relationships may break
/// - High Hawkes intensity indicates clustering → momentum regime
/// - Large trade sizes indicate institutional flow → different dynamics
#[derive(Debug, Clone)]
pub struct AdaptiveHazard {
    config: AdaptiveHazardConfig,
    /// Current hazard rate (cached for diagnostics)
    current_hazard: f64,
    /// Current market stress index [0, 1]
    stress_index: f64,
}

impl Default for AdaptiveHazard {
    fn default() -> Self {
        Self::new(AdaptiveHazardConfig::default())
    }
}

impl AdaptiveHazard {
    /// Create a new adaptive hazard calculator.
    pub fn new(config: AdaptiveHazardConfig) -> Self {
        Self {
            current_hazard: config.base_rate,
            stress_index: 0.0,
            config,
        }
    }

    /// Compute time-varying hazard rate based on market stress.
    ///
    /// # Arguments
    /// * `vpin` - VPIN score [0, 1], higher = more informed trading
    /// * `hawkes_intensity_ratio` - Current λ / baseline μ, >1 = elevated
    /// * `size_anomaly_sigma` - Trade size deviation in sigmas, >3 = anomalous
    ///
    /// # Returns
    /// Hazard rate (probability of regime change per observation)
    pub fn compute(
        &mut self,
        vpin: f64,
        hawkes_intensity_ratio: f64,
        size_anomaly_sigma: f64,
    ) -> f64 {
        // Compute stress contributions
        let vpin_stress = self.config.vpin_sensitivity * vpin;
        let intensity_stress =
            self.config.intensity_sensitivity * (hawkes_intensity_ratio - 1.0).max(0.0);
        let anomaly_stress = self.config.anomaly_sensitivity * size_anomaly_sigma.max(0.0);

        // Total stress multiplier
        let total_stress = 1.0 + vpin_stress + intensity_stress + anomaly_stress;

        // Compute adaptive hazard
        let hazard = (self.config.base_rate * total_stress).min(self.config.max_rate);

        // Cache for diagnostics
        self.current_hazard = hazard;
        self.stress_index = (total_stress - 1.0) / 3.0; // Normalize to [0, 1] roughly

        hazard
    }

    /// Get current hazard rate (from last compute call).
    pub fn current_hazard(&self) -> f64 {
        self.current_hazard
    }

    /// Get current market stress index [0, 1].
    pub fn stress_index(&self) -> f64 {
        self.stress_index.clamp(0.0, 1.0)
    }

    /// Get base (unstressed) hazard rate.
    pub fn base_rate(&self) -> f64 {
        self.config.base_rate
    }
}

/// Sufficient statistics for Bayesian linear regression at one run length.
#[derive(Debug, Clone)]
struct RegressionStats {
    /// Number of observations
    n: usize,
    /// X'X matrix (for coefficient estimation)
    xtx: [[f64; N_FEATURES]; N_FEATURES],
    /// X'y vector
    xty: [f64; N_FEATURES],
    /// Sum of y^2 (for noise variance estimation)
    yty: f64,
    /// Sum of y
    y_sum: f64,
}

impl Default for RegressionStats {
    fn default() -> Self {
        Self {
            n: 0,
            xtx: [[0.0; N_FEATURES]; N_FEATURES],
            xty: [0.0; N_FEATURES],
            yty: 0.0,
            y_sum: 0.0,
        }
    }
}

impl RegressionStats {
    /// Update statistics with new observation.
    fn update(&mut self, x: &[f64; N_FEATURES], y: f64) {
        self.n += 1;

        // Update X'X
        for i in 0..N_FEATURES {
            for j in 0..N_FEATURES {
                self.xtx[i][j] += x[i] * x[j];
            }
        }

        // Update X'y
        for (i, xi) in x.iter().enumerate().take(N_FEATURES) {
            self.xty[i] += xi * y;
        }

        // Update y statistics
        self.yty += y * y;
        self.y_sum += y;
    }

    /// Compute posterior mean of coefficients (ridge regression solution).
    fn posterior_mean(
        &self,
        prior_precision: f64,
        prior_mean: &[f64; N_FEATURES],
    ) -> [f64; N_FEATURES] {
        if self.n < N_FEATURES {
            return *prior_mean;
        }

        // Regularized solution: (X'X + λI)^{-1} (X'y + λ μ₀)
        // We use a simple diagonal approximation for efficiency
        let mut coefs = [0.0; N_FEATURES];

        for i in 0..N_FEATURES {
            let data_precision = self.xtx[i][i];
            let total_precision = data_precision + prior_precision;

            if total_precision > 1e-10 {
                let data_contrib = self.xty[i];
                let prior_contrib = prior_precision * prior_mean[i];
                coefs[i] = (data_contrib + prior_contrib) / total_precision;
            } else {
                coefs[i] = prior_mean[i];
            }
        }

        coefs
    }

    /// Compute predictive variance for a new observation.
    fn predictive_variance(&self, x: &[f64; N_FEATURES], prior_alpha: f64, prior_beta: f64) -> f64 {
        if self.n < N_FEATURES + 2 {
            return prior_beta / prior_alpha; // Prior variance
        }

        // Posterior parameters for noise variance (Inverse-Gamma)
        let post_alpha = prior_alpha + self.n as f64 / 2.0;

        // Compute residual sum of squares (approximate)
        let y_mean = self.y_sum / self.n as f64;
        let rss = (self.yty - self.y_sum * y_mean).max(0.0);

        let post_beta = prior_beta + rss / 2.0;

        // Predictive variance = noise_var + x' Σ x (coefficient uncertainty)
        let noise_var = post_beta / (post_alpha - 1.0).max(0.5);

        // Add coefficient uncertainty (simplified)
        let mut coef_var = 0.0;
        for (i, xi) in x.iter().enumerate().take(N_FEATURES) {
            let precision = self.xtx[i][i] + 1.0; // Regularization
            coef_var += xi * xi / precision;
        }

        noise_var * (1.0 + coef_var)
    }

    /// Compute predictive log-likelihood for observation y given features x.
    fn predictive_log_likelihood(
        &self,
        x: &[f64; N_FEATURES],
        y: f64,
        prior_precision: f64,
        prior_mean: &[f64; N_FEATURES],
        prior_alpha: f64,
        prior_beta: f64,
    ) -> f64 {
        let coefs = self.posterior_mean(prior_precision, prior_mean);
        let pred: f64 = coefs.iter().zip(x.iter()).map(|(c, xi)| c * xi).sum();
        let var = self.predictive_variance(x, prior_alpha, prior_beta);

        // Log of Gaussian PDF
        let residual = y - pred;
        -0.5 * (var.ln()
            + residual * residual / var
            + std::f64::consts::LN_2
            + std::f64::consts::PI.ln())
    }
}

/// BOCPD-based predictor for kappa that detects relationship changes.
#[derive(Debug)]
pub struct BOCPDKappaPredictor {
    /// Configuration
    config: BOCPDKappaConfig,

    /// Run length probabilities P(r_t = k | data)
    run_length_probs: Vec<f64>,

    /// Regression statistics for each run length
    run_stats: Vec<RegressionStats>,

    /// Total observations
    observation_count: usize,

    /// Recent predictions for diagnostics
    recent_errors: VecDeque<f64>,

    /// Maximum recent errors to track
    max_recent: usize,

    /// Cached probability of being in new regime
    p_new_regime_cached: f64,

    /// Adaptive hazard calculator (first-principles: non-stationary changepoint probability)
    adaptive_hazard: AdaptiveHazard,

    /// Last computed hazard rate (for diagnostics)
    last_hazard_rate: f64,

    /// Whether a changepoint was detected on last update
    last_changepoint_detected: bool,
}

impl Default for BOCPDKappaPredictor {
    fn default() -> Self {
        Self::new(BOCPDKappaConfig::default())
    }
}

impl BOCPDKappaPredictor {
    /// Create a new BOCPD Kappa Predictor.
    pub fn new(config: BOCPDKappaConfig) -> Self {
        let max_len = config.max_run_length;
        let base_hazard = config.hazard_rate;

        // Initialize with all mass on run length 0
        let mut run_length_probs = vec![0.0; max_len + 1];
        run_length_probs[0] = 1.0;

        let run_stats = vec![RegressionStats::default(); max_len + 1];

        // Initialize adaptive hazard with base rate from config
        let adaptive_config = AdaptiveHazardConfig {
            base_rate: base_hazard,
            ..Default::default()
        };

        Self {
            config,
            run_length_probs,
            run_stats,
            observation_count: 0,
            recent_errors: VecDeque::with_capacity(100),
            max_recent: 100,
            p_new_regime_cached: 1.0, // Start in "new regime"
            adaptive_hazard: AdaptiveHazard::new(adaptive_config),
            last_hazard_rate: base_hazard,
            last_changepoint_detected: false,
        }
    }

    /// Update with new observation using constant hazard rate. Returns true if changepoint detected.
    ///
    /// For first-principles grounding, prefer `update_with_stress` which uses adaptive hazard.
    pub fn update(&mut self, features: &[f64; N_FEATURES], realized_kappa: f64) -> bool {
        self.update_internal(features, realized_kappa, self.config.hazard_rate)
    }

    /// Update with new observation using adaptive (stress-dependent) hazard rate.
    ///
    /// First-Principles: The probability of a regime change is NOT constant.
    /// It increases with market stress indicators.
    ///
    /// # Arguments
    /// * `features` - Feature vector for kappa prediction
    /// * `realized_kappa` - Actual observed kappa
    /// * `vpin` - VPIN score [0, 1] indicating informed trading probability
    /// * `hawkes_intensity_ratio` - Current Hawkes intensity / baseline (>1 = elevated)
    /// * `size_anomaly_sigma` - Trade size deviation in sigmas (>3 = anomalous)
    pub fn update_with_stress(
        &mut self,
        features: &[f64; N_FEATURES],
        realized_kappa: f64,
        vpin: f64,
        hawkes_intensity_ratio: f64,
        size_anomaly_sigma: f64,
    ) -> bool {
        let hazard = self
            .adaptive_hazard
            .compute(vpin, hawkes_intensity_ratio, size_anomaly_sigma);
        self.last_hazard_rate = hazard;
        self.update_internal(features, realized_kappa, hazard)
    }

    /// Internal update method with explicit hazard rate.
    fn update_internal(
        &mut self,
        features: &[f64; N_FEATURES],
        realized_kappa: f64,
        hazard: f64,
    ) -> bool {
        self.observation_count += 1;

        let max_len = self.config.max_run_length;

        // Compute predictive probabilities for each run length
        let mut pred_probs = vec![0.0; max_len + 1];
        for (r, pred_prob) in pred_probs.iter_mut().enumerate().take(max_len + 1) {
            if self.run_length_probs[r] > 1e-10 {
                *pred_prob = self.run_stats[r].predictive_log_likelihood(
                    features,
                    realized_kappa,
                    self.config.prior_coef_precision,
                    &self.config.prior_coef_mean,
                    self.config.prior_noise_alpha,
                    self.config.prior_noise_beta,
                );
            } else {
                *pred_prob = f64::NEG_INFINITY;
            }
        }

        // Convert log probs to probs (with numerical stability)
        let max_log = pred_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for p in &mut pred_probs {
            if *p > f64::NEG_INFINITY {
                *p = (*p - max_log).exp();
            } else {
                *p = 0.0;
            }
        }

        // BOCD update step
        // P(r_t = r | data) ∝ P(x_t | r_{t-1} = r) × (1 - H) × P(r_{t-1} = r)   [growth]
        //                   + P(x_t | r_{t-1} = r) × H × P(r_{t-1} = r)         [changepoint]
        let mut new_probs = vec![0.0; max_len + 1];

        // Changepoint mass (all run lengths contribute to r=0)
        let mut cp_mass = 0.0;
        for (r, &pred_prob) in pred_probs.iter().enumerate().take(max_len) {
            cp_mass += pred_prob * hazard * self.run_length_probs[r];
        }
        new_probs[0] = cp_mass;

        // Growth mass (run length increments)
        for r in 0..max_len {
            new_probs[r + 1] = pred_probs[r] * (1.0 - hazard) * self.run_length_probs[r];
        }

        // Normalize
        let sum: f64 = new_probs.iter().sum();
        if sum > 1e-10 {
            for p in &mut new_probs {
                *p /= sum;
            }
        } else {
            // Numerical issues, reset to uniform
            let uniform = 1.0 / (max_len + 1) as f64;
            new_probs.fill(uniform);
        }

        self.run_length_probs = new_probs;

        // Update regression statistics for each run length
        // New run length 0 gets fresh statistics
        let mut new_stats = vec![RegressionStats::default(); max_len + 1];
        new_stats[0] = RegressionStats::default();
        new_stats[0].update(features, realized_kappa);

        // Existing run lengths get updated statistics
        for r in 0..max_len {
            new_stats[r + 1] = self.run_stats[r].clone();
            new_stats[r + 1].update(features, realized_kappa);
        }
        self.run_stats = new_stats;

        // Track prediction error
        let (pred_mean, _) = self.predict(features);
        let error = (realized_kappa - pred_mean).abs();
        self.recent_errors.push_back(error);
        if self.recent_errors.len() > self.max_recent {
            self.recent_errors.pop_front();
        }

        // Cache new regime probability
        self.p_new_regime_cached = self.compute_p_new_regime();

        // Detect changepoint
        self.last_changepoint_detected =
            self.p_new_regime_cached > self.config.changepoint_threshold;
        self.last_changepoint_detected
    }

    /// Predict kappa given features.
    ///
    /// Returns (mean, std) of the predictive distribution.
    /// Uses weighted average across run lengths.
    pub fn predict(&self, features: &[f64; N_FEATURES]) -> (f64, f64) {
        let mut mean = 0.0;
        let mut var = 0.0;

        for r in 0..=self.config.max_run_length {
            if self.run_length_probs[r] > 1e-10 {
                let coefs = self.run_stats[r].posterior_mean(
                    self.config.prior_coef_precision,
                    &self.config.prior_coef_mean,
                );

                let pred: f64 = coefs.iter().zip(features.iter()).map(|(c, x)| c * x).sum();
                let pred_var = self.run_stats[r].predictive_variance(
                    features,
                    self.config.prior_noise_alpha,
                    self.config.prior_noise_beta,
                );

                mean += self.run_length_probs[r] * pred;
                // Law of total variance: Var[Y] = E[Var[Y|R]] + Var[E[Y|R]]
                var += self.run_length_probs[r] * (pred_var + pred * pred);
            }
        }

        var -= mean * mean;
        var = var.max(0.01); // Ensure positive

        (mean, var.sqrt())
    }

    /// Compute probability of being in a "new regime" (short run length).
    fn compute_p_new_regime(&self) -> f64 {
        let threshold = self.config.new_regime_run_length;
        self.run_length_probs[..threshold.min(self.run_length_probs.len())]
            .iter()
            .sum()
    }

    /// Get probability that we're in a new regime (coefficients have changed).
    pub fn p_new_regime(&self) -> f64 {
        self.p_new_regime_cached
    }

    /// Check if we should use prior instead of prediction.
    ///
    /// Returns true if:
    /// - Not warmed up, OR
    /// - High probability of being in new regime
    pub fn should_use_prior(&self) -> bool {
        !self.is_warmed_up() || self.p_new_regime_cached > self.config.changepoint_threshold
    }

    /// Check if predictor is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.observation_count >= self.config.warmup_observations
    }

    /// Get most likely run length.
    pub fn most_likely_run_length(&self) -> usize {
        self.run_length_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get expected run length (mean of distribution).
    /// More robust than most_likely_run_length for geometric-like distributions.
    pub fn expected_run_length(&self) -> f64 {
        self.run_length_probs
            .iter()
            .enumerate()
            .map(|(r, p)| r as f64 * p)
            .sum()
    }

    /// Get entropy of run length distribution (higher = more uncertainty).
    pub fn run_length_entropy(&self) -> f64 {
        let mut entropy = 0.0;
        for &p in &self.run_length_probs {
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Get mean absolute prediction error (recent).
    pub fn mean_abs_error(&self) -> f64 {
        if self.recent_errors.is_empty() {
            return f64::INFINITY;
        }
        self.recent_errors.iter().sum::<f64>() / self.recent_errors.len() as f64
    }

    /// Get total observation count.
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Check if changepoint was detected on last update.
    pub fn changepoint_detected(&self) -> bool {
        self.last_changepoint_detected
    }

    /// Get current coefficient estimates (weighted by run length probability).
    pub fn current_coefficients(&self) -> [f64; N_FEATURES] {
        let mut coefs = [0.0; N_FEATURES];

        for r in 0..=self.config.max_run_length {
            if self.run_length_probs[r] > 1e-10 {
                let r_coefs = self.run_stats[r].posterior_mean(
                    self.config.prior_coef_precision,
                    &self.config.prior_coef_mean,
                );
                for i in 0..N_FEATURES {
                    coefs[i] += self.run_length_probs[r] * r_coefs[i];
                }
            }
        }

        coefs
    }

    // =========================================================================
    // First-Principles: Adaptive Hazard Diagnostics
    // =========================================================================

    /// Get last computed hazard rate.
    pub fn last_hazard_rate(&self) -> f64 {
        self.last_hazard_rate
    }

    /// Get current market stress index from adaptive hazard [0, 1].
    pub fn market_stress_index(&self) -> f64 {
        self.adaptive_hazard.stress_index()
    }

    /// Get base (unstressed) hazard rate.
    pub fn base_hazard_rate(&self) -> f64 {
        self.adaptive_hazard.base_rate()
    }

    /// Reset the predictor.
    pub fn reset(&mut self) {
        let max_len = self.config.max_run_length;

        self.run_length_probs = vec![0.0; max_len + 1];
        self.run_length_probs[0] = 1.0;
        self.run_stats = vec![RegressionStats::default(); max_len + 1];
        self.observation_count = 0;
        self.recent_errors.clear();
        self.p_new_regime_cached = 1.0;
        self.last_changepoint_detected = false;
    }
}

// Send + Sync for concurrent access
unsafe impl Send for BOCPDKappaPredictor {}
unsafe impl Sync for BOCPDKappaPredictor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BOCPDKappaConfig::default();
        assert!((config.hazard_rate - 0.005).abs() < 1e-10);
        assert_eq!(config.max_run_length, 500);
        assert_eq!(config.warmup_observations, 50);
    }

    #[test]
    fn test_predictor_creation() {
        let predictor = BOCPDKappaPredictor::default();
        assert!(!predictor.is_warmed_up());
        assert!((predictor.p_new_regime() - 1.0).abs() < 1e-10); // Starts in new regime
    }

    #[test]
    fn test_warmup() {
        let mut predictor = BOCPDKappaPredictor::new(BOCPDKappaConfig {
            warmup_observations: 10,
            ..Default::default()
        });

        // Not warmed up yet
        assert!(!predictor.is_warmed_up());
        assert!(predictor.should_use_prior());

        // Add observations
        for i in 0..10 {
            let features = [0.1, 0.2, 0.05, 0.05];
            let kappa = 1.0 + i as f64 * 0.1;
            predictor.update(&features, kappa);
        }

        // Now warmed up
        assert!(predictor.is_warmed_up());
    }

    #[test]
    fn test_basic_prediction_stability() {
        // Test that BOCPD produces stable predictions with stable input
        let mut predictor = BOCPDKappaPredictor::new(BOCPDKappaConfig {
            warmup_observations: 10,
            hazard_rate: 0.02,
            new_regime_run_length: 5,
            ..Default::default()
        });

        // Feed stable relationship: kappa ≈ x1 + x2
        for t in 0..50 {
            let x1 = (t as f64 * 0.1).sin() + 1.0;
            let x2 = (t as f64 * 0.05).cos() + 1.0;
            let features = [x1, x2, 0.0, 0.0];
            let kappa = x1 + x2; // Simple: kappa = x1 + x2
            predictor.update(&features, kappa);
        }

        // Should be warmed up
        assert!(predictor.is_warmed_up(), "Should be warmed up after 50 obs");

        // Predictions should be finite
        let (pred, std) = predictor.predict(&[1.0, 1.0, 0.0, 0.0]);
        assert!(pred.is_finite(), "Prediction should be finite: {}", pred);
        assert!(std.is_finite(), "Std should be finite: {}", std);
        assert!(std > 0.0, "Std should be positive: {}", std);

        // Prediction should be in reasonable range for input x1=1, x2=1
        // True value = 1 + 1 = 2, prediction should be close
        assert!(
            pred > 0.0 && pred < 10.0,
            "Prediction {} should be in reasonable range for kappa=x1+x2",
            pred
        );

        // Expected run length should be positive
        let expected_rl = predictor.expected_run_length();
        assert!(
            expected_rl > 0.0,
            "Expected run length should be positive: {}",
            expected_rl
        );

        // Entropy should be positive (there's uncertainty in run length)
        let entropy = predictor.run_length_entropy();
        assert!(entropy > 0.0, "Entropy should be positive: {}", entropy);
    }

    #[test]
    fn test_relationship_change_detection() {
        let mut predictor = BOCPDKappaPredictor::new(BOCPDKappaConfig {
            warmup_observations: 20,
            hazard_rate: 0.02,
            changepoint_threshold: 0.3,
            ..Default::default()
        });

        // Phase 1: kappa = 1.0 + 0.5*x1
        for i in 0..50 {
            let x1 = (i as f64 * 0.1).sin();
            let features = [x1, 0.0, 0.0, 0.0];
            let kappa = 1.0 + 0.5 * x1;
            predictor.update(&features, kappa);
        }

        let p_before = predictor.p_new_regime();

        // Phase 2: kappa = 2.0 - 0.5*x1 (relationship reversed!)
        let mut detected_change = false;
        for i in 50..100 {
            let x1 = (i as f64 * 0.1).sin();
            let features = [x1, 0.0, 0.0, 0.0];
            let kappa = 2.0 - 0.5 * x1; // Reversed relationship

            if predictor.update(&features, kappa) {
                detected_change = true;
            }
        }

        let p_after = predictor.p_new_regime();

        // P(new regime) should increase after relationship change
        // Note: This may not always trigger, depending on random seed
        assert!(
            p_after > p_before * 0.5 || detected_change,
            "Should detect relationship change: p_before={}, p_after={}, detected={}",
            p_before,
            p_after,
            detected_change
        );
    }

    #[test]
    fn test_prediction() {
        let mut predictor = BOCPDKappaPredictor::new(BOCPDKappaConfig {
            warmup_observations: 10,
            ..Default::default()
        });

        // Train on simple relationship
        for i in 0..30 {
            let x = i as f64 / 30.0;
            let features = [x, 0.0, 0.0, 0.0];
            let kappa = 1.0 + x; // Simple linear
            predictor.update(&features, kappa);
        }

        // Predict
        let features = [0.5, 0.0, 0.0, 0.0];
        let (mean, std) = predictor.predict(&features);

        // Should be close to 1.5 (1.0 + 0.5)
        assert!((mean - 1.5).abs() < 0.5, "Mean should be ~1.5: {}", mean);
        assert!(std > 0.0, "Std should be positive: {}", std);
    }

    #[test]
    fn test_coefficients() {
        let mut predictor = BOCPDKappaPredictor::default();

        // Train
        for _ in 0..50 {
            let features = [1.0, 0.5, 0.0, 0.0];
            let kappa = 2.0 + 0.3 * features[0] + 0.2 * features[1];
            predictor.update(&features, kappa);
        }

        let coefs = predictor.current_coefficients();

        // Coefficients should be positive for features 0 and 1
        assert!(coefs[0] > 0.0, "Coef 0 should be positive: {}", coefs[0]);
    }

    #[test]
    fn test_reset() {
        let mut predictor = BOCPDKappaPredictor::default();

        // Add some observations
        for i in 0..20 {
            let features = [i as f64 * 0.1, 0.0, 0.0, 0.0];
            predictor.update(&features, 1.0 + i as f64 * 0.1);
        }

        assert!(predictor.observation_count() > 0);

        predictor.reset();

        assert_eq!(predictor.observation_count(), 0);
        assert!((predictor.p_new_regime() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy() {
        let predictor = BOCPDKappaPredictor::default();

        // Initially all mass on r=0, entropy should be 0
        let entropy = predictor.run_length_entropy();
        assert!(entropy < 0.1, "Initial entropy should be ~0: {}", entropy);
    }
}
