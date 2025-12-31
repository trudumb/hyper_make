//! Bayesian fill probability model for ladder quoting.
//!
//! Provides empirically-calibrated fill probability estimates using:
//! - **First-passage theory**: P(fill|δ,τ) = 2×Φ(-δ/(σ×√τ))
//! - **Bayesian posterior**: Beta-Binomial conjugate updates from observed fills
//! - **Depth bucketing**: Groups observations by depth for stable estimation
//!
//! # Mathematical Foundation
//!
//! The theoretical fill probability derives from first-passage time of
//! Brownian motion. For a diffusion process S(t) = S₀ × exp(σW_t - σ²t/2),
//! the probability of price reaching depth δ within time τ is:
//!
//! ```text
//! P(fill | δ, τ) = 2 × Φ(-δ / (σ × √τ))
//! ```
//!
//! Where Φ is the standard normal CDF.
//!
//! The Bayesian model uses this as a prior and updates based on observed
//! fill rates at each depth bucket.
//!
//! # Usage
//!
//! ```rust,ignore
//! let model = BayesianFillModel::new(2.0, 2.0, 0.0001, 10.0);
//!
//! // Update with observations
//! model.record_fill(5.0, true);  // Filled at 5bp depth
//! model.record_fill(5.0, false); // Attempt at 5bp, no fill
//!
//! // Get posterior probability
//! let p_fill = model.fill_probability(5.0);
//! ```

use std::collections::HashMap;
use tracing::debug;

use crate::EPSILON;

/// Standard normal CDF approximation (Abramowitz & Stegun)
fn std_normal_cdf(x: f64) -> f64 {
    // Rational approximation for standard normal CDF
    // Accurate to ~10^-7
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();

    let t = 1.0 / (1.0 + P * x_abs);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x_abs * x_abs / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

/// First-passage fill probability model
///
/// Computes theoretical fill probability from price diffusion:
/// P(fill | δ, τ) = 2 × Φ(-δ / (σ × √τ))
#[derive(Debug, Clone)]
pub struct FirstPassageFillModel {
    /// Volatility (per-second)
    sigma: f64,
    /// Time horizon (seconds)
    tau: f64,
}

impl FirstPassageFillModel {
    /// Create a new first-passage model
    pub fn new(sigma: f64, tau: f64) -> Self {
        Self { sigma, tau }
    }

    /// Update model parameters
    pub fn update(&mut self, sigma: f64, tau: f64) {
        self.sigma = sigma;
        self.tau = tau;
    }

    /// Compute fill probability at given depth
    ///
    /// Uses first-passage formula: P(fill|δ,τ) = 2×Φ(-δ/(σ×√τ))
    ///
    /// # Arguments
    /// * `depth_bps` - Depth from mid in basis points
    ///
    /// # Returns
    /// Fill probability in [0, 1]
    pub fn probability(&self, depth_bps: f64) -> f64 {
        if depth_bps < EPSILON {
            return 1.0; // At mid, always fills
        }

        let depth_frac = depth_bps / 10000.0;
        let tau_safe = self.tau.max(0.001); // Avoid division by zero
        let sigma_safe = self.sigma.max(EPSILON);

        // z = -δ / (σ × √τ)
        let z = -depth_frac / (sigma_safe * tau_safe.sqrt());

        // P(fill) = 2 × Φ(z)
        let p = 2.0 * std_normal_cdf(z);

        // Clamp to valid probability range
        p.clamp(0.0, 1.0)
    }

    /// Compute fill probability with explicit sigma/tau
    pub fn probability_with_params(&self, depth_bps: f64, sigma: f64, tau: f64) -> f64 {
        if depth_bps < EPSILON {
            return 1.0;
        }

        let depth_frac = depth_bps / 10000.0;
        let tau_safe = tau.max(0.001);
        let sigma_safe = sigma.max(EPSILON);

        let z = -depth_frac / (sigma_safe * tau_safe.sqrt());
        (2.0 * std_normal_cdf(z)).clamp(0.0, 1.0)
    }
}

impl Default for FirstPassageFillModel {
    fn default() -> Self {
        Self {
            sigma: 0.0001,  // 1bp/sec default
            tau: 10.0,      // 10 second horizon
        }
    }
}

/// Depth bucket for grouping observations
///
/// Buckets are 2bp wide: [0-2), [2-4), [4-6), etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthBucket(u32);

impl DepthBucket {
    /// Bucket width in basis points
    const WIDTH_BPS: f64 = 2.0;

    /// Create bucket from depth in basis points
    pub fn from_bps(depth_bps: f64) -> Self {
        let bucket_idx = (depth_bps / Self::WIDTH_BPS).floor() as u32;
        Self(bucket_idx)
    }

    /// Get the center depth of this bucket in bps
    pub fn center_bps(&self) -> f64 {
        (self.0 as f64 + 0.5) * Self::WIDTH_BPS
    }

    /// Get the lower bound of this bucket in bps
    pub fn lower_bps(&self) -> f64 {
        self.0 as f64 * Self::WIDTH_BPS
    }

    /// Get the upper bound of this bucket in bps
    pub fn upper_bps(&self) -> f64 {
        (self.0 as f64 + 1.0) * Self::WIDTH_BPS
    }
}

/// Observation counts for a depth bucket
#[derive(Debug, Clone, Copy, Default)]
struct BucketObservations {
    /// Number of fills at this depth
    fills: u64,
    /// Total attempts (fills + non-fills)
    attempts: u64,
}

impl BucketObservations {
    /// Record a new observation
    fn record(&mut self, filled: bool) {
        self.attempts += 1;
        if filled {
            self.fills += 1;
        }
    }
}

/// Bayesian fill probability model with Beta-Binomial conjugate updates
///
/// Maintains a prior belief about fill probability and updates it
/// based on observed fills at each depth bucket. Uses first-passage
/// theory as the prior for unobserved depths.
#[derive(Debug, Clone)]
pub struct BayesianFillModel {
    /// Prior parameters (α₀, β₀) for Beta distribution
    prior_alpha: f64,
    prior_beta: f64,

    /// Observations by depth bucket
    observations: HashMap<DepthBucket, BucketObservations>,

    /// First-passage theoretical model for fallback
    theoretical: FirstPassageFillModel,

    /// Minimum observations before trusting empirical data
    min_observations: u64,

    /// Maximum age of observations in seconds (for decay) - reserved for future use
    #[allow(dead_code)]
    max_age_secs: f64,
}

impl BayesianFillModel {
    /// Create a new Bayesian fill model
    ///
    /// # Arguments
    /// * `prior_alpha` - Beta prior α parameter (default 2.0 = mild prior)
    /// * `prior_beta` - Beta prior β parameter (default 2.0 = mild prior)
    /// * `sigma` - Initial volatility estimate for theoretical model
    /// * `tau` - Time horizon for theoretical model
    pub fn new(prior_alpha: f64, prior_beta: f64, sigma: f64, tau: f64) -> Self {
        Self {
            prior_alpha: prior_alpha.max(1.0),
            prior_beta: prior_beta.max(1.0),
            observations: HashMap::new(),
            theoretical: FirstPassageFillModel::new(sigma, tau),
            min_observations: 5,
            max_age_secs: 3600.0, // 1 hour default
        }
    }

    /// Create with default parameters
    pub fn with_defaults(sigma: f64, tau: f64) -> Self {
        Self::new(2.0, 2.0, sigma, tau)
    }

    /// Update theoretical model parameters
    pub fn update_params(&mut self, sigma: f64, tau: f64) {
        self.theoretical.update(sigma, tau);
    }

    /// Record a fill/no-fill observation at a given depth
    ///
    /// # Arguments
    /// * `depth_bps` - Depth from mid in basis points
    /// * `filled` - Whether the order was filled
    pub fn record_observation(&mut self, depth_bps: f64, filled: bool) {
        let bucket = DepthBucket::from_bps(depth_bps);
        let obs = self.observations.entry(bucket).or_default();
        obs.record(filled);
    }

    /// Get posterior fill probability at a given depth
    ///
    /// Uses Bayesian posterior if sufficient observations exist,
    /// otherwise falls back to first-passage theoretical model.
    ///
    /// # Arguments
    /// * `depth_bps` - Depth from mid in basis points
    ///
    /// # Returns
    /// Posterior mean fill probability
    pub fn fill_probability(&self, depth_bps: f64) -> f64 {
        let bucket = DepthBucket::from_bps(depth_bps);

        if let Some(obs) = self.observations.get(&bucket) {
            if obs.attempts >= self.min_observations {
                // Posterior mean from Beta-Binomial conjugate update:
                // P(fill | data) ~ Beta(α₀ + fills, β₀ + non-fills)
                // Posterior mean = (α₀ + fills) / (α₀ + β₀ + attempts)
                let posterior_alpha = self.prior_alpha + obs.fills as f64;
                let posterior_beta = self.prior_beta + (obs.attempts - obs.fills) as f64;
                return posterior_alpha / (posterior_alpha + posterior_beta);
            }
        }

        // Fall back to theoretical first-passage model
        self.theoretical.probability(depth_bps)
    }

    /// Get fill probability with explicit sigma/tau override
    pub fn fill_probability_with_params(
        &self,
        depth_bps: f64,
        sigma: f64,
        tau: f64,
    ) -> f64 {
        let bucket = DepthBucket::from_bps(depth_bps);

        if let Some(obs) = self.observations.get(&bucket) {
            if obs.attempts >= self.min_observations {
                // Use empirical data when available
                let posterior_alpha = self.prior_alpha + obs.fills as f64;
                let posterior_beta = self.prior_beta + (obs.attempts - obs.fills) as f64;
                return posterior_alpha / (posterior_alpha + posterior_beta);
            }
        }

        // Fall back to theoretical with provided params
        self.theoretical.probability_with_params(depth_bps, sigma, tau)
    }

    /// Get posterior uncertainty (standard deviation) at a given depth
    ///
    /// Higher uncertainty for buckets with fewer observations.
    pub fn fill_uncertainty(&self, depth_bps: f64) -> f64 {
        let bucket = DepthBucket::from_bps(depth_bps);

        if let Some(obs) = self.observations.get(&bucket) {
            if obs.attempts > 0 {
                let alpha = self.prior_alpha + obs.fills as f64;
                let beta = self.prior_beta + (obs.attempts - obs.fills) as f64;
                let total = alpha + beta;

                // Beta distribution variance: αβ / ((α+β)²(α+β+1))
                let variance = (alpha * beta) / (total.powi(2) * (total + 1.0));
                return variance.sqrt();
            }
        }

        // High uncertainty for unobserved buckets
        0.5
    }

    /// Get 95% credible interval for fill probability
    ///
    /// Returns (lower, upper) bounds of the posterior credible interval.
    pub fn fill_credible_interval(&self, depth_bps: f64) -> (f64, f64) {
        let mean = self.fill_probability(depth_bps);
        let std = self.fill_uncertainty(depth_bps);

        // Approximate 95% CI as mean ± 2×std (for Beta, this is approximate)
        let lower = (mean - 2.0 * std).max(0.0);
        let upper = (mean + 2.0 * std).min(1.0);

        (lower, upper)
    }

    /// Get number of observations in a bucket
    pub fn observations_at_depth(&self, depth_bps: f64) -> (u64, u64) {
        let bucket = DepthBucket::from_bps(depth_bps);
        self.observations
            .get(&bucket)
            .map(|o| (o.fills, o.attempts))
            .unwrap_or((0, 0))
    }

    /// Get total observations across all buckets
    pub fn total_observations(&self) -> (u64, u64) {
        self.observations.values().fold((0, 0), |acc, obs| {
            (acc.0 + obs.fills, acc.1 + obs.attempts)
        })
    }

    /// Check if model has sufficient data for reliable estimates
    pub fn is_warmed_up(&self) -> bool {
        let (_, total_attempts) = self.total_observations();
        total_attempts >= 20 // Need at least 20 total observations
    }

    /// Clear all observations (for testing or reset)
    pub fn clear(&mut self) {
        self.observations.clear();
    }

    /// Log model diagnostics
    pub fn log_diagnostics(&self) {
        let (total_fills, total_attempts) = self.total_observations();
        let num_buckets = self.observations.len();
        let overall_rate = if total_attempts > 0 {
            total_fills as f64 / total_attempts as f64
        } else {
            0.0
        };

        debug!(
            total_fills,
            total_attempts,
            num_buckets,
            overall_fill_rate = %format!("{:.2}%", overall_rate * 100.0),
            is_warmed_up = self.is_warmed_up(),
            "Bayesian fill model diagnostics"
        );
    }
}

impl Default for BayesianFillModel {
    fn default() -> Self {
        Self::with_defaults(0.0001, 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_normal_cdf() {
        // Test known values
        assert!((std_normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((std_normal_cdf(-1.96) - 0.025).abs() < 0.01);
        assert!((std_normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!(std_normal_cdf(-5.0) < 0.001);
        assert!(std_normal_cdf(5.0) > 0.999);
    }

    #[test]
    fn test_first_passage_at_touch() {
        let model = FirstPassageFillModel::new(0.0001, 10.0);

        // At touch (depth = 0), probability should be 1.0
        assert!((model.probability(0.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_first_passage_depth_decay() {
        // Use higher sigma for visible probability differences across depths
        let model = FirstPassageFillModel::new(0.002, 10.0); // 20bp/sec volatility

        // Probability should decrease with depth
        let p_shallow = model.probability(2.0);
        let p_medium = model.probability(10.0);
        let p_deep = model.probability(50.0);

        assert!(p_shallow > p_medium);
        assert!(p_medium > p_deep);
        // With high sigma, even deep levels have some probability
        assert!(p_deep >= 0.0);
    }

    #[test]
    fn test_first_passage_volatility_effect() {
        let model_low_vol = FirstPassageFillModel::new(0.00005, 10.0);
        let model_high_vol = FirstPassageFillModel::new(0.0002, 10.0);

        // Higher volatility should increase fill probability at same depth
        let p_low = model_low_vol.probability(20.0);
        let p_high = model_high_vol.probability(20.0);

        assert!(p_high > p_low);
    }

    #[test]
    fn test_depth_bucket() {
        let bucket = DepthBucket::from_bps(5.0);
        assert!((bucket.center_bps() - 5.0).abs() < 0.01);
        assert!((bucket.lower_bps() - 4.0).abs() < 0.01);
        assert!((bucket.upper_bps() - 6.0).abs() < 0.01);

        // Same bucket for nearby depths
        let bucket1 = DepthBucket::from_bps(4.5);
        let bucket2 = DepthBucket::from_bps(5.5);
        assert_eq!(bucket1, bucket2);

        // Different buckets for distant depths
        let bucket3 = DepthBucket::from_bps(10.0);
        assert_ne!(bucket, bucket3);
    }

    #[test]
    fn test_bayesian_prior_only() {
        let model = BayesianFillModel::with_defaults(0.0001, 10.0);

        // Without observations, should fall back to theoretical
        let p = model.fill_probability(10.0);
        let p_theoretical = model.theoretical.probability(10.0);

        assert!((p - p_theoretical).abs() < EPSILON);
    }

    #[test]
    fn test_bayesian_with_observations() {
        let mut model = BayesianFillModel::new(1.0, 1.0, 0.0001, 10.0);
        model.min_observations = 3;

        // Record observations at 5bp depth
        for _ in 0..4 {
            model.record_observation(5.0, true); // 4 fills
        }
        model.record_observation(5.0, false); // 1 non-fill

        // Should use empirical data: (1 + 4) / (1 + 1 + 5) = 5/7 ≈ 0.714
        let p = model.fill_probability(5.0);
        assert!((p - 5.0 / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_bayesian_uncertainty() {
        let mut model = BayesianFillModel::with_defaults(0.0001, 10.0);
        model.min_observations = 1;

        // High uncertainty for unobserved depths
        let u_unobserved = model.fill_uncertainty(100.0);
        assert!(u_unobserved > 0.3);

        // Record many observations at 5bp
        for _ in 0..100 {
            model.record_observation(5.0, true);
        }

        // Low uncertainty for well-observed depths
        let u_observed = model.fill_uncertainty(5.0);
        assert!(u_observed < u_unobserved);
    }

    #[test]
    fn test_bayesian_credible_interval() {
        let mut model = BayesianFillModel::with_defaults(0.0001, 10.0);

        // Record some observations
        for _ in 0..20 {
            model.record_observation(5.0, true);
        }
        for _ in 0..10 {
            model.record_observation(5.0, false);
        }

        let (lower, upper) = model.fill_credible_interval(5.0);
        let mean = model.fill_probability(5.0);

        assert!(lower < mean);
        assert!(mean < upper);
        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
    }

    #[test]
    fn test_bucket_observations() {
        // Test via BayesianFillModel's public interface
        let mut model = BayesianFillModel::with_defaults(0.0001, 10.0);
        model.min_observations = 1;

        model.record_observation(5.0, true);
        model.record_observation(5.0, true);
        model.record_observation(5.0, false);

        let (fills, attempts) = model.observations_at_depth(5.0);
        assert_eq!(fills, 2);
        assert_eq!(attempts, 3);
    }

    #[test]
    fn test_bayesian_warmup() {
        let mut model = BayesianFillModel::with_defaults(0.0001, 10.0);

        assert!(!model.is_warmed_up());

        for i in 0..25 {
            model.record_observation(i as f64, i % 2 == 0);
        }

        assert!(model.is_warmed_up());
    }

    #[test]
    fn test_total_observations() {
        let mut model = BayesianFillModel::with_defaults(0.0001, 10.0);

        model.record_observation(5.0, true);
        model.record_observation(5.0, false);
        model.record_observation(10.0, true);

        let (fills, attempts) = model.total_observations();
        assert_eq!(fills, 2);
        assert_eq!(attempts, 3);
    }

    #[test]
    fn test_realistic_btc_scenario() {
        // Simulate realistic BTC volatility and horizon
        // BTC volatility is typically 2-5% daily, which is ~0.001-0.002 per second
        let sigma = 0.001; // 10bp/sec (realistic for volatile BTC)
        let tau = 10.0; // 10 second horizon

        let model = BayesianFillModel::with_defaults(sigma, tau);

        // At 5bp depth, should have reasonable fill probability
        // z = -0.0005 / (0.001 * √10) = -0.158, Φ(-0.158) ≈ 0.437, 2*0.437 ≈ 0.87
        let p_5bp = model.fill_probability(5.0);
        assert!(p_5bp > 0.1, "p_5bp = {} should be > 0.1", p_5bp);

        // At 50bp depth, should have lower fill probability
        // z = -0.005 / (0.001 * √10) = -1.58, Φ(-1.58) ≈ 0.057, 2*0.057 ≈ 0.11
        let p_50bp = model.fill_probability(50.0);
        assert!(p_50bp < p_5bp, "p_50bp = {} should be < p_5bp = {}", p_50bp, p_5bp);
    }
}
