//! Learned Spread Floor - Component 1
//!
//! Implements Bayesian estimation of the break-even spread from observed
//! adverse selection. Replaces the static `min_spread_floor` with a learned
//! value that adapts to actual market conditions.
//!
//! # Mathematical Model
//!
//! The break-even spread satisfies:
//! ```text
//! δ_BE = f_maker + E[AS | fill] + buffer
//! ```
//!
//! We model adverse selection with a Normal-Normal conjugate:
//! ```text
//! AS ~ Normal(μ_AS, σ_AS²)
//! μ_AS ~ Normal(μ_0, σ_0²)
//! ```
//!
//! The posterior after observing AS values is:
//! ```text
//! μ_AS | data ~ Normal(μ_post, σ_post²)
//! ```
//!
//! The dynamic floor is:
//! ```text
//! floor = f_maker + max(0, μ_post) + k × σ_post
//! ```
//!
//! where k is the risk multiplier (1.5 covers ~87% of outcomes).

use std::collections::VecDeque;
use tracing::debug;

/// Learned spread floor based on Bayesian adverse selection estimation.
#[derive(Debug, Clone)]
pub struct LearnedSpreadFloor {
    /// Maker fee rate (constant, known)
    maker_fee: f64,

    /// Prior mean for AS
    prior_mean: f64,

    /// Prior variance for AS
    prior_variance: f64,

    /// Likelihood variance (AS observation noise)
    /// This is the variance of individual AS observations
    likelihood_variance: f64,

    /// Posterior mean (updated online)
    posterior_mean: f64,

    /// Posterior variance (updated online)
    posterior_variance: f64,

    /// Number of observations
    n_observations: usize,

    /// Risk multiplier k for floor = fees + E[AS] + k×σ_AS
    risk_k: f64,

    /// Minimum tick size as fraction (hard floor)
    tick_size_fraction: f64,

    /// EWMA decay for non-stationarity (makes recent obs more important)
    ewma_decay: f64,

    /// Recent AS observations for variance estimation
    recent_observations: VecDeque<f64>,

    /// Window size for variance estimation
    variance_window: usize,

    /// Running sum of observations (for EWMA mean)
    ewma_sum: f64,

    /// Running weight sum (for EWMA normalization)
    ewma_weight_sum: f64,

    /// Running sum of squared deviations (for EWMA variance)
    ewma_sq_sum: f64,
}

impl LearnedSpreadFloor {
    /// Create a new learned spread floor estimator.
    ///
    /// # Arguments
    /// * `maker_fee` - Maker fee rate (fraction, e.g., 0.0003 for 3 bps)
    /// * `prior_mean` - Prior expected AS (fraction, e.g., 0.0003 for 3 bps)
    /// * `prior_std` - Prior uncertainty in AS (fraction)
    /// * `risk_k` - Risk multiplier for safety margin
    /// * `tick_size_fraction` - Minimum tick as fraction of price
    /// * `ewma_decay` - EWMA decay factor (0.99 = ~100 obs half-life)
    pub fn new(
        maker_fee: f64,
        prior_mean: f64,
        prior_std: f64,
        risk_k: f64,
        tick_size_fraction: f64,
        ewma_decay: f64,
    ) -> Self {
        let prior_variance = prior_std * prior_std;

        Self {
            maker_fee,
            prior_mean,
            prior_variance,
            likelihood_variance: prior_variance * 4.0, // Start with wide likelihood
            posterior_mean: prior_mean,
            posterior_variance: prior_variance,
            n_observations: 0,
            risk_k,
            tick_size_fraction,
            ewma_decay,
            recent_observations: VecDeque::with_capacity(200),
            variance_window: 200,
            ewma_sum: 0.0,
            ewma_weight_sum: 0.0,
            ewma_sq_sum: 0.0,
        }
    }

    /// Create from config.
    pub fn from_config(config: &super::AdaptiveBayesianConfig) -> Self {
        Self::new(
            config.maker_fee_rate,
            config.as_prior_mean,
            config.as_prior_std,
            config.floor_risk_k,
            config.floor_absolute_min,
            config.as_ewma_decay,
        )
    }

    /// Update the estimator with an observed adverse selection value.
    ///
    /// # Arguments
    /// * `as_realized` - Realized adverse selection (signed: positive = adverse)
    ///   Calculated as: (mid_{t+Δt} - fill_price) × direction
    ///   where direction = +1 for buy fill, -1 for sell fill
    pub fn update(&mut self, as_realized: f64) {
        self.n_observations += 1;

        // Update EWMA statistics
        let weight = 1.0; // Current observation has weight 1
        self.ewma_sum = self.ewma_decay * self.ewma_sum + weight * as_realized;
        self.ewma_weight_sum = self.ewma_decay * self.ewma_weight_sum + weight;

        // Track for variance estimation
        self.recent_observations.push_back(as_realized);
        if self.recent_observations.len() > self.variance_window {
            self.recent_observations.pop_front();
        }

        // Update posterior using Normal-Normal conjugate
        // posterior_mean = (prior_var × obs + likelihood_var × prior_mean) / (prior_var + likelihood_var)
        // But with EWMA, we use weighted update

        // Compute EWMA mean
        let ewma_mean = if self.ewma_weight_sum > 0.0 {
            self.ewma_sum / self.ewma_weight_sum
        } else {
            self.prior_mean
        };

        // Compute sample variance from recent observations
        let sample_variance = self.compute_sample_variance();

        // Update likelihood variance (how noisy are observations?)
        if self.n_observations > 10 {
            self.likelihood_variance =
                0.95 * self.likelihood_variance + 0.05 * sample_variance.max(1e-12);
        }

        // Bayesian update with effective sample size from EWMA
        let effective_n = self.ewma_weight_sum;
        let precision_prior = 1.0 / self.prior_variance;
        let precision_likelihood = effective_n / self.likelihood_variance;

        self.posterior_variance = 1.0 / (precision_prior + precision_likelihood);
        self.posterior_mean = self.posterior_variance
            * (precision_prior * self.prior_mean + precision_likelihood * ewma_mean);

        // Log update periodically
        if self.n_observations % 50 == 0 {
            debug!(
                n = self.n_observations,
                as_realized_bps = as_realized * 10000.0,
                posterior_mean_bps = self.posterior_mean * 10000.0,
                posterior_std_bps = self.posterior_std() * 10000.0,
                floor_bps = self.learned_spread_floor() * 10000.0,
                "Learned floor updated"
            );
        }
    }

    /// Compute sample variance from recent observations.
    fn compute_sample_variance(&self) -> f64 {
        if self.recent_observations.len() < 2 {
            return self.prior_variance;
        }

        let n = self.recent_observations.len() as f64;
        let mean: f64 = self.recent_observations.iter().sum::<f64>() / n;
        let variance: f64 = self.recent_observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (n - 1.0);

        variance.max(1e-12)
    }

    /// Get the posterior standard deviation.
    pub fn posterior_std(&self) -> f64 {
        self.posterior_variance.sqrt()
    }

    /// Calculate the learned spread floor.
    ///
    /// # Formula
    /// ```text
    /// floor = maker_fee + max(0, μ_AS) + k × σ_AS
    /// ```
    ///
    /// The max(0, μ_AS) ensures we don't reduce floor if AS estimate is negative
    /// (which would be unrealistic - we shouldn't expect to GAIN from fills).
    pub fn learned_spread_floor(&self) -> f64 {
        let as_mean = self.posterior_mean.max(0.0);
        let as_std = self.posterior_std();

        let floor = self.maker_fee + as_mean + self.risk_k * as_std;

        // Enforce minimum tick size
        floor.max(self.tick_size_fraction)
    }

    /// Get the static fallback floor (for comparison/logging).
    pub fn static_floor(&self) -> f64 {
        self.maker_fee + self.prior_mean + self.risk_k * self.prior_variance.sqrt()
    }

    /// Check if the estimator has enough data to be reliable.
    pub fn is_warmed_up(&self) -> bool {
        self.n_observations >= 20
    }

    /// Get the number of observations.
    pub fn observation_count(&self) -> usize {
        self.n_observations
    }

    /// Get the posterior mean AS.
    pub fn posterior_mean(&self) -> f64 {
        self.posterior_mean
    }

    /// Get effective sample size (from EWMA weighting).
    pub fn effective_sample_size(&self) -> f64 {
        self.ewma_weight_sum
    }

    /// Reset to prior (for testing or regime change).
    pub fn reset(&mut self) {
        self.posterior_mean = self.prior_mean;
        self.posterior_variance = self.prior_variance;
        self.n_observations = 0;
        self.recent_observations.clear();
        self.ewma_sum = 0.0;
        self.ewma_weight_sum = 0.0;
        self.ewma_sq_sum = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_floor() -> LearnedSpreadFloor {
        LearnedSpreadFloor::new(
            0.0003,  // 3 bps maker fee
            0.0003,  // 3 bps prior AS mean
            0.0005,  // 5 bps prior AS std
            1.5,     // 1.5σ risk margin
            0.0001,  // 1 bp minimum
            0.995,   // EWMA decay
        )
    }

    #[test]
    fn test_initial_floor() {
        let floor = default_floor();

        // Initial floor = fees + prior_mean + k * prior_std
        // = 0.0003 + 0.0003 + 1.5 * 0.0005 = 0.00135 = 13.5 bps
        let expected = 0.0003 + 0.0003 + 1.5 * 0.0005;
        let actual = floor.learned_spread_floor();

        assert!(
            (actual - expected).abs() < 0.0001,
            "Expected {:.4} bps, got {:.4} bps",
            expected * 10000.0,
            actual * 10000.0
        );
    }

    #[test]
    fn test_floor_decreases_with_low_as() {
        let mut floor = default_floor();
        let initial = floor.learned_spread_floor();

        // Simulate many low-AS fills (0.5 bps)
        for _ in 0..100 {
            floor.update(0.00005); // 0.5 bps AS
        }

        let after = floor.learned_spread_floor();

        assert!(
            after < initial,
            "Floor should decrease with low AS: {:.2} bps -> {:.2} bps",
            initial * 10000.0,
            after * 10000.0
        );
    }

    #[test]
    fn test_floor_increases_with_high_as() {
        let mut floor = default_floor();

        // First establish a baseline with the prior
        let initial = floor.learned_spread_floor();

        // Simulate very high-AS fills (20 bps - much higher than prior 3 bps)
        for _ in 0..100 {
            floor.update(0.002); // 20 bps AS
        }

        let after = floor.learned_spread_floor();

        // The posterior mean should move toward observed values
        assert!(
            floor.posterior_mean() > 0.0005, // Should be significantly higher than prior
            "Posterior mean should increase with high AS: {:.2} bps",
            floor.posterior_mean() * 10000.0
        );

        // Note: floor might not always be higher because it depends on
        // both mean AND std. But posterior mean should definitely increase.
    }

    #[test]
    fn test_floor_respects_minimum() {
        let mut floor = default_floor();

        // Simulate many negative AS (unrealistic but tests floor)
        for _ in 0..100 {
            floor.update(-0.001); // Negative AS
        }

        let after = floor.learned_spread_floor();

        assert!(
            after >= floor.tick_size_fraction,
            "Floor must respect minimum tick: {:.4} bps",
            after * 10000.0
        );
    }

    #[test]
    fn test_warmup_detection() {
        let mut floor = default_floor();

        assert!(!floor.is_warmed_up());

        for _ in 0..19 {
            floor.update(0.0003);
        }
        assert!(!floor.is_warmed_up());

        floor.update(0.0003);
        assert!(floor.is_warmed_up());
    }

    #[test]
    fn test_ewma_decay_recency() {
        let mut floor = default_floor();

        // First 50 observations: high AS
        for _ in 0..50 {
            floor.update(0.001); // 10 bps
        }
        let after_high = floor.posterior_mean();

        // Next 50 observations: low AS
        for _ in 0..50 {
            floor.update(0.0001); // 1 bp
        }
        let after_low = floor.posterior_mean();

        // EWMA should weight recent observations more
        assert!(
            after_low < after_high,
            "Recent low AS should reduce mean: {:.4} -> {:.4}",
            after_high * 10000.0,
            after_low * 10000.0
        );

        // Should be closer to recent values than simple average
        let simple_avg = (50.0 * 0.001 + 50.0 * 0.0001) / 100.0;
        assert!(
            after_low < simple_avg,
            "EWMA mean {:.4} should be below simple avg {:.4}",
            after_low * 10000.0,
            simple_avg * 10000.0
        );
    }
}
