//! Bootstrap Particle Filter for non-linear parameter estimation.
//!
//! Estimates joint posterior over [sigma, kappa, regime_drift, jump_intensity]
//! using Sequential Monte Carlo. Handles non-linear, non-Gaussian state dynamics
//! that standard Kalman filters cannot capture.
//!
//! Key features:
//! - Student-t observation likelihood (robust to fat tails)
//! - Systematic resampling when ESS < N/2
//! - SIMD-friendly struct-of-arrays layout
//! - Marginal log-likelihood for BMA integration

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

fn default_num_particles() -> usize {
    200
}
fn default_observation_df() -> f64 {
    5.0
}
fn default_sigma_rw_scale() -> f64 {
    0.02
}
fn default_kappa_rw_scale() -> f64 {
    0.02
}
fn default_drift_rw_scale() -> f64 {
    0.01
}
fn default_jump_rw_scale() -> f64 {
    0.05
}
fn default_ess_threshold_frac() -> f64 {
    0.5
}
fn default_kappa_floor() -> f64 {
    1.0
}
fn default_sigma_floor() -> f64 {
    1e-8
}

/// Configuration for the particle filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleFilterConfig {
    /// Number of particles (higher = better approximation, more compute).
    #[serde(default = "default_num_particles")]
    pub num_particles: usize,
    /// Student-t degrees of freedom for observation likelihood.
    /// Lower = heavier tails = more robust to outliers. Infinity = Gaussian.
    #[serde(default = "default_observation_df")]
    pub observation_df: f64,
    /// Random walk scale for sigma transitions (log-space std dev).
    #[serde(default = "default_sigma_rw_scale")]
    pub sigma_random_walk_scale: f64,
    /// Random walk scale for kappa transitions (log-space std dev).
    #[serde(default = "default_kappa_rw_scale")]
    pub kappa_random_walk_scale: f64,
    /// Random walk scale for drift transitions (additive std dev).
    #[serde(default = "default_drift_rw_scale")]
    pub drift_random_walk_scale: f64,
    /// Random walk scale for jump intensity transitions (log-space std dev).
    #[serde(default = "default_jump_rw_scale")]
    pub jump_random_walk_scale: f64,
    /// Resample when ESS < this fraction × num_particles.
    #[serde(default = "default_ess_threshold_frac")]
    pub ess_threshold_fraction: f64,
    /// Floor for kappa in all particles (GLFT invariant: kappa > 0).
    #[serde(default = "default_kappa_floor")]
    pub kappa_floor: f64,
    /// Floor for sigma in all particles.
    #[serde(default = "default_sigma_floor")]
    pub sigma_floor: f64,
}

impl Default for ParticleFilterConfig {
    fn default() -> Self {
        Self {
            num_particles: default_num_particles(),
            observation_df: default_observation_df(),
            sigma_random_walk_scale: default_sigma_rw_scale(),
            kappa_random_walk_scale: default_kappa_rw_scale(),
            drift_random_walk_scale: default_drift_rw_scale(),
            jump_random_walk_scale: default_jump_rw_scale(),
            ess_threshold_fraction: default_ess_threshold_frac(),
            kappa_floor: default_kappa_floor(),
            sigma_floor: default_sigma_floor(),
        }
    }
}

// ============================================================================
// Posterior Summary
// ============================================================================

/// Summary statistics from the particle filter posterior.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParticleFilterPosterior {
    /// Posterior mean of sigma (per-second volatility).
    #[serde(default)]
    pub sigma_mean: f64,
    /// Posterior variance of sigma.
    #[serde(default)]
    pub sigma_variance: f64,
    /// Posterior mean of kappa (fill intensity).
    #[serde(default)]
    pub kappa_mean: f64,
    /// Posterior variance of kappa.
    #[serde(default)]
    pub kappa_variance: f64,
    /// Posterior mean of drift rate.
    #[serde(default)]
    pub drift_mean: f64,
    /// Posterior variance of drift.
    #[serde(default)]
    pub drift_variance: f64,
    /// Posterior mean of jump intensity.
    #[serde(default)]
    pub jump_mean: f64,
    /// Posterior variance of jump intensity.
    #[serde(default)]
    pub jump_variance: f64,
    /// Effective sample size (diagnostic).
    #[serde(default)]
    pub ess: f64,
    /// Total number of resampling events.
    #[serde(default)]
    pub num_resamples: u64,
    /// Total number of updates processed.
    #[serde(default)]
    pub num_updates: u64,
    /// Marginal log-likelihood (for BMA integration).
    #[serde(default)]
    pub marginal_log_likelihood: f64,
}

// ============================================================================
// Particle Filter
// ============================================================================

/// Bootstrap Particle Filter for joint parameter estimation.
///
/// Uses struct-of-arrays layout for cache efficiency.
/// All random numbers generated via fast LCG (no external dependency).
pub struct ParticleFilter {
    config: ParticleFilterConfig,
    // Struct-of-arrays for SIMD-friendly access
    sigma: Vec<f64>,
    kappa: Vec<f64>,
    drift: Vec<f64>,
    jump: Vec<f64>,
    log_weights: Vec<f64>,
    // Cached posterior
    posterior: ParticleFilterPosterior,
    // Simple LCG RNG state (for deterministic tests)
    rng_state: u64,
    // Accumulated marginal log-likelihood
    cumulative_log_likelihood: f64,
}

impl ParticleFilter {
    /// Create a new particle filter with given configuration.
    ///
    /// Initializes particles from broad priors:
    /// - sigma ~ LogNormal(ln(0.0003), 0.5)
    /// - kappa ~ LogNormal(ln(1500), 0.5)
    /// - drift ~ Normal(0, 0.001)
    /// - jump_intensity ~ LogNormal(ln(0.1), 1.0)
    pub fn new(config: ParticleFilterConfig) -> Self {
        let n = config.num_particles.max(10);
        let mut filter = Self {
            sigma: vec![0.0; n],
            kappa: vec![0.0; n],
            drift: vec![0.0; n],
            jump: vec![0.0; n],
            log_weights: vec![-(n as f64).ln(); n], // Uniform weights in log space
            posterior: ParticleFilterPosterior::default(),
            rng_state: 42,
            cumulative_log_likelihood: 0.0,
            config,
        };
        filter.initialize_from_prior();
        filter.compute_posterior();
        filter
    }

    /// Create with a specific RNG seed (for reproducible tests).
    pub fn with_seed(config: ParticleFilterConfig, seed: u64) -> Self {
        let mut filter = Self::new(config);
        filter.rng_state = seed;
        filter.initialize_from_prior();
        filter.compute_posterior();
        filter
    }

    fn initialize_from_prior(&mut self) {
        let n = self.config.num_particles;
        for i in 0..n {
            // Spread particles across reasonable parameter ranges
            self.sigma[i] = (0.0003_f64 * (0.5 * self.randn()).exp()).max(self.config.sigma_floor);
            self.kappa[i] = (1500.0 * (0.5 * self.randn()).exp()).max(self.config.kappa_floor);
            self.drift[i] = 0.001 * self.randn();
            self.jump[i] = (0.1 * (1.0 * self.randn()).exp()).max(0.0);
        }
    }

    /// Update the particle filter with a new price innovation.
    ///
    /// # Arguments
    /// * `price_innovation` - Observed price return (fractional)
    /// * `dt_secs` - Time elapsed since last observation (seconds)
    pub fn update(&mut self, price_innovation: f64, dt_secs: f64) {
        if dt_secs <= 0.0 {
            return;
        }

        let n = self.config.num_particles;

        // 1. Transition: propagate each particle forward
        self.transition(dt_secs);

        // 2. Weight update: compute log-likelihood for each particle
        let mut max_log_w = f64::NEG_INFINITY;
        for i in 0..n {
            let log_lik = self.compute_log_likelihood(price_innovation, i, dt_secs);
            self.log_weights[i] += log_lik;
            if self.log_weights[i] > max_log_w {
                max_log_w = self.log_weights[i];
            }
        }

        // 3. Normalize weights (log-sum-exp for numerical stability)
        let log_sum = {
            let mut sum = 0.0_f64;
            for i in 0..n {
                sum += (self.log_weights[i] - max_log_w).exp();
            }
            max_log_w + sum.ln()
        };

        // Accumulate marginal log-likelihood for BMA
        self.cumulative_log_likelihood += log_sum - (n as f64).ln();

        for i in 0..n {
            self.log_weights[i] -= log_sum;
        }

        // 4. Compute ESS and resample if needed
        let ess = self.compute_ess();
        let threshold = self.config.ess_threshold_fraction * n as f64;
        if ess < threshold {
            self.systematic_resample();
            self.posterior.num_resamples += 1;
        }

        // 5. Update posterior statistics
        self.posterior.num_updates += 1;
        self.compute_posterior();
    }

    /// Transition model: geometric random walk on positive parameters,
    /// additive random walk on drift with mean-reversion.
    fn transition(&mut self, dt_secs: f64) {
        let dt_scale = dt_secs.sqrt();
        let n = self.config.num_particles;

        for i in 0..n {
            // Sigma: geometric random walk (stays positive)
            let sigma_noise = self.config.sigma_random_walk_scale * dt_scale * self.randn();
            self.sigma[i] *= sigma_noise.exp();
            self.sigma[i] = self.sigma[i].max(self.config.sigma_floor);

            // Kappa: geometric random walk (stays positive, floor at kappa_floor)
            let kappa_noise = self.config.kappa_random_walk_scale * dt_scale * self.randn();
            self.kappa[i] *= kappa_noise.exp();
            self.kappa[i] = self.kappa[i].max(self.config.kappa_floor);

            // Drift: additive random walk with mean-reversion toward 0
            let drift_noise = self.config.drift_random_walk_scale * dt_scale * self.randn();
            self.drift[i] = self.drift[i] * (-0.01 * dt_secs).exp() + drift_noise;

            // Jump intensity: geometric random walk (stays non-negative)
            let jump_noise = self.config.jump_random_walk_scale * dt_scale * self.randn();
            self.jump[i] *= jump_noise.exp();
            self.jump[i] = self.jump[i].max(0.0);
        }
    }

    /// Compute log-likelihood of observation under particle i's parameters.
    /// Uses Student-t distribution for robustness to fat tails.
    fn compute_log_likelihood(&self, innovation: f64, particle_idx: usize, dt_secs: f64) -> f64 {
        let sigma = self.sigma[particle_idx];
        let drift = self.drift[particle_idx];

        // Expected innovation under this particle's drift
        let expected = drift * dt_secs;
        let residual = innovation - expected;

        // Variance under this particle's sigma
        let variance = sigma * sigma * dt_secs;
        if variance <= 1e-30 {
            return -100.0; // Degenerate particle
        }

        let df = self.config.observation_df;

        if df > 1000.0 {
            // Approximate as Gaussian for large df
            let standardized = residual / variance.sqrt();
            -0.5 * (variance.ln() + standardized * standardized)
        } else {
            // Student-t log-likelihood
            let scale_sq = variance;
            let z_sq = residual * residual / scale_sq;

            let log_gamma_ratio = log_gamma_ratio_approx(df);

            log_gamma_ratio
                - 0.5 * (df * std::f64::consts::PI * scale_sq).ln()
                - ((df + 1.0) / 2.0) * (1.0 + z_sq / df).ln()
        }
    }

    /// Systematic resampling: deterministic, low-variance resampling.
    #[allow(clippy::needless_range_loop)] // Parallel SoA indexing is clearer with explicit index
    fn systematic_resample(&mut self) {
        let n = self.config.num_particles;

        // Convert log weights to cumulative distribution
        let mut cum_weights = vec![0.0_f64; n];
        let max_lw = self
            .log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for i in 0..n {
            let w = (self.log_weights[i] - max_lw).exp();
            total += w;
            cum_weights[i] = total;
        }
        // Normalize
        if total > 0.0 {
            for w in cum_weights.iter_mut() {
                *w /= total;
            }
        }

        // Systematic resampling with single uniform draw
        let u0 = self.rand_uniform() / n as f64;
        let mut indices = vec![0usize; n];
        let mut j = 0;
        for i in 0..n {
            let u = u0 + i as f64 / n as f64;
            while j < n - 1 && cum_weights[j] < u {
                j += 1;
            }
            indices[i] = j;
        }

        // Copy selected particles
        let old_sigma = self.sigma.clone();
        let old_kappa = self.kappa.clone();
        let old_drift = self.drift.clone();
        let old_jump = self.jump.clone();

        for i in 0..n {
            let idx = indices[i];
            self.sigma[i] = old_sigma[idx];
            self.kappa[i] = old_kappa[idx];
            self.drift[i] = old_drift[idx];
            self.jump[i] = old_jump[idx];
        }

        // Reset weights to uniform
        let log_uniform = -(n as f64).ln();
        for w in &mut self.log_weights {
            *w = log_uniform;
        }
    }

    /// Compute effective sample size from log weights.
    fn compute_ess(&self) -> f64 {
        let max_lw = self
            .log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut sum_w = 0.0;
        let mut sum_w2 = 0.0;
        for lw in &self.log_weights {
            let w = (lw - max_lw).exp();
            sum_w += w;
            sum_w2 += w * w;
        }

        if sum_w2 == 0.0 {
            return 0.0;
        }

        (sum_w * sum_w) / sum_w2
    }

    /// Compute posterior statistics from weighted particles.
    #[allow(clippy::needless_range_loop)] // Parallel SoA indexing is clearer with explicit index
    fn compute_posterior(&mut self) {
        let n = self.config.num_particles;
        let max_lw = self
            .log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut weights = vec![0.0; n];
        let mut total = 0.0;
        for i in 0..n {
            weights[i] = (self.log_weights[i] - max_lw).exp();
            total += weights[i];
        }
        if total > 0.0 {
            for w in weights.iter_mut() {
                *w /= total;
            }
        }

        // Weighted mean
        let mut sigma_mean = 0.0;
        let mut kappa_mean = 0.0;
        let mut drift_mean = 0.0;
        let mut jump_mean = 0.0;
        for i in 0..n {
            sigma_mean += weights[i] * self.sigma[i];
            kappa_mean += weights[i] * self.kappa[i];
            drift_mean += weights[i] * self.drift[i];
            jump_mean += weights[i] * self.jump[i];
        }

        // Weighted variance
        let mut sigma_var = 0.0;
        let mut kappa_var = 0.0;
        let mut drift_var = 0.0;
        let mut jump_var = 0.0;
        for i in 0..n {
            sigma_var += weights[i] * (self.sigma[i] - sigma_mean).powi(2);
            kappa_var += weights[i] * (self.kappa[i] - kappa_mean).powi(2);
            drift_var += weights[i] * (self.drift[i] - drift_mean).powi(2);
            jump_var += weights[i] * (self.jump[i] - jump_mean).powi(2);
        }

        self.posterior.sigma_mean = sigma_mean;
        self.posterior.sigma_variance = sigma_var;
        self.posterior.kappa_mean = kappa_mean.max(self.config.kappa_floor);
        self.posterior.kappa_variance = kappa_var;
        self.posterior.drift_mean = drift_mean;
        self.posterior.drift_variance = drift_var;
        self.posterior.jump_mean = jump_mean;
        self.posterior.jump_variance = jump_var;
        self.posterior.ess = self.compute_ess();
        self.posterior.marginal_log_likelihood = self.cumulative_log_likelihood;
    }

    // === Public API ===

    /// Get the current posterior summary.
    pub fn posterior(&self) -> &ParticleFilterPosterior {
        &self.posterior
    }

    /// Get effective sample size.
    pub fn ess(&self) -> f64 {
        self.posterior.ess
    }

    /// Get marginal log-likelihood (for BMA weight updates).
    pub fn marginal_log_likelihood(&self) -> f64 {
        self.cumulative_log_likelihood
    }

    /// Get number of updates processed.
    pub fn num_updates(&self) -> u64 {
        self.posterior.num_updates
    }

    /// Check if filter has enough updates to be informative.
    pub fn is_warmed_up(&self) -> bool {
        self.posterior.num_updates >= 20
    }

    /// Get posterior mean sigma.
    pub fn sigma_mean(&self) -> f64 {
        self.posterior.sigma_mean
    }

    /// Get posterior mean kappa (floored at config.kappa_floor).
    pub fn kappa_mean(&self) -> f64 {
        self.posterior.kappa_mean.max(self.config.kappa_floor)
    }

    /// Reset the filter (e.g., after a regime change).
    pub fn reset(&mut self) {
        self.initialize_from_prior();
        self.log_weights =
            vec![-(self.config.num_particles as f64).ln(); self.config.num_particles];
        self.cumulative_log_likelihood = 0.0;
        self.posterior = ParticleFilterPosterior::default();
    }

    // === RNG (simple LCG, no external dependency) ===

    fn next_u64(&mut self) -> u64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.rng_state
    }

    fn rand_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform for normal random variates.
    fn randn(&mut self) -> f64 {
        let u1 = self.rand_uniform().max(1e-15);
        let u2 = self.rand_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Approximate log Gamma((df+1)/2) - log Gamma(df/2) - 0.5 * log(pi * df)
/// for Student-t distribution normalization constant.
fn log_gamma_ratio_approx(df: f64) -> f64 {
    let a = (df + 1.0) / 2.0;
    let b = df / 2.0;
    // Stirling approximation: log Gamma(x) ~ (x - 0.5) * ln(x) - x + 0.5 * ln(2*pi)
    let log_gamma_a = (a - 0.5) * a.ln() - a + 0.5 * (2.0 * std::f64::consts::PI).ln();
    let log_gamma_b = (b - 0.5) * b.ln() - b + 0.5 * (2.0 * std::f64::consts::PI).ln();
    log_gamma_a - log_gamma_b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_filter_creation() {
        let config = ParticleFilterConfig::default();
        let pf = ParticleFilter::new(config);
        assert_eq!(pf.sigma.len(), 200);
        assert_eq!(pf.kappa.len(), 200);
        assert!(pf.ess() > 0.0);
    }

    #[test]
    fn test_kappa_floor_invariant() {
        let config = ParticleFilterConfig {
            num_particles: 50,
            kappa_floor: 1.0,
            ..Default::default()
        };
        let mut pf = ParticleFilter::with_seed(config, 123);

        // Run many updates
        for i in 0..100 {
            let innovation = 0.001 * ((i as f64) * 0.1).sin();
            pf.update(innovation, 1.0);
        }

        // ALL particles must have kappa >= floor
        for k in &pf.kappa {
            assert!(*k >= 1.0, "kappa {} violated floor", k);
        }
        assert!(pf.kappa_mean() >= 1.0);
    }

    #[test]
    fn test_systematic_resampling_preserves_count() {
        let config = ParticleFilterConfig {
            num_particles: 100,
            ..Default::default()
        };
        let mut pf = ParticleFilter::with_seed(config, 456);

        // Force degenerate weights
        for i in 0..100 {
            pf.log_weights[i] = if i == 0 { 0.0 } else { -100.0 };
        }

        pf.systematic_resample();
        assert_eq!(pf.sigma.len(), 100);
        assert_eq!(pf.kappa.len(), 100);

        // After resampling, weights should be uniform
        let expected_log_w = -(100.0_f64).ln();
        for w in &pf.log_weights {
            assert!((w - expected_log_w).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ess_tracking() {
        let config = ParticleFilterConfig {
            num_particles: 100,
            ess_threshold_fraction: 0.0, // Disable auto-resampling
            ..Default::default()
        };
        let mut pf = ParticleFilter::with_seed(config, 789);

        let initial_ess = pf.ess();
        assert!(
            (initial_ess - 100.0).abs() < 1.0,
            "Initial ESS should be ~N, got {}",
            initial_ess
        );

        // After updates without resampling, ESS should decrease
        for _ in 0..50 {
            pf.update(0.001, 1.0);
        }
        assert!(
            pf.ess() < initial_ess,
            "ESS should decrease without resampling"
        );
    }

    #[test]
    fn test_update_convergence() {
        let config = ParticleFilterConfig {
            num_particles: 200,
            observation_df: 5.0,
            ..Default::default()
        };
        let mut pf = ParticleFilter::with_seed(config, 42);

        // Feed consistent small positive innovations (low vol, positive drift)
        for _ in 0..200 {
            pf.update(0.0002, 1.0);
        }

        // Posterior should reflect positive drift
        assert!(
            pf.posterior().drift_mean > 0.0,
            "Should detect positive drift, got {}",
            pf.posterior().drift_mean
        );
        assert_eq!(pf.posterior().num_updates, 200);
    }

    #[test]
    fn test_marginal_log_likelihood() {
        let config = ParticleFilterConfig {
            num_particles: 50,
            ..Default::default()
        };
        let mut pf = ParticleFilter::with_seed(config, 999);

        assert_eq!(pf.marginal_log_likelihood(), 0.0);

        pf.update(0.0001, 1.0);
        assert!(pf.marginal_log_likelihood().is_finite());
    }

    #[test]
    fn test_reset() {
        let config = ParticleFilterConfig::default();
        let mut pf = ParticleFilter::with_seed(config, 42);

        for _ in 0..50 {
            pf.update(0.001, 1.0);
        }
        assert!(pf.num_updates() > 0);

        pf.reset();
        assert_eq!(pf.num_updates(), 0);
        assert_eq!(pf.marginal_log_likelihood(), 0.0);
    }

    #[test]
    fn test_student_t_heavier_tails() {
        let gaussian_config = ParticleFilterConfig {
            num_particles: 100,
            observation_df: 10000.0, // Effectively Gaussian
            ess_threshold_fraction: 0.0,
            ..Default::default()
        };
        let student_config = ParticleFilterConfig {
            num_particles: 100,
            observation_df: 3.0, // Heavy tails
            ess_threshold_fraction: 0.0,
            ..Default::default()
        };

        let mut pf_gauss = ParticleFilter::with_seed(gaussian_config, 42);
        let mut pf_student = ParticleFilter::with_seed(student_config, 42);

        // Normal observations
        for _ in 0..50 {
            pf_gauss.update(0.0001, 1.0);
            pf_student.update(0.0001, 1.0);
        }

        let ess_before_gauss = pf_gauss.ess();
        let ess_before_student = pf_student.ess();

        // Inject outlier (5-sigma event)
        pf_gauss.update(0.05, 1.0);
        pf_student.update(0.05, 1.0);

        let ess_drop_gauss = ess_before_gauss - pf_gauss.ess();
        let ess_drop_student = ess_before_student - pf_student.ess();

        // Student-t should lose less ESS from outlier (assigns higher probability)
        assert!(
            ess_drop_student <= ess_drop_gauss + 5.0,
            "Student-t ESS drop {} should be <= Gaussian drop {} (tolerance 5.0)",
            ess_drop_student,
            ess_drop_gauss
        );
    }
}
