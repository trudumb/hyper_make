//! Particle Filter for Stochastic Volatility with Regime Switching
//!
//! This module implements a bootstrap particle filter for estimating latent
//! volatility state with regime switching. The model is:
//!
//! ```text
//! State equation:
//!   log(σ_t) = log(σ_{t-1}) + κ_r × (θ_r - log(σ_{t-1})) × Δt + ξ_r × √Δt × ε_t
//!
//! where r ∈ {Low, Normal, High, Extreme} is the regime.
//!
//! Observation equation:
//!   r_t ~ N(0, σ_t²)
//!
//! Regime transitions follow a sticky Markov chain.
//! ```
//!
//! # Algorithm
//!
//! Bootstrap particle filter with systematic resampling:
//! 1. Propagate particles via state transition (Euler-Maruyama)
//! 2. Weight particles by observation likelihood
//! 3. Resample when effective sample size (ESS) < N/2
//!
//! # Usage
//!
//! ```ignore
//! let mut filter = VolatilityFilter::new(VolFilterConfig::default());
//!
//! // Feed returns
//! filter.on_return(0.001, 0.1);  // 10 bps return, 0.1s delta
//!
//! // Get estimates
//! let sigma = filter.sigma();           // Posterior mean
//! let regime = filter.regime();         // MAP regime
//! let (lo, hi) = filter.sigma_credible_interval(0.95);
//! ```

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::volatility::VolatilityRegime;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the volatility particle filter
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VolFilterConfig {
    /// Number of particles (default: 500)
    pub n_particles: usize,

    /// Effective sample size threshold for resampling (fraction of N, default: 0.5)
    pub resampling_ess_threshold: f64,

    /// Prior probability for each regime [Low, Normal, High, Extreme]
    pub regime_prior: [f64; 4],

    /// Regime transition matrix (row = from, col = to)
    /// Rows should sum to 1.0
    pub transition_matrix: [[f64; 4]; 4],

    /// Regime-specific parameters [Low, Normal, High, Extreme]
    pub regime_params: [VolRegimeParams; 4],

    /// Initial log-volatility mean (default: ln(0.001) ≈ -6.9 for 10 bps/sqrt(s))
    pub initial_log_vol_mean: f64,

    /// Initial log-volatility std (default: 1.0)
    pub initial_log_vol_std: f64,

    /// Minimum observations before estimates are reliable
    pub min_observations: usize,
}

/// Parameters for a single volatility regime
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct VolRegimeParams {
    /// Mean reversion speed (κ) - how fast vol reverts to theta
    pub kappa: f64,

    /// Long-run mean (θ) in log-vol space
    pub theta: f64,

    /// Vol of vol (ξ) - how noisy the vol process is
    pub xi: f64,
}

impl Default for VolRegimeParams {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            theta: -6.9, // ln(0.001) ≈ 10 bps/sqrt(s)
            xi: 0.5,
        }
    }
}

impl Default for VolFilterConfig {
    fn default() -> Self {
        // Default regime parameters tuned for crypto markets
        let regime_params = [
            // Low volatility: fast reversion to low level
            VolRegimeParams {
                kappa: 2.0,
                theta: -7.6, // ~5 bps/sqrt(s)
                xi: 0.3,
            },
            // Normal volatility: moderate reversion
            VolRegimeParams {
                kappa: 1.0,
                theta: -6.9, // ~10 bps/sqrt(s)
                xi: 0.5,
            },
            // High volatility: slower reversion to higher level
            VolRegimeParams {
                kappa: 0.5,
                theta: -5.5, // ~40 bps/sqrt(s)
                xi: 0.8,
            },
            // Extreme volatility: very slow reversion, high vol-of-vol
            VolRegimeParams {
                kappa: 0.3,
                theta: -4.6, // ~100 bps/sqrt(s)
                xi: 1.2,
            },
        ];

        // Sticky diagonal transition matrix with faster adaptation
        // Higher values on diagonal = more persistent regimes
        // But not too sticky to allow regime detection
        let transition_matrix = [
            // From Low:     to Low   Normal  High   Extreme
            [0.90, 0.08, 0.019, 0.001],
            // From Normal:  to Low   Normal  High   Extreme
            [0.05, 0.88, 0.06, 0.01],
            // From High:    to Low   Normal  High   Extreme
            [0.02, 0.10, 0.82, 0.06],
            // From Extreme: to Low   Normal  High   Extreme
            [0.02, 0.08, 0.20, 0.70],
        ];

        Self {
            n_particles: 500,
            resampling_ess_threshold: 0.5,
            regime_prior: [0.15, 0.55, 0.25, 0.05],
            transition_matrix,
            regime_params,
            initial_log_vol_mean: -6.9,
            initial_log_vol_std: 1.0,
            min_observations: 20,
        }
    }
}

// ============================================================================
// Particle Structure
// ============================================================================

/// A single particle representing a possible volatility state
#[derive(Debug, Clone, Copy)]
pub struct VolParticle {
    /// Log volatility (natural log of σ in return units)
    pub log_vol: f64,

    /// Current regime
    pub regime: VolatilityRegime,
}

impl VolParticle {
    /// Get volatility in linear space
    pub fn sigma(&self) -> f64 {
        self.log_vol.exp()
    }

    /// Get volatility in basis points per sqrt(second)
    pub fn sigma_bps_per_sqrt_s(&self) -> f64 {
        self.sigma() * 10000.0
    }
}

// ============================================================================
// Volatility Filter
// ============================================================================

/// Bootstrap particle filter for stochastic volatility with regime switching
#[derive(Debug)]
pub struct VolatilityFilter {
    /// Particles representing possible states
    particles: Vec<VolParticle>,

    /// Particle weights (normalized)
    weights: Vec<f64>,

    /// Configuration
    config: VolFilterConfig,

    /// Random number generator
    rng: SmallRng,

    /// Number of observations processed
    observation_count: usize,

    /// Cached posterior statistics (updated after each observation)
    cache: FilterCache,
}

#[derive(Debug, Clone, Default)]
struct FilterCache {
    /// Posterior mean of sigma (linear space)
    sigma_mean: f64,

    /// Posterior std of sigma (linear space)
    sigma_std: f64,

    /// MAP regime estimate
    map_regime: VolatilityRegime,

    /// Regime probabilities [Low, Normal, High, Extreme]
    regime_probs: [f64; 4],

    /// 95% credible interval for sigma
    ci_95: (f64, f64),
}

impl VolatilityFilter {
    /// Create a new volatility filter with the given configuration
    pub fn new(config: VolFilterConfig) -> Self {
        let mut rng = SmallRng::from_entropy();
        let n = config.n_particles;

        // Initialize particles from prior
        let mut particles = Vec::with_capacity(n);
        for _ in 0..n {
            // Sample initial regime from prior
            let regime = sample_regime_from_probs(&config.regime_prior, &mut rng);

            // Sample initial log-vol from Gaussian prior
            let log_vol = config.initial_log_vol_mean
                + config.initial_log_vol_std * sample_standard_normal(&mut rng);

            particles.push(VolParticle { log_vol, regime });
        }

        // Uniform weights initially
        let weights = vec![1.0 / n as f64; n];

        let mut filter = Self {
            particles,
            weights,
            config,
            rng,
            observation_count: 0,
            cache: FilterCache::default(),
        };

        // Initialize cache
        filter.update_cache();

        filter
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(VolFilterConfig::default())
    }

    /// Update filter with a new return observation
    ///
    /// # Arguments
    /// * `r` - Return (e.g., 0.001 for 10 bps)
    /// * `dt` - Time delta in seconds
    pub fn on_return(&mut self, r: f64, dt: f64) {
        if dt <= 0.0 || !r.is_finite() || !dt.is_finite() {
            return;
        }

        // Step 1: Propagate particles
        self.propagate(dt);

        // Step 2: Weight particles by observation likelihood
        self.weight(r);

        // Step 3: Resample if ESS is too low
        self.resample_if_needed();

        // Update observation count and cache
        self.observation_count += 1;
        self.update_cache();
    }

    /// Propagate particles forward by dt seconds
    fn propagate(&mut self, dt: f64) {
        let sqrt_dt = dt.sqrt();

        for particle in &mut self.particles {
            // Get regime parameters
            let params = &self.config.regime_params[regime_to_index(particle.regime)];

            // Euler-Maruyama step for log-vol:
            // d(log σ) = κ(θ - log σ)dt + ξ dW
            let drift = params.kappa * (params.theta - particle.log_vol) * dt;
            let diffusion = params.xi * sqrt_dt * sample_standard_normal(&mut self.rng);
            particle.log_vol += drift + diffusion;

            // Clamp to reasonable bounds
            particle.log_vol = particle.log_vol.clamp(-12.0, 0.0); // ~0.6 bps to ~10000 bps

            // Regime transition
            let from_idx = regime_to_index(particle.regime);
            particle.regime =
                sample_regime_from_probs(&self.config.transition_matrix[from_idx], &mut self.rng);
        }
    }

    /// Weight particles by observation likelihood
    fn weight(&mut self, r: f64) {
        let r_sq = r * r;

        for (i, particle) in self.particles.iter().enumerate() {
            let sigma = particle.sigma();
            let var = sigma * sigma;

            // Gaussian log-likelihood: -0.5 * (log(2πσ²) + r²/σ²)
            let log_likelihood = -0.5 * ((2.0 * PI * var).ln() + r_sq / var);

            // Update weight (in log space for numerical stability)
            self.weights[i] = (self.weights[i].ln() + log_likelihood).exp();
        }

        // Normalize weights
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            for w in &mut self.weights {
                *w /= sum;
            }
        } else {
            // Reset to uniform if weights collapsed
            let n = self.particles.len();
            for w in &mut self.weights {
                *w = 1.0 / n as f64;
            }
        }
    }

    /// Resample particles if effective sample size is too low
    fn resample_if_needed(&mut self) {
        let ess = self.effective_sample_size();
        let threshold = self.config.resampling_ess_threshold * self.particles.len() as f64;

        if ess < threshold {
            self.systematic_resample();
        }
    }

    /// Compute effective sample size: 1 / Σw²
    fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.weights.iter().map(|w| w * w).sum();
        if sum_sq > 0.0 {
            1.0 / sum_sq
        } else {
            0.0
        }
    }

    /// Systematic resampling (low variance)
    fn systematic_resample(&mut self) {
        let n = self.particles.len();

        // Compute cumulative weights
        let mut cumsum = vec![0.0; n + 1];
        for i in 0..n {
            cumsum[i + 1] = cumsum[i] + self.weights[i];
        }

        // Generate systematic samples
        let u0: f64 = self.rng.gen::<f64>() / n as f64;
        let mut new_particles = Vec::with_capacity(n);

        let mut j = 0;
        for i in 0..n {
            let u = u0 + i as f64 / n as f64;
            while j < n && cumsum[j + 1] < u {
                j += 1;
            }
            j = j.min(n - 1);
            new_particles.push(self.particles[j]);
        }

        self.particles = new_particles;

        // Reset to uniform weights
        for w in &mut self.weights {
            *w = 1.0 / n as f64;
        }
    }

    /// Update cached statistics
    fn update_cache(&mut self) {
        let _n = self.particles.len() as f64;

        // Compute weighted mean and variance of sigma
        let mut sigma_sum = 0.0;
        let mut sigma_sq_sum = 0.0;
        let mut regime_counts = [0.0f64; 4];

        for (i, particle) in self.particles.iter().enumerate() {
            let w = self.weights[i];
            let sigma = particle.sigma();

            sigma_sum += w * sigma;
            sigma_sq_sum += w * sigma * sigma;
            regime_counts[regime_to_index(particle.regime)] += w;
        }

        self.cache.sigma_mean = sigma_sum;
        self.cache.sigma_std = (sigma_sq_sum - sigma_sum * sigma_sum).sqrt().max(0.0);
        self.cache.regime_probs = regime_counts;

        // MAP regime
        let mut max_prob = 0.0;
        let mut max_regime = VolatilityRegime::Normal;
        for (i, &prob) in regime_counts.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_regime = index_to_regime(i);
            }
        }
        self.cache.map_regime = max_regime;

        // Compute 95% credible interval via weighted quantiles
        self.cache.ci_95 = self.compute_credible_interval(0.95);
    }

    /// Compute credible interval for sigma
    fn compute_credible_interval(&self, level: f64) -> (f64, f64) {
        let alpha = (1.0 - level) / 2.0;

        // Collect (sigma, weight) pairs and sort by sigma
        let mut sorted: Vec<(f64, f64)> = self
            .particles
            .iter()
            .zip(self.weights.iter())
            .map(|(p, &w)| (p.sigma(), w))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Find quantiles
        let mut cumsum = 0.0;
        let mut lower = sorted[0].0;
        let mut upper = sorted.last().unwrap().0;

        for (sigma, w) in &sorted {
            if cumsum < alpha && cumsum + w >= alpha {
                lower = *sigma;
            }
            if cumsum < 1.0 - alpha && cumsum + w >= 1.0 - alpha {
                upper = *sigma;
            }
            cumsum += w;
        }

        (lower, upper)
    }

    // ========================================================================
    // Public Getters
    // ========================================================================

    /// Current volatility estimate (posterior mean in linear space)
    pub fn sigma(&self) -> f64 {
        self.cache.sigma_mean
    }

    /// Current volatility in basis points per sqrt(second)
    pub fn sigma_bps_per_sqrt_s(&self) -> f64 {
        self.cache.sigma_mean * 10000.0
    }

    /// Current regime (MAP estimate)
    pub fn regime(&self) -> VolatilityRegime {
        self.cache.map_regime
    }

    /// Regime probabilities [Low, Normal, High, Extreme]
    pub fn regime_probabilities(&self) -> [f64; 4] {
        self.cache.regime_probs
    }

    /// Confidence in current regime estimate (probability of MAP regime)
    pub fn regime_confidence(&self) -> f64 {
        self.cache.regime_probs[regime_to_index(self.cache.map_regime)]
    }

    /// Posterior standard deviation of sigma
    pub fn sigma_std(&self) -> f64 {
        self.cache.sigma_std
    }

    /// Credible interval for sigma at given level (e.g., 0.95 for 95%)
    pub fn sigma_credible_interval(&self, level: f64) -> (f64, f64) {
        if (level - 0.95).abs() < 0.001 {
            self.cache.ci_95
        } else {
            self.compute_credible_interval(level)
        }
    }

    /// Effective sample size (measure of particle diversity)
    pub fn ess(&self) -> f64 {
        self.effective_sample_size()
    }

    /// Number of observations processed
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Check if filter has enough observations to be reliable
    pub fn is_warmed_up(&self) -> bool {
        self.observation_count >= self.config.min_observations
    }

    /// Confidence in sigma estimate (based on ESS and observation count)
    pub fn sigma_confidence(&self) -> f64 {
        let ess_ratio = self.effective_sample_size() / self.particles.len() as f64;
        let obs_ratio = (self.observation_count as f64 / self.config.min_observations as f64).min(1.0);
        ess_ratio * obs_ratio
    }

    /// Probability of regime change within horizon_ms milliseconds
    pub fn regime_change_probability(&self, horizon_ms: u64) -> f64 {
        let current_regime = self.cache.map_regime;
        let current_idx = regime_to_index(current_regime);

        // Approximate: use transition matrix raised to power based on horizon
        // For simplicity, assume ~100ms per transition step
        let steps = (horizon_ms / 100).max(1) as usize;

        // Probability of staying in current regime after n steps ≈ p^n
        let stay_prob = self.config.transition_matrix[current_idx][current_idx].powi(steps as i32);

        1.0 - stay_prob
    }

    /// Reset the filter to initial state
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone());
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn regime_to_index(regime: VolatilityRegime) -> usize {
    match regime {
        VolatilityRegime::Low => 0,
        VolatilityRegime::Normal => 1,
        VolatilityRegime::High => 2,
        VolatilityRegime::Extreme => 3,
    }
}

fn index_to_regime(index: usize) -> VolatilityRegime {
    match index {
        0 => VolatilityRegime::Low,
        1 => VolatilityRegime::Normal,
        2 => VolatilityRegime::High,
        _ => VolatilityRegime::Extreme,
    }
}

fn sample_regime_from_probs<R: Rng>(probs: &[f64; 4], rng: &mut R) -> VolatilityRegime {
    let u: f64 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return index_to_regime(i);
        }
    }
    VolatilityRegime::Normal // Fallback
}

fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    // Box-Muller transform
    let u1: f64 = rng.gen::<f64>().max(1e-10);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_initialization() {
        let filter = VolatilityFilter::default_config();
        assert_eq!(filter.particles.len(), 500);
        assert_eq!(filter.weights.len(), 500);
        assert!((filter.weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(!filter.is_warmed_up());
    }

    #[test]
    fn test_filter_warmup() {
        let mut filter = VolatilityFilter::default_config();
        assert!(!filter.is_warmed_up());

        // Feed observations
        for _ in 0..25 {
            filter.on_return(0.0001, 0.1); // Small return
        }

        assert!(filter.is_warmed_up());
    }

    #[test]
    fn test_filter_responds_to_returns() {
        let mut filter = VolatilityFilter::default_config();

        // Warm up with small returns
        for _ in 0..30 {
            filter.on_return(0.0001, 0.1);
        }
        let sigma_calm = filter.sigma();

        // Feed large returns
        for _ in 0..30 {
            filter.on_return(0.01, 0.1); // 100 bps returns
        }
        let sigma_volatile = filter.sigma();

        // Volatility should increase
        assert!(
            sigma_volatile > sigma_calm * 1.5,
            "Expected sigma to increase after large returns: {} vs {}",
            sigma_volatile,
            sigma_calm
        );
    }

    #[test]
    fn test_regime_detection_calm() {
        let mut filter = VolatilityFilter::default_config();

        // Feed consistently small returns (many observations to let regime adapt)
        for _ in 0..200 {
            filter.on_return(0.00005, 0.1); // 0.5 bps returns
        }

        // Volatility estimate should be low (< 20 bps/sqrt(s))
        let sigma_bps = filter.sigma_bps_per_sqrt_s();
        assert!(
            sigma_bps < 20.0,
            "Expected low volatility after calm period, got {} bps/sqrt(s)",
            sigma_bps
        );

        // Regime should not be Extreme after consistent calm returns
        let regime = filter.regime();
        assert!(
            regime != VolatilityRegime::Extreme,
            "Did not expect Extreme regime after calm period, got {:?}",
            regime
        );
    }

    #[test]
    fn test_regime_detection_volatile() {
        let mut filter = VolatilityFilter::default_config();

        // Feed consistently large returns
        for _ in 0..100 {
            filter.on_return(0.02, 0.1); // 200 bps returns
        }

        // Should be in High or Extreme regime
        let regime = filter.regime();
        assert!(
            regime == VolatilityRegime::High || regime == VolatilityRegime::Extreme,
            "Expected High or Extreme regime, got {:?}",
            regime
        );
    }

    #[test]
    fn test_credible_interval() {
        let mut filter = VolatilityFilter::default_config();

        // Warm up
        for _ in 0..50 {
            filter.on_return(0.001, 0.1);
        }

        let (lo, hi) = filter.sigma_credible_interval(0.95);
        let mean = filter.sigma();

        // Mean should be within CI
        assert!(lo <= mean && mean <= hi, "Mean {} not in CI [{}, {}]", mean, lo, hi);

        // CI should be reasonable (not infinite)
        assert!(lo > 0.0);
        assert!(hi < 1.0); // Less than 10000 bps
    }

    #[test]
    fn test_ess_after_resampling() {
        let mut filter = VolatilityFilter::default_config();

        // Feed returns to trigger resampling
        for _ in 0..100 {
            filter.on_return(0.001, 0.1);
        }

        // ESS should be reasonable after resampling
        let ess = filter.ess();
        assert!(
            ess > 100.0,
            "ESS {} should be reasonable after observations",
            ess
        );
    }

    #[test]
    fn test_regime_change_probability() {
        let filter = VolatilityFilter::default_config();

        // Longer horizon should have higher change probability
        let p_100ms = filter.regime_change_probability(100);
        let p_1000ms = filter.regime_change_probability(1000);

        assert!(
            p_1000ms > p_100ms,
            "Longer horizon should have higher change prob: {} vs {}",
            p_1000ms,
            p_100ms
        );
    }

    #[test]
    fn test_sigma_confidence() {
        let mut filter = VolatilityFilter::default_config();

        // Initially low confidence
        let conf_initial = filter.sigma_confidence();
        assert!(conf_initial < 0.5, "Initial confidence should be low");

        // After warmup, higher confidence
        for _ in 0..50 {
            filter.on_return(0.001, 0.1);
        }
        let conf_warmed = filter.sigma_confidence();
        assert!(
            conf_warmed > conf_initial,
            "Warmed up confidence should be higher"
        );
    }

    #[test]
    fn test_reset() {
        let mut filter = VolatilityFilter::default_config();

        // Feed some observations
        for _ in 0..50 {
            filter.on_return(0.01, 0.1);
        }
        assert!(filter.is_warmed_up());

        // Reset
        filter.reset();

        // Should be back to initial state
        assert!(!filter.is_warmed_up());
        assert_eq!(filter.observation_count(), 0);
    }
}
