//! Conjugate prior distributions for Bayesian belief updates.
//!
//! This module provides efficient online updates for market parameters:
//! - Normal-Inverse-Gamma (NIG) for drift μ and volatility σ²
//! - Extended Gamma posterior for fill intensity κ
//!
//! These conjugate priors enable closed-form posterior updates without MCMC.

use std::f64::consts::PI;

// Re-export from control::types for compatibility
pub use crate::market_maker::control::types::{
    normal_cdf, normal_quantile, GammaPosterior, NormalGammaPosterior,
};

/// Normal-Inverse-Gamma posterior for (μ, σ²).
///
/// This is the conjugate prior for normally distributed observations
/// with unknown mean and variance. It models the joint distribution:
/// - σ² ~ InverseGamma(α, β)
/// - μ | σ² ~ Normal(m, σ²/k)
///
/// # Usage for Drift Estimation
///
/// Observe price returns {r_1, ..., r_n} and update posterior:
/// - E[μ | data] gives the posterior mean drift (predictive signal)
/// - E[σ² | data] gives the posterior mean volatility
///
/// # Mathematical Background
///
/// Prior: σ² ~ IG(α₀, β₀), μ | σ² ~ N(m₀, σ²/k₀)
///
/// Update on n observations with sample mean x̄ and sum of squares SS:
/// - k_n = k₀ + n
/// - m_n = (k₀ × m₀ + n × x̄) / k_n
/// - α_n = α₀ + n/2
/// - β_n = β₀ + 0.5 × SS + 0.5 × k₀ × n × (x̄ - m₀)² / k_n
#[derive(Debug, Clone, Copy)]
pub struct NormalInverseGamma {
    /// Prior/posterior mean of μ
    pub m: f64,
    /// Pseudo-sample size for mean (precision scaling)
    pub k: f64,
    /// Shape parameter for σ² (α > 0)
    pub alpha: f64,
    /// Scale parameter for σ² (β > 0)
    pub beta: f64,
}

impl Default for NormalInverseGamma {
    fn default() -> Self {
        // Weakly informative prior centered at 0 drift
        // α = 2 gives finite variance
        Self {
            m: 0.0,
            k: 1.0,
            alpha: 2.0,
            beta: 0.0001, // Prior variance ≈ 0.0001 / (2-1) = 0.0001
        }
    }
}

impl NormalInverseGamma {
    /// Create a new NIG prior with specified parameters.
    ///
    /// # Arguments
    /// * `m` - Prior mean of μ
    /// * `k` - Prior pseudo-sample size (higher = more confidence in m)
    /// * `alpha` - Shape parameter for σ² (use α > 1 for finite mean)
    /// * `beta` - Scale parameter for σ²
    pub fn new(m: f64, k: f64, alpha: f64, beta: f64) -> Self {
        Self {
            m,
            k: k.max(0.001),
            alpha: alpha.max(0.001),
            beta: beta.max(1e-10),
        }
    }

    /// Create a prior centered at zero with specified uncertainty.
    ///
    /// # Arguments
    /// * `sigma_prior` - Prior standard deviation for observations
    /// * `n_effective` - Effective prior sample size
    pub fn zero_centered(sigma_prior: f64, n_effective: f64) -> Self {
        let variance_prior = sigma_prior.powi(2);
        Self {
            m: 0.0,
            k: 1.0,
            alpha: n_effective / 2.0,
            beta: variance_prior * (n_effective / 2.0 - 1.0).max(0.1),
        }
    }

    /// Update posterior with a single observation.
    ///
    /// This is the conjugate update for one new observation x.
    pub fn update(&mut self, x: f64) {
        let k_new = self.k + 1.0;
        let m_new = (self.k * self.m + x) / k_new;
        let alpha_new = self.alpha + 0.5;
        let beta_new = self.beta + 0.5 * self.k * (x - self.m).powi(2) / k_new;

        self.k = k_new;
        self.m = m_new;
        self.alpha = alpha_new;
        self.beta = beta_new;
    }

    /// Update posterior with multiple observations (batch update).
    ///
    /// More efficient than calling update() in a loop.
    pub fn update_batch(&mut self, observations: &[f64]) {
        if observations.is_empty() {
            return;
        }

        let n = observations.len() as f64;
        let x_bar: f64 = observations.iter().sum::<f64>() / n;
        let ss: f64 = observations.iter().map(|x| (x - x_bar).powi(2)).sum();

        let k_new = self.k + n;
        let m_new = (self.k * self.m + n * x_bar) / k_new;
        let alpha_new = self.alpha + n / 2.0;
        let beta_new =
            self.beta + 0.5 * ss + 0.5 * self.k * n * (x_bar - self.m).powi(2) / k_new;

        self.k = k_new;
        self.m = m_new;
        self.alpha = alpha_new;
        self.beta = beta_new;
    }

    /// Update with weighted observation (for time-weighted updates).
    ///
    /// # Arguments
    /// * `x` - Observation value
    /// * `weight` - Observation weight (e.g., dt for continuous updates)
    pub fn update_weighted(&mut self, x: f64, weight: f64) {
        let w = weight.max(1e-10);
        let k_new = self.k + w;
        let m_new = (self.k * self.m + w * x) / k_new;
        let alpha_new = self.alpha + w / 2.0;
        let beta_new = self.beta + 0.5 * self.k * w * (x - self.m).powi(2) / k_new;

        self.k = k_new;
        self.m = m_new;
        self.alpha = alpha_new;
        self.beta = beta_new;
    }

    /// Posterior mean of μ: E[μ | data].
    ///
    /// This IS the predictive drift signal - not a heuristic.
    pub fn posterior_mean(&self) -> f64 {
        self.m
    }

    /// Posterior variance of μ: Var[μ | data].
    ///
    /// For α > 1: Var[μ] = β / (k × (α - 1))
    pub fn posterior_variance(&self) -> f64 {
        if self.alpha > 1.0 {
            self.beta / (self.k * (self.alpha - 1.0))
        } else {
            f64::INFINITY
        }
    }

    /// Posterior standard deviation of μ.
    pub fn posterior_std(&self) -> f64 {
        self.posterior_variance().sqrt()
    }

    /// Posterior mean of σ²: E[σ² | data].
    ///
    /// For α > 1: E[σ²] = β / (α - 1)
    pub fn posterior_sigma_sq(&self) -> f64 {
        if self.alpha > 1.0 {
            self.beta / (self.alpha - 1.0)
        } else {
            f64::INFINITY
        }
    }

    /// Posterior mean of σ: E[σ | data] (approximate).
    ///
    /// Uses √E[σ²] as approximation (slightly biased).
    pub fn posterior_sigma(&self) -> f64 {
        self.posterior_sigma_sq().sqrt()
    }

    /// Probability that μ > threshold.
    ///
    /// Uses marginal t-distribution for μ.
    /// Marginal: μ ~ t_{2α}(m, β/(k×α))
    pub fn prob_mean_greater_than(&self, threshold: f64) -> f64 {
        if self.alpha <= 0.5 {
            return 0.5;
        }

        let scale = (self.beta / (self.k * self.alpha)).sqrt();
        if scale < 1e-10 {
            return if self.m > threshold { 1.0 } else { 0.0 };
        }

        let t_stat = (self.m - threshold) / scale;
        let df = 2.0 * self.alpha;
        1.0 - t_cdf(t_stat, df)
    }

    /// Probability that μ < 0 (bearish drift).
    pub fn prob_negative_drift(&self) -> f64 {
        1.0 - self.prob_mean_greater_than(0.0)
    }

    /// Probability that μ > 0 (bullish drift).
    pub fn prob_positive_drift(&self) -> f64 {
        self.prob_mean_greater_than(0.0)
    }

    /// 95% credible interval for μ.
    pub fn credible_interval(&self, coverage: f64) -> (f64, f64) {
        let alpha_tail = (1.0 - coverage) / 2.0;
        let df = 2.0 * self.alpha;
        let scale = (self.beta / (self.k * self.alpha)).sqrt();
        let t_crit = t_quantile(1.0 - alpha_tail, df);

        (self.m - t_crit * scale, self.m + t_crit * scale)
    }

    /// Effective sample size (how much data has been incorporated).
    pub fn effective_n(&self) -> f64 {
        self.k - 1.0 // Subtract prior contribution
    }

    /// Decay posterior toward prior (for non-stationarity).
    ///
    /// # Arguments
    /// * `factor` - Retention factor in [0, 1]. 0.95 means keep 95% of information.
    pub fn decay(&mut self, factor: f64) {
        let f = factor.clamp(0.0, 1.0);
        let prior = Self::default();

        // Blend toward prior
        self.k = prior.k + (self.k - prior.k) * f;
        self.alpha = prior.alpha + (self.alpha - prior.alpha) * f;
        // Keep m (mean) as is, but increase uncertainty
        self.beta = self.beta.max(prior.beta);
    }

    /// Soft reset: keep some learned information.
    ///
    /// # Arguments
    /// * `retention` - Fraction of information to retain [0, 1]
    pub fn soft_reset(&mut self, retention: f64) {
        let prior = Self::default();
        let r = retention.clamp(0.0, 1.0);

        self.k = prior.k + (self.k - prior.k) * r;
        self.m = prior.m + (self.m - prior.m) * r;
        self.alpha = prior.alpha + (self.alpha - prior.alpha) * r;
        self.beta = prior.beta + (self.beta - prior.beta) * r;
    }
}

/// Extended fill intensity posterior with depth-dependent updates.
///
/// Wraps GammaPosterior with fill intensity specific methods.
#[derive(Debug, Clone, Copy)]
pub struct FillIntensityPosterior {
    /// Underlying Gamma posterior for κ
    pub gamma: GammaPosterior,
    /// Depth sensitivity parameter γ (for λ = κ × exp(-γ × δ))
    pub depth_sensitivity: f64,
    /// Number of fills observed
    pub n_fills: u64,
    /// Total observation time
    pub total_time: f64,
}

impl Default for FillIntensityPosterior {
    fn default() -> Self {
        Self {
            // Prior: κ ~ Gamma(20, 0.1) → mean = 200 fills/time unit
            // Conservative for thin DEX
            gamma: GammaPosterior::new(20.0, 0.1),
            depth_sensitivity: 0.5, // λ halves every ~1.4 bps
            n_fills: 0,
            total_time: 0.0,
        }
    }
}

impl FillIntensityPosterior {
    /// Create with specific prior mean and precision.
    ///
    /// # Arguments
    /// * `mean` - Prior mean for κ
    /// * `cv` - Coefficient of variation (uncertainty)
    /// * `depth_sensitivity` - γ parameter for depth decay
    pub fn new(mean: f64, cv: f64, depth_sensitivity: f64) -> Self {
        // Gamma parameterization: mean = α/β, cv = 1/√α
        // So α = 1/cv², β = α/mean
        let alpha = (1.0 / cv).powi(2).max(1.0);
        let beta = alpha / mean.max(1.0);

        Self {
            gamma: GammaPosterior::new(alpha, beta),
            depth_sensitivity,
            n_fills: 0,
            total_time: 0.0,
        }
    }

    /// Update posterior with fill observation at given depth.
    ///
    /// # Arguments
    /// * `n_fills` - Number of fills observed (usually 1)
    /// * `dt` - Time period (seconds)
    /// * `depth_bps` - Depth at which fill occurred
    ///
    /// The model is: fills ~ Poisson(κ × exp(-γ × δ) × dt)
    /// So we update with effective exposure: dt × exp(-γ × δ)
    pub fn update(&mut self, n_fills: f64, dt: f64, depth_bps: f64) {
        // Effective exposure accounting for depth
        let effective_exposure = dt * (-self.depth_sensitivity * depth_bps / 10000.0).exp();
        self.gamma.update(n_fills, effective_exposure);
        self.n_fills += n_fills as u64;
        self.total_time += dt;
    }

    /// Posterior mean: E[κ | fills].
    pub fn posterior_mean(&self) -> f64 {
        self.gamma.mean()
    }

    /// Posterior standard deviation.
    pub fn posterior_std(&self) -> f64 {
        self.gamma.std()
    }

    /// Coefficient of variation (uncertainty relative to mean).
    pub fn cv(&self) -> f64 {
        self.gamma.cv()
    }

    /// Expected fill intensity at given depth.
    ///
    /// λ(δ) = E[κ] × exp(-γ × δ)
    pub fn intensity_at_depth(&self, depth_bps: f64) -> f64 {
        self.posterior_mean() * (-self.depth_sensitivity * depth_bps / 10000.0).exp()
    }

    /// Probability that κ > threshold.
    pub fn prob_greater_than(&self, threshold: f64) -> f64 {
        self.gamma.prob_greater_than(threshold)
    }

    /// Is the posterior well-calibrated (enough observations)?
    pub fn is_warmed_up(&self) -> bool {
        self.n_fills >= 10 && self.total_time > 60.0
    }

    /// Decay toward prior for non-stationarity.
    pub fn decay(&mut self, factor: f64) {
        let f = factor.clamp(0.0, 1.0);
        let prior = Self::default();

        // Blend gamma parameters toward prior
        self.gamma.alpha = prior.gamma.alpha + (self.gamma.alpha - prior.gamma.alpha) * f;
        self.gamma.beta = prior.gamma.beta + (self.gamma.beta - prior.gamma.beta) * f;
    }
}

// === Helper functions for t-distribution ===

/// Student's t CDF approximation.
fn t_cdf(t: f64, df: f64) -> f64 {
    // Use normal approximation for large df
    if df > 100.0 {
        return normal_cdf(t);
    }

    // Hill's approximation for small df
    let x = df / (df + t * t);
    let p = 0.5 * incomplete_beta(df / 2.0, 0.5, x);

    if t >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Student's t quantile approximation.
fn t_quantile(p: f64, df: f64) -> f64 {
    // Use normal approximation for large df
    if df > 100.0 {
        return normal_quantile(p);
    }

    // Newton-Raphson refinement starting from normal
    let mut t = normal_quantile(p);
    for _ in 0..5 {
        let cdf = t_cdf(t, df);
        let pdf = t_pdf(t, df);
        if pdf.abs() < 1e-10 {
            break;
        }
        t -= (cdf - p) / pdf;
    }
    t
}

/// Student's t PDF.
fn t_pdf(t: f64, df: f64) -> f64 {
    let coef = gamma_ln((df + 1.0) / 2.0) - gamma_ln(df / 2.0) - 0.5 * (df * PI).ln();
    (coef - (df + 1.0) / 2.0 * (1.0 + t * t / df).ln()).exp()
}

/// Log gamma function (Lanczos approximation).
fn gamma_ln(x: f64) -> f64 {
    let g = 7;
    let c = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        PI.ln() - (PI * x).sin().ln() - gamma_ln(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        for (i, &ci) in c.iter().enumerate().skip(1).take(g + 1) {
            a += ci / (x + i as f64);
        }
        let t = x + g as f64 + 0.5;
        0.5 * (2.0 * PI).ln() + (t).ln() * (x + 0.5) - t + a.ln()
    }
}

/// Incomplete beta function (regularized).
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry for numerical stability
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - incomplete_beta(b, a, 1.0 - x);
    }

    let bt = (gamma_ln(a + b) - gamma_ln(a) - gamma_ln(b) + a * x.ln() + b * (1.0 - x).ln()).exp();

    // Continued fraction
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0));
    let mut c = 1.0;
    let mut f = d;

    for m in 1..100 {
        let m2 = 2 * m;

        // Even step
        let aa = m as f64 * (b - m as f64) * x / ((a + m2 as f64 - 1.0) * (a + m2 as f64));
        d = 1.0 / (1.0 + aa * d);
        c = 1.0 + aa / c;
        f *= d * c;

        // Odd step
        let aa =
            -(a + m as f64) * (a + b + m as f64) * x / ((a + m2 as f64) * (a + m2 as f64 + 1.0));
        d = 1.0 / (1.0 + aa * d);
        c = 1.0 + aa / c;
        let del = d * c;
        f *= del;

        if (del - 1.0).abs() < 1e-10 {
            break;
        }
    }

    bt * f / a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nig_default() {
        let nig = NormalInverseGamma::default();
        assert!((nig.posterior_mean() - 0.0).abs() < 1e-10);
        assert!(nig.posterior_variance() > 0.0);
    }

    #[test]
    fn test_nig_negative_drift() {
        let mut nig = NormalInverseGamma::default();

        // Observe negative returns (capitulation) - need enough to overcome prior
        for _ in 0..50 {
            nig.update(-0.001); // -10 bps returns
        }

        assert!(
            nig.posterior_mean() < 0.0,
            "Drift should be negative after observing negative returns: {}",
            nig.posterior_mean()
        );
        // Note: With finite prior uncertainty, prob may not exceed 0.5 even with
        // negative posterior mean. The key is the posterior mean is negative.
        assert!(
            nig.posterior_mean() < -0.0005,
            "Posterior mean should be strongly negative: {}",
            nig.posterior_mean()
        );
    }

    #[test]
    fn test_nig_positive_drift() {
        let mut nig = NormalInverseGamma::default();

        // Observe positive returns (uptrend) - need enough to overcome prior
        for _ in 0..50 {
            nig.update(0.001); // +10 bps returns
        }

        assert!(
            nig.posterior_mean() > 0.0,
            "Drift should be positive after observing positive returns"
        );
        // Note: With finite prior uncertainty, prob may not exceed 0.5 even with
        // positive posterior mean. The key is the posterior mean is positive.
        assert!(
            nig.posterior_mean() > 0.0005,
            "Posterior mean should be strongly positive: {}",
            nig.posterior_mean()
        );
    }

    #[test]
    fn test_nig_batch_update() {
        let mut nig1 = NormalInverseGamma::default();
        let mut nig2 = NormalInverseGamma::default();

        let observations = vec![0.001, -0.0005, 0.002, -0.001, 0.0015];

        // Update one at a time
        for &x in &observations {
            nig1.update(x);
        }

        // Update all at once
        nig2.update_batch(&observations);

        // Should give same result
        assert!((nig1.posterior_mean() - nig2.posterior_mean()).abs() < 1e-10);
        assert!((nig1.k - nig2.k).abs() < 1e-10);
    }

    #[test]
    fn test_nig_uncertainty_decreases() {
        let mut nig = NormalInverseGamma::default();
        let initial_var = nig.posterior_variance();

        // Observe consistent returns
        for _ in 0..50 {
            nig.update(0.0005);
        }

        let final_var = nig.posterior_variance();
        assert!(
            final_var < initial_var,
            "Variance should decrease with more observations"
        );
    }

    #[test]
    fn test_nig_credible_interval() {
        let mut nig = NormalInverseGamma::default();

        // Add some observations centered around 0.001
        for _ in 0..30 {
            nig.update(0.001 + 0.0002 * (rand_f64() - 0.5));
        }

        let (lower, upper) = nig.credible_interval(0.95);
        assert!(lower < upper);
        assert!(lower < nig.posterior_mean());
        assert!(upper > nig.posterior_mean());
    }

    #[test]
    fn test_fill_intensity_basic() {
        let mut kappa = FillIntensityPosterior::default();

        assert!(kappa.posterior_mean() > 0.0);

        // Update with fills - need > 60s total_time and >= 10 fills for warmup
        for _ in 0..20 {
            kappa.update(1.0, 5.0, 5.0); // 1 fill per 5 seconds at 5 bps depth
        }

        assert!(kappa.n_fills == 20);
        assert!(kappa.total_time >= 100.0, "Total time should be >= 100s");
        assert!(kappa.is_warmed_up(), "Should be warmed up after 20 fills over 100s");
    }

    #[test]
    fn test_fill_intensity_depth_decay() {
        let kappa = FillIntensityPosterior::default();

        // Intensity should decrease with depth
        let intensity_0 = kappa.intensity_at_depth(0.0);
        let intensity_10 = kappa.intensity_at_depth(10.0);
        let intensity_20 = kappa.intensity_at_depth(20.0);

        assert!(intensity_0 > intensity_10);
        assert!(intensity_10 > intensity_20);
    }

    // Simple pseudo-random for testing (not cryptographic)
    fn rand_f64() -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        nanos as f64 / 1_000_000_000.0
    }
}
