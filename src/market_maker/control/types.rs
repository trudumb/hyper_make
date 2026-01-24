//! Probability distribution types for Bayesian belief tracking.
//!
//! These conjugate priors enable efficient online updates from trading outcomes.

use std::f64::consts::PI;

/// Gamma distribution posterior for rate parameters (λ, fill rates).
///
/// Conjugate prior for Poisson observations (fill counts).
/// Gamma(α, β) has mean α/β and variance α/β².
#[derive(Debug, Clone, Copy)]
pub struct GammaPosterior {
    /// Shape parameter (prior fills + observed fills)
    pub alpha: f64,
    /// Rate parameter (prior time + observed time)
    pub beta: f64,
}

impl Default for GammaPosterior {
    fn default() -> Self {
        // Weakly informative prior: α=2, β=1 → mean=2, var=2
        Self {
            alpha: 2.0,
            beta: 1.0,
        }
    }
}

impl GammaPosterior {
    /// Create with specific parameters.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// Posterior mean: E[λ] = α/β
    pub fn mean(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Posterior variance: Var[λ] = α/β²
    pub fn variance(&self) -> f64 {
        self.alpha / (self.beta * self.beta)
    }

    /// Posterior standard deviation
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Coefficient of variation (uncertainty relative to mean)
    pub fn cv(&self) -> f64 {
        1.0 / self.alpha.sqrt()
    }

    /// Posterior mode: (α-1)/β for α > 1
    pub fn mode(&self) -> f64 {
        if self.alpha > 1.0 {
            (self.alpha - 1.0) / self.beta
        } else {
            0.0
        }
    }

    /// Update with Poisson observation: n events in time t.
    ///
    /// Posterior: Gamma(α + n, β + t)
    pub fn update(&mut self, n_events: f64, time_elapsed: f64) {
        self.alpha += n_events;
        self.beta += time_elapsed;
    }

    /// Credible interval [lower, upper] for given coverage.
    pub fn credible_interval(&self, coverage: f64) -> (f64, f64) {
        // Use gamma quantile function approximation
        let p_lower = (1.0 - coverage) / 2.0;
        let p_upper = 1.0 - p_lower;

        // Wilson-Hilferty approximation for gamma quantiles
        let lower = self.gamma_quantile(p_lower);
        let upper = self.gamma_quantile(p_upper);

        (lower, upper)
    }

    /// Approximate gamma quantile using Wilson-Hilferty transformation.
    fn gamma_quantile(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        // Normal quantile (Beasley-Springer-Moro approximation)
        let z = normal_quantile(p);

        // Wilson-Hilferty: X ~ Gamma(α, β) ≈ β * (1 + z*√(1/(9α)) - 1/(9α))³ * α
        let term = 1.0 / (9.0 * self.alpha);
        let cube = (1.0 - term + z * term.sqrt()).powi(3);
        (self.alpha * cube / self.beta).max(0.0)
    }

    /// Probability that λ > threshold.
    pub fn prob_greater_than(&self, threshold: f64) -> f64 {
        1.0 - self.cdf(threshold)
    }

    /// CDF at x using incomplete gamma function approximation.
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        // Regularized incomplete gamma function approximation
        let z = self.beta * x;
        incomplete_gamma(self.alpha, z)
    }
}

/// Normal-Gamma posterior for mean and precision of normal distribution.
///
/// Conjugate prior for normal observations with unknown mean and variance.
/// Models the joint distribution of (μ, τ) where τ = 1/σ².
#[derive(Debug, Clone, Copy)]
pub struct NormalGammaPosterior {
    /// Posterior mean location
    pub mu: f64,
    /// Pseudo-observations for mean
    pub kappa: f64,
    /// Shape for precision
    pub alpha: f64,
    /// Scale for precision
    pub beta: f64,
}

impl Default for NormalGammaPosterior {
    fn default() -> Self {
        // Weakly informative: μ~0, broad uncertainty
        Self {
            mu: 0.0,
            kappa: 1.0,
            alpha: 2.0,
            beta: 1.0,
        }
    }
}

impl NormalGammaPosterior {
    /// Create with specific parameters.
    pub fn new(mu: f64, kappa: f64, alpha: f64, beta: f64) -> Self {
        Self {
            mu,
            kappa,
            alpha,
            beta,
        }
    }

    /// Posterior mean of μ: E[μ] = μ
    pub fn mean_of_mean(&self) -> f64 {
        self.mu
    }

    /// Posterior mean of precision: E[τ] = α/β
    pub fn mean_precision(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Posterior mean of variance: E[σ²] ≈ β/(α-1) for α > 1
    pub fn mean_variance(&self) -> f64 {
        if self.alpha > 1.0 {
            self.beta / (self.alpha - 1.0)
        } else {
            f64::INFINITY
        }
    }

    /// Posterior std of the mean (marginal t-distribution).
    pub fn std_of_mean(&self) -> f64 {
        // Marginal is t-distribution with 2α degrees of freedom
        // Variance = β/(α*κ) * (α/(α-1)) for α > 1
        if self.alpha > 1.0 {
            let variance =
                self.beta / (self.alpha * self.kappa) * (self.alpha / (self.alpha - 1.0));
            variance.sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Update with a single observation x.
    pub fn update(&mut self, x: f64) {
        let kappa_new = self.kappa + 1.0;
        let mu_new = (self.kappa * self.mu + x) / kappa_new;
        let alpha_new = self.alpha + 0.5;
        let beta_new = self.beta + 0.5 * self.kappa * (x - self.mu).powi(2) / kappa_new;

        self.mu = mu_new;
        self.kappa = kappa_new;
        self.alpha = alpha_new;
        self.beta = beta_new;
    }

    /// Update with multiple observations (batch update).
    pub fn update_batch(&mut self, observations: &[f64]) {
        if observations.is_empty() {
            return;
        }

        let n = observations.len() as f64;
        let x_bar: f64 = observations.iter().sum::<f64>() / n;
        let ss: f64 = observations.iter().map(|x| (x - x_bar).powi(2)).sum();

        let kappa_new = self.kappa + n;
        let mu_new = (self.kappa * self.mu + n * x_bar) / kappa_new;
        let alpha_new = self.alpha + n / 2.0;
        let beta_new =
            self.beta + 0.5 * ss + 0.5 * self.kappa * n * (x_bar - self.mu).powi(2) / kappa_new;

        self.mu = mu_new;
        self.kappa = kappa_new;
        self.alpha = alpha_new;
        self.beta = beta_new;
    }

    /// Probability that mean > threshold (using marginal t-distribution).
    pub fn prob_mean_greater_than(&self, threshold: f64) -> f64 {
        if self.alpha <= 0.5 {
            return 0.5; // Improper posterior
        }

        let t_stat = (self.mu - threshold) / self.std_of_mean();
        let df = 2.0 * self.alpha;
        1.0 - t_cdf(t_stat, df)
    }

    /// Credible interval for the mean.
    pub fn credible_interval(&self, coverage: f64) -> (f64, f64) {
        let alpha_tail = (1.0 - coverage) / 2.0;
        let df = 2.0 * self.alpha;
        let t_crit = t_quantile(1.0 - alpha_tail, df);
        let se = self.std_of_mean();

        (self.mu - t_crit * se, self.mu + t_crit * se)
    }
}

/// Dirichlet posterior for categorical/multinomial parameters.
///
/// Conjugate prior for categorical observations (model selection).
/// Dirichlet(α₁, ..., αₖ) has mean αᵢ / Σαⱼ.
#[derive(Debug, Clone)]
pub struct DirichletPosterior {
    /// Concentration parameters (prior + observed counts)
    pub alpha: Vec<f64>,
}

impl Default for DirichletPosterior {
    fn default() -> Self {
        // Uniform prior over 3 categories
        Self::uniform(3)
    }
}

impl DirichletPosterior {
    /// Create uniform prior with k categories.
    pub fn uniform(k: usize) -> Self {
        Self {
            alpha: vec![1.0; k],
        }
    }

    /// Create with specific concentrations.
    pub fn new(alpha: Vec<f64>) -> Self {
        Self { alpha }
    }

    /// Number of categories.
    pub fn k(&self) -> usize {
        self.alpha.len()
    }

    /// Posterior mean for category i: E[πᵢ] = αᵢ / Σαⱼ
    pub fn mean(&self, i: usize) -> f64 {
        if i >= self.alpha.len() {
            return 0.0;
        }
        let sum: f64 = self.alpha.iter().sum();
        self.alpha[i] / sum
    }

    /// All posterior means.
    pub fn means(&self) -> Vec<f64> {
        let sum: f64 = self.alpha.iter().sum();
        self.alpha.iter().map(|a| a / sum).collect()
    }

    /// Posterior mode (MAP estimate) for category i.
    pub fn mode(&self, i: usize) -> f64 {
        if i >= self.alpha.len() {
            return 0.0;
        }
        let sum: f64 = self.alpha.iter().sum();
        let k = self.alpha.len() as f64;
        ((self.alpha[i] - 1.0) / (sum - k)).max(0.0)
    }

    /// Variance of posterior mean for category i.
    pub fn variance(&self, i: usize) -> f64 {
        if i >= self.alpha.len() {
            return 0.0;
        }
        let sum: f64 = self.alpha.iter().sum();
        let mean_i = self.alpha[i] / sum;
        mean_i * (1.0 - mean_i) / (sum + 1.0)
    }

    /// Entropy of the posterior mean (measure of uncertainty).
    pub fn entropy(&self) -> f64 {
        let means = self.means();
        -means
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Update with observation of category i.
    pub fn update(&mut self, i: usize) {
        if i < self.alpha.len() {
            self.alpha[i] += 1.0;
        }
    }

    /// Update with weighted observation.
    pub fn update_weighted(&mut self, i: usize, weight: f64) {
        if i < self.alpha.len() {
            self.alpha[i] += weight;
        }
    }

    /// Update with counts for each category.
    pub fn update_batch(&mut self, counts: &[f64]) {
        for (i, &count) in counts.iter().enumerate() {
            if i < self.alpha.len() {
                self.alpha[i] += count;
            }
        }
    }

    /// Most likely category (highest posterior mean).
    pub fn argmax(&self) -> usize {
        self.alpha
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Probability that category i is the best (highest true probability).
    /// Uses closed-form approximation (n_samples parameter reserved for future Monte Carlo).
    pub fn prob_best(&self, i: usize, _n_samples: usize) -> f64 {
        if i >= self.alpha.len() {
            return 0.0;
        }

        // Simple approximation using posterior means and variances
        let means = self.means();
        let my_mean = means[i];

        // Count how many times we're likely to be best
        let mut wins = 0.0;
        for (j, &other_mean) in means.iter().enumerate() {
            if j == i {
                continue;
            }
            // Approximate probability we beat this category
            let diff = my_mean - other_mean;
            let var_sum = self.variance(i) + self.variance(j);
            if var_sum > 0.0 {
                let z = diff / var_sum.sqrt();
                wins += normal_cdf(z);
            } else if diff > 0.0 {
                wins += 1.0;
            }
        }

        // Normalize to probability
        (wins / (self.alpha.len() - 1) as f64).min(1.0)
    }

    /// Decay all counts (for non-stationarity).
    pub fn decay(&mut self, factor: f64) {
        for a in &mut self.alpha {
            // Keep at least 1.0 (uniform prior)
            *a = (*a * factor).max(1.0);
        }
    }
}

/// Discrete probability distribution over N states.
#[derive(Debug, Clone)]
pub struct DiscreteDistribution<const N: usize> {
    /// Probabilities (must sum to 1)
    pub probs: [f64; N],
}

impl<const N: usize> Default for DiscreteDistribution<N> {
    fn default() -> Self {
        let p = 1.0 / N as f64;
        Self { probs: [p; N] }
    }
}

impl<const N: usize> DiscreteDistribution<N> {
    /// Create with specific probabilities.
    pub fn new(probs: [f64; N]) -> Self {
        Self { probs }
    }

    /// Create with all probability on state i.
    pub fn certain(i: usize) -> Self {
        let mut probs = [0.0; N];
        if i < N {
            probs[i] = 1.0;
        }
        Self { probs }
    }

    /// Entropy of the distribution.
    pub fn entropy(&self) -> f64 {
        -self
            .probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Most likely state.
    pub fn argmax(&self) -> usize {
        self.probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Normalize probabilities to sum to 1.
    pub fn normalize(&mut self) {
        let sum: f64 = self.probs.iter().sum();
        if sum > 0.0 {
            for p in &mut self.probs {
                *p /= sum;
            }
        }
    }
}

// === Helper functions ===

/// Standard normal CDF approximation (Abramowitz & Stegun).
pub fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2π)
    let p = d
        * (-x * x / 2.0).exp()
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    if x >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Standard normal quantile approximation (Beasley-Springer-Moro).
pub fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-10 {
        return 0.0;
    }

    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (c[0] + q * (c[1] + q * (c[2] + q * (c[3] + q * (c[4] + q * c[5])))))
            / (1.0 + q * (d[0] + q * (d[1] + q * (d[2] + q * d[3]))))
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (a[0] + r * (a[1] + r * (a[2] + r * (a[3] + r * (a[4] + r * a[5])))))
            / (1.0 + r * (b[0] + r * (b[1] + r * (b[2] + r * (b[3] + r * b[4])))))
            * q
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(c[0] + q * (c[1] + q * (c[2] + q * (c[3] + q * (c[4] + q * c[5])))))
            / (1.0 + q * (d[0] + q * (d[1] + q * (d[2] + q * d[3]))))
    }
}

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
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        PI.ln() - (PI * x).sin().ln() - gamma_ln(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        for i in 1..=g + 1 {
            a += c[i] / (x + i as f64);
        }
        let t = x + g as f64 + 0.5;
        0.5 * (2.0 * PI).ln() + (t).ln() * (x + 0.5) - t + a.ln()
    }
}

/// Incomplete gamma function (regularized).
fn incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    // Use series for x < a + 1
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;
        for n in 1..100 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < sum.abs() * 1e-10 {
                break;
            }
        }
        sum * (-x + a * x.ln() - gamma_ln(a)).exp()
    } else {
        // Use continued fraction for x >= a + 1
        1.0 - incomplete_gamma_cf(a, x)
    }
}

/// Incomplete gamma continued fraction.
fn incomplete_gamma_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..100 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-10 {
            break;
        }
    }

    (-x + a * x.ln() - gamma_ln(a)).exp() * h
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
    fn test_gamma_posterior() {
        let mut gamma = GammaPosterior::new(2.0, 1.0);
        assert!((gamma.mean() - 2.0).abs() < 1e-10);

        // Update with 3 events in 1 time unit
        gamma.update(3.0, 1.0);
        assert!((gamma.mean() - 2.5).abs() < 1e-10); // (2+3)/(1+1) = 2.5
    }

    #[test]
    fn test_normal_gamma_posterior() {
        let mut ng = NormalGammaPosterior::new(0.0, 1.0, 2.0, 1.0);

        // Update with observation
        ng.update(2.0);
        assert!((ng.mu - 1.0).abs() < 1e-10); // (1*0 + 2)/2 = 1

        // Check probability computation returns valid probability in [0, 1]
        let p = ng.prob_mean_greater_than(0.0);
        assert!(p >= 0.0 && p <= 1.0);

        // Check that std_of_mean works
        let std = ng.std_of_mean();
        assert!(std > 0.0 && std.is_finite());
    }

    #[test]
    fn test_dirichlet_posterior() {
        let mut dir = DirichletPosterior::uniform(3);

        // All means should be 1/3
        for i in 0..3 {
            assert!((dir.mean(i) - 1.0 / 3.0).abs() < 1e-10);
        }

        // Update category 0
        dir.update(0);
        assert!(dir.mean(0) > dir.mean(1));
        assert_eq!(dir.argmax(), 0);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }

    #[test]
    fn test_discrete_distribution() {
        let dist = DiscreteDistribution::<3>::default();
        assert!((dist.probs[0] - 1.0 / 3.0).abs() < 1e-10);
        // When all probs are equal, argmax can return any valid index
        let idx = dist.argmax();
        assert!(idx < 3);

        // Test with non-uniform distribution
        let non_uniform = DiscreteDistribution::<3>::new([0.1, 0.6, 0.3]);
        assert_eq!(non_uniform.argmax(), 1); // Index 1 has highest prob
    }
}
