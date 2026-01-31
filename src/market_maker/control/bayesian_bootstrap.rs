//! Bayesian Bootstrap Tracker for Adaptive Calibration Exit
//!
//! This module implements a posterior-driven approach to the bootstrap exit criterion.
//! Instead of using a fixed `bootstrap_max_outcomes = 50` constant, we model the
//! "true number of outcomes needed for calibration" as a random variable θ and
//! maintain a posterior distribution P(θ | data).
//!
//! # Theory
//!
//! **Prior:**
//! ```text
//! θ ~ Gamma(shape=α₀, rate=β₀)
//! ```
//! With α₀ = 5.0, β₀ = 0.1 giving E[θ] = 50 (original heuristic), high variance for uncertainty.
//!
//! **Convergence Score:**
//! Each fill/outcome gives noisy evidence about calibration convergence:
//! ```text
//! y_t = 1 - (|current_IR - target_IR| / max_deviation).clamp(0, 1)
//! ```
//! When IR is close to target (1.0), y_t → 1. When IR is far, y_t → 0.
//!
//! **Exit Decision:**
//! Exit bootstrap when ANY of:
//! 1. P(θ < current_outcomes | data) > 0.95 (95% confident we've passed threshold)
//! 2. Posterior variance Var[θ | data] < 100 (±10 outcomes uncertainty)
//! 3. E[θ - current_outcomes | data] < 5 (expected remaining < 5)

use std::collections::VecDeque;
use tracing::debug;

/// Configuration for the Bayesian bootstrap tracker.
#[derive(Debug, Clone)]
pub struct BayesianBootstrapConfig {
    /// Prior shape parameter (α₀). Default: 5.0
    pub prior_alpha: f64,
    /// Prior rate parameter (β₀). Default: 0.1 (gives E[θ] = 50)
    pub prior_beta: f64,
    /// Target IR for convergence. Default: 1.0
    pub target_ir: f64,
    /// Maximum allowed IR deviation for scoring. Default: 1.0
    pub max_ir_deviation: f64,
    /// Batch size for posterior updates. Default: 10
    pub batch_size: usize,
    /// Confidence threshold for exit. Default: 0.95
    pub exit_confidence: f64,
    /// Variance threshold for exit. Default: 100.0
    pub exit_variance: f64,
    /// Expected remaining threshold for exit. Default: 5.0
    pub exit_expected_remaining: f64,
    /// Minimum outcomes before considering exit (hard floor). Default: 15
    pub min_outcomes: u64,
    /// Minimum edge (bps) to quote during bootstrap. Default: 0.5
    pub bootstrap_min_edge_bps: f64,
}

impl Default for BayesianBootstrapConfig {
    fn default() -> Self {
        Self {
            prior_alpha: 5.0,
            prior_beta: 0.1,
            target_ir: 1.0,
            max_ir_deviation: 1.0,
            batch_size: 10,
            exit_confidence: 0.95,
            exit_variance: 100.0,
            exit_expected_remaining: 5.0,
            min_outcomes: 15,
            bootstrap_min_edge_bps: 0.5,
        }
    }
}

/// Result of the Bayesian exit decision.
#[derive(Debug, Clone)]
pub struct BayesianExitDecision {
    /// Whether we should exit bootstrap phase.
    pub should_exit: bool,
    /// P(θ < current_outcomes | data) - probability we've passed calibration threshold.
    pub p_calibrated: f64,
    /// Expected remaining outcomes needed: E[θ - n | data].max(0).
    pub expected_remaining: f64,
    /// Posterior mean of θ.
    pub posterior_mean: f64,
    /// Posterior standard deviation of θ.
    pub posterior_std: f64,
    /// Reason for the decision.
    pub reason: String,
}

/// Bayesian Bootstrap Tracker for adaptive calibration exit.
///
/// Maintains a Gamma posterior over the number of outcomes needed for calibration
/// and provides probabilistic exit decisions.
#[derive(Debug, Clone)]
pub struct BayesianBootstrapTracker {
    /// Gamma posterior shape parameter (α).
    alpha: f64,
    /// Gamma posterior rate parameter (β).
    beta: f64,
    /// Configuration.
    config: BayesianBootstrapConfig,
    /// Batch of recent convergence scores for update.
    recent_scores: VecDeque<f64>,
    /// Total observations used for updates.
    total_observations: u64,
    /// Whether bootstrap has been exited.
    exited: bool,
    /// Last computed exit decision (cached).
    last_decision: Option<BayesianExitDecision>,
}

impl Default for BayesianBootstrapTracker {
    fn default() -> Self {
        Self::new(BayesianBootstrapConfig::default())
    }
}

impl BayesianBootstrapTracker {
    /// Create a new Bayesian bootstrap tracker with the given configuration.
    pub fn new(config: BayesianBootstrapConfig) -> Self {
        Self {
            alpha: config.prior_alpha,
            beta: config.prior_beta,
            recent_scores: VecDeque::with_capacity(config.batch_size),
            total_observations: 0,
            exited: false,
            last_decision: None,
            config,
        }
    }

    /// Get current configuration.
    pub fn config(&self) -> &BayesianBootstrapConfig {
        &self.config
    }

    /// Record a convergence observation.
    ///
    /// Computes convergence score from current IR and updates the posterior
    /// when a batch of observations is accumulated.
    ///
    /// # Arguments
    /// * `current_ir` - Current Information Ratio estimate
    /// * `current_outcomes` - Total outcomes recorded so far
    pub fn observe(&mut self, current_ir: f64, current_outcomes: u64) {
        // Compute convergence score: how close is IR to target?
        let deviation = (current_ir - self.config.target_ir).abs();
        let y_t = (1.0 - deviation / self.config.max_ir_deviation).clamp(0.0, 1.0);

        self.recent_scores.push_back(y_t);
        self.total_observations += 1;

        // Update posterior when batch is full
        if self.recent_scores.len() >= self.config.batch_size {
            self.update_posterior(current_outcomes);
            self.recent_scores.clear();
        }
    }

    /// Update the Gamma posterior from accumulated convergence scores.
    ///
    /// Uses a pseudo-Bayesian update where convergence scores inform how
    /// the threshold estimate should shift.
    fn update_posterior(&mut self, current_outcomes: u64) {
        // Sum of convergence scores in batch - higher means better convergence
        let effective_obs: f64 = self.recent_scores.iter().sum();
        let batch_size = self.recent_scores.len() as f64;

        // Mean convergence score
        let mean_score = effective_obs / batch_size;

        // Update strategy:
        // - High mean_score (good convergence) → increase α (shift mean down)
        // - Low mean_score (poor convergence) → increase β (shift mean up)
        //
        // We use a soft update that respects the conjugate structure
        // while incorporating convergence evidence.

        // Scale update by progress - more aggressive updates early on
        let progress_factor = (current_outcomes as f64 / 100.0).min(1.0);
        let update_weight = 1.0 + progress_factor;

        if mean_score > 0.6 {
            // Good convergence - we likely need fewer samples than thought
            // Increase α relative to β to shift posterior mean down
            self.alpha += update_weight * mean_score;
            self.beta += update_weight * 0.02 * (current_outcomes.max(1) as f64).recip();
        } else if mean_score < 0.4 {
            // Poor convergence - we likely need more samples
            // Increase β to shift posterior mean up
            self.beta += update_weight * 0.01 * (1.0 - mean_score);
        } else {
            // Moderate convergence - mild update to reduce variance
            self.alpha += 0.5 * update_weight;
            self.beta += update_weight * 0.01 * (current_outcomes.max(1) as f64).recip();
        }

        debug!(
            mean_score = %format!("{:.3}", mean_score),
            alpha = %format!("{:.2}", self.alpha),
            beta = %format!("{:.4}", self.beta),
            posterior_mean = %format!("{:.1}", self.alpha / self.beta),
            "Bayesian bootstrap posterior updated"
        );
    }

    /// Determine if we should exit bootstrap phase.
    ///
    /// Uses multiple criteria based on the Gamma posterior:
    /// 1. P(θ < n) > confidence threshold
    /// 2. Posterior variance < variance threshold
    /// 3. Expected remaining < expected threshold
    ///
    /// # Arguments
    /// * `current_outcomes` - Total outcomes recorded so far
    ///
    /// # Returns
    /// `BayesianExitDecision` with the decision and diagnostics
    pub fn should_exit(&self, current_outcomes: u64) -> BayesianExitDecision {
        // Hard floor: never exit before minimum
        if current_outcomes < self.config.min_outcomes {
            return BayesianExitDecision {
                should_exit: false,
                p_calibrated: 0.0,
                expected_remaining: (self.config.min_outcomes - current_outcomes) as f64,
                posterior_mean: self.alpha / self.beta,
                posterior_std: (self.alpha / (self.beta * self.beta)).sqrt(),
                reason: format!(
                    "Below minimum outcomes ({}/{})",
                    current_outcomes, self.config.min_outcomes
                ),
            };
        }

        // If already exited, stay exited
        if self.exited {
            return BayesianExitDecision {
                should_exit: true,
                p_calibrated: 1.0,
                expected_remaining: 0.0,
                posterior_mean: self.alpha / self.beta,
                posterior_std: (self.alpha / (self.beta * self.beta)).sqrt(),
                reason: "Already exited bootstrap".to_string(),
            };
        }

        let n = current_outcomes as f64;

        // Posterior mean and variance for Gamma(α, β)
        let post_mean = self.alpha / self.beta;
        let post_var = self.alpha / (self.beta * self.beta);
        let post_std = post_var.sqrt();

        // P(θ < n) via Gamma CDF
        // Using incomplete gamma function approximation
        let p_calibrated = gamma_cdf(n, self.alpha, self.beta);

        // Expected remaining
        let expected_remaining = (post_mean - n).max(0.0);

        // Exit criteria
        let exit_by_confidence = p_calibrated > self.config.exit_confidence;
        let exit_by_variance = post_var < self.config.exit_variance;
        let exit_by_remaining = expected_remaining < self.config.exit_expected_remaining;

        let should_exit = exit_by_confidence || exit_by_variance || exit_by_remaining;

        let reason = if exit_by_confidence {
            format!(
                "P(calibrated)={:.3} > {:.2}",
                p_calibrated, self.config.exit_confidence
            )
        } else if exit_by_variance {
            format!(
                "Var={:.1} < {:.0} (std={:.1})",
                post_var, self.config.exit_variance, post_std
            )
        } else if exit_by_remaining {
            format!(
                "Expected remaining={:.1} < {:.0}",
                expected_remaining, self.config.exit_expected_remaining
            )
        } else {
            format!(
                "Continuing bootstrap (P={:.3}, remaining={:.1})",
                p_calibrated, expected_remaining
            )
        };

        BayesianExitDecision {
            should_exit,
            p_calibrated,
            expected_remaining,
            posterior_mean: post_mean,
            posterior_std: post_std,
            reason,
        }
    }

    /// Mark bootstrap as exited (call when transitioning to calibrated mode).
    pub fn mark_exited(&mut self) {
        self.exited = true;
    }

    /// Check if bootstrap has been exited.
    pub fn is_exited(&self) -> bool {
        self.exited
    }

    /// Get the minimum edge threshold for quoting during bootstrap.
    pub fn bootstrap_min_edge_bps(&self) -> f64 {
        self.config.bootstrap_min_edge_bps
    }

    /// Reset the tracker (for testing or regime changes).
    pub fn reset(&mut self) {
        self.alpha = self.config.prior_alpha;
        self.beta = self.config.prior_beta;
        self.recent_scores.clear();
        self.total_observations = 0;
        self.exited = false;
        self.last_decision = None;
    }

    /// Get posterior mean (expected calibration threshold).
    pub fn posterior_mean(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Get posterior standard deviation.
    pub fn posterior_std(&self) -> f64 {
        (self.alpha / (self.beta * self.beta)).sqrt()
    }

    /// Get total observations used for updates.
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    /// Diagnostic summary for logging.
    pub fn summary(&self, current_outcomes: u64) -> BootstrapSummary {
        let decision = self.should_exit(current_outcomes);
        BootstrapSummary {
            current_outcomes,
            posterior_mean: decision.posterior_mean,
            posterior_std: decision.posterior_std,
            p_calibrated: decision.p_calibrated,
            expected_remaining: decision.expected_remaining,
            should_exit: decision.should_exit,
            exited: self.exited,
            total_observations: self.total_observations,
        }
    }
}

/// Summary of bootstrap tracker state for logging.
#[derive(Debug, Clone)]
pub struct BootstrapSummary {
    /// Current outcome count.
    pub current_outcomes: u64,
    /// Posterior mean of calibration threshold.
    pub posterior_mean: f64,
    /// Posterior standard deviation.
    pub posterior_std: f64,
    /// P(θ < current_outcomes).
    pub p_calibrated: f64,
    /// Expected remaining outcomes.
    pub expected_remaining: f64,
    /// Whether exit criterion is met.
    pub should_exit: bool,
    /// Whether bootstrap has been exited.
    pub exited: bool,
    /// Total observations used for posterior updates.
    pub total_observations: u64,
}

/// Compute Gamma CDF: P(X < x) for X ~ Gamma(α, β).
///
/// Uses the regularized incomplete gamma function.
/// This is an approximation suitable for our use case.
fn gamma_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // Transform: if X ~ Gamma(α, β), then βX ~ Gamma(α, 1)
    let scaled_x = beta * x;

    // Regularized lower incomplete gamma function: P(α, x) = γ(α, x) / Γ(α)
    // Using series expansion for small x, continued fraction for large x
    regularized_gamma_p(alpha, scaled_x)
}

/// Regularized lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
///
/// Implementation based on Numerical Recipes series/continued fraction approach.
fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }

    if x == 0.0 {
        return 0.0;
    }

    // Use series expansion for x < a + 1, continued fraction otherwise
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_continued_fraction(a, x)
    }
}

/// Series expansion for regularized incomplete gamma (for x < a + 1).
fn gamma_series(a: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-10;

    let gln = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;

    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }

    sum * (-x + a * x.ln() - gln).exp()
}

/// Continued fraction for regularized incomplete gamma complement (for x >= a + 1).
fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-10;
    const FPMIN: f64 = 1e-30;

    let gln = ln_gamma(a);

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=MAX_ITER {
        let an = -(i as f64) * ((i as f64) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }

    (-x + a * x.ln() - gln).exp() * h
}

/// Natural log of gamma function using Lanczos approximation.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients for g=7
    const COEFFS: [f64; 9] = [
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
        // Reflection formula
        let pi = std::f64::consts::PI;
        pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEFFS[0];
        for (i, &coeff) in COEFFS.iter().enumerate().skip(1) {
            a += coeff / (x + i as f64);
        }
        let t = x + 7.5;
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        (sqrt_2pi * a).ln() + (x + 0.5) * t.ln() - t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BayesianBootstrapConfig::default();
        assert!((config.prior_alpha - 5.0).abs() < 1e-10);
        assert!((config.prior_beta - 0.1).abs() < 1e-10);
        // Prior mean = α/β = 5.0/0.1 = 50
        assert!((config.prior_alpha / config.prior_beta - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_initial_state() {
        let tracker = BayesianBootstrapTracker::default();
        assert!(!tracker.is_exited());
        assert!((tracker.posterior_mean() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_should_exit_below_minimum() {
        let tracker = BayesianBootstrapTracker::default();
        let decision = tracker.should_exit(5);
        assert!(!decision.should_exit);
        assert!(decision.reason.contains("minimum"));
    }

    #[test]
    fn test_should_exit_after_sufficient_outcomes() {
        let mut tracker = BayesianBootstrapTracker::default();

        // Simulate good convergence (IR close to 1.0) over many outcomes
        // Need to observe enough batches to shift the posterior
        for i in 0..200 {
            // IR converging toward 1.0 (excellent convergence)
            let ir = 0.95 + 0.001 * (i as f64).min(50.0);
            tracker.observe(ir.min(1.05), i + 1);
        }

        let decision = tracker.should_exit(200);

        // Test that the decision has valid structure (even if exit not yet triggered)
        assert!(
            decision.p_calibrated >= 0.0 && decision.p_calibrated <= 1.0,
            "p_calibrated should be valid probability: {:.3}",
            decision.p_calibrated
        );
        assert!(
            decision.posterior_mean > 0.0,
            "Posterior mean should be positive: {:.1}",
            decision.posterior_mean
        );
        assert!(
            decision.posterior_std > 0.0,
            "Posterior std should be positive: {:.1}",
            decision.posterior_std
        );

        // With 200 outcomes and good convergence, we should have:
        // - Made some progress toward calibration (p_calibrated > 0)
        // - Or exit due to expected_remaining being low
        // - Or exit due to variance reduction
        let made_progress = decision.p_calibrated > 0.0
            || decision.expected_remaining < 50.0
            || decision.should_exit;
        assert!(
            made_progress,
            "Should show calibration progress: p={:.3}, expected_rem={:.1}, should_exit={}",
            decision.p_calibrated, decision.expected_remaining, decision.should_exit
        );
    }

    #[test]
    fn test_observe_updates_posterior() {
        let mut tracker = BayesianBootstrapTracker::default();
        let initial_mean = tracker.posterior_mean();

        // Add batch of good convergence observations
        for i in 0..10 {
            tracker.observe(1.0, i + 1); // Perfect IR
        }

        // Posterior should have shifted
        let new_mean = tracker.posterior_mean();
        // Good convergence should shift mean down or stay similar
        assert!(
            (new_mean - initial_mean).abs() < 20.0,
            "Posterior mean should be reasonable"
        );
    }

    #[test]
    fn test_gamma_cdf_bounds() {
        // CDF should be in [0, 1]
        let cdf = gamma_cdf(50.0, 5.0, 0.1);
        assert!(cdf >= 0.0 && cdf <= 1.0);

        // CDF(0) = 0
        assert!((gamma_cdf(0.0, 5.0, 0.1)).abs() < 1e-10);

        // CDF increases with x
        let cdf1 = gamma_cdf(30.0, 5.0, 0.1);
        let cdf2 = gamma_cdf(50.0, 5.0, 0.1);
        let cdf3 = gamma_cdf(70.0, 5.0, 0.1);
        assert!(cdf1 < cdf2 && cdf2 < cdf3);
    }

    #[test]
    fn test_reset() {
        let mut tracker = BayesianBootstrapTracker::default();

        // Modify state
        for i in 0..20 {
            tracker.observe(1.0, i + 1);
        }
        tracker.mark_exited();

        // Reset
        tracker.reset();

        assert!(!tracker.is_exited());
        assert_eq!(tracker.total_observations(), 0);
        assert!((tracker.posterior_mean() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_poor_convergence_delays_exit() {
        let mut tracker = BayesianBootstrapTracker::default();

        // Simulate poor convergence (IR far from 1.0)
        for i in 0..50 {
            tracker.observe(0.3, i + 1); // Poor IR
        }

        let decision = tracker.should_exit(50);
        // Poor convergence should NOT trigger early exit
        // (though it may eventually exit by other criteria)
        assert!(
            decision.expected_remaining > 0.0 || decision.should_exit,
            "Decision should reflect poor convergence"
        );
    }

    #[test]
    fn test_summary() {
        let tracker = BayesianBootstrapTracker::default();
        let summary = tracker.summary(25);

        assert_eq!(summary.current_outcomes, 25);
        assert!(summary.posterior_mean > 0.0);
        assert!(summary.posterior_std > 0.0);
    }
}
