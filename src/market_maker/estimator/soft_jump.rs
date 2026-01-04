//! Soft Jump Classification Module
//!
//! Replaces the binary `is_jump()` classification with a probabilistic mixture model
//! that outputs P(jump) ∈ [0,1] for each return observation.

// Allow dead code since this is V2 infrastructure being built incrementally
#![allow(dead_code)]
//!
//! ## The Bug (Original Binary Classification)
//! ```text
//! // WRONG: Binary decision creates discontinuity at threshold
//! fn is_jump(&self, log_return: f64, sigma_clean: f64) -> bool {
//!     log_return.abs() > self.config.jump_threshold_sigmas * sigma_clean
//! }
//! ```
//!
//! ## The Fix (Mixture Model)
//! Model returns as a two-component mixture:
//! - Diffusion component: N(0, σ_diffusion) - normal price movements
//! - Jump component: N(0, σ_jump) where σ_jump ≈ 3-5× σ_diffusion
//!
//! P(jump | return) = P(jump) × P(return | jump) / P(return)
//!
//! This provides:
//! - Smooth toxicity score (rolling average of P(jump))
//! - No discontinuity at arbitrary threshold
//! - Adaptive learning of jump size distribution

use super::tick_ewma::TickEWMA;

/// Soft jump classifier using a two-component Gaussian mixture model.
///
/// Instead of binary "is this a jump?", we compute P(jump) ∈ [0,1]
/// for each return observation and track a rolling toxicity score.
#[derive(Debug, Clone)]
pub(crate) struct SoftJumpClassifier {
    /// Prior probability of a jump (learned)
    pi: f64,

    /// Diffusion volatility (normal price movements)
    sigma_diffusion: f64,

    /// Jump volatility (typically 3-5× σ_diffusion)
    sigma_jump: f64,

    /// EWMA alpha for updating pi
    pi_alpha: f64,

    /// Ratio σ_jump / σ_diffusion (typically 3-5)
    jump_vol_ratio: f64,

    /// EWMA for tracking rolling toxicity score
    toxicity_ewma: TickEWMA,

    /// Minimum observations before classifier is calibrated
    min_observations: usize,

    /// Observation count
    observation_count: usize,

    /// Last computed jump probability
    last_jump_prob: f64,
}

impl SoftJumpClassifier {
    /// Create a new soft jump classifier.
    ///
    /// # Arguments
    /// * `initial_pi` - Initial prior probability of jump (e.g., 0.05 = 5%)
    /// * `jump_vol_ratio` - Ratio of jump vol to diffusion vol (e.g., 4.0)
    /// * `pi_alpha` - EWMA alpha for updating pi (e.g., 0.01)
    /// * `toxicity_half_life` - Half-life in ticks for toxicity EWMA
    pub(crate) fn new(
        initial_pi: f64,
        jump_vol_ratio: f64,
        pi_alpha: f64,
        toxicity_half_life: f64,
    ) -> Self {
        Self {
            pi: initial_pi,
            sigma_diffusion: 0.0001, // Will be updated
            sigma_jump: 0.0001 * jump_vol_ratio,
            pi_alpha,
            jump_vol_ratio,
            toxicity_ewma: TickEWMA::new_uninitialized(toxicity_half_life),
            min_observations: 20,
            observation_count: 0,
            last_jump_prob: 0.0,
        }
    }

    /// Create with default parameters for market making
    pub(crate) fn default_params() -> Self {
        Self::new(
            0.05, // 5% prior jump probability
            4.0,  // Jumps are 4x larger than normal moves
            0.01, // Slow learning rate for pi
            50.0, // 50-tick half-life for toxicity
        )
    }

    /// Update the diffusion volatility estimate (from bipower variation)
    pub(crate) fn update_sigma(&mut self, sigma_diffusion: f64) {
        if sigma_diffusion > 1e-12 {
            self.sigma_diffusion = sigma_diffusion;
            self.sigma_jump = sigma_diffusion * self.jump_vol_ratio;
        }
    }

    /// Compute P(jump | return) using Bayes' rule on mixture model.
    ///
    /// P(jump | r) = P(jump) × P(r | jump) / [P(jump) × P(r | jump) + P(diff) × P(r | diff)]
    pub(crate) fn jump_probability(&self, log_return: f64) -> f64 {
        if self.sigma_diffusion < 1e-12 || self.sigma_jump < 1e-12 {
            return 0.0;
        }

        // Log-likelihoods for numerical stability
        let ll_diff = gaussian_log_pdf(log_return, 0.0, self.sigma_diffusion);
        let ll_jump = gaussian_log_pdf(log_return, 0.0, self.sigma_jump);

        let log_prior_jump = self.pi.max(1e-10).ln();
        let log_prior_diff = (1.0 - self.pi).max(1e-10).ln();

        let log_post_jump = log_prior_jump + ll_jump;
        let log_post_diff = log_prior_diff + ll_diff;

        // Softmax for numerical stability
        let max_log = log_post_jump.max(log_post_diff);
        let exp_jump = (log_post_jump - max_log).exp();
        let exp_diff = (log_post_diff - max_log).exp();

        exp_jump / (exp_jump + exp_diff)
    }

    /// Process a new return observation.
    ///
    /// Updates:
    /// 1. Jump probability for this observation
    /// 2. Prior pi (EWMA update)
    /// 3. Rolling toxicity score
    pub(crate) fn on_return(&mut self, log_return: f64) {
        if self.sigma_diffusion < 1e-12 {
            return;
        }

        // Compute P(jump | return)
        let p_jump = self.jump_probability(log_return);
        self.last_jump_prob = p_jump;

        // Update prior pi (EWMA)
        self.pi = self.pi_alpha * p_jump + (1.0 - self.pi_alpha) * self.pi;

        // Clamp pi to reasonable range
        self.pi = self.pi.clamp(0.001, 0.5);

        // Update toxicity EWMA
        self.toxicity_ewma.update(p_jump);

        self.observation_count += 1;
    }

    /// Get current toxicity score [0, 1].
    ///
    /// This is the rolling average of P(jump) over recent observations.
    /// Higher values indicate more toxic (jump-dominated) regimes.
    pub(crate) fn toxicity_score(&self) -> f64 {
        if self.observation_count < self.min_observations {
            return self.pi; // Return prior during warmup
        }
        self.toxicity_ewma.value().clamp(0.0, 1.0)
    }

    /// Get current learned prior probability of jump.
    pub(crate) fn pi(&self) -> f64 {
        self.pi
    }

    /// Get last computed jump probability.
    pub(crate) fn last_jump_probability(&self) -> f64 {
        self.last_jump_prob
    }

    /// Check if classifier is calibrated (has enough observations).
    pub(crate) fn is_calibrated(&self) -> bool {
        self.observation_count >= self.min_observations
    }

    /// Check if in toxic regime (toxicity > threshold).
    ///
    /// Recommended threshold: 0.15 (15% of returns are jump-like)
    pub(crate) fn is_toxic_regime(&self, threshold: f64) -> bool {
        self.toxicity_score() > threshold
    }

    /// Get the effective jump ratio (similar to RV/BV).
    ///
    /// Maps toxicity to a ratio for backwards compatibility.
    /// toxicity=0 → ratio=1.0, toxicity=0.5 → ratio=3.0
    pub(crate) fn effective_jump_ratio(&self) -> f64 {
        1.0 + 4.0 * self.toxicity_score()
    }

    /// Get current diffusion sigma.
    pub(crate) fn sigma_diffusion(&self) -> f64 {
        self.sigma_diffusion
    }

    /// Get current jump sigma.
    pub(crate) fn sigma_jump(&self) -> f64 {
        self.sigma_jump
    }
}

/// Log probability density of Gaussian N(μ, σ)
fn gaussian_log_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    const LOG_2PI: f64 = 1.8378770664093453; // ln(2π)
    let z = (x - mu) / sigma;
    -0.5 * (LOG_2PI + 2.0 * sigma.ln() + z * z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_log_pdf() {
        // Standard normal at 0
        let ll = gaussian_log_pdf(0.0, 0.0, 1.0);
        // Should be -0.5 * ln(2π) ≈ -0.919
        assert!((ll - (-0.919)).abs() < 0.01);

        // 1 sigma away
        let ll_1sigma = gaussian_log_pdf(1.0, 0.0, 1.0);
        // Should be -0.5 * (ln(2π) + 1) ≈ -1.419
        assert!((ll_1sigma - (-1.419)).abs() < 0.01);
    }

    #[test]
    fn test_soft_jump_small_returns() {
        let mut classifier = SoftJumpClassifier::default_params();
        classifier.update_sigma(0.0001); // 1 bp/sqrt(tick)

        // Small return (0.5 sigma) should have low jump probability
        let p_jump = classifier.jump_probability(0.00005);
        assert!(p_jump < 0.1, "Small return should have low P(jump)");
    }

    #[test]
    fn test_soft_jump_large_returns() {
        let mut classifier = SoftJumpClassifier::default_params();
        classifier.update_sigma(0.0001);

        // Large return (5 sigma) should have high jump probability
        let p_jump = classifier.jump_probability(0.0005);
        assert!(p_jump > 0.5, "Large return should have high P(jump)");
    }

    #[test]
    fn test_toxicity_tracking() {
        let mut classifier = SoftJumpClassifier::default_params();
        classifier.update_sigma(0.0001);

        // Feed small returns - toxicity should stay low
        for _ in 0..50 {
            classifier.on_return(0.00003);
        }
        assert!(
            classifier.toxicity_score() < 0.1,
            "Small returns should have low toxicity"
        );

        // Feed large returns - toxicity should increase
        for _ in 0..50 {
            classifier.on_return(0.0005);
        }
        assert!(
            classifier.toxicity_score() > 0.3,
            "Large returns should increase toxicity"
        );
    }

    #[test]
    fn test_pi_learning() {
        let mut classifier = SoftJumpClassifier::new(0.1, 4.0, 0.1, 10.0);
        classifier.update_sigma(0.0001);

        // Feed small returns - pi should decrease
        for _ in 0..100 {
            classifier.on_return(0.00002);
        }
        assert!(
            classifier.pi() < 0.1,
            "Small returns should decrease pi from 0.1"
        );

        // Feed large returns - pi should increase
        for _ in 0..100 {
            classifier.on_return(0.0005);
        }
        assert!(
            classifier.pi() > 0.15,
            "Large returns should increase pi toward 0.5"
        );
    }

    #[test]
    fn test_effective_jump_ratio() {
        let mut classifier = SoftJumpClassifier::default_params();
        classifier.update_sigma(0.0001);

        // Initial ratio should be near 1.0
        // (toxicity starts at pi = 0.05, so ratio ≈ 1.2)
        let initial_ratio = classifier.effective_jump_ratio();
        assert!(initial_ratio > 1.0 && initial_ratio < 1.5);

        // After toxic flow, ratio should increase
        for _ in 0..100 {
            classifier.on_return(0.0005);
        }
        let toxic_ratio = classifier.effective_jump_ratio();
        assert!(toxic_ratio > 2.0, "Toxic flow should increase jump ratio");
    }
}
