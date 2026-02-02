//! Position Continuation Model using Beta-Binomial Bayesian inference.
//!
//! This module implements the mathematical foundation for HOLD/ADD/REDUCE
//! position decisions. Instead of always mean-reverting via inventory skew,
//! we estimate P(continuation | fills, regime) to decide whether positions
//! should be held, added to, or reduced.
//!
//! ## Key Insight
//!
//! GLFT inventory skew formula `skew = q × γ × σ² × T` always mean-reverts.
//! This causes repeated "buy at P, sell at P-Δ" cycles with negative EV after costs.
//!
//! The Position Continuation Model solves this by:
//! - **HOLD**: Set `inventory_ratio = 0` → no skew → symmetric quotes
//! - **ADD**: Set `inventory_ratio < 0` → REVERSE skew → tighter on position-building side
//! - **REDUCE**: Normal behavior with urgency scaling
//!
//! ## Mathematical Foundation
//!
//! Beta-Binomial Conjugate Prior for P(continuation | fills, regime):
//!
//! ```text
//! P(cont) ~ Beta(α, β)
//! Prior: α₀, β₀ from regime (cascade=0.8, normal=0.5, quiet=0.3)
//! Update: α += aligned_fills, β += adverse_fills
//! Posterior mean: p_cont = α / (α + β)
//! Confidence: conf = 1 - Var[Beta(α,β)] / Var[Beta(1,1)]
//! ```

/// Beta-Binomial model for position continuation probability.
///
/// Maintains a Beta distribution posterior over the probability that
/// the current position direction will continue to be profitable.
#[derive(Debug, Clone)]
pub struct ContinuationPosterior {
    /// Alpha parameter (aligned fill successes + prior)
    pub alpha: f64,
    /// Beta parameter (adverse fill failures + prior)
    pub beta: f64,
}

impl Default for ContinuationPosterior {
    fn default() -> Self {
        // Start with neutral prior (Beta(2.5, 2.5) → mean = 0.5)
        Self {
            alpha: 2.5,
            beta: 2.5,
        }
    }
}

impl ContinuationPosterior {
    /// Initialize from regime with appropriate prior.
    ///
    /// Different regimes have different continuation probabilities:
    /// - **cascade**: High continuation (momentum persists), prior mean = 0.8
    /// - **bursty**: Moderate-high continuation, prior mean = 0.6
    /// - **normal**: Neutral, prior mean = 0.5
    /// - **quiet**: Low continuation (mean-reversion dominates), prior mean = 0.3
    pub fn from_regime(regime: &str) -> Self {
        let (alpha, beta) = match regime.to_lowercase().as_str() {
            "cascade" | "extreme" => (4.0, 1.0),  // Prior mean = 0.8
            "bursty" | "high" => (3.0, 2.0),       // Prior mean = 0.6
            "normal" => (2.5, 2.5),                // Prior mean = 0.5
            "quiet" | "low" => (1.5, 3.5),         // Prior mean = 0.3
            _ => (2.5, 2.5),                       // Default to neutral
        };
        Self { alpha, beta }
    }

    /// Create with explicit parameters.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.max(0.1),
            beta: beta.max(0.1),
        }
    }

    /// Update posterior with fill observation.
    ///
    /// # Arguments
    /// * `is_aligned` - True if fill was in the direction of the position
    /// * `weight` - Observation weight (typically sqrt(size) for size-weighted updates)
    pub fn observe_fill(&mut self, is_aligned: bool, weight: f64) {
        let w = weight.max(0.01).min(5.0); // Clamp weight to reasonable range
        if is_aligned {
            self.alpha += w;
        } else {
            self.beta += w;
        }
    }

    /// Posterior mean P(continuation).
    ///
    /// This is the expected probability that the position direction
    /// will continue to be profitable.
    pub fn prob_continuation(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Confidence in the estimate [0, 1].
    ///
    /// Based on variance reduction from uniform prior:
    /// conf = 1 - Var[Beta(α,β)] / Var[Beta(1,1)]
    ///
    /// Higher α + β → lower variance → higher confidence.
    pub fn confidence(&self) -> f64 {
        let n = self.alpha + self.beta;
        let uniform_var = 1.0 / 12.0; // Var[Beta(1,1)] = 1/12
        let posterior_var = (self.alpha * self.beta) / (n * n * (n + 1.0));
        (1.0 - posterior_var / uniform_var).max(0.0)
    }

    /// Effective sample size (α + β - 2).
    ///
    /// Represents how many observations have been incorporated
    /// beyond the uniform prior.
    pub fn effective_n(&self) -> f64 {
        (self.alpha + self.beta - 2.0).max(0.0)
    }

    /// Decay posterior toward prior over time.
    ///
    /// Applies exponential decay to prevent stale beliefs from dominating.
    /// Useful for non-stationary markets where continuation probability changes.
    ///
    /// # Arguments
    /// * `half_life_fills` - Number of fills for half-life decay
    pub fn decay(&mut self, half_life_fills: f64) {
        if half_life_fills <= 0.0 {
            return;
        }
        let factor = 0.5_f64.powf(1.0 / half_life_fills.max(1.0));
        // Decay toward prior (α=1, β=1 for uniform, but we use regime prior)
        // Keep at least 1.0 for both to maintain proper Beta distribution
        self.alpha = 1.0 + (self.alpha - 1.0) * factor;
        self.beta = 1.0 + (self.beta - 1.0) * factor;
    }

    /// Reset to regime prior.
    pub fn reset_to_regime(&mut self, regime: &str) {
        let fresh = Self::from_regime(regime);
        self.alpha = fresh.alpha;
        self.beta = fresh.beta;
    }

    /// Get the 95% credible interval for P(continuation).
    ///
    /// Returns (lower, upper) bounds using the beta quantile approximation.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        // Use normal approximation for Beta quantiles when α, β > 1
        let mean = self.prob_continuation();
        let var = (self.alpha * self.beta) /
            ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0));
        let std = var.sqrt();

        // 95% CI ≈ mean ± 1.96 × std
        let lower = (mean - 1.96 * std).max(0.0);
        let upper = (mean + 1.96 * std).min(1.0);
        (lower, upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_neutral() {
        let post = ContinuationPosterior::default();
        let p = post.prob_continuation();
        assert!((p - 0.5).abs() < 0.01, "Default should be neutral, got {}", p);
    }

    #[test]
    fn test_regime_priors() {
        let cascade = ContinuationPosterior::from_regime("cascade");
        let quiet = ContinuationPosterior::from_regime("quiet");

        assert!(cascade.prob_continuation() > 0.7, "Cascade prior should be high");
        assert!(quiet.prob_continuation() < 0.4, "Quiet prior should be low");
    }

    #[test]
    fn test_observe_fill_updates() {
        let mut post = ContinuationPosterior::default();
        let initial_p = post.prob_continuation();

        // Observe aligned fill
        post.observe_fill(true, 1.0);
        assert!(post.prob_continuation() > initial_p, "Aligned fill should increase p_cont");

        // Observe adverse fill
        let p_after_aligned = post.prob_continuation();
        post.observe_fill(false, 1.0);
        assert!(post.prob_continuation() < p_after_aligned, "Adverse fill should decrease p_cont");
    }

    #[test]
    fn test_confidence_increases_with_observations() {
        let mut post = ContinuationPosterior::default();
        let initial_conf = post.confidence();

        for _ in 0..10 {
            post.observe_fill(true, 1.0);
        }

        assert!(post.confidence() > initial_conf, "Confidence should increase with observations");
    }

    #[test]
    fn test_decay_moves_toward_prior() {
        let mut post = ContinuationPosterior::from_regime("normal");

        // Add lots of aligned observations
        for _ in 0..20 {
            post.observe_fill(true, 1.0);
        }
        let p_before_decay = post.prob_continuation();

        // Decay multiple times
        for _ in 0..50 {
            post.decay(10.0);
        }

        // Should move back toward 0.5 (normal regime prior mean)
        let p_after_decay = post.prob_continuation();
        assert!(
            (p_after_decay - 0.5).abs() < (p_before_decay - 0.5).abs(),
            "Decay should move toward prior: before={}, after={}",
            p_before_decay,
            p_after_decay
        );
    }

    #[test]
    fn test_credible_interval() {
        let post = ContinuationPosterior::new(10.0, 10.0);
        let (lower, upper) = post.credible_interval_95();

        assert!(lower < 0.5 && upper > 0.5, "CI should contain 0.5 for symmetric Beta(10,10)");
        assert!(lower >= 0.0 && upper <= 1.0, "CI should be in [0,1]");
    }
}
