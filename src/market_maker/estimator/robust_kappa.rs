//! Robust kappa estimator using Student-t likelihood.
//!
//! This module provides a kappa (κ) estimator that resists outliers by using
//! a heavy-tailed Student-t distribution instead of the standard exponential.
//!
//! ## The Problem with Exponential Likelihood
//!
//! The standard Bayesian kappa estimator uses exponential likelihood:
//! ```text
//! P(δ | κ) = κ × exp(-κ × δ)
//! ```
//!
//! This is fragile to outliers because:
//! - One trade at 500 bps from mid gives L ∝ κ × exp(-0.05κ)
//! - This likelihood is maximized at κ → 0
//! - A single liquidation cascade can collapse the entire estimate
//!
//! ## The Solution: Student-t Distribution
//!
//! Student-t has heavier tails than exponential:
//! ```text
//! P(δ | κ, ν) ∝ (1 + (κδ)²/ν)^(-(ν+1)/2)
//! ```
//!
//! Where ν (degrees of freedom) controls tail weight:
//! - ν → ∞: converges to Gaussian
//! - ν = 1: Cauchy (very heavy tails)
//! - ν ≈ 3-5: good for market data
//!
//! ## Implementation: Iteratively Reweighted Least Squares (IRLS)
//!
//! Instead of full Bayesian inference, we use IRLS for computational efficiency:
//! 1. Initialize weights wᵢ = 1
//! 2. Compute κ̂ = Σwᵢ / Σ(wᵢ × δᵢ) (weighted MLE for exponential)
//! 3. Update weights: wᵢ = (ν + 1) / (ν + (κ̂ × δᵢ)²)
//! 4. Repeat until convergence
//!
//! Outliers get downweighted → they don't collapse κ.

use std::collections::VecDeque;
use tracing::debug;

/// Default degrees of freedom (moderate heavy tails)
const DEFAULT_NU: f64 = 4.0;

/// Maximum iterations for IRLS convergence
const MAX_IRLS_ITERATIONS: usize = 10;

/// Convergence threshold for IRLS
const IRLS_CONVERGENCE_THRESHOLD: f64 = 0.01;

/// Maximum κ bound
const MAX_KAPPA: f64 = 10000.0;

/// Minimum κ bound
const MIN_KAPPA: f64 = 50.0;

/// Minimum distance to prevent division issues
const MIN_DISTANCE: f64 = 0.00001; // 0.1 bps

/// Robust kappa estimator using Student-t likelihood.
///
/// Resists outliers by using heavy-tailed distribution instead of exponential.
/// Implements iteratively reweighted least squares (IRLS) for fast inference.
#[derive(Debug, Clone)]
pub(crate) struct RobustKappaEstimator {
    /// Current κ estimate
    kappa: f64,

    /// Degrees of freedom (tail heaviness parameter)
    /// Lower ν = heavier tails = more robust to outliers
    nu: f64,

    /// Observations: (distance, weight, timestamp)
    observations: VecDeque<(f64, f64, u64)>,

    /// Window in milliseconds
    window_ms: u64,

    /// Effective sample size (accounts for downweighted outliers)
    effective_sample_size: f64,

    /// Prior κ for regularization
    prior_kappa: f64,

    /// Prior strength (effective number of prior observations)
    prior_strength: f64,

    /// Number of observations added
    observation_count: u64,

    /// Number of outliers detected (weight < 0.5)
    outlier_count: u64,
}

impl RobustKappaEstimator {
    /// Create a new robust kappa estimator.
    ///
    /// # Arguments
    /// * `prior_kappa` - Prior expected value of κ (e.g., 2000 for liquid markets)
    /// * `prior_strength` - Effective sample size of prior (e.g., 10)
    /// * `nu` - Degrees of freedom for Student-t (e.g., 4.0)
    /// * `window_ms` - Rolling window in milliseconds
    pub(crate) fn new(prior_kappa: f64, prior_strength: f64, nu: f64, window_ms: u64) -> Self {
        Self {
            kappa: prior_kappa,
            nu: nu.max(1.0), // ν must be > 0
            observations: VecDeque::with_capacity(1000),
            window_ms,
            effective_sample_size: prior_strength,
            prior_kappa,
            prior_strength,
            observation_count: 0,
            outlier_count: 0,
        }
    }

    /// Create with default parameters for liquid markets.
    pub(crate) fn default_liquid() -> Self {
        Self::new(2000.0, 10.0, DEFAULT_NU, 300_000) // 5 min window
    }

    /// Create with default parameters for illiquid markets.
    #[allow(dead_code)] // Used in tests; production uses explicit config
    pub(crate) fn default_illiquid() -> Self {
        // Heavier tails (lower ν) and stronger prior for illiquid markets
        Self::new(1500.0, 20.0, 3.0, 600_000) // 10 min window
    }

    /// Add a trade observation.
    ///
    /// # Arguments
    /// * `timestamp_ms` - Trade timestamp
    /// * `distance` - Distance from mid as fraction (e.g., 0.001 = 10 bps)
    pub(crate) fn on_trade(&mut self, timestamp_ms: u64, distance: f64) {
        // Apply minimum distance floor
        let distance = distance.abs().max(MIN_DISTANCE);

        // Expire old observations
        self.expire_old(timestamp_ms);

        // Add new observation with initial weight = 1
        self.observations.push_back((distance, 1.0, timestamp_ms));
        self.observation_count += 1;

        // Recompute κ using IRLS
        self.update_kappa();

        // Log periodically
        if self.observation_count.is_multiple_of(50) {
            debug!(
                kappa = %format!("{:.0}", self.kappa),
                ess = %format!("{:.1}", self.effective_sample_size),
                observations = self.observations.len(),
                outliers = self.outlier_count,
                "Robust kappa updated"
            );
        }
    }

    /// Expire observations outside the rolling window.
    fn expire_old(&mut self, now: u64) {
        let cutoff = now.saturating_sub(self.window_ms);

        while let Some(&(_, _, ts)) = self.observations.front() {
            if ts < cutoff {
                self.observations.pop_front();
            } else {
                break;
            }
        }
    }

    /// Update κ using Iteratively Reweighted Least Squares (IRLS).
    fn update_kappa(&mut self) {
        if self.observations.is_empty() {
            self.kappa = self.prior_kappa;
            self.effective_sample_size = self.prior_strength;
            return;
        }

        // Initialize with prior-weighted mean
        let mut kappa_est = self.kappa;
        let _n = self.observations.len() as f64;

        for _iteration in 0..MAX_IRLS_ITERATIONS {
            // Step 1: Compute weights based on current κ estimate
            // w_i = (ν + 1) / (ν + (κ × δ_i)²)
            let mut sum_weights = self.prior_strength; // Prior contributes

            self.outlier_count = 0;

            for (distance, weight, _) in self.observations.iter_mut() {
                let z = kappa_est * *distance;
                let new_weight = (self.nu + 1.0) / (self.nu + z * z);

                // Clamp weight to [0.01, 1.0] for numerical stability
                *weight = new_weight.clamp(0.01, 1.0);

                // Track outliers (weight < 0.5)
                if *weight < 0.5 {
                    self.outlier_count += 1;
                }

                sum_weights += *weight;
            }

            // Step 2: Update κ estimate
            // For exponential MLE: κ = n / Σδ_i
            // Weighted: κ = Σw_i / Σ(w_i × δ_i)
            // But we want inverse: κ = Σw_i / (Σw_i / Σ(w_i/δ_i)) = Σ(w_i/δ_i) * Σw_i / n...
            //
            // Actually, for exponential with rate κ: MLE is κ̂ = n / Σδ_i
            // Weighted version: κ̂ = Σw_i / Σ(w_i × δ_i)
            let sum_weighted_distance: f64 =
                self.observations.iter().map(|(d, w, _)| w * d).sum::<f64>()
                    + self.prior_strength * (1.0 / self.prior_kappa); // Prior term

            let new_kappa = sum_weights / sum_weighted_distance;
            let new_kappa = new_kappa.clamp(MIN_KAPPA, MAX_KAPPA);

            // Check convergence
            let rel_change = (new_kappa - kappa_est).abs() / kappa_est.max(1.0);
            kappa_est = new_kappa;

            if rel_change < IRLS_CONVERGENCE_THRESHOLD {
                break;
            }
        }

        self.kappa = kappa_est;

        // Compute effective sample size
        // ESS = (Σw_i)² / Σw_i²
        let sum_w: f64 = self.observations.iter().map(|(_, w, _)| w).sum();
        let sum_w2: f64 = self.observations.iter().map(|(_, w, _)| w * w).sum();

        self.effective_sample_size = if sum_w2 > 0.0 {
            sum_w * sum_w / sum_w2 + self.prior_strength
        } else {
            self.prior_strength
        };
    }

    /// Get current κ estimate.
    pub(crate) fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Get confidence based on effective sample size.
    ///
    /// Returns a value in [0, 1] where:
    /// - 0.0 = no data beyond prior
    /// - 1.0 = many high-quality observations
    pub(crate) fn confidence(&self) -> f64 {
        // Confidence grows with ESS, saturating around 50 observations
        let ess_beyond_prior = (self.effective_sample_size - self.prior_strength).max(0.0);
        let conf = 1.0 - (-ess_beyond_prior / 30.0).exp();
        conf.clamp(0.0, 1.0)
    }

    /// Get effective sample size.
    pub(crate) fn effective_sample_size(&self) -> f64 {
        self.effective_sample_size
    }

    /// Get number of outliers detected (weight < 0.5).
    pub(crate) fn outlier_count(&self) -> u64 {
        self.outlier_count
    }

    /// Get degrees of freedom parameter.
    pub(crate) fn nu(&self) -> f64 {
        self.nu
    }

    /// Check if estimator is warmed up.
    pub(crate) fn is_warmed_up(&self) -> bool {
        self.effective_sample_size >= self.prior_strength + 10.0
    }

    /// Get observation count.
    pub(crate) fn observation_count(&self) -> u64 {
        self.observation_count
    }
}

impl Default for RobustKappaEstimator {
    fn default() -> Self {
        Self::default_liquid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_update() {
        let mut estimator = RobustKappaEstimator::new(2000.0, 10.0, 4.0, 300_000);

        // Add some normal observations around 10 bps
        for i in 0..20 {
            estimator.on_trade(i * 1000, 0.001); // 10 bps
        }

        // κ should be approximately 1/0.001 = 1000
        let kappa = estimator.kappa();
        assert!(
            kappa > 500.0 && kappa < 2000.0,
            "Expected κ around 1000, got {}",
            kappa
        );
    }

    #[test]
    fn test_outlier_resistance() {
        let mut estimator = RobustKappaEstimator::new(2000.0, 10.0, 4.0, 300_000);

        // Add 20 normal observations at 10 bps
        for i in 0..20 {
            estimator.on_trade(i * 1000, 0.001); // 10 bps
        }

        let kappa_before = estimator.kappa();

        // Add one extreme outlier at 500 bps
        estimator.on_trade(21 * 1000, 0.05); // 500 bps!

        let kappa_after = estimator.kappa();

        // κ should NOT collapse dramatically
        // With standard exponential, this would cause a massive drop
        let change_pct = (kappa_before - kappa_after).abs() / kappa_before * 100.0;
        assert!(
            change_pct < 30.0,
            "κ dropped {}% (from {} to {}), should be resistant to outliers",
            change_pct,
            kappa_before,
            kappa_after
        );
    }

    #[test]
    fn test_multiple_outliers() {
        let mut estimator = RobustKappaEstimator::new(2000.0, 10.0, 4.0, 300_000);

        // Add 20 normal observations at 10 bps
        for i in 0..20 {
            estimator.on_trade(i * 1000, 0.001);
        }

        let kappa_before = estimator.kappa();

        // Add 5 outliers (like a liquidation cascade)
        for i in 0..5 {
            estimator.on_trade((21 + i) * 1000, 0.03); // 300 bps
        }

        let kappa_after = estimator.kappa();

        // Should still resist the cascade
        let change_pct = (kappa_before - kappa_after).abs() / kappa_before * 100.0;
        assert!(
            change_pct < 50.0,
            "κ dropped {}% after liquidation cascade, should be more resistant",
            change_pct
        );

        // Should have detected outliers
        assert!(
            estimator.outlier_count() > 0,
            "Should detect outliers in cascade"
        );
    }

    #[test]
    fn test_prior_dominates_with_no_data() {
        let estimator = RobustKappaEstimator::new(2000.0, 10.0, 4.0, 300_000);

        assert_eq!(estimator.kappa(), 2000.0);
        assert!(!estimator.is_warmed_up());
    }

    #[test]
    fn test_window_expiry() {
        let mut estimator = RobustKappaEstimator::new(2000.0, 10.0, 4.0, 1000); // 1 second window

        // Add observations
        estimator.on_trade(0, 0.001);
        estimator.on_trade(500, 0.001);
        estimator.on_trade(1500, 0.001); // This should expire the first one

        // First observation should be expired
        assert_eq!(estimator.observations.len(), 2);
    }

    #[test]
    fn test_effective_sample_size() {
        let mut estimator = RobustKappaEstimator::new(2000.0, 10.0, 4.0, 300_000);

        // Add homogeneous observations (all should get high weight)
        for i in 0..20 {
            estimator.on_trade(i * 1000, 0.001);
        }

        // ESS should be close to n + prior_strength when weights are uniform
        let ess = estimator.effective_sample_size();
        assert!(
            ess > 20.0,
            "ESS should be at least 20 for 20 uniform observations, got {}",
            ess
        );
    }

    #[test]
    fn test_confidence_growth() {
        let mut estimator = RobustKappaEstimator::new(2000.0, 10.0, 4.0, 300_000);

        assert!(
            estimator.confidence() < 0.1,
            "Initial confidence should be low"
        );

        for i in 0..50 {
            estimator.on_trade(i * 1000, 0.001);
        }

        assert!(
            estimator.confidence() > 0.5,
            "Confidence should grow with observations"
        );
    }
}
