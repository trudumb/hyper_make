//! Parameter Covariance Tracking Module
//!
//! Tracks joint (κ, σ) uncertainty for proper spread uncertainty quantification.

// Allow dead code since this is V2 infrastructure being built incrementally
#![allow(dead_code)]
//!
//! In GLFT, the optimal spread δ* depends on both κ and σ:
//! δ* = (1/γ) × ln(1 + γ/κ)
//!
//! If κ and σ are correlated (they often are during volatility regimes),
//! the uncertainty in spread calculations should account for this correlation.

use super::tick_ewma::TickEWMA;

/// Tracks rolling covariance between two parameters using EWMA.
///
/// Maintains online estimates of:
/// - Mean of each parameter
/// - Variance of each parameter
/// - Covariance between parameters
/// - Correlation coefficient
#[derive(Debug, Clone)]
pub(crate) struct ParameterCovariance {
    /// EWMA of κ
    mean_kappa: f64,
    /// EWMA of σ
    mean_sigma: f64,
    /// EWMA of κ²
    mean_kappa_sq: f64,
    /// EWMA of σ²
    mean_sigma_sq: f64,
    /// EWMA of κ × σ
    mean_kappa_sigma: f64,
    /// EWMA decay factor
    alpha: f64,
    /// Observation count
    observation_count: usize,
    /// Minimum observations before covariance is valid
    min_observations: usize,
}

impl ParameterCovariance {
    /// Create a new covariance tracker.
    ///
    /// # Arguments
    /// * `alpha` - EWMA decay factor (e.g., 0.02 for 50-tick half-life)
    pub(crate) fn new(alpha: f64) -> Self {
        Self {
            mean_kappa: 0.0,
            mean_sigma: 0.0,
            mean_kappa_sq: 0.0,
            mean_sigma_sq: 0.0,
            mean_kappa_sigma: 0.0,
            alpha,
            observation_count: 0,
            min_observations: 20,
        }
    }

    /// Create with half-life in ticks.
    pub(crate) fn with_half_life(half_life_ticks: f64) -> Self {
        let alpha = 1.0 - 2.0_f64.powf(-1.0 / half_life_ticks);
        Self::new(alpha)
    }

    /// Update with new (κ, σ) observation.
    pub(crate) fn update(&mut self, kappa: f64, sigma: f64) {
        if self.observation_count == 0 {
            // Initialize with first observation
            self.mean_kappa = kappa;
            self.mean_sigma = sigma;
            self.mean_kappa_sq = kappa * kappa;
            self.mean_sigma_sq = sigma * sigma;
            self.mean_kappa_sigma = kappa * sigma;
        } else {
            // EWMA update
            self.mean_kappa = self.alpha * kappa + (1.0 - self.alpha) * self.mean_kappa;
            self.mean_sigma = self.alpha * sigma + (1.0 - self.alpha) * self.mean_sigma;
            self.mean_kappa_sq =
                self.alpha * (kappa * kappa) + (1.0 - self.alpha) * self.mean_kappa_sq;
            self.mean_sigma_sq =
                self.alpha * (sigma * sigma) + (1.0 - self.alpha) * self.mean_sigma_sq;
            self.mean_kappa_sigma =
                self.alpha * (kappa * sigma) + (1.0 - self.alpha) * self.mean_kappa_sigma;
        }
        self.observation_count += 1;
    }

    /// Get variance of κ.
    pub(crate) fn variance_kappa(&self) -> f64 {
        (self.mean_kappa_sq - self.mean_kappa.powi(2)).max(0.0)
    }

    /// Get variance of σ.
    pub(crate) fn variance_sigma(&self) -> f64 {
        (self.mean_sigma_sq - self.mean_sigma.powi(2)).max(0.0)
    }

    /// Get covariance Cov(κ, σ).
    pub(crate) fn covariance(&self) -> f64 {
        self.mean_kappa_sigma - self.mean_kappa * self.mean_sigma
    }

    /// Get correlation coefficient ρ(κ, σ) ∈ [-1, 1].
    pub(crate) fn correlation(&self) -> f64 {
        let var_k = self.variance_kappa();
        let var_s = self.variance_sigma();
        let denom = (var_k * var_s).sqrt();
        if denom < 1e-12 {
            return 0.0;
        }
        (self.covariance() / denom).clamp(-1.0, 1.0)
    }

    /// Get standard deviation of κ.
    pub(crate) fn std_kappa(&self) -> f64 {
        self.variance_kappa().sqrt()
    }

    /// Get standard deviation of σ.
    pub(crate) fn std_sigma(&self) -> f64 {
        self.variance_sigma().sqrt()
    }

    /// Get mean κ.
    pub(crate) fn mean_kappa(&self) -> f64 {
        self.mean_kappa
    }

    /// Get mean σ.
    pub(crate) fn mean_sigma(&self) -> f64 {
        self.mean_sigma
    }

    /// Check if covariance estimate is valid.
    pub(crate) fn is_valid(&self) -> bool {
        self.observation_count >= self.min_observations
    }

    /// Compute spread uncertainty from parameter uncertainty.
    ///
    /// Using delta method for δ* = (1/γ) × ln(1 + γ/κ):
    /// ∂δ*/∂κ ≈ -1/(κ(κ+γ)) for small γ/κ
    ///
    /// Var(δ*) ≈ (∂δ*/∂κ)² Var(κ)
    pub(crate) fn spread_uncertainty(&self, gamma: f64) -> f64 {
        let kappa = self.mean_kappa;
        if kappa < 1e-6 || gamma < 1e-12 {
            return 0.0;
        }

        // Derivative of GLFT spread w.r.t. kappa
        // δ* = (1/γ) × ln(1 + γ/κ)
        // dδ*/dκ = (1/γ) × (-γ/κ²) / (1 + γ/κ) = -1/(κ(κ + γ))
        let d_delta_d_kappa = -1.0 / (kappa * (kappa + gamma));

        // Standard error of spread
        (d_delta_d_kappa.powi(2) * self.variance_kappa()).sqrt()
    }

    /// Reset to initial state.
    pub(crate) fn reset(&mut self) {
        self.mean_kappa = 0.0;
        self.mean_sigma = 0.0;
        self.mean_kappa_sq = 0.0;
        self.mean_sigma_sq = 0.0;
        self.mean_kappa_sigma = 0.0;
        self.observation_count = 0;
    }
}

/// Extended covariance tracker for multiple parameters.
///
/// Tracks covariances among (κ, σ, λ_jump) for full uncertainty propagation.
#[derive(Debug, Clone)]
pub(crate) struct MultiParameterCovariance {
    /// (κ, σ) covariance
    kappa_sigma: ParameterCovariance,

    /// EWMA of λ (jump intensity)
    mean_lambda: TickEWMA,

    /// EWMA of λ × κ
    mean_lambda_kappa: TickEWMA,

    /// EWMA of λ × σ
    mean_lambda_sigma: TickEWMA,
}

impl MultiParameterCovariance {
    pub(crate) fn new(half_life_ticks: f64) -> Self {
        Self {
            kappa_sigma: ParameterCovariance::with_half_life(half_life_ticks),
            mean_lambda: TickEWMA::new_uninitialized(half_life_ticks),
            mean_lambda_kappa: TickEWMA::new_uninitialized(half_life_ticks),
            mean_lambda_sigma: TickEWMA::new_uninitialized(half_life_ticks),
        }
    }

    pub(crate) fn update(&mut self, kappa: f64, sigma: f64, lambda: f64) {
        self.kappa_sigma.update(kappa, sigma);
        self.mean_lambda.update(lambda);
        self.mean_lambda_kappa.update(lambda * kappa);
        self.mean_lambda_sigma.update(lambda * sigma);
    }

    pub(crate) fn correlation_kappa_sigma(&self) -> f64 {
        self.kappa_sigma.correlation()
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.kappa_sigma.is_valid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_positive_correlation() {
        let mut cov = ParameterCovariance::with_half_life(10.0);

        // Feed positively correlated data: high κ with high σ
        for i in 0..100 {
            let kappa = 500.0 + 100.0 * (i as f64 / 100.0);
            let sigma = 0.0001 + 0.00005 * (i as f64 / 100.0);
            cov.update(kappa, sigma);
        }

        assert!(cov.is_valid());
        assert!(
            cov.correlation() > 0.5,
            "Positively correlated data should have positive correlation"
        );
    }

    #[test]
    fn test_covariance_negative_correlation() {
        let mut cov = ParameterCovariance::with_half_life(10.0);

        // Feed negatively correlated data: high κ with low σ
        for i in 0..100 {
            let kappa = 500.0 + 100.0 * (i as f64 / 100.0);
            let sigma = 0.00015 - 0.00005 * (i as f64 / 100.0);
            cov.update(kappa, sigma);
        }

        assert!(cov.is_valid());
        assert!(
            cov.correlation() < -0.5,
            "Negatively correlated data should have negative correlation"
        );
    }

    #[test]
    fn test_covariance_uncorrelated() {
        let mut cov = ParameterCovariance::with_half_life(20.0);

        // Feed uncorrelated data (κ oscillates, σ is random-ish)
        for i in 0..200 {
            let kappa = 500.0 + 50.0 * ((i as f64 * 0.3).sin());
            let sigma = 0.0001 + 0.00002 * ((i as f64 * 0.7).cos());
            cov.update(kappa, sigma);
        }

        assert!(cov.is_valid());
        // Should be roughly uncorrelated (close to 0)
        assert!(
            cov.correlation().abs() < 0.5,
            "Uncorrelated data should have near-zero correlation"
        );
    }

    #[test]
    fn test_spread_uncertainty() {
        let mut cov = ParameterCovariance::with_half_life(10.0);

        // Feed data with variance in kappa
        for i in 0..100 {
            let kappa = 500.0 + 50.0 * ((i as f64 * 0.5).sin());
            cov.update(kappa, 0.0001);
        }

        let spread_std = cov.spread_uncertainty(0.3);
        assert!(spread_std > 0.0, "Should have positive spread uncertainty");
        assert!(
            spread_std < 0.01,
            "Spread uncertainty should be reasonable"
        );
    }

    #[test]
    fn test_variance_calculation() {
        let mut cov = ParameterCovariance::new(0.1);

        // Feed constant data - variance should be 0
        for _ in 0..50 {
            cov.update(500.0, 0.0001);
        }

        assert!(
            cov.variance_kappa() < 1e-6,
            "Constant data should have near-zero variance"
        );
        assert!(
            cov.variance_sigma() < 1e-12,
            "Constant data should have near-zero variance"
        );
    }
}
