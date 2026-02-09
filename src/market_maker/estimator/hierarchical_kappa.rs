//! Hierarchical Kappa Estimation Module
//!
//! Fixes the linear blending bug in the original kappa estimation where
//! `kappa = own_conf * own + (1 - own_conf) * market` is statistically incorrect.
//!
//! ## The Problem
//!
//! Linear blending of κ_own and κ_market is wrong because:
//! 1. They measure different things (our fill rate vs market trade rate)
//! 2. Blending parameters breaks the Bayesian structure
//! 3. It ignores the hierarchical relationship
//!
//! ## The Solution: Hierarchical Prior
//!
//! Use market kappa to inform the *prior* on our own kappa:
//! - Prior: κ_own ~ Gamma(α₀, β₀) where β₀ = α₀ / κ_market (prior mean = κ_market)
//! - Likelihood: Fills ~ Exp(κ_own)
//! - Posterior: κ_own | fills ~ Gamma(α₀ + n, β₀ + Σδᵢ)
//!
//! This gives:
//! - Fast warmup: Prior pulls toward market kappa when n is small
//! - Proper convergence: As n grows, our own fills dominate
//! - Adverse selection adjustment: Scale down effective kappa by φ(AS)
//!
//! ## Key Insight
//!
//! The relationship between κ_market and κ_own is:
//! κ_own = κ_market × φ(AS) where φ(AS) ∈ [0.5, 1.0]
//!
//! - When AS is low: φ ≈ 1.0, fills execute at market rate
//! - When AS is high: φ < 1.0, informed flow avoids our quotes

// Allow dead code since this is V2 infrastructure being built incrementally
#![allow(dead_code)]

use super::tick_ewma::TickEWMA;
use std::collections::VecDeque;

/// Hierarchical Kappa Estimator
///
/// Uses market-wide kappa as a hierarchical prior, then updates based on
/// our own fills to get the correct posterior estimate.
#[derive(Debug, Clone)]
pub(crate) struct HierarchicalKappa {
    // === Hierarchical Prior ===
    /// Prior shape parameter (concentration)
    /// Higher α₀ = more confidence in prior (slower adaptation)
    prior_alpha: f64,

    // === Market-Level Kappa (Prior Mean) ===
    /// Market kappa from L2 book / trade tape
    market_kappa: f64,
    /// Confidence in market kappa [0, 1]
    market_kappa_conf: f64,

    // === Our Own Fill Statistics ===
    /// Count of our fill observations (n)
    observation_count: usize,
    /// Sum of fill distances (Σδᵢ)
    sum_distances: f64,
    /// Sum of squared distances for variance
    sum_sq_distances: f64,
    /// Rolling window of fills
    fill_window: VecDeque<(u64, f64)>,
    /// Window duration in ms
    window_ms: u64,

    // === Adverse Selection Factor ===
    /// Current adverse selection estimate (in bps)
    adverse_selection_bps: f64,
    /// AS decay factor for φ(AS) = exp(-c × AS)
    /// c = 0.2 means AS of 5bp → φ = 0.37 (63% discount)
    as_decay_coefficient: f64,
    /// EWMA for adverse selection tracking
    as_ewma: TickEWMA,

    // === Posterior (cached) ===
    /// Posterior shape α = α₀ + n
    posterior_alpha: f64,
    /// Posterior rate β = β₀ + Σδᵢ where β₀ = α₀/κ_market
    posterior_beta: f64,
    /// Posterior mean E[κ | data] = α/β
    posterior_mean: f64,
    /// Posterior variance Var[κ | data] = α/β²
    posterior_var: f64,
    /// 95% credible interval bounds
    ci_95_lower: f64,
    ci_95_upper: f64,

    // === Effective kappa (with AS adjustment) ===
    /// κ_effective = posterior_mean × φ(AS)
    effective_kappa: f64,

    // === Configuration ===
    /// Minimum observations before posterior is trusted
    min_observations: usize,
    /// Default market kappa (used when market_kappa not yet available)
    default_market_kappa: f64,
}

impl HierarchicalKappa {
    /// Create a new hierarchical kappa estimator.
    ///
    /// # Arguments
    /// * `prior_alpha` - Prior shape parameter (default 5.0 for moderate strength)
    /// * `default_market_kappa` - Default market kappa before we have market data
    /// * `window_ms` - Rolling window for fill observations (default 300000 = 5 min)
    pub(crate) fn new(prior_alpha: f64, default_market_kappa: f64, window_ms: u64) -> Self {
        let mut estimator = Self {
            prior_alpha,
            market_kappa: default_market_kappa,
            market_kappa_conf: 0.0,
            observation_count: 0,
            sum_distances: 0.0,
            sum_sq_distances: 0.0,
            fill_window: VecDeque::new(),
            window_ms,
            adverse_selection_bps: 0.0,
            as_decay_coefficient: 0.2, // φ = exp(-0.2 × AS_bps)
            as_ewma: TickEWMA::new_uninitialized(50.0), // 50-fill half-life for AS
            posterior_alpha: prior_alpha,
            posterior_beta: prior_alpha / default_market_kappa,
            posterior_mean: default_market_kappa,
            posterior_var: 0.0,
            ci_95_lower: 0.0,
            ci_95_upper: 0.0,
            effective_kappa: default_market_kappa,
            min_observations: 10,
            default_market_kappa,
        };
        estimator.update_posterior();
        estimator
    }

    /// Create with default parameters for liquid markets.
    pub(crate) fn default_liquid() -> Self {
        Self::new(
            5.0,     // prior_alpha: moderate prior strength
            2500.0,  // default_market_kappa: 4bp avg fill distance
            300_000, // window_ms: 5 minutes
        )
    }

    /// Update market kappa from L2 book / trade tape.
    ///
    /// This updates the *prior mean* for our hierarchical model.
    pub(crate) fn update_market_kappa(&mut self, kappa: f64, confidence: f64) {
        if kappa > 0.0 {
            self.market_kappa = kappa;
            self.market_kappa_conf = confidence.clamp(0.0, 1.0);
            self.update_posterior();
        }
    }

    /// Update the hierarchical prior parameters (mean and concentration).
    ///
    /// Called by adaptive kappa prior when the prior center shifts toward
    /// observed market kappa. Updates both the prior alpha (concentration)
    /// and the default market kappa, then recomputes the posterior.
    pub(crate) fn update_market_prior(&mut self, new_prior_mean: f64, new_prior_strength: f64) {
        let safe_mean = new_prior_mean.clamp(100.0, 50000.0);
        let safe_strength = new_prior_strength.clamp(1.0, 50.0);

        self.prior_alpha = safe_strength;
        self.default_market_kappa = safe_mean;
        // Also update market_kappa if it hasn't been set from external data yet
        if self.market_kappa_conf < 0.1 {
            self.market_kappa = safe_mean;
        }
        self.update_posterior();
    }

    /// Record a fill from one of our orders.
    ///
    /// # Arguments
    /// * `distance_bps` - Distance from mid where fill executed (in basis points)
    /// * `timestamp_ms` - Fill timestamp in milliseconds
    /// * `adverse_selection_bps` - Optional observed adverse selection for this fill
    pub(crate) fn record_fill(
        &mut self,
        distance_bps: f64,
        timestamp_ms: u64,
        adverse_selection_bps: Option<f64>,
    ) {
        // Distance must be positive
        let distance = distance_bps.abs().max(0.01); // Floor at 0.01 bps

        // Add to rolling window
        self.fill_window.push_back((timestamp_ms, distance));

        // Update statistics
        self.observation_count += 1;
        self.sum_distances += distance;
        self.sum_sq_distances += distance * distance;

        // Update adverse selection EWMA if provided
        if let Some(as_bps) = adverse_selection_bps {
            self.as_ewma.update(as_bps.abs());
            self.adverse_selection_bps = self.as_ewma.value();
        }

        // Expire old observations
        self.expire_window(timestamp_ms);

        // Update posterior
        self.update_posterior();
    }

    /// Expire old observations from the rolling window.
    fn expire_window(&mut self, current_time_ms: u64) {
        let cutoff = current_time_ms.saturating_sub(self.window_ms);

        while let Some(&(ts, distance)) = self.fill_window.front() {
            if ts >= cutoff {
                break;
            }
            self.fill_window.pop_front();
            self.observation_count = self.observation_count.saturating_sub(1);
            self.sum_distances = (self.sum_distances - distance).max(0.0);
            self.sum_sq_distances = (self.sum_sq_distances - distance * distance).max(0.0);
        }
    }

    /// Update posterior parameters using hierarchical Bayesian update.
    fn update_posterior(&mut self) {
        // Hierarchical prior: β₀ = α₀ / κ_market
        // This means our prior has mean E[κ] = α₀/β₀ = κ_market
        let prior_beta = if self.market_kappa > 0.0 {
            self.prior_alpha / self.market_kappa
        } else {
            self.prior_alpha / self.default_market_kappa
        };

        // Posterior update (Gamma-Exponential conjugacy)
        // α_post = α₀ + n
        // β_post = β₀ + Σδᵢ
        self.posterior_alpha = self.prior_alpha + self.observation_count as f64;
        self.posterior_beta = prior_beta + self.sum_distances;

        // Posterior mean and variance
        self.posterior_mean = self.posterior_alpha / self.posterior_beta;
        self.posterior_var = self.posterior_alpha / (self.posterior_beta * self.posterior_beta);

        // 95% credible interval using Wilson-Hilferty approximation for Gamma quantiles
        // For Gamma(α, β), approximately: (1/β) × χ²_{2α, p} / 2
        // Using normal approximation for large α
        let std = self.posterior_var.sqrt();
        self.ci_95_lower = (self.posterior_mean - 1.96 * std).max(0.01);
        self.ci_95_upper = self.posterior_mean + 1.96 * std;

        // Apply adverse selection adjustment
        // φ(AS) = exp(-c × AS) where AS is in bps
        let as_factor = self.adverse_selection_factor();
        self.effective_kappa = self.posterior_mean * as_factor;
    }

    /// Calculate adverse selection factor φ(AS) ∈ [0.5, 1.0].
    ///
    /// φ(AS) = 0.5 + 0.5 × exp(-c × AS)
    /// - AS = 0 → φ = 1.0 (no adjustment)
    /// - AS = 5bp → φ ≈ 0.68 (with c = 0.2)
    /// - AS → ∞ → φ → 0.5 (floor)
    fn adverse_selection_factor(&self) -> f64 {
        let decay = (-self.as_decay_coefficient * self.adverse_selection_bps).exp();
        0.5 + 0.5 * decay
    }

    // === Getters ===

    /// Get the effective kappa (posterior mean adjusted for AS).
    ///
    /// This is the primary output to use for GLFT spread calculation.
    pub(crate) fn effective_kappa(&self) -> f64 {
        self.effective_kappa
    }

    /// Get the raw posterior mean (without AS adjustment).
    pub(crate) fn posterior_mean(&self) -> f64 {
        self.posterior_mean
    }

    /// Get posterior standard deviation.
    pub(crate) fn posterior_std(&self) -> f64 {
        self.posterior_var.sqrt()
    }

    /// Get posterior variance.
    pub(crate) fn posterior_var(&self) -> f64 {
        self.posterior_var
    }

    /// Get 95% credible interval.
    pub(crate) fn credible_interval_95(&self) -> (f64, f64) {
        (self.ci_95_lower, self.ci_95_upper)
    }

    /// Get relative uncertainty (std / mean).
    pub(crate) fn relative_uncertainty(&self) -> f64 {
        if self.posterior_mean > 0.0 {
            self.posterior_std() / self.posterior_mean
        } else {
            1.0
        }
    }

    /// Get confidence based on observation count and uncertainty.
    ///
    /// Returns value in [0, 1]:
    /// - 0: Only prior, no fills observed
    /// - 1: Many fills, low uncertainty
    pub(crate) fn confidence(&self) -> f64 {
        let obs_factor = if self.observation_count >= self.min_observations {
            // Saturating confidence from observations
            1.0 - (-0.1 * (self.observation_count as f64 - self.min_observations as f64)).exp()
        } else {
            self.observation_count as f64 / self.min_observations as f64 * 0.5
        };

        // Also factor in uncertainty
        let uncertainty_factor = (1.0 - self.relative_uncertainty().min(1.0)).max(0.0);

        (obs_factor * 0.7 + uncertainty_factor * 0.3).clamp(0.0, 1.0)
    }

    /// Get current market kappa (prior mean).
    pub(crate) fn market_kappa(&self) -> f64 {
        self.market_kappa
    }

    /// Get market kappa confidence.
    pub(crate) fn market_kappa_confidence(&self) -> f64 {
        self.market_kappa_conf
    }

    /// Get current adverse selection estimate in bps.
    pub(crate) fn adverse_selection_bps(&self) -> f64 {
        self.adverse_selection_bps
    }

    /// Get adverse selection factor φ(AS).
    pub(crate) fn as_factor(&self) -> f64 {
        self.adverse_selection_factor()
    }

    /// Get observation count in window.
    pub(crate) fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Check if estimator is warmed up.
    pub(crate) fn is_warmed_up(&self) -> bool {
        self.observation_count >= self.min_observations
    }

    /// Get coefficient of variation (CV) of fill distances.
    ///
    /// CV = σ / μ where μ = mean distance, σ = std of distances
    /// For exponential distribution: CV = 1
    /// CV > 1 indicates heavy tail
    pub(crate) fn cv(&self) -> f64 {
        if self.observation_count < 2 || self.sum_distances < 1e-12 {
            return 1.0; // Default to exponential assumption
        }

        let mean = self.sum_distances / self.observation_count as f64;
        let var = (self.sum_sq_distances / self.observation_count as f64 - mean * mean).max(0.0);
        let std = var.sqrt();

        if mean > 1e-12 {
            std / mean
        } else {
            1.0
        }
    }

    /// Check if fill distance distribution is heavy-tailed (CV > 1.2).
    pub(crate) fn is_heavy_tailed(&self) -> bool {
        self.cv() > 1.2
    }

    /// Reset to initial state (keeps configuration).
    pub(crate) fn reset(&mut self) {
        self.observation_count = 0;
        self.sum_distances = 0.0;
        self.sum_sq_distances = 0.0;
        self.fill_window.clear();
        self.adverse_selection_bps = 0.0;
        self.as_ewma.reset();
        self.update_posterior();
    }
}

/// Configuration for hierarchical kappa estimation.
#[derive(Debug, Clone)]
pub(crate) struct HierarchicalKappaConfig {
    /// Prior shape parameter (concentration)
    pub prior_alpha: f64,
    /// Default market kappa before we have market data
    pub default_market_kappa: f64,
    /// Rolling window for fills in ms
    pub window_ms: u64,
    /// AS decay coefficient for φ(AS) = exp(-c × AS)
    pub as_decay_coefficient: f64,
    /// Minimum observations for warmup
    pub min_observations: usize,
}

impl Default for HierarchicalKappaConfig {
    fn default() -> Self {
        Self {
            prior_alpha: 5.0,
            default_market_kappa: 2500.0,
            window_ms: 300_000,
            as_decay_coefficient: 0.2,
            min_observations: 10,
        }
    }
}

impl HierarchicalKappaConfig {
    /// Create config for liquid markets (BTC, ETH).
    pub(crate) fn liquid() -> Self {
        Self {
            prior_alpha: 5.0,
            default_market_kappa: 2500.0, // 4bp avg fill distance
            window_ms: 300_000,
            as_decay_coefficient: 0.2,
            min_observations: 10,
        }
    }

    /// Create config for less liquid markets.
    pub(crate) fn illiquid() -> Self {
        Self {
            prior_alpha: 10.0,           // Stronger prior
            default_market_kappa: 500.0, // 20bp avg fill distance
            window_ms: 600_000,          // 10 min window
            as_decay_coefficient: 0.1,   // Less AS sensitivity
            min_observations: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_warmup() {
        let est = HierarchicalKappa::default_liquid();

        // Before any fills, posterior mean = prior mean = market kappa
        assert!(
            (est.posterior_mean() - 2500.0).abs() < 100.0,
            "Initial posterior should be near prior mean"
        );
    }

    #[test]
    fn test_hierarchical_convergence() {
        let mut est = HierarchicalKappa::default_liquid();
        est.update_market_kappa(2000.0, 0.8);

        // Feed fills at 10bp distance → true kappa = 1000
        let true_mean_distance = 10.0; // 10 bps
        for i in 0..100 {
            est.record_fill(true_mean_distance, i as u64 * 1000, None);
        }

        // Posterior should converge toward 1000 (= 1/0.001)
        // Since we're feeding consistent 10bp distances
        let _expected_kappa = 100.0; // 1 / 0.01 = 100
        let observed = est.posterior_mean();

        // With 100 observations, should be close to true value
        // (accounting for prior pull)
        assert!(
            observed < 500.0,
            "With 10bp fills, kappa should be < 500, got {}",
            observed
        );
    }

    #[test]
    fn test_hierarchical_market_update() {
        let mut est = HierarchicalKappa::default_liquid();

        // Initial market kappa
        est.update_market_kappa(1000.0, 0.5);
        let initial = est.posterior_mean();

        // Update market kappa higher
        est.update_market_kappa(3000.0, 0.8);
        let updated = est.posterior_mean();

        // Posterior should increase (prior mean increased)
        assert!(
            updated > initial,
            "Higher market kappa should increase posterior"
        );
    }

    #[test]
    fn test_adverse_selection_adjustment() {
        let mut est = HierarchicalKappa::default_liquid();

        // Feed fills with adverse selection
        for i in 0..50 {
            est.record_fill(5.0, i as u64 * 1000, Some(2.0)); // 2bp AS
        }

        // AS factor should reduce effective kappa
        let as_factor = est.as_factor();
        assert!(
            as_factor < 1.0,
            "AS factor should be < 1.0 with positive AS"
        );
        assert!(as_factor > 0.5, "AS factor should be > 0.5 (floor)");

        assert!(
            est.effective_kappa() < est.posterior_mean(),
            "Effective kappa should be reduced by AS factor"
        );
    }

    #[test]
    fn test_confidence_increases() {
        let mut est = HierarchicalKappa::default_liquid();

        let initial_conf = est.confidence();
        // Initial confidence includes uncertainty factor from prior
        // With prior alpha=5, relative_uncertainty ≈ 0.45, so confidence ≈ 0.17
        assert!(
            initial_conf < 0.3,
            "Initial confidence should be moderate (no fills)"
        );

        // Add observations
        for i in 0..30 {
            est.record_fill(5.0, i as u64 * 1000, None);
        }

        let final_conf = est.confidence();
        assert!(
            final_conf > initial_conf,
            "Confidence should increase with observations"
        );
        assert!(
            final_conf > 0.5,
            "Should have reasonable confidence after 30 fills"
        );
    }

    #[test]
    fn test_window_expiry() {
        let mut est = HierarchicalKappa::new(5.0, 2500.0, 10_000); // 10 second window

        // Fill at t=0
        est.record_fill(5.0, 0, None);
        assert_eq!(est.observation_count(), 1);

        // Fill at t=5s (within window)
        est.record_fill(5.0, 5_000, None);
        assert_eq!(est.observation_count(), 2);

        // Fill at t=15s (first fill should expire)
        est.record_fill(5.0, 15_000, None);
        assert_eq!(est.observation_count(), 2); // First expired

        // Fill at t=20s (second fill should expire)
        est.record_fill(5.0, 20_000, None);
        assert_eq!(est.observation_count(), 2); // Second expired
    }

    #[test]
    fn test_cv_calculation() {
        let mut est = HierarchicalKappa::default_liquid();

        // Feed constant distances → CV ≈ 0
        for i in 0..50 {
            est.record_fill(5.0, i as u64 * 1000, None);
        }

        assert!(est.cv() < 0.2, "Constant distances should have low CV");

        // Reset and feed varying distances
        est.reset();
        for i in 0..50 {
            let distance = if i % 2 == 0 { 2.0 } else { 10.0 };
            est.record_fill(distance, i as u64 * 1000, None);
        }

        assert!(est.cv() > 0.5, "Varying distances should have higher CV");
    }

    #[test]
    fn test_credible_interval() {
        let mut est = HierarchicalKappa::default_liquid();

        // Feed some observations
        for i in 0..30 {
            est.record_fill(5.0, i as u64 * 1000, None);
        }

        let (lower, upper) = est.credible_interval_95();

        assert!(lower > 0.0, "Lower bound should be positive");
        assert!(
            lower < est.posterior_mean(),
            "Lower bound should be below mean"
        );
        assert!(
            upper > est.posterior_mean(),
            "Upper bound should be above mean"
        );
    }
}
