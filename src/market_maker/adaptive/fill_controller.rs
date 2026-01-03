//! Fill Rate Controller - Component 4
//!
//! Implements a Gamma-Poisson conjugate model for fill rate estimation and
//! provides a spread ceiling to ensure minimum fill activity.
//!
//! # Mathematical Model
//!
//! Target a minimum fill rate ρ_target (fills per second).
//!
//! ## Fill Rate Model
//! ```text
//! fills_in_Δt ~ Poisson(ρ × Δt)
//! ρ ~ Gamma(α, β)  # Conjugate prior
//!
//! Posterior after observing k fills in time Δt:
//! α_post = α + k
//! β_post = β + Δt
//! E[ρ] = α_post / β_post
//! ```
//!
//! ## Spread Ceiling
//! ```text
//! ρ(δ) = ρ_0 × exp(-κ × δ)    # Fill rate decays with spread
//! δ_target = (1/κ) × ln(ρ_0 / ρ_target)
//! ```
//!
//! # Purpose
//!
//! If we're quoting so wide that we never get fills, we make no profit.
//! The fill controller provides a soft ceiling on spread to ensure minimum activity.

use tracing::debug;

/// Fill rate controller using Gamma-Poisson conjugate model.
#[derive(Debug, Clone)]
pub struct FillRateController {
    /// Target fill rate (fills per second)
    target_fill_rate: f64,

    /// Gamma posterior shape parameter (α)
    alpha: f64,

    /// Gamma posterior rate parameter (β)
    beta: f64,

    /// Prior shape parameter
    prior_alpha: f64,

    /// Prior rate parameter
    prior_beta: f64,

    /// Current kappa estimate (from BlendedKappaEstimator)
    current_kappa: f64,

    /// Ceiling multiplier (allow spread to exceed fill-optimal by this factor)
    ceiling_mult: f64,

    /// Minimum observation time before controller activates (seconds)
    min_observation_time: f64,

    /// Total observation time (seconds)
    observation_time: f64,

    /// Total fills observed
    total_fills: usize,

    /// EWMA decay for non-stationarity
    decay: f64,

    /// Last logged fill rate (for change detection)
    last_logged_rate: f64,
}

impl FillRateController {
    /// Create a new fill rate controller.
    ///
    /// # Arguments
    /// * `target_fill_rate` - Target fills per second (e.g., 0.02 = 1 per 50s)
    /// * `ceiling_mult` - Multiplier on target spread (e.g., 1.5 allows 50% wider)
    /// * `min_observation_time` - Seconds before controller activates
    /// * `decay` - EWMA decay factor
    pub fn new(
        target_fill_rate: f64,
        ceiling_mult: f64,
        min_observation_time: f64,
        decay: f64,
    ) -> Self {
        // Prior: expect 1 fill per 60 seconds initially
        let prior_alpha = 1.0;
        let prior_beta = 60.0;

        Self {
            target_fill_rate,
            alpha: prior_alpha,
            beta: prior_beta,
            prior_alpha,
            prior_beta,
            current_kappa: 500.0, // Default until updated
            ceiling_mult,
            min_observation_time,
            observation_time: 0.0,
            total_fills: 0,
            decay,
            last_logged_rate: 0.0,
        }
    }

    /// Create from config.
    pub fn from_config(config: &super::AdaptiveBayesianConfig) -> Self {
        Self::new(
            config.target_fill_rate,
            config.fill_ceiling_mult,
            config.fill_min_observation_secs,
            config.fill_rate_decay,
        )
    }

    /// Update after a time period with observed fills.
    ///
    /// # Arguments
    /// * `fills` - Number of fills in this period
    /// * `elapsed_secs` - Duration of this period in seconds
    /// * `kappa` - Current kappa estimate (for spread calculation)
    pub fn update(&mut self, fills: usize, elapsed_secs: f64, kappa: f64) {
        if elapsed_secs <= 0.0 {
            return;
        }

        self.observation_time += elapsed_secs;
        self.total_fills += fills;
        self.current_kappa = kappa;

        // EWMA update for non-stationarity
        // decay^t gives weight to old observations
        self.alpha = self.decay * self.alpha + fills as f64;
        self.beta = self.decay * self.beta + elapsed_secs;

        // Ensure minimum values (regularization toward prior)
        self.alpha = self.alpha.max(self.prior_alpha * 0.1);
        self.beta = self.beta.max(self.prior_beta * 0.1);

        // Log significant changes
        let current_rate = self.observed_fill_rate();
        let rate_change = (current_rate - self.last_logged_rate).abs();
        if rate_change > 0.01 || self.total_fills.is_multiple_of(10) && self.total_fills > 0 {
            debug!(
                total_fills = self.total_fills,
                observation_time = self.observation_time,
                fill_rate = current_rate,
                target_rate = self.target_fill_rate,
                ceiling = ?self.spread_ceiling(),
                "Fill rate controller updated"
            );
            self.last_logged_rate = current_rate;
        }
    }

    /// Get the observed fill rate (Gamma posterior mean).
    pub fn observed_fill_rate(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Get the spread ceiling based on fill rate target.
    ///
    /// Returns `None` if:
    /// - Not enough observation time
    /// - Already meeting fill rate target
    ///
    /// Returns `Some(δ_ceiling)` if fill rate is below target.
    pub fn spread_ceiling(&self) -> Option<f64> {
        // Don't activate until we have enough observation time
        if self.observation_time < self.min_observation_time {
            return None;
        }

        let rho_observed = self.observed_fill_rate();

        // If we're meeting or exceeding target, no ceiling needed
        if rho_observed >= self.target_fill_rate {
            return None;
        }

        // Estimate base fill rate (at zero spread)
        // This is a rough approximation: ρ_0 ≈ κ × ρ_observed
        // Rationale: if ρ(δ) = ρ_0 × exp(-κδ), then ρ_0 = ρ(0)
        // We don't know current δ, but we assume it's approximately 1/κ (one unit of spread)
        // So ρ_observed ≈ ρ_0 × exp(-1) → ρ_0 ≈ ρ_observed × e
        let rho_0 = rho_observed * std::f64::consts::E * 2.0; // Generous estimate

        // Target spread: δ = (1/κ) × ln(ρ_0 / ρ_target)
        if rho_0 <= self.target_fill_rate || self.current_kappa <= 0.0 {
            // Can't achieve target even at zero spread
            return Some(0.0001); // Very tight ceiling
        }

        let delta_target = (rho_0 / self.target_fill_rate).ln() / self.current_kappa;

        // Apply ceiling multiplier (allow some buffer)
        Some(delta_target.max(0.0) * self.ceiling_mult)
    }

    /// Get a soft adjustment factor based on fill rate.
    ///
    /// Returns a multiplier ∈ (0, 1] to apply to spread:
    /// - 1.0 if meeting fill target
    /// - < 1.0 if below target (tighten)
    ///
    /// This is an alternative to hard ceiling for smoother adjustment.
    pub fn spread_adjustment_factor(&self) -> f64 {
        if self.observation_time < self.min_observation_time {
            return 1.0;
        }

        let rho_observed = self.observed_fill_rate();

        if rho_observed >= self.target_fill_rate {
            1.0 // No adjustment needed
        } else if rho_observed <= 0.0 {
            0.5 // Very aggressive tightening
        } else {
            // Smooth adjustment: sqrt(observed / target)
            (rho_observed / self.target_fill_rate).sqrt().min(1.0)
        }
    }

    /// Check if the controller is active (has enough data).
    pub fn is_active(&self) -> bool {
        self.observation_time >= self.min_observation_time
    }

    /// Check if we're meeting the fill rate target.
    pub fn is_meeting_target(&self) -> bool {
        self.is_active() && self.observed_fill_rate() >= self.target_fill_rate
    }

    /// Get total observation time.
    pub fn observation_time(&self) -> f64 {
        self.observation_time
    }

    /// Get total fills observed.
    pub fn total_fills(&self) -> usize {
        self.total_fills
    }

    /// Get target fill rate.
    pub fn target_fill_rate(&self) -> f64 {
        self.target_fill_rate
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.alpha = self.prior_alpha;
        self.beta = self.prior_beta;
        self.observation_time = 0.0;
        self.total_fills = 0;
    }

    /// Update kappa estimate.
    pub fn set_kappa(&mut self, kappa: f64) {
        self.current_kappa = kappa;
    }

    /// Get posterior uncertainty (coefficient of variation).
    pub fn uncertainty(&self) -> f64 {
        // For Gamma(α, β), CV = 1/√α
        1.0 / self.alpha.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_controller() -> FillRateController {
        FillRateController::new(
            0.02,  // target: 1 fill per 50 seconds
            1.5,   // ceiling_mult
            120.0, // 2 minutes warmup
            0.995, // decay
        )
    }

    #[test]
    fn test_initial_state() {
        let ctrl = default_controller();

        assert!(!ctrl.is_active());
        assert_eq!(ctrl.total_fills(), 0);
        assert!(ctrl.spread_ceiling().is_none());
    }

    #[test]
    fn test_fill_rate_estimation() {
        let mut ctrl = default_controller();

        // Simulate 2 minutes with 2 fills
        ctrl.update(2, 120.0, 500.0);

        // Should be active now
        assert!(ctrl.is_active());

        // Fill rate should be approximately 2/120 = 0.0167
        let rate = ctrl.observed_fill_rate();
        assert!(
            rate > 0.01 && rate < 0.03,
            "Fill rate should be ~0.017, got {}",
            rate
        );
    }

    #[test]
    fn test_ceiling_when_below_target() {
        let mut ctrl = default_controller();

        // Simulate 5 minutes with only 1 fill (below target)
        ctrl.update(1, 300.0, 500.0);

        // Should provide a ceiling
        let ceiling = ctrl.spread_ceiling();
        assert!(
            ceiling.is_some(),
            "Should provide ceiling when below target"
        );

        let c = ceiling.unwrap();
        assert!(c > 0.0, "Ceiling should be positive: {}", c);
    }

    #[test]
    fn test_no_ceiling_when_meeting_target() {
        let mut ctrl = default_controller();

        // Simulate 2 minutes with many fills (above target)
        ctrl.update(10, 120.0, 500.0);

        // Should not provide ceiling
        assert!(
            ctrl.spread_ceiling().is_none(),
            "Should not provide ceiling when meeting target"
        );
    }

    #[test]
    fn test_ewma_decay() {
        let mut ctrl = default_controller();

        // First period: high fill rate
        ctrl.update(10, 60.0, 500.0);
        let rate_after_high = ctrl.observed_fill_rate();

        // Second period: zero fills
        ctrl.update(0, 120.0, 500.0);
        let rate_after_low = ctrl.observed_fill_rate();

        // Rate should decrease but not to zero (EWMA smoothing)
        assert!(
            rate_after_low < rate_after_high,
            "Rate should decrease: {} -> {}",
            rate_after_high,
            rate_after_low
        );
        assert!(
            rate_after_low > 0.0,
            "Rate should not go to zero due to EWMA"
        );
    }

    #[test]
    fn test_adjustment_factor() {
        let mut ctrl = default_controller();

        // Simulate low fill rate
        ctrl.update(1, 300.0, 500.0);

        let factor = ctrl.spread_adjustment_factor();
        assert!(
            factor < 1.0,
            "Factor should be < 1 when below target: {}",
            factor
        );
        assert!(factor > 0.0, "Factor should be positive: {}", factor);
    }

    #[test]
    fn test_meeting_target_detection() {
        let mut ctrl = default_controller();

        // Below target
        ctrl.update(1, 120.0, 500.0);
        assert!(!ctrl.is_meeting_target());

        // Above target
        ctrl.update(20, 60.0, 500.0);
        assert!(ctrl.is_meeting_target());
    }
}
