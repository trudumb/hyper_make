//! Parameter change rate limiters for convergence stability.
//!
//! Prevents feedback loop cascades where rapid parameter changes amplify noise
//! into runaway spread widening or aggressive quoting.
//!
//! Rate limits:
//! - Gamma: max 20% change per minute (risk aversion changes slowly)
//! - Sigma: max 30% change per minute (allow fast response to vol spikes)
//! - Kappa: max 15% change per minute (fill intensity changes slowly)

use serde::{Deserialize, Serialize};

/// Configuration for parameter rate limiting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterConfig {
    /// Maximum fractional change per minute for gamma (0.20 = 20%)
    #[serde(default = "default_gamma_max_change_per_min")]
    pub gamma_max_change_per_min: f64,
    /// Maximum fractional change per minute for sigma (0.30 = 30%)
    #[serde(default = "default_sigma_max_change_per_min")]
    pub sigma_max_change_per_min: f64,
    /// Maximum fractional change per minute for kappa (0.15 = 15%)
    #[serde(default = "default_kappa_max_change_per_min")]
    pub kappa_max_change_per_min: f64,
}

fn default_gamma_max_change_per_min() -> f64 {
    0.20
}
fn default_sigma_max_change_per_min() -> f64 {
    0.30
}
fn default_kappa_max_change_per_min() -> f64 {
    0.15
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            gamma_max_change_per_min: default_gamma_max_change_per_min(),
            sigma_max_change_per_min: default_sigma_max_change_per_min(),
            kappa_max_change_per_min: default_kappa_max_change_per_min(),
        }
    }
}

/// Tracks parameter values and limits their rate of change.
///
/// Uses exponential moving rate tracking per parameter with hard clamps
/// on update magnitude.
#[derive(Debug, Clone)]
pub struct ParameterRateLimiter {
    config: RateLimiterConfig,
    /// Last accepted gamma value
    last_gamma: f64,
    /// Last accepted sigma value
    last_sigma: f64,
    /// Last accepted kappa value
    last_kappa: f64,
    /// Timestamp of last gamma update (seconds since some epoch)
    last_gamma_time_s: f64,
    /// Timestamp of last sigma update
    last_sigma_time_s: f64,
    /// Timestamp of last kappa update
    last_kappa_time_s: f64,
    /// Whether the limiter has been initialized with first values
    initialized: bool,
}

impl ParameterRateLimiter {
    pub fn new(config: RateLimiterConfig) -> Self {
        Self {
            config,
            last_gamma: 0.0,
            last_sigma: 0.0,
            last_kappa: 0.0,
            last_gamma_time_s: 0.0,
            last_sigma_time_s: 0.0,
            last_kappa_time_s: 0.0,
            initialized: false,
        }
    }

    /// Apply rate limiting to a new gamma value.
    ///
    /// Returns the clamped gamma (always positive, respects max change rate).
    pub fn limit_gamma(&mut self, proposed: f64, now_s: f64) -> f64 {
        if !self.initialized || self.last_gamma <= 0.0 {
            self.last_gamma = proposed.max(0.01);
            self.last_gamma_time_s = now_s;
            return self.last_gamma;
        }
        let result = self.limit_param(
            proposed,
            self.last_gamma,
            self.last_gamma_time_s,
            now_s,
            self.config.gamma_max_change_per_min,
        );
        self.last_gamma = result.max(0.01); // gamma > 0 invariant
        self.last_gamma_time_s = now_s;
        self.last_gamma
    }

    /// Apply rate limiting to a new sigma value.
    ///
    /// Returns the clamped sigma (always positive, allows faster changes for vol spikes).
    pub fn limit_sigma(&mut self, proposed: f64, now_s: f64) -> f64 {
        if !self.initialized || self.last_sigma <= 0.0 {
            self.last_sigma = proposed.max(1e-9);
            self.last_sigma_time_s = now_s;
            return self.last_sigma;
        }
        let result = self.limit_param(
            proposed,
            self.last_sigma,
            self.last_sigma_time_s,
            now_s,
            self.config.sigma_max_change_per_min,
        );
        self.last_sigma = result.max(1e-9);
        self.last_sigma_time_s = now_s;
        self.last_sigma
    }

    /// Apply rate limiting to a new kappa value.
    ///
    /// Returns the clamped kappa (always >= 1.0, strictest rate limit).
    pub fn limit_kappa(&mut self, proposed: f64, now_s: f64) -> f64 {
        if !self.initialized || self.last_kappa <= 0.0 {
            self.last_kappa = proposed.max(1.0);
            self.last_kappa_time_s = now_s;
            return self.last_kappa;
        }
        let result = self.limit_param(
            proposed,
            self.last_kappa,
            self.last_kappa_time_s,
            now_s,
            self.config.kappa_max_change_per_min,
        );
        self.last_kappa = result.max(1.0); // kappa > 0 invariant
        self.last_kappa_time_s = now_s;
        self.last_kappa
    }

    /// Initialize with first observed values (no rate limiting applied).
    pub fn initialize(&mut self, gamma: f64, sigma: f64, kappa: f64, now_s: f64) {
        self.last_gamma = gamma.max(0.01);
        self.last_sigma = sigma.max(1e-9);
        self.last_kappa = kappa.max(1.0);
        self.last_gamma_time_s = now_s;
        self.last_sigma_time_s = now_s;
        self.last_kappa_time_s = now_s;
        self.initialized = true;
    }

    /// Core rate limiting logic.
    ///
    /// Computes the maximum allowed change based on elapsed time and
    /// max change rate, then clamps the proposed value.
    fn limit_param(
        &self,
        proposed: f64,
        last: f64,
        last_time_s: f64,
        now_s: f64,
        max_change_per_min: f64,
    ) -> f64 {
        let dt_s = (now_s - last_time_s).max(0.0);
        let dt_min = dt_s / 60.0;

        // Scale allowed change by elapsed time (proportional to dt)
        // Cap at 1 minute worth of change to prevent catch-up after long gaps
        let allowed_fraction = (max_change_per_min * dt_min).min(max_change_per_min);

        let max_val = last * (1.0 + allowed_fraction);
        let min_val = last * (1.0 - allowed_fraction);

        proposed.clamp(min_val, max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_value_passthrough() {
        let mut limiter = ParameterRateLimiter::new(RateLimiterConfig::default());
        limiter.initialize(0.5, 0.001, 2000.0, 0.0);

        // First values should pass through unchanged
        assert!((limiter.last_gamma - 0.5).abs() < 1e-10);
        assert!((limiter.last_sigma - 0.001).abs() < 1e-10);
        assert!((limiter.last_kappa - 2000.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_rate_limited() {
        let mut limiter = ParameterRateLimiter::new(RateLimiterConfig::default());
        limiter.initialize(0.5, 0.001, 2000.0, 0.0);

        // Try to double gamma instantly (0.5 → 1.0)
        // At dt=1s: allowed change = 20%/min * (1/60) min = 0.33%
        let result = limiter.limit_gamma(1.0, 1.0);
        assert!(result < 0.6, "Gamma should be rate-limited, got {result}");
        assert!(result > 0.5, "Gamma should increase slightly, got {result}");
    }

    #[test]
    fn test_sigma_allows_faster_changes() {
        let mut limiter = ParameterRateLimiter::new(RateLimiterConfig::default());
        limiter.initialize(0.5, 0.001, 2000.0, 0.0);

        // After 30 seconds, sigma should allow ~15% change (30%/min * 0.5min)
        let result = limiter.limit_sigma(0.002, 30.0); // Trying to double
        let change_pct = (result - 0.001) / 0.001;
        assert!(
            change_pct <= 0.16,
            "Sigma change should be <=15%, got {:.1}%",
            change_pct * 100.0
        );
        assert!(change_pct > 0.10, "Sigma should allow ~15% in 30s");
    }

    #[test]
    fn test_kappa_floor_maintained() {
        let mut limiter = ParameterRateLimiter::new(RateLimiterConfig::default());
        limiter.initialize(0.5, 0.001, 2000.0, 0.0);

        // Even with extreme downward pressure, kappa stays >= 1.0
        let result = limiter.limit_kappa(0.0, 600.0); // Full minute elapsed
        assert!(result >= 1.0, "Kappa must stay >= 1.0, got {result}");
    }

    #[test]
    fn test_full_minute_allows_max_change() {
        let mut limiter = ParameterRateLimiter::new(RateLimiterConfig::default());
        limiter.initialize(0.5, 0.001, 2000.0, 0.0);

        // After 60 seconds (1 minute), gamma can change by full 20%
        let result = limiter.limit_gamma(1.0, 60.0);
        let change_pct = (result - 0.5) / 0.5;
        assert!(
            (change_pct - 0.20).abs() < 0.01,
            "After 1 min, gamma should change by ~20%, got {:.1}%",
            change_pct * 100.0
        );
    }
}
