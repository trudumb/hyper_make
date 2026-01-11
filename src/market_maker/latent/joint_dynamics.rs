//! Joint Parameter Dynamics
//!
//! Models the correlated evolution of latent market parameters:
//!
//! - σ (volatility) ↔ κ (order flow intensity) correlation
//! - σ ↔ AS (adverse selection) correlation
//! - κ ↔ P(informed) correlation
//!
//! # Model
//!
//! Parameters evolve according to correlated Ornstein-Uhlenbeck processes:
//!
//! ```text
//! dX = κ_x(θ_x - X)dt + σ_x dW_x
//! dY = κ_y(θ_y - Y)dt + σ_y dW_y
//!
//! where E[dW_x dW_y] = ρ_{xy} dt
//! ```
//!
//! The joint dynamics enable:
//! 1. Better uncertainty quantification
//! 2. Regime detection via parameter clustering
//! 3. Predictive parameter forecasting
//!
//! # Usage
//!
//! ```ignore
//! let mut dynamics = JointDynamics::new(JointDynamicsConfig::default());
//!
//! // Update with observations
//! dynamics.update(&JointObservation {
//!     sigma: 0.15,
//!     kappa: 100.0,
//!     p_informed: 0.2,
//!     as_bps: 3.0,
//!     timestamp_ms: 1000,
//! });
//!
//! // Get correlation structure
//! let corr = dynamics.correlations();
//! let state = dynamics.current_state();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for joint dynamics model
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JointDynamicsConfig {
    /// Half-life in observations for EWMA
    pub ewma_half_life: usize,

    /// Minimum observations for valid estimates
    pub min_observations: usize,

    /// Maximum observations to store
    pub max_observations: usize,

    /// Mean reversion speed for sigma
    pub kappa_sigma: f64,

    /// Long-run mean for sigma (bps/s)
    pub theta_sigma: f64,

    /// Mean reversion speed for kappa
    pub kappa_kappa: f64,

    /// Long-run mean for kappa
    pub theta_kappa: f64,

    /// Forecast horizon (seconds)
    pub forecast_horizon_s: f64,
}

impl Default for JointDynamicsConfig {
    fn default() -> Self {
        Self {
            ewma_half_life: 100,
            min_observations: 30,
            max_observations: 500,
            kappa_sigma: 0.1,   // Mean revert over ~10 observations
            theta_sigma: 10.0,  // 10 bps/s long-run vol
            kappa_kappa: 0.05,  // Slower mean reversion for kappa
            theta_kappa: 100.0, // 100 units long-run kappa
            forecast_horizon_s: 60.0,
        }
    }
}

// ============================================================================
// Joint Observation
// ============================================================================

/// A single observation of all parameters
#[derive(Debug, Clone, Copy, Default)]
pub struct JointObservation {
    /// Volatility (bps per second)
    pub sigma: f64,

    /// Order flow intensity (kappa)
    pub kappa: f64,

    /// Probability of informed trading
    pub p_informed: f64,

    /// Adverse selection (bps)
    pub as_bps: f64,

    /// Flow momentum (-1 to 1)
    pub flow_momentum: f64,

    /// Timestamp (milliseconds)
    pub timestamp_ms: u64,
}

// ============================================================================
// State Covariance
// ============================================================================

/// Covariance matrix for parameter estimates
#[derive(Debug, Clone, Default)]
pub struct StateCovariance {
    /// Variance of sigma
    pub var_sigma: f64,

    /// Variance of kappa
    pub var_kappa: f64,

    /// Variance of p_informed
    pub var_p_informed: f64,

    /// Variance of AS
    pub var_as: f64,

    /// Covariance(sigma, kappa)
    pub cov_sigma_kappa: f64,

    /// Covariance(sigma, AS)
    pub cov_sigma_as: f64,

    /// Covariance(kappa, p_informed)
    pub cov_kappa_pinformed: f64,

    /// Number of observations
    pub n_obs: usize,
}

impl StateCovariance {
    /// Correlation between sigma and kappa
    pub fn corr_sigma_kappa(&self) -> f64 {
        let denom = (self.var_sigma * self.var_kappa).sqrt();
        if denom > 0.0 {
            (self.cov_sigma_kappa / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Correlation between sigma and AS
    pub fn corr_sigma_as(&self) -> f64 {
        let denom = (self.var_sigma * self.var_as).sqrt();
        if denom > 0.0 {
            (self.cov_sigma_as / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Correlation between kappa and p_informed
    pub fn corr_kappa_pinformed(&self) -> f64 {
        let denom = (self.var_kappa * self.var_p_informed).sqrt();
        if denom > 0.0 {
            (self.cov_kappa_pinformed / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Check if covariance is valid
    pub fn is_valid(&self) -> bool {
        self.n_obs >= 10
            && self.var_sigma > 0.0
            && self.var_kappa > 0.0
            && self.var_sigma.is_finite()
            && self.var_kappa.is_finite()
    }
}

// ============================================================================
// Parameter Correlations
// ============================================================================

/// Correlation structure between parameters
#[derive(Debug, Clone, Default)]
pub struct ParameterCorrelations {
    /// σ ↔ κ: Typically negative (high vol → lower kappa)
    pub sigma_kappa: f64,

    /// σ ↔ AS: Typically positive (high vol → more AS)
    pub sigma_as: f64,

    /// κ ↔ P(informed): Typically positive (active markets → informed)
    pub kappa_p_informed: f64,

    /// AS ↔ P(informed): Typically positive
    pub as_p_informed: f64,

    /// Confidence in correlation estimates
    pub confidence: f64,
}

impl ParameterCorrelations {
    /// Check if correlations suggest toxic conditions
    pub fn is_toxic(&self) -> bool {
        // High correlation between AS and informed probability
        // Combined with positive sigma-AS correlation
        self.sigma_as > 0.3 && self.as_p_informed > 0.5
    }
}

// ============================================================================
// Latent State
// ============================================================================

/// Current latent state estimate
#[derive(Debug, Clone, Default)]
pub struct LatentStateEstimate {
    /// Filtered sigma estimate
    pub sigma: f64,

    /// Filtered kappa estimate
    pub kappa: f64,

    /// Filtered p_informed estimate
    pub p_informed: f64,

    /// Filtered AS estimate
    pub as_bps: f64,

    /// Filtered flow momentum
    pub flow_momentum: f64,

    /// Regime indicator (0=calm, 1=normal, 2=volatile, 3=extreme)
    pub regime: u8,

    /// Uncertainty (combined standard error)
    pub uncertainty: f64,

    /// Timestamp
    pub timestamp_ms: u64,
}

impl LatentStateEstimate {
    /// Detect regime from parameter values
    fn detect_regime(sigma: f64, p_informed: f64, as_bps: f64) -> u8 {
        // Simple regime detection based on thresholds
        let vol_score = if sigma < 5.0 {
            0
        } else if sigma < 15.0 {
            1
        } else if sigma < 30.0 {
            2
        } else {
            3
        };
        let informed_score = if p_informed < 0.1 {
            0
        } else if p_informed < 0.25 {
            1
        } else if p_informed < 0.5 {
            2
        } else {
            3
        };
        let as_score = if as_bps < 1.0 {
            0
        } else if as_bps < 3.0 {
            1
        } else if as_bps < 6.0 {
            2
        } else {
            3
        };

        // Take max of scores
        vol_score.max(informed_score).max(as_score)
    }
}

// ============================================================================
// Joint Dynamics Model
// ============================================================================

/// Joint dynamics model for correlated parameter evolution
#[derive(Debug)]
pub struct JointDynamics {
    /// Configuration
    config: JointDynamicsConfig,

    /// Recent observations
    observations: VecDeque<JointObservation>,

    /// Running means
    mean_sigma: f64,
    mean_kappa: f64,
    mean_p_informed: f64,
    mean_as: f64,
    mean_momentum: f64,

    /// Running second moments
    mean_sigma_sq: f64,
    mean_kappa_sq: f64,
    mean_p_informed_sq: f64,
    mean_as_sq: f64,

    /// Running cross products
    mean_sigma_kappa: f64,
    mean_sigma_as: f64,
    mean_kappa_pinformed: f64,
    mean_as_pinformed: f64,

    /// EWMA alpha
    ewma_alpha: f64,

    /// Observation count
    observation_count: usize,

    /// Last observation timestamp
    last_timestamp_ms: u64,
}

impl JointDynamics {
    /// Create new joint dynamics model
    pub fn new(config: JointDynamicsConfig) -> Self {
        let ewma_alpha = 1.0 - 0.5f64.powf(1.0 / config.ewma_half_life as f64);

        Self {
            config,
            observations: VecDeque::with_capacity(500),
            mean_sigma: 10.0,     // Default
            mean_kappa: 100.0,    // Default
            mean_p_informed: 0.1, // Default
            mean_as: 2.0,         // Default
            mean_momentum: 0.0,
            mean_sigma_sq: 100.0,
            mean_kappa_sq: 10000.0,
            mean_p_informed_sq: 0.01,
            mean_as_sq: 4.0,
            mean_sigma_kappa: 1000.0,
            mean_sigma_as: 20.0,
            mean_kappa_pinformed: 10.0,
            mean_as_pinformed: 0.2,
            ewma_alpha,
            observation_count: 0,
            last_timestamp_ms: 0,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(JointDynamicsConfig::default())
    }

    /// Update with new observation
    pub fn update(&mut self, obs: &JointObservation) {
        // Store observation
        if self.observations.len() >= self.config.max_observations {
            self.observations.pop_front();
        }
        self.observations.push_back(*obs);

        let alpha = self.ewma_alpha;

        // Update means
        if self.observation_count == 0 {
            self.mean_sigma = obs.sigma;
            self.mean_kappa = obs.kappa;
            self.mean_p_informed = obs.p_informed;
            self.mean_as = obs.as_bps;
            self.mean_momentum = obs.flow_momentum;

            self.mean_sigma_sq = obs.sigma * obs.sigma;
            self.mean_kappa_sq = obs.kappa * obs.kappa;
            self.mean_p_informed_sq = obs.p_informed * obs.p_informed;
            self.mean_as_sq = obs.as_bps * obs.as_bps;

            self.mean_sigma_kappa = obs.sigma * obs.kappa;
            self.mean_sigma_as = obs.sigma * obs.as_bps;
            self.mean_kappa_pinformed = obs.kappa * obs.p_informed;
            self.mean_as_pinformed = obs.as_bps * obs.p_informed;
        } else {
            self.mean_sigma = alpha * obs.sigma + (1.0 - alpha) * self.mean_sigma;
            self.mean_kappa = alpha * obs.kappa + (1.0 - alpha) * self.mean_kappa;
            self.mean_p_informed = alpha * obs.p_informed + (1.0 - alpha) * self.mean_p_informed;
            self.mean_as = alpha * obs.as_bps + (1.0 - alpha) * self.mean_as;
            self.mean_momentum = alpha * obs.flow_momentum + (1.0 - alpha) * self.mean_momentum;

            self.mean_sigma_sq = alpha * obs.sigma * obs.sigma + (1.0 - alpha) * self.mean_sigma_sq;
            self.mean_kappa_sq = alpha * obs.kappa * obs.kappa + (1.0 - alpha) * self.mean_kappa_sq;
            self.mean_p_informed_sq =
                alpha * obs.p_informed * obs.p_informed + (1.0 - alpha) * self.mean_p_informed_sq;
            self.mean_as_sq = alpha * obs.as_bps * obs.as_bps + (1.0 - alpha) * self.mean_as_sq;

            self.mean_sigma_kappa =
                alpha * obs.sigma * obs.kappa + (1.0 - alpha) * self.mean_sigma_kappa;
            self.mean_sigma_as =
                alpha * obs.sigma * obs.as_bps + (1.0 - alpha) * self.mean_sigma_as;
            self.mean_kappa_pinformed =
                alpha * obs.kappa * obs.p_informed + (1.0 - alpha) * self.mean_kappa_pinformed;
            self.mean_as_pinformed =
                alpha * obs.as_bps * obs.p_informed + (1.0 - alpha) * self.mean_as_pinformed;
        }

        self.last_timestamp_ms = obs.timestamp_ms;
        self.observation_count += 1;
    }

    // ========================================================================
    // State Access
    // ========================================================================

    /// Get current filtered state estimate
    pub fn current_state(&self) -> LatentStateEstimate {
        let cov = self.covariance();
        let uncertainty = if cov.is_valid() {
            (cov.var_sigma.sqrt() / self.mean_sigma.max(0.1)
                + cov.var_kappa.sqrt() / self.mean_kappa.max(0.1)
                + cov.var_p_informed.sqrt() / self.mean_p_informed.max(0.01))
                / 3.0
        } else {
            1.0
        };

        LatentStateEstimate {
            sigma: self.mean_sigma,
            kappa: self.mean_kappa,
            p_informed: self.mean_p_informed,
            as_bps: self.mean_as,
            flow_momentum: self.mean_momentum,
            regime: LatentStateEstimate::detect_regime(
                self.mean_sigma,
                self.mean_p_informed,
                self.mean_as,
            ),
            uncertainty: uncertainty.clamp(0.0, 1.0),
            timestamp_ms: self.last_timestamp_ms,
        }
    }

    /// Get covariance structure
    pub fn covariance(&self) -> StateCovariance {
        StateCovariance {
            var_sigma: (self.mean_sigma_sq - self.mean_sigma * self.mean_sigma).max(0.0),
            var_kappa: (self.mean_kappa_sq - self.mean_kappa * self.mean_kappa).max(0.0),
            var_p_informed: (self.mean_p_informed_sq - self.mean_p_informed * self.mean_p_informed)
                .max(0.0),
            var_as: (self.mean_as_sq - self.mean_as * self.mean_as).max(0.0),
            cov_sigma_kappa: self.mean_sigma_kappa - self.mean_sigma * self.mean_kappa,
            cov_sigma_as: self.mean_sigma_as - self.mean_sigma * self.mean_as,
            cov_kappa_pinformed: self.mean_kappa_pinformed - self.mean_kappa * self.mean_p_informed,
            n_obs: self.observation_count,
        }
    }

    /// Get parameter correlations
    pub fn correlations(&self) -> ParameterCorrelations {
        let cov = self.covariance();
        let confidence = if self.is_warmed_up() {
            (self.observation_count as f64 / 100.0).tanh()
        } else {
            0.0
        };

        // AS ↔ P(informed) correlation
        let as_pinformed = {
            let denom = (cov.var_as * cov.var_p_informed).sqrt();
            if denom > 0.0 {
                let cov_as_pinformed = self.mean_as_pinformed - self.mean_as * self.mean_p_informed;
                (cov_as_pinformed / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        };

        ParameterCorrelations {
            sigma_kappa: cov.corr_sigma_kappa(),
            sigma_as: cov.corr_sigma_as(),
            kappa_p_informed: cov.corr_kappa_pinformed(),
            as_p_informed: as_pinformed,
            confidence,
        }
    }

    /// Forecast state at future time
    pub fn forecast(&self, horizon_s: f64) -> LatentStateEstimate {
        // Mean reversion forecast
        let decay_sigma = (-self.config.kappa_sigma * horizon_s).exp();
        let decay_kappa = (-self.config.kappa_kappa * horizon_s).exp();

        let forecast_sigma =
            self.config.theta_sigma + decay_sigma * (self.mean_sigma - self.config.theta_sigma);
        let forecast_kappa =
            self.config.theta_kappa + decay_kappa * (self.mean_kappa - self.config.theta_kappa);

        // P(informed) and AS are harder to forecast, use current values with decay
        let forecast_p_informed = self.mean_p_informed * 0.9_f64.powf(horizon_s / 60.0);
        let forecast_as = self.mean_as;

        // Uncertainty increases with horizon
        let base_uncertainty = self.current_state().uncertainty;
        let horizon_factor = 1.0 + (horizon_s / 60.0).sqrt() * 0.5;

        LatentStateEstimate {
            sigma: forecast_sigma,
            kappa: forecast_kappa,
            p_informed: forecast_p_informed,
            as_bps: forecast_as,
            flow_momentum: self.mean_momentum * 0.5_f64.powf(horizon_s / 10.0),
            regime: LatentStateEstimate::detect_regime(
                forecast_sigma,
                forecast_p_informed,
                forecast_as,
            ),
            uncertainty: (base_uncertainty * horizon_factor).clamp(0.0, 1.0),
            timestamp_ms: self.last_timestamp_ms + (horizon_s * 1000.0) as u64,
        }
    }

    // ========================================================================
    // Status Methods
    // ========================================================================

    /// Number of observations
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Check if model is warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.observation_count >= self.config.min_observations
    }

    /// Reset model
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone());
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let dynamics = JointDynamics::default_config();
        assert_eq!(dynamics.observation_count(), 0);
        assert!(!dynamics.is_warmed_up());
    }

    #[test]
    fn test_update() {
        let mut dynamics = JointDynamics::default_config();

        let obs = JointObservation {
            sigma: 15.0,
            kappa: 150.0,
            p_informed: 0.2,
            as_bps: 3.0,
            flow_momentum: 0.5,
            timestamp_ms: 1000,
        };

        dynamics.update(&obs);
        assert_eq!(dynamics.observation_count(), 1);

        let state = dynamics.current_state();
        assert!((state.sigma - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_covariance_computation() {
        let mut dynamics = JointDynamics::default_config();

        // Add correlated observations: high sigma → high AS
        for i in 0..100 {
            let base_sigma = 10.0 + (i as f64 / 10.0);
            let obs = JointObservation {
                sigma: base_sigma,
                kappa: 100.0 - (i as f64 / 2.0), // Negative correlation with sigma
                p_informed: 0.1 + (i as f64 / 500.0),
                as_bps: base_sigma * 0.2, // Positive correlation with sigma
                flow_momentum: 0.0,
                timestamp_ms: i * 1000,
            };
            dynamics.update(&obs);
        }

        let cov = dynamics.covariance();
        let corr = dynamics.correlations();

        assert!(cov.var_sigma > 0.0, "Variance should be positive");
        assert!(cov.var_kappa > 0.0, "Variance should be positive");

        // Sigma-AS should be positively correlated
        assert!(
            corr.sigma_as > 0.0,
            "Sigma-AS should be positive: {}",
            corr.sigma_as
        );

        // Sigma-kappa should be negatively correlated
        assert!(
            corr.sigma_kappa < 0.0,
            "Sigma-kappa should be negative: {}",
            corr.sigma_kappa
        );
    }

    #[test]
    fn test_regime_detection() {
        // Calm regime
        let regime_calm = LatentStateEstimate::detect_regime(3.0, 0.05, 0.5);
        assert_eq!(regime_calm, 0);

        // Normal regime
        let regime_normal = LatentStateEstimate::detect_regime(10.0, 0.15, 2.0);
        assert_eq!(regime_normal, 1);

        // Volatile regime
        let regime_volatile = LatentStateEstimate::detect_regime(20.0, 0.3, 4.0);
        assert_eq!(regime_volatile, 2);

        // Extreme regime
        let regime_extreme = LatentStateEstimate::detect_regime(50.0, 0.6, 8.0);
        assert_eq!(regime_extreme, 3);
    }

    #[test]
    fn test_forecast() {
        let mut dynamics = JointDynamics::default_config();

        // Add some observations
        for i in 0..50 {
            let obs = JointObservation {
                sigma: 20.0, // Above long-run mean of 10
                kappa: 80.0, // Below long-run mean of 100
                p_informed: 0.3,
                as_bps: 4.0,
                flow_momentum: 0.2,
                timestamp_ms: i * 1000,
            };
            dynamics.update(&obs);
        }

        let current = dynamics.current_state();
        let forecast = dynamics.forecast(60.0);

        // Sigma should mean-revert toward theta_sigma (10.0)
        assert!(
            forecast.sigma < current.sigma,
            "Sigma should mean-revert down"
        );

        // Kappa should mean-revert toward theta_kappa (100.0)
        assert!(
            forecast.kappa > current.kappa,
            "Kappa should mean-revert up"
        );

        // Uncertainty should increase with horizon
        assert!(
            forecast.uncertainty >= current.uncertainty,
            "Uncertainty should increase"
        );
    }

    #[test]
    fn test_warmup() {
        let mut dynamics = JointDynamics::default_config();

        for i in 0..35 {
            let obs = JointObservation {
                sigma: 10.0,
                kappa: 100.0,
                p_informed: 0.1,
                as_bps: 2.0,
                timestamp_ms: i * 1000,
                ..Default::default()
            };
            dynamics.update(&obs);
        }

        assert!(dynamics.is_warmed_up());
    }

    #[test]
    fn test_reset() {
        let mut dynamics = JointDynamics::default_config();

        for i in 0..50 {
            let obs = JointObservation {
                sigma: 15.0,
                kappa: 120.0,
                timestamp_ms: i * 1000,
                ..Default::default()
            };
            dynamics.update(&obs);
        }

        assert!(dynamics.observation_count() > 0);

        dynamics.reset();

        assert_eq!(dynamics.observation_count(), 0);
        assert!(!dynamics.is_warmed_up());
    }

    #[test]
    fn test_toxic_detection() {
        let toxic_corr = ParameterCorrelations {
            sigma_kappa: -0.5,
            sigma_as: 0.6, // High positive
            kappa_p_informed: 0.3,
            as_p_informed: 0.7, // High positive
            confidence: 0.8,
        };

        assert!(toxic_corr.is_toxic());

        let calm_corr = ParameterCorrelations {
            sigma_kappa: -0.2,
            sigma_as: 0.1,
            kappa_p_informed: 0.2,
            as_p_informed: 0.3,
            confidence: 0.8,
        };

        assert!(!calm_corr.is_toxic());
    }
}
