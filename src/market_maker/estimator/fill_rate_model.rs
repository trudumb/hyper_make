//! Bayesian Fill Rate Model
//!
//! Models fill probability as a function of quote depth and market state.
//!
//! # Model
//!
//! ```text
//! log(λ(δ)) = log(λ_0) - δ/δ_char + Σ β_i × x_i
//! ```
//!
//! Where:
//! - λ(δ) = fill rate at depth δ (fills per second)
//! - λ_0 = baseline fill rate
//! - δ_char = characteristic depth (decay constant)
//! - x_i = market features (volatility, spread, imbalance, regime)
//! - β_i = learned coefficients
//!
//! # Bayesian Estimation
//!
//! Uses online Bayesian regression with exponential forgetting:
//! - Gamma prior on λ_0 (conjugate for exponential fill times)
//! - Normal priors on feature coefficients
//! - Rolling window of observations for adaptation
//!
//! # Usage
//!
//! ```ignore
//! let mut model = FillRateModel::new(FillRateConfig::default());
//!
//! // Observe fills and non-fills
//! model.observe(&FillObservation {
//!     depth_bps: 5.0,
//!     filled: true,
//!     duration_s: 0.5,
//!     state: MarketState::default(),
//! });
//!
//! // Query expected fill rate
//! let rate = model.fill_rate(5.0, &state);
//! let optimal = model.optimal_depth(0.5, &state); // 50% fill rate target
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for fill rate model
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FillRateConfig {
    /// Prior mean for baseline fill rate (fills/second)
    pub lambda_0_prior_mean: f64,

    /// Prior strength for lambda_0 (effective observations)
    pub lambda_0_prior_strength: f64,

    /// Prior mean for characteristic depth (bps)
    pub delta_char_prior_mean: f64,

    /// Prior strength for delta_char (effective observations)
    pub delta_char_prior_strength: f64,

    /// Half-life in observations for exponential forgetting
    pub observation_half_life: usize,

    /// Minimum observations before model is reliable
    pub min_observations: usize,

    /// Maximum observations to keep in buffer
    pub buffer_size: usize,

    /// Regularization strength for feature coefficients
    pub coefficient_regularization: f64,

    /// Maximum depth to consider (bps)
    pub max_depth_bps: f64,

    /// Minimum fill rate estimate (prevents zero)
    pub min_fill_rate: f64,
}

impl Default for FillRateConfig {
    fn default() -> Self {
        Self {
            // MAINNET OPTIMIZED: Higher baseline for liquid markets
            lambda_0_prior_mean: 0.8, // 0.8 fills/second (increased from 0.1)
            lambda_0_prior_strength: 10.0, // Moderate confidence
            delta_char_prior_mean: 10.0, // 10 bps characteristic depth
            delta_char_prior_strength: 5.0, // Lower confidence
            // MAINNET OPTIMIZED: Faster adaptation with abundant data
            observation_half_life: 100, // Reduced from 200 - faster convergence
            min_observations: 30,       // 30 fills/non-fills needed
            buffer_size: 500,           // Keep recent observations
            coefficient_regularization: 0.1, // L2 regularization
            max_depth_bps: 100.0,       // Cap depth at 100 bps
            min_fill_rate: 0.001,       // Floor at 0.1% per second
        }
    }
}

// ============================================================================
// Market State Features
// ============================================================================

/// Market state for conditioning fill rate predictions
#[derive(Debug, Clone, Copy, Default)]
pub struct MarketState {
    /// Volatility (per-second, bps)
    pub sigma_bps: f64,

    /// Current spread (bps)
    pub spread_bps: f64,

    /// Book imbalance (-1 to 1)
    pub book_imbalance: f64,

    /// Flow imbalance (-1 to 1)
    pub flow_imbalance: f64,

    /// Regime (0=low, 1=normal, 2=high, 3=extreme)
    pub regime: u8,

    /// Hour of day (UTC, 0-23)
    pub hour_utc: u8,

    /// Is bid side (affects imbalance interpretation)
    pub is_bid: bool,
}

impl MarketState {
    /// Extract feature vector for regression
    fn to_features(self) -> [f64; 6] {
        [
            self.sigma_bps.ln().clamp(-5.0, 5.0),         // Log volatility
            (self.spread_bps / 10.0).clamp(-2.0, 2.0),    // Normalized spread
            self.book_imbalance,                          // Book pressure
            self.flow_imbalance,                          // Trade flow
            (self.regime as f64 - 1.5) / 1.5,             // Regime (-1 to 1)
            ((self.hour_utc as f64 - 12.0) / 12.0).cos(), // Hour seasonality
        ]
    }
}

// ============================================================================
// Fill Observation
// ============================================================================

/// A single fill/no-fill observation
#[derive(Debug, Clone, Copy)]
pub struct FillObservation {
    /// Depth at which order was placed (bps from mid)
    pub depth_bps: f64,

    /// Whether the order was filled
    pub filled: bool,

    /// Duration of observation (seconds)
    /// For filled orders: time until fill
    /// For unfilled: time until cancel
    pub duration_s: f64,

    /// Market state at time of observation
    pub state: MarketState,

    /// Quote size (for size-dependent models)
    pub size: f64,
}

impl Default for FillObservation {
    fn default() -> Self {
        Self {
            depth_bps: 5.0,
            filled: false,
            duration_s: 1.0,
            state: MarketState::default(),
            size: 1.0,
        }
    }
}

// ============================================================================
// Bayesian Estimate
// ============================================================================

/// Bayesian estimate with uncertainty
#[derive(Debug, Clone, Copy)]
pub struct BayesianEstimate {
    /// Posterior mean
    pub mean: f64,

    /// Posterior variance
    pub variance: f64,

    /// Number of observations
    pub n_obs: f64,
}

impl BayesianEstimate {
    fn new(prior_mean: f64, prior_strength: f64) -> Self {
        Self {
            mean: prior_mean,
            variance: prior_mean.powi(2) / prior_strength,
            n_obs: prior_strength,
        }
    }

    /// 95% credible interval
    pub fn credible_interval(&self) -> (f64, f64) {
        let std = self.variance.sqrt();
        (self.mean - 1.96 * std, self.mean + 1.96 * std)
    }

    /// Update with new observation using Bayesian update
    fn update(&mut self, observation: f64, weight: f64) {
        let precision = 1.0 / self.variance;
        let obs_precision = weight / (observation.powi(2).max(0.01));

        let new_precision = precision + obs_precision;
        let new_mean = (precision * self.mean + obs_precision * observation) / new_precision;

        self.mean = new_mean;
        self.variance = 1.0 / new_precision;
        self.n_obs += weight;
    }

    /// Apply forgetting factor
    fn forget(&mut self, factor: f64) {
        self.variance /= factor;
        self.n_obs *= factor;
    }
}

// ============================================================================
// Feature Coefficients
// ============================================================================

/// Coefficients for market state features
#[derive(Debug, Clone)]
pub struct FillRateCoefficients {
    /// Coefficient values [log_vol, spread, book_imb, flow_imb, regime, hour]
    pub values: [f64; 6],

    /// Coefficient variances
    pub variances: [f64; 6],

    /// Names for debugging
    pub names: [&'static str; 6],
}

impl Default for FillRateCoefficients {
    fn default() -> Self {
        Self {
            // Expected signs:
            // log_vol: negative (higher vol = lower fills at same depth)
            // spread: positive (wider market spread = easier to fill)
            // book_imb: depends on side (buy imbalance helps asks)
            // flow_imb: depends on side
            // regime: negative (high regime = fewer fills)
            // hour: varies with liquidity
            values: [0.0; 6],
            variances: [1.0; 6],
            names: [
                "log_vol", "spread", "book_imb", "flow_imb", "regime", "hour_cos",
            ],
        }
    }
}

impl FillRateCoefficients {
    /// Compute linear combination β'x
    fn dot(&self, features: &[f64; 6]) -> f64 {
        self.values
            .iter()
            .zip(features.iter())
            .map(|(b, x)| b * x)
            .sum()
    }

    /// Update coefficients with stochastic gradient descent
    fn sgd_update(&mut self, features: &[f64; 6], gradient: f64, learning_rate: f64, l2_reg: f64) {
        for (i, &feat) in features.iter().enumerate() {
            let grad = gradient * feat + l2_reg * self.values[i];
            self.values[i] -= learning_rate * grad;
            // Update variance estimate (approximate)
            self.variances[i] = 0.99 * self.variances[i] + 0.01 * grad.powi(2).max(0.01);
        }
    }
}

// ============================================================================
// Sufficient Statistics
// ============================================================================

/// Running statistics for online estimation
#[derive(Debug, Clone, Default)]
struct FillRateStats {
    /// Weighted count of fills
    n_fills: f64,

    /// Weighted count of non-fills
    n_non_fills: f64,

    /// Sum of fill times
    sum_fill_time: f64,

    /// Sum of squared fill times
    sum_fill_time_sq: f64,

    /// Sum of depths for fills
    sum_depth_fills: f64,

    /// Sum of depths for non-fills
    sum_depth_non_fills: f64,
}

// ============================================================================
// Fill Rate Model
// ============================================================================

/// Bayesian fill rate model with feature conditioning
#[derive(Debug)]
pub struct FillRateModel {
    /// Baseline fill rate estimate
    lambda_0: BayesianEstimate,

    /// Characteristic depth estimate
    delta_char: BayesianEstimate,

    /// Feature coefficients
    coefficients: FillRateCoefficients,

    /// Recent observations buffer
    observation_buffer: VecDeque<FillObservation>,

    /// Running statistics
    stats: FillRateStats,

    /// Configuration
    config: FillRateConfig,

    /// Forgetting factor (computed from half-life)
    forgetting_factor: f64,

    /// Number of observations processed
    observation_count: usize,
}

impl FillRateModel {
    /// Create new fill rate model
    pub fn new(config: FillRateConfig) -> Self {
        let lambda_0 =
            BayesianEstimate::new(config.lambda_0_prior_mean, config.lambda_0_prior_strength);
        let delta_char = BayesianEstimate::new(
            config.delta_char_prior_mean,
            config.delta_char_prior_strength,
        );
        let forgetting_factor = 0.5f64.powf(1.0 / config.observation_half_life as f64);

        Self {
            lambda_0,
            delta_char,
            coefficients: FillRateCoefficients::default(),
            observation_buffer: VecDeque::with_capacity(config.buffer_size),
            stats: FillRateStats::default(),
            config,
            forgetting_factor,
            observation_count: 0,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(FillRateConfig::default())
    }

    /// Observe a fill or non-fill event
    pub fn observe(&mut self, obs: &FillObservation) {
        // Apply forgetting to old statistics
        self.apply_forgetting();

        // Update statistics
        if obs.filled {
            self.stats.n_fills += 1.0;
            self.stats.sum_fill_time += obs.duration_s;
            self.stats.sum_fill_time_sq += obs.duration_s.powi(2);
            self.stats.sum_depth_fills += obs.depth_bps;
        } else {
            self.stats.n_non_fills += 1.0;
            self.stats.sum_depth_non_fills += obs.depth_bps;
        }

        // Store observation
        if self.observation_buffer.len() >= self.config.buffer_size {
            self.observation_buffer.pop_front();
        }
        self.observation_buffer.push_back(*obs);

        // Update parameters
        self.update_parameters(obs);

        self.observation_count += 1;
    }

    /// Apply forgetting factor to statistics
    fn apply_forgetting(&mut self) {
        let f = self.forgetting_factor;
        self.stats.n_fills *= f;
        self.stats.n_non_fills *= f;
        self.stats.sum_fill_time *= f;
        self.stats.sum_fill_time_sq *= f;
        self.stats.sum_depth_fills *= f;
        self.stats.sum_depth_non_fills *= f;

        self.lambda_0.forget(f);
        self.delta_char.forget(f);
    }

    /// Update model parameters from observation
    fn update_parameters(&mut self, obs: &FillObservation) {
        let features = obs.state.to_features();
        let depth_norm = obs.depth_bps / self.delta_char.mean;

        // Predicted log fill rate
        let log_lambda_pred =
            self.lambda_0.mean.ln() - depth_norm + self.coefficients.dot(&features);
        let _lambda_pred = log_lambda_pred.exp().clamp(self.config.min_fill_rate, 10.0);

        // Update baseline rate from fills
        if obs.filled && obs.duration_s > 0.0 {
            // Fill time follows exponential distribution with rate λ
            // MLE: λ = 1 / mean_fill_time
            let obs_rate = 1.0 / obs.duration_s;
            // Account for depth
            let adjusted_rate = obs_rate * (depth_norm).exp();
            self.lambda_0.update(adjusted_rate, 1.0);
        }

        // Update characteristic depth
        if self.stats.n_fills > 1.0 && self.stats.n_non_fills > 1.0 {
            let avg_depth_fills = self.stats.sum_depth_fills / self.stats.n_fills;
            let avg_depth_non = self.stats.sum_depth_non_fills / self.stats.n_non_fills;

            // δ_char relates to where fills vs non-fills separate
            if avg_depth_non > avg_depth_fills {
                let estimated_delta_char = (avg_depth_non + avg_depth_fills) / 2.0;
                self.delta_char
                    .update(estimated_delta_char.clamp(1.0, 50.0), 0.1);
            }
        }

        // Update coefficients with SGD
        // Gradient of log-likelihood for logistic regression
        let y = if obs.filled { 1.0 } else { 0.0 };
        let fill_prob = self.fill_probability_internal(obs.depth_bps, &features, obs.duration_s);
        let gradient = fill_prob - y;

        let learning_rate = 0.01 / (1.0 + self.observation_count as f64 / 1000.0);
        self.coefficients.sgd_update(
            &features,
            gradient,
            learning_rate,
            self.config.coefficient_regularization,
        );
    }

    /// Internal fill probability calculation
    fn fill_probability_internal(
        &self,
        depth_bps: f64,
        features: &[f64; 6],
        duration_s: f64,
    ) -> f64 {
        let lambda = self.fill_rate_internal(depth_bps, features);

        // P(fill) = 1 - exp(-λ * t) for exponential fill times
        (1.0 - (-lambda * duration_s).exp()).clamp(0.0, 1.0)
    }

    /// Internal fill rate calculation
    fn fill_rate_internal(&self, depth_bps: f64, features: &[f64; 6]) -> f64 {
        let depth_clamped = depth_bps.clamp(0.0, self.config.max_depth_bps);
        let depth_norm = depth_clamped / self.delta_char.mean;

        let log_lambda = self.lambda_0.mean.ln() - depth_norm + self.coefficients.dot(features);

        log_lambda.exp().clamp(self.config.min_fill_rate, 10.0)
    }

    // ========================================================================
    // Public API
    // ========================================================================

    /// Predicted fill rate at given depth and market state (fills/second)
    pub fn fill_rate(&self, depth_bps: f64, state: &MarketState) -> f64 {
        let features = state.to_features();
        self.fill_rate_internal(depth_bps, &features)
    }

    /// Probability of fill within given duration
    pub fn fill_probability(&self, depth_bps: f64, state: &MarketState, duration_s: f64) -> f64 {
        let lambda = self.fill_rate(depth_bps, state);
        (1.0 - (-lambda * duration_s).exp()).clamp(0.0, 1.0)
    }

    /// Find optimal depth for target fill rate
    pub fn optimal_depth(&self, target_rate: f64, state: &MarketState) -> f64 {
        let features = state.to_features();
        let feature_contrib = self.coefficients.dot(&features);

        // Solve: target = λ_0 * exp(-δ/δ_char + feature_contrib)
        // log(target) = log(λ_0) - δ/δ_char + feature_contrib
        // δ = δ_char * (log(λ_0) - log(target) + feature_contrib)

        let log_lambda_0 = self.lambda_0.mean.ln();
        let log_target = target_rate.clamp(self.config.min_fill_rate, 10.0).ln();

        let depth = self.delta_char.mean * (log_lambda_0 - log_target + feature_contrib);

        depth.clamp(0.0, self.config.max_depth_bps)
    }

    /// Find depth for target fill probability over duration
    pub fn depth_for_probability(
        &self,
        target_prob: f64,
        state: &MarketState,
        duration_s: f64,
    ) -> f64 {
        // P(fill) = 1 - exp(-λt) => λ = -ln(1-P)/t
        let target_prob_clamped = target_prob.clamp(0.01, 0.99);
        let target_rate = -(1.0 - target_prob_clamped).ln() / duration_s;

        self.optimal_depth(target_rate, state)
    }

    /// Get baseline fill rate estimate
    pub fn lambda_0(&self) -> &BayesianEstimate {
        &self.lambda_0
    }

    /// Get characteristic depth estimate
    pub fn delta_char(&self) -> &BayesianEstimate {
        &self.delta_char
    }

    /// Get feature coefficients
    pub fn coefficients(&self) -> &FillRateCoefficients {
        &self.coefficients
    }

    /// Total observations processed
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Check if model has enough data
    pub fn is_warmed_up(&self) -> bool {
        self.observation_count >= self.config.min_observations
    }

    /// Fill rate statistics
    pub fn fill_statistics(&self) -> FillRateStatistics {
        let total = self.stats.n_fills + self.stats.n_non_fills;
        let fill_rate = if total > 0.0 {
            self.stats.n_fills / total
        } else {
            0.0
        };

        let mean_fill_time = if self.stats.n_fills > 0.0 {
            self.stats.sum_fill_time / self.stats.n_fills
        } else {
            0.0
        };

        FillRateStatistics {
            empirical_fill_rate: fill_rate,
            mean_fill_time_s: mean_fill_time,
            n_fills: self.stats.n_fills as usize,
            n_non_fills: self.stats.n_non_fills as usize,
            lambda_0_mean: self.lambda_0.mean,
            lambda_0_ci: self.lambda_0.credible_interval(),
            delta_char_mean: self.delta_char.mean,
            delta_char_ci: self.delta_char.credible_interval(),
        }
    }

    /// Reset model to initial state
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone());
    }

    // === Checkpoint persistence ===

    /// Extract Bayesian posterior state for checkpoint persistence.
    pub fn to_checkpoint(&self) -> crate::market_maker::checkpoint::FillRateCheckpoint {
        crate::market_maker::checkpoint::FillRateCheckpoint {
            lambda_0_mean: self.lambda_0.mean,
            lambda_0_variance: self.lambda_0.variance,
            lambda_0_n_obs: self.lambda_0.n_obs,
            delta_char_mean: self.delta_char.mean,
            delta_char_variance: self.delta_char.variance,
            delta_char_n_obs: self.delta_char.n_obs,
            observation_count: self.observation_count,
        }
    }

    /// Restore Bayesian posterior state from a checkpoint.
    pub fn restore_checkpoint(&mut self, cp: &crate::market_maker::checkpoint::FillRateCheckpoint) {
        self.lambda_0.mean = cp.lambda_0_mean;
        self.lambda_0.variance = cp.lambda_0_variance;
        self.lambda_0.n_obs = cp.lambda_0_n_obs;
        self.delta_char.mean = cp.delta_char_mean;
        self.delta_char.variance = cp.delta_char_variance;
        self.delta_char.n_obs = cp.delta_char_n_obs;
        self.observation_count = cp.observation_count;
    }
}

/// Summary statistics for fill rate model
#[derive(Debug, Clone)]
pub struct FillRateStatistics {
    /// Empirical fill rate (fills / total observations)
    pub empirical_fill_rate: f64,

    /// Mean time to fill (seconds)
    pub mean_fill_time_s: f64,

    /// Number of fills observed
    pub n_fills: usize,

    /// Number of non-fills observed
    pub n_non_fills: usize,

    /// Baseline fill rate (posterior mean)
    pub lambda_0_mean: f64,

    /// Baseline fill rate 95% CI
    pub lambda_0_ci: (f64, f64),

    /// Characteristic depth (posterior mean)
    pub delta_char_mean: f64,

    /// Characteristic depth 95% CI
    pub delta_char_ci: (f64, f64),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_initialization() {
        let model = FillRateModel::default_config();
        assert_eq!(model.observation_count(), 0);
        assert!(!model.is_warmed_up());

        // Prior values should be set (mainnet-optimized: lambda_0 = 0.8)
        assert!((model.lambda_0.mean - 0.8).abs() < 0.01);
        assert!((model.delta_char.mean - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_fill_rate_decreases_with_depth() {
        let model = FillRateModel::default_config();
        let state = MarketState::default();

        let rate_5 = model.fill_rate(5.0, &state);
        let rate_10 = model.fill_rate(10.0, &state);
        let rate_20 = model.fill_rate(20.0, &state);

        assert!(
            rate_5 > rate_10,
            "Rate at 5 bps should exceed rate at 10 bps"
        );
        assert!(
            rate_10 > rate_20,
            "Rate at 10 bps should exceed rate at 20 bps"
        );
    }

    #[test]
    fn test_fill_probability_increases_with_time() {
        let model = FillRateModel::default_config();
        let state = MarketState::default();

        let prob_1s = model.fill_probability(5.0, &state, 1.0);
        let prob_5s = model.fill_probability(5.0, &state, 5.0);
        let prob_10s = model.fill_probability(5.0, &state, 10.0);

        assert!(prob_1s < prob_5s, "P(fill) should increase with time");
        assert!(prob_5s < prob_10s, "P(fill) should increase with time");
        assert!(prob_10s < 1.0, "P(fill) should be less than 1");
    }

    #[test]
    fn test_observation_updates_model() {
        let mut model = FillRateModel::default_config();

        // Add some fills at close depths
        for _ in 0..50 {
            model.observe(&FillObservation {
                depth_bps: 3.0,
                filled: true,
                duration_s: 0.5,
                state: MarketState::default(),
                size: 1.0,
            });
        }

        // Add some non-fills at far depths
        for _ in 0..50 {
            model.observe(&FillObservation {
                depth_bps: 15.0,
                filled: false,
                duration_s: 5.0,
                state: MarketState::default(),
                size: 1.0,
            });
        }

        let stats = model.fill_statistics();
        assert!(
            stats.empirical_fill_rate > 0.4,
            "Should have ~50% fill rate"
        );
        assert!(
            stats.mean_fill_time_s > 0.0,
            "Mean fill time should be positive"
        );
    }

    #[test]
    fn test_optimal_depth_calculation() {
        let model = FillRateModel::default_config();
        let state = MarketState::default();

        // Higher target rate should give smaller depth
        let depth_high = model.optimal_depth(0.5, &state); // Want fast fills
        let depth_low = model.optimal_depth(0.05, &state); // OK with slow fills

        assert!(
            depth_high < depth_low,
            "Higher target rate should need tighter depth: {} vs {}",
            depth_high,
            depth_low
        );
    }

    #[test]
    fn test_depth_for_probability() {
        let model = FillRateModel::default_config();
        let state = MarketState::default();

        // Higher fill probability target should give smaller depth
        // Use 20 second window so 80% is achievable with default λ_0 = 0.1
        // At depth 0, P(fill in 20s) = 1 - e^(-0.1*20) = 86%
        let depth_80 = model.depth_for_probability(0.8, &state, 20.0);
        let depth_20 = model.depth_for_probability(0.2, &state, 20.0);

        assert!(
            depth_80 < depth_20,
            "80% probability needs tighter depth than 20%: {} vs {}",
            depth_80,
            depth_20
        );
    }

    #[test]
    fn test_warmup() {
        let mut model = FillRateModel::default_config();

        for i in 0..40 {
            model.observe(&FillObservation {
                depth_bps: 5.0,
                filled: i % 2 == 0,
                duration_s: 1.0,
                ..Default::default()
            });
        }

        assert!(model.is_warmed_up());
    }

    #[test]
    fn test_market_state_features() {
        let state = MarketState {
            sigma_bps: 10.0,
            spread_bps: 5.0,
            book_imbalance: 0.5,
            flow_imbalance: -0.3,
            regime: 1,
            hour_utc: 12,
            is_bid: true,
        };

        let features = state.to_features();
        assert_eq!(features.len(), 6);

        // Features should be bounded
        for f in &features {
            assert!(f.is_finite(), "Feature should be finite");
        }
    }

    #[test]
    fn test_credible_interval() {
        let estimate = BayesianEstimate::new(0.1, 10.0);
        let (lo, hi) = estimate.credible_interval();

        assert!(lo < estimate.mean, "Lower bound should be below mean");
        assert!(hi > estimate.mean, "Upper bound should be above mean");
        assert!(
            lo > 0.0 || hi > 0.0,
            "At least one bound should be positive"
        );
    }

    #[test]
    fn test_reset() {
        let mut model = FillRateModel::default_config();

        for _ in 0..50 {
            model.observe(&FillObservation {
                depth_bps: 5.0,
                filled: true,
                duration_s: 0.5,
                ..Default::default()
            });
        }

        assert!(model.observation_count() > 0);

        model.reset();

        assert_eq!(model.observation_count(), 0);
        assert!(!model.is_warmed_up());
    }
}
