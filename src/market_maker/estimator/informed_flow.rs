//! Online EM-based Flow Decomposition
//!
//! Decomposes trade flow into three components via mixture modeling:
//!
//! 1. **Informed traders**: Large sizes, clustered arrivals, high price impact
//! 2. **Noise traders**: Random sizes, Poisson arrivals, low/temporary impact
//! 3. **Forced traders**: Predictable flow (liquidations, rebalances)
//!
//! # Algorithm
//!
//! Online EM (Expectation-Maximization) with exponential forgetting:
//!
//! ```text
//! E-step: Compute responsibilities γ_{t,k} = P(component_k | trade_t)
//! M-step: Update component parameters with forgetting factor λ
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut estimator = InformedFlowEstimator::new(InformedFlowConfig::default());
//!
//! // Feed trades with features
//! let features = TradeFeatures {
//!     size: 1.5,
//!     inter_arrival_ms: 250,
//!     price_impact_bps: 2.5,
//!     book_imbalance: 0.3,
//!     is_buy: true,
//! };
//! estimator.on_trade(&features);
//!
//! // Get decomposition
//! let decomp = estimator.decomposition();
//! println!("P(informed) = {:.2}%", decomp.p_informed * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::f64::consts::PI;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the informed flow estimator
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InformedFlowConfig {
    /// Number of mixture components (default: 3)
    pub n_components: usize,

    /// EM update interval in trades (default: 50)
    pub em_update_interval: usize,

    /// Half-life in trades for exponential forgetting (default: 500)
    pub observation_half_life: usize,

    /// Minimum observations before estimates are reliable (default: 100)
    pub min_observations: usize,

    /// Buffer size for trade features (default: 1000)
    pub buffer_size: usize,

    /// Price impact horizon in milliseconds (default: 1000)
    pub impact_horizon_ms: u64,

    /// Prior probabilities for components [informed, noise, forced]
    pub prior_probs: [f64; 3],
}

impl Default for InformedFlowConfig {
    fn default() -> Self {
        Self {
            n_components: 3,
            em_update_interval: 50,
            observation_half_life: 500,
            min_observations: 100,
            buffer_size: 1000,
            impact_horizon_ms: 1000,
            // Prior: 5% informed, 85% noise, 10% forced
            prior_probs: [0.05, 0.85, 0.10],
        }
    }
}

// ============================================================================
// Component Parameters
// ============================================================================

/// Parameters for a single mixture component
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct ComponentParams {
    // Size distribution (log-normal)
    /// Mean of log-size
    pub mu_size: f64,
    /// Std of log-size
    pub sigma_size: f64,

    // Arrival process
    /// Baseline arrival rate (trades per second)
    pub lambda_0: f64,
    /// Self-excitation parameter (Hawkes α)
    pub alpha: f64,
    /// Decay rate (Hawkes β)
    pub beta: f64,

    // Price impact
    /// Expected impact per unit size (bps)
    pub impact_per_unit: f64,
    /// Impact variance
    pub impact_variance: f64,
}

impl Default for ComponentParams {
    fn default() -> Self {
        Self {
            mu_size: 0.0,
            sigma_size: 1.0,
            lambda_0: 0.1,
            alpha: 0.0,
            beta: 1.0,
            impact_per_unit: 1.0,
            impact_variance: 4.0,
        }
    }
}

impl ComponentParams {
    /// Create default informed trader parameters
    pub fn informed() -> Self {
        Self {
            mu_size: 1.0,       // Larger than average
            sigma_size: 0.8,    // Less variance (more consistent)
            lambda_0: 0.05,     // Lower baseline
            alpha: 0.5,         // High self-excitation (clustering)
            beta: 2.0,          // Moderate decay
            impact_per_unit: 3.0,  // High impact
            impact_variance: 2.0,
        }
    }

    /// Create default noise trader parameters
    pub fn noise() -> Self {
        Self {
            mu_size: -0.5,      // Smaller than average
            sigma_size: 1.2,    // Higher variance (random)
            lambda_0: 0.2,      // Higher baseline (frequent)
            alpha: 0.1,         // Low self-excitation (Poisson-like)
            beta: 5.0,          // Fast decay
            impact_per_unit: 0.5,  // Low impact
            impact_variance: 1.0,
        }
    }

    /// Create default forced trader parameters
    pub fn forced() -> Self {
        Self {
            mu_size: 0.5,       // Medium-large
            sigma_size: 0.5,    // Low variance (predictable size)
            lambda_0: 0.02,     // Rare baseline
            alpha: 0.8,         // Very high clustering (liquidation cascades)
            beta: 1.0,          // Slow decay (sustained pressure)
            impact_per_unit: 2.0,  // Medium-high impact
            impact_variance: 3.0,
        }
    }
}

// ============================================================================
// Trade Features
// ============================================================================

/// Features extracted from a trade for flow classification
#[derive(Debug, Clone, Copy)]
pub struct TradeFeatures {
    /// Trade size (in asset units, normalized by typical size)
    pub size: f64,

    /// Time since last trade (milliseconds)
    pub inter_arrival_ms: u64,

    /// Price impact in basis points (measured after impact_horizon_ms)
    pub price_impact_bps: f64,

    /// Book imbalance at time of trade (-1 to 1)
    pub book_imbalance: f64,

    /// Trade direction
    pub is_buy: bool,

    /// Timestamp (milliseconds)
    pub timestamp_ms: u64,
}

impl Default for TradeFeatures {
    fn default() -> Self {
        Self {
            size: 1.0,
            inter_arrival_ms: 1000,
            price_impact_bps: 0.0,
            book_imbalance: 0.0,
            is_buy: true,
            timestamp_ms: 0,
        }
    }
}

// ============================================================================
// Flow Decomposition Result
// ============================================================================

/// Result of flow decomposition
#[derive(Debug, Clone, Copy, Default)]
pub struct FlowDecomposition {
    /// Probability trade is from informed trader
    pub p_informed: f64,

    /// Probability trade is noise
    pub p_noise: f64,

    /// Probability trade is forced (liquidation/rebalance)
    pub p_forced: f64,

    /// Confidence in decomposition (based on recent consistency)
    pub confidence: f64,

    /// Number of observations used
    pub n_observations: usize,
}

impl FlowDecomposition {
    /// Check if primarily informed flow
    pub fn is_informed_dominated(&self) -> bool {
        self.p_informed > self.p_noise && self.p_informed > self.p_forced
    }

    /// Check if primarily noise flow
    pub fn is_noise_dominated(&self) -> bool {
        self.p_noise > self.p_informed && self.p_noise > self.p_forced
    }

    /// Check if primarily forced flow
    pub fn is_forced_dominated(&self) -> bool {
        self.p_forced > self.p_informed && self.p_forced > self.p_noise
    }

    /// Toxicity score: how unfavorable is this flow for market makers
    /// Higher = more adverse selection expected
    pub fn toxicity_score(&self) -> f64 {
        // Informed flow is most toxic, forced is somewhat toxic
        self.p_informed * 1.0 + self.p_forced * 0.5
    }
}

// ============================================================================
// Informed Flow Estimator
// ============================================================================

/// Online EM-based estimator for trade flow decomposition
#[derive(Debug)]
pub struct InformedFlowEstimator {
    /// Component parameters [informed, noise, forced]
    components: [ComponentParams; 3],

    /// Component mixing weights (π_k)
    mixing_weights: [f64; 3],

    /// Recent responsibilities (for tracking)
    responsibilities: VecDeque<[f64; 3]>,

    /// Trade feature buffer for delayed M-step
    feature_buffer: VecDeque<TradeFeatures>,

    /// Sufficient statistics for online EM
    sufficient_stats: SufficientStats,

    /// Configuration
    config: InformedFlowConfig,

    /// Number of observations processed
    observation_count: usize,

    /// Trades since last EM update
    trades_since_update: usize,

    /// Exponential forgetting factor
    forgetting_factor: f64,

    /// Last trade timestamp
    last_trade_ms: u64,
}

/// Sufficient statistics for online EM
#[derive(Debug, Clone, Default)]
struct SufficientStats {
    // Per-component stats
    n: [f64; 3],            // Weighted count
    sum_log_size: [f64; 3], // Σ γ × log(size)
    sum_log_size_sq: [f64; 3], // Σ γ × log(size)²
    sum_inter_arrival: [f64; 3], // Σ γ × inter_arrival
    sum_impact: [f64; 3],   // Σ γ × impact
    sum_impact_sq: [f64; 3], // Σ γ × impact²
}

impl InformedFlowEstimator {
    /// Create a new estimator with default configuration
    pub fn new(config: InformedFlowConfig) -> Self {
        // Initialize components with informed/noise/forced priors
        let components = [
            ComponentParams::informed(),
            ComponentParams::noise(),
            ComponentParams::forced(),
        ];

        let forgetting_factor = 0.5f64.powf(1.0 / config.observation_half_life as f64);

        Self {
            components,
            mixing_weights: config.prior_probs,
            responsibilities: VecDeque::with_capacity(config.buffer_size),
            feature_buffer: VecDeque::with_capacity(config.buffer_size),
            sufficient_stats: SufficientStats::default(),
            config,
            observation_count: 0,
            trades_since_update: 0,
            forgetting_factor,
            last_trade_ms: 0,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(InformedFlowConfig::default())
    }

    /// Process a new trade observation
    pub fn on_trade(&mut self, features: &TradeFeatures) {
        // Calculate inter-arrival time
        let inter_arrival_ms = if self.last_trade_ms > 0 {
            features.timestamp_ms.saturating_sub(self.last_trade_ms)
        } else {
            features.inter_arrival_ms
        };
        self.last_trade_ms = features.timestamp_ms;

        // Create feature copy with computed inter-arrival
        let mut feat = *features;
        feat.inter_arrival_ms = inter_arrival_ms;

        // E-step: compute responsibilities
        let resp = self.e_step(&feat);

        // Store for tracking
        if self.responsibilities.len() >= self.config.buffer_size {
            self.responsibilities.pop_front();
        }
        self.responsibilities.push_back(resp);

        // Store feature for M-step
        if self.feature_buffer.len() >= self.config.buffer_size {
            self.feature_buffer.pop_front();
        }
        self.feature_buffer.push_back(feat);

        // Update sufficient statistics
        self.update_sufficient_stats(&feat, &resp);

        self.observation_count += 1;
        self.trades_since_update += 1;

        // Periodic M-step
        if self.trades_since_update >= self.config.em_update_interval {
            self.m_step();
            self.trades_since_update = 0;
        }
    }

    /// E-step: compute responsibilities P(component | trade)
    fn e_step(&self, features: &TradeFeatures) -> [f64; 3] {
        let mut log_likelihoods = [0.0f64; 3];

        for (k, component) in self.components.iter().enumerate() {
            // Size likelihood (log-normal)
            let log_size = (features.size.max(0.01)).ln();
            let size_ll = gaussian_log_pdf(
                log_size,
                component.mu_size,
                component.sigma_size,
            );

            // Inter-arrival likelihood (exponential approximation)
            let arrival_rate = component.lambda_0
                + component.alpha * (-(component.beta * features.inter_arrival_ms as f64 / 1000.0)).exp();
            let inter_arrival_s = features.inter_arrival_ms as f64 / 1000.0;
            let arrival_ll = arrival_rate.ln() - arrival_rate * inter_arrival_s;

            // Impact likelihood (Gaussian)
            let expected_impact = component.impact_per_unit * features.size;
            let impact_ll = gaussian_log_pdf(
                features.price_impact_bps,
                expected_impact,
                component.impact_variance.sqrt(),
            );

            // Combine log-likelihoods with mixing weight
            log_likelihoods[k] = self.mixing_weights[k].ln() + size_ll + arrival_ll + impact_ll;
        }

        // Softmax to get responsibilities
        let max_ll = log_likelihoods.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut responsibilities = [0.0f64; 3];
        let mut sum = 0.0;

        for k in 0..3 {
            responsibilities[k] = (log_likelihoods[k] - max_ll).exp();
            sum += responsibilities[k];
        }

        // Normalize
        if sum > 0.0 {
            for r in &mut responsibilities {
                *r /= sum;
            }
        } else {
            // Fallback to prior
            responsibilities = self.config.prior_probs;
        }

        responsibilities
    }

    /// Update sufficient statistics with forgetting
    fn update_sufficient_stats(&mut self, features: &TradeFeatures, resp: &[f64; 3]) {
        let lambda = self.forgetting_factor;
        let log_size = (features.size.max(0.01)).ln();
        let inter_arrival_s = features.inter_arrival_ms as f64 / 1000.0;

        for k in 0..3 {
            let gamma = resp[k];

            // Decay old stats and add new
            self.sufficient_stats.n[k] = lambda * self.sufficient_stats.n[k] + gamma;
            self.sufficient_stats.sum_log_size[k] =
                lambda * self.sufficient_stats.sum_log_size[k] + gamma * log_size;
            self.sufficient_stats.sum_log_size_sq[k] =
                lambda * self.sufficient_stats.sum_log_size_sq[k] + gamma * log_size * log_size;
            self.sufficient_stats.sum_inter_arrival[k] =
                lambda * self.sufficient_stats.sum_inter_arrival[k] + gamma * inter_arrival_s;
            self.sufficient_stats.sum_impact[k] =
                lambda * self.sufficient_stats.sum_impact[k] + gamma * features.price_impact_bps;
            self.sufficient_stats.sum_impact_sq[k] =
                lambda * self.sufficient_stats.sum_impact_sq[k]
                    + gamma * features.price_impact_bps * features.price_impact_bps;
        }
    }

    /// M-step: update component parameters from sufficient statistics
    fn m_step(&mut self) {
        let total_n: f64 = self.sufficient_stats.n.iter().sum();
        if total_n < 1.0 {
            return;
        }

        for k in 0..3 {
            let n_k = self.sufficient_stats.n[k];
            if n_k < 0.1 {
                continue;
            }

            // Update mixing weight
            self.mixing_weights[k] = n_k / total_n;

            // Update size distribution (log-normal)
            let mean_log_size = self.sufficient_stats.sum_log_size[k] / n_k;
            let mean_log_size_sq = self.sufficient_stats.sum_log_size_sq[k] / n_k;
            let var_log_size = (mean_log_size_sq - mean_log_size * mean_log_size).max(0.01);

            self.components[k].mu_size = mean_log_size;
            self.components[k].sigma_size = var_log_size.sqrt().clamp(0.1, 3.0);

            // Update arrival rate (simplified - just baseline)
            let mean_inter_arrival = self.sufficient_stats.sum_inter_arrival[k] / n_k;
            if mean_inter_arrival > 0.0 {
                self.components[k].lambda_0 = (1.0 / mean_inter_arrival).clamp(0.001, 10.0);
            }

            // Update impact model
            let mean_impact = self.sufficient_stats.sum_impact[k] / n_k;
            let mean_impact_sq = self.sufficient_stats.sum_impact_sq[k] / n_k;
            let var_impact = (mean_impact_sq - mean_impact * mean_impact).max(0.01);

            // impact_per_unit = mean_impact / mean_size (approximate)
            let mean_size = (mean_log_size + 0.5 * var_log_size).exp();
            if mean_size > 0.01 {
                self.components[k].impact_per_unit = (mean_impact / mean_size).clamp(-10.0, 10.0);
            }
            self.components[k].impact_variance = var_impact.clamp(0.1, 100.0);
        }

        // Normalize mixing weights
        let sum: f64 = self.mixing_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.mixing_weights {
                *w /= sum;
            }
        }
    }

    // ========================================================================
    // Public Getters
    // ========================================================================

    /// Get current flow decomposition (based on recent trades)
    pub fn decomposition(&self) -> FlowDecomposition {
        if self.responsibilities.is_empty() {
            return FlowDecomposition {
                p_informed: self.config.prior_probs[0],
                p_noise: self.config.prior_probs[1],
                p_forced: self.config.prior_probs[2],
                confidence: 0.0,
                n_observations: 0,
            };
        }

        // Average recent responsibilities
        let n_recent = self.responsibilities.len().min(50);
        let recent: Vec<_> = self.responsibilities.iter().rev().take(n_recent).collect();

        let mut avg = [0.0f64; 3];
        for resp in &recent {
            for k in 0..3 {
                avg[k] += resp[k];
            }
        }
        for k in 0..3 {
            avg[k] /= n_recent as f64;
        }

        // Compute confidence based on consistency
        let mut variance = 0.0;
        for resp in &recent {
            for k in 0..3 {
                let diff = resp[k] - avg[k];
                variance += diff * diff;
            }
        }
        variance /= (n_recent * 3) as f64;
        let confidence = (1.0 - variance.sqrt() * 2.0).clamp(0.0, 1.0);

        FlowDecomposition {
            p_informed: avg[0],
            p_noise: avg[1],
            p_forced: avg[2],
            confidence,
            n_observations: self.observation_count,
        }
    }

    /// Predict probability that next trade is informed
    pub fn p_next_informed(&self) -> f64 {
        self.mixing_weights[0]
    }

    /// Predict probability that next trade is noise
    pub fn p_next_noise(&self) -> f64 {
        self.mixing_weights[1]
    }

    /// Predict probability that next trade is forced
    pub fn p_next_forced(&self) -> f64 {
        self.mixing_weights[2]
    }

    /// Get component parameters
    pub fn component_params(&self) -> &[ComponentParams; 3] {
        &self.components
    }

    /// Get mixing weights
    pub fn mixing_weights(&self) -> &[f64; 3] {
        &self.mixing_weights
    }

    /// Number of observations processed
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Check if estimator has enough observations
    pub fn is_warmed_up(&self) -> bool {
        self.observation_count >= self.config.min_observations
    }

    /// Reset estimator to initial state
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone());
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Gaussian log probability density
fn gaussian_log_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * z * z
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimator_initialization() {
        let estimator = InformedFlowEstimator::default_config();
        assert_eq!(estimator.observation_count(), 0);
        assert!(!estimator.is_warmed_up());

        let decomp = estimator.decomposition();
        assert!((decomp.p_informed - 0.05).abs() < 0.01);
        assert!((decomp.p_noise - 0.85).abs() < 0.01);
        assert!((decomp.p_forced - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_estimator_warmup() {
        let mut estimator = InformedFlowEstimator::default_config();

        for i in 0..150 {
            let features = TradeFeatures {
                size: 1.0,
                inter_arrival_ms: 1000,
                price_impact_bps: 1.0,
                timestamp_ms: i * 1000,
                ..Default::default()
            };
            estimator.on_trade(&features);
        }

        assert!(estimator.is_warmed_up());
    }

    #[test]
    fn test_informed_flow_detection() {
        let mut estimator = InformedFlowEstimator::default_config();

        // Feed trades that look informed: large, clustered, high impact
        // Informed component expects impact_per_unit = 3.0, so for size 5,
        // expected impact is ~15 bps (vs forced's 10 bps, noise's 2.5 bps)
        for i in 0..200 {
            let features = TradeFeatures {
                size: 5.0,               // Large size
                inter_arrival_ms: 100,   // Clustered (fast)
                price_impact_bps: 15.0,  // High impact matching informed expectations
                timestamp_ms: i * 100,
                ..Default::default()
            };
            estimator.on_trade(&features);
        }

        let decomp = estimator.decomposition();
        assert!(
            decomp.p_informed > 0.3,
            "Expected informed flow > 0.3, got {}",
            decomp.p_informed
        );
    }

    #[test]
    fn test_noise_flow_detection() {
        let mut estimator = InformedFlowEstimator::default_config();

        // Feed trades that look like noise: small, random, low impact
        for i in 0..200 {
            let features = TradeFeatures {
                size: 0.3,               // Small size
                inter_arrival_ms: 5000,  // Sparse (slow)
                price_impact_bps: 0.5,   // Low impact
                timestamp_ms: i * 5000,
                ..Default::default()
            };
            estimator.on_trade(&features);
        }

        let decomp = estimator.decomposition();
        assert!(
            decomp.p_noise > 0.5,
            "Expected noise flow > 0.5, got {}",
            decomp.p_noise
        );
    }

    #[test]
    fn test_flow_decomposition_sums_to_one() {
        let mut estimator = InformedFlowEstimator::default_config();

        for i in 0..100 {
            let features = TradeFeatures {
                size: 1.0,
                inter_arrival_ms: 1000,
                price_impact_bps: 2.0,
                timestamp_ms: i * 1000,
                ..Default::default()
            };
            estimator.on_trade(&features);
        }

        let decomp = estimator.decomposition();
        let sum = decomp.p_informed + decomp.p_noise + decomp.p_forced;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Decomposition should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_toxicity_score() {
        let decomp = FlowDecomposition {
            p_informed: 0.5,
            p_noise: 0.3,
            p_forced: 0.2,
            confidence: 0.8,
            n_observations: 100,
        };

        // toxicity = 0.5 * 1.0 + 0.2 * 0.5 = 0.6
        let toxicity = decomp.toxicity_score();
        assert!((toxicity - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut estimator = InformedFlowEstimator::default_config();

        for i in 0..150 {
            let features = TradeFeatures {
                size: 1.0,
                timestamp_ms: i * 1000,
                ..Default::default()
            };
            estimator.on_trade(&features);
        }
        assert!(estimator.is_warmed_up());

        estimator.reset();
        assert!(!estimator.is_warmed_up());
        assert_eq!(estimator.observation_count(), 0);
    }

    #[test]
    fn test_component_params() {
        let informed = ComponentParams::informed();
        let noise = ComponentParams::noise();
        let forced = ComponentParams::forced();

        // Informed should have larger size mean
        assert!(informed.mu_size > noise.mu_size);

        // Informed should have higher clustering (alpha)
        assert!(informed.alpha > noise.alpha);

        // Informed should have higher impact
        assert!(informed.impact_per_unit > noise.impact_per_unit);

        // Forced should have highest clustering
        assert!(forced.alpha > informed.alpha);
    }
}
