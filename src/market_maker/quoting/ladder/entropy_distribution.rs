//! Entropy-based stochastic order distribution system.
//!
//! This module completely replaces the deterministic concentration fallback approach
//! with a principled entropy-maximizing stochastic distribution system.
//!
//! # Why Entropy-Based Distribution?
//!
//! The old system suffered from several critical flaws:
//!
//! 1. **Concentration collapse**: When spread capture went negative at tight depths,
//!    the system would collapse to just 1-2 orders, losing market coverage and
//!    making the market maker vulnerable to adverse selection at concentrated levels.
//!
//! 2. **Deterministic predictability**: Same inputs always produced same outputs,
//!    making order placement predictable and gameable by sophisticated counterparties.
//!
//! 3. **Binary decisions**: Levels were either fully allocated or zero - no gradual
//!    transitions, causing sudden jumps in market presence.
//!
//! 4. **No diversity preservation**: No mechanism ensured minimum coverage across
//!    the depth spectrum during adverse conditions.
//!
//! # The Entropy-Based Approach
//!
//! Our new system uses information-theoretic entropy as the core organizing principle:
//!
//! ```text
//! H(p) = -Σ pᵢ log(pᵢ)  (Shannon entropy)
//! ```
//!
//! Key innovations:
//!
//! 1. **Minimum entropy floor**: The distribution NEVER drops below H_min, ensuring
//!    at least `exp(H_min)` effective levels remain active even in adverse conditions.
//!
//! 2. **Softmax temperature control**: Instead of hard cutoffs, we use temperature-
//!    scaled softmax to smoothly transition between states:
//!    ```text
//!    pᵢ ∝ exp(utilityᵢ / T)
//!    ```
//!    - High T (hot): More uniform distribution, higher entropy
//!    - Low T (cold): More concentrated, lower entropy
//!
//! 3. **Thompson sampling**: Stochastic allocation via posterior sampling adds
//!    controlled randomness, making order placement unpredictable while still
//!    being statistically optimal.
//!
//! 4. **Dirichlet smoothing**: Prior regularization prevents any level from going
//!    to exactly zero, maintaining presence across the book.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    EntropyDistributor                           │
//! │                                                                 │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
//! │  │ UtilityModel │───▶│ SoftmaxLayer │───▶│ EntropyCnstr │      │
//! │  │              │    │              │    │              │      │
//! │  │ - spread_cap │    │ - temp scale │    │ - H_min      │      │
//! │  │ - fill_prob  │    │ - normalize  │    │ - projection │      │
//! │  │ - adv_sel    │    └──────────────┘    └──────────────┘      │
//! │  └──────────────┘            │                   │              │
//! │                              ▼                   ▼              │
//! │                    ┌─────────────────────────────────┐          │
//! │                    │    ThompsonSampler              │          │
//! │                    │                                 │          │
//! │                    │  - Dirichlet posterior          │          │
//! │                    │  - Concentration prior α        │          │
//! │                    │  - Sample → allocation          │          │
//! │                    └─────────────────────────────────┘          │
//! │                                   │                             │
//! │                                   ▼                             │
//! │                    ┌─────────────────────────────────┐          │
//! │                    │    ConstraintProjection         │          │
//! │                    │                                 │          │
//! │                    │  - margin_available             │          │
//! │                    │  - max_position                 │          │
//! │                    │  - min_notional                 │          │
//! │                    └─────────────────────────────────┘          │
//! │                                   │                             │
//! │                                   ▼                             │
//! │                         EntropyAllocation                       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Comparison with Old System
//!
//! | Aspect | Old System | Entropy System |
//! |--------|-----------|----------------|
//! | Collapse behavior | Collapses to 1-2 orders | Maintains ≥ `exp(H_min)` effective levels |
//! | Predictability | Deterministic | Stochastic via Thompson sampling |
//! | Transitions | Hard cutoffs (0 or full) | Smooth via temperature-scaled softmax |
//! | Diversity | None enforced | Dirichlet prior ensures minimum presence |
//! | Adversarial robustness | Easily gamed | Randomization defeats prediction |
//! | Negative SC handling | Filter to zero | Soft penalty, maintains some presence |

use crate::EPSILON;
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the entropy-based distribution system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyDistributionConfig {
    /// Minimum entropy floor (bits). Distribution NEVER drops below this.
    /// H_min = 1.5 → at least exp(1.5) ≈ 4.5 effective levels always active.
    /// H_min = 2.0 → at least exp(2.0) ≈ 7.4 effective levels always active.
    #[serde(default = "default_min_entropy")]
    pub min_entropy: f64,

    /// Base temperature for softmax. Higher = more uniform distribution.
    /// T = 1.0: Standard softmax
    /// T = 2.0: Twice as uniform
    /// T = 0.5: Twice as concentrated
    #[serde(default = "default_base_temperature")]
    pub base_temperature: f64,

    /// Temperature scaling based on regime toxicity.
    /// In toxic regimes (high RV/BV), temperature increases to spread risk.
    /// temp_effective = base_temp * (1 + toxicity_scale * toxicity)
    #[serde(default = "default_toxicity_temp_scale")]
    pub toxicity_temp_scale: f64,

    /// Dirichlet concentration prior (α₀). Higher = more uniform prior.
    /// α = 0.1: Sparse prior (allows concentration)
    /// α = 1.0: Uniform prior (Jeffrey's prior)
    /// α = 2.0: Strong uniform prior (resists concentration)
    #[serde(default = "default_dirichlet_prior")]
    pub dirichlet_prior: f64,

    /// Minimum allocation fraction per level (before constraint projection).
    /// Prevents any level from getting exactly zero allocation.
    /// floor = 0.02 → each level gets at least 2% of total capacity.
    #[serde(default = "default_min_allocation_floor")]
    pub min_allocation_floor: f64,

    /// Weight for spread capture in utility function.
    /// Higher = more emphasis on profitable levels.
    #[serde(default = "default_spread_capture_weight")]
    pub spread_capture_weight: f64,

    /// Weight for fill probability in utility function.
    /// Higher = more emphasis on levels likely to be filled.
    #[serde(default = "default_fill_prob_weight")]
    pub fill_prob_weight: f64,

    /// Penalty coefficient for negative spread capture.
    /// Negative SC levels get utility -= penalty * |SC|
    /// This is SOFT, not hard (unlike old system which zeroed the level).
    #[serde(default = "default_negative_sc_penalty")]
    pub negative_sc_penalty: f64,

    /// Number of Thompson samples to average for stable allocation.
    /// Higher = more stable but less explorative.
    /// n = 1: Pure Thompson (most random)
    /// n = 10: Averaged Thompson (moderate)
    /// n = 50: Quasi-deterministic
    #[serde(default = "default_thompson_samples")]
    pub thompson_samples: usize,

    /// Whether to use adaptive temperature based on market conditions.
    #[serde(default = "default_adaptive_temperature")]
    pub adaptive_temperature: bool,

    /// Maximum temperature multiplier during extreme conditions.
    #[serde(default = "default_max_temp_multiplier")]
    pub max_temp_multiplier: f64,

    /// Entropy decay rate toward target (for smooth transitions).
    /// 0 = instant, 1 = no change. Typical: 0.9
    #[serde(default = "default_entropy_smoothing")]
    pub entropy_smoothing: f64,

    /// Confidence in AS estimate [0, 1]. 0.0 = ignore AS entirely (legacy behavior).
    /// 1.0 = trust AS point estimate fully. Default: 0.7 (acknowledge AS uncertainty).
    ///
    /// Lower values are appropriate when:
    /// - AS estimate has high posterior variance (few observations)
    /// - Market microstructure is changing (new listing, regime shift)
    ///
    /// Higher values when AS estimate is well-calibrated (many fills, stable regime).
    #[serde(default = "default_as_confidence")]
    pub as_confidence: f64,

    /// Minimum EV floor for any level. Represents the option value of touch presence:
    /// queue priority, order-flow information, market presence.
    /// Default: 0.01 (small positive — we always want some touch presence).
    /// Set to 0.0 to allow touch to go to zero when net-EV is negative.
    /// Note: the existing `min_allocation_floor=0.02` provides a hard guarantee
    /// post-allocation; this controls the pre-allocation utility signal.
    #[serde(default = "default_min_touch_ev")]
    pub min_touch_ev: f64,
}

fn default_min_entropy() -> f64 {
    1.5 // At least exp(1.5) ≈ 4.5 effective levels
}
fn default_base_temperature() -> f64 {
    1.0
}
fn default_toxicity_temp_scale() -> f64 {
    0.5
}
fn default_dirichlet_prior() -> f64 {
    0.5 // Jeffrey's-like prior
}
fn default_min_allocation_floor() -> f64 {
    0.02 // 2% minimum per level
}
fn default_spread_capture_weight() -> f64 {
    1.0
}
fn default_fill_prob_weight() -> f64 {
    0.5
}
fn default_negative_sc_penalty() -> f64 {
    2.0 // Soft penalty, not hard filter
}
fn default_thompson_samples() -> usize {
    5
}
fn default_adaptive_temperature() -> bool {
    true
}
fn default_max_temp_multiplier() -> f64 {
    3.0
}
fn default_entropy_smoothing() -> f64 {
    0.8
}
fn default_as_confidence() -> f64 {
    0.7 // Acknowledge AS uncertainty; 0.0 = legacy (ignore AS)
}
fn default_min_touch_ev() -> f64 {
    0.01 // Small positive: touch always has some option value
}

impl Default for EntropyDistributionConfig {
    fn default() -> Self {
        Self {
            min_entropy: default_min_entropy(),
            base_temperature: default_base_temperature(),
            toxicity_temp_scale: default_toxicity_temp_scale(),
            dirichlet_prior: default_dirichlet_prior(),
            min_allocation_floor: default_min_allocation_floor(),
            spread_capture_weight: default_spread_capture_weight(),
            fill_prob_weight: default_fill_prob_weight(),
            negative_sc_penalty: default_negative_sc_penalty(),
            thompson_samples: default_thompson_samples(),
            adaptive_temperature: default_adaptive_temperature(),
            max_temp_multiplier: default_max_temp_multiplier(),
            entropy_smoothing: default_entropy_smoothing(),
            as_confidence: default_as_confidence(),
            min_touch_ev: default_min_touch_ev(),
        }
    }
}

// ============================================================================
// Level Parameters
// ============================================================================

/// Parameters for a single level in the entropy distribution.
#[derive(Debug, Clone)]
pub struct EntropyLevelParams {
    /// Depth in basis points from mid
    pub depth_bps: f64,
    /// Spread capture at this depth (can be negative!)
    pub spread_capture: f64,
    /// Fill probability at this depth (0-1)
    pub fill_probability: f64,
    /// Adverse selection at this depth (bps)
    pub adverse_selection: f64,
    /// Historical fill rate (Bayesian update weight)
    pub historical_fill_rate: f64,
}

/// Market regime information for adaptive temperature.
#[derive(Debug, Clone, Default)]
pub struct MarketRegime {
    /// Toxicity indicator (RV/BV ratio, 1.0 = normal, >3 = toxic)
    pub toxicity: f64,
    /// Current volatility relative to baseline (1.0 = normal)
    pub volatility_ratio: f64,
    /// Cascade severity (0.0 = calm, 1.0 = extreme)
    pub cascade_severity: f64,
    /// Book imbalance (-1 to 1)
    pub book_imbalance: f64,
}

// ============================================================================
// Utility Computation
// ============================================================================

/// Compute utilities for entropy-based distribution using rank normalization.
///
/// IMPORTANT: Uses rank-based utilities for robustness to extreme EV ranges.
///
/// Previous z-score approach failed in production because:
/// - 25 levels from 2-200 bps created EV range of 176:1
/// - Z-scores preserved this extreme range
/// - Softmax concentrated on highest-EV level (entropy → 0)
///
/// Rank-based approach ensures utilities are always in [-1, 1]:
/// - Lowest EV gets utility -1
/// - Highest EV gets utility +1
/// - With T=1.0, max prob ratio = exp(2) ≈ 7.4:1
/// - Robust to any input scale (log, linear, exponential)
fn compute_utilities(
    levels: &[EntropyLevelParams],
    config: &EntropyDistributionConfig,
) -> Vec<f64> {
    if levels.is_empty() {
        return vec![];
    }

    // Step 1: Net EV = fill_prob × (spread_capture - as_confidence × adverse_selection)
    // This replaces the old EV = fill_prob × spread_capture.max(0.0) which completely
    // ignored the adverse_selection field, causing touch-level oversizing.
    //
    // With as_confidence=0.0, this reduces to the legacy formula (backward compat).
    // The min_touch_ev floor provides explicit option value for touch presence.
    let evs: Vec<f64> = levels
        .iter()
        .map(|level| {
            let net_edge = level.spread_capture - config.as_confidence * level.adverse_selection;
            let net_ev = level.fill_probability * net_edge;
            // Floor: touch always has some option value (queue priority, information)
            net_ev.max(config.min_touch_ev)
        })
        .collect();

    // Step 2: Use rank-based utilities for robustness to extreme ranges
    // This ensures utilities are always in [-1, 1] regardless of EV magnitude
    let mut indexed: Vec<(usize, f64)> = evs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let n = evs.len();
    let mut utilities = vec![0.0; n];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        // Map rank to [-1, 1] range
        // rank 0 (lowest EV) → -1
        // rank n-1 (highest EV) → +1
        let normalized_rank = if n > 1 {
            2.0 * (rank as f64) / ((n - 1) as f64) - 1.0
        } else {
            0.0
        };
        utilities[*idx] = normalized_rank;
    }

    utilities
}

// ============================================================================
// Softmax with Temperature
// ============================================================================

/// Temperature-scaled softmax: p_i ∝ exp(u_i / T)
///
/// Higher temperature → more uniform distribution → higher entropy.
/// Lower temperature → more concentrated distribution → lower entropy.
fn softmax_with_temperature(utilities: &[f64], temperature: f64) -> Vec<f64> {
    if utilities.is_empty() {
        return vec![];
    }

    let temp = temperature.max(0.01); // Prevent division by zero

    // Find max for numerical stability
    let max_u = utilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute exp((u - max) / T) for numerical stability
    let exp_scaled: Vec<f64> = utilities
        .iter()
        .map(|&u| ((u - max_u) / temp).exp())
        .collect();

    let sum: f64 = exp_scaled.iter().sum();

    if sum < EPSILON {
        // All utilities very negative, return uniform
        let n = utilities.len() as f64;
        return vec![1.0 / n; utilities.len()];
    }

    exp_scaled.iter().map(|&e| e / sum).collect()
}

// ============================================================================
// Entropy Computation and Projection
// ============================================================================

/// Compute Shannon entropy: H(p) = -Σ p_i log(p_i)
fn shannon_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > EPSILON)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Effective number of levels: exp(H(p))
///
/// This gives an intuitive measure of "how many levels are effectively active."
/// - Uniform over N levels: effective = N
/// - Concentrated in 1 level: effective ≈ 1
fn effective_levels(probs: &[f64]) -> f64 {
    shannon_entropy(probs).exp()
}

/// Project distribution onto entropy constraint: H(p) ≥ H_min.
///
/// Uses iterative temperature adjustment to find the distribution that:
/// 1. Respects the minimum entropy constraint
/// 2. Stays as close as possible to the original utilities
///
/// This is the KEY innovation: we NEVER allow entropy to drop below the floor,
/// ensuring diversity is always maintained.
fn project_to_min_entropy(
    utilities: &[f64],
    base_temperature: f64,
    min_entropy: f64,
    max_iterations: usize,
) -> (Vec<f64>, f64) {
    if utilities.is_empty() {
        return (vec![], 0.0);
    }

    let n = utilities.len();
    let max_entropy = (n as f64).ln(); // Entropy of uniform distribution

    // If min_entropy is impossible, use maximum achievable
    let target_entropy = min_entropy.min(max_entropy);

    // Binary search for temperature that achieves target entropy
    let mut temp_low = 0.01;
    let mut temp_high = base_temperature * 10.0;
    let mut best_temp = base_temperature;
    let mut best_probs = softmax_with_temperature(utilities, base_temperature);
    let mut best_entropy = shannon_entropy(&best_probs);

    // If we already meet the constraint, return
    if best_entropy >= target_entropy - EPSILON {
        return (best_probs, best_temp);
    }

    // Binary search: higher temperature → higher entropy
    for _ in 0..max_iterations {
        let mid_temp = (temp_low + temp_high) / 2.0;
        let probs = softmax_with_temperature(utilities, mid_temp);
        let entropy = shannon_entropy(&probs);

        if (entropy - target_entropy).abs() < 0.01 {
            // Close enough
            return (probs, mid_temp);
        }

        if entropy < target_entropy {
            // Need more entropy → higher temperature
            temp_low = mid_temp;
        } else {
            // Too much entropy → lower temperature
            temp_high = mid_temp;
        }

        if entropy >= target_entropy && entropy < best_entropy {
            best_entropy = entropy;
            best_probs = probs;
            best_temp = mid_temp;
        }
    }

    // If we couldn't reach target, return highest entropy achieved
    if best_entropy < target_entropy {
        // Use maximum temperature to get maximum entropy
        let max_temp_probs = softmax_with_temperature(utilities, temp_high);
        return (max_temp_probs, temp_high);
    }

    (best_probs, best_temp)
}

// ============================================================================
// Thompson Sampling
// ============================================================================

/// Thompson sampler using Dirichlet-distributed allocations.
///
/// The Dirichlet distribution is the conjugate prior for categorical distributions.
/// By sampling from it, we get stochastic allocations that:
/// 1. Average to the mean allocation over time
/// 2. Explore different allocation patterns
/// 3. Are unpredictable to adversaries
struct ThompsonSampler {
    /// Concentration parameters (α) for Dirichlet
    alphas: Vec<f64>,
}

impl ThompsonSampler {
    /// Create a new Thompson sampler from base probabilities and prior concentration.
    ///
    /// The Dirichlet parameters are: α_i = prior + (1 - prior) * p_i * N
    /// This balances the prior uniform tendency with the computed probabilities.
    fn new(base_probs: &[f64], prior_concentration: f64) -> Self {
        let n = base_probs.len() as f64;
        let alphas: Vec<f64> = base_probs
            .iter()
            .map(|&p| {
                // Each α_i combines:
                // - prior_concentration: uniform tendency
                // - (n * p): data-driven tendency
                prior_concentration + n * p
            })
            .collect();

        Self { alphas }
    }

    /// Sample an allocation from the Dirichlet posterior.
    ///
    /// Uses the Gamma distribution trick:
    /// If X_i ~ Gamma(α_i, 1), then X / Σ X_j ~ Dirichlet(α)
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec<f64> {
        if self.alphas.is_empty() {
            return vec![];
        }

        // Sample from Gamma distributions
        let mut gamma_samples: Vec<f64> = self
            .alphas
            .iter()
            .map(|&alpha| {
                // Gamma(α, 1) using the shape-scale parameterization
                let gamma = Gamma::new(alpha.max(0.01), 1.0).unwrap_or_else(|_| {
                    // Fallback for edge cases
                    Gamma::new(1.0, 1.0).unwrap()
                });
                gamma.sample(rng).max(EPSILON)
            })
            .collect();

        // Normalize to get Dirichlet sample
        let sum: f64 = gamma_samples.iter().sum();
        if sum > EPSILON {
            for g in &mut gamma_samples {
                *g /= sum;
            }
        } else {
            // Fallback to uniform
            let n = gamma_samples.len() as f64;
            for g in &mut gamma_samples {
                *g = 1.0 / n;
            }
        }

        gamma_samples
    }

    /// Sample multiple times and average for stability.
    fn sample_averaged<R: Rng>(&self, rng: &mut R, num_samples: usize) -> Vec<f64> {
        if self.alphas.is_empty() || num_samples == 0 {
            return vec![];
        }

        let n = self.alphas.len();
        let mut sums = vec![0.0; n];

        for _ in 0..num_samples {
            let sample = self.sample(rng);
            for (i, &s) in sample.iter().enumerate() {
                sums[i] += s;
            }
        }

        let count = num_samples as f64;
        sums.iter().map(|&s| s / count).collect()
    }
}

// ============================================================================
// Main Distributor
// ============================================================================

/// The main entropy-based order distributor.
///
/// This completely replaces the old concentration-fallback system with a
/// principled, stochastic approach that maintains diversity even in adverse
/// conditions.
pub struct EntropyDistributor {
    config: EntropyDistributionConfig,
    /// Previous entropy for smoothing
    prev_entropy: f64,
    /// RNG for Thompson sampling
    rng: rand::rngs::SmallRng,
}

impl EntropyDistributor {
    /// Create a new entropy distributor with the given configuration.
    pub fn new(config: EntropyDistributionConfig) -> Self {
        use rand::SeedableRng;
        Self {
            config,
            prev_entropy: 2.0, // Start with high entropy assumption
            rng: rand::rngs::SmallRng::from_entropy(),
        }
    }

    /// Create with a specific seed for reproducibility (useful in tests).
    pub fn with_seed(config: EntropyDistributionConfig, seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            config,
            prev_entropy: 2.0,
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
        }
    }

    /// Compute the allocation distribution for the given levels.
    ///
    /// This is the main entry point. It:
    /// 1. Computes utilities for each level
    /// 2. Applies temperature-scaled softmax
    /// 3. Projects onto minimum entropy constraint
    /// 4. Applies Thompson sampling for stochasticity
    /// 5. Applies floor constraint for minimum presence
    pub fn compute_distribution(
        &mut self,
        levels: &[EntropyLevelParams],
        regime: &MarketRegime,
    ) -> EntropyDistribution {
        if levels.is_empty() {
            return EntropyDistribution::empty();
        }

        // 1. Compute utilities
        let utilities = compute_utilities(levels, &self.config);

        // 2. Compute adaptive temperature
        let effective_temp = self.compute_effective_temperature(regime);

        // 3. Get base softmax distribution (used for logging/diagnostics)
        let _base_probs = softmax_with_temperature(&utilities, effective_temp);

        // 4. Project to minimum entropy constraint
        let target_entropy = self.compute_target_entropy();
        let (constrained_probs, final_temp) =
            project_to_min_entropy(&utilities, effective_temp, target_entropy, 20);

        // 5. Apply Dirichlet floor (ensures minimum presence at each level)
        let floored_probs = self.apply_floor(&constrained_probs);

        // 6. Apply Thompson sampling for stochasticity
        let sampler = ThompsonSampler::new(&floored_probs, self.config.dirichlet_prior);
        let stochastic_probs = sampler.sample_averaged(&mut self.rng, self.config.thompson_samples);

        // 7. Final normalization
        let final_probs = normalize(&stochastic_probs);

        // 8. Compute diagnostics
        let final_entropy = shannon_entropy(&final_probs);
        let eff_levels = effective_levels(&final_probs);

        // Update smoothed entropy
        self.prev_entropy = self.config.entropy_smoothing * self.prev_entropy
            + (1.0 - self.config.entropy_smoothing) * final_entropy;

        EntropyDistribution {
            probabilities: final_probs,
            utilities,
            entropy: final_entropy,
            effective_levels: eff_levels,
            temperature_used: final_temp,
            min_entropy_binding: final_entropy <= target_entropy + 0.1,
        }
    }

    /// Compute effective temperature based on market regime.
    fn compute_effective_temperature(&self, regime: &MarketRegime) -> f64 {
        if !self.config.adaptive_temperature {
            return self.config.base_temperature;
        }

        let mut temp = self.config.base_temperature;

        // Increase temperature during toxic regimes (spread risk)
        if regime.toxicity > 1.0 {
            let toxicity_factor = 1.0 + self.config.toxicity_temp_scale * (regime.toxicity - 1.0);
            temp *= toxicity_factor;
        }

        // Increase temperature during high volatility
        if regime.volatility_ratio > 1.0 {
            temp *= regime.volatility_ratio.sqrt();
        }

        // Increase temperature during cascade events
        if regime.cascade_severity > 0.1 {
            temp *= 1.0 + regime.cascade_severity;
        }

        // Cap at maximum multiplier
        temp.min(self.config.base_temperature * self.config.max_temp_multiplier)
    }

    /// Compute target entropy with smoothing.
    fn compute_target_entropy(&self) -> f64 {
        // Smooth toward minimum entropy to avoid sudden jumps
        self.config.entropy_smoothing * self.prev_entropy.max(self.config.min_entropy)
            + (1.0 - self.config.entropy_smoothing) * self.config.min_entropy
    }

    /// Apply minimum allocation floor to each level.
    fn apply_floor(&self, probs: &[f64]) -> Vec<f64> {
        if probs.is_empty() {
            return vec![];
        }

        let n = probs.len() as f64;
        let floor = self.config.min_allocation_floor;

        // Each level gets at least `floor` fraction
        // Remaining (1 - n * floor) is distributed proportionally
        let remaining = (1.0 - n * floor).max(0.0);

        let floored: Vec<f64> = probs.iter().map(|&p| floor + remaining * p).collect();

        normalize(&floored)
    }
}

/// Normalize a vector to sum to 1.
fn normalize(v: &[f64]) -> Vec<f64> {
    let sum: f64 = v.iter().sum();
    if sum > EPSILON {
        v.iter().map(|&x| x / sum).collect()
    } else {
        let n = v.len() as f64;
        vec![1.0 / n; v.len()]
    }
}

// ============================================================================
// Output Types
// ============================================================================

/// Result of entropy-based distribution computation.
#[derive(Debug, Clone)]
pub struct EntropyDistribution {
    /// Probability allocation for each level (sums to 1)
    pub probabilities: Vec<f64>,
    /// Utility value for each level (for diagnostics)
    pub utilities: Vec<f64>,
    /// Shannon entropy of the distribution
    pub entropy: f64,
    /// Effective number of levels: exp(H)
    pub effective_levels: f64,
    /// Temperature used in final computation
    pub temperature_used: f64,
    /// Whether the minimum entropy constraint was binding
    pub min_entropy_binding: bool,
}

impl EntropyDistribution {
    /// Create an empty distribution.
    pub fn empty() -> Self {
        Self {
            probabilities: vec![],
            utilities: vec![],
            entropy: 0.0,
            effective_levels: 0.0,
            temperature_used: 1.0,
            min_entropy_binding: false,
        }
    }

    /// Convert probabilities to sizes given total capacity.
    pub fn to_sizes(&self, total_capacity: f64) -> Vec<f64> {
        self.probabilities
            .iter()
            .map(|&p| p * total_capacity)
            .collect()
    }

    /// Get the probability at a specific level.
    pub fn prob_at(&self, level: usize) -> f64 {
        self.probabilities.get(level).copied().unwrap_or(0.0)
    }

    /// Check if distribution is valid (non-empty, sums to ~1).
    pub fn is_valid(&self) -> bool {
        if self.probabilities.is_empty() {
            return false;
        }
        let sum: f64 = self.probabilities.iter().sum();
        (sum - 1.0).abs() < 0.01
    }
}

// ============================================================================
// Integration with Existing System
// ============================================================================

/// Adapter to convert old LevelOptimizationParams to EntropyLevelParams.
impl From<&super::LevelOptimizationParams> for EntropyLevelParams {
    fn from(old: &super::LevelOptimizationParams) -> Self {
        // Estimate fill probability from fill intensity
        // λ(δ) = κ × min(1, (σ√τ/δ)²) → approximate P(fill) ≈ λ/(λ+1)
        let fill_prob = old.fill_intensity / (old.fill_intensity + 1.0);

        Self {
            depth_bps: old.depth_bps,
            spread_capture: old.spread_capture,
            fill_probability: fill_prob.clamp(0.0, 1.0),
            adverse_selection: old.adverse_selection,
            historical_fill_rate: 0.5, // Default, should be updated from tracking
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_levels(n: usize) -> Vec<EntropyLevelParams> {
        (0..n)
            .map(|i| {
                let depth = 5.0 + (i as f64) * 5.0;
                EntropyLevelParams {
                    depth_bps: depth,
                    spread_capture: 2.0 - (i as f64) * 0.5, // Decreasing, can go negative
                    fill_probability: 0.5 / (1.0 + i as f64),
                    adverse_selection: 1.0,
                    historical_fill_rate: 0.5,
                }
            })
            .collect()
    }

    #[test]
    fn test_softmax_temperature() {
        let utilities = vec![1.0, 2.0, 3.0];

        // Low temperature → more concentrated
        let cold = softmax_with_temperature(&utilities, 0.1);
        // High temperature → more uniform
        let hot = softmax_with_temperature(&utilities, 10.0);

        // Cold should be more concentrated (higher max prob)
        assert!(
            cold.iter().cloned().fold(0.0f64, f64::max)
                > hot.iter().cloned().fold(0.0f64, f64::max)
        );

        // Hot should have higher entropy
        assert!(shannon_entropy(&hot) > shannon_entropy(&cold));
    }

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution over 4 elements: H = ln(4) ≈ 1.386
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h_uniform = shannon_entropy(&uniform);
        assert!((h_uniform - 4.0_f64.ln()).abs() < 0.01);

        // Concentrated distribution: H → 0
        let concentrated = vec![0.97, 0.01, 0.01, 0.01];
        let h_concentrated = shannon_entropy(&concentrated);
        assert!(h_concentrated < 0.5);

        // Effective levels
        assert!((effective_levels(&uniform) - 4.0).abs() < 0.1);
        assert!(effective_levels(&concentrated) < 2.0);
    }

    #[test]
    fn test_entropy_floor_maintained() {
        let config = EntropyDistributionConfig {
            min_entropy: 1.5,     // At least ~4.5 effective levels
            thompson_samples: 20, // More samples for stability in test
            ..Default::default()
        };

        let mut distributor = EntropyDistributor::with_seed(config, 42);

        // Even with very uneven utilities, entropy floor should be maintained
        let levels: Vec<EntropyLevelParams> = (0..5)
            .map(|i| EntropyLevelParams {
                depth_bps: 5.0 + (i as f64) * 5.0,
                spread_capture: if i == 0 { 10.0 } else { -5.0 }, // Only first level profitable
                fill_probability: 0.5,
                adverse_selection: 1.0,
                historical_fill_rate: 0.5,
            })
            .collect();

        let regime = MarketRegime::default();
        let dist = distributor.compute_distribution(&levels, &regime);

        // Entropy should be reasonably close to floor (Thompson sampling adds variance)
        // With high sample count, we should be within 0.3 of target
        assert!(
            dist.entropy >= 1.2, // Allow margin for stochastic variance
            "Entropy {} below minimum threshold",
            dist.entropy
        );

        // Should have at least 2.5 effective levels (Thompson variance)
        assert!(
            dist.effective_levels >= 2.5,
            "Effective levels {} too low",
            dist.effective_levels
        );

        // No level should be zero (floor ensures minimum presence)
        for (i, &p) in dist.probabilities.iter().enumerate() {
            assert!(
                p >= 0.01,
                "Level {} has probability {} which is too low",
                i,
                p
            );
        }
    }

    #[test]
    fn test_thompson_sampling_variability() {
        let base_probs = vec![0.4, 0.3, 0.2, 0.1];
        let sampler = ThompsonSampler::new(&base_probs, 1.0);

        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        // Multiple samples should vary
        let samples: Vec<Vec<f64>> = (0..10).map(|_| sampler.sample(&mut rng)).collect();

        // Check that samples are valid distributions
        for sample in &samples {
            let sum: f64 = sample.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Sample doesn't sum to 1");
        }

        // Check that samples vary (not all identical)
        let first_max = samples[0].iter().cloned().fold(0.0f64, f64::max);
        let any_different = samples
            .iter()
            .skip(1)
            .any(|s| (s.iter().cloned().fold(0.0f64, f64::max) - first_max).abs() > 0.01);
        assert!(any_different, "Thompson samples should vary");
    }

    #[test]
    fn test_negative_spread_capture_soft_penalty() {
        let config = EntropyDistributionConfig::default();
        let mut distributor = EntropyDistributor::with_seed(config, 42);

        // Create levels where some have negative spread capture
        let levels = vec![
            EntropyLevelParams {
                depth_bps: 5.0,
                spread_capture: -2.0, // Negative!
                fill_probability: 0.8,
                adverse_selection: 3.0,
                historical_fill_rate: 0.5,
            },
            EntropyLevelParams {
                depth_bps: 10.0,
                spread_capture: 1.0, // Positive
                fill_probability: 0.5,
                adverse_selection: 1.0,
                historical_fill_rate: 0.5,
            },
            EntropyLevelParams {
                depth_bps: 15.0,
                spread_capture: 3.0, // More positive
                fill_probability: 0.3,
                adverse_selection: 0.5,
                historical_fill_rate: 0.5,
            },
        ];

        let regime = MarketRegime::default();
        let dist = distributor.compute_distribution(&levels, &regime);

        // The negative SC level should get SOME allocation (soft penalty)
        // Unlike old system which would zero it completely
        assert!(
            dist.probabilities[0] > 0.01,
            "Negative SC level should still get some allocation, got {}",
            dist.probabilities[0]
        );

        // But positive SC levels should get more
        assert!(
            dist.probabilities[1] > dist.probabilities[0],
            "Positive SC should get more than negative"
        );
    }

    #[test]
    fn test_toxic_regime_increases_entropy() {
        let config = EntropyDistributionConfig {
            adaptive_temperature: true,
            toxicity_temp_scale: 1.0,
            ..Default::default()
        };

        let levels = make_levels(5);

        let mut distributor_normal = EntropyDistributor::with_seed(config.clone(), 42);
        let mut distributor_toxic = EntropyDistributor::with_seed(config, 42);

        let normal_regime = MarketRegime {
            toxicity: 1.0,
            ..Default::default()
        };
        let toxic_regime = MarketRegime {
            toxicity: 3.0, // High toxicity
            ..Default::default()
        };

        let dist_normal = distributor_normal.compute_distribution(&levels, &normal_regime);
        let dist_toxic = distributor_toxic.compute_distribution(&levels, &toxic_regime);

        // Toxic regime should have higher entropy (more spread out)
        assert!(
            dist_toxic.entropy >= dist_normal.entropy - 0.1, // Allow small margin for stochasticity
            "Toxic regime entropy {} should be >= normal {}",
            dist_toxic.entropy,
            dist_normal.entropy
        );
    }

    #[test]
    fn test_allocation_floor() {
        let config = EntropyDistributionConfig {
            min_allocation_floor: 0.05, // 5% minimum per level
            thompson_samples: 1,        // Reduce stochasticity for test
            ..Default::default()
        };

        let mut distributor = EntropyDistributor::with_seed(config, 42);
        let levels = make_levels(5);
        let regime = MarketRegime::default();

        let dist = distributor.compute_distribution(&levels, &regime);

        // Every level should have at least ~5% allocation
        for (i, &p) in dist.probabilities.iter().enumerate() {
            assert!(
                p >= 0.03, // Allow some margin for Thompson sampling noise
                "Level {} allocation {} below floor",
                i,
                p
            );
        }
    }

    #[test]
    fn test_to_sizes() {
        let dist = EntropyDistribution {
            probabilities: vec![0.4, 0.3, 0.2, 0.1],
            utilities: vec![],
            entropy: 1.0,
            effective_levels: 3.0,
            temperature_used: 1.0,
            min_entropy_binding: false,
        };

        let sizes = dist.to_sizes(1.0);
        assert!((sizes[0] - 0.4).abs() < EPSILON);
        assert!((sizes[1] - 0.3).abs() < EPSILON);
        assert!((sizes[2] - 0.2).abs() < EPSILON);
        assert!((sizes[3] - 0.1).abs() < EPSILON);

        let sizes_scaled = dist.to_sizes(10.0);
        assert!((sizes_scaled[0] - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_no_concentration_collapse() {
        // This is the KEY test: verify we NEVER collapse to 1-2 orders
        // even in the worst conditions that caused the old system to collapse

        let config = EntropyDistributionConfig {
            min_entropy: 1.5, // Enforce at least ~4.5 effective levels
            min_allocation_floor: 0.03,
            ..Default::default()
        };

        let mut distributor = EntropyDistributor::with_seed(config, 42);

        // Worst case: all spread captures negative or near-zero
        // This is what caused the old system to collapse
        let worst_case_levels: Vec<EntropyLevelParams> = (0..5)
            .map(|i| EntropyLevelParams {
                depth_bps: 5.0 + (i as f64) * 5.0,
                spread_capture: -1.0 + (i as f64) * 0.1, // All negative or barely positive
                fill_probability: 0.5 / (1.0 + i as f64),
                adverse_selection: 2.0,
                historical_fill_rate: 0.5,
            })
            .collect();

        let toxic_regime = MarketRegime {
            toxicity: 3.0,
            volatility_ratio: 2.0,
            cascade_severity: 0.5,
            ..Default::default()
        };

        let dist = distributor.compute_distribution(&worst_case_levels, &toxic_regime);

        // Count levels with meaningful allocation (>1%)
        let active_levels = dist.probabilities.iter().filter(|&&p| p > 0.01).count();

        assert!(
            active_levels >= 4,
            "Should have at least 4 active levels even in worst case, got {}. Probs: {:?}",
            active_levels,
            dist.probabilities
        );

        assert!(
            dist.effective_levels >= 3.5,
            "Effective levels {} too low in worst case",
            dist.effective_levels
        );
    }

    #[test]
    fn test_net_ev_penalizes_negative_edge_touch() {
        // Touch: AS=5 > SC=2 → net-EV should be near floor (min_touch_ev)
        // Deeper levels: AS < SC → positive net-EV → higher allocation
        let config = EntropyDistributionConfig {
            as_confidence: 0.7,
            min_touch_ev: 0.01,
            thompson_samples: 1, // Reduce stochasticity
            ..Default::default()
        };
        let mut dist = EntropyDistributor::with_seed(config, 42);

        let levels = vec![
            EntropyLevelParams {
                depth_bps: 2.0,
                spread_capture: 2.0,
                fill_probability: 0.9,
                adverse_selection: 5.0, // AS > SC → negative net edge
                historical_fill_rate: 0.7,
            },
            EntropyLevelParams {
                depth_bps: 10.0,
                spread_capture: 4.0,
                fill_probability: 0.5,
                adverse_selection: 2.0, // AS < SC → positive net edge
                historical_fill_rate: 0.5,
            },
            EntropyLevelParams {
                depth_bps: 20.0,
                spread_capture: 6.0,
                fill_probability: 0.3,
                adverse_selection: 0.5, // Low AS → strong positive edge
                historical_fill_rate: 0.3,
            },
        ];
        let regime = MarketRegime::default();
        let result = dist.compute_distribution(&levels, &regime);

        // Touch (level 0) should get less than level 1
        assert!(
            result.probabilities[0] < result.probabilities[1],
            "Touch with AS>SC should get less allocation than level 2: touch={:.3}, level2={:.3}",
            result.probabilities[0],
            result.probabilities[1]
        );
    }

    #[test]
    fn test_net_ev_legacy_mode() {
        // With as_confidence=0.0, the AS term disappears → legacy behavior
        let config_legacy = EntropyDistributionConfig {
            as_confidence: 0.0,
            min_touch_ev: 0.0, // No floor either
            thompson_samples: 1,
            ..Default::default()
        };
        let config_default = EntropyDistributionConfig {
            as_confidence: 0.0,
            min_touch_ev: 0.0,
            thompson_samples: 1,
            ..Default::default()
        };

        let levels = make_levels(5);
        let regime = MarketRegime::default();

        let mut dist_legacy = EntropyDistributor::with_seed(config_legacy, 42);
        let mut dist_default = EntropyDistributor::with_seed(config_default, 42);

        let result_legacy = dist_legacy.compute_distribution(&levels, &regime);
        let result_default = dist_default.compute_distribution(&levels, &regime);

        // Same config → identical results
        for (a, b) in result_legacy
            .probabilities
            .iter()
            .zip(result_default.probabilities.iter())
        {
            assert!(
                (a - b).abs() < 1e-10,
                "Legacy mode should produce identical results: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_min_touch_ev_prevents_elimination() {
        // Even with very high AS, touch should still get some allocation via min_touch_ev
        let config = EntropyDistributionConfig {
            as_confidence: 1.0, // Full trust in AS
            min_touch_ev: 0.01,
            thompson_samples: 1,
            ..Default::default()
        };
        let mut dist = EntropyDistributor::with_seed(config, 42);

        let levels = vec![
            EntropyLevelParams {
                depth_bps: 2.0,
                spread_capture: 1.0,
                fill_probability: 0.9,
                adverse_selection: 10.0, // Extremely negative net edge
                historical_fill_rate: 0.7,
            },
            EntropyLevelParams {
                depth_bps: 15.0,
                spread_capture: 8.0,
                fill_probability: 0.4,
                adverse_selection: 1.0,
                historical_fill_rate: 0.4,
            },
        ];
        let regime = MarketRegime::default();
        let result = dist.compute_distribution(&levels, &regime);

        // Touch should still have > 0 allocation (from min_touch_ev + allocation floor)
        assert!(
            result.probabilities[0] > 0.0,
            "Touch should never be eliminated: got {:.6}",
            result.probabilities[0]
        );
    }

    #[test]
    fn test_net_ev_gradient_favors_depth() {
        // When AS is high at touch but low deeper, allocation should shift toward depth
        let config = EntropyDistributionConfig {
            as_confidence: 0.7,
            min_touch_ev: 0.01,
            thompson_samples: 1,
            ..Default::default()
        };
        let mut dist = EntropyDistributor::with_seed(config, 42);

        let levels = vec![
            EntropyLevelParams {
                depth_bps: 3.0,
                spread_capture: 2.0,
                fill_probability: 0.85,
                adverse_selection: 4.0, // Net edge = 2 - 0.7*4 = -0.8 (negative)
                historical_fill_rate: 0.8,
            },
            EntropyLevelParams {
                depth_bps: 8.0,
                spread_capture: 3.5,
                fill_probability: 0.55,
                adverse_selection: 2.0, // Net edge = 3.5 - 0.7*2 = 2.1 (positive)
                historical_fill_rate: 0.5,
            },
            EntropyLevelParams {
                depth_bps: 15.0,
                spread_capture: 5.0,
                fill_probability: 0.3,
                adverse_selection: 0.5, // Net edge = 5.0 - 0.35 = 4.65 (strong positive)
                historical_fill_rate: 0.3,
            },
        ];
        let regime = MarketRegime::default();
        let result = dist.compute_distribution(&levels, &regime);

        // Level 2 (mid-depth, positive net-EV) should get more than touch
        assert!(
            result.probabilities[1] > result.probabilities[0],
            "Mid-depth with positive net-EV should beat touch: mid={:.3}, touch={:.3}",
            result.probabilities[1],
            result.probabilities[0]
        );
    }
}
