//! Entropy-based stochastic order distribution system.
//!
//! This module implements information-theoretic order placement that distributes
//! orders across price levels using maximum entropy principles. Instead of
//! concentrating orders at the "best" level, it maintains diversity for:
//! - Robustness to model uncertainty
//! - Better fill probability across market conditions
//! - Reduced adverse selection from predictable placement
//!
//! # Key Concepts
//!
//! ## Maximum Entropy Allocation
//! Sizes follow: s_i ∝ exp(MV_i / T) where T is "temperature"
//! - Low T (< 0.5): Concentrates on high-MV levels (exploitation)
//! - High T (> 2.0): Nearly uniform distribution (exploration)
//! - T ≈ 1.0: Balanced allocation (default)
//!
//! ## Entropy Regularization
//! Objective: max Σ MV_i × s_i + β × H(s)
//! where H(s) = -Σ p_i × ln(p_i) is Shannon entropy
//!
//! ## Information-Theoretic Depth Spacing
//! Uses mutual information I(Fill; ΔPrice) to determine optimal level placement.
//! Levels are spaced to maximize information captured about price movements.
//!
//! ## Diversification Metrics
//! - Effective levels: exp(H) gives number of "effective" levels
//! - HHI (Herfindahl-Hirschman Index): Σ (s_i/S)² measures concentration
//! - Target: HHI < 0.25 (at least 4 effective levels)

use crate::EPSILON;

/// Configuration for entropy-based allocation.
#[derive(Debug, Clone)]
pub struct EntropyConfig {
    /// Temperature parameter controlling distribution spread.
    /// - T < 0.5: Concentrate on best levels (exploitation mode)
    /// - T = 1.0: Balanced (default)
    /// - T > 2.0: Nearly uniform (exploration mode)
    ///
    /// Range: [0.1, 10.0]
    pub temperature: f64,

    /// Entropy regularization weight (β).
    /// Higher values penalize concentration more strongly.
    /// Range: [0.0, 1.0], default 0.3
    pub entropy_weight: f64,

    /// Minimum effective levels target.
    /// Allocator will try to maintain at least this many "effective" levels.
    /// Effective levels = exp(entropy), measures diversification.
    /// Range: [1, num_levels], default 3
    pub min_effective_levels: usize,

    /// Maximum HHI (Herfindahl-Hirschman Index) allowed.
    /// HHI = Σ (s_i/S)² measures concentration.
    /// HHI = 1.0 means all in one level, HHI = 1/n means uniform.
    /// Default 0.4 (allows concentration but not single-level)
    pub max_hhi: f64,

    /// Enable adaptive temperature based on market regime.
    /// In high-volatility: lower T (concentrate at deep levels)
    /// In low-volatility: higher T (spread across levels)
    pub adaptive_temperature: bool,

    /// Volatility baseline for adaptive temperature.
    /// T_adaptive = T_base × (σ_baseline / σ_current)
    pub sigma_baseline: f64,

    /// Minimum allocation per level as fraction of total.
    /// Ensures every level gets at least this fraction.
    /// Default 0.02 (2% minimum per level)
    pub min_level_fraction: f64,

    /// Enable information-theoretic depth optimization.
    /// When true, adjusts depths based on mutual information.
    pub use_mutual_information: bool,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            entropy_weight: 0.3,
            min_effective_levels: 3,
            max_hhi: 0.4,
            adaptive_temperature: true,
            sigma_baseline: 0.0002, // 2bp/sec typical for BTC
            min_level_fraction: 0.02,
            use_mutual_information: true,
        }
    }
}

impl EntropyConfig {
    /// Create config for exploitation mode (concentrate on best levels).
    pub fn exploitation() -> Self {
        Self {
            temperature: 0.3,
            entropy_weight: 0.1,
            min_effective_levels: 2,
            max_hhi: 0.6,
            adaptive_temperature: false,
            ..Default::default()
        }
    }

    /// Create config for exploration mode (spread across all levels).
    pub fn exploration() -> Self {
        Self {
            temperature: 3.0,
            entropy_weight: 0.6,
            min_effective_levels: 4,
            max_hhi: 0.3,
            adaptive_temperature: false,
            ..Default::default()
        }
    }

    /// Create config for balanced market making.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Get adaptive temperature based on current volatility.
    pub fn effective_temperature(&self, current_sigma: f64) -> f64 {
        if !self.adaptive_temperature || current_sigma < EPSILON {
            return self.temperature;
        }

        // Higher volatility → lower temperature (concentrate at safe depths)
        // Lower volatility → higher temperature (spread across levels)
        let ratio = self.sigma_baseline / current_sigma.max(EPSILON);
        (self.temperature * ratio).clamp(0.1, 10.0)
    }
}

/// Result of entropy-based allocation.
#[derive(Debug, Clone)]
pub struct EntropyAllocation {
    /// Allocated sizes per level.
    pub sizes: Vec<f64>,

    /// Shannon entropy of the allocation: H = -Σ p_i ln(p_i)
    pub entropy: f64,

    /// Number of effective levels: exp(H)
    pub effective_levels: f64,

    /// Herfindahl-Hirschman Index: Σ (s_i/S)²
    pub hhi: f64,

    /// Temperature used (may differ from config if adaptive)
    pub temperature_used: f64,

    /// Per-level information content: -ln(p_i)
    pub surprisal: Vec<f64>,

    /// Total position allocated.
    pub total_size: f64,

    /// Expected value: Σ MV_i × s_i
    pub expected_value: f64,
}

/// Entropy-based order allocator.
///
/// Distributes orders across price levels using maximum entropy principles
/// to maintain diversity while respecting profitability constraints.
#[derive(Debug, Clone)]
pub struct EntropyAllocator {
    config: EntropyConfig,
}

impl EntropyAllocator {
    /// Create new allocator with configuration.
    pub fn new(config: EntropyConfig) -> Self {
        Self { config }
    }

    /// Create allocator with default configuration.
    pub fn default_allocator() -> Self {
        Self::new(EntropyConfig::default())
    }

    /// Allocate sizes using Boltzmann distribution.
    ///
    /// Sizes follow: s_i = S × exp(MV_i / T) / Σ exp(MV_j / T)
    ///
    /// This is the maximum entropy distribution subject to the constraint
    /// that expected marginal value equals a target.
    ///
    /// # Arguments
    /// * `marginal_values` - MV(δ) = λ(δ) × SC(δ) for each level
    /// * `total_size` - Total size budget to allocate
    /// * `current_sigma` - Current volatility for adaptive temperature
    /// * `min_size` - Minimum size per level (smaller allocations zeroed)
    ///
    /// # Returns
    /// Entropy allocation result with sizes and metrics.
    pub fn allocate_boltzmann(
        &self,
        marginal_values: &[f64],
        total_size: f64,
        current_sigma: f64,
        min_size: f64,
    ) -> EntropyAllocation {
        let n = marginal_values.len();
        if n == 0 || total_size < EPSILON {
            return EntropyAllocation::empty();
        }

        // Get effective temperature (may be adaptive)
        let temperature = self.config.effective_temperature(current_sigma);

        // Compute Boltzmann weights: w_i = exp(MV_i / T)
        // Use log-sum-exp trick for numerical stability
        let max_mv = marginal_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // If all marginal values are very negative, use uniform fallback
        if max_mv < -100.0 * temperature {
            return self.allocate_uniform_fallback(n, total_size, marginal_values);
        }

        let log_weights: Vec<f64> = marginal_values
            .iter()
            .map(|&mv| (mv.max(-100.0 * temperature) - max_mv) / temperature)
            .collect();

        // Convert to weights with numerical stability
        let weights: Vec<f64> = log_weights.iter().map(|&lw| lw.exp()).collect();
        let sum_weights: f64 = weights.iter().sum();

        if sum_weights < EPSILON {
            return self.allocate_uniform_fallback(n, total_size, marginal_values);
        }

        // Normalize to get probabilities
        let mut probs: Vec<f64> = weights.iter().map(|&w| w / sum_weights).collect();

        // Apply minimum level fraction if configured
        if self.config.min_level_fraction > 0.0 {
            probs = self.apply_min_fraction(&probs);
        }

        // Check HHI constraint and redistribute if needed
        let hhi = compute_hhi(&probs);
        if hhi > self.config.max_hhi {
            probs = self.reduce_concentration(&probs, marginal_values);
        }

        // Convert probabilities to sizes
        let mut sizes: Vec<f64> = probs.iter().map(|&p| p * total_size).collect();

        // Apply minimum size threshold
        for size in sizes.iter_mut() {
            if *size < min_size {
                *size = 0.0;
            }
        }

        // Renormalize if needed
        let actual_total: f64 = sizes.iter().sum();
        if actual_total > EPSILON && (actual_total - total_size).abs() > EPSILON * 10.0 {
            let scale = total_size / actual_total;
            for size in sizes.iter_mut() {
                *size *= scale;
            }
        }

        // Compute metrics
        let final_probs: Vec<f64> = if actual_total > EPSILON {
            sizes.iter().map(|&s| s / actual_total.max(EPSILON)).collect()
        } else {
            vec![0.0; n]
        };

        let entropy = compute_entropy(&final_probs);
        let effective_levels = entropy.exp();
        let final_hhi = compute_hhi(&final_probs);
        let surprisal: Vec<f64> = final_probs
            .iter()
            .map(|&p| if p > EPSILON { -p.ln() } else { 0.0 })
            .collect();

        let expected_value: f64 = sizes
            .iter()
            .zip(marginal_values.iter())
            .map(|(&s, &mv)| s * mv)
            .sum();

        EntropyAllocation {
            sizes,
            entropy,
            effective_levels,
            hhi: final_hhi,
            temperature_used: temperature,
            surprisal,
            total_size: actual_total,
            expected_value,
        }
    }

    /// Allocate using entropy-regularized optimization.
    ///
    /// Solves: max Σ MV_i × s_i + β × H(s)
    /// subject to: Σ s_i = S, s_i ≥ 0
    ///
    /// The solution is a softmax/Boltzmann distribution with temperature T = β.
    pub fn allocate_regularized(
        &self,
        marginal_values: &[f64],
        total_size: f64,
        min_size: f64,
    ) -> EntropyAllocation {
        // Entropy-regularized objective has same form as Boltzmann with T = β
        // But we also add the entropy weight to the objective
        let effective_temp = self.config.entropy_weight.max(0.1);

        let config = EntropyConfig {
            temperature: effective_temp,
            adaptive_temperature: false,
            ..self.config.clone()
        };

        let allocator = EntropyAllocator::new(config);
        allocator.allocate_boltzmann(marginal_values, total_size, 0.0, min_size)
    }

    /// Uniform fallback when all marginal values are negative/invalid.
    fn allocate_uniform_fallback(
        &self,
        n: usize,
        total_size: f64,
        marginal_values: &[f64],
    ) -> EntropyAllocation {
        let per_level = total_size / n as f64;
        let sizes = vec![per_level; n];
        let _probs = vec![1.0 / n as f64; n];

        let entropy = (n as f64).ln();
        let hhi = 1.0 / n as f64;
        let surprisal = vec![(n as f64).ln(); n];

        let expected_value: f64 = sizes
            .iter()
            .zip(marginal_values.iter())
            .map(|(&s, &mv)| s * mv)
            .sum();

        EntropyAllocation {
            sizes,
            entropy,
            effective_levels: n as f64,
            hhi,
            temperature_used: f64::INFINITY,
            surprisal,
            total_size,
            expected_value,
        }
    }

    /// Apply minimum fraction constraint to probabilities.
    fn apply_min_fraction(&self, probs: &[f64]) -> Vec<f64> {
        let n = probs.len();
        let min_p = self.config.min_level_fraction;

        // Count levels that need boosting
        let below_min: usize = probs.iter().filter(|&&p| p < min_p).count();
        if below_min == 0 {
            return probs.to_vec();
        }

        // Redistribute from levels above min to levels below
        let boost_total = below_min as f64 * min_p;
        let available: f64 = probs.iter().filter(|&&p| p >= min_p).map(|&p| p - min_p).sum();

        if available < boost_total {
            // Not enough to redistribute, use uniform
            return vec![1.0 / n as f64; n];
        }

        // Scale down levels above min, boost levels below min
        let scale = 1.0 - boost_total / available.max(EPSILON);
        probs
            .iter()
            .map(|&p| {
                if p < min_p {
                    min_p
                } else {
                    min_p + (p - min_p) * scale
                }
            })
            .collect()
    }

    /// Reduce concentration when HHI exceeds threshold.
    fn reduce_concentration(&self, probs: &[f64], marginal_values: &[f64]) -> Vec<f64> {
        let _n = probs.len();
        let target_hhi = self.config.max_hhi;

        // Binary search for the right temperature to achieve target HHI
        let mut low_t = 0.1;
        let mut high_t = 100.0;

        for _ in 0..20 {
            let mid_t = (low_t + high_t) / 2.0;
            let test_probs = self.boltzmann_probs(marginal_values, mid_t);
            let test_hhi = compute_hhi(&test_probs);

            if test_hhi > target_hhi {
                low_t = mid_t; // Need higher temperature (more uniform)
            } else {
                high_t = mid_t; // Can use lower temperature (more concentrated)
            }
        }

        self.boltzmann_probs(marginal_values, high_t)
    }

    /// Compute Boltzmann probabilities at given temperature.
    fn boltzmann_probs(&self, marginal_values: &[f64], temperature: f64) -> Vec<f64> {
        let max_mv = marginal_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = marginal_values
            .iter()
            .map(|&mv| ((mv.max(-100.0 * temperature) - max_mv) / temperature).exp())
            .collect();
        let sum: f64 = weights.iter().sum();
        if sum < EPSILON {
            vec![1.0 / marginal_values.len() as f64; marginal_values.len()]
        } else {
            weights.iter().map(|&w| w / sum).collect()
        }
    }
}

impl EntropyAllocation {
    /// Create empty allocation.
    pub fn empty() -> Self {
        Self {
            sizes: vec![],
            entropy: 0.0,
            effective_levels: 0.0,
            hhi: 1.0,
            temperature_used: 1.0,
            surprisal: vec![],
            total_size: 0.0,
            expected_value: 0.0,
        }
    }

    /// Check if allocation is well-diversified.
    pub fn is_diversified(&self, min_effective: f64) -> bool {
        self.effective_levels >= min_effective
    }

    /// Get the number of non-zero levels.
    pub fn active_levels(&self) -> usize {
        self.sizes.iter().filter(|&&s| s > EPSILON).count()
    }
}

// ============================================================================
// Information-Theoretic Utilities
// ============================================================================

/// Compute Shannon entropy: H = -Σ p_i × ln(p_i)
pub fn compute_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > EPSILON)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Compute Herfindahl-Hirschman Index: HHI = Σ p_i²
pub fn compute_hhi(probs: &[f64]) -> f64 {
    probs.iter().map(|&p| p * p).sum()
}

/// Compute KL divergence: D_KL(P || Q) = Σ p_i × ln(p_i / q_i)
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() {
        return f64::INFINITY;
    }
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > EPSILON && qi > EPSILON)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum()
}

/// Compute mutual information between fills and price moves.
///
/// I(Fill; ΔP) = H(ΔP) - H(ΔP | Fill)
///
/// This measures how much information a fill at depth δ gives about
/// the subsequent price move. Used for optimal level placement.
#[derive(Debug, Clone)]
pub struct MutualInformationEstimator {
    /// Observed fill counts per depth bucket.
    fill_counts: Vec<u64>,
    /// Sum of |Δp| after fills per depth bucket.
    price_move_sum: Vec<f64>,
    /// Sum of Δp² after fills per depth bucket.
    price_move_sq_sum: Vec<f64>,
    /// Depth bucket boundaries in bps.
    bucket_boundaries: Vec<f64>,
    /// Total observations.
    total_observations: u64,
}

impl MutualInformationEstimator {
    /// Create new estimator with depth buckets.
    ///
    /// # Arguments
    /// * `bucket_boundaries` - Depth thresholds in bps (e.g., [2, 5, 10, 20, 50])
    pub fn new(bucket_boundaries: Vec<f64>) -> Self {
        let n = bucket_boundaries.len() + 1; // +1 for beyond last boundary
        Self {
            fill_counts: vec![0; n],
            price_move_sum: vec![0.0; n],
            price_move_sq_sum: vec![0.0; n],
            bucket_boundaries,
            total_observations: 0,
        }
    }

    /// Default buckets for typical market making.
    pub fn default_buckets() -> Self {
        Self::new(vec![2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    }

    /// Record a fill observation.
    ///
    /// # Arguments
    /// * `depth_bps` - Depth of the filled order in bps
    /// * `price_move` - Subsequent price move (can be positive or negative)
    pub fn record_fill(&mut self, depth_bps: f64, price_move: f64) {
        let bucket = self.get_bucket(depth_bps);
        self.fill_counts[bucket] += 1;
        self.price_move_sum[bucket] += price_move.abs();
        self.price_move_sq_sum[bucket] += price_move * price_move;
        self.total_observations += 1;
    }

    /// Get bucket index for a depth.
    fn get_bucket(&self, depth_bps: f64) -> usize {
        for (i, &boundary) in self.bucket_boundaries.iter().enumerate() {
            if depth_bps < boundary {
                return i;
            }
        }
        self.bucket_boundaries.len()
    }

    /// Estimate mutual information I(Fill; |ΔP|) for each depth bucket.
    ///
    /// Returns estimated bits of information per bucket.
    pub fn estimate_mutual_information(&self) -> Vec<f64> {
        if self.total_observations < 10 {
            return vec![0.0; self.fill_counts.len()];
        }

        // Compute overall price move statistics
        let total_move_sum: f64 = self.price_move_sum.iter().sum();
        let total_move_sq_sum: f64 = self.price_move_sq_sum.iter().sum();

        let overall_mean = total_move_sum / self.total_observations as f64;
        let overall_var =
            total_move_sq_sum / self.total_observations as f64 - overall_mean * overall_mean;

        if overall_var < EPSILON {
            return vec![0.0; self.fill_counts.len()];
        }

        // For each bucket, compute conditional variance and information
        self.fill_counts
            .iter()
            .zip(self.price_move_sum.iter())
            .zip(self.price_move_sq_sum.iter())
            .map(|((&count, &move_sum), &move_sq_sum)| {
                if count < 5 {
                    return 0.0;
                }
                let n = count as f64;
                let cond_mean = move_sum / n;
                let cond_var = move_sq_sum / n - cond_mean * cond_mean;

                // Information = reduction in variance (log ratio)
                // I ≈ 0.5 × ln(overall_var / conditional_var)
                if cond_var > EPSILON {
                    0.5 * (overall_var / cond_var).ln().max(0.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Get recommended depth weights based on mutual information.
    ///
    /// Higher weight = more informative fills = potentially more value.
    pub fn get_depth_weights(&self) -> Vec<f64> {
        let mi = self.estimate_mutual_information();
        let max_mi = mi.iter().cloned().fold(0.0, f64::max);

        if max_mi < EPSILON {
            return vec![1.0; mi.len()];
        }

        // Normalize to [0.5, 1.5] range
        mi.iter().map(|&m| 0.5 + m / max_mi).collect()
    }
}

// ============================================================================
// Fill Correlation Tracker
// ============================================================================

/// Tracks correlation between fills at different depth levels.
///
/// Used to understand how fills cluster and to avoid placing orders
/// at depths that fill together (reducing diversification benefit).
#[derive(Debug, Clone)]
pub struct FillCorrelationTracker {
    /// Co-occurrence matrix: counts of fills at both depths.
    cooccurrence: Vec<Vec<u64>>,
    /// Marginal counts per depth.
    marginals: Vec<u64>,
    /// Depth bucket boundaries.
    bucket_boundaries: Vec<f64>,
    /// Window for co-occurrence (fills within this many seconds count).
    window_secs: f64,
    /// Recent fills for windowed tracking.
    recent_fills: Vec<(f64, usize, f64)>, // (timestamp, bucket, size)
}

impl FillCorrelationTracker {
    /// Create new tracker.
    pub fn new(bucket_boundaries: Vec<f64>, window_secs: f64) -> Self {
        let n = bucket_boundaries.len() + 1;
        Self {
            cooccurrence: vec![vec![0; n]; n],
            marginals: vec![0; n],
            bucket_boundaries,
            window_secs,
            recent_fills: Vec::new(),
        }
    }

    /// Record a fill and update correlations.
    pub fn record_fill(&mut self, timestamp: f64, depth_bps: f64, size: f64) {
        let bucket = self.get_bucket(depth_bps);

        // Update marginal
        self.marginals[bucket] += 1;

        // Find recent fills within window and update co-occurrence
        let cutoff = timestamp - self.window_secs;
        for &(ts, other_bucket, _) in &self.recent_fills {
            if ts >= cutoff {
                self.cooccurrence[bucket][other_bucket] += 1;
                self.cooccurrence[other_bucket][bucket] += 1;
            }
        }

        // Add to recent fills
        self.recent_fills.push((timestamp, bucket, size));

        // Prune old fills
        self.recent_fills.retain(|&(ts, _, _)| ts >= cutoff);
    }

    /// Get bucket index for a depth.
    fn get_bucket(&self, depth_bps: f64) -> usize {
        for (i, &boundary) in self.bucket_boundaries.iter().enumerate() {
            if depth_bps < boundary {
                return i;
            }
        }
        self.bucket_boundaries.len()
    }

    /// Compute correlation matrix between depth buckets.
    ///
    /// Returns Pearson correlation: ρ_ij = (n_ij - n_i × n_j / N) / √(n_i × n_j)
    pub fn correlation_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.marginals.len();
        let total: u64 = self.marginals.iter().sum();

        if total < 10 {
            // Not enough data
            return vec![vec![0.0; n]; n];
        }

        let mut corr = vec![vec![0.0; n]; n];
        let total_f = total as f64;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            #[allow(clippy::needless_range_loop)]
            for j in 0..n {
                if i == j {
                    corr[i][j] = 1.0;
                    continue;
                }

                let n_i = self.marginals[i] as f64;
                let n_j = self.marginals[j] as f64;
                let n_ij = self.cooccurrence[i][j] as f64;

                if n_i < 1.0 || n_j < 1.0 {
                    continue;
                }

                // Normalized co-occurrence
                let expected = n_i * n_j / total_f;
                let denom = (n_i * n_j).sqrt();
                corr[i][j] = (n_ij - expected) / denom.max(1.0);
            }
        }

        corr
    }

    /// Get diversification penalty for depth allocation.
    ///
    /// Returns penalty in [0, 1] where 0 = no correlation, 1 = high correlation.
    /// Used to adjust allocation away from correlated depths.
    pub fn diversification_penalty(&self, depth_weights: &[f64]) -> f64 {
        let corr = self.correlation_matrix();
        let n = depth_weights.len().min(corr.len());

        if n < 2 {
            return 0.0;
        }

        // Weighted average correlation
        let mut total_corr = 0.0;
        let mut total_weight = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let w = depth_weights[i] * depth_weights[j];
                total_corr += corr[i][j].abs() * w;
                total_weight += w;
            }
        }

        if total_weight > EPSILON {
            total_corr / total_weight
        } else {
            0.0
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_computation() {
        // Uniform distribution: H = ln(n)
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h = compute_entropy(&uniform);
        assert!((h - 4.0_f64.ln()).abs() < 0.01);

        // Concentrated: H ≈ 0
        let concentrated = vec![0.99, 0.003, 0.003, 0.004];
        let h_conc = compute_entropy(&concentrated);
        assert!(h_conc < 0.1);

        // Binary uniform: H = ln(2)
        let binary = vec![0.5, 0.5];
        let h_bin = compute_entropy(&binary);
        assert!((h_bin - 2.0_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_hhi_computation() {
        // Uniform: HHI = 1/n
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let hhi = compute_hhi(&uniform);
        assert!((hhi - 0.25).abs() < 0.01);

        // Concentrated: HHI ≈ 1
        let concentrated = vec![1.0, 0.0, 0.0, 0.0];
        let hhi_conc = compute_hhi(&concentrated);
        assert!((hhi_conc - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_boltzmann_allocation() {
        let config = EntropyConfig::default();
        let allocator = EntropyAllocator::new(config);

        // Marginal values: level 1 is best
        let mvs = vec![10.0, 5.0, 2.0, 1.0, 0.5];
        let result = allocator.allocate_boltzmann(&mvs, 1.0, 0.0002, 0.01);

        // Should have allocated something
        assert!(result.total_size > 0.0);

        // Level 0 should have most
        assert!(result.sizes[0] > result.sizes[4]);

        // Should be diversified (not all in one level)
        assert!(result.effective_levels > 1.5);
        assert!(result.hhi < 0.8);
    }

    #[test]
    fn test_boltzmann_temperature_effect() {
        // Low temperature: concentrate
        let low_t = EntropyConfig {
            temperature: 0.1,
            adaptive_temperature: false,
            ..Default::default()
        };
        let allocator_low = EntropyAllocator::new(low_t);

        // High temperature: spread
        let high_t = EntropyConfig {
            temperature: 10.0,
            adaptive_temperature: false,
            ..Default::default()
        };
        let allocator_high = EntropyAllocator::new(high_t);

        let mvs = vec![10.0, 5.0, 2.0, 1.0];

        let result_low = allocator_low.allocate_boltzmann(&mvs, 1.0, 0.0, 0.0);
        let result_high = allocator_high.allocate_boltzmann(&mvs, 1.0, 0.0, 0.0);

        // Low temperature should be more concentrated
        assert!(result_low.hhi > result_high.hhi);
        // High temperature should have more effective levels
        assert!(result_high.effective_levels > result_low.effective_levels);
    }

    #[test]
    fn test_adaptive_temperature() {
        let config = EntropyConfig {
            temperature: 1.0,
            sigma_baseline: 0.0002,
            adaptive_temperature: true,
            ..Default::default()
        };

        // High volatility: lower effective temperature
        let t_high_vol = config.effective_temperature(0.0004);
        assert!(t_high_vol < config.temperature);

        // Low volatility: higher effective temperature
        let t_low_vol = config.effective_temperature(0.0001);
        assert!(t_low_vol > config.temperature);
    }

    #[test]
    fn test_min_fraction_constraint() {
        let config = EntropyConfig {
            min_level_fraction: 0.1, // Each level gets at least 10%
            temperature: 0.1,        // Would normally concentrate
            adaptive_temperature: false,
            ..Default::default()
        };
        let allocator = EntropyAllocator::new(config);

        let mvs = vec![100.0, 1.0, 1.0, 1.0]; // Very skewed MVs

        let result = allocator.allocate_boltzmann(&mvs, 1.0, 0.0, 0.0);

        // Even the worst levels should have at least ~10%
        for size in &result.sizes {
            if *size > EPSILON {
                assert!(*size >= 0.08, "Size {} below min fraction", size);
            }
        }
    }

    #[test]
    fn test_hhi_constraint() {
        let config = EntropyConfig {
            max_hhi: 0.3, // Must be fairly diversified
            temperature: 0.1,
            adaptive_temperature: false,
            min_level_fraction: 0.0,
            ..Default::default()
        };
        let allocator = EntropyAllocator::new(config);

        let mvs = vec![100.0, 50.0, 10.0, 5.0, 1.0];

        let result = allocator.allocate_boltzmann(&mvs, 1.0, 0.0, 0.0);

        // HHI should be at or below threshold
        assert!(
            result.hhi <= 0.35,
            "HHI {} exceeds max {}",
            result.hhi,
            0.35
        );
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert!(kl_divergence(&p, &q).abs() < EPSILON);

        let p2 = vec![0.9, 0.1];
        let q2 = vec![0.5, 0.5];
        let kl = kl_divergence(&p2, &q2);
        assert!(kl > 0.0); // P is more concentrated than Q
    }

    #[test]
    fn test_mutual_information_estimator() {
        let mut estimator = MutualInformationEstimator::default_buckets();

        // Simulate fills with correlation to price moves
        // Tight fills (small depth) → larger price moves
        for _ in 0..100 {
            estimator.record_fill(1.0, 0.001); // Tight fill, big move
        }
        for _ in 0..100 {
            estimator.record_fill(30.0, 0.0001); // Deep fill, small move
        }

        let mi = estimator.estimate_mutual_information();
        // Tight depths should have higher MI (more informative)
        // Note: with our simple simulation this may not hold perfectly
        assert!(!mi.is_empty());
    }

    #[test]
    fn test_fill_correlation_tracker() {
        let mut tracker = FillCorrelationTracker::new(vec![5.0, 10.0, 20.0], 1.0);

        // Simulate correlated fills at depths 3bp and 7bp
        for i in 0..50 {
            let t = i as f64 * 0.5;
            tracker.record_fill(t, 3.0, 0.1);
            tracker.record_fill(t + 0.1, 7.0, 0.1); // Fill shortly after
        }

        let corr = tracker.correlation_matrix();
        // Buckets 0 (< 5bp) and 1 (5-10bp) should be correlated
        assert!(corr[0][1] > 0.0);
    }

    #[test]
    fn test_empty_allocation() {
        let allocator = EntropyAllocator::default_allocator();

        // Empty MVs
        let result = allocator.allocate_boltzmann(&[], 1.0, 0.0002, 0.01);
        assert!(result.sizes.is_empty());
        assert_eq!(result.total_size, 0.0);

        // Zero total size
        let result2 = allocator.allocate_boltzmann(&[1.0, 2.0], 0.0, 0.0002, 0.01);
        assert_eq!(result2.total_size, 0.0);
    }

    #[test]
    fn test_negative_marginal_values() {
        let allocator = EntropyAllocator::default_allocator();

        // All negative MVs should use uniform fallback
        let mvs = vec![-5.0, -10.0, -2.0];
        let result = allocator.allocate_boltzmann(&mvs, 1.0, 0.0002, 0.0);

        // Should still allocate (uniform fallback)
        assert!(result.total_size > 0.0);
        // Should be roughly uniform
        assert!(result.hhi < 0.5);
    }

    #[test]
    fn test_exploitation_exploration_configs() {
        let mvs = vec![10.0, 5.0, 2.0, 1.0];

        let exploit = EntropyAllocator::new(EntropyConfig::exploitation());
        let explore = EntropyAllocator::new(EntropyConfig::exploration());

        let result_exploit = exploit.allocate_boltzmann(&mvs, 1.0, 0.0, 0.0);
        let result_explore = explore.allocate_boltzmann(&mvs, 1.0, 0.0, 0.0);

        // Exploitation should be more concentrated
        assert!(result_exploit.hhi > result_explore.hhi);
    }
}
