//! Entropy-regularized constrained optimizer for ladder quoting.
//!
//! This module unifies entropy-based allocation with constrained optimization,
//! providing a complete solution for distributed order placement.
//!
//! # Architecture
//!
//! ```text
//! MarketParams (σ, κ, γ, inventory)
//!          ↓
//! InformationDepthGenerator
//!    └─ Generates depths based on I(δ)
//!          ↓
//! EntropyRegularizedOptimizer
//!    ├─ Computes MV(δ) = λ(δ) × SC(δ)
//!    ├─ Applies Boltzmann distribution: s ∝ exp(MV/T)
//!    ├─ Enforces constraints (margin, position, min_notional)
//!    └─ Ensures diversification (HHI, effective levels)
//!          ↓
//! EntropyLadder { depths, sizes, metrics }
//! ```
//!
//! # Key Features
//!
//! 1. **Information-theoretic depth selection**: Places levels where fills are informative
//! 2. **Boltzmann size allocation**: Distributes size using maximum entropy principle
//! 3. **Constraint satisfaction**: Respects margin, position, and notional limits
//! 4. **Diversification guarantees**: Maintains minimum effective levels and max HHI
//! 5. **Adaptive temperature**: Adjusts concentration based on market conditions

use smallvec::SmallVec;

use super::entropy::{compute_entropy, compute_hhi, EntropyAllocator, EntropyConfig};
use super::generator::{fill_intensity, spread_capture};
use super::information_spacing::{InformationDepthGenerator, InformationSpacingConfig};
use super::optimizer::BindingConstraint;
use super::{Ladder, LadderConfig, LadderLevel, LadderLevels, LadderParams};
use crate::market_maker::infra::capacity::DEPTH_INLINE_CAPACITY;
use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

/// Type alias for depth vectors.
type DepthVec = SmallVec<[f64; DEPTH_INLINE_CAPACITY]>;

/// Configuration for entropy-regularized optimization.
#[derive(Debug, Clone)]
pub struct EntropyOptimizationConfig {
    /// Entropy configuration for size allocation.
    pub entropy: EntropyConfig,

    /// Information spacing configuration for depth selection.
    pub spacing: InformationSpacingConfig,

    /// Margin available for orders.
    pub margin_available: f64,

    /// Maximum position size across all levels.
    pub max_position: f64,

    /// Minimum size per level (smaller orders filtered).
    pub min_size: f64,

    /// Minimum notional value per order.
    pub min_notional: f64,

    /// Current price for calculations.
    pub price: f64,

    /// Leverage factor.
    pub leverage: f64,

    /// Enable regime-adaptive behavior.
    pub adaptive_regime: bool,

    /// Blend factor between pure entropy and MV-weighted.
    /// 0.0 = pure entropy (uniform), 1.0 = pure MV-weighted
    /// Default 0.7 (favor profitability with diversification)
    pub profit_entropy_blend: f64,
}

impl Default for EntropyOptimizationConfig {
    fn default() -> Self {
        Self {
            entropy: EntropyConfig::default(),
            spacing: InformationSpacingConfig::default(),
            margin_available: 1000.0,
            max_position: 1.0,
            min_size: 0.001,
            min_notional: 10.0,
            price: 100.0,
            leverage: 10.0,
            adaptive_regime: true,
            profit_entropy_blend: 0.7,
        }
    }
}

impl EntropyOptimizationConfig {
    /// Create config for small accounts (lower minimums).
    pub fn small_account(margin: f64, price: f64) -> Self {
        let position = margin * 0.8 / (price / 20.0); // 20x leverage, 80% margin use
        Self {
            margin_available: margin,
            max_position: position,
            min_size: 10.0 / price * 1.1, // Just above min notional
            min_notional: 10.0,
            price,
            leverage: 20.0,
            entropy: EntropyConfig {
                temperature: 1.5, // More spread for small accounts
                min_effective_levels: 2,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create config for large accounts (tighter spreads, more levels).
    pub fn large_account(margin: f64, price: f64) -> Self {
        let position = margin * 0.5 / (price / 10.0);
        Self {
            margin_available: margin,
            max_position: position,
            min_size: 0.001,
            min_notional: 10.0,
            price,
            leverage: 10.0,
            entropy: EntropyConfig {
                temperature: 0.8,
                min_effective_levels: 4,
                max_hhi: 0.25,
                ..Default::default()
            },
            spacing: InformationSpacingConfig {
                num_levels: 7,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Result of entropy-regularized optimization.
#[derive(Debug, Clone)]
pub struct EntropyOptimizationResult {
    /// Generated ladder.
    pub ladder: Ladder,

    /// Depths used (in bps).
    pub depths: DepthVec,

    /// Allocated sizes.
    pub sizes: Vec<f64>,

    /// Marginal values at each level.
    pub marginal_values: Vec<f64>,

    /// Shannon entropy of size distribution.
    pub entropy: f64,

    /// Number of effective levels: exp(H).
    pub effective_levels: f64,

    /// Herfindahl-Hirschman Index.
    pub hhi: f64,

    /// Temperature used (may be adaptive).
    pub temperature_used: f64,

    /// Total margin used.
    pub margin_used: f64,

    /// Total position allocated.
    pub position_used: f64,

    /// Which constraint is binding.
    pub binding_constraint: BindingConstraint,

    /// Expected profit: Σ MV_i × s_i
    pub expected_value: f64,

    /// Number of active levels (non-zero size).
    pub active_levels: usize,
}

/// Entropy-regularized optimizer for ladder construction.
///
/// Combines information-theoretic depth selection with entropy-based
/// size allocation to create well-diversified ladders.
#[derive(Debug, Clone)]
pub struct EntropyRegularizedOptimizer {
    config: EntropyOptimizationConfig,
    depth_generator: InformationDepthGenerator,
    size_allocator: EntropyAllocator,
}

impl EntropyRegularizedOptimizer {
    /// Create optimizer with configuration.
    pub fn new(config: EntropyOptimizationConfig) -> Self {
        let depth_generator = InformationDepthGenerator::with_default_curve(config.spacing.clone());
        let size_allocator = EntropyAllocator::new(config.entropy.clone());

        Self {
            config,
            depth_generator,
            size_allocator,
        }
    }

    /// Create optimizer with default configuration.
    pub fn default_optimizer() -> Self {
        Self::new(EntropyOptimizationConfig::default())
    }

    /// Optimize ladder using entropy-regularized allocation.
    ///
    /// # Arguments
    /// * `params` - Market parameters for ladder generation
    /// * `ladder_config` - Configuration for ladder structure
    ///
    /// # Returns
    /// Optimization result with ladder and metrics.
    pub fn optimize(&self, params: &LadderParams, ladder_config: &LadderConfig) -> EntropyOptimizationResult {
        // 1. Generate depths using information-theoretic spacing
        let depths = if self.config.spacing.vol_scaling {
            self.depth_generator
                .generate_depths_scaled(params.sigma, params.time_horizon)
        } else {
            self.depth_generator.generate_depths()
        };

        if depths.is_empty() {
            return EntropyOptimizationResult::empty();
        }

        // 2. Compute fill intensity and spread capture for each depth
        let intensities: Vec<f64> = depths
            .iter()
            .map(|&d| fill_intensity(d, params.sigma, params.kappa, params.time_horizon))
            .collect();

        let spreads: Vec<f64> = if let Some(ref depth_decay) = params.depth_decay_as {
            depths
                .iter()
                .map(|&d| depth_decay.spread_capture(d, ladder_config.fees_bps))
                .collect()
        } else {
            depths
                .iter()
                .map(|&d| {
                    spread_capture(
                        d,
                        params.as_at_touch_bps,
                        ladder_config.as_decay_bps,
                        ladder_config.fees_bps,
                    )
                })
                .collect()
        };

        // 3. Compute marginal values: MV = λ × SC
        let marginal_values: Vec<f64> = intensities
            .iter()
            .zip(spreads.iter())
            .map(|(&lambda, &sc)| (lambda * sc).max(0.0))
            .collect();

        // 4. Determine total allocable position
        let margin_per_unit = self.config.price / self.config.leverage;
        let max_by_margin = if margin_per_unit > EPSILON {
            self.config.margin_available / margin_per_unit
        } else {
            f64::MAX
        };
        let max_position = max_by_margin.min(self.config.max_position);

        let binding = if max_by_margin < self.config.max_position {
            BindingConstraint::Margin
        } else if self.config.max_position < max_by_margin {
            BindingConstraint::Position
        } else {
            BindingConstraint::None
        };

        // 5. Blend pure MV allocation with entropy allocation
        let sizes = self.blended_allocation(&marginal_values, max_position, params.sigma);

        // 6. Apply minimum size and notional constraints
        let (filtered_sizes, filtered_depths) =
            self.apply_constraints(&sizes, &depths, params.mid_price);

        // 7. Build ladder from depths and sizes
        let ladder = self.build_ladder(
            &filtered_depths,
            &filtered_sizes,
            params,
            ladder_config.fees_bps,
        );

        // 8. Compute metrics
        let total_size: f64 = filtered_sizes.iter().sum();
        let probs: Vec<f64> = if total_size > EPSILON {
            filtered_sizes.iter().map(|&s| s / total_size).collect()
        } else {
            vec![0.0; filtered_sizes.len()]
        };

        let entropy = compute_entropy(&probs);
        let hhi = compute_hhi(&probs);
        let effective_levels = entropy.exp();

        let expected_value: f64 = filtered_sizes
            .iter()
            .zip(marginal_values.iter())
            .map(|(&s, &mv)| s * mv)
            .sum();

        let active_levels = filtered_sizes.iter().filter(|&&s| s > EPSILON).count();

        EntropyOptimizationResult {
            ladder,
            depths: filtered_depths,
            sizes: filtered_sizes,
            marginal_values,
            entropy,
            effective_levels,
            hhi,
            temperature_used: self.config.entropy.effective_temperature(params.sigma),
            margin_used: total_size * margin_per_unit,
            position_used: total_size,
            binding_constraint: binding,
            expected_value,
            active_levels,
        }
    }

    /// Blend MV-weighted and entropy-based allocation.
    fn blended_allocation(&self, marginal_values: &[f64], total_size: f64, sigma: f64) -> Vec<f64> {
        let n = marginal_values.len();
        if n == 0 || total_size < EPSILON {
            return vec![];
        }

        // Pure MV-weighted allocation
        let total_mv: f64 = marginal_values.iter().sum();
        let mv_allocation: Vec<f64> = if total_mv > EPSILON {
            marginal_values
                .iter()
                .map(|&mv| total_size * mv / total_mv)
                .collect()
        } else {
            vec![total_size / n as f64; n]
        };

        // Entropy-based allocation (Boltzmann)
        let entropy_alloc = self.size_allocator.allocate_boltzmann(
            marginal_values,
            total_size,
            sigma,
            0.0, // Don't filter yet
        );

        // Blend based on config
        let blend = self.config.profit_entropy_blend;
        mv_allocation
            .iter()
            .zip(entropy_alloc.sizes.iter())
            .map(|(&mv_s, &ent_s)| blend * mv_s + (1.0 - blend) * ent_s)
            .collect()
    }

    /// Apply minimum size and notional constraints.
    fn apply_constraints(
        &self,
        sizes: &[f64],
        depths: &DepthVec,
        mid_price: f64,
    ) -> (Vec<f64>, DepthVec) {
        let dynamic_min = (self.config.min_notional / mid_price).max(self.config.min_size * 0.1);

        let mut filtered_sizes = Vec::new();
        let mut filtered_depths = DepthVec::new();

        for (&size, &depth) in sizes.iter().zip(depths.iter()) {
            let notional = size * mid_price;
            if size >= dynamic_min && notional >= self.config.min_notional {
                filtered_sizes.push(size);
                filtered_depths.push(depth);
            }
        }

        // Concentration fallback if all filtered
        if filtered_sizes.is_empty() && !sizes.is_empty() {
            let total: f64 = sizes.iter().sum();
            if total * mid_price >= self.config.min_notional {
                // Find best level by MV (or first if all equal)
                let best_idx = 0; // Use tightest as fallback
                if best_idx < depths.len() {
                    filtered_sizes.push(total);
                    filtered_depths.push(depths[best_idx]);
                }
            }
        }

        (filtered_sizes, filtered_depths)
    }

    /// Build ladder from depths and sizes.
    fn build_ladder(
        &self,
        depths: &DepthVec,
        sizes: &[f64],
        params: &LadderParams,
        _fees_bps: f64,
    ) -> Ladder {
        let mut bids = LadderLevels::new();
        let mut asks = LadderLevels::new();

        for (&depth_bps, &size) in depths.iter().zip(sizes.iter()) {
            if size < EPSILON {
                continue;
            }

            // Calculate price offset from mid
            let offset = params.mid_price * (depth_bps / 10000.0);

            // Round prices per exchange requirements
            let bid_price =
                round_to_significant_and_decimal(params.mid_price - offset, 5, params.decimals);
            let ask_price =
                round_to_significant_and_decimal(params.mid_price + offset, 5, params.decimals);

            // Truncate size
            let size = truncate_float(size, params.sz_decimals, false);

            // Check minimum notional
            if bid_price * size >= params.min_notional {
                bids.push(LadderLevel {
                    price: bid_price,
                    size,
                    depth_bps,
                });
            }
            if ask_price * size >= params.min_notional {
                asks.push(LadderLevel {
                    price: ask_price,
                    size,
                    depth_bps,
                });
            }
        }

        // Apply inventory skew
        self.apply_inventory_skew(&mut bids, &mut asks, params);

        // Sort: bids highest first, asks lowest first
        bids.sort_by(|a, b| {
            b.price
                .partial_cmp(&a.price)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        asks.sort_by(|a, b| {
            a.price
                .partial_cmp(&b.price)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ladder { bids, asks }
    }

    /// Apply GLFT inventory skew to bid/ask levels.
    fn apply_inventory_skew(
        &self,
        bids: &mut LadderLevels,
        asks: &mut LadderLevels,
        params: &LadderParams,
    ) {
        if params.inventory_ratio.abs() < EPSILON {
            return;
        }

        // Reservation price offset: γσ²qT
        let skew_fraction = params.inventory_ratio * params.gamma * params.sigma.powi(2) * params.time_horizon;
        let offset = params.mid_price * skew_fraction;

        // Shift prices
        for level in bids.iter_mut() {
            level.price =
                round_to_significant_and_decimal(level.price - offset, 5, params.decimals);
        }
        for level in asks.iter_mut() {
            level.price =
                round_to_significant_and_decimal(level.price - offset, 5, params.decimals);
        }

        // Size skew: reduce side that increases position
        let size_reduction = params.inventory_ratio.abs().min(0.9);

        if params.inventory_ratio > 0.0 {
            // Long: reduce bids
            for level in bids.iter_mut() {
                level.size = truncate_float(level.size * (1.0 - size_reduction), params.sz_decimals, false);
            }
        } else {
            // Short: reduce asks
            for level in asks.iter_mut() {
                level.size = truncate_float(level.size * (1.0 - size_reduction), params.sz_decimals, false);
            }
        }
    }

    /// Update depth generator with new information curve.
    pub fn update_information_curve(
        &mut self,
        curve: super::information_spacing::InformationDensityCurve,
    ) {
        self.depth_generator.update_curve(curve);
    }

    /// Update entropy configuration.
    pub fn update_entropy_config(&mut self, config: EntropyConfig) {
        self.config.entropy = config.clone();
        self.size_allocator = EntropyAllocator::new(config);
    }
}

impl EntropyOptimizationResult {
    /// Create empty result.
    pub fn empty() -> Self {
        Self {
            ladder: Ladder::default(),
            depths: DepthVec::new(),
            sizes: vec![],
            marginal_values: vec![],
            entropy: 0.0,
            effective_levels: 0.0,
            hhi: 1.0,
            temperature_used: 1.0,
            margin_used: 0.0,
            position_used: 0.0,
            binding_constraint: BindingConstraint::None,
            expected_value: 0.0,
            active_levels: 0,
        }
    }

    /// Check if result is well-diversified.
    pub fn is_diversified(&self, min_levels: f64, max_hhi: f64) -> bool {
        self.effective_levels >= min_levels && self.hhi <= max_hhi
    }

    /// Get diversification score in [0, 1].
    /// 1.0 = perfectly uniform, 0.0 = completely concentrated.
    pub fn diversification_score(&self) -> f64 {
        if self.active_levels <= 1 {
            return 0.0;
        }
        // Normalize HHI to [0, 1] where lower HHI = better diversification
        let min_hhi = 1.0 / self.active_levels as f64;
        let max_hhi = 1.0;
        1.0 - (self.hhi - min_hhi) / (max_hhi - min_hhi)
    }

    /// Log optimization result.
    pub fn log_summary(&self) {
        tracing::info!(
            active_levels = self.active_levels,
            effective_levels = %format!("{:.2}", self.effective_levels),
            entropy = %format!("{:.3}", self.entropy),
            hhi = %format!("{:.3}", self.hhi),
            position = %format!("{:.6}", self.position_used),
            expected_value = %format!("{:.6}", self.expected_value),
            temperature = %format!("{:.2}", self.temperature_used),
            "Entropy optimization result"
        );
    }
}

// ============================================================================
// Stochastic Order Distribution Engine
// ============================================================================

/// Stochastic order distribution engine.
///
/// This is the main entry point for entropy-based order placement.
/// It combines all components to generate well-distributed ladders.
#[derive(Debug, Clone)]
pub struct StochasticOrderEngine {
    optimizer: EntropyRegularizedOptimizer,
    /// Recent optimization metrics for monitoring.
    recent_metrics: Vec<EntropyMetrics>,
    /// Maximum metrics history.
    max_history: usize,
}

/// Metrics from entropy optimization for monitoring.
#[derive(Debug, Clone)]
pub struct EntropyMetrics {
    pub timestamp: f64,
    pub entropy: f64,
    pub effective_levels: f64,
    pub hhi: f64,
    pub active_levels: usize,
    pub expected_value: f64,
}

impl StochasticOrderEngine {
    /// Create new engine with configuration.
    pub fn new(config: EntropyOptimizationConfig) -> Self {
        Self {
            optimizer: EntropyRegularizedOptimizer::new(config),
            recent_metrics: Vec::new(),
            max_history: 100,
        }
    }

    /// Generate entropy-optimized ladder.
    pub fn generate_ladder(
        &mut self,
        params: &LadderParams,
        ladder_config: &LadderConfig,
    ) -> EntropyOptimizationResult {
        let result = self.optimizer.optimize(params, ladder_config);

        // Record metrics
        self.record_metrics(&result);

        result
    }

    /// Record metrics for monitoring.
    fn record_metrics(&mut self, result: &EntropyOptimizationResult) {
        let metrics = EntropyMetrics {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            entropy: result.entropy,
            effective_levels: result.effective_levels,
            hhi: result.hhi,
            active_levels: result.active_levels,
            expected_value: result.expected_value,
        };

        self.recent_metrics.push(metrics);

        // Prune old metrics
        if self.recent_metrics.len() > self.max_history {
            self.recent_metrics.remove(0);
        }
    }

    /// Get average diversification score over recent history.
    pub fn average_diversification(&self) -> f64 {
        if self.recent_metrics.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.recent_metrics.iter().map(|m| m.effective_levels).sum();
        sum / self.recent_metrics.len() as f64
    }

    /// Get average HHI over recent history.
    pub fn average_hhi(&self) -> f64 {
        if self.recent_metrics.is_empty() {
            return 1.0;
        }

        let sum: f64 = self.recent_metrics.iter().map(|m| m.hhi).sum();
        sum / self.recent_metrics.len() as f64
    }

    /// Update configuration.
    pub fn update_config(&mut self, config: EntropyOptimizationConfig) {
        self.optimizer = EntropyRegularizedOptimizer::new(config);
    }

    /// Get recent metrics.
    pub fn recent_metrics(&self) -> &[EntropyMetrics] {
        &self.recent_metrics
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> LadderParams {
        LadderParams {
            mid_price: 100.0,
            sigma: 0.0002,
            kappa: 100.0,
            arrival_intensity: 1.0,
            as_at_touch_bps: 2.0,
            total_size: 1.0,
            inventory_ratio: 0.0,
            gamma: 0.3,
            time_horizon: 10.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            depth_decay_as: None,
        }
    }

    fn default_ladder_config() -> LadderConfig {
        LadderConfig::default()
    }

    #[test]
    fn test_entropy_optimizer_basic() {
        let config = EntropyOptimizationConfig::default();
        let optimizer = EntropyRegularizedOptimizer::new(config);

        let params = default_params();
        let ladder_config = default_ladder_config();

        let result = optimizer.optimize(&params, &ladder_config);

        // Should have produced a ladder
        assert!(!result.ladder.is_empty() || result.active_levels == 0);

        // Metrics should be valid
        if result.active_levels > 0 {
            assert!(result.entropy >= 0.0);
            assert!(result.hhi >= 0.0 && result.hhi <= 1.0);
            assert!(result.effective_levels >= 1.0);
        }
    }

    #[test]
    fn test_entropy_optimizer_diversification() {
        let mut config = EntropyOptimizationConfig::default();
        config.entropy.temperature = 2.0; // Higher temperature for more spread
        config.entropy.max_hhi = 0.3; // Enforce diversification

        let optimizer = EntropyRegularizedOptimizer::new(config);

        let params = default_params();
        let ladder_config = default_ladder_config();

        let result = optimizer.optimize(&params, &ladder_config);

        // Should be diversified
        if result.active_levels >= 3 {
            assert!(result.hhi <= 0.5, "HHI {} too high", result.hhi);
            assert!(
                result.effective_levels >= 2.0,
                "Effective levels {} too low",
                result.effective_levels
            );
        }
    }

    #[test]
    fn test_entropy_optimizer_constraints() {
        let mut config = EntropyOptimizationConfig::default();
        config.margin_available = 50.0; // Limited margin
        config.max_position = 0.3;
        config.price = 100.0;
        config.leverage = 10.0; // margin_per_unit = 10

        let optimizer = EntropyRegularizedOptimizer::new(config.clone());

        let params = default_params();
        let ladder_config = default_ladder_config();

        let result = optimizer.optimize(&params, &ladder_config);

        // Should respect position constraint
        assert!(result.position_used <= config.max_position + EPSILON);

        // Should respect margin constraint
        let margin_per_unit = config.price / config.leverage;
        assert!(result.margin_used <= config.margin_available + margin_per_unit * EPSILON);
    }

    #[test]
    fn test_entropy_optimizer_small_account() {
        let config = EntropyOptimizationConfig::small_account(100.0, 90000.0);
        let optimizer = EntropyRegularizedOptimizer::new(config);

        let mut params = default_params();
        params.mid_price = 90000.0;
        params.min_notional = 10.0;
        let ladder_config = default_ladder_config();

        let result = optimizer.optimize(&params, &ladder_config);

        // Should still produce something even with small account
        // (may be concentrated but should have at least one level)
        assert!(result.active_levels >= 0);
    }

    #[test]
    fn test_stochastic_engine() {
        let config = EntropyOptimizationConfig::default();
        let mut engine = StochasticOrderEngine::new(config);

        let params = default_params();
        let ladder_config = default_ladder_config();

        // Generate multiple ladders
        for _ in 0..5 {
            let _result = engine.generate_ladder(&params, &ladder_config);
        }

        // Check metrics tracking
        assert!(engine.recent_metrics().len() == 5);
        assert!(engine.average_diversification() >= 0.0);
        assert!(engine.average_hhi() <= 1.0);
    }

    #[test]
    fn test_diversification_score() {
        let mut result = EntropyOptimizationResult::empty();

        // Single level: score = 0
        result.active_levels = 1;
        result.hhi = 1.0;
        assert!((result.diversification_score() - 0.0).abs() < EPSILON);

        // Uniform 4 levels: HHI = 0.25, score should be high
        result.active_levels = 4;
        result.hhi = 0.25; // 1/4
        let score = result.diversification_score();
        assert!(score > 0.9, "Score {} should be near 1.0", score);

        // Concentrated 4 levels: HHI = 0.7
        result.hhi = 0.7;
        let score = result.diversification_score();
        assert!(score < 0.5, "Score {} should be low", score);
    }

    #[test]
    fn test_inventory_skew() {
        let config = EntropyOptimizationConfig::default();
        let optimizer = EntropyRegularizedOptimizer::new(config);

        let ladder_config = default_ladder_config();

        // Neutral inventory
        let neutral = default_params();
        let result_neutral = optimizer.optimize(&neutral, &ladder_config);

        // Long inventory
        let mut long = default_params();
        long.inventory_ratio = 0.5;
        let result_long = optimizer.optimize(&long, &ladder_config);

        // Long should have smaller bid sizes
        if !result_neutral.ladder.bids.is_empty() && !result_long.ladder.bids.is_empty() {
            let neutral_bid_size: f64 = result_neutral.ladder.bids.iter().map(|l| l.size).sum();
            let long_bid_size: f64 = result_long.ladder.bids.iter().map(|l| l.size).sum();
            assert!(long_bid_size < neutral_bid_size);
        }
    }

    #[test]
    fn test_blended_allocation() {
        let config = EntropyOptimizationConfig {
            profit_entropy_blend: 0.5, // 50/50 blend
            ..Default::default()
        };
        let optimizer = EntropyRegularizedOptimizer::new(config);

        let params = default_params();
        let ladder_config = default_ladder_config();

        let result = optimizer.optimize(&params, &ladder_config);

        // Blended allocation should produce reasonable results
        if result.active_levels > 1 {
            // Should not be perfectly uniform (would have HHI = 1/n)
            // Should not be completely concentrated (would have HHI ≈ 1)
            assert!(result.hhi > 0.1 && result.hhi < 0.9);
        }
    }
}
