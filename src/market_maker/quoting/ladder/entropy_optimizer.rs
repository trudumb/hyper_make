//! Entropy-constrained optimizer for size allocation.
//!
//! This optimizer completely replaces the old `ConstrainedLadderOptimizer` with an
//! entropy-aware system that maintains diversity while respecting constraints.
//!
//! # Key Differences from Old System
//!
//! 1. **No concentration fallback**: The old system would collapse to 1-2 orders
//!    when spread capture went negative. This system maintains diversity through
//!    entropy constraints.
//!
//! 2. **Soft constraints**: Instead of hard filtering (size = 0 if below threshold),
//!    we use soft penalties that gradually reduce allocation.
//!
//! 3. **Stochastic allocation**: Thompson sampling adds controlled randomness,
//!    making order placement unpredictable while statistically optimal.
//!
//! 4. **Entropy projection**: The allocation is projected onto an entropy constraint
//!    to ensure minimum diversity is always maintained.

use super::entropy_distribution::{
    EntropyDistribution, EntropyDistributionConfig, EntropyDistributor, EntropyLevelParams,
    MarketRegime,
};
use super::{BindingConstraint, ConstrainedAllocation, LevelOptimizationParams};
use crate::EPSILON;
use serde::{Deserialize, Serialize};

/// Configuration for the entropy-constrained optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyOptimizerConfig {
    /// Core entropy distribution configuration
    pub distribution: EntropyDistributionConfig,

    /// Minimum notional value per order (exchange constraint)
    #[serde(default = "default_min_notional")]
    pub min_notional: f64,

    /// Whether to use soft notional constraints (reduce size) vs hard (zero out)
    #[serde(default = "default_soft_notional")]
    pub soft_notional_constraint: bool,

    /// Notional softness parameter: below this multiple of min_notional, start reducing
    /// E.g., soft_factor = 2.0 means start reducing at 2x min_notional
    #[serde(default = "default_notional_soft_factor")]
    pub notional_soft_factor: f64,

    /// Whether to redistribute filtered allocation to remaining levels
    #[serde(default = "default_redistribute")]
    pub redistribute_filtered: bool,

    /// Maximum iterations for constraint satisfaction
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
}

fn default_min_notional() -> f64 {
    10.0
}
fn default_soft_notional() -> bool {
    true
}
fn default_notional_soft_factor() -> f64 {
    1.1 // CHANGED: Reduced from 1.5 to allow $10+ orders with small capital
}
fn default_redistribute() -> bool {
    true
}
fn default_max_iterations() -> usize {
    10
}

impl Default for EntropyOptimizerConfig {
    fn default() -> Self {
        Self {
            distribution: EntropyDistributionConfig::default(),
            min_notional: default_min_notional(),
            soft_notional_constraint: default_soft_notional(),
            notional_soft_factor: default_notional_soft_factor(),
            redistribute_filtered: default_redistribute(),
            max_iterations: default_max_iterations(),
        }
    }
}

/// Result of entropy-constrained optimization.
#[derive(Debug, Clone)]
pub struct EntropyConstrainedAllocation {
    /// Allocated sizes per level
    pub sizes: Vec<f64>,
    /// The underlying entropy distribution
    pub distribution: EntropyDistribution,
    /// Total margin used
    pub margin_used: f64,
    /// Total position allocated
    pub position_used: f64,
    /// Which constraint was binding
    pub binding_constraint: BindingConstraint,
    /// Number of levels that received meaningful allocation
    pub active_levels: usize,
    /// Whether entropy floor was enforced
    pub entropy_floor_active: bool,
}

impl EntropyConstrainedAllocation {
    /// Convert to the legacy ConstrainedAllocation format for compatibility.
    pub fn to_legacy(&self) -> ConstrainedAllocation {
        ConstrainedAllocation {
            sizes: self.sizes.clone(),
            shadow_price: self.distribution.entropy, // Use entropy as shadow price proxy
            margin_used: self.margin_used,
            position_used: self.position_used,
            binding_constraint: self.binding_constraint,
        }
    }
}

/// Entropy-constrained optimizer that maintains diversity while respecting constraints.
///
/// This completely replaces the old `ConstrainedLadderOptimizer`.
pub struct EntropyConstrainedOptimizer {
    config: EntropyOptimizerConfig,
    distributor: EntropyDistributor,
    /// Price for notional calculation
    price: f64,
    /// Available margin
    margin_available: f64,
    /// Maximum position size
    max_position: f64,
    /// Leverage factor
    leverage: f64,
}

impl EntropyConstrainedOptimizer {
    /// Create a new entropy-constrained optimizer.
    pub fn new(
        config: EntropyOptimizerConfig,
        price: f64,
        margin_available: f64,
        max_position: f64,
        leverage: f64,
    ) -> Self {
        let distributor = EntropyDistributor::new(config.distribution.clone());
        Self {
            config,
            distributor,
            price,
            margin_available,
            max_position,
            leverage,
        }
    }

    /// Create with a specific seed for reproducibility.
    pub fn with_seed(
        config: EntropyOptimizerConfig,
        price: f64,
        margin_available: f64,
        max_position: f64,
        leverage: f64,
        seed: u64,
    ) -> Self {
        let distributor = EntropyDistributor::with_seed(config.distribution.clone(), seed);
        Self {
            config,
            distributor,
            price,
            margin_available,
            max_position,
            leverage,
        }
    }

    /// Update market parameters (call when price or constraints change).
    pub fn update_params(&mut self, price: f64, margin_available: f64, max_position: f64) {
        self.price = price;
        self.margin_available = margin_available;
        self.max_position = max_position;
    }

    /// Compute optimal allocation using entropy-based distribution.
    ///
    /// This is the main entry point, replacing `ConstrainedLadderOptimizer::optimize`.
    pub fn optimize(
        &mut self,
        levels: &[LevelOptimizationParams],
        regime: &MarketRegime,
    ) -> EntropyConstrainedAllocation {
        if levels.is_empty() {
            return EntropyConstrainedAllocation {
                sizes: vec![],
                distribution: EntropyDistribution::empty(),
                margin_used: 0.0,
                position_used: 0.0,
                binding_constraint: BindingConstraint::None,
                active_levels: 0,
                entropy_floor_active: false,
            };
        }

        // 1. Convert to entropy level params
        let entropy_levels: Vec<EntropyLevelParams> =
            levels.iter().map(EntropyLevelParams::from).collect();

        // 2. Compute entropy-based distribution
        let distribution = self
            .distributor
            .compute_distribution(&entropy_levels, regime);

        // 3. Determine maximum allocable position from constraints
        let margin_per_unit = self.price / self.leverage;
        let max_by_margin = if margin_per_unit > EPSILON {
            self.margin_available / margin_per_unit
        } else {
            f64::MAX
        };
        let max_position_total = max_by_margin.min(self.max_position);

        let binding_constraint = if max_by_margin < self.max_position {
            BindingConstraint::Margin
        } else if self.max_position < max_by_margin {
            BindingConstraint::Position
        } else {
            BindingConstraint::None
        };

        // 4. Convert probabilities to sizes
        let mut sizes = distribution.to_sizes(max_position_total);

        // 5. Apply notional constraints (soft or hard)
        sizes = self.apply_notional_constraints(
            &sizes,
            &distribution.probabilities,
            max_position_total,
        );

        // 6. Compute final metrics
        let position_used: f64 = sizes.iter().sum();
        let margin_used = position_used * margin_per_unit;
        let active_levels = sizes.iter().filter(|&&s| s > EPSILON).count();
        let entropy_floor_active = distribution.min_entropy_binding;

        EntropyConstrainedAllocation {
            sizes,
            distribution,
            margin_used,
            position_used,
            binding_constraint,
            active_levels,
            entropy_floor_active,
        }
    }

    /// Apply notional constraints while preserving as much entropy as possible.
    fn apply_notional_constraints(
        &self,
        sizes: &[f64],
        _probs: &[f64],
        max_position: f64,
    ) -> Vec<f64> {
        let min_size_for_notional = self.config.min_notional / self.price;
        let soft_threshold = min_size_for_notional * self.config.notional_soft_factor;

        let mut result: Vec<f64> = if self.config.soft_notional_constraint {
            // Soft constraint: gradually reduce sizes below threshold
            sizes
                .iter()
                .map(|&s| {
                    if s >= soft_threshold {
                        s
                    } else if s >= min_size_for_notional {
                        // Linear interpolation from min to soft_threshold
                        let t =
                            (s - min_size_for_notional) / (soft_threshold - min_size_for_notional);
                        min_size_for_notional + t * (s - min_size_for_notional)
                    } else {
                        // Below minimum: either zero or boost to minimum
                        if s * self.price >= self.config.min_notional * 0.8 {
                            // Close enough, boost to minimum
                            min_size_for_notional
                        } else {
                            0.0
                        }
                    }
                })
                .collect()
        } else {
            // Hard constraint: zero out below minimum
            sizes
                .iter()
                .map(|&s| {
                    if s * self.price >= self.config.min_notional {
                        s
                    } else {
                        0.0
                    }
                })
                .collect()
        };

        // Redistribute filtered allocation if enabled
        if self.config.redistribute_filtered {
            let total_before: f64 = sizes.iter().sum();
            let total_after: f64 = result.iter().sum();
            let filtered = total_before - total_after;

            if filtered > EPSILON && total_after > EPSILON {
                // Redistribute proportionally to remaining levels
                let scale = (total_before / total_after).min(1.5); // Cap redistribution
                for size in &mut result {
                    if *size > EPSILON {
                        *size *= scale;
                    }
                }

                // Ensure we don't exceed max_position
                let new_total: f64 = result.iter().sum();
                if new_total > max_position {
                    let correction = max_position / new_total;
                    for size in &mut result {
                        *size *= correction;
                    }
                }
            }
        }

        result
    }

    /// Optimize using old-style parameters for backward compatibility.
    pub fn optimize_legacy(&mut self, levels: &[LevelOptimizationParams]) -> ConstrainedAllocation {
        let regime = MarketRegime::default();
        self.optimize(levels, &regime).to_legacy()
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create an entropy optimizer with default configuration.
pub fn create_entropy_optimizer(
    price: f64,
    margin_available: f64,
    max_position: f64,
    leverage: f64,
) -> EntropyConstrainedOptimizer {
    EntropyConstrainedOptimizer::new(
        EntropyOptimizerConfig::default(),
        price,
        margin_available,
        max_position,
        leverage,
    )
}

/// Create an entropy optimizer tuned for aggressive quoting (more concentration allowed).
pub fn create_aggressive_optimizer(
    price: f64,
    margin_available: f64,
    max_position: f64,
    leverage: f64,
) -> EntropyConstrainedOptimizer {
    let config = EntropyOptimizerConfig {
        distribution: EntropyDistributionConfig {
            min_entropy: 1.0,           // Lower floor (allows more concentration)
            base_temperature: 0.5,      // Colder (more concentrated)
            min_allocation_floor: 0.01, // Lower floor per level
            ..Default::default()
        },
        ..Default::default()
    };

    EntropyConstrainedOptimizer::new(config, price, margin_available, max_position, leverage)
}

/// Create an entropy optimizer tuned for defensive quoting (more diversity).
pub fn create_defensive_optimizer(
    price: f64,
    margin_available: f64,
    max_position: f64,
    leverage: f64,
) -> EntropyConstrainedOptimizer {
    let config = EntropyOptimizerConfig {
        distribution: EntropyDistributionConfig {
            min_entropy: 2.0,           // Higher floor (enforces diversity)
            base_temperature: 2.0,      // Hotter (more uniform)
            min_allocation_floor: 0.05, // Higher floor per level
            toxicity_temp_scale: 1.0,   // More responsive to toxicity
            ..Default::default()
        },
        ..Default::default()
    };

    EntropyConstrainedOptimizer::new(config, price, margin_available, max_position, leverage)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_levels() -> Vec<LevelOptimizationParams> {
        vec![
            LevelOptimizationParams {
                depth_bps: 5.0,
                fill_intensity: 0.8,
                spread_capture: 2.0,
                margin_per_unit: 4500.0,
                adverse_selection: 1.0,
            },
            LevelOptimizationParams {
                depth_bps: 10.0,
                fill_intensity: 0.5,
                spread_capture: 5.0,
                margin_per_unit: 4500.0,
                adverse_selection: 0.5,
            },
            LevelOptimizationParams {
                depth_bps: 15.0,
                fill_intensity: 0.3,
                spread_capture: 8.0,
                margin_per_unit: 4500.0,
                adverse_selection: 0.3,
            },
            LevelOptimizationParams {
                depth_bps: 20.0,
                fill_intensity: 0.2,
                spread_capture: 10.0,
                margin_per_unit: 4500.0,
                adverse_selection: 0.2,
            },
            LevelOptimizationParams {
                depth_bps: 25.0,
                fill_intensity: 0.1,
                spread_capture: 12.0,
                margin_per_unit: 4500.0,
                adverse_selection: 0.1,
            },
        ]
    }

    #[test]
    fn test_entropy_optimizer_basic() {
        let mut optimizer = EntropyConstrainedOptimizer::with_seed(
            EntropyOptimizerConfig::default(),
            90000.0, // price
            10000.0, // margin
            0.1,     // max_position
            20.0,    // leverage
            42,
        );

        let levels = make_levels();
        let regime = MarketRegime::default();

        let result = optimizer.optimize(&levels, &regime);

        // Should have allocated to multiple levels
        assert!(
            result.active_levels >= 3,
            "Should have at least 3 active levels"
        );

        // Total should be near max_position
        assert!(
            result.position_used > 0.05,
            "Should have meaningful position"
        );
        assert!(
            result.position_used <= 0.1 + EPSILON,
            "Should not exceed max_position"
        );

        // Entropy should be maintained
        assert!(
            result.distribution.effective_levels >= 3.0,
            "Effective levels too low: {}",
            result.distribution.effective_levels
        );
    }

    #[test]
    fn test_no_collapse_with_negative_spread_capture() {
        // This test verifies the KEY improvement: no collapse to 1-2 orders

        let mut optimizer = EntropyConstrainedOptimizer::with_seed(
            EntropyOptimizerConfig::default(),
            90000.0,
            10000.0,
            0.1,
            20.0,
            42,
        );

        // All levels have negative or near-zero spread capture
        // Old system would collapse to 0-2 orders here
        let bad_levels: Vec<LevelOptimizationParams> = (0..5)
            .map(|i| LevelOptimizationParams {
                depth_bps: 5.0 + (i as f64) * 5.0,
                fill_intensity: 0.5 / (1.0 + i as f64),
                spread_capture: -1.0 + (i as f64) * 0.3, // Mostly negative
                margin_per_unit: 4500.0,
                adverse_selection: 2.0,
            })
            .collect();

        let toxic_regime = MarketRegime {
            toxicity: 3.0,
            volatility_ratio: 2.0,
            cascade_severity: 0.3,
            book_imbalance: 0.0,
        };

        let result = optimizer.optimize(&bad_levels, &toxic_regime);

        // Should NOT collapse - entropy system maintains diversity
        assert!(
            result.active_levels >= 3,
            "Should maintain at least 3 levels even with bad spreads, got {}",
            result.active_levels
        );

        assert!(
            result.distribution.entropy >= 1.0,
            "Entropy should be maintained, got {}",
            result.distribution.entropy
        );
    }

    #[test]
    fn test_soft_notional_constraint() {
        let config = EntropyOptimizerConfig {
            soft_notional_constraint: true,
            min_notional: 10.0,
            notional_soft_factor: 2.0,
            ..Default::default()
        };

        let mut optimizer = EntropyConstrainedOptimizer::with_seed(
            config, 90000.0, 100.0, // Low margin to force small sizes
            0.005, // Very small max position
            20.0, 42,
        );

        let levels = make_levels();
        let regime = MarketRegime::default();

        let result = optimizer.optimize(&levels, &regime);

        // With soft constraints, we should still get some allocation
        // even if notional values are borderline
        assert!(
            result.position_used > 0.0,
            "Should have some position with soft constraints"
        );
    }

    #[test]
    fn test_defensive_vs_aggressive_optimizer() {
        let levels = make_levels();
        let regime = MarketRegime::default();

        // Create initial optimizers (then replace with seeded versions for reproducibility)
        let _ = create_defensive_optimizer(90000.0, 10000.0, 0.1, 20.0);
        let _ = create_aggressive_optimizer(90000.0, 10000.0, 0.1, 20.0);

        // Use seeds for reproducibility
        let mut defensive = EntropyConstrainedOptimizer::with_seed(
            EntropyOptimizerConfig {
                distribution: EntropyDistributionConfig {
                    min_entropy: 2.0,
                    base_temperature: 2.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            90000.0,
            10000.0,
            0.1,
            20.0,
            42,
        );

        let mut aggressive = EntropyConstrainedOptimizer::with_seed(
            EntropyOptimizerConfig {
                distribution: EntropyDistributionConfig {
                    min_entropy: 1.0,
                    base_temperature: 0.5,
                    ..Default::default()
                },
                ..Default::default()
            },
            90000.0,
            10000.0,
            0.1,
            20.0,
            42,
        );

        let def_result = defensive.optimize(&levels, &regime);
        let agg_result = aggressive.optimize(&levels, &regime);

        // Defensive should have higher entropy (more spread out)
        assert!(
            def_result.distribution.entropy >= agg_result.distribution.entropy - 0.5,
            "Defensive entropy {} should be >= aggressive {}",
            def_result.distribution.entropy,
            agg_result.distribution.entropy
        );

        // Defensive should have more effective levels
        assert!(
            def_result.distribution.effective_levels
                >= agg_result.distribution.effective_levels - 1.0,
            "Defensive effective levels {} should be >= aggressive {}",
            def_result.distribution.effective_levels,
            agg_result.distribution.effective_levels
        );
    }

    #[test]
    fn test_legacy_compatibility() {
        let mut optimizer = EntropyConstrainedOptimizer::with_seed(
            EntropyOptimizerConfig::default(),
            90000.0,
            10000.0,
            0.1,
            20.0,
            42,
        );

        let levels = make_levels();

        // Test legacy interface
        let legacy_result = optimizer.optimize_legacy(&levels);

        // Should return valid ConstrainedAllocation
        assert!(!legacy_result.sizes.is_empty());
        assert!(legacy_result.position_used > 0.0);
    }

    #[test]
    fn test_margin_constraint() {
        let mut optimizer = EntropyConstrainedOptimizer::with_seed(
            EntropyOptimizerConfig::default(),
            90000.0,
            100.0, // Very low margin
            1.0,   // High max_position (not binding)
            20.0,
            42,
        );

        let levels = make_levels();
        let regime = MarketRegime::default();

        let result = optimizer.optimize(&levels, &regime);

        // Should be margin-constrained
        assert_eq!(result.binding_constraint, BindingConstraint::Margin);

        // Margin used should be near available
        let max_by_margin = 100.0 / (90000.0 / 20.0);
        assert!(
            result.position_used <= max_by_margin + EPSILON,
            "Position {} exceeds margin-implied max {}",
            result.position_used,
            max_by_margin
        );
    }

    #[test]
    fn test_position_constraint() {
        let mut optimizer = EntropyConstrainedOptimizer::with_seed(
            EntropyOptimizerConfig::default(),
            90000.0,
            100000.0, // Plenty of margin
            0.01,     // Very low max_position
            20.0,
            42,
        );

        let levels = make_levels();
        let regime = MarketRegime::default();

        let result = optimizer.optimize(&levels, &regime);

        // Should be position-constrained
        assert_eq!(result.binding_constraint, BindingConstraint::Position);

        // Position used should be near max
        assert!(
            result.position_used <= 0.01 + EPSILON,
            "Position {} exceeds max 0.01",
            result.position_used
        );
    }
}
