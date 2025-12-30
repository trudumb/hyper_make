//! Constrained variational ladder optimization.
//!
//! Implements greedy allocation by marginal value subject to margin and position constraints.

use crate::EPSILON;

/// Parameters for a single ladder level used in optimization.
#[derive(Debug, Clone)]
pub struct LevelOptimizationParams {
    /// Depth in basis points
    pub depth_bps: f64,
    /// Fill intensity λ(δ) at this depth
    pub fill_intensity: f64,
    /// Spread capture SC(δ) at this depth
    pub spread_capture: f64,
    /// Margin required per unit size at this level
    pub margin_per_unit: f64,
}

/// Constrained ladder optimizer implementing the variational calculus solution.
///
/// Solves: max Σ λ(δᵢ) × SC(δᵢ) × sᵢ
/// subject to:
///   - Σ margin(sᵢ) ≤ M_available (margin constraint)
///   - sᵢ ≥ min_notional (minimum order size)
///   - Σ sᵢ ≤ max_position (position limit)
///
/// The Lagrangian gives: λ(δ) × SC(δ) = λ* (constant) at optimum.
/// Uses greedy allocation ranked by marginal value.
#[derive(Debug, Clone)]
pub struct ConstrainedLadderOptimizer {
    /// Available margin for placing orders
    pub margin_available: f64,
    /// Maximum position size across all levels
    pub max_position: f64,
    /// Minimum size per level (orders below are skipped)
    pub min_size: f64,
    /// Minimum notional value per order
    pub min_notional: f64,
    /// Price for notional calculation
    pub price: f64,
    /// Leverage factor (margin_per_unit = price / leverage)
    pub leverage: f64,
}

impl ConstrainedLadderOptimizer {
    /// Create new optimizer with constraints.
    pub fn new(
        margin_available: f64,
        max_position: f64,
        min_size: f64,
        min_notional: f64,
        price: f64,
        leverage: f64,
    ) -> Self {
        Self {
            margin_available,
            max_position,
            min_size,
            min_notional,
            price,
            leverage,
        }
    }

    /// Compute optimal size allocation across levels.
    ///
    /// Implements greedy allocation by marginal value:
    /// 1. Compute MV(δ) = λ(δ) × SC(δ) for each level
    /// 2. Sort levels by marginal value (descending)
    /// 3. Allocate capital greedily until constraints bind
    /// 4. Return sizes and shadow price λ* for diagnostics
    pub fn optimize(&self, levels: &[LevelOptimizationParams]) -> ConstrainedAllocation {
        if levels.is_empty() {
            return ConstrainedAllocation {
                sizes: vec![],
                shadow_price: 0.0,
                margin_used: 0.0,
                position_used: 0.0,
                binding_constraint: BindingConstraint::None,
            };
        }

        // 1. Compute marginal value MV(δ) = λ(δ) × SC(δ) for each level
        let marginal_values: Vec<f64> = levels
            .iter()
            .map(|l| (l.fill_intensity * l.spread_capture).max(0.0))
            .collect();

        // 2. Sort levels by marginal value (greedy allocation order)
        let mut sorted_indices: Vec<(usize, f64)> = marginal_values
            .iter()
            .enumerate()
            .map(|(i, &mv)| (i, mv))
            .collect();
        sorted_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Greedy allocation until constraints bind
        let mut sizes = vec![0.0; levels.len()];
        let mut margin_used = 0.0;
        let mut position_used = 0.0;
        let mut shadow_price = 0.0;
        let mut binding_constraint = BindingConstraint::None;

        for &(idx, mv) in &sorted_indices {
            if mv <= EPSILON {
                // No more profitable levels
                break;
            }

            // Margin per unit at this level
            let margin_per_unit = levels[idx].margin_per_unit;

            // How much size can we allocate at this level?
            let max_by_margin = if margin_per_unit > EPSILON {
                (self.margin_available - margin_used) / margin_per_unit
            } else {
                f64::MAX
            };
            let max_by_position = self.max_position - position_used;

            // The size we can allocate (respecting both constraints)
            let max_allocable = max_by_margin.min(max_by_position);

            if max_allocable <= EPSILON {
                // Constraints are binding
                shadow_price = mv;
                binding_constraint = if max_by_margin < max_by_position {
                    BindingConstraint::Margin
                } else {
                    BindingConstraint::Position
                };
                break;
            }

            // Allocate up to max_allocable, respecting min_size and min_notional
            let notional = max_allocable * self.price;
            let size = if max_allocable < self.min_size || notional < self.min_notional {
                0.0 // Skip this level (too small)
            } else {
                max_allocable
            };

            if size > EPSILON {
                sizes[idx] = size;
                margin_used += size * margin_per_unit;
                position_used += size;
            }
        }

        // If we exhausted all levels without binding, shadow price is 0
        if binding_constraint == BindingConstraint::None {
            // Find the lowest MV of allocated levels (marginal value of last allocated)
            shadow_price = sorted_indices
                .iter()
                .filter(|(idx, _)| sizes[*idx] > EPSILON)
                .map(|(_, mv)| *mv)
                .next_back()
                .unwrap_or(0.0);
        }

        ConstrainedAllocation {
            sizes,
            shadow_price,
            margin_used,
            position_used,
            binding_constraint,
        }
    }

    /// Compute level params from depths, intensities, and spread captures.
    pub fn build_level_params(
        &self,
        depths: &[f64],
        intensities: &[f64],
        spreads: &[f64],
    ) -> Vec<LevelOptimizationParams> {
        let margin_per_unit = self.price / self.leverage;

        depths
            .iter()
            .zip(intensities.iter())
            .zip(spreads.iter())
            .map(|((&depth, &intensity), &spread)| LevelOptimizationParams {
                depth_bps: depth,
                fill_intensity: intensity,
                spread_capture: spread,
                margin_per_unit,
            })
            .collect()
    }
}

/// Result of constrained optimization.
#[derive(Debug, Clone)]
pub struct ConstrainedAllocation {
    /// Optimal sizes per level
    pub sizes: Vec<f64>,
    /// Shadow price λ* (marginal value at binding constraint)
    /// Economic interpretation: return per unit of relaxed constraint
    pub shadow_price: f64,
    /// Total margin used
    pub margin_used: f64,
    /// Total position allocated
    pub position_used: f64,
    /// Which constraint is binding
    pub binding_constraint: BindingConstraint,
}

/// Which constraint binds in the optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingConstraint {
    /// No constraint binding (all levels allocated)
    None,
    /// Margin constraint binding
    Margin,
    /// Position constraint binding
    Position,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constrained_optimizer_basic() {
        let optimizer = ConstrainedLadderOptimizer::new(
            1000.0, // margin_available
            0.5,    // max_position
            0.01,   // min_size
            10.0,   // min_notional
            100.0,  // price
            10.0,   // leverage
        );

        let levels = vec![
            LevelOptimizationParams {
                depth_bps: 5.0,
                fill_intensity: 10.0,
                spread_capture: 2.0,
                margin_per_unit: 10.0, // 100 / 10 leverage
            },
            LevelOptimizationParams {
                depth_bps: 10.0,
                fill_intensity: 5.0,
                spread_capture: 5.0,
                margin_per_unit: 10.0,
            },
            LevelOptimizationParams {
                depth_bps: 20.0,
                fill_intensity: 2.0,
                spread_capture: 10.0,
                margin_per_unit: 10.0,
            },
        ];

        let result = optimizer.optimize(&levels);

        // Should have allocated to some levels
        assert!(result.sizes.iter().any(|&s| s > 0.0));
        assert!(result.margin_used > 0.0);
        assert!(result.position_used > 0.0);
    }

    #[test]
    fn test_constrained_optimizer_margin_binding() {
        // Very limited margin
        let optimizer = ConstrainedLadderOptimizer::new(
            50.0,  // margin_available (only $50)
            10.0,  // max_position (plenty of room)
            0.01,  // min_size
            10.0,  // min_notional
            100.0, // price
            10.0,  // leverage (margin_per_unit = 10)
        );

        let levels = vec![
            LevelOptimizationParams {
                depth_bps: 5.0,
                fill_intensity: 10.0,
                spread_capture: 5.0,
                margin_per_unit: 10.0,
            },
            LevelOptimizationParams {
                depth_bps: 10.0,
                fill_intensity: 5.0,
                spread_capture: 10.0,
                margin_per_unit: 10.0,
            },
        ];

        let result = optimizer.optimize(&levels);

        // Should be margin-constrained
        assert!(result.margin_used <= 50.0 + EPSILON);
        // Position should be limited by margin: 50 / 10 = 5 max
        assert!(result.position_used <= 5.0 + EPSILON);
    }

    #[test]
    fn test_constrained_optimizer_position_binding() {
        // Plenty of margin, limited position
        let optimizer = ConstrainedLadderOptimizer::new(
            10000.0, // margin_available (plenty)
            0.1,     // max_position (very limited)
            0.01,    // min_size
            10.0,    // min_notional
            100.0,   // price
            10.0,    // leverage
        );

        let levels = vec![LevelOptimizationParams {
            depth_bps: 5.0,
            fill_intensity: 10.0,
            spread_capture: 5.0,
            margin_per_unit: 10.0,
        }];

        let result = optimizer.optimize(&levels);

        // Should be position-constrained
        assert!(result.position_used <= 0.1 + EPSILON);
        // Should not have used much margin
        assert!(result.margin_used <= 10.0 + EPSILON); // 0.1 * 10 margin_per_unit
    }

    #[test]
    fn test_constrained_optimizer_greedy_ordering() {
        let optimizer = ConstrainedLadderOptimizer::new(
            100.0, // margin_available
            1.0,   // max_position
            0.01,  // min_size
            10.0,  // min_notional
            100.0, // price
            10.0,  // leverage (margin_per_unit = 10)
        );

        // Level 1 has highest marginal value (10 * 10 = 100)
        // Level 2 has lower marginal value (5 * 5 = 25)
        // Level 3 has lowest marginal value (2 * 2 = 4)
        let levels = vec![
            LevelOptimizationParams {
                depth_bps: 20.0,
                fill_intensity: 2.0,
                spread_capture: 2.0,
                margin_per_unit: 10.0,
            },
            LevelOptimizationParams {
                depth_bps: 5.0,
                fill_intensity: 10.0,
                spread_capture: 10.0, // Highest MV
                margin_per_unit: 10.0,
            },
            LevelOptimizationParams {
                depth_bps: 10.0,
                fill_intensity: 5.0,
                spread_capture: 5.0,
                margin_per_unit: 10.0,
            },
        ];

        let result = optimizer.optimize(&levels);

        // Level 1 (index 1) should get allocation first due to highest MV
        // With margin = 100 and margin_per_unit = 10, can allocate up to 10 units total
        // But position limit is 1.0, so only 1 unit total
        assert!(result.sizes[1] > 0.0); // Highest MV level
        assert!((result.position_used - 1.0).abs() < 0.1); // Should hit position limit
    }

    #[test]
    fn test_constrained_optimizer_empty_levels() {
        let optimizer = ConstrainedLadderOptimizer::new(1000.0, 10.0, 0.01, 10.0, 100.0, 10.0);

        let result = optimizer.optimize(&[]);

        assert!(result.sizes.is_empty());
        assert_eq!(result.margin_used, 0.0);
        assert_eq!(result.position_used, 0.0);
        assert_eq!(result.binding_constraint, BindingConstraint::None);
    }

    #[test]
    fn test_constrained_optimizer_negative_spread_capture() {
        let optimizer = ConstrainedLadderOptimizer::new(1000.0, 10.0, 0.01, 10.0, 100.0, 10.0);

        // All levels have negative spread capture (unprofitable)
        let levels = vec![LevelOptimizationParams {
            depth_bps: 2.0,
            fill_intensity: 10.0,
            spread_capture: -5.0, // Negative!
            margin_per_unit: 10.0,
        }];

        let result = optimizer.optimize(&levels);

        // Should not allocate to unprofitable levels
        assert!(result.sizes.iter().all(|&s| s <= EPSILON));
        assert_eq!(result.margin_used, 0.0);
    }

    #[test]
    fn test_constrained_optimizer_build_level_params() {
        let optimizer = ConstrainedLadderOptimizer::new(
            1000.0, 10.0, 0.01, 10.0, 100.0, 5.0, // 5x leverage
        );

        let depths = vec![5.0, 10.0, 20.0];
        let intensities = vec![10.0, 5.0, 2.0];
        let spreads = vec![2.0, 5.0, 10.0];

        let params = optimizer.build_level_params(&depths, &intensities, &spreads);

        assert_eq!(params.len(), 3);
        assert!((params[0].margin_per_unit - 20.0).abs() < 0.01); // 100 / 5
        assert!((params[1].depth_bps - 10.0).abs() < 0.01);
        assert!((params[2].spread_capture - 10.0).abs() < 0.01);
    }
}
