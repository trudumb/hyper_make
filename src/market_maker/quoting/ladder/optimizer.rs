//! Constrained ladder optimization with multiple allocation strategies.
//!
//! Supports two allocation methods:
//! 1. **Proportional MV**: Sizes proportional to marginal value λ(δ) × SC(δ)
//! 2. **Kelly-Stochastic**: Uses first-passage fill probability and Kelly criterion
//!
//! The Kelly-Stochastic approach is more sophisticated and mathematically grounded:
//! - Fill probability: P(δ,τ) = 2Φ(-δ/(σ√τ)) from Brownian first-passage time
//! - Kelly sizing: f*(δ) = E[R|fill] / Var[R|fill]
//! - Properly accounts for the intuition that "closer to mid = higher fill probability"

use crate::EPSILON;

/// Parameters for a single ladder level used in optimization.
#[derive(Debug, Clone)]
pub struct LevelOptimizationParams {
    /// Depth in basis points
    pub depth_bps: f64,
    /// Fill intensity λ(δ) at this depth (used by proportional allocation)
    pub fill_intensity: f64,
    /// Spread capture SC(δ) at this depth
    pub spread_capture: f64,
    /// Margin required per unit size at this level
    pub margin_per_unit: f64,
    /// Adverse selection at this depth (bps)
    pub adverse_selection: f64,
}

/// Parameters for Kelly-Stochastic allocation.
///
/// This is a more sophisticated allocation method based on:
/// - First-passage time theory for fill probability
/// - Kelly criterion for optimal sizing under uncertainty
#[derive(Debug, Clone)]
pub struct KellyStochasticParams {
    /// Volatility per second (e.g., 0.0001 = 1bp/sec)
    pub sigma: f64,
    /// Time horizon in seconds for fill probability calculation
    pub time_horizon: f64,
    /// Informed trader probability at the touch (0.0-1.0)
    pub alpha_touch: f64,
    /// Characteristic depth for alpha decay in bps
    /// α(δ) = α_touch × exp(-δ/alpha_decay_bps)
    pub alpha_decay_bps: f64,
    /// Kelly fraction (0.25 = quarter Kelly, recommended 0.25-0.5)
    /// Lower values are more conservative
    pub kelly_fraction: f64,
}

impl Default for KellyStochasticParams {
    fn default() -> Self {
        Self {
            sigma: 0.0001,         // 1bp/sec volatility (typical for BTC)
            time_horizon: 10.0,    // 10 second holding period
            alpha_touch: 0.15,     // 15% informed probability at touch
            alpha_decay_bps: 10.0, // Alpha decays with 10bp characteristic
            kelly_fraction: 0.25,  // Quarter Kelly (conservative)
        }
    }
}

/// Constrained ladder optimizer implementing proportional allocation.
///
/// Solves: max Σ λ(δᵢ) × SC(δᵢ) × sᵢ
/// subject to:
///   - Σ margin(sᵢ) ≤ M_available (margin constraint)
///   - sᵢ ≥ min_notional (minimum order size)
///   - Σ sᵢ ≤ max_position (position limit)
///
/// Uses proportional allocation: sizes proportional to marginal value MV(δ) = λ(δ) × SC(δ).
/// This distributes capital across all profitable levels for robust ladder quoting.
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
    /// Implements proportional allocation by marginal value:
    /// 1. Compute MV(δ) = λ(δ) × SC(δ) for each level
    /// 2. Allocate sizes proportional to MV, respecting margin/position constraints
    /// 3. Filter out levels below min_size or min_notional thresholds
    /// 4. Return sizes and shadow price λ* for diagnostics
    ///
    /// This distributes capital across ALL profitable levels for robust ladder quoting,
    /// rather than concentrating in a single level.
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

        let total_mv: f64 = marginal_values.iter().sum();
        if total_mv <= EPSILON {
            // No profitable levels
            return ConstrainedAllocation {
                sizes: vec![0.0; levels.len()],
                shadow_price: 0.0,
                margin_used: 0.0,
                position_used: 0.0,
                binding_constraint: BindingConstraint::None,
            };
        }

        // 2. Determine maximum allocable position from constraints
        // Use first level's margin_per_unit (they should all be the same)
        let margin_per_unit = levels[0].margin_per_unit;
        let max_by_margin = if margin_per_unit > EPSILON {
            self.margin_available / margin_per_unit
        } else {
            f64::MAX
        };
        let max_position_total = max_by_margin.min(self.max_position);

        // Determine binding constraint
        let binding_constraint = if max_by_margin < self.max_position {
            BindingConstraint::Margin
        } else if self.max_position < max_by_margin {
            BindingConstraint::Position
        } else {
            BindingConstraint::None
        };

        // 3. Calculate dynamic minimum size from notional requirement
        // This ensures we can quote with any capital level that meets exchange minimums
        let dynamic_min_size = (self.min_notional / self.price).max(self.min_size * 0.1);
        let effective_min_size = dynamic_min_size.max(1e-8); // Absolute floor for precision

        // 4. Allocate sizes proportional to marginal value
        let mut raw_sizes: Vec<f64> = marginal_values
            .iter()
            .map(|&mv| max_position_total * mv / total_mv)
            .collect();

        // 5. Filter out levels below effective min_size or min_notional thresholds
        for size in raw_sizes.iter_mut() {
            let notional = *size * self.price;
            if *size < effective_min_size || notional < self.min_notional {
                *size = 0.0;
            }
        }

        // 6. CONCENTRATION FALLBACK: If all levels filtered out but capacity exists,
        // concentrate into the single best level to guarantee at least one quote
        let valid_count = raw_sizes.iter().filter(|&&s| s > EPSILON).count();
        if valid_count == 0 && max_position_total >= effective_min_size {
            // Find best level by marginal value
            if let Some((best_idx, best_mv)) = marginal_values
                .iter()
                .enumerate()
                .filter(|(_, &mv)| mv > EPSILON)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                // Concentrate entire capacity into best level
                raw_sizes = vec![0.0; levels.len()];
                let concentrated_size = max_position_total.max(effective_min_size);
                let notional = concentrated_size * self.price;
                // Only place if meets notional minimum
                if notional >= self.min_notional {
                    raw_sizes[best_idx] = concentrated_size;
                    tracing::info!(
                        levels = levels.len(),
                        best_level = best_idx,
                        concentrated_size = %format!("{:.6}", concentrated_size),
                        notional = %format!("{:.2}", notional),
                        max_position_total = %format!("{:.6}", max_position_total),
                        best_mv = %format!("{:.6}", best_mv),
                        "Concentration fallback: single quote at best level"
                    );
                }
            }
        }

        // 7. Compute actual margin and position used
        let position_used: f64 = raw_sizes.iter().sum();
        let margin_used = position_used * margin_per_unit;

        // 8. Compute shadow price (marginal value of lowest allocated level)
        let shadow_price = marginal_values
            .iter()
            .zip(raw_sizes.iter())
            .filter(|(_, &sz)| sz > EPSILON)
            .map(|(&mv, _)| mv)
            .fold(f64::MAX, f64::min); // Minimum MV among allocated levels
        let shadow_price = if shadow_price == f64::MAX {
            0.0
        } else {
            shadow_price
        };

        ConstrainedAllocation {
            sizes: raw_sizes,
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
        adverse_selections: &[f64],
    ) -> Vec<LevelOptimizationParams> {
        let margin_per_unit = self.price / self.leverage;

        depths
            .iter()
            .zip(intensities.iter())
            .zip(spreads.iter())
            .zip(adverse_selections.iter())
            .map(
                |(((&depth, &intensity), &spread), &as_bps)| LevelOptimizationParams {
                    depth_bps: depth,
                    fill_intensity: intensity,
                    spread_capture: spread,
                    margin_per_unit,
                    adverse_selection: as_bps,
                },
            )
            .collect()
    }

    /// Kelly-Stochastic allocation using first-passage fill probability and Kelly criterion.
    ///
    /// This is a more sophisticated allocation than proportional MV:
    /// 1. Fill probability from Brownian first-passage time: P(δ,τ) = 2Φ(-δ/(σ√τ))
    /// 2. Kelly sizing: f*(δ) = E[R|fill] / Var[R|fill]
    /// 3. Depth-dependent informed probability: α(δ) = α₀ × exp(-δ/δ_char)
    ///
    /// The result properly captures the intuition that closer to mid = higher fill probability,
    /// and allocates more capital where fills are likely AND profitable.
    pub fn optimize_kelly_stochastic(
        &self,
        levels: &[LevelOptimizationParams],
        kelly_params: &KellyStochasticParams,
    ) -> ConstrainedAllocation {
        if levels.is_empty() {
            return ConstrainedAllocation {
                sizes: vec![],
                shadow_price: 0.0,
                margin_used: 0.0,
                position_used: 0.0,
                binding_constraint: BindingConstraint::None,
            };
        }

        // 1. Compute Kelly-weighted values for each level
        let kelly_values: Vec<f64> = levels
            .iter()
            .map(|level| {
                // Fill probability from first-passage time: P = 2Φ(-δ/(σ√τ))
                let p_fill = fill_probability_stochastic(
                    level.depth_bps,
                    kelly_params.sigma,
                    kelly_params.time_horizon,
                );

                // Informed probability decays with depth: α(δ) = α₀ × exp(-δ/δ_char)
                let alpha = kelly_params.alpha_touch
                    * (-level.depth_bps / kelly_params.alpha_decay_bps).exp();

                // Kelly fraction: f* = E[R|fill] / Var[R|fill]
                let kelly_f =
                    kelly_fraction_at_level(level.spread_capture, level.adverse_selection, alpha);

                // Weight by fill probability and fractional Kelly
                let weighted = p_fill * kelly_f * kelly_params.kelly_fraction;
                weighted.max(0.0)
            })
            .collect();

        let total_kelly: f64 = kelly_values.iter().sum();
        if total_kelly <= EPSILON {
            return ConstrainedAllocation {
                sizes: vec![0.0; levels.len()],
                shadow_price: 0.0,
                margin_used: 0.0,
                position_used: 0.0,
                binding_constraint: BindingConstraint::None,
            };
        }

        // 2. Determine maximum allocable position from constraints
        let margin_per_unit = levels[0].margin_per_unit;
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

        // 3. Calculate dynamic minimum size from notional requirement
        // This ensures we can quote with any capital level that meets exchange minimums
        let dynamic_min_size = (self.min_notional / self.price).max(self.min_size * 0.1);
        let effective_min_size = dynamic_min_size.max(1e-8); // Absolute floor for precision

        // 4. Allocate sizes proportional to Kelly-weighted values
        let mut raw_sizes: Vec<f64> = kelly_values
            .iter()
            .map(|&kv| max_position_total * kv / total_kelly)
            .collect();

        // 5. Filter out levels below effective min_size or min_notional thresholds
        for size in raw_sizes.iter_mut() {
            let notional = *size * self.price;
            if *size < effective_min_size || notional < self.min_notional {
                *size = 0.0;
            }
        }

        // 6. CONCENTRATION FALLBACK: If all levels filtered out but capacity exists,
        // concentrate into the single best level to guarantee at least one quote
        let valid_count = raw_sizes.iter().filter(|&&s| s > EPSILON).count();
        if valid_count == 0 && max_position_total >= effective_min_size {
            // Find best level by Kelly value
            if let Some((best_idx, best_kv)) = kelly_values
                .iter()
                .enumerate()
                .filter(|(_, &kv)| kv > EPSILON)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                // Concentrate entire capacity into best level
                raw_sizes = vec![0.0; levels.len()];
                let concentrated_size = max_position_total.max(effective_min_size);
                let notional = concentrated_size * self.price;
                // Only place if meets notional minimum
                if notional >= self.min_notional {
                    raw_sizes[best_idx] = concentrated_size;
                    tracing::info!(
                        levels = levels.len(),
                        best_level = best_idx,
                        concentrated_size = %format!("{:.6}", concentrated_size),
                        notional = %format!("{:.2}", notional),
                        max_position_total = %format!("{:.6}", max_position_total),
                        best_kelly = %format!("{:.6}", best_kv),
                        "Concentration fallback (Kelly): single quote at best level"
                    );
                }
            }
        }

        // 7. Compute actual margin and position used
        let position_used: f64 = raw_sizes.iter().sum();
        let margin_used = position_used * margin_per_unit;

        // 8. Shadow price is the minimum Kelly value among allocated levels
        let shadow_price = kelly_values
            .iter()
            .zip(raw_sizes.iter())
            .filter(|(_, &sz)| sz > EPSILON)
            .map(|(&kv, _)| kv)
            .fold(f64::MAX, f64::min);
        let shadow_price = if shadow_price == f64::MAX {
            0.0
        } else {
            shadow_price
        };

        ConstrainedAllocation {
            sizes: raw_sizes,
            shadow_price,
            margin_used,
            position_used,
            binding_constraint,
        }
    }
}

// ============================================================================
// Stochastic Fill Probability Functions
// ============================================================================

/// First-passage time fill probability for Brownian motion.
///
/// For a Brownian motion with volatility σ, the probability of reaching
/// depth δ before time τ is: P(τ_δ < τ) = 2Φ(-δ/(σ√τ))
///
/// This properly captures the intuition that:
/// - Closer to mid = higher fill probability
/// - Higher volatility = higher fill probability at all depths
/// - Longer time horizon = higher fill probability
pub(crate) fn fill_probability_stochastic(depth_bps: f64, sigma: f64, time_horizon: f64) -> f64 {
    let depth = depth_bps / 10000.0; // Convert bps to fraction
    let sigma_sqrt_t = sigma * time_horizon.sqrt();

    if sigma_sqrt_t < 1e-12 || depth < EPSILON {
        return if depth < EPSILON { 1.0 } else { 0.0 };
    }

    // P(τ_δ < τ) = 2Φ(-δ/(σ√τ))
    2.0 * normal_cdf(-depth / sigma_sqrt_t)
}

/// Standard normal CDF approximation.
///
/// Uses Abramowitz and Stegun approximation (formula 7.1.26) with error < 1.5e-7.
fn normal_cdf(x: f64) -> f64 {
    // Handle extreme values
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    // Abramowitz and Stegun approximation for erf
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs() / std::f64::consts::SQRT_2;

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();

    0.5 * (1.0 + sign * y)
}

/// Kelly fraction at a given level.
///
/// f*(δ) = E[R|fill] / Var[R|fill]
///
/// Where:
/// - E[R|fill] = (1-α) × SC - α × AS
/// - Var[R|fill] = (SC + AS)² × α × (1-α)
///
/// This tells us the optimal fraction of capital to allocate at this level.
fn kelly_fraction_at_level(spread_capture: f64, adverse_selection: f64, alpha: f64) -> f64 {
    // Expected return given fill:
    // If noise trader: profit = spread_capture
    // If informed trader: loss = adverse_selection
    // E[R] = (1-α) × SC - α × AS
    let expected = (1.0 - alpha) * spread_capture - alpha * adverse_selection;

    // Variance of return given fill:
    // Two outcomes with probabilities (1-α) and α
    // Var = (SC + AS)² × α × (1-α)
    let total_range = spread_capture + adverse_selection;
    let variance = total_range.powi(2) * alpha * (1.0 - alpha);

    // Kelly: f* = E[R] / Var[R]
    if variance < 1e-10 || expected <= 0.0 {
        return 0.0;
    }

    (expected / variance).max(0.0)
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

    /// Helper to create level params with default adverse_selection
    fn level(depth: f64, fill_int: f64, spread_cap: f64, margin: f64) -> LevelOptimizationParams {
        LevelOptimizationParams {
            depth_bps: depth,
            fill_intensity: fill_int,
            spread_capture: spread_cap,
            margin_per_unit: margin,
            adverse_selection: 1.0, // Default AS of 1bp
        }
    }

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
            level(5.0, 10.0, 2.0, 10.0),
            level(10.0, 5.0, 5.0, 10.0),
            level(20.0, 2.0, 10.0, 10.0),
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

        let levels = vec![level(5.0, 10.0, 5.0, 10.0), level(10.0, 5.0, 10.0, 10.0)];

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

        let levels = vec![level(5.0, 10.0, 5.0, 10.0)];

        let result = optimizer.optimize(&levels);

        // Should be position-constrained
        assert!(result.position_used <= 0.1 + EPSILON);
        // Should not have used much margin
        assert!(result.margin_used <= 10.0 + EPSILON); // 0.1 * 10 margin_per_unit
    }

    #[test]
    fn test_constrained_optimizer_proportional_allocation() {
        let optimizer = ConstrainedLadderOptimizer::new(
            100.0, // margin_available
            1.0,   // max_position
            0.01,  // min_size
            1.0,   // min_notional (lowered to allow small allocations)
            100.0, // price
            10.0,  // leverage (margin_per_unit = 10)
        );

        // Level 0: MV = 5 * 5 = 25
        // Level 1: MV = 10 * 10 = 100 (highest)
        // Level 2: MV = 5 * 5 = 25
        // Total MV = 150, proportions: 16.7%, 66.7%, 16.7%
        let levels = vec![
            level(20.0, 5.0, 5.0, 10.0),
            level(5.0, 10.0, 10.0, 10.0), // Highest MV
            level(10.0, 5.0, 5.0, 10.0),
        ];

        let result = optimizer.optimize(&levels);

        // With proportional allocation, ALL levels should get some size
        // Size proportional to MV: level[1] should get most (100/150 ≈ 67%)
        assert!(result.sizes[0] > 0.0, "Level 0 should have allocation");
        assert!(result.sizes[1] > 0.0, "Level 1 should have allocation");
        assert!(result.sizes[2] > 0.0, "Level 2 should have allocation");

        // Level 1 should have the most (highest MV)
        assert!(result.sizes[1] > result.sizes[0]);
        assert!(result.sizes[1] > result.sizes[2]);

        // Total position should be at the limit (1.0) since margin allows it
        // margin_available/margin_per_unit = 100/10 = 10, but max_position = 1.0
        assert!((result.position_used - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_constrained_optimizer_distributes_across_levels() {
        // Key test: verify we get MULTIPLE levels allocated, not just one
        let optimizer = ConstrainedLadderOptimizer::new(
            1000.0, // margin_available
            0.5,    // max_position
            0.01,   // min_size
            1.0,    // min_notional (low to allow small sizes)
            100.0,  // price
            10.0,   // leverage
        );

        let levels = vec![
            level(5.0, 10.0, 5.0, 10.0),
            level(10.0, 5.0, 8.0, 10.0),
            level(20.0, 3.0, 10.0, 10.0),
        ];

        let result = optimizer.optimize(&levels);

        // Count non-zero allocations
        let allocated_levels = result.sizes.iter().filter(|&&s| s > EPSILON).count();

        // Should have allocated to ALL 3 levels (the key fix!)
        assert_eq!(
            allocated_levels, 3,
            "Expected 3 levels allocated, got {}. Sizes: {:?}",
            allocated_levels, result.sizes
        );
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
            adverse_selection: 1.0,
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
        let adverse = vec![1.0, 0.5, 0.2];

        let params = optimizer.build_level_params(&depths, &intensities, &spreads, &adverse);

        assert_eq!(params.len(), 3);
        assert!((params[0].margin_per_unit - 20.0).abs() < 0.01); // 100 / 5
        assert!((params[1].depth_bps - 10.0).abs() < 0.01);
        assert!((params[2].spread_capture - 10.0).abs() < 0.01);
        assert!((params[2].adverse_selection - 0.2).abs() < 0.01);
    }

    // ==================== Kelly-Stochastic Tests ====================

    #[test]
    fn test_fill_probability_stochastic() {
        // At touch (0 depth), fill probability should be 1
        let p_touch = fill_probability_stochastic(0.0, 0.0001, 10.0);
        assert!((p_touch - 1.0).abs() < 0.01);

        // Tight depth (2 bps) with typical volatility
        let p_tight = fill_probability_stochastic(2.0, 0.0001, 10.0);
        assert!(p_tight > 0.1); // Should have decent probability

        // Wide depth (20 bps) should have very low probability
        let p_wide = fill_probability_stochastic(20.0, 0.0001, 10.0);
        assert!(p_wide < 0.01); // Should be near zero

        // Higher volatility increases fill probability
        let p_high_vol = fill_probability_stochastic(10.0, 0.001, 10.0);
        let p_low_vol = fill_probability_stochastic(10.0, 0.0001, 10.0);
        assert!(p_high_vol > p_low_vol);
    }

    #[test]
    fn test_kelly_fraction_at_level() {
        // With zero alpha (no informed traders), variance = 0, so Kelly returns 0
        // This is a degenerate case - in practice alpha should never be exactly 0
        let kelly_no_informed = kelly_fraction_at_level(5.0, 1.0, 0.0);
        assert!(kelly_no_informed == 0.0); // Variance is 0 when alpha = 0

        // With very small alpha (1%), should be positive (realistic case)
        let kelly_tiny_alpha = kelly_fraction_at_level(5.0, 1.0, 0.01);
        assert!(kelly_tiny_alpha > 0.0);

        // With 100% alpha (all informed), expected return is negative
        let kelly_all_informed = kelly_fraction_at_level(5.0, 10.0, 1.0);
        assert!(kelly_all_informed <= 0.0);

        // With moderate alpha and good spread capture, should be positive
        let kelly_moderate = kelly_fraction_at_level(10.0, 2.0, 0.15);
        assert!(kelly_moderate > 0.0);

        // Negative spread capture should give zero
        let kelly_unprofitable = kelly_fraction_at_level(-5.0, 1.0, 0.1);
        assert!(kelly_unprofitable <= 0.0);
    }

    #[test]
    fn test_kelly_stochastic_concentrates_at_touch() {
        let optimizer = ConstrainedLadderOptimizer::new(1000.0, 0.5, 0.001, 1.0, 100.0, 10.0);

        // Create levels at increasing depths
        let levels = vec![
            LevelOptimizationParams {
                depth_bps: 2.0,
                fill_intensity: 0.0, // Ignored by Kelly-Stochastic
                spread_capture: 2.0,
                margin_per_unit: 10.0,
                adverse_selection: 1.0,
            },
            LevelOptimizationParams {
                depth_bps: 10.0,
                fill_intensity: 0.0,
                spread_capture: 8.0,
                margin_per_unit: 10.0,
                adverse_selection: 0.5,
            },
            LevelOptimizationParams {
                depth_bps: 20.0,
                fill_intensity: 0.0,
                spread_capture: 15.0,
                margin_per_unit: 10.0,
                adverse_selection: 0.2,
            },
        ];

        let kelly_params = KellyStochasticParams::default();
        let result = optimizer.optimize_kelly_stochastic(&levels, &kelly_params);

        // Tightest level should get most size (highest fill probability)
        // Even though spread capture is lower
        assert!(
            result.sizes[0] > result.sizes[2],
            "Tight level should get more than deep level. Sizes: {:?}",
            result.sizes
        );
    }

    #[test]
    fn test_normal_cdf_accuracy() {
        // Test known values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((normal_cdf(1.0) - 0.8413).abs() < 0.01);
        assert!((normal_cdf(-1.0) - 0.1587).abs() < 0.01);
        assert!((normal_cdf(2.0) - 0.9772).abs() < 0.01);
        assert!((normal_cdf(-2.0) - 0.0228).abs() < 0.01);

        // Extreme values
        assert!(normal_cdf(10.0) > 0.999);
        assert!(normal_cdf(-10.0) < 0.001);
    }

    #[test]
    fn test_concentration_fallback_small_capacity() {
        // Scenario: Extremely small position capacity where even with dynamic min_size,
        // proportional allocation still produces sub-minimum sizes.
        // The concentration fallback should put entire capacity in single best level.
        //
        // With dynamic min_size = max($10/$90000, 0.001*0.1) = max(0.000111, 0.0001) = 0.000111
        // If capacity = 0.0003 and 5 levels, each gets 0.00006 < 0.000111 → all filtered → concentrate
        let optimizer = ConstrainedLadderOptimizer::new(
            50.0,     // margin_available ($50)
            0.0003,   // max_position (extremely small - less than min_notional worth)
            0.001,    // min_size (hardcoded, but dynamic calculation should override)
            10.0,     // min_notional ($10 exchange minimum)
            90000.0,  // price ($90k BTC)
            20.0,     // leverage
        );

        // Create 5 levels with varying marginal values
        let levels = vec![
            LevelOptimizationParams {
                depth_bps: 5.0,
                fill_intensity: 0.8,
                spread_capture: 0.0005,
                margin_per_unit: 90000.0 / 20.0,
                adverse_selection: 0.0001,
            },
            LevelOptimizationParams {
                depth_bps: 10.0,
                fill_intensity: 0.6,
                spread_capture: 0.001,
                margin_per_unit: 90000.0 / 20.0,
                adverse_selection: 0.0001,
            },
            LevelOptimizationParams {
                depth_bps: 15.0,
                fill_intensity: 0.4,
                spread_capture: 0.0015,
                margin_per_unit: 90000.0 / 20.0,
                adverse_selection: 0.0001,
            },
            LevelOptimizationParams {
                depth_bps: 20.0,
                fill_intensity: 0.3,
                spread_capture: 0.002,
                margin_per_unit: 90000.0 / 20.0,
                adverse_selection: 0.0001,
            },
            LevelOptimizationParams {
                depth_bps: 25.0,
                fill_intensity: 0.2,
                spread_capture: 0.0025,
                margin_per_unit: 90000.0 / 20.0,
                adverse_selection: 0.0001,
            },
        ];

        let allocation = optimizer.optimize(&levels);

        // With 0.0003 BTC capacity split 5 ways = 0.00006 each < 0.000111 dynamic min
        // But concentration gives 0.0003 BTC which is 0.0003 * 90000 = $27 > $10 min_notional
        // So we should get exactly 1 non-zero level
        let non_zero_count = allocation.sizes.iter().filter(|&&s| s > 0.0001).count();
        assert_eq!(non_zero_count, 1, "Expected concentration into single level");

        // The single level should have ~full capacity
        let total_size: f64 = allocation.sizes.iter().sum();
        assert!(
            total_size >= 0.0002,
            "Expected concentrated size near capacity, got {}",
            total_size
        );

        // Notional should meet minimum ($27 > $10)
        let max_size = allocation.sizes.iter().cloned().fold(0.0, f64::max);
        let notional = max_size * 90000.0;
        assert!(
            notional >= 10.0,
            "Expected notional >= $10, got {}",
            notional
        );
    }

    #[test]
    fn test_dynamic_min_size_allows_small_orders() {
        // Test that with dynamic min_size, small accounts CAN quote across multiple levels
        // Previously: 0.002633 BTC / 5 levels = 0.000527 < 0.001 hardcoded → all filtered
        // Now: dynamic min = $10/$90k = 0.000111, so 0.000527 > 0.000111 → passes
        let optimizer = ConstrainedLadderOptimizer::new(
            250.0,    // margin_available ($250)
            0.002633, // max_position (same as log analysis)
            0.001,    // min_size (hardcoded, would filter if used)
            10.0,     // min_notional ($10 exchange minimum)
            90000.0,  // price ($90k BTC)
            20.0,     // leverage
        );

        let levels = vec![
            LevelOptimizationParams {
                depth_bps: 5.0,
                fill_intensity: 0.8,
                spread_capture: 0.0005,
                margin_per_unit: 90000.0 / 20.0,
                adverse_selection: 0.0001,
            },
            LevelOptimizationParams {
                depth_bps: 10.0,
                fill_intensity: 0.6,
                spread_capture: 0.001,
                margin_per_unit: 90000.0 / 20.0,
                adverse_selection: 0.0001,
            },
            LevelOptimizationParams {
                depth_bps: 15.0,
                fill_intensity: 0.4,
                spread_capture: 0.0015,
                margin_per_unit: 90000.0 / 20.0,
                adverse_selection: 0.0001,
            },
        ];

        let allocation = optimizer.optimize(&levels);

        // All 3 levels should now pass with dynamic min_size
        let non_zero_count = allocation.sizes.iter().filter(|&&s| s > 0.0001).count();
        assert_eq!(
            non_zero_count, 3,
            "Expected all 3 levels to pass with dynamic min_size"
        );

        // Total should be ~full capacity
        let total_size: f64 = allocation.sizes.iter().sum();
        assert!(
            (total_size - 0.002633).abs() < 0.0001,
            "Expected total near capacity, got {}",
            total_size
        );
    }

    #[test]
    fn test_dynamic_min_size_from_notional() {
        // Test that dynamic min_size is calculated from min_notional/price
        // rather than using hardcoded min_size when it's too restrictive
        let optimizer = ConstrainedLadderOptimizer::new(
            1000.0,    // margin_available
            0.005,     // max_position
            0.001,     // min_size (hardcoded 0.001 BTC = $90 at $90k)
            10.0,      // min_notional ($10 = 0.000111 BTC at $90k)
            90000.0,   // price
            20.0,      // leverage
        );

        // Single level that would be filtered by 0.001 min_size
        // but should pass dynamic min_size from notional
        let levels = vec![LevelOptimizationParams {
            depth_bps: 10.0,
            fill_intensity: 0.5,
            spread_capture: 0.001,
            margin_per_unit: 90000.0 / 20.0,
            adverse_selection: 0.0001,
        }];

        let allocation = optimizer.optimize(&levels);

        // With 0.005 BTC capacity and single level, should get full allocation
        // not filtered out by hardcoded 0.001 min_size
        assert!(
            allocation.sizes[0] > 0.0001,
            "Expected allocation, got {}",
            allocation.sizes[0]
        );
    }
}
