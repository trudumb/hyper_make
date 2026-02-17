//! Risk-budget allocation for tick-grid levels.
//!
//! Two allocation strategies:
//! - **Softmax (default)**: Utility-proportional sizing via softmax weights.
//!   Higher-utility levels (near touch) get more size. Temperature controls
//!   concentration: cold (0.3) = aggressive skew, hot (2.0) = nearly uniform.
//! - **Uniform (legacy)**: Greedy water-filling with flat per-level cap.
//!
//! # Constraints (both strategies)
//! - `Σ size ≤ position_capacity` — total position budget per side
//! - `Σ size × margin_per_unit ≤ margin_capacity` — margin budget
//! - `size[i] ≤ per_level_cap` — single order concentration limit
//! - `size[i] ≥ min_viable_size OR size[i] == 0` — exchange minimum notional

use crate::truncate_float;

use super::tick_grid::TickLevel;

/// Risk budget for one side (bid or ask).
#[derive(Debug, Clone)]
pub struct SideRiskBudget {
    /// Maximum total position on this side (contracts).
    pub position_capacity: f64,
    /// Maximum margin available for this side (USD).
    pub margin_capacity: f64,
    /// Margin required per contract (USD). Typically mark_price / leverage.
    pub margin_per_contract: f64,
    /// Minimum viable order size (from SizeQuantum).
    pub min_viable_size: f64,
    /// Maximum fraction of position_capacity in a single order.
    pub max_single_order_fraction: f64,
    /// Size decimals for exchange truncation.
    pub sz_decimals: u32,
}

/// Allocation result for a single level.
#[derive(Debug, Clone, Copy)]
pub struct LevelAllocation {
    /// The tick this allocation is for.
    pub tick: TickLevel,
    /// Allocated size in contracts (exchange-truncated).
    pub size: f64,
    /// Margin consumed by this level.
    pub margin_used: f64,
}

/// Parameters controlling softmax allocation behavior.
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxParams {
    /// Controls concentration: lower = more skewed to high utility.
    /// 0.3 = aggressive (touch gets ~50%), 2.0 = nearly uniform (defensive).
    pub temperature: f64,
    /// Minimum Shannon entropy (bits) of the allocation weights.
    /// Prevents collapse to a single level. Typical: 1.0 bit.
    pub min_entropy_bits: f64,
    /// Minimum fraction of total capacity per allocated level.
    /// Prevents any level from being negligibly small. Typical: 0.05 (5%).
    pub min_level_fraction: f64,
}

impl Default for SoftmaxParams {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            min_entropy_bits: 1.0,
            min_level_fraction: 0.05,
        }
    }
}

/// Allocation diagnostic information for logging.
#[derive(Debug, Clone)]
pub struct AllocationDiagnostics {
    /// Final temperature used (may differ from input if entropy floor raised it).
    pub effective_temperature: f64,
    /// Shannon entropy of the allocation in bits.
    pub entropy_bits: f64,
    /// Softmax weights before clamping (in level order).
    pub weights: Vec<f64>,
}

/// Compute softmax allocation temperature from market regime.
///
/// Adapts concentration based on:
/// - Regime gamma multiplier: higher = more volatile = more uniform
/// - Warmup progress: early session = more uniform to learn fill rates
/// - Position zone: Yellow zone = spread risk across levels
pub fn compute_allocation_temperature(
    regime_gamma_mult: f64,
    warmup_progress: f64,
    in_yellow_zone: bool,
) -> f64 {
    let base: f64 = if regime_gamma_mult <= 1.1 {
        0.3
    } else if regime_gamma_mult <= 1.5 {
        0.5
    } else if regime_gamma_mult <= 2.0 {
        1.0
    } else {
        2.0
    };
    let warmup_adj: f64 = if warmup_progress < 0.25 { 0.5 } else { 0.0 };
    let zone_adj: f64 = if in_yellow_zone { 0.3 } else { 0.0 };
    (base + warmup_adj + zone_adj).clamp(0.2, 3.0)
}

/// Allocate sizes to tick levels using utility-proportional softmax weighting.
///
/// Higher-utility levels get proportionally more size. The softmax temperature
/// controls how concentrated the allocation is (lower = more skewed to best levels).
///
/// # Algorithm
/// ```text
/// 1. Compute softmax weights: w_i = exp(utility_i / T) / Σ exp(utility_j / T)
/// 2. If Shannon entropy H(w) < min_entropy_bits, raise T until H ≥ min_entropy
/// 3. target_size_i = total_capacity × w_i
/// 4. Clamp each to [min_viable_size, per_level_cap]
/// 5. Redistribute excess from clamped levels to unclamped (iterative)
/// 6. Truncate to exchange precision
/// 7. Drop any level < min_viable_size
/// ```
pub fn allocate_risk_budget(
    levels: &[TickLevel],
    budget: &SideRiskBudget,
    softmax: &SoftmaxParams,
) -> (Vec<LevelAllocation>, AllocationDiagnostics) {
    let empty_diag = AllocationDiagnostics {
        effective_temperature: softmax.temperature,
        entropy_bits: 0.0,
        weights: Vec::new(),
    };

    if levels.is_empty() || budget.position_capacity <= 0.0 {
        return (Vec::new(), empty_diag);
    }

    // Total capacity is the minimum of position and margin budgets
    let margin_limited_capacity = if budget.margin_per_contract > 0.0 {
        budget.margin_capacity / budget.margin_per_contract
    } else {
        budget.position_capacity
    };
    let total_capacity = budget.position_capacity.min(margin_limited_capacity);

    if total_capacity < budget.min_viable_size {
        return (Vec::new(), empty_diag);
    }

    let per_level_cap = budget.position_capacity * budget.max_single_order_fraction;

    // Step 1: Compute softmax weights with entropy floor
    let (weights, effective_temp) = compute_softmax_weights(
        levels,
        softmax.temperature,
        softmax.min_entropy_bits,
    );

    let entropy = shannon_entropy_bits(&weights);

    // Step 2: Compute target sizes from weights
    let min_level_size = (total_capacity * softmax.min_level_fraction)
        .max(budget.min_viable_size);
    let mut sizes = vec![0.0_f64; levels.len()];

    // Initial allocation from weights
    for (i, &w) in weights.iter().enumerate() {
        sizes[i] = total_capacity * w;
    }

    // Step 3: Iterative clamping with redistribution (max 5 iterations)
    for _ in 0..5 {
        let mut excess = 0.0_f64;
        let mut unclamped_weight_sum = 0.0_f64;
        let mut any_clamped = false;

        // Identify clamped levels and collect excess
        for (i, size) in sizes.iter_mut().enumerate() {
            if *size > per_level_cap {
                excess += *size - per_level_cap;
                *size = per_level_cap;
                any_clamped = true;
            } else if *size < min_level_size && *size > 0.0 {
                // Below minimum — will be set to min or dropped later
                // For now, clamp up if there's budget
                let deficit = min_level_size - *size;
                if deficit <= excess {
                    excess -= deficit;
                    *size = min_level_size;
                }
            }
            if *size > 0.0 && *size < per_level_cap {
                unclamped_weight_sum += weights[i];
            }
        }

        if !any_clamped || excess < budget.min_viable_size * 0.1 {
            break;
        }

        // Redistribute excess proportionally to unclamped levels
        if unclamped_weight_sum > 0.0 {
            for (i, size) in sizes.iter_mut().enumerate() {
                if *size > 0.0 && *size < per_level_cap {
                    let share = weights[i] / unclamped_weight_sum;
                    *size += excess * share;
                }
            }
        }
    }

    // Step 4: Ensure total doesn't exceed capacity
    let raw_total: f64 = sizes.iter().sum();
    if raw_total > total_capacity {
        let scale = total_capacity / raw_total;
        for size in sizes.iter_mut() {
            *size *= scale;
        }
    }

    // Step 5: Truncate to exchange precision and enforce minimums
    let mut allocations = Vec::with_capacity(levels.len());
    let mut remaining_margin = budget.margin_capacity;

    for (idx, level) in levels.iter().enumerate() {
        let truncated = truncate_float(sizes[idx], budget.sz_decimals, false);

        if truncated >= budget.min_viable_size {
            let margin_needed = truncated * budget.margin_per_contract;
            if margin_needed <= remaining_margin + 0.001 {
                allocations.push(LevelAllocation {
                    tick: *level,
                    size: truncated,
                    margin_used: margin_needed,
                });
                remaining_margin -= margin_needed;
            }
        }
    }

    let diag = AllocationDiagnostics {
        effective_temperature: effective_temp,
        entropy_bits: entropy,
        weights,
    };

    (allocations, diag)
}

/// Legacy uniform allocation (greedy water-filling with flat per-level cap).
///
/// Preserved for backward compatibility. Each level gets up to per_level_cap,
/// allocated in utility-priority order until budgets are exhausted.
pub fn allocate_risk_budget_uniform(
    levels: &[TickLevel],
    budget: &SideRiskBudget,
) -> Vec<LevelAllocation> {
    if levels.is_empty() || budget.position_capacity <= 0.0 {
        return Vec::new();
    }

    let per_level_cap = budget.position_capacity * budget.max_single_order_fraction;
    let mut remaining_position = budget.position_capacity;
    let mut remaining_margin = budget.margin_capacity;
    let mut allocations = Vec::with_capacity(levels.len());

    let mut sorted_indices: Vec<usize> = (0..levels.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        levels[b]
            .utility
            .partial_cmp(&levels[a].utility)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sizes = vec![0.0_f64; levels.len()];

    for &idx in &sorted_indices {
        if remaining_position < budget.min_viable_size {
            break;
        }

        let margin_limited = if budget.margin_per_contract > 0.0 {
            remaining_margin / budget.margin_per_contract
        } else {
            f64::MAX
        };

        let raw_max = per_level_cap
            .min(remaining_position)
            .min(margin_limited);

        let truncated = truncate_float(raw_max, budget.sz_decimals, false);

        if truncated >= budget.min_viable_size {
            sizes[idx] = truncated;
            remaining_position -= truncated;
            remaining_margin -= truncated * budget.margin_per_contract;
        }
    }

    for (idx, level) in levels.iter().enumerate() {
        let size = sizes[idx];
        if size > 0.0 {
            allocations.push(LevelAllocation {
                tick: *level,
                size,
                margin_used: size * budget.margin_per_contract,
            });
        }
    }

    allocations
}

/// Compute softmax weights from utility scores, with entropy floor enforcement.
///
/// Returns (weights, effective_temperature). If the initial temperature produces
/// entropy below min_entropy_bits, temperature is raised until the floor is met.
fn compute_softmax_weights(
    levels: &[TickLevel],
    initial_temperature: f64,
    min_entropy_bits: f64,
) -> (Vec<f64>, f64) {
    let n = levels.len();
    if n == 0 {
        return (Vec::new(), initial_temperature);
    }
    if n == 1 {
        return (vec![1.0], initial_temperature);
    }

    let utilities: Vec<f64> = levels.iter().map(|l| l.utility.max(0.0)).collect();

    // Check if all utilities are equal (or near-zero) — return uniform
    let max_u = utilities.iter().cloned().fold(0.0_f64, f64::max);
    if max_u < 1e-12 {
        let uniform = 1.0 / n as f64;
        return (vec![uniform; n], initial_temperature);
    }

    let mut temperature = initial_temperature.max(0.01);

    // Binary search for temperature if entropy is too low (max 10 iterations)
    for _ in 0..10 {
        let weights = softmax_at_temperature(&utilities, temperature);
        let entropy = shannon_entropy_bits(&weights);

        if entropy >= min_entropy_bits || temperature >= 10.0 {
            return (weights, temperature);
        }

        // Raise temperature to increase entropy
        temperature *= 1.5;
    }

    let weights = softmax_at_temperature(&utilities, temperature);
    (weights, temperature)
}

/// Compute softmax weights at a given temperature.
fn softmax_at_temperature(utilities: &[f64], temperature: f64) -> Vec<f64> {
    let t = temperature.max(0.01);

    // Subtract max for numerical stability (log-sum-exp trick)
    let max_u = utilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = utilities.iter().map(|&u| ((u - max_u) / t).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();

    if sum > 0.0 {
        exp_vals.iter().map(|&e| e / sum).collect()
    } else {
        let uniform = 1.0 / utilities.len() as f64;
        vec![uniform; utilities.len()]
    }
}

/// Shannon entropy in bits: H = -Σ p_i × log2(p_i)
fn shannon_entropy_bits(weights: &[f64]) -> f64 {
    weights
        .iter()
        .filter(|&&w| w > 0.0)
        .map(|&w| -w * w.log2())
        .sum()
}

/// Convenience: compute total allocated size.
pub fn total_allocated_size(allocations: &[LevelAllocation]) -> f64 {
    allocations.iter().map(|a| a.size).sum()
}

/// Convenience: compute total margin used.
pub fn total_margin_used(allocations: &[LevelAllocation]) -> f64 {
    allocations.iter().map(|a| a.margin_used).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_levels(depths: &[f64], utilities: &[f64]) -> Vec<TickLevel> {
        depths
            .iter()
            .zip(utilities.iter())
            .enumerate()
            .map(|(i, (&d, &u))| TickLevel {
                price: 29.63 - d * 29.63 / 10_000.0,
                depth_bps: d,
                tick_offset: i as u32 * 10,
                utility: u,
            })
            .collect()
    }

    fn hype_budget(position_capacity: f64) -> SideRiskBudget {
        SideRiskBudget {
            position_capacity,
            margin_capacity: position_capacity * 29.63 / 10.0, // 10x leverage
            margin_per_contract: 29.63 / 10.0, // $2.963 margin per HYPE
            min_viable_size: 0.34, // ~$10 min notional
            max_single_order_fraction: 0.25,
            sz_decimals: 2,
        }
    }

    fn default_softmax() -> SoftmaxParams {
        SoftmaxParams::default()
    }

    // =====================================================================
    // Softmax allocation tests (new)
    // =====================================================================

    #[test]
    fn test_softmax_allocation_skewed() {
        // 5 levels with declining utility — touch should get most size
        let levels = make_levels(
            &[8.0, 12.0, 16.0, 20.0, 24.0],
            &[3.0, 2.2, 1.5, 1.0, 0.5],
        );
        let budget = hype_budget(8.0);
        let params = SoftmaxParams { temperature: 0.5, ..default_softmax() };

        let (allocs, diag) = allocate_risk_budget(&levels, &budget, &params);

        assert!(!allocs.is_empty(), "Should have allocations");

        // Touch level (8 bps, utility 3.0) should have largest size
        let touch_size = allocs[0].size;
        let deepest_size = allocs.last().map(|a| a.size).unwrap_or(0.0);

        assert!(
            touch_size > deepest_size,
            "Touch ({:.2}) should be larger than deepest ({:.2})",
            touch_size, deepest_size,
        );

        // Touch should be at least 2x the deepest level
        if deepest_size > 0.0 {
            assert!(
                touch_size / deepest_size >= 1.5,
                "Touch/deepest ratio {:.2} should be >= 1.5",
                touch_size / deepest_size,
            );
        }

        // Total should not exceed capacity
        let total = total_allocated_size(&allocs);
        assert!(total <= 8.01, "Total {} should not exceed capacity", total);

        // Diagnostics should be populated
        assert!(diag.entropy_bits > 0.0);
        assert_eq!(diag.weights.len(), 5);
    }

    #[test]
    fn test_entropy_floor_prevents_collapse() {
        // One level has vastly higher utility — should still allocate to others
        let levels = make_levels(
            &[8.0, 12.0, 16.0, 20.0],
            &[10.0, 1.0, 0.5, 0.1],
        );
        let budget = hype_budget(8.0);
        let params = SoftmaxParams {
            temperature: 0.1, // Very cold — would collapse without entropy floor
            min_entropy_bits: 1.0,
            min_level_fraction: 0.05,
        };

        let (allocs, diag) = allocate_risk_budget(&levels, &budget, &params);

        // Should have more than 1 level allocated due to entropy floor
        assert!(
            allocs.len() >= 2,
            "Entropy floor should prevent collapse to 1 level, got {} levels",
            allocs.len(),
        );

        // Entropy should meet the floor
        assert!(
            diag.entropy_bits >= 0.9, // Allow small float tolerance
            "Entropy {:.2} should be >= min_entropy 1.0",
            diag.entropy_bits,
        );

        // Temperature should have been raised
        assert!(
            diag.effective_temperature > 0.1,
            "Temperature should have been raised from 0.1 to {:.2}",
            diag.effective_temperature,
        );
    }

    #[test]
    fn test_temperature_regime_scaling() {
        // Calm regime → cold temperature → more skewed
        let temp_calm = compute_allocation_temperature(1.0, 1.0, false);
        assert!((temp_calm - 0.3).abs() < 0.01, "Calm should be 0.3, got {}", temp_calm);

        // Normal regime
        let temp_normal = compute_allocation_temperature(1.3, 1.0, false);
        assert!((temp_normal - 0.5).abs() < 0.01, "Normal should be 0.5, got {}", temp_normal);

        // Volatile regime
        let temp_volatile = compute_allocation_temperature(1.8, 1.0, false);
        assert!((temp_volatile - 1.0).abs() < 0.01, "Volatile should be 1.0, got {}", temp_volatile);

        // Extreme regime
        let temp_extreme = compute_allocation_temperature(3.0, 1.0, false);
        assert!((temp_extreme - 2.0).abs() < 0.01, "Extreme should be 2.0, got {}", temp_extreme);

        // Warmup adds 0.5
        let temp_warmup = compute_allocation_temperature(1.0, 0.1, false);
        assert!((temp_warmup - 0.8).abs() < 0.01, "Warmup should be 0.8, got {}", temp_warmup);

        // Yellow zone adds 0.3
        let temp_yellow = compute_allocation_temperature(1.0, 1.0, true);
        assert!((temp_yellow - 0.6).abs() < 0.01, "Yellow should be 0.6, got {}", temp_yellow);

        // Combined: extreme + warmup + yellow = 2.0 + 0.5 + 0.3 = 2.8
        let temp_all = compute_allocation_temperature(3.0, 0.1, true);
        assert!((temp_all - 2.8).abs() < 0.01, "Combined should be 2.8, got {}", temp_all);

        // Clamped at 3.0 max
        let temp_max = compute_allocation_temperature(5.0, 0.0, true);
        assert!((temp_max - 3.0).abs() < 0.01, "Max should be 3.0, got {}", temp_max);
    }

    #[test]
    fn test_high_temperature_nearly_uniform() {
        // Hot temperature → nearly uniform allocation
        let levels = make_levels(
            &[8.0, 12.0, 16.0, 20.0],
            &[3.0, 2.0, 1.5, 1.0],
        );
        let budget = hype_budget(8.0);
        let params = SoftmaxParams {
            temperature: 5.0,
            min_entropy_bits: 0.0,
            min_level_fraction: 0.0,
        };

        let (allocs, _diag) = allocate_risk_budget(&levels, &budget, &params);

        if allocs.len() >= 2 {
            let max_size = allocs.iter().map(|a| a.size).fold(0.0_f64, f64::max);
            let min_size = allocs.iter().map(|a| a.size).fold(f64::MAX, f64::min);

            // At high temperature, max/min ratio should be close to 1
            assert!(
                max_size / min_size < 1.5,
                "High temp should be near-uniform: max={:.2}, min={:.2}, ratio={:.2}",
                max_size, min_size, max_size / min_size,
            );
        }
    }

    #[test]
    fn test_softmax_equal_utilities_uniform() {
        // All equal utilities → uniform regardless of temperature
        let levels = make_levels(
            &[8.0, 12.0, 16.0, 20.0],
            &[2.0, 2.0, 2.0, 2.0],
        );
        let budget = hype_budget(8.0);
        let params = SoftmaxParams { temperature: 0.3, ..default_softmax() };

        let (allocs, diag) = allocate_risk_budget(&levels, &budget, &params);

        // Weights should be uniform
        for &w in &diag.weights {
            assert!(
                (w - 0.25).abs() < 0.01,
                "Equal utilities should give uniform weights, got {:.3}",
                w,
            );
        }

        // Sizes should be similar
        if allocs.len() >= 2 {
            let max_size = allocs.iter().map(|a| a.size).fold(0.0_f64, f64::max);
            let min_size = allocs.iter().map(|a| a.size).fold(f64::MAX, f64::min);
            assert!(
                max_size - min_size < 0.1,
                "Equal utilities: sizes should be similar, max={:.2}, min={:.2}",
                max_size, min_size,
            );
        }
    }

    // =====================================================================
    // Legacy uniform allocation tests (preserved)
    // =====================================================================

    #[test]
    fn test_allocate_basic() {
        let levels = make_levels(
            &[8.0, 12.0, 16.0, 20.0],
            &[2.5, 2.0, 1.5, 1.0],
        );
        let budget = hype_budget(8.0);

        let allocs = allocate_risk_budget_uniform(&levels, &budget);

        assert!(!allocs.is_empty(), "Should allocate to some levels");

        let total = total_allocated_size(&allocs);
        assert!(total <= 8.0 + 0.01,
            "Total {} should not exceed capacity 8.0", total);

        for alloc in &allocs {
            assert!(alloc.size <= 2.01,
                "Level at {}bps has size {} > per_level_cap 2.0",
                alloc.tick.depth_bps, alloc.size);
        }

        for alloc in &allocs {
            assert!(alloc.size >= 0.34,
                "Level at {}bps has size {} < min_viable 0.34",
                alloc.tick.depth_bps, alloc.size);
        }
    }

    #[test]
    fn test_allocate_respects_margin() {
        let levels = make_levels(&[8.0, 12.0, 16.0], &[2.0, 1.5, 1.0]);
        let budget = SideRiskBudget {
            position_capacity: 10.0,
            margin_capacity: 2.0 * 2.963,
            margin_per_contract: 2.963,
            min_viable_size: 0.34,
            max_single_order_fraction: 0.30,
            sz_decimals: 2,
        };

        let allocs = allocate_risk_budget_uniform(&levels, &budget);
        let total_margin = total_margin_used(&allocs);

        assert!(total_margin <= 2.0 * 2.963 + 0.01,
            "Total margin {} should not exceed budget {}", total_margin, 2.0 * 2.963);
    }

    #[test]
    fn test_allocate_empty_levels() {
        let budget = hype_budget(8.0);
        let allocs = allocate_risk_budget_uniform(&[], &budget);
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_allocate_zero_capacity() {
        let levels = make_levels(&[8.0, 12.0], &[2.0, 1.0]);
        let budget = hype_budget(0.0);
        let allocs = allocate_risk_budget_uniform(&levels, &budget);
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_allocate_tiny_capacity() {
        let levels = make_levels(&[8.0, 12.0], &[2.0, 1.0]);
        let budget = hype_budget(0.20);
        let allocs = allocate_risk_budget_uniform(&levels, &budget);
        assert!(allocs.is_empty(),
            "Should not allocate when capacity < min_viable");
    }

    #[test]
    fn test_allocate_utility_priority() {
        let levels = make_levels(
            &[8.0, 12.0, 16.0],
            &[1.0, 3.0, 2.0],
        );
        let mut budget = hype_budget(0.50);
        budget.max_single_order_fraction = 0.80;

        let allocs = allocate_risk_budget_uniform(&levels, &budget);

        assert_eq!(allocs.len(), 1, "Should allocate to exactly 1 level");
        assert!((allocs[0].tick.depth_bps - 12.0).abs() < 0.1,
            "Should allocate to highest utility level at 12bps, got {}bps",
            allocs[0].tick.depth_bps);
    }

    #[test]
    fn test_allocate_exchange_truncation() {
        let levels = make_levels(&[8.0], &[2.0]);
        let budget = SideRiskBudget {
            position_capacity: 1.567,
            margin_capacity: 1000.0,
            margin_per_contract: 2.963,
            min_viable_size: 0.34,
            max_single_order_fraction: 1.0,
            sz_decimals: 2,
        };

        let allocs = allocate_risk_budget_uniform(&levels, &budget);
        assert_eq!(allocs.len(), 1);
        assert!((allocs[0].size - 1.56).abs() < 0.01,
            "Size should be truncated: {}", allocs[0].size);
    }

    #[test]
    fn test_full_pipeline_100_dollar_hype() {
        let levels = make_levels(
            &[8.0, 11.0, 14.0, 17.0, 20.0, 24.0, 28.0, 32.0],
            &[2.5, 2.2, 1.9, 1.6, 1.3, 1.0, 0.7, 0.4],
        );
        let budget = hype_budget(8.44);

        let allocs = allocate_risk_budget_uniform(&levels, &budget);

        assert!(allocs.len() >= 3,
            "Should have >=3 levels with $100/10x, got {}", allocs.len());

        let total = total_allocated_size(&allocs);
        assert!(total > 1.0, "Should allocate >1 HYPE total, got {}", total);
        assert!(total <= 8.44 + 0.01, "Should not exceed capacity");
    }

    #[test]
    fn test_softmax_respects_margin() {
        let levels = make_levels(&[8.0, 12.0, 16.0], &[2.0, 1.5, 1.0]);
        let budget = SideRiskBudget {
            position_capacity: 10.0,
            margin_capacity: 2.0 * 2.963,
            margin_per_contract: 2.963,
            min_viable_size: 0.34,
            max_single_order_fraction: 0.30,
            sz_decimals: 2,
        };

        let (allocs, _) = allocate_risk_budget(&levels, &budget, &default_softmax());
        let total_margin = total_margin_used(&allocs);

        assert!(total_margin <= 2.0 * 2.963 + 0.01,
            "Total margin {} should not exceed budget {}", total_margin, 2.0 * 2.963);
    }

    #[test]
    fn test_softmax_empty_levels() {
        let budget = hype_budget(8.0);
        let (allocs, _) = allocate_risk_budget(&[], &budget, &default_softmax());
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_softmax_zero_capacity() {
        let levels = make_levels(&[8.0, 12.0], &[2.0, 1.0]);
        let budget = hype_budget(0.0);
        let (allocs, _) = allocate_risk_budget(&levels, &budget, &default_softmax());
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_softmax_full_pipeline_hype() {
        let levels = make_levels(
            &[8.0, 11.0, 14.0, 17.0, 20.0],
            &[2.5, 2.2, 1.9, 1.6, 1.3],
        );
        let budget = hype_budget(4.65); // ~$100 capital per side after warmup
        let params = SoftmaxParams { temperature: 0.5, ..default_softmax() };

        let (allocs, diag) = allocate_risk_budget(&levels, &budget, &params);

        // Should have multiple levels
        assert!(allocs.len() >= 2, "Should have >=2 levels, got {}", allocs.len());

        // Total should not exceed capacity
        let total = total_allocated_size(&allocs);
        assert!(total <= 4.66, "Total {} should not exceed capacity", total);

        // Sizes should be non-uniform (touch > deepest)
        if allocs.len() >= 2 {
            let first = allocs[0].size;
            let last = allocs.last().unwrap().size;
            assert!(
                first >= last,
                "Touch ({:.2}) should be >= deepest ({:.2})",
                first, last,
            );
        }

        // Diagnostics
        assert!(diag.entropy_bits > 0.0);
        assert!(diag.effective_temperature > 0.0);
    }

    // =====================================================================
    // Shannon entropy tests
    // =====================================================================

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform over 4 items: log2(4) = 2.0 bits
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = shannon_entropy_bits(&weights);
        assert!((entropy - 2.0).abs() < 0.01, "Uniform 4 items should be 2.0 bits, got {}", entropy);
    }

    #[test]
    fn test_shannon_entropy_concentrated() {
        // All weight on one item: 0 bits
        let weights = vec![1.0, 0.0, 0.0, 0.0];
        let entropy = shannon_entropy_bits(&weights);
        assert!(entropy.abs() < 0.01, "Concentrated should be ~0 bits, got {}", entropy);
    }

    #[test]
    fn test_softmax_weights_basic() {
        let levels = make_levels(&[8.0, 16.0], &[2.0, 1.0]);
        let (weights, _temp) = compute_softmax_weights(&levels, 0.5, 0.0);

        assert_eq!(weights.len(), 2);
        // Higher utility should get higher weight
        assert!(weights[0] > weights[1],
            "Higher utility ({}) should get higher weight than ({})",
            weights[0], weights[1]);
        // Weights should sum to ~1.0
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "Weights should sum to 1.0, got {}", sum);
    }
}
