//! Risk-budget allocation for tick-grid levels.
//!
//! Replaces the sequential `position_cap → per_level_cap → truncation → min_notional_filter → fallback`
//! chain with a joint greedy water-filling optimization that satisfies all constraints by construction.
//!
//! # Constraints
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

/// Allocate sizes to tick levels using greedy water-filling.
///
/// Levels are processed in order of utility per risk (highest first).
/// Each level gets the maximum size it can support, subject to remaining budgets.
///
/// # Algorithm
/// ```text
/// Sort levels by utility descending
/// For each level:
///   per_level_cap = min(position_capacity × max_single_order_fraction, remaining_position)
///   margin_limited = remaining_margin / margin_per_contract
///   max_size = min(per_level_cap, margin_limited)
///   size = truncate_to_exchange(max_size)
///   if size < min_viable_size: size = 0  // can't afford this level
///   Deduct from remaining budgets
/// ```
pub fn allocate_risk_budget(
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

    // Process levels in order (already sorted by depth from select_optimal_ticks,
    // but we want to allocate by utility). Sort a copy by utility descending.
    let mut sorted_indices: Vec<usize> = (0..levels.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        levels[b]
            .utility
            .partial_cmp(&levels[a].utility)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // First pass: allocate by utility priority
    let mut sizes = vec![0.0_f64; levels.len()];

    for &idx in &sorted_indices {
        if remaining_position < budget.min_viable_size {
            break; // No more position budget
        }

        // Compute max size for this level
        let margin_limited = if budget.margin_per_contract > 0.0 {
            remaining_margin / budget.margin_per_contract
        } else {
            f64::MAX
        };

        let raw_max = per_level_cap
            .min(remaining_position)
            .min(margin_limited);

        // Truncate to exchange precision
        let truncated = truncate_float(raw_max, budget.sz_decimals, false);

        if truncated >= budget.min_viable_size {
            sizes[idx] = truncated;
            remaining_position -= truncated;
            remaining_margin -= truncated * budget.margin_per_contract;
        }
    }

    // Build allocation results in original level order (depth order)
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

    #[test]
    fn test_allocate_basic() {
        let levels = make_levels(
            &[8.0, 12.0, 16.0, 20.0],
            &[2.5, 2.0, 1.5, 1.0],
        );
        let budget = hype_budget(8.0); // 8 HYPE per side

        let allocs = allocate_risk_budget(&levels, &budget);

        // Should have allocations
        assert!(!allocs.is_empty(), "Should allocate to some levels");

        // Total should not exceed capacity
        let total = total_allocated_size(&allocs);
        assert!(total <= 8.0 + 0.01,
            "Total {} should not exceed capacity 8.0", total);

        // Each level should not exceed per_level_cap (25% of 8 = 2)
        for alloc in &allocs {
            assert!(alloc.size <= 2.01,
                "Level at {}bps has size {} > per_level_cap 2.0",
                alloc.tick.depth_bps, alloc.size);
        }

        // Each allocated level should meet min_viable_size
        for alloc in &allocs {
            assert!(alloc.size >= 0.34,
                "Level at {}bps has size {} < min_viable 0.34",
                alloc.tick.depth_bps, alloc.size);
        }
    }

    #[test]
    fn test_allocate_respects_margin() {
        let levels = make_levels(&[8.0, 12.0, 16.0], &[2.0, 1.5, 1.0]);
        // Very tight margin: only enough for ~2 HYPE total
        let budget = SideRiskBudget {
            position_capacity: 10.0,
            margin_capacity: 2.0 * 2.963, // $5.926 margin = ~2 HYPE at 10x
            margin_per_contract: 2.963,
            min_viable_size: 0.34,
            max_single_order_fraction: 0.30,
            sz_decimals: 2,
        };

        let allocs = allocate_risk_budget(&levels, &budget);
        let total_margin = total_margin_used(&allocs);

        assert!(total_margin <= 2.0 * 2.963 + 0.01,
            "Total margin {} should not exceed budget {}", total_margin, 2.0 * 2.963);
    }

    #[test]
    fn test_allocate_empty_levels() {
        let budget = hype_budget(8.0);
        let allocs = allocate_risk_budget(&[], &budget);
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_allocate_zero_capacity() {
        let levels = make_levels(&[8.0, 12.0], &[2.0, 1.0]);
        let budget = hype_budget(0.0);
        let allocs = allocate_risk_budget(&levels, &budget);
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_allocate_tiny_capacity() {
        let levels = make_levels(&[8.0, 12.0], &[2.0, 1.0]);
        let budget = hype_budget(0.20); // Below min_viable_size
        let allocs = allocate_risk_budget(&levels, &budget);
        // Can't allocate anything since 0.20 < min_viable 0.34
        assert!(allocs.is_empty(),
            "Should not allocate when capacity < min_viable");
    }

    #[test]
    fn test_allocate_utility_priority() {
        // Level 2 has highest utility — should get allocated first
        let levels = make_levels(
            &[8.0, 12.0, 16.0],
            &[1.0, 3.0, 2.0],
        );
        // Only enough for 1 level — use higher single-order fraction so
        // per_level_cap (0.50 × 0.80 = 0.40) exceeds min_viable_size (0.34).
        let mut budget = hype_budget(0.50);
        budget.max_single_order_fraction = 0.80;

        let allocs = allocate_risk_budget(&levels, &budget);

        // Should allocate to the highest-utility level (12 bps)
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
            max_single_order_fraction: 1.0, // Allow full allocation
            sz_decimals: 2,
        };

        let allocs = allocate_risk_budget(&levels, &budget);
        assert_eq!(allocs.len(), 1);
        // 1.567 truncated to 2 decimals = 1.56
        assert!((allocs[0].size - 1.56).abs() < 0.01,
            "Size should be truncated: {}", allocs[0].size);
    }

    #[test]
    fn test_full_pipeline_100_dollar_hype() {
        // Integration test: $100 capital, 10x leverage, HYPE at $29.63
        // max_position ≈ 16.87 (after WS0 fix)
        // Per side ≈ 8.44 HYPE
        let levels = make_levels(
            &[8.0, 11.0, 14.0, 17.0, 20.0, 24.0, 28.0, 32.0],
            &[2.5, 2.2, 1.9, 1.6, 1.3, 1.0, 0.7, 0.4],
        );
        let budget = hype_budget(8.44);

        let allocs = allocate_risk_budget(&levels, &budget);

        // Should have multiple levels allocated
        assert!(allocs.len() >= 3,
            "Should have >=3 levels with $100/10x, got {}", allocs.len());

        let total = total_allocated_size(&allocs);
        assert!(total > 1.0, "Should allocate >1 HYPE total, got {}", total);
        assert!(total <= 8.44 + 0.01, "Should not exceed capacity");
    }
}
