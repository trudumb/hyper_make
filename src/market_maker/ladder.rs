//! Ladder quoting engine for multi-level quote generation.
//!
//! Implements multi-level quote ladders with:
//! - Geometric or linear depth spacing
//! - Fill intensity modeling: λ(δ) = σ²/δ²
//! - Spread capture: SC(δ) = δ - AS₀ × exp(-δ/δ_char) - fees
//! - Size allocation proportional to marginal value: λ(δ) × SC(δ)
//! - GLFT inventory skew applied to entire ladder
//! - **Constrained variational optimization** for capital-efficient allocation

use serde::{Deserialize, Serialize};

use super::adverse_selection::DepthDecayAS;
use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

/// A single quote level in the ladder
#[derive(Debug, Clone, Copy)]
pub struct LadderLevel {
    /// Price of this level
    pub price: f64,
    /// Size at this level
    pub size: f64,
    /// Distance from mid in basis points
    pub depth_bps: f64,
}

/// Multi-level quote ladder
#[derive(Debug, Clone, Default)]
pub struct Ladder {
    /// Bid levels, sorted best-to-worst (highest price first)
    pub bids: Vec<LadderLevel>,
    /// Ask levels, sorted best-to-worst (lowest price first)
    pub asks: Vec<LadderLevel>,
}

/// Configuration for ladder generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LadderConfig {
    /// Number of levels per side (default: 5)
    pub num_levels: usize,
    /// Minimum depth from mid in basis points (default: 2 bps)
    pub min_depth_bps: f64,
    /// Maximum depth from mid in basis points (default: 50 bps)
    pub max_depth_bps: f64,
    /// Use geometric spacing (true) or linear spacing (false)
    pub geometric_spacing: bool,
    /// Minimum size per level (orders below this are skipped)
    pub min_level_size: f64,
    /// Trading fees in basis points (maker + taker / 2)
    pub fees_bps: f64,
    /// Adverse selection decay characteristic depth in bps
    pub as_decay_bps: f64,
}

impl Default for LadderConfig {
    fn default() -> Self {
        Self {
            num_levels: 5,
            min_depth_bps: 2.0,
            max_depth_bps: 50.0,
            geometric_spacing: true,
            min_level_size: 0.001,
            fees_bps: 0.5,
            as_decay_bps: 10.0,
        }
    }
}

/// Parameters for ladder generation derived from market state
#[derive(Debug, Clone)]
pub struct LadderParams {
    /// Mid/fair price to quote around
    pub mid_price: f64,
    /// Volatility (per-second)
    pub sigma: f64,
    /// Order flow intensity (κ)
    pub kappa: f64,
    /// Trade arrival intensity (volume ticks per second)
    pub arrival_intensity: f64,
    /// Adverse selection at touch in basis points (legacy, used if depth_decay_as is None)
    pub as_at_touch_bps: f64,
    /// Total size budget per side
    pub total_size: f64,
    /// Inventory ratio q/Q_max in [-1, 1]
    pub inventory_ratio: f64,
    /// Risk aversion parameter (γ)
    pub gamma: f64,
    /// Time horizon T = 1/λ (seconds)
    pub time_horizon: f64,
    /// Price decimals for rounding
    pub decimals: u32,
    /// Size decimals for truncation
    pub sz_decimals: u32,
    /// Minimum order notional value
    pub min_notional: f64,
    /// Optional depth-dependent AS model (calibrated from fills)
    /// If Some, uses first-principles exponential decay: AS(δ) = AS₀ × exp(-δ/δ_char)
    /// If None, falls back to legacy config-based decay
    pub depth_decay_as: Option<DepthDecayAS>,
}

impl Ladder {
    /// Generate a ladder using the GLFT-based approach.
    ///
    /// Steps:
    /// 1. Compute depth spacing (geometric or linear)
    /// 2. Calculate fill intensity λ(δ) and spread capture SC(δ) for each depth
    /// 3. Allocate sizes proportional to marginal value λ(δ) × SC(δ)
    /// 4. Build raw ladder with price/size at each level
    /// 5. Apply inventory skew (shift prices and reduce sizes on one side)
    pub fn generate(config: &LadderConfig, params: &LadderParams) -> Self {
        // 1. Compute depth spacing
        let depths = compute_depths(config);

        // 2. Compute fill intensity and spread capture for each depth
        let intensities: Vec<f64> = depths
            .iter()
            .map(|&d| fill_intensity(d, params.sigma, params.kappa))
            .collect();

        // Use calibrated depth-dependent AS if available, else legacy config
        let spreads: Vec<f64> = if let Some(ref depth_decay) = params.depth_decay_as {
            // First-principles: use calibrated AS(δ) = AS₀ × exp(-δ/δ_char)
            depths
                .iter()
                .map(|&d| depth_decay.spread_capture(d, config.fees_bps))
                .collect()
        } else {
            // Legacy: use config-based AS decay
            depths
                .iter()
                .map(|&d| {
                    spread_capture(
                        d,
                        params.as_at_touch_bps,
                        config.as_decay_bps,
                        config.fees_bps,
                    )
                })
                .collect()
        };

        // 3. Allocate sizes proportional to marginal value
        let sizes = allocate_sizes(
            &intensities,
            &spreads,
            params.total_size,
            config.min_level_size,
        );

        // 4. Build raw ladder (before skew)
        let mut ladder = build_raw_ladder(
            &depths,
            &sizes,
            params.mid_price,
            params.decimals,
            params.sz_decimals,
            params.min_notional,
        );

        // 5. Apply inventory skew (with re-rounding for exchange precision)
        apply_inventory_skew(
            &mut ladder,
            params.inventory_ratio,
            params.gamma,
            params.sigma,
            params.time_horizon,
            params.mid_price,
            params.decimals,
            params.sz_decimals,
        );

        ladder
    }

    /// Check if ladder is empty (no valid quotes on either side)
    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// Total bid size across all levels
    pub fn total_bid_size(&self) -> f64 {
        self.bids.iter().map(|l| l.size).sum()
    }

    /// Total ask size across all levels
    pub fn total_ask_size(&self) -> f64 {
        self.asks.iter().map(|l| l.size).sum()
    }

    /// Number of bid levels
    pub fn num_bids(&self) -> usize {
        self.bids.len()
    }

    /// Number of ask levels
    pub fn num_asks(&self) -> usize {
        self.asks.len()
    }

    /// Best bid price (highest)
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Best ask price (lowest)
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }
}

/// Compute depth levels using geometric or linear spacing.
///
/// Geometric: δ_k = δ_min × r^(k-1) where r = (δ_max/δ_min)^(1/(K-1))
/// Linear: δ_k = δ_min + k × (δ_max - δ_min) / (K-1)
fn compute_depths(config: &LadderConfig) -> Vec<f64> {
    let k = config.num_levels;
    if k == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![config.min_depth_bps];
    }

    if config.geometric_spacing {
        // Geometric spacing: r = (max/min)^(1/(K-1))
        let r = (config.max_depth_bps / config.min_depth_bps).powf(1.0 / (k - 1) as f64);
        (0..k)
            .map(|i| config.min_depth_bps * r.powi(i as i32))
            .collect()
    } else {
        // Linear spacing
        let step = (config.max_depth_bps - config.min_depth_bps) / (k - 1) as f64;
        (0..k)
            .map(|i| config.min_depth_bps + step * i as f64)
            .collect()
    }
}

/// Fill intensity model: λ(δ) = σ²/δ² × κ
///
/// Diffusion-driven fills: probability of price reaching depth δ
/// scales as σ²/δ². Multiply by κ (order flow intensity) for scaling.
fn fill_intensity(depth_bps: f64, sigma: f64, kappa: f64) -> f64 {
    let depth = depth_bps / 10000.0; // Convert bps to fraction
    if depth < EPSILON {
        return 0.0;
    }
    // σ is per-second, so σ² gives variance per second
    let diffusion_term = sigma.powi(2) / depth.powi(2);
    diffusion_term * kappa
}

/// Spread capture: SC(δ) = δ - AS₀ × exp(-δ/δ_char) - fees
///
/// Expected profit from capturing the spread at depth δ:
/// - δ: raw spread captured (in bps)
/// - AS₀ × exp(-δ/δ_char): adverse selection cost that decays with depth
/// - fees: trading fees
fn spread_capture(depth_bps: f64, as_at_touch_bps: f64, as_decay_bps: f64, fees_bps: f64) -> f64 {
    let as_cost = as_at_touch_bps * (-depth_bps / as_decay_bps).exp();
    depth_bps - as_cost - fees_bps
}

/// Allocate sizes proportional to marginal value: λ(δ) × SC(δ)
///
/// Sizes are normalized to sum to total_size, with levels below
/// min_size set to zero (orders too small to be worth placing).
fn allocate_sizes(
    intensities: &[f64],
    spreads: &[f64],
    total_size: f64,
    min_size: f64,
) -> Vec<f64> {
    // Compute marginal value at each depth
    let marginal_values: Vec<f64> = intensities
        .iter()
        .zip(spreads.iter())
        .map(|(&lambda, &sc)| (lambda * sc).max(0.0)) // Only positive values
        .collect();

    let total: f64 = marginal_values.iter().sum();
    if total <= EPSILON {
        // No profitable levels
        return vec![0.0; marginal_values.len()];
    }

    // Normalize to total_size
    marginal_values
        .iter()
        .map(|&v| {
            let raw = total_size * v / total;
            if raw < min_size {
                0.0
            } else {
                raw
            }
        })
        .collect()
}

// ============================================================================
// Constrained Variational Ladder Optimization (Phase 2)
// ============================================================================

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

/// Build raw ladder from depths and sizes (before inventory skew).
///
/// Creates symmetric bid/ask levels around mid price, applying
/// proper price rounding and size truncation per exchange requirements.
fn build_raw_ladder(
    depths: &[f64],
    sizes: &[f64],
    mid: f64,
    decimals: u32,
    sz_decimals: u32,
    min_notional: f64,
) -> Ladder {
    let mut bids = Vec::new();
    let mut asks = Vec::new();

    for (&depth_bps, &size) in depths.iter().zip(sizes.iter()) {
        if size < EPSILON {
            continue;
        }

        // Calculate price offset from mid
        let offset = mid * (depth_bps / 10000.0);

        // Round prices per exchange requirements
        let bid_price = round_to_significant_and_decimal(mid - offset, 5, decimals);
        let ask_price = round_to_significant_and_decimal(mid + offset, 5, decimals);

        // Truncate size
        let size = truncate_float(size, sz_decimals, false);

        // Check minimum notional
        if bid_price * size >= min_notional {
            bids.push(LadderLevel {
                price: bid_price,
                size,
                depth_bps,
            });
        }
        if ask_price * size >= min_notional {
            asks.push(LadderLevel {
                price: ask_price,
                size,
                depth_bps,
            });
        }
    }

    // Sort: bids highest first (best bid), asks lowest first (best ask)
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

/// Apply GLFT inventory skew to the ladder.
///
/// 1. Shift all prices by reservation price offset: γσ²qT
/// 2. Reduce sizes on the side that would increase inventory
/// 3. Re-round prices and truncate sizes to maintain exchange precision
#[allow(clippy::too_many_arguments)]
fn apply_inventory_skew(
    ladder: &mut Ladder,
    inventory_ratio: f64,
    gamma: f64,
    sigma: f64,
    time_horizon: f64,
    mid: f64,
    decimals: u32,
    sz_decimals: u32,
) {
    if inventory_ratio.abs() < EPSILON {
        return; // No inventory, no skew needed
    }

    // Reservation price offset: γσ²qT (as fraction of mid)
    // Positive inventory → positive skew → shift prices down
    let skew_fraction = inventory_ratio * gamma * sigma.powi(2) * time_horizon;
    let offset = mid * skew_fraction;

    // Shift all prices by reservation price offset and RE-ROUND to exchange precision
    for level in &mut ladder.bids {
        level.price = round_to_significant_and_decimal(level.price - offset, 5, decimals);
    }
    for level in &mut ladder.asks {
        level.price = round_to_significant_and_decimal(level.price - offset, 5, decimals);
    }

    // Size skew: reduce side that increases position
    // Cap reduction at 90% to never completely remove quotes
    let size_reduction = inventory_ratio.abs().min(0.9);

    if inventory_ratio > 0.0 {
        // Long inventory: reduce bid sizes (don't want to buy more)
        for level in &mut ladder.bids {
            level.size = truncate_float(level.size * (1.0 - size_reduction), sz_decimals, false);
        }
    } else {
        // Short inventory: reduce ask sizes (don't want to sell more)
        for level in &mut ladder.asks {
            level.size = truncate_float(level.size * (1.0 - size_reduction), sz_decimals, false);
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
    fn test_default_config() {
        let config = LadderConfig::default();
        assert_eq!(config.num_levels, 5);
        assert!((config.min_depth_bps - 2.0).abs() < 0.01);
        assert!((config.max_depth_bps - 50.0).abs() < 0.01);
        assert!(config.geometric_spacing);
    }

    #[test]
    fn test_compute_depths_geometric() {
        let config = LadderConfig {
            num_levels: 5,
            min_depth_bps: 2.0,
            max_depth_bps: 32.0,
            geometric_spacing: true,
            ..Default::default()
        };
        let depths = compute_depths(&config);

        assert_eq!(depths.len(), 5);
        // Geometric spacing with ratio 2: 2, 4, 8, 16, 32
        assert!((depths[0] - 2.0).abs() < 0.01);
        assert!((depths[1] - 4.0).abs() < 0.01);
        assert!((depths[2] - 8.0).abs() < 0.01);
        assert!((depths[3] - 16.0).abs() < 0.01);
        assert!((depths[4] - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_depths_linear() {
        let config = LadderConfig {
            num_levels: 5,
            min_depth_bps: 2.0,
            max_depth_bps: 10.0,
            geometric_spacing: false,
            ..Default::default()
        };
        let depths = compute_depths(&config);

        assert_eq!(depths.len(), 5);
        // Linear spacing: 2, 4, 6, 8, 10
        assert!((depths[0] - 2.0).abs() < 0.01);
        assert!((depths[1] - 4.0).abs() < 0.01);
        assert!((depths[2] - 6.0).abs() < 0.01);
        assert!((depths[3] - 8.0).abs() < 0.01);
        assert!((depths[4] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_depths_single_level() {
        let config = LadderConfig {
            num_levels: 1,
            min_depth_bps: 5.0,
            max_depth_bps: 50.0,
            geometric_spacing: true,
            ..Default::default()
        };
        let depths = compute_depths(&config);

        assert_eq!(depths.len(), 1);
        assert!((depths[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_depths_zero_levels() {
        let config = LadderConfig {
            num_levels: 0,
            ..Default::default()
        };
        let depths = compute_depths(&config);
        assert!(depths.is_empty());
    }

    #[test]
    fn test_fill_intensity() {
        // Higher intensity at tighter depths
        let i_tight = fill_intensity(2.0, 0.001, 100.0);
        let i_wide = fill_intensity(20.0, 0.001, 100.0);

        assert!(i_tight > i_wide);

        // Intensity at 0 depth should be 0 (avoid division by zero)
        let i_zero = fill_intensity(0.0, 0.001, 100.0);
        assert!((i_zero).abs() < EPSILON);
    }

    #[test]
    fn test_spread_capture() {
        // At touch (2 bps) with high AS, spread capture should be negative
        let sc_touch = spread_capture(2.0, 5.0, 10.0, 0.5);
        assert!(sc_touch < 0.0); // Unprofitable at touch

        // At deeper levels, AS decays and spread capture becomes positive
        let sc_deep = spread_capture(20.0, 5.0, 10.0, 0.5);
        assert!(sc_deep > 15.0); // Should be close to depth - fees

        // With zero AS, spread capture = depth - fees
        let sc_no_as = spread_capture(10.0, 0.0, 10.0, 0.5);
        assert!((sc_no_as - 9.5).abs() < 0.01);
    }

    #[test]
    fn test_allocate_sizes() {
        let intensities = vec![10.0, 5.0, 2.0];
        let spreads = vec![1.0, 2.0, 3.0]; // marginal: 10, 10, 6

        let sizes = allocate_sizes(&intensities, &spreads, 1.0, 0.0);

        assert_eq!(sizes.len(), 3);
        let total: f64 = sizes.iter().sum();
        assert!((total - 1.0).abs() < 0.01); // Should sum to total_size
    }

    #[test]
    fn test_allocate_sizes_with_min_size() {
        let intensities = vec![10.0, 0.1, 0.1];
        let spreads = vec![1.0, 1.0, 1.0]; // marginal: 10, 0.1, 0.1

        let sizes = allocate_sizes(&intensities, &spreads, 1.0, 0.1);

        // First level should get most of the size
        // Small levels should be zeroed out
        assert!(sizes[0] > 0.8);
        assert!(sizes[1] < 0.1 || sizes[1] == 0.0);
    }

    #[test]
    fn test_allocate_sizes_all_negative() {
        let intensities = vec![1.0, 1.0, 1.0];
        let spreads = vec![-1.0, -2.0, -3.0]; // All negative spread capture

        let sizes = allocate_sizes(&intensities, &spreads, 1.0, 0.0);

        // All marginal values should be clamped to 0, so all sizes should be 0
        assert!(sizes.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_build_raw_ladder() {
        let depths = vec![2.0, 5.0, 10.0];
        let sizes = vec![0.5, 0.3, 0.2];
        let mid = 100.0;

        let ladder = build_raw_ladder(&depths, &sizes, mid, 2, 4, 10.0);

        assert_eq!(ladder.bids.len(), 3);
        assert_eq!(ladder.asks.len(), 3);

        // Best bid should be highest (closest to mid)
        assert!(ladder.bids[0].price > ladder.bids[1].price);
        // Best ask should be lowest (closest to mid)
        assert!(ladder.asks[0].price < ladder.asks[1].price);
    }

    #[test]
    fn test_build_raw_ladder_min_notional() {
        let depths = vec![2.0, 5.0];
        let sizes = vec![0.001, 0.5]; // First level too small for $10 notional at $100

        let ladder = build_raw_ladder(&depths, &sizes, 100.0, 2, 4, 10.0);

        // Only second level should make it (0.5 * $100 = $50 > $10)
        assert_eq!(ladder.bids.len(), 1);
        assert_eq!(ladder.asks.len(), 1);
    }

    #[test]
    fn test_inventory_skew_long() {
        let mut ladder = Ladder {
            bids: vec![LadderLevel {
                price: 100.0,
                size: 1.0,
                depth_bps: 2.0,
            }],
            asks: vec![LadderLevel {
                price: 100.2,
                size: 1.0,
                depth_bps: 2.0,
            }],
        };

        // Long inventory: bids should move down, bid sizes reduced
        apply_inventory_skew(&mut ladder, 0.5, 0.3, 0.01, 10.0, 100.0, 2, 4);

        assert!(ladder.bids[0].price < 100.0); // Price shifted down
        assert!(ladder.bids[0].size < 1.0); // Size reduced
        assert!((ladder.asks[0].size - 1.0).abs() < 0.01); // Ask size unchanged
    }

    #[test]
    fn test_inventory_skew_short() {
        let mut ladder = Ladder {
            bids: vec![LadderLevel {
                price: 100.0,
                size: 1.0,
                depth_bps: 2.0,
            }],
            asks: vec![LadderLevel {
                price: 100.2,
                size: 1.0,
                depth_bps: 2.0,
            }],
        };

        // Short inventory: prices shift up, ask sizes reduced
        apply_inventory_skew(&mut ladder, -0.5, 0.3, 0.01, 10.0, 100.0, 2, 4);

        assert!(ladder.bids[0].price > 100.0); // Price shifted up
        assert!(ladder.asks[0].size < 1.0); // Ask size reduced
        assert!((ladder.bids[0].size - 1.0).abs() < 0.01); // Bid size unchanged
    }

    #[test]
    fn test_inventory_skew_zero() {
        let mut ladder = Ladder {
            bids: vec![LadderLevel {
                price: 100.0,
                size: 1.0,
                depth_bps: 2.0,
            }],
            asks: vec![LadderLevel {
                price: 100.2,
                size: 1.0,
                depth_bps: 2.0,
            }],
        };

        // Zero inventory: no changes
        apply_inventory_skew(&mut ladder, 0.0, 0.3, 0.01, 10.0, 100.0, 2, 4);

        assert!((ladder.bids[0].price - 100.0).abs() < EPSILON);
        assert!((ladder.bids[0].size - 1.0).abs() < EPSILON);
        assert!((ladder.asks[0].size - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_inventory_skew_price_precision() {
        // Test that prices remain properly rounded after skew (BTC-like: 0 decimals)
        let mut ladder = Ladder {
            bids: vec![LadderLevel {
                price: 87833.0,
                size: 0.02,
                depth_bps: 2.0,
            }],
            asks: vec![LadderLevel {
                price: 87850.0,
                size: 0.02,
                depth_bps: 2.0,
            }],
        };

        // Apply skew with 0 decimal places (BTC)
        apply_inventory_skew(&mut ladder, 0.5, 0.3, 0.0001, 10.0, 87840.0, 0, 5);

        // Prices should be integers (no fractional part)
        assert_eq!(ladder.bids[0].price, ladder.bids[0].price.round());
        assert_eq!(ladder.asks[0].price, ladder.asks[0].price.round());
    }

    #[test]
    fn test_ladder_generate_full() {
        let config = LadderConfig {
            num_levels: 3,
            min_depth_bps: 5.0,
            max_depth_bps: 20.0,
            geometric_spacing: false,
            min_level_size: 0.01,
            fees_bps: 0.5,
            as_decay_bps: 10.0,
        };

        let params = LadderParams {
            mid_price: 100.0,
            sigma: 0.001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            as_at_touch_bps: 2.0, // Low AS so all levels are profitable
            total_size: 1.0,
            inventory_ratio: 0.0,
            gamma: 0.3,
            time_horizon: 10.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            depth_decay_as: None, // Use legacy AS config
        };

        let ladder = Ladder::generate(&config, &params);

        // Should have quotes on both sides
        assert!(!ladder.is_empty());
        assert!(ladder.num_bids() > 0);
        assert!(ladder.num_asks() > 0);

        // Total sizes should be close to total_size (accounting for min_notional filtering)
        let total_bid = ladder.total_bid_size();
        let total_ask = ladder.total_ask_size();
        assert!(total_bid > 0.0);
        assert!(total_ask > 0.0);

        // Best bid < mid < best ask
        assert!(ladder.best_bid().unwrap() < params.mid_price);
        assert!(ladder.best_ask().unwrap() > params.mid_price);
    }

    #[test]
    fn test_ladder_generate_with_inventory() {
        let config = LadderConfig::default();

        let params_neutral = LadderParams {
            mid_price: 1000.0,
            sigma: 0.001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            as_at_touch_bps: 1.0,
            total_size: 1.0,
            inventory_ratio: 0.0,
            gamma: 0.5,
            time_horizon: 10.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            depth_decay_as: None,
        };

        let params_long = LadderParams {
            inventory_ratio: 0.5,
            ..params_neutral.clone()
        };

        let ladder_neutral = Ladder::generate(&config, &params_neutral);
        let ladder_long = Ladder::generate(&config, &params_long);

        // Long inventory should have smaller bid sizes
        assert!(ladder_long.total_bid_size() < ladder_neutral.total_bid_size());
        // Ask sizes should be similar
        assert!((ladder_long.total_ask_size() - ladder_neutral.total_ask_size()).abs() < 0.5);
    }

    #[test]
    fn test_ladder_with_depth_decay_as() {
        use super::DepthDecayAS;

        let config = LadderConfig {
            num_levels: 5,
            min_depth_bps: 2.0,
            max_depth_bps: 20.0,
            geometric_spacing: true,
            min_level_size: 0.01,
            fees_bps: 0.5,
            as_decay_bps: 10.0, // Ignored when depth_decay_as is Some
        };

        // Calibrated AS model with higher AS at touch
        let depth_decay = DepthDecayAS::new(5.0, 8.0); // AS₀=5bp, δ_char=8bp

        let params = LadderParams {
            mid_price: 100.0,
            sigma: 0.001,
            kappa: 100.0,
            arrival_intensity: 1.0,
            as_at_touch_bps: 2.0, // Ignored when depth_decay_as is Some
            total_size: 1.0,
            inventory_ratio: 0.0,
            gamma: 0.3,
            time_horizon: 10.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
            depth_decay_as: Some(depth_decay),
        };

        let ladder = Ladder::generate(&config, &params);

        // Should have quotes on both sides
        assert!(!ladder.is_empty());

        // With high AS at touch (5bp), shallow levels should have less size allocated
        // because their spread capture is lower (or negative)
        // Deep levels should get more size because AS decays

        // Verify total size is close to target
        let total_bid = ladder.total_bid_size();
        let total_ask = ladder.total_ask_size();
        assert!(total_bid > 0.0);
        assert!(total_ask > 0.0);
    }

    // ========================================================================
    // ConstrainedLadderOptimizer Tests
    // ========================================================================

    #[test]
    fn test_constrained_optimizer_basic() {
        let optimizer = ConstrainedLadderOptimizer::new(
            1000.0,  // margin_available
            0.5,     // max_position
            0.01,    // min_size
            10.0,    // min_notional
            100.0,   // price
            10.0,    // leverage
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
            50.0,    // margin_available (only $50)
            10.0,    // max_position (plenty of room)
            0.01,    // min_size
            10.0,    // min_notional
            100.0,   // price
            10.0,    // leverage (margin_per_unit = 10)
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

        let levels = vec![
            LevelOptimizationParams {
                depth_bps: 5.0,
                fill_intensity: 10.0,
                spread_capture: 5.0,
                margin_per_unit: 10.0,
            },
        ];

        let result = optimizer.optimize(&levels);

        // Should be position-constrained
        assert!(result.position_used <= 0.1 + EPSILON);
        // Should not have used much margin
        assert!(result.margin_used <= 10.0 + EPSILON); // 0.1 * 10 margin_per_unit
    }

    #[test]
    fn test_constrained_optimizer_greedy_ordering() {
        let optimizer = ConstrainedLadderOptimizer::new(
            100.0,   // margin_available
            1.0,     // max_position
            0.01,    // min_size
            10.0,    // min_notional
            100.0,   // price
            10.0,    // leverage (margin_per_unit = 10)
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
        let optimizer = ConstrainedLadderOptimizer::new(
            1000.0, 10.0, 0.01, 10.0, 100.0, 10.0,
        );

        let result = optimizer.optimize(&[]);

        assert!(result.sizes.is_empty());
        assert_eq!(result.margin_used, 0.0);
        assert_eq!(result.position_used, 0.0);
        assert_eq!(result.binding_constraint, BindingConstraint::None);
    }

    #[test]
    fn test_constrained_optimizer_negative_spread_capture() {
        let optimizer = ConstrainedLadderOptimizer::new(
            1000.0, 10.0, 0.01, 10.0, 100.0, 10.0,
        );

        // All levels have negative spread capture (unprofitable)
        let levels = vec![
            LevelOptimizationParams {
                depth_bps: 2.0,
                fill_intensity: 10.0,
                spread_capture: -5.0, // Negative!
                margin_per_unit: 10.0,
            },
        ];

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
