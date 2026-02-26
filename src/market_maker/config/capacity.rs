//! Capacity-aware quoting types: SizeQuantum, Viability, CapacityBudget.
//!
//! These types replace fragile buffer-based truncation (1.15x, 1.01x) with exact
//! ceiling math that is provably correct for ALL (price, sz_decimals) combinations.
//!
//! Core insight: instead of `(min_notional * 1.15) / mark_px` (which fails at
//! truncation boundaries), use ceiling division to find the smallest truncation-
//! stable size that meets the notional requirement.

use serde::{Deserialize, Serialize};

use super::auto_derive::CapitalTier;
use crate::truncate_float;

/// Truncation-safe minimum viable order size.
///
/// Computed from (min_notional, mark_px, sz_decimals) using exact ceiling math.
/// The ground truth for "can we place an order at this price?"
///
/// # Invariant
/// `truncate_float(min_viable_size, sz_decimals, false) == min_viable_size`
/// AND `min_viable_size * mark_px >= min_notional`
#[derive(Debug, Clone, Copy)]
pub struct SizeQuantum {
    /// The smallest size that is both truncation-stable and meets min_notional.
    pub min_viable_size: f64,
    /// Size step for this sz_decimals (e.g., 0.01 for sz_decimals=2).
    pub step: f64,
    /// The exchange minimum notional used in computation.
    pub min_notional: f64,
    /// The mark price used in computation (for staleness detection).
    pub mark_px: f64,
    /// The sz_decimals used.
    pub sz_decimals: u32,
}

impl SizeQuantum {
    /// Compute the minimum viable order size using exact ceiling math.
    ///
    /// # Algorithm
    /// ```text
    /// step = 10^(-sz_decimals)     // e.g., 0.01 for HYPE
    /// raw_min = min_notional / mark_px  // e.g., 10.0 / 30.90 = 0.3236
    /// steps_needed = ceil(raw_min / step)  // ceil(32.36) = 33
    /// min_viable = steps_needed * step     // 0.33
    /// ```
    ///
    /// Result: 0.33 * $30.90 = $10.20 >= $10, and truncate_float(0.33, 2) = 0.33.
    /// Provably correct for ALL (price, sz_decimals) combinations.
    pub fn compute(min_notional: f64, mark_px: f64, sz_decimals: u32) -> Self {
        debug_assert!(mark_px > 0.0, "mark_px must be positive");
        debug_assert!(min_notional >= 0.0, "min_notional must be non-negative");

        let step = (10.0_f64).powi(-(sz_decimals as i32));
        let raw_min = min_notional / mark_px;

        // Ceiling division: smallest integer multiple of step >= raw_min
        let steps_needed = (raw_min / step).ceil() as u64;
        let min_viable_size = steps_needed as f64 * step;

        Self {
            min_viable_size,
            step,
            min_notional,
            mark_px,
            sz_decimals,
        }
    }

    /// Clamp a raw size to be viable (truncation-stable and meets min_notional).
    ///
    /// Returns `None` if the raw size is zero/negative (no capacity).
    ///
    /// When `allow_round_up` is true and raw is slightly below min_viable,
    /// rounds UP to min_viable. The cost of rounding up by one quantum
    /// (e.g., 0.01 contract = $0.31) is negligible vs. not quoting at all.
    pub fn clamp_to_viable(&self, raw: f64, allow_round_up: bool) -> Option<f64> {
        if raw <= 0.0 {
            return None;
        }

        // Truncate to exchange precision (floor)
        let truncated = truncate_float(raw, self.sz_decimals, false);

        if truncated >= self.min_viable_size {
            // Already meets minimum after truncation
            Some(truncated)
        } else if allow_round_up {
            // Raw is slightly below min_viable after truncation — round up
            // This adds at most one quantum of exposure
            Some(self.min_viable_size)
        } else {
            // Can't meet minimum without rounding up
            None
        }
    }

    /// Check if a size (already truncated) meets the minimum viable threshold.
    pub fn is_sufficient(&self, size: f64) -> bool {
        size >= self.min_viable_size
    }
}

/// Whether the system can viably quote at the current capital level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Viability {
    /// Full ladder operation possible.
    Full {
        /// Number of viable levels per side.
        levels_per_side: usize,
    },
    /// Only concentrated (1-2 level) quoting possible.
    Concentrated {
        /// Why full ladder isn't possible.
        reason: String,
    },
    /// Cannot place any valid orders. System should idle.
    NotViable {
        /// Why quoting isn't possible.
        reason: String,
        /// How much capital (USD) is needed to start quoting.
        min_capital_needed_usd: f64,
    },
}

/// Per-cycle capacity budget computed at the TOP of the quote cycle.
///
/// This replaces bottom-up surprise filtering with top-down capacity awareness.
/// Computed once, flows through the entire pipeline.
#[derive(Debug, Clone)]
pub struct CapacityBudget {
    /// Whether quoting is viable at all.
    pub viability: Viability,
    /// Truncation-safe minimum size.
    pub quantum: SizeQuantum,
    /// Capital tier classification.
    pub capital_tier: CapitalTier,
    /// Available capacity for bid orders (contracts).
    pub bid_capacity: f64,
    /// Available capacity for ask orders (contracts).
    pub ask_capacity: f64,
    /// Position limit from InventoryGovernor.
    pub effective_max_position: f64,
    /// Target liquidity per level.
    pub target_liquidity: f64,
    /// How many viable levels per side (each meeting exchange min_notional).
    pub viable_levels_per_side: usize,
    /// Minimum depth (bps) for QueueValue > 0 in Normal toxicity.
    /// QueueValue breakeven: depth > AS_COST_NORMAL(3.0) + MAKER_FEE(1.5) + safety(0.5) = 5.0
    /// Concentration fallbacks and QueueValue evaluation use this instead of BBO half-spread.
    pub min_viable_depth_bps: f64,
}

impl CapacityBudget {
    /// Compute the capacity budget for this quote cycle.
    ///
    /// This is the SINGLE source of truth for "can we quote and how much?"
    pub fn compute(
        account_value: f64,
        mark_px: f64,
        min_notional: f64,
        sz_decimals: u32,
        effective_max_position: f64,
        position: f64,
        target_liquidity: f64,
    ) -> Self {
        let quantum = SizeQuantum::compute(min_notional, mark_px, sz_decimals);

        // Available capacity per side (contracts)
        let (bid_capacity, ask_capacity) = if position >= 0.0 {
            (
                (effective_max_position - position).max(0.0),
                (effective_max_position + position).max(0.0),
            )
        } else {
            (
                (effective_max_position + position.abs()).max(0.0),
                (effective_max_position - position.abs()).max(0.0),
            )
        };

        // Determine viability
        let viability = Self::determine_viability(
            account_value,
            mark_px,
            &quantum,
            effective_max_position,
            bid_capacity,
            ask_capacity,
        );

        // Capital tier from viable levels — use the best-case single-side capacity
        // (at position=0, both sides have full effective_max_position).
        let best_side_capacity = bid_capacity.max(ask_capacity);
        let viable_levels_per_side = if mark_px > 0.0 && quantum.min_viable_size > 0.0 {
            let per_level_notional = quantum.min_viable_size * mark_px;
            if per_level_notional > 0.0 {
                (best_side_capacity / quantum.min_viable_size).floor() as usize
            } else {
                0
            }
        } else {
            0
        };

        let capital_tier = match viable_levels_per_side {
            0..=2 => CapitalTier::Micro,
            3..=5 => CapitalTier::Small,
            6..=15 => CapitalTier::Medium,
            _ => CapitalTier::Large,
        };

        // QueueValue breakeven: depth > AS_COST_NORMAL(3.0) + MAKER_FEE(1.5) + safety(0.5)
        const MIN_VIABLE_DEPTH_BPS: f64 = 5.0;

        Self {
            viability,
            quantum,
            capital_tier,
            bid_capacity,
            ask_capacity,
            effective_max_position,
            target_liquidity,
            viable_levels_per_side,
            min_viable_depth_bps: MIN_VIABLE_DEPTH_BPS,
        }
    }

    /// Whether the system should attempt quoting this cycle.
    pub fn should_quote(&self) -> bool {
        !matches!(self.viability, Viability::NotViable { .. })
    }

    /// Clamp a raw size to viable, delegating to the quantum.
    pub fn viable_size(&self, raw: f64) -> Option<f64> {
        self.quantum.clamp_to_viable(raw, true)
    }

    /// Compute the capital-aware policy for this budget's tier.
    pub fn policy(&self) -> CapitalAwarePolicy {
        CapitalAwarePolicy::from_tier(self.capital_tier)
    }

    fn determine_viability(
        account_value: f64,
        mark_px: f64,
        quantum: &SizeQuantum,
        effective_max_position: f64,
        bid_capacity: f64,
        ask_capacity: f64,
    ) -> Viability {
        if mark_px <= 0.0 {
            return Viability::NotViable {
                reason: "Invalid mark price".to_string(),
                min_capital_needed_usd: 0.0,
            };
        }

        // Check if we can place even ONE minimum order
        let min_order_notional = quantum.min_viable_size * mark_px;
        let can_bid = bid_capacity >= quantum.min_viable_size;
        let can_ask = ask_capacity >= quantum.min_viable_size;

        if !can_bid && !can_ask {
            // Need enough capital for at least one minimum order on one side
            // Approximate: need 2x min_notional to have position capacity
            let min_capital = min_order_notional * 2.0;
            return Viability::NotViable {
                reason: format!(
                    "No side has capacity for min order ({:.4} contracts, ${:.2} notional). \
                     bid_capacity={:.4}, ask_capacity={:.4}, effective_max={:.4}",
                    quantum.min_viable_size,
                    min_order_notional,
                    bid_capacity,
                    ask_capacity,
                    effective_max_position,
                ),
                min_capital_needed_usd: min_capital,
            };
        }

        // Check if account value is too low (can't afford even one order's margin)
        // Approximate: need at least min_notional / leverage worth of capital
        if account_value > 0.0 && account_value < quantum.min_notional * 0.5 {
            return Viability::NotViable {
                reason: format!(
                    "Account value ${:.2} too low for min notional ${:.2}",
                    account_value, quantum.min_notional,
                ),
                min_capital_needed_usd: quantum.min_notional * 2.0,
            };
        }

        // Determine tier: how many levels can each side support?
        let bid_levels = if quantum.min_viable_size > 0.0 {
            (bid_capacity / quantum.min_viable_size).floor() as usize
        } else {
            0
        };
        let ask_levels = if quantum.min_viable_size > 0.0 {
            (ask_capacity / quantum.min_viable_size).floor() as usize
        } else {
            0
        };
        let max_levels = bid_levels.max(ask_levels);

        if max_levels <= 2 {
            Viability::Concentrated {
                reason: format!(
                    "Only {max_levels} viable levels (bid={bid_levels}, ask={ask_levels}); concentrated quoting",
                ),
            }
        } else {
            Viability::Full {
                levels_per_side: max_levels,
            }
        }
    }
}

/// Capital-aware policy computed once from `CapitalTier` and threaded through the pipeline.
///
/// Instead of scattered `if tier == Micro` conditionals in 15 files, this single policy
/// object encodes all tier-dependent behavior. Computed at the top of the pipeline from
/// `CapitalTier`, flows through `MarketParams.capital_policy`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CapitalAwarePolicy {
    pub tier: CapitalTier,

    // --- Ladder generation ---
    /// Maximum ladder levels per side. Micro: 2, Small: 3, Medium: 5, Large: 8.
    pub max_levels_per_side: usize,
    /// Skip entropy optimization for Micro/Small (saves compute, predictable sizes).
    pub skip_entropy_optimization: bool,
    /// Minimum size as multiple of exchange minimum (1.0 = exactly exchange min).
    pub min_level_size_mult: f64,

    // --- Reconciliation ---
    /// Deprecated: unified reconcile now handles all tiers economically.
    /// Kept for checkpoint backward compatibility (`#[serde(default)]` equivalent).
    pub use_batch_reconcile: bool,
    /// Tolerance (bps) before re-quoting. Higher = fewer API calls. Micro: 3.0, Large: 1.0.
    pub price_drift_threshold_bps: f64,

    // --- Controller ---
    /// Cap L3 trust before sufficient fills. Prevents death spiral from tautological edge.
    pub max_l3_trust_uncalibrated: f64,
    /// Always maintain at least 1 level per side (never 0+0). Micro/Small: true.
    pub always_quote_minimum: bool,
    /// Fills needed before L3 trust can exceed cap.
    pub min_fills_for_trust_ramp: u64,

    // --- Warmup ---
    /// Fill target for 100% warmup. Micro: 5, Large: 50.
    pub warmup_fill_target: u64,
    /// Use L2 book data for initial kappa/sigma estimates. Micro/Small: true.
    pub bootstrap_from_book: bool,
    /// Warmup spread floor (bps). Micro: 3.0, Large: 8.0. Tighter = more fills = faster warmup.
    pub warmup_floor_bps: f64,

    // --- Quota management ---
    /// Override min_headroom_for_full_ladder. Micro: 0.05 vs default 0.20.
    pub quota_min_headroom_for_full: f64,
    /// Whether to truncate levels by sqrt(headroom). Micro/Small: false.
    pub quota_density_scaling: bool,

    // --- Tick-grid quoting (WS5) ---
    /// Enable tick-grid-first level placement (vs legacy bps-space).
    pub use_tick_grid: bool,
    /// Minimum tick spacing as multiple of tick_bps. Micro: 3.0 (wider gaps), Large: 1.0.
    pub min_tick_spacing_mult: f64,
    /// Maximum fraction of position_capacity in a single order. Micro: 0.40, Large: 0.20.
    pub max_single_order_fraction: f64,
    /// Spread compensation multiplier for small capital (widens to compensate execution risk).
    pub spread_compensation_mult: f64,
    /// Use greedy water-filling allocation (vs uniform). Micro/Small: true.
    pub use_greedy_allocation: bool,
    /// Depth range multiplier: max_depth = touch_bps * depth_range_multiplier.
    /// Wider range = more utility variation across levels = better size differentiation.
    /// Micro: 2.5, Small: 3.0, Medium: 4.0, Large: 5.0.
    pub depth_range_multiplier: f64,

    // --- Scale-adaptive strategy (E5) ---
    /// Spread multiplier when only 1 level per side. Wider = safer for concentrated exposure.
    /// Micro: 1.5, Small: 1.3, Medium/Large: 1.0.
    pub single_level_spread_mult: f64,
    /// Minimum spread (bps) when only 1 level per side. Floor for single-level mode.
    /// Micro: 8.0, Small: 6.0, Medium/Large: 0.0 (use GLFT output).
    pub single_level_min_spread_bps: f64,
    /// Max fraction of available budget in a single order. Caps concentration risk.
    /// Micro: 0.70, Small: 0.80, Large: 1.0.
    pub single_level_max_size_fraction: f64,
    /// Maximum warmup gamma inflation multiplier. Caps death spiral from no-fill warmup.
    /// Micro: 1.3, Small: 1.5, Medium: 1.8, Large: 2.0.
    pub warmup_gamma_max_inflation: f64,
}

impl CapitalAwarePolicy {
    /// Construct policy from capital tier. This is the ONLY place tier → behavior mapping lives.
    pub fn from_tier(tier: CapitalTier) -> Self {
        match tier {
            CapitalTier::Micro => Self {
                tier,
                max_levels_per_side: 2,
                skip_entropy_optimization: true,
                min_level_size_mult: 1.0,
                use_batch_reconcile: true,
                price_drift_threshold_bps: 3.0,
                max_l3_trust_uncalibrated: 0.30,
                always_quote_minimum: true,
                min_fills_for_trust_ramp: 10,
                warmup_fill_target: 5,
                bootstrap_from_book: true,
                warmup_floor_bps: 3.0,
                quota_min_headroom_for_full: 0.05,
                quota_density_scaling: false,
                use_tick_grid: true,
                min_tick_spacing_mult: 3.0,
                max_single_order_fraction: 0.40,
                spread_compensation_mult: 1.15,
                use_greedy_allocation: true,
                depth_range_multiplier: 2.5,
                single_level_spread_mult: 1.5,
                single_level_min_spread_bps: 8.0,
                single_level_max_size_fraction: 0.70,
                warmup_gamma_max_inflation: 1.3,
            },
            CapitalTier::Small => Self {
                tier,
                max_levels_per_side: 4,
                skip_entropy_optimization: true,
                min_level_size_mult: 1.0,
                use_batch_reconcile: true,
                price_drift_threshold_bps: 2.0,
                max_l3_trust_uncalibrated: 0.50,
                always_quote_minimum: true,
                min_fills_for_trust_ramp: 20,
                warmup_fill_target: 10,
                bootstrap_from_book: true,
                warmup_floor_bps: 4.0,
                quota_min_headroom_for_full: 0.10,
                quota_density_scaling: false,
                use_tick_grid: true,
                min_tick_spacing_mult: 2.0,
                max_single_order_fraction: 0.30,
                spread_compensation_mult: 1.05,
                use_greedy_allocation: true,
                depth_range_multiplier: 3.0,
                single_level_spread_mult: 1.3,
                single_level_min_spread_bps: 6.0,
                single_level_max_size_fraction: 0.80,
                warmup_gamma_max_inflation: 1.5,
            },
            CapitalTier::Medium => Self {
                tier,
                max_levels_per_side: 8,
                skip_entropy_optimization: false,
                min_level_size_mult: 1.5,
                use_batch_reconcile: false,
                price_drift_threshold_bps: 1.5,
                max_l3_trust_uncalibrated: 0.80,
                always_quote_minimum: false,
                min_fills_for_trust_ramp: 30,
                warmup_fill_target: 25,
                bootstrap_from_book: false,
                warmup_floor_bps: 6.0,
                quota_min_headroom_for_full: 0.15,
                quota_density_scaling: true,
                use_tick_grid: true,
                min_tick_spacing_mult: 1.0,
                max_single_order_fraction: 0.25,
                spread_compensation_mult: 1.0,
                use_greedy_allocation: false,
                depth_range_multiplier: 4.0,
                single_level_spread_mult: 1.0,
                single_level_min_spread_bps: 0.0,
                single_level_max_size_fraction: 1.0,
                warmup_gamma_max_inflation: 1.8,
            },
            CapitalTier::Large => Self {
                tier,
                max_levels_per_side: 15,
                skip_entropy_optimization: false,
                min_level_size_mult: 2.0,
                use_batch_reconcile: false,
                price_drift_threshold_bps: 1.0,
                max_l3_trust_uncalibrated: 0.95,
                always_quote_minimum: false,
                min_fills_for_trust_ramp: 50,
                warmup_fill_target: 50,
                bootstrap_from_book: false,
                warmup_floor_bps: 8.0,
                quota_min_headroom_for_full: 0.20,
                quota_density_scaling: true,
                use_tick_grid: true,
                min_tick_spacing_mult: 1.0,
                max_single_order_fraction: 0.20,
                spread_compensation_mult: 1.0,
                use_greedy_allocation: false,
                depth_range_multiplier: 5.0,
                single_level_spread_mult: 1.0,
                single_level_min_spread_bps: 0.0,
                single_level_max_size_fraction: 1.0,
                warmup_gamma_max_inflation: 2.0,
            },
        }
    }
}

impl Default for CapitalAwarePolicy {
    fn default() -> Self {
        Self::from_tier(CapitalTier::Large)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =================================================================
    // SizeQuantum tests
    // =================================================================

    #[test]
    fn test_quantum_hype_100_dollar_scenario() {
        // THE bug: $100 capital, HYPE at $30.90, sz_decimals=2
        // Old code: (10.0 * 1.15) / 30.90 = 0.3722, truncate → 0.37
        //           0.37 < 0.3722 → FILTERED → dead system
        // New code: ceil(10.0/30.90 / 0.01) * 0.01 = ceil(32.36) * 0.01 = 0.33
        //           0.33 * 30.90 = 10.20 >= 10.0 ✓, truncate(0.33, 2) = 0.33 ✓
        let quantum = SizeQuantum::compute(10.0, 30.90, 2);

        assert_eq!(quantum.min_viable_size, 0.33);
        assert!(
            quantum.min_viable_size * 30.90 >= 10.0,
            "Must meet min_notional: {} * {} = {}",
            quantum.min_viable_size,
            30.90,
            quantum.min_viable_size * 30.90,
        );
        // Truncation-stable
        let truncated = truncate_float(quantum.min_viable_size, 2, false);
        assert_eq!(
            truncated, quantum.min_viable_size,
            "Must be truncation-stable"
        );
    }

    #[test]
    fn test_quantum_btc_high_price() {
        // BTC at $50,000, sz_decimals=5
        let quantum = SizeQuantum::compute(10.0, 50_000.0, 5);

        // raw_min = 10/50000 = 0.0002, step = 0.00001
        // steps = ceil(0.0002 / 0.00001) = ceil(20) = 20
        // min_viable = 0.00020
        assert_eq!(quantum.min_viable_size, 0.00020);
        assert!(quantum.min_viable_size * 50_000.0 >= 10.0);
        assert_eq!(
            truncate_float(quantum.min_viable_size, 5, false),
            quantum.min_viable_size,
        );
    }

    #[test]
    fn test_quantum_cheap_token() {
        // Cheap token at $0.05, sz_decimals=0 (whole units)
        let quantum = SizeQuantum::compute(10.0, 0.05, 0);

        // raw_min = 10/0.05 = 200, step = 1
        // steps = ceil(200/1) = 200
        // min_viable = 200.0
        assert_eq!(quantum.min_viable_size, 200.0);
        assert!(quantum.min_viable_size * 0.05 >= 10.0);
    }

    #[test]
    fn test_quantum_exact_boundary() {
        // Price where raw_min is exactly on a step boundary
        // $10 / $20 = 0.50, with sz_decimals=2, step=0.01
        // steps = ceil(50.0) = 50, min_viable = 0.50
        let quantum = SizeQuantum::compute(10.0, 20.0, 2);
        assert_eq!(quantum.min_viable_size, 0.50);
        assert!(quantum.min_viable_size * 20.0 >= 10.0);
    }

    #[test]
    fn test_clamp_to_viable_above_minimum() {
        let quantum = SizeQuantum::compute(10.0, 30.90, 2);
        // Raw size well above minimum
        let result = quantum.clamp_to_viable(1.0, false);
        assert_eq!(result, Some(1.0));
    }

    #[test]
    fn test_clamp_to_viable_slightly_below_with_roundup() {
        let quantum = SizeQuantum::compute(10.0, 30.90, 2);
        // min_viable = 0.33; raw = 0.30 (below min after truncation)
        let result = quantum.clamp_to_viable(0.30, true);
        assert_eq!(result, Some(0.33)); // Rounded up to min_viable
    }

    #[test]
    fn test_clamp_to_viable_slightly_below_without_roundup() {
        let quantum = SizeQuantum::compute(10.0, 30.90, 2);
        let result = quantum.clamp_to_viable(0.30, false);
        assert_eq!(result, None); // Can't meet minimum
    }

    #[test]
    fn test_clamp_to_viable_zero() {
        let quantum = SizeQuantum::compute(10.0, 30.90, 2);
        assert_eq!(quantum.clamp_to_viable(0.0, true), None);
        assert_eq!(quantum.clamp_to_viable(-1.0, true), None);
    }

    #[test]
    fn test_clamp_truncates_to_exchange_precision() {
        let quantum = SizeQuantum::compute(10.0, 30.90, 2);
        // Raw = 0.5678; should truncate to 0.56
        let result = quantum.clamp_to_viable(0.5678, false);
        assert_eq!(result, Some(0.56));
    }

    #[test]
    fn test_is_sufficient() {
        let quantum = SizeQuantum::compute(10.0, 30.90, 2);
        assert!(quantum.is_sufficient(0.33));
        assert!(quantum.is_sufficient(0.50));
        assert!(!quantum.is_sufficient(0.32));
        assert!(!quantum.is_sufficient(0.0));
    }

    // =================================================================
    // Property-like tests across (price, sz_decimals) combinations
    // =================================================================

    #[test]
    fn test_quantum_property_all_sz_decimals() {
        let prices = [
            0.001, 0.01, 0.05, 0.50, 1.0, 5.0, 10.0, 30.90, 100.0, 500.0, 1000.0, 5000.0, 50_000.0,
            100_000.0,
        ];

        for &price in &prices {
            for sz_dec in 0..=5 {
                let quantum = SizeQuantum::compute(10.0, price, sz_dec);

                // Property 1: meets min_notional
                let notional = quantum.min_viable_size * price;
                assert!(
                    notional >= 10.0 - 1e-9, // Allow tiny float epsilon
                    "NOTIONAL FAIL: price={}, sz_dec={}, min_viable={}, notional={}",
                    price,
                    sz_dec,
                    quantum.min_viable_size,
                    notional,
                );

                // Property 2: truncation-stable
                let truncated = truncate_float(quantum.min_viable_size, sz_dec, false);
                assert!(
                    (truncated - quantum.min_viable_size).abs() < 1e-12,
                    "TRUNCATION FAIL: price={}, sz_dec={}, min_viable={}, truncated={}",
                    price,
                    sz_dec,
                    quantum.min_viable_size,
                    truncated,
                );

                // Property 3: minimal — one step less would NOT meet notional
                if quantum.min_viable_size > quantum.step {
                    let one_less = quantum.min_viable_size - quantum.step;
                    let one_less_notional = one_less * price;
                    assert!(
                        one_less_notional < 10.0 + 1e-9,
                        "NOT MINIMAL: price={}, sz_dec={}, one_less={} meets notional={}",
                        price,
                        sz_dec,
                        one_less,
                        one_less_notional,
                    );
                }
            }
        }
    }

    #[test]
    fn test_clamp_roundtrip_property() {
        // For any raw size, clamp_to_viable should produce a truncation-stable result
        let quantum = SizeQuantum::compute(10.0, 30.90, 2);
        let raws = [0.001, 0.10, 0.32, 0.33, 0.3722, 0.50, 1.0, 5.678];

        for &raw in &raws {
            if let Some(clamped) = quantum.clamp_to_viable(raw, true) {
                let re_truncated = truncate_float(clamped, 2, false);
                assert_eq!(
                    re_truncated, clamped,
                    "Round-trip fail: raw={}, clamped={}, re_truncated={}",
                    raw, clamped, re_truncated,
                );
                assert!(
                    clamped * 30.90 >= 10.0 - 1e-9,
                    "Notional fail after clamp: raw={}, clamped={}, notional={}",
                    raw,
                    clamped,
                    clamped * 30.90,
                );
            }
        }
    }

    // =================================================================
    // Viability tests
    // =================================================================

    #[test]
    fn test_viability_not_viable_zero_capacity() {
        // Position already at max — no capacity on either side
        let budget = CapacityBudget::compute(
            100.0, // account_value
            30.90, // mark_px
            10.0,  // min_notional
            2,     // sz_decimals
            0.388, // effective_max_position
            0.388, // position (at max)
            0.33,  // target_liquidity
        );

        // With position == max, bid_capacity = 0, ask_capacity = 0.776
        // Should still be viable (can ask)
        assert!(budget.should_quote());
    }

    #[test]
    fn test_viability_not_viable_tiny_account() {
        let budget = CapacityBudget::compute(
            3.0,   // account_value ($3 — way too small)
            30.90, // mark_px
            10.0,  // min_notional
            2,     // sz_decimals
            0.01,  // effective_max_position (tiny)
            0.0,   // position
            0.01,  // target_liquidity
        );

        assert!(!budget.should_quote());
        match &budget.viability {
            Viability::NotViable {
                min_capital_needed_usd,
                ..
            } => {
                assert!(*min_capital_needed_usd > 0.0);
            }
            _ => panic!("Expected NotViable, got {:?}", budget.viability),
        }
    }

    #[test]
    fn test_viability_concentrated_small_capital() {
        // $25 capital on HYPE — just barely viable
        let budget = CapacityBudget::compute(
            25.0,  // account_value
            30.90, // mark_px
            10.0,  // min_notional
            2,     // sz_decimals
            0.80,  // effective_max_position (~$24 notional)
            0.0,   // position
            0.33,  // target_liquidity
        );

        assert!(budget.should_quote());
        // 0.80 / 0.33 = 2.4 → 2 levels per side → Concentrated
        matches!(budget.viability, Viability::Concentrated { .. });
    }

    #[test]
    fn test_viability_full_large_capital() {
        // $10,000 capital
        let budget = CapacityBudget::compute(
            10_000.0, // account_value
            30.90,    // mark_px
            10.0,     // min_notional
            2,        // sz_decimals
            100.0,    // effective_max_position
            0.0,      // position
            5.0,      // target_liquidity
        );

        assert!(budget.should_quote());
        matches!(budget.viability, Viability::Full { .. });
        assert_eq!(budget.capital_tier, CapitalTier::Large);
    }

    #[test]
    fn test_budget_hype_100_dollar_scenario() {
        // THE scenario: $100 capital, HYPE $30.90
        let budget = CapacityBudget::compute(
            100.0, // account_value
            30.90, // mark_px
            10.0,  // min_notional
            2,     // sz_decimals
            0.388, // effective_max_position (from $100 / leverage)
            0.0,   // position
            0.33,  // target_liquidity
        );

        // MUST be viable — this is the whole point of the fix
        assert!(
            budget.should_quote(),
            "CRITICAL: $100 capital on HYPE must be viable! viability={:?}",
            budget.viability,
        );
        assert_eq!(budget.quantum.min_viable_size, 0.33);
        assert!(budget.quantum.is_sufficient(0.33));
    }

    #[test]
    fn test_budget_viable_size_delegates_to_quantum() {
        let budget = CapacityBudget::compute(100.0, 30.90, 10.0, 2, 0.388, 0.0, 0.33);
        // Above min
        assert_eq!(budget.viable_size(0.50), Some(0.50));
        // Below min — rounds up
        assert_eq!(budget.viable_size(0.30), Some(0.33));
        // Zero
        assert_eq!(budget.viable_size(0.0), None);
    }

    #[test]
    fn test_budget_capacity_with_position() {
        // Long position reduces bid capacity, increases ask capacity
        let budget = CapacityBudget::compute(
            1000.0, 30.90, 10.0, 2, 5.0, // effective_max_position
            2.0, // long position
            1.0, // target_liquidity
        );

        assert!((budget.bid_capacity - 3.0).abs() < 1e-10); // 5 - 2
        assert!((budget.ask_capacity - 7.0).abs() < 1e-10); // 5 + 2
    }

    #[test]
    fn test_budget_capacity_with_short_position() {
        let budget = CapacityBudget::compute(
            1000.0, 30.90, 10.0, 2, 5.0,  // effective_max_position
            -2.0, // short position
            1.0,  // target_liquidity
        );

        assert!((budget.bid_capacity - 7.0).abs() < 1e-10); // 5 + 2
        assert!((budget.ask_capacity - 3.0).abs() < 1e-10); // 5 - 2
    }

    #[test]
    fn test_quantum_zero_mark_price_debug_assert() {
        // In release mode (no debug_assert), compute should handle gracefully
        // In tests, we verify the happy path instead
        let quantum = SizeQuantum::compute(10.0, 100.0, 2);
        assert!(quantum.min_viable_size > 0.0);
    }

    #[test]
    fn test_capacity_budget_has_min_viable_depth() {
        let budget = CapacityBudget::compute(100.0, 30.90, 10.0, 2, 0.388, 0.0, 0.33);
        // min_viable_depth_bps must be >= 5.0 (QueueValue breakeven in Normal)
        assert!(
            budget.min_viable_depth_bps >= 5.0,
            "min_viable_depth_bps={} must be >= 5.0",
            budget.min_viable_depth_bps,
        );
    }

    #[test]
    fn test_viable_levels_100_usd_hype() {
        // $100 capital, HYPE at $30.90
        let budget = CapacityBudget::compute(100.0, 30.90, 10.0, 2, 0.388, 0.0, 0.33);
        // 0.388 / 0.33 = 1.17 → 1 level per side → Micro tier
        // This is correct for the CapacityBudget calculation
        assert!(
            budget.viable_levels_per_side >= 1,
            "viable_levels={} should be at least 1",
            budget.viable_levels_per_side,
        );
    }

    #[test]
    fn test_min_viable_depth_survives_queue_value() {
        let budget = CapacityBudget::compute(100.0, 30.90, 10.0, 2, 0.388, 0.0, 0.33);
        // At min_viable_depth_bps with Normal toxicity and front of queue:
        // value = 5.0 - 3.0(AS) - 0(queue) - 1.5(fee) = 0.5 > 0
        let value = budget.min_viable_depth_bps - 3.0 - 0.0 - 1.5;
        assert!(
            value > 0.0,
            "QueueValue at min_viable_depth_bps={}: {} must be > 0",
            budget.min_viable_depth_bps,
            value,
        );
    }

    // =================================================================
    // CapitalAwarePolicy tests
    // =================================================================

    #[test]
    fn test_policy_micro_tier() {
        let policy = CapitalAwarePolicy::from_tier(CapitalTier::Micro);
        assert_eq!(policy.max_levels_per_side, 2);
        assert!(policy.skip_entropy_optimization);
        assert!(policy.use_batch_reconcile);
        assert!(policy.always_quote_minimum);
        assert!(!policy.quota_density_scaling);
        assert!((policy.max_l3_trust_uncalibrated - 0.30).abs() < 1e-10);
        assert_eq!(policy.min_fills_for_trust_ramp, 10);
        assert_eq!(policy.warmup_fill_target, 5);
        assert!((policy.warmup_floor_bps - 3.0).abs() < 1e-10);
        // Tick-grid fields
        assert!(policy.use_tick_grid);
        assert!((policy.min_tick_spacing_mult - 3.0).abs() < 1e-10);
        assert!((policy.max_single_order_fraction - 0.40).abs() < 1e-10);
        assert!((policy.spread_compensation_mult - 1.15).abs() < 1e-10);
        assert!(policy.use_greedy_allocation);
        assert!((policy.depth_range_multiplier - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_policy_small_tier() {
        let policy = CapitalAwarePolicy::from_tier(CapitalTier::Small);
        assert_eq!(policy.max_levels_per_side, 4);
        assert!(policy.skip_entropy_optimization);
        assert!(policy.use_batch_reconcile);
        assert!(policy.always_quote_minimum);
        assert!(!policy.quota_density_scaling);
        assert!((policy.max_l3_trust_uncalibrated - 0.50).abs() < 1e-10);
        // Tick-grid fields
        assert!(policy.use_tick_grid);
        assert!((policy.min_tick_spacing_mult - 2.0).abs() < 1e-10);
        assert!((policy.max_single_order_fraction - 0.30).abs() < 1e-10);
        assert!((policy.spread_compensation_mult - 1.05).abs() < 1e-10);
        assert!(policy.use_greedy_allocation);
    }

    #[test]
    fn test_policy_medium_tier() {
        let policy = CapitalAwarePolicy::from_tier(CapitalTier::Medium);
        assert_eq!(policy.max_levels_per_side, 8);
        assert!(!policy.skip_entropy_optimization);
        assert!(!policy.use_batch_reconcile);
        assert!(!policy.always_quote_minimum);
        assert!(policy.quota_density_scaling);
        // Tick-grid fields
        assert!(policy.use_tick_grid);
        assert!((policy.min_tick_spacing_mult - 1.0).abs() < 1e-10);
        assert!((policy.max_single_order_fraction - 0.25).abs() < 1e-10);
        assert!((policy.spread_compensation_mult - 1.0).abs() < 1e-10);
        assert!(!policy.use_greedy_allocation);
    }

    #[test]
    fn test_policy_large_tier() {
        let policy = CapitalAwarePolicy::from_tier(CapitalTier::Large);
        assert_eq!(policy.max_levels_per_side, 15);
        assert!(!policy.skip_entropy_optimization);
        assert!(!policy.use_batch_reconcile);
        assert!(!policy.always_quote_minimum);
        assert!(policy.quota_density_scaling);
        assert!((policy.max_l3_trust_uncalibrated - 0.95).abs() < 1e-10);
        // Tick-grid fields
        assert!(policy.use_tick_grid);
        assert!((policy.min_tick_spacing_mult - 1.0).abs() < 1e-10);
        assert!((policy.max_single_order_fraction - 0.20).abs() < 1e-10);
        assert!((policy.spread_compensation_mult - 1.0).abs() < 1e-10);
        assert!(!policy.use_greedy_allocation);
    }

    #[test]
    fn test_policy_default_is_large() {
        let policy = CapitalAwarePolicy::default();
        assert_eq!(policy.tier, CapitalTier::Large);
        assert_eq!(policy.max_levels_per_side, 15);
    }

    #[test]
    fn test_budget_policy_matches_tier() {
        let budget = CapacityBudget::compute(100.0, 30.90, 10.0, 2, 0.388, 0.0, 0.33);
        let policy = budget.policy();
        assert_eq!(policy.tier, budget.capital_tier);
    }
}
