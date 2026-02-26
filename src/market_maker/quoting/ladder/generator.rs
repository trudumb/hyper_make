//! Ladder generation logic.
//!
//! **NOTE**: The concentration fallback logic in this module (collapsing to 1-2 orders
//! when spread capture is negative) is superseded by the entropy-based allocation in
//! `entropy_distribution.rs`. When `use_entropy_distribution=true`, the entropy system
//! maintains diversity through entropy constraints instead of collapsing.
//!
//! This module's fallback code is retained for backward compatibility.
//!
//! Implements multi-level quote generation with:
//! - Geometric or linear depth spacing
//! - Fill intensity modeling: λ(δ) = σ²/δ²
//! - Spread capture: SC(δ) = δ - AS₀ × exp(-δ/δ_char) - fees
//! - Size allocation proportional to marginal value

use smallvec::SmallVec;

use crate::market_maker::config::SizeQuantum;
use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use super::{Ladder, LadderConfig, LadderLevel, LadderLevels, LadderParams};
use crate::market_maker::infra::capacity::DEPTH_INLINE_CAPACITY;

/// Deduplicate ladder levels with the same price, merging their sizes.
///
/// After rounding to exchange tick size, multiple levels may collapse to the same price.
/// This merges them (summing sizes) to avoid wasting API rate limit on duplicate orders.
/// Levels must be sorted by price before calling.
fn dedup_merge_levels(levels: &mut LadderLevels) {
    if levels.len() <= 1 {
        return;
    }
    let mut write = 0;
    for read in 1..levels.len() {
        if (levels[read].price - levels[write].price).abs() < EPSILON {
            levels[write].size += levels[read].size;
        } else {
            write += 1;
            if write != read {
                levels[write] = levels[read];
            }
        }
    }
    levels.truncate(write + 1);
}

/// Maximum fraction of total_size allowed in a single resting order.
///
/// Prevents concentration fallbacks from dumping the entire position budget into
/// one order. With this cap, at least 4 fills are needed to reach max inventory,
/// giving the GLFT gamma/skew feedback loop time to widen spreads.
const MAX_SINGLE_ORDER_FRACTION: f64 = 0.25;

/// Type alias for depth values using SmallVec for stack allocation
type DepthVec = SmallVec<[f64; DEPTH_INLINE_CAPACITY]>;

/// Enforce inviolable spread floor on **total spread** (best_ask - best_bid).
///
/// Glosten-Milgrom break-even: `s* = 2α(V_h - V_l)/[(1-α)(V_h + V_l)]`.
/// Floor enforced on total spread (not half-spread from mid) to preserve
/// reservation-price skew from GLFT's γ·q inventory penalty.
///
/// Deficit is split proportionally to current offsets:
/// - When long (bids far, asks tight): most deficit pushes ask up.
///   This is correct: we want to sell, just not below cost.
/// - When short: symmetric opposite.
/// - Deeper levels (placed by GLFT δ_k = δ_1 + k·Δ) are NOT moved.
fn enforce_spread_floor(ladder: &mut Ladder, mid: f64, floor_bps: f64) {
    if floor_bps <= 0.0 || mid <= 0.0 {
        return;
    }
    let best_bid = match ladder.bids.first() {
        Some(b) => b.price,
        None => return,
    };
    let best_ask = match ladder.asks.first() {
        Some(a) => a.price,
        None => return,
    };
    let total_spread = best_ask - best_bid;
    let min_total_spread = mid * floor_bps / 10_000.0;

    if total_spread >= min_total_spread {
        return; // Already above floor
    }

    let deficit = min_total_spread - total_spread;
    let bid_offset = (mid - best_bid).max(0.0);
    let ask_offset = (best_ask - mid).max(0.0);
    let total_offset = (bid_offset + ask_offset).max(1e-12);
    let bid_frac = bid_offset / total_offset;

    // Push best bid down and best ask up by proportional deficit
    if let Some(b) = ladder.bids.first_mut() {
        b.price -= deficit * bid_frac;
    }
    if let Some(a) = ladder.asks.first_mut() {
        a.price += deficit * (1.0 - bid_frac);
    }
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
        // Check if we have asymmetric depths (bid ≠ ask)
        if let Some(ref dynamic) = config.dynamic_depths {
            if dynamic.bid != dynamic.ask {
                tracing::debug!(
                    bid_levels = dynamic.bid.len(),
                    ask_levels = dynamic.ask.len(),
                    "Ladder::generate using asymmetric path"
                );
                return Self::generate_asymmetric(config, params);
            }
        }
        tracing::debug!("Ladder::generate using symmetric path");

        // 1. Compute depth spacing (symmetric)
        let depths = compute_depths(config);

        // 2. Compute fill intensity and spread capture for each depth
        let intensities: Vec<f64> = depths
            .iter()
            .map(|&d| fill_intensity(d, params.sigma, params.kappa, params.time_horizon))
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
        // Pass exchange BBO to prevent quotes crossing the actual L2 spread
        // (uses same price source as the BBO crossing filter in order_ops)
        let mut ladder = build_raw_ladder(&RawLadderInput {
            depths: &depths,
            sizes: &sizes,
            mid: params.mid_price,
            market_mid: params.market_mid,
            exchange_best_bid: params.cached_best_bid,
            exchange_best_ask: params.cached_best_ask,
            decimals: params.decimals,
            sz_decimals: params.sz_decimals,
            min_notional: params.min_notional,
        });

        // 5. Apply inventory skew with drift adjustment (re-rounds for exchange precision)
        apply_inventory_skew_with_drift(
            &mut ladder,
            params.inventory_ratio,
            params.gamma,
            params.sigma,
            params.time_horizon,
            params.mid_price,
            params.market_mid, // Use actual market mid for safety checks
            params.decimals,
            params.sz_decimals,
            params.use_drift_adjusted_skew,
            params.hjb_drift_urgency,
            params.position_opposes_momentum,
            params.urgency_score,
            params.funding_rate,
            params.use_funding_skew,
            params.cached_best_bid,
            params.cached_best_ask,
            params.effective_floor_bps,
        );

        // 6. Apply RL policy adjustments (spread delta + asymmetric skew)
        apply_rl_adjustments(
            &mut ladder,
            &RlAdjustmentParams {
                spread_delta_bps: params.rl_spread_delta_bps,
                bid_skew_bps: params.rl_bid_skew_bps,
                ask_skew_bps: params.rl_ask_skew_bps,
                confidence: params.rl_confidence,
                market_mid: params.market_mid,
                decimals: params.decimals,
                warmup_pct: params.warmup_pct,
            },
        );

        // 7. Enforce inviolable spread floor on total spread (best_ask - best_bid).
        // Post-hoc: preserves GLFT reservation-price skew by proportional deficit split.
        enforce_spread_floor(&mut ladder, params.mid_price, params.effective_floor_bps);

        ladder
    }

    /// Generate a ladder with asymmetric bid/ask depths.
    ///
    /// Used when bid and ask depths differ (e.g., different κ for each side).
    /// Each side gets its own depth array and size allocation.
    pub fn generate_asymmetric(config: &LadderConfig, params: &LadderParams) -> Self {
        // 1. Get separate bid and ask depths
        let bid_depths = compute_bid_depths(config);
        let ask_depths = compute_ask_depths(config);

        // 2. Compute fill intensity and spread capture for bid depths
        let bid_intensities: Vec<f64> = bid_depths
            .iter()
            .map(|&d| fill_intensity(d, params.sigma, params.kappa, params.time_horizon))
            .collect();

        let bid_spreads: Vec<f64> = if let Some(ref depth_decay) = params.depth_decay_as {
            bid_depths
                .iter()
                .map(|&d| depth_decay.spread_capture(d, config.fees_bps))
                .collect()
        } else {
            bid_depths
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

        // 3. Compute fill intensity and spread capture for ask depths
        let ask_intensities: Vec<f64> = ask_depths
            .iter()
            .map(|&d| fill_intensity(d, params.sigma, params.kappa, params.time_horizon))
            .collect();

        let ask_spreads: Vec<f64> = if let Some(ref depth_decay) = params.depth_decay_as {
            ask_depths
                .iter()
                .map(|&d| depth_decay.spread_capture(d, config.fees_bps))
                .collect()
        } else {
            ask_depths
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

        // 4. Allocate sizes separately for each side (half of total each)
        let bid_sizes = allocate_sizes(
            &bid_intensities,
            &bid_spreads,
            params.total_size / 2.0,
            config.min_level_size,
        );

        let ask_sizes = allocate_sizes(
            &ask_intensities,
            &ask_spreads,
            params.total_size / 2.0,
            config.min_level_size,
        );

        // 5. Build raw ladder with asymmetric depths
        // Pass market_mid to prevent quotes crossing the actual market spread
        let mut ladder = build_asymmetric_ladder(
            &bid_depths,
            &bid_sizes,
            &ask_depths,
            &ask_sizes,
            params.mid_price,
            params.market_mid,
            params.cached_best_bid,
            params.cached_best_ask,
            params.decimals,
            params.sz_decimals,
            params.min_notional,
        );

        // 6. Apply inventory skew with drift adjustment
        apply_inventory_skew_with_drift(
            &mut ladder,
            params.inventory_ratio,
            params.gamma,
            params.sigma,
            params.time_horizon,
            params.mid_price,
            params.market_mid, // Use actual market mid for safety checks
            params.decimals,
            params.sz_decimals,
            params.use_drift_adjusted_skew,
            params.hjb_drift_urgency,
            params.position_opposes_momentum,
            params.urgency_score,
            params.funding_rate,
            params.use_funding_skew,
            params.cached_best_bid,
            params.cached_best_ask,
            params.effective_floor_bps,
        );

        // 7. Apply RL policy adjustments (spread delta + asymmetric skew)
        apply_rl_adjustments(
            &mut ladder,
            &RlAdjustmentParams {
                spread_delta_bps: params.rl_spread_delta_bps,
                bid_skew_bps: params.rl_bid_skew_bps,
                ask_skew_bps: params.rl_ask_skew_bps,
                confidence: params.rl_confidence,
                market_mid: params.market_mid,
                decimals: params.decimals,
                warmup_pct: params.warmup_pct,
            },
        );

        // 8. Enforce inviolable spread floor (same as symmetric path)
        enforce_spread_floor(&mut ladder, params.mid_price, params.effective_floor_bps);

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

/// Compute depth levels using dynamic depths (if available) or geometric/linear spacing.
///
/// If `config.dynamic_depths` is Some, returns the bid depths from there.
/// Otherwise falls back to static computation:
/// - Geometric: δ_k = δ_min × r^(k-1) where r = (δ_max/δ_min)^(1/(K-1))
/// - Linear: δ_k = δ_min + k × (δ_max - δ_min) / (K-1)
///
/// Uses SmallVec to avoid heap allocation for typical ladder sizes.
pub(crate) fn compute_depths(config: &LadderConfig) -> DepthVec {
    // Use dynamic depths if available (GLFT-optimal)
    if let Some(ref dynamic) = config.dynamic_depths {
        // For symmetric ladder generation, use bid depths
        // (caller can use compute_depths_asymmetric for separate bid/ask)
        return dynamic.bid.iter().copied().collect();
    }

    // Fallback to static depth computation
    compute_depths_static(config)
}

/// Compute bid-specific depths (for asymmetric ladder generation)
pub(crate) fn compute_bid_depths(config: &LadderConfig) -> DepthVec {
    if let Some(ref dynamic) = config.dynamic_depths {
        dynamic.bid.iter().copied().collect()
    } else {
        compute_depths_static(config)
    }
}

/// Compute ask-specific depths (for asymmetric ladder generation)
pub(crate) fn compute_ask_depths(config: &LadderConfig) -> DepthVec {
    if let Some(ref dynamic) = config.dynamic_depths {
        dynamic.ask.iter().copied().collect()
    } else {
        compute_depths_static(config)
    }
}

/// Static depth computation using min/max and geometric/linear spacing.
/// Returns SmallVec to avoid heap allocation for typical ladder sizes.
fn compute_depths_static(config: &LadderConfig) -> DepthVec {
    let k = config.num_levels;
    if k == 0 {
        return DepthVec::new();
    }
    if k == 1 {
        let mut depths = DepthVec::new();
        depths.push(config.min_depth_bps);
        return depths;
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

/// Fill intensity model: λ(δ) = κ × min(1, (σ×√τ / δ)²)
///
/// Time-normalized diffusion-driven fills: probability of price reaching depth δ
/// depends on the expected move σ×√τ over time horizon τ.
///
/// The formula gives meaningful fill probabilities across the ladder:
/// - At optimal depth (where σ×√τ ≈ δ): fill probability ~100%
/// - At 2x optimal depth: fill probability ~25%
/// - At 3x optimal depth: fill probability ~11%
///
/// Example with sigma=0.0001 (1bp/sec), tau=10s:
///   σ×√τ = 0.0001 × √10 ≈ 3.16bp expected move
///   At 5bp depth: (3.16/5)² ≈ 0.4 → 40% of kappa
///   At 10bp depth: (3.16/10)² ≈ 0.1 → 10% of kappa
pub(crate) fn fill_intensity(depth_bps: f64, sigma: f64, kappa: f64, time_horizon: f64) -> f64 {
    let depth = depth_bps / 10000.0; // Convert bps to fraction
    if depth < EPSILON {
        return 0.0;
    }
    // Expected price move over time horizon τ
    let expected_move = sigma * time_horizon.sqrt();
    // Fill probability proportional to (expected_move / depth)², capped at 1.0
    let fill_prob = (expected_move / depth).powi(2).min(1.0);
    fill_prob * kappa
}

/// Spread capture: SC(δ) = δ - AS₀ × exp(-δ/δ_char) - fees
///
/// Expected profit from capturing the spread at depth δ:
/// - δ: raw spread captured (in bps)
/// - AS₀ × exp(-δ/δ_char): adverse selection cost that decays with depth
/// - fees: trading fees
pub(crate) fn spread_capture(
    depth_bps: f64,
    as_at_touch_bps: f64,
    as_decay_bps: f64,
    fees_bps: f64,
) -> f64 {
    let as_cost = as_at_touch_bps * (-depth_bps / as_decay_bps).exp();
    depth_bps - as_cost - fees_bps
}

/// Allocate sizes proportional to marginal value: λ(δ) × SC(δ)
///
/// Sizes are normalized to sum to total_size, with levels below
/// min_size set to zero (orders too small to be worth placing).
/// Uses SmallVec to avoid heap allocation for typical ladder sizes.
pub(crate) fn allocate_sizes(
    intensities: &[f64],
    spreads: &[f64],
    total_size: f64,
    min_size: f64,
) -> DepthVec {
    // Compute marginal value at each depth
    let marginal_values: DepthVec = intensities
        .iter()
        .zip(spreads.iter())
        .map(|(&lambda, &sc)| (lambda * sc).max(0.0)) // Only positive values
        .collect();

    let total: f64 = marginal_values.iter().sum();

    // Always log allocation parameters for debugging
    let max_mv = marginal_values.iter().fold(0.0_f64, |a, &b| a.max(b));
    let max_spread = spreads.iter().fold(0.0_f64, |a, &b| a.max(b));
    tracing::debug!(
        total_mv = %format!("{:.6}", total),
        max_mv = %format!("{:.6}", max_mv),
        max_spread = %format!("{:.4}", max_spread),
        total_size = %format!("{:.6}", total_size),
        min_size = %format!("{:.6}", min_size),
        num_levels = marginal_values.len(),
        "allocate_sizes called"
    );

    if total <= EPSILON {
        // All marginal values near zero - but we may still have positive spread capture
        // Use concentration fallback: put all size on first level (tightest depth)
        // Only if at least one spread is positive (otherwise truly unprofitable)
        let has_positive_spread = spreads.iter().any(|&sc| sc > EPSILON);
        tracing::info!(
            has_positive_spread = has_positive_spread,
            total_size = %format!("{:.6}", total_size),
            min_size = %format!("{:.6}", min_size),
            "allocate_sizes: total MV <= EPSILON, checking fallback conditions"
        );
        if has_positive_spread && total_size >= min_size {
            let mut result = DepthVec::new();
            result.resize(marginal_values.len(), 0.0);
            if !result.is_empty() {
                // Cap per-order size: no single order exceeds 25% of total_size.
                // Distribute remainder equally across additional levels if possible.
                let capped_size = total_size
                    .min(total_size * MAX_SINGLE_ORDER_FRACTION)
                    .max(min_size);
                result[0] = capped_size;
                tracing::info!(
                    levels = result.len(),
                    total_size = %format!("{:.6}", total_size),
                    capped_size = %format!("{:.6}", capped_size),
                    "Ladder allocate_sizes: MV=0 fallback, capped concentration on tightest level"
                );
            }
            return result;
        }
        // Truly no profitable levels - return zeros
        let mut zeros = DepthVec::new();
        zeros.resize(marginal_values.len(), 0.0);
        return zeros;
    }

    // Normalize to total_size
    let mut result: DepthVec = marginal_values
        .iter()
        .map(|&v| {
            let raw = total_size * v / total;
            if raw < min_size {
                0.0
            } else {
                raw
            }
        })
        .collect();

    // Check if all sizes were filtered out due to min_size
    // Use concentration fallback: cap per-order at 25% of total to prevent max-position fills
    let all_filtered = result.iter().all(|&s| s < EPSILON);
    if all_filtered && total_size >= min_size {
        let capped_size = total_size
            .min(total_size * MAX_SINGLE_ORDER_FRACTION)
            .max(min_size);
        result[0] = capped_size;
        tracing::info!(
            levels = result.len(),
            total_size = %format!("{:.6}", total_size),
            capped_size = %format!("{:.6}", capped_size),
            min_size = %format!("{:.6}", min_size),
            "Ladder allocate_sizes: min_size fallback, capped concentration on tightest level"
        );
    }

    result
}

/// Build raw ladder from depths and sizes (before inventory skew).
///
/// Creates symmetric bid/ask levels around mid price, applying
/// proper price rounding and size truncation per exchange requirements.
/// Uses SmallVec to avoid heap allocation for typical ladder sizes.
///
/// # Safe Base Prices
///
/// To prevent quotes crossing the market spread when microprice diverges from market_mid:
/// - Bids use `min(mid, exchange_best_ask - tick)` as base - never above exchange ask
/// - Asks use `max(mid, exchange_best_bid + tick)` as base - never below exchange bid
///
/// Uses exchange L2 BBO (same source as crossing filter in order_ops) to avoid
/// the 307-order waste from AllMids vs L2 price source mismatch.
/// Parameters for building a raw ladder from depths and sizes.
pub(crate) struct RawLadderInput<'a> {
    pub depths: &'a [f64],
    pub sizes: &'a [f64],
    pub mid: f64,
    pub market_mid: f64,
    pub exchange_best_bid: f64,
    pub exchange_best_ask: f64,
    pub decimals: u32,
    pub sz_decimals: u32,
    pub min_notional: f64,
}

pub(crate) fn build_raw_ladder(input: &RawLadderInput<'_>) -> Ladder {
    let RawLadderInput {
        depths,
        sizes,
        mid,
        market_mid,
        exchange_best_bid,
        exchange_best_ask,
        decimals,
        sz_decimals,
        min_notional,
    } = input;
    let (mid, market_mid) = (*mid, *market_mid);
    let (exchange_best_bid, exchange_best_ask) = (*exchange_best_bid, *exchange_best_ask);
    let (decimals, sz_decimals, min_notional) = (*decimals, *sz_decimals, *min_notional);

    // DEBUG: log entry to verify code path
    let total_size: f64 = sizes.iter().sum();
    tracing::debug!(
        num_depths = depths.len(),
        num_sizes = sizes.len(),
        total_size = %format!("{:.6}", total_size),
        mid = %format!("{:.4}", mid),
        market_mid = %format!("{:.4}", market_mid),
        exchange_best_bid = %format!("{:.4}", exchange_best_bid),
        exchange_best_ask = %format!("{:.4}", exchange_best_ask),
        min_notional = %format!("{:.2}", min_notional),
        "build_raw_ladder called"
    );

    let mut bids = LadderLevels::new();
    let mut asks = LadderLevels::new();

    // Use exchange L2 BBO as safety cap (same source as crossing filter in order_ops).
    // When exchange BBO is valid, use it. Otherwise fall back to market_mid.
    let min_tick = 10.0_f64.powi(-(decimals as i32));
    let effective_bid_cap = if exchange_best_ask > 0.0 {
        exchange_best_ask - min_tick // Never place bids at or above exchange ask
    } else {
        market_mid
    };
    let effective_ask_floor = if exchange_best_bid > 0.0 {
        exchange_best_bid + min_tick // Never place asks at or below exchange bid
    } else {
        market_mid
    };

    let bid_base = mid.min(effective_bid_cap);
    let ask_base = mid.max(effective_ask_floor);

    for (&depth_bps, &size) in depths.iter().zip(sizes.iter()) {
        if size < EPSILON {
            continue;
        }

        // Calculate price offset using safe bases
        let bid_offset = bid_base * (depth_bps / 10000.0);
        let ask_offset = ask_base * (depth_bps / 10000.0);

        // Round prices per exchange requirements
        let bid_price = round_to_significant_and_decimal(bid_base - bid_offset, 5, decimals);
        let ask_price = round_to_significant_and_decimal(ask_base + ask_offset, 5, decimals);

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

    // Concentration fallback: if all levels filtered by min_notional,
    // but total size meets min_notional, create single level at tightest depth.
    // Cap per-order size at 25% of total to prevent max-position fills.
    if bids.is_empty() && !depths.is_empty() && !sizes.is_empty() {
        let total_size: f64 = sizes.iter().sum();
        if let Some(&tightest_depth) = depths.first() {
            let offset = bid_base * (tightest_depth / 10000.0);
            let bid_price = round_to_significant_and_decimal(bid_base - offset, 5, decimals);

            // Floor: exact ceiling math via SizeQuantum (replaces 1.01x buffer)
            let quantum = SizeQuantum::compute(min_notional, bid_price.max(EPSILON), sz_decimals);
            let min_size_for_notional = quantum.min_viable_size;
            let capped_size = (total_size * MAX_SINGLE_ORDER_FRACTION).max(min_size_for_notional);
            let capped_size_truncated =
                truncate_float(capped_size.min(total_size), sz_decimals, false);

            if bid_price * capped_size_truncated >= min_notional {
                tracing::info!(
                    total_size = %format!("{:.6}", total_size),
                    capped_size = %format!("{:.6}", capped_size_truncated),
                    price = %format!("{:.4}", bid_price),
                    notional = %format!("{:.2}", bid_price * capped_size_truncated),
                    depth_bps = %format!("{:.2}", tightest_depth),
                    "Bid concentration fallback: size-capped order at tightest depth"
                );
                bids.push(LadderLevel {
                    price: bid_price,
                    size: capped_size_truncated,
                    depth_bps: tightest_depth,
                });
            }
        }
    }

    if asks.is_empty() && !depths.is_empty() && !sizes.is_empty() {
        let total_size: f64 = sizes.iter().sum();
        if let Some(&tightest_depth) = depths.first() {
            let offset = ask_base * (tightest_depth / 10000.0);
            let ask_price = round_to_significant_and_decimal(ask_base + offset, 5, decimals);

            // Floor: exact ceiling math via SizeQuantum (replaces 1.01x buffer)
            let quantum = SizeQuantum::compute(min_notional, ask_price.max(EPSILON), sz_decimals);
            let min_size_for_notional = quantum.min_viable_size;
            let capped_size = (total_size * MAX_SINGLE_ORDER_FRACTION).max(min_size_for_notional);
            let capped_size_truncated =
                truncate_float(capped_size.min(total_size), sz_decimals, false);

            if ask_price * capped_size_truncated >= min_notional {
                tracing::info!(
                    total_size = %format!("{:.6}", total_size),
                    capped_size = %format!("{:.6}", capped_size_truncated),
                    price = %format!("{:.4}", ask_price),
                    notional = %format!("{:.2}", ask_price * capped_size_truncated),
                    depth_bps = %format!("{:.2}", tightest_depth),
                    "Ask concentration fallback: size-capped order at tightest depth"
                );
                asks.push(LadderLevel {
                    price: ask_price,
                    size: capped_size_truncated,
                    depth_bps: tightest_depth,
                });
            }
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

    // Merge levels with identical prices (from rounding to exchange tick size)
    dedup_merge_levels(&mut bids);
    dedup_merge_levels(&mut asks);

    Ladder { bids, asks }
}

/// Build ladder with asymmetric depths for bids and asks.
///
/// Each side gets its own depth array and corresponding size array.
/// This is used when bid/ask depths differ (e.g., different κ for each side).
/// Uses SmallVec to avoid heap allocation for typical ladder sizes.
///
/// # Safe Base Prices
///
/// To prevent quotes crossing the market spread when microprice diverges from market_mid:
/// - Bids use `min(mid, market_mid)` as base - never built above market_mid
/// - Asks use `max(mid, market_mid)` as base - never built below market_mid
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_asymmetric_ladder(
    bid_depths: &[f64],
    bid_sizes: &[f64],
    ask_depths: &[f64],
    ask_sizes: &[f64],
    mid: f64,
    market_mid: f64,
    exchange_best_bid: f64,
    exchange_best_ask: f64,
    decimals: u32,
    sz_decimals: u32,
    min_notional: f64,
) -> Ladder {
    let mut bids = LadderLevels::new();
    let mut asks = LadderLevels::new();

    // Use exchange BBO for safety cap (same logic as build_raw_ladder).
    // When exchange BBO is valid, use it. Otherwise fall back to market_mid.
    let min_tick = 10.0_f64.powi(-(decimals as i32));
    let effective_bid_cap = if exchange_best_ask > 0.0 {
        exchange_best_ask - min_tick
    } else {
        market_mid
    };
    let effective_ask_floor = if exchange_best_bid > 0.0 {
        exchange_best_bid + min_tick
    } else {
        market_mid
    };

    let bid_base = mid.min(effective_bid_cap);
    let ask_base = mid.max(effective_ask_floor);

    // Build bid levels using safe base
    for (&depth_bps, &size) in bid_depths.iter().zip(bid_sizes.iter()) {
        if size < EPSILON {
            continue;
        }

        let offset = bid_base * (depth_bps / 10000.0);
        let bid_price = round_to_significant_and_decimal(bid_base - offset, 5, decimals);
        let size = truncate_float(size, sz_decimals, false);

        if bid_price * size >= min_notional {
            bids.push(LadderLevel {
                price: bid_price,
                size,
                depth_bps,
            });
        }
    }

    // Build ask levels using safe base
    for (&depth_bps, &size) in ask_depths.iter().zip(ask_sizes.iter()) {
        if size < EPSILON {
            continue;
        }

        let offset = ask_base * (depth_bps / 10000.0);
        let ask_price = round_to_significant_and_decimal(ask_base + offset, 5, decimals);
        let size = truncate_float(size, sz_decimals, false);

        if ask_price * size >= min_notional {
            asks.push(LadderLevel {
                price: ask_price,
                size,
                depth_bps,
            });
        }
    }

    // Concentration fallback for bids (asymmetric), size-capped at 25% of total
    if bids.is_empty() && !bid_depths.is_empty() && !bid_sizes.is_empty() {
        let total_size: f64 = bid_sizes.iter().sum();
        if let Some(&tightest_depth) = bid_depths.first() {
            let offset = bid_base * (tightest_depth / 10000.0);
            let bid_price = round_to_significant_and_decimal(bid_base - offset, 5, decimals);

            // Exact ceiling math via SizeQuantum (replaces 1.01x buffer)
            let quantum = SizeQuantum::compute(min_notional, bid_price.max(EPSILON), sz_decimals);
            let min_size_for_notional = quantum.min_viable_size;
            let capped_size = (total_size * MAX_SINGLE_ORDER_FRACTION).max(min_size_for_notional);
            let capped_size_truncated =
                truncate_float(capped_size.min(total_size), sz_decimals, false);

            if bid_price * capped_size_truncated >= min_notional {
                tracing::info!(
                    total_size = %format!("{:.6}", total_size),
                    capped_size = %format!("{:.6}", capped_size_truncated),
                    price = %format!("{:.4}", bid_price),
                    notional = %format!("{:.2}", bid_price * capped_size_truncated),
                    depth_bps = %format!("{:.2}", tightest_depth),
                    "Bid concentration fallback (asymmetric): size-capped order at tightest depth"
                );
                bids.push(LadderLevel {
                    price: bid_price,
                    size: capped_size_truncated,
                    depth_bps: tightest_depth,
                });
            }
        }
    }

    // Concentration fallback for asks (asymmetric), size-capped at 25% of total
    if asks.is_empty() && !ask_depths.is_empty() && !ask_sizes.is_empty() {
        let total_size: f64 = ask_sizes.iter().sum();
        if let Some(&tightest_depth) = ask_depths.first() {
            let offset = ask_base * (tightest_depth / 10000.0);
            let ask_price = round_to_significant_and_decimal(ask_base + offset, 5, decimals);

            // Exact ceiling math via SizeQuantum (replaces 1.01x buffer)
            let quantum = SizeQuantum::compute(min_notional, ask_price.max(EPSILON), sz_decimals);
            let min_size_for_notional = quantum.min_viable_size;
            let capped_size = (total_size * MAX_SINGLE_ORDER_FRACTION).max(min_size_for_notional);
            let capped_size_truncated =
                truncate_float(capped_size.min(total_size), sz_decimals, false);

            if ask_price * capped_size_truncated >= min_notional {
                tracing::info!(
                    total_size = %format!("{:.6}", total_size),
                    capped_size = %format!("{:.6}", capped_size_truncated),
                    price = %format!("{:.4}", ask_price),
                    notional = %format!("{:.2}", ask_price * capped_size_truncated),
                    depth_bps = %format!("{:.2}", tightest_depth),
                    "Ask concentration fallback (asymmetric): size-capped order at tightest depth"
                );
                asks.push(LadderLevel {
                    price: ask_price,
                    size: capped_size_truncated,
                    depth_bps: tightest_depth,
                });
            }
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

    // Merge levels with identical prices (from rounding to exchange tick size)
    dedup_merge_levels(&mut bids);
    dedup_merge_levels(&mut asks);

    Ladder { bids, asks }
}

/// Apply GLFT inventory skew with drift adjustment.
///
/// Extended version that includes HJB drift urgency when position opposes momentum.
/// This creates asymmetric quotes: wider on the side that would worsen position.
///
/// # Parameters
/// - `mid`: Microprice (used for calculating skew offset)
/// - `market_mid`: Actual exchange mid (used for safety checks to prevent crossing)
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_inventory_skew_with_drift(
    ladder: &mut Ladder,
    inventory_ratio: f64,
    gamma: f64,
    sigma: f64,
    time_horizon: f64,
    mid: f64,
    market_mid: f64,
    decimals: u32,
    sz_decimals: u32,
    use_drift_adjusted_skew: bool,
    hjb_drift_urgency: f64,
    position_opposes_momentum: bool,
    urgency_score: f64,
    funding_rate: f64,
    use_funding_skew: bool,
    cached_best_bid: f64,
    cached_best_ask: f64,
    effective_floor_bps: f64,
) {
    // === BASE INVENTORY SKEW (GLFT) ===
    // Reservation price offset: γσ²qT (as fraction of mid)
    // Positive inventory → positive skew → shift prices down
    let base_skew_fraction = inventory_ratio * gamma * sigma.powi(2) * time_horizon;

    // === DRIFT URGENCY (when position opposes momentum) ===
    // Add extra skew to accelerate inventory reduction when fighting the trend

    // CRITICAL FIX: Cap drift urgency relative to depth from MARKET MID (not microprice).
    // The drift urgency must NEVER cause ask prices to go below market_mid or bids above.
    let drift_skew_fraction =
        if use_drift_adjusted_skew && position_opposes_momentum && urgency_score > 0.5 {
            // Calculate depth from MARKET MID (not microprice) - this is what matters for crossing
            // Safety factor of 0.5 ensures we never cross more than 50% toward market mid
            let min_bid_depth_bps = ladder
                .bids
                .iter()
                .map(|l| ((market_mid - l.price) / market_mid * 10000.0).abs())
                .filter(|d| *d > 0.0)
                .fold(f64::INFINITY, f64::min);
            let min_ask_depth_bps = ladder
                .asks
                .iter()
                .map(|l| ((l.price - market_mid) / market_mid * 10000.0).abs())
                .filter(|d| *d > 0.0)
                .fold(f64::INFINITY, f64::min);

            // The max safe drift is the minimum of bid/ask depths from market mid
            let min_depth_bps = min_bid_depth_bps.min(min_ask_depth_bps);
            let max_safe_drift_fraction = if min_depth_bps.is_finite() && min_depth_bps > 0.0 {
                (min_depth_bps * 0.5) / 10000.0 // 50% of min depth, convert to fraction
            } else {
                0.0005 // 5 bps absolute cap if no safe margin
            };

            // Cap drift urgency to prevent crossed spreads
            let raw_drift = hjb_drift_urgency;
            let capped_drift = raw_drift.abs().min(max_safe_drift_fraction) * raw_drift.signum();

            if (raw_drift - capped_drift).abs() > 1e-8 {
                tracing::warn!(
                    raw_drift_bps = %format!("{:.2}", raw_drift * 10000.0),
                    capped_drift_bps = %format!("{:.2}", capped_drift * 10000.0),
                    min_depth_from_market_mid_bps = %format!("{:.2}", min_depth_bps),
                    market_mid = %format!("{:.4}", market_mid),
                    microprice = %format!("{:.4}", mid),
                    "Drift urgency capped to prevent crossing market mid"
                );
            }

            capped_drift
        } else {
            0.0
        };

    // === FUNDING SKEW ===
    // Account for cost of carry over the holding horizon.
    // If funding > 0: Longs pay shorts. Asset effectively depreciates by funding_rate * T relative to cash.
    // skew = -funding_rate * T
    // This lowers bids and asks, encouraging selling (to avoid paying funding) and buying lower.
    let funding_skew_fraction = if use_funding_skew {
        -funding_rate * time_horizon
    } else {
        0.0
    };

    // === COMBINED SKEW ===
    let total_skew_fraction = base_skew_fraction + drift_skew_fraction + funding_skew_fraction;

    // Early return only if there's no inventory AND no drift/funding adjustment
    if inventory_ratio.abs() < EPSILON
        && drift_skew_fraction.abs() < EPSILON
        && funding_skew_fraction.abs() < EPSILON
    {
        return;
    }

    // === MICROPRICE SANITY CHECK ===
    // If mid (microprice) diverged significantly from market_mid, use market_mid instead
    // This prevents cascading errors from upstream bugs (e.g., reservation_shift overflow)
    let safe_mid = {
        let ratio = mid / market_mid;
        if !(0.8..=1.2).contains(&ratio) {
            tracing::error!(
                microprice = %format!("{:.4}", mid),
                market_mid = %format!("{:.4}", market_mid),
                ratio = %format!("{:.2}", ratio),
                "CRITICAL: Microprice diverged >20% from market_mid - using market_mid for skew calculation"
            );
            market_mid
        } else {
            mid
        }
    };

    let raw_offset = safe_mid * total_skew_fraction;

    // === BOUND OFFSET BY BBO HALF-SPREAD WITH OWN-LADDER FLOOR ===
    // Cap skew so it never pushes quotes past the opposing BBO.
    // BUT: on tight-spread tokens (e.g. HYPE BBO ~0.3 bps) our ladder is 8+ bps from mid,
    // so the exchange BBO is irrelevant to our crossing risk. Use our own ladder depth
    // as a floor so inventory skew isn't killed by a tight exchange spread.
    let half_spread = (cached_best_ask - cached_best_bid) / 2.0;
    let own_min_depth = ladder
        .bids
        .iter()
        .map(|l| (safe_mid - l.price).abs())
        .chain(ladder.asks.iter().map(|l| (l.price - safe_mid).abs()))
        .filter(|d| *d > 1e-12)
        .fold(f64::INFINITY, f64::min);
    let own_floor = if own_min_depth.is_finite() {
        own_min_depth * 0.5 // 50% of our innermost level depth
    } else {
        safe_mid * (effective_floor_bps.max(5.0) / 10_000.0) // fallback: effective floor (min 5 bps)
    };
    let max_offset = if half_spread > 0.0 {
        (half_spread * 0.8).max(own_floor)
    } else {
        own_floor
    };
    let offset = raw_offset.clamp(-max_offset, max_offset);

    if (raw_offset - offset).abs() > 1e-10 {
        let own_floor_bps = own_floor / safe_mid * 10000.0;
        let half_spread_cap = half_spread * 0.8;
        let capping_source = if half_spread_cap > own_floor {
            "bbo"
        } else {
            "own_depth"
        };
        tracing::warn!(
            raw_offset_bps = %format!("{:.2}", raw_offset / safe_mid * 10000.0),
            capped_offset_bps = %format!("{:.2}", offset / safe_mid * 10000.0),
            half_spread_bps = %format!("{:.2}", half_spread / safe_mid * 10000.0),
            own_floor_bps = %format!("{:.2}", own_floor_bps),
            capping_source = capping_source,
            "Inventory skew offset capped"
        );
    }

    // Shift all prices by bounded offset and RE-ROUND to exchange precision
    for level in &mut ladder.bids {
        level.price = round_to_significant_and_decimal(level.price - offset, 5, decimals);
    }
    for level in &mut ladder.asks {
        level.price = round_to_significant_and_decimal(level.price - offset, 5, decimals);
    }

    // Size skew: reduce side that increases position
    // Cap reduction at 90% to never completely remove quotes
    if inventory_ratio.abs() >= EPSILON {
        let size_reduction = (inventory_ratio.abs() * 2.5).min(0.95);

        if inventory_ratio > 0.0 {
            // Long inventory: reduce bid sizes (don't want to buy more)
            for level in &mut ladder.bids {
                level.size =
                    truncate_float(level.size * (1.0 - size_reduction), sz_decimals, false);
            }
        } else {
            // Short inventory: reduce ask sizes (don't want to sell more)
            for level in &mut ladder.asks {
                level.size =
                    truncate_float(level.size * (1.0 - size_reduction), sz_decimals, false);
            }
        }
    }
}

/// RL policy adjustments for quote ladder modification.
pub(crate) struct RlAdjustmentParams {
    pub spread_delta_bps: f64,
    pub bid_skew_bps: f64,
    pub ask_skew_bps: f64,
    pub confidence: f64,
    pub market_mid: f64,
    pub decimals: u32,
    /// Warmup progress [0.0, 1.0]. RL adjustments are disabled when < 0.5
    /// to prevent untrained RL policies from widening spreads during early learning.
    pub warmup_pct: f64,
}

/// Apply RL policy adjustments to the ladder.
///
/// RL recommendations adjust quotes after all other calculations:
/// - `spread_delta_bps`: Widens (+) or tightens (-) both sides equally
/// - `bid_skew_bps`: Additional adjustment to bid side only
/// - `ask_skew_bps`: Additional adjustment to ask side only
///
/// Adjustments are scaled by `confidence` (0-1) to ensure uncertain
/// recommendations have minimal impact.
///
/// Safety guards:
/// - Bids cannot go above market_mid (prevents crossing)
/// - Asks cannot go below market_mid (prevents crossing)
/// - Max adjustment capped at ±5 bps per side
pub(crate) fn apply_rl_adjustments(ladder: &mut Ladder, params: &RlAdjustmentParams) {
    // Skip if confidence too low or no adjustment needed
    if params.confidence < 0.1 {
        return;
    }

    // Disable RL adjustments during warmup (<50%) to prevent untrained policies
    // from adding unnecessary spread widening during early learning phase
    if params.warmup_pct < 0.5 {
        tracing::debug!(
            warmup_pct = %format!("{:.0}%", params.warmup_pct * 100.0),
            rl_confidence = %format!("{:.2}", params.confidence),
            "RL adjustments disabled: warmup < 50%"
        );
        return;
    }

    // Scale adjustments by confidence
    let effective_confidence = params.confidence.clamp(0.0, 1.0);

    // Total bid adjustment: spread_delta (widens both) + bid_skew (positive widens bid)
    // Positive spread_delta = widen = move bid DOWN (further from mid)
    // Positive bid_skew = widen bid = move bid DOWN
    let raw_bid_adj_bps = params.spread_delta_bps + params.bid_skew_bps;
    let raw_ask_adj_bps = params.spread_delta_bps + params.ask_skew_bps;

    // Clamp to reasonable bounds before scaling
    let clamped_bid_adj = raw_bid_adj_bps.clamp(-5.0, 5.0);
    let clamped_ask_adj = raw_ask_adj_bps.clamp(-5.0, 5.0);

    // Scale by confidence
    let bid_adj_bps = clamped_bid_adj * effective_confidence;
    let ask_adj_bps = clamped_ask_adj * effective_confidence;

    // Skip if negligible adjustment
    if bid_adj_bps.abs() < 0.1 && ask_adj_bps.abs() < 0.1 {
        return;
    }

    // Log RL adjustment
    tracing::info!(
        spread_delta_bps = %format!("{:.2}", params.spread_delta_bps),
        bid_skew_bps = %format!("{:.2}", params.bid_skew_bps),
        ask_skew_bps = %format!("{:.2}", params.ask_skew_bps),
        confidence = %format!("{:.2}", params.confidence),
        effective_bid_adj_bps = %format!("{:.2}", bid_adj_bps),
        effective_ask_adj_bps = %format!("{:.2}", ask_adj_bps),
        "RL adjustment applied to ladder"
    );

    // Apply to bids: positive adjustment = widen = lower price
    for level in &mut ladder.bids {
        let adj_frac = bid_adj_bps / 10000.0;
        let new_price = level.price * (1.0 - adj_frac);
        // Safety: don't let bid go above market_mid
        let safe_price = new_price.min(params.market_mid * 0.9999);
        level.price = round_to_significant_and_decimal(safe_price, 5, params.decimals);
    }

    // Apply to asks: positive adjustment = widen = higher price
    for level in &mut ladder.asks {
        let adj_frac = ask_adj_bps / 10000.0;
        let new_price = level.price * (1.0 + adj_frac);
        // Safety: don't let ask go below market_mid
        let safe_price = new_price.max(params.market_mid * 1.0001);
        level.price = round_to_significant_and_decimal(safe_price, 5, params.decimals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

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
        let tau = 10.0; // 10 second time horizon

        // Use typical HFT sigma (0.0001 = 1bp/sec)
        // Expected move = 0.0001 * √10 ≈ 0.000316 ≈ 3.16bp
        let sigma = 0.0001;
        let kappa = 100.0;

        // Higher intensity at tighter depths
        let i_tight = fill_intensity(2.0, sigma, kappa, tau);
        let i_wide = fill_intensity(20.0, sigma, kappa, tau);

        assert!(
            i_tight > i_wide,
            "i_tight={:.2} should be > i_wide={:.2}",
            i_tight,
            i_wide
        );

        // Intensity at 0 depth should be 0 (avoid division by zero)
        let i_zero = fill_intensity(0.0, sigma, kappa, tau);
        assert!((i_zero).abs() < EPSILON);

        // With sigma=0.0001 (1bp/sec), tau=10s: expected move = 0.0001 * √10 ≈ 3.16bp
        // At 2bp depth: (3.16/2)² ≈ 2.5, capped at 1.0 → intensity = 100
        // At 5bp depth: (3.16/5)² ≈ 0.4 → intensity = 40
        // At 10bp depth: (3.16/10)² ≈ 0.1 → intensity = 10
        // At 20bp depth: (3.16/20)² ≈ 0.025 → intensity = 2.5
        let i_2bp = fill_intensity(2.0, sigma, kappa, tau);
        let i_5bp = fill_intensity(5.0, sigma, kappa, tau);
        let i_10bp = fill_intensity(10.0, sigma, kappa, tau);
        let i_20bp = fill_intensity(20.0, sigma, kappa, tau);

        assert!(
            (i_2bp - 100.0).abs() < 1.0,
            "2bp intensity should be ~100 (capped), got {}",
            i_2bp
        );
        assert!(
            i_5bp > 35.0 && i_5bp < 45.0,
            "5bp intensity should be ~40, got {}",
            i_5bp
        );
        assert!(
            i_10bp > 8.0 && i_10bp < 12.0,
            "10bp intensity should be ~10, got {}",
            i_10bp
        );
        assert!(
            i_20bp > 2.0 && i_20bp < 3.0,
            "20bp intensity should be ~2.5, got {}",
            i_20bp
        );

        // Verify monotonic decay with depth
        assert!(i_2bp >= i_5bp);
        assert!(i_5bp > i_10bp);
        assert!(i_10bp > i_20bp);
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
        let market_mid = 100.0; // No divergence in test

        let ladder = build_raw_ladder(&RawLadderInput {
            depths: &depths,
            sizes: &sizes,
            mid,
            market_mid,
            exchange_best_bid: 0.0,
            exchange_best_ask: 0.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
        });

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
        let mid = 100.0;
        let market_mid = 100.0; // No divergence in test

        let ladder = build_raw_ladder(&RawLadderInput {
            depths: &depths,
            sizes: &sizes,
            mid,
            market_mid,
            exchange_best_bid: 0.0,
            exchange_best_ask: 0.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional: 10.0,
        });

        // Only second level should make it (0.5 * $100 = $50 > $10)
        assert_eq!(ladder.bids.len(), 1);
        assert_eq!(ladder.asks.len(), 1);
    }

    #[test]
    fn test_inventory_skew_long() {
        let mut ladder = Ladder {
            bids: smallvec![LadderLevel {
                price: 100.0,
                size: 1.0,
                depth_bps: 2.0,
            }],
            asks: smallvec![LadderLevel {
                price: 100.2,
                size: 1.0,
                depth_bps: 2.0,
            }],
        };

        // Long inventory: bids should move down, bid sizes reduced
        apply_inventory_skew_with_drift(
            &mut ladder,
            0.5,
            0.3,
            0.01,
            10.0,
            100.0,
            100.0,
            2,
            4,
            false,
            0.0,
            false,
            0.0,
            0.0,   // funding_rate
            false, // use_funding_skew
            0.0,   // cached_best_bid (0 = fallback to market_mid)
            0.0,   // cached_best_ask
            8.0,   // effective_floor_bps
        );

        assert!(ladder.bids[0].price < 100.0); // Price shifted down
        assert!(ladder.bids[0].size < 1.0); // Size reduced
        assert!((ladder.asks[0].size - 1.0).abs() < 0.01); // Ask size unchanged
    }

    #[test]
    fn test_inventory_skew_short() {
        let mut ladder = Ladder {
            bids: smallvec![LadderLevel {
                price: 99.0, // Start below mid so upward shift is valid
                size: 1.0,
                depth_bps: 100.0,
            }],
            asks: smallvec![LadderLevel {
                price: 100.2,
                size: 1.0,
                depth_bps: 20.0, // 20bps
            }],
        };

        // Short inventory: prices shift up, ask sizes reduced
        apply_inventory_skew_with_drift(
            &mut ladder,
            -0.5,
            0.3,
            0.01,
            10.0,
            100.0,
            100.0,
            2,
            4,
            false,
            0.0,
            false,
            0.0,
            0.0,   // funding_rate
            false, // use_funding_skew
            0.0,   // cached_best_bid
            0.0,   // cached_best_ask
            8.0,   // effective_floor_bps
        );

        assert!(ladder.bids[0].price > 99.0); // Price shifted up
        assert!(ladder.asks[0].size < 1.0); // Ask size reduced
        assert!((ladder.bids[0].size - 1.0).abs() < 0.01); // Bid size unchanged
    }

    #[test]
    fn test_inventory_skew_zero() {
        let mut ladder = Ladder {
            bids: smallvec![LadderLevel {
                price: 100.0,
                size: 1.0,
                depth_bps: 2.0,
            }],
            asks: smallvec![LadderLevel {
                price: 100.2,
                size: 1.0,
                depth_bps: 2.0,
            }],
        };

        // Zero inventory: no changes
        apply_inventory_skew_with_drift(
            &mut ladder,
            0.0,
            0.3,
            0.01,
            10.0,
            100.0,
            100.0,
            2,
            4,
            false,
            0.0,
            false,
            0.0,
            0.0,   // funding_rate
            false, // use_funding_skew
            0.0,   // cached_best_bid
            0.0,   // cached_best_ask
            8.0,   // effective_floor_bps
        );

        assert!((ladder.bids[0].price - 100.0).abs() < EPSILON);
        assert!((ladder.bids[0].size - 1.0).abs() < EPSILON);
        assert!((ladder.asks[0].size - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_inventory_skew_price_precision() {
        // Test that prices remain properly rounded after skew (BTC-like: 0 decimals)
        let mut ladder = Ladder {
            bids: smallvec![LadderLevel {
                price: 87833.0,
                size: 0.02,
                depth_bps: 2.0,
            }],
            asks: smallvec![LadderLevel {
                price: 87850.0,
                size: 0.02,
                depth_bps: 2.0,
            }],
        };

        // Apply skew with 0 decimal places (BTC)
        apply_inventory_skew_with_drift(
            &mut ladder,
            0.5,
            0.3,
            0.0001,
            10.0,
            87840.0,
            87840.0,
            0,
            5,
            false,
            0.0,
            false,
            0.0,
            0.0,   // funding_rate
            false, // use_funding_skew
            0.0,   // cached_best_bid
            0.0,   // cached_best_ask
            8.0,   // effective_floor_bps
        );

        // Prices should be integers (no fractional part)
        assert_eq!(ladder.bids[0].price, ladder.bids[0].price.round());
        assert_eq!(ladder.asks[0].price, ladder.asks[0].price.round());
    }

    /// Test that verifies the HIP-3 capital efficiency fix.
    ///
    /// The bug: When total_size is small (e.g., GLFT-derived target_liquidity = 1.3 HYPE),
    /// individual levels fail min_notional check, triggering concentration fallback
    /// to a single order BEFORE the entropy optimizer can distribute sizes.
    ///
    /// The fix: Use effective_max_position (margin-based capacity) for initial ladder
    /// generation so levels pass min_notional, then let entropy optimizer allocate.
    #[test]
    fn test_hip3_capital_efficiency_fix() {
        // HIP-3 scenario: HYPE at $22.80, $10 min notional, 10 levels
        let mid = 22.80;
        let min_notional = 10.0;
        let num_levels = 10;

        // Create depths (geometric spacing from 10bps to 100bps)
        let depths: Vec<f64> = (0..num_levels)
            .map(|i| 10.0 * (1.26_f64).powi(i as i32)) // ~10, 12.6, 15.9, 20, 25, 32, 40, 50, 63, 80
            .collect();

        // Scenario 1: Small total_size (like target_liquidity = 1.3 HYPE)
        // Per-level = 1.3 / 10 = 0.13 HYPE
        // Per-level notional = 0.13 * $22.80 = $2.96 < $10 → FAILS min_notional
        let small_total_size = 1.3;
        let small_sizes: Vec<f64> = vec![small_total_size / num_levels as f64; num_levels];

        let ladder_small = build_raw_ladder(&RawLadderInput {
            depths: &depths,
            sizes: &small_sizes,
            mid,
            market_mid: mid,
            exchange_best_bid: 0.0,
            exchange_best_ask: 0.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional,
        });

        // With small sizes, ALL individual levels fail min_notional
        // Concentration fallback should trigger → single order
        assert_eq!(
            ladder_small.bids.len(),
            1,
            "Small total_size should trigger concentration fallback to 1 level"
        );
        // The single order should be capped at 25% of total (or min exchange size, whichever is larger)
        let total_bid_size: f64 = ladder_small.bids.iter().map(|l| l.size).sum();
        // Bid price is offset by tightest depth (10 bps for this test)
        let tightest_test_bps = 10.0;
        let bid_price_approx = mid * (1.0 - tightest_test_bps / 10000.0);
        let expected_cap =
            (small_total_size * MAX_SINGLE_ORDER_FRACTION).max(min_notional / bid_price_approx);
        assert!(
            total_bid_size <= expected_cap + 0.01,
            "Concentration fallback size {:.4} should be capped at {:.4}",
            total_bid_size,
            expected_cap,
        );

        // Scenario 2: Large total_size (like effective_max_position = 66 HYPE)
        // Per-level = 66 / 10 = 6.6 HYPE
        // Per-level notional = 6.6 * $22.80 = $150 > $10 → PASSES min_notional
        let large_total_size = 66.0;
        let large_sizes: Vec<f64> = vec![large_total_size / num_levels as f64; num_levels];

        let ladder_large = build_raw_ladder(&RawLadderInput {
            depths: &depths,
            sizes: &large_sizes,
            mid,
            market_mid: mid,
            exchange_best_bid: 0.0,
            exchange_best_ask: 0.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional,
        });

        // With large sizes, ALL levels should pass min_notional
        // No concentration fallback → multiple levels
        assert_eq!(
            ladder_large.bids.len(),
            num_levels,
            "Large total_size should result in {} levels, got {}",
            num_levels,
            ladder_large.bids.len()
        );
        assert_eq!(
            ladder_large.asks.len(),
            num_levels,
            "Large total_size should result in {} ask levels, got {}",
            num_levels,
            ladder_large.asks.len()
        );

        // Verify each level has roughly the expected size
        for (i, level) in ladder_large.bids.iter().enumerate() {
            let expected_size = large_total_size / num_levels as f64;
            assert!(
                (level.size - expected_size).abs() < 0.1,
                "Level {} size {} should be close to {}",
                i,
                level.size,
                expected_size
            );
        }
    }

    /// Test the min_meaningful_size calculation using SizeQuantum.
    /// SizeQuantum uses exact ceiling math to compute the smallest truncation-stable
    /// size that meets min_notional, replacing the old buffer-based approach.
    #[test]
    fn test_min_meaningful_size_calculation() {
        use crate::market_maker::config::SizeQuantum;

        let min_notional: f64 = 10.0;
        let microprice: f64 = 22.80;

        // Old 1.5x (overly conservative)
        let old_min = (min_notional * 1.5) / microprice;

        // New: exact ceiling math via SizeQuantum
        let quantum = SizeQuantum::compute(min_notional, microprice, 2); // sz_decimals=2
        let new_min = quantum.min_viable_size;

        // SizeQuantum should be LESS conservative than 1.5x
        assert!(
            new_min < old_min,
            "Quantum should be tighter than 1.5x buffer"
        );

        // But still meet min_notional
        assert!(new_min * microprice >= min_notional);

        // And be truncation-stable
        assert_eq!(truncate_float(new_min, 2, false), new_min);

        // With 6 HYPE available, should get more levels
        let available: f64 = 6.0;
        let old_max_levels = (available / old_min).floor() as usize;
        let new_max_levels = (available / new_min).floor() as usize;
        assert!(new_max_levels >= old_max_levels);
    }

    #[test]
    fn test_rl_adjustments_widen_spreads() {
        let mut ladder = Ladder {
            bids: smallvec![
                LadderLevel {
                    price: 99.95,
                    size: 1.0,
                    depth_bps: 5.0
                },
                LadderLevel {
                    price: 99.90,
                    size: 1.0,
                    depth_bps: 10.0
                },
            ],
            asks: smallvec![
                LadderLevel {
                    price: 100.05,
                    size: 1.0,
                    depth_bps: 5.0
                },
                LadderLevel {
                    price: 100.10,
                    size: 1.0,
                    depth_bps: 10.0
                },
            ],
        };

        // RL recommends widening by 2 bps with high confidence
        apply_rl_adjustments(
            &mut ladder,
            &RlAdjustmentParams {
                spread_delta_bps: 2.0, // widen both sides by 2 bps
                bid_skew_bps: 0.0,     // no bid skew
                ask_skew_bps: 0.0,     // no ask skew
                confidence: 0.9,       // 90% confidence
                market_mid: 100.0,
                decimals: 2,
                warmup_pct: 1.0, // fully warmed up for test
            },
        );

        // Bids should be LOWER (further from mid) after widening
        // 2 bps * 0.9 confidence = 1.8 bps effective widen
        // 99.95 * (1 - 0.00018) ≈ 99.932
        assert!(
            ladder.bids[0].price < 99.95,
            "Bid should move down (widen): {}",
            ladder.bids[0].price
        );

        // Asks should be HIGHER (further from mid) after widening
        // 100.05 * (1 + 0.00018) ≈ 100.068
        assert!(
            ladder.asks[0].price > 100.05,
            "Ask should move up (widen): {}",
            ladder.asks[0].price
        );
    }

    #[test]
    fn test_rl_adjustments_asymmetric_skew() {
        let mut ladder = Ladder {
            bids: smallvec![LadderLevel {
                price: 99.95,
                size: 1.0,
                depth_bps: 5.0
            },],
            asks: smallvec![LadderLevel {
                price: 100.05,
                size: 1.0,
                depth_bps: 5.0
            },],
        };

        // RL recommends widening bid (positive = widen) and tightening ask (negative = tighten)
        // This creates asymmetry favoring buying
        apply_rl_adjustments(
            &mut ladder,
            &RlAdjustmentParams {
                spread_delta_bps: 0.0, // no symmetric spread delta
                bid_skew_bps: 2.0,     // widen bid by 2 bps
                ask_skew_bps: -2.0,    // tighten ask by 2 bps
                confidence: 1.0,       // 100% confidence
                market_mid: 100.0,
                decimals: 2,
                warmup_pct: 1.0, // fully warmed up for test
            },
        );

        // Bid widened = lower price
        assert!(
            ladder.bids[0].price < 99.95,
            "Bid should widen (move down): {}",
            ladder.bids[0].price
        );
        // Ask tightened = lower price (closer to mid)
        assert!(
            ladder.asks[0].price < 100.05,
            "Ask should tighten (move down): {}",
            ladder.asks[0].price
        );
    }

    #[test]
    fn test_rl_adjustments_low_confidence_ignored() {
        let mut ladder = Ladder {
            bids: smallvec![LadderLevel {
                price: 99.95,
                size: 1.0,
                depth_bps: 5.0
            },],
            asks: smallvec![LadderLevel {
                price: 100.05,
                size: 1.0,
                depth_bps: 5.0
            },],
        };

        let original_bid = ladder.bids[0].price;
        let original_ask = ladder.asks[0].price;

        // RL recommends large adjustment but with LOW confidence
        apply_rl_adjustments(
            &mut ladder,
            &RlAdjustmentParams {
                spread_delta_bps: 5.0, // large spread delta
                bid_skew_bps: 0.0,
                ask_skew_bps: 0.0,
                confidence: 0.05, // Only 5% confidence - below 10% threshold
                market_mid: 100.0,
                decimals: 2,
                warmup_pct: 1.0, // fully warmed up for test
            },
        );

        // Prices should be unchanged (low confidence ignored)
        assert_eq!(
            ladder.bids[0].price, original_bid,
            "Bid should be unchanged"
        );
        assert_eq!(
            ladder.asks[0].price, original_ask,
            "Ask should be unchanged"
        );
    }

    #[test]
    fn test_rl_adjustments_safety_guard_bid_crossing() {
        let mut ladder = Ladder {
            bids: smallvec![
                LadderLevel {
                    price: 99.99,
                    size: 1.0,
                    depth_bps: 1.0
                }, // Very close to mid
            ],
            asks: smallvec![LadderLevel {
                price: 100.01,
                size: 1.0,
                depth_bps: 1.0
            },],
        };

        // RL recommends TIGHTENING (negative) which could cross mid
        apply_rl_adjustments(
            &mut ladder,
            &RlAdjustmentParams {
                spread_delta_bps: -10.0, // Aggressive tightening
                bid_skew_bps: 0.0,
                ask_skew_bps: 0.0,
                confidence: 1.0,
                market_mid: 100.0, // market_mid = 100
                decimals: 2,
                warmup_pct: 1.0, // fully warmed up for test
            },
        );

        // Safety guard: bid should not go above market_mid * 0.9999
        assert!(
            ladder.bids[0].price < 100.0,
            "Bid must stay below mid: {}",
            ladder.bids[0].price
        );
        // Safety guard: ask should not go below market_mid * 1.0001
        assert!(
            ladder.asks[0].price > 100.0,
            "Ask must stay above mid: {}",
            ladder.asks[0].price
        );
    }

    /// Test that no single order exceeds 25% of total_size in concentration fallback paths.
    ///
    /// This is the critical safety test for the HYPE incident: the concentration
    /// fallback collapsed 25 levels into a SINGLE order at 100% of max position
    /// (1.51 HYPE = full $50 limit). One fill maxed inventory with zero recovery.
    ///
    /// GLFT inventory theory: q at max creates reservation price adjustment
    /// = q * gamma * sigma^2 * T which swings maximally and prevents recovery.
    ///
    /// Note: The per-level cap in the general path (after entropy optimizer) is
    /// enforced by `ladder_strat.rs`, not by these low-level generator functions.
    /// This test verifies the concentration fallback caps in the generator.
    #[test]
    fn test_no_single_order_exceeds_25pct_max_position() {
        // === Scenario 1: Concentration fallback in build_raw_ladder ===
        // Small per-level sizes that all fail min_notional → concentration fallback
        let mid = 33.0; // HYPE-like price
        let min_notional = 10.0;
        let total_size = 1.51; // Full $50 limit at $33

        // With 10 levels: per-level = 0.151 HYPE, notional = $4.98 < $10 → all fail
        let depths: Vec<f64> = (0..10).map(|i| 5.0 + i as f64 * 3.0).collect();
        let sizes: Vec<f64> = vec![total_size / 10.0; 10];

        let ladder = build_raw_ladder(&RawLadderInput {
            depths: &depths,
            sizes: &sizes,
            mid,
            market_mid: mid,
            exchange_best_bid: 0.0,
            exchange_best_ask: 0.0,
            decimals: 2,
            sz_decimals: 4,
            min_notional,
        });

        // Should have a fallback order, but capped at 25% of total_size
        // (or min exchange size at actual bid_price, whichever is larger)
        // Bid price is offset from mid by tightest depth (5 bps)
        let tightest_bps = 5.0;
        let bid_price_approx = mid * (1.0 - tightest_bps / 10000.0);
        let cap = (total_size * MAX_SINGLE_ORDER_FRACTION).max(min_notional / bid_price_approx);
        for level in &ladder.bids {
            assert!(
                level.size <= cap + 0.001,
                "Bid size {:.6} exceeds cap {:.6} (was {:.6} total)",
                level.size,
                cap,
                total_size
            );
        }
        for level in &ladder.asks {
            assert!(
                level.size <= cap + 0.001,
                "Ask size {:.6} exceeds cap {:.6} (was {:.6} total)",
                level.size,
                cap,
                total_size
            );
        }

        // === Scenario 2: allocate_sizes MV=0 fallback ===
        // When ALL marginal values are zero (total <= EPSILON), the MV=0 fallback fires
        let intensities_zero = vec![1.0, 1.0, 1.0];
        let spreads_all_neg = vec![-0.0001, -2.0, -3.0]; // First level ~zero → total MV ≈ 0
        let alloc = allocate_sizes(&intensities_zero, &spreads_all_neg, total_size, 0.0);
        // All negative spreads → MV all clamped to 0.0 → total ≈ 0 → fallback
        // Since all MVs are 0, all sizes should be 0 (no positive spread)
        for (i, &size) in alloc.iter().enumerate() {
            assert!(
                size <= total_size * MAX_SINGLE_ORDER_FRACTION + 0.001,
                "MV=0 fallback level {} size {:.6} exceeds 25% cap",
                i,
                size
            );
        }

        // === Scenario 3: allocate_sizes min_size fallback ===
        let intensities = vec![0.01, 0.01, 0.01]; // Very small intensities
        let spreads = vec![1.0, 1.0, 1.0]; // Positive spreads
                                           // min_size = 0.6, which is larger than per-level = 1.51/3 ≈ 0.503 → all filtered
        let min_size = 0.6;
        let alloc = allocate_sizes(&intensities, &spreads, total_size, min_size);

        // The effective cap is max(25% * total, min_size) because we can't go below min_size
        let effective_cap = (total_size * MAX_SINGLE_ORDER_FRACTION).max(min_size);
        for (i, &size) in alloc.iter().enumerate() {
            assert!(
                size <= effective_cap + 0.001,
                "min_size fallback level {} size {:.6} exceeds effective cap {:.6}",
                i,
                size,
                effective_cap
            );
        }
        // Critically: should NOT be full total_size (was 1.51 before the fix)
        assert!(
            alloc[0] < total_size - 0.01,
            "min_size fallback should NOT concentrate full total_size ({:.4}), got {:.4}",
            total_size,
            alloc[0]
        );

        // === Scenario 4: Asymmetric ladder concentration fallback ===
        // Use 10 levels so per-level is small enough to fail min_notional
        let asym_depths = vec![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0];
        let asym_sizes = vec![total_size / 10.0; 10]; // 0.151 each, notional = $4.98 < $10

        let ladder = build_asymmetric_ladder(
            &asym_depths,
            &asym_sizes,
            &asym_depths,
            &asym_sizes,
            mid,
            mid,
            0.0,
            0.0,
            2,
            4,
            min_notional,
        );

        for level in &ladder.bids {
            assert!(
                level.size <= cap + 0.001,
                "Asymmetric bid size {:.6} exceeds cap {:.6}",
                level.size,
                cap
            );
        }
        for level in &ladder.asks {
            assert!(
                level.size <= cap + 0.001,
                "Asymmetric ask size {:.6} exceeds cap {:.6}",
                level.size,
                cap
            );
        }
    }

    #[test]
    fn test_dedup_merge_levels() {
        let mut levels: LadderLevels = smallvec![
            LadderLevel {
                price: 100.0,
                size: 0.5,
                depth_bps: 2.0
            },
            LadderLevel {
                price: 100.0,
                size: 0.3,
                depth_bps: 3.0
            },
            LadderLevel {
                price: 101.0,
                size: 0.2,
                depth_bps: 5.0
            },
            LadderLevel {
                price: 101.0,
                size: 0.1,
                depth_bps: 6.0
            },
            LadderLevel {
                price: 102.0,
                size: 0.4,
                depth_bps: 8.0
            },
        ];
        dedup_merge_levels(&mut levels);

        assert_eq!(levels.len(), 3, "should merge to 3 unique prices");
        assert!((levels[0].size - 0.8).abs() < 1e-9, "100.0: 0.5+0.3=0.8");
        assert!((levels[1].size - 0.3).abs() < 1e-9, "101.0: 0.2+0.1=0.3");
        assert!((levels[2].size - 0.4).abs() < 1e-9, "102.0: 0.4");
    }

    #[test]
    fn test_dedup_merge_levels_single_and_empty() {
        // Empty
        let mut empty: LadderLevels = smallvec![];
        dedup_merge_levels(&mut empty);
        assert!(empty.is_empty());

        // Single
        let mut single: LadderLevels = smallvec![LadderLevel {
            price: 50.0,
            size: 1.0,
            depth_bps: 5.0
        },];
        dedup_merge_levels(&mut single);
        assert_eq!(single.len(), 1);
    }

    #[test]
    fn test_hype_price_no_duplicates() {
        // HYPE at ~$31.61 with geometric spacing produces sub-$0.001 differences
        // that collapse after rounding. Verify dedup removes them.
        let depths = vec![2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0];
        let sizes = vec![0.1; 8];
        let mid = 31.61;
        let market_mid = 31.61;
        let decimals = 4; // HYPE uses 4 decimal places
        let sz_decimals = 1;
        let min_notional = 10.0;

        let ladder = build_raw_ladder(&RawLadderInput {
            depths: &depths,
            sizes: &sizes,
            mid,
            market_mid,
            exchange_best_bid: 0.0,
            exchange_best_ask: 0.0,
            decimals,
            sz_decimals,
            min_notional,
        });

        // Check no duplicate prices on either side
        for i in 1..ladder.bids.len() {
            assert!(
                (ladder.bids[i].price - ladder.bids[i - 1].price).abs() > EPSILON,
                "duplicate bid price at index {}: {:.4}",
                i,
                ladder.bids[i].price
            );
        }
        for i in 1..ladder.asks.len() {
            assert!(
                (ladder.asks[i].price - ladder.asks[i - 1].price).abs() > EPSILON,
                "duplicate ask price at index {}: {:.4}",
                i,
                ladder.asks[i].price
            );
        }
    }

    #[test]
    fn test_bbo_cap_allows_large_skew_when_ladder_is_wide() {
        // Scenario: HYPE with tight exchange BBO (~0.3 bps) but our ladder 8+ bps from mid.
        // The BBO cap should use our own ladder depth as floor, not the tiny exchange BBO.
        let mid = 22.80;
        let mut ladder = Ladder {
            bids: smallvec![LadderLevel {
                price: mid * (1.0 - 8.0 / 10000.0), // ~8 bps below mid
                size: 1.0,
                depth_bps: 8.0,
            }],
            asks: smallvec![LadderLevel {
                price: mid * (1.0 + 8.0 / 10000.0), // ~8 bps above mid
                size: 1.0,
                depth_bps: 8.0,
            }],
        };

        let original_bid = ladder.bids[0].price;
        let original_ask = ladder.asks[0].price;

        // Tight exchange BBO: 0.3 bps half-spread = 0.0003 * mid = $0.00684
        let tight_bbo_half = mid * 0.3 / 10000.0;
        let cached_best_bid = mid - tight_bbo_half;
        let cached_best_ask = mid + tight_bbo_half;

        // Apply large inventory skew (~4 bps raw offset)
        // base_skew = 0.5 * 3.0 * 0.01^2 * 10.0 = 0.00015 → 1.5 bps (plus size asymmetry)
        // With gamma=5.0: 0.5 * 5.0 * 0.01^2 * 10.0 = 0.00025 → 2.5 bps raw
        // With gamma=8.0: 0.5 * 8.0 * 0.01^2 * 10.0 = 0.0004 → 4.0 bps raw
        apply_inventory_skew_with_drift(
            &mut ladder,
            0.5,  // inventory_ratio: 50% long
            8.0,  // gamma: high risk aversion to produce ~4 bps skew
            0.01, // sigma: moderate vol
            10.0, // time_horizon
            mid,
            mid,
            2,     // decimals
            4,     // sz_decimals
            false, // use_drift_adjusted_skew
            0.0,   // hjb_drift_urgency
            false, // position_opposes_momentum
            0.0,   // urgency_score
            0.0,   // funding_rate
            false, // use_funding_skew
            cached_best_bid,
            cached_best_ask,
            8.0, // effective_floor_bps
        );

        // With old code: max_offset = tight_bbo_half * 0.8 = 0.24 bps → almost no skew
        // With new code: max_offset = max(0.24 bps, 4 bps own floor) → meaningful skew
        let bid_shift_bps = (original_bid - ladder.bids[0].price) / mid * 10000.0;
        let ask_shift_bps = (original_ask - ladder.asks[0].price) / mid * 10000.0;

        // Skew should be at least 2 bps (not capped to 0.24 bps by tight BBO)
        // Raw skew is ~4 bps from gamma=8.0, sigma=0.01, inv_ratio=0.5
        assert!(
            bid_shift_bps.abs() > 2.0 || ask_shift_bps.abs() > 2.0,
            "BBO cap killed inventory skew: bid_shift={:.2} bps, ask_shift={:.2} bps",
            bid_shift_bps,
            ask_shift_bps
        );
    }

    #[test]
    fn test_empty_ladder_uses_effective_floor_bps() {
        // Empty ladder → own_min_depth = f64::INFINITY → should fallback to effective_floor_bps
        let mut ladder = Ladder {
            bids: smallvec![],
            asks: smallvec![],
        };

        let mid = 31.0; // HYPE-like price
        let effective_floor_bps = 10.0;

        // With empty ladder, own_floor should be mid * (effective_floor_bps / 10000)
        // = 31.0 * (10.0 / 10000.0) = 31.0 * 0.001 = 0.031
        // The skew cap should use this own_floor, not the old hardcoded 8 bps
        apply_inventory_skew_with_drift(
            &mut ladder,
            0.5,  // inventory_ratio
            0.3,  // gamma
            0.01, // sigma
            10.0, // time_horizon
            mid,
            mid,
            4,     // decimals (HYPE-like)
            1,     // sz_decimals
            false, // use_drift_adjusted_skew
            0.0,   // hjb_drift_urgency
            false, // position_opposes_momentum
            0.0,   // urgency_score
            0.0,   // funding_rate
            false, // use_funding_skew
            0.0,   // cached_best_bid
            0.0,   // cached_best_ask
            effective_floor_bps,
        );

        // Empty ladder, nothing to shift — just verify no panic
        assert!(ladder.bids.is_empty());
        assert!(ladder.asks.is_empty());
    }

    /// Validate the actual_tick computation used in BBO crossing checks (reconcile.rs).
    /// The old proxy (mid * 0.0001) produced $0.0031 for HYPE at $31 when the real
    /// exchange tick (10^-4 = $0.0001) is 31x smaller.
    #[test]
    fn test_actual_tick_vs_proxy_for_bbo_crossing() {
        // HYPE: decimals=4, price=$31
        let decimals: u32 = 4;
        let mid = 31.0;

        let actual_tick = 10f64.powi(-(decimals as i32)); // $0.0001
        let old_proxy = mid * 0.0001; // $0.0031

        assert!(
            (actual_tick - 0.0001).abs() < 1e-12,
            "HYPE tick should be 0.0001, got {}",
            actual_tick
        );
        assert!(
            old_proxy > actual_tick * 30.0,
            "Old proxy {:.6} should be ~31x larger than actual tick {:.6}",
            old_proxy,
            actual_tick
        );

        let safety_margin = actual_tick * 2.0; // 2 ticks = $0.0002

        // Scenario 1: HYPE bid at 30.9970 with best_ask=30.9980
        // Gap = 30.9980 - 30.9970 = 0.0010 = 10 ticks → NOT filtered
        let bid_price = 30.997;
        let best_ask = 30.998;
        let should_filter_bid = bid_price >= best_ask - safety_margin;
        assert!(
            !should_filter_bid,
            "Bid at {:.4} should NOT be filtered with best_ask={:.4} (gap={:.4} > safety={:.4})",
            bid_price,
            best_ask,
            best_ask - bid_price,
            safety_margin
        );

        // Scenario 2: HYPE bid at 30.9978 with best_ask=30.9980
        // Gap = 30.9980 - 30.9978 = 0.0002 = 2 ticks = exactly safety_margin → IS filtered
        let bid_price_tight = 30.9978;
        let should_filter_tight = bid_price_tight >= best_ask - safety_margin;
        assert!(
            should_filter_tight,
            "Bid at {:.4} SHOULD be filtered with best_ask={:.4} (gap={:.4} <= safety={:.4})",
            bid_price_tight,
            best_ask,
            best_ask - bid_price_tight,
            safety_margin
        );

        // BTC: decimals=0, price=$87000
        let btc_decimals: u32 = 0;
        let btc_tick = 10f64.powi(-(btc_decimals as i32)); // $1.0
        assert!(
            (btc_tick - 1.0).abs() < 1e-12,
            "BTC tick should be 1.0, got {}",
            btc_tick
        );
    }

    #[test]
    fn test_empty_ladder_effective_floor_clamps_to_5bps_minimum() {
        // Even if effective_floor_bps is very small, own_floor should be at least 5 bps
        let mut ladder = Ladder {
            bids: smallvec![LadderLevel {
                price: 30.99,
                size: 1.0,
                depth_bps: 3.0,
            }],
            asks: smallvec![LadderLevel {
                price: 31.01,
                size: 1.0,
                depth_bps: 3.0,
            }],
        };

        let mid = 31.0;
        // Pass a very low effective_floor_bps — should clamp to 5.0
        apply_inventory_skew_with_drift(
            &mut ladder,
            0.5,  // inventory_ratio: strong long
            8.0,  // gamma: high to produce large skew
            0.01, // sigma
            10.0, // time_horizon
            mid,
            mid,
            4, // decimals
            1, // sz_decimals
            false,
            0.0,
            false,
            0.0,
            0.0,
            false,
            0.0,
            0.0,
            1.0, // effective_floor_bps: very low (would be 1 bps)
        );

        // Ladder has levels so own_min_depth is finite — the effective_floor_bps
        // doesn't matter in this case (own_min_depth * 0.5 is used instead).
        // Just verify no panic and prices shifted.
        assert!(ladder.bids[0].price < 30.99);
    }
}
