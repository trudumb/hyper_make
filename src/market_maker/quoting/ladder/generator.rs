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

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use super::{Ladder, LadderConfig, LadderLevel, LadderLevels, LadderParams};
use crate::market_maker::infra::capacity::DEPTH_INLINE_CAPACITY;

/// Type alias for depth values using SmallVec for stack allocation
type DepthVec = SmallVec<[f64; DEPTH_INLINE_CAPACITY]>;

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
        let mut ladder = build_raw_ladder(
            &depths,
            &sizes,
            params.mid_price,
            params.decimals,
            params.sz_decimals,
            params.min_notional,
        );

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
        );

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
        let mut ladder = build_asymmetric_ladder(
            &bid_depths,
            &bid_sizes,
            &ask_depths,
            &ask_sizes,
            params.mid_price,
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
                result[0] = total_size; // Concentrate on tightest level
                tracing::info!(
                    levels = result.len(),
                    total_size = %format!("{:.6}", total_size),
                    "Ladder allocate_sizes: MV=0 fallback, concentrating on tightest level"
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
    // Use concentration fallback: put all size on first level (tightest depth)
    let all_filtered = result.iter().all(|&s| s < EPSILON);
    if all_filtered && total_size >= min_size {
        result[0] = total_size;
        tracing::info!(
            levels = result.len(),
            total_size = %format!("{:.6}", total_size),
            min_size = %format!("{:.6}", min_size),
            "Ladder allocate_sizes: min_size fallback, concentrating on tightest level"
        );
    }

    result
}

/// Build raw ladder from depths and sizes (before inventory skew).
///
/// Creates symmetric bid/ask levels around mid price, applying
/// proper price rounding and size truncation per exchange requirements.
/// Uses SmallVec to avoid heap allocation for typical ladder sizes.
pub(crate) fn build_raw_ladder(
    depths: &[f64],
    sizes: &[f64],
    mid: f64,
    decimals: u32,
    sz_decimals: u32,
    min_notional: f64,
) -> Ladder {
    // DEBUG: log entry to verify code path
    let total_size: f64 = sizes.iter().sum();
    tracing::debug!(
        num_depths = depths.len(),
        num_sizes = sizes.len(),
        total_size = %format!("{:.6}", total_size),
        mid = %format!("{:.4}", mid),
        min_notional = %format!("{:.2}", min_notional),
        "build_raw_ladder called"
    );

    let mut bids = LadderLevels::new();
    let mut asks = LadderLevels::new();

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

    // Concentration fallback: if all levels filtered by min_notional,
    // but total size meets min_notional, create single level at tightest depth
    if bids.is_empty() && !depths.is_empty() && !sizes.is_empty() {
        let total_size: f64 = sizes.iter().sum();
        let total_size_truncated = truncate_float(total_size, sz_decimals, false);
        if let Some(&tightest_depth) = depths.first() {
            let offset = mid * (tightest_depth / 10000.0);
            let bid_price = round_to_significant_and_decimal(mid - offset, 5, decimals);

            if bid_price * total_size_truncated >= min_notional {
                tracing::info!(
                    total_size = %format!("{:.6}", total_size_truncated),
                    price = %format!("{:.4}", bid_price),
                    notional = %format!("{:.2}", bid_price * total_size_truncated),
                    depth_bps = %format!("{:.2}", tightest_depth),
                    "Bid concentration fallback: single order at tightest depth"
                );
                bids.push(LadderLevel {
                    price: bid_price,
                    size: total_size_truncated,
                    depth_bps: tightest_depth,
                });
            }
        }
    }

    if asks.is_empty() && !depths.is_empty() && !sizes.is_empty() {
        let total_size: f64 = sizes.iter().sum();
        let total_size_truncated = truncate_float(total_size, sz_decimals, false);
        if let Some(&tightest_depth) = depths.first() {
            let offset = mid * (tightest_depth / 10000.0);
            let ask_price = round_to_significant_and_decimal(mid + offset, 5, decimals);

            if ask_price * total_size_truncated >= min_notional {
                tracing::info!(
                    total_size = %format!("{:.6}", total_size_truncated),
                    price = %format!("{:.4}", ask_price),
                    notional = %format!("{:.2}", ask_price * total_size_truncated),
                    depth_bps = %format!("{:.2}", tightest_depth),
                    "Ask concentration fallback: single order at tightest depth"
                );
                asks.push(LadderLevel {
                    price: ask_price,
                    size: total_size_truncated,
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

    Ladder { bids, asks }
}

/// Build ladder with asymmetric depths for bids and asks.
///
/// Each side gets its own depth array and corresponding size array.
/// This is used when bid/ask depths differ (e.g., different κ for each side).
/// Uses SmallVec to avoid heap allocation for typical ladder sizes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_asymmetric_ladder(
    bid_depths: &[f64],
    bid_sizes: &[f64],
    ask_depths: &[f64],
    ask_sizes: &[f64],
    mid: f64,
    decimals: u32,
    sz_decimals: u32,
    min_notional: f64,
) -> Ladder {
    let mut bids = LadderLevels::new();
    let mut asks = LadderLevels::new();

    // Build bid levels
    for (&depth_bps, &size) in bid_depths.iter().zip(bid_sizes.iter()) {
        if size < EPSILON {
            continue;
        }

        let offset = mid * (depth_bps / 10000.0);
        let bid_price = round_to_significant_and_decimal(mid - offset, 5, decimals);
        let size = truncate_float(size, sz_decimals, false);

        if bid_price * size >= min_notional {
            bids.push(LadderLevel {
                price: bid_price,
                size,
                depth_bps,
            });
        }
    }

    // Build ask levels
    for (&depth_bps, &size) in ask_depths.iter().zip(ask_sizes.iter()) {
        if size < EPSILON {
            continue;
        }

        let offset = mid * (depth_bps / 10000.0);
        let ask_price = round_to_significant_and_decimal(mid + offset, 5, decimals);
        let size = truncate_float(size, sz_decimals, false);

        if ask_price * size >= min_notional {
            asks.push(LadderLevel {
                price: ask_price,
                size,
                depth_bps,
            });
        }
    }

    // Concentration fallback for bids (asymmetric)
    if bids.is_empty() && !bid_depths.is_empty() && !bid_sizes.is_empty() {
        let total_size: f64 = bid_sizes.iter().sum();
        let total_size_truncated = truncate_float(total_size, sz_decimals, false);
        if let Some(&tightest_depth) = bid_depths.first() {
            let offset = mid * (tightest_depth / 10000.0);
            let bid_price = round_to_significant_and_decimal(mid - offset, 5, decimals);

            if bid_price * total_size_truncated >= min_notional {
                tracing::info!(
                    total_size = %format!("{:.6}", total_size_truncated),
                    price = %format!("{:.4}", bid_price),
                    notional = %format!("{:.2}", bid_price * total_size_truncated),
                    depth_bps = %format!("{:.2}", tightest_depth),
                    "Bid concentration fallback (asymmetric): single order at tightest depth"
                );
                bids.push(LadderLevel {
                    price: bid_price,
                    size: total_size_truncated,
                    depth_bps: tightest_depth,
                });
            }
        }
    }

    // Concentration fallback for asks (asymmetric)
    if asks.is_empty() && !ask_depths.is_empty() && !ask_sizes.is_empty() {
        let total_size: f64 = ask_sizes.iter().sum();
        let total_size_truncated = truncate_float(total_size, sz_decimals, false);
        if let Some(&tightest_depth) = ask_depths.first() {
            let offset = mid * (tightest_depth / 10000.0);
            let ask_price = round_to_significant_and_decimal(mid + offset, 5, decimals);

            if ask_price * total_size_truncated >= min_notional {
                tracing::info!(
                    total_size = %format!("{:.6}", total_size_truncated),
                    price = %format!("{:.4}", ask_price),
                    notional = %format!("{:.2}", ask_price * total_size_truncated),
                    depth_bps = %format!("{:.2}", tightest_depth),
                    "Ask concentration fallback (asymmetric): single order at tightest depth"
                );
                asks.push(LadderLevel {
                    price: ask_price,
                    size: total_size_truncated,
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

    // === COMBINED SKEW ===
    let total_skew_fraction = base_skew_fraction + drift_skew_fraction;

    // Early return only if there's no inventory AND no drift adjustment
    if inventory_ratio.abs() < EPSILON && drift_skew_fraction.abs() < EPSILON {
        return;
    }

    let offset = mid * total_skew_fraction;

    // Shift all prices by combined offset and RE-ROUND to exchange precision
    for level in &mut ladder.bids {
        level.price = round_to_significant_and_decimal(level.price - offset, 5, decimals);
    }
    for level in &mut ladder.asks {
        level.price = round_to_significant_and_decimal(level.price - offset, 5, decimals);
    }

    // === FINAL SAFETY CHECK: Prevent crossed quotes ===
    // Use market_mid (actual exchange mid) NOT microprice for safety check
    // This ensures quotes never cross the actual market spread
    for level in &mut ladder.bids {
        if level.price > market_mid {
            tracing::error!(
                bid_price = %format!("{:.4}", level.price),
                market_mid = %format!("{:.4}", market_mid),
                microprice = %format!("{:.4}", mid),
                offset = %format!("{:.6}", offset),
                "CRITICAL: Bid crossed market mid - adjusting to 1bp below"
            );
            level.price = round_to_significant_and_decimal(market_mid * 0.9999, 5, decimals);
        }
    }
    for level in &mut ladder.asks {
        if level.price < market_mid {
            tracing::error!(
                ask_price = %format!("{:.4}", level.price),
                market_mid = %format!("{:.4}", market_mid),
                microprice = %format!("{:.4}", mid),
                offset = %format!("{:.6}", offset),
                "CRITICAL: Ask crossed market mid - adjusting to 1bp above"
            );
            level.price = round_to_significant_and_decimal(market_mid * 1.0001, 5, decimals);
        }
    }

    // Size skew: reduce side that increases position
    // Cap reduction at 90% to never completely remove quotes
    if inventory_ratio.abs() >= EPSILON {
        let size_reduction = inventory_ratio.abs().min(0.9);

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
        );

        // Prices should be integers (no fractional part)
        assert_eq!(ladder.bids[0].price, ladder.bids[0].price.round());
        assert_eq!(ladder.asks[0].price, ladder.asks[0].price.round());
    }
}
