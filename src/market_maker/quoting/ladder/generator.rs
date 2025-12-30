//! Ladder generation logic.
//!
//! Implements multi-level quote generation with:
//! - Geometric or linear depth spacing
//! - Fill intensity modeling: λ(δ) = σ²/δ²
//! - Spread capture: SC(δ) = δ - AS₀ × exp(-δ/δ_char) - fees
//! - Size allocation proportional to marginal value

use crate::{round_to_significant_and_decimal, truncate_float, EPSILON};

use super::{Ladder, LadderConfig, LadderLevel, LadderParams};

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
pub(crate) fn compute_depths(config: &LadderConfig) -> Vec<f64> {
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
pub(crate) fn fill_intensity(depth_bps: f64, sigma: f64, kappa: f64) -> f64 {
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
pub(crate) fn allocate_sizes(
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

/// Build raw ladder from depths and sizes (before inventory skew).
///
/// Creates symmetric bid/ask levels around mid price, applying
/// proper price rounding and size truncation per exchange requirements.
pub(crate) fn build_raw_ladder(
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
pub(crate) fn apply_inventory_skew(
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
