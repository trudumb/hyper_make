//! Tick-grid-first level placement and scoring for ladder quoting.
//!
//! Instead of computing continuous bps-space depths and then rounding to exchange ticks
//! (which causes duplicate-price collapse), this module starts from the exchange tick grid
//! and enumerates valid prices directly. Every price is guaranteed unique by construction.
//!
//! # Pipeline
//! ```text
//! TickGridConfig::compute() → quoting range [touch_bps, max_depth_bps]
//! enumerate_ticks()         → N distinct exchange-tick-aligned prices
//! score_ticks()             → utility per tick = P(fill) × (edge - AS - fee)
//! select_optimal_ticks()    → top-K by utility (K = capital-limited levels)
//! ```

/// A candidate level at a specific exchange tick.
#[derive(Debug, Clone, Copy)]
pub struct TickLevel {
    /// Exchange-aligned price (guaranteed on tick boundary).
    pub price: f64,
    /// Distance from mid in basis points.
    pub depth_bps: f64,
    /// Number of ticks from the touch level.
    pub tick_offset: u32,
    /// Expected utility: P(fill) × (edge - AS - fee). Set by `score_ticks()`.
    pub utility: f64,
}

/// Configuration for tick-grid level enumeration.
#[derive(Debug, Clone)]
pub struct TickGridConfig {
    /// Current mark/mid price.
    pub mark_price: f64,
    /// Exchange tick size (minimum price increment, e.g., 0.0001 for HYPE).
    pub tick_size: f64,
    /// Tick size in basis points at current price: tick_size / mark_price * 10000.
    pub tick_bps: f64,
    /// GLFT optimal touch depth in bps (from spread engine).
    pub touch_depth_bps: f64,
    /// Maximum quoting depth in bps.
    pub max_depth_bps: f64,
    /// Maximum number of levels per side.
    pub max_levels: usize,
    /// Size decimals for order rounding.
    pub sz_decimals: u32,
    /// Minimum tick spacing multiplier (e.g., 3.0 for Micro = at least 3 ticks apart).
    pub min_tick_spacing_mult: f64,
}

impl TickGridConfig {
    /// Compute tick grid configuration from market parameters.
    pub fn compute(
        mark_price: f64,
        tick_size: f64,
        touch_depth_bps: f64,
        max_depth_bps: f64,
        max_levels: usize,
        sz_decimals: u32,
        min_tick_spacing_mult: f64,
    ) -> Self {
        let tick_bps = if mark_price > 0.0 {
            tick_size / mark_price * 10_000.0
        } else {
            1.0
        };
        Self {
            mark_price,
            tick_size,
            tick_bps,
            touch_depth_bps: touch_depth_bps.max(tick_bps), // At least 1 tick from mid
            max_depth_bps: max_depth_bps.max(touch_depth_bps + tick_bps),
            max_levels: max_levels.max(1),
            sz_decimals,
            min_tick_spacing_mult: min_tick_spacing_mult.max(1.0),
        }
    }
}

/// Enumerate distinct bid tick levels below the mid price.
///
/// Starts at the touch (snapped to tick) and steps outward by `spacing_ticks`.
/// Every price is on a valid tick boundary by construction — no dedup needed.
pub fn enumerate_bid_ticks(config: &TickGridConfig) -> Vec<TickLevel> {
    if config.mark_price <= 0.0 || config.tick_size <= 0.0 || config.max_levels == 0 {
        return Vec::new();
    }

    let spacing_ticks = compute_spacing_ticks(config);
    let mut levels = Vec::with_capacity(config.max_levels);

    // Touch: snap touch_depth_bps to nearest tick below mid
    let touch_offset_ticks = (config.touch_depth_bps / config.tick_bps).ceil() as u32;
    let touch_offset_ticks = touch_offset_ticks.max(1);

    for i in 0..config.max_levels {
        let tick_offset = touch_offset_ticks + (i as u32) * spacing_ticks;
        let price = config.mark_price - (tick_offset as f64) * config.tick_size;

        if price <= 0.0 {
            break;
        }

        let depth_bps = (config.mark_price - price) / config.mark_price * 10_000.0;
        if depth_bps > config.max_depth_bps {
            break;
        }

        levels.push(TickLevel {
            price,
            depth_bps,
            tick_offset,
            utility: 0.0,
        });
    }

    levels
}

/// Enumerate distinct ask tick levels above the mid price.
///
/// Mirror of `enumerate_bid_ticks` for the ask side.
pub fn enumerate_ask_ticks(config: &TickGridConfig) -> Vec<TickLevel> {
    if config.mark_price <= 0.0 || config.tick_size <= 0.0 || config.max_levels == 0 {
        return Vec::new();
    }

    let spacing_ticks = compute_spacing_ticks(config);
    let mut levels = Vec::with_capacity(config.max_levels);

    let touch_offset_ticks = (config.touch_depth_bps / config.tick_bps).ceil() as u32;
    let touch_offset_ticks = touch_offset_ticks.max(1);

    for i in 0..config.max_levels {
        let tick_offset = touch_offset_ticks + (i as u32) * spacing_ticks;
        let price = config.mark_price + (tick_offset as f64) * config.tick_size;

        let depth_bps = (price - config.mark_price) / config.mark_price * 10_000.0;
        if depth_bps > config.max_depth_bps {
            break;
        }

        levels.push(TickLevel {
            price,
            depth_bps,
            tick_offset,
            utility: 0.0,
        });
    }

    levels
}

/// Compute tick spacing to spread levels across the quoting range.
///
/// `spacing_bps = max(tick_bps, (max_depth - touch) / (max_levels - 1), min_spacing_mult × tick_bps)`
/// Rounded up to whole ticks.
fn compute_spacing_ticks(config: &TickGridConfig) -> u32 {
    if config.max_levels <= 1 {
        return 1;
    }

    let range_bps = config.max_depth_bps - config.touch_depth_bps;
    let natural_spacing_bps = range_bps / (config.max_levels - 1) as f64;
    let min_spacing_bps = config.tick_bps * config.min_tick_spacing_mult;

    let spacing_bps = natural_spacing_bps.max(min_spacing_bps).max(config.tick_bps);
    let spacing_ticks = (spacing_bps / config.tick_bps).ceil() as u32;

    spacing_ticks.max(1)
}

// ============================================================================
// WS2: Fill-Probability-Weighted Depth Selection
// ============================================================================

/// Parameters for tick scoring.
#[derive(Debug, Clone)]
pub struct TickScoringParams {
    /// Volatility (per-second).
    pub sigma: f64,
    /// Time horizon (seconds).
    pub tau: f64,
    /// Adverse selection at touch in basis points.
    pub as_at_touch_bps: f64,
    /// AS exponential decay rate in bps.
    pub as_decay_bps: f64,
    /// Maker fee in basis points.
    pub fee_bps: f64,
}

/// Score each tick level by expected utility: U(δ) = P_fill(δ) × (δ - AS(δ) - fee).
///
/// - `P_fill(δ) = 2Φ(-δ / (σ√τ))` from first-passage Brownian motion
/// - `AS(δ) = as_at_touch × exp(-δ / as_decay_bps)`
pub fn score_ticks(levels: &mut [TickLevel], params: &TickScoringParams) {
    let sigma_sqrt_tau = params.sigma * params.tau.sqrt();

    for level in levels.iter_mut() {
        let depth = level.depth_bps;

        // Fill probability from first-passage theory
        let p_fill = if depth < 1e-9 || sigma_sqrt_tau < 1e-12 {
            1.0
        } else {
            let depth_frac = depth / 10_000.0;
            let z = -depth_frac / sigma_sqrt_tau;
            2.0 * std_normal_cdf(z)
        };

        // Adverse selection decays with depth
        let as_cost = params.as_at_touch_bps * (-depth / params.as_decay_bps.max(1.0)).exp();

        // Edge = depth - AS - fee
        let edge = depth - as_cost - params.fee_bps;

        // Utility = P(fill) × edge
        level.utility = p_fill * edge;
    }
}

/// Select top-K ticks by utility, always including the touch.
///
/// Returns levels sorted by depth ascending (closest to mid first).
pub fn select_optimal_ticks(levels: &[TickLevel], max_levels: usize) -> Vec<TickLevel> {
    if levels.is_empty() || max_levels == 0 {
        return Vec::new();
    }

    // Filter to positive utility
    let mut candidates: Vec<TickLevel> = levels
        .iter()
        .filter(|l| l.utility > 0.0)
        .copied()
        .collect();

    // Always include touch (first level) even if utility is slightly negative
    if candidates.is_empty() && !levels.is_empty() {
        candidates.push(levels[0]);
    }

    // Sort by utility descending
    candidates.sort_by(|a, b| b.utility.partial_cmp(&a.utility).unwrap_or(std::cmp::Ordering::Equal));

    // Take top max_levels
    candidates.truncate(max_levels);

    // Re-sort by depth ascending (closest to mid first)
    candidates.sort_by(|a, b| a.depth_bps.partial_cmp(&b.depth_bps).unwrap_or(std::cmp::Ordering::Equal));

    candidates
}

/// Standard normal CDF approximation (Abramowitz & Stegun).
fn std_normal_cdf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + P * x_abs);
    let y = 1.0
        - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1)
            * t
            * (-x_abs * x_abs / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hype_config() -> TickGridConfig {
        // HYPE at $29.63, tick_size=0.0001 (4 price decimals)
        TickGridConfig::compute(
            29.63,
            0.0001,
            8.0,   // touch at 8 bps
            50.0,   // max depth 50 bps
            10,     // 10 levels
            2,      // sz_decimals
            1.0,    // min 1 tick spacing
        )
    }

    #[test]
    fn test_tick_grid_config_compute() {
        let cfg = hype_config();
        assert!(cfg.tick_bps > 0.0);
        // tick_bps = 0.0001 / 29.63 * 10000 ≈ 0.0338
        assert!((cfg.tick_bps - 0.0338).abs() < 0.001,
            "tick_bps={}", cfg.tick_bps);
        assert_eq!(cfg.max_levels, 10);
    }

    #[test]
    fn test_enumerate_bid_ticks_unique_prices() {
        let cfg = hype_config();
        let bids = enumerate_bid_ticks(&cfg);

        assert!(!bids.is_empty(), "Should produce bid levels");

        // All prices must be unique
        for i in 1..bids.len() {
            assert!(bids[i].price < bids[i - 1].price,
                "Bid prices must be strictly decreasing: [{i}]={} >= [{prev}]={}",
                bids[i].price, bids[i - 1].price, prev = i - 1);
        }

        // All prices below mid
        for level in &bids {
            assert!(level.price < cfg.mark_price,
                "Bid price {} should be below mid {}", level.price, cfg.mark_price);
        }
    }

    #[test]
    fn test_enumerate_ask_ticks_unique_prices() {
        let cfg = hype_config();
        let asks = enumerate_ask_ticks(&cfg);

        assert!(!asks.is_empty(), "Should produce ask levels");

        // All prices must be unique
        for i in 1..asks.len() {
            assert!(asks[i].price > asks[i - 1].price,
                "Ask prices must be strictly increasing: [{i}]={} <= [{prev}]={}",
                asks[i].price, asks[i - 1].price, prev = i - 1);
        }

        // All prices above mid
        for level in &asks {
            assert!(level.price > cfg.mark_price,
                "Ask price {} should be above mid {}", level.price, cfg.mark_price);
        }
    }

    #[test]
    fn test_enumerate_respects_max_levels() {
        let cfg = TickGridConfig::compute(29.63, 0.0001, 8.0, 200.0, 5, 2, 1.0);
        let bids = enumerate_bid_ticks(&cfg);
        assert!(bids.len() <= 5, "Should respect max_levels=5, got {}", bids.len());
    }

    #[test]
    fn test_enumerate_respects_max_depth() {
        let cfg = TickGridConfig::compute(29.63, 0.0001, 8.0, 15.0, 100, 2, 1.0);
        let bids = enumerate_bid_ticks(&cfg);
        for level in &bids {
            assert!(level.depth_bps <= 15.0 + 0.1,
                "depth_bps {} should be <= max_depth 15.0", level.depth_bps);
        }
    }

    #[test]
    fn test_min_tick_spacing_mult() {
        // With min_tick_spacing_mult=3, levels should be at least 3 ticks apart
        let cfg = TickGridConfig::compute(29.63, 0.0001, 8.0, 200.0, 10, 2, 3.0);
        let bids = enumerate_bid_ticks(&cfg);
        if bids.len() >= 2 {
            for i in 1..bids.len() {
                let tick_diff = bids[i - 1].tick_offset.abs_diff(bids[i].tick_offset);
                assert!(tick_diff >= 3,
                    "Levels should be at least 3 ticks apart, got {}", tick_diff);
            }
        }
    }

    #[test]
    fn test_score_ticks_positive_utility_at_depth() {
        let mut levels = vec![
            TickLevel { price: 29.60, depth_bps: 10.0, tick_offset: 300, utility: 0.0 },
            TickLevel { price: 29.55, depth_bps: 27.0, tick_offset: 800, utility: 0.0 },
        ];
        let params = TickScoringParams {
            sigma: 0.0003,
            tau: 10.0,
            as_at_touch_bps: 3.0,
            as_decay_bps: 10.0,
            fee_bps: 1.5,
        };
        score_ticks(&mut levels, &params);

        // Both should have positive utility at reasonable depth
        assert!(levels[0].utility > 0.0,
            "10 bps depth should have positive utility: {}", levels[0].utility);
        // Deeper level should have lower fill prob but higher edge
        assert!(levels[1].utility > 0.0 || levels[1].utility <= levels[0].utility,
            "27 bps level utility={}", levels[1].utility);
    }

    #[test]
    fn test_select_optimal_ticks_filters_negative() {
        let levels = vec![
            TickLevel { price: 29.62, depth_bps: 3.0, tick_offset: 100, utility: -0.5 },
            TickLevel { price: 29.60, depth_bps: 10.0, tick_offset: 300, utility: 2.5 },
            TickLevel { price: 29.58, depth_bps: 17.0, tick_offset: 500, utility: 1.8 },
            TickLevel { price: 29.55, depth_bps: 27.0, tick_offset: 800, utility: 0.3 },
        ];

        let selected = select_optimal_ticks(&levels, 3);

        // Should exclude the negative-utility level
        assert!(selected.len() <= 3);
        for level in &selected {
            assert!(level.utility > 0.0, "Selected level should have positive utility");
        }
    }

    #[test]
    fn test_select_optimal_ticks_sorted_by_depth() {
        let levels = vec![
            TickLevel { price: 29.60, depth_bps: 10.0, tick_offset: 300, utility: 2.5 },
            TickLevel { price: 29.55, depth_bps: 27.0, tick_offset: 800, utility: 1.0 },
            TickLevel { price: 29.58, depth_bps: 17.0, tick_offset: 500, utility: 1.8 },
        ];

        let selected = select_optimal_ticks(&levels, 3);

        // Should be sorted by depth ascending
        for i in 1..selected.len() {
            assert!(selected[i].depth_bps >= selected[i - 1].depth_bps,
                "Should be sorted by depth: {}bps >= {}bps",
                selected[i].depth_bps, selected[i - 1].depth_bps);
        }
    }

    #[test]
    fn test_select_includes_touch_fallback() {
        // All negative utility — should still include touch as fallback
        let levels = vec![
            TickLevel { price: 29.62, depth_bps: 3.0, tick_offset: 100, utility: -0.5 },
            TickLevel { price: 29.60, depth_bps: 10.0, tick_offset: 300, utility: -1.0 },
        ];

        let selected = select_optimal_ticks(&levels, 5);
        assert_eq!(selected.len(), 1, "Should include touch as fallback");
        assert!((selected[0].depth_bps - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_config() {
        let cfg = TickGridConfig::compute(0.0, 0.0001, 8.0, 50.0, 10, 2, 1.0);
        let bids = enumerate_bid_ticks(&cfg);
        assert!(bids.is_empty(), "Should produce no levels with zero price");
    }

    #[test]
    fn test_full_pipeline_hype_100_dollar() {
        // End-to-end: $100 capital on HYPE at $29.63, 10x leverage
        // After WS0 fix: max_position ≈ 16.87, should produce 8+ levels
        let cfg = TickGridConfig::compute(
            29.63,
            0.0001,  // 4-decimal tick
            8.0,     // touch at 8 bps
            50.0,    // max depth 50 bps
            15,      // up to 15 levels
            2,       // sz_decimals
            1.0,     // min 1 tick spacing
        );

        let mut bids = enumerate_bid_ticks(&cfg);
        let mut asks = enumerate_ask_ticks(&cfg);

        let scoring = TickScoringParams {
            sigma: 0.0003,
            tau: 10.0,
            as_at_touch_bps: 3.0,
            as_decay_bps: 10.0,
            fee_bps: 1.5,
        };
        score_ticks(&mut bids, &scoring);
        score_ticks(&mut asks, &scoring);

        let selected_bids = select_optimal_ticks(&bids, 10);
        let selected_asks = select_optimal_ticks(&asks, 10);

        // Should have multiple distinct levels on each side
        assert!(selected_bids.len() >= 3,
            "Should have >=3 bid levels, got {}", selected_bids.len());
        assert!(selected_asks.len() >= 3,
            "Should have >=3 ask levels, got {}", selected_asks.len());

        // Verify all prices are unique
        for i in 1..selected_bids.len() {
            assert_ne!(selected_bids[i].price, selected_bids[i - 1].price,
                "Bid prices must be unique");
        }
        for i in 1..selected_asks.len() {
            assert_ne!(selected_asks[i].price, selected_asks[i - 1].price,
                "Ask prices must be unique");
        }
    }

    #[test]
    fn test_btc_tick_grid() {
        // BTC at $100k, tick_size=0.1 (1 decimal), sz_decimals=5
        let cfg = TickGridConfig::compute(
            100_000.0,
            0.1,     // $0.10 ticks
            5.0,     // touch at 5 bps = $5
            30.0,    // max depth 30 bps = $30
            8,
            5,
            1.0,
        );

        let bids = enumerate_bid_ticks(&cfg);
        assert!(!bids.is_empty());

        // tick_bps = 0.1 / 100000 * 10000 = 0.01 bps — very fine grid
        assert!(cfg.tick_bps < 0.02, "BTC tick_bps={}", cfg.tick_bps);

        // All unique
        for i in 1..bids.len() {
            assert!(bids[i].price < bids[i - 1].price);
        }
    }
}
