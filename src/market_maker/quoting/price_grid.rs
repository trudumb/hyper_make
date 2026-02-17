//! Price grid quantization to reduce order churn.
//!
//! Sub-tick price oscillations cause cancel+replace cycles that consume
//! API quota and lose queue position. This module quantizes ladder prices
//! to a grid aligned with exchange tick boundaries, so that two prices
//! resolving to the same grid point keep the resting order untouched.
//!
//! Grid spacing adapts to volatility (wider in high-vol regimes) and
//! quota pressure (wider when rate limit headroom is low).

use serde::{Deserialize, Serialize};

// ── Configuration ──────────────────────────────────────────────────────

/// Configuration for price grid quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceGridConfig {
    /// Master enable/disable. Defaults to `false` for safe rollout.
    #[serde(default)]
    pub enabled: bool,

    /// Minimum grid spacing as multiple of tick size (default 2.0).
    #[serde(default = "default_min_tick_multiple")]
    pub min_tick_multiple: f64,

    /// Sigma divisor for adaptive spacing (default 4.0).
    /// Grid spacing = max(min_tick_multiple * tick_bps, sigma_bps / sigma_divisor).
    #[serde(default = "default_sigma_divisor")]
    pub sigma_divisor: f64,

    /// Quota pressure scaling factor (default 2.0).
    /// Under pressure: spacing *= 1 + (1 - headroom) * pressure_scaling.
    #[serde(default = "default_pressure_scaling")]
    pub pressure_scaling: f64,
}

fn default_min_tick_multiple() -> f64 {
    2.0
}
fn default_sigma_divisor() -> f64 {
    4.0
}
fn default_pressure_scaling() -> f64 {
    2.0
}

impl Default for PriceGridConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_tick_multiple: default_min_tick_multiple(),
            sigma_divisor: default_sigma_divisor(),
            pressure_scaling: default_pressure_scaling(),
        }
    }
}

// ── Price Grid ─────────────────────────────────────────────────────────

/// Price grid for quantizing order prices to reduce churn.
///
/// All ladder prices are snapped to the nearest grid point, then
/// tick-aligned. Two prices that resolve to the same grid point are
/// considered identical for reconciliation purposes.
#[derive(Debug, Clone)]
pub struct PriceGrid {
    /// Grid spacing in bps.
    pub spacing_bps: f64,
    /// Reference mid price for grid alignment.
    pub reference_mid: f64,
    /// Exchange tick size (absolute price units).
    pub tick_size: f64,
    /// Precomputed step = reference_mid * spacing_bps / 10_000,
    /// clamped to at least one tick.
    pub step: f64,
}

impl PriceGrid {
    /// Create a grid for the current market state.
    ///
    /// # Arguments
    /// * `mid` — current mid price (absolute)
    /// * `tick_size` — exchange tick size (absolute price units)
    /// * `sigma_bps` — recent realised volatility in basis points
    ///   (e.g. 1-minute sigma × 10 000)
    /// * `config` — grid configuration
    /// * `headroom_pct` — rate-limit headroom fraction [0.0, 1.0]
    pub fn for_current_state(
        mid: f64,
        tick_size: f64,
        sigma_bps: f64,
        config: &PriceGridConfig,
        headroom_pct: f64,
    ) -> Self {
        let tick_bps = if mid > 0.0 {
            tick_size / mid * 10_000.0
        } else {
            1.0
        };

        let base_spacing = (config.min_tick_multiple * tick_bps)
            .max(sigma_bps / config.sigma_divisor)
            .max(0.1); // absolute minimum

        // Widen under quota pressure
        let pressure_mult = if headroom_pct < 1.0 {
            1.0 + (1.0 - headroom_pct).max(0.0) * config.pressure_scaling
        } else {
            1.0
        };

        let spacing = base_spacing * pressure_mult;
        let step = (mid * spacing / 10_000.0).max(tick_size); // step >= tick

        Self {
            spacing_bps: spacing,
            reference_mid: mid,
            tick_size,
            step,
        }
    }

    /// Snap a price to the nearest grid point, then align to tick size.
    pub fn snap(&self, price: f64) -> f64 {
        if self.step <= 0.0 || self.tick_size <= 0.0 {
            return price;
        }

        // Round to nearest grid step
        let offset = price - self.reference_mid;
        let grid_steps = (offset / self.step).round();
        let snapped = self.reference_mid + grid_steps * self.step;

        // Align to tick
        (snapped / self.tick_size).round() * self.tick_size
    }

    /// Snap a bid price DOWN to the nearest grid point, then floor to tick.
    ///
    /// Bids snap conservatively downward so we never bid higher than intended.
    pub fn snap_bid(&self, price: f64) -> f64 {
        if self.step <= 0.0 || self.tick_size <= 0.0 {
            return price;
        }

        // Floor to grid step (bid snaps DOWN)
        let offset = price - self.reference_mid;
        let grid_steps = (offset / self.step).floor();
        let snapped = self.reference_mid + grid_steps * self.step;

        // Floor-align to tick
        (snapped / self.tick_size).floor() * self.tick_size
    }

    /// Snap an ask price UP to the nearest grid point, then ceil to tick.
    ///
    /// Asks snap conservatively upward so we never ask lower than intended.
    pub fn snap_ask(&self, price: f64) -> f64 {
        if self.step <= 0.0 || self.tick_size <= 0.0 {
            return price;
        }

        // Ceil to grid step (ask snaps UP)
        let offset = price - self.reference_mid;
        let grid_steps = (offset / self.step).ceil();
        let snapped = self.reference_mid + grid_steps * self.step;

        // Ceil-align to tick
        (snapped / self.tick_size).ceil() * self.tick_size
    }

    /// Check if two prices resolve to the same grid point.
    pub fn same_point(&self, a: f64, b: f64) -> bool {
        if self.step <= 0.0 {
            return (a - b).abs() < f64::EPSILON;
        }
        let a_snapped = self.snap(a);
        let b_snapped = self.snap(b);
        (a_snapped - b_snapped).abs() < self.tick_size * 0.5
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> PriceGridConfig {
        PriceGridConfig::default()
    }

    /// Helper: compute tick size for a given mid price (Hyperliquid 5-sig-fig rule).
    fn tick_for_mid(mid: f64) -> f64 {
        if mid <= 0.0 {
            return 0.01;
        }
        let integer_digits = (mid.log10().floor() as i32 + 1).max(1);
        let decimal_places = (5 - integer_digits).max(0);
        10_f64.powi(-decimal_places)
    }

    #[test]
    fn test_snap_to_grid() {
        let mid = 30.0;
        let tick = tick_for_mid(mid); // 0.01
        let grid = PriceGrid::for_current_state(mid, tick, 5.0, &default_cfg(), 1.0);

        // A price slightly off-grid should snap
        let snapped = grid.snap(30.013);
        // Must be on tick boundary
        assert!(
            (snapped / tick - (snapped / tick).round()).abs() < 1e-9,
            "snapped {} not on tick {}",
            snapped,
            tick
        );
    }

    #[test]
    fn test_snap_preserves_tick() {
        let mid = 100.0;
        let tick = tick_for_mid(mid); // 0.01
        let grid = PriceGrid::for_current_state(mid, tick, 10.0, &default_cfg(), 1.0);

        for offset in [0.003, 0.007, 0.012, 0.025, 0.049, 0.1] {
            let p = mid + offset;
            let s = grid.snap(p);
            let remainder = (s / tick) - (s / tick).round();
            assert!(
                remainder.abs() < 1e-9,
                "price {} → snapped {} not on tick {}",
                p,
                s,
                tick
            );
        }
    }

    #[test]
    fn test_same_point_symmetric() {
        let mid = 30.0;
        let tick = 0.01;
        let grid = PriceGrid::for_current_state(mid, tick, 5.0, &default_cfg(), 1.0);

        let a = 30.05;
        let b = 30.052;
        assert_eq!(
            grid.same_point(a, b),
            grid.same_point(b, a),
            "same_point must be symmetric"
        );
    }

    #[test]
    fn test_same_point_adjacent_grid() {
        let mid = 30.0;
        let tick = 0.01;
        let cfg = default_cfg();
        let grid = PriceGrid::for_current_state(mid, tick, 5.0, &cfg, 1.0);

        // Two prices separated by more than one full grid step should NOT be the same point
        let a = mid;
        let b = mid + 2.0 * grid.step;
        assert!(
            !grid.same_point(a, b),
            "prices {} and {} are {} steps apart, should NOT be same point (step={})",
            a,
            b,
            2.0,
            grid.step
        );
    }

    #[test]
    fn test_same_point_within_grid() {
        let mid = 30.0;
        let tick = 0.01;
        let cfg = default_cfg();
        let grid = PriceGrid::for_current_state(mid, tick, 5.0, &cfg, 1.0);

        // Two prices within the same grid cell should be same_point
        let a = mid + 0.001;
        let b = mid + 0.002;
        assert!(
            grid.same_point(a, b),
            "prices {} and {} within same grid cell (step={}) should be same_point",
            a,
            b,
            grid.step
        );
    }

    #[test]
    fn test_adaptive_spacing_sigma() {
        let mid = 30.0;
        let tick = 0.01;
        let cfg = default_cfg();

        let grid_low = PriceGrid::for_current_state(mid, tick, 2.0, &cfg, 1.0);
        let grid_high = PriceGrid::for_current_state(mid, tick, 40.0, &cfg, 1.0);

        assert!(
            grid_high.spacing_bps >= grid_low.spacing_bps,
            "higher sigma ({}) should give wider spacing ({} vs {})",
            40.0,
            grid_high.spacing_bps,
            grid_low.spacing_bps
        );
    }

    #[test]
    fn test_adaptive_spacing_pressure() {
        let mid = 30.0;
        let tick = 0.01;
        let cfg = default_cfg();

        let grid_relaxed = PriceGrid::for_current_state(mid, tick, 5.0, &cfg, 1.0);
        let grid_pressured = PriceGrid::for_current_state(mid, tick, 5.0, &cfg, 0.3);

        assert!(
            grid_pressured.spacing_bps > grid_relaxed.spacing_bps,
            "lower headroom ({}) should give wider spacing ({} vs {})",
            0.3,
            grid_pressured.spacing_bps,
            grid_relaxed.spacing_bps
        );
    }

    #[test]
    fn test_default_config() {
        let cfg = PriceGridConfig::default();
        assert!(!cfg.enabled, "default must be disabled for safe rollout");
        assert!(
            cfg.min_tick_multiple > 0.0,
            "min_tick_multiple must be positive"
        );
        assert!(cfg.sigma_divisor > 0.0, "sigma_divisor must be positive");
        assert!(
            cfg.pressure_scaling >= 0.0,
            "pressure_scaling must be non-negative"
        );
    }

    #[test]
    fn test_degenerate_inputs() {
        let cfg = default_cfg();

        // Zero mid — should not panic
        let grid = PriceGrid::for_current_state(0.0, 0.01, 5.0, &cfg, 1.0);
        let s = grid.snap(100.0);
        assert!(s.is_finite(), "snap with zero mid should produce finite result");

        // Zero tick — should return price unchanged
        let grid = PriceGrid::for_current_state(30.0, 0.0, 5.0, &cfg, 1.0);
        let p = 30.05;
        let s = grid.snap(p);
        assert_eq!(s, p, "snap with zero tick should return price unchanged");

        // Negative mid — should not panic
        let grid = PriceGrid::for_current_state(-10.0, 0.01, 5.0, &cfg, 1.0);
        let s = grid.snap(30.0);
        assert!(s.is_finite(), "snap with negative mid should produce finite result");

        // Zero sigma — should not panic, spacing falls back to tick-based
        let grid = PriceGrid::for_current_state(30.0, 0.01, 0.0, &cfg, 1.0);
        assert!(
            grid.spacing_bps > 0.0,
            "spacing should be positive even with zero sigma"
        );

        // Zero headroom — should widen, not panic
        let grid = PriceGrid::for_current_state(30.0, 0.01, 5.0, &cfg, 0.0);
        assert!(
            grid.spacing_bps > 0.0,
            "spacing should be positive with zero headroom"
        );
    }

    #[test]
    fn test_snap_bid_down_snap_ask_up() {
        let mid = 30.0;
        let tick = 0.01;
        let grid = PriceGrid::for_current_state(mid, tick, 5.0, &default_cfg(), 1.0);

        // A price between grid points: bid should snap down, ask should snap up
        let price = mid + grid.step * 0.3; // 30% into the first grid cell above mid
        let bid = grid.snap_bid(price);
        let ask = grid.snap_ask(price);

        assert!(
            bid <= price,
            "snap_bid({}) = {} should be <= price",
            price,
            bid
        );
        assert!(
            ask >= price,
            "snap_ask({}) = {} should be >= price",
            price,
            ask
        );
        assert!(
            bid < ask,
            "snap_bid ({}) should be < snap_ask ({}) for price between grid points",
            bid,
            ask
        );
    }

    #[test]
    fn test_snap_bid_ask_at_mid() {
        let mid = 30.0;
        let tick = 0.01;
        let grid = PriceGrid::for_current_state(mid, tick, 5.0, &default_cfg(), 1.0);

        // At exactly mid, floor(0 steps) = 0, ceil(0 steps) = 0 — both snap to mid
        let bid = grid.snap_bid(mid);
        let ask = grid.snap_ask(mid);

        // Both should resolve to mid (floor and ceil of 0 are both 0)
        assert!(
            (bid - mid).abs() < tick,
            "snap_bid at mid should be near mid, got {}",
            bid
        );
        assert!(
            (ask - mid).abs() < tick,
            "snap_ask at mid should be near mid, got {}",
            ask
        );
    }

    #[test]
    fn test_snap_bid_ask_tick_alignment() {
        let mid = 100.0;
        let tick = tick_for_mid(mid); // 0.01
        let grid = PriceGrid::for_current_state(mid, tick, 10.0, &default_cfg(), 1.0);

        for offset in [0.003, 0.007, 0.012, 0.025, 0.049, 0.1] {
            let p = mid + offset;

            let bid = grid.snap_bid(p);
            let bid_remainder = (bid / tick) - (bid / tick).round();
            assert!(
                bid_remainder.abs() < 1e-9,
                "snap_bid({}) = {} not on tick {}",
                p,
                bid,
                tick
            );

            let ask = grid.snap_ask(p);
            let ask_remainder = (ask / tick) - (ask / tick).round();
            assert!(
                ask_remainder.abs() < 1e-9,
                "snap_ask({}) = {} not on tick {}",
                p,
                ask,
                tick
            );
        }
    }

    #[test]
    fn test_snap_bid_ask_degenerate_inputs() {
        // Zero tick — should return price unchanged
        let grid = PriceGrid::for_current_state(30.0, 0.0, 5.0, &default_cfg(), 1.0);
        let p = 30.05;
        assert_eq!(
            grid.snap_bid(p),
            p,
            "snap_bid with zero tick should return price unchanged"
        );
        assert_eq!(
            grid.snap_ask(p),
            p,
            "snap_ask with zero tick should return price unchanged"
        );

        // Zero step — construct directly
        let grid = PriceGrid {
            spacing_bps: 0.0,
            reference_mid: 30.0,
            tick_size: 0.01,
            step: 0.0,
        };
        assert_eq!(
            grid.snap_bid(p),
            p,
            "snap_bid with zero step should return price unchanged"
        );
        assert_eq!(
            grid.snap_ask(p),
            p,
            "snap_ask with zero step should return price unchanged"
        );
    }

    #[test]
    fn test_snap_bid_always_le_snap_ask() {
        let mid = 30.0;
        let tick = 0.01;
        let grid = PriceGrid::for_current_state(mid, tick, 5.0, &default_cfg(), 1.0);

        // For many prices around mid, snap_bid should always be <= snap_ask
        for i in -20..=20 {
            let price = mid + (i as f64) * 0.007;
            let bid = grid.snap_bid(price);
            let ask = grid.snap_ask(price);
            assert!(
                bid <= ask,
                "snap_bid({}) = {} > snap_ask = {} — invariant violated",
                price,
                bid,
                ask
            );
        }
    }
}
