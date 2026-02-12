//! EWMA-based trade flow tracker for Hyperliquid native trade streams.
//!
//! Computes directional volume imbalance at multiple horizons (1s, 5s, 30s, 5m)
//! and average trade sizes from the HL trade stream. Replaces the hardcoded zeros
//! for `imbalance_30s`, `avg_buy_size`, `avg_sell_size`, `size_ratio` in
//! `FlowFeatureVec` (handlers.rs).
//!
//! # EWMA Design
//!
//! Alpha values approximate the responsiveness of each horizon:
//! - 1s:  alpha = 0.20 (fast, ~5 trades effective window)
//! - 5s:  alpha = 0.05 (~20 trades)
//! - 30s: alpha = 0.01 (~100 trades)
//! - 5m:  alpha = 0.002 (~500 trades)
//!
//! Imbalance formula: `(buy_ewma - sell_ewma) / (buy_ewma + sell_ewma + EPSILON)`
//! Returns values in [-1, 1] where positive = buy pressure.

/// Epsilon to prevent division by zero in imbalance calculations.
const EPSILON: f64 = 1e-12;

/// Minimum trade count before tracker is considered warmed up.
const MIN_WARMUP_TRADES: usize = 20;

/// EWMA alpha for ~1s horizon (fast).
const ALPHA_1S: f64 = 0.20;
/// EWMA alpha for ~5s horizon.
const ALPHA_5S: f64 = 0.05;
/// EWMA alpha for ~30s horizon.
const ALPHA_30S: f64 = 0.01;
/// EWMA alpha for ~5m horizon (slow).
const ALPHA_5M: f64 = 0.002;

/// Tracks trade flow with EWMA at multiple time horizons.
///
/// Provides imbalance, size ratios, and directional flow features
/// from the Hyperliquid trade stream. Each horizon uses a separate
/// EWMA pair (buy volume, sell volume) to capture flow dynamics
/// at different timescales.
#[derive(Debug, Clone)]
pub struct TradeFlowTracker {
    // EWMA buy/sell volume at each horizon
    buy_ewma_1s: f64,
    sell_ewma_1s: f64,
    buy_ewma_5s: f64,
    sell_ewma_5s: f64,
    buy_ewma_30s: f64,
    sell_ewma_30s: f64,
    buy_ewma_5m: f64,
    sell_ewma_5m: f64,

    // Total counts for warmup and average size
    trade_count: usize,
    total_buy_volume: f64,
    total_sell_volume: f64,
    total_buy_count: usize,
    total_sell_count: usize,
}

impl TradeFlowTracker {
    /// Create a new tracker with all state initialized to zero.
    pub fn new() -> Self {
        Self {
            buy_ewma_1s: 0.0,
            sell_ewma_1s: 0.0,
            buy_ewma_5s: 0.0,
            sell_ewma_5s: 0.0,
            buy_ewma_30s: 0.0,
            sell_ewma_30s: 0.0,
            buy_ewma_5m: 0.0,
            sell_ewma_5m: 0.0,
            trade_count: 0,
            total_buy_volume: 0.0,
            total_sell_volume: 0.0,
            total_buy_count: 0,
            total_sell_count: 0,
        }
    }

    /// Process a trade event, updating all EWMA horizons and cumulative stats.
    ///
    /// # Arguments
    /// * `size` - Absolute trade size in asset units (must be >= 0)
    /// * `is_buy` - Whether this trade was buyer-initiated
    pub fn on_trade(&mut self, size: f64, is_buy: bool) {
        let size = size.abs(); // Defensive: ensure non-negative
        self.trade_count += 1;

        if is_buy {
            self.total_buy_volume += size;
            self.total_buy_count += 1;

            self.buy_ewma_1s = Self::ewma_update(self.buy_ewma_1s, size, ALPHA_1S);
            self.sell_ewma_1s = Self::ewma_update(self.sell_ewma_1s, 0.0, ALPHA_1S);

            self.buy_ewma_5s = Self::ewma_update(self.buy_ewma_5s, size, ALPHA_5S);
            self.sell_ewma_5s = Self::ewma_update(self.sell_ewma_5s, 0.0, ALPHA_5S);

            self.buy_ewma_30s = Self::ewma_update(self.buy_ewma_30s, size, ALPHA_30S);
            self.sell_ewma_30s = Self::ewma_update(self.sell_ewma_30s, 0.0, ALPHA_30S);

            self.buy_ewma_5m = Self::ewma_update(self.buy_ewma_5m, size, ALPHA_5M);
            self.sell_ewma_5m = Self::ewma_update(self.sell_ewma_5m, 0.0, ALPHA_5M);
        } else {
            self.total_sell_volume += size;
            self.total_sell_count += 1;

            self.buy_ewma_1s = Self::ewma_update(self.buy_ewma_1s, 0.0, ALPHA_1S);
            self.sell_ewma_1s = Self::ewma_update(self.sell_ewma_1s, size, ALPHA_1S);

            self.buy_ewma_5s = Self::ewma_update(self.buy_ewma_5s, 0.0, ALPHA_5S);
            self.sell_ewma_5s = Self::ewma_update(self.sell_ewma_5s, size, ALPHA_5S);

            self.buy_ewma_30s = Self::ewma_update(self.buy_ewma_30s, 0.0, ALPHA_30S);
            self.sell_ewma_30s = Self::ewma_update(self.sell_ewma_30s, size, ALPHA_30S);

            self.buy_ewma_5m = Self::ewma_update(self.buy_ewma_5m, 0.0, ALPHA_5M);
            self.sell_ewma_5m = Self::ewma_update(self.sell_ewma_5m, size, ALPHA_5M);
        }
    }

    /// Trade flow imbalance at the 1s horizon, in [-1, 1].
    pub fn imbalance_at_1s(&self) -> f64 {
        Self::imbalance(self.buy_ewma_1s, self.sell_ewma_1s)
    }

    /// Trade flow imbalance at the 5s horizon, in [-1, 1].
    pub fn imbalance_at_5s(&self) -> f64 {
        Self::imbalance(self.buy_ewma_5s, self.sell_ewma_5s)
    }

    /// Trade flow imbalance at the 30s horizon, in [-1, 1].
    pub fn imbalance_at_30s(&self) -> f64 {
        Self::imbalance(self.buy_ewma_30s, self.sell_ewma_30s)
    }

    /// Trade flow imbalance at the 5m horizon, in [-1, 1].
    pub fn imbalance_at_5m(&self) -> f64 {
        Self::imbalance(self.buy_ewma_5m, self.sell_ewma_5m)
    }

    /// Average buy trade size (total buy volume / buy count).
    /// Returns 0.0 if no buys have been recorded.
    pub fn avg_buy_size(&self) -> f64 {
        if self.total_buy_count == 0 {
            return 0.0;
        }
        self.total_buy_volume / self.total_buy_count as f64
    }

    /// Average sell trade size (total sell volume / sell count).
    /// Returns 0.0 if no sells have been recorded.
    pub fn avg_sell_size(&self) -> f64 {
        if self.total_sell_count == 0 {
            return 0.0;
        }
        self.total_sell_volume / self.total_sell_count as f64
    }

    /// Size ratio: avg_buy / (avg_buy + avg_sell + epsilon).
    /// Centered at 0.5 when buy and sell sizes are equal.
    /// Returns 0.5 if no trades have been recorded.
    pub fn size_ratio(&self) -> f64 {
        let avg_buy = self.avg_buy_size();
        let avg_sell = self.avg_sell_size();
        if avg_buy == 0.0 && avg_sell == 0.0 {
            return 0.5;
        }
        avg_buy / (avg_buy + avg_sell + EPSILON)
    }

    /// Whether the tracker has seen enough trades to produce reliable estimates.
    pub fn is_warmed_up(&self) -> bool {
        self.trade_count >= MIN_WARMUP_TRADES
    }

    /// Total number of trades processed.
    pub fn trade_count(&self) -> usize {
        self.trade_count
    }

    // ---- Internal helpers ----

    /// Standard EWMA update: new = alpha * observation + (1 - alpha) * old.
    #[inline]
    fn ewma_update(old: f64, observation: f64, alpha: f64) -> f64 {
        alpha * observation + (1.0 - alpha) * old
    }

    /// Compute normalized imbalance from buy/sell EWMA values.
    /// Returns value in [-1, 1].
    #[inline]
    fn imbalance(buy: f64, sell: f64) -> f64 {
        (buy - sell) / (buy + sell + EPSILON)
    }
}

impl Default for TradeFlowTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tracker_all_zeros() {
        let tracker = TradeFlowTracker::new();
        assert_eq!(tracker.imbalance_at_1s(), 0.0);
        assert_eq!(tracker.imbalance_at_5s(), 0.0);
        assert_eq!(tracker.imbalance_at_30s(), 0.0);
        assert_eq!(tracker.imbalance_at_5m(), 0.0);
        assert_eq!(tracker.avg_buy_size(), 0.0);
        assert_eq!(tracker.avg_sell_size(), 0.0);
        assert_eq!(tracker.size_ratio(), 0.5);
        assert!(!tracker.is_warmed_up());
        assert_eq!(tracker.trade_count(), 0);
    }

    #[test]
    fn test_buy_heavy_positive_imbalance() {
        let mut tracker = TradeFlowTracker::new();

        // 10 buys of size 1.0
        for _ in 0..10 {
            tracker.on_trade(1.0, true);
        }
        // 2 sells of size 1.0
        for _ in 0..2 {
            tracker.on_trade(1.0, false);
        }

        // All horizons should show positive imbalance (buy pressure)
        assert!(
            tracker.imbalance_at_1s() > 0.0,
            "1s imbalance should be positive, got {}",
            tracker.imbalance_at_1s()
        );
        assert!(
            tracker.imbalance_at_5s() > 0.0,
            "5s imbalance should be positive, got {}",
            tracker.imbalance_at_5s()
        );
        assert!(
            tracker.imbalance_at_30s() > 0.0,
            "30s imbalance should be positive, got {}",
            tracker.imbalance_at_30s()
        );
        assert!(
            tracker.imbalance_at_5m() > 0.0,
            "5m imbalance should be positive, got {}",
            tracker.imbalance_at_5m()
        );
    }

    #[test]
    fn test_sell_heavy_negative_imbalance() {
        let mut tracker = TradeFlowTracker::new();

        // 2 buys then 10 sells
        for _ in 0..2 {
            tracker.on_trade(1.0, true);
        }
        for _ in 0..10 {
            tracker.on_trade(1.0, false);
        }

        // All horizons should show negative imbalance (sell pressure)
        assert!(
            tracker.imbalance_at_1s() < 0.0,
            "1s imbalance should be negative, got {}",
            tracker.imbalance_at_1s()
        );
        assert!(
            tracker.imbalance_at_5s() < 0.0,
            "5s imbalance should be negative, got {}",
            tracker.imbalance_at_5s()
        );
        assert!(
            tracker.imbalance_at_30s() < 0.0,
            "30s imbalance should be negative, got {}",
            tracker.imbalance_at_30s()
        );
        assert!(
            tracker.imbalance_at_5m() < 0.0,
            "5m imbalance should be negative, got {}",
            tracker.imbalance_at_5m()
        );
    }

    #[test]
    fn test_imbalance_bounded() {
        let mut tracker = TradeFlowTracker::new();

        // All buys — should approach +1 but never exceed
        for _ in 0..100 {
            tracker.on_trade(1.0, true);
        }
        assert!(tracker.imbalance_at_1s() <= 1.0);
        assert!(tracker.imbalance_at_1s() > 0.9); // With 100 buys, 1s should be near 1

        // Reset and all sells — should approach -1
        let mut tracker2 = TradeFlowTracker::new();
        for _ in 0..100 {
            tracker2.on_trade(1.0, false);
        }
        assert!(tracker2.imbalance_at_1s() >= -1.0);
        assert!(tracker2.imbalance_at_1s() < -0.9);
    }

    #[test]
    fn test_warmup_threshold() {
        let mut tracker = TradeFlowTracker::new();
        assert!(!tracker.is_warmed_up());

        for i in 0..MIN_WARMUP_TRADES {
            tracker.on_trade(1.0, i % 2 == 0);
            if i < MIN_WARMUP_TRADES - 1 {
                assert!(!tracker.is_warmed_up());
            }
        }
        assert!(tracker.is_warmed_up());
        assert_eq!(tracker.trade_count(), MIN_WARMUP_TRADES);
    }

    #[test]
    fn test_avg_buy_sell_size() {
        let mut tracker = TradeFlowTracker::new();

        // 3 buys of varying sizes
        tracker.on_trade(2.0, true);
        tracker.on_trade(4.0, true);
        tracker.on_trade(6.0, true);
        // 2 sells of varying sizes
        tracker.on_trade(1.0, false);
        tracker.on_trade(3.0, false);

        let expected_avg_buy = (2.0 + 4.0 + 6.0) / 3.0;
        let expected_avg_sell = (1.0 + 3.0) / 2.0;

        assert!(
            (tracker.avg_buy_size() - expected_avg_buy).abs() < 1e-10,
            "avg_buy_size: expected {}, got {}",
            expected_avg_buy,
            tracker.avg_buy_size()
        );
        assert!(
            (tracker.avg_sell_size() - expected_avg_sell).abs() < 1e-10,
            "avg_sell_size: expected {}, got {}",
            expected_avg_sell,
            tracker.avg_sell_size()
        );
    }

    #[test]
    fn test_size_ratio_centered() {
        let mut tracker = TradeFlowTracker::new();

        // Equal size buys and sells
        for _ in 0..10 {
            tracker.on_trade(1.0, true);
            tracker.on_trade(1.0, false);
        }

        // size_ratio should be ~0.5
        assert!(
            (tracker.size_ratio() - 0.5).abs() < 0.01,
            "size_ratio should be ~0.5, got {}",
            tracker.size_ratio()
        );
    }

    #[test]
    fn test_size_ratio_asymmetric() {
        let mut tracker = TradeFlowTracker::new();

        // Large buys, small sells
        for _ in 0..5 {
            tracker.on_trade(10.0, true);
            tracker.on_trade(1.0, false);
        }

        // avg_buy=10.0, avg_sell=1.0, ratio = 10/(10+1) ≈ 0.909
        assert!(
            tracker.size_ratio() > 0.8,
            "size_ratio should be > 0.8 for large buys, got {}",
            tracker.size_ratio()
        );
    }

    #[test]
    fn test_faster_horizons_respond_faster() {
        let mut tracker = TradeFlowTracker::new();

        // Build up some sell pressure first
        for _ in 0..50 {
            tracker.on_trade(1.0, false);
        }

        // Now flip to buy pressure
        for _ in 0..10 {
            tracker.on_trade(1.0, true);
        }

        // The faster horizon (1s) should have more positive imbalance
        // than the slower horizon (5m) after the flip
        assert!(
            tracker.imbalance_at_1s() > tracker.imbalance_at_5m(),
            "1s ({}) should respond faster than 5m ({}) to regime change",
            tracker.imbalance_at_1s(),
            tracker.imbalance_at_5m()
        );
        assert!(
            tracker.imbalance_at_5s() > tracker.imbalance_at_30s(),
            "5s ({}) should respond faster than 30s ({})",
            tracker.imbalance_at_5s(),
            tracker.imbalance_at_30s()
        );
    }

    #[test]
    fn test_negative_size_treated_as_positive() {
        let mut tracker = TradeFlowTracker::new();
        tracker.on_trade(-5.0, true);
        assert!(
            (tracker.avg_buy_size() - 5.0).abs() < 1e-10,
            "Negative size should be abs'd: got {}",
            tracker.avg_buy_size()
        );
    }

    #[test]
    fn test_default_impl() {
        let tracker = TradeFlowTracker::default();
        assert_eq!(tracker.trade_count(), 0);
        assert!(!tracker.is_warmed_up());
    }

    #[test]
    fn test_ewma_decay_on_opposite_trades() {
        let mut tracker = TradeFlowTracker::new();

        // Single buy of size 10
        tracker.on_trade(10.0, true);
        let imb_after_buy = tracker.imbalance_at_1s();
        assert!(imb_after_buy > 0.9); // Should be strongly positive

        // Many sells of size 0.1 — buy EWMA decays, sell EWMA grows
        for _ in 0..50 {
            tracker.on_trade(0.1, false);
        }

        // Imbalance should have decayed and possibly flipped negative
        assert!(
            tracker.imbalance_at_1s() < imb_after_buy,
            "Imbalance should decay after opposing trades"
        );
    }
}
