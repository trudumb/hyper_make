//! Binance Flow Analyzer - Parallel flow analysis for cross-venue signals.
//!
//! This module extracts stochastic flow features from Binance trades,
//! mirroring the features computed from Hyperliquid trades. The joint
//! analysis of both venues enables:
//!
//! - **Agreement detection**: Both venues showing same directional pressure
//! - **Divergence detection**: Venues disagree, indicating dislocation
//! - **Intensity ratio**: Where is price discovery happening?
//! - **Max toxicity**: Worst-case VPIN across venues
//!
//! # Architecture
//!
//! ```text
//! Binance @aggTrade → BinanceFlowAnalyzer → FlowFeatureVec
//!                                              │
//!                                              ▼
//!                                        CrossVenueAnalyzer
//!                                              │
//!                                              ▼
//!                                     BivariateFlowObservation
//! ```
//!
//! # Features Computed
//!
//! | Feature | Process | Meaning |
//! |---------|---------|---------|
//! | VPIN | Bucket-based | Toxicity probability |
//! | Volume imbalance | Rolling sum | Directional pressure |
//! | Trade intensity | Hawkes-like EWMA | Clustering, cascade risk |
//! | Avg trade size | EWMA by side | Informed vs noise |

use super::vpin::{VpinConfig, VpinEstimator};
use crate::market_maker::infra::BinanceTradeUpdate;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for Binance flow analyzer.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BinanceFlowConfig {
    /// VPIN bucket volume (in BTC equivalents).
    /// Binance has more liquidity, so larger buckets are appropriate.
    /// Default: 5.0 BTC
    ///
    /// NOTE: If vpin_adaptive is true, this becomes the initial bucket volume
    /// which is then dynamically adjusted based on observed trading activity.
    pub vpin_bucket_volume: f64,

    /// Number of VPIN buckets in rolling window.
    /// Default: 50
    pub vpin_n_buckets: usize,

    /// Enable adaptive VPIN bucket sizing.
    /// When enabled, bucket volume is dynamically adjusted to maintain
    /// approximately equal-time buckets (per Easley, Lopez de Prado, O'Hara).
    /// Default: false (for backward compatibility)
    #[serde(default)]
    pub vpin_adaptive: bool,

    /// Target average time per bucket in seconds (when adaptive is enabled).
    /// Default: 30.0 seconds
    #[serde(default = "default_vpin_target_seconds")]
    pub vpin_target_bucket_seconds: f64,

    /// Rolling window sizes for volume imbalance (milliseconds).
    /// Default: [1000, 5000, 30000, 300000] (1s, 5s, 30s, 5m)
    pub imbalance_windows_ms: [u64; 4],

    /// Intensity decay rate per second (Hawkes-like).
    /// Higher = faster decay, less memory.
    /// Default: 2.0 (half-life ~ 350ms)
    pub intensity_decay_rate: f64,

    /// EWMA alpha for trade size tracking.
    /// Default: 0.05 (slow adaptation)
    pub size_ema_alpha: f64,

    /// Maximum trades to keep in history buffer.
    /// Default: 1000
    pub max_trade_history: usize,
}

fn default_vpin_target_seconds() -> f64 {
    30.0
}

impl Default for BinanceFlowConfig {
    fn default() -> Self {
        Self {
            vpin_bucket_volume: 5.0,
            vpin_n_buckets: 50,
            vpin_adaptive: false,
            vpin_target_bucket_seconds: 30.0,
            imbalance_windows_ms: [1000, 5000, 30000, 300_000],
            intensity_decay_rate: 2.0,
            size_ema_alpha: 0.05,
            max_trade_history: 1000,
        }
    }
}

impl BinanceFlowConfig {
    /// Config for liquid BTC markets.
    pub fn btc() -> Self {
        Self::default()
    }

    /// Config for less liquid markets (smaller buckets).
    pub fn altcoin() -> Self {
        Self {
            vpin_bucket_volume: 1.0,
            vpin_n_buckets: 30,
            ..Default::default()
        }
    }

    /// Config with adaptive VPIN bucket sizing.
    ///
    /// This is the recommended config for production as it
    /// automatically calibrates bucket size to market conditions.
    ///
    /// # Arguments
    /// * `initial_bucket_volume` - Starting bucket size (will be adapted)
    /// * `target_bucket_seconds` - Target time per bucket (30-60s recommended)
    pub fn adaptive(initial_bucket_volume: f64, target_bucket_seconds: f64) -> Self {
        Self {
            vpin_bucket_volume: initial_bucket_volume,
            vpin_n_buckets: 50,
            vpin_adaptive: true,
            vpin_target_bucket_seconds: target_bucket_seconds,
            ..Default::default()
        }
    }
}

/// Flow features vector - standardized feature set for cross-venue comparison.
///
/// Both BinanceFlowAnalyzer and Hyperliquid estimators produce this
/// common feature vector for the CrossVenueAnalyzer.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlowFeatureVec {
    /// VPIN toxicity [0, 1]. Higher = more informed flow.
    pub vpin: f64,

    /// VPIN velocity (rate of change). Positive = toxicity increasing.
    pub vpin_velocity: f64,

    /// Volume imbalance at 1s window [-1, 1].
    /// Positive = buy pressure, negative = sell pressure.
    pub imbalance_1s: f64,

    /// Volume imbalance at 5s window [-1, 1].
    pub imbalance_5s: f64,

    /// Volume imbalance at 30s window [-1, 1].
    pub imbalance_30s: f64,

    /// Volume imbalance at 5m window [-1, 1].
    pub imbalance_5m: f64,

    /// Trade intensity (events per second, Hawkes-smoothed).
    pub intensity: f64,

    /// Average buy trade size (EWMA).
    pub avg_buy_size: f64,

    /// Average sell trade size (EWMA).
    pub avg_sell_size: f64,

    /// Size ratio: avg_buy / avg_sell. >1 = larger buys, <1 = larger sells.
    pub size_ratio: f64,

    /// Order flow direction [-1, 1] from VPIN (weighted recent buckets).
    pub order_flow_direction: f64,

    /// Timestamp of last update (ms).
    pub timestamp_ms: i64,

    /// Number of trades processed.
    pub trade_count: u64,

    /// Confidence [0, 1] based on data sufficiency.
    pub confidence: f64,
}

impl FlowFeatureVec {
    /// Get primary directional signal (uses 5s imbalance by default).
    pub fn directional_signal(&self) -> f64 {
        self.imbalance_5s
    }

    /// Check if features are valid (enough data).
    pub fn is_valid(&self) -> bool {
        self.confidence > 0.3 && self.trade_count >= 50
    }
}

/// Rolling volume accumulator for a specific time window.
#[derive(Debug)]
struct RollingVolume {
    window_ms: u64,
    buy_volume: VecDeque<(i64, f64)>,  // (timestamp_ms, volume)
    sell_volume: VecDeque<(i64, f64)>,
}

impl RollingVolume {
    fn new(window_ms: u64) -> Self {
        Self {
            window_ms,
            buy_volume: VecDeque::with_capacity(1000),
            sell_volume: VecDeque::with_capacity(1000),
        }
    }

    fn add(&mut self, timestamp_ms: i64, volume: f64, is_buy: bool) {
        let queue = if is_buy {
            &mut self.buy_volume
        } else {
            &mut self.sell_volume
        };
        queue.push_back((timestamp_ms, volume));

        // Prune old entries
        let cutoff = timestamp_ms - self.window_ms as i64;
        while let Some(&(ts, _)) = self.buy_volume.front() {
            if ts < cutoff {
                self.buy_volume.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(ts, _)) = self.sell_volume.front() {
            if ts < cutoff {
                self.sell_volume.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get volume imbalance [-1, 1].
    fn imbalance(&self) -> f64 {
        let buy_sum: f64 = self.buy_volume.iter().map(|(_, v)| v).sum();
        let sell_sum: f64 = self.sell_volume.iter().map(|(_, v)| v).sum();
        let total = buy_sum + sell_sum;
        if total < 1e-12 {
            0.0
        } else {
            (buy_sum - sell_sum) / total
        }
    }

    /// Get total volume in window.
    fn total_volume(&self) -> f64 {
        let buy_sum: f64 = self.buy_volume.iter().map(|(_, v)| v).sum();
        let sell_sum: f64 = self.sell_volume.iter().map(|(_, v)| v).sum();
        buy_sum + sell_sum
    }
}

/// EWMA tracker for trade sizes.
#[derive(Debug)]
struct SizeEwma {
    value: f64,
    alpha: f64,
    initialized: bool,
}

impl SizeEwma {
    fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha: alpha.clamp(0.001, 1.0),
            initialized: false,
        }
    }

    fn update(&mut self, size: f64) {
        if !self.initialized {
            self.value = size;
            self.initialized = true;
        } else {
            self.value = self.alpha * size + (1.0 - self.alpha) * self.value;
        }
    }

    fn value(&self) -> f64 {
        if self.initialized {
            self.value
        } else {
            1.0 // Default to prevent division by zero
        }
    }
}

/// Binance flow analyzer - extracts flow features from Binance trades.
#[derive(Debug)]
pub struct BinanceFlowAnalyzer {
    config: BinanceFlowConfig,

    /// VPIN estimator (reuses existing implementation)
    vpin: VpinEstimator,

    /// Rolling volume accumulators for different windows
    volume_1s: RollingVolume,
    volume_5s: RollingVolume,
    volume_30s: RollingVolume,
    volume_5m: RollingVolume,

    /// Trade intensity (Hawkes-like EWMA)
    intensity: f64,
    last_trade_time_ms: i64,

    /// Trade size tracking by side
    avg_buy_size: SizeEwma,
    avg_sell_size: SizeEwma,

    /// Trade count
    trade_count: u64,

    /// Latest mid price (for VPIN classification)
    latest_mid: f64,
}

impl BinanceFlowAnalyzer {
    /// Create a new Binance flow analyzer.
    pub fn new(config: BinanceFlowConfig) -> Self {
        let vpin_config = if config.vpin_adaptive {
            // Use adaptive VPIN configuration
            VpinConfig {
                bucket_volume: config.vpin_bucket_volume,
                n_buckets: config.vpin_n_buckets,
                velocity_alpha: 0.1,
                min_trades_per_bucket: 3, // Lower threshold for adaptive
                use_tick_rule: false,
                adaptive_enabled: true,
                target_bucket_seconds: config.vpin_target_bucket_seconds,
                volume_rate_alpha: 0.1,
                min_bucket_multiplier: 0.05,
                max_bucket_multiplier: 20.0,
                adaptive_warmup_buckets: 10,
            }
        } else {
            // Static bucket sizing
            VpinConfig {
                bucket_volume: config.vpin_bucket_volume,
                n_buckets: config.vpin_n_buckets,
                velocity_alpha: 0.1,
                min_trades_per_bucket: 5,
                use_tick_rule: false,
                ..VpinConfig::default()
            }
        };

        Self {
            volume_1s: RollingVolume::new(config.imbalance_windows_ms[0]),
            volume_5s: RollingVolume::new(config.imbalance_windows_ms[1]),
            volume_30s: RollingVolume::new(config.imbalance_windows_ms[2]),
            volume_5m: RollingVolume::new(config.imbalance_windows_ms[3]),
            vpin: VpinEstimator::new(vpin_config),
            intensity: 0.0,
            last_trade_time_ms: 0,
            avg_buy_size: SizeEwma::new(config.size_ema_alpha),
            avg_sell_size: SizeEwma::new(config.size_ema_alpha),
            trade_count: 0,
            latest_mid: 0.0,
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(BinanceFlowConfig::default())
    }

    /// Update latest mid price (for VPIN classification).
    pub fn set_mid_price(&mut self, mid: f64) {
        if mid > 0.0 {
            self.latest_mid = mid;
        }
    }

    /// Process a Binance trade update.
    pub fn on_trade(&mut self, trade: &BinanceTradeUpdate) {
        let timestamp_ms = trade.timestamp_ms;
        let size = trade.quantity;
        let price = trade.price;

        // is_buyer_maker = true means seller aggressed (sell trade)
        let is_buy = !trade.is_buyer_maker;

        // Update mid price from trade if not set
        if self.latest_mid <= 0.0 {
            self.latest_mid = price;
        }

        // Update VPIN
        self.vpin.on_trade(size, price, self.latest_mid, timestamp_ms as u64);

        // Update rolling volume accumulators
        self.volume_1s.add(timestamp_ms, size, is_buy);
        self.volume_5s.add(timestamp_ms, size, is_buy);
        self.volume_30s.add(timestamp_ms, size, is_buy);
        self.volume_5m.add(timestamp_ms, size, is_buy);

        // Update trade intensity (Hawkes-like exponential kernel)
        if self.last_trade_time_ms > 0 {
            let dt_secs = (timestamp_ms - self.last_trade_time_ms).max(1) as f64 / 1000.0;
            let decay = (-self.config.intensity_decay_rate * dt_secs).exp();
            self.intensity = self.intensity * decay + 1.0; // +1 for this trade
        } else {
            self.intensity = 1.0;
        }
        self.last_trade_time_ms = timestamp_ms;

        // Update size tracking
        if is_buy {
            self.avg_buy_size.update(size);
        } else {
            self.avg_sell_size.update(size);
        }

        self.trade_count += 1;
    }

    /// Get current VPIN toxicity [0, 1].
    pub fn vpin(&self) -> f64 {
        self.vpin.vpin()
    }

    /// Get VPIN velocity (rate of change).
    pub fn vpin_velocity(&self) -> f64 {
        self.vpin.vpin_velocity()
    }

    /// Get volume imbalance for a specific window.
    pub fn volume_imbalance(&self, window_idx: usize) -> f64 {
        match window_idx {
            0 => self.volume_1s.imbalance(),
            1 => self.volume_5s.imbalance(),
            2 => self.volume_30s.imbalance(),
            3 => self.volume_5m.imbalance(),
            _ => 0.0,
        }
    }

    /// Get current trade intensity (events per second equivalent).
    pub fn trade_intensity(&self) -> f64 {
        self.intensity
    }

    /// Get order flow direction from VPIN.
    pub fn order_flow_direction(&self) -> f64 {
        self.vpin.order_flow_direction()
    }

    /// Get complete flow feature vector.
    pub fn flow_features(&self) -> FlowFeatureVec {
        let avg_buy = self.avg_buy_size.value();
        let avg_sell = self.avg_sell_size.value();
        let size_ratio = if avg_sell > 1e-12 {
            avg_buy / avg_sell
        } else {
            1.0
        };

        // Confidence based on trade count and VPIN bucket count
        let trade_conf = (self.trade_count as f64 / 100.0).min(1.0);
        let vpin_conf = (self.vpin.bucket_count() as f64 / 10.0).min(1.0);
        let confidence = trade_conf * vpin_conf;

        FlowFeatureVec {
            vpin: self.vpin.vpin(),
            vpin_velocity: self.vpin.vpin_velocity(),
            imbalance_1s: self.volume_1s.imbalance(),
            imbalance_5s: self.volume_5s.imbalance(),
            imbalance_30s: self.volume_30s.imbalance(),
            imbalance_5m: self.volume_5m.imbalance(),
            intensity: self.intensity,
            avg_buy_size: avg_buy,
            avg_sell_size: avg_sell,
            size_ratio,
            order_flow_direction: self.vpin.order_flow_direction(),
            timestamp_ms: self.last_trade_time_ms,
            trade_count: self.trade_count,
            confidence,
        }
    }

    /// Check if analyzer is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.trade_count >= 50 && self.vpin.bucket_count() >= 5
    }

    /// Get total volume in 5m window.
    pub fn total_volume_5m(&self) -> f64 {
        self.volume_5m.total_volume()
    }

    /// Get trade count.
    pub fn trade_count(&self) -> u64 {
        self.trade_count
    }

    /// Reset the analyzer state.
    pub fn reset(&mut self) {
        self.vpin.reset();
        self.volume_1s = RollingVolume::new(self.config.imbalance_windows_ms[0]);
        self.volume_5s = RollingVolume::new(self.config.imbalance_windows_ms[1]);
        self.volume_30s = RollingVolume::new(self.config.imbalance_windows_ms[2]);
        self.volume_5m = RollingVolume::new(self.config.imbalance_windows_ms[3]);
        self.intensity = 0.0;
        self.last_trade_time_ms = 0;
        self.avg_buy_size = SizeEwma::new(self.config.size_ema_alpha);
        self.avg_sell_size = SizeEwma::new(self.config.size_ema_alpha);
        self.trade_count = 0;
        self.latest_mid = 0.0;
    }

    /// Get configuration.
    pub fn config(&self) -> &BinanceFlowConfig {
        &self.config
    }
}

impl Default for BinanceFlowAnalyzer {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(timestamp_ms: i64, price: f64, quantity: f64, is_buyer_maker: bool) -> BinanceTradeUpdate {
        BinanceTradeUpdate {
            timestamp_ms,
            price,
            quantity,
            is_buyer_maker,
            trade_id: timestamp_ms as u64,
        }
    }

    #[test]
    fn test_binance_flow_analyzer_basic() {
        let mut analyzer = BinanceFlowAnalyzer::default();

        // Feed some trades
        for i in 0..100 {
            let trade = make_trade(
                1000 + i * 10,
                50000.0 + (i as f64 * 0.1),
                0.1,
                i % 2 == 0, // Alternating buy/sell
            );
            analyzer.on_trade(&trade);
        }

        // Should be warmed up
        assert!(analyzer.trade_count() >= 100);

        // Get features
        let features = analyzer.flow_features();
        assert!(features.trade_count >= 100);
        assert!(features.imbalance_1s.abs() <= 1.0);
        assert!(features.intensity > 0.0);
    }

    #[test]
    fn test_volume_imbalance_buy_pressure() {
        let mut analyzer = BinanceFlowAnalyzer::default();

        // All buy trades (is_buyer_maker = false means buy aggressor)
        for i in 0..50 {
            let trade = make_trade(
                1000 + i * 10,
                50000.0,
                1.0,
                false, // Buy aggressor
            );
            analyzer.on_trade(&trade);
        }

        let features = analyzer.flow_features();
        // Should have positive imbalance (buy pressure)
        assert!(features.imbalance_1s > 0.5, "Expected buy pressure, got {}", features.imbalance_1s);
    }

    #[test]
    fn test_volume_imbalance_sell_pressure() {
        let mut analyzer = BinanceFlowAnalyzer::default();

        // All sell trades (is_buyer_maker = true means sell aggressor)
        for i in 0..50 {
            let trade = make_trade(
                1000 + i * 10,
                50000.0,
                1.0,
                true, // Sell aggressor
            );
            analyzer.on_trade(&trade);
        }

        let features = analyzer.flow_features();
        // Should have negative imbalance (sell pressure)
        assert!(features.imbalance_1s < -0.5, "Expected sell pressure, got {}", features.imbalance_1s);
    }

    #[test]
    fn test_trade_intensity() {
        let mut analyzer = BinanceFlowAnalyzer::default();

        // Rapid burst of trades
        for i in 0..20 {
            let trade = make_trade(
                1000 + i, // 1ms apart
                50000.0,
                0.1,
                false,
            );
            analyzer.on_trade(&trade);
        }

        let intensity_after_burst = analyzer.trade_intensity();

        // Wait 1 second (simulated)
        let trade = make_trade(2000, 50000.0, 0.1, false);
        analyzer.on_trade(&trade);

        let intensity_after_wait = analyzer.trade_intensity();

        // Intensity should be lower after waiting
        assert!(
            intensity_after_wait < intensity_after_burst,
            "Intensity should decay: {} -> {}",
            intensity_after_burst,
            intensity_after_wait
        );
    }

    #[test]
    fn test_size_tracking() {
        let mut analyzer = BinanceFlowAnalyzer::default();

        // Large buys, small sells
        for i in 0..50 {
            // Large buy
            let buy_trade = make_trade(1000 + i * 20, 50000.0, 10.0, false);
            analyzer.on_trade(&buy_trade);

            // Small sell
            let sell_trade = make_trade(1000 + i * 20 + 10, 50000.0, 1.0, true);
            analyzer.on_trade(&sell_trade);
        }

        let features = analyzer.flow_features();
        // Size ratio should be > 1 (larger buys than sells)
        assert!(
            features.size_ratio > 1.0,
            "Expected size_ratio > 1, got {}",
            features.size_ratio
        );
    }

    #[test]
    fn test_flow_feature_vec_validity() {
        let mut features = FlowFeatureVec::default();
        assert!(!features.is_valid()); // Not valid initially

        features.confidence = 0.5;
        features.trade_count = 100;
        assert!(features.is_valid()); // Now valid
    }

    #[test]
    fn test_reset() {
        let mut analyzer = BinanceFlowAnalyzer::default();

        // Add trades
        for i in 0..50 {
            let trade = make_trade(1000 + i * 10, 50000.0, 0.1, false);
            analyzer.on_trade(&trade);
        }

        assert!(analyzer.trade_count() > 0);

        // Reset
        analyzer.reset();

        assert_eq!(analyzer.trade_count(), 0);
        assert_eq!(analyzer.trade_intensity(), 0.0);
    }
}
