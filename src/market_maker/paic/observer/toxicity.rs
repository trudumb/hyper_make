//! Flow Toxicity Estimation for PAIC.
//!
//! Estimates the probability that recent order flow is "toxic" (informed trading).
//!
//! Key signals:
//! - VPIN (Volume-synchronized Probability of INformed trading)
//! - OFI (Order Flow Imbalance)
//! - Trade clustering (bursts of one-sided flow)

use std::collections::VecDeque;
use std::time::Instant;

/// Configuration for toxicity estimation.
#[derive(Debug, Clone)]
pub struct ToxicityConfig {
    /// Window for OFI calculation (milliseconds)
    pub ofi_window_ms: u64,
    /// VPIN bucket size (volume units)
    pub vpin_bucket_volume: f64,
    /// Number of buckets for VPIN
    pub vpin_num_buckets: usize,
    /// EWMA alpha for smoothing
    pub ewma_alpha: f64,
    /// Threshold for "toxic" classification
    pub toxic_threshold: f64,
}

impl Default for ToxicityConfig {
    fn default() -> Self {
        Self {
            ofi_window_ms: 1000,   // 1 second
            vpin_bucket_volume: 1.0, // 1 BTC equivalent
            vpin_num_buckets: 50,
            ewma_alpha: 0.1,
            toxic_threshold: 0.1,
        }
    }
}

/// Trade record for toxicity calculation.
#[derive(Debug, Clone)]
struct TradeRecord {
    timestamp: Instant,
    volume: f64,
    is_buy: bool,
}

/// Toxicity estimator using OFI and VPIN-style metrics.
#[derive(Debug)]
pub struct ToxicityEstimator {
    config: ToxicityConfig,

    /// Recent trades for OFI
    trades: VecDeque<TradeRecord>,

    /// Volume buckets for VPIN: (buy_volume, sell_volume) per bucket
    vpin_buckets: VecDeque<(f64, f64)>,

    /// Current bucket accumulator
    current_bucket_buy: f64,
    current_bucket_sell: f64,
    current_bucket_total: f64,

    /// Smoothed toxicity score [0, 1]
    toxicity: f64,

    /// Smoothed OFI [-1, 1]
    ofi: f64,

    /// Is current flow considered toxic?
    is_toxic: bool,
}

impl ToxicityEstimator {
    /// Create a new toxicity estimator.
    pub fn new(config: ToxicityConfig) -> Self {
        Self {
            config,
            trades: VecDeque::with_capacity(1000),
            vpin_buckets: VecDeque::with_capacity(50),
            current_bucket_buy: 0.0,
            current_bucket_sell: 0.0,
            current_bucket_total: 0.0,
            toxicity: 0.0,
            ofi: 0.0,
            is_toxic: false,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(ToxicityConfig::default())
    }

    /// Process a trade.
    pub fn on_trade(&mut self, volume: f64, is_buy: bool) {
        let now = Instant::now();

        // Record trade
        self.trades.push_back(TradeRecord {
            timestamp: now,
            volume,
            is_buy,
        });

        // Update bucket accumulators
        if is_buy {
            self.current_bucket_buy += volume;
        } else {
            self.current_bucket_sell += volume;
        }
        self.current_bucket_total += volume;

        // Check if bucket is complete
        if self.current_bucket_total >= self.config.vpin_bucket_volume {
            self.complete_bucket();
        }

        // Expire old trades
        let window_secs = self.config.ofi_window_ms as f64 / 1000.0;
        while let Some(front) = self.trades.front() {
            if now.duration_since(front.timestamp).as_secs_f64() > window_secs {
                self.trades.pop_front();
            } else {
                break;
            }
        }

        // Update metrics
        self.update_ofi();
        self.update_toxicity();
    }

    /// Complete a VPIN bucket.
    fn complete_bucket(&mut self) {
        // Push completed bucket
        self.vpin_buckets.push_back((self.current_bucket_buy, self.current_bucket_sell));

        // Maintain bucket window
        while self.vpin_buckets.len() > self.config.vpin_num_buckets {
            self.vpin_buckets.pop_front();
        }

        // Reset accumulators
        self.current_bucket_buy = 0.0;
        self.current_bucket_sell = 0.0;
        self.current_bucket_total = 0.0;
    }

    /// Update OFI (Order Flow Imbalance).
    fn update_ofi(&mut self) {
        if self.trades.is_empty() {
            return;
        }

        let (buy_vol, sell_vol): (f64, f64) = self.trades.iter().fold((0.0, 0.0), |(b, s), t| {
            if t.is_buy {
                (b + t.volume, s)
            } else {
                (b, s + t.volume)
            }
        });

        let total = buy_vol + sell_vol;
        if total > 0.0 {
            let raw_ofi = (buy_vol - sell_vol) / total;
            // EWMA smoothing
            self.ofi = self.config.ewma_alpha * raw_ofi + (1.0 - self.config.ewma_alpha) * self.ofi;
        }
    }

    /// Update toxicity score.
    fn update_toxicity(&mut self) {
        // VPIN-style toxicity: sum of absolute imbalances / total volume
        if self.vpin_buckets.is_empty() {
            return;
        }

        let (sum_imbalance, total_volume): (f64, f64) = self
            .vpin_buckets
            .iter()
            .fold((0.0, 0.0), |(imb, vol), (buy, sell)| {
                let bucket_total = buy + sell;
                let bucket_imbalance = (buy - sell).abs();
                (imb + bucket_imbalance, vol + bucket_total)
            });

        if total_volume > 0.0 {
            let raw_toxicity = sum_imbalance / total_volume;
            // EWMA smoothing
            self.toxicity =
                self.config.ewma_alpha * raw_toxicity + (1.0 - self.config.ewma_alpha) * self.toxicity;
        }

        // Update toxic flag
        self.is_toxic = self.toxicity > self.config.toxic_threshold;
    }

    /// Get current toxicity score [0, 1].
    pub fn toxicity(&self) -> f64 {
        self.toxicity
    }

    /// Get current OFI [-1, 1].
    pub fn ofi(&self) -> f64 {
        self.ofi
    }

    /// Check if current flow is toxic.
    pub fn is_toxic(&self) -> bool {
        self.is_toxic
    }

    /// Get signed toxicity (direction × magnitude).
    ///
    /// Positive = toxic buying, Negative = toxic selling.
    pub fn signed_toxicity(&self) -> f64 {
        self.ofi.signum() * self.toxicity
    }

    /// Check if flow is toxic for a specific side.
    ///
    /// `is_our_bid`: true if we're asking about our bid order.
    /// Returns true if toxic flow is likely to hit our order adversely.
    pub fn is_toxic_for_side(&self, is_our_bid: bool) -> bool {
        if !self.is_toxic {
            return false;
        }
        // Toxic selling (OFI < 0) is bad for bids
        // Toxic buying (OFI > 0) is bad for asks
        (is_our_bid && self.ofi < -0.05) || (!is_our_bid && self.ofi > 0.05)
    }

    /// Get number of trades in window.
    pub fn trade_count(&self) -> usize {
        self.trades.len()
    }

    /// Get number of VPIN buckets.
    pub fn bucket_count(&self) -> usize {
        self.vpin_buckets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toxicity_estimator_new() {
        let estimator = ToxicityEstimator::default_config();
        assert!((estimator.toxicity() - 0.0).abs() < 0.01);
        assert!((estimator.ofi() - 0.0).abs() < 0.01);
        assert!(!estimator.is_toxic());
    }

    #[test]
    fn test_balanced_flow() {
        let mut estimator = ToxicityEstimator::default_config();

        // Balanced buy/sell flow
        for _ in 0..10 {
            estimator.on_trade(0.1, true);
            estimator.on_trade(0.1, false);
        }

        // OFI should be near zero
        assert!(estimator.ofi().abs() < 0.2);
        // Not toxic
        assert!(!estimator.is_toxic());
    }

    #[test]
    fn test_toxic_buying() {
        let mut estimator = ToxicityEstimator::default_config();

        // Heavy one-sided buying
        for _ in 0..50 {
            estimator.on_trade(0.1, true);
        }

        // OFI should be positive
        assert!(estimator.ofi() > 0.5);
        // Should be toxic
        assert!(estimator.toxicity() > 0.5);
    }

    #[test]
    fn test_toxic_for_side() {
        let mut estimator = ToxicityEstimator::new(ToxicityConfig {
            toxic_threshold: 0.1,
            ..Default::default()
        });

        // Heavy selling (toxic for bids)
        for _ in 0..50 {
            estimator.on_trade(0.1, false);
        }

        // Should be toxic for bids
        assert!(estimator.is_toxic_for_side(true));
        // Not toxic for asks
        assert!(!estimator.is_toxic_for_side(false));
    }
}
