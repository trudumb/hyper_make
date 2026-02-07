//! VPIN (Volume-Synchronized Probability of Informed Trading) Estimator
//!
//! VPIN is a real-time estimate of order flow toxicity based on the idea that
//! informed traders cluster their trades to exploit information advantages.
//!
//! ## Theory
//!
//! VPIN buckets trades by volume (not time) and classifies buy/sell volume using
//! the bulk volume classification method. The key insight is that informed flow
//! tends to be unidirectional within a bucket.
//!
//! ```text
//! VPIN = Σ|V_buy - V_sell| / (n_buckets × bucket_volume)
//! ```
//!
//! High VPIN (>0.7) indicates toxic flow where informed traders dominate.
//! Low VPIN (<0.3) indicates noise-dominated markets (good for market making).
//!
//! ## Usage
//!
//! ```ignore
//! let mut vpin = VpinEstimator::new(VpinConfig::default());
//!
//! // On each trade
//! if let Some(vpin_value) = vpin.on_trade(size, price, mid_price) {
//!     // Bucket completed, new VPIN value available
//!     println!("VPIN: {:.2}", vpin_value);
//! }
//!
//! // Access current estimates
//! let toxicity = vpin.vpin();
//! let velocity = vpin.vpin_velocity();
//! ```
//!
//! ## References
//!
//! - Easley, Lopez de Prado, O'Hara (2012): "The Volume Clock"
//! - Easley, Lopez de Prado, O'Hara (2011): "The Microstructure of the Flash Crash"

use std::collections::VecDeque;

/// Configuration for VPIN estimator.
#[derive(Debug, Clone)]
pub struct VpinConfig {
    /// Volume per bucket (in base asset units).
    /// Smaller buckets = more responsive, noisier.
    /// Larger buckets = smoother, slower to react.
    /// Default: 1.0 (1 BTC or equivalent)
    ///
    /// NOTE: If adaptive_enabled is true, this becomes the initial bucket volume
    /// which is then dynamically adjusted based on observed trading activity.
    pub bucket_volume: f64,

    /// Number of buckets in rolling window.
    /// VPIN is computed over this many recent buckets.
    /// Default: 50
    pub n_buckets: usize,

    /// EWMA alpha for velocity calculation.
    /// Default: 0.1
    pub velocity_alpha: f64,

    /// Minimum trades per bucket for validity.
    /// Buckets with fewer trades are marked low-confidence.
    /// Default: 5
    pub min_trades_per_bucket: usize,

    /// Whether to use tick rule for buy/sell classification.
    /// If false, uses bulk volume classification (price vs mid).
    /// Default: false (use bulk classification)
    pub use_tick_rule: bool,

    /// Enable adaptive bucket sizing based on trading volume.
    /// When enabled, bucket_volume is dynamically adjusted to target
    /// approximately equal-time buckets (per Easley, Lopez de Prado, O'Hara).
    /// Default: false
    pub adaptive_enabled: bool,

    /// Target average time per bucket in seconds.
    /// The algorithm adjusts bucket_volume to achieve this target.
    /// Academic default: 30-60 seconds per bucket.
    /// Default: 30.0
    pub target_bucket_seconds: f64,

    /// EWMA alpha for volume rate estimation.
    /// Lower = smoother adaptation, higher = faster response.
    /// Default: 0.1
    pub volume_rate_alpha: f64,

    /// Minimum bucket volume as fraction of initial.
    /// Prevents buckets from becoming too small (noisy).
    /// Default: 0.05 (5% of initial)
    pub min_bucket_multiplier: f64,

    /// Maximum bucket volume as fraction of initial.
    /// Prevents buckets from becoming too large (unresponsive).
    /// Default: 20.0 (20x initial)
    pub max_bucket_multiplier: f64,

    /// Minimum buckets before adaptive sizing kicks in.
    /// Allows warm-up period with initial bucket size.
    /// Default: 10
    pub adaptive_warmup_buckets: usize,

    /// Maximum age of most recent bucket before VPIN is considered stale (ms).
    /// Default: 60_000 (60 seconds). Set to 0 to disable staleness check.
    pub staleness_threshold_ms: u64,
}

impl Default for VpinConfig {
    fn default() -> Self {
        Self {
            bucket_volume: 1.0,
            n_buckets: 50,
            velocity_alpha: 0.1,
            min_trades_per_bucket: 5,
            use_tick_rule: false,
            // Adaptive bucket sizing (disabled by default for backward compatibility)
            adaptive_enabled: false,
            target_bucket_seconds: 30.0,
            volume_rate_alpha: 0.1,
            min_bucket_multiplier: 0.05,
            max_bucket_multiplier: 20.0,
            adaptive_warmup_buckets: 10,
            staleness_threshold_ms: 60_000,
        }
    }
}

impl VpinConfig {
    /// Config for liquid markets (BTC/ETH on major exchanges).
    pub fn liquid() -> Self {
        Self {
            bucket_volume: 5.0,
            n_buckets: 50,
            min_trades_per_bucket: 10,
            ..Default::default()
        }
    }

    /// Config for less liquid markets (altcoins, DEX).
    pub fn illiquid() -> Self {
        Self {
            bucket_volume: 0.5,
            n_buckets: 30,
            min_trades_per_bucket: 3,
            ..Default::default()
        }
    }

    /// Config for high-frequency (faster response).
    pub fn high_frequency() -> Self {
        Self {
            bucket_volume: 0.2,
            n_buckets: 100,
            velocity_alpha: 0.2,
            min_trades_per_bucket: 2,
            ..Default::default()
        }
    }

    /// Config with adaptive bucket sizing enabled.
    ///
    /// This is the recommended config for production use as it
    /// automatically calibrates bucket size to market conditions.
    ///
    /// # Arguments
    /// * `initial_bucket_volume` - Starting bucket size (will be adapted)
    /// * `target_bucket_seconds` - Target time per bucket (30-60s recommended)
    pub fn adaptive(initial_bucket_volume: f64, target_bucket_seconds: f64) -> Self {
        Self {
            bucket_volume: initial_bucket_volume,
            n_buckets: 50,
            min_trades_per_bucket: 3,
            velocity_alpha: 0.1,
            use_tick_rule: false,
            adaptive_enabled: true,
            target_bucket_seconds,
            volume_rate_alpha: 0.1,
            min_bucket_multiplier: 0.05,
            max_bucket_multiplier: 20.0,
            adaptive_warmup_buckets: 10,
            staleness_threshold_ms: 60_000,
        }
    }
}

/// A completed volume bucket.
#[derive(Debug, Clone, Copy)]
struct Bucket {
    /// Buy volume in this bucket
    buy_volume: f64,
    /// Total volume in this bucket
    total_volume: f64,
    /// Number of trades in bucket
    n_trades: usize,
    /// Timestamp when bucket completed (ms)
    completed_at_ms: u64,
}

impl Bucket {
    /// Order imbalance for this bucket: |V_buy - V_sell| / V_total
    fn imbalance(&self) -> f64 {
        if self.total_volume < 1e-12 {
            return 0.0;
        }
        let sell_volume = self.total_volume - self.buy_volume;
        (self.buy_volume - sell_volume).abs() / self.total_volume
    }

    /// Signed imbalance: positive = buy pressure, negative = sell pressure
    fn signed_imbalance(&self) -> f64 {
        if self.total_volume < 1e-12 {
            return 0.0;
        }
        let sell_volume = self.total_volume - self.buy_volume;
        (self.buy_volume - sell_volume) / self.total_volume
    }
}

/// VPIN estimator state.
#[derive(Debug)]
pub struct VpinEstimator {
    config: VpinConfig,

    /// Completed buckets (ring buffer)
    buckets: VecDeque<Bucket>,

    /// Current bucket accumulation
    current_buy_volume: f64,
    current_total_volume: f64,
    current_n_trades: usize,

    /// Previous VPIN value (for velocity calculation)
    prev_vpin: f64,

    /// VPIN velocity (rate of change)
    vpin_velocity: f64,

    /// Total trades processed
    total_trades: u64,

    /// Total buckets completed
    total_buckets: u64,

    /// Last trade timestamp
    last_trade_ms: u64,

    /// Last trade price (for tick rule)
    last_price: Option<f64>,

    // === Adaptive bucket sizing state ===
    /// Initial bucket volume (used for clamping)
    initial_bucket_volume: f64,

    /// EWMA of volume rate (volume per second)
    volume_rate_ewma: f64,

    /// Start timestamp of current bucket (for duration calculation)
    current_bucket_start_ms: u64,

    /// Current effective bucket volume (may be adapted)
    effective_bucket_volume: f64,
}

impl VpinEstimator {
    /// Create a new VPIN estimator.
    pub fn new(config: VpinConfig) -> Self {
        let initial_bucket_volume = config.bucket_volume;
        Self {
            buckets: VecDeque::with_capacity(config.n_buckets + 1),
            current_buy_volume: 0.0,
            current_total_volume: 0.0,
            current_n_trades: 0,
            prev_vpin: 0.0,
            vpin_velocity: 0.0,
            total_trades: 0,
            total_buckets: 0,
            last_trade_ms: 0,
            last_price: None,
            // Adaptive state
            initial_bucket_volume,
            volume_rate_ewma: 0.0,
            current_bucket_start_ms: 0,
            effective_bucket_volume: initial_bucket_volume,
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(VpinConfig::default())
    }

    /// Process a trade and potentially complete a bucket.
    ///
    /// # Arguments
    /// * `size` - Trade size (always positive)
    /// * `price` - Trade price
    /// * `mid` - Mid price at time of trade
    /// * `timestamp_ms` - Trade timestamp (epoch ms)
    ///
    /// # Returns
    /// Some(vpin) if a bucket was completed, None otherwise.
    pub fn on_trade(&mut self, size: f64, price: f64, mid: f64, timestamp_ms: u64) -> Option<f64> {
        if size <= 0.0 {
            return None;
        }

        // Track bucket start time for adaptive sizing
        if self.current_bucket_start_ms == 0 {
            self.current_bucket_start_ms = timestamp_ms;
        }

        // Classify as buy or sell
        let is_buy = self.classify_trade(price, mid);

        // Update current bucket
        if is_buy {
            self.current_buy_volume += size;
        }
        self.current_total_volume += size;
        self.current_n_trades += 1;
        self.total_trades += 1;
        self.last_trade_ms = timestamp_ms;
        self.last_price = Some(price);

        // Use effective bucket volume (may be adapted)
        let bucket_volume = self.effective_bucket_volume;

        // Check if bucket is complete
        let mut bucket_completed = false;
        while self.current_total_volume >= bucket_volume {
            // How much volume goes into this bucket?
            let overflow = self.current_total_volume - bucket_volume;

            // Proportionally split buy volume if there's overflow
            let bucket_buy = if overflow > 1e-12 {
                let bucket_frac = bucket_volume / self.current_total_volume;
                self.current_buy_volume * bucket_frac
            } else {
                self.current_buy_volume
            };

            // Calculate bucket duration for adaptive sizing
            let bucket_duration_ms = if self.current_bucket_start_ms > 0 {
                timestamp_ms.saturating_sub(self.current_bucket_start_ms)
            } else {
                0
            };

            // Complete the bucket
            let bucket = Bucket {
                buy_volume: bucket_buy,
                total_volume: bucket_volume,
                n_trades: self.current_n_trades,
                completed_at_ms: timestamp_ms,
            };

            // Add to ring buffer
            if self.buckets.len() >= self.config.n_buckets {
                self.buckets.pop_front();
            }
            self.buckets.push_back(bucket);
            self.total_buckets += 1;
            bucket_completed = true;

            // Update adaptive bucket sizing
            if self.config.adaptive_enabled && bucket_duration_ms > 0 {
                self.update_adaptive_bucket_size(bucket_duration_ms, bucket_volume);
            }

            // Carry over excess to next bucket
            if overflow > 1e-12 {
                let overflow_frac = overflow / self.current_total_volume;
                self.current_buy_volume *= overflow_frac;
                self.current_total_volume = overflow;
                self.current_n_trades = 1; // At least one trade contributed to overflow
            } else {
                self.current_buy_volume = 0.0;
                self.current_total_volume = 0.0;
                self.current_n_trades = 0;
            }

            // Reset bucket start time for next bucket
            self.current_bucket_start_ms = timestamp_ms;
        }

        if bucket_completed {
            let new_vpin = self.compute_vpin();

            // Update velocity (EWMA of change)
            let change = new_vpin - self.prev_vpin;
            self.vpin_velocity = self.config.velocity_alpha * change
                + (1.0 - self.config.velocity_alpha) * self.vpin_velocity;

            self.prev_vpin = new_vpin;
            Some(new_vpin)
        } else {
            None
        }
    }

    /// Update adaptive bucket size based on observed bucket duration.
    ///
    /// The goal is to maintain approximately equal-time buckets by adjusting
    /// volume to match the target duration.
    fn update_adaptive_bucket_size(&mut self, bucket_duration_ms: u64, bucket_volume: f64) {
        // Convert to seconds
        let duration_secs = bucket_duration_ms as f64 / 1000.0;

        // Estimate volume rate (volume per second)
        let volume_rate = if duration_secs > 0.001 {
            bucket_volume / duration_secs
        } else {
            return; // Skip pathologically short buckets
        };

        // Update EWMA of volume rate
        let alpha = self.config.volume_rate_alpha;
        if self.volume_rate_ewma <= 0.0 {
            // First update - initialize directly
            self.volume_rate_ewma = volume_rate;
        } else {
            self.volume_rate_ewma = alpha * volume_rate + (1.0 - alpha) * self.volume_rate_ewma;
        }

        // Only adapt after warmup period
        if self.total_buckets < self.config.adaptive_warmup_buckets as u64 {
            return;
        }

        // Calculate target bucket volume to achieve target duration
        // bucket_volume = volume_rate × target_seconds
        let target_bucket_volume = self.volume_rate_ewma * self.config.target_bucket_seconds;

        // Clamp to min/max multipliers of initial bucket volume
        let min_volume = self.initial_bucket_volume * self.config.min_bucket_multiplier;
        let max_volume = self.initial_bucket_volume * self.config.max_bucket_multiplier;

        self.effective_bucket_volume = target_bucket_volume.clamp(min_volume, max_volume);
    }

    /// Classify trade as buy or sell.
    fn classify_trade(&self, price: f64, mid: f64) -> bool {
        if self.config.use_tick_rule {
            // Tick rule: compare to last price
            match self.last_price {
                Some(last) if price > last => true,
                Some(last) if price < last => false,
                _ => price >= mid, // Fallback to mid comparison
            }
        } else {
            // Bulk volume classification: compare to mid
            price >= mid
        }
    }

    /// Compute VPIN over rolling window of buckets.
    fn compute_vpin(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }

        let sum_imbalance: f64 = self.buckets.iter().map(|b| b.imbalance()).sum();

        sum_imbalance / self.buckets.len() as f64
    }

    /// Get current VPIN value [0, 1].
    ///
    /// Higher values indicate more toxic (informed) flow.
    pub fn vpin(&self) -> f64 {
        self.compute_vpin()
    }

    /// Get VPIN velocity (rate of change).
    ///
    /// Positive = toxicity increasing (danger).
    /// Negative = toxicity decreasing (improving).
    pub fn vpin_velocity(&self) -> f64 {
        self.vpin_velocity
    }

    /// Get signed order flow direction.
    ///
    /// Returns weighted average of signed imbalances.
    /// Positive = net buying pressure.
    /// Negative = net selling pressure.
    pub fn order_flow_direction(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }

        // Weight recent buckets more heavily
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let n = self.buckets.len();

        for (i, bucket) in self.buckets.iter().enumerate() {
            let weight = (i + 1) as f64 / n as f64; // Linear weighting
            weighted_sum += weight * bucket.signed_imbalance();
            weight_sum += weight;
        }

        if weight_sum > 1e-12 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Check if VPIN estimate is valid (enough buckets).
    pub fn is_valid(&self) -> bool {
        self.buckets.len() >= self.config.n_buckets / 2
    }

    /// Check if VPIN estimate is stale (most recent bucket too old).
    pub fn is_stale(&self, current_time_ms: u64) -> bool {
        if self.config.staleness_threshold_ms == 0 {
            return false; // Staleness check disabled
        }
        match self.buckets.back() {
            Some(bucket) => {
                current_time_ms.saturating_sub(bucket.completed_at_ms)
                    > self.config.staleness_threshold_ms
            }
            None => true, // No completed buckets = stale
        }
    }

    /// Get confidence in VPIN estimate [0, 1].
    ///
    /// Based on bucket count and trade density.
    pub fn confidence(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }

        // Bucket count confidence
        let bucket_conf = (self.buckets.len() as f64 / self.config.n_buckets as f64).min(1.0);

        // Trade density confidence
        let avg_trades = self.buckets.iter().map(|b| b.n_trades).sum::<usize>() as f64
            / self.buckets.len() as f64;
        let density_conf = (avg_trades / self.config.min_trades_per_bucket as f64).min(1.0);

        bucket_conf * density_conf
    }

    /// Get number of completed buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Get total trades processed.
    pub fn total_trades(&self) -> u64 {
        self.total_trades
    }

    /// Get total buckets completed.
    pub fn total_buckets(&self) -> u64 {
        self.total_buckets
    }

    /// Get current bucket fill percentage [0, 1].
    pub fn current_bucket_fill(&self) -> f64 {
        self.current_total_volume / self.effective_bucket_volume
    }

    /// Get configuration.
    pub fn config(&self) -> &VpinConfig {
        &self.config
    }

    /// Reset the estimator state.
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.current_buy_volume = 0.0;
        self.current_total_volume = 0.0;
        self.current_n_trades = 0;
        self.prev_vpin = 0.0;
        self.vpin_velocity = 0.0;
        self.total_trades = 0;
        self.total_buckets = 0;
        self.last_trade_ms = 0;
        self.last_price = None;
        // Reset adaptive state
        self.volume_rate_ewma = 0.0;
        self.current_bucket_start_ms = 0;
        self.effective_bucket_volume = self.initial_bucket_volume;
    }

    /// Get recent bucket imbalances for diagnostics.
    pub fn recent_imbalances(&self, n: usize) -> Vec<f64> {
        self.buckets
            .iter()
            .rev()
            .take(n)
            .map(|b| b.imbalance())
            .collect()
    }

    /// Update bucket volume (for adaptive sizing).
    /// Note: This updates the base/initial bucket volume.
    /// If adaptive sizing is enabled, the effective volume will be adjusted
    /// based on trading activity.
    pub fn set_bucket_volume(&mut self, bucket_volume: f64) {
        if bucket_volume > 0.0 {
            self.config.bucket_volume = bucket_volume;
            self.initial_bucket_volume = bucket_volume;
            self.effective_bucket_volume = bucket_volume;
        }
    }

    // === Adaptive sizing diagnostics ===

    /// Get current effective bucket volume.
    /// This may differ from config.bucket_volume if adaptive sizing is enabled.
    pub fn effective_bucket_volume(&self) -> f64 {
        self.effective_bucket_volume
    }

    /// Get estimated volume rate (volume per second).
    /// Returns 0.0 if not enough data.
    pub fn volume_rate(&self) -> f64 {
        self.volume_rate_ewma
    }

    /// Check if adaptive sizing is enabled and active.
    pub fn is_adaptive_active(&self) -> bool {
        self.config.adaptive_enabled
            && self.total_buckets >= self.config.adaptive_warmup_buckets as u64
    }

    /// Get the adaptation ratio: effective_bucket_volume / initial_bucket_volume.
    /// Values > 1.0 mean buckets have grown (high volume), < 1.0 means shrunk.
    pub fn adaptation_ratio(&self) -> f64 {
        if self.initial_bucket_volume > 0.0 {
            self.effective_bucket_volume / self.initial_bucket_volume
        } else {
            1.0
        }
    }
}

impl Default for VpinEstimator {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_basic() {
        let config = VpinConfig {
            bucket_volume: 10.0,
            n_buckets: 5,
            ..Default::default()
        };
        let mut vpin = VpinEstimator::new(config);

        // Add trades to complete a bucket (all buys)
        for i in 0..10 {
            let _ = vpin.on_trade(1.0, 100.01, 100.0, i * 1000);
        }

        // First bucket should have VPIN = 1.0 (all buys)
        assert!(vpin.bucket_count() >= 1);
        assert!(vpin.vpin() > 0.9, "All-buy bucket should have high VPIN");
    }

    #[test]
    fn test_vpin_balanced() {
        let config = VpinConfig {
            bucket_volume: 10.0,
            n_buckets: 5,
            ..Default::default()
        };
        let mut vpin = VpinEstimator::new(config);

        // Add balanced trades (half buy, half sell)
        for i in 0..20 {
            let price = if i % 2 == 0 { 100.01 } else { 99.99 };
            let _ = vpin.on_trade(1.0, price, 100.0, i * 1000);
        }

        // Should have low VPIN (balanced)
        assert!(vpin.bucket_count() >= 1);
        assert!(vpin.vpin() < 0.3, "Balanced trades should have low VPIN");
    }

    #[test]
    fn test_vpin_velocity() {
        let config = VpinConfig {
            bucket_volume: 5.0,
            n_buckets: 10,
            velocity_alpha: 0.3,
            ..Default::default()
        };
        let mut vpin = VpinEstimator::new(config);

        // Start with balanced trades
        for i in 0..50 {
            let price = if i % 2 == 0 { 100.01 } else { 99.99 };
            let _ = vpin.on_trade(1.0, price, 100.0, i * 1000);
        }

        let vpin_before = vpin.vpin();

        // Switch to all buys (toxic flow)
        for i in 50..100 {
            let _ = vpin.on_trade(1.0, 100.01, 100.0, i * 1000);
        }

        let vpin_after = vpin.vpin();

        // VPIN should increase and velocity should be positive
        assert!(
            vpin_after > vpin_before,
            "VPIN should increase with unidirectional flow"
        );
        assert!(
            vpin.vpin_velocity() > 0.0,
            "Velocity should be positive when toxicity increases"
        );
    }

    #[test]
    fn test_order_flow_direction() {
        let config = VpinConfig {
            bucket_volume: 5.0,
            n_buckets: 5,
            ..Default::default()
        };
        let mut vpin = VpinEstimator::new(config);

        // Add all buy trades
        for i in 0..25 {
            let _ = vpin.on_trade(1.0, 100.01, 100.0, i * 1000);
        }

        assert!(
            vpin.order_flow_direction() > 0.5,
            "Buy-dominated flow should be positive"
        );

        // Reset and add all sell trades
        vpin.reset();
        for i in 0..25 {
            let _ = vpin.on_trade(1.0, 99.99, 100.0, i * 1000);
        }

        assert!(
            vpin.order_flow_direction() < -0.5,
            "Sell-dominated flow should be negative"
        );
    }

    #[test]
    fn test_bucket_overflow() {
        let config = VpinConfig {
            bucket_volume: 3.0,
            n_buckets: 5,
            ..Default::default()
        };
        let mut vpin = VpinEstimator::new(config);

        // Single large trade that creates multiple buckets
        let result = vpin.on_trade(10.0, 100.01, 100.0, 1000);

        assert!(result.is_some(), "Should complete at least one bucket");
        assert!(
            vpin.bucket_count() >= 3,
            "10.0 volume should create 3+ buckets of size 3.0"
        );
    }

    #[test]
    fn test_confidence() {
        let config = VpinConfig {
            bucket_volume: 5.0,
            n_buckets: 10,
            min_trades_per_bucket: 5,
            ..Default::default()
        };
        let mut vpin = VpinEstimator::new(config);

        // Initially no confidence
        assert!(
            vpin.confidence() < 0.1,
            "Should have low confidence initially"
        );

        // Add enough trades to fill buckets
        for i in 0..100 {
            let _ = vpin.on_trade(1.0, 100.01, 100.0, i * 100);
        }

        assert!(
            vpin.confidence() > 0.5,
            "Should have higher confidence with more data"
        );
    }

    #[test]
    fn test_tick_rule_classification() {
        let config = VpinConfig {
            bucket_volume: 5.0,
            n_buckets: 5,
            use_tick_rule: true,
            ..Default::default()
        };
        let mut vpin = VpinEstimator::new(config);

        // Rising prices = buys
        for i in 0..10 {
            let price = 100.0 + i as f64 * 0.01;
            let _ = vpin.on_trade(1.0, price, 100.0, i * 1000);
        }

        assert!(
            vpin.vpin() > 0.7,
            "Rising prices should classify as buys (high VPIN)"
        );
    }
}
