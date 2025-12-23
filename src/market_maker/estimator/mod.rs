//! Econometric parameter estimation for GLFT strategy from live market data.
//!
//! Features a robust HFT pipeline:
//! - Volume Clock: Adaptive volume-based sampling (normalizes by economic activity)
//! - VWAP Pre-Averaging: Filters bid-ask bounce noise
//! - Bipower Variation: Jump-robust volatility estimation (σ = √BV)
//! - Regime Detection: RV/BV ratio identifies toxic jump regimes
//! - Weighted Kappa: Proximity-weighted L2 book regression

mod arrival;
mod bipower;
mod config;
mod flow;
mod kappa;
mod momentum;
mod volume_clock;

#[cfg(test)]
mod tests;

// Re-export public types
pub use config::EstimatorConfig;

use arrival::VolumeTickArrivalEstimator;
use bipower::MultiScaleBipowerEstimator;
use flow::TradeFlowTracker;
use kappa::WeightedKappaEstimator;
use momentum::MomentumDetector;
use volume_clock::VolumeBucketAccumulator;

use tracing::debug;

/// Econometric parameter estimator with multi-timescale variance, momentum detection,
/// and trade flow tracking for robust adverse selection protection.
///
/// Pipeline:
/// 1. Raw trades → Volume Clock → Normalized volume buckets with VWAP
/// 2. VWAP returns → Multi-Scale Bipower → sigma_clean, sigma_total, sigma_effective
/// 3. VWAP returns → Momentum Detector → falling/rising knife scores
/// 4. Trade tape → Flow Tracker → buy/sell imbalance
/// 5. L2 Book → Weighted Kappa → Order book depth decay
/// 6. Regime detection: fast jump_ratio > threshold = toxic
#[derive(Debug)]
pub struct ParameterEstimator {
    config: EstimatorConfig,
    /// Volume bucket accumulator (volume clock)
    bucket_accumulator: VolumeBucketAccumulator,
    /// Multi-timescale bipower estimator (replaces single-scale)
    multi_scale: MultiScaleBipowerEstimator,
    /// Momentum detector for falling/rising knife patterns
    momentum: MomentumDetector,
    /// Trade flow tracker for buy/sell imbalance
    flow: TradeFlowTracker,
    /// Weighted kappa estimator
    kappa: WeightedKappaEstimator,
    /// Volume tick arrival estimator
    arrival: VolumeTickArrivalEstimator,
    /// Current mid price
    #[allow(dead_code)]
    current_mid: f64,
    /// Current timestamp for momentum queries
    current_time_ms: u64,
}

impl ParameterEstimator {
    /// Create a new parameter estimator with the given config.
    pub fn new(config: EstimatorConfig) -> Self {
        let bucket_accumulator = VolumeBucketAccumulator::new(&config);
        let multi_scale = MultiScaleBipowerEstimator::new(&config);
        let momentum = MomentumDetector::new(config.momentum_window_ms);
        let flow = TradeFlowTracker::new(config.trade_flow_window_ms, config.trade_flow_alpha);
        let kappa = WeightedKappaEstimator::new(
            config.kappa_half_life_updates,
            config.default_kappa,
            config.kappa_max_distance,
            config.kappa_max_levels,
        );
        let arrival = VolumeTickArrivalEstimator::new(
            config.medium_half_life_ticks, // Use medium timescale
            config.default_arrival_intensity,
        );

        Self {
            config,
            bucket_accumulator,
            multi_scale,
            momentum,
            flow,
            kappa,
            arrival,
            current_mid: 0.0,
            current_time_ms: 0,
        }
    }

    /// Update current mid price.
    pub fn on_mid_update(&mut self, mid_price: f64) {
        self.current_mid = mid_price;
    }

    /// Process a new trade (feeds into volume clock AND flow tracker).
    ///
    /// # Arguments
    /// * `timestamp_ms` - Trade timestamp
    /// * `price` - Trade price
    /// * `size` - Trade size
    /// * `is_buy_aggressor` - Whether buyer was the taker (if available from exchange)
    pub fn on_trade(
        &mut self,
        timestamp_ms: u64,
        price: f64,
        size: f64,
        is_buy_aggressor: Option<bool>,
    ) {
        self.current_time_ms = timestamp_ms;

        // Track trade flow if we know aggressor side
        if let Some(is_buy) = is_buy_aggressor {
            self.flow.on_trade(timestamp_ms, size, is_buy);
        }

        // Feed into volume bucket accumulator
        if let Some(bucket) = self.bucket_accumulator.on_trade(timestamp_ms, price, size) {
            // Get log return BEFORE updating multi_scale (it will update last_vwap)
            let log_return = self.multi_scale.last_log_return(&bucket);

            // Bucket completed - update estimators
            self.multi_scale.on_bucket(&bucket);
            self.arrival.on_bucket(&bucket);

            // Update momentum detector with signed return
            if let Some(ret) = log_return {
                self.momentum.on_bucket(bucket.end_time_ms, ret);
            }

            debug!(
                vwap = %format!("{:.4}", bucket.vwap),
                volume = %format!("{:.4}", bucket.volume),
                duration_ms = bucket.end_time_ms.saturating_sub(bucket.start_time_ms),
                tick = self.multi_scale.tick_count(),
                sigma_clean = %format!("{:.6}", self.multi_scale.sigma_clean()),
                sigma_total = %format!("{:.6}", self.multi_scale.sigma_total()),
                jump_ratio = %format!("{:.2}", self.multi_scale.jump_ratio_fast()),
                "Volume bucket completed"
            );
        }
    }

    /// Legacy on_trade without aggressor info (backward compatibility).
    pub fn on_trade_legacy(&mut self, timestamp_ms: u64, price: f64, size: f64) {
        self.on_trade(timestamp_ms, price, size, None);
    }

    /// Process L2 order book update for kappa estimation.
    /// bids and asks are slices of (price, size) tuples, best first.
    pub fn on_l2_book(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        self.kappa.update(bids, asks, mid);
    }

    // === Volatility Accessors ===

    /// Get clean volatility (σ_clean) - per-second, NOT annualized.
    /// Based on Bipower Variation, robust to jumps.
    /// Use for base spread pricing (continuous risk).
    pub fn sigma(&self) -> f64 {
        self.multi_scale.sigma_clean()
    }

    /// Get clean volatility - alias for sigma()
    pub fn sigma_clean(&self) -> f64 {
        self.multi_scale.sigma_clean()
    }

    /// Get total volatility (σ_total) - includes jumps.
    /// Based on Realized Variance, captures full price risk.
    pub fn sigma_total(&self) -> f64 {
        self.multi_scale.sigma_total()
    }

    /// Get effective volatility (σ_effective) - blended.
    /// Blends clean and total based on jump regime.
    /// Use for inventory skew (reacts appropriately to jumps).
    pub fn sigma_effective(&self) -> f64 {
        self.multi_scale.sigma_effective()
    }

    // === Order Book Accessors ===

    /// Get current order book depth decay estimate (κ).
    pub fn kappa(&self) -> f64 {
        self.kappa.kappa()
    }

    /// Get current order arrival intensity (volume ticks per second).
    pub fn arrival_intensity(&self) -> f64 {
        self.arrival.ticks_per_second()
    }

    // === Regime Detection ===

    /// Get fast RV/BV jump ratio.
    /// - ≈ 1.0: Normal diffusion (safe to market make)
    /// - > 1.5: Jumps present (toxic environment)
    pub fn jump_ratio(&self) -> f64 {
        self.multi_scale.jump_ratio_fast()
    }

    /// Check if currently in toxic (jump) regime.
    pub fn is_toxic_regime(&self) -> bool {
        self.multi_scale.jump_ratio_fast() > self.config.jump_ratio_threshold
    }

    // === Directional Flow Accessors ===

    /// Get signed momentum in bps over momentum window.
    /// Negative = market falling, Positive = market rising.
    pub fn momentum_bps(&self) -> f64 {
        self.momentum.momentum_bps(self.current_time_ms)
    }

    /// Get falling knife score [0, 3].
    /// > 0.5 = some downward momentum
    /// > 1.0 = severe downward momentum (protect bids!)
    pub fn falling_knife_score(&self) -> f64 {
        self.momentum.falling_knife_score(self.current_time_ms)
    }

    /// Get rising knife score [0, 3].
    /// > 0.5 = some upward momentum
    /// > 1.0 = severe upward momentum (protect asks!)
    pub fn rising_knife_score(&self) -> f64 {
        self.momentum.rising_knife_score(self.current_time_ms)
    }

    /// Get trade flow imbalance [-1, 1].
    /// Negative = sell pressure, Positive = buy pressure.
    pub fn flow_imbalance(&self) -> f64 {
        self.flow.imbalance()
    }

    // === Warmup ===

    /// Check if estimator has collected enough data.
    pub fn is_warmed_up(&self) -> bool {
        self.multi_scale.tick_count() >= self.config.min_volume_ticks
            && self.kappa.update_count() >= self.config.min_l2_updates
    }

    /// Get current warmup progress.
    /// Returns (volume_ticks, min_volume_ticks, l2_updates, min_l2_updates)
    pub fn warmup_progress(&self) -> (usize, usize, usize, usize) {
        (
            self.multi_scale.tick_count(),
            self.config.min_volume_ticks,
            self.kappa.update_count(),
            self.config.min_l2_updates,
        )
    }

    /// Get simplified warmup progress for legacy compatibility.
    /// Returns (current_samples, min_samples) based on volume ticks.
    pub fn warmup_progress_simple(&self) -> (usize, usize) {
        (self.multi_scale.tick_count(), self.config.min_volume_ticks)
    }
}
