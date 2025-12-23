//! Econometric parameter estimation for GLFT strategy from live market data.
//!
//! Features a robust HFT pipeline:
//! - Volume Clock: Adaptive volume-based sampling (normalizes by economic activity)
//! - VWAP Pre-Averaging: Filters bid-ask bounce noise
//! - Bipower Variation: Jump-robust volatility estimation (σ = √BV)
//! - Regime Detection: RV/BV ratio identifies toxic jump regimes
//! - Weighted Kappa: Proximity-weighted L2 book regression

use std::collections::VecDeque;
use tracing::debug;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for econometric parameter estimation pipeline.
#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    // === Volume Clock ===
    /// Initial volume bucket threshold (in asset units, e.g., 1.0 BTC)
    pub initial_bucket_volume: f64,
    /// Rolling window for adaptive bucket calculation (seconds)
    pub volume_window_secs: f64,
    /// Percentile of rolling volume for adaptive bucket (e.g., 0.01 = 1%)
    pub volume_percentile: f64,
    /// Minimum bucket volume floor
    pub min_bucket_volume: f64,
    /// Maximum bucket volume ceiling
    pub max_bucket_volume: f64,

    // === EWMA ===
    /// Half-life for variance EWMA (in volume ticks, not time)
    pub variance_half_life_ticks: f64,
    /// Half-life for kappa EWMA (in L2 updates)
    pub kappa_half_life_updates: f64,

    // === Regime Detection ===
    /// Jump detection threshold: RV/BV ratio > threshold = toxic regime
    pub jump_ratio_threshold: f64,

    // === Kappa Estimation ===
    /// Maximum distance from mid for kappa regression (as fraction, e.g., 0.01 = 1%)
    pub kappa_max_distance: f64,
    /// Maximum number of L2 levels to use per side
    pub kappa_max_levels: usize,

    // === Warmup ===
    /// Minimum volume ticks before volatility estimates are valid
    pub min_volume_ticks: usize,
    /// Minimum L2 updates before kappa is valid
    pub min_l2_updates: usize,

    // === Defaults ===
    /// Default sigma during warmup (per-second volatility)
    pub default_sigma: f64,
    /// Default kappa during warmup (order book decay)
    pub default_kappa: f64,
    /// Default arrival intensity during warmup (ticks per second)
    pub default_arrival_intensity: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            // Volume Clock - tuned for testnet/low-activity markets
            initial_bucket_volume: 0.01, // Start very small (0.01 BTC)
            volume_window_secs: 300.0,   // 5 minutes
            volume_percentile: 0.01,     // 1% of 5-min volume
            min_bucket_volume: 0.001,    // Floor at 0.001 BTC
            max_bucket_volume: 10.0,     // Cap at 10 BTC

            // EWMA
            variance_half_life_ticks: 50.0, // 50 volume ticks
            kappa_half_life_updates: 30.0,  // 30 L2 updates

            // Regime Detection
            jump_ratio_threshold: 3.0, // RV/BV > 3.0 = toxic

            // Kappa
            kappa_max_distance: 0.01, // 1% from mid
            kappa_max_levels: 15,

            // Warmup - reasonable for testnet/low-activity
            min_volume_ticks: 10,
            min_l2_updates: 5,

            // Defaults
            default_sigma: 0.0001,          // 0.01% per-second
            default_kappa: 100.0,           // Moderate depth decay
            default_arrival_intensity: 0.5, // 0.5 ticks per second
        }
    }
}

impl EstimatorConfig {
    /// Create config from legacy fields (for backward compatibility with bin/market_maker.rs)
    pub fn from_legacy(
        _window_ms: u64,
        _min_trades: usize,
        default_sigma: f64,
        default_kappa: f64,
        default_arrival_intensity: f64,
        _decay_secs: u64,
        min_warmup_trades: usize,
    ) -> Self {
        Self {
            default_sigma,
            default_kappa,
            default_arrival_intensity,
            min_volume_ticks: min_warmup_trades.max(10),
            ..Default::default()
        }
    }
}

// ============================================================================
// Volume Clock (Data Normalization)
// ============================================================================

/// A completed volume bucket with VWAP.
#[derive(Debug, Clone)]
struct VolumeBucket {
    /// Start timestamp of bucket (ms)
    start_time_ms: u64,
    /// End timestamp of bucket (ms)
    end_time_ms: u64,
    /// VWAP: sum(price * size) / sum(size)
    vwap: f64,
    /// Total volume in bucket
    volume: f64,
}

/// Accumulates trades until volume threshold is reached.
/// Implements adaptive volume clock for normalized economic sampling.
#[derive(Debug)]
struct VolumeBucketAccumulator {
    /// Current bucket start time
    start_time_ms: Option<u64>,
    /// Sum of (price * size) for current bucket
    price_volume_sum: f64,
    /// Sum of sizes for current bucket
    volume_sum: f64,
    /// Current adaptive threshold
    threshold: f64,
    /// Rolling volume tracker for adaptive threshold: (timestamp_ms, volume)
    rolling_volumes: VecDeque<(u64, f64)>,
    /// Config reference
    initial_bucket_volume: f64,
    volume_window_secs: f64,
    volume_percentile: f64,
    min_bucket_volume: f64,
    max_bucket_volume: f64,
}

impl VolumeBucketAccumulator {
    fn new(config: &EstimatorConfig) -> Self {
        Self {
            start_time_ms: None,
            price_volume_sum: 0.0,
            volume_sum: 0.0,
            threshold: config.initial_bucket_volume,
            rolling_volumes: VecDeque::new(),
            initial_bucket_volume: config.initial_bucket_volume,
            volume_window_secs: config.volume_window_secs,
            volume_percentile: config.volume_percentile,
            min_bucket_volume: config.min_bucket_volume,
            max_bucket_volume: config.max_bucket_volume,
        }
    }

    /// Add a trade. Returns Some(VolumeBucket) if bucket completed.
    fn on_trade(&mut self, time_ms: u64, price: f64, size: f64) -> Option<VolumeBucket> {
        if price <= 0.0 || size <= 0.0 {
            return None;
        }

        // Initialize bucket start if needed
        if self.start_time_ms.is_none() {
            self.start_time_ms = Some(time_ms);
        }

        // Accumulate
        self.price_volume_sum += price * size;
        self.volume_sum += size;

        // Update rolling volumes for adaptive threshold
        self.rolling_volumes.push_back((time_ms, size));
        let cutoff = time_ms.saturating_sub((self.volume_window_secs * 1000.0) as u64);
        while self
            .rolling_volumes
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.rolling_volumes.pop_front();
        }

        // Check if bucket is complete
        if self.volume_sum >= self.threshold {
            let vwap = self.price_volume_sum / self.volume_sum;
            let bucket = VolumeBucket {
                start_time_ms: self.start_time_ms.unwrap(),
                end_time_ms: time_ms,
                vwap,
                volume: self.volume_sum,
            };

            // Reset for next bucket
            self.price_volume_sum = 0.0;
            self.volume_sum = 0.0;
            self.start_time_ms = None;

            // Update adaptive threshold based on rolling volume
            self.update_adaptive_threshold();

            Some(bucket)
        } else {
            None
        }
    }

    /// Calculate adaptive threshold as percentile of recent volume.
    fn update_adaptive_threshold(&mut self) {
        let total_rolling: f64 = self.rolling_volumes.iter().map(|(_, v)| v).sum();

        if total_rolling > 0.0 {
            let target = total_rolling * self.volume_percentile;
            self.threshold = target.clamp(self.min_bucket_volume, self.max_bucket_volume);
        } else {
            self.threshold = self.initial_bucket_volume;
        }
    }

    /// Get current bucket fill progress (0.0 to 1.0+)
    #[allow(dead_code)]
    fn bucket_progress(&self) -> f64 {
        if self.threshold > 0.0 {
            self.volume_sum / self.threshold
        } else {
            0.0
        }
    }
}

// ============================================================================
// Bipower Variation (Jump-Robust Volatility)
// ============================================================================

/// Bipower variation estimator for jump-robust volatility.
///
/// Tracks both:
/// - RV (Realized Variance): EWMA of r² (includes jumps)
/// - BV (Bipower Variation): EWMA of (π/2) × |r_t| × |r_{t-1}| (robust to jumps)
///
/// Key insight: BV ≈ RV in normal markets, but BV << RV when jumps occur.
/// Jump ratio = RV/BV detects toxic regimes.
#[derive(Debug)]
struct BipowerVariationEstimator {
    /// EWMA decay factor (per tick)
    alpha: f64,
    /// Realized variance (includes jumps): EWMA of r²
    ewma_rv: f64,
    /// Bipower variation (excludes jumps): EWMA of (π/2)|r_t||r_{t-1}|
    ewma_bv: f64,
    /// Last absolute log return (for BV calculation)
    last_abs_return: Option<f64>,
    /// Last VWAP (for log return calculation)
    last_vwap: Option<f64>,
    /// Number of ticks processed
    tick_count: usize,
}

impl BipowerVariationEstimator {
    fn new(half_life_ticks: f64, default_sigma: f64) -> Self {
        let alpha = (2.0_f64.ln() / half_life_ticks).clamp(0.001, 1.0);
        let default_var = default_sigma.powi(2);
        Self {
            alpha,
            ewma_rv: default_var,
            ewma_bv: default_var,
            last_abs_return: None,
            last_vwap: None,
            tick_count: 0,
        }
    }

    /// Process a completed volume bucket.
    fn on_bucket(&mut self, bucket: &VolumeBucket) {
        if let Some(last_vwap) = self.last_vwap {
            if bucket.vwap > 0.0 && last_vwap > 0.0 {
                let log_return = (bucket.vwap / last_vwap).ln();
                let abs_return = log_return.abs();

                // RV: EWMA of r²
                let rv_observation = log_return.powi(2);
                self.ewma_rv = self.alpha * rv_observation + (1.0 - self.alpha) * self.ewma_rv;

                // BV: EWMA of (π/2) × |r_t| × |r_{t-1}|
                // This is the unbiased estimator for continuous variance under Gaussian assumption
                if let Some(last_abs) = self.last_abs_return {
                    let bv_observation = std::f64::consts::FRAC_PI_2 * abs_return * last_abs;
                    self.ewma_bv = self.alpha * bv_observation + (1.0 - self.alpha) * self.ewma_bv;
                }

                self.last_abs_return = Some(abs_return);
                self.tick_count += 1;
            }
        }
        self.last_vwap = Some(bucket.vwap);
    }

    /// Get clean volatility: sqrt(Bipower Variation).
    /// This is robust to jumps - measures only the continuous component.
    fn sigma(&self) -> f64 {
        self.ewma_bv.sqrt().clamp(1e-7, 0.01)
    }

    /// Get realized variance (includes jumps).
    #[allow(dead_code)]
    fn realized_variance(&self) -> f64 {
        self.ewma_rv
    }

    /// Get bipower variation (excludes jumps).
    #[allow(dead_code)]
    fn bipower_variation(&self) -> f64 {
        self.ewma_bv
    }

    /// Get jump ratio: RV / BV.
    /// - ratio ≈ 1.0: Normal diffusion (safe market making)
    /// - ratio >> 1.0: Jumps present (toxic environment)
    fn jump_ratio(&self) -> f64 {
        if self.ewma_bv > 1e-12 {
            (self.ewma_rv / self.ewma_bv).clamp(0.1, 100.0)
        } else {
            1.0
        }
    }

    fn tick_count(&self) -> usize {
        self.tick_count
    }
}

// ============================================================================
// Weighted Kappa Estimator (Improved L2 Analysis)
// ============================================================================

/// Weighted linear regression kappa estimator.
///
/// Improvements over simple kappa:
/// - Truncates to orders within max_distance of mid (ignores fake far orders)
/// - Uses first N levels only (focuses on relevant liquidity)
/// - Weights by proximity to mid (closer levels matter more)
#[derive(Debug)]
struct WeightedKappaEstimator {
    alpha: f64,
    kappa: f64,
    max_distance: f64,
    max_levels: usize,
    update_count: usize,
}

impl WeightedKappaEstimator {
    fn new(
        half_life_updates: f64,
        default_kappa: f64,
        max_distance: f64,
        max_levels: usize,
    ) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_updates).clamp(0.001, 1.0),
            kappa: default_kappa,
            max_distance,
            max_levels,
            update_count: 0,
        }
    }

    /// Update kappa from L2 order book.
    ///
    /// Uses instantaneous depth (size at each level) to fit exponential decay:
    /// L(δ) = A × exp(-κ × δ)  =>  ln(L) = ln(A) - κ × δ
    fn update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if mid <= 0.0 {
            return;
        }

        // Collect points: (distance, size_at_level, weight)
        let mut points: Vec<(f64, f64, f64)> = Vec::new();

        // Process bids (truncate and limit levels)
        for (i, (price, size)) in bids.iter().enumerate() {
            if i >= self.max_levels || *price <= 0.0 || *size <= 0.0 {
                break;
            }
            let distance = (mid - price) / mid;
            if distance > self.max_distance {
                break; // Too far from mid
            }
            // Weight by proximity: closer to mid = higher weight
            let weight = 1.0 / (1.0 + distance * 100.0);
            if distance > 1e-6 {
                points.push((distance, *size, weight));
            }
        }

        // Process asks (same logic)
        for (i, (price, size)) in asks.iter().enumerate() {
            if i >= self.max_levels || *price <= 0.0 || *size <= 0.0 {
                break;
            }
            let distance = (price - mid) / mid;
            if distance > self.max_distance {
                break;
            }
            let weight = 1.0 / (1.0 + distance * 100.0);
            if distance > 1e-6 {
                points.push((distance, *size, weight));
            }
        }

        // Need at least 4 points for meaningful regression
        if points.len() < 4 {
            return;
        }

        // Weighted linear regression on (distance, ln(size))
        // Model: ln(size) = ln(A) - κ × distance
        // Slope = -κ, so κ = -slope
        if let Some(slope) = weighted_linear_regression_slope(&points) {
            // In real order books, liquidity often INCREASES slightly with distance
            // (more limit orders stacked further from mid), giving positive slope.
            // Use absolute value and a reasonable default if slope is wrong sign.
            let kappa_estimated = if slope < 0.0 {
                // Negative slope = liquidity decays with distance (expected in theory)
                (-slope).clamp(1.0, 10000.0)
            } else {
                // Positive slope = liquidity increases with distance (common in practice)
                // Use a moderate default based on typical market structure
                50.0
            };

            // EWMA update
            self.kappa = self.alpha * kappa_estimated + (1.0 - self.alpha) * self.kappa;
            self.update_count += 1;

            debug!(
                points = points.len(),
                slope = %format!("{:.4}", slope),
                kappa_new = %format!("{:.2}", kappa_estimated),
                kappa_ewma = %format!("{:.2}", self.kappa),
                "Kappa updated from L2 book"
            );
        }
    }

    fn kappa(&self) -> f64 {
        self.kappa.clamp(1.0, 10000.0)
    }

    fn update_count(&self) -> usize {
        self.update_count
    }
}

/// Weighted linear regression to get slope.
/// Points are (x, y, weight). Fits ln(y) ~ a + b*x, returns b.
fn weighted_linear_regression_slope(points: &[(f64, f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxx = 0.0;
    let mut sum_wxy = 0.0;

    for (x, y, w) in points {
        if *y <= 0.0 {
            continue;
        }
        let ln_y = y.ln();
        sum_w += w;
        sum_wx += w * x;
        sum_wy += w * ln_y;
        sum_wxx += w * x * x;
        sum_wxy += w * x * ln_y;
    }

    let denominator = sum_w * sum_wxx - sum_wx * sum_wx;
    if denominator.abs() < 1e-12 {
        return None;
    }

    Some((sum_w * sum_wxy - sum_wx * sum_wy) / denominator)
}

// ============================================================================
// Volume Tick Arrival Estimator
// ============================================================================

/// Arrival intensity estimator based on volume clock ticks.
/// Measures volume ticks per second (not raw trades per second).
#[derive(Debug)]
struct VolumeTickArrivalEstimator {
    alpha: f64,
    intensity: f64, // Volume ticks per second
    last_tick_ms: Option<u64>,
    tick_count: usize,
}

impl VolumeTickArrivalEstimator {
    fn new(half_life_ticks: f64, default_intensity: f64) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 1.0),
            intensity: default_intensity,
            last_tick_ms: None,
            tick_count: 0,
        }
    }

    fn on_bucket(&mut self, bucket: &VolumeBucket) {
        if let Some(last_ms) = self.last_tick_ms {
            let interval_secs = (bucket.end_time_ms.saturating_sub(last_ms)) as f64 / 1000.0;
            if interval_secs > 0.001 {
                let rate = 1.0 / interval_secs;
                let rate_clamped = rate.clamp(0.001, 100.0);
                self.intensity = self.alpha * rate_clamped + (1.0 - self.alpha) * self.intensity;
                self.tick_count += 1;
            }
        }
        self.last_tick_ms = Some(bucket.end_time_ms);
    }

    fn ticks_per_second(&self) -> f64 {
        self.intensity.clamp(0.001, 100.0)
    }
}

// ============================================================================
// Main Orchestrator: ParameterEstimator
// ============================================================================

/// Econometric parameter estimator with volume clock, bipower variation, and regime detection.
///
/// Pipeline:
/// 1. Raw trades → Volume Clock → Normalized volume buckets with VWAP
/// 2. VWAP returns → Bipower Variation → Jump-robust σ + RV/BV ratio
/// 3. L2 Book → Weighted Kappa → Order book depth decay
/// 4. Regime detection: RV/BV > threshold = toxic
#[derive(Debug)]
pub struct ParameterEstimator {
    config: EstimatorConfig,
    /// Volume bucket accumulator (volume clock)
    bucket_accumulator: VolumeBucketAccumulator,
    /// Bipower variation estimator (for sigma and jump detection)
    bipower: BipowerVariationEstimator,
    /// Weighted kappa estimator
    kappa: WeightedKappaEstimator,
    /// Volume tick arrival estimator
    arrival: VolumeTickArrivalEstimator,
    /// Current mid price
    #[allow(dead_code)]
    current_mid: f64,
}

impl ParameterEstimator {
    /// Create a new parameter estimator with the given config.
    pub fn new(config: EstimatorConfig) -> Self {
        let bucket_accumulator = VolumeBucketAccumulator::new(&config);
        let bipower =
            BipowerVariationEstimator::new(config.variance_half_life_ticks, config.default_sigma);
        let kappa = WeightedKappaEstimator::new(
            config.kappa_half_life_updates,
            config.default_kappa,
            config.kappa_max_distance,
            config.kappa_max_levels,
        );
        let arrival = VolumeTickArrivalEstimator::new(
            config.variance_half_life_ticks,
            config.default_arrival_intensity,
        );

        Self {
            config,
            bucket_accumulator,
            bipower,
            kappa,
            arrival,
            current_mid: 0.0,
        }
    }

    /// Update current mid price.
    pub fn on_mid_update(&mut self, mid_price: f64) {
        self.current_mid = mid_price;
    }

    /// Process a new trade (feeds into volume clock).
    /// Note: Now requires size parameter for volume-based sampling.
    pub fn on_trade(&mut self, timestamp_ms: u64, price: f64, size: f64) {
        // Feed into volume bucket accumulator
        if let Some(bucket) = self.bucket_accumulator.on_trade(timestamp_ms, price, size) {
            // Bucket completed - update estimators with VWAP
            self.bipower.on_bucket(&bucket);
            self.arrival.on_bucket(&bucket);

            debug!(
                vwap = %format!("{:.4}", bucket.vwap),
                volume = %format!("{:.4}", bucket.volume),
                duration_ms = bucket.end_time_ms.saturating_sub(bucket.start_time_ms),
                tick = self.bipower.tick_count(),
                "Volume bucket completed"
            );
        }
    }

    /// Legacy on_trade without size (for backward compatibility).
    /// Uses a default size of 1.0 - not recommended for production.
    #[allow(dead_code)]
    pub fn on_trade_legacy(&mut self, timestamp_ms: u64, price: f64) {
        self.on_trade(timestamp_ms, price, 1.0);
    }

    /// Process L2 order book update for kappa estimation.
    /// bids and asks are slices of (price, size) tuples, best first.
    pub fn on_l2_book(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        self.kappa.update(bids, asks, mid);
    }

    // === Accessors ===

    /// Get current volatility estimate (σ) - per-second, NOT annualized.
    /// This is sqrt(Bipower Variation), which is robust to jumps.
    pub fn sigma(&self) -> f64 {
        self.bipower.sigma()
    }

    /// Get current order book depth decay estimate (κ).
    pub fn kappa(&self) -> f64 {
        self.kappa.kappa()
    }

    /// Get current order arrival intensity (volume ticks per second).
    pub fn arrival_intensity(&self) -> f64 {
        self.arrival.ticks_per_second()
    }

    /// Get RV/BV jump ratio.
    /// - ≈ 1.0: Normal diffusion (safe to market make)
    /// - > 3.0: Jumps dominating (toxic environment)
    pub fn jump_ratio(&self) -> f64 {
        self.bipower.jump_ratio()
    }

    /// Check if currently in toxic (jump) regime.
    pub fn is_toxic_regime(&self) -> bool {
        self.bipower.jump_ratio() > self.config.jump_ratio_threshold
    }

    /// Check if estimator has collected enough data.
    pub fn is_warmed_up(&self) -> bool {
        self.bipower.tick_count() >= self.config.min_volume_ticks
            && self.kappa.update_count() >= self.config.min_l2_updates
    }

    /// Get current warmup progress.
    /// Returns (volume_ticks, min_volume_ticks, l2_updates, min_l2_updates)
    pub fn warmup_progress(&self) -> (usize, usize, usize, usize) {
        (
            self.bipower.tick_count(),
            self.config.min_volume_ticks,
            self.kappa.update_count(),
            self.config.min_l2_updates,
        )
    }

    /// Get simplified warmup progress for legacy compatibility.
    /// Returns (current_samples, min_samples) based on volume ticks.
    pub fn warmup_progress_simple(&self) -> (usize, usize) {
        (self.bipower.tick_count(), self.config.min_volume_ticks)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> EstimatorConfig {
        EstimatorConfig {
            initial_bucket_volume: 1.0,
            min_volume_ticks: 5,
            min_l2_updates: 3,
            variance_half_life_ticks: 10.0,
            kappa_half_life_updates: 10.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_volume_bucket_accumulation() {
        let config = EstimatorConfig {
            initial_bucket_volume: 10.0,
            ..Default::default()
        };
        let mut acc = VolumeBucketAccumulator::new(&config);

        // Should not complete bucket yet
        assert!(acc.on_trade(1000, 100.0, 3.0).is_none());
        assert!(acc.on_trade(2000, 101.0, 3.0).is_none());

        // Should complete bucket at 10+ volume
        let bucket = acc.on_trade(3000, 102.0, 5.0);
        assert!(bucket.is_some());

        let b = bucket.unwrap();
        assert!((b.volume - 11.0).abs() < 0.01);
        // VWAP = (100*3 + 101*3 + 102*5) / 11 = 1113/11 ≈ 101.18
        assert!((b.vwap - 101.18).abs() < 0.1);
    }

    #[test]
    fn test_vwap_calculation() {
        let config = EstimatorConfig {
            initial_bucket_volume: 5.0,
            ..Default::default()
        };
        let mut acc = VolumeBucketAccumulator::new(&config);

        // Trades at different prices
        acc.on_trade(1000, 100.0, 2.0); // 200, vol = 2
        acc.on_trade(2000, 110.0, 2.0); // 220, vol = 4

        // Total: 420 / 4 = 105.0, but need 5 volume
        let bucket = acc.on_trade(2000, 110.0, 0.0); // Force check with zero size
        assert!(bucket.is_none()); // Not enough volume yet

        // Add more to complete (need 1 more)
        let bucket = acc.on_trade(3000, 120.0, 1.0); // 120, vol = 5
        assert!(bucket.is_some());

        let b = bucket.unwrap();
        // VWAP = (200 + 220 + 120) / 5 = 540/5 = 108.0
        assert!((b.vwap - 108.0).abs() < 0.1);
    }

    #[test]
    fn test_bipower_no_jumps() {
        let mut bv = BipowerVariationEstimator::new(10.0, 0.001);

        // Feed stable returns (no jumps) - small oscillations
        let vwaps = [100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1];
        for (i, vwap) in vwaps.iter().enumerate() {
            bv.on_bucket(&VolumeBucket {
                start_time_ms: i as u64 * 1000,
                end_time_ms: (i + 1) as u64 * 1000,
                vwap: *vwap,
                volume: 1.0,
            });
        }

        // Jump ratio should be close to 1.0 (no jumps)
        let ratio = bv.jump_ratio();
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Expected ratio ~1.0 for no jumps, got {}",
            ratio
        );
    }

    #[test]
    fn test_bipower_with_jump() {
        let mut bv = BipowerVariationEstimator::new(5.0, 0.001);

        // Feed returns with a sudden jump
        // Normal, normal, JUMP, normal, normal
        let vwaps = [100.0, 100.1, 100.0, 105.0, 105.1, 105.0, 105.1];
        for (i, vwap) in vwaps.iter().enumerate() {
            bv.on_bucket(&VolumeBucket {
                start_time_ms: i as u64 * 1000,
                end_time_ms: (i + 1) as u64 * 1000,
                vwap: *vwap,
                volume: 1.0,
            });
        }

        // Jump ratio should be elevated (RV > BV due to jump)
        let ratio = bv.jump_ratio();
        assert!(
            ratio > 1.5,
            "Expected elevated ratio due to jump, got {}",
            ratio
        );
    }

    #[test]
    fn test_weighted_kappa_estimator() {
        let mut kappa = WeightedKappaEstimator::new(10.0, 100.0, 0.01, 15);
        let mid = 100.0;

        // Synthetic book where depth increases at each level
        // This is typical: more liquidity accumulates further from mid
        // Cumulative depth at level i: sum of sizes from 0 to i
        let bids: Vec<(f64, f64)> = (1..=10)
            .map(|i| {
                let price = mid - i as f64 * 0.05; // 99.95, 99.90, ...
                let size = 1.0; // Constant size at each level
                (price, size)
            })
            .collect();

        let asks: Vec<(f64, f64)> = (1..=10)
            .map(|i| {
                let price = mid + i as f64 * 0.05; // 100.05, 100.10, ...
                let size = 1.0;
                (price, size)
            })
            .collect();

        // Run multiple updates to converge
        for _ in 0..30 {
            kappa.update(&bids, &asks, mid);
        }

        // Kappa should be in a reasonable range (changed from default 100)
        let k = kappa.kappa();
        assert!(
            k > 1.0 && k < 10000.0,
            "Kappa should be in valid range, got {}",
            k
        );
        // Verify it's updating (not stuck at default)
        assert!(
            k != 100.0,
            "Kappa should have changed from default 100, got {}",
            k
        );
    }

    #[test]
    fn test_regime_detection() {
        let mut config = make_config();
        config.jump_ratio_threshold = 2.0;
        let estimator = ParameterEstimator::new(config);

        // Initially not toxic (default ratio = 1.0)
        assert!(!estimator.is_toxic_regime());
    }

    #[test]
    fn test_full_pipeline_warmup() {
        let config = make_config();
        let mut estimator = ParameterEstimator::new(config);

        assert!(!estimator.is_warmed_up());

        // Feed trades to fill buckets (need 5 volume ticks)
        let mut time = 1000u64;
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 0.5;
            estimator.on_trade(time, price, 0.5); // 0.5 per trade, 2 trades per bucket
            time += 100;
        }

        // Feed L2 books (need 3 updates)
        let bids = vec![(99.9, 5.0), (99.8, 10.0), (99.7, 15.0)];
        let asks = vec![(100.1, 5.0), (100.2, 10.0), (100.3, 15.0)];
        for _ in 0..5 {
            estimator.on_l2_book(&bids, &asks, 100.0);
        }

        // Should be warmed up
        assert!(estimator.is_warmed_up());

        // Params should be in reasonable ranges
        let sigma = estimator.sigma();
        let kappa = estimator.kappa();
        let ratio = estimator.jump_ratio();

        assert!(sigma > 0.0, "sigma should be positive");
        assert!(kappa > 1.0, "kappa should be > 1");
        assert!(ratio > 0.0, "jump_ratio should be positive");
    }

    #[test]
    fn test_adaptive_bucket_threshold() {
        let config = EstimatorConfig {
            initial_bucket_volume: 1.0,
            volume_window_secs: 10.0,
            volume_percentile: 0.1, // 10% of rolling volume
            min_bucket_volume: 0.5,
            max_bucket_volume: 10.0,
            ..Default::default()
        };
        let mut acc = VolumeBucketAccumulator::new(&config);

        // Fill several buckets to build up rolling volume
        let mut time = 0u64;
        for _ in 0..10 {
            while acc.on_trade(time, 100.0, 0.5).is_none() {
                time += 100;
            }
        }

        // Threshold should have adapted based on rolling volume
        // With 10 buckets of ~1.0 volume each in 10 seconds,
        // and 10% percentile, threshold should be around 1.0
        assert!(acc.threshold >= 0.5);
        assert!(acc.threshold <= 10.0);
    }

    #[test]
    fn test_arrival_intensity() {
        let config = make_config();
        let mut estimator = ParameterEstimator::new(config);

        // Feed trades at consistent rate to fill buckets
        let mut time = 0u64;
        for _ in 0..50 {
            estimator.on_trade(time, 100.0, 1.0); // Each trade = 1 bucket
            time += 500; // 500ms between buckets = 2 ticks/sec
        }

        let intensity = estimator.arrival_intensity();
        // Should be close to 2 ticks/sec
        assert!(
            intensity > 1.0 && intensity < 5.0,
            "Expected ~2 ticks/sec, got {}",
            intensity
        );
    }
}
