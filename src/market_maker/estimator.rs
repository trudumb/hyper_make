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

    // === Multi-Timescale EWMA ===
    /// Fast half-life for variance EWMA (~2 seconds, reacts to crashes)
    pub fast_half_life_ticks: f64,
    /// Medium half-life for variance EWMA (~10 seconds)
    pub medium_half_life_ticks: f64,
    /// Slow half-life for variance EWMA (~60 seconds, baseline)
    pub slow_half_life_ticks: f64,
    /// Half-life for kappa EWMA (in L2 updates)
    pub kappa_half_life_updates: f64,

    // === Momentum Detection ===
    /// Window for momentum calculation (milliseconds)
    pub momentum_window_ms: u64,

    // === Trade Flow Tracking ===
    /// Window for trade flow imbalance calculation (milliseconds)
    pub trade_flow_window_ms: u64,
    /// EWMA alpha for smoothing flow imbalance
    pub trade_flow_alpha: f64,

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

            // Multi-Timescale EWMA
            fast_half_life_ticks: 5.0,     // ~2 seconds - reacts to crashes
            medium_half_life_ticks: 20.0,  // ~10 seconds
            slow_half_life_ticks: 100.0,   // ~60 seconds - baseline
            kappa_half_life_updates: 30.0, // 30 L2 updates

            // Momentum Detection
            momentum_window_ms: 500, // Track signed returns over 500ms

            // Trade Flow Tracking
            trade_flow_window_ms: 1000, // Track buy/sell imbalance over 1s
            trade_flow_alpha: 0.1,      // EWMA smoothing for flow

            // Regime Detection - true toxic flow detection
            jump_ratio_threshold: 3.0, // RV/BV > 3.0 = toxic (lower values are normal bid-ask bounce)

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

    /// Create config with custom toxic threshold
    pub fn with_toxic_threshold(mut self, threshold: f64) -> Self {
        self.jump_ratio_threshold = threshold;
        self
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
// Single-Scale Bipower Variation (Building Block)
// ============================================================================

/// Single-timescale RV/BV tracker - building block for multi-scale estimator.
#[derive(Debug)]
struct SingleScaleBipower {
    /// EWMA decay factor (per tick)
    alpha: f64,
    /// Realized variance (includes jumps): EWMA of r²
    rv: f64,
    /// Bipower variation (excludes jumps): EWMA of (π/2)|r_t||r_{t-1}|
    bv: f64,
    /// Last absolute log return (for BV calculation)
    last_abs_return: Option<f64>,
}

impl SingleScaleBipower {
    fn new(half_life_ticks: f64, default_var: f64) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 1.0),
            rv: default_var,
            bv: default_var,
            last_abs_return: None,
        }
    }

    fn update(&mut self, log_return: f64) {
        let abs_return = log_return.abs();

        // RV: EWMA of r²
        let rv_obs = log_return.powi(2);
        self.rv = self.alpha * rv_obs + (1.0 - self.alpha) * self.rv;

        // BV: EWMA of (π/2) × |r_t| × |r_{t-1}|
        if let Some(last_abs) = self.last_abs_return {
            let bv_obs = std::f64::consts::FRAC_PI_2 * abs_return * last_abs;
            self.bv = self.alpha * bv_obs + (1.0 - self.alpha) * self.bv;
        }

        self.last_abs_return = Some(abs_return);
    }

    /// Total volatility including jumps (√RV)
    fn sigma_total(&self) -> f64 {
        self.rv.sqrt().clamp(1e-7, 0.05)
    }

    /// Clean volatility excluding jumps (√BV)
    fn sigma_clean(&self) -> f64 {
        self.bv.sqrt().clamp(1e-7, 0.05)
    }

    /// Jump ratio: RV/BV (1.0 = normal, >2 = jumps)
    fn jump_ratio(&self) -> f64 {
        if self.bv > 1e-12 {
            (self.rv / self.bv).clamp(0.1, 100.0)
        } else {
            1.0
        }
    }
}

// ============================================================================
// Multi-Timescale Bipower Estimator
// ============================================================================

/// Multi-timescale volatility with fast/medium/slow components.
///
/// Fast (~2s): Reacts quickly to crashes, used for early warning
/// Medium (~10s): Balanced responsiveness
/// Slow (~60s): Stable baseline for pricing
#[derive(Debug)]
struct MultiScaleBipowerEstimator {
    fast: SingleScaleBipower,   // ~5 ticks / ~2 seconds
    medium: SingleScaleBipower, // ~20 ticks / ~10 seconds
    slow: SingleScaleBipower,   // ~100 ticks / ~60 seconds
    last_vwap: Option<f64>,
    tick_count: usize,
}

impl MultiScaleBipowerEstimator {
    fn new(config: &EstimatorConfig) -> Self {
        let default_var = config.default_sigma.powi(2);
        Self {
            fast: SingleScaleBipower::new(config.fast_half_life_ticks, default_var),
            medium: SingleScaleBipower::new(config.medium_half_life_ticks, default_var),
            slow: SingleScaleBipower::new(config.slow_half_life_ticks, default_var),
            last_vwap: None,
            tick_count: 0,
        }
    }

    /// Process a completed volume bucket.
    fn on_bucket(&mut self, bucket: &VolumeBucket) {
        if let Some(prev_vwap) = self.last_vwap {
            if bucket.vwap > 0.0 && prev_vwap > 0.0 {
                let log_return = (bucket.vwap / prev_vwap).ln();
                self.fast.update(log_return);
                self.medium.update(log_return);
                self.slow.update(log_return);
                self.tick_count += 1;
            }
        }
        self.last_vwap = Some(bucket.vwap);
    }

    /// Get the log return for the most recent bucket (for momentum tracking)
    fn last_log_return(&self, bucket: &VolumeBucket) -> Option<f64> {
        self.last_vwap.and_then(|prev| {
            if bucket.vwap > 0.0 && prev > 0.0 {
                Some((bucket.vwap / prev).ln())
            } else {
                None
            }
        })
    }

    /// Clean sigma (BV-based) for spread pricing.
    /// Uses slow timescale for stability.
    fn sigma_clean(&self) -> f64 {
        self.slow.sigma_clean()
    }

    /// Total sigma (RV-based) for risk assessment.
    /// Blends fast + slow: uses fast when market is accelerating.
    fn sigma_total(&self) -> f64 {
        let fast = self.fast.sigma_total();
        let slow = self.slow.sigma_total();

        // If fast >> slow, market is accelerating - trust fast more
        let ratio = fast / slow.max(1e-9);

        if ratio > 1.5 {
            // Acceleration: blend toward fast
            let weight = ((ratio - 1.0) / 3.0).clamp(0.0, 0.7);
            weight * fast + (1.0 - weight) * slow
        } else {
            // Stable: prefer slow for less noise
            0.2 * fast + 0.8 * slow
        }
    }

    /// Effective sigma for inventory skew.
    /// Blends clean and total based on jump regime.
    fn sigma_effective(&self) -> f64 {
        let clean = self.sigma_clean();
        let total = self.sigma_total();
        let jump_ratio = self.jump_ratio_fast();

        // At ratio=1: pure clean (no jumps)
        // At ratio=3: 67% total (jumps dominant)
        // At ratio=5: 80% total
        let jump_weight = 1.0 - (1.0 / jump_ratio.max(1.0));
        let jump_weight = jump_weight.clamp(0.0, 0.85);

        (1.0 - jump_weight) * clean + jump_weight * total
    }

    /// Fast jump ratio (detects recent jumps quickly)
    fn jump_ratio_fast(&self) -> f64 {
        self.fast.jump_ratio()
    }

    /// Medium jump ratio (more stable signal)
    #[allow(dead_code)]
    fn jump_ratio_medium(&self) -> f64 {
        self.medium.jump_ratio()
    }

    fn tick_count(&self) -> usize {
        self.tick_count
    }
}

// ============================================================================
// Momentum Detector (Directional Flow)
// ============================================================================

/// Detects directional momentum from signed VWAP returns.
///
/// Tracks signed (not absolute) returns to detect falling/rising knife patterns.
#[derive(Debug)]
struct MomentumDetector {
    /// Recent (timestamp_ms, log_return) pairs
    returns: VecDeque<(u64, f64)>,
    /// Window for momentum calculation (ms)
    window_ms: u64,
}

impl MomentumDetector {
    fn new(window_ms: u64) -> Self {
        Self {
            returns: VecDeque::with_capacity(100),
            window_ms,
        }
    }

    /// Add a new VWAP-based return
    fn on_bucket(&mut self, end_time_ms: u64, log_return: f64) {
        self.returns.push_back((end_time_ms, log_return));

        // Expire old returns (keep 2x window for safety)
        let cutoff = end_time_ms.saturating_sub(self.window_ms * 2);
        while self
            .returns
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.returns.pop_front();
        }
    }

    /// Signed momentum in bps over the configured window
    fn momentum_bps(&self, now_ms: u64) -> f64 {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        let sum: f64 = self
            .returns
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, r)| r)
            .sum();
        sum * 10_000.0 // Convert to bps
    }

    /// Falling knife score: 0 = normal, 1+ = severe downward momentum
    fn falling_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);

        // Only trigger on negative momentum
        if momentum >= 0.0 {
            return 0.0;
        }

        // Score: -20 bps = 1.0, -40 bps = 2.0, etc.
        (momentum.abs() / 20.0).clamp(0.0, 3.0)
    }

    /// Rising knife score (for protecting asks during pumps)
    fn rising_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);

        if momentum <= 0.0 {
            return 0.0;
        }

        (momentum / 20.0).clamp(0.0, 3.0)
    }
}

// ============================================================================
// Trade Flow Tracker (Buy/Sell Imbalance)
// ============================================================================

/// Tracks buy vs sell aggressor imbalance from trade tape.
///
/// Uses the trade side field from Hyperliquid ("B" = buy aggressor, "S" = sell aggressor)
/// to detect directional order flow before it shows up in price.
#[derive(Debug)]
struct TradeFlowTracker {
    /// (timestamp_ms, signed_volume): positive = buy aggressor
    trades: VecDeque<(u64, f64)>,
    /// Rolling window (ms)
    window_ms: u64,
    /// EWMA smoothed imbalance
    ewma_imbalance: f64,
    /// EWMA alpha
    alpha: f64,
}

impl TradeFlowTracker {
    fn new(window_ms: u64, alpha: f64) -> Self {
        Self {
            trades: VecDeque::with_capacity(500),
            window_ms,
            ewma_imbalance: 0.0,
            alpha,
        }
    }

    /// Add a trade from the tape.
    /// is_buy_aggressor: true if buyer was taker (lifted the ask)
    fn on_trade(&mut self, timestamp_ms: u64, size: f64, is_buy_aggressor: bool) {
        let signed = if is_buy_aggressor { size } else { -size };
        self.trades.push_back((timestamp_ms, signed));

        // Expire old trades
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self
            .trades
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.trades.pop_front();
        }

        // Update EWMA
        let instant = self.compute_instant_imbalance();
        self.ewma_imbalance = self.alpha * instant + (1.0 - self.alpha) * self.ewma_imbalance;
    }

    /// Compute instantaneous imbalance: (buy - sell) / total
    fn compute_instant_imbalance(&self) -> f64 {
        let (buy_vol, sell_vol) =
            self.trades.iter().fold(
                (0.0, 0.0),
                |(b, s), (_, v)| {
                    if *v > 0.0 {
                        (b + v, s)
                    } else {
                        (b, s - v)
                    }
                },
            );
        let total = buy_vol + sell_vol;
        if total < 1e-12 {
            0.0
        } else {
            (buy_vol - sell_vol) / total
        }
    }

    /// Smoothed flow imbalance [-1, 1]
    /// Negative = sell pressure, Positive = buy pressure
    fn imbalance(&self) -> f64 {
        self.ewma_imbalance.clamp(-1.0, 1.0)
    }

    /// Is there dominant selling (for bid protection)?
    #[allow(dead_code)]
    fn is_sell_pressure(&self) -> bool {
        self.ewma_imbalance < -0.25
    }

    /// Is there dominant buying (for ask protection)?
    #[allow(dead_code)]
    fn is_buy_pressure(&self) -> bool {
        self.ewma_imbalance > 0.25
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

    #[allow(dead_code)]
    fn kappa(&self) -> f64 {
        self.kappa.clamp(1.0, 10000.0)
    }

    #[allow(dead_code)]
    fn update_count(&self) -> usize {
        self.update_count
    }
}

// ============================================================================
// Fill Rate Kappa Estimator (Trade Distance Distribution)
// ============================================================================

/// Estimates κ from trade execution distance distribution.
///
/// This is the CORRECT κ for the GLFT formula:
/// λ(δ) = A × exp(-κ × δ)  where λ is fill rate at distance δ
///
/// For exponential distribution: E[δ] = 1/κ, so κ = 1/E[δ]
///
/// This measures WHERE trades actually execute relative to mid,
/// NOT the shape of the order book (which is a different concept).
#[derive(Debug)]
struct FillRateKappaEstimator {
    /// Rolling observations: (distance, volume, timestamp)
    observations: VecDeque<(f64, f64, u64)>,
    /// Rolling window (ms)
    window_ms: u64,
    /// Running volume-weighted distance sum (for efficiency)
    volume_weighted_distance: f64,
    /// Running total volume
    total_volume: f64,
    /// EWMA smoothed kappa
    kappa: f64,
    /// EWMA alpha
    alpha: f64,
    /// Update count for warmup
    update_count: usize,
}

impl FillRateKappaEstimator {
    fn new(window_ms: u64, half_life_ticks: f64, default_kappa: f64) -> Self {
        Self {
            observations: VecDeque::with_capacity(1000),
            window_ms,
            volume_weighted_distance: 0.0,
            total_volume: 0.0,
            kappa: default_kappa,
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 0.5),
            update_count: 0,
        }
    }

    /// Process a trade. mid must be the mid price at time of trade.
    fn on_trade(&mut self, timestamp_ms: u64, price: f64, size: f64, mid: f64) {
        if mid <= 0.0 || size <= 0.0 || price <= 0.0 {
            return;
        }

        // Distance from mid as fraction
        let distance = ((price - mid) / mid).abs();

        // Minimum distance floor to avoid division issues
        // Trades AT the mid get a small floor distance
        let distance = distance.max(0.00001); // 0.1 bps floor

        // Add observation
        self.observations.push_back((distance, size, timestamp_ms));
        self.volume_weighted_distance += distance * size;
        self.total_volume += size;

        // Expire old observations
        self.expire_old(timestamp_ms);

        // Update kappa estimate
        self.update_kappa();
    }

    fn expire_old(&mut self, now: u64) {
        let cutoff = now.saturating_sub(self.window_ms);
        while let Some((dist, size, ts)) = self.observations.front() {
            if *ts < cutoff {
                self.volume_weighted_distance -= dist * size;
                self.total_volume -= size;
                self.observations.pop_front();
            } else {
                break;
            }
        }

        // Ensure running sums don't go negative due to float precision
        self.volume_weighted_distance = self.volume_weighted_distance.max(0.0);
        self.total_volume = self.total_volume.max(0.0);
    }

    fn update_kappa(&mut self) {
        if self.total_volume < 1e-9 {
            return;
        }

        // Volume-weighted average distance
        let avg_distance = self.volume_weighted_distance / self.total_volume;

        // For exponential distribution λ(δ) = A·exp(-κδ):
        // E[δ] = 1/κ (mean of exponential)
        // Therefore: κ = 1/E[δ]
        if avg_distance > 1e-8 {
            let kappa_instant = 1.0 / avg_distance;

            // Clamp to reasonable range
            // κ = 1000 means avg distance = 10 bps
            // κ = 5000 means avg distance = 2 bps
            // κ = 500 means avg distance = 20 bps
            let kappa_clamped = kappa_instant.clamp(100.0, 10000.0);

            // EWMA update
            self.kappa = self.alpha * kappa_clamped + (1.0 - self.alpha) * self.kappa;
            self.update_count += 1;

            if self.update_count.is_multiple_of(100) {
                debug!(
                    observations = self.observations.len(),
                    avg_distance_bps = %format!("{:.2}", avg_distance * 10000.0),
                    kappa_instant = %format!("{:.0}", kappa_instant),
                    kappa_ewma = %format!("{:.0}", self.kappa),
                    "Fill-rate kappa updated from trade distances"
                );
            }
        }
    }

    fn kappa(&self) -> f64 {
        self.kappa.clamp(100.0, 10000.0)
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
// Book Structure Estimator (L2 Order Book Analysis)
// ============================================================================

/// Analyzes L2 order book structure for auxiliary quote adjustments.
///
/// Provides two key signals:
/// 1. **Book Imbalance** [-1, 1]: Bid/ask depth asymmetry
///    - Positive = more bids than asks (buying pressure)
///    - Used for directional skew adjustment
///
/// 2. **Liquidity Gamma Multiplier** [1.0, 2.0]: Thin book detection
///    - Scales γ up when near-touch liquidity is below average
///    - Protects against adverse selection in thin markets
#[derive(Debug)]
struct BookStructureEstimator {
    /// EWMA smoothed book imbalance [-1, 1]
    imbalance: f64,
    /// Current near-touch depth (within 10 bps of mid)
    near_touch_depth: f64,
    /// Rolling reference depth for comparison
    reference_depth: f64,
    /// EWMA smoothing factor
    alpha: f64,
    /// Number of levels to consider for imbalance
    imbalance_levels: usize,
    /// Maximum distance for near-touch liquidity (as fraction)
    near_touch_distance: f64,
}

impl BookStructureEstimator {
    fn new() -> Self {
        Self {
            imbalance: 0.0,
            near_touch_depth: 0.0,
            reference_depth: 1.0, // Start with 1.0 to avoid division issues
            alpha: 0.1,
            imbalance_levels: 5,
            near_touch_distance: 0.001, // 10 bps
        }
    }

    /// Update with new L2 book data.
    fn update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        if mid <= 0.0 {
            return;
        }

        // 1. Calculate bid/ask imbalance from top N levels
        let bid_depth: f64 = bids
            .iter()
            .take(self.imbalance_levels)
            .map(|(_, sz)| sz)
            .sum();
        let ask_depth: f64 = asks
            .iter()
            .take(self.imbalance_levels)
            .map(|(_, sz)| sz)
            .sum();

        let total = bid_depth + ask_depth;
        if total > 1e-9 {
            let instant_imbalance = (bid_depth - ask_depth) / total;
            self.imbalance = self.alpha * instant_imbalance + (1.0 - self.alpha) * self.imbalance;
        }

        // 2. Calculate near-touch liquidity (depth within 10 bps of mid)
        let bid_near: f64 = bids
            .iter()
            .take_while(|(px, _)| (mid - px) / mid <= self.near_touch_distance)
            .map(|(_, sz)| sz)
            .sum();
        let ask_near: f64 = asks
            .iter()
            .take_while(|(px, _)| (px - mid) / mid <= self.near_touch_distance)
            .map(|(_, sz)| sz)
            .sum();
        self.near_touch_depth = bid_near + ask_near;

        // 3. Update reference depth (slow-moving average)
        // Use very slow decay to establish "normal" liquidity baseline
        self.reference_depth =
            0.99 * self.reference_depth + 0.01 * self.near_touch_depth.max(0.001);
    }

    /// Get current book imbalance [-1, 1].
    /// Positive = more bids (buying pressure), Negative = more asks (selling pressure).
    fn imbalance(&self) -> f64 {
        self.imbalance.clamp(-1.0, 1.0)
    }

    /// Get gamma multiplier for thin book conditions [1.0, 2.0].
    /// Returns > 1.0 when near-touch liquidity is below reference.
    fn gamma_multiplier(&self) -> f64 {
        if self.near_touch_depth >= self.reference_depth {
            1.0
        } else {
            // Thin book → scale gamma up (wider spreads for protection)
            // sqrt scaling: 4x thinner → 2x gamma
            let ratio = self.reference_depth / self.near_touch_depth.max(0.001);
            ratio.sqrt().clamp(1.0, 2.0)
        }
    }
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

/// Econometric parameter estimator with multi-timescale variance, momentum detection,
/// and trade flow tracking for robust adverse selection protection.
///
/// Pipeline:
/// 1. Raw trades → Volume Clock → Normalized volume buckets with VWAP
/// 2. VWAP returns → Multi-Scale Bipower → sigma_clean, sigma_total, sigma_effective
/// 3. VWAP returns → Momentum Detector → falling/rising knife scores
/// 4. Trade tape → Flow Tracker → buy/sell imbalance
/// 5. Trade tape → Fill Rate Kappa → trade distance distribution (CORRECT κ for GLFT)
/// 6. L2 Book → Book analysis for auxiliary adjustments (imbalance, etc.)
/// 7. Regime detection: fast jump_ratio > threshold = toxic
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
    /// Fill-rate kappa estimator from trade distances (PRIMARY κ for GLFT)
    fill_rate_kappa: FillRateKappaEstimator,
    /// Weighted kappa estimator from L2 book (kept for auxiliary use)
    #[allow(dead_code)]
    book_kappa: WeightedKappaEstimator,
    /// Volume tick arrival estimator
    arrival: VolumeTickArrivalEstimator,
    /// Current mid price
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

        // Fill-rate kappa from trade distances (PRIMARY for GLFT)
        // Use 5-minute window, medium half-life, default to 2000 (typical for liquid markets)
        let fill_rate_kappa = FillRateKappaEstimator::new(
            300_000, // 5 minute window
            config.medium_half_life_ticks,
            2000.0, // Default: avg trade distance = 5 bps
        );

        // Book kappa (kept for auxiliary use, may remove later)
        let book_kappa = WeightedKappaEstimator::new(
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
            fill_rate_kappa,
            book_kappa,
            arrival,
            current_mid: 0.0,
            current_time_ms: 0,
        }
    }

    /// Update current mid price.
    pub fn on_mid_update(&mut self, mid_price: f64) {
        self.current_mid = mid_price;
    }

    /// Process a new trade (feeds into volume clock, flow tracker, AND fill-rate kappa).
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

        // Feed into fill-rate kappa estimator (trade distance from mid)
        // This is the PRIMARY κ source for GLFT
        if self.current_mid > 0.0 {
            self.fill_rate_kappa
                .on_trade(timestamp_ms, price, size, self.current_mid);
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
                kappa = %format!("{:.0}", self.fill_rate_kappa.kappa()),
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
        self.current_mid = mid;
        self.book_kappa.update(bids, asks, mid);
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

    /// Get current fill-rate kappa estimate (κ).
    /// This is estimated from trade execution distance distribution,
    /// which is the CORRECT κ for the GLFT formula.
    pub fn kappa(&self) -> f64 {
        self.fill_rate_kappa.kappa()
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
            && self.fill_rate_kappa.update_count() >= self.config.min_l2_updates
    }

    /// Get current warmup progress.
    /// Returns (volume_ticks, min_volume_ticks, l2_updates, min_l2_updates)
    pub fn warmup_progress(&self) -> (usize, usize, usize, usize) {
        (
            self.multi_scale.tick_count(),
            self.config.min_volume_ticks,
            self.fill_rate_kappa.update_count(),
            self.config.min_l2_updates,
        )
    }

    /// Get simplified warmup progress for legacy compatibility.
    /// Returns (current_samples, min_samples) based on volume ticks.
    pub fn warmup_progress_simple(&self) -> (usize, usize) {
        (self.multi_scale.tick_count(), self.config.min_volume_ticks)
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
            fast_half_life_ticks: 5.0,
            medium_half_life_ticks: 10.0,
            slow_half_life_ticks: 50.0,
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
    fn test_single_scale_bipower_no_jumps() {
        let mut bv = SingleScaleBipower::new(10.0, 0.001_f64.powi(2));

        // Feed stable returns (no jumps) - small oscillations
        let vwaps: [f64; 8] = [100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1];
        let mut last_vwap: f64 = vwaps[0];
        for vwap in vwaps.iter().skip(1) {
            let log_return = (vwap / last_vwap).ln();
            bv.update(log_return);
            last_vwap = *vwap;
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
    fn test_single_scale_bipower_with_jump() {
        let mut bv = SingleScaleBipower::new(5.0, 0.001_f64.powi(2));

        // Feed returns with a sudden jump
        // Normal, normal, JUMP, normal, normal
        let vwaps: [f64; 7] = [100.0, 100.1, 100.0, 105.0, 105.1, 105.0, 105.1];
        let mut last_vwap: f64 = vwaps[0];
        for vwap in vwaps.iter().skip(1) {
            let log_return = (vwap / last_vwap).ln();
            bv.update(log_return);
            last_vwap = *vwap;
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

        // Feed initial L2 book to set current_mid (needed for fill-rate kappa)
        let bids = vec![(99.9, 5.0), (99.8, 10.0), (99.7, 15.0)];
        let asks = vec![(100.1, 5.0), (100.2, 10.0), (100.3, 15.0)];
        estimator.on_l2_book(&bids, &asks, 100.0);

        // Feed trades to fill buckets (need 5 volume ticks)
        // These will also feed into fill_rate_kappa since current_mid is set
        let mut time = 1000u64;
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 0.5;
            // Alternate buy/sell to simulate balanced flow
            let is_buy = i % 2 == 0;
            estimator.on_trade(time, price, 0.5, Some(is_buy)); // 0.5 per trade, 2 trades per bucket
            time += 100;
        }

        // Feed more L2 books (total needs min_l2_updates which is checked via fill_rate_kappa)
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
        for i in 0..50 {
            let is_buy = i % 2 == 0;
            estimator.on_trade(time, 100.0, 1.0, Some(is_buy)); // Each trade = 1 bucket
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
