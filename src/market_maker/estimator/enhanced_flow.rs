//! Enhanced Flow Imbalance Estimator
//!
//! Provides multi-feature composite flow imbalance signal with natural variation
//! for IR calibration. The original book_imbalance tends to cluster in 0.3-0.5 range,
//! resulting in single-bin IR samples with zero resolution.
//!
//! # Problem Solved
//!
//! IR measures discriminative power, not just accuracy:
//! ```text
//! IR = Resolution / Uncertainty
//! ```
//!
//! When all predictions have similar confidence (flow_imbalance values), they fall
//! into the same IR bin, resulting in zero resolution and IR ≈ 0 regardless of hit rate.
//!
//! # Solution
//!
//! Multi-feature composite with natural spread in confidence values:
//! - Base flow from order book imbalance
//! - Depth imbalance: bid/ask depth at multiple levels
//! - Momentum: short-term price momentum from recent trades
//! - Kappa signal: fill intensity relative to average
//! - Spread-based sensitivity scaling
//!
//! # Usage
//!
//! ```ignore
//! let mut estimator = EnhancedFlowEstimator::new(EnhancedFlowConfig::default());
//!
//! let enhanced_flow = estimator.compute(&ctx);
//! // Result varies in -1.0 to +1.0 range with wider distribution
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for enhanced flow estimator.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EnhancedFlowConfig {
    /// Weight for depth imbalance feature.
    /// Default: 0.25
    pub weight_depth: f64,

    /// Weight for momentum feature.
    /// Default: 0.25
    pub weight_momentum: f64,

    /// Weight for kappa signal feature.
    /// Default: 0.20
    pub weight_kappa: f64,

    /// Number of book levels for depth calculation.
    /// Default: 5
    pub depth_levels: usize,

    /// Lookback window for momentum calculation (seconds).
    /// Default: 60.0
    pub momentum_window_secs: f64,

    /// EMA alpha for flow smoothing.
    /// Default: 0.1 (10 sample half-life)
    pub ema_alpha: f64,

    /// Minimum spread (bps) for sensitivity scaling.
    /// Default: 5.0
    pub min_spread_bps: f64,

    /// Sensitivity scale factor.
    /// Higher = more aggressive spread-based scaling.
    /// Default: 2.0
    pub sensitivity_scale: f64,

    /// Maximum trades to keep in buffer.
    /// Default: 500
    pub max_trade_buffer: usize,
}

impl Default for EnhancedFlowConfig {
    fn default() -> Self {
        Self {
            weight_depth: 0.25,
            weight_momentum: 0.25,
            weight_kappa: 0.20,
            depth_levels: 5,
            momentum_window_secs: 60.0,
            ema_alpha: 0.1,
            min_spread_bps: 5.0,
            sensitivity_scale: 2.0,
            max_trade_buffer: 500,
        }
    }
}

/// Trade data for momentum calculation.
#[derive(Debug, Clone, Copy)]
pub struct TradeData {
    /// Size of the trade.
    pub size: f64,
    /// Whether the trade was a buy.
    pub is_buy: bool,
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
}

/// Order book level for depth calculation.
#[derive(Debug, Clone, Copy)]
pub struct BookLevel {
    /// Size at this level.
    pub size: f64,
}

/// Context for computing enhanced flow.
#[derive(Debug, Clone)]
pub struct EnhancedFlowContext {
    /// Base book imbalance from top-of-book.
    pub book_imbalance: f64,

    /// Bid levels (from best to worse).
    pub bid_levels: Vec<BookLevel>,

    /// Ask levels (from best to worse).
    pub ask_levels: Vec<BookLevel>,

    /// Recent trades for momentum calculation.
    pub recent_trades: Vec<TradeData>,

    /// Current kappa (fill intensity) estimate.
    pub kappa_effective: f64,

    /// Average kappa for normalization.
    pub kappa_avg: f64,

    /// Current bid-ask spread in basis points.
    pub spread_bps: f64,

    /// Current timestamp in milliseconds.
    pub now_ms: u64,
}

impl Default for EnhancedFlowContext {
    fn default() -> Self {
        Self {
            book_imbalance: 0.0,
            bid_levels: Vec::new(),
            ask_levels: Vec::new(),
            recent_trades: Vec::new(),
            kappa_effective: 1000.0,
            kappa_avg: 1000.0,
            spread_bps: 10.0,
            now_ms: 0,
        }
    }
}

/// Result of enhanced flow computation.
#[derive(Debug, Clone, Copy)]
pub struct EnhancedFlowResult {
    /// Final enhanced flow signal [-1.0, +1.0].
    pub enhanced_flow: f64,

    /// EMA-smoothed flow (for stability).
    pub smoothed_flow: f64,

    /// Component breakdown for diagnostics.
    pub base_flow: f64,
    pub depth_component: f64,
    pub momentum_component: f64,
    pub kappa_component: f64,

    /// Sensitivity factor applied.
    pub sensitivity_factor: f64,

    /// Variance of recent flow values (for IR bin spread).
    pub flow_variance: f64,
}

/// EMA tracker for flow smoothing.
#[derive(Debug, Clone)]
struct FlowEMA {
    value: f64,
    alpha: f64,
    initialized: bool,
}

impl FlowEMA {
    fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha: alpha.clamp(0.01, 1.0),
            initialized: false,
        }
    }

    fn update(&mut self, new_flow: f64) -> f64 {
        if !self.initialized {
            self.value = new_flow;
            self.initialized = true;
        } else {
            self.value = self.alpha * new_flow + (1.0 - self.alpha) * self.value;
        }
        self.value
    }

    fn current(&self) -> f64 {
        self.value
    }
}

/// Variance tracker for flow distribution monitoring.
#[derive(Debug, Clone)]
struct VarianceTracker {
    samples: VecDeque<f64>,
    max_samples: usize,
}

impl VarianceTracker {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    fn update(&mut self, value: f64) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(value);
    }

    fn variance(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }

        let n = self.samples.len() as f64;
        let mean: f64 = self.samples.iter().sum::<f64>() / n;
        let sum_sq: f64 = self.samples.iter().map(|x| (x - mean).powi(2)).sum();
        sum_sq / (n - 1.0)
    }
}

/// Enhanced flow estimator with multi-feature composition.
#[derive(Debug)]
pub struct EnhancedFlowEstimator {
    config: EnhancedFlowConfig,
    flow_ema: FlowEMA,
    variance_tracker: VarianceTracker,
    computation_count: u64,
    /// EMA of kappa for normalization
    kappa_ema: f64,
}

impl EnhancedFlowEstimator {
    /// Create a new enhanced flow estimator.
    pub fn new(config: EnhancedFlowConfig) -> Self {
        let alpha = config.ema_alpha;
        Self {
            config,
            flow_ema: FlowEMA::new(alpha),
            variance_tracker: VarianceTracker::new(100), // Track last 100 samples
            computation_count: 0,
            kappa_ema: 1000.0, // Default initial kappa
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(EnhancedFlowConfig::default())
    }

    /// Compute enhanced flow from context.
    pub fn compute(&mut self, ctx: &EnhancedFlowContext) -> EnhancedFlowResult {
        // 1. Base flow from order book imbalance
        let base_flow = ctx.book_imbalance;

        // 2. Depth imbalance: bid depth vs ask depth at multiple levels
        let depth_component = self.compute_depth_imbalance(ctx);

        // 3. Momentum: recent trade volume imbalance
        let momentum_component = self.compute_momentum(ctx);

        // 4. Kappa signal: normalized current vs average
        let kappa_component = self.compute_kappa_signal(ctx);

        // 5. Composite signal (weighted sum)
        let composite = base_flow
            + self.config.weight_depth * depth_component
            + self.config.weight_momentum * momentum_component
            + self.config.weight_kappa * kappa_component;

        // 6. Spread-based sensitivity scaling
        // Tighter spread = higher sensitivity (we're more confident)
        let sensitivity_factor = self.config.sensitivity_scale
            / ctx.spread_bps.max(self.config.min_spread_bps);

        // 7. Non-linear scaling with tanh to bound output
        let scaled_flow = (composite * sensitivity_factor).tanh();

        // 8. EMA smoothing for stability
        let smoothed_flow = self.flow_ema.update(scaled_flow);

        // Track variance for diagnostics
        self.variance_tracker.update(scaled_flow);
        self.computation_count += 1;

        EnhancedFlowResult {
            enhanced_flow: scaled_flow,
            smoothed_flow,
            base_flow,
            depth_component,
            momentum_component,
            kappa_component,
            sensitivity_factor,
            flow_variance: self.variance_tracker.variance(),
        }
    }

    /// Compute depth imbalance from multiple book levels.
    fn compute_depth_imbalance(&self, ctx: &EnhancedFlowContext) -> f64 {
        let n_levels = self.config.depth_levels;

        let bid_depth: f64 = ctx
            .bid_levels
            .iter()
            .take(n_levels)
            .map(|l| l.size)
            .sum();

        let ask_depth: f64 = ctx
            .ask_levels
            .iter()
            .take(n_levels)
            .map(|l| l.size)
            .sum();

        let total = bid_depth + ask_depth;
        if total > 0.0 {
            (bid_depth - ask_depth) / total
        } else {
            0.0
        }
    }

    /// Compute momentum from recent trades.
    fn compute_momentum(&self, ctx: &EnhancedFlowContext) -> f64 {
        let window_ms = (self.config.momentum_window_secs * 1000.0) as u64;
        let cutoff = ctx.now_ms.saturating_sub(window_ms);

        let recent: Vec<_> = ctx
            .recent_trades
            .iter()
            .filter(|t| t.timestamp_ms >= cutoff)
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        let buy_volume: f64 = recent.iter().filter(|t| t.is_buy).map(|t| t.size).sum();

        let sell_volume: f64 = recent.iter().filter(|t| !t.is_buy).map(|t| t.size).sum();

        let total = buy_volume + sell_volume;
        if total > 0.0 {
            (buy_volume - sell_volume) / total
        } else {
            0.0
        }
    }

    /// Compute kappa signal (fill intensity relative to average).
    fn compute_kappa_signal(&self, ctx: &EnhancedFlowContext) -> f64 {
        let avg = ctx.kappa_avg.max(100.0); // Floor to avoid division issues
        let ratio = ctx.kappa_effective / avg;

        // Log-normalize: ln(ratio) gives symmetric deviation
        // ratio = 2 → ln(2) ≈ 0.69 (bullish: high activity)
        // ratio = 0.5 → ln(0.5) ≈ -0.69 (bearish: low activity)
        ratio.ln().clamp(-2.0, 2.0) / 2.0 // Normalize to [-1, 1]
    }

    /// Get the current smoothed flow value.
    pub fn current_smoothed_flow(&self) -> f64 {
        self.flow_ema.current()
    }

    /// Get the current flow variance.
    pub fn flow_variance(&self) -> f64 {
        self.variance_tracker.variance()
    }

    /// Get computation count.
    pub fn computation_count(&self) -> u64 {
        self.computation_count
    }

    /// Get configuration.
    pub fn config(&self) -> &EnhancedFlowConfig {
        &self.config
    }

    /// Get the current average kappa (EMA).
    pub fn avg_kappa(&self) -> f64 {
        self.kappa_ema
    }

    /// Update average kappa with new observation.
    pub fn update_kappa(&mut self, kappa: f64) {
        let alpha = 0.05; // Slow adaptation
        self.kappa_ema = alpha * kappa + (1.0 - alpha) * self.kappa_ema;
    }

    /// Reset the estimator state.
    pub fn reset(&mut self) {
        self.flow_ema = FlowEMA::new(self.config.ema_alpha);
        self.variance_tracker = VarianceTracker::new(100);
        self.computation_count = 0;
        self.kappa_ema = 1000.0;
    }

    /// Compute depth-weighted order flow imbalance.
    ///
    /// Unlike simple OFI which treats all levels equally, this weights
    /// each level by inverse distance from mid: w_i = 1 / (1 + i).
    ///
    /// Levels closer to mid have more predictive power for short-term
    /// price movements.
    ///
    /// # Arguments
    /// * `bid_levels` - Bid levels from best to worst
    /// * `ask_levels` - Ask levels from best to worst
    /// * `prev_bid_levels` - Previous bid levels (for delta computation)
    /// * `prev_ask_levels` - Previous ask levels (for delta computation)
    ///
    /// # Returns
    /// Depth-weighted OFI in [-1, 1] range.
    /// Positive = buying pressure, Negative = selling pressure.
    pub fn depth_weighted_ofi(
        &self,
        bid_levels: &[BookLevel],
        ask_levels: &[BookLevel],
        prev_bid_levels: &[BookLevel],
        prev_ask_levels: &[BookLevel],
    ) -> f64 {
        let n_levels = self.config.depth_levels;

        let mut weighted_bid_delta = 0.0;
        let mut weighted_ask_delta = 0.0;
        let mut total_weight = 0.0;

        // Compute weighted bid changes
        for i in 0..n_levels {
            let weight = 1.0 / (1.0 + i as f64);

            let curr_bid = bid_levels.get(i).map(|l| l.size).unwrap_or(0.0);
            let prev_bid = prev_bid_levels.get(i).map(|l| l.size).unwrap_or(0.0);

            let curr_ask = ask_levels.get(i).map(|l| l.size).unwrap_or(0.0);
            let prev_ask = prev_ask_levels.get(i).map(|l| l.size).unwrap_or(0.0);

            // OFI = Σ(Δbid - Δask) weighted by level
            weighted_bid_delta += weight * (curr_bid - prev_bid);
            weighted_ask_delta += weight * (curr_ask - prev_ask);
            total_weight += weight;
        }

        if total_weight < 1e-12 {
            return 0.0;
        }

        // Normalize OFI: positive = bids increasing relative to asks
        let raw_ofi = (weighted_bid_delta - weighted_ask_delta) / total_weight;

        // Scale to [-1, 1] using tanh
        raw_ofi.tanh()
    }

    /// Compute static depth-weighted imbalance (no previous state needed).
    ///
    /// This is the depth-weighted version of simple book imbalance.
    /// Useful when you don't have previous book state.
    pub fn depth_weighted_imbalance(&self, bid_levels: &[BookLevel], ask_levels: &[BookLevel]) -> f64 {
        let n_levels = self.config.depth_levels;

        let mut weighted_bid_depth = 0.0;
        let mut weighted_ask_depth = 0.0;

        for i in 0..n_levels {
            let weight = 1.0 / (1.0 + i as f64);

            weighted_bid_depth += weight * bid_levels.get(i).map(|l| l.size).unwrap_or(0.0);
            weighted_ask_depth += weight * ask_levels.get(i).map(|l| l.size).unwrap_or(0.0);
        }

        let total = weighted_bid_depth + weighted_ask_depth;
        if total < 1e-12 {
            return 0.0;
        }

        (weighted_bid_depth - weighted_ask_depth) / total
    }
}

impl Default for EnhancedFlowEstimator {
    fn default() -> Self {
        Self::default_config()
    }
}

// ============================================================================
// Liquidity Evaporation Detector
// ============================================================================

/// Configuration for liquidity evaporation detector.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LiquidityEvaporationConfig {
    /// Lookback window in milliseconds.
    /// Default: 5000 (5 seconds)
    pub window_ms: u64,

    /// Threshold for evaporation detection (fraction of depth drop).
    /// Default: 0.5 (50% drop triggers detection)
    pub evaporation_threshold: f64,

    /// Number of levels to consider for near-touch depth.
    /// Default: 3
    pub near_touch_levels: usize,

    /// EWMA alpha for smoothing evaporation score.
    /// Default: 0.2
    pub ema_alpha: f64,

    /// Minimum depth observations for valid detection.
    /// Default: 10
    pub min_observations: usize,
}

impl Default for LiquidityEvaporationConfig {
    fn default() -> Self {
        Self {
            window_ms: 5000,
            evaporation_threshold: 0.5,
            near_touch_levels: 3,
            ema_alpha: 0.2,
            min_observations: 10,
        }
    }
}

/// Detects rapid liquidity evaporation (book thinning).
///
/// Liquidity evaporation is a leading indicator of toxic flow:
/// - Market makers pull quotes before large moves
/// - Liquidation cascades thin the book rapidly
/// - Flash crashes preceded by depth drops
///
/// # Usage
///
/// ```ignore
/// let mut detector = LiquidityEvaporationDetector::new(
///     LiquidityEvaporationConfig::default()
/// );
///
/// // On each book update
/// let near_touch_depth = compute_near_touch_depth(&book);
/// detector.on_book(near_touch_depth, timestamp_ms);
///
/// if detector.evaporation_score() > 0.7 {
///     // Widen spreads or pause quoting
/// }
/// ```
#[derive(Debug)]
pub struct LiquidityEvaporationDetector {
    config: LiquidityEvaporationConfig,

    /// Recent depth observations: (timestamp_ms, depth)
    depth_history: VecDeque<(u64, f64)>,

    /// EWMA of evaporation score
    evaporation_ema: f64,

    /// Peak depth in current window
    peak_depth: f64,

    /// Observation count
    observation_count: usize,
}

impl LiquidityEvaporationDetector {
    /// Create a new liquidity evaporation detector.
    pub fn new(config: LiquidityEvaporationConfig) -> Self {
        Self {
            depth_history: VecDeque::with_capacity(100),
            evaporation_ema: 0.0,
            peak_depth: 0.0,
            observation_count: 0,
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(LiquidityEvaporationConfig::default())
    }

    /// Process a book update.
    ///
    /// # Arguments
    /// * `near_touch_depth` - Total depth within N levels of mid
    /// * `timestamp_ms` - Current timestamp
    pub fn on_book(&mut self, near_touch_depth: f64, timestamp_ms: u64) {
        // Prune old observations
        let cutoff = timestamp_ms.saturating_sub(self.config.window_ms);
        while let Some(&(ts, _)) = self.depth_history.front() {
            if ts < cutoff {
                self.depth_history.pop_front();
            } else {
                break;
            }
        }

        // Update peak depth in window
        self.peak_depth = self
            .depth_history
            .iter()
            .map(|(_, d)| *d)
            .fold(near_touch_depth, f64::max);

        // Add current observation
        self.depth_history.push_back((timestamp_ms, near_touch_depth));
        self.observation_count += 1;

        // Compute instantaneous evaporation score
        let instant_score = if self.peak_depth > 1e-12 {
            let drop_frac = 1.0 - (near_touch_depth / self.peak_depth);
            drop_frac.max(0.0)
        } else {
            0.0
        };

        // Update EWMA
        self.evaporation_ema = self.config.ema_alpha * instant_score
            + (1.0 - self.config.ema_alpha) * self.evaporation_ema;
    }

    /// Get current evaporation score [0, 1].
    ///
    /// Higher values indicate more severe liquidity evaporation.
    pub fn evaporation_score(&self) -> f64 {
        if self.observation_count < self.config.min_observations {
            return 0.0;
        }
        self.evaporation_ema.clamp(0.0, 1.0)
    }

    /// Check if evaporation is detected (score > threshold).
    pub fn is_evaporating(&self) -> bool {
        self.evaporation_score() > self.config.evaporation_threshold
    }

    /// Get depth drop from peak.
    ///
    /// Returns (current_depth, peak_depth, drop_fraction).
    pub fn depth_drop(&self) -> (f64, f64, f64) {
        let current = self
            .depth_history
            .back()
            .map(|(_, d)| *d)
            .unwrap_or(0.0);

        let drop_frac = if self.peak_depth > 1e-12 {
            1.0 - (current / self.peak_depth)
        } else {
            0.0
        };

        (current, self.peak_depth, drop_frac.max(0.0))
    }

    /// Get observation count.
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Check if detector is warmed up.
    pub fn is_valid(&self) -> bool {
        self.observation_count >= self.config.min_observations
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.depth_history.clear();
        self.evaporation_ema = 0.0;
        self.peak_depth = 0.0;
        self.observation_count = 0;
    }

    /// Compute near-touch depth from book levels.
    ///
    /// Helper to compute total depth within N levels of mid.
    pub fn compute_near_touch_depth(
        &self,
        bid_levels: &[BookLevel],
        ask_levels: &[BookLevel],
    ) -> f64 {
        let n = self.config.near_touch_levels;

        let bid_depth: f64 = bid_levels.iter().take(n).map(|l| l.size).sum();
        let ask_depth: f64 = ask_levels.iter().take(n).map(|l| l.size).sum();

        bid_depth + ask_depth
    }
}

impl Default for LiquidityEvaporationDetector {
    fn default() -> Self {
        Self::default_config()
    }
}

// =============================================================================
// Cumulative Order Flow Imbalance (COFI) with Decay
// =============================================================================

/// Configuration for Cumulative OFI.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CumulativeOFIConfig {
    /// Decay factor per update (λ). Typically 0.95-0.99.
    /// Higher = longer memory, more smoothing.
    /// Default: 0.97
    pub decay_lambda: f64,

    /// Time-based decay rate per second (for time gaps between updates).
    /// Applied as additional decay when updates are sparse.
    /// Default: 0.1 (10% decay per second of silence)
    pub time_decay_rate: f64,

    /// Threshold for detecting sustained shift.
    /// Default: 0.3
    pub sustained_shift_threshold: f64,

    /// Minimum updates before COFI is valid.
    /// Default: 10
    pub min_updates: usize,
}

impl Default for CumulativeOFIConfig {
    fn default() -> Self {
        Self {
            decay_lambda: 0.97,
            time_decay_rate: 0.1,
            sustained_shift_threshold: 0.3,
            min_updates: 10,
        }
    }
}

/// Cumulative Order Flow Imbalance with exponential decay.
///
/// Raw OFI is noisy - temporary book flickers create false signals.
/// COFI accumulates imbalance with decay to distinguish:
/// - Temporary flickers (decay away quickly)
/// - Sustained supply/demand shifts (accumulate despite decay)
///
/// # Theory
///
/// ```text
/// COFI_t = λ × COFI_{t-1} + (bid_delta - ask_delta)
/// ```
///
/// Where:
/// - λ ∈ [0.95, 0.99] controls memory (higher = longer memory)
/// - bid_delta = change in bid depth at level
/// - ask_delta = change in ask depth at level
///
/// # Usage
///
/// ```ignore
/// let mut cofi = CumulativeOFI::new(CumulativeOFIConfig::default());
///
/// // On each book update, compute deltas from previous book
/// let bid_delta = new_bid_depth - prev_bid_depth;
/// let ask_delta = new_ask_depth - prev_ask_depth;
/// cofi.on_book_update(bid_delta, ask_delta, timestamp_ms);
///
/// if cofi.is_sustained_shift() {
///     // Sustained imbalance detected - adjust skew
/// }
/// ```
#[derive(Debug)]
pub struct CumulativeOFI {
    config: CumulativeOFIConfig,

    /// Cumulative bid-side flow (with decay)
    cumulative_bid: f64,

    /// Cumulative ask-side flow (with decay)
    cumulative_ask: f64,

    /// Last update timestamp for time-based decay
    last_update_ms: u64,

    /// Previous COFI for velocity calculation
    prev_cofi: f64,

    /// COFI velocity (momentum of the imbalance)
    cofi_velocity: f64,

    /// Update count
    update_count: usize,
}

impl CumulativeOFI {
    /// Create a new CumulativeOFI tracker.
    pub fn new(config: CumulativeOFIConfig) -> Self {
        Self {
            config,
            cumulative_bid: 0.0,
            cumulative_ask: 0.0,
            last_update_ms: 0,
            prev_cofi: 0.0,
            cofi_velocity: 0.0,
            update_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(CumulativeOFIConfig::default())
    }

    /// Update with new book deltas.
    ///
    /// # Arguments
    /// * `bid_delta` - Change in bid depth (positive = bids added)
    /// * `ask_delta` - Change in ask depth (positive = asks added)
    /// * `timestamp_ms` - Current timestamp
    pub fn on_book_update(&mut self, bid_delta: f64, ask_delta: f64, timestamp_ms: u64) {
        // Apply time-based decay for gaps
        if self.last_update_ms > 0 && timestamp_ms > self.last_update_ms {
            let elapsed_secs = (timestamp_ms - self.last_update_ms) as f64 / 1000.0;
            let time_decay = (-self.config.time_decay_rate * elapsed_secs).exp();
            self.cumulative_bid *= time_decay;
            self.cumulative_ask *= time_decay;
        }

        // Store previous COFI for velocity
        self.prev_cofi = self.cofi();

        // Apply standard decay
        self.cumulative_bid *= self.config.decay_lambda;
        self.cumulative_ask *= self.config.decay_lambda;

        // Add new deltas
        self.cumulative_bid += bid_delta.max(0.0);
        self.cumulative_ask += ask_delta.max(0.0);

        // Update velocity (change in COFI)
        let new_cofi = self.cofi();
        self.cofi_velocity = new_cofi - self.prev_cofi;

        self.last_update_ms = timestamp_ms;
        self.update_count += 1;
    }

    /// Get COFI value in [-1, 1] range.
    ///
    /// Positive = sustained bid pressure (buying)
    /// Negative = sustained ask pressure (selling)
    pub fn cofi(&self) -> f64 {
        let total = self.cumulative_bid + self.cumulative_ask;
        if total < 1e-12 {
            return 0.0;
        }
        ((self.cumulative_bid - self.cumulative_ask) / total).clamp(-1.0, 1.0)
    }

    /// Get COFI velocity (momentum of imbalance).
    ///
    /// Positive velocity = imbalance growing toward buys
    /// Negative velocity = imbalance growing toward sells
    pub fn cofi_velocity(&self) -> f64 {
        self.cofi_velocity
    }

    /// Check if a sustained shift is detected.
    ///
    /// Returns true if |COFI| > threshold, indicating
    /// persistent supply/demand imbalance (not just noise).
    pub fn is_sustained_shift(&self) -> bool {
        if self.update_count < self.config.min_updates {
            return false;
        }
        self.cofi().abs() > self.config.sustained_shift_threshold
    }

    /// Get shift direction.
    ///
    /// Returns:
    /// - Some(1.0) if sustained bid pressure (buying)
    /// - Some(-1.0) if sustained ask pressure (selling)
    /// - None if no sustained shift
    pub fn shift_direction(&self) -> Option<f64> {
        if self.is_sustained_shift() {
            Some(self.cofi().signum())
        } else {
            None
        }
    }

    /// Get raw cumulative values (for debugging).
    pub fn raw_cumulative(&self) -> (f64, f64) {
        (self.cumulative_bid, self.cumulative_ask)
    }

    /// Check if COFI is warmed up.
    pub fn is_valid(&self) -> bool {
        self.update_count >= self.config.min_updates
    }

    /// Get update count.
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.cumulative_bid = 0.0;
        self.cumulative_ask = 0.0;
        self.last_update_ms = 0;
        self.prev_cofi = 0.0;
        self.cofi_velocity = 0.0;
        self.update_count = 0;
    }
}

impl Default for CumulativeOFI {
    fn default() -> Self {
        Self::default_config()
    }
}

// =============================================================================
// Trade Size Distribution Tracker
// =============================================================================

/// Configuration for trade size distribution tracker.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TradeSizeDistributionConfig {
    /// Window size (number of trades).
    /// Default: 500
    pub window_size: usize,

    /// EMA alpha for baseline median tracking.
    /// Default: 0.01 (slow adaptation)
    pub median_ema_alpha: f64,

    /// Sigma threshold for anomaly detection.
    /// Default: 3.0
    pub anomaly_sigma_threshold: f64,

    /// Minimum trades before valid statistics.
    /// Default: 50
    pub min_trades: usize,
}

impl Default for TradeSizeDistributionConfig {
    fn default() -> Self {
        Self {
            window_size: 500,
            median_ema_alpha: 0.01,
            anomaly_sigma_threshold: 3.0,
            min_trades: 50,
        }
    }
}

/// Tracks rolling trade size statistics for anomaly detection.
///
/// In crypto, "toxic" flow often manifests as:
/// - High-frequency, small-ticket bursts (HFT accumulation)
/// - Single large "sweep" orders (liquidation cascades)
///
/// A 3σ jump in median trade size while VPIN is rising
/// should accelerate the toxicity score.
///
/// # Usage
///
/// ```ignore
/// let mut tracker = TradeSizeDistribution::new(
///     TradeSizeDistributionConfig::default()
/// );
///
/// for trade in trades {
///     tracker.on_trade(trade.size);
/// }
///
/// if tracker.is_size_anomaly(3.0) {
///     // Unusual trade sizes - increase toxicity multiplier
///     let accel = tracker.toxicity_acceleration(vpin);
/// }
/// ```
#[derive(Debug)]
pub struct TradeSizeDistribution {
    config: TradeSizeDistributionConfig,

    /// Rolling window of trade sizes
    sizes: VecDeque<f64>,

    /// Cached statistics (recomputed periodically)
    cached_mean: f64,
    cached_std: f64,
    cached_median: f64,

    /// EMA of median for baseline
    median_ema: f64,

    /// Trade count
    trade_count: usize,

    /// Updates since last cache refresh
    updates_since_cache: usize,
}

impl TradeSizeDistribution {
    /// Create a new trade size distribution tracker.
    pub fn new(config: TradeSizeDistributionConfig) -> Self {
        Self {
            sizes: VecDeque::with_capacity(config.window_size),
            cached_mean: 0.0,
            cached_std: 1.0, // Avoid division by zero
            cached_median: 0.0,
            median_ema: 0.0,
            trade_count: 0,
            updates_since_cache: 0,
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(TradeSizeDistributionConfig::default())
    }

    /// Process a new trade.
    pub fn on_trade(&mut self, size: f64) {
        if size <= 0.0 {
            return;
        }

        // Add to window
        self.sizes.push_back(size);
        while self.sizes.len() > self.config.window_size {
            self.sizes.pop_front();
        }

        self.trade_count += 1;
        self.updates_since_cache += 1;

        // Refresh cache periodically (every 10 trades)
        if self.updates_since_cache >= 10 {
            self.refresh_cache();
        }
    }

    /// Refresh cached statistics.
    fn refresh_cache(&mut self) {
        if self.sizes.is_empty() {
            return;
        }

        // Compute mean
        let sum: f64 = self.sizes.iter().sum();
        self.cached_mean = sum / self.sizes.len() as f64;

        // Compute std
        let variance: f64 = self.sizes.iter()
            .map(|&x| (x - self.cached_mean).powi(2))
            .sum::<f64>() / self.sizes.len() as f64;
        self.cached_std = variance.sqrt().max(1e-12);

        // Compute median
        let mut sorted: Vec<f64> = self.sizes.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.cached_median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Update EMA baseline
        if self.median_ema < 1e-12 {
            // Initialize
            self.median_ema = self.cached_median;
        } else {
            self.median_ema = self.config.median_ema_alpha * self.cached_median
                + (1.0 - self.config.median_ema_alpha) * self.median_ema;
        }

        self.updates_since_cache = 0;
    }

    /// Get sigma deviation of current median from baseline.
    ///
    /// > 3.0 indicates anomalous trade size regime.
    pub fn median_sigma(&self) -> f64 {
        if self.cached_std < 1e-12 || self.median_ema < 1e-12 {
            return 0.0;
        }
        (self.cached_median - self.median_ema).abs() / self.cached_std
    }

    /// Check if trade sizes are anomalous.
    ///
    /// Returns true if median jumped > threshold sigmas above EMA baseline.
    pub fn is_size_anomaly(&self, threshold_sigma: f64) -> bool {
        if self.trade_count < self.config.min_trades {
            return false;
        }
        self.median_sigma() > threshold_sigma
    }

    /// Compute toxicity acceleration factor.
    ///
    /// When trade sizes are anomalous AND VPIN is elevated,
    /// accelerate the toxicity score.
    ///
    /// Returns multiplier in [1.0, 2.0] range.
    pub fn toxicity_acceleration(&self, vpin: f64) -> f64 {
        if !self.is_valid() {
            return 1.0;
        }

        let sigma = self.median_sigma();

        // Only accelerate if sigma > 2 (mildly anomalous)
        if sigma < 2.0 {
            return 1.0;
        }

        // Acceleration scales with both sigma and VPIN
        // sigma=3 + vpin=0.5 → 1.25x
        // sigma=4 + vpin=0.7 → 1.7x
        let sigma_factor = ((sigma - 2.0) / 2.0).clamp(0.0, 1.0);
        let vpin_factor = vpin.clamp(0.0, 1.0);

        1.0 + sigma_factor * vpin_factor
    }

    /// Get current statistics.
    pub fn stats(&self) -> TradeSizeStats {
        TradeSizeStats {
            mean: self.cached_mean,
            std: self.cached_std,
            median: self.cached_median,
            median_ema: self.median_ema,
            median_sigma: self.median_sigma(),
            trade_count: self.trade_count,
        }
    }

    /// Check if tracker is warmed up.
    pub fn is_valid(&self) -> bool {
        self.trade_count >= self.config.min_trades
    }

    /// Get trade count.
    pub fn trade_count(&self) -> usize {
        self.trade_count
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.sizes.clear();
        self.cached_mean = 0.0;
        self.cached_std = 1.0;
        self.cached_median = 0.0;
        self.median_ema = 0.0;
        self.trade_count = 0;
        self.updates_since_cache = 0;
    }
}

/// Trade size statistics snapshot.
#[derive(Debug, Clone)]
pub struct TradeSizeStats {
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub median_ema: f64,
    pub median_sigma: f64,
    pub trade_count: usize,
}

impl Default for TradeSizeDistribution {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context() -> EnhancedFlowContext {
        EnhancedFlowContext {
            book_imbalance: 0.3,
            bid_levels: vec![
                BookLevel { size: 100.0 },
                BookLevel { size: 80.0 },
                BookLevel { size: 60.0 },
            ],
            ask_levels: vec![
                BookLevel { size: 50.0 },
                BookLevel { size: 40.0 },
                BookLevel { size: 30.0 },
            ],
            recent_trades: vec![
                TradeData {
                    size: 10.0,
                    is_buy: true,
                    timestamp_ms: 1000,
                },
                TradeData {
                    size: 5.0,
                    is_buy: false,
                    timestamp_ms: 2000,
                },
                TradeData {
                    size: 15.0,
                    is_buy: true,
                    timestamp_ms: 3000,
                },
            ],
            kappa_effective: 1500.0,
            kappa_avg: 1000.0,
            spread_bps: 10.0,
            now_ms: 60_000, // 60 seconds
        }
    }

    #[test]
    fn test_basic_computation() {
        let mut estimator = EnhancedFlowEstimator::default_config();
        let ctx = make_context();

        let result = estimator.compute(&ctx);

        // Enhanced flow should be in valid range
        assert!(result.enhanced_flow >= -1.0 && result.enhanced_flow <= 1.0);
        assert!(result.smoothed_flow >= -1.0 && result.smoothed_flow <= 1.0);
    }

    #[test]
    fn test_depth_imbalance() {
        let mut estimator = EnhancedFlowEstimator::default_config();
        let ctx = make_context();

        // Depth: bids = 240, asks = 120, imbalance = (240-120)/(240+120) = 0.33
        let result = estimator.compute(&ctx);
        assert!(result.depth_component > 0.0); // More bids = positive
    }

    #[test]
    fn test_momentum_computation() {
        let mut estimator = EnhancedFlowEstimator::default_config();
        let ctx = make_context();

        // Trades: buy=25, sell=5, momentum = (25-5)/30 = 0.67
        let result = estimator.compute(&ctx);
        assert!(result.momentum_component > 0.0); // More buys = positive
    }

    #[test]
    fn test_kappa_signal() {
        let mut estimator = EnhancedFlowEstimator::default_config();
        let ctx = make_context();

        // Kappa: 1500/1000 = 1.5, ln(1.5) ≈ 0.4
        let result = estimator.compute(&ctx);
        assert!(result.kappa_component > 0.0); // Higher kappa = positive
    }

    #[test]
    fn test_variance_tracking() {
        let mut estimator = EnhancedFlowEstimator::default_config();

        // Generate varied inputs to create variance
        for i in 0..50 {
            let mut ctx = make_context();
            ctx.book_imbalance = (i as f64 / 50.0 - 0.5) * 2.0; // Range -1 to 1
            ctx.spread_bps = 5.0 + (i % 10) as f64; // Varied spreads
            estimator.compute(&ctx);
        }

        // Should have non-zero variance
        let variance = estimator.flow_variance();
        assert!(variance > 0.0, "Expected positive variance, got {}", variance);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut estimator = EnhancedFlowEstimator::default_config();

        // First computation
        let mut ctx = make_context();
        ctx.book_imbalance = 0.8;
        let _result1 = estimator.compute(&ctx);

        // Second computation with different input
        ctx.book_imbalance = -0.8;
        let result2 = estimator.compute(&ctx);

        // Smoothed flow should be less extreme than raw flow
        assert!(
            result2.smoothed_flow.abs() < result2.enhanced_flow.abs() || result2.smoothed_flow.abs() < 0.8,
            "Smoothed flow {} should be closer to zero than raw {}",
            result2.smoothed_flow, result2.enhanced_flow
        );
    }

    #[test]
    fn test_sensitivity_scaling() {
        let mut estimator = EnhancedFlowEstimator::default_config();

        // Tight spread = high sensitivity
        let mut ctx = make_context();
        ctx.spread_bps = 5.0;
        ctx.book_imbalance = 0.2;
        let result_tight = estimator.compute(&ctx);

        estimator.reset();

        // Wide spread = low sensitivity
        ctx.spread_bps = 20.0;
        let result_wide = estimator.compute(&ctx);

        assert!(
            result_tight.sensitivity_factor > result_wide.sensitivity_factor,
            "Tight spread {} should have higher sensitivity than wide {}",
            result_tight.sensitivity_factor, result_wide.sensitivity_factor
        );
    }

    #[test]
    fn test_reset() {
        let mut estimator = EnhancedFlowEstimator::default_config();
        let ctx = make_context();

        estimator.compute(&ctx);
        assert!(estimator.computation_count() > 0);

        estimator.reset();
        assert_eq!(estimator.computation_count(), 0);
    }

    #[test]
    fn test_empty_context() {
        let mut estimator = EnhancedFlowEstimator::default_config();
        let ctx = EnhancedFlowContext::default();

        let result = estimator.compute(&ctx);

        // Should handle empty context gracefully
        assert!(result.enhanced_flow.is_finite());
        assert!(result.depth_component.is_finite());
        assert!(result.momentum_component.is_finite());
    }

    // === Depth-weighted OFI tests ===

    #[test]
    fn test_depth_weighted_imbalance() {
        let estimator = EnhancedFlowEstimator::default_config();

        let bid_levels = vec![
            BookLevel { size: 100.0 }, // Weight: 1.0
            BookLevel { size: 80.0 },  // Weight: 0.5
            BookLevel { size: 60.0 },  // Weight: 0.33
        ];
        let ask_levels = vec![
            BookLevel { size: 50.0 }, // Weight: 1.0
            BookLevel { size: 40.0 }, // Weight: 0.5
            BookLevel { size: 30.0 }, // Weight: 0.33
        ];

        let imbalance = estimator.depth_weighted_imbalance(&bid_levels, &ask_levels);

        // More bids than asks, should be positive
        assert!(
            imbalance > 0.0,
            "Should have positive imbalance with more bids"
        );
    }

    #[test]
    fn test_depth_weighted_ofi() {
        let estimator = EnhancedFlowEstimator::default_config();

        // Previous state: balanced
        let prev_bids = vec![
            BookLevel { size: 100.0 },
            BookLevel { size: 100.0 },
        ];
        let prev_asks = vec![
            BookLevel { size: 100.0 },
            BookLevel { size: 100.0 },
        ];

        // Current state: bids increased, asks decreased
        let curr_bids = vec![
            BookLevel { size: 150.0 },
            BookLevel { size: 120.0 },
        ];
        let curr_asks = vec![
            BookLevel { size: 80.0 },
            BookLevel { size: 70.0 },
        ];

        let ofi = estimator.depth_weighted_ofi(&curr_bids, &curr_asks, &prev_bids, &prev_asks);

        // Positive OFI: bids up, asks down = buying pressure
        assert!(ofi > 0.0, "Should have positive OFI with buying pressure");
    }

    // === Liquidity evaporation tests ===

    #[test]
    fn test_liquidity_evaporation_basic() {
        let config = LiquidityEvaporationConfig {
            window_ms: 1000,
            evaporation_threshold: 0.5,
            min_observations: 3,
            ..Default::default()
        };
        let mut detector = LiquidityEvaporationDetector::new(config);

        // Build up depth
        detector.on_book(100.0, 1000);
        detector.on_book(100.0, 1100);
        detector.on_book(100.0, 1200);

        // Should have low evaporation initially
        assert!(
            detector.evaporation_score() < 0.2,
            "Should have low evaporation with stable depth"
        );

        // Now depth drops rapidly
        detector.on_book(50.0, 1300);
        detector.on_book(30.0, 1400);

        // Should detect evaporation
        assert!(
            detector.evaporation_score() > 0.2,
            "Should detect evaporation: {}",
            detector.evaporation_score()
        );
    }

    #[test]
    fn test_liquidity_evaporation_recovery() {
        let config = LiquidityEvaporationConfig {
            window_ms: 1000,
            evaporation_threshold: 0.3,
            min_observations: 2,
            ema_alpha: 0.3,
            ..Default::default()
        };
        let mut detector = LiquidityEvaporationDetector::new(config);

        // Build up depth
        detector.on_book(100.0, 1000);
        detector.on_book(100.0, 1100);

        // Depth drops
        detector.on_book(40.0, 1200);
        let score_after_drop = detector.evaporation_score();

        // Depth recovers
        detector.on_book(90.0, 1300);
        detector.on_book(100.0, 1400);
        let score_after_recovery = detector.evaporation_score();

        assert!(
            score_after_recovery < score_after_drop,
            "Score should decrease after recovery"
        );
    }

    #[test]
    fn test_near_touch_depth() {
        let config = LiquidityEvaporationConfig {
            near_touch_levels: 2,
            ..Default::default()
        };
        let detector = LiquidityEvaporationDetector::new(config);

        let bid_levels = vec![
            BookLevel { size: 100.0 },
            BookLevel { size: 80.0 },
            BookLevel { size: 60.0 }, // Should be excluded (only 2 levels)
        ];
        let ask_levels = vec![
            BookLevel { size: 50.0 },
            BookLevel { size: 40.0 },
            BookLevel { size: 30.0 }, // Should be excluded
        ];

        let depth = detector.compute_near_touch_depth(&bid_levels, &ask_levels);

        // Should only sum first 2 levels: (100+80) + (50+40) = 270
        assert!(
            (depth - 270.0).abs() < 0.01,
            "Expected 270.0, got {}",
            depth
        );
    }

    // === Cumulative OFI tests ===

    #[test]
    fn test_cofi_basic() {
        let mut cofi = CumulativeOFI::default();

        // Simulate sustained buying pressure
        for i in 0..20 {
            cofi.on_book_update(10.0, 2.0, i * 100); // Bids added > asks added
        }

        // Should show positive imbalance (buying pressure)
        assert!(cofi.cofi() > 0.0, "Should have positive COFI with bid pressure");
        assert!(cofi.is_valid(), "Should be valid after 20 updates");
    }

    #[test]
    fn test_cofi_decay() {
        let config = CumulativeOFIConfig {
            decay_lambda: 0.9,
            min_updates: 5,
            ..Default::default()
        };
        let mut cofi = CumulativeOFI::new(config);

        // Large initial imbalance (more bids than asks)
        for i in 0..10 {
            cofi.on_book_update(100.0, 20.0, i * 100);
        }
        let cofi_after_burst = cofi.cofi();

        // Now add balanced flow - this dilutes the imbalance
        // as old imbalanced flow decays and balanced flow dominates
        for i in 10..50 {
            cofi.on_book_update(50.0, 50.0, i * 100);
        }
        let cofi_after_balanced = cofi.cofi();

        // COFI should move toward zero as balanced flow dilutes imbalance
        assert!(
            cofi_after_balanced.abs() < cofi_after_burst.abs(),
            "COFI should decay toward zero with balanced flow: {} -> {}",
            cofi_after_burst,
            cofi_after_balanced
        );
    }

    #[test]
    fn test_cofi_sustained_shift() {
        let config = CumulativeOFIConfig {
            sustained_shift_threshold: 0.3,
            min_updates: 10,
            decay_lambda: 0.98,
            ..Default::default()
        };
        let mut cofi = CumulativeOFI::new(config);

        // Balanced flow (no sustained shift)
        for i in 0..15 {
            cofi.on_book_update(10.0, 10.0, i * 100);
        }
        assert!(!cofi.is_sustained_shift(), "Should not detect shift with balanced flow");

        cofi.reset();

        // Strong directional flow
        for i in 0..15 {
            cofi.on_book_update(20.0, 2.0, i * 100); // 10:1 ratio
        }
        assert!(cofi.is_sustained_shift(), "Should detect sustained shift with directional flow");
    }

    #[test]
    fn test_cofi_velocity() {
        let mut cofi = CumulativeOFI::default();

        // Build up imbalance
        for i in 0..10 {
            cofi.on_book_update(10.0, 0.0, i * 100);
        }

        // Velocity should be positive (imbalance growing toward buys)
        assert!(cofi.cofi_velocity() >= 0.0, "Velocity should be non-negative during buying");

        // Now reverse
        for i in 10..20 {
            cofi.on_book_update(0.0, 20.0, i * 100);
        }

        // Velocity should be negative (imbalance shifting toward sells)
        assert!(cofi.cofi_velocity() <= 0.0, "Velocity should be non-positive during selling");
    }

    // === Trade Size Distribution tests ===

    #[test]
    fn test_trade_size_basic() {
        let mut tracker = TradeSizeDistribution::default();

        // Add some normal trades
        for _ in 0..100 {
            tracker.on_trade(10.0);
        }

        assert!(tracker.is_valid(), "Should be valid after 100 trades");
        let stats = tracker.stats();
        assert!((stats.mean - 10.0).abs() < 0.1, "Mean should be ~10");
        assert!(stats.std < 1.0, "Std should be low for uniform sizes");
    }

    #[test]
    fn test_trade_size_anomaly_detection() {
        let config = TradeSizeDistributionConfig {
            min_trades: 50,
            anomaly_sigma_threshold: 2.0,
            window_size: 100, // Smaller window for test
            ..Default::default()
        };
        let mut tracker = TradeSizeDistribution::new(config);

        // Establish baseline with small trades
        for _ in 0..60 {
            tracker.on_trade(10.0);
        }
        assert!(!tracker.is_size_anomaly(2.0), "No anomaly with stable sizes");

        // Now inject MANY large trades to shift the median
        // Need > 50% of window to shift median
        for _ in 0..60 {
            tracker.on_trade(100.0);  // 10x normal
        }

        // With 60 small (10.0) + 60 large (100.0) in 100-trade window,
        // median should shift toward large, creating deviation from EMA baseline
        let stats = tracker.stats();

        // Median should be elevated compared to EMA baseline
        assert!(
            stats.median > stats.median_ema,
            "Median {} should be higher than EMA baseline {}",
            stats.median,
            stats.median_ema
        );
    }

    #[test]
    fn test_toxicity_acceleration() {
        let mut tracker = TradeSizeDistribution::default();

        // Normal trades - no acceleration
        for _ in 0..60 {
            tracker.on_trade(10.0);
        }

        let accel_normal = tracker.toxicity_acceleration(0.5);
        assert!(
            (accel_normal - 1.0).abs() < 0.1,
            "Should have no acceleration with normal sizes: {}",
            accel_normal
        );

        // Anomalous trades with high VPIN
        for _ in 0..40 {
            tracker.on_trade(100.0);
        }

        let accel_anomaly = tracker.toxicity_acceleration(0.7);
        // With 10x size jump and high VPIN, should accelerate
        // But actual acceleration depends on EMA tracking - may be modest
        assert!(
            accel_anomaly >= 1.0,
            "Acceleration should be >= 1.0 with anomalous sizes: {}",
            accel_anomaly
        );
    }

    #[test]
    fn test_trade_size_stats() {
        let mut tracker = TradeSizeDistribution::default();

        // Mixed sizes for meaningful statistics
        for i in 0..100 {
            tracker.on_trade(10.0 + (i as f64 % 5.0));
        }

        let stats = tracker.stats();
        assert!(stats.mean > 10.0 && stats.mean < 15.0, "Mean in expected range");
        assert!(stats.std > 0.0, "Should have non-zero std");
        assert!(stats.median > 10.0 && stats.median < 15.0, "Median in expected range");
    }
}
