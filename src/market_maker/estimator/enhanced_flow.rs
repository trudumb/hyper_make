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
}

impl Default for EnhancedFlowEstimator {
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
        let result1 = estimator.compute(&ctx);

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
}
