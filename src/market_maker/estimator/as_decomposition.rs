//! Adverse Selection Decomposition
//!
//! Decomposes total adverse selection into:
//!
//! 1. **Permanent component**: Price impact that persists (information)
//! 2. **Temporary component**: Price impact that reverses (liquidity)
//! 3. **Timing component**: Cost of market timing/latency
//!
//! # Model
//!
//! ```text
//! AS_total = AS_permanent + AS_temporary + AS_timing
//!
//! AS_permanent ≈ E[Δp(5min) | fill]  (long-horizon impact)
//! AS_temporary = E[Δp(1s) | fill] - AS_permanent
//! AS_timing = cost from execution delay
//! ```
//!
//! # Horizons
//!
//! - 1 second: Immediate impact (includes temporary)
//! - 5 seconds: Short-term impact
//! - 30 seconds: Medium-term impact
//! - 5 minutes: Permanent proxy (information content)
//!
//! # Usage
//!
//! ```ignore
//! let mut decomp = ASDecomposition::new(ASDecompConfig::default());
//!
//! // Record fills with market state
//! decomp.on_fill(&fill_info);
//!
//! // Update with price observations
//! decomp.on_price_update(new_mid, timestamp_ms);
//!
//! // Get decomposition
//! let permanent = decomp.permanent_as_bps();
//! let temporary = decomp.temporary_as_bps();
//! let total = decomp.total_as_bps();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Constants
// ============================================================================

/// Standard measurement horizons in milliseconds
pub const HORIZONS_MS: [u64; 4] = [1_000, 5_000, 30_000, 300_000];

/// Horizon names for display
pub const HORIZON_NAMES: [&str; 4] = ["1s", "5s", "30s", "5min"];

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for AS decomposition
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ASDecompConfig {
    /// Measurement horizons in milliseconds
    pub horizons_ms: Vec<u64>,

    /// Half-life in fills for EWMA (default: 100 fills)
    pub ewma_half_life_fills: usize,

    /// Minimum fills before estimates are reliable
    pub min_fills: usize,

    /// Maximum pending measurements to track
    pub max_pending: usize,

    /// Permanent horizon index (default: 3 = 5min)
    pub permanent_horizon_idx: usize,

    /// Immediate horizon index (default: 0 = 1s)
    pub immediate_horizon_idx: usize,

    /// Minimum significance level for decomposition
    pub min_significance: f64,
}

impl Default for ASDecompConfig {
    fn default() -> Self {
        Self {
            horizons_ms: HORIZONS_MS.to_vec(),
            ewma_half_life_fills: 100,
            min_fills: 30,
            max_pending: 1000,
            permanent_horizon_idx: 3,
            immediate_horizon_idx: 0,
            min_significance: 0.5,
        }
    }
}

// ============================================================================
// Fill Information
// ============================================================================

/// Information about a fill for AS measurement
#[derive(Debug, Clone, Copy)]
pub struct FillInfo {
    /// Unique fill identifier
    pub fill_id: u64,

    /// Fill timestamp (milliseconds)
    pub timestamp_ms: u64,

    /// Mid price at time of fill
    pub fill_mid: f64,

    /// Fill size
    pub size: f64,

    /// Whether this was a buy fill (we sold to a buyer)
    pub is_buy: bool,

    /// Optional: volatility at time of fill (for normalization)
    pub sigma_bps: Option<f64>,

    /// Optional: regime at time of fill
    pub regime: Option<u8>,
}

impl Default for FillInfo {
    fn default() -> Self {
        Self {
            fill_id: 0,
            timestamp_ms: 0,
            fill_mid: 0.0,
            size: 1.0,
            is_buy: true,
            sigma_bps: None,
            regime: None,
        }
    }
}

// ============================================================================
// Pending Measurement
// ============================================================================

/// A pending AS measurement waiting for price observations
#[derive(Debug, Clone)]
struct PendingMeasurement {
    #[allow(dead_code)]
    fill_id: u64,
    fill_time_ms: u64,
    fill_mid: f64,
    #[allow(dead_code)]
    size: f64,
    is_buy: bool,
    #[allow(dead_code)]
    sigma_bps: Option<f64>,
    horizons_measured: [bool; 4],
}

// ============================================================================
// Rolling Statistics
// ============================================================================

/// EWMA-based rolling statistics for AS estimation
#[derive(Debug, Clone, Default)]
struct RollingStats {
    /// EWMA of AS values
    mean: f64,

    /// EWMA of squared AS values (for variance)
    mean_sq: f64,

    /// Count of observations
    count: usize,

    /// Sum of weights (for proper normalization)
    weight_sum: f64,
}

impl RollingStats {
    fn new() -> Self {
        Self::default()
    }

    /// Update with new observation using EWMA
    fn update(&mut self, value: f64, alpha: f64) {
        if self.count == 0 {
            self.mean = value;
            self.mean_sq = value * value;
            self.weight_sum = 1.0;
        } else {
            self.mean = alpha * value + (1.0 - alpha) * self.mean;
            self.mean_sq = alpha * value * value + (1.0 - alpha) * self.mean_sq;
            self.weight_sum = alpha + (1.0 - alpha) * self.weight_sum;
        }
        self.count += 1;
    }

    /// Get current mean
    fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current variance
    fn variance(&self) -> f64 {
        (self.mean_sq - self.mean * self.mean).max(0.0)
    }

    /// Get standard error
    fn std_error(&self) -> f64 {
        if self.count < 2 {
            f64::INFINITY
        } else {
            (self.variance() / self.count as f64).sqrt()
        }
    }

    /// T-statistic for mean != 0
    fn t_stat(&self) -> f64 {
        let se = self.std_error();
        if se > 0.0 && se.is_finite() {
            self.mean / se
        } else {
            0.0
        }
    }
}

// ============================================================================
// AS Decomposition Result
// ============================================================================

/// Result of AS decomposition
#[derive(Debug, Clone, Default)]
pub struct ASDecompResult {
    /// Total adverse selection (immediate horizon)
    pub total_bps: f64,

    /// Permanent component (information)
    pub permanent_bps: f64,

    /// Temporary component (liquidity)
    pub temporary_bps: f64,

    /// Timing component
    pub timing_bps: f64,

    /// Percentage that is permanent
    pub permanent_pct: f64,

    /// Percentage that is temporary
    pub temporary_pct: f64,

    /// Statistical significance of decomposition
    pub significance: f64,

    /// Number of fills measured
    pub n_fills: usize,

    /// AS by horizon (1s, 5s, 30s, 5min)
    pub by_horizon_bps: [f64; 4],

    /// Variance by horizon
    pub by_horizon_var: [f64; 4],
}

impl ASDecompResult {
    /// Check if results are statistically significant
    pub fn is_significant(&self) -> bool {
        self.significance > 0.5 && self.n_fills >= 30
    }

    /// Fraction of AS that is informational (permanent)
    pub fn information_fraction(&self) -> f64 {
        if self.total_bps.abs() > 0.01 {
            (self.permanent_bps / self.total_bps).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

// ============================================================================
// AS Decomposition Estimator
// ============================================================================

/// Multi-horizon adverse selection decomposition estimator
#[derive(Debug)]
pub struct ASDecomposition {
    /// Configuration
    config: ASDecompConfig,

    /// Pending fills awaiting measurement
    pending: VecDeque<PendingMeasurement>,

    /// Statistics by horizon
    stats_by_horizon: [RollingStats; 4],

    /// Statistics by horizon and side (buy/sell)
    stats_buy: [RollingStats; 4],
    stats_sell: [RollingStats; 4],

    /// Total fills measured
    fills_measured: usize,

    /// EWMA alpha (computed from half-life)
    ewma_alpha: f64,

    /// Last price observation
    last_mid: f64,
    last_timestamp_ms: u64,
}

impl ASDecomposition {
    /// Create new AS decomposition estimator
    pub fn new(config: ASDecompConfig) -> Self {
        let ewma_alpha = 1.0 - 0.5f64.powf(1.0 / config.ewma_half_life_fills as f64);

        Self {
            config,
            pending: VecDeque::with_capacity(1000),
            stats_by_horizon: [
                RollingStats::new(),
                RollingStats::new(),
                RollingStats::new(),
                RollingStats::new(),
            ],
            stats_buy: [
                RollingStats::new(),
                RollingStats::new(),
                RollingStats::new(),
                RollingStats::new(),
            ],
            stats_sell: [
                RollingStats::new(),
                RollingStats::new(),
                RollingStats::new(),
                RollingStats::new(),
            ],
            fills_measured: 0,
            ewma_alpha,
            last_mid: 0.0,
            last_timestamp_ms: 0,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ASDecompConfig::default())
    }

    /// Record a new fill for AS measurement
    pub fn on_fill(&mut self, fill: &FillInfo) {
        // Enforce max pending limit
        while self.pending.len() >= self.config.max_pending {
            self.pending.pop_front();
        }

        // Add pending measurement
        self.pending.push_back(PendingMeasurement {
            fill_id: fill.fill_id,
            fill_time_ms: fill.timestamp_ms,
            fill_mid: fill.fill_mid,
            size: fill.size,
            is_buy: fill.is_buy,
            sigma_bps: fill.sigma_bps,
            horizons_measured: [false; 4],
        });
    }

    /// Process price update and measure AS for mature fills
    pub fn on_price_update(&mut self, mid: f64, timestamp_ms: u64) {
        self.last_mid = mid;
        self.last_timestamp_ms = timestamp_ms;

        // Check each pending fill for mature measurements
        let horizons = &self.config.horizons_ms;

        for pending in self.pending.iter_mut() {
            let elapsed_ms = timestamp_ms.saturating_sub(pending.fill_time_ms);

            for (h_idx, &horizon_ms) in horizons.iter().enumerate() {
                if h_idx >= 4 {
                    break;
                }

                // Check if this horizon is ready to measure
                if !pending.horizons_measured[h_idx] && elapsed_ms >= horizon_ms {
                    // Calculate AS: price move in direction of fill
                    let price_change_bps = if pending.fill_mid > 0.0 {
                        (mid - pending.fill_mid) / pending.fill_mid * 10_000.0
                    } else {
                        0.0
                    };

                    // AS is positive when price moves against market maker
                    // If we sold (is_buy = true, buyer took our ask), AS = price increase
                    // If we bought (is_buy = false, seller hit our bid), AS = price decrease
                    let as_bps = if pending.is_buy {
                        price_change_bps // Price went up after we sold
                    } else {
                        -price_change_bps // Price went down after we bought
                    };

                    // Update statistics
                    self.stats_by_horizon[h_idx].update(as_bps, self.ewma_alpha);

                    if pending.is_buy {
                        self.stats_buy[h_idx].update(as_bps, self.ewma_alpha);
                    } else {
                        self.stats_sell[h_idx].update(as_bps, self.ewma_alpha);
                    }

                    pending.horizons_measured[h_idx] = true;

                    // Count fill as measured once all horizons done
                    if h_idx == horizons.len() - 1 || h_idx == 3 {
                        self.fills_measured += 1;
                    }
                }
            }
        }

        // Remove fully measured fills (all horizons complete)
        self.pending.retain(|p| {
            !p.horizons_measured
                .iter()
                .take(self.config.horizons_ms.len().min(4))
                .all(|&m| m)
        });
    }

    // ========================================================================
    // Decomposition Methods
    // ========================================================================

    /// Get AS at specific horizon
    pub fn as_at_horizon(&self, horizon_idx: usize) -> f64 {
        if horizon_idx < 4 {
            self.stats_by_horizon[horizon_idx].mean()
        } else {
            0.0
        }
    }

    /// Get AS variance at specific horizon
    pub fn as_variance_at_horizon(&self, horizon_idx: usize) -> f64 {
        if horizon_idx < 4 {
            self.stats_by_horizon[horizon_idx].variance()
        } else {
            0.0
        }
    }

    /// Total AS (immediate horizon, typically 1s)
    pub fn total_as_bps(&self) -> f64 {
        self.as_at_horizon(self.config.immediate_horizon_idx)
    }

    /// Permanent AS (long horizon, typically 5min)
    pub fn permanent_as_bps(&self) -> f64 {
        self.as_at_horizon(self.config.permanent_horizon_idx)
    }

    /// Temporary AS (immediate - permanent)
    pub fn temporary_as_bps(&self) -> f64 {
        let immediate = self.total_as_bps();
        let permanent = self.permanent_as_bps();
        (immediate - permanent).max(0.0)
    }

    /// Timing AS (residual, typically small or zero)
    pub fn timing_as_bps(&self) -> f64 {
        // Timing component is typically estimated from execution delay
        // Here we use a simple approximation based on AS volatility
        let var = self.as_variance_at_horizon(0);
        var.sqrt() * 0.1 // ~10% of immediate AS std
    }

    /// Get full decomposition result
    pub fn decomposition(&self) -> ASDecompResult {
        let total = self.total_as_bps();
        let permanent = self.permanent_as_bps();
        let temporary = self.temporary_as_bps();
        let timing = self.timing_as_bps();

        let total_abs = total.abs().max(0.01);
        let permanent_pct = (permanent / total_abs * 100.0).clamp(0.0, 100.0);
        let temporary_pct = (temporary / total_abs * 100.0).clamp(0.0, 100.0);

        // Compute significance from t-stats
        let t_permanent = self.stats_by_horizon[self.config.permanent_horizon_idx].t_stat();
        let significance = (t_permanent.abs() / 2.0).tanh(); // Sigmoid-like transform

        ASDecompResult {
            total_bps: total,
            permanent_bps: permanent,
            temporary_bps: temporary,
            timing_bps: timing,
            permanent_pct,
            temporary_pct,
            significance,
            n_fills: self.fills_measured,
            by_horizon_bps: [
                self.as_at_horizon(0),
                self.as_at_horizon(1),
                self.as_at_horizon(2),
                self.as_at_horizon(3),
            ],
            by_horizon_var: [
                self.as_variance_at_horizon(0),
                self.as_variance_at_horizon(1),
                self.as_variance_at_horizon(2),
                self.as_variance_at_horizon(3),
            ],
        }
    }

    // ========================================================================
    // Side-Specific Methods
    // ========================================================================

    /// AS for buy fills (we sold to buyer)
    pub fn as_buy_bps(&self, horizon_idx: usize) -> f64 {
        if horizon_idx < 4 {
            self.stats_buy[horizon_idx].mean()
        } else {
            0.0
        }
    }

    /// AS for sell fills (we bought from seller)
    pub fn as_sell_bps(&self, horizon_idx: usize) -> f64 {
        if horizon_idx < 4 {
            self.stats_sell[horizon_idx].mean()
        } else {
            0.0
        }
    }

    /// Asymmetry: difference between buy and sell AS
    pub fn as_asymmetry_bps(&self, horizon_idx: usize) -> f64 {
        self.as_buy_bps(horizon_idx) - self.as_sell_bps(horizon_idx)
    }

    // ========================================================================
    // Status Methods
    // ========================================================================

    /// Number of fills measured
    pub fn fills_measured(&self) -> usize {
        self.fills_measured
    }

    /// Number of pending measurements
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if estimator is warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.fills_measured >= self.config.min_fills
    }

    /// Reset estimator
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone());
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_estimator() -> ASDecomposition {
        ASDecomposition::default_config()
    }

    #[test]
    fn test_initialization() {
        let est = create_estimator();
        assert_eq!(est.fills_measured(), 0);
        assert!(!est.is_warmed_up());
        assert_eq!(est.pending_count(), 0);
    }

    #[test]
    fn test_fill_recording() {
        let mut est = create_estimator();

        let fill = FillInfo {
            fill_id: 1,
            timestamp_ms: 1000,
            fill_mid: 100.0,
            size: 1.0,
            is_buy: true,
            ..Default::default()
        };

        est.on_fill(&fill);
        assert_eq!(est.pending_count(), 1);
    }

    #[test]
    fn test_as_measurement_buy_fill() {
        let mut est = create_estimator();

        // Record a buy fill at $100
        let fill = FillInfo {
            fill_id: 1,
            timestamp_ms: 0,
            fill_mid: 100.0,
            is_buy: true,
            ..Default::default()
        };
        est.on_fill(&fill);

        // Price moves up to $101 after 1 second (100 bps adverse for seller)
        est.on_price_update(101.0, 1000);

        // AS should be positive (we sold before price went up)
        assert!(
            est.as_at_horizon(0) > 0.0,
            "AS should be positive: got {}",
            est.as_at_horizon(0)
        );
    }

    #[test]
    fn test_as_measurement_sell_fill() {
        let mut est = create_estimator();

        // Record a sell fill at $100 (we bought)
        let fill = FillInfo {
            fill_id: 1,
            timestamp_ms: 0,
            fill_mid: 100.0,
            is_buy: false,
            ..Default::default()
        };
        est.on_fill(&fill);

        // Price moves down to $99 after 1 second (100 bps adverse for buyer)
        est.on_price_update(99.0, 1000);

        // AS should be positive (we bought before price went down)
        assert!(
            est.as_at_horizon(0) > 0.0,
            "AS should be positive: got {}",
            est.as_at_horizon(0)
        );
    }

    #[test]
    fn test_multi_horizon_measurement() {
        let mut est = create_estimator();

        // Fill at time 0, price 100
        let fill = FillInfo {
            fill_id: 1,
            timestamp_ms: 0,
            fill_mid: 100.0,
            is_buy: true,
            ..Default::default()
        };
        est.on_fill(&fill);

        // Updates at each horizon
        est.on_price_update(101.0, 1000); // 1s: +100 bps
        est.on_price_update(101.5, 5000); // 5s: +150 bps
        est.on_price_update(102.0, 30000); // 30s: +200 bps
        est.on_price_update(102.5, 300000); // 5min: +250 bps

        // Check all horizons measured
        assert!(est.as_at_horizon(0) > 0.0, "1s should be measured");
        assert!(est.as_at_horizon(1) > 0.0, "5s should be measured");
        assert!(est.as_at_horizon(2) > 0.0, "30s should be measured");
        assert!(est.as_at_horizon(3) > 0.0, "5min should be measured");
    }

    #[test]
    fn test_decomposition() {
        let mut est = create_estimator();

        // Add multiple fills with varying outcomes
        for i in 0..50 {
            let fill = FillInfo {
                fill_id: i,
                timestamp_ms: i * 400_000, // Spread out in time
                fill_mid: 100.0,
                is_buy: true,
                ..Default::default()
            };
            est.on_fill(&fill);

            // Immediate impact high, permanent lower (mean reversion)
            let t0 = i * 400_000;
            est.on_price_update(103.0, t0 + 1000); // 3% immediate
            est.on_price_update(102.0, t0 + 5000); // 2% at 5s
            est.on_price_update(101.0, t0 + 30000); // 1% at 30s
            est.on_price_update(100.5, t0 + 300000); // 0.5% permanent
        }

        let decomp = est.decomposition();

        // Temporary should be larger than permanent (mean reversion)
        assert!(
            decomp.temporary_bps > 0.0,
            "Should have temporary component"
        );
        assert!(decomp.n_fills > 0, "Should have measured fills");
    }

    #[test]
    fn test_warmup() {
        let mut est = create_estimator();

        for i in 0..35 {
            let fill = FillInfo {
                fill_id: i,
                timestamp_ms: i * 400_000,
                fill_mid: 100.0,
                is_buy: true,
                ..Default::default()
            };
            est.on_fill(&fill);

            let t0 = i * 400_000;
            est.on_price_update(101.0, t0 + 1000);
            est.on_price_update(101.0, t0 + 5000);
            est.on_price_update(101.0, t0 + 30000);
            est.on_price_update(101.0, t0 + 300000);
        }

        assert!(est.is_warmed_up());
    }

    #[test]
    fn test_reset() {
        let mut est = create_estimator();

        let fill = FillInfo {
            fill_id: 1,
            timestamp_ms: 0,
            fill_mid: 100.0,
            is_buy: true,
            ..Default::default()
        };
        est.on_fill(&fill);
        est.on_price_update(101.0, 1000);

        assert!(est.pending_count() > 0 || est.fills_measured() > 0);

        est.reset();

        assert_eq!(est.fills_measured(), 0);
        assert_eq!(est.pending_count(), 0);
    }

    #[test]
    fn test_asymmetry() {
        let mut est = create_estimator();

        // Buy fills with high AS
        for i in 0..20 {
            let fill = FillInfo {
                fill_id: i,
                timestamp_ms: i * 400_000,
                fill_mid: 100.0,
                is_buy: true,
                ..Default::default()
            };
            est.on_fill(&fill);
            let t0 = i * 400_000;
            est.on_price_update(102.0, t0 + 1000);
            est.on_price_update(102.0, t0 + 300000);
        }

        // Sell fills with low AS
        for i in 20..40 {
            let fill = FillInfo {
                fill_id: i,
                timestamp_ms: i * 400_000,
                fill_mid: 100.0,
                is_buy: false,
                ..Default::default()
            };
            est.on_fill(&fill);
            let t0 = i * 400_000;
            est.on_price_update(99.5, t0 + 1000);
            est.on_price_update(99.5, t0 + 300000);
        }

        // Check asymmetry
        let buy_as = est.as_buy_bps(0);
        let sell_as = est.as_sell_bps(0);

        assert!(buy_as > sell_as, "Buy AS should exceed sell AS");
    }
}
