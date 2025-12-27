//! Adverse selection measurement and prediction for market making.
//!
//! Measures the ground truth adverse selection cost (E[Δp | fill]) and provides
//! spread adjustment recommendations to compensate for informed flow.
//!
//! Key components:
//! - Ground truth measurement: Track price movement 1 second after each fill
//! - Realized AS tracking: EWMA of signed price movements conditional on fills
//! - Predicted α(t): Probability that next trade is informed (from signals)
//! - Spread adjustment: Recommended spread widening based on realized AS

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::debug;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for adverse selection estimation.
#[derive(Debug, Clone)]
pub struct AdverseSelectionConfig {
    /// Horizon for measuring price impact after fill (milliseconds)
    /// Default: 1000ms (1 second)
    pub measurement_horizon_ms: u64,

    /// EWMA decay factor for realized AS (0.0 = no decay, 1.0 = instant decay)
    /// Default: 0.05 (20-trade half-life)
    pub ewma_alpha: f64,

    /// Minimum fills required before AS estimates are valid
    /// Default: 20
    pub min_fills_warmup: usize,

    /// Maximum pending fills to track (memory bound)
    /// Default: 1000
    pub max_pending_fills: usize,

    /// Spread adjustment multiplier (how much to widen based on AS)
    /// spread_adjustment = multiplier × realized_as
    /// Default: 2.0 (widen by 2x the AS cost)
    pub spread_adjustment_multiplier: f64,

    /// Maximum spread adjustment (as fraction of mid price)
    /// Default: 0.005 (0.5%)
    pub max_spread_adjustment: f64,

    /// Minimum spread adjustment (floor)
    /// Default: 0.0 (no floor)
    pub min_spread_adjustment: f64,

    // === Alpha Prediction Weights ===
    /// Weight for volatility surprise signal in α prediction
    pub alpha_volatility_weight: f64,
    /// Weight for flow imbalance magnitude signal in α prediction
    pub alpha_flow_weight: f64,
    /// Weight for jump ratio signal in α prediction
    pub alpha_jump_weight: f64,
}

impl Default for AdverseSelectionConfig {
    fn default() -> Self {
        Self {
            measurement_horizon_ms: 1000,
            ewma_alpha: 0.05,
            min_fills_warmup: 20,
            max_pending_fills: 1000,
            spread_adjustment_multiplier: 2.0,
            max_spread_adjustment: 0.005,
            min_spread_adjustment: 0.0,
            // Alpha prediction weights (simple linear combination)
            alpha_volatility_weight: 0.3,
            alpha_flow_weight: 0.4,
            alpha_jump_weight: 0.3,
        }
    }
}

// ============================================================================
// Pending Fill Tracking
// ============================================================================

/// A fill waiting for price resolution to measure adverse selection.
#[derive(Debug, Clone)]
struct PendingFill {
    /// Trade ID for deduplication
    tid: u64,
    /// Fill timestamp
    fill_time: Instant,
    /// Mid price at time of fill
    fill_mid: f64,
    /// Fill size (for future size-weighted AS)
    #[allow(dead_code)]
    size: f64,
    /// True if this was a buy fill (we got lifted on our ask)
    is_buy: bool,
}

// ============================================================================
// Adverse Selection Estimator
// ============================================================================

/// Estimates adverse selection costs from fill data.
///
/// Measures ground truth E[Δp | fill] by tracking price movement after fills,
/// and provides spread adjustment recommendations.
#[derive(Debug)]
pub struct AdverseSelectionEstimator {
    config: AdverseSelectionConfig,

    /// Fills waiting for price resolution
    pending_fills: VecDeque<PendingFill>,

    /// Total fills measured (after resolution)
    fills_measured: usize,

    // === Realized AS Tracking (EWMA) ===
    /// EWMA of adverse selection for buy fills (price went up after we got lifted)
    /// Positive = adverse, we bought and price moved against us
    realized_as_buy: f64,

    /// EWMA of adverse selection for sell fills (price went down after we got hit)
    /// Positive = adverse, we sold and price moved against us
    realized_as_sell: f64,

    /// EWMA of overall adverse selection (magnitude, both sides)
    realized_as_total: f64,

    /// Count of buy fills measured
    buy_fills_measured: usize,

    /// Count of sell fills measured
    sell_fills_measured: usize,

    // === Signal Cache for Alpha Prediction ===
    /// Latest volatility surprise (realized_vol / expected_vol - 1.0)
    cached_vol_surprise: f64,

    /// Latest flow imbalance magnitude |flow_imbalance|
    cached_flow_magnitude: f64,

    /// Latest jump ratio (RV/BV)
    cached_jump_ratio: f64,
}

impl AdverseSelectionEstimator {
    /// Create a new adverse selection estimator.
    pub fn new(config: AdverseSelectionConfig) -> Self {
        Self {
            config,
            pending_fills: VecDeque::new(),
            fills_measured: 0,
            realized_as_buy: 0.0,
            realized_as_sell: 0.0,
            realized_as_total: 0.0,
            buy_fills_measured: 0,
            sell_fills_measured: 0,
            cached_vol_surprise: 0.0,
            cached_flow_magnitude: 0.0,
            cached_jump_ratio: 1.0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(AdverseSelectionConfig::default())
    }

    /// Record a fill for future AS measurement.
    ///
    /// Call this immediately when a fill is received, with the current mid price.
    pub fn record_fill(&mut self, tid: u64, size: f64, is_buy: bool, current_mid: f64) {
        // Enforce max pending fills (FIFO eviction)
        while self.pending_fills.len() >= self.config.max_pending_fills {
            self.pending_fills.pop_front();
        }

        self.pending_fills.push_back(PendingFill {
            tid,
            fill_time: Instant::now(),
            fill_mid: current_mid,
            size,
            is_buy,
        });

        debug!(
            tid = tid,
            size = size,
            is_buy = is_buy,
            mid = current_mid,
            pending = self.pending_fills.len(),
            "AS: Recorded fill for measurement"
        );
    }

    /// Update estimator with current mid price.
    ///
    /// Call this on each mid price update to resolve pending fills
    /// whose measurement horizon has elapsed.
    pub fn update(&mut self, current_mid: f64) {
        let now = Instant::now();
        let horizon = Duration::from_millis(self.config.measurement_horizon_ms);

        // Process all pending fills whose horizon has elapsed
        while let Some(front) = self.pending_fills.front() {
            if now.duration_since(front.fill_time) < horizon {
                break; // Not ready yet
            }

            let fill = self.pending_fills.pop_front().unwrap();
            self.measure_fill_impact(&fill, current_mid);
        }
    }

    /// Measure the adverse selection impact of a resolved fill.
    fn measure_fill_impact(&mut self, fill: &PendingFill, current_mid: f64) {
        // Calculate price change (as fraction of fill price)
        let price_change = (current_mid - fill.fill_mid) / fill.fill_mid;

        // Adverse selection from the MM's perspective:
        // - If we got LIFTED (sold to aggressor = is_buy=true for our fill):
        //   Price going UP after = adverse (we sold too cheap)
        // - If we got HIT (bought from aggressor = is_buy=false for our fill):
        //   Price going DOWN after = adverse (we bought too expensive)
        //
        // Note: is_buy from the fill perspective means WE bought (got filled on our bid)
        // So for AS: if is_buy=true and price goes DOWN, that's adverse for us
        //           if is_buy=false and price goes UP, that's adverse for us

        let signed_as = if fill.is_buy {
            // We bought (bid got hit), price going DOWN is adverse for us
            -price_change // Positive AS if price dropped
        } else {
            // We sold (ask got lifted), price going UP is adverse for us
            price_change // Positive AS if price rose
        };

        // Update EWMA
        let alpha = self.config.ewma_alpha;

        if fill.is_buy {
            self.realized_as_buy = alpha * signed_as + (1.0 - alpha) * self.realized_as_buy;
            self.buy_fills_measured += 1;
        } else {
            self.realized_as_sell = alpha * signed_as + (1.0 - alpha) * self.realized_as_sell;
            self.sell_fills_measured += 1;
        }

        // Total AS (magnitude)
        self.realized_as_total =
            alpha * signed_as.abs() + (1.0 - alpha) * self.realized_as_total;
        self.fills_measured += 1;

        debug!(
            tid = fill.tid,
            is_buy = fill.is_buy,
            fill_mid = fill.fill_mid,
            current_mid = current_mid,
            price_change_bps = price_change * 10000.0,
            signed_as_bps = signed_as * 10000.0,
            realized_as_total_bps = self.realized_as_total * 10000.0,
            fills = self.fills_measured,
            "AS: Measured fill impact"
        );
    }

    /// Update signal cache for alpha prediction.
    ///
    /// Call this with latest market parameters to update prediction signals.
    pub fn update_signals(
        &mut self,
        sigma_realized: f64,
        sigma_expected: f64,
        flow_imbalance: f64,
        jump_ratio: f64,
    ) {
        // Volatility surprise: how much higher is realized vol vs expected
        self.cached_vol_surprise = if sigma_expected > 1e-10 {
            (sigma_realized / sigma_expected) - 1.0
        } else {
            0.0
        };

        // Flow imbalance magnitude
        self.cached_flow_magnitude = flow_imbalance.abs();

        // Jump ratio (RV/BV)
        self.cached_jump_ratio = jump_ratio;
    }

    // ========================================================================
    // Public Query Methods
    // ========================================================================

    /// Check if estimator has enough data for valid estimates.
    pub fn is_warmed_up(&self) -> bool {
        self.fills_measured >= self.config.min_fills_warmup
    }

    /// Get the number of fills measured.
    pub fn fills_measured(&self) -> usize {
        self.fills_measured
    }

    /// Get the number of pending fills awaiting resolution.
    pub fn pending_count(&self) -> usize {
        self.pending_fills.len()
    }

    /// Get realized adverse selection for buy fills (EWMA).
    /// Positive = adverse (price moved against us after buying).
    pub fn realized_as_buy(&self) -> f64 {
        self.realized_as_buy
    }

    /// Get realized adverse selection for sell fills (EWMA).
    /// Positive = adverse (price moved against us after selling).
    pub fn realized_as_sell(&self) -> f64 {
        self.realized_as_sell
    }

    /// Get total realized adverse selection (magnitude, both sides).
    pub fn realized_as_total(&self) -> f64 {
        self.realized_as_total
    }

    /// Get realized AS in basis points.
    pub fn realized_as_bps(&self) -> f64 {
        self.realized_as_total * 10000.0
    }

    /// Get recommended spread adjustment (as fraction of mid price).
    ///
    /// This is the amount to widen spreads to compensate for adverse selection.
    /// Returns 0.0 if not warmed up.
    pub fn spread_adjustment(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }

        let raw_adjustment =
            self.realized_as_total * self.config.spread_adjustment_multiplier;

        // Clamp to configured bounds
        raw_adjustment
            .max(self.config.min_spread_adjustment)
            .min(self.config.max_spread_adjustment)
    }

    /// Get spread adjustment in basis points.
    pub fn spread_adjustment_bps(&self) -> f64 {
        self.spread_adjustment() * 10000.0
    }

    /// Get predicted alpha: P(next trade is informed).
    ///
    /// Uses a simple weighted combination of signals:
    /// - Volatility surprise (higher realized vs expected = more informed)
    /// - Flow imbalance magnitude (extreme imbalance = directional informed)
    /// - Jump ratio (high RV/BV = toxic informed flow)
    ///
    /// Returns value in [0, 1].
    pub fn predicted_alpha(&self) -> f64 {
        let cfg = &self.config;

        // Normalize signals to [0, 1] range
        let vol_signal = (self.cached_vol_surprise.max(0.0) / 2.0).min(1.0);
        let flow_signal = self.cached_flow_magnitude; // Already in [0, 1]
        let jump_signal = ((self.cached_jump_ratio - 1.0).max(0.0) / 4.0).min(1.0);

        // Weighted combination
        let raw_alpha = cfg.alpha_volatility_weight * vol_signal
            + cfg.alpha_flow_weight * flow_signal
            + cfg.alpha_jump_weight * jump_signal;

        // Sigmoid squashing to [0, 1]
        1.0 / (1.0 + (-4.0 * (raw_alpha - 0.5)).exp())
    }

    /// Get asymmetric spread adjustment for bid/ask.
    ///
    /// Returns (bid_adjustment, ask_adjustment) where each is the amount
    /// to widen that side based on side-specific AS.
    pub fn asymmetric_adjustment(&self) -> (f64, f64) {
        if !self.is_warmed_up() {
            return (0.0, 0.0);
        }

        let mult = self.config.spread_adjustment_multiplier;
        let max_adj = self.config.max_spread_adjustment;

        // If we're seeing more AS on buys, widen bids more
        let bid_adj = (self.realized_as_buy.abs() * mult).min(max_adj);
        // If we're seeing more AS on sells, widen asks more
        let ask_adj = (self.realized_as_sell.abs() * mult).min(max_adj);

        (bid_adj, ask_adj)
    }

    /// Get diagnostic summary for logging.
    pub fn summary(&self) -> AdverseSelectionSummary {
        AdverseSelectionSummary {
            fills_measured: self.fills_measured,
            pending_fills: self.pending_fills.len(),
            is_warmed_up: self.is_warmed_up(),
            realized_as_bps: self.realized_as_bps(),
            realized_as_buy_bps: self.realized_as_buy * 10000.0,
            realized_as_sell_bps: self.realized_as_sell * 10000.0,
            spread_adjustment_bps: self.spread_adjustment_bps(),
            predicted_alpha: self.predicted_alpha(),
        }
    }
}

/// Summary of adverse selection state for logging.
#[derive(Debug, Clone)]
pub struct AdverseSelectionSummary {
    pub fills_measured: usize,
    pub pending_fills: usize,
    pub is_warmed_up: bool,
    pub realized_as_bps: f64,
    pub realized_as_buy_bps: f64,
    pub realized_as_sell_bps: f64,
    pub spread_adjustment_bps: f64,
    pub predicted_alpha: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_estimator() -> AdverseSelectionEstimator {
        let config = AdverseSelectionConfig {
            measurement_horizon_ms: 10, // 10ms for fast tests
            ewma_alpha: 0.5,            // High alpha for visible changes
            min_fills_warmup: 3,
            max_pending_fills: 100,
            spread_adjustment_multiplier: 2.0,
            max_spread_adjustment: 0.01,
            min_spread_adjustment: 0.0,
            ..Default::default()
        };
        AdverseSelectionEstimator::new(config)
    }

    #[test]
    fn test_record_fill() {
        let mut est = make_estimator();

        est.record_fill(1, 1.0, true, 100.0);
        assert_eq!(est.pending_count(), 1);
        assert_eq!(est.fills_measured(), 0);
        assert!(!est.is_warmed_up());
    }

    #[test]
    fn test_adverse_selection_buy() {
        let mut est = make_estimator();

        // Record a buy fill at mid=100
        est.record_fill(1, 1.0, true, 100.0);

        // Wait for horizon
        std::thread::sleep(std::time::Duration::from_millis(15));

        // Price dropped to 99 (1% adverse for our buy)
        est.update(99.0);

        assert_eq!(est.fills_measured(), 1);
        // We bought at 100, price went to 99 = -1% change
        // For buy, AS = -price_change = 0.01 (positive = adverse)
        assert!(est.realized_as_buy > 0.0);
    }

    #[test]
    fn test_adverse_selection_sell() {
        let mut est = make_estimator();

        // Record a sell fill at mid=100
        est.record_fill(1, 1.0, false, 100.0);

        // Wait for horizon
        std::thread::sleep(std::time::Duration::from_millis(15));

        // Price rose to 101 (1% adverse for our sell)
        est.update(101.0);

        assert_eq!(est.fills_measured(), 1);
        // We sold at 100, price went to 101 = +1% change
        // For sell, AS = price_change = 0.01 (positive = adverse)
        assert!(est.realized_as_sell > 0.0);
    }

    #[test]
    fn test_warmup() {
        let mut est = make_estimator();

        for i in 0..3 {
            est.record_fill(i as u64, 1.0, true, 100.0);
            std::thread::sleep(std::time::Duration::from_millis(15));
            est.update(99.0); // Adverse for buys
        }

        assert!(est.is_warmed_up());
        assert!(est.spread_adjustment() > 0.0);
    }

    #[test]
    fn test_spread_adjustment_clamping() {
        let mut est = make_estimator();

        // Create extreme AS
        for i in 0..5 {
            est.record_fill(i as u64, 1.0, true, 100.0);
            std::thread::sleep(std::time::Duration::from_millis(15));
            est.update(90.0); // 10% adverse
        }

        // Should be clamped to max
        assert!(est.spread_adjustment() <= est.config.max_spread_adjustment);
    }

    #[test]
    fn test_predicted_alpha() {
        let mut est = make_estimator();

        // Low signals = low alpha
        est.update_signals(0.0001, 0.0001, 0.0, 1.0);
        let low_alpha = est.predicted_alpha();

        // High signals = high alpha
        est.update_signals(0.0003, 0.0001, 0.8, 4.0);
        let high_alpha = est.predicted_alpha();

        assert!(high_alpha > low_alpha);
        assert!(low_alpha >= 0.0 && low_alpha <= 1.0);
        assert!(high_alpha >= 0.0 && high_alpha <= 1.0);
    }

    #[test]
    fn test_max_pending_fills() {
        let config = AdverseSelectionConfig {
            max_pending_fills: 5,
            ..Default::default()
        };
        let mut est = AdverseSelectionEstimator::new(config);

        // Add more fills than max
        for i in 0..10 {
            est.record_fill(i as u64, 1.0, true, 100.0);
        }

        // Should be capped at max
        assert_eq!(est.pending_count(), 5);
    }
}
