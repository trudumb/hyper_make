//! Adverse selection estimator.
//!
//! Measures ground truth E[Δp | fill] by tracking price movement after fills,
//! and provides spread adjustment recommendations.

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::debug;

use super::AdverseSelectionConfig;

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
    /// Horizons already measured (to avoid double-counting)
    horizons_measured: [bool; NUM_HORIZONS],
}

/// Number of AS measurement horizons
const NUM_HORIZONS: usize = 6;

/// Multi-horizon AS measurement horizons (milliseconds)
///
/// Extended from [500, 1000, 2000] to cover actual hold times (80s-10min observed).
/// The longer horizons (60s-10min) match measured fill→close durations.
const HORIZONS_MS: [u64; NUM_HORIZONS] = [
    500,     // 0.5s - microstructure noise
    2_000,   // 2s - immediate adverse selection
    10_000,  // 10s - short-term price discovery
    60_000,  // 60s (1 min) - matches observed hold times
    180_000, // 180s (3 min) - medium-term adverse selection
    600_000, // 600s (10 min) - longest typical hold time
];

/// Tracks AS statistics for a single horizon
#[derive(Debug, Clone, Default)]
struct HorizonStats {
    /// EWMA of AS magnitude
    as_ewma: f64,
    /// EWMA of AS variance (for stability scoring)
    as_var_ewma: f64,
    /// Count of measurements
    count: usize,
}

/// Estimates adverse selection costs from fill data.
///
/// Measures ground truth E[Δp | fill] by tracking price movement after fills,
/// and provides spread adjustment recommendations.
///
/// Features multi-horizon measurement (500ms, 1000ms, 2000ms) with automatic
/// horizon selection based on estimate stability.
#[derive(Debug)]
pub struct AdverseSelectionEstimator {
    config: AdverseSelectionConfig,

    /// Fills waiting for price resolution
    pending_fills: VecDeque<PendingFill>,

    /// Total fills measured (after resolution at primary horizon)
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

    // === Multi-Horizon AS Tracking ===
    /// Stats for each horizon (500ms to 10min)
    horizon_stats: [HorizonStats; NUM_HORIZONS],

    /// Index of currently selected best horizon (0 to NUM_HORIZONS-1)
    best_horizon_idx: usize,

    // === Signal Cache for Alpha Prediction ===
    /// Latest volatility surprise (realized_vol / expected_vol - 1.0)
    cached_vol_surprise: f64,

    /// Latest flow imbalance magnitude |flow_imbalance|
    cached_flow_magnitude: f64,

    /// Latest jump ratio (RV/BV)
    cached_jump_ratio: f64,

    // === Informed Fill Tracking for Parameter Learning ===
    /// Count of fills classified as "informed" (adverse move > threshold at 500ms)
    informed_fills_count: usize,

    /// Count of fills classified as "uninformed" (adverse move <= threshold at 500ms)
    uninformed_fills_count: usize,

    /// Threshold in bps for classifying a fill as "informed"
    informed_threshold_bps: f64,

    /// EWMA of recent adverse selection magnitude (in bps) for spread widening.
    /// Smoothed over ~10 fills (alpha ≈ 0.1) to react to AS regime changes.
    recent_as_ewma_bps: f64,
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
            horizon_stats: Default::default(),
            best_horizon_idx: 3, // Default to 60s (60000ms) - matches observed hold times
            cached_vol_surprise: 0.0,
            cached_flow_magnitude: 0.0,
            cached_jump_ratio: 1.0,
            informed_fills_count: 0,
            uninformed_fills_count: 0,
            informed_threshold_bps: 5.0, // Default: 5 bps = "informed"
            recent_as_ewma_bps: 0.0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(AdverseSelectionConfig::default())
    }

    /// Record a fill for future AS measurement via multi-horizon markout.
    ///
    /// `current_mid` should be the mid price at **order placement time** (mid_at_placement).
    /// The AS estimator measures how much the market moved after placement by comparing
    /// this anchor price against future mid prices at 500ms/1s/2s horizons.
    ///
    /// Previously this used mid_at_fill, which was tautological: both `fill_mid` and
    /// the first `update()` call used nearly identical mid prices, giving AS ≈ 0 always.
    /// Using mid_at_placement captures the full adverse selection cost including pre-fill
    /// movement that occurred while the order was resting.
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
            horizons_measured: [false; NUM_HORIZONS],
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
    /// at multiple horizons (500ms, 1000ms, 2000ms).
    pub fn update(&mut self, current_mid: f64) {
        let now = Instant::now();
        let alpha = self.config.ewma_alpha;
        let _best_horizon_idx = self.best_horizon_idx;

        // Collect measurements to process (avoid borrow issues)
        let mut measurements: Vec<(usize, f64, bool)> = Vec::new(); // (horizon_idx, signed_as, is_buy)
        let mut fills_to_remove = Vec::new();

        // First pass: measure at each horizon
        for (fill_idx, fill) in self.pending_fills.iter_mut().enumerate() {
            let elapsed = now.duration_since(fill.fill_time);

            for (horizon_idx, &horizon_ms) in HORIZONS_MS.iter().enumerate() {
                if fill.horizons_measured[horizon_idx] {
                    continue; // Already measured at this horizon
                }

                if elapsed >= Duration::from_millis(horizon_ms) {
                    // Measure AS at this horizon
                    let price_change = (current_mid - fill.fill_mid) / fill.fill_mid;
                    let signed_as = if fill.is_buy {
                        -price_change // We bought, price going DOWN is adverse
                    } else {
                        price_change // We sold, price going UP is adverse
                    };

                    fill.horizons_measured[horizon_idx] = true;

                    // Collect for later processing - use first horizon (500ms) for legacy stats
                    // to maintain backward compatibility with fills_measured counter
                    if horizon_idx == 0 {
                        measurements.push((fill.tid as usize, signed_as, fill.is_buy));
                    }

                    // Update horizon stats inline (these are separate from self borrow)
                    let stats = &mut self.horizon_stats[horizon_idx];
                    let prev_as = stats.as_ewma;
                    stats.as_ewma = alpha * signed_as.abs() + (1.0 - alpha) * stats.as_ewma;
                    let deviation = (signed_as.abs() - prev_as).powi(2);
                    stats.as_var_ewma = alpha * deviation + (1.0 - alpha) * stats.as_var_ewma;
                    stats.count += 1;
                }
            }

            // Check if all horizons measured
            if fill.horizons_measured.iter().all(|&m| m) {
                fills_to_remove.push(fill_idx);
            }
        }

        // Remove fully measured fills (in reverse order to preserve indices)
        for idx in fills_to_remove.into_iter().rev() {
            self.pending_fills.remove(idx);
        }

        // Second pass: update legacy stats from collected measurements
        for (_tid, signed_as, is_buy) in measurements {
            if is_buy {
                self.realized_as_buy = alpha * signed_as + (1.0 - alpha) * self.realized_as_buy;
                self.buy_fills_measured += 1;
            } else {
                self.realized_as_sell = alpha * signed_as + (1.0 - alpha) * self.realized_as_sell;
                self.sell_fills_measured += 1;
            }
            self.realized_as_total =
                alpha * signed_as.abs() + (1.0 - alpha) * self.realized_as_total;
            self.fills_measured += 1;

            // Classify fill as informed/uninformed for parameter learning
            // signed_as is in fractional form, threshold is in bps (1 bp = 0.0001)
            let adverse_move_bps = signed_as.abs() * 10000.0;
            if adverse_move_bps > self.informed_threshold_bps {
                self.informed_fills_count += 1;
            } else {
                self.uninformed_fills_count += 1;
            }

            // Update rolling AS severity EWMA (alpha=0.1, ~10 fill half-life)
            const AS_SEVERITY_ALPHA: f64 = 0.1;
            self.recent_as_ewma_bps = AS_SEVERITY_ALPHA * adverse_move_bps
                + (1.0 - AS_SEVERITY_ALPHA) * self.recent_as_ewma_bps;
        }

        // Periodically update best horizon selection
        self.update_best_horizon();
    }

    /// Update best horizon selection based on variance (stability).
    fn update_best_horizon(&mut self) {
        // Need at least 20 measurements per horizon before comparing
        let min_count = 20;
        if self.horizon_stats.iter().any(|s| s.count < min_count) {
            return;
        }

        // Find horizon with lowest variance (most stable)
        let best_idx = self
            .horizon_stats
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.as_var_ewma
                    .partial_cmp(&b.as_var_ewma)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(1);

        if best_idx != self.best_horizon_idx {
            debug!(
                prev_horizon_ms = HORIZONS_MS[self.best_horizon_idx],
                new_horizon_ms = HORIZONS_MS[best_idx],
                variance_prev = %format!("{:.6}", self.horizon_stats[self.best_horizon_idx].as_var_ewma),
                variance_new = %format!("{:.6}", self.horizon_stats[best_idx].as_var_ewma),
                "AS: Switching to more stable horizon"
            );
            self.best_horizon_idx = best_idx;
        }
    }

    /// Get the currently selected measurement horizon in milliseconds.
    pub fn current_horizon_ms(&self) -> u64 {
        HORIZONS_MS[self.best_horizon_idx]
    }

    /// Get AS estimates at all horizons (for diagnostics).
    pub fn horizon_as_bps(&self) -> [(u64, f64); NUM_HORIZONS] {
        [
            (HORIZONS_MS[0], self.horizon_stats[0].as_ewma * 10000.0),
            (HORIZONS_MS[1], self.horizon_stats[1].as_ewma * 10000.0),
            (HORIZONS_MS[2], self.horizon_stats[2].as_ewma * 10000.0),
            (HORIZONS_MS[3], self.horizon_stats[3].as_ewma * 10000.0),
            (HORIZONS_MS[4], self.horizon_stats[4].as_ewma * 10000.0),
            (HORIZONS_MS[5], self.horizon_stats[5].as_ewma * 10000.0),
        ]
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

        let raw_adjustment = self.realized_as_total * self.config.spread_adjustment_multiplier;

        // Clamp to configured bounds
        raw_adjustment
            .max(self.config.min_spread_adjustment)
            .min(self.config.max_spread_adjustment)
    }

    /// Get spread adjustment in basis points.
    pub fn spread_adjustment_bps(&self) -> f64 {
        self.spread_adjustment() * 10000.0
    }

    /// Per-side spread adjustment for bids (based on buy-side AS).
    /// When buy fills show high AS, widen bid side more.
    pub fn spread_adjustment_bid(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }
        let raw = self.realized_as_buy * self.config.spread_adjustment_multiplier;
        raw.max(self.config.min_spread_adjustment)
            .min(self.config.max_spread_adjustment)
    }

    /// Per-side spread adjustment for asks (based on sell-side AS).
    /// When sell fills show high AS, widen ask side more.
    pub fn spread_adjustment_ask(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }
        let raw = self.realized_as_sell * self.config.spread_adjustment_multiplier;
        raw.max(self.config.min_spread_adjustment)
            .min(self.config.max_spread_adjustment)
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

    // ========================================================================
    // Multi-Horizon AS Accessors
    // ========================================================================

    /// Get AS at 500ms horizon (in bps).
    pub fn as_500ms_bps(&self) -> f64 {
        self.horizon_stats[0].as_ewma * 10000.0
    }

    /// Get AS at 1000ms horizon (in bps).
    pub fn as_1000ms_bps(&self) -> f64 {
        self.horizon_stats[1].as_ewma * 10000.0
    }

    /// Get AS at 2000ms horizon (in bps).
    pub fn as_2000ms_bps(&self) -> f64 {
        self.horizon_stats[2].as_ewma * 10000.0
    }

    /// Get current best horizon in milliseconds.
    pub fn best_horizon_ms(&self) -> u64 {
        HORIZONS_MS[self.best_horizon_idx]
    }

    /// Get AS at the currently selected best horizon (in bps).
    pub fn best_horizon_as_bps(&self) -> f64 {
        self.horizon_stats[self.best_horizon_idx].as_ewma * 10000.0
    }


    // === Informed Fill Classification for Parameter Learning ===

    /// Get the count of fills classified as "informed" (adverse move > threshold).
    pub fn informed_fills_count(&self) -> usize {
        self.informed_fills_count
    }

    /// Get the count of fills classified as "uninformed" (adverse move <= threshold).
    pub fn uninformed_fills_count(&self) -> usize {
        self.uninformed_fills_count
    }

    /// Get the total fills classified (informed + uninformed).
    pub fn total_classified_fills(&self) -> usize {
        self.informed_fills_count + self.uninformed_fills_count
    }

    /// Get the empirical alpha_touch = P(informed | fill).
    /// Returns None if no fills have been classified yet.
    pub fn empirical_alpha_touch(&self) -> Option<f64> {
        let total = self.total_classified_fills();
        if total == 0 {
            None
        } else {
            Some(self.informed_fills_count as f64 / total as f64)
        }
    }

    /// Consume the informed/uninformed counts for parameter learning.
    /// Returns (informed_count, uninformed_count) and resets both to zero.
    /// This allows periodic batch updates to the Bayesian parameter.
    pub fn take_informed_counts(&mut self) -> (usize, usize) {
        let informed = self.informed_fills_count;
        let uninformed = self.uninformed_fills_count;
        self.informed_fills_count = 0;
        self.uninformed_fills_count = 0;
        (informed, uninformed)
    }

    /// Set the threshold (in bps) for classifying fills as "informed".
    /// Default is 5 bps.
    pub fn set_informed_threshold_bps(&mut self, threshold_bps: f64) {
        self.informed_threshold_bps = threshold_bps;
    }

    /// Get the current informed threshold in bps.
    pub fn informed_threshold_bps(&self) -> f64 {
        self.informed_threshold_bps
    }

    /// Spread multiplier based on recent AS severity (rolling EWMA over ~10 fills).
    ///
    /// Returns a multiplier to widen spreads when recent fills show persistent
    /// adverse selection:
    /// - > 5 bps AS: 1.5x (heavy AS, widen 50%)
    /// - > 3 bps AS: 1.25x (moderate AS, widen 25%)
    /// - otherwise: 1.0x (normal)
    pub fn recent_as_severity_mult(&self) -> f64 {
        if self.recent_as_ewma_bps > 5.0 {
            1.5
        } else if self.recent_as_ewma_bps > 3.0 {
            1.25
        } else {
            1.0
        }
    }

    /// Get the current rolling AS severity in bps (for diagnostics).
    pub fn recent_as_severity_bps(&self) -> f64 {
        self.recent_as_ewma_bps
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_estimator() -> AdverseSelectionEstimator {
        let config = AdverseSelectionConfig {
            measurement_horizon_ms: 500, // Match first multi-horizon (500ms)
            ewma_alpha: 0.5,             // High alpha for visible changes
            min_fills_warmup: 3,
            max_pending_fills: 100,
            spread_adjustment_multiplier: 2.0,
            max_spread_adjustment: 0.01,
            min_spread_adjustment: 0.0,
            ..Default::default()
        };
        AdverseSelectionEstimator::new(config)
    }

    // Sleep duration that exceeds the first horizon (500ms)
    const TEST_HORIZON_SLEEP: std::time::Duration = std::time::Duration::from_millis(510);

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

        // Wait for first multi-horizon (500ms)
        std::thread::sleep(TEST_HORIZON_SLEEP);

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

        // Wait for first multi-horizon (500ms)
        std::thread::sleep(TEST_HORIZON_SLEEP);

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
            std::thread::sleep(TEST_HORIZON_SLEEP);
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
            std::thread::sleep(TEST_HORIZON_SLEEP);
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

    #[test]
    fn test_informed_fill_classification() {
        let mut est = make_estimator();
        
        // Set threshold to 5 bps (default)
        assert_eq!(est.informed_threshold_bps(), 5.0);
        
        // Initially no fills classified
        assert_eq!(est.informed_fills_count(), 0);
        assert_eq!(est.uninformed_fills_count(), 0);
        assert_eq!(est.empirical_alpha_touch(), None);
        
        // Add an informed fill (adverse move > 5 bps = 0.05%)
        est.record_fill(1, 1.0, true, 100.0);
        std::thread::sleep(TEST_HORIZON_SLEEP);
        // Price drops 0.1% = 10 bps adverse (> 5 bps threshold)
        est.update(99.9);
        
        assert_eq!(est.informed_fills_count(), 1);
        assert_eq!(est.uninformed_fills_count(), 0);
        
        // Add an uninformed fill (adverse move < 5 bps)
        est.record_fill(2, 1.0, true, 100.0);
        std::thread::sleep(TEST_HORIZON_SLEEP);
        // Price drops 0.02% = 2 bps adverse (< 5 bps threshold)
        est.update(99.98);
        
        assert_eq!(est.informed_fills_count(), 1);
        assert_eq!(est.uninformed_fills_count(), 1);
        
        // Check empirical alpha_touch = 1/2 = 0.5
        assert!((est.empirical_alpha_touch().unwrap() - 0.5).abs() < 0.01);
        
        // Take counts resets them
        let (informed, uninformed) = est.take_informed_counts();
        assert_eq!(informed, 1);
        assert_eq!(uninformed, 1);
        assert_eq!(est.informed_fills_count(), 0);
        assert_eq!(est.uninformed_fills_count(), 0);
    }

    #[test]
    fn test_informed_threshold_adjustment() {
        let mut est = make_estimator();
        
        // Can adjust threshold
        est.set_informed_threshold_bps(10.0);
        assert_eq!(est.informed_threshold_bps(), 10.0);
        
        // With higher threshold, same fill becomes uninformed
        est.record_fill(1, 1.0, true, 100.0);
        std::thread::sleep(TEST_HORIZON_SLEEP);
        // Price drops 0.08% = 8 bps adverse (< 10 bps threshold now)
        est.update(99.92);
        
        assert_eq!(est.informed_fills_count(), 0);
        assert_eq!(est.uninformed_fills_count(), 1);
    }
}
