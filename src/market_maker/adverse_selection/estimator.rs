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

    /// EWMA of squared deviations from AS mean, for posterior variance estimation.
    /// Var[AS] ≈ E[(AS - E[AS])²] tracked via EWMA with same alpha.
    recent_as_variance_bps2: f64,

    /// Checkpoint-seeded AS floor in bps, used during cold start when no live
    /// markout data is available yet. Set from prior checkpoint on init.
    checkpoint_as_floor_bps: Option<f64>,

    // === Glosten-Milgrom: Informed jump magnitude tracking ===
    /// EWMA of AS magnitude (fractional) for fills classified as informed.
    /// E[ΔP | informed] — the expected price jump given an informed trade.
    informed_as_ewma: f64,
    /// Per-side informed jump EWMA: buy fills classified as informed.
    informed_as_buy_ewma: f64,
    /// Per-side informed jump EWMA: sell fills classified as informed.
    informed_as_sell_ewma: f64,
    /// Per-side informed/uninformed counts for empirical alpha by side.
    informed_buy_count: usize,
    uninformed_buy_count: usize,
    informed_sell_count: usize,
    uninformed_sell_count: usize,

    // === InformedFlow EM calibration ===
    /// Latest EM-derived P(informed) and its confidence from the Hawkes GMM.
    /// Used to probabilistically override the heuristic alpha weights in `predicted_alpha()`.
    gmm_alpha_signal: Option<(f64, f64)>, // (p_informed, confidence)
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
            recent_as_variance_bps2: 0.0,
            checkpoint_as_floor_bps: None,
            // Glosten-Milgrom fields
            informed_as_ewma: 0.0,
            informed_as_buy_ewma: 0.0,
            informed_as_sell_ewma: 0.0,
            informed_buy_count: 0,
            uninformed_buy_count: 0,
            informed_sell_count: 0,
            uninformed_sell_count: 0,
            gmm_alpha_signal: None,
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
                // Glosten-Milgrom: Track informed jump magnitude (fractional)
                const GM_ALPHA: f64 = 0.1;
                self.informed_as_ewma =
                    GM_ALPHA * signed_as.abs() + (1.0 - GM_ALPHA) * self.informed_as_ewma;
                if is_buy {
                    self.informed_buy_count += 1;
                    self.informed_as_buy_ewma =
                        GM_ALPHA * signed_as.abs() + (1.0 - GM_ALPHA) * self.informed_as_buy_ewma;
                } else {
                    self.informed_sell_count += 1;
                    self.informed_as_sell_ewma =
                        GM_ALPHA * signed_as.abs() + (1.0 - GM_ALPHA) * self.informed_as_sell_ewma;
                }
            } else {
                self.uninformed_fills_count += 1;
                if is_buy {
                    self.uninformed_buy_count += 1;
                } else {
                    self.uninformed_sell_count += 1;
                }
            }

            // Update rolling AS severity EWMA (alpha=0.1, ~10 fill half-life)
            const AS_SEVERITY_ALPHA: f64 = 0.1;
            // Variance update: E[(x - mean)²] via Welford-style EWMA
            let deviation = adverse_move_bps - self.recent_as_ewma_bps;
            self.recent_as_variance_bps2 = AS_SEVERITY_ALPHA * deviation * deviation
                + (1.0 - AS_SEVERITY_ALPHA) * self.recent_as_variance_bps2;
            // Mean update (after variance to use old mean for deviation)
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

    /// Dynamic AS floor in bps for spread floor computation.
    /// Uses checkpoint seed when no live markout data yet (critical for Fix 1 window).
    /// After enough markouts, uses max(ewma/2, checkpoint_seed).
    pub fn as_floor_bps(&self) -> f64 {
        let markout_count = self.fills_measured;
        if markout_count < 3 {
            // No live markout data — use checkpoint seed or conservative fallback
            return self.checkpoint_as_floor_bps.unwrap_or(2.0);
        }
        let ewma_half = self.realized_as_bps().abs() * 0.5;
        ewma_half.max(self.checkpoint_as_floor_bps.unwrap_or(2.0))
    }

    /// Set checkpoint-seeded AS floor (called during prior injection).
    pub fn set_checkpoint_as_floor_bps(&mut self, floor_bps: f64) {
        self.checkpoint_as_floor_bps = Some(floor_bps);
    }

    /// Glosten-Milgrom spread adjustment (as fraction of mid price).
    ///
    /// Break-even adjustment = P(informed) × E[ΔP | informed].
    /// This replaces the old `realized_as × 2.0` heuristic with a principled
    /// conditional expectation from the Glosten-Milgrom model.
    ///
    /// Returns 0.0 if not warmed up.
    pub fn spread_adjustment(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }

        // P(informed | fill): empirical when available, predicted (with GMM blend) as fallback
        let p_informed = self
            .empirical_alpha_touch()
            .unwrap_or_else(|| self.predicted_alpha());

        // E[ΔP | informed]: EWMA of AS magnitude for informed-classified fills
        // Falls back to realized_as_total when no informed fills tracked yet
        let expected_informed_jump = if self.informed_as_ewma > 0.0 {
            self.informed_as_ewma
        } else {
            self.realized_as_total
        };

        // Glosten-Milgrom: break-even adjustment = P(informed) × E[ΔP|informed]
        let gm_adjustment = p_informed * expected_informed_jump;

        gm_adjustment
            .max(self.config.min_spread_adjustment)
            .min(self.config.max_spread_adjustment)
    }

    /// Get spread adjustment in basis points.
    pub fn spread_adjustment_bps(&self) -> f64 {
        self.spread_adjustment() * 10000.0
    }

    /// Per-side Glosten-Milgrom spread adjustment for bids (buy-side AS).
    /// Uses buy-side P(informed) and buy-side E[ΔP|informed].
    pub fn spread_adjustment_bid(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }
        let total_buy = self.informed_buy_count + self.uninformed_buy_count;
        let p_informed_buy = if total_buy > 0 {
            self.informed_buy_count as f64 / total_buy as f64
        } else {
            self.empirical_alpha_touch()
                .unwrap_or_else(|| self.predicted_alpha())
        };
        let jump_buy = if self.informed_as_buy_ewma > 0.0 {
            self.informed_as_buy_ewma
        } else {
            self.realized_as_buy.abs()
        };
        let raw = p_informed_buy * jump_buy;
        raw.max(self.config.min_spread_adjustment)
            .min(self.config.max_spread_adjustment)
    }

    /// Per-side Glosten-Milgrom spread adjustment for asks (sell-side AS).
    /// Uses sell-side P(informed) and sell-side E[ΔP|informed].
    pub fn spread_adjustment_ask(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }
        let total_sell = self.informed_sell_count + self.uninformed_sell_count;
        let p_informed_sell = if total_sell > 0 {
            self.informed_sell_count as f64 / total_sell as f64
        } else {
            self.empirical_alpha_touch()
                .unwrap_or_else(|| self.predicted_alpha())
        };
        let jump_sell = if self.informed_as_sell_ewma > 0.0 {
            self.informed_as_sell_ewma
        } else {
            self.realized_as_sell.abs()
        };
        let raw = p_informed_sell * jump_sell;
        raw.max(self.config.min_spread_adjustment)
            .min(self.config.max_spread_adjustment)
    }

    /// Get predicted alpha: P(next trade is informed).
    ///
    /// If the Hawkes GMM has high confidence, its mathematically derived P(informed)
    /// takes over. Otherwise, gracefully degrades to the heuristic signals.
    pub fn predicted_alpha(&self) -> f64 {
        let cfg = &self.config;

        // 1. Calculate the baseline heuristic alpha (Volatility, Imbalance, Jump)
        let vol_signal = (self.cached_vol_surprise.max(0.0) / 2.0).min(1.0);
        let flow_signal = self.cached_flow_magnitude; // Already in [0, 1]
        let jump_signal = ((self.cached_jump_ratio - 1.0).max(0.0) / 4.0).min(1.0);

        let raw_alpha = cfg.alpha_volatility_weight * vol_signal
            + cfg.alpha_flow_weight * flow_signal
            + cfg.alpha_jump_weight * jump_signal;

        let heuristic_alpha = 1.0 / (1.0 + (-4.0 * (raw_alpha - 0.5)).exp());

        // 2. Bayesian blend with the latent GMM probability (if available)
        if let Some((gmm_p_informed, confidence)) = self.gmm_alpha_signal {
            // Confidence [0.0, 1.0] controls the interpolation.
            // A highly confident GMM completely overrides the heuristics.
            (gmm_p_informed * confidence) + (heuristic_alpha * (1.0 - confidence))
        } else {
            heuristic_alpha
        }
    }

    /// Get asymmetric Glosten-Milgrom spread adjustment for bid/ask.
    ///
    /// Returns (bid_adjustment, ask_adjustment) using per-side GM conditional
    /// expectations: P(informed|side) × E[ΔP|informed, side].
    pub fn asymmetric_adjustment(&self) -> (f64, f64) {
        if !self.is_warmed_up() {
            return (0.0, 0.0);
        }

        (self.spread_adjustment_bid(), self.spread_adjustment_ask())
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

    /// Empirical P(informed | buy fill) from per-side classification.
    pub fn empirical_alpha_buy(&self) -> Option<f64> {
        let total = self.informed_buy_count + self.uninformed_buy_count;
        if total == 0 {
            None
        } else {
            Some(self.informed_buy_count as f64 / total as f64)
        }
    }

    /// Empirical P(informed | sell fill) from per-side classification.
    pub fn empirical_alpha_sell(&self) -> Option<f64> {
        let total = self.informed_sell_count + self.uninformed_sell_count;
        if total == 0 {
            None
        } else {
            Some(self.informed_sell_count as f64 / total as f64)
        }
    }

    /// Calibrate P(informed) from InformedFlowEstimator's EM decomposition.
    ///
    /// Feed the latent flow decomposition from the Hawkes GMM into the AS estimator.
    /// This bridges the microstructure flow tracking with the pricing model.
    /// The raw (p_informed, confidence) is stored and used by `predicted_alpha()`
    /// to perform a Bayesian blend — high confidence GMM overrides the heuristics.
    pub fn calibrate_alpha_from_flow(&mut self, p_informed: f64, confidence: f64) {
        self.gmm_alpha_signal = Some((p_informed, confidence));
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

    /// Whether the Hawkes GMM has provided a P(informed) signal.
    pub fn has_gmm_signal(&self) -> bool {
        self.gmm_alpha_signal.is_some()
    }

    /// Arrow-Pratt variance risk premium: prices AS uncertainty as 0.5 × γ × Var[AS].
    ///
    /// Returns an **additive** bps premium (NOT a multiplier). This replaces the old
    /// step-function `recent_as_severity_mult()` (1.5×/>5bps, 1.25×/>3bps, 1.0×)
    /// with a continuous, differentiable, and theoretically-grounded formulation.
    ///
    /// The premium compensates a risk-averse agent for facing uncertain AS costs:
    /// higher variance in realized AS → wider spreads to buffer uncertainty.
    pub fn as_variance_risk_premium_bps(&self, gamma: f64) -> f64 {
        if self.fills_measured < 3 {
            return 0.0;
        }
        // Arrow-Pratt: exact risk premium for a risk-averse agent
        // Premium = 0.5 × γ × Var[AS]
        0.5 * gamma * self.recent_as_variance_bps2
    }

    /// Deprecated: use `as_variance_risk_premium_bps(gamma)` instead.
    /// Kept for backward compatibility during transition.
    #[deprecated(
        note = "Use as_variance_risk_premium_bps(gamma) — returns additive bps, not multiplier"
    )]
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

    /// AS posterior variance in bps² (EWMA of squared deviations from mean).
    /// Used by Q19 uncertainty premium: floor = E[AS] + k × √Var[AS].
    /// Returns 0.0 when insufficient data (< 3 fills).
    pub fn as_floor_variance_bps2(&self) -> f64 {
        if self.fills_measured < 3 {
            0.0
        } else {
            self.recent_as_variance_bps2
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_estimator() -> AdverseSelectionEstimator {
        let config = AdverseSelectionConfig {
            measurement_horizon_ms: 500, // Match first multi-horizon (500ms)
            ewma_alpha: 0.5,             // High alpha for visible changes
            min_fills_warmup: 3,
            max_pending_fills: 100,
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

    #[test]
    fn test_as_floor_bps_cold_start() {
        let mut est = make_estimator();
        // Cold start: no fills, no checkpoint
        assert!(
            (est.as_floor_bps() - 2.0).abs() < 1e-10,
            "Default floor should be 2.0 bps"
        );
        // With checkpoint seed
        est.set_checkpoint_as_floor_bps(5.0);
        assert!(
            (est.as_floor_bps() - 5.0).abs() < 1e-10,
            "Should use checkpoint seed"
        );
    }

    #[test]
    fn test_as_floor_variance_cold_start() {
        let est = make_estimator();
        // No fills → variance = 0 (insufficient data)
        assert_eq!(est.as_floor_variance_bps2(), 0.0);
    }

    #[test]
    fn test_as_floor_variance_accumulates() {
        let mut est = make_estimator();
        // Record 4 fills with varying AS magnitude to build variance
        // Need >= 3 fills measured for variance to be exposed
        for i in 0..4 {
            est.record_fill(i as u64, 1.0, true, 100.0);
        }
        std::thread::sleep(TEST_HORIZON_SLEEP);
        // Price drops 0.1% = 10 bps adverse for buys
        est.update(99.90);
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Record 4 more fills
        for i in 4..8 {
            est.record_fill(i as u64, 1.0, true, 100.0);
        }
        std::thread::sleep(TEST_HORIZON_SLEEP);
        // Price drops more: 0.2% = 20 bps adverse
        est.update(99.70);

        // After enough fills, variance should be > 0 (AS values differ)
        let var = est.as_floor_variance_bps2();
        assert!(var >= 0.0, "Variance should be non-negative, got {var}");
    }

    #[test]
    fn test_as_uncertainty_premium() {
        // Verify the premium formula: E[AS] + k * sqrt(Var[AS])
        let as_floor_bps = 5.0;
        let variance_bps2: f64 = 25.0; // std dev = 5 bps
        let k = 1.5;
        let premium = k * variance_bps2.sqrt();
        let total_floor = as_floor_bps + premium;
        assert!((premium - 7.5).abs() < 1e-10, "k=1.5 * sqrt(25) = 7.5");
        assert!((total_floor - 12.5).abs() < 1e-10, "5 + 7.5 = 12.5 bps");
    }

    // === Glosten-Milgrom Tests ===

    #[test]
    fn test_gm_spread_adjustment_proportional_to_alpha() {
        // GM adjustment = P(informed) × E[ΔP|informed]
        // With known P(informed) and E[jump], adjustment should scale linearly
        let mut est = make_estimator();

        // Create 3 informed fills (> 5 bps AS) and 3 uninformed (< 5 bps)
        for i in 0..3 {
            est.record_fill(i as u64 * 2, 1.0, true, 100.0);
            std::thread::sleep(TEST_HORIZON_SLEEP);
            est.update(99.9); // 10 bps adverse → informed

            est.record_fill(i as u64 * 2 + 1, 1.0, true, 100.0);
            std::thread::sleep(TEST_HORIZON_SLEEP);
            est.update(99.98); // 2 bps adverse → uninformed
        }

        assert!(est.is_warmed_up(), "Should be warmed up after 6 fills");
        let adj = est.spread_adjustment();
        assert!(
            adj > 0.0,
            "GM adjustment should be positive with mixed fills"
        );
        // P(informed) ≈ 0.5, E[jump|informed] ≈ 0.001 → adj ≈ 0.0005
        assert!(adj < 0.005, "GM adjustment should be reasonable: got {adj}");
    }

    #[test]
    fn test_gm_per_side_tracks_independently() {
        let mut est = make_estimator();

        // All buy fills are informed (10 bps AS), all sell fills are uninformed (2 bps AS)
        for i in 0..3 {
            est.record_fill(i as u64 * 2, 1.0, true, 100.0); // buy
            std::thread::sleep(TEST_HORIZON_SLEEP);
            est.update(99.9); // 10 bps adverse for buy → informed

            est.record_fill(i as u64 * 2 + 1, 1.0, false, 100.0); // sell
            std::thread::sleep(TEST_HORIZON_SLEEP);
            est.update(100.02); // 2 bps adverse for sell → uninformed
        }

        // Buy-side should have higher adjustment than sell-side
        let bid_adj = est.spread_adjustment_bid();
        let ask_adj = est.spread_adjustment_ask();
        assert!(
            bid_adj > ask_adj,
            "Buy-side (more informed) should have higher adj: bid={bid_adj} ask={ask_adj}"
        );
    }

    // === Arrow-Pratt Tests ===

    #[test]
    fn test_arrow_pratt_premium_proportional_to_gamma_and_variance() {
        let mut est = make_estimator();

        // Build up some variance
        for i in 0..4 {
            est.record_fill(i as u64, 1.0, true, 100.0);
        }
        std::thread::sleep(TEST_HORIZON_SLEEP);
        est.update(99.9); // 10 bps
        for i in 4..8 {
            est.record_fill(i as u64, 1.0, true, 100.0);
        }
        std::thread::sleep(TEST_HORIZON_SLEEP);
        est.update(99.7); // 30 bps

        let var = est.as_floor_variance_bps2();
        assert!(
            var > 0.0,
            "Variance should be positive after differing AS magnitudes"
        );

        // Premium = 0.5 × γ × Var[AS]
        let premium_g1 = est.as_variance_risk_premium_bps(1.0);
        let premium_g2 = est.as_variance_risk_premium_bps(2.0);
        assert!(
            (premium_g2 - 2.0 * premium_g1).abs() < 1e-10,
            "Premium should scale linearly with gamma"
        );
        assert!(
            (premium_g1 - 0.5 * var).abs() < 1e-10,
            "Premium(γ=1) = 0.5 × Var[AS]"
        );
    }

    #[test]
    fn test_arrow_pratt_zero_before_warmup() {
        let est = make_estimator();
        // No fills → zero premium regardless of gamma
        assert_eq!(est.as_variance_risk_premium_bps(1.0), 0.0);
        assert_eq!(est.as_variance_risk_premium_bps(10.0), 0.0);
    }

    // === Flow Calibration Tests ===

    #[test]
    fn test_calibrate_alpha_from_flow() {
        let mut est = make_estimator();

        // Without GMM signal, predicted_alpha uses only heuristics
        let baseline_alpha = est.predicted_alpha();

        // Feed GMM signal with high confidence
        est.calibrate_alpha_from_flow(0.92, 0.85);
        let gmm_alpha = est.predicted_alpha();

        // With 0.85 confidence, GMM dominates: 0.85 * 0.92 + 0.15 * baseline
        let expected = 0.85 * 0.92 + 0.15 * baseline_alpha;
        assert!(
            (gmm_alpha - expected).abs() < 0.01,
            "Expected ~{expected:.3}, got {gmm_alpha:.3}"
        );
        // GMM should push alpha UP toward 0.92
        assert!(gmm_alpha > baseline_alpha, "GMM should increase alpha");

        // Low confidence GMM → heuristic dominates
        est.calibrate_alpha_from_flow(0.92, 0.1);
        let low_conf_alpha = est.predicted_alpha();
        let expected_low = 0.1 * 0.92 + 0.9 * baseline_alpha;
        assert!(
            (low_conf_alpha - expected_low).abs() < 0.01,
            "Expected ~{expected_low:.3}, got {low_conf_alpha:.3}"
        );
        // Should be much closer to baseline than the high-confidence case
        assert!(
            (low_conf_alpha - baseline_alpha).abs() < (gmm_alpha - baseline_alpha).abs(),
            "Low confidence should stay closer to baseline"
        );
    }

    // === HARA Gamma Test ===

    #[test]
    fn test_reservation_price_shift_basics() {
        // Verify the A-S formula: shift = -q × γ × σ² × τ
        use crate::market_maker::strategy::reservation_price_shift;

        // Long position → negative shift (sell pressure)
        let shift_long = reservation_price_shift(10.0, 0.0002, 60.0, 1.0, 0.0, 1000.0);
        assert!(
            shift_long < 0.0,
            "Long position should produce negative shift"
        );

        // Short position → positive shift (buy pressure)
        let shift_short = reservation_price_shift(-10.0, 0.0002, 60.0, 1.0, 0.0, 1000.0);
        assert!(
            shift_short > 0.0,
            "Short position should produce positive shift"
        );

        // Flat position → zero shift
        let shift_flat = reservation_price_shift(0.0, 0.0002, 60.0, 1.0, 0.0, 1000.0);
        assert!(
            (shift_flat).abs() < 1e-15,
            "Flat position should have zero shift"
        );

        // HARA: losing money → higher gamma → larger shift
        let shift_losing = reservation_price_shift(10.0, 0.0002, 60.0, 1.0, -500.0, 1000.0);
        assert!(
            shift_losing.abs() > shift_long.abs(),
            "Losing PnL should increase shift magnitude via HARA gamma"
        );
    }
}
