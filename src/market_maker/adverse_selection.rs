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
//! - **Depth-Dependent AS**: Exponential decay model AS(δ) = AS₀ × exp(-δ/δ_char)
//!   calibrated from fill history by depth bucket

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::debug;

// ============================================================================
// Depth-Dependent Adverse Selection Model (First Principles)
// ============================================================================

/// Depth buckets for AS calibration (in basis points)
const DEPTH_BUCKETS: [(f64, f64); 4] = [
    (0.0, 2.0),   // Touch: 0-2 bps
    (2.0, 5.0),   // Near: 2-5 bps
    (5.0, 10.0),  // Mid: 5-10 bps
    (10.0, 100.0), // Deep: 10+ bps
];

/// Pending fill awaiting AS measurement (for depth-aware calibration).
#[derive(Debug, Clone)]
struct PendingDepthFill {
    /// Trade ID for deduplication
    tid: u64,
    /// Fill time for horizon tracking
    fill_time: Instant,
    /// Mid price at fill time (for AS calculation)
    fill_mid: f64,
    /// Fill price
    fill_price: f64,
    /// Fill size
    size: f64,
    /// True if this was a buy fill
    is_buy: bool,
    /// Depth from mid at fill time in basis points
    depth_bps: f64,
}

/// A fill with associated depth information for calibration.
#[derive(Debug, Clone)]
pub struct FillWithDepth {
    /// Trade ID for deduplication
    pub tid: u64,
    /// Fill size
    pub size: f64,
    /// True if this was a buy fill
    pub is_buy: bool,
    /// Depth from mid at fill time in basis points
    pub depth_bps: f64,
    /// Realized adverse selection (measured after horizon)
    pub realized_as: f64,
}

/// Depth-dependent adverse selection model.
///
/// Implements the first-principles formula: AS(δ) = AS₀ × exp(-δ/δ_char)
///
/// Where:
/// - AS₀ = adverse selection at the touch (calibrated from fills at depth 0-2bp)
/// - δ_char = characteristic depth where AS decays to 37% (e^-1)
/// - Deep levels have exponentially less informed flow
///
/// This model captures the empirical fact that informed traders prioritize
/// execution speed over price improvement, so they hit the touch more than
/// deep levels.
#[derive(Debug, Clone)]
pub struct DepthDecayAS {
    /// AS at touch in basis points (calibrated from fills at depth 0-2bp)
    pub as_touch_bps: f64,
    /// Characteristic depth for exponential decay in basis points
    /// At δ = δ_char, AS decays to ~37% of touch AS
    pub delta_char_bps: f64,
    /// Calibration confidence (0-1), based on fill count and R²
    pub confidence: f64,
    /// Fill history by bucket for ongoing calibration
    bucket_fills: [Vec<f64>; 4], // AS measurements per bucket
    /// Total fills used for calibration
    total_calibration_fills: usize,
    /// Minimum fills per bucket before calibration starts
    min_fills_per_bucket: usize,
    /// EWMA decay for bucket AS estimates
    ewma_alpha: f64,
    /// EWMA AS estimate per bucket
    bucket_as_ewma: [f64; 4],
    /// Pending fills awaiting AS measurement (stochastic module integration)
    pending_fills: std::collections::VecDeque<PendingDepthFill>,
    /// Measurement horizon in milliseconds (default 1000ms)
    measurement_horizon_ms: u64,
    /// Maximum pending fills to track
    max_pending_fills: usize,
}

impl Default for DepthDecayAS {
    fn default() -> Self {
        Self {
            // Conservative defaults before calibration
            as_touch_bps: 3.0,      // 3 bps AS at touch (typical)
            delta_char_bps: 8.0,    // AS decays to 37% by 8 bps depth
            confidence: 0.0,        // No confidence until calibrated
            bucket_fills: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            total_calibration_fills: 0,
            min_fills_per_bucket: 10, // Need 10 fills per bucket
            ewma_alpha: 0.1,
            bucket_as_ewma: [0.0; 4],
            // Stochastic module integration - pending fill tracking
            pending_fills: std::collections::VecDeque::new(),
            measurement_horizon_ms: 1000, // 1 second AS measurement horizon
            max_pending_fills: 100,       // Track up to 100 pending fills
        }
    }
}

impl DepthDecayAS {
    /// Create with custom initial parameters.
    pub fn new(as_touch_bps: f64, delta_char_bps: f64) -> Self {
        Self {
            as_touch_bps,
            delta_char_bps,
            ..Default::default()
        }
    }

    /// Adverse selection at arbitrary depth: AS(δ) = AS₀ × exp(-δ/δ_char)
    ///
    /// This is the core first-principles formula from the manual.
    pub fn as_at_depth(&self, depth_bps: f64) -> f64 {
        self.as_touch_bps * (-depth_bps / self.delta_char_bps).exp()
    }

    /// Expected spread capture at depth: SC(δ) = δ - AS(δ) - fees
    ///
    /// Positive spread capture means profitable, negative means losing money.
    pub fn spread_capture(&self, depth_bps: f64, fees_bps: f64) -> f64 {
        depth_bps - self.as_at_depth(depth_bps) - fees_bps
    }

    /// Find depth where spread capture becomes positive (break-even depth)
    ///
    /// Solves: δ - AS₀ × exp(-δ/δ_char) - fees = 0
    /// Using Newton-Raphson iteration.
    pub fn break_even_depth(&self, fees_bps: f64) -> f64 {
        // Initial guess: AS₀ + fees (would be break-even with no decay)
        let mut delta = self.as_touch_bps + fees_bps;

        for _ in 0..10 {
            let f = delta - self.as_touch_bps * (-delta / self.delta_char_bps).exp() - fees_bps;
            let f_prime = 1.0 + (self.as_touch_bps / self.delta_char_bps)
                * (-delta / self.delta_char_bps).exp();

            if f_prime.abs() < 1e-10 {
                break;
            }

            let delta_new = delta - f / f_prime;
            if (delta_new - delta).abs() < 0.01 {
                return delta_new.max(0.0);
            }
            delta = delta_new;
        }

        delta.max(0.0)
    }

    /// Record a fill for calibration.
    ///
    /// Call this after measuring realized AS for a fill at a known depth.
    pub fn record_fill(&mut self, fill: &FillWithDepth) {
        // Find the bucket for this depth
        let bucket_idx = self.get_bucket_index(fill.depth_bps);

        // Update EWMA for this bucket
        let alpha = self.ewma_alpha;
        self.bucket_as_ewma[bucket_idx] =
            alpha * fill.realized_as.abs() + (1.0 - alpha) * self.bucket_as_ewma[bucket_idx];

        // Store for batch calibration (keep last 100 per bucket)
        if self.bucket_fills[bucket_idx].len() >= 100 {
            self.bucket_fills[bucket_idx].remove(0);
        }
        self.bucket_fills[bucket_idx].push(fill.realized_as.abs());

        self.total_calibration_fills += 1;

        // Attempt recalibration periodically
        if self.total_calibration_fills.is_multiple_of(20) {
            self.calibrate();
        }
    }

    /// Get bucket index for a given depth.
    fn get_bucket_index(&self, depth_bps: f64) -> usize {
        for (i, (min, max)) in DEPTH_BUCKETS.iter().enumerate() {
            if depth_bps >= *min && depth_bps < *max {
                return i;
            }
        }
        3 // Deep bucket for anything beyond 10bp
    }

    /// Calibrate AS₀ and δ_char from fill history.
    ///
    /// Uses weighted least squares on log-transformed AS values:
    /// ln(AS) = ln(AS₀) - δ/δ_char
    ///
    /// This is a linear regression: y = a + bx where
    /// - y = ln(AS)
    /// - x = δ (bucket midpoint)
    /// - a = ln(AS₀)
    /// - b = -1/δ_char
    pub fn calibrate(&mut self) {
        // Check if we have enough fills in each bucket
        let has_enough = self.bucket_fills.iter()
            .all(|b| b.len() >= self.min_fills_per_bucket);

        if !has_enough {
            return;
        }

        // Compute mean AS per bucket
        let bucket_means: Vec<f64> = self.bucket_fills.iter()
            .map(|fills| {
                if fills.is_empty() {
                    0.0
                } else {
                    fills.iter().sum::<f64>() / fills.len() as f64
                }
            })
            .collect();

        // Bucket midpoints for regression
        let bucket_midpoints: Vec<f64> = DEPTH_BUCKETS.iter()
            .map(|(min, max)| (min + max) / 2.0)
            .collect();

        // Weighted least squares on log(AS) = log(AS₀) - δ/δ_char
        // Weight by number of fills in each bucket
        let weights: Vec<f64> = self.bucket_fills.iter()
            .map(|b| b.len() as f64)
            .collect();

        // Filter out buckets with zero or very small AS (can't take log)
        let valid_data: Vec<(f64, f64, f64)> = bucket_midpoints.iter()
            .zip(bucket_means.iter())
            .zip(weights.iter())
            .filter(|((_, &as_val), _)| as_val > 0.001) // 0.1 bps minimum
            .map(|((&depth, &as_val), &w)| (depth, as_val.ln(), w))
            .collect();

        if valid_data.len() < 2 {
            return; // Need at least 2 points for regression
        }

        // Weighted linear regression: y = a + bx
        let sum_w: f64 = valid_data.iter().map(|(_, _, w)| w).sum();
        let sum_wx: f64 = valid_data.iter().map(|(x, _, w)| w * x).sum();
        let sum_wy: f64 = valid_data.iter().map(|(_, y, w)| w * y).sum();
        let sum_wxx: f64 = valid_data.iter().map(|(x, _, w)| w * x * x).sum();
        let sum_wxy: f64 = valid_data.iter().map(|(x, y, w)| w * x * y).sum();

        let denom = sum_w * sum_wxx - sum_wx * sum_wx;
        if denom.abs() < 1e-10 {
            return;
        }

        let b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
        let a = (sum_wy - b * sum_wx) / sum_w;

        // Extract parameters
        let new_as_touch = a.exp(); // AS₀ = exp(a)
        let new_delta_char = if b.abs() > 1e-10 { -1.0 / b } else { self.delta_char_bps };

        // Compute R² for confidence
        let y_mean = sum_wy / sum_w;
        let ss_tot: f64 = valid_data.iter()
            .map(|(_, y, w)| w * (y - y_mean).powi(2))
            .sum();
        let ss_res: f64 = valid_data.iter()
            .map(|(x, y, w)| {
                let y_pred = a + b * x;
                w * (y - y_pred).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 1e-10 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        // Only update if results are sensible
        if new_as_touch > 0.1 && new_as_touch < 50.0
            && new_delta_char > 1.0 && new_delta_char < 100.0
            && r_squared > 0.3
        {
            // Smooth update to avoid jumps
            let smooth = 0.2;
            self.as_touch_bps = smooth * new_as_touch + (1.0 - smooth) * self.as_touch_bps;
            self.delta_char_bps = smooth * new_delta_char + (1.0 - smooth) * self.delta_char_bps;
            self.confidence = r_squared.sqrt(); // sqrt(R²) as confidence

            debug!(
                as_touch = self.as_touch_bps,
                delta_char = self.delta_char_bps,
                confidence = self.confidence,
                r_squared = r_squared,
                fills = self.total_calibration_fills,
                "DepthDecayAS: Calibrated from fill history"
            );
        }
    }

    /// Check if model is calibrated with sufficient confidence.
    pub fn is_calibrated(&self) -> bool {
        self.confidence > 0.5 && self.total_calibration_fills >= 40
    }

    /// Get summary for diagnostics.
    pub fn summary(&self) -> DepthDecayASSummary {
        DepthDecayASSummary {
            as_touch_bps: self.as_touch_bps,
            delta_char_bps: self.delta_char_bps,
            confidence: self.confidence,
            total_fills: self.total_calibration_fills,
            bucket_counts: self.bucket_fills.iter().map(|b| b.len()).collect(),
            bucket_as_bps: self.bucket_as_ewma.to_vec(),
        }
    }

    // ========================================================================
    // Stochastic Module Integration: Pending Fill Tracking
    // ========================================================================

    /// Record a pending fill for depth-aware AS calibration.
    ///
    /// Call this immediately when a fill is received, with fill price and current mid.
    /// The fill will be resolved later in `resolve_pending_fills` after the
    /// measurement horizon elapses.
    ///
    /// # Arguments
    /// * `tid` - Trade ID for deduplication
    /// * `fill_price` - Price at which the fill executed
    /// * `size` - Fill size
    /// * `is_buy` - True if this was a buy fill
    /// * `current_mid` - Mid price at fill time
    pub fn record_pending_fill(
        &mut self,
        tid: u64,
        fill_price: f64,
        size: f64,
        is_buy: bool,
        current_mid: f64,
    ) {
        // Compute depth from mid in basis points
        let depth_bps = if current_mid > 0.0 {
            ((fill_price - current_mid) / current_mid).abs() * 10000.0
        } else {
            0.0
        };

        // Enforce max pending fills (FIFO eviction)
        while self.pending_fills.len() >= self.max_pending_fills {
            self.pending_fills.pop_front();
        }

        self.pending_fills.push_back(PendingDepthFill {
            tid,
            fill_time: Instant::now(),
            fill_mid: current_mid,
            fill_price,
            size,
            is_buy,
            depth_bps,
        });

        tracing::debug!(
            tid = tid,
            fill_price = fill_price,
            mid = current_mid,
            depth_bps = depth_bps,
            is_buy = is_buy,
            pending = self.pending_fills.len(),
            "DepthDecayAS: Recorded pending fill for calibration"
        );
    }

    /// Resolve pending fills whose measurement horizon has elapsed.
    ///
    /// Call this on each mid price update to resolve fills and feed them
    /// into the depth-dependent AS calibration.
    ///
    /// # Arguments
    /// * `current_mid` - Current mid price for AS calculation
    pub fn resolve_pending_fills(&mut self, current_mid: f64) {
        let now = Instant::now();
        let horizon = std::time::Duration::from_millis(self.measurement_horizon_ms);

        // Process all pending fills whose horizon has elapsed
        while let Some(front) = self.pending_fills.front() {
            if now.duration_since(front.fill_time) < horizon {
                break; // Not ready yet
            }

            let fill = self.pending_fills.pop_front().unwrap();

            // Calculate realized adverse selection (price change as fraction)
            let price_change = (current_mid - fill.fill_mid) / fill.fill_mid;

            // AS from MM's perspective:
            // - Buy fill: price going DOWN is adverse (we bought, price dropped)
            // - Sell fill: price going UP is adverse (we sold, price rose)
            let realized_as = if fill.is_buy {
                -price_change // Positive AS if price dropped after buying
            } else {
                price_change // Positive AS if price rose after selling
            };

            // Convert to basis points
            let realized_as_bps = realized_as * 10000.0;

            // Create FillWithDepth and feed to calibration
            let fill_with_depth = FillWithDepth {
                tid: fill.tid,
                size: fill.size,
                is_buy: fill.is_buy,
                depth_bps: fill.depth_bps,
                realized_as: realized_as_bps,
            };

            self.record_fill(&fill_with_depth);

            tracing::debug!(
                tid = fill.tid,
                depth_bps = fill.depth_bps,
                realized_as_bps = realized_as_bps,
                is_buy = fill.is_buy,
                "DepthDecayAS: Resolved fill for calibration"
            );
        }
    }

    /// Get number of pending fills awaiting resolution.
    pub fn pending_count(&self) -> usize {
        self.pending_fills.len()
    }
}

/// Summary of depth-decay AS model for logging.
#[derive(Debug, Clone)]
pub struct DepthDecayASSummary {
    pub as_touch_bps: f64,
    pub delta_char_bps: f64,
    pub confidence: f64,
    pub total_fills: usize,
    pub bucket_counts: Vec<usize>,
    pub bucket_as_bps: Vec<f64>,
}

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
        self.realized_as_total = alpha * signed_as.abs() + (1.0 - alpha) * self.realized_as_total;
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

    // ========================================================================
    // DepthDecayAS Tests
    // ========================================================================

    #[test]
    fn test_depth_decay_as_formula() {
        let model = DepthDecayAS::new(5.0, 10.0); // AS₀=5bp, δ_char=10bp

        // At touch (0 bps): AS = 5.0 * exp(0) = 5.0
        assert!((model.as_at_depth(0.0) - 5.0).abs() < 0.01);

        // At characteristic depth (10 bps): AS = 5.0 * exp(-1) ≈ 1.84
        assert!((model.as_at_depth(10.0) - 5.0 * (-1.0_f64).exp()).abs() < 0.01);

        // At 2x characteristic depth (20 bps): AS = 5.0 * exp(-2) ≈ 0.68
        assert!((model.as_at_depth(20.0) - 5.0 * (-2.0_f64).exp()).abs() < 0.01);

        // Deep (50 bps): AS ≈ 0 (effectively)
        assert!(model.as_at_depth(50.0) < 0.1);
    }

    #[test]
    fn test_depth_decay_spread_capture() {
        let model = DepthDecayAS::new(5.0, 10.0);
        let fees = 0.5; // 0.5 bps fees

        // At touch: SC = 0 - 5.0 - 0.5 = -5.5 (unprofitable)
        assert!(model.spread_capture(0.0, fees) < 0.0);

        // At 5 bps: SC = 5 - 5*exp(-0.5) - 0.5 ≈ 5 - 3.03 - 0.5 = 1.47
        let sc_5 = model.spread_capture(5.0, fees);
        assert!(sc_5 > 0.0);

        // At 20 bps: SC ≈ 20 - 0.68 - 0.5 = 18.82 (very profitable)
        let sc_20 = model.spread_capture(20.0, fees);
        assert!(sc_20 > 15.0);
    }

    #[test]
    fn test_depth_decay_break_even() {
        let model = DepthDecayAS::new(5.0, 10.0);
        let fees = 0.5;

        let be = model.break_even_depth(fees);

        // At break-even, spread capture should be ~0
        let sc_at_be = model.spread_capture(be, fees);
        assert!(sc_at_be.abs() < 0.1); // Within 0.1 bps of break-even
    }

    #[test]
    fn test_depth_decay_calibration() {
        let mut model = DepthDecayAS::default();

        // Simulate fills with known AS pattern (higher at touch, decaying)
        // Touch bucket: 0-2bp → high AS (~4bp)
        for i in 0..15 {
            model.record_fill(&FillWithDepth {
                tid: i,
                size: 1.0,
                is_buy: true,
                depth_bps: 1.0,
                realized_as: 0.0004, // 4 bps
            });
        }

        // Near bucket: 2-5bp → medium AS (~2bp)
        for i in 15..30 {
            model.record_fill(&FillWithDepth {
                tid: i,
                size: 1.0,
                is_buy: true,
                depth_bps: 3.5,
                realized_as: 0.0002, // 2 bps
            });
        }

        // Mid bucket: 5-10bp → lower AS (~1bp)
        for i in 30..45 {
            model.record_fill(&FillWithDepth {
                tid: i,
                size: 1.0,
                is_buy: true,
                depth_bps: 7.5,
                realized_as: 0.0001, // 1 bp
            });
        }

        // Deep bucket: 10+bp → very low AS (~0.5bp)
        for i in 45..60 {
            model.record_fill(&FillWithDepth {
                tid: i,
                size: 1.0,
                is_buy: true,
                depth_bps: 15.0,
                realized_as: 0.00005, // 0.5 bps
            });
        }

        // Model should have calibrated
        assert!(model.total_calibration_fills >= 40);

        // AS at touch should be positive
        assert!(model.as_touch_bps > 0.0);

        // Delta char should be reasonable (between 1 and 100)
        assert!(model.delta_char_bps > 1.0 && model.delta_char_bps < 100.0);
    }

    #[test]
    fn test_depth_decay_bucket_assignment() {
        let model = DepthDecayAS::default();

        // Test bucket boundaries
        assert_eq!(model.get_bucket_index(0.5), 0); // Touch
        assert_eq!(model.get_bucket_index(1.5), 0); // Touch
        assert_eq!(model.get_bucket_index(2.5), 1); // Near
        assert_eq!(model.get_bucket_index(4.0), 1); // Near
        assert_eq!(model.get_bucket_index(6.0), 2); // Mid
        assert_eq!(model.get_bucket_index(9.0), 2); // Mid
        assert_eq!(model.get_bucket_index(15.0), 3); // Deep
        assert_eq!(model.get_bucket_index(100.0), 3); // Deep
    }

    #[test]
    fn test_depth_decay_summary() {
        let model = DepthDecayAS::new(4.0, 12.0);
        let summary = model.summary();

        assert!((summary.as_touch_bps - 4.0).abs() < 0.01);
        assert!((summary.delta_char_bps - 12.0).abs() < 0.01);
        assert_eq!(summary.total_fills, 0);
        assert_eq!(summary.bucket_counts.len(), 4);
    }
}
