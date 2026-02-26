//! Book dynamics tracker: detects informed liquidity pulling from L2 depth changes.
//!
//! Tracks exponential moving averages of bid/ask depth and their change rates
//! to detect when informed agents are thinning one side of the book.
//!
//! Key signals:
//! - **Thinning direction**: which side is losing depth faster ([-1, 1])
//!   - Positive = asks thinning faster → bearish
//!   - Negative = bids thinning faster → bullish
//! - **Depth persistence**: how stable the book is (1.0 = stable, 0.0 = volatile)

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Time constant for depth EMA (seconds).
const TAU_S: f64 = 5.0;

/// Minimum denominator to avoid division by zero.
const EPSILON: f64 = 1e-12;

/// Number of updates required before signals are meaningful.
const WARMUP_COUNT: u64 = 10;

/// Tracks L2 book depth changes over time to detect informed pulling (thinning).
///
/// Uses time-weighted EMAs of bid and ask depth, plus change-rate EMAs,
/// to produce two signals: thinning direction and depth persistence.
///
/// Also tracks multi-depth book imbalance (shallow vs deep) for Book Pressure
/// Gradient (BPG) computation and ΔBIM (rate of change of imbalance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookDynamicsTracker {
    /// EMA of bid-side depth.
    depth_bid_ema: f64,
    /// EMA of ask-side depth.
    depth_ask_ema: f64,
    /// EMA of bid depth change rate (units/second).
    bid_change_rate_ema: f64,
    /// EMA of ask depth change rate (units/second).
    ask_change_rate_ema: f64,
    /// EMA of squared depth changes (for variance estimate).
    depth_change_sq_ema: f64,
    /// EMA of mean total depth (bid + ask).
    mean_depth_ema: f64,
    /// Previous bid depth snapshot.
    last_bid_depth: f64,
    /// Previous ask depth snapshot.
    last_ask_depth: f64,
    /// Timestamp of last update (not serialized — reset on restore).
    #[serde(skip)]
    last_update: Option<Instant>,
    /// Total number of updates received.
    update_count: u64,

    // === Multi-depth BIM for Book Pressure Gradient (Phase 7) ===
    /// EWMA of shallow book imbalance (top 3 levels, ~10 bps from mid).
    #[serde(default)]
    bim_shallow_ema: f64,
    /// EWMA of deep book imbalance (all levels, ~50 bps from mid).
    #[serde(default)]
    bim_deep_ema: f64,
    /// Previous shallow BIM for ΔBIM computation.
    #[serde(default)]
    prev_bim_shallow: f64,
}

/// EWMA decay factor for iceberg hidden ratio (0.05 new observation weight).
const ICEBERG_ALPHA: f64 = 0.05;

/// Threshold for considering a side to have meaningful hidden liquidity support.
const HIDDEN_SUPPORT_THRESHOLD: f64 = 0.3;

/// Detects iceberg (hidden) liquidity by comparing fill sizes against book depth decreases.
///
/// When a fill occurs for X units but the book only decreases by Y < X, the difference
/// `(X - Y) / X` indicates hidden depth that refilled the level. High hidden ratios
/// imply passive support at that price level, allowing tighter quoting.
///
/// Each side maintains an EWMA hidden ratio in [0, 1]:
/// - 0.0 → no hidden liquidity detected
/// - 1.0 → all fills are hidden (extreme iceberg)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergDetector {
    /// EWMA of hidden fraction on the bid side.
    #[serde(default)]
    hidden_ratio_bid: f64,
    /// EWMA of hidden fraction on the ask side.
    #[serde(default)]
    hidden_ratio_ask: f64,
    /// Snapshot of bid depth before a fill (for before/after comparison).
    #[serde(default)]
    last_bid_depth: f64,
    /// Snapshot of ask depth before a fill (for before/after comparison).
    #[serde(default)]
    last_ask_depth: f64,
}

impl IcebergDetector {
    /// Create a new detector with zeroed fields.
    pub fn new() -> Self {
        Self {
            hidden_ratio_bid: 0.0,
            hidden_ratio_ask: 0.0,
            last_bid_depth: 0.0,
            last_ask_depth: 0.0,
        }
    }

    /// Snapshot current book depths before a potential fill.
    ///
    /// Must be called before `on_fill_with_book_update` so the detector can
    /// compare pre-fill vs post-fill depth.
    pub fn snapshot_book(&mut self, bid_depth: f64, ask_depth: f64) {
        self.last_bid_depth = bid_depth;
        self.last_ask_depth = ask_depth;
    }

    /// Record a fill and compare with the new book depth to detect hidden liquidity.
    ///
    /// `is_buy` — true if a buy order filled (consuming ask-side depth).
    /// `fill_size` — size of the fill in base units.
    /// `new_depth_at_level` — the book depth on the consumed side *after* the fill.
    ///
    /// If `fill_size > depth_decrease`, the excess suggests hidden (iceberg) depth
    /// that refilled the level.
    pub fn on_fill_with_book_update(
        &mut self,
        is_buy: bool,
        fill_size: f64,
        new_depth_at_level: f64,
    ) {
        if fill_size <= 0.0 {
            return;
        }

        if is_buy {
            // Buy fills consume ask-side depth
            let depth_decrease = (self.last_ask_depth - new_depth_at_level).max(0.0);
            let hidden_frac = ((fill_size - depth_decrease) / fill_size).clamp(0.0, 1.0);
            self.hidden_ratio_ask =
                self.hidden_ratio_ask * (1.0 - ICEBERG_ALPHA) + hidden_frac * ICEBERG_ALPHA;
            self.last_ask_depth = new_depth_at_level;
        } else {
            // Sell fills consume bid-side depth
            let depth_decrease = (self.last_bid_depth - new_depth_at_level).max(0.0);
            let hidden_frac = ((fill_size - depth_decrease) / fill_size).clamp(0.0, 1.0);
            self.hidden_ratio_bid =
                self.hidden_ratio_bid * (1.0 - ICEBERG_ALPHA) + hidden_frac * ICEBERG_ALPHA;
            self.last_bid_depth = new_depth_at_level;
        }
    }

    /// Returns the EWMA hidden liquidity ratio for the given side.
    ///
    /// Value in [0, 1]: 0 = no hidden, 1 = fully hidden.
    pub fn hidden_liquidity_ratio(&self, is_buy: bool) -> f64 {
        if is_buy {
            self.hidden_ratio_ask
        } else {
            self.hidden_ratio_bid
        }
    }

    /// Whether both sides show significant hidden liquidity (> threshold).
    ///
    /// Bilateral support suggests a well-supported book where tighter quoting is safer.
    pub fn has_bilateral_support(&self) -> bool {
        self.hidden_ratio_bid > HIDDEN_SUPPORT_THRESHOLD
            && self.hidden_ratio_ask > HIDDEN_SUPPORT_THRESHOLD
    }
}

impl Default for IcebergDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl BookDynamicsTracker {
    pub fn new() -> Self {
        Self {
            depth_bid_ema: 0.0,
            depth_ask_ema: 0.0,
            bid_change_rate_ema: 0.0,
            ask_change_rate_ema: 0.0,
            depth_change_sq_ema: 0.0,
            mean_depth_ema: 0.0,
            last_bid_depth: 0.0,
            last_ask_depth: 0.0,
            last_update: None,
            update_count: 0,
            bim_shallow_ema: 0.0,
            bim_deep_ema: 0.0,
            prev_bim_shallow: 0.0,
        }
    }

    /// Update with a new L2 depth snapshot.
    ///
    /// `bid_depth` and `ask_depth` are the total quantities resting on each side
    /// (e.g., sum of top-N levels). `now` is the current timestamp.
    pub fn update(&mut self, bid_depth: f64, ask_depth: f64, now: Instant) {
        if let Some(prev_time) = self.last_update {
            let dt_s = now.duration_since(prev_time).as_secs_f64();

            // Time-varying EMA alpha: alpha = 1 - exp(-dt / tau)
            let alpha = 1.0 - (-dt_s / TAU_S).exp();

            // Update depth EMAs
            self.depth_bid_ema += alpha * (bid_depth - self.depth_bid_ema);
            self.depth_ask_ema += alpha * (ask_depth - self.depth_ask_ema);

            // Depth change rates (units per second)
            let safe_dt = dt_s.max(1e-6);
            let bid_change_rate = (bid_depth - self.last_bid_depth) / safe_dt;
            let ask_change_rate = (ask_depth - self.last_ask_depth) / safe_dt;

            // Update change rate EMAs
            self.bid_change_rate_ema += alpha * (bid_change_rate - self.bid_change_rate_ema);
            self.ask_change_rate_ema += alpha * (ask_change_rate - self.ask_change_rate_ema);

            // Track total depth change magnitude squared for persistence
            let total_change_rate = bid_change_rate.abs() + ask_change_rate.abs();
            let change_sq = total_change_rate * total_change_rate;
            self.depth_change_sq_ema += alpha * (change_sq - self.depth_change_sq_ema);

            // Mean total depth
            let total_depth = bid_depth + ask_depth;
            self.mean_depth_ema += alpha * (total_depth - self.mean_depth_ema);
        } else {
            // First update: seed EMAs directly
            self.depth_bid_ema = bid_depth;
            self.depth_ask_ema = ask_depth;
            self.mean_depth_ema = bid_depth + ask_depth;
        }

        self.last_bid_depth = bid_depth;
        self.last_ask_depth = ask_depth;
        self.last_update = Some(now);
        self.update_count += 1;
    }

    /// Thinning direction: which side is losing depth faster.
    ///
    /// Returns a value in [-1, 1]:
    /// - Positive → asks thinning faster (bearish signal)
    /// - Negative → bids thinning faster (bullish signal)
    /// - Near zero → symmetric book
    ///
    /// Returns 0.0 if not warmed up.
    pub fn thinning_direction(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }

        // ask_change_rate negative = asks thinning; bid_change_rate negative = bids thinning
        // We want: positive output when asks thin faster → use (bid_rate - ask_rate)
        // If bids stable (rate~0) and asks dropping (rate<0): bid_rate - ask_rate > 0 → positive (bearish)
        // If asks stable (rate~0) and bids dropping (rate<0): bid_rate - ask_rate < 0 → negative (bullish)
        let diff = self.bid_change_rate_ema - self.ask_change_rate_ema;
        let denom = self.bid_change_rate_ema.abs() + self.ask_change_rate_ema.abs() + EPSILON;
        (diff / denom).clamp(-1.0, 1.0)
    }

    /// Depth persistence: how stable the book is.
    ///
    /// Returns a value in [0, 1]:
    /// - 1.0 → stable book (low variance in depth changes)
    /// - 0.0 → rapidly changing book
    ///
    /// Returns 1.0 (neutral/stable) if not warmed up.
    pub fn depth_persistence(&self) -> f64 {
        if !self.is_warmed_up() {
            return 1.0;
        }

        // persistence = 1 - sqrt(variance) / (mean_depth + epsilon)
        // depth_change_sq_ema is EMA of squared change rates → approximates variance
        let volatility = self.depth_change_sq_ema.sqrt();
        let persistence = 1.0 - volatility / (self.mean_depth_ema + EPSILON);
        persistence.clamp(0.0, 1.0)
    }

    /// Update multi-depth book imbalance from L2 snapshot.
    ///
    /// `bid_depth_shallow` / `ask_depth_shallow` = top 3 levels (~10 bps).
    /// `bid_depth_deep` / `ask_depth_deep` = all levels (~50 bps).
    pub fn update_imbalances(
        &mut self,
        bid_depth_shallow: f64,
        ask_depth_shallow: f64,
        bid_depth_deep: f64,
        ask_depth_deep: f64,
    ) {
        let alpha = 0.1; // Same EWMA alpha as BookStructureEstimator

        let total_shallow = bid_depth_shallow + ask_depth_shallow;
        if total_shallow > EPSILON {
            let bim_shallow = (bid_depth_shallow - ask_depth_shallow) / total_shallow;
            self.prev_bim_shallow = self.bim_shallow_ema;
            self.bim_shallow_ema += alpha * (bim_shallow - self.bim_shallow_ema);
        }

        let total_deep = bid_depth_deep + ask_depth_deep;
        if total_deep > EPSILON {
            let bim_deep = (bid_depth_deep - ask_depth_deep) / total_deep;
            self.bim_deep_ema += alpha * (bim_deep - self.bim_deep_ema);
        }
    }

    /// Rate of change of book imbalance (ΔBIM).
    ///
    /// Rapidly deteriorating bid side (ΔBIM << 0) precedes price drops.
    /// Uses the difference between bid and ask change rate EMAs as a proxy.
    /// Returns 0.0 if not warmed up.
    pub fn book_imbalance_delta(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }
        // Change in shallow BIM: positive = bids strengthening, negative = bids weakening
        (self.bim_shallow_ema - self.prev_bim_shallow).clamp(-1.0, 1.0)
    }

    /// Book Pressure Gradient: BIM(shallow) - BIM(deep).
    ///
    /// - BPG > 0 → support concentrated near touch (fragile, can be swept)
    /// - BPG < 0 → support distributed deep (resilient)
    ///
    /// Returns 0.0 if not warmed up.
    pub fn book_pressure_gradient(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }
        (self.bim_shallow_ema - self.bim_deep_ema).clamp(-1.0, 1.0)
    }

    /// Current shallow book imbalance (top 3 levels).
    pub fn bim_shallow(&self) -> f64 {
        self.bim_shallow_ema
    }

    /// Whether the tracker has received enough updates to produce meaningful signals.
    pub fn is_warmed_up(&self) -> bool {
        self.update_count >= WARMUP_COUNT
    }

    /// Total number of updates received.
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Current EMA of bid-side depth.
    pub fn depth_bid_ema(&self) -> f64 {
        self.depth_bid_ema
    }

    /// Current EMA of ask-side depth.
    pub fn depth_ask_ema(&self) -> f64 {
        self.depth_ask_ema
    }
}

impl Default for BookDynamicsTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Sweep Detection ───

use std::collections::VecDeque;

/// Detection window for sweep events (milliseconds).
const SWEEP_WINDOW_MS: u64 = 3000;

/// A single fill event for sweep tracking.
#[derive(Debug, Clone, Copy)]
struct SweepFill {
    timestamp_ms: u64,
    is_buy: bool,
    size: f64,
    levels_crossed: u32,
}

/// Detects multi-level sweeps: rapid fills that cross multiple price levels.
///
/// Sweeps are a strong indicator of informed trading — a large order that
/// eats through multiple book levels within a short time window indicates
/// aggressive directional intent. This produces a drift observation for
/// the Kalman filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepDetector {
    /// Detection window in ms.
    #[serde(default = "default_sweep_window")]
    tau_sweep_ms: u64,
    /// Recent fills within the detection window.
    #[serde(skip)]
    recent_fills: VecDeque<SweepFill>,
}

fn default_sweep_window() -> u64 {
    SWEEP_WINDOW_MS
}

impl SweepDetector {
    /// Create a new sweep detector with default 3s window.
    pub fn new() -> Self {
        Self {
            tau_sweep_ms: SWEEP_WINDOW_MS,
            recent_fills: VecDeque::with_capacity(64),
        }
    }

    /// Record a fill that may be part of a sweep.
    ///
    /// `levels_crossed` = how many distinct price levels this fill consumed.
    /// For a single-level fill, pass 1. For a sweep across 3 levels, pass 3.
    pub fn record_fill(&mut self, is_buy: bool, size: f64, levels_crossed: u32, now_ms: u64) {
        // Expire old fills
        let cutoff = now_ms.saturating_sub(self.tau_sweep_ms);
        while let Some(f) = self.recent_fills.front() {
            if f.timestamp_ms < cutoff {
                self.recent_fills.pop_front();
            } else {
                break;
            }
        }

        self.recent_fills.push_back(SweepFill {
            timestamp_ms: now_ms,
            is_buy,
            size,
            levels_crossed,
        });
    }

    /// Sweep scores for bid and ask sides.
    ///
    /// Returns `(bid_sweep_score, ask_sweep_score)` where each is the
    /// sum of `size × levels_crossed` for that side within the window.
    pub fn sweep_scores(&self, now_ms: u64) -> (f64, f64) {
        let cutoff = now_ms.saturating_sub(self.tau_sweep_ms);
        let mut bid_score = 0.0;
        let mut ask_score = 0.0;

        for f in &self.recent_fills {
            if f.timestamp_ms < cutoff {
                continue;
            }
            let score = f.size * f.levels_crossed as f64;
            if f.is_buy {
                // Buy sweep consumes ask side → bearish for holders
                ask_score += score;
            } else {
                // Sell sweep consumes bid side → bullish pressure
                bid_score += score;
            }
        }

        (bid_score, ask_score)
    }

    /// Sweep signal as a drift observation (z, R) for the Kalman filter.
    ///
    /// Buy sweeps → bearish (negative z). Sell sweeps → bullish (positive z).
    /// Returns None if no sweeps detected in the window.
    pub fn drift_observation(&self, now_ms: u64) -> Option<(f64, f64)> {
        let (bid_score, ask_score) = self.sweep_scores(now_ms);
        let total = bid_score + ask_score;

        if total < 0.01 {
            return None; // No significant sweep activity
        }

        // Imbalance: positive = more sell sweeps (bullish), negative = more buy sweeps (bearish)
        let imbalance = (bid_score - ask_score) / total;
        let alpha_sweep = 0.4;
        let z = imbalance * alpha_sweep;

        // R inversely proportional to total sweep magnitude.
        // Large sweeps = high confidence → low R.
        let sigma_sweep = 2.0;
        let r = sigma_sweep * sigma_sweep / total.max(0.1);

        Some((z, r))
    }
}

impl Default for SweepDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Helper: advance `now` by `millis` milliseconds.
    fn advance(now: Instant, millis: u64) -> Instant {
        now + Duration::from_millis(millis)
    }

    #[test]
    fn test_new_tracker() {
        let t = BookDynamicsTracker::new();
        assert_eq!(t.update_count(), 0);
        assert!(!t.is_warmed_up());
        assert_eq!(t.thinning_direction(), 0.0);
        assert_eq!(t.depth_persistence(), 1.0);
        assert_eq!(t.depth_bid_ema(), 0.0);
        assert_eq!(t.depth_ask_ema(), 0.0);
    }

    #[test]
    fn test_warmup_required() {
        let mut t = BookDynamicsTracker::new();
        let mut now = Instant::now();

        // Feed 9 updates — should NOT be warmed up
        for i in 0..9 {
            t.update(100.0, 100.0, now);
            now = advance(now, 500);
            assert!(
                !t.is_warmed_up(),
                "should not be warmed up after {} updates",
                i + 1
            );
        }

        // 10th update → warmed up
        t.update(100.0, 100.0, now);
        assert!(t.is_warmed_up());
        assert_eq!(t.update_count(), 10);
    }

    #[test]
    fn test_bid_thinning_bullish() {
        let mut t = BookDynamicsTracker::new();
        let mut now = Instant::now();

        // Warmup with balanced book
        for _ in 0..10 {
            t.update(100.0, 100.0, now);
            now = advance(now, 500);
        }

        // Now drop bid depth while asks stay constant
        for _ in 0..20 {
            t.update(50.0, 100.0, now);
            now = advance(now, 500);
        }

        let dir = t.thinning_direction();
        assert!(
            dir < -0.1,
            "bid thinning should produce negative (bullish) direction, got {dir}"
        );
    }

    #[test]
    fn test_ask_thinning_bearish() {
        let mut t = BookDynamicsTracker::new();
        let mut now = Instant::now();

        // Warmup with balanced book
        for _ in 0..10 {
            t.update(100.0, 100.0, now);
            now = advance(now, 500);
        }

        // Now drop ask depth while bids stay constant
        for _ in 0..20 {
            t.update(100.0, 50.0, now);
            now = advance(now, 500);
        }

        let dir = t.thinning_direction();
        assert!(
            dir > 0.1,
            "ask thinning should produce positive (bearish) direction, got {dir}"
        );
    }

    #[test]
    fn test_symmetric_book_neutral() {
        let mut t = BookDynamicsTracker::new();
        let mut now = Instant::now();

        // Feed identical bid/ask depths throughout
        for _ in 0..20 {
            t.update(100.0, 100.0, now);
            now = advance(now, 500);
        }

        let dir = t.thinning_direction();
        assert!(
            dir.abs() < 0.05,
            "symmetric book should have near-zero thinning direction, got {dir}"
        );
    }

    #[test]
    fn test_stable_book_high_persistence() {
        let mut t = BookDynamicsTracker::new();
        let mut now = Instant::now();

        // Feed constant depth — no changes
        for _ in 0..30 {
            t.update(100.0, 100.0, now);
            now = advance(now, 500);
        }

        let p = t.depth_persistence();
        assert!(p > 0.9, "stable book should have high persistence, got {p}");
    }

    #[test]
    fn test_volatile_book_low_persistence() {
        let mut t = BookDynamicsTracker::new();
        let mut now = Instant::now();

        // Feed rapidly alternating depth
        for i in 0..30 {
            let depth = if i % 2 == 0 { 200.0 } else { 20.0 };
            t.update(depth, depth, now);
            now = advance(now, 500);
        }

        let p = t.depth_persistence();
        let stable_p = {
            let mut stable = BookDynamicsTracker::new();
            let mut stable_now = Instant::now();
            for _ in 0..30 {
                stable.update(100.0, 100.0, stable_now);
                stable_now = advance(stable_now, 500);
            }
            stable.depth_persistence()
        };

        assert!(
            p < stable_p,
            "volatile book ({p}) should have lower persistence than stable ({stable_p})"
        );
    }

    // ─── IcebergDetector tests ───

    #[test]
    fn test_iceberg_new_zeroed() {
        let d = IcebergDetector::new();
        assert_eq!(d.hidden_liquidity_ratio(true), 0.0);
        assert_eq!(d.hidden_liquidity_ratio(false), 0.0);
        assert!(!d.has_bilateral_support());
    }

    #[test]
    fn test_iceberg_hidden_detected_when_fill_exceeds_decrease() {
        let mut d = IcebergDetector::new();

        // Snapshot: ask depth = 100
        d.snapshot_book(100.0, 100.0);

        // Buy fill of 10 units, but ask depth only dropped to 95 (decrease = 5)
        // hidden_frac = (10 - 5) / 10 = 0.5
        d.on_fill_with_book_update(true, 10.0, 95.0);

        let ratio = d.hidden_liquidity_ratio(true);
        // First observation: EWMA = 0.0 * 0.95 + 0.5 * 0.05 = 0.025
        assert!((ratio - 0.025).abs() < 1e-9, "expected 0.025, got {ratio}");
    }

    #[test]
    fn test_iceberg_no_hidden_when_fill_equals_decrease() {
        let mut d = IcebergDetector::new();

        // Snapshot: bid depth = 100
        d.snapshot_book(100.0, 100.0);

        // Sell fill of 10 units, bid depth drops to 90 (decrease = 10, matches fill)
        // hidden_frac = (10 - 10) / 10 = 0.0
        d.on_fill_with_book_update(false, 10.0, 90.0);

        let ratio = d.hidden_liquidity_ratio(false);
        assert!(ratio.abs() < 1e-9, "expected 0.0, got {ratio}");
    }

    #[test]
    fn test_iceberg_ewma_decay_over_time() {
        let mut d = IcebergDetector::new();

        // First: a fill with heavy hidden liquidity
        d.snapshot_book(100.0, 100.0);
        // Buy fill of 10, ask only drops to 100 (no decrease at all)
        // hidden_frac = (10 - 0) / 10 = 1.0
        d.on_fill_with_book_update(true, 10.0, 100.0);

        let ratio_after_hidden = d.hidden_liquidity_ratio(true);
        // EWMA = 0.0 * 0.95 + 1.0 * 0.05 = 0.05
        assert!(
            (ratio_after_hidden - 0.05).abs() < 1e-9,
            "expected 0.05, got {ratio_after_hidden}"
        );

        // Now feed many fills with zero hidden fraction to decay the EWMA
        for _ in 0..20 {
            d.snapshot_book(100.0, 100.0);
            // Buy fill of 10, ask drops to 90 (decrease = 10 = fill, hidden_frac = 0)
            d.on_fill_with_book_update(true, 10.0, 90.0);
        }

        let ratio_decayed = d.hidden_liquidity_ratio(true);
        assert!(
            ratio_decayed < ratio_after_hidden,
            "EWMA should decay: {ratio_decayed} should be < {ratio_after_hidden}"
        );
        assert!(
            ratio_decayed < 0.02,
            "after 20 zero-hidden fills, ratio should be small, got {ratio_decayed}"
        );
    }

    #[test]
    fn test_iceberg_bilateral_support() {
        let mut d = IcebergDetector::new();

        // Repeatedly feed fills with full hidden liquidity on both sides
        // to build up EWMA above the 0.3 threshold
        for _ in 0..100 {
            d.snapshot_book(100.0, 100.0);
            // Buy: ask stays at 100 → hidden_frac = 1.0
            d.on_fill_with_book_update(true, 10.0, 100.0);
            // Sell: bid stays at 100 → hidden_frac = 1.0
            d.snapshot_book(100.0, 100.0);
            d.on_fill_with_book_update(false, 10.0, 100.0);
        }

        assert!(
            d.hidden_liquidity_ratio(true) > HIDDEN_SUPPORT_THRESHOLD,
            "ask hidden ratio should exceed threshold"
        );
        assert!(
            d.hidden_liquidity_ratio(false) > HIDDEN_SUPPORT_THRESHOLD,
            "bid hidden ratio should exceed threshold"
        );
        assert!(
            d.has_bilateral_support(),
            "both sides above threshold should report bilateral support"
        );
    }

    #[test]
    fn test_iceberg_zero_fill_ignored() {
        let mut d = IcebergDetector::new();
        d.snapshot_book(100.0, 100.0);

        // Zero-size fill should be a no-op
        d.on_fill_with_book_update(true, 0.0, 95.0);
        assert_eq!(d.hidden_liquidity_ratio(true), 0.0);

        // Negative-size fill should also be a no-op
        d.on_fill_with_book_update(false, -5.0, 95.0);
        assert_eq!(d.hidden_liquidity_ratio(false), 0.0);
    }

    #[test]
    fn test_iceberg_default_impl() {
        let d = IcebergDetector::default();
        assert_eq!(d.hidden_ratio_bid, 0.0);
        assert_eq!(d.hidden_ratio_ask, 0.0);
        assert_eq!(d.last_bid_depth, 0.0);
        assert_eq!(d.last_ask_depth, 0.0);
    }

    // ─── Phase 7: ΔBIM, BPG, and Sweep tests ───

    #[test]
    fn test_bim_delta_negative_when_bids_collapse() {
        let mut t = BookDynamicsTracker::new();
        let mut now = Instant::now();

        // Warmup with balanced book
        for _ in 0..12 {
            t.update(100.0, 100.0, now);
            t.update_imbalances(50.0, 50.0, 100.0, 100.0);
            now = advance(now, 500);
        }

        // Bids collapse at shallow level
        for _ in 0..5 {
            t.update(40.0, 100.0, now);
            t.update_imbalances(10.0, 50.0, 40.0, 100.0);
            now = advance(now, 500);
        }

        let dbim = t.book_imbalance_delta();
        assert!(
            dbim < 0.0,
            "bid collapse should produce negative ΔBIM, got {dbim}"
        );
    }

    #[test]
    fn test_bpg_fragile_support() {
        let mut t = BookDynamicsTracker::new();
        let mut now = Instant::now();

        // Warmup
        for _ in 0..12 {
            t.update(100.0, 100.0, now);
            now = advance(now, 500);
        }

        // Shallow: heavy bids, light asks. Deep: balanced.
        // BIM_shallow > 0, BIM_deep ≈ 0 → BPG > 0 (fragile support near touch)
        for _ in 0..20 {
            t.update_imbalances(80.0, 20.0, 100.0, 100.0);
            t.update(100.0, 100.0, now);
            now = advance(now, 500);
        }

        let bpg = t.book_pressure_gradient();
        assert!(
            bpg > 0.0,
            "shallow bid-heavy vs balanced deep should produce BPG > 0, got {bpg}"
        );
    }

    #[test]
    fn test_bpg_resilient_support() {
        let mut t = BookDynamicsTracker::new();
        let mut now = Instant::now();

        // Warmup
        for _ in 0..12 {
            t.update(100.0, 100.0, now);
            now = advance(now, 500);
        }

        // Shallow: balanced. Deep: heavy bids.
        // BIM_shallow ≈ 0, BIM_deep > 0 → BPG < 0 (resilient deep support)
        for _ in 0..20 {
            t.update_imbalances(50.0, 50.0, 200.0, 100.0);
            t.update(100.0, 100.0, now);
            now = advance(now, 500);
        }

        let bpg = t.book_pressure_gradient();
        assert!(
            bpg < 0.0,
            "deep bid-heavy vs balanced shallow should produce BPG < 0, got {bpg}"
        );
    }

    #[test]
    fn test_bpg_not_warmed_up_returns_zero() {
        let t = BookDynamicsTracker::new();
        assert_eq!(t.book_pressure_gradient(), 0.0);
        assert_eq!(t.book_imbalance_delta(), 0.0);
    }

    // ─── SweepDetector tests ───

    #[test]
    fn test_sweep_no_fills_no_observation() {
        let det = SweepDetector::new();
        assert!(det.drift_observation(1000).is_none());
    }

    #[test]
    fn test_sweep_single_fill_no_sweep() {
        let mut det = SweepDetector::new();
        det.record_fill(true, 1.0, 1, 1000);

        // Single small fill at 1 level — low score
        let (bid, ask) = det.sweep_scores(1000);
        assert_eq!(bid, 0.0); // buy fills → ask_score
        assert!((ask - 1.0).abs() < 1e-9); // 1.0 * 1 = 1.0
    }

    #[test]
    fn test_sweep_multi_level_detection() {
        let mut det = SweepDetector::new();

        // 3 buy fills crossing multiple levels within the 3s window
        det.record_fill(true, 5.0, 3, 1000); // 5 * 3 = 15
        det.record_fill(true, 3.0, 2, 1500); // 3 * 2 = 6
        det.record_fill(true, 2.0, 2, 2000); // 2 * 2 = 4

        let (bid, ask) = det.sweep_scores(2500);
        assert_eq!(bid, 0.0); // no sell sweeps
        assert!((ask - 25.0).abs() < 1e-9); // 15 + 6 + 4 = 25

        // Should produce bearish drift observation (buy sweeps)
        let obs = det.drift_observation(2500);
        assert!(obs.is_some());
        let (z, _r) = obs.unwrap();
        assert!(
            z < 0.0,
            "buy sweep should produce negative z (bearish), got {z}"
        );
    }

    #[test]
    fn test_sweep_expiry() {
        let mut det = SweepDetector::new();

        // Fill at t=0
        det.record_fill(true, 10.0, 5, 0);

        // At t=2s, still in window
        let (_, ask) = det.sweep_scores(2000);
        assert!(ask > 0.0);

        // At t=4s, expired (window is 3s)
        det.record_fill(false, 0.1, 1, 4000); // trigger cleanup
        let (_, ask) = det.sweep_scores(4000);
        assert!(ask < 1.0, "old sweep should have expired, got ask={ask}");
    }

    #[test]
    fn test_sweep_bidirectional() {
        let mut det = SweepDetector::new();

        // Equal buy and sell sweeps
        det.record_fill(true, 5.0, 2, 1000); // ask_score = 10
        det.record_fill(false, 5.0, 2, 1500); // bid_score = 10

        let obs = det.drift_observation(2000);
        assert!(obs.is_some());
        let (z, _r) = obs.unwrap();
        // Balanced sweeps → z ≈ 0
        assert!(
            z.abs() < 0.1,
            "balanced sweeps should produce near-zero z, got {z}"
        );
    }
}
