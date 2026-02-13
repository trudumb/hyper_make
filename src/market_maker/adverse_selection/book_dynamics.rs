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
        assert!(
            p > 0.9,
            "stable book should have high persistence, got {p}"
        );
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
        assert!(
            (ratio - 0.025).abs() < 1e-9,
            "expected 0.025, got {ratio}"
        );
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
        assert!(
            ratio.abs() < 1e-9,
            "expected 0.0, got {ratio}"
        );
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
}
