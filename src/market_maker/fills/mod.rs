//! Unified fill processing pipeline.
//!
//! This module provides a single source of truth for fill deduplication and
//! the FillProcessor for orchestrating fill handling across modules.
//!
//! # Architecture
//!
//! ```text
//! WebSocket: UserFills
//!          |
//!          v
//! +------------------+
//! |   FillProcessor  |
//! |  (single dedup)  |
//! +------------------+
//!          |
//!     Updates all modules via FillState bundle:
//!     +----+----+----+----+----+----+
//!     |    |    |    |    |    |    |
//!     v    v    v    v    v    v    v
//! Position Orders AS   Queue PnL  Estimator Metrics
//! Tracker  Manager     Tracker    Tracker
//! ```
//!
//! # Key Components
//!
//! - **FillEvent**: Unified fill data structure with helper methods
//! - **FillProcessor**: Orchestrates fill handling with centralized deduplication
//! - **FillState**: Bundle of mutable references for processing
//! - **FillDeduplicator**: Tracks seen trade IDs to prevent double-processing
//!
//! # Benefits
//!
//! - **Single deduplication**: No duplicate tracking across modules
//! - **Clear data flow**: All modules updated through FillState bundle
//! - **Testable**: Processor can be tested in isolation

mod consumer;
mod dedup;
mod processor;

pub use consumer::{FillConsumer, FillConsumerBox};
pub use dedup::FillDeduplicator;
pub use processor::{
    FillProcessingResult, FillProcessor, FillSignalSnapshot, FillSignalStore, FillState,
    PendingMarkout, TombstoneOrder,
};

use std::time::Instant;

/// Pending fill outcome for adverse selection markout tracking.
///
/// After each fill, we record the mid price and wait 5 seconds.
/// Then we check whether the mid moved against us (adverse selection)
/// and feed the outcome to the pre-fill classifier and model gating.
#[derive(Debug, Clone)]
pub struct PendingFillOutcome {
    /// Fill timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
    /// Fill price
    pub fill_price: f64,
    /// Whether this was a buy fill (our bid was filled)
    pub is_buy: bool,
    /// Mid price at the time of fill
    pub mid_at_fill: f64,
}

/// A unified fill event containing all data needed by modules.
///
/// This is the single source of truth for fill data, passed to the FillProcessor.
#[derive(Debug, Clone)]
pub struct FillEvent {
    /// Trade ID (unique identifier for deduplication)
    pub tid: u64,
    /// Order ID
    pub oid: u64,
    /// Client Order ID (for deterministic fill matching - Phase 1 Fix)
    /// This is the UUID we generated before order placement.
    pub cloid: Option<String>,
    /// Fill size (always positive)
    pub size: f64,
    /// Fill price
    pub price: f64,
    /// Was this a buy (we got filled on our bid)
    pub is_buy: bool,
    /// Mid price at fill time (for spread capture calculation)
    pub mid_at_fill: f64,
    /// Placement price (where we placed the order, for depth calculation)
    /// None if order was untracked
    pub placement_price: Option<f64>,
    /// Fill timestamp
    pub timestamp: Instant,
    /// Asset being traded
    pub asset: String,
}

impl FillEvent {
    /// Create a new fill event (legacy, without CLOID).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tid: u64,
        oid: u64,
        size: f64,
        price: f64,
        is_buy: bool,
        mid_at_fill: f64,
        placement_price: Option<f64>,
        asset: String,
    ) -> Self {
        Self {
            tid,
            oid,
            cloid: None,
            size,
            price,
            is_buy,
            mid_at_fill,
            placement_price,
            timestamp: Instant::now(),
            asset,
        }
    }

    /// Create a new fill event with CLOID for deterministic tracking.
    #[allow(clippy::too_many_arguments)]
    pub fn with_cloid(
        tid: u64,
        oid: u64,
        cloid: Option<String>,
        size: f64,
        price: f64,
        is_buy: bool,
        mid_at_fill: f64,
        placement_price: Option<f64>,
        asset: String,
    ) -> Self {
        Self {
            tid,
            oid,
            cloid,
            size,
            price,
            is_buy,
            mid_at_fill,
            placement_price,
            timestamp: Instant::now(),
            asset,
        }
    }

    /// Calculate depth from mid in basis points.
    ///
    /// Uses placement price if available, otherwise fill price.
    pub fn depth_bps(&self) -> f64 {
        let reference_price = self.placement_price.unwrap_or(self.price);
        if self.mid_at_fill > 0.0 {
            ((reference_price - self.mid_at_fill).abs() / self.mid_at_fill) * 10_000.0
        } else {
            0.0
        }
    }

    /// Calculate spread capture in dollars.
    ///
    /// Positive = captured spread (filled better than mid).
    /// Negative = gave up spread (filled worse than mid).
    pub fn spread_capture(&self) -> f64 {
        if self.is_buy {
            // Buy: mid - price (positive if we bought below mid)
            (self.mid_at_fill - self.price) * self.size
        } else {
            // Sell: price - mid (positive if we sold above mid)
            (self.price - self.mid_at_fill) * self.size
        }
    }

    /// Get signed size (positive for buy, negative for sell).
    pub fn signed_size(&self) -> f64 {
        if self.is_buy {
            self.size
        } else {
            -self.size
        }
    }

    /// Get timestamp in milliseconds since epoch.
    pub fn timestamp_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// Result of processing a fill through the pipeline.
#[derive(Debug, Clone, Default)]
pub struct FillResult {
    /// Was this a new fill (not a duplicate)?
    pub is_new: bool,
    /// Number of consumers that processed the fill
    pub consumers_notified: usize,
    /// Any errors from consumers (consumer name -> error message)
    pub errors: Vec<(String, String)>,
}

impl FillResult {
    /// Create a result for a duplicate fill.
    pub fn duplicate() -> Self {
        Self {
            is_new: false,
            consumers_notified: 0,
            errors: Vec::new(),
        }
    }

    /// Create a result for a new fill.
    pub fn new_fill(consumers_notified: usize) -> Self {
        Self {
            is_new: true,
            consumers_notified,
            errors: Vec::new(),
        }
    }

    /// Add an error from a consumer.
    pub fn add_error(&mut self, consumer: &str, error: String) {
        self.errors.push((consumer.to_string(), error));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_event_depth_bps() {
        let fill = FillEvent::new(
            1,
            100,
            1.0,
            50000.0,       // fill price
            true,          // is_buy
            50010.0,       // mid
            Some(50005.0), // placement price
            "BTC".to_string(),
        );

        // Depth should be based on placement price: |50005 - 50010| / 50010 * 10000 ≈ 1.0 bps
        let depth = fill.depth_bps();
        assert!((depth - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_fill_event_spread_capture_buy() {
        let fill = FillEvent::new(
            1,
            100,
            1.0,
            50000.0, // fill price
            true,    // is_buy
            50010.0, // mid
            None,
            "BTC".to_string(),
        );

        // Buy at 50000, mid was 50010 = captured $10
        assert!((fill.spread_capture() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_event_spread_capture_sell() {
        let fill = FillEvent::new(
            1,
            100,
            1.0,
            50010.0, // fill price
            false,   // is_sell
            50000.0, // mid
            None,
            "BTC".to_string(),
        );

        // Sell at 50010, mid was 50000 = captured $10
        assert!((fill.spread_capture() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fill_event_signed_size() {
        let buy_fill = FillEvent::new(1, 100, 1.0, 50000.0, true, 50000.0, None, "BTC".to_string());
        assert!((buy_fill.signed_size() - 1.0).abs() < f64::EPSILON);

        let sell_fill = FillEvent::new(
            2,
            101,
            1.0,
            50000.0,
            false,
            50000.0,
            None,
            "BTC".to_string(),
        );
        assert!((sell_fill.signed_size() - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fill_result_duplicate() {
        let result = FillResult::duplicate();
        assert!(!result.is_new);
        assert_eq!(result.consumers_notified, 0);
    }

    #[test]
    fn test_fill_result_new() {
        let result = FillResult::new_fill(5);
        assert!(result.is_new);
        assert_eq!(result.consumers_notified, 5);
    }

    #[test]
    fn test_pending_fill_outcome_creation() {
        let outcome = PendingFillOutcome {
            timestamp_ms: 1_700_000_000_000,
            fill_price: 50_000.0,
            is_buy: true,
            mid_at_fill: 50_005.0,
        };
        assert!(outcome.is_buy);
        assert!((outcome.fill_price - 50_000.0).abs() < f64::EPSILON);
        assert!((outcome.mid_at_fill - 50_005.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pending_fill_outcome_queue_drain() {
        use std::collections::VecDeque;

        let mut queue: VecDeque<PendingFillOutcome> = VecDeque::new();
        let base_ms = 1_700_000_000_000u64;

        // Push 3 outcomes: t=0, t=3s, t=6s
        for i in 0..3 {
            queue.push_back(PendingFillOutcome {
                timestamp_ms: base_ms + i * 3_000,
                fill_price: 50_000.0,
                is_buy: i % 2 == 0,
                mid_at_fill: 50_000.0,
            });
        }
        assert_eq!(queue.len(), 3);

        // At t=4.9s, only the first (t=0) should be expired (>= 5s not met)
        let now_ms = base_ms + 4_999;
        let expired: Vec<_> = std::iter::from_fn(|| {
            if let Some(front) = queue.front() {
                if now_ms.saturating_sub(front.timestamp_ms) >= 5_000 {
                    return queue.pop_front();
                }
            }
            None
        })
        .collect();
        assert_eq!(expired.len(), 0);
        assert_eq!(queue.len(), 3);

        // At t=5s, the first outcome (t=0) should expire
        let now_ms = base_ms + 5_000;
        let expired: Vec<_> = std::iter::from_fn(|| {
            if let Some(front) = queue.front() {
                if now_ms.saturating_sub(front.timestamp_ms) >= 5_000 {
                    return queue.pop_front();
                }
            }
            None
        })
        .collect();
        assert_eq!(expired.len(), 1);
        assert!(expired[0].is_buy);
        assert_eq!(queue.len(), 2);

        // At t=11s, both remaining (t=3s and t=6s) should expire
        let now_ms = base_ms + 11_000;
        let expired: Vec<_> = std::iter::from_fn(|| {
            if let Some(front) = queue.front() {
                if now_ms.saturating_sub(front.timestamp_ms) >= 5_000 {
                    return queue.pop_front();
                }
            }
            None
        })
        .collect();
        assert_eq!(expired.len(), 2);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_pending_fill_outcome_adverse_classification() {
        let mid_at_fill = 50_000.0;

        // Buy fill: mid dropped 2 bps -> adverse
        let mid_now_buy_adverse = mid_at_fill * (1.0 - 2.0 / 10_000.0); // 49999.0
        let mid_change_bps = ((mid_now_buy_adverse - mid_at_fill) / mid_at_fill) * 10_000.0;
        let was_adverse_buy = mid_change_bps < -1.0; // Mid dropped > 1 bps
        assert!(was_adverse_buy);

        // Buy fill: mid rose 2 bps -> not adverse
        let mid_now_buy_good = mid_at_fill * (1.0 + 2.0 / 10_000.0);
        let mid_change_bps = ((mid_now_buy_good - mid_at_fill) / mid_at_fill) * 10_000.0;
        let was_adverse_buy = mid_change_bps < -1.0;
        assert!(!was_adverse_buy);

        // Sell fill: mid rose 2 bps -> adverse
        let mid_now_sell_adverse = mid_at_fill * (1.0 + 2.0 / 10_000.0);
        let mid_change_bps = ((mid_now_sell_adverse - mid_at_fill) / mid_at_fill) * 10_000.0;
        let was_adverse_sell = mid_change_bps > 1.0; // Mid rose > 1 bps
        assert!(was_adverse_sell);

        // Sell fill: mid dropped 2 bps -> not adverse
        let mid_now_sell_good = mid_at_fill * (1.0 - 2.0 / 10_000.0);
        let mid_change_bps = ((mid_now_sell_good - mid_at_fill) / mid_at_fill) * 10_000.0;
        let was_adverse_sell = mid_change_bps > 1.0;
        assert!(!was_adverse_sell);
    }
}
