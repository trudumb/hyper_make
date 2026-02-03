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
    PendingMarkout,
};

use std::time::Instant;

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
}
