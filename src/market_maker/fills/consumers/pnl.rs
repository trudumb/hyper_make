//! P&L tracking consumer.
//!
//! Tracks realized/unrealized P&L and spread capture from fills.

use crate::market_maker::fills::{FillConsumer, FillEvent};
use crate::market_maker::pnl::{PnLConfig, PnLTracker};

/// Consumer that updates P&L tracking.
///
/// Wraps the existing PnLTracker to integrate with the fill pipeline.
pub struct PnLConsumer {
    tracker: PnLTracker,
}

impl PnLConsumer {
    /// Create a new P&L consumer with default config.
    pub fn new() -> Self {
        Self {
            tracker: PnLTracker::new(PnLConfig::default()),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: PnLConfig) -> Self {
        Self {
            tracker: PnLTracker::new(config),
        }
    }

    /// Get a reference to the underlying tracker.
    pub fn tracker(&self) -> &PnLTracker {
        &self.tracker
    }

    /// Get a mutable reference to the underlying tracker.
    pub fn tracker_mut(&mut self) -> &mut PnLTracker {
        &mut self.tracker
    }
}

impl Default for PnLConsumer {
    fn default() -> Self {
        Self::new()
    }
}

impl FillConsumer for PnLConsumer {
    fn on_fill(&mut self, fill: &FillEvent) -> Option<String> {
        self.tracker.record_fill(
            fill.tid,
            fill.price,
            fill.size,
            fill.is_buy,
            fill.mid_at_fill,
        );
        None
    }

    fn name(&self) -> &'static str {
        "PnL"
    }

    /// P&L tracking is second priority (after position)
    fn priority(&self) -> u32 {
        10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fill(tid: u64, price: f64, size: f64, is_buy: bool, mid: f64) -> FillEvent {
        FillEvent::new(
            tid,
            100,
            size,
            price,
            is_buy,
            mid,
            None,
            "BTC".to_string(),
        )
    }

    #[test]
    fn test_records_fill() {
        let mut consumer = PnLConsumer::new();
        consumer.on_fill(&make_fill(1, 50000.0, 1.0, true, 50010.0));
        assert_eq!(consumer.tracker().fill_count(), 1);
    }

    #[test]
    fn test_spread_capture() {
        let mut consumer = PnLConsumer::new();
        // Buy at 50000, mid was 50010 = captured $10
        consumer.on_fill(&make_fill(1, 50000.0, 1.0, true, 50010.0));

        let summary = consumer.tracker().summary(50010.0);
        assert!(summary.spread_capture > 0.0);
    }

    #[test]
    fn test_priority() {
        let consumer = PnLConsumer::new();
        assert_eq!(consumer.priority(), 10);
    }
}
