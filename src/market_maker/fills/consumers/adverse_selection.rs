//! Adverse selection consumer.
//!
//! Records fills for AS measurement in the AdverseSelectionEstimator.

use crate::market_maker::adverse_selection::AdverseSelectionEstimator;
use crate::market_maker::fills::{FillConsumer, FillEvent};

/// Consumer that records fills for adverse selection measurement.
///
/// Wraps the existing AdverseSelectionEstimator to integrate with the fill pipeline.
pub struct AdverseSelectionConsumer {
    estimator: AdverseSelectionEstimator,
}

impl AdverseSelectionConsumer {
    /// Create a new AS consumer with default config.
    pub fn new() -> Self {
        Self {
            estimator: AdverseSelectionEstimator::default_config(),
        }
    }

    /// Create with existing estimator.
    pub fn with_estimator(estimator: AdverseSelectionEstimator) -> Self {
        Self { estimator }
    }

    /// Get a reference to the underlying estimator.
    pub fn estimator(&self) -> &AdverseSelectionEstimator {
        &self.estimator
    }

    /// Get a mutable reference to the underlying estimator.
    pub fn estimator_mut(&mut self) -> &mut AdverseSelectionEstimator {
        &mut self.estimator
    }

    /// Update estimator with current mid price.
    ///
    /// This should be called on each mid price update to resolve
    /// pending fills whose measurement horizon has elapsed.
    pub fn update(&mut self, current_mid: f64) {
        self.estimator.update(current_mid);
    }

    /// Get realized adverse selection in basis points.
    pub fn realized_as_bps(&self) -> f64 {
        self.estimator.realized_as_bps()
    }
}

impl Default for AdverseSelectionConsumer {
    fn default() -> Self {
        Self::new()
    }
}

impl FillConsumer for AdverseSelectionConsumer {
    fn on_fill(&mut self, fill: &FillEvent) -> Option<String> {
        self.estimator.record_fill(
            fill.tid,
            fill.size,
            fill.is_buy,
            fill.mid_at_fill,
        );
        None
    }

    fn name(&self) -> &'static str {
        "AdverseSelection"
    }

    /// AS tracking is medium priority
    fn priority(&self) -> u32 {
        50
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fill(tid: u64, size: f64, is_buy: bool, mid: f64) -> FillEvent {
        FillEvent::new(
            tid,
            100,
            size,
            mid, // fill at mid for simplicity
            is_buy,
            mid,
            None,
            "BTC".to_string(),
        )
    }

    #[test]
    fn test_records_fill() {
        let mut consumer = AdverseSelectionConsumer::new();
        let result = consumer.on_fill(&make_fill(1, 1.0, true, 50000.0));
        assert!(result.is_none());
    }

    #[test]
    fn test_priority() {
        let consumer = AdverseSelectionConsumer::new();
        assert_eq!(consumer.priority(), 50);
    }
}
