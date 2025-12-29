//! Estimator consumer.
//!
//! Records fills to the ParameterEstimator for kappa (fill rate) estimation.

use crate::market_maker::fills::{FillConsumer, FillEvent};

/// Callback type for notifying the estimator of fills.
///
/// Parameters: (timestamp_ms, placement_price, fill_price, size, is_buy)
pub(crate) type EstimatorCallback = Box<dyn FnMut(u64, f64, f64, f64, bool) + Send + Sync>;

/// Consumer that notifies the ParameterEstimator of fills.
///
/// The estimator uses fill data to update kappa (order fill rate decay)
/// estimates. This consumer uses a callback pattern to avoid circular
/// dependencies with the estimator module.
pub struct EstimatorConsumer {
    callback: Option<EstimatorCallback>,
    fills_sent: usize,
}

impl EstimatorConsumer {
    /// Create a new estimator consumer with no callback.
    ///
    /// Use `set_callback` to wire up the actual estimator.
    pub fn new() -> Self {
        Self {
            callback: None,
            fills_sent: 0,
        }
    }

    /// Create with callback.
    pub fn with_callback(callback: EstimatorCallback) -> Self {
        Self {
            callback: Some(callback),
            fills_sent: 0,
        }
    }

    /// Set the callback function.
    ///
    /// This is typically called during setup to wire the consumer
    /// to the actual ParameterEstimator.
    pub fn set_callback(&mut self, callback: EstimatorCallback) {
        self.callback = Some(callback);
    }

    /// Get number of fills sent to estimator.
    pub fn fills_sent(&self) -> usize {
        self.fills_sent
    }
}

impl Default for EstimatorConsumer {
    fn default() -> Self {
        Self::new()
    }
}

impl FillConsumer for EstimatorConsumer {
    fn on_fill(&mut self, fill: &FillEvent) -> Option<String> {
        // Only process if we have a placement price (tracked order)
        let placement_price = match fill.placement_price {
            Some(p) => p,
            None => return None, // Skip untracked orders
        };

        if let Some(ref mut callback) = self.callback {
            callback(
                fill.timestamp_ms(),
                placement_price,
                fill.price,
                fill.size,
                fill.is_buy,
            );
            self.fills_sent += 1;
        }

        None
    }

    fn name(&self) -> &'static str {
        "Estimator"
    }

    /// Estimator update is medium priority (after position, before metrics)
    fn priority(&self) -> u32 {
        30
    }

    /// Only enabled if callback is set.
    fn is_enabled(&self) -> bool {
        self.callback.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn make_fill(tid: u64, price: f64, placement: Option<f64>) -> FillEvent {
        FillEvent::new(
            tid,
            100,
            1.0,
            price,
            true,
            price,
            placement,
            "BTC".to_string(),
        )
    }

    #[test]
    fn test_disabled_without_callback() {
        let consumer = EstimatorConsumer::new();
        assert!(!consumer.is_enabled());
    }

    #[test]
    fn test_enabled_with_callback() {
        let consumer = EstimatorConsumer::with_callback(Box::new(|_, _, _, _, _| {}));
        assert!(consumer.is_enabled());
    }

    #[test]
    fn test_skips_untracked_fills() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let mut consumer = EstimatorConsumer::with_callback(Box::new(move |_, _, _, _, _| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
        }));

        // Fill without placement price should be skipped
        consumer.on_fill(&make_fill(1, 50000.0, None));
        assert_eq!(call_count.load(Ordering::SeqCst), 0);

        // Fill with placement price should be processed
        consumer.on_fill(&make_fill(2, 50000.0, Some(50005.0)));
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_priority() {
        let consumer = EstimatorConsumer::new();
        assert_eq!(consumer.priority(), 30);
    }
}
