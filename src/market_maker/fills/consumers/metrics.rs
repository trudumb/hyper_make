//! Metrics consumer.
//!
//! Records fill metrics to Prometheus and optional external recorder.

use crate::market_maker::config::MarketMakerMetricsRecorder;
use crate::market_maker::fills::{FillConsumer, FillEvent};
use crate::market_maker::metrics::PrometheusMetrics;
use std::sync::Arc;

/// Consumer that records fill metrics.
///
/// Updates both Prometheus metrics and an optional external metrics recorder.
pub struct MetricsConsumer {
    prometheus: PrometheusMetrics,
    external: Option<Arc<dyn MarketMakerMetricsRecorder>>,
}

impl MetricsConsumer {
    /// Create a new metrics consumer.
    pub fn new(prometheus: PrometheusMetrics) -> Self {
        Self {
            prometheus,
            external: None,
        }
    }

    /// Create with external metrics recorder.
    pub fn with_external(
        prometheus: PrometheusMetrics,
        external: Arc<dyn MarketMakerMetricsRecorder>,
    ) -> Self {
        Self {
            prometheus,
            external: Some(external),
        }
    }

    /// Get a reference to Prometheus metrics.
    pub fn prometheus(&self) -> &PrometheusMetrics {
        &self.prometheus
    }

    /// Get a mutable reference to Prometheus metrics.
    pub fn prometheus_mut(&mut self) -> &mut PrometheusMetrics {
        &mut self.prometheus
    }
}

impl FillConsumer for MetricsConsumer {
    fn on_fill(&mut self, fill: &FillEvent) -> Option<String> {
        // Update Prometheus
        self.prometheus.record_fill(fill.size, fill.is_buy);

        // Update external recorder if present
        if let Some(ref external) = self.external {
            external.record_fill(fill.size, fill.is_buy);
        }

        None
    }

    fn name(&self) -> &'static str {
        "Metrics"
    }

    /// Metrics are low priority (can be last)
    fn priority(&self) -> u32 {
        200
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fill(tid: u64, size: f64, is_buy: bool) -> FillEvent {
        FillEvent::new(
            tid,
            100,
            size,
            50000.0,
            is_buy,
            50000.0,
            None,
            "BTC".to_string(),
        )
    }

    #[test]
    fn test_priority() {
        let consumer = MetricsConsumer::new(PrometheusMetrics::new());
        assert_eq!(consumer.priority(), 200);
    }

    #[test]
    fn test_records_fill() {
        let mut consumer = MetricsConsumer::new(PrometheusMetrics::new());
        let result = consumer.on_fill(&make_fill(1, 1.0, true));
        assert!(result.is_none());
    }
}
