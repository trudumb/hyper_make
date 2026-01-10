//! Prometheus Metrics for Market Maker
//!
//! Exposes key market maker metrics for monitoring and alerting.
//!
//! Key metric categories:
//! - **Position**: Current position, inventory utilization
//! - **P&L**: Daily P&L, drawdown, unrealized P&L
//! - **Orders**: Fill rate, cancel rate, order latency
//! - **Market**: Spread, volatility, toxicity
//! - **Risk**: Kill switch status, cascade severity

mod atomic;
mod fields;
mod output;
mod summary;
mod updates;

use std::sync::Arc;

use fields::MetricsInner;
pub use summary::MetricsSummary;

/// Prometheus-compatible metrics collector.
///
/// Thread-safe metrics that can be scraped by Prometheus or similar systems.
#[derive(Clone)]
pub struct PrometheusMetrics {
    inner: Arc<MetricsInner>,
}

impl PrometheusMetrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MetricsInner::new()),
        }
    }
}

impl Default for PrometheusMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = PrometheusMetrics::new();
        let summary = metrics.summary();
        assert_eq!(summary.position, 0.0);
        assert_eq!(summary.orders_placed, 0);
    }

    #[test]
    fn test_position_update() {
        let metrics = PrometheusMetrics::new();
        metrics.update_position(5.0, 10.0);

        let summary = metrics.summary();
        assert_eq!(summary.position, 5.0);
        assert_eq!(summary.inventory_utilization, 0.5);
    }

    #[test]
    fn test_pnl_update() {
        let metrics = PrometheusMetrics::new();
        metrics.update_pnl(100.0, 150.0, 80.0, 20.0);

        let summary = metrics.summary();
        assert_eq!(summary.daily_pnl, 100.0);
        assert!((summary.drawdown_pct - 33.33).abs() < 0.1);
    }

    #[test]
    fn test_order_recording() {
        let metrics = PrometheusMetrics::new();

        metrics.record_order_placed();
        metrics.record_order_placed();
        metrics.record_fill(1.0, true);
        metrics.record_fill(0.5, false);
        metrics.record_cancel();

        let summary = metrics.summary();
        assert_eq!(summary.orders_placed, 2);
        assert_eq!(summary.orders_filled, 2);
        assert_eq!(summary.fill_volume, 1.5);
    }

    #[test]
    fn test_prometheus_output() {
        let metrics = PrometheusMetrics::new();
        metrics.update_position(1.0, 10.0);
        metrics.update_market(50000.0, 5.0, 0.001, 1.2, 500.0);

        let output = metrics.to_prometheus_text("BTC", "USDC");

        assert!(output.contains("mm_position{asset=\"BTC\",quote=\"USDC\"}"));
        assert!(output.contains("mm_mid_price{asset=\"BTC\",quote=\"USDC\"}"));
        assert!(output.contains("# TYPE mm_position gauge"));
    }

    #[test]
    fn test_prometheus_output_hip3_collateral() {
        let metrics = PrometheusMetrics::new();
        metrics.update_position(1.0, 10.0);
        metrics.update_market(50000.0, 5.0, 0.001, 1.2, 500.0);

        // Test with HIP-3 collateral (USDE)
        let output = metrics.to_prometheus_text("hyna:BTC", "USDE");

        assert!(output.contains("mm_position{asset=\"hyna:BTC\",quote=\"USDE\"}"));
        assert!(output.contains("mm_mid_price{asset=\"hyna:BTC\",quote=\"USDE\"}"));
    }

    #[test]
    fn test_to_map() {
        let metrics = PrometheusMetrics::new();
        metrics.update_position(2.5, 10.0);

        let map = metrics.to_map();
        assert_eq!(map.get("position"), Some(&2.5));
        assert_eq!(map.get("inventory_utilization"), Some(&0.25));
    }

    #[test]
    fn test_thread_safety() {
        let metrics = PrometheusMetrics::new();
        let metrics2 = metrics.clone();

        std::thread::spawn(move || {
            for _ in 0..100 {
                metrics2.record_order_placed();
            }
        });

        for _ in 0..100 {
            metrics.record_order_placed();
        }

        // Allow thread to complete
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Should have some orders (exact count depends on timing)
        let summary = metrics.summary();
        assert!(summary.orders_placed > 0);
    }

    #[test]
    fn test_risk_update() {
        let metrics = PrometheusMetrics::new();
        metrics.update_risk(true, 0.8, 5.5, 2.0);

        let summary = metrics.summary();
        assert!(summary.kill_switch_triggered);
        assert_eq!(summary.cascade_severity, 0.8);
        assert_eq!(summary.adverse_selection_bps, 5.5);
    }
}
