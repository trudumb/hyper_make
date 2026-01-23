//! Edge maintenance and A/B testing module.
//!
//! This module provides infrastructure for maintaining trading edge over time:
//!
//! ## Signal Health Monitoring
//!
//! Tracks the health of trading signals using mutual information (MI) metrics.
//! Signals degrade over time as arbitrageurs compete away edge. This module
//! provides early warning when signals are losing predictive power.
//!
//! ```ignore
//! let mut monitor = SignalHealthMonitor::new(0.5, 0.75);
//! monitor.register_signal(EdgeSignalKind::LeadLag, "BTC Lead-Lag", 0.8);
//!
//! // Update with new MI measurement
//! monitor.update_signal(EdgeSignalKind::LeadLag, 0.6, now_ms);
//!
//! // Check health
//! if !monitor.all_healthy() {
//!     for signal in monitor.stale_signals() {
//!         warn!("Signal {:?} is stale - consider removing", signal);
//!     }
//! }
//! ```
//!
//! ## A/B Testing
//!
//! Enables rigorous comparison of control solvers using statistical tests.
//! Supports:
//! - Deterministic random allocation (reproducible)
//! - PnL, fill rate, and adverse selection tracking
//! - Z-test for statistical significance
//! - Auto-promotion when treatment wins
//!
//! ```ignore
//! let config = ABTestConfig {
//!     name: "glft_vs_custom".to_string(),
//!     treatment_allocation: 0.1,
//!     min_samples: 100,
//!     auto_promote: true,
//!     ..Default::default()
//! };
//!
//! let test = ABTest::new(config, now_ms);
//!
//! // For each quote cycle
//! match test.allocate() {
//!     ABVariant::Control => use_control_solver(),
//!     ABVariant::Treatment => use_treatment_solver(),
//! }
//!
//! // After trade completes
//! test.record_control_trade(pnl_bps, filled, was_adverse);
//! ```
//!
//! ## Key Insight
//!
//! All alpha decays. The competitive advantage in market making comes from:
//! 1. Detecting signal decay early (signal health)
//! 2. Rigorously testing improvements (A/B testing)
//! 3. Continuously iterating on the edge source
//!
//! This module provides the infrastructure for (1) and (2).

pub mod ab_testing;
pub mod signal_health;

// Re-export key types
pub use ab_testing::{ABMetrics, ABTest, ABTestConfig, ABTestManager, ABVariant};
pub use signal_health::{
    EdgeSignalKind, HealthStatus, SignalHealth, SignalHealthMonitor, SignalHealthSummary,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify key types are accessible
        let _monitor = SignalHealthMonitor::new(0.5, 0.75);
        let _manager = ABTestManager::new();
        let _config = ABTestConfig::default();
    }

    #[test]
    fn test_integration_signal_with_ab() {
        // Test that signal health can inform A/B test decisions
        let mut monitor = SignalHealthMonitor::new(0.5, 0.75);
        monitor.register_signal(EdgeSignalKind::LeadLag, "Lead-Lag".to_string(), 1.0);

        let manager = ABTestManager::new();
        let config = ABTestConfig {
            name: "new_lead_lag_model".to_string(),
            treatment_allocation: 0.1,
            min_samples: 50,
            ..Default::default()
        };

        manager.create_test(config, 0).unwrap();

        // If signal is degraded, we might want to test a new model
        monitor.update_signal(EdgeSignalKind::LeadLag, 0.6, 1000);

        if monitor
            .get_health(EdgeSignalKind::LeadLag)
            .unwrap()
            .is_degraded()
        {
            // Could trigger A/B test for new model here
            let test = manager.get_test("new_lead_lag_model").unwrap();
            assert!(!test.is_concluded());
        }
    }
}
