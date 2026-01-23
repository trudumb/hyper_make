//! Signal health monitoring for edge maintenance.
//!
//! Tracks the health of trading signals using mutual information (MI) metrics.
//! Signals are considered:
//! - **Healthy**: MI >= 75% of baseline
//! - **Degraded**: MI between 50-75% of baseline
//! - **Stale**: MI < 50% of baseline
//!
//! ## Key Insight
//!
//! All alpha decays. This module provides early warning when signals
//! are losing predictive power, allowing proactive parameter adjustment
//! or signal retirement before edge turns negative.

use std::collections::HashMap;

/// Health status of a single trading signal.
#[derive(Debug, Clone)]
pub struct SignalHealth {
    /// Signal name for logging
    pub name: String,
    /// Current mutual information value
    pub mi_current: f64,
    /// Baseline MI when signal was added
    pub mi_baseline: f64,
    /// Historical MI values for trend analysis
    pub mi_history: Vec<f64>,
    /// Estimated decay rate (% per period)
    pub decay_rate: f64,
    /// Last update timestamp (epoch ms)
    pub last_update: u64,
}

impl SignalHealth {
    /// Create a new signal health tracker.
    ///
    /// # Arguments
    /// * `name` - Signal identifier for logging
    /// * `baseline_mi` - Initial mutual information value (established when signal added)
    pub fn new(name: String, baseline_mi: f64) -> Self {
        Self {
            name,
            mi_current: baseline_mi,
            mi_baseline: baseline_mi,
            mi_history: vec![baseline_mi],
            decay_rate: 0.0,
            last_update: 0,
        }
    }

    /// Update signal health with new MI measurement.
    ///
    /// # Arguments
    /// * `mi` - New mutual information value
    /// * `timestamp` - Epoch milliseconds
    pub fn update(&mut self, mi: f64, timestamp: u64) {
        self.mi_current = mi;
        self.mi_history.push(mi);
        self.last_update = timestamp;

        // Keep history bounded (last 100 periods)
        if self.mi_history.len() > 100 {
            self.mi_history.remove(0);
        }

        // Recalculate decay rate
        self.decay_rate = self.estimated_decay_rate();
    }

    /// Check if signal is stale (MI < 50% of baseline).
    ///
    /// Stale signals should be removed or significantly downweighted.
    pub fn is_stale(&self) -> bool {
        self.relative_strength() < 0.5
    }

    /// Check if signal is degraded (MI < 75% of baseline).
    ///
    /// Degraded signals need attention but may still provide value.
    pub fn is_degraded(&self) -> bool {
        self.relative_strength() < 0.75
    }

    /// Calculate relative strength (mi_current / mi_baseline).
    ///
    /// Returns 1.0 if signal is as strong as baseline, 0.0 if no information.
    pub fn relative_strength(&self) -> f64 {
        if self.mi_baseline > 1e-10 {
            (self.mi_current / self.mi_baseline).clamp(0.0, 2.0)
        } else {
            0.0
        }
    }

    /// Estimate decay rate using linear regression on history.
    ///
    /// Returns the estimated percentage decline per period.
    /// Positive values indicate decay, negative values indicate improvement.
    pub fn estimated_decay_rate(&self) -> f64 {
        if self.mi_history.len() < 3 {
            return 0.0;
        }

        // Simple linear regression: y = mx + b
        let n = self.mi_history.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &mi) in self.mi_history.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += mi;
            sum_xy += x * mi;
            sum_xx += x * x;
        }

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        // Slope m
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;

        // Convert to percentage decay per period
        // Negative slope means decay
        if self.mi_baseline > 1e-10 {
            -slope / self.mi_baseline * 100.0
        } else {
            0.0
        }
    }
}

/// Type of trading signal for categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeSignalKind {
    /// Cross-exchange lead-lag signal (Binance -> Hyperliquid)
    LeadLag,
    /// Funding rate prediction
    FundingPrediction,
    /// Market regime detection (HMM)
    RegimeDetection,
    /// Adverse selection prediction
    AdverseSelection,
    /// Fill probability estimation
    FillProbability,
    /// Custom signal with numeric ID
    Custom(u32),
}

impl std::fmt::Display for EdgeSignalKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeSignalKind::LeadLag => write!(f, "LeadLag"),
            EdgeSignalKind::FundingPrediction => write!(f, "FundingPrediction"),
            EdgeSignalKind::RegimeDetection => write!(f, "RegimeDetection"),
            EdgeSignalKind::AdverseSelection => write!(f, "AdverseSelection"),
            EdgeSignalKind::FillProbability => write!(f, "FillProbability"),
            EdgeSignalKind::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// Monitor for tracking health of multiple signals.
///
/// Aggregates signal health across the system and provides
/// summary statistics for operational monitoring.
#[derive(Debug)]
pub struct SignalHealthMonitor {
    /// Map of signal type to health tracker
    signals: HashMap<EdgeSignalKind, SignalHealth>,
    /// Threshold for stale classification (default 0.5)
    stale_threshold: f64,
    /// Threshold for degraded classification (default 0.75)
    degraded_threshold: f64,
}

impl SignalHealthMonitor {
    /// Create a new signal health monitor.
    ///
    /// # Arguments
    /// * `stale_threshold` - Relative strength below which signal is stale (default 0.5)
    /// * `degraded_threshold` - Relative strength below which signal is degraded (default 0.75)
    pub fn new(stale_threshold: f64, degraded_threshold: f64) -> Self {
        Self {
            signals: HashMap::new(),
            stale_threshold: stale_threshold.clamp(0.0, 1.0),
            degraded_threshold: degraded_threshold.clamp(0.0, 1.0),
        }
    }

    /// Register a new signal for monitoring.
    ///
    /// # Arguments
    /// * `signal_type` - Type of signal
    /// * `name` - Human-readable name for logging
    /// * `baseline_mi` - Initial mutual information value
    pub fn register_signal(&mut self, signal_type: EdgeSignalKind, name: String, baseline_mi: f64) {
        let health = SignalHealth::new(name, baseline_mi);
        self.signals.insert(signal_type, health);
    }

    /// Update signal health with new measurement.
    ///
    /// # Arguments
    /// * `signal_type` - Type of signal to update
    /// * `mi` - New mutual information value
    /// * `timestamp` - Epoch milliseconds
    pub fn update_signal(&mut self, signal_type: EdgeSignalKind, mi: f64, timestamp: u64) {
        if let Some(health) = self.signals.get_mut(&signal_type) {
            health.update(mi, timestamp);
        }
    }

    /// Get health status for a specific signal.
    pub fn get_health(&self, signal_type: EdgeSignalKind) -> Option<&SignalHealth> {
        self.signals.get(&signal_type)
    }

    /// Get list of stale signals.
    pub fn stale_signals(&self) -> Vec<EdgeSignalKind> {
        self.signals
            .iter()
            .filter(|(_, health)| health.relative_strength() < self.stale_threshold)
            .map(|(signal_type, _)| *signal_type)
            .collect()
    }

    /// Get list of degraded signals.
    pub fn degraded_signals(&self) -> Vec<EdgeSignalKind> {
        self.signals
            .iter()
            .filter(|(_, health)| {
                let rs = health.relative_strength();
                rs >= self.stale_threshold && rs < self.degraded_threshold
            })
            .map(|(signal_type, _)| *signal_type)
            .collect()
    }

    /// Check if all signals are healthy.
    pub fn all_healthy(&self) -> bool {
        self.signals
            .values()
            .all(|health| health.relative_strength() >= self.degraded_threshold)
    }

    /// Get the weakest signal by relative strength.
    ///
    /// Returns (signal_type, relative_strength) or None if no signals.
    pub fn weakest_signal(&self) -> Option<(EdgeSignalKind, f64)> {
        self.signals
            .iter()
            .min_by(|a, b| {
                a.1.relative_strength()
                    .partial_cmp(&b.1.relative_strength())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(signal_type, health)| (*signal_type, health.relative_strength()))
    }

    /// Generate summary of signal health.
    pub fn summary(&self) -> SignalHealthSummary {
        let total = self.signals.len();
        let stale = self.stale_signals().len();
        let degraded = self.degraded_signals().len();
        let healthy = total.saturating_sub(stale).saturating_sub(degraded);

        let avg_relative_strength = if total > 0 {
            self.signals
                .values()
                .map(|h| h.relative_strength())
                .sum::<f64>()
                / total as f64
        } else {
            1.0
        };

        SignalHealthSummary {
            total_signals: total,
            healthy_signals: healthy,
            degraded_signals: degraded,
            stale_signals: stale,
            avg_relative_strength,
        }
    }
}

impl Default for SignalHealthMonitor {
    fn default() -> Self {
        Self::new(0.5, 0.75)
    }
}

/// Summary statistics for signal health.
#[derive(Debug, Clone, Default)]
pub struct SignalHealthSummary {
    /// Total number of registered signals
    pub total_signals: usize,
    /// Number of healthy signals (RS >= degraded_threshold)
    pub healthy_signals: usize,
    /// Number of degraded signals (stale_threshold <= RS < degraded_threshold)
    pub degraded_signals: usize,
    /// Number of stale signals (RS < stale_threshold)
    pub stale_signals: usize,
    /// Average relative strength across all signals
    pub avg_relative_strength: f64,
}

impl SignalHealthSummary {
    /// Check if system is in healthy state.
    pub fn is_healthy(&self) -> bool {
        self.stale_signals == 0 && self.degraded_signals == 0
    }

    /// Get health status as a simple enum.
    pub fn status(&self) -> HealthStatus {
        if self.stale_signals > 0 {
            HealthStatus::Critical
        } else if self.degraded_signals > 0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Good
        }
    }
}

/// Overall health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All signals healthy
    Good,
    /// Some signals degraded but none stale
    Warning,
    /// At least one signal is stale
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_health_creation() {
        let health = SignalHealth::new("test_signal".to_string(), 0.5);

        assert_eq!(health.name, "test_signal");
        assert_eq!(health.mi_baseline, 0.5);
        assert_eq!(health.mi_current, 0.5);
        assert!((health.relative_strength() - 1.0).abs() < 1e-10);
        assert!(!health.is_stale());
        assert!(!health.is_degraded());
    }

    #[test]
    fn test_signal_health_update() {
        let mut health = SignalHealth::new("test".to_string(), 1.0);

        // Update with degraded MI
        health.update(0.6, 1000);

        assert_eq!(health.mi_current, 0.6);
        assert!((health.relative_strength() - 0.6).abs() < 1e-10);
        assert!(!health.is_stale()); // 60% >= 50%
        assert!(health.is_degraded()); // 60% < 75%
        assert_eq!(health.mi_history.len(), 2);
    }

    #[test]
    fn test_signal_stale_detection() {
        let mut health = SignalHealth::new("test".to_string(), 1.0);

        // Update with stale MI
        health.update(0.4, 1000);

        assert!(health.is_stale());
        assert!(health.is_degraded());
    }

    #[test]
    fn test_decay_rate_estimation() {
        let mut health = SignalHealth::new("test".to_string(), 1.0);

        // Simulate decay: 1.0, 0.9, 0.8, 0.7, 0.6
        for (i, mi) in [0.9, 0.8, 0.7, 0.6].iter().enumerate() {
            health.update(*mi, (i + 1) as u64 * 1000);
        }

        // Should detect positive decay rate (values decreasing)
        let decay = health.estimated_decay_rate();
        assert!(decay > 0.0, "Expected positive decay rate, got {}", decay);
    }

    #[test]
    fn test_signal_type_display() {
        assert_eq!(format!("{}", EdgeSignalKind::LeadLag), "LeadLag");
        assert_eq!(format!("{}", EdgeSignalKind::Custom(42)), "Custom(42)");
    }

    #[test]
    fn test_monitor_register_and_update() {
        let mut monitor = SignalHealthMonitor::new(0.5, 0.75);

        monitor.register_signal(EdgeSignalKind::LeadLag, "Lead-Lag".to_string(), 0.8);
        monitor.register_signal(
            EdgeSignalKind::FillProbability,
            "Fill Prob".to_string(),
            0.6,
        );

        assert_eq!(monitor.summary().total_signals, 2);
        assert!(monitor.all_healthy());

        // Degrade one signal
        monitor.update_signal(EdgeSignalKind::LeadLag, 0.3, 1000);

        assert!(!monitor.all_healthy());
        assert_eq!(monitor.stale_signals().len(), 1);
    }

    #[test]
    fn test_weakest_signal() {
        let mut monitor = SignalHealthMonitor::new(0.5, 0.75);

        monitor.register_signal(EdgeSignalKind::LeadLag, "Lead-Lag".to_string(), 1.0);
        monitor.register_signal(
            EdgeSignalKind::FillProbability,
            "Fill Prob".to_string(),
            1.0,
        );

        // Degrade LeadLag more than FillProbability
        monitor.update_signal(EdgeSignalKind::LeadLag, 0.4, 1000);
        monitor.update_signal(EdgeSignalKind::FillProbability, 0.7, 1000);

        let (weakest, strength) = monitor.weakest_signal().unwrap();
        assert_eq!(weakest, EdgeSignalKind::LeadLag);
        assert!((strength - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_summary_statistics() {
        let mut monitor = SignalHealthMonitor::new(0.5, 0.75);

        // Add 3 signals with different health levels
        monitor.register_signal(EdgeSignalKind::LeadLag, "LL".to_string(), 1.0);
        monitor.register_signal(EdgeSignalKind::FillProbability, "FP".to_string(), 1.0);
        monitor.register_signal(EdgeSignalKind::AdverseSelection, "AS".to_string(), 1.0);

        // Make one healthy, one degraded, one stale
        monitor.update_signal(EdgeSignalKind::LeadLag, 0.9, 1000); // Healthy (90%)
        monitor.update_signal(EdgeSignalKind::FillProbability, 0.6, 1000); // Degraded (60%)
        monitor.update_signal(EdgeSignalKind::AdverseSelection, 0.3, 1000); // Stale (30%)

        let summary = monitor.summary();

        assert_eq!(summary.total_signals, 3);
        assert_eq!(summary.healthy_signals, 1);
        assert_eq!(summary.degraded_signals, 1);
        assert_eq!(summary.stale_signals, 1);
        assert_eq!(summary.status(), HealthStatus::Critical);
    }

    #[test]
    fn test_summary_status() {
        let summary_good = SignalHealthSummary {
            total_signals: 3,
            healthy_signals: 3,
            degraded_signals: 0,
            stale_signals: 0,
            avg_relative_strength: 0.95,
        };
        assert_eq!(summary_good.status(), HealthStatus::Good);

        let summary_warning = SignalHealthSummary {
            total_signals: 3,
            healthy_signals: 2,
            degraded_signals: 1,
            stale_signals: 0,
            avg_relative_strength: 0.85,
        };
        assert_eq!(summary_warning.status(), HealthStatus::Warning);

        let summary_critical = SignalHealthSummary {
            total_signals: 3,
            healthy_signals: 1,
            degraded_signals: 1,
            stale_signals: 1,
            avg_relative_strength: 0.6,
        };
        assert_eq!(summary_critical.status(), HealthStatus::Critical);
    }

    #[test]
    fn test_history_bounded() {
        let mut health = SignalHealth::new("test".to_string(), 1.0);

        // Add 150 updates
        for i in 0..150 {
            health.update(0.9, i as u64 * 1000);
        }

        // History should be bounded to 100
        assert!(health.mi_history.len() <= 100);
    }

    #[test]
    fn test_zero_baseline() {
        let health = SignalHealth::new("test".to_string(), 0.0);

        // Should handle zero baseline gracefully
        assert_eq!(health.relative_strength(), 0.0);
        assert_eq!(health.estimated_decay_rate(), 0.0);
    }
}
