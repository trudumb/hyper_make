//! Data staleness monitor.

use std::time::Duration;

use crate::market_maker::risk::{RiskAction, RiskAssessment, RiskMonitor, RiskState};

/// Monitors market data freshness.
pub struct DataStalenessMonitor {
    /// Threshold for kill switch
    stale_threshold: Duration,
    /// Warning threshold (fraction of kill threshold)
    warning_threshold: f64,
}

impl DataStalenessMonitor {
    /// Create a new data staleness monitor.
    ///
    /// # Arguments
    ///
    /// * `stale_threshold` - Duration after which data is considered stale
    pub fn new(stale_threshold: Duration) -> Self {
        Self {
            stale_threshold,
            warning_threshold: 0.5,
        }
    }

    /// Create with custom warning threshold.
    pub fn with_warning_threshold(mut self, threshold: f64) -> Self {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

impl RiskMonitor for DataStalenessMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        let stale_secs = self.stale_threshold.as_secs_f64();
        let data_age_secs = state.data_age.as_secs_f64();

        if state.data_age > self.stale_threshold {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Data stale: {:.1}s since last update (threshold: {:.1}s)",
                    data_age_secs, stale_secs
                ),
            )
            .with_metric(data_age_secs)
            .with_threshold(stale_secs);
        }

        let staleness_ratio = data_age_secs / stale_secs;

        // At 50% of threshold, recommend pulling quotes
        if staleness_ratio > self.warning_threshold {
            return RiskAssessment::high(
                self.name(),
                RiskAction::PullQuotes,
                format!(
                    "Data aging: {:.1}s since last update ({:.0}% of threshold)",
                    data_age_secs,
                    staleness_ratio * 100.0
                ),
            )
            .with_metric(data_age_secs)
            .with_threshold(stale_secs);
        }

        RiskAssessment::ok(self.name())
    }

    fn name(&self) -> &'static str {
        "DataStalenessMonitor"
    }

    fn priority(&self) -> u32 {
        20 // High priority - stale data is dangerous
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::market_maker::risk::RiskSeverity;

    #[test]
    fn test_fresh_data() {
        let monitor = DataStalenessMonitor::new(Duration::from_secs(30));
        let now = Instant::now();
        let state = RiskState {
            data_age: Duration::from_secs(5),
            last_data_time: now - Duration::from_secs(5),
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_warning_level() {
        let monitor =
            DataStalenessMonitor::new(Duration::from_secs(30)).with_warning_threshold(0.5);
        let now = Instant::now();
        let state = RiskState {
            data_age: Duration::from_secs(20), // 67% of threshold
            last_data_time: now - Duration::from_secs(20),
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
    }

    #[test]
    fn test_stale() {
        let monitor = DataStalenessMonitor::new(Duration::from_secs(30));
        let now = Instant::now();
        let state = RiskState {
            data_age: Duration::from_secs(35),
            last_data_time: now - Duration::from_secs(35),
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }
}
