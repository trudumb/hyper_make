//! Data staleness monitor.
//!
//! Monitors market data freshness while respecting reconnection attempts.
//! When WebSocket is actively reconnecting, gives grace period before triggering
//! the kill switch to allow recovery.

use std::time::Duration;

use crate::market_maker::risk::{RiskAction, RiskAssessment, RiskMonitor, RiskState};

/// Monitors market data freshness.
pub struct DataStalenessMonitor {
    /// Threshold for kill switch (when not reconnecting)
    stale_threshold: Duration,
    /// Warning threshold (fraction of kill threshold)
    warning_threshold: f64,
    /// Maximum reconnection attempts before triggering kill switch
    /// (default: 5 attempts, ~30-60 seconds with exponential backoff)
    max_reconnection_attempts: u32,
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
            max_reconnection_attempts: 5,
        }
    }

    /// Create with custom warning threshold.
    pub fn with_warning_threshold(mut self, threshold: f64) -> Self {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Create with custom max reconnection attempts before kill switch.
    pub fn with_max_reconnection_attempts(mut self, attempts: u32) -> Self {
        self.max_reconnection_attempts = attempts.max(1);
        self
    }
}

impl RiskMonitor for DataStalenessMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        let stale_secs = self.stale_threshold.as_secs_f64();
        let data_age_secs = state.data_age.as_secs_f64();

        // If connection has permanently failed, trigger kill switch immediately
        if state.connection_failed {
            return RiskAssessment::critical(
                self.name(),
                "Connection permanently failed after max retries".to_string(),
            )
            .with_metric(data_age_secs)
            .with_threshold(stale_secs);
        }

        // Check if data is stale
        if state.data_age > self.stale_threshold {
            // If actively reconnecting, give grace period
            if state.is_reconnecting {
                // Still within grace period - pull quotes but don't kill
                if state.reconnection_attempt < self.max_reconnection_attempts {
                    return RiskAssessment::high(
                        self.name(),
                        RiskAction::PullQuotes,
                        format!(
                            "Reconnecting (attempt {}/{}), data stale: {:.1}s",
                            state.reconnection_attempt,
                            self.max_reconnection_attempts,
                            data_age_secs
                        ),
                    )
                    .with_metric(data_age_secs)
                    .with_threshold(stale_secs);
                }

                // Exceeded max reconnection attempts - trigger kill switch
                return RiskAssessment::critical(
                    self.name(),
                    format!(
                        "Reconnection failed after {} attempts, data stale: {:.1}s",
                        state.reconnection_attempt, data_age_secs
                    ),
                )
                .with_metric(data_age_secs)
                .with_threshold(stale_secs);
            }

            // Not reconnecting but data is stale - this shouldn't happen normally,
            // but trigger kill switch as a safety measure
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
    fn test_stale_not_reconnecting() {
        let monitor = DataStalenessMonitor::new(Duration::from_secs(30));
        let now = Instant::now();
        let state = RiskState {
            data_age: Duration::from_secs(35),
            last_data_time: now - Duration::from_secs(35),
            is_reconnecting: false,
            reconnection_attempt: 0,
            connection_failed: false,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }

    #[test]
    fn test_stale_but_reconnecting_within_grace() {
        let monitor = DataStalenessMonitor::new(Duration::from_secs(30))
            .with_max_reconnection_attempts(5);
        let now = Instant::now();
        let state = RiskState {
            data_age: Duration::from_secs(45), // Stale
            last_data_time: now - Duration::from_secs(45),
            is_reconnecting: true,
            reconnection_attempt: 2, // Attempt 2 of 5 - within grace
            connection_failed: false,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        // Should be High (pull quotes) not Critical (kill)
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
        assert!(!assessment.should_kill());
    }

    #[test]
    fn test_stale_reconnection_exhausted() {
        let monitor = DataStalenessMonitor::new(Duration::from_secs(30))
            .with_max_reconnection_attempts(5);
        let now = Instant::now();
        let state = RiskState {
            data_age: Duration::from_secs(120), // Very stale
            last_data_time: now - Duration::from_secs(120),
            is_reconnecting: true,
            reconnection_attempt: 6, // Exceeded max attempts
            connection_failed: false,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        // Should be Critical (kill) after exceeding max attempts
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }

    #[test]
    fn test_connection_permanently_failed() {
        let monitor = DataStalenessMonitor::new(Duration::from_secs(30));
        let now = Instant::now();
        let state = RiskState {
            data_age: Duration::from_secs(60),
            last_data_time: now - Duration::from_secs(60),
            is_reconnecting: false,
            reconnection_attempt: 10,
            connection_failed: true, // Permanently failed
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        // Should immediately kill when connection has permanently failed
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }
}
