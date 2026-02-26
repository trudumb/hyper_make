//! Rate limit error monitor.

use crate::market_maker::risk::{RiskAction, RiskAssessment, RiskMonitor, RiskState};

/// Monitors exchange rate limit errors.
pub struct RateLimitMonitor {
    /// Maximum allowed rate limit errors before kill
    max_errors: u32,
    /// Warning threshold (errors)
    warning_threshold: u32,
}

impl RateLimitMonitor {
    /// Create a new rate limit monitor.
    ///
    /// # Arguments
    ///
    /// * `max_errors` - Maximum allowed rate limit errors
    pub fn new(max_errors: u32) -> Self {
        Self {
            max_errors,
            warning_threshold: max_errors.saturating_sub(1),
        }
    }

    /// Create with custom warning threshold.
    pub fn with_warning_threshold(mut self, threshold: u32) -> Self {
        self.warning_threshold = threshold;
        self
    }
}

impl Default for RateLimitMonitor {
    fn default() -> Self {
        Self::new(3) // Default: 3 errors triggers kill
    }
}

impl RiskMonitor for RateLimitMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        let errors = state.rate_limit_errors;

        if errors > self.max_errors {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Rate limit errors {} exceeds maximum {}",
                    errors, self.max_errors
                ),
            )
            .with_metric(errors as f64)
            .with_threshold(self.max_errors as f64);
        }

        if errors >= self.warning_threshold {
            // Back off - pull quotes to reduce API calls
            return RiskAssessment::high(
                self.name(),
                RiskAction::PullQuotes,
                format!(
                    "Rate limit errors {} approaching limit {}",
                    errors, self.max_errors
                ),
            )
            .with_metric(errors as f64)
            .with_threshold(self.max_errors as f64);
        }

        if errors > 0 {
            return RiskAssessment::warn(
                self.name(),
                format!("Rate limit errors: {} (limit: {})", errors, self.max_errors),
            )
            .with_metric(errors as f64)
            .with_threshold(self.max_errors as f64);
        }

        RiskAssessment::ok(self.name())
    }

    fn name(&self) -> &'static str {
        "RateLimitMonitor"
    }

    fn priority(&self) -> u32 {
        15 // High priority - rate limits can cascade
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::risk::RiskSeverity;

    #[test]
    fn test_no_errors() {
        let monitor = RateLimitMonitor::default();
        let state = RiskState {
            rate_limit_errors: 0,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_some_errors() {
        let monitor = RateLimitMonitor::new(5);
        let state = RiskState {
            rate_limit_errors: 1,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Medium);
    }

    #[test]
    fn test_warning_level() {
        let monitor = RateLimitMonitor::new(3);
        let state = RiskState {
            rate_limit_errors: 2, // One below max
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
    }

    #[test]
    fn test_over_limit() {
        let monitor = RateLimitMonitor::new(3);
        let state = RiskState {
            rate_limit_errors: 5,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }
}
