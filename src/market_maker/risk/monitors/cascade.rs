//! Liquidation cascade monitor.

use crate::market_maker::risk::{RiskAction, RiskAssessment, RiskMonitor, RiskState};

/// Monitors liquidation cascade severity.
pub struct CascadeMonitor {
    /// Threshold for pulling quotes (severity level)
    pull_threshold: f64,
    /// Threshold for kill switch (severity level)
    kill_threshold: f64,
    /// Threshold for widening spreads (severity level)
    widen_threshold: f64,
    /// Spread widening factor per unit of severity
    widen_factor: f64,
}

impl CascadeMonitor {
    /// Create a new cascade monitor.
    ///
    /// # Arguments
    ///
    /// * `pull_threshold` - Cascade severity at which to pull quotes
    /// * `kill_threshold` - Cascade severity at which to trigger kill switch
    pub fn new(pull_threshold: f64, kill_threshold: f64) -> Self {
        Self {
            pull_threshold,
            kill_threshold,
            widen_threshold: pull_threshold * 0.5,
            widen_factor: 0.5, // +50% spread per unit severity above threshold
        }
    }

    /// Create with custom widen threshold.
    pub fn with_widen_threshold(mut self, threshold: f64, factor: f64) -> Self {
        self.widen_threshold = threshold;
        self.widen_factor = factor;
        self
    }
}

impl Default for CascadeMonitor {
    fn default() -> Self {
        Self::new(0.8, 5.0) // Default: pull at 0.8, kill at 5.0
    }
}

impl RiskMonitor for CascadeMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        let severity = state.cascade_severity;

        // Kill switch at extreme severity
        if severity > self.kill_threshold {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Extreme liquidation cascade: severity {:.2} > {:.2}",
                    severity, self.kill_threshold
                ),
            )
            .with_metric(severity)
            .with_threshold(self.kill_threshold);
        }

        // Pull quotes at high severity
        if severity > self.pull_threshold {
            return RiskAssessment::high(
                self.name(),
                RiskAction::PullQuotes,
                format!(
                    "Liquidation cascade detected: severity {:.2} > {:.2}",
                    severity, self.pull_threshold
                ),
            )
            .with_metric(severity)
            .with_threshold(self.pull_threshold);
        }

        // Widen spreads at moderate severity
        if severity > self.widen_threshold {
            let excess = severity - self.widen_threshold;
            let widen_factor = 1.0 + excess * self.widen_factor;

            return RiskAssessment::high(
                self.name(),
                RiskAction::WidenSpreads(widen_factor),
                format!(
                    "Elevated cascade risk: severity {severity:.2}, widening spreads by {widen_factor:.1}x"
                ),
            )
            .with_metric(severity)
            .with_threshold(self.widen_threshold);
        }

        RiskAssessment::ok(self.name())
    }

    fn name(&self) -> &'static str {
        "CascadeMonitor"
    }

    fn priority(&self) -> u32 {
        30 // Medium-high priority
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::risk::RiskSeverity;

    #[test]
    fn test_calm() {
        let monitor = CascadeMonitor::default();
        let state = RiskState {
            cascade_severity: 0.2,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_widen_spreads() {
        let monitor = CascadeMonitor::new(0.8, 5.0).with_widen_threshold(0.4, 0.5);
        let state = RiskState {
            cascade_severity: 0.6, // Above widen threshold
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        if let RiskAction::WidenSpreads(factor) = assessment.action {
            assert!(factor > 1.0);
        } else {
            panic!("Expected WidenSpreads action");
        }
    }

    #[test]
    fn test_pull_quotes() {
        let monitor = CascadeMonitor::new(0.8, 5.0);
        let state = RiskState {
            cascade_severity: 1.0, // Above pull threshold
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
    }

    #[test]
    fn test_kill() {
        let monitor = CascadeMonitor::new(0.8, 5.0);
        let state = RiskState {
            cascade_severity: 6.0, // Above kill threshold
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }
}
