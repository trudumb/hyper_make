//! Daily loss limit monitor.

use crate::market_maker::risk::{RiskAssessment, RiskMonitor, RiskState};

/// Monitors daily P&L against loss limit.
pub struct LossMonitor {
    /// Maximum allowed daily loss (positive value)
    max_daily_loss: f64,
    /// Warning threshold (fraction of limit, e.g., 0.8 = 80%)
    warning_threshold: f64,
}

impl LossMonitor {
    /// Create a new loss monitor.
    ///
    /// # Arguments
    ///
    /// * `max_daily_loss` - Maximum allowed daily loss in USD (positive)
    pub fn new(max_daily_loss: f64) -> Self {
        Self {
            max_daily_loss,
            warning_threshold: 0.8,
        }
    }

    /// Create with custom warning threshold.
    pub fn with_warning_threshold(mut self, threshold: f64) -> Self {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

impl RiskMonitor for LossMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        let loss = -state.daily_pnl; // Convert to positive loss

        if loss <= 0.0 {
            // In profit or breakeven
            return RiskAssessment::ok(self.name());
        }

        if loss > self.max_daily_loss {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Daily loss ${:.2} exceeds limit ${:.2}",
                    loss, self.max_daily_loss
                ),
            )
            .with_metric(loss)
            .with_threshold(self.max_daily_loss);
        }

        let loss_ratio = loss / self.max_daily_loss;
        if loss_ratio > self.warning_threshold {
            return RiskAssessment::warn(
                self.name(),
                format!(
                    "Daily loss ${:.2} is {:.0}% of limit ${:.2}",
                    loss,
                    loss_ratio * 100.0,
                    self.max_daily_loss
                ),
            )
            .with_metric(loss)
            .with_threshold(self.max_daily_loss);
        }

        RiskAssessment::ok(self.name())
    }

    fn name(&self) -> &'static str {
        "LossMonitor"
    }

    fn priority(&self) -> u32 {
        0 // Highest priority - P&L is critical
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::risk::RiskSeverity;

    #[test]
    fn test_in_profit() {
        let monitor = LossMonitor::new(500.0);
        let state = RiskState {
            daily_pnl: 100.0,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_small_loss() {
        let monitor = LossMonitor::new(500.0);
        let state = RiskState {
            daily_pnl: -100.0, // 20% of limit
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_warning_level() {
        let monitor = LossMonitor::new(500.0).with_warning_threshold(0.8);
        let state = RiskState {
            daily_pnl: -450.0, // 90% of limit
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Medium);
    }

    #[test]
    fn test_over_limit() {
        let monitor = LossMonitor::new(500.0);
        let state = RiskState {
            daily_pnl: -600.0, // Over limit
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }
}
