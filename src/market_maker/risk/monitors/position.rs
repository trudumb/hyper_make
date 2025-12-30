//! Position limit monitor.

use crate::market_maker::risk::{RiskAction, RiskAssessment, RiskMonitor, RiskState};

/// Monitors position size and value against limits.
pub struct PositionMonitor {
    /// Hard limit multiplier (kill switch at this multiple of soft limit)
    hard_limit_multiplier: f64,
    /// Warning threshold (fraction of limit)
    warning_threshold: f64,
}

impl PositionMonitor {
    /// Create a new position monitor.
    pub fn new() -> Self {
        Self {
            hard_limit_multiplier: 2.0, // Kill at 2x soft limit
            warning_threshold: 0.8,
        }
    }

    /// Create with custom hard limit multiplier.
    pub fn with_hard_limit(mut self, multiplier: f64) -> Self {
        self.hard_limit_multiplier = multiplier.max(1.0);
        self
    }

    /// Create with custom warning threshold.
    pub fn with_warning_threshold(mut self, threshold: f64) -> Self {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

impl Default for PositionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskMonitor for PositionMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        // Check position value (USD) - primary limit
        let value_utilization = state.value_utilization();
        let hard_value_limit = state.max_position_value * self.hard_limit_multiplier;

        // Kill switch at hard limit (2x soft limit by default)
        if state.position_value > hard_value_limit {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Position value ${:.2} exceeds hard limit ${:.2}",
                    state.position_value, hard_value_limit
                ),
            )
            .with_metric(state.position_value)
            .with_threshold(hard_value_limit);
        }

        // Reduce-only mode at soft limit (1x)
        if value_utilization > 1.0 {
            return RiskAssessment::high(
                self.name(),
                RiskAction::ReduceOnly,
                format!(
                    "Position value ${:.2} exceeds soft limit ${:.2} - reduce only",
                    state.position_value, state.max_position_value
                ),
            )
            .with_metric(state.position_value)
            .with_threshold(state.max_position_value);
        }

        // Warning near limit
        if value_utilization > self.warning_threshold {
            return RiskAssessment::warn(
                self.name(),
                format!(
                    "Position value ${:.2} is {:.0}% of limit ${:.2}",
                    state.position_value,
                    value_utilization * 100.0,
                    state.max_position_value
                ),
            )
            .with_metric(state.position_value)
            .with_threshold(state.max_position_value);
        }

        // Also check contract limit
        let inventory_utilization = state.inventory_utilization();
        if inventory_utilization > 1.0 {
            return RiskAssessment::high(
                self.name(),
                RiskAction::ReduceOnly,
                format!(
                    "Position {:.6} exceeds limit {:.6} - reduce only",
                    state.position.abs(),
                    state.max_position
                ),
            )
            .with_metric(state.position.abs())
            .with_threshold(state.max_position);
        }

        if inventory_utilization > self.warning_threshold {
            return RiskAssessment::warn(
                self.name(),
                format!(
                    "Position {:.6} is {:.0}% of limit {:.6}",
                    state.position.abs(),
                    inventory_utilization * 100.0,
                    state.max_position
                ),
            )
            .with_metric(state.position.abs())
            .with_threshold(state.max_position);
        }

        // Check worst-case positions from pending exposure
        // This warns if all resting orders filling would breach limits
        if state.worst_case_exceeds_limits() {
            let breach_side = if state.worst_case_exceeds_long_limit() {
                "long"
            } else {
                "short"
            };
            let worst_position = if state.worst_case_exceeds_long_limit() {
                state.worst_case_max_position
            } else {
                state.worst_case_min_position
            };
            return RiskAssessment::warn(
                self.name(),
                format!(
                    "Worst-case {} position {:.6} would exceed limit {:.6} (pending bid={:.6}, ask={:.6})",
                    breach_side,
                    worst_position.abs(),
                    state.max_position,
                    state.pending_bid_exposure,
                    state.pending_ask_exposure
                ),
            )
            .with_metric(worst_position.abs())
            .with_threshold(state.max_position);
        }

        RiskAssessment::ok(self.name())
    }

    fn name(&self) -> &'static str {
        "PositionMonitor"
    }

    fn priority(&self) -> u32 {
        10 // High priority
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::risk::RiskSeverity;

    #[test]
    fn test_under_limit() {
        let monitor = PositionMonitor::new();
        let state = RiskState {
            position: 0.5,
            max_position: 1.0,
            position_value: 5000.0,
            max_position_value: 10000.0,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_warning_level() {
        let monitor = PositionMonitor::new().with_warning_threshold(0.8);
        let state = RiskState {
            position: 0.5,
            max_position: 1.0,
            position_value: 9000.0, // 90% of limit
            max_position_value: 10000.0,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Medium);
    }

    #[test]
    fn test_reduce_only() {
        let monitor = PositionMonitor::new();
        let state = RiskState {
            position: 1.2,
            max_position: 1.0,
            position_value: 12000.0, // Over soft limit
            max_position_value: 10000.0,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::ReduceOnly));
    }

    #[test]
    fn test_hard_limit_kill() {
        let monitor = PositionMonitor::new().with_hard_limit(2.0);
        let state = RiskState {
            position: 2.5,
            max_position: 1.0,
            position_value: 25000.0, // Over 2x soft limit (hard limit)
            max_position_value: 10000.0,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }

    #[test]
    fn test_worst_case_position_warning() {
        let monitor = PositionMonitor::new();
        // Currently at 50% utilization but pending orders would exceed
        let state = RiskState {
            position: 0.5,
            max_position: 1.0,
            position_value: 5000.0,
            max_position_value: 10000.0,
            ..Default::default()
        }
        .with_pending_exposure(0.7, 0.2); // Would go to 1.2 long if all bids fill

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Medium);
        assert!(assessment.description.contains("Worst-case"));
        assert!(assessment.description.contains("long"));
    }

    #[test]
    fn test_worst_case_position_ok() {
        let monitor = PositionMonitor::new();
        // Pending orders won't exceed limits
        let state = RiskState {
            position: 0.5,
            max_position: 1.0,
            position_value: 5000.0,
            max_position_value: 10000.0,
            ..Default::default()
        }
        .with_pending_exposure(0.3, 0.2); // Would go to 0.8 long max

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }
}
