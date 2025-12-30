//! Risk aggregator for unified risk evaluation.
//!
//! Collects risk monitors and provides a single evaluation point.

use super::{RiskAction, RiskAssessment, RiskMonitor, RiskMonitorBox, RiskSeverity, RiskState};

/// Aggregated risk evaluation result.
#[derive(Debug, Clone)]
pub struct AggregatedRisk {
    /// Individual assessments from each monitor
    pub assessments: Vec<RiskAssessment>,
    /// Highest severity across all monitors
    pub max_severity: RiskSeverity,
    /// Most severe action recommended
    pub primary_action: RiskAction,
    /// All kill reasons (if any monitors triggered kill)
    pub kill_reasons: Vec<String>,
    /// Should enter reduce-only mode?
    pub reduce_only: bool,
    /// Should pull all quotes?
    pub pull_quotes: bool,
    /// Spread widening factor (max across monitors)
    pub spread_factor: f64,
}

impl Default for AggregatedRisk {
    fn default() -> Self {
        Self {
            assessments: Vec::new(),
            max_severity: RiskSeverity::None,
            primary_action: RiskAction::None,
            kill_reasons: Vec::new(),
            reduce_only: false,
            pull_quotes: false,
            spread_factor: 1.0,
        }
    }
}

impl AggregatedRisk {
    /// Is any kill condition triggered?
    pub fn should_kill(&self) -> bool {
        !self.kill_reasons.is_empty()
    }

    /// Get actionable assessments only.
    pub fn actionable(&self) -> impl Iterator<Item = &RiskAssessment> {
        self.assessments.iter().filter(|a| a.is_actionable())
    }

    /// Get summary string for logging.
    pub fn summary(&self) -> String {
        if self.should_kill() {
            format!(
                "KILL: {} reasons - {:?}",
                self.kill_reasons.len(),
                self.kill_reasons
            )
        } else if self.max_severity >= RiskSeverity::High {
            format!(
                "HIGH RISK: {} actionable, reduce_only={}, pull_quotes={}, spread_factor={:.2}",
                self.assessments
                    .iter()
                    .filter(|a| a.is_actionable())
                    .count(),
                self.reduce_only,
                self.pull_quotes,
                self.spread_factor
            )
        } else {
            format!("OK: max_severity={:?}", self.max_severity)
        }
    }
}

/// Aggregates multiple risk monitors for unified evaluation.
///
/// # Example
///
/// ```ignore
/// let aggregator = RiskAggregator::new()
///     .with_monitor(Box::new(LossMonitor::new(500.0)))
///     .with_monitor(Box::new(PositionMonitor::new(10000.0)));
///
/// let state = RiskState::default();
/// let result = aggregator.evaluate(&state);
///
/// if result.should_kill() {
///     // Handle kill switch
/// }
/// ```
pub struct RiskAggregator {
    /// Registered monitors (sorted by priority)
    monitors: Vec<RiskMonitorBox>,
}

impl Default for RiskAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskAggregator {
    /// Create a new empty aggregator.
    pub fn new() -> Self {
        Self {
            monitors: Vec::new(),
        }
    }

    /// Add a monitor (maintains priority ordering).
    pub fn with_monitor(mut self, monitor: RiskMonitorBox) -> Self {
        self.monitors.push(monitor);
        self.monitors.sort_by_key(|m| m.priority());
        self
    }

    /// Add a monitor (mutable version).
    pub fn add_monitor(&mut self, monitor: RiskMonitorBox) {
        self.monitors.push(monitor);
        self.monitors.sort_by_key(|m| m.priority());
    }

    /// Evaluate all monitors against the current state.
    pub fn evaluate(&self, state: &RiskState) -> AggregatedRisk {
        let mut result = AggregatedRisk::default();

        for monitor in &self.monitors {
            if !monitor.is_enabled() {
                continue;
            }

            let assessment = monitor.evaluate(state);

            // Update max severity
            if assessment.severity > result.max_severity {
                result.max_severity = assessment.severity;
            }

            // Process action
            match &assessment.action {
                RiskAction::None => {}
                RiskAction::Warn(_) => {}
                RiskAction::ReduceOnly => {
                    result.reduce_only = true;
                }
                RiskAction::WidenSpreads(factor) => {
                    result.spread_factor = result.spread_factor.max(*factor);
                }
                RiskAction::PullQuotes => {
                    result.pull_quotes = true;
                }
                RiskAction::Kill(reason) => {
                    result.kill_reasons.push(reason.clone());
                }
            }

            // Update primary action (highest severity action wins)
            if assessment.severity > RiskSeverity::None {
                match (&result.primary_action, &assessment.action) {
                    (RiskAction::None, action) => {
                        result.primary_action = action.clone();
                    }
                    (RiskAction::Kill(_), _) => {
                        // Keep kill action
                    }
                    (_, RiskAction::Kill(_)) => {
                        result.primary_action = assessment.action.clone();
                    }
                    (RiskAction::PullQuotes, _) => {
                        // Keep pull quotes unless kill
                    }
                    (_, RiskAction::PullQuotes) => {
                        result.primary_action = assessment.action.clone();
                    }
                    (RiskAction::ReduceOnly, _) => {
                        // Keep reduce only unless higher priority
                    }
                    (_, RiskAction::ReduceOnly) => {
                        result.primary_action = assessment.action.clone();
                    }
                    _ => {}
                }
            }

            result.assessments.push(assessment);
        }

        result
    }

    /// Get monitor names for debugging.
    pub fn monitor_names(&self) -> Vec<&'static str> {
        self.monitors.iter().map(|m| m.name()).collect()
    }

    /// Get number of registered monitors.
    pub fn monitor_count(&self) -> usize {
        self.monitors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test monitor that always returns OK
    struct OkMonitor;
    impl RiskMonitor for OkMonitor {
        fn evaluate(&self, _state: &RiskState) -> RiskAssessment {
            RiskAssessment::ok("OkMonitor")
        }
        fn name(&self) -> &'static str {
            "OkMonitor"
        }
    }

    // Test monitor that always returns critical
    struct CriticalMonitor {
        reason: &'static str,
    }
    impl RiskMonitor for CriticalMonitor {
        fn evaluate(&self, _state: &RiskState) -> RiskAssessment {
            RiskAssessment::critical("CriticalMonitor", self.reason)
        }
        fn name(&self) -> &'static str {
            "CriticalMonitor"
        }
        fn priority(&self) -> u32 {
            0 // High priority
        }
    }

    // Test monitor that recommends reduce-only
    struct ReduceOnlyMonitor;
    impl RiskMonitor for ReduceOnlyMonitor {
        fn evaluate(&self, _state: &RiskState) -> RiskAssessment {
            RiskAssessment::high("ReduceOnlyMonitor", RiskAction::ReduceOnly, "Over limit")
        }
        fn name(&self) -> &'static str {
            "ReduceOnlyMonitor"
        }
    }

    #[test]
    fn test_empty_aggregator() {
        let aggregator = RiskAggregator::new();
        let result = aggregator.evaluate(&RiskState::default());

        assert_eq!(result.max_severity, RiskSeverity::None);
        assert!(!result.should_kill());
        assert!(!result.reduce_only);
    }

    #[test]
    fn test_all_ok() {
        let aggregator = RiskAggregator::new()
            .with_monitor(Box::new(OkMonitor))
            .with_monitor(Box::new(OkMonitor));

        let result = aggregator.evaluate(&RiskState::default());

        assert_eq!(result.max_severity, RiskSeverity::None);
        assert!(!result.should_kill());
        assert_eq!(result.assessments.len(), 2);
    }

    #[test]
    fn test_one_critical() {
        let aggregator = RiskAggregator::new()
            .with_monitor(Box::new(OkMonitor))
            .with_monitor(Box::new(CriticalMonitor {
                reason: "Test kill",
            }));

        let result = aggregator.evaluate(&RiskState::default());

        assert_eq!(result.max_severity, RiskSeverity::Critical);
        assert!(result.should_kill());
        assert_eq!(result.kill_reasons.len(), 1);
        assert_eq!(result.kill_reasons[0], "Test kill");
    }

    #[test]
    fn test_reduce_only() {
        let aggregator = RiskAggregator::new().with_monitor(Box::new(ReduceOnlyMonitor));

        let result = aggregator.evaluate(&RiskState::default());

        assert!(result.reduce_only);
        assert!(!result.should_kill());
    }

    #[test]
    fn test_priority_ordering() {
        let aggregator = RiskAggregator::new()
            .with_monitor(Box::new(OkMonitor)) // priority 100
            .with_monitor(Box::new(CriticalMonitor { reason: "Test" })); // priority 0

        // Critical should be first due to lower priority
        let names = aggregator.monitor_names();
        assert_eq!(names[0], "CriticalMonitor");
        assert_eq!(names[1], "OkMonitor");
    }
}
