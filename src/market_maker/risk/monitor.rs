//! Risk monitor trait and assessment types.
//!
//! Defines the interface for risk monitors and their outputs.

use super::RiskState;

/// Type alias for boxed monitor.
pub type RiskMonitorBox = Box<dyn RiskMonitor>;

/// Severity level of a risk assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskSeverity {
    /// No risk detected
    None,
    /// Low risk - informational
    Low,
    /// Medium risk - may warrant attention
    Medium,
    /// High risk - action recommended
    High,
    /// Critical risk - immediate action required
    Critical,
}

impl RiskSeverity {
    /// Is this severity actionable (High or Critical)?
    pub fn is_actionable(&self) -> bool {
        matches!(self, RiskSeverity::High | RiskSeverity::Critical)
    }

    /// Should trigger kill switch (Critical only)?
    pub fn should_kill(&self) -> bool {
        matches!(self, RiskSeverity::Critical)
    }
}

/// Recommended action from a risk assessment.
#[derive(Debug, Clone, PartialEq)]
pub enum RiskAction {
    /// No action needed
    None,
    /// Log a warning
    Warn(String),
    /// Enter reduce-only mode
    ReduceOnly,
    /// Widen spreads by the given factor
    WidenSpreads(f64),
    /// Pull all quotes
    PullQuotes,
    /// Trigger kill switch with reason
    Kill(String),
}

impl RiskAction {
    /// Is this a kill action?
    pub fn is_kill(&self) -> bool {
        matches!(self, RiskAction::Kill(_))
    }

    /// Is this a quote-affecting action?
    pub fn affects_quotes(&self) -> bool {
        matches!(
            self,
            RiskAction::ReduceOnly
                | RiskAction::WidenSpreads(_)
                | RiskAction::PullQuotes
                | RiskAction::Kill(_)
        )
    }
}

/// Result of a risk monitor evaluation.
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Name of the monitor that produced this assessment
    pub monitor: &'static str,
    /// Severity of the risk
    pub severity: RiskSeverity,
    /// Recommended action
    pub action: RiskAction,
    /// Human-readable description
    pub description: String,
    /// Numeric metric value (for tracking)
    pub metric_value: Option<f64>,
    /// Threshold that was exceeded (if any)
    pub threshold: Option<f64>,
}

impl RiskAssessment {
    /// Create a "no risk" assessment.
    pub fn ok(monitor: &'static str) -> Self {
        Self {
            monitor,
            severity: RiskSeverity::None,
            action: RiskAction::None,
            description: String::new(),
            metric_value: None,
            threshold: None,
        }
    }

    /// Create a warning assessment.
    pub fn warn(monitor: &'static str, description: impl Into<String>) -> Self {
        let desc = description.into();
        Self {
            monitor,
            severity: RiskSeverity::Medium,
            action: RiskAction::Warn(desc.clone()),
            description: desc,
            metric_value: None,
            threshold: None,
        }
    }

    /// Create a high-severity assessment.
    pub fn high(monitor: &'static str, action: RiskAction, description: impl Into<String>) -> Self {
        Self {
            monitor,
            severity: RiskSeverity::High,
            action,
            description: description.into(),
            metric_value: None,
            threshold: None,
        }
    }

    /// Create a critical/kill assessment.
    pub fn critical(monitor: &'static str, reason: impl Into<String>) -> Self {
        let reason = reason.into();
        Self {
            monitor,
            severity: RiskSeverity::Critical,
            action: RiskAction::Kill(reason.clone()),
            description: reason,
            metric_value: None,
            threshold: None,
        }
    }

    /// Builder-style method to add metric value.
    pub fn with_metric(mut self, value: f64) -> Self {
        self.metric_value = Some(value);
        self
    }

    /// Builder-style method to add threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Is this assessment actionable?
    pub fn is_actionable(&self) -> bool {
        self.severity.is_actionable()
    }

    /// Should trigger kill switch?
    pub fn should_kill(&self) -> bool {
        self.severity.should_kill()
    }
}

/// Trait for risk monitors.
///
/// Each monitor evaluates one aspect of risk based on the unified RiskState.
/// Monitors are stateless - all state is captured in RiskState.
///
/// # Thread Safety
///
/// Monitors are `Send + Sync` to allow for potential parallelization.
pub trait RiskMonitor: Send + Sync {
    /// Evaluate risk based on current state.
    ///
    /// # Arguments
    ///
    /// * `state` - The unified risk state snapshot
    ///
    /// # Returns
    ///
    /// A risk assessment with severity, action, and description.
    fn evaluate(&self, state: &RiskState) -> RiskAssessment;

    /// Monitor name for logging and debugging.
    fn name(&self) -> &'static str;

    /// Priority for ordering (lower = evaluated first).
    ///
    /// Default is 100. Critical monitors (loss, position) should be 0-10,
    /// informational monitors can be 200+.
    fn priority(&self) -> u32 {
        100
    }

    /// Whether this monitor is enabled.
    ///
    /// Disabled monitors are skipped during evaluation.
    fn is_enabled(&self) -> bool {
        true
    }
}

// Blanket implementation for Box<dyn RiskMonitor>
impl RiskMonitor for Box<dyn RiskMonitor> {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        (**self).evaluate(state)
    }

    fn name(&self) -> &'static str {
        (**self).name()
    }

    fn priority(&self) -> u32 {
        (**self).priority()
    }

    fn is_enabled(&self) -> bool {
        (**self).is_enabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(RiskSeverity::None < RiskSeverity::Low);
        assert!(RiskSeverity::Low < RiskSeverity::Medium);
        assert!(RiskSeverity::Medium < RiskSeverity::High);
        assert!(RiskSeverity::High < RiskSeverity::Critical);
    }

    #[test]
    fn test_severity_actionable() {
        assert!(!RiskSeverity::None.is_actionable());
        assert!(!RiskSeverity::Low.is_actionable());
        assert!(!RiskSeverity::Medium.is_actionable());
        assert!(RiskSeverity::High.is_actionable());
        assert!(RiskSeverity::Critical.is_actionable());
    }

    #[test]
    fn test_assessment_ok() {
        let assessment = RiskAssessment::ok("TestMonitor");
        assert_eq!(assessment.severity, RiskSeverity::None);
        assert!(!assessment.is_actionable());
        assert!(!assessment.should_kill());
    }

    #[test]
    fn test_assessment_critical() {
        let assessment = RiskAssessment::critical("TestMonitor", "Max loss exceeded");
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.is_actionable());
        assert!(assessment.should_kill());
    }
}
