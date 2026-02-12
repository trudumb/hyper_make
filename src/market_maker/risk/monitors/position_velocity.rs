//! Position velocity monitor for detecting rapid position accumulation or whipsaws.
//!
//! Tracks how fast position is changing relative to the max position limit.
//! Rapid position changes can indicate:
//! - Toxic flow causing one-sided fills
//! - Position whipsaws from oscillating signals
//! - Runaway accumulation from a stuck inventory skew
//!
//! The monitor reads `position_velocity_1m` from `RiskState`, which is the
//! absolute position change per minute (in contracts) normalized by max_position.

use crate::market_maker::risk::{RiskAction, RiskAssessment, RiskMonitor, RiskState};

/// Monitors position velocity to detect rapid accumulation or whipsaws.
///
/// Thresholds are expressed as fractions of max_position per minute:
/// - `warn_threshold`: warn and start widening spreads (default: 0.50 = 50%/min)
/// - `pull_threshold`: pull all quotes (default: 1.00 = 100%/min)
/// - `kill_threshold`: trigger kill switch (default: 2.00 = 200%/min)
///
/// The spread widening factor scales linearly between warn and pull thresholds,
/// from 1.5x at warn to 3.0x at pull.
pub struct PositionVelocityMonitor {
    /// Threshold (fraction of max_position per minute) for warning + spread widening
    warn_threshold: f64,
    /// Threshold for pulling all quotes
    pull_threshold: f64,
    /// Threshold for kill switch
    kill_threshold: f64,
}

impl PositionVelocityMonitor {
    /// Create a new position velocity monitor.
    ///
    /// # Arguments
    ///
    /// * `warn_threshold` - Fraction of max_position/min that triggers widening (e.g., 0.50)
    /// * `pull_threshold` - Fraction of max_position/min that triggers quote pull (e.g., 1.00)
    /// * `kill_threshold` - Fraction of max_position/min that triggers kill switch (e.g., 2.00)
    pub fn new(warn_threshold: f64, pull_threshold: f64, kill_threshold: f64) -> Self {
        Self {
            warn_threshold,
            pull_threshold,
            kill_threshold,
        }
    }

    /// Create from a single base threshold.
    ///
    /// Derives pull (2x base) and kill (4x base) from the warn threshold.
    pub fn from_base_threshold(base: f64) -> Self {
        Self::new(base, base * 2.0, base * 4.0)
    }
}

impl Default for PositionVelocityMonitor {
    fn default() -> Self {
        // 20%/min warn, 50%/min pull, 100%/min kill
        // Tightened from 50/100/200: Episode 2 accumulated at ~38%/min without even
        // triggering WARN. These thresholds catch one-sided fill cascades earlier.
        Self::new(0.20, 0.50, 1.00)
    }
}

impl RiskMonitor for PositionVelocityMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        // position_velocity_1m is abs(position_change) / max_position per minute
        // A value of 1.0 means the entire max position turned over in one minute
        let velocity = state.position_velocity_1m;

        // Kill switch at extreme velocity
        if velocity > self.kill_threshold {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Extreme position velocity: {:.2}x max_position/min > {:.2}x threshold",
                    velocity, self.kill_threshold
                ),
            )
            .with_metric(velocity)
            .with_threshold(self.kill_threshold);
        }

        // Pull quotes at high velocity
        if velocity > self.pull_threshold {
            return RiskAssessment::high(
                self.name(),
                RiskAction::PullQuotes,
                format!(
                    "Rapid position change: {:.2}x max_position/min > {:.2}x pull threshold",
                    velocity, self.pull_threshold
                ),
            )
            .with_metric(velocity)
            .with_threshold(self.pull_threshold);
        }

        // Widen spreads as velocity approaches pull threshold
        if velocity > self.warn_threshold {
            // Linear interpolation: 1.5x at warn_threshold, 3.0x at pull_threshold
            let progress =
                (velocity - self.warn_threshold) / (self.pull_threshold - self.warn_threshold);
            let widen_factor = 1.5 + progress * 1.5; // 1.5x to 3.0x

            return RiskAssessment::high(
                self.name(),
                RiskAction::WidenSpreads(widen_factor),
                format!(
                    "Position accumulating: {:.2}x max_position/min > {:.2}x warn (widening {:.1}x)",
                    velocity, self.warn_threshold, widen_factor
                ),
            )
            .with_metric(velocity)
            .with_threshold(self.warn_threshold);
        }

        RiskAssessment::ok(self.name())
    }

    fn name(&self) -> &'static str {
        "PositionVelocityMonitor"
    }

    fn priority(&self) -> u32 {
        12 // Between PositionMonitor (10) and RateLimitMonitor (15)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::risk::RiskSeverity;

    fn make_state(velocity: f64) -> RiskState {
        RiskState {
            position_velocity_1m: velocity,
            ..Default::default()
        }
    }

    #[test]
    fn test_normal_velocity_passes() {
        let monitor = PositionVelocityMonitor::default();
        let state = make_state(0.05); // 5% of max/min — well below 20% warn

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_zero_velocity_is_ok() {
        let monitor = PositionVelocityMonitor::default();
        let state = make_state(0.0);

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_warn_threshold_triggers_widen() {
        let monitor = PositionVelocityMonitor::default();
        let state = make_state(0.30); // 30% of max/min — above warn (0.20)

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::WidenSpreads(_)));

        // Verify the widening factor is in range [1.5, 3.0]
        if let RiskAction::WidenSpreads(factor) = assessment.action {
            assert!(factor >= 1.5, "widen factor {factor} should be >= 1.5");
            assert!(factor <= 3.0, "widen factor {factor} should be <= 3.0");
        }
    }

    #[test]
    fn test_warn_boundary_below() {
        let monitor = PositionVelocityMonitor::new(0.50, 1.00, 2.00);
        let state = make_state(0.49); // Just below warn threshold

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_warn_boundary_above() {
        let monitor = PositionVelocityMonitor::new(0.50, 1.00, 2.00);
        let state = make_state(0.51); // Just above warn threshold

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::WidenSpreads(_)));
    }

    #[test]
    fn test_pull_threshold_triggers_pull_quotes() {
        let monitor = PositionVelocityMonitor::default();
        let state = make_state(0.60); // 60% of max/min — above pull (0.50)

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
    }

    #[test]
    fn test_pull_boundary_above() {
        let monitor = PositionVelocityMonitor::new(0.50, 1.00, 2.00);
        let state = make_state(1.01); // Just above pull threshold

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
    }

    #[test]
    fn test_kill_threshold_triggers_kill() {
        let monitor = PositionVelocityMonitor::default();
        let state = make_state(1.20); // 120% of max/min — above kill (1.00)

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }

    #[test]
    fn test_extreme_velocity_triggers_kill() {
        let monitor = PositionVelocityMonitor::new(0.50, 1.00, 2.00);
        let state = make_state(5.0); // 5x max/min — extreme

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }

    #[test]
    fn test_widen_factor_at_warn_boundary() {
        let monitor = PositionVelocityMonitor::new(0.50, 1.00, 2.00);
        // At exactly warn threshold + epsilon, factor should be ~1.5
        let state = make_state(0.501);

        let assessment = monitor.evaluate(&state);
        if let RiskAction::WidenSpreads(factor) = assessment.action {
            assert!(
                (factor - 1.5).abs() < 0.05,
                "at warn boundary, factor {factor} should be near 1.5"
            );
        }
    }

    #[test]
    fn test_widen_factor_near_pull_boundary() {
        let monitor = PositionVelocityMonitor::new(0.50, 1.00, 2.00);
        // Near pull threshold, factor should approach 3.0
        let state = make_state(0.99);

        let assessment = monitor.evaluate(&state);
        if let RiskAction::WidenSpreads(factor) = assessment.action {
            assert!(
                (factor - 3.0).abs() < 0.1,
                "near pull boundary, factor {factor} should be near 3.0"
            );
        }
    }

    #[test]
    fn test_from_base_threshold() {
        let monitor = PositionVelocityMonitor::from_base_threshold(0.25);

        // Warn at 0.25
        let assessment = monitor.evaluate(&make_state(0.30));
        assert!(matches!(assessment.action, RiskAction::WidenSpreads(_)));

        // Pull at 0.50
        let assessment = monitor.evaluate(&make_state(0.55));
        assert!(matches!(assessment.action, RiskAction::PullQuotes));

        // Kill at 1.00
        let assessment = monitor.evaluate(&make_state(1.10));
        assert!(assessment.should_kill());
    }

    #[test]
    fn test_recovery_after_velocity_drops() {
        let monitor = PositionVelocityMonitor::default();

        // First: rapid accumulation (above pull threshold 0.50)
        let state = make_state(0.60);
        let assessment = monitor.evaluate(&state);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));

        // Then: velocity drops to normal (below warn 0.20)
        let state = make_state(0.05);
        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_metric_and_threshold_populated() {
        let monitor = PositionVelocityMonitor::default();
        let state = make_state(0.30); // Above warn (0.20), triggers WidenSpreads

        let assessment = monitor.evaluate(&state);
        assert!(assessment.metric_value.is_some());
        assert!(assessment.threshold.is_some());
        assert!((assessment.metric_value.unwrap() - 0.30).abs() < f64::EPSILON);
        assert!((assessment.threshold.unwrap() - 0.20).abs() < f64::EPSILON);
    }

    #[test]
    fn test_priority() {
        let monitor = PositionVelocityMonitor::default();
        assert_eq!(monitor.priority(), 12);
    }
}
