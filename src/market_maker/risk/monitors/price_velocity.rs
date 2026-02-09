//! Price velocity monitor for flash crash protection.

use crate::market_maker::risk::{RiskAction, RiskAssessment, RiskMonitor, RiskState};

/// Monitors price velocity (abs(mid_delta / mid) per second) to detect flash crashes.
///
/// Flash crashes can wipe out a market maker in seconds. This monitor triggers
/// PullQuotes when price is moving too fast, giving us time to reassess.
///
/// Default threshold: 5% per second (0.05). For BTC at $100k, that's $5k/s.
pub struct PriceVelocityMonitor {
    /// Velocity threshold above which quotes are pulled (fraction per second)
    pull_threshold: f64,
    /// Velocity threshold for kill switch (fraction per second)
    kill_threshold: f64,
}

impl PriceVelocityMonitor {
    /// Create a new price velocity monitor.
    ///
    /// # Arguments
    ///
    /// * `pull_threshold` - Velocity (fraction/s) at which to pull quotes
    /// * `kill_threshold` - Velocity (fraction/s) at which to trigger kill switch
    pub fn new(pull_threshold: f64, kill_threshold: f64) -> Self {
        Self {
            pull_threshold,
            kill_threshold,
        }
    }
}

impl Default for PriceVelocityMonitor {
    fn default() -> Self {
        Self::new(0.05, 0.15) // Pull at 5%/s, kill at 15%/s
    }
}

impl RiskMonitor for PriceVelocityMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        let velocity = state.price_velocity_1s;

        // Kill switch at extreme velocity
        if velocity > self.kill_threshold {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Extreme price velocity: {:.4}/s > {:.4}/s threshold",
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
                    "Flash crash detected: price velocity {:.4}/s > {:.4}/s threshold",
                    velocity, self.pull_threshold
                ),
            )
            .with_metric(velocity)
            .with_threshold(self.pull_threshold);
        }

        RiskAssessment::ok(self.name())
    }

    fn name(&self) -> &'static str {
        "PriceVelocityMonitor"
    }

    fn priority(&self) -> u32 {
        10 // High priority — flash crashes need immediate response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::risk::RiskSeverity;

    fn make_state(velocity: f64) -> RiskState {
        RiskState {
            price_velocity_1s: velocity,
            ..Default::default()
        }
    }

    #[test]
    fn test_normal_velocity_passes() {
        let monitor = PriceVelocityMonitor::default();
        let state = make_state(0.001); // 0.1%/s — normal

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_threshold_boundary_below() {
        let monitor = PriceVelocityMonitor::new(0.05, 0.15);
        let state = make_state(0.049); // Just below pull threshold

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_threshold_boundary_above() {
        let monitor = PriceVelocityMonitor::new(0.05, 0.15);
        let state = make_state(0.051); // Just above pull threshold

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
    }

    #[test]
    fn test_flash_crash_triggers_pull() {
        let monitor = PriceVelocityMonitor::default();
        let state = make_state(0.08); // 8%/s — flash crash

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
        assert!(assessment.metric_value.unwrap() > 0.0);
        assert!(assessment.threshold.unwrap() > 0.0);
    }

    #[test]
    fn test_extreme_velocity_triggers_kill() {
        let monitor = PriceVelocityMonitor::new(0.05, 0.15);
        let state = make_state(0.20); // 20%/s — extreme

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }

    #[test]
    fn test_negative_velocity_is_abs() {
        // price_velocity_1s is already absolute (computed as abs outside monitor)
        // but test that the monitor correctly handles large positive values
        // representing either a crash (price drop) or rally (price spike)
        let monitor = PriceVelocityMonitor::default();

        // A sharp rally should also trigger — we care about speed, not direction
        let state = make_state(0.07); // 7%/s rally
        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
    }

    #[test]
    fn test_recovery_after_velocity_drops() {
        let monitor = PriceVelocityMonitor::default();

        // First: flash crash
        let state = make_state(0.08);
        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);

        // Then: velocity drops to normal
        let state = make_state(0.01);
        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_zero_velocity_is_ok() {
        let monitor = PriceVelocityMonitor::default();
        let state = make_state(0.0);

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }
}
