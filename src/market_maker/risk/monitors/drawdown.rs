//! Drawdown monitor.

use crate::market_maker::risk::{RiskAssessment, RiskMonitor, RiskState};

/// Monitors peak-to-trough drawdown.
pub struct DrawdownMonitor {
    /// Maximum allowed drawdown (fraction, 0.05 = 5%)
    max_drawdown: f64,
    /// Warning threshold (fraction of limit)
    warning_threshold: f64,
    /// Minimum peak PnL (USD) before drawdown check activates.
    /// Drawdown from peak is only meaningful when the peak represents
    /// a significant sample of fills, not spread noise.
    min_peak_for_drawdown: f64,
    /// Maximum absolute drawdown in USD before triggering critical.
    /// Safety net for small-capital scenarios where percentage check is bypassed.
    max_absolute_drawdown: f64,
}

impl DrawdownMonitor {
    /// Create a new drawdown monitor.
    ///
    /// # Arguments
    ///
    /// * `max_drawdown` - Maximum allowed drawdown fraction (e.g., 0.05 = 5%)
    pub fn new(max_drawdown: f64) -> Self {
        Self {
            max_drawdown: max_drawdown.clamp(0.0, 1.0),
            warning_threshold: 0.7,
            min_peak_for_drawdown: 1.0,
            max_absolute_drawdown: 5.0, // Conservative $5 default
        }
    }

    /// Create with custom warning threshold.
    pub fn with_warning_threshold(mut self, threshold: f64) -> Self {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set minimum peak PnL before drawdown monitoring activates.
    pub fn with_min_peak(mut self, min_peak: f64) -> Self {
        self.min_peak_for_drawdown = min_peak;
        self
    }

    /// Set maximum absolute drawdown in USD.
    ///
    /// Safety net for small-capital scenarios where percentage drawdown is
    /// bypassed because peak PnL is below `min_peak_for_drawdown`.
    pub fn with_max_absolute_drawdown(mut self, max_absolute: f64) -> Self {
        self.max_absolute_drawdown = max_absolute;
        self
    }
}

impl RiskMonitor for DrawdownMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        // No drawdown if no peak or at peak
        if state.peak_pnl <= 0.0 || state.daily_pnl >= state.peak_pnl {
            return RiskAssessment::ok(self.name());
        }

        // Absolute drawdown safety net: catches small-capital scenarios where
        // percentage check is disabled (peak < min_peak_for_drawdown).
        let absolute_drawdown = state.peak_pnl - state.daily_pnl;
        if absolute_drawdown > self.max_absolute_drawdown && absolute_drawdown > 0.0 {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Absolute drawdown ${:.2} exceeds max ${:.2}",
                    absolute_drawdown, self.max_absolute_drawdown
                ),
            )
            .with_metric(absolute_drawdown)
            .with_threshold(self.max_absolute_drawdown);
        }

        // Drawdown is meaningless when peak is spread noise (e.g., $0.02 from one fill).
        // check_daily_loss (LossMonitor) still protects against catastrophic losses.
        if state.peak_pnl < self.min_peak_for_drawdown {
            return RiskAssessment::ok(self.name());
        }

        let drawdown = state.drawdown();

        if drawdown > self.max_drawdown {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Drawdown {:.1}% exceeds limit {:.1}%",
                    drawdown * 100.0,
                    self.max_drawdown * 100.0
                ),
            )
            .with_metric(drawdown)
            .with_threshold(self.max_drawdown);
        }

        let drawdown_ratio = drawdown / self.max_drawdown;
        if drawdown_ratio > self.warning_threshold {
            return RiskAssessment::warn(
                self.name(),
                format!(
                    "Drawdown {:.1}% is {:.0}% of limit {:.1}%",
                    drawdown * 100.0,
                    drawdown_ratio * 100.0,
                    self.max_drawdown * 100.0
                ),
            )
            .with_metric(drawdown)
            .with_threshold(self.max_drawdown);
        }

        RiskAssessment::ok(self.name())
    }

    fn name(&self) -> &'static str {
        "DrawdownMonitor"
    }

    fn priority(&self) -> u32 {
        5 // High priority - drawdown is critical
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::risk::RiskSeverity;

    #[test]
    fn test_no_peak() {
        let monitor = DrawdownMonitor::new(0.05);
        let state = RiskState {
            peak_pnl: 0.0,
            daily_pnl: -100.0,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_at_peak() {
        let monitor = DrawdownMonitor::new(0.05);
        let state = RiskState {
            peak_pnl: 100.0,
            daily_pnl: 100.0,
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_small_drawdown() {
        let monitor = DrawdownMonitor::new(0.10); // 10% limit
        let state = RiskState {
            account_value: 1000.0,
            peak_pnl: 1000.0,
            daily_pnl: 980.0, // drawdown = (1000-980)/1000 = 2%
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
    }

    #[test]
    fn test_warning_drawdown() {
        let monitor = DrawdownMonitor::new(0.10).with_warning_threshold(0.7); // 10% limit
        let state = RiskState {
            account_value: 1000.0,
            peak_pnl: 1000.0,
            daily_pnl: 920.0, // drawdown = (1000-920)/1000 = 8% = 80% of 10% limit
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Medium);
    }

    #[test]
    fn test_over_limit() {
        let monitor = DrawdownMonitor::new(0.05); // 5% limit
        let state = RiskState {
            account_value: 1000.0,
            peak_pnl: 1000.0,
            daily_pnl: 900.0, // drawdown = (1000-900)/1000 = 10% > 5% limit
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }
}
