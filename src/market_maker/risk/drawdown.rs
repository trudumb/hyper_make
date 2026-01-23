//! Drawdown tracking and management.
//!
//! Tracks equity drawdown from peak and from session start.
//! Provides position sizing multipliers and pause signals based on drawdown levels.
//!
//! Defense-first: position sizes are reduced as drawdown increases.

/// Drawdown severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DrawdownLevel {
    /// Normal operation - no significant drawdown.
    Normal,

    /// Warning level - drawdown is notable but manageable.
    /// Typically > 1% drawdown.
    Warning,

    /// Critical level - significant drawdown requiring action.
    /// Typically > 2% drawdown.
    Critical,

    /// Emergency level - severe drawdown requiring immediate action.
    /// Typically > 3% drawdown, should pause trading.
    Emergency,
}

impl DrawdownLevel {
    /// Returns true if this level requires any action.
    pub fn requires_action(&self) -> bool {
        !matches!(self, DrawdownLevel::Normal)
    }

    /// Returns true if trading should be paused at this level.
    pub fn should_pause(&self) -> bool {
        matches!(self, DrawdownLevel::Emergency)
    }

    /// Returns true if positions should be reduced at this level.
    pub fn should_reduce_position(&self) -> bool {
        matches!(self, DrawdownLevel::Warning | DrawdownLevel::Critical | DrawdownLevel::Emergency)
    }
}

impl std::fmt::Display for DrawdownLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DrawdownLevel::Normal => write!(f, "Normal"),
            DrawdownLevel::Warning => write!(f, "Warning"),
            DrawdownLevel::Critical => write!(f, "Critical"),
            DrawdownLevel::Emergency => write!(f, "Emergency"),
        }
    }
}

/// Configuration for drawdown thresholds.
#[derive(Debug, Clone)]
pub struct DrawdownConfig {
    /// Threshold for warning level as a fraction (0.0 to 1.0).
    /// Example: 0.01 = 1% drawdown triggers warning.
    pub warning_threshold: f64,

    /// Threshold for critical level as a fraction.
    /// Example: 0.02 = 2% drawdown triggers critical.
    pub critical_threshold: f64,

    /// Threshold for emergency level as a fraction.
    /// Example: 0.03 = 3% drawdown triggers emergency.
    pub emergency_threshold: f64,

    /// Position size multiplier to apply at warning level.
    /// Example: 0.5 = reduce to 50% of normal size.
    pub position_reduce_at_warning: f64,

    /// Position size multiplier to apply at critical level.
    /// Example: 0.25 = reduce to 25% of normal size.
    pub position_reduce_at_critical: f64,
}

impl Default for DrawdownConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 0.01,           // 1% drawdown
            critical_threshold: 0.02,          // 2% drawdown
            emergency_threshold: 0.03,         // 3% drawdown
            position_reduce_at_warning: 0.5,   // 50% size at warning
            position_reduce_at_critical: 0.25, // 25% size at critical
        }
    }
}

impl DrawdownConfig {
    /// Create config with custom thresholds.
    pub fn new(warning: f64, critical: f64, emergency: f64) -> Self {
        Self {
            warning_threshold: warning.clamp(0.0, 1.0),
            critical_threshold: critical.clamp(warning, 1.0),
            emergency_threshold: emergency.clamp(critical, 1.0),
            ..Default::default()
        }
    }

    /// Builder method for position reduction multipliers.
    pub fn with_position_reduction(mut self, at_warning: f64, at_critical: f64) -> Self {
        self.position_reduce_at_warning = at_warning.clamp(0.0, 1.0);
        self.position_reduce_at_critical = at_critical.clamp(0.0, self.position_reduce_at_warning);
        self
    }

    /// Validate that thresholds are consistent.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.warning_threshold <= 0.0 {
            return Err("warning_threshold must be positive");
        }
        if self.critical_threshold <= self.warning_threshold {
            return Err("critical_threshold must be greater than warning_threshold");
        }
        if self.emergency_threshold <= self.critical_threshold {
            return Err("emergency_threshold must be greater than critical_threshold");
        }
        if self.position_reduce_at_warning <= 0.0 || self.position_reduce_at_warning > 1.0 {
            return Err("position_reduce_at_warning must be in (0, 1]");
        }
        if self.position_reduce_at_critical > self.position_reduce_at_warning {
            return Err("position_reduce_at_critical must not exceed position_reduce_at_warning");
        }
        Ok(())
    }
}

/// Drawdown tracker for monitoring equity changes.
///
/// Tracks both peak-to-trough drawdown and drawdown from session start.
#[derive(Debug, Clone)]
pub struct DrawdownTracker {
    /// Configuration
    config: DrawdownConfig,

    /// Peak equity observed (high-water mark)
    peak_equity: f64,

    /// Current equity
    current_equity: f64,

    /// Equity at session start
    session_start_equity: f64,
}

impl DrawdownTracker {
    /// Create a new drawdown tracker.
    ///
    /// # Arguments
    /// - `config`: Drawdown configuration
    /// - `initial_equity`: Starting equity value
    pub fn new(config: DrawdownConfig, initial_equity: f64) -> Self {
        Self {
            config,
            peak_equity: initial_equity,
            current_equity: initial_equity,
            session_start_equity: initial_equity,
        }
    }

    /// Create with default config.
    pub fn with_defaults(initial_equity: f64) -> Self {
        Self::new(DrawdownConfig::default(), initial_equity)
    }

    /// Get the configuration.
    pub fn config(&self) -> &DrawdownConfig {
        &self.config
    }

    /// Get current equity.
    pub fn current_equity(&self) -> f64 {
        self.current_equity
    }

    /// Get peak equity (high-water mark).
    pub fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    /// Get session start equity.
    pub fn session_start_equity(&self) -> f64 {
        self.session_start_equity
    }

    /// Update the current equity value.
    ///
    /// Automatically updates peak if current exceeds it.
    pub fn update_equity(&mut self, equity: f64) {
        self.current_equity = equity;

        // Update peak if new high
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
    }

    /// Calculate current drawdown from peak as a fraction.
    ///
    /// Returns 0.0 if at or above peak, positive fraction otherwise.
    /// Example: 0.05 means 5% below peak.
    pub fn drawdown_pct(&self) -> f64 {
        if self.peak_equity <= 0.0 || self.current_equity >= self.peak_equity {
            0.0
        } else {
            (self.peak_equity - self.current_equity) / self.peak_equity
        }
    }

    /// Calculate drawdown from session start as a fraction.
    ///
    /// Can be negative if currently above session start (in profit).
    /// Positive values indicate loss from session start.
    pub fn drawdown_from_session_start(&self) -> f64 {
        if self.session_start_equity <= 0.0 {
            0.0
        } else {
            (self.session_start_equity - self.current_equity) / self.session_start_equity
        }
    }

    /// Get the current drawdown level based on peak drawdown.
    pub fn level(&self) -> DrawdownLevel {
        let dd = self.drawdown_pct();

        if dd >= self.config.emergency_threshold {
            DrawdownLevel::Emergency
        } else if dd >= self.config.critical_threshold {
            DrawdownLevel::Critical
        } else if dd >= self.config.warning_threshold {
            DrawdownLevel::Warning
        } else {
            DrawdownLevel::Normal
        }
    }

    /// Get the position size multiplier based on current drawdown.
    ///
    /// Returns:
    /// - 1.0 at Normal level (full size)
    /// - `position_reduce_at_warning` at Warning level
    /// - `position_reduce_at_critical` at Critical level
    /// - 0.0 at Emergency level (no new positions)
    pub fn position_multiplier(&self) -> f64 {
        match self.level() {
            DrawdownLevel::Normal => 1.0,
            DrawdownLevel::Warning => self.config.position_reduce_at_warning,
            DrawdownLevel::Critical => self.config.position_reduce_at_critical,
            DrawdownLevel::Emergency => 0.0,
        }
    }

    /// Returns true if trading should be paused.
    pub fn should_pause(&self) -> bool {
        self.level().should_pause()
    }

    /// Reset session tracking with new starting equity.
    ///
    /// Call this at the start of each trading session.
    /// Does NOT reset the peak - use `reset_peak` for that.
    pub fn reset_session(&mut self, equity: f64) {
        self.session_start_equity = equity;
        self.current_equity = equity;
    }

    /// Reset peak to current equity.
    ///
    /// Use this sparingly - typically only after investigating
    /// and resolving the cause of drawdown.
    pub fn reset_peak(&mut self) {
        self.peak_equity = self.current_equity;
    }

    /// Reset everything to a new starting point.
    pub fn reset_all(&mut self, equity: f64) {
        self.peak_equity = equity;
        self.current_equity = equity;
        self.session_start_equity = equity;
    }

    /// Get a summary of current drawdown state.
    pub fn summary(&self) -> DrawdownSummary {
        DrawdownSummary {
            current_equity: self.current_equity,
            peak_equity: self.peak_equity,
            session_start_equity: self.session_start_equity,
            drawdown_pct: self.drawdown_pct(),
            drawdown_from_session: self.drawdown_from_session_start(),
            level: self.level(),
            position_multiplier: self.position_multiplier(),
        }
    }
}

// Ensure Send + Sync for thread safety
unsafe impl Send for DrawdownTracker {}
unsafe impl Sync for DrawdownTracker {}

/// Summary of drawdown state for reporting.
#[derive(Debug, Clone)]
pub struct DrawdownSummary {
    /// Current equity value
    pub current_equity: f64,
    /// Peak equity (high-water mark)
    pub peak_equity: f64,
    /// Equity at session start
    pub session_start_equity: f64,
    /// Drawdown from peak as fraction
    pub drawdown_pct: f64,
    /// Drawdown from session start as fraction
    pub drawdown_from_session: f64,
    /// Current drawdown level
    pub level: DrawdownLevel,
    /// Position size multiplier
    pub position_multiplier: f64,
}

impl std::fmt::Display for DrawdownSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Drawdown: {:.2}% from peak (level: {}), {:.2}% from session, position mult: {:.0}%",
            self.drawdown_pct * 100.0,
            self.level,
            self.drawdown_from_session * 100.0,
            self.position_multiplier * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DrawdownConfig::default();
        assert_eq!(config.warning_threshold, 0.01);
        assert_eq!(config.critical_threshold, 0.02);
        assert_eq!(config.emergency_threshold, 0.03);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let valid = DrawdownConfig::new(0.01, 0.02, 0.03);
        assert!(valid.validate().is_ok());

        // Invalid: warning >= critical
        let invalid = DrawdownConfig {
            warning_threshold: 0.02,
            critical_threshold: 0.02,
            emergency_threshold: 0.03,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        // Invalid: critical >= emergency
        let invalid2 = DrawdownConfig {
            warning_threshold: 0.01,
            critical_threshold: 0.03,
            emergency_threshold: 0.03,
            ..Default::default()
        };
        assert!(invalid2.validate().is_err());
    }

    #[test]
    fn test_initial_state() {
        let tracker = DrawdownTracker::with_defaults(10_000.0);

        assert_eq!(tracker.current_equity(), 10_000.0);
        assert_eq!(tracker.peak_equity(), 10_000.0);
        assert_eq!(tracker.session_start_equity(), 10_000.0);
        assert!((tracker.drawdown_pct() - 0.0).abs() < f64::EPSILON);
        assert_eq!(tracker.level(), DrawdownLevel::Normal);
        assert!((tracker.position_multiplier() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_equity_update_increases_peak() {
        let mut tracker = DrawdownTracker::with_defaults(10_000.0);

        // Increase equity - should update peak
        tracker.update_equity(11_000.0);
        assert_eq!(tracker.peak_equity(), 11_000.0);
        assert_eq!(tracker.current_equity(), 11_000.0);

        // Decrease equity - peak should stay
        tracker.update_equity(10_500.0);
        assert_eq!(tracker.peak_equity(), 11_000.0);
        assert_eq!(tracker.current_equity(), 10_500.0);
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut tracker = DrawdownTracker::with_defaults(10_000.0);

        // Go to peak then drawdown
        tracker.update_equity(10_000.0);
        assert!((tracker.drawdown_pct() - 0.0).abs() < f64::EPSILON);

        // 5% drawdown
        tracker.update_equity(9_500.0);
        assert!((tracker.drawdown_pct() - 0.05).abs() < 0.001);

        // New peak, then 10% drawdown
        tracker.update_equity(10_000.0);
        tracker.update_equity(9_000.0);
        assert!((tracker.drawdown_pct() - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_drawdown_levels() {
        let config = DrawdownConfig::new(0.01, 0.02, 0.03);
        let mut tracker = DrawdownTracker::new(config, 10_000.0);

        // Normal (0.5% drawdown)
        tracker.update_equity(9_950.0);
        assert_eq!(tracker.level(), DrawdownLevel::Normal);

        // Warning (1.5% drawdown)
        tracker.update_equity(9_850.0);
        assert_eq!(tracker.level(), DrawdownLevel::Warning);

        // Critical (2.5% drawdown)
        tracker.update_equity(9_750.0);
        assert_eq!(tracker.level(), DrawdownLevel::Critical);

        // Emergency (3.5% drawdown)
        tracker.update_equity(9_650.0);
        assert_eq!(tracker.level(), DrawdownLevel::Emergency);
    }

    #[test]
    fn test_position_multiplier() {
        let config = DrawdownConfig::default()
            .with_position_reduction(0.5, 0.25);
        let mut tracker = DrawdownTracker::new(config, 10_000.0);

        // Normal - full size
        assert!((tracker.position_multiplier() - 1.0).abs() < f64::EPSILON);

        // Warning - 50%
        tracker.update_equity(9_850.0); // 1.5% drawdown
        assert!((tracker.position_multiplier() - 0.5).abs() < f64::EPSILON);

        // Critical - 25%
        tracker.update_equity(9_750.0); // 2.5% drawdown
        assert!((tracker.position_multiplier() - 0.25).abs() < f64::EPSILON);

        // Emergency - 0%
        tracker.update_equity(9_650.0); // 3.5% drawdown
        assert!((tracker.position_multiplier() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_should_pause() {
        let mut tracker = DrawdownTracker::with_defaults(10_000.0);

        // Normal - don't pause
        assert!(!tracker.should_pause());

        // Emergency - pause
        tracker.update_equity(9_600.0); // 4% drawdown
        assert!(tracker.should_pause());
    }

    #[test]
    fn test_session_drawdown() {
        let mut tracker = DrawdownTracker::with_defaults(10_000.0);

        // Profit from session start
        tracker.update_equity(10_500.0);
        assert!(tracker.drawdown_from_session_start() < 0.0); // Negative = profit

        // Loss from session start
        tracker.update_equity(9_500.0);
        assert!((tracker.drawdown_from_session_start() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_reset_session() {
        let mut tracker = DrawdownTracker::with_defaults(10_000.0);

        // Some trading
        tracker.update_equity(11_000.0);
        tracker.update_equity(10_500.0);

        // Reset session with new start
        tracker.reset_session(10_500.0);

        assert_eq!(tracker.session_start_equity(), 10_500.0);
        assert_eq!(tracker.current_equity(), 10_500.0);
        // Peak should NOT be reset
        assert_eq!(tracker.peak_equity(), 11_000.0);
    }

    #[test]
    fn test_reset_peak() {
        let mut tracker = DrawdownTracker::with_defaults(10_000.0);

        // Go to high peak then drawdown
        tracker.update_equity(12_000.0);
        tracker.update_equity(10_000.0);

        // Big drawdown showing
        assert!(tracker.drawdown_pct() > 0.15);

        // Reset peak
        tracker.reset_peak();
        assert_eq!(tracker.peak_equity(), 10_000.0);
        assert!((tracker.drawdown_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset_all() {
        let mut tracker = DrawdownTracker::with_defaults(10_000.0);

        tracker.update_equity(12_000.0);
        tracker.update_equity(11_000.0);

        tracker.reset_all(15_000.0);

        assert_eq!(tracker.peak_equity(), 15_000.0);
        assert_eq!(tracker.current_equity(), 15_000.0);
        assert_eq!(tracker.session_start_equity(), 15_000.0);
    }

    #[test]
    fn test_drawdown_level_ordering() {
        assert!(DrawdownLevel::Normal < DrawdownLevel::Warning);
        assert!(DrawdownLevel::Warning < DrawdownLevel::Critical);
        assert!(DrawdownLevel::Critical < DrawdownLevel::Emergency);
    }

    #[test]
    fn test_drawdown_level_actions() {
        assert!(!DrawdownLevel::Normal.requires_action());
        assert!(DrawdownLevel::Warning.requires_action());
        assert!(DrawdownLevel::Critical.requires_action());
        assert!(DrawdownLevel::Emergency.requires_action());

        assert!(!DrawdownLevel::Normal.should_pause());
        assert!(!DrawdownLevel::Warning.should_pause());
        assert!(!DrawdownLevel::Critical.should_pause());
        assert!(DrawdownLevel::Emergency.should_pause());

        assert!(!DrawdownLevel::Normal.should_reduce_position());
        assert!(DrawdownLevel::Warning.should_reduce_position());
        assert!(DrawdownLevel::Critical.should_reduce_position());
        assert!(DrawdownLevel::Emergency.should_reduce_position());
    }

    #[test]
    fn test_summary() {
        let mut tracker = DrawdownTracker::with_defaults(10_000.0);
        tracker.update_equity(9_850.0); // 1.5% drawdown

        let summary = tracker.summary();
        assert_eq!(summary.current_equity, 9_850.0);
        assert_eq!(summary.peak_equity, 10_000.0);
        assert!((summary.drawdown_pct - 0.015).abs() < 0.001);
        assert_eq!(summary.level, DrawdownLevel::Warning);
        assert!((summary.position_multiplier - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_zero_equity_edge_cases() {
        let mut tracker = DrawdownTracker::with_defaults(0.0);

        // Should not panic on zero equity
        assert!((tracker.drawdown_pct() - 0.0).abs() < f64::EPSILON);
        assert!((tracker.drawdown_from_session_start() - 0.0).abs() < f64::EPSILON);

        tracker.update_equity(100.0);
        // Peak is now 100, session start still 0
        assert!((tracker.drawdown_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_negative_equity_handling() {
        let mut tracker = DrawdownTracker::with_defaults(10_000.0);

        // Negative equity (extreme loss scenario)
        tracker.update_equity(-1_000.0);

        // Should still calculate valid drawdown (though extreme)
        let dd = tracker.drawdown_pct();
        assert!(dd > 1.0); // More than 100% drawdown
    }

    #[test]
    fn test_display_implementations() {
        assert_eq!(format!("{}", DrawdownLevel::Warning), "Warning");
        assert_eq!(format!("{}", DrawdownLevel::Emergency), "Emergency");

        let tracker = DrawdownTracker::with_defaults(10_000.0);
        let summary = tracker.summary();
        let display = format!("{}", summary);
        assert!(display.contains("Drawdown:"));
        assert!(display.contains("from peak"));
    }
}
