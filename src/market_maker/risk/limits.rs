//! Risk limits and checking.
//!
//! Provides position and order size limit checking with soft/hard thresholds.
//! Defense-first: when uncertain, suggest tighter limits.

/// Configuration for risk limits.
///
/// All limits are expressed as positive values (notional in USD, sizes in contracts/coins).
#[derive(Debug, Clone)]
pub struct RiskLimits {
    /// Hard limit for position notional (USD) - cancel all quotes if breached.
    /// This is the absolute maximum allowed position value.
    pub max_position_notional: f64,

    /// Soft limit for position notional (USD) - start reducing quote sizes.
    /// Typically 50-80% of max_position_notional.
    pub soft_position_threshold: f64,

    /// Maximum allowed drawdown as a fraction (0.0 to 1.0).
    /// Example: 0.02 = 2% daily drawdown limit.
    pub max_drawdown_pct: f64,

    /// Maximum position as percentage of total open interest.
    /// Prevents concentration risk. Example: 0.01 = 1% of OI.
    pub max_concentration_pct: f64,

    /// Maximum single order size (contracts/coins).
    /// Prevents fat-finger errors and limits per-order risk.
    pub max_order_size: f64,

    /// Minimum spread in basis points - never quote tighter than this.
    /// Defense mechanism during volatile conditions.
    pub min_spread_bps: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position_notional: 10_000.0,  // $10k hard limit
            soft_position_threshold: 7_000.0, // $7k soft limit (70%)
            max_drawdown_pct: 0.02,           // 2% daily drawdown
            max_concentration_pct: 0.01,      // 1% of open interest
            max_order_size: 1.0,              // 1 contract default
            min_spread_bps: 5.0,              // 5 bps minimum spread
        }
    }
}

impl RiskLimits {
    /// Create new risk limits with custom position limits.
    pub fn new(max_notional: f64, soft_threshold: f64) -> Self {
        Self {
            max_position_notional: max_notional,
            soft_position_threshold: soft_threshold,
            ..Default::default()
        }
    }

    /// Builder method to set max drawdown.
    pub fn with_max_drawdown(mut self, pct: f64) -> Self {
        self.max_drawdown_pct = pct.clamp(0.0, 1.0);
        self
    }

    /// Builder method to set concentration limit.
    pub fn with_max_concentration(mut self, pct: f64) -> Self {
        self.max_concentration_pct = pct.clamp(0.0, 1.0);
        self
    }

    /// Builder method to set max order size.
    pub fn with_max_order_size(mut self, size: f64) -> Self {
        self.max_order_size = size.max(0.0);
        self
    }

    /// Builder method to set minimum spread.
    pub fn with_min_spread_bps(mut self, bps: f64) -> Self {
        self.min_spread_bps = bps.max(0.0);
        self
    }

    /// Validate that limits are internally consistent.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_position_notional <= 0.0 {
            return Err("max_position_notional must be positive");
        }
        if self.soft_position_threshold <= 0.0 {
            return Err("soft_position_threshold must be positive");
        }
        if self.soft_position_threshold > self.max_position_notional {
            return Err("soft_position_threshold must not exceed max_position_notional");
        }
        if self.max_order_size <= 0.0 {
            return Err("max_order_size must be positive");
        }
        Ok(())
    }
}

/// Result of a risk check.
#[derive(Debug, Clone, PartialEq)]
pub enum RiskCheckResult {
    /// All checks passed - proceed normally.
    Ok,

    /// Soft limit breached - reduce exposure but don't cancel.
    SoftLimitBreached { current: f64, limit: f64 },

    /// Hard limit breached - cancel all quotes immediately.
    HardLimitBreached { current: f64, limit: f64 },
}

impl RiskCheckResult {
    /// Returns true if this result indicates a problem.
    pub fn is_breach(&self) -> bool {
        !matches!(self, RiskCheckResult::Ok)
    }

    /// Returns true if this is a hard limit breach requiring immediate action.
    pub fn is_hard_breach(&self) -> bool {
        matches!(self, RiskCheckResult::HardLimitBreached { .. })
    }

    /// Returns true if this is a soft limit breach requiring position reduction.
    pub fn is_soft_breach(&self) -> bool {
        matches!(self, RiskCheckResult::SoftLimitBreached { .. })
    }
}

/// Risk checker that validates positions and orders against limits.
///
/// Thread-safe: all methods take `&self` and limits are immutable after construction.
#[derive(Debug, Clone)]
pub struct RiskChecker {
    limits: RiskLimits,
}

impl RiskChecker {
    /// Create a new risk checker with the given limits.
    pub fn new(limits: RiskLimits) -> Self {
        Self { limits }
    }

    /// Get the configured limits.
    pub fn limits(&self) -> &RiskLimits {
        &self.limits
    }

    /// Check if a position notional value is within limits.
    ///
    /// Returns:
    /// - `Ok` if below soft threshold
    /// - `SoftLimitBreached` if between soft and hard limits
    /// - `HardLimitBreached` if above hard limit
    pub fn check_position(&self, position_notional: f64) -> RiskCheckResult {
        let abs_notional = position_notional.abs();

        if abs_notional > self.limits.max_position_notional {
            RiskCheckResult::HardLimitBreached {
                current: abs_notional,
                limit: self.limits.max_position_notional,
            }
        } else if abs_notional > self.limits.soft_position_threshold {
            RiskCheckResult::SoftLimitBreached {
                current: abs_notional,
                limit: self.limits.soft_position_threshold,
            }
        } else {
            RiskCheckResult::Ok
        }
    }

    /// Check if an order size is within limits.
    ///
    /// Returns:
    /// - `Ok` if order size is acceptable
    /// - `HardLimitBreached` if order size exceeds maximum
    pub fn check_order_size(&self, size: f64) -> RiskCheckResult {
        let abs_size = size.abs();

        if abs_size > self.limits.max_order_size {
            RiskCheckResult::HardLimitBreached {
                current: abs_size,
                limit: self.limits.max_order_size,
            }
        } else {
            RiskCheckResult::Ok
        }
    }

    /// Calculate suggested size multiplier based on position notional.
    ///
    /// Returns a value between 0.0 and 1.0:
    /// - 1.0 when below soft threshold (full size)
    /// - Linearly decreasing from 1.0 to 0.0 between soft and hard limits
    /// - 0.0 when at or above hard limit
    ///
    /// This allows graceful position reduction as limits are approached.
    pub fn suggested_size_multiplier(&self, position_notional: f64) -> f64 {
        let abs_notional = position_notional.abs();

        if abs_notional <= self.limits.soft_position_threshold {
            // Below soft limit - full size
            1.0
        } else if abs_notional >= self.limits.max_position_notional {
            // At or above hard limit - no new positions
            0.0
        } else {
            // Between soft and hard limits - linear interpolation
            let range = self.limits.max_position_notional - self.limits.soft_position_threshold;
            let excess = abs_notional - self.limits.soft_position_threshold;
            1.0 - (excess / range)
        }
    }

    /// Check concentration against open interest.
    ///
    /// # Arguments
    /// - `position_size`: Current position size (contracts)
    /// - `open_interest`: Total market open interest (contracts)
    ///
    /// Returns:
    /// - `Ok` if concentration is acceptable
    /// - `SoftLimitBreached` if approaching limit (80%+)
    /// - `HardLimitBreached` if at or above limit
    pub fn check_concentration(&self, position_size: f64, open_interest: f64) -> RiskCheckResult {
        if open_interest <= 0.0 {
            // No OI data - be defensive, assume OK but caller should handle
            return RiskCheckResult::Ok;
        }

        let concentration = position_size.abs() / open_interest;

        if concentration >= self.limits.max_concentration_pct {
            RiskCheckResult::HardLimitBreached {
                current: concentration,
                limit: self.limits.max_concentration_pct,
            }
        } else if concentration >= self.limits.max_concentration_pct * 0.8 {
            RiskCheckResult::SoftLimitBreached {
                current: concentration,
                limit: self.limits.max_concentration_pct,
            }
        } else {
            RiskCheckResult::Ok
        }
    }

    /// Check if a proposed spread is wide enough.
    ///
    /// # Arguments
    /// - `spread_bps`: Proposed spread in basis points
    ///
    /// Returns the minimum of proposed spread and min_spread_bps.
    /// This enforces the minimum spread floor.
    pub fn enforce_min_spread(&self, spread_bps: f64) -> f64 {
        spread_bps.max(self.limits.min_spread_bps)
    }
}

// Ensure Send + Sync for thread safety
unsafe impl Send for RiskChecker {}
unsafe impl Sync for RiskChecker {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_limits() {
        let limits = RiskLimits::default();
        assert_eq!(limits.max_position_notional, 10_000.0);
        assert_eq!(limits.soft_position_threshold, 7_000.0);
        assert_eq!(limits.max_drawdown_pct, 0.02);
        assert!(limits.validate().is_ok());
    }

    #[test]
    fn test_limits_validation() {
        // Invalid: soft > hard
        let invalid = RiskLimits {
            soft_position_threshold: 15_000.0,
            max_position_notional: 10_000.0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        // Invalid: zero max
        let invalid2 = RiskLimits {
            max_position_notional: 0.0,
            ..Default::default()
        };
        assert!(invalid2.validate().is_err());
    }

    #[test]
    fn test_check_position_ok() {
        let checker = RiskChecker::new(RiskLimits::default());
        let result = checker.check_position(5_000.0);
        assert_eq!(result, RiskCheckResult::Ok);
    }

    #[test]
    fn test_check_position_soft_breach() {
        let checker = RiskChecker::new(RiskLimits::default());
        let result = checker.check_position(8_000.0);
        assert!(result.is_soft_breach());
        if let RiskCheckResult::SoftLimitBreached { current, limit } = result {
            assert_eq!(current, 8_000.0);
            assert_eq!(limit, 7_000.0);
        }
    }

    #[test]
    fn test_check_position_hard_breach() {
        let checker = RiskChecker::new(RiskLimits::default());
        let result = checker.check_position(12_000.0);
        assert!(result.is_hard_breach());
        if let RiskCheckResult::HardLimitBreached { current, limit } = result {
            assert_eq!(current, 12_000.0);
            assert_eq!(limit, 10_000.0);
        }
    }

    #[test]
    fn test_check_position_negative() {
        // Negative positions should use absolute value
        let checker = RiskChecker::new(RiskLimits::default());
        let result = checker.check_position(-12_000.0);
        assert!(result.is_hard_breach());
    }

    #[test]
    fn test_check_order_size() {
        let checker = RiskChecker::new(RiskLimits::default());
        assert_eq!(checker.check_order_size(0.5), RiskCheckResult::Ok);
        assert!(checker.check_order_size(1.5).is_hard_breach());
    }

    #[test]
    fn test_suggested_size_multiplier() {
        let limits = RiskLimits::new(10_000.0, 5_000.0);
        let checker = RiskChecker::new(limits);

        // Below soft limit - full size
        assert!((checker.suggested_size_multiplier(3_000.0) - 1.0).abs() < f64::EPSILON);

        // At soft limit - still full size
        assert!((checker.suggested_size_multiplier(5_000.0) - 1.0).abs() < f64::EPSILON);

        // Midway between soft and hard - 50%
        assert!((checker.suggested_size_multiplier(7_500.0) - 0.5).abs() < f64::EPSILON);

        // At hard limit - zero
        assert!((checker.suggested_size_multiplier(10_000.0) - 0.0).abs() < f64::EPSILON);

        // Above hard limit - still zero
        assert!((checker.suggested_size_multiplier(15_000.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_concentration_check() {
        let limits = RiskLimits::default().with_max_concentration(0.01); // 1%
        let checker = RiskChecker::new(limits);

        // 0.5% of OI - OK
        assert_eq!(
            checker.check_concentration(50.0, 10_000.0),
            RiskCheckResult::Ok
        );

        // 0.85% of OI - soft breach (>80% of limit)
        assert!(checker.check_concentration(85.0, 10_000.0).is_soft_breach());

        // 1.5% of OI - hard breach
        assert!(checker
            .check_concentration(150.0, 10_000.0)
            .is_hard_breach());

        // Zero OI - defensive OK
        assert_eq!(checker.check_concentration(100.0, 0.0), RiskCheckResult::Ok);
    }

    #[test]
    fn test_enforce_min_spread() {
        let limits = RiskLimits::default().with_min_spread_bps(5.0);
        let checker = RiskChecker::new(limits);

        // Spread above minimum - unchanged
        assert!((checker.enforce_min_spread(10.0) - 10.0).abs() < f64::EPSILON);

        // Spread below minimum - raised to minimum
        assert!((checker.enforce_min_spread(3.0) - 5.0).abs() < f64::EPSILON);

        // Spread at minimum - unchanged
        assert!((checker.enforce_min_spread(5.0) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_risk_check_result_methods() {
        assert!(!RiskCheckResult::Ok.is_breach());
        assert!(!RiskCheckResult::Ok.is_hard_breach());
        assert!(!RiskCheckResult::Ok.is_soft_breach());

        let soft = RiskCheckResult::SoftLimitBreached {
            current: 1.0,
            limit: 0.5,
        };
        assert!(soft.is_breach());
        assert!(!soft.is_hard_breach());
        assert!(soft.is_soft_breach());

        let hard = RiskCheckResult::HardLimitBreached {
            current: 1.0,
            limit: 0.5,
        };
        assert!(hard.is_breach());
        assert!(hard.is_hard_breach());
        assert!(!hard.is_soft_breach());
    }
}
