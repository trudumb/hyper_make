//! Margin-Aware Position Sizing
//!
//! Ensures orders don't exceed available margin or leverage limits.
//!
//! Key features:
//! - **Available Margin Tracking**: Monitors account margin for sizing decisions
//! - **Leverage Limits**: Respects configured maximum leverage
//! - **Position Limits**: Enforces notional position limits
//! - **Order Size Adjustment**: Dynamically adjusts order sizes to fit within margin

use std::time::{Duration, Instant};

/// Configuration for margin-aware position sizing.
#[derive(Debug, Clone)]
pub struct MarginConfig {
    /// Maximum leverage to use (e.g., 5.0 = 5x)
    pub max_leverage: f64,
    /// Buffer factor for margin (0.8 = use only 80% of available)
    pub margin_buffer: f64,
    /// Maximum notional position value in USD
    pub max_notional_position: f64,
    /// Maximum single order notional value in USD
    pub max_order_notional: f64,
    /// Minimum margin refresh interval
    pub refresh_interval: Duration,
}

impl Default for MarginConfig {
    fn default() -> Self {
        Self {
            max_leverage: 3.0,                // Conservative 3x max leverage
            margin_buffer: 0.8,               // Use 80% of available margin
            max_notional_position: 100_000.0, // $100k max position
            max_order_notional: 10_000.0,     // $10k max per order
            refresh_interval: Duration::from_secs(10),
        }
    }
}

/// Current margin state.
#[derive(Debug, Clone, Default)]
pub struct MarginState {
    /// Account equity/value
    pub account_value: f64,
    /// Total margin currently used
    pub margin_used: f64,
    /// Total notional position value
    pub total_notional: f64,
    /// Available margin for new positions
    pub available_margin: f64,
    /// Current effective leverage
    pub current_leverage: f64,
    /// Last update timestamp
    pub last_updated: Option<Instant>,
}

impl MarginState {
    /// Create from user state response values.
    ///
    /// # Arguments
    /// - `account_value`: Total account value in USD
    /// - `margin_used`: Total margin currently in use
    /// - `total_notional`: Total notional position value
    pub fn from_values(account_value: f64, margin_used: f64, total_notional: f64) -> Self {
        let available_margin = (account_value - margin_used).max(0.0);
        let current_leverage = if account_value > 0.0 {
            total_notional / account_value
        } else {
            0.0
        };

        Self {
            account_value,
            margin_used,
            total_notional,
            available_margin,
            current_leverage,
            last_updated: Some(Instant::now()),
        }
    }

    /// Check if margin state is stale.
    pub fn is_stale(&self, max_age: Duration) -> bool {
        match self.last_updated {
            Some(t) => t.elapsed() > max_age,
            None => true,
        }
    }
}

/// Sizing result with margin-adjusted values.
#[derive(Debug, Clone)]
pub struct SizingResult {
    /// Original requested size
    pub requested_size: f64,
    /// Adjusted size after margin constraints
    pub adjusted_size: f64,
    /// Whether size was reduced due to margin
    pub was_constrained: bool,
    /// Reason for constraint (if any)
    pub constraint_reason: Option<String>,
}

/// Margin-aware position sizer.
///
/// Adjusts position sizes based on:
/// - Available margin
/// - Leverage limits
/// - Notional position limits
pub struct MarginAwareSizer {
    config: MarginConfig,
    state: MarginState,
}

impl MarginAwareSizer {
    /// Create a new margin-aware sizer.
    pub fn new(config: MarginConfig) -> Self {
        Self {
            config,
            state: MarginState::default(),
        }
    }

    /// Update margin state from exchange data.
    ///
    /// # Arguments
    /// - `account_value`: Total account value in USD
    /// - `margin_used`: Total margin currently in use
    /// - `total_notional`: Total notional position value
    pub fn update_state(&mut self, account_value: f64, margin_used: f64, total_notional: f64) {
        self.state = MarginState::from_values(account_value, margin_used, total_notional);
    }

    /// Check if margin state needs refresh.
    pub fn needs_refresh(&self) -> bool {
        self.state.is_stale(self.config.refresh_interval)
    }

    /// Get current margin state.
    pub fn state(&self) -> &MarginState {
        &self.state
    }

    /// Calculate maximum order size based on margin constraints.
    ///
    /// # Arguments
    /// - `price`: Current price of the asset
    /// - `is_increasing_position`: Whether this order increases position size
    ///
    /// # Returns
    /// Maximum size that can be ordered while respecting margin limits
    pub fn max_order_size(&self, price: f64, is_increasing_position: bool) -> f64 {
        if price <= 0.0 {
            return 0.0;
        }

        // Start with max order notional limit
        let mut max_notional = self.config.max_order_notional;

        // If increasing position, also consider margin constraints
        if is_increasing_position {
            // Available margin with buffer
            let usable_margin = self.state.available_margin * self.config.margin_buffer;

            // Notional we can add based on available margin and leverage
            // margin_required = notional / leverage
            // notional = margin_required * leverage
            let margin_based_notional = usable_margin * self.config.max_leverage;
            max_notional = max_notional.min(margin_based_notional);

            // Also respect max notional position limit
            let remaining_notional = self.config.max_notional_position - self.state.total_notional;
            max_notional = max_notional.min(remaining_notional.max(0.0));
        }

        // Convert notional to size
        (max_notional / price).max(0.0)
    }

    /// Adjust order size to fit within margin constraints.
    ///
    /// # Arguments
    /// - `size`: Requested order size
    /// - `price`: Current price of the asset
    /// - `current_position`: Current position (positive = long, negative = short)
    /// - `is_buy`: Whether this is a buy order
    ///
    /// # Returns
    /// SizingResult with adjusted size and constraint info
    pub fn adjust_size(
        &self,
        size: f64,
        price: f64,
        current_position: f64,
        is_buy: bool,
    ) -> SizingResult {
        if size <= 0.0 {
            return SizingResult {
                requested_size: size,
                adjusted_size: 0.0,
                was_constrained: false,
                constraint_reason: None,
            };
        }

        // Determine if this order increases or decreases position
        let is_increasing = if is_buy {
            current_position >= 0.0 // Buying when long or flat = increasing
        } else {
            current_position <= 0.0 // Selling when short or flat = increasing
        };

        let max_size = self.max_order_size(price, is_increasing);
        let adjusted_size = size.min(max_size);

        let (was_constrained, constraint_reason) = if adjusted_size < size {
            let reason = if is_increasing {
                if adjusted_size == 0.0 {
                    "No available margin".to_string()
                } else {
                    format!(
                        "Reduced from {} to {} due to margin limits",
                        size, adjusted_size
                    )
                }
            } else {
                format!(
                    "Reduced from {} to {} due to order size limit",
                    size, adjusted_size
                )
            };
            (true, Some(reason))
        } else {
            (false, None)
        };

        SizingResult {
            requested_size: size,
            adjusted_size,
            was_constrained,
            constraint_reason,
        }
    }

    /// Check if we can place an order of given size.
    ///
    /// # Arguments
    /// - `size`: Order size
    /// - `price`: Order price
    /// - `current_position`: Current position
    /// - `is_buy`: Whether this is a buy order
    ///
    /// # Returns
    /// (can_place, reason)
    pub fn can_place_order(
        &self,
        size: f64,
        price: f64,
        current_position: f64,
        is_buy: bool,
    ) -> (bool, Option<String>) {
        if size <= 0.0 {
            return (false, Some("Size must be positive".to_string()));
        }

        if price <= 0.0 {
            return (false, Some("Price must be positive".to_string()));
        }

        // Check order notional limit
        let order_notional = size * price;
        if order_notional > self.config.max_order_notional {
            return (
                false,
                Some(format!(
                    "Order notional ${:.2} exceeds limit ${:.2}",
                    order_notional, self.config.max_order_notional
                )),
            );
        }

        // Determine if this increases position
        let is_increasing = if is_buy {
            current_position >= 0.0
        } else {
            current_position <= 0.0
        };

        if is_increasing {
            // Check leverage limit
            let new_notional = self.state.total_notional + order_notional;
            let new_leverage = if self.state.account_value > 0.0 {
                new_notional / self.state.account_value
            } else {
                f64::INFINITY
            };

            if new_leverage > self.config.max_leverage {
                return (
                    false,
                    Some(format!(
                        "Would exceed max leverage: {:.2}x > {:.2}x",
                        new_leverage, self.config.max_leverage
                    )),
                );
            }

            // Check notional position limit
            if new_notional > self.config.max_notional_position {
                return (
                    false,
                    Some(format!(
                        "Would exceed max notional: ${:.2} > ${:.2}",
                        new_notional, self.config.max_notional_position
                    )),
                );
            }

            // Check margin
            let required_margin = order_notional / self.config.max_leverage;
            let usable_margin = self.state.available_margin * self.config.margin_buffer;
            if required_margin > usable_margin {
                return (
                    false,
                    Some(format!(
                        "Insufficient margin: need ${:.2}, have ${:.2}",
                        required_margin, usable_margin
                    )),
                );
            }
        }

        (true, None)
    }

    /// Get summary for logging/diagnostics.
    pub fn summary(&self) -> MarginSummary {
        MarginSummary {
            account_value: self.state.account_value,
            margin_used: self.state.margin_used,
            available_margin: self.state.available_margin,
            total_notional: self.state.total_notional,
            current_leverage: self.state.current_leverage,
            max_leverage: self.config.max_leverage,
            margin_utilization: if self.state.account_value > 0.0 {
                self.state.margin_used / self.state.account_value
            } else {
                0.0
            },
            is_stale: self.needs_refresh(),
        }
    }
}

/// Summary of margin status.
#[derive(Debug, Clone)]
pub struct MarginSummary {
    pub account_value: f64,
    pub margin_used: f64,
    pub available_margin: f64,
    pub total_notional: f64,
    pub current_leverage: f64,
    pub max_leverage: f64,
    pub margin_utilization: f64,
    pub is_stale: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sizer() -> MarginAwareSizer {
        let config = MarginConfig {
            max_leverage: 5.0,
            margin_buffer: 0.8,
            max_notional_position: 50_000.0,
            max_order_notional: 10_000.0,
            refresh_interval: Duration::from_secs(10),
        };
        let mut sizer = MarginAwareSizer::new(config);
        // Set initial state: $10k account, $2k margin used, $10k notional position
        sizer.update_state(10_000.0, 2_000.0, 10_000.0);
        sizer
    }

    #[test]
    fn test_default_config() {
        let config = MarginConfig::default();
        assert_eq!(config.max_leverage, 3.0);
        assert_eq!(config.margin_buffer, 0.8);
    }

    #[test]
    fn test_margin_state_from_values() {
        let state = MarginState::from_values(10_000.0, 2_000.0, 15_000.0);
        assert_eq!(state.account_value, 10_000.0);
        assert_eq!(state.margin_used, 2_000.0);
        assert_eq!(state.available_margin, 8_000.0);
        assert_eq!(state.current_leverage, 1.5);
        assert!(state.last_updated.is_some());
    }

    #[test]
    fn test_max_order_size_when_increasing() {
        let sizer = create_sizer();
        let price = 50_000.0;

        // Max order size when increasing position
        let max_size = sizer.max_order_size(price, true);

        // Available margin = $8k * 0.8 = $6.4k
        // Max notional from margin = $6.4k * 5 = $32k
        // Remaining notional before max = $50k - $10k = $40k
        // Max order notional limit = $10k
        // Limiting factor: max_order_notional = $10k
        // Max size = $10k / $50k = 0.2
        assert!((max_size - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_max_order_size_when_reducing() {
        let sizer = create_sizer();
        let price = 50_000.0;

        // Max order size when reducing position (no margin constraints)
        let max_size = sizer.max_order_size(price, false);

        // Only limited by max_order_notional = $10k
        // Max size = $10k / $50k = 0.2
        assert!((max_size - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_adjust_size_unconstrained() {
        let sizer = create_sizer();
        let price = 50_000.0;
        let size = 0.1; // $5k notional

        let result = sizer.adjust_size(size, price, 0.5, true);

        assert_eq!(result.requested_size, 0.1);
        assert_eq!(result.adjusted_size, 0.1);
        assert!(!result.was_constrained);
        assert!(result.constraint_reason.is_none());
    }

    #[test]
    fn test_adjust_size_constrained() {
        let sizer = create_sizer();
        let price = 50_000.0;
        let size = 0.5; // $25k notional - exceeds $10k limit

        let result = sizer.adjust_size(size, price, 0.5, true);

        assert_eq!(result.requested_size, 0.5);
        assert!(result.adjusted_size < 0.5);
        assert!(result.was_constrained);
        assert!(result.constraint_reason.is_some());
    }

    #[test]
    fn test_can_place_order_valid() {
        let sizer = create_sizer();
        let price = 50_000.0;
        let size = 0.1; // $5k notional

        let (can_place, reason) = sizer.can_place_order(size, price, 0.5, true);

        assert!(can_place);
        assert!(reason.is_none());
    }

    #[test]
    fn test_can_place_order_exceeds_notional() {
        let sizer = create_sizer();
        let price = 50_000.0;
        let size = 0.3; // $15k notional - exceeds $10k limit

        let (can_place, reason) = sizer.can_place_order(size, price, 0.5, true);

        assert!(!can_place);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("exceeds limit"));
    }

    #[test]
    fn test_can_place_order_exceeds_leverage() {
        let config = MarginConfig {
            max_leverage: 2.0, // Lower leverage limit
            margin_buffer: 0.8,
            max_notional_position: 100_000.0,
            max_order_notional: 100_000.0, // High limit
            refresh_interval: Duration::from_secs(10),
        };
        let mut sizer = MarginAwareSizer::new(config);
        sizer.update_state(10_000.0, 0.0, 15_000.0); // Already at 1.5x

        let price = 50_000.0;
        let size = 0.2; // $10k notional - would push to 2.5x

        let (can_place, reason) = sizer.can_place_order(size, price, 0.5, true);

        assert!(!can_place);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("leverage"));
    }

    #[test]
    fn test_can_place_order_reducing_position() {
        let config = MarginConfig {
            max_leverage: 1.0, // Very low leverage
            margin_buffer: 0.8,
            max_notional_position: 50_000.0,
            max_order_notional: 100_000.0,
            refresh_interval: Duration::from_secs(10),
        };
        let mut sizer = MarginAwareSizer::new(config);
        sizer.update_state(10_000.0, 8_000.0, 40_000.0); // Almost at max

        let price = 50_000.0;
        let size = 0.5; // $25k notional

        // Reducing a long position (selling when long)
        let (can_place, reason) = sizer.can_place_order(size, price, 1.0, false);

        // Should be allowed since we're reducing, not increasing
        assert!(can_place);
        assert!(reason.is_none());
    }

    #[test]
    fn test_summary() {
        let sizer = create_sizer();
        let summary = sizer.summary();

        assert_eq!(summary.account_value, 10_000.0);
        assert_eq!(summary.margin_used, 2_000.0);
        assert_eq!(summary.available_margin, 8_000.0);
        assert_eq!(summary.current_leverage, 1.0);
        assert_eq!(summary.max_leverage, 5.0);
        assert!((summary.margin_utilization - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_is_stale() {
        let config = MarginConfig {
            refresh_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let sizer = MarginAwareSizer::new(config);

        // Initially stale (no state set)
        assert!(sizer.needs_refresh());
    }

    #[test]
    fn test_zero_account_value() {
        let mut sizer = MarginAwareSizer::new(MarginConfig::default());
        sizer.update_state(0.0, 0.0, 0.0);

        let state = sizer.state();
        assert_eq!(state.current_leverage, 0.0);

        let summary = sizer.summary();
        assert_eq!(summary.margin_utilization, 0.0);
    }
}
