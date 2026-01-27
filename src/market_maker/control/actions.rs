//! Action types for the stochastic controller.
//!
//! Actions represent the full set of possible decisions the controller can make,
//! going beyond simple quote/no-quote to include inventory management.

use crate::market_maker::quoting::Ladder;

/// Action that the stochastic controller can take.
///
/// Layer 3 adds terminal and inventory management actions that Layer 2
/// (LearningModule) doesn't know about.
#[derive(Debug, Clone)]
pub enum Action {
    /// Place quotes with the given ladder.
    Quote {
        /// The ladder of quotes to place
        ladder: Ladder,
        /// Expected value of this action
        expected_value: f64,
    },

    /// Don't quote this cycle.
    NoQuote {
        /// Reason for not quoting
        reason: NoQuoteReason,
    },

    /// Aggressively unwind inventory (terminal or risk condition).
    DumpInventory {
        /// Urgency factor (higher = more aggressive)
        urgency: f64,
        /// Target position to reach
        target_position: f64,
    },

    /// Build inventory towards a target (e.g., for funding capture).
    BuildInventory {
        /// Target position
        target: f64,
        /// How aggressively to build (spreads tighter on target side)
        aggressiveness: f64,
    },

    // NOTE: DefensiveQuote has been removed. All uncertainty is now handled
    // through gamma scaling (kappa_ci_width flows through uncertainty_scalar).
    // The GLFT formula naturally widens spreads when gamma increases due to uncertainty.
    // This eliminates arbitrary spread_multiplier and size_fraction values.

    /// Wait to learn more before acting.
    WaitToLearn {
        /// Expected information gain from waiting
        expected_info_gain: f64,
        /// How many cycles to wait
        suggested_wait_cycles: u32,
    },
}

impl Default for Action {
    fn default() -> Self {
        Self::NoQuote {
            reason: NoQuoteReason::NotReady,
        }
    }
}

impl Action {
    /// Check if this action involves quoting.
    pub fn is_quoting(&self) -> bool {
        matches!(self, Self::Quote { .. } | Self::BuildInventory { .. })
    }

    /// Check if this action is a wait/no-action.
    pub fn is_passive(&self) -> bool {
        matches!(self, Self::NoQuote { .. } | Self::WaitToLearn { .. })
    }

    /// Check if this is an urgent inventory action.
    pub fn is_urgent(&self) -> bool {
        matches!(self, Self::DumpInventory { urgency, .. } if *urgency > 2.0)
    }

    /// Get the expected value if available.
    pub fn expected_value(&self) -> Option<f64> {
        match self {
            Self::Quote { expected_value, .. } => Some(*expected_value),
            _ => None,
        }
    }
}

/// Reasons for not quoting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoQuoteReason {
    /// System not ready (warming up, no data)
    NotReady,
    /// Negative expected edge
    NegativeEdge,
    /// Model health degraded
    ModelDegraded,
    /// High uncertainty, waiting to learn
    HighUncertainty,
    /// Changepoint detected, beliefs resetting
    ChangepointDetected,
    /// Risk limit reached
    RiskLimit,
    /// Rate limit constraint
    RateLimited,
    /// Data staleness
    StaleData,
    /// Cascade/liquidation detected
    CascadeDetected,
    /// Information value: better to wait
    InformationValue,
}

impl std::fmt::Display for NoQuoteReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotReady => write!(f, "not ready"),
            Self::NegativeEdge => write!(f, "negative edge"),
            Self::ModelDegraded => write!(f, "model degraded"),
            Self::HighUncertainty => write!(f, "high uncertainty"),
            Self::ChangepointDetected => write!(f, "changepoint detected"),
            Self::RiskLimit => write!(f, "risk limit"),
            Self::RateLimited => write!(f, "rate limited"),
            Self::StaleData => write!(f, "stale data"),
            Self::CascadeDetected => write!(f, "cascade detected"),
            Self::InformationValue => write!(f, "waiting to learn"),
        }
    }
}

// NOTE: DefensiveReason enum has been removed. All uncertainty is now handled
// through gamma scaling (kappa_ci_width flows through uncertainty_scalar).
// The GLFT formula naturally widens spreads when gamma increases due to uncertainty.

/// Configuration for action generation.
#[derive(Debug, Clone)]
pub struct ActionConfig {
    /// Urgency threshold for terminal zone (session time fraction)
    pub terminal_zone_start: f64,
    /// Position fraction that triggers dump
    pub dump_position_threshold: f64,
    /// Maximum dump urgency
    pub max_dump_urgency: f64,
    /// Funding capture threshold (rate in bps)
    pub funding_capture_threshold: f64,
    /// Maximum funding position (as fraction of max position)
    pub max_funding_position_fraction: f64,
    // NOTE: defensive_spread_range and defensive_size_range have been removed.
    // All uncertainty is now handled through gamma scaling.
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            terminal_zone_start: 0.95,
            dump_position_threshold: 0.8,
            max_dump_urgency: 10.0,
            funding_capture_threshold: 5.0, // 5 bps
            max_funding_position_fraction: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_classification() {
        let quote = Action::Quote {
            ladder: Ladder::default(),
            expected_value: 1.0,
        };
        assert!(quote.is_quoting());
        assert!(!quote.is_passive());

        let no_quote = Action::NoQuote {
            reason: NoQuoteReason::NegativeEdge,
        };
        assert!(!no_quote.is_quoting());
        assert!(no_quote.is_passive());

        let dump = Action::DumpInventory {
            urgency: 5.0,
            target_position: 0.0,
        };
        assert!(dump.is_urgent());
    }

    #[test]
    fn test_no_quote_reason_display() {
        assert_eq!(NoQuoteReason::NegativeEdge.to_string(), "negative edge");
        assert_eq!(
            NoQuoteReason::ChangepointDetected.to_string(),
            "changepoint detected"
        );
    }
}
