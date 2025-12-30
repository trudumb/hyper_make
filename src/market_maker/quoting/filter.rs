//! Quote filtering for position limits and reduce-only mode.
//!
//! Extracts the duplicated reduce-only logic from mod.rs into a unified filter.

use crate::market_maker::config::Quote;
use tracing::warn;

/// Reason for entering reduce-only mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReduceOnlyReason {
    /// Position exceeds max_position limit.
    OverPositionLimit,
    /// Position value exceeds max_position_value limit.
    OverValueLimit,
}

/// Result of reduce-only filtering.
#[derive(Debug, Clone)]
pub struct ReduceOnlyResult {
    /// Whether quotes were filtered.
    pub was_filtered: bool,
    /// Reason for filtering (if any).
    pub reason: Option<ReduceOnlyReason>,
    /// Side that was filtered (true = bid, false = ask).
    pub filtered_bids: bool,
    /// Side that was filtered (true = ask, false = bid).
    pub filtered_asks: bool,
}

impl ReduceOnlyResult {
    fn no_filtering() -> Self {
        Self {
            was_filtered: false,
            reason: None,
            filtered_bids: false,
            filtered_asks: false,
        }
    }

    fn filtered_bids(reason: ReduceOnlyReason) -> Self {
        Self {
            was_filtered: true,
            reason: Some(reason),
            filtered_bids: true,
            filtered_asks: false,
        }
    }

    fn filtered_asks(reason: ReduceOnlyReason) -> Self {
        Self {
            was_filtered: true,
            reason: Some(reason),
            filtered_bids: false,
            filtered_asks: true,
        }
    }
}

/// Configuration for reduce-only filtering.
pub struct ReduceOnlyConfig {
    /// Current position.
    pub position: f64,
    /// Maximum allowed position.
    pub max_position: f64,
    /// Current mid price (for value calculation).
    pub mid_price: f64,
    /// Maximum allowed position value.
    pub max_position_value: f64,
    /// Asset name (for logging).
    pub asset: String,
}

/// Quote filtering utilities.
///
/// Provides reduce-only filtering for both single quotes and ladders.
pub struct QuoteFilter;

impl QuoteFilter {
    /// Apply reduce-only logic to ladder quotes.
    ///
    /// When position exceeds limits, clears the side that would increase exposure.
    /// Returns a result describing what was filtered.
    pub fn apply_reduce_only_ladder(
        bids: &mut Vec<Quote>,
        asks: &mut Vec<Quote>,
        config: &ReduceOnlyConfig,
    ) -> ReduceOnlyResult {
        let position = config.position;
        let position_value = position.abs() * config.mid_price;

        let over_position_limit = position.abs() > config.max_position;
        let over_value_limit = position_value > config.max_position_value;

        if !over_position_limit && !over_value_limit {
            return ReduceOnlyResult::no_filtering();
        }

        let reason = if over_value_limit {
            ReduceOnlyReason::OverValueLimit
        } else {
            ReduceOnlyReason::OverPositionLimit
        };

        if position > 0.0 {
            // Long position over max: only allow sells (no bids)
            bids.clear();
            Self::log_reduce_only(position, config, reason, true);
            ReduceOnlyResult::filtered_bids(reason)
        } else {
            // Short position over max: only allow buys (no asks)
            asks.clear();
            Self::log_reduce_only(position, config, reason, false);
            ReduceOnlyResult::filtered_asks(reason)
        }
    }

    /// Apply reduce-only logic to single quotes.
    ///
    /// When position exceeds limits, clears the side that would increase exposure.
    /// Returns a result describing what was filtered.
    pub fn apply_reduce_only_single(
        bid: &mut Option<Quote>,
        ask: &mut Option<Quote>,
        config: &ReduceOnlyConfig,
    ) -> ReduceOnlyResult {
        let position = config.position;
        let position_value = position.abs() * config.mid_price;

        let over_position_limit = position.abs() > config.max_position;
        let over_value_limit = position_value > config.max_position_value;

        if !over_position_limit && !over_value_limit {
            return ReduceOnlyResult::no_filtering();
        }

        let reason = if over_value_limit {
            ReduceOnlyReason::OverValueLimit
        } else {
            ReduceOnlyReason::OverPositionLimit
        };

        if position > 0.0 {
            // Long position over max: only allow sells (no bids)
            *bid = None;
            Self::log_reduce_only(position, config, reason, true);
            ReduceOnlyResult::filtered_bids(reason)
        } else {
            // Short position over max: only allow buys (no asks)
            *ask = None;
            Self::log_reduce_only(position, config, reason, false);
            ReduceOnlyResult::filtered_asks(reason)
        }
    }

    /// Check if we should be in reduce-only mode.
    pub fn is_reduce_only(config: &ReduceOnlyConfig) -> Option<ReduceOnlyReason> {
        let position = config.position;
        let position_value = position.abs() * config.mid_price;

        if position_value > config.max_position_value {
            Some(ReduceOnlyReason::OverValueLimit)
        } else if position.abs() > config.max_position {
            Some(ReduceOnlyReason::OverPositionLimit)
        } else {
            None
        }
    }

    fn log_reduce_only(
        position: f64,
        config: &ReduceOnlyConfig,
        reason: ReduceOnlyReason,
        is_bid_side: bool,
    ) {
        let side_name = if is_bid_side { "bids" } else { "asks" };
        let position_type = if position > 0.0 { "long" } else { "short" };

        match reason {
            ReduceOnlyReason::OverValueLimit => {
                let position_value = position.abs() * config.mid_price;
                warn!(
                    position = %format!("{:.6}", position),
                    position_value = %format!("${:.2}", position_value),
                    limit = %format!("${:.2}", config.max_position_value),
                    "Position value over limit ({}) - reduce-only mode, cancelling {}",
                    position_type,
                    side_name
                );
            }
            ReduceOnlyReason::OverPositionLimit => {
                warn!(
                    position = %format!("{:.6}", position),
                    max_position = %format!("{:.6}", config.max_position),
                    "Over max position ({}) - reduce-only mode, cancelling {}",
                    position_type,
                    side_name
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(position: f64) -> ReduceOnlyConfig {
        // Set value limit high enough that it won't trigger by default
        // Position 10 * 50000 = $500k, so set limit to $1M
        ReduceOnlyConfig {
            position,
            max_position: 10.0,
            mid_price: 50000.0,
            max_position_value: 1_000_000.0, // $1M (high enough to not trigger)
            asset: "BTC".to_string(),
        }
    }

    #[test]
    fn test_no_filtering_when_within_limits() {
        let config = make_config(5.0); // 5 BTC, under max_position of 10
        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        assert!(!result.was_filtered);
        assert_eq!(bids.len(), 1);
        assert_eq!(asks.len(), 1);
    }

    #[test]
    fn test_filter_bids_when_long_over_limit() {
        let config = make_config(15.0); // Over max_position of 10
        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        assert!(result.was_filtered);
        assert!(result.filtered_bids);
        assert!(!result.filtered_asks);
        assert_eq!(result.reason, Some(ReduceOnlyReason::OverPositionLimit));
        assert!(bids.is_empty());
        assert_eq!(asks.len(), 1);
    }

    #[test]
    fn test_filter_asks_when_short_over_limit() {
        let config = make_config(-15.0); // Short over max_position of 10
        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        assert!(result.was_filtered);
        assert!(!result.filtered_bids);
        assert!(result.filtered_asks);
        assert_eq!(result.reason, Some(ReduceOnlyReason::OverPositionLimit));
        assert_eq!(bids.len(), 1);
        assert!(asks.is_empty());
    }

    #[test]
    fn test_value_limit_takes_precedence() {
        // Position value: 5 * 50000 = $250k
        // Set value limit to $100k so it triggers
        let mut config = make_config(5.0);
        config.max_position_value = 100_000.0; // Lower than position value

        let mut bids = vec![Quote::new(49000.0, 0.1)];
        let mut asks = vec![Quote::new(51000.0, 0.1)];

        let result = QuoteFilter::apply_reduce_only_ladder(&mut bids, &mut asks, &config);

        assert!(result.was_filtered);
        assert_eq!(result.reason, Some(ReduceOnlyReason::OverValueLimit));
    }

    #[test]
    fn test_single_quote_filtering() {
        let config = make_config(15.0); // Over max_position of 10
        let mut bid = Some(Quote::new(49000.0, 0.1));
        let mut ask = Some(Quote::new(51000.0, 0.1));

        let result = QuoteFilter::apply_reduce_only_single(&mut bid, &mut ask, &config);

        assert!(result.was_filtered);
        assert!(bid.is_none());
        assert!(ask.is_some());
    }

    #[test]
    fn test_is_reduce_only() {
        let normal_config = make_config(5.0);
        assert!(QuoteFilter::is_reduce_only(&normal_config).is_none());

        let over_position = make_config(15.0);
        assert_eq!(
            QuoteFilter::is_reduce_only(&over_position),
            Some(ReduceOnlyReason::OverPositionLimit)
        );
    }
}
