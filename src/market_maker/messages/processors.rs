//! Message processors.
//!
//! Contains logic extracted from handle_message for each message type.

use tracing::{debug, warn};

/// Result of processing a fill message.
#[derive(Debug, Clone)]
pub struct FillProcessingResult {
    /// Trade ID
    pub tid: u64,
    /// Order ID
    pub oid: u64,
    /// Fill amount
    pub amount: f64,
    /// Was this a buy
    pub is_buy: bool,
    /// Fill price
    pub fill_price: f64,
    /// Placement price (None if untracked)
    pub placement_price: Option<f64>,
    /// Was the fill new (not duplicate)
    pub is_new: bool,
    /// Was the order fully filled
    pub is_complete: bool,
}

/// Result of processing a trade message.
#[derive(Debug, Clone, Default)]
pub struct TradeProcessingResult {
    /// Number of trades processed
    pub trades_processed: usize,
    /// Number of trades skipped (wrong asset or quality issue)
    pub trades_skipped: usize,
}

/// Result of processing an L2 book message.
#[derive(Debug, Clone)]
pub struct L2BookProcessingResult {
    /// Best bid price
    pub best_bid: Option<f64>,
    /// Best ask price
    pub best_ask: Option<f64>,
    /// Whether book was valid (not crossed)
    pub is_valid: bool,
}

impl Default for L2BookProcessingResult {
    fn default() -> Self {
        Self {
            best_bid: None,
            best_ask: None,
            is_valid: true,
        }
    }
}

/// Check position utilization and log warnings at thresholds.
pub fn check_position_thresholds(position: f64, max_position: f64, asset: &str) {
    let utilization = (position.abs() / max_position).min(1.0);
    let utilization_pct = utilization * 100.0;

    if utilization >= 0.90 {
        warn!(
            position = %format!("{:.6}", position),
            max_position = %format!("{:.6}", max_position),
            utilization = %format!("{:.1}%", utilization_pct),
            asset = %asset,
            "Position at 90%+ of max - approaching reduce-only mode"
        );
    } else if utilization >= 0.75 {
        warn!(
            position = %format!("{:.6}", position),
            max_position = %format!("{:.6}", max_position),
            utilization = %format!("{:.1}%", utilization_pct),
            asset = %asset,
            "Position at 75%+ of max"
        );
    } else if utilization >= 0.50 {
        debug!(
            position = %format!("{:.6}", position),
            max_position = %format!("{:.6}", max_position),
            utilization = %format!("{:.1}%", utilization_pct),
            asset = %asset,
            "Position at 50%+ of max"
        );
    }
}

/// Parse L2 book level strings to (price, size) tuples.
///
/// Takes vectors of (price_string, size_string) pairs.
pub fn parse_l2_level_strings(levels: &[(String, String)]) -> Vec<(f64, f64)> {
    levels
        .iter()
        .filter_map(|(px_str, sz_str)| {
            let px: f64 = px_str.parse().ok()?;
            let sz: f64 = sz_str.parse().ok()?;
            Some((px, sz))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_processing_result_default() {
        let result = TradeProcessingResult::default();
        assert_eq!(result.trades_processed, 0);
        assert_eq!(result.trades_skipped, 0);
    }

    #[test]
    fn test_l2_book_result_default() {
        let result = L2BookProcessingResult::default();
        assert!(result.is_valid);
        assert!(result.best_bid.is_none());
    }
}
