//! L2Book message handler.
//!
//! Processes order book updates from the WebSocket feed.

use crate::prelude::Result;
use crate::ws::message_types::L2Book;
use tracing::{debug, warn};

use super::context::MessageContext;
use super::processors::L2BookProcessingResult;
use crate::market_maker::estimator::ParameterEstimator;
use crate::market_maker::events::{L2Level, ParsedL2Book};
use crate::market_maker::infra::{AnomalyType, DataQualityMonitor, PrometheusMetrics};
use crate::market_maker::process_models::SpreadProcessEstimator;
use crate::market_maker::tracking::QueuePositionTracker;
use crate::types::OrderBookLevel;

/// Mutable references needed by L2Book handler.
pub struct L2BookState<'a> {
    /// Parameter estimator
    pub estimator: &'a mut ParameterEstimator,
    /// Queue position tracker
    pub queue_tracker: &'a mut QueuePositionTracker,
    /// Spread process estimator
    pub spread_tracker: &'a mut SpreadProcessEstimator,
    /// Data quality monitor
    pub data_quality: &'a mut DataQualityMonitor,
    /// Prometheus metrics
    pub prometheus: &'a mut PrometheusMetrics,
}

/// Process an L2Book message.
///
/// Updates:
/// - Estimator (kappa, book imbalance)
/// - Queue tracker (depth-ahead for orders)
/// - Spread tracker (spread dynamics)
/// - Data quality validation
///
/// Returns processing result with best bid/ask.
pub fn process_l2_book(
    l2_book: &L2Book,
    ctx: &MessageContext,
    state: &mut L2BookState,
) -> Result<L2BookProcessingResult> {
    let mut result = L2BookProcessingResult::default();

    // Validate asset and mid price
    if l2_book.data.coin != *ctx.asset || ctx.latest_mid <= 0.0 {
        return Ok(result);
    }

    // Need at least 2 sides (bids and asks)
    if l2_book.data.levels.len() < 2 {
        return Ok(result);
    }

    // Parse L2 levels
    let parsed = parse_l2_book(&l2_book.data.levels)?;

    // Get best bid/ask
    result.best_bid = parsed.best_bid();
    result.best_ask = parsed.best_ask();

    // Data quality check for crossed book
    if let (Some(best_bid), Some(best_ask)) = (result.best_bid, result.best_ask) {
        if let Err(anomaly) = state
            .data_quality
            .check_l2_book(&ctx.asset, 0, best_bid, best_ask)
        {
            warn!(anomaly = %anomaly, "L2 book quality issue");
            state.prometheus.record_data_quality_issue();
            if matches!(anomaly, AnomalyType::CrossedBook) {
                state.prometheus.record_crossed_book();
                result.is_valid = false;
                return Ok(result);
            }
        }
    }

    // Update estimator and trackers
    let bids = parsed.bids_as_tuples();
    let asks = parsed.asks_as_tuples();
    state.estimator.on_l2_book(&bids, &asks, ctx.latest_mid);

    if let (Some(best_bid), Some(best_ask)) = (result.best_bid, result.best_ask) {
        state
            .queue_tracker
            .update_from_book(best_bid, best_ask, state.estimator.sigma_clean());
        state
            .spread_tracker
            .update(best_bid, best_ask, state.estimator.sigma_clean());
    }

    debug!(
        asset = %ctx.asset,
        best_bid = ?result.best_bid,
        best_ask = ?result.best_ask,
        spread_bps = ?parsed.spread_bps(),
        kappa = %format!("{:.2}", state.estimator.kappa()),
        "L2Book processed"
    );

    Ok(result)
}

/// Parse L2 book levels from WebSocket format.
fn parse_l2_book(levels: &[Vec<OrderBookLevel>]) -> Result<ParsedL2Book> {
    let bids: Vec<L2Level> = levels[0]
        .iter()
        .filter_map(|level| {
            let px: f64 = level.px.parse().ok()?;
            let sz: f64 = level.sz.parse().ok()?;
            Some(L2Level {
                price: px,
                size: sz,
            })
        })
        .collect();

    let asks: Vec<L2Level> = levels[1]
        .iter()
        .filter_map(|level| {
            let px: f64 = level.px.parse().ok()?;
            let sz: f64 = level.sz.parse().ok()?;
            Some(L2Level {
                price: px,
                size: sz,
            })
        })
        .collect();

    Ok(ParsedL2Book { bids, asks })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::{DataQualityConfig, EstimatorConfig, QueueConfig, SpreadConfig};
    use std::sync::Arc;

    fn make_test_state<'a>(
        estimator: &'a mut ParameterEstimator,
        queue_tracker: &'a mut QueuePositionTracker,
        spread_tracker: &'a mut SpreadProcessEstimator,
        data_quality: &'a mut DataQualityMonitor,
        prometheus: &'a mut PrometheusMetrics,
    ) -> L2BookState<'a> {
        L2BookState {
            estimator,
            queue_tracker,
            spread_tracker,
            data_quality,
            prometheus,
        }
    }

    #[test]
    fn test_parse_l2_book() {
        use crate::types::OrderBookLevel;

        let levels = vec![
            vec![OrderBookLevel {
                px: "49900.0".to_string(),
                sz: "1.0".to_string(),
                n: 1,
            }],
            vec![OrderBookLevel {
                px: "50100.0".to_string(),
                sz: "1.0".to_string(),
                n: 1,
            }],
        ];

        let parsed = parse_l2_book(&levels).unwrap();
        assert_eq!(parsed.best_bid(), Some(49900.0));
        assert_eq!(parsed.best_ask(), Some(50100.0));
    }

    #[test]
    fn test_process_filters_wrong_asset() {
        let mut estimator = ParameterEstimator::new(EstimatorConfig::default());
        let mut queue_tracker = QueuePositionTracker::new(QueueConfig::default());
        let mut spread_tracker = SpreadProcessEstimator::new(SpreadConfig::default());
        let mut data_quality = DataQualityMonitor::new(DataQualityConfig::default());
        let mut prometheus = PrometheusMetrics::new();

        let mut state = make_test_state(
            &mut estimator,
            &mut queue_tracker,
            &mut spread_tracker,
            &mut data_quality,
            &mut prometheus,
        );

        let ctx = MessageContext::new(Arc::from("BTC"), 50000.0, 0.0, 1.0, false);

        // Create L2Book for wrong asset
        let l2_book = L2Book {
            data: crate::types::L2BookData {
                coin: "ETH".to_string(),
                levels: vec![vec![], vec![]],
                time: 1000,
            },
        };

        let result = process_l2_book(&l2_book, &ctx, &mut state).unwrap();

        assert!(result.best_bid.is_none());
        assert!(result.best_ask.is_none());
    }
}
