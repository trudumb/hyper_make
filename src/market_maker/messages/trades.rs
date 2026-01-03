//! Trades message handler.
//!
//! Processes trade feed updates from the WebSocket.

use crate::prelude::Result;
use crate::ws::message_types::Trades;
use tracing::{debug, info, warn};

use super::context::MessageContext;
use super::processors::TradeProcessingResult;
use crate::market_maker::estimator::ParameterEstimator;
use crate::market_maker::infra::{DataQualityMonitor, PrometheusMetrics};
use crate::market_maker::process_models::HawkesOrderFlowEstimator;

/// Mutable references needed by Trades handler.
pub struct TradesState<'a> {
    /// Parameter estimator
    pub estimator: &'a mut ParameterEstimator,
    /// Hawkes order flow estimator
    pub hawkes: &'a mut HawkesOrderFlowEstimator,
    /// Data quality monitor
    pub data_quality: &'a mut DataQualityMonitor,
    /// Prometheus metrics
    pub prometheus: &'a mut PrometheusMetrics,
    /// Last logged warmup progress (for throttling)
    pub last_warmup_log: &'a mut usize,
}

/// Process a Trades message.
///
/// Updates:
/// - Estimator (volatility, flow imbalance)
/// - Hawkes order flow
/// - Data quality validation
///
/// Returns the number of trades processed and skipped.
pub fn process_trades(
    trades: &Trades,
    ctx: &MessageContext,
    state: &mut TradesState,
) -> Result<TradeProcessingResult> {
    let mut result = TradeProcessingResult::default();

    for trade in &trades.data {
        // Skip trades for other assets
        if trade.coin != *ctx.asset {
            result.trades_skipped += 1;
            continue;
        }

        // Parse trade data
        let price: f64 = match trade.px.parse() {
            Ok(p) => p,
            Err(_) => {
                result.trades_skipped += 1;
                continue;
            }
        };
        let size: f64 = match trade.sz.parse() {
            Ok(s) => s,
            Err(_) => {
                result.trades_skipped += 1;
                continue;
            }
        };

        // Data quality validation
        if let Err(anomaly) =
            state
                .data_quality
                .check_trade(&ctx.asset, 0, trade.time, price, size, ctx.latest_mid)
        {
            warn!(anomaly = %anomaly, price = %price, size = %size, "Trade quality issue");
            state.prometheus.record_data_quality_issue();
            result.trades_skipped += 1;
            continue;
        }

        // Update estimators
        let is_buy_aggressor = trade.side == "B";
        state
            .estimator
            .on_trade(trade.time, price, size, Some(is_buy_aggressor));
        state.hawkes.record_trade(is_buy_aggressor, size);

        result.trades_processed += 1;
    }

    // Warmup progress logging
    log_warmup_progress(state.estimator, state.last_warmup_log);

    debug!(
        asset = %ctx.asset,
        processed = result.trades_processed,
        skipped = result.trades_skipped,
        "Trades processed"
    );

    Ok(result)
}

/// Log estimator warmup progress (throttled).
///
/// Progress is logged:
/// - Immediately on first trade (so user knows trades are being received)
/// - Every tick during early warmup (ticks 0-5)
/// - Every 5 ticks during later warmup
fn log_warmup_progress(estimator: &ParameterEstimator, last_warmup_log: &mut usize) {
    if estimator.is_warmed_up() {
        return;
    }

    let (vol_ticks, min_vol, trade_obs, min_trades) = estimator.warmup_progress();

    // Determine logging threshold: more frequent early, less frequent later
    let log_threshold = if vol_ticks < 5 { 1 } else { 5 };

    // Log on first trade or when threshold crossed
    let should_log =
        (*last_warmup_log == 0 && trade_obs > 0) || (vol_ticks >= *last_warmup_log + log_threshold);

    if should_log {
        info!(
            "Warming up: {}/{} volume ticks, {}/{} trade observations",
            vol_ticks, min_vol, trade_obs, min_trades
        );
        *last_warmup_log = vol_ticks.max(1); // Prevent re-logging at 0
    }

    if vol_ticks >= min_vol && trade_obs >= min_trades {
        info!(
            "Warmup complete! σ={:.6}, κ={:.2}, jump_ratio={:.2}",
            estimator.sigma(),
            estimator.kappa(),
            estimator.jump_ratio()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::{DataQualityConfig, EstimatorConfig, HawkesConfig};
    use std::sync::Arc;

    fn make_test_state<'a>(
        estimator: &'a mut ParameterEstimator,
        hawkes: &'a mut HawkesOrderFlowEstimator,
        data_quality: &'a mut DataQualityMonitor,
        prometheus: &'a mut PrometheusMetrics,
        last_warmup_log: &'a mut usize,
    ) -> TradesState<'a> {
        TradesState {
            estimator,
            hawkes,
            data_quality,
            prometheus,
            last_warmup_log,
        }
    }

    #[test]
    fn test_process_trades_filters_wrong_asset() {
        let mut estimator = ParameterEstimator::new(EstimatorConfig::default());
        let mut hawkes = HawkesOrderFlowEstimator::new(HawkesConfig::default());
        let mut data_quality = DataQualityMonitor::new(DataQualityConfig::default());
        let mut prometheus = PrometheusMetrics::new();
        let mut last_warmup_log = 0;

        let mut state = make_test_state(
            &mut estimator,
            &mut hawkes,
            &mut data_quality,
            &mut prometheus,
            &mut last_warmup_log,
        );

        let ctx = MessageContext::new(Arc::from("BTC"), 50000.0, 0.0, 1.0, false, Arc::from("USDC"));

        // Create trades for wrong asset
        let trades = Trades {
            data: vec![crate::types::Trade {
                coin: "ETH".to_string(),
                side: "B".to_string(),
                px: "3000.0".to_string(),
                sz: "1.0".to_string(),
                time: 1000,
                hash: "hash".to_string(),
                tid: 1,
                users: ("user1".to_string(), "user2".to_string()),
            }],
        };

        let result = process_trades(&trades, &ctx, &mut state).unwrap();

        assert_eq!(result.trades_processed, 0);
        assert_eq!(result.trades_skipped, 1);
    }
}
