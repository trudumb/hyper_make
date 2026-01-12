//! UserFills message handler.
//!
//! Processes fill notifications from the WebSocket feed.

use crate::prelude::Result;
use crate::ws::message_types::UserFills;
use tracing::{debug, warn};

use super::context::MessageContext;
use crate::market_maker::fills::{FillEvent, FillProcessor, FillState};
use crate::market_maker::tracking::{OrderManager, QueuePositionTracker};

/// A fill observation for Bayesian learning.
#[derive(Debug, Clone, Copy)]
pub struct FillObservation {
    /// Depth from mid in basis points
    pub depth_bps: f64,
    /// Whether the order filled (true) or was cancelled (false)
    pub filled: bool,
}

/// Result of processing UserFills message.
#[derive(Debug, Default)]
pub struct UserFillsResult {
    /// Number of fills processed
    pub fills_processed: usize,
    /// Number of fills skipped (wrong asset, duplicate)
    pub fills_skipped: usize,
    /// Order IDs that were cleaned up
    pub cleaned_oids: Vec<u64>,
    /// Whether quotes should be updated after fills
    pub should_update_quotes: bool,
    /// Number of fills for untracked orders (Phase 4: triggers reconciliation)
    pub unmatched_fills: usize,
    /// Fill observations for Bayesian fill probability learning.
    /// Each entry contains (depth_bps, filled=true) for the strategy to learn from.
    pub fill_observations: Vec<FillObservation>,
    /// Total USD volume from new fills (for rate limit budget calculation)
    pub total_volume_usd: f64,
}

/// Process UserFills through the FillProcessor.
///
/// This function creates FillEvents and delegates to FillProcessor for:
/// - Deduplication
/// - Position updates
/// - Order tracking
/// - Adverse selection measurement
/// - P&L tracking
/// - Queue tracking updates
pub fn process_user_fills<'a>(
    user_fills: &UserFills,
    ctx: &MessageContext,
    fill_processor: &mut FillProcessor,
    fill_state: &mut FillState<'a>,
) -> Result<UserFillsResult> {
    let mut result = UserFillsResult::default();

    // Need valid mid price
    if !ctx.has_mid() {
        return Ok(result);
    }

    // Process each fill
    for fill in &user_fills.data.fills {
        // Filter by asset
        if fill.coin != *ctx.asset {
            result.fills_skipped += 1;
            continue;
        }

        // Parse fill data
        let amount: f64 = match fill.sz.parse() {
            Ok(a) => a,
            Err(_) => {
                result.fills_skipped += 1;
                continue;
            }
        };
        let fill_price: f64 = fill.px.parse().unwrap_or(ctx.latest_mid);
        let is_buy = fill.side.eq("B");

        // Validate fee_token matches expected collateral for this DEX
        // This catches misconfiguration where we're trading on wrong DEX
        if fill.fee_token != *ctx.expected_collateral {
            warn!(
                asset = %ctx.asset,
                tid = fill.tid,
                expected_collateral = %ctx.expected_collateral,
                actual_fee_token = %fill.fee_token,
                "Fill fee_token mismatch - trading on wrong DEX or collateral misconfigured"
            );
        }

        // Create fill event
        let fill_event = FillEvent::new(
            fill.tid,
            fill.oid,
            amount,
            fill_price,
            is_buy,
            ctx.latest_mid,
            None,
            ctx.asset.to_string(),
        );

        // Process through unified processor
        let process_result = fill_processor.process(&fill_event, fill_state);

        if process_result.fill_result.is_new {
            result.fills_processed += 1;
            // Phase 4: Track unmatched fills for reconciliation triggering
            if !process_result.order_found {
                result.unmatched_fills += 1;
            }

            // Track USD volume for rate limit budget calculation
            let fill_volume_usd = amount * fill_price;
            result.total_volume_usd += fill_volume_usd;

            // Record fill observation for Bayesian learning.
            // We need to compute depth from the placement price if available.
            if let Some(placement_price) = process_result.placement_price {
                let depth_bps = if ctx.latest_mid > 0.0 {
                    ((placement_price - ctx.latest_mid).abs() / ctx.latest_mid) * 10_000.0
                } else {
                    0.0
                };
                result.fill_observations.push(FillObservation {
                    depth_bps,
                    filled: true, // This is a fill, not a cancel
                });
            }
        } else {
            result.fills_skipped += 1;
        }
    }

    // Signal that quotes should be updated if we processed fills
    result.should_update_quotes = result.fills_processed > 0;

    debug!(
        asset = %ctx.asset,
        processed = result.fills_processed,
        skipped = result.fills_skipped,
        "UserFills processed"
    );

    Ok(result)
}

/// Cleanup completed orders and update queue tracker.
pub fn cleanup_orders(
    orders: &mut OrderManager,
    queue_tracker: &mut QueuePositionTracker,
) -> Vec<u64> {
    let removed_oids = orders.cleanup();
    for oid in &removed_oids {
        queue_tracker.order_removed(*oid);
    }
    removed_oids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::config::MetricsRecorder;
    use crate::market_maker::control::StochasticController;
    use crate::market_maker::{
        AdverseSelectionConfig, AdverseSelectionEstimator, DepthDecayAS, EstimatorConfig,
        ParameterEstimator, PnLConfig, PnLTracker, PositionTracker, PrometheusMetrics, QueueConfig,
    };
    use std::sync::Arc;

    #[test]
    fn test_process_filters_wrong_asset() {
        let ctx = MessageContext::new(
            Arc::from("BTC"),
            50000.0,
            0.0,
            1.0,
            false,
            Arc::from("USDC"),
        );

        let mut fill_processor = FillProcessor::new();
        let mut position = PositionTracker::new(0.0);
        let mut orders = OrderManager::new();
        let mut adverse_selection =
            AdverseSelectionEstimator::new(AdverseSelectionConfig::default());
        let mut depth_decay_as = DepthDecayAS::default();
        let mut queue_tracker = QueuePositionTracker::new(QueueConfig::default());
        let mut estimator = ParameterEstimator::new(EstimatorConfig::default());
        let mut pnl_tracker = PnLTracker::new(PnLConfig::default());
        let mut prometheus = PrometheusMetrics::new();
        let metrics: MetricsRecorder = None;
        let mut learning = crate::market_maker::learning::LearningModule::default();
        let mut stochastic_controller = StochasticController::default();

        let mut fill_state = FillState {
            position: &mut position,
            orders: &mut orders,
            adverse_selection: &mut adverse_selection,
            depth_decay_as: &mut depth_decay_as,
            queue_tracker: &mut queue_tracker,
            estimator: &mut estimator,
            pnl_tracker: &mut pnl_tracker,
            prometheus: &mut prometheus,
            metrics: &metrics,
            latest_mid: 50000.0,
            asset: "BTC",
            max_position: 1.0,
            calibrate_depth_as: false,
            learning: &mut learning,
            stochastic_controller: &mut stochastic_controller,
            fee_bps: 1.5,
        };

        // Create fill for wrong asset
        let user_fills = UserFills {
            data: crate::types::UserFillsData {
                fills: vec![crate::types::TradeInfo {
                    coin: "ETH".to_string(),
                    oid: 1,
                    tid: 1,
                    px: "3000.0".to_string(),
                    sz: "1.0".to_string(),
                    side: "B".to_string(),
                    time: 1000,
                    closed_pnl: "0".to_string(),
                    hash: "hash".to_string(),
                    start_position: "0".to_string(),
                    dir: "L".to_string(),
                    crossed: false,
                    fee: "0".to_string(),
                    fee_token: "USDC".to_string(),
                    cloid: None,
                    builder_fee: None, // HIP-3: no builder fee for validator perps
                }],
                user: alloy::primitives::Address::ZERO,
                is_snapshot: Some(false),
            },
        };

        let result =
            process_user_fills(&user_fills, &ctx, &mut fill_processor, &mut fill_state).unwrap();

        assert_eq!(result.fills_processed, 0);
        assert_eq!(result.fills_skipped, 1);
    }

    #[test]
    fn test_process_skips_without_mid() {
        let ctx = MessageContext::new(Arc::from("BTC"), -1.0, 0.0, 1.0, false, Arc::from("USDC"));

        let mut fill_processor = FillProcessor::new();
        let mut position = PositionTracker::new(0.0);
        let mut orders = OrderManager::new();
        let mut adverse_selection =
            AdverseSelectionEstimator::new(AdverseSelectionConfig::default());
        let mut depth_decay_as = DepthDecayAS::default();
        let mut queue_tracker = QueuePositionTracker::new(QueueConfig::default());
        let mut estimator = ParameterEstimator::new(EstimatorConfig::default());
        let mut pnl_tracker = PnLTracker::new(PnLConfig::default());
        let mut prometheus = PrometheusMetrics::new();
        let metrics: MetricsRecorder = None;
        let mut learning = crate::market_maker::learning::LearningModule::default();
        let mut stochastic_controller = StochasticController::default();

        let mut fill_state = FillState {
            position: &mut position,
            orders: &mut orders,
            adverse_selection: &mut adverse_selection,
            depth_decay_as: &mut depth_decay_as,
            queue_tracker: &mut queue_tracker,
            estimator: &mut estimator,
            pnl_tracker: &mut pnl_tracker,
            prometheus: &mut prometheus,
            metrics: &metrics,
            latest_mid: -1.0,
            asset: "BTC",
            max_position: 1.0,
            calibrate_depth_as: false,
            learning: &mut learning,
            stochastic_controller: &mut stochastic_controller,
            fee_bps: 1.5,
        };

        let user_fills = UserFills {
            data: crate::types::UserFillsData {
                fills: vec![],
                user: alloy::primitives::Address::ZERO,
                is_snapshot: Some(false),
            },
        };

        let result =
            process_user_fills(&user_fills, &ctx, &mut fill_processor, &mut fill_state).unwrap();

        assert_eq!(result.fills_processed, 0);
        assert!(!result.should_update_quotes);
    }
}
