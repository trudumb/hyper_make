//! AllMids message handler.
//!
//! Processes mid price updates from the WebSocket feed.

use crate::prelude::Result;
use crate::ws::message_types::AllMids;
use tracing::debug;

use super::context::MessageContext;
use crate::market_maker::adverse_selection::{AdverseSelectionEstimator, DepthDecayAS};
use crate::market_maker::estimator::ParameterEstimator;
use crate::market_maker::infra::{ConnectionHealthMonitor, ConnectionSupervisor};
use crate::market_maker::process_models::{HJBInventoryController, LiquidationCascadeDetector};
use crate::market_maker::StochasticConfig;

/// Mutable references needed by AllMids handler.
pub struct AllMidsState<'a> {
    /// Parameter estimator
    pub estimator: &'a mut ParameterEstimator,
    /// Connection health monitor (low-level)
    pub connection_health: &'a mut ConnectionHealthMonitor,
    /// Connection supervisor (high-level)
    pub connection_supervisor: &'a ConnectionSupervisor,
    /// Adverse selection estimator
    pub adverse_selection: &'a mut AdverseSelectionEstimator,
    /// Depth-dependent AS model
    pub depth_decay_as: &'a mut DepthDecayAS,
    /// Liquidation cascade detector
    pub liquidation_detector: &'a mut LiquidationCascadeDetector,
    /// HJB inventory controller
    pub hjb_controller: &'a mut HJBInventoryController,
    /// Stochastic module configuration
    pub stochastic_config: &'a StochasticConfig,
    /// Latest mid price (mutable to update)
    pub latest_mid: &'a mut f64,
}

/// Result of processing AllMids message.
#[derive(Debug)]
pub struct AllMidsResult {
    /// The new mid price
    pub mid: f64,
    /// Whether we should update quotes
    pub should_update_quotes: bool,
}

/// Process an AllMids message.
///
/// Updates:
/// - Mid price
/// - Estimator
/// - Connection health
/// - Adverse selection (resolves pending fills)
/// - Liquidation detector
/// - HJB controller
/// - Depth decay AS (if calibration enabled)
pub fn process_all_mids(
    all_mids: &AllMids,
    ctx: &MessageContext,
    state: &mut AllMidsState,
) -> Result<Option<AllMidsResult>> {
    // Get mid for our asset
    let mids = &all_mids.data.mids;
    let Some(mid_str) = mids.get(&*ctx.asset) else {
        return Ok(None);
    };

    let mid: f64 = mid_str
        .parse()
        .map_err(|_| crate::Error::FloatStringParse)?;

    // Update latest mid
    *state.latest_mid = mid;
    state.estimator.on_mid_update(mid);

    // Connection health tracking (both low-level monitor and high-level supervisor)
    state.connection_health.record_data_received();
    state.connection_supervisor.record_market_data();

    // Tier 1: Update AS estimator (resolves pending fills)
    state.adverse_selection.update(mid);
    state.adverse_selection.update_signals(
        state.estimator.sigma_total(),
        state.estimator.sigma_clean(),
        state.estimator.flow_imbalance(),
        state.estimator.jump_ratio(),
    );

    // Tier 1: Periodic update of liquidation detector
    state.liquidation_detector.update();

    // Stochastic modules
    state
        .hjb_controller
        .update_sigma(state.estimator.sigma_clean());
    if state.stochastic_config.calibrate_depth_as {
        state.depth_decay_as.resolve_pending_fills(mid);
    }

    debug!(
        asset = %ctx.asset,
        mid = %format!("{:.2}", mid),
        sigma = %format!("{:.6}", state.estimator.sigma_clean()),
        "AllMids processed"
    );

    Ok(Some(AllMidsResult {
        mid,
        should_update_quotes: true,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::{
        AdverseSelectionConfig, EstimatorConfig, HJBConfig, LiquidationConfig,
    };
    use std::sync::Arc;

    fn make_test_state<'a>(
        estimator: &'a mut ParameterEstimator,
        connection_health: &'a mut ConnectionHealthMonitor,
        connection_supervisor: &'a ConnectionSupervisor,
        adverse_selection: &'a mut AdverseSelectionEstimator,
        depth_decay_as: &'a mut DepthDecayAS,
        liquidation_detector: &'a mut LiquidationCascadeDetector,
        hjb_controller: &'a mut HJBInventoryController,
        stochastic_config: &'a StochasticConfig,
        latest_mid: &'a mut f64,
    ) -> AllMidsState<'a> {
        AllMidsState {
            estimator,
            connection_health,
            connection_supervisor,
            adverse_selection,
            depth_decay_as,
            liquidation_detector,
            hjb_controller,
            stochastic_config,
            latest_mid,
        }
    }

    #[test]
    fn test_process_updates_mid() {
        let mut estimator = ParameterEstimator::new(EstimatorConfig::default());
        let mut connection_health = ConnectionHealthMonitor::new();
        let connection_supervisor = ConnectionSupervisor::new();
        let mut adverse_selection =
            AdverseSelectionEstimator::new(AdverseSelectionConfig::default());
        let mut depth_decay_as = DepthDecayAS::default();
        let mut liquidation_detector =
            LiquidationCascadeDetector::new(LiquidationConfig::default());
        let mut hjb_controller = HJBInventoryController::new(HJBConfig::default());
        let stochastic_config = StochasticConfig::default();
        let mut latest_mid = 0.0;

        let mut state = make_test_state(
            &mut estimator,
            &mut connection_health,
            &connection_supervisor,
            &mut adverse_selection,
            &mut depth_decay_as,
            &mut liquidation_detector,
            &mut hjb_controller,
            &stochastic_config,
            &mut latest_mid,
        );

        let ctx = MessageContext::new(Arc::from("BTC"), 0.0, 0.0, 1.0, false, Arc::from("USDC"));

        // Create AllMids with BTC mid price
        let mut mids = std::collections::HashMap::new();
        mids.insert("BTC".to_string(), "50000.0".to_string());
        let all_mids = AllMids {
            data: crate::types::AllMidsData { mids },
        };

        let result = process_all_mids(&all_mids, &ctx, &mut state).unwrap();

        assert!(result.is_some());
        let result = result.unwrap();
        assert!((result.mid - 50000.0).abs() < f64::EPSILON);
        assert!(result.should_update_quotes);
        assert!((latest_mid - 50000.0).abs() < f64::EPSILON);
    }
}
