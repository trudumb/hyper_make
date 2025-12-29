//! Fill processor - orchestrates fill state updates across modules.
//!
//! This module provides a clean interface for processing fills, replacing the
//! scattered fill handling logic in mod.rs. It uses the FillPipeline for
//! deduplication while providing direct state access for modules that need it.
//!
//! # Design
//!
//! The FillProcessor takes a FillState bundle containing mutable references to
//! all modules that need to be updated on fill. This avoids the ownership issues
//! that would arise from having long-lived references stored in the processor.
//!
//! ```ignore
//! let processor = FillProcessor::new();
//! let mut state = FillState { position, orders, ... };
//! let result = processor.process(fill_event, &mut state);
//! ```

use super::dedup::FillDeduplicator;
use super::{FillEvent, FillResult};
use crate::market_maker::adverse_selection::{AdverseSelectionEstimator, DepthDecayAS};
use crate::market_maker::estimator::ParameterEstimator;
use crate::market_maker::messages;
use crate::market_maker::metrics::PrometheusMetrics;
use crate::market_maker::order_manager::{OrderManager, Side};
use crate::market_maker::pnl::PnLTracker;
use crate::market_maker::position::PositionTracker;
use crate::market_maker::queue::QueuePositionTracker;
use crate::market_maker::config::MetricsRecorder;
use tracing::{debug, info, warn};

/// Fill processing result with additional context.
#[derive(Debug, Clone)]
pub struct FillProcessingResult {
    /// Base fill result (is_new, consumers_notified)
    pub fill_result: FillResult,
    /// Was the order tracked in OrderManager?
    pub order_found: bool,
    /// Was this a new fill for the order (not already recorded)?
    pub is_new_fill: bool,
    /// Is the order now fully filled?
    pub is_complete: bool,
    /// Placement price if found
    pub placement_price: Option<f64>,
}

impl FillProcessingResult {
    /// Create result for a duplicate fill.
    pub fn duplicate() -> Self {
        Self {
            fill_result: FillResult::duplicate(),
            order_found: false,
            is_new_fill: false,
            is_complete: false,
            placement_price: None,
        }
    }
}

/// State bundle for fill processing.
///
/// Contains mutable references to all modules that need to be updated on fill.
/// This pattern avoids ownership issues by borrowing for the duration of processing.
pub struct FillState<'a> {
    // Core state
    /// Position tracker
    pub position: &'a mut PositionTracker,
    /// Order manager
    pub orders: &'a mut OrderManager,

    // Tier 1 modules
    /// Adverse selection estimator
    pub adverse_selection: &'a mut AdverseSelectionEstimator,
    /// Depth-dependent AS model
    pub depth_decay_as: &'a mut DepthDecayAS,
    /// Queue position tracker
    pub queue_tracker: &'a mut QueuePositionTracker,

    // Tier 2 modules
    /// Parameter estimator
    pub estimator: &'a mut ParameterEstimator,
    /// P&L tracker
    pub pnl_tracker: &'a mut PnLTracker,

    // Infrastructure
    /// Prometheus metrics
    pub prometheus: &'a mut PrometheusMetrics,
    /// Optional metrics recorder (MetricsRecorder is Option<Arc<dyn ...>>)
    pub metrics: &'a MetricsRecorder,

    // Context
    /// Current mid price
    pub latest_mid: f64,
    /// Asset being traded
    pub asset: &'a str,
    /// Max position for threshold warnings
    pub max_position: f64,
    /// Whether depth AS calibration is enabled
    pub calibrate_depth_as: bool,
}

/// Fill processor - coordinates fill handling across modules.
///
/// Provides centralized fill deduplication and orchestrates updates to all
/// modules that need fill information. Replaces the 170-line fill handling
/// block in mod.rs with a clean, testable interface.
pub struct FillProcessor {
    /// Centralized deduplication
    deduplicator: FillDeduplicator,
}

impl FillProcessor {
    /// Create a new fill processor.
    pub fn new() -> Self {
        Self {
            deduplicator: FillDeduplicator::new(),
        }
    }

    /// Create with custom deduplication capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            deduplicator: FillDeduplicator::with_capacity(capacity),
        }
    }

    /// Process a fill through all modules.
    ///
    /// This is the main entry point for fill handling. It:
    /// 1. Deduplicates by trade ID
    /// 2. Updates position
    /// 3. Updates order tracking
    /// 4. Determines placement price (from orders or pending)
    /// 5. Records to all analytics modules if placement is known
    /// 6. Logs fill and threshold warnings
    ///
    /// Returns detailed processing result.
    pub fn process(&mut self, fill: &FillEvent, state: &mut FillState) -> FillProcessingResult {
        // Step 1: Centralized deduplication
        if !self.deduplicator.check_and_mark(fill.tid) {
            debug!(tid = fill.tid, "Fill is duplicate, skipping");
            return FillProcessingResult::duplicate();
        }

        // Step 2: Update position
        state.position.process_fill(fill.size, fill.is_buy);

        // Step 3: Update order tracking
        let (order_found, is_new_fill, is_complete) = state
            .orders
            .process_fill(fill.oid, fill.tid, fill.size);

        // Step 4: Determine placement price
        let placement_price = if order_found {
            // Order found by OID - get placement price directly
            state.orders.get_order(fill.oid).map(|o| o.price)
        } else {
            // Order not found by OID - check pending orders by (side, fill_price)
            // This handles the race condition when fill arrives before OID is registered
            let side = if fill.is_buy { Side::Buy } else { Side::Sell };
            if let Some(pending) = state.orders.get_pending(side, fill.price) {
                debug!(
                    oid = fill.oid,
                    tid = fill.tid,
                    fill_price = fill.price,
                    placement_price = pending.price,
                    "Fill matched to pending order (immediate fill race condition)"
                );
                Some(pending.price)
            } else {
                // Truly untracked - not in orders or pending
                warn!(
                    "[Fill] Untracked order filled: oid={} tid={} {} {} {} | position updated to {}",
                    fill.oid,
                    fill.tid,
                    if fill.is_buy { "bought" } else { "sold" },
                    fill.size,
                    state.asset,
                    state.position.position()
                );
                None
            }
        };

        // Step 5: Process fill if we have placement info
        if placement_price.is_some() && (!order_found || is_new_fill) {
            self.record_fill_analytics(fill, placement_price, state);
            self.update_queue_tracking(fill, order_found, is_complete, state);
        }

        // Step 6: Position threshold warnings
        messages::check_position_thresholds(
            state.position.position(),
            state.max_position,
            state.asset,
        );

        FillProcessingResult {
            fill_result: FillResult::new_fill(1), // All modules updated as one
            order_found,
            is_new_fill,
            is_complete,
            placement_price,
        }
    }

    /// Record fill to all analytics modules.
    fn record_fill_analytics(
        &self,
        fill: &FillEvent,
        placement_price: Option<f64>,
        state: &mut FillState,
    ) {
        let placement = placement_price.unwrap_or(fill.price);

        // Tier 1: Adverse selection measurement
        state.adverse_selection.record_fill(
            fill.tid,
            fill.size,
            fill.is_buy,
            state.latest_mid,
        );

        // Stochastic: Depth-aware AS calibration
        if state.calibrate_depth_as {
            state.depth_decay_as.record_pending_fill(
                fill.tid,
                fill.price,
                fill.size,
                fill.is_buy,
                state.latest_mid,
            );
        }

        // Estimator: Feed own fill rate for kappa
        let timestamp_ms = fill.timestamp_ms();
        state.estimator.on_own_fill(
            timestamp_ms,
            placement,
            fill.price,
            fill.size,
            fill.is_buy,
        );

        // Tier 2: P&L tracking
        state.pnl_tracker.record_fill(
            fill.tid,
            fill.price,
            fill.size,
            fill.is_buy,
            state.latest_mid,
        );

        // Infrastructure: Prometheus metrics
        state.prometheus.record_fill(fill.size, fill.is_buy);

        // Log fill summary
        let pnl_summary = state.pnl_tracker.summary(state.latest_mid);
        info!(
            "[Fill] {} {} {} | oid={} tid={} | position: {} | AS: {:.2}bps | P&L: ${:.2}",
            if fill.is_buy { "bought" } else { "sold" },
            fill.size,
            state.asset,
            fill.oid,
            fill.tid,
            state.position.position(),
            state.adverse_selection.realized_as_bps(),
            pnl_summary.total_pnl
        );

        // Optional metrics recorder
        if let Some(m) = state.metrics {
            m.record_fill(fill.size, fill.is_buy);
            m.update_position(state.position.position());
        }
    }

    /// Update queue tracking based on fill completeness.
    fn update_queue_tracking(
        &self,
        fill: &FillEvent,
        order_found: bool,
        is_complete: bool,
        state: &mut FillState,
    ) {
        if order_found && !is_complete {
            // Partial fill - update queue position
            state.queue_tracker.order_partially_filled(fill.oid, fill.size);
        }
        // Note: For complete fills, cleanup() in mod.rs handles removal
        // Note: For pending-matched fills, queue tracking is handled when order is finalized
    }

    /// Check if a trade ID has already been processed.
    pub fn is_duplicate(&self, tid: u64) -> bool {
        self.deduplicator.is_duplicate(tid)
    }

    /// Get the number of tracked fills.
    pub fn tracked_fill_count(&self) -> usize {
        self.deduplicator.len()
    }

    /// Clear the deduplication cache.
    ///
    /// Use with caution - could cause double-processing.
    pub fn clear_dedup_cache(&mut self) {
        self.deduplicator.clear();
    }
}

impl Default for FillProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::adverse_selection::AdverseSelectionConfig;
    use crate::market_maker::pnl::PnLConfig;
    use crate::market_maker::queue::QueueConfig;
    use crate::market_maker::EstimatorConfig;

    fn make_fill(tid: u64, oid: u64, size: f64, price: f64, is_buy: bool) -> FillEvent {
        FillEvent::new(
            tid,
            oid,
            size,
            price,
            is_buy,
            price, // mid_at_fill = price for simplicity
            Some(price), // placement_price = price
            "BTC".to_string(),
        )
    }

    fn make_test_state<'a>(
        position: &'a mut PositionTracker,
        orders: &'a mut OrderManager,
        adverse_selection: &'a mut AdverseSelectionEstimator,
        depth_decay_as: &'a mut DepthDecayAS,
        queue_tracker: &'a mut QueuePositionTracker,
        estimator: &'a mut ParameterEstimator,
        pnl_tracker: &'a mut PnLTracker,
        prometheus: &'a mut PrometheusMetrics,
        metrics: &'a MetricsRecorder,
    ) -> FillState<'a> {
        FillState {
            position,
            orders,
            adverse_selection,
            depth_decay_as,
            queue_tracker,
            estimator,
            pnl_tracker,
            prometheus,
            metrics,
            latest_mid: 50000.0,
            asset: "BTC",
            max_position: 10.0,
            calibrate_depth_as: true,
        }
    }

    #[test]
    fn test_processor_new_fill() {
        let mut processor = FillProcessor::new();
        let mut position = PositionTracker::new(0.0);
        let mut orders = OrderManager::new();
        let mut adverse_selection = AdverseSelectionEstimator::new(AdverseSelectionConfig::default());
        let mut depth_decay_as = DepthDecayAS::default();
        let mut queue_tracker = QueuePositionTracker::new(QueueConfig::default());
        let mut estimator = ParameterEstimator::new(EstimatorConfig::default());
        let mut pnl_tracker = PnLTracker::new(PnLConfig::default());
        let mut prometheus = PrometheusMetrics::new();
        let metrics: MetricsRecorder = None;

        let mut state = make_test_state(
            &mut position,
            &mut orders,
            &mut adverse_selection,
            &mut depth_decay_as,
            &mut queue_tracker,
            &mut estimator,
            &mut pnl_tracker,
            &mut prometheus,
            &metrics,
        );

        let fill = make_fill(1, 100, 1.0, 50000.0, true);
        let result = processor.process(&fill, &mut state);

        assert!(result.fill_result.is_new);
        // Position should be updated
        assert!((position.position() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_processor_duplicate_rejected() {
        let mut processor = FillProcessor::new();
        let mut position = PositionTracker::new(0.0);
        let mut orders = OrderManager::new();
        let mut adverse_selection = AdverseSelectionEstimator::new(AdverseSelectionConfig::default());
        let mut depth_decay_as = DepthDecayAS::default();
        let mut queue_tracker = QueuePositionTracker::new(QueueConfig::default());
        let mut estimator = ParameterEstimator::new(EstimatorConfig::default());
        let mut pnl_tracker = PnLTracker::new(PnLConfig::default());
        let mut prometheus = PrometheusMetrics::new();
        let metrics: MetricsRecorder = None;

        let mut state = make_test_state(
            &mut position,
            &mut orders,
            &mut adverse_selection,
            &mut depth_decay_as,
            &mut queue_tracker,
            &mut estimator,
            &mut pnl_tracker,
            &mut prometheus,
            &metrics,
        );

        let fill1 = make_fill(1, 100, 1.0, 50000.0, true);
        let fill2 = make_fill(1, 100, 1.0, 50000.0, true); // Same TID

        let result1 = processor.process(&fill1, &mut state);
        let result2 = processor.process(&fill2, &mut state);

        assert!(result1.fill_result.is_new);
        assert!(!result2.fill_result.is_new); // Duplicate rejected

        // Position should only be updated once
        assert!((position.position() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_processor_is_duplicate() {
        let mut processor = FillProcessor::new();

        assert!(!processor.is_duplicate(1));

        // Mark as seen
        processor.deduplicator.check_and_mark(1);

        assert!(processor.is_duplicate(1));
        assert!(!processor.is_duplicate(2));
    }

    #[test]
    fn test_processor_clear_cache() {
        let mut processor = FillProcessor::new();
        processor.deduplicator.check_and_mark(1);
        assert!(processor.is_duplicate(1));

        processor.clear_dedup_cache();
        assert!(!processor.is_duplicate(1));
    }
}
