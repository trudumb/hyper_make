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
use crate::market_maker::config::MetricsRecorder;
use crate::market_maker::estimator::ParameterEstimator;
use crate::market_maker::infra::PrometheusMetrics;
use crate::market_maker::messages;
use crate::market_maker::tracking::{
    OrderManager, PnLTracker, PositionTracker, QueuePositionTracker, Side,
};
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
    /// Was this a WebSocket confirmation for an immediate fill (position already updated)?
    pub is_immediate_fill_confirmation: bool,
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
            is_immediate_fill_confirmation: false,
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
    /// OIDs of orders that filled immediately (from API response).
    /// When WebSocket fill arrives for these OIDs, skip position update
    /// since position was already updated from API response.
    immediate_fill_oids: std::collections::HashSet<u64>,
}

impl FillProcessor {
    /// Create a new fill processor.
    pub fn new() -> Self {
        Self {
            deduplicator: FillDeduplicator::new(),
            immediate_fill_oids: std::collections::HashSet::new(),
        }
    }

    /// Create with custom deduplication capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            deduplicator: FillDeduplicator::with_capacity(capacity),
            immediate_fill_oids: std::collections::HashSet::new(),
        }
    }

    /// Pre-register an OID as having filled immediately from API response.
    ///
    /// When an order fills immediately (API returns filled=true before WebSocket),
    /// position is updated from the API response. This method registers the OID
    /// so that when the WebSocket fill arrives later, we skip position update
    /// to prevent double-counting.
    ///
    /// The OID is automatically removed when the corresponding WebSocket fill
    /// is processed, or can be manually cleared via `clear_immediate_fill_oid()`.
    pub fn pre_register_immediate_fill(&mut self, oid: u64) {
        self.immediate_fill_oids.insert(oid);
        debug!(oid, "Pre-registered immediate fill OID for dedup");
    }

    /// Check if an OID was pre-registered as an immediate fill.
    pub fn is_immediate_fill(&self, oid: u64) -> bool {
        self.immediate_fill_oids.contains(&oid)
    }

    /// Clear an OID from the immediate fill set (after WebSocket confirmation).
    pub fn clear_immediate_fill_oid(&mut self, oid: u64) {
        self.immediate_fill_oids.remove(&oid);
    }

    /// Get count of pending immediate fill OIDs.
    pub fn immediate_fill_count(&self) -> usize {
        self.immediate_fill_oids.len()
    }

    /// Process a fill through all modules.
    ///
    /// This is the main entry point for fill handling. It:
    /// 1. Deduplicates by trade ID
    /// 2. Checks for immediate fill (position already updated from API)
    /// 3. Updates position (unless immediate fill)
    /// 4. Updates order tracking
    /// 5. Determines placement price (from orders or pending)
    /// 6. Records to all analytics modules if placement is known
    /// 7. Logs fill and threshold warnings
    ///
    /// Returns detailed processing result.
    pub fn process(&mut self, fill: &FillEvent, state: &mut FillState) -> FillProcessingResult {
        // Step 1: Centralized deduplication
        if !self.deduplicator.check_and_mark(fill.tid) {
            debug!(tid = fill.tid, "Fill is duplicate, skipping");
            return FillProcessingResult::duplicate();
        }

        // Step 2: Check if this is a WebSocket confirmation for an immediate fill
        // If so, position was already updated from API response - skip position update!
        let is_immediate_fill_confirmation = self.immediate_fill_oids.remove(&fill.oid);
        if is_immediate_fill_confirmation {
            info!(
                oid = fill.oid,
                tid = fill.tid,
                size = fill.size,
                "WebSocket fill for immediate-fill order - position already updated, skipping"
            );
        }

        // Step 3: Update position (unless already updated from API for immediate fills)
        if !is_immediate_fill_confirmation {
            state.position.process_fill(fill.size, fill.is_buy);
        }

        // Step 4: Update order tracking
        let (order_found, is_new_fill, is_complete) =
            state.orders.process_fill(fill.oid, fill.tid, fill.size);

        // Step 5: Determine placement price using CLOID-first lookup (Phase 1 Fix)
        //
        // Lookup priority:
        // 1. By OID (if order already tracked)
        // 2. By CLOID (primary - deterministic, eliminates timing race)
        // 3. By (side, price) (fallback - for edge cases where CLOID missing)
        let placement_price = if order_found {
            // Order found by OID - get placement price directly
            state.orders.get_order(fill.oid).map(|o| o.price)
        } else {
            // Order not found by OID - try CLOID lookup first (Phase 1 Fix)
            let side = if fill.is_buy { Side::Buy } else { Side::Sell };

            // Check if fill has CLOID (TradeInfo.cloid field)
            if let Some(ref cloid) = fill.cloid {
                if let Some(pending) = state.orders.get_pending_by_cloid(cloid) {
                    debug!(
                        oid = fill.oid,
                        tid = fill.tid,
                        cloid = %cloid,
                        fill_price = fill.price,
                        placement_price = pending.price,
                        "Fill matched to pending order by CLOID (deterministic lookup)"
                    );
                    Some(pending.price)
                } else {
                    // CLOID provided but not found in pending - fall through to price lookup
                    None
                }
            } else {
                None
            }
            .or_else(|| {
                // Fallback: Check pending orders by (side, fill_price)
                // This handles the race condition when fill arrives before OID is registered
                // or when CLOID is not provided in the fill
                if let Some(pending) = state.orders.get_pending(side, fill.price) {
                    debug!(
                        oid = fill.oid,
                        tid = fill.tid,
                        fill_price = fill.price,
                        placement_price = pending.price,
                        "Fill matched to pending order by price (fallback lookup)"
                    );
                    Some(pending.price)
                } else {
                    // Truly untracked - not in orders, not by CLOID, not by price
                    warn!(
                        "[Fill] Untracked order filled: oid={} tid={} cloid={:?} {} {} {} | position updated to {}",
                        fill.oid,
                        fill.tid,
                        fill.cloid,
                        if fill.is_buy { "bought" } else { "sold" },
                        fill.size,
                        state.asset,
                        state.position.position()
                    );
                    None
                }
            })
        };

        // Step 6: Process fill if we have placement info
        if placement_price.is_some() && (!order_found || is_new_fill) {
            self.record_fill_analytics(fill, placement_price, state);
            self.update_queue_tracking(fill, order_found, is_complete, state);
        }

        // Step 7: Position threshold warnings
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
            is_immediate_fill_confirmation,
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
        state
            .adverse_selection
            .record_fill(fill.tid, fill.size, fill.is_buy, state.latest_mid);

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
        state
            .estimator
            .on_own_fill(timestamp_ms, placement, fill.price, fill.size, fill.is_buy);

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
            state
                .queue_tracker
                .order_partially_filled(fill.oid, fill.size);
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
    use crate::market_maker::tracking::{PnLConfig, QueueConfig};
    use crate::market_maker::EstimatorConfig;

    fn make_fill(tid: u64, oid: u64, size: f64, price: f64, is_buy: bool) -> FillEvent {
        FillEvent::new(
            tid,
            oid,
            size,
            price,
            is_buy,
            price,       // mid_at_fill = price for simplicity
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
        let mut adverse_selection =
            AdverseSelectionEstimator::new(AdverseSelectionConfig::default());
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
        let mut adverse_selection =
            AdverseSelectionEstimator::new(AdverseSelectionConfig::default());
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

    #[test]
    fn test_immediate_fill_registration() {
        let mut processor = FillProcessor::new();

        // Initially no immediate fills
        assert!(!processor.is_immediate_fill(100));
        assert_eq!(processor.immediate_fill_count(), 0);

        // Register an immediate fill
        processor.pre_register_immediate_fill(100);
        assert!(processor.is_immediate_fill(100));
        assert_eq!(processor.immediate_fill_count(), 1);

        // Register another
        processor.pre_register_immediate_fill(200);
        assert!(processor.is_immediate_fill(200));
        assert_eq!(processor.immediate_fill_count(), 2);

        // Clear one
        processor.clear_immediate_fill_oid(100);
        assert!(!processor.is_immediate_fill(100));
        assert!(processor.is_immediate_fill(200));
        assert_eq!(processor.immediate_fill_count(), 1);
    }

    #[test]
    fn test_immediate_fill_skips_position_update() {
        let mut processor = FillProcessor::new();
        let mut position = PositionTracker::new(1.0); // Start with position 1.0 (from API)
        let mut orders = OrderManager::new();
        let mut adverse_selection =
            AdverseSelectionEstimator::new(AdverseSelectionConfig::default());
        let mut depth_decay_as = DepthDecayAS::default();
        let mut queue_tracker = QueuePositionTracker::new(QueueConfig::default());
        let mut estimator = ParameterEstimator::new(EstimatorConfig::default());
        let mut pnl_tracker = PnLTracker::new(PnLConfig::default());
        let mut prometheus = PrometheusMetrics::new();
        let metrics: MetricsRecorder = None;

        // Pre-register OID 100 as immediate fill (simulating API returned filled=true)
        processor.pre_register_immediate_fill(100);

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

        // Now WebSocket fill arrives for OID 100
        let fill = make_fill(1, 100, 1.0, 50000.0, true);
        let result = processor.process(&fill, &mut state);

        // Should be marked as immediate fill confirmation
        assert!(result.is_immediate_fill_confirmation);
        assert!(result.fill_result.is_new);

        // Position should NOT be updated (should stay at 1.0, not become 2.0)
        assert!(
            (position.position() - 1.0).abs() < f64::EPSILON,
            "Position was {}, expected 1.0 (should not have been updated)",
            position.position()
        );

        // OID should be cleared from immediate fill set
        assert!(!processor.is_immediate_fill(100));
    }

    #[test]
    fn test_normal_fill_updates_position() {
        let mut processor = FillProcessor::new();
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

        // No pre-registration - this is a normal fill

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

        // Should NOT be marked as immediate fill confirmation
        assert!(!result.is_immediate_fill_confirmation);
        assert!(result.fill_result.is_new);

        // Position SHOULD be updated
        assert!(
            (position.position() - 1.0).abs() < f64::EPSILON,
            "Position was {}, expected 1.0 (should have been updated)",
            position.position()
        );
    }

    #[test]
    fn test_immediate_fill_then_additional_fills() {
        // Scenario: Order partially fills immediately, then more fills arrive via WS
        let mut processor = FillProcessor::new();
        let mut position = PositionTracker::new(0.5); // Start with 0.5 from API immediate fill
        let mut orders = OrderManager::new();
        let mut adverse_selection =
            AdverseSelectionEstimator::new(AdverseSelectionConfig::default());
        let mut depth_decay_as = DepthDecayAS::default();
        let mut queue_tracker = QueuePositionTracker::new(QueueConfig::default());
        let mut estimator = ParameterEstimator::new(EstimatorConfig::default());
        let mut pnl_tracker = PnLTracker::new(PnLConfig::default());
        let mut prometheus = PrometheusMetrics::new();
        let metrics: MetricsRecorder = None;

        // Pre-register OID 100 as immediate fill
        processor.pre_register_immediate_fill(100);

        // First WS fill arrives (confirmation of immediate fill)
        let fill1 = make_fill(1, 100, 0.5, 50000.0, true);
        let result1 = {
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
            processor.process(&fill1, &mut state)
        };

        assert!(result1.is_immediate_fill_confirmation);
        // Position stays at 0.5 (no double-count)
        assert!((position.position() - 0.5).abs() < f64::EPSILON);

        // Second WS fill arrives (additional fill, not pre-registered)
        let fill2 = make_fill(2, 100, 0.3, 50000.0, true);
        let result2 = {
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
            processor.process(&fill2, &mut state)
        };

        assert!(!result2.is_immediate_fill_confirmation); // Not an immediate fill
        // Position should now be 0.5 + 0.3 = 0.8
        assert!(
            (position.position() - 0.8).abs() < f64::EPSILON,
            "Position was {}, expected 0.8",
            position.position()
        );
    }
}
