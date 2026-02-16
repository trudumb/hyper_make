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
use crate::market_maker::adverse_selection::{
    AdverseSelectionEstimator, DepthDecayAS, EnhancedASClassifier, PreFillASClassifier,
};
use crate::market_maker::config::MetricsRecorder;
use crate::market_maker::control::{PositionPnLTracker, StochasticController};
use crate::market_maker::tracking::calibration_wiring::ModelCalibrationOrchestrator;
use crate::market_maker::estimator::ParameterEstimator;
use crate::market_maker::infra::PrometheusMetrics;
use crate::market_maker::messages;
use crate::market_maker::strategy::MarketParams;
use crate::market_maker::tracking::{
    OrderManager, PnLTracker, PositionTracker, QueuePositionTracker, Side,
};
use serde::Serialize;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tracing::{debug, info, warn};

// ============================================================================
// Signal Diagnostic Infrastructure (Phase 1)
// ============================================================================

/// Snapshot of all signals at the moment of a fill.
///
/// This struct captures the complete signal state when a fill occurs,
/// enabling post-hoc analysis of which signals predicted toxic fills.
/// Exported to JSON for Python analysis scripts.
#[derive(Debug, Clone, Serialize)]
pub struct FillSignalSnapshot {
    // === Fill Metadata ===
    /// Trade ID (unique identifier)
    pub tid: u64,
    /// Order ID
    pub oid: u64,
    /// Fill timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
    /// Fill side ("bid" = we bought, "ask" = we sold)
    pub side: String,
    /// Fill price
    pub price: f64,
    /// Fill size
    pub size: f64,
    /// Asset being traded
    pub asset: String,

    // === Pre-Fill Classifier Signals ===
    /// Pre-fill toxicity score for this side [0, 1]
    pub pre_fill_toxicity: f64,
    /// Orderbook imbalance at fill time [-1, 1]
    pub orderbook_imbalance: f64,
    /// Trade flow imbalance at fill time [-1, 1]
    pub trade_flow_imbalance: f64,

    // === Regime Signals ===
    /// HMM regime (0=quiet, 1=trending, 2=volatile)
    pub hmm_regime: u8,
    /// HMM regime confidence [0, 1]
    pub hmm_confidence: f64,
    /// Changepoint probability [0, 1]
    pub cp_prob: f64,
    /// Is toxic regime active?
    pub is_toxic_regime: bool,
    /// Jump ratio (RV/BV)
    pub jump_ratio: f64,

    // === Directional Signals ===
    /// Continuation probability [0, 1]
    pub continuation_p: f64,
    /// Drift probability bearish [0, 1]
    pub drift_prob_bearish: f64,
    /// Momentum signal (bps)
    pub momentum_bps: f64,
    /// Falling knife score [0, 3]
    pub falling_knife_score: f64,
    /// Rising knife score [0, 3]
    pub rising_knife_score: f64,

    // === Market State ===
    /// Mid price at fill time
    pub mid_price: f64,
    /// Market spread (bps)
    pub spread_bps: f64,
    /// Volatility (per-second)
    pub volatility: f64,
    /// Funding rate (8h)
    pub funding_rate_8h: f64,
    /// Current position
    pub position: f64,

    // === Hawkes/Flow Signals ===
    /// Hawkes flow imbalance [-1, 1]
    pub hawkes_imbalance: f64,
    /// Hawkes cluster probability [0, 1]
    pub hawkes_p_cluster: f64,
    /// Hawkes excitation penalty [0.5, 1.0]
    pub hawkes_excitation_penalty: f64,

    // === Kappa/Fill Rate ===
    /// Estimated kappa (fill intensity)
    pub kappa: f64,
    /// Kappa uncertainty (std dev)
    pub kappa_uncertainty: f64,
    /// Adaptive kappa from Bayesian system
    pub adaptive_kappa: f64,

    // === Markout Outcomes (filled asynchronously) ===
    /// Price move 500ms after fill (bps, from fill side perspective)
    /// Negative = adverse selection (price moved against us)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markout_500ms_bps: Option<f64>,
    /// Price move 2s after fill (bps)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markout_2s_bps: Option<f64>,
    /// Price move 10s after fill (bps)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markout_10s_bps: Option<f64>,
}

impl FillSignalSnapshot {
    /// Create a new snapshot from a fill event and market params.
    pub fn from_fill_and_params(
        fill: &FillEvent,
        params: &MarketParams,
        asset: &str,
    ) -> Self {
        let side = if fill.is_buy { "bid" } else { "ask" };
        
        // Get pre-fill toxicity for the correct side
        let pre_fill_toxicity = if fill.is_buy {
            params.pre_fill_toxicity_bid
        } else {
            params.pre_fill_toxicity_ask
        };

        Self {
            // Fill metadata
            tid: fill.tid,
            oid: fill.oid,
            timestamp_ms: fill.timestamp_ms(),
            side: side.to_string(),
            price: fill.price,
            size: fill.size,
            asset: asset.to_string(),

            // Pre-fill classifier signals
            pre_fill_toxicity,
            orderbook_imbalance: params.book_imbalance,
            trade_flow_imbalance: params.flow_imbalance,

            // Regime signals
            hmm_regime: match params.volatility_regime {
                crate::market_maker::estimator::VolatilityRegime::Low => 0,
                crate::market_maker::estimator::VolatilityRegime::Normal => 0,
                crate::market_maker::estimator::VolatilityRegime::High => 1,
                crate::market_maker::estimator::VolatilityRegime::Extreme => 2,
            },
            hmm_confidence: 0.8, // TODO: wire up actual HMM confidence
            cp_prob: 0.0, // Will be filled from stochastic controller
            is_toxic_regime: params.is_toxic_regime,
            jump_ratio: params.jump_ratio,

            // Directional signals
            continuation_p: params.p_momentum_continue,
            drift_prob_bearish: if params.momentum_bps < 0.0 {
                params.p_momentum_continue
            } else {
                1.0 - params.p_momentum_continue
            },
            momentum_bps: params.momentum_bps,
            falling_knife_score: params.falling_knife_score,
            rising_knife_score: params.rising_knife_score,

            // Market state
            mid_price: fill.mid_at_fill,
            spread_bps: params.market_spread_bps,
            volatility: params.sigma,
            funding_rate_8h: params.funding_rate,
            position: 0.0, // Will be filled from FillState

            // Hawkes/flow signals
            hawkes_imbalance: params.hawkes_imbalance,
            hawkes_p_cluster: params.hawkes_p_cluster,
            hawkes_excitation_penalty: params.hawkes_excitation_penalty,

            // Kappa
            kappa: params.kappa,
            kappa_uncertainty: params.kappa_uncertainty,
            adaptive_kappa: params.adaptive_kappa,

            // Markouts (filled later)
            markout_500ms_bps: None,
            markout_2s_bps: None,
            markout_10s_bps: None,
        }
    }

    /// Set the position at fill time.
    pub fn with_position(mut self, position: f64) -> Self {
        self.position = position;
        self
    }

    /// Set the changepoint probability.
    pub fn with_cp_prob(mut self, cp_prob: f64) -> Self {
        self.cp_prob = cp_prob;
        self
    }
}

/// Pending markout tracking for a fill.
#[derive(Debug, Clone)]
pub struct PendingMarkout {
    /// Trade ID for matching
    pub tid: u64,
    /// Fill price for markout calculation
    pub fill_price: f64,
    /// Fill side (true = buy/bid, false = sell/ask)
    pub is_buy: bool,
    /// Fill timestamp
    pub fill_time: Instant,
    /// Timestamp in ms for horizon calculations
    pub fill_time_ms: u64,
    /// 500ms markout (filled when available)
    pub markout_500ms_bps: Option<f64>,
    /// 2s markout (filled when available)
    pub markout_2s_bps: Option<f64>,
    /// 10s markout (filled when available)
    pub markout_10s_bps: Option<f64>,
}

impl PendingMarkout {
    /// Create a new pending markout.
    pub fn new(tid: u64, fill_price: f64, is_buy: bool, fill_time_ms: u64) -> Self {
        Self {
            tid,
            fill_price,
            is_buy,
            fill_time: Instant::now(),
            fill_time_ms,
            markout_500ms_bps: None,
            markout_2s_bps: None,
            markout_10s_bps: None,
        }
    }

    /// Update markouts based on current mid price and elapsed time.
    /// Returns true if all markouts are now filled.
    pub fn update(&mut self, current_mid: f64, current_time_ms: u64) -> bool {
        let elapsed_ms = current_time_ms.saturating_sub(self.fill_time_ms);
        
        // Calculate markout: (current_mid - fill_price) / fill_price * 10000
        // For buys: positive markout = price went up = good
        // For sells: positive markout = price went down = good
        // We want: negative = adverse selection (price moved against us)
        let raw_markout_bps = (current_mid - self.fill_price) / self.fill_price * 10000.0;
        
        // Adjust sign: for sells, flip sign so negative = AS
        let markout_bps = if self.is_buy {
            raw_markout_bps  // Buy: price up = good (+), price down = bad (-)
        } else {
            -raw_markout_bps // Sell: price down = good (+), price up = bad (-)
        };

        // Fill markouts at appropriate horizons
        if self.markout_500ms_bps.is_none() && elapsed_ms >= 500 {
            self.markout_500ms_bps = Some(markout_bps);
        }
        if self.markout_2s_bps.is_none() && elapsed_ms >= 2000 {
            self.markout_2s_bps = Some(markout_bps);
        }
        if self.markout_10s_bps.is_none() && elapsed_ms >= 10000 {
            self.markout_10s_bps = Some(markout_bps);
        }

        // Return true if all markouts filled
        self.markout_500ms_bps.is_some()
            && self.markout_2s_bps.is_some()
            && self.markout_10s_bps.is_some()
    }

    /// Check if this markout is stale (>30s old with no complete data).
    pub fn is_stale(&self) -> bool {
        self.fill_time.elapsed().as_secs() > 30 && self.markout_10s_bps.is_none()
    }
}

/// Store for fill signal snapshots with export capability.
#[derive(Debug)]
pub struct FillSignalStore {
    /// Completed snapshots ready for export
    snapshots: VecDeque<FillSignalSnapshot>,
    /// Pending markouts waiting for price updates
    pending_markouts: VecDeque<PendingMarkout>,
    /// Maximum snapshots to keep in memory
    capacity: usize,
    /// Total fills recorded (for stats)
    total_recorded: u64,
    /// Total fills exported
    total_exported: u64,
    /// Export path (if configured)
    export_path: Option<String>,
    /// Last export time
    last_export: Instant,
}

impl FillSignalStore {
    /// Create a new store with default capacity.
    pub fn new() -> Self {
        Self {
            snapshots: VecDeque::with_capacity(1000),
            pending_markouts: VecDeque::with_capacity(100),
            capacity: 1000,
            total_recorded: 0,
            total_exported: 0,
            export_path: None,
            last_export: Instant::now(),
        }
    }

    /// Create with custom capacity and export path.
    pub fn with_config(capacity: usize, export_path: Option<String>) -> Self {
        Self {
            snapshots: VecDeque::with_capacity(capacity),
            pending_markouts: VecDeque::with_capacity(100),
            capacity,
            total_recorded: 0,
            total_exported: 0,
            export_path,
            last_export: Instant::now(),
        }
    }

    /// Record a new fill snapshot and start markout tracking.
    pub fn record(&mut self, snapshot: FillSignalSnapshot) {
        // Start tracking markout for this fill
        let pending = PendingMarkout::new(
            snapshot.tid,
            snapshot.price,
            snapshot.side == "bid",
            snapshot.timestamp_ms,
        );
        self.pending_markouts.push_back(pending);

        // Store snapshot (markouts will be filled later)
        self.snapshots.push_back(snapshot);
        self.total_recorded += 1;

        // Evict oldest if over capacity
        while self.snapshots.len() > self.capacity {
            self.snapshots.pop_front();
        }
    }

    /// Update markouts with new mid price.
    /// Call this on every AllMids update.
    pub fn update_markouts(&mut self, current_mid: f64, current_time_ms: u64) {
        // Update all pending markouts
        let mut completed_tids = Vec::new();
        
        for pending in self.pending_markouts.iter_mut() {
            if pending.update(current_mid, current_time_ms) {
                completed_tids.push((
                    pending.tid,
                    pending.markout_500ms_bps,
                    pending.markout_2s_bps,
                    pending.markout_10s_bps,
                ));
            }
        }

        // Remove completed and stale markouts
        self.pending_markouts.retain(|p| !p.is_stale() && p.markout_10s_bps.is_none());

        // Update snapshots with completed markouts
        for (tid, m500, m2s, m10s) in completed_tids {
            if let Some(snapshot) = self.snapshots.iter_mut().find(|s| s.tid == tid) {
                snapshot.markout_500ms_bps = m500;
                snapshot.markout_2s_bps = m2s;
                snapshot.markout_10s_bps = m10s;
            }
        }
    }

    /// Export snapshots to JSON file.
    pub fn export_to_json(&mut self, path: &str) -> std::io::Result<usize> {
        // Only export snapshots with complete markouts
        let complete: Vec<_> = self.snapshots
            .iter()
            .filter(|s| s.markout_10s_bps.is_some())
            .cloned()
            .collect();

        if complete.is_empty() {
            return Ok(0);
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write as JSON array
        serde_json::to_writer_pretty(&mut writer, &complete)
            .map_err(std::io::Error::other)?;
        writer.flush()?;

        let count = complete.len();
        self.total_exported += count as u64;
        self.last_export = Instant::now();

        info!(
            exported = count,
            total_recorded = self.total_recorded,
            total_exported = self.total_exported,
            path = %path,
            "Exported fill signal snapshots"
        );

        Ok(count)
    }

    /// Check if export is due (every 5 minutes or 100 fills).
    pub fn should_export(&self) -> bool {
        let fills_since_export = self.total_recorded - self.total_exported;
        let time_since_export = self.last_export.elapsed().as_secs();
        
        fills_since_export >= 100 || time_since_export >= 300
    }

    /// Get the configured export path.
    pub fn export_path(&self) -> Option<&str> {
        self.export_path.as_deref()
    }

    /// Set export path.
    pub fn set_export_path(&mut self, path: String) {
        self.export_path = Some(path);
    }

    /// Get stats for logging.
    pub fn stats(&self) -> (u64, u64, usize, usize) {
        (
            self.total_recorded,
            self.total_exported,
            self.snapshots.len(),
            self.pending_markouts.len(),
        )
    }
}

impl Default for FillSignalStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Original Fill Processing Infrastructure
// ============================================================================

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
    /// Pre-fill adverse selection classifier (online learning)
    pub pre_fill_classifier: &'a mut PreFillASClassifier,
    /// Enhanced AS classifier with microstructure features
    pub enhanced_classifier: &'a mut EnhancedASClassifier,

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

    // Learning
    /// Closed-loop learning module
    pub learning: &'a mut crate::market_maker::learning::LearningModule,

    // Layer 3: Stochastic Controller
    /// POMDP-based sequential decision-making controller
    pub stochastic_controller: &'a mut StochasticController,

    // Calibrated Thresholds
    /// Position P&L tracker for deriving position thresholds from actual P&L data
    pub position_pnl: &'a mut PositionPnLTracker,

    // Fee configuration for edge calculation
    /// Fee in basis points for edge calculation
    pub fee_bps: f64,
    
    // Bayesian learning
    /// Theoretical edge estimator for Bayesian alpha updates
    pub theoretical_edge: &'a mut crate::market_maker::control::TheoreticalEdgeEstimator,

    // Calibration
    /// Model calibration orchestrator for fill probability and AS predictions
    pub model_calibration: &'a mut ModelCalibrationOrchestrator,

    // Signal diagnostics (Phase 1)
    /// Fill signal store for diagnostic export
    pub signal_store: &'a mut FillSignalStore,
    /// Current market params for signal capture (set before process() call)
    pub market_params: Option<&'a MarketParams>,
}


/// A record of a recently cancelled order, used to handle the cancel-fill race condition.
///
/// When an order is cancelled but a fill arrives after (the exchange filled it before
/// processing our cancel), we can match the fill to this tombstone instead of logging
/// an "untracked order" warning. This allows proper attribution of the fill.
#[derive(Debug, Clone)]
pub struct TombstoneOrder {
    /// Side of the cancelled order
    pub side: Side,
    /// Limit price of the cancelled order
    pub price: f64,
    /// Original size of the cancelled order
    pub size: f64,
    /// When the cancellation was recorded
    pub cancelled_at: Instant,
}

/// Time-to-live for tombstone entries (2 seconds).
/// Fills arriving after this window are truly unexpected.
const TOMBSTONE_TTL: std::time::Duration = std::time::Duration::from_secs(2);

/// Fill processor - coordinates fill handling across modules.
///
/// Provides centralized fill deduplication and orchestrates updates to all
/// modules that need fill information. Replaces the 170-line fill handling
/// block in mod.rs with a clean, testable interface.
pub struct FillProcessor {
    /// Centralized deduplication
    deduplicator: FillDeduplicator,
    /// Remaining immediate fill amounts per OID.
    ///
    /// When an order fills immediately (API returns filled=true before WebSocket),
    /// we track how much was filled so we can properly dedup WS fills.
    /// Maps OID -> remaining amount to skip.
    ///
    /// For full immediate fills: register full size, first WS fill skips all.
    /// For partial immediate fills: register partial size, WS fills decrement
    /// until remaining reaches 0, then subsequent fills update position normally.
    immediate_fill_amounts: std::collections::HashMap<u64, f64>,
    /// Tombstone records for recently cancelled orders.
    ///
    /// When an order is cancelled, we record its metadata here so that if a fill
    /// arrives for it (cancel-fill race), we can match and attribute correctly
    /// instead of logging an "untracked order" warning.
    recently_cancelled: std::collections::HashMap<u64, TombstoneOrder>,
}

impl FillProcessor {
    /// Create a new fill processor.
    pub fn new() -> Self {
        Self {
            deduplicator: FillDeduplicator::new(),
            immediate_fill_amounts: std::collections::HashMap::new(),
            recently_cancelled: std::collections::HashMap::new(),
        }
    }

    /// Create with custom deduplication capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            deduplicator: FillDeduplicator::with_capacity(capacity),
            immediate_fill_amounts: std::collections::HashMap::new(),
            recently_cancelled: std::collections::HashMap::new(),
        }
    }

    /// Pre-register an immediate fill amount for an OID.
    ///
    /// When an order fills immediately (API returns filled=true before WebSocket),
    /// position is updated from the API response. This method registers the amount
    /// so that when WebSocket fills arrive later, we skip position updates up to
    /// that amount to prevent double-counting.
    ///
    /// For partial immediate fills, multiple WS fills may arrive. We decrement
    /// the remaining amount with each fill until it reaches 0, then subsequent
    /// fills update position normally.
    ///
    /// # Arguments
    /// * `oid` - Order ID
    /// * `amount` - Amount that was filled immediately (from API response)
    pub fn pre_register_immediate_fill(&mut self, oid: u64, amount: f64) {
        // Add to existing amount if already registered (shouldn't happen normally)
        let entry = self.immediate_fill_amounts.entry(oid).or_insert(0.0);
        *entry += amount;
        debug!(
            oid,
            amount,
            total = *entry,
            "Pre-registered immediate fill amount for dedup"
        );
    }

    /// Check if an OID has remaining immediate fill amount to skip.
    pub fn is_immediate_fill(&self, oid: u64) -> bool {
        self.immediate_fill_amounts
            .get(&oid)
            .map(|&amt| amt > 1e-10)
            .unwrap_or(false)
    }

    /// Get remaining immediate fill amount for an OID.
    pub fn get_immediate_fill_remaining(&self, oid: u64) -> f64 {
        self.immediate_fill_amounts
            .get(&oid)
            .copied()
            .unwrap_or(0.0)
    }

    /// Clear an OID from the immediate fill tracking.
    pub fn clear_immediate_fill_oid(&mut self, oid: u64) {
        self.immediate_fill_amounts.remove(&oid);
    }

    /// Get count of OIDs with pending immediate fill amounts.
    pub fn immediate_fill_count(&self) -> usize {
        self.immediate_fill_amounts.len()
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
    #[tracing::instrument(name = "fill_processing", skip_all, fields(oid = fill.oid, tid = fill.tid, side = %if fill.is_buy { "buy" } else { "sell" }))]
    pub fn process(&mut self, fill: &FillEvent, state: &mut FillState) -> FillProcessingResult {
        // Step 1: Centralized deduplication
        if !self.deduplicator.check_and_mark(fill.tid) {
            debug!(tid = fill.tid, "Fill is duplicate, skipping");
            return FillProcessingResult::duplicate();
        }

        // Step 2: Check if this fill (or part of it) was already counted from API response.
        // For partial immediate fills, we track the remaining amount to skip.
        let remaining_immediate = self
            .immediate_fill_amounts
            .get(&fill.oid)
            .copied()
            .unwrap_or(0.0);
        let skip_amount = remaining_immediate.min(fill.size);
        let update_amount = fill.size - skip_amount;
        let is_immediate_fill_confirmation = skip_amount > 1e-10;

        if is_immediate_fill_confirmation {
            // Decrement the remaining immediate fill amount
            if let Some(remaining) = self.immediate_fill_amounts.get_mut(&fill.oid) {
                *remaining -= skip_amount;
                if *remaining <= 1e-10 {
                    // All immediate fill amount consumed, remove entry
                    self.immediate_fill_amounts.remove(&fill.oid);
                }
            }

            if update_amount > 1e-10 {
                info!(
                    oid = fill.oid,
                    tid = fill.tid,
                    fill_size = fill.size,
                    skip_amount,
                    update_amount,
                    "Partial immediate fill dedup - skipping {} of {}, updating position by {}",
                    skip_amount,
                    fill.size,
                    update_amount
                );
            } else {
                info!(
                    oid = fill.oid,
                    tid = fill.tid,
                    size = fill.size,
                    "WebSocket fill for immediate-fill order - position already updated, skipping"
                );
            }
        }

        // Step 3: Update position by the non-immediate portion only
        if update_amount > 1e-10 {
            state.position.process_fill(update_amount, fill.is_buy);
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
                } else if let Some(tombstone) = self.recently_cancelled.remove(&fill.oid) {
                    // Cancel-fill race: order was cancelled but exchange filled it first.
                    // Use tombstone data for attribution.
                    info!(
                        "[Fill] Fill matched to recently cancelled order: oid={} tid={} side={:?} price={} size={} | tombstone age={:?}",
                        fill.oid,
                        fill.tid,
                        tombstone.side,
                        tombstone.price,
                        fill.size,
                        tombstone.cancelled_at.elapsed(),
                    );
                    Some(tombstone.price)
                } else {
                    // Truly untracked - not in orders, not by CLOID, not by price, not in tombstones
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
        // FIX: Use mid_at_placement (the mid when order was created) instead of latest_mid.
        // Previously both record_fill and update() used latest_mid, making AS = 0 (tautological).
        // Now we anchor to placement mid: AS = |mid_at_measurement - mid_at_placement| gives
        // the full adverse movement since we posted the order.
        let as_reference_mid = if fill.mid_at_placement > 0.0 {
            fill.mid_at_placement
        } else {
            state.latest_mid // Fallback for fills without placement tracking
        };
        state
            .adverse_selection
            .record_fill(fill.tid, fill.size, fill.is_buy, as_reference_mid);

        // Stochastic: Depth-aware AS calibration
        if state.calibrate_depth_as {
            state.depth_decay_as.record_pending_fill(
                fill.tid,
                fill.price,
                fill.size,
                fill.is_buy,
                as_reference_mid,
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

        // Learning: Update closed-loop learning module
        // Build minimal market params from estimator for learning update
        let params = MarketParams::from_estimator(
            state.estimator,
            state.latest_mid,
            state.position.position(),
            state.max_position,
        );

        if state.learning.is_enabled() {
            state
                .learning
                .on_fill(fill, &params, state.position.position());
        }

        // Update P&L peak tracking after fill
        let pnl_summary = state.pnl_tracker.summary_update_peak(state.latest_mid);

        // Layer 3: Update stochastic controller beliefs
        // This is the critical feedback loop that was missing!
        if state.stochastic_controller.is_enabled() {
            // Use actual drawdown from P&L tracking
            let drawdown = pnl_summary.drawdown();

            // Get learning module output for controller
            let learning_output =
                state
                    .learning
                    .output(&params, state.position.position(), drawdown);

            // Get realized adverse selection
            let realized_as_bps = state.adverse_selection.realized_as_bps();

            // Update controller beliefs and changepoint detection
            state
                .stochastic_controller
                .on_fill(fill, &learning_output, realized_as_bps);

            let cp_summary = state.stochastic_controller.changepoint_summary();
            debug!(
                realized_as_bps = %format!("{:.2}", realized_as_bps),
                belief_edge = %format!("{:.2}", state.stochastic_controller.belief().expected_edge()),
                n_fills = state.stochastic_controller.belief().n_fills,
                changepoint_prob_5 = %format!("{:.3}", cp_summary.cp_prob_5),
                drawdown = %format!("{:.2}", drawdown),
                "Layer 3: Updated beliefs from fill"
            );

            // Record calibration outcomes for fill probability and AS predictions.
            // The prediction IDs are stored on the TrackedOrder when the quote was placed.
            if let Some(order) = state.orders.get_order(fill.oid) {
                // Record fill probability outcome (prediction was made at quote time)
                if let Some(fill_pred_id) = order.fill_prediction_id {
                    state.model_calibration.fill_model.record_outcome(fill_pred_id, true);
                    debug!(
                        oid = fill.oid,
                        prediction_id = fill_pred_id,
                        "Recorded fill probability outcome: filled=true"
                    );
                }

                // Record adverse selection outcome with realized AS bps
                if let Some(as_pred_id) = order.as_prediction_id {
                    state.model_calibration.as_model.record_outcome(as_pred_id, realized_as_bps);
                    debug!(
                        oid = fill.oid,
                        prediction_id = as_pred_id,
                        realized_as_bps = %format!("{:.2}", realized_as_bps),
                        "Recorded AS prediction outcome"
                    );
                }
            }

            // === PreFillASClassifier Online Learning ===
            // Record outcome for online weight learning. This allows the classifier
            // to adapt its signal weights based on actual fill outcomes.
            let is_bid = fill.is_buy;
            let was_adverse = realized_as_bps > 3.0; // Same threshold as adverse_threshold_default
            state.pre_fill_classifier.record_outcome(
                is_bid,
                was_adverse,
                Some(realized_as_bps),
            );
            debug!(
                is_bid = is_bid,
                was_adverse = was_adverse,
                realized_as_bps = %format!("{:.2}", realized_as_bps),
                "Recorded pre-fill classifier outcome for online learning"
            );

            // === EnhancedASClassifier Online Learning ===
            // Record outcome for microstructure-based classifier with z-score features
            state.enhanced_classifier.record_outcome(
                is_bid,
                was_adverse,
                Some(realized_as_bps),
            );
            debug!(
                is_bid = is_bid,
                was_adverse = was_adverse,
                realized_as_bps = %format!("{:.2}", realized_as_bps),
                "Recorded enhanced classifier outcome for online learning"
            );
        }

        // === Calibrated P&L Tracking (IR-Based Thresholds) ===
        // Record P&L by position quantile and regime for threshold derivation.
        // This enables deriving position thresholds from actual P&L data.
        if state.max_position > 0.0 {
            let position_ratio = state.position.position().abs() / state.max_position;
            // Get regime from estimator (0=calm, 1=normal, 2=volatile/cascade)
            let regime = match state.estimator.volatility_regime() {
                crate::market_maker::estimator::VolatilityRegime::Low => 0,
                crate::market_maker::estimator::VolatilityRegime::Normal => 0,
                crate::market_maker::estimator::VolatilityRegime::High => 1,
                crate::market_maker::estimator::VolatilityRegime::Extreme => 2,
            };
            // P&L in bps: (fill_pnl / notional) * 10000
            let notional = fill.price * fill.size;
            if notional > 0.0 {
                let pnl_bps = (pnl_summary.total_pnl / notional) * 10000.0;
                state.position_pnl.record(position_ratio, regime, pnl_bps);
            }
        }

        // Log fill summary
        let as_bps = state.adverse_selection.realized_as_bps();
        info!(
            "[Fill] {} {} {} | oid={} tid={} | position: {} | AS: {:.2}bps | P&L: ${:.2}",
            if fill.is_buy { "bought" } else { "sold" },
            fill.size,
            state.asset,
            fill.oid,
            fill.tid,
            state.position.position(),
            as_bps,
            pnl_summary.total_pnl
        );

        // Record fill for dashboard display
        state
            .prometheus
            .record_fill_for_dashboard(pnl_summary.total_pnl, fill.is_buy, as_bps);

        // Record P&L attribution for dashboard breakdown
        // Note: adverse_selection and fees are typically negative values
        state.prometheus.record_pnl_attribution(
            pnl_summary.spread_capture,
            -pnl_summary.adverse_selection, // Convert to negative (it's tracked as positive loss)
            -pnl_summary.fees,              // Convert to negative (fees are costs)
        );

        // Record calibration data for dashboard prediction quality tracking
        // Note: We record that a fill happened (did_fill=true)
        // Using current kappa to estimate fill probability: p = kappa / (kappa + 1000)
        // This is a rough heuristic - proper calibration would track prediction at order time
        let kappa = state.estimator.kappa();
        let fill_prob_estimate = kappa / (kappa + 1000.0);
        let regime = format!("{:?}", state.estimator.volatility_regime());
        state
            .prometheus
            .record_fill_calibration(fill_prob_estimate, true, &regime);

        // Record AS calibration: consider "adverse" if AS exceeds 2 bps threshold
        let as_threshold_bps = 2.0;
        let was_adverse = as_bps > as_threshold_bps;
        // Use toxic regime flag as AS probability proxy
        let as_prob_estimate = if state.estimator.is_toxic_regime() {
            0.5
        } else {
            0.1
        };
        state
            .prometheus
            .record_as_calibration(as_prob_estimate, was_adverse, &regime);

        // === Bayesian Alpha Update ===
        // Update the theoretical edge estimator with fill outcome for adaptive learning.
        // This is the key feedback loop: did book imbalance predict price direction?
        //
        // For a buy fill:
        //   - Predicted direction: +1 if book_imbalance > 0 (bullish), -1 otherwise
        //   - Actual direction: determined by price movement after fill
        //
        // We use adverse selection as a proxy for price direction:
        //   - Positive AS (price moved against us) = prediction was wrong
        //   - Negative AS (price moved in our favor) = prediction was correct
        let book_imbalance = state.estimator.book_imbalance();
        let predicted_direction = if book_imbalance > 0.0 { 1i8 } else { -1 };
        
        // For buys: negative AS means price went up (correct if imbalance was positive)
        // For sells: negative AS means price went down (correct if imbalance was negative)
        // Simplified: was_adverse = false means our fill was "correct" (price moved in our favor)
        let actual_direction = if fill.is_buy {
            if was_adverse { -1i8 } else { 1 }  // Adverse on buy = price dropped
        } else if was_adverse { 1i8 } else { -1 };
        
        state.theoretical_edge.update_from_fill(
            book_imbalance,
            predicted_direction,
            actual_direction,
            timestamp_ms,
        );

        // Log learning update
        let summary = state.theoretical_edge.bayesian_summary();
        tracing::debug!(
            alpha = %format!("{:.3}", summary.alpha),
            uncertainty = %format!("{:.3}", summary.uncertainty),
            acc = %format!("{:.2}", summary.empirical_accuracy),
            fills = summary.total_fills,
            lambda = %format!("{:.2}", summary.fill_rate_per_hour),
            "Bayesian learning update"
        );

        // Optional metrics recorder
        if let Some(m) = state.metrics {
            m.record_fill(fill.size, fill.is_buy);
            m.update_position(state.position.position());
        }

        // === Signal Diagnostic Capture (Phase 1) ===
        // Record fill signal snapshot for calibration analysis.
        // Uses provided market_params if available, otherwise builds from estimator.
        let snapshot_params = state.market_params.cloned().unwrap_or(params);
        let cp_prob = state.stochastic_controller.changepoint_summary().cp_prob_5;
        
        let snapshot = FillSignalSnapshot::from_fill_and_params(fill, &snapshot_params, state.asset)
            .with_position(state.position.position())
            .with_cp_prob(cp_prob);
        
        state.signal_store.record(snapshot);
        
        // Log diagnostic capture
        let (recorded, exported, stored, pending) = state.signal_store.stats();
        if recorded % 10 == 0 {
            debug!(
                total_recorded = recorded,
                total_exported = exported,
                in_memory = stored,
                pending_markouts = pending,
                "Signal diagnostic store stats"
            );
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

    /// Record a recently cancelled order as a tombstone.
    ///
    /// Called when an order is cancelled so that if a fill arrives for this OID
    /// within the TTL window (cancel-fill race condition), we can match and
    /// attribute it correctly instead of logging an "untracked order" warning.
    pub fn record_cancelled_order(&mut self, oid: u64, side: Side, price: f64, size: f64) {
        self.recently_cancelled.insert(
            oid,
            TombstoneOrder {
                side,
                price,
                size,
                cancelled_at: Instant::now(),
            },
        );
    }

    /// Remove tombstone entries older than the TTL (2 seconds).
    ///
    /// Should be called periodically (e.g. each quote cycle) to prevent unbounded growth.
    pub fn cleanup_tombstones(&mut self) {
        let now = Instant::now();
        self.recently_cancelled
            .retain(|_, tombstone| now.duration_since(tombstone.cancelled_at) < TOMBSTONE_TTL);
    }

    /// Number of active tombstone entries (for diagnostics).
    pub fn tombstone_count(&self) -> usize {
        self.recently_cancelled.len()
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
        pre_fill_classifier: &'a mut PreFillASClassifier,
        enhanced_classifier: &'a mut EnhancedASClassifier,
        estimator: &'a mut ParameterEstimator,
        pnl_tracker: &'a mut PnLTracker,
        prometheus: &'a mut PrometheusMetrics,
        metrics: &'a MetricsRecorder,
        learning: &'a mut crate::market_maker::learning::LearningModule,
        stochastic_controller: &'a mut StochasticController,
        position_pnl: &'a mut PositionPnLTracker,
        theoretical_edge: &'a mut crate::market_maker::control::TheoreticalEdgeEstimator,
        model_calibration: &'a mut ModelCalibrationOrchestrator,
        signal_store: &'a mut FillSignalStore,
    ) -> FillState<'a> {
        FillState {
            position,
            orders,
            adverse_selection,
            depth_decay_as,
            queue_tracker,
            pre_fill_classifier,
            enhanced_classifier,
            estimator,
            pnl_tracker,
            prometheus,
            metrics,
            latest_mid: 50000.0,
            asset: "BTC",
            max_position: 10.0,
            calibrate_depth_as: true,
            learning,
            stochastic_controller,
            position_pnl,
            fee_bps: 1.5,
            theoretical_edge,
            model_calibration,
            signal_store,
            market_params: None,
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
        let mut learning = crate::market_maker::learning::LearningModule::default();
        let mut stochastic_controller = StochasticController::default();
        let mut position_pnl = PositionPnLTracker::default();
        let mut theoretical_edge = crate::market_maker::control::TheoreticalEdgeEstimator::new();
        let mut model_calibration = ModelCalibrationOrchestrator::default();
        let mut signal_store = FillSignalStore::new();
        let mut pre_fill_classifier = PreFillASClassifier::default();
        let mut enhanced_classifier = EnhancedASClassifier::default_config();

        let mut state = make_test_state(
            &mut position,
            &mut orders,
            &mut adverse_selection,
            &mut depth_decay_as,
            &mut queue_tracker,
            &mut pre_fill_classifier,
            &mut enhanced_classifier,
            &mut estimator,
            &mut pnl_tracker,
            &mut prometheus,
            &metrics,
            &mut learning,
            &mut stochastic_controller,
            &mut position_pnl,
            &mut theoretical_edge,
            &mut model_calibration,
            &mut signal_store,
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
        let mut learning = crate::market_maker::learning::LearningModule::default();
        let mut stochastic_controller = StochasticController::default();
        let mut position_pnl = PositionPnLTracker::default();
        let mut theoretical_edge = crate::market_maker::control::TheoreticalEdgeEstimator::new();
        let mut model_calibration = ModelCalibrationOrchestrator::default();
        let mut signal_store = FillSignalStore::new();
        let mut pre_fill_classifier = PreFillASClassifier::default();
        let mut enhanced_classifier = EnhancedASClassifier::default_config();

        let mut state = make_test_state(
            &mut position,
            &mut orders,
            &mut adverse_selection,
            &mut depth_decay_as,
            &mut queue_tracker,
            &mut pre_fill_classifier,
            &mut enhanced_classifier,
            &mut estimator,
            &mut pnl_tracker,
            &mut prometheus,
            &metrics,
            &mut learning,
            &mut stochastic_controller,
            &mut position_pnl,
            &mut theoretical_edge,
            &mut model_calibration,
            &mut signal_store,
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

        // Register an immediate fill with amount
        processor.pre_register_immediate_fill(100, 1.0);
        assert!(processor.is_immediate_fill(100));
        assert_eq!(processor.immediate_fill_count(), 1);
        assert!((processor.get_immediate_fill_remaining(100) - 1.0).abs() < 1e-10);

        // Register another
        processor.pre_register_immediate_fill(200, 0.5);
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
        let mut learning = crate::market_maker::learning::LearningModule::default();
        let mut stochastic_controller = StochasticController::default();
        let mut position_pnl = PositionPnLTracker::default();
        let mut theoretical_edge = crate::market_maker::control::TheoreticalEdgeEstimator::new();
        let mut model_calibration = ModelCalibrationOrchestrator::default();
        let mut signal_store = FillSignalStore::new();
        let mut pre_fill_classifier = PreFillASClassifier::default();
        let mut enhanced_classifier = EnhancedASClassifier::default_config();

        // Pre-register OID 100 as immediate fill with amount 1.0 (simulating API returned filled=true)
        processor.pre_register_immediate_fill(100, 1.0);

        let mut state = make_test_state(
            &mut position,
            &mut orders,
            &mut adverse_selection,
            &mut depth_decay_as,
            &mut queue_tracker,
            &mut pre_fill_classifier,
            &mut enhanced_classifier,
            &mut estimator,
            &mut pnl_tracker,
            &mut prometheus,
            &metrics,
            &mut learning,
            &mut stochastic_controller,
            &mut position_pnl,
            &mut theoretical_edge,
            &mut model_calibration,
            &mut signal_store,
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
        let mut learning = crate::market_maker::learning::LearningModule::default();
        let mut stochastic_controller = StochasticController::default();
        let mut position_pnl = PositionPnLTracker::default();
        let mut theoretical_edge = crate::market_maker::control::TheoreticalEdgeEstimator::new();
        let mut model_calibration = ModelCalibrationOrchestrator::default();
        let mut signal_store = FillSignalStore::new();
        let mut pre_fill_classifier = PreFillASClassifier::default();
        let mut enhanced_classifier = EnhancedASClassifier::default_config();

        // No pre-registration - this is a normal fill

        let mut state = make_test_state(
            &mut position,
            &mut orders,
            &mut adverse_selection,
            &mut depth_decay_as,
            &mut queue_tracker,
            &mut pre_fill_classifier,
            &mut enhanced_classifier,
            &mut estimator,
            &mut pnl_tracker,
            &mut prometheus,
            &metrics,
            &mut learning,
            &mut stochastic_controller,
            &mut position_pnl,
            &mut theoretical_edge,
            &mut model_calibration,
            &mut signal_store,
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
        // Scenario: Order partially fills immediately (0.5), then more fills arrive via WS
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
        let mut learning = crate::market_maker::learning::LearningModule::default();
        let mut stochastic_controller = StochasticController::default();
        let mut position_pnl = PositionPnLTracker::default();
        let mut theoretical_edge = crate::market_maker::control::TheoreticalEdgeEstimator::new();
        let mut model_calibration = ModelCalibrationOrchestrator::default();
        let mut signal_store = FillSignalStore::new();
        let mut pre_fill_classifier = PreFillASClassifier::default();
        let mut enhanced_classifier = EnhancedASClassifier::default_config();

        // Pre-register OID 100 with the AMOUNT that was filled immediately (0.5)
        processor.pre_register_immediate_fill(100, 0.5);

        // First WS fill arrives (confirmation of immediate fill)
        let fill1 = make_fill(1, 100, 0.5, 50000.0, true);
        let result1 = {
            let mut state = make_test_state(
                &mut position,
                &mut orders,
                &mut adverse_selection,
                &mut depth_decay_as,
                &mut queue_tracker,
                &mut pre_fill_classifier,
                &mut enhanced_classifier,
                &mut estimator,
                &mut pnl_tracker,
                &mut prometheus,
                &metrics,
                &mut learning,
                &mut stochastic_controller,
                &mut position_pnl,
                &mut theoretical_edge,
                &mut model_calibration,
                &mut signal_store,
            );
            processor.process(&fill1, &mut state)
        };

        assert!(result1.is_immediate_fill_confirmation);
        // Position stays at 0.5 (no double-count)
        assert!((position.position() - 0.5).abs() < f64::EPSILON);

        // Second WS fill arrives (resting portion filled, not pre-registered)
        let fill2 = make_fill(2, 100, 0.3, 50000.0, true);
        let result2 = {
            let mut state = make_test_state(
                &mut position,
                &mut orders,
                &mut adverse_selection,
                &mut depth_decay_as,
                &mut queue_tracker,
                &mut pre_fill_classifier,
                &mut enhanced_classifier,
                &mut estimator,
                &mut pnl_tracker,
                &mut prometheus,
                &metrics,
                &mut learning,
                &mut stochastic_controller,
                &mut position_pnl,
                &mut theoretical_edge,
                &mut model_calibration,
                &mut signal_store,
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

    #[test]
    fn test_partial_immediate_fill_multiple_ws_fills() {
        // Scenario: Order for 1.0 partially fills immediately (0.6), WS sends two fills
        // Fill 1: 0.4 (part of immediate)
        // Fill 2: 0.3 (remaining immediate 0.2 + resting 0.1)
        // Fill 3: 0.3 (all resting)
        let mut processor = FillProcessor::new();
        let mut position = PositionTracker::new(0.6); // Start with 0.6 from API immediate fill
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
        let mut position_pnl = PositionPnLTracker::default();
        let mut theoretical_edge = crate::market_maker::control::TheoreticalEdgeEstimator::new();
        let mut model_calibration = ModelCalibrationOrchestrator::default();
        let mut signal_store = FillSignalStore::new();
        let mut pre_fill_classifier = PreFillASClassifier::default();
        let mut enhanced_classifier = EnhancedASClassifier::default_config();

        // Pre-register OID 100 with the AMOUNT that was filled immediately (0.6)
        processor.pre_register_immediate_fill(100, 0.6);
        assert!((processor.get_immediate_fill_remaining(100) - 0.6).abs() < 1e-10);

        // First WS fill: 0.4 (all from immediate portion)
        let fill1 = make_fill(1, 100, 0.4, 50000.0, true);
        let result1 = {
            let mut state = make_test_state(
                &mut position,
                &mut orders,
                &mut adverse_selection,
                &mut depth_decay_as,
                &mut queue_tracker,
                &mut pre_fill_classifier,
                &mut enhanced_classifier,
                &mut estimator,
                &mut pnl_tracker,
                &mut prometheus,
                &metrics,
                &mut learning,
                &mut stochastic_controller,
                &mut position_pnl,
                &mut theoretical_edge,
                &mut model_calibration,
                &mut signal_store,
            );
            processor.process(&fill1, &mut state)
        };

        assert!(result1.is_immediate_fill_confirmation);
        // Position stays at 0.6 (skipped full 0.4)
        assert!(
            (position.position() - 0.6).abs() < f64::EPSILON,
            "Position was {}, expected 0.6",
            position.position()
        );
        // Remaining immediate: 0.6 - 0.4 = 0.2
        assert!((processor.get_immediate_fill_remaining(100) - 0.2).abs() < 1e-10);

        // Second WS fill: 0.3 (0.2 from remaining immediate + 0.1 from resting)
        let fill2 = make_fill(2, 100, 0.3, 50000.0, true);
        let result2 = {
            let mut state = make_test_state(
                &mut position,
                &mut orders,
                &mut adverse_selection,
                &mut depth_decay_as,
                &mut queue_tracker,
                &mut pre_fill_classifier,
                &mut enhanced_classifier,
                &mut estimator,
                &mut pnl_tracker,
                &mut prometheus,
                &metrics,
                &mut learning,
                &mut stochastic_controller,
                &mut position_pnl,
                &mut theoretical_edge,
                &mut model_calibration,
                &mut signal_store,
            );
            processor.process(&fill2, &mut state)
        };

        // This is still marked as immediate fill confirmation because we skipped part of it
        assert!(result2.is_immediate_fill_confirmation);
        // Position should be 0.6 + 0.1 = 0.7 (skipped 0.2, updated 0.1)
        assert!(
            (position.position() - 0.7).abs() < 1e-10,
            "Position was {}, expected 0.7",
            position.position()
        );
        // Remaining immediate should be 0 (and OID removed from map)
        assert!(!processor.is_immediate_fill(100));

        // Third WS fill: 0.3 (all from resting)
        let fill3 = make_fill(3, 100, 0.3, 50000.0, true);
        let result3 = {
            let mut state = make_test_state(
                &mut position,
                &mut orders,
                &mut adverse_selection,
                &mut depth_decay_as,
                &mut queue_tracker,
                &mut pre_fill_classifier,
                &mut enhanced_classifier,
                &mut estimator,
                &mut pnl_tracker,
                &mut prometheus,
                &metrics,
                &mut learning,
                &mut stochastic_controller,
                &mut position_pnl,
                &mut theoretical_edge,
                &mut model_calibration,
                &mut signal_store,
            );
            processor.process(&fill3, &mut state)
        };

        assert!(!result3.is_immediate_fill_confirmation);
        // Position should be 0.7 + 0.3 = 1.0
        assert!(
            (position.position() - 1.0).abs() < 1e-10,
            "Position was {}, expected 1.0",
            position.position()
        );
    }

    #[test]
    fn test_tombstone_records_order() {
        let mut processor = FillProcessor::new();
        assert_eq!(processor.tombstone_count(), 0);

        processor.record_cancelled_order(100, Side::Buy, 50000.0, 1.0);
        assert_eq!(processor.tombstone_count(), 1);

        processor.record_cancelled_order(200, Side::Sell, 51000.0, 0.5);
        assert_eq!(processor.tombstone_count(), 2);

        // Verify tombstone data is correct
        let tombstone = processor.recently_cancelled.get(&100).unwrap();
        assert_eq!(tombstone.side, Side::Buy);
        assert!((tombstone.price - 50000.0).abs() < f64::EPSILON);
        assert!((tombstone.size - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tombstone_expires_after_ttl() {
        let mut processor = FillProcessor::new();

        // Insert a tombstone with an artificially old timestamp
        processor.recently_cancelled.insert(
            100,
            TombstoneOrder {
                side: Side::Buy,
                price: 50000.0,
                size: 1.0,
                cancelled_at: Instant::now() - std::time::Duration::from_secs(3),
            },
        );
        assert_eq!(processor.tombstone_count(), 1);

        // Insert a fresh tombstone
        processor.record_cancelled_order(200, Side::Sell, 51000.0, 0.5);
        assert_eq!(processor.tombstone_count(), 2);

        // Cleanup should remove the expired one, keep the fresh one
        processor.cleanup_tombstones();
        assert_eq!(processor.tombstone_count(), 1);
        assert!(processor.recently_cancelled.contains_key(&200));
        assert!(!processor.recently_cancelled.contains_key(&100));
    }

    #[test]
    fn test_fill_matches_tombstone() {
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
        let mut learning = crate::market_maker::learning::LearningModule::default();
        let mut stochastic_controller = StochasticController::default();
        let mut position_pnl = PositionPnLTracker::default();
        let mut theoretical_edge = crate::market_maker::control::TheoreticalEdgeEstimator::new();
        let mut model_calibration = ModelCalibrationOrchestrator::default();
        let mut signal_store = FillSignalStore::new();
        let mut pre_fill_classifier = PreFillASClassifier::default();
        let mut enhanced_classifier = EnhancedASClassifier::default_config();

        // Record a tombstone for OID 100 (simulating a cancelled order)
        processor.record_cancelled_order(100, Side::Buy, 49950.0, 1.0);

        let mut state = make_test_state(
            &mut position,
            &mut orders,
            &mut adverse_selection,
            &mut depth_decay_as,
            &mut queue_tracker,
            &mut pre_fill_classifier,
            &mut enhanced_classifier,
            &mut estimator,
            &mut pnl_tracker,
            &mut prometheus,
            &metrics,
            &mut learning,
            &mut stochastic_controller,
            &mut position_pnl,
            &mut theoretical_edge,
            &mut model_calibration,
            &mut signal_store,
        );

        // Fill arrives for the cancelled order (cancel-fill race)
        let fill = make_fill(1, 100, 1.0, 50000.0, true);
        let result = processor.process(&fill, &mut state);

        // Fill should be processed and matched to tombstone
        assert!(result.fill_result.is_new);
        // Placement price should come from the tombstone
        assert_eq!(result.placement_price, Some(49950.0));
        // Tombstone should be consumed (removed)
        assert_eq!(processor.tombstone_count(), 0);
        // Position should be updated
        assert!((position.position() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fill_without_tombstone_still_warns() {
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
        let mut learning = crate::market_maker::learning::LearningModule::default();
        let mut stochastic_controller = StochasticController::default();
        let mut position_pnl = PositionPnLTracker::default();
        let mut theoretical_edge = crate::market_maker::control::TheoreticalEdgeEstimator::new();
        let mut model_calibration = ModelCalibrationOrchestrator::default();
        let mut signal_store = FillSignalStore::new();
        let mut pre_fill_classifier = PreFillASClassifier::default();
        let mut enhanced_classifier = EnhancedASClassifier::default_config();

        // No tombstone recorded — this is a truly untracked fill

        let mut state = make_test_state(
            &mut position,
            &mut orders,
            &mut adverse_selection,
            &mut depth_decay_as,
            &mut queue_tracker,
            &mut pre_fill_classifier,
            &mut enhanced_classifier,
            &mut estimator,
            &mut pnl_tracker,
            &mut prometheus,
            &metrics,
            &mut learning,
            &mut stochastic_controller,
            &mut position_pnl,
            &mut theoretical_edge,
            &mut model_calibration,
            &mut signal_store,
        );

        // Fill arrives for unknown OID (no tombstone, no tracked order)
        let fill = make_fill(1, 999, 1.0, 50000.0, true);
        let result = processor.process(&fill, &mut state);

        // Fill should still be processed (position updated)
        assert!(result.fill_result.is_new);
        assert!((position.position() - 1.0).abs() < f64::EPSILON);
        // But placement price should be None (untracked)
        assert_eq!(result.placement_price, None);
        // No tombstones should exist
        assert_eq!(processor.tombstone_count(), 0);
    }
}
