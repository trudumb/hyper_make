//! Order state types and structures.
//!
//! Contains core types for order lifecycle management:
//! - `Side`: Buy/Sell side enumeration
//! - `OrderState`: Order lifecycle state machine
//! - `TrackedOrder`: Order with full lifecycle metadata
//! - `PendingOrder`: Order awaiting OID assignment
//! - `LadderAction`: Reconciliation actions

use std::time::{Duration, Instant};

use smallvec::SmallVec;

use crate::market_maker::infra::capacity::FILL_TID_INLINE_CAPACITY;
use crate::EPSILON;
use serde::{Deserialize, Serialize};

/// Side of an order (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// Convert from Hyperliquid side string ("B" or "A"/"S").
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "B" => Some(Side::Buy),
            "A" | "S" => Some(Side::Sell),
            _ => None,
        }
    }
}

/// Order lifecycle state.
///
/// State transitions:
/// ```text
/// Resting ─────────────────────────────────────────► Filled
///    │                                                  ▲
///    │ (partial fill)                                   │
///    ▼                                                  │
/// PartialFilled ───────────────────────────────────────┘
///    │                                                  ▲
///    │ (cancel requested)                               │
///    ▼                                                  │
/// CancelPending ──────────────────────────────────────► FilledDuringCancel
///    │                                                  ▲
///    │ (cancel confirmed)                               │
///    ▼                                                  │
/// CancelConfirmed ─────────────────────────────────────┘
///    │
///    │ (fill window expired, no late fills)
///    ▼
/// Cancelled
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    /// Order is resting on the exchange, no fills yet
    Resting,
    /// Order has been partially filled but still has remaining size
    PartialFilled,
    /// Cancel request has been sent, awaiting confirmation.
    /// Fills can still arrive during this window!
    CancelPending,
    /// Cancel has been confirmed by exchange, waiting for fill window to expire.
    /// This is the critical state that prevents premature removal - late fills
    /// from WebSocket can still arrive during this window.
    CancelConfirmed,
    /// Fill arrived during cancel process (CancelPending or CancelConfirmed).
    /// Terminal state - safe to remove immediately.
    FilledDuringCancel,
    /// Fully filled. Terminal state - safe to remove immediately.
    Filled,
    /// Cancel confirmed and fill window has expired with no late fills.
    /// Terminal state - safe to remove immediately.
    Cancelled,
    /// Order filled immediately on placement (API reported filled before WS).
    ///
    /// This state bridges the race condition between REST API response and WebSocket
    /// fill notification. Position was already updated from API response; when WS
    /// fill arrives, it will be deduplicated.
    ///
    /// Transitions to Filled when WS confirmation arrives (or after timeout).
    FilledImmediately,
}

/// Action to take for ladder reconciliation.
#[derive(Debug, Clone)]
pub enum LadderAction {
    /// Place a new order at the specified price and size
    Place { side: Side, price: f64, size: f64 },
    /// Cancel an existing order by ID
    Cancel { oid: u64 },
    /// Modify an existing order (preserves queue position when possible)
    Modify {
        oid: u64,
        new_price: f64,
        new_size: f64,
        side: Side,
    },
}

/// A pending order awaiting OID from exchange.
///
/// Used to bridge the race condition between order placement and fill notification.
/// When an order is placed, we store it by CLOID (primary) and (side, price_key) (fallback)
/// so that if a fill arrives before the OID is known, we can still find the placement price.
///
/// # CLOID Tracking (Primary)
///
/// Client Order IDs (CLOIDs) are generated before placement and provide deterministic
/// order matching. The SDK returns CLOIDs in fill notifications via `TradeInfo.cloid`.
/// This eliminates the timing-dependent race condition between REST response and WebSocket fills.
///
/// # Price-Based Fallback
///
/// For backward compatibility and edge cases, we also store by (side, price_key).
/// This handles fills where CLOID might be missing from the exchange response.
#[derive(Debug, Clone)]
pub struct PendingOrder {
    /// Side of the order
    pub side: Side,
    /// Limit price
    pub price: f64,
    /// Intended order size
    pub size: f64,
    /// When the order was submitted
    pub placed_at: Instant,
    /// Client Order ID (UUID) - generated before placement for deterministic tracking
    pub cloid: Option<String>,
    /// Calibration prediction ID for fill probability model
    pub fill_prediction_id: Option<u64>,
    /// Calibration prediction ID for adverse selection model
    pub as_prediction_id: Option<u64>,
    /// Depth from mid in basis points (for calibration)
    pub depth_bps: Option<f64>,
}

impl PendingOrder {
    /// Create a new pending order (legacy, without CLOID).
    pub fn new(side: Side, price: f64, size: f64) -> Self {
        Self {
            side,
            price,
            size,
            placed_at: Instant::now(),
            cloid: None,
            fill_prediction_id: None,
            as_prediction_id: None,
            depth_bps: None,
        }
    }

    /// Create a new pending order with CLOID tracking.
    pub fn with_cloid(side: Side, price: f64, size: f64, cloid: String) -> Self {
        Self {
            side,
            price,
            size,
            placed_at: Instant::now(),
            cloid: Some(cloid),
            fill_prediction_id: None,
            as_prediction_id: None,
            depth_bps: None,
        }
    }

    /// Create a new pending order with full calibration tracking.
    pub fn with_calibration(
        side: Side,
        price: f64,
        size: f64,
        cloid: String,
        fill_prediction_id: Option<u64>,
        as_prediction_id: Option<u64>,
        depth_bps: Option<f64>,
    ) -> Self {
        Self {
            side,
            price,
            size,
            placed_at: Instant::now(),
            cloid: Some(cloid),
            fill_prediction_id,
            as_prediction_id,
            depth_bps,
        }
    }
}

/// Convert a price to an integer key for HashMap lookup.
/// Uses fixed-point representation with 8 decimal places.
#[inline]
pub(crate) fn price_to_key(price: f64) -> u64 {
    (price * 1e8).round() as u64
}

/// A tracked order with its current state and lifecycle metadata.
#[derive(Debug, Clone)]
pub struct TrackedOrder {
    /// Order ID from the exchange
    pub oid: u64,
    /// Client order ID (optional, for CLOID tracking)
    pub cloid: Option<String>,
    /// Side of the order
    pub side: Side,
    /// Limit price
    pub price: f64,
    /// Original size
    pub size: f64,
    /// Amount filled so far
    pub filled: f64,
    /// Total fill value (for average price calculation)
    /// Sum of (fill_price × fill_size) for all fills
    fill_value: f64,
    /// Order lifecycle state
    pub state: OrderState,
    /// When the order was placed (for age tracking)
    pub placed_at: Instant,
    /// When the order entered its current state (for fill window timing)
    pub state_changed_at: Instant,
    /// When the last fill occurred (for activity tracking)
    pub last_fill_at: Option<Instant>,
    /// Trade IDs of fills processed for this order (for dedup at order level).
    /// Uses SmallVec to avoid heap allocation for typical 1-4 fills per order.
    pub fill_tids: SmallVec<[u64; FILL_TID_INLINE_CAPACITY]>,
    /// Calibration prediction ID for fill probability model
    pub fill_prediction_id: Option<u64>,
    /// Calibration prediction ID for adverse selection model
    pub as_prediction_id: Option<u64>,
    /// Depth from mid in basis points (for calibration outcome recording)
    pub depth_bps: Option<f64>,
}

impl TrackedOrder {
    /// Create a new tracked order in Resting state.
    pub fn new(oid: u64, side: Side, price: f64, size: f64) -> Self {
        let now = Instant::now();
        Self {
            oid,
            cloid: None,
            side,
            price,
            size,
            filled: 0.0,
            fill_value: 0.0,
            state: OrderState::Resting,
            placed_at: now,
            state_changed_at: now,
            last_fill_at: None,
            fill_tids: SmallVec::new(),
            fill_prediction_id: None,
            as_prediction_id: None,
            depth_bps: None,
        }
    }

    /// Create a new tracked order with a client order ID.
    pub fn with_cloid(oid: u64, cloid: String, side: Side, price: f64, size: f64) -> Self {
        let mut order = Self::new(oid, side, price, size);
        order.cloid = Some(cloid);
        order
    }

    /// Create a tracked order from a pending order when OID is assigned.
    /// Preserves calibration prediction IDs for outcome recording.
    pub fn from_pending(oid: u64, pending: &PendingOrder) -> Self {
        let now = Instant::now();
        Self {
            oid,
            cloid: pending.cloid.clone(),
            side: pending.side,
            price: pending.price,
            size: pending.size,
            filled: 0.0,
            fill_value: 0.0,
            state: OrderState::Resting,
            placed_at: pending.placed_at,
            state_changed_at: now,
            last_fill_at: None,
            fill_tids: SmallVec::new(),
            fill_prediction_id: pending.fill_prediction_id,
            as_prediction_id: pending.as_prediction_id,
            depth_bps: pending.depth_bps,
        }
    }

    /// Get remaining size (unfilled).
    pub fn remaining(&self) -> f64 {
        (self.size - self.filled).max(0.0)
    }

    /// Check if the order is fully filled.
    pub fn is_filled(&self) -> bool {
        self.remaining() <= EPSILON
    }

    /// Transition to a new state, recording the timestamp.
    pub fn transition_to(&mut self, new_state: OrderState) {
        self.state = new_state;
        self.state_changed_at = Instant::now();
    }

    /// Record a fill on this order (without price for backward compatibility).
    /// Returns true if this is a new fill, false if duplicate (already processed).
    pub fn record_fill(&mut self, tid: u64, amount: f64) -> bool {
        // Use placement price as fill price (conservative estimate)
        self.record_fill_with_price(tid, amount, self.price)
    }

    /// Record a fill on this order with the actual fill price.
    /// Returns true if this is a new fill, false if duplicate (already processed).
    pub fn record_fill_with_price(&mut self, tid: u64, amount: f64, fill_price: f64) -> bool {
        if self.fill_tids.contains(&tid) {
            return false; // Duplicate fill
        }
        self.fill_tids.push(tid);
        self.filled += amount;
        self.fill_value += fill_price * amount;
        self.last_fill_at = Some(Instant::now());
        true
    }

    /// Get the average fill price for this order.
    /// Returns the placement price if no fills yet.
    pub fn average_fill_price(&self) -> f64 {
        if self.filled > EPSILON {
            self.fill_value / self.filled
        } else {
            self.price // No fills yet, return placement price
        }
    }

    /// Get time since last fill (if any).
    pub fn time_since_last_fill(&self) -> Option<Duration> {
        self.last_fill_at.map(|t| t.elapsed())
    }

    /// Get order age (time since placement).
    pub fn age(&self) -> Duration {
        self.placed_at.elapsed()
    }

    /// Get fill count.
    pub fn fill_count(&self) -> usize {
        self.fill_tids.len()
    }

    /// Check if the fill window has expired (for CancelConfirmed state).
    pub fn fill_window_expired(&self, window: Duration) -> bool {
        if self.state == OrderState::CancelConfirmed {
            self.state_changed_at.elapsed() >= window
        } else {
            false
        }
    }

    /// Check if order is in a terminal state (ready for cleanup).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            OrderState::Filled
                | OrderState::Cancelled
                | OrderState::FilledDuringCancel
                | OrderState::FilledImmediately
        )
    }

    /// Check if order can receive fills (not yet fully cancelled).
    pub fn can_receive_fills(&self) -> bool {
        matches!(
            self.state,
            OrderState::Resting
                | OrderState::PartialFilled
                | OrderState::CancelPending
                | OrderState::CancelConfirmed
        )
    }

    /// Check if order is actively quoting (not in cancel process).
    pub fn is_active(&self) -> bool {
        matches!(self.state, OrderState::Resting | OrderState::PartialFilled)
    }
}

/// Configuration for order state management.
#[derive(Debug, Clone)]
pub struct OrderManagerConfig {
    /// Time to wait after cancel confirmation for potential late fills.
    /// Hyperliquid typically processes within 2 seconds, so 5s is safe.
    pub fill_window_duration: Duration,
    /// Maximum time an order can be in CancelPending before considered stuck.
    pub cancel_timeout: Duration,
}

impl Default for OrderManagerConfig {
    fn default() -> Self {
        Self {
            fill_window_duration: Duration::from_secs(5),
            cancel_timeout: Duration::from_secs(30),
        }
    }
}
