//! Order state types and structures.
//!
//! Contains core types for order lifecycle management:
//! - `Side`: Buy/Sell side enumeration
//! - `OrderState`: Order lifecycle state machine
//! - `TrackedOrder`: Order with full lifecycle metadata
//! - `PendingOrder`: Order awaiting OID assignment
//! - `LadderAction`: Reconciliation actions

use std::time::{Duration, Instant};

use crate::EPSILON;

/// Side of an order (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
}

/// Action to take for ladder reconciliation.
#[derive(Debug, Clone)]
pub enum LadderAction {
    /// Place a new order at the specified price and size
    Place { side: Side, price: f64, size: f64 },
    /// Cancel an existing order by ID
    Cancel { oid: u64 },
}

/// A pending order awaiting OID from exchange.
///
/// Used to bridge the race condition between order placement and fill notification.
/// When an order is placed, we store it by (side, price_key) so that if a fill
/// arrives before the OID is known, we can still find the placement price.
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
}

impl PendingOrder {
    /// Create a new pending order.
    pub fn new(side: Side, price: f64, size: f64) -> Self {
        Self {
            side,
            price,
            size,
            placed_at: Instant::now(),
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
    /// Side of the order
    pub side: Side,
    /// Limit price
    pub price: f64,
    /// Original size
    pub size: f64,
    /// Amount filled so far
    pub filled: f64,
    /// Order lifecycle state
    pub state: OrderState,
    /// When the order was placed (for age tracking)
    pub placed_at: Instant,
    /// When the order entered its current state (for fill window timing)
    pub state_changed_at: Instant,
    /// Trade IDs of fills processed for this order (for dedup at order level)
    pub fill_tids: Vec<u64>,
}

impl TrackedOrder {
    /// Create a new tracked order in Resting state.
    pub fn new(oid: u64, side: Side, price: f64, size: f64) -> Self {
        let now = Instant::now();
        Self {
            oid,
            side,
            price,
            size,
            filled: 0.0,
            state: OrderState::Resting,
            placed_at: now,
            state_changed_at: now,
            fill_tids: Vec::new(),
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

    /// Record a fill on this order.
    /// Returns true if this is a new fill, false if duplicate (already processed).
    pub fn record_fill(&mut self, tid: u64, amount: f64) -> bool {
        if self.fill_tids.contains(&tid) {
            return false; // Duplicate fill
        }
        self.fill_tids.push(tid);
        self.filled += amount;
        true
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
            OrderState::Filled | OrderState::Cancelled | OrderState::FilledDuringCancel
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
