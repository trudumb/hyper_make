//! Queue position state types.

use std::time::Instant;

/// Queue position state for a single order.
#[derive(Debug, Clone)]
pub struct OrderQueuePosition {
    /// Order ID
    pub oid: u64,
    /// Price level
    pub price: f64,
    /// Our order size
    pub size: f64,
    /// Estimated depth ahead of us in queue
    pub depth_ahead: f64,
    /// Time when order was placed
    pub placed_at: Instant,
    /// Time of last queue update
    pub last_update: Instant,
    /// True if this is a bid (buy order)
    pub is_bid: bool,
}

impl OrderQueuePosition {
    /// Create a new queue position for an order.
    pub fn new(oid: u64, price: f64, size: f64, depth_ahead: f64, is_bid: bool) -> Self {
        let now = Instant::now();
        Self {
            oid,
            price,
            size,
            depth_ahead,
            placed_at: now,
            last_update: now,
            is_bid,
        }
    }

    /// Get time since order was placed (seconds).
    pub fn age_seconds(&self) -> f64 {
        self.placed_at.elapsed().as_secs_f64()
    }

    /// Get time since last update (seconds).
    pub fn time_since_update(&self) -> f64 {
        self.last_update.elapsed().as_secs_f64()
    }
}

/// Summary of a single queue position.
#[derive(Debug, Clone)]
pub struct QueuePositionSummary {
    pub oid: u64,
    pub price: f64,
    pub depth_ahead: f64,
    pub age_seconds: f64,
}

/// Summary of all queue positions.
#[derive(Debug, Clone)]
pub struct QueueSummary {
    pub total_orders: usize,
    pub bid_positions: Vec<QueuePositionSummary>,
    pub ask_positions: Vec<QueuePositionSummary>,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
}
