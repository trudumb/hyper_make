//! Execution quality tracking module.
//!
//! This module provides tools for tracking and analyzing order execution quality
//! in a market making system.
//!
//! # Components
//!
//! - **FillTracker**: Tracks fill quality metrics (fill rate, adverse selection, latency)
//! - **OrderLifecycleTracker**: Tracks order lifecycle from placement to terminal state
//!
//! # Usage
//!
//! ```ignore
//! use hyper_make::market_maker::execution::{
//!     FillTracker, FillRecord, FillMetrics, Side,
//!     OrderLifecycleTracker, OrderEvent, OrderState, OrderLifecycle,
//! };
//!
//! // Fill tracking
//! let mut fill_tracker = FillTracker::new(1000);
//! fill_tracker.record_quote();
//! fill_tracker.record_fill(FillRecord::new(
//!     current_time_ms(),
//!     Side::Bid,
//!     50000.0,
//!     0.01,
//!     5,
//!     15.0,
//! ));
//! let metrics = fill_tracker.metrics();
//!
//! // Order lifecycle tracking
//! let lifecycle_tracker = OrderLifecycleTracker::new(1000);
//! lifecycle_tracker.create_order(CreateOrderParams {
//!     order_id: 12345,
//!     client_order_id: "cloid-uuid".to_string(),
//!     symbol: "BTC".to_string(),
//!     side: Side::Bid,
//!     price: 50000.0,
//!     size: 0.01,
//!     timestamp_ms: current_time_ms(),
//! });
//! lifecycle_tracker.update_order(12345, OrderEvent::new(
//!     current_time_ms() + 100,
//!     OrderState::Filled,
//!     0.01,
//!     0.0,
//! ));
//! let analysis = lifecycle_tracker.cancel_analysis();
//! ```
//!
//! # Thread Safety
//!
//! `OrderLifecycleTracker` is thread-safe (Send + Sync) and uses internal
//! RwLock for concurrent access. `FillTracker` is not thread-safe and should
//! be protected externally if used from multiple threads.
//!
//! # Metrics
//!
//! ## Fill Metrics
//!
//! - **fill_rate**: Fills / quotes placed (0.0 to 1.0)
//! - **adverse_selection_rate**: % of fills where price moved against within 1s
//! - **queue_position_mean**: Average queue position at fill time
//! - **latency_p50_ms / latency_p99_ms**: Latency percentiles
//!
//! ## Cancel Analysis
//!
//! - **total_cancels**: Total cancelled orders
//! - **cancel_before_any_fill**: Orders cancelled before receiving any fill
//! - **cancel_after_partial**: Orders cancelled after partial fill
//! - **avg_time_to_cancel_ms**: Average time from placement to cancel

mod fill_tracker;
mod order_lifecycle;

// Re-export key types
pub use fill_tracker::{FillMetrics, FillRecord, FillTracker, Side};
pub use order_lifecycle::{
    CancelAnalysis, CreateOrderParams, FillStatistics, OrderEvent, OrderLifecycle,
    OrderLifecycleTracker, OrderState,
};
