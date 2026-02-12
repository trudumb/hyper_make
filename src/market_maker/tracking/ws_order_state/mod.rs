//! WebSocket-based order state management.
//!
//! This module provides end-to-end order lifecycle management using WebSocket
//! for both order submission and real-time state updates. It is designed to
//! replace REST-based order placement with lower-latency WS post requests.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                   WsOrderStateManager                        │
//! │  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
//! │  │ InflightReqs   │  │ TrackedOrders  │  │ProcessedTIDs  │  │
//! │  │ (request_id→)  │  │    (oid→)      │  │ (dedup set)   │  │
//! │  └────────────────┘  └────────────────┘  └───────────────┘  │
//! └─────────────────────────────┬────────────────────────────────┘
//!                               │
//!          ┌────────────────────┼────────────────────┐
//!          │                    │                    │
//!          ▼                    ▼                    ▼
//!  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
//!  │  WS Post API  │    │ orderUpdates  │    │  userFills    │
//!  │ (submit order)│    │ (state sync)  │    │ (fill events) │
//!  └───────────────┘    └───────────────┘    └───────────────┘
//! ```
//!
//! # Key Features
//!
//! - **WS Post Submission**: Orders submitted via WebSocket post requests for
//!   lower latency than REST (typically 5-20ms vs 50-100ms)
//!
//! - **Request ID Correlation**: Each WS post request has a unique ID for
//!   matching responses to requests
//!
//! - **CLOID Correlation**: Client Order IDs link orders across events
//!   (placement, updates, fills)
//!
//! - **Trade ID Deduplication**: Fills are deduplicated by trade ID to prevent
//!   double-counting
//!
//! - **Timeout Recovery**: Timed out requests can be recovered via REST fallback
//!
//! - **Reconciliation**: Periodic sync with exchange state catches any drift
//!
//! # Usage
//!
//! ```rust,ignore
//! use hyperliquid_rust_sdk::market_maker::tracking::ws_order_state::*;
//!
//! // Create manager
//! let mut mgr = WsOrderStateManager::new();
//!
//! // Prepare order for WS submission
//! let spec = WsOrderSpec::new("BTC".to_string(), 50000.0, 0.01, true);
//! let (request_id, inflight) = mgr.prepare_order(spec)?;
//!
//! // Send via WebSocket (caller responsibility)
//! // ws.send(build_ws_post_message(request_id, &inflight))?;
//!
//! // Handle response
//! mgr.handle_ws_response(&response);
//!
//! // Handle orderUpdates
//! mgr.handle_order_update(&update);
//!
//! // Handle fills
//! mgr.handle_fill(&fill, &mut position_tracker);
//! ```
//!
//! # Module Structure
//!
//! - `types`: Core types (InflightRequest, WsPostRequest, etc.)
//! - `manager`: WsOrderStateManager implementation
//! - `reconcile`: Reconciliation with exchange state

mod manager;
mod reconcile;
mod types;

pub use manager::WsOrderStateManager;
pub use reconcile::ExchangeOrderInfo;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::tracking::order_manager::Side;
    use crate::market_maker::tracking::PositionTracker;

    #[test]
    fn test_full_order_lifecycle() {
        let mut mgr = WsOrderStateManager::new();
        let mut position = PositionTracker::new(0.0);

        // 1. Prepare order
        let spec = WsOrderSpec::new("BTC".to_string(), 50000.0, 0.01, true);
        let (request_id, inflight) = mgr.prepare_order(spec).unwrap();
        assert_eq!(request_id, 1);
        assert_eq!(mgr.inflight_count(), 1);

        // 2. Simulate WS response (order placed successfully)
        // In real code, this would come from parsing the WS message
        let oid = 12345u64;
        let cloid = inflight.cloid.clone();

        // Manually add the order as if response was processed
        let order = crate::market_maker::tracking::TrackedOrder::with_cloid(
            oid,
            cloid.clone(),
            Side::Buy,
            50000.0,
            0.01,
            0.0,
        );
        mgr.add_order(order);

        // 3. Verify order is tracked
        assert!(mgr.get_order(oid).is_some());
        let tracked = mgr.get_order(oid).unwrap();
        assert_eq!(tracked.side, Side::Buy);
        assert!((tracked.price - 50000.0).abs() < 1e-10);

        // 4. Simulate orderUpdate (open)
        let update = WsOrderUpdateEvent {
            oid,
            cloid: Some(cloid.clone()),
            status: "open".to_string(),
            size: 0.01,
            orig_size: 0.01,
            price: 50000.0,
            coin: "BTC".to_string(),
            is_buy: true,
            status_timestamp: 0,
        };
        mgr.handle_order_update(&update);
        assert_eq!(
            mgr.get_order(oid).unwrap().state,
            crate::market_maker::tracking::OrderState::Resting
        );

        // 5. Simulate fill
        let fill = WsFillEvent {
            oid,
            tid: 99999,
            size: 0.01,
            price: 50000.0,
            is_buy: true,
            coin: "BTC".to_string(),
            cloid: Some(cloid.clone()),
            timestamp: 0,
        };
        let fill_result = mgr.handle_fill(&fill, &mut position);
        assert_eq!(fill_result, Some(0.01));

        // 6. Simulate orderUpdate (filled)
        let update = WsOrderUpdateEvent {
            oid,
            cloid: Some(cloid),
            status: "filled".to_string(),
            size: 0.0,
            orig_size: 0.01,
            price: 50000.0,
            coin: "BTC".to_string(),
            is_buy: true,
            status_timestamp: 0,
        };
        mgr.handle_order_update(&update);
        assert_eq!(
            mgr.get_order(oid).unwrap().state,
            crate::market_maker::tracking::OrderState::Filled
        );

        // 7. Position should be updated
        assert!((position.position() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_cancel_lifecycle() {
        let mut mgr = WsOrderStateManager::new();

        // Add an order
        let order = crate::market_maker::tracking::TrackedOrder::new(123, Side::Buy, 50000.0, 0.01, 0.0);
        mgr.add_order(order);

        // Prepare cancel
        let request_id = mgr.prepare_cancel("BTC", 123).unwrap();
        assert!(request_id > 0);

        // Order should be in CancelPending
        assert_eq!(
            mgr.get_order(123).unwrap().state,
            crate::market_maker::tracking::OrderState::CancelPending
        );

        // Simulate orderUpdate (canceled)
        let update = WsOrderUpdateEvent {
            oid: 123,
            cloid: None,
            status: "canceled".to_string(),
            size: 0.0,
            orig_size: 0.01,
            price: 50000.0,
            coin: "BTC".to_string(),
            is_buy: true,
            status_timestamp: 0,
        };
        mgr.handle_order_update(&update);

        // Order should be Cancelled
        assert_eq!(
            mgr.get_order(123).unwrap().state,
            crate::market_maker::tracking::OrderState::Cancelled
        );
    }
}
