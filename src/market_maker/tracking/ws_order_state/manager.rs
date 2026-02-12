//! WebSocket-based order state manager.
//!
//! This module provides end-to-end order state management using WebSocket
//! for both order submission and state updates. It replaces REST-based order
//! placement with lower-latency WS post requests while receiving real-time
//! state updates via orderUpdates and userFills subscriptions.

use super::types::*;
use crate::market_maker::tracking::order_manager::{OrderState, Side, TrackedOrder};
use crate::market_maker::tracking::PositionTracker;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// WebSocket-based order state manager.
///
/// Manages the complete order lifecycle using WebSocket for:
/// - Order submission via WS post requests
/// - State tracking via orderUpdates subscription
/// - Fill processing via userFills subscription
///
/// Key features:
/// - Request ID correlation for WS post responses
/// - CLOID correlation for cross-event matching
/// - Trade ID deduplication for fills
/// - Timeout handling with REST fallback
pub struct WsOrderStateManager {
    /// Inflight WS post requests awaiting response (request_id → request)
    inflight: HashMap<u64, InflightRequest>,

    /// CLOID → request_id mapping for inflight correlation
    cloid_to_request: HashMap<String, u64>,

    /// Tracked orders after OID assignment (oid → order)
    orders: HashMap<u64, TrackedOrder>,

    /// CLOID → OID mapping for event correlation
    cloid_to_oid: HashMap<String, u64>,

    /// Fill deduplication by trade ID
    processed_tids: HashSet<u64>,

    /// TID → timestamp for retention cleanup
    tid_timestamps: HashMap<u64, Instant>,

    /// Request ID generator
    request_id_gen: RequestIdGenerator,

    /// Configuration
    config: WsOrderStateConfig,

    /// Last cleanup time
    last_cleanup: Instant,
}

impl WsOrderStateManager {
    /// Create a new WS order state manager with default config.
    pub fn new() -> Self {
        Self::with_config(WsOrderStateConfig::default())
    }

    /// Create a new WS order state manager with custom config.
    pub fn with_config(config: WsOrderStateConfig) -> Self {
        Self {
            inflight: HashMap::new(),
            cloid_to_request: HashMap::new(),
            orders: HashMap::new(),
            cloid_to_oid: HashMap::new(),
            processed_tids: HashSet::new(),
            tid_timestamps: HashMap::new(),
            request_id_gen: RequestIdGenerator::new(),
            config,
            last_cleanup: Instant::now(),
        }
    }

    // =========================================================================
    // Order Submission
    // =========================================================================

    /// Prepare a single order for WS post submission.
    ///
    /// Returns the request ID and serialized message to send.
    /// The caller should send this via WebSocket and await the response.
    pub fn prepare_order(&mut self, spec: WsOrderSpec) -> Result<(u64, InflightRequest), String> {
        // Check inflight limit
        if self.inflight.len() >= self.config.max_inflight_requests {
            return Err("Max inflight requests reached".to_string());
        }

        let request_id = self.request_id_gen.next();
        let cloid = spec
            .cloid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let inflight = InflightRequest::new(
            request_id,
            cloid.clone(),
            spec.asset.clone(),
            spec.price,
            spec.size,
            RequestType::Place {
                is_buy: spec.is_buy,
            },
        );

        // Store in inflight maps
        self.inflight.insert(request_id, inflight);
        self.cloid_to_request.insert(cloid.clone(), request_id);

        // Return a copy for the caller
        let inflight_copy = InflightRequest::new(
            request_id,
            cloid,
            spec.asset,
            spec.price,
            spec.size,
            RequestType::Place {
                is_buy: spec.is_buy,
            },
        );

        Ok((request_id, inflight_copy))
    }

    /// Prepare bulk orders for WS post submission.
    ///
    /// Returns the request ID and list of inflight requests.
    pub fn prepare_bulk_orders(
        &mut self,
        specs: Vec<WsOrderSpec>,
    ) -> Result<(u64, Vec<InflightRequest>), String> {
        if specs.is_empty() {
            return Err("Empty order specs".to_string());
        }

        if self.inflight.len() + specs.len() > self.config.max_inflight_requests {
            return Err("Would exceed max inflight requests".to_string());
        }

        let request_id = self.request_id_gen.next();
        let is_buy = specs[0].is_buy;
        let mut inflight_copies = Vec::with_capacity(specs.len());

        for spec in specs {
            let cloid = spec
                .cloid
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

            let inflight = InflightRequest::new(
                request_id,
                cloid.clone(),
                spec.asset.clone(),
                spec.price,
                spec.size,
                RequestType::BulkPlace {
                    count: 1,
                    is_buy: spec.is_buy,
                },
            );

            self.cloid_to_request.insert(cloid.clone(), request_id);
            inflight_copies.push(inflight);
        }

        // Store the first inflight as representative for the bulk request
        if let Some(first) = inflight_copies.first() {
            let bulk_inflight = InflightRequest::new(
                request_id,
                first.cloid.clone(),
                first.asset.clone(),
                first.price,
                first.size,
                RequestType::BulkPlace {
                    count: inflight_copies.len(),
                    is_buy,
                },
            );
            self.inflight.insert(request_id, bulk_inflight);
        }

        Ok((request_id, inflight_copies))
    }

    /// Prepare a cancel request for WS post submission.
    pub fn prepare_cancel(&mut self, asset: &str, oid: u64) -> Result<u64, String> {
        if self.inflight.len() >= self.config.max_inflight_requests {
            return Err("Max inflight requests reached".to_string());
        }

        let request_id = self.request_id_gen.next();
        let cloid = uuid::Uuid::new_v4().to_string();

        let inflight = InflightRequest::new(
            request_id,
            cloid.clone(),
            asset.to_string(),
            0.0,
            0.0,
            RequestType::Cancel { oid },
        );

        self.inflight.insert(request_id, inflight);

        // Mark order as cancel pending
        if let Some(order) = self.orders.get_mut(&oid) {
            order.transition_to(OrderState::CancelPending);
        }

        Ok(request_id)
    }

    // =========================================================================
    // Response Handling
    // =========================================================================

    /// Handle a WS post response.
    ///
    /// This processes the response and updates internal state accordingly.
    pub fn handle_ws_response(&mut self, response: &WsPostResponse) -> Vec<ActionResult> {
        let request_id = response.data.id;

        let Some(inflight) = self.inflight.remove(&request_id) else {
            warn!(
                request_id = request_id,
                "WS post response for unknown request"
            );
            return vec![ActionResult::empty()];
        };

        // Clean up CLOID mapping
        self.cloid_to_request.remove(&inflight.cloid);

        match &response.data.response {
            WsResponsePayload::Action { payload } => {
                self.process_action_response(inflight, payload)
            }
            WsResponsePayload::Info { payload: _ } => {
                debug!(request_id = request_id, "Info response received");
                vec![ActionResult::empty()]
            }
            WsResponsePayload::Error { payload } => {
                error!(
                    request_id = request_id,
                    cloid = %inflight.cloid,
                    error = %payload,
                    "WS post request failed"
                );
                vec![ActionResult::error(payload.clone())]
            }
        }
    }

    fn process_action_response(
        &mut self,
        inflight: InflightRequest,
        payload: &serde_json::Value,
    ) -> Vec<ActionResult> {
        // Extract values before consuming inflight to avoid borrow conflicts
        let request_type = inflight.request_type.clone();

        // Parse the response based on request type
        match request_type {
            RequestType::Place { is_buy } => self.process_place_response(inflight, payload, is_buy),
            RequestType::BulkPlace { count, is_buy } => {
                self.process_bulk_place_response(inflight, payload, count, is_buy)
            }
            RequestType::Cancel { oid } => self.process_cancel_response(oid, payload),
            RequestType::Modify { original_oid } => {
                self.process_modify_response(inflight, payload, original_oid)
            }
            RequestType::BulkCancel { oids } => self.process_bulk_cancel_response(oids, payload),
        }
    }

    fn process_place_response(
        &mut self,
        inflight: InflightRequest,
        payload: &serde_json::Value,
        is_buy: bool,
    ) -> Vec<ActionResult> {
        // Try to extract OID from response
        // Expected format: {"status":"ok","response":{"type":"order","data":{"statuses":[{"resting":{"oid":123}}]}}}
        let oid = self.extract_oid_from_response(payload);
        let filled = self.extract_filled_from_response(payload);
        let resting_size = self.extract_resting_size_from_response(payload, inflight.size);

        if let Some(oid) = oid {
            let side = if is_buy { Side::Buy } else { Side::Sell };

            // Create tracked order
            let mut tracked = TrackedOrder::with_cloid(
                oid,
                inflight.cloid.clone(),
                side,
                inflight.price,
                inflight.size,
                0.0, // mid not available in WS state manager
            );

            if filled {
                tracked.transition_to(OrderState::FilledImmediately);
                tracked.filled = inflight.size;
            }

            self.orders.insert(oid, tracked);
            self.cloid_to_oid.insert(inflight.cloid.clone(), oid);

            info!(
                oid = oid,
                cloid = %inflight.cloid,
                filled = filled,
                "Order placed via WS post"
            );

            vec![ActionResult::single(
                oid,
                Some(inflight.cloid),
                filled,
                resting_size,
            )]
        } else {
            let error = self.extract_error_from_response(payload);
            error!(
                cloid = %inflight.cloid,
                error = ?error,
                "Order placement failed"
            );
            vec![ActionResult::error(
                error.unwrap_or_else(|| "Unknown error".to_string()),
            )]
        }
    }

    fn process_bulk_place_response(
        &mut self,
        _inflight: InflightRequest,
        payload: &serde_json::Value,
        count: usize,
        is_buy: bool,
    ) -> Vec<ActionResult> {
        // Extract statuses array from bulk response
        let statuses = payload
            .get("response")
            .and_then(|r| r.get("data"))
            .and_then(|d| d.get("statuses"))
            .and_then(|s| s.as_array());

        let mut results = Vec::with_capacity(count);

        if let Some(statuses) = statuses {
            for status in statuses {
                let oid = status
                    .get("resting")
                    .and_then(|r| r.get("oid"))
                    .and_then(|o| o.as_u64())
                    .or_else(|| {
                        status
                            .get("filled")
                            .and_then(|f| f.get("oid"))
                            .and_then(|o| o.as_u64())
                    });

                let filled = status.get("filled").is_some();

                if let Some(oid) = oid {
                    let cloid = uuid::Uuid::new_v4().to_string(); // Bulk orders may not have individual CLOIDs
                    let side = if is_buy { Side::Buy } else { Side::Sell };

                    let tracked = TrackedOrder::with_cloid(oid, cloid.clone(), side, 0.0, 0.0, 0.0);

                    self.orders.insert(oid, tracked);
                    self.cloid_to_oid.insert(cloid.clone(), oid);

                    results.push(ActionResult::single(oid, Some(cloid), filled, 0.0));
                } else if let Some(error) = status.get("error").and_then(|e| e.as_str()) {
                    results.push(ActionResult::error(error.to_string()));
                } else {
                    results.push(ActionResult::error("Unknown status".to_string()));
                }
            }
        } else {
            // Single error for the whole batch
            let error = self.extract_error_from_response(payload);
            results.push(ActionResult::error(
                error.unwrap_or_else(|| "Bulk order failed".to_string()),
            ));
        }

        results
    }

    fn process_cancel_response(
        &mut self,
        oid: u64,
        payload: &serde_json::Value,
    ) -> Vec<ActionResult> {
        // Check if cancel was successful
        let success = payload
            .get("status")
            .and_then(|s| s.as_str())
            .map(|s| s == "ok")
            .unwrap_or(false);

        if success {
            if let Some(order) = self.orders.get_mut(&oid) {
                order.transition_to(OrderState::CancelConfirmed);
            }
            info!(oid = oid, "Cancel confirmed via WS post");
            vec![ActionResult::single(oid, None, false, 0.0)]
        } else {
            if let Some(order) = self.orders.get_mut(&oid) {
                // Revert to previous state
                order.transition_to(OrderState::Resting);
            }
            let error = self.extract_error_from_response(payload);
            warn!(oid = oid, error = ?error, "Cancel failed");
            vec![ActionResult::error(
                error.unwrap_or_else(|| "Cancel failed".to_string()),
            )]
        }
    }

    fn process_modify_response(
        &mut self,
        inflight: InflightRequest,
        payload: &serde_json::Value,
        original_oid: u64,
    ) -> Vec<ActionResult> {
        // Modify results in a new OID
        let new_oid = self.extract_oid_from_response(payload);

        if let Some(new_oid) = new_oid {
            // Remove old order
            if let Some(old_order) = self.orders.remove(&original_oid) {
                // Create new tracked order with updated details
                let side = old_order.side;
                let tracked = TrackedOrder::with_cloid(
                    new_oid,
                    inflight.cloid.clone(),
                    side,
                    inflight.price,
                    inflight.size,
                    old_order.mid_at_placement, // preserve mid from original order
                );

                self.orders.insert(new_oid, tracked);
                self.cloid_to_oid.insert(inflight.cloid.clone(), new_oid);

                // Remove old CLOID mapping
                if let Some(old_cloid) = old_order.cloid {
                    self.cloid_to_oid.remove(&old_cloid);
                }
            }

            info!(
                old_oid = original_oid,
                new_oid = new_oid,
                "Order modified via WS post"
            );
            vec![ActionResult::single(
                new_oid,
                Some(inflight.cloid),
                false,
                inflight.size,
            )]
        } else {
            let error = self.extract_error_from_response(payload);
            warn!(oid = original_oid, error = ?error, "Modify failed");
            vec![ActionResult::error(
                error.unwrap_or_else(|| "Modify failed".to_string()),
            )]
        }
    }

    fn process_bulk_cancel_response(
        &mut self,
        oids: Vec<u64>,
        _payload: &serde_json::Value,
    ) -> Vec<ActionResult> {
        // Mark all as cancelled for now (simplified)
        for oid in &oids {
            if let Some(order) = self.orders.get_mut(oid) {
                order.transition_to(OrderState::CancelConfirmed);
            }
        }
        vec![ActionResult::empty()]
    }

    // =========================================================================
    // Order Updates Handler
    // =========================================================================

    /// Handle an orderUpdates event from WebSocket subscription.
    pub fn handle_order_update(&mut self, event: &WsOrderUpdateEvent) {
        let oid = event.oid;

        // Try to find order by OID, or by CLOID if not found
        // First resolve the actual OID (may be mapped via CLOID)
        let actual_oid = if self.orders.contains_key(&oid) {
            Some(oid)
        } else {
            event
                .cloid
                .as_ref()
                .and_then(|c| self.cloid_to_oid.get(c).copied())
                .filter(|&mapped_oid| self.orders.contains_key(&mapped_oid))
        };

        let Some(actual_oid) = actual_oid else {
            // Could be from a previous session or race condition
            debug!(
                oid = oid,
                status = %event.status,
                "orderUpdate for unknown order"
            );
            return;
        };

        // Safe to unwrap since we verified the key exists above
        let order = self.orders.get_mut(&actual_oid).unwrap();

        // Update state based on status
        match event.status.as_str() {
            "open" => {
                if !matches!(order.state, OrderState::Resting | OrderState::PartialFilled) {
                    order.transition_to(OrderState::Resting);
                }
                // Update size if different (could be post-partial-fill)
                if (order.size - event.size).abs() > 1e-10 {
                    order.size = event.size;
                }
            }
            "filled" => {
                order.transition_to(OrderState::Filled);
                order.filled = event.orig_size;
            }
            "canceled" | "marginCanceled" => {
                // Only transition if not already in a terminal state.
                // marginCanceled: exchange cancelled due to insufficient margin.
                if !order.is_terminal() {
                    if event.status == "marginCanceled" {
                        warn!(
                            oid = oid,
                            "Order margin-cancelled by exchange — insufficient margin"
                        );
                    }
                    order.transition_to(OrderState::Cancelled);
                }
            }
            status => {
                warn!(
                    oid = oid,
                    status = status,
                    "Unknown order status in orderUpdate"
                );
            }
        }

        debug!(
            oid = oid,
            status = %event.status,
            state = ?order.state,
            "Order state updated from orderUpdate"
        );
    }

    // =========================================================================
    // Fill Processing
    // =========================================================================

    /// Handle a fill event from userFills subscription.
    ///
    /// Returns the fill size if it was a new (non-duplicate) fill.
    pub fn handle_fill(
        &mut self,
        event: &WsFillEvent,
        position_tracker: &mut PositionTracker,
    ) -> Option<f64> {
        // Dedup by trade ID
        if self.processed_tids.contains(&event.tid) {
            debug!(tid = event.tid, "Duplicate fill blocked");
            return None;
        }

        // Record the TID
        self.processed_tids.insert(event.tid);
        self.tid_timestamps.insert(event.tid, Instant::now());

        // Update position
        position_tracker.process_fill(event.size, event.is_buy);

        // Update order tracking
        if let Some(order) = self.orders.get_mut(&event.oid) {
            // record_fill_with_price handles dedup internally and updates filled amount
            if !order.record_fill_with_price(event.tid, event.size, event.price) {
                // Already processed at order level
                return None;
            }

            // State transition
            let remaining = order.remaining();
            if remaining < 1e-10 {
                if matches!(
                    order.state,
                    OrderState::CancelPending | OrderState::CancelConfirmed
                ) {
                    order.transition_to(OrderState::FilledDuringCancel);
                } else {
                    order.transition_to(OrderState::Filled);
                }
            } else if order.state == OrderState::Resting {
                order.transition_to(OrderState::PartialFilled);
            }

            info!(
                oid = event.oid,
                tid = event.tid,
                size = event.size,
                price = event.price,
                filled = order.filled,
                remaining = remaining,
                state = ?order.state,
                "Fill processed"
            );
        } else {
            // Order not tracked - could be from previous session
            warn!(oid = event.oid, tid = event.tid, "Fill for unknown order");
        }

        Some(event.size)
    }

    // =========================================================================
    // Query Methods
    // =========================================================================

    /// Get a tracked order by OID.
    pub fn get_order(&self, oid: u64) -> Option<&TrackedOrder> {
        self.orders.get(&oid)
    }

    /// Get a mutable reference to a tracked order.
    pub fn get_order_mut(&mut self, oid: u64) -> Option<&mut TrackedOrder> {
        self.orders.get_mut(&oid)
    }

    /// Get order by CLOID.
    pub fn get_order_by_cloid(&self, cloid: &str) -> Option<&TrackedOrder> {
        self.cloid_to_oid
            .get(cloid)
            .and_then(|oid| self.orders.get(oid))
    }

    /// Get all active orders for a side.
    pub fn get_orders_by_side(&self, side: Side) -> Vec<&TrackedOrder> {
        self.orders
            .values()
            .filter(|o| o.side == side && o.is_active())
            .collect()
    }

    /// Get all order IDs.
    pub fn order_ids(&self) -> Vec<u64> {
        self.orders.keys().copied().collect()
    }

    /// Get all active (open) order IDs.
    pub fn open_order_ids(&self) -> HashSet<u64> {
        self.orders
            .values()
            .filter(|o| o.is_active())
            .map(|o| o.oid)
            .collect()
    }

    /// Get count of inflight requests.
    pub fn inflight_count(&self) -> usize {
        self.inflight.len()
    }

    /// Get count of tracked orders.
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Check if a CLOID has an inflight request.
    pub fn is_inflight(&self, cloid: &str) -> bool {
        self.cloid_to_request.contains_key(cloid)
    }

    // =========================================================================
    // Cleanup and Maintenance
    // =========================================================================

    /// Clean up terminal orders and expired TIDs.
    ///
    /// Returns IDs of orders that were removed.
    pub fn cleanup(&mut self) -> Vec<u64> {
        let mut removed = Vec::new();

        // Clean up terminal orders past fill window
        let orders_to_remove: Vec<u64> = self
            .orders
            .iter()
            .filter(|(_, order)| {
                if order.is_terminal() {
                    // For terminal orders (Filled, Cancelled, etc.), check state_changed_at
                    order.state_changed_at.elapsed() >= self.config.fill_window_duration
                } else if order.state == OrderState::CancelConfirmed {
                    order.fill_window_expired(self.config.cancel_grace_period)
                } else {
                    false
                }
            })
            .map(|(oid, _)| *oid)
            .collect();

        for oid in orders_to_remove {
            if let Some(order) = self.orders.remove(&oid) {
                if let Some(cloid) = order.cloid {
                    self.cloid_to_oid.remove(&cloid);
                }
                removed.push(oid);
            }
        }

        // Clean up expired TIDs
        if self.last_cleanup.elapsed() > Duration::from_secs(60) {
            let cutoff = Instant::now() - self.config.tid_retention;
            let expired_tids: Vec<u64> = self
                .tid_timestamps
                .iter()
                .filter(|(_, ts)| **ts < cutoff)
                .map(|(tid, _)| *tid)
                .collect();

            for tid in expired_tids {
                self.processed_tids.remove(&tid);
                self.tid_timestamps.remove(&tid);
            }

            self.last_cleanup = Instant::now();
        }

        removed
    }

    /// Check for timed out inflight requests.
    ///
    /// Returns inflight requests that have timed out.
    pub fn check_timeouts(&mut self) -> Vec<InflightRequest> {
        let mut timed_out = Vec::new();

        let expired_ids: Vec<u64> = self
            .inflight
            .iter()
            .filter(|(_, req)| req.is_timed_out(self.config.ws_post_timeout))
            .map(|(id, _)| *id)
            .collect();

        for id in expired_ids {
            if let Some(req) = self.inflight.remove(&id) {
                self.cloid_to_request.remove(&req.cloid);
                timed_out.push(req);
            }
        }

        timed_out
    }

    /// Add an order directly (for REST fallback or reconciliation).
    pub fn add_order(&mut self, order: TrackedOrder) {
        let oid = order.oid;
        if let Some(ref cloid) = order.cloid {
            self.cloid_to_oid.insert(cloid.clone(), oid);
        }
        self.orders.insert(oid, order);
    }

    /// Remove an order.
    pub fn remove_order(&mut self, oid: u64) -> Option<TrackedOrder> {
        if let Some(order) = self.orders.remove(&oid) {
            if let Some(ref cloid) = order.cloid {
                self.cloid_to_oid.remove(cloid);
            }
            Some(order)
        } else {
            None
        }
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn extract_oid_from_response(&self, payload: &serde_json::Value) -> Option<u64> {
        payload
            .get("response")
            .and_then(|r| r.get("data"))
            .and_then(|d| d.get("statuses"))
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
            .and_then(|status| {
                status
                    .get("resting")
                    .and_then(|r| r.get("oid"))
                    .and_then(|o| o.as_u64())
                    .or_else(|| {
                        status
                            .get("filled")
                            .and_then(|f| f.get("oid"))
                            .and_then(|o| o.as_u64())
                    })
            })
    }

    fn extract_filled_from_response(&self, payload: &serde_json::Value) -> bool {
        payload
            .get("response")
            .and_then(|r| r.get("data"))
            .and_then(|d| d.get("statuses"))
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
            .map(|status| status.get("filled").is_some())
            .unwrap_or(false)
    }

    fn extract_resting_size_from_response(&self, payload: &serde_json::Value, default: f64) -> f64 {
        payload
            .get("response")
            .and_then(|r| r.get("data"))
            .and_then(|d| d.get("statuses"))
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
            .and_then(|status| {
                status
                    .get("resting")
                    .and_then(|r| r.get("sz"))
                    .and_then(|s| s.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
            })
            .unwrap_or(default)
    }

    fn extract_error_from_response(&self, payload: &serde_json::Value) -> Option<String> {
        payload
            .get("response")
            .and_then(|r| r.get("data"))
            .and_then(|d| d.get("statuses"))
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
            .and_then(|status| status.get("error"))
            .and_then(|e| e.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                payload
                    .get("status")
                    .filter(|s| s.as_str() != Some("ok"))
                    .and_then(|_| payload.get("response"))
                    .and_then(|r| r.as_str())
                    .map(|s| s.to_string())
            })
    }
}

impl Default for WsOrderStateManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_order() {
        let mut mgr = WsOrderStateManager::new();
        let spec = WsOrderSpec::new("BTC".to_string(), 50000.0, 0.01, true);

        let result = mgr.prepare_order(spec);
        assert!(result.is_ok());

        let (request_id, inflight) = result.unwrap();
        assert_eq!(request_id, 1);
        assert_eq!(inflight.asset, "BTC");
        assert_eq!(inflight.price, 50000.0);
        assert_eq!(inflight.size, 0.01);

        // Should be in inflight map
        assert_eq!(mgr.inflight_count(), 1);
    }

    #[test]
    fn test_max_inflight_limit() {
        let config = WsOrderStateConfig {
            max_inflight_requests: 2,
            ..Default::default()
        };
        let mut mgr = WsOrderStateManager::with_config(config);

        // First two should succeed
        assert!(mgr
            .prepare_order(WsOrderSpec::new("BTC".to_string(), 50000.0, 0.01, true))
            .is_ok());
        assert!(mgr
            .prepare_order(WsOrderSpec::new("BTC".to_string(), 49000.0, 0.01, true))
            .is_ok());

        // Third should fail
        let result = mgr.prepare_order(WsOrderSpec::new("BTC".to_string(), 48000.0, 0.01, true));
        assert!(result.is_err());
    }

    #[test]
    fn test_handle_fill_dedup() {
        let mut mgr = WsOrderStateManager::new();
        let mut position = PositionTracker::new(0.0);

        // Add a tracked order
        let order = TrackedOrder::new(123, Side::Buy, 50000.0, 0.02, 0.0);
        mgr.add_order(order);

        let fill = WsFillEvent {
            oid: 123,
            tid: 456,
            size: 0.01,
            price: 50000.0,
            is_buy: true,
            coin: "BTC".to_string(),
            cloid: None,
            timestamp: 0,
        };

        // First fill should process
        let result1 = mgr.handle_fill(&fill, &mut position);
        assert_eq!(result1, Some(0.01));

        // Duplicate should be blocked
        let result2 = mgr.handle_fill(&fill, &mut position);
        assert_eq!(result2, None);

        // Order should have only one fill recorded
        let order = mgr.get_order(123).unwrap();
        assert_eq!(order.fill_count(), 1);
        assert!((order.filled - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_handle_order_update() {
        let mut mgr = WsOrderStateManager::new();

        // Add a tracked order
        let order = TrackedOrder::new(123, Side::Buy, 50000.0, 0.01, 0.0);
        mgr.add_order(order);

        // Simulate filled update
        let update = WsOrderUpdateEvent {
            oid: 123,
            cloid: None,
            status: "filled".to_string(),
            size: 0.0,
            orig_size: 0.01,
            price: 50000.0,
            coin: "BTC".to_string(),
            is_buy: true,
            status_timestamp: 0,
        };

        mgr.handle_order_update(&update);

        let order = mgr.get_order(123).unwrap();
        assert_eq!(order.state, OrderState::Filled);
    }

    #[test]
    fn test_cleanup_terminal_orders() {
        let config = WsOrderStateConfig {
            fill_window_duration: Duration::from_millis(10),
            ..Default::default()
        };
        let mut mgr = WsOrderStateManager::with_config(config);

        // Add a filled order
        let mut order = TrackedOrder::new(123, Side::Buy, 50000.0, 0.01, 0.0);
        order.transition_to(OrderState::Filled);
        mgr.add_order(order);

        // Immediately - should not clean up
        let removed = mgr.cleanup();
        assert!(removed.is_empty());

        // Wait for fill window
        std::thread::sleep(Duration::from_millis(20));

        // Now should clean up
        let removed = mgr.cleanup();
        assert_eq!(removed, vec![123]);
        assert!(mgr.get_order(123).is_none());
    }
}
