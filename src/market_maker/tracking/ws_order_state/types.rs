//! Types for WebSocket-based order state management.

use crate::market_maker::tracking::order_manager::Side;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Configuration for WebSocket order state management.
#[derive(Debug, Clone)]
pub struct WsOrderStateConfig {
    /// Timeout for WS post responses (default: 5s)
    pub ws_post_timeout: Duration,

    /// Grace period after cancel before removing order (default: 2s)
    pub cancel_grace_period: Duration,

    /// Maximum inflight requests (rate limit protection, default: 100)
    pub max_inflight_requests: usize,

    /// How long to keep processed TIDs for dedup (default: 5min)
    pub tid_retention: Duration,

    /// Fill window duration for late fills (default: 2s)
    pub fill_window_duration: Duration,

    /// Whether to use WS post for orders (false = use REST)
    pub use_ws_post: bool,
}

impl Default for WsOrderStateConfig {
    fn default() -> Self {
        Self {
            ws_post_timeout: Duration::from_secs(5),
            cancel_grace_period: Duration::from_secs(2),
            max_inflight_requests: 100,
            tid_retention: Duration::from_secs(300),
            fill_window_duration: Duration::from_secs(2),
            use_ws_post: true,
        }
    }
}

/// Type of order request in flight.
#[derive(Debug, Clone)]
pub enum RequestType {
    /// Single order placement
    Place { is_buy: bool },
    /// Order modification
    Modify { original_oid: u64 },
    /// Order cancellation
    Cancel { oid: u64 },
    /// Bulk order placement
    BulkPlace { count: usize, is_buy: bool },
    /// Bulk order cancellation
    BulkCancel { oids: Vec<u64> },
}

impl RequestType {
    pub fn is_buy(&self) -> Option<bool> {
        match self {
            RequestType::Place { is_buy } | RequestType::BulkPlace { is_buy, .. } => Some(*is_buy),
            _ => None,
        }
    }
}

/// Tracks an in-flight WS post request awaiting response.
#[derive(Debug)]
pub struct InflightRequest {
    /// Unique request ID for WS post correlation
    pub request_id: u64,
    /// Client order ID for cross-event correlation
    pub cloid: String,
    /// Asset being traded
    pub asset: String,
    /// Limit price
    pub price: f64,
    /// Order size
    pub size: f64,
    /// Type of request
    pub request_type: RequestType,
    /// When the request was sent
    pub sent_at: Instant,
    /// Number of retry attempts
    pub retry_count: u8,
}

impl InflightRequest {
    pub fn new(
        request_id: u64,
        cloid: String,
        asset: String,
        price: f64,
        size: f64,
        request_type: RequestType,
    ) -> Self {
        Self {
            request_id,
            cloid,
            asset,
            price,
            size,
            request_type,
            sent_at: Instant::now(),
            retry_count: 0,
        }
    }

    /// Check if this request has timed out.
    pub fn is_timed_out(&self, timeout: Duration) -> bool {
        self.sent_at.elapsed() > timeout
    }

    /// Get the side of the request (if applicable).
    pub fn side(&self) -> Option<Side> {
        self.request_type
            .is_buy()
            .map(|is_buy| if is_buy { Side::Buy } else { Side::Sell })
    }
}

/// WS post request message format.
#[derive(Debug, Serialize)]
pub struct WsPostRequest {
    pub method: &'static str,
    pub id: u64,
    pub request: WsRequestPayload,
}

impl WsPostRequest {
    pub fn new(id: u64, payload: WsRequestPayload) -> Self {
        Self {
            method: "post",
            id,
            request: payload,
        }
    }
}

/// Payload for WS post request.
#[derive(Debug, Serialize)]
pub struct WsRequestPayload {
    #[serde(rename = "type")]
    pub request_type: String,
    pub payload: serde_json::Value,
}

impl WsRequestPayload {
    pub fn action(payload: serde_json::Value) -> Self {
        Self {
            request_type: "action".to_string(),
            payload,
        }
    }

    pub fn info(payload: serde_json::Value) -> Self {
        Self {
            request_type: "info".to_string(),
            payload,
        }
    }
}

/// WS post response message format.
#[derive(Debug, Deserialize)]
pub struct WsPostResponse {
    pub channel: String,
    pub data: WsPostResponseData,
}

/// Response data from WS post.
#[derive(Debug, Deserialize)]
pub struct WsPostResponseData {
    pub id: u64,
    pub response: WsResponsePayload,
}

/// Payload of WS post response.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum WsResponsePayload {
    #[serde(rename = "action")]
    Action { payload: serde_json::Value },
    #[serde(rename = "info")]
    Info { payload: serde_json::Value },
    #[serde(rename = "error")]
    Error { payload: String },
}

/// Result of parsing an action response.
#[derive(Debug)]
pub struct ActionResult {
    /// Order IDs from successful placements
    pub oids: Vec<u64>,
    /// CLOIDs for each order
    pub cloids: Vec<Option<String>>,
    /// Whether orders were filled immediately
    pub filled: Vec<bool>,
    /// Resting sizes
    pub resting_sizes: Vec<f64>,
    /// Error messages (if any)
    pub errors: Vec<Option<String>>,
}

impl ActionResult {
    pub fn empty() -> Self {
        Self {
            oids: Vec::new(),
            cloids: Vec::new(),
            filled: Vec::new(),
            resting_sizes: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn single(oid: u64, cloid: Option<String>, filled: bool, resting_size: f64) -> Self {
        Self {
            oids: vec![oid],
            cloids: vec![cloid],
            filled: vec![filled],
            resting_sizes: vec![resting_size],
            errors: vec![None],
        }
    }

    pub fn error(error: String) -> Self {
        Self {
            oids: vec![0],
            cloids: vec![None],
            filled: vec![false],
            resting_sizes: vec![0.0],
            errors: vec![Some(error)],
        }
    }
}

/// Atomic counter for generating unique request IDs.
#[derive(Debug, Default)]
pub struct RequestIdGenerator {
    counter: AtomicU64,
}

impl RequestIdGenerator {
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(1),
        }
    }

    pub fn next(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::SeqCst)
    }
}

/// Statistics from reconciliation.
#[derive(Debug, Default)]
pub struct ReconcileStats {
    /// Orders recovered from timeout
    pub recovered: usize,
    /// Orphan orders cancelled
    pub orphans_cancelled: usize,
    /// Stale local orders removed
    pub stale_removed: usize,
    /// Total time taken
    pub duration: Duration,
}

/// Fill event from WebSocket.
#[derive(Debug, Clone)]
pub struct WsFillEvent {
    /// Order ID
    pub oid: u64,
    /// Trade ID (for deduplication)
    pub tid: u64,
    /// Fill size
    pub size: f64,
    /// Fill price
    pub price: f64,
    /// Is buy side
    pub is_buy: bool,
    /// Asset/coin
    pub coin: String,
    /// Client order ID (if available)
    pub cloid: Option<String>,
    /// Timestamp (ms since epoch)
    pub timestamp: u64,
}

/// Order update event from WebSocket.
#[derive(Debug, Clone)]
pub struct WsOrderUpdateEvent {
    /// Order ID
    pub oid: u64,
    /// Client order ID (if available)
    pub cloid: Option<String>,
    /// Order status ("open", "filled", "canceled", etc.)
    pub status: String,
    /// Current size remaining
    pub size: f64,
    /// Original size
    pub orig_size: f64,
    /// Limit price
    pub price: f64,
    /// Asset/coin
    pub coin: String,
    /// Is buy side
    pub is_buy: bool,
    /// Status timestamp (ms since epoch)
    pub status_timestamp: u64,
}

/// Order specification for placement.
#[derive(Debug, Clone)]
pub struct WsOrderSpec {
    pub asset: String,
    pub price: f64,
    pub size: f64,
    pub is_buy: bool,
    pub cloid: Option<String>,
    pub reduce_only: bool,
}

impl WsOrderSpec {
    pub fn new(asset: String, price: f64, size: f64, is_buy: bool) -> Self {
        Self {
            asset,
            price,
            size,
            is_buy,
            cloid: None,
            reduce_only: false,
        }
    }

    pub fn with_cloid(mut self, cloid: String) -> Self {
        self.cloid = Some(cloid);
        self
    }

    pub fn reduce_only(mut self) -> Self {
        self.reduce_only = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inflight_request_timeout() {
        let req = InflightRequest::new(
            1,
            "test-cloid".to_string(),
            "BTC".to_string(),
            50000.0,
            0.01,
            RequestType::Place { is_buy: true },
        );

        // Should not be timed out immediately
        assert!(!req.is_timed_out(Duration::from_secs(5)));

        // Would be timed out with 0 timeout
        assert!(req.is_timed_out(Duration::ZERO));
    }

    #[test]
    fn test_request_id_generator() {
        let gen = RequestIdGenerator::new();
        let id1 = gen.next();
        let id2 = gen.next();
        let id3 = gen.next();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[test]
    fn test_action_result_single() {
        let result = ActionResult::single(123, Some("cloid".to_string()), false, 0.01);
        assert_eq!(result.oids.len(), 1);
        assert_eq!(result.oids[0], 123);
        assert!(!result.filled[0]);
    }
}
