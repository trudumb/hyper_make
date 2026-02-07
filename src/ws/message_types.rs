use serde::{Deserialize, Serialize};

use crate::types::{
    ActiveAssetCtxData, ActiveAssetDataData, ActiveSpotAssetCtxData, AllMidsData, BboData,
    CandleData, L2BookData, NotificationData, OrderUpdate, Trade, UserData, UserFillsData,
    UserFundingsData, UserNonFundingLedgerUpdatesData, WebData2Data,
};

#[derive(Deserialize, Clone, Debug)]
pub struct Trades {
    pub data: Vec<Trade>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct L2Book {
    pub data: L2BookData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct AllMids {
    pub data: AllMidsData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct User {
    pub data: UserData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct UserFills {
    pub data: UserFillsData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct CandleMessage {
    pub data: CandleData,
}

// Type alias for backwards compatibility
pub type Candle = CandleMessage;

#[derive(Deserialize, Clone, Debug)]
pub struct OrderUpdates {
    pub data: Vec<OrderUpdate>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct UserFundings {
    pub data: UserFundingsData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct UserNonFundingLedgerUpdates {
    pub data: UserNonFundingLedgerUpdatesData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Notification {
    pub data: NotificationData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct WebData2 {
    pub data: WebData2Data,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ActiveAssetCtx {
    pub data: ActiveAssetCtxData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ActiveSpotAssetCtx {
    pub data: ActiveSpotAssetCtxData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ActiveAssetData {
    pub data: ActiveAssetDataData,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Bbo {
    pub data: BboData,
}

/// OpenOrders WebSocket message (snapshot of current open orders).
#[derive(Deserialize, Clone, Debug)]
pub struct OpenOrders {
    pub data: OpenOrdersData,
}

/// Data payload for OpenOrders WebSocket message.
#[derive(Deserialize, Clone, Debug)]
pub struct OpenOrdersData {
    pub dex: String,
    pub user: alloy::primitives::Address,
    pub orders: Vec<OpenOrderEntry>,
}

/// Single order entry in OpenOrders snapshot.
#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct OpenOrderEntry {
    pub coin: String,
    pub limit_px: String,
    pub oid: u64,
    pub side: String,
    pub sz: String,
    pub timestamp: u64,
    pub orig_sz: String,
    pub cloid: Option<String>,
}

// =============================================================================
// WebSocket Posting Types (Phase 4)
// =============================================================================

/// WebSocket post request wrapper.
/// Used to send actions (orders, cancels) or info requests over WebSocket.
#[derive(Serialize, Clone, Debug)]
pub struct WsPostRequest {
    pub method: &'static str, // Always "post"
    pub id: u64,
    pub request: WsPostPayload,
}

impl WsPostRequest {
    /// Create a new WS post request for an action.
    pub fn action(id: u64, payload: serde_json::Value) -> Self {
        Self {
            method: "post",
            id,
            request: WsPostPayload::Action { payload },
        }
    }

    /// Create a new WS post request for an info query.
    pub fn info(id: u64, payload: serde_json::Value) -> Self {
        Self {
            method: "post",
            id,
            request: WsPostPayload::Info { payload },
        }
    }
}

/// Payload for a WS post request (either action or info).
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum WsPostPayload {
    /// Signed action (order, cancel, etc.)
    Action { payload: serde_json::Value },
    /// Info request (l2Book, openOrders, etc.)
    Info { payload: serde_json::Value },
}

/// WebSocket post response wrapper.
#[derive(Deserialize, Clone, Debug)]
pub struct WsPostResponse {
    pub channel: String, // Always "post"
    pub data: WsPostResponseData,
}

/// Data payload of a WS post response.
#[derive(Deserialize, Clone, Debug)]
pub struct WsPostResponseData {
    pub id: u64,
    pub response: WsPostResponsePayload,
}

/// The actual response payload (action result, info result, or error).
#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum WsPostResponsePayload {
    /// Successful action response
    Action { payload: serde_json::Value },
    /// Successful info response
    Info { payload: serde_json::Value },
    /// Error response (HTTP status code + description)
    Error { payload: String },
}

impl WsPostResponsePayload {
    /// Check if this is an error response.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get error message if this is an error.
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { payload } => Some(payload),
            _ => None,
        }
    }
}
