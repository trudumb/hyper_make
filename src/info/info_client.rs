use std::collections::HashMap;

use alloy::primitives::Address;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;

use std::time::Duration;

use crate::{
    info::response_structs::{
        ActiveAssetDataResponse, CandlesSnapshotResponse, FundingHistoryResponse,
        L2SnapshotResponse, OpenOrdersResponse, OrderStatusResponse, RecentTradesResponse,
        ReferralResponse, UserFeesResponse, UserFillsResponse, UserFundingResponse,
        UserRateLimitResponse, UserStateResponse, UserTokenBalanceResponse,
    },
    meta::{AssetContext, Meta, PerpDex, PerpDexLimits, SpotMeta, SpotMetaAndAssetCtxs},
    prelude::*,
    req::HttpClient,
    types::OrderInfo,
    ws::{message_types::WsPostResponseData, Subscription, WsManager},
    BaseUrl, Error, Message,
};

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CandleSnapshotRequest {
    coin: String,
    interval: String,
    start_time: u64,
    end_time: u64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "camelCase")]
pub enum InfoRequest {
    /// Get user's clearinghouse state.
    /// For HIP-3 builder DEXs, include `dex` parameter to query DEX-specific state.
    #[serde(rename = "clearinghouseState")]
    #[serde(rename_all = "camelCase")]
    UserState {
        user: Address,
        /// Optional HIP-3 DEX name for DEX-specific clearinghouse state.
        /// If None, returns validator perps state.
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    #[serde(rename = "batchClearinghouseStates")]
    UserStates {
        users: Vec<Address>,
    },
    #[serde(rename = "spotClearinghouseState")]
    UserTokenBalances {
        user: Address,
    },
    UserFees {
        user: Address,
    },
    /// Get user's open orders. Use `dex` for HIP-3 builder DEXs.
    #[serde(rename_all = "camelCase")]
    OpenOrders {
        user: Address,
        /// Optional HIP-3 DEX name (e.g., "hyna", "flx").
        /// If None, returns validator perps orders only.
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    OrderStatus {
        user: Address,
        oid: u64,
    },
    /// Get perp metadata. Use `dex` for HIP-3 builder DEXs.
    #[serde(rename_all = "camelCase")]
    Meta {
        /// Optional HIP-3 DEX name (e.g., "hyena", "felix").
        /// If None, returns validator perps.
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    /// Get perp metadata with asset contexts.
    #[serde(rename_all = "camelCase")]
    MetaAndAssetCtxs {
        /// Optional HIP-3 DEX name.
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    /// List all available HIP-3 DEXs.
    PerpDexs,
    /// Get open interest limits for a HIP-3 DEX.
    #[serde(rename_all = "camelCase")]
    PerpDexLimits {
        dex: String,
    },
    SpotMeta,
    SpotMetaAndAssetCtxs,
    /// Get all mid prices. Use `dex` for HIP-3 builder DEXs.
    #[serde(rename_all = "camelCase")]
    AllMids {
        /// Optional HIP-3 DEX name.
        #[serde(skip_serializing_if = "Option::is_none")]
        dex: Option<String>,
    },
    UserFills {
        user: Address,
    },
    #[serde(rename_all = "camelCase")]
    FundingHistory {
        coin: String,
        start_time: u64,
        end_time: Option<u64>,
    },
    #[serde(rename_all = "camelCase")]
    UserFunding {
        user: Address,
        start_time: u64,
        end_time: Option<u64>,
    },
    L2Book {
        coin: String,
    },
    RecentTrades {
        coin: String,
    },
    #[serde(rename_all = "camelCase")]
    CandleSnapshot {
        req: CandleSnapshotRequest,
    },
    Referral {
        user: Address,
    },
    HistoricalOrders {
        user: Address,
    },
    ActiveAssetData {
        user: Address,
        coin: String,
    },
    /// Query user rate limit status.
    /// Returns cumulative volume, requests used/cap, and surplus.
    #[serde(rename = "userRateLimit")]
    UserRateLimit {
        user: Address,
    },
}

#[derive(Debug)]
pub struct InfoClient {
    pub http_client: HttpClient,
    pub(crate) ws_manager: Option<WsManager>,
    reconnect: bool,
}

impl InfoClient {
    pub async fn new(client: Option<Client>, base_url: Option<BaseUrl>) -> Result<InfoClient> {
        Self::new_internal(client, base_url, false).await
    }

    pub async fn with_reconnect(
        client: Option<Client>,
        base_url: Option<BaseUrl>,
    ) -> Result<InfoClient> {
        Self::new_internal(client, base_url, true).await
    }

    async fn new_internal(
        client: Option<Client>,
        base_url: Option<BaseUrl>,
        reconnect: bool,
    ) -> Result<InfoClient> {
        let client = client.unwrap_or_default();
        let base_url = base_url.unwrap_or(BaseUrl::Mainnet).get_url();

        Ok(InfoClient {
            http_client: HttpClient { client, base_url },
            ws_manager: None,
            reconnect,
        })
    }

    pub async fn subscribe(
        &mut self,
        subscription: Subscription,
        sender_channel: UnboundedSender<std::sync::Arc<Message>>,
    ) -> Result<u32> {
        if self.ws_manager.is_none() {
            let ws_manager = WsManager::new(
                format!("ws{}/ws", &self.http_client.base_url[4..]),
                self.reconnect,
            )
            .await?;
            self.ws_manager = Some(ws_manager);
        }

        let identifier =
            serde_json::to_string(&subscription).map_err(|e| Error::JsonParse(e.to_string()))?;

        self.ws_manager
            .as_mut()
            .ok_or(Error::WsManagerNotFound)?
            .add_subscription(identifier, sender_channel)
            .await
    }

    pub async fn unsubscribe(&mut self, subscription_id: u32) -> Result<()> {
        if self.ws_manager.is_none() {
            let ws_manager = WsManager::new(
                format!("ws{}/ws", &self.http_client.base_url[4..]),
                self.reconnect,
            )
            .await?;
            self.ws_manager = Some(ws_manager);
        }

        self.ws_manager
            .as_mut()
            .ok_or(Error::WsManagerNotFound)?
            .remove_subscription(subscription_id)
            .await
    }

    /// Ensure the WebSocket connection is established.
    ///
    /// This is useful for clients that only use WS POST (no subscriptions)
    /// but need the connection to be active.
    pub async fn ensure_connected(&mut self) -> Result<()> {
        if self.ws_manager.is_none() {
            let ws_manager = WsManager::new(
                format!("ws{}/ws", &self.http_client.base_url[4..]),
                self.reconnect,
            )
            .await?;
            self.ws_manager = Some(ws_manager);
        }
        Ok(())
    }

    async fn send_info_request<T: for<'a> Deserialize<'a>>(
        &self,
        info_request: InfoRequest,
    ) -> Result<T> {
        let data =
            serde_json::to_string(&info_request).map_err(|e| Error::JsonParse(e.to_string()))?;

        let return_data = self.http_client.post("/info", data).await?;
        serde_json::from_str(&return_data).map_err(|e| Error::JsonParse(e.to_string()))
    }

    /// Get user's open orders for validator perps.
    ///
    /// For HIP-3 builder DEXs, use `open_orders_for_dex()` instead.
    pub async fn open_orders(&self, address: Address) -> Result<Vec<OpenOrdersResponse>> {
        let input = InfoRequest::OpenOrders {
            user: address,
            dex: None,
        };
        self.send_info_request(input).await
    }

    /// Get user's open orders for a specific HIP-3 DEX.
    ///
    /// For HIP-3 builder-deployed perps, orders are per-DEX.
    /// Without the `dex` parameter, the API returns validator perps orders only.
    ///
    /// # Arguments
    /// * `address` - User's wallet address
    /// * `dex` - Optional HIP-3 DEX name (e.g., "hyna", "flx"). None for validator perps.
    pub async fn open_orders_for_dex(
        &self,
        address: Address,
        dex: Option<&str>,
    ) -> Result<Vec<OpenOrdersResponse>> {
        let input = InfoRequest::OpenOrders {
            user: address,
            dex: dex.map(|s| s.to_string()),
        };
        self.send_info_request(input).await
    }

    /// Get user's clearinghouse state for validator perps.
    ///
    /// For HIP-3 builder DEXs, use `user_state_for_dex()` instead.
    pub async fn user_state(&self, address: Address) -> Result<UserStateResponse> {
        let input = InfoRequest::UserState {
            user: address,
            dex: None,
        };
        self.send_info_request(input).await
    }

    /// Get user's clearinghouse state for a specific HIP-3 DEX.
    ///
    /// For HIP-3 builder-deployed perps, the clearinghouse state is per-DEX.
    /// Without the `dex` parameter, the API returns validator perps state.
    ///
    /// # Arguments
    /// * `address` - User's wallet address
    /// * `dex` - Optional HIP-3 DEX name (e.g., "hyna", "flx"). None for validator perps.
    pub async fn user_state_for_dex(
        &self,
        address: Address,
        dex: Option<&str>,
    ) -> Result<UserStateResponse> {
        let input = InfoRequest::UserState {
            user: address,
            dex: dex.map(String::from),
        };
        self.send_info_request(input).await
    }

    pub async fn user_states(&self, addresses: Vec<Address>) -> Result<Vec<UserStateResponse>> {
        let input = InfoRequest::UserStates { users: addresses };
        self.send_info_request(input).await
    }

    pub async fn user_token_balances(&self, address: Address) -> Result<UserTokenBalanceResponse> {
        let input = InfoRequest::UserTokenBalances { user: address };
        self.send_info_request(input).await
    }

    pub async fn user_fees(&self, address: Address) -> Result<UserFeesResponse> {
        let input = InfoRequest::UserFees { user: address };
        self.send_info_request(input).await
    }

    /// Get perp metadata for validator perps (default).
    ///
    /// For HIP-3 builder DEXs, use `meta_for_dex()` instead.
    pub async fn meta(&self) -> Result<Meta> {
        let input = InfoRequest::Meta { dex: None };
        self.send_info_request(input).await
    }

    /// Get perp metadata for a specific DEX.
    ///
    /// # Arguments
    /// - `dex`: Optional DEX name (e.g., "hyena", "felix").
    ///   If None, returns validator perps (same as `meta()`).
    ///
    /// # Example
    /// ```ignore
    /// // Get Hyena DEX metadata
    /// let meta = info_client.meta_for_dex(Some("hyena")).await?;
    /// ```
    pub async fn meta_for_dex(&self, dex: Option<&str>) -> Result<Meta> {
        let input = InfoRequest::Meta {
            dex: dex.map(String::from),
        };
        self.send_info_request(input).await
    }

    /// Get perp metadata with asset contexts for validator perps (default).
    pub async fn meta_and_asset_contexts(&self) -> Result<(Meta, Vec<AssetContext>)> {
        let input = InfoRequest::MetaAndAssetCtxs { dex: None };
        self.send_info_request(input).await
    }

    /// Get perp metadata with asset contexts for a specific DEX.
    pub async fn meta_and_asset_contexts_for_dex(
        &self,
        dex: Option<&str>,
    ) -> Result<(Meta, Vec<AssetContext>)> {
        let input = InfoRequest::MetaAndAssetCtxs {
            dex: dex.map(String::from),
        };
        self.send_info_request(input).await
    }

    /// List all available HIP-3 DEXs.
    ///
    /// Returns a vector where each element is either a `PerpDex` or `None`
    /// (for unregistered DEX IDs).
    ///
    /// # Example
    /// ```ignore
    /// let dexs = info_client.perp_dexs().await?;
    /// for dex in dexs.iter().flatten() {
    ///     println!("{} - {}", dex.name, dex.full_name);
    /// }
    /// ```
    pub async fn perp_dexs(&self) -> Result<Vec<Option<PerpDex>>> {
        let input = InfoRequest::PerpDexs;
        self.send_info_request(input).await
    }

    /// Get open interest limits for a specific HIP-3 DEX.
    ///
    /// # Arguments
    /// - `dex`: DEX name (e.g., "hyena", "felix")
    pub async fn perp_dex_limits(&self, dex: &str) -> Result<PerpDexLimits> {
        let input = InfoRequest::PerpDexLimits {
            dex: dex.to_string(),
        };
        self.send_info_request(input).await
    }

    pub async fn spot_meta(&self) -> Result<SpotMeta> {
        let input = InfoRequest::SpotMeta;
        self.send_info_request(input).await
    }

    pub async fn spot_meta_and_asset_contexts(&self) -> Result<Vec<SpotMetaAndAssetCtxs>> {
        let input = InfoRequest::SpotMetaAndAssetCtxs;
        self.send_info_request(input).await
    }

    /// Get all mid prices for validator perps (default).
    pub async fn all_mids(&self) -> Result<HashMap<String, String>> {
        let input = InfoRequest::AllMids { dex: None };
        self.send_info_request(input).await
    }

    /// Get all mid prices for a specific DEX.
    ///
    /// # Arguments
    /// - `dex`: Optional DEX name. If None, returns validator perps.
    pub async fn all_mids_for_dex(&self, dex: Option<&str>) -> Result<HashMap<String, String>> {
        let input = InfoRequest::AllMids {
            dex: dex.map(String::from),
        };
        self.send_info_request(input).await
    }

    pub async fn user_fills(&self, address: Address) -> Result<Vec<UserFillsResponse>> {
        let input = InfoRequest::UserFills { user: address };
        self.send_info_request(input).await
    }

    pub async fn funding_history(
        &self,
        coin: String,
        start_time: u64,
        end_time: Option<u64>,
    ) -> Result<Vec<FundingHistoryResponse>> {
        let input = InfoRequest::FundingHistory {
            coin,
            start_time,
            end_time,
        };
        self.send_info_request(input).await
    }

    pub async fn user_funding_history(
        &self,
        user: Address,
        start_time: u64,
        end_time: Option<u64>,
    ) -> Result<Vec<UserFundingResponse>> {
        let input = InfoRequest::UserFunding {
            user,
            start_time,
            end_time,
        };
        self.send_info_request(input).await
    }

    pub async fn recent_trades(&self, coin: String) -> Result<Vec<RecentTradesResponse>> {
        let input = InfoRequest::RecentTrades { coin };
        self.send_info_request(input).await
    }

    pub async fn l2_snapshot(&self, coin: String) -> Result<L2SnapshotResponse> {
        let input = InfoRequest::L2Book { coin };
        self.send_info_request(input).await
    }

    pub async fn candles_snapshot(
        &self,
        coin: String,
        interval: String,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<CandlesSnapshotResponse>> {
        let input = InfoRequest::CandleSnapshot {
            req: CandleSnapshotRequest {
                coin,
                interval,
                start_time,
                end_time,
            },
        };
        self.send_info_request(input).await
    }

    pub async fn query_order_by_oid(
        &self,
        address: Address,
        oid: u64,
    ) -> Result<OrderStatusResponse> {
        let input = InfoRequest::OrderStatus { user: address, oid };
        self.send_info_request(input).await
    }

    pub async fn query_referral_state(&self, address: Address) -> Result<ReferralResponse> {
        let input = InfoRequest::Referral { user: address };
        self.send_info_request(input).await
    }

    pub async fn historical_orders(&self, address: Address) -> Result<Vec<OrderInfo>> {
        let input = InfoRequest::HistoricalOrders { user: address };
        self.send_info_request(input).await
    }

    pub async fn active_asset_data(
        &self,
        user: Address,
        coin: String,
    ) -> Result<ActiveAssetDataResponse> {
        let input = InfoRequest::ActiveAssetData { user, coin };
        self.send_info_request(input).await
    }

    /// Query user rate limit status.
    ///
    /// Returns cumulative volume, requests used, requests cap, and surplus.
    /// Hyperliquid allows 1 request per $1 USD traded, with a 10,000 request
    /// initial buffer.
    ///
    /// # Example
    /// ```ignore
    /// let limits = info_client.user_rate_limit(address).await?;
    /// println!("Used: {}/{}, Headroom: {:.1}%",
    ///     limits.n_requests_used,
    ///     limits.n_requests_cap,
    ///     limits.headroom_pct() * 100.0
    /// );
    /// ```
    pub async fn user_rate_limit(&self, address: Address) -> Result<UserRateLimitResponse> {
        let input = InfoRequest::UserRateLimit { user: address };
        self.send_info_request(input).await
    }

    /// Force a WebSocket reconnection.
    ///
    /// Call this when the ConnectionSupervisor detects stale data and
    /// recommends reconnection. This proactively restarts the connection
    /// rather than waiting for ping/pong timeout.
    ///
    /// Returns Ok(()) if reconnection was initiated, Err if no WebSocket
    /// manager exists (subscriptions were never set up).
    pub async fn reconnect(&self) -> Result<()> {
        if let Some(ref ws) = self.ws_manager {
            ws.force_reconnect().await;
            Ok(())
        } else {
            Err(crate::Error::WsManagerNotFound)
        }
    }

    /// Send an action request via WebSocket POST.
    ///
    /// This provides lower latency than REST for order placement, cancellation,
    /// and modification operations. The WebSocket connection must be established
    /// (typically via a prior `subscribe()` call).
    ///
    /// # Arguments
    /// * `payload` - The signed action payload (order, cancel, modify, etc.)
    /// * `timeout` - Maximum time to wait for response
    ///
    /// # Returns
    /// The response data if successful, or an error on timeout/failure.
    ///
    /// # Example
    /// ```ignore
    /// let payload = serde_json::json!({
    ///     "type": "order",
    ///     "orders": [...],
    ///     "grouping": "na"
    /// });
    /// let response = info_client.ws_post_action(payload, Duration::from_secs(5)).await?;
    /// ```
    pub async fn ws_post_action(
        &self,
        payload: serde_json::Value,
        timeout: Duration,
    ) -> Result<WsPostResponseData> {
        if let Some(ref ws) = self.ws_manager {
            ws.post(payload, true, timeout).await
        } else {
            Err(crate::Error::WsManagerNotFound)
        }
    }

    /// Send an info request via WebSocket POST.
    ///
    /// This provides lower latency than REST for info queries like L2 book,
    /// open orders, etc. The WebSocket connection must be established.
    ///
    /// # Arguments
    /// * `payload` - The info request payload
    /// * `timeout` - Maximum time to wait for response
    ///
    /// # Returns
    /// The response data if successful, or an error on timeout/failure.
    pub async fn ws_post_info(
        &self,
        payload: serde_json::Value,
        timeout: Duration,
    ) -> Result<WsPostResponseData> {
        if let Some(ref ws) = self.ws_manager {
            ws.post(payload, false, timeout).await
        } else {
            Err(crate::Error::WsManagerNotFound)
        }
    }

    /// Check if WebSocket manager is available for WS POST.
    pub fn has_ws_manager(&self) -> bool {
        self.ws_manager.is_some()
    }
}
