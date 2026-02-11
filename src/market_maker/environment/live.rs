//! Live trading environment — wraps `HyperliquidExecutor` + full WS subscription.
//!
//! This is the **E_live** in the mathematical structure:
//! `S →[T,π,E_live]→ S'`
//!
//! The `LiveEnvironment` holds the real exchange executor and produces
//! observations from the full set of WebSocket channels.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::info;

use alloy::primitives::Address;

use crate::prelude::Result;
use crate::{InfoClient, Message, Subscription};

use super::stream::LiveObservationStream;
use super::TradingEnvironment;
use crate::market_maker::{
    CancelResult, HyperliquidExecutor, ModifyResult, ModifySpec, OrderExecutor, OrderResult,
    OrderSpec,
};

/// Configuration for the live environment's WS subscriptions.
#[derive(Clone, Debug)]
pub struct LiveEnvironmentConfig {
    /// Asset to trade (e.g., "BTC", "HYPE").
    pub asset: String,
    /// User's exchange address.
    pub user_address: Address,
    /// Optional HIP-3 DEX name (e.g., "hyena").
    pub dex: Option<String>,
}

/// Live trading environment.
///
/// Wraps `HyperliquidExecutor` for order execution and optionally `InfoClient`
/// for WebSocket subscriptions.
///
/// # Construction
///
/// - [`from_executor`](Self::from_executor): Phase 3 — executor-only, WS handled externally
/// - [`new`](Self::new): Phase 4+ — full environment with WS observation stream
pub struct LiveEnvironment {
    /// The real exchange executor.
    executor: HyperliquidExecutor,
    /// WS subscription client (optional during Phase 3 transition).
    info_client: Option<InfoClient>,
    /// Subscription configuration (needed for observation_stream).
    config: Option<LiveEnvironmentConfig>,
}

impl LiveEnvironment {
    /// Create a full live environment with WS subscription support.
    ///
    /// This enables `observation_stream()` to subscribe to all 9 WS channels.
    pub fn new(
        executor: HyperliquidExecutor,
        info_client: InfoClient,
        config: LiveEnvironmentConfig,
    ) -> Self {
        Self {
            executor,
            info_client: Some(info_client),
            config: Some(config),
        }
    }

    /// Create a live environment from executor only (Phase 3 transition).
    ///
    /// WS subscriptions are handled externally by MarketMaker's event loop.
    /// `observation_stream()` will return an error if called.
    pub fn from_executor(executor: HyperliquidExecutor) -> Self {
        Self {
            executor,
            info_client: None,
            config: None,
        }
    }

    /// Access the underlying InfoClient, if available.
    pub fn info_client(&self) -> Option<&InfoClient> {
        self.info_client.as_ref()
    }

    /// Mutably access the underlying InfoClient, if available.
    pub fn info_client_mut(&mut self) -> Option<&mut InfoClient> {
        self.info_client.as_mut()
    }
}

#[async_trait]
impl TradingEnvironment for LiveEnvironment {
    type ObservationStream = LiveObservationStream;

    async fn observation_stream(&mut self) -> Result<Self::ObservationStream> {
        let info_client = self
            .info_client
            .as_mut()
            .ok_or_else(|| crate::Error::GenericRequest(
                "LiveEnvironment::observation_stream() requires InfoClient. \
                 Use LiveEnvironment::new() instead of from_executor().".to_string()
            ))?;
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| crate::Error::GenericRequest(
                "LiveEnvironment missing config".to_string()
            ))?;

        let (sender, receiver) = mpsc::unbounded_channel::<Arc<Message>>();

        // Subscribe to all 9 WS channels
        info_client
            .subscribe(
                Subscription::UserFills {
                    user: config.user_address,
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::AllMids {
                    dex: config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::Trades {
                    coin: config.asset.clone(),
                    dex: config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::L2Book {
                    coin: config.asset.clone(),
                    dex: config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::OrderUpdates {
                    user: config.user_address,
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::OpenOrders {
                    user: config.user_address,
                    dex: config.dex.clone(),
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::ActiveAssetData {
                    user: config.user_address,
                    coin: config.asset.clone(),
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::WebData2 {
                    user: config.user_address,
                },
                sender.clone(),
            )
            .await?;

        info_client
            .subscribe(
                Subscription::UserNonFundingLedgerUpdates {
                    user: config.user_address,
                },
                sender.clone(),
            )
            .await?;

        drop(sender); // Done subscribing

        info!(
            asset = %config.asset,
            dex = ?config.dex,
            "LiveEnvironment: subscribed to all 9 WS channels"
        );

        Ok(LiveObservationStream::new(receiver))
    }

    // === Delegate all order execution to HyperliquidExecutor ===

    async fn place_order(
        &self,
        asset: &str,
        price: f64,
        size: f64,
        is_buy: bool,
        cloid: Option<String>,
        post_only: bool,
    ) -> OrderResult {
        self.executor
            .place_order(asset, price, size, is_buy, cloid, post_only)
            .await
    }

    async fn place_bulk_orders(&self, asset: &str, orders: Vec<OrderSpec>) -> Vec<OrderResult> {
        self.executor.place_bulk_orders(asset, orders).await
    }

    async fn place_ioc_reduce_order(
        &self,
        asset: &str,
        size: f64,
        is_buy: bool,
        slippage_bps: u32,
        mid_price: f64,
    ) -> OrderResult {
        self.executor
            .place_ioc_reduce_order(asset, size, is_buy, slippage_bps, mid_price)
            .await
    }

    async fn cancel_order(&self, asset: &str, oid: u64) -> CancelResult {
        self.executor.cancel_order(asset, oid).await
    }

    async fn cancel_bulk_orders(&self, asset: &str, oids: Vec<u64>) -> Vec<CancelResult> {
        self.executor.cancel_bulk_orders(asset, oids).await
    }

    async fn modify_order(
        &self,
        asset: &str,
        oid: u64,
        new_price: f64,
        new_size: f64,
        is_buy: bool,
        post_only: bool,
    ) -> ModifyResult {
        self.executor
            .modify_order(asset, oid, new_price, new_size, is_buy, post_only)
            .await
    }

    async fn modify_bulk_orders(
        &self,
        asset: &str,
        modifies: Vec<ModifySpec>,
    ) -> Vec<ModifyResult> {
        self.executor.modify_bulk_orders(asset, modifies).await
    }

    async fn sync_state(&mut self) -> Result<()> {
        // Live environment sync is handled by MarketMaker's startup sequence
        // (sync_open_orders, sync_position_from_exchange, cancel_all_orders_on_startup).
        // Those methods will be migrated here in Phase 4.
        Ok(())
    }

    fn is_live(&self) -> bool {
        true
    }
}
