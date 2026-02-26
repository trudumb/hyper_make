//! Order-related methods for ExchangeClient.
//!
//! This module contains methods for placing and managing orders:
//! - Single and bulk order placement
//! - Market orders (open/close)
//! - Builder fee integration

use alloy::signers::local::PrivateKeySigner;
use log::debug;

use crate::{
    exchange::{
        actions::BulkOrder,
        order::{MarketCloseParams, MarketOrderParams},
        BuilderInfo, ClientLimit, ClientOrder, ClientOrderRequest,
    },
    helpers::next_nonce,
    info::info_client::InfoClient,
    prelude::*,
    signature::sign_l1_action,
    BaseUrl, Error, ExchangeResponseStatus,
};

use super::exchange_client::{Actions, ExchangeClient};

impl ExchangeClient {
    /// Place a single order.
    pub async fn order(
        &self,
        order: ClientOrderRequest,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        self.bulk_order(vec![order], wallet).await
    }

    /// Place a single order with builder fee.
    pub async fn order_with_builder(
        &self,
        order: ClientOrderRequest,
        wallet: Option<&PrivateKeySigner>,
        builder: BuilderInfo,
    ) -> Result<ExchangeResponseStatus> {
        self.bulk_order_with_builder(vec![order], wallet, builder)
            .await
    }

    /// Place multiple orders in a single request.
    pub async fn bulk_order(
        &self,
        orders: Vec<ClientOrderRequest>,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let timestamp = next_nonce();

        let mut transformed_orders = Vec::new();

        for order in orders {
            transformed_orders.push(order.convert(&self.coin_to_asset)?);
        }

        let action = Actions::Order(BulkOrder {
            orders: transformed_orders,
            grouping: "na".to_string(),
            builder: None,
        });
        let connection_id = action.hash(timestamp, self.vault_address)?;
        let action = serde_json::to_value(&action).map_err(|e| Error::JsonParse(e.to_string()))?;

        let is_mainnet = self.http_client.is_mainnet();
        let signature = sign_l1_action(wallet, connection_id, is_mainnet)?;
        self.post(action, signature, timestamp).await
    }

    /// Place multiple orders with builder fee in a single request.
    pub async fn bulk_order_with_builder(
        &self,
        orders: Vec<ClientOrderRequest>,
        wallet: Option<&PrivateKeySigner>,
        mut builder: BuilderInfo,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let timestamp = next_nonce();

        builder.builder = builder.builder.to_lowercase();

        let mut transformed_orders = Vec::new();

        for order in orders {
            transformed_orders.push(order.convert(&self.coin_to_asset)?);
        }

        let action = Actions::Order(BulkOrder {
            orders: transformed_orders,
            grouping: "na".to_string(),
            builder: Some(builder),
        });
        let connection_id = action.hash(timestamp, self.vault_address)?;
        let action = serde_json::to_value(&action).map_err(|e| Error::JsonParse(e.to_string()))?;

        let is_mainnet = self.http_client.is_mainnet();
        let signature = sign_l1_action(wallet, connection_id, is_mainnet)?;
        self.post(action, signature, timestamp).await
    }

    /// Open a market position.
    pub async fn market_open(
        &self,
        params: MarketOrderParams<'_>,
    ) -> Result<ExchangeResponseStatus> {
        let slippage = params.slippage.unwrap_or(0.05); // Default 5% slippage
        let (px, sz_decimals) = self
            .calculate_slippage_price(params.asset, params.is_buy, slippage, params.px)
            .await?;

        let order = ClientOrderRequest {
            asset: params.asset.to_string(),
            is_buy: params.is_buy,
            reduce_only: false,
            limit_px: px,
            sz: round_to_decimals(params.sz, sz_decimals),
            cloid: params.cloid,
            order_type: ClientOrder::Limit(ClientLimit {
                tif: "Ioc".to_string(),
            }),
        };

        self.order(order, params.wallet).await
    }

    /// Open a market position with builder fee.
    pub async fn market_open_with_builder(
        &self,
        params: MarketOrderParams<'_>,
        builder: BuilderInfo,
    ) -> Result<ExchangeResponseStatus> {
        let slippage = params.slippage.unwrap_or(0.05); // Default 5% slippage
        let (px, sz_decimals) = self
            .calculate_slippage_price(params.asset, params.is_buy, slippage, params.px)
            .await?;

        let order = ClientOrderRequest {
            asset: params.asset.to_string(),
            is_buy: params.is_buy,
            reduce_only: false,
            limit_px: px,
            sz: round_to_decimals(params.sz, sz_decimals),
            cloid: params.cloid,
            order_type: ClientOrder::Limit(ClientLimit {
                tif: "Ioc".to_string(),
            }),
        };

        self.order_with_builder(order, params.wallet, builder).await
    }

    /// Close a market position.
    pub async fn market_close(
        &self,
        params: MarketCloseParams<'_>,
    ) -> Result<ExchangeResponseStatus> {
        let slippage = params.slippage.unwrap_or(0.05); // Default 5% slippage
        let wallet = params.wallet.unwrap_or(&self.wallet);

        let base_url = match self.http_client.base_url.as_str() {
            "https://api.hyperliquid.xyz" => BaseUrl::Mainnet,
            "https://api.hyperliquid-testnet.xyz" => BaseUrl::Testnet,
            _ => return Err(Error::GenericRequest("Invalid base URL".to_string())),
        };
        let info_client = InfoClient::new(None, Some(base_url)).await?;
        let user_state = info_client.user_state(wallet.address()).await?;

        let position = user_state
            .asset_positions
            .iter()
            .find(|p| p.position.coin == params.asset)
            .ok_or(Error::AssetNotFound)?;

        let szi = position
            .position
            .szi
            .parse::<f64>()
            .map_err(|_| Error::FloatStringParse)?;

        let (px, sz_decimals) = self
            .calculate_slippage_price(params.asset, szi < 0.0, slippage, params.px)
            .await?;

        let sz = round_to_decimals(params.sz.unwrap_or_else(|| szi.abs()), sz_decimals);

        let order = ClientOrderRequest {
            asset: params.asset.to_string(),
            is_buy: szi < 0.0,
            reduce_only: true,
            limit_px: px,
            sz,
            cloid: params.cloid,
            order_type: ClientOrder::Limit(ClientLimit {
                tif: "Ioc".to_string(),
            }),
        };

        self.order(order, Some(wallet)).await
    }

    /// Calculate the slippage-adjusted price for market orders.
    pub(crate) async fn calculate_slippage_price(
        &self,
        asset: &str,
        is_buy: bool,
        slippage: f64,
        px: Option<f64>,
    ) -> Result<(f64, u32)> {
        let base_url = match self.http_client.base_url.as_str() {
            "https://api.hyperliquid.xyz" => BaseUrl::Mainnet,
            "https://api.hyperliquid-testnet.xyz" => BaseUrl::Testnet,
            _ => return Err(Error::GenericRequest("Invalid base URL".to_string())),
        };
        let info_client = InfoClient::new(None, Some(base_url)).await?;
        let meta = info_client.meta().await?;

        let asset_meta = meta
            .universe
            .iter()
            .find(|a| a.name == asset)
            .ok_or(Error::AssetNotFound)?;

        let sz_decimals = asset_meta.sz_decimals;
        let max_decimals: u32 = if self.coin_to_asset[asset] < 10000 {
            6
        } else {
            8
        };
        let price_decimals = max_decimals.saturating_sub(sz_decimals);

        let px = if let Some(px) = px {
            px
        } else {
            let all_mids = info_client.all_mids().await?;
            all_mids
                .get(asset)
                .ok_or(Error::AssetNotFound)?
                .parse::<f64>()
                .map_err(|_| Error::FloatStringParse)?
        };

        debug!("px before slippage: {px:?}");
        let slippage_factor = if is_buy {
            1.0 + slippage
        } else {
            1.0 - slippage
        };
        let px = px * slippage_factor;

        // Round to the correct number of decimal places and significant figures
        let px = round_to_significant_and_decimal(px, 5, price_decimals);

        debug!("px after slippage: {px:?}");
        Ok((px, sz_decimals))
    }
}

fn round_to_decimals(value: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (value * factor).round() / factor
}

fn round_to_significant_and_decimal(value: f64, sig_figs: u32, max_decimals: u32) -> f64 {
    let abs_value = value.abs();
    let magnitude = abs_value.log10().floor() as i32;
    let scale = 10f64.powi(sig_figs as i32 - magnitude - 1);
    let rounded = (abs_value * scale).round() / scale;
    round_to_decimals(rounded.copysign(value), max_decimals)
}
