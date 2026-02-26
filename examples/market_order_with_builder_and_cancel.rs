use std::{env, thread::sleep, time::Duration};

use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{
    BaseUrl, BuilderInfo, ExchangeClient, ExchangeDataStatus, ExchangeResponseStatus,
    MarketCloseParams, MarketOrderParams,
};
use log::info;

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let private_key = env::var("HYPERLIQUID_PRIVATE_KEY")
        .expect("HYPERLIQUID_PRIVATE_KEY must be set in environment or .env file");
    let wallet: PrivateKeySigner = private_key.parse().expect("Invalid private key");

    let exchange_client = ExchangeClient::new(None, wallet, Some(BaseUrl::Testnet), None, None)
        .await
        .unwrap();

    // Market open order
    let market_open_params = MarketOrderParams {
        asset: "ETH",
        is_buy: true,
        sz: 0.01,
        px: None,
        slippage: Some(0.01), // 1% slippage
        cloid: None,
        wallet: None,
    };

    let fee = 1;
    let builder = "0x1ab189B7801140900C711E458212F9c76F8dAC79";

    let response = exchange_client
        .market_open_with_builder(
            market_open_params,
            BuilderInfo {
                builder: builder.to_string(),
                fee,
            },
        )
        .await
        .unwrap();
    info!("Market open order placed: {response:?}");

    let response = match response {
        ExchangeResponseStatus::Ok(exchange_response) => exchange_response,
        ExchangeResponseStatus::Err(e) => panic!("Error with exchange response: {e}"),
    };
    let status = response.data.unwrap().statuses[0].clone();
    match status {
        ExchangeDataStatus::Filled(order) => info!("Order filled: {order:?}"),
        ExchangeDataStatus::Resting(order) => info!("Order resting: {order:?}"),
        _ => panic!("Unexpected status: {status:?}"),
    };

    // Wait for a while before closing the position
    sleep(Duration::from_secs(10));

    // Market close order
    let market_close_params = MarketCloseParams {
        asset: "ETH",
        sz: None, // Close entire position
        px: None,
        slippage: Some(0.01), // 1% slippage
        cloid: None,
        wallet: None,
    };

    let response = exchange_client
        .market_close(market_close_params)
        .await
        .unwrap();
    info!("Market close order placed: {response:?}");

    let response = match response {
        ExchangeResponseStatus::Ok(exchange_response) => exchange_response,
        ExchangeResponseStatus::Err(e) => panic!("Error with exchange response: {e}"),
    };
    let status = response.data.unwrap().statuses[0].clone();
    match status {
        ExchangeDataStatus::Filled(order) => info!("Close order filled: {order:?}"),
        ExchangeDataStatus::Resting(order) => info!("Close order resting: {order:?}"),
        _ => panic!("Unexpected status: {status:?}"),
    };
}
