use std::env;

use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{BaseUrl, ExchangeClient, InfoClient};
use log::info;

#[tokio::main]
async fn main() {
    // Example assumes you already have a position on ETH so you can update margin
    let _ = dotenvy::dotenv();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let private_key = env::var("HYPERLIQUID_PRIVATE_KEY")
        .expect("HYPERLIQUID_PRIVATE_KEY must be set in environment or .env file");
    let wallet: PrivateKeySigner = private_key.parse().expect("Invalid private key");

    let address = wallet.address();
    let exchange_client = ExchangeClient::new(None, wallet, Some(BaseUrl::Testnet), None, None)
        .await
        .unwrap();
    let info_client = InfoClient::new(None, Some(BaseUrl::Testnet)).await.unwrap();

    let response = exchange_client
        .update_leverage(5, "ETH", false, None)
        .await
        .unwrap();
    info!("Update leverage response: {response:?}");

    let response = exchange_client
        .update_isolated_margin(1.0, "ETH", None)
        .await
        .unwrap();

    info!("Update isolated margin response: {response:?}");

    let user_state = info_client.user_state(address).await.unwrap();
    info!("User state: {user_state:?}");
}
