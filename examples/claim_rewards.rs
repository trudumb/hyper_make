use std::env;

use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{BaseUrl, ExchangeClient, ExchangeResponseStatus};
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

    let response = exchange_client.claim_rewards(None).await.unwrap();

    match response {
        ExchangeResponseStatus::Ok(exchange_response) => {
            info!("Rewards claimed successfully: {exchange_response:?}");
        }
        ExchangeResponseStatus::Err(e) => {
            info!("Failed to claim rewards: {e}");
        }
    }
}
