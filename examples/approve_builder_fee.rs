use std::env;

use alloy::{primitives::address, signers::local::PrivateKeySigner};
use hyperliquid_rust_sdk::{BaseUrl, ExchangeClient};
use log::info;

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let private_key = env::var("HYPERLIQUID_PRIVATE_KEY")
        .expect("HYPERLIQUID_PRIVATE_KEY must be set in environment or .env file");
    let wallet: PrivateKeySigner = private_key.parse().expect("Invalid private key");

    let exchange_client =
        ExchangeClient::new(None, wallet.clone(), Some(BaseUrl::Testnet), None, None)
            .await
            .unwrap();

    let max_fee_rate = "0.1%";
    let builder = address!("0x1ab189B7801140900C711E458212F9c76F8dAC79");

    let resp = exchange_client
        .approve_builder_fee(builder, max_fee_rate.to_string(), Some(&wallet))
        .await;
    info!("resp: {resp:#?}");
}
