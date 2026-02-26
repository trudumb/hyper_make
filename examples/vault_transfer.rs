use std::env;

use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{BaseUrl, ExchangeClient};
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

    let usd = 5_000_000; // at least 5 USD
    let is_deposit = true;

    let res = exchange_client
        .vault_transfer(
            is_deposit,
            usd,
            Some(
                "0x1962905b0a2d0ce7907ae1a0d17f3e4a1f63dfb7"
                    .parse()
                    .unwrap(),
            ),
            None,
        )
        .await
        .unwrap();
    info!("Vault transfer result: {res:?}");
}
