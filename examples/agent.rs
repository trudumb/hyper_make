use std::env;

use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{BaseUrl, ClientLimit, ClientOrder, ClientOrderRequest, ExchangeClient};
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

    /*
        Create a new wallet with the agent.
        This agent cannot transfer or withdraw funds, but can for example place orders.
    */

    let (agent_private_key, response) = exchange_client.approve_agent(None).await.unwrap();
    info!("Agent creation response: {response:?}");

    let agent_wallet = PrivateKeySigner::from_bytes(&agent_private_key).unwrap();

    info!("Agent address: {:?}", agent_wallet.address());

    let exchange_client =
        ExchangeClient::new(None, agent_wallet, Some(BaseUrl::Testnet), None, None)
            .await
            .unwrap();

    let order = ClientOrderRequest {
        asset: "ETH".to_string(),
        is_buy: true,
        reduce_only: false,
        limit_px: 1795.0,
        sz: 0.01,
        cloid: None,
        order_type: ClientOrder::Limit(ClientLimit {
            tif: "Gtc".to_string(),
        }),
    };

    let response = exchange_client.order(order, None).await.unwrap();

    info!("Order placed: {response:?}");
}
