use futures_util::{SinkExt, StreamExt};
use std::env;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    // Basic logging setup
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Load private key from env (standard for this project)
    let private_key =
        env::var("HYPERLIQUID_PRIVATE_KEY").expect("HYPERLIQUID_PRIVATE_KEY must be set");
    let wallet: alloy::signers::local::PrivateKeySigner =
        private_key.parse().expect("Invalid private key");
    let user_address = wallet.address();

    log::info!("Starting WS Debugger for user: {}", user_address);

    // Create a channel for the WS manager (even if we don't fully use the manager, we need the channel)
    let (_sender, _receiver) = mpsc::unbounded_channel::<String>();

    // Connect to Mainnet WS (using ExchangeClient as a helper to get the URL)
    // NOTE: This is a simplified direct connection to test the subscription
    let url = "wss://api.hyperliquid.xyz/ws";
    log::info!("Connecting to {}", url);

    let (ws_stream, _) = tokio_tungstenite::connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();

    // Subscribe to WebData2
    let sub = serde_json::json!({
        "method": "subscribe",
        "subscription": {
            "type": "webData2",
            "user": user_address
        }
    });

    let sub_msg = tokio_tungstenite::tungstenite::Message::Text(sub.to_string());
    write.send(sub_msg).await?;
    log::info!("Subscribed to WebData2");

    // Read loop
    while let Some(msg) = read.next().await {
        let msg = msg?;
        if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
            // Log everything that looks like data
            if !text.contains("\"channel\":\"pong\"") {
                log::info!("RECEIVED: {}", text);

                // If we get a valid WebData2 message, break after a few
                if text.contains("webData2") {
                    log::info!("Got WebData2! Analyze the JSON above.");
                    // Keep running for a bit to see if updates come
                }
            }
        }
    }

    Ok(())
}
