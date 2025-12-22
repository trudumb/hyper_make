use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use tokio::sync::mpsc::unbounded_channel;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await.unwrap();

    let (sender, mut receiver) = unbounded_channel();
    let subscription_id = info_client
        .subscribe(
            Subscription::ActiveAssetCtx {
                coin: "@107".to_string(), //spot index for hype token
            },
            sender,
        )
        .await
        .unwrap();

    info!("Subscribed to spot price for HYPE. Press Ctrl+C to stop.");

    loop {
        tokio::select! {
            Some(Message::ActiveSpotAssetCtx(spot_ctx)) = receiver.recv() => {
                info!("Received spot price data: {spot_ctx:?}");
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Shutting down...");
                break;
            }
        }
    }

    info_client.unsubscribe(subscription_id).await.unwrap();
    info!("Unsubscribed and exiting cleanly.");
}
