use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use tokio::sync::mpsc::unbounded_channel;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut info_client = InfoClient::new(None, Some(BaseUrl::Testnet)).await.unwrap();
    let coin = "BTC".to_string();

    let (sender, mut receiver) = unbounded_channel();
    let subscription_id = info_client
        .subscribe(Subscription::ActiveAssetCtx { coin }, sender)
        .await
        .unwrap();

    info!("Subscribed to active asset context for BTC. Press Ctrl+C to stop.");

    loop {
        tokio::select! {
            Some(Message::ActiveAssetCtx(active_asset_ctx)) = receiver.recv() => {
                info!("Received active asset ctx: {active_asset_ctx:?}");
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
