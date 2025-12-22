use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use tokio::sync::mpsc::unbounded_channel;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut info_client = InfoClient::new(None, Some(BaseUrl::Testnet)).await.unwrap();

    let (sender, mut receiver) = unbounded_channel();
    let subscription_id = info_client
        .subscribe(Subscription::AllMids, sender)
        .await
        .unwrap();

    info!("Subscribed to all mids. Press Ctrl+C to stop.");

    loop {
        tokio::select! {
            Some(Message::AllMids(all_mids)) = receiver.recv() => {
                info!("Received mids data: {all_mids:?}");
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
