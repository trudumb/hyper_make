use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use tokio::sync::mpsc::unbounded_channel;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut info_client = InfoClient::new(None, Some(BaseUrl::Testnet)).await.unwrap();

    let (sender, mut receiver) = unbounded_channel();
    let subscription_id = info_client
        .subscribe(
            Subscription::L2Book {
                coin: "ETH".to_string(),
            },
            sender,
        )
        .await
        .unwrap();

    info!("Subscribed to L2 book for ETH. Press Ctrl+C to stop.");

    loop {
        tokio::select! {
            Some(Message::L2Book(l2_book)) = receiver.recv() => {
                info!("Received l2 book data: {l2_book:?}");
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
