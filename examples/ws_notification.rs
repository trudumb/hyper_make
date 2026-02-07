use alloy::primitives::address;
use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use std::sync::Arc;
use tokio::sync::mpsc::unbounded_channel;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut info_client = InfoClient::new(None, Some(BaseUrl::Testnet)).await.unwrap();
    let user = address!("0xc64cc00b46101bd40aa1c3121195e85c0b0918d8");

    let (sender, mut receiver) = unbounded_channel::<Arc<Message>>();
    let subscription_id = info_client
        .subscribe(Subscription::Notification { user }, sender)
        .await
        .unwrap();

    info!("Subscribed to notifications. Press Ctrl+C to stop.");

    loop {
        tokio::select! {
            Some(arc_msg) = receiver.recv() => {
                if let Message::Notification(notification) = &*arc_msg {
                    info!("Received notification data: {notification:?}");
                }
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
