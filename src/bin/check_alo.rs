use std::env;
use std::sync::Arc;
use tokio::time::Duration;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use hyperliquid_rust_sdk::{BaseUrl, InfoClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    
    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    // Get configuration
    let private_key = env::var("HYPERLIQUID_PRIVATE_KEY")
        .expect("HYPERLIQUID_PRIVATE_KEY must be set");
    let wallet: alloy::signers::local::PrivateKeySigner = private_key.parse()
        .expect("Invalid private key");
    let user_address = wallet.address();

    // Determine network (default to Testnet for safety)
    let args: Vec<String> = env::args().collect();
    let network = if args.len() > 1 && args[1].to_lowercase() == "mainnet" {
        info!("Using MAINNET");
        BaseUrl::Mainnet
    } else {
        info!("Using TESTNET");
        BaseUrl::Testnet
    };

    info!("Checking orders for user: {}", user_address);

    let info_client = InfoClient::new(None, Some(network)).await?;

    // Fetch open orders
    info!("Fetching open orders...");
    let open_orders = info_client.open_orders(user_address).await?;

    if open_orders.is_empty() {
        info!("No open orders found.");
        return Ok(());
    }

    info!("Found {} open orders. Checking TIF status...", open_orders.len());

    let mut alo_count = 0;
    let mut non_alo_count = 0;

    for order in open_orders {
        // Query detailed info for each order to get TIF
        match info_client.query_order_by_oid(user_address, order.oid).await {
            Ok(status_response) => {
                if let Some(order_info) = status_response.order {
                    let tif = order_info.order.tif.unwrap_or_else(|| "Unknown".to_string());
                    let is_alo = tif == "Alo";
                    
                    if is_alo {
                        alo_count += 1;
                    } else {
                        non_alo_count += 1;
                    }

                    info!(
                        "OID: {}, Coin: {}, Side: {}, Price: {}, Size: {}, TIF: {} [{}]",
                        order.oid,
                        order.coin,
                        order.side,
                        order.limit_px,
                        order.sz,
                        tif,
                        if is_alo { "OK (Post-Only)" } else { "WARNING (Not ALO)" }
                    );
                } else {
                    warn!("OID: {} - Order info not found (might be filled/cancelled)", order.oid);
                }
            }
            Err(e) => {
                warn!("Failed to query OID {}: {}", order.oid, e);
            }
        }
        // Rate limit nice-ness
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    info!("Summary:");
    info!("  Total Open Orders: {}", alo_count + non_alo_count);
    info!("  ALO (Post-Only):   {}", alo_count);
    info!("  Other (Gtc/Ioc):   {}", non_alo_count);

    if non_alo_count == 0 {
        info!("SUCCESS: All open orders are Post-Only.");
    } else {
        warn!("FAILURE: Found non-ALO orders!");
    }

    Ok(())
}
