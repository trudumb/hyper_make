//! Diagnostic binary to check user balance on mainnet.
//!
//! Run with: cargo run --bin check_balance

use hyperliquid_rust_sdk::{BaseUrl, InfoClient};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get wallet address from environment or use a test address
    let wallet_key =
        env::var("HL_PRIVATE_KEY").expect("HL_PRIVATE_KEY environment variable not set");

    // Parse wallet address from private key
    let wallet = wallet_key.parse::<alloy::signers::local::PrivateKeySigner>()?;
    let user_address = wallet.address();

    println!("=== Balance Check for {} ===\n", user_address);

    // Create info client for mainnet
    let info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await?;

    // Query user state
    println!("Querying user state...");
    let user_state = info_client.user_state(user_address).await?;

    println!("\n--- margin_summary (isolated margin) ---");
    println!(
        "  account_value:     {}",
        user_state.margin_summary.account_value
    );
    println!(
        "  total_margin_used: {}",
        user_state.margin_summary.total_margin_used
    );
    println!(
        "  total_ntl_pos:     {}",
        user_state.margin_summary.total_ntl_pos
    );
    println!(
        "  total_raw_usd:     {}",
        user_state.margin_summary.total_raw_usd
    );

    println!("\n--- cross_margin_summary (cross margin) ---");
    println!(
        "  account_value:     {}",
        user_state.cross_margin_summary.account_value
    );
    println!(
        "  total_margin_used: {}",
        user_state.cross_margin_summary.total_margin_used
    );
    println!(
        "  total_ntl_pos:     {}",
        user_state.cross_margin_summary.total_ntl_pos
    );
    println!(
        "  total_raw_usd:     {}",
        user_state.cross_margin_summary.total_raw_usd
    );

    println!("\n--- withdrawable ---");
    println!("  {}", user_state.withdrawable);

    // Also query spot balances
    println!("\n--- Spot Balances ---");
    let spot_balances = info_client.user_token_balances(user_address).await?;
    for balance in &spot_balances.balances {
        let total: f64 = balance.total.parse().unwrap_or(0.0);
        if total > 0.0 {
            println!(
                "  {}: total={}, hold={}",
                balance.coin, balance.total, balance.hold
            );
        }
    }

    // Print asset positions
    println!("\n--- Asset Positions ---");
    if user_state.asset_positions.is_empty() {
        println!("  No positions");
    } else {
        for pos in &user_state.asset_positions {
            println!(
                "  {}: szi={}, entry_px={:?}, liquidation_px={:?}",
                pos.position.coin,
                pos.position.szi,
                pos.position.entry_px,
                pos.position.liquidation_px
            );
        }
    }

    println!("\n=== Done ===");

    Ok(())
}
