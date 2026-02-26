//! Diagnostic binary to check user balance on mainnet.
//!
//! Run with: cargo run --bin check_balance
//! Run with DEX: cargo run --bin check_balance -- --dex hyna

use hyperliquid_rust_sdk::{BaseUrl, InfoClient};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file
    dotenvy::dotenv().ok();

    // Get wallet address from environment (try both common names)
    let wallet_key = env::var("HYPERLIQUID_PRIVATE_KEY")
        .or_else(|_| env::var("HL_PRIVATE_KEY"))
        .expect("HYPERLIQUID_PRIVATE_KEY or HL_PRIVATE_KEY environment variable not set");

    // Parse wallet address from private key
    let wallet = wallet_key.parse::<alloy::signers::local::PrivateKeySigner>()?;
    let user_address = wallet.address();

    // Check for --dex argument
    let args: Vec<String> = env::args().collect();
    let dex = args
        .iter()
        .position(|a| a == "--dex")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    println!("=== Balance Check for {user_address} ===");
    if let Some(dex_name) = dex {
        println!("=== HIP-3 DEX: {dex_name} ===");
    }
    println!();

    // Create info client for mainnet
    let info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await?;

    // Query user state (validator perps OR HIP-3 DEX)
    println!(
        "Querying user state{}...",
        dex.map(|d| format!(" for DEX '{d}'")).unwrap_or_default()
    );
    let user_state = info_client.user_state_for_dex(user_address, dex).await?;

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

    // Print asset positions with full margin details
    println!("\n--- Asset Positions (with margin details) ---");
    if user_state.asset_positions.is_empty() {
        println!("  No positions");
    } else {
        let mut total_margin_in_positions = 0.0;
        let mut total_unrealized_pnl = 0.0;
        for pos in &user_state.asset_positions {
            let margin: f64 = pos.position.margin_used.parse().unwrap_or(0.0);
            let pnl: f64 = pos.position.unrealized_pnl.parse().unwrap_or(0.0);
            total_margin_in_positions += margin;
            total_unrealized_pnl += pnl;
            println!(
                "  {}: szi={}, margin_used={}, unrealized_pnl={}, position_value={}, entry_px={:?}",
                pos.position.coin,
                pos.position.szi,
                pos.position.margin_used,
                pos.position.unrealized_pnl,
                pos.position.position_value,
                pos.position.entry_px,
            );
        }
        println!("\n  --- Position Totals ---");
        println!("  Total margin in positions: {total_margin_in_positions:.4}");
        println!("  Total unrealized PnL:      {total_unrealized_pnl:.4}");
        println!(
            "  Position equity:           {:.4}",
            total_margin_in_positions + total_unrealized_pnl
        );
    }

    // Calculate potential total account value
    println!("\n--- Calculated Account Value ---");
    let margin_account_value: f64 = user_state
        .margin_summary
        .account_value
        .parse()
        .unwrap_or(0.0);
    let withdrawable: f64 = user_state.withdrawable.parse().unwrap_or(0.0);
    let total_raw_usd: f64 = user_state
        .margin_summary
        .total_raw_usd
        .parse()
        .unwrap_or(0.0);

    println!("  margin_summary.account_value: {margin_account_value:.4}");
    println!("  margin_summary.total_raw_usd: {total_raw_usd:.4}");
    println!("  withdrawable:                 {withdrawable:.4}");
    println!("  ---");
    println!(
        "  account_value + withdrawable: {:.4}",
        margin_account_value + withdrawable
    );
    println!(
        "  total_raw_usd + withdrawable: {:.4}",
        total_raw_usd + withdrawable
    );

    println!("\n=== Done ===");

    Ok(())
}
