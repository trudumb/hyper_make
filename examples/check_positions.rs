//! Check open positions on mainnet.
//!
//! Usage:
//!   cargo run --example check_positions
//!   cargo run --example check_positions -- 0xYourAddress
//!   cargo run --example check_positions -- 0xYourAddress --testnet

use alloy::primitives::Address;
use hyperliquid_rust_sdk::{BaseUrl, InfoClient};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut address_str: Option<String> = None;
    let mut use_testnet = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--testnet" => use_testnet = true,
            s if s.starts_with("0x") => address_str = Some(s.to_string()),
            _ => {}
        }
        i += 1;
    }

    // Get address from arg or env
    let address_str = address_str.or_else(|| env::var("HYPERLIQUID_ADDRESS").ok());

    let Some(address_str) = address_str else {
        eprintln!("Usage: check_positions [ADDRESS] [--testnet]");
        eprintln!("  ADDRESS: Ethereum address (0x...) or set HYPERLIQUID_ADDRESS env var");
        eprintln!("  --testnet: Use testnet instead of mainnet");
        std::process::exit(1);
    };

    let address: Address = address_str.parse()?;
    let network = if use_testnet {
        BaseUrl::Testnet
    } else {
        BaseUrl::Mainnet
    };
    let network_name = if use_testnet { "TESTNET" } else { "MAINNET" };

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Hyperliquid Position Checker - {}", network_name);
    println!("═══════════════════════════════════════════════════════════");
    println!("  Address: {}", address);
    println!("═══════════════════════════════════════════════════════════\n");

    let info_client = InfoClient::new(None, Some(network)).await?;

    // Query user state
    let user_state = info_client.user_state(address).await?;

    // Display margin summary
    println!("📊 Account Summary:");
    println!(
        "   Account Value:     ${}",
        user_state.margin_summary.account_value
    );
    println!(
        "   Total Margin Used: ${}",
        user_state.margin_summary.total_margin_used
    );
    println!(
        "   Total Position:    ${}",
        user_state.margin_summary.total_ntl_pos
    );
    println!("   Withdrawable:      ${}", user_state.withdrawable);
    println!();

    // Display positions
    let positions: Vec<_> = user_state
        .asset_positions
        .iter()
        .filter(|p| p.position.szi.parse::<f64>().unwrap_or(0.0).abs() > 1e-9)
        .collect();

    if positions.is_empty() {
        println!("📭 No open positions\n");
    } else {
        println!("📈 Open Positions ({}):", positions.len());
        println!(
            "   {:<6} {:>12} {:>12} {:>12} {:>12} {:>10}",
            "Asset", "Size", "Entry", "Mark", "PnL", "Leverage"
        );
        println!("   {}", "-".repeat(72));

        for p in &positions {
            let pos = &p.position;
            let size: f64 = pos.szi.parse().unwrap_or(0.0);
            let entry: f64 = pos
                .entry_px
                .as_ref()
                .map(|s| s.parse().unwrap_or(0.0))
                .unwrap_or(0.0);
            let pnl: f64 = pos.unrealized_pnl.parse().unwrap_or(0.0);
            let leverage = &pos.leverage;

            // Get mark price (position_value / size)
            let pos_value: f64 = pos.position_value.parse().unwrap_or(0.0);
            let mark = if size.abs() > 1e-9 {
                pos_value / size.abs()
            } else {
                0.0
            };

            let pnl_str = if pnl >= 0.0 {
                format!("+${:.2}", pnl)
            } else {
                format!("-${:.2}", pnl.abs())
            };

            println!(
                "   {:<6} {:>12.5} {:>12.2} {:>12.2} {:>12} {:>10}",
                pos.coin,
                size,
                entry,
                mark,
                pnl_str,
                format!("{}x", leverage.value)
            );
        }
        println!();
    }

    // Query open orders
    let open_orders = info_client.open_orders(address).await?;

    if open_orders.is_empty() {
        println!("📭 No open orders\n");
    } else {
        println!("📋 Open Orders ({}):", open_orders.len());
        println!(
            "   {:<6} {:>6} {:>12} {:>12} {:>16}",
            "Asset", "Side", "Size", "Price", "OID"
        );
        println!("   {}", "-".repeat(58));

        for order in &open_orders {
            println!(
                "   {:<6} {:>6} {:>12} {:>12} {:>16}",
                order.coin, order.side, order.sz, order.limit_px, order.oid
            );
        }
        println!();
    }

    Ok(())
}
