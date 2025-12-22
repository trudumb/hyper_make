/*
This is an example of a basic market making strategy.

We subscribe to the current mid price and build a market around this price. Whenever our market becomes outdated, we place and cancel orders to renew it.
*/
use alloy::signers::local::PrivateKeySigner;
use hyperliquid_rust_sdk::{BaseUrl, MarketMaker, MarketMakerInput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    // Key was randomly generated for testing and shouldn't be used with any real funds
    let wallet: PrivateKeySigner =
        "e908f86dbb4d55ac876378565aafeabc187f6690f046459397b17d9b9a19688e"
            .parse()
            .expect("Invalid private key");
    let market_maker_input = MarketMakerInput {
        asset: "ETH".to_string(),
        target_liquidity: 0.25,
        max_bps_diff: 2,
        half_spread: 1,
        max_absolute_position_size: 0.5,
        decimals: 1,
        wallet,
        base_url: Some(BaseUrl::Testnet),
    };
    let mut market_maker = MarketMaker::new(market_maker_input).await?;
    market_maker.start().await?;
    Ok(())
}
