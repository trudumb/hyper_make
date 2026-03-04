//! Hyperliquid Perp WebSocket feed for cross-venue lead-lag signal.
//!
//! Streams perp BBO and trade data from Hyperliquid's perpetual market
//! for assets where the perp leads the HIP-3 spot book.
//!
//! # Architecture
//!
//! ```text
//! HL Perp WS (wss://api.hyperliquid.xyz/ws)
//!     │
//!     ├── BBO subscription (best bid/ask, ~50ms updates)
//!     ├── Trades subscription (individual fills)
//!     │
//!     ▼
//! HlPerpFeed
//!     │
//!     ├── Computes mid price, spread
//!     ├── Computes perp-spot basis (premium/discount)
//!     ├── Sends to signal integrator via channel
//!     │
//!     ▼
//! Quote Engine (skews quotes based on perp signal)
//! ```
//!
//! # Key Insight
//!
//! For HIP-3 assets like HYPE, the perp market has:
//! - Higher liquidity (deeper books, tighter spreads)
//! - More informed flow (arb bots, systematic traders)
//! - Funding settlement every 8h creating predictable flow patterns
//!
//! Perp premium/discount to spot = directional signal.
//!
//! # Usage
//!
//! ```ignore
//! let (tx, rx) = tokio::sync::mpsc::channel(1000);
//! let feed = HlPerpFeed::for_asset("HYPE", tx);
//! tokio::spawn(feed.run());
//!
//! while let Some(update) = rx.recv().await {
//!     match update {
//!         HlPerpUpdate::Price(p) => signal_integrator.on_hl_perp_price(p),
//!         HlPerpUpdate::Trade(t) => signal_integrator.on_hl_perp_trade(t),
//!     }
//! }
//! ```

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tracing::{debug, error, info, trace, warn};

/// Price update from HL perp (from BBO subscription).
#[derive(Debug, Clone)]
pub struct HlPerpPriceUpdate {
    /// Coin identifier (e.g., "HYPE", "BTC") — distinguishes asset perp from BTC feed.
    pub coin: String,
    /// Timestamp in milliseconds (exchange time).
    pub timestamp_ms: u64,
    /// Mid price (best_bid + best_ask) / 2.
    pub mid_price: f64,
    /// Best bid price.
    pub best_bid: f64,
    /// Best ask price.
    pub best_ask: f64,
    /// Spread in basis points.
    pub spread_bps: f64,
}

/// Trade update from HL perp (from Trades subscription).
#[derive(Debug, Clone)]
pub struct HlPerpTradeUpdate {
    /// Coin identifier (e.g., "HYPE", "BTC") — distinguishes asset perp from BTC feed.
    pub coin: String,
    /// Timestamp in milliseconds (trade time).
    pub timestamp_ms: u64,
    /// Trade price.
    pub price: f64,
    /// Trade size (always positive).
    pub size: f64,
    /// Trade side: true = buy aggressor, false = sell aggressor.
    pub is_buy: bool,
    /// Trade ID.
    pub tid: u64,
}

/// Combined update from HL perp feed — either price or trade.
#[derive(Debug, Clone)]
pub enum HlPerpUpdate {
    /// BBO price update.
    Price(HlPerpPriceUpdate),
    /// Trade update.
    Trade(HlPerpTradeUpdate),
}

/// Configuration for HL perp feed.
#[derive(Debug, Clone)]
pub struct HlPerpFeedConfig {
    /// Asset to subscribe to (e.g., "HYPE").
    pub coin: String,
    /// WebSocket URL.
    pub ws_url: String,
    /// Reconnect delay on disconnect.
    pub reconnect_delay_ms: u64,
    /// Maximum reconnect attempts (0 = unlimited).
    pub max_reconnect_attempts: u32,
    /// Whether to subscribe to trade stream.
    pub enable_trades: bool,
}

impl Default for HlPerpFeedConfig {
    fn default() -> Self {
        Self {
            coin: "HYPE".to_string(),
            ws_url: "wss://api.hyperliquid.xyz/ws".to_string(),
            reconnect_delay_ms: 1000,
            max_reconnect_attempts: 0, // Unlimited — kill switch is the safety net
            enable_trades: true,
        }
    }
}

impl HlPerpFeedConfig {
    /// Create config for a specific asset.
    pub fn for_asset(coin: &str) -> Self {
        Self {
            coin: coin.to_uppercase(),
            ..Default::default()
        }
    }

    /// Create config with testnet URL.
    pub fn testnet(coin: &str) -> Self {
        Self {
            coin: coin.to_uppercase(),
            ws_url: "wss://api.hyperliquid-testnet.xyz/ws".to_string(),
            ..Default::default()
        }
    }
}

/// HL BBO message (channel: "bbo").
#[derive(Debug, Deserialize)]
struct HlBboMessage {
    #[allow(dead_code)]
    channel: String,
    data: HlBboData,
}

/// BBO data payload.
#[derive(Debug, Deserialize)]
struct HlBboData {
    #[allow(dead_code)]
    coin: String,
    time: u64,
    bbo: Vec<Option<HlOrderLevel>>,
}

/// Single order book level.
#[derive(Debug, Deserialize)]
struct HlOrderLevel {
    px: String,
    #[allow(dead_code)]
    sz: String,
    #[allow(dead_code)]
    n: u64,
}

/// HL Trades message (channel: "trades").
#[derive(Debug, Deserialize)]
struct HlTradesMessage {
    #[allow(dead_code)]
    channel: String,
    data: Vec<HlTradeData>,
}

/// Single trade data.
#[derive(Debug, Deserialize)]
struct HlTradeData {
    #[allow(dead_code)]
    coin: String,
    side: String,
    px: String,
    sz: String,
    time: u64,
    tid: u64,
}

/// Hyperliquid perpetual price feed.
pub struct HlPerpFeed {
    config: HlPerpFeedConfig,
    tx: mpsc::Sender<HlPerpUpdate>,
    reconnect_attempts: u32,
}

impl HlPerpFeed {
    /// Create a new HL perp feed.
    pub fn new(config: HlPerpFeedConfig, tx: mpsc::Sender<HlPerpUpdate>) -> Self {
        Self {
            config,
            tx,
            reconnect_attempts: 0,
        }
    }

    /// Create with default config for an asset (BBO + trades).
    pub fn for_asset(coin: &str, tx: mpsc::Sender<HlPerpUpdate>) -> Self {
        Self::new(HlPerpFeedConfig::for_asset(coin), tx)
    }

    /// Run the feed (blocking, should be spawned).
    pub async fn run(mut self) {
        info!(coin = %self.config.coin, "Starting HL perp feed");

        loop {
            match self.connect_and_stream().await {
                Ok(()) => {
                    info!(coin = %self.config.coin, "HL perp feed disconnected normally");
                    break;
                }
                Err(e) => {
                    self.reconnect_attempts += 1;
                    if self.config.max_reconnect_attempts > 0
                        && self.reconnect_attempts > self.config.max_reconnect_attempts
                    {
                        error!(
                            coin = %self.config.coin,
                            attempts = self.reconnect_attempts,
                            "HL perp feed: max reconnect attempts exceeded"
                        );
                        break;
                    }

                    warn!(
                        coin = %self.config.coin,
                        error = %e,
                        attempt = self.reconnect_attempts,
                        "HL perp feed error, reconnecting..."
                    );

                    let delay = Duration::from_millis(
                        self.config.reconnect_delay_ms * (self.reconnect_attempts as u64).min(30),
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// Connect to WebSocket and stream data.
    async fn connect_and_stream(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let url = &self.config.ws_url;
        info!(url = %url, coin = %self.config.coin, "Connecting to HL perp WebSocket...");

        let (ws_stream, response) = connect_async(url).await?;
        info!(
            status = %response.status(),
            "Connected to HL perp WebSocket"
        );

        self.reconnect_attempts = 0;
        let (mut write, mut read) = ws_stream.split();

        // Subscribe to BBO (perp = no dex parameter)
        let bbo_sub = format!(
            r#"{{"method":"subscribe","subscription":{{"type":"bbo","coin":"{}"}}}}"#,
            self.config.coin,
        );
        write
            .send(tokio_tungstenite::tungstenite::Message::Text(bbo_sub))
            .await?;
        info!(coin = %self.config.coin, "Subscribed to HL perp BBO");

        // Subscribe to trades if enabled
        if self.config.enable_trades {
            let trades_sub = format!(
                r#"{{"method":"subscribe","subscription":{{"type":"trades","coin":"{}"}}}}"#,
                self.config.coin,
            );
            write
                .send(tokio_tungstenite::tungstenite::Message::Text(trades_sub))
                .await?;
            info!(coin = %self.config.coin, "Subscribed to HL perp trades");
        }

        while let Some(msg) = read.next().await {
            match msg {
                Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                    if let Err(e) = self.process_message(&text) {
                        debug!(error = %e, "Failed to process HL perp message");
                    }
                }
                Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => {
                    info!("HL perp WebSocket closed by server");
                    break;
                }
                Ok(tokio_tungstenite::tungstenite::Message::Ping(data)) => {
                    debug!("Received ping from HL, length={}", data.len());
                }
                Ok(_) => {}
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }

        Ok(())
    }

    /// Process a WebSocket message by detecting channel type.
    fn process_message(&self, text: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Detect channel from the JSON
        // HL messages: {"channel": "bbo", "data": {...}} or {"channel": "trades", "data": [...]}
        // Also: {"channel": "subscriptionResponse", "data": {...}} for sub confirmations

        // Quick channel detection
        if text.contains("\"bbo\"")
            && text.contains("\"data\"")
            && !text.contains("subscriptionResponse")
        {
            if let Ok(msg) = serde_json::from_str::<HlBboMessage>(text) {
                return self.process_bbo(&msg.data);
            }
        }

        if text.contains("\"trades\"")
            && text.contains("\"data\"")
            && !text.contains("subscriptionResponse")
        {
            if let Ok(msg) = serde_json::from_str::<HlTradesMessage>(text) {
                for trade in &msg.data {
                    let _ = self.process_trade(trade);
                }
                return Ok(());
            }
        }

        // Subscription confirmations - log and ignore
        if text.contains("subscriptionResponse") {
            trace!(
                "HL perp subscription response: {}",
                &text[..text.len().min(200)]
            );
        }

        Ok(())
    }

    /// Process a BBO update.
    fn process_bbo(
        &self,
        data: &HlBboData,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // bbo[0] = bid, bbo[1] = ask
        let bid = data.bbo.first().and_then(|b| b.as_ref());
        let ask = data.bbo.get(1).and_then(|a| a.as_ref());

        let (bid_level, ask_level) = match (bid, ask) {
            (Some(b), Some(a)) => (b, a),
            _ => return Ok(()), // Incomplete BBO
        };

        let best_bid: f64 = bid_level.px.parse()?;
        let best_ask: f64 = ask_level.px.parse()?;

        if best_bid <= 0.0 || best_ask <= 0.0 || best_ask <= best_bid {
            return Ok(()); // Invalid data
        }

        let mid_price = (best_bid + best_ask) / 2.0;
        let spread_bps = (best_ask - best_bid) / mid_price * 10_000.0;

        let update = HlPerpUpdate::Price(HlPerpPriceUpdate {
            coin: self.config.coin.clone(),
            timestamp_ms: data.time,
            mid_price,
            best_bid,
            best_ask,
            spread_bps,
        });

        if self.tx.try_send(update).is_err() {
            debug!("HL perp update channel full, dropping BBO update");
        }

        Ok(())
    }

    /// Process a single trade.
    fn process_trade(
        &self,
        trade: &HlTradeData,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let price: f64 = trade.px.parse()?;
        let size: f64 = trade.sz.parse()?;

        if price <= 0.0 || size <= 0.0 {
            return Ok(());
        }

        // HL trade side: "B" = buy, "A" = sell (ask)
        let is_buy = trade.side == "B";

        let update = HlPerpUpdate::Trade(HlPerpTradeUpdate {
            coin: self.config.coin.clone(),
            timestamp_ms: trade.time,
            price,
            size,
            is_buy,
            tid: trade.tid,
        });

        if self.tx.try_send(update).is_err() {
            trace!("HL perp update channel full, dropping trade update");
        }

        Ok(())
    }
}

/// Check if an asset should use the HL perp lead-lag signal.
///
/// Returns true for HIP-3 DEX assets where the HL perp for the same
/// underlying has meaningful price discovery. Returns false for:
/// - Assets already being traded as perps (would be self-referential)
/// - Assets without an HL perp market
pub fn should_use_hl_perp_lead(asset: &str, is_hip3: bool) -> bool {
    if !is_hip3 {
        return false; // Only HIP-3 spot assets benefit from perp lead-lag
    }

    // Strip DEX prefix (e.g., "hyna:HYPE" → "HYPE")
    let base = asset.split(':').next_back().unwrap_or(asset);

    // Assets with liquid HL perp markets
    matches!(
        base.to_uppercase().as_str(),
        "HYPE" | "PURR" | "JEFF" | "FARM" | "PIP" | "CATBAL" | "RAGE"
    )
}

/// Resolve the HL perp coin name from an asset string.
///
/// Strips DEX prefix and normalizes to uppercase.
pub fn resolve_hl_perp_coin(asset: &str) -> String {
    let base = asset.split(':').next_back().unwrap_or(asset);
    base.to_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_use_hl_perp_lead() {
        // HIP-3 assets with perp → should use
        assert!(should_use_hl_perp_lead("HYPE", true));
        assert!(should_use_hl_perp_lead("hyna:HYPE", true));
        assert!(should_use_hl_perp_lead("PURR", true));

        // Non-HIP-3 → should not use (self-referential)
        assert!(!should_use_hl_perp_lead("HYPE", false));
        assert!(!should_use_hl_perp_lead("BTC", false));

        // Unknown HIP-3 asset → should not use (no perp market)
        assert!(!should_use_hl_perp_lead("RANDOMTOKEN", true));
    }

    #[test]
    fn test_resolve_hl_perp_coin() {
        assert_eq!(resolve_hl_perp_coin("HYPE"), "HYPE");
        assert_eq!(resolve_hl_perp_coin("hyna:HYPE"), "HYPE");
        assert_eq!(resolve_hl_perp_coin("hyna:hype"), "HYPE");
    }

    #[test]
    fn test_config_defaults() {
        let config = HlPerpFeedConfig::default();
        assert_eq!(config.coin, "HYPE");
        assert_eq!(config.max_reconnect_attempts, 0);
        assert!(config.enable_trades);
    }

    #[test]
    fn test_config_for_asset() {
        let config = HlPerpFeedConfig::for_asset("purr");
        assert_eq!(config.coin, "PURR");
    }

    #[test]
    fn test_config_testnet() {
        let config = HlPerpFeedConfig::testnet("HYPE");
        assert!(config.ws_url.contains("testnet"));
    }

    #[test]
    fn test_bbo_processing() {
        let (tx, mut rx) = mpsc::channel(10);
        let feed = HlPerpFeed::for_asset("HYPE", tx);

        let bbo_data = HlBboData {
            coin: "HYPE".to_string(),
            time: 1700000000000,
            bbo: vec![
                Some(HlOrderLevel {
                    px: "32.50".to_string(),
                    sz: "100.0".to_string(),
                    n: 5,
                }),
                Some(HlOrderLevel {
                    px: "32.52".to_string(),
                    sz: "80.0".to_string(),
                    n: 3,
                }),
            ],
        };

        feed.process_bbo(&bbo_data).unwrap();

        let update = rx.try_recv().unwrap();
        match update {
            HlPerpUpdate::Price(p) => {
                assert_eq!(p.coin, "HYPE");
                assert_eq!(p.timestamp_ms, 1700000000000);
                assert!((p.mid_price - 32.51).abs() < 0.001);
                assert!((p.best_bid - 32.50).abs() < 0.001);
                assert!((p.best_ask - 32.52).abs() < 0.001);
                assert!(p.spread_bps > 0.0);
            }
            _ => panic!("Expected Price update"),
        }
    }

    #[test]
    fn test_trade_processing() {
        let (tx, mut rx) = mpsc::channel(10);
        let feed = HlPerpFeed::for_asset("HYPE", tx);

        let trade = HlTradeData {
            coin: "HYPE".to_string(),
            side: "B".to_string(),
            px: "32.50".to_string(),
            sz: "10.0".to_string(),
            time: 1700000000000,
            tid: 12345,
        };

        feed.process_trade(&trade).unwrap();

        let update = rx.try_recv().unwrap();
        match update {
            HlPerpUpdate::Trade(t) => {
                assert_eq!(t.coin, "HYPE");
                assert!(t.is_buy);
                assert!((t.price - 32.50).abs() < 0.001);
                assert!((t.size - 10.0).abs() < 0.001);
                assert_eq!(t.tid, 12345);
            }
            _ => panic!("Expected Trade update"),
        }
    }

    #[test]
    fn test_invalid_bbo_rejected() {
        let (tx, mut rx) = mpsc::channel(10);
        let feed = HlPerpFeed::for_asset("HYPE", tx);

        // Ask <= Bid → should be rejected
        let bbo_data = HlBboData {
            coin: "HYPE".to_string(),
            time: 1700000000000,
            bbo: vec![
                Some(HlOrderLevel {
                    px: "32.52".to_string(),
                    sz: "100.0".to_string(),
                    n: 5,
                }),
                Some(HlOrderLevel {
                    px: "32.50".to_string(),
                    sz: "80.0".to_string(),
                    n: 3,
                }),
            ],
        };

        feed.process_bbo(&bbo_data).unwrap();
        assert!(rx.try_recv().is_err(), "Invalid BBO should be dropped");
    }

    #[test]
    fn test_incomplete_bbo_rejected() {
        let (tx, mut rx) = mpsc::channel(10);
        let feed = HlPerpFeed::for_asset("HYPE", tx);

        // Only bid, no ask
        let bbo_data = HlBboData {
            coin: "HYPE".to_string(),
            time: 1700000000000,
            bbo: vec![Some(HlOrderLevel {
                px: "32.50".to_string(),
                sz: "100.0".to_string(),
                n: 5,
            })],
        };

        feed.process_bbo(&bbo_data).unwrap();
        assert!(rx.try_recv().is_err(), "Incomplete BBO should be dropped");
    }
}
