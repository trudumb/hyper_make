//! Binance WebSocket price feed for lead-lag signal.
//!
//! Streams BTC perpetual mid prices from Binance Futures to detect
//! price movements before they propagate to Hyperliquid.
//!
//! # Architecture
//!
//! ```text
//! Binance WS (wss://fstream.binance.com)
//!     │
//!     ├── @bookTicker stream (best bid/ask, ~10ms updates)
//!     │
//!     ▼
//! BinanceFeed
//!     │
//!     ├── Computes mid price
//!     ├── Sends to LagAnalyzer via channel
//!     │
//!     ▼
//! Quote Engine (skews quotes based on lag signal)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let (tx, rx) = tokio::sync::mpsc::channel(1000);
//! let feed = BinanceFeed::new("BTCUSDT", tx);
//! tokio::spawn(feed.run());
//!
//! // In quote engine, receive price updates
//! while let Some(update) = rx.recv().await {
//!     lag_analyzer.add_signal(update.timestamp_ms, update.mid_price);
//! }
//! ```

use futures_util::StreamExt;
use serde::Deserialize;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tracing::{debug, error, info, trace, warn};

/// Price update from Binance (from @bookTicker stream).
#[derive(Debug, Clone, Copy)]
pub struct BinancePriceUpdate {
    /// Timestamp in milliseconds (exchange time).
    pub timestamp_ms: i64,
    /// Mid price (best_bid + best_ask) / 2.
    pub mid_price: f64,
    /// Best bid price.
    pub best_bid: f64,
    /// Best ask price.
    pub best_ask: f64,
    /// Spread in basis points.
    pub spread_bps: f64,
}

/// Trade update from Binance (from @aggTrade stream).
///
/// Aggregate trades are compressed trades that have the same price and were
/// within 100ms of each other.
#[derive(Debug, Clone, Copy)]
pub struct BinanceTradeUpdate {
    /// Timestamp in milliseconds (trade time, not event time).
    pub timestamp_ms: i64,
    /// Trade price.
    pub price: f64,
    /// Trade quantity (always positive).
    pub quantity: f64,
    /// True if the trade was a sell aggressor (buyer is maker).
    /// Note: Binance "m" field means "is buyer maker", so true = sell aggressor.
    pub is_buyer_maker: bool,
    /// Aggregate trade ID.
    pub trade_id: u64,
}

/// Combined update from Binance feed - either price or trade.
#[derive(Debug, Clone)]
pub enum BinanceUpdate {
    /// Price update from @bookTicker.
    Price(BinancePriceUpdate),
    /// Trade update from @aggTrade.
    Trade(BinanceTradeUpdate),
}

/// Binance book ticker message (best bid/ask).
#[derive(Debug, Deserialize)]
struct BookTickerMessage {
    /// Event type.
    #[allow(dead_code)]
    e: String,
    /// Event time (ms).
    #[serde(rename = "E")]
    event_time: i64,
    /// Symbol.
    #[allow(dead_code)]
    s: String,
    /// Best bid price.
    b: String,
    /// Best bid quantity.
    #[serde(rename = "B")]
    _bid_qty: String,
    /// Best ask price.
    a: String,
    /// Best ask quantity.
    #[serde(rename = "A")]
    _ask_qty: String,
}

/// Binance aggregate trade message.
#[derive(Debug, Deserialize)]
struct AggTradeMessage {
    /// Event type.
    #[allow(dead_code)]
    e: String,
    /// Event time (ms).
    #[serde(rename = "E")]
    _event_time: i64,
    /// Symbol.
    #[allow(dead_code)]
    s: String,
    /// Aggregate trade ID.
    a: u64,
    /// Price.
    p: String,
    /// Quantity.
    q: String,
    /// First trade ID.
    #[allow(dead_code)]
    f: u64,
    /// Last trade ID.
    #[allow(dead_code)]
    l: u64,
    /// Trade time (ms).
    #[serde(rename = "T")]
    trade_time: i64,
    /// Is buyer maker (true = sell aggressor).
    m: bool,
}

/// Configuration for Binance feed.
#[derive(Debug, Clone)]
pub struct BinanceFeedConfig {
    /// Symbol to subscribe to (e.g., "btcusdt").
    pub symbol: String,
    /// WebSocket URL (defaults to Binance Futures).
    pub ws_url: String,
    /// Reconnect delay on disconnect.
    pub reconnect_delay: Duration,
    /// Maximum reconnect attempts before giving up.
    pub max_reconnect_attempts: u32,
    /// Stale threshold - if no update for this long, consider feed stale.
    pub stale_threshold: Duration,
    /// Whether to subscribe to @aggTrade stream (for flow analysis).
    pub enable_trades: bool,
}

impl Default for BinanceFeedConfig {
    fn default() -> Self {
        Self {
            symbol: "btcusdt".to_string(),
            ws_url: "wss://fstream.binance.com/ws".to_string(),
            reconnect_delay: Duration::from_secs(1),
            max_reconnect_attempts: 10,
            stale_threshold: Duration::from_secs(5),
            enable_trades: true, // Enable trade stream by default
        }
    }
}

impl BinanceFeedConfig {
    /// Create config for a specific symbol.
    pub fn for_symbol(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_lowercase(),
            ..Default::default()
        }
    }

    /// Create config for price-only mode (no trades).
    pub fn price_only(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_lowercase(),
            enable_trades: false,
            ..Default::default()
        }
    }

    /// Get the full WebSocket URL with stream subscription.
    /// Uses combined stream endpoint when trades are enabled.
    pub fn stream_url(&self) -> String {
        if self.enable_trades {
            // Combined streams endpoint
            format!(
                "{}/stream?streams={}@bookTicker/{}@aggTrade",
                self.ws_url.trim_end_matches("/ws"),
                self.symbol,
                self.symbol
            )
        } else {
            // Single stream endpoint (legacy behavior)
            format!("{}/{}@bookTicker", self.ws_url, self.symbol)
        }
    }
}

/// Resolve a Hyperliquid asset name to its Binance Futures symbol.
///
/// Returns `Some("btcusdt")` for assets with a meaningful Binance equivalent,
/// `None` for Hyperliquid-native tokens (HYPE, PURR, etc.) where cross-venue
/// signals would be pure noise.
///
/// # Priority
/// 1. Explicit `override_symbol` (from CLI `--binance-symbol`)
/// 2. Known asset mapping (BTC→btcusdt, ETH→ethusdt, etc.)
/// 3. `None` → caller should disable Binance feed
pub fn resolve_binance_symbol(asset: &str, override_symbol: Option<&str>) -> Option<String> {
    // Explicit override takes priority
    if let Some(sym) = override_symbol {
        return Some(sym.to_lowercase());
    }

    // Strip DEX prefix (e.g., "hyna:BTC" → "BTC")
    let base_asset = asset.split(':').next_back().unwrap_or(asset);

    // Map known assets to their Binance Futures USDT perpetual symbols.
    // Only include assets with genuine cross-venue price discovery on Binance.
    match base_asset.to_uppercase().as_str() {
        // Major pairs — strong lead-lag signal
        "BTC" => Some("btcusdt".to_string()),
        "ETH" => Some("ethusdt".to_string()),
        // Large-cap alts — moderate lead-lag signal
        "SOL" => Some("solusdt".to_string()),
        "DOGE" => Some("dogeusdt".to_string()),
        "AVAX" => Some("avaxusdt".to_string()),
        "LINK" => Some("linkusdt".to_string()),
        "MATIC" | "POL" => Some("maticusdt".to_string()),
        "ARB" => Some("arbusdt".to_string()),
        "OP" => Some("opusdt".to_string()),
        "SUI" => Some("suiusdt".to_string()),
        "APT" => Some("aptusdt".to_string()),
        "WIF" => Some("wifusdt".to_string()),
        "PEPE" => Some("pepeusdt".to_string()),
        "SEI" => Some("seiusdt".to_string()),
        "TIA" => Some("tiausdt".to_string()),
        "INJ" => Some("injusdt".to_string()),
        "JUP" => Some("jupusdt".to_string()),
        "RENDER" | "RNDR" => Some("renderusdt".to_string()),
        "FET" => Some("fetusdt".to_string()),
        "STX" => Some("stxusdt".to_string()),
        "NEAR" => Some("nearusdt".to_string()),
        "FIL" => Some("filusdt".to_string()),
        "ATOM" => Some("atomusdt".to_string()),
        "DOT" => Some("dotusdt".to_string()),
        "ADA" => Some("adausdt".to_string()),
        "XRP" => Some("xrpusdt".to_string()),
        "LTC" => Some("ltcusdt".to_string()),
        "BCH" => Some("bchusdt".to_string()),
        "UNI" => Some("uniusdt".to_string()),
        "AAVE" => Some("aaveusdt".to_string()),
        "MKR" => Some("mkrusdt".to_string()),
        "CRV" => Some("crvusdt".to_string()),
        "LDO" => Some("ldousdt".to_string()),
        "BONK" => Some("bonkusdt".to_string()),
        // Hyperliquid-native tokens — NO Binance equivalent
        // HYPE, PURR, JEFF, and other HL-native tokens return None.
        // Feeding BTC data for these would inject noise into the lead-lag estimator.
        _ => None,
    }
}

/// Binance price feed for lead-lag signal.
pub struct BinanceFeed {
    config: BinanceFeedConfig,
    tx: mpsc::Sender<BinanceUpdate>,
    reconnect_attempts: u32,
}

impl BinanceFeed {
    /// Create a new Binance feed with combined updates.
    pub fn new(config: BinanceFeedConfig, tx: mpsc::Sender<BinanceUpdate>) -> Self {
        Self {
            config,
            tx,
            reconnect_attempts: 0,
        }
    }

    /// Create with default config for a symbol (includes trades).
    pub fn for_symbol(symbol: &str, tx: mpsc::Sender<BinanceUpdate>) -> Self {
        Self::new(BinanceFeedConfig::for_symbol(symbol), tx)
    }

    /// Create with price-only config for a symbol.
    pub fn for_symbol_price_only(symbol: &str, tx: mpsc::Sender<BinanceUpdate>) -> Self {
        Self::new(BinanceFeedConfig::price_only(symbol), tx)
    }

    /// Run the feed (blocking, should be spawned).
    pub async fn run(mut self) {
        loop {
            match self.connect_and_stream().await {
                Ok(()) => {
                    info!(symbol = %self.config.symbol, "Binance feed disconnected normally");
                    break;
                }
                Err(e) => {
                    self.reconnect_attempts += 1;
                    if self.reconnect_attempts > self.config.max_reconnect_attempts {
                        error!(
                            symbol = %self.config.symbol,
                            attempts = self.reconnect_attempts,
                            "Max reconnect attempts exceeded, giving up"
                        );
                        break;
                    }

                    warn!(
                        symbol = %self.config.symbol,
                        error = %e,
                        attempt = self.reconnect_attempts,
                        "Binance feed error, reconnecting..."
                    );

                    tokio::time::sleep(self.config.reconnect_delay).await;
                }
            }
        }
    }

    /// Connect to WebSocket and stream prices.
    async fn connect_and_stream(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let url = self.config.stream_url();
        info!(url = %url, "Connecting to Binance WebSocket...");

        let (ws_stream, response) = connect_async(&url).await?;
        info!(
            status = %response.status(),
            "Connected to Binance WebSocket"
        );

        // Reset reconnect counter on successful connection
        self.reconnect_attempts = 0;

        let (_, mut read) = ws_stream.split();

        while let Some(msg) = read.next().await {
            match msg {
                Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                    if let Err(e) = self.process_message(&text).await {
                        debug!(error = %e, "Failed to process Binance message");
                    }
                }
                Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => {
                    info!("Binance WebSocket closed by server");
                    break;
                }
                Ok(tokio_tungstenite::tungstenite::Message::Ping(data)) => {
                    debug!("Received ping from Binance, length={}", data.len());
                }
                Ok(_) => {
                    // Ignore other message types
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }

        Ok(())
    }

    /// Process a WebSocket message.
    async fn process_message(
        &self,
        text: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Try to detect message type from the JSON
        // Combined streams wrap messages in {"stream": "...", "data": {...}}
        if let Ok(wrapper) = serde_json::from_str::<CombinedStreamWrapper>(text) {
            // Combined stream format
            if wrapper.stream.ends_with("@bookTicker") {
                self.process_book_ticker(&wrapper.data)?;
            } else if wrapper.stream.ends_with("@aggTrade") {
                self.process_agg_trade(&wrapper.data)?;
            }
        } else {
            // Single stream format (legacy - direct bookTicker)
            self.process_book_ticker(text)?;
        }

        Ok(())
    }

    /// Process a book ticker message.
    fn process_book_ticker(&self, data: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let msg: BookTickerMessage = serde_json::from_str(data)?;

        // Parse prices
        let best_bid: f64 = msg.b.parse()?;
        let best_ask: f64 = msg.a.parse()?;

        if best_bid <= 0.0 || best_ask <= 0.0 || best_ask <= best_bid {
            return Ok(()); // Invalid data
        }

        let mid_price = (best_bid + best_ask) / 2.0;
        let spread_bps = (best_ask - best_bid) / mid_price * 10_000.0;

        let update = BinanceUpdate::Price(BinancePriceUpdate {
            timestamp_ms: msg.event_time,
            mid_price,
            best_bid,
            best_ask,
            spread_bps,
        });

        // Send to channel (non-blocking, drop if full)
        if self.tx.try_send(update).is_err() {
            debug!("Binance update channel full, dropping price update");
        }

        Ok(())
    }

    /// Process an aggregate trade message.
    fn process_agg_trade(&self, data: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let msg: AggTradeMessage = serde_json::from_str(data)?;

        // Parse trade data
        let price: f64 = msg.p.parse()?;
        let quantity: f64 = msg.q.parse()?;

        if price <= 0.0 || quantity <= 0.0 {
            return Ok(()); // Invalid data
        }

        let update = BinanceUpdate::Trade(BinanceTradeUpdate {
            timestamp_ms: msg.trade_time,
            price,
            quantity,
            is_buyer_maker: msg.m,
            trade_id: msg.a,
        });

        // Send to channel (non-blocking, drop if full)
        if self.tx.try_send(update).is_err() {
            trace!("Binance update channel full, dropping trade update");
        }

        Ok(())
    }
}

/// Wrapper for combined stream messages.
#[derive(Debug, Deserialize)]
struct CombinedStreamWrapper {
    /// Stream name (e.g., "btcusdt@bookTicker" or "btcusdt@aggTrade").
    stream: String,
    /// Raw JSON data for the stream-specific message.
    #[serde(deserialize_with = "deserialize_raw_value")]
    data: String,
}

/// Deserialize raw JSON value as a string for later parsing.
fn deserialize_raw_value<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(value.to_string())
}

/// Lead-lag signal derived from Binance feed.
///
/// This struct tracks the relationship between Binance and Hyperliquid prices
/// and provides actionable signals for quote skewing.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeadLagSignal {
    /// Latest Binance mid price.
    pub binance_mid: f64,
    /// Latest Hyperliquid mid price.
    pub hl_mid: f64,
    /// Price difference (Binance - HL) in basis points.
    pub diff_bps: f64,
    /// Optimal lag in milliseconds (positive = Binance leads).
    pub optimal_lag_ms: i64,
    /// Mutual information at optimal lag (higher = stronger signal).
    pub mi_bits: f64,
    /// Whether the signal is actionable (enough data, significant MI, stable).
    pub is_actionable: bool,
    /// Suggested skew direction: +1 = skew asks wider (bullish signal), -1 = skew bids wider.
    pub skew_direction: i8,
    /// Suggested skew magnitude in basis points.
    pub skew_magnitude_bps: f64,
    /// Stability confidence [0, 1] from LeadLagStabilityGate.
    /// Higher = more consistent lag readings, more trustworthy signal.
    pub stability_confidence: f64,
}

impl LeadLagSignal {
    /// Compute signal from current state.
    ///
    /// Now requires stability_confidence from LeadLagStabilityGate.
    /// Signals are only actionable when lag is stable and causal.
    pub fn compute(
        binance_mid: f64,
        hl_mid: f64,
        optimal_lag_ms: i64,
        mi_bits: f64,
        min_mi_threshold: f64,
        stability_confidence: f64,
    ) -> Self {
        if binance_mid <= 0.0 || hl_mid <= 0.0 {
            return Self::default();
        }

        let diff_bps = (binance_mid - hl_mid) / hl_mid * 10_000.0;

        // Signal is actionable if:
        // 1. Binance leads (positive lag - implementation note: positive lag means
        //    we look at past signal to predict current target, i.e., signal leads)
        // 2. MI is above threshold
        // 3. Price difference is significant (> 1 bps)
        // 4. NEW: Stability confidence is high enough (> 0.3)
        let is_actionable = optimal_lag_ms > 25 // Binance leads by at least 25ms
            && mi_bits > min_mi_threshold
            && diff_bps.abs() > 1.0
            && stability_confidence > 0.3; // NEW: require stability

        // Skew direction based on price difference
        // If Binance is higher, HL will likely move up -> skew asks wider (don't sell cheap)
        let skew_direction = if diff_bps > 1.0 {
            1 // Bullish
        } else if diff_bps < -1.0 {
            -1 // Bearish
        } else {
            0 // Neutral
        };

        // Skew magnitude scales with MI confidence, price difference, AND stability
        // Capped at 5 bps to avoid over-skewing
        let skew_magnitude_bps = if is_actionable {
            // Scale by stability confidence to reduce magnitude when unstable
            (diff_bps.abs() * mi_bits * stability_confidence).min(5.0)
        } else {
            0.0
        };

        Self {
            binance_mid,
            hl_mid,
            diff_bps,
            optimal_lag_ms,
            mi_bits,
            is_actionable,
            skew_direction,
            skew_magnitude_bps,
            stability_confidence,
        }
    }

    /// Legacy compute without stability (for backward compatibility).
    /// Assumes stability_confidence = 1.0 if not provided.
    pub fn compute_legacy(
        binance_mid: f64,
        hl_mid: f64,
        optimal_lag_ms: i64,
        mi_bits: f64,
        min_mi_threshold: f64,
    ) -> Self {
        Self::compute(
            binance_mid,
            hl_mid,
            optimal_lag_ms,
            mi_bits,
            min_mi_threshold,
            1.0, // Assume stable
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binance_feed_config_default() {
        let config = BinanceFeedConfig::default();
        assert_eq!(config.symbol, "btcusdt");
        assert!(config.ws_url.contains("binance.com"));
    }

    #[test]
    fn test_binance_feed_config_for_symbol() {
        let config = BinanceFeedConfig::for_symbol("ETHUSDT");
        assert_eq!(config.symbol, "ethusdt"); // Lowercased
    }

    #[test]
    fn test_stream_url_combined() {
        let config = BinanceFeedConfig::for_symbol("btcusdt");
        let url = config.stream_url();
        // Combined stream endpoint with both bookTicker and aggTrade
        assert!(url.contains("btcusdt@bookTicker"));
        assert!(url.contains("btcusdt@aggTrade"));
        assert!(url.contains("/stream?streams="));
    }

    #[test]
    fn test_stream_url_price_only() {
        let config = BinanceFeedConfig::price_only("btcusdt");
        let url = config.stream_url();
        // Single stream endpoint (legacy)
        assert!(url.contains("btcusdt@bookTicker"));
        assert!(!url.contains("aggTrade"));
    }

    #[test]
    fn test_binance_trade_update_parsing() {
        // Verify BinanceTradeUpdate struct fields
        let trade = BinanceTradeUpdate {
            timestamp_ms: 1704067200000,
            price: 50000.0,
            quantity: 0.5,
            is_buyer_maker: true, // sell aggressor
            trade_id: 12345,
        };
        assert_eq!(trade.timestamp_ms, 1704067200000);
        assert_eq!(trade.price, 50000.0);
        assert!(trade.is_buyer_maker); // true = sell aggressor
    }

    #[test]
    fn test_binance_update_enum() {
        // Price update
        let price_update = BinanceUpdate::Price(BinancePriceUpdate {
            timestamp_ms: 1704067200000,
            mid_price: 50000.0,
            best_bid: 49999.0,
            best_ask: 50001.0,
            spread_bps: 0.4,
        });
        assert!(matches!(price_update, BinanceUpdate::Price(_)));

        // Trade update
        let trade_update = BinanceUpdate::Trade(BinanceTradeUpdate {
            timestamp_ms: 1704067200000,
            price: 50000.0,
            quantity: 0.5,
            is_buyer_maker: false, // buy aggressor
            trade_id: 12345,
        });
        assert!(matches!(trade_update, BinanceUpdate::Trade(_)));
    }

    #[test]
    fn test_lead_lag_signal_compute_bullish() {
        // Use compute_legacy for backward-compatible tests (assumes stability=1.0)
        let signal = LeadLagSignal::compute_legacy(
            50100.0, // Binance higher
            50000.0, // HL lower
            100,     // Binance leads by 100ms (positive = signal leads)
            0.15,    // Strong MI
            0.05,    // MI threshold
        );

        assert!(signal.is_actionable);
        assert_eq!(signal.skew_direction, 1); // Bullish
        assert!(signal.diff_bps > 0.0);
        assert!(signal.skew_magnitude_bps > 0.0);
    }

    #[test]
    fn test_lead_lag_signal_compute_bearish() {
        let signal = LeadLagSignal::compute_legacy(
            49900.0, // Binance lower
            50000.0, // HL higher
            100,     // Binance leads by 100ms
            0.15,    // Strong MI
            0.05,    // MI threshold
        );

        assert!(signal.is_actionable);
        assert_eq!(signal.skew_direction, -1); // Bearish
        assert!(signal.diff_bps < 0.0);
    }

    #[test]
    fn test_lead_lag_signal_not_actionable_low_mi() {
        let signal = LeadLagSignal::compute_legacy(
            50100.0,
            50000.0,
            100,
            0.01, // Low MI
            0.05, // Threshold
        );

        assert!(!signal.is_actionable);
        assert_eq!(signal.skew_magnitude_bps, 0.0);
    }

    #[test]
    fn test_lead_lag_signal_not_actionable_hl_leads() {
        let signal = LeadLagSignal::compute_legacy(
            50100.0,
            50000.0,
            -100, // HL leads (negative lag = target leads signal)
            0.15,
            0.05,
        );

        assert!(!signal.is_actionable);
    }

    #[test]
    fn test_lead_lag_signal_with_stability() {
        // Test with low stability - should not be actionable
        let signal_low_stability = LeadLagSignal::compute(
            50100.0,
            50000.0,
            100,  // Binance leads
            0.15, // Strong MI
            0.05, // Threshold
            0.1,  // Low stability
        );
        assert!(!signal_low_stability.is_actionable);

        // Test with high stability - should be actionable
        let signal_high_stability = LeadLagSignal::compute(
            50100.0,
            50000.0,
            100,  // Binance leads
            0.15, // Strong MI
            0.05, // Threshold
            0.8,  // High stability
        );
        assert!(signal_high_stability.is_actionable);
        assert!(signal_high_stability.stability_confidence > 0.5);
    }

    #[test]
    fn test_resolve_binance_symbol_known_assets() {
        assert_eq!(resolve_binance_symbol("BTC", None), Some("btcusdt".to_string()));
        assert_eq!(resolve_binance_symbol("ETH", None), Some("ethusdt".to_string()));
        assert_eq!(resolve_binance_symbol("SOL", None), Some("solusdt".to_string()));
        assert_eq!(resolve_binance_symbol("DOGE", None), Some("dogeusdt".to_string()));
    }

    #[test]
    fn test_resolve_binance_symbol_case_insensitive() {
        // Asset names should work regardless of case
        assert_eq!(resolve_binance_symbol("btc", None), Some("btcusdt".to_string()));
        assert_eq!(resolve_binance_symbol("Eth", None), Some("ethusdt".to_string()));
    }

    #[test]
    fn test_resolve_binance_symbol_hl_native() {
        // Hyperliquid-native tokens have no Binance equivalent
        assert_eq!(resolve_binance_symbol("HYPE", None), None);
        assert_eq!(resolve_binance_symbol("PURR", None), None);
        assert_eq!(resolve_binance_symbol("JEFF", None), None);
    }

    #[test]
    fn test_resolve_binance_symbol_dex_prefix_stripped() {
        // HIP-3 DEX-prefixed assets should strip prefix
        assert_eq!(resolve_binance_symbol("hyna:BTC", None), Some("btcusdt".to_string()));
        assert_eq!(resolve_binance_symbol("hyna:ETH", None), Some("ethusdt".to_string()));
        assert_eq!(resolve_binance_symbol("hyna:HYPE", None), None);
    }

    #[test]
    fn test_resolve_binance_symbol_override() {
        // Explicit override takes priority over mapping
        assert_eq!(
            resolve_binance_symbol("HYPE", Some("btcusdt")),
            Some("btcusdt".to_string()),
        );
        // Override lowercases
        assert_eq!(
            resolve_binance_symbol("BTC", Some("ETHUSDT")),
            Some("ethusdt".to_string()),
        );
    }
}
