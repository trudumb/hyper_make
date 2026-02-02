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
use tracing::{debug, error, info, warn};

/// Price update from Binance.
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
}

impl Default for BinanceFeedConfig {
    fn default() -> Self {
        Self {
            symbol: "btcusdt".to_string(),
            ws_url: "wss://fstream.binance.com/ws".to_string(),
            reconnect_delay: Duration::from_secs(1),
            max_reconnect_attempts: 10,
            stale_threshold: Duration::from_secs(5),
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

    /// Get the full WebSocket URL with stream subscription.
    pub fn stream_url(&self) -> String {
        format!("{}/{}@bookTicker", self.ws_url, self.symbol)
    }
}

/// Binance price feed for lead-lag signal.
pub struct BinanceFeed {
    config: BinanceFeedConfig,
    tx: mpsc::Sender<BinancePriceUpdate>,
    reconnect_attempts: u32,
}

impl BinanceFeed {
    /// Create a new Binance feed.
    pub fn new(config: BinanceFeedConfig, tx: mpsc::Sender<BinancePriceUpdate>) -> Self {
        Self {
            config,
            tx,
            reconnect_attempts: 0,
        }
    }

    /// Create with default config for a symbol.
    pub fn for_symbol(symbol: &str, tx: mpsc::Sender<BinancePriceUpdate>) -> Self {
        Self::new(BinanceFeedConfig::for_symbol(symbol), tx)
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
        let msg: BookTickerMessage = serde_json::from_str(text)?;

        // Parse prices
        let best_bid: f64 = msg.b.parse()?;
        let best_ask: f64 = msg.a.parse()?;

        if best_bid <= 0.0 || best_ask <= 0.0 || best_ask <= best_bid {
            return Ok(()); // Invalid data
        }

        let mid_price = (best_bid + best_ask) / 2.0;
        let spread_bps = (best_ask - best_bid) / mid_price * 10_000.0;

        let update = BinancePriceUpdate {
            timestamp_ms: msg.event_time,
            mid_price,
            best_bid,
            best_ask,
            spread_bps,
        };

        // Send to channel (non-blocking, drop if full)
        if self.tx.try_send(update).is_err() {
            debug!("Binance price channel full, dropping update");
        }

        Ok(())
    }
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
    /// Optimal lag in milliseconds (negative = Binance leads).
    pub optimal_lag_ms: i64,
    /// Mutual information at optimal lag (higher = stronger signal).
    pub mi_bits: f64,
    /// Whether the signal is actionable (enough data, significant MI).
    pub is_actionable: bool,
    /// Suggested skew direction: +1 = skew asks wider (bullish signal), -1 = skew bids wider.
    pub skew_direction: i8,
    /// Suggested skew magnitude in basis points.
    pub skew_magnitude_bps: f64,
}

impl LeadLagSignal {
    /// Compute signal from current state.
    pub fn compute(
        binance_mid: f64,
        hl_mid: f64,
        optimal_lag_ms: i64,
        mi_bits: f64,
        min_mi_threshold: f64,
    ) -> Self {
        if binance_mid <= 0.0 || hl_mid <= 0.0 {
            return Self::default();
        }

        let diff_bps = (binance_mid - hl_mid) / hl_mid * 10_000.0;

        // Signal is actionable if:
        // 1. Binance leads (negative lag)
        // 2. MI is above threshold
        // 3. Price difference is significant (> 1 bps)
        let is_actionable = optimal_lag_ms < -25 // Binance leads by at least 25ms
            && mi_bits > min_mi_threshold
            && diff_bps.abs() > 1.0;

        // Skew direction based on price difference
        // If Binance is higher, HL will likely move up -> skew asks wider (don't sell cheap)
        let skew_direction = if diff_bps > 1.0 {
            1 // Bullish
        } else if diff_bps < -1.0 {
            -1 // Bearish
        } else {
            0 // Neutral
        };

        // Skew magnitude scales with MI confidence and price difference
        // Capped at 5 bps to avoid over-skewing
        let skew_magnitude_bps = if is_actionable {
            (diff_bps.abs() * mi_bits).min(5.0)
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
        }
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
    fn test_stream_url() {
        let config = BinanceFeedConfig::for_symbol("btcusdt");
        let url = config.stream_url();
        assert!(url.contains("btcusdt@bookTicker"));
    }

    #[test]
    fn test_lead_lag_signal_compute_bullish() {
        let signal = LeadLagSignal::compute(
            50100.0, // Binance higher
            50000.0, // HL lower
            -100,    // Binance leads by 100ms
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
        let signal = LeadLagSignal::compute(
            49900.0, // Binance lower
            50000.0, // HL higher
            -100,    // Binance leads by 100ms
            0.15,    // Strong MI
            0.05,    // MI threshold
        );

        assert!(signal.is_actionable);
        assert_eq!(signal.skew_direction, -1); // Bearish
        assert!(signal.diff_bps < 0.0);
    }

    #[test]
    fn test_lead_lag_signal_not_actionable_low_mi() {
        let signal = LeadLagSignal::compute(
            50100.0,
            50000.0,
            -100,
            0.01, // Low MI
            0.05, // Threshold
        );

        assert!(!signal.is_actionable);
        assert_eq!(signal.skew_magnitude_bps, 0.0);
    }

    #[test]
    fn test_lead_lag_signal_not_actionable_hl_leads() {
        let signal = LeadLagSignal::compute(
            50100.0,
            50000.0,
            100, // HL leads (positive lag)
            0.15,
            0.05,
        );

        assert!(!signal.is_actionable);
    }
}
