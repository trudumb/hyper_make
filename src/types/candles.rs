//! Candlestick/OHLCV data types.

use serde::{Deserialize, Serialize};

/// Candlestick data with OHLCV values.
/// Consolidates `CandlesSnapshotResponse` (info) and `CandleData` (ws).
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Candle {
    /// Opening time of the candle
    #[serde(rename = "t")]
    pub time_open: u64,
    /// Closing time of the candle
    #[serde(rename = "T")]
    pub time_close: u64,
    /// Symbol/coin
    #[serde(rename = "s")]
    pub coin: String,
    /// Interval (e.g., "1m", "1h")
    #[serde(rename = "i")]
    pub interval: String,
    /// Open price
    #[serde(rename = "o")]
    pub open: String,
    /// Close price
    #[serde(rename = "c")]
    pub close: String,
    /// High price
    #[serde(rename = "h")]
    pub high: String,
    /// Low price
    #[serde(rename = "l")]
    pub low: String,
    /// Volume
    #[serde(rename = "v")]
    pub volume: String,
    /// Number of trades
    #[serde(rename = "n")]
    pub num_trades: u64,
}

// Type alias for backwards compatibility (used by WebSocket)
pub type CandleData = Candle;
