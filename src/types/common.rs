//! Common shared types used across the SDK.

use serde::{Deserialize, Serialize};

/// Order book price level with quantity and order count.
/// Consolidates `Level` (info) and `BookLevel` (ws).
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct OrderBookLevel {
    /// Price at this level
    pub px: String,
    /// Total size at this level
    pub sz: String,
    /// Number of orders at this level
    pub n: u64,
}

// Type aliases for backwards compatibility
pub type Level = OrderBookLevel;
pub type BookLevel = OrderBookLevel;

/// Leverage configuration for a position.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Leverage {
    #[serde(rename = "type")]
    pub type_string: String,
    pub value: u32,
    pub raw_usd: Option<String>,
}
