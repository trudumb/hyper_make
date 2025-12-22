//! Order book types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::OrderBookLevel;

/// L2 order book snapshot data.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct L2BookData {
    pub coin: String,
    pub time: u64,
    pub levels: Vec<Vec<OrderBookLevel>>,
}

/// All mid prices across assets.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct AllMidsData {
    pub mids: HashMap<String, String>,
}

/// Best bid/offer data.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct BboData {
    pub coin: String,
    pub time: u64,
    pub bbo: Vec<Option<OrderBookLevel>>,
}
