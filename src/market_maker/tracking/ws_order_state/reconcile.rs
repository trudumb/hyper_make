//! Reconciliation logic for WebSocket order state manager.
//!
//! This module provides types and utilities for synchronization between
//! local order state and exchange state.

use crate::market_maker::tracking::order_manager::Side;

/// Exchange order information for reconciliation.
#[derive(Debug, Clone)]
pub struct ExchangeOrderInfo {
    pub oid: u64,
    pub cloid: Option<String>,
    pub coin: String,
    pub side: String,
    pub price: f64,
    pub size: f64,
    pub orig_size: f64,
}

impl ExchangeOrderInfo {
    /// Parse side string to Side enum.
    pub fn side_enum(&self) -> Side {
        if self.side == "B" || self.side.to_lowercase() == "buy" {
            Side::Buy
        } else {
            Side::Sell
        }
    }

    /// Calculate filled amount.
    pub fn filled_amount(&self) -> f64 {
        self.orig_size - self.size
    }
}
