//! Unified event types for market maker.
//!
//! These types represent the core events that flow through the market maker:
//! - Market data events (AllMids, Trades, L2Book)
//! - Fill events (already defined in fills module)

use std::time::Instant;

/// Market data event types.
#[derive(Debug, Clone)]
pub enum MarketDataType {
    /// All mid prices update
    AllMids,
    /// Trade feed update
    Trades,
    /// L2 order book update
    L2Book,
}

/// Unified market data event.
///
/// Represents a single market data update that triggers processing across
/// multiple consumers. This abstraction allows for:
/// - Consistent event handling across different data types
/// - Priority-based consumer ordering
/// - Future event replay/sourcing capabilities
#[derive(Debug, Clone)]
pub struct MarketDataEvent {
    /// Type of market data
    pub event_type: MarketDataType,
    /// Timestamp when event was received
    pub received_at: Instant,
    /// Asset symbol this event relates to
    pub asset: String,
    /// Current mid price (if available)
    pub mid_price: Option<f64>,
}

impl MarketDataEvent {
    /// Create a new market data event.
    pub fn new(event_type: MarketDataType, asset: String, mid_price: Option<f64>) -> Self {
        Self {
            event_type,
            received_at: Instant::now(),
            asset,
            mid_price,
        }
    }

    /// Create an AllMids event.
    pub fn all_mids(asset: String, mid_price: f64) -> Self {
        Self::new(MarketDataType::AllMids, asset, Some(mid_price))
    }

    /// Create a Trades event.
    pub fn trades(asset: String) -> Self {
        Self::new(MarketDataType::Trades, asset, None)
    }

    /// Create an L2Book event.
    pub fn l2_book(asset: String) -> Self {
        Self::new(MarketDataType::L2Book, asset, None)
    }

    /// Time since event was received.
    pub fn age(&self) -> std::time::Duration {
        self.received_at.elapsed()
    }
}

/// Parsed trade data from WebSocket.
#[derive(Debug, Clone)]
pub struct ParsedTrade {
    /// Trade price
    pub price: f64,
    /// Trade size
    pub size: f64,
    /// True if buyer was aggressor
    pub is_buy_aggressor: bool,
    /// Trade timestamp (milliseconds)
    pub timestamp_ms: u64,
}

/// Parsed L2 book level.
#[derive(Debug, Clone)]
pub struct L2Level {
    /// Price level
    pub price: f64,
    /// Size at this level
    pub size: f64,
}

/// Parsed L2 book data.
#[derive(Debug, Clone)]
pub struct ParsedL2Book {
    /// Bid levels (best first)
    pub bids: Vec<L2Level>,
    /// Ask levels (best first)
    pub asks: Vec<L2Level>,
}

impl ParsedL2Book {
    /// Get best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price from book.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Check if book is crossed (bid >= ask).
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => bid >= ask,
            _ => false,
        }
    }

    /// Get spread in basis points.
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                let mid = (bid + ask) / 2.0;
                if mid > 0.0 {
                    Some((ask - bid) / mid * 10000.0)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Convert to tuple format for estimator.
    pub fn bids_as_tuples(&self) -> Vec<(f64, f64)> {
        self.bids.iter().map(|l| (l.price, l.size)).collect()
    }

    /// Convert to tuple format for estimator.
    pub fn asks_as_tuples(&self) -> Vec<(f64, f64)> {
        self.asks.iter().map(|l| (l.price, l.size)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data_event_creation() {
        let event = MarketDataEvent::all_mids("BTC".to_string(), 50000.0);
        assert!(matches!(event.event_type, MarketDataType::AllMids));
        assert_eq!(event.mid_price, Some(50000.0));
        assert_eq!(event.asset, "BTC");
    }

    #[test]
    fn test_parsed_l2_book() {
        let book = ParsedL2Book {
            bids: vec![
                L2Level {
                    price: 49900.0,
                    size: 1.0,
                },
                L2Level {
                    price: 49800.0,
                    size: 2.0,
                },
            ],
            asks: vec![
                L2Level {
                    price: 50100.0,
                    size: 1.0,
                },
                L2Level {
                    price: 50200.0,
                    size: 2.0,
                },
            ],
        };

        assert_eq!(book.best_bid(), Some(49900.0));
        assert_eq!(book.best_ask(), Some(50100.0));
        assert_eq!(book.mid_price(), Some(50000.0));
        assert!(!book.is_crossed());
        assert!(book.spread_bps().unwrap() > 0.0);
    }

    #[test]
    fn test_crossed_book() {
        let book = ParsedL2Book {
            bids: vec![L2Level {
                price: 50200.0,
                size: 1.0,
            }],
            asks: vec![L2Level {
                price: 50100.0,
                size: 1.0,
            }],
        };

        assert!(book.is_crossed());
    }
}
