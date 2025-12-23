//! Trade flow tracker for buy/sell imbalance detection.
//!
//! Uses the trade side field from exchange data to detect
//! directional order flow before it shows up in price.

use std::collections::VecDeque;

/// Tracks buy vs sell aggressor imbalance from trade tape.
///
/// Uses the trade side field from Hyperliquid ("B" = buy aggressor, "S" = sell aggressor)
/// to detect directional order flow before it shows up in price.
#[derive(Debug)]
pub(crate) struct TradeFlowTracker {
    /// (timestamp_ms, signed_volume): positive = buy aggressor
    trades: VecDeque<(u64, f64)>,
    /// Rolling window (ms)
    window_ms: u64,
    /// EWMA smoothed imbalance
    ewma_imbalance: f64,
    /// EWMA alpha
    alpha: f64,
}

impl TradeFlowTracker {
    pub(super) fn new(window_ms: u64, alpha: f64) -> Self {
        Self {
            trades: VecDeque::with_capacity(500),
            window_ms,
            ewma_imbalance: 0.0,
            alpha,
        }
    }

    /// Add a trade from the tape.
    /// is_buy_aggressor: true if buyer was taker (lifted the ask)
    pub(super) fn on_trade(&mut self, timestamp_ms: u64, size: f64, is_buy_aggressor: bool) {
        let signed = if is_buy_aggressor { size } else { -size };
        self.trades.push_back((timestamp_ms, signed));

        // Expire old trades
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self
            .trades
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.trades.pop_front();
        }

        // Update EWMA
        let instant = self.compute_instant_imbalance();
        self.ewma_imbalance = self.alpha * instant + (1.0 - self.alpha) * self.ewma_imbalance;
    }

    /// Compute instantaneous imbalance: (buy - sell) / total
    fn compute_instant_imbalance(&self) -> f64 {
        let (buy_vol, sell_vol) =
            self.trades.iter().fold(
                (0.0, 0.0),
                |(b, s), (_, v)| {
                    if *v > 0.0 {
                        (b + v, s)
                    } else {
                        (b, s - v)
                    }
                },
            );
        let total = buy_vol + sell_vol;
        if total < 1e-12 {
            0.0
        } else {
            (buy_vol - sell_vol) / total
        }
    }

    /// Smoothed flow imbalance [-1, 1]
    /// Negative = sell pressure, Positive = buy pressure
    pub(super) fn imbalance(&self) -> f64 {
        self.ewma_imbalance.clamp(-1.0, 1.0)
    }

    /// Is there dominant selling (for bid protection)?
    #[allow(dead_code)]
    pub(super) fn is_sell_pressure(&self) -> bool {
        self.ewma_imbalance < -0.25
    }

    /// Is there dominant buying (for ask protection)?
    #[allow(dead_code)]
    pub(super) fn is_buy_pressure(&self) -> bool {
        self.ewma_imbalance > 0.25
    }
}
