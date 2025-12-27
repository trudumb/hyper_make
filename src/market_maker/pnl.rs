//! P&L Attribution - decompose profits and losses into components.
//!
//! Breaks down total P&L into:
//! - **Spread Capture**: Revenue from bid-ask spread
//! - **Adverse Selection**: Loss from trading against informed flow
//! - **Inventory Carry**: Gain/loss from holding positions
//! - **Funding**: Cost/revenue from perpetual funding
//! - **Fees**: Exchange trading fees
//!
//! This helps diagnose what's driving performance and optimize strategy.

use std::collections::VecDeque;
use std::time::Instant;

/// Configuration for P&L attribution.
#[derive(Debug, Clone)]
pub struct PnLConfig {
    /// Fee rate (as fraction, e.g., 0.0002 = 2 bps)
    pub fee_rate: f64,
    /// Maximum history entries to keep
    pub max_history: usize,
}

impl Default for PnLConfig {
    fn default() -> Self {
        Self {
            fee_rate: 0.0002, // 2 bps maker fee
            max_history: 1000,
        }
    }
}

/// A single fill record for P&L attribution.
#[derive(Debug, Clone, Copy)]
pub struct FillRecord {
    /// Trade ID
    pub tid: u64,
    /// Fill price
    pub price: f64,
    /// Fill size (always positive)
    pub size: f64,
    /// Was this a buy?
    pub is_buy: bool,
    /// Mid price at time of fill
    pub mid_at_fill: f64,
    /// Mid price 1 second after fill (for AS measurement)
    pub mid_after_1s: Option<f64>,
    /// Timestamp
    pub timestamp: Instant,
}

/// Decomposed P&L components.
#[derive(Debug, Clone, Default)]
pub struct PnLComponents {
    /// Revenue from spread capture (fill price vs mid)
    pub spread_capture: f64,
    /// Loss from adverse selection (mid movement after fill)
    pub adverse_selection: f64,
    /// Gain/loss from inventory carry (unrealized P&L)
    pub inventory_carry: f64,
    /// Funding payments (positive = paid, negative = received)
    pub funding: f64,
    /// Trading fees paid
    pub fees: f64,
    /// Total realized P&L
    pub realized_pnl: f64,
    /// Total unrealized P&L
    pub unrealized_pnl: f64,
}

impl PnLComponents {
    /// Get total P&L (realized + unrealized).
    pub fn total(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl
    }

    /// Get net trading P&L (spread - AS - fees).
    pub fn net_trading(&self) -> f64 {
        self.spread_capture - self.adverse_selection - self.fees
    }
}

/// P&L attribution tracker.
pub struct PnLTracker {
    config: PnLConfig,

    /// Fill history
    fills: VecDeque<FillRecord>,

    /// Current position
    position: f64,

    /// Average entry price (for unrealized P&L)
    avg_entry_price: f64,

    /// Total cost basis
    cost_basis: f64,

    /// Cumulative spread capture
    total_spread_capture: f64,

    /// Cumulative adverse selection
    total_adverse_selection: f64,

    /// Cumulative funding
    total_funding: f64,

    /// Cumulative fees
    total_fees: f64,

    /// Cumulative realized P&L
    total_realized_pnl: f64,

    /// Number of fills
    fill_count: usize,

    /// Start time
    start_time: Instant,
}

impl PnLTracker {
    /// Create a new P&L tracker.
    pub fn new(config: PnLConfig) -> Self {
        Self {
            config,
            fills: VecDeque::with_capacity(1000),
            position: 0.0,
            avg_entry_price: 0.0,
            cost_basis: 0.0,
            total_spread_capture: 0.0,
            total_adverse_selection: 0.0,
            total_funding: 0.0,
            total_fees: 0.0,
            total_realized_pnl: 0.0,
            fill_count: 0,
            start_time: Instant::now(),
        }
    }

    /// Record a fill.
    ///
    /// # Arguments
    /// - `tid`: Trade ID
    /// - `price`: Fill price
    /// - `size`: Fill size (always positive)
    /// - `is_buy`: Was this a buy?
    /// - `mid_at_fill`: Mid price at time of fill
    pub fn record_fill(
        &mut self,
        tid: u64,
        price: f64,
        size: f64,
        is_buy: bool,
        mid_at_fill: f64,
    ) {
        let now = Instant::now();

        // Calculate spread capture: how much we beat the mid
        // Buy at price < mid = positive spread capture
        // Sell at price > mid = positive spread capture
        let spread_capture = if is_buy {
            (mid_at_fill - price) * size
        } else {
            (price - mid_at_fill) * size
        };
        self.total_spread_capture += spread_capture;

        // Calculate fees
        let notional = price * size;
        let fee = notional * self.config.fee_rate;
        self.total_fees += fee;

        // Update position and cost basis
        let _signed_size = if is_buy { size } else { -size };

        if is_buy {
            // Buying increases position
            if self.position >= 0.0 {
                // Adding to long position
                let new_cost = self.cost_basis + price * size;
                let new_position = self.position + size;
                self.avg_entry_price = new_cost / new_position;
                self.cost_basis = new_cost;
                self.position = new_position;
            } else {
                // Reducing short position (realize P&L)
                let close_size = size.min(self.position.abs());
                let realized = (self.avg_entry_price - price) * close_size;
                self.total_realized_pnl += realized;

                let remaining = size - close_size;
                if remaining > 0.0 {
                    // Flipping to long
                    self.position = remaining;
                    self.avg_entry_price = price;
                    self.cost_basis = price * remaining;
                } else {
                    self.position += size;
                    self.cost_basis = self.avg_entry_price * self.position.abs();
                }
            }
        } else {
            // Selling decreases position
            if self.position <= 0.0 {
                // Adding to short position
                let new_cost = self.cost_basis + price * size;
                let new_position = self.position - size;
                self.avg_entry_price = new_cost / new_position.abs();
                self.cost_basis = new_cost;
                self.position = new_position;
            } else {
                // Reducing long position (realize P&L)
                let close_size = size.min(self.position);
                let realized = (price - self.avg_entry_price) * close_size;
                self.total_realized_pnl += realized;

                let remaining = size - close_size;
                if remaining > 0.0 {
                    // Flipping to short
                    self.position = -remaining;
                    self.avg_entry_price = price;
                    self.cost_basis = price * remaining;
                } else {
                    self.position -= size;
                    self.cost_basis = self.avg_entry_price * self.position.abs();
                }
            }
        }

        // Store fill for AS measurement
        let fill = FillRecord {
            tid,
            price,
            size,
            is_buy,
            mid_at_fill,
            mid_after_1s: None, // Will be updated later
            timestamp: now,
        };
        self.fills.push_back(fill);
        self.fill_count += 1;

        // Trim history
        while self.fills.len() > self.config.max_history {
            self.fills.pop_front();
        }
    }

    /// Update adverse selection measurement for a fill.
    ///
    /// Call this ~1 second after a fill with the current mid price.
    pub fn update_adverse_selection(&mut self, tid: u64, mid_after: f64) {
        if let Some(fill) = self.fills.iter_mut().find(|f| f.tid == tid) {
            if fill.mid_after_1s.is_none() {
                fill.mid_after_1s = Some(mid_after);

                // Calculate adverse selection
                // If we bought and price went up, that's good (negative AS)
                // If we bought and price went down, that's bad (positive AS)
                let price_move = mid_after - fill.mid_at_fill;
                let as_cost = if fill.is_buy {
                    -price_move * fill.size // Bought, price down = cost
                } else {
                    price_move * fill.size // Sold, price up = cost
                };
                self.total_adverse_selection += as_cost.max(0.0);
            }
        }
    }

    /// Record a funding payment.
    ///
    /// Positive = paid (cost), Negative = received (revenue)
    pub fn record_funding(&mut self, amount: f64) {
        self.total_funding += amount;
    }

    /// Get unrealized P&L at current mid price.
    pub fn unrealized_pnl(&self, current_mid: f64) -> f64 {
        if self.position.abs() < 1e-9 {
            return 0.0;
        }

        if self.position > 0.0 {
            (current_mid - self.avg_entry_price) * self.position
        } else {
            (self.avg_entry_price - current_mid) * self.position.abs()
        }
    }

    /// Get full P&L attribution.
    pub fn attribution(&self, current_mid: f64) -> PnLComponents {
        let unrealized = self.unrealized_pnl(current_mid);

        PnLComponents {
            spread_capture: self.total_spread_capture,
            adverse_selection: self.total_adverse_selection,
            inventory_carry: unrealized, // Inventory carry = unrealized P&L
            funding: self.total_funding,
            fees: self.total_fees,
            realized_pnl: self.total_realized_pnl,
            unrealized_pnl: unrealized,
        }
    }

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Get average entry price.
    pub fn avg_entry_price(&self) -> f64 {
        self.avg_entry_price
    }

    /// Get fill count.
    pub fn fill_count(&self) -> usize {
        self.fill_count
    }

    /// Get summary.
    pub fn summary(&self, current_mid: f64) -> PnLSummary {
        let attr = self.attribution(current_mid);

        PnLSummary {
            position: self.position,
            avg_entry_price: self.avg_entry_price,
            fill_count: self.fill_count,
            spread_capture: attr.spread_capture,
            adverse_selection: attr.adverse_selection,
            funding: attr.funding,
            fees: attr.fees,
            realized_pnl: attr.realized_pnl,
            unrealized_pnl: attr.unrealized_pnl,
            total_pnl: attr.total(),
            uptime_secs: self.start_time.elapsed().as_secs_f64(),
        }
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.fills.clear();
        self.position = 0.0;
        self.avg_entry_price = 0.0;
        self.cost_basis = 0.0;
        self.total_spread_capture = 0.0;
        self.total_adverse_selection = 0.0;
        self.total_funding = 0.0;
        self.total_fees = 0.0;
        self.total_realized_pnl = 0.0;
        self.fill_count = 0;
        self.start_time = Instant::now();
    }
}

/// P&L summary for reporting.
#[derive(Debug, Clone)]
pub struct PnLSummary {
    pub position: f64,
    pub avg_entry_price: f64,
    pub fill_count: usize,
    pub spread_capture: f64,
    pub adverse_selection: f64,
    pub funding: f64,
    pub fees: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub uptime_secs: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PnLConfig::default();
        assert_eq!(config.fee_rate, 0.0002);
    }

    #[test]
    fn test_new_tracker() {
        let tracker = PnLTracker::new(PnLConfig::default());
        assert_eq!(tracker.position(), 0.0);
        assert_eq!(tracker.fill_count(), 0);
    }

    #[test]
    fn test_buy_fill() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Buy 1 BTC at $50000, mid was $50010 (10 bps better)
        tracker.record_fill(1, 50000.0, 1.0, true, 50010.0);

        assert_eq!(tracker.position(), 1.0);
        assert_eq!(tracker.avg_entry_price(), 50000.0);
        assert!((tracker.total_spread_capture - 10.0).abs() < 0.01); // $10 spread capture
    }

    #[test]
    fn test_sell_fill() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Sell 1 BTC at $50010, mid was $50000 (10 bps better)
        tracker.record_fill(1, 50010.0, 1.0, false, 50000.0);

        assert_eq!(tracker.position(), -1.0);
        assert!((tracker.total_spread_capture - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_close_position_realized_pnl() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Buy 1 BTC at $50000
        tracker.record_fill(1, 50000.0, 1.0, true, 50000.0);

        // Sell 1 BTC at $51000 (profit)
        tracker.record_fill(2, 51000.0, 1.0, false, 51000.0);

        assert_eq!(tracker.position(), 0.0);
        assert!((tracker.total_realized_pnl - 1000.0).abs() < 0.01); // $1000 profit
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Buy 1 BTC at $50000
        tracker.record_fill(1, 50000.0, 1.0, true, 50000.0);

        // Current mid is $51000
        let unrealized = tracker.unrealized_pnl(51000.0);
        assert!((unrealized - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_fees() {
        let config = PnLConfig {
            fee_rate: 0.001, // 10 bps
            ..Default::default()
        };
        let mut tracker = PnLTracker::new(config);

        // Trade $50000 notional
        tracker.record_fill(1, 50000.0, 1.0, true, 50000.0);

        assert!((tracker.total_fees - 50.0).abs() < 0.01); // $50 fee
    }

    #[test]
    fn test_attribution() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        tracker.record_fill(1, 50000.0, 1.0, true, 50010.0);
        tracker.record_funding(5.0); // Paid $5 funding

        let attr = tracker.attribution(50100.0);

        assert!(attr.spread_capture > 0.0);
        assert!(attr.fees > 0.0);
        assert_eq!(attr.funding, 5.0);
        assert!(attr.unrealized_pnl > 0.0); // Price went up
    }

    #[test]
    fn test_adverse_selection_measurement() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Buy at mid $50000
        tracker.record_fill(1, 50000.0, 1.0, true, 50000.0);

        // Price dropped to $49900 after 1s (adverse selection!)
        tracker.update_adverse_selection(1, 49900.0);

        assert!((tracker.total_adverse_selection - 100.0).abs() < 0.01); // $100 AS cost
    }

    #[test]
    fn test_summary() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        tracker.record_fill(1, 50000.0, 1.0, true, 50010.0);

        let summary = tracker.summary(50100.0);
        assert_eq!(summary.fill_count, 1);
        assert!(summary.total_pnl > 0.0);
    }
}
