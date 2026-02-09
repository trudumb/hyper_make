//! Unified risk state snapshot.
//!
//! Captures all risk-relevant data at a point in time for evaluation by monitors.

use std::time::{Duration, Instant};

/// Unified risk state snapshot.
///
/// This is the single source of truth for all risk-relevant data.
/// All monitors evaluate this same snapshot to ensure consistency.
#[derive(Debug, Clone)]
pub struct RiskState {
    // === P&L Metrics ===
    /// Current realized + unrealized P&L
    pub daily_pnl: f64,
    /// Peak P&L for drawdown calculation
    pub peak_pnl: f64,
    /// Realized P&L from closed trades
    pub realized_pnl: f64,
    /// Unrealized P&L from open position
    pub unrealized_pnl: f64,

    // === Position Metrics ===
    /// Current position size (contracts)
    pub position: f64,
    /// Maximum allowed position
    pub max_position: f64,
    /// Current mid price
    pub mid_price: f64,
    /// Position value in USD (|position| × mid_price)
    pub position_value: f64,
    /// Maximum allowed position value in USD
    pub max_position_value: f64,
    /// Account equity/value in USD
    pub account_value: f64,

    // === Volatility/Market Metrics ===
    /// Current volatility estimate (per-second σ)
    pub sigma: f64,
    /// Confidence in sigma estimate (0.0-1.0)
    pub sigma_confidence: f64,
    /// Jump ratio (RV/BV) - toxic flow indicator
    pub jump_ratio: f64,
    /// Is market in toxic regime?
    pub is_toxic_regime: bool,

    // === Price Velocity ===
    /// Absolute price velocity over 1 second: abs(mid_delta / mid) / elapsed_s
    pub price_velocity_1s: f64,

    // === Cascade/Tail Risk Metrics ===
    /// Liquidation cascade severity (0.0 = calm, 1.0+ = cascade)
    pub cascade_severity: f64,
    /// Tail risk multiplier from cascade detection
    pub tail_risk_multiplier: f64,
    /// Should quotes be pulled due to cascade?
    pub should_pull_quotes: bool,

    // === Adverse Selection Metrics ===
    /// Realized adverse selection in basis points
    pub adverse_selection_bps: f64,
    /// Probability of informed flow (α)
    pub alpha: f64,

    // === Pending Exposure Metrics ===
    /// Pending bid exposure (total remaining size on buy orders)
    pub pending_bid_exposure: f64,
    /// Pending ask exposure (total remaining size on sell orders)
    pub pending_ask_exposure: f64,
    /// Net pending change (bid - ask; positive = net long if all fill)
    pub net_pending_change: f64,
    /// Worst-case max position if all bids fill and no asks fill
    pub worst_case_max_position: f64,
    /// Worst-case min position if all asks fill and no bids fill
    pub worst_case_min_position: f64,

    // === Data Freshness ===
    /// Time since last market data update
    pub data_age: Duration,
    /// Last data update timestamp
    pub last_data_time: Instant,
    /// Rate limit errors count
    pub rate_limit_errors: u32,

    // === Connection State ===
    /// Whether WebSocket is actively reconnecting
    pub is_reconnecting: bool,
    /// Current reconnection attempt number (0 = not reconnecting)
    pub reconnection_attempt: u32,
    /// Whether connection has permanently failed
    pub connection_failed: bool,

    // === Timestamp ===
    /// When this snapshot was taken
    pub timestamp: Instant,
}

impl RiskState {
    /// Create a new risk state with the given parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        daily_pnl: f64,
        peak_pnl: f64,
        position: f64,
        max_position: f64,
        mid_price: f64,
        max_position_value: f64,
        account_value: f64,
        sigma: f64,
        cascade_severity: f64,
        last_data_time: Instant,
    ) -> Self {
        let now = Instant::now();
        Self {
            daily_pnl,
            peak_pnl,
            realized_pnl: 0.0,
            unrealized_pnl: daily_pnl,
            position,
            max_position,
            mid_price,
            position_value: position.abs() * mid_price,
            max_position_value,
            account_value,
            sigma,
            sigma_confidence: 1.0,
            jump_ratio: 1.0,
            is_toxic_regime: false,
            price_velocity_1s: 0.0,
            cascade_severity,
            tail_risk_multiplier: 1.0,
            should_pull_quotes: false,
            adverse_selection_bps: 0.0,
            alpha: 0.0,
            pending_bid_exposure: 0.0,
            pending_ask_exposure: 0.0,
            net_pending_change: 0.0,
            worst_case_max_position: position,
            worst_case_min_position: position,
            data_age: now.duration_since(last_data_time),
            last_data_time,
            rate_limit_errors: 0,
            is_reconnecting: false,
            reconnection_attempt: 0,
            connection_failed: false,
            timestamp: now,
        }
    }

    /// Builder-style method to set realized/unrealized PnL.
    pub fn with_pnl_breakdown(mut self, realized: f64, unrealized: f64) -> Self {
        self.realized_pnl = realized;
        self.unrealized_pnl = unrealized;
        self
    }

    /// Builder-style method to set volatility metrics.
    pub fn with_volatility(mut self, sigma: f64, confidence: f64, jump_ratio: f64) -> Self {
        self.sigma = sigma;
        self.sigma_confidence = confidence;
        self.jump_ratio = jump_ratio;
        self.is_toxic_regime = jump_ratio > 3.0;
        self
    }

    /// Builder-style method to set cascade metrics.
    pub fn with_cascade(mut self, severity: f64, tail_risk_mult: f64, should_pull: bool) -> Self {
        self.cascade_severity = severity;
        self.tail_risk_multiplier = tail_risk_mult;
        self.should_pull_quotes = should_pull;
        self
    }

    /// Builder-style method to set adverse selection metrics.
    pub fn with_adverse_selection(mut self, as_bps: f64, alpha: f64) -> Self {
        self.adverse_selection_bps = as_bps;
        self.alpha = alpha;
        self
    }

    /// Builder-style method to set rate limit errors.
    pub fn with_rate_limit_errors(mut self, errors: u32) -> Self {
        self.rate_limit_errors = errors;
        self
    }

    /// Builder-style method to set price velocity.
    pub fn with_price_velocity(mut self, velocity_1s: f64) -> Self {
        self.price_velocity_1s = velocity_1s;
        self
    }

    /// Builder-style method to set connection state.
    pub fn with_connection_state(
        mut self,
        is_reconnecting: bool,
        attempt: u32,
        failed: bool,
    ) -> Self {
        self.is_reconnecting = is_reconnecting;
        self.reconnection_attempt = attempt;
        self.connection_failed = failed;
        self
    }

    /// Builder-style method to set pending exposure metrics.
    ///
    /// # Arguments
    /// - `bid_exposure`: Total remaining size on buy orders
    /// - `ask_exposure`: Total remaining size on sell orders
    pub fn with_pending_exposure(mut self, bid_exposure: f64, ask_exposure: f64) -> Self {
        self.pending_bid_exposure = bid_exposure;
        self.pending_ask_exposure = ask_exposure;
        self.net_pending_change = bid_exposure - ask_exposure;
        self.worst_case_max_position = self.position + bid_exposure;
        self.worst_case_min_position = self.position - ask_exposure;
        self
    }

    // === Computed Properties ===

    /// Calculate inventory utilization (0.0 to 1.0+).
    pub fn inventory_utilization(&self) -> f64 {
        if self.max_position > 0.0 {
            self.position.abs() / self.max_position
        } else {
            0.0
        }
    }

    /// Calculate position value utilization (0.0 to 1.0+).
    pub fn value_utilization(&self) -> f64 {
        if self.max_position_value > 0.0 {
            self.position_value / self.max_position_value
        } else {
            0.0
        }
    }

    /// Calculate drawdown from peak as fraction of account value (0.0 to 1.0+).
    pub fn drawdown(&self) -> f64 {
        if self.account_value <= 0.0 {
            return 0.0;
        }
        if self.peak_pnl <= 0.0 {
            // No profit yet — no drawdown from peak
            return 0.0;
        }
        (self.peak_pnl - self.daily_pnl) / self.account_value
    }

    /// Is data stale (beyond threshold)?
    pub fn is_data_stale(&self, threshold: Duration) -> bool {
        self.data_age > threshold
    }

    /// Is position over limit (contracts)?
    pub fn is_over_position_limit(&self) -> bool {
        self.position.abs() > self.max_position
    }

    /// Is position value over limit (USD)?
    pub fn is_over_value_limit(&self) -> bool {
        self.position_value > self.max_position_value
    }

    /// Should enter reduce-only mode?
    pub fn should_reduce_only(&self) -> bool {
        self.is_over_position_limit() || self.is_over_value_limit()
    }

    // === Pending Exposure Properties ===

    /// Would worst-case max position exceed limits?
    ///
    /// Returns true if all buy orders filling would exceed max_position.
    pub fn worst_case_exceeds_long_limit(&self) -> bool {
        self.worst_case_max_position > self.max_position
    }

    /// Would worst-case min position exceed limits (short)?
    ///
    /// Returns true if all sell orders filling would exceed max_position on short side.
    pub fn worst_case_exceeds_short_limit(&self) -> bool {
        self.worst_case_min_position < -self.max_position
    }

    /// Would worst-case position on either side exceed limits?
    pub fn worst_case_exceeds_limits(&self) -> bool {
        self.worst_case_exceeds_long_limit() || self.worst_case_exceeds_short_limit()
    }

    /// Get the worst-case position value (max absolute position × mid_price).
    pub fn worst_case_position_value(&self) -> f64 {
        let max_abs = self
            .worst_case_max_position
            .abs()
            .max(self.worst_case_min_position.abs());
        max_abs * self.mid_price
    }

    /// Is the book balanced (roughly equal bid/ask exposure)?
    ///
    /// Returns true if net pending change is within 10% of total exposure.
    pub fn is_book_balanced(&self) -> bool {
        let total_exposure = self.pending_bid_exposure + self.pending_ask_exposure;
        if total_exposure < 1e-9 {
            return true; // No orders, considered balanced
        }
        (self.net_pending_change.abs() / total_exposure) < 0.1
    }
}

impl Default for RiskState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            daily_pnl: 0.0,
            peak_pnl: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            position: 0.0,
            max_position: 1.0,
            mid_price: 0.0,
            position_value: 0.0,
            max_position_value: 10_000.0,
            account_value: 10_000.0,
            sigma: 0.0002,
            sigma_confidence: 0.0,
            jump_ratio: 1.0,
            is_toxic_regime: false,
            price_velocity_1s: 0.0,
            cascade_severity: 0.0,
            tail_risk_multiplier: 1.0,
            should_pull_quotes: false,
            adverse_selection_bps: 0.0,
            alpha: 0.0,
            pending_bid_exposure: 0.0,
            pending_ask_exposure: 0.0,
            net_pending_change: 0.0,
            worst_case_max_position: 0.0,
            worst_case_min_position: 0.0,
            data_age: Duration::ZERO,
            last_data_time: now,
            rate_limit_errors: 0,
            is_reconnecting: false,
            reconnection_attempt: 0,
            connection_failed: false,
            timestamp: now,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_utilization() {
        let state = RiskState {
            position: 0.5,
            max_position: 1.0,
            ..Default::default()
        };
        assert!((state.inventory_utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_value_utilization() {
        let state = RiskState {
            position_value: 5000.0,
            max_position_value: 10000.0,
            ..Default::default()
        };
        assert!((state.value_utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_drawdown_normal_case() {
        // 10% drawdown on $1000 account: peak=$100, current=$0 => (100-0)/1000 = 0.10
        let state = RiskState {
            daily_pnl: 0.0,
            peak_pnl: 100.0,
            account_value: 1000.0,
            ..Default::default()
        };
        assert!((state.drawdown() - 0.10).abs() < f64::EPSILON);
    }

    #[test]
    fn test_drawdown_tiny_peak_no_explosion() {
        // Bug case: peak=$0.001, loss to -$0.01 on $1000 account
        // Old code: (0.001 - (-0.01)) / 0.001 = 11.0 (1100%!) -- blows up
        // New code: (0.001 - (-0.01)) / 1000 = 0.000011 (~0.001%) -- correct
        let state = RiskState {
            daily_pnl: -0.01,
            peak_pnl: 0.001,
            account_value: 1000.0,
            ..Default::default()
        };
        let dd = state.drawdown();
        assert!(dd < 0.001, "drawdown should be tiny, got {dd}");
        assert!(dd > 0.0, "drawdown should be positive");
    }

    #[test]
    fn test_drawdown_zero_account_value_returns_zero() {
        let state = RiskState {
            daily_pnl: -10.0,
            peak_pnl: 50.0,
            account_value: 0.0,
            ..Default::default()
        };
        assert_eq!(state.drawdown(), 0.0);
    }

    #[test]
    fn test_drawdown_no_profit_yet_returns_zero() {
        // No profit yet (peak_pnl <= 0) -- drawdown should be 0
        let state = RiskState {
            daily_pnl: -5.0,
            peak_pnl: 0.0,
            account_value: 1000.0,
            ..Default::default()
        };
        assert_eq!(state.drawdown(), 0.0);
    }

    #[test]
    fn test_reduce_only() {
        let mut state = RiskState::default();
        state.position = 0.5;
        state.max_position = 1.0;
        state.position_value = 5000.0;
        state.max_position_value = 10000.0;
        assert!(!state.should_reduce_only());

        state.position = 1.5; // Over limit
        assert!(state.should_reduce_only());
    }

    // === Pending Exposure Tests ===

    #[test]
    fn test_with_pending_exposure() {
        let state = RiskState::default().with_pending_exposure(1.0, 0.5);

        assert!((state.pending_bid_exposure - 1.0).abs() < f64::EPSILON);
        assert!((state.pending_ask_exposure - 0.5).abs() < f64::EPSILON);
        assert!((state.net_pending_change - 0.5).abs() < f64::EPSILON); // 1.0 - 0.5
                                                                        // Default position is 0, so worst case:
        assert!((state.worst_case_max_position - 1.0).abs() < f64::EPSILON); // 0 + 1.0
        assert!((state.worst_case_min_position - (-0.5)).abs() < f64::EPSILON); // 0 - 0.5
    }

    #[test]
    fn test_worst_case_exceeds_long_limit() {
        let mut state = RiskState::default();
        state.position = 0.5;
        state.max_position = 1.0;

        // Within limits
        let state = state.with_pending_exposure(0.3, 0.2);
        assert!(!state.worst_case_exceeds_long_limit()); // 0.5 + 0.3 = 0.8 < 1.0

        // Exceeds limits
        let state = RiskState {
            position: 0.5,
            max_position: 1.0,
            ..Default::default()
        }
        .with_pending_exposure(0.6, 0.2); // 0.5 + 0.6 = 1.1 > 1.0
        assert!(state.worst_case_exceeds_long_limit());
    }

    #[test]
    fn test_worst_case_exceeds_short_limit() {
        let state = RiskState {
            position: -0.5,
            max_position: 1.0,
            ..Default::default()
        }
        .with_pending_exposure(0.2, 0.6); // -0.5 - 0.6 = -1.1 < -1.0
        assert!(state.worst_case_exceeds_short_limit());
    }

    #[test]
    fn test_worst_case_position_value() {
        let state = RiskState {
            position: 0.5,
            mid_price: 100.0,
            ..Default::default()
        }
        .with_pending_exposure(1.0, 0.5);

        // worst_case_max_position = 0.5 + 1.0 = 1.5
        // worst_case_min_position = 0.5 - 0.5 = 0.0
        // max absolute = max(|1.5|, |0.0|) = 1.5
        // value = 1.5 * 100 = 150
        assert!((state.worst_case_position_value() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_book_balanced() {
        // Balanced book (within 10%)
        let state = RiskState::default().with_pending_exposure(1.0, 0.95);
        assert!(state.is_book_balanced()); // net = 0.05, total = 1.95, ratio = 2.5% < 10%

        // Unbalanced book
        let state = RiskState::default().with_pending_exposure(1.0, 0.5);
        assert!(!state.is_book_balanced()); // net = 0.5, total = 1.5, ratio = 33% > 10%

        // Empty book is balanced
        let state = RiskState::default().with_pending_exposure(0.0, 0.0);
        assert!(state.is_book_balanced());
    }
}
