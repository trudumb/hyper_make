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

    // === Data Freshness ===
    /// Time since last market data update
    pub data_age: Duration,
    /// Last data update timestamp
    pub last_data_time: Instant,
    /// Rate limit errors count
    pub rate_limit_errors: u32,

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
            cascade_severity,
            tail_risk_multiplier: 1.0,
            should_pull_quotes: false,
            adverse_selection_bps: 0.0,
            alpha: 0.0,
            data_age: now.duration_since(last_data_time),
            last_data_time,
            rate_limit_errors: 0,
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

    /// Calculate drawdown from peak (0.0 to 1.0+).
    pub fn drawdown(&self) -> f64 {
        if self.peak_pnl > 0.0 {
            (self.peak_pnl - self.daily_pnl) / self.peak_pnl
        } else {
            0.0
        }
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
            cascade_severity: 0.0,
            tail_risk_multiplier: 1.0,
            should_pull_quotes: false,
            adverse_selection_bps: 0.0,
            alpha: 0.0,
            data_age: Duration::ZERO,
            last_data_time: now,
            rate_limit_errors: 0,
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
    fn test_drawdown() {
        let state = RiskState {
            daily_pnl: 80.0,
            peak_pnl: 100.0,
            ..Default::default()
        };
        assert!((state.drawdown() - 0.2).abs() < f64::EPSILON);
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
}
