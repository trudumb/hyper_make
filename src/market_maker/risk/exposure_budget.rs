//! Centralized position exposure budget.
//!
//! Tracks ALL exposure sources — resting orders, in-flight placements, and current
//! position — to compute worst-case positions and available capacity per side.
//!
//! # Problem
//!
//! Without centralized tracking, 3 simultaneous buy fills can push position to 111%
//! of max_position. Each subsystem independently checks position limits but none
//! accounts for the aggregate worst case.
//!
//! # Architecture
//!
//! ```text
//! ExposureBudget::snapshot() called once per quote cycle
//!   → worst_case_long = position + resting_bids + inflight_bids
//!   → worst_case_short = position - resting_asks - inflight_asks
//!   → available_bid_budget = (max_position - worst_case_long).max(0)
//!   → available_ask_budget = (max_position + worst_case_short).max(0)
//! ```

use serde::{Deserialize, Serialize};

/// Snapshot of exposure budget at a point in time.
///
/// Computed once per cycle and passed through MarketParams to all consumers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExposureSnapshot {
    /// Current position (signed: positive = long, negative = short)
    pub position: f64,
    /// Total size of resting bid orders (confirmed on exchange)
    pub resting_bid_exposure: f64,
    /// Total size of resting ask orders (confirmed on exchange)
    pub resting_ask_exposure: f64,
    /// Total size of in-flight bid placements (sent, not yet ack'd)
    pub inflight_bid_exposure: f64,
    /// Total size of in-flight ask placements (sent, not yet ack'd)
    pub inflight_ask_exposure: f64,
    /// Max position limit
    pub max_position: f64,
}

impl ExposureSnapshot {
    /// Worst-case long position if all bids fill simultaneously.
    pub fn worst_case_long(&self) -> f64 {
        self.position + self.resting_bid_exposure + self.inflight_bid_exposure
    }

    /// Worst-case short position if all asks fill simultaneously.
    /// Returns a negative number for short positions.
    pub fn worst_case_short(&self) -> f64 {
        self.position - self.resting_ask_exposure - self.inflight_ask_exposure
    }

    /// Available capacity for new buy orders before worst-case hits max_position.
    pub fn available_bid_budget(&self) -> f64 {
        (self.max_position - self.worst_case_long()).max(0.0)
    }

    /// Available capacity for new sell orders before worst-case hits -max_position.
    pub fn available_ask_budget(&self) -> f64 {
        (self.max_position + self.worst_case_short()).max(0.0)
    }

    /// Whether the worst-case long position exceeds max_position.
    pub fn long_overexposed(&self) -> bool {
        self.worst_case_long() > self.max_position
    }

    /// Whether the worst-case short position exceeds -max_position.
    pub fn short_overexposed(&self) -> bool {
        self.worst_case_short() < -self.max_position
    }

    /// Total exposure ratio: max(|worst_case_long|, |worst_case_short|) / max_position.
    pub fn exposure_ratio(&self) -> f64 {
        if self.max_position <= 0.0 {
            return f64::INFINITY;
        }
        let worst = self
            .worst_case_long()
            .abs()
            .max(self.worst_case_short().abs());
        worst / self.max_position
    }
}

impl Default for ExposureSnapshot {
    fn default() -> Self {
        Self {
            position: 0.0,
            resting_bid_exposure: 0.0,
            resting_ask_exposure: 0.0,
            inflight_bid_exposure: 0.0,
            inflight_ask_exposure: 0.0,
            max_position: 1.0,
        }
    }
}

/// Centralized exposure budget tracker.
///
/// Lives on QuoteEngine and is updated each cycle from OrderManager state.
/// Provides the authoritative `ExposureSnapshot` that all downstream consumers use.
#[derive(Debug, Clone)]
pub struct ExposureBudget {
    /// Current max position (may change with regime/margin)
    max_position: f64,
}

impl ExposureBudget {
    pub fn new(max_position: f64) -> Self {
        Self { max_position }
    }

    /// Update the max position limit (e.g., from margin recomputation).
    pub fn set_max_position(&mut self, max_position: f64) {
        self.max_position = max_position;
    }

    /// Compute a point-in-time exposure snapshot.
    ///
    /// Called once per quote cycle with the latest order manager state.
    pub fn snapshot(
        &self,
        position: f64,
        resting_bid_exposure: f64,
        resting_ask_exposure: f64,
        inflight_bid_exposure: f64,
        inflight_ask_exposure: f64,
    ) -> ExposureSnapshot {
        ExposureSnapshot {
            position,
            resting_bid_exposure,
            resting_ask_exposure,
            inflight_bid_exposure,
            inflight_ask_exposure,
            max_position: self.max_position,
        }
    }

    /// Current max position.
    pub fn max_position(&self) -> f64 {
        self.max_position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worst_case_long() {
        let snap = ExposureSnapshot {
            position: 30.0,
            resting_bid_exposure: 15.0,
            resting_ask_exposure: 10.0,
            inflight_bid_exposure: 5.0,
            inflight_ask_exposure: 0.0,
            max_position: 50.0,
        };
        // worst_case_long = 30 + 15 + 5 = 50
        assert!((snap.worst_case_long() - 50.0).abs() < 1e-10);
        assert!((snap.available_bid_budget() - 0.0).abs() < 1e-10);
        assert!(!snap.long_overexposed());
    }

    #[test]
    fn test_worst_case_overexposed() {
        let snap = ExposureSnapshot {
            position: 30.0,
            resting_bid_exposure: 25.0,
            resting_ask_exposure: 0.0,
            inflight_bid_exposure: 0.0,
            inflight_ask_exposure: 0.0,
            max_position: 50.0,
        };
        // worst_case_long = 30 + 25 = 55 > 50
        assert!(snap.long_overexposed());
        assert!((snap.available_bid_budget() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_available_budgets_short_position() {
        let snap = ExposureSnapshot {
            position: -20.0,
            resting_bid_exposure: 5.0,
            resting_ask_exposure: 10.0,
            inflight_bid_exposure: 0.0,
            inflight_ask_exposure: 0.0,
            max_position: 50.0,
        };
        // worst_case_long = -20 + 5 = -15 → available_bid = 50 - (-15) = 65
        assert!((snap.available_bid_budget() - 65.0).abs() < 1e-10);
        // worst_case_short = -20 - 10 = -30 → available_ask = 50 + (-30) = 20
        assert!((snap.available_ask_budget() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_exposure_ratio() {
        let snap = ExposureSnapshot {
            position: 40.0,
            resting_bid_exposure: 10.0,
            resting_ask_exposure: 0.0,
            inflight_bid_exposure: 0.0,
            inflight_ask_exposure: 0.0,
            max_position: 50.0,
        };
        // worst_case_long = 50, ratio = 50/50 = 1.0
        assert!((snap.exposure_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_budget_snapshot() {
        let budget = ExposureBudget::new(100.0);
        let snap = budget.snapshot(50.0, 20.0, 15.0, 5.0, 3.0);
        assert!((snap.worst_case_long() - 75.0).abs() < 1e-10);
        assert!((snap.worst_case_short() - 32.0).abs() < 1e-10);
        assert!((snap.available_bid_budget() - 25.0).abs() < 1e-10);
        assert!((snap.available_ask_budget() - 132.0).abs() < 1e-10);
    }
}
