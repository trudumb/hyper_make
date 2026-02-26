//! Reconcile outcome tracking for economic order reconciliation.
//!
//! Tracks decisions made by the reconciler (latch, modify, cancel+place, etc.)
//! and correlates them with fill outcomes to learn:
//! - Fill rates by action type
//! - Edge captured by action type
//! - Calibration bias of P(fill) predictions
//! - Suggested latch threshold adjustments

use std::collections::HashMap;
use std::time::Instant;

/// Number of distinct action types tracked.
const NUM_ACTION_TYPES: usize = 6;

/// Action types for outcome tracking.
/// Mirrors scorer::ActionType but defined here to avoid circular dependency during initial build.
/// Will be unified with scorer::ActionType in W6 integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum ReconcileActionType {
    Latch,
    ModifySize,
    ModifyPrice,
    CancelPlace,
    NewPlace,
    StaleCancel,
}

impl ReconcileActionType {
    fn index(self) -> usize {
        match self {
            Self::Latch => 0,
            Self::ModifySize => 1,
            Self::ModifyPrice => 2,
            Self::CancelPlace => 3,
            Self::NewPlace => 4,
            Self::StaleCancel => 5,
        }
    }

    /// All action types in index order.
    fn all() -> [Self; NUM_ACTION_TYPES] {
        [
            Self::Latch,
            Self::ModifySize,
            Self::ModifyPrice,
            Self::CancelPlace,
            Self::NewPlace,
            Self::StaleCancel,
        ]
    }
}

/// EWMA rate tracker for fill rates and edge.
#[derive(Debug, Clone)]
struct EwmaRate {
    rate: f64,
    alpha: f64,
    count: u64,
}

impl EwmaRate {
    fn new(alpha: f64) -> Self {
        Self {
            rate: 0.5,
            alpha,
            count: 0,
        }
    }

    fn update(&mut self, value: f64) {
        if self.count == 0 {
            self.rate = value;
        } else {
            self.rate = self.alpha * value + (1.0 - self.alpha) * self.rate;
        }
        self.count += 1;
    }

    fn rate(&self) -> f64 {
        self.rate
    }

    fn count(&self) -> u64 {
        self.count
    }
}

/// A recorded reconciliation decision awaiting outcome.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ReconcileDecision {
    pub oid: u64,
    pub action: ReconcileActionType,
    pub predicted_p_fill: f64,
    pub predicted_queue_value_bps: f64,
    pub price_drift_bps: f64,
    pub decided_at: Instant,
}

/// Summary statistics for logging.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OutcomeStats {
    pub total_decisions: u64,
    pub total_fills: u64,
    pub total_unfilled: u64,
    pub pending_count: usize,
    pub calibration_bias: f64,
    pub fill_rates: [(ReconcileActionType, f64, u64); NUM_ACTION_TYPES],
}

/// Tracks reconciliation decisions and correlates with fill outcomes.
#[allow(dead_code)]
pub struct ReconcileOutcomeTracker {
    /// Pending decisions awaiting fill/unfill outcome.
    pending_decisions: HashMap<u64, ReconcileDecision>,
    /// Fill rate EWMA per action type.
    fill_rates: [EwmaRate; NUM_ACTION_TYPES],
    /// Edge captured EWMA per action type.
    edge_by_action: [EwmaRate; NUM_ACTION_TYPES],
    /// EWMA of P(fill) prediction error (predicted - actual).
    calibration_bias: f64,
    /// EWMA alpha for calibration bias.
    bias_alpha: f64,
    /// Total decisions recorded.
    total_decisions: u64,
    /// Total fills observed.
    total_fills: u64,
    /// Total unfilled outcomes observed.
    total_unfilled: u64,
    /// Maximum age for pending decisions before auto-expiry (seconds).
    max_pending_age_secs: f64,
}

impl Default for ReconcileOutcomeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ReconcileOutcomeTracker {
    /// Create a new tracker with default parameters.
    /// EWMA alpha=0.02 (~50-sample half-life), bias alpha=0.05 (~20-sample), max pending age 300s.
    #[allow(dead_code)]
    pub fn new() -> Self {
        const RATE_ALPHA: f64 = 0.02;
        Self {
            pending_decisions: HashMap::new(),
            fill_rates: std::array::from_fn(|_| EwmaRate::new(RATE_ALPHA)),
            edge_by_action: std::array::from_fn(|_| EwmaRate::new(RATE_ALPHA)),
            calibration_bias: 0.0,
            bias_alpha: 0.05,
            total_decisions: 0,
            total_fills: 0,
            total_unfilled: 0,
            max_pending_age_secs: 300.0,
        }
    }

    /// Record a reconciliation decision. Stores it as pending until fill/unfill outcome arrives.
    #[allow(dead_code)]
    pub fn record_decision(&mut self, decision: ReconcileDecision) {
        self.total_decisions += 1;
        self.pending_decisions.insert(decision.oid, decision);
    }

    /// Record that an order was filled. Updates fill rate, edge, and calibration bias
    /// for the action type that produced this order.
    #[allow(dead_code)]
    pub fn record_fill(&mut self, oid: u64, realized_edge_bps: f64) {
        if let Some(decision) = self.pending_decisions.remove(&oid) {
            let idx = decision.action.index();

            // Update fill rate: 1.0 = filled
            self.fill_rates[idx].update(1.0);

            // Update edge captured for this action type
            self.edge_by_action[idx].update(realized_edge_bps);

            // Update calibration bias: predicted - actual (actual=1.0 for fill)
            let error = decision.predicted_p_fill - 1.0;
            self.calibration_bias =
                self.bias_alpha * error + (1.0 - self.bias_alpha) * self.calibration_bias;

            self.total_fills += 1;
        }
    }

    /// Record that an order expired or was cancelled without filling.
    /// Updates fill rate and calibration bias for the action type.
    #[allow(dead_code)]
    pub fn record_unfilled(&mut self, oid: u64) {
        if let Some(decision) = self.pending_decisions.remove(&oid) {
            let idx = decision.action.index();

            // Update fill rate: 0.0 = unfilled
            self.fill_rates[idx].update(0.0);

            // Update calibration bias: predicted - actual (actual=0.0 for unfilled)
            let error = decision.predicted_p_fill;
            self.calibration_bias =
                self.bias_alpha * error + (1.0 - self.bias_alpha) * self.calibration_bias;

            self.total_unfilled += 1;
        }
    }

    /// Suggest a latch threshold adjustment in bps based on observed latch fill rate.
    ///
    /// If latch fill rate is high (>0.6), we are latching orders that fill easily
    /// and could tolerate a wider latch threshold (positive adjustment).
    /// If latch fill rate is low (<0.3), we are latching orders that rarely fill
    /// and should tighten the threshold (negative adjustment).
    /// Linear interpolation between -1.0 bps and +1.0 bps.
    #[allow(dead_code)]
    pub fn suggested_latch_adjustment_bps(&self) -> f64 {
        let latch_idx = ReconcileActionType::Latch.index();
        let latch_rate = &self.fill_rates[latch_idx];

        // Need minimum observations before suggesting adjustments
        const MIN_OBS: u64 = 20;
        if latch_rate.count() < MIN_OBS {
            return 0.0;
        }

        let rate = latch_rate.rate();

        // Linear map: rate 0.3 -> -1.0 bps, rate 0.6 -> +1.0 bps
        // Midpoint at 0.45 -> 0.0 bps
        const LOW_RATE: f64 = 0.3;
        const HIGH_RATE: f64 = 0.6;
        const MAX_ADJ_BPS: f64 = 1.0;

        if rate < LOW_RATE {
            -MAX_ADJ_BPS
        } else if rate > HIGH_RATE {
            MAX_ADJ_BPS
        } else {
            // Linear interpolation: [0.3, 0.6] -> [-1.0, +1.0]
            let t = (rate - LOW_RATE) / (HIGH_RATE - LOW_RATE); // [0, 1]
            -MAX_ADJ_BPS + 2.0 * MAX_ADJ_BPS * t
        }
    }

    /// Get the current EWMA fill rate for a given action type.
    #[allow(dead_code)]
    pub fn fill_rate_for_action(&self, action: &ReconcileActionType) -> f64 {
        self.fill_rates[action.index()].rate()
    }

    /// Get the current calibration bias (EWMA of predicted - actual).
    /// Positive = over-predicting fill probability, negative = under-predicting.
    #[allow(dead_code)]
    pub fn calibration_bias(&self) -> f64 {
        self.calibration_bias
    }

    /// Number of pending decisions awaiting outcome.
    #[allow(dead_code)]
    pub fn pending_count(&self) -> usize {
        self.pending_decisions.len()
    }

    /// Get summary statistics for logging/diagnostics.
    #[allow(dead_code)]
    pub fn stats(&self) -> OutcomeStats {
        let all = ReconcileActionType::all();
        let mut fill_rates = [(ReconcileActionType::Latch, 0.0_f64, 0_u64); NUM_ACTION_TYPES];
        for (i, &action) in all.iter().enumerate() {
            fill_rates[i] = (
                action,
                self.fill_rates[i].rate(),
                self.fill_rates[i].count(),
            );
        }

        OutcomeStats {
            total_decisions: self.total_decisions,
            total_fills: self.total_fills,
            total_unfilled: self.total_unfilled,
            pending_count: self.pending_decisions.len(),
            calibration_bias: self.calibration_bias,
            fill_rates,
        }
    }

    /// Remove pending decisions older than `max_pending_age_secs`.
    /// Returns the number of stale decisions removed.
    #[allow(dead_code)]
    pub fn cleanup_stale_decisions(&mut self) -> usize {
        let max_age_secs = self.max_pending_age_secs;
        let before = self.pending_decisions.len();
        self.pending_decisions
            .retain(|_, d| d.decided_at.elapsed().as_secs_f64() < max_age_secs);
        before - self.pending_decisions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decision(
        oid: u64,
        action: ReconcileActionType,
        predicted_p_fill: f64,
    ) -> ReconcileDecision {
        ReconcileDecision {
            oid,
            action,
            predicted_p_fill,
            predicted_queue_value_bps: 0.5,
            price_drift_bps: 0.0,
            decided_at: Instant::now(),
        }
    }

    #[test]
    fn test_fill_recording_updates_correct_action_type() {
        let mut tracker = ReconcileOutcomeTracker::new();

        // Record a latch decision and a cancel+place decision
        tracker.record_decision(make_decision(1, ReconcileActionType::Latch, 0.5));
        tracker.record_decision(make_decision(2, ReconcileActionType::CancelPlace, 0.7));

        // Fill the latch order
        tracker.record_fill(1, 2.0);

        // Latch fill rate should move toward 1.0 (was 0.5 default, now first obs = 1.0)
        assert!((tracker.fill_rate_for_action(&ReconcileActionType::Latch) - 1.0).abs() < 1e-9);

        // CancelPlace should still be at default 0.5 (no outcome yet)
        assert!(
            (tracker.fill_rate_for_action(&ReconcileActionType::CancelPlace) - 0.5).abs() < 1e-9
        );

        assert_eq!(tracker.total_fills, 1);
        assert_eq!(tracker.pending_count(), 1); // decision 2 still pending
    }

    #[test]
    fn test_unfilled_recording_decreases_fill_rate() {
        let mut tracker = ReconcileOutcomeTracker::new();

        // Record and fill first to set rate to 1.0
        tracker.record_decision(make_decision(1, ReconcileActionType::ModifyPrice, 0.5));
        tracker.record_fill(1, 1.0);
        let rate_after_fill = tracker.fill_rate_for_action(&ReconcileActionType::ModifyPrice);
        assert!((rate_after_fill - 1.0).abs() < 1e-9);

        // Now record an unfilled outcome
        tracker.record_decision(make_decision(2, ReconcileActionType::ModifyPrice, 0.5));
        tracker.record_unfilled(2);

        // Rate should decrease (EWMA of [1.0, 0.0] with alpha=0.02)
        let rate_after_unfill = tracker.fill_rate_for_action(&ReconcileActionType::ModifyPrice);
        assert!(rate_after_unfill < rate_after_fill);
        assert_eq!(tracker.total_unfilled, 1);
    }

    #[test]
    fn test_calibration_bias_converges_toward_prediction_error() {
        let mut tracker = ReconcileOutcomeTracker::new();

        // Consistently predict 0.8 but never fill -> bias should go positive
        for i in 0..50 {
            tracker.record_decision(make_decision(i, ReconcileActionType::NewPlace, 0.8));
            tracker.record_unfilled(i);
        }

        // Bias = EWMA of (predicted - actual) = EWMA of (0.8 - 0.0) = ~0.8
        // With alpha=0.05 and 50 iterations, should converge close to 0.8
        let bias = tracker.calibration_bias();
        assert!(
            bias > 0.5,
            "bias should be positive (over-predicting): {}",
            bias
        );
        assert!(bias < 0.85, "bias should converge near 0.8: {}", bias);
    }

    #[test]
    fn test_latch_adjustment_high_fill_rate_widens() {
        let mut tracker = ReconcileOutcomeTracker::new();

        // Record 30 latch decisions that all fill -> high fill rate
        for i in 0..30 {
            tracker.record_decision(make_decision(i, ReconcileActionType::Latch, 0.5));
            tracker.record_fill(i, 1.0);
        }

        let rate = tracker.fill_rate_for_action(&ReconcileActionType::Latch);
        assert!(rate > 0.6, "fill rate should be high: {}", rate);

        let adj = tracker.suggested_latch_adjustment_bps();
        assert!(
            adj > 0.0,
            "high fill rate should suggest wider latch (positive bps): {}",
            adj
        );
    }

    #[test]
    fn test_latch_adjustment_low_fill_rate_tightens() {
        let mut tracker = ReconcileOutcomeTracker::new();

        // Record 30 latch decisions that never fill -> low fill rate
        for i in 0..30 {
            tracker.record_decision(make_decision(i, ReconcileActionType::Latch, 0.5));
            tracker.record_unfilled(i);
        }

        let rate = tracker.fill_rate_for_action(&ReconcileActionType::Latch);
        assert!(rate < 0.3, "fill rate should be low: {}", rate);

        let adj = tracker.suggested_latch_adjustment_bps();
        assert!(
            adj < 0.0,
            "low fill rate should suggest tighter latch (negative bps): {}",
            adj
        );
    }

    #[test]
    fn test_stale_decision_cleanup() {
        let mut tracker = ReconcileOutcomeTracker::new();

        // Insert a decision with a very short max age
        tracker.max_pending_age_secs = 0.0; // expire immediately
        tracker.record_decision(make_decision(1, ReconcileActionType::Latch, 0.5));
        tracker.record_decision(make_decision(2, ReconcileActionType::NewPlace, 0.5));

        assert_eq!(tracker.pending_count(), 2);

        let removed = tracker.cleanup_stale_decisions();
        assert_eq!(removed, 2);
        assert_eq!(tracker.pending_count(), 0);
    }

    #[test]
    fn test_stats_includes_all_action_types() {
        let mut tracker = ReconcileOutcomeTracker::new();

        // Record one decision per action type
        tracker.record_decision(make_decision(1, ReconcileActionType::Latch, 0.5));
        tracker.record_decision(make_decision(2, ReconcileActionType::ModifySize, 0.4));
        tracker.record_decision(make_decision(3, ReconcileActionType::ModifyPrice, 0.6));
        tracker.record_decision(make_decision(4, ReconcileActionType::CancelPlace, 0.7));
        tracker.record_decision(make_decision(5, ReconcileActionType::NewPlace, 0.3));
        tracker.record_decision(make_decision(6, ReconcileActionType::StaleCancel, 0.1));

        // Fill some, leave others
        tracker.record_fill(1, 1.0);
        tracker.record_fill(3, 2.0);
        tracker.record_unfilled(5);

        let stats = tracker.stats();
        assert_eq!(stats.total_decisions, 6);
        assert_eq!(stats.total_fills, 2);
        assert_eq!(stats.total_unfilled, 1);
        assert_eq!(stats.pending_count, 3); // 2, 4, 6 still pending
        assert_eq!(stats.fill_rates.len(), NUM_ACTION_TYPES);

        // Verify action type ordering matches
        assert_eq!(stats.fill_rates[0].0, ReconcileActionType::Latch);
        assert_eq!(stats.fill_rates[1].0, ReconcileActionType::ModifySize);
        assert_eq!(stats.fill_rates[2].0, ReconcileActionType::ModifyPrice);
        assert_eq!(stats.fill_rates[3].0, ReconcileActionType::CancelPlace);
        assert_eq!(stats.fill_rates[4].0, ReconcileActionType::NewPlace);
        assert_eq!(stats.fill_rates[5].0, ReconcileActionType::StaleCancel);

        // Latch had 1 fill -> rate should be 1.0 (first obs)
        assert!((stats.fill_rates[0].1 - 1.0).abs() < 1e-9);
        assert_eq!(stats.fill_rates[0].2, 1); // 1 observation
    }

    #[test]
    fn test_latch_adjustment_insufficient_observations_returns_zero() {
        let mut tracker = ReconcileOutcomeTracker::new();

        // Only 5 observations, below MIN_OBS threshold of 20
        for i in 0..5 {
            tracker.record_decision(make_decision(i, ReconcileActionType::Latch, 0.5));
            tracker.record_fill(i, 1.0);
        }

        let adj = tracker.suggested_latch_adjustment_bps();
        assert!(
            adj.abs() < 1e-9,
            "should return 0.0 with insufficient observations: {}",
            adj
        );
    }

    #[test]
    fn test_duplicate_oid_overwrites_pending() {
        let mut tracker = ReconcileOutcomeTracker::new();

        tracker.record_decision(make_decision(1, ReconcileActionType::Latch, 0.3));
        tracker.record_decision(make_decision(1, ReconcileActionType::CancelPlace, 0.9));

        // Should have overwritten: only 1 pending
        assert_eq!(tracker.pending_count(), 1);

        // Fill should use the latest action type (CancelPlace)
        tracker.record_fill(1, 1.5);

        // CancelPlace rate should have been updated (first obs = 1.0)
        assert!(
            (tracker.fill_rate_for_action(&ReconcileActionType::CancelPlace) - 1.0).abs() < 1e-9
        );
        // Latch rate should remain at default 0.5
        assert!((tracker.fill_rate_for_action(&ReconcileActionType::Latch) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_fill_for_unknown_oid_is_noop() {
        let mut tracker = ReconcileOutcomeTracker::new();

        // Fill an oid that was never recorded
        tracker.record_fill(999, 1.0);
        assert_eq!(tracker.total_fills, 0);
        assert_eq!(tracker.pending_count(), 0);
    }
}
