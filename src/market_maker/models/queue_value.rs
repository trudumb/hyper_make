//! Queue value heuristic for level-by-level quote filtering.
//!
//! Estimates the expected value of quoting at a given depth level:
//!
//! ```text
//! queue_value(depth_bps, toxicity, features) =
//!     depth_bps                           // spread captured
//!   - expected_AS(toxicity)               // regime-dependent: 1/3/8 bps
//!   - queue_penalty(queue_rank)           // 30% worse at back of queue
//!   - maker_fee_bps                       // 1.5 bps on HL
//! ```
//!
//! Negative queue value → don't quote that level. Online learning via
//! `observe_outcome()` from markout data refines the heuristic over time.

use crate::market_maker::adverse_selection::toxicity_regime::ToxicityRegime;
use crate::market_maker::config::auto_derive::CapitalTier;

/// Maker fee on Hyperliquid in basis points.
const MAKER_FEE_BPS: f64 = 1.5;

/// Expected AS cost by toxicity regime (basis points).
const AS_COST_BENIGN_BPS: f64 = 1.0;
const AS_COST_NORMAL_BPS: f64 = 3.0;
const AS_COST_TOXIC_BPS: f64 = 8.0;

/// Queue rank penalty: quoting at back of queue costs up to 30% of spread.
const QUEUE_PENALTY_MAX: f64 = 0.30;

/// Queue value heuristic with online learning.
#[derive(Debug, Clone)]
pub struct QueueValueHeuristic {
    /// EWMA of prediction errors for bias correction
    prediction_bias_bps: f64,
    /// Number of outcome observations
    outcome_count: usize,
    /// EWMA smoothing factor
    alpha: f64,
}

impl Default for QueueValueHeuristic {
    fn default() -> Self {
        Self::new()
    }
}

impl QueueValueHeuristic {
    pub fn new() -> Self {
        Self {
            prediction_bias_bps: 0.0,
            outcome_count: 0,
            alpha: 0.05,
        }
    }

    /// Compute queue value for a ladder level.
    ///
    /// Returns the expected edge in bps for quoting at `depth_bps` from mid.
    /// Negative → don't quote that level.
    ///
    /// - `depth_bps`: Distance from mid to quote level in basis points
    /// - `toxicity`: Current toxicity regime
    /// - `queue_rank`: Position in queue [0, 1] (0 = front)
    pub fn queue_value(
        &self,
        depth_bps: f64,
        toxicity: ToxicityRegime,
        queue_rank: f64,
    ) -> f64 {
        let expected_as = match toxicity {
            ToxicityRegime::Benign => AS_COST_BENIGN_BPS,
            ToxicityRegime::Normal => AS_COST_NORMAL_BPS,
            ToxicityRegime::Toxic => AS_COST_TOXIC_BPS,
        };

        let queue_penalty = depth_bps * QUEUE_PENALTY_MAX * queue_rank.clamp(0.0, 1.0);

        let raw_value = depth_bps - expected_as - queue_penalty - MAKER_FEE_BPS;

        // Apply learned bias correction (positive bias = we overpredict value)
        raw_value - self.prediction_bias_bps
    }

    /// Returns true if quoting at this depth has positive expected value.
    ///
    /// During warmup (< 10 observations), uses a lenient threshold of -1.0 bps
    /// to avoid filtering orders before the heuristic has calibrated. This prevents
    /// the "no fills → no calibration → no fills" death spiral.
    pub fn should_quote(
        &self,
        depth_bps: f64,
        toxicity: ToxicityRegime,
        queue_rank: f64,
    ) -> bool {
        let threshold = if self.outcome_count < 10 { -1.0 } else { 0.0 };
        self.queue_value(depth_bps, toxicity, queue_rank) > threshold
    }

    /// Observe a fill outcome for online learning.
    ///
    /// - `predicted_value_bps`: The queue value we predicted when quoting
    /// - `realized_pnl_bps`: Actual PnL from the fill (markout - fees)
    pub fn observe_outcome(&mut self, predicted_value_bps: f64, realized_pnl_bps: f64) {
        let error = predicted_value_bps - realized_pnl_bps;
        self.prediction_bias_bps =
            self.alpha * error + (1.0 - self.alpha) * self.prediction_bias_bps;
        self.outcome_count += 1;
    }

    /// Number of outcomes observed.
    pub fn outcome_count(&self) -> usize {
        self.outcome_count
    }

    /// Current prediction bias in bps.
    pub fn prediction_bias_bps(&self) -> f64 {
        self.prediction_bias_bps
    }

    /// Returns the minimum depth (bps) that produces positive queue value.
    ///
    /// QueueValue breakeven: depth > expected_AS + maker_fee + prediction_bias + safety_margin
    pub fn minimum_viable_depth(&self, toxicity: ToxicityRegime) -> f64 {
        let expected_as = match toxicity {
            ToxicityRegime::Benign => AS_COST_BENIGN_BPS,
            ToxicityRegime::Normal => AS_COST_NORMAL_BPS,
            ToxicityRegime::Toxic => AS_COST_TOXIC_BPS,
        };
        let warmup_buffer = if self.outcome_count < 10 { 1.0 } else { 0.0 };
        expected_as + MAKER_FEE_BPS + self.prediction_bias_bps + 0.5 - warmup_buffer
    }

    /// Create a QueueValueHeuristic calibrated for the given capital tier.
    ///
    /// Small capital tiers quote at wider depths (GLFT-optimal, not BBO),
    /// so they face lower adverse selection costs.
    pub fn for_capital_tier(tier: CapitalTier) -> Self {
        let h = Self::new();
        // For Micro/Small tiers, we don't adjust the heuristic itself —
        // the fix is in the edge evaluation depth (quote_engine passes GLFT depth
        // instead of BBO half-spread). The heuristic remains universal.
        // This method exists as a future extension point.
        let _ = tier;
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_value_benign_front_of_queue() {
        let heuristic = QueueValueHeuristic::new();
        // 8 bps depth, benign, front of queue
        // 8 - 1.0 - 0 - 1.5 = 5.5 bps
        let value = heuristic.queue_value(8.0, ToxicityRegime::Benign, 0.0);
        assert!((value - 5.5).abs() < 1e-10);
        assert!(heuristic.should_quote(8.0, ToxicityRegime::Benign, 0.0));
    }

    #[test]
    fn test_negative_value_toxic() {
        let heuristic = QueueValueHeuristic::new();
        // 4 bps depth, toxic regime, middle of queue
        // 4 - 8 - (4 * 0.3 * 0.5) - 1.5 = 4 - 8 - 0.6 - 1.5 = -6.1
        let value = heuristic.queue_value(4.0, ToxicityRegime::Toxic, 0.5);
        assert!(value < 0.0);
        assert!(!heuristic.should_quote(4.0, ToxicityRegime::Toxic, 0.5));
    }

    #[test]
    fn test_queue_rank_penalty() {
        let heuristic = QueueValueHeuristic::new();
        let front = heuristic.queue_value(10.0, ToxicityRegime::Normal, 0.0);
        let back = heuristic.queue_value(10.0, ToxicityRegime::Normal, 1.0);
        // Back of queue should be worse
        assert!(front > back);
        // Difference should be 10 * 0.3 * 1.0 = 3.0 bps
        assert!((front - back - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_online_learning_bias() {
        let mut heuristic = QueueValueHeuristic::new();

        // Consistently overpredict by 2 bps
        for _ in 0..20 {
            heuristic.observe_outcome(5.0, 3.0); // predicted 5, realized 3
        }

        // Bias should be positive (we overpredict)
        assert!(heuristic.prediction_bias_bps() > 0.0);

        // Same level should now have lower predicted value
        let value_before = QueueValueHeuristic::new().queue_value(8.0, ToxicityRegime::Benign, 0.0);
        let value_after = heuristic.queue_value(8.0, ToxicityRegime::Benign, 0.0);
        assert!(value_after < value_before);
    }

    #[test]
    fn test_marginal_level_toxic_filtered() {
        let heuristic = QueueValueHeuristic::new();
        // 3 bps depth in toxic regime should be negative
        assert!(!heuristic.should_quote(3.0, ToxicityRegime::Toxic, 0.0));
        // But same level in benign should be positive
        assert!(heuristic.should_quote(3.0, ToxicityRegime::Benign, 0.0));
    }

    #[test]
    fn test_wide_level_always_quoted() {
        let heuristic = QueueValueHeuristic::new();
        // 20 bps depth should be positive even in toxic, back of queue
        // 20 - 8 - (20 * 0.3 * 1.0) - 1.5 = 20 - 8 - 6 - 1.5 = 4.5
        assert!(heuristic.should_quote(20.0, ToxicityRegime::Toxic, 1.0));
    }

    #[test]
    fn test_minimum_viable_depth_normal() {
        let heuristic = QueueValueHeuristic::new();
        let depth = heuristic.minimum_viable_depth(ToxicityRegime::Normal);
        // During warmup (0 observations): 3.0 (AS) + 1.5 (fee) + 0.0 (bias) + 0.5 (safety) - 1.0 (warmup buffer) = 4.0
        assert!((depth - 4.0).abs() < 1e-10, "depth={}", depth);
    }

    #[test]
    fn test_minimum_viable_depth_normal_calibrated() {
        let mut heuristic = QueueValueHeuristic::new();
        // Simulate 10+ observations to exit warmup
        for _ in 0..10 {
            heuristic.observe_outcome(5.0, 5.0); // zero-bias observations
        }
        let depth = heuristic.minimum_viable_depth(ToxicityRegime::Normal);
        // Post-warmup: 3.0 (AS) + 1.5 (fee) + ~0.0 (bias) + 0.5 (safety) = 5.0
        assert!((depth - 5.0).abs() < 0.1, "depth={}", depth);
    }

    #[test]
    fn test_minimum_viable_depth_toxic() {
        let heuristic = QueueValueHeuristic::new();
        let depth = heuristic.minimum_viable_depth(ToxicityRegime::Toxic);
        // During warmup: 8.0 (AS) + 1.5 (fee) + 0.0 (bias) + 0.5 (safety) - 1.0 (warmup buffer) = 9.0
        assert!((depth - 9.0).abs() < 1e-10, "depth={}", depth);
    }

    #[test]
    fn test_queue_value_positive_at_5_bps_normal() {
        let heuristic = QueueValueHeuristic::new();
        // 5 - 3 - 0 - 1.5 = 0.5 > 0
        assert!(heuristic.should_quote(5.0, ToxicityRegime::Normal, 0.0));
    }

    #[test]
    fn test_for_capital_tier_wider_depth_positive_value() {
        let heuristic = QueueValueHeuristic::for_capital_tier(CapitalTier::Micro);
        // At GLFT-optimal depth (5 bps) instead of BBO (1 bps), value is positive:
        // 5 - 3(AS) - 0 - 1.5(fee) = 0.5 bps > 0
        assert!(heuristic.should_quote(5.0, ToxicityRegime::Normal, 0.0));
        // But at BBO depth (1 bps), still negative:
        // 1 - 3 - 0 - 1.5 = -3.5
        assert!(!heuristic.should_quote(1.0, ToxicityRegime::Normal, 0.0));
    }
}
