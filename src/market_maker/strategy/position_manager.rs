//! Position Decision Engine - HOLD/ADD/REDUCE for informed position management.
//!
//! This module implements the core logic for deciding whether to hold, add to,
//! or reduce a position based on Bayesian continuation probability.
//!
//! ## Problem
//!
//! GLFT inventory skew formula always mean-reverts:
//! ```text
//! skew = inventory_ratio × γ × σ² × T
//! ```
//! With q > 0, skew > 0, meaning we always quote tighter on the ASK side.
//! This leads to repeated fills that close the position prematurely.
//!
//! ## Solution
//!
//! The Position Continuation Model transforms inventory_ratio based on
//! Bayesian P(continuation):
//!
//! | Action | inventory_ratio | When |
//! |--------|-----------------|------|
//! | HOLD | 0.0 | p_cont > 0.50, aligned, conf > 0.6 |
//! | ADD | -kelly × sign | p_cont > 0.65, aligned, conf > 0.7, edge > 2×costs |
//! | REDUCE | +urgency × sign | Default (current behavior) |

use crate::market_maker::stochastic::ContinuationPosterior;

/// Configuration for PositionDecisionEngine.
#[derive(Debug, Clone)]
pub struct PositionDecisionConfig {
    /// P(continuation) threshold for HOLD action (default: 0.50)
    pub hold_threshold: f64,
    /// P(continuation) threshold for ADD action (default: 0.65)
    pub add_threshold: f64,
    /// Minimum confidence for HOLD action (default: 0.60)
    pub conf_hold: f64,
    /// Minimum confidence for ADD action (default: 0.7)
    pub conf_add: f64,
    /// Cost in bps for Kelly calculation (spread + fees) (default: 3.0)
    pub cost_bps: f64,
    /// Maximum Kelly fraction (default: 0.3)
    pub max_kelly: f64,
    /// Decay half-life in fills (default: 20.0)
    pub decay_half_life: f64,
    /// Minimum inventory ratio to consider position significant (default: 0.01)
    pub min_significant_position: f64,
    /// Maximum inventory ratio to consider adding (default: 0.5)
    pub max_position_for_add: f64,
    /// Minimum belief confidence to use for alignment (default: 0.3)
    pub min_belief_confidence: f64,
    /// Use fused continuation probability (multi-signal fusion) (default: true)
    pub use_fused_probability: bool,
}

impl Default for PositionDecisionConfig {
    fn default() -> Self {
        Self {
            // P0 FIX (2026-02-02): Lower hold_threshold to reduce premature REDUCE actions
            // Analysis showed 10.84 bps adverse selection from exiting trends too early
            hold_threshold: 0.50, // was 0.55 - allows more HOLD during aligned trends
            add_threshold: 0.65,
            // P0 FIX (2026-02-02): Raise conf_hold to require more signal confidence
            // This prevents flipping on noisy belief updates
            conf_hold: 0.60, // was 0.50 - requires higher confidence for HOLD
            conf_add: 0.7,
            cost_bps: 3.0,
            max_kelly: 0.3,
            decay_half_life: 20.0,
            min_significant_position: 0.01,
            max_position_for_add: 0.5,
            min_belief_confidence: 0.3,
            use_fused_probability: true, // Use enhanced multi-signal fusion by default
        }
    }
}

/// Position Decision Engine - determines HOLD/ADD/REDUCE based on Bayesian continuation.
#[derive(Debug, Clone)]
pub struct PositionDecisionEngine {
    /// Beta-Binomial posterior for continuation probability
    pub continuation: ContinuationPosterior,
    /// Configuration
    pub config: PositionDecisionConfig,
    /// Current regime name (for reset)
    current_regime: String,
    /// Fill count since last decay
    fills_since_decay: u64,
}

impl Default for PositionDecisionEngine {
    fn default() -> Self {
        Self::new(PositionDecisionConfig::default())
    }
}

impl PositionDecisionEngine {
    /// Create new PositionDecisionEngine with specific config
    pub fn new(config: PositionDecisionConfig) -> Self {
        Self {
            continuation: ContinuationPosterior::new(5.0, 5.0),
            config,
            current_regime: String::from("normal"),
            fills_since_decay: 0,
        }
    }

    /// Update posterior with fill observation.
    ///
    /// # Arguments
    /// * `fill_side_sign` - +1 for buy fill, -1 for sell fill
    /// * `position_sign` - +1 for long position, -1 for short
    /// * `size` - Fill size (used for weight calculation)
    pub fn observe_fill(&mut self, fill_side_sign: f64, position_sign: f64, size: f64) {
        let is_aligned = fill_side_sign * position_sign > 0.0;
        let weight = size.sqrt().clamp(0.1, 3.0); // Size-weighted update
        self.continuation.observe_fill(is_aligned, weight);

        // Apply decay periodically
        self.fills_since_decay += 1;
        if self.fills_since_decay >= 5 {
            self.continuation.decay(self.config.decay_half_life);
            self.fills_since_decay = 0;
        }
    }

    /// Reset for new regime.
    pub fn reset_for_regime(&mut self, regime: &str) {
        self.continuation.reset_to_regime(regime);
        self.current_regime = regime.to_string();
        self.fills_since_decay = 0;
    }

    /// Get current P(continuation).
    ///
    /// Returns fused probability if configured, otherwise simple Beta mean.
    pub fn prob_continuation(&self) -> f64 {
        if self.config.use_fused_probability {
            self.continuation.prob_continuation_fused()
        } else {
            self.continuation.prob_continuation()
        }
    }

    /// Get current confidence.
    ///
    /// Returns fused confidence if configured, otherwise variance-based.
    pub fn confidence(&self) -> f64 {
        if self.config.use_fused_probability {
            self.continuation.confidence_fused()
        } else {
            self.continuation.confidence()
        }
    }

    /// Update external signals for multi-signal fusion.
    ///
    /// This should be called each quote cycle with the latest signal values
    /// from BOCD, momentum model, trend tracker, and HMM.
    ///
    /// # Arguments
    /// * `changepoint_prob` - From BOCD `changepoint_probability(5)`
    /// * `changepoint_entropy` - From BOCD `run_length_entropy()`
    /// * `momentum_continuation` - From MomentumModel `continuation_probability()`
    /// * `trend_agreement` - From TrendSignal `timeframe_agreement`
    /// * `trend_confidence` - From TrendSignal `trend_confidence`
    /// * `regime_probs` - From HMM `regime_probabilities()` [quiet, normal, bursty, cascade]
    /// * `trend_position_alignment` - [-1,1] trend alignment with position direction
    #[allow(clippy::too_many_arguments)] // Signal passthrough; struct refactor deferred
    pub fn update_signals(
        &mut self,
        changepoint_prob: f64,
        changepoint_entropy: f64,
        momentum_continuation: f64,
        trend_agreement: f64,
        trend_confidence: f64,
        regime_probs: [f64; 4],
        trend_position_alignment: f64,
    ) {
        self.continuation.update_signals(
            changepoint_prob,
            changepoint_entropy,
            momentum_continuation,
            trend_agreement,
            trend_confidence,
            regime_probs,
            trend_position_alignment,
        );
    }

    /// Get diagnostic summary of continuation signals.
    pub fn signal_summary(
        &self,
    ) -> crate::market_maker::stochastic::continuation::ContinuationSignalSummary {
        self.continuation.signal_summary()
    }

    /// Get current regime.
    pub fn current_regime(&self) -> &str {
        &self.current_regime
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_regime_reset() {
        let mut engine = PositionDecisionEngine::default();

        // Build up some state
        for _ in 0..10 {
            engine.observe_fill(1.0, 1.0, 1.0);
        }
        let _p_before = engine.prob_continuation();

        // Reset to cascade regime
        engine.reset_for_regime("cascade");

        // Also update signals to reflect cascade regime
        // (in production, this would come from HMM and other signal sources)
        engine.update_signals(
            0.0,                  // No changepoint
            2.0,                  // Normal entropy
            0.7,                  // High momentum continuation (typical in cascade)
            0.5,                  // Some trend agreement
            0.5,                  // Some trend confidence
            [0.0, 0.0, 0.2, 0.8], // Cascade-dominant regime
            0.0,                  // Neutral alignment
        );

        // Should have elevated continuation probability reflecting cascade regime
        let p_after = engine.prob_continuation();
        assert!(
            p_after > 0.6,
            "Cascade regime with supportive signals should have elevated prior: {}",
            p_after
        );
        assert_eq!(engine.current_regime(), "cascade");
    }

    #[test]
    fn test_signal_update_affects_decision() {
        let mut engine = PositionDecisionEngine::default();

        // Build up some fills
        for _ in 0..15 {
            engine.observe_fill(1.0, 1.0, 1.0);
        }
        let p_before = engine.prob_continuation();

        // Simulate changepoint detection
        engine.update_signals(
            0.8,                  // High changepoint
            0.5,                  // Low entropy
            0.5,                  // Neutral momentum
            0.0,                  // No trend
            0.0,                  // No trend confidence
            [0.2, 0.5, 0.2, 0.1], // Normal regime
            0.0,                  // Neutral alignment
        );

        let p_after = engine.prob_continuation();

        // Changepoint should reduce continuation probability
        assert!(
            p_after < p_before,
            "Changepoint should reduce p_cont: before={}, after={}",
            p_before,
            p_after
        );
    }
}
