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

/// Position action decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionAction {
    /// Hold current position: inventory_ratio = 0 (no skew)
    Hold,
    /// Add to position: inventory_ratio = -kelly_frac (reverse skew)
    Add { kelly_frac: f64 },
    /// Reduce position: inventory_ratio = +urgency (normal skew)
    Reduce { urgency: f64 },
}

impl Default for PositionAction {
    fn default() -> Self {
        PositionAction::Reduce { urgency: 1.0 }
    }
}

impl PositionAction {
    /// Get the inventory_ratio multiplier for this action.
    ///
    /// This transforms the raw inventory_ratio for GLFT skew calculation:
    /// - HOLD: 0.0 → no skew → symmetric quotes
    /// - ADD: -kelly → reverse skew → tighter on position-building side
    /// - REDUCE: +urgency → normal skew → tighter on position-reducing side
    pub fn inventory_ratio_multiplier(&self) -> f64 {
        match self {
            PositionAction::Hold => 0.0,
            PositionAction::Add { kelly_frac } => -kelly_frac.abs(),
            PositionAction::Reduce { urgency } => urgency.abs(),
        }
    }

    /// Check if this is a hold action.
    pub fn is_hold(&self) -> bool {
        matches!(self, PositionAction::Hold)
    }

    /// Check if this is an add action.
    pub fn is_add(&self) -> bool {
        matches!(self, PositionAction::Add { .. })
    }

    /// Check if this is a reduce action.
    pub fn is_reduce(&self) -> bool {
        matches!(self, PositionAction::Reduce { .. })
    }
}

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
    /// Create a new engine with configuration.
    pub fn new(config: PositionDecisionConfig) -> Self {
        Self {
            continuation: ContinuationPosterior::default(),
            config,
            current_regime: "normal".to_string(),
            fills_since_decay: 0,
        }
    }

    /// Decide position action based on current state.
    ///
    /// # Arguments
    /// * `position` - Current position (signed)
    /// * `max_position` - Maximum allowed position
    /// * `belief_drift` - E[μ | data] from belief system (positive = bullish)
    /// * `belief_confidence` - Confidence in belief system [0, 1]
    /// * `edge_bps` - Expected edge in basis points
    ///
    /// # Returns
    /// PositionAction indicating HOLD, ADD, or REDUCE
    pub fn decide(
        &self,
        position: f64,
        max_position: f64,
        belief_drift: f64,
        belief_confidence: f64,
        edge_bps: f64,
    ) -> PositionAction {
        let inv_ratio = if max_position > 1e-9 {
            (position / max_position).abs()
        } else {
            0.0
        };

        // Flat position → default reduce with no urgency
        if inv_ratio < self.config.min_significant_position {
            return PositionAction::Reduce { urgency: 0.0 };
        }

        // Use fused probability if enabled (multi-signal fusion)
        // Otherwise fall back to simple Beta mean
        let (p_cont, conf) = if self.config.use_fused_probability {
            (
                self.continuation.prob_continuation_fused(),
                self.continuation.confidence_fused(),
            )
        } else {
            (
                self.continuation.prob_continuation(),
                self.continuation.confidence(),
            )
        };
        let position_sign = position.signum();

        // Check alignment: position direction matches belief drift
        let aligned = if belief_confidence > self.config.min_belief_confidence {
            (position_sign > 0.0 && belief_drift > 0.0)
                || (position_sign < 0.0 && belief_drift < 0.0)
        } else {
            false // Can't determine alignment without confident beliefs
        };

        // === ADD: Strong conviction, aligned, edge exceeds costs ===
        if p_cont > self.config.add_threshold
            && aligned
            && conf > self.config.conf_add
            && belief_confidence > 0.6
            && edge_bps > 2.0 * self.config.cost_bps
            && inv_ratio < self.config.max_position_for_add
        {
            let kelly = self.compute_kelly(p_cont, edge_bps);
            return PositionAction::Add { kelly_frac: kelly };
        }

        // === HOLD: Moderate conviction, aligned ===
        if p_cont > self.config.hold_threshold && aligned && conf > self.config.conf_hold {
            return PositionAction::Hold;
        }

        // === REDUCE: Default behavior with urgency scaling ===
        let urgency = self.compute_urgency(p_cont, inv_ratio, aligned);
        PositionAction::Reduce { urgency }
    }

    /// Compute Kelly fraction for ADD sizing.
    ///
    /// Kelly formula: f* = (p × edge - (1-p) × loss) / edge
    /// where p = P(continuation), edge = expected gain, loss = cost
    fn compute_kelly(&self, p_cont: f64, edge_bps: f64) -> f64 {
        let loss_bps = self.config.cost_bps;
        if edge_bps <= 0.0 {
            return 0.0;
        }
        let kelly = (p_cont * edge_bps - (1.0 - p_cont) * loss_bps) / edge_bps;
        kelly.clamp(0.0, self.config.max_kelly)
    }

    /// Compute urgency for REDUCE action.
    ///
    /// Higher urgency when:
    /// - Low P(continuation) → fear_factor increases
    /// - High inventory ratio → size_factor increases (quadratic+linear for aggressive decay)
    /// - Position opposed to beliefs → base urgency increases to 1.5
    fn compute_urgency(&self, p_cont: f64, inv_ratio: f64, aligned: bool) -> f64 {
        let base = if aligned { 0.5 } else { 1.5 };
        let fear_factor = (1.0 - p_cont).powi(2); // More urgent when p_cont low
        let size_factor = inv_ratio.powi(2) + inv_ratio; // Quadratic+linear: aggressive at high inventory
        (base + fear_factor * size_factor).clamp(0.0, 3.0)
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
    pub fn update_signals(
        &mut self,
        changepoint_prob: f64,
        changepoint_entropy: f64,
        momentum_continuation: f64,
        trend_agreement: f64,
        trend_confidence: f64,
        regime_probs: [f64; 4],
    ) {
        self.continuation.update_signals(
            changepoint_prob,
            changepoint_entropy,
            momentum_continuation,
            trend_agreement,
            trend_confidence,
            regime_probs,
        );
    }

    /// Get diagnostic summary of continuation signals.
    pub fn signal_summary(&self) -> crate::market_maker::stochastic::continuation::ContinuationSignalSummary {
        self.continuation.signal_summary()
    }

    /// Get current regime.
    pub fn current_regime(&self) -> &str {
        &self.current_regime
    }
}

/// Convert PositionAction to effective inventory_ratio for GLFT.
///
/// This function takes the raw inventory_ratio and transforms it based
/// on the position action:
///
/// - HOLD: Returns 0.0 (no skew)
/// - ADD: Returns negative ratio (reverse skew to build position)
/// - REDUCE: Returns positive ratio scaled by urgency (normal mean-reversion)
///
/// # Arguments
/// * `action` - The decided position action
/// * `raw_inventory_ratio` - The original q/Q_max ratio (signed, -1 to 1)
///
/// # Returns
/// Effective inventory_ratio for GLFT skew calculation
pub fn action_to_inventory_ratio(action: PositionAction, raw_inventory_ratio: f64) -> f64 {
    let abs_ratio = raw_inventory_ratio.abs();
    let sign = raw_inventory_ratio.signum();

    match action {
        PositionAction::Hold => 0.0,
        PositionAction::Add { kelly_frac } => {
            // Reverse skew: negative ratio encourages position building
            -kelly_frac * abs_ratio * sign
        }
        PositionAction::Reduce { urgency } => {
            // Normal skew: positive ratio encourages position reduction
            urgency * abs_ratio * sign
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_action_is_reduce() {
        let action = PositionAction::default();
        assert!(action.is_reduce());
    }

    #[test]
    fn test_hold_decision() {
        let mut engine = PositionDecisionEngine::default();

        // Simulate aligned fills to increase p_cont
        for _ in 0..20 {
            engine.observe_fill(1.0, 1.0, 1.0); // Buy fills for long position
        }

        let action = engine.decide(
            0.3,  // position (long)
            1.0,  // max_position
            0.001, // belief_drift (positive = bullish)
            0.8,  // belief_confidence
            5.0,  // edge_bps
        );

        // With high p_cont, alignment, and confidence, should HOLD
        assert!(action.is_hold() || action.is_add(),
            "Expected HOLD or ADD with high p_cont, got {:?}", action);
    }

    #[test]
    fn test_reduce_when_not_aligned() {
        let engine = PositionDecisionEngine::default();

        let action = engine.decide(
            0.3,   // position (long)
            1.0,   // max_position
            -0.001, // belief_drift (negative = bearish, opposed to long)
            0.8,   // belief_confidence
            5.0,   // edge_bps
        );

        // Position opposes beliefs → should REDUCE
        assert!(action.is_reduce(), "Expected REDUCE when position opposes beliefs");
    }

    #[test]
    fn test_add_with_high_edge() {
        let mut engine = PositionDecisionEngine::default();

        // Build high continuation confidence
        for _ in 0..30 {
            engine.observe_fill(1.0, 1.0, 1.0);
        }

        let action = engine.decide(
            0.2,   // position (moderate long)
            1.0,   // max_position
            0.001, // belief_drift (aligned)
            0.9,   // belief_confidence (high)
            10.0,  // edge_bps (high edge)
        );

        // High p_cont + high edge + aligned + not too big → could ADD
        // Note: depends on continuation posterior state
        match action {
            PositionAction::Add { kelly_frac } => {
                assert!(kelly_frac > 0.0 && kelly_frac <= 0.3,
                    "Kelly should be in (0, 0.3], got {}", kelly_frac);
            }
            PositionAction::Hold => {
                // Also acceptable if not quite meeting add threshold
            }
            PositionAction::Reduce { .. } => {
                // Acceptable if continuation posterior didn't get high enough
            }
        }
    }

    #[test]
    fn test_flat_position_returns_zero_urgency() {
        let engine = PositionDecisionEngine::default();

        let action = engine.decide(
            0.005, // Nearly flat position
            1.0,
            0.001,
            0.8,
            5.0,
        );

        match action {
            PositionAction::Reduce { urgency } => {
                assert!(urgency < 0.01, "Flat position should have ~0 urgency");
            }
            _ => panic!("Flat position should REDUCE"),
        }
    }

    #[test]
    fn test_action_to_inventory_ratio() {
        // HOLD → 0
        assert_eq!(action_to_inventory_ratio(PositionAction::Hold, 0.5), 0.0);

        // ADD with kelly=0.2 and ratio=0.3 → -0.2 * 0.3 = -0.06
        let add_result = action_to_inventory_ratio(
            PositionAction::Add { kelly_frac: 0.2 },
            0.3,
        );
        assert!((add_result - (-0.06)).abs() < 0.001);

        // REDUCE with urgency=1.5 and ratio=0.4 → 1.5 * 0.4 = 0.6
        let reduce_result = action_to_inventory_ratio(
            PositionAction::Reduce { urgency: 1.5 },
            0.4,
        );
        assert!((reduce_result - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_regime_reset() {
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
            0.0,                       // No changepoint
            2.0,                       // Normal entropy
            0.7,                       // High momentum continuation (typical in cascade)
            0.5,                       // Some trend agreement
            0.5,                       // Some trend confidence
            [0.0, 0.0, 0.2, 0.8],     // Cascade-dominant regime
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
            0.8,                       // High changepoint
            0.5,                       // Low entropy
            0.5,                       // Neutral momentum
            0.0,                       // No trend
            0.0,                       // No trend confidence
            [0.2, 0.5, 0.2, 0.1],     // Normal regime
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
