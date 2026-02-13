//! Regime state machine — single source of truth for all regime-dependent parameters.
//!
//! The `RegimeState` is updated early in each quote cycle from HMM probabilities,
//! BOCPD changepoint detection, and the regime kappa estimator. All downstream
//! components read regime-dependent parameters from `RegimeState::params()` rather
//! than computing their own regime adjustments.

use serde::{Deserialize, Serialize};

/// Controller objective — determines whether the controller optimizes for
/// spread capture (mean-reverting) or inventory minimization (trending/toxic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ControllerObjective {
    /// Tight spread, high skew gain, maximize spread capture.
    #[default]
    MeanRevert,
    /// Wide spread, minimize inventory, defensive posture.
    TrendingToxic,
}

/// Discrete market regime — drives all regime-dependent parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    Calm,     // Low vol, mean-reverting
    Normal,   // Typical conditions
    Volatile, // Elevated vol, trending possible
    Extreme,  // Cascade / liquidation regime
}

impl Default for MarketRegime {
    fn default() -> Self {
        Self::Normal
    }
}

impl MarketRegime {
    /// Map from HMM regime index (0=low, 1=normal, 2=high, 3=extreme).
    pub fn from_hmm_index(idx: usize) -> Self {
        match idx {
            0 => Self::Calm,
            1 => Self::Normal,
            2 => Self::Volatile,
            _ => Self::Extreme,
        }
    }

    /// Return the default `RegimeParams` for this regime.
    pub fn default_params(self) -> RegimeParams {
        match self {
            Self::Calm => RegimeParams {
                kappa: 3000.0,
                spread_floor_bps: 3.0,
                skew_gain: 1.5,
                max_position_fraction: 1.0,
                emergency_cp_threshold: 0.9,
                reduce_only_fraction: 0.7,
                size_multiplier: 1.0,
                as_expected_bps: 1.0,
                risk_premium_bps: 0.5,
                controller_objective: ControllerObjective::MeanRevert,
            },
            Self::Normal => RegimeParams {
                kappa: 2000.0,
                spread_floor_bps: 5.0,
                skew_gain: 1.0,
                max_position_fraction: 0.8,
                emergency_cp_threshold: 0.8,
                reduce_only_fraction: 0.5,
                size_multiplier: 0.8,
                as_expected_bps: 1.0,
                risk_premium_bps: 1.0,
                controller_objective: ControllerObjective::MeanRevert,
            },
            Self::Volatile => RegimeParams {
                kappa: 1000.0,
                spread_floor_bps: 10.0,
                skew_gain: 0.5,
                max_position_fraction: 0.5,
                emergency_cp_threshold: 0.6,
                reduce_only_fraction: 0.3,
                size_multiplier: 0.5,
                as_expected_bps: 3.0,
                risk_premium_bps: 3.0,
                controller_objective: ControllerObjective::TrendingToxic,
            },
            Self::Extreme => RegimeParams {
                kappa: 500.0,
                spread_floor_bps: 20.0,
                skew_gain: 0.3,
                max_position_fraction: 0.3,
                emergency_cp_threshold: 0.4,
                reduce_only_fraction: 0.2,
                size_multiplier: 0.3,
                as_expected_bps: 5.0,
                risk_premium_bps: 6.0,
                controller_objective: ControllerObjective::TrendingToxic,
            },
        }
    }
}

/// Regime-conditioned parameters — every value here is regime-dependent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeParams {
    /// Regime-conditioned kappa for GLFT spread formula.
    /// Higher kappa = tighter spreads. kappa > 0.0 invariant.
    pub kappa: f64,
    /// Minimum spread floor for this regime (bps).
    pub spread_floor_bps: f64,
    /// How aggressively to skew quotes (0.5 = conservative, 2.0 = aggressive).
    pub skew_gain: f64,
    /// Max position as fraction of config.max_position (0.3 = 30%).
    pub max_position_fraction: f64,
    /// Changepoint probability threshold for triggering emergency pull.
    pub emergency_cp_threshold: f64,
    /// Position fraction above which reduce-only mode activates.
    pub reduce_only_fraction: f64,
    /// Quote size scaling factor (0.3 = 30% of normal).
    pub size_multiplier: f64,
    /// Expected adverse selection cost for this regime (bps).
    /// Updated via EWMA from markout measurements.
    #[serde(default)]
    pub as_expected_bps: f64,
    /// Additive risk premium for this regime (bps).
    #[serde(default)]
    pub risk_premium_bps: f64,
    /// Controller objective for this regime.
    #[serde(default)]
    pub controller_objective: ControllerObjective,
}

impl RegimeParams {
    /// EWMA update of expected adverse selection from markout measurements.
    ///
    /// Blends new markout AS observation into the regime's running estimate.
    /// Uses a conservative EWMA weight (0.05) to avoid overreacting to noise.
    pub fn update_as_from_markout(&mut self, markout_as_bps: f64) {
        const EWMA_WEIGHT: f64 = 0.05;
        let clamped = markout_as_bps.clamp(0.0, 50.0); // Sanity: AS can't be negative or > 50 bps
        self.as_expected_bps = (1.0 - EWMA_WEIGHT) * self.as_expected_bps + EWMA_WEIGHT * clamped;
    }
}

impl Default for RegimeParams {
    fn default() -> Self {
        MarketRegime::Normal.default_params()
    }
}

/// Number of consecutive cycles required before a regime transition is accepted.
const HYSTERESIS_CYCLES: u32 = 5;

/// Regime state machine with hysteresis to prevent oscillation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeState {
    /// Currently active regime.
    #[serde(default)]
    pub regime: MarketRegime,
    /// Parameters derived from the active regime.
    #[serde(default)]
    pub params: RegimeParams,
    /// HMM posterior probability for the active regime.
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    /// Cycles remaining before a new transition is allowed.
    #[serde(default)]
    pub transition_cooldown_cycles: u32,
    /// Consecutive cycles the proposed regime has been dominant.
    #[serde(default)]
    consecutive_regime_count: u32,
    /// The proposed regime (may differ from active during hysteresis).
    #[serde(default)]
    proposed_regime: MarketRegime,
}

fn default_confidence() -> f64 {
    0.5
}

impl Default for RegimeState {
    fn default() -> Self {
        Self {
            regime: MarketRegime::Normal,
            params: MarketRegime::Normal.default_params(),
            confidence: 0.5,
            transition_cooldown_cycles: 0,
            consecutive_regime_count: 0,
            proposed_regime: MarketRegime::Normal,
        }
    }
}

impl RegimeState {
    /// Create a new RegimeState defaulting to Normal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the regime state from HMM probabilities and kappa estimator output.
    ///
    /// Returns `true` if the regime actually transitioned.
    ///
    /// # Arguments
    /// * `hmm_probs` - HMM posterior probabilities [p_low, p_normal, p_high, p_extreme]
    /// * `bocpd_cp` - BOCPD changepoint probability (0.0-1.0)
    /// * `kappa_effective` - Regime-blended kappa from signal integration
    pub fn update(
        &mut self,
        hmm_probs: &[f64; 4],
        bocpd_cp: f64,
        kappa_effective: f64,
    ) -> bool {
        // Determine proposed regime from HMM argmax
        let (max_idx, max_prob) = hmm_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((1, &0.25));

        let mut proposed = MarketRegime::from_hmm_index(max_idx);

        // Override: high changepoint probability forces Volatile or Extreme
        if bocpd_cp > 0.7 && proposed == MarketRegime::Calm {
            proposed = MarketRegime::Volatile;
        }
        if bocpd_cp > 0.9 {
            proposed = MarketRegime::Extreme;
        }

        // Tick cooldown
        if self.transition_cooldown_cycles > 0 {
            self.transition_cooldown_cycles -= 1;
        }

        // Hysteresis: count consecutive cycles in proposed regime
        if proposed == self.proposed_regime {
            self.consecutive_regime_count += 1;
        } else {
            self.proposed_regime = proposed;
            self.consecutive_regime_count = 1;
        }

        // Transition only when:
        // 1. Proposed differs from current
        // 2. Consecutive count >= HYSTERESIS_CYCLES
        // 3. Cooldown expired
        // Exception: transitions TO Extreme are always immediate (safety first)
        let should_transition = proposed != self.regime
            && (proposed == MarketRegime::Extreme
                || (self.consecutive_regime_count >= HYSTERESIS_CYCLES
                    && self.transition_cooldown_cycles == 0));

        if should_transition {
            self.regime = proposed;
            self.params = proposed.default_params();
            self.confidence = *max_prob;
            self.transition_cooldown_cycles = HYSTERESIS_CYCLES;
            self.consecutive_regime_count = 0;

            // Override kappa with kappa_effective when available and meaningful
            if kappa_effective > 0.0 {
                // Blend: 50% regime default + 50% observed kappa_effective
                self.params.kappa = 0.5 * self.params.kappa + 0.5 * kappa_effective;
                // Enforce kappa > 0.0 invariant
                self.params.kappa = self.params.kappa.max(1.0);
            }

            true
        } else {
            // Even without transition, update confidence and blend kappa
            self.confidence = *max_prob;

            if kappa_effective > 0.0 {
                // Slow blend toward kappa_effective (10% per cycle)
                self.params.kappa = 0.9 * self.params.kappa + 0.1 * kappa_effective;
                self.params.kappa = self.params.kappa.max(1.0);
            }

            false
        }
    }

    /// Get a human-readable label for the current regime.
    pub fn regime_label(&self) -> &'static str {
        match self.regime {
            MarketRegime::Calm => "Calm",
            MarketRegime::Normal => "Normal",
            MarketRegime::Volatile => "Volatile",
            MarketRegime::Extreme => "Extreme",
        }
    }

    /// Get the spread adjustment in bps relative to Normal baseline.
    /// Positive = wider than Normal, negative = tighter than Normal.
    pub fn spread_adjustment_bps(&self) -> f64 {
        self.params.spread_floor_bps - MarketRegime::Normal.default_params().spread_floor_bps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_normal() {
        let state = RegimeState::new();
        assert_eq!(state.regime, MarketRegime::Normal);
        assert!((state.params.kappa - 2000.0).abs() < f64::EPSILON);
        assert!((state.confidence - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hysteresis_prevents_premature_transition() {
        let mut state = RegimeState::new();

        // 4 cycles of Calm should NOT trigger transition (need 5)
        let calm_probs = [0.8, 0.1, 0.05, 0.05];
        for _ in 0..4 {
            let changed = state.update(&calm_probs, 0.0, 2000.0);
            assert!(!changed, "Should not transition before hysteresis threshold");
            assert_eq!(state.regime, MarketRegime::Normal);
        }

        // 5th cycle triggers transition
        let changed = state.update(&calm_probs, 0.0, 2000.0);
        assert!(changed, "Should transition after 5 consecutive cycles");
        assert_eq!(state.regime, MarketRegime::Calm);
    }

    #[test]
    fn test_extreme_transitions_immediately() {
        let mut state = RegimeState::new();

        // Extreme regime should bypass hysteresis (safety first)
        let extreme_probs = [0.0, 0.05, 0.1, 0.85];
        let changed = state.update(&extreme_probs, 0.0, 500.0);
        assert!(changed, "Extreme should transition immediately");
        assert_eq!(state.regime, MarketRegime::Extreme);
        assert!(state.params.kappa < 600.0, "Extreme kappa should be low");
    }

    #[test]
    fn test_bocpd_overrides_calm_to_volatile() {
        let mut state = RegimeState::new();

        // HMM says Calm but BOCPD says changepoint — should be Volatile
        let calm_probs = [0.8, 0.1, 0.05, 0.05];
        for _ in 0..5 {
            state.update(&calm_probs, 0.75, 2000.0);
        }
        assert_eq!(state.regime, MarketRegime::Volatile);
    }

    #[test]
    fn test_bocpd_extreme_override() {
        let mut state = RegimeState::new();

        // BOCPD > 0.9 forces Extreme regardless of HMM
        let normal_probs = [0.1, 0.8, 0.05, 0.05];
        let changed = state.update(&normal_probs, 0.95, 1000.0);
        assert!(changed, "BOCPD > 0.9 should force Extreme");
        assert_eq!(state.regime, MarketRegime::Extreme);
    }

    #[test]
    fn test_cooldown_prevents_oscillation() {
        let mut state = RegimeState::new();

        // Transition to Calm
        let calm_probs = [0.8, 0.1, 0.05, 0.05];
        for _ in 0..5 {
            state.update(&calm_probs, 0.0, 3000.0);
        }
        assert_eq!(state.regime, MarketRegime::Calm);

        // Immediately try to go back to Normal — cooldown should prevent
        // Cooldown was set to HYSTERESIS_CYCLES (5) on transition to Calm.
        // Each cycle ticks it down. Transition requires cooldown == 0 AND
        // consecutive count >= HYSTERESIS_CYCLES, so first 4 cycles should not transition.
        let normal_probs = [0.1, 0.8, 0.05, 0.05];
        for i in 0..4 {
            let cooldown_before = state.transition_cooldown_cycles;
            let changed = state.update(&normal_probs, 0.0, 2000.0);
            assert!(
                !changed,
                "Cycle {i}: cooldown was {cooldown_before}, should prevent transition"
            );
            assert_eq!(state.regime, MarketRegime::Calm, "Should still be Calm after cycle {i}");
        }
    }

    #[test]
    fn test_kappa_blending_during_steady_state() {
        let mut state = RegimeState::new();
        let initial_kappa = state.params.kappa;

        // Feed kappa_effective = 1500 for several cycles (no regime change)
        let normal_probs = [0.1, 0.8, 0.05, 0.05];
        for _ in 0..10 {
            state.update(&normal_probs, 0.0, 1500.0);
        }

        // Kappa should have blended toward 1500 (10% per cycle)
        assert!(
            state.params.kappa < initial_kappa,
            "Kappa should blend toward kappa_effective (1500 < 2000)"
        );
        assert!(
            state.params.kappa > 1400.0,
            "Kappa should still be above kappa_effective due to blending"
        );
    }

    #[test]
    fn test_kappa_always_positive() {
        let mut state = RegimeState::new();

        // Feed kappa_effective = 0.0 (invalid) — should be ignored
        let normal_probs = [0.1, 0.8, 0.05, 0.05];
        state.update(&normal_probs, 0.0, 0.0);
        assert!(state.params.kappa > 0.0, "Kappa must always be > 0.0");

        // Feed very small kappa
        for _ in 0..100 {
            state.update(&normal_probs, 0.0, 0.1);
        }
        assert!(
            state.params.kappa >= 1.0,
            "Kappa clamped to >= 1.0, got {}",
            state.params.kappa
        );
    }

    #[test]
    fn test_regime_params_ordering() {
        // Calm should have highest kappa (tightest spreads)
        // Extreme should have lowest kappa (widest spreads)
        let calm = MarketRegime::Calm.default_params();
        let normal = MarketRegime::Normal.default_params();
        let volatile = MarketRegime::Volatile.default_params();
        let extreme = MarketRegime::Extreme.default_params();

        assert!(calm.kappa > normal.kappa);
        assert!(normal.kappa > volatile.kappa);
        assert!(volatile.kappa > extreme.kappa);

        assert!(calm.spread_floor_bps < normal.spread_floor_bps);
        assert!(normal.spread_floor_bps < volatile.spread_floor_bps);
        assert!(volatile.spread_floor_bps < extreme.spread_floor_bps);

        assert!(calm.size_multiplier > extreme.size_multiplier);
        assert!(calm.max_position_fraction > extreme.max_position_fraction);
    }

    #[test]
    fn test_spread_adjustment_bps() {
        let mut state = RegimeState::new();

        // Normal regime: adjustment = 0 (baseline)
        assert!((state.spread_adjustment_bps()).abs() < f64::EPSILON);

        // Transition to Extreme
        let extreme_probs = [0.0, 0.05, 0.1, 0.85];
        state.update(&extreme_probs, 0.0, 500.0);
        assert!(
            state.spread_adjustment_bps() > 10.0,
            "Extreme should have positive spread adjustment"
        );
    }

    #[test]
    fn test_interrupted_transition_resets_count() {
        let mut state = RegimeState::new();

        // 3 cycles of Calm
        let calm_probs = [0.8, 0.1, 0.05, 0.05];
        for _ in 0..3 {
            state.update(&calm_probs, 0.0, 2000.0);
        }
        assert_eq!(state.regime, MarketRegime::Normal);

        // Interrupted by Volatile — counter should reset
        let volatile_probs = [0.05, 0.1, 0.8, 0.05];
        state.update(&volatile_probs, 0.0, 1000.0);

        // Now 4 more cycles of Calm should NOT be enough (counter was reset)
        for _ in 0..4 {
            state.update(&calm_probs, 0.0, 2000.0);
        }
        assert_eq!(
            state.regime,
            MarketRegime::Normal,
            "Interrupted transition should reset consecutive count"
        );

        // One more makes 5 from the reset
        let changed = state.update(&calm_probs, 0.0, 2000.0);
        assert!(changed);
        assert_eq!(state.regime, MarketRegime::Calm);
    }

    #[test]
    fn test_controller_objective_per_regime() {
        assert_eq!(
            MarketRegime::Calm.default_params().controller_objective,
            ControllerObjective::MeanRevert
        );
        assert_eq!(
            MarketRegime::Normal.default_params().controller_objective,
            ControllerObjective::MeanRevert
        );
        assert_eq!(
            MarketRegime::Volatile.default_params().controller_objective,
            ControllerObjective::TrendingToxic
        );
        assert_eq!(
            MarketRegime::Extreme.default_params().controller_objective,
            ControllerObjective::TrendingToxic
        );
    }

    #[test]
    fn test_as_expected_bps_ordering() {
        let calm = MarketRegime::Calm.default_params();
        let normal = MarketRegime::Normal.default_params();
        let volatile = MarketRegime::Volatile.default_params();
        let extreme = MarketRegime::Extreme.default_params();

        assert!(calm.as_expected_bps <= normal.as_expected_bps);
        assert!(normal.as_expected_bps < volatile.as_expected_bps);
        assert!(volatile.as_expected_bps < extreme.as_expected_bps);
    }

    #[test]
    fn test_risk_premium_ordering() {
        let calm = MarketRegime::Calm.default_params();
        let normal = MarketRegime::Normal.default_params();
        let volatile = MarketRegime::Volatile.default_params();
        let extreme = MarketRegime::Extreme.default_params();

        assert!(calm.risk_premium_bps <= normal.risk_premium_bps);
        assert!(normal.risk_premium_bps < volatile.risk_premium_bps);
        assert!(volatile.risk_premium_bps < extreme.risk_premium_bps);
    }

    #[test]
    fn test_update_as_from_markout() {
        let mut params = MarketRegime::Normal.default_params();
        let initial_as = params.as_expected_bps;

        // Feed high AS observation
        params.update_as_from_markout(10.0);
        assert!(
            params.as_expected_bps > initial_as,
            "AS should increase toward observation"
        );
        assert!(
            params.as_expected_bps < 10.0,
            "AS should not jump to observation (EWMA smoothing)"
        );

        // Feed many observations at 10.0 — should converge
        for _ in 0..200 {
            params.update_as_from_markout(10.0);
        }
        assert!(
            (params.as_expected_bps - 10.0).abs() < 0.1,
            "AS should converge to observation after many updates, got {}",
            params.as_expected_bps
        );
    }

    #[test]
    fn test_update_as_from_markout_clamps() {
        let mut params = MarketRegime::Normal.default_params();

        // Negative AS clamped to 0
        params.update_as_from_markout(-5.0);
        assert!(params.as_expected_bps >= 0.0);

        // Extreme AS clamped to 50
        for _ in 0..200 {
            params.update_as_from_markout(100.0);
        }
        assert!(
            params.as_expected_bps <= 50.0,
            "AS should be clamped to 50 bps, got {}",
            params.as_expected_bps
        );
    }

    #[test]
    fn test_regime_transition_preserves_objective() {
        let mut state = RegimeState::new();
        assert_eq!(
            state.params.controller_objective,
            ControllerObjective::MeanRevert
        );

        // Transition to Extreme
        let extreme_probs = [0.0, 0.05, 0.1, 0.85];
        state.update(&extreme_probs, 0.0, 500.0);
        assert_eq!(
            state.params.controller_objective,
            ControllerObjective::TrendingToxic
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let state = RegimeState {
            regime: MarketRegime::Volatile,
            params: MarketRegime::Volatile.default_params(),
            confidence: 0.72,
            transition_cooldown_cycles: 3,
            consecutive_regime_count: 2,
            proposed_regime: MarketRegime::Extreme,
        };

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: RegimeState = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.regime, MarketRegime::Volatile);
        assert!((deserialized.confidence - 0.72).abs() < f64::EPSILON);
        assert_eq!(deserialized.transition_cooldown_cycles, 3);
    }
}
