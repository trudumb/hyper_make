//! Regime state machine — single source of truth for all regime-dependent parameters.
//!
//! The `RegimeState` is updated early in each quote cycle from HMM probabilities,
//! BOCPD changepoint detection, and the regime kappa estimator. All downstream
//! components read regime-dependent parameters from `RegimeState::params()` rather
//! than computing their own regime adjustments.
//!
//! ## Continuous Blending Architecture
//!
//! All regime parameters are computed as belief-weighted averages across all 4 regimes:
//!
//!   `P_effective = Σ belief[i] × P_regime[i]`
//!
//! An EWMA smoother (α=0.05) prevents sudden jumps. The discrete label only changes
//! when conviction conditions are met (margin, KL divergence, dwell time).

use crate::market_maker::estimator::BlendedParameters;
use serde::{Deserialize, Serialize};
use std::time::Instant;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MarketRegime {
    Calm,     // Low vol, mean-reverting
    #[default]
    Normal,   // Typical conditions
    Volatile, // Elevated vol, trending possible
    Extreme,  // Cascade / liquidation regime
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

    /// Convert regime to index (0=Calm, 1=Normal, 2=Volatile, 3=Extreme).
    pub fn to_index(self) -> usize {
        match self {
            Self::Calm => 0,
            Self::Normal => 1,
            Self::Volatile => 2,
            Self::Extreme => 3,
        }
    }

    /// Severity level (0=Calm through 3=Extreme) for escalation/de-escalation logic.
    fn severity(self) -> u8 {
        match self {
            Self::Calm => 0,
            Self::Normal => 1,
            Self::Volatile => 2,
            Self::Extreme => 3,
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
                gamma_multiplier: 1.0,
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
                gamma_multiplier: 1.2,
            },
            Self::Volatile => RegimeParams {
                kappa: 1000.0,
                spread_floor_bps: 10.0,
                skew_gain: 0.5,
                max_position_fraction: 0.7,
                emergency_cp_threshold: 0.6,
                reduce_only_fraction: 0.3,
                size_multiplier: 0.5,
                as_expected_bps: 3.0,
                risk_premium_bps: 3.0,
                controller_objective: ControllerObjective::TrendingToxic,
                gamma_multiplier: 2.0,
            },
            Self::Extreme => RegimeParams {
                kappa: 500.0,
                spread_floor_bps: 20.0,
                skew_gain: 0.3,
                max_position_fraction: 0.5,
                emergency_cp_threshold: 0.4,
                reduce_only_fraction: 0.2,
                size_multiplier: 0.3,
                as_expected_bps: 5.0,
                risk_premium_bps: 6.0,
                controller_objective: ControllerObjective::TrendingToxic,
                gamma_multiplier: 3.0,
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
    /// Gamma multiplier for this regime.
    /// Routes regime risk through gamma instead of spread floor clamping.
    /// Higher gamma → wider spreads via GLFT formula δ = (1/γ)ln(1 + γ/κ).
    #[serde(default = "default_gamma_multiplier")]
    pub gamma_multiplier: f64,
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

/// Belief-weighted blended parameters across all 4 regimes.
///
/// This is the primary output consumed by downstream components (GLFT, risk overlay, etc.).
/// All numeric fields are computed as `Σ belief[i] × param_regime[i]`, then EWMA-smoothed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendedRegimeParams {
    /// Blended kappa for GLFT formula. kappa > 0.0 invariant.
    pub kappa: f64,
    /// Blended spread floor (bps).
    pub spread_floor_bps: f64,
    /// Blended skew gain.
    pub skew_gain: f64,
    /// Blended max position fraction.
    pub max_position_fraction: f64,
    /// Blended emergency changepoint threshold.
    pub emergency_cp_threshold: f64,
    /// Blended reduce-only fraction.
    pub reduce_only_fraction: f64,
    /// Blended quote size multiplier.
    pub size_multiplier: f64,
    /// Blended expected adverse selection (bps).
    pub as_expected_bps: f64,
    /// Blended additive risk premium (bps).
    pub risk_premium_bps: f64,
    /// Blended gamma multiplier.
    pub gamma_multiplier: f64,
}

impl Default for BlendedRegimeParams {
    fn default() -> Self {
        let normal = MarketRegime::Normal.default_params();
        Self {
            kappa: normal.kappa,
            spread_floor_bps: normal.spread_floor_bps,
            skew_gain: normal.skew_gain,
            max_position_fraction: normal.max_position_fraction,
            emergency_cp_threshold: normal.emergency_cp_threshold,
            reduce_only_fraction: normal.reduce_only_fraction,
            size_multiplier: normal.size_multiplier,
            as_expected_bps: normal.as_expected_bps,
            risk_premium_bps: normal.risk_premium_bps,
            gamma_multiplier: normal.gamma_multiplier,
        }
    }
}

impl BlendedRegimeParams {
    /// Compute belief-weighted average across all 4 regimes.
    ///
    /// `beliefs` is `[p_calm, p_normal, p_volatile, p_extreme]`.
    fn from_beliefs(beliefs: &[f64; 4]) -> Self {
        let params: [RegimeParams; 4] = [
            MarketRegime::Calm.default_params(),
            MarketRegime::Normal.default_params(),
            MarketRegime::Volatile.default_params(),
            MarketRegime::Extreme.default_params(),
        ];

        let mut kappa = 0.0;
        let mut spread_floor_bps = 0.0;
        let mut skew_gain = 0.0;
        let mut max_position_fraction = 0.0;
        let mut emergency_cp_threshold = 0.0;
        let mut reduce_only_fraction = 0.0;
        let mut size_multiplier = 0.0;
        let mut as_expected_bps = 0.0;
        let mut risk_premium_bps = 0.0;
        let mut gamma_multiplier = 0.0;

        for (i, b) in beliefs.iter().enumerate() {
            kappa += b * params[i].kappa;
            spread_floor_bps += b * params[i].spread_floor_bps;
            skew_gain += b * params[i].skew_gain;
            max_position_fraction += b * params[i].max_position_fraction;
            emergency_cp_threshold += b * params[i].emergency_cp_threshold;
            reduce_only_fraction += b * params[i].reduce_only_fraction;
            size_multiplier += b * params[i].size_multiplier;
            as_expected_bps += b * params[i].as_expected_bps;
            risk_premium_bps += b * params[i].risk_premium_bps;
            gamma_multiplier += b * params[i].gamma_multiplier;
        }

        // Enforce kappa > 0.0 invariant
        kappa = kappa.max(1.0);

        Self {
            kappa,
            spread_floor_bps,
            skew_gain,
            max_position_fraction,
            emergency_cp_threshold,
            reduce_only_fraction,
            size_multiplier,
            as_expected_bps,
            risk_premium_bps,
            gamma_multiplier,
        }
    }

    /// EWMA blend: `self = (1-alpha) * self + alpha * target`.
    fn ewma_toward(&mut self, target: &BlendedRegimeParams, alpha: f64) {
        let a = alpha;
        let b = 1.0 - alpha;
        self.kappa = (b * self.kappa + a * target.kappa).max(1.0);
        self.spread_floor_bps = b * self.spread_floor_bps + a * target.spread_floor_bps;
        self.skew_gain = b * self.skew_gain + a * target.skew_gain;
        self.max_position_fraction =
            b * self.max_position_fraction + a * target.max_position_fraction;
        self.emergency_cp_threshold =
            b * self.emergency_cp_threshold + a * target.emergency_cp_threshold;
        self.reduce_only_fraction =
            b * self.reduce_only_fraction + a * target.reduce_only_fraction;
        self.size_multiplier = b * self.size_multiplier + a * target.size_multiplier;
        self.as_expected_bps = b * self.as_expected_bps + a * target.as_expected_bps;
        self.risk_premium_bps = b * self.risk_premium_bps + a * target.risk_premium_bps;
        self.gamma_multiplier = b * self.gamma_multiplier + a * target.gamma_multiplier;
    }
}

/// Conviction-based label transition state.
///
/// The label only transitions when three conditions are ALL met:
/// 1. Conviction margin: `belief[proposed] - belief[current] > CONVICTION_MARGIN`
/// 2. KL divergence from uniform: `KL(belief || uniform) > KL_THRESHOLD_NATS`
/// 3. Dwell time: regime-dependent minimum time since last transition
#[derive(Debug, Clone)]
struct ConvictionState {
    current_label: MarketRegime,
    last_transition: Instant,
}

/// Minimum conviction margin: proposed belief must exceed current belief by this much.
const CONVICTION_MARGIN: f64 = 0.20;

/// Minimum KL divergence from uniform distribution (nats).
const KL_THRESHOLD_NATS: f64 = 0.3;

/// Dwell time for escalation (Calm->Normal->Volatile): 30 seconds.
const DWELL_ESCALATION_S: f64 = 30.0;

/// Dwell time for de-escalation (Volatile->Normal->Calm): 120 seconds.
const DWELL_DEESCALATION_S: f64 = 120.0;

/// Dwell time for transition TO Extreme: 0 seconds (immediate, safety first).
const DWELL_TO_EXTREME_S: f64 = 0.0;

/// Dwell time for transition FROM Extreme: 300 seconds (5 minutes).
const DWELL_FROM_EXTREME_S: f64 = 300.0;

impl Default for ConvictionState {
    fn default() -> Self {
        Self {
            current_label: MarketRegime::Normal,
            last_transition: Instant::now(),
        }
    }
}

impl ConvictionState {
    /// Determine if the label should transition from current to proposed.
    fn should_transition(
        &self,
        proposed: MarketRegime,
        beliefs: &[f64; 4],
        now: Instant,
    ) -> bool {
        if proposed == self.current_label {
            return false;
        }

        // Condition 1: conviction margin
        let current_belief = beliefs[self.current_label.to_index()];
        let proposed_belief = beliefs[proposed.to_index()];
        if proposed_belief - current_belief <= CONVICTION_MARGIN {
            return false;
        }

        // Condition 2: KL divergence from uniform
        let kl = kl_from_uniform(beliefs);
        if kl <= KL_THRESHOLD_NATS {
            return false;
        }

        // Condition 3: dwell time
        let elapsed_s = now.duration_since(self.last_transition).as_secs_f64();
        let required_dwell_s = self.required_dwell_s(proposed);
        elapsed_s >= required_dwell_s
    }

    fn required_dwell_s(&self, proposed: MarketRegime) -> f64 {
        if proposed == MarketRegime::Extreme {
            return DWELL_TO_EXTREME_S;
        }
        if self.current_label == MarketRegime::Extreme {
            return DWELL_FROM_EXTREME_S;
        }
        if proposed.severity() > self.current_label.severity() {
            DWELL_ESCALATION_S
        } else {
            DWELL_DEESCALATION_S
        }
    }

    fn transition_to(&mut self, regime: MarketRegime, now: Instant) {
        self.current_label = regime;
        self.last_transition = now;
    }
}

/// Compute KL divergence from uniform distribution: `KL(p || uniform)`.
///
/// Returns value in nats (natural log). KL = 0 when beliefs are perfectly uniform.
fn kl_from_uniform(beliefs: &[f64; 4]) -> f64 {
    let uniform = 0.25_f64;
    let mut kl = 0.0;
    for &p in beliefs {
        if p > 1e-12 {
            kl += p * (p / uniform).ln();
        }
    }
    kl
}

/// EWMA smoothing factor for blended parameter updates.
const BLENDING_ALPHA: f64 = 0.05;

/// Number of consecutive cycles required before a regime transition is accepted.
/// Kept for backward compatibility but superseded by ConvictionState.
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
    /// EWMA-smoothed blended kappa from regime belief probabilities.
    /// Populated after the first call to `update_from_blender()`.
    #[serde(default)]
    blended_kappa: Option<f64>,
    /// EWMA-smoothed blended gamma multiplier (blended.gamma / gamma_base).
    /// Populated after the first call to `update_from_blender()`.
    #[serde(default)]
    blended_gamma_multiplier: Option<f64>,
    /// EWMA-smoothed blended spread multiplier from regime beliefs.
    /// Populated after the first call to `update_from_blender()`.
    #[serde(default)]
    blended_spread_multiplier: Option<f64>,
    /// Continuously blended parameters — primary output for downstream consumers.
    #[serde(default)]
    pub blended: BlendedRegimeParams,
    /// Conviction state for label transitions (transient, not checkpointed).
    #[serde(skip)]
    conviction_state: ConvictionState,
    /// Conviction: margin between top-2 beliefs. Higher = more confident in regime.
    /// Computed as `sorted_beliefs[0] - sorted_beliefs[1]`.
    #[serde(default)]
    pub conviction: f64,
}

fn default_gamma_multiplier() -> f64 {
    1.0
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
            blended_kappa: None,
            blended_gamma_multiplier: None,
            blended_spread_multiplier: None,
            blended: BlendedRegimeParams::default(),
            conviction_state: ConvictionState::default(),
            conviction: 0.0,
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
    /// Performs continuous blending of all parameters, plus conviction-based
    /// label transitions. Returns `true` if the discrete label transitioned.
    ///
    /// # Arguments
    /// * `hmm_probs` - HMM posterior probabilities [p_calm, p_normal, p_volatile, p_extreme]
    /// * `bocpd_cp` - BOCPD changepoint probability (0.0-1.0)
    /// * `kappa_effective` - Regime-blended kappa from signal integration
    /// * `kappa_confidence` - Posterior confidence in kappa estimate [0, 1].
    ///   From `1/(1+CV)` of `BayesianKappaEstimator`. Low fills → high CV → low
    ///   confidence → trust belief-weighted kappa. Many fills → tight posterior →
    ///   trust observed kappa_effective.
    pub fn update(
        &mut self,
        hmm_probs: &[f64; 4],
        bocpd_cp: f64,
        kappa_effective: f64,
        kappa_confidence: f64,
    ) -> bool {
        let now = Instant::now();

        // Step 1: Modify beliefs with changepoint evidence
        let mut beliefs = *hmm_probs;
        Self::apply_changepoint_evidence_to(&mut beliefs, bocpd_cp);

        // Step 2: Compute instantaneous blended params from beliefs
        let target = BlendedRegimeParams::from_beliefs(&beliefs);

        // Step 3: EWMA smooth toward target
        self.blended.ewma_toward(&target, BLENDING_ALPHA);

        // Step 4: Blend kappa with kappa_effective weighted by posterior confidence.
        // Low fills → high CV → low confidence → trust belief-weighted kappa.
        // Many fills → tight posterior → trust observed kappa_effective.
        // Cap at 0.95: always retain 5% regime belief weight (robustness constraint).
        //
        // NOTE: For quadratic loss, the Bayes-optimal weight is
        // w = var_prior / (var_prior + var_posterior) (inverse-variance weighting).
        // 1/(1+CV) approximates this but isn't exact. The approximation error is
        // dominated by regime mixture variance and other noise sources.
        // The 0.95 cap also partially compensates: even if the CV mapping overstates
        // certainty (CV near 0 → confidence near 1), the cap prevents the data-driven
        // estimate from fully dominating the regime belief.
        if kappa_effective > 0.0 {
            let w = kappa_confidence.clamp(0.0, 0.95);
            self.blended.kappa = ((1.0 - w) * self.blended.kappa + w * kappa_effective).max(1.0);
        }

        // Step 5: Update confidence to max belief
        self.confidence = beliefs.iter().cloned().fold(0.0_f64, f64::max);

        // Step 6: Conviction-based label transition
        let proposed = Self::argmax_regime(&beliefs);
        let transitioned = self.conviction_state.should_transition(proposed, &beliefs, now);

        if transitioned {
            self.conviction_state.transition_to(proposed, now);
            self.regime = proposed;
            self.params = proposed.default_params();
            // Override params.kappa with blended kappa for compatibility
            self.params.kappa = self.blended.kappa;
            // Legacy hysteresis fields
            self.transition_cooldown_cycles = HYSTERESIS_CYCLES;
            self.consecutive_regime_count = 0;

            true
        } else {
            // Keep discrete params.kappa in sync with blended
            self.params.kappa = self.blended.kappa;
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
        self.blended.spread_floor_bps - MarketRegime::Normal.default_params().spread_floor_bps
    }

    /// Inject BOCPD changepoint evidence into beliefs before blending.
    ///
    /// Standalone method for external callers that want to inject CP evidence
    /// outside the normal `update()` cycle.
    pub fn inject_changepoint_evidence(&mut self, cp_probability: f64) {
        // Guard: skip boost entirely for low cp_probability to prevent
        // continuous belief erosion from noisy changepoint detectors.
        // Without this, cp_prob=0.1 every cycle slowly shifts beliefs toward Volatile.
        if cp_probability < 0.3 {
            return;
        }
        let mut beliefs = [0.25, 0.25, 0.25, 0.25];
        Self::apply_changepoint_evidence_to(&mut beliefs, cp_probability);
        let target = BlendedRegimeParams::from_beliefs(&beliefs);
        self.blended.ewma_toward(&target, BLENDING_ALPHA);
    }

    /// Apply changepoint evidence to a beliefs array and normalize.
    ///
    /// * `cp > 0.5`: boost Volatile + Extreme proportionally
    /// * `cp > 0.9`: force `belief[Extreme]` >= 0.6
    fn apply_changepoint_evidence_to(beliefs: &mut [f64; 4], cp_probability: f64) {
        if cp_probability > 0.5 {
            let boost = cp_probability - 0.5;
            beliefs[2] += boost * 0.5; // Half to Volatile
            beliefs[3] += boost * 0.5; // Half to Extreme
        }

        if cp_probability > 0.9 && beliefs[3] < 0.6 {
            beliefs[3] = 0.6;
        }

        // Normalize
        let sum: f64 = beliefs.iter().sum();
        if sum > 1e-12 {
            for b in beliefs.iter_mut() {
                *b /= sum;
            }
        }
    }

    /// Get the argmax regime from beliefs.
    fn argmax_regime(beliefs: &[f64; 4]) -> MarketRegime {
        let (max_idx, _) = beliefs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((1, &0.25));
        MarketRegime::from_hmm_index(max_idx)
    }

    /// Update blended regime parameters from the `RegimeParameterBlender` output.
    ///
    /// EWMA-smooths kappa, gamma multiplier, and spread multiplier from the
    /// blender's belief-weighted output. On the first call, values are initialized
    /// directly (no smoothing). Conviction is updated from belief entropy.
    ///
    /// # Arguments
    /// * `blended` - Output from `RegimeParameterBlender::blend()`
    /// * `gamma_base` - Base gamma from `RegimeParameterConfig::gamma_base` for
    ///   computing the gamma multiplier (`blended.gamma / gamma_base`)
    /// * `ewma_alpha` - EWMA smoothing factor (0.15 default, ~7-update half-life)
    pub fn update_from_blender(
        &mut self,
        blended: &BlendedParameters,
        gamma_base: f64,
        ewma_alpha: f64,
    ) {
        let alpha = ewma_alpha.clamp(0.01, 1.0);

        // Blended kappa: enforce kappa > 0.0 invariant
        let new_kappa = blended.kappa.max(1.0);
        self.blended_kappa = Some(match self.blended_kappa {
            Some(prev) => ((1.0 - alpha) * prev + alpha * new_kappa).max(1.0),
            None => new_kappa,
        });

        // Gamma multiplier: blended.gamma / gamma_base
        // Guard: gamma_base must be > 0 to avoid division by zero
        let safe_gamma_base = gamma_base.max(1e-6);
        let new_gamma_mult = (blended.gamma / safe_gamma_base).max(0.01);
        self.blended_gamma_multiplier = Some(match self.blended_gamma_multiplier {
            Some(prev) => (1.0 - alpha) * prev + alpha * new_gamma_mult,
            None => new_gamma_mult,
        });

        // Spread multiplier: direct from blended output
        let new_spread_mult = blended.spread_multiplier.max(0.1);
        self.blended_spread_multiplier = Some(match self.blended_spread_multiplier {
            Some(prev) => (1.0 - alpha) * prev + alpha * new_spread_mult,
            None => new_spread_mult,
        });

        // Update conviction from belief entropy.
        // Max entropy for 4 regimes = ln(4) ~ 1.386 nats.
        // Conviction = 1.0 - (entropy / max_entropy), so concentrated beliefs -> high conviction.
        let max_entropy = 4.0_f64.ln();
        let entropy = blended.belief.entropy();
        self.confidence = (1.0_f64 - entropy / max_entropy).clamp(0.0, 1.0);
    }

    /// EWMA-smoothed blended kappa from regime beliefs.
    /// Returns `None` until the first call to `update_from_blender()`.
    pub fn blended_kappa(&self) -> Option<f64> {
        self.blended_kappa
    }

    /// EWMA-smoothed blended gamma multiplier (blended.gamma / gamma_base).
    /// Returns `None` until the first call to `update_from_blender()`.
    pub fn blended_gamma_multiplier(&self) -> Option<f64> {
        self.blended_gamma_multiplier
    }

    /// EWMA-smoothed blended spread multiplier from regime beliefs.
    /// Returns `None` until the first call to `update_from_blender()`.
    pub fn blended_spread_multiplier(&self) -> Option<f64> {
        self.blended_spread_multiplier
    }

    /// Update blended parameters from RegimeParameterBlender output.
    ///
    /// EWMA smooths all params with the given alpha (typical: 0.15).
    /// On the first call (when fields are None), values are initialized directly.
    ///
    /// # Arguments
    /// * `blended_kappa` - Belief-weighted kappa from blender
    /// * `blended_gamma_mult` - Belief-weighted gamma multiplier
    /// * `blended_spread_mult` - Belief-weighted spread multiplier
    /// * `alpha` - EWMA smoothing factor, clamped to [0.01, 0.5]
    pub fn update_blended_from_blender(
        &mut self,
        blended_kappa: f64,
        blended_gamma_mult: f64,
        blended_spread_mult: f64,
        alpha: f64,
    ) {
        let a = alpha.clamp(0.01, 0.5);

        // Blended kappa: enforce kappa > 0.0 invariant
        self.blended_kappa = Some(
            match self.blended_kappa {
                Some(prev) => (1.0 - a) * prev + a * blended_kappa,
                None => blended_kappa,
            }
            .max(1.0),
        );

        // Gamma multiplier: enforce >= 0.1
        self.blended_gamma_multiplier = Some(
            match self.blended_gamma_multiplier {
                Some(prev) => (1.0 - a) * prev + a * blended_gamma_mult,
                None => blended_gamma_mult,
            }
            .max(0.1),
        );

        // Spread multiplier: enforce >= 0.1
        self.blended_spread_multiplier = Some(
            match self.blended_spread_multiplier {
                Some(prev) => (1.0 - a) * prev + a * blended_spread_mult,
                None => blended_spread_mult,
            }
            .max(0.1),
        );
    }

    /// Compute conviction from beliefs array [4] as margin between top-2 beliefs.
    ///
    /// Conviction = sorted_beliefs[0] - sorted_beliefs[1].
    /// Range: 0.0 (uniform) to ~1.0 (concentrated in one regime).
    pub fn update_conviction(&mut self, beliefs: &[f64; 4]) {
        let mut sorted = *beliefs;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        self.conviction = (sorted[0] - sorted[1]).max(0.0);
    }

    /// Compute blended gamma multiplier from regime probabilities.
    ///
    /// Interpolates gamma multipliers across all 4 regimes weighted by belief
    /// probabilities, producing a continuous value instead of the discrete
    /// {1.0, 1.2, 2.0, 3.0} jumps from the active regime label.
    ///
    /// # Arguments
    /// * `regime_probs` - `[p_calm, p_normal, p_volatile, p_extreme]`
    ///
    /// # Returns
    /// Belief-weighted gamma multiplier, always >= 1.0.
    ///
    /// # Example
    /// With probs `[0.2, 0.3, 0.3, 0.2]`:
    /// `0.2*1.0 + 0.3*1.2 + 0.3*2.0 + 0.2*3.0 = 0.2 + 0.36 + 0.6 + 0.6 = 1.76`
    pub fn blended_gamma_multiplier_from_probs(regime_probs: &[f64; 4]) -> f64 {
        const GAMMA_MULTIPLIERS: [f64; 4] = [1.0, 1.2, 2.0, 3.0];
        let blended: f64 = regime_probs
            .iter()
            .zip(GAMMA_MULTIPLIERS.iter())
            .map(|(p, m)| p * m)
            .sum();
        // Enforce gamma_multiplier >= 1.0 invariant
        blended.max(1.0)
    }

    /// Get the continuously blended gamma multiplier.
    ///
    /// Returns the blended value from `self.blended.gamma_multiplier` which
    /// is EWMA-smoothed across regime probability updates. Falls back to the
    /// discrete regime gamma multiplier if blending has not been initialized.
    pub fn effective_gamma_multiplier(&self) -> f64 {
        self.blended.gamma_multiplier.max(1.0)
    }

    /// Produce a unified regime snapshot for downstream consumers.
    /// Single source of truth: kappa_effective, gamma_multiplier, label, position fraction.
    /// Replaces 4 independent regime channels (HMM, RegimeKappa, ThresholdKappa, BOCPD).
    pub fn unified_regime(&self) -> UnifiedRegime {
        UnifiedRegime {
            kappa_effective: self.blended.kappa.max(1.0),
            gamma_multiplier: self.effective_gamma_multiplier(),
            regime_label: self.regime,
            max_position_fraction: self.blended.max_position_fraction.clamp(0.3, 1.0),
            risk_premium_bps: self.blended.risk_premium_bps.max(0.0),
            spread_floor_bps: self.blended.spread_floor_bps.max(0.0),
            conviction: self.conviction.clamp(0.0, 1.0),
        }
    }
}


/// Unified regime snapshot — single source of truth for all regime-dependent decisions.
/// Replaces 4 independent regime channels (HMM, RegimeKappa, ThresholdKappa, BOCPD)
/// with one coherent output.
///
/// Produced by `RegimeState::unified_regime()` once per quote cycle.
/// All downstream consumers should use this instead of querying regime state directly.
#[derive(Debug, Clone, Copy)]
pub struct UnifiedRegime {
    /// Blended kappa for GLFT spread formula. Always > 0.
    /// Higher kappa = tighter spreads (kappa=3250 → 4.6 bps, kappa=8000 → 2.75 bps).
    pub kappa_effective: f64,
    /// Gamma multiplier for risk aversion [1.0, 3.0].
    /// Routes regime risk through GLFT gamma: δ = (1/γ)ln(1 + γ/κ).
    pub gamma_multiplier: f64,
    /// Discrete label for logging and threshold-based decisions.
    pub regime_label: MarketRegime,
    /// Position limit fraction [0.3, 1.0] — feeds PositionLimits.regime_fraction.
    pub max_position_fraction: f64,
    /// Additive risk premium for this regime state (bps).
    /// Feeds SpreadComposition.risk_premium_bps.
    pub risk_premium_bps: f64,
    /// Regime-conditioned spread floor (bps).
    pub spread_floor_bps: f64,
    /// Conviction in current regime [0.0, 1.0]. Higher = more confident.
    pub conviction: f64,
}

impl Default for UnifiedRegime {
    fn default() -> Self {
        Self {
            kappa_effective: 2000.0,
            gamma_multiplier: 1.0,
            regime_label: MarketRegime::Normal,
            max_position_fraction: 1.0,
            risk_premium_bps: 0.0,
            spread_floor_bps: 0.0,
            conviction: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a RegimeState with a specific conviction label and timestamp.
    fn state_with_conviction(label: MarketRegime, age_s: f64) -> RegimeState {
        let now = Instant::now();
        let past = now - std::time::Duration::from_secs_f64(age_s);
        let mut state = RegimeState::new();
        state.regime = label;
        state.params = label.default_params();
        state.blended = BlendedRegimeParams::from_beliefs(&match label {
            MarketRegime::Calm => [1.0, 0.0, 0.0, 0.0],
            MarketRegime::Normal => [0.0, 1.0, 0.0, 0.0],
            MarketRegime::Volatile => [0.0, 0.0, 1.0, 0.0],
            MarketRegime::Extreme => [0.0, 0.0, 0.0, 1.0],
        });
        state.conviction_state = ConvictionState {
            current_label: label,
            last_transition: past,
        };
        state
    }

    #[test]
    fn test_default_is_normal() {
        let state = RegimeState::new();
        assert_eq!(state.regime, MarketRegime::Normal);
        assert!((state.blended.kappa - 2000.0).abs() < f64::EPSILON);
        assert!((state.confidence - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_blended_stable_with_near_uniform_beliefs() {
        // Near-uniform beliefs -> label stays, blended params are weighted average
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);

        let probs = [0.30, 0.28, 0.22, 0.20];
        for _ in 0..50 {
            let changed = state.update(&probs, 0.0, 0.0, 0.0);
            assert!(!changed, "Label should NOT change with near-uniform beliefs");
        }
        assert_eq!(state.regime, MarketRegime::Normal);

        let calm_kappa = MarketRegime::Calm.default_params().kappa;
        let extreme_kappa = MarketRegime::Extreme.default_params().kappa;
        assert!(
            state.blended.kappa > extreme_kappa && state.blended.kappa < calm_kappa,
            "Blended kappa {} should be between Extreme ({}) and Calm ({})",
            state.blended.kappa, extreme_kappa, calm_kappa
        );
    }

    #[test]
    fn test_strong_conviction_triggers_transition() {
        // Normal→Calm is de-escalation, needs 120s dwell
        let mut state = state_with_conviction(MarketRegime::Normal, 121.0);

        let probs = [0.80, 0.05, 0.10, 0.05];
        let changed = state.update(&probs, 0.0, 0.0, 0.0);
        assert!(changed, "Should transition with strong conviction + sufficient dwell");
        assert_eq!(state.regime, MarketRegime::Calm);
    }

    #[test]
    fn test_extreme_detection_immediate() {
        let mut state = state_with_conviction(MarketRegime::Normal, 0.0);

        let probs = [0.05, 0.10, 0.15, 0.70];
        let changed = state.update(&probs, 0.0, 500.0, 0.5);
        assert!(changed, "Extreme should transition immediately (0s dwell)");
        assert_eq!(state.regime, MarketRegime::Extreme);
    }

    #[test]
    fn test_deescalation_from_extreme_needs_300s() {
        let mut state = state_with_conviction(MarketRegime::Extreme, 100.0);

        let probs = [0.05, 0.80, 0.10, 0.05];
        let changed = state.update(&probs, 0.0, 2000.0, 0.5);
        assert!(!changed, "Should NOT de-escalate from Extreme with only 100s dwell");
        assert_eq!(state.regime, MarketRegime::Extreme);

        let mut state2 = state_with_conviction(MarketRegime::Extreme, 301.0);
        let changed2 = state2.update(&probs, 0.0, 2000.0, 0.5);
        assert!(changed2, "Should de-escalate from Extreme after 300s dwell");
        assert_eq!(state2.regime, MarketRegime::Normal);
    }

    #[test]
    fn test_bocpd_forces_extreme_belief() {
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);

        // HMM already leans Extreme, BOCPD > 0.9 pushes it over conviction threshold
        let probs = [0.05, 0.10, 0.25, 0.60];
        let changed = state.update(&probs, 0.95, 1000.0, 0.5);
        assert!(changed, "BOCPD > 0.9 should force Extreme transition");
        assert_eq!(state.regime, MarketRegime::Extreme);
    }

    #[test]
    fn test_ewma_smoothing_gradual() {
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);
        let initial_kappa = state.blended.kappa;
        let extreme_kappa = MarketRegime::Extreme.default_params().kappa;

        let extreme_probs = [0.0, 0.05, 0.10, 0.85];
        state.update(&extreme_probs, 0.0, 0.0, 0.0);
        let after_1 = state.blended.kappa;
        assert!(
            (after_1 - initial_kappa).abs() < (initial_kappa - extreme_kappa).abs() * 0.15,
            "After 1 update, kappa {} should still be close to initial {}", after_1, initial_kappa
        );

        for _ in 0..19 {
            state.update(&extreme_probs, 0.0, 0.0, 0.0);
        }
        assert!(
            state.blended.kappa < initial_kappa * 0.7,
            "After 20 updates, kappa {} should be well below initial {}", state.blended.kappa, initial_kappa
        );
    }

    #[test]
    fn test_kl_uniform_no_transition() {
        let mut state = state_with_conviction(MarketRegime::Normal, 600.0);

        let uniform = [0.25, 0.25, 0.25, 0.25];
        for _ in 0..20 {
            let changed = state.update(&uniform, 0.0, 0.0, 0.0);
            assert!(!changed, "Uniform beliefs should never trigger transition (KL=0)");
        }
        assert_eq!(state.regime, MarketRegime::Normal);
    }

    #[test]
    fn test_kl_from_uniform_computation() {
        let uniform = [0.25, 0.25, 0.25, 0.25];
        assert!(kl_from_uniform(&uniform).abs() < 1e-10);

        let concentrated = [0.97, 0.01, 0.01, 0.01];
        let kl = kl_from_uniform(&concentrated);
        assert!(kl > 1.0, "Concentrated beliefs should have high KL, got {kl}");
    }

    #[test]
    fn test_kappa_always_positive() {
        let mut state = RegimeState::new();

        let normal_probs = [0.1, 0.8, 0.05, 0.05];
        state.update(&normal_probs, 0.0, 0.0, 0.0);
        assert!(state.blended.kappa > 0.0, "Kappa must always be > 0.0");

        for _ in 0..100 {
            state.update(&normal_probs, 0.0, 0.1, 0.5);
        }
        assert!(state.blended.kappa >= 1.0, "Kappa clamped to >= 1.0, got {}", state.blended.kappa);
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

        // Normal regime: adjustment = 0 (baseline, blended starts at Normal defaults)
        assert!((state.spread_adjustment_bps()).abs() < f64::EPSILON);

        // Feed many extreme updates to EWMA toward extreme spread floor
        let extreme_probs = [0.0, 0.05, 0.1, 0.85];
        for _ in 0..100 {
            state.update(&extreme_probs, 0.0, 500.0, 0.5);
        }
        assert!(
            state.spread_adjustment_bps() > 5.0,
            "After many extreme updates, spread adjustment {} should be positive",
            state.spread_adjustment_bps()
        );
    }

    #[test]
    fn test_conviction_needs_margin_above_threshold() {
        // With conviction-based transitions, insufficient margin prevents transition
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);

        // Beliefs where Calm leads but margin < CONVICTION_MARGIN (0.20)
        let low_margin = [0.35, 0.30, 0.20, 0.15];
        for _ in 0..20 {
            let changed = state.update(&low_margin, 0.0, 0.0, 0.0);
            assert!(!changed, "Margin 0.05 < 0.20 threshold should not trigger transition");
        }
        assert_eq!(state.regime, MarketRegime::Normal);
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
        // Use conviction helper: Normal with enough dwell time
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);
        assert_eq!(
            state.params.controller_objective,
            ControllerObjective::MeanRevert
        );

        // Transition to Extreme (0s dwell, strong conviction)
        let extreme_probs = [0.0, 0.05, 0.1, 0.85];
        let changed = state.update(&extreme_probs, 0.0, 500.0, 0.5);
        assert!(changed, "Should transition to Extreme");
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
            blended_kappa: Some(1800.0),
            blended_gamma_multiplier: Some(1.5),
            blended_spread_multiplier: Some(1.2),
            blended: BlendedRegimeParams::default(),
            conviction_state: ConvictionState::default(),
            conviction: 0.35,
        };

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: RegimeState = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.regime, MarketRegime::Volatile);
        assert!((deserialized.confidence - 0.72).abs() < f64::EPSILON);
        assert_eq!(deserialized.transition_cooldown_cycles, 3);
        assert!((deserialized.blended_kappa.unwrap() - 1800.0).abs() < f64::EPSILON);
        assert!((deserialized.blended_gamma_multiplier.unwrap() - 1.5).abs() < f64::EPSILON);
        assert!((deserialized.blended_spread_multiplier.unwrap() - 1.2).abs() < f64::EPSILON);
        assert!((deserialized.conviction - 0.35).abs() < f64::EPSILON);
    }

    #[test]
    fn test_serde_backward_compat_missing_blended_fields() {
        // Old checkpoint without blended fields should deserialize with None defaults
        let json = r#"{"regime":"Normal","params":{"kappa":2000.0,"spread_floor_bps":5.0,"skew_gain":1.0,"max_position_fraction":0.8,"emergency_cp_threshold":0.8,"reduce_only_fraction":0.5,"size_multiplier":0.8,"as_expected_bps":1.0,"risk_premium_bps":1.0,"controller_objective":"MeanRevert","gamma_multiplier":1.2},"confidence":0.5,"transition_cooldown_cycles":0,"consecutive_regime_count":0,"proposed_regime":"Normal"}"#;
        let deserialized: RegimeState = serde_json::from_str(json).unwrap();
        assert!(deserialized.blended_kappa.is_none());
        assert!(deserialized.blended_gamma_multiplier.is_none());
        assert!(deserialized.blended_spread_multiplier.is_none());
    }

    /// Helper to build a BlendedParameters for testing.
    fn test_blended(kappa: f64, gamma: f64, spread_mult: f64) -> BlendedParameters {
        use crate::market_maker::estimator::RegimeBeliefState;
        BlendedParameters {
            kappa,
            gamma,
            spread_multiplier: spread_mult,
            kelly_multiplier: 1.0,
            belief: RegimeBeliefState::default(),
        }
    }

    #[test]
    fn test_update_from_blender_first_call_initializes() {
        let mut state = RegimeState::new();

        // Before any call, all accessors return None
        assert!(state.blended_kappa().is_none());
        assert!(state.blended_gamma_multiplier().is_none());
        assert!(state.blended_spread_multiplier().is_none());

        let blended = test_blended(2500.0, 0.15, 1.3);
        let gamma_base = 0.1;
        state.update_from_blender(&blended, gamma_base, 0.15);

        // First call initializes directly without smoothing
        let kappa = state.blended_kappa().unwrap();
        assert!(
            (kappa - 2500.0).abs() < f64::EPSILON,
            "First call should initialize kappa directly, got {kappa}"
        );
        let gamma_mult = state.blended_gamma_multiplier().unwrap();
        assert!(
            (gamma_mult - 1.5).abs() < 1e-10,
            "First call gamma_mult should be 0.15/0.1 = 1.5, got {gamma_mult}"
        );
        let spread_mult = state.blended_spread_multiplier().unwrap();
        assert!(
            (spread_mult - 1.3).abs() < f64::EPSILON,
            "First call spread_mult should be 1.3, got {spread_mult}"
        );
    }

    #[test]
    fn test_update_from_blender_ewma_convergence() {
        let mut state = RegimeState::new();
        let gamma_base = 0.1;
        let alpha = 0.15;

        // Initialize with kappa=2000
        let initial = test_blended(2000.0, 0.1, 1.0);
        state.update_from_blender(&initial, gamma_base, alpha);
        assert!((state.blended_kappa().unwrap() - 2000.0).abs() < f64::EPSILON);

        // Now feed kappa=1000 repeatedly — should converge toward 1000
        let target = test_blended(1000.0, 0.2, 2.0);
        for _ in 0..100 {
            state.update_from_blender(&target, gamma_base, alpha);
        }

        let final_kappa = state.blended_kappa().unwrap();
        assert!(
            (final_kappa - 1000.0).abs() < 1.0,
            "After 100 updates with alpha=0.15, kappa should converge to 1000, got {final_kappa}"
        );

        let final_gamma = state.blended_gamma_multiplier().unwrap();
        assert!(
            (final_gamma - 2.0).abs() < 0.01,
            "Gamma mult should converge to 0.2/0.1=2.0, got {final_gamma}"
        );

        let final_spread = state.blended_spread_multiplier().unwrap();
        assert!(
            (final_spread - 2.0).abs() < 0.01,
            "Spread mult should converge to 2.0, got {final_spread}"
        );
    }

    #[test]
    fn test_update_from_blender_ewma_gradual() {
        let mut state = RegimeState::new();
        let gamma_base = 0.1;
        let alpha = 0.15;

        // Initialize at 2000
        let initial = test_blended(2000.0, 0.1, 1.0);
        state.update_from_blender(&initial, gamma_base, alpha);

        // One update toward 1000 — should move only ~15%
        let target = test_blended(1000.0, 0.1, 1.0);
        state.update_from_blender(&target, gamma_base, alpha);

        let after_one = state.blended_kappa().unwrap();
        // Expected: (1-0.15)*2000 + 0.15*1000 = 1700 + 150 = 1850
        assert!(
            (after_one - 1850.0).abs() < 1.0,
            "After 1 EWMA update: expected ~1850, got {after_one}"
        );
    }

    #[test]
    fn test_update_from_blender_conviction_from_entropy() {
        use crate::market_maker::estimator::RegimeBeliefState;

        let mut state = RegimeState::new();
        let gamma_base = 0.1;

        // Uniform beliefs -> max entropy -> conviction ~0
        let uniform_blended = BlendedParameters {
            kappa: 2000.0,
            gamma: 0.1,
            spread_multiplier: 1.0,
            kelly_multiplier: 1.0,
            belief: RegimeBeliefState::uniform(),
        };
        state.update_from_blender(&uniform_blended, gamma_base, 0.15);
        assert!(
            state.confidence < 0.05,
            "Uniform beliefs should give near-zero conviction, got {}",
            state.confidence
        );

        // Concentrated beliefs (pure Normal) -> zero entropy -> conviction ~1
        let mut concentrated = RegimeBeliefState::default();
        concentrated.p_low = 0.0;
        concentrated.p_normal = 1.0;
        concentrated.p_high = 0.0;
        concentrated.p_extreme = 0.0;
        let concentrated_blended = BlendedParameters {
            kappa: 2000.0,
            gamma: 0.1,
            spread_multiplier: 1.0,
            kelly_multiplier: 1.0,
            belief: concentrated,
        };
        state.update_from_blender(&concentrated_blended, gamma_base, 0.15);
        assert!(
            state.confidence > 0.95,
            "Concentrated beliefs should give near-1.0 conviction, got {}",
            state.confidence
        );
    }

    // --- Tests for update_blended_from_blender and update_conviction ---

    #[test]
    fn test_update_blended_from_blender_ewma_smoothing() {
        let mut state = RegimeState::new();
        let alpha = 0.15;

        // Initialize at kappa=2000
        state.update_blended_from_blender(2000.0, 1.0, 1.0, alpha);
        assert!((state.blended_kappa().unwrap() - 2000.0).abs() < f64::EPSILON);

        // One update toward kappa=1000: expected = 0.85*2000 + 0.15*1000 = 1850
        state.update_blended_from_blender(1000.0, 0.5, 0.5, alpha);
        let kappa = state.blended_kappa().unwrap();
        assert!(
            (kappa - 1850.0).abs() < 1.0,
            "Expected ~1850 after one EWMA step, got {kappa}"
        );

        let gamma = state.blended_gamma_multiplier().unwrap();
        // 0.85*1.0 + 0.15*0.5 = 0.925
        assert!(
            (gamma - 0.925).abs() < 0.01,
            "Expected ~0.925 gamma mult, got {gamma}"
        );

        let spread = state.blended_spread_multiplier().unwrap();
        // 0.85*1.0 + 0.15*0.5 = 0.925
        assert!(
            (spread - 0.925).abs() < 0.01,
            "Expected ~0.925 spread mult, got {spread}"
        );
    }

    #[test]
    fn test_update_blended_from_blender_first_call_initializes_directly() {
        let mut state = RegimeState::new();

        // Before any call, all fields are None
        assert!(state.blended_kappa().is_none());
        assert!(state.blended_gamma_multiplier().is_none());
        assert!(state.blended_spread_multiplier().is_none());

        // First call should initialize directly (no EWMA smoothing)
        state.update_blended_from_blender(3000.0, 2.0, 1.5, 0.15);
        assert!((state.blended_kappa().unwrap() - 3000.0).abs() < f64::EPSILON);
        assert!((state.blended_gamma_multiplier().unwrap() - 2.0).abs() < f64::EPSILON);
        assert!((state.blended_spread_multiplier().unwrap() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_blended_from_blender_convergence() {
        let mut state = RegimeState::new();
        let alpha = 0.15;

        // Initialize at kappa=2000
        state.update_blended_from_blender(2000.0, 1.0, 1.0, alpha);

        // Feed kappa=500 for 100 iterations — should converge
        for _ in 0..100 {
            state.update_blended_from_blender(500.0, 0.5, 2.0, alpha);
        }

        let kappa = state.blended_kappa().unwrap();
        assert!(
            (kappa - 500.0).abs() < 1.0,
            "Kappa should converge to 500, got {kappa}"
        );
        let gamma = state.blended_gamma_multiplier().unwrap();
        assert!(
            (gamma - 0.5).abs() < 0.01,
            "Gamma mult should converge to 0.5, got {gamma}"
        );
        let spread = state.blended_spread_multiplier().unwrap();
        assert!(
            (spread - 2.0).abs() < 0.01,
            "Spread mult should converge to 2.0, got {spread}"
        );
    }

    #[test]
    fn test_update_blended_from_blender_kappa_clamped() {
        let mut state = RegimeState::new();

        // Feed kappa=0 — should be clamped to 1.0
        state.update_blended_from_blender(0.0, 0.5, 0.5, 0.15);
        assert!(
            state.blended_kappa().unwrap() >= 1.0,
            "Kappa must be >= 1.0, got {}",
            state.blended_kappa().unwrap()
        );

        // Feed negative kappa — should still clamp to 1.0
        state.update_blended_from_blender(-100.0, 0.0, 0.0, 0.15);
        assert!(
            state.blended_kappa().unwrap() >= 1.0,
            "Kappa must be >= 1.0 even with negative input, got {}",
            state.blended_kappa().unwrap()
        );

        // Gamma and spread also clamped
        assert!(
            state.blended_gamma_multiplier().unwrap() >= 0.1,
            "Gamma mult must be >= 0.1"
        );
        assert!(
            state.blended_spread_multiplier().unwrap() >= 0.1,
            "Spread mult must be >= 0.1"
        );
    }

    #[test]
    fn test_update_conviction_from_beliefs() {
        let mut state = RegimeState::new();
        assert!((state.conviction - 0.0).abs() < f64::EPSILON);

        // Concentrated beliefs: [0.8, 0.1, 0.05, 0.05]
        // Sorted: [0.8, 0.1, 0.05, 0.05], margin = 0.8 - 0.1 = 0.7
        state.update_conviction(&[0.8, 0.1, 0.05, 0.05]);
        assert!(
            (state.conviction - 0.7).abs() < 1e-10,
            "Conviction should be 0.7, got {}",
            state.conviction
        );

        // Uniform beliefs: [0.25, 0.25, 0.25, 0.25]
        // margin = 0.25 - 0.25 = 0.0
        state.update_conviction(&[0.25, 0.25, 0.25, 0.25]);
        assert!(
            state.conviction.abs() < 1e-10,
            "Uniform beliefs should give conviction ~0, got {}",
            state.conviction
        );

        // Two dominant: [0.45, 0.45, 0.05, 0.05]
        // margin = 0.45 - 0.45 = 0.0
        state.update_conviction(&[0.45, 0.45, 0.05, 0.05]);
        assert!(
            state.conviction.abs() < 1e-10,
            "Two equal dominant beliefs should give conviction ~0, got {}",
            state.conviction
        );
    }

    #[test]
    fn test_serde_backward_compat_missing_conviction_field() {
        // Old checkpoint JSON without the conviction field should deserialize with default 0.0
        let json = r#"{"regime":"Normal","params":{"kappa":2000.0,"spread_floor_bps":5.0,"skew_gain":1.0,"max_position_fraction":0.8,"emergency_cp_threshold":0.8,"reduce_only_fraction":0.5,"size_multiplier":0.8,"as_expected_bps":1.0,"risk_premium_bps":1.0,"controller_objective":"MeanRevert","gamma_multiplier":1.2},"confidence":0.5,"transition_cooldown_cycles":0,"consecutive_regime_count":0,"proposed_regime":"Normal","blended":{"kappa":2000.0,"spread_floor_bps":5.0,"skew_gain":1.0,"max_position_fraction":0.8,"emergency_cp_threshold":0.8,"reduce_only_fraction":0.5,"size_multiplier":0.8,"as_expected_bps":1.0,"risk_premium_bps":1.0,"gamma_multiplier":1.2}}"#;
        let deserialized: RegimeState = serde_json::from_str(json).unwrap();
        assert!(
            deserialized.conviction.abs() < f64::EPSILON,
            "Missing conviction should default to 0.0"
        );
        assert!(deserialized.blended_kappa.is_none());
    }

    #[test]
    fn test_unified_regime_from_normal() {
        let state = state_with_conviction(MarketRegime::Normal, 100.0);
        let unified = state.unified_regime();
        assert_eq!(unified.regime_label, MarketRegime::Normal);
        assert!(unified.kappa_effective >= 1.0, "kappa must be positive");
        assert!(unified.gamma_multiplier >= 1.0, "gamma mult >= 1.0");
        assert!(unified.max_position_fraction >= 0.3 && unified.max_position_fraction <= 1.0);
        assert!(unified.conviction >= 0.0 && unified.conviction <= 1.0);
    }

    #[test]
    fn test_unified_regime_from_extreme() {
        let state = state_with_conviction(MarketRegime::Extreme, 100.0);
        let unified = state.unified_regime();
        assert_eq!(unified.regime_label, MarketRegime::Extreme);
        // Extreme regime should have higher gamma multiplier and lower position fraction
        let normal = state_with_conviction(MarketRegime::Normal, 100.0).unified_regime();
        assert!(unified.gamma_multiplier >= normal.gamma_multiplier,
            "extreme gamma_mult {} should >= normal {}",
            unified.gamma_multiplier, normal.gamma_multiplier);
        assert!(unified.max_position_fraction <= normal.max_position_fraction,
            "extreme position fraction {} should <= normal {}",
            unified.max_position_fraction, normal.max_position_fraction);
    }

    #[test]
    fn test_unified_regime_default() {
        let ur = UnifiedRegime::default();
        assert_eq!(ur.regime_label, MarketRegime::Normal);
        assert!(ur.kappa_effective > 0.0);
        assert!((ur.gamma_multiplier - 1.0).abs() < f64::EPSILON);
        assert!((ur.max_position_fraction - 1.0).abs() < f64::EPSILON);
    }

    // === Kappa confidence-weighted blend tests ===

    #[test]
    fn test_kappa_blend_low_confidence() {
        // conf=0.2 → ~80% belief + 20% effective
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);
        // Set blended kappa to a known value
        state.blended.kappa = 2000.0;

        let probs = [0.0, 1.0, 0.0, 0.0]; // Pure Normal
        state.update(&probs, 0.0, 1000.0, 0.2);

        // w=0.2: kappa = 0.8 * blended + 0.2 * 1000
        // blended was 2000 but EWMA moved it slightly; the key check:
        // with low confidence, kappa should stay closer to belief (higher) than to effective (1000)
        assert!(
            state.blended.kappa > 1500.0,
            "Low confidence should keep kappa closer to belief: {}",
            state.blended.kappa
        );
    }

    #[test]
    fn test_kappa_blend_high_confidence() {
        // conf=0.9 → ~10% belief + 90% effective
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);
        state.blended.kappa = 2000.0;

        let probs = [0.0, 1.0, 0.0, 0.0];
        state.update(&probs, 0.0, 1000.0, 0.9);

        // w=0.9: kappa = 0.1 * blended + 0.9 * 1000
        // With high confidence, kappa should be pulled strongly toward effective (1000)
        assert!(
            state.blended.kappa < 1300.0,
            "High confidence should pull kappa toward effective: {}",
            state.blended.kappa
        );
    }

    #[test]
    fn test_kappa_blend_zero_confidence() {
        // conf=0.0 → 100% belief, 0% effective
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);
        let initial_kappa = state.blended.kappa;

        let probs = [0.0, 1.0, 0.0, 0.0];
        state.update(&probs, 0.0, 1000.0, 0.0);

        // w=0.0: kappa = 1.0 * blended + 0.0 * 1000
        // EWMA still moves blended toward target, but the kappa_effective blend has no effect
        // The initial blended was ~2000 (Normal default), and we're feeding Normal beliefs
        // So blended should stay very close to where EWMA alone takes it
        let after_ewma_only = (1.0 - BLENDING_ALPHA) * initial_kappa
            + BLENDING_ALPHA * MarketRegime::Normal.default_params().kappa;
        assert!(
            (state.blended.kappa - after_ewma_only).abs() < 1.0,
            "Zero confidence: kappa {} should match pure EWMA {}",
            state.blended.kappa, after_ewma_only
        );
    }

    #[test]
    fn test_kappa_blend_cap_at_95() {
        // conf=1.0 → capped at 0.95, retains 5% belief weight
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);
        state.blended.kappa = 2000.0;

        let probs = [0.0, 1.0, 0.0, 0.0];
        state.update(&probs, 0.0, 1000.0, 1.0);

        // w capped at 0.95: kappa = 0.05 * blended + 0.95 * 1000
        // blended was EWMA'd first toward Normal(2000), so ≈2000
        // Result ≈ 0.05 * 2000 + 0.95 * 1000 = 100 + 950 = 1050
        assert!(
            state.blended.kappa > 1000.0,
            "Cap at 0.95 should retain some belief weight: {}",
            state.blended.kappa
        );
        // But also very close to effective
        assert!(
            state.blended.kappa < 1100.0,
            "Cap at 0.95 with effective=1000 should be near 1050: {}",
            state.blended.kappa
        );
    }

    #[test]
    fn test_kappa_blend_zero_effective_skips_blend() {
        // kappa_effective=0.0 → no blend (existing guard)
        let mut state = state_with_conviction(MarketRegime::Normal, 60.0);
        let initial_kappa = state.blended.kappa;

        let probs = [0.0, 1.0, 0.0, 0.0];
        state.update(&probs, 0.0, 0.0, 0.9);

        // EWMA moves toward Normal target, but no kappa_effective blend
        let after_ewma = (1.0 - BLENDING_ALPHA) * initial_kappa
            + BLENDING_ALPHA * MarketRegime::Normal.default_params().kappa;
        assert!(
            (state.blended.kappa - after_ewma).abs() < 1.0,
            "Zero kappa_effective should skip blend: {} vs {}",
            state.blended.kappa, after_ewma
        );
    }
}
