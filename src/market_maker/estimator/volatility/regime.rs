//! Volatility regime classification and tracking.
//!
//! Four-state regime with asymmetric hysteresis to prevent rapid switching.
//! Self-calibrating: learns baseline and thresholds from observed market data.

// HMM-based regime blending infrastructure - reserved for future integration
#![allow(dead_code)]

use tracing::{debug, info};

/// Volatility regime classification.
///
/// Four states with hysteresis to prevent rapid switching:
/// - Low: Very quiet market (σ < 0.5 × baseline)
/// - Normal: Standard market conditions
/// - High: Elevated volatility (σ > 1.5 × baseline)
/// - Extreme: Crisis/toxic conditions (σ > 3 × baseline OR high jump ratio)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VolatilityRegime {
    /// Very quiet market - can tighten spreads
    Low,
    /// Normal market conditions
    #[default]
    Normal,
    /// Elevated volatility - widen spreads
    High,
    /// Crisis/toxic - maximum caution, consider pulling quotes
    Extreme,
}

impl VolatilityRegime {
    /// Get spread multiplier for this regime.
    ///
    /// Used to scale spreads based on volatility state.
    pub(crate) fn spread_multiplier(&self) -> f64 {
        match self {
            Self::Low => 0.8,     // Tighter spreads in quiet markets
            Self::Normal => 1.0,  // Base case
            Self::High => 1.5,    // Wider spreads for elevated vol
            Self::Extreme => 2.5, // Much wider spreads in crisis
        }
    }

    /// Get gamma multiplier for this regime.
    ///
    /// Risk aversion increases with volatility.
    pub(crate) fn gamma_multiplier(&self) -> f64 {
        match self {
            Self::Low => 0.8,
            Self::Normal => 1.0,
            Self::High => 1.5,
            Self::Extreme => 1.8, // Reduced from 3.0 - still conservative but not punitive
        }
    }

    /// Get Kelly fraction multiplier for this regime.
    ///
    /// In high volatility, we want to be more conservative (lower Kelly fraction).
    /// In low volatility, we can be more aggressive (higher Kelly fraction).
    ///
    /// Returns a multiplier to apply to the base Kelly fraction (0.25 default):
    /// - Low: 1.5x → 0.375 effective (can be more aggressive in quiet markets)
    /// - Normal: 1.0x → 0.25 effective (standard quarter Kelly)
    /// - High: 0.5x → 0.125 effective (more conservative)
    /// - Extreme: 0.25x → 0.0625 effective (very conservative, near flat)
    pub fn kelly_fraction_multiplier(&self) -> f64 {
        match self {
            Self::Low => 1.5,      // More aggressive in quiet markets
            Self::Normal => 1.0,   // Standard
            Self::High => 0.5,     // More conservative
            Self::Extreme => 0.25, // Very conservative
        }
    }

    /// Check if quotes should be pulled (extreme regime).
    #[allow(dead_code)]
    pub(crate) fn should_consider_pulling_quotes(&self) -> bool {
        matches!(self, Self::Extreme)
    }

    /// Get numeric index for regime (for belief state indexing).
    pub fn index(&self) -> usize {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Extreme => 3,
        }
    }

    /// Create regime from index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Low,
            1 => Self::Normal,
            2 => Self::High,
            _ => Self::Extreme,
        }
    }
}

// ============================================================================
// Regime Belief State (HMM-style soft probabilities)
// ============================================================================

/// HMM-style belief state for regime probabilities.
///
/// Instead of hard regime switches, maintains probability distribution over regimes.
/// This allows for smooth parameter blending per the Small Fish Strategy:
/// - γ_effective = P(calm) × γ_calm + P(volatile) × γ_volatile + P(cascade) × γ_cascade
///
/// Key insight: Single parameter values are almost always wrong. Use the belief state
/// to blend parameters based on regime probabilities.
///
/// Integration path with `RegimeHMM`:
/// ```ignore
/// let belief = hmm.to_belief_state();
/// let blended = blender.blend_all(&belief);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RegimeBeliefState {
    /// Probability of Low (calm) regime
    pub p_low: f64,
    /// Probability of Normal regime
    pub p_normal: f64,
    /// Probability of High (volatile) regime
    pub p_high: f64,
    /// Probability of Extreme (cascade) regime
    pub p_extreme: f64,
}

impl Default for RegimeBeliefState {
    fn default() -> Self {
        // Start with Normal being most likely
        Self {
            p_low: 0.1,
            p_normal: 0.7,
            p_high: 0.15,
            p_extreme: 0.05,
        }
    }
}

impl RegimeBeliefState {
    /// Create belief state from hard regime (all probability on one state).
    pub(crate) fn from_regime(regime: VolatilityRegime) -> Self {
        match regime {
            VolatilityRegime::Low => Self {
                p_low: 1.0,
                p_normal: 0.0,
                p_high: 0.0,
                p_extreme: 0.0,
            },
            VolatilityRegime::Normal => Self {
                p_low: 0.0,
                p_normal: 1.0,
                p_high: 0.0,
                p_extreme: 0.0,
            },
            VolatilityRegime::High => Self {
                p_low: 0.0,
                p_normal: 0.0,
                p_high: 1.0,
                p_extreme: 0.0,
            },
            VolatilityRegime::Extreme => Self {
                p_low: 0.0,
                p_normal: 0.0,
                p_high: 0.0,
                p_extreme: 1.0,
            },
        }
    }

    /// Create uniform belief state (maximum uncertainty).
    pub(crate) fn uniform() -> Self {
        Self {
            p_low: 0.25,
            p_normal: 0.25,
            p_high: 0.25,
            p_extreme: 0.25,
        }
    }

    /// Normalize probabilities to sum to 1.0.
    pub(crate) fn normalize(&mut self) {
        let sum = self.p_low + self.p_normal + self.p_high + self.p_extreme;
        if sum > 1e-9 {
            self.p_low /= sum;
            self.p_normal /= sum;
            self.p_high /= sum;
            self.p_extreme /= sum;
        } else {
            *self = Self::default();
        }
    }

    /// Get probability for a specific regime.
    pub(crate) fn probability(&self, regime: VolatilityRegime) -> f64 {
        match regime {
            VolatilityRegime::Low => self.p_low,
            VolatilityRegime::Normal => self.p_normal,
            VolatilityRegime::High => self.p_high,
            VolatilityRegime::Extreme => self.p_extreme,
        }
    }

    /// Get the most likely regime.
    pub(crate) fn most_likely(&self) -> VolatilityRegime {
        let probs = [
            (self.p_low, VolatilityRegime::Low),
            (self.p_normal, VolatilityRegime::Normal),
            (self.p_high, VolatilityRegime::High),
            (self.p_extreme, VolatilityRegime::Extreme),
        ];
        probs
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(_, r)| r)
            .unwrap_or(VolatilityRegime::Normal)
    }

    /// Get belief entropy (measure of uncertainty).
    /// Higher entropy = more uncertain about regime.
    pub(crate) fn entropy(&self) -> f64 {
        let probs = [self.p_low, self.p_normal, self.p_high, self.p_extreme];
        -probs
            .iter()
            .filter(|&&p| p > 1e-9)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Check if we're confident about the regime (low entropy).
    pub(crate) fn is_confident(&self) -> bool {
        // Max entropy for 4 states is ln(4) ≈ 1.386
        // Consider confident if entropy < 0.5 (one state dominates)
        self.entropy() < 0.5
    }

    /// Get probabilities as array [p_low, p_normal, p_high, p_extreme].
    pub(crate) fn as_array(&self) -> [f64; 4] {
        [self.p_low, self.p_normal, self.p_high, self.p_extreme]
    }
}

// ============================================================================
// Regime Parameter Blender (Soft HMM-style blending)
// ============================================================================

/// Configuration for regime-dependent parameters.
///
/// Per Small Fish Strategy (lines 262-265):
/// - κ varies 10x between calm and cascade
/// - γ varies 5x between calm and cascade
#[derive(Debug, Clone, Copy)]
pub struct RegimeParameterConfig {
    /// Base gamma (risk aversion) for Normal regime
    pub gamma_base: f64,
    /// Base kappa (fill intensity) for Normal regime
    pub kappa_base: f64,
}

impl Default for RegimeParameterConfig {
    fn default() -> Self {
        Self {
            gamma_base: 0.1,
            kappa_base: 1.0,
        }
    }
}

/// Regime-aware parameter blender using HMM belief state probabilities.
///
/// **Key principle from Small Fish Strategy:**
/// Use soft belief state probabilities, NOT hard regime switches.
///
/// γ_effective = P(low) × γ_low + P(normal) × γ_normal + P(high) × γ_high + P(extreme) × γ_extreme
/// κ_effective = P(low) × κ_low + P(normal) × κ_normal + P(high) × κ_high + P(extreme) × κ_extreme
///
/// This provides smooth transitions and handles regime uncertainty appropriately.
#[derive(Debug, Clone)]
pub struct RegimeParameterBlender {
    /// Base parameter configuration
    config: RegimeParameterConfig,

    /// Gamma multipliers per regime (risk aversion)
    /// Per Small Fish: γ varies 5x between calm and cascade
    gamma_multipliers: [f64; 4],

    /// Kappa multipliers per regime (fill intensity)
    /// Per Small Fish: κ varies 10x between calm and cascade
    kappa_multipliers: [f64; 4],
}

impl Default for RegimeParameterBlender {
    fn default() -> Self {
        Self::new(RegimeParameterConfig::default())
    }
}

impl RegimeParameterBlender {
    /// Create new blender with given base configuration.
    ///
    /// Default multipliers (per Small Fish Strategy):
    /// - Gamma: Low=0.5, Normal=1.0, High=2.0, Extreme=2.5 (5x range)
    /// - Kappa: Low=2.0, Normal=1.0, High=0.5, Extreme=0.2 (10x range)
    pub(crate) fn new(config: RegimeParameterConfig) -> Self {
        Self {
            config,
            // Gamma increases with regime severity (more risk averse in volatile markets)
            // 5x range: 0.5 to 2.5
            gamma_multipliers: [0.5, 1.0, 2.0, 2.5],
            // Kappa decreases with regime severity (fewer fills in volatile markets)
            // 10x range: 2.0 to 0.2
            kappa_multipliers: [2.0, 1.0, 0.5, 0.2],
        }
    }

    /// Create blender with custom multipliers.
    pub(crate) fn with_multipliers(
        config: RegimeParameterConfig,
        gamma_multipliers: [f64; 4],
        kappa_multipliers: [f64; 4],
    ) -> Self {
        Self {
            config,
            gamma_multipliers,
            kappa_multipliers,
        }
    }

    /// Blend gamma (risk aversion) using belief state probabilities.
    ///
    /// γ_effective = Σ P(regime_i) × γ_base × γ_multiplier_i
    ///
    /// Higher gamma = wider spreads = more conservative
    pub(crate) fn blend_gamma(&self, belief: &RegimeBeliefState) -> f64 {
        let probs = belief.as_array();
        let blended_multiplier: f64 = probs
            .iter()
            .zip(self.gamma_multipliers.iter())
            .map(|(&p, &m)| p * m)
            .sum();

        self.config.gamma_base * blended_multiplier
    }

    /// Blend kappa (fill intensity) using belief state probabilities.
    ///
    /// κ_effective = Σ P(regime_i) × κ_base × κ_multiplier_i
    ///
    /// Higher kappa = more fills expected = can quote tighter
    pub(crate) fn blend_kappa(&self, belief: &RegimeBeliefState) -> f64 {
        let probs = belief.as_array();
        let blended_multiplier: f64 = probs
            .iter()
            .zip(self.kappa_multipliers.iter())
            .map(|(&p, &m)| p * m)
            .sum();

        self.config.kappa_base * blended_multiplier
    }

    /// Blend spread multiplier using belief state.
    pub(crate) fn blend_spread_multiplier(&self, belief: &RegimeBeliefState) -> f64 {
        let spread_multipliers = [0.8, 1.0, 1.5, 2.5]; // Low, Normal, High, Extreme
        let probs = belief.as_array();
        probs
            .iter()
            .zip(spread_multipliers.iter())
            .map(|(&p, &m)| p * m)
            .sum()
    }

    /// Blend Kelly fraction multiplier using belief state.
    pub(crate) fn blend_kelly_multiplier(&self, belief: &RegimeBeliefState) -> f64 {
        let kelly_multipliers = [1.5, 1.0, 0.5, 0.25]; // Low, Normal, High, Extreme
        let probs = belief.as_array();
        probs
            .iter()
            .zip(kelly_multipliers.iter())
            .map(|(&p, &m)| p * m)
            .sum()
    }

    /// Get all blended parameters at once.
    pub(crate) fn blend_all(&self, belief: &RegimeBeliefState) -> BlendedParameters {
        BlendedParameters {
            gamma: self.blend_gamma(belief),
            kappa: self.blend_kappa(belief),
            spread_multiplier: self.blend_spread_multiplier(belief),
            kelly_multiplier: self.blend_kelly_multiplier(belief),
            belief: *belief,
        }
    }

    /// Get gamma for a specific regime (without blending).
    pub(crate) fn gamma_for_regime(&self, regime: VolatilityRegime) -> f64 {
        self.config.gamma_base * self.gamma_multipliers[regime.index()]
    }

    /// Get kappa for a specific regime (without blending).
    pub(crate) fn kappa_for_regime(&self, regime: VolatilityRegime) -> f64 {
        self.config.kappa_base * self.kappa_multipliers[regime.index()]
    }

    /// Verify the 10x kappa range (calm to cascade).
    pub(crate) fn kappa_range_ratio(&self) -> f64 {
        self.kappa_multipliers[0] / self.kappa_multipliers[3]
    }

    /// Verify the 5x gamma range (calm to cascade).
    pub(crate) fn gamma_range_ratio(&self) -> f64 {
        self.gamma_multipliers[3] / self.gamma_multipliers[0]
    }
}

/// Result of blending all parameters for a given belief state.
#[derive(Debug, Clone, Copy)]
pub struct BlendedParameters {
    /// Blended gamma (risk aversion)
    pub gamma: f64,
    /// Blended kappa (fill intensity)
    pub kappa: f64,
    /// Blended spread multiplier
    pub spread_multiplier: f64,
    /// Blended Kelly fraction multiplier
    pub kelly_multiplier: f64,
    /// The belief state used for blending
    pub belief: RegimeBeliefState,
}

impl BlendedParameters {
    /// Check if parameters suggest defensive posture.
    pub(crate) fn is_defensive(&self) -> bool {
        self.spread_multiplier > 1.5 || self.kelly_multiplier < 0.5
    }
}

// ============================================================================
// Belief State Updater (Simple HMM-style updates)
// ============================================================================

/// Updates belief state based on observations (simplified HMM).
///
/// Uses soft updates based on how well observations match each regime,
/// rather than hard switches.
#[derive(Debug, Clone)]
pub(crate) struct BeliefStateUpdater {
    /// Learning rate for belief updates
    learning_rate: f64,
    /// Baseline sigma for regime classification
    baseline_sigma: f64,
    /// Regime thresholds
    low_threshold: f64,
    high_threshold: f64,
    extreme_threshold: f64,
    /// Jump ratio threshold
    jump_threshold: f64,
}

impl BeliefStateUpdater {
    /// Create new updater with default thresholds.
    pub(crate) fn new(baseline_sigma: f64) -> Self {
        Self {
            learning_rate: 0.1,
            baseline_sigma,
            low_threshold: 0.5,
            high_threshold: 1.5,
            extreme_threshold: 3.0,
            jump_threshold: 2.0,
        }
    }

    /// Create with custom learning rate.
    pub(crate) fn with_learning_rate(baseline_sigma: f64, learning_rate: f64) -> Self {
        Self {
            learning_rate: learning_rate.clamp(0.01, 0.5),
            baseline_sigma,
            low_threshold: 0.5,
            high_threshold: 1.5,
            extreme_threshold: 3.0,
            jump_threshold: 2.0,
        }
    }

    /// Update baseline sigma (e.g., from EWMA).
    pub(crate) fn update_baseline(&mut self, new_baseline: f64) {
        if new_baseline > 1e-9 {
            self.baseline_sigma = new_baseline;
        }
    }

    /// Compute observation likelihood for each regime (soft classification).
    fn observation_likelihoods(&self, sigma: f64, jump_ratio: f64) -> [f64; 4] {
        let sigma_ratio = sigma / self.baseline_sigma.max(1e-9);

        // Use Gaussian-like likelihood based on distance from regime center
        // Low regime centered at low_threshold/2
        let low_center = self.low_threshold / 2.0;
        let low_likelihood = (-((sigma_ratio - low_center).powi(2)) / 0.5).exp();

        // Normal regime centered at 1.0
        let normal_likelihood = (-((sigma_ratio - 1.0).powi(2)) / 0.5).exp();

        // High regime centered at (high + extreme) / 2
        let high_center = (self.high_threshold + self.extreme_threshold) / 2.0;
        let high_likelihood = (-((sigma_ratio - high_center).powi(2)) / 1.0).exp();

        // Extreme regime: high likelihood when sigma_ratio > extreme_threshold OR jump_ratio high
        let extreme_sigma = if sigma_ratio > self.extreme_threshold {
            1.0
        } else {
            (-(self.extreme_threshold - sigma_ratio).powi(2) / 2.0).exp()
        };
        let extreme_jump = if jump_ratio > self.jump_threshold {
            1.0
        } else {
            (jump_ratio / self.jump_threshold).powi(2)
        };
        let extreme_likelihood = extreme_sigma.max(extreme_jump);

        // Normalize to sum to 1
        let sum = low_likelihood + normal_likelihood + high_likelihood + extreme_likelihood;
        if sum > 1e-9 {
            [
                low_likelihood / sum,
                normal_likelihood / sum,
                high_likelihood / sum,
                extreme_likelihood / sum,
            ]
        } else {
            [0.1, 0.7, 0.15, 0.05] // Default
        }
    }

    /// Update belief state based on new observation.
    ///
    /// Uses Bayesian-like update: new_belief ∝ prior × likelihood
    pub(crate) fn update(&self, belief: &mut RegimeBeliefState, sigma: f64, jump_ratio: f64) {
        let likelihoods = self.observation_likelihoods(sigma, jump_ratio);

        // Weighted update: belief = (1-α) × belief + α × likelihood
        belief.p_low =
            (1.0 - self.learning_rate) * belief.p_low + self.learning_rate * likelihoods[0];
        belief.p_normal =
            (1.0 - self.learning_rate) * belief.p_normal + self.learning_rate * likelihoods[1];
        belief.p_high =
            (1.0 - self.learning_rate) * belief.p_high + self.learning_rate * likelihoods[2];
        belief.p_extreme =
            (1.0 - self.learning_rate) * belief.p_extreme + self.learning_rate * likelihoods[3];

        // Ensure normalization
        belief.normalize();
    }

    /// Fast path: update and return blended parameters.
    pub(crate) fn update_and_blend(
        &self,
        belief: &mut RegimeBeliefState,
        sigma: f64,
        jump_ratio: f64,
        blender: &RegimeParameterBlender,
    ) -> BlendedParameters {
        self.update(belief, sigma, jump_ratio);
        blender.blend_all(belief)
    }
}

/// Result of warmup calibration from observed market data.
#[derive(Debug, Clone)]
struct CalibrationResult {
    /// Baseline volatility (median of observations)
    baseline: f64,
    /// Extreme threshold multiplier (P95/median, min 3.0)
    extreme_threshold: f64,
    /// Low threshold multiplier (P10/median, max 0.7)
    low_threshold: f64,
    /// High threshold multiplier (P75/median, min 1.3)
    high_threshold: f64,
}

/// Self-calibrating warmup that learns volatility thresholds from observed data.
///
/// Instead of hardcoding regime thresholds, this collects observations during warmup
/// and computes percentile-based thresholds that adapt to the actual market.
#[derive(Debug)]
struct WarmupCalibrator {
    /// Collected sigma observations
    observations: Vec<f64>,
    /// Minimum observations before calibration (default: 50)
    min_observations: usize,
    /// Whether calibration has been performed
    is_calibrated: bool,
}

impl WarmupCalibrator {
    fn new(min_observations: usize) -> Self {
        Self {
            observations: Vec::with_capacity(min_observations + 10),
            min_observations,
            is_calibrated: false,
        }
    }

    /// Add an observation and return calibration result when ready.
    fn add_observation(&mut self, sigma: f64) -> Option<CalibrationResult> {
        if self.is_calibrated {
            return None;
        }

        // Filter out clearly invalid observations
        if sigma > 1e-9 && sigma < 0.1 {
            self.observations.push(sigma);
        }

        if self.observations.len() >= self.min_observations {
            let result = self.compute_calibration();
            self.is_calibrated = true;
            Some(result)
        } else {
            None
        }
    }

    /// Compute calibration from collected observations.
    fn compute_calibration(&self) -> CalibrationResult {
        let mut sorted = self.observations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        let median = sorted[len / 2];
        let p10 = sorted[(len as f64 * 0.10) as usize];
        let p75 = sorted[(len as f64 * 0.75) as usize];
        let p95 = sorted[(len as f64 * 0.95) as usize];

        CalibrationResult {
            baseline: median,
            // Extreme: P95/median but at least 3.0 to avoid false positives
            extreme_threshold: (p95 / median).max(3.0),
            // Low: P10/median but at most 0.7 to ensure it's truly quiet
            low_threshold: (p10 / median).min(0.7),
            // High: P75/median but at least 1.3 to distinguish from normal
            high_threshold: (p75 / median).max(1.3),
        }
    }

    /// Get warmup progress as (current, required).
    fn progress(&self) -> (usize, usize) {
        (self.observations.len(), self.min_observations)
    }
}

/// Tracks volatility regime with asymmetric hysteresis to prevent rapid switching.
///
/// Transitions between states require sustained conditions to trigger,
/// preventing oscillation at boundaries.
///
/// **Self-calibrating:** During warmup, collects volatility observations and
/// computes percentile-based thresholds that adapt to the actual market.
/// This eliminates the need for hardcoded "magic numbers" that may not match
/// different assets or market conditions.
#[derive(Debug)]
pub(crate) struct VolatilityRegimeTracker {
    /// Current regime state
    regime: VolatilityRegime,
    /// Baseline volatility (for regime thresholds)
    baseline_sigma: f64,
    /// Consecutive updates in potential new regime (for hysteresis)
    transition_count: u32,
    /// Minimum transitions for escalating to higher regime (fast: vol spikes)
    min_transitions_escalate: u32,
    /// Minimum transitions for de-escalating to lower regime (slow: vol mean-reverts)
    min_transitions_deescalate: u32,
    /// Thresholds relative to baseline
    low_threshold: f64, // σ < baseline × low_threshold → Low
    high_threshold: f64,    // σ > baseline × high_threshold → High
    extreme_threshold: f64, // σ > baseline × extreme_threshold → Extreme
    /// Jump ratio threshold for Extreme regime
    jump_threshold: f64,
    /// Pending regime (for hysteresis tracking)
    pending_regime: Option<VolatilityRegime>,
    /// Self-calibrating warmup (learns thresholds from observed data)
    calibrator: WarmupCalibrator,
}

impl VolatilityRegimeTracker {
    /// Number of observations required for calibration.
    const CALIBRATION_OBSERVATIONS: usize = 50;

    pub(crate) fn new(baseline_sigma: f64) -> Self {
        Self {
            regime: VolatilityRegime::Normal,
            baseline_sigma,
            transition_count: 0,
            // Balanced hysteresis: escalate (3 ticks), de-escalate (4 ticks)
            min_transitions_escalate: 3,
            min_transitions_deescalate: 4,
            // Initial thresholds (will be overwritten by calibrator)
            // These serve as fallback if calibration fails or during warmup
            low_threshold: 0.5,
            high_threshold: 1.5,
            extreme_threshold: 5.0,
            jump_threshold: 3.0,
            pending_regime: None,
            calibrator: WarmupCalibrator::new(Self::CALIBRATION_OBSERVATIONS),
        }
    }

    /// Determine if a transition is escalating (moving to higher risk regime).
    fn is_escalation(&self, from: VolatilityRegime, to: VolatilityRegime) -> bool {
        let from_level = match from {
            VolatilityRegime::Low => 0,
            VolatilityRegime::Normal => 1,
            VolatilityRegime::High => 2,
            VolatilityRegime::Extreme => 3,
        };
        let to_level = match to {
            VolatilityRegime::Low => 0,
            VolatilityRegime::Normal => 1,
            VolatilityRegime::High => 2,
            VolatilityRegime::Extreme => 3,
        };
        to_level > from_level
    }

    /// Get required transitions for a regime change (asymmetric).
    fn required_transitions(&self, from: VolatilityRegime, to: VolatilityRegime) -> u32 {
        if self.is_escalation(from, to) {
            self.min_transitions_escalate
        } else {
            self.min_transitions_deescalate
        }
    }

    /// Update regime based on current volatility and jump ratio.
    ///
    /// Uses asymmetric hysteresis:
    /// - Escalation (to higher risk): 3 ticks - react quickly to vol spikes
    /// - De-escalation (to lower risk): 4 ticks - confirm vol has truly subsided
    ///
    /// **Self-calibrating:** During warmup, collects sigma observations and
    /// computes percentile-based thresholds when enough data is collected.
    pub(crate) fn update(&mut self, sigma: f64, jump_ratio: f64) {
        // Self-calibration: collect observations and calibrate when ready
        if let Some(calibration) = self.calibrator.add_observation(sigma) {
            self.baseline_sigma = calibration.baseline;
            self.low_threshold = calibration.low_threshold;
            self.high_threshold = calibration.high_threshold;
            self.extreme_threshold = calibration.extreme_threshold;
            info!(
                baseline = %format!("{:.6}", calibration.baseline),
                low_threshold = %format!("{:.2}", calibration.low_threshold),
                high_threshold = %format!("{:.2}", calibration.high_threshold),
                extreme_threshold = %format!("{:.2}", calibration.extreme_threshold),
                observations = self.calibrator.progress().0,
                "Volatility regime self-calibrated from observed market data"
            );
        }

        // Determine target regime based on current conditions
        let target = self.classify(sigma, jump_ratio);

        // Check if target matches pending transition
        if let Some(pending) = self.pending_regime {
            if pending == target {
                self.transition_count += 1;
                // Use asymmetric hysteresis: fast escalation, slow de-escalation
                let required = self.required_transitions(self.regime, target);
                if self.transition_count >= required {
                    // Transition confirmed
                    if self.regime != target {
                        let direction = if self.is_escalation(self.regime, target) {
                            "escalation"
                        } else {
                            "de-escalation"
                        };
                        debug!(
                            from = ?self.regime,
                            to = ?target,
                            sigma = %format!("{:.6}", sigma),
                            jump_ratio = %format!("{:.2}", jump_ratio),
                            direction = direction,
                            ticks_required = required,
                            "Volatility regime transition"
                        );
                    }
                    self.regime = target;
                    self.pending_regime = None;
                    self.transition_count = 0;
                }
            } else {
                // Target changed, reset hysteresis
                self.pending_regime = Some(target);
                self.transition_count = 1;
            }
        } else if target != self.regime {
            // Start new pending transition
            self.pending_regime = Some(target);
            self.transition_count = 1;
        }
    }

    /// Classify conditions into target regime.
    fn classify(&self, sigma: f64, jump_ratio: f64) -> VolatilityRegime {
        // Jump ratio overrides to Extreme
        if jump_ratio > self.jump_threshold {
            return VolatilityRegime::Extreme;
        }

        // Volatility-based classification
        let sigma_ratio = sigma / self.baseline_sigma.max(1e-9);

        if sigma_ratio < self.low_threshold {
            VolatilityRegime::Low
        } else if sigma_ratio > self.extreme_threshold {
            VolatilityRegime::Extreme
        } else if sigma_ratio > self.high_threshold {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Normal
        }
    }

    /// Get current regime.
    pub(crate) fn regime(&self) -> VolatilityRegime {
        self.regime
    }

    /// Update baseline volatility (e.g., from long-term EWMA).
    pub(crate) fn update_baseline(&mut self, new_baseline: f64) {
        if new_baseline > 1e-9 {
            // Slow update to baseline (EWMA with long half-life)
            self.baseline_sigma = 0.99 * self.baseline_sigma + 0.01 * new_baseline;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_regime_multipliers() {
        assert!((VolatilityRegime::Low.spread_multiplier() - 0.8).abs() < 0.01);
        assert!((VolatilityRegime::Normal.spread_multiplier() - 1.0).abs() < 0.01);
        assert!((VolatilityRegime::High.spread_multiplier() - 1.5).abs() < 0.01);
        assert!((VolatilityRegime::Extreme.spread_multiplier() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_volatility_regime_gamma_multipliers() {
        assert!((VolatilityRegime::Low.gamma_multiplier() - 0.8).abs() < 0.01);
        assert!((VolatilityRegime::Normal.gamma_multiplier() - 1.0).abs() < 0.01);
        assert!((VolatilityRegime::High.gamma_multiplier() - 1.5).abs() < 0.01);
        assert!((VolatilityRegime::Extreme.gamma_multiplier() - 1.8).abs() < 0.01);
        // Reduced from 3.0
    }

    #[test]
    fn test_warmup_calibrator() {
        let mut calibrator = WarmupCalibrator::new(10); // Small for testing

        // Add observations simulating realistic BTC volatility
        let observations = [
            0.00020, 0.00022, 0.00018, 0.00025, 0.00030, 0.00019, 0.00021, 0.00023, 0.00028,
            0.00024,
        ];

        for (i, &sigma) in observations.iter().enumerate() {
            let result = calibrator.add_observation(sigma);
            if i < 9 {
                assert!(
                    result.is_none(),
                    "Should not calibrate before min observations"
                );
            } else {
                assert!(result.is_some(), "Should calibrate at min observations");
                let cal = result.unwrap();
                // Median should be around 0.000225 (middle of sorted values)
                assert!(cal.baseline > 0.00020 && cal.baseline < 0.00025);
                // Extreme threshold should be at least 3.0
                assert!(cal.extreme_threshold >= 3.0);
            }
        }

        // Calibrator should not produce more results
        assert!(calibrator.add_observation(0.00025).is_none());
    }

    // ========================================================================
    // Regime Belief State Tests
    // ========================================================================

    #[test]
    fn test_belief_state_default() {
        let belief = RegimeBeliefState::default();
        // Should sum to 1.0
        let sum = belief.p_low + belief.p_normal + belief.p_high + belief.p_extreme;
        assert!((sum - 1.0).abs() < 0.001, "Probabilities should sum to 1.0");
        // Normal should be most likely by default
        assert!(belief.p_normal > belief.p_low);
        assert!(belief.p_normal > belief.p_high);
        assert!(belief.p_normal > belief.p_extreme);
    }

    #[test]
    fn test_belief_state_from_regime() {
        let belief = RegimeBeliefState::from_regime(VolatilityRegime::High);
        assert!((belief.p_high - 1.0).abs() < 0.001);
        assert!(belief.p_low < 0.001);
        assert!(belief.p_normal < 0.001);
        assert!(belief.p_extreme < 0.001);
    }

    #[test]
    fn test_belief_state_uniform() {
        let belief = RegimeBeliefState::uniform();
        assert!((belief.p_low - 0.25).abs() < 0.001);
        assert!((belief.p_normal - 0.25).abs() < 0.001);
        assert!((belief.p_high - 0.25).abs() < 0.001);
        assert!((belief.p_extreme - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_belief_state_normalize() {
        let mut belief = RegimeBeliefState {
            p_low: 0.2,
            p_normal: 0.4,
            p_high: 0.2,
            p_extreme: 0.2,
        };
        belief.normalize();
        let sum = belief.p_low + belief.p_normal + belief.p_high + belief.p_extreme;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_belief_state_most_likely() {
        let belief = RegimeBeliefState {
            p_low: 0.1,
            p_normal: 0.2,
            p_high: 0.6,
            p_extreme: 0.1,
        };
        assert_eq!(belief.most_likely(), VolatilityRegime::High);
    }

    #[test]
    fn test_belief_state_entropy() {
        // Uniform = max entropy
        let uniform = RegimeBeliefState::uniform();
        let max_entropy = uniform.entropy();
        assert!(max_entropy > 1.3, "Uniform should have high entropy");

        // Concentrated = low entropy
        let concentrated = RegimeBeliefState::from_regime(VolatilityRegime::Normal);
        let low_entropy = concentrated.entropy();
        assert!(low_entropy < 0.1, "Concentrated should have low entropy");

        assert!(max_entropy > low_entropy);
    }

    #[test]
    fn test_belief_state_confidence() {
        let confident = RegimeBeliefState::from_regime(VolatilityRegime::Normal);
        assert!(confident.is_confident());

        let uncertain = RegimeBeliefState::uniform();
        assert!(!uncertain.is_confident());
    }

    // ========================================================================
    // Regime Parameter Blender Tests
    // ========================================================================

    #[test]
    fn test_blender_default() {
        let blender = RegimeParameterBlender::default();
        // Verify 10x kappa range (calm to cascade)
        let ratio = blender.kappa_range_ratio();
        assert!(
            (ratio - 10.0).abs() < 0.1,
            "Kappa should vary 10x, got {}",
            ratio
        );

        // Verify 5x gamma range (calm to cascade)
        let gamma_ratio = blender.gamma_range_ratio();
        assert!(
            (gamma_ratio - 5.0).abs() < 0.1,
            "Gamma should vary 5x, got {}",
            gamma_ratio
        );
    }

    #[test]
    fn test_blend_gamma_pure_regimes() {
        let config = RegimeParameterConfig {
            gamma_base: 0.1,
            kappa_base: 1.0,
        };
        let blender = RegimeParameterBlender::new(config);

        // Pure Low regime
        let low_belief = RegimeBeliefState::from_regime(VolatilityRegime::Low);
        let gamma_low = blender.blend_gamma(&low_belief);
        assert!(
            (gamma_low - 0.05).abs() < 0.001,
            "Low gamma should be 0.1 * 0.5 = 0.05"
        );

        // Pure Normal regime
        let normal_belief = RegimeBeliefState::from_regime(VolatilityRegime::Normal);
        let gamma_normal = blender.blend_gamma(&normal_belief);
        assert!(
            (gamma_normal - 0.1).abs() < 0.001,
            "Normal gamma should be 0.1 * 1.0 = 0.1"
        );

        // Pure Extreme regime
        let extreme_belief = RegimeBeliefState::from_regime(VolatilityRegime::Extreme);
        let gamma_extreme = blender.blend_gamma(&extreme_belief);
        assert!(
            (gamma_extreme - 0.25).abs() < 0.001,
            "Extreme gamma should be 0.1 * 2.5 = 0.25"
        );
    }

    #[test]
    fn test_blend_kappa_pure_regimes() {
        let config = RegimeParameterConfig {
            gamma_base: 0.1,
            kappa_base: 1.0,
        };
        let blender = RegimeParameterBlender::new(config);

        // Pure Low regime (high kappa = more fills)
        let low_belief = RegimeBeliefState::from_regime(VolatilityRegime::Low);
        let kappa_low = blender.blend_kappa(&low_belief);
        assert!(
            (kappa_low - 2.0).abs() < 0.001,
            "Low kappa should be 1.0 * 2.0 = 2.0"
        );

        // Pure Extreme regime (low kappa = fewer fills)
        let extreme_belief = RegimeBeliefState::from_regime(VolatilityRegime::Extreme);
        let kappa_extreme = blender.blend_kappa(&extreme_belief);
        assert!(
            (kappa_extreme - 0.2).abs() < 0.001,
            "Extreme kappa should be 1.0 * 0.2 = 0.2"
        );
    }

    #[test]
    fn test_blend_gamma_mixed_belief() {
        let blender = RegimeParameterBlender::default();

        // 50% Normal, 50% High
        let mixed = RegimeBeliefState {
            p_low: 0.0,
            p_normal: 0.5,
            p_high: 0.5,
            p_extreme: 0.0,
        };

        let gamma = blender.blend_gamma(&mixed);
        // Expected: 0.1 * (0.5 * 1.0 + 0.5 * 2.0) = 0.1 * 1.5 = 0.15
        assert!(
            (gamma - 0.15).abs() < 0.001,
            "Mixed gamma should be 0.15, got {}",
            gamma
        );
    }

    #[test]
    fn test_blend_all() {
        let blender = RegimeParameterBlender::default();
        let belief = RegimeBeliefState::default();

        let params = blender.blend_all(&belief);
        assert!(params.gamma > 0.0);
        assert!(params.kappa > 0.0);
        assert!(params.spread_multiplier > 0.0);
        assert!(params.kelly_multiplier > 0.0);
    }

    #[test]
    fn test_blended_parameters_defensive() {
        let blender = RegimeParameterBlender::default();

        // Extreme regime should be defensive
        let extreme_belief = RegimeBeliefState::from_regime(VolatilityRegime::Extreme);
        let params = blender.blend_all(&extreme_belief);
        assert!(params.is_defensive(), "Extreme regime should be defensive");

        // Low regime should not be defensive
        let low_belief = RegimeBeliefState::from_regime(VolatilityRegime::Low);
        let params_low = blender.blend_all(&low_belief);
        assert!(
            !params_low.is_defensive(),
            "Low regime should not be defensive"
        );
    }

    // ========================================================================
    // Belief State Updater Tests
    // ========================================================================

    #[test]
    fn test_belief_updater_new() {
        let updater = BeliefStateUpdater::new(0.0002);
        assert!((updater.baseline_sigma - 0.0002).abs() < 1e-9);
    }

    #[test]
    fn test_belief_updater_low_volatility() {
        let updater = BeliefStateUpdater::new(0.001);
        let mut belief = RegimeBeliefState::default();

        // Feed very low sigma observations
        for _ in 0..20 {
            updater.update(&mut belief, 0.0002, 1.0); // sigma_ratio = 0.2
        }

        // Should shift toward Low regime
        assert!(
            belief.p_low > belief.p_extreme,
            "Low prob {} should exceed Extreme prob {} after low vol observations",
            belief.p_low,
            belief.p_extreme
        );
    }

    #[test]
    fn test_belief_updater_high_jump_ratio() {
        let updater = BeliefStateUpdater::new(0.001);
        let mut belief = RegimeBeliefState::default();

        // Feed high jump ratio observations
        for _ in 0..20 {
            updater.update(&mut belief, 0.001, 5.0); // high jump ratio
        }

        // Should shift toward Extreme regime
        assert!(
            belief.p_extreme > belief.p_low,
            "Extreme prob {} should exceed Low prob {} after high jump observations",
            belief.p_extreme,
            belief.p_low
        );
    }

    #[test]
    fn test_belief_updater_normal_conditions() {
        let updater = BeliefStateUpdater::new(0.001);
        let mut belief = RegimeBeliefState::uniform();

        // Feed normal sigma observations
        for _ in 0..30 {
            updater.update(&mut belief, 0.001, 1.0); // sigma_ratio = 1.0
        }

        // Should shift toward Normal regime
        assert!(
            belief.p_normal > belief.p_extreme,
            "Normal prob {} should exceed Extreme prob {} after normal observations",
            belief.p_normal,
            belief.p_extreme
        );
    }

    #[test]
    fn test_update_and_blend() {
        let updater = BeliefStateUpdater::new(0.001);
        let blender = RegimeParameterBlender::default();
        let mut belief = RegimeBeliefState::default();

        // Update and get blended params in one call
        let params = updater.update_and_blend(&mut belief, 0.001, 1.0, &blender);

        assert!(params.gamma > 0.0);
        assert!(params.kappa > 0.0);
    }

    #[test]
    fn test_regime_index_roundtrip() {
        for regime in [
            VolatilityRegime::Low,
            VolatilityRegime::Normal,
            VolatilityRegime::High,
            VolatilityRegime::Extreme,
        ] {
            let idx = regime.index();
            let restored = VolatilityRegime::from_index(idx);
            assert_eq!(regime, restored);
        }
    }

    #[test]
    fn test_small_fish_kappa_gamma_ranges() {
        // Per Small Fish Strategy lines 262-265:
        // - κ varies 10x between calm and cascade
        // - γ varies 5x between calm and cascade

        let blender = RegimeParameterBlender::default();

        // Kappa: Low=2.0, Extreme=0.2 -> 10x range
        let kappa_low = blender.kappa_for_regime(VolatilityRegime::Low);
        let kappa_extreme = blender.kappa_for_regime(VolatilityRegime::Extreme);
        let kappa_ratio = kappa_low / kappa_extreme;
        assert!(
            (kappa_ratio - 10.0).abs() < 0.1,
            "Kappa ratio should be 10x: {} / {} = {}",
            kappa_low,
            kappa_extreme,
            kappa_ratio
        );

        // Gamma: Low=0.5, Extreme=2.5 -> 5x range
        let gamma_low = blender.gamma_for_regime(VolatilityRegime::Low);
        let gamma_extreme = blender.gamma_for_regime(VolatilityRegime::Extreme);
        let gamma_ratio = gamma_extreme / gamma_low;
        assert!(
            (gamma_ratio - 5.0).abs() < 0.1,
            "Gamma ratio should be 5x: {} / {} = {}",
            gamma_extreme,
            gamma_low,
            gamma_ratio
        );
    }

    #[test]
    fn test_soft_blending_vs_hard_switch() {
        // Demonstrate that soft blending gives smoother transitions than hard switches
        let blender = RegimeParameterBlender::default();

        // Hard switch: instant jump
        let low_gamma = blender.gamma_for_regime(VolatilityRegime::Low);
        let normal_gamma = blender.gamma_for_regime(VolatilityRegime::Normal);
        let hard_jump = (normal_gamma - low_gamma).abs();

        // Soft blend: gradual transition
        let belief_90_low = RegimeBeliefState {
            p_low: 0.9,
            p_normal: 0.1,
            p_high: 0.0,
            p_extreme: 0.0,
        };
        let belief_50_50 = RegimeBeliefState {
            p_low: 0.5,
            p_normal: 0.5,
            p_high: 0.0,
            p_extreme: 0.0,
        };
        let belief_10_low = RegimeBeliefState {
            p_low: 0.1,
            p_normal: 0.9,
            p_high: 0.0,
            p_extreme: 0.0,
        };

        let gamma_90_low = blender.blend_gamma(&belief_90_low);
        let gamma_50_50 = blender.blend_gamma(&belief_50_50);
        let gamma_10_low = blender.blend_gamma(&belief_10_low);

        // Soft blending should give intermediate values
        assert!(gamma_90_low < gamma_50_50);
        assert!(gamma_50_50 < gamma_10_low);
        assert!(gamma_10_low < normal_gamma);

        // Soft steps should be smaller than hard jump
        let soft_step = (gamma_50_50 - gamma_90_low).abs();
        assert!(
            soft_step < hard_jump,
            "Soft step {} should be smaller than hard jump {}",
            soft_step,
            hard_jump
        );
    }
}
