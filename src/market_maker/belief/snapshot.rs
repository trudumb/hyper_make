//! BeliefSnapshot and component belief structures.
//!
//! This module defines the read-only snapshot that consumers use to
//! access the centralized belief state. The snapshot is a point-in-time
//! copy that can be read without locks.
//!
//! ## Design
//!
//! The snapshot is divided into logical belief groups:
//! - **DriftVolatilityBeliefs**: Price drift and volatility posteriors
//! - **KappaBeliefs**: Fill intensity estimation
//! - **ContinuationBeliefs**: Position continuation model
//! - **RegimeBeliefs**: Market regime (HMM)
//! - **ChangepointBeliefs**: BOCD state
//! - **EdgeBeliefs**: Expected trading edge
//! - **CalibrationState**: Model quality metrics

use super::Regime;

/// Complete snapshot of all belief state.
///
/// This is the primary interface for consumers. All fields are
/// public for easy access without getter methods.
#[derive(Debug, Clone, Default)]
pub struct BeliefSnapshot {
    // === Core Bayesian Posteriors ===
    /// Drift and volatility beliefs (Normal-Inverse-Gamma)
    pub drift_vol: DriftVolatilityBeliefs,

    /// Fill intensity beliefs (confidence-weighted blend)
    pub kappa: KappaBeliefs,

    /// Position continuation beliefs (Beta-Binomial + signals)
    pub continuation: ContinuationBeliefs,

    // === Regime Detection ===
    /// Regime beliefs (4-state HMM)
    pub regime: RegimeBeliefs,

    /// Changepoint beliefs (BOCD)
    pub changepoint: ChangepointBeliefs,

    // === Edge Estimates ===
    /// Trading edge beliefs
    pub edge: EdgeBeliefs,

    // === Microstructure (Phase 1: Alpha-generating) ===
    /// Microstructure-based toxicity signals
    pub microstructure: MicrostructureBeliefs,

    // === Cross-Venue (Bivariate Flow Model) ===
    /// Cross-venue beliefs from joint Binance + Hyperliquid analysis
    pub cross_venue: CrossVenueBeliefs,

    // === Calibration ===
    /// Model calibration metrics
    pub calibration: CalibrationState,

    // === Meta ===
    /// Statistics about the belief state
    pub stats: BeliefStats,
}

impl BeliefSnapshot {
    /// Check if beliefs are warmed up (enough observations).
    pub fn is_warmed_up(&self) -> bool {
        self.stats.is_warmed_up
    }

    /// Get overall confidence in beliefs [0, 1].
    pub fn overall_confidence(&self) -> f64 {
        // Geometric mean of component confidences
        let confidences = [
            self.drift_vol.confidence,
            self.kappa.confidence,
            self.regime.confidence,
        ];
        let product: f64 = confidences.iter().product();
        product.powf(1.0 / confidences.len() as f64)
    }

    /// Get the current regime.
    pub fn current_regime(&self) -> Regime {
        self.regime.current
    }

    /// Get predictive bias β_t = E[μ | data].
    pub fn predictive_bias(&self) -> f64 {
        self.drift_vol.expected_drift
    }

    /// Get effective kappa (EWMA-smoothed).
    pub fn kappa_effective(&self) -> f64 {
        self.kappa.kappa_effective
    }

    /// Get fused continuation probability.
    pub fn p_continuation(&self) -> f64 {
        self.continuation.p_fused
    }

    /// Get expected edge.
    pub fn expected_edge(&self) -> f64 {
        self.edge.expected_edge
    }

    /// Get fill prediction information ratio.
    pub fn fill_ir(&self) -> f64 {
        self.calibration.fill.information_ratio
    }

    /// Get AS prediction information ratio.
    pub fn as_ir(&self) -> f64 {
        self.calibration.adverse_selection.information_ratio
    }
}

// =============================================================================
// Component Belief Structures
// =============================================================================

/// Drift and volatility beliefs from Normal-Inverse-Gamma posterior.
#[derive(Debug, Clone)]
pub struct DriftVolatilityBeliefs {
    /// E[μ | data] - posterior mean of drift (THE predictive signal)
    pub expected_drift: f64,

    /// Uncertainty in drift estimate (posterior std of μ)
    pub drift_uncertainty: f64,

    /// E[σ | data] - posterior mean of volatility
    pub expected_sigma: f64,

    /// Probability that drift is negative (bearish)
    pub prob_bearish: f64,

    /// Probability that drift is positive (bullish)
    pub prob_bullish: f64,

    /// Confidence in drift estimate [0, 1]
    pub confidence: f64,

    /// Number of price observations
    pub n_observations: u64,

    // === Fat-Tail Skewness Fields (Phase 2A: Statistical Refinements) ===

    /// Skewness of the sigma posterior distribution.
    ///
    /// Positive = right-skewed (vol spike risk).
    /// Typical values: 0.5 to 2.0 for vol processes.
    /// Use for asymmetric spread adjustment.
    pub sigma_skewness: f64,

    /// Excess kurtosis of sigma posterior (fat tails indicator).
    ///
    /// 0 = Gaussian, >0 = fat tails.
    /// High kurtosis means extreme moves more likely than Gaussian predicts.
    pub sigma_kurtosis: f64,

    /// Skewness of drift belief distribution.
    ///
    /// Asymmetry in drift belief; affects directional confidence.
    pub drift_skewness: f64,
}

impl Default for DriftVolatilityBeliefs {
    fn default() -> Self {
        Self {
            expected_drift: 0.0,
            drift_uncertainty: 0.01,
            expected_sigma: 0.02, // ~2% daily vol
            prob_bearish: 0.5,
            prob_bullish: 0.5,
            confidence: 0.0,
            n_observations: 0,
            // Phase 2A: Fat-tail skewness (neutral defaults)
            sigma_skewness: 0.5,   // Slight positive (typical for vol)
            sigma_kurtosis: 0.0,   // Gaussian
            drift_skewness: 0.0,   // Symmetric
        }
    }
}

impl DriftVolatilityBeliefs {
    /// Check if vol spike risk is elevated based on skewness.
    ///
    /// Returns true if sigma_skewness > 1.0 (right-skewed distribution).
    /// When true, consider widening spreads asymmetrically.
    pub fn has_vol_spike_risk(&self) -> bool {
        self.sigma_skewness > 1.0
    }

    /// Check if distribution has fat tails.
    ///
    /// Returns true if sigma_kurtosis > 3.0 (excess kurtosis).
    /// Fat tails mean extreme moves are more likely than Gaussian predicts.
    pub fn has_fat_tails(&self) -> bool {
        self.sigma_kurtosis > 3.0
    }

    /// Get asymmetric spread adjustment factor for bid side.
    ///
    /// When sigma is right-skewed (vol spike risk), tighten bid spread.
    /// Returns a multiplier in [0.8, 1.0].
    pub fn bid_spread_factor(&self, sensitivity: f64) -> f64 {
        let skew_factor = (self.sigma_skewness * sensitivity).tanh();
        (1.0 - skew_factor * 0.1).clamp(0.8, 1.0)
    }

    /// Get asymmetric spread adjustment factor for ask side.
    ///
    /// When sigma is right-skewed (vol spike risk), widen ask spread.
    /// Returns a multiplier in [1.0, 1.2].
    pub fn ask_spread_factor(&self, sensitivity: f64) -> f64 {
        let skew_factor = (self.sigma_skewness * sensitivity).tanh();
        (1.0 + skew_factor * 0.1).clamp(1.0, 1.2)
    }

    /// Get a "tail risk" score [0, 1] combining skewness and kurtosis.
    ///
    /// Higher values indicate more extreme event risk.
    pub fn tail_risk_score(&self) -> f64 {
        let skew_contrib = (self.sigma_skewness / 2.0).min(1.0);
        let kurt_contrib = (self.sigma_kurtosis / 6.0).min(1.0);
        (0.6 * skew_contrib + 0.4 * kurt_contrib).clamp(0.0, 1.0)
    }
}

/// Fill intensity (kappa) beliefs from confidence-weighted blending.
///
/// ## Uncertainty Propagation (Phase 2)
///
/// Includes posterior uncertainty for kappa and derived spread confidence
/// intervals. Use `spread_ci_lower` and `spread_ci_upper` to assess the
/// range of plausible optimal spreads.
#[derive(Debug, Clone)]
pub struct KappaBeliefs {
    /// EWMA-smoothed effective kappa
    pub kappa_effective: f64,

    /// Raw (unsmoothed) kappa
    pub kappa_raw: f64,

    /// Component breakdown: (kappa, weight)
    pub components: KappaComponents,

    /// Overall confidence in kappa estimate [0, 1]
    pub confidence: f64,

    /// Is this from warmup (prior-dominated)?
    pub is_warmup: bool,

    /// Number of own fills observed
    pub n_own_fills: usize,

    // === Uncertainty Fields (Phase 2: Alpha-generating) ===

    /// Standard deviation of kappa posterior.
    ///
    /// Higher values indicate more uncertainty in fill intensity.
    /// Use for spread widening when uncertain.
    pub kappa_std: f64,

    /// Lower bound of 95% CI for optimal spread (bps).
    ///
    /// Derived from joint (κ, σ) uncertainty using delta method.
    pub spread_ci_lower: f64,

    /// Upper bound of 95% CI for optimal spread (bps).
    ///
    /// When uncertain, quote at `spread_ci_upper` for safety.
    pub spread_ci_upper: f64,

    /// Correlation between kappa and sigma.
    ///
    /// Negative correlation (typical) means spread widening during vol.
    /// Use for covariance-aware spread calculation.
    pub kappa_sigma_corr: f64,

    // === Skew-Adjusted CIs (Phase 2A: Statistical Refinements) ===

    /// Skew-adjusted lower bound of spread CI.
    ///
    /// Tighter than spread_ci_lower when sigma is left-skewed (vol drop likely).
    /// Adjustment: lower × (1 - σ_skew × skew_sensitivity) for positive skew.
    pub spread_ci_lower_skew_adjusted: f64,

    /// Skew-adjusted upper bound of spread CI.
    ///
    /// Wider than spread_ci_upper when sigma is right-skewed (vol spike likely).
    /// This is the DEFENSIVE bound - quote here when vol spike risk is high.
    /// Adjustment: upper × (1 + σ_skew × skew_sensitivity) for positive skew.
    pub spread_ci_upper_skew_adjusted: f64,
}

impl Default for KappaBeliefs {
    fn default() -> Self {
        Self {
            kappa_effective: 2000.0,
            kappa_raw: 2000.0,
            components: KappaComponents::default(),
            confidence: 0.0,
            is_warmup: true,
            n_own_fills: 0,
            // Uncertainty defaults
            kappa_std: 500.0, // High uncertainty initially
            spread_ci_lower: 3.0, // ~3 bps min
            spread_ci_upper: 10.0, // ~10 bps max
            kappa_sigma_corr: 0.0, // Unknown correlation
            // Phase 2A: Skew-adjusted CIs (default = same as base CI)
            spread_ci_lower_skew_adjusted: 3.0,
            spread_ci_upper_skew_adjusted: 10.0,
        }
    }
}

impl KappaBeliefs {
    /// Get spread uncertainty (half-width of CI in bps).
    pub fn spread_uncertainty(&self) -> f64 {
        (self.spread_ci_upper - self.spread_ci_lower) / 2.0
    }

    /// Get recommended spread adjustment factor for uncertainty.
    ///
    /// When uncertainty is high, this returns > 1.0 to widen spreads.
    /// Formula: 1.0 + uncertainty_scale × (spread_uncertainty / base_spread)
    pub fn uncertainty_adjustment(&self, base_spread_bps: f64, uncertainty_scale: f64) -> f64 {
        if base_spread_bps < 1e-6 {
            return 1.0;
        }
        let relative_uncertainty = self.spread_uncertainty() / base_spread_bps;
        1.0 + uncertainty_scale * relative_uncertainty.min(1.0)
    }

    /// Get coefficient of variation for kappa (CV = σ/μ).
    ///
    /// CV > 0.5 indicates high uncertainty, consider widening.
    pub fn kappa_cv(&self) -> f64 {
        if self.kappa_effective < 1e-6 {
            return 1.0;
        }
        self.kappa_std / self.kappa_effective
    }

    // === Skew-Adjusted CI Methods (Phase 2A) ===

    /// Get skew-adjusted spread uncertainty.
    ///
    /// This is the half-width of the asymmetric CI.
    pub fn skew_adjusted_spread_uncertainty(&self) -> f64 {
        (self.spread_ci_upper_skew_adjusted - self.spread_ci_lower_skew_adjusted) / 2.0
    }

    /// Get the defensive spread (upper skew-adjusted CI).
    ///
    /// Use this spread when vol spike risk is elevated.
    /// This is wider than the base CI upper bound.
    pub fn defensive_spread(&self) -> f64 {
        self.spread_ci_upper_skew_adjusted
    }

    /// Get the aggressive spread (lower skew-adjusted CI).
    ///
    /// Use this spread when vol is expected to drop.
    /// This is tighter than the base CI lower bound.
    pub fn aggressive_spread(&self) -> f64 {
        self.spread_ci_lower_skew_adjusted
    }

    /// Check if skew adjustment is significant.
    ///
    /// Returns true if skew-adjusted CI differs from base CI by > 10%.
    pub fn is_skew_significant(&self) -> bool {
        let upper_diff = (self.spread_ci_upper_skew_adjusted - self.spread_ci_upper).abs();
        let lower_diff = (self.spread_ci_lower_skew_adjusted - self.spread_ci_lower).abs();

        let base_width = self.spread_ci_upper - self.spread_ci_lower;
        if base_width < 0.1 {
            return false;
        }

        (upper_diff / base_width > 0.1) || (lower_diff / base_width > 0.1)
    }

    /// Get recommended spread given vol spike risk level.
    ///
    /// - `vol_risk = 0.0` → use aggressive (lower) spread
    /// - `vol_risk = 0.5` → use midpoint
    /// - `vol_risk = 1.0` → use defensive (upper) spread
    pub fn recommended_spread(&self, vol_risk: f64) -> f64 {
        let vol_risk = vol_risk.clamp(0.0, 1.0);
        self.spread_ci_lower_skew_adjusted
            + vol_risk * (self.spread_ci_upper_skew_adjusted - self.spread_ci_lower_skew_adjusted)
    }
}

/// Breakdown of kappa components.
#[derive(Debug, Clone, Default)]
pub struct KappaComponents {
    /// Own-fill kappa (Bayesian posterior)
    pub own: (f64, f64), // (kappa, weight)
    /// Book-structure kappa (exponential decay regression)
    pub book: (f64, f64),
    /// Robust kappa (Student-t market trades)
    pub robust: (f64, f64),
    /// Prior kappa (regularization)
    pub prior: (f64, f64),
}

/// Position continuation beliefs from Beta-Binomial + signal fusion.
#[derive(Debug, Clone)]
pub struct ContinuationBeliefs {
    /// Raw fill-based posterior mean P(cont)
    pub p_fill_raw: f64,

    /// Fused probability after signal integration
    pub p_fused: f64,

    /// Confidence in fused estimate [0, 1]
    pub confidence_fused: f64,

    /// Changepoint discount applied to fill posterior
    pub changepoint_discount: f64,

    /// Signal summary for diagnostics
    pub signal_summary: ContinuationSignals,
}

impl Default for ContinuationBeliefs {
    fn default() -> Self {
        Self {
            p_fill_raw: 0.5,
            p_fused: 0.5,
            confidence_fused: 0.0,
            changepoint_discount: 0.0,
            signal_summary: ContinuationSignals::default(),
        }
    }
}

/// Individual signal contributions to continuation probability.
#[derive(Debug, Clone, Default)]
pub struct ContinuationSignals {
    /// Momentum continuation probability
    pub p_momentum: f64,
    /// Trend-based probability
    pub p_trend: f64,
    /// Regime-based prior
    pub p_regime: f64,
    /// Trend confidence
    pub trend_confidence: f64,
}

/// Regime beliefs from 4-state HMM.
#[derive(Debug, Clone)]
pub struct RegimeBeliefs {
    /// [quiet, normal, bursty, cascade] probabilities
    pub probs: [f64; 4],

    /// Most likely regime
    pub current: Regime,

    /// Confidence in regime classification [0, 1]
    pub confidence: f64,

    /// Probability of transition in next period
    pub transition_prob: f64,
}

impl Default for RegimeBeliefs {
    fn default() -> Self {
        Self {
            probs: [0.2, 0.5, 0.2, 0.1], // Prior: mostly normal
            current: Regime::Normal,
            confidence: 0.0,
            transition_prob: 0.1,
        }
    }
}

impl RegimeBeliefs {
    /// Blend a parameter across regimes.
    ///
    /// # Arguments
    /// * `values` - [quiet, normal, bursty, cascade] parameter values
    pub fn blend(&self, values: [f64; 4]) -> f64 {
        self.probs
            .iter()
            .zip(values.iter())
            .map(|(p, v)| p * v)
            .sum()
    }
}

/// Changepoint beliefs from BOCD.
#[derive(Debug, Clone)]
pub struct ChangepointBeliefs {
    /// P(changepoint in last 1 observation)
    pub prob_1: f64,

    /// P(changepoint in last 5 observations)
    pub prob_5: f64,

    /// P(changepoint in last 10 observations)
    pub prob_10: f64,

    /// Most likely run length
    pub run_length: usize,

    /// Entropy of run length distribution
    pub entropy: f64,

    /// Detection result (None/Pending/Confirmed)
    pub result: ChangepointResult,

    /// Trust factor for learning module [0, 1]
    /// Lower when changepoint is likely (stale model)
    pub learning_trust: f64,

    /// Number of changepoint observations processed
    pub observation_count: usize,

    /// Whether BOCD has enough observations for reliable detection
    /// (typically requires ~10 observations to avoid false positives)
    pub is_warmed_up: bool,
}

impl Default for ChangepointBeliefs {
    fn default() -> Self {
        Self {
            prob_1: 0.0,
            prob_5: 0.1,
            prob_10: 0.2,
            run_length: 10,
            entropy: 2.0,
            result: ChangepointResult::None,
            learning_trust: 1.0,
            observation_count: 0,
            is_warmed_up: false,
        }
    }
}

/// Changepoint detection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChangepointResult {
    /// No changepoint detected
    #[default]
    None,
    /// Threshold exceeded, awaiting confirmation
    Pending(usize),
    /// Confirmed changepoint
    Confirmed,
}

impl ChangepointResult {
    /// Check if any change detected.
    pub fn is_detected(&self) -> bool {
        !matches!(self, ChangepointResult::None)
    }

    /// Check if confirmed.
    pub fn is_confirmed(&self) -> bool {
        matches!(self, ChangepointResult::Confirmed)
    }
}

/// Trading edge beliefs.
///
/// ## Toxicity Adjustment (Phase 3: Medium Alpha)
///
/// The `toxicity_adjusted_edge` field accounts for informed trader probability,
/// providing a more realistic edge estimate during toxic flow periods.
///
/// Formula: `toxicity_adjusted_edge = expected_edge × (1 - α × toxicity_score)`
///
/// where:
/// - `α` = toxicity penalty coefficient (typically 0.5-0.8)
/// - `toxicity_score` = combined VPIN + soft_jump toxicity [0, 1]
#[derive(Debug, Clone)]
pub struct EdgeBeliefs {
    /// Expected edge (bps) - raw estimate
    pub expected_edge: f64,

    /// Toxicity-adjusted edge (bps).
    ///
    /// This is the edge estimate after accounting for informed flow.
    /// Use this for actual quoting decisions.
    pub toxicity_adjusted_edge: f64,

    /// Combined toxicity score [0, 1].
    ///
    /// Fuses VPIN and soft_jump toxicity for robust detection.
    /// Higher values = more informed flow = lower effective edge.
    pub toxicity_score: f64,

    /// Uncertainty in edge estimate (bps)
    pub uncertainty: f64,

    /// Edge by volatility regime [low, normal, high]
    pub by_regime: [f64; 3],

    /// Probability of positive edge
    pub p_positive: f64,

    /// Probability of positive toxicity-adjusted edge
    pub p_positive_adjusted: f64,

    /// Adverse selection bias correction
    pub as_bias: f64,

    /// Epistemic uncertainty (model disagreement)
    pub epistemic_uncertainty: f64,

    /// Should we quote? Based on toxicity-adjusted edge.
    ///
    /// False when toxicity is too high or edge is negative.
    pub should_quote: bool,
}

impl Default for EdgeBeliefs {
    fn default() -> Self {
        Self {
            expected_edge: 0.0,
            toxicity_adjusted_edge: 0.0,
            toxicity_score: 0.0,
            uncertainty: 2.0, // 2 bps uncertainty
            by_regime: [1.0, 0.5, -0.5], // Positive in calm, negative in volatile
            p_positive: 0.5,
            p_positive_adjusted: 0.5,
            as_bias: 0.0,
            epistemic_uncertainty: 0.5,
            should_quote: true,
        }
    }
}

impl EdgeBeliefs {
    /// Get the recommended edge for quoting (toxicity-adjusted).
    pub fn effective_edge(&self) -> f64 {
        self.toxicity_adjusted_edge
    }

    /// Get toxicity penalty factor [0, 1].
    ///
    /// 0 = full edge retained, 1 = no edge (fully toxic)
    pub fn toxicity_penalty(&self) -> f64 {
        self.toxicity_score
    }

    /// Check if edge is profitable after toxicity adjustment.
    pub fn is_profitable(&self) -> bool {
        self.toxicity_adjusted_edge > 0.0
    }

    /// Get conservative edge (lower bound of CI).
    ///
    /// Use when uncertain and want to be defensive.
    pub fn conservative_edge(&self) -> f64 {
        self.toxicity_adjusted_edge - 1.96 * self.uncertainty
    }

    /// Compute expected PnL per trade (in bps).
    ///
    /// Accounts for toxicity and adverse selection.
    pub fn expected_pnl_bps(&self) -> f64 {
        self.toxicity_adjusted_edge - self.as_bias
    }
}

// =============================================================================
// Microstructure Beliefs (Phase 1: Alpha-Generating)
// =============================================================================

/// Microstructure-based toxicity and flow signals.
///
/// These signals provide real-time estimates of order flow toxicity
/// and market quality. They are leading indicators of adverse selection.
///
/// ## Key Signals
///
/// - **VPIN**: Volume-synchronized probability of informed trading
/// - **Depth OFI**: Depth-weighted order flow imbalance
/// - **Liquidity Evaporation**: Rapid depth drops near the touch
///
/// ## Usage
///
/// ```ignore
/// let beliefs = belief_state.snapshot();
///
/// // Use VPIN for toxicity assessment
/// if beliefs.microstructure.vpin > 0.7 {
///     // High informed flow - widen spreads
/// }
///
/// // Use liquidity evaporation for defense
/// if beliefs.microstructure.liquidity_evaporation > 0.5 {
///     // Book thinning - reduce position
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MicrostructureBeliefs {
    /// VPIN: Volume-synchronized probability of informed trading [0, 1].
    ///
    /// Higher values indicate more toxic (informed) flow.
    /// - < 0.3: Safe, noise-dominated market
    /// - 0.3-0.5: Normal
    /// - 0.5-0.7: Elevated toxicity
    /// - > 0.7: Dangerous, likely informed flow
    pub vpin: f64,

    /// VPIN velocity (rate of change).
    ///
    /// Positive = toxicity increasing (danger).
    /// Negative = toxicity decreasing (improving).
    pub vpin_velocity: f64,

    /// Depth-weighted order flow imbalance [-1, 1].
    ///
    /// Unlike simple OFI, this weights levels by distance from mid.
    /// Positive = buying pressure, Negative = selling pressure.
    pub depth_ofi: f64,

    /// Liquidity evaporation score [0, 1].
    ///
    /// Measures how much near-touch depth has dropped from recent peak.
    /// High values indicate market makers pulling quotes (danger signal).
    pub liquidity_evaporation: f64,

    /// Order flow direction from VPIN buckets [-1, 1].
    ///
    /// Signed measure of recent order flow direction.
    pub order_flow_direction: f64,

    /// Confidence in microstructure estimates [0, 1].
    ///
    /// Based on data quantity and quality.
    pub confidence: f64,

    /// Number of VPIN buckets completed.
    pub vpin_buckets: usize,

    /// Whether microstructure signals are valid (enough data).
    pub is_valid: bool,

    // === Phase 1A: Toxic Volume Refinements ===
    /// Trade size sigma: deviation of current median from baseline.
    ///
    /// > 3.0 indicates anomalous trade size regime (potential sweep).
    pub trade_size_sigma: f64,

    /// Toxicity acceleration factor [1.0, 2.0].
    ///
    /// Multiplier for VPIN when trade sizes are anomalous.
    /// 1.0 = normal, 2.0 = maximum acceleration.
    pub toxicity_acceleration: f64,

    /// Cumulative OFI with decay [-1, 1].
    ///
    /// Distinguishes temporary book flickers from sustained shifts.
    /// Positive = sustained bid pressure, Negative = sustained ask pressure.
    pub cofi: f64,

    /// COFI velocity (momentum of the imbalance).
    pub cofi_velocity: f64,

    /// Whether a sustained supply/demand shift is detected.
    pub is_sustained_shift: bool,
}

impl Default for MicrostructureBeliefs {
    fn default() -> Self {
        Self {
            vpin: 0.0,
            vpin_velocity: 0.0,
            depth_ofi: 0.0,
            liquidity_evaporation: 0.0,
            order_flow_direction: 0.0,
            confidence: 0.0,
            vpin_buckets: 0,
            is_valid: false,
            // Phase 1A fields
            trade_size_sigma: 0.0,
            toxicity_acceleration: 1.0, // No acceleration by default
            cofi: 0.0,
            cofi_velocity: 0.0,
            is_sustained_shift: false,
        }
    }
}

impl MicrostructureBeliefs {
    /// Check if market is in toxic state.
    ///
    /// Combines multiple signals for robust toxicity detection.
    /// Enhanced with Phase 1A refinements: trade size anomaly and COFI.
    pub fn is_toxic(&self) -> bool {
        if !self.is_valid {
            return false; // Can't assess without data
        }

        // High VPIN or rapidly increasing VPIN (with toxicity acceleration)
        let effective_vpin = self.vpin * self.toxicity_acceleration;
        let vpin_toxic = effective_vpin > 0.6 || (effective_vpin > 0.4 && self.vpin_velocity > 0.1);

        // Significant liquidity evaporation
        let evaporation_toxic = self.liquidity_evaporation > 0.5;

        // Trade size anomaly (large orders + high VPIN = danger)
        let size_anomaly_toxic = self.trade_size_sigma > 3.0 && self.vpin > 0.4;

        // Sustained directional shift with high VPIN
        let shift_toxic = self.is_sustained_shift && self.vpin > 0.5;

        vpin_toxic || evaporation_toxic || size_anomaly_toxic || shift_toxic
    }

    /// Get combined toxicity score [0, 1].
    ///
    /// Weighted combination of all toxicity signals.
    /// Enhanced with Phase 1A: trade size anomaly and COFI.
    pub fn toxicity_score(&self) -> f64 {
        if !self.is_valid {
            return 0.5; // Uncertain default
        }

        // Apply toxicity acceleration from trade size anomaly
        let accelerated_vpin = (self.vpin * self.toxicity_acceleration).min(1.0);

        // Base weights: VPIN (0.4), velocity (0.15), evaporation (0.2)
        // New weights: trade_size (0.1), COFI magnitude (0.15)
        let base = 0.4 * accelerated_vpin
            + 0.15 * (self.vpin_velocity.max(0.0) * 2.0).min(1.0) // Scale velocity to [0, 1]
            + 0.2 * self.liquidity_evaporation
            + 0.1 * (self.trade_size_sigma / 5.0).min(1.0) // Normalize sigma to [0, 1]
            + 0.15 * self.cofi.abs(); // COFI magnitude

        base.clamp(0.0, 1.0)
    }

    /// Get spread multiplier based on microstructure [1.0, 3.0].
    ///
    /// Use to widen spreads when toxicity is elevated.
    pub fn spread_multiplier(&self) -> f64 {
        let toxicity = self.toxicity_score();
        // Linear scaling: toxicity 0 -> 1.0x, toxicity 1 -> 3.0x
        1.0 + 2.0 * toxicity
    }

    /// Get effective VPIN with toxicity acceleration.
    pub fn effective_vpin(&self) -> f64 {
        (self.vpin * self.toxicity_acceleration).min(1.0)
    }
}

// =============================================================================
// Cross-Venue Beliefs (Bivariate Flow Model)
// =============================================================================

/// Cross-venue beliefs from joint Binance + Hyperliquid flow analysis.
///
/// The key insight is that neither exchange is definitively the leader -
/// the signal comes from the **joint relationship**:
/// - Agreement (both show same pressure) → high confidence
/// - Divergence → uncertainty, market dislocation
/// - Intensity ratio → where is price discovery happening?
///
/// ## Signal Interpretation
///
/// | State | Interpretation | Action |
/// |-------|----------------|--------|
/// | Both buying, high agreement | Strong bullish | Lean long, aggressive bids |
/// | Both selling, high agreement | Strong bearish | Lean short, aggressive asks |
/// | Binance buying, HL selling | Dislocation | Widen spreads, reduce size |
/// | High intensity on Binance | Price discovery there | Weight Binance signal more |
/// | Both VPIN > 0.6 | Informed traders active | Widen significantly |
///
/// ## Usage
///
/// ```ignore
/// let beliefs = belief_state.snapshot();
/// let cv = &beliefs.cross_venue;
///
/// // Use agreement for confidence boost
/// if cv.agreement > 0.7 {
///     // High confidence in direction
///     let direction_adjustment = cv.direction * cv.confidence * max_bias_bps;
/// }
///
/// // Widen when venues disagree
/// let uncertainty_mult = 1.0 + (1.0 - cv.confidence) * 0.5;
///
/// // Widen when either venue shows toxicity
/// let toxicity_mult = 1.0 + cv.max_toxicity * toxicity_sensitivity;
/// ```
#[derive(Debug, Clone)]
pub struct CrossVenueBeliefs {
    /// Joint direction belief from bivariate analysis [-1, +1].
    ///
    /// Positive = bullish (both venues buying), Negative = bearish.
    /// When venues agree, this reflects the consensus direction.
    /// When venues disagree, this decays toward neutral.
    pub direction: f64,

    /// Confidence in direction based on venue agreement [0, 1].
    ///
    /// High when both venues show same direction (agreement > 0.5).
    /// Low when venues diverge (uncertainty, market dislocation).
    pub confidence: f64,

    /// Where is price discovery happening? [0, 1].
    ///
    /// 0 = all activity on Hyperliquid
    /// 0.5 = balanced activity
    /// 1 = all activity on Binance
    ///
    /// When discovery_venue > 0.6, weight Binance signals more heavily.
    pub discovery_venue: f64,

    /// Maximum toxicity across both venues [0, 1].
    ///
    /// max(vpin_binance, vpin_hl).
    /// Use for spread widening - when either venue is toxic, widen.
    pub max_toxicity: f64,

    /// Average toxicity across venues [0, 1].
    ///
    /// (vpin_binance + vpin_hl) / 2.
    /// Use for baseline toxicity assessment.
    pub avg_toxicity: f64,

    /// Agreement score between venues [-1, 1].
    ///
    /// +1 = perfect agreement (same direction, same magnitude)
    /// 0 = uncorrelated
    /// -1 = perfect disagreement (opposite directions)
    pub agreement: f64,

    /// Imbalance divergence (binance - hl) [-2, 2].
    ///
    /// Measures the difference in directional pressure between venues.
    /// Large positive = Binance more bullish than HL.
    /// Large negative = HL more bullish than Binance.
    pub divergence: f64,

    /// Intensity ratio: λ_B / (λ_B + λ_H) [0, 1].
    ///
    /// Where is the trading activity concentrated?
    /// \> 0.6 = Binance is the action (follow Binance signals)
    /// \< 0.4 = HL is the action (rely on local flow)
    pub intensity_ratio: f64,

    /// Rolling correlation of imbalances [-1, 1].
    ///
    /// Measures how correlated the imbalance signals are over time.
    /// High correlation = stable relationship, use cross-venue signals.
    /// Low correlation = regime change, rely on venue-specific signals.
    pub imbalance_correlation: f64,

    /// Whether a toxicity alert is active (either venue VPIN > 0.7).
    pub toxicity_alert: bool,

    /// Whether a divergence alert is active (large venue disagreement).
    pub divergence_alert: bool,

    /// Whether cross-venue beliefs are valid (have data from both venues).
    pub is_valid: bool,

    /// Number of cross-venue observations processed.
    pub observation_count: u64,

    /// Timestamp of last cross-venue update (epoch ms).
    pub last_update_ms: u64,
}

impl Default for CrossVenueBeliefs {
    fn default() -> Self {
        Self {
            direction: 0.0,
            confidence: 0.0,
            discovery_venue: 0.5,
            max_toxicity: 0.0,
            avg_toxicity: 0.0,
            agreement: 0.0,
            divergence: 0.0,
            intensity_ratio: 0.5,
            imbalance_correlation: 0.0,
            toxicity_alert: false,
            divergence_alert: false,
            is_valid: false,
            observation_count: 0,
            last_update_ms: 0,
        }
    }
}

impl CrossVenueBeliefs {
    /// Get spread multiplier based on cross-venue state [1.0, 2.5].
    ///
    /// Widens spreads when:
    /// - Venues disagree (low confidence)
    /// - Either venue shows toxicity
    /// - Correlation is low (unstable relationship)
    pub fn spread_multiplier(&self) -> f64 {
        if !self.is_valid {
            return 1.0; // No adjustment without data
        }

        // Base multiplier from uncertainty (low confidence = wider spreads)
        let uncertainty_mult = 1.0 + (1.0 - self.confidence) * 0.3;

        // Toxicity multiplier
        let toxicity_mult = 1.0 + self.max_toxicity * 0.5;

        // Correlation multiplier (low correlation = unstable, widen)
        let corr_mult = 1.0 + (1.0 - self.imbalance_correlation.abs()) * 0.2;

        (uncertainty_mult * toxicity_mult * corr_mult).min(2.5)
    }

    /// Get skew recommendation based on cross-venue direction [-1, 1].
    ///
    /// Returns a directional skew multiplier:
    /// - Positive: lean long (tighter bids, wider asks)
    /// - Negative: lean short (wider bids, tighter asks)
    ///
    /// The magnitude is scaled by confidence and reduced by toxicity.
    pub fn skew_recommendation(&self) -> f64 {
        if !self.is_valid {
            return 0.0;
        }

        // Direction scaled by confidence, reduced by toxicity
        let toxicity_discount = 1.0 - self.max_toxicity;
        self.direction * self.confidence * toxicity_discount
    }

    /// Check if cross-venue signals suggest caution.
    ///
    /// True when any of:
    /// - Toxicity alert is active
    /// - Divergence alert is active
    /// - Confidence is very low (< 0.3)
    pub fn requires_caution(&self) -> bool {
        self.toxicity_alert || self.divergence_alert || self.confidence < 0.3
    }

    /// Get the dominant venue for signal weighting.
    ///
    /// Returns:
    /// - Some(true) if Binance is dominant (discovery_venue > 0.6)
    /// - Some(false) if HL is dominant (discovery_venue < 0.4)
    /// - None if balanced
    pub fn dominant_venue(&self) -> Option<bool> {
        if self.discovery_venue > 0.6 {
            Some(true) // Binance dominant
        } else if self.discovery_venue < 0.4 {
            Some(false) // HL dominant
        } else {
            None // Balanced
        }
    }

    /// Get signal quality score [0, 1].
    ///
    /// High when:
    /// - Venues agree (high confidence)
    /// - Correlation is strong
    /// - Toxicity is low
    pub fn signal_quality(&self) -> f64 {
        if !self.is_valid {
            return 0.0;
        }

        let agreement_score = (self.agreement + 1.0) / 2.0; // Map [-1,1] to [0,1]
        let corr_score = self.imbalance_correlation.abs();
        let toxicity_score = 1.0 - self.max_toxicity;

        (0.4 * agreement_score + 0.3 * corr_score + 0.3 * toxicity_score).clamp(0.0, 1.0)
    }
}

// =============================================================================
// Calibration State
// =============================================================================

/// Model calibration state for tracking prediction quality.
#[derive(Debug, Clone, Default)]
pub struct CalibrationState {
    /// Fill probability calibration
    pub fill: CalibrationMetrics,

    /// Adverse selection calibration
    pub adverse_selection: CalibrationMetrics,

    /// Signal quality (mutual information scores)
    pub signal_quality: std::collections::HashMap<String, f64>,

    /// Pending predictions awaiting outcomes
    pub pending_count: usize,

    /// Linked (resolved) predictions
    pub linked_count: usize,

    /// Latency-adjusted calibration metrics (Phase 7.1)
    pub latency: LatencyCalibration,
}

/// Latency-adjusted calibration metrics.
///
/// Tracks signal value decay and computes IR only for "fresh" predictions.
/// This helps identify when our processing latency exceeds the signal's alpha duration.
#[derive(Debug, Clone)]
pub struct LatencyCalibration {
    /// VPIN latency-adjusted IR (computed only for fresh signals)
    pub vpin_latency_adjusted_ir: f64,

    /// Time until VPIN signal value drops below 0.5 (ms)
    pub vpin_alpha_duration_ms: f64,

    /// Flow signal latency-adjusted IR
    pub flow_latency_adjusted_ir: f64,

    /// Time until flow signal drops below 0.5 (ms)
    pub flow_alpha_duration_ms: f64,

    /// Maximum processing time to capture alpha (ms)
    /// This is the minimum of all signal alpha durations.
    pub signal_latency_budget_ms: f64,

    /// True if our processing latency exceeds the signal's alpha duration.
    /// When true, we're providing "free options" to faster participants.
    pub is_latency_constrained: bool,

    /// Mean latency from signal to quote (ms)
    pub mean_signal_to_quote_ms: f64,

    /// P95 latency from signal to quote (ms)
    pub p95_signal_to_quote_ms: f64,

    /// Ratio of fresh_ir to all_ir (>1 = freshness matters)
    pub ir_degradation_ratio: f64,
}

impl Default for LatencyCalibration {
    fn default() -> Self {
        Self {
            vpin_latency_adjusted_ir: 0.0,
            vpin_alpha_duration_ms: 50.0, // Conservative default
            flow_latency_adjusted_ir: 0.0,
            flow_alpha_duration_ms: 100.0, // Conservative default
            signal_latency_budget_ms: 50.0,
            is_latency_constrained: false,
            mean_signal_to_quote_ms: 0.0,
            p95_signal_to_quote_ms: 0.0,
            ir_degradation_ratio: 1.0,
        }
    }
}

impl LatencyCalibration {
    /// Check if a specific signal is latency-constrained.
    pub fn is_signal_constrained(&self, signal_name: &str, processing_latency_ms: f64) -> bool {
        match signal_name {
            "vpin" => processing_latency_ms > self.vpin_alpha_duration_ms,
            "flow" | "ofi" | "cofi" => processing_latency_ms > self.flow_alpha_duration_ms,
            _ => processing_latency_ms > self.signal_latency_budget_ms,
        }
    }

    /// Get alpha duration for a signal.
    pub fn alpha_duration(&self, signal_name: &str) -> f64 {
        match signal_name {
            "vpin" => self.vpin_alpha_duration_ms,
            "flow" | "ofi" | "cofi" => self.flow_alpha_duration_ms,
            _ => self.signal_latency_budget_ms,
        }
    }

    /// Check if freshness matters (ir_degradation_ratio > 1.2)
    pub fn freshness_matters(&self) -> bool {
        self.ir_degradation_ratio > 1.2
    }
}

/// Metrics for calibration quality assessment.
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Brier score (lower is better, 0 = perfect)
    pub brier_score: f64,

    /// Information ratio (higher is better, >1 = adds value)
    pub information_ratio: f64,

    /// Base rate (empirical positive rate)
    pub base_rate: f64,

    /// Number of resolved predictions
    pub n_samples: usize,

    /// Is the model calibrated? (IR > 1.0 with enough samples)
    pub is_calibrated: bool,
}

impl Default for CalibrationMetrics {
    fn default() -> Self {
        Self {
            brier_score: 0.25, // Random baseline for binary
            information_ratio: 0.0,
            base_rate: 0.5,
            n_samples: 0,
            is_calibrated: false,
        }
    }
}

impl CalibrationMetrics {
    /// Check if the model adds value (IR > 1.0 with confidence).
    pub fn adds_value(&self) -> bool {
        self.is_calibrated && self.information_ratio > 1.0
    }

    /// Get a quality score [0, 1].
    pub fn quality_score(&self) -> f64 {
        if self.n_samples < 30 {
            return 0.0; // Not enough data
        }
        // Score based on IR, saturating at 2.0
        (self.information_ratio / 2.0).clamp(0.0, 1.0)
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics about the belief state.
#[derive(Debug, Clone)]
pub struct BeliefStats {
    /// Total number of price observations
    pub n_price_obs: u64,

    /// Total number of fills
    pub n_fills: u64,

    /// Total number of market trades observed
    pub n_market_trades: u64,

    /// Last update timestamp (epoch ms)
    pub last_update_ms: u64,

    /// Time since initialization (seconds)
    pub uptime_secs: f64,

    /// Is the belief system warmed up?
    pub is_warmed_up: bool,

    /// Warmup progress [0, 1]
    pub warmup_progress: f64,
}

impl Default for BeliefStats {
    fn default() -> Self {
        Self {
            n_price_obs: 0,
            n_fills: 0,
            n_market_trades: 0,
            last_update_ms: 0,
            uptime_secs: 0.0,
            is_warmed_up: false,
            warmup_progress: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_snapshot_default() {
        let snapshot = BeliefSnapshot::default();
        assert!(!snapshot.is_warmed_up());
        assert_eq!(snapshot.current_regime(), Regime::Normal);
        assert!((snapshot.predictive_bias() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_belief_snapshot_overall_confidence() {
        let mut snapshot = BeliefSnapshot::default();
        snapshot.drift_vol.confidence = 0.8;
        snapshot.kappa.confidence = 0.9;
        snapshot.regime.confidence = 0.7;

        let conf = snapshot.overall_confidence();
        // Geometric mean of 0.8, 0.9, 0.7
        let expected = (0.8 * 0.9 * 0.7_f64).powf(1.0 / 3.0);
        assert!((conf - expected).abs() < 1e-6);
    }

    #[test]
    fn test_regime_beliefs_blend() {
        let regime = RegimeBeliefs {
            probs: [0.5, 0.3, 0.1, 0.1],
            ..Default::default()
        };

        // Blend gamma: quiet=1.0, normal=2.0, bursty=3.0, cascade=4.0
        let blended = regime.blend([1.0, 2.0, 3.0, 4.0]);
        // 0.5*1 + 0.3*2 + 0.1*3 + 0.1*4 = 0.5 + 0.6 + 0.3 + 0.4 = 1.8
        assert!((blended - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_changepoint_result_detection() {
        assert!(!ChangepointResult::None.is_detected());
        assert!(!ChangepointResult::None.is_confirmed());

        assert!(ChangepointResult::Pending(2).is_detected());
        assert!(!ChangepointResult::Pending(2).is_confirmed());

        assert!(ChangepointResult::Confirmed.is_detected());
        assert!(ChangepointResult::Confirmed.is_confirmed());
    }

    #[test]
    fn test_calibration_metrics_adds_value() {
        let mut metrics = CalibrationMetrics::default();
        assert!(!metrics.adds_value());

        metrics.is_calibrated = true;
        metrics.information_ratio = 0.5;
        assert!(!metrics.adds_value());

        metrics.information_ratio = 1.5;
        assert!(metrics.adds_value());
    }

    #[test]
    fn test_calibration_metrics_quality_score() {
        let mut metrics = CalibrationMetrics::default();

        // Not enough samples
        metrics.n_samples = 10;
        assert!((metrics.quality_score() - 0.0).abs() < 1e-10);

        // Enough samples, IR = 0
        metrics.n_samples = 100;
        metrics.information_ratio = 0.0;
        assert!((metrics.quality_score() - 0.0).abs() < 1e-10);

        // IR = 1.0 → quality = 0.5
        metrics.information_ratio = 1.0;
        assert!((metrics.quality_score() - 0.5).abs() < 1e-10);

        // IR = 2.0 → quality = 1.0
        metrics.information_ratio = 2.0;
        assert!((metrics.quality_score() - 1.0).abs() < 1e-10);

        // IR > 2.0 → quality capped at 1.0
        metrics.information_ratio = 3.0;
        assert!((metrics.quality_score() - 1.0).abs() < 1e-10);
    }

    // === Phase 2A: Skewness Tests ===

    #[test]
    fn test_drift_vol_skewness_defaults() {
        let dv = DriftVolatilityBeliefs::default();

        // Default: slight positive skewness (typical for vol)
        assert!((dv.sigma_skewness - 0.5).abs() < 1e-10);
        assert!((dv.sigma_kurtosis - 0.0).abs() < 1e-10);
        assert!((dv.drift_skewness - 0.0).abs() < 1e-10);

        // Not elevated at defaults
        assert!(!dv.has_vol_spike_risk());
        assert!(!dv.has_fat_tails());
    }

    #[test]
    fn test_drift_vol_vol_spike_risk() {
        let mut dv = DriftVolatilityBeliefs::default();

        // Low skewness: no spike risk
        dv.sigma_skewness = 0.5;
        assert!(!dv.has_vol_spike_risk());

        // High skewness: spike risk
        dv.sigma_skewness = 1.5;
        assert!(dv.has_vol_spike_risk());
    }

    #[test]
    fn test_drift_vol_fat_tails() {
        let mut dv = DriftVolatilityBeliefs::default();

        // Low kurtosis: no fat tails
        dv.sigma_kurtosis = 2.0;
        assert!(!dv.has_fat_tails());

        // High kurtosis: fat tails
        dv.sigma_kurtosis = 5.0;
        assert!(dv.has_fat_tails());
    }

    #[test]
    fn test_drift_vol_spread_factors() {
        let mut dv = DriftVolatilityBeliefs::default();
        let sensitivity = 1.0;

        // Neutral skewness: factors near 1.0
        dv.sigma_skewness = 0.0;
        let bid = dv.bid_spread_factor(sensitivity);
        let ask = dv.ask_spread_factor(sensitivity);
        assert!((bid - 1.0).abs() < 0.01);
        assert!((ask - 1.0).abs() < 0.01);

        // High positive skewness: tighter bid, wider ask
        dv.sigma_skewness = 2.0;
        let bid_high = dv.bid_spread_factor(sensitivity);
        let ask_high = dv.ask_spread_factor(sensitivity);
        assert!(bid_high < 1.0); // Tighter
        assert!(ask_high > 1.0); // Wider
    }

    #[test]
    fn test_drift_vol_tail_risk_score() {
        let mut dv = DriftVolatilityBeliefs::default();

        // Low skewness and kurtosis: low tail risk
        dv.sigma_skewness = 0.0;
        dv.sigma_kurtosis = 0.0;
        assert!(dv.tail_risk_score() < 0.1);

        // High skewness and kurtosis: high tail risk
        dv.sigma_skewness = 2.0;
        dv.sigma_kurtosis = 6.0;
        assert!(dv.tail_risk_score() > 0.8);
    }

    #[test]
    fn test_kappa_skew_adjusted_defaults() {
        let kappa = KappaBeliefs::default();

        // Default: skew-adjusted CIs equal base CIs
        assert!((kappa.spread_ci_lower_skew_adjusted - kappa.spread_ci_lower).abs() < 1e-10);
        assert!((kappa.spread_ci_upper_skew_adjusted - kappa.spread_ci_upper).abs() < 1e-10);
    }

    #[test]
    fn test_kappa_defensive_aggressive_spread() {
        let mut kappa = KappaBeliefs::default();
        kappa.spread_ci_lower_skew_adjusted = 3.0;
        kappa.spread_ci_upper_skew_adjusted = 12.0;

        assert!((kappa.aggressive_spread() - 3.0).abs() < 1e-10);
        assert!((kappa.defensive_spread() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_kappa_recommended_spread() {
        let mut kappa = KappaBeliefs::default();
        kappa.spread_ci_lower_skew_adjusted = 4.0;
        kappa.spread_ci_upper_skew_adjusted = 10.0;

        // vol_risk = 0 → lower bound
        assert!((kappa.recommended_spread(0.0) - 4.0).abs() < 1e-10);

        // vol_risk = 1 → upper bound
        assert!((kappa.recommended_spread(1.0) - 10.0).abs() < 1e-10);

        // vol_risk = 0.5 → midpoint
        assert!((kappa.recommended_spread(0.5) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_kappa_skew_significant() {
        let mut kappa = KappaBeliefs::default();
        kappa.spread_ci_lower = 3.0;
        kappa.spread_ci_upper = 10.0;

        // Same as base: not significant
        kappa.spread_ci_lower_skew_adjusted = 3.0;
        kappa.spread_ci_upper_skew_adjusted = 10.0;
        assert!(!kappa.is_skew_significant());

        // Large difference: significant
        kappa.spread_ci_upper_skew_adjusted = 12.0; // 2 bps diff on 7 bps width = 28%
        assert!(kappa.is_skew_significant());
    }
}
