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
#[derive(Debug, Clone)]
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

    // === Calibration ===
    /// Model calibration metrics
    pub calibration: CalibrationState,

    // === Meta ===
    /// Statistics about the belief state
    pub stats: BeliefStats,
}

impl Default for BeliefSnapshot {
    fn default() -> Self {
        Self {
            drift_vol: DriftVolatilityBeliefs::default(),
            kappa: KappaBeliefs::default(),
            continuation: ContinuationBeliefs::default(),
            regime: RegimeBeliefs::default(),
            changepoint: ChangepointBeliefs::default(),
            edge: EdgeBeliefs::default(),
            calibration: CalibrationState::default(),
            stats: BeliefStats::default(),
        }
    }
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
        }
    }
}

/// Fill intensity (kappa) beliefs from confidence-weighted blending.
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
        }
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
#[derive(Debug, Clone)]
pub struct EdgeBeliefs {
    /// Expected edge (bps)
    pub expected_edge: f64,

    /// Uncertainty in edge estimate (bps)
    pub uncertainty: f64,

    /// Edge by volatility regime [low, normal, high]
    pub by_regime: [f64; 3],

    /// Probability of positive edge
    pub p_positive: f64,

    /// Adverse selection bias correction
    pub as_bias: f64,

    /// Epistemic uncertainty (model disagreement)
    pub epistemic_uncertainty: f64,
}

impl Default for EdgeBeliefs {
    fn default() -> Self {
        Self {
            expected_edge: 0.0,
            uncertainty: 2.0, // 2 bps uncertainty
            by_regime: [1.0, 0.5, -0.5], // Positive in calm, negative in volatile
            p_positive: 0.5,
            as_bias: 0.0,
            epistemic_uncertainty: 0.5,
        }
    }
}

// =============================================================================
// Calibration State
// =============================================================================

/// Model calibration state for tracking prediction quality.
#[derive(Debug, Clone)]
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
}

impl Default for CalibrationState {
    fn default() -> Self {
        Self {
            fill: CalibrationMetrics::default(),
            adverse_selection: CalibrationMetrics::default(),
            signal_quality: std::collections::HashMap::new(),
            pending_count: 0,
            linked_count: 0,
        }
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
}
