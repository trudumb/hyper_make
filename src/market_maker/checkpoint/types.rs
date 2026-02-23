//! Checkpoint data types for persisting learned state across sessions.
//!
//! Each checkpoint struct captures the minimum learning state needed to warm-start
//! a component. Ephemeral state (cached values, timestamps, rolling windows) is NOT
//! persisted — it rebuilds from live data within seconds.

use serde::{Deserialize, Deserializer, Serialize};

use crate::market_maker::calibration::parameter_learner::LearnedParameters;

/// Deserialize an f64 field that may be `null` in JSON.
///
/// When a checkpoint is hand-edited or corrupted, f64 fields can become `null`.
/// Without this, serde fails with "invalid type: null, expected f64".
/// This deserializer treats `null` as `0.0` (the f64 default).
fn deserialize_f64_or_null<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<f64>::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}

/// Deserialize a u64 field that may be `null` in JSON.
fn deserialize_u64_or_null<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<u64>::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}

/// Deserialize a usize field that may be `null` in JSON.
fn deserialize_usize_or_null<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<usize>::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}
use crate::market_maker::estimator::calibration_coordinator::CalibrationCoordinator;
use crate::market_maker::learning::spread_bandit::SpreadBanditCheckpoint;
use crate::market_maker::ComponentParams;

/// Type alias: CalibrationCoordinator is directly serializable and acts as its own checkpoint.
pub type CalibrationCoordinatorCheckpoint = CalibrationCoordinator;

/// Readiness verdict for a checkpoint — can it safely drive live trading?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PriorVerdict {
    /// All estimators converged, safe for live at full confidence.
    Ready,
    /// Partial convergence — live with defensive spreads (wider, smaller size).
    Marginal,
    /// Insufficient data — live would be reckless.
    #[default]
    Insufficient,
}

/// Calibration readiness snapshot, stamped into each saved checkpoint.
/// Captures estimator convergence at save time so the `run` command
/// can make a go/no-go decision without re-deriving from raw state.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PriorReadiness {
    pub verdict: PriorVerdict,
    /// Observation counts per estimator at assessment time
    pub vol_observations: usize,
    pub kappa_observations: usize,
    pub as_learning_samples: usize,
    pub regime_observations: usize,
    pub fill_rate_observations: usize,
    pub kelly_fills: usize,
    /// Session duration (seconds) at assessment time
    pub session_duration_s: f64,
    /// How many of the 5 core estimators met min_observations
    pub estimators_ready: u8,
}

/// Quote outcome tracker checkpoint for fill rate persistence.
///
/// Persists the BinnedFillRate empirical P(fill|spread) so the
/// tracker doesn't restart from zero on restart.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuoteOutcomeCheckpoint {
    /// Binned fill rate data: Vec of (lo_bps, hi_bps, fills, total) per bin
    #[serde(default)]
    pub bins: Vec<(f64, f64, u64, u64)>,
}

/// Complete checkpoint bundle containing all model state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointBundle {
    #[serde(default)]
    pub metadata: CheckpointMetadata,
    /// LearnedParameters — already Serialize/Deserialize, ~20 Bayesian posteriors
    #[serde(default)]
    pub learned_params: LearnedParameters,
    /// Pre-fill adverse selection classifier learning state
    #[serde(default)]
    pub pre_fill: PreFillCheckpoint,
    /// Enhanced AS classifier learning state
    #[serde(default)]
    pub enhanced: EnhancedCheckpoint,
    /// Volatility filter sufficient statistics
    #[serde(default)]
    pub vol_filter: VolFilterCheckpoint,
    /// Regime HMM belief state + transition counts
    #[serde(default)]
    pub regime_hmm: RegimeHMMCheckpoint,
    /// Informed flow mixture model parameters
    #[serde(default)]
    pub informed_flow: InformedFlowCheckpoint,
    /// Fill rate model Bayesian regression posteriors
    #[serde(default)]
    pub fill_rate: FillRateCheckpoint,
    /// BayesianKappaEstimator — own fills
    #[serde(default)]
    pub kappa_own: KappaCheckpoint,
    /// BayesianKappaEstimator — bid fills
    #[serde(default)]
    pub kappa_bid: KappaCheckpoint,
    /// BayesianKappaEstimator — ask fills
    #[serde(default)]
    pub kappa_ask: KappaCheckpoint,
    /// Momentum model — continuation probabilities by magnitude
    #[serde(default)]
    pub momentum: MomentumCheckpoint,
    /// Kelly win/loss tracker state for position sizing persistence
    #[serde(default)]
    pub kelly_tracker: KellyTrackerCheckpoint,
    /// Model ensemble weights for prediction persistence
    #[serde(default)]
    pub ensemble_weights: EnsembleWeightsCheckpoint,
    /// Contextual bandit spread optimizer state (replaces RL MDP)
    #[serde(default)]
    pub spread_bandit: SpreadBanditCheckpoint,
    /// Baseline tracker EWMA for counterfactual reward centering
    #[serde(default)]
    pub baseline_tracker: BaselineTrackerCheckpoint,
    /// Quote outcome tracker fill rate bins
    #[serde(default)]
    pub quote_outcomes: QuoteOutcomeCheckpoint,
    /// Kill switch state for persistence across restarts
    #[serde(default)]
    pub kill_switch: KillSwitchCheckpoint,
    /// Calibration readiness assessment, stamped at save time
    #[serde(default)]
    pub readiness: PriorReadiness,
    /// Calibration coordinator state for L2-derived kappa blending
    #[serde(default)]
    pub calibration_coordinator: CalibrationCoordinatorCheckpoint,
    /// Kappa orchestrator warmup state (cumulative fills, graduation flag)
    #[serde(default)]
    pub kappa_orchestrator: KappaOrchestratorCheckpoint,
    /// Prior confidence [0,1] from injection — how much to trust this prior.
    /// 0.0 = cold-start, 1.0 = fully calibrated and fresh.
    #[serde(default)]
    pub prior_confidence: f64,
    /// Bayesian fair value model learned parameters (α_book, β_flow, σ_noise).
    /// Posterior state (μ, σ²) is NOT persisted — it resets relative to current mid.
    #[serde(default)]
    pub bayesian_fair_value: crate::market_maker::belief::BayesianFairValueCheckpoint,
}

/// Checkpoint metadata for versioning and diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Schema version for forward compatibility
    #[serde(default)]
    pub version: u32,
    /// Timestamp when checkpoint was saved (ms since epoch)
    #[serde(default, deserialize_with = "deserialize_u64_or_null")]
    pub timestamp_ms: u64,
    /// Asset this checkpoint is for (e.g., "ETH" or "hyna:HYPE")
    #[serde(default)]
    pub asset: String,
    /// How long the session ran before this checkpoint (seconds)
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub session_duration_s: f64,
    /// Canonical base symbol without DEX prefix (e.g., "HYPE" for both "HYPE" and "hyna:HYPE")
    #[serde(default)]
    pub base_symbol: String,
    /// Source mode: "paper", "live", or "cold"
    #[serde(default)]
    pub source_mode: String,
    /// Timestamp of parent prior used at injection time (ms since epoch), 0 if cold-start
    #[serde(default)]
    pub parent_timestamp_ms: u64,
    /// Cumulative session count in the prior chain (0 = first session)
    #[serde(default)]
    pub chain_depth: u32,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            version: 0,
            timestamp_ms: 0,
            asset: String::new(),
            session_duration_s: 0.0,
            base_symbol: String::new(),
            source_mode: String::new(),
            parent_timestamp_ms: 0,
            chain_depth: 0,
        }
    }
}

/// PreFillASClassifier learning state.
///
/// Captures online learning weights and sufficient statistics.
/// Ephemeral state (cached_toxicity, signal values, timestamps) rebuilds from live data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFillCheckpoint {
    /// Learned signal weights [imbalance, flow, regime, funding, changepoint]
    #[serde(default)]
    pub learned_weights: [f64; 5],
    /// Running sum of (signal_i * outcome) for each signal
    #[serde(default)]
    pub signal_outcome_sum: [f64; 5],
    /// Running sum of signal_i^2 for each signal
    #[serde(default)]
    pub signal_sq_sum: [f64; 5],
    /// Number of learning samples processed
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub learning_samples: usize,
    /// Regime probabilities for soft blending
    #[serde(default)]
    pub regime_probs: [f64; 4],

    // === EWMA normalizer state (added for z-score normalization fix) ===
    /// EWMA mean/variance for imbalance (log bid/ask ratio)
    #[serde(default)]
    pub imbalance_ewma_mean: f64,
    #[serde(default = "default_ewma_var")]
    pub imbalance_ewma_var: f64,
    /// EWMA mean/variance for trade flow direction
    #[serde(default)]
    pub flow_ewma_mean: f64,
    #[serde(default = "default_ewma_var")]
    pub flow_ewma_var: f64,
    /// EWMA mean/variance for funding rate
    #[serde(default)]
    pub funding_ewma_mean: f64,
    #[serde(default = "default_ewma_var")]
    pub funding_ewma_var: f64,
    /// Previous regime trust (for delta computation)
    #[serde(default = "default_regime_trust")]
    pub regime_trust_prev: f64,
    /// EWMA mean for regime instability (1 - trust)
    #[serde(default)]
    pub regime_ewma_mean: f64,
    /// EWMA variance for regime instability
    #[serde(default = "default_ewma_var")]
    pub regime_ewma_var: f64,
    /// EWMA mean for changepoint probability
    #[serde(default)]
    pub changepoint_ewma_mean: f64,
    /// EWMA variance for changepoint probability
    #[serde(default = "default_ewma_var")]
    pub changepoint_ewma_var: f64,
    /// Number of normalizer observations
    #[serde(default)]
    pub normalizer_obs_count: usize,
    /// AS bias correction EWMA in bps (predicted - realized)
    #[serde(default)]
    pub bias_correction_bps: f64,
    /// Number of AS bias observations
    #[serde(default)]
    pub bias_observation_count: usize,
}

fn default_ewma_var() -> f64 {
    1.0
}

fn default_regime_trust() -> f64 {
    1.0
}

fn default_momentum_prior() -> f64 {
    0.5
}

impl Default for PreFillCheckpoint {
    fn default() -> Self {
        Self {
            learned_weights: [0.30, 0.25, 0.25, 0.10, 0.10],
            signal_outcome_sum: [0.0; 5],
            signal_sq_sum: [0.0; 5],
            learning_samples: 0,
            regime_probs: [0.1, 0.7, 0.15, 0.05],
            // EWMA normalizer state
            imbalance_ewma_mean: 0.0,
            imbalance_ewma_var: 1.0,
            flow_ewma_mean: 0.0,
            flow_ewma_var: 1.0,
            funding_ewma_mean: 0.0,
            funding_ewma_var: 1.0,
            regime_trust_prev: 1.0,
            regime_ewma_mean: 0.0,
            regime_ewma_var: 1.0,
            changepoint_ewma_mean: 0.0,
            changepoint_ewma_var: 1.0,
            normalizer_obs_count: 0,
            bias_correction_bps: 0.0,
            bias_observation_count: 0,
        }
    }
}

/// EnhancedASClassifier learning state.
///
/// The nested MicrostructureExtractor is NOT persisted — it rebuilds from
/// live trades within ~100 trades (~30 seconds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedCheckpoint {
    /// Learned feature weights (10 features)
    #[serde(default)]
    pub learned_weights: [f64; 10],
    /// Weight gradient momentum
    #[serde(default)]
    pub weight_gradients: [f64; 10],
    /// Number of learning samples
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub learning_samples: usize,
    /// Sum of predictions (for calibration tracking)
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub prediction_sum: f64,
    /// Sum of outcomes (for calibration tracking)
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub outcome_sum: f64,
    /// Correct predictions count
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub correct_predictions: usize,
    /// Total predictions count
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub total_predictions: usize,
}

impl Default for EnhancedCheckpoint {
    fn default() -> Self {
        Self {
            learned_weights: [0.1; 10],
            weight_gradients: [0.0; 10],
            learning_samples: 0,
            prediction_sum: 0.0,
            outcome_sum: 0.0,
            correct_predictions: 0,
            total_predictions: 0,
        }
    }
}

/// VolatilityFilter sufficient statistics.
///
/// Saves summary statistics instead of full particle cloud (200 bytes vs 20KB).
/// On restore: reinitialize particles centered on saved sigma_mean with sigma_std spread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolFilterCheckpoint {
    /// Posterior mean of sigma
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub sigma_mean: f64,
    /// Posterior standard deviation of sigma
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub sigma_std: f64,
    /// Regime probabilities [low, normal, high, extreme]
    #[serde(default)]
    pub regime_probs: [f64; 4],
    /// Total observations processed
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub observation_count: usize,
    /// Q18: Number of quote-cycle intervals where a fill occurred (bias tracker)
    #[serde(default)]
    pub bias_fill_intervals: u64,
    /// Q18: Number of quote-cycle intervals without a fill (bias tracker)
    #[serde(default)]
    pub bias_nonfill_intervals: u64,
}

impl Default for VolFilterCheckpoint {
    fn default() -> Self {
        Self {
            sigma_mean: 0.0005, // Reasonable crypto vol prior (~5 bps/sqrt(s))
            sigma_std: 0.0002, // Non-degenerate prior spread
            regime_probs: [0.1, 0.7, 0.15, 0.05],
            observation_count: 0,
            bias_fill_intervals: 0,
            bias_nonfill_intervals: 0,
        }
    }
}

/// RegimeHMM belief state and learned transitions.
///
/// VecDeque observation buffers (vol_buffer, spread_buffer) are NOT persisted —
/// they refill from live data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeHMMCheckpoint {
    /// Current belief state probabilities [4 regimes]
    #[serde(default)]
    pub belief: [f64; 4],
    /// Learned transition counts [from][to] (Bayesian sufficient statistics)
    #[serde(default)]
    pub transition_counts: [[f64; 4]; 4],
    /// Total observations processed
    #[serde(default, deserialize_with = "deserialize_u64_or_null")]
    pub observation_count: u64,
    /// Number of recalibrations performed
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub recalibration_count: usize,
}

impl Default for RegimeHMMCheckpoint {
    fn default() -> Self {
        Self {
            belief: [0.1, 0.7, 0.15, 0.05],
            transition_counts: [[0.0; 4]; 4],
            observation_count: 0,
            recalibration_count: 0,
        }
    }
}

/// InformedFlowEstimator mixture model state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformedFlowCheckpoint {
    /// Component parameters for 3-component mixture [informed, noise, forced]
    #[serde(default)]
    pub component_params: [ComponentParams; 3],
    /// Mixing weights for the 3 components
    #[serde(default)]
    pub mixing_weights: [f64; 3],
    /// Total observations processed
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub observation_count: usize,
}

impl Default for InformedFlowCheckpoint {
    fn default() -> Self {
        Self {
            component_params: [
                ComponentParams::default(),
                ComponentParams::default(),
                ComponentParams::default(),
            ],
            mixing_weights: [0.25, 0.50, 0.25],
            observation_count: 0,
        }
    }
}

/// FillRateModel Bayesian regression posteriors.
///
/// Saves the BayesianEstimate (mean/variance/n_obs) for λ₀ and δ*.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillRateCheckpoint {
    /// Posterior mean for base fill rate λ₀
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub lambda_0_mean: f64,
    /// Posterior variance for base fill rate λ₀
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub lambda_0_variance: f64,
    /// Effective observations for λ₀
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub lambda_0_n_obs: f64,
    /// Posterior mean for characteristic distance δ*
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub delta_char_mean: f64,
    /// Posterior variance for characteristic distance δ*
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub delta_char_variance: f64,
    /// Effective observations for δ*
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub delta_char_n_obs: f64,
    /// Total observations processed
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub observation_count: usize,
}

impl Default for FillRateCheckpoint {
    fn default() -> Self {
        Self {
            lambda_0_mean: 0.1,
            lambda_0_variance: 0.001,
            lambda_0_n_obs: 10.0,
            delta_char_mean: 10.0,
            delta_char_variance: 20.0,
            delta_char_n_obs: 5.0,
            observation_count: 0,
        }
    }
}

/// BayesianKappaEstimator sufficient statistics.
///
/// The rolling observation window (VecDeque) is NOT persisted — the sufficient
/// statistics (observation_count, sum_distances) fully determine the posterior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KappaCheckpoint {
    /// Prior/posterior shape parameter α
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub prior_alpha: f64,
    /// Prior/posterior rate parameter β
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub prior_beta: f64,
    /// Total observations
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub observation_count: usize,
    /// Sum of distances (Σδᵢ)
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub sum_distances: f64,
    /// Sum of squared distances (Σδᵢ²) for variance
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub sum_sq_distances: f64,
    /// Cached posterior mean κ̂ = α/β
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub kappa_posterior_mean: f64,
    /// Cumulative observation count (never decremented by rolling window expiry).
    /// Used by the calibration gate instead of `observation_count` which is
    /// a rolling-window count that can drop below thresholds.
    #[serde(default)]
    pub total_observations: usize,
}

impl Default for KappaCheckpoint {
    fn default() -> Self {
        Self {
            prior_alpha: 10.0,
            prior_beta: 0.02,
            observation_count: 0,
            sum_distances: 0.0,
            sum_sq_distances: 0.0,
            kappa_posterior_mean: 500.0,
            total_observations: 0,
        }
    }
}

/// KappaOrchestrator warmup state.
///
/// Persists cumulative fill count and warmup graduation so the
/// orchestrator doesn't re-enter warmup mode after every restart
/// (which causes kappa to swing 2-3x as the blending formula changes).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KappaOrchestratorCheckpoint {
    /// Whether the orchestrator has ever exited warmup mode.
    /// Once true, warmup is never re-entered (prevents kappa swings).
    #[serde(default)]
    pub has_exited_warmup: bool,
    /// Cumulative own-fill count (never decremented by rolling window expiry).
    /// Used for warmup gate instead of rolling-window observation_count.
    #[serde(default)]
    pub total_own_fills: u64,
}

/// MomentumModel learned continuation probabilities.
///
/// The observation VecDeque is NOT persisted — it's a rolling window.
/// The valuable state is the per-magnitude continuation probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumCheckpoint {
    /// Continuation probability by magnitude bucket [0-10, 10-20, ..., 90+ bps]
    #[serde(default)]
    pub continuation_by_magnitude: [f64; 10],
    /// Observation counts per magnitude bucket
    #[serde(default)]
    pub counts_by_magnitude: [usize; 10],
    /// Prior probability of momentum continuation
    #[serde(default = "default_momentum_prior", deserialize_with = "deserialize_f64_or_null")]
    pub prior_continuation: f64,
}

impl Default for MomentumCheckpoint {
    fn default() -> Self {
        Self {
            continuation_by_magnitude: [0.5; 10],
            counts_by_magnitude: [0; 10],
            prior_continuation: 0.5,
        }
    }
}

/// Kelly win/loss tracker learned state.
///
/// Persists EWMA of wins/losses and counts so Kelly sizing
/// doesn't restart from priors after a restart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyTrackerCheckpoint {
    /// EWMA of win sizes (bps)
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub ewma_wins: f64,
    /// Count of wins
    #[serde(default, deserialize_with = "deserialize_u64_or_null")]
    pub n_wins: u64,
    /// EWMA of loss sizes (bps)
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub ewma_losses: f64,
    /// Count of losses
    #[serde(default, deserialize_with = "deserialize_u64_or_null")]
    pub n_losses: u64,
    /// EWMA decay factor
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub decay: f64,
    /// Prediction horizon (ms) used when these observations were collected.
    /// On restore, if this differs from the current config horizon, the
    /// Kelly state is discarded and reset to priors (stale-observation invalidation).
    #[serde(default)]
    pub horizon_ms: u64,
}

impl Default for KellyTrackerCheckpoint {
    fn default() -> Self {
        Self {
            ewma_wins: 5.0,
            n_wins: 0,
            ewma_losses: 3.0,
            n_losses: 0,
            decay: 0.99,
            horizon_ms: 0,
        }
    }
}

/// Model ensemble weight state.
///
/// Persists softmax weights learned from fill outcomes
/// so the ensemble doesn't restart from uniform priors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleWeightsCheckpoint {
    /// Softmax weights per model [GLFT, Empirical, Funding]
    #[serde(default)]
    pub model_weights: Vec<f64>,
    /// Total weight updates performed
    #[serde(default, deserialize_with = "deserialize_usize_or_null")]
    pub total_updates: usize,
}

impl Default for EnsembleWeightsCheckpoint {
    fn default() -> Self {
        Self {
            model_weights: vec![0.5, 0.3, 0.2],
            total_updates: 0,
        }
    }
}


/// Baseline tracker checkpoint for counterfactual reward centering.
///
/// Persists the EWMA baseline so the bandit doesn't lose its fee-drag estimate
/// on restart (avoids ~50-fill warmup period).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineTrackerCheckpoint {
    /// EWMA of realized edge (typically ~-1.5 bps from fee drag)
    #[serde(default)]
    pub ewma_reward: f64,
    /// Number of observations
    #[serde(default)]
    pub n_observations: u64,
}

impl Default for BaselineTrackerCheckpoint {
    fn default() -> Self {
        Self {
            ewma_reward: 0.0,
            n_observations: 0,
        }
    }
}

/// Kill switch state for checkpoint persistence.
///
/// Allows restoring triggered state after restart so the system
/// doesn't accidentally resume trading after an emergency shutdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchCheckpoint {
    /// Whether the kill switch was triggered
    #[serde(default)]
    pub triggered: bool,
    /// Reasons for triggering (may have multiple)
    #[serde(default)]
    pub trigger_reasons: Vec<String>,
    /// Daily P&L at checkpoint time
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub daily_pnl: f64,
    /// Peak P&L for drawdown calculation
    #[serde(default, deserialize_with = "deserialize_f64_or_null")]
    pub peak_pnl: f64,
    /// Timestamp when kill switch was triggered (ms since epoch), 0 if not triggered
    #[serde(default, deserialize_with = "deserialize_u64_or_null")]
    pub triggered_at_ms: u64,
    /// Timestamp when checkpoint was saved (ms since epoch).
    /// Used to detect trading day boundaries and reset daily P&L on new day.
    #[serde(default, deserialize_with = "deserialize_u64_or_null")]
    pub saved_at_ms: u64,

    /// Q20: Consecutive stuck cycles at checkpoint time.
    #[serde(default)]
    pub position_stuck_cycles: u32,

    /// Q20: Cumulative unrealized adverse selection cost in USD.
    #[serde(default)]
    pub unrealized_as_cost_usd: f64,
}

impl Default for KillSwitchCheckpoint {
    fn default() -> Self {
        Self {
            triggered: false,
            trigger_reasons: Vec::new(),
            daily_pnl: 0.0,
            peak_pnl: 0.0,
            triggered_at_ms: 0,
            saved_at_ms: 0,
            position_stuck_cycles: 0,
            unrealized_as_cost_usd: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_bundle_roundtrip_json() {
        // Create a bundle with non-default values
        let bundle = CheckpointBundle {
            metadata: CheckpointMetadata {
                version: 1,
                timestamp_ms: 1700000000000,
                asset: "ETH".to_string(),
                session_duration_s: 3600.0,
                ..Default::default()
            },
            learned_params: LearnedParameters::default(),
            pre_fill: PreFillCheckpoint {
                learned_weights: [0.35, 0.20, 0.20, 0.15, 0.10],
                signal_outcome_sum: [1.0, 2.0, 3.0, 4.0, 5.0],
                signal_sq_sum: [10.0, 20.0, 30.0, 40.0, 50.0],
                learning_samples: 1000,
                regime_probs: [0.05, 0.60, 0.25, 0.10],
                ..PreFillCheckpoint::default()
            },
            enhanced: EnhancedCheckpoint::default(),
            vol_filter: VolFilterCheckpoint {
                sigma_mean: 0.02,
                sigma_std: 0.005,
                regime_probs: [0.1, 0.5, 0.3, 0.1],
                observation_count: 500,
                bias_fill_intervals: 0,
                bias_nonfill_intervals: 0,
            },
            regime_hmm: RegimeHMMCheckpoint::default(),
            informed_flow: InformedFlowCheckpoint::default(),
            fill_rate: FillRateCheckpoint::default(),
            kappa_own: KappaCheckpoint {
                prior_alpha: 15.0,
                prior_beta: 0.03,
                observation_count: 200,
                sum_distances: 4.0,
                sum_sq_distances: 0.08,
                kappa_posterior_mean: 500.0,
                total_observations: 200,
            },
            kappa_bid: KappaCheckpoint::default(),
            kappa_ask: KappaCheckpoint::default(),
            momentum: MomentumCheckpoint::default(),
            kelly_tracker: KellyTrackerCheckpoint {
                ewma_wins: 7.5,
                n_wins: 42,
                ewma_losses: 2.1,
                n_losses: 18,
                decay: 0.98,
                horizon_ms: 500,
            },
            ensemble_weights: EnsembleWeightsCheckpoint {
                model_weights: vec![0.6, 0.25, 0.15],
                total_updates: 100,
            },
            quote_outcomes: QuoteOutcomeCheckpoint { bins: vec![(0.0, 2.0, 5, 20), (2.0, 4.0, 10, 30)] },
            spread_bandit: SpreadBanditCheckpoint::default(),
            baseline_tracker: BaselineTrackerCheckpoint {
                ewma_reward: -1.5,
                n_observations: 42,
            },
            kill_switch: KillSwitchCheckpoint {
                triggered: true,
                trigger_reasons: vec!["Max daily loss exceeded".to_string()],
                daily_pnl: -100.0,
                peak_pnl: 50.0,
                triggered_at_ms: 1700000000000,
                saved_at_ms: 1700000000000,
                position_stuck_cycles: 0,
                unrealized_as_cost_usd: 0.0,
            },
            readiness: PriorReadiness::default(),
            calibration_coordinator: CalibrationCoordinatorCheckpoint::default(),
            kappa_orchestrator: KappaOrchestratorCheckpoint::default(),
            prior_confidence: 0.0,
            bayesian_fair_value: Default::default(),
        };

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&bundle).expect("serialize");

        // Deserialize back
        let restored: CheckpointBundle = serde_json::from_str(&json).expect("deserialize");

        // Verify key fields survived round-trip
        assert_eq!(restored.metadata.version, 1);
        assert_eq!(restored.metadata.asset, "ETH");
        assert_eq!(restored.pre_fill.learning_samples, 1000);
        assert_eq!(restored.pre_fill.learned_weights[0], 0.35);
        assert_eq!(restored.vol_filter.sigma_mean, 0.02);
        assert_eq!(restored.vol_filter.observation_count, 500);
        assert_eq!(restored.kappa_own.prior_alpha, 15.0);
        assert_eq!(restored.kappa_own.observation_count, 200);
        // Kelly tracker round-trip
        assert_eq!(restored.kelly_tracker.n_wins, 42);
        assert_eq!(restored.kelly_tracker.n_losses, 18);
        assert_eq!(restored.kelly_tracker.ewma_wins, 7.5);
        assert_eq!(restored.kelly_tracker.decay, 0.98);
        // Ensemble weights round-trip
        assert_eq!(restored.ensemble_weights.model_weights, vec![0.6, 0.25, 0.15]);
        assert_eq!(restored.ensemble_weights.total_updates, 100);
        // Quote outcomes round-trip
        assert_eq!(restored.quote_outcomes.bins.len(), 2);
        assert_eq!(restored.quote_outcomes.bins[0].2, 5);
        // Kill switch round-trip
        assert!(restored.kill_switch.triggered);
        assert_eq!(restored.kill_switch.trigger_reasons.len(), 1);
        assert_eq!(restored.kill_switch.daily_pnl, -100.0);
        assert_eq!(restored.kill_switch.peak_pnl, 50.0);
        assert_eq!(restored.kill_switch.triggered_at_ms, 1700000000000);
        // Readiness round-trip
        assert_eq!(restored.readiness.verdict, PriorVerdict::Insufficient);
        // Calibration coordinator round-trip
        assert_eq!(restored.calibration_coordinator.fill_count(), 0);
        assert_eq!(
            restored.calibration_coordinator.phase(),
            crate::market_maker::estimator::calibration_coordinator::CalibrationPhase::Cold,
        );
    }

    #[test]
    fn test_defaults_are_sane() {
        let pre_fill = PreFillCheckpoint::default();
        let sum: f64 = pre_fill.learned_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Default weights should sum to ~1.0");

        let regime_sum: f64 = pre_fill.regime_probs.iter().sum();
        assert!((regime_sum - 1.0).abs() < 0.01, "Regime probs should sum to ~1.0");

        let momentum = MomentumCheckpoint::default();
        assert_eq!(momentum.prior_continuation, 0.5);
    }

    #[test]
    fn test_checkpoint_bundle_backward_compat_no_kill_switch() {
        // Simulate a checkpoint JSON from before kill_switch was added.
        // The kill_switch field should default to KillSwitchCheckpoint::default().
        let bundle = CheckpointBundle {
            metadata: CheckpointMetadata {
                version: 1,
                timestamp_ms: 1700000000000,
                asset: "ETH".to_string(),
                session_duration_s: 100.0,
                ..Default::default()
            },
            learned_params: LearnedParameters::default(),
            pre_fill: PreFillCheckpoint::default(),
            enhanced: EnhancedCheckpoint::default(),
            vol_filter: VolFilterCheckpoint::default(),
            regime_hmm: RegimeHMMCheckpoint::default(),
            informed_flow: InformedFlowCheckpoint::default(),
            fill_rate: FillRateCheckpoint::default(),
            kappa_own: KappaCheckpoint::default(),
            kappa_bid: KappaCheckpoint::default(),
            kappa_ask: KappaCheckpoint::default(),
            momentum: MomentumCheckpoint::default(),
            kelly_tracker: KellyTrackerCheckpoint::default(),
            ensemble_weights: EnsembleWeightsCheckpoint::default(),
            quote_outcomes: QuoteOutcomeCheckpoint::default(),
            spread_bandit: SpreadBanditCheckpoint::default(),
            baseline_tracker: BaselineTrackerCheckpoint::default(),
            kill_switch: KillSwitchCheckpoint::default(),
            readiness: PriorReadiness::default(),
            calibration_coordinator: CalibrationCoordinatorCheckpoint::default(),
            kappa_orchestrator: KappaOrchestratorCheckpoint::default(),
            prior_confidence: 0.0,
            bayesian_fair_value: Default::default(),
        };

        // Serialize to JSON
        let json = serde_json::to_string(&bundle).expect("serialize");

        // Remove the kill_switch field to simulate an old checkpoint
        let json_without_ks: serde_json::Value =
            serde_json::from_str(&json).expect("parse");
        let mut map = json_without_ks.as_object().unwrap().clone();
        map.remove("kill_switch");
        let old_json = serde_json::to_string(&map).expect("re-serialize");

        // Deserialize — should use default for missing kill_switch field
        let restored: CheckpointBundle =
            serde_json::from_str(&old_json).expect("deserialize old format");

        assert!(!restored.kill_switch.triggered);
        assert!(restored.kill_switch.trigger_reasons.is_empty());
        assert_eq!(restored.kill_switch.daily_pnl, 0.0);
        assert_eq!(restored.kill_switch.peak_pnl, 0.0);
        assert_eq!(restored.kill_switch.triggered_at_ms, 0);
    }

    #[test]
    fn test_checkpoint_bundle_backward_compat_no_readiness() {
        let bundle = CheckpointBundle {
            metadata: CheckpointMetadata {
                version: 1,
                timestamp_ms: 1700000000000,
                asset: "ETH".to_string(),
                session_duration_s: 100.0,
                ..Default::default()
            },
            learned_params: LearnedParameters::default(),
            pre_fill: PreFillCheckpoint::default(),
            enhanced: EnhancedCheckpoint::default(),
            vol_filter: VolFilterCheckpoint::default(),
            regime_hmm: RegimeHMMCheckpoint::default(),
            informed_flow: InformedFlowCheckpoint::default(),
            fill_rate: FillRateCheckpoint::default(),
            kappa_own: KappaCheckpoint::default(),
            kappa_bid: KappaCheckpoint::default(),
            kappa_ask: KappaCheckpoint::default(),
            momentum: MomentumCheckpoint::default(),
            kelly_tracker: KellyTrackerCheckpoint::default(),
            ensemble_weights: EnsembleWeightsCheckpoint::default(),
            quote_outcomes: QuoteOutcomeCheckpoint::default(),
            spread_bandit: SpreadBanditCheckpoint::default(),
            baseline_tracker: BaselineTrackerCheckpoint::default(),
            kill_switch: KillSwitchCheckpoint::default(),
            readiness: PriorReadiness::default(),
            calibration_coordinator: CalibrationCoordinatorCheckpoint::default(),
            kappa_orchestrator: KappaOrchestratorCheckpoint::default(),
            prior_confidence: 0.0,
            bayesian_fair_value: Default::default(),
        };
        let json = serde_json::to_string(&bundle).expect("serialize");
        let mut map: serde_json::Value = serde_json::from_str(&json).expect("parse");
        map.as_object_mut().unwrap().remove("readiness");
        let old_json = serde_json::to_string(&map).expect("re-serialize");
        let restored: CheckpointBundle = serde_json::from_str(&old_json).expect("deserialize old format");
        assert_eq!(restored.readiness.verdict, PriorVerdict::Insufficient);
        assert_eq!(restored.readiness.estimators_ready, 0);
    }

    #[test]
    fn test_checkpoint_bundle_backward_compat_no_calibration_coordinator() {
        let bundle = CheckpointBundle {
            metadata: CheckpointMetadata {
                version: 1,
                timestamp_ms: 1700000000000,
                asset: "ETH".to_string(),
                session_duration_s: 100.0,
                ..Default::default()
            },
            learned_params: LearnedParameters::default(),
            pre_fill: PreFillCheckpoint::default(),
            enhanced: EnhancedCheckpoint::default(),
            vol_filter: VolFilterCheckpoint::default(),
            regime_hmm: RegimeHMMCheckpoint::default(),
            informed_flow: InformedFlowCheckpoint::default(),
            fill_rate: FillRateCheckpoint::default(),
            kappa_own: KappaCheckpoint::default(),
            kappa_bid: KappaCheckpoint::default(),
            kappa_ask: KappaCheckpoint::default(),
            momentum: MomentumCheckpoint::default(),
            kelly_tracker: KellyTrackerCheckpoint::default(),
            ensemble_weights: EnsembleWeightsCheckpoint::default(),
            quote_outcomes: QuoteOutcomeCheckpoint::default(),
            spread_bandit: SpreadBanditCheckpoint::default(),
            baseline_tracker: BaselineTrackerCheckpoint::default(),
            kill_switch: KillSwitchCheckpoint::default(),
            readiness: PriorReadiness::default(),
            calibration_coordinator: CalibrationCoordinatorCheckpoint::default(),
            kappa_orchestrator: KappaOrchestratorCheckpoint::default(),
            prior_confidence: 0.0,
            bayesian_fair_value: Default::default(),
        };
        let json = serde_json::to_string(&bundle).expect("serialize");
        let mut map: serde_json::Value = serde_json::from_str(&json).expect("parse");
        map.as_object_mut().unwrap().remove("calibration_coordinator");
        let old_json = serde_json::to_string(&map).expect("re-serialize");
        let restored: CheckpointBundle =
            serde_json::from_str(&old_json).expect("deserialize old format without calibration_coordinator");
        // Should default to Cold phase with 0 fills
        assert_eq!(
            restored.calibration_coordinator.phase(),
            crate::market_maker::estimator::calibration_coordinator::CalibrationPhase::Cold,
        );
        assert_eq!(restored.calibration_coordinator.fill_count(), 0);
        assert!(!restored.calibration_coordinator.is_seeded());
    }

    #[test]
    fn test_checkpoint_bundle_from_empty_json() {
        // An empty JSON object should deserialize successfully with all defaults
        let restored: CheckpointBundle =
            serde_json::from_str("{}").expect("deserialize from empty JSON");
        assert_eq!(restored.metadata.version, 0);
        assert_eq!(restored.metadata.asset, "");
        assert_eq!(restored.pre_fill.learning_samples, 0);
        assert_eq!(restored.vol_filter.observation_count, 0);
        assert_eq!(restored.kappa_own.observation_count, 0);
        assert_eq!(restored.fill_rate.observation_count, 0);
        assert!(!restored.kill_switch.triggered);
    }

    #[test]
    fn test_checkpoint_bundle_null_f64_fields() {
        // Simulate JSON with null values for f64 fields — should not panic
        let json = r#"{
            "metadata": {"version": 1, "timestamp_ms": null, "asset": "ETH", "session_duration_s": null},
            "fill_rate": {
                "lambda_0_mean": null,
                "lambda_0_variance": null,
                "lambda_0_n_obs": null,
                "delta_char_mean": null,
                "delta_char_variance": null,
                "delta_char_n_obs": null,
                "observation_count": null
            },
            "kappa_own": {
                "prior_alpha": null,
                "prior_beta": null,
                "observation_count": null,
                "sum_distances": null,
                "sum_sq_distances": null,
                "kappa_posterior_mean": null
            },
            "kill_switch": {
                "triggered": false,
                "trigger_reasons": [],
                "daily_pnl": null,
                "peak_pnl": null,
                "triggered_at_ms": null,
                "saved_at_ms": null
            }
        }"#;
        let restored: CheckpointBundle =
            serde_json::from_str(json).expect("deserialize with null f64 fields");
        assert_eq!(restored.metadata.timestamp_ms, 0);
        assert_eq!(restored.metadata.session_duration_s, 0.0);
        assert_eq!(restored.fill_rate.lambda_0_mean, 0.0);
        assert_eq!(restored.fill_rate.observation_count, 0);
        assert_eq!(restored.kappa_own.prior_alpha, 0.0);
        assert_eq!(restored.kill_switch.daily_pnl, 0.0);
        assert_eq!(restored.kill_switch.triggered_at_ms, 0);
    }
}
