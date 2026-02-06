//! Checkpoint data types for persisting learned state across sessions.
//!
//! Each checkpoint struct captures the minimum learning state needed to warm-start
//! a component. Ephemeral state (cached values, timestamps, rolling windows) is NOT
//! persisted — it rebuilds from live data within seconds.

use serde::{Deserialize, Serialize};

use crate::market_maker::calibration::parameter_learner::LearnedParameters;
use crate::market_maker::ComponentParams;

/// Complete checkpoint bundle containing all model state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointBundle {
    pub metadata: CheckpointMetadata,
    /// LearnedParameters — already Serialize/Deserialize, ~20 Bayesian posteriors
    pub learned_params: LearnedParameters,
    /// Pre-fill adverse selection classifier learning state
    pub pre_fill: PreFillCheckpoint,
    /// Enhanced AS classifier learning state
    pub enhanced: EnhancedCheckpoint,
    /// Volatility filter sufficient statistics
    pub vol_filter: VolFilterCheckpoint,
    /// Regime HMM belief state + transition counts
    pub regime_hmm: RegimeHMMCheckpoint,
    /// Informed flow mixture model parameters
    pub informed_flow: InformedFlowCheckpoint,
    /// Fill rate model Bayesian regression posteriors
    pub fill_rate: FillRateCheckpoint,
    /// BayesianKappaEstimator — own fills
    pub kappa_own: KappaCheckpoint,
    /// BayesianKappaEstimator — bid fills
    pub kappa_bid: KappaCheckpoint,
    /// BayesianKappaEstimator — ask fills
    pub kappa_ask: KappaCheckpoint,
    /// Momentum model — continuation probabilities by magnitude
    pub momentum: MomentumCheckpoint,
}

/// Checkpoint metadata for versioning and diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Schema version for forward compatibility
    pub version: u32,
    /// Timestamp when checkpoint was saved (ms since epoch)
    pub timestamp_ms: u64,
    /// Asset this checkpoint is for (e.g., "ETH")
    pub asset: String,
    /// How long the session ran before this checkpoint (seconds)
    pub session_duration_s: f64,
}

/// PreFillASClassifier learning state.
///
/// Captures online learning weights and sufficient statistics.
/// Ephemeral state (cached_toxicity, signal values, timestamps) rebuilds from live data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFillCheckpoint {
    /// Learned signal weights [imbalance, flow, regime, funding, changepoint]
    pub learned_weights: [f64; 5],
    /// Running sum of (signal_i * outcome) for each signal
    pub signal_outcome_sum: [f64; 5],
    /// Running sum of signal_i^2 for each signal
    pub signal_sq_sum: [f64; 5],
    /// Number of learning samples processed
    pub learning_samples: usize,
    /// Regime probabilities for soft blending
    pub regime_probs: [f64; 4],
}

impl Default for PreFillCheckpoint {
    fn default() -> Self {
        Self {
            learned_weights: [0.30, 0.25, 0.25, 0.10, 0.10],
            signal_outcome_sum: [0.0; 5],
            signal_sq_sum: [0.0; 5],
            learning_samples: 0,
            regime_probs: [0.1, 0.7, 0.15, 0.05],
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
    pub learned_weights: [f64; 10],
    /// Weight gradient momentum
    pub weight_gradients: [f64; 10],
    /// Number of learning samples
    pub learning_samples: usize,
    /// Sum of predictions (for calibration tracking)
    pub prediction_sum: f64,
    /// Sum of outcomes (for calibration tracking)
    pub outcome_sum: f64,
    /// Correct predictions count
    pub correct_predictions: usize,
    /// Total predictions count
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
    pub sigma_mean: f64,
    /// Posterior standard deviation of sigma
    pub sigma_std: f64,
    /// Regime probabilities [low, normal, high, extreme]
    pub regime_probs: [f64; 4],
    /// Total observations processed
    pub observation_count: usize,
}

impl Default for VolFilterCheckpoint {
    fn default() -> Self {
        Self {
            sigma_mean: 0.0,
            sigma_std: 0.0,
            regime_probs: [0.1, 0.7, 0.15, 0.05],
            observation_count: 0,
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
    pub belief: [f64; 4],
    /// Learned transition counts [from][to] (Bayesian sufficient statistics)
    pub transition_counts: [[f64; 4]; 4],
    /// Total observations processed
    pub observation_count: u64,
    /// Number of recalibrations performed
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
    pub component_params: [ComponentParams; 3],
    /// Mixing weights for the 3 components
    pub mixing_weights: [f64; 3],
    /// Total observations processed
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
    pub lambda_0_mean: f64,
    /// Posterior variance for base fill rate λ₀
    pub lambda_0_variance: f64,
    /// Effective observations for λ₀
    pub lambda_0_n_obs: f64,
    /// Posterior mean for characteristic distance δ*
    pub delta_char_mean: f64,
    /// Posterior variance for characteristic distance δ*
    pub delta_char_variance: f64,
    /// Effective observations for δ*
    pub delta_char_n_obs: f64,
    /// Total observations processed
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
    pub prior_alpha: f64,
    /// Prior/posterior rate parameter β
    pub prior_beta: f64,
    /// Total observations
    pub observation_count: usize,
    /// Sum of distances (Σδᵢ)
    pub sum_distances: f64,
    /// Sum of squared distances (Σδᵢ²) for variance
    pub sum_sq_distances: f64,
    /// Cached posterior mean κ̂ = α/β
    pub kappa_posterior_mean: f64,
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
        }
    }
}

/// MomentumModel learned continuation probabilities.
///
/// The observation VecDeque is NOT persisted — it's a rolling window.
/// The valuable state is the per-magnitude continuation probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumCheckpoint {
    /// Continuation probability by magnitude bucket [0-10, 10-20, ..., 90+ bps]
    pub continuation_by_magnitude: [f64; 10],
    /// Observation counts per magnitude bucket
    pub counts_by_magnitude: [usize; 10],
    /// Prior probability of momentum continuation
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
            },
            learned_params: LearnedParameters::default(),
            pre_fill: PreFillCheckpoint {
                learned_weights: [0.35, 0.20, 0.20, 0.15, 0.10],
                signal_outcome_sum: [1.0, 2.0, 3.0, 4.0, 5.0],
                signal_sq_sum: [10.0, 20.0, 30.0, 40.0, 50.0],
                learning_samples: 1000,
                regime_probs: [0.05, 0.60, 0.25, 0.10],
            },
            enhanced: EnhancedCheckpoint::default(),
            vol_filter: VolFilterCheckpoint {
                sigma_mean: 0.02,
                sigma_std: 0.005,
                regime_probs: [0.1, 0.5, 0.3, 0.1],
                observation_count: 500,
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
            },
            kappa_bid: KappaCheckpoint::default(),
            kappa_ask: KappaCheckpoint::default(),
            momentum: MomentumCheckpoint::default(),
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
}
