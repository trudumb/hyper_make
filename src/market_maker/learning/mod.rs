//! Closed-loop learning system for market making.
//!
//! This module implements a 5-level architecture that transforms
//! open-loop parameter estimation into closed-loop control where
//! fills are labeled data that update model confidence.
//!
//! ## Architecture
//!
//! ```text
//! Level 0: ParameterEstimator (exists) - σ, κ, microprice estimates
//! Level 1: ModelConfidenceTracker     - track prediction vs realization
//! Level 2: ModelEnsemble              - multiple edge models, weighted
//! Level 2.5: AdaptiveEnsemble         - dynamic IR-based model weighting
//! Level 3: DecisionEngine             - formal P(edge > 0) criterion
//! Level 4: ExecutionOptimizer         - utility-maximizing ladder
//! Level 5: CrossAssetSignals          - BTC lead-lag, funding divergence
//! ```
//!
//! ## Core Insight
//!
//! Every fill is a prediction that gets scored:
//! - At fill time: record `predicted_edge_bps` and `state`
//! - After horizon (1s): measure `realized_as_bps` and `realized_edge_bps`
//! - Update confidence tracker and ensemble weights
//!
//! This is online learning with labels from trading outcomes.

pub mod adaptive_ensemble;
pub mod baseline_tracker;
pub mod competitor_model;
pub mod confidence;
pub mod cross_asset;
pub mod decision;
pub mod ensemble;
pub mod execution;
pub mod experience;
pub mod quote_outcome;
pub mod rl_agent;
pub mod spread_bandit;
pub mod types;

// Re-export key types
pub use adaptive_ensemble::{AdaptiveEnsemble, EnsembleSummary, ModelPerformance};
pub use baseline_tracker::BaselineTracker;
pub use competitor_model::{
    BayesianGamma, CompetitorModel, CompetitorModelConfig, CompetitorSummary, MarketEvent, Side,
    SnipeTracker,
};
pub use confidence::{
    AggregateConfidence, EdgeBiasSummary, EdgeBiasTracker, ModelConfidenceTracker,
};
pub use decision::DecisionEngine;
pub use ensemble::{EdgeModel, ModelEnsemble};
pub use execution::ExecutionOptimizer;
pub use experience::{ExperienceLogger, ExperienceParams, ExperienceRecord, ExperienceSource};
pub use rl_agent::{
    ExplorationStrategy, MDPAction, MDPState, QLearningAgent, QLearningConfig,
    RLPolicyRecommendation, Reward, RewardConfig, SimToRealConfig,
};
pub use spread_bandit::{BanditSelection, SpreadBandit, SpreadBanditCheckpoint, SpreadContext};
pub use types::*;

use crate::market_maker::calibration::{CalibrationSample, CoefficientEstimator};
use crate::market_maker::fills::FillEvent;
use crate::market_maker::quoting::Ladder;
use crate::market_maker::strategy::{
    CalibratedRiskModel, MarketParams, RiskFeatures, RiskModelConfig, WinLossTracker,
};

/// Configuration for the LearningModule.
#[derive(Debug, Clone)]
pub struct LearningConfig {
    /// Whether learning is enabled
    pub enabled: bool,
    /// Prediction horizon in milliseconds (time to measure outcome)
    pub prediction_horizon_ms: u64,
    /// Minimum predictions before updating weights
    pub min_predictions_for_update: usize,
    /// Whether to use decision engine as quote filter
    pub use_decision_filter: bool,
    /// Log model health every N quote cycles
    pub health_log_interval: usize,
    /// Fee rate in basis points
    pub fee_bps: f64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prediction_horizon_ms: 500, // 500ms — aligned with AS measurement horizon
            min_predictions_for_update: 20,
            use_decision_filter: true, // Enabled by default - GLFT math is proven
            health_log_interval: 100,
            fee_bps: 1.5,
        }
    }
}

/// The learning module for closed-loop market making.
///
/// Integrates into MarketMaker without requiring Arc wrapping.
/// Tracks predictions, updates model weights, and provides decision support.
pub struct LearningModule {
    /// Configuration
    config: LearningConfig,

    // === Level 1 ===
    /// Model confidence tracker
    confidence_tracker: ModelConfidenceTracker,

    // === Level 2 ===
    /// Model ensemble
    ensemble: ModelEnsemble,

    // === Level 3 ===
    /// Decision engine
    decision_engine: DecisionEngine,

    // === Level 4 ===
    /// Execution optimizer
    execution_optimizer: ExecutionOptimizer,

    // === Feedback loop ===
    /// Pending predictions waiting for outcome
    pending_predictions: Vec<PendingPrediction>,

    /// Latest market mid for AS calculation
    last_mid: f64,

    /// Quote cycle counter for periodic logging
    quote_cycle_count: usize,

    // === Risk Model Calibration ===
    /// Coefficient estimator for calibrating risk model from fills
    coefficient_estimator: CoefficientEstimator,

    /// Kelly sizer for tracking win/loss ratios
    kelly_tracker: WinLossTracker,

    /// Risk model config for feature normalization consistency
    risk_model_config: RiskModelConfig,
}

impl Default for LearningModule {
    fn default() -> Self {
        Self::new(LearningConfig::default())
    }
}

impl LearningModule {
    /// Create a new learning module.
    pub fn new(config: LearningConfig) -> Self {
        Self {
            config,
            confidence_tracker: ModelConfidenceTracker::new(),
            ensemble: ModelEnsemble::new(),
            decision_engine: DecisionEngine::default(),
            execution_optimizer: ExecutionOptimizer::default(),
            pending_predictions: Vec::new(),
            last_mid: 0.0,
            quote_cycle_count: 0,
            coefficient_estimator: CoefficientEstimator::default(),
            kelly_tracker: WinLossTracker::default(),
            risk_model_config: RiskModelConfig::default(),
        }
    }

    /// Create a new learning module with custom risk model config.
    pub fn with_risk_model_config(
        config: LearningConfig,
        risk_model_config: RiskModelConfig,
    ) -> Self {
        Self {
            config,
            confidence_tracker: ModelConfidenceTracker::new(),
            ensemble: ModelEnsemble::new(),
            decision_engine: DecisionEngine::default(),
            execution_optimizer: ExecutionOptimizer::default(),
            pending_predictions: Vec::new(),
            last_mid: 0.0,
            quote_cycle_count: 0,
            coefficient_estimator: CoefficientEstimator::default(),
            kelly_tracker: WinLossTracker::default(),
            risk_model_config,
        }
    }

    /// Update the risk model config (e.g., after calibrating baselines).
    pub fn update_risk_model_config(&mut self, config: RiskModelConfig) {
        self.risk_model_config = config;
    }

    /// Check if learning is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the learning config for checkpoint serialization.
    pub fn config(&self) -> &LearningConfig {
        &self.config
    }

    /// Update the last known mid price.
    pub fn update_mid(&mut self, mid: f64) {
        self.last_mid = mid;
    }

    /// Handle a fill event - record prediction for later scoring.
    ///
    /// Called from FillProcessor with current market params and position.
    pub fn on_fill(&mut self, fill: &FillEvent, params: &MarketParams, position: f64) {
        if !self.config.enabled {
            return;
        }

        // 1. Build market state at fill time
        let state = self.build_market_state(params, position);

        // 2. Get ensemble prediction for this state
        let prediction = self.ensemble.predict_edge(&state);

        // 3. Record pending prediction
        let pending = PendingPrediction {
            timestamp_ms: fill.timestamp_ms(),
            fill: fill.clone(),
            predicted_edge_bps: prediction.mean,
            predicted_uncertainty: prediction.std,
            state,
            depth_bps: fill.depth_bps(),
            predicted_fill_prob: 1.0, // Filled, so probability was validated
        };
        self.pending_predictions.push(pending);

        // 4. Score any matured predictions
        self.score_matured_predictions();
    }

    /// Score predictions that have reached their measurement horizon.
    fn score_matured_predictions(&mut self) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let horizon_ms = self.config.prediction_horizon_ms;

        // Find matured predictions
        let mut matured = Vec::new();
        self.pending_predictions.retain(|pred| {
            if now_ms.saturating_sub(pred.timestamp_ms) >= horizon_ms {
                matured.push(pred.clone());
                false
            } else {
                true
            }
        });

        // Score each matured prediction
        for pred in matured {
            let outcome = self.measure_outcome(&pred);

            // Update confidence tracker
            self.confidence_tracker.record_edge_prediction(
                pred.predicted_edge_bps,
                pred.predicted_uncertainty,
                outcome.realized_edge_bps,
            );

            // Update ensemble weights
            self.ensemble.update_weights(&outcome);

            // === Risk Model Calibration Pipeline ===
            // Record sample for coefficient estimator (log-additive gamma calibration)
            // Use config to ensure baselines match those used in GLFTStrategy
            let features =
                RiskFeatures::from_state(&outcome.prediction.state, &self.risk_model_config);
            let sample = CalibrationSample {
                timestamp_ms: outcome.prediction.timestamp_ms,
                features,
                realized_as_bps: outcome.realized_as_bps,
                realized_edge_bps: outcome.realized_edge_bps,
                is_buy: outcome.prediction.fill.is_buy,
                depth_bps: outcome.prediction.depth_bps,
            };
            self.coefficient_estimator.record_sample(sample);

            // Update Kelly win/loss tracker
            let is_win = outcome.realized_edge_bps > 0.0;
            if is_win {
                self.kelly_tracker.record_win(outcome.realized_edge_bps);
            } else {
                self.kelly_tracker.record_loss(-outcome.realized_edge_bps);
            }

            // Diagnostic: log per-fill Kelly edge decomposition
            let spread_captured_bps = if outcome.prediction.fill.quoted_half_spread_bps > 0.0 {
                outcome.prediction.fill.quoted_half_spread_bps * 10000.0
            } else {
                outcome.prediction.depth_bps
            };
            tracing::debug!(
                spread_captured_bps = %format!("{:.2}", spread_captured_bps),
                as_measured_bps = %format!("{:.2}", outcome.realized_as_bps),
                fee_bps = %format!("{:.2}", self.config.fee_bps),
                edge_bps = %format!("{:.2}", outcome.realized_edge_bps),
                horizon_ms = self.config.prediction_horizon_ms,
                is_win = is_win,
                "Kelly edge decomposition"
            );
        }
    }

    /// Measure the outcome of a prediction.
    ///
    /// Adverse selection is measured as the price move from the *placement-time mid*
    /// to the *horizon-time mid*. Previously this used `self.last_mid` at both
    /// fill time and horizon time, which made AS tautologically equal to depth
    /// (same reference point). Now we use `fill.mid_at_placement` when available,
    /// falling back to `self.last_mid` for backward compatibility with old fills.
    fn measure_outcome(&self, pred: &PendingPrediction) -> TradingOutcome {
        let current_mid = self.last_mid;

        // Use placement-time mid for AS reference when available.
        // When mid_at_placement is 0.0 (unset / old fill events), fall back to
        // current mid (legacy behavior, still tautological but safe).
        let reference_mid = if pred.fill.mid_at_placement > 0.0 {
            pred.fill.mid_at_placement
        } else {
            current_mid
        };

        // AS = how much mid moved against us from placement to horizon.
        // For buys: mid dropping after we bought = adverse (positive AS)
        // For sells: mid rising after we sold = adverse (positive AS)
        let price_move = if pred.fill.is_buy {
            // Buy: AS positive when mid dropped (we bought, price fell)
            // reference_mid - current_mid > 0 means mid fell
            (reference_mid - current_mid) / reference_mid * 10_000.0
        } else {
            // Sell: AS positive when mid rose (we sold, price rose)
            // current_mid - reference_mid > 0 means mid rose
            (current_mid - reference_mid) / reference_mid * 10_000.0
        };

        let realized_as_bps = price_move.max(0.0);

        // Spread captured: use quoted half-spread when available (from placement mid),
        // otherwise fall back to depth_bps() which uses fill-time mid.
        let spread_captured_bps = if pred.fill.quoted_half_spread_bps > 0.0 {
            pred.fill.quoted_half_spread_bps
        } else {
            pred.fill.depth_bps()
        };
        let realized_edge_bps = spread_captured_bps - realized_as_bps - self.config.fee_bps;

        TradingOutcome {
            prediction: pred.clone(),
            realized_as_bps,
            realized_edge_bps,
            price_at_horizon: current_mid,
            horizon_elapsed_ms: self.config.prediction_horizon_ms,
        }
    }

    /// Build market state from current parameters.
    fn build_market_state(&self, params: &MarketParams, position: f64) -> MarketState {
        MarketState {
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            microprice: params.microprice,
            market_mid: params.market_mid,
            sigma: params.sigma,
            sigma_total: params.sigma_total,
            sigma_effective: params.sigma_effective,
            kappa: params.kappa,
            kappa_bid: params.kappa_bid,
            kappa_ask: params.kappa_ask,
            book_imbalance: params.book_imbalance,
            flow_imbalance: params.flow_imbalance,
            momentum_bps: params.momentum_bps,
            p_informed: params.p_informed,
            toxicity_score: params.toxicity_score,
            jump_ratio: params.jump_ratio,
            volatility_regime: match params.volatility_regime {
                crate::market_maker::estimator::VolatilityRegime::Low => VolatilityRegime::Low,
                crate::market_maker::estimator::VolatilityRegime::Normal => {
                    VolatilityRegime::Normal
                }
                crate::market_maker::estimator::VolatilityRegime::High => VolatilityRegime::High,
                crate::market_maker::estimator::VolatilityRegime::Extreme => {
                    VolatilityRegime::Extreme
                }
            },
            predicted_as_bps: params.total_as_bps,
            as_permanent_bps: params.as_permanent_bps,
            as_temporary_bps: params.as_temporary_bps,
            funding_rate: params.funding_rate,
            predicted_funding_cost: params.predicted_funding_cost,
            position,
            max_position: params.dynamic_max_position,
            cross_signal: None,
            actual_quoted_spread_bps: if params.market_spread_bps > 0.0 {
                Some(params.market_spread_bps)
            } else {
                None
            },
        }
    }

    /// Main quote cycle - decide whether to quote and optimize ladder.
    ///
    /// Called with current market params and position.
    pub fn quote_cycle(&mut self, params: &MarketParams, position: f64) -> Option<Ladder> {
        // 1. Build current market state
        let state = self.build_market_state(params, position);

        // 2. Get ensemble prediction
        let prediction = self.ensemble.predict_edge(&state);

        // 3. Check model health
        let health = self.confidence_tracker.model_health();

        // 4. Get drawdown (placeholder - should come from P&L tracker)
        let current_drawdown = 0.0;

        // 5. Decision (pass realized_vol for A-S reservation shift)
        let decision =
            self.decision_engine
                .should_quote(&prediction, &health, current_drawdown, params.sigma);

        // 6. Log decision
        tracing::debug!(
            predicted_edge = %format!("{:.2}bp", prediction.mean),
            uncertainty = %format!("{:.2}bp", prediction.std),
            disagreement = %format!("{:.2}bp", prediction.disagreement),
            model_health = ?health.overall,
            decision = ?decision,
            "Quote cycle decision"
        );

        // 7. Generate ladder if quoting
        match decision {
            QuoteDecision::Quote {
                size_fraction,
                expected_edge,
                ..
            } => {
                let ladder =
                    self.execution_optimizer
                        .optimize_ladder(params, size_fraction, expected_edge);
                Some(ladder)
            }
            QuoteDecision::ReducedSize { fraction, .. } => {
                let ladder =
                    self.execution_optimizer
                        .optimize_ladder(params, fraction, prediction.mean);
                Some(ladder)
            }
            QuoteDecision::NoQuote { reason } => {
                tracing::info!(reason = %reason, "Not quoting");
                None
            }
        }
    }

    /// Get the current model health.
    pub fn model_health(&self) -> ModelHealth {
        self.confidence_tracker.model_health()
    }

    /// Check if decision filter is enabled.
    pub fn use_decision_filter(&self) -> bool {
        self.config.enabled && self.config.use_decision_filter
    }

    /// Evaluate whether to quote based on decision engine.
    ///
    /// Returns the QuoteDecision which can be:
    /// - Quote: Full size with confidence and expected edge
    /// - ReducedSize: Partial size due to model disagreement
    /// - NoQuote: Don't quote with reason
    ///
    /// The caller should apply the decision to their quote sizing.
    pub fn evaluate_decision(
        &self,
        params: &MarketParams,
        position: f64,
        current_drawdown: f64,
    ) -> QuoteDecision {
        // Build current market state
        let state = self.build_market_state(params, position);

        // Get ensemble prediction
        let prediction = self.ensemble.predict_edge(&state);

        // Get model health
        let health = self.confidence_tracker.model_health();

        // Get decision from engine (pass realized_vol for A-S reservation shift)
        let decision =
            self.decision_engine
                .should_quote(&prediction, &health, current_drawdown, params.sigma);

        // Log the decision
        tracing::debug!(
            predicted_edge = %format!("{:.2}bp", prediction.mean),
            uncertainty = %format!("{:.2}bp", prediction.std),
            disagreement = %format!("{:.2}bp", prediction.disagreement),
            model_health = ?health.overall,
            drawdown = %format!("{:.2}%", current_drawdown * 100.0),
            decision = ?decision,
            "Decision engine evaluation"
        );

        decision
    }

    /// Track quote cycle and check if we should log model health.
    /// Returns true if health should be logged this cycle.
    /// Also scores any matured predictions periodically (not just on fills).
    pub fn should_log_health(&mut self) -> bool {
        self.quote_cycle_count += 1;

        // Score matured predictions every cycle (not just on fills)
        // This is critical for sparse fill environments
        self.score_matured_predictions();

        if self.config.health_log_interval == 0 {
            return false;
        }
        self.quote_cycle_count
            .is_multiple_of(self.config.health_log_interval)
    }

    /// Log detailed calibration metrics for diagnostics.
    /// Call this when should_log_health() returns true.
    pub fn log_calibration_report(&self) {
        let tracker = &self.confidence_tracker;
        tracing::info!(
            target: "layer2::calibration",
            vol_rmse = %format!("{:.4}", tracker.vol_rmse()),
            vol_bias = %format!("{:.4}", tracker.vol_bias()),
            as_rmse = %format!("{:.2}", tracker.as_rmse()),
            as_bias = %format!("{:.2}", tracker.as_bias()),
            edge_rmse = %format!("{:.2}", tracker.edge_rmse()),
            edge_bias = %format!("{:.2}", tracker.edge_bias()),
            n_edge_obs = tracker.n_edge_observations(),
            "[Calibration] Model confidence tracker metrics"
        );
    }

    /// Get the number of pending predictions.
    pub fn pending_predictions_count(&self) -> usize {
        self.pending_predictions.len()
    }

    /// Get confidence tracker for inspection.
    pub fn confidence_tracker(&self) -> &ModelConfidenceTracker {
        &self.confidence_tracker
    }

    /// Get ensemble for inspection.
    pub fn ensemble(&self) -> &ModelEnsemble {
        &self.ensemble
    }

    // === Risk Model Calibration Accessors ===

    /// Get the fitted calibrated risk model (if available).
    ///
    /// Returns Some(model) if coefficient estimator has enough samples.
    pub fn fitted_risk_model(&self) -> Option<&CalibratedRiskModel> {
        self.coefficient_estimator.fitted_model()
    }

    /// Check if risk model calibration is warmed up.
    pub fn risk_model_warmed_up(&self) -> bool {
        self.coefficient_estimator.is_warmed_up()
    }

    /// Get number of calibration samples.
    pub fn calibration_sample_count(&self) -> usize {
        self.coefficient_estimator.n_samples()
    }

    /// Get R² of risk model calibration.
    pub fn risk_model_r_squared(&self) -> f64 {
        self.coefficient_estimator.r_squared()
    }

    /// Get Kelly tracker win rate.
    pub fn kelly_win_rate(&self) -> f64 {
        self.kelly_tracker.win_rate()
    }

    /// Get Kelly tracker odds ratio.
    pub fn kelly_odds_ratio(&self) -> f64 {
        self.kelly_tracker.odds_ratio()
    }

    /// Get Kelly tracker statistics (avg_win, avg_loss, total_trades).
    pub fn kelly_stats(&self) -> (f64, f64, u64) {
        (
            self.kelly_tracker.avg_win(),
            self.kelly_tracker.avg_loss(),
            self.kelly_tracker.total_trades(),
        )
    }

    /// Check if Kelly tracker is warmed up.
    pub fn kelly_warmed_up(&self) -> bool {
        self.kelly_tracker.is_warmed_up()
    }

    /// Get Kelly-optimal position fraction for sizing.
    ///
    /// Returns Some(fraction) when tracker has enough data (50+ fills),
    /// None during warmup. Fraction is clamped to [0.05, 0.30].
    pub fn kelly_recommendation(&self) -> Option<f64> {
        if !self.kelly_tracker.is_warmed_up() {
            return None;
        }

        let p = self.kelly_tracker.win_rate();
        let b = self.kelly_tracker.odds_ratio();

        if b <= 0.0 || p <= 0.0 {
            return None;
        }

        let q = 1.0 - p;
        let f_full = (p * b - q) / b;

        if f_full <= 0.0 {
            return Some(0.05); // Minimum floor even with marginal edge
        }

        // 15% fractional Kelly, clamped
        let fraction = (0.15 * f_full).clamp(0.05, 0.30);
        Some(fraction)
    }

    /// Get the Kelly win/loss tracker for cloning to strategy.
    pub fn kelly_tracker(&self) -> &WinLossTracker {
        &self.kelly_tracker
    }

    /// Get current ensemble weights for checkpoint persistence.
    pub fn ensemble_weights(&self) -> Vec<f64> {
        self.ensemble.current_weights()
    }

    /// Get total ensemble weight updates for checkpoint persistence.
    pub fn ensemble_total_updates(&self) -> usize {
        self.ensemble.total_updates()
    }

    /// Restore Kelly tracker and ensemble weights from checkpoint.
    ///
    /// Kelly state is invalidated if the checkpoint horizon differs from the
    /// current config horizon — observations measured under a different window
    /// don't inform the current posterior (stale-observation invalidation).
    pub fn restore_from_checkpoint(
        &mut self,
        kelly: &crate::market_maker::checkpoint::KellyTrackerCheckpoint,
        ensemble: &crate::market_maker::checkpoint::EnsembleWeightsCheckpoint,
    ) {
        // Stale-observation invalidation: if horizon changed, discard Kelly state
        // and start from priors. This prevents 63 stale 1000ms losses from
        // contaminating a fresh 500ms model.
        if kelly.horizon_ms != 0 && kelly.horizon_ms != self.config.prediction_horizon_ms {
            tracing::warn!(
                checkpoint_horizon_ms = kelly.horizon_ms,
                config_horizon_ms = self.config.prediction_horizon_ms,
                stale_wins = kelly.n_wins,
                stale_losses = kelly.n_losses,
                "Kelly horizon mismatch — discarding stale checkpoint, starting from priors"
            );
            // Don't restore — keep default priors (ewma_wins=5.0, ewma_losses=3.0)
        } else {
            self.kelly_tracker.restore_from_checkpoint(
                kelly.ewma_wins,
                kelly.n_wins,
                kelly.ewma_losses,
                kelly.n_losses,
                kelly.decay,
            );
            if kelly.horizon_ms > 0 {
                tracing::info!(
                    horizon_ms = kelly.horizon_ms,
                    n_wins = kelly.n_wins,
                    n_losses = kelly.n_losses,
                    "Kelly tracker restored from checkpoint (horizon matches)"
                );
            }
        }
        self.ensemble
            .restore_weights(&ensemble.model_weights, ensemble.total_updates);
    }

    /// Get the risk model config for consistency with strategy.
    pub fn risk_model_config(&self) -> &RiskModelConfig {
        &self.risk_model_config
    }

    /// Update realized edge statistics from EdgeTracker for A-S feedback.
    ///
    /// Feeds realized edge outcomes back into the decision engine's p_positive
    /// calculation. When realized edge is persistently negative, the blended
    /// P(edge > 0) shifts below 0.5, increasing information asymmetry and
    /// reducing size — all within the existing A-S framework.
    pub fn update_realized_edge_stats(&mut self, mean_bps: f64, std_bps: f64, count: usize) {
        let config = self.decision_engine.config().clone();
        self.decision_engine
            .set_config(decision::DecisionEngineConfig {
                realized_edge_mean: mean_bps,
                realized_edge_std: std_bps,
                realized_edge_n: count,
                ..config
            });
    }

    /// Generate output for Layer 3 (StochasticController).
    ///
    /// This bundles all Layer 2 outputs into a single structure for the controller.
    pub fn output(
        &self,
        params: &MarketParams,
        position: f64,
        current_drawdown: f64,
    ) -> crate::market_maker::control::LearningModuleOutput {
        use crate::market_maker::control::{
            GaussianEstimate, LearningModuleOutput, ModelPrediction,
        };

        // Build market state
        let state = self.build_market_state(params, position);

        // Get ensemble prediction
        let prediction = self.ensemble.predict_edge(&state);

        // Get model health
        let health = self.confidence_tracker.model_health();

        // Get decision from engine (pass realized_vol for A-S reservation shift)
        let decision =
            self.decision_engine
                .should_quote(&prediction, &health, current_drawdown, params.sigma);

        // Convert model contributions
        let model_predictions: Vec<ModelPrediction> = prediction
            .model_contributions
            .iter()
            .map(|wp| ModelPrediction {
                name: wp.model.clone(),
                mean: wp.mean,
                std: wp.std,
                weight: wp.weight,
                performance: 1.0, // Could track this in confidence tracker
            })
            .collect();

        // Calculate p_positive_edge
        let p_positive = if prediction.std > 1e-9 {
            crate::market_maker::control::types::normal_cdf(prediction.mean / prediction.std)
        } else if prediction.mean > 0.0 {
            1.0
        } else {
            0.0
        };

        LearningModuleOutput {
            edge_prediction: GaussianEstimate::new(prediction.mean, prediction.std),
            model_predictions,
            model_disagreement: prediction.disagreement,
            model_health: health,
            calibration: self.confidence_tracker.edge_calibration_score().clone(),
            as_bias: self.confidence_tracker.as_bias(),
            myopic_decision: decision,
            p_positive_edge: p_positive,
            position,
            max_position: params.dynamic_max_position,
            drawdown: current_drawdown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_config_default() {
        let config = LearningConfig::default();
        assert!(config.enabled);
        assert_eq!(config.prediction_horizon_ms, 500);
        assert_eq!(config.min_predictions_for_update, 20);
        assert_eq!(config.fee_bps, 1.5);
    }

    #[test]
    fn test_learning_module_creation() {
        let module = LearningModule::default();
        assert!(module.is_enabled());
        assert_eq!(module.pending_predictions_count(), 0);
    }

    #[test]
    fn test_kelly_stale_invalidation_horizon_mismatch() {
        // Restore with horizon_ms=1000 into config with 500ms → state resets to priors
        let mut module = LearningModule::default(); // config has 500ms
        assert_eq!(module.config.prediction_horizon_ms, 500);

        let stale_kelly = crate::market_maker::checkpoint::KellyTrackerCheckpoint {
            ewma_wins: 10.0,
            n_wins: 0,
            ewma_losses: 20.0,
            n_losses: 63,
            decay: 0.99,
            horizon_ms: 1000, // Mismatch!
        };
        let ensemble = crate::market_maker::checkpoint::EnsembleWeightsCheckpoint::default();
        module.restore_from_checkpoint(&stale_kelly, &ensemble);

        // Should NOT have restored the stale state — should be at default priors
        assert_eq!(module.kelly_tracker.total_trades(), 0);
        assert_eq!(module.kelly_tracker.avg_win(), 5.0); // Default prior
        assert_eq!(module.kelly_tracker.avg_loss(), 3.0); // Default prior
    }

    #[test]
    fn test_kelly_fresh_restore_horizon_matches() {
        // Restore with horizon_ms=500 into config with 500ms → state preserved
        let mut module = LearningModule::default();

        let fresh_kelly = crate::market_maker::checkpoint::KellyTrackerCheckpoint {
            ewma_wins: 8.0,
            n_wins: 30,
            ewma_losses: 4.0,
            n_losses: 20,
            decay: 0.99,
            horizon_ms: 500, // Matches config
        };
        let ensemble = crate::market_maker::checkpoint::EnsembleWeightsCheckpoint::default();
        module.restore_from_checkpoint(&fresh_kelly, &ensemble);

        // Should have restored the state
        assert_eq!(module.kelly_tracker.total_trades(), 50);
    }

    #[test]
    fn test_kelly_restore_legacy_zero_horizon() {
        // Old checkpoints have horizon_ms=0 (default) — should restore (backward compat)
        let mut module = LearningModule::default();

        let legacy_kelly = crate::market_maker::checkpoint::KellyTrackerCheckpoint {
            ewma_wins: 7.0,
            n_wins: 25,
            ewma_losses: 3.5,
            n_losses: 15,
            decay: 0.99,
            horizon_ms: 0, // Legacy — no horizon recorded
        };
        let ensemble = crate::market_maker::checkpoint::EnsembleWeightsCheckpoint::default();
        module.restore_from_checkpoint(&legacy_kelly, &ensemble);

        // Should have restored (0 == "unknown", don't invalidate)
        assert_eq!(module.kelly_tracker.total_trades(), 40);
    }
}
