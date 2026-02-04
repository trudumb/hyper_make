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
pub mod competitor_model;
pub mod confidence;
pub mod decision;
pub mod ensemble;
pub mod execution;
pub mod rl_agent;
pub mod types;

// Re-export key types
pub use adaptive_ensemble::{AdaptiveEnsemble, EnsembleSummary, ModelPerformance};
pub use confidence::{AggregateConfidence, EdgeBiasSummary, EdgeBiasTracker, ModelConfidenceTracker};
pub use decision::DecisionEngine;
pub use ensemble::{EdgeModel, ModelEnsemble};
pub use execution::ExecutionOptimizer;
pub use rl_agent::{
    MDPAction, MDPState, QLearningAgent, QLearningConfig, RLPolicyRecommendation,
    Reward, RewardConfig, ExplorationStrategy,
};
pub use competitor_model::{
    CompetitorModel, CompetitorModelConfig, CompetitorSummary,
    MarketEvent, Side, BayesianGamma, SnipeTracker,
};
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
            prediction_horizon_ms: 1000, // 1 second
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
    pub fn with_risk_model_config(config: LearningConfig, risk_model_config: RiskModelConfig) -> Self {
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
            let features = RiskFeatures::from_state(&outcome.prediction.state, &self.risk_model_config);
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
            if outcome.realized_edge_bps > 0.0 {
                self.kelly_tracker.record_win(outcome.realized_edge_bps);
            } else {
                self.kelly_tracker.record_loss(-outcome.realized_edge_bps);
            }
        }
    }

    /// Measure the outcome of a prediction.
    fn measure_outcome(&self, pred: &PendingPrediction) -> TradingOutcome {
        // Calculate realized adverse selection
        // AS = price_move_against_fill / fill_price
        let fill_price = pred.fill.price;
        let current_mid = self.last_mid;

        let price_move = if pred.fill.is_buy {
            // For buys, AS is positive if price went down (we bought high)
            (fill_price - current_mid) / fill_price * 10000.0
        } else {
            // For sells, AS is positive if price went up (we sold low)
            (current_mid - fill_price) / fill_price * 10000.0
        };

        let realized_as_bps = price_move.max(0.0);

        // Calculate realized edge
        // Note: depth_bps() returns basis points, spread_capture() returns dollars
        let spread_captured_bps = pred.fill.depth_bps();
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

    /// Get the Kelly win/loss tracker for cloning to strategy.
    pub fn kelly_tracker(&self) -> &WinLossTracker {
        &self.kelly_tracker
    }

    /// Get the risk model config for consistency with strategy.
    pub fn risk_model_config(&self) -> &RiskModelConfig {
        &self.risk_model_config
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
        } else {
            if prediction.mean > 0.0 {
                1.0
            } else {
                0.0
            }
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
        assert_eq!(config.prediction_horizon_ms, 1000);
        assert_eq!(config.min_predictions_for_update, 20);
        assert_eq!(config.fee_bps, 1.5);
    }

    #[test]
    fn test_learning_module_creation() {
        let module = LearningModule::default();
        assert!(module.is_enabled());
        assert_eq!(module.pending_predictions_count(), 0);
    }
}
