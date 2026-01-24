//! Layer 3: Stochastic Controller for optimal sequential decision-making.
//!
//! This module sits on top of the LearningModule (Layer 2) and adds:
//! - BeliefState: Proper Bayesian posteriors over latent parameters
//! - ValueFunction: Basis function approximation for V(s)
//! - OptimalController: Q-value maximization over actions
//! - ChangepointDetector: BOCD for regime shift detection
//! - InformationValue: When to wait vs act
//!
//! ## Architecture
//!
//! ```text
//! Layer 1: ParameterEstimator → σ, κ, microprice
//!     ↓
//! Layer 2: LearningModule → edge predictions, model health, calibration
//!     ↓
//! Layer 3: StochasticController → optimal sequential decisions
//!     ↓
//! Layer 4: Execution → order management
//! ```
//!
//! ## Key Insight
//!
//! Layer 2 is myopic - it optimizes for immediate expected value.
//! Layer 3 considers future consequences:
//! - Terminal conditions (session end, funding)
//! - Information value (wait to learn)
//! - Regime changes (distrust learned models)
//! - Position management (sequential effects)

pub mod actions;
pub mod belief;
pub mod changepoint;
pub mod controller;
pub mod information;
pub mod interface;
pub mod quote_gate;
pub mod simulation;
pub mod state;
pub mod traits;
pub mod types;
pub mod value;

// Re-export key types
pub use actions::{Action, ActionConfig, DefensiveReason, NoQuoteReason};
pub use belief::BeliefState;
pub use changepoint::{ChangepointConfig, ChangepointDetector};
pub use quote_gate::{
    NoQuoteReason as QuoteGateNoQuoteReason, QuoteDecision as QuoteGateDecision, QuoteGate,
    QuoteGateConfig, QuoteGateInput,
};
pub use controller::{ControllerConfig, OptimalController};
pub use information::{InformationConfig, InformationValue};
pub use interface::{GaussianEstimate, LearningModuleOutput, ModelPrediction, TradingState};
pub use simulation::{
    CascadeScenario, HistoricalReplay, MarketScenario, MeanRevertingScenario, MonteCarloResult,
    SimulationConfig, SimulationEngine, SimulationResult, TrendingScenario,
};
pub use state::{ControlState, StateConfig};
pub use traits::{
    BeliefProvider, ControlOutput, ControlReason, ControlSolver, ControlStateProvider,
    MarketMicrostructure, ObservableState, StateSnapshot, ValueFunctionSolver,
};
pub use types::{DirichletPosterior, DiscreteDistribution, GammaPosterior, NormalGammaPosterior};

use crate::market_maker::fills::FillEvent;
use crate::market_maker::learning::{Health, QuoteDecision};
use crate::market_maker::quoting::Ladder;
use tracing::{debug, info, warn};

/// Configuration for the stochastic controller.
#[derive(Debug, Clone)]
pub struct StochasticControllerConfig {
    /// Whether the controller is enabled
    pub enabled: bool,
    /// Controller configuration
    pub controller: ControllerConfig,
    /// Changepoint detection configuration
    pub changepoint: ChangepointConfig,
    /// Information value configuration
    pub information: InformationConfig,
    /// State configuration
    pub state: StateConfig,
    /// Action configuration
    pub action: ActionConfig,
    /// Logging interval (quote cycles)
    pub log_interval: usize,
}

impl Default for StochasticControllerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            controller: ControllerConfig::default(),
            changepoint: ChangepointConfig::default(),
            information: InformationConfig::default(),
            state: StateConfig::default(),
            action: ActionConfig::default(),
            log_interval: 100,
        }
    }
}

/// The stochastic controller (Layer 3).
///
/// Orchestrates optimal sequential decision-making on top of the learning module.
#[derive(Debug)]
pub struct StochasticController {
    /// Configuration
    config: StochasticControllerConfig,

    /// Belief state
    belief: BeliefState,

    /// Optimal controller (value function + Q-value computation)
    controller: OptimalController,

    /// Changepoint detector
    changepoint: ChangepointDetector,

    /// Information value calculator
    info_value: InformationValue,

    /// Current learning trust level [0, 1]
    learning_trust: f64,

    /// Quote cycle counter for logging
    cycle_count: usize,

    /// Last action taken
    last_action: Option<Action>,

    /// Consecutive wait cycles
    wait_cycles: u32,

    /// Previous control state for TD learning
    /// Stored when an action is taken, used when fill arrives
    prev_state: Option<ControlState>,

    /// Previous action taken (for TD learning)
    prev_action_taken: Option<state::ActionTaken>,
}

impl Default for StochasticController {
    fn default() -> Self {
        Self::new(StochasticControllerConfig::default())
    }
}

impl StochasticController {
    /// Create a new stochastic controller.
    pub fn new(config: StochasticControllerConfig) -> Self {
        Self {
            controller: OptimalController::new(config.controller.clone()),
            changepoint: ChangepointDetector::new(config.changepoint.clone()),
            info_value: InformationValue::new(config.information.clone()),
            belief: BeliefState::default(),
            learning_trust: 1.0,
            cycle_count: 0,
            last_action: None,
            wait_cycles: 0,
            prev_state: None,
            prev_action_taken: None,
            config,
        }
    }

    /// Check if controller is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Main entry point - called each quote cycle.
    ///
    /// Takes learning module output and trading state, returns optimal action.
    pub fn act(
        &mut self,
        learning_output: &LearningModuleOutput,
        trading_state: &TradingState,
    ) -> Action {
        if !self.config.enabled {
            return self.myopic_to_action(&learning_output.myopic_decision);
        }

        self.cycle_count += 1;

        // 1. Update belief state from learning module output
        self.update_beliefs_from_learning(learning_output);

        // 2. Check for changepoint (regime shift)
        self.update_changepoint(learning_output);
        let trust_learning = self.assess_learning_trust(learning_output);
        self.learning_trust = trust_learning;

        // 3. Build control state
        let control_state = ControlState {
            wealth: trading_state.wealth,
            position: trading_state.position,
            margin_used: trading_state.margin_used,
            time: trading_state.session_time,
            belief: self.belief.clone(),
            vol_regime: DiscreteDistribution::default(),
            time_to_funding: trading_state.time_to_funding,
            predicted_funding: trading_state.predicted_funding,
            learning_trust: trust_learning,
            model_health: learning_output.model_health.clone(),
            reduce_only: trading_state.reduce_only,
            drawdown: trading_state.drawdown,
        };

        // 4. Compute optimal action
        let (optimal_action, expected_value) = self.controller.optimal_action(&control_state);

        // 5. Compare to myopic action from Layer 2
        let action = self.controller.reconcile_with_myopic(
            optimal_action,
            &learning_output.myopic_decision,
            &control_state,
        );

        // 6. Information value check: should we wait?
        // Skip WaitToLearn during L3 warmup - we need data first before making wait decisions
        // Using myopic actions during warmup allows position building and calibration
        let should_check_wait = self.changepoint.is_warmed_up();

        if should_check_wait && self.info_value.should_wait(&control_state, &action) {
            self.wait_cycles += 1;

            if self.wait_cycles <= self.config.information.max_wait_cycles {
                let info_gain = self.info_value.expected_info_gain(&control_state);
                debug!(
                    wait_cycles = self.wait_cycles,
                    info_gain = %format!("{:.4}", info_gain),
                    uncertainty = %format!("{:.2}", control_state.edge_uncertainty()),
                    "Waiting to learn"
                );

                return Action::WaitToLearn {
                    expected_info_gain: info_gain,
                    suggested_wait_cycles: self.info_value.recommended_wait_cycles(&control_state),
                };
            }
        } else if should_check_wait {
            self.wait_cycles = 0;
        }

        // 7. Log periodically
        if self.should_log() {
            self.log_state(&control_state, &action, expected_value);
        }

        // 8. Store state for TD learning on fill
        // Convert action to ActionTaken for value function learning
        let action_taken = match &action {
            Action::Quote { ladder, .. } => {
                // Extract spread from level 0 if available (first bid + first ask)
                let spread_bps = ladder
                    .bids
                    .first()
                    .zip(ladder.asks.first())
                    .map(|(bid, ask)| bid.depth_bps + ask.depth_bps)
                    .unwrap_or(8.0);
                let size = ladder.bids.first().map(|b| b.size).unwrap_or(0.0)
                    + ladder.asks.first().map(|a| a.size).unwrap_or(0.0);
                state::ActionTaken::Quoted { spread_bps, size }
            }
            Action::DefensiveQuote {
                spread_multiplier,
                size_fraction,
                ..
            } => {
                state::ActionTaken::Quoted {
                    spread_bps: 8.0 * spread_multiplier, // Base spread * multiplier
                    size: *size_fraction,
                }
            }
            Action::DumpInventory {
                target_position, ..
            } => state::ActionTaken::DumpedInventory {
                amount: *target_position,
            },
            Action::BuildInventory { .. } => state::ActionTaken::Quoted {
                spread_bps: 8.0,
                size: 1.0,
            },
            Action::NoQuote { .. } | Action::WaitToLearn { .. } => state::ActionTaken::NoQuote,
        };
        self.prev_state = Some(control_state);
        self.prev_action_taken = Some(action_taken);

        self.last_action = Some(action.clone());
        action
    }

    /// Handle a fill event.
    ///
    /// Updates beliefs and performs TD(0) value function learning.
    pub fn on_fill(
        &mut self,
        fill: &FillEvent,
        learning_output: &LearningModuleOutput,
        realized_as_bps: f64,
    ) {
        if !self.config.enabled {
            return;
        }

        // Calculate realized edge (reward for TD learning)
        let spread_captured = fill.spread_capture();
        let fee_bps = 1.5; // Should come from config
        let realized_edge = spread_captured - realized_as_bps - fee_bps;

        // Time since last update (approximate)
        let time_elapsed = 0.001; // 1ms placeholder

        // Update belief state
        let result =
            self.belief
                .update_with_diagnostics(realized_as_bps, realized_edge, time_elapsed);

        // Update changepoint detector with edge observation
        self.changepoint.update(realized_edge);

        // TD(0) Value Function Learning
        // If we have a previous state, create transition and update
        if let (Some(prev_state), Some(action_taken)) =
            (self.prev_state.take(), self.prev_action_taken.take())
        {
            // Build current control state (after fill)
            let current_state = ControlState {
                wealth: prev_state.wealth + realized_edge, // Updated wealth
                position: prev_state.position + if fill.is_buy { fill.size } else { -fill.size },
                time: prev_state.time, // Time is updated externally
                margin_used: prev_state.margin_used,
                drawdown: prev_state.drawdown,
                time_to_funding: prev_state.time_to_funding,
                predicted_funding: prev_state.predicted_funding, // Carry forward
                vol_regime: prev_state.vol_regime.clone(),       // Carry forward
                // Convert Health enum to trust score [0, 1]
                learning_trust: match learning_output.model_health.overall {
                    Health::Good => 1.0,
                    Health::Warning => 0.5,
                    Health::Degraded => 0.1,
                },
                reduce_only: prev_state.reduce_only,
                belief: self.belief.clone(),
                model_health: learning_output.model_health.clone(),
            };

            // Create state transition for TD learning
            let transition = state::StateTransition {
                from: prev_state,
                to: current_state,
                action_taken,
                reward: realized_edge, // Realized edge is our reward signal
            };

            // Update value function with TD(0)
            self.controller.update_value_function(&transition);

            debug!(
                reward = %format!("{:.4}", realized_edge),
                n_updates = self.controller.value_function_updates(),
                "TD(0): Updated value function from fill"
            );
        }

        if result.significant {
            debug!(
                prior_edge = %format!("{:.2}", result.prior_edge),
                posterior_edge = %format!("{:.2}", result.posterior_edge),
                info_gain = %format!("{:.4}", result.info_gain),
                "Significant belief update from fill"
            );
        }
    }

    /// Update beliefs from learning module output.
    fn update_beliefs_from_learning(&mut self, output: &LearningModuleOutput) {
        self.belief.update_from_learning(output);

        // Update information value tracker
        let control_state = ControlState {
            belief: self.belief.clone(),
            ..Default::default()
        };
        self.info_value.update(&control_state);
    }

    /// Update changepoint detection.
    fn update_changepoint(&mut self, output: &LearningModuleOutput) {
        // Use edge prediction as observation
        let obs = output.edge_prediction.mean;
        self.changepoint.update(obs);

        // Check if we should reset beliefs
        if self.changepoint.should_reset_beliefs() {
            warn!("Changepoint detected - resetting beliefs");
            self.belief.soft_reset(0.3); // Keep 30% of learned information
        }
    }

    /// Assess how much to trust the learning module.
    ///
    /// Uses entropy-based trust scaling for principled warmup behavior:
    /// - During warmup: trust scales with inverse entropy (more concentrated = more trust)
    /// - After warmup: trust based on changepoint probability
    ///
    /// First-principles: Trust should reflect how much we believe the learned
    /// parameters are relevant to the current regime. Entropy measures how
    /// confident BOCD is about the run length (regime age).
    fn assess_learning_trust(&self, output: &LearningModuleOutput) -> f64 {
        // Use entropy for principled warmup scaling
        let entropy = self.changepoint.run_length_entropy();
        let max_entropy = (self.changepoint.max_run_length() as f64).ln();

        // Normalized entropy ∈ [0, 1]
        // 0 = maximally concentrated (high confidence in run length)
        // 1 = maximally spread (no idea about run length)
        let normalized_entropy = (entropy / max_entropy).clamp(0.0, 1.0);

        let changepoint_factor = if !self.changepoint.is_warmed_up() {
            // During warmup: scale from 0.5 to 1.0 as entropy decreases
            // Low entropy = distribution concentrating = can trust more
            // High entropy = still uncertain = use baseline trust
            0.5 + 0.5 * (1.0 - normalized_entropy)
        } else {
            // Post-warmup: use changepoint probability for regime detection
            let changepoint_prob = self.changepoint.changepoint_probability(5);
            if changepoint_prob > 0.5 {
                // Recent changepoint detected - distrust learning module
                // Scale trust inversely with changepoint probability
                // Floor of 0.1 prevents complete paralysis
                (0.5 * (1.0 - changepoint_prob)).max(0.1)
            } else {
                // Stable regime - trust based on inverse probability
                1.0 - changepoint_prob
            }
        };

        // Model health
        let health_trust = match output.model_health.overall {
            Health::Good => 1.0,
            Health::Warning => 0.7,
            Health::Degraded => 0.3,
        };

        // AS bias check
        let as_trust = if output.as_bias > 1.0 {
            0.5 // Underestimating AS is dangerous
        } else {
            1.0
        };

        // Combine factors multiplicatively
        health_trust * as_trust * changepoint_factor
    }

    /// Convert myopic decision to action.
    fn myopic_to_action(&self, decision: &QuoteDecision) -> Action {
        match decision {
            QuoteDecision::Quote {
                expected_edge,
                size_fraction: _,
                ..
            } => Action::Quote {
                ladder: Ladder::default(),
                expected_value: *expected_edge,
            },
            QuoteDecision::ReducedSize {
                fraction,
                reason: _,
            } => Action::DefensiveQuote {
                spread_multiplier: 1.0,
                size_fraction: *fraction,
                reason: DefensiveReason::ModelDisagreement,
            },
            QuoteDecision::NoQuote { reason: _ } => Action::NoQuote {
                reason: NoQuoteReason::NegativeEdge,
            },
        }
    }

    /// Check if we should log this cycle.
    fn should_log(&self) -> bool {
        self.config.log_interval > 0 && self.cycle_count.is_multiple_of(self.config.log_interval)
    }

    /// Log current state.
    fn log_state(&self, state: &ControlState, action: &Action, expected_value: f64) {
        let cp_summary = self.changepoint.summary();

        info!(
            cycle = self.cycle_count,
            position = %format!("{:.4}", state.position),
            time = %format!("{:.3}", state.time),
            expected_edge = %format!("{:.2}bp", state.expected_edge()),
            uncertainty = %format!("{:.2}bp", state.edge_uncertainty()),
            confidence = %format!("{:.2}", state.confidence()),
            learning_trust = %format!("{:.2}", self.learning_trust),
            cp_prob = %format!("{:.3}", cp_summary.cp_prob_5),
            expected_value = %format!("{:.4}", expected_value),
            action = ?action,
            "Stochastic controller state"
        );

        // Log detailed changepoint state for diagnostics
        info!(
            target: "layer3::changepoint",
            p_now = %format!("{:.4}", cp_summary.cp_prob_1),
            p_5 = %format!("{:.4}", cp_summary.cp_prob_5),
            p_10 = %format!("{:.4}", cp_summary.cp_prob_10),
            run_length = cp_summary.most_likely_run,
            entropy = %format!("{:.4}", cp_summary.entropy),
            detected = cp_summary.detected,
            "[Changepoint] BOCD state"
        );
    }

    /// Get current belief state.
    pub fn belief(&self) -> &BeliefState {
        &self.belief
    }

    /// Get current learning trust.
    pub fn learning_trust(&self) -> f64 {
        self.learning_trust
    }

    /// Get changepoint summary.
    pub fn changepoint_summary(&self) -> changepoint::ChangepointSummary {
        self.changepoint.summary()
    }

    /// Get last action taken.
    pub fn last_action(&self) -> Option<&Action> {
        self.last_action.as_ref()
    }

    /// Reset the controller.
    pub fn reset(&mut self) {
        self.belief = BeliefState::default();
        self.changepoint.reset();
        self.learning_trust = 1.0;
        self.cycle_count = 0;
        self.wait_cycles = 0;
        self.last_action = None;
    }

    /// Generate comprehensive system health report.
    ///
    /// Shows the full Layer 1→L2→L3 pipeline state for diagnostics.
    pub fn system_health_report(
        &self,
        learning_output: &LearningModuleOutput,
    ) -> SystemHealthReport {
        let cp_summary = self.changepoint.summary();

        SystemHealthReport {
            // Layer 2 (LearningModule) outputs
            layer2_model_health: learning_output.model_health.clone(),
            layer2_quote_decision: format!("{:?}", learning_output.myopic_decision),
            layer2_edge_prediction: EdgePredictionReport {
                mean: learning_output.edge_prediction.mean,
                std: learning_output.edge_prediction.std,
                confidence: learning_output.p_positive_edge, // Use P(edge > 0) as confidence
            },

            // Layer 3 (StochasticController) outputs
            layer3_belief: BeliefReport {
                expected_edge: self.belief.expected_edge(),
                uncertainty: self.belief.total_edge_uncertainty,
                confidence: self.belief.confidence(),
                n_fills: self.belief.n_fills as u32,
            },
            layer3_changepoint: ChangepointReport {
                prob_5: cp_summary.cp_prob_5,
                prob_10: cp_summary.cp_prob_10,
                run_length: cp_summary.most_likely_run as u32,
            },
            layer3_value_function: ValueFunctionReport {
                n_updates: self.controller.value_function_updates(),
            },
            layer3_learning_trust: self.learning_trust,
            layer3_cycle_count: self.cycle_count,
            layer3_last_action: self.last_action.as_ref().map(|a| format!("{:?}", a)),

            // Meta
            overall_health: self.assess_overall_health(learning_output),
        }
    }

    /// Assess overall system health.
    fn assess_overall_health(&self, output: &LearningModuleOutput) -> OverallHealth {
        let cp_prob = self.changepoint.changepoint_probability(5);

        // Critical thresholds
        if cp_prob > 0.8 {
            return OverallHealth::Critical("Regime change detected".to_string());
        }
        if matches!(output.model_health.overall, Health::Degraded) {
            return OverallHealth::Critical("Model health degraded".to_string());
        }

        // Warning thresholds
        if cp_prob > 0.5 {
            return OverallHealth::Warning("Elevated changepoint probability".to_string());
        }
        if matches!(output.model_health.overall, Health::Warning) {
            return OverallHealth::Warning("Model health warning".to_string());
        }
        if self.learning_trust < 0.5 {
            return OverallHealth::Warning("Low learning trust".to_string());
        }

        OverallHealth::Good
    }

    /// Log comprehensive health report.
    pub fn log_health_report(&self, learning_output: &LearningModuleOutput) {
        let report = self.system_health_report(learning_output);

        info!(
            target: "layer3::health",
            // Layer 2 summary
            l2_edge_mean = %format!("{:.2}bp", report.layer2_edge_prediction.mean),
            l2_edge_std = %format!("{:.2}bp", report.layer2_edge_prediction.std),
            l2_decision = %report.layer2_quote_decision,

            // Layer 3 belief
            l3_belief_edge = %format!("{:.2}bp", report.layer3_belief.expected_edge),
            l3_belief_conf = %format!("{:.2}", report.layer3_belief.confidence),
            l3_n_fills = report.layer3_belief.n_fills,

            // Layer 3 meta-learning
            l3_cp_prob = %format!("{:.3}", report.layer3_changepoint.prob_5),
            l3_vf_updates = report.layer3_value_function.n_updates,
            l3_trust = %format!("{:.2}", report.layer3_learning_trust),

            // Overall
            overall = ?report.overall_health,
            "System health report"
        );
    }
}

/// Comprehensive system health report for Layer1→L2→L3 diagnostics.
#[derive(Debug, Clone)]
pub struct SystemHealthReport {
    // Layer 2 outputs
    pub layer2_model_health: crate::market_maker::learning::types::ModelHealth,
    pub layer2_quote_decision: String,
    pub layer2_edge_prediction: EdgePredictionReport,

    // Layer 3 outputs
    pub layer3_belief: BeliefReport,
    pub layer3_changepoint: ChangepointReport,
    pub layer3_value_function: ValueFunctionReport,
    pub layer3_learning_trust: f64,
    pub layer3_cycle_count: usize,
    pub layer3_last_action: Option<String>,

    // Overall assessment
    pub overall_health: OverallHealth,
}

/// Edge prediction summary.
#[derive(Debug, Clone)]
pub struct EdgePredictionReport {
    pub mean: f64,
    pub std: f64,
    pub confidence: f64,
}

/// Belief state summary.
#[derive(Debug, Clone)]
pub struct BeliefReport {
    pub expected_edge: f64,
    pub uncertainty: f64,
    pub confidence: f64,
    pub n_fills: u32,
}

/// Changepoint detector summary.
#[derive(Debug, Clone)]
pub struct ChangepointReport {
    pub prob_5: f64,
    pub prob_10: f64,
    pub run_length: u32,
}

/// Value function summary.
#[derive(Debug, Clone)]
pub struct ValueFunctionReport {
    pub n_updates: usize,
}

/// Overall system health status.
#[derive(Debug, Clone)]
pub enum OverallHealth {
    Good,
    Warning(String),
    Critical(String),
}

/// Result of applying the stochastic controller.
#[derive(Debug, Clone)]
pub struct ControllerResult {
    /// The action to take
    pub action: Action,
    /// Expected value of the action
    pub expected_value: f64,
    /// Whether Layer 3 overrode Layer 2
    pub overrode_myopic: bool,
    /// Reason for override (if any)
    pub override_reason: Option<String>,
}

impl StochasticController {
    /// Get detailed result with diagnostics.
    pub fn act_with_details(
        &mut self,
        learning_output: &LearningModuleOutput,
        trading_state: &TradingState,
    ) -> ControllerResult {
        let myopic_action = self.myopic_to_action(&learning_output.myopic_decision);
        let action = self.act(learning_output, trading_state);

        let (overrode, reason) = self.check_override(&myopic_action, &action);

        ControllerResult {
            action,
            expected_value: 0.0, // Would come from Q-value
            overrode_myopic: overrode,
            override_reason: reason,
        }
    }

    /// Check if we overrode the myopic decision.
    fn check_override(&self, myopic: &Action, actual: &Action) -> (bool, Option<String>) {
        // Compare action types
        let myopic_type = std::mem::discriminant(myopic);
        let actual_type = std::mem::discriminant(actual);

        if myopic_type != actual_type {
            let reason = match actual {
                Action::DumpInventory { .. } => "terminal/position management",
                Action::BuildInventory { .. } => "funding capture",
                Action::WaitToLearn { .. } => "information value",
                Action::DefensiveQuote { .. } => "uncertainty/risk",
                Action::NoQuote { .. } => "no quoting condition",
                Action::Quote { .. } => "optimal quote",
            };
            (true, Some(reason.to_string()))
        } else {
            (false, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_controller_creation() {
        let controller = StochasticController::default();
        assert!(controller.is_enabled());
        assert_eq!(controller.cycle_count, 0);
    }

    #[test]
    fn test_disabled_controller() {
        let config = StochasticControllerConfig {
            enabled: false,
            ..Default::default()
        };
        let mut controller = StochasticController::new(config);

        let output = LearningModuleOutput::default();
        let state = TradingState::default();

        let action = controller.act(&output, &state);

        // Should just return myopic action
        assert!(matches!(action, Action::NoQuote { .. }));
    }

    #[test]
    fn test_learning_trust_assessment() {
        let mut controller = StochasticController::default();

        // Feed some observations to the changepoint detector so it's not in "fresh" state
        // Fresh state has 100% changepoint probability which gives 0 trust
        for i in 0..10 {
            controller.changepoint.update(i as f64 * 0.1);
        }

        let mut output = LearningModuleOutput::default();
        output.model_health.overall = Health::Good;
        output.as_bias = 0.0;

        let trust = controller.assess_learning_trust(&output);
        // With good health and some observations, trust should be reasonable
        assert!(trust > 0.5);

        output.model_health.overall = Health::Degraded;
        let trust = controller.assess_learning_trust(&output);
        assert!(trust < 0.5);
    }

    #[test]
    fn test_reset() {
        let mut controller = StochasticController::default();

        // Simulate some activity
        controller.cycle_count = 100;
        controller.learning_trust = 0.5;

        controller.reset();

        assert_eq!(controller.cycle_count, 0);
        assert!((controller.learning_trust - 1.0).abs() < 1e-10);
    }
}
