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
pub mod state;
pub mod types;
pub mod value;

// Re-export key types
pub use actions::{Action, ActionConfig, DefensiveReason, NoQuoteReason};
pub use belief::BeliefState;
pub use changepoint::{ChangepointConfig, ChangepointDetector};
pub use controller::{ControllerConfig, OptimalController};
pub use information::{InformationConfig, InformationValue};
pub use interface::{GaussianEstimate, LearningModuleOutput, ModelPrediction, TradingState};
pub use state::{ControlState, StateConfig};
pub use types::{DirichletPosterior, DiscreteDistribution, GammaPosterior, NormalGammaPosterior};

use crate::market_maker::fills::FillEvent;
use crate::market_maker::learning::{Health, ModelHealth, QuoteDecision};
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
        if self.info_value.should_wait(&control_state, &action) {
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
        } else {
            self.wait_cycles = 0;
        }

        // 7. Log periodically
        if self.should_log() {
            self.log_state(&control_state, &action, expected_value);
        }

        self.last_action = Some(action.clone());
        action
    }

    /// Handle a fill event.
    pub fn on_fill(
        &mut self,
        fill: &FillEvent,
        learning_output: &LearningModuleOutput,
        realized_as_bps: f64,
    ) {
        if !self.config.enabled {
            return;
        }

        // Calculate realized edge
        let spread_captured = fill.spread_capture();
        let fee_bps = 1.5; // Should come from config
        let realized_edge = spread_captured - realized_as_bps - fee_bps;

        // Time since last update (approximate)
        let time_elapsed = 0.001; // 1ms placeholder

        // Update belief state
        let result = self
            .belief
            .update_with_diagnostics(realized_as_bps, realized_edge, time_elapsed);

        // Update changepoint detector with edge observation
        self.changepoint.update(realized_edge);

        // Update controller value function
        // (Simplified - in production would track full state transitions)

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
    fn assess_learning_trust(&self, output: &LearningModuleOutput) -> f64 {
        // Changepoint detection
        let changepoint_prob = self.changepoint.changepoint_probability(5);
        if changepoint_prob > 0.5 {
            // Recent changepoint - distrust learning module
            return 0.5 * (1.0 - changepoint_prob);
        }

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

        // Combine factors
        health_trust * as_trust * (1.0 - changepoint_prob)
    }

    /// Convert myopic decision to action.
    fn myopic_to_action(&self, decision: &QuoteDecision) -> Action {
        match decision {
            QuoteDecision::Quote {
                expected_edge,
                size_fraction,
                ..
            } => Action::Quote {
                ladder: Ladder::default(),
                expected_value: *expected_edge,
            },
            QuoteDecision::ReducedSize { fraction, reason } => Action::DefensiveQuote {
                spread_multiplier: 1.0,
                size_fraction: *fraction,
                reason: DefensiveReason::ModelDisagreement,
            },
            QuoteDecision::NoQuote { reason } => Action::NoQuote {
                reason: NoQuoteReason::NegativeEdge,
            },
        }
    }

    /// Check if we should log this cycle.
    fn should_log(&self) -> bool {
        self.config.log_interval > 0 && self.cycle_count % self.config.log_interval == 0
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
