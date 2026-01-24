//! Optimal controller using Q-value maximization.
//!
//! The controller implements the core optimization:
//!   a* = argmax_a Q(s, a)
//! where Q(s, a) = r(s, a) + γ E[V(s') | s, a]

use super::actions::{Action, ActionConfig, DefensiveReason, NoQuoteReason};
use super::state::ControlState;
use super::value::{ActionOutcome, ExpectedValueComputer, ValueFunction};
use crate::market_maker::learning::QuoteDecision;
use crate::market_maker::quoting::Ladder;

/// Optimal controller for the stochastic control problem.
#[derive(Debug, Clone)]
pub struct OptimalController {
    /// Value function for computing expected future value
    pub value_fn: ValueFunction,
    /// Expected value computer for Monte Carlo integration
    expected_value: ExpectedValueComputer,
    /// Configuration
    config: ControllerConfig,
}

/// Controller configuration.
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Maximum position size
    pub max_position: f64,
    /// Survival probability threshold for quoting
    pub survival_prob_threshold: f64,
    /// Discount factor for future rewards
    pub gamma: f64,
    /// Terminal zone threshold (session time fraction)
    pub terminal_zone: f64,
    /// Funding capture threshold (bps)
    pub funding_threshold: f64,
    /// Maximum funding position fraction
    pub max_funding_fraction: f64,
    /// Action configuration
    pub action_config: ActionConfig,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            max_position: 10.0,
            survival_prob_threshold: 0.95,
            gamma: 0.99,
            terminal_zone: 0.95,
            funding_threshold: 5.0,
            max_funding_fraction: 0.5,
            action_config: ActionConfig::default(),
        }
    }
}

impl Default for OptimalController {
    fn default() -> Self {
        let value_fn = ValueFunction::default();
        Self {
            expected_value: ExpectedValueComputer::new(value_fn.clone()),
            value_fn,
            config: ControllerConfig::default(),
        }
    }
}

impl OptimalController {
    /// Create a new controller.
    pub fn new(config: ControllerConfig) -> Self {
        let value_fn = ValueFunction::default();
        Self {
            expected_value: ExpectedValueComputer::new(value_fn.clone()),
            value_fn,
            config,
        }
    }

    /// Compute optimal action for the current state.
    ///
    /// Returns (action, expected_value).
    pub fn optimal_action(&self, state: &ControlState) -> (Action, f64) {
        // Generate candidate actions
        let candidates = self.generate_candidate_actions(state);

        // Compute Q-value for each
        let mut best_action = Action::default();
        let mut best_q = f64::NEG_INFINITY;

        for action in candidates {
            let q = self.q_value(state, &action);
            if q > best_q {
                best_q = q;
                best_action = action;
            }
        }

        (best_action, best_q)
    }

    /// Compute Q(s, a) = r(s, a) + γ E[V(s') | s, a].
    pub fn q_value(&self, state: &ControlState, action: &Action) -> f64 {
        // Immediate reward
        let immediate_reward = self.immediate_reward(state, action);

        // Expected future value
        let action_outcome = self.action_to_outcome(action, state);
        let future_value = self
            .expected_value
            .expected_future_value(state, &action_outcome);

        immediate_reward + self.config.gamma * future_value
    }

    /// Compute immediate reward for taking action in state.
    fn immediate_reward(&self, state: &ControlState, action: &Action) -> f64 {
        match action {
            Action::Quote { expected_value, .. } => *expected_value,

            Action::NoQuote { reason } => {
                // Small negative reward for not acting (opportunity cost)
                let base_cost = -0.01;
                // But less if it's for a good reason
                let reason_bonus = match reason {
                    NoQuoteReason::HighUncertainty => 0.005,
                    NoQuoteReason::InformationValue => 0.005,
                    NoQuoteReason::ChangepointDetected => 0.003,
                    _ => 0.0,
                };
                base_cost + reason_bonus
            }

            Action::DumpInventory { urgency, .. } => {
                // Negative reward proportional to urgency (forced action is costly)
                let position_risk = state.position.powi(2) / (self.config.max_position.powi(2));
                -0.05 * urgency * (1.0 - position_risk) // Less costly if position is risky anyway
            }

            Action::BuildInventory { target, .. } => {
                // Reward for positioning towards funding
                let funding_value = state.predicted_funding.abs() * 0.01;
                let position_delta = (target - state.position).abs();
                funding_value - 0.01 * position_delta
            }

            Action::DefensiveQuote {
                spread_multiplier, ..
            } => {
                // Reduced expected value from wider spreads
                let edge = state.expected_edge();
                let fill_prob_reduction = 1.0 / spread_multiplier; // Wider spreads fill less
                edge * fill_prob_reduction * 0.5
            }

            Action::WaitToLearn {
                expected_info_gain, ..
            } => {
                // Value of information minus opportunity cost
                expected_info_gain * 0.1 - 0.02
            }
        }
    }

    /// Convert action to outcome for value computation.
    fn action_to_outcome(&self, action: &Action, state: &ControlState) -> ActionOutcome {
        match action {
            Action::Quote { .. } | Action::DefensiveQuote { .. } => {
                // Estimate fill probability from belief
                let fill_prob = self.estimate_fill_prob(state);
                let fill_size = 0.1; // Placeholder

                ActionOutcome::Quoted {
                    fill_prob,
                    fill_size,
                    fill_side: state.position < 0.0, // Buy if short, sell if long
                }
            }
            Action::DumpInventory {
                target_position, ..
            } => ActionOutcome::DumpedInventory {
                amount: (state.position - target_position).abs(),
            },
            Action::BuildInventory { target, .. } => ActionOutcome::Quoted {
                fill_prob: 0.8, // Aggressive building has high fill prob
                fill_size: (target - state.position).abs() * 0.1,
                fill_side: *target > state.position,
            },
            Action::NoQuote { .. } | Action::WaitToLearn { .. } => ActionOutcome::NoAction,
        }
    }

    /// Estimate fill probability from belief state.
    fn estimate_fill_prob(&self, state: &ControlState) -> f64 {
        // Use fill rate belief
        let lambda = state.belief.expected_fill_rate();

        // Convert to probability in one time step
        let dt = 0.001;
        1.0 - (-lambda * dt).exp()
    }

    /// Generate candidate actions to consider.
    fn generate_candidate_actions(&self, state: &ControlState) -> Vec<Action> {
        let mut candidates = Vec::new();

        // 1. Always consider not quoting
        candidates.push(Action::NoQuote {
            reason: NoQuoteReason::NegativeEdge,
        });

        // 2. Consider quoting if conditions allow
        if state.can_quote() {
            candidates.push(Action::Quote {
                ladder: Ladder::default(),
                expected_value: state.expected_edge() * 0.1, // Simple estimate
            });

            // 3. Consider defensive quoting
            candidates.push(Action::DefensiveQuote {
                spread_multiplier: 1.5,
                size_fraction: 0.5,
                reason: DefensiveReason::RegimeUncertainty,
            });
        }

        // 4. Consider inventory dump if position is large
        let position_fraction = state.abs_inventory() / self.config.max_position;
        if position_fraction > 0.5 {
            candidates.push(Action::DumpInventory {
                urgency: (position_fraction - 0.5) * 2.0 * state.urgency(),
                target_position: state.position * 0.5, // Reduce by half
            });
        }

        // 5. Consider funding positioning
        if state.funding_approaching(1.0)
            && state.predicted_funding.abs() > self.config.funding_threshold
        {
            // Build position opposite to funding direction
            let target = -state.predicted_funding.signum()
                * self.config.max_position
                * self.config.max_funding_fraction;

            candidates.push(Action::BuildInventory {
                target,
                aggressiveness: 0.5,
            });
        }

        // 6. Consider waiting if high uncertainty
        if state.edge_uncertainty() > state.expected_edge().abs() {
            candidates.push(Action::WaitToLearn {
                expected_info_gain: 0.1,
                suggested_wait_cycles: 5,
            });
        }

        candidates
    }

    /// Reconcile optimal action with Layer 2's myopic decision.
    pub fn reconcile_with_myopic(
        &self,
        optimal: Action,
        myopic: &QuoteDecision,
        state: &ControlState,
    ) -> Action {
        // Layer 3 overrides in specific situations:

        // 1. Terminal zone - force inventory reduction
        if state.is_terminal(self.config.terminal_zone) && state.abs_inventory() > 0.5 {
            return self.terminal_action(state);
        }

        // 2. Funding approaching - position for capture
        if state.funding_approaching(0.5) {
            if let Some(action) = self.funding_action(state) {
                return action;
            }
        }

        // 3. High model uncertainty - be more conservative
        if state.belief.epistemic_uncertainty > 0.5 {
            return self.conservative_action(myopic, state);
        }

        // 4. Information value - prefer waiting
        if matches!(optimal, Action::WaitToLearn { .. })
            && !matches!(myopic, QuoteDecision::NoQuote { .. })
        {
            // Check if waiting is actually valuable
            let wait_value = self.value_of_waiting(state);
            let act_value = self.value_of_acting(state, myopic);

            if wait_value > act_value {
                return optimal;
            }
        }

        // 5. Position limits - Layer 2 might not account for sequential effects
        if self.would_exceed_position(myopic, state) {
            return self.position_constrained_action(myopic, state);
        }

        // Otherwise, trust Layer 2 (convert myopic to action)
        self.myopic_to_action(myopic, state)
    }

    /// Terminal zone action: reduce inventory urgently.
    fn terminal_action(&self, state: &ControlState) -> Action {
        let time_remaining = state.time_remaining();
        let urgency =
            (1.0 / time_remaining.max(0.01)).min(self.config.action_config.max_dump_urgency);

        Action::DumpInventory {
            urgency,
            target_position: 0.0,
        }
    }

    /// Funding capture action.
    fn funding_action(&self, state: &ControlState) -> Option<Action> {
        let funding_rate = state.predicted_funding;

        // If funding is significant, position accordingly
        if funding_rate.abs() > self.config.funding_threshold {
            // Negative funding = longs pay shorts → we want to be short
            // Positive funding = shorts pay longs → we want to be long
            let target = -funding_rate.signum()
                * self.config.max_position
                * self.config.max_funding_fraction;

            // Only if we're not already positioned
            if (state.position - target).abs() > 0.1 {
                return Some(Action::BuildInventory {
                    target,
                    aggressiveness: 0.5,
                });
            }
        }

        None
    }

    /// Conservative action when models disagree.
    fn conservative_action(&self, myopic: &QuoteDecision, _state: &ControlState) -> Action {
        match myopic {
            QuoteDecision::Quote {
                size_fraction,
                expected_edge: _,
                ..
            } => {
                // Reduce size and widen spreads
                Action::DefensiveQuote {
                    spread_multiplier: 1.5,
                    size_fraction: size_fraction * 0.5,
                    reason: DefensiveReason::ModelDisagreement,
                }
            }
            QuoteDecision::ReducedSize { fraction, .. } => Action::DefensiveQuote {
                spread_multiplier: 1.3,
                size_fraction: fraction * 0.5,
                reason: DefensiveReason::ModelDisagreement,
            },
            QuoteDecision::NoQuote { reason: _ } => Action::NoQuote {
                reason: NoQuoteReason::ModelDegraded,
            },
        }
    }

    /// Value of waiting to learn.
    fn value_of_waiting(&self, state: &ControlState) -> f64 {
        // Expected reduction in uncertainty
        let current_uncertainty = state.edge_uncertainty();
        let expected_uncertainty = current_uncertainty * 0.9; // Assume 10% reduction

        // Value of uncertainty reduction
        let info_value = (current_uncertainty - expected_uncertainty) * state.expected_edge().abs();

        // Minus opportunity cost
        let opportunity_cost = state.expected_edge().max(0.0) * 0.01;

        info_value - opportunity_cost
    }

    /// Value of acting now.
    fn value_of_acting(&self, state: &ControlState, myopic: &QuoteDecision) -> f64 {
        match myopic {
            QuoteDecision::Quote { expected_edge, .. } => *expected_edge * 0.1,
            QuoteDecision::ReducedSize { fraction, .. } => state.expected_edge() * fraction * 0.1,
            QuoteDecision::NoQuote { .. } => 0.0,
        }
    }

    /// Check if myopic action would exceed position limits.
    fn would_exceed_position(&self, myopic: &QuoteDecision, state: &ControlState) -> bool {
        let expected_fill = match myopic {
            QuoteDecision::Quote { size_fraction, .. } => *size_fraction * 0.1,
            QuoteDecision::ReducedSize { fraction, .. } => *fraction * 0.1,
            QuoteDecision::NoQuote { .. } => 0.0,
        };

        (state.position.abs() + expected_fill) > self.config.max_position * 0.95
    }

    /// Action constrained by position limits.
    fn position_constrained_action(&self, myopic: &QuoteDecision, state: &ControlState) -> Action {
        // Calculate how much room we have
        let room = (self.config.max_position * 0.95 - state.position.abs()).max(0.0);
        let max_fraction = room / self.config.max_position;

        if max_fraction < 0.1 {
            // Too close to limit, don't quote
            Action::NoQuote {
                reason: NoQuoteReason::RiskLimit,
            }
        } else {
            // Reduce size proportionally
            match myopic {
                QuoteDecision::Quote {
                    expected_edge: _, ..
                } => Action::DefensiveQuote {
                    spread_multiplier: 1.0,
                    size_fraction: max_fraction,
                    reason: DefensiveReason::PositionLimitApproaching,
                },
                _ => Action::NoQuote {
                    reason: NoQuoteReason::RiskLimit,
                },
            }
        }
    }

    /// Convert myopic decision to action.
    fn myopic_to_action(&self, myopic: &QuoteDecision, _state: &ControlState) -> Action {
        match myopic {
            QuoteDecision::Quote {
                size_fraction: _,
                expected_edge,
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

    /// Update value function with observed transition.
    pub fn update(&mut self, from: ControlState, to: ControlState, reward: f64) {
        use super::state::{ActionTaken, StateTransition};

        let transition = StateTransition {
            from,
            to,
            action_taken: ActionTaken::NoQuote, // Placeholder
            reward,
        };

        self.value_fn.update_td(&transition);
    }

    /// Update value function with a full state transition (for TD(0) learning).
    ///
    /// This is called when a fill is observed to update the value function
    /// with the actual reward (realized edge).
    pub fn update_value_function(&mut self, transition: &super::state::StateTransition) {
        self.value_fn.update_td(transition);

        // Also sync the expected value computer's value function
        // (it maintains its own copy for Monte Carlo integration)
        self.expected_value = ExpectedValueComputer::new(self.value_fn.clone());
    }

    /// Get number of value function updates performed.
    pub fn value_function_updates(&self) -> usize {
        self.value_fn.n_updates as usize
    }

    /// Get current value function weights for diagnostics.
    pub fn value_function_weights(&self) -> &[f64] {
        self.value_fn.weights()
    }
}

// === Trait implementations for trait-based architecture ===

use super::traits::{ControlOutput, ControlSolver, ControlStateProvider, ValueFunctionSolver};
use crate::market_maker::strategy::MarketParams;

impl ControlSolver for OptimalController {
    fn solve(
        &self,
        state: &dyn ControlStateProvider,
        _market_params: &MarketParams,
    ) -> ControlOutput {
        // Convert trait object to ControlState for compatibility with existing logic
        // This creates a temporary ControlState from the provider
        let control_state = self.state_from_provider(state);
        let (action, expected_value) = self.optimal_action(&control_state);

        let confidence = state.overall_confidence();

        ControlOutput {
            action,
            expected_value,
            confidence,
        }
    }

    fn name(&self) -> &'static str {
        "OptimalController"
    }
}

impl ValueFunctionSolver for OptimalController {
    fn q_value(&self, state: &dyn ControlStateProvider, action: &Action) -> f64 {
        let control_state = self.state_from_provider(state);
        self.q_value(&control_state, action)
    }

    fn n_updates(&self) -> usize {
        self.value_function_updates()
    }
}

impl OptimalController {
    /// Create a ControlState from a ControlStateProvider.
    ///
    /// This is a helper for the trait-based interface to work with
    /// the existing concrete-type methods.
    fn state_from_provider(&self, provider: &dyn ControlStateProvider) -> ControlState {
        use super::belief::BeliefState;
        use super::types::DiscreteDistribution;
        use crate::market_maker::learning::ModelHealth;

        ControlState {
            wealth: provider.wealth(),
            position: provider.position(),
            margin_used: provider.margin_used(),
            time: provider.time(),
            belief: BeliefState::default(), // We use trait methods instead
            vol_regime: {
                let probs = provider.regime_probs();
                DiscreteDistribution { probs }
            },
            time_to_funding: provider.time_to_funding(),
            predicted_funding: provider.predicted_funding(),
            learning_trust: provider.learning_trust(),
            model_health: if provider.is_model_degraded() {
                let mut health = ModelHealth::default();
                health.overall = crate::market_maker::learning::Health::Degraded;
                health
            } else {
                ModelHealth::default()
            },
            reduce_only: provider.reduce_only(),
            drawdown: provider.drawdown(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_default() {
        let controller = OptimalController::default();
        let state = ControlState::default();

        let (action, _value) = controller.optimal_action(&state);
        // Should return some action
        assert!(!matches!(action, Action::DumpInventory { .. }));
    }

    #[test]
    fn test_terminal_action() {
        let controller = OptimalController::default();
        let mut state = ControlState::default();
        state.time = 0.98;
        state.position = 5.0;

        let action = controller.terminal_action(&state);
        assert!(matches!(action, Action::DumpInventory { .. }));
    }

    #[test]
    fn test_funding_action() {
        let controller = OptimalController::default();
        let mut state = ControlState::default();
        state.time_to_funding = 0.5;
        state.predicted_funding = 10.0;
        state.position = 0.0;

        let action = controller.funding_action(&state);
        assert!(action.is_some());

        if let Some(Action::BuildInventory { target, .. }) = action {
            assert!(target < 0.0); // Should go short to collect positive funding
        }
    }

    #[test]
    fn test_q_value() {
        let controller = OptimalController::default();
        let state = ControlState::default();
        let action = Action::NoQuote {
            reason: NoQuoteReason::NegativeEdge,
        };

        let q = controller.q_value(&state, &action);
        // Q-value should be finite
        assert!(q.is_finite());
    }
}
