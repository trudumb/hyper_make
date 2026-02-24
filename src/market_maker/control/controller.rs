//! Optimal controller using Q-value maximization.
//!
//! The controller implements the core optimization:
//!   a* = argmax_a Q(s, a)
//! where Q(s, a) = r(s, a) + γ E[V(s') | s, a]

use super::actions::{Action, ActionConfig, NoQuoteReason};
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
            Action::Quote { expected_value, .. } => {
                // Use realized edge from last fill as the reward signal when available.
                // This grounds the TD learning in actual P&L rather than synthetic
                // expected_value estimates that may be miscalibrated.
                // Fallback to expected_value when no fill has been observed yet.
                if state.last_realized_edge_bps != 0.0 {
                    // Convert bps to fraction for consistency with other rewards
                    state.last_realized_edge_bps / 10000.0
                } else {
                    *expected_value
                }
            }

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
                // Negative reward = spread crossing cost scaled by urgency.
                // Uses actual market spread instead of arbitrary -0.05 constant.
                // The spread cost is the price of immediacy when dumping.
                let spread_cost = if state.market_spread_bps > 0.0 {
                    state.market_spread_bps / 10000.0
                } else {
                    0.0005 // Fallback: 5 bps if no market data
                };
                -spread_cost * urgency
            }

            Action::BuildInventory { target, .. } => {
                // Reward for positioning towards funding
                let funding_value = state.predicted_funding.abs() * 0.01;
                let position_delta = (target - state.position).abs();
                funding_value - 0.01 * position_delta
            }

            Action::WaitToLearn { .. } => {
                // Zero immediate reward: the true opportunity cost of waiting
                // is captured by the discount factor in the TD update.
                // No arbitrary constants needed.
                0.0
            }
        }
    }

    /// Convert action to outcome for value computation.
    fn action_to_outcome(&self, action: &Action, state: &ControlState) -> ActionOutcome {
        match action {
            Action::Quote { .. } => {
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
        // NOTE: DefensiveQuote has been removed. All uncertainty is now handled
        // through gamma scaling (kappa_ci_width flows through uncertainty_scalar).
        // The GLFT formula naturally widens spreads when gamma increases.
        if state.can_quote() {
            candidates.push(Action::Quote {
                ladder: Box::new(Ladder::default()),
                expected_value: state.expected_edge() * 0.1, // Simple estimate
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

        // 6. WaitToLearn removed: cold-start deadlock — can only learn by quoting.
        // The Action::WaitToLearn variant is kept for checkpoint backward compat but never generated.

        candidates
    }

    /// Reconcile optimal action with Layer 2's myopic decision.
    pub fn reconcile_with_myopic(
        &self,
        _optimal: Action,
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

        // 4. WaitToLearn removed: cold-start deadlock — can only learn by quoting.

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
    ///
    /// NOTE: We now use Quote with full size instead of DefensiveQuote.
    /// Uncertainty is handled through gamma scaling (kappa_ci_width → uncertainty_scalar).
    /// The GLFT formula naturally widens spreads when gamma increases due to uncertainty.
    fn conservative_action(&self, myopic: &QuoteDecision, state: &ControlState) -> Action {
        match myopic {
            QuoteDecision::Quote { expected_edge, .. } => {
                // Quote with full size - gamma already handles uncertainty
                Action::Quote {
                    ladder: Box::new(Ladder::default()),
                    expected_value: *expected_edge,
                }
            }
            QuoteDecision::ReducedSize { .. } => {
                // Quote with full size - gamma handles the risk
                Action::Quote {
                    ladder: Box::new(Ladder::default()),
                    expected_value: state.expected_edge(),
                }
            }
            QuoteDecision::NoQuote { reason: _ } => Action::NoQuote {
                reason: NoQuoteReason::ModelDegraded,
            },
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
    /// Action constrained by position limits.
    ///
    /// NOTE: We use Quote instead of DefensiveQuote. Position sizing is handled
    /// by the GLFT strategy's cascade_size_factor and inventory scaling, not
    /// by arbitrary size_fraction multipliers in the controller.
    ///
    /// CRITICAL: When over position limit, we MUST still quote reduce-only to work
    /// off the position. The strategy layer (GLFT/ladder) handles this by:
    /// - inventory_scalar increasing gamma to widen spreads
    /// - Only quoting the side that reduces position
    fn position_constrained_action(&self, myopic: &QuoteDecision, state: &ControlState) -> Action {
        // Calculate how much room we have (negative = over limit)
        let room = self.config.max_position * 0.95 - state.position.abs();
        let max_fraction = room / self.config.max_position;

        // ALWAYS allow quoting - the strategy layer handles position limits via:
        // 1. inventory_scalar increases gamma dramatically when at/over limit
        // 2. Skew pushes quotes to reduce position (aggressive on reduce side)
        // 3. reduce_only mode in ladder_strat blocks quotes that would increase position
        //
        // Previous bug: Blocking ALL quotes when over limit left position stuck!
        // The right behavior is to let the strategy quote reduce-only.
        match myopic {
            QuoteDecision::Quote { expected_edge, .. } => {
                // Log if over limit - strategy will handle with reduce-only
                if max_fraction < 0.0 {
                    tracing::debug!(
                        position = %state.position,
                        max_position = %self.config.max_position,
                        room = %room,
                        "Over position limit - strategy will quote reduce-only"
                    );
                }
                Action::Quote {
                    ladder: Box::new(Ladder::default()),
                    expected_value: *expected_edge,
                }
            }
            QuoteDecision::ReducedSize { .. } => Action::Quote {
                ladder: Box::new(Ladder::default()),
                expected_value: state.expected_edge(),
            },
            _ => Action::NoQuote {
                reason: NoQuoteReason::RiskLimit,
            },
        }
    }

    /// Convert myopic decision to action.
    ///
    /// NOTE: Both Quote and ReducedSize now map to Action::Quote.
    /// Size reduction is handled by GLFT's inventory_scalar and cascade_size_factor,
    /// not by arbitrary size_fraction values.
    fn myopic_to_action(&self, myopic: &QuoteDecision, state: &ControlState) -> Action {
        match myopic {
            QuoteDecision::Quote { expected_edge, .. } => Action::Quote {
                ladder: Box::new(Ladder::default()),
                expected_value: *expected_edge,
            },
            QuoteDecision::ReducedSize { .. } => {
                // ReducedSize is converted to Quote - gamma handles the risk
                Action::Quote {
                    ladder: Box::new(Ladder::default()),
                    expected_value: state.expected_edge(),
                }
            }
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
                ModelHealth {
                    overall: crate::market_maker::learning::Health::Degraded,
                    ..Default::default()
                }
            } else {
                ModelHealth::default()
            },
            reduce_only: provider.reduce_only(),
            drawdown: provider.drawdown(),
            rate_limit_headroom: provider.rate_limit_headroom(),
            last_realized_edge_bps: 0.0, // Set from TradingState path, not provider
            market_spread_bps: 0.0,      // Set from TradingState path, not provider
            drift_rate: 0.0,
            ou_uncertainty: 0.0,
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

    #[test]
    fn test_quote_reward_uses_realized_edge() {
        let controller = OptimalController::default();
        let mut state = ControlState::default();
        // Set realized edge from last fill: +3 bps
        state.last_realized_edge_bps = 3.0;

        let action = Action::Quote {
            ladder: Box::new(crate::market_maker::quoting::Ladder::default()),
            expected_value: 0.1, // This should be ignored when realized edge is available
        };

        let reward = controller.immediate_reward(&state, &action);
        // Should use realized edge: 3 bps = 0.0003
        assert!(
            (reward - 0.0003).abs() < 1e-8,
            "Quote reward should use realized edge (0.0003), got {}",
            reward
        );
    }

    #[test]
    fn test_quote_reward_falls_back_to_expected_value() {
        let controller = OptimalController::default();
        let state = ControlState::default(); // last_realized_edge_bps = 0.0

        let action = Action::Quote {
            ladder: Box::new(crate::market_maker::quoting::Ladder::default()),
            expected_value: 0.05,
        };

        let reward = controller.immediate_reward(&state, &action);
        // No realized edge → should use expected_value
        assert!(
            (reward - 0.05).abs() < 1e-8,
            "Quote reward should fall back to expected_value (0.05), got {}",
            reward
        );
    }

    #[test]
    fn test_dump_inventory_reward_uses_market_spread() {
        let controller = OptimalController::default();
        let mut state = ControlState::default();
        state.position = 5.0;
        state.market_spread_bps = 10.0; // 10 bps spread

        let action = Action::DumpInventory {
            urgency: 1.0,
            target_position: 0.0,
        };

        let reward = controller.immediate_reward(&state, &action);
        // Should be: -(10/10000) * 1.0 = -0.001
        assert!(
            (reward - (-0.001)).abs() < 1e-8,
            "Dump reward should be -spread_cost * urgency = -0.001, got {}",
            reward
        );
    }

    #[test]
    fn test_dump_inventory_reward_scales_with_urgency() {
        let controller = OptimalController::default();
        let mut state = ControlState::default();
        state.position = 5.0;
        state.market_spread_bps = 10.0;

        let low_urgency = Action::DumpInventory {
            urgency: 0.5,
            target_position: 0.0,
        };
        let high_urgency = Action::DumpInventory {
            urgency: 2.0,
            target_position: 0.0,
        };

        let low_reward = controller.immediate_reward(&state, &low_urgency);
        let high_reward = controller.immediate_reward(&state, &high_urgency);

        assert!(
            high_reward < low_reward,
            "Higher urgency should be more costly: low={}, high={}",
            low_reward,
            high_reward
        );
    }

    #[test]
    fn test_wait_to_learn_reward_is_zero() {
        let controller = OptimalController::default();
        let state = ControlState::default();

        let action = Action::WaitToLearn {
            expected_info_gain: 0.5,
            suggested_wait_cycles: 3,
        };

        let reward = controller.immediate_reward(&state, &action);
        assert!(
            (reward - 0.0).abs() < 1e-10,
            "WaitToLearn reward should be exactly 0.0, got {}",
            reward
        );
    }
}
