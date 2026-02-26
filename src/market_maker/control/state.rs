//! Control state for the stochastic controller.
//!
//! The control state bundles all information needed for optimal decision-making:
//! observable state, belief state, and exogenous factors.

use super::belief::BeliefState;
use super::interface::TradingState;
use super::types::DiscreteDistribution;
use crate::market_maker::learning::ModelHealth;

/// Complete state for the stochastic control problem.
///
/// This is the state vector that the value function operates on.
#[derive(Debug, Clone)]
pub struct ControlState {
    // === Observable state ===
    /// Current wealth (cumulative P&L)
    pub wealth: f64,

    /// Current position
    pub position: f64,

    /// Margin currently used
    pub margin_used: f64,

    /// Session time as fraction [0, 1]
    pub time: f64,

    // === Belief state ===
    /// Full Bayesian belief state
    pub belief: BeliefState,

    // === Exogenous state ===
    /// Volatility regime distribution
    pub vol_regime: DiscreteDistribution<3>,

    /// Time to next funding (hours)
    pub time_to_funding: f64,

    /// Predicted funding rate (bps)
    pub predicted_funding: f64,

    // === Meta-state ===
    /// Trust in learning module [0, 1]
    pub learning_trust: f64,

    /// Model health from Layer 2
    pub model_health: ModelHealth,

    /// Whether reduce-only mode is active
    pub reduce_only: bool,

    /// Current drawdown
    pub drawdown: f64,

    /// Rate limit headroom fraction [0, 1]
    pub rate_limit_headroom: f64,

    /// Last realized edge from a fill (bps). Updated after each fill.
    /// Used as the TD reward signal for Quote actions instead of
    /// synthetic expected_value from the action itself.
    pub last_realized_edge_bps: f64,

    /// Current market spread (bps). Used to compute spread crossing
    /// cost for DumpInventory actions.
    pub market_spread_bps: f64,

    /// Current drift rate per second from OU process.
    /// Positive = price trending up, negative = trending down.
    pub drift_rate: f64,

    /// OU process uncertainty (sqrt of posterior variance).
    /// Higher values = less confident in drift estimate.
    pub ou_uncertainty: f64,
}

impl Default for ControlState {
    fn default() -> Self {
        Self {
            wealth: 0.0,
            position: 0.0,
            margin_used: 0.0,
            time: 0.0,
            belief: BeliefState::default(),
            vol_regime: DiscreteDistribution::default(),
            time_to_funding: 8.0,
            predicted_funding: 0.0,
            learning_trust: 1.0,
            model_health: ModelHealth::default(),
            reduce_only: false,
            drawdown: 0.0,
            rate_limit_headroom: 1.0,
            last_realized_edge_bps: 0.0,
            market_spread_bps: 0.0,
            drift_rate: 0.0,
            ou_uncertainty: 0.0,
        }
    }
}

impl ControlState {
    /// Create control state from trading state and belief.
    pub fn from_trading_state(trading: &TradingState, belief: BeliefState) -> Self {
        Self {
            wealth: trading.wealth,
            position: trading.position,
            margin_used: trading.margin_used,
            time: trading.session_time,
            belief,
            vol_regime: DiscreteDistribution::default(),
            time_to_funding: trading.time_to_funding,
            predicted_funding: trading.predicted_funding,
            learning_trust: 1.0,
            model_health: ModelHealth::default(),
            reduce_only: trading.reduce_only,
            drawdown: trading.drawdown,
            rate_limit_headroom: trading.rate_limit_headroom,
            last_realized_edge_bps: trading.last_realized_edge_bps,
            market_spread_bps: trading.market_spread_bps,
            drift_rate: 0.0,
            ou_uncertainty: 0.0,
        }
    }

    // === Time features ===

    /// Time remaining in session [0, 1].
    pub fn time_remaining(&self) -> f64 {
        (1.0 - self.time).max(0.0)
    }

    /// Urgency factor (increases as time runs out).
    pub fn urgency(&self) -> f64 {
        if self.time_remaining() > 0.01 {
            1.0 / self.time_remaining()
        } else {
            100.0 // Max urgency
        }
    }

    /// Whether in terminal zone.
    pub fn is_terminal(&self, threshold: f64) -> bool {
        self.time > threshold
    }

    // === Position features ===

    /// Signed inventory (positive = long).
    pub fn inventory(&self) -> f64 {
        self.position
    }

    /// Absolute inventory.
    pub fn abs_inventory(&self) -> f64 {
        self.position.abs()
    }

    /// Inventory as fraction of max (if known).
    pub fn inventory_fraction(&self, max_position: f64) -> f64 {
        if max_position > 0.0 {
            self.position.abs() / max_position
        } else {
            0.0
        }
    }

    /// Whether position is risky (needs reduction).
    pub fn position_is_risky(&self, threshold_fraction: f64, max_position: f64) -> bool {
        self.inventory_fraction(max_position) > threshold_fraction
    }

    // === Regime features ===

    /// Get volatility regime probabilities.
    pub fn regime_probs(&self) -> [f64; 3] {
        self.vol_regime.probs
    }

    /// Most likely regime (0=Low, 1=Normal, 2=High).
    pub fn current_regime(&self) -> usize {
        self.vol_regime.argmax()
    }

    /// Regime entropy (uncertainty over regime).
    pub fn regime_entropy(&self) -> f64 {
        self.vol_regime.entropy()
    }

    /// Probability of high volatility regime.
    pub fn p_high_vol(&self) -> f64 {
        self.vol_regime.probs[2]
    }

    // === Funding features ===

    /// Whether funding is approaching.
    pub fn funding_approaching(&self, hours_threshold: f64) -> bool {
        self.time_to_funding < hours_threshold
    }

    /// Funding opportunity (signed: positive = go short to collect).
    pub fn funding_opportunity(&self) -> f64 {
        if self.time_to_funding < 1.0 {
            self.predicted_funding // Positive funding = shorts collect
        } else {
            0.0
        }
    }

    // === Belief features ===

    /// Expected edge from belief.
    pub fn expected_edge(&self) -> f64 {
        self.belief.expected_edge()
    }

    /// Edge uncertainty.
    pub fn edge_uncertainty(&self) -> f64 {
        self.belief.total_edge_uncertainty
    }

    /// Probability of positive edge.
    pub fn p_positive_edge(&self) -> f64 {
        self.belief.p_positive_edge()
    }

    /// Overall confidence.
    pub fn confidence(&self) -> f64 {
        self.belief.confidence() * self.learning_trust
    }

    // === Risk features ===

    /// Whether we can quote (not in reduce-only and not degraded).
    pub fn can_quote(&self) -> bool {
        !self.reduce_only && !self.model_health.is_degraded()
    }

    /// Risk score [0, 1] combining various factors.
    pub fn risk_score(&self) -> f64 {
        let drawdown_risk = self.drawdown;
        let position_risk = self.abs_inventory() / 10.0; // Assume max ~10
        let model_risk = if self.model_health.is_degraded() {
            1.0
        } else {
            0.0
        };
        let trust_risk = 1.0 - self.learning_trust;

        // Weighted combination
        (0.4 * drawdown_risk + 0.3 * position_risk + 0.2 * model_risk + 0.1 * trust_risk)
            .clamp(0.0, 1.0)
    }
}

/// State transition model.
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// Starting state
    pub from: ControlState,
    /// Ending state
    pub to: ControlState,
    /// Action taken
    pub action_taken: ActionTaken,
    /// Reward received
    pub reward: f64,
}

/// Action taken in transition (simplified for recording).
#[derive(Debug, Clone, Copy)]
pub enum ActionTaken {
    /// Quoted with parameters
    Quoted { spread_bps: f64, size: f64 },
    /// Did not quote
    NoQuote,
    /// Dumped inventory
    DumpedInventory { amount: f64 },
}

impl ControlState {
    /// Compute reward for a state transition.
    pub fn compute_reward(&self, next: &ControlState, action: ActionTaken) -> f64 {
        // P&L change
        let pnl_change = next.wealth - self.wealth;

        // Risk penalty
        let risk_penalty = next.risk_score() * 0.1;

        // Time penalty (prefer acting sooner)
        let time_penalty = (next.time - self.time) * 0.01;

        // Inventory holding cost (quadratic)
        let holding_cost = self.position.powi(2) * 0.001;

        // Action-specific adjustments
        let action_cost = match action {
            ActionTaken::Quoted { .. } => 0.0,
            ActionTaken::NoQuote => 0.01, // Small cost for not acting
            ActionTaken::DumpedInventory { .. } => 0.05, // Cost for forced liquidation
        };

        pnl_change - risk_penalty - time_penalty - holding_cost - action_cost
    }
}

/// Configuration for state construction.
#[derive(Debug, Clone)]
pub struct StateConfig {
    /// Terminal zone threshold
    pub terminal_threshold: f64,
    /// Funding approaching threshold (hours)
    pub funding_threshold: f64,
    /// Position risk threshold
    pub position_risk_threshold: f64,
    /// Maximum position for normalization
    pub max_position: f64,
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            terminal_threshold: 0.95,
            funding_threshold: 1.0,
            position_risk_threshold: 0.8,
            max_position: 10.0,
        }
    }
}

// === Trait implementations for trait-based architecture ===

use super::traits::{BeliefProvider, ControlStateProvider, MarketMicrostructure, ObservableState};

impl ObservableState for ControlState {
    fn position(&self) -> f64 {
        self.position
    }

    fn wealth(&self) -> f64 {
        self.wealth
    }

    fn margin_used(&self) -> f64 {
        self.margin_used
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn time_to_funding(&self) -> f64 {
        self.time_to_funding
    }

    fn predicted_funding(&self) -> f64 {
        self.predicted_funding
    }

    fn drawdown(&self) -> f64 {
        self.drawdown
    }

    fn reduce_only(&self) -> bool {
        self.reduce_only
    }

    fn learning_trust(&self) -> f64 {
        self.learning_trust
    }

    fn rate_limit_headroom(&self) -> f64 {
        self.rate_limit_headroom
    }
}

impl BeliefProvider for ControlState {
    fn expected_fill_rate(&self) -> f64 {
        self.belief.expected_fill_rate()
    }

    fn fill_rate_uncertainty(&self) -> f64 {
        self.belief.fill_rate_uncertainty()
    }

    fn expected_as(&self) -> f64 {
        self.belief.expected_as()
    }

    fn as_uncertainty(&self) -> f64 {
        self.belief.as_uncertainty()
    }

    fn expected_edge(&self) -> f64 {
        self.belief.expected_edge()
    }

    fn edge_uncertainty(&self) -> f64 {
        self.belief.total_edge_uncertainty
    }

    fn p_positive_edge(&self) -> f64 {
        self.belief.p_positive_edge()
    }

    fn confidence(&self) -> f64 {
        self.belief.confidence()
    }

    fn epistemic_uncertainty(&self) -> f64 {
        self.belief.epistemic_uncertainty
    }

    fn n_fills(&self) -> u64 {
        self.belief.n_fills
    }

    fn regime_probs(&self) -> [f64; 3] {
        self.vol_regime.probs
    }

    fn current_regime(&self) -> usize {
        self.vol_regime.argmax()
    }

    fn regime_entropy(&self) -> f64 {
        self.vol_regime.entropy()
    }
}

impl MarketMicrostructure for ControlState {
    fn is_model_degraded(&self) -> bool {
        self.model_health.is_degraded()
    }

    fn can_quote(&self) -> bool {
        !self.reduce_only && !self.model_health.is_degraded()
    }

    fn risk_score(&self) -> f64 {
        let drawdown_risk = self.drawdown;
        let position_risk = self.position.abs() / 10.0;
        let model_risk = if self.model_health.is_degraded() {
            1.0
        } else {
            0.0
        };
        let trust_risk = 1.0 - self.learning_trust;

        (0.4 * drawdown_risk + 0.3 * position_risk + 0.2 * model_risk + 0.1 * trust_risk)
            .clamp(0.0, 1.0)
    }
}

impl ControlStateProvider for ControlState {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_state_default() {
        let state = ControlState::default();
        assert_eq!(state.time, 0.0);
        assert_eq!(state.position, 0.0);
        assert!(!state.reduce_only);
    }

    #[test]
    fn test_time_features() {
        let mut state = ControlState {
            time: 0.5,
            ..Default::default()
        };
        assert!((state.time_remaining() - 0.5).abs() < 1e-10);
        assert!(!state.is_terminal(0.95));

        state.time = 0.98;
        assert!(state.is_terminal(0.95));
        assert!(state.urgency() > 10.0);
    }

    #[test]
    fn test_position_features() {
        let mut state = ControlState {
            position: 5.0,
            ..Default::default()
        };
        assert_eq!(state.abs_inventory(), 5.0);
        assert!((state.inventory_fraction(10.0) - 0.5).abs() < 1e-10);
        assert!(!state.position_is_risky(0.8, 10.0));

        state.position = 9.0;
        assert!(state.position_is_risky(0.8, 10.0));
    }

    #[test]
    fn test_funding_features() {
        let mut state = ControlState {
            time_to_funding: 0.5,
            predicted_funding: 10.0,
            ..Default::default()
        };

        assert!(state.funding_approaching(1.0));
        assert!((state.funding_opportunity() - 10.0).abs() < 1e-10);

        state.time_to_funding = 5.0;
        assert!(!state.funding_approaching(1.0));
        assert!((state.funding_opportunity() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_risk_score() {
        let mut state = ControlState::default();
        let initial_risk = state.risk_score();

        state.drawdown = 0.5;
        let elevated_risk = state.risk_score();

        assert!(elevated_risk > initial_risk);
    }
}
