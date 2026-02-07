//! Trait-based architecture for control state providers.
//!
//! This module defines traits that decouple Solver logic from concrete State structs,
//! enabling better testing and modularity. The key traits are:
//!
//! - `ObservableState`: Access to observable market state (position, wealth, time)
//! - `BeliefProvider`: Access to Bayesian belief state (edge, fill rate, uncertainty)
//! - `MarketMicrostructure`: Access to market microstructure parameters
//! - `ControlStateProvider`: Aggregates all state access traits
//! - `ControlSolver`: Interface for solvers that compute optimal actions
//!
//! ## Design Goals
//!
//! 1. **Testability**: Use `StateSnapshot` in unit tests without real market data
//! 2. **Modularity**: Solvers depend on traits, not concrete types
//! 3. **Simulation**: Use snapshots for what-if analysis and Monte Carlo
//! 4. **Immutability**: Trait methods are read-only accessors

use super::actions::Action;
use crate::market_maker::strategy::MarketParams;

/// Provides access to observable state (position, wealth, time).
///
/// These are hard facts that can be directly measured, not inferred.
pub trait ObservableState {
    /// Current position (signed: positive = long, negative = short)
    fn position(&self) -> f64;

    /// Absolute position
    fn abs_position(&self) -> f64 {
        self.position().abs()
    }

    /// Current wealth (cumulative P&L)
    fn wealth(&self) -> f64;

    /// Margin currently used
    fn margin_used(&self) -> f64;

    /// Session time as fraction [0, 1]
    fn time(&self) -> f64;

    /// Time remaining in session [0, 1]
    fn time_remaining(&self) -> f64 {
        (1.0 - self.time()).max(0.0)
    }

    /// Urgency factor (increases as time runs out)
    fn urgency(&self) -> f64 {
        if self.time_remaining() > 0.01 {
            1.0 / self.time_remaining()
        } else {
            100.0 // Max urgency
        }
    }

    /// Whether in terminal zone
    fn is_terminal(&self, threshold: f64) -> bool {
        self.time() > threshold
    }

    /// Position as fraction of max position
    fn inventory_fraction(&self, max_position: f64) -> f64 {
        if max_position > 0.0 {
            self.position().abs() / max_position
        } else {
            0.0
        }
    }

    /// Whether position is risky (needs reduction)
    fn position_is_risky(&self, threshold_fraction: f64, max_position: f64) -> bool {
        self.inventory_fraction(max_position) > threshold_fraction
    }

    /// Time to next funding (hours)
    fn time_to_funding(&self) -> f64;

    /// Predicted funding rate (bps)
    fn predicted_funding(&self) -> f64;

    /// Whether funding is approaching
    fn funding_approaching(&self, hours_threshold: f64) -> bool {
        self.time_to_funding() < hours_threshold
    }

    /// Funding opportunity (signed: positive = go short to collect)
    fn funding_opportunity(&self) -> f64 {
        if self.time_to_funding() < 1.0 {
            self.predicted_funding()
        } else {
            0.0
        }
    }

    /// Current drawdown
    fn drawdown(&self) -> f64;

    /// Whether reduce-only mode is active
    fn reduce_only(&self) -> bool;

    /// Trust in learning module [0, 1]
    fn learning_trust(&self) -> f64;
}

/// Provides access to Bayesian belief state.
///
/// These are inferred quantities with associated uncertainty.
pub trait BeliefProvider {
    /// Expected fill rate (lambda)
    fn expected_fill_rate(&self) -> f64;

    /// Fill rate uncertainty (std)
    fn fill_rate_uncertainty(&self) -> f64;

    /// Expected adverse selection (bps)
    fn expected_as(&self) -> f64;

    /// Adverse selection uncertainty (std)
    fn as_uncertainty(&self) -> f64;

    /// Expected edge (bps)
    fn expected_edge(&self) -> f64;

    /// Edge uncertainty (std)
    fn edge_uncertainty(&self) -> f64;

    /// Probability of positive edge
    fn p_positive_edge(&self) -> f64;

    /// Overall confidence [0, 1]
    fn confidence(&self) -> f64;

    /// Epistemic uncertainty (model disagreement)
    fn epistemic_uncertainty(&self) -> f64;

    /// Number of fills observed
    fn n_fills(&self) -> u64;

    /// Volatility regime probabilities [low, normal, high]
    fn regime_probs(&self) -> [f64; 3];

    /// Most likely regime (0=Low, 1=Normal, 2=High)
    fn current_regime(&self) -> usize;

    /// Regime entropy (uncertainty over regime)
    fn regime_entropy(&self) -> f64;

    /// Probability of high volatility regime
    fn p_high_vol(&self) -> f64 {
        self.regime_probs()[2]
    }
}

/// Provides access to market microstructure parameters.
///
/// These come from the parameter estimator and market params.
pub trait MarketMicrostructure {
    /// Whether model health is degraded
    fn is_model_degraded(&self) -> bool;

    /// Whether we can quote (not in reduce-only and not degraded)
    fn can_quote(&self) -> bool;

    /// Risk score [0, 1] combining various factors
    fn risk_score(&self) -> f64;
}

/// Aggregates all state access traits.
///
/// Implement this trait to provide a complete view of the control state.
/// Both `ControlState` and `StateSnapshot` implement this trait.
pub trait ControlStateProvider: ObservableState + BeliefProvider + MarketMicrostructure {
    /// Get the overall confidence (belief confidence * learning trust)
    fn overall_confidence(&self) -> f64 {
        self.confidence() * self.learning_trust()
    }
}

/// Output from a control solver.
#[derive(Debug, Clone)]
pub struct ControlOutput {
    /// The recommended action
    pub action: Action,
    /// Expected value of the action
    pub expected_value: f64,
    /// Confidence in the recommendation [0, 1]
    pub confidence: f64,
}

/// Reason for a control decision.
#[derive(Debug, Clone)]
pub enum ControlReason {
    /// Normal quoting based on edge
    NormalQuoting,
    /// Terminal zone requires inventory reduction
    TerminalZone,
    /// Funding opportunity to capture
    FundingCapture,
    /// High uncertainty - be defensive
    HighUncertainty,
    /// Changepoint detected - distrust models
    ChangepointDetected,
    /// Information value - wait to learn
    InformationValue,
    /// Position limit approaching
    PositionLimit,
    /// Risk limit reached
    RiskLimit,
    /// Model degraded
    ModelDegraded,
}

/// Interface for solvers that compute optimal actions.
///
/// This trait decouples the solver logic from concrete state types.
/// Solvers receive state through the `ControlStateProvider` trait.
pub trait ControlSolver: Send + Sync {
    /// Compute the optimal action given the current state.
    ///
    /// # Arguments
    /// * `state` - State provider (trait object for abstraction)
    /// * `market_params` - Market parameters (kept concrete per design)
    ///
    /// # Returns
    /// * `ControlOutput` with the recommended action and metadata
    fn solve(
        &self,
        state: &dyn ControlStateProvider,
        market_params: &MarketParams,
    ) -> ControlOutput;

    /// Get the name of this solver for logging.
    fn name(&self) -> &'static str;
}

/// Interface for solvers that also maintain a value function.
///
/// Extends `ControlSolver` with value function learning capabilities.
pub trait ValueFunctionSolver: ControlSolver {
    /// Get the Q-value for a state-action pair.
    fn q_value(&self, state: &dyn ControlStateProvider, action: &Action) -> f64;

    /// Get the number of value function updates performed.
    fn n_updates(&self) -> usize;
}

/// Snapshot of control state for simulation and testing.
///
/// This is a concrete struct that implements `ControlStateProvider`,
/// allowing it to be used as a drop-in replacement for real state
/// during simulations, what-if analysis, and unit tests.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    // Observable state
    pub position: f64,
    pub wealth: f64,
    pub margin_used: f64,
    pub time: f64,
    pub time_to_funding: f64,
    pub predicted_funding: f64,
    pub drawdown: f64,
    pub reduce_only: bool,
    pub learning_trust: f64,

    // Belief state
    pub expected_fill_rate: f64,
    pub fill_rate_uncertainty: f64,
    pub expected_as: f64,
    pub as_uncertainty: f64,
    pub expected_edge: f64,
    pub edge_uncertainty: f64,
    pub p_positive_edge: f64,
    pub confidence: f64,
    pub epistemic_uncertainty: f64,
    pub n_fills: u64,
    pub regime_probs: [f64; 3],

    // Market microstructure
    pub is_model_degraded: bool,
}

impl Default for StateSnapshot {
    fn default() -> Self {
        Self {
            position: 0.0,
            wealth: 0.0,
            margin_used: 0.0,
            time: 0.0,
            time_to_funding: 8.0,
            predicted_funding: 0.0,
            drawdown: 0.0,
            reduce_only: false,
            learning_trust: 1.0,
            expected_fill_rate: 1.0,
            fill_rate_uncertainty: 0.5,
            expected_as: 0.0,
            as_uncertainty: 1.0,
            expected_edge: 0.0,
            edge_uncertainty: 1.0,
            p_positive_edge: 0.5,
            confidence: 0.5,
            epistemic_uncertainty: 0.5,
            n_fills: 0,
            regime_probs: [0.33, 0.34, 0.33],
            is_model_degraded: false,
        }
    }
}

impl ObservableState for StateSnapshot {
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
}

impl BeliefProvider for StateSnapshot {
    fn expected_fill_rate(&self) -> f64 {
        self.expected_fill_rate
    }

    fn fill_rate_uncertainty(&self) -> f64 {
        self.fill_rate_uncertainty
    }

    fn expected_as(&self) -> f64 {
        self.expected_as
    }

    fn as_uncertainty(&self) -> f64 {
        self.as_uncertainty
    }

    fn expected_edge(&self) -> f64 {
        self.expected_edge
    }

    fn edge_uncertainty(&self) -> f64 {
        self.edge_uncertainty
    }

    fn p_positive_edge(&self) -> f64 {
        self.p_positive_edge
    }

    fn confidence(&self) -> f64 {
        self.confidence
    }

    fn epistemic_uncertainty(&self) -> f64 {
        self.epistemic_uncertainty
    }

    fn n_fills(&self) -> u64 {
        self.n_fills
    }

    fn regime_probs(&self) -> [f64; 3] {
        self.regime_probs
    }

    fn current_regime(&self) -> usize {
        self.regime_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1)
    }

    fn regime_entropy(&self) -> f64 {
        self.regime_probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }
}

impl MarketMicrostructure for StateSnapshot {
    fn is_model_degraded(&self) -> bool {
        self.is_model_degraded
    }

    fn can_quote(&self) -> bool {
        !self.reduce_only && !self.is_model_degraded
    }

    fn risk_score(&self) -> f64 {
        let drawdown_risk = self.drawdown;
        let position_risk = self.position.abs() / 10.0; // Assume max ~10
        let model_risk = if self.is_model_degraded { 1.0 } else { 0.0 };
        let trust_risk = 1.0 - self.learning_trust;

        (0.4 * drawdown_risk + 0.3 * position_risk + 0.2 * model_risk + 0.1 * trust_risk)
            .clamp(0.0, 1.0)
    }
}

impl ControlStateProvider for StateSnapshot {}

impl StateSnapshot {
    /// Create a snapshot from a ControlStateProvider.
    ///
    /// This is useful for capturing the current state for later analysis.
    pub fn from_provider(provider: &dyn ControlStateProvider) -> Self {
        Self {
            position: provider.position(),
            wealth: provider.wealth(),
            margin_used: provider.margin_used(),
            time: provider.time(),
            time_to_funding: provider.time_to_funding(),
            predicted_funding: provider.predicted_funding(),
            drawdown: provider.drawdown(),
            reduce_only: provider.reduce_only(),
            learning_trust: provider.learning_trust(),
            expected_fill_rate: provider.expected_fill_rate(),
            fill_rate_uncertainty: provider.fill_rate_uncertainty(),
            expected_as: provider.expected_as(),
            as_uncertainty: provider.as_uncertainty(),
            expected_edge: provider.expected_edge(),
            edge_uncertainty: provider.edge_uncertainty(),
            p_positive_edge: provider.p_positive_edge(),
            confidence: provider.confidence(),
            epistemic_uncertainty: provider.epistemic_uncertainty(),
            n_fills: provider.n_fills(),
            regime_probs: provider.regime_probs(),
            is_model_degraded: provider.is_model_degraded(),
        }
    }

    /// Create a snapshot with modified position for what-if analysis.
    pub fn with_position(mut self, position: f64) -> Self {
        self.position = position;
        self
    }

    /// Create a snapshot with modified time for terminal analysis.
    pub fn with_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    /// Create a snapshot with modified edge for sensitivity analysis.
    pub fn with_edge(mut self, edge: f64, uncertainty: f64) -> Self {
        self.expected_edge = edge;
        self.edge_uncertainty = uncertainty;
        // Approximate normal CDF for P(edge > 0)
        // Using polynomial approximation of erf
        self.p_positive_edge = if uncertainty > 0.0 {
            let z = edge / (uncertainty * std::f64::consts::SQRT_2);
            let t = 1.0 / (1.0 + 0.3275911 * z.abs());
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let erf_approx = 1.0
                - (a1 * t + a2 * t.powi(2) + a3 * t.powi(3) + a4 * t.powi(4) + a5 * t.powi(5))
                    * (-z * z).exp();
            let erf_val = if z >= 0.0 { erf_approx } else { -erf_approx };
            0.5 * (1.0 + erf_val)
        } else if edge > 0.0 {
            1.0
        } else {
            0.0
        };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_snapshot_default() {
        let snapshot = StateSnapshot::default();

        assert_eq!(snapshot.position(), 0.0);
        assert_eq!(snapshot.wealth(), 0.0);
        assert_eq!(snapshot.time(), 0.0);
        assert!((snapshot.time_remaining() - 1.0).abs() < 1e-10);
        assert!(snapshot.can_quote());
    }

    #[test]
    fn test_state_snapshot_from_provider() {
        let original = StateSnapshot {
            position: 5.0,
            wealth: 100.0,
            time: 0.5,
            expected_edge: 2.0,
            ..Default::default()
        };

        let copy = StateSnapshot::from_provider(&original);

        assert_eq!(copy.position(), original.position());
        assert_eq!(copy.wealth(), original.wealth());
        assert_eq!(copy.time(), original.time());
        assert_eq!(copy.expected_edge(), original.expected_edge());
    }

    #[test]
    fn test_state_snapshot_with_modifiers() {
        let snapshot = StateSnapshot::default()
            .with_position(3.0)
            .with_time(0.8)
            .with_edge(5.0, 2.0);

        assert_eq!(snapshot.position(), 3.0);
        assert_eq!(snapshot.time(), 0.8);
        assert_eq!(snapshot.expected_edge(), 5.0);
        assert_eq!(snapshot.edge_uncertainty(), 2.0);
        assert!(snapshot.p_positive_edge() > 0.9); // 5.0 / 2.0 = 2.5 std above 0
    }

    #[test]
    fn test_observable_state_methods() {
        let snapshot = StateSnapshot {
            time: 0.9,
            position: 8.0,
            time_to_funding: 0.5,
            predicted_funding: 5.0,
            ..Default::default()
        };

        assert!(snapshot.is_terminal(0.85));
        assert!(!snapshot.is_terminal(0.95));
        assert!((snapshot.time_remaining() - 0.1).abs() < 1e-10);
        assert!(snapshot.urgency() > 9.0);
        assert!((snapshot.inventory_fraction(10.0) - 0.8).abs() < 1e-10);
        assert!(snapshot.position_is_risky(0.7, 10.0));
        assert!(snapshot.funding_approaching(1.0));
        assert!((snapshot.funding_opportunity() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_belief_provider_methods() {
        let snapshot = StateSnapshot {
            regime_probs: [0.1, 0.2, 0.7],
            ..Default::default()
        };

        assert_eq!(snapshot.current_regime(), 2); // High vol
        assert!((snapshot.p_high_vol() - 0.7).abs() < 1e-10);

        // Entropy should be positive
        assert!(snapshot.regime_entropy() > 0.0);
    }

    #[test]
    fn test_market_microstructure_methods() {
        let snapshot = StateSnapshot {
            reduce_only: false,
            is_model_degraded: false,
            drawdown: 0.1,
            position: 5.0,
            learning_trust: 0.8,
            ..Default::default()
        };

        assert!(snapshot.can_quote());

        let risk = snapshot.risk_score();
        assert!(risk > 0.0);
        assert!(risk < 1.0);

        // Degraded model should prevent quoting
        let degraded = StateSnapshot {
            is_model_degraded: true,
            ..Default::default()
        };
        assert!(!degraded.can_quote());
    }

    #[test]
    fn test_control_state_provider_overall_confidence() {
        let snapshot = StateSnapshot {
            confidence: 0.8,
            learning_trust: 0.9,
            ..Default::default()
        };

        let overall = snapshot.overall_confidence();
        assert!((overall - 0.72).abs() < 1e-10); // 0.8 * 0.9
    }
}
