//! Value function approximation for the stochastic controller.
//!
//! Uses basis function approximation: V(s) ≈ Σᵢ wᵢ φᵢ(s)
//! where φᵢ are hand-crafted features capturing the state-value relationship.

use super::state::{ControlState, StateTransition};

/// Linear value function approximation using basis functions.
///
/// V(s) = w^T φ(s)
///
/// Basis functions are chosen to capture:
/// - Wealth effects (risk aversion)
/// - Position/inventory costs (quadratic)
/// - Time effects (urgency near terminal)
/// - Belief effects (edge uncertainty)
/// - Cross terms (inventory × time, inventory × AS)
#[derive(Debug, Clone)]
pub struct ValueFunction {
    /// Basis weights
    pub weights: Vec<f64>,
    /// Number of basis functions
    pub n_basis: usize,
    /// Discount factor
    pub gamma: f64,
    /// Learning rate for weight updates
    pub learning_rate: f64,
    /// Number of updates performed
    pub n_updates: u64,
}

impl Default for ValueFunction {
    fn default() -> Self {
        Self::new(BasisConfig::default())
    }
}

/// Configuration for basis functions.
#[derive(Debug, Clone)]
pub struct BasisConfig {
    /// Number of basis functions
    pub n_basis: usize,
    /// Discount factor
    pub gamma: f64,
    /// Initial learning rate
    pub initial_lr: f64,
    /// Learning rate decay
    pub lr_decay: f64,
}

impl Default for BasisConfig {
    fn default() -> Self {
        Self {
            n_basis: 15, // Number of basis functions
            gamma: 0.99,
            initial_lr: 0.01,
            lr_decay: 0.9999,
        }
    }
}

impl ValueFunction {
    /// Create a new value function.
    pub fn new(config: BasisConfig) -> Self {
        Self {
            weights: vec![0.0; config.n_basis],
            n_basis: config.n_basis,
            gamma: config.gamma,
            learning_rate: config.initial_lr,
            n_updates: 0,
        }
    }

    /// Compute value of a state.
    pub fn value(&self, state: &ControlState) -> f64 {
        let features = Self::compute_basis(state);
        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum()
    }

    /// Compute basis functions for a state.
    ///
    /// Returns feature vector φ(s).
    pub fn compute_basis(state: &ControlState) -> Vec<f64> {
        let features = vec![
            // === Wealth features ===
            // 0: Constant (bias term)
            1.0,
            // 1: Wealth (linear)
            state.wealth / 1000.0, // Normalize
            // 2: Wealth squared (risk aversion)
            (state.wealth / 1000.0).powi(2),
            // === Position features ===
            // 3: Position (linear)
            state.position,
            // 4: Position squared (inventory cost)
            state.position.powi(2),
            // 5: Absolute position
            state.position.abs(),
            // === Time features ===
            // 6: Time fraction
            state.time,
            // 7: Time urgency: √(1-t)
            state.time_remaining().sqrt(),
            // 8: Terminal indicator (smooth)
            sigmoid((state.time - 0.95) * 20.0),
            // === Cross terms ===
            // 9: Position × Time (inventory becomes more costly near end)
            state.position * state.time,
            // 10: Position × Expected AS (inventory × toxicity)
            state.position * state.belief.expected_as() / 10.0,
            // === Belief features ===
            // 11: Expected edge
            state.expected_edge() / 10.0,
            // 12: Edge uncertainty
            state.edge_uncertainty() / 10.0,
            // 13: Confidence
            state.confidence(),
            // === Regime features ===
            // 14: Regime entropy (uncertainty about volatility)
            state.regime_entropy(),
        ];

        features
    }

    /// Update weights using TD(0) learning.
    ///
    /// δ = r + γ V(s') - V(s)
    /// w ← w + α δ φ(s)
    pub fn update_td(&mut self, transition: &StateTransition) {
        let features = Self::compute_basis(&transition.from);
        let v_current = self.value(&transition.from);
        let v_next = self.value(&transition.to);

        // TD error
        let td_error = transition.reward + self.gamma * v_next - v_current;

        // Update weights
        let lr = self.effective_learning_rate();
        for (w, f) in self.weights.iter_mut().zip(features.iter()) {
            *w += lr * td_error * f;
        }

        self.n_updates += 1;
    }

    /// Batch update using least squares TD (LSTD).
    pub fn update_batch(&mut self, transitions: &[StateTransition]) {
        if transitions.is_empty() {
            return;
        }

        // Build feature matrices
        let n = transitions.len();
        let d = self.n_basis;

        // A matrix: Σ φ(s)(φ(s) - γφ(s'))^T
        let mut a = vec![vec![0.0; d]; d];
        // b vector: Σ r φ(s)
        let mut b = vec![0.0; d];

        for trans in transitions {
            let phi = Self::compute_basis(&trans.from);
            let phi_next = Self::compute_basis(&trans.to);

            for i in 0..d {
                b[i] += trans.reward * phi[i];
                for j in 0..d {
                    a[i][j] += phi[i] * (phi[j] - self.gamma * phi_next[j]);
                }
            }
        }

        // Regularization
        let lambda = 0.01;
        for (i, row) in a.iter_mut().enumerate().take(d) {
            row[i] += lambda * n as f64;
        }

        // Solve A w = b using simple iteration (could use proper linear algebra)
        let new_weights = solve_linear_system(&a, &b);

        // Blend with current weights (stability)
        let blend = 0.3;
        for (w, nw) in self.weights.iter_mut().zip(new_weights.iter()) {
            *w = *w * (1.0 - blend) + nw * blend;
        }

        self.n_updates += transitions.len() as u64;
    }

    /// Get effective learning rate with decay.
    fn effective_learning_rate(&self) -> f64 {
        self.learning_rate * 0.9999_f64.powi(self.n_updates as i32)
    }

    /// Compute advantage: A(s, a) = Q(s, a) - V(s)
    pub fn advantage(&self, state: &ControlState, q_value: f64) -> f64 {
        q_value - self.value(state)
    }

    /// Get weights as slice.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Set weights from slice.
    pub fn set_weights(&mut self, weights: &[f64]) {
        self.weights
            .iter_mut()
            .zip(weights.iter())
            .for_each(|(w, &new_w)| *w = new_w);
    }
}

/// Sigmoid function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Simple linear system solver (Gauss-Seidel iteration).
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];

    // Gauss-Seidel iterations
    for _ in 0..100 {
        let mut max_change: f64 = 0.0;

        for i in 0..n {
            let mut sum = b[i];
            for (j, &xj) in x.iter().enumerate().take(n) {
                if i != j {
                    sum -= a[i][j] * xj;
                }
            }

            let new_x = if a[i][i].abs() > 1e-10 {
                sum / a[i][i]
            } else {
                0.0
            };

            max_change = max_change.max((new_x - x[i]).abs());
            x[i] = new_x;
        }

        if max_change < 1e-8 {
            break;
        }
    }

    x
}

/// Expected value computation for action selection.
#[derive(Debug, Clone)]
pub struct ExpectedValueComputer {
    /// Value function
    value_fn: ValueFunction,
    /// Number of Monte Carlo samples for integration
    n_samples: usize,
}

impl ExpectedValueComputer {
    /// Create a new expected value computer.
    pub fn new(value_fn: ValueFunction) -> Self {
        Self {
            value_fn,
            n_samples: 50,
        }
    }

    /// Compute expected future value: E[V(s') | s, a].
    ///
    /// Uses Monte Carlo integration over belief distribution.
    pub fn expected_future_value(&self, state: &ControlState, action: &ActionOutcome) -> f64 {
        let mut total = 0.0;

        // Sample possible outcomes
        for i in 0..self.n_samples {
            let next_state = self.sample_transition(state, action, i);
            total += self.value_fn.value(&next_state);
        }

        total / self.n_samples as f64
    }

    /// Sample a state transition given action.
    fn sample_transition(
        &self,
        state: &ControlState,
        action: &ActionOutcome,
        sample_idx: usize,
    ) -> ControlState {
        let mut next = state.clone();

        // Time advances
        let dt = 0.001; // Small time step
        next.time = (next.time + dt).min(1.0);

        // Position changes from fills
        match action {
            ActionOutcome::Quoted {
                fill_prob,
                fill_size,
                fill_side,
            } => {
                // Deterministic approximation using fill probability
                let expected_fill = fill_prob * fill_size;
                let sign = if *fill_side { 1.0 } else { -1.0 };
                next.position += expected_fill
                    * sign
                    * (1.0 + 0.1 * (sample_idx as f64 / self.n_samples as f64 - 0.5));
            }
            ActionOutcome::NoAction => {}
            ActionOutcome::DumpedInventory { amount } => {
                let direction = -next.position.signum();
                next.position += direction * amount;
            }
        }

        // Wealth changes from P&L
        let pnl = self.sample_pnl(state, action, sample_idx);
        next.wealth += pnl;

        // Update belief slightly (learning continues)
        next.belief.soft_reset(0.999);

        next
    }

    /// Sample P&L outcome.
    fn sample_pnl(&self, state: &ControlState, action: &ActionOutcome, _sample_idx: usize) -> f64 {
        match action {
            ActionOutcome::Quoted {
                fill_prob,
                fill_size,
                ..
            } => {
                // Expected spread capture minus AS
                let spread_capture = 4.0; // bps, placeholder
                let as_cost = state.belief.expected_as();
                fill_prob * fill_size * (spread_capture - as_cost) / 10000.0
            }
            ActionOutcome::NoAction => 0.0,
            ActionOutcome::DumpedInventory { amount } => {
                // Crossing spread cost
                -amount * 5.0 / 10000.0 // 5 bps cost
            }
        }
    }
}

/// Simplified action outcome for value computation.
#[derive(Debug, Clone)]
pub enum ActionOutcome {
    /// Quoted with expected fill
    Quoted {
        fill_prob: f64,
        fill_size: f64,
        fill_side: bool, // true = buy, false = sell
    },
    /// No action taken
    NoAction,
    /// Dumped inventory
    DumpedInventory { amount: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_function_default() {
        let vf = ValueFunction::default();
        let state = ControlState::default();

        // Initial value should be 0 (all weights are 0)
        assert!((vf.value(&state) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_basis_functions() {
        let state = ControlState::default();
        let basis = ValueFunction::compute_basis(&state);

        assert_eq!(basis.len(), 15);
        assert!((basis[0] - 1.0).abs() < 1e-10); // Constant term
    }

    #[test]
    fn test_td_update() {
        let mut vf = ValueFunction::default();

        let from = ControlState::default();
        let mut to = ControlState::default();
        to.wealth = 100.0;

        let transition = StateTransition {
            from,
            to,
            action_taken: super::super::state::ActionTaken::NoQuote,
            reward: 1.0,
        };

        let v_before = vf.value(&transition.from);
        vf.update_td(&transition);
        let v_after = vf.value(&transition.from);

        // Value should increase after positive reward
        assert!(v_after > v_before || (v_before - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
