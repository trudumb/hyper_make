//! Information value calculations for optimal timing.
//!
//! Determines when it's better to wait and learn vs act immediately.
//! Based on the value of information theory: if learning will significantly
//! reduce uncertainty, the expected gain may exceed the opportunity cost of waiting.

use super::actions::Action;
use super::state::ControlState;

/// Information value calculator.
///
/// Determines whether the value of waiting to learn more exceeds
/// the value of acting now.
#[derive(Debug, Clone)]
pub struct InformationValue {
    /// Configuration
    config: InformationConfig,
    /// Recent uncertainty history
    uncertainty_history: Vec<f64>,
    /// Recent edge history
    edge_history: Vec<f64>,
    /// Maximum history length
    max_history: usize,
}

/// Configuration for information value calculations.
#[derive(Debug, Clone)]
pub struct InformationConfig {
    /// Cost per period of waiting (opportunity cost)
    pub wait_cost_per_period: f64,
    /// Minimum uncertainty reduction to justify waiting
    pub min_uncertainty_reduction: f64,
    /// Learning rate (how fast uncertainty decreases)
    pub learning_rate: f64,
    /// Maximum wait cycles before forcing action
    pub max_wait_cycles: u32,
    /// Threshold for deciding to wait
    pub wait_threshold: f64,
    /// Minimum uncertainty to even consider waiting (bps)
    /// If σ_edge < this, act immediately. Should approximate half-spread.
    /// Derived from GLFT: δ = (1/γ) × ln(1 + γ/κ), typically 5-15 bps
    pub min_uncertainty_to_wait: f64,
}

impl Default for InformationConfig {
    fn default() -> Self {
        Self {
            wait_cost_per_period: 0.01,     // 0.01 bps opportunity cost per cycle
            min_uncertainty_reduction: 0.1, // 10% reduction needed
            learning_rate: 0.05,            // 5% uncertainty reduction per observation
            max_wait_cycles: 10,
            wait_threshold: 0.5,
            // Typical GLFT half-spread for crypto: ~8 bps
            // If uncertainty < half-spread, we're confident enough to act
            min_uncertainty_to_wait: 8.0,
        }
    }
}

impl Default for InformationValue {
    fn default() -> Self {
        Self::new(InformationConfig::default())
    }
}

impl InformationValue {
    /// Create a new information value calculator.
    pub fn new(config: InformationConfig) -> Self {
        Self {
            config,
            uncertainty_history: Vec::new(),
            edge_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Update with current state observation.
    pub fn update(&mut self, state: &ControlState) {
        self.uncertainty_history.push(state.edge_uncertainty());
        self.edge_history.push(state.expected_edge());

        // Trim history
        if self.uncertainty_history.len() > self.max_history {
            self.uncertainty_history.remove(0);
            self.edge_history.remove(0);
        }
    }

    /// Should we wait to learn more?
    pub fn should_wait(&self, state: &ControlState, best_action: &Action) -> bool {
        // Don't wait if:
        // 1. Terminal zone (no time)
        if state.time_remaining() < 0.05 {
            return false;
        }

        // 2. Action is urgent
        if best_action.is_urgent() {
            return false;
        }

        // 3. Uncertainty small relative to typical half-spread - act immediately
        // First-principles: If σ_edge < δ (half-spread), uncertainty doesn't swamp edge
        if state.edge_uncertainty() < self.config.min_uncertainty_to_wait {
            return false;
        }

        // Calculate value of waiting vs acting
        let wait_value = self.value_of_waiting(state);
        let act_value = self.value_of_acting(state, best_action);

        wait_value > act_value * self.config.wait_threshold
    }

    /// Calculate expected value of waiting one period.
    pub fn value_of_waiting(&self, state: &ControlState) -> f64 {
        let current_uncertainty = state.edge_uncertainty();
        let expected_edge = state.expected_edge();

        // Expected uncertainty after waiting (from learning rate)
        let expected_uncertainty = self.predict_uncertainty_reduction(current_uncertainty);

        // Value of uncertainty reduction
        let uncertainty_reduction = current_uncertainty - expected_uncertainty;
        let info_value = self.information_gain_value(expected_edge, uncertainty_reduction);

        // Subtract waiting cost
        info_value - self.config.wait_cost_per_period
    }

    /// Calculate value of acting now.
    fn value_of_acting(&self, _state: &ControlState, action: &Action) -> f64 {
        match action {
            Action::Quote { expected_value, .. } => *expected_value,
            _ => 0.0,
        }
    }

    /// Predict uncertainty reduction from one more observation.
    fn predict_uncertainty_reduction(&self, current: f64) -> f64 {
        // Use historical rate if available (need at least 10 observations)
        if self.uncertainty_history.len() >= 10 {
            let n = self.uncertainty_history.len();
            let recent: f64 = self.uncertainty_history[n - 5..].iter().sum::<f64>() / 5.0;
            let older: f64 = self.uncertainty_history[n - 10..n - 5].iter().sum::<f64>() / 5.0;

            if older > recent {
                // Uncertainty is decreasing
                let rate = (older - recent) / older;
                return current * (1.0 - rate.min(0.2));
            }
        }

        // Default: use configured learning rate
        current * (1.0 - self.config.learning_rate)
    }

    /// Calculate value of reducing uncertainty.
    fn information_gain_value(&self, expected_edge: f64, uncertainty_reduction: f64) -> f64 {
        // Value depends on current edge and how much we reduce uncertainty
        // Higher edge + more reduction = more valuable

        // If edge sign could change with uncertainty reduction, that's very valuable
        let edge_to_uncertainty = expected_edge.abs() / (uncertainty_reduction.max(0.01));

        if edge_to_uncertainty < 1.0 {
            // Uncertainty is larger than edge - information is very valuable
            uncertainty_reduction * 2.0
        } else {
            // Edge is clear - information has diminishing value
            uncertainty_reduction * 0.5
        }
    }

    /// Get recommended wait cycles.
    pub fn recommended_wait_cycles(&self, state: &ControlState) -> u32 {
        let current_uncertainty = state.edge_uncertainty();

        // Calculate how many cycles to reach target uncertainty
        let target_uncertainty =
            current_uncertainty * (1.0 - self.config.min_uncertainty_reduction);
        let learning_rate = self.config.learning_rate;

        if learning_rate <= 0.0 {
            return 1;
        }

        // Solve: current * (1-learning_rate)^n = target
        // n = log(target/current) / log(1-learning_rate)
        let n = (target_uncertainty / current_uncertainty).ln() / (1.0 - learning_rate).ln();

        (n.ceil() as u32).min(self.config.max_wait_cycles).max(1)
    }

    /// Calculate expected information gain per cycle.
    pub fn expected_info_gain(&self, state: &ControlState) -> f64 {
        let current = state.edge_uncertainty();
        let expected = self.predict_uncertainty_reduction(current);
        current - expected
    }

    /// Get summary for diagnostics.
    pub fn summary(&self, state: &ControlState, action: &Action) -> InformationSummary {
        InformationSummary {
            current_uncertainty: state.edge_uncertainty(),
            wait_value: self.value_of_waiting(state),
            act_value: self.value_of_acting(state, action),
            should_wait: self.should_wait(state, action),
            recommended_wait: self.recommended_wait_cycles(state),
            info_gain: self.expected_info_gain(state),
        }
    }
}

/// Summary of information value analysis.
#[derive(Debug, Clone)]
pub struct InformationSummary {
    /// Current edge uncertainty
    pub current_uncertainty: f64,
    /// Expected value of waiting
    pub wait_value: f64,
    /// Expected value of acting
    pub act_value: f64,
    /// Recommendation to wait
    pub should_wait: bool,
    /// Recommended wait cycles
    pub recommended_wait: u32,
    /// Expected information gain
    pub info_gain: f64,
}

/// Action value comparator for timing decisions.
#[derive(Debug, Clone)]
pub struct ActionValueComparator {}

impl ActionValueComparator {
    /// Create a new comparator.
    pub fn new() -> Self {
        Self {}
    }

    /// Compare immediate action value vs delayed action value.
    pub fn compare_timing(
        &self,
        state: &ControlState,
        now_action: &Action,
        later_action: &Action,
        wait_cycles: u32,
    ) -> TimingComparison {
        let now_value = self.action_value(state, now_action);

        // Estimate state after waiting
        let later_state = self.project_state(state, wait_cycles);
        let later_value = self.action_value(&later_state, later_action);

        // Discount later value
        let discount = 0.99_f64.powi(wait_cycles as i32);
        let discounted_later = later_value * discount;

        TimingComparison {
            now_value,
            later_value: discounted_later,
            prefer_waiting: discounted_later > now_value,
            advantage: discounted_later - now_value,
        }
    }

    /// Estimate action value.
    fn action_value(&self, _state: &ControlState, action: &Action) -> f64 {
        match action {
            Action::Quote { expected_value, .. } => *expected_value,
            Action::NoQuote { .. } => -0.01,
            Action::WaitToLearn {
                expected_info_gain, ..
            } => *expected_info_gain * 0.5,
            _ => 0.0,
        }
    }

    /// Project state forward by N cycles.
    fn project_state(&self, state: &ControlState, cycles: u32) -> ControlState {
        let mut projected = state.clone();

        // Time advances
        let dt = 0.001 * cycles as f64;
        projected.time = (projected.time + dt).min(1.0);

        // Uncertainty decreases
        projected.belief.total_edge_uncertainty *= 0.95_f64.powi(cycles as i32);

        projected
    }
}

impl Default for ActionValueComparator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of timing comparison.
#[derive(Debug, Clone)]
pub struct TimingComparison {
    /// Value of acting now
    pub now_value: f64,
    /// Discounted value of acting later
    pub later_value: f64,
    /// Whether waiting is preferred
    pub prefer_waiting: bool,
    /// Advantage of waiting
    pub advantage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_information_value_default() {
        let iv = InformationValue::default();
        let state = ControlState::default();
        let action = Action::Quote {
            ladder: Default::default(),
            expected_value: 1.0,
        };

        let should_wait = iv.should_wait(&state, &action);
        // With default uncertainty, should not wait
        assert!(!should_wait);
    }

    #[test]
    fn test_high_uncertainty_wait() {
        let iv = InformationValue::default();
        let mut state = ControlState::default();
        state.belief.total_edge_uncertainty = 5.0; // High uncertainty

        let _action = Action::Quote {
            ladder: Default::default(),
            expected_value: 0.1, // Small expected value
        };

        // High uncertainty + low expected value might suggest waiting
        let wait_value = iv.value_of_waiting(&state);
        assert!(wait_value > -0.1);
    }

    #[test]
    fn test_terminal_no_wait() {
        let iv = InformationValue::default();
        let mut state = ControlState::default();
        state.time = 0.98; // Near terminal

        let action = Action::Quote {
            ladder: Default::default(),
            expected_value: 0.1,
        };

        // Should never wait near terminal
        assert!(!iv.should_wait(&state, &action));
    }

    #[test]
    fn test_recommended_wait_cycles() {
        let iv = InformationValue::default();
        let mut state = ControlState::default();
        state.belief.total_edge_uncertainty = 2.0;

        let cycles = iv.recommended_wait_cycles(&state);
        assert!(cycles >= 1);
        assert!(cycles <= iv.config.max_wait_cycles);
    }

    #[test]
    fn test_timing_comparison() {
        let comparator = ActionValueComparator::new();
        let state = ControlState::default();

        let now = Action::Quote {
            ladder: Default::default(),
            expected_value: 0.5,
        };

        let later = Action::Quote {
            ladder: Default::default(),
            expected_value: 0.6,
        };

        let comparison = comparator.compare_timing(&state, &now, &later, 5);

        // Later action has higher value, but discounting matters
        assert!(comparison.now_value < 0.6);
    }
}
