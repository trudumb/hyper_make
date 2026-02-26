//! Offline Simulation Engine for strategy validation.
//!
//! This module exploits the trait-based architecture to run solvers against
//! simulated market data without going live. Solvers receive `StateSnapshot`
//! objects through `dyn ControlStateProvider` and cannot distinguish simulation
//! from production.
//!
//! ## Key Components
//!
//! - `SimulationEngine`: Core simulation loop
//! - `MarketScenario`: Trait for generating state sequences
//! - `FillModel`: Simulates fill outcomes based on actions
//! - `SimulationResult`: Aggregated metrics from simulation runs
//!
//! ## Usage
//!
//! ```ignore
//! let scenario = TrendingScenario::new(0.001, 100); // 10bps/step trend, 100 steps
//! let solver = OptimalController::default();
//! let config = SimulationConfig::default();
//!
//! let result = SimulationEngine::run(&solver, &scenario, &config);
//! println!("Simulated P&L: {:.2} bps", result.total_pnl_bps);
//! ```

use super::actions::Action;
use super::traits::{ControlSolver, StateSnapshot};
use crate::market_maker::strategy::MarketParams;

use std::collections::VecDeque;

/// Configuration for simulation runs.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Initial position
    pub initial_position: f64,
    /// Initial wealth
    pub initial_wealth: f64,
    /// Maximum position allowed
    pub max_position: f64,
    /// Base fill probability for quotes
    pub base_fill_prob: f64,
    /// Fill probability decay per bps of spread
    pub fill_prob_decay_per_bps: f64,
    /// Adverse selection cost (bps) per fill
    pub as_cost_bps: f64,
    /// Maker fee (bps)
    pub maker_fee_bps: f64,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Whether to record full trajectory
    pub record_trajectory: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            initial_position: 0.0,
            initial_wealth: 0.0,
            max_position: 10.0,
            base_fill_prob: 0.3,
            fill_prob_decay_per_bps: 0.02,
            as_cost_bps: 1.5,
            maker_fee_bps: 1.5,
            seed: 42,
            record_trajectory: false,
        }
    }
}

/// Result of a single simulation step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// The state at this step
    pub state: StateSnapshot,
    /// Action taken by solver
    pub action: Action,
    /// Whether bid filled
    pub bid_filled: bool,
    /// Whether ask filled
    pub ask_filled: bool,
    /// P&L from this step (bps)
    pub step_pnl_bps: f64,
    /// Position after this step
    pub position_after: f64,
}

/// Aggregated results from a simulation run.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Total P&L (bps)
    pub total_pnl_bps: f64,
    /// Number of steps simulated
    pub n_steps: usize,
    /// Number of fills (bid + ask)
    pub n_fills: usize,
    /// Number of bid fills
    pub n_bid_fills: usize,
    /// Number of ask fills
    pub n_ask_fills: usize,
    /// Final position
    pub final_position: f64,
    /// Maximum position reached
    pub max_position_reached: f64,
    /// Maximum drawdown (bps)
    pub max_drawdown_bps: f64,
    /// Sharpe ratio (annualized, assuming 1-minute steps)
    pub sharpe_ratio: f64,
    /// Average spread quoted (bps)
    pub avg_spread_bps: f64,
    /// Fill rate (fills per step)
    pub fill_rate: f64,
    /// Full trajectory (if recorded)
    pub trajectory: Option<Vec<StepResult>>,
    /// Action distribution
    pub action_counts: ActionCounts,
}

/// Count of each action type taken.
#[derive(Debug, Clone, Default)]
pub struct ActionCounts {
    pub quote: usize,
    pub no_quote: usize,
    pub dump_inventory: usize,
    pub build_inventory: usize,
    pub wait_to_learn: usize,
}

/// Trait for market scenarios that generate state sequences.
///
/// Implement this to create custom scenarios for testing.
pub trait MarketScenario: Send + Sync {
    /// Get the number of steps in this scenario.
    fn n_steps(&self) -> usize;

    /// Generate the state at a given step.
    ///
    /// # Arguments
    /// * `step` - The step index (0-indexed)
    /// * `current_position` - Current position (for state consistency)
    /// * `current_wealth` - Current wealth (for state consistency)
    fn state_at(&self, step: usize, current_position: f64, current_wealth: f64) -> StateSnapshot;

    /// Get market parameters at a given step.
    fn market_params_at(&self, step: usize) -> MarketParams;

    /// Name of this scenario for logging.
    fn name(&self) -> &str;
}

/// Simple fill model based on spread width.
#[derive(Debug, Clone)]
pub struct SimpleFillModel {
    base_prob: f64,
    decay_per_bps: f64,
    rng_state: u64,
}

impl SimpleFillModel {
    pub fn new(base_prob: f64, decay_per_bps: f64, seed: u64) -> Self {
        Self {
            base_prob,
            decay_per_bps,
            rng_state: seed,
        }
    }

    /// Simple LCG random number generator for reproducibility.
    fn next_random(&mut self) -> f64 {
        // LCG parameters from Numerical Recipes
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Simulate whether a quote at given spread fills.
    pub fn would_fill(&mut self, spread_bps: f64) -> bool {
        let fill_prob = (self.base_prob - self.decay_per_bps * spread_bps).max(0.01);
        self.next_random() < fill_prob
    }
}

/// The core simulation engine.
pub struct SimulationEngine;

impl SimulationEngine {
    /// Run a simulation with the given solver and scenario.
    pub fn run(
        solver: &dyn ControlSolver,
        scenario: &dyn MarketScenario,
        config: &SimulationConfig,
    ) -> SimulationResult {
        let mut position = config.initial_position;
        let mut wealth = config.initial_wealth;
        let mut fill_model = SimpleFillModel::new(
            config.base_fill_prob,
            config.fill_prob_decay_per_bps,
            config.seed,
        );

        let mut total_pnl_bps: f64 = 0.0;
        let mut n_fills: usize = 0;
        let mut n_bid_fills: usize = 0;
        let mut n_ask_fills: usize = 0;
        let mut max_position: f64 = position.abs();
        let mut peak_wealth: f64 = wealth;
        let mut max_drawdown: f64 = 0.0;
        let mut spread_sum: f64 = 0.0;
        let mut spread_count: usize = 0;
        let mut action_counts = ActionCounts::default();
        let mut trajectory = if config.record_trajectory {
            Some(Vec::with_capacity(scenario.n_steps()))
        } else {
            None
        };

        // For Sharpe calculation
        let mut returns: Vec<f64> = Vec::with_capacity(scenario.n_steps());

        for step in 0..scenario.n_steps() {
            // Generate state for this step
            let state = scenario.state_at(step, position, wealth);
            let market_params = scenario.market_params_at(step);

            // Get solver's decision
            let output = solver.solve(&state, &market_params);
            let action = output.action;

            // Count action types
            match &action {
                Action::Quote { .. } => action_counts.quote += 1,
                Action::NoQuote { .. } => action_counts.no_quote += 1,
                Action::DumpInventory { .. } => action_counts.dump_inventory += 1,
                Action::BuildInventory { .. } => action_counts.build_inventory += 1,
                Action::WaitToLearn { .. } => action_counts.wait_to_learn += 1,
            }

            // Simulate fills and P&L
            let (bid_filled, ask_filled, step_pnl, spread_bps) = Self::simulate_step(
                &action,
                &mut fill_model,
                config,
                position,
                config.max_position,
            );

            // Update position
            if bid_filled {
                position += 1.0; // Simplified: 1 unit per fill
                n_bid_fills += 1;
                n_fills += 1;
            }
            if ask_filled {
                position -= 1.0;
                n_ask_fills += 1;
                n_fills += 1;
            }

            // Clamp position
            position = position.clamp(-config.max_position, config.max_position);

            // Update wealth and tracking
            wealth += step_pnl;
            total_pnl_bps += step_pnl;
            returns.push(step_pnl);

            max_position = max_position.max(position.abs());
            peak_wealth = peak_wealth.max(wealth);
            let drawdown = peak_wealth - wealth;
            max_drawdown = max_drawdown.max(drawdown);

            if let Some(spread) = spread_bps {
                spread_sum += spread;
                spread_count += 1;
            }

            // Record trajectory if requested
            if let Some(ref mut traj) = trajectory {
                traj.push(StepResult {
                    state,
                    action,
                    bid_filled,
                    ask_filled,
                    step_pnl_bps: step_pnl,
                    position_after: position,
                });
            }
        }

        // Calculate Sharpe ratio
        let sharpe = Self::calculate_sharpe(&returns);

        SimulationResult {
            total_pnl_bps,
            n_steps: scenario.n_steps(),
            n_fills,
            n_bid_fills,
            n_ask_fills,
            final_position: position,
            max_position_reached: max_position,
            max_drawdown_bps: max_drawdown,
            sharpe_ratio: sharpe,
            avg_spread_bps: if spread_count > 0 {
                spread_sum / spread_count as f64
            } else {
                0.0
            },
            fill_rate: n_fills as f64 / scenario.n_steps() as f64,
            trajectory,
            action_counts,
        }
    }

    /// Simulate a single step's fills and P&L.
    fn simulate_step(
        action: &Action,
        fill_model: &mut SimpleFillModel,
        config: &SimulationConfig,
        position: f64,
        max_position: f64,
    ) -> (bool, bool, f64, Option<f64>) {
        match action {
            Action::Quote {
                ladder,
                expected_value: _,
            } => {
                // Extract spread from ladder (simplified)
                let spread_bps = ladder
                    .bids
                    .first()
                    .zip(ladder.asks.first())
                    .map(|(b, a)| b.depth_bps + a.depth_bps)
                    .unwrap_or(8.0);

                let can_buy = position < max_position;
                let can_sell = position > -max_position;

                let bid_filled = can_buy && fill_model.would_fill(spread_bps / 2.0);
                let ask_filled = can_sell && fill_model.would_fill(spread_bps / 2.0);

                // P&L: spread capture - AS - fees
                let mut pnl = 0.0;
                if bid_filled {
                    pnl += spread_bps / 2.0 - config.as_cost_bps - config.maker_fee_bps;
                }
                if ask_filled {
                    pnl += spread_bps / 2.0 - config.as_cost_bps - config.maker_fee_bps;
                }

                (bid_filled, ask_filled, pnl, Some(spread_bps))
            }

            Action::DumpInventory { urgency, .. } => {
                // Aggressive exit - pays spread + slippage
                let slippage_bps = 2.0 * urgency;
                let pnl = -slippage_bps - config.maker_fee_bps;
                // Position reduction simulated externally
                (false, false, pnl, None)
            }

            Action::BuildInventory { .. } => {
                // Aggressive entry - pays spread
                let pnl = -4.0 - config.maker_fee_bps; // Half-spread to cross
                (true, false, pnl, None) // Simplified: always fills one side
            }

            Action::NoQuote { .. } | Action::WaitToLearn { .. } => {
                // No fills, no P&L (but opportunity cost not modeled here)
                (false, false, 0.0, None)
            }
        }
    }

    /// Calculate annualized Sharpe ratio.
    fn calculate_sharpe(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        // Annualize assuming 1-minute steps, ~525600 minutes/year
        let annualization_factor = (525600.0_f64).sqrt();
        (mean / std) * annualization_factor
    }

    /// Run multiple simulations with different seeds for Monte Carlo analysis.
    pub fn monte_carlo(
        solver: &dyn ControlSolver,
        scenario: &dyn MarketScenario,
        config: &SimulationConfig,
        n_runs: usize,
    ) -> MonteCarloResult {
        let mut results = Vec::with_capacity(n_runs);

        for i in 0..n_runs {
            let mut run_config = config.clone();
            run_config.seed = config.seed.wrapping_add(i as u64);
            results.push(Self::run(solver, scenario, &run_config));
        }

        MonteCarloResult::from_runs(results)
    }

    /// Run parameter sweep over a range of values.
    pub fn parameter_sweep<F, P>(
        solver_factory: F,
        scenario: &dyn MarketScenario,
        config: &SimulationConfig,
        param_values: &[P],
    ) -> Vec<(P, SimulationResult)>
    where
        F: Fn(&P) -> Box<dyn ControlSolver>,
        P: Clone,
    {
        param_values
            .iter()
            .map(|param| {
                let solver = solver_factory(param);
                let result = Self::run(solver.as_ref(), scenario, config);
                (param.clone(), result)
            })
            .collect()
    }
}

/// Results from Monte Carlo simulation.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Number of runs
    pub n_runs: usize,
    /// Mean P&L (bps)
    pub mean_pnl_bps: f64,
    /// Std of P&L (bps)
    pub std_pnl_bps: f64,
    /// 5th percentile P&L (VaR)
    pub pnl_p5: f64,
    /// 95th percentile P&L
    pub pnl_p95: f64,
    /// Mean Sharpe ratio
    pub mean_sharpe: f64,
    /// Mean fill rate
    pub mean_fill_rate: f64,
    /// Probability of positive P&L
    pub prob_positive: f64,
}

impl MonteCarloResult {
    fn from_runs(results: Vec<SimulationResult>) -> Self {
        let n = results.len();
        if n == 0 {
            return Self {
                n_runs: 0,
                mean_pnl_bps: 0.0,
                std_pnl_bps: 0.0,
                pnl_p5: 0.0,
                pnl_p95: 0.0,
                mean_sharpe: 0.0,
                mean_fill_rate: 0.0,
                prob_positive: 0.0,
            };
        }

        let mut pnls: Vec<f64> = results.iter().map(|r| r.total_pnl_bps).collect();
        pnls.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_pnl = pnls.iter().sum::<f64>() / n as f64;
        let variance =
            pnls.iter().map(|p| (p - mean_pnl).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

        let p5_idx = (n as f64 * 0.05).floor() as usize;
        let p95_idx = (n as f64 * 0.95).floor() as usize;

        Self {
            n_runs: n,
            mean_pnl_bps: mean_pnl,
            std_pnl_bps: variance.sqrt(),
            pnl_p5: pnls[p5_idx.min(n - 1)],
            pnl_p95: pnls[p95_idx.min(n - 1)],
            mean_sharpe: results.iter().map(|r| r.sharpe_ratio).sum::<f64>() / n as f64,
            mean_fill_rate: results.iter().map(|r| r.fill_rate).sum::<f64>() / n as f64,
            prob_positive: pnls.iter().filter(|&&p| p > 0.0).count() as f64 / n as f64,
        }
    }
}

// ============================================================================
// Built-in Market Scenarios
// ============================================================================

/// Trending market scenario.
///
/// Simulates a market with consistent drift in one direction.
#[derive(Debug, Clone)]
pub struct TrendingScenario {
    /// Drift per step (as fraction, e.g., 0.0001 = 1 bps)
    pub drift_per_step: f64,
    /// Number of steps
    pub steps: usize,
    /// Base volatility
    pub base_sigma: f64,
    /// Base edge estimate
    pub base_edge_bps: f64,
}

impl TrendingScenario {
    pub fn new(drift_per_step: f64, steps: usize) -> Self {
        Self {
            drift_per_step,
            steps,
            base_sigma: 0.0002,
            base_edge_bps: 2.0,
        }
    }

    pub fn bullish(steps: usize) -> Self {
        Self::new(0.0001, steps) // 1 bps/step upward drift
    }

    pub fn bearish(steps: usize) -> Self {
        Self::new(-0.0001, steps) // 1 bps/step downward drift
    }
}

impl MarketScenario for TrendingScenario {
    fn n_steps(&self) -> usize {
        self.steps
    }

    fn state_at(&self, step: usize, current_position: f64, current_wealth: f64) -> StateSnapshot {
        let time_fraction = step as f64 / self.steps as f64;
        let momentum = self.drift_per_step * 10000.0; // Convert to bps

        StateSnapshot {
            position: current_position,
            wealth: current_wealth,
            time: time_fraction,
            expected_edge: self.base_edge_bps - momentum.abs() * 0.1, // Edge decreases with strong trend
            edge_uncertainty: 1.0 + momentum.abs() * 0.5,
            p_positive_edge: if self.base_edge_bps > momentum.abs() * 0.1 {
                0.6
            } else {
                0.4
            },
            confidence: 0.7,
            ..Default::default()
        }
    }

    fn market_params_at(&self, step: usize) -> MarketParams {
        MarketParams {
            sigma: self.base_sigma,
            momentum_bps: self.drift_per_step * 10000.0 * (step as f64 + 1.0),
            p_momentum_continue: 0.7, // High continuation prob in trend
            ..Default::default()
        }
    }

    fn name(&self) -> &str {
        "TrendingScenario"
    }
}

/// Mean-reverting market scenario.
///
/// Simulates a market that oscillates around a mean.
#[derive(Debug, Clone)]
pub struct MeanRevertingScenario {
    /// Oscillation period (steps)
    pub period: usize,
    /// Amplitude (as fraction)
    pub amplitude: f64,
    /// Number of steps
    pub steps: usize,
    /// Base edge estimate
    pub base_edge_bps: f64,
}

impl MeanRevertingScenario {
    pub fn new(period: usize, amplitude: f64, steps: usize) -> Self {
        Self {
            period,
            amplitude,
            steps,
            base_edge_bps: 3.0,
        }
    }
}

impl MarketScenario for MeanRevertingScenario {
    fn n_steps(&self) -> usize {
        self.steps
    }

    fn state_at(&self, step: usize, current_position: f64, current_wealth: f64) -> StateSnapshot {
        let phase = 2.0 * std::f64::consts::PI * step as f64 / self.period as f64;
        let price_deviation = self.amplitude * phase.sin();

        StateSnapshot {
            position: current_position,
            wealth: current_wealth,
            time: step as f64 / self.steps as f64,
            expected_edge: self.base_edge_bps + price_deviation.abs() * 10000.0 * 0.5, // Higher edge at extremes
            edge_uncertainty: 0.8,
            p_positive_edge: 0.65,
            confidence: 0.75,
            ..Default::default()
        }
    }

    fn market_params_at(&self, step: usize) -> MarketParams {
        let phase = 2.0 * std::f64::consts::PI * step as f64 / self.period as f64;

        MarketParams {
            sigma: 0.0001 + 0.00005 * phase.cos().abs(),
            flow_imbalance: -phase.sin() * 0.3, // Counter-flow at extremes
            ..Default::default()
        }
    }

    fn name(&self) -> &str {
        "MeanRevertingScenario"
    }
}

/// Liquidation cascade scenario.
///
/// Simulates a sudden market stress event with toxic flow.
#[derive(Debug, Clone)]
pub struct CascadeScenario {
    /// Steps before cascade
    pub calm_steps: usize,
    /// Duration of cascade (steps)
    pub cascade_steps: usize,
    /// Recovery steps after cascade
    pub recovery_steps: usize,
    /// Cascade intensity
    pub intensity: f64,
}

impl CascadeScenario {
    pub fn new(calm: usize, cascade: usize, recovery: usize, intensity: f64) -> Self {
        Self {
            calm_steps: calm,
            cascade_steps: cascade,
            recovery_steps: recovery,
            intensity,
        }
    }

    /// Standard cascade: 50 calm, 10 cascade, 40 recovery
    pub fn standard() -> Self {
        Self::new(50, 10, 40, 2.0)
    }

    /// Severe cascade
    pub fn severe() -> Self {
        Self::new(30, 20, 50, 5.0)
    }
}

impl MarketScenario for CascadeScenario {
    fn n_steps(&self) -> usize {
        self.calm_steps + self.cascade_steps + self.recovery_steps
    }

    fn state_at(&self, step: usize, current_position: f64, current_wealth: f64) -> StateSnapshot {
        let total = self.n_steps();
        let in_cascade = step >= self.calm_steps && step < self.calm_steps + self.cascade_steps;
        let in_recovery = step >= self.calm_steps + self.cascade_steps;

        let (edge, uncertainty, confidence) = if in_cascade {
            // During cascade: negative edge, high uncertainty, low confidence
            (-5.0 * self.intensity, 10.0, 0.2)
        } else if in_recovery {
            // Recovery: edge recovering, uncertainty decreasing
            let recovery_progress =
                (step - self.calm_steps - self.cascade_steps) as f64 / self.recovery_steps as f64;
            (
                2.0 * recovery_progress,
                5.0 * (1.0 - recovery_progress) + 1.0,
                0.3 + 0.4 * recovery_progress,
            )
        } else {
            // Calm: normal conditions
            (3.0, 1.0, 0.7)
        };

        StateSnapshot {
            position: current_position,
            wealth: current_wealth,
            time: step as f64 / total as f64,
            expected_edge: edge,
            edge_uncertainty: uncertainty,
            p_positive_edge: if edge > 0.0 { 0.6 } else { 0.3 },
            confidence,
            regime_probs: if in_cascade {
                [0.0, 0.1, 0.9] // High vol regime
            } else if in_recovery {
                [0.1, 0.5, 0.4]
            } else {
                [0.2, 0.6, 0.2]
            },
            is_model_degraded: in_cascade,
            ..Default::default()
        }
    }

    fn market_params_at(&self, step: usize) -> MarketParams {
        let in_cascade = step >= self.calm_steps && step < self.calm_steps + self.cascade_steps;

        let mut params = MarketParams::default();

        if in_cascade {
            params.sigma = 0.001 * self.intensity;
            params.should_pull_quotes = self.intensity > 3.0;
            params.is_toxic_regime = true;
            params.cascade_intensity = 0.7; // was size_factor = 0.3
            params.tail_risk_intensity = ((self.intensity - 1.0) / 4.0).clamp(0.0, 1.0);
            params.flow_imbalance = -0.8; // Heavy sell pressure
        } else {
            params.sigma = 0.0002;
            params.cascade_intensity = 0.0;
            params.tail_risk_intensity = 0.0;
        }

        params
    }

    fn name(&self) -> &str {
        "CascadeScenario"
    }
}

/// Historical replay scenario.
///
/// Replays a sequence of pre-recorded states.
#[derive(Debug, Clone)]
pub struct HistoricalReplay {
    states: Vec<StateSnapshot>,
    params: Vec<MarketParams>,
    name: String,
}

impl HistoricalReplay {
    pub fn new(states: Vec<StateSnapshot>, params: Vec<MarketParams>, name: String) -> Self {
        assert_eq!(
            states.len(),
            params.len(),
            "States and params must have same length"
        );
        Self {
            states,
            params,
            name,
        }
    }

    /// Create from a VecDeque (common in live systems).
    pub fn from_deque(
        states: VecDeque<StateSnapshot>,
        params: VecDeque<MarketParams>,
        name: String,
    ) -> Self {
        Self::new(
            states.into_iter().collect(),
            params.into_iter().collect(),
            name,
        )
    }
}

impl MarketScenario for HistoricalReplay {
    fn n_steps(&self) -> usize {
        self.states.len()
    }

    fn state_at(&self, step: usize, current_position: f64, current_wealth: f64) -> StateSnapshot {
        let mut state = self.states[step].clone();
        // Override position and wealth to track simulation state
        state.position = current_position;
        state.wealth = current_wealth;
        state
    }

    fn market_params_at(&self, step: usize) -> MarketParams {
        self.params[step].clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::control::controller::OptimalController;

    #[test]
    fn test_trending_scenario() {
        let scenario = TrendingScenario::bullish(100);
        assert_eq!(scenario.n_steps(), 100);

        let state = scenario.state_at(50, 0.0, 0.0);
        assert!(state.time > 0.0);
    }

    #[test]
    fn test_cascade_scenario() {
        let scenario = CascadeScenario::standard();
        assert_eq!(scenario.n_steps(), 100);

        // Check cascade detection
        let calm_state = scenario.state_at(25, 0.0, 0.0);
        let cascade_state = scenario.state_at(55, 0.0, 0.0);
        let recovery_state = scenario.state_at(80, 0.0, 0.0);

        assert!(calm_state.expected_edge > 0.0);
        assert!(cascade_state.expected_edge < 0.0);
        assert!(cascade_state.is_model_degraded);
        assert!(!recovery_state.is_model_degraded);
    }

    #[test]
    fn test_simulation_engine_basic() {
        let scenario = MeanRevertingScenario::new(20, 0.001, 50);
        let solver = OptimalController::default();
        let config = SimulationConfig::default();

        let result = SimulationEngine::run(&solver, &scenario, &config);

        assert_eq!(result.n_steps, 50);
        // n_fills is usize, always >= 0
        assert!(result.total_pnl_bps.is_finite());
    }

    #[test]
    fn test_simulation_with_trajectory() {
        let scenario = TrendingScenario::bullish(20);
        let solver = OptimalController::default();
        let config = SimulationConfig {
            record_trajectory: true,
            ..Default::default()
        };

        let result = SimulationEngine::run(&solver, &scenario, &config);

        assert!(result.trajectory.is_some());
        assert_eq!(result.trajectory.as_ref().unwrap().len(), 20);
    }

    #[test]
    fn test_monte_carlo() {
        let scenario = MeanRevertingScenario::new(10, 0.0005, 30);
        let solver = OptimalController::default();
        let config = SimulationConfig::default();

        let mc_result = SimulationEngine::monte_carlo(&solver, &scenario, &config, 10);

        assert_eq!(mc_result.n_runs, 10);
        assert!(mc_result.prob_positive >= 0.0 && mc_result.prob_positive <= 1.0);
    }

    #[test]
    fn test_fill_model_determinism() {
        let mut model1 = SimpleFillModel::new(0.3, 0.02, 42);
        let mut model2 = SimpleFillModel::new(0.3, 0.02, 42);

        // Same seed should produce same results
        for _ in 0..10 {
            assert_eq!(model1.would_fill(5.0), model2.would_fill(5.0));
        }
    }

    #[test]
    fn test_action_counts() {
        let scenario = CascadeScenario::standard();
        let solver = OptimalController::default();
        let config = SimulationConfig::default();

        let result = SimulationEngine::run(&solver, &scenario, &config);

        let total_actions = result.action_counts.quote
            + result.action_counts.no_quote
            + result.action_counts.dump_inventory
            + result.action_counts.build_inventory
            + result.action_counts.wait_to_learn;

        assert_eq!(total_actions, scenario.n_steps());
    }
}
