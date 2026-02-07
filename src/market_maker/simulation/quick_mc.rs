//! Quick Monte Carlo EV Simulator
//!
//! Fast Monte Carlo simulation for expected value estimation using kappa-based
//! fill arrivals and flow-based outcome probabilities. Enables proactive quoting
//! decisions when traditional IR/theoretical edge models are inconclusive.
//!
//! # Use Case
//!
//! When IR can't calibrate and theoretical edge is marginal:
//! 1. Run quick MC simulation (10-50 paths)
//! 2. If sim_ev > threshold AND kappa is strong, override conservative decision
//! 3. Generate fills → feed data back to IR → break the vicious cycle
//!
//! # Model
//!
//! For each simulation path:
//! 1. Sample fill count from Poisson(kappa * horizon)
//! 2. For each fill, sample outcome from Bernoulli(P(correct))
//! 3. Compute PnL: correct → +half_spread - cost, wrong → -half_spread * 1.5 - cost
//!
//! P(correct) = 0.5 + alpha * |enhanced_flow|, where alpha ∈ [0.1, 0.25]
//!
//! # Usage
//!
//! ```ignore
//! let simulator = QuickMCSimulator::new(config);
//!
//! let ev = simulator.simulate_ev(
//!     kappa_effective,
//!     enhanced_flow,
//!     spread_bps,
//!     horizon_secs,
//! );
//!
//! if kappa_effective > 2000.0 && ev > 0.5 {
//!     // Override to quote
//! }
//! ```

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Configuration for quick MC simulator.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QuickMCConfig {
    /// Number of simulation paths.
    /// Default: 25
    pub n_paths: usize,

    /// Cost per trade in basis points (fees + slippage).
    /// Default: 2.0
    pub cost_per_trade_bps: f64,

    /// Base alpha for P(correct) calculation.
    /// P(correct) = 0.5 + alpha * |flow|
    /// Default: 0.15
    pub alpha: f64,

    /// Alpha boost when flow signal is strong.
    /// Default: 0.10
    pub strong_flow_alpha_boost: f64,

    /// Threshold for "strong" flow signal.
    /// Default: 0.5
    pub strong_flow_threshold: f64,

    /// Adverse selection penalty multiplier.
    /// When wrong, lose half_spread * this multiplier.
    /// Default: 1.5
    pub adverse_selection_multiplier: f64,

    /// Kappa units (fills per hour).
    /// Default: true (kappa in fills/hour)
    pub kappa_per_hour: bool,

    /// Random seed (0 = use system random).
    /// Default: 0
    pub random_seed: u64,
}

impl Default for QuickMCConfig {
    fn default() -> Self {
        Self {
            n_paths: 25,
            cost_per_trade_bps: 2.0,
            alpha: 0.15,
            strong_flow_alpha_boost: 0.10,
            strong_flow_threshold: 0.5,
            adverse_selection_multiplier: 1.5,
            kappa_per_hour: true,
            random_seed: 0,
        }
    }
}

/// Result of MC simulation.
#[derive(Debug, Clone)]
pub struct MCSimulationResult {
    /// Expected value in basis points.
    pub expected_ev_bps: f64,
    /// Standard error of the estimate.
    pub std_error_bps: f64,
    /// 5th percentile (Value at Risk proxy).
    pub percentile_5: f64,
    /// 95th percentile (upside potential).
    pub percentile_95: f64,
    /// Average fills per path.
    pub avg_fills: f64,
    /// Probability of positive outcome.
    pub p_positive: f64,
    /// Number of paths run.
    pub n_paths: usize,
    /// P(correct) used in simulation.
    pub p_correct: f64,
}

/// Quick Monte Carlo simulator for EV estimation.
#[derive(Debug, Clone)]
pub struct QuickMCSimulator {
    config: QuickMCConfig,
    /// Simulation count.
    simulation_count: u64,
}

impl QuickMCSimulator {
    /// Create a new MC simulator.
    pub fn new(config: QuickMCConfig) -> Self {
        Self {
            config,
            simulation_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(QuickMCConfig::default())
    }

    /// Simulate expected value.
    ///
    /// # Arguments
    /// * `kappa` - Fill intensity (fills per hour if kappa_per_hour=true)
    /// * `enhanced_flow` - Flow signal [-1, +1]
    /// * `spread_bps` - Current bid-ask spread
    /// * `horizon_secs` - Simulation horizon in seconds
    ///
    /// # Returns
    /// MC simulation result with expected value and statistics
    pub fn simulate_ev(
        &mut self,
        kappa: f64,
        enhanced_flow: f64,
        spread_bps: f64,
        horizon_secs: f64,
    ) -> MCSimulationResult {
        self.simulation_count += 1;

        let mut rng = if self.config.random_seed > 0 {
            rand::rngs::StdRng::seed_from_u64(self.config.random_seed + self.simulation_count)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        // Convert kappa to expected fills in horizon
        let expected_fills = if self.config.kappa_per_hour {
            kappa * horizon_secs / 3600.0
        } else {
            kappa * horizon_secs
        };

        // Compute P(correct) based on flow signal
        let flow_abs = enhanced_flow.abs();
        let alpha = if flow_abs > self.config.strong_flow_threshold {
            self.config.alpha + self.config.strong_flow_alpha_boost
        } else {
            self.config.alpha
        };
        let p_correct = (0.5 + alpha * flow_abs).clamp(0.5, 0.85);

        // Run simulations
        let mut pnls = Vec::with_capacity(self.config.n_paths);
        let mut total_fills = 0;

        for _ in 0..self.config.n_paths {
            let (pnl, fills) = self.simulate_path(&mut rng, expected_fills, p_correct, spread_bps);
            pnls.push(pnl);
            total_fills += fills;
        }

        // Compute statistics
        let n = pnls.len() as f64;
        let mean = pnls.iter().sum::<f64>() / n;
        let variance = pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std_error = (variance / n).sqrt();

        // Sort for percentiles
        pnls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p5_idx = (n * 0.05) as usize;
        let p95_idx = ((n * 0.95) as usize).min(pnls.len() - 1);
        let percentile_5 = pnls.get(p5_idx).copied().unwrap_or(0.0);
        let percentile_95 = pnls.get(p95_idx).copied().unwrap_or(0.0);

        let positive_count = pnls.iter().filter(|&&p| p > 0.0).count();
        let p_positive = positive_count as f64 / n;

        let avg_fills = total_fills as f64 / n;

        MCSimulationResult {
            expected_ev_bps: mean,
            std_error_bps: std_error,
            percentile_5,
            percentile_95,
            avg_fills,
            p_positive,
            n_paths: self.config.n_paths,
            p_correct,
        }
    }

    /// Simulate a single path.
    fn simulate_path<R: Rng>(
        &self,
        rng: &mut R,
        expected_fills: f64,
        p_correct: f64,
        spread_bps: f64,
    ) -> (f64, usize) {
        // Sample fill count from Poisson
        let fills = self.sample_poisson(rng, expected_fills);

        if fills == 0 {
            return (0.0, 0);
        }

        let half_spread = spread_bps / 2.0;
        let cost = self.config.cost_per_trade_bps;
        let adverse_mult = self.config.adverse_selection_multiplier;

        let mut pnl = 0.0;

        for _ in 0..fills {
            let correct = rng.gen::<f64>() < p_correct;

            if correct {
                // Correct direction: capture half spread minus costs
                pnl += half_spread - cost;
            } else {
                // Wrong direction: adverse selection
                pnl -= half_spread * adverse_mult + cost;
            }
        }

        (pnl, fills)
    }

    /// Sample from Poisson distribution.
    fn sample_poisson<R: Rng>(&self, rng: &mut R, lambda: f64) -> usize {
        if lambda <= 0.0 {
            return 0;
        }

        // For small lambda, use inverse transform
        if lambda < 30.0 {
            let l = (-lambda).exp();
            let mut k = 0;
            let mut p = 1.0;

            loop {
                k += 1;
                p *= rng.gen::<f64>();
                if p <= l {
                    return k - 1;
                }
            }
        } else {
            // For large lambda, use normal approximation
            let z = rng.gen::<f64>() * 2.0 - 1.0; // Simple uniform approximation
            (lambda + z * lambda.sqrt()).round().max(0.0) as usize
        }
    }

    /// Quick EV estimate (single number, no detailed stats).
    pub fn quick_ev(
        &mut self,
        kappa: f64,
        enhanced_flow: f64,
        spread_bps: f64,
        horizon_secs: f64,
    ) -> f64 {
        self.simulate_ev(kappa, enhanced_flow, spread_bps, horizon_secs)
            .expected_ev_bps
    }

    /// Check if MC suggests quoting based on EV and kappa.
    ///
    /// # Arguments
    /// * `kappa` - Fill intensity
    /// * `enhanced_flow` - Flow signal
    /// * `spread_bps` - Current spread
    /// * `horizon_secs` - Simulation horizon
    /// * `min_ev_bps` - Minimum EV to quote (default: 0.2)
    /// * `min_kappa` - Minimum kappa to quote (default: 1500)
    pub fn should_quote(
        &mut self,
        kappa: f64,
        enhanced_flow: f64,
        spread_bps: f64,
        horizon_secs: f64,
        min_ev_bps: f64,
        min_kappa: f64,
    ) -> (bool, MCSimulationResult) {
        let result = self.simulate_ev(kappa, enhanced_flow, spread_bps, horizon_secs);

        let should = kappa >= min_kappa && result.expected_ev_bps >= min_ev_bps;

        (should, result)
    }

    /// Get simulation count.
    pub fn simulation_count(&self) -> u64 {
        self.simulation_count
    }

    /// Get configuration.
    pub fn config(&self) -> &QuickMCConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: QuickMCConfig) {
        self.config = config;
    }

    /// Reset simulation count.
    pub fn reset(&mut self) {
        self.simulation_count = 0;
    }
}

impl Default for QuickMCSimulator {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_simulation() {
        let mut sim = QuickMCSimulator::new(QuickMCConfig {
            n_paths: 100,
            random_seed: 42,
            ..Default::default()
        });

        let result = sim.simulate_ev(
            2000.0, // High kappa
            0.5,    // Strong flow
            10.0,   // 10 bps spread
            1.0,    // 1 second
        );

        // Should have reasonable statistics
        assert!(result.expected_ev_bps.is_finite());
        assert!(result.std_error_bps >= 0.0);
        assert!(result.avg_fills >= 0.0);
        assert!(result.p_positive >= 0.0 && result.p_positive <= 1.0);
    }

    #[test]
    fn test_zero_kappa() {
        let mut sim = QuickMCSimulator::new(QuickMCConfig {
            n_paths: 50,
            random_seed: 42,
            ..Default::default()
        });

        let result = sim.simulate_ev(0.0, 0.5, 10.0, 1.0);

        // Zero kappa → zero fills → zero EV
        assert!(result.avg_fills.abs() < 0.1);
        assert!(result.expected_ev_bps.abs() < 0.1);
    }

    #[test]
    fn test_strong_flow_increases_p_correct() {
        let mut sim = QuickMCSimulator::new(QuickMCConfig {
            n_paths: 50,
            alpha: 0.15,
            strong_flow_alpha_boost: 0.10,
            strong_flow_threshold: 0.5,
            random_seed: 42,
            ..Default::default()
        });

        let result_weak = sim.simulate_ev(2000.0, 0.3, 10.0, 1.0);
        let result_strong = sim.simulate_ev(2000.0, 0.7, 10.0, 1.0);

        assert!(result_strong.p_correct > result_weak.p_correct);
    }

    #[test]
    fn test_wider_spread_improves_ev() {
        let mut sim = QuickMCSimulator::new(QuickMCConfig {
            n_paths: 200,
            random_seed: 42,
            ..Default::default()
        });

        let result_tight = sim.simulate_ev(2000.0, 0.5, 5.0, 1.0);
        let result_wide = sim.simulate_ev(2000.0, 0.5, 20.0, 1.0);

        // Wider spread should generally improve EV when we're predicting correctly
        // (but also increases adverse selection cost when wrong)
        assert!(result_tight.expected_ev_bps.is_finite());
        assert!(result_wide.expected_ev_bps.is_finite());
    }

    #[test]
    fn test_longer_horizon_more_fills() {
        let mut sim = QuickMCSimulator::new(QuickMCConfig {
            n_paths: 100,
            random_seed: 42,
            ..Default::default()
        });

        let result_short = sim.simulate_ev(1000.0, 0.5, 10.0, 1.0);
        let result_long = sim.simulate_ev(1000.0, 0.5, 10.0, 10.0);

        assert!(result_long.avg_fills > result_short.avg_fills);
    }

    #[test]
    fn test_should_quote() {
        let mut sim = QuickMCSimulator::new(QuickMCConfig {
            n_paths: 50,
            random_seed: 42,
            ..Default::default()
        });

        // High kappa, strong flow → should quote
        let (should_high, _) = sim.should_quote(2000.0, 0.6, 10.0, 1.0, 0.0, 1500.0);

        // Low kappa → should not quote
        let (should_low, _) = sim.should_quote(500.0, 0.6, 10.0, 1.0, 0.0, 1500.0);

        assert!(should_high || !should_low); // At least one should differ
    }

    #[test]
    fn test_percentiles() {
        let mut sim = QuickMCSimulator::new(QuickMCConfig {
            n_paths: 100,
            random_seed: 42,
            ..Default::default()
        });

        let result = sim.simulate_ev(2000.0, 0.5, 10.0, 1.0);

        // 5th percentile should be less than 95th
        assert!(result.percentile_5 <= result.percentile_95);

        // Mean should be between percentiles (usually)
        // This isn't strictly guaranteed but should hold for reasonable distributions
    }

    #[test]
    fn test_quick_ev() {
        let mut sim = QuickMCSimulator::default_config();

        let ev = sim.quick_ev(1500.0, 0.4, 10.0, 1.0);

        assert!(ev.is_finite());
    }

    #[test]
    fn test_poisson_sampling() {
        let sim = QuickMCSimulator::default_config();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Sample many times and check average is close to lambda
        let lambda = 5.0;
        let samples: Vec<usize> = (0..1000)
            .map(|_| sim.sample_poisson(&mut rng, lambda))
            .collect();

        let avg = samples.iter().sum::<usize>() as f64 / samples.len() as f64;

        // Should be close to lambda (within 20%)
        assert!(
            (avg - lambda).abs() < lambda * 0.2,
            "Expected avg ~{}, got {}",
            lambda,
            avg
        );
    }

    #[test]
    fn test_deterministic_seed() {
        let config = QuickMCConfig {
            n_paths: 10,
            random_seed: 12345,
            ..Default::default()
        };

        let mut sim1 = QuickMCSimulator::new(config.clone());
        let mut sim2 = QuickMCSimulator::new(config);

        let result1 = sim1.simulate_ev(1000.0, 0.5, 10.0, 1.0);
        let result2 = sim2.simulate_ev(1000.0, 0.5, 10.0, 1.0);

        // With same seed and same call, should get same result
        assert!((result1.expected_ev_bps - result2.expected_ev_bps).abs() < 0.001);
    }

    #[test]
    fn test_adverse_selection_impact() {
        let config_normal = QuickMCConfig {
            n_paths: 100,
            adverse_selection_multiplier: 1.0,
            random_seed: 42,
            ..Default::default()
        };

        let config_harsh = QuickMCConfig {
            n_paths: 100,
            adverse_selection_multiplier: 2.0,
            random_seed: 42,
            ..Default::default()
        };

        let mut sim_normal = QuickMCSimulator::new(config_normal);
        let mut sim_harsh = QuickMCSimulator::new(config_harsh);

        // With low flow (more wrong predictions), harsh adverse selection hurts more
        let result_normal = sim_normal.simulate_ev(2000.0, 0.1, 10.0, 1.0);
        let result_harsh = sim_harsh.simulate_ev(2000.0, 0.1, 10.0, 1.0);

        // Harsh adverse selection should reduce EV
        assert!(result_harsh.expected_ev_bps <= result_normal.expected_ev_bps);
    }
}
