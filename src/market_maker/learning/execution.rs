//! Level 4: Execution Optimization
//!
//! Proper ladder optimization using utility maximization.
//! max E[PnL] - 0.5 × γ × Var[PnL] subject to constraints.

use smallvec::SmallVec;

use crate::market_maker::infra::capacity::LADDER_LEVEL_INLINE_CAPACITY;
use crate::market_maker::quoting::{Ladder, LadderLevel};
use crate::market_maker::strategy::MarketParams;

/// Configuration for execution optimization.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Risk aversion for utility function
    pub risk_aversion: f64,
    /// Fee rate in bps
    pub fee_bps: f64,
    /// Number of ladder levels
    pub num_levels: usize,
    /// Minimum depth in bps
    pub min_depth_bps: f64,
    /// Maximum depth in bps
    pub max_depth_bps: f64,
    /// Minimum size per level
    pub min_size: f64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            risk_aversion: 0.5,
            fee_bps: 1.5,
            num_levels: 5,
            min_depth_bps: 3.0,
            max_depth_bps: 50.0,
            min_size: 0.001,
        }
    }
}

/// Execution optimizer that finds utility-maximizing ladders.
pub struct ExecutionOptimizer {
    config: ExecutionConfig,
}

impl Default for ExecutionOptimizer {
    fn default() -> Self {
        Self::new(ExecutionConfig::default())
    }
}

impl ExecutionOptimizer {
    /// Create a new optimizer.
    pub fn new(config: ExecutionConfig) -> Self {
        Self { config }
    }

    /// Optimize ladder given decision parameters.
    ///
    /// Uses grid search over depth configurations to find
    /// utility-maximizing ladder.
    pub fn optimize_ladder(
        &self,
        params: &MarketParams,
        size_fraction: f64,
        expected_edge: f64,
    ) -> Ladder {
        // Grid of base depths to try
        let base_depths = [3.0, 5.0, 8.0, 10.0, 15.0];
        // Grid of depth ratios between levels
        let depth_ratios = [1.3, 1.5, 1.8, 2.0];

        let mut best_ladder = self.build_default_ladder(params, size_fraction);
        let mut best_utility = f64::NEG_INFINITY;

        for &base_depth in &base_depths {
            for &ratio in &depth_ratios {
                let candidate = self.build_ladder(params, base_depth, ratio, size_fraction);
                let utility = self.evaluate_utility(&candidate, params, expected_edge);

                if utility > best_utility {
                    best_utility = utility;
                    best_ladder = candidate;
                }
            }
        }

        best_ladder
    }

    /// Build a ladder with given parameters.
    fn build_ladder(
        &self,
        params: &MarketParams,
        base_depth_bps: f64,
        depth_ratio: f64,
        size_fraction: f64,
    ) -> Ladder {
        let mut bids: SmallVec<[LadderLevel; LADDER_LEVEL_INLINE_CAPACITY]> = SmallVec::new();
        let mut asks: SmallVec<[LadderLevel; LADDER_LEVEL_INLINE_CAPACITY]> = SmallVec::new();

        // Total size to distribute
        let total_size = params.dynamic_max_position * size_fraction;

        // Geometric size decay
        let size_decay: f64 = 0.6;
        let size_sum: f64 = (0..self.config.num_levels)
            .map(|i| size_decay.powi(i as i32))
            .sum();

        for i in 0..self.config.num_levels {
            // Depth increases geometrically
            let depth_bps = (base_depth_bps * depth_ratio.powi(i as i32))
                .clamp(self.config.min_depth_bps, self.config.max_depth_bps);

            // Size decreases geometrically
            let size =
                (total_size * size_decay.powi(i as i32) / size_sum).max(self.config.min_size);

            // Calculate prices
            let bid_price = params.microprice * (1.0 - depth_bps / 10000.0);
            let ask_price = params.microprice * (1.0 + depth_bps / 10000.0);

            bids.push(LadderLevel {
                price: bid_price,
                size,
                depth_bps,
            });

            asks.push(LadderLevel {
                price: ask_price,
                size,
                depth_bps,
            });
        }

        Ladder { bids, asks }
    }

    /// Build default ladder when optimization fails.
    fn build_default_ladder(&self, params: &MarketParams, size_fraction: f64) -> Ladder {
        self.build_ladder(params, 8.0, 1.5, size_fraction)
    }

    /// Evaluate utility of a ladder.
    ///
    /// Utility = E[PnL] - 0.5 × γ × Var[PnL]
    fn evaluate_utility(&self, ladder: &Ladder, params: &MarketParams, _expected_edge: f64) -> f64 {
        let expected_pnl = self.expected_pnl(ladder, params);
        let variance = self.pnl_variance(ladder, params);

        expected_pnl - 0.5 * self.config.risk_aversion * variance
    }

    /// Calculate expected P&L for a ladder.
    ///
    /// E[PnL] = Σ λ(δ) × [δ - AS(δ) - fees] × size
    fn expected_pnl(&self, ladder: &Ladder, params: &MarketParams) -> f64 {
        let mut total = 0.0;

        // Bid side
        for level in &ladder.bids {
            let fill_rate = self.fill_rate_at_depth(level.depth_bps, params);
            let spread_capture = level.depth_bps;
            let as_at_depth = self.as_at_depth(level.depth_bps, params);
            let edge = spread_capture - as_at_depth - self.config.fee_bps;
            total += fill_rate * edge * level.size;
        }

        // Ask side
        for level in &ladder.asks {
            let fill_rate = self.fill_rate_at_depth(level.depth_bps, params);
            let spread_capture = level.depth_bps;
            let as_at_depth = self.as_at_depth(level.depth_bps, params);
            let edge = spread_capture - as_at_depth - self.config.fee_bps;
            total += fill_rate * edge * level.size;
        }

        total
    }

    /// Calculate P&L variance for a ladder.
    ///
    /// Note: Fills are correlated with adverse moves, so variance
    /// is higher than if fills were independent.
    fn pnl_variance(&self, ladder: &Ladder, params: &MarketParams) -> f64 {
        let mut total_variance = 0.0;

        // Simple model: variance from each level is fill_rate × edge_variance × size²
        let edge_variance = params.sigma_effective.powi(2) * 10000.0 * 10000.0;

        for level in ladder.bids.iter().chain(ladder.asks.iter()) {
            let fill_rate = self.fill_rate_at_depth(level.depth_bps, params);
            total_variance += fill_rate * edge_variance * level.size.powi(2);
        }

        // Add covariance between levels (fills are correlated)
        // Approximation: add 20% for cross-level correlation
        total_variance *= 1.2;

        total_variance
    }

    /// Estimate fill rate at given depth.
    fn fill_rate_at_depth(&self, depth_bps: f64, params: &MarketParams) -> f64 {
        // λ(δ) = κ × min(1, (σ×√τ / δ)²)
        let depth_frac = depth_bps / 10000.0;
        if depth_frac < 0.0001 {
            return 1.0;
        }

        let expected_move = params.sigma_effective * params.kelly_time_horizon.sqrt();
        let fill_prob = (expected_move / depth_frac).powi(2).min(1.0);

        fill_prob * params.kappa.min(10.0)
    }

    /// Estimate adverse selection at given depth.
    fn as_at_depth(&self, depth_bps: f64, params: &MarketParams) -> f64 {
        // AS typically decreases with depth (further from touch = less informed)
        // Use depth decay from params if available
        let base_as = params.total_as_bps;

        // Get characteristic depth from calibrated model if available
        let delta_char = params
            .depth_decay_as
            .as_ref()
            .map(|d| d.delta_char_bps)
            .unwrap_or(10.0); // Default 10 bps characteristic depth

        // AS(δ) = AS_base × exp(-δ / δ_char)
        base_as * (-depth_bps / delta_char).exp()
    }

    /// Get config reference.
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    /// Update config.
    pub fn set_config(&mut self, config: ExecutionConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> MarketParams {
        MarketParams {
            microprice: 100.0,
            market_mid: 100.0,
            sigma: 0.01,
            sigma_effective: 0.01,
            kappa: 1.0,
            kelly_time_horizon: 1.0,
            dynamic_max_position: 1.0,
            total_as_bps: 2.0,
            depth_decay_as: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_build_ladder() {
        let optimizer = ExecutionOptimizer::default();
        let params = default_params();

        let ladder = optimizer.build_ladder(&params, 8.0, 1.5, 1.0);

        assert_eq!(ladder.bids.len(), 5);
        assert_eq!(ladder.asks.len(), 5);

        // Check depths are increasing
        for i in 1..ladder.bids.len() {
            assert!(ladder.bids[i].depth_bps > ladder.bids[i - 1].depth_bps);
        }

        // Check sizes are decreasing
        for i in 1..ladder.bids.len() {
            assert!(ladder.bids[i].size <= ladder.bids[i - 1].size);
        }
    }

    #[test]
    fn test_expected_pnl() {
        let optimizer = ExecutionOptimizer::default();
        let params = default_params();

        let ladder = optimizer.build_ladder(&params, 8.0, 1.5, 1.0);
        let expected_pnl = optimizer.expected_pnl(&ladder, &params);

        // With 8bp depth, 2bp AS, 1.5bp fees → ~4.5bp edge
        // Should be positive
        assert!(
            expected_pnl > 0.0,
            "Expected positive P&L, got {expected_pnl}"
        );
    }

    #[test]
    fn test_optimize_ladder() {
        let optimizer = ExecutionOptimizer::default();
        let params = default_params();

        let ladder = optimizer.optimize_ladder(&params, 0.5, 3.0);

        // Should produce valid ladder
        assert!(!ladder.bids.is_empty());
        assert!(!ladder.asks.is_empty());
    }

    #[test]
    fn test_fill_rate_decreases_with_depth() {
        let optimizer = ExecutionOptimizer::default();
        // Use lower sigma so depths 50-200 bps show decay
        let params = MarketParams {
            sigma_effective: 0.001, // 0.1% - much lower
            kappa: 1.0,
            kelly_time_horizon: 1.0,
            ..Default::default()
        };

        // With sigma=0.1% and depth in bps: fill_prob = (sigma / depth_frac)^2
        // At 50 bps: (0.001 / 0.005)^2 = 0.04
        // At 100 bps: (0.001 / 0.01)^2 = 0.01
        // At 200 bps: (0.001 / 0.02)^2 = 0.0025
        let rate_50bps = optimizer.fill_rate_at_depth(50.0, &params);
        let rate_100bps = optimizer.fill_rate_at_depth(100.0, &params);
        let rate_200bps = optimizer.fill_rate_at_depth(200.0, &params);

        assert!(
            rate_50bps > rate_100bps,
            "Fill rate should decrease with depth: {rate_50bps} vs {rate_100bps}"
        );
        assert!(
            rate_100bps > rate_200bps,
            "Fill rate should decrease with depth: {rate_100bps} vs {rate_200bps}"
        );
    }

    #[test]
    fn test_as_decreases_with_depth() {
        let optimizer = ExecutionOptimizer::default();
        let params = default_params();

        let as_5bps = optimizer.as_at_depth(5.0, &params);
        let as_10bps = optimizer.as_at_depth(10.0, &params);
        let as_20bps = optimizer.as_at_depth(20.0, &params);

        assert!(as_5bps > as_10bps, "AS should decrease with depth");
        assert!(as_10bps > as_20bps, "AS should decrease with depth");
    }
}
