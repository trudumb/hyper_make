//! RL Edge Model — wraps the Q-learning agent as an `EdgeModel` ensemble member.
//!
//! Converts `MarketState` into `MDPState` and queries the Q-table for edge predictions.
//! This replaces the separate RL override system with a unified ensemble pipeline.

use std::sync::{Arc, RwLock};

use super::ensemble::EdgeModel;
use super::rl_agent::{MDPState, QLearningAgent};
use super::types::MarketState;

/// Baseline sigma for vol_ratio bucketing (annualized 1% per-second vol).
const VOL_BASELINE_SIGMA: f64 = 0.01;

/// Default Hawkes branching ratio when not available from MarketState.
const DEFAULT_HAWKES_BRANCHING: f64 = 0.5;

/// Minimum observations before Q-value uncertainty drops below "very uncertain".
const MIN_OBS_FOR_MODERATE_CONFIDENCE: u64 = 5;

/// Uncertainty floor in bps (even well-observed states have irreducible noise).
const MIN_UNCERTAINTY_BPS: f64 = 0.5;

/// High uncertainty fallback in bps (cold agent or lock contention).
const HIGH_UNCERTAINTY_BPS: f64 = 5.0;

/// Wraps a shared `QLearningAgent` as an `EdgeModel` for ensemble integration.
pub struct RLEdgeModel {
    agent: Arc<RwLock<QLearningAgent>>,
}

impl RLEdgeModel {
    /// Create a new RLEdgeModel wrapping a shared Q-learning agent.
    pub fn new(agent: Arc<RwLock<QLearningAgent>>) -> Self {
        Self { agent }
    }

    /// Convert `MarketState` to `MDPState` for Q-table lookup.
    fn to_mdp_state(state: &MarketState) -> MDPState {
        let vol_ratio = if VOL_BASELINE_SIGMA > 0.0 {
            state.sigma_effective / VOL_BASELINE_SIGMA
        } else {
            1.0
        };

        MDPState::from_continuous(
            state.position,
            state.max_position,
            state.book_imbalance,
            vol_ratio,
            state.p_informed,
            DEFAULT_HAWKES_BRANCHING,
            state.momentum_bps,
        )
    }
}

impl EdgeModel for RLEdgeModel {
    fn predict_edge(&self, state: &MarketState) -> (f64, f64) {
        let Ok(mut agent) = self.agent.write() else {
            // Lock contention — return zero edge with high uncertainty
            return (0.0, HIGH_UNCERTAINTY_BPS);
        };

        let mdp_state = Self::to_mdp_state(state);
        let stats = agent.get_q_stats(&mdp_state);

        // Q-value mean is in reward units (bps). Use directly as edge prediction.
        let mean_bps = stats.best_q_mean;

        // Uncertainty: high when few observations, otherwise from Q-value posterior std.
        let std_bps = if stats.best_q_count < MIN_OBS_FOR_MODERATE_CONFIDENCE {
            HIGH_UNCERTAINTY_BPS
        } else {
            stats.best_q_std.max(MIN_UNCERTAINTY_BPS)
        };

        (mean_bps, std_bps)
    }

    fn name(&self) -> &str {
        "RL"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::learning::rl_agent::QLearningConfig;

    fn make_rl_edge_model() -> RLEdgeModel {
        let agent = QLearningAgent::new(QLearningConfig::default());
        RLEdgeModel::new(Arc::new(RwLock::new(agent)))
    }

    #[test]
    fn test_rl_edge_model_returns_valid_predictions() {
        let model = make_rl_edge_model();
        let state = MarketState::default();

        let (mean, std) = model.predict_edge(&state);

        // Mean should be finite
        assert!(mean.is_finite(), "mean should be finite, got {mean}");
        // Std should be positive and finite
        assert!(std > 0.0, "std should be positive, got {std}");
        assert!(std.is_finite(), "std should be finite, got {std}");
    }

    #[test]
    fn test_cold_agent_returns_high_uncertainty() {
        let model = make_rl_edge_model();
        let state = MarketState::default();

        let (_mean, std) = model.predict_edge(&state);

        // Cold agent (no training data) should return high uncertainty
        assert!(
            std >= HIGH_UNCERTAINTY_BPS,
            "cold agent std ({std}) should be >= {HIGH_UNCERTAINTY_BPS}"
        );
    }

    #[test]
    fn test_name_returns_rl() {
        let model = make_rl_edge_model();
        assert_eq!(model.name(), "RL");
    }

    #[test]
    fn test_to_mdp_state_conversion() {
        // Verify the MarketState -> MDPState conversion produces sensible results
        let state = MarketState {
            position: 0.5,
            max_position: 1.0,
            book_imbalance: 0.2,
            sigma_effective: 0.02, // 2x baseline -> High vol
            p_informed: 0.1,      // Low adverse
            momentum_bps: 10.0,   // Bullish drift
            ..Default::default()
        };

        let mdp = RLEdgeModel::to_mdp_state(&state);

        // Position 0.5 / 1.0 = 50% -> Long bucket
        assert_eq!(
            mdp.inventory,
            crate::market_maker::learning::rl_agent::InventoryBucket::Long
        );
        // Imbalance 0.2 -> Buy bucket
        assert_eq!(
            mdp.imbalance,
            crate::market_maker::learning::rl_agent::ImbalanceBucket::Buy
        );
        // Vol ratio 2.0 -> High bucket
        assert_eq!(
            mdp.volatility,
            crate::market_maker::learning::rl_agent::VolatilityBucket::High
        );
        // p_informed 0.1 -> Low adverse bucket
        assert_eq!(
            mdp.adverse,
            crate::market_maker::learning::rl_agent::AdverseBucket::Low
        );
        // Momentum 10.0 bps -> Bullish drift
        assert_eq!(
            mdp.drift,
            crate::market_maker::learning::rl_agent::DriftBucket::Bullish
        );
    }

    #[test]
    fn test_poisoned_lock_returns_fallback() {
        let agent = Arc::new(RwLock::new(QLearningAgent::new(QLearningConfig::default())));

        // Poison the lock by panicking inside a write guard
        let agent_clone = Arc::clone(&agent);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = agent_clone.write().unwrap();
            panic!("intentional panic to poison lock");
        }));
        assert!(result.is_err());

        let model = RLEdgeModel::new(agent);
        let state = MarketState::default();

        let (mean, std) = model.predict_edge(&state);
        assert_eq!(mean, 0.0, "poisoned lock should return zero mean");
        assert_eq!(
            std, HIGH_UNCERTAINTY_BPS,
            "poisoned lock should return high uncertainty"
        );
    }
}
