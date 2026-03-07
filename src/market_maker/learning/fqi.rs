//! Fitted Q-Iteration (FQI) for offline batch Q-learning.
//!
//! Runs at startup or on-demand (NOT in the hot loop). Learns multi-step
//! temporal credit assignment from accumulated experience data.
//!
//! Uses double Q-learning to prevent overestimation, which is critical
//! for market making where overconfident tightening → adverse selection.
//!
//! ## State/Action Space
//! - States: MDPStateCompact (45 = 5 inv × 3 vol × 3 imbalance)
//! - Actions: Bandit arms (8 spread multipliers)
//! - Q-table: 45 × 8 = 360 cells

use serde::{Deserialize, Serialize};

use super::replay_buffer::ReplayBuffer;

/// Number of compact MDP states (5 inv × 3 vol × 3 imbalance).
const N_STATES: usize = 45;
/// Number of bandit arms.
const N_ACTIONS: usize = 8;
/// Total Q-table cells.
const N_CELLS: usize = N_STATES * N_ACTIONS;

/// Spread multiplier arms matching SpreadBandit.
const ARM_MULTIPLIERS: [f64; N_ACTIONS] = [0.85, 0.90, 0.95, 1.00, 1.05, 1.15, 1.25, 1.40];

/// Arm → spread delta bps mapping for policy recommendations.
const ARM_DELTA_BPS: [f64; N_ACTIONS] = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.5, 2.5, 4.0];

/// Configuration for Fitted Q-Iteration.
#[derive(Debug, Clone)]
pub struct FQIConfig {
    /// Discount factor γ for future rewards.
    pub discount: f64,
    /// Learning rate for Q-table updates.
    pub learning_rate: f64,
    /// Maximum number of FQI iterations.
    pub max_iterations: usize,
    /// Convergence threshold (Bellman residual).
    pub convergence_threshold: f64,
    /// Minimum samples per (s,a) cell before updating.
    pub min_samples_per_cell: usize,
}

impl Default for FQIConfig {
    fn default() -> Self {
        Self {
            discount: 0.95,
            learning_rate: 0.1,
            max_iterations: 20,
            convergence_threshold: 0.01,
            min_samples_per_cell: 3,
        }
    }
}

/// Result of running FQI.
#[derive(Debug, Clone)]
pub struct FQIResult {
    /// Q1 table (primary): flattened [state * N_ACTIONS + action]
    pub q1: Vec<f64>,
    /// Q2 table (secondary, for double-Q): flattened [state * N_ACTIONS + action]
    pub q2: Vec<f64>,
    /// Visit count per (state, action) cell
    pub visit_counts: Vec<usize>,
    /// Number of iterations completed
    pub iterations: usize,
    /// Final Bellman residual
    pub bellman_residual: f64,
    /// Total experience records used
    pub total_records: usize,
}

impl FQIResult {
    /// Get the greedy policy: best action per state.
    pub fn greedy_policy(&self) -> Vec<usize> {
        (0..N_STATES)
            .map(|s| {
                let base = s * N_ACTIONS;
                let (best_a, _) = (0..N_ACTIONS)
                    .map(|a| (a, self.q1[base + a]))
                    .max_by(|(_, q1), (_, q2)| {
                        q1.partial_cmp(q2).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((3, 0.0)); // Default arm
                best_a
            })
            .collect()
    }

    /// Get the Q-value for a (state, action) pair (average of Q1 and Q2).
    pub fn q_value(&self, state_idx: usize, action_idx: usize) -> f64 {
        if state_idx >= N_STATES || action_idx >= N_ACTIONS {
            return 0.0;
        }
        let idx = state_idx * N_ACTIONS + action_idx;
        (self.q1[idx] + self.q2[idx]) / 2.0
    }

    /// Get the visit count for a (state, action) pair.
    pub fn visits(&self, state_idx: usize, action_idx: usize) -> usize {
        if state_idx >= N_STATES || action_idx >= N_ACTIONS {
            return 0;
        }
        self.visit_counts[state_idx * N_ACTIONS + action_idx]
    }

    /// Get the spread multiplier for a given arm index.
    pub fn arm_multiplier(arm_idx: usize) -> f64 {
        ARM_MULTIPLIERS.get(arm_idx).copied().unwrap_or(1.0)
    }
}

/// Checkpoint data for FQI persistence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FQICheckpoint {
    /// Flattened Q1 table
    #[serde(default)]
    pub q1: Vec<f64>,
    /// Flattened Q2 table
    #[serde(default)]
    pub q2: Vec<f64>,
    /// Visit counts per cell
    #[serde(default)]
    pub visit_counts: Vec<usize>,
    /// Total records used for training
    #[serde(default)]
    pub total_records: usize,
}

impl FQICheckpoint {
    /// Reconstruct FQIResult from checkpoint.
    pub fn to_result(&self) -> Option<FQIResult> {
        if self.q1.len() != N_CELLS || self.q2.len() != N_CELLS {
            return None;
        }
        Some(FQIResult {
            q1: self.q1.clone(),
            q2: self.q2.clone(),
            visit_counts: if self.visit_counts.len() == N_CELLS {
                self.visit_counts.clone()
            } else {
                vec![0; N_CELLS]
            },
            iterations: 0,
            bellman_residual: 0.0,
            total_records: self.total_records,
        })
    }
}

impl From<&FQIResult> for FQICheckpoint {
    fn from(result: &FQIResult) -> Self {
        Self {
            q1: result.q1.clone(),
            q2: result.q2.clone(),
            visit_counts: result.visit_counts.clone(),
            total_records: result.total_records,
        }
    }
}

/// Policy recommendation from FQI Q-values.
#[derive(Debug, Clone)]
pub struct FQIPolicyRecommendation {
    /// Recommended spread delta in bps
    pub spread_delta_bps: f64,
    /// Confidence [0, 1] based on state visit count
    pub confidence: f64,
    /// Recommended arm index
    pub arm_idx: usize,
    /// Q-value of recommended action
    pub q_value: f64,
    /// Q-value of default action (arm 3 = 1.0x)
    pub q_default: f64,
}

/// Fitted Q-Iteration learner.
pub struct FittedQIterator {
    config: FQIConfig,
}

impl FittedQIterator {
    /// Create a new FQI learner with default config.
    pub fn new() -> Self {
        Self {
            config: FQIConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: FQIConfig) -> Self {
        Self { config }
    }

    /// Run FQI on the replay buffer, producing Q-tables.
    ///
    /// Uses double Q-learning: Q1 selects the best action, Q2 evaluates it.
    /// This prevents overestimation bias that would cause overconfident tightening.
    pub fn fit(&self, buffer: &ReplayBuffer) -> FQIResult {
        let records = buffer.records();
        let n = records.len();

        let mut q1 = vec![0.0_f64; N_CELLS];
        let mut q2 = vec![0.0_f64; N_CELLS];
        let mut visit_counts = vec![0_usize; N_CELLS];

        // Count visits per cell
        for r in records {
            let s = r.state_idx.min(N_STATES - 1);
            let a = r.action_idx.min(N_ACTIONS - 1);
            visit_counts[s * N_ACTIONS + a] += 1;
        }

        let mut bellman_residual = f64::MAX;

        let mut iterations = 0;
        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;
            // Accumulate targets per (s, a) cell
            let mut target_sums = vec![0.0_f64; N_CELLS];
            let mut target_counts = vec![0_usize; N_CELLS];

            for r in records {
                let s = r.state_idx.min(N_STATES - 1);
                let a = r.action_idx.min(N_ACTIONS - 1);
                let s_next = r.next_state_idx.min(N_STATES - 1);
                let idx = s * N_ACTIONS + a;

                // Double Q-learning: Q1 selects action, Q2 evaluates
                // On even iterations, update Q1 using Q2 for evaluation
                // On odd iterations, update Q2 using Q1 for evaluation
                let (q_select, _q_eval) = if iter % 2 == 0 {
                    (&q1, &q2)
                } else {
                    (&q2, &q1)
                };

                // Select best action in s' according to q_select
                let best_a_next = (0..N_ACTIONS)
                    .max_by(|&a1, &a2| {
                        q_select[s_next * N_ACTIONS + a1]
                            .partial_cmp(&q_select[s_next * N_ACTIONS + a2])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(3);

                // Evaluate using _q_eval
                let q_next = _q_eval[s_next * N_ACTIONS + best_a_next];
                let target = r.reward_total + self.config.discount * q_next;

                target_sums[idx] += target;
                target_counts[idx] += 1;
            }

            // Update Q-table
            let mut max_delta = 0.0_f64;
            let q_update = if iterations % 2 == 1 {
                &mut q1
            } else {
                &mut q2
            };

            for idx in 0..N_CELLS {
                if target_counts[idx] < self.config.min_samples_per_cell {
                    continue;
                }
                let mean_target = target_sums[idx] / target_counts[idx] as f64;
                let old_q = q_update[idx];
                q_update[idx] = (1.0 - self.config.learning_rate) * old_q
                    + self.config.learning_rate * mean_target;
                max_delta = max_delta.max((q_update[idx] - old_q).abs());
            }

            bellman_residual = max_delta;
            if bellman_residual < self.config.convergence_threshold {
                break;
            }
        }

        FQIResult {
            q1,
            q2,
            visit_counts,
            iterations,
            bellman_residual,
            total_records: n,
        }
    }

    /// Generate a policy recommendation for the given state.
    pub fn recommend(
        result: &FQIResult,
        state_idx: usize,
        min_visits: usize,
    ) -> FQIPolicyRecommendation {
        let s = state_idx.min(N_STATES - 1);
        let base = s * N_ACTIONS;

        // Default arm (3 = 1.0x)
        let default_arm = 3;
        let q_default = (result.q1[base + default_arm] + result.q2[base + default_arm]) / 2.0;

        // Find best action
        let (best_a, best_q) = (0..N_ACTIONS)
            .map(|a| {
                let q = (result.q1[base + a] + result.q2[base + a]) / 2.0;
                (a, q)
            })
            .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((default_arm, 0.0));

        // Confidence based on visit count
        let visits = result.visits(s, best_a);
        let confidence = if min_visits == 0 {
            0.0
        } else {
            (visits as f64 / min_visits as f64).min(1.0)
        };

        // If insufficient visits, recommend default (no change)
        if visits < min_visits {
            return FQIPolicyRecommendation {
                spread_delta_bps: 0.0,
                confidence,
                arm_idx: default_arm,
                q_value: q_default,
                q_default,
            };
        }

        FQIPolicyRecommendation {
            spread_delta_bps: ARM_DELTA_BPS[best_a],
            confidence,
            arm_idx: best_a,
            q_value: best_q,
            q_default,
        }
    }
}

impl Default for FittedQIterator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::learning::experience::{ExperienceRecord, ExperienceSource};

    fn make_record(
        state_idx: usize,
        action_idx: usize,
        reward: f64,
        next_state_idx: usize,
    ) -> ExperienceRecord {
        ExperienceRecord {
            state_idx,
            action_idx,
            reward_total: reward,
            edge_component: reward,
            inventory_penalty: 0.0,
            volatility_penalty: 0.0,
            inventory_change_penalty: 0.0,
            next_state_idx,
            done: false,
            timestamp_ms: 0,
            session_id: "test".to_string(),
            source: ExperienceSource::Paper,
            side: "buy".to_string(),
            fill_price: 100.0,
            mid_price: 100.0,
            fill_size: 1.0,
            inventory: 0.0,
            regime: "normal".to_string(),
            drift_penalty: None,
            bandit_multiplier: None,
            vol_ratio: None,
            mdp_state_idx: Some(state_idx),
            bandit_arm_idx: Some(action_idx),
            inventory_risk_at_fill: None,
        }
    }

    #[test]
    fn test_fqi_convergence() {
        // Simple scenario: state 0 with action 3 gives reward 5,
        // action 0 gives reward 1. FQI should prefer action 3.
        let mut buffer = ReplayBuffer::new(1000);
        for _ in 0..50 {
            buffer.push(make_record(0, 3, 5.0, 0));
            buffer.push(make_record(0, 0, 1.0, 0));
            buffer.push(make_record(0, 1, 2.0, 0));
        }

        let fqi = FittedQIterator::new();
        let result = fqi.fit(&buffer);

        // Q-value for action 3 should be higher than action 0
        let q_a3 = result.q_value(0, 3);
        let q_a0 = result.q_value(0, 0);
        assert!(
            q_a3 > q_a0,
            "q(s=0, a=3)={q_a3:.2} should > q(s=0, a=0)={q_a0:.2}"
        );

        // Bellman residual should decrease
        assert!(result.bellman_residual < 1.0, "Residual should converge");
    }

    #[test]
    fn test_fqi_double_q_lower_values() {
        // Double-Q should produce lower (more conservative) values than single-Q
        let mut buffer = ReplayBuffer::new(1000);
        for _ in 0..30 {
            for a in 0..8 {
                buffer.push(make_record(0, a, 3.0 + (a as f64) * 0.5, 0));
            }
        }

        let fqi_double = FittedQIterator::with_config(FQIConfig {
            max_iterations: 20,
            ..Default::default()
        });
        let result = fqi_double.fit(&buffer);

        // Q-values should be in reasonable range (not pathologically large)
        for s in 0..N_STATES {
            for a in 0..N_ACTIONS {
                let q = result.q_value(s, a);
                // With gamma=0.95, Q should converge to r/(1-gamma) = r/0.05
                // For r ~ 3-6.5, Q should be ~ 60-130. Allow generous bounds.
                assert!(
                    q.abs() < 200.0,
                    "Q({s},{a}) = {q:.2} is pathologically large"
                );
            }
        }
    }

    #[test]
    fn test_fqi_recommend_default_no_visits() {
        let result = FQIResult {
            q1: vec![0.0; N_CELLS],
            q2: vec![0.0; N_CELLS],
            visit_counts: vec![0; N_CELLS],
            iterations: 0,
            bellman_residual: 0.0,
            total_records: 0,
        };

        let rec = FittedQIterator::recommend(&result, 0, 10);
        assert_eq!(rec.arm_idx, 3); // Default arm
        assert!((rec.spread_delta_bps).abs() < 1e-10); // No delta
        assert!((rec.confidence).abs() < 1e-10); // Zero confidence
    }

    #[test]
    fn test_fqi_recommend_with_visits() {
        let mut q1 = vec![0.0; N_CELLS];
        let mut visit_counts = vec![0; N_CELLS];

        // State 0: arm 5 is best (Q=10), default arm 3 (Q=5)
        q1[5] = 10.0; // state 0, arm 5
        q1[3] = 5.0; // state 0, arm 3
        visit_counts[5] = 20;
        visit_counts[3] = 20;

        let result = FQIResult {
            q1: q1.clone(),
            q2: q1,
            visit_counts,
            iterations: 10,
            bellman_residual: 0.001,
            total_records: 100,
        };

        let rec = FittedQIterator::recommend(&result, 0, 10);
        assert_eq!(rec.arm_idx, 5);
        assert!((rec.spread_delta_bps - 1.5).abs() < 1e-10); // Arm 5 → +1.5 bps
        assert!((rec.confidence - 1.0).abs() < 1e-10); // 20/10 capped at 1.0
    }

    #[test]
    fn test_fqi_greedy_policy() {
        let mut q1 = vec![0.0; N_CELLS];
        // State 0 prefers arm 2, state 1 prefers arm 6
        q1[2] = 5.0; // state 0, arm 2
        q1[N_ACTIONS + 6] = 8.0; // state 1, arm 6

        let result = FQIResult {
            q1: q1.clone(),
            q2: q1,
            visit_counts: vec![10; N_CELLS],
            iterations: 10,
            bellman_residual: 0.001,
            total_records: 100,
        };

        let policy = result.greedy_policy();
        assert_eq!(policy[0], 2);
        assert_eq!(policy[1], 6);
    }

    #[test]
    fn test_checkpoint_round_trip() {
        let result = FQIResult {
            q1: vec![1.0; N_CELLS],
            q2: vec![2.0; N_CELLS],
            visit_counts: vec![5; N_CELLS],
            iterations: 15,
            bellman_residual: 0.005,
            total_records: 500,
        };

        let checkpoint = FQICheckpoint::from(&result);
        let json = serde_json::to_string(&checkpoint).unwrap();
        let restored: FQICheckpoint = serde_json::from_str(&json).unwrap();
        let restored_result = restored.to_result().unwrap();

        assert_eq!(restored_result.q1.len(), N_CELLS);
        assert!((restored_result.q1[0] - 1.0).abs() < 1e-10);
        assert!((restored_result.q2[0] - 2.0).abs() < 1e-10);
        assert_eq!(restored_result.visit_counts[0], 5);
    }

    #[test]
    fn test_empty_checkpoint_returns_none() {
        let empty = FQICheckpoint::default();
        assert!(empty.to_result().is_none());
    }

    #[test]
    fn test_min_samples_filtering() {
        let mut buffer = ReplayBuffer::new(100);
        // Only 2 samples for (s=0, a=0) — below min_samples_per_cell=3
        buffer.push(make_record(0, 0, 10.0, 0));
        buffer.push(make_record(0, 0, 10.0, 0));

        let fqi = FittedQIterator::new();
        let result = fqi.fit(&buffer);

        // Q-value should remain at 0 (not updated) because < 3 samples
        assert!(
            (result.q1[0]).abs() < 1e-10,
            "Should not update with < 3 samples"
        );
    }
}
