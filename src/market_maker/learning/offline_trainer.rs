//! Offline RL trainer for batch replay on logged experience data.
//!
//! Reads experience JSONL files and performs multi-epoch Q-learning
//! replay to train or refine RL policies without live trading.

use serde::{Deserialize, Serialize};

use super::experience::ExperienceRecord;
use super::rl_agent::{MDPAction, MDPState, QLearningAgent, QLearningConfig, Reward};
use crate::market_maker::checkpoint::types::RLCheckpoint;

/// Configuration for offline training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineTrainerConfig {
    /// Maximum number of training epochs
    #[serde(default = "default_max_epochs")]
    pub max_epochs: usize,
    /// Whether to shuffle experiences each epoch
    #[serde(default = "default_shuffle")]
    pub shuffle: bool,
    /// Discount factor for Q-learning
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// Convergence threshold: stop when mean-Q delta < this for patience epochs
    #[serde(default = "default_convergence_threshold")]
    pub convergence_threshold: f64,
    /// Number of consecutive epochs below threshold before stopping
    #[serde(default = "default_convergence_patience")]
    pub convergence_patience: usize,
}

fn default_max_epochs() -> usize { 10 }
fn default_shuffle() -> bool { true }
fn default_gamma() -> f64 { 0.95 }
fn default_convergence_threshold() -> f64 { 0.01 }
fn default_convergence_patience() -> usize { 3 }

impl Default for OfflineTrainerConfig {
    fn default() -> Self {
        Self {
            max_epochs: default_max_epochs(),
            shuffle: default_shuffle(),
            gamma: default_gamma(),
            convergence_threshold: default_convergence_threshold(),
            convergence_patience: default_convergence_patience(),
        }
    }
}

/// Metrics for a single training epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    /// Epoch number (0-indexed)
    pub epoch: usize,
    /// Average reward across all experiences in this epoch
    pub avg_reward: f64,
    /// Number of unique states visited
    pub states_visited: usize,
    /// Total Q-table updates performed this epoch
    pub total_updates: u64,
    /// Mean absolute change in Q-values vs previous epoch
    pub convergence_metric: f64,
}

/// Complete training history returned by the trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Number of epochs actually completed
    pub epochs_completed: usize,
    /// Per-epoch metrics
    pub epoch_metrics: Vec<EpochMetrics>,
    /// Whether training converged before max_epochs
    pub converged: bool,
    /// Final number of unique states with observations
    pub final_states_visited: usize,
    /// Total experiences processed (across all epochs)
    pub total_experiences_processed: u64,
}

/// Offline RL trainer that replays experience data for Q-learning.
pub struct OfflineTrainer {
    agent: QLearningAgent,
    config: OfflineTrainerConfig,
}

impl OfflineTrainer {
    /// Create a new offline trainer with the given config.
    pub fn new(config: OfflineTrainerConfig) -> Self {
        let ql_config = QLearningConfig {
            gamma: config.gamma,
            ..Default::default()
        };
        Self {
            agent: QLearningAgent::new(ql_config),
            config,
        }
    }

    /// Create a trainer initialized with a prior from an existing checkpoint.
    pub fn with_prior(config: OfflineTrainerConfig, checkpoint: &RLCheckpoint, weight: f64) -> Self {
        let mut trainer = Self::new(config);
        // Restore and then discount the prior
        let mut prior_agent = QLearningAgent::default();
        prior_agent.restore_from_checkpoint(checkpoint);
        let prior_q = prior_agent.export_q_table();
        trainer.agent.import_q_table_as_prior(&prior_q, weight);
        trainer
    }

    /// Run multi-epoch training on the given experience records.
    ///
    /// Returns a training history with per-epoch metrics and convergence info.
    pub fn train(&mut self, experiences: &[ExperienceRecord]) -> TrainingHistory {
        let mut epoch_metrics = Vec::with_capacity(self.config.max_epochs);
        let mut consecutive_below_threshold = 0;
        let mut converged = false;
        let mut total_experiences_processed: u64 = 0;

        // Pre-compute indices for shuffling
        let mut indices: Vec<usize> = (0..experiences.len()).collect();

        for epoch in 0..self.config.max_epochs {
            // Snapshot mean Q-values before this epoch for convergence check
            let pre_epoch_mean_q = self.mean_q_value();

            // Optional shuffle with epoch-based seed (deterministic per epoch)
            if self.config.shuffle {
                fisher_yates_shuffle(&mut indices, epoch as u64);
            }

            let mut epoch_reward_sum = 0.0;

            for &idx in &indices {
                let exp = &experiences[idx];

                let state = MDPState::from_index(exp.state_idx);
                let action = MDPAction::from_index(exp.action_idx);
                let reward = Reward {
                    total: exp.reward_total,
                    edge_component: exp.edge_component,
                    inventory_penalty: exp.inventory_penalty,
                    volatility_penalty: exp.volatility_penalty,
                    inventory_change_penalty: exp.inventory_change_penalty,
                };
                let next_state = MDPState::from_index(exp.next_state_idx);

                self.agent.update(state, action, reward, next_state, exp.done);
                epoch_reward_sum += exp.reward_total;
            }

            total_experiences_processed += experiences.len() as u64;

            // Compute epoch metrics
            let post_epoch_mean_q = self.mean_q_value();
            let convergence_metric = (post_epoch_mean_q - pre_epoch_mean_q).abs();
            let states_visited = self.agent.summary().states_visited;
            let avg_reward = if experiences.is_empty() {
                0.0
            } else {
                epoch_reward_sum / experiences.len() as f64
            };

            epoch_metrics.push(EpochMetrics {
                epoch,
                avg_reward,
                states_visited,
                total_updates: self.agent.total_updates(),
                convergence_metric,
            });

            // Check convergence
            if convergence_metric < self.config.convergence_threshold {
                consecutive_below_threshold += 1;
                if consecutive_below_threshold >= self.config.convergence_patience {
                    converged = true;
                    break;
                }
            } else {
                consecutive_below_threshold = 0;
            }
        }

        TrainingHistory {
            epochs_completed: epoch_metrics.len(),
            final_states_visited: self.agent.summary().states_visited,
            epoch_metrics,
            converged,
            total_experiences_processed,
        }
    }

    /// Export the trained Q-table as a checkpoint.
    pub fn to_checkpoint(&self) -> RLCheckpoint {
        self.agent.to_checkpoint()
    }

    /// Get a reference to the underlying agent for inspection.
    pub fn agent(&self) -> &QLearningAgent {
        &self.agent
    }

    /// Compute mean Q-value across all visited state-action pairs.
    fn mean_q_value(&mut self) -> f64 {
        let q_table = self.agent.export_q_table();
        let mut sum = 0.0;
        let mut count = 0u64;
        for actions in q_table.values() {
            for q_val in actions {
                if q_val.count() > 0 {
                    sum += q_val.mean();
                    count += 1;
                }
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }
}

/// Fisher-Yates shuffle with a simple deterministic seed.
///
/// Uses a basic LCG PRNG for reproducibility per epoch.
fn fisher_yates_shuffle(indices: &mut [usize], seed: u64) {
    let mut rng_state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

    for i in (1..indices.len()).rev() {
        // Generate next pseudo-random number
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
}

/// Read experience records from a JSONL file.
///
/// Skips lines that fail to parse (logs warning).
pub fn read_experience_file(path: &str) -> std::io::Result<Vec<ExperienceRecord>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut records = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<ExperienceRecord>(&line) {
            Ok(record) => records.push(record),
            Err(e) => {
                eprintln!("Warning: skipping line {} in {}: {}", line_num + 1, path, e);
            }
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::learning::experience::ExperienceSource;

    fn make_experience(state_idx: usize, action_idx: usize, reward: f64, next_state_idx: usize) -> ExperienceRecord {
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
            mid_price: 99.95,
            fill_size: 1.0,
            inventory: 0.0,
            regime: "normal".to_string(),
        }
    }

    #[test]
    fn test_offline_trainer_basic() {
        let config = OfflineTrainerConfig {
            max_epochs: 3,
            shuffle: false,
            ..Default::default()
        };

        let mut trainer = OfflineTrainer::new(config);

        // Create a small set of experiences
        let experiences = vec![
            make_experience(0, 0, 1.0, 1),
            make_experience(1, 2, -0.5, 0),
            make_experience(0, 1, 0.3, 2),
        ];

        let history = trainer.train(&experiences);

        assert_eq!(history.epochs_completed, 3);
        assert!(!history.converged);
        assert_eq!(history.epoch_metrics.len(), 3);
        assert!(history.final_states_visited > 0);
        assert_eq!(history.total_experiences_processed, 9); // 3 experiences × 3 epochs
    }

    #[test]
    fn test_offline_trainer_convergence() {
        let config = OfflineTrainerConfig {
            max_epochs: 100,
            shuffle: false,
            convergence_threshold: 0.001,
            convergence_patience: 3,
            ..Default::default()
        };

        let mut trainer = OfflineTrainer::new(config);

        // Create identical experiences — should converge quickly
        let experiences: Vec<_> = (0..50)
            .map(|_| make_experience(0, 0, 1.0, 0))
            .collect();

        let history = trainer.train(&experiences);

        assert!(history.converged, "Should converge with identical experiences");
        assert!(history.epochs_completed < 100);
    }

    #[test]
    fn test_offline_trainer_to_checkpoint() {
        let config = OfflineTrainerConfig {
            max_epochs: 2,
            shuffle: false,
            ..Default::default()
        };

        let mut trainer = OfflineTrainer::new(config);
        let experiences = vec![make_experience(0, 0, 1.0, 1)];
        trainer.train(&experiences);

        let checkpoint = trainer.to_checkpoint();
        assert!(!checkpoint.q_entries.is_empty());
        assert!(checkpoint.total_observations > 0);
    }

    #[test]
    fn test_offline_trainer_with_prior() {
        // First train to get a checkpoint
        let config = OfflineTrainerConfig {
            max_epochs: 2,
            shuffle: false,
            ..Default::default()
        };
        let mut trainer1 = OfflineTrainer::new(config.clone());
        let experiences = vec![make_experience(0, 0, 1.0, 1)];
        trainer1.train(&experiences);
        let checkpoint = trainer1.to_checkpoint();

        // Now train with that checkpoint as prior
        let mut trainer2 = OfflineTrainer::with_prior(config, &checkpoint, 0.3);
        let history = trainer2.train(&experiences);
        assert!(history.final_states_visited > 0);
    }

    #[test]
    fn test_fisher_yates_deterministic() {
        let mut a = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut b = a.clone();
        fisher_yates_shuffle(&mut a, 42);
        fisher_yates_shuffle(&mut b, 42);
        assert_eq!(a, b, "Same seed should produce same shuffle");

        let mut c = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        fisher_yates_shuffle(&mut c, 99);
        assert_ne!(a, c, "Different seeds should produce different shuffles");
    }

    #[test]
    fn test_epoch_metrics_tracked() {
        let config = OfflineTrainerConfig {
            max_epochs: 5,
            shuffle: false,
            ..Default::default()
        };

        let mut trainer = OfflineTrainer::new(config);
        let experiences = vec![
            make_experience(0, 0, 1.0, 1),
            make_experience(1, 1, -0.5, 2),
            make_experience(2, 2, 0.8, 0),
        ];

        let history = trainer.train(&experiences);

        for (i, metrics) in history.epoch_metrics.iter().enumerate() {
            assert_eq!(metrics.epoch, i);
            assert!(metrics.states_visited > 0);
        }
    }
}
