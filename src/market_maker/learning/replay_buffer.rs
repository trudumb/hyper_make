//! Replay buffer for offline learning from accumulated experience.
//!
//! Loads JSONL experience files into a structured buffer with sampling
//! and statistics for Fitted Q-Iteration.

use std::collections::HashMap;
use std::path::Path;

use super::experience::ExperienceRecord;

/// Replay buffer holding experience records for offline learning.
pub struct ReplayBuffer {
    records: Vec<ExperienceRecord>,
    capacity: usize,
}

/// Aggregate statistics over the replay buffer.
#[derive(Debug, Clone)]
pub struct ReplayStatistics {
    /// Total number of records in the buffer
    pub total_records: usize,
    /// Mean reward across all records
    pub mean_reward: f64,
    /// Standard deviation of reward
    pub std_reward: f64,
    /// Visit count per state index
    pub state_distribution: HashMap<usize, usize>,
    /// Visit count per action index
    pub action_distribution: HashMap<usize, usize>,
    /// Mean reward per state index
    pub reward_by_state: HashMap<usize, f64>,
    /// Mean reward per action index
    pub reward_by_action: HashMap<usize, f64>,
}

impl ReplayBuffer {
    /// Create an empty buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            records: Vec::with_capacity(capacity.min(10_000)),
            capacity,
        }
    }

    /// Load experience records from all JSONL files in a directory.
    ///
    /// Corrupt lines are skipped with a warning, not a crash.
    /// Returns the number of records loaded.
    pub fn load_from_dir(&mut self, dir: &Path) -> std::io::Result<usize> {
        use std::io::BufRead;
        let mut loaded = 0;

        if !dir.exists() {
            return Ok(0);
        }

        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let file = std::fs::File::open(entry.path())?;
            let reader = std::io::BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<ExperienceRecord>(&line) {
                    Ok(record) => {
                        self.push(record);
                        loaded += 1;
                    }
                    Err(e) => {
                        tracing::warn!(
                            file = %entry.path().display(),
                            error = %e,
                            "Skipping corrupt experience record"
                        );
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Push a single record, evicting the oldest if at capacity.
    pub fn push(&mut self, record: ExperienceRecord) {
        if self.records.len() >= self.capacity {
            self.records.remove(0);
        }
        self.records.push(record);
    }

    /// Number of records currently stored.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get all records as a slice.
    pub fn records(&self) -> &[ExperienceRecord] {
        &self.records
    }

    /// Sample a uniform random batch of the given size.
    ///
    /// Returns up to `batch_size` records (fewer if buffer is smaller).
    /// Uses a deterministic seed for reproducibility in tests.
    pub fn sample_batch(&self, batch_size: usize, seed: u64) -> Vec<&ExperienceRecord> {
        if self.records.is_empty() {
            return Vec::new();
        }
        let n = batch_size.min(self.records.len());
        // Simple LCG-based sampling for determinism without pulling in rand
        let mut rng_state = seed;
        let mut indices: Vec<usize> = (0..self.records.len()).collect();
        // Fisher-Yates shuffle (partial, only first n)
        for i in 0..n {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = i + (rng_state as usize % (self.records.len() - i));
            indices.swap(i, j);
        }
        indices.truncate(n);
        indices.iter().map(|&i| &self.records[i]).collect()
    }

    /// Compute aggregate statistics over the buffer.
    pub fn statistics(&self) -> ReplayStatistics {
        if self.records.is_empty() {
            return ReplayStatistics {
                total_records: 0,
                mean_reward: 0.0,
                std_reward: 0.0,
                state_distribution: HashMap::new(),
                action_distribution: HashMap::new(),
                reward_by_state: HashMap::new(),
                reward_by_action: HashMap::new(),
            };
        }

        let n = self.records.len() as f64;
        let sum_r: f64 = self.records.iter().map(|r| r.reward_total).sum();
        let mean_r = sum_r / n;
        let var_r: f64 = self
            .records
            .iter()
            .map(|r| (r.reward_total - mean_r).powi(2))
            .sum::<f64>()
            / n;
        let std_r = var_r.sqrt();

        let mut state_dist: HashMap<usize, usize> = HashMap::new();
        let mut action_dist: HashMap<usize, usize> = HashMap::new();
        let mut state_rewards: HashMap<usize, (f64, usize)> = HashMap::new();
        let mut action_rewards: HashMap<usize, (f64, usize)> = HashMap::new();

        for r in &self.records {
            *state_dist.entry(r.state_idx).or_insert(0) += 1;
            *action_dist.entry(r.action_idx).or_insert(0) += 1;

            let se = state_rewards.entry(r.state_idx).or_insert((0.0, 0));
            se.0 += r.reward_total;
            se.1 += 1;

            let ae = action_rewards.entry(r.action_idx).or_insert((0.0, 0));
            ae.0 += r.reward_total;
            ae.1 += 1;
        }

        let reward_by_state: HashMap<usize, f64> = state_rewards
            .into_iter()
            .map(|(k, (sum, count))| (k, sum / count as f64))
            .collect();
        let reward_by_action: HashMap<usize, f64> = action_rewards
            .into_iter()
            .map(|(k, (sum, count))| (k, sum / count as f64))
            .collect();

        ReplayStatistics {
            total_records: self.records.len(),
            mean_reward: mean_r,
            std_reward: std_r,
            state_distribution: state_dist,
            action_distribution: action_dist,
            reward_by_state,
            reward_by_action,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::learning::experience::ExperienceSource;

    fn make_record(state_idx: usize, action_idx: usize, reward: f64) -> ExperienceRecord {
        ExperienceRecord {
            state_idx,
            action_idx,
            reward_total: reward,
            edge_component: reward,
            inventory_penalty: 0.0,
            volatility_penalty: 0.0,
            inventory_change_penalty: 0.0,
            next_state_idx: (state_idx + 1) % 45,
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
    fn test_push_and_capacity() {
        let mut buf = ReplayBuffer::new(3);
        buf.push(make_record(0, 0, 1.0));
        buf.push(make_record(1, 1, 2.0));
        buf.push(make_record(2, 2, 3.0));
        assert_eq!(buf.len(), 3);

        // Exceeding capacity evicts oldest
        buf.push(make_record(3, 3, 4.0));
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.records()[0].state_idx, 1); // Oldest (0) evicted
    }

    #[test]
    fn test_sample_batch() {
        let mut buf = ReplayBuffer::new(100);
        for i in 0..20 {
            buf.push(make_record(i % 45, i % 8, i as f64));
        }
        let batch = buf.sample_batch(5, 42);
        assert_eq!(batch.len(), 5);

        // Deterministic — same seed gives same result
        let batch2 = buf.sample_batch(5, 42);
        assert_eq!(
            batch.iter().map(|r| r.state_idx).collect::<Vec<_>>(),
            batch2.iter().map(|r| r.state_idx).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_sample_batch_larger_than_buffer() {
        let mut buf = ReplayBuffer::new(100);
        buf.push(make_record(0, 0, 1.0));
        buf.push(make_record(1, 1, 2.0));
        let batch = buf.sample_batch(10, 1);
        assert_eq!(batch.len(), 2); // Capped at buffer size
    }

    #[test]
    fn test_statistics() {
        let mut buf = ReplayBuffer::new(100);
        buf.push(make_record(0, 0, 1.0));
        buf.push(make_record(0, 1, 3.0));
        buf.push(make_record(1, 0, 5.0));

        let stats = buf.statistics();
        assert_eq!(stats.total_records, 3);
        assert!((stats.mean_reward - 3.0).abs() < 1e-10);
        assert_eq!(stats.state_distribution[&0], 2);
        assert_eq!(stats.state_distribution[&1], 1);
        assert_eq!(stats.action_distribution[&0], 2);
        assert_eq!(stats.action_distribution[&1], 1);
        // Mean reward for state 0 = (1+3)/2 = 2
        assert!((stats.reward_by_state[&0] - 2.0).abs() < 1e-10);
        // Mean reward for state 1 = 5
        assert!((stats.reward_by_state[&1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_buffer() {
        let buf = ReplayBuffer::new(100);
        assert!(buf.is_empty());
        let stats = buf.statistics();
        assert_eq!(stats.total_records, 0);
        assert!((stats.mean_reward).abs() < 1e-10);
        let batch = buf.sample_batch(10, 1);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_load_from_nonexistent_dir() {
        let mut buf = ReplayBuffer::new(100);
        let result = buf.load_from_dir(Path::new("/tmp/nonexistent_xq9z"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_load_from_dir_with_jsonl() {
        let dir = std::env::temp_dir().join("test_replay_buffer_load");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Write a JSONL file with 2 records + 1 corrupt line
        let record = make_record(5, 3, 2.5);
        let json = serde_json::to_string(&record).unwrap();
        let content = format!("{json}\nNOT_VALID_JSON\n{json}\n");
        std::fs::write(dir.join("test.jsonl"), content).unwrap();

        let mut buf = ReplayBuffer::new(100);
        let loaded = buf.load_from_dir(&dir).unwrap();
        assert_eq!(loaded, 2); // 2 valid, 1 skipped
        assert_eq!(buf.len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
