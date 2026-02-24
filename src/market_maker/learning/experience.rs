//! Experience logging for RL training data collection.
//!
//! Captures every (state, action, reward, next_state) tuple as structured JSONL
//! for offline training, replay analysis, and live streaming to training services.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::rl_agent::{MDPAction, MDPState, Reward};

/// Source of an experience record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperienceSource {
    /// Paper trading simulation
    Paper,
    /// Live trading
    Live,
}

impl std::fmt::Display for ExperienceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Paper => write!(f, "paper"),
            Self::Live => write!(f, "live"),
        }
    }
}

/// A single SARSA experience tuple for RL training.
///
/// ~400 bytes JSON per record. Designed for JSONL serialization
/// and offline batch replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceRecord {
    // === Core SARSA tuple ===
    /// Discretized state index (0..674)
    pub state_idx: usize,
    /// Action index (0..24)
    pub action_idx: usize,
    /// Total reward
    pub reward_total: f64,
    /// Edge component of reward
    pub edge_component: f64,
    /// Inventory penalty component
    pub inventory_penalty: f64,
    /// Volatility penalty component
    pub volatility_penalty: f64,
    /// Inventory change penalty component
    pub inventory_change_penalty: f64,
    /// Next state index after transition
    pub next_state_idx: usize,
    /// Whether this is a terminal state
    pub done: bool,

    // === Metadata ===
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
    /// Session identifier
    pub session_id: String,
    /// Source: paper or live
    pub source: ExperienceSource,
    /// Fill side (buy/sell)
    pub side: String,

    // === Market context ===
    /// Fill price
    pub fill_price: f64,
    /// Mid price at time of fill
    pub mid_price: f64,
    /// Fill size
    pub fill_size: f64,
    /// Current inventory position
    pub inventory: f64,
    /// Current market regime
    pub regime: String,
}

/// Parameters for constructing an `ExperienceRecord` from RL components.
pub struct ExperienceParams {
    /// Current MDP state
    pub state: MDPState,
    /// Action taken
    pub action: MDPAction,
    /// Reward received
    pub reward: Reward,
    /// Next MDP state after transition
    pub next_state: MDPState,
    /// Whether this is a terminal state
    pub done: bool,
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
    /// Session identifier
    pub session_id: String,
    /// Source: paper or live
    pub source: ExperienceSource,
    /// Fill side (buy/sell)
    pub side: String,
    /// Fill price
    pub fill_price: f64,
    /// Mid price at time of fill
    pub mid_price: f64,
    /// Fill size
    pub fill_size: f64,
    /// Current inventory position
    pub inventory: f64,
    /// Current market regime
    pub regime: String,
}

impl ExperienceRecord {
    /// Create a new experience record from RL components.
    pub fn from_params(params: ExperienceParams) -> Self {
        Self {
            state_idx: params.state.to_index(),
            action_idx: params.action.to_index(),
            reward_total: params.reward.total,
            edge_component: params.reward.edge_component,
            inventory_penalty: params.reward.inventory_penalty,
            volatility_penalty: params.reward.volatility_penalty,
            inventory_change_penalty: params.reward.inventory_change_penalty,
            next_state_idx: params.next_state.to_index(),
            done: params.done,
            timestamp_ms: params.timestamp_ms,
            session_id: params.session_id,
            source: params.source,
            side: params.side,
            fill_price: params.fill_price,
            mid_price: params.mid_price,
            fill_size: params.fill_size,
            inventory: params.inventory,
            regime: params.regime,
        }
    }
}

/// JSONL file writer for RL experience persistence.
///
/// Writes one JSON object per line in append mode. Follows the same
/// pattern as `AnalyticsLogger` in `analytics/persistence.rs`.
pub struct ExperienceLogger {
    writer: BufWriter<File>,
    /// Optional channel for live streaming to training service
    live_tx: Option<tokio::sync::mpsc::Sender<ExperienceRecord>>,
}

impl ExperienceLogger {
    /// Create a new experience logger writing to the given directory.
    ///
    /// Creates the directory if it doesn't exist. File name includes
    /// source, timestamp, and session ID for uniqueness.
    pub fn new(
        output_dir: &str,
        source: ExperienceSource,
        session_id: &str,
    ) -> std::io::Result<Self> {
        let dir = PathBuf::from(output_dir);
        std::fs::create_dir_all(&dir)?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        let filename = format!("{source}_{timestamp}_{session_id}.jsonl");
        let path = dir.join(filename);

        let file = OpenOptions::new().create(true).append(true).open(path)?;

        Ok(Self {
            writer: BufWriter::new(file),
            live_tx: None,
        })
    }

    /// Attach a live streaming channel to the training service.
    pub fn with_live_tx(mut self, tx: tokio::sync::mpsc::Sender<ExperienceRecord>) -> Self {
        self.live_tx = Some(tx);
        self
    }

    /// Log an experience record as one JSONL line.
    ///
    /// Also sends to live training service if a channel is attached.
    /// Never panics on I/O errors — caller should use `let _ =`.
    pub fn log(&mut self, record: &ExperienceRecord) -> std::io::Result<()> {
        let json = serde_json::to_string(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(self.writer, "{json}")?;

        // Non-blocking send to training service
        if let Some(ref tx) = self.live_tx {
            let _ = tx.try_send(record.clone());
        }

        Ok(())
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experience_record_serialization() {
        let record = ExperienceRecord {
            state_idx: 42,
            action_idx: 7,
            reward_total: 0.5,
            edge_component: 1.2,
            inventory_penalty: -0.3,
            volatility_penalty: -0.2,
            inventory_change_penalty: -0.2,
            next_state_idx: 43,
            done: false,
            timestamp_ms: 1234567890,
            session_id: "test-session".to_string(),
            source: ExperienceSource::Paper,
            side: "buy".to_string(),
            fill_price: 100.0,
            mid_price: 99.95,
            fill_size: 1.0,
            inventory: 2.5,
            regime: "normal".to_string(),
        };

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: ExperienceRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.state_idx, 42);
        assert_eq!(deserialized.action_idx, 7);
        assert!((deserialized.reward_total - 0.5).abs() < 1e-10);
        assert_eq!(deserialized.source, ExperienceSource::Paper);
    }

    #[test]
    fn test_experience_source_display() {
        assert_eq!(format!("{}", ExperienceSource::Paper), "paper");
        assert_eq!(format!("{}", ExperienceSource::Live), "live");
    }

    #[test]
    fn test_experience_logger_writes_jsonl() {
        let dir = std::env::temp_dir().join("test_experience_logger");
        let _ = std::fs::remove_dir_all(&dir);

        let mut logger =
            ExperienceLogger::new(dir.to_str().unwrap(), ExperienceSource::Paper, "test").unwrap();

        let record = ExperienceRecord {
            state_idx: 0,
            action_idx: 0,
            reward_total: 1.0,
            edge_component: 1.5,
            inventory_penalty: -0.2,
            volatility_penalty: -0.1,
            inventory_change_penalty: -0.2,
            next_state_idx: 1,
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
        };

        logger.log(&record).unwrap();
        logger.flush().unwrap();

        // Verify file exists and has content
        let files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(files.len(), 1);

        let content = std::fs::read_to_string(files[0].path()).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1);

        // Verify it's valid JSON
        let parsed: ExperienceRecord = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed.state_idx, 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
