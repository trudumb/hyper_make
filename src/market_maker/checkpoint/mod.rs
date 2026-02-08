//! Checkpoint persistence for warm-starting the market maker.
//!
//! Saves learned model state (classifier weights, Bayesian posteriors, regime beliefs)
//! to JSON files on disk. On restart, loads the latest checkpoint to avoid cold-starting.
//!
//! # Architecture
//!
//! ```text
//! Live Market Maker (runs continuously)
//!   │ saves every 5 min + on shutdown
//!   ▼
//! data/checkpoints/latest/checkpoint.json
//!   │ loads on startup (warm-start)
//!   ▼
//! Live Market Maker (next session)
//! ```
//!
//! # Design Decisions
//!
//! - **JSON format**: Human-readable, forward-compatible, version field for migrations
//! - **Separate checkpoint structs**: Extracts only learning state, not full struct serialization
//! - **Atomic writes**: Write to `.tmp`, then `fs::rename` to prevent corruption
//! - **7-day retention**: ~100KB each, old checkpoints cleaned up automatically

pub mod types;
pub mod prediction_reader;

pub use types::*;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use tracing::{debug, info};

/// Manages checkpoint persistence to disk.
///
/// Saves and loads `CheckpointBundle` to a directory structure:
/// ```text
/// base_dir/
/// ├── latest/
/// │   └── checkpoint.json
/// └── 1700000000000/     (timestamped backups)
///     └── checkpoint.json
/// ```
pub struct CheckpointManager {
    base_dir: PathBuf,
}

impl CheckpointManager {
    /// Create a new CheckpointManager.
    ///
    /// Creates the base directory if it doesn't exist.
    pub fn new(base_dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    /// Save a checkpoint bundle atomically.
    ///
    /// 1. Writes to `latest/checkpoint.json.tmp`
    /// 2. Renames to `latest/checkpoint.json` (atomic on same filesystem)
    /// 3. Creates a timestamped backup copy
    pub fn save_all(&self, bundle: &CheckpointBundle) -> io::Result<()> {
        let latest_dir = self.base_dir.join("latest");
        fs::create_dir_all(&latest_dir)?;

        let checkpoint_path = latest_dir.join("checkpoint.json");
        let tmp_path = latest_dir.join("checkpoint.json.tmp");

        // Serialize to JSON
        let json = serde_json::to_string_pretty(bundle).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("JSON serialize: {e}"))
        })?;

        // Atomic write: tmp file → rename
        fs::write(&tmp_path, &json)?;
        fs::rename(&tmp_path, &checkpoint_path)?;

        // Create timestamped backup
        let backup_dir = self.base_dir.join(bundle.metadata.timestamp_ms.to_string());
        fs::create_dir_all(&backup_dir)?;
        fs::write(backup_dir.join("checkpoint.json"), &json)?;

        info!(
            asset = %bundle.metadata.asset,
            version = bundle.metadata.version,
            pre_fill_samples = bundle.pre_fill.learning_samples,
            enhanced_samples = bundle.enhanced.learning_samples,
            vol_filter_obs = bundle.vol_filter.observation_count,
            kappa_own_obs = bundle.kappa_own.observation_count,
            kelly_wins = bundle.kelly_tracker.n_wins,
            kelly_losses = bundle.kelly_tracker.n_losses,
            rl_observations = bundle.rl_q_table.total_observations,
            "Checkpoint saved to {}",
            checkpoint_path.display()
        );

        Ok(())
    }

    /// Load the latest checkpoint bundle.
    ///
    /// Returns `Ok(None)` if no checkpoint exists.
    /// Returns `Err` only on actual I/O or parse errors.
    pub fn load_latest(&self) -> io::Result<Option<CheckpointBundle>> {
        let checkpoint_path = self.base_dir.join("latest/checkpoint.json");

        if !checkpoint_path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&checkpoint_path)?;
        let bundle: CheckpointBundle = serde_json::from_str(&json).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("JSON deserialize: {e}"),
            )
        })?;

        info!(
            asset = %bundle.metadata.asset,
            version = bundle.metadata.version,
            timestamp_ms = bundle.metadata.timestamp_ms,
            session_duration_s = bundle.metadata.session_duration_s,
            pre_fill_samples = bundle.pre_fill.learning_samples,
            enhanced_samples = bundle.enhanced.learning_samples,
            "Loaded checkpoint from {}",
            checkpoint_path.display()
        );

        Ok(Some(bundle))
    }

    /// Clean up old timestamped checkpoint backups.
    ///
    /// Keeps the `latest/` directory and removes backups older than `keep_days`.
    /// Returns the number of directories removed.
    pub fn cleanup_old(&self, keep_days: u32) -> io::Result<usize> {
        let cutoff_ms = {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            now.saturating_sub(keep_days as u64 * 86_400_000)
        };

        let mut removed = 0;
        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Skip "latest" and non-numeric directories
            if name_str == "latest" {
                continue;
            }

            // Try to parse directory name as timestamp
            if let Ok(ts) = name_str.parse::<u64>() {
                if ts < cutoff_ms {
                    fs::remove_dir_all(entry.path())?;
                    debug!(
                        timestamp_ms = ts,
                        "Removed old checkpoint: {}",
                        entry.path().display()
                    );
                    removed += 1;
                }
            }
        }

        if removed > 0 {
            info!(
                removed = removed,
                keep_days = keep_days,
                "Cleaned up old checkpoints"
            );
        }

        Ok(removed)
    }

    /// Get the base directory path.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::calibration::parameter_learner::LearnedParameters;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn make_test_dir() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("checkpoint_test_{}", id));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn make_test_bundle() -> CheckpointBundle {
        CheckpointBundle {
            metadata: CheckpointMetadata {
                version: 1,
                timestamp_ms: 1700000000000,
                asset: "ETH".to_string(),
                session_duration_s: 3600.0,
            },
            learned_params: LearnedParameters::default(),
            pre_fill: PreFillCheckpoint {
                learned_weights: [0.35, 0.20, 0.20, 0.15, 0.10],
                signal_outcome_sum: [1.0, 2.0, 3.0, 4.0, 5.0],
                signal_sq_sum: [10.0, 20.0, 30.0, 40.0, 50.0],
                learning_samples: 1000,
                regime_probs: [0.05, 0.60, 0.25, 0.10],
                ..PreFillCheckpoint::default()
            },
            enhanced: EnhancedCheckpoint::default(),
            vol_filter: VolFilterCheckpoint::default(),
            regime_hmm: RegimeHMMCheckpoint::default(),
            informed_flow: InformedFlowCheckpoint::default(),
            fill_rate: FillRateCheckpoint::default(),
            kappa_own: KappaCheckpoint::default(),
            kappa_bid: KappaCheckpoint::default(),
            kappa_ask: KappaCheckpoint::default(),
            momentum: MomentumCheckpoint::default(),
            kelly_tracker: KellyTrackerCheckpoint::default(),
            ensemble_weights: EnsembleWeightsCheckpoint::default(),
            rl_q_table: RLCheckpoint::default(),
        }
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = make_test_dir();
        let manager = CheckpointManager::new(dir.clone()).expect("create manager");

        let bundle = make_test_bundle();
        manager.save_all(&bundle).expect("save");

        let loaded = manager.load_latest().expect("load").expect("should exist");

        assert_eq!(loaded.metadata.version, 1);
        assert_eq!(loaded.metadata.asset, "ETH");
        assert_eq!(loaded.pre_fill.learning_samples, 1000);
        assert_eq!(loaded.pre_fill.learned_weights[0], 0.35);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_latest_when_empty() {
        let dir = make_test_dir();
        let manager = CheckpointManager::new(dir.clone()).expect("create manager");

        let loaded = manager.load_latest().expect("load");
        assert!(loaded.is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cleanup_old() {
        let dir = make_test_dir();
        let manager = CheckpointManager::new(dir.clone()).expect("create manager");

        // Create an "old" backup (timestamp from 2023)
        let old_dir = dir.join("1600000000000");
        fs::create_dir_all(&old_dir).unwrap();
        fs::write(old_dir.join("checkpoint.json"), "{}").unwrap();

        // Create a "recent" backup (very recent timestamp)
        let recent_dir = dir.join("9999999999999");
        fs::create_dir_all(&recent_dir).unwrap();
        fs::write(recent_dir.join("checkpoint.json"), "{}").unwrap();

        let removed = manager.cleanup_old(7).expect("cleanup");
        assert_eq!(removed, 1, "Should remove old backup only");
        assert!(!old_dir.exists(), "Old backup should be removed");
        assert!(recent_dir.exists(), "Recent backup should be kept");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_atomic_write_creates_no_tmp() {
        let dir = make_test_dir();
        let manager = CheckpointManager::new(dir.clone()).expect("create manager");

        let bundle = make_test_bundle();
        manager.save_all(&bundle).expect("save");

        // .tmp file should not exist after successful save
        let tmp_path = dir.join("latest/checkpoint.json.tmp");
        assert!(!tmp_path.exists(), ".tmp file should be cleaned up after rename");

        let _ = fs::remove_dir_all(&dir);
    }
}
