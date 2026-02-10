//! JSONL file logging for analytics data.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use super::attribution::SignalPnLAttributor;
use super::edge_metrics::EdgeSnapshot;
use super::sharpe::SharpeSummary;
use super::CycleContributions;

/// JSONL file writer for analytics persistence.
///
/// Writes one JSON object per line to separate files for each data type.
/// Uses `BufWriter` for efficient I/O and append mode for crash safety.
pub struct AnalyticsLogger {
    sharpe_writer: BufWriter<File>,
    contributions_writer: BufWriter<File>,
    signal_pnl_writer: BufWriter<File>,
    edge_writer: BufWriter<File>,
}

impl AnalyticsLogger {
    /// Create a new analytics logger writing to the given directory.
    ///
    /// Creates the directory if it doesn't exist. Opens four JSONL files
    /// in append mode.
    pub fn new(output_dir: &str) -> std::io::Result<Self> {
        let dir = PathBuf::from(output_dir);
        std::fs::create_dir_all(&dir)?;

        let open_append = |name: &str| -> std::io::Result<BufWriter<File>> {
            let path = dir.join(name);
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            Ok(BufWriter::new(file))
        };

        Ok(Self {
            sharpe_writer: open_append("sharpe_metrics.jsonl")?,
            contributions_writer: open_append("signal_contributions.jsonl")?,
            signal_pnl_writer: open_append("signal_pnl.jsonl")?,
            edge_writer: open_append("edge_validation.jsonl")?,
        })
    }

    /// Log a Sharpe summary as one JSONL line.
    pub fn log_sharpe(&mut self, summary: &SharpeSummary) -> std::io::Result<()> {
        let json = serde_json::to_string(summary)?;
        writeln!(self.sharpe_writer, "{json}")
    }

    /// Log cycle contributions as one JSONL line.
    pub fn log_contributions(&mut self, cycle: &CycleContributions) -> std::io::Result<()> {
        let json = serde_json::to_string(cycle)?;
        writeln!(self.contributions_writer, "{json}")
    }

    /// Log a snapshot of per-signal PnL stats as one JSONL line.
    pub fn log_signal_pnl(&mut self, attributor: &SignalPnLAttributor) -> std::io::Result<()> {
        let snapshot: std::collections::HashMap<String, SignalPnlSnapshot> = attributor
            .signal_names()
            .into_iter()
            .map(|name| {
                let snap = SignalPnlSnapshot {
                    active_pnl_bps: attributor.signal_pnl(&name),
                    inactive_pnl_bps: attributor.signal_pnl_inactive(&name),
                    marginal_value_bps: attributor.marginal_value(&name),
                };
                (name, snap)
            })
            .collect();
        let json = serde_json::to_string(&snapshot)?;
        writeln!(self.signal_pnl_writer, "{json}")
    }

    /// Log an edge snapshot as one JSONL line.
    pub fn log_edge(&mut self, snapshot: &EdgeSnapshot) -> std::io::Result<()> {
        let json = serde_json::to_string(snapshot)?;
        writeln!(self.edge_writer, "{json}")
    }

    /// Flush all writers to disk.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.sharpe_writer.flush()?;
        self.contributions_writer.flush()?;
        self.signal_pnl_writer.flush()?;
        self.edge_writer.flush()?;
        Ok(())
    }
}

/// Serializable snapshot of per-signal PnL stats for JSONL output.
#[derive(serde::Serialize)]
struct SignalPnlSnapshot {
    active_pnl_bps: f64,
    inactive_pnl_bps: f64,
    marginal_value_bps: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_dir(name: &str) -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "analytics_test_{}_{}_{name}",
            std::process::id(),
            id
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn test_creates_files() {
        let dir = test_dir("creates_files");
        let _logger = AnalyticsLogger::new(dir.to_str().unwrap()).unwrap();

        assert!(dir.join("sharpe_metrics.jsonl").exists());
        assert!(dir.join("signal_contributions.jsonl").exists());
        assert!(dir.join("signal_pnl.jsonl").exists());
        assert!(dir.join("edge_validation.jsonl").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_jsonl_format() {
        let dir = test_dir("jsonl_format");
        let mut logger = AnalyticsLogger::new(dir.to_str().unwrap()).unwrap();

        let summary = SharpeSummary {
            sharpe_1h: 1.5,
            sharpe_24h: 2.0,
            sharpe_7d: 1.8,
            sharpe_all: 1.9,
            count: 100,
            mean_return_bps: 3.5,
            std_return_bps: 2.1,
            elapsed_secs: 3600.0,
        };
        logger.log_sharpe(&summary).unwrap();
        logger.flush().unwrap();

        let content = std::fs::read_to_string(dir.join("sharpe_metrics.jsonl")).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 1);

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed["count"], 100);
        assert_eq!(parsed["sharpe_1h"], 1.5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_append_mode() {
        let dir = test_dir("append_mode");

        // Write first entry
        {
            let mut logger = AnalyticsLogger::new(dir.to_str().unwrap()).unwrap();
            let snap = EdgeSnapshot {
                timestamp_ns: 1000,
                predicted_spread_bps: 5.0,
                realized_spread_bps: 4.0,
                predicted_as_bps: 1.0,
                realized_as_bps: 0.5,
                fee_bps: 1.5,
                predicted_edge_bps: 2.5,
                realized_edge_bps: 2.0,
                gross_edge_bps: 3.5,
            };
            logger.log_edge(&snap).unwrap();
            logger.flush().unwrap();
        }

        // Open again and write second entry
        {
            let mut logger = AnalyticsLogger::new(dir.to_str().unwrap()).unwrap();
            let snap = EdgeSnapshot {
                timestamp_ns: 2000,
                predicted_spread_bps: 6.0,
                realized_spread_bps: 5.0,
                predicted_as_bps: 1.0,
                realized_as_bps: 0.5,
                fee_bps: 1.5,
                predicted_edge_bps: 3.5,
                realized_edge_bps: 3.0,
                gross_edge_bps: 4.5,
            };
            logger.log_edge(&snap).unwrap();
            logger.flush().unwrap();
        }

        let content = std::fs::read_to_string(dir.join("edge_validation.jsonl")).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 2, "Expected 2 lines in append mode");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_log_signal_pnl() {
        let dir = test_dir("signal_pnl");
        let mut logger = AnalyticsLogger::new(dir.to_str().unwrap()).unwrap();

        let mut attributor = SignalPnLAttributor::new();
        let cycle = CycleContributions {
            cycle_id: 1,
            timestamp_ns: 1000,
            contributions: vec![super::super::attribution::SignalContribution {
                signal_name: "TestSignal".to_string(),
                spread_adjustment_bps: 1.0,
                skew_adjustment_bps: 0.5,
                was_active: true,
                gating_weight: 1.0,
                raw_value: 42.0,
            }],
            total_spread_mult: 1.0,
            combined_skew_bps: 0.5,
        };
        attributor.record_cycle(&cycle, 5.0);

        logger.log_signal_pnl(&attributor).unwrap();
        logger.flush().unwrap();

        let content = std::fs::read_to_string(dir.join("signal_pnl.jsonl")).unwrap();
        assert!(!content.trim().is_empty());
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert!(parsed["TestSignal"].is_object());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
