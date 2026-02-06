//! Prediction log reader for batch retraining.
//!
//! Reads JSONL prediction logs written by PredictionLogger and filters
//! to resolved records (where outcomes are known).

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use crate::market_maker::simulation::prediction::PredictionRecord;

/// Read all resolved predictions from a JSONL file.
///
/// Filters to records where `outcomes.is_some()` — i.e., predictions
/// that have been matched with actual outcomes.
pub fn read_resolved(path: &Path) -> io::Result<Vec<PredictionRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<PredictionRecord>(&line) {
            Ok(record) => {
                if record.outcomes.is_some() {
                    records.push(record);
                }
            }
            Err(e) => {
                tracing::warn!(
                    line = line_num + 1,
                    error = %e,
                    "Skipping malformed prediction record"
                );
            }
        }
    }

    Ok(records)
}

/// Read all predictions (resolved and unresolved) from a JSONL file.
pub fn read_all(path: &Path) -> io::Result<Vec<PredictionRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<PredictionRecord>(&line) {
            Ok(record) => records.push(record),
            Err(e) => {
                tracing::warn!(
                    line = line_num + 1,
                    error = %e,
                    "Skipping malformed prediction record"
                );
            }
        }
    }

    Ok(records)
}
