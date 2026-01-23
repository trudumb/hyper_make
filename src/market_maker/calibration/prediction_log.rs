//! Prediction logging with outcome resolution.
//!
//! This module provides thread-safe prediction logging that allows:
//! - Recording predictions with metadata
//! - Resolving outcomes after the fact
//! - Querying resolved predictions by type or regime
//!
//! ## Example
//!
//! ```ignore
//! let log = PredictionLog::new(10000);
//!
//! // Record a fill probability prediction
//! let id = log.record(PredictionRecord {
//!     timestamp: 1234567890,
//!     prediction_type: PredictionType::FillProbability,
//!     predicted_prob: 0.75,
//!     outcome: None,
//!     regime: 0,
//!     metadata: HashMap::new(),
//! });
//!
//! // Later, resolve the outcome
//! log.resolve(id, true);  // Order was filled
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Type of prediction being made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictionType {
    /// Probability of order fill at given spread
    FillProbability,
    /// Probability of adverse selection (toxic flow)
    AdverseSelection,
    /// Probability of regime transition
    RegimeTransition,
    /// Probability of positive edge direction
    EdgeDirection,
}

impl PredictionType {
    /// Get all prediction types for iteration.
    pub fn all() -> &'static [PredictionType] {
        &[
            PredictionType::FillProbability,
            PredictionType::AdverseSelection,
            PredictionType::RegimeTransition,
            PredictionType::EdgeDirection,
        ]
    }

    /// Get a human-readable name for the prediction type.
    pub fn name(&self) -> &'static str {
        match self {
            PredictionType::FillProbability => "fill_probability",
            PredictionType::AdverseSelection => "adverse_selection",
            PredictionType::RegimeTransition => "regime_transition",
            PredictionType::EdgeDirection => "edge_direction",
        }
    }
}

/// A single prediction record with optional outcome.
#[derive(Debug, Clone)]
pub struct PredictionRecord {
    /// Record ID (assigned by the log)
    pub id: u64,
    /// Timestamp of the prediction (epoch millis or similar)
    pub timestamp: u64,
    /// Type of prediction
    pub prediction_type: PredictionType,
    /// Predicted probability [0, 1]
    pub predicted_prob: f64,
    /// Actual outcome (None until resolved)
    pub outcome: Option<bool>,
    /// Regime at time of prediction (0=calm, 1=volatile, 2=cascade)
    pub regime: usize,
    /// Additional metadata for debugging/analysis
    pub metadata: HashMap<String, f64>,
}

impl PredictionRecord {
    /// Create a new prediction record.
    ///
    /// Note: `id` will be assigned when the record is added to a log.
    pub fn new(
        timestamp: u64,
        prediction_type: PredictionType,
        predicted_prob: f64,
        regime: usize,
    ) -> Self {
        Self {
            id: 0, // Will be assigned by log
            timestamp,
            prediction_type,
            predicted_prob: predicted_prob.clamp(0.0, 1.0),
            outcome: None,
            regime: regime.min(2),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the record.
    pub fn with_metadata(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Check if the outcome has been resolved.
    pub fn is_resolved(&self) -> bool {
        self.outcome.is_some()
    }

    /// Calculate Brier score if resolved.
    pub fn brier_score(&self) -> Option<f64> {
        self.outcome.map(|o| {
            let outcome_val = if o { 1.0 } else { 0.0 };
            (self.predicted_prob - outcome_val).powi(2)
        })
    }
}

/// Thread-safe prediction log with capacity limit.
///
/// The log maintains predictions in order and supports:
/// - O(1) record insertion
/// - O(1) outcome resolution by ID
/// - O(n) queries for resolved/typed predictions
///
/// When capacity is exceeded, oldest records are discarded (FIFO).
pub struct PredictionLog {
    /// All prediction records
    records: RwLock<Vec<PredictionRecord>>,
    /// Maximum capacity before eviction
    capacity: usize,
    /// Next record ID
    next_id: AtomicU64,
    /// Index mapping record_id -> vec_index for fast resolution
    id_to_index: RwLock<HashMap<u64, usize>>,
}

impl std::fmt::Debug for PredictionLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let records = self.records.read().unwrap();
        f.debug_struct("PredictionLog")
            .field("capacity", &self.capacity)
            .field("len", &records.len())
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

impl PredictionLog {
    /// Create a new prediction log with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            records: RwLock::new(Vec::with_capacity(capacity.min(10000))),
            capacity: capacity.max(1), // At least 1
            next_id: AtomicU64::new(0),
            id_to_index: RwLock::new(HashMap::new()),
        }
    }

    /// Record a new prediction, returning its ID.
    ///
    /// If capacity is exceeded, the oldest record is evicted.
    pub fn record(&self, mut record: PredictionRecord) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        record.id = id;

        let mut records = self.records.write().unwrap();
        let mut id_map = self.id_to_index.write().unwrap();

        // Evict oldest if at capacity
        if records.len() >= self.capacity {
            if let Some(old) = records.first() {
                id_map.remove(&old.id);
            }
            records.remove(0);

            // Reindex remaining records
            id_map.clear();
            for (idx, rec) in records.iter().enumerate() {
                id_map.insert(rec.id, idx);
            }
        }

        let index = records.len();
        id_map.insert(id, index);
        records.push(record);

        id
    }

    /// Resolve the outcome for a prediction by ID.
    ///
    /// Returns true if the record was found and updated, false otherwise.
    pub fn resolve(&self, record_id: u64, outcome: bool) -> bool {
        let id_map = self.id_to_index.read().unwrap();
        if let Some(&index) = id_map.get(&record_id) {
            drop(id_map); // Release read lock before write
            let mut records = self.records.write().unwrap();
            if index < records.len() && records[index].id == record_id {
                records[index].outcome = Some(outcome);
                return true;
            }
        }
        false
    }

    /// Get all resolved predictions.
    pub fn get_resolved(&self) -> Vec<PredictionRecord> {
        let records = self.records.read().unwrap();
        records
            .iter()
            .filter(|r| r.is_resolved())
            .cloned()
            .collect()
    }

    /// Get predictions of a specific type (resolved or not).
    pub fn get_by_type(&self, pred_type: PredictionType) -> Vec<PredictionRecord> {
        let records = self.records.read().unwrap();
        records
            .iter()
            .filter(|r| r.prediction_type == pred_type)
            .cloned()
            .collect()
    }

    /// Get resolved predictions of a specific type.
    pub fn get_resolved_by_type(&self, pred_type: PredictionType) -> Vec<PredictionRecord> {
        let records = self.records.read().unwrap();
        records
            .iter()
            .filter(|r| r.prediction_type == pred_type && r.is_resolved())
            .cloned()
            .collect()
    }

    /// Get predictions by regime (resolved or not).
    pub fn get_by_regime(&self, regime: usize) -> Vec<PredictionRecord> {
        let records = self.records.read().unwrap();
        records
            .iter()
            .filter(|r| r.regime == regime)
            .cloned()
            .collect()
    }

    /// Get count of all records.
    pub fn len(&self) -> usize {
        self.records.read().unwrap().len()
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.records.read().unwrap().is_empty()
    }

    /// Get count of resolved records.
    pub fn resolved_count(&self) -> usize {
        self.records
            .read()
            .unwrap()
            .iter()
            .filter(|r| r.is_resolved())
            .count()
    }

    /// Get count of unresolved records.
    pub fn unresolved_count(&self) -> usize {
        self.records
            .read()
            .unwrap()
            .iter()
            .filter(|r| !r.is_resolved())
            .count()
    }

    /// Calculate the resolution rate.
    pub fn resolution_rate(&self) -> f64 {
        let len = self.len();
        if len == 0 {
            return 0.0;
        }
        self.resolved_count() as f64 / len as f64
    }

    /// Clear all records.
    pub fn clear(&self) {
        let mut records = self.records.write().unwrap();
        let mut id_map = self.id_to_index.write().unwrap();
        records.clear();
        id_map.clear();
    }

    /// Calculate aggregate Brier score for all resolved predictions.
    pub fn aggregate_brier_score(&self) -> Option<f64> {
        let records = self.records.read().unwrap();
        let scores: Vec<f64> = records.iter().filter_map(|r| r.brier_score()).collect();

        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    /// Calculate base rate (overall outcome frequency) for resolved predictions.
    pub fn base_rate(&self) -> Option<f64> {
        let records = self.records.read().unwrap();
        let resolved: Vec<_> = records.iter().filter(|r| r.is_resolved()).collect();

        if resolved.is_empty() {
            None
        } else {
            let positives = resolved.iter().filter(|r| r.outcome == Some(true)).count();
            Some(positives as f64 / resolved.len() as f64)
        }
    }
}

// Safety: PredictionLog uses RwLock which is Send + Sync
unsafe impl Send for PredictionLog {}
unsafe impl Sync for PredictionLog {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_type_all() {
        let all = PredictionType::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&PredictionType::FillProbability));
        assert!(all.contains(&PredictionType::AdverseSelection));
    }

    #[test]
    fn test_prediction_type_name() {
        assert_eq!(PredictionType::FillProbability.name(), "fill_probability");
        assert_eq!(PredictionType::AdverseSelection.name(), "adverse_selection");
    }

    #[test]
    fn test_prediction_record_new() {
        let record = PredictionRecord::new(1234, PredictionType::FillProbability, 0.75, 0);

        assert_eq!(record.timestamp, 1234);
        assert_eq!(record.prediction_type, PredictionType::FillProbability);
        assert_eq!(record.predicted_prob, 0.75);
        assert_eq!(record.regime, 0);
        assert!(!record.is_resolved());
        assert!(record.brier_score().is_none());
    }

    #[test]
    fn test_prediction_record_clamping() {
        // Probability should be clamped to [0, 1]
        let record_high = PredictionRecord::new(0, PredictionType::FillProbability, 1.5, 0);
        assert_eq!(record_high.predicted_prob, 1.0);

        let record_low = PredictionRecord::new(0, PredictionType::FillProbability, -0.5, 0);
        assert_eq!(record_low.predicted_prob, 0.0);

        // Regime should be clamped to max 2
        let record_regime = PredictionRecord::new(0, PredictionType::FillProbability, 0.5, 5);
        assert_eq!(record_regime.regime, 2);
    }

    #[test]
    fn test_prediction_record_with_metadata() {
        let record = PredictionRecord::new(0, PredictionType::FillProbability, 0.5, 0)
            .with_metadata("spread_bps", 5.0)
            .with_metadata("volatility", 0.02);

        assert_eq!(record.metadata.get("spread_bps"), Some(&5.0));
        assert_eq!(record.metadata.get("volatility"), Some(&0.02));
    }

    #[test]
    fn test_prediction_record_brier_score() {
        let mut record = PredictionRecord::new(0, PredictionType::FillProbability, 0.8, 0);

        // Unresolved: no Brier score
        assert!(record.brier_score().is_none());

        // Resolved with true: (0.8 - 1.0)^2 = 0.04
        record.outcome = Some(true);
        let score = record.brier_score().unwrap();
        assert!((score - 0.04).abs() < 1e-10);

        // Resolved with false: (0.8 - 0.0)^2 = 0.64
        record.outcome = Some(false);
        let score = record.brier_score().unwrap();
        assert!((score - 0.64).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_log_new() {
        let log = PredictionLog::new(100);
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
        assert_eq!(log.capacity, 100);
    }

    #[test]
    fn test_prediction_log_record_and_resolve() {
        let log = PredictionLog::new(100);

        let record = PredictionRecord::new(1234, PredictionType::FillProbability, 0.75, 0);
        let id = log.record(record);

        assert_eq!(log.len(), 1);
        assert_eq!(log.resolved_count(), 0);
        assert_eq!(log.unresolved_count(), 1);

        // Resolve the prediction
        assert!(log.resolve(id, true));
        assert_eq!(log.resolved_count(), 1);
        assert_eq!(log.unresolved_count(), 0);

        // Resolving non-existent ID returns false
        assert!(!log.resolve(9999, false));
    }

    #[test]
    fn test_prediction_log_capacity_eviction() {
        let log = PredictionLog::new(3);

        // Add 3 records
        let id1 = log.record(PredictionRecord::new(1, PredictionType::FillProbability, 0.5, 0));
        let id2 = log.record(PredictionRecord::new(2, PredictionType::FillProbability, 0.6, 0));
        let id3 = log.record(PredictionRecord::new(3, PredictionType::FillProbability, 0.7, 0));

        assert_eq!(log.len(), 3);

        // Add 4th record - should evict oldest
        let _id4 = log.record(PredictionRecord::new(4, PredictionType::FillProbability, 0.8, 0));

        assert_eq!(log.len(), 3);

        // First record should be evicted
        assert!(!log.resolve(id1, true)); // Can't find it

        // Others should still be there
        assert!(log.resolve(id2, true));
        assert!(log.resolve(id3, true));
    }

    #[test]
    fn test_prediction_log_get_by_type() {
        let log = PredictionLog::new(100);

        log.record(PredictionRecord::new(1, PredictionType::FillProbability, 0.5, 0));
        log.record(PredictionRecord::new(2, PredictionType::AdverseSelection, 0.3, 0));
        log.record(PredictionRecord::new(3, PredictionType::FillProbability, 0.7, 0));

        let fills = log.get_by_type(PredictionType::FillProbability);
        assert_eq!(fills.len(), 2);

        let as_preds = log.get_by_type(PredictionType::AdverseSelection);
        assert_eq!(as_preds.len(), 1);

        let regime = log.get_by_type(PredictionType::RegimeTransition);
        assert_eq!(regime.len(), 0);
    }

    #[test]
    fn test_prediction_log_get_by_regime() {
        let log = PredictionLog::new(100);

        log.record(PredictionRecord::new(1, PredictionType::FillProbability, 0.5, 0));
        log.record(PredictionRecord::new(2, PredictionType::FillProbability, 0.5, 1));
        log.record(PredictionRecord::new(3, PredictionType::FillProbability, 0.5, 2));
        log.record(PredictionRecord::new(4, PredictionType::FillProbability, 0.5, 0));

        assert_eq!(log.get_by_regime(0).len(), 2);
        assert_eq!(log.get_by_regime(1).len(), 1);
        assert_eq!(log.get_by_regime(2).len(), 1);
    }

    #[test]
    fn test_prediction_log_get_resolved() {
        let log = PredictionLog::new(100);

        let id1 = log.record(PredictionRecord::new(1, PredictionType::FillProbability, 0.5, 0));
        let _id2 = log.record(PredictionRecord::new(2, PredictionType::FillProbability, 0.6, 0));
        let id3 = log.record(PredictionRecord::new(3, PredictionType::FillProbability, 0.7, 0));

        log.resolve(id1, true);
        log.resolve(id3, false);

        let resolved = log.get_resolved();
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn test_prediction_log_resolution_rate() {
        let log = PredictionLog::new(100);

        assert_eq!(log.resolution_rate(), 0.0); // Empty

        let id1 = log.record(PredictionRecord::new(1, PredictionType::FillProbability, 0.5, 0));
        log.record(PredictionRecord::new(2, PredictionType::FillProbability, 0.6, 0));

        assert_eq!(log.resolution_rate(), 0.0);

        log.resolve(id1, true);
        assert!((log.resolution_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_log_aggregate_brier_score() {
        let log = PredictionLog::new(100);

        assert!(log.aggregate_brier_score().is_none()); // No resolved predictions

        let id1 = log.record(PredictionRecord::new(1, PredictionType::FillProbability, 0.9, 0));
        let id2 = log.record(PredictionRecord::new(2, PredictionType::FillProbability, 0.1, 0));

        log.resolve(id1, true); // Brier = (0.9 - 1)^2 = 0.01
        log.resolve(id2, false); // Brier = (0.1 - 0)^2 = 0.01

        let score = log.aggregate_brier_score().unwrap();
        assert!((score - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_log_base_rate() {
        let log = PredictionLog::new(100);

        assert!(log.base_rate().is_none()); // No resolved predictions

        let id1 = log.record(PredictionRecord::new(1, PredictionType::FillProbability, 0.5, 0));
        let id2 = log.record(PredictionRecord::new(2, PredictionType::FillProbability, 0.5, 0));
        let id3 = log.record(PredictionRecord::new(3, PredictionType::FillProbability, 0.5, 0));
        let id4 = log.record(PredictionRecord::new(4, PredictionType::FillProbability, 0.5, 0));

        log.resolve(id1, true);
        log.resolve(id2, true);
        log.resolve(id3, false);
        log.resolve(id4, true);

        let rate = log.base_rate().unwrap();
        assert!((rate - 0.75).abs() < 1e-10); // 3 out of 4
    }

    #[test]
    fn test_prediction_log_clear() {
        let log = PredictionLog::new(100);

        log.record(PredictionRecord::new(1, PredictionType::FillProbability, 0.5, 0));
        log.record(PredictionRecord::new(2, PredictionType::FillProbability, 0.6, 0));

        assert_eq!(log.len(), 2);

        log.clear();

        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
    }

    #[test]
    fn test_prediction_log_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let log = Arc::new(PredictionLog::new(1000));

        let mut handles = vec![];

        // Spawn multiple writers
        for i in 0..4 {
            let log_clone = Arc::clone(&log);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let ts = (i * 100 + j) as u64;
                    log_clone.record(PredictionRecord::new(
                        ts,
                        PredictionType::FillProbability,
                        0.5,
                        0,
                    ));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(log.len(), 400);
    }
}
