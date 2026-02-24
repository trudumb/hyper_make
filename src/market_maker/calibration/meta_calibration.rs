//! Meta-calibration: Tracks whether model confidence intervals are actually calibrated.
//!
//! For each model's 95% CI, records whether the true value fell within the interval.
//! If a model claims 95% confidence but only covers 70% of outcomes, it's overconfident.
//! EWMA tracking detects calibration drift in recent data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tracks calibration of confidence intervals for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCalibrationTracker {
    /// Model name for logging
    model_name: String,
    /// EWMA of coverage (fraction of observations within CI)
    ewma_coverage: f64,
    /// EWMA decay factor
    decay: f64,
    /// Target coverage (e.g., 0.95 for 95% CI)
    target_coverage: f64,
    /// Total observations
    total_observations: usize,
    /// Total hits (within CI)
    total_hits: usize,
}

impl ModelCalibrationTracker {
    pub fn new(model_name: &str, target_coverage: f64, decay: f64) -> Self {
        Self {
            model_name: model_name.to_string(),
            ewma_coverage: target_coverage, // Start at target (optimistic prior)
            decay,
            target_coverage,
            total_observations: 0,
            total_hits: 0,
        }
    }

    /// Record whether an observation fell within the model's confidence interval.
    /// `within_ci`: true if the realized value was within the predicted CI.
    pub fn record(&mut self, within_ci: bool) {
        let hit = if within_ci { 1.0 } else { 0.0 };
        self.ewma_coverage = self.decay * self.ewma_coverage + (1.0 - self.decay) * hit;
        self.total_observations += 1;
        if within_ci {
            self.total_hits += 1;
        }
    }

    /// Current EWMA coverage (should be close to target_coverage if well-calibrated)
    pub fn coverage(&self) -> f64 {
        self.ewma_coverage
    }

    /// Whether the model appears overconfident (coverage significantly below target)
    pub fn is_overconfident(&self) -> bool {
        self.total_observations >= 50 && self.ewma_coverage < self.target_coverage - 0.15
    }

    /// Whether the model appears underconfident (coverage significantly above target).
    /// Uses raw coverage rather than EWMA since underconfidence is a systemic pattern
    /// (CIs too wide), not an urgent recent drift like overconfidence.
    pub fn is_underconfident(&self) -> bool {
        self.total_observations >= 50 && self.raw_coverage() > self.target_coverage + 0.04
    }

    /// Raw (non-EWMA) coverage
    pub fn raw_coverage(&self) -> f64 {
        if self.total_observations == 0 {
            return self.target_coverage;
        }
        self.total_hits as f64 / self.total_observations as f64
    }

    /// Model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Target coverage level
    pub fn target_coverage(&self) -> f64 {
        self.target_coverage
    }

    /// Total observations recorded
    pub fn total_observations(&self) -> usize {
        self.total_observations
    }
}

/// Aggregates meta-calibration across multiple models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCalibrationTracker {
    /// Per-model trackers
    trackers: HashMap<String, ModelCalibrationTracker>,
    /// Default decay for new trackers
    #[serde(default = "default_decay")]
    default_decay: f64,
    /// Alert threshold: warn if coverage drops below this
    #[serde(default = "default_alert_threshold")]
    alert_threshold: f64,
}

fn default_decay() -> f64 {
    0.995
}

fn default_alert_threshold() -> f64 {
    0.80
}

impl MetaCalibrationTracker {
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
            default_decay: default_decay(),
            alert_threshold: default_alert_threshold(),
        }
    }

    /// Record a CI observation for a model.
    /// Creates a tracker if this is the first observation for this model.
    pub fn record(&mut self, model_name: &str, target_coverage: f64, within_ci: bool) {
        let decay = self.default_decay;
        let tracker = self
            .trackers
            .entry(model_name.to_string())
            .or_insert_with(|| ModelCalibrationTracker::new(model_name, target_coverage, decay));
        tracker.record(within_ci);
    }

    /// Get models that appear overconfident
    pub fn overconfident_models(&self) -> Vec<&str> {
        self.trackers
            .values()
            .filter(|t| t.is_overconfident())
            .map(|t| t.model_name())
            .collect()
    }

    /// Get coverage summary for all models: (name, ewma_coverage, raw_coverage, is_overconfident)
    pub fn summary(&self) -> Vec<(String, f64, f64, bool)> {
        self.trackers
            .values()
            .map(|t| {
                (
                    t.model_name().to_string(),
                    t.coverage(),
                    t.raw_coverage(),
                    t.is_overconfident(),
                )
            })
            .collect()
    }

    /// Check if any model has EWMA coverage below alert threshold
    pub fn has_alerts(&self) -> bool {
        self.trackers
            .values()
            .any(|t| t.total_observations() >= 50 && t.coverage() < self.alert_threshold)
    }

    /// Get tracker for a specific model
    pub fn get(&self, model_name: &str) -> Option<&ModelCalibrationTracker> {
        self.trackers.get(model_name)
    }
}

impl Default for MetaCalibrationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_calibration_perfect_coverage() {
        let mut tracker = MetaCalibrationTracker::new();
        for _ in 0..200 {
            tracker.record("kappa", 0.95, true);
        }
        let t = tracker.get("kappa").unwrap();
        // EWMA with decay=0.995 starting at 0.95 converges toward 1.0;
        // after 200 observations: ~0.982
        assert!(
            t.coverage() > 0.97,
            "Coverage should be near 1.0 with all hits"
        );
        assert!(t.raw_coverage() == 1.0);
        assert!(!t.is_overconfident());
    }

    #[test]
    fn test_meta_calibration_overconfident_model_detected() {
        let mut tracker = MetaCalibrationTracker::new();
        // 70% hit rate on a 95% CI => overconfident
        for i in 0..200 {
            let within = i % 10 < 7; // 70% hit rate
            tracker.record("sigma", 0.95, within);
        }
        let t = tracker.get("sigma").unwrap();
        assert!(
            t.is_overconfident(),
            "70% coverage on 95% CI should be overconfident"
        );
        assert!(
            t.coverage() < 0.80,
            "EWMA coverage should reflect recent ~70% rate"
        );
        assert!(!tracker.overconfident_models().is_empty());
    }

    #[test]
    fn test_meta_calibration_underconfident_model_detected() {
        let mut tracker = MetaCalibrationTracker::new();
        // 100% hit rate on a 95% CI => underconfident (wasting edge with too-wide CIs)
        for _ in 0..200 {
            tracker.record("as_model", 0.95, true);
        }
        let t = tracker.get("as_model").unwrap();
        assert!(
            t.is_underconfident(),
            "100% coverage on 95% CI should be underconfident"
        );
        assert!(!t.is_overconfident());
    }

    #[test]
    fn test_meta_calibration_ewma_tracks_recent_regime() {
        let mut tracker = MetaCalibrationTracker::new();
        // 500 good observations at ~95% hit rate
        for i in 0..500 {
            let within = i % 20 != 0; // 95% hit rate
            tracker.record("regime", 0.95, within);
        }
        let t = tracker.get("regime").unwrap();
        let coverage_after_good = t.coverage();
        assert!(
            coverage_after_good > 0.90,
            "Should be well-calibrated after good period"
        );

        // 100 bad observations at 50% hit rate
        for i in 0..100 {
            let within = i % 2 == 0; // 50% hit rate
            tracker.record("regime", 0.95, within);
        }
        let t = tracker.get("regime").unwrap();
        let coverage_after_bad = t.coverage();
        let raw = t.raw_coverage();

        // EWMA should drop well below raw coverage
        assert!(
            coverage_after_bad < raw,
            "EWMA ({:.3}) should be below raw ({:.3}) after bad recent period",
            coverage_after_bad,
            raw
        );
        // EWMA should reflect recent regime shift
        assert!(
            coverage_after_bad < 0.85,
            "EWMA should drop significantly after bad observations, got {:.3}",
            coverage_after_bad
        );
    }

    #[test]
    fn test_meta_calibration_per_model_isolation() {
        let mut tracker = MetaCalibrationTracker::new();

        // Model A: overconfident (60% hit on 95% CI)
        for i in 0..200 {
            let within = i % 10 < 6; // 60% hit rate
            tracker.record("model_a", 0.95, within);
        }

        // Model B: perfect calibration (95% hit on 95% CI)
        for i in 0..200 {
            let within = i % 20 != 0; // 95% hit rate
            tracker.record("model_b", 0.95, within);
        }

        let a = tracker.get("model_a").unwrap();
        let b = tracker.get("model_b").unwrap();

        assert!(a.is_overconfident(), "Model A should be overconfident");
        assert!(!b.is_overconfident(), "Model B should not be overconfident");
        assert!(
            !b.is_underconfident(),
            "Model B should not be underconfident"
        );

        let overconfident = tracker.overconfident_models();
        assert!(overconfident.contains(&"model_a"));
        assert!(!overconfident.contains(&"model_b"));
    }
}
