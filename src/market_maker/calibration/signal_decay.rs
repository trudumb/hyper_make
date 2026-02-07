//! Signal decay tracking for latency-adjusted calibration.
//!
//! This module tracks how signal value decays over time. A signal like VPIN
//! may be highly predictive when fresh (0-10ms) but worthless after 50ms.
//!
//! Key concepts:
//! - **Half-life**: Time for signal value to decay by 50%
//! - **Alpha duration**: Time until signal provides less than threshold edge
//! - **Latency-adjusted IR**: IR computed only for "fresh" predictions
//!
//! ## Signal Decay Model
//!
//! ```text
//! value(t) = value_0 × 2^(-t/half_life) + floor × (1 - 2^(-t/half_life))
//! ```
//!
//! This exponential decay converges to `floor` as t → ∞.
//!
//! ## Usage
//!
//! ```ignore
//! let mut tracker = SignalDecayTracker::new();
//!
//! // Configure signal
//! tracker.register_signal("vpin", SignalDecayConfig {
//!     half_life_ms: 10.0,
//!     floor: 0.1,
//!     freshness_multiplier: 2.0,
//! });
//!
//! // Emit signal
//! tracker.emit("vpin", 0.8, now_ms);
//!
//! // Record outcome
//! tracker.record_outcome("vpin", 5.0, now_ms + 15);  // 15ms delay
//!
//! // Check latency-adjusted IR
//! if let Some(ir) = tracker.latency_adjusted_ir("vpin") {
//!     if ir > 1.0 {
//!         // Signal still provides edge despite latency
//!     }
//! }
//! ```

use std::collections::{HashMap, VecDeque};

use super::InformationRatioTracker;

/// Configuration for signal decay behavior.
#[derive(Debug, Clone)]
pub struct SignalDecayConfig {
    /// Half-life in milliseconds (time for signal value to drop by 50%).
    /// Typical values: 10ms for VPIN, 50ms for OFI, 500ms for regime signals.
    pub half_life_ms: f64,

    /// Minimum signal value after full decay [0, 1].
    /// Represents the baseline predictive power when signal is "stale".
    pub floor: f64,

    /// Freshness threshold multiplier (default: 2.0).
    /// A signal is "fresh" if age < freshness_multiplier × half_life.
    pub freshness_multiplier: f64,
}

impl Default for SignalDecayConfig {
    fn default() -> Self {
        Self {
            half_life_ms: 50.0,    // 50ms default
            floor: 0.1,           // 10% residual value
            freshness_multiplier: 2.0, // Fresh if age < 100ms
        }
    }
}

impl SignalDecayConfig {
    /// Create config for a fast-decaying signal like VPIN.
    pub fn fast() -> Self {
        Self {
            half_life_ms: 10.0,
            floor: 0.05,
            freshness_multiplier: 2.0,
        }
    }

    /// Create config for a medium-decay signal like OFI.
    pub fn medium() -> Self {
        Self {
            half_life_ms: 50.0,
            floor: 0.15,
            freshness_multiplier: 2.0,
        }
    }

    /// Create config for a slow-decaying signal like regime.
    pub fn slow() -> Self {
        Self {
            half_life_ms: 500.0,
            floor: 0.3,
            freshness_multiplier: 3.0,
        }
    }

    /// Calculate signal value at a given age.
    ///
    /// Uses exponential decay: value(t) = v0 × 2^(-t/τ) + floor × (1 - 2^(-t/τ))
    pub fn decayed_value(&self, initial_value: f64, age_ms: f64) -> f64 {
        if age_ms <= 0.0 {
            return initial_value;
        }

        let decay_factor = 2_f64.powf(-age_ms / self.half_life_ms);
        initial_value * decay_factor + self.floor * (1.0 - decay_factor)
    }

    /// Check if signal is still "fresh" given its age.
    pub fn is_fresh(&self, age_ms: f64) -> bool {
        age_ms < self.half_life_ms * self.freshness_multiplier
    }

    /// Calculate time until signal drops below threshold value.
    ///
    /// Returns infinity if threshold <= floor (never drops below).
    pub fn time_to_threshold(&self, initial_value: f64, threshold: f64) -> f64 {
        if threshold <= self.floor {
            return f64::INFINITY;
        }
        if initial_value <= threshold {
            return 0.0;
        }

        // Solve: threshold = v0 × 2^(-t/τ) + floor × (1 - 2^(-t/τ))
        // threshold - floor = (v0 - floor) × 2^(-t/τ)
        // 2^(-t/τ) = (threshold - floor) / (v0 - floor)
        // -t/τ = log2((threshold - floor) / (v0 - floor))
        // t = -τ × log2((threshold - floor) / (v0 - floor))

        let ratio = (threshold - self.floor) / (initial_value - self.floor);
        -self.half_life_ms * ratio.log2()
    }
}

/// A recorded signal emission.
#[derive(Debug, Clone)]
pub struct SignalEmission {
    /// Signal value at emission [0, 1] or [-1, 1] depending on signal type.
    pub value: f64,

    /// Direction prediction: positive = price up, negative = price down.
    /// For non-directional signals, use 0.0.
    pub direction: f64,

    /// Timestamp of emission (epoch ms).
    pub timestamp_ms: u64,

    /// Unique ID for linking to outcomes.
    pub id: u64,

    /// Whether outcome has been recorded.
    pub resolved: bool,
}

/// A recorded signal outcome.
#[derive(Debug, Clone)]
pub struct SignalOutcome {
    /// ID of the emission this outcome resolves.
    pub emission_id: u64,

    /// Actual price move in bps.
    pub actual_move_bps: f64,

    /// Whether the direction prediction was correct.
    pub direction_correct: bool,

    /// Delay from emission to outcome (ms).
    pub delay_ms: u64,

    /// Decayed signal value at outcome time.
    pub decayed_value: f64,

    /// Was the signal still "fresh" when outcome occurred?
    pub was_fresh: bool,
}

/// Latency statistics for a signal.
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    /// Mean latency from emission to outcome (ms).
    pub mean_latency_ms: f64,

    /// Standard deviation of latency.
    pub std_latency_ms: f64,

    /// 95th percentile latency.
    pub p95_latency_ms: f64,

    /// Fraction of outcomes that occurred while signal was fresh.
    pub fresh_ratio: f64,

    /// Total emissions.
    pub n_emissions: usize,

    /// Total resolved outcomes.
    pub n_outcomes: usize,
}

/// Tracks signal value decay and computes latency-adjusted IR.
#[derive(Debug)]
pub struct SignalDecayTracker {
    /// Configuration per signal.
    configs: HashMap<String, SignalDecayConfig>,

    /// Recent emissions per signal (for linking to outcomes).
    emissions: HashMap<String, VecDeque<SignalEmission>>,

    /// Resolved outcomes per signal.
    outcomes: HashMap<String, VecDeque<SignalOutcome>>,

    /// IR tracker for fresh predictions only.
    fresh_ir: HashMap<String, InformationRatioTracker>,

    /// IR tracker for all predictions (for comparison).
    all_ir: HashMap<String, InformationRatioTracker>,

    /// Running latency statistics.
    latency_stats: HashMap<String, LatencyAccumulator>,

    /// Maximum emissions to keep per signal.
    max_emissions: usize,

    /// Maximum outcomes to keep per signal.
    max_outcomes: usize,

    /// Next emission ID.
    next_id: u64,
}

/// Internal accumulator for latency statistics.
#[derive(Debug, Default)]
struct LatencyAccumulator {
    sum: f64,
    sum_sq: f64,
    latencies: Vec<f64>,
    fresh_count: usize,
    total_count: usize,
}

impl LatencyAccumulator {
    fn update(&mut self, latency_ms: f64, was_fresh: bool) {
        self.sum += latency_ms;
        self.sum_sq += latency_ms * latency_ms;
        self.latencies.push(latency_ms);
        self.total_count += 1;
        if was_fresh {
            self.fresh_count += 1;
        }

        // Keep latencies bounded for percentile calculation
        if self.latencies.len() > 10000 {
            self.latencies.drain(0..5000);
        }
    }

    fn stats(&self, n_emissions: usize) -> LatencyStats {
        if self.total_count == 0 {
            return LatencyStats::default();
        }

        let mean = self.sum / self.total_count as f64;
        let variance = (self.sum_sq / self.total_count as f64) - mean * mean;
        let std = variance.max(0.0).sqrt();

        // Calculate P95
        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_idx = ((sorted.len() as f64) * 0.95).floor() as usize;
        let p95 = sorted.get(p95_idx).copied().unwrap_or(mean);

        LatencyStats {
            mean_latency_ms: mean,
            std_latency_ms: std,
            p95_latency_ms: p95,
            fresh_ratio: self.fresh_count as f64 / self.total_count as f64,
            n_emissions,
            n_outcomes: self.total_count,
        }
    }
}

impl Default for SignalDecayTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalDecayTracker {
    /// Create a new signal decay tracker.
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            emissions: HashMap::new(),
            outcomes: HashMap::new(),
            fresh_ir: HashMap::new(),
            all_ir: HashMap::new(),
            latency_stats: HashMap::new(),
            max_emissions: 10000,
            max_outcomes: 10000,
            next_id: 1,
        }
    }

    /// Create with custom capacity limits.
    pub fn with_capacity(max_emissions: usize, max_outcomes: usize) -> Self {
        Self {
            max_emissions,
            max_outcomes,
            ..Self::new()
        }
    }

    /// Register a signal with its decay configuration.
    pub fn register_signal(&mut self, name: &str, config: SignalDecayConfig) {
        self.configs.insert(name.to_string(), config);
        self.emissions.insert(name.to_string(), VecDeque::new());
        self.outcomes.insert(name.to_string(), VecDeque::new());
        self.fresh_ir.insert(name.to_string(), InformationRatioTracker::new(10));
        self.all_ir.insert(name.to_string(), InformationRatioTracker::new(10));
        self.latency_stats.insert(name.to_string(), LatencyAccumulator::default());
    }

    /// Emit a signal observation.
    ///
    /// Returns the emission ID for later linking to outcomes.
    pub fn emit(&mut self, signal_name: &str, value: f64, direction: f64, timestamp_ms: u64) -> Option<u64> {
        // Get or create config (use default if not registered)
        if !self.configs.contains_key(signal_name) {
            self.register_signal(signal_name, SignalDecayConfig::default());
        }

        let emissions = self.emissions.get_mut(signal_name)?;

        let id = self.next_id;
        self.next_id += 1;

        let emission = SignalEmission {
            value,
            direction,
            timestamp_ms,
            id,
            resolved: false,
        };

        emissions.push_back(emission);

        // Trim old emissions
        while emissions.len() > self.max_emissions {
            emissions.pop_front();
        }

        Some(id)
    }

    /// Record an outcome for a signal.
    ///
    /// This matches the outcome to the most recent unresolved emission
    /// and computes decay-adjusted metrics.
    pub fn record_outcome(
        &mut self,
        signal_name: &str,
        actual_move_bps: f64,
        timestamp_ms: u64,
    ) -> Option<SignalOutcome> {
        let config = self.configs.get(signal_name)?.clone();
        let emissions = self.emissions.get_mut(signal_name)?;

        // Find most recent unresolved emission
        let emission_idx = emissions.iter().rposition(|e| !e.resolved)?;
        let emission = &mut emissions[emission_idx];

        let delay_ms = timestamp_ms.saturating_sub(emission.timestamp_ms);
        let decayed_value = config.decayed_value(emission.value, delay_ms as f64);
        let was_fresh = config.is_fresh(delay_ms as f64);

        // Direction correct if sign matches
        let direction_correct = if emission.direction.abs() < 1e-10 {
            // Non-directional signal: always "correct"
            true
        } else {
            (emission.direction > 0.0) == (actual_move_bps > 0.0)
        };

        emission.resolved = true;

        let outcome = SignalOutcome {
            emission_id: emission.id,
            actual_move_bps,
            direction_correct,
            delay_ms,
            decayed_value,
            was_fresh,
        };

        // Update IR trackers
        // Use decayed value as the "predicted probability" of direction being correct
        // For directional signals, higher |value| = higher confidence in direction
        let predicted_prob = (decayed_value.abs() + 1.0) / 2.0; // Map to [0.5, 1.0]
        let predicted_prob_fresh = (emission.value.abs() + 1.0) / 2.0;

        if let Some(all_ir) = self.all_ir.get_mut(signal_name) {
            all_ir.update(predicted_prob, direction_correct);
        }

        if was_fresh {
            if let Some(fresh_ir) = self.fresh_ir.get_mut(signal_name) {
                fresh_ir.update(predicted_prob_fresh, direction_correct);
            }
        }

        // Update latency stats
        if let Some(stats) = self.latency_stats.get_mut(signal_name) {
            stats.update(delay_ms as f64, was_fresh);
        }

        // Store outcome
        if let Some(outcomes) = self.outcomes.get_mut(signal_name) {
            outcomes.push_back(outcome.clone());
            while outcomes.len() > self.max_outcomes {
                outcomes.pop_front();
            }
        }

        Some(outcome)
    }

    /// Get the latency-adjusted IR for a signal.
    ///
    /// This computes IR only for predictions where the signal was still "fresh"
    /// when the outcome was recorded.
    pub fn latency_adjusted_ir(&self, signal_name: &str) -> Option<f64> {
        let tracker = self.fresh_ir.get(signal_name)?;
        if tracker.n_samples() < 30 {
            return None;
        }
        Some(tracker.information_ratio())
    }

    /// Get the all-outcomes IR for comparison.
    pub fn all_outcomes_ir(&self, signal_name: &str) -> Option<f64> {
        let tracker = self.all_ir.get(signal_name)?;
        if tracker.n_samples() < 30 {
            return None;
        }
        Some(tracker.information_ratio())
    }

    /// Get the IR degradation ratio.
    ///
    /// Returns (latency_adjusted_ir / all_outcomes_ir).
    /// A ratio > 1.0 means the signal provides more edge when fresh.
    pub fn ir_degradation_ratio(&self, signal_name: &str) -> Option<f64> {
        let fresh = self.latency_adjusted_ir(signal_name)?;
        let all = self.all_outcomes_ir(signal_name)?;

        if all < 1e-10 {
            return None;
        }

        Some(fresh / all)
    }

    /// Get the alpha duration for a signal.
    ///
    /// Returns the time in milliseconds until the signal's expected value
    /// drops below the given threshold.
    pub fn alpha_duration_ms(&self, signal_name: &str, threshold: f64) -> Option<f64> {
        let config = self.configs.get(signal_name)?;

        // Use typical initial value (e.g., mean of recent emissions)
        let emissions = self.emissions.get(signal_name)?;
        if emissions.is_empty() {
            return None;
        }

        let mean_value: f64 = emissions.iter().map(|e| e.value.abs()).sum::<f64>()
            / emissions.len() as f64;

        Some(config.time_to_threshold(mean_value, threshold))
    }

    /// Get latency statistics for a signal.
    pub fn latency_stats(&self, signal_name: &str) -> Option<LatencyStats> {
        let stats = self.latency_stats.get(signal_name)?;
        let n_emissions = self.emissions.get(signal_name).map(|e| e.len()).unwrap_or(0);
        Some(stats.stats(n_emissions))
    }

    /// Check if the system is latency-constrained for a signal.
    ///
    /// Returns true if our processing latency exceeds the signal's alpha duration.
    pub fn is_latency_constrained(&self, signal_name: &str, processing_latency_ms: f64) -> Option<bool> {
        let config = self.configs.get(signal_name)?;
        let alpha_duration = self.alpha_duration_ms(signal_name, 0.5)?;
        Some(processing_latency_ms > alpha_duration || !config.is_fresh(processing_latency_ms))
    }

    /// Get estimated signal value at current time.
    pub fn current_value(&self, signal_name: &str, current_timestamp_ms: u64) -> Option<f64> {
        let config = self.configs.get(signal_name)?;
        let emissions = self.emissions.get(signal_name)?;
        let latest = emissions.back()?;

        let age = current_timestamp_ms.saturating_sub(latest.timestamp_ms) as f64;
        Some(config.decayed_value(latest.value, age))
    }

    /// Get registered signal names.
    pub fn signal_names(&self) -> Vec<&String> {
        self.configs.keys().collect()
    }

    /// Get config for a signal.
    pub fn config(&self, signal_name: &str) -> Option<&SignalDecayConfig> {
        self.configs.get(signal_name)
    }

    /// Clear all data for a signal.
    pub fn clear(&mut self, signal_name: &str) {
        if let Some(emissions) = self.emissions.get_mut(signal_name) {
            emissions.clear();
        }
        if let Some(outcomes) = self.outcomes.get_mut(signal_name) {
            outcomes.clear();
        }
        if let Some(ir) = self.fresh_ir.get_mut(signal_name) {
            ir.clear();
        }
        if let Some(ir) = self.all_ir.get_mut(signal_name) {
            ir.clear();
        }
        if let Some(stats) = self.latency_stats.get_mut(signal_name) {
            *stats = LatencyAccumulator::default();
        }
    }

    /// Get total emission count across all signals.
    pub fn total_emissions(&self) -> usize {
        self.emissions.values().map(|e| e.len()).sum()
    }

    /// Get total outcome count across all signals.
    pub fn total_outcomes(&self) -> usize {
        self.outcomes.values().map(|o| o.len()).sum()
    }
}

// Send + Sync for concurrent access
unsafe impl Send for SignalDecayTracker {}
unsafe impl Sync for SignalDecayTracker {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_config_default() {
        let config = SignalDecayConfig::default();
        assert_eq!(config.half_life_ms, 50.0);
        assert_eq!(config.floor, 0.1);
        assert_eq!(config.freshness_multiplier, 2.0);
    }

    #[test]
    fn test_decay_value() {
        let config = SignalDecayConfig {
            half_life_ms: 10.0,
            floor: 0.0,
            freshness_multiplier: 2.0,
        };

        // At t=0, value unchanged
        assert!((config.decayed_value(1.0, 0.0) - 1.0).abs() < 1e-10);

        // At t=half_life, value = 0.5
        assert!((config.decayed_value(1.0, 10.0) - 0.5).abs() < 1e-10);

        // At t=2*half_life, value = 0.25
        assert!((config.decayed_value(1.0, 20.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_decay_with_floor() {
        let config = SignalDecayConfig {
            half_life_ms: 10.0,
            floor: 0.2,
            freshness_multiplier: 2.0,
        };

        // At large t, should approach floor
        let value = config.decayed_value(1.0, 1000.0);
        assert!((value - 0.2).abs() < 0.01, "Value should approach floor: {}", value);
    }

    #[test]
    fn test_is_fresh() {
        let config = SignalDecayConfig {
            half_life_ms: 10.0,
            floor: 0.0,
            freshness_multiplier: 2.0,
        };

        assert!(config.is_fresh(0.0));
        assert!(config.is_fresh(15.0));
        assert!(!config.is_fresh(25.0));  // > 2 * half_life
    }

    #[test]
    fn test_time_to_threshold() {
        let config = SignalDecayConfig {
            half_life_ms: 10.0,
            floor: 0.0,
            freshness_multiplier: 2.0,
        };

        // Time to reach 0.5 from 1.0 should be half_life
        let t = config.time_to_threshold(1.0, 0.5);
        assert!((t - 10.0).abs() < 1e-10, "Time to 0.5 should be half_life: {}", t);

        // Time to reach 0.25 from 1.0 should be 2*half_life
        let t2 = config.time_to_threshold(1.0, 0.25);
        assert!((t2 - 20.0).abs() < 1e-10, "Time to 0.25 should be 2*half_life: {}", t2);
    }

    #[test]
    fn test_time_to_threshold_with_floor() {
        let config = SignalDecayConfig {
            half_life_ms: 10.0,
            floor: 0.3,
            freshness_multiplier: 2.0,
        };

        // Threshold below floor -> infinity
        assert_eq!(config.time_to_threshold(1.0, 0.2), f64::INFINITY);

        // Threshold above initial -> 0
        assert_eq!(config.time_to_threshold(0.5, 0.6), 0.0);
    }

    #[test]
    fn test_tracker_emit_and_outcome() {
        let mut tracker = SignalDecayTracker::new();
        tracker.register_signal("vpin", SignalDecayConfig::fast());

        // Emit signal
        let id = tracker.emit("vpin", 0.8, 1.0, 1000).unwrap();
        assert_eq!(id, 1);

        // Record outcome after 5ms
        let outcome = tracker.record_outcome("vpin", 10.0, 1005).unwrap();
        assert_eq!(outcome.delay_ms, 5);
        assert!(outcome.was_fresh);  // 5ms < 2 * 10ms
        assert!(outcome.direction_correct);  // direction=1.0, move=10.0 (both positive)
    }

    #[test]
    fn test_tracker_fresh_vs_stale() {
        let mut tracker = SignalDecayTracker::new();
        tracker.register_signal("vpin", SignalDecayConfig {
            half_life_ms: 10.0,
            floor: 0.1,
            freshness_multiplier: 2.0,
        });

        // Fresh emission (resolved quickly)
        tracker.emit("vpin", 0.9, 1.0, 1000);
        let outcome1 = tracker.record_outcome("vpin", 10.0, 1010).unwrap();
        assert!(outcome1.was_fresh);

        // Stale emission (resolved slowly)
        tracker.emit("vpin", 0.9, 1.0, 2000);
        let outcome2 = tracker.record_outcome("vpin", 10.0, 2050).unwrap();
        assert!(!outcome2.was_fresh);  // 50ms > 2 * 10ms
    }

    #[test]
    fn test_latency_stats() {
        let mut tracker = SignalDecayTracker::new();
        tracker.register_signal("vpin", SignalDecayConfig::medium());

        // Add some emissions with varying delays
        for i in 0..100 {
            tracker.emit("vpin", 0.7, 1.0, i * 100);
            tracker.record_outcome("vpin", 5.0, i * 100 + 20 + (i % 30));
        }

        let stats = tracker.latency_stats("vpin").unwrap();
        assert!(stats.mean_latency_ms > 20.0);
        assert!(stats.mean_latency_ms < 60.0);
        assert!(stats.fresh_ratio > 0.9);  // Most should be fresh
        assert_eq!(stats.n_outcomes, 100);
    }

    #[test]
    fn test_ir_computation() {
        let mut tracker = SignalDecayTracker::new();
        tracker.register_signal("vpin", SignalDecayConfig::medium());

        // Add predictions with consistent direction correctness
        for i in 0..200 {
            let direction = if i % 3 == 0 { -1.0 } else { 1.0 };
            let move_bps = if i % 3 == 0 { -5.0 } else { 5.0 };  // Mostly correct

            tracker.emit("vpin", 0.8, direction, i * 100);
            tracker.record_outcome("vpin", move_bps, i * 100 + 30);
        }

        // Should have enough samples for IR
        let fresh_ir = tracker.latency_adjusted_ir("vpin");
        let all_ir = tracker.all_outcomes_ir("vpin");

        assert!(fresh_ir.is_some());
        assert!(all_ir.is_some());
    }

    #[test]
    fn test_alpha_duration() {
        let mut tracker = SignalDecayTracker::new();
        tracker.register_signal("vpin", SignalDecayConfig {
            half_life_ms: 10.0,
            floor: 0.0,
            freshness_multiplier: 2.0,
        });

        // Emit some signals to establish mean value
        for i in 0..10 {
            tracker.emit("vpin", 0.8, 1.0, i * 100);
        }

        let alpha_ms = tracker.alpha_duration_ms("vpin", 0.4).unwrap();
        // Should be about 10ms (half-life) since 0.8 -> 0.4 is one half-life
        assert!((alpha_ms - 10.0).abs() < 2.0, "Alpha duration: {}", alpha_ms);
    }

    #[test]
    fn test_is_latency_constrained() {
        let mut tracker = SignalDecayTracker::new();
        tracker.register_signal("vpin", SignalDecayConfig::fast());

        for i in 0..10 {
            tracker.emit("vpin", 0.8, 1.0, i * 100);
        }

        // Fast processing: not constrained
        let constrained_fast = tracker.is_latency_constrained("vpin", 5.0).unwrap();
        assert!(!constrained_fast);

        // Slow processing: constrained
        let constrained_slow = tracker.is_latency_constrained("vpin", 50.0).unwrap();
        assert!(constrained_slow);
    }

    #[test]
    fn test_current_value() {
        let mut tracker = SignalDecayTracker::new();
        tracker.register_signal("vpin", SignalDecayConfig {
            half_life_ms: 10.0,
            floor: 0.0,
            freshness_multiplier: 2.0,
        });

        tracker.emit("vpin", 1.0, 1.0, 1000);

        // At same time: full value
        let v0 = tracker.current_value("vpin", 1000).unwrap();
        assert!((v0 - 1.0).abs() < 1e-10);

        // After half-life: half value
        let v1 = tracker.current_value("vpin", 1010).unwrap();
        assert!((v1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_clear() {
        let mut tracker = SignalDecayTracker::new();
        tracker.register_signal("vpin", SignalDecayConfig::default());

        tracker.emit("vpin", 0.8, 1.0, 1000);
        tracker.record_outcome("vpin", 5.0, 1020);

        tracker.clear("vpin");

        assert_eq!(tracker.total_emissions(), 0);
        assert_eq!(tracker.total_outcomes(), 0);
    }

    #[test]
    fn test_auto_register() {
        let mut tracker = SignalDecayTracker::new();

        // Emit without registering first
        let id = tracker.emit("new_signal", 0.7, 1.0, 1000);
        assert!(id.is_some());

        // Should auto-register with defaults
        assert!(tracker.config("new_signal").is_some());
    }
}
