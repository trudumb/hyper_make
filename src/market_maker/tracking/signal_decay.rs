//! Signal Decay Tracking
//!
//! Monitors the degradation of predictive signals over time.
//! Alpha decays as arbitrageurs discover and exploit patterns.
//!
//! # Key Metrics
//!
//! - **MI Trend**: Slope of mutual information over time
//! - **Half-Life**: Time for signal MI to decay by 50%
//! - **Staleness**: Whether signal has decayed below usefulness threshold
//!
//! # Alert Thresholds
//!
//! - Half-life < 30 days: Warning (signal aging quickly)
//! - Half-life < 7 days: Critical (remove signal soon)
//! - MI dropped > 50% in past week: Investigate cause
//!
//! # Usage
//!
//! ```ignore
//! let mut report = SignalDecayReport::new();
//!
//! // Record daily MI measurements for each signal
//! report.record("book_imbalance", 0.045); // 0.045 bits
//! report.record("trade_flow", 0.032);     // 0.032 bits
//!
//! // Check for decaying signals
//! for alert in report.alerts() {
//!     if alert.severity == AlertSeverity::Critical {
//!         log::error!("Signal {} is stale: {}", alert.signal_name, alert.message);
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::collections::VecDeque;

/// Configuration for signal decay tracking.
#[derive(Debug, Clone)]
pub struct SignalDecayConfig {
    /// Maximum history length per signal (days)
    pub max_history_days: usize,
    /// Minimum observations before computing trends
    pub min_observations: usize,
    /// Warning threshold: half-life in days
    pub warning_half_life_days: f64,
    /// Critical threshold: half-life in days
    pub critical_half_life_days: f64,
    /// Minimum MI threshold (below = stale)
    pub min_mi_threshold: f64,
    /// Significant MI drop threshold (fraction)
    pub significant_drop_threshold: f64,
}

impl Default for SignalDecayConfig {
    fn default() -> Self {
        Self {
            max_history_days: 90,            // 3 months of history
            min_observations: 7,             // 1 week of data
            warning_half_life_days: 30.0,    // Warning if < 1 month
            critical_half_life_days: 7.0,    // Critical if < 1 week
            min_mi_threshold: 0.005,         // 0.005 bits minimum
            significant_drop_threshold: 0.5, // 50% drop is significant
        }
    }
}

/// A timestamped MI observation.
#[derive(Debug, Clone, Copy)]
struct MiObservation {
    /// Unix timestamp in milliseconds
    timestamp_ms: i64,
    /// Mutual information in bits
    mi_bits: f64,
}

/// Tracks decay for a single signal.
#[derive(Debug, Clone)]
pub struct SignalDecayTracker {
    signal_name: String,
    history: VecDeque<MiObservation>,
    max_history: usize,
}

impl SignalDecayTracker {
    /// Create a new tracker for a signal.
    pub fn new(signal_name: &str, max_history: usize) -> Self {
        Self {
            signal_name: signal_name.to_string(),
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Record an MI observation.
    pub fn add_observation(&mut self, timestamp_ms: i64, mi_bits: f64) {
        if !mi_bits.is_finite() || mi_bits < 0.0 {
            return;
        }

        self.history.push_back(MiObservation {
            timestamp_ms,
            mi_bits,
        });

        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Get signal name.
    pub fn signal_name(&self) -> &str {
        &self.signal_name
    }

    /// Get number of observations.
    pub fn observation_count(&self) -> usize {
        self.history.len()
    }

    /// Get the most recent MI value.
    pub fn current_mi(&self) -> Option<f64> {
        self.history.back().map(|obs| obs.mi_bits)
    }

    /// Compute the trend (slope) of MI over time.
    ///
    /// Returns MI change per day (negative = decaying).
    pub fn trend(&self) -> f64 {
        if self.history.len() < 3 {
            return 0.0;
        }

        // Linear regression of MI on time
        let n = self.history.len() as f64;
        let first_ts = self.history.front().map(|o| o.timestamp_ms).unwrap_or(0);

        let mut sum_t = 0.0;
        let mut sum_mi = 0.0;
        let mut sum_t2 = 0.0;
        let mut sum_t_mi = 0.0;

        for obs in &self.history {
            // Convert to days since first observation
            let t = (obs.timestamp_ms - first_ts) as f64 / (1000.0 * 86400.0);
            sum_t += t;
            sum_mi += obs.mi_bits;
            sum_t2 += t * t;
            sum_t_mi += t * obs.mi_bits;
        }

        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_t_mi - sum_t * sum_mi) / denom
    }

    /// Estimate half-life in days.
    ///
    /// Returns None if MI is not decaying or insufficient data.
    pub fn half_life_days(&self) -> Option<f64> {
        let trend = self.trend();

        if trend >= 0.0 {
            return None; // Not decaying
        }

        // Average MI
        let avg_mi: f64 =
            self.history.iter().map(|o| o.mi_bits).sum::<f64>() / self.history.len() as f64;

        if avg_mi <= 0.0 {
            return None;
        }

        // Time to decay to half: t_half = avg_mi / (2 * |trend|)
        // But exponential decay: y = y0 * exp(-k*t), half-life = ln(2)/k
        // Approximating k ≈ |trend| / avg_mi
        let k = trend.abs() / avg_mi;
        if k > 0.0 {
            Some(0.693 / k) // ln(2) ≈ 0.693
        } else {
            None
        }
    }

    /// Check if signal is stale (below minimum MI threshold).
    pub fn is_stale(&self, min_mi: f64) -> bool {
        match self.current_mi() {
            Some(mi) => mi < min_mi,
            None => true, // No data = stale
        }
    }

    /// Compute MI change over the last N days.
    ///
    /// Returns (change, percentage_change) where negative = decay.
    pub fn recent_change(&self, days: f64) -> Option<(f64, f64)> {
        if self.history.len() < 2 {
            return None;
        }

        let now_ms = self.history.back()?.timestamp_ms;
        let threshold_ms = now_ms - (days * 86400.0 * 1000.0) as i64;

        // Find earliest observation within window
        let baseline = self
            .history
            .iter()
            .find(|o| o.timestamp_ms >= threshold_ms)?;

        let current = self.history.back()?;

        if baseline.mi_bits > 0.0 {
            let change = current.mi_bits - baseline.mi_bits;
            let pct_change = change / baseline.mi_bits;
            Some((change, pct_change))
        } else {
            None
        }
    }

    /// Get history as (days_ago, mi_bits) pairs for plotting.
    pub fn history_for_plot(&self) -> Vec<(f64, f64)> {
        if self.history.is_empty() {
            return Vec::new();
        }

        let now_ms = self.history.back().map(|o| o.timestamp_ms).unwrap_or(0);

        self.history
            .iter()
            .map(|obs| {
                let days_ago = (now_ms - obs.timestamp_ms) as f64 / (1000.0 * 86400.0);
                (days_ago, obs.mi_bits)
            })
            .collect()
    }
}

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    /// Informational only
    Info,
    /// Signal is aging, monitor closely
    Warning,
    /// Signal is near stale, consider removal
    Critical,
}

/// An alert about signal health.
#[derive(Debug, Clone)]
pub struct SignalAlert {
    /// Signal name
    pub signal_name: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Human-readable message
    pub message: String,
    /// Current MI (bits)
    pub current_mi: f64,
    /// Half-life if known (days)
    pub half_life_days: Option<f64>,
    /// Recent change percentage
    pub recent_change_pct: Option<f64>,
}

/// Aggregated signal decay report.
#[derive(Debug, Clone)]
pub struct SignalDecayReport {
    trackers: HashMap<String, SignalDecayTracker>,
    config: SignalDecayConfig,
}

impl SignalDecayReport {
    /// Create a new report with default config.
    pub fn new() -> Self {
        Self::with_config(SignalDecayConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: SignalDecayConfig) -> Self {
        Self {
            trackers: HashMap::new(),
            config,
        }
    }

    /// Record an MI observation for a signal.
    pub fn record(&mut self, signal_name: &str, mi_bits: f64) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        self.record_at(signal_name, now_ms, mi_bits);
    }

    /// Record an MI observation with explicit timestamp.
    pub fn record_at(&mut self, signal_name: &str, timestamp_ms: i64, mi_bits: f64) {
        let tracker = self
            .trackers
            .entry(signal_name.to_string())
            .or_insert_with(|| SignalDecayTracker::new(signal_name, self.config.max_history_days));

        tracker.add_observation(timestamp_ms, mi_bits);
    }

    /// Get all active alerts.
    pub fn alerts(&self) -> Vec<SignalAlert> {
        let mut alerts = Vec::new();

        for (name, tracker) in &self.trackers {
            if tracker.observation_count() < self.config.min_observations {
                continue;
            }

            let current_mi = tracker.current_mi().unwrap_or(0.0);
            let half_life = tracker.half_life_days();
            let recent_change = tracker.recent_change(7.0); // Last 7 days

            // Check for stale signal
            if tracker.is_stale(self.config.min_mi_threshold) {
                alerts.push(SignalAlert {
                    signal_name: name.clone(),
                    severity: AlertSeverity::Critical,
                    message: format!(
                        "Signal MI ({:.4} bits) below minimum threshold ({:.4})",
                        current_mi, self.config.min_mi_threshold
                    ),
                    current_mi,
                    half_life_days: half_life,
                    recent_change_pct: recent_change.map(|(_, pct)| pct),
                });
                continue;
            }

            // Check half-life
            if let Some(hl) = half_life {
                if hl < self.config.critical_half_life_days {
                    alerts.push(SignalAlert {
                        signal_name: name.clone(),
                        severity: AlertSeverity::Critical,
                        message: format!("Signal decaying rapidly (half-life: {hl:.1} days)"),
                        current_mi,
                        half_life_days: Some(hl),
                        recent_change_pct: recent_change.map(|(_, pct)| pct),
                    });
                    continue;
                } else if hl < self.config.warning_half_life_days {
                    alerts.push(SignalAlert {
                        signal_name: name.clone(),
                        severity: AlertSeverity::Warning,
                        message: format!("Signal aging (half-life: {hl:.1} days)"),
                        current_mi,
                        half_life_days: Some(hl),
                        recent_change_pct: recent_change.map(|(_, pct)| pct),
                    });
                    continue;
                }
            }

            // Check for sudden drops
            if let Some((_, pct_change)) = recent_change {
                if pct_change < -self.config.significant_drop_threshold {
                    alerts.push(SignalAlert {
                        signal_name: name.clone(),
                        severity: AlertSeverity::Warning,
                        message: format!(
                            "MI dropped {:.1}% in last 7 days",
                            pct_change.abs() * 100.0
                        ),
                        current_mi,
                        half_life_days: half_life,
                        recent_change_pct: Some(pct_change),
                    });
                }
            }
        }

        // Sort by severity (critical first)
        alerts.sort_by_key(|a| match a.severity {
            AlertSeverity::Critical => 0,
            AlertSeverity::Warning => 1,
            AlertSeverity::Info => 2,
        });

        alerts
    }

    /// Generate a text report of all signal health.
    pub fn generate_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push("=== Signal Decay Report ===".to_string());
        lines.push(String::new());

        // Sort signals by current MI (lowest first = most concerning)
        let mut signals: Vec<_> = self.trackers.iter().collect();
        signals.sort_by(|a, b| {
            let mi_a = a.1.current_mi().unwrap_or(0.0);
            let mi_b = b.1.current_mi().unwrap_or(0.0);
            mi_a.partial_cmp(&mi_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (name, tracker) in signals {
            let mi = tracker.current_mi().unwrap_or(0.0);
            let trend = tracker.trend();
            let half_life = tracker.half_life_days();
            let obs_count = tracker.observation_count();

            let trend_str = if trend > 0.001 {
                "↑"
            } else if trend < -0.001 {
                "↓"
            } else {
                "→"
            };

            let half_life_str = half_life
                .map(|hl| format!("{hl:.0}d"))
                .unwrap_or_else(|| "N/A".to_string());

            lines.push(format!(
                "{name}: MI={mi:.4} bits {trend_str} (half-life: {half_life_str}, n={obs_count})"
            ));
        }

        lines.push(String::new());

        // Add alerts
        let alerts = self.alerts();
        if !alerts.is_empty() {
            lines.push("--- Alerts ---".to_string());
            for alert in alerts {
                let severity = match alert.severity {
                    AlertSeverity::Critical => "[CRITICAL]",
                    AlertSeverity::Warning => "[WARNING]",
                    AlertSeverity::Info => "[INFO]",
                };
                lines.push(format!(
                    "{} {}: {}",
                    severity, alert.signal_name, alert.message
                ));
            }
        } else {
            lines.push("No alerts.".to_string());
        }

        lines.join("\n")
    }

    /// Get a specific tracker.
    pub fn get_tracker(&self, signal_name: &str) -> Option<&SignalDecayTracker> {
        self.trackers.get(signal_name)
    }

    /// Get all signal names being tracked.
    pub fn signal_names(&self) -> Vec<&str> {
        self.trackers.keys().map(|s| s.as_str()).collect()
    }

    /// Rank signals by current MI (highest first).
    pub fn ranked_signals(&self) -> Vec<(&str, f64)> {
        let mut signals: Vec<_> = self
            .trackers
            .iter()
            .filter_map(|(name, tracker)| tracker.current_mi().map(|mi| (name.as_str(), mi)))
            .collect();

        signals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        signals
    }

    /// Check if any signals are critical.
    pub fn has_critical_alerts(&self) -> bool {
        self.alerts()
            .iter()
            .any(|a| a.severity == AlertSeverity::Critical)
    }

    /// Clear all tracking data.
    pub fn reset(&mut self) {
        self.trackers.clear();
    }
}

impl Default for SignalDecayReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_decay_tracker() {
        let mut tracker = SignalDecayTracker::new("test_signal", 100);

        // Add observations with decaying MI
        let base_ts = 0i64;
        for i in 0..30 {
            let ts = base_ts + i * 86400 * 1000; // Daily observations
            let mi = 0.1 - (i as f64) * 0.002; // Decaying from 0.1 to 0.04
            tracker.add_observation(ts, mi);
        }

        // Should detect decay
        let trend = tracker.trend();
        assert!(trend < 0.0, "Trend should be negative for decaying signal");

        // Half-life should be computable
        let half_life = tracker.half_life_days();
        assert!(half_life.is_some(), "Should compute half-life");

        // Current MI
        let current = tracker.current_mi().unwrap();
        assert!(current < 0.1, "Final MI should be less than initial");
    }

    #[test]
    fn test_signal_decay_report() {
        let mut report = SignalDecayReport::new();

        let base_ts = 0i64;

        // Add a healthy signal
        for i in 0..30 {
            let ts = base_ts + i * 86400 * 1000;
            report.record_at("healthy", ts, 0.08 + (i as f64 * 0.001)); // Slightly increasing
        }

        // Add a decaying signal
        for i in 0..30 {
            let ts = base_ts + i * 86400 * 1000;
            report.record_at("decaying", ts, 0.08 - (i as f64 * 0.002)); // Decaying
        }

        // Add a stale signal
        for i in 0..30 {
            let ts = base_ts + i * 86400 * 1000;
            report.record_at("stale", ts, 0.003); // Already below threshold
        }

        // Check alerts
        let alerts = report.alerts();
        assert!(!alerts.is_empty(), "Should have alerts");

        // Check ranking
        let ranked = report.ranked_signals();
        assert_eq!(ranked.len(), 3);

        // Generate report
        let text = report.generate_report();
        assert!(text.contains("Signal Decay Report"));
    }

    #[test]
    fn test_half_life_calculation() {
        let mut tracker = SignalDecayTracker::new("test", 100);

        // Exponential decay: MI = 0.1 * exp(-0.05 * day)
        // Half-life = ln(2) / 0.05 ≈ 13.9 days
        for i in 0..50 {
            let ts = i * 86400 * 1000;
            let mi = 0.1 * (-0.05 * i as f64).exp();
            tracker.add_observation(ts, mi);
        }

        let half_life = tracker.half_life_days();
        assert!(half_life.is_some());

        let hl = half_life.unwrap();
        // Should be approximately 14 days
        assert!(
            (hl - 14.0).abs() < 5.0,
            "Half-life {:.1} should be close to 14 days",
            hl
        );
    }
}
