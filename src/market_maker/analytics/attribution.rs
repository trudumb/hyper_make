//! Signal PnL attribution via active/inactive conditional analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-cycle record of what a single signal contributed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalContribution {
    pub signal_name: String,
    pub spread_adjustment_bps: f64,
    pub skew_adjustment_bps: f64,
    pub was_active: bool,
    pub gating_weight: f64,
    pub raw_value: f64,
}

/// All signal contributions for one quote cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleContributions {
    pub cycle_id: u64,
    pub timestamp_ns: u64,
    pub contributions: Vec<SignalContribution>,
    pub total_spread_mult: f64,
    pub combined_skew_bps: f64,
}

/// Internal per-signal tracking stats.
#[derive(Debug, Clone, Default)]
struct SignalStats {
    active_pnl_sum: f64,
    active_count: u64,
    inactive_pnl_sum: f64,
    inactive_count: u64,
}

/// Tracks per-signal PnL using active/inactive conditional analysis.
///
/// For each cycle, PnL is attributed to each signal's active and inactive
/// buckets. The marginal value of a signal is the difference in per-fill PnL
/// between when the signal is active vs inactive.
#[derive(Debug, Clone)]
pub struct SignalPnLAttributor {
    stats: HashMap<String, SignalStats>,
}

impl SignalPnLAttributor {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    /// Record a cycle's contributions along with its PnL outcome.
    pub fn record_cycle(&mut self, contributions: &CycleContributions, pnl_bps: f64) {
        for contrib in &contributions.contributions {
            let entry = self.stats.entry(contrib.signal_name.clone()).or_default();
            if contrib.was_active {
                entry.active_pnl_sum += pnl_bps;
                entry.active_count += 1;
            } else {
                entry.inactive_pnl_sum += pnl_bps;
                entry.inactive_count += 1;
            }
        }
    }

    /// Cumulative PnL when a signal was active.
    pub fn signal_pnl(&self, signal: &str) -> f64 {
        self.stats.get(signal).map_or(0.0, |s| s.active_pnl_sum)
    }

    /// Cumulative PnL when a signal was inactive.
    pub fn signal_pnl_inactive(&self, signal: &str) -> f64 {
        self.stats.get(signal).map_or(0.0, |s| s.inactive_pnl_sum)
    }

    /// Marginal value: active PnL/fill minus inactive PnL/fill.
    pub fn marginal_value(&self, signal: &str) -> f64 {
        let stats = match self.stats.get(signal) {
            Some(s) => s,
            None => return 0.0,
        };

        let active_avg = if stats.active_count > 0 {
            stats.active_pnl_sum / stats.active_count as f64
        } else {
            0.0
        };
        let inactive_avg = if stats.inactive_count > 0 {
            stats.inactive_pnl_sum / stats.inactive_count as f64
        } else {
            0.0
        };

        active_avg - inactive_avg
    }

    /// All tracked signal names.
    pub fn signal_names(&self) -> Vec<String> {
        self.stats.keys().cloned().collect()
    }

    /// Human-readable attribution report.
    pub fn format_report(&self) -> String {
        let mut names: Vec<&String> = self.stats.keys().collect();
        names.sort();

        if names.is_empty() {
            return "No attribution data".to_string();
        }

        let mut lines = vec!["Signal PnL Attribution:".to_string()];
        for name in &names {
            let stats = &self.stats[*name];
            let marginal = self.marginal_value(name);
            lines.push(format!(
                "  {name}: marginal={marginal:.2}bps active_pnl={:.1}bps({} fills) inactive_pnl={:.1}bps({} fills)",
                stats.active_pnl_sum, stats.active_count,
                stats.inactive_pnl_sum, stats.inactive_count,
            ));
        }
        lines.join("\n")
    }
}

impl Default for SignalPnLAttributor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cycle(cycle_id: u64, signals: &[(&str, bool)]) -> CycleContributions {
        CycleContributions {
            cycle_id,
            timestamp_ns: cycle_id * 1_000_000_000,
            contributions: signals
                .iter()
                .map(|(name, active)| SignalContribution {
                    signal_name: name.to_string(),
                    spread_adjustment_bps: 0.0,
                    skew_adjustment_bps: 0.0,
                    was_active: *active,
                    gating_weight: if *active { 1.0 } else { 0.0 },
                    raw_value: 0.0,
                })
                .collect(),
            total_spread_mult: 1.0,
            combined_skew_bps: 0.0,
        }
    }

    #[test]
    fn test_marginal_value() {
        let mut attributor = SignalPnLAttributor::new();

        // Signal A active => +5 bps each time
        for i in 0..10 {
            let cycle = make_cycle(i, &[("SignalA", true)]);
            attributor.record_cycle(&cycle, 5.0);
        }
        // Signal A inactive => -2 bps each time
        for i in 10..20 {
            let cycle = make_cycle(i, &[("SignalA", false)]);
            attributor.record_cycle(&cycle, -2.0);
        }

        // Marginal = 5.0 - (-2.0) = 7.0
        let marginal = attributor.marginal_value("SignalA");
        assert!(
            (marginal - 7.0).abs() < 1e-10,
            "Expected 7.0, got {marginal}"
        );
    }

    #[test]
    fn test_active_inactive_split() {
        let mut attributor = SignalPnLAttributor::new();

        let active_cycle = make_cycle(1, &[("Flow", true)]);
        attributor.record_cycle(&active_cycle, 10.0);

        let inactive_cycle = make_cycle(2, &[("Flow", false)]);
        attributor.record_cycle(&inactive_cycle, -3.0);

        assert!((attributor.signal_pnl("Flow") - 10.0).abs() < 1e-10);
        assert!((attributor.signal_pnl_inactive("Flow") - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_signals() {
        let mut attributor = SignalPnLAttributor::new();

        // Both signals active, PnL = 8
        let cycle = make_cycle(1, &[("LeadLag", true), ("InformedFlow", true)]);
        attributor.record_cycle(&cycle, 8.0);

        // Only LeadLag active, PnL = 3
        let cycle = make_cycle(2, &[("LeadLag", true), ("InformedFlow", false)]);
        attributor.record_cycle(&cycle, 3.0);

        let names = attributor.signal_names();
        assert_eq!(names.len(), 2);

        // LeadLag: always active, avg = (8+3)/2 = 5.5
        assert!((attributor.signal_pnl("LeadLag") - 11.0).abs() < 1e-10);

        let report = attributor.format_report();
        assert!(report.contains("LeadLag"));
        assert!(report.contains("InformedFlow"));
    }

    #[test]
    fn test_unknown_signal() {
        let attributor = SignalPnLAttributor::new();
        assert_eq!(attributor.signal_pnl("NonExistent"), 0.0);
        assert_eq!(attributor.marginal_value("NonExistent"), 0.0);
    }

    #[test]
    fn test_format_report_empty() {
        let attributor = SignalPnLAttributor::new();
        assert_eq!(attributor.format_report(), "No attribution data");
    }
}
