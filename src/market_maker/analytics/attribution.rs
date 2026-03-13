//! Signal PnL attribution via online univariate regression.
//!
//! For each signal, tracks Cov(adjustment, fill_pnl) / Var(adjustment) —
//! the OLS slope coefficient measuring "bps of PnL per bps of signal adjustment."

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

/// Minimum variance threshold to report a regression result.
/// Below this, the signal has insufficient variation for attribution.
const MIN_VARIANCE_EPSILON: f64 = 1e-12;

/// Minimum observations needed for a signal's marginal value to be reported.
/// Below this, the regression has too few samples for statistical significance.
const MIN_MARGINAL_OBSERVATIONS: u64 = 20;

/// Online running statistics for univariate regression of fill PnL on signal adjustment.
#[derive(Debug, Clone, Default)]
struct SignalRegression {
    /// Sum of x (signal adjustment bps).
    sum_x: f64,
    /// Sum of y (fill PnL bps).
    sum_y: f64,
    /// Sum of x*y.
    sum_xy: f64,
    /// Sum of x^2.
    sum_x2: f64,
    /// Sum of y^2 (for R²).
    sum_y2: f64,
    /// Number of observations.
    count: u64,
}

impl SignalRegression {
    fn add(&mut self, x: f64, y: f64) {
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xy += x * y;
        self.sum_x2 += x * x;
        self.sum_y2 += y * y;
        self.count += 1;
    }

    /// Variance of x: E[x²] - E[x]²
    fn var_x(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        (self.sum_x2 / n) - (self.sum_x / n).powi(2)
    }

    /// Variance of y: E[y²] - E[y]²
    fn var_y(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        (self.sum_y2 / n) - (self.sum_y / n).powi(2)
    }

    /// Covariance of x and y: E[xy] - E[x]E[y]
    fn cov_xy(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        (self.sum_xy / n) - (self.sum_x / n) * (self.sum_y / n)
    }

    /// OLS slope: Cov(x,y) / Var(x), or 0.0 if insufficient variance.
    fn slope(&self) -> f64 {
        let var_x = self.var_x();
        if var_x < MIN_VARIANCE_EPSILON {
            return 0.0;
        }
        self.cov_xy() / var_x
    }

    /// R² = Cov(x,y)² / (Var(x) * Var(y)), measuring attribution quality.
    fn r_squared(&self) -> f64 {
        let var_x = self.var_x();
        let var_y = self.var_y();
        if var_x < MIN_VARIANCE_EPSILON || var_y < MIN_VARIANCE_EPSILON {
            return 0.0;
        }
        let cov = self.cov_xy();
        (cov * cov) / (var_x * var_y)
    }

    /// Whether we have enough data and variance to report meaningful results.
    fn has_sufficient_data(&self) -> bool {
        self.count >= 10 && self.var_x() >= MIN_VARIANCE_EPSILON
    }
}

/// Tracks per-signal PnL attribution using online univariate regression.
///
/// For each fill, regresses PnL (bps) on each signal's total adjustment magnitude
/// (spread_adjustment_bps + skew_adjustment_bps). The slope coefficient gives
/// "bps of PnL per bps of signal adjustment" — a continuous, signal-specific measure.
#[derive(Debug, Clone)]
pub struct SignalPnLAttributor {
    stats: HashMap<String, SignalRegression>,
}

impl SignalPnLAttributor {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    /// Record a cycle's contributions along with its PnL outcome.
    ///
    /// Uses each signal's total adjustment (spread + skew) as the regressor
    /// and fill PnL as the response variable.
    pub fn record_cycle(&mut self, contributions: &CycleContributions, pnl_bps: f64) {
        for contrib in &contributions.contributions {
            let x = contrib.spread_adjustment_bps + contrib.skew_adjustment_bps;
            let entry = self.stats.entry(contrib.signal_name.clone()).or_default();
            entry.add(x, pnl_bps);
        }
    }

    /// Cumulative PnL when a signal was active (backward compat — returns sum_y).
    pub fn signal_pnl(&self, signal: &str) -> f64 {
        self.stats.get(signal).map_or(0.0, |s| s.sum_y)
    }

    /// Cumulative PnL when a signal was inactive (backward compat — returns 0.0).
    pub fn signal_pnl_inactive(&self, _signal: &str) -> f64 {
        // No longer tracked with regression approach. Return 0.0 for backward compat.
        0.0
    }

    /// Marginal value: OLS slope of PnL on signal adjustment.
    ///
    /// Returns Cov(adjustment, pnl) / Var(adjustment), or 0.0 if insufficient data.
    pub fn marginal_value(&self, signal: &str) -> f64 {
        self.stats.get(signal).map_or(0.0, |s| s.slope())
    }

    /// Gated marginal value: returns `None` if the signal has fewer than
    /// `MIN_MARGINAL_OBSERVATIONS` observations.
    pub fn marginal_value_gated(&self, signal: &str) -> Option<f64> {
        let reg = self.stats.get(signal)?;
        if reg.count < MIN_MARGINAL_OBSERVATIONS {
            return None;
        }
        Some(reg.slope())
    }

    /// R² of the regression for a given signal.
    ///
    /// Measures what fraction of PnL variance is explained by this signal's adjustment.
    /// Low R² means the marginal value is unreliable.
    pub fn signal_r_squared(&self, signal: &str) -> f64 {
        self.stats.get(signal).map_or(0.0, |s| s.r_squared())
    }

    /// Whether a signal has sufficient data for reliable attribution.
    pub fn signal_has_sufficient_data(&self, signal: &str) -> bool {
        self.stats
            .get(signal)
            .is_some_and(|s| s.has_sufficient_data())
    }

    /// Number of observations for a signal.
    pub fn signal_count(&self, signal: &str) -> u64 {
        self.stats.get(signal).map_or(0, |s| s.count)
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

        let mut lines = vec!["Signal PnL Attribution (regression):".to_string()];
        for name in &names {
            let reg = &self.stats[*name];
            if reg.count < MIN_MARGINAL_OBSERVATIONS {
                lines.push(format!(
                    "  {name}: marginal=N/A (n={}, need {MIN_MARGINAL_OBSERVATIONS}) [insufficient samples]",
                    reg.count,
                ));
            } else {
                let slope = reg.slope();
                let r2 = reg.r_squared();
                let quality = if !reg.has_sufficient_data() {
                    " [insufficient data]"
                } else if r2 < 0.01 {
                    " [weak]"
                } else {
                    ""
                };
                lines.push(format!(
                    "  {name}: marginal={slope:.2}bps/bps R²={r2:.3} n={}{quality}",
                    reg.count,
                ));
            }
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

    fn make_cycle(cycle_id: u64, signals: &[(&str, f64, f64)]) -> CycleContributions {
        CycleContributions {
            cycle_id,
            timestamp_ns: cycle_id * 1_000_000_000,
            contributions: signals
                .iter()
                .map(|(name, spread_adj, skew_adj)| SignalContribution {
                    signal_name: name.to_string(),
                    spread_adjustment_bps: *spread_adj,
                    skew_adjustment_bps: *skew_adj,
                    was_active: (*spread_adj + *skew_adj).abs() > 0.0,
                    gating_weight: 1.0,
                    raw_value: *spread_adj + *skew_adj,
                })
                .collect(),
            total_spread_mult: 1.0,
            combined_skew_bps: 0.0,
        }
    }

    #[test]
    fn test_marginal_value_positive_correlation() {
        let mut attributor = SignalPnLAttributor::new();

        // Signal A adjustment is positively correlated with PnL:
        // Higher adjustment → higher PnL
        for i in 0..100 {
            let x = (i as f64) * 0.1; // 0.0 to 9.9 bps adjustment
            let pnl = 2.0 * x + 1.0; // PnL = 2*adjustment + 1 (slope=2)
            let cycle = make_cycle(i, &[("SignalA", x, 0.0)]);
            attributor.record_cycle(&cycle, pnl);
        }

        let marginal = attributor.marginal_value("SignalA");
        assert!(
            (marginal - 2.0).abs() < 0.01,
            "Expected slope ~2.0, got {marginal}"
        );

        let r2 = attributor.signal_r_squared("SignalA");
        assert!(r2 > 0.99, "Expected R² ~1.0 for perfect linear, got {r2}");
    }

    #[test]
    fn test_marginal_value_negative_correlation() {
        let mut attributor = SignalPnLAttributor::new();

        // Negative relationship: more spread widening → worse PnL (overcautious)
        for i in 0..100 {
            let x = (i as f64) * 0.1;
            let pnl = -0.5 * x + 3.0; // slope = -0.5
            let cycle = make_cycle(i, &[("Cautious", x, 0.0)]);
            attributor.record_cycle(&cycle, pnl);
        }

        let marginal = attributor.marginal_value("Cautious");
        assert!(
            (marginal - (-0.5)).abs() < 0.01,
            "Expected slope ~-0.5, got {marginal}"
        );
    }

    #[test]
    fn test_zero_variance_returns_zero() {
        let mut attributor = SignalPnLAttributor::new();

        // Signal always has same adjustment (zero variance in x)
        for i in 0..20 {
            let cycle = make_cycle(i, &[("Constant", 5.0, 0.0)]);
            attributor.record_cycle(&cycle, (i as f64) * 0.5);
        }

        // No variance in adjustment → slope undefined → returns 0.0
        assert_eq!(attributor.marginal_value("Constant"), 0.0);
        assert!(!attributor.signal_has_sufficient_data("Constant"));
    }

    #[test]
    fn test_multiple_signals_differentiated() {
        let mut attributor = SignalPnLAttributor::new();

        // Two signals with different adjustment patterns
        for i in 0..100 {
            let x_lead = (i as f64) * 0.2; // LeadLag varies 0-20
            let x_vpin = ((i % 10) as f64) * 0.5; // VPIN varies 0-4.5 cyclically
            let pnl = 1.5 * x_lead + 0.3 * x_vpin;
            let cycle = make_cycle(i, &[("LeadLag", 0.0, x_lead), ("VPIN", x_vpin, 0.0)]);
            attributor.record_cycle(&cycle, pnl);
        }

        let marginal_ll = attributor.marginal_value("LeadLag");
        let marginal_vpin = attributor.marginal_value("VPIN");

        // They should be different (the whole point of the fix)
        assert!(
            (marginal_ll - marginal_vpin).abs() > 0.1,
            "Signals should have different marginals: LeadLag={marginal_ll:.2}, VPIN={marginal_vpin:.2}"
        );
    }

    #[test]
    fn test_skew_plus_spread_used_as_x() {
        let mut attributor = SignalPnLAttributor::new();

        // Signal has both spread and skew adjustments
        for i in 0..50 {
            let spread_adj = (i as f64) * 0.1;
            let skew_adj = (i as f64) * 0.2;
            let total_x = spread_adj + skew_adj; // 0.3 * i
            let pnl = 1.0 * total_x;
            let cycle = make_cycle(i, &[("Combined", spread_adj, skew_adj)]);
            attributor.record_cycle(&cycle, pnl);
        }

        let marginal = attributor.marginal_value("Combined");
        assert!(
            (marginal - 1.0).abs() < 0.01,
            "Expected slope ~1.0, got {marginal}"
        );
    }

    #[test]
    fn test_unknown_signal() {
        let attributor = SignalPnLAttributor::new();
        assert_eq!(attributor.signal_pnl("NonExistent"), 0.0);
        assert_eq!(attributor.marginal_value("NonExistent"), 0.0);
        assert_eq!(attributor.signal_r_squared("NonExistent"), 0.0);
    }

    #[test]
    fn test_format_report_empty() {
        let attributor = SignalPnLAttributor::new();
        assert_eq!(attributor.format_report(), "No attribution data");
    }

    #[test]
    fn test_format_report_with_data() {
        let mut attributor = SignalPnLAttributor::new();
        for i in 0..20 {
            let x = (i as f64) * 0.5;
            let pnl = 2.0 * x + 1.0;
            let cycle = make_cycle(i, &[("TestSignal", x, 0.0)]);
            attributor.record_cycle(&cycle, pnl);
        }

        let report = attributor.format_report();
        assert!(report.contains("TestSignal"));
        assert!(report.contains("marginal="));
        assert!(report.contains("R²="));
    }

    #[test]
    fn test_r_squared_noisy() {
        let mut attributor = SignalPnLAttributor::new();

        // Signal with some noise — R² should be less than 1.0
        for i in 0..200 {
            let x = (i as f64) * 0.1;
            let noise = if i % 3 == 0 { 2.0 } else { -1.0 };
            let pnl = 1.0 * x + noise;
            let cycle = make_cycle(i, &[("Noisy", x, 0.0)]);
            attributor.record_cycle(&cycle, pnl);
        }

        let r2 = attributor.signal_r_squared("Noisy");
        assert!(
            r2 > 0.5,
            "R² should be decent with moderate noise, got {r2}"
        );
        assert!(r2 < 1.0, "R² should be < 1.0 with noise, got {r2}");
    }

    #[test]
    fn test_marginal_value_gated_insufficient() {
        let mut attributor = SignalPnLAttributor::new();
        // Only 10 observations — below MIN_MARGINAL_OBSERVATIONS (20)
        for i in 0..10 {
            let x = (i as f64) * 0.5;
            let pnl = 2.0 * x + 1.0;
            let cycle = make_cycle(i, &[("FewSamples", x, 0.0)]);
            attributor.record_cycle(&cycle, pnl);
        }
        assert!(attributor.marginal_value_gated("FewSamples").is_none());
        // But ungated still returns a value
        assert!(attributor.marginal_value("FewSamples") != 0.0);
    }

    #[test]
    fn test_marginal_value_gated_sufficient() {
        let mut attributor = SignalPnLAttributor::new();
        // 30 observations — above MIN_MARGINAL_OBSERVATIONS (20)
        for i in 0..30 {
            let x = (i as f64) * 0.5;
            let pnl = 2.0 * x + 1.0;
            let cycle = make_cycle(i, &[("EnoughSamples", x, 0.0)]);
            attributor.record_cycle(&cycle, pnl);
        }
        let gated = attributor.marginal_value_gated("EnoughSamples");
        assert!(gated.is_some());
        assert!((gated.unwrap() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_insufficient_data_flag() {
        let mut attributor = SignalPnLAttributor::new();

        // Only 5 observations — below threshold
        for i in 0..5 {
            let cycle = make_cycle(i, &[("Few", (i as f64), 0.0)]);
            attributor.record_cycle(&cycle, (i as f64) * 2.0);
        }

        assert!(!attributor.signal_has_sufficient_data("Few"));
        assert_eq!(attributor.signal_count("Few"), 5);
    }
}
