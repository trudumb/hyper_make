//! Post-session summary report.
//!
//! Gathers all session metrics and writes a JSON summary + human-readable table
//! on shutdown.

use std::time::Instant;

use serde::Serialize;

/// Tracks time spent in each volatility regime (4 buckets: Low, Normal, High, Extreme).
#[derive(Debug, Clone, Default)]
pub struct RegimeTimeTracker {
    /// Accumulated seconds per regime index [Low=0, Normal=1, High=2, Extreme=3].
    pub regime_time_secs: [f64; 4],
    /// Last observed regime index.
    last_regime_idx: usize,
    /// When the last regime was entered.
    last_regime_time: Option<Instant>,
}

impl RegimeTimeTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with the current regime index. Accumulates elapsed time into the previous regime.
    pub fn update(&mut self, regime_idx: usize) {
        let now = Instant::now();
        if let Some(last_time) = self.last_regime_time {
            let elapsed = now.duration_since(last_time).as_secs_f64();
            let idx = self.last_regime_idx.min(3);
            self.regime_time_secs[idx] += elapsed;
        }
        self.last_regime_idx = regime_idx.min(3);
        self.last_regime_time = Some(now);
    }

    /// Finalize: flush elapsed time for the current regime.
    pub fn finalize(&mut self) {
        if let Some(last_time) = self.last_regime_time {
            let elapsed = last_time.elapsed().as_secs_f64();
            let idx = self.last_regime_idx.min(3);
            self.regime_time_secs[idx] += elapsed;
            self.last_regime_time = None;
        }
    }

    /// Total tracked seconds.
    pub fn total_secs(&self) -> f64 {
        self.regime_time_secs.iter().sum()
    }

    /// Percentage of time in each regime.
    pub fn percentages(&self) -> [f64; 4] {
        let total = self.total_secs();
        if total < 1e-9 {
            return [0.0; 4];
        }
        [
            self.regime_time_secs[0] / total * 100.0,
            self.regime_time_secs[1] / total * 100.0,
            self.regime_time_secs[2] / total * 100.0,
            self.regime_time_secs[3] / total * 100.0,
        ]
    }
}

/// Tracks order-to-fill latency percentiles via a ring buffer.
#[derive(Debug, Clone)]
pub struct LatencyTracker {
    /// Recent latencies in milliseconds.
    latencies_ms: Vec<f64>,
    /// Maximum capacity.
    capacity: usize,
}

impl LatencyTracker {
    pub fn new(capacity: usize) -> Self {
        Self {
            latencies_ms: Vec::with_capacity(capacity.min(10_000)),
            capacity: capacity.min(10_000),
        }
    }

    pub fn record(&mut self, latency_ms: f64) {
        if self.latencies_ms.len() >= self.capacity {
            self.latencies_ms.remove(0);
        }
        self.latencies_ms.push(latency_ms);
    }

    pub fn count(&self) -> usize {
        self.latencies_ms.len()
    }

    /// Compute percentile (0-100). Returns None if empty.
    pub fn percentile(&self, pct: f64) -> Option<f64> {
        if self.latencies_ms.is_empty() {
            return None;
        }
        let mut sorted = self.latencies_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Per-regime KPI tracker with 4 buckets (Low, Normal, High, Extreme).
#[derive(Debug, Clone, Default)]
pub struct RegimeKPITracker {
    /// Fills per regime.
    pub fills: [u64; 4],
    /// Quote cycles per regime.
    pub cycles: [u64; 4],
    /// Sum of spread_bps per regime (for averaging).
    pub spread_sum_bps: [f64; 4],
    /// Sum of AS bps per regime (for averaging).
    pub as_sum_bps: [f64; 4],
}

impl RegimeKPITracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_cycle(&mut self, regime_idx: usize, spread_bps: f64) {
        let idx = regime_idx.min(3);
        self.cycles[idx] += 1;
        self.spread_sum_bps[idx] += spread_bps;
    }

    pub fn record_fill(&mut self, regime_idx: usize, as_bps: f64) {
        let idx = regime_idx.min(3);
        self.fills[idx] += 1;
        self.as_sum_bps[idx] += as_bps;
    }

    pub fn avg_spread_bps(&self, regime_idx: usize) -> f64 {
        let idx = regime_idx.min(3);
        if self.cycles[idx] == 0 {
            return 0.0;
        }
        self.spread_sum_bps[idx] / self.cycles[idx] as f64
    }

    pub fn avg_as_bps(&self, regime_idx: usize) -> f64 {
        let idx = regime_idx.min(3);
        if self.fills[idx] == 0 {
            return 0.0;
        }
        self.as_sum_bps[idx] / self.fills[idx] as f64
    }
}

/// Complete session summary written to JSON on shutdown.
#[derive(Debug, Clone, Serialize)]
pub struct SessionSummary {
    // Metadata
    pub asset: String,
    pub mode: String,
    pub duration_secs: f64,
    pub quote_cycles: u64,

    // PnL
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub spread_capture: f64,
    pub adverse_selection: f64,
    pub fees: f64,
    pub funding: f64,

    // Fills
    pub total_fills: usize,
    pub fill_rate_pct: f64,

    // Sharpe
    pub sharpe_1h: f64,
    pub sharpe_all: f64,
    pub sharpe_ci_lo: f64,
    pub sharpe_ci_hi: f64,
    pub equity_sharpe: f64,
    pub max_drawdown_pct: f64,

    // Estimator state
    pub sigma: f64,
    pub kappa: f64,
    pub gamma: f64,
    pub vol_regime: String,

    // Regime time breakdown
    pub regime_pct_low: f64,
    pub regime_pct_normal: f64,
    pub regime_pct_high: f64,
    pub regime_pct_extreme: f64,

    // Per-regime KPIs
    pub regime_fills: [u64; 4],
    pub regime_avg_spread_bps: [f64; 4],
    pub regime_avg_as_bps: [f64; 4],

    // Latency
    pub fill_latency_p50_ms: Option<f64>,
    pub fill_latency_p95_ms: Option<f64>,
    pub fill_latency_p99_ms: Option<f64>,

    // Risk
    pub kill_switch_triggered: bool,
    pub kill_switch_reasons: Vec<String>,

    // Signals
    pub signal_marginals: Vec<(String, f64)>,
}

/// Print a human-readable session summary to stdout.
pub fn print_session_summary(s: &SessionSummary) {
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!(
        "║               SESSION SUMMARY: {}                      ║",
        s.asset
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║ Mode: {:<10} Duration: {:.0}s  Cycles: {:<8}        ║",
        s.mode, s.duration_secs, s.quote_cycles
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ PnL                                                    ║");
    println!(
        "║   Total:     ${:<10.4}  (R: ${:.4} + U: ${:.4})    ║",
        s.total_pnl, s.realized_pnl, s.unrealized_pnl
    );
    println!(
        "║   Spread:    ${:<10.4}  AS: ${:<10.4}                ║",
        s.spread_capture, s.adverse_selection
    );
    println!(
        "║   Fees:      ${:<10.4}  Funding: ${:<10.4}           ║",
        s.fees, s.funding
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Fills & Sharpe                                         ║");
    println!(
        "║   Fills: {:<6} Fill Rate: {:.1}%                       ║",
        s.total_fills, s.fill_rate_pct
    );
    println!(
        "║   Sharpe(1h): {:.2}  Sharpe(all): {:.2} [{:.2}, {:.2}]  ║",
        s.sharpe_1h, s.sharpe_all, s.sharpe_ci_lo, s.sharpe_ci_hi
    );
    println!(
        "║   Equity Sharpe: {:.2}  Max DD: {:.2}%                  ║",
        s.equity_sharpe, s.max_drawdown_pct
    );
    if let (Some(p50), Some(p95)) = (s.fill_latency_p50_ms, s.fill_latency_p95_ms) {
        println!(
            "║   Latency P50: {:.0}ms  P95: {:.0}ms  P99: {:.0}ms       ║",
            p50,
            p95,
            s.fill_latency_p99_ms.unwrap_or(0.0)
        );
    }
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Estimator State                                        ║");
    println!(
        "║   σ: {:.6}  κ: {:.0}  γ: {:.4}  Regime: {}           ║",
        s.sigma, s.kappa, s.gamma, s.vol_regime
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ Regime Breakdown                                       ║");
    let labels = ["Low", "Normal", "High", "Extreme"];
    for (i, label) in labels.iter().enumerate() {
        let pct = [
            s.regime_pct_low,
            s.regime_pct_normal,
            s.regime_pct_high,
            s.regime_pct_extreme,
        ][i];
        if pct > 0.1 {
            println!(
                "║   {:<8} {:.1}%  fills={} spread={:.1}bps as={:.1}bps  ║",
                label, pct, s.regime_fills[i], s.regime_avg_spread_bps[i], s.regime_avg_as_bps[i]
            );
        }
    }
    if s.kill_switch_triggered {
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ ⚠ KILL SWITCH TRIGGERED                                ║");
        for reason in &s.kill_switch_reasons {
            println!(
                "║   - {}                                               ║",
                reason
            );
        }
    }
    if !s.signal_marginals.is_empty() {
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ Signal Marginal Values                                 ║");
        for (name, value) in &s.signal_marginals {
            println!(
                "║   {:<20} {:.1} bps                         ║",
                name, value
            );
        }
    } else {
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ Signal Marginal Values          (insufficient data)    ║");
    }
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_time_tracker() {
        let mut tracker = RegimeTimeTracker::new();
        tracker.update(0); // Low
        std::thread::sleep(std::time::Duration::from_millis(10));
        tracker.update(1); // Normal
        std::thread::sleep(std::time::Duration::from_millis(10));
        tracker.finalize();

        assert!(tracker.regime_time_secs[0] > 0.0);
        assert!(tracker.regime_time_secs[1] > 0.0);
        assert_eq!(tracker.regime_time_secs[2], 0.0);
        let pct = tracker.percentages();
        let total: f64 = pct.iter().sum();
        assert!((total - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_latency_tracker() {
        let mut tracker = LatencyTracker::new(100);
        for i in 0..50 {
            tracker.record(i as f64);
        }
        assert_eq!(tracker.count(), 50);
        let p50 = tracker.percentile(50.0).unwrap();
        assert!((p50 - 25.0).abs() < 2.0);
        let p99 = tracker.percentile(99.0).unwrap();
        assert!(p99 > 40.0);
    }

    #[test]
    fn test_regime_kpi_tracker() {
        let mut tracker = RegimeKPITracker::new();
        tracker.record_cycle(1, 10.0);
        tracker.record_cycle(1, 12.0);
        tracker.record_fill(1, 3.0);
        assert_eq!(tracker.cycles[1], 2);
        assert_eq!(tracker.fills[1], 1);
        assert!((tracker.avg_spread_bps(1) - 11.0).abs() < 0.01);
        assert!((tracker.avg_as_bps(1) - 3.0).abs() < 0.01);
    }
}
