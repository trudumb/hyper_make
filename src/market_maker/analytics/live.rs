//! Live analytics bundle for the production market maker.
//!
//! Bundles Sharpe tracking, per-signal attribution, and optional file logging
//! into a single struct that can be wired into the live event loop.
//!
//! The paper trader initializes these components individually; this module
//! provides a unified interface so the live market maker can do the same
//! with minimal boilerplate.

use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tracing::{info, warn};

use super::attribution::{CycleContributions, SignalContribution, SignalPnLAttributor};
use super::edge_metrics::EdgeSnapshot;
use super::persistence::AnalyticsLogger;
use super::sharpe::{PerSignalSharpeTracker, SharpeSummary, SharpeTracker};
use crate::market_maker::strategy::SignalContributionRecord;

/// Summary snapshot returned by `LiveAnalytics::summary()`.
#[derive(Debug, Clone)]
pub struct LiveAnalyticsSummary {
    /// Sharpe ratio statistics (1h, 24h, all-time).
    pub sharpe: SharpeSummary,
    /// Mean realized edge in bps (from EdgeTracker, if provided externally).
    pub mean_realized_edge_bps: f64,
    /// Per-signal marginal values in bps.
    pub signal_marginals: Vec<(String, f64)>,
    /// Total fills tracked by the Sharpe tracker.
    pub fill_count: usize,
}

/// Bundled analytics components for the live market maker.
///
/// Provides analytics for both paper and live environments. The `EdgeTracker` is NOT included
/// here because it already lives in `Tier2Components`.
pub struct LiveAnalytics {
    /// Return-based Sharpe ratio tracker.
    sharpe_tracker: SharpeTracker,
    /// Per-signal Sharpe ratio tracker.
    signal_sharpe: PerSignalSharpeTracker,
    /// Per-signal PnL attribution (active vs inactive).
    signal_attributor: SignalPnLAttributor,
    /// Optional JSONL file logger for offline analysis.
    logger: Option<AnalyticsLogger>,
    /// Most recent quote cycle's signal contributions.
    /// Retained so that when a fill arrives, we can attribute it.
    last_cycle_contributions: Option<CycleContributions>,
    /// Cycle counter for contribution records.
    cycle_counter: u64,
    /// Last time a periodic summary was logged.
    last_summary_time: Instant,
    /// Interval between periodic summary logs.
    summary_interval_secs: u64,
}

impl LiveAnalytics {
    /// Create a new `LiveAnalytics` bundle.
    ///
    /// If `log_dir` is `Some`, an `AnalyticsLogger` is created that writes
    /// JSONL files to that directory. If creation fails, a warning is logged
    /// and file logging is silently disabled.
    pub fn new(log_dir: Option<PathBuf>) -> Self {
        let logger = log_dir.and_then(|dir| {
            match AnalyticsLogger::new(dir.to_str().unwrap_or("data/analytics")) {
                Ok(l) => Some(l),
                Err(e) => {
                    warn!("Failed to initialize analytics logger: {}", e);
                    None
                }
            }
        });

        Self {
            sharpe_tracker: SharpeTracker::new(),
            signal_sharpe: PerSignalSharpeTracker::new(),
            signal_attributor: SignalPnLAttributor::new(),
            logger,
            last_cycle_contributions: None,
            cycle_counter: 0,
            last_summary_time: Instant::now(),
            summary_interval_secs: 30,
        }
    }

    /// Record a fill event and update all relevant trackers.
    ///
    /// # Arguments
    /// * `fill_pnl_bps` - PnL of the fill in basis points.
    /// * `edge_snapshot` - Optional edge snapshot for file logging (caller
    ///   builds this from the fill context; the `EdgeTracker` in
    ///   `Tier2Components` should also receive it).
    pub fn record_fill(&mut self, fill_pnl_bps: f64, edge_snapshot: Option<&EdgeSnapshot>) {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        // 1. Sharpe tracker
        self.sharpe_tracker.add_return(fill_pnl_bps, timestamp_ns);

        // 2. Per-signal Sharpe + attribution (if we have cycle contributions)
        if let Some(ref contribs) = self.last_cycle_contributions {
            for contrib in &contribs.contributions {
                if contrib.was_active {
                    self.signal_sharpe
                        .add_signal_return(&contrib.signal_name, fill_pnl_bps, timestamp_ns);
                }
            }
            self.signal_attributor.record_cycle(contribs, fill_pnl_bps);
        }

        // 3. Edge logging (file only; EdgeTracker lives in Tier2Components)
        if let (Some(ref mut logger), Some(snap)) = (&mut self.logger, edge_snapshot) {
            let _ = logger.log_edge(snap);
        }
    }

    /// Record the current quote cycle's signal contributions.
    ///
    /// Call this once per quote cycle with the output of
    /// `signal_integrator.get_signals()`. The contributions are retained
    /// so the next `record_fill` can attribute PnL to the active signals.
    pub fn record_quote_cycle(&mut self, signals: &crate::market_maker::strategy::IntegratedSignals) {
        if let Some(ref record) = signals.signal_contributions {
            let timestamp_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);

            self.cycle_counter += 1;

            let contributions = build_cycle_contributions(
                self.cycle_counter,
                timestamp_ns,
                record,
                signals.total_spread_mult,
                signals.combined_skew_bps,
            );

            if let Some(ref mut logger) = self.logger {
                let _ = logger.log_contributions(&contributions);
            }

            self.last_cycle_contributions = Some(contributions);
        }
    }

    /// Log a periodic analytics summary if the interval has elapsed.
    ///
    /// Returns `true` if a summary was logged, `false` otherwise.
    /// The `mean_realized_edge_bps` should come from the `EdgeTracker`
    /// in `Tier2Components`.
    pub fn maybe_log_summary(&mut self, mean_realized_edge_bps: f64) -> bool {
        if self.last_summary_time.elapsed().as_secs() < self.summary_interval_secs {
            return false;
        }

        let sharpe_summary = self.sharpe_tracker.summary();
        let (sharpe_all, sharpe_lo, sharpe_hi) = self.sharpe_tracker.sharpe_with_confidence(0.90);
        let data_quality = if self.sharpe_tracker.has_sufficient_data() {
            ""
        } else {
            " (insufficient data)"
        };
        info!(
            "[ANALYTICS] Sharpe(1h)={:.2} Sharpe(24h)={:.2} Sharpe(all)={:.2} [{:.2}, {:.2}]{} Fills={} Edge={:.1}bps",
            sharpe_summary.sharpe_1h,
            sharpe_summary.sharpe_24h,
            sharpe_all,
            sharpe_lo,
            sharpe_hi,
            data_quality,
            sharpe_summary.count,
            mean_realized_edge_bps,
        );

        if !self.signal_attributor.signal_names().is_empty() {
            let signal_parts: Vec<String> = self
                .signal_attributor
                .signal_names()
                .iter()
                .map(|name| {
                    format!(
                        "{}={:.1}bps",
                        name,
                        self.signal_attributor.marginal_value(name)
                    )
                })
                .collect();
            info!("[ANALYTICS] Signal marginal: {}", signal_parts.join(" "));
        }

        // Persist to disk
        if let Some(ref mut logger) = self.logger {
            let _ = logger.log_sharpe(&sharpe_summary);
            let _ = logger.log_signal_pnl(&self.signal_attributor);
            let _ = logger.flush();
        }

        self.last_summary_time = Instant::now();
        true
    }

    /// Flush all buffered data to disk.
    pub fn flush(&mut self) {
        if let Some(ref mut logger) = self.logger {
            let _ = logger.flush();
        }
    }

    /// Get a summary snapshot of the current analytics state.
    ///
    /// `mean_realized_edge_bps` should come from the `EdgeTracker`
    /// in `Tier2Components`.
    pub fn summary(&self, mean_realized_edge_bps: f64) -> LiveAnalyticsSummary {
        let sharpe = self.sharpe_tracker.summary();
        let signal_marginals: Vec<(String, f64)> = self
            .signal_attributor
            .signal_names()
            .iter()
            .map(|name| (name.clone(), self.signal_attributor.marginal_value(name)))
            .collect();

        LiveAnalyticsSummary {
            fill_count: sharpe.count,
            sharpe,
            mean_realized_edge_bps,
            signal_marginals,
        }
    }

    /// Get a reference to the Sharpe tracker.
    pub fn sharpe_tracker(&self) -> &SharpeTracker {
        &self.sharpe_tracker
    }

    /// Get a reference to the per-signal Sharpe tracker.
    pub fn signal_sharpe(&self) -> &PerSignalSharpeTracker {
        &self.signal_sharpe
    }

    /// Get a reference to the signal PnL attributor.
    pub fn signal_attributor(&self) -> &SignalPnLAttributor {
        &self.signal_attributor
    }

    /// Get the per-signal Sharpe report.
    pub fn signal_sharpe_report(&self) -> String {
        self.signal_sharpe.format_report()
    }

    /// Get the signal attribution report.
    pub fn attribution_report(&self) -> String {
        self.signal_attributor.format_report()
    }
}

/// Build a `CycleContributions` from a `SignalContributionRecord`.
///
/// Converts the raw signal contribution record into the attribution format.
fn build_cycle_contributions(
    cycle_id: u64,
    timestamp_ns: u64,
    record: &SignalContributionRecord,
    total_spread_mult: f64,
    combined_skew_bps: f64,
) -> CycleContributions {
    CycleContributions {
        cycle_id,
        timestamp_ns,
        contributions: vec![
            SignalContribution {
                signal_name: "LeadLag".to_string(),
                spread_adjustment_bps: 0.0,
                skew_adjustment_bps: record.lead_lag_skew_bps,
                was_active: record.lead_lag_active,
                gating_weight: record.lead_lag_gating_weight,
                raw_value: record.lead_lag_skew_bps,
            },
            SignalContribution {
                signal_name: "InformedFlow".to_string(),
                spread_adjustment_bps: (record.informed_flow_spread_mult - 1.0) * 100.0,
                skew_adjustment_bps: 0.0,
                was_active: record.informed_flow_active,
                gating_weight: record.informed_flow_gating_weight,
                raw_value: record.informed_flow_spread_mult,
            },
            SignalContribution {
                signal_name: "RegimeDetection".to_string(),
                spread_adjustment_bps: 0.0,
                skew_adjustment_bps: 0.0,
                was_active: record.regime_active,
                gating_weight: if record.regime_active { 1.0 } else { 0.0 },
                raw_value: record.regime_kappa_effective,
            },
            SignalContribution {
                signal_name: "CrossVenue".to_string(),
                spread_adjustment_bps: (record.cross_venue_spread_mult - 1.0) * 100.0,
                skew_adjustment_bps: record.cross_venue_skew_bps,
                was_active: record.cross_venue_active,
                gating_weight: if record.cross_venue_active { 1.0 } else { 0.0 },
                raw_value: record.cross_venue_spread_mult,
            },
            SignalContribution {
                signal_name: "VPIN".to_string(),
                spread_adjustment_bps: (record.vpin_spread_mult - 1.0) * 100.0,
                skew_adjustment_bps: 0.0,
                was_active: record.vpin_active,
                gating_weight: if record.vpin_active { 1.0 } else { 0.0 },
                raw_value: record.vpin_spread_mult,
            },
            SignalContribution {
                signal_name: "BuyPressure".to_string(),
                spread_adjustment_bps: 0.0,
                skew_adjustment_bps: record.buy_pressure_skew_bps,
                was_active: record.buy_pressure_active,
                gating_weight: if record.buy_pressure_active { 1.0 } else { 0.0 },
                raw_value: record.buy_pressure_skew_bps,
            },
        ],
        total_spread_mult,
        combined_skew_bps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_analytics_new_no_logger() {
        let analytics = LiveAnalytics::new(None);
        assert!(analytics.logger.is_none());
        assert_eq!(analytics.sharpe_tracker.count(), 0);
        assert!(analytics.last_cycle_contributions.is_none());
    }

    #[test]
    fn test_live_analytics_new_with_logger() {
        let dir = tempfile::tempdir().unwrap();
        let analytics = LiveAnalytics::new(Some(dir.path().to_path_buf()));
        assert!(analytics.logger.is_some());
    }

    #[test]
    fn test_record_fill_updates_sharpe() {
        let mut analytics = LiveAnalytics::new(None);
        analytics.record_fill(2.5, None);
        analytics.record_fill(-1.0, None);
        assert_eq!(analytics.sharpe_tracker.count(), 2);
    }

    #[test]
    fn test_record_fill_with_contributions_updates_attribution() {
        let mut analytics = LiveAnalytics::new(None);

        // Set up cycle contributions
        analytics.last_cycle_contributions = Some(CycleContributions {
            cycle_id: 1,
            timestamp_ns: 0,
            contributions: vec![
                SignalContribution {
                    signal_name: "LeadLag".to_string(),
                    spread_adjustment_bps: 0.0,
                    skew_adjustment_bps: 1.5,
                    was_active: true,
                    gating_weight: 0.8,
                    raw_value: 1.5,
                },
                SignalContribution {
                    signal_name: "VPIN".to_string(),
                    spread_adjustment_bps: 2.0,
                    skew_adjustment_bps: 0.0,
                    was_active: false,
                    gating_weight: 0.0,
                    raw_value: 1.02,
                },
            ],
            total_spread_mult: 1.02,
            combined_skew_bps: 1.5,
        });

        analytics.record_fill(3.0, None);

        // LeadLag was active, so it should have a signal Sharpe entry
        assert!(analytics.signal_sharpe.signal_sharpe("LeadLag").is_some());
        // VPIN was inactive, so no entry in signal_sharpe
        assert!(analytics.signal_sharpe.signal_sharpe("VPIN").is_none());

        // Attribution should have recorded both signals
        assert!(!analytics.signal_attributor.signal_names().is_empty());
    }

    #[test]
    fn test_record_quote_cycle_from_integrated_signals() {
        let mut analytics = LiveAnalytics::new(None);

        let mut signals = crate::market_maker::strategy::IntegratedSignals::default();
        signals.total_spread_mult = 1.05;
        signals.combined_skew_bps = 2.0;
        signals.signal_contributions = Some(SignalContributionRecord {
            lead_lag_skew_bps: 1.0,
            lead_lag_active: true,
            lead_lag_gating_weight: 0.9,
            informed_flow_spread_mult: 1.02,
            informed_flow_active: true,
            informed_flow_gating_weight: 0.7,
            regime_kappa_effective: 5000.0,
            regime_active: true,
            cross_venue_spread_mult: 1.0,
            cross_venue_skew_bps: 0.0,
            cross_venue_active: false,
            vpin_spread_mult: 1.03,
            vpin_active: true,
            buy_pressure_skew_bps: 0.5,
            buy_pressure_active: true,
            ..Default::default()
        });

        analytics.record_quote_cycle(&signals);

        assert!(analytics.last_cycle_contributions.is_some());
        let contribs = analytics.last_cycle_contributions.as_ref().unwrap();
        assert_eq!(contribs.cycle_id, 1);
        assert_eq!(contribs.contributions.len(), 6);
        assert_eq!(contribs.total_spread_mult, 1.05);
        assert_eq!(contribs.combined_skew_bps, 2.0);
    }

    #[test]
    fn test_summary_empty() {
        let analytics = LiveAnalytics::new(None);
        let summary = analytics.summary(0.0);
        assert_eq!(summary.fill_count, 0);
        assert!(summary.signal_marginals.is_empty());
    }

    #[test]
    fn test_build_cycle_contributions() {
        let record = SignalContributionRecord {
            lead_lag_skew_bps: 1.5,
            lead_lag_active: true,
            lead_lag_gating_weight: 0.9,
            informed_flow_spread_mult: 1.1,
            informed_flow_active: true,
            informed_flow_gating_weight: 0.8,
            regime_kappa_effective: 3000.0,
            regime_active: false,
            cross_venue_spread_mult: 1.0,
            cross_venue_skew_bps: 0.0,
            cross_venue_active: false,
            vpin_spread_mult: 1.05,
            vpin_active: true,
            buy_pressure_skew_bps: -0.5,
            buy_pressure_active: true,
            ..Default::default()
        };

        let contribs = build_cycle_contributions(42, 1234567890, &record, 1.15, 1.0);

        assert_eq!(contribs.cycle_id, 42);
        assert_eq!(contribs.contributions.len(), 6);

        // LeadLag
        let ll = &contribs.contributions[0];
        assert_eq!(ll.signal_name, "LeadLag");
        assert_eq!(ll.skew_adjustment_bps, 1.5);
        assert!(ll.was_active);

        // InformedFlow spread adjustment = (1.1 - 1.0) * 100 = 10.0
        let inf = &contribs.contributions[1];
        assert_eq!(inf.signal_name, "InformedFlow");
        assert!((inf.spread_adjustment_bps - 10.0).abs() < 0.01);

        // Regime inactive => gating_weight = 0.0
        let regime = &contribs.contributions[2];
        assert!(!regime.was_active);
        assert_eq!(regime.gating_weight, 0.0);
    }

    #[test]
    fn test_maybe_log_summary_respects_interval() {
        let mut analytics = LiveAnalytics::new(None);
        // Just created, so interval hasn't elapsed yet
        assert!(!analytics.maybe_log_summary(0.0));
    }

    #[test]
    fn test_flush_no_panic_without_logger() {
        let mut analytics = LiveAnalytics::new(None);
        analytics.flush(); // Should not panic
    }
}
