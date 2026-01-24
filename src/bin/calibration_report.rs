//! Daily Calibration Report Generator
//!
//! Analyzes model performance and generates health reports.
//! Tracks Information Ratio, Brier scores, and signal health for
//! daily review of model calibration.
//!
//! Usage:
//!   calibration_report [OPTIONS]
//!
//! Options:
//!   --output <FILE>    Output file for JSON report (optional)
//!   --format <FMT>     Output format: ascii, json, both (default: ascii)
//!   --days <N>         Days of history to analyze (default: 1)
//!
//! Example:
//!   calibration_report --format both --output daily_report.json

use clap::Parser;
use serde::{Deserialize, Serialize};

use hyperliquid_rust_sdk::market_maker::{
    BrierScoreTracker, EdgeSignalKind, InformationRatioTracker, SignalHealthMonitor,
};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser)]
#[command(name = "calibration_report")]
#[command(version, about = "Generate daily model calibration reports")]
struct Cli {
    /// Output file for JSON report
    #[arg(short, long)]
    output: Option<String>,

    /// Output format: ascii, json, both
    #[arg(short, long, default_value = "ascii")]
    format: String,

    /// Days of history to analyze
    #[arg(short, long, default_value = "1")]
    days: u32,

    /// Simulate demo data (for testing without live data)
    #[arg(long)]
    demo: bool,
}

// ============================================================================
// Report Data Structures
// ============================================================================

/// Report for a single model's calibration metrics.
#[derive(Debug, Serialize, Deserialize)]
struct ModelReport {
    /// Model name
    name: String,
    /// Information Ratio (>1.0 means model adds value)
    information_ratio: f64,
    /// Brier score (lower is better, 0 = perfect)
    brier_score: f64,
    /// Calibration error (mean absolute deviation from perfect calibration)
    calibration_error: f64,
    /// Number of predictions in the evaluation window
    n_predictions: usize,
    /// Model health status
    status: String,
    /// Detailed status message
    status_detail: String,
}

impl ModelReport {
    /// Determine health status based on metrics.
    fn compute_status(ir: f64, brier: f64, n_predictions: usize) -> (String, String) {
        if n_predictions < 100 {
            return (
                "insufficient_data".to_string(),
                format!("Need 100+ samples, have {}", n_predictions),
            );
        }

        if ir >= 1.2 && brier < 0.2 {
            ("healthy".to_string(), "Strong predictive value".to_string())
        } else if ir >= 1.0 && brier < 0.25 {
            ("healthy".to_string(), "Acceptable calibration".to_string())
        } else if ir >= 0.9 {
            (
                "degraded".to_string(),
                format!("IR marginal ({:.2}), monitor closely", ir),
            )
        } else if ir < 0.9 {
            (
                "stale".to_string(),
                format!("IR below threshold ({:.2} < 1.0), consider removal", ir),
            )
        } else {
            (
                "degraded".to_string(),
                format!("Brier score high ({:.2})", brier),
            )
        }
    }
}

/// Summary statistics for the report.
#[derive(Debug, Serialize, Deserialize)]
struct ReportSummary {
    /// Total number of models evaluated
    total_models: usize,
    /// Number of healthy models
    healthy_models: usize,
    /// Number of degraded models
    degraded_models: usize,
    /// Number of stale models
    stale_models: usize,
    /// Average Information Ratio across all models
    average_ir: f64,
    /// Average Brier score across all models
    average_brier: f64,
    /// Recommended action based on overall health
    recommendation: String,
}

/// Signal health report for edge maintenance.
#[derive(Debug, Serialize, Deserialize)]
struct SignalReport {
    /// Signal type name
    signal_type: String,
    /// Current mutual information relative to baseline (1.0 = baseline)
    relative_strength: f64,
    /// Estimated decay rate (% per day)
    decay_rate_per_day: f64,
    /// Health status
    status: String,
}

/// Complete calibration report.
#[derive(Debug, Serialize, Deserialize)]
struct CalibrationReport {
    /// ISO 8601 timestamp when report was generated
    generated_at: String,
    /// Number of days of history analyzed
    period_days: u32,
    /// Individual model reports
    models: Vec<ModelReport>,
    /// Signal health reports
    signals: Vec<SignalReport>,
    /// Overall system health
    overall_health: String,
    /// Active alerts
    alerts: Vec<String>,
    /// Summary statistics
    summary: ReportSummary,
}

// ============================================================================
// Report Generation
// ============================================================================

/// Generate a calibration report.
///
/// In production, this would load actual prediction logs from storage.
/// For now, we demonstrate the report structure with simulated data
/// or return empty reports if no demo flag is set.
fn generate_report(days: u32, demo: bool) -> CalibrationReport {
    let mut alerts = Vec::new();

    let (models, signals) = if demo {
        // Generate demo data for illustration
        (
            generate_demo_model_reports(),
            generate_demo_signal_reports(),
        )
    } else {
        // In production: load from prediction log files
        // For now, generate trackers and report their state
        (
            generate_live_model_reports(),
            generate_live_signal_reports(),
        )
    };

    // Count health states
    let healthy = models.iter().filter(|m| m.status == "healthy").count();
    let degraded = models.iter().filter(|m| m.status == "degraded").count();
    let stale = models.iter().filter(|m| m.status == "stale").count();

    // Calculate averages
    let total = models.len().max(1);
    let avg_ir = models.iter().map(|m| m.information_ratio).sum::<f64>() / total as f64;
    let avg_brier = models.iter().map(|m| m.brier_score).sum::<f64>() / total as f64;

    // Generate alerts
    for model in &models {
        if model.information_ratio < 1.0 && model.n_predictions >= 100 {
            alerts.push(format!(
                "MODEL ALERT: {} IR below 1.0 ({:.2}) - model may be adding noise",
                model.name, model.information_ratio
            ));
        }
        if model.brier_score > 0.3 && model.n_predictions >= 100 {
            alerts.push(format!(
                "MODEL ALERT: {} Brier score high ({:.2}) - poor calibration",
                model.name, model.brier_score
            ));
        }
    }

    for signal in &signals {
        if signal.relative_strength < 0.5 {
            alerts.push(format!(
                "SIGNAL ALERT: {} is stale (strength {:.0}% of baseline)",
                signal.signal_type,
                signal.relative_strength * 100.0
            ));
        } else if signal.decay_rate_per_day > 5.0 {
            alerts.push(format!(
                "SIGNAL ALERT: {} decaying rapidly ({:.1}%/day)",
                signal.signal_type, signal.decay_rate_per_day
            ));
        }
    }

    // Determine overall health
    let overall_health = if stale > 0 || alerts.len() > 2 {
        "CRITICAL"
    } else if degraded > 0 || !alerts.is_empty() {
        "WARNING"
    } else {
        "HEALTHY"
    };

    // Generate recommendation
    let recommendation = if stale > 0 {
        "CRITICAL: Stale models detected. Consider widening spreads or pausing trading.".to_string()
    } else if degraded > 1 {
        "WARNING: Multiple degraded models. Review model parameters and reduce position sizes."
            .to_string()
    } else if degraded == 1 {
        "CAUTION: One model degraded. Monitor closely and consider reducing reliance.".to_string()
    } else if healthy == total && avg_ir > 1.1 {
        "OPTIMAL: All models healthy with strong IR. Continue trading normally.".to_string()
    } else {
        "OK: Models within acceptable parameters. Continue monitoring.".to_string()
    };

    CalibrationReport {
        generated_at: chrono::Utc::now().to_rfc3339(),
        period_days: days,
        models,
        signals,
        overall_health: overall_health.to_string(),
        alerts,
        summary: ReportSummary {
            total_models: total,
            healthy_models: healthy,
            degraded_models: degraded,
            stale_models: stale,
            average_ir: avg_ir,
            average_brier: avg_brier,
            recommendation,
        },
    }
}

/// Generate demo model reports for illustration.
fn generate_demo_model_reports() -> Vec<ModelReport> {
    vec![
        {
            let (status, status_detail) = ModelReport::compute_status(1.35, 0.18, 1250);
            ModelReport {
                name: "fill_probability".to_string(),
                information_ratio: 1.35,
                brier_score: 0.18,
                calibration_error: 0.045,
                n_predictions: 1250,
                status,
                status_detail,
            }
        },
        {
            let (status, status_detail) = ModelReport::compute_status(1.22, 0.21, 890);
            ModelReport {
                name: "adverse_selection".to_string(),
                information_ratio: 1.22,
                brier_score: 0.21,
                calibration_error: 0.062,
                n_predictions: 890,
                status,
                status_detail,
            }
        },
        {
            let (status, status_detail) = ModelReport::compute_status(1.08, 0.25, 2400);
            ModelReport {
                name: "regime_detection".to_string(),
                information_ratio: 1.08,
                brier_score: 0.25,
                calibration_error: 0.078,
                n_predictions: 2400,
                status,
                status_detail,
            }
        },
        {
            let (status, status_detail) = ModelReport::compute_status(0.92, 0.32, 450);
            ModelReport {
                name: "lead_lag".to_string(),
                information_ratio: 0.92,
                brier_score: 0.32,
                calibration_error: 0.112,
                n_predictions: 450,
                status,
                status_detail,
            }
        },
    ]
}

/// Generate live model reports from actual trackers.
fn generate_live_model_reports() -> Vec<ModelReport> {
    // Create trackers (in production, these would be loaded from storage)
    let fill_brier = BrierScoreTracker::new(1000);
    let fill_ir = InformationRatioTracker::new(10);

    let as_brier = BrierScoreTracker::new(1000);
    let as_ir = InformationRatioTracker::new(10);

    // Build reports with current state (empty since no data loaded)
    vec![
        {
            let n = fill_ir.n_samples();
            let ir = fill_ir.information_ratio();
            let brier = fill_brier.score();
            let cal_err = fill_ir.calibration_error();
            let (status, status_detail) = ModelReport::compute_status(ir, brier, n);
            ModelReport {
                name: "fill_probability".to_string(),
                information_ratio: ir,
                brier_score: brier,
                calibration_error: cal_err,
                n_predictions: n,
                status,
                status_detail,
            }
        },
        {
            let n = as_ir.n_samples();
            let ir = as_ir.information_ratio();
            let brier = as_brier.score();
            let cal_err = as_ir.calibration_error();
            let (status, status_detail) = ModelReport::compute_status(ir, brier, n);
            ModelReport {
                name: "adverse_selection".to_string(),
                information_ratio: ir,
                brier_score: brier,
                calibration_error: cal_err,
                n_predictions: n,
                status,
                status_detail,
            }
        },
    ]
}

/// Generate demo signal reports for illustration.
fn generate_demo_signal_reports() -> Vec<SignalReport> {
    vec![
        SignalReport {
            signal_type: "LeadLag".to_string(),
            relative_strength: 0.85,
            decay_rate_per_day: 1.2,
            status: "healthy".to_string(),
        },
        SignalReport {
            signal_type: "FundingPrediction".to_string(),
            relative_strength: 0.92,
            decay_rate_per_day: 0.5,
            status: "healthy".to_string(),
        },
        SignalReport {
            signal_type: "RegimeDetection".to_string(),
            relative_strength: 0.78,
            decay_rate_per_day: 2.1,
            status: "healthy".to_string(),
        },
        SignalReport {
            signal_type: "AdverseSelection".to_string(),
            relative_strength: 0.65,
            decay_rate_per_day: 3.8,
            status: "degraded".to_string(),
        },
    ]
}

/// Generate live signal reports from actual monitor.
fn generate_live_signal_reports() -> Vec<SignalReport> {
    let mut monitor = SignalHealthMonitor::new(0.5, 0.75);

    // Register known signal types (in production, load from config)
    monitor.register_signal(EdgeSignalKind::LeadLag, "Lead-Lag".to_string(), 1.0);
    monitor.register_signal(
        EdgeSignalKind::FundingPrediction,
        "Funding".to_string(),
        1.0,
    );
    monitor.register_signal(EdgeSignalKind::RegimeDetection, "Regime".to_string(), 1.0);
    monitor.register_signal(
        EdgeSignalKind::AdverseSelection,
        "Adverse Selection".to_string(),
        1.0,
    );
    monitor.register_signal(
        EdgeSignalKind::FillProbability,
        "Fill Probability".to_string(),
        1.0,
    );

    // Build reports from current state
    let mut reports = Vec::new();

    for signal_kind in [
        EdgeSignalKind::LeadLag,
        EdgeSignalKind::FundingPrediction,
        EdgeSignalKind::RegimeDetection,
        EdgeSignalKind::AdverseSelection,
        EdgeSignalKind::FillProbability,
    ] {
        if let Some(health) = monitor.get_health(signal_kind) {
            let status = if health.is_stale() {
                "stale"
            } else if health.is_degraded() {
                "degraded"
            } else {
                "healthy"
            };

            reports.push(SignalReport {
                signal_type: format!("{}", signal_kind),
                relative_strength: health.relative_strength(),
                decay_rate_per_day: health.decay_rate,
                status: status.to_string(),
            });
        }
    }

    reports
}

// ============================================================================
// Output Formatting
// ============================================================================

/// Print ASCII formatted report to stdout.
fn print_ascii(report: &CalibrationReport) {
    let border = "=".repeat(72);
    let separator = "-".repeat(72);

    println!("{}", border);
    println!("{:^72}", "DAILY CALIBRATION REPORT");
    println!("{}", border);
    println!(
        " Generated: {}",
        &report.generated_at[..19].replace('T', " ")
    );
    println!(" Period: {} day(s)", report.period_days);
    println!(
        " Overall Health: {}",
        format_health_status(&report.overall_health)
    );
    println!("{}", separator);

    // Model Performance Section
    println!(" MODEL PERFORMANCE");
    println!("{}", separator);
    println!(
        " {:20} {:>8} {:>8} {:>8} {:>9} {:14}",
        "Model", "IR", "Brier", "CalErr", "N Preds", "Status"
    );
    println!("{}", separator);

    for model in &report.models {
        let status_icon = match model.status.as_str() {
            "healthy" => "[OK]",
            "degraded" => "[!!]",
            "stale" => "[XX]",
            _ => "[??]",
        };

        println!(
            " {:20} {:>8.2} {:>8.3} {:>8.3} {:>9} {} {}",
            truncate_str(&model.name, 20),
            model.information_ratio,
            model.brier_score,
            model.calibration_error,
            model.n_predictions,
            status_icon,
            truncate_str(&model.status, 10)
        );
    }

    println!("{}", separator);

    // Signal Health Section
    if !report.signals.is_empty() {
        println!(" SIGNAL HEALTH");
        println!("{}", separator);
        println!(
            " {:20} {:>12} {:>12} {:>12}",
            "Signal", "Strength", "Decay/Day", "Status"
        );
        println!("{}", separator);

        for signal in &report.signals {
            let status_icon = match signal.status.as_str() {
                "healthy" => "[OK]",
                "degraded" => "[!!]",
                "stale" => "[XX]",
                _ => "[??]",
            };

            println!(
                " {:20} {:>11.0}% {:>11.1}% {} {}",
                truncate_str(&signal.signal_type, 20),
                signal.relative_strength * 100.0,
                signal.decay_rate_per_day,
                status_icon,
                signal.status
            );
        }

        println!("{}", separator);
    }

    // Summary Section
    println!(" SUMMARY");
    println!("{}", separator);
    println!(
        " Models: {} total | {} healthy | {} degraded | {} stale",
        report.summary.total_models,
        report.summary.healthy_models,
        report.summary.degraded_models,
        report.summary.stale_models
    );
    println!(
        " Average IR: {:.2} | Average Brier: {:.3}",
        report.summary.average_ir, report.summary.average_brier
    );
    println!("{}", separator);

    // Alerts Section
    if !report.alerts.is_empty() {
        println!(" ALERTS ({} active)", report.alerts.len());
        println!("{}", separator);
        for alert in &report.alerts {
            println!(" * {}", alert);
        }
        println!("{}", separator);
    }

    // Recommendation
    println!(" RECOMMENDATION");
    println!("{}", separator);
    println!(" {}", report.summary.recommendation);
    println!("{}", border);
}

/// Print JSON formatted report to stdout.
fn print_json(report: &CalibrationReport) {
    match serde_json::to_string_pretty(report) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error serializing report to JSON: {}", e),
    }
}

/// Format health status for display.
fn format_health_status(status: &str) -> String {
    match status {
        "HEALTHY" => "HEALTHY [All systems nominal]".to_string(),
        "WARNING" => "WARNING [Issues detected - review alerts]".to_string(),
        "CRITICAL" => "CRITICAL [Immediate action required]".to_string(),
        _ => status.to_string(),
    }
}

/// Truncate a string to a maximum length.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    let cli = Cli::parse();

    // Generate the report
    let report = generate_report(cli.days, cli.demo);

    // Output based on format
    match cli.format.as_str() {
        "json" => print_json(&report),
        "ascii" => print_ascii(&report),
        "both" => {
            print_ascii(&report);
            println!("\n--- JSON Output ---\n");
            print_json(&report);
        }
        _ => {
            eprintln!(
                "Unknown format '{}'. Using ascii. Valid formats: ascii, json, both",
                cli.format
            );
            print_ascii(&report);
        }
    }

    // Write to file if specified
    if let Some(output_path) = cli.output {
        match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&output_path, json) {
                    eprintln!("Failed to write report to {}: {}", output_path, e);
                } else {
                    println!("\nReport written to: {}", output_path);
                }
            }
            Err(e) => eprintln!("Failed to serialize report: {}", e),
        }
    }

    // Exit with appropriate code based on health
    let exit_code = match report.overall_health.as_str() {
        "HEALTHY" => 0,
        "WARNING" => 1,
        "CRITICAL" => 2,
        _ => 0,
    };

    std::process::exit(exit_code);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_report_compute_status_healthy() {
        let (status, _) = ModelReport::compute_status(1.5, 0.15, 500);
        assert_eq!(status, "healthy");
    }

    #[test]
    fn test_model_report_compute_status_degraded() {
        let (status, _) = ModelReport::compute_status(0.95, 0.20, 500);
        assert_eq!(status, "degraded");
    }

    #[test]
    fn test_model_report_compute_status_stale() {
        let (status, _) = ModelReport::compute_status(0.8, 0.35, 500);
        assert_eq!(status, "stale");
    }

    #[test]
    fn test_model_report_compute_status_insufficient_data() {
        let (status, _) = ModelReport::compute_status(1.5, 0.15, 50);
        assert_eq!(status, "insufficient_data");
    }

    #[test]
    fn test_generate_demo_report() {
        let report = generate_report(1, true);

        assert!(!report.models.is_empty());
        assert!(!report.signals.is_empty());
        assert!(!report.generated_at.is_empty());
        assert_eq!(report.period_days, 1);
    }

    #[test]
    fn test_generate_live_report() {
        let report = generate_report(1, false);

        // Live reports should have model entries even if empty
        assert!(!report.models.is_empty());
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 8), "hello...");
        assert_eq!(truncate_str("hi", 2), "hi");
    }

    #[test]
    fn test_format_health_status() {
        assert!(format_health_status("HEALTHY").contains("nominal"));
        assert!(format_health_status("WARNING").contains("review"));
        assert!(format_health_status("CRITICAL").contains("action"));
    }

    #[test]
    fn test_report_serialization() {
        let report = generate_report(1, true);
        let json = serde_json::to_string(&report);
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("generated_at"));
        assert!(json_str.contains("models"));
        assert!(json_str.contains("summary"));
    }

    #[test]
    fn test_alerts_generation() {
        let report = generate_report(1, true);

        // Demo data includes a model with IR < 1.0, should generate alert
        let has_ir_alert = report.alerts.iter().any(|a| a.contains("IR below 1.0"));
        assert!(has_ir_alert, "Expected IR alert for demo data");
    }

    #[test]
    fn test_summary_calculation() {
        let report = generate_report(1, true);

        // Verify summary math
        let total = report.summary.healthy_models
            + report.summary.degraded_models
            + report.summary.stale_models;

        // Total should account for all non-insufficient models
        assert!(total <= report.summary.total_models);
        assert!(report.summary.average_ir > 0.0);
    }
}
