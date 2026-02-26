# Daily Calibration Report — Implementation Reference

## Report Generator

```rust
struct DailyReportGenerator {
    prediction_store: PredictionStorage,
    signal_store: SignalStorage,
    config: ReportConfig,
}

struct ReportConfig {
    // Thresholds
    min_ir_warning: f64,          // 1.0
    min_ir_critical: f64,         // 0.8
    signal_decay_warning: f64,    // 0.15 (15% drop)
    max_cascade_loss: f64,        // $50

    // Comparison windows
    comparison_window_days: usize, // 7
}

impl DailyReportGenerator {
    fn generate(&self, date: NaiveDate) -> DailyReport {
        // Load data
        let predictions = self.prediction_store.load_for_date(date);
        let predictions_7d = self.prediction_store.load_range(
            date - Duration::days(7),
            date,
        );

        // Compute sections
        let pnl = self.compute_pnl_attribution(&predictions);
        let pnl_7d = self.compute_pnl_attribution(&predictions_7d);

        let calibration = self.compute_calibration_metrics(&predictions);
        let calibration_7d = self.compute_calibration_metrics(&predictions_7d);

        let signal_health = self.compute_signal_health(date);
        let regime_dist = self.compute_regime_distribution(&predictions);
        let conditional = self.compute_conditional_issues(&predictions);

        // Generate action items
        let actions = self.generate_action_items(
            &calibration,
            &calibration_7d,
            &signal_health,
            &regime_dist,
            &pnl,
        );

        // Determine overall status
        let status = self.determine_status(&actions);

        DailyReport {
            date,
            status,
            pnl,
            pnl_comparison: pnl_7d,
            calibration,
            calibration_comparison: calibration_7d,
            signal_health,
            regime_distribution: regime_dist,
            conditional_issues: conditional,
            action_items: actions,
        }
    }

    fn generate_action_items(
        &self,
        cal: &CalibrationMetrics,
        cal_7d: &CalibrationMetrics,
        signals: &SignalHealth,
        regimes: &RegimeDistribution,
        pnl: &PnLAttribution,
    ) -> Vec<ActionItem> {
        let mut items = Vec::new();

        // Check model IRs
        for (model, metrics) in &cal.by_model {
            if metrics.information_ratio < self.config.min_ir_critical {
                items.push(ActionItem {
                    priority: Priority::High,
                    category: "Model Calibration".to_string(),
                    message: format!(
                        "{} IR = {:.2} < {:.2} (critical threshold)",
                        model, metrics.information_ratio, self.config.min_ir_critical
                    ),
                    recommendation: "Consider disabling model or retraining immediately".to_string(),
                });
            } else if metrics.information_ratio < self.config.min_ir_warning {
                items.push(ActionItem {
                    priority: Priority::High,
                    category: "Model Calibration".to_string(),
                    message: format!(
                        "{} IR = {:.2} < {:.2} (warning threshold)",
                        model, metrics.information_ratio, self.config.min_ir_warning
                    ),
                    recommendation: "Check conditional calibration, consider retraining".to_string(),
                });
            }
        }

        // Check regime PnL
        if let Some(cascade_pnl) = regimes.pnl_by_regime.get(&MarketRegime::Cascade) {
            if *cascade_pnl < -self.config.max_cascade_loss {
                items.push(ActionItem {
                    priority: Priority::Medium,
                    category: "Regime Performance".to_string(),
                    message: format!("Cascade regime loss: ${:.2}", cascade_pnl),
                    recommendation: "Review liquidation detector thresholds".to_string(),
                });
            }
        }

        // Check signal decay
        for signal in &signals.signals {
            let decay_rate = (signal.mi_7d_ago - signal.mi_today) / signal.mi_7d_ago;
            if decay_rate > self.config.signal_decay_warning {
                items.push(ActionItem {
                    priority: Priority::Medium,
                    category: "Signal Health".to_string(),
                    message: format!(
                        "Signal {} MI dropped {:.1}%",
                        signal.name, decay_rate * 100.0
                    ),
                    recommendation: "Run signal audit, consider removing from features".to_string(),
                });
            }
        }

        // Sort by priority
        items.sort_by_key(|i| match i.priority {
            Priority::High => 0,
            Priority::Medium => 1,
            Priority::Low => 2,
        });

        items
    }

    fn determine_status(&self, actions: &[ActionItem]) -> ReportStatus {
        let high_count = actions.iter()
            .filter(|a| a.priority == Priority::High)
            .count();
        let medium_count = actions.iter()
            .filter(|a| a.priority == Priority::Medium)
            .count();

        if high_count > 0 {
            ReportStatus::Critical
        } else if medium_count > 2 {
            ReportStatus::Warning
        } else if medium_count > 0 {
            ReportStatus::Caution
        } else {
            ReportStatus::Ok
        }
    }
}
```

## Automation

```rust
// Cron job: Run at 6:00 AM daily
fn daily_report_job() {
    let generator = DailyReportGenerator::new();
    let alert_manager = AlertManager::from_config();

    let yesterday = Local::now().date_naive() - Duration::days(1);

    // Generate report
    let report = generator.generate(yesterday);

    // Save to file
    let report_path = format!("reports/{}.md", yesterday);
    std::fs::write(&report_path, report.to_markdown())?;

    // Send alerts
    alert_manager.send_report_alerts(&report);

    // Update dashboard
    update_grafana_dashboard(&report);
}
```
