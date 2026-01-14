---
name: Daily Calibration Report
description: Automated daily health check of all model components for early warning of degradation and drift
---

# Daily Calibration Report Skill

## Purpose

Automated daily health check of all model components. This is your early warning system for:

- Model degradation
- Regime shifts
- Parameter drift
- Unexpected losses

## When to Use

- Run automatically every day at market close or 6am
- Manually when debugging performance issues
- After deploying model changes
- When something "feels off"

## Prerequisites

- `measurement-infrastructure` collecting data
- `calibration-analysis` for metrics computation
- At least 24 hours of production data

---

## Report Sections

### 1. Executive Summary

```
╔════════════════════════════════════════════════════════════════════╗
║             DAILY CALIBRATION REPORT - 2024-01-15                  ║
╠════════════════════════════════════════════════════════════════════╣
║  Overall Status: ⚠️  WARNING                                        ║
║                                                                    ║
║  PnL:           $347.82  (+12% vs 7d avg)                         ║
║  Sharpe (1d):   2.1                                               ║
║  Fill Rate:     847 fills                                         ║
║  Uptime:        99.7%                                             ║
║                                                                    ║
║  Issues:                                                           ║
║  • Adverse selection model IR dropped to 0.92 (threshold: 1.0)    ║
║  • Cascade regime PnL: -$45 (investigate)                         ║
╚════════════════════════════════════════════════════════════════════╝
```

### 2. PnL Attribution

```
PnL Attribution
═══════════════════════════════════════════════════════════════════
                        Today      7d Avg     Δ         Status
───────────────────────────────────────────────────────────────────
Gross PnL:              $347.82    $310.45    +12.0%    ✓ OK
├── Spread Capture:     $512.30    $478.22    +7.1%     ✓ OK
├── Adverse Selection:  -$89.45    -$102.33   +12.6%    ✓ IMPROVED
├── Inventory Cost:     -$23.18    -$18.44    -25.7%    ⚠ ELEVATED
└── Fees:               -$51.85    -$47.00    -10.3%    ✓ OK

PnL by Hour:
00:00 ████████████████ $48.22
01:00 ███████████████  $45.11
02:00 ████████████     $36.89
...
23:00 ██████████████████ $52.44
```

### 3. Model Calibration

```
Model Calibration Metrics
═══════════════════════════════════════════════════════════════════
                           Brier    IR      Δ IR (7d)   Status
───────────────────────────────────────────────────────────────────
Fill Prediction (1s):      0.142    1.34    -0.05       ✓ OK
Fill Prediction (10s):     0.118    1.52    +0.03       ✓ OK
Adverse Selection:         0.198    0.92    -0.15       ⚠ WARNING
Direction (1s):            0.241    1.08    +0.02       ✓ OK
Regime Detection:          0.156    1.28    -0.02       ✓ OK

Calibration Curves:
Fill (1s):  0.1 ▪▪▪▪▪  0.3 ▪▪▪▪▪  0.5 ▪▪▪▪▪  0.7 ▪▪▪▪▪  0.9 ▪▪▪▪▪
Predicted:  ─────────────────────────────────────────────────────
Realized:   ╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶╶
            (Good calibration - lines overlap)
```

### 4. Signal Health

```
Signal Mutual Information (vs 7d ago)
═══════════════════════════════════════════════════════════════════
Signal                      MI Today   MI 7d Ago   Δ        Status
───────────────────────────────────────────────────────────────────
binance_return_100ms        0.087      0.092       -5.4%    ✓ OK
trade_imbalance_1s          0.064      0.068       -5.9%    ✓ OK
microprice_imbalance        0.048      0.044       +9.1%    ✓ OK
funding_x_imbalance         0.039      0.051       -23.5%   ⚠ DECAY
open_interest_change_1m     0.022      0.024       -8.3%    ✓ OK
lead_lag_r_squared          0.31       0.34        -8.8%    ✓ OK

⚠ funding_x_imbalance MI dropped 23.5% - investigate interaction
```

### 5. Regime Distribution

```
Regime Distribution
═══════════════════════════════════════════════════════════════════
            Time %    PnL         PnL/hr     Fills    AS Rate
───────────────────────────────────────────────────────────────────
Quiet:      62.4%     $285.33     $19.05     412      18.2%
Trending:   24.1%     $98.44      $17.02     289      32.4%
Volatile:   12.3%     $9.05       $3.06      134      41.2%
Cascade:    1.2%      -$45.00     -$156.25   12       75.0%

⚠ Cascade PnL negative - review liquidation detector thresholds
```

### 6. Conditional Calibration Issues

```
Conditional Calibration Analysis
═══════════════════════════════════════════════════════════════════

By Volatility Regime:
  • Low vol:    Fill IR = 1.42 ✓
  • Med vol:    Fill IR = 1.31 ✓
  • High vol:   Fill IR = 0.88 ⚠ (model overconfident in high vol)

By Funding State:
  • High +ve:   AS IR = 0.76 ⚠ (classifier fails in squeeze)
  • Normal:     AS IR = 1.15 ✓
  • High -ve:   AS IR = 0.82 ⚠ (classifier fails in squeeze)

By Time of Day:
  • 00:00-08:00: Fill IR = 1.38 ✓
  • 08:00-16:00: Fill IR = 1.29 ✓
  • 16:00-24:00: Fill IR = 1.41 ✓
```

### 7. Actionable Items

```
═══════════════════════════════════════════════════════════════════
ACTIONABLE ITEMS (Prioritized)
═══════════════════════════════════════════════════════════════════

[HIGH] Adverse Selection Model IR = 0.92 < 1.0
  → Model adding noise in some conditions
  → Check conditional calibration by funding state
  → Consider retraining or reducing model weight
  
[MEDIUM] Cascade Regime Loss: -$45
  → Review liquidation detector thresholds
  → Current threshold: OI drop > 2%
  → Consider tightening to 1.5% or widening spreads earlier
  
[MEDIUM] Signal Decay: funding_x_imbalance MI -23.5%
  → This interaction signal is losing predictive power
  → Run signal audit to investigate cause
  → May need to remove from feature set
  
[LOW] Inventory Cost +25.7% vs average
  → Holding positions longer than usual
  → Check inventory skew parameters
  → May indicate trending market with asymmetric fills
```

---

## Implementation

### Report Generator

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

### Alerting Integration

```rust
struct AlertManager {
    slack_webhook: Option<String>,
    email_recipients: Vec<String>,
    pagerduty_key: Option<String>,
}

impl AlertManager {
    fn send_report_alerts(&self, report: &DailyReport) {
        match report.status {
            ReportStatus::Critical => {
                // PagerDuty + Slack + Email
                self.send_pagerduty(&report);
                self.send_slack(&report, SlackPriority::Urgent);
                self.send_email(&report);
            }
            ReportStatus::Warning => {
                // Slack + Email
                self.send_slack(&report, SlackPriority::Warning);
                self.send_email(&report);
            }
            ReportStatus::Caution => {
                // Slack only
                self.send_slack(&report, SlackPriority::Info);
            }
            ReportStatus::Ok => {
                // Daily summary to Slack
                self.send_slack(&report, SlackPriority::Info);
            }
        }
    }
    
    fn send_slack(&self, report: &DailyReport, priority: SlackPriority) {
        let Some(webhook) = &self.slack_webhook else { return };
        
        let emoji = match report.status {
            ReportStatus::Critical => "🚨",
            ReportStatus::Warning => "⚠️",
            ReportStatus::Caution => "📊",
            ReportStatus::Ok => "✅",
        };
        
        let message = format!(
            "{} *Daily Report {}*\nPnL: ${:.2} | Sharpe: {:.1} | Status: {:?}\n{}",
            emoji,
            report.date,
            report.pnl.gross_pnl,
            report.sharpe,
            report.status,
            report.action_items.iter()
                .take(3)
                .map(|a| format!("• [{}] {}", a.priority, a.message))
                .collect::<Vec<_>>()
                .join("\n")
        );
        
        // Send to Slack webhook
        send_slack_message(webhook, &message, priority);
    }
}
```

### Automation

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

---

## Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Model IR | < 1.0 | < 0.8 | Retrain or disable |
| Brier Score | > 0.20 | > 0.25 | Investigate calibration |
| Daily Loss | > $200 | > $500 | Reduce size, investigate |
| Cascade Loss | > $50 | > $100 | Tighten liquidation detection |
| Signal MI Decay | > 15%/week | > 30%/week | Remove signal |
| Lead-Lag R² | < 0.20 | < 0.15 | Reduce reliance |

---

## Dashboard Integration

Export metrics to Grafana/Prometheus:

```rust
fn export_metrics(report: &DailyReport) {
    // PnL metrics
    gauge!("mm.pnl.gross", report.pnl.gross_pnl);
    gauge!("mm.pnl.spread_capture", report.pnl.spread_capture);
    gauge!("mm.pnl.adverse_selection", report.pnl.adverse_selection);
    
    // Model metrics
    for (model, metrics) in &report.calibration.by_model {
        gauge!("mm.model.ir", metrics.information_ratio, "model" => model);
        gauge!("mm.model.brier", metrics.brier_score, "model" => model);
    }
    
    // Regime metrics
    for (regime, pnl) in &report.regime_distribution.pnl_by_regime {
        gauge!("mm.regime.pnl", *pnl, "regime" => format!("{:?}", regime));
    }
    
    // Signal metrics
    for signal in &report.signal_health.signals {
        gauge!("mm.signal.mi", signal.mi_today, "signal" => &signal.name);
    }
}
```

---

## Dependencies

- **Requires**: measurement-infrastructure, calibration-analysis, all model components
- **Enables**: Automated monitoring, early warning, performance tracking

## Common Mistakes

1. **Not running daily**: Problems compound if undetected
2. **Ignoring warnings**: "It's probably fine" leads to losses
3. **Too many alerts**: Alert fatigue defeats the purpose
4. **No historical comparison**: Need 7d context for anomaly detection
5. **Missing conditional analysis**: Aggregate metrics hide regime-specific issues

## Operational Checklist

- [ ] Report job scheduled (cron, systemd timer, etc.)
- [ ] Slack webhook configured
- [ ] Email alerts configured
- [ ] PagerDuty integration for critical alerts
- [ ] Dashboard panels created
- [ ] Historical reports archived
- [ ] Weekly review meeting scheduled
