# Daily Calibration Report — Alerting Reference

## AlertManager

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

## Alert Routing

| Report Status | PagerDuty | Slack | Email |
|--------------|-----------|-------|-------|
| Critical | Yes (page) | Urgent | Yes |
| Warning | No | Warning | Yes |
| Caution | No | Info | No |
| Ok | No | Summary | No |
