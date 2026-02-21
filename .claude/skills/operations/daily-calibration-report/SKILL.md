---
name: daily-calibration-report
description: Automated daily health check of all model components for early warning of degradation and drift. Use when setting up daily reports, debugging performance issues, or reviewing model health after deployment. Covers PnL attribution, calibration metrics, signal decay, regime analysis, and alerting.
requires:
  - calibration-analysis
disable-model-invocation: true
context: fork
agent: analytics
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
║  Overall Status: WARNING                                           ║
║                                                                    ║
║  PnL:           $347.82  (+12% vs 7d avg)                         ║
║  Sharpe (1d):   2.1                                               ║
║  Fill Rate:     847 fills                                         ║
║  Uptime:        99.7%                                             ║
║                                                                    ║
║  Issues:                                                           ║
║  - Adverse selection model IR dropped to 0.92 (threshold: 1.0)    ║
║  - Cascade regime PnL: -$45 (investigate)                         ║
╚════════════════════════════════════════════════════════════════════╝
```

### 2. PnL Attribution

Breaks down daily PnL into: spread capture, adverse selection, inventory cost, fees. Compared against 7-day rolling average. Hourly PnL histogram for time-of-day analysis.

### 3. Model Calibration

Per-model metrics: Brier Score, Information Ratio, 7-day IR trend. Models tracked: Fill Prediction (1s, 10s), Adverse Selection, Direction (1s), Regime Detection. Includes calibration curve visualization (predicted vs realized).

### 4. Signal Health

Mutual information per signal vs 7 days ago. Flags signals with >15% MI decay for investigation. Tracks lead-lag R-squared for cross-exchange signals.

### 5. Regime Distribution

Time fraction, PnL, PnL/hour, fill count, and AS rate per regime (Quiet/Trending/Volatile/Cascade). Cascade losses are highlighted for investigation.

### 6. Conditional Calibration Issues

Sliced calibration by: volatility regime, funding state, time of day. Flags conditions where model IR drops below 1.0 — these are where the model adds noise.

### 7. Actionable Items

Prioritized list generated from all sections:

- **HIGH**: Model IR < 1.0, daily loss exceeding limit, AS > 50% of spread capture
- **MEDIUM**: Cascade losses, signal decay > 15%, conditional calibration issues
- **LOW**: Elevated inventory cost, minor parameter drift

See [references/implementation.md](./references/implementation.md) for the `DailyReportGenerator` struct, `generate()` pipeline, action item generation, and status determination logic.

---

## Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Model IR | < 1.0 | < 0.8 | Retrain or disable |
| Brier Score | > 0.20 | > 0.25 | Investigate calibration |
| Daily Loss | > $200 | > $500 | Reduce size, investigate |
| Cascade Loss | > $50 | > $100 | Tighten liquidation detection |
| Signal MI Decay | > 15%/week | > 30%/week | Remove signal |
| Lead-Lag R-squared | < 0.20 | < 0.15 | Reduce reliance |

See [references/alerting.md](./references/alerting.md) for the `AlertManager` implementation and alert routing (Slack/Email/PagerDuty).

---

## Dashboard Integration

Export metrics to Grafana/Prometheus for real-time monitoring. Key panels: PnL Waterfall, Model IR Trend, Signal MI Heatmap, Regime Timeline, Calibration Curve.

See [references/dashboard.md](./references/dashboard.md) for `export_metrics()` code and panel definitions.

---

## Operational Checklist

- [ ] Report job scheduled (cron, systemd timer, etc.)
- [ ] Slack webhook configured
- [ ] Email alerts configured
- [ ] PagerDuty integration for critical alerts
- [ ] Dashboard panels created
- [ ] Historical reports archived
- [ ] Weekly review meeting scheduled

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

## Supporting Files

- [references/implementation.md](./references/implementation.md) — DailyReportGenerator, generate() pipeline, action items, status logic
- [references/alerting.md](./references/alerting.md) — AlertManager, Slack/Email/PagerDuty routing
- [references/dashboard.md](./references/dashboard.md) — Grafana export, metric panels
