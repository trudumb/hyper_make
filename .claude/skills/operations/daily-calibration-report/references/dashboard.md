# Daily Calibration Report — Dashboard Reference

## Grafana/Prometheus Export

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

## Key Dashboard Panels

| Panel | Metric | Alert |
|-------|--------|-------|
| PnL Waterfall | gross, spread, AS, inventory, fees | Daily loss > $500 |
| Model IR Trend | IR per model over 30d | IR < 1.0 for any model |
| Signal MI Heatmap | MI per signal per day | Decay > 15%/week |
| Regime Timeline | Regime probability over time | Cascade duration > 5 min |
| Calibration Curve | Predicted vs realized by bin | Deviation > 10% in any bin |
