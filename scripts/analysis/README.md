# Analysis Scripts

Scripts for analyzing paper trading sessions and generating calibration reports.

## Scripts

### analyze_session.sh

Quick shell-based analysis of a paper trading session.

```bash
./scripts/analysis/analyze_session.sh logs/paper_trading_BTC_2026-01-22_15-30-00
```

**Output:**
- Basic statistics (log lines, errors, warnings)
- Session timing
- Fill analysis
- Parameter estimation samples
- Risk events
- Health summary

### calibration_report.py

Python-based calibration analysis following Small Fish methodology.

```bash
python3 scripts/analysis/calibration_report.py logs/paper_trading_BTC_2026-01-22_15-30-00
```

**Output Files:**
- `calibration_report.json` - Machine-readable metrics
- `calibration_report.md` - Human-readable Markdown report

**Metrics Computed:**
- Brier score decomposition (reliability, resolution, uncertainty)
- Information Ratio (IR > 1.0 means model adds value)
- Calibration curve
- PnL attribution
- Small Fish validation checklist

## Usage Workflow

1. **Run paper trading session:**
   ```bash
   ./scripts/paper_trading.sh BTC 3600 --report
   ```

2. **Quick analysis:**
   ```bash
   ./scripts/analysis/analyze_session.sh logs/paper_trading_BTC_*/
   ```

3. **Full calibration report:**
   ```bash
   python3 scripts/analysis/calibration_report.py logs/paper_trading_BTC_*/
   ```

4. **Review report:**
   ```bash
   cat logs/paper_trading_BTC_*/calibration_report.md
   ```

## Key Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| Samples | ≥200 | Statistical significance |
| Information Ratio | >1.0 | Model adds value |
| Calibration Error | <0.1 | Predictions match outcomes |
| Brier Score | <0.25 | Better than random |

## Requirements

- `analyze_session.sh`: bash, standard Unix tools
- `calibration_report.py`: Python 3.6+ (no external dependencies)
