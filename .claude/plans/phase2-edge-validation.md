# Phase 2: Edge Validation — Data Collection & Statistical Proof

## Prerequisites
Phase 1 complete: calibration flowing, spreads < 8 bps, signals wired, paper trader stable for 1800s+.

## Mission
Run paper trader 24/7 for 7+ days. Accumulate 5,000+ fills. Prove with statistical rigor that the system has positive edge net of fees.

---

## Week 1: Data Collection

### Setup
```bash
# Run paper trader in screen/tmux with checkpoint persistence
cargo run --release --bin paper_trader -- \
    --paper-mode \
    --duration 604800 \
    --report \
    --checkpoint-interval 300 \
    2>&1 | tee logs/paper_trading/week1_$(date +%Y%m%d).log
```

### Daily Monitoring Checklist
| Check | Command | Target |
|-------|---------|--------|
| System alive | `grep "Quote Cycles" logs/paper_trading/*.log \| tail -1` | Increasing |
| Fill count | `grep "total_fills" logs/paper_trading/*.log \| tail -1` | > 500/day |
| Spread tracking | `grep "optimal_spread_bps" logs/paper_trading/*.log \| tail -1` | < 5 bps |
| Warmup | `grep "warmup_pct" logs/paper_trading/*.log \| tail -1` | Increasing daily |
| Kill switches | `grep "KILL" logs/paper_trading/*.log` | Zero |
| Calibration | `cat logs/paper_trading/calibration_report.json` | Brier < 0.25 |

### Data Files Generated
- `logs/paper_trading/predictions.jsonl` — every quote cycle's predictions + outcomes
- `logs/paper_trading/calibration_report.json` — Brier scores, IR, conditional calibration
- `data/checkpoints/paper/BTC/latest/checkpoint.json` — learned parameters
- Per-fill attribution logs

---

## Week 1-2: Statistical Analysis

### Analysis 1: Sharpe Ratio
```
Sharpe = mean(daily_pnl) / std(daily_pnl) × sqrt(365)
```

**Requirements**:
- At least 7 daily observations
- `Sharpe > 2.0` for go-live
- `Sharpe > 3.0` for confident go-live
- `Sharpe < 1.0` → do not proceed, return to Phase 1

### Analysis 2: Fill Quality Distribution
For each fill, compute `fill_quality = spread_capture_bps - adverse_selection_bps - fee_bps`:
- **Mean fill quality > 1.0 bps**: System captures edge
- **Mean fill quality < 0**: System is being adversely selected
- **Distribution**: Should be right-skewed (many small wins, few large losses)

### Analysis 3: Per-Signal PnL Attribution
For each signal, compute incremental PnL contribution:
| Signal | Metric | Go-Live Threshold |
|--------|--------|-------------------|
| Lead-lag | PnL when signal active vs inactive | > $0 contribution |
| Regime HMM | Drawdown reduction in cascade periods | > 30% reduction |
| Inventory skew | Position mean-reversion speed | < 5 min to half-life |
| AS classifier | Brier score on fill toxicity | < 0.20 |
| Informed flow | Spread widening prevents AS | Conditional win rate > 55% |

### Analysis 4: Conditional Calibration
From calibration_report.json:
- **Fill probability Brier**: How well do we predict fills?
- **AS Brier**: How well do we predict adverse selection?
- **Conditional by regime**: Does accuracy vary by market conditions?
- **Information Ratio**: Signal value above baseline

### Analysis 5: Drawdown Analysis
- **Max drawdown**: Should be < 3% of simulated capital
- **Max drawdown duration**: Time from peak to recovery
- **Drawdown during cascades**: Should be smaller than without regime detection
- **Recovery pattern**: Consistent mean-reversion, not lucky rebounds

---

## Go/No-Go Decision

### GREEN (Proceed to Phase 3)
| Metric | Threshold |
|--------|-----------|
| Paper Sharpe (7d) | > 2.0 |
| Fill quality mean | > 1.0 bps |
| Brier score (fill) | < 0.20 |
| AS prediction IR | > 0.5 |
| Max drawdown | < 3% |
| Win rate | > 52% |
| Daily fill count | > 300 |
| Two-sided ratio | > 40/60 |

### YELLOW (Iterate, Re-run Phase 1)
- Sharpe 1.0-2.0: Edge exists but weak. Tighten spreads or improve signals.
- Brier 0.20-0.30: Models learning but not calibrated. Need more data or model fixes.
- Max drawdown 3-5%: Risk management needs tuning.

### RED (Do Not Proceed)
- Sharpe < 1.0: No proven edge. Return to fundamentals.
- Fill quality < 0: Being adversely selected. Widen spreads or improve AS model.
- Max drawdown > 5%: Risk management broken. Fix before any live capital.
- Fill count < 100/day: Not competitive. Spreads too wide or fill sim unrealistic.
