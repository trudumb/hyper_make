---
name: Calibration Analysis
description: Systematic analysis of model predictions vs realized outcomes to identify model weaknesses
user-invocable: false
---

# Calibration Analysis Skill

## Purpose

Systematic analysis of model predictions vs realized outcomes. This tells you exactly where your models are wrong and by how much.

## When to Use

- Daily model health checks
- Debugging unexpected PnL patterns
- Deciding which model component to improve next
- Validating changes before/after deployment
- Any time you want to know "is this model actually working?"

## Prerequisites

- `measurement-infrastructure` must be implemented and logging data
- At least 24 hours of prediction data (preferably 1 week+)

---

## Key Metrics

### 1. Brier Score

The mean squared error of probability predictions:

```
BS = (1/N) Σ (pᵢ - oᵢ)²
```

Where:
- pᵢ = predicted probability
- oᵢ ∈ {0, 1} = actual outcome

**Interpretation:**
- BS = 0: Perfect predictions
- BS = 0.25: Predicting 50% for everything (random)
- BS > 0.25: Worse than random

### 2. Brier Score Decomposition

Split Brier score into interpretable components:

```
BS = Reliability - Resolution + Uncertainty

Reliability = (1/N) Σₖ nₖ(p̄ₖ - ōₖ)²
  - Measures calibration quality
  - Lower is better
  - If your 70% predictions hit 70%, this is 0

Resolution = (1/N) Σₖ nₖ(ōₖ - ō)²
  - Measures discrimination ability
  - Higher is better
  - Do your high predictions differ from low predictions?

Uncertainty = ō(1 - ō)
  - Base rate variance
  - Not controllable, just inherent difficulty
```

### 3. Information Ratio

```
IR = Resolution / Uncertainty
```

**Interpretation:**
- IR > 1.0: Model predictions carry useful information
- IR ≈ 1.0: Model is roughly as good as predicting base rate
- IR < 1.0: Model is adding noise (REMOVE IT)

---

## Implementation

### Brier Score Decomposition

`compute_brier_decomposition(predictions, outcomes, num_bins)` bins predictions and computes reliability, resolution, uncertainty, and IR. See [implementation.md](./implementation.md) for full code.

### Calibration Curve

`build_calibration_curve(predictions, outcomes, num_bins)` sorts predictions into equal-sized bins, computes realized rates, and adds Wilson score 95% CIs. See [implementation.md](./implementation.md) for full code.

### Conditional Calibration

Slice calibration by conditioning variables (`VolatilityQuartile`, `FundingRegime`, `TimeOfDay`, `InventoryState`, `RecentFillRate`, `BookImbalance`, `Regime`). Groups records by condition, requires min 100 samples per slice, computes Brier decomposition per slice. Regime slicing uses cascade/volatile/trending/quiet thresholds at 0.5 probability. See [implementation.md](./implementation.md) for full code.

---

## PnL Attribution

Decompose daily PnL into four components: **spread capture** (revenue from bid-ask), **adverse selection** (loss from fills before adverse moves), **inventory cost** (mark-to-market), and **fees** (1.5 bps maker). Also breaks down PnL by regime (quiet/trending/volatile/cascade) with time fractions. See [implementation.md](./implementation.md) for full `PnLAttribution` struct and `compute_pnl_attribution()` code.

---

## Daily Report Template

```
=== Calibration Report: {date} ===

PnL Attribution
───────────────────────────────────────────
Gross PnL:              ${gross_pnl:>10.2}
├── Spread Capture:     ${spread_capture:>10.2}  {spread_status}
├── Adverse Selection:  ${adverse_selection:>10.2}  {as_status}
├── Inventory Cost:     ${inventory_cost:>10.2}
└── Fees:               ${fees:>10.2}

Model Calibration
───────────────────────────────────────────
                        Brier   IR      Status
Fill Prediction (1s):   {fp_brier:.3}  {fp_ir:.2}   {fp_status}
Fill Prediction (10s):  {fp10_brier:.3} {fp10_ir:.2}  {fp10_status}
Adverse Selection:      {as_brier:.3}  {as_ir:.2}   {as_status_model}
Volatility (RMSE):      {vol_rmse:.6}          {vol_status}

Regime Distribution
───────────────────────────────────────────
            Time    PnL         PnL/Hour
Quiet:      {quiet_time:>4.0%}   ${quiet_pnl:>8.2}   ${quiet_rate:>6.2}/hr
Trending:   {trend_time:>4.0%}   ${trend_pnl:>8.2}   ${trend_rate:>6.2}/hr
Volatile:   {vol_time:>4.0%}   ${vol_pnl:>8.2}   ${vol_rate:>6.2}/hr
Cascade:    {casc_time:>4.0%}   ${casc_pnl:>8.2}   ${casc_rate:>6.2}/hr

Conditional Calibration Issues
───────────────────────────────────────────
{conditional_issues}

Actionable Items
───────────────────────────────────────────
{action_items}
```

### Report Generation

`generate_daily_report(date)` loads prediction records, computes PnL attribution, runs Brier decomposition on fill (1s/10s) and adverse selection models, checks conditional calibration across regime/volatility/funding slices, and generates prioritized action items. Action item priorities: HIGH for AS > 50% of spread capture or IR < 1.0; MEDIUM for cascade losses or conditional issues. See [implementation.md](./implementation.md) for full code.

---

## Alert Thresholds

Default alert thresholds:

| Threshold | Default | Meaning |
|-----------|---------|---------|
| `min_information_ratio` | 1.0 | Below this, model is useless |
| `max_brier_score` | 0.25 | Above this, worse than random |
| `max_daily_loss` | $500 | Dollar amount |
| `max_adverse_selection_ratio` | 0.7 | AS / spread_capture |
| `max_cascade_loss` | $100 | Dollar amount in cascade regime |

`check_alerts(report, thresholds)` emits `Alert::Critical` for IR below threshold or daily loss exceeding limit. See [implementation.md](./implementation.md) for full `AlertThresholds` struct and `check_alerts()` code.

---

## Common Patterns

### "Model is well-calibrated overall but fails in regime X"

This is the most common pattern. Solution:
1. Identify which regime has IR < 1.0
2. Either: train regime-specific model, or
3. Fall back to wider spreads / simpler model in that regime

### "Calibration looks good but still losing money"

Possible causes:
1. Good calibration on the wrong metric (e.g., calibrated fill prediction but adverse selection is the real problem)
2. Execution slippage not captured in calibration
3. Latency effects (predictions stale by the time orders placed)

### "IR > 1 but Brier score is high"

Model has good discrimination (can tell high from low) but poor calibration (predictions don't match frequencies). Fix with isotonic regression or Platt scaling.

---

## Dependencies

- **Requires**: measurement-infrastructure (prediction logs with outcomes)
- **Enables**: All model improvement work, daily-calibration-report

## Next Steps

After analyzing calibration:
1. Identify weakest component (lowest IR or biggest PnL drag)
2. Read that component's skill file
3. Use signal-audit to identify better features
4. Implement improvement
5. Re-run calibration to validate

## Supporting Files

- [implementation.md](./implementation.md) -- All Rust code: Brier decomposition, calibration curve, conditional calibration, PnL attribution, report generation, action items, alert thresholds
