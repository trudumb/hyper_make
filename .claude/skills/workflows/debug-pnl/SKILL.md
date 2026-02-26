---
name: debug-pnl
description: Step-by-step workflow for diagnosing why the market maker is losing money
disable-model-invocation: true
context: fork
agent: general-purpose
argument-hint: "[asset] [timerange]"
---

# Debug PnL Workflow

Structured diagnostic for market maker PnL issues. Follow these steps in order.

## Step 1: Identify the Drag

Check PnL attribution to find which component is causing losses. Read the relevant analytics:

```
src/market_maker/analytics/attribution.rs   — PnL decomposition
src/market_maker/analytics/edge_metrics.rs  — Edge tracking
src/market_maker/analytics/sharpe.rs        — Sharpe ratio
```

PnL decomposes into:
- **Spread capture** (positive) — edge from bid-ask spread
- **Adverse selection** (negative) — fills on wrong side of price move
- **Inventory cost** (negative) — holding risk
- **Fees** (negative) — exchange maker fees (1.5 bps on Hyperliquid)

The largest negative component is your priority target.

## Step 2: Route to Component

| Dominant Drag | Check These | Read Skill |
|---------------|-------------|------------|
| Adverse selection | `adverse_selection/`, `calibration/` | `adverse-selection-classifier` |
| Spread too tight | `strategy/signal_integration.rs` | `quote-engine` |
| Inventory cost | `strategy/position_manager.rs`, `control/` | `stochastic-controller` |
| Fill rate collapsed | `estimator/kappa.rs`, `simulation/fill_sim.rs` | `fill-intensity-hawkes` |
| Regime misclassification | `estimator/regime_hmm.rs` | `regime-detection-hmm` |
| Cascade losses | `risk/circuit_breaker.rs`, `risk/monitors/cascade.rs` | `risk-management` |

## Step 3: Check Calibration

For the identified component, check its calibration metrics:

```
src/market_maker/calibration/brier_score.rs       — Brier Score
src/market_maker/calibration/information_ratio.rs  — Information Ratio
src/market_maker/calibration/conditional_metrics.rs — Conditional calibration
src/market_maker/calibration/model_gating.rs       — Model weight/gating
```

Key thresholds:
- **IR < 1.0**: model adding noise, not signal
- **IR < 0.8**: model actively harmful, consider disabling
- **Brier > 0.25**: severe miscalibration

## Step 4: Conditional Analysis

Check if the issue is regime-specific:

1. Does the model fail only in high-vol? Check `conditional_metrics.rs` by volatility bucket
2. Does it fail near funding settlement? Check by `time_to_funding_settlement_s`
3. Does it fail at specific times of day? Check by hour
4. Does it fail at specific position sizes? Check by inventory level

Regime-specific failures suggest the model needs regime-dependent parameters.

## Step 5: Validate Fix

After identifying and fixing the issue:

1. Check calibration metrics before/after
2. Run the paper trader to validate in simulation
3. Monitor live metrics for at least 1 hour after deployment
4. Watch for regression in other components — fixes often shift PnL between components

## Common Patterns

### "Making money in quiet, losing in volatile"
- Gamma too low for high-vol regime
- Cascade detection too slow
- Spread floor not wide enough

### "Fills are profitable but too few"
- Kappa estimate too low (spreads too wide)
- Check `fill_rate_model.rs` for stale estimates
- Calibration gamma too conservative

### "Fills are plentiful but adverse"
- Pre-fill toxicity classifier miscalibrated
- Check `pre_fill_classifier.rs` z-score computation
- Lead-lag signal may have decayed

### "Position builds up, can't unwind"
- Inventory skew not aggressive enough
- Position guard soft threshold too high
- Check for asymmetric fill rates (buying but not selling)
