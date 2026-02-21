---
name: learning-pipeline
description: Documents the 9 learning feedback loops, SpreadBandit Thompson Sampling, adaptive ensemble, confidence tracking, and baseline tracker. Use when debugging learning behavior, tuning reward attribution, investigating model weight decay, or understanding how fills translate into parameter updates.
requires:
  - measurement-infrastructure
user-invocable: false
---

# Learning Pipeline Skill

## Architecture Overview

Every fill is labeled training data. The system has 5 core learning components that all receive 1:1 reward attribution from each fill event:

```
FILL EVENT
  |
  +-> SpreadBandit.update_from_pending(reward)     # Spread selection learning
  +-> AdaptiveEnsemble.update_performance(ir,brier) # Model weight learning
  +-> KappaEstimator.on_own_fill(ts, price)        # Fill intensity learning
  +-> ReconcileOutcomeTracker.record_fill(oid,edge) # Reconcile action learning
  +-> PreFillClassifier.record_outcome(adverse,mag) # Adverse selection learning
```

## The 9 Feedback Loops

| # | Loop | Trigger | Updates | File |
|---|------|---------|---------|------|
| 1 | Kappa from own fills | `on_own_fill()` | Fill intensity (Hawkes) | `estimator/mod.rs` |
| 2 | AS markout queue | Fill → pending outcomes | Pending fill list | `orchestrator/handlers.rs` |
| 3 | AS outcome feedback | Markout resolved | `record_outcome()` on classifier | `orchestrator/handlers.rs` |
| 4 | Calibration progress | Periodic | `calibration_controller` update | `orchestrator/handlers.rs` |
| 5 | Sigma update | Trade/L2 events | Realized volatility | `orchestrator/handlers.rs` |
| 6 | Regime update | Trade/L2 events | HMM belief state | `orchestrator/handlers.rs` |
| 7 | Quote outcome tracking | Fill + 30s expiry | Fill rate bins, edge estimation | `learning/quote_outcome.rs` |
| 8 | Spread bandit update | Fill event | Context-arm posterior | `learning/spread_bandit.rs` |
| 9 | Ensemble weight update | Fill event | IR-based model weights | `learning/adaptive_ensemble.rs` |

See `references/feedback-loops.md` for detailed loop descriptions.

## Core Components

### SpreadBandit (Thompson Sampling)

81 contexts (3 regimes x 3 positions x 3 vols x 3 flows) x 8 arms (multipliers: 0.85-1.40).

- **Posterior**: Normal-Gamma conjugate per (context, arm) cell
- **Selection**: Sample from posterior, pick highest sampled reward
- **Forgetting**: `factor=0.995`, half-life ~138 obs, only when `n >= 10`
- **Cold start**: Arm 3 (mult 1.0 = pure GLFT) when `max_obs < 3`
- **Reward**: `baseline_adjusted_edge_bps` (actual edge minus EWMA baseline)

Key methods: `select_arm(context)`, `update_from_pending(reward)`, `best_arm(context)`

### QuoteOutcomeTracker (Unbiased Edge)

Solves survivorship bias by tracking ALL quotes (filled AND unfilled).

- **Bins**: 8 fine + 4 coarse with hierarchical shrinkage
- **Fill rate**: Beta posterior per bin: `P(fill) = alpha / (alpha + beta)`
- **Expected edge**: `E[edge] = P(fill) x E[edge|fill]`
- **Optimal spread**: `argmax(expected_edge)` via grid search
- **E[PnL] reconciliation**: Tracks prediction accuracy via `epnl_at_registration`

Key methods: `register_quote()`, `on_fill()`, `expire_old_quotes()`, `optimal_spread_bps()`

### AdaptiveEnsemble (Dynamic Model Weights)

Softmax over Information Ratio with water-filling floor.

- **Weight formula**: `w[i] = exp(IR[i] / T) / sum(exp(IR[j] / T))`
- **Temperature**: 0.5 (concentrated) to 2.0 (uniform)
- **Floor**: `min_weight` via iterative water-filling
- **Decay**: EWMA blend `ir_new = 0.995 * ir_old + 0.005 * ir_measured`
- **Cold start**: `min_predictions_for_weight = 20`

Key methods: `update_performance()`, `compute_weights()`, `weighted_average()`, `summary()`

### BaselineTracker (Counterfactual Reward)

EWMA baseline subtraction centers rewards around zero for RL/bandit.

- **Formula**: `ewma = 0.99 * ewma + 0.01 * reward`
- **Output**: `counterfactual = actual - baseline` (or `actual` if not warmed up)
- **Warmup**: `min_observations = 10`

### EdgeBiasTracker (Calibration Health)

Detects systematic edge miscalibration.

- **Input**: `(predicted_edge_bps, realized_edge_bps)`
- **Alert**: `should_recalibrate()` when `|ewma_bias| > 1.5 bps`

## Data Flow

```
QUOTE CYCLE:
  SpreadBandit.select_arm(context) → pending selection
  QuoteOutcomeTracker.register_quote() → pending quote
  Quote published with spread_bps

FILL EVENT (handlers.rs):
  QuoteOutcomeTracker.on_fill() → resolve pending, update fill rate bins
  SpreadBandit.update_from_pending(reward) → update cell posterior
  AdaptiveEnsemble.update_performance() → update IR, recompute weights
  KappaEstimator.on_own_fill() → update Hawkes intensity
  ReconcileOutcomeTracker.record_fill() → update action EV estimates

EXPIRY (30s timeout):
  QuoteOutcomeTracker.expire_old_quotes() → mark as Expired, update bins
```

## Checkpoint Persistence

All components persist via `#[serde(default)]`:
- `SpreadBanditCheckpoint`: cells with (context_idx, arm_idx, mu_n, kappa_n, alpha, beta, n)
- `QuoteOutcomeCheckpoint`: bins with (lo_bps, hi_bps, observed_fills, observed_total)
- `BaselineTracker`: (ewma_reward, n_observations)
- `AdaptiveEnsemble`: HashMap of ModelPerformance (IR, Brier, n_predictions, weight)

## Key File Map

| Component | File |
|-----------|------|
| SpreadBandit | `learning/spread_bandit.rs` |
| QuoteOutcomeTracker | `learning/quote_outcome.rs` |
| BaselineTracker | `learning/baseline_tracker.rs` |
| AdaptiveEnsemble | `learning/adaptive_ensemble.rs` |
| EdgeBiasTracker | `learning/confidence.rs` |
| DecisionEngine | `learning/decision.rs` |
| CompetitorModel | `learning/competitor_model.rs` |
| Fill integration | `orchestrator/handlers.rs` |
