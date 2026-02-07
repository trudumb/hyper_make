---
name: Adverse Selection Classifier
description: Predict which incoming trades are informed before they hit you, enabling dynamic spread/kappa adjustment
---

# Adverse Selection Classifier Skill

## Purpose

Predict which incoming trades are informed before they hit you, enabling dynamic spread/kappa adjustment to protect against toxic flow. This is often the highest-value model for PnL improvement because adverse selection is typically the largest drag on market maker profits.

## When to Use

- Building adverse selection prediction from scratch
- Debugging why you're losing money on fills
- Adding liquidation cascade detection
- Tuning when to widen spreads vs stay tight
- Any time adverse selection is your biggest PnL drag

## Prerequisites

- `measurement-infrastructure` with post-fill price tracking
- `signal-audit` to identify predictive features
- Historical trades with ex-post price evolution

---

## Core Concept: Trade Classification

Every trade comes from a latent source:

1. **Noise traders**: No information, mean-reverting flow
2. **Informed traders**: Have directional information, toxic to market makers
3. **Liquidations**: Forced selling/buying, extremely toxic but predictable
4. **Arbitrageurs**: Cross-exchange spread closure, predictable timing

You can't observe the true type, but you can:
- Infer it from ex-post price movement (for training)
- Predict it from observable features (for real-time use)

---

## Labeling Strategy

Since true intent is unobservable, label based on ex-post outcomes:

Labels: `Informed`, `Noise`, `Liquidation`, `Arbitrage`. Priority: check liquidation context first (OI drop >2% + extreme funding), then arbitrage (cross-exchange spread >5bps closed within 500ms), then informed vs noise based on ex-post 10s price move (>5bps in trade direction = informed).

See [implementation.md](./implementation.md) for full labeling code.

### Label Distribution (Typical)

| Label | Frequency | Toxicity |
|-------|-----------|----------|
| Noise | 60-70% | Low (slightly positive expectation) |
| Informed | 20-30% | Medium (negative expectation) |
| Arbitrage | 5-10% | Medium-High (predictable timing) |
| Liquidation | 1-5% | Extreme (can be catastrophic) |

---

## Feature Engineering

Six feature groups, 26 features total. Each group has a struct + compute function:

| Group | Fields | Key Signals |
|-------|--------|-------------|
| **Size** (5) | `size_zscore`, `size_quantile`, `size_vs_depth`, `size_vs_typical`, `size_raw` | Large relative to depth = likely informed |
| **Timing** (4) | `time_since_last_trade_ms`, `trades_in_last_1s`, `trade_rate_zscore`, `is_burst` | Bursts (>10/sec) cluster with toxic flow |
| **Aggression** (4) | `is_aggressor`, `crossed_spread_bps`, `depth_consumed_pct`, `sweeping_multiple_levels` | Sweeping multiple levels = strongly informed |
| **Flow** (4) | `signed_volume_imbalance_1s/10s`, `flow_autocorrelation`, `trade_aligns_with_flow` | Flow autocorrelation detects persistent direction |
| **Hyperliquid** (5) | `funding_rate`, `oi_change_1m/5m`, `trade_against_funding`, `near_liquidation_level`, `time_to_settlement_s` | Trade against funding often = informed |
| **Cross-Exchange** (4) | `binance_hl_spread_bps`, `binance_price_change_100ms/500ms`, `binance_leading` | Binance leading >2bps = arb flow incoming |

`TradeFeatures` combines all groups and provides `to_vector()` -> `Vec<f64>` with appropriate scaling (log for time, bps for funding, cyclical sin for settlement time).

See [implementation.md](./implementation.md) for all struct definitions and compute functions.

---

## Classifier Architecture

### Option 1: Logistic Regression (Interpretable)

`P(informed) = sigmoid(w . x + b)` -- simple dot product of weights with feature vector. Good starting point for interpretability.

### Option 2: Small MLP (More Expressive)

2-hidden-layer MLP (input -> 32 -> 16 -> 4 classes) with ReLU activations and softmax output. `predict_informed_prob()` returns `P(informed) + P(liquidation)` since both are toxic to market makers.

See [implementation.md](./implementation.md) for both classifier implementations.

---

## Training

Standard mini-batch training with Adam optimizer (lr=0.001), batch size 64, 100 epochs with early stopping on validation AUC. Evaluation uses binary toxic vs non-toxic classification (Informed + Liquidation = toxic).

See [implementation.md](./implementation.md) for training loop and AUC computation code.

---

## Real-Time Integration

`AdverseSelectionAdjuster` wraps the classifier for real-time use:

- **`on_trade()`**: Classifies each trade and updates an EMA of informed probability (alpha ~0.1, decays over ~10 trades)
- **`get_kappa_adjustment()`**: Reduces kappa proportional to informed flow intensity (floors at 0.3 -- never reduce by more than 70%)
- **`get_spread_adjustment_bps()`**: Widens spread proportional to informed intensity
- **`should_go_defensive()`**: Returns true when informed intensity exceeds `high_toxicity_threshold`

See [implementation.md](./implementation.md) for full `AdverseSelectionAdjuster` code.

---

## Liquidation Detector (Specialized Subsystem)

Liquidations deserve their own detector because they're:
- Highly predictable from OI + funding
- Extremely toxic
- Need fast response (pull quotes, don't just widen)

`LiquidationDetector` tracks OI and funding history in ring buffers and computes cascade probability from four additive signals:

| Signal | Contribution | Typical Threshold |
|--------|-------------|-------------------|
| Rapid OI drop (1m) | +0.3 | OI change < -2% |
| Rapid OI drop (5m) | +0.3 | OI change < -5% |
| Extreme funding | +0.2 | Funding percentile > 95th or < 5th |
| Funding-direction squeeze | +0.2 | Extreme funding + OI drop > 1% |

Decision thresholds: `is_cascade_active()` at prob > 0.5, `should_pull_quotes()` at prob > 0.7. Max probability capped at 0.95.

See [implementation.md](./implementation.md) for full `LiquidationDetector` code.

---

## Validation

### Key Metrics

Track `ClassifierValidation` with: AUC-ROC, precision@50% recall, per-class confusion matrix/precision/recall, Brier score, information ratio, and economic metrics (PnL with/without classifier, improvement %).

See [implementation.md](./implementation.md) for the `ClassifierValidation` struct definition.

### Acceptance Criteria

- AUC > 0.65 (informed vs noise classification)
- Information Ratio > 1.0 (adds value vs base rate)
- PnL improvement > 10% vs no classifier

---

## Dependencies

- **Requires**: measurement-infrastructure, signal-audit
- **Enables**: Dynamic kappa adjustment, spread widening on toxic flow

## Common Mistakes

1. **Overfitting to size**: Large trades aren't always informed
2. **Ignoring timing**: Clusters of trades are more toxic than isolated ones
3. **Missing cross-exchange**: Arbitrage flow is predictable from Binance
4. **Static thresholds**: Liquidation conditions vary with market state
5. **Binary thinking**: Use probabilities, not hard classifications

## Next Steps

1. Build labeled dataset from historical fills + price evolution
2. Run signal audit to identify predictive features
3. Train classifier (start with logistic regression for interpretability)
4. Validate AUC > 0.65, IR > 1.0
5. Integrate into quote engine
6. Build liquidation detector as separate fast path
7. Monitor classifier decay (retrain monthly)

## Supporting Files

- [implementation.md](./implementation.md) -- All Rust code: feature structs, classifiers, training loop, real-time integration, liquidation detector, validation
