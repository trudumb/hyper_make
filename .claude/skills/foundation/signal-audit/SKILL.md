---
name: Signal Audit
description: Systematically measure the predictive information content of all available signals before building models
user-invocable: false
---

# Signal Audit Skill

## Purpose

Systematically measure the predictive information content of all available signals before building models. This prevents wasting effort on low-value features and identifies high-value signals you might be ignoring.

## When to Use

- Before building any new predictive model
- When adding a new data source (cross-exchange feed, new API field)
- Quarterly review of signal value decay
- Debugging why a model stopped working
- Deciding which features to include in a model

## Prerequisites

- `measurement-infrastructure` implemented (for outcome data)
- Historical market data with candidate signals
- Defined prediction targets

---

## Core Concept: Mutual Information

Mutual information measures how many bits of information signal X provides about target Y:

```
I(X; Y) = H(Y) - H(Y|X)
```

Where H is entropy.

**Key properties:**
- I(X; Y) ≥ 0 (always non-negative)
- I(X; Y) = 0 if and only if X and Y are independent
- Works for non-linear relationships (unlike correlation)
- Units are bits (or nats if using natural log)

---

## Signal Catalog

### Book-Derived Signals

- **Basic**: `spread_bps`, `mid_price`
- **Imbalance**: `microprice_imbalance` (L1), `book_imbalance_l5` (top 5), `book_pressure` (weighted depth asymmetry)
- **Depth**: `depth_at_1bps`, `depth_at_5bps`, `depth_at_10bps`
- **Shape**: `book_slope_bid`, `book_slope_ask` (depth increase rate away from mid)

### Trade-Derived Signals

- **Volume imbalance**: `trade_imbalance_{1s,10s,60s}` (net signed volume)
- **Intensity**: `trade_arrival_rate`, `volume_rate`
- **Size distribution**: `avg_trade_size`, `trade_size_std`, `large_trade_count_1m` (> 2 sigma)
- **Aggression**: `aggressor_imbalance`

### Hyperliquid-Specific Signals

- **Funding**: `funding_rate`, `funding_rate_change_{1h,8h}`, `predicted_funding_rate`, `time_to_funding_settlement_s`
- **Open Interest**: `open_interest`, `open_interest_change_{1m,5m,1h}`, `oi_momentum`
- **Vault**: `hlp_vault_position`

### Cross-Exchange Signals

- **Binance**: `binance_mid`, `binance_spread_bps`, `binance_hl_basis_bps`
- **Lead indicators**: `binance_return_{100ms,500ms,1s}`
- **Volume ratio**: `binance_volume_ratio` (Binance / HL)

### Composite Signals

- **Interactions**: `funding_x_imbalance`, `oi_x_funding`, `basis_x_imbalance`
- **Momentum**: `price_momentum_{1m,5m}`, `volume_momentum`

See [implementation.md](./implementation.md) for full struct definitions.

---

## Prediction Targets

- **Direction**: `PriceDirection{1s,10s,60s}` -- sign of future return
- **Magnitude**: `AbsReturn{1s,10s}`, `Volatility1m`
- **Fill-related**: `FillWithin{1s,10s}`, `TimeToNextFill`
- **Adverse selection**: `AdverseOnNextFill`, `InformedFlow`
- **Regime**: `RegimeTransition` (will regime change in next minute?)

---

## Mutual Information Estimation

### k-NN Estimator (Kraskov et al.)

For continuous variables, use the k-nearest-neighbor estimator (`estimate_mutual_information`). Normalizes inputs to [0,1], builds k-d trees in joint and marginal spaces, then computes MI via digamma functions on neighbor counts. Typical k = 3-10.

### For Binary Targets

Use the simpler binned estimator (`estimate_mi_binary_target`). Bins the continuous variable, counts joint/marginal frequencies, and computes MI from the frequency table.

See [implementation.md](./implementation.md) for full code of both estimators.

---

## Signal Analysis Framework

`SignalAnalysisResult` captures per-signal metrics:

| Field | Description |
|-------|-------------|
| `mutual_information_bits` | Raw MI in bits |
| `mutual_information_normalized` | MI / H(Y), fraction of target entropy explained |
| `correlation`, `correlation_abs` | Linear relationship (for comparison) |
| `auc_roc` | ROC AUC for binary targets |
| `optimal_lag_ms`, `mi_at_optimal_lag` | Best lag and MI at that lag |
| `mi_by_regime`, `regime_variance_ratio` | Per-regime MI and max/min ratio |
| `mi_trend_30d` | Stationarity trend |

`analyze_signal()` computes MI, correlation, optimal lag (candidate lags: -500ms to +500ms), and per-regime MI (min 100 samples per regime). `find_optimal_lag()` sweeps candidate lags and returns the one with highest MI.

See [implementation.md](./implementation.md) for full code.

---

## Signal Audit Report

`generate_signal_audit_report()` analyzes all signals against a target, sorts by MI descending, and produces a table with actionable insights. The report flags: (1) highest-MI signal, (2) regime-conditional signals (variance ratio > 2x), (3) signals with 20%+ more MI at a lag, (4) high-correlation but low-MI signals (possibly spurious).

See [implementation.md](./implementation.md) for full code.

### Example Report Output

```
=== Signal Audit Report ===
Target: PriceDirection1s

Signal                      MI (bits)  Corr    Opt Lag   Regime Var
─────────────────────────────────────────────────────────────────────
binance_return_100ms        0.0890     0.31    -150ms    2.3x
trade_imbalance_1s          0.0670     0.24       0ms    1.4x
microprice_imbalance        0.0450     0.19       0ms    1.2x
funding_x_imbalance         0.0410     0.15       0ms    3.1x
open_interest_change_1m     0.0230     0.08       0ms    1.1x
book_pressure               0.0180     0.11       0ms    1.3x
funding_rate                0.0120     0.05       0ms    1.8x

ACTIONABLE INSIGHTS:
1. binance_return_100ms has highest MI (0.089 bits) - prioritize if not already used
2. funding_x_imbalance has 3.1x higher MI in some regimes - consider regime conditioning
3. binance_return_100ms has 20%+ more MI at -150ms lag - incorporate lag in feature
```

---

## Signal Quality Thresholds

Typical thresholds:

| Parameter | Typical Value | Purpose |
|-----------|--------------|---------|
| `min_mi_bits` | 0.01 | Minimum MI to include in model |
| `min_samples` | 1000 | Minimum samples for reliable MI estimate |
| `max_regime_variance` | 3.0 | Above this, require regime conditioning |
| `min_correlation` | 0.05 | Sanity check floor |

`filter_signals()` keeps signals above MI and correlation thresholds. `flag_regime_conditional()` flags signals with regime variance ratio above threshold.

See [implementation.md](./implementation.md) for full code.

---

## Tracking Signal Decay

Signals lose value over time as:
- Other participants discover them
- Market structure changes
- Regime shifts

`compute_signal_decay()` fits a linear regression on historical MI values over time and computes a half-life (days until MI drops 50%). Action thresholds:

- **< 30 days**: URGENT -- signal decaying rapidly, investigate or replace
- **< 90 days**: WARNING -- monitor closely
- **>= 90 days**: OK -- signal stable

See [implementation.md](./implementation.md) for full code.

---

## Dependencies

- **Requires**: measurement-infrastructure (for outcome data), historical market data
- **Enables**: All model skills (by identifying which features to use)

## Common Mistakes

1. **Using correlation instead of MI**: Correlation misses non-linear relationships
2. **Not checking lag**: Some signals lead the target and are more valuable at a lag
3. **Ignoring regime conditioning**: A signal useless overall might be gold in specific regimes
4. **Not tracking decay**: Signals that worked last year might be worthless now
5. **Too few samples**: MI estimation needs 1000+ samples for reliability

## Next Steps

After signal audit:
1. Select top signals for your target (MI > 0.01 bits)
2. Flag regime-conditional signals for special handling
3. Incorporate optimal lags into feature engineering
4. Read the relevant model skill to build the predictor
5. Set up decay tracking for production monitoring

## Supporting Files

- [implementation.md](./implementation.md) -- All Rust code: signal structs, MI estimators, analysis framework, report generation, quality thresholds, decay tracking
