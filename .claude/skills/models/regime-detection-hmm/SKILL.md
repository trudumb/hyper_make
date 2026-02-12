---
name: regime-detection-hmm
description: Bayesian belief tracking over market states (Quiet/Trending/Volatile/Cascade) using Hidden Markov Models. Use when implementing regime-dependent parameters, building aggressive/defensive decisions, debugging regime-specific losses, or replacing hard-coded volatility thresholds with smooth probability blending.
user-invocable: false
---

# Regime Detection (HMM) Skill

## Purpose

Replace crude regime proxies (warmup multipliers, hard thresholds) with proper Bayesian belief tracking over market states. This enables:

- Smooth parameter blending between regimes
- Anticipating regime transitions
- Regime-specific model deployment
- Principled uncertainty handling

## When to Use

- Implementing regime-dependent parameters
- Building the "should I be aggressive or defensive" decision
- Debugging regime-specific losses
- Replacing hard-coded volatility thresholds

## Prerequisites

- `measurement-infrastructure` for validation
- Historical market data with regime labels (can be derived)
- Understanding of your loss patterns by market condition

---

## State Space Definition

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MarketRegime {
    Quiet,      // Low volatility, balanced flow, normal fill rates
    Trending,   // Directional momentum, elevated adverse selection
    Volatile,   // High volatility, wide spreads, uncertain direction
    Cascade,    // Liquidation cascade, extreme toxicity
}
```

### Regime Characteristics

| Regime | Volatility | Trade Rate | Imbalance | Adverse Selection | Duration |
|--------|-----------|------------|-----------|-------------------|----------|
| Quiet | Low (sigma < 0.01%) | Low-Normal | Balanced | Low (~20%) | Hours |
| Trending | Medium | Normal-High | Directional | Medium (~40%) | 10-60 min |
| Volatile | High (sigma > 0.03%) | High | Variable | Medium (~35%) | Minutes-Hours |
| Cascade | Extreme | Very High | Extreme | Very High (~70%) | Minutes |

---

## Hidden Markov Model Specification

### Emission Model

Each regime has a `RegimeEmissionModel` with log-normal volatility, log-normal trade intensity, normal imbalance, and beta-approximated adverse selection rate. The `log_likelihood()` method computes the joint log-probability of an observation vector under that regime.

See [implementation.md](./implementation.md#emission-model) for full code.

### Transition Matrix

`HMMParams` holds a 4x4 transition matrix, per-regime emission models, and an initial state distribution. Default parameters encode: Quiet is sticky (0.95 self-transition), Cascade is short-lived (0.30 self-transition, 0.60 reverts to Volatile).

See [implementation.md](./implementation.md#hmm-parameters-and-default-transition-matrix) for full code.

---

## Online Filtering (Forward Algorithm)

`OnlineHMMFilter` maintains a belief state `[f64; 4]` updated with each observation via predict-then-update:

1. **Predict**: propagate belief through the transition matrix
2. **Update**: weight by emission likelihood, then normalize

Key API:
- `update(&mut self, obs: &ObservationVector)` -- Bayesian belief update
- `most_likely_regime(&self) -> MarketRegime` -- argmax over belief
- `regime_probabilities(&self) -> HashMap<MarketRegime, f64>` -- full distribution
- `probability_of(&self, regime: MarketRegime) -> f64` -- single regime query

`ObservationVector` carries: `timestamp_ns`, `volatility`, `trade_intensity`, `imbalance`, `adverse_selection_rate`.

See [implementation.md](./implementation.md#online-hmm-filter-forward-algorithm) for full code.

---

## Regime-Specific Parameters

Each regime maps to a `RegimeParams` struct controlling gamma (risk aversion), kappa_multiplier, spread_floor_bps, max_inventory, and quote_size_multiplier.

| Regime | gamma | kappa_mult | spread_floor_bps | max_inventory | quote_size_mult |
|--------|-------|-----------|-----------------|---------------|-----------------|
| Quiet | 0.3 | 1.0 | 5.0 | 1.0 | 1.0 |
| Trending | 0.5 | 0.7 | 10.0 | 0.5 | 0.8 |
| Volatile | 0.8 | 1.5 | 15.0 | 0.3 | 0.6 |
| Cascade | 2.0 | 5.0 | 50.0 | 0.1 | 0.3 |

### Parameter Blending

Don't hard-switch between regimes -- blend by belief probability. Use linear blending for additive quantities (spread_floor, max_inventory, quote_size) and log-space blending for multiplicative quantities (gamma, kappa_multiplier).

See [implementation.md](./implementation.md#regime-specific-parameters-and-blending) for full code.

---

## Parameter Learning (Baum-Welch)

Periodically re-estimate HMM parameters from historical data using the EM algorithm:

1. **E-Step**: Forward pass (alpha) and backward pass (beta) with scaling for numerical stability
2. **Compute posteriors**: gamma[t][i] = P(state_t = i | all obs), xi[t][i][j] = joint transition posterior
3. **M-Step**: Update transition matrix (normalize rows), emission means/stds (weighted by gamma), initial distribution

`train_hmm()` runs Baum-Welch iterations until log-likelihood convergence (delta < 0.001).

See [implementation.md](./implementation.md#parameter-learning-baum-welch) for full code.

---

## Integration with Quote Engine

In `QuoteEngine::generate_quotes()`:
1. Build `ObservationVector` from current market data
2. Call `hmm_filter.update(&obs)` to update beliefs
3. Call `blend_params_by_belief()` for smoothed regime parameters
4. Apply blended gamma, kappa, spread_floor, max_inventory to GLFT formula

See [implementation.md](./implementation.md#quote-engine-integration) for full code.

---

## Validation

### Regime Prediction Accuracy

Run the filter over historical observations with ex-post regime labels. Track accuracy and confusion matrix across all 4 regimes. Cascade detection recall is the most important metric -- missing a cascade is expensive.

See [implementation.md](./implementation.md#regime-prediction-accuracy) for full code.

### Economic Value

The real test: does regime detection improve PnL? Backtest with and without HMM regime detection, comparing PnL, Sharpe ratio, and max drawdown.

See [implementation.md](./implementation.md#economic-value-backtest) for full code.

---

## Dependencies

- **Requires**: measurement-infrastructure, historical market data
- **Enables**: Regime-specific parameters, smooth parameter blending

## Common Mistakes

1. **Hard switching**: Use probability blending, not argmax
2. **Too many states**: 4 is usually enough; more = overfitting
3. **Ignoring transition dynamics**: Time in regime matters
4. **Static parameters**: Retrain Baum-Welch periodically
5. **Observation selection**: Include adverse selection rate, not just vol/intensity

## Next Steps

1. Define 4 regimes based on your trading experience
2. Set reasonable initial emission parameters
3. Implement online filter
4. Implement parameter blending
5. Validate on historical data
6. Set up weekly Baum-Welch retraining
7. Monitor regime distribution in production

## Supporting Files

- [implementation.md](./implementation.md) -- All Rust code: state space enum, emission model, HMM params, online filter, regime params, parameter blending, Baum-Welch learning, quote engine integration, validation
