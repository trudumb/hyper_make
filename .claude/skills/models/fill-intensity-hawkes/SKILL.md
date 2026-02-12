---
name: fill-intensity-hawkes
description: Model fill intensity using Hawkes processes for state-dependent kappa estimation. Use when upgrading from simple fill-rate kappa, building fill probability predictions, incorporating queue position dynamics, or adding Hyperliquid-specific features (funding, OI) to fill rate models.
user-invocable: false
---

# Fill Intensity Model (Hawkes Process) Skill

## Purpose

Model the full intensity process for fills, conditional on exchange-specific state. This replaces simple frequency-based kappa estimation with a proper probabilistic model that captures:

- Self-exciting dynamics (trades beget trades)
- State-dependent baseline intensity
- Queue position effects
- Hyperliquid-specific features (funding, OI)

## When to Use

- Upgrading from simple "fills per second" kappa estimation
- Building fill probability predictions
- Incorporating queue position dynamics
- Adding exchange-specific features to fill rate estimation

## Prerequisites

- `measurement-infrastructure` for prediction logging and validation
- `signal-audit` to identify which features have predictive power for fills
- Historical fill data with market state

---

## Theoretical Foundation

### Standard Hawkes Process

A self-exciting point process where each event increases the probability of future events:

```
λ(t) = μ + ∫₀ᵗ α·e^(-β(t-s)) dN(s)
```

Where:
- λ(t) = instantaneous intensity (expected events per unit time)
- μ = baseline intensity
- α = excitation magnitude (how much each event increases intensity)
- β = decay rate (how quickly excitation fades)
- N(s) = counting process (number of events by time s)

**Stationarity condition:** α/β < 1 (excitation must decay faster than it accumulates)

### Intuition for Market Making

When a trade occurs:
1. It consumes liquidity (queue moves forward)
2. It signals activity (other participants may trade)
3. Both effects increase your fill probability temporarily

The Hawkes process captures this clustering of fills.

---

## Extensions for Hyperliquid

The standard Hawkes is too simple. We extend it with:

### Extension 1: State-Dependent Baseline

The baseline intensity μ varies with market state:

```
μ(t) = μ₀ · exp(w_F·F(t) + w_OI·ΔOI(t) + w_τ·τ(t) + w_σ·σ(t))
```

Where:
- F(t) = funding rate (extreme funding → more activity)
- ΔOI(t) = OI change rate (new positions opening/closing)
- τ(t) = time to funding settlement (cyclical feature)
- σ(t) = current volatility

`compute_baseline_intensity()` multiplies `mu_0` by an exponential of weighted market features (funding, OI change, settlement phase, volatility). See [implementation.md](./implementation.md) for full code.

### Extension 2: Trade-Type-Dependent Excitation

Different trades have different excitation effects:

`compute_excitation()` computes per-trade alpha as `alpha_base * size_mult * side_mult * aggressor_mult`. Size effect is sublinear (exponent 0.3-0.5, capped at 3x), same-side trades get ~1.5x, aggressors get ~1.2x. See [implementation.md](./implementation.md) for full code.

### Extension 3: Queue-Position-Dependent Kernel

The kernel shouldn't just depend on time—it should depend on queue consumption:

`adaptive_kernel()` returns `exp(-beta * time) * (1 + queue_sensitivity * consumed_fraction)` -- standard temporal decay boosted by queue consumption. See [implementation.md](./implementation.md) for full code.

---

## Full Model API

### Core Structs

`HyperliquidFillIntensityModel` contains `BaselineParams` (mu_0, w_funding, w_oi, w_settlement, w_volatility), `ExcitationParams` (alpha_base, size_exponent, same_side_mult, aggressor_mult, median_trade_size), `KernelParams` (beta, queue_sensitivity), depth_half_life_bps, and regime_multipliers. See [implementation.md](./implementation.md) for full struct definitions.

### Key Methods

- **`intensity_at(t, recent_trades, queue_position, queue_history, market_state, our_side) -> f64`** -- Computes `(mu(t) + sum(alpha_i * kernel_i)) * regime_mult`. Sums excitation from trades within 60s window.
- **`expected_fills_in_window(t_start, t_end, depth_bps, ...) -> f64`** -- Numerical integration of intensity over time window with depth decay (`exp(-0.693 * depth / half_life)`).
- **`fill_probability(...) -> f64`** -- Returns `1 - exp(-expected_fills)` (Poisson survival).

See [implementation.md](./implementation.md) for full struct definitions and method bodies.

---

## Parameter Estimation

### Batch Estimation (MLE)

For initial parameter estimation or periodic retraining. The log-likelihood is `sum(log(lambda(t_i))) - integral(lambda(t), 0, T)`, computed via numerical integration (1000 steps). Optimized with L-BFGS (100 iterations, tolerance 1e-6).

Key functions: `hawkes_log_likelihood()` and `fit_hawkes_model()`. See [implementation.md](./implementation.md) for full code.

### Online Estimation

`OnlineHawkesEstimator` performs real-time parameter updates via stochastic gradient:

- **`on_fill()`**: Computes innovation `(1 - predicted * dt)`, applies gradient step to `mu_0` (clamped to 0.5x-2.0x per step), decays learning rate.
- **`on_no_fill()`**: If expected fills > 0.5 but none observed, reduces `mu_0` proportionally.

See [implementation.md](./implementation.md) for full `OnlineHawkesEstimator` struct and methods.

---

## Converting Intensity to Kappa

The GLFT formula uses kappa: fill rate per unit spread. Convert from Hawkes intensity:

`intensity_to_kappa()` estimates kappa via finite difference: perturbs depth by 0.5 bps, measures fill rate change over a 1-second window, divides by depth change in fraction. Floors at 100.0 to prevent GLFT division issues. See [implementation.md](./implementation.md) for full code.

---

## Validation

### Calibration Checks

`validate_fill_model()` extracts per-level fill predictions vs actual outcomes from `PredictionRecord` data, then computes Brier score decomposition and calibration curve (20 bins). See [implementation.md](./implementation.md) for full code.

### Key Metrics

- **Brier Score**: Should be < 0.15 for good fill prediction
- **Information Ratio**: Must be > 1.0 (or model is adding noise)
- **Calibration Curve**: Should follow y=x diagonal

---

## Regime-Specific Parameters

Default regime multipliers:

| Regime   | Multiplier | Rationale |
|----------|------------|-----------|
| Quiet    | 1.0        | Baseline |
| Trending | 0.8        | One side gets picked off, other doesn't fill |
| Volatile | 2.0        | High activity, lots of fills |
| Cascade  | 5.0        | Extreme activity, fills are certain but toxic |

These are set via `default_regime_multipliers()`. See [implementation.md](./implementation.md) for code.

---

## Integration Points

### With Quote Engine

`QuoteEngine::compute_kappa()` chains: `intensity_to_kappa()` at 10bps reference depth, then multiplies by adverse selection adjustment and regime-blended kappa multiplier. Result is floored at `MIN_KAPPA`. See [implementation.md](./implementation.md) for full code.

---

## Dependencies

- **Requires**: measurement-infrastructure, signal-audit (to identify features)
- **Enables**: Better kappa estimation for GLFT, fill probability predictions

## Common Mistakes

1. **Ignoring queue position**: Fill probability depends heavily on queue position
2. **Same α for all trades**: Different trade types excite differently
3. **Forgetting depth decay**: Fills at 20bps are much rarer than at 2bps
4. **Not updating online**: Parameters drift; need continuous adaptation
5. **Overcomplicating**: Start with basic Hawkes, add extensions incrementally

## Next Steps

1. Implement basic Hawkes with MLE estimation
2. Validate against measurement infrastructure
3. Add state-dependent baseline (funding, OI)
4. Add trade-type excitation
5. Add queue-position kernel
6. Set up online updating
7. Integrate into quote engine via `intensity_to_kappa`

## Supporting Files

- [implementation.md](./implementation.md) -- All Rust code: baseline intensity, excitation kernel, full model structs and methods, MLE estimation, online estimator, intensity-to-kappa conversion, validation, regime multipliers, quote engine integration
