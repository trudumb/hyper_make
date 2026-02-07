---
name: add-signal
description: End-to-end workflow for adding a new predictive signal to the market maker
disable-model-invocation: true
context: fork
agent: general-purpose
argument-hint: "[signal-name] [prediction-target]"
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

# Add Signal Workflow

Step-by-step process for adding a new predictive signal. Follow measurement-before-modeling: define what you're predicting, log it, measure baseline, then build.

## Step 1: Define the Signal

Before writing code, answer:

1. **What does it predict?** (price direction, fill probability, adverse selection, volatility)
2. **At what horizon?** (100ms, 1s, 10s, 60s)
3. **What is the prediction target metric?** (Brier Score for binary, IR for continuous)
4. **What is the baseline?** (naive predictor, e.g., always predict 50% for binary)

## Step 2: Implement the Estimator

Create a new file in `src/market_maker/estimator/`:

```rust
// src/market_maker/estimator/my_signal.rs

pub struct MySignalEstimator {
    // State for online estimation
    // Use EWMA, rolling windows, or Bayesian updates
}

impl MySignalEstimator {
    pub fn new(config: MySignalConfig) -> Self { ... }

    /// Update with new market data. Called on every relevant event.
    pub fn update(&mut self, data: &MarketData) { ... }

    /// Get current signal value, normalized to a useful range.
    /// Document units in the return type or name.
    pub fn signal_value(&self) -> f64 { ... }

    /// Confidence in the signal (0.0 = no data, 1.0 = fully warmed up)
    pub fn confidence(&self) -> f64 { ... }
}
```

### Conventions

- Use `_bps` suffix for basis point values
- Use `_s` suffix for seconds
- Use EWMA with configurable halflife for online stats
- Add `#[serde(default)]` to any checkpoint fields for backward compat
- Include a `warmup_observations: usize` counter

## Step 3: Register in mod.rs

Add the module to `src/market_maker/estimator/mod.rs`:

```rust
pub mod my_signal;
pub use my_signal::MySignalEstimator;
```

## Step 4: Wire into Signal Integration

`src/market_maker/strategy/signal_integration.rs` is the central hub. Add your signal:

1. Add field to the struct that holds estimators
2. Call `update()` in the appropriate update method
3. Add signal output to `compute_signals()` or equivalent
4. Include in prediction logging

**IMPORTANT**: Only the `strategy` teammate edits `signal_integration.rs` in team mode. Others propose changes via messages.

## Step 5: Add Model Gating

In `src/market_maker/calibration/model_gating.rs`:

1. Add your signal to the gating system
2. Set initial weight to 0.0 (disabled until validated)
3. Define MI threshold for activation
4. Signal should only contribute when `should_use_model()` returns true

## Step 6: Add Prediction Logging

Ensure your signal's predictions are logged for calibration:

1. Add to `PredictionRecord` in checkpoint/prediction system
2. Log both the signal value and the prediction target outcome
3. Include market state conditioning variables

## Step 7: Set Up Calibration

In `src/market_maker/calibration/`:

1. Add Brier Score tracking for your signal's predictions
2. Add Information Ratio computation
3. Set up conditional calibration (by regime, volatility, time of day)
4. Define warning thresholds (IR < 1.0 = warning, IR < 0.8 = critical)

## Step 8: Verify

```bash
# Must pass
cargo clippy -- -D warnings
cargo test --lib

# Check signal compiles and integrates
cargo build
```

## Step 9: Validate Live

1. Run paper trader with signal enabled (weight = 0.0 initially)
2. Collect at least 24h of prediction data
3. Compute Brier Score and IR
4. If IR > 1.0, gradually increase weight via model gating
5. Monitor for 1 week before full deployment

## Checklist

- [ ] Signal has clear prediction target and horizon
- [ ] Estimator implements online update (no batch processing)
- [ ] Units documented in variable names
- [ ] Warmup tracking with confidence metric
- [ ] Registered in `estimator/mod.rs`
- [ ] Wired into `signal_integration.rs`
- [ ] Model gating configured (starts at weight 0.0)
- [ ] Prediction logging enabled
- [ ] Calibration metrics defined (Brier, IR, conditional)
- [ ] Checkpoint fields use `#[serde(default)]`
- [ ] Clippy clean, tests pass
