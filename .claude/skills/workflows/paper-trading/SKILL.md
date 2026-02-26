---
name: paper-trading
description: Set up, debug, and validate the paper trading system
disable-model-invocation: true
context: fork
agent: general-purpose
argument-hint: "[asset]"
---

# Paper Trading Workflow

Guide for setting up, debugging, and validating the paper trading system. The paper trader uses simulated fills against real market data to validate strategies without risking capital.

## Key Files

```
src/bin/paper_trader.rs                    # Main binary
src/market_maker/simulation/fill_sim.rs    # Fill probability simulator
src/market_maker/simulation/outcome.rs     # Fill outcome tracking
src/market_maker/simulation/calibration.rs # Fill sim calibration
```

## Setup

The paper trader connects to real market data feeds but simulates fills locally:

1. Real L2 book, trades, and mid-price updates from exchange
2. Simulated fill logic based on `FillSimulator`
3. Local position and PnL tracking
4. All estimators and models run on real data

### Configuration

Key settings in paper trader config:
- `ignore_book_depth: true` — paper trader orders don't exist in real book, so queue position must use flat model
- `fill_prob_base` — base fill probability (typically 0.3-0.5)
- `adverse_fill_ratio` — fraction of fills that are adverse (for testing robustness)

## Three Critical Feedback Loops

These must ALL be working for the paper trader to produce meaningful results. If any is broken, the system won't learn from its simulated experience.

### 1. Kappa from Own Fills

```
Simulated fill -> estimator.on_own_fill() -> kappa update
```

- In `on_simulated_fill()`, call `estimator.on_own_fill(side, price, size, mid)`
- Without this: `own_fills=0` forever, kappa stuck at prior values
- Symptom: kappa never changes regardless of fill behavior

### 2. Adverse Selection Outcome Feedback

```
Simulated fill -> wait 5s -> check mid price -> record_outcome()
```

- Uses `PendingFillOutcome` deque, checked in `update_mid()`
- Each fill gets a markout check: did mid move against us > 1 bps?
- Calls `pre_fill_classifier.record_outcome(was_adverse)`
- Without this: AS classifier never learns, toxicity scores meaningless

### 3. Calibration Controller

```
kappa_confidence + as_progress -> cal_progress -> cal_gamma
```

- `CalibrationController` tracks progress via `(as_progress + kappa_progress) / 2`
- `cal_gamma` starts at 0.3 (fill-hungry = tight spreads) and rises toward 1.0
- Driven by `CalibrationController` imported via market_maker re-export
- Without this: `calibration_progress` stuck at 0.0, gamma never adjusts

## Common Issues

### Zero Fills

**Root cause**: `compute_fill_probability()` uses real L2 book depth for queue position, but paper trader orders don't exist in the real book.

**Fix**: Set `ignore_book_depth: true` on `FillSimulatorConfig`. This uses a flat queue model instead of trying to estimate queue position in the real book.

**Verify**: After fix, you should see fills within 1-2 minutes of starting.

### Margin Bug

**Root cause**: `update_state()` never called, so margin tracking thinks there's no available margin.

**Fix**: Ensure `update_state()` is called on position changes to keep margin tracker in sync.

### Kappa Stuck at Prior

**Symptom**: `kappa_confidence` stays at 0, kappa never updates from initial values.

**Root cause**: `estimator.on_own_fill()` not being called when simulated fills occur.

**Fix**: Wire `on_own_fill()` call in `on_simulated_fill()` handler.

### Toxicity Always 0.5

**Symptom**: Pre-fill toxicity scores are always near 0.5 (neutral).

**Root cause**: `pre_fill_classifier.record_outcome()` never called, so classifier has no training data.

**Fix**: Implement `PendingFillOutcome` deque with 5s markout check in `update_mid()`.

### Cal Progress Stuck at 0

**Symptom**: `calibration_progress` always 0.0, `cal_gamma` at initial value.

**Root cause**: `CalibrationController` not instantiated or not being updated.

**Fix**: Import and instantiate `CalibrationController`, feed it kappa_confidence and as_progress.

## Validation Checklist

After starting the paper trader, verify these within the first 30 minutes:

- [ ] Fills occurring (at least a few per minute in active market)
- [ ] `kappa_confidence` increasing from 0
- [ ] `calibration_progress` > 0
- [ ] PnL tracking working (non-zero realized + unrealized)
- [ ] Adverse selection outcomes being recorded
- [ ] Regime detection updating (not stuck on one regime)
- [ ] Checkpoint saves occurring (every 5 minutes)

## Running

```bash
# Build
cargo build --release --bin paper_trader

# Run (user executes manually)
RUST_LOG=info ./target/release/paper_trader
```

## Interpreting Results

- **Positive PnL in quiet markets**: good baseline, strategy captures spread
- **Negative PnL in volatile**: expected, but magnitude matters — should be small
- **High adverse selection rate (> 40%)**: AS classifier needs work
- **Low fill rate**: kappa estimate may be too low, or spreads too wide
- **Calibration progress > 50%**: system is learning, can trust parameter estimates
