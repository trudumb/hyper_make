# Session: Calibration-Aware Fill Rate Controller

**Date**: 2026-01-04
**Status**: Complete ✅

## Problem Statement

User identified a **calibration bootstrap problem**: The market maker quotes "competitively" (tight spreads) but receives no fills, meaning stochastic features that require fills to calibrate (adverse selection, kappa, Thompson sampling) can never warm up.

**Root Cause**: Current warmup logic only applies to VOLATILITY warmup (from market trades), not fill-based calibration. After volatility warmup completes, system quotes tight even with 0 own fills.

## Solution: Calibration-Aware Fill Rate Controller

Implemented a controller that targets a minimum fill rate during calibration warmup, reducing gamma (tighter quotes) to attract fills. Once calibration completes, the controller phases out.

### Key Formula
```
fill_hungry_mult = min(1.0, actual_rate / target_rate)
fill_hungry_mult = max(0.3, fill_hungry_mult)  // Cap at 70% gamma reduction
blended_mult = fill_hungry_mult + (1 - fill_hungry_mult) × calibration_progress
```

## Files Changed

### Created
- `src/market_maker/estimator/calibration_controller.rs` (~200 lines)
  - `CalibrationController` struct
  - `CalibrationControllerConfig` struct
  - Methods: `record_fill()`, `update_calibration_status()`, `gamma_multiplier()`, `calibration_progress()`, `is_calibrated()`, `fill_count()`

### Modified
1. **`src/market_maker/estimator/mod.rs`**
   - Added module declaration and exports

2. **`src/market_maker/strategy/market_params.rs`**
   - Added fields: `calibration_gamma_mult`, `calibration_progress`, `calibration_complete`

3. **`src/market_maker/strategy/glft.rs`**
   - Added `calibration_scalar` to `effective_gamma()` calculation

4. **`src/market_maker/config.rs`**
   - Added to `StochasticConfig`:
     - `enable_calibration_fill_rate: bool` (default: true)
     - `target_fill_rate_per_hour: f64` (default: 10.0)
     - `min_fill_hungry_gamma: f64` (default: 0.3)

5. **`src/market_maker/core/components.rs`**
   - Added `calibration_controller: CalibrationController` to `StochasticComponents`
   - Updated constructors to initialize from config

6. **`src/market_maker/strategy/params.rs`**
   - Added calibration fields to `ParameterSources`
   - Added calibration values to `ParameterAggregator::build()` output

7. **`src/market_maker/mod.rs`**
   - Wire calibration values into `ParameterSources` construction
   - Call `record_fill()` when fills occur
   - Call `update_calibration_status()` before each quote cycle
   - Update Prometheus metrics

8. **`src/market_maker/infra/metrics.rs`**
   - Added metrics: `mm_calibration_gamma_mult`, `mm_calibration_progress`, `mm_calibration_fill_count`, `mm_calibration_complete`
   - Added `update_calibration()` method

## Configuration

```rust
// StochasticConfig defaults:
enable_calibration_fill_rate: true,    // Enable fill-rate targeting
target_fill_rate_per_hour: 10.0,       // 10 fills/hour (~2/level for 5 levels)
min_fill_hungry_gamma: 0.3,            // Max 70% gamma reduction
```

## Expected Behavior

| State | Fill Rate | Calibration | Gamma Mult | Quote Width |
|-------|-----------|-------------|------------|-------------|
| Cold start | 0 fills/hr | 0% | 0.30 | 70% tighter than GLFT optimal |
| Warming up | 5 fills/hr | 25% | 0.55 | 45% tighter |
| Half-calibrated | 10 fills/hr | 50% | 0.75 | 25% tighter |
| Nearly calibrated | 10 fills/hr | 90% | 0.95 | 5% tighter |
| Fully calibrated | any | 100% | 1.00 | Normal GLFT |

## Verification

1. **Logs**: Look for "Fill-hungry mode active" during warmup
2. **Metrics**: `mm_calibration_gamma_mult < 1.0` initially, → 1.0 as calibrated
3. **Fills**: Should see fills happen earlier than before
4. **Transition**: Once calibrated, logs stop and quotes widen to GLFT optimal

## Build Status

- ✅ `cargo build` passes
- ✅ `cargo test` passes (593 tests)
- ✅ `cargo clippy -- -D warnings` passes

## Relation to Previous Work

This complements the entropy-based order distribution system (session_2026-01-04_entropy_integration_complete.md) by ensuring the market maker can actually receive fills to calibrate those features.
