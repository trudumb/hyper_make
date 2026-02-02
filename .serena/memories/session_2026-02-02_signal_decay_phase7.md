# Session 2026-02-02: Signal Decay Tracking (Phase 7.1)

## Summary
Implemented Phase 7.1 of the alpha-generating architecture improvements: Signal Decay Tracking for latency-adjusted calibration.

## Problem Addressed
The user's critique identified a blind spot: "Your plan mentions 'CalibrationState' (Brier/IR), but it doesn't account for Signal Decay... You need a Latency-Adjusted IR."

Signals like VPIN are highly predictive when fresh (0-10ms) but worthless after 50ms. If our processing latency exceeds the signal's alpha duration, we're providing "free options" to faster participants.

## Implementation

### New File: `src/market_maker/calibration/signal_decay.rs`
- `SignalDecayConfig` - per-signal decay parameters (half_life_ms, floor, freshness_multiplier)
- `SignalDecayTracker` - tracks signal emissions and outcomes
- `SignalEmission` / `SignalOutcome` - data structures for tracking
- `LatencyAccumulator` - internal stats accumulator

Key methods:
- `emit()` - record signal emission with direction prediction
- `record_outcome()` - link outcome to emission, compute decay-adjusted value
- `latency_adjusted_ir()` - IR computed only for "fresh" signals
- `alpha_duration_ms()` - time until signal value drops below threshold
- `is_latency_constrained()` - checks if processing latency exceeds alpha duration

### New Struct: `LatencyCalibration` in `snapshot.rs`
Added to `CalibrationState`:
```rust
pub struct LatencyCalibration {
    pub vpin_latency_adjusted_ir: f64,
    pub vpin_alpha_duration_ms: f64,
    pub flow_latency_adjusted_ir: f64,
    pub flow_alpha_duration_ms: f64,
    pub signal_latency_budget_ms: f64,
    pub is_latency_constrained: bool,
    pub mean_signal_to_quote_ms: f64,
    pub p95_signal_to_quote_ms: f64,
    pub ir_degradation_ratio: f64,
}
```

## Files Modified
1. `src/market_maker/calibration/signal_decay.rs` (NEW) - 18 tests
2. `src/market_maker/calibration/mod.rs` - Added module and exports
3. `src/market_maker/belief/snapshot.rs` - Added LatencyCalibration struct
4. `src/market_maker/belief/central.rs` - Import and use LatencyCalibration

## Tests
- 18 signal_decay tests passing
- 54 belief tests passing
- 1772 total tests passing

## Remaining Work
Wire SignalDecayTracker into orchestrator handlers to track real emissions/outcomes during production.

## V2 Refinements Progress
| Phase | Status |
|-------|--------|
| 1A.2 COFI | ✅ Done |
| 1A.1 Trade Size | ✅ Done |
| 7.1 Signal Decay | ✅ Done |
| 2A.1 Belief Skewness | Pending |
| 4A.2 Funding Magnitude | Pending |
| 4A.1 BOCPD | Pending |
| 6A.1 PCA | Pending |
| 6A.2 RL Tuning | Pending |

## Key Formulas

### Signal Decay
```
value(t) = value_0 × 2^(-t/half_life) + floor × (1 - 2^(-t/half_life))
```

### Latency-Adjusted IR
IR computed only for predictions where `signal_age < 2 × half_life`.
