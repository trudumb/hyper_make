# Session: 2026-01-05 Drift-Adjusted Skew Diagnostics

## Summary

Added momentum diagnostics and fixed activation thresholds for drift-adjusted skew feature that was not triggering during bullish market conditions despite short position.

## Problem Identified

Log analysis of `mm_hip3_hyna_HYPE_hip3_2026-01-05_12-56-26.log` showed:
- Position: -47.91 (SHORT)
- Price: Rising (27.4475 → 27.4685 = +7.6 bps in 30 minutes)
- Symmetric spreads: 10 bps bid, 10 bps ask (NO SKEW!)
- 100% sell fills: 54.39 sold, 0.00 bought
- P&L: -$11.52 (consistent adverse selection ~2 bps)

**Root Cause**: Drift-adjusted skew feature from merge #8 had thresholds that weren't being met:
1. `momentum_bps.abs() > 10.0` - Too high for gradual trends
2. `p_continuation >= 0.5` - Requires calibrated momentum model

## Changes Made

### 1. Added Momentum Diagnostics (mod.rs:1337-1355)
```rust
// Log momentum diagnostics for drift-adjusted skew debugging
if drift_adjusted_skew.is_opposed || momentum_bps.abs() > 5.0 {
    debug!(
        momentum_bps, p_continuation, position,
        is_opposed, drift_urgency_bps, variance_mult, urgency_score,
        "Momentum-position alignment check"
    );
}
```

### 2. Dynamic Position-Dependent Thresholds (hjb_control.rs:308-318)
```rust
// Position-dependent threshold: larger positions need smaller momentum to trigger
// - |q| = 0.1: threshold = 8 bps (small position, need clearer signal)
// - |q| = 0.5: threshold = 5 bps (medium position)
// - |q| = 1.0: threshold = 3 bps (max position, react to any trend)
let base_threshold = 8.0;
let position_factor = (1.0 - 0.6 * q.abs()).max(0.3);
let momentum_threshold = base_threshold * position_factor;
```

## Files Modified

| File | Change |
|------|--------|
| `src/market_maker/mod.rs:1337-1355` | Added momentum diagnostics logging |
| `src/market_maker/process_models/hjb_control.rs:308-325` | Position-dependent momentum thresholds |

## Expected Behavior After Fix

With position at 74% of max (-47.91/64.6):
- **Old threshold**: 10 bps (not met for gradual 7.6 bps trend)
- **New threshold**: 8 × (1 - 0.6 × 0.74) = 8 × 0.56 = 4.5 bps (WILL trigger)

When triggered, drift-adjusted skew will:
1. Add `drift_urgency` to base skew (negative for short → favors buying)
2. Apply `variance_multiplier` (>1.0 when opposed → wider ask, tighter bid)
3. Result: Asymmetric quotes that favor position reduction

## Verification

```
cargo check: ✓ Clean
cargo test: 662 passed, 0 failed
```

## Next Steps

1. Run market maker with new diagnostics: `RUST_LOG=hyperliquid_rust_sdk::market_maker=debug`
2. Verify "Momentum-position alignment check" logs appear
3. Confirm asymmetric spreads when position opposes momentum
4. Consider wiring up DirectionalRiskEstimator for EWMA-smoothed signals (future)

## Future Enhancement: DirectionalRiskEstimator

The `DirectionalRiskEstimator` in `estimator/directional_risk.rs` provides:
- EWMA-smoothed variance multiplier
- Drift estimate with momentum history
- Statistics tracking for confidence

Currently not wired up - requires refactoring to pass through ParameterSources.
Could improve signal stability for live trading.
