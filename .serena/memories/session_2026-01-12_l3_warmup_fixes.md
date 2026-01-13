# Session: 2026-01-12 L3 Stochastic Controller Warmup Fixes

## Summary
Fixed Layer 3 stochastic controller bugs causing constant changepoint detection, zero learning trust, and perpetual WaitToLearn state at startup.

## Root Cause Analysis

From log analysis (`logs/mm_mainnet_BTC_2026-01-12_22-18-30.log`):

### Problem 1: Changepoint Over-Firing
- At startup, `run_length_probs = [1.0]` (100% probability of run length 0)
- This was interpreted as 100% changepoint probability
- `should_reset_beliefs()` fired every cycle, constantly resetting beliefs

### Problem 2: Learning Trust Zero
- `assess_learning_trust()` calculated: `0.5 * (1.0 - 1.0) = 0`
- Zero trust caused defensive behavior throughout operation

### Problem 3: Perpetual WaitToLearn
- High uncertainty (from belief resets) triggered `should_wait()` 
- WaitToLearn returned every cycle, but didn't cancel existing orders
- Orders placed in early cycles stayed on book indefinitely

### Problem 4: Beliefs Never Accumulated
- Every cycle: changepoint fires → beliefs reset → high uncertainty → WaitToLearn
- No opportunity for the system to learn from fills

## Fixes Implemented

### Fix 1: Changepoint Warmup Period (`changepoint.rs`)
Added warmup period before changepoint detection activates:
- New field: `warmup_observations: usize` (default: 20)
- New field: `observation_count: usize`
- `is_warmed_up()` method checks `observation_count >= warmup_observations`
- `changepoint_detected()` returns false during warmup
- `should_reset_beliefs()` returns false during warmup
- Summary includes `warmed_up` and `observation_count` for diagnostics

### Fix 2: Learning Trust Floor (`control/mod.rs`)
Improved `assess_learning_trust()`:
- During warmup: use baseline trust of 0.7 instead of 0
- After warmup with high changepoint prob: clamp to minimum 0.1
- Prevents complete paralysis from zero trust

### Fix 3: Skip WaitToLearn During Warmup (`control/mod.rs`)
Modified `act()` method:
- Check `changepoint.is_warmed_up()` before evaluating WaitToLearn
- During warmup: use myopic actions to allow position building
- Only consider waiting after sufficient data gathered

### Fix 4: Beliefs Only Reset on Genuine Changepoints
Covered by Fix 1 - `should_reset_beliefs()` returns false during warmup

## Files Modified

| File | Changes |
|------|---------|
| `src/market_maker/control/changepoint.rs:95-109` | Added `warmup_observations` to config |
| `src/market_maker/control/changepoint.rs:85-88` | Added warmup fields to detector |
| `src/market_maker/control/changepoint.rs:150-158` | Initialize warmup in constructor, add `is_warmed_up()` |
| `src/market_maker/control/changepoint.rs:222-224` | Increment observation_count in `update()` |
| `src/market_maker/control/changepoint.rs:238-261` | Add warmup checks to `changepoint_detected()` and `should_reset_beliefs()` |
| `src/market_maker/control/changepoint.rs:284-289` | Reset observation_count in `reset()` |
| `src/market_maker/control/changepoint.rs:292-326` | Add warmup status to summary |
| `src/market_maker/control/mod.rs:381-415` | Improved `assess_learning_trust()` with warmup handling |
| `src/market_maker/control/mod.rs:206-230` | Skip WaitToLearn during warmup |

## Expected Behavior After Fix

1. **First 20 observations**: Warmup period
   - Changepoint detection disabled
   - Learning trust starts at 0.7
   - WaitToLearn skipped
   - Myopic actions used (allows position building)
   
2. **After warmup**: Normal operation
   - Changepoint detection active
   - Learning trust based on actual changepoint probability
   - WaitToLearn evaluated normally
   - Beliefs reset only on genuine regime changes

## Verification

```bash
cargo build  # ✅ Passed
cargo test changepoint  # ✅ 5 tests passed
cargo test control::tests  # ✅ 4 tests passed
```

## Log Indicators

Before fix:
```
l3_trust":"0.00","l3_cp_prob":"1.000","l3_action":"WaitToLearn
Changepoint detected - resetting beliefs  # Every second
```

After fix (expected):
```
l3_trust":"0.70","l3_cp_prob":"1.000","l3_action":"Quote  # During warmup
# No changepoint logs until warmup complete
```

## Related Memories
- `session_2026-01-12_layer3_production_wiring`
- `session_2026-01-12_stochastic_controller_layer3`
