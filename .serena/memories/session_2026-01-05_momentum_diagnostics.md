# Session: 2026-01-05 Momentum Diagnostics Analysis

## Summary
Investigated why drift-adjusted skew feature wasn't activating during 10-minute HIP-3 test. Found that short-term momentum signals differ from medium-term price trends.

## Test Results

### 10-minute HIP-3 Test
- **Asset**: HYPE on hyna DEX
- **Duration**: ~9 minutes
- **Fills**: 14 (ALL BUYS, 0 sells)
- **Position**: 36.06 HYPE long
- **P&L**: -$4.18 (negative - caught wrong side of trend)
- **Price**: Fell 114 bps ($26.81 → $26.50) over test period

### Key Finding: Momentum vs Trend Mismatch
The drift-adjusted skew only activates when position **opposes** momentum:
- `is_opposed = (q * momentum_bps < 0)` 

During the test:
- **Position**: LONG (q > 0)
- **Short-term momentum**: +14 bps (POSITIVE - indicating short-term rise)
- **Medium-term trend**: -114 bps (NEGATIVE - falling)
- **Result**: `is_opposed = false` → no drift urgency applied

**Insight**: Within a downtrend, there are short-term bounces that register as positive momentum. The momentum detector window (~1s-5s) captures these bounces, not the longer trend.

## Technical Details

### Momentum Detection Flow
1. `MomentumDetector.on_bucket()` receives VWAP returns from volume buckets
2. `momentum_bps()` sums returns over `momentum_window_ms` (default: 5000ms)
3. Returns signed momentum in bps (+ = rising, - = falling)

### Drift Adjustment Logic (hjb_control.rs)
```rust
// is_opposed: position opposes momentum direction
let is_opposed = q * momentum_bps < 0.0;

// Only apply drift urgency when:
// 1. is_opposed == true (position vs momentum)
// 2. momentum exceeds dynamic threshold (3-8 bps based on position size)
// 3. p_continuation >= min_continuation_prob (0.5)
```

### Why No Activation
With `momentum_bps = +14` (positive, short-term bounce):
- Long position (q > 0) + positive momentum → NOT opposed
- Drift urgency = 0
- No additional skew to accelerate position reduction

## Potential Improvements

### Option 1: Multi-Timeframe Momentum
Track momentum at multiple windows:
- Short (5s): Current behavior, for quick reactions
- Medium (30-60s): For trend confirmation
- Apply drift adjustment if EITHER window shows opposition

### Option 2: Cumulative Adverse Fill Direction
Track direction of recent fills:
- If position building in one direction (all buys/sells)
- AND adverse selection is negative
- Consider this as implicit "trend opposition"

### Option 3: Momentum Trend (derivative)
Track if momentum is increasing or decreasing:
- Even if momentum = +14 bps
- If momentum was +30 bps a minute ago → falling momentum
- Could indicate turning point

### Option 4: Hybrid Price-Level Check
Compare current position entry price vs current mid:
- If long AND current_mid < avg_entry_price → position underwater
- Apply urgency even without "momentum opposition"

## Inventory Skew Still Working
Note: The standard inventory skew IS working correctly:
- Quote asymmetry observed: ask closer (5.5 bps) vs bid (8.9 bps)
- This naturally encourages selling to reduce position
- But not aggressive enough in sustained trends

## Files Modified
- `src/market_maker/mod.rs:1357-1375` - Updated logging condition (>10 bps threshold)

## Verification
- Added INFO-level diagnostics to confirm momentum values
- Confirmed momentum_bps, p_continuation, is_opposed all computing correctly
- EWMA warming up correctly (~20s to warm)

## Next Steps
1. Consider implementing Option 2 (cumulative fill direction) as enhancement
2. Potentially add medium-timeframe momentum signal
3. The current system works but relies on momentum opposition which may not trigger in sustained trends with bounces
