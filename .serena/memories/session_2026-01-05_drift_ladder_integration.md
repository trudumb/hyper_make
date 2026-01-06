# Session: 2026-01-05 Drift-Adjusted Skew Integration into Ladder Generator

## Summary

Fixed critical bug where drift-adjusted skew was computed but never applied to ladder quotes, causing symmetric quotes even when position opposed market momentum (14:1 long:short fill ratio despite opposition detection).

## Root Cause

The multi-timeframe trend detection correctly computed drift urgency (~15.79 bps), but the ladder generator ignored it:

| Step | Status | Location |
|------|--------|----------|
| 1. Compute drift | ✅ | `hjb_control.rs:optimal_skew_with_trend()` |
| 2. Pass to ParameterSources | ✅ | `mod.rs:1404` |
| 3. Transfer to MarketParams | ✅ | `params.rs:852-858` |
| 4. GLFT uses drift | ✅ | `glft.rs:615-678` (single-quote mode) |
| 5. LadderParams struct | ❌ **MISSING** | `ladder_strat.rs:407-421` |
| 6. apply_inventory_skew() | ❌ **IGNORES** | `generator.rs:696-744` |

**Evidence from logs:**
- `is_opposed: true` correctly detected
- `drift_urgency_bps: 15.79` correctly calculated
- Quotes symmetric: `bid_from_mid_bps: 10.2`, `ask_from_mid_bps: 10.2`
- Position grew: 5.64 → 9.42 → 14.99 → 29.96 LONG into falling market

## Changes Made

### 1. Added drift fields to LadderParams (`quoting/ladder/mod.rs:209-220`)

```rust
// === Drift-Adjusted Skew (First Principles Extension) ===
pub use_drift_adjusted_skew: bool,
pub hjb_drift_urgency: f64,          // Fractional (0.001579 = 15.79 bps)
pub position_opposes_momentum: bool,
pub directional_variance_mult: f64,
pub urgency_score: f64,
```

### 2. Populated drift fields in ladder_strat.rs (~line 421)

Connected MarketParams drift fields to LadderParams when constructing ladder parameters.

### 3. Created `apply_inventory_skew_with_drift()` in generator.rs

Extended function that applies combined GLFT + drift skew:

```rust
let base_skew_fraction = inventory_ratio * gamma * sigma.powi(2) * time_horizon;

let drift_skew_fraction = if use_drift_adjusted_skew
    && position_opposes_momentum
    && urgency_score > 0.5
{
    hjb_drift_urgency  // Already fractional
} else {
    0.0
};

let total_skew_fraction = base_skew_fraction + drift_skew_fraction;
let offset = mid * total_skew_fraction;
```

### 4. Updated call sites in generator.rs

Both `generate()` (line 99) and `generate_asymmetric()` (line 204) now use `apply_inventory_skew_with_drift()` with full drift parameters.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `src/market_maker/quoting/ladder/mod.rs` | 209-220 | Added drift fields to LadderParams |
| `src/market_maker/quoting/ladder/mod.rs` | 386-463 | Added drift skew unit test |
| `src/market_maker/strategy/ladder_strat.rs` | ~421 | Populated drift fields from MarketParams |
| `src/market_maker/quoting/ladder/generator.rs` | 99-112 | Updated to use drift function |
| `src/market_maker/quoting/ladder/generator.rs` | 204-217 | Updated asymmetric generation |
| `src/market_maker/quoting/ladder/generator.rs` | 735-810 | Added `apply_inventory_skew_with_drift()` |

## Expected Behavior After Fix

**Before (symmetric despite opposition):**
```
market_mid: 26.75
bid: 26.72 (10.2 bps from mid)
ask: 26.78 (10.2 bps from mid)
```

**After (asymmetric with 15.79 bps drift):**
```
market_mid: 26.75
bid: 26.68 (26 bps from mid) - WIDER, discourages buying
ask: 26.75 (0 bps from mid)  - TIGHTER, encourages selling
```

## Verification

✅ cargo build (success, 1 warning about backward-compat wrapper)
✅ cargo test --lib (676 passed)
✅ New test `test_ladder_with_drift_adjusted_skew` validates asymmetric quote shift

## Skew Direction Logic

For LONG position with bearish momentum:
- `drift_urgency > 0` → positive total skew
- Both prices shift DOWN by `mid × total_skew`
- Bid becomes WIDER from mid (harder to buy more)
- Ask becomes TIGHTER to mid (easier to sell)

This correctly reduces the long position by making sells more attractive.

## Session Checkpoint

**Completed Tasks:**
- ✅ Added drift fields to `LadderParams` struct
- ✅ Connected `MarketParams` drift data to ladder generation
- ✅ Created `apply_inventory_skew_with_drift()` function
- ✅ Updated both `generate()` and `generate_asymmetric()` call sites
- ✅ Added unit test for drift-adjusted ladder behavior
- ✅ All 676 tests passing

**Cross-Session Context:**
This fix completes the multi-timeframe trend detection feature chain:
1. `session_2026-01-05_multi_timeframe_trend_detection` - Created TrendPersistenceTracker
2. `session_2026-01-05_momentum_diagnostics` - Added logging for opposition detection
3. `session_2026-01-05_drift_skew_diagnostics` - Traced data flow to find the gap
4. **This session** - Connected drift to ladder generator

**Next Session Recommendations:**
- Run live test to confirm asymmetric quotes appear when `is_opposed=true`
- Monitor fill ratio (should move toward 1:1 vs previous 14:1 long bias)
- Consider adding drift urgency to quote logging for visibility
