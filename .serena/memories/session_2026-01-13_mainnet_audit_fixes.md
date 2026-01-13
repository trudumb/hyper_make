# Session: 2026-01-13 Mainnet BTC Log Audit and Fixes

## Summary
Audited mainnet BTC market maker log (`mm_mainnet_BTC_2026-01-13_15-13-53.log`) and systematically fixed 4 identified issues: microprice divergence during warmup, untracked fills (67% failure rate), rate limit pressure, and ladder level configuration.

## Issues Identified from Log Audit

### 1. Microprice Divergence (~200 bps during warmup) - CRITICAL
- **Root Cause**: EMA initialization captured outlier values when beta coefficients had high variance during early warmup
- **Symptom**: 10 "CRITICAL: Bid crossed market mid" errors in logs

### 2. Untracked Fills (18/27 = 67% failure rate) - CRITICAL
- **Root Cause**: CLOID was not being passed from UserFills WebSocket message to FillEvent
- **Symptom**: 18 "Untracked order filled" warnings

### 3. Rate Limit Pressure (1.8-4.8% headroom) - MEDIUM
- **Root Cause**: Early session has only 10K base budget; rate limit check was tied to impulse_enabled flag
- **Symptom**: Low headroom warnings in logs

### 4. Ladder Level Reduction (25→17/13) - WORKING AS DESIGNED
- **Assessment**: System correctly reduces levels when margin is insufficient - not a bug

## Changes Made

### Issue 1: Microprice Divergence Fix
**Files Modified:**
- `src/market_maker/estimator/microprice.rs` (lines 609-656)
  - Changed EMA initialization to use `mid` instead of `raw_microprice`
  - Added divergence reset when EMA deviates >100 bps from mid
  - Added warning log when reset occurs

- `src/market_maker/strategy/market_params.rs` (line 1273)
  - Removed `.max(mid)` asymmetric bias
  - Changed from: `microprice: estimator.microprice().max(mid)`
  - Changed to: `microprice: estimator.microprice()`

### Issue 2: Untracked Fills Fix
**Files Modified:**
- `src/market_maker/messages/user_fills.rs` (lines 96-106)
  - Changed `FillEvent::new()` to `FillEvent::with_cloid()`
  - Passes `fill.cloid.clone()` from WebSocket message to FillEvent
  - Enables deterministic CLOID-first lookup in fill processor

### Issue 3: Rate Limit Handling Fix
**Files Modified:**
- `src/market_maker/orchestrator/reconcile.rs` (lines 556-604)
  - Added unconditional rate limit check at start of `reconcile_ladder_smart()`
  - Refreshes rate limit cache every 60 seconds (regardless of impulse_enabled)
  - Throttles reconciliation when headroom < 10% (buffer before 5% hard limit)
  - Uses `proactive_rate_tracker.can_modify()` for timing check

## Files Summary

| File | Change Type | Lines |
|------|-------------|-------|
| `src/market_maker/estimator/microprice.rs` | EMA init fix, divergence reset | 609-656 |
| `src/market_maker/strategy/market_params.rs` | Remove .max(mid) bias | 1273 |
| `src/market_maker/messages/user_fills.rs` | Pass CLOID to FillEvent | 96-106 |
| `src/market_maker/orchestrator/reconcile.rs` | Unconditional rate limit check | 556-604 |

## Verification
- ✅ Build succeeded
- ✅ 823 library tests passed
- ✅ No new clippy warnings from changes (existing warnings are format string style)

## Technical Details

### Microprice EMA Fix Logic
```rust
// BEFORE: First observation initialized EMA with potentially bad value
if prev_bits == EMA_NONE {
    self.ema_microprice_bits.store(raw_microprice.to_bits(), Ordering::Relaxed);
}

// AFTER: Initialize EMA with mid (stable reference), add divergence reset
if prev_bits == EMA_NONE {
    self.ema_microprice_bits.store(mid.to_bits(), Ordering::Relaxed);
    raw_microprice
} else {
    let prev = f64::from_bits(prev_bits);
    let ema_divergence_bps = ((prev - mid) / mid).abs() * 10_000.0;
    if ema_divergence_bps > 100.0 {
        // Reset EMA when too far from mid
        self.ema_microprice_bits.store(mid.to_bits(), Ordering::Relaxed);
        return raw_microprice;
    }
    // ... normal EMA update
}
```

### CLOID Fill Tracking Fix
```rust
// BEFORE: Created FillEvent without CLOID
let fill_event = FillEvent::new(fill.tid, fill.oid, amount, ...);

// AFTER: Pass CLOID for deterministic tracking
let fill_event = FillEvent::with_cloid(
    fill.tid,
    fill.oid,
    fill.cloid.clone(),  // Extract from WS message
    amount,
    ...
);
```

### Rate Limit Check Addition
Added at start of `reconcile_ladder_smart()`:
- Cache refresh every 60 seconds via `user_rate_limit()` API
- Throttle when `headroom < 10%` AND `!can_modify()` (timing check)
- Runs regardless of `impulse_enabled` flag

## Next Steps
- Monitor mainnet session for:
  - Zero "Bid crossed market mid" errors (Issue 1 fix)
  - Reduced "Untracked order filled" warnings (Issue 2 fix)
  - Improved rate limit headroom over time (Issue 3 fix)
- Consider reducing `num_levels` config from 25 to ~15 for typical margin levels (Issue 4)

## Session Stats
- Log analyzed: 6.7 MB, ~18 minutes of trading
- Fills: 27 total (9 tracked before fix, 18 untracked)
- Critical errors: 10 (microprice-related)
- All issues addressed with first-principles fixes
