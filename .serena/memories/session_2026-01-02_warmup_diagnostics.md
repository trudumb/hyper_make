# Session: 2026-01-02 Warmup Diagnostics for HIP-3

## Problem Analyzed
**Log:** `logs/mm_hyna_BTC_2026-01-02_22-32-20.log.console`
**Issue:** Market maker running 120+ seconds with `orders_placed=0` despite continuous quote cycles

## Root Cause
The estimator warmup gate in `mod.rs:1012` blocks ALL order placement until:
- Volume ticks >= 20 (from `VolumeBucketAccumulator`)
- Trade observations >= 50 (from `market_kappa`)

HIP-3 DEX markets have low volume, so these thresholds are never reached.

The `warmup_pct=10%` in logs is from adaptive spreads module (different system) - NOT the blocking gate.

## Fix Implemented
Added diagnostic logging when warmup blocks order placement:

**File: `src/market_maker/mod.rs`**
- Added `last_warmup_block_log: Option<std::time::Instant>` field (line 100)
- Added initialization `last_warmup_block_log: None` (line 228)
- Added WARN log every 10 seconds in `update_quotes()` (lines 1013-1029):
  ```
  Warmup incomplete - no orders placed (waiting for market data)
  volume_ticks=X, volume_ticks_required=20,
  trade_observations=Y, trade_observations_required=50
  ```

## Verification
- `cargo check` - passes
- `cargo clippy -- -D warnings` - passes

## New Finding: Size Optimization Issue

The warmup check IS passing (defaults are 10 ticks / 5 observations). The issue is:
1. Ladder diagnostics log (so generate_ladder is called)
2. But `orders_placed=0` means ladder comes back empty or gets filtered

Root cause: Line 740-741 in ladder_strat.rs filters out levels where `size <= EPSILON`.
The constrained optimizer may be returning zero-size levels.

Need to add diagnostics to show:
- Available margin
- Available position capacity  
- Optimizer allocation results
- Why sizes might be zero

## Next Steps (Future Session)
Implement Option 1: Relaxed warmup for HIP-3 DEX
- Reduce `min_volume_ticks` from 20 to 5
- Reduce `min_trade_observations` from 50 to 10
- Auto-detect HIP-3 mode and apply relaxed settings

## Test Command
```bash
./scripts/test_hip3.sh BTC hyna 300
```

## Key Files
- `src/market_maker/mod.rs` - Warmup gate and new diagnostics
- `src/market_maker/estimator/parameter_estimator.rs` - `warmup_progress()` method (already existed)
- `src/market_maker/messages/trades.rs` - Existing warmup logging (only triggers on trades)
