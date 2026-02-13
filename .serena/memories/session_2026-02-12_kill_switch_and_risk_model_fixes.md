# Session 2026-02-12 Evening: Kill Switch Daily Reset + Risk Model Blend Fix

## Context
After the Feb 12 live audit (6 systemic bugs found), attempted to restart mainnet HYPE trading.
The MM immediately self-killed on startup because the checkpoint carried forward yesterday's -$5.02 daily loss.

## Bug 1: Kill Switch Checkpoint Carries Stale Daily Loss Across Days

### Root Cause
`restore_from_checkpoint()` in `risk/kill_switch.rs` unconditionally copied `daily_pnl` and `peak_pnl` 
from the previous session's checkpoint. No concept of "trading day" — a new session on a new UTC day 
would inherit yesterday's cumulative loss and immediately trigger the daily loss limit.

### Fix (3 files)
1. **`checkpoint/types.rs`**: Added `saved_at_ms: u64` field to `KillSwitchCheckpoint` 
   - `#[serde(default)]` for backward compat (old checkpoints get 0 → treated as different day → safe reset)
2. **`risk/kill_switch.rs` — `to_checkpoint()`**: Now records `saved_at_ms` timestamp
3. **`risk/kill_switch.rs` — `restore_from_checkpoint()`**: 
   - Checks if checkpoint was saved on same UTC day via `is_same_utc_day()` helper
   - Different day → resets daily_pnl and peak_pnl to 0
   - Filters daily-scoped trigger reasons ("daily loss", "drawdown") on new day
   - Manual shutdown and cascade reasons still persist within 24h (as before)
   
### Helper functions added
- `is_same_utc_day(ts1_ms, ts2_ms)` — compares UTC day numbers
- `is_daily_scoped_reason(reason)` — matches "daily loss" or "drawdown" strings
- `const MS_PER_DAY: u64 = 24 * 60 * 60 * 1000`

### Tests added (7 new + 1 updated)
- `test_kill_switch_new_day_resets_daily_pnl` — core case
- `test_kill_switch_same_day_preserves_daily_pnl` — same-day still works
- `test_kill_switch_new_day_non_daily_reason_persists_within_24h` — manual shutdown survives
- `test_kill_switch_new_day_daily_loss_dropped_but_manual_kept` — mixed reasons filtered correctly
- `test_kill_switch_new_day_drawdown_not_restored` — drawdown is daily-scoped
- `test_kill_switch_old_checkpoint_without_saved_at_resets_pnl` — backward compat
- `test_is_same_utc_day` — helper validation
- `test_is_daily_scoped_reason` — helper validation
- Updated `test_kill_switch_restore_expired_trigger_ignored` — P&L now correctly resets for 25h-old checkpoint

## Bug 2: risk_model_blend=0.0 for Default Spread Profile (Multiplicative Gamma)

### Root Cause
In `src/bin/market_maker.rs` line ~1831, the strategy construction used:
```rust
SpreadProfile::Default => RiskModelConfig::default()
```
`RiskModelConfig::default()` has `risk_model_blend: 0.0` (legacy multiplicative).
Only `Hip3` and `Aggressive` profiles got `risk_model_blend: 1.0`.

The `StochasticConfig::default()` correctly sets `risk_model_blend: 1.0`, but this value was 
never wired through to `RiskModelConfig` — it was ignored entirely.

### Fix
Changed `SpreadProfile::Default` branch to:
```rust
SpreadProfile::Default => RiskModelConfig {
    use_calibrated_risk_model: true,
    risk_model_blend: 1.0,
    ..Default::default()
},
```

### Impact
- Prevents multiplicative gamma explosion (1.2^7 = 3.6x) that was happening on all non-HIP3 runs
- The Feb 9 fix (commit c622f6f) was only partially effective — it fixed the config default but not the wiring

## Validation
- All 46 kill switch tests pass
- Clippy clean (zero warnings)
- Mainnet HYPE session started successfully at 02:37 UTC
- Log confirms: `risk_model_blend: 1.0`, `daily P&L reset to zero`, `no reasons survive — not re-triggering`
- First quote cycle: 23 bps total spread, 4 bid + 4 ask levels, position flat

## Config Observations (market_maker_live.toml)
- `max_daily_loss = $5` on $100 capital (5%) — tight, one bad sequence triggers
- `inventory_skew_factor = 0.0` — no inventory skew at all
- `use_calibrated_as = false` — AS model disabled (Phase 1)
- Checkpoint had all-zero estimator observations — the 2s self-kill session saved an empty checkpoint
- Readiness: "Insufficient" — fresh run starts with cold priors

## Files Changed
- `src/market_maker/checkpoint/types.rs` — KillSwitchCheckpoint struct + Default
- `src/market_maker/risk/kill_switch.rs` — to_checkpoint, restore_from_checkpoint, 2 helpers, 8 tests
- `src/bin/market_maker.rs` — risk_model_cfg construction for Default profile
