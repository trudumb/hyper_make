# Anti-Accumulation V2 — Live Mainnet Bug Fixes (2026-02-12)

## Context
V1 deployed on mainnet for ~30 min. Cascade tracker fired 25 times, 55 fills, BBO cap warning 153 times.
Two critical bugs identified from live logs.

## Bugs Fixed

### BUG 1: Cascade WIDEN widens REDUCING side (P0)
**File**: `orchestrator/quote_engine.rs:2192, 2214`
**Root cause**: WIDEN had no position check. When long (+3.28), sell WIDEN fired → widened sells → harder to reduce long (WRONG).
**Fix**: Added `&& self.position.position() >= 0.0` on buy WIDEN, `<= 0.0` on sell WIDEN. Now WIDEN only fires on accumulating side.

### BUG 2: BBO half-spread cap kills inventory skew (P0)  
**File**: `quoting/ladder/generator.rs:986-991`
**Root cause**: Cap at `80% of exchange BBO half-spread`. HYPE BBO is 0.16-0.65 bps → max skew 0.13-0.52 bps. But GLFT computes 4+ bps. 97% of skew deleted.
**Fix**: `(half_spread * 0.8).max(own_floor)` where `own_floor = 50% of innermost ladder level depth`. No new parameters needed.

### ISSUE 3: record_fill() counts reducing fills (P1)
**File**: `mod.rs` FillCascadeTracker
**Root cause**: Selling out of a long (healthy!) counted toward sell cascade → Bug 1 made it counterproductive.
**Fix**: `record_fill(side, position)` only counts accumulating fills (buying when long/flat, selling when short/flat).

### ISSUE 4: Longer cooldowns + cancel on cascade trigger (P1)
**Files**: `mod.rs`, `orchestrator/handlers.rs`
**Root cause**: 15s widen cooldown too short, pre-cascade orders filled during cascade (cancel-fill race).
**Fix**: Cooldowns 15→30s (widen), 30→60s (suppress). New `CascadeEvent` enum returned on threshold crossing. Handler cancels resting accumulating-side orders immediately.

## V2 vs V1 Live Comparison (first 5 min)

| Metric | V1 (old session) | V2 (new session) |
|--------|-------------------|-------------------|
| BBO cap warnings | 153 | 1 |
| Cascade events | 25 | 2 |
| Sharpe | 384 | 743 |
| Edge bps | 2.3 | 4.6 |
| Fills | 55 (30 min) | 11 (5 min) |
| Position control | -5.04 to +3.28 | -5.04 → +0.87 (reduced inherited short) |

## Test Results
Tests: 2,411 → 2,418 (+7). Clippy clean.
- 6 FillCascadeTracker unit tests (reducing not counted, accumulating triggers, suppress, mixed, cooldowns, widen logic)
- 1 BBO cap test (4 bps skew survives with tight 0.3 bps exchange BBO)

## Key Files Modified
- `orchestrator/quote_engine.rs` — Position check on WIDEN
- `quoting/ladder/generator.rs` — BBO cap floor using own ladder depth
- `mod.rs` (FillCascadeTracker) — Position-aware recording, cooldowns, CascadeEvent return type
- `orchestrator/handlers.rs` — Pass position, cancel on cascade trigger, import CascadeEvent

## Known Issues
- Kill switch checkpoint persistence blocks restart with inherited over-limit position. Must clear checkpoint manually.
- Position started at -5.04 from prior session, needed `--max-position-usd 200` override.
