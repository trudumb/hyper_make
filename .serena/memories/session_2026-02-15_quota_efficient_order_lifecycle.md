# Session 2026-02-15: Quota-Efficient Order Lifecycle Fix

## Problem
Market maker fundamentally unprofitable due to quota death spiral:
- Low API quota headroom (9%) → `wide_two_sided_decision()` applied `1/sqrt(headroom)` = 3.34x spread widening
- Wider spreads → uncompetitive quotes → few fills → no quota recovery → more widening
- Position accumulated (6 consecutive same-side fills) → InventoryGovernor widened further
- 80 orders placed, 72 cancelled (90%), $0.01 PnL in 8.5 min session

## Root Causes Fixed (6)
1. **Quota death spiral** — `wide_two_sided_decision()` widened spreads by `1/sqrt(headroom)`
2. **Dead shadow pricing** — `quota_shadow_spread_bps` computed but never consumed
3. **Multiplicative spread stacking** — AS(1.5)×quota(5)×risk(3)×cascade(5) = 112x worst case
4. **FillCascadeTracker position gate** — only counted "accumulating" fills
5. **Kill switch capital mismatch** — $500 max_daily_loss on $100 capital
6. **No cross-venue safety** — no protective mode when Binance feed absent

## Implementation (6 Phases, 5 agents + lead)

### Phase 1a: Neuter wide_two_sided_decision (quote_gate.rs)
- Agent: strategy-quote-gate
- Function returns `None` immediately, 6 tests updated

### Phase 1b/1c: Cycle throttling + cap multipliers (quote_engine.rs, mod.rs)
- Lead implementation
- Quota-aware cycle intervals: >=30% headroom→1s, 10-30%→3s, 5-10%→5s, <5%→10s
- `WidenSpreads` multiplier capped at 2.0 (was uncapped)
- `NoQuote` widening capped at 2.0 (was doubling each time)
- Wired `signal_position_limit_mult` from no-signal safety mode

### Phase 2: Additive spread composition (glft.rs)
- Agent: strategy-glft
- Old: `half_spread * spread_widening_mult * bandit_mult` (multiplicative chain)
- New: `half_spread * bandit_mult + widening_addon + quota_addon` (additive)
- Bandit clamped [0.8, 1.5], widening excess capped at 1.0 (doubles at most)
- Quota addon = `quota_shadow_spread_bps / 10000` (capped at 50 bps)
- Worst case: ~3.5x base (was 10x+)
- No hard cap on total spread (was causing test failures at 50 bps)
- New test: `test_additive_spread_composition_capped`

### Phase 3: Quota-aware reconciliation (reconcile.rs, mod.rs)
- Lead implementation
- Latch threshold widens at <30% headroom: `(base*2.0).clamp(3.0, 15.0)` (was `base.clamp(1.0, 10.0)`)
- 2-second empty ladder recovery cooldown via `last_empty_ladder_recovery: Option<Instant>`

### Phase 4: FillCascadeTracker fix (mod.rs)
- Agent: general-cascade
- Removed `is_accumulating` gate — counts ALL same-side fills
- Thresholds lowered: widen=2 (was 3), suppress=4 (was 5)
- Module-level constants: `CASCADE_WIDEN_THRESHOLD`, `CASCADE_SUPPRESS_THRESHOLD`
- 6 tests updated

### Phase 5: Capital-proportional kill switch (kill_switch.rs)
- Agent: risk-killswitch
- `KillSwitchConfig::for_capital(account_value)`:
  - max_daily_loss = 5% of capital (min $1)
  - max_absolute_drawdown = 3% (min $0.50)
  - min_peak_for_drawdown = 0.5% (min $0.50)
  - max_position_value = 3x capital
- `validate_for_capital()` logs warn at >20%, error at >100%
- 4 new tests

### Phase 6: No-signal safety mode (signal_integration.rs)
- Agent: signals-safety
- Uses `Cell<bool>` for interior mutability (get_signals takes `&self`)
- Tracks cross-venue availability via `has_had_cross_venue` flag
- `signal_position_limit_mult()` returns 0.30 when no cross-venue signal
- Wired into quote_engine.rs via `effective_max_position *= signal_limit_mult`
- 2 new tests

## Validation
- clippy clean (fixed `manual_clamp` lint, unused variable)
- 2616 tests pass, 4 pre-existing failures unchanged
- Plan file: `.claude/plans/streamed-wiggling-kettle.md`

## Key Debugging: GLFT Test Failures
- Phase 2 initially caused 6 test failures
- Root cause: 50 bps hard cap (`.min(0.0050)`) on additive composition
- Base GLFT spread was 100+ bps with test params (kappa_bid defaults to 100)
- Cap clipped both toxic and non-toxic cases equally → 0 bps difference
- Fix: removed per-side hard cap, each addon bounded individually instead

## Files Changed
| File | Agent/Lead |
|------|-----------|
| control/quote_gate.rs | strategy-quote-gate |
| orchestrator/quote_engine.rs | lead |
| strategy/glft.rs | strategy-glft |
| orchestrator/reconcile.rs | lead |
| mod.rs (FillCascadeTracker + fields) | general-cascade + lead |
| risk/kill_switch.rs | risk-killswitch |
| strategy/signal_integration.rs | signals-safety |
