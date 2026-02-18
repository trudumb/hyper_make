# Session 2026-02-18: Post-Implementation Audit Remediation

## Context
Three code-reviewer agents audited all Sprint 1-6 changes (27 tasks, ~800 lines added).
Found 2 FAIL and 6 WARN issues. This session fixed 7 of 8 (W5 skipped — already correct).

## Tests: 3006 passed, 0 failed. Clippy clean.

## Fixes Applied

### F1: Dead `total_spread_mult` mutations → additive risk premium (CRITICAL)
**Files**: `quote_engine.rs`, `signal_integration.rs`
**Problem**: Three sprint features mutated `signals.total_spread_mult` (OI vol, funding settlement, cancel-race) but this field was NEVER consumed by the spread pipeline. All three features were dead code.
**Fix**: 
- Added `signal_risk_premium_bps: f64` field to `IntegratedSignals` (derives Default → 0.0)
- Replaced 3 multiplicative `*=` mutations with additive bps:
  - OI vol: `oi_excess * 3.0` bps (was `1.0 + oi_excess * 0.3` mult)
  - Funding settlement: `settlement_proximity * 1.5` bps (was `1.0 + proximity * 0.15` mult)
  - Cancel-race: `cancel_race_excess.min(5.0)` bps (was `1.0 + excess/50` mult)
- Added `total_risk_premium += signals.signal_risk_premium_bps` in Phase 5 accumulation block
- Pipeline: `signal_risk_premium_bps` → `total_risk_premium_bps` → `solve_min_gamma()` → GLFT spread

### F2: Cancel-race tracker blind to 90% of cancels
**File**: `order_ops.rs`
**Problem**: `record_cancel_request()` only in `initiate_bulk_cancel()`, not `initiate_and_track_cancel()`.
**Fix**: Added `cancel_race_tracker.record_cancel_request(oid, cancel_ts_ms)` before `initiate_cancel()` in single-order path.

### W1: Kelly applied twice (compound reduction)
**File**: `quote_engine.rs`
**Problem**: Kelly multiplied both `effective_max_position` (line ~1540) and `kelly_adjusted_liquidity` (line ~1939).
**Fix**: Removed Kelly from liquidity path. `kelly_adjusted_liquidity = decision_adjusted_liquidity` directly.

### W2: Emergency bypass unreachable for StaleCancel
**File**: `budget_allocator.rs`
**Problem**: `is_emergency()` required `value_bps > 10.0` but StaleCancel value is always negative (`-ev_keep - api_cost`).
**Fix**: `is_emergency()` now triggers on ALL StaleCancel actions (removed value threshold).

### W3: BayesianEstimate variance floor
**File**: `learning/cross_asset.rs`
**Fix**: Added `.max(1e-10)` after precision update to prevent zero-variance.

### W4: BTC self-referential lead-lag
**File**: `mod.rs`
**Fix**: Conditional: `for_btc()` when asset contains "BTC", `for_altcoin()` otherwise.

### W6: Regime-dependent cross-asset skew clamp
**File**: `quote_engine.rs`
**Fix**: Calm/Normal: ±3 bps, Volatile: ±5 bps, Extreme: ±6 bps (was hard ±3.0).

### W5: Cancel-on-toxicity race window (SKIPPED)
**Analysis**: Toxicity cancel clears ladder → reconcile runs in same `update_quotes()` call → exchange cancel fires immediately. Race window is within single function, not across cycles. No fix needed.

## Key Lessons
1. **Multiplicative spread fields logged but unconsumed**: Always trace field from mutation to consumption. `total_spread_mult` was analytics-only, not pipeline input.
2. **StaleCancel EV is always negative**: `value = -ev_keep - api_cost`. Any positive threshold on StaleCancel value is unreachable.
3. **Kelly dedup**: Capital deployment sizing (position limits) and order sizing (liquidity) are separate concerns. Apply Kelly to one, not both.
4. **Variance collapse**: Conjugate normal updates converge precision to infinity. Floor prevents downstream division-by-zero.

## Architecture Note
The `total_risk_premium_bps` pipeline is the correct way to add spread widening:
```
quote_engine.rs accumulates → market_params.total_risk_premium_bps → glft.rs:solve_min_gamma() → spread
```
Contributions: regime (base), hawkes sync (0-3 bps), toxicity (0-5 bps), staleness (0-3 bps), signal-derived (OI/funding/cancel-race, 0-~10 bps).
