# Churn Reduction Implementation - Session State

**Last Updated:** 2026-01-31
**Plan File:** `.claude/plans/fluffy-skipping-wren.md`

## Summary

Implementing a three-part first-principles redesign to reduce order churn from 94% to 20-40%.

## Completed Work

### Phase 1: OU Drift Model ✅
- **File:** `src/market_maker/process_models/hjb/ou_drift.rs` (NEW)
- Ornstein-Uhlenbeck mean-reverting drift with threshold gating
- Replaces EWMA for noise filtering
- Key structs: `OUDriftConfig`, `OUDriftEstimator`, `OUUpdateResult`, `OUDriftSummary`
- All 5 unit tests passing

### Phase 2: HJB Queue Value Integration ✅
- **File:** `src/market_maker/process_models/hjb/queue_value.rs` (NEW)
- Queue value formula: `v(q) = (s/2) × exp(-α×q) - β×q`
- Key structs: `HJBQueueValueConfig`, `HJBQueueValueCalculator`, `OrderQueueValue`
- All 5 unit tests passing

**Modified files:**
- `src/market_maker/process_models/hjb/config.rs` - Added OU and queue value config params
- `src/market_maker/process_models/hjb/controller.rs` - Integrated OU drift estimator
- `src/market_maker/process_models/hjb/skew.rs` - Added queue value methods
- `src/market_maker/tracking/queue/tracker.rs` - Added HJB queue value integration
- `src/market_maker/tracking/order_manager/reconcile.rs` - Wired queue value into priority matching

### Phase 3: Event-Driven Architecture (Partial) ✅
- **File:** `src/market_maker/events/quote_trigger.rs` (NEW)
- Event types: `QuoteUpdateEvent`, `ReconcileScope`, `QuoteUpdateTrigger`, `EventDrivenConfig`
- All 3 unit tests passing

- **File:** `src/market_maker/orchestrator/event_accumulator.rs` (NEW)
- Event accumulation and triggering logic
- Key structs: `AffectedTracker`, `EventAccumulator`, `EventAccumulatorStats`
- All 5 unit tests passing

**Module exports updated:**
- `src/market_maker/process_models/hjb/mod.rs`
- `src/market_maker/events/mod.rs`
- `src/market_maker/orchestrator/mod.rs`

## Remaining Work

### Task #9: Integrate event-driven updates in event loop (PENDING)
- Modify `src/market_maker/orchestrator/event_loop.rs`
- Replace timed polling with event-triggered updates
- Add partial reconciliation support

### Task #11: Fix failing tests (IN PROGRESS)
Two HJB tests failing due to OU drift warmup timing:
- `test_hjb_drift_warmup`
- `test_hjb_drift_adjusted_skew_uses_smoothed`

**Root cause:** When `use_ou_drift=true`, warmup requires 20 OU updates. In fast test loops, timestamps may be identical (same millisecond), causing OU to skip count increments.

**Fix needed in** `src/market_maker/process_models/hjb/ou_drift.rs` line 169-179:
```rust
// Current: skips update_count when dt_ms == 0
// Fix: increment update_count even when dt_ms == 0 for warmup purposes
if dt_ms == 0 {
    self.update_count += 1;  // ADD THIS
    if !self.warmed_up && self.update_count >= 20 {
        self.warmed_up = true;  // ADD THIS
    }
    return OUUpdateResult { ... };
}
```

## Compilation Status

- `cargo check` passes with warnings (dead code for unused event accumulator)
- 18 tests passing, 2 failing (the warmup tests mentioned above)

## Configuration Defaults

```rust
// HJB Queue Value (in HJBConfig)
use_queue_value: true,
queue_alpha: 0.1,
queue_beta: 0.02,
queue_modify_cost_bps: 3.0,

// OU Drift (in HJBConfig)
use_ou_drift: true,
ou_theta: 0.5,           // ~1.4s half-life
ou_reconcile_k: 2.0,     // 2σ threshold

// Event-Driven (in EventDrivenConfig)
enabled: true,
mid_move_threshold_bps: 5.0,
fill_immediate_trigger: true,
queue_depletion_p_fill: 0.05,
fallback_interval: Duration::from_secs(5),
max_pending_events: 10,
min_reconcile_interval: Duration::from_millis(100),
```

## Next Session Checklist

1. [ ] Fix OU drift warmup test issue (see fix above)
2. [ ] Run `cargo test --lib hjb` to verify fix
3. [ ] Complete Task #9: Wire event accumulator into event_loop.rs
4. [ ] Run full test suite: `cargo test --lib`
5. [ ] Paper trade test on testnet (30 min)

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/market_maker/process_models/hjb/ou_drift.rs` | OU drift estimator |
| `src/market_maker/process_models/hjb/queue_value.rs` | Queue value calculator |
| `src/market_maker/events/quote_trigger.rs` | Event types and config |
| `src/market_maker/orchestrator/event_accumulator.rs` | Event accumulation |
| `src/market_maker/orchestrator/event_loop.rs` | Main loop (needs integration) |
| `.claude/plans/fluffy-skipping-wren.md` | Full implementation plan |
