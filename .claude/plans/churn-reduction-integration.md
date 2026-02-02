# Churn Reduction Integration Plan

## Status Summary

### ✅ Completed

1. **OU Drift Model** (`src/market_maker/process_models/hjb/ou_drift.rs`)
   - Kalman filter-style updates with threshold gating
   - Integrated into HJB controller via `use_ou_drift` config flag
   - Called in `update_momentum_signals()`

2. **HJB Queue Value** (`src/market_maker/tracking/queue/tracker.rs`)
   - Queue value formula: `v(q) = (s/2) × exp(-α×q) - β×q`
   - `should_preserve_by_hjb_value()` method implemented
   - Used in `priority_based_matching()` in reconcile.rs

3. **Event Types** (`src/market_maker/events/quote_trigger.rs`)
   - `QuoteUpdateEvent` enum with all event types
   - `ReconcileScope` enum
   - `EventDrivenConfig` with defaults

4. **Event Accumulator** (`src/market_maker/orchestrator/event_accumulator.rs`)
   - All event accumulation logic implemented
   - Priority handling, debouncing, fallback timer
   - ✅ Dead code warnings fixed with `#[allow(dead_code)]` annotations

5. **EventAccumulator Field** (`src/market_maker/mod.rs`)
   - ✅ Added `event_accumulator` field to MarketMaker struct
   - ✅ Initialized with `EventAccumulator::default_config()`

6. **Event Handler Wiring** (`src/market_maker/orchestrator/handlers.rs`)
   - ✅ `handle_all_mids()` now calls `event_accumulator.on_mid_update()` when enabled
   - ✅ `handle_user_fills()` now calls `event_accumulator.on_fill()` when enabled
   - ✅ Added `check_event_accumulator()` method for main loop integration
   - Graceful fallback to timed polling when event-driven mode is disabled

7. **Main Event Loop Integration** (`src/market_maker/orchestrator/event_loop.rs`)
   - ✅ Added call to `check_event_accumulator()` after each message is processed
   - Event-driven reconciliation now checks `should_trigger()` and `check_fallback()`

### 🔄 Optional Enhancements (Future Work)

1. **Partial Reconciliation** (`ReconcileScope` support)
   - Currently all triggers use full reconciliation
   - Could add `partial_update_quotes(scope)` for side-only or levels-only reconciliation
   - Low priority - full reconciliation works fine, just less efficient

2. **Queue Depletion Events**
   - `on_queue_depletion()` is implemented but not wired
   - Would need integration with queue tracker updates
   - Low priority - fills already trigger high-priority events

3. **Signal Change Events**
   - `on_signal_change()` is implemented but not wired
   - Would need integration with signal/estimator updates
   - Low priority - mid price moves already capture most signal changes

4. **Volatility Spike Events**
   - `on_volatility_update()` is implemented but not wired
   - Would need integration with sigma estimator
   - Low priority - mid price moves during volatility spikes trigger anyway

## Configuration Defaults

```rust
// EventDrivenConfig (in src/market_maker/events/quote_trigger.rs)
EventDrivenConfig {
    enabled: true,                              // Event-driven mode enabled
    mid_move_threshold_bps: 5.0,               // 5 bps price move triggers
    fill_immediate_trigger: true,               // Fills always trigger
    queue_depletion_p_fill: 0.05,              // P(fill) < 5% triggers
    signal_change_threshold: 0.3,               // 30% signal change triggers
    volatility_spike_ratio: 1.5,               // 50% vol change triggers
    fallback_interval: Duration::from_secs(5), // 5s fallback timer
    min_reconcile_interval: Duration::from_millis(100), // 100ms debounce
    max_pending_events: 10,                    // Max events before forced trigger
}

// HJB Config (already in place)
use_ou_drift: true,
ou_theta: 0.5,       // ~1.4s half-life
ou_reconcile_k: 2.0, // 2σ threshold

// HJB Queue Value (already in place)
use_hjb_queue_value: true,
hjb_queue_alpha: 0.1,
hjb_queue_beta: 0.02,
hjb_queue_modify_cost_bps: 3.0,
```

## How It Works

### Flow Overview

```
Message Arrives
       │
       ▼
┌──────────────────┐
│ handle_message() │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────┐
│ handler records event to       │
│ EventAccumulator               │
│ - on_mid_update() for AllMids  │
│ - on_fill() for UserFills      │
└────────┬───────────────────────┘
         │
         ▼
┌───────────────────────────────────┐
│ check_event_accumulator()         │
│ - Checks should_trigger()         │
│ - Checks check_fallback()         │
│ - Calls update_quotes() if needed │
│ - Resets accumulator after        │
└───────────────────────────────────┘
```

### Event Triggering Logic

1. **Mid Price Move > 5 bps**: Immediate trigger (priority 70)
2. **Fill Received**: Immediate trigger (priority 90)
3. **High Priority Event (≥80)**: Immediate trigger
4. **Accumulated Price Move > threshold**: Trigger
5. **Max Pending Events (10)**: Trigger
6. **Fallback Timer (5s)**: Trigger if any events pending

### Debouncing

- Minimum 100ms between reconciliations
- Small price moves (<5 bps) accumulate rather than trigger
- Events are buffered and merged for efficient reconciliation

## Expected Impact

| Metric | Before | After (Est.) |
|--------|--------|--------------|
| Churn ratio | 94% | 20-40% |
| Quote cycles/sec | ~1.0 | 0.1-0.3 |
| Queue preservation | Poor | Good |
| Fill rate | Baseline | +15-25% |
| API budget usage | High | -50-70% |

## Verification

1. **Build**: ✅ `cargo build` passes
2. **Tests**: ✅ `cargo test --lib` - all 1593 tests pass
3. **Paper Trade**: Monitor these log messages:
   - `"Event accumulator: triggering quote update"` - shows when reconciliation happens
   - `"Event accumulator stats (churn reduction)"` - shows filter ratio every 100 reconciles
   - `"Event accumulator: fallback timer triggered"` - shows fallback activations
   - `"HJB queue value: preserving order"` - shows queue preservation working
   - `"OU drift reconciled: false"` - shows noise filtering

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `orchestrator/mod.rs` | Fix unused imports | ✅ Done |
| `mod.rs` | Add EventAccumulator field | ✅ Done |
| `orchestrator/event_accumulator.rs` | Add dead_code annotations | ✅ Done |
| `orchestrator/handlers.rs` | Wire event recording + check method | ✅ Done |
| `orchestrator/event_loop.rs` | Call check_event_accumulator() | ✅ Done |
| `process_models/hjb/mod.rs` | Fix OU drift tests | ✅ Done |
| `estimator/enhanced_flow.rs` | Fix unused variable | ✅ Done |
