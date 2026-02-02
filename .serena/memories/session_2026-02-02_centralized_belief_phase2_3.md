# Session: Centralized Belief State Phases 2-3 Implementation

**Date**: 2026-02-02
**Task**: Implement Phases 2 and 3 of the centralized belief state architecture

## Summary

Implemented Phases 2 and 3 of the centralized belief state refactoring plan:
- Phase 2: Added `CentralBeliefState` field to `MarketMaker`
- Phase 3: Added dual-write paths for all belief updates

## Changes Made

### Phase 2: Add CentralBeliefState to MarketMaker

**File: `src/market_maker/mod.rs`**

1. Added import:
```rust
use belief::{CentralBeliefConfig, CentralBeliefState};
```

2. Added field to `MarketMaker<S, E>` struct:
```rust
// === Centralized Belief State (Phase 2) ===
/// Single source of truth for all Bayesian beliefs.
/// Replaces fragmented belief modules with unified state management.
/// Consumers read via snapshot() for consistent point-in-time views.
central_beliefs: CentralBeliefState,
```

3. Initialized in `new()`:
```rust
central_beliefs: CentralBeliefState::new(CentralBeliefConfig::default()),
```

4. Added accessor methods:
```rust
pub fn central_beliefs(&self) -> &CentralBeliefState
pub fn central_beliefs_mut(&mut self) -> &mut CentralBeliefState
```

### Phase 3: Dual-Write Updates

**File: `src/market_maker/orchestrator/quote_engine.rs`**

Added imports:
```rust
use crate::market_maker::belief::BeliefUpdate;
```

Added dual-writes for:
1. **Price returns** (after `beliefs_builder.observe_price()`):
```rust
self.central_beliefs.update(BeliefUpdate::PriceReturn {
    return_frac: price_return,
    dt_secs: dt,
    timestamp_ms,
});
```

2. **Regime updates** (after HMM regime_probs read):
```rust
self.central_beliefs.update(BeliefUpdate::RegimeUpdate {
    probs: regime_probs,
    features: None,
});
```

3. **Changepoint observations** (after regime update):
```rust
self.central_beliefs.update(BeliefUpdate::ChangepointObs {
    observation: market_params.sigma,
});
```

**File: `src/market_maker/orchestrator/handlers.rs`**

Added imports:
```rust
use super::super::belief::BeliefUpdate;
```

Added dual-writes for:

4. **Own fills** (in `handle_user_fills`, after `calibration_controller.record_fill()`):
```rust
self.central_beliefs.update(BeliefUpdate::OwnFill {
    price: fill_price,
    size: fill_size,
    mid: self.latest_mid,
    is_buy,
    is_aligned,
    realized_as_bps,
    realized_edge_bps,
    timestamp_ms,
    order_id: Some(fill.oid),
});
```

5. **Market trades** (in `handle_trades`, inside trade caching loop):
```rust
self.central_beliefs.update(BeliefUpdate::MarketTrade {
    price: trade_price,
    mid: self.latest_mid,
    timestamp_ms,
});
```

## Verification

- `cargo check --lib` passes
- `cargo test belief --lib` - 54 tests pass
- `cargo test --lib` - 1737 tests pass

## Next Steps (Phases 4-8)

- **Phase 4**: Migrate QuoteEngine reads from scattered sources to `central_beliefs.snapshot()`
- **Phase 5**: Migrate ParameterAggregator to use `build_from_beliefs()`
- **Phase 6**: Migrate QuoteGate to use BeliefSnapshot
- **Phase 7**: Remove old dual-write paths (HIGH RISK)
- **Phase 8**: Cleanup deprecated code (HIGH RISK)

## Architecture Notes

The centralized belief state receives updates from:
- Quote engine: price returns, regime, changepoint
- Fill handler: own fills
- Trade handler: market trades

Consumers will migrate to reading via `central_beliefs.snapshot()` which provides
a point-in-time consistent view of all beliefs.
