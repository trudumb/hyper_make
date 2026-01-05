# Session: 2026-01-04 Queue Position & Rate Limit Optimization

## Summary
Implemented comprehensive queue position and rate limit optimization to reduce order churn by ~50%. Added configurable reconciliation tolerances, queue-aware reconciliation, and microprice EMA smoothing.

## Problem Statement
Log analysis revealed excessive order churn:
- 205 orders placed, 97 cancelled in 60 seconds
- Every modify returns new OID (Hyperliquid price mods reset queue)
- Spread volatility 88-194 bps causing constant updates
- Rate limit exhaustion risk

**Critical Discovery**: On Hyperliquid, only SIZE-only modifications preserve queue position. Price changes always return new OID.

## Changes Made

### P0: Increased Reconciliation Tolerances (Low Risk)
**File**: `src/market_maker/tracking/order_manager/reconcile.rs:34-43`

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `skip_price_tolerance_bps` | 1 | 10 | Reduce churn 40-60% |
| `max_modify_price_bps` | 10 | 50 | No queue benefit anyway |

### P1: Made ReconcileConfig Configurable
- Added `ReconcileConfig` field to `MarketMakerConfig` (`config.rs:180`)
- Added CLI arguments (`bin/market_maker.rs:128-155`):
  - `--skip-price-tolerance-bps`
  - `--max-modify-price-bps`
  - `--skip-size-tolerance-pct`
  - `--max-modify-size-pct`
  - `--use-queue-aware`
  - `--queue-horizon-seconds`
- Wired config through in `mod.rs:2187`
- Exported `ReconcileConfig` from `lib.rs`

### P2: Queue-Aware Reconciliation
**File**: `src/market_maker/mod.rs:2218-2300`

Added integration with `QueuePositionTracker`:
- New fields in `ReconcileConfig`:
  - `use_queue_aware: bool` (default: false)
  - `queue_horizon_seconds: f64` (default: 1.0)
- After standard reconciliation, checks "skipped" orders
- If `queue_tracker.should_refresh(oid, horizon)` returns true, forces CANCEL+PLACE
- Refreshes orders with low fill probability even if price is within tolerance

### P3: Microprice EMA Smoothing
**File**: `src/market_maker/estimator/microprice.rs`

Added EMA smoothing to reduce microprice volatility:
- New fields (lines 130-138):
  - `ema_alpha: f64` (smoothing factor, default 0.2)
  - `ema_microprice_bits: AtomicU64` (thread-safe storage)
  - `ema_min_change_bps: f64` (noise filter, default 2.0)
- Modified `microprice()` method (lines 609-638):
  - Applies EMA: `smoothed = alpha * raw + (1-alpha) * prev`
  - Noise filter: skips update if change < min_change_bps
  - Uses `AtomicU64` for `Sync` compatibility
- Added configuration to `StochasticConfig` (config.rs:584-596):
  - `microprice_ema_alpha`
  - `microprice_ema_min_change_bps`

## Files Modified

| File | Changes |
|------|---------|
| `tracking/order_manager/reconcile.rs` | P0 defaults, P2 queue-aware fields |
| `config.rs` | P1 ReconcileConfig field, P3 EMA settings in StochasticConfig |
| `mod.rs` | P1/P2 config wiring (line 2187), P2 queue-aware logic (lines 2218-2300) |
| `bin/market_maker.rs` | P1 CLI arguments, P3 EMA wiring |
| `estimator/microprice.rs` | P3 EMA fields and smoothing logic |
| `estimator/parameter_estimator.rs` | P3 `set_microprice_ema()` method |
| `lib.rs` | ReconcileConfig export |

## Key Implementation Details

### Thread-Safe EMA Storage
Used `AtomicU64` with `f64::to_bits()`/`from_bits()` for Sync-safe interior mutability:
```rust
const EMA_NONE: u64 = u64::MAX; // Sentinel for "no value"
ema_microprice_bits: AtomicU64,

// In microprice():
let prev_bits = self.ema_microprice_bits.load(Ordering::Relaxed);
if prev_bits == EMA_NONE {
    self.ema_microprice_bits.store(raw.to_bits(), Ordering::Relaxed);
} else {
    let prev = f64::from_bits(prev_bits);
    let smoothed = alpha * raw + (1.0 - alpha) * prev;
    self.ema_microprice_bits.store(smoothed.to_bits(), Ordering::Relaxed);
}
```

### Queue-Aware Integration
Integrated at the `reconcile_ladder_smart` level where `tier1.queue_tracker` is accessible:
```rust
if reconcile_config.use_queue_aware {
    for order in &current_bids {
        if !bid_action_oids.contains(&order.oid)
            && self.tier1.queue_tracker.should_refresh(order.oid, horizon)
        {
            bid_actions.push(LadderAction::Cancel { oid: order.oid });
            // ... place matching level
        }
    }
}
```

## Verification
- ✅ All 597 tests pass
- ✅ Clippy passes with zero warnings
- ✅ Build successful
- ✅ CLI `--help` shows new arguments

## Expected Impact

| Metric | Current | Target |
|--------|---------|--------|
| Orders/min | ~205 | <100 |
| Cancels/min | ~97 | <50 |
| API calls/min | ~300 | <150 |

## Rollback

| Phase | Rollback Method |
|-------|-----------------|
| P0 | Revert defaults to 1/10 |
| P1 | N/A - only adds optionality |
| P2 | `--use-queue-aware=false` (default) |
| P3 | `microprice_ema_alpha: 0.0` disables EMA |

## CLI Usage Examples

```bash
# Default behavior (P0 tolerances active)
cargo run --bin market_maker -- --asset BTC

# Enable queue-aware reconciliation
cargo run --bin market_maker -- --asset BTC --use-queue-aware --queue-horizon-seconds 2.0

# Adjust tolerances manually
cargo run --bin market_maker -- --asset BTC --skip-price-tolerance-bps 15 --max-modify-price-bps 75

# Check available options
cargo run --bin market_maker -- --help | grep -E "(skip-price|max-modify|queue)"
```

## Next Steps
1. Monitor live performance to validate ~50% churn reduction
2. Consider enabling `use_queue_aware: true` by default after validation
3. Tune EMA parameters based on microprice std dev analysis
