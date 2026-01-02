# Session: Historical Fill Snapshot Fix

## Date: 2026-01-02

## Problem Diagnosed

**Issue**: "Untracked order filled" warnings flooding logs at startup

**Root Cause**: When subscribing to `UserFills` WebSocket feed, Hyperliquid sends a **snapshot** of recent historical fills from previous sessions. The code was processing these as new fills, causing:

1. "Untracked order filled" warnings - OIDs from previous sessions aren't in current `OrderManager`
2. False position updates - Historical fills incorrectly modifying position
3. Reconciliation spam - Each unmatched fill triggering reconciliation

**Evidence from logs**:
- OIDs like 45929151455, 45929151456 were from previous MM session
- All fills arrived within milliseconds at startup (19:50:09.963-977)
- `cloid=None` indicating these weren't placed in current session
- Position jumping around as historical fills were replayed

## Fix Applied

**File**: `src/market_maker/mod.rs` (lines 707-716)

```rust
// Skip snapshot fills - these are historical fills from previous sessions.
// Position is already loaded from exchange at startup, so processing these
// would cause "untracked order filled" warnings for orders we didn't place.
if user_fills.data.is_snapshot.unwrap_or(false) {
    debug!(
        fills = user_fills.data.fills.len(),
        "Skipping UserFills snapshot (historical fills from previous sessions)"
    );
    return Ok(());
}
```

**Rationale**:
- Position is already loaded from exchange at startup via `load_initial_state()`
- Snapshot fills are historical - they happened before this session
- We only want to process **new** fills for orders we placed in current session
- `is_snapshot` field in `UserFillsData` indicates initial snapshot vs live updates

## Related Fixes in Same Session

### 1. Logging Improvements

- **OID tracking log**: Changed from INFO to DEBUG in `mod.rs:1948-1953`
- **Quote cycle summary**: Added consolidated log with spread info at `mod.rs:2189-2200`
- **get_quote_summary()**: Added method to `OrderManager` at `manager.rs:649-665`

### 2. Reconciliation Debounce

**File**: `src/market_maker/infra/reconciliation.rs` (lines 176-181)

```rust
// oid=0 with size=0 is a dedup artifact, not a real unmatched fill
if oid == 0 && size.abs() < 1e-9 {
    tracing::debug!("Skipping reconciliation for oid=0 size=0 (dedup artifact)");
    return;
}
```

## Key Insight

The `UserFillsData` struct has an `is_snapshot: Option<bool>` field that distinguishes:
- `is_snapshot = true`: Initial historical fill snapshot on subscription
- `is_snapshot = false` or `None`: Live fill updates

This pattern applies to other WebSocket subscriptions too (UserFundings, etc.).

## Files Modified

1. `src/market_maker/mod.rs` - Skip snapshot fills, logging improvements
2. `src/market_maker/tracking/order_manager/manager.rs` - Added `get_quote_summary()`
3. `src/market_maker/infra/reconciliation.rs` - Debounce oid=0 warnings
