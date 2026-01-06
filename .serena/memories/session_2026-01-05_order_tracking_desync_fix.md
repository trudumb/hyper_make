# Session: 2026-01-05 Order Tracking Desync Fix

## Summary
Fixed critical bug where market maker orders on exchange were completely out of sync with calculated quotes. Orders at 26.38-26.89 remained on exchange while MM calculated quotes at 26.55-26.60 without updating them.

## Root Cause Analysis
The reconciliation system (`reconcile_ladder_smart()`) only operates on orders tracked locally via `OrderManager.get_all_by_side()`. When orders become "orphaned" (exist on exchange but not in local tracking), they cannot be modified by the reconciliation logic.

Key evidence from logs:
- 2,794 quote cycles but only 5 modifies and 46 placements (~98% skip rate)
- "Stale order in tracking (not on exchange)" warnings
- "Untracked order filled" warnings showing fills on unknown orders
- MM calculated quotes (26.55-26.60) but exchange had orders at 26.38-26.89

Contributing factors:
1. **Order state confusion**: Orders may get removed from local tracking before cancelled on exchange
2. **SafetySync too slow**: Only ran every 60 seconds, missing orphans for too long
3. **Possible HIP-3 asset name mismatch**: Safety sync filters by `o.coin == self.config.asset`

## Changes Made

### 1. Diagnostic Logging for Order Counts
**File**: `src/market_maker/mod.rs:2330-2338`

Added INFO-level logging to show order tracking state during reconciliation:
```rust
info!(
    local_bids = current_bids.len(),
    local_asks = current_asks.len(),
    target_bid_levels = bid_levels.len(),
    target_ask_levels = ask_levels.len(),
    "[Reconcile] Order counts"
);
```

Purpose: Reveals if local tracking has lost sync (e.g., local_bids=0 but orders exist on exchange).

### 2. HIP-3 Asset Matching Diagnostic
**File**: `src/market_maker/mod.rs:3463-3500`

Added detection for asset name mismatch in SafetySync:
```rust
if total_exchange_orders > 0 && matching_orders.is_empty() {
    let sample_coins: Vec<_> = exchange_orders.iter().take(3).map(|o| &o.coin).collect();
    warn!(
        total_exchange_orders = total_exchange_orders,
        our_asset = %self.config.asset,
        sample_coins = ?sample_coins,
        "[SafetySync] No matching orders! Possible HIP-3 asset name mismatch"
    );
}
```

Purpose: Detect if HIP-3 DEX returns different asset names than expected.

### 3. Increased SafetySync Frequency
**File**: `src/market_maker/mod.rs:551-553`

Changed interval from 60 seconds to 15 seconds:
```rust
let mut sync_interval = tokio::time::interval(Duration::from_secs(15));
```

Purpose: Faster orphan detection and state reconciliation.

### 4. Drift Detection with Forced Reconciliation
**File**: `src/market_maker/mod.rs:2340-2434`

Added automatic detection of price drift between existing orders and targets:
- Calculates max drift in bps between current orders and target levels
- If drift > 100 bps, triggers forced cancel-all + replace-all
- Logs drift amounts for debugging

```rust
const MAX_ACCEPTABLE_DRIFT_BPS: f64 = 100.0;

let bid_drift = ...; // Calculate max drift for bids
let ask_drift = ...; // Calculate max drift for asks

if max_drift > MAX_ACCEPTABLE_DRIFT_BPS {
    // Cancel all orders
    self.initiate_bulk_cancel(all_oids).await;
    
    // Place new orders at target prices
    let order_specs = bid_levels.iter().map(|l| OrderSpec::new(...))...;
    self.executor.place_bulk_orders(&self.config.asset, order_specs).await;
}
```

Purpose: Automatically recover from desync states without waiting for SafetySync.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `src/market_maker/mod.rs` | 551-553 | SafetySync interval 60s → 15s |
| `src/market_maker/mod.rs` | 2330-2338 | Added order count diagnostic logging |
| `src/market_maker/mod.rs` | 2340-2434 | Added drift detection + forced reconciliation |
| `src/market_maker/mod.rs` | 3463-3500 | Added HIP-3 asset matching diagnostic |

## Key Log Messages to Watch

| Message | Meaning |
|---------|---------|
| `[Reconcile] Order counts` | Shows local vs target order counts |
| `[Reconcile] Large price drift detected!` | Drift > 100 bps, forcing full reconciliation |
| `[Reconcile] Cancelled all stale orders due to drift` | Orders cancelled in drift recovery |
| `[Reconcile] Placed new orders after drift correction` | Fresh orders placed |
| `[SafetySync] No matching orders!` | HIP-3 asset name mismatch detected |

## Verification

```bash
cargo check   # ✅ Compiles with 1 warning (dead_code)
cargo build --release   # ✅ Build successful
```

## Technical Notes

### OrderSpec and OrderResult Types
- `OrderSpec::new(price, size, is_buy)` - creates order specification
- `OrderResult.oid` is `u64` (0 if failed), not `Option<u64>`
- Check `result.oid != 0` for successful order placement

### Reconciliation Flow
1. Get locally tracked orders via `OrderManager.get_all_by_side()`
2. Compare to target ladder levels from strategy
3. Match orders to targets within 100 bps
4. Skip (within tolerance), Modify (small change), or Cancel+Place (large change)

Problem: Step 1 returns empty if orders were removed from local tracking but still exist on exchange.

## Related Sessions
- `session_2026-01-02_orphan_order_fixes` - Original orphan tracker implementation
- `session_2026-01-02_orphan_tracker_implementation` - OrphanTracker design

## Next Steps
1. Run MM with new build and observe log messages
2. Verify drift detection triggers on actual desync scenarios
3. Monitor if HIP-3 asset mismatch warning appears
4. Consider adding exchange state polling if issues persist
