# Session: HIP-3 Open Orders API Fix

**Date:** 2026-01-03
**Status:** Complete
**Issue:** Order state management broken on HIP-3 DEXs - all orders incorrectly marked as stale

## Problem Summary

When running the market maker on HIP-3 DEXs (e.g., `hyna:BTC`), every safety sync cycle found ALL locally tracked orders as "stale" (tracked locally but not on exchange):

```
[SafetySync] Stale order in tracking (not on exchange): oid=285064225307 - removing
[SafetySync] Stale order in tracking (not on exchange): oid=285064225308 - removing
... (8-10 orders every 11 seconds)
```

This caused the SafetySync to continuously remove all orders from local tracking, breaking order state management.

## Root Cause

The `OpenOrders` InfoRequest variant lacked a `dex` parameter. While other API endpoints like `Meta`, `AllMids`, and `UserState` were updated for HIP-3 support with optional DEX parameters, `OpenOrders` was missed:

```rust
// BEFORE (broken for HIP-3):
OpenOrders {
    user: Address,
}

// AFTER (HIP-3 aware):
OpenOrders {
    user: Address,
    #[serde(skip_serializing_if = "Option::is_none")]
    dex: Option<String>,
}
```

Without the `dex` parameter, the API only returned validator perps orders, not HIP-3 DEX orders. Since HIP-3 orders weren't returned, SafetySync thought they didn't exist on the exchange.

## Solution Implemented

### 1. Added DEX parameter to OpenOrders InfoRequest
Location: `src/info/info_client.rs:58-66`

### 2. Added `open_orders_for_dex()` method
Location: `src/info/info_client.rs:244-254`

```rust
pub async fn open_orders_for_dex(
    &self,
    address: Address,
    dex: Option<&str>,
) -> Result<Vec<OpenOrdersResponse>>
```

### 3. Updated MarketMaker to use DEX-aware method
All `open_orders()` calls in the market maker now use `open_orders_for_dex()`:
- `sync_open_orders()` - line 283
- `cancel_all_orders_on_startup()` - lines 358, 413
- Safety sync (Step 5: Exchange reconciliation) - line 2921

### 4. Updated market_maker binary
- `--status` command handler - line 1403

## Files Modified

1. `src/info/info_client.rs`
   - Added `dex` field to `OpenOrders` variant
   - Added `open_orders_for_dex()` method

2. `src/market_maker/mod.rs`
   - Updated 4 call sites to use `open_orders_for_dex()`

3. `src/bin/market_maker.rs`
   - Updated `--status` to use DEX-aware open_orders

## Backward Compatibility

- `open_orders()` still exists and passes `dex: None`
- Examples using `open_orders()` continue to work unchanged
- Validator perps behavior is identical (dex: None)

## Testing

- Build: ✓
- Clippy: ✓

## Related Sessions

- `session_2026-01-02_hip3_dex_support_complete` - Original HIP-3 implementation
- `session_2026-01-03_hip3_asset_index_fix` - Fixed asset index formula
- `session_2026-01-02_orphan_order_fixes` - CLOID mismatch fix
