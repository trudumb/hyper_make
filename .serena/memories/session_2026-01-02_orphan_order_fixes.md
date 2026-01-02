# Session: Orphan Order and Capacity Warning Fixes (2026-01-02)

## Session Summary

Fixed two production issues identified via log troubleshooting:
1. **CLOID mismatch** in modify fallback causing orphan orders
2. **Incorrect threshold** for low sell/buy capacity warnings

## Problems Solved

### 1. Modify Fallback CLOID Mismatch (CRITICAL)

**Problem:** When modify failed and fell back to cancel+place, orphan orders were created because the CLOID didn't match.

**Root Cause:** 
```rust
// Line 1759: Caller creates CLOID-A
let cloid = uuid::Uuid::new_v4().to_string();
self.orders.add_pending_with_cloid(..., cloid.clone());

// Line 1767: But place_order() generated its own CLOID-B!
self.executor.place_order(asset, price, size, is_buy).await;
// Result: pending order with CLOID-A, but API returns CLOID-B
// finalize_pending_by_cloid() fails to find matching pending order
// → order becomes orphaned
```

**Fix:** Added optional `cloid` parameter to `place_order()`:
```rust
// In OrderExecutor trait (executor.rs):
async fn place_order(
    &self,
    asset: &str,
    price: f64,
    size: f64,
    is_buy: bool,
    cloid: Option<String>,  // NEW: Allow caller to specify CLOID
) -> OrderResult;

// In modify fallback path (mod.rs):
self.executor.place_order(..., Some(cloid.clone())).await;
```

### 2. Low Sell Capacity Warning Threshold (MEDIUM)

**Problem:** Spurious "Low sell capacity" warnings when position was tiny but had adequate capacity.

**Root Cause:**
```rust
// Old logic (WRONG):
if position > 0.0 && summary.available_sell < self.effective_max_position * 0.1 {
    warn!("Low sell capacity...");
}
// position=0.014, available_sell=0.016, effective_max=0.65
// 0.016 < 0.065 → warning triggered even though we can close!
```

**Fix:** Base threshold on actual position size with headroom:
```rust
// New logic (CORRECT):
const CAPACITY_HEADROOM: f64 = 1.5;
if position > 0.0 && summary.available_sell < position.abs() * CAPACITY_HEADROOM {
    warn!("Low sell capacity - insufficient to close long position");
}
// position=0.014, available_sell=0.016, required=0.021
// 0.016 < 0.021 → no warning (we can close with 1.14x headroom)
```

## Key Files Modified

### `src/market_maker/infra/executor.rs`
- Added `cloid: Option<String>` parameter to `OrderExecutor::place_order()` trait
- Updated `HyperliquidExecutor::place_order()` to use caller-provided CLOID when available

### `src/market_maker/mod.rs`
- Updated modify fallback path to pass CLOID to `place_order()`
- Updated `place_single_quote()` to pass `None` for CLOID
- Fixed capacity warning logic to use position-based threshold

## Testing

- All 448 tests pass
- Clippy clean
- Formatter applied

## Key Patterns Learned

### CLOID Consistency
When pre-registering pending orders with CLOID, ALWAYS pass the same CLOID to the API call:
```rust
let cloid = uuid::Uuid::new_v4().to_string();
self.orders.add_pending_with_cloid(..., cloid.clone());  // Pre-register
self.executor.place_order(..., Some(cloid.clone())).await;  // SAME CLOID!
```

### Capacity Warning Threshold
Warn about capacity only when it's insufficient to close the current position:
```rust
// Good: position-relative threshold
available < position.abs() * 1.5

// Bad: max-position-relative threshold (too conservative for small positions)
available < max_position * 0.1
```
