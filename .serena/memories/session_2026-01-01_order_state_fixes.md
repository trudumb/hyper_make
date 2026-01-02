# Session: Order State Management Fixes (2026-01-01)

## Session Summary

This session focused on troubleshooting and fixing critical order state management issues in the market maker system.

## Problems Solved

### 1. Modify Fallback Orphan Orders
**Problem:** When modify failed and fell back to cancel+place, the replacement order wasn't being tracked properly.
**Root Cause:** The fallback path ignored the `place_order` result and didn't add the new order to OrderManager.
**Fix:** Complete rewrite of modify fallback in `mod.rs` (lines 1664-1850) to:
- Detect "filled" and "canceled" errors and transition states appropriately
- Call `initiate_cancel()` before exchange cancel
- Use CLOID-based pending→finalize flow for replacement orders
- Properly handle partial immediate fills in fallback path

### 2. Modify Race Condition with Fills
**Problem:** Orders filled on exchange but WS notification delayed caused "Cannot modify canceled or filled order" errors.
**Root Cause:** Race condition between fill and modify attempt.
**Fix:** Added error detection in modify result handler:
- Detect "filled" → transition to FilledImmediately
- Detect "canceled" → remove from tracking

### 3. Partial Immediate Fill State Management
**Problem:** Partial immediate fills were marked as `FilledImmediately` (terminal state), causing the resting portion to become orphaned.
**Root Cause:** `FilledImmediately` is NOT in `is_active()` (only `Resting | PartialFilled`), so orders wouldn't be included in `get_all_by_side()` for reconciliation.
**Fix:** Use `PartialFilled` state for partial immediate fills so the resting portion remains in reconciliation.

### 4. Partial Immediate Fill Deduplication
**Problem:** `immediate_fill_oids: HashSet<u64>` only tracked OID, not amount. For partial fills, couldn't distinguish which WS fill corresponded to immediate vs resting portion.
**Root Cause:** First WS fill would remove OID from set, but that might be the wrong fill.
**Fix:** Changed to `immediate_fill_amounts: HashMap<u64, f64>` to track AMOUNT per OID:
- `pre_register_immediate_fill(oid, amount)` now takes amount
- `process()` calculates `skip_amount = min(remaining, fill.size)`
- Decrements remaining amount as WS fills arrive
- Only skips position update for the immediate portion

## Key Files Modified

### `src/market_maker/mod.rs`
- Lines 1457-1488: Immediate fill registration now passes amount
- Lines 1664-1850: Complete modify fallback rewrite
- Lines 1781-1793: Modify fallback immediate fill registration

### `src/market_maker/fills/processor.rs`
- Changed `immediate_fill_oids: HashSet<u64>` → `immediate_fill_amounts: HashMap<u64, f64>`
- Updated `pre_register_immediate_fill(oid, amount)` signature
- Updated `process()` to handle amount-based dedup
- Added `get_immediate_fill_remaining()` helper
- Added `test_partial_immediate_fill_multiple_ws_fills` test

### `src/market_maker/tracking/order_manager/types.rs`
- `OrderState` enum reference: `is_active()` returns `Resting | PartialFilled` only
- `is_terminal()` includes `FilledImmediately`

## Key Patterns Learned

### Order State Machine
```
Resting → PartialFilled → Filled (normal lifecycle)
Resting → FilledImmediately (full immediate fill, terminal)
Resting → CancelPending → CancelConfirmed | Cancelled | FilledDuringCancel
PartialFilled → stays active for reconciliation
```

### Immediate Fill Dedup Pattern
```rust
// Register amount, not just OID
pre_register_immediate_fill(oid, actual_fill_amount);

// In process():
let remaining = immediate_fill_amounts.get(&oid).unwrap_or(0.0);
let skip_amount = remaining.min(fill.size);
let update_amount = fill.size - skip_amount;
// Only update position by update_amount
```

### CLOID-Based Order Tracking
- Use CLOID for deterministic order matching
- `add_pending_with_cloid()` before API call
- `finalize_pending_by_cloid()` after successful placement
- Eliminates timing race between fill and order registration

## Tests Added

- `test_partial_immediate_fill_multiple_ws_fills`: Verifies correct position tracking when:
  - Order for 1.0 partially fills 0.6 immediately
  - Multiple WS fills arrive with different sizes
  - Each fill correctly decrements remaining and updates position

## Technical Notes

### WsFill Interface (from Hyperliquid)
```typescript
interface WsFill {
  oid: number;    // order id
  tid: number;    // unique trade id (for dedup)
  sz: string;     // size of this fill
  time: number;   // timestamp
  // ... other fields
}
```

### Critical Insight
Each WS fill has its own TID and represents a discrete trade. The API immediate fill gives total amount but not individual TIDs. Must match by decrementing amount, not by OID presence.
