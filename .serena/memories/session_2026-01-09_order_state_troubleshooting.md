# Session: 2026-01-09 Order State Management Troubleshooting

## Summary
Analyzed order rejection logs and implemented fixes for WebSocket state sync race conditions and rate limiter backoff issues. Added error classification system for smarter rejection handling.

## Root Causes Identified

### Issue 1: Leverage/Margin Tier Violation
- MM requested 40x leverage but exchange limited to 25x due to position size
- MM continued calculating with 40x, causing ALL buy orders to be rejected with `PerpMaxPosition`
- Fix: Parse leverage response to extract actual allowed leverage

### Issue 2: WS State Sync Race Condition
- Orders placed via WS POST immediately added to `ws_state`
- `openOrders` WS subscription snapshot arrives before exchange processed new orders
- Fresh orders incorrectly removed from state, creating mismatch

### Issue 3: Backoff Counter Cascade
- Each order in bulk rejection individually incremented counter
- 25 rejections in one batch immediately triggered 120s backoff
- Fix: Treat batch rejections as single event

## Changes Made

### 1. Leverage Response Parsing
**File:** `src/bin/market_maker.rs`
- Added `parse_leverage_from_error()` helper function
- Modified leverage setting to use actual (exchange-limited) leverage
- Pattern: `"...is Nx..."` extraction from error message

### 2. Grace Period for Fresh Orders
**File:** `src/market_maker/mod.rs:~1386-1410`
```rust
const GRACE_PERIOD_SECS: u64 = 2;
let grace_period = std::time::Duration::from_secs(GRACE_PERIOD_SECS);

// Skip removal of orders younger than grace period
let (stale_oids, skipped_fresh): (Vec<u64>, Vec<u64>) = current_ws_oids
    .iter()
    .filter(|oid| !ws_open_oids.contains(oid))
    .partition(|oid| {
        self.ws_state.get_order(**oid)
            .map(|o| o.placed_at.elapsed() > grace_period)
            .unwrap_or(true)
    });
```

### 3. Batch Rejection Recording
**File:** `src/market_maker/infra/rate_limit.rs:247-300`
- Added `record_batch_rejection()` method
- Increments `consecutive_rejections` by 1 (not count)
- Preserves `total_rejections` count for metrics accuracy
- Updated both calling locations in `mod.rs` to aggregate rejections

### 4. Error Classification System
**File:** `src/market_maker/infra/rate_limit.rs:20-96`
```rust
pub enum RejectionErrorType {
    PositionLimit,  // Skip side entirely (PerpMaxPosition)
    Margin,         // Transient, use backoff
    PriceError,     // May retry with adjusted price (BadAloPx)
    Other,          // Default handling
}
```
- Added `classify()` method for error categorization
- Added `should_skip_side()`, `should_backoff()`, `is_transient()` helpers
- Added `classify_and_record_batch()` for comprehensive handling

## Files Modified

| File | Changes |
|------|---------|
| `src/bin/market_maker.rs` | Added leverage parsing from error message |
| `src/market_maker/mod.rs` | Grace period for WS sync, batch rejection handling in 2 locations |
| `src/market_maker/infra/rate_limit.rs` | Added RejectionErrorType, record_batch_rejection, classify_and_record_batch |

## Tests Added
- `test_batch_rejection_counts_as_single_event`
- `test_batch_rejection_triggers_backoff_at_threshold`
- `test_batch_rejection_zero_count`
- `test_error_classification_position_limit`
- `test_error_classification_margin`
- `test_error_classification_price_error`
- `test_error_classification_other`
- `test_classify_and_record_batch`

## Verification
- ✅ `cargo check` passes
- ✅ `cargo fmt` applied
- ✅ 693 library tests pass
- ✅ All 19 rate_limit tests pass (including 8 new tests)

## Metrics Impact
| Metric | Before | After |
|--------|--------|-------|
| Backoff cascade from 25 rejections | 120s immediately | 5s (single increment) |
| Fresh order removal in WS sync | Immediate | 2s grace period |
| Error type awareness | None | 4 categories |

## Pending Tasks (Phase 2)
- Add margin tier data parsing from meta endpoint
- Implement MarginTierCalculator for dynamic leverage

## Key Insights
1. WebSocket-to-WebSocket timing issues require grace periods, not just REST timing fixes
2. Bulk order rejections should be treated as single events for backoff purposes
3. Error classification enables smarter handling (skip side vs backoff vs retry)
4. Exchange may reduce leverage at runtime based on position size - must parse response
