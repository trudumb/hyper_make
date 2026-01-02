# Session: OrphanTracker Implementation (Phase 7)
**Date**: 2026-01-02
**Focus**: Prevent false orphan detection during order lifecycle

## Problem Solved

The market maker had a critical race condition where valid orders were being cancelled as "orphans":

1. `place_bulk_orders()` registers pending orders with CLOIDs
2. API returns with OIDs, `finalize_pending_by_cloid()` moves to tracked
3. `safety_sync()` runs and queries exchange vs local
4. **Race**: Orders on exchange but not yet finalized locally → detected as orphans → cancelled!

Log evidence (from mm.log analysis):
- 40+ orphan cancellations in 2-minute session
- 35+ stale local removals
- 6 state mismatches
- Orders placed, then immediately cancelled as orphans

## Solution: OrphanTracker

New module `src/market_maker/infra/orphan_tracker.rs` provides:

### 1. Expected CLOID Protection

Orders in-flight (API called but not finalized) are protected:

```rust
// Before API call
self.infra.orphan_tracker.register_expected_cloids(&cloids);

// When API returns with OID
self.infra.orphan_tracker.record_oid_for_cloid(&cloid, oid);

// After finalization
self.infra.orphan_tracker.mark_finalized(&cloid, oid);

// On failure
self.infra.orphan_tracker.mark_failed(&cloid);
```

### 2. Orphan Grace Period

Even if order appears orphaned, wait 5 seconds before cancelling:

```rust
let (aged_orphans, new_count) = self.infra.orphan_tracker.filter_aged_orphans(&candidates);

// Only cancel orders that have been orphaned for > grace_period
for oid in aged_orphans {
    // Cancel...
    self.infra.orphan_tracker.clear_orphan(oid);
}
```

### 3. Recently Finalized Protection

Orders just finalized are protected for 10 seconds (protection_period):

```rust
pub fn protected_oids(&self) -> HashSet<u64> {
    // Includes: expected OIDs + recently finalized OIDs
}
```

## Configuration

```rust
pub struct OrphanTrackerConfig {
    pub orphan_grace_period: Duration,         // 5 sec default
    pub expected_cloid_ttl: Duration,          // 30 sec default
    pub finalized_protection_period: Duration, // 10 sec default
}
```

## Files Modified

1. **New: `src/market_maker/infra/orphan_tracker.rs`** (~330 lines)
   - `OrphanTracker` struct with all tracking logic
   - `OrphanTrackerConfig` configuration
   - Unit tests for all functionality

2. **`src/market_maker/infra/mod.rs`**
   - Added `orphan_tracker` module

3. **`src/market_maker/core/components.rs`**
   - Added `orphan_tracker: OrphanTracker` to `InfraComponents`
   - Updated constructors

4. **`src/market_maker/mod.rs`** (multiple locations)
   - `place_ladder_orders()`: register_expected_cloids, record_oid, mark_finalized, mark_failed
   - Modify fallback path: same integration
   - `place_bulk_ladder_orders()`: same integration
   - `safety_sync()`: Use filter_aged_orphans instead of immediate cancellation

## Order Placement Flow with Tracker

```
T=0ms   add_pending_with_cloid(cloid)
        register_expected_cloids([cloid])   ← NEW: Protect CLOID
        
T=1ms   API call to exchange
        
T=500ms API returns with OID
        record_oid_for_cloid(cloid, oid)    ← NEW: Associate OID
        
T=501ms finalize_pending_by_cloid(cloid)
        mark_finalized(cloid, oid)          ← NEW: Move to protection
        
T=10s   safety_sync()
        filter_aged_orphans(candidates)     ← NEW: Grace period check
        cleanup()                           ← NEW: Expire old entries
```

## Safety Sync Changes

Before:
```rust
let orphans = SafetyAuditor::find_orphans(&exchange_oids, &local_oids);
for oid in orphans {
    // Cancel immediately!
}
```

After:
```rust
self.infra.orphan_tracker.cleanup();

let candidates = SafetyAuditor::find_orphans(&exchange_oids, &local_oids);
let (aged_orphans, new_count) = self.infra.orphan_tracker.filter_aged_orphans(&candidates);

if new_count > 0 {
    debug!("New potential orphans detected - starting grace period");
}

for oid in aged_orphans {
    // Cancel only if aged past grace period
    self.infra.orphan_tracker.clear_orphan(oid);
}
```

## Testing

- All 463 tests pass
- Clippy clean
- Build successful
- New unit tests in orphan_tracker.rs

## Expected Impact

- **Eliminated**: False orphan detection during order finalization window
- **Reduced**: Unnecessary order cancellations
- **Improved**: Order state consistency
- **Added**: Grace period for legitimate orphans (e.g., from previous session)

## Configuration Tuning

If orphan detection is still too aggressive:
- Increase `orphan_grace_period` (default 5s → 10s)
- Increase `finalized_protection_period` (default 10s → 20s)

If orphan cleanup is too slow:
- Decrease `orphan_grace_period` (default 5s → 3s)
