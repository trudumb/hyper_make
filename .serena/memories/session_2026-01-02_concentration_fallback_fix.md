# Session: 2026-01-02 Concentration Fallback Fix

## Problem
Market maker with small accounts ($237) failed to place ANY orders on HIP-3 BTC ($90k) due to:
1. Position capacity (0.002633 BTC) split across 5 ladder levels
2. Each level < 0.001 BTC hardcoded min_level_size → all filtered out
3. Result: `orders_placed=0` despite valid margin

## Root Cause Analysis
From log `mm_hyna_BTC_2026-01-02_22-57-57.log`:
```
All bid levels filtered out (sizes too small)
  available_position=0.002633, min_level_size=0.001000
```

The min_level_size (0.001 BTC = $90 notional) was 9x stricter than exchange minimum ($10 notional).

When 0.002633 BTC split across 5 levels:
- Level 0: 0.00053 BTC < 0.001 → FILTERED
- Level 1: 0.00028 BTC < 0.001 → FILTERED
- ...all levels filtered

## Fix Implemented
**File: `src/market_maker/quoting/ladder/optimizer.rs`**

### 1. Dynamic min_size calculation
```rust
// Calculate from notional requirement instead of hardcoded value
let dynamic_min_size = (self.min_notional / self.price).max(self.min_size * 0.1);
let effective_min_size = dynamic_min_size.max(1e-8);
```

At $90k BTC with $10 min_notional: min_size = 0.000111 BTC (vs old 0.001)

### 2. Concentration Fallback
When proportional allocation produces sub-minimum sizes for ALL levels:
```rust
if valid_count == 0 && max_position_total >= effective_min_size {
    // Concentrate entire capacity into single best level
    if let Some((best_idx, best_mv)) = marginal_values.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    {
        raw_sizes[best_idx] = max_position_total;
    }
}
```

### 3. Added to both optimizers
- `optimize()` (proportional MV allocation)
- `optimize_kelly_stochastic()` (Kelly criterion allocation)

## Expected Behavior After Fix
With $237 account on HIP-3 BTC:
- Before: 0 orders (all levels filtered)
- After: 1 order at best level with size ~0.002633 BTC ($237 notional)

## Key Changes
- `optimize()`: lines 166-214 - dynamic min_size + concentration fallback
- `optimize_kelly_stochastic()`: lines 337-394 - same changes
- INFO log when concentration triggered: "Concentration fallback: single quote at best level"

## Verification
- `cargo check` ✓
- `cargo clippy -- -D warnings` ✓  
- `cargo test` ✓ (541 passed, 3 new tests added)

## New Tests Added
1. `test_concentration_fallback_small_capacity` - verifies concentration into single level when capacity is extremely limited
2. `test_dynamic_min_size_allows_small_orders` - verifies small accounts can now quote across multiple levels
3. `test_dynamic_min_size_from_notional` - verifies dynamic min_size calculation from notional

## Test Command
```bash
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug timeout 60 cargo run --bin market_maker -- \
  --network mainnet --asset BTC --dex hyna --log-file logs/test_concentration.log
```

Look for: "Concentration fallback" INFO log instead of "All bid levels filtered out" WARN.
