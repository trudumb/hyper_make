# Session: Multi-Level Ladder Fix for Low-Price Assets

**Date**: 2026-01-03
**Focus**: Auto-calculate position/liquidity to support 5 ladder levels

## Problem

HYPE at ~$25 with default `--max-position 0.5`:
- Total per-side capacity: 0.5 HYPE = $12.50
- Min per-level requirement: 0.4 HYPE = $10 (min_notional)
- Result: Only 1 order per side (concentration fallback)

Even after increasing to 2.0 HYPE (5 × min_viable), still only 3 levels placed because:
- Size allocation uses **marginal-value weighting** (lambda × spread_capture)
- Outer levels get exponentially less size (~6% for level 5)
- Level 5 at 0.13 HYPE × $25 = $3.25 < $10 min_notional

## Root Cause

In `src/bin/market_maker.rs` line 938, the original calculation assumed equal allocation:
```rust
let min_for_ladder = min_viable_liquidity * num_ladder_levels as f64;
```

But actual allocation is proportional to marginal value, with outer levels getting ~5-10% of total.

## Solution

Modified `src/bin/market_maker.rs` (lines 934-948) to account for decay:

```rust
// With 5 levels and typical decay, smallest fraction ≈ 0.065 (6.5%).
// Safety factor: use 0.05 (5%) to ensure all levels pass after truncation.
let num_ladder_levels = 5_u8;
let smallest_level_fraction = 0.05; // Conservative: outer level gets ~5% of total
let min_for_ladder = min_notional / (mark_px * smallest_level_fraction);
```

For HYPE at $25:
- `min_for_ladder = $10 / ($25 × 0.05) = 8.0 HYPE`
- Smallest level gets 5% = 0.4 HYPE = $10 ✓

## Key Changes

1. **`src/bin/market_maker.rs`**:
   - Auto-increases `max_position` from 0.5 to 8.0 HYPE
   - Auto-increases `target_liquidity` from 0.25 to 8.0 HYPE
   - Uses `smallest_level_fraction = 0.05` to ensure all 5 levels pass min_notional
   - Added logging for auto-increase events

## Results

| Metric | Before | After |
|--------|--------|-------|
| Bid levels | 1 | 5 |
| Ask levels | 1 | 5 |
| Total orders | 2 | 10 |
| "concentration fallback" | Yes | No |
| min_for_ladder | 2.0 HYPE | 8.0 HYPE |

## Key Insight: Marginal-Value Weighted Allocation

The ladder generator doesn't allocate sizes equally. It uses:
```
size[i] ∝ lambda(depth[i]) × spread_capture(depth[i])
```

Where:
- `lambda(d)` = fill intensity, decays exponentially with depth
- `spread_capture(d)` = profit per fill, also decays with depth

Result: Level 1 gets ~40%, Level 5 gets ~5% of total size.

## Math Summary

| Asset | Price | 5% fraction | min_for_ladder | Result |
|-------|-------|-------------|----------------|--------|
| HYPE  | $25   | 0.05 HYPE   | 8.0 HYPE       | 5 levels ✓ |
| BTC   | $95K  | 0.000005 BTC| 0.0021 BTC     | 5 levels ✓ |

Formula: `min_for_ladder = min_notional / (price × 0.05)`

## Files Modified

- `src/bin/market_maker.rs` (lines 934-973): Position limit calculation with decay-aware sizing

## Test Command

```bash
RUST_LOG=hyperliquid_rust_sdk::market_maker=info timeout 60 cargo run --release --bin market_maker -- \
  --asset HYPE --dex hyna --network mainnet \
  --log-file logs/mm_hyna_HYPE_5levels.log
```

Expected output:
- `"Auto-increased max_position to support 5 ladder levels"`
- `"Bulk order placed","resting":5` (twice, for bids and asks)
- `"Bulk cancel completed","cancelled":10` (on shutdown)
