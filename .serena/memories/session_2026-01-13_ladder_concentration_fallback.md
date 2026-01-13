# Session: 2026-01-13 Ladder Concentration Fallback Fix

## Summary
Fixed min_notional issue where ladder was completely empty when margin was tight but still above exchange minimum. Added concentration fallback after entropy optimizer filtering.

## Problem
When running with limited margin (~$209), the system was:
1. Configuring 25 ladder levels (default)
2. Reducing to 19 bid / 13 ask based on margin constraints
3. Entropy optimizer distributing size across these levels
4. Each level ending up with ~0.00016 BTC (~$14.7)
5. After geometric decay, levels falling below $10 min_notional
6. All levels being filtered out, resulting in empty ladder

**Log evidence:**
```json
{"message":"Reducing ladder levels due to tight exchange limits","configured":25,"effective_bid":19,"effective_ask":13}
{"message":"Ladder completely empty: all levels below min_notional or margin constraint"}
```

## Root Cause
The concentration fallback in `generator.rs::build_raw_ladder()` was being bypassed because:
1. `Ladder::generate()` creates levels with some sizes
2. Entropy optimizer redistributes sizes across all levels
3. Filtering happens after optimization, not after generation
4. No fallback existed at the filtering stage

## Solution
Added concentration fallback in `ladder_strat.rs` after step 9 (filtering):

**File:** `src/market_maker/strategy/ladder_strat.rs:1043-1146`

When all levels are filtered out but total available size meets min_notional:
- Create single concentrated order at tightest depth
- Use full available size for that side
- Log the fallback for visibility

```rust
// 10. CONCENTRATION FALLBACK: If all levels filtered out but total size
// meets min_notional, create single concentrated order at tightest depth.
if bids_before > 0 && ladder.bids.is_empty() {
    let total_bid_size = truncate_float(available_for_bids, config.sz_decimals, false);
    let bid_notional = total_bid_size * market_params.microprice;

    if total_bid_size > min_size_for_order && bid_notional >= config.min_notional {
        // Create concentrated order at tightest depth
        let offset = market_params.microprice * (tightest_depth_bps / 10000.0);
        let bid_price = round_to_significant_and_decimal(
            market_params.microprice - offset, 5, config.decimals,
        );

        ladder.bids.push(LadderLevel {
            price: bid_price,
            size: total_bid_size,
            depth_bps: tightest_depth_bps,
        });

        info!("Bid concentration fallback: collapsed to single order at tightest depth");
    }
}
```

## Files Modified

| File | Changes |
|------|---------|
| `src/market_maker/strategy/ladder_strat.rs:5` | Added `round_to_significant_and_decimal` import |
| `src/market_maker/strategy/ladder_strat.rs:13` | Added `LadderLevel` import |
| `src/market_maker/strategy/ladder_strat.rs:1043-1146` | Added concentration fallback after filtering |

## Expected Log Behavior

**Before fix:**
```
Ladder completely empty: all levels below min_notional or margin constraint
```

**After fix:**
```
Bid concentration fallback: collapsed to single order at tightest depth
Ask concentration fallback: collapsed to single order at tightest depth
```

Or if truly below min_notional:
```
Ladder completely empty: available size below min_notional (no fallback possible)
```

## Verification

```bash
cargo build           # ✅ Passed
cargo test ladder     # ✅ 83 tests passed
cargo test            # ✅ 828 tests passed
```

## Design Rationale

**Why add fallback at filtering stage instead of fixing entropy optimizer?**

1. The entropy optimizer is working as designed - it distributes size across levels for diversity
2. The issue is that with tight margin, the distributed sizes fall below exchange minimum
3. The concentration fallback is the mathematically correct response: when margin doesn't support multi-level quoting, use single-level quoting
4. This maintains the first-principles architecture while handling edge cases gracefully

**Trade-off:**
- Single concentrated order means less diversity and higher queue position uncertainty
- But this is better than no orders at all when margin is tight
- As margin grows, the system will naturally return to multi-level quoting

## Related Files
- `src/market_maker/quoting/ladder/generator.rs:525-571` - Original concentration fallback in build_raw_ladder
- `src/market_maker/quoting/ladder/mod.rs:128` - Default num_levels = 25

## Pre-existing Clippy Issues (Not Addressed)

The following clippy issues existed before this session:
- `excessive_precision` in changepoint.rs (gamma_ln Lanczos coefficients)
- `needless_range_loop` in control/value.rs
- `collapsible_else_if` in learning/mod.rs
