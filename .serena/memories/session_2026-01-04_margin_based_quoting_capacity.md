# Session: Margin-Based Quoting Capacity Fix

**Date:** 2026-01-04
**Status:** ✅ COMPLETE

## Summary

Fixed quoting capacity to use margin-based values instead of CLI `--max-position` parameter. The `--max-position` flag is now optional - when omitted, the system uses margin-based capacity for quoting and liquidation proximity for reduce-only triggers.

## Problem

The ladder optimizer was using `max_position_total: 0.500000` (from CLI `--max-position 0.5`) instead of the margin-based capacity (~46 HYPE based on $233 margin × 5x leverage / $4.95 price).

This caused:
- All liquidity concentrated at inner levels
- "Concentration fallback triggered" warnings
- Inefficient use of available margin

## Root Cause Analysis

Three issues were identified:

1. **Priority order in `effective_max_position()`** (market_params.rs):
   - Was checking `dynamic_max_position` FIRST
   - Should check `margin_quoting_capacity` first (hard solvency constraint)

2. **Exchange limits refresh** (mod.rs:3327):
   - Was passing `config.max_position` (user CLI value)
   - Should pass `effective_max_position` (margin-based)

3. **Timing issue** (mod.rs quote_cycle):
   - `update_local_max` was called AFTER building `sources`
   - Market params had stale exchange limits from previous cycle

## Fixes Applied

### 1. market_params.rs - Fixed priority order
```rust
pub fn effective_max_position(&self, static_fallback: f64) -> f64 {
    const EPSILON: f64 = 1e-9;

    // PRIORITY 1: Margin-based quoting capacity (HARD solvency constraint)
    if self.margin_quoting_capacity > EPSILON {
        if self.dynamic_limit_valid && self.dynamic_max_position > EPSILON {
            return self.margin_quoting_capacity.min(self.dynamic_max_position);
        }
        return self.margin_quoting_capacity;
    }

    // PRIORITY 2: Dynamic volatility-adjusted limit from kill switch
    if self.dynamic_limit_valid && self.dynamic_max_position > EPSILON {
        return self.dynamic_max_position;
    }

    // Last resort: static fallback (only during early warmup)
    static_fallback
}
```

### 2. mod.rs:3324-3333 - Fixed refresh_exchange_limits
```rust
let local_max = if self.effective_max_position > 0.0 {
    self.effective_max_position
} else {
    self.config.max_position
};
self.infra.exchange_limits.update_from_response(&asset_data, local_max);
```

### 3. mod.rs:1174-1210 - Pre-compute before building sources
```rust
// CRITICAL: Pre-compute effective_max_position BEFORE building sources
let margin_quoting_capacity = (margin_state.available_margin * max_leverage / self.latest_mid).max(0.0);
let dynamic_max_position = dynamic_max_position_value / self.latest_mid;

let pre_effective_max_position = {
    if margin_quoting_capacity > EPSILON {
        if dynamic_limit_valid && dynamic_max_position > EPSILON {
            margin_quoting_capacity.min(dynamic_max_position)
        } else {
            margin_quoting_capacity
        }
    } else if dynamic_limit_valid && dynamic_max_position > EPSILON {
        dynamic_max_position
    } else {
        self.config.max_position
    }
};

// Update BEFORE building sources
self.infra.exchange_limits.update_local_max(pre_effective_max_position);
```

## Verification

Logs now show:
```
Quoting capacity: user max_position is for reduce-only only
  effective_max_position: 47.25 HYPE
  margin_quoting_capacity: 47.25
  
Ladder spread diagnostics
  bid_levels: 5, ask_levels: 5
```

## Liquidation Proximity Reduce-Only

Discovery: The liquidation proximity-based reduce-only was ALREADY FULLY IMPLEMENTED:

**Priority order in filter.rs:**
1. `ApproachingLiquidation` - buffer_ratio < 0.5 (HIGHEST)
2. `OverMarginUtilization` - margin_used/account_value > 80%
3. `OverValueLimit` - position_value > max_position_value
4. `OverPositionLimit` - position > max_position (fallback)

**Buffer ratio formula:**
```
distance_to_liq = |mark_price - liquidation_px| / mark_price
buffer_ratio = distance / (distance + maintenance_margin_rate)
  - 0.0 = at liquidation
  - 0.5 = trigger threshold (default)
  - 1.0 = completely safe
```

## Files Modified

- `src/market_maker/strategy/market_params.rs` - Priority order fix
- `src/market_maker/mod.rs` - Pre-computation and exchange limits
- `src/market_maker/strategy/ladder_strat.rs` - INFO-level logging
- `WORKFLOW.md` - Documentation updates

## Usage

**Default (capital-efficient):**
```bash
./target/debug/market_maker --network mainnet --asset HYPE --dex hyna
```

**With optional hard cap:**
```bash
./target/debug/market_maker --network mainnet --asset HYPE --dex hyna --max-position 10.0
```

## Related Sessions

- `session_2026-01-04_dynamic_reduce_only_liquidation` - Initial investigation
- `session_2026-01-03_capital_efficient_position_limits` - Earlier work on position limits
