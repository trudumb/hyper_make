# Controller-Derived Position Sizing Fix

## Date: 2026-01-04

## Problem
Ladder quotes were broken - the multi-level ladder was collapsing to a single-level "concentration fallback" every cycle. Log analysis revealed:
- User's `--max-position 0.5` was being used as the quoting capacity cap
- With 0.5 / 5 levels = 0.1 contracts per level = ~$2.53 notional (below $10 minimum)
- All levels filtered out → Kelly optimizer fell back to single-level concentration

## Root Cause
Two distinct concepts were conflated:
1. **Quoting Capacity** - how much the optimizer can allocate (should be margin-constrained)
2. **Reduce-Only Trigger** - when to stop adding exposure (user config via `--max-position`)

User's `--max-position` was being used for BOTH, when it should only control the reduce-only filter.

## Solution
Separate quoting capacity (controller-derived) from reduce-only trigger (user config):

### Files Modified

**1. `src/market_maker/strategy/market_params.rs`**
- Added `margin_quoting_capacity: f64` field
  - Formula: `margin_available × leverage / microprice`
  - The HARD solvency constraint from account margin
- Added `quoting_capacity()` method
  - Returns margin-based capacity for ladder allocation
  - Falls back to dynamic_max_position or computes from margin_available
- Updated `effective_max_position()` to use controller-derived values

**2. `src/market_maker/strategy/params.rs`**
- Added computation of `margin_quoting_capacity` from `margin_sizer.state().available_margin`
- Updated `dynamic_max_position` to fall back to margin-based calculation

**3. `src/market_maker/strategy/ladder_strat.rs`**
- Updated `generate_ladder()` to call `market_params.quoting_capacity()` instead of using user's `max_position`
- Added debug logging when controller-derived capacity differs from user config

## Key Insight
From stochastic control theory (GLFT/Kelly):
- Position limits should derive from the **controller** (margin constraints, volatility, Kelly criterion)
- User's `--max-position` is a **policy override** for reduce-only mode, not a quoting capacity cap
- The Kelly optimizer should allocate based on what the account can actually support (margin × leverage / price)

## Verification
- `cargo build` ✓
- `cargo clippy -- -D warnings` ✓

## Expected Behavior After Fix
- Ladder will allocate based on `margin_quoting_capacity` (~23.5 contracts with $237.50 account, 5x leverage, ~$25 price)
- User's `--max-position 0.5` will only trigger reduce-only mode when position exceeds 0.5
- Multi-level ladder should work correctly instead of concentrating to single level
