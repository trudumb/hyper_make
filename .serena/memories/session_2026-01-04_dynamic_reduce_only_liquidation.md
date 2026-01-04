# Dynamic Reduce-Only Based on Liquidation Proximity

## Date: 2026-01-04

## Problem
User-specified `--max-position` was being used as the reduce-only trigger, which is arbitrary. Reduce-only mode should trigger based on dynamic exchange-derived liquidation proximity.

## Solution
Replaced static `max_position` trigger with dynamic calculation based on Hyperliquid's `liquidation_px` API field.

### Key Formulas (from Hyperliquid docs)
```
maintenance_margin = 0.5 × initial_margin_at_max_leverage
maintenance_margin_rate = 1 / (2 × max_leverage)

distance_to_liq = |mark_price - liquidation_px| / mark_price
buffer_ratio = distance / (distance + maintenance_margin_rate)
  - 0.0 = at liquidation
  - 1.0 = very safe

TRIGGER: reduce-only when buffer_ratio < threshold (default 0.5)
```

### Files Modified

**1. `src/market_maker/infra/margin.rs`**
- Added fields to `MarginState`:
  - `liquidation_price: Option<f64>` - from exchange API
  - `mark_price: f64`
  - `position_size: f64`
  - `maintenance_margin_rate: f64` - calculated as `0.5 / max_leverage`
- Added methods:
  - `from_values_with_liquidation()` - full constructor with liquidation data
  - `distance_to_liquidation()` - `|mark - liq| / mark`
  - `liquidation_buffer_ratio()` - normalized 0-1 safety metric
  - `maintenance_margin_required()` - `|pos| × price × rate`
  - `margin_cushion()` and `margin_cushion_ratio()`
  - `is_approaching_liquidation(threshold)` - threshold check
- Added `update_state_with_liquidation()` to `MarginAwareSizer`

**2. `src/market_maker/quoting/filter.rs`**
- Added `ReduceOnlyReason::ApproachingLiquidation` variant (highest priority)
- Added fields to `ReduceOnlyConfig`:
  - `liquidation_price: Option<f64>`
  - `liquidation_buffer_ratio: Option<f64>`
  - `liquidation_trigger_threshold: f64` (default 0.5)
- Added `DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD` constant
- Updated priority order in all filter functions:
  1. **ApproachingLiquidation** (NEW - highest priority)
  2. OverMarginUtilization
  3. OverValueLimit
  4. OverPositionLimit (fallback only)
- Added logging for liquidation-triggered reduce-only

**3. `src/market_maker/quoting/mod.rs`**
- Re-exported `DEFAULT_LIQUIDATION_TRIGGER_THRESHOLD`

**4. `src/market_maker/mod.rs`**
- Updated `refresh_margin_state()` to extract `liquidation_px` from API response:
  - Parses from `user_state.asset_positions[].position.liquidation_px`
  - Calls `update_state_with_liquidation()` instead of `update_state()`
- Updated both `ReduceOnlyConfig` constructions to include new fields

### Backward Compatibility
- If `liquidation_px` is None (no position, API issue): falls back to existing triggers
- User's `--max-position` still works as a hard cap override
- Existing margin utilization (80%) trigger remains as secondary check

## Verification
- `cargo build` ✓
- `cargo clippy -- -D warnings` ✓

## Priority Order After Fix
1. **ApproachingLiquidation** - `buffer_ratio < 0.5` (exchange-derived)
2. **OverMarginUtilization** - `margin_used/account_value > 80%`
3. **OverValueLimit** - `position_value > max_position_value`
4. **OverPositionLimit** - `position > max_position` (fallback only)
