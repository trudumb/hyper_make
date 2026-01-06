# Session: Two-Sided Margin Allocation Fix

## Date: 2026-01-06

## Problem
Order rejections with "Insufficient margin" errors despite having available margin. The market maker was trying to allocate ~$700 of margin when only ~$400 was available.

### Log Evidence
```
margin_available=397.49, leverage=10.0, microprice=27.11
→ margin_quoting_capacity = 397.49 × 10 / 27.11 = 146.6 contracts

Order rejected: sz=65.19 → error=Insufficient margin to place order
```

## Root Cause
**Both bid and ask optimizers independently received the FULL available_margin**, allowing each to allocate 100% of margin. When orders were placed on both sides simultaneously, total margin required exceeded available by ~2x.

### Bug Location
`src/market_maker/strategy/ladder_strat.rs` around lines 769-775 (bids) and 872-878 (asks):

```rust
// BUG: Both optimizers got full available_margin
let mut entropy_optimizer = EntropyConstrainedOptimizer::new(
    entropy_config,
    market_params.microprice,
    available_margin,  // ← FULL margin for BOTH bids AND asks!
    available_for_bids,
    leverage,
);
```

### Mathematical Analysis
- Account: $397.49 margin
- Per-side capacity (buggy): 397.49 × 10 / 27.11 = 146.6 contracts per side
- Total required: 146.6 contracts × 2 sides × $27.11 / 10 = ~$795
- Available: $397.49
- **Result**: Order rejection

## Solution
Implemented **inventory-weighted margin split** between bid and ask sides:

1. When LONG (positive inventory): allocate more margin to asks (reduce position faster)
2. When SHORT (negative inventory): allocate more margin to bids (reduce position faster)
3. When FLAT: 50/50 split

### Key Code Changes

**Added margin split calculation** (lines 545-581):
```rust
// Inventory-weighted margin split
let inventory_ratio = (position / effective_max_position).clamp(-1.0, 1.0);
const MARGIN_SPLIT_SENSITIVITY: f64 = 0.2;
let ask_margin_weight = (0.5 + inventory_ratio * MARGIN_SPLIT_SENSITIVITY).clamp(0.3, 0.7);
let bid_margin_weight = 1.0 - ask_margin_weight;
let margin_for_bids = available_margin * bid_margin_weight;
let margin_for_asks = available_margin * ask_margin_weight;
```

**Updated optimizers to use split margins**:
- Bid optimizer: `margin_for_bids` (line 772)
- Ask optimizer: `margin_for_asks` (line 875)
- Legacy bid optimizer: `margin_for_bids` (line 707)
- Legacy ask optimizer: `margin_for_asks` (line 716)

## First-Principles Justification

From GLFT/Kelly stochastic control theory:
- **Margin is a SHARED resource** between bid and ask sides
- **Position limits are ASYMMETRIC** (separate for each side based on current position)
- **Optimal allocation favors inventory reduction** to minimize holding risk

The inventory-weighted split allocates:
- Range: 30% to 70% per side (never fully one-sided to maintain two-sided quoting)
- When long: 30% to bids, 70% to asks (encourages selling)
- When short: 70% to bids, 30% to asks (encourages buying)
- When flat: 50/50 split

## Files Modified

| File | Change |
|------|--------|
| `src/market_maker/strategy/ladder_strat.rs:545-581` | Added inventory-weighted margin split calculation with INFO logging |
| `src/market_maker/strategy/ladder_strat.rs:772` | Bid entropy optimizer uses `margin_for_bids` |
| `src/market_maker/strategy/ladder_strat.rs:875` | Ask entropy optimizer uses `margin_for_asks` |
| `src/market_maker/strategy/ladder_strat.rs:707` | Legacy bid optimizer uses `margin_for_bids` |
| `src/market_maker/strategy/ladder_strat.rs:716` | Legacy ask optimizer uses `margin_for_asks` |

## Verification
- `cargo build` ✓
- `cargo test --lib entropy_optimizer` (7 tests pass) ✓

## Expected Behavior After Fix
- With $397.49 margin and 50/50 split:
  - Bids: ~$199 margin → ~73 contracts
  - Asks: ~$199 margin → ~73 contracts
  - Total: ~146 contracts (same capacity, but distributed correctly)
- No more "Insufficient margin" rejections for normal two-sided quoting
- Inventory-weighted allocation provides risk-aware position reduction

## Related Sessions
- `session_2026-01-04_controller_derived_position_sizing` - Introduced margin_quoting_capacity
- `session_2026-01-02_stochastic_constraints` - Stochastic constraint implementation
- `tight_spread_first_principles` - Mathematical foundations
