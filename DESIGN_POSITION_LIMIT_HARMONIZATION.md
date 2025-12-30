# Design: First-Principles Position Limit Harmonization

**Date**: 2025-12-30
**Status**: Design Complete
**Author**: Claude (Opus 4.5)

---

## Executive Summary

This design eliminates the static `config.max_position` as a runtime limit, replacing ALL position limit logic with first-principles derived limits based on account equity, volatility, and leverage constraints.

### The Problem

Two parallel position limit systems exist:
- **Static**: `config.max_position = 0.05` (arbitrary, not equity-aware)
- **Dynamic**: `market_params.dynamic_max_position = 0.283` (first-principles derived)

The **reduce-only filter** uses static (0.05), while **strategies** use dynamic (0.283), causing:
- Position grows to 0.088 BTC (within dynamic limit)
- Reduce-only triggers at 0.05 (incorrect)
- One-sided quoting, unable to trade properly

### The Solution

1. **Remove static limit from runtime decisions** - `config.max_position` becomes ONLY a fallback during warmup
2. **Cache effective limit in MarketMaker** - Updated each quote cycle from first-principles calculation
3. **All consumers use cached effective limit** - Single source of truth

---

## Architecture

### Current Flow (Problematic)

```
                   ┌──────────────────────────┐
                   │  config.max_position     │
                   │  (static: 0.05)          │
                   └─────────┬────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ReduceOnlyConf │   │ MessageContext│   │ UserFillsState│
│ max_pos: 0.05 │   │ max_pos: 0.05 │   │ max_pos: 0.05 │
└───────────────┘   └───────────────┘   └───────────────┘
        │
        ▼
  REDUCE-ONLY TRIGGERS AT 0.05 ✗

                   ┌──────────────────────────┐
                   │ market_params.dynamic_   │
                   │ max_position (0.283)     │
                   └─────────┬────────────────┘
                             │
                             ▼
               ┌───────────────────────┐
               │ LadderStrategy/GLFT   │
               │ uses effective: 0.283 │
               └───────────────────────┘
                             │
                             ▼
                   SIZES ORDERS FOR 0.283 ✓
```

### New Flow (Harmonized)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FIRST-PRINCIPLES CALCULATION                      │
│  calculate_dynamic_max_position_value(                              │
│      account_value,                                                  │
│      sigma,                                                          │
│      time_horizon,                                                   │
│      sigma_confidence,                                               │
│      &dynamic_risk_config                                            │
│  )                                                                   │
│                                                                      │
│  leverage_limit = account_value × max_leverage                       │
│  volatility_limit = (equity × risk_fraction) / (num_sigmas × σ × √T)│
│  max_position_VALUE = min(leverage_limit, volatility_limit)          │
│  max_position_SIZE = max_position_VALUE / mid_price                  │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │ MarketMaker.effective_max_pos │
             │ (cached, updated each cycle)  │
             └───────────────┬───────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
     ▼                       ▼                       ▼
┌────────────┐      ┌────────────────┐      ┌───────────────┐
│ReduceOnly  │      │ MessageContext │      │ UserFillsState│
│ Config     │      │                │      │               │
└────────────┘      └────────────────┘      └───────────────┘
     │                       │                       │
     └───────────────────────┼───────────────────────┘
                             │
                             ▼
              ALL USE SAME EFFECTIVE LIMIT ✓
```

---

## Detailed Design

### 1. New Field in MarketMaker

**File**: `src/market_maker/mod.rs`

```rust
pub struct MarketMaker<S, E> {
    // ... existing fields ...

    /// Effective max position SIZE (contracts), updated each quote cycle.
    ///
    /// Derived from first principles:
    /// - During warmup: Uses config.max_position as fallback
    /// - After warmup: Uses dynamic_max_position_value / mid_price
    ///
    /// This is THE source of truth for all position limit checks.
    /// NEVER use config.max_position directly for runtime decisions.
    effective_max_position: f64,
}
```

**Initialization**:
```rust
impl<S, E> MarketMaker<S, E> {
    pub fn new(...) -> Self {
        Self {
            // ...
            effective_max_position: config.max_position, // Fallback during warmup
        }
    }
}
```

### 2. Update Logic in Quote Cycle

**File**: `src/market_maker/mod.rs` - `update_quotes()`

```rust
async fn update_quotes(&mut self) -> Result<()> {
    // ... build market_params ...

    // CRITICAL: Update cached effective limit from first principles
    // This happens EVERY quote cycle to reflect current equity/volatility
    self.effective_max_position = market_params.effective_max_position(self.config.max_position);

    // ... rest of function uses self.effective_max_position ...
}
```

### 3. Consumer Updates

All consumers that currently use `self.config.max_position` will use `self.effective_max_position`:

| Location | Current | New |
|----------|---------|-----|
| `mod.rs:658` - UserFillsState | `self.config.max_position` | `self.effective_max_position` |
| `mod.rs:701` - MessageContext | `self.config.max_position` | `self.effective_max_position` |
| `mod.rs:827` - ReduceOnlyConfig (ladder) | `self.config.max_position` | `self.effective_max_position` |
| `mod.rs:871` - ReduceOnlyConfig (single) | `self.config.max_position` | `self.effective_max_position` |

### 4. Exception: Strategy Inputs

Lines 805 and 863 (strategy `calculate_ladder` and `calculate_quotes`) should **continue using `config.max_position`** because:
- Strategies internally call `market_params.effective_max_position(static_fallback)`
- They need the fallback value to implement the same logic

```rust
// KEEP AS-IS (strategies handle effective internally)
let ladder = self.strategy.calculate_ladder(
    &quote_config,
    self.position.position(),
    self.config.max_position,  // <-- KEEP: strategy uses as fallback
    self.config.target_liquidity,
    &market_params,
);
```

### 5. MessageContext Enhancement

**File**: `src/market_maker/messages/context.rs`

The `MessageContext` receives `max_position` from the caller. No changes to the struct needed, but callers must pass the effective limit:

```rust
// In mod.rs - handle_l2_book(), handle_trades(), etc.
let ctx = messages::MessageContext::new(
    self.config.asset.clone(),
    self.latest_mid,
    self.position.position(),
    self.effective_max_position,  // Changed from self.config.max_position
    self.estimator.is_warmed_up(),
);
```

### 6. Update safety_sync()

**File**: `src/market_maker/mod.rs` - `safety_sync()`

Line 1801 also uses `self.config.max_position` for reduce-only check:

```rust
let (reduce_only, reason) = safety::SafetyAuditor::check_reduce_only(
    position,
    position_value,
    self.effective_max_position,  // Changed from self.config.max_position
    max_position_value,
);
```

---

## First-Principles Derivation (Reference)

The dynamic position limit is derived from two constraints:

### Leverage Constraint
```
position_value ≤ account_value × max_leverage
```
This is the **hard ceiling** from exchange rules.

### Volatility Constraint (VaR-based)
```
position_value ≤ (equity × risk_fraction) / (num_sigmas × σ × √T)

where:
- risk_fraction = 0.5 (50% of equity at risk in 5-sigma move)
- num_sigmas = 5.0 (99.99997% confidence)
- σ = per-second volatility from bipower variation
- T = expected holding time (1 / arrival_intensity)
```

### Final Limit
```
max_position_value = min(leverage_limit, volatility_limit)
max_position_SIZE = max_position_value / mid_price
```

### Bayesian Regularization
When volatility estimate has low confidence, we blend with prior:
```
σ_effective = confidence × σ_observed + (1 - confidence) × σ_prior
```

---

## Configuration Changes

### Deprecate `config.max_position` Runtime Usage

The field remains for:
1. **Warmup fallback** - Before margin state is fetched
2. **Strategy fallback** - Internal to effective_max_position()
3. **CLI default** - User-friendly starting point

But it is **NEVER used directly for runtime limit enforcement** after warmup.

### New Logging

Add DEBUG log when effective limit changes significantly:

```rust
if (new_effective - self.effective_max_position).abs() > 0.01 {
    debug!(
        old = %format!("{:.6}", self.effective_max_position),
        new = %format!("{:.6}", new_effective),
        dynamic_valid = market_params.dynamic_limit_valid,
        "Effective max position updated"
    );
}
self.effective_max_position = new_effective;
```

---

## Validation Criteria

After implementation, verify:

1. **Single Source of Truth**
   - `mm_inventory_utilization` metric = `|position| / effective_max_position`
   - All modules see same limit value

2. **No False Reduce-Only**
   - Reduce-only only triggers when position exceeds EFFECTIVE limit
   - No spurious triggers at old static threshold

3. **Dynamic Scaling Works**
   - Larger account → larger position limit
   - Higher volatility → smaller position limit
   - Leverage cap respected

4. **Safe Fallback During Warmup**
   - Before first margin refresh: uses static fallback
   - After margin refresh: uses dynamic limit

---

## Migration Path

### Phase 1: Add Cached Field (This PR)
1. Add `effective_max_position` field to MarketMaker
2. Initialize to `config.max_position`
3. Update in `update_quotes()` from `market_params.effective_max_position()`

### Phase 2: Update Consumers (This PR)
1. Change ReduceOnlyConfig construction (2 places)
2. Change MessageContext construction (3 places)
3. Change UserFillsState construction (1 place)
4. Change safety_sync() reduce-only check (1 place)

### Phase 3: Verify (This PR)
1. Run test suite (404 tests)
2. Run with DEBUG logging
3. Verify metrics show correct utilization

### Phase 4: Future Cleanup
1. Consider renaming `config.max_position` to `max_position_fallback`
2. Add deprecation warning in config parsing
3. Document that runtime uses dynamic limits only

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/market_maker/mod.rs` | Add field, update in update_quotes(), change 7 usages |
| `src/market_maker/messages/context.rs` | No changes (receives value from caller) |
| `src/market_maker/fills/processor.rs` | No changes (receives value from caller) |
| `src/market_maker/quoting/filter.rs` | No changes (receives value from caller) |

Total: **~20 lines changed** in one file.

---

## Test Plan

1. **Unit Tests**: Existing 404 tests should pass unchanged
2. **Integration Test**: Run market maker with DEBUG logging
3. **Verify Logs**:
   - `effective_max_pos` in quote inputs should equal dynamic when valid
   - Reduce-only warnings should use effective limit
   - Position utilization should be calculated against effective limit

---

## Appendix: Complete Usage Map

### Uses `config.max_position` (CHANGE to `effective_max_position`):
- `mod.rs:658` - UserFillsState.max_position
- `mod.rs:701` - MessageContext.max_position
- `mod.rs:827` - ReduceOnlyConfig.max_position (ladder)
- `mod.rs:871` - ReduceOnlyConfig.max_position (single)
- `mod.rs:1801` - safety_sync reduce-only check
- `mod.rs:1892/1899` - capacity checks (consider changing)

### Uses `config.max_position` (KEEP as strategy fallback):
- `mod.rs:760` - ParameterSources.max_position (input to market_params)
- `mod.rs:805` - strategy.calculate_ladder() (strategy handles internally)
- `mod.rs:863` - strategy.calculate_quotes() (strategy handles internally)
- `mod.rs:1957` - another strategy call

### Uses `market_params.effective_max_position()` (ALREADY CORRECT):
- Strategy internals (ladder_strat.rs, glft.rs)
- Logging at mod.rs:778
