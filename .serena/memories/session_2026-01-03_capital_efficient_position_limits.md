# Session: Capital-Efficient Position Limits

**Date**: 2026-01-03
**Duration**: ~45 minutes
**Status**: ✅ Complete

## Problem Statement

Position runaway kill switch triggered at -16.52 HYPE despite margin allowing ~400 HYPE. Root cause: arbitrary `max_position` config value (0.5) was auto-increased to 8.0 for ladder support, breaking reduce-only protection.

User insight: "Why have arbitrary max_position? Goal is to capture as much spread with the collateral available that our models let via utility constraint."

## Theoretical Foundation: GLFT Optimal Control

From Guéant-Lehalle-Fernandez-Tapia (2013):
```
max E[W_T] - γ ∫₀ᵀ q²(t) dt
subject to: Margin(q(t)) ≤ Collateral  ∀t
```

Two fundamentally different constraints:
| Constraint | Type | Source | Role |
|------------|------|--------|------|
| **Margin** | Hard/Physical | Collateral × Leverage | Solvency |
| **Utility (γ)** | Soft/Economic | Risk preference | Position comfort |
| ~~max_position~~ | ~~Arbitrary~~ | ~~User config~~ | ~~Should NOT exist~~ |

## Solution: Capital-First Design

### 1. Position Limits from Capital (Not User Config)

```rust
// TRUE hard constraint: what margin allows
let margin_based_limit = (account_value * leverage * 0.5) / mark_price;

// User can only REDUCE below margin, not increase
let effective_max = match max_position_override {
    Some(user_limit) => user_limit.min(margin_based_limit),
    None => margin_based_limit,  // Capital-efficient default
};
```

### 2. Reduce-Only Based on Margin Utilization

```rust
const MARGIN_UTILIZATION_THRESHOLD: f64 = 0.8;  // 80%

let margin_utilization = margin_used / account_value;
if margin_utilization > MARGIN_UTILIZATION_THRESHOLD {
    // Reduce-only mode - only allow position-reducing orders
}
```

### 3. Kill Switch Uses Margin-Based Limits

```rust
fn check_position_runaway(&self, state: &KillSwitchState) -> Option<KillReason> {
    let margin_based_limit = (state.account_value * state.leverage * 0.5) / state.mid_price;
    let runaway_threshold = margin_based_limit * 2.0;  // 2× = 100% of margin
    
    if state.position.abs() > runaway_threshold {
        Some(KillReason::PositionRunaway { ... })
    }
}
```

## Files Modified

| File | Changes |
|------|---------|
| `src/bin/market_maker.rs` | Made `max_position` optional, margin-based default |
| `src/market_maker/quoting/filter.rs` | Added `OverMarginUtilization` reason, 80% threshold |
| `src/market_maker/risk/kill_switch.rs` | Added margin fields to state, margin-based runaway |
| `src/market_maker/mod.rs` | Updated ReduceOnlyConfig and KillSwitchState creation |
| `src/market_maker/infra/margin.rs` | Added `max_leverage()` getter |

## Key Code Locations

- **Position limit calculation**: `src/bin/market_maker.rs:920-1007`
- **Reduce-only logic**: `src/market_maker/quoting/filter.rs:115-200`
- **Kill switch runaway check**: `src/market_maker/risk/kill_switch.rs:498-540`
- **Config structs**: `src/market_maker/quoting/filter.rs:58-72` (ReduceOnlyConfig)

## Risk Control Philosophy

| Control | Type | Purpose |
|---------|------|---------|
| **Margin** | Hard constraint | Solvency - can't exceed collateral |
| **Gamma (γ)** | Soft constraint | Controls position comfort via inventory penalty |
| **Reduce-only** | Safety valve | Triggers at 80% margin utilization |
| **Kill switch** | Emergency | Triggers at 2× margin capacity |

## Test Results

All 541 tests pass. Build successful.

## Related Sessions

- `session_2026-01-03_multi_level_ladder_fix` - Initial ladder fix that exposed the problem
- `session_2026-01-02_stochastic_constraints` - Stochastic optimization foundations

## Tags

#capital-efficiency #margin #glft #position-limits #kill-switch #reduce-only
