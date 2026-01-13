# Session: 2026-01-13 L3 Stochastic Controller Proper Fixes

## Summary
Replaced band-aid fixes with first-principles solutions for the Layer 3 stochastic controller.
All changes address root causes rather than suppressing symptoms.

## Changes Made

### 1. WaitToLearn Orchestrator Integration (P0)
**File:** `src/market_maker/orchestrator/quote_engine.rs:545-567`

**Problem:** WaitToLearn action was returned but never handled - orders stayed on book indefinitely.

**Fix:** Cancel all resting orders when WaitToLearn is triggered:
```rust
Action::WaitToLearn { ... } => {
    let bid_orders = self.orders.get_all_by_side(Side::Buy);
    let ask_orders = self.orders.get_all_by_side(Side::Sell);
    let resting_oids: Vec<u64> = bid_orders.iter().chain(ask_orders.iter()).map(|o| o.oid).collect();
    if !resting_oids.is_empty() {
        self.executor.cancel_bulk_orders(&self.config.asset, resting_oids).await;
    }
    return Ok(());
}
```

### 2. Entropy-Based Warmup (P1)
**File:** `src/market_maker/control/changepoint.rs:170-196`

**Problem:** Magic number 20 for warmup observations, not derived from data quality.

**Fix:** Hybrid approach using entropy:
- Minimum 10 observations (escape degenerate r=0 state)
- Then: entropy < threshold (distribution concentrated) OR hard cap reached
- Added `warmup_entropy_threshold: f64` to config (default: 2.0)
- Added `max_run_length()` getter for entropy normalization

**First-principles:** BOCD needs time for run length distribution to spread and reconcentrate. 
Entropy naturally measures this: high entropy = uncertain, low entropy = concentrated.

### 3. Epistemicity Ratio for Disagreement (P2)
**File:** `src/market_maker/learning/decision.rs:123-144`

**Problem:** Original check `disagreement > 1.5 × std` was mathematically impossible.
By variance decomposition: `std² = weighted_var + disagreement²`, so `disagreement ≤ std` always.

**Fix:** Use valid epistemicity ratio:
```rust
let epistemicity_ratio = ensemble.disagreement / ensemble.std.max(0.001);
if epistemicity_ratio > 0.5 {  // >50% of uncertainty is model disagreement
    return QuoteDecision::ReducedSize { fraction: 0.5, ... };
}
```

**First-principles:** Epistemicity ratio ∈ [0, 1] measures what fraction of total uncertainty comes from model disagreement (epistemic) vs within-model uncertainty (aleatoric).

### 4. Principled Learning Trust (P1.5)
**File:** `src/market_maker/control/mod.rs:385-439`

**Problem:** Magic 0.7 trust floor during warmup, not derived from information theory.

**Fix:** Entropy-based trust scaling:
```rust
let normalized_entropy = (entropy / max_entropy).clamp(0.0, 1.0);
let changepoint_factor = if !self.changepoint.is_warmed_up() {
    // Scale from 0.5 to 1.0 as entropy decreases
    0.5 + 0.5 * (1.0 - normalized_entropy)
} else {
    // Post-warmup: use changepoint probability
    (1.0 - changepoint_prob).max(0.1)
};
```

**First-principles:** Trust should reflect confidence in regime stability.
Entropy measures how confident BOCD is about run length (regime age).

## Files Modified

| File | Changes |
|------|---------|
| `orchestrator/quote_engine.rs:545-567` | WaitToLearn cancels resting orders |
| `control/changepoint.rs:85-90,113-136,166,170-200` | Entropy-based warmup config/implementation |
| `control/changepoint.rs:469-547` | Updated tests for entropy-based warmup |
| `learning/decision.rs:123-144` | Epistemicity ratio instead of impossible threshold |
| `learning/decision.rs:250-282` | Updated test for permissive defaults |
| `control/mod.rs:385-439` | Entropy-based trust scaling |

## Verification

```bash
cargo build           # ✅ Passed
cargo test            # ✅ 821 tests passed
cargo test changepoint # ✅ 7 tests passed
cargo test decision   # ✅ 9 tests passed  
cargo test control    # ✅ 60 tests passed
```

## Expected Log Behavior

Before fixes (band-aids):
```
l3_trust":"0.70"  # Magic constant
```

After fixes (principled):
```
l3_trust":"0.52"→"0.78"→"0.95"  # Scales with entropy concentration
```

When WaitToLearn triggers:
```
WaitToLearn: cancelling resting orders to prevent stale quotes (count=12)
```

When epistemicity is high:
```
High epistemic uncertainty: 65% of std from model disagreement
```

## Pre-existing Clippy Issues (Not Addressed)

The following clippy issues existed before this session and are unrelated to L3 fixes:
- `large_enum_variant` in actions.rs (Ladder struct is 1224 bytes)
- `excessive_precision` in changepoint.rs (gamma_ln coefficients)
- `unnecessary_cast` in belief.rs
- `deprecated` warnings for ConstrainedLadderOptimizer in optimizer.rs tests

## Related Memories
- `session_2026-01-12_l3_warmup_fixes` - Original band-aid fixes being replaced
- `session_2026-01-12_stochastic_controller_layer3` - L3 architecture overview
