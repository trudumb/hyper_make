# Session: Warmup Bypass Fix for Adaptive Bayesian System

## Date: 2026-01-02

## Problem Solved
The adaptive Bayesian market making system had a **chicken-and-egg warmup problem**:
- Adaptive system required 20+ fills to "warm up" before using adaptive values
- During warmup, fell back to legacy multiplicative gamma → wide spreads (20+ bps)
- Wide spreads = few fills = system never warms up

## Solution: Use Adaptive Values Immediately via Bayesian Priors

The key insight: **Our Bayesian priors already provide reasonable starting values.** We were unnecessarily gating their use behind warmup completion.

### Changes Made

#### 1. `src/market_maker/adaptive/calculator.rs`
Added three new methods:
```rust
pub fn can_provide_estimates(&self) -> bool {
    true  // Always - priors work immediately
}

pub fn warmup_progress(&self) -> f64 {
    // 0.0 to 1.0 based on floor/kappa/gamma progress
    floor_progress * 0.4 + kappa_progress * 0.4 + gamma_progress * 0.2
}

pub fn warmup_uncertainty_factor(&self) -> f64 {
    // Starts at 1.2 (20% wider), decays to 1.0
    1.0 + (1.0 - self.warmup_progress()) * 0.2
}
```

#### 2. `src/market_maker/strategy/market_params.rs`
Added new fields:
- `adaptive_can_estimate: bool` → TRUE immediately (use priors)
- `adaptive_warmup_progress: f64` → 0% to 100%
- `adaptive_uncertainty_factor: f64` → 1.2 → 1.0

#### 3. `src/market_maker/strategy/glft.rs`
Changed all gating from `adaptive_warmed_up` to `adaptive_can_estimate`:
- Section 1: Gamma selection
- Section 1a: Kappa selection
- Section 2d: Spread floor
- Added Section 2f: Warmup uncertainty scaling

## Expected Spread Calculation (From Priors)

Prior-based floor:
```
floor = fees + E[AS] + k×σ_AS
      = 3 bps + 3 bps + 1.5×5 bps = 13.5 bps
```

Prior-based GLFT:
```
δ = (1/γ) × ln(1 + γ/κ) + fees
  = (1/0.3) × ln(1 + 0.3/2500) + 3 bps ≈ 7 bps
```

With uncertainty factor (1.2): **~8.5 bps spreads from startup**

## Behavior Comparison

| Metric | Before | After |
|--------|--------|-------|
| T=0 spreads | 20+ bps (legacy) | ~8-10 bps (priors) |
| Warmup time | Never (chicken-egg) | Progressive |
| First fills | Very rare | Normal rate |

## Key Insight

The distinction matters:
- `is_warmed_up()` → For CONFIDENCE reporting (full calibration after 20+ fills)
- `can_provide_estimates()` → For USAGE decision (always true with priors)

## Files Modified
- `src/market_maker/adaptive/calculator.rs` - New warmup methods
- `src/market_maker/strategy/market_params.rs` - New fields
- `src/market_maker/strategy/params.rs` - Wire new fields
- `src/market_maker/strategy/glft.rs` - Use adaptive_can_estimate
- `src/market_maker/mod.rs` - Updated logging

## Tests
All 38 adaptive tests + 14 GLFT tests pass.

## Related Sessions
- `session_2026-01-01_ws_order_state_integration.md` - Previous WS work
- `design_tight_spread_system.md` - Tight spread architecture
- `session_checkpoint_profitability_fixes.md` - Earlier profitability work
