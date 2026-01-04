# Stochastic Parameters Deep Dive: Why Gamma=0.5 and Kappa=2000 Never Update

## Executive Summary

After tracing the complete data flow through the market maker codebase, I've identified **critical bottlenecks** that prevent the Bayesian parameters from updating dynamically. The parameters ARE designed to update, but there are **implementation gaps** preventing data from reaching the estimators.

---

## Root Cause Analysis

### 1. KAPPA = 2000 (Static)

**Direct Cause:** `kappa_prior_mean × warmup_factor = 2500 × 0.8 = 2000`

**File:** `src/market_maker/adaptive/blended_kappa.rs:304-314`

```rust
pub fn kappa(&self) -> f64 {
    let w = self.blend_weight();
    let blended = (1.0 - w) * self.book_kappa + w * self.own_kappa();

    // Apply warmup conservatism if still early
    if self.own_fill_count < self.blend_min_fills {  // < 10 fills
        blended * self.warmup_factor  // × 0.8
    } else {
        blended
    }
}
```

**Why it stays at 2000:**

| Component | Value | Reason |
|-----------|-------|--------|
| `own_fill_count` | 0 | No fills reaching BlendedKappaEstimator |
| `blend_weight()` | ~0.12 | Sigmoid(0-10/5) is low |
| `book_kappa` | 2500 | **L2 updates not flowing to BlendedKappa** |
| `own_kappa` | 2500 | No own fills recorded |
| `blended` | 2500 | 0.88×2500 + 0.12×2500 |
| **Output** | **2000** | 2500 × 0.8 warmup_factor |

### 2. GAMMA = 0.500 (Static)

**Direct Cause:** Legacy multiplicative path with thin book scaling

**File:** `src/market_maker/strategy/ladder_strat.rs:306-322`

```rust
let gamma = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
    // Adaptive path (NOT being used or not warmed up)
    adaptive_gamma * tail_risk_multiplier
} else {
    // Legacy path (BEING USED)
    let base_gamma = self.effective_gamma(market_params, position, effective_max_position);
    let gamma_with_liq = base_gamma * market_params.liquidity_gamma_mult;
    gamma_with_liq * market_params.tail_risk_multiplier
};
```

**Calculation breakdown (from your screenshot: `book_depth_usd = "7996"`):**

```
gamma_base = 0.3 (from RiskConfig)
book_depth_multiplier = 1 + 0.5 × (1 - 8000/50000) = 1.42 (thin book!)
effective_gamma = 0.3 × 1.42 = 0.426
liquidity_gamma_mult ≈ 1.17 (from book structure)
gamma = 0.426 × 1.17 × 1.0 ≈ 0.50
```

**Why it stays at 0.5:**
- The thin book ($8k vs $50k threshold) triggers protective scaling
- These multipliers are **recomputed each cycle** but the underlying market conditions (thin book) don't change
- Adaptive gamma isn't being used (either disabled OR `ShrinkageGamma.is_warmed_up() = false`)

---

## Critical Bottlenecks Identified

### Bottleneck #1: BlendedKappaEstimator Never Receives L2 Updates

**Symptom:** `book_kappa` stays at prior (2500)

**Evidence:** Searching for `adaptive_spreads.on_l2` returns NO matches:
```bash
grep -r "adaptive_spreads.on_l2\|adaptive_spreads.kappa_mut" src/market_maker/
# No matches found
```

**The `BlendedKappaEstimator.on_l2_update()` method exists but is NEVER called:**

```rust
// src/market_maker/adaptive/blended_kappa.rs:183-203
pub fn on_l2_update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
    // This estimates kappa from L2 book structure
    // But it's NEVER CALLED from the market maker!
}
```

**Compare with ParameterEstimator which DOES receive L2:**
```rust
// src/market_maker/messages/l2_book.rs:84
state.estimator.on_l2_book(&bids, &asks, ctx.latest_mid);  // ✓ Called
// state.adaptive_spreads.on_l2_update(...);  // ✗ Missing!
```

### Bottleneck #2: Adaptive Fill Processing Gated by Config

**File:** `src/market_maker/mod.rs:911`

```rust
if self.stochastic.stochastic_config.use_adaptive_spreads {  // Must be true!
    for fill in &user_fills.data.fills {
        // ... process fills for adaptive system
        self.stochastic.adaptive_spreads.on_fill_simple(...);
    }
}
```

If `use_adaptive_spreads = false`, the `AdaptiveSpreadCalculator` NEVER receives fill data.

**Note:** Config defaults to `true` (`config.rs:568`), but MarketParams defaults to `false` (`market_params.rs:479`). The actual value depends on which code path populates the struct.

### Bottleneck #3: ShrinkageGamma Warmup Gate

**File:** `src/market_maker/adaptive/calculator.rs:484-489`

```rust
pub fn gamma(&self, gamma_base: f64, ...) -> f64 {
    if self.config.enable_shrinkage_gamma && self.gamma.is_warmed_up() {
        self.last_gamma  // Learned gamma
    } else {
        gamma_base  // Returns 0.3 unchanged!
    }
}
```

**The `ShrinkageGamma` module has its own warmup requirement:**
- Needs ~20 standardized observations
- Until warmed up, just returns `gamma_base` (0.3)
- The 0.5 you see comes from the LEGACY multiplicative path, not the adaptive system

### Bottleneck #4: Two Separate Kappa Systems (Not Synchronized)

There are **TWO independent kappa estimators**:

| System | Location | Data Source | Used By |
|--------|----------|-------------|---------|
| ParameterEstimator.kappa() | `estimator/parameter_estimator.rs` | Trades, L2 book, Own fills | Legacy path |
| BlendedKappaEstimator.kappa() | `adaptive/blended_kappa.rs` | **Nothing currently** | Adaptive path |

The ParameterEstimator IS receiving data correctly:
```rust
// Trades → market_kappa
state.estimator.on_trade(...)  // ✓ Working

// Own fills → own_kappa
state.estimator.on_own_fill(...)  // ✓ Working

// L2 book → book structure
state.estimator.on_l2_book(...)  // ✓ Working
```

But the BlendedKappaEstimator is **isolated**:
```rust
// adaptive_spreads.kappa.on_l2_update(...)  // ✗ Never called
// adaptive_spreads.on_fill(...)  // Only if use_adaptive_spreads=true
```

---

## Data Flow Diagrams

### Current Flow (Broken)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
└──────────────────────┬────────────────────┬─────────────────────────┘
                       │                    │
                   Trades              L2 Book
                       │                    │
                       ▼                    ▼
          ┌─────────────────────────────────────────────┐
          │         ParameterEstimator                  │  ✓ RECEIVING DATA
          │  - market_kappa (from trades)              │
          │  - own_kappa (from fills)                  │
          │  - book_structure (from L2)                │
          └─────────────────────────────────────────────┘
                       │
                       ▼
          ┌─────────────────────────────────────────────┐
          │         MarketParams.kappa                  │  ✓ POPULATED
          │         MarketParams.sigma                  │
          │         (from ParameterEstimator)           │
          └─────────────────────────────────────────────┘


          ┌─────────────────────────────────────────────┐
          │      AdaptiveSpreadCalculator               │  ✗ ISOLATED
          │  - BlendedKappaEstimator                    │
          │    └─ book_kappa = STUCK AT 2500           │
          │    └─ own_kappa = STUCK AT 2500            │
          │  - ShrinkageGamma                           │
          │    └─ NOT WARMED UP                        │
          └─────────────────────────────────────────────┘
                       │
                       ▼
          ┌─────────────────────────────────────────────┐
          │         MarketParams.adaptive_kappa         │  = 2000 (STATIC)
          │         MarketParams.adaptive_gamma         │  = 0.3 (unused)
          └─────────────────────────────────────────────┘


          ┌─────────────────────────────────────────────┐
          │         LadderStrategy                      │
          │                                             │
          │  IF adaptive_mode:                          │
          │    gamma = adaptive_gamma → 0.3             │
          │    kappa = adaptive_kappa → 2000            │
          │                                             │
          │  ELSE (legacy):                             │  ← CURRENT PATH
          │    gamma = effective_gamma() → 0.5          │
          │    kappa = estimator.kappa() × (1-α)        │
          └─────────────────────────────────────────────┘
```

### Expected Flow (Fixed)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
└──────────────────────┬────────────────────┬─────────────────────────┘
                       │                    │
                   Trades              L2 Book
                       │                    │
          ┌────────────┼────────────────────┼──────────────┐
          │            │                    │              │
          ▼            ▼                    ▼              ▼
┌───────────────────────────┐    ┌───────────────────────────────────┐
│   ParameterEstimator      │    │   AdaptiveSpreadCalculator         │
│   (for diagnostics)       │    │                                   │
└───────────────────────────┘    │   BlendedKappaEstimator            │
                                 │     └─ on_l2_update(bids, asks)   │ ← FIX
                                 │     └─ on_own_fill(...)           │ ← FIX
                                 │   ShrinkageGamma                   │
                                 │     └─ warm up from signals       │
                                 └───────────────────────────────────┘
                                              │
                                              ▼
                                 ┌───────────────────────────────────┐
                                 │  MarketParams.adaptive_kappa      │ → DYNAMIC
                                 │  MarketParams.adaptive_gamma      │ → DYNAMIC
                                 └───────────────────────────────────┘
```

---

## Required Fixes

### Fix #1: Wire L2 Updates to BlendedKappaEstimator

**File:** `src/market_maker/messages/l2_book.rs`

Add after line 84:
```rust
// Feed L2 data to adaptive spread calculator's kappa estimator
if let Some(adaptive) = &mut state.stochastic.adaptive_spreads {
    adaptive.on_l2_update(&bids, &asks, ctx.latest_mid);
}
```

Or in `mod.rs` after processing L2:
```rust
self.stochastic.adaptive_spreads.on_l2_update(&bids, &asks, self.latest_mid);
```

### Fix #2: Always Feed Fills to Adaptive System

**File:** `src/market_maker/mod.rs:911`

Change from:
```rust
if self.stochastic.stochastic_config.use_adaptive_spreads {
```

To:
```rust
// Always feed fills to adaptive system for learning (even if not using adaptive quotes)
{
```

This allows the adaptive system to learn even when not actively used for quoting.

### Fix #3: Reduce ShrinkageGamma Warmup Requirement

**File:** `src/market_maker/adaptive/shrinkage_gamma.rs`

Find the warmup check and reduce the threshold, or use a progressive approach:
```rust
pub fn is_warmed_up(&self) -> bool {
    // Current: needs ~20 observations
    // Proposed: start with partial learning after 5 observations
    self.observation_count >= 5
}
```

### Fix #4: Unify Kappa Estimators OR Pass Through

**Option A:** Use ParameterEstimator.kappa() as the fallback for BlendedKappa:
```rust
// In calculator.rs kappa() method
pub fn kappa(&self, book_kappa_fallback: f64) -> f64 {
    if self.kappa.own_fill_count() >= self.config.kappa_blend_min_fills {
        self.kappa.kappa()  // Use BlendedKappa when warmed up
    } else {
        book_kappa_fallback  // Use ParameterEstimator's kappa during warmup
    }
}
```

**Option B:** Remove BlendedKappaEstimator and just use ParameterEstimator:
```rust
// In params.rs
adaptive_kappa: sources.estimator.kappa(),  // Use existing estimator
```

---

## Diagnostic Commands

### Check Warmup Status

Add logging to see why adaptive isn't being used:

```bash
RUST_LOG=hyperliquid_rust_sdk::market_maker::strategy::ladder_strat=trace \
  cargo run --bin market_maker -- --asset HYPE --dex hyna
```

Look for:
```
"Ladder using ADAPTIVE gamma"  ← If you see this, adaptive is working
"Ladder spread diagnostics"    ← If no ADAPTIVE log, legacy path is used
```

### Check BlendedKappa State

Add to quote cycle logging:
```rust
info!(
    blended_own_fills = self.stochastic.adaptive_spreads.kappa_mut().own_fill_count(),
    blended_book_updates = self.stochastic.adaptive_spreads.kappa_mut().book_update_count,
    "Adaptive kappa state"
);
```

### Check ShrinkageGamma State

```rust
info!(
    gamma_warmed_up = self.stochastic.adaptive_spreads.gamma_warmed_up(),
    gamma_obs_count = self.stochastic.adaptive_spreads.gamma_observation_count(),
    "Adaptive gamma state"
);
```

---

## Summary

| Parameter | Current Value | Why Static | Fix |
|-----------|---------------|------------|-----|
| kappa | 2000 | BlendedKappa not receiving L2/fills | Wire L2 updates to adaptive system |
| gamma | 0.500 | Using legacy path due to thin book | Either fix adaptive warmup OR accept legacy is correct |

The **core issue** is that the `AdaptiveSpreadCalculator` was designed to be self-contained with its own estimators, but the **data feeds were never connected**. The ParameterEstimator receives all the data and works correctly, but the adaptive system is isolated.

**Recommended Priority:**
1. Fix #1 (L2 → BlendedKappa) - High impact, simple fix
2. Fix #2 (Always feed fills) - Ensures learning happens
3. Fix #4 (Unify kappa sources) - Architectural simplification
4. Fix #3 (Reduce warmup) - Quality of life improvement
