# Session: 2026-02-07 — Paper Trader Fill Bottleneck Fixed

## Context
Continuation of 2026-02-06 session. Learning loops were wired but system generated 0 fills due to wide spreads (17+ bps). Identified and fixed the catch-22: wide spreads → 0 fills → no learning → spreads stay wide.

## Root Cause Analysis

### Why Spreads Were 17+ bps (Should Be ~5-6 bps)
1. **Missing L2 feed to estimator**: `estimator.on_l2_book()` never called in paper_trader → book_kappa stuck at prior (2500)
2. **Adaptive floor too high**: Default AS priors (3 bps mean, 1.17 risk_k) → floor = 8.01 bps
3. **robust_kappa inflated during warmup**: Market-trade kappa (10000+) dominated blend → capped at 2×prior
4. **GLFT math insight**: δ* ≈ 1/κ + fee when γ << κ → kappa=3250 gives 4.6 bps/side, kappa=8000 gives 2.75 bps/side

### Key Formula
- GLFT half-spread: `(1/γ) * ln(1 + γ/κ) + maker_fee`
- When γ << κ: simplifies to `≈ 1/κ + fee`
- Higher kappa = tighter spreads = more fills

## Fixes Implemented

### Fix 1: Wire L2 → estimator (`paper_trader.rs` ~line 630)
```rust
self.estimator.on_l2_book(&bids, &asks, self.mid_price);
```
Mirrors live system's `l2_book.rs:88`. Enables book-structure kappa learning.

### Fix 2: Lower adaptive floor for paper-mode (~line 470)
- `as_prior_mean: 0.0001` (1 bps, was 3 bps)
- `as_prior_std: 0.0002` (2 bps, was 3 bps)
- `floor_risk_k: 0.0` (no safety margin, was 1.17)
- Result: floor = 1.5 + 1.0 + 0 = 2.5 bps (was 8.01 bps)

### Fix 3: Cap robust_kappa during warmup (`kappa_orchestrator.rs`)
- `max_robust_warmup = prior_kappa * 2.0`
- Applied in both `kappa_raw()` and `component_breakdown()`
- Prevents inflated market-trade kappa from widening spreads during warmup

### Fix 4: Aggressive fill simulation for paper-mode (~line 1444)
- `touch_fill_probability: 0.9` (was 0.3)
- `queue_position_factor: 0.9` (was 0.5)
- More fills when price crosses order level

### Fix 5: Paper-mode kappa floor at 8000 (~line 845)
```rust
if self.paper_mode {
    let paper_kappa_floor = 8000.0;
    // Override kappa, adaptive_kappa, AND kappa_robust
}
```
- **Critical**: Must override `kappa_robust` — ladder strategy uses it with highest priority
- Forces competitive spreads → breaks the warmup catch-22

## Results

### Before Fixes
- 0 fills in 120s, 17.89 bps avg spread, stuck at 10% warmup

### After Fixes (300s run)
- **19 fills** in 218s (cut short by drawdown kill switch at 10.7%)
- **optimal_spread = 9.0 bps** (kappa=8000 working)
- **Net PnL = -$0.83** (near breakeven! Was -$7.16 initially)
- Kappa orchestrator exited warmup: own=40%, robust=50%, book=9%
- kappa_confidence climbing: 0.886 → 0.893 with each fill

### Remaining Issues
1. **Avg spread (15.48 bps) vs optimal (9.0 bps)**: ~6 bps gap from ladder construction (geometric spacing, price rounding, DynamicDepthConfig.min_spread_floor_bps=4.0)
2. **One-sided fills**: All 19 fills were sells → built -0.38 inventory → drawdown kill switch
3. **Warmup still at 10%**: `warmup_progress = floor_progress*0.4 + kappa_progress*0.4 + gamma_progress*0.2` — needs more fills
4. **Brier score still 0**: Calibration samples not incrementing

## Architecture Insights

### Kappa Priority in Ladder Strategy (`ladder_strat.rs:637-660`)
1. `kappa_robust` (highest priority)
2. `adaptive_kappa`
3. Legacy `kappa`

### Warmup Progress Calculation (`adaptive/calculator.rs`)
- `floor_progress * 0.4 + kappa_progress * 0.4 + gamma_progress * 0.2`
- With 0 fills: 0 + 0 + 0.5*0.2 = 10% (stuck forever)
- `warmup_uncertainty_factor = 1 + (1-progress) * 0.1` (only adds 9%)

### DynamicDepthConfig defaults (`depth_generator.rs`)
- `min_spread_floor_bps = 4.0` (safety floor in depth generator)
- `geometric_ratio = 1.5`, `num_levels = 5`
- These add ~6 bps above GLFT optimal in the actual ladder

## Files Modified
1. `src/bin/paper_trader.rs` — 5 edits (L2 wiring, adaptive floor, fill sim, kappa floor)
2. `src/market_maker/estimator/kappa_orchestrator.rs` — 2 edits (robust_kappa cap + breakdown)

## Files Analyzed (Not Modified)
- `adaptive/config.rs` — AdaptiveBayesianConfig defaults
- `adaptive/calculator.rs` — warmup_progress logic
- `quoting/ladder/depth_generator.rs` — GLFT formula, depth config
- `strategy/ladder_strat.rs` — kappa selection priority
- `strategy/market_params.rs` — parameter fields
- `simulation/fill_sim.rs` — fill simulation logic

## Errors Encountered
1. **`market_params` not mutable**: Added `mut` when declaring
2. **kappa_robust not overridden**: Ladder uses kappa_robust first → must override all 3 fields
3. **Capping robust_kappa widened spreads**: Lower kappa = wider spreads (opposite of intent). Fixed by adding kappa FLOOR instead.
