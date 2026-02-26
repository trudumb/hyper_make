# Session 2026-02-17: Cold-Start Spread Widening Fixes

## Context
Continuing from previous session that fixed P0-A (queue_factor=0.0) and P0-B (double-decrement fill drop) in the paper fill simulator. Paper trader still got 0 fills — quotes placed at 30+ bps from mid.

## Root Causes Found

### Root Cause #1: Pre-fill AS Multiplier Double-Counting
- **Toxicity applied TWICE**:
  1. `ladder_strat.rs` lines 1272-1281: `pre_fill_spread_mult_bid/ask` multiplies depths (1.5-2.0x)
  2. `quote_engine.rs` lines 2629-2643: `ToxicityRegime::Normal` widens spreads by `(1.0 + bid_tox)` (1.5x)
- **Combined**: 1.98x × 1.5x ≈ 3x total, pushing 7 bps to 21+ bps
- **Fix**: Removed `ToxicityRegime::Normal` spread widening in quote_engine.rs — depth multiplier in ladder_strat.rs already handles it

### Root Cause #2: AS Model Warmup Bypass
- `update_count` in pre_fill_classifier incremented by book/trade events (not fills)
- `warmup_prior_min_updates=10` exceeded before any fills arrive
- Raw uncalibrated signals produce 1.5-2.0x multipliers at cold start
- Enhanced classifier returns 0.5 at low confidence, blended at 30% weight
- **Fix (v1)**: Warmup AS cap — linear ramp from 1.15x→3.0x with warmup progress
- **Fix (v2, this session)**: Reduced cap to 1.0x at cold start (no AS widening until calibrated)

### Root Cause #3: Regime Kappa Domination During Warmup
- Regime kappa blend weight = 60% even at cold start
- Regime classifier uncalibrated → defaults to Normal=2000
- Market-derived kappa from book/robust estimators (potentially 3000-5000 for HYPE) gets 40% weight
- Result: kappa pulled DOWN to ~2000 → GLFT ≈ 6.5 bps per side minimum
- **Fix**: Warmup ramp from 20% regime weight at 0% warmup → 60% at 100%

### Root Cause #4: PriceGrid Snapping
- `snap_bid()` floors to grid step → pushes bids DOWN (+1-3 bps from optimal)
- `snap_ask()` ceils to grid step → pushes asks UP (+1-3 bps from optimal)
- `base_spacing = max(min_tick_multiple * tick_bps, sigma_bps / sigma_divisor, 0.1)`
- **Not fixed** — needed for exchange tick compliance, impact is 1-3 bps

## Full Kappa Pipeline at Cold Start (traced)
1. `kappa_orchestrator.kappa_raw()` → warmup blend: 0.4×book + 0.3×capped_robust + 0.3×prior(2000)
2. `parameter_estimator.apply_kappa_floor()` → clamp
3. `market_params.kappa_robust` in generate_ladder
4. **Regime blending**: (1-w)×kappa_robust + w×regime_kappa(Normal=2000)
   - w was 0.6 fixed → now 0.2 at cold start, ramps to 0.6
5. AS alpha feedback: only if `as_warmed_up && alpha > 0.3`
6. Final kappa drives GLFT: δ* = (1/γ)ln(1+γ/κ) + fee

## Expected Spread After All Fixes
- κ ≈ 2500-3000 at cold start (more market signal, less regime default)
- GLFT: 1/2500 + 0.00015 = 5.5 bps → with kappa=3000: 4.8 bps
- No AS widening at cold start (cap=1.0x)
- PriceGrid: +1-2 bps
- **Total: ~6-7 bps per side** (vs previous 30+ bps, then 11 bps)

## Files Modified This Session
- `src/market_maker/strategy/ladder_strat.rs`:
  - Regime kappa blend weight: warmup ramp 20%→60% (was fixed 60%)
  - Warmup AS cap: 1.0x at cold start (was 1.15x)
- `src/market_maker/orchestrator/quote_engine.rs` (previous session, carried forward):
  - ToxicityRegime::Normal spread widening removed (was double-counting)

## Key Architecture Insights
- `SpreadComposition` (warmup_addon_bps, coordinator_uncertainty_premium_bps) only used by dashboard/tests, NOT by generate_ladder
- The actual spread in generate_ladder is: GLFT depths → floor clamp → position zone → kappa cap → AS multiplier → PriceGrid snap
- `kappa_orchestrator.kappa_raw()` has separate warmup logic: caps robust at 2×prior, uses book 40% + robust 30% + prior 30%
- `BlendedKappaEstimator.warmup_factor=0.8` is on a DIFFERENT code path (not the robust kappa path used by default)

## Verification Status
- Clippy: CLEAN
- Tests: Not yet run (cargo test filter issue — tests exist but `cargo test test_name` returns 0 matches, likely module path issue)
- Paper trading: NOT yet run with these latest changes

## Remaining Issues
- HYPE BBO is ~3-5 bps per side; our quotes at ~6-7 bps may still need occasional volatility for fills
- Over 3600s this should produce fills — HYPE moves >7 bps regularly
- PriceGrid adds unnecessary widening during warmup (1-3 bps) — could be bypassed
- The `kappa_prior_mean=2500` and `prior_kappa=2000` may both be too low for HYPE specifically
- Need to run the full 3600s paper trading test to prove calibration gate passage
