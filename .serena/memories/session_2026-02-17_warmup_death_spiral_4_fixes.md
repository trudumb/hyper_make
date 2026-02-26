# Session 2026-02-17: Warmup Death Spiral — 4 Fixes

## Problem
With `--max-position 10 --capital-usd 1000`, the system produced only 0-1 ladder levels instead of the expected 4-5 per side. This prevented all fills → no calibration → perpetual death spiral.

## Root Cause Chain (4 issues)

### Issue 1: Paper Mode Position Ramp (market_maker.rs)
Paper auto-calibration used `StochasticConfig::default()` which has `enable_position_ramp: true` with `ramp_initial_fraction: 0.10`.
- Effect: effective_max_position = 10.0 * 0.10 = 1.0 at startup, slowly ramping over 1800s
- Fix: Set `enable_position_ramp: false, enable_performance_gating: false` in paper mode's StochasticConfig

### Issue 2: Regime Fraction from Checkpoint (quote_engine.rs:1418-1441)
`self.stochastic.regime_state.params.max_position_fraction` loaded from checkpoint could be 0.175 (unclamped raw RegimeParams).
- Effect: effective_max = 10.0 * 0.175 = 1.75 (even after paper ramp fix)
- Fix: Use `unified_regime().max_position_fraction` (clamped [0.3, 1.0]) AND bypass entirely when `fill_count < 10`

### Issue 3: ExecutionMode is_warmup False (quote_engine.rs:2510)
`is_warmup: !self.estimator.is_warmed_up()` returned FALSE because checkpoint had vol_filter_obs=110 > min_volume_ticks=10.
- Effect: When ToxicityRegime::Toxic + flat position, `select_mode()` returned `ExecutionMode::Flat` for Medium capital → cleared all levels
- Fix: Changed to `is_warmup: self.tier1.adverse_selection.fills_measured() < 10`

### Issue 4: Toxicity Filter is_warmup False (quote_engine.rs:2596)
Same issue as #3 — `!self.estimator.is_warmed_up()` returned FALSE from checkpoint.
- Effect: Toxic + flat → cleared both sides instead of widening spreads 1.5x
- Fix: Changed to `let is_warmup = self.tier1.adverse_selection.fills_measured() < 10`

## Key Learning: `estimator.is_warmed_up()` is Unreliable for Cold-Start Detection
The parameter estimator checks `tick_count >= 10 && trade_count >= 5`. Checkpoint data satisfies these thresholds immediately.
**CORRECT warmup signal for cold-start protection**: `fills_measured() < N` (actual fill count from adverse_selection classifier).
This is the **same pattern** as the regime fraction fix — `fills_measured()` is the canonical warm-start indicator across the codebase.

## Result
- effective_max_position: 10.0 (was 1.0-1.75)
- Ladder levels: 4/4 consistently (was 0-1)
- Empty cycles: 0% (was >50%)
- Spreads: ~16 bps/side (GLFT optimal)

## Files Modified
- `src/bin/market_maker.rs:2582` — paper mode StochasticConfig with ramp/gating disabled
- `src/market_maker/orchestrator/quote_engine.rs:1418-1441` — regime fraction bypass + unified_regime()
- `src/market_maker/orchestrator/quote_engine.rs:2510` — execution mode is_warmup from fill_count
- `src/market_maker/orchestrator/quote_engine.rs:2596` — toxicity filter is_warmup from fill_count
