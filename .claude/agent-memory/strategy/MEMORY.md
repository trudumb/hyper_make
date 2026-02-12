# Strategy Agent Memory

## Per-Order Size Cap (2026-02-08)

### Problem: HYPE Incident
Concentration fallback collapsed 25 levels into 1 order at 100% max position (1.51 HYPE = full $50 limit). One fill maxed inventory with zero recovery capacity.

### Fix: MAX_SINGLE_ORDER_FRACTION = 0.25
- `ladder_strat.rs` step 8b: per-level cap after entropy optimizer
- `ladder_strat.rs` concentration fallback: bid/ask capped
- `generator.rs` 6 fallback paths capped (allocate_sizes x2, build_raw_ladder x2, build_asymmetric_ladder x2)
- Floor = max(min_meaningful_size or min_notional*1.01/price, 25%*max_position) to survive truncation
- Test: `test_no_single_order_exceeds_25pct_max_position` covers all 4 scenarios

### Key Insight
`truncate_float()` rounds DOWN — when computing min_size_for_notional, must use 1.01x buffer or the truncated size * price < min_notional at boundary

## Edge Prediction Overshoot Analysis (2026-02-07)

### 40x Overshoot Root Cause (predicted 58.4 bps, realized 1.44 bps)

Four bugs in `src/bin/paper_trader.rs` (lead-owned), all in EdgeSnapshot construction:

1. **predicted_spread == realized_spread**: Both use `fill_depth_bps * 2.0` (line 2033-2034)
2. **predicted_edge uses raw depth, not GLFT spread**: `fill_depth_bps * 2.0` (~30 bps) instead of GLFT half-spread (~5.5 bps)
3. **AS estimator never fed fills**: `adverse_selection.record_fill()` never called, so `best_horizon_as_bps()` always returns 0.0
4. **realized_as = min(depth, predicted_as) = 0**: Since predicted_as=0, realized_as=0 always

Compound effect: predicted_edge = 30 - 0 - 1.5 = 28.5 bps vs realized ~0.7 bps = 40x

### Earlier Fixes (2.5x overshoot phase)
1. **adverse_prior raised from 0.15 to 0.25** in `TheoreticalEdgeConfig::default()`
2. **Fill dampening floor tightened from 0.5 to 0.3** in `calculate_edge()` and `calculate_edge_enhanced()`

### Pending Fix (Needs Coordination)
3. **Spread capture uses market spread, not our spread** in TheoreticalEdgeEstimator
   - `spread_edge_bps = spread_bps / 2.0` uses L2 market spread
   - Should use min(market_spread, our_GLFT_spread)
   - Requires passing GLFT-computed half-spread to `calculate_edge()` callers

### Key Formula Reference
- GLFT half-spread: delta* = (1/gamma) * ln(1 + gamma/kappa) + vol_comp + fee
- gamma=0.07, kappa=8000 -> 2.75 bps/side = 5.5 bps total (paper mode)
- BayesianAlphaTracker: Beta(2,6) prior -> mean 0.25

### File Ownership
- `control/theoretical_edge.rs` - edge estimation (strategy-owned)
- `control/hybrid_ev.rs` - IR/theoretical blending (strategy-owned)
- `control/quote_gate.rs` - quote decision logic (strategy-owned)
- `strategy/glft.rs` - GLFT formula and quoting (strategy-owned)
- `latent/edge_surface.rs` - 225-cell grid edge surface (NOT strategy-owned, in latent/)

### Test Coverage
- theoretical_edge: 30 tests
- hybrid_ev: 9 tests
- quote_gate: 42 tests
- All pass after changes

## Self-Consistent Gamma + Edge at Actual Spread (2026-02-11)

### Phase 1: solve_min_gamma()
- `glft.rs`: Binary search for min gamma where `half_spread(gamma) >= target_floor`
- Wired in `effective_gamma()` after RL multiplier, before final clamp
- **Key limitation**: GLFT half-spread has a MAXIMUM w.r.t. gamma when sigma is tiny.
  With sigma=0.0001, vol_comp is ~0, and GLFT term `(1/gamma)*ln(1+gamma/kappa)` is bounded.
  At kappa=5000: max achievable spread ~3.5 bps. Floor of 5 bps is UNREACHABLE via gamma alone.
  `solve_min_gamma` returns `hi=100.0` in these cases (clamped by gamma_max).
- **Test fixture lesson**: `test_market_params()` must set `sigma_effective` = `sigma`.
  Default `sigma_effective=0.0001` causes self-consistent gamma to hit gamma_max.
  Set `min_spread_floor=0.00001` in test strategies to isolate from the gamma floor.

### Phase 2: actual_quoted_spread_bps
- `types.rs`: New `Option<f64>` field on `MarketState`
- `ensemble.rs`: `GLFTEdgeModel::predict_edge()` uses `actual_quoted_spread_bps/2` when available
- `mod.rs`: Wired from `params.market_spread_bps` in `build_market_state()`
- Fixes: edge prediction at theoretical 2.87 bps (always negative) vs actual 7.42 bps (positive)

### Serena Tool Reliability Note
Serena `replace_content` sometimes reports OK but edits don't persist on disk (WSL2 filesystem).
Always verify with `grep` after Serena edits. Prefer Claude's Edit tool for reliability.
