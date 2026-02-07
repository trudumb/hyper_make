# Strategy Agent Memory

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
