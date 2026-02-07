# Strategy Agent Memory

## Edge Prediction Overshoot Analysis (2026-02-07)

Predicted 7.24 bps vs realized 2.87 bps (2.5x overshoot). Three root causes identified and two fixed:

### Fixes Applied
1. **adverse_prior raised from 0.15 to 0.25** in `TheoreticalEdgeConfig::default()`
   - 0.15 underestimates BTC's 20-30% informed flow on Hyperliquid
   - ~1-2 bps overshoot contribution
2. **Fill dampening floor tightened from 0.5 to 0.3** in both `calculate_edge()` and `calculate_edge_enhanced()`
   - Range now [0.3, 1.0] instead of [0.5, 1.0]
   - ~1 bps overshoot contribution

### Pending Fix (Needs Coordination)
3. **Spread capture uses market spread, not our spread** - biggest contributor (~2 bps)
   - `spread_edge_bps = spread_bps / 2.0` uses L2 market spread
   - Should use min(market_spread, our_GLFT_spread) since we capture at most our own half-spread
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
