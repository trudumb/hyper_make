# Strategy Agent Memory

## Current State
- Additive spread composition: `half_spread * bandit_mult + widening_addon + quota_addon`
- `solve_min_gamma()` binary search for spread floor enforcement
- Inventory + signal skew wired in `signal_integration.rs` (Feb 12)
- Stuck-short fix: sqrt(q_ratio) reducing threshold (Feb 23)
- Edge accountability: edge uncertainty → gamma, calibration quality → gamma (Feb 25)

## Open Issues
- RL agent is observe-only (gamma/omega multipliers populate MarketParams but GLFT never reads them)
- Spread capture in TheoreticalEdgeEstimator uses market spread, not our GLFT spread
- `solve_min_gamma` hits gamma_max when sigma is tiny (GLFT term bounded, max ~3.5 bps at kappa=5000)

## Key Gotchas
- `sigma_effective` must equal `sigma` in test fixtures (default 0.0001 causes gamma_max)
- Set `min_spread_floor=0.00001` in tests to isolate from gamma floor
- `truncate_float()` rounds DOWN — use 1.01x buffer for min_notional calculations
- Skew clamp at 80% half-spread prevents quote crossing
- `signal_integration.rs` is STRATEGY-owned (not lead)
