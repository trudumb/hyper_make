# Code Reviewer Memory

## Review Checklist
1. Calibration metrics validated before/after
2. No hardcoded parameters (must be regime-dependent or configurable)
3. Units documented in variable names (`_bps`, `_s`, `_8h`)
4. Risk limits respected (`inventory.abs() <= max_inventory`)
5. Spread invariants (`ask > bid`, both correct side of microprice)
6. `#[serde(default)]` on all checkpoint fields
7. `kappa > 0.0` and `gamma > 0.0` in all formula paths
8. No binary side-clearing — route through graduated widening

## Known Patterns to Watch
- "Config struct defined but not enforced" — trace from definition to hot path usage
- "Replace vs blend" — any import/reload using `=` instead of blending deserves scrutiny
- EWMA update-before-compute ordering
- `estimator.is_warmed_up()` unreliable post-checkpoint — prefer `fills_measured() < N`
- RL agent: observe-only (gamma/omega multipliers not consumed by GLFT)
- `estimator.sigma()` returns fractional per-second sigma — must `* 10000` for bps

## Key Audit Results
- Phase 6 calibration audit: PASS with CONCERNS (sigma cap, kappa stagnation risk)
- Phase 8 RL audit: RL agent NOT driving live quotes (observe-only)
- Risk audit: 8.5/10 production-ready (3 HIGH, 3 MEDIUM open)
