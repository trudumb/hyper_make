# Session: 2026-01-13 Data-Driven Constants Plan

## Summary
Analysis of GLFT ad-hoc multiplier cleanup status and creation of future improvements plan for deriving remaining heuristic constants from empirical data.

## Context
The `/sc:analyze` command was run to assess the status of ad-hoc multiplier cleanup in the GLFT strategy. The analysis confirmed that major cleanup work was already completed (documented in `session_2026-01-13_ad_hoc_multiplier_cleanup.md`).

## Key Findings

### Cleanup Status: COMPLETE
The three main problems identified have been resolved:

| Problem | Resolution |
|---------|------------|
| Redundant volatility scaling | `regime_scalar` removed; `vol_scalar` handles via continuous relationship |
| Contradictory spread regime mult | `spread_regime_mult` (1.3/1.1/1.0/0.95/0.9) removed entirely |
| Arbitrary Hawkes thresholds | 0.8/0.9 cutoffs replaced with continuous: `1 + 2.0 × (percentile - 0.5)` |

### Remaining Heuristic Constants
These are well-documented with marked TODOs for future derivation:

| Constant | Location | Value | Future Work |
|----------|----------|-------|-------------|
| Tail multiplier | glft.rs:384 | `(2.0-CV).clamp(0.5,1.0)` | Fit from kappa_cv ↔ fill AS regression |
| Alpha cap | glft.rs:380 | 0.5 (50%) | Review literature; use soft sigmoid |
| Jump premium cap | glft.rs:462 | 50 bps | Derive from historical crash data |
| Hawkes skew coeff | glft.rs:635 | 0.00005 | Regress E[Δp \| imbalance] |
| Funding skew coeff | glft.rs:663 | 0.01 | Formula: cost × E[T] / spread |
| Floor risk k | adaptive/config.rs:143 | 1.17σ | Target 95% coverage (k=1.645) |

## Implementation Plan Created

### Phase 1: Instrumentation
Add logging for:
- `kappa_cv` with fill outcomes
- `jump_ratio` during extreme events
- `hawkes_imbalance` pre-fill values
- `holding_time` from position changes
- `fill_edge` = price - fair - fees

### Phase 2: Data Collection
Collect over 2-3 weeks:
- 10K+ fills for regression
- 7+ days of kappa_cv distribution
- Extreme event samples (jump_ratio > 3)

### Phase 3: Analysis & Derivation
For each constant:
- Run regressions / empirical distributions
- Derive principled coefficients
- A/B test against current values

### Phase 4: Implementation
- Update constants with derived values
- Add derivation documentation
- Create tracking for ongoing recalibration

## Verification
- ✅ `cargo build`: Compiles successfully
- ✅ `cargo test`: 823 tests pass (7 additional from examples)

## Files Created
- `/home/trudumb/.claude/plans/modular-popping-sonnet.md` - Full analysis and plan
- This session memory

## Next Steps
1. Begin Phase 1 instrumentation when ready to collect data
2. Consider priority order: P1 constants (tail, alpha, jump) have highest impact
3. Book depth threshold can be made relative without data collection

## Key Principle
**Current pattern (correct):**
```rust
let gamma_adjusted = gamma_base * principled_scaling_factor(uncertainty, conditions);
let optimal_spread = glft_formula(gamma_adjusted, kappa);
// Trust the math
```

**Avoid anti-pattern:**
```rust
let actual_spread = optimal_spread * arbitrary_multiplier; // Don't do this
```
