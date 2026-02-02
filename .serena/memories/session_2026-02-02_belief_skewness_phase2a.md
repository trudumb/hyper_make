# Session 2026-02-02: Belief Skewness Tracking (Phase 2A.1)

## Summary
Implemented Phase 2A.1 of the alpha-generating architecture improvements: Fat-tail belief skewness tracking for asymmetric spread adjustment.

## Problem Addressed
Gaussian assumptions miss fat tails in volatility. Right-skewed vol distributions indicate vol spike risk, which should lead to asymmetric spread widening.

## Implementation

### DriftVolatilityBeliefs (snapshot.rs)
Added fields:
- `sigma_skewness` - Skewness of sigma posterior (positive = right-skewed = vol spike risk)
- `sigma_kurtosis` - Excess kurtosis (fat tails indicator)
- `drift_skewness` - Asymmetry in drift belief

Added methods:
- `has_vol_spike_risk()` - True if sigma_skewness > 1.0
- `has_fat_tails()` - True if sigma_kurtosis > 3.0
- `bid_spread_factor(sensitivity)` - Returns [0.8, 1.0] multiplier
- `ask_spread_factor(sensitivity)` - Returns [1.0, 1.2] multiplier
- `tail_risk_score()` - Combined skew+kurtosis score [0, 1]

### KappaBeliefs (snapshot.rs)
Added fields:
- `spread_ci_lower_skew_adjusted` - Tighter when vol expected to drop
- `spread_ci_upper_skew_adjusted` - Wider when vol spike risk elevated

Added methods:
- `defensive_spread()` - Upper skew-adjusted CI (quote here when risky)
- `aggressive_spread()` - Lower skew-adjusted CI
- `recommended_spread(vol_risk)` - Interpolate based on risk level
- `is_skew_significant()` - True if skew adjustment > 10% of CI width

### InternalState (central.rs)
Added `sigma_skewness` field to track skewness over time.

### Skewness Computation (central.rs)
In `build_drift_vol_beliefs()`:
- Compute sigma_skewness from Inverse-Gamma posterior: `4 × sqrt(2 / (α - 3))`
- Compute sigma_kurtosis from Inverse-Gamma: `6×(5α-11)/((α-3)(α-4))`
- drift_skewness: simple heuristic based on drift direction

In `build_kappa_beliefs()`:
- Compute skew-adjusted CIs: `ci × (1 ± skew_factor × 0.5)`

## Files Modified
1. `src/market_maker/belief/snapshot.rs` - Added fields, methods, and 9 tests
2. `src/market_maker/belief/central.rs` - Added sigma_skewness to InternalState and computation

## Tests
- 9 new skewness tests in snapshot.rs
- 1781 total tests passing

## Key Formulas

### Sigma Skewness (Inverse-Gamma)
```
skewness = 4 × √(2 / (α - 3)) for α > 3
```
Where α = n_obs/2 + prior_α

### Asymmetric Spread Adjustment
```
bid_factor = 1.0 - tanh(σ_skew × sensitivity) × 0.1
ask_factor = 1.0 + tanh(σ_skew × sensitivity) × 0.1
```

### Skew-Adjusted CI
```
ci_lower_adjusted = ci_lower × (1 - skew_factor × 0.5)
ci_upper_adjusted = ci_upper × (1 + skew_factor × 0.5)
```

## V2 Refinements Progress
| Phase | Status |
|-------|--------|
| 1A.2 COFI | ✅ Done |
| 1A.1 Trade Size | ✅ Done |
| 7.1 Signal Decay | ✅ Done |
| 2A.1 Belief Skewness | ✅ Done |
| 4A.2 Funding Magnitude | Pending |
| 4A.1 BOCPD | Pending |
| 6A.1 PCA | Pending |
| 6A.2 RL Tuning | Pending |
