# Session: LadderStrategy Adaptive Spreads Fix

## Date: 2026-01-02

## Problem
Order history showed ~16 bps half-spreads when we expected ~8-10 bps from adaptive priors.

**Evidence:**
- Best bid: 89,685 → 15.9 bps from mid (89,828)
- Best ask: 89,971 → 15.9 bps from mid

## Root Cause
`LadderStrategy.generate_ladder()` was NOT updated when we added adaptive spreads to `GLFTStrategy.calculate_quotes()`.

The ladder strategy was still using:
- Old multiplicative gamma (not adaptive log-additive)
- Book-based kappa (not adaptive blended kappa)
- Static spread floor (not adaptive Bayesian floor)

**Critical Lines (before fix):**
```rust
// Lines 286-289: Uses OLD multiplicative gamma
let base_gamma = self.effective_gamma(market_params, position, effective_max_position);
let gamma_with_liq = base_gamma * market_params.liquidity_gamma_mult;
let gamma = gamma_with_liq * market_params.tail_risk_multiplier;

// Lines 291-294: Uses book-based kappa - NOT adaptive!
let alpha = market_params.predicted_alpha.min(0.5);
let kappa = market_params.kappa * (1.0 - alpha);

// Lines 335-336: Uses static floor - NOT adaptive!
let effective_floor_frac = market_params.effective_spread_floor(self.risk_config.min_spread_floor);
```

## Solution
Updated `LadderStrategy.generate_ladder()` to use adaptive values when enabled, mirroring GLFT strategy logic.

### Changes Made to `src/market_maker/strategy/ladder_strat.rs`

1. **Gamma calculation (lines 286-305)**:
   ```rust
   let gamma = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
       // Adaptive gamma: log-additive scaling
       market_params.adaptive_gamma * market_params.tail_risk_multiplier
   } else {
       // Legacy: multiplicative RiskConfig gamma
       ...
   };
   ```

2. **Kappa calculation (lines 307-323)**:
   ```rust
   let kappa = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
       market_params.adaptive_kappa
   } else {
       // Legacy: book-based kappa with AS adjustment
       ...
   };
   ```

3. **Floor calculation (lines 359-376)**:
   ```rust
   let effective_floor_frac = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
       market_params.adaptive_spread_floor
   } else {
       market_params.effective_spread_floor(self.risk_config.min_spread_floor)
   };
   ```

4. **Uncertainty factor scaling (lines 425-445)**:
   Added warmup uncertainty scaling (1.2 → 1.0) to ladder depths.

5. **Logging (lines 453-465)**:
   Added `adaptive_mode` and `warmup_pct` to debug log.

## Expected Result
- **Before**: ~16 bps half-spreads (legacy multiplicative gamma)
- **After**: ~8-10 bps half-spreads (adaptive priors + uncertainty)

## Key Insight
When adding new features like adaptive spreads, need to update ALL strategy implementations:
- `GLFTStrategy.calculate_quotes()` ✓
- `LadderStrategy.generate_ladder()` ✓ (this fix)

The LadderStrategy is used for multi-level quoting (ladder orders), which is the default production mode.

## Tests
- 61 ladder-related tests pass
- Full compilation successful

## Related Sessions
- `session_2026-01-02_warmup_bypass_fix.md` - Added adaptive values via priors
- `design_tight_spread_system.md` - Tight spread architecture
