# Session: Magic Number Elimination with Bayesian Learning

**Date**: 2026-02-02
**Objective**: Replace 170+ magic numbers with statistically grounded, Bayesian-learned parameters

## Summary

Implemented comprehensive infrastructure to replace arbitrary magic numbers with:
1. **Bayesian Regularized Parameters**: Online learning with priors to prevent overfitting
2. **First-Principles Derivations**: Mathematical formulas for GLFT, Kelly, VaR parameters
3. **Historical Calibration**: Batch calibration from fill/trade logs

## Files Created

### 1. `src/market_maker/calibration/parameter_learner.rs`

Core Bayesian parameter learning infrastructure:

- `BayesianParam`: Generic parameter with prior + posterior
  - Supports Beta, Gamma, Normal, InverseGamma, LogNormal families
  - Conjugate updates for online learning
  - Shrinkage estimation toward prior with few samples
  
- `LearnedParameters`: Collection of all learned parameters
  - Tier 1 (P&L Critical): alpha_touch, gamma_base, spread_floor_bps, etc.
  - Tier 2 (Risk): max_daily_loss_fraction, max_drawdown, cascade_oi_threshold
  - Tier 3 (Calibration): kappa, hawkes_mu/alpha/beta, regime_sticky_diagonal
  - Tier 4 (Microstructure): kalman_q/r, depth_spacing_ratio, microprice_decay

Key features:
- `estimate()`: Shrinkage estimate (blends MLE and prior)
- `credible_interval_95()`: Uncertainty quantification
- `is_calibrated()`: Check if enough data for reliable estimates
- `data_weight()`: How much the estimate is driven by data vs prior

### 2. `src/market_maker/calibration/historical_calibrator.rs`

Batch calibration from historical data:

- `HistoricalCalibrator`: Loads fills, snapshots, trades and calibrates all parameters
- `FillRecord`, `MarketSnapshot`, `TradeRecord`: Data structures for calibration
- `PowerAnalysis`: Sample size requirements for different parameter types

Calibration methods:
- `calibrate_alpha_touch()`: From fill adverse selection analysis
- `calibrate_kappa()`: From book depth and trade rates
- `calibrate_hawkes()`: Method of moments from trade arrivals
- `calibrate_toxic_hours()`: From hourly adverse selection patterns
- `calibrate_spread_floor()`: From fill profitability by depth
- `calibrate_cascade_threshold()`: From OI drops before bad fills

### 3. `src/market_maker/calibration/derived_constants.rs`

First-principles derivations:

- `derive_gamma_from_glft()`: γ from target spread, κ, σ, T
- `derive_spread_floor()`: δ_min = fee + AS + latency_slippage
- `derive_max_daily_loss()`: Kelly-scaled from account × f × 2σ
- `derive_max_drawdown()`: VaR_99 × horizon_factor
- `derive_ewma_alpha()`: From target half-life
- `derive_kalman_noise()`: Q and R from tick volatility and spread
- `derive_hazard_rate()`: 1/E[regime_duration]
- `derive_toxic_hour_multiplier()`: (toxic_AS + fee) / (normal_AS + fee)
- `derive_quote_latch_threshold()`: 2 × (fee + slippage)
- `derive_confidence_threshold()`: Cost-sensitive ROC optimization
- `derive_depth_spacing_ratio()`: From fill intensity curve
- `derive_reduce_only_threshold()`: From margin analysis

## Files Modified

### `src/market_maker/config/stochastic.rs`

Added fields for learned parameters integration:
- `use_learned_parameters: bool` - Enable Bayesian learning
- `learned_param_min_observations: usize` - Power analysis derived (100)
- `learned_param_max_cv: f64` - Maximum acceptable CV (0.5)
- `learned_param_staleness_hours: f64` - Signal decay half-life (4.0)

### `src/market_maker/strategy/risk_config.rs`

Added GLFT-derived methods:
- `derive_gamma_from_glft()`: Compute optimal γ from target spread
- `derive_spread_floor()`: Compute minimum spread from components

### `src/market_maker/risk/kill_switch.rs`

Added Kelly-scaled constructor:
- `KillSwitchConfig::from_account_kelly()`: Derives limits from account value

### `src/market_maker/estimator/kappa_orchestrator.rs`

Documented prior derivations:
- `prior_kappa = 2000.0`: Historical median, Gamma(4, 0.002) prior
- `prior_strength = 10.0`: Pseudo-observations for regularization
- `robust_nu = 4.0`: From kurtosis analysis (ν ≈ 6/excess_kurtosis)
- `KAPPA_EWMA_ALPHA = 0.9`: From half-life analysis

### `src/market_maker/control/bayesian_bootstrap.rs`

Documented prior derivations:
- `prior_alpha/beta`: Gamma(5, 0.1) → E[θ]=50, power analysis based
- Exit thresholds: From statistical requirements for IR estimation

### `src/market_maker/calibration/mod.rs`

Updated exports for new modules and functions.

## Key Design Decisions

### 1. Bayesian Regularization vs Pure MLE

**Problem**: Pure MLE overfits with small samples
**Solution**: Use informative priors based on domain knowledge (the old magic numbers)

Shrinkage formula:
```
θ_posterior = w × θ_MLE + (1-w) × θ_prior
where w = n / (n + prior_strength)
```

### 2. Prior Elicitation Strategy

| Parameter Type | Prior Family | Reasoning |
|----------------|--------------|-----------|
| Probabilities | Beta(α, β) | Bounded [0,1], conjugate to Binomial |
| Rates | Gamma(shape, rate) | Positive, conjugate to Poisson/Exponential |
| Unbounded | Normal(μ, σ²) | Regularizes toward prior mean |
| Variances | InverseGamma | Conjugate to Normal variance |
| Multiplicative | LogNormal | Positive, multiplicative scale |

### 3. Derivation Documentation

Every parameter now has:
1. **DERIVATION comment**: Mathematical formula or statistical basis
2. **Domain knowledge**: Why this prior/formula makes sense
3. **Input requirements**: What data is needed

## Usage Example

```rust
use crate::market_maker::calibration::{LearnedParameters, BayesianParam};

// Create with defaults (priors based on old magic numbers)
let mut params = LearnedParameters::default();

// Record fills and update alpha_touch
for fill in observed_fills {
    let is_informed = fill.adverse_move_bps < -5.0;
    if is_informed {
        params.alpha_touch.observe_beta(1, 0);
    } else {
        params.alpha_touch.observe_beta(0, 1);
    }
}

// Get regularized estimate (shrinks toward 0.25 with few samples)
let alpha = params.alpha_touch.estimate(); // Bayesian posterior mean

// Check calibration status
let status = params.calibration_status();
if status.tier1_ready {
    // Use learned parameters
} else {
    // Fall back to priors (original magic numbers)
}
```

## Completed Phases

### Phase 1: Infrastructure ✅
- Created `parameter_learner.rs` with `BayesianParam` and `LearnedParameters`
- Created `historical_calibrator.rs` for batch calibration
- Created `derived_constants.rs` with first-principles derivations
- 21 tests passing

### Phase 2: Integration ✅
- Wired `LearnedParameters` into `StochasticComponents`
- Added helper methods: `update_alpha_touch()`, `learned_alpha_touch()`, etc.
- Added config flags: `use_learned_parameters`, `learned_param_min_observations`

### Phase 3: Online Learning ✅
- Extended `AdverseSelectionEstimator` with informed fill classification
- Added fields: `informed_fills_count`, `uninformed_fills_count`, `informed_threshold_bps`
- Wired periodic alpha_touch updates from AS classifier in event loop
- Wired periodic kappa updates from fill rate observations in event loop
- Added 2 new AS estimator tests for informed fill tracking

### Phase 4: Persistence ✅
- Added `Serialize`/`Deserialize` to `BayesianParam`, `LearnedParameters`, `CalibrationStatus`
- Added `save_to_file()`, `load_from_file()`, `load_or_default()` methods
- Added `default_path()` helper for consistent file naming
- Added 3 new persistence tests (roundtrip, missing file, default path)

## Test Results
- All 1838 tests passing

## Remaining Tasks

### Phase 5: Logging & Monitoring
- Add periodic logging of learned parameter evolution
- Add Prometheus metrics for parameter tracking
- Add calibration status to health dashboard

### Phase 6: Use Learned Parameters in Quoting
- Replace magic numbers in GLFT calculations with `learned_params.kappa.estimate()`
- Replace alpha in Kelly with `learned_params.alpha_touch.estimate()`
- Add conditional logic to use learned vs default based on calibration status



1. **Integration**: Wire LearnedParameters into actual quoting logic
2. **Persistence**: Save/load calibrated parameters across sessions  
3. **Logging**: Add structured logging for parameter evolution
4. **A/B Testing**: Compare learned vs static parameters in paper trading
5. **Regime Conditioning**: Separate parameters by regime

## Technical Notes

- All new code compiles cleanly (only 1 unrelated warning)
- Tests included for BayesianParam and PowerAnalysis
- Thread-safe (no Arc/Mutex needed for BayesianParam updates)
