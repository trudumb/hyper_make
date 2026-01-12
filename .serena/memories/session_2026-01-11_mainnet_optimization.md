# Session: 2026-01-11 Mainnet Optimization

## Summary
Implemented mainnet-focused improvements to market maker parameters after correcting testnet overfitting. Added universal safety improvements (book kappa reliability, dynamic warmup) and mainnet-optimized tuning (tighter spreads, faster estimation). Testnet validation showed 44% more fills.

## Key Insight
**Testnet testing validates integration, not parameter tuning.** The initial improvement plan was wrong because it optimized for testnet's illiquidity (18 fills/hour). Mainnet BTC on Hyperliquid is highly liquid, requiring different parameter calibration.

## Changes Made

### Part 1: Universal Improvements (Regime-Agnostic)

**1. Book Kappa Reliability Check** - `src/market_maker/adaptive/blended_kappa.rs`
- Added validation: when `book_kappa > 3× prior` OR `book_update_count < 10`, fall back to prior
- Prevents spread collapse on thin books
- Added `book_kappa_reliable()` method for status reporting

**2. Dynamic Warmup Thresholds** - `src/market_maker/adaptive/calculator.rs`
- Warmup thresholds now adapt based on observed fill rate
- Liquid markets (≥0.005 fills/sec): 20 floor obs, 10 kappa fills
- Illiquid markets (<0.005 fills/sec): 10 floor obs, 5 kappa fills

### Part 2: Mainnet Optimizations

**3. RiskConfig Tuning** - `src/market_maker/strategy/risk_config.rs`
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `gamma_base` | 0.3 | 0.15 | Less risk aversion on deep books |
| `min_spread_floor` | 8 bps | 5 bps | Tighter quotes on liquid markets |
| `inventory_threshold` | 0.3 | 0.5 | Easier position exits |
| `max_volatility_multiplier` | 3.0 | 2.0 | Less extreme scaling |

**4. Fill Rate Model** - `src/market_maker/estimator/fill_rate_model.rs`
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `lambda_0_prior_mean` | 0.1 | 0.8 | Mainnet baseline fill rate |
| `observation_half_life` | 200 | 100 | Faster convergence |

**5. Adverse Selection** - `src/market_maker/adverse_selection/mod.rs`
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `measurement_horizon_ms` | 1000 | 500 | Faster mean reversion |
| `ewma_alpha` | 0.05 | 0.12 | Faster adaptation |

**6. Calibration Controller** - `src/market_maker/estimator/calibration_controller.rs`
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `target_fill_rate_per_hour` | 10 | 60 | Liquid market target |
| `min_gamma_mult` | 0.3 | 0.6 | Protect edge, don't sacrifice spreads |
| `lookback_secs` | 3600 | 900 | Faster regime detection |

**Bug Fix:** Calibration controller fill rate calculation now properly accounts for lookback window size (was dividing by hourly rate, not window-adjusted rate).

## Files Modified
| File | Changes |
|------|---------|
| `src/market_maker/adaptive/blended_kappa.rs` | Book κ reliability check, new method |
| `src/market_maker/adaptive/calculator.rs` | Dynamic warmup thresholds |
| `src/market_maker/strategy/risk_config.rs` | 4 parameter changes |
| `src/market_maker/estimator/fill_rate_model.rs` | 2 parameter + clippy fixes |
| `src/market_maker/adverse_selection/mod.rs` | 2 parameter changes |
| `src/market_maker/estimator/calibration_controller.rs` | 3 params + bug fix |
| `src/market_maker/estimator/parameter_estimator.rs` | Unused import cleanup |
| `src/market_maker/estimator/informed_flow.rs` | Clippy loop fixes |
| `src/bin/parameter_estimator.rs` | Clippy fix |

## Verification Results

### Test Results
- **752 tests pass** ✅
- **Clippy clean** ✅
- **Cargo fmt clean** ✅

### Testnet Validation (1 hour)
| Metric | Previous | After | Change |
|--------|----------|-------|--------|
| Fills | 18 | 26 | +44% ✅ |
| Warmup | 86% | 90% | +4% ✅ |
| Realized AS | 0.47 bps | 0.44 bps | Better ✅ |
| Kill Switch | No | No | ✅ |

## Design Principles Applied

1. **Universal improvements add safety, not remove it** - Book kappa check prevents spread collapse
2. **Mainnet is the target, testnet validates integration** - Don't overfit to testnet conditions
3. **Liquid markets need different calibration** - Higher fill rate expectations, tighter spreads
4. **Faster estimation with abundant data** - Shorter half-lives, higher EWMA alpha

## Next Steps
- Mainnet validation with tiny position size
- Monitor P&L per fill metric on production
- Consider auto-regime detection (future enhancement)
