---
allowed-tools: Glob, Grep, Read
description: Track market maker production gap implementation progress
---

Check implementation status of the 9 production-ready market maker gaps and report progress.

## Instructions

For each of the 9 gaps, check if the corresponding module exists and has implementation:

### Gap 1: Adverse Selection
- Check: `src/market_maker/adverse_selection.rs`
- Look for: `AdverseSelectionEstimator` struct, `spread_adjustment()` method
- Integration: `MarketParams.as_spread_adjustment`, `MarketParams.predicted_alpha`

### Gap 2: Queue Position
- Check: `src/market_maker/queue.rs`
- Look for: `QueuePositionTracker` struct, `fill_probability()` method
- Integration: `queue_tracker` field in MarketMaker, updated on L2Book

### Gap 3: Liquidation Cascade
- Check: `src/market_maker/liquidation.rs`
- Look for: `LiquidationCascadeDetector` struct, `should_pull_quotes()` method
- Integration: `MarketParams.tail_risk_multiplier`, `MarketParams.should_pull_quotes`

### Gap 4: Hawkes Order Flow
- Check: `src/market_maker/hawkes.rs`
- Look for: `HawkesOrderFlowEstimator` struct, dual intensity model (`lambda_buy`, `lambda_sell`)
- Integration: `MarketParams.hawkes_buy_intensity`, `hawkes_sell_intensity`, `hawkes_imbalance`

### Gap 5: Funding Rate
- Check: `src/market_maker/funding.rs`
- Look for: `FundingRateEstimator` struct, `funding_cost()` method
- Integration: `MarketParams.funding_rate`, `MarketParams.predicted_funding_cost`

### Gap 6: Correlation
- Check: `src/market_maker/correlation.rs`
- Look for: `CorrelationEstimator` struct, multi-scale EWMA (`CorrelationHorizon`)
- Integration: `correlation` field in MarketMaker (Optional, for multi-asset)

### Gap 7: Volatility Regime
- Check: `src/market_maker/estimator.rs`
- Look for: `VolatilityRegime` enum with 4 states (Low, Normal, High, Extreme)
- Integration: `MarketParams.volatility_regime`, gamma/spread multipliers

### Gap 8: Spread Process
- Check: `src/market_maker/spread.rs`
- Look for: `SpreadProcessEstimator` struct, `spread_percentile()` method
- Integration: `MarketParams.fair_spread`, `spread_percentile`, `spread_regime`

### Gap 9: Calibration
- Check: `src/market_maker/calibration.rs` and `src/bin/calibrate.rs`
- Look for: `ModelCalibrator` struct, MLE fitting functions
- Integration: Offline tool for parameter estimation

## Output Format

Report as a table:
| Gap | Status | File | Key Components |
|-----|--------|------|----------------|

Then show:
- Completion percentage (X/9 gaps complete)
- Next recommended gap to implement
- Any integration status (wired into mod.rs, strategy.rs)

## Current Status (Auto-Updated)

| Gap | Status | File | Key Components |
|-----|--------|------|----------------|
| 1. Adverse Selection | ✅ Complete | `adverse_selection.rs` | `AdverseSelectionEstimator`, `spread_adjustment()`, `predicted_alpha()` |
| 2. Queue Position | ✅ Complete | `queue.rs` | `QueuePositionTracker`, `fill_probability()`, `should_refresh()` |
| 3. Liquidation Cascade | ✅ Complete | `liquidation.rs` | `LiquidationCascadeDetector`, `should_pull_quotes()`, `tail_risk_multiplier()` |
| 4. Hawkes Order Flow | ✅ Complete | `hawkes.rs` | `HawkesOrderFlowEstimator`, `lambda_buy()`, `lambda_sell()`, dual intensity |
| 5. Funding Rate | ✅ Complete | `funding.rs` | `FundingRateEstimator`, `funding_cost()`, EWMA prediction |
| 6. Correlation | ✅ Complete | `correlation.rs` | `CorrelationEstimator`, `CorrelationHorizon`, multi-scale EWMA |
| 7. Volatility Regime | ✅ Complete | `estimator.rs` | `VolatilityRegime` (Low/Normal/High/Extreme), `gamma_multiplier()` |
| 8. Spread Process | ✅ Complete | `spread.rs` | `SpreadProcessEstimator`, `spread_percentile()`, `SpreadRegime` |
| 9. Calibration | ❌ Not Started | N/A | Offline calibration tool not implemented |

**Completion: 8/9 gaps complete (89%)**

### Integration Status

All 8 implemented gaps are fully integrated into the MarketMaker:

1. **mod.rs integration**: All estimators are instantiated in `MarketMaker::new()` and stored as fields
2. **MarketParams integration**: All outputs flow through `MarketParams` struct to strategy
3. **Event processing**: Updates on AllMids, Trades, L2Book, UserFills messages
4. **Kill switch integration**: Cascade severity triggers kill switch
5. **Prometheus metrics**: Key metrics exported for monitoring

### Recent Additions (This Session)

- **Ladder quoting**: Multi-level GLFT ladder with bulk order placement
- **Margin-aware sizing**: Real-time margin state from exchange
- **Bulk orders**: Single API call for all ladder levels
- **Stale data fix**: Threshold increased to 30s for ladder operations

### Next Recommended Gap

**Gap 9: Calibration** - Create offline calibration tool for:
- Adverse selection model parameters (AS decay, depth buckets)
- Hawkes process parameters (alpha, beta, gamma, mu)
- Funding rate prediction coefficients
- Volatility regime thresholds

Implementation approach:
1. Create `src/bin/calibrate.rs` binary
2. Add `src/market_maker/calibration.rs` module
3. Historical data replay with MLE fitting
4. Output config files for production use
