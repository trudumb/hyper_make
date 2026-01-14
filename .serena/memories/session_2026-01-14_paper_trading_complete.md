# Session: Paper Trading Simulation System - Complete

**Date**: 2026-01-14
**Status**: Complete
**Focus**: Full paper trading infrastructure with dashboard integration

## Executive Summary

Built a comprehensive paper trading/simulation system for the Hyperliquid market maker that:
- Generates quotes exactly as the live system would (GLFT formula)
- Tracks quotes WITHOUT placing real orders
- Simulates fills based on market trade flow
- Logs predictions for calibration analysis
- Provides PnL attribution and live dashboard

## Architecture

```
Market Data (WebSocket) → Quote Engine → SimulationExecutor → PredictionLogger
                                              ↓
                                        FillSimulator ← Market Trades
                                              ↓
                                        OutcomeTracker → CalibrationMetrics
                                              ↓
                                        Dashboard API → Web UI
```

## Files Created

### Simulation Module (`src/market_maker/simulation/`)
| File | Purpose |
|------|---------|
| `mod.rs` | Module exports |
| `prediction.rs` | PredictionRecord, MarketStateSnapshot, ModelPredictions, PredictionLogger |
| `executor.rs` | SimulationExecutor implementing OrderExecutor trait |
| `fill_sim.rs` | FillSimulator (probabilistic), AggressiveFillSimulator (trade-through) |
| `calibration.rs` | BrierDecomposition, CalibrationAnalyzer, CalibrationCurve |
| `outcome.rs` | PnLDecomposition, OutcomeTracker, RegimeAttribution |

### Binary
| File | Purpose |
|------|---------|
| `src/bin/paper_trader.rs` | CLI tool with WebSocket subscriptions, quote generation, dashboard API |

### Script
| File | Purpose |
|------|---------|
| `scripts/paper_trading.sh` | Shell wrapper with dashboard server startup |

## Key Features

### 1. SimulationExecutor
- Implements `OrderExecutor` trait
- Tracks virtual orders in memory
- No real exchange interaction
- Thread-safe order management

### 2. Fill Simulation
- Probabilistic model based on:
  - Trade price vs order price
  - Trade size and queue position
  - Order age (FIFO approximation)
- Aggressive mode for trade-through fills

### 3. Prediction Logging
- JSONL output for analysis
- Captures per-cycle:
  - Market state (σ, κ, regime, spread)
  - Model predictions (fill rate, adverse selection)
  - Ladder levels with fill probabilities

### 4. Calibration Analysis
- Brier score decomposition (reliability, resolution, uncertainty)
- Calibration curves (predicted vs realized)
- Conditional slices by regime

### 5. PnL Attribution
- Spread capture
- Adverse selection (1s price impact)
- Inventory cost (mark-to-market)
- Fee cost

### 6. Dashboard Integration
- HTTP API at `/api/dashboard`
- Real-time state updates every quote cycle
- Compatible with existing `mm-dashboard-fixed.html`

## Bug Fixes

1. **Integer overflow** in `prediction.rs:548`
   - Changed `now_ns - r.timestamp_ns` to `now_ns.saturating_sub(r.timestamp_ns)`

2. **Module visibility** - Made `metrics` module public in `infra/mod.rs`

3. **Side enum** - Added Serialize/Deserialize, fixed Bid/Ask → Buy/Sell

4. **Unreachable patterns** - Removed catch-all arms from Side match statements

## Usage

```bash
# Quick 1-minute test
./scripts/paper_trading.sh BTC 60

# With dashboard (opens browser UI)
./scripts/paper_trading.sh BTC 300 --dashboard

# Verbose logging
./scripts/paper_trading.sh BTC 300 --verbose

# Full calibration report
./scripts/paper_trading.sh BTC 3600 --report
```

## Dashboard URLs (when --dashboard enabled)
- Web UI: http://localhost:3000/mm-dashboard-fixed.html
- API: http://localhost:8080/api/dashboard

## Dependencies Added
- Uses existing: axum, tower-http, serde_json, chrono
- No new crate dependencies

## Testing Status
- ✅ Compiles without errors
- ✅ WebSocket connections work
- ✅ Quote generation functional
- ✅ Fill simulation working
- ✅ Dashboard API responds
- ⚠️ Playwright verification pending (port conflicts during testing)

## Future Enhancements
- Add fill history to dashboard
- Implement regime history tracking
- Enhance calibration metrics display
- Create automated integration tests
