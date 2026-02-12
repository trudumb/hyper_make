---
name: quote-engine
description: Wire all model components into the quote generation pipeline. Use when building the quote engine, adding new model components, debugging quote generation, or understanding the full data flow from market data through GLFT optimal spread to final bid/ask prices.
user-invocable: false
---

# Quote Engine Integration Skill

## Purpose

Wire together all model components into a coherent quote generation pipeline. This skill describes how the foundation and model layers combine to produce actual quotes.

## When to Use

- Building the quote engine from scratch
- Adding a new model component to the pipeline
- Debugging quote generation issues
- Understanding the full data flow

## Prerequisites

All foundation and model skills should be understood:
- `measurement-infrastructure` (logging)
- `calibration-analysis` (validation)
- `signal-audit` (feature selection)
- `fill-intensity-hawkes` (kappa)
- `adverse-selection-classifier` (toxicity)
- `regime-detection-hmm` (state tracking)
- `lead-lag-estimator` (cross-exchange)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Market Data Feeds                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Hyperliquid  │  │   Binance    │  │  Hyperliquid State   │   │
│  │  Book/Trade  │  │  Book/Trade  │  │  (Funding, OI, etc)  │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
└─────────┼─────────────────┼─────────────────────┼───────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Model Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Lead-Lag   │  │    HMM      │  │  Adverse    │              │
│  │  Estimator  │  │   Filter    │  │  Selection  │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         │         ┌──────┴──────┐         │                      │
│         │         │   Hawkes    │         │                      │
│         │         │  Intensity  │         │                      │
│         │         └──────┬──────┘         │                      │
│         │                │                │                      │
│  ┌──────┴────────────────┴────────────────┴──────┐              │
│  │              Parameter Blender                 │              │
│  └────────────────────┬──────────────────────────┘              │
└───────────────────────┼─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Quote Engine                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. Check liquidation cascade → defensive if active      │    │
│  │  2. Compute adjusted microprice (local + lead-lag)       │    │
│  │  3. Get regime-blended parameters (γ, κ, floors)         │    │
│  │  4. Compute kappa (Hawkes × adverse × regime)            │    │
│  │  5. GLFT optimal spread: δ = (1/γ)ln(1 + γ/κ) + fee      │    │
│  │  6. Apply adjustments (floor, inventory skew, lead-lag)  │    │
│  │  7. Output final quotes                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Measurement Layer                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Log predictions → Async outcome matching → Storage      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Market Data Input

`MarketData` aggregates all feeds: Hyperliquid L2 book/trades, state (funding, OI), Binance book/trades, and current position. See [implementation.md](./implementation.md) for full struct definition.

Key fields: `hl_bids/asks`, `hl_recent_trades`, `funding_rate`, `open_interest`, `binance_mid`, `current_inventory`, `inventory_age_s`.

### Quote Output

`QuoteSet` contains final bid/ask prices and sizes, plus diagnostics for logging: microprice, adjusted microprice, half_spread_bps, inventory/lead-lag skew, regime info, kappa, gamma, and liquidation probability. See [implementation.md](./implementation.md) for full struct definition.

---

## Quote Engine Implementation

### QuoteEngine Components

The `QuoteEngine` struct holds all model components and state:

- **Model components**: `OnlineHMMFilter`, `RegimeConditionedLeadLag`, `HyperliquidFillIntensityModel`, `MLPClassifier`, `LiquidationDetector`
- **State estimators**: `MicropriceEstimator`, `BiPowerVolatilityEstimator`
- **Tracking**: `AdverseSelectionAdjuster`, `RingBuffer<Trade>`
- **Config**: `QuoteEngineConfig` (base_gamma, maker_fee_bps, max_inventory, max_quote_size, min_spread_bps, skew limits, defensive thresholds)
- **Measurement**: `PredictionLogger`

### generate_quotes() Pipeline

The core method `generate_quotes(&mut data) -> QuoteSet` runs 11 steps per cycle:

1. **Update all models** -- HMM filter, lead-lag, liquidation detector, microprice, volatility, adverse selection
2. **Cascade check** -- if `liquidation_detector.should_pull_quotes()`, return defensive quotes
3. **Regime-blended params** -- `blend_params_by_belief(&hmm_filter)` returns gamma, kappa_multiplier, spread_floor
4. **Adjusted microprice** -- local microprice + lead-lag predicted remaining move from Binance
5. **Compute kappa** -- Hawkes intensity x adverse selection adjustment x regime multiplier (floor 100.0)
6. **GLFT spread** -- `(1/gamma) * ln(1 + gamma/kappa) + fee`
7. **Apply adjustments** -- spread floor, HMM uncertainty multiplier (up to 20%), inventory skew, lead-lag skew
8. **Final prices** -- `bid = microprice * (1 - bid_depth)`, `ask = microprice * (1 + ask_depth)`
9. **Compute sizes** -- reduce size on side that would increase inventory (up to 80% reduction)
10. **Build output** -- populate QuoteSet with all diagnostics
11. **Log predictions** -- async prediction logging for calibration

### Key Helper Methods

- **`compute_kappa()`** -- combines Hawkes intensity, adverse selection adjustment, and regime multiplier; floors at 100.0 to prevent GLFT blow-up
- **`compute_inventory_skew()`** -- linear skew proportional to `inventory / max_inventory`, capped at `max_inventory_skew_bps`
- **`compute_sizes()`** -- reduces quote size on the side that would increase inventory exposure
- **`defensive_quotes()`** -- wide spread (50 bps default), minimal size (10% of max), used during liquidation cascades

See [implementation.md](./implementation.md) for full Rust code of all structs and methods.

---

## Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `base_gamma` | 0.5 | Risk aversion |
| `maker_fee_bps` | 1.5 | Hyperliquid maker fee |
| `max_inventory` | 1.0 | 1 BTC |
| `max_quote_size` | 0.1 | 0.1 BTC per side |
| `min_spread_bps` | 5.0 | Absolute floor |
| `max_inventory_skew_bps` | 10.0 | Skew cap |
| `max_lead_lag_skew_bps` | 5.0 | Cross-exchange skew cap |
| `liquidation_threshold` | 0.5 | Cascade trigger |
| `defensive_spread_bps` | 50.0 | Spread during cascades |

See [implementation.md](./implementation.md) for the `Default` impl.

---

## Testing Strategy

### Unit Tests

Four core invariant tests that must always pass:

1. **`spread_is_always_positive`** -- `ask > bid` across 1000 random market states
2. **`quotes_straddle_microprice`** -- bid below and ask above adjusted microprice
3. **`inventory_skew_direction`** -- long inventory widens bid, tightens ask
4. **`cascade_triggers_defensive`** -- rapid OI drops trigger wide defensive spreads (>= 25 bps)

See [implementation.md](./implementation.md) for full test code.

### Integration Tests

**`replay_historical_day`** -- loads a full day of market data, runs the quote engine tick-by-tick, simulates fills against historical trades, and reports spread capture vs adverse selection PnL. See [implementation.md](./implementation.md) for full test code.

### Paper Trading Validation

Before live deployment:

1. Run paper trading for 1 week minimum
2. Compare against calibration predictions:
   - Fill rates should match predicted fill rates
   - Adverse selection should match predicted toxicity
3. PnL should be positive (or understand why not)
4. No unexpected regime behavior

---

## Operational Checklist

Before enabling:

- [ ] All model components trained on recent data
- [ ] Calibration metrics acceptable (IR > 1.0 for each model)
- [ ] Paper trading shows positive PnL
- [ ] Circuit breakers tested
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set
- [ ] Daily report automation running

---

## Dependencies

- **Requires**: All foundation and model skills
- **Enables**: Live trading, daily-calibration-report

## Common Mistakes

1. **Not logging predictions**: Can't debug without measurement
2. **Missing defensive mode**: Cascade protection is critical
3. **Hard-coded parameters**: Everything should be regime-dependent
4. **Synchronous model updates**: Keep hot path fast
5. **Ignoring uncertainty**: Widen when HMM is uncertain

## Next Steps

1. Implement basic quote engine (Steps 1-8)
2. Add measurement integration (Step 11)
3. Unit test all invariants
4. Integration test on historical data
5. Paper trade for 1 week
6. Enable live with small size
7. Gradually increase size as confidence grows

## Supporting Files

- [implementation.md](./implementation.md) -- All Rust code: MarketData struct, QuoteSet struct, QuoteEngine full implementation (generate_quotes pipeline, update_models, compute_kappa, compute_inventory_skew, compute_sizes, defensive_quotes, log_predictions), configuration defaults, unit tests, integration tests
