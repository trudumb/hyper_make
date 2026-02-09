---
name: Measurement Infrastructure
description: Build the prediction logging and outcome tracking system that all other model improvements depend on
user-invocable: false
---

# Measurement Infrastructure Skill

## Purpose

Build the prediction logging and outcome tracking system that all other model improvements depend on. This is the foundation - without proper measurement, you can't know if anything else is working.

**READ THIS SKILL FIRST before any model work.**

## When to Use

- Setting up a new trading system from scratch
- Adding prediction logging to an existing quote engine
- Building the data pipeline for calibration analysis
- Any time you're about to build a predictive model

## Core Principle

Every quote cycle, your system makes implicit predictions:
- "If I place a bid at this price, the probability of fill in 1s is X%"
- "If I get filled, the probability of adverse selection is Y%"
- "The price will move Z bps in the next 10 seconds"

These predictions must be **recorded** with enough granularity to diagnose failures.

---

## Schema Definitions

### 1. Prediction Record (Top Level)

```rust
struct PredictionRecord {
    timestamp_ns: u64,
    quote_cycle_id: u64,
    market_state: MarketStateSnapshot,
    predictions: ModelPredictions,
    outcomes: Option<ObservedOutcomes>,
}
```

### 2. Market State Snapshot

Capture everything the model could condition on:

```rust
struct MarketStateSnapshot {
    // L2 Book State
    bid_levels: Vec<(f64, f64)>,  // (price, size) top N
    ask_levels: Vec<(f64, f64)>,
    spread_bps: f64,
    microprice: f64,
    book_imbalance: f64,  // (bid_size - ask_size) / total

    // Kappa
    kappa_book: f64,
    kappa_robust: f64,
    kappa_own: f64,
    kappa_final: f64,

    // Volatility
    sigma_bipower: f64,
    sigma_realized_1m: f64,
    sigma_realized_5m: f64,

    // Gamma
    gamma_base: f64,
    gamma_effective: f64,

    // Hyperliquid-specific
    funding_rate: f64,
    time_to_funding_settlement_s: f64,
    open_interest: f64,
    open_interest_delta_1m: f64,

    // Cross-exchange
    binance_mid: Option<f64>,
    binance_hl_basis_bps: Option<f64>,

    // Position
    inventory: f64,
    inventory_age_s: f64,

    // Regime
    regime_quiet_prob: f64,
    regime_trending_prob: f64,
    regime_volatile_prob: f64,
    regime_cascade_prob: f64,
}
```

### 3. Model Predictions

```rust
struct ModelPredictions {
    levels: Vec<LevelPrediction>,
    expected_fill_rate_1s: f64,
    expected_adverse_selection_bps: f64,
    predicted_price_direction_1s: f64,  // [-1, 1]
    direction_confidence: f64,
}

struct LevelPrediction {
    side: Side,
    price: f64,
    size: f64,
    depth_from_mid_bps: f64,
    p_fill_1s: f64,
    p_fill_10s: f64,
    p_adverse_given_fill: f64,
    expected_pnl_given_fill: f64,
}
```

### 4. Observed Outcomes

```rust
struct ObservedOutcomes {
    fills: Vec<FillOutcome>,
    price_1s_later: f64,
    price_10s_later: f64,
    price_60s_later: f64,
    realized_adverse_selection_bps: f64,
}

struct FillOutcome {
    level_index: usize,
    fill_timestamp_ns: u64,
    fill_price: f64,
    fill_size: f64,
    mark_price_at_fill: f64,
    mark_price_1s_later: f64,
    mark_price_10s_later: f64,
}
```

---

## Implementation Checklist

### Step 1: Instrument Quote Generation

Capture market state BEFORE computing quotes, extract predictions, log as JSONL (one record per line). Outcomes are filled asynchronously via the outcome matcher.

### Step 2: Build Async Outcome Matcher

Track pending predictions in a HashMap by cycle_id. On fill events, match to the originating prediction. On price updates, fill in price evolution fields. Flush completed records (age > max_horizon_s) to JSONL.

### Step 3: Storage Layer

Use **JSONL** (one JSON record per line) for persistence. This is what the actual codebase uses — see `src/market_maker/analytics/persistence.rs`. Files are written to `logs/` with rotation by date. The `let _ =` pattern ensures logging never crashes the trader.

---

## What to Log vs What to Skip

### Must Log (Critical)
- All fill probability predictions
- All fill outcomes
- Post-fill price evolution (for adverse selection)
- Market state at prediction time
- Regime probabilities

### Should Log (Important)
- Kappa inputs and outputs
- Gamma inputs and outputs
- Queue position estimates
- Cross-exchange state

### Can Skip (Space Optimization)
- Full L2 book beyond top 5 levels
- Sub-100ms price updates
- Predictions for orders that were never placed

---

## Dependencies

- **Requires**: Your existing quote engine, market data feed
- **Enables**: calibration-analysis, signal-audit, all model skills

## Common Mistakes

1. **Logging only aggregates**: You need per-prediction granularity to diagnose issues
2. **Missing market state**: Without conditioning variables, you can't do conditional calibration
3. **Synchronous outcome filling**: This blocks the hot path; must be async
4. **Not logging "boring" periods**: Quiet market data is just as important for calibration
5. **Forgetting to log predictions for orders that didn't fill**: These are negative examples

---

## Next Steps

Once this infrastructure is in place:
1. Read `calibration-analysis/SKILL.md` to analyze the logged data
2. Read `signal-audit/SKILL.md` to measure signal information content
3. Then proceed to specific model skills
