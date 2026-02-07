---
name: Lead-Lag Estimator
description: Exploit the fact that Binance BTC perpetual leads Hyperliquid for microprice adjustment and quote skew
---

# Cross-Exchange Lead-Lag Model Skill

## Purpose

Exploit the fact that Binance BTC perpetual leads Hyperliquid by 50-500ms. This is potentially your highest-value proprietary signal because:

- It's based on physical reality (Binance has 10-100x more volume)
- It's predictive of short-term direction
- It enables microprice adjustment and quote skew
- It's continuously measurable and adaptable

## When to Use

- You have a cross-exchange data feed (Binance websocket)
- Building directional microprice adjustment
- Adding quote skew based on recent price moves elsewhere
- Improving adverse selection detection (arbitrage flow)

## Prerequisites

- Real-time Binance BTC perpetual market data feed
- Synchronized timestamps (NTP or better)
- `measurement-infrastructure` for validation
- `signal-audit` to quantify information content

---

## Core Observation

Price discovery happens on Binance first because:
1. Higher volume (10-100x Hyperliquid)
2. More sophisticated participants
3. Lower latency infrastructure
4. Deeper liquidity

When Binance price moves, Hyperliquid follows with a lag of 50-500ms depending on:
- Volatility regime (high vol = faster arbitrage = shorter lag)
- Time of day (more participants = shorter lag)
- Move size (larger moves propagate faster)

---

## Lead-Lag Estimation

### Basic Model

```
HL_return(t) = beta * Binance_return(t - lag) + epsilon
```

Where:
- lag = optimal lead time (typically 100-300ms)
- beta = transmission coefficient (typically 0.6-0.9)
- epsilon = noise (Hyperliquid-specific flow)

### Estimator Implementation

`LeadLagEstimator` uses ring buffers for Binance/HL price returns, grid-search over candidate lags (-100ms to 500ms), and online linear regression with exponential smoothing. Key fields: `lag_estimate_ms`, `beta_estimate`, `r_squared`. Re-estimates every second once `min_samples` (100) are collected.

See [implementation.md](./implementation.md#leadlagestimator) for full code.

### Regime-Conditioned Lead-Lag

The lead-lag relationship varies with volatility. `RegimeConditionedLeadLag` maintains separate `LeadLagEstimator` instances per `VolatilityRegime` (Low/Medium/High), classified by vol thresholds (1bp, 3bp, 10bp per 100ms).

Key method: `predict_remaining_move(binance_return, time_since_move_ms)` -- returns expected remaining HL move scaled by R-squared confidence, or 0.0 if R-squared < 0.1 or outside the lag window.

See [implementation.md](./implementation.md#regimeconditionedleadlag) for full code.

---

## Integration into Quote Engine

### Microprice Adjustment

`compute_adjusted_microprice` shifts the local microprice by the predicted remaining move from the lead-lag model: `microprice * (1.0 + predicted_move)`.

See [implementation.md](./implementation.md#microprice-adjustment) for full code.

### Quote Skew

`compute_lead_lag_skew_bps` computes directional skew when Binance moves and the lag hasn't elapsed. Skew magnitude = `move_bps * sqrt(R^2) * time_decay * 0.5`, capped at `max_skew_bps`. Returns 0.0 if R-squared < 0.2 or outside lag window.

See [implementation.md](./implementation.md#quote-skew) for full code.

### Full Integration

`QuoteEngine::generate_quotes_with_lead_lag` ties everything together: updates the regime, feeds prices, detects significant Binance moves (>1 bps), adjusts microprice, computes lead-lag skew, then applies both to bid/ask depths alongside inventory skew.

See [implementation.md](./implementation.md#full-quote-engine-integration) for full code.

---

## Monitoring and Decay Detection

### Signal Quality Tracking

`LeadLagMonitor` tracks R-squared history and checks both current level and 24h decay rate. Thresholds: warning at R-squared < 0.15, critical at R-squared < 0.10, decay warning at >10%/day.

See [implementation.md](./implementation.md#signal-quality-monitor) for full code.

### Expected Decay Pattern

Lead-lag signal decays as:
1. More arbitrageurs enter Hyperliquid
2. Technology improves (lower latency)
3. Liquidity deepens

**Typical half-life: 6-12 months**

Plan to:
- Monitor R-squared continuously
- Reduce reliance as R-squared drops
- Eventually this edge will be arbitraged away

---

## Validation

### Backtesting

`backtest_lead_lag_value` runs the quote engine with lead-lag enabled/disabled to measure PnL, Sharpe, and adverse selection rate improvements.

See [implementation.md](./implementation.md#backtesting) for full code.

### Key Metrics

- **R-squared > 0.2**: Signal is useful
- **R-squared > 0.3**: Signal is strong
- **R-squared > 0.4**: Signal is exceptional (rare)

Compare PnL with/without lead-lag:
- **Expected improvement: 10-30%** in spread capture
- **Adverse selection should decrease** (you anticipate moves)

---

## Dependencies

- **Requires**: Binance websocket feed, synchronized timestamps
- **Enables**: Microprice adjustment, quote skew, arbitrage flow detection

## Common Mistakes

1. **Wrong timestamp sync**: Even 10ms error ruins the estimate
2. **Too aggressive skew**: Don't rely too heavily on a decaying signal
3. **Ignoring regime**: Low-vol has different lag than high-vol
4. **Not monitoring decay**: This edge will shrink over time
5. **Symmetric treatment**: Lag may differ for up vs down moves

## Data Requirements

- Binance BTC-PERP book ticker: ~10ms resolution
- Hyperliquid BTC book ticker: ~50ms resolution minimum
- Timestamps: NTP-synchronized or better
- Storage: ~10GB/month for tick data

## Next Steps

1. Set up Binance websocket feed
2. Implement basic LeadLagEstimator
3. Validate R-squared > 0.2 on historical data
4. Add regime conditioning
5. Integrate microprice adjustment
6. Add quote skew
7. Set up decay monitoring
8. Plan for eventual edge erosion

## Supporting Files

- [implementation.md](./implementation.md) -- All Rust code: LeadLagEstimator (with grid-search regression), RegimeConditionedLeadLag (per-regime estimators, predict_remaining_move), microprice adjustment, quote skew, full quote engine integration, signal quality monitor, backtesting framework
