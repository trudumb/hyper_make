# Phase 3: Production Hardening — Before Live Capital

## Prerequisites
Phase 2 complete: Paper Sharpe > 2.0, Brier < 0.20, max drawdown < 3%, 5000+ fills collected.

## Mission
Harden the system for real money. Every failure mode must be anticipated. A bug here costs real capital.

---

## Agent A: Exchange Integration Stress Test

### Files Owned
- `src/market_maker/infra/ws_executor.rs`
- `src/market_maker/infra/reconnection.rs`
- `src/market_maker/infra/rate_limit/`
- `src/market_maker/execution/order_lifecycle.rs`

### Tests Required

#### A1: WebSocket Reconnection
- Simulate disconnect after 100 messages
- Verify: reconnects within 5s, resumes order state, no orphaned orders
- Verify: rate limit budget resets correctly on reconnect
- Verify: stale data detection fires if reconnect takes > 30s

#### A2: Order Lifecycle
- Place → Fill → verify inventory update
- Place → Cancel → verify no phantom position
- Place → Partial Fill → Cancel remainder → verify accounting
- Place → Reject → verify no position change
- Rapid place/cancel (10/s) → verify no rate limit violation

#### A3: Exchange Outage
- Simulate 60s of no data → verify kill switch fires
- Simulate intermittent data (every 5s) → verify degraded mode
- Simulate order rejection storm → verify graceful backoff

#### A4: Latency Profiling
- Measure: quote generation time (target < 1ms)
- Measure: order placement round-trip (target < 100ms)
- Measure: Binance→Hyperliquid signal propagation (target < 50ms)

### Checkpoints
| # | Test | Pass Criteria |
|---|------|--------------|
| 1 | Reconnection test | Reconnects 10/10 times, < 5s each |
| 2 | Order lifecycle | All 5 scenarios pass |
| 3 | Exchange outage | Kill switch fires correctly |
| 4 | Latency profiling | All targets met |

---

## Agent B: Risk & Capital Management

### Files Owned
- `src/market_maker/risk/aggregator.rs`
- `src/market_maker/risk/kill_switch.rs`
- `src/market_maker/risk/position_guard.rs`
- `src/market_maker/infra/margin.rs`

### Production Risk Configuration

#### B1: Position Limits
```rust
// Based on account equity and Kelly criterion
max_position_btc = account_equity / (leverage * btc_price) * kelly_fraction
// Example: $10K equity, 3x leverage, $100K BTC, 0.5 Kelly
// max_position = 10000 / (3 * 100000) * 0.5 = 0.0167 BTC
```

Start ultra-conservative: 0.01 BTC max position ($1000 notional at $100K BTC).

#### B2: Kill Switch Thresholds (Production)
| Trigger | Threshold | Action |
|---------|-----------|--------|
| Daily loss | $50 (0.5% of $10K) | Kill all, cancel all |
| Drawdown | 2% | Kill all, cancel all |
| Position runaway | 2× max_position | Kill all |
| Stale data | 10 seconds | Pull quotes |
| Cascade severity | > 1.5 | Pull quotes |
| Rate limit errors | 2 consecutive | Kill all |

#### B3: Gradual Re-entry After Kill
- After kill switch: wait 5 minutes
- Re-enter with 50% position limits
- If killed again within 1 hour: wait until next day
- If killed 3 times in a day: manual review required

#### B4: Margin Monitoring
- Track margin utilization every tick
- Alert at 60% utilization
- Reduce position at 80%
- Kill at 90%

### Checkpoints
| # | Test | Pass Criteria |
|---|------|--------------|
| 1 | Position limits enforce correctly | Cannot exceed max_position |
| 2 | Kill switch fires at all thresholds | Each trigger tested |
| 3 | Gradual re-entry works | Re-enters at 50% after 5 min |
| 4 | Margin monitoring accurate | Matches exchange margin API |

---

## Agent C: Latency Optimization

### Files Owned
- `src/market_maker/infra/binance_feed.rs`
- `src/market_maker/infra/ws_executor.rs` (parsing only)
- Hot path profiling

### Optimizations

#### C1: Binance Feed Zero-Copy Parsing
- Current: JSON parsing with serde
- Target: Zero-copy with simd-json or manual parsing for aggTrade
- Expected improvement: 5-20ms reduction in feed-to-signal latency

#### C2: Quote Generation Hot Path
- Profile `generate_ladder()` end-to-end
- Target: < 500μs per quote cycle
- Identify allocations in hot path, replace with arena allocation

#### C3: Order Batching
- Batch cancel-and-replace into single API call where possible
- Reduce API round-trips from 2 (cancel + place) to 1 (modify)

### Checkpoints
| # | Optimization | Measurement |
|---|-------------|-------------|
| 1 | Feed parsing profiled | Baseline latency measured |
| 2 | Zero-copy implemented | Latency reduced by > 5ms |
| 3 | Quote generation profiled | < 500μs confirmed |
| 4 | Order batching | API calls reduced by > 30% |

---

## Agent D: Monitoring & Alerting

### Files Owned
- `src/bin/health_dashboard.rs`
- `src/market_maker/monitoring/`
- New: alerting infrastructure

### Requirements

#### D1: Real-Time Dashboard
Display every 10 seconds:
```
[LIVE] PnL=$12.34 Position=0.005BTC Spread=4.2bps Fills=47 Regime=Normal
       Kappa=8234 Gamma=0.07 WinRate=54% FillRate=3.2/min
       Signals: lead_lag=+0.8bps alpha=0.12 falling_knife=0.0
       Risk: drawdown=0.3% margin_util=15% kill=OK
```

#### D2: Alerts (to stdout/file, expandable to Telegram/Discord)
| Alert | Condition | Severity |
|-------|-----------|----------|
| Drawdown warning | > 1% | WARN |
| Drawdown critical | > 1.5% | CRITICAL |
| No fills 5 min | fill_count unchanged 5 min | WARN |
| Signal degradation | lead_lag MI < 0.05 for 10 min | WARN |
| Spread blow-up | avg_spread > 20 bps for 2 min | WARN |
| Inventory imbalance | abs(inventory) > 0.7 × max | WARN |
| Regime transition | regime changed | INFO |
| Kill switch armed | any monitor at HIGH | CRITICAL |

#### D3: Post-Mortem Logging
On every kill switch trigger, write detailed state dump:
```json
{
  "timestamp": "2026-02-07T12:34:56Z",
  "trigger": "drawdown",
  "value": 0.021,
  "threshold": 0.02,
  "position": 0.015,
  "unrealized_pnl": -210.50,
  "regime": "High",
  "recent_fills": [...last 10 fills...],
  "signal_state": {...},
  "market_state": {...}
}
```

### Checkpoints
| # | Feature | Verification |
|---|---------|-------------|
| 1 | Dashboard displays all fields | Manual inspection |
| 2 | Alerts fire at correct thresholds | Simulate each condition |
| 3 | Post-mortem logging works | Trigger kill switch, verify dump |
| 4 | Log rotation | Logs don't grow unbounded |

---

## Live Deployment Protocol

### Day 1: Observation Only
```bash
# Connect to live feeds, run strategy, but do NOT place orders
cargo run --release --bin market_maker -- --observe-only --asset BTC
```
Verify: signals produce values, quotes would be competitive, no errors.

### Day 2-3: Micro Position ($100 notional max)
- max_position: 0.001 BTC
- max_daily_loss: $5
- Run for full trading day
- Compare live fills to paper predictions

### Week 1: Small Position ($1K notional max)
- max_position: 0.01 BTC
- max_daily_loss: $20
- Compare live Sharpe to paper Sharpe (should be within 0.5)

### Week 2-4: Scale Up
- Increase position 2× per week if live Sharpe > 1.5
- Monitor: slippage vs paper, fill quality vs paper, AS vs paper
- If live Sharpe < paper Sharpe by > 1.0: stop and investigate

### Steady State
- Position sized to Kelly criterion (typically 10-30% of max theoretical)
- Daily P&L review
- Weekly signal health review
- Monthly full recalibration

---

## Emergency Procedures

### Scenario: Unexpected Loss > 2× Daily Limit
1. Kill switch fires automatically
2. Cancel all open orders (verify via API)
3. Check: is position flat? If not, market-close immediately
4. Check exchange status (outage? delisting?)
5. Do NOT restart until root cause identified

### Scenario: Exchange API Down
1. Kill switch fires on stale data
2. Orders remain on exchange book (can't cancel)
3. Monitor exchange status page
4. When API returns: immediately cancel all, verify position
5. Resume only after confirming state consistency

### Scenario: Signal Degradation
1. Lead-lag MI drops to 0 (edge gone)
2. Widen spreads 2× (reduce fill rate, reduce exposure)
3. Monitor for 1 hour
4. If signal doesn't recover: switch to defensive mode (wider spreads, smaller positions)
5. Investigate: has the cross-venue relationship changed?
