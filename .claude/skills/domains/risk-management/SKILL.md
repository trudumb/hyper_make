---
name: Risk Management
description: Layered risk system with monitors, circuit breakers, kill switch, and position guards. Read when working on risk/, safety/, or monitoring/ modules, or when debugging position limits, emergency shutdowns, or spread widening.
user-invocable: false
---

# Risk Management Skill

## Purpose

Understand and modify the layered risk management system that protects the market maker from catastrophic loss. This system gates every quote cycle — if risk says "no", no quotes go out.

**Defense first**: when uncertain, widen spreads. Missing a trade is cheap; getting run over in a cascade is not.

## When to Use

- Working on `risk/`, `safety/`, or `monitoring/` modules
- Debugging why quotes are being gated or spreads are widening
- Adding new risk monitors or circuit breakers
- Investigating position limit breaches or kill switch triggers
- Understanding why the system stopped quoting

## Module Map

All paths relative to `src/market_maker/`:

```
risk/
  mod.rs              # Re-exports, RiskState, RiskAggregator, RiskMonitor trait
  state.rs            # RiskState — unified snapshot of all risk-relevant data
  monitor.rs          # RiskMonitor trait, RiskSeverity, RiskAction, RiskAssessment
  aggregator.rs       # AggregatedRisk — collects assessments, computes max severity
  limits.rs           # RiskLimits, RiskChecker — soft/hard position & order limits
  circuit_breaker.rs  # CircuitBreakerMonitor — market condition triggers
  drawdown.rs         # DrawdownTracker — equity drawdown with daily/lifetime thresholds
  kill_switch.rs      # KillSwitch — immediate trading halt
  position_guard.rs   # PositionGuard — inventory limits & soft threshold scaling
  reentry.rs          # ReentryController — controlled re-engagement after drawdown
  monitors/
    cascade.rs        # CascadeMonitor — OI-based liquidation cascade detection
    data_staleness.rs # DataStalenessMonitor — feed staleness triggers cautious mode
    drawdown.rs       # DrawdownMonitor — position-specific drawdown
    loss.rs           # LossMonitor — cumulative loss monitoring
    position.rs       # PositionMonitor — concentration & leverage limits
    rate_limit.rs     # RateLimitMonitor — order rate limit tracking
safety/
  auditor.rs          # SafetyAuditor — periodic state reconciliation
monitoring/
  alerter.rs          # Alerter — thread-safe alerting with deduplication
  dashboard.rs        # DashboardState — real-time terminal display
  postmortem.rs       # Post-trade analysis
```

---

## Architecture

### Data Flow

All monitors evaluate the **same `RiskState` snapshot** in a single pass. This prevents race conditions, stale data, and partial updates.

```
Market Data + Position State + PnL
  -> RiskState (unified snapshot)
  -> [LossMonitor, PositionMonitor, CascadeMonitor, ...]
  -> RiskAggregator (max severity, spread factor, kill reasons)
  -> Quote Engine (gates quotes, widens spreads, or kills trading)
```

### RiskMonitor Trait

Every risk monitor implements:

```rust
trait RiskMonitor {
    fn name(&self) -> &str;
    fn evaluate(&self, state: &RiskState) -> RiskAssessment;
}

struct RiskAssessment {
    severity: RiskSeverity,     // Normal, Caution, Warning, Critical, Emergency
    action: RiskAction,         // Continue, WidenSpreads(factor), ReduceSize, PullQuotes, KillSwitch
    reason: String,
    spread_multiplier: f64,     // 1.0 = no change, 2.0 = double spreads
}
```

### Severity Escalation

```
Normal -> Caution -> Warning -> Critical -> Emergency
  noop    log only   widen 1.5x  pull quotes  kill switch
```

The `RiskAggregator` takes the **maximum severity** across all monitors.

---

## Monitor Catalog

### 1. LossMonitor (`monitors/loss.rs`)
- Tracks cumulative realized + unrealized PnL
- **Thresholds**: warning at 50% of daily limit, critical at 80%, emergency at 100%

### 2. PositionMonitor (`monitors/position.rs`)
- Hard invariant: `inventory.abs() <= max_inventory`
- Soft threshold at ~70% — begins reducing quote sizes
- Monitors concentration across assets in multi-asset mode

### 3. CascadeMonitor (`monitors/cascade.rs`)
- Detects liquidation cascades via OI drops > 2% in 1 minute
- Immediately widens spreads and may pull quotes
- **Critical for crypto**: cascades can move price 5-10% in seconds

### 4. DataStalenessMonitor (`monitors/data_staleness.rs`)
- Monitors feed freshness (L2 book, trades, Binance)
- Triggers cautious mode if data older than 5s
- Prevents quoting on stale information

### 5. DrawdownMonitor (`monitors/drawdown.rs`)
- Peak-to-trough drawdown per position and aggregate
- Daily and lifetime drawdown limits via `DrawdownTracker` with high-water mark

### 6. RateLimitMonitor (`monitors/rate_limit.rs`)
- Proactive: slows down before hitting exchange rate limits
- Reactive: backs off after receiving rate limit rejections

---

## Circuit Breaker System

`CircuitBreakerMonitor` in `circuit_breaker.rs` handles market-condition triggers:

| Trigger | Detection | Response |
|---------|-----------|----------|
| OI drop > threshold | `open_interest_delta_1m` | Widen spreads, may pull quotes |
| Funding extreme | `abs(funding_rate) > threshold` | Widen spreads |
| Spread blowout | Market spread > 5x normal | Reduce size or pause |
| Fill collapse | Fill rate drops to 0 | Check connectivity |
| Model degradation | IR < critical threshold | Reduce model weight |

---

## Kill Switch

`KillSwitch` in `kill_switch.rs` — last line of defense:

- **Immediate**: cancels all orders, closes positions at market
- **Triggered by**: Emergency-level assessments from any monitor
- **Requires manual reset**: system won't restart automatically
- **Logs everything**: full state dump for post-mortem

---

## Position Guard

`PositionGuard` enforces inventory limits with soft/hard thresholds:

- 0-70% of limit: full quote sizes on both sides
- 70-100% of limit: linearly reduce quote size on the expanding side
- At 100%: only quotes that reduce position (reduce-only mode)

---

## Re-entry After Drawdown

`ReentryController` in `reentry.rs`:

1. After significant drawdown, don't jump back to full size
2. Gradually increase position limits over time
3. Reset if another drawdown occurs during recovery
4. Configurable recovery period and scaling curve

---

## Safety Auditor

`SafetyAuditor` in `safety/auditor.rs` — periodic reconciliation:

- Order cleanup (expired fill windows)
- Stale pending detection (orders not on exchange)
- Stuck cancel detection (cancel requests that didn't execute)
- Orphan reconciliation (exchange orders not in local tracking)
- Reduce-only reporting

---

## Quote Engine Integration

Risk assessments flow into the quote engine via `AggregatedRisk`:
```
spread_factor = max(1.0, risk.spread_multiplier)
effective_spread = base_spread * spread_factor

if risk.action == PullQuotes { return no_quotes; }
if risk.action == KillSwitch { cancel_all_and_halt(); }
```

---

## Key Invariants

These must ALWAYS hold — violations are bugs:

1. `inventory.abs() <= max_inventory`
2. `ask_price > bid_price`
3. `spread >= min_spread_bps` (never tighter than fee + minimum edge)
4. Kill switch state persists across restarts

---

## Common Debugging

### "Why did it stop quoting?"
1. Check `DashboardState` for active risk triggers
2. Look at `RiskAggregator` — which monitor returned highest severity?
3. Check kill switch (requires manual reset)
4. Verify data freshness — stale data gates quotes

### "Why are spreads so wide?"
1. Check `spread_multiplier` in `AggregatedRisk`
2. Cascade monitor — OI drop detected?
3. Drawdown state — in recovery mode?
4. Regime detection — high-vol regime naturally widens

### "Position limit breach"
1. Check `PositionGuard` — hard limit hit?
2. Verify reduce-only mode active
3. Check for orphan fills
4. Run `SafetyAuditor` for full reconciliation

---

## Adding a New Risk Monitor

1. Implement `RiskMonitor` trait in `risk/monitors/`
2. Add to monitor list in `RiskAggregator`
3. Use `RiskState` fields only — don't add new data sources to hot path
4. Default to `RiskSeverity::Normal` when uncertain
5. Include clear `reason` strings for debugging
6. Add tests covering all severity transitions
