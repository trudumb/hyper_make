---
name: live-incident-response
description: Structured triage for live market maker incidents. Use when MM stops quoting, position spikes unexpectedly, kill switch triggers, spreads blow up, rate limits hit, or data goes stale. Decision-tree diagnosis with exact file:line references from past incidents.
disable-model-invocation: true
argument-hint: "[symptom-description]"
---

# Live Incident Response

Structured triage for Hyperliquid market maker incidents. Each section is a decision tree
that walks through diagnosis of a specific symptom, with exact file:line references
from the codebase and known past incidents.

**Principles**:
- Defense first: when uncertain, widen spreads or pull quotes. Missing a trade is cheap.
- Diagnose before acting: understand the root cause before resetting anything.
- Record everything: save state dumps, log the incident in memory, update thresholds.

**Quick Reference**: Jump to the symptom that matches your incident:
- [A. MM Stopped Quoting](#a-mm-stopped-quoting)
- [B. Position Spiked](#b-position-spiked)
- [C. Kill Switch Triggered](#c-kill-switch-triggered)
- [D. Spreads Too Wide](#d-spreads-too-wide)
- [E. Rate Limited](#e-rate-limited)
- [F. Data Stale](#f-data-stale)
- [G. Post-Incident Checklist](#g-post-incident-checklist)

---

## A. MM Stopped Quoting

When the market maker is running but no orders appear on the book, check these in order.
Each step either identifies the root cause or directs you to the next check.

### Step 1: Is the kill switch triggered?

**File**: `src/market_maker/risk/kill_switch.rs`
- `KillSwitch::is_triggered()` (line 443) -- atomic bool, fast check
- `KillSwitch::trigger_reasons()` (line 448) -- get all reasons

The kill switch is latching: once triggered, it stays triggered until manually reset via
`KillSwitch::reset()` (line 461). It persists across restarts via checkpoint
(`restore_from_checkpoint()`, line 917) with a 24-hour expiry window.

**Check**: Look for log lines containing "kill switch triggered" or call `summary()` (line 866).

If triggered, go to [Section C: Kill Switch Triggered](#c-kill-switch-triggered).

### Step 2: Check the risk monitor system

**File**: `src/market_maker/risk/aggregator.rs`
- `RiskAggregator` evaluates all monitors and takes **maximum severity** (line 170-171).
- One Critical monitor overrides any number of Normal monitors.

The eight risk monitors to check (all in `src/market_maker/risk/monitors/`):

| Monitor | File | What it checks | Triggers at |
|---------|------|---------------|-------------|
| `LossMonitor` | `loss.rs:34` | Daily PnL vs limit | Loss > `max_daily_loss` |
| `DrawdownMonitor` | `drawdown.rs:44` | Peak-to-trough drawdown | Drawdown > `max_drawdown` (skips if peak < `min_peak_for_drawdown`) |
| `PositionMonitor` | `position.rs:42` | Position size/value | Value > 2x soft limit = kill; 1x-2x = reduce-only |
| `PositionVelocityMonitor` | `position_velocity.rs:63` | Position change rate | 50%/min warn, 100%/min pull, 200%/min kill |
| `DataStalenessMonitor` | `data_staleness.rs:49` | Market data freshness | Data age > `stale_threshold` (default 30s, production 10s) |
| `CascadeMonitor` | `cascade.rs:47` | Liquidation cascade severity | Pull at 0.8, kill at 5.0 |
| `PriceVelocityMonitor` | `price_velocity.rs:39` | Flash crash detection | Pull at 5%/s, kill at 15%/s |
| `RateLimitMonitor` | `rate_limit.rs:39` | Exchange API errors | Kill at 3 errors (production: 2) |

**Action**: Check which monitor returned `RiskAction::PullQuotes` or `RiskAssessment::critical`.
The most severe assessment wins. A single `PullQuotes` from any monitor stops quoting.

### Step 3: Check the data quality gate

**File**: `src/market_maker/infra/data_quality.rs`
- `DataQualityMonitor::should_gate_quotes()` (line 287) -- called at quote_engine.rs:32

Three `QuoteGateReason` variants (line 72-79):
- `NoDataReceived` -- no L2 data for this asset yet (startup)
- `StaleData { age_ms, threshold_ms }` -- data age exceeds 15s threshold
- `CrossedBook { best_bid, best_ask }` -- impossible book state (bid >= ask)

This gate fires BEFORE any spread calculation. If gated, no quotes are generated at all.

### Step 4: Check the quote gate (directional)

**File**: `src/market_maker/control/quote_gate.rs`
- `QuoteDecision` enum (line 41) controls what gets quoted
- `NoQuoteReason` enum (line 78): `Warmup`, `NoEdgeFlat`, `Cascade`, `Manual`, `QuotaExhausted`, `ToxicFlow`

**Key**: `NoEdgeFlat` means no directional edge and flat position. The system is designed
to not quote when it has no view. This is intentional behavior, not a bug -- but it can
cause extended periods of no quoting if the edge signal is too conservative.

### Step 5: Check known historical causes

These are bugs that were found and fixed, but similar patterns may recur.

#### 5a. WaitToLearn Deadlock (Fixed Feb 10)

**File**: `src/market_maker/orchestrator/quote_engine.rs`
- The L3 readiness level previously required learning to complete before quoting.
- But learning requires fills, and fills require quoting -- deadlock.
- **Fix**: Changed to fall-through (quote anyway even if not fully learned).

**Pattern to watch for**: Any code path that gates quoting on a condition that can only be
satisfied by quoting. This is a cold-start deadlock.

#### 5b. Kappa Threshold Skip (Fixed Feb 10)

**File**: `src/market_maker/orchestrator/quote_engine.rs`
- Raw kappa from L2 book was ~6400-7700, compared against threshold of 5000.
- But `kappa_effective` (the blended value) was ~3250.
- **Fix**: Removed the feature entirely (redundant with rate limiter).

**Key lesson**: `self.estimator.kappa()` returns RAW market kappa. The kappa orchestrator's
`kappa_effective` is the blended value. Any code comparing kappa to thresholds must use
`kappa_effective`, not raw kappa.

#### 5c. Permanent Signal Staleness (Fixed Feb 9)

**File**: `src/market_maker/strategy/signal_integration.rs`
- `staleness_spread_multiplier()` (line 999) returns up to 2.0x when signals are stale.
- If `use_lead_lag` or `use_cross_venue` is true but no Binance feed exists (e.g., HIP-3
  DEX tokens), these signals never warm up, causing permanent 2.0x staleness.
- **Fix**: `disable_binance_signals()` (line 1040) sets both flags to false.

**Check**: Are Binance-dependent signals enabled for an asset with no Binance listing?

#### 5d. Cold-Start Staleness (Fixed Feb 10)

**File**: `src/market_maker/strategy/signal_integration.rs`
- `staleness_spread_multiplier()` (line 999-1033) checks `observation_counts() != (0, 0)`.
- Before the fix, cold start (observation_count == 0) would trigger the 2.0x multiplier.
- **Fix**: Added guard -- only penalize staleness if the signal was ever warmed up.

**Check**: Is `observation_count` zero for a signal that is flagged as stale?

#### 5e. Duplicate on_trade Handler (Fixed Feb 10)

**File**: `src/market_maker/orchestrator/handlers.rs`
- Two separate `on_trade` calls double-counted trades for kappa estimation.
- **Fix**: Removed the duplicate (the canonical one is in `messages/trades.rs`).

---

## B. Position Spiked

Position changed rapidly or unexpectedly. This is dangerous because it indicates either
a control failure or an external event (liquidation).

### Step 1: Cancel-Fill Race

The most common cause. An order is cancelled, but the cancel arrives at the exchange
after the order has already been filled.

**Pattern**: Cancel request sent -> fill arrives -> position spikes -> whipsaw.

**Diagnosis**: Check fill tracker for fills from recently-cancelled order CLOIDs.
The safety auditor (`src/market_maker/safety/`) runs periodic reconciliation between
local and exchange state.

**Impact**: Causes position whipsaw. The system handles this gracefully via reduce-only
mode, but the whipsaw itself costs money.

### Step 2: Inventory-Forcing One-Sided Quoting

**File**: `src/market_maker/control/quote_gate.rs`
- When API headroom drops below 10%, the system enters inventory-forcing mode.
- Only quotes the reducing side, trying to flatten position.
- If fills happen on one side only, position can drift rapidly.

**Diagnosis**: Check `rate_limit_headroom_pct` in logs. Below 10% triggers one-sided.

### Step 3: Orphan Fills

Fills from orders the system doesn't recognize. Could be from:
- A previous session's orders that weren't cancelled on shutdown
- Manual orders placed outside the MM
- Exchange-side order matching anomalies

**Diagnosis**: Check fill tracker for fills with unknown CLOIDs. The safety auditor
compares local state vs exchange state periodically.

### Step 4: Liquidation Detection

**File**: `src/market_maker/risk/kill_switch.rs`
- `check_liquidation()` (line 569) -- detects position jumps without corresponding fills.
- Threshold: position changes by > 20% of `max_position_contracts` with no fill in 5 seconds.
- Triggers `KillReason::LiquidationDetected` (line 303).

If this fired, the exchange may have liquidated the position. Check exchange UI immediately.

---

## C. Kill Switch Triggered

The kill switch latches on and blocks all trading. Understanding WHY it triggered is
critical before deciding whether to reset.

### Step 1: Read the Kill Reason

**File**: `src/market_maker/risk/kill_switch.rs`
- `KillReason` enum (line 282-307):
  - `MaxLoss { loss, limit }` -- daily PnL exceeded limit
  - `MaxDrawdown { drawdown, limit }` -- peak-to-trough exceeded limit
  - `MaxPosition { value, limit }` -- position value exceeded 2x soft limit
  - `PositionRunaway { contracts, limit }` -- contracts exceeded margin-based limit
  - `StaleData { elapsed, threshold }` -- no market data within threshold
  - `RateLimit { count, limit }` -- too many API errors
  - `CascadeDetected { severity }` -- liquidation cascade too intense
  - `Manual { reason }` -- operator triggered
  - `LiquidationDetected { position_delta, max_position }` -- unexplained position jump

### Step 2: Known Drawdown Display Bug

**File**: `src/market_maker/risk/kill_switch.rs`
- `summary()` (line 866-886) computes drawdown as `(peak_pnl - daily_pnl) / peak_pnl`.
- This is the DISPLAY formula. It can show 748% when peak_pnl is tiny (e.g., $0.02).
- The ACTUAL kill switch check at line 704-728 uses `min_peak_for_drawdown` guard
  (default $1.00, production = `max(1.0, max_position_value * 0.02)`).
- The risk state `drawdown()` method (`src/market_maker/risk/state.rs:253`) uses
  `account_value` as denominator -- this is the correct formula.

**Key**: The kill switch summary may show absurd drawdown percentages. Check the actual
`RiskState::drawdown()` (state.rs:253-262) which divides by `account_value`, not `peak_pnl`.

### Step 3: Recovery Procedure

1. **Verify actual drawdown**: Read `RiskState::drawdown()`, not the summary.
2. **Check position**: Is `inventory.abs() <= max_inventory`?
3. **Save checkpoint**: `KillSwitch::to_checkpoint()` (line 889) before any reset.
4. **When safe to reset**: Call `KillSwitch::reset()` (line 461).
5. **When NOT to reset**:
   - Real drawdown > 2% of account value
   - Position > max_inventory
   - Cascade still active (severity > 0.8)
   - Data still stale (no market data flowing)
   - You don't understand why it triggered

### Step 4: Checkpoint Persistence

**File**: `src/market_maker/risk/kill_switch.rs`
- `to_checkpoint()` (line 889) captures triggered state.
- `restore_from_checkpoint()` (line 917) re-triggers if checkpoint is < 24 hours old.
- Transient reasons (PositionRunaway) are NOT re-triggered on restore (line 948-951).
- Stale checkpoints (> 24 hours) are ignored (line 938-941).

---

## D. Spreads Too Wide

The market maker is quoting but with spreads so wide that fills are unlikely.
This is usually caused by multiplicative compounding of spread multipliers.

### The Spread Composition Chain

**File**: `src/market_maker/orchestrator/quote_engine.rs` (lines 992-1107)

Seven multiplicative factors are composed together. Each one independently widens
the spread. Because they multiply, several modest 1.5x factors compound to 3.4x+.

See [references/spread-composition.md](references/spread-composition.md) for the full chain.

### Diagnosis Steps

1. **Check logs**: The quote engine logs all multiplier components at line 1102-1107:
   ```
   toxicity=1.20x, defensive=1.00x, staleness=2.00x, model_gating=1.50x, total=3.60x
   ```

2. **Identify the dominant factor**: Usually one multiplier is much larger than the rest.

3. **Common patterns**:
   - `staleness=2.0x` -- Binance signals enabled but no feed (see A.5c)
   - `model_gating=1.5-2.0x` -- Model IR is poor, gating is widening (check calibration)
   - `toxicity=1.5-3.0x` -- VPIN or informed flow is elevated (may be correct)
   - `defensive=2.0-5.0x` -- Recent fills had negative gross edge (check edge_metrics)
   - Circuit breaker varies by type: OI cascade = cancel all, funding/spread/fill = 1.5-2.0x

4. **The multiplicative cascade problem**: Several independent 1.5x multipliers compound:
   - staleness 1.5x * model_gating 1.5x * toxicity 1.5x = 3.375x total
   - This is capped at `max_composed_spread_mult` (default 10.0x, line 1096-1097)
   - But even 3-4x is usually too wide to get filled

5. **Additive spread adjustments** are separate from the multiplicative chain:
   - Signal integration cap at 20 bps (`signal_integration.rs`)
   - This prevents the old 3.4x cascade bug from additive factors

### Resolution

- If `staleness` is the culprit: check data feeds, consider `disable_binance_signals()`.
- If `model_gating` is the culprit: check model calibration, Brier scores, IR.
- If `toxicity` is the culprit: this may be correct -- market IS toxic. Wait it out.
- If `defensive` is the culprit: check `EdgeMetricsTracker`, recent fill quality.
- If ALL factors are moderately elevated: this is the compounding problem. Consider
  reducing `max_composed_spread_mult` or switching to additive composition.

---

## E. Rate Limited

The exchange is rejecting API requests. This degrades quoting quality and can eventually
trigger the kill switch.

### Rate Limit Architecture

**Files**:
- `src/market_maker/infra/rate_limit/mod.rs` -- module overview
- `src/market_maker/infra/rate_limit/proactive.rs` -- proactive tracking
- `src/market_maker/infra/rate_limit/rejection.rs` -- rejection handling with backoff
- `src/market_maker/infra/rate_limit/error_type.rs` -- error classification

### Quota Tier Behavior

The system adjusts behavior based on API headroom percentage:

| Headroom | Tier | Behavior |
|----------|------|----------|
| >= 50% | Full | Normal quoting, no shadow spread |
| 20-50% | Normal | Mild shadow spread (0.5-2.5 bps) |
| 10-20% | Minimal | Noticeable shadow spread (2.5-5 bps), reduced ladder levels |
| 5-10% | Conservation | Aggressive shadow spread (5-10 bps), inventory-forcing disabled |
| 1-5% | Critical | Prohibitive shadow spread (10-50 bps), epsilon probes blocked |
| < 1% | Exhausted | Hard veto on all quotes (`quote_gate.rs`) |

**File**: `src/market_maker/control/quote_gate.rs`
- Shadow spread: `continuous_shadow_spread_bps()` (line 885) = `lambda / headroom.max(0.01)`
- Ladder density: `continuous_ladder_levels()` (line 901) = `sqrt(headroom / min_headroom) * max_levels`
- Inventory-forcing activation: headroom < 10% (line 1041)
- Spread multiplier: `1/sqrt(headroom)` capped at 5.0x (line 1056-1057)
- Epsilon probe: blocked below 10% headroom (line 987)

### Known Issue: HIP-3 7% Headroom

HIP-3 (hyna DEX token) has been observed at 7% API headroom permanently. This forces
the system into a degraded state:

- Inventory-forcing mode (one-sided quoting)
- ~3.78x spread multiplier (1/sqrt(0.07))
- Position whipsaw from one-sided fills

**Root cause**: Unknown account tier limitation on Hyperliquid. Investigate API quota.

### Workarounds

- Reduce `quote_levels` (fewer orders = fewer API calls)
- Increase `min_change_threshold` (avoid trivial order modifications)
- Increase `quote_interval_ms` (less frequent quote cycles)
- Switch to passive mode (reduce-only on one side)

---

## F. Data Stale

No market data is flowing. This is dangerous because the MM may be quoting on stale prices.

### Three Data Feeds to Check

1. **Hyperliquid L2/Trades** -- primary market data feed
   - L2 book snapshots and trade stream
   - Feeds into kappa estimation, signal integration, quote engine

2. **Hyperliquid State** -- account state feed
   - Position, balance, open orders
   - Feeds into risk management, reconciliation

3. **Binance** -- cross-venue reference feed (optional)
   - Lead-lag estimation, cross-venue flow
   - Not available for HIP-3/DEX tokens

### Connection Health Infrastructure

**File**: `src/market_maker/infra/reconnection.rs`
- `ConnectionState` enum (line 48-59): `Healthy`, `Stale`, `Reconnecting`, `Disconnected`, `Failed`
- `ConnectionHealthMonitor` (line 78) tracks per-connection health
- `is_data_stale()` (line 144) checks against `stale_data_threshold` (default 30s)
- State machine: `Healthy` -> `Stale` -> `Reconnecting` -> `Healthy` or `Failed`
- Exponential backoff: 1s initial, 2x multiplier, 60s max, 10 max attempts (line 32-43)

**File**: `src/market_maker/infra/connection_supervisor.rs`
- `ConnectionSupervisor` provides application-level supervision
- `SupervisorEvent` enum (line 50-61): `Healthy`, `MarketDataStale`, `UserEventStale`, `ReconnectRecommended`, `Critical`
- Market data stale threshold: 10s (line 24)
- Requires 2 consecutive stale readings before signaling reconnect (line 33)

### Data Staleness Monitor (Risk System)

**File**: `src/market_maker/risk/monitors/data_staleness.rs`
- `DataStalenessMonitor` (line 12) watches `RiskState::data_age`
- Kill switch threshold: `stale_data_threshold` (default 30s, production 10s)
- Reconnection grace: up to 5 attempts before triggering kill switch (line 19)
- If actively reconnecting, pulls quotes but doesn't kill (line 67-80)
- If connection permanently failed, triggers kill switch immediately (line 55-61)

### Data Quality Gate

**File**: `src/market_maker/infra/data_quality.rs`
- `should_gate_quotes()` (line 287) -- immediate quote blocking at 15s threshold
- This fires BEFORE the 30s kill switch threshold
- Three reasons: `NoDataReceived`, `StaleData`, `CrossedBook` (line 72-79)

### Diagnosis Flow

1. **Check connection state**: Is the WebSocket connected?
2. **Check reconnection count**: How many attempts have been made?
3. **Check data quality gate**: Is `should_gate_quotes()` firing?
4. **Check which feed is stale**: HL market data? HL state? Binance?
5. **Check network**: Is the machine online? DNS resolving? Firewall blocking?
6. **Check exchange status**: Is Hyperliquid having an outage?

### Recovery

- If single feed: the connection supervisor should auto-reconnect (up to 10 attempts)
- If all feeds: likely network or machine issue, not exchange
- If permanent failure after 10 attempts: check exchange status, restart MM
- After recovery: verify positions match between local state and exchange

---

## G. Post-Incident Checklist

After every incident, complete these steps. Skipping them leads to repeat incidents.

### Immediate (During Incident)

- [ ] Save full state dump: kill switch state, position, PnL, all multipliers
- [ ] Screenshot or log the relevant log lines
- [ ] Note the exact time (UTC) the incident started and ended
- [ ] Record which monitors/gates were active

### Within 1 Hour

- [ ] Record incident in Serena memory: `.serena/memories/incident_YYYY-MM-DD_<symptom>.md`
- [ ] Update MEMORY.md with incident details and root cause
- [ ] If new bug found: file it in Known Issues section of MEMORY.md
- [ ] If threshold needs updating: note the old and new values with reasoning

### Within 24 Hours

- [ ] Root cause analysis: trace the exact code path that led to the incident
- [ ] Check if the same pattern exists elsewhere in the codebase
- [ ] Write test case that reproduces the root cause
- [ ] Consider adding to pre-flight checks if it could be caught earlier
- [ ] Update monitoring thresholds if the current ones are too sensitive or too loose

### Pre-Flight Checks (Before Next Trading Session)

- [ ] All feeds connected and receiving data
- [ ] Kill switch is NOT triggered
- [ ] Position matches exchange state
- [ ] API headroom > 20%
- [ ] No circuit breakers active
- [ ] Checkpoint loaded successfully (or cleanly skipped if > 24h old)
- [ ] Spread multiplier < 2.0x at startup

---

## Reference: Risk System Architecture

```
                     +-----------------+
                     |  KillSwitch     |  <- Latching emergency stop
                     |  kill_switch.rs |     Once triggered, requires manual reset
                     +--------+--------+
                              |
                     +--------v--------+
                     | RiskAggregator  |  <- Takes MAX severity across all monitors
                     | aggregator.rs   |
                     +--------+--------+
                              |
          +-------------------+-------------------+
          |         |         |         |         |
     +----v--+ +---v---+ +---v---+ +---v---+ +---v---+
     | Loss  | | Draw  | | Pos   | | Data  | | Casc  | ... (8 monitors)
     +-------+ +-------+ +-------+ +-------+ +-------+
                              |
                     +--------v--------+
                     | DataQualityGate |  <- Pre-quote data validation (15s)
                     | data_quality.rs |
                     +--------+--------+
                              |
                     +--------v--------+
                     | SpreadMultiplier|  <- 7 multiplicative factors
                     | quote_engine.rs |     Lines 992-1107
                     +--------+--------+
                              |
                     +--------v--------+
                     |   QuoteGate     |  <- Directional decision
                     |  quote_gate.rs  |     Quote/NoQuote/OneSide
                     +-----------------+
```

---

## Reference: Key File Map

See [references/incident-history.md](references/incident-history.md#key-file-map) for the complete file:line reference table covering all risk monitors, data quality gates, spread chain components, and quote gate infrastructure.
