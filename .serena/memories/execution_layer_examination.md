# Execution Layer & Exchange Integration Examination

**Date**: 2026-02-16 | **Scope**: Live trading vs paper simulation gap analysis

---

## 1. ORDER EXECUTION LAYER

### A. Order Placement Pipeline
**File**: `src/market_maker/infra/executor.rs`

**Core Types**:
- `OrderResult`: Resting size, OID, immediate fill flag, CLOID, error message
- `OrderSpec`: Price, size, is_buy, cloid (optional), post_only flag
- `ModifyResult`: New OID, resting size, success flag, error message
- `CancelResult`: Enum with 4 outcomes (Cancelled, AlreadyCancelled, AlreadyFilled, Failed)

**Key Design Decision**: `CancelResult::AlreadyFilled` is SEPARATE from `AlreadyCancelled`
- `safe_to_remove_tracking()` returns FALSE for AlreadyFilled
- **Reason**: Fill notification will arrive separately; removing order from tracking loses the fill

**Critical Logic**:
```
OrderResult.error field is populated when order fails
→ Caller can query error message to decide: retry? reconcile? widen spreads?
```

### B. WebSocket Order State Manager
**File**: `src/market_maker/tracking/ws_order_state/manager.rs`

**Dual-Path Architecture**:
1. **WS POST requests** (lower latency): Orders submitted via WebSocket post, await response
2. **State tracking** via orderUpdates + userFills subscriptions (real-time)

**Request Correlation Strategy**:
- Request ID → tracks inflight requests
- CLOID (UUID) → cross-event matching (order request → orderUpdate → userFill)
- CLOID→OID mapping → handles state transitions

**Inflight Request Management**:
- `max_inflight_requests` config (prevents unbounded queueing)
- Timeout handling with REST fallback
- Fill deduplication by trade ID (`processed_tids` HashSet)

**Known Issue** (from memory): Cancel-fill race condition
- Orders cancelled but filled first → position spikes
- **Root cause**: Exchange may execute fill BEFORE processing cancel
- **Mitigation**: Reconciliation + watchdog monitors for unexpected position changes

---

## 2. FILL PROCESSING PIPELINE

### A. Fill Consumption & Deduplication
**File**: `src/market_maker/fills/dedup.rs`

**FillDeduplicator**:
- Fixed-size HashSet with FIFO eviction (default: 10K fills)
- Trade ID (tid) is the unique key
- Prevents double-processing when fills arrive via multiple channels

**Single Source of Truth**:
```
WebSocket: UserFills
    ↓
FillDeduplicator.check_and_mark(tid)  ← Returns: is_new? (bool)
    ↓
FillProcessor
    ↓
Updates: Position, Orders, AS Classifier, Queue Tracker, PnL, Estimators
```

**CRITICAL**: No multiple dedup mechanisms scattered across modules
- Previously: PositionTracker, AdverseSelectionEstimator, DepthDecayAS all had their own HashSets
- Now: Single authoritative dedup pass

### B. Fill Event Structure
**File**: `src/market_maker/fills/mod.rs`

```rust
pub struct FillEvent {
    pub tid: u64,                      // Trade ID (dedup key)
    pub oid: u64,                      // Order ID
    pub cloid: Option<String>,         // Client Order ID (UUID for deterministic tracking)
    pub size: f64,                     // Always positive
    pub price: f64,
    pub is_buy: bool,
    pub mid_at_fill: f64,             // Price at fill time (spread calculation)
    pub placement_price: Option<f64>,  // Where we placed the order (depth calc)
    pub asset: String,
    pub mid_at_placement: f64,         // CRITICAL: For adverse selection measurement
    pub quoted_half_spread_bps: f64,   // |fill_price - mid_at_placement| / mid_at_placement * 10000
}
```

**Key Field: `mid_at_placement`**
- Came from TrackedOrder when order was placed
- Used to calculate correct adverse selection (AS)
- **Problem in earlier version**: AS tautology — mid_at_fill was used instead
  - Result: AS always ~0 because we compare fill_price vs mid_at_fill (contemporaneous)
  - **Fix**: `mid_at_placement` set at order placement, compared at fill

### C. Fill Signal Capture
**File**: `src/market_maker/fills/processor.rs` (lines 45-231)

**FillSignalSnapshot**: Captures 47+ fields at moment of fill:
- Pre-fill toxicity (bid/ask separately)
- Orderbook imbalance, trade flow imbalance
- HMM regime, changepoint probability, toxic regime flag
- Continuation probability, momentum signals
- Markout outcomes (500ms, 2s, 10s — filled asynchronously)

**Markout Tracking**:
- `PendingMarkout` struct waits for price updates
- Updated via `update_markouts(current_mid)` called on AllMids events
- Stale after 30s with incomplete data

**Export**: JSON JSONL to `data/analytics/` for Python post-analysis

**Design Philosophy**: Don't measure signal efficacy retroactively
- Record signal state AT FILL TIME
- Later: compare predicted signals vs actual adverse selection

### D. Fill State Bundle
**File**: `src/market_maker/fills/processor.rs` (implied in design)

```rust
pub struct FillState<'a> {
    // Mutable references to all modules that need updating
    position: &'a mut PositionTracker,
    orders: &'a mut OrderManager,
    as_classifier: &'a mut EnhancedASClassifier,
    queue_tracker: &'a mut QueuePositionTracker,
    pnl: &'a mut PnLTracker,
    estimator: &'a mut ParameterEstimator,
    metrics: &'a mut PrometheusMetrics,
}
```

**Benefits**:
- Compiler enforces all modules updated (missing one = compile error)
- Clear data flow: fill → all modules in one transaction
- Testable: can inject mock state

---

## 3. EXCHANGE STATE RECONCILIATION

### A. Position Reconciliation
**File**: `src/market_maker/infra/reconciliation.rs`

**Triggers** (4 categories):
1. **Background Timer**: Every 10s (was 60s, accelerated due to drift issues)
2. **Order Rejection**: Position-related error → immediate sync
3. **Unmatched Fill**: Fill for oid not in OrderManager → drift detected
4. **Large Position Change**: >5% of max_position → anomaly flag

**Design Philosophy**: Event-driven + low-overhead background
- Expensive operation (REST call to exchange for account state)
- Minimized with minimum 2s interval between event-triggered syncs

**Drift Detection** (potential causes):
- Fill arrived before order was tracked (out-of-order WS messages)
- Cancel-fill race: order filled after cancel processed
- Connection dropout: fills missed during disconnect
- Stale local cache: exchange partial fill not reflected locally

### B. Order Manager Reconciliation
**File**: `src/market_maker/tracking/order_manager/reconcile.rs` (lines 1-200+)

**Ladder Reconciliation** (quote cycle):
- Compare current orders (from WS orderUpdates) vs target ladder (from quote engine)
- Generate minimal cancel/place/modify actions

**Smart Thresholds**:
```rust
skip_price_tolerance_bps: 20,    // UNCHANGED if price within 20 bps
skip_size_tolerance_pct: 0.10,   // UNCHANGED if size within 10%
max_modify_price_bps: 50,        // Modify (not cancel+place) if price Δ ≤ 50 bps
max_modify_size_pct: 0.50,       // Modify if size Δ ≤ 50%
```

**Queue-Aware Reconciliation**:
- `use_queue_aware: true` by default
- Uses `QueueValueComparator` to evaluate: is old order's queue position worth preserving?
- If P(fill) * expected_value > replacement order edge → KEEP old order (no modify)

**Design Principle**: Queue position > price optimality
- Order with 1s queue history at slightly suboptimal price beats fresh order at mid
- Adaptive thresholds scale with vol and kappa

**HJB Queue Value Integration**:
- Optional: uses continuation value formula to value queue positions
- Parameters: α (decay), β (linear cost), modify_cost_bps threshold

---

## 4. WEBSOCKET INFRASTRUCTURE

### A. Connection Supervision
**File**: `src/market_maker/infra/connection_supervisor.rs`

**Two-Layer Health Monitoring**:

1. **Low-level** (WsManager): Ping/pong at 30s intervals
   - Timeout after 90s pong-less
   - Exponential backoff on reconnect (1s → 60s max)

2. **High-level** (ConnectionSupervisor): Market data staleness
   - Threshold: 10s without market data → Stale
   - Threshold: 60s without user events → Stale
   - Requires 2 consecutive stale readings before signaling reconnect

**Auto-Clear Reconnection State**:
```rust
// CRITICAL FIX (line 164-169):
if was_reconnecting && data_arrived {
    self.health_monitor.reconnection_success()
    // Prevents attempt counter accumulation
}
```

**Why This Matters**:
- Attempt counter increments on each reconnection
- Eventually hit `max_consecutive_failures` limit
- If stale data detection false-positive → kill switch triggers
- **Fix**: Clear counter on first successful data post-reconnect

### B. Rate Limiting (Proactive)
**File**: `src/market_maker/infra/rate_limit/proactive.rs`

**Hyperliquid Rate Limits** (per docs):
- IP: 1200 weight/minute
- Address: 1 request per 1 USDC traded + 10K buffer
- Batched: 1 IP weight but n address requests
- On 429: 1 request per 10 seconds ONLY

**CumulativeQuotaState** (Death Spiral Prevention):
```rust
pub struct CumulativeQuotaState {
    consecutive_exhaustions: u32,
    backoff_until: Option<Instant>,
}

// Exponential backoff: 30s, 60s, 120s, 240s... max 600s (10 min)
// With 50% jitter to prevent herd behavior
```

**Triggers**:
- Track IP weight usage (rolling 60s window)
- Warning threshold: 80% of limit
- Action: 429 backoff with exponential multiplier (1.5x)

**Quota Budget** (address level):
- `address_initial_buffer: 10K`
- `requests_per_usd_traded: 1.0`
- As volume traded → more quota available
- **Problem** (from memory): hyna DEX quota tight at 7% API usage → one-sided quoting only

### C. WebSocket Message Flow
**File**: `src/ws/ws_manager.rs`

**Subscription Types**:
- AllMids: Mid prices (with optional HIP-3 DEX)
- L2Book: Order book updates
- Trades: Trade feed
- OrderUpdates: Own order state changes
- UserFills: Fill notifications
- UserFundings: Funding rate updates
- Notification: General alerts

**Message Correlation**:
```
WsPostRequest (order placement via WS)
    ↓ (correlate by request_id)
WsPostResponse (immediate ack)
    ↓ (use CLOID to match)
OrderUpdates (OID assignment)
    ↓ (match tid to oid)
UserFills (fill notification)
```

**Critical Gap**: Order confirmation timing
- Rest: order immediately resting after POST response
- Reality: OID may arrive slightly later in orderUpdates
- If fill arrives before orderUpdates → unmatched fill → reconciliation triggered

---

## 5. DATA QUALITY GATES

### A. Data Quality Monitor
**File**: `src/market_maker/infra/data_quality.rs`

**Checks**:
- Sequence gaps: Track per-asset sequence numbers
- Timestamp regression: Reject messages going backwards
- Stale data: Age > 30s → gate quotes
- Crossed books: Best bid >= best ask → gate quotes
- Invalid prices: Deviation > 20% from reference → reject
- Invalid sizes: Zero, negative, or non-finite → reject

**QuoteGateReason**:
```rust
pub enum QuoteGateReason {
    NoDataReceived,
    StaleData { age_ms: u64, threshold_ms: u64 },
    CrossedBook { best_bid: f64, best_ask: f64 },
}
```

**Design Philosophy**: Fail-safe quoting
- When unsure, don't quote (missing a trade < getting run over)
- Stale data → no quotes until refresh

### B. Ancillary Staleness Checks
**Multiple Layers**:

1. **Connection supervisor**: Market data staleness (10s threshold)
2. **Data quality monitor**: Per-asset staleness (30s threshold)
3. **Quote engine**: Checks both before generating ladder
4. **Risk monitor**: Kill switch on stale data + other conditions

**Redundancy Rationale**: Belt-and-suspenders
- Connection issue might not be obvious to data quality monitor
- Data quality might miss slow feeds (last tick 45s old, but trickling)

---

## 6. PAPER SIMULATION GAPS (vs Live)

### A. Cancel-Fill Race Condition
**Simulated**: Perfect causality
- Cancel issued → immediately removed from book
- Fill only arrives for resting orders

**Live**:
- Exchange asynchronous processing
- Cancel and fill both in-flight simultaneously
- Fill can execute before cancel processed
- Result: Position spike (both sides filled)

**Mitigation Needed**:
- Reconciliation on unmatched fills
- Watchdog monitor for spike detection
- Quick cancellation of opposing orders

### B. Connection Dropouts & Reconnection
**Simulated**: Continuous perfect data
- No message loss
- No stale data periods
- Immediate market data availability

**Live**:
- 500ms-10s+ dropouts (typical)
- Fills missed during disconnect
- Market data stale for reconnection duration
- Reconnection may not fully clear stale state

**Gaps in Paper Trader**:
- `FillSimulator` assumes deterministic execution
- No Binance-Hyperliquid lead-lag (Binance leads by 50-500ms)
- No quota pressure (can requote freely)
- No microstructure (no crushed spread when overwhelmed)

### C. Microstructure & Latency
**Simulated**: Negligible latency, infinite liquidity

**Live**:
- Order placement latency: 50-200ms (WS POST vs REST)
- Market impact: Large orders tighten spreads or incur fees
- Queue position decay: Positions change between placement and fill
- Adverse selection timing: Movement happens between mid_at_placement and fill

### D. Order Manager State Divergence
**Simulated**: OrderManager perfectly mirrors exchange state

**Live**:
- Out-of-order messages: orderUpdates may arrive before userFills or vice versa
- Partial fills: size changes mid-cycle, multiple updates per order
- Order rejects: "Invalid leverage" not detected until placement attempt
- Status anomalies: Order in "pending" state longer than expected

### E. Risk Calculation Divergence
**Simulated**: Risk thresholds are constants

**Live**:
- Capital util changes (resting orders consume margin)
- Leverage varies per-asset
- Liquidation price moves dynamically
- Risk model may be stale (old fills in calibration buffer)

---

## 7. CRITICAL FAILURE MODES NOT SIMULATED

### A. WS Disconnect During Order Cycle
```
Cycle N: Generate ladder → Place 10 orders via WS POST
↓ (5 orders placed successfully)
[DISCONNECT]
↓ (remaining 5 orders lost, no OID ever assigned)
[RECONNECT + DATA RESUME]
↓ (position reflects 5 fills, but we think 0 were placed — reconciliation)
```

**Paper Trader Gap**: Assumes 100% delivery

### B. Quota Exhaustion Death Spiral
```
Normal: 1200 weight/min available, 1100 being used (alright)
↓
OI drop → cascade trading → 50 orders/min instead of 10
↓
Quota exhaustion: 1200 weight used, API returns 429
↓
Exponential backoff: 10s, 20s, 40s... eventually 10min backoff
↓
System cannot quote, position drifts, kill switch triggers
```

**Paper Trader Gap**: No quota simulation, unlimited order placement

### C. Stale Data Leading to Mispricing
```
Mid price: 50000 BTC
↓
[10s data gap during volatile period]
↓
Resume with: mid 50000 (stale)
↓
Actually: mid 50300 (market rallied while stale)
↓
Quote: bid at 49950 (appears -50 bps, actually -350 bps from true mid)
↓
Adverse selection + margin consumed
```

**Paper Trader Gap**: Uses real Binance data (no gaps), but replay assumes continuous data

### D. Order State Divergence Over Time
```
Local state: 3 resting orders
Exchange state: 1 order (2 got crushed in rapid price action)
↓
Next reconciliation: "What? 2 orders gone? Maybe cancelled by risk?"
↓
Quote engine: Thinks spread is narrower than it is
↓
Reprice too tight, get run over
```

**Paper Trader Gap**: OrderManager always in sync with simulated orders

### E. Fill Timestamp Mismatch
```
Order placed: t=1000ms (mid = 50000)
↓
Market rallies: t=1050ms (mid = 50100)
↓
Fill arrives: t=1100ms (fill price = 50080, but mid_at_fill = 50100)
↓
Adverse selection = 50100 - 50080 = 20 bps (good)
↓
But: mid_at_placement was NEVER SET
↓
AS tautology: compare mid_at_fill vs mid_at_placement (both 50100) → 0 bps
```

**Paper Trader Gap**: Timestamp alignment is perfect, mid_at_placement always available

---

## 8. ARCHITECTURAL INSIGHTS & RECOMMENDATIONS

### Key Strength: Layered Safety
1. FillDeduplicator (prevent double-process)
2. FillState bundle (compiler-enforced updates)
3. OrderManager reconciliation (queue preservation + threshold awareness)
4. PositionReconciler (drift detection + multi-trigger)
5. DataQualityMonitor (stale data rejection)
6. ConnectionSupervisor (auto-clear reconnection state)

### Key Weakness: Reconciliation Latency
- Background interval: 10s (vs optimal ~500ms)
- Reason: REST call expensive, but can't afford 10s drift in cascade
- **Opportunity**: Event-triggered sync on unmatched fill is great, but sync interval min=2s still allows drift

### Key Risk: Quota Death Spiral
- Currently: exponential backoff with jitter (good)
- Missing: circuit breaker to STOP quoting before exhaustion
- Missing: Quota reservation system (reserve quota for cancel+recover)
- From memory: hyna DEX quota at 7% API usage → forced one-sided quoting

### Data Flow Integrity
- **Best practice**: CLOID for deterministic matching (good)
- **Missing**: Timeout-based order deconfirmation (if no orderUpdate in 5s, assume stale)
- **Missing**: Duplicate order detection (same CLOID + price within 1 min → likely duplicate)

### Fill Signal Capture
- **Excellent**: FillSignalSnapshot with 47+ fields + async markout tracking
- **Missing**: Snapshot of exchange-level order state at fill (queue depth, OI, funding rate at exact fill time)
- **Missing**: Attribution of fill to specific market-making action (was it from base ladder? spike defense? recovery?)

---

## 9. RECOMMENDATIONS FOR LIVE TRADING

### P0: Quota Management
- [ ] Implement quota reservation (mandatory 5% headroom for cancels)
- [ ] Add circuit breaker: stop new quotes when headroom < 10%
- [ ] Track per-asset quota usage separately (in case some assets are quota hogs)

### P0: Reconciliation Speed
- [ ] Reduce background interval from 10s → 3s (cost is one REST call every 3s)
- [ ] Make event-triggered sync synchronous (not async) — priority over quote delay

### P1: Timeout Detection
- [ ] OrderManager: if order placed >5s ago but no orderUpdate → mark stale, don't quote until resolved
- [ ] FillConsumer: if fill correlated but no reconciliation within 1s → escalate to risk monitor

### P1: Microstructure Simulation
- [ ] Paper trader: add Binance lead-lag (delay HL mid by 50-500ms)
- [ ] Paper trader: add realistic quote rejection rate (2-5% at peak periods)
- [ ] Paper trader: add order confirmation delay (100-300ms for OID assignment)

### P2: Kill Switch Hardening
- [ ] Separate "stale data" from "position divergence" triggers
- [ ] Use `reconnection_success()` to auto-clear attempt counter (already done ✓)
- [ ] Add "quota exhaustion" as a separate kill switch trigger (not lumped with risk)

### P2: Fill Attribution
- [ ] Tag each fill with generation source (base_ladder, spike_defense, recovery)
- [ ] Export this to FillSignalSnapshot
- [ ] Enables post-analysis: did spike defense fills have higher AS? Good or bad signal?

---

## 10. SUMMARY TABLE: Live vs Paper Gaps

| Component | Paper | Live | Gap | Risk Level |
|-----------|-------|------|-----|-----------|
| Fill Dedup | HashSet (perfect) | HashSet + WS | Perfect | Low |
| Order State | Deterministic | 3+ msg streams | Message order | **Medium** |
| Cancel-Fill | Never race | Can race | Race possible | **High** |
| Position | Always correct | Event-driven sync | 10s drift | **High** |
| Quota | Unlimited | 1200 weight/min | Exhaustion spiral | **Critical** |
| Data Quality | Perfect | Stale/crossed/gaps | Rejection gates | **Medium** |
| Latency | 1ms | 50-200ms | Price slip | **Medium** |
| Connection | Always up | Dropout 500ms-10s+ | Stale state | **Medium** |
| Reconciliation | N/A | 10s background + event | Drift window | **Medium** |

---

## Files Examined
- `src/market_maker/execution/`: order_lifecycle.rs, fill_tracker.rs, state_machine.rs, mode_handlers.rs
- `src/market_maker/fills/`: dedup.rs, processor.rs, consumer.rs, mod.rs
- `src/market_maker/tracking/`: order_manager/reconcile.rs, ws_order_state/manager.rs
- `src/market_maker/infra/`: executor.rs, connection_supervisor.rs, data_quality.rs, reconciliation.rs, rate_limit/proactive.rs
- `src/exchange/`: exchange_client.rs
- `src/ws/`: ws_manager.rs, message_types.rs
- `src/market_maker/messages/`: processors.rs

**Total Lines Analyzed**: ~2,500+ LOC
