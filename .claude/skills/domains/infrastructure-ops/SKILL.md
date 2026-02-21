---
name: infrastructure-ops
description: WebSocket management, event loop, rate limiting, reconnection, recovery, metrics, and order execution infrastructure. Use when working on orchestrator/, infra/, messages/, core/, fills/, or execution/ modules, debugging connectivity or order placement, adding message handlers, or investigating stale data and latency issues.
requires:
  - quote-engine
user-invocable: false
---

# Infrastructure Operations Skill

## Purpose

Understand and modify the operational infrastructure that connects market data to order execution. Covers the event loop, WebSocket management, rate limiting, reconnection, recovery, metrics, and order execution.

## When to Use

- Working on `orchestrator/`, `infra/`, `messages/`, `core/`, `fills/`, or `execution/` modules
- Debugging connectivity, order placement, or feed issues
- Adding new message handlers or data sources
- Understanding the event loop and quote cycle
- Investigating "orders not sending" or "stale data"

## Module Map

All paths relative to `src/market_maker/`:

```
orchestrator/
  event_loop.rs         # Main event loop — sync, startup, recovery, dispatch
  handlers.rs           # WS message dispatching (L2 book, trades, fills, user events)
  quote_engine.rs       # Quote generation — warmup, estimator validation, strategy
  reconcile.rs          # ReconciledOrder — order diff/reconciliation
  order_ops.rs          # Place, cancel, modify order abstractions
  recovery.rs           # Safety sync, state refresh, stuck order recovery
  event_accumulator.rs  # EventAccumulator — event-driven quote triggering

infra/
  connection_supervisor.rs  # Proactive WS health monitoring
  reconnection.rs           # Feed reconnection state machine
  data_quality.rs           # Market data anomaly detection & validation
  margin.rs                 # MarginTracker — margin-aware position sizing
  ws_executor.rs            # WS POST executor with REST fallback
  arena.rs                  # Arena allocator for quote cycle latency
  capacity.rs               # Pre-allocation sizes
  orphan_tracker.rs         # Prevents false orphan detection
  logging.rs                # Structured multi-stream logging
  recovery.rs               # Stuck position recovery
  reconciliation.rs         # Event-driven position sync
  sync_health.rs            # Feed sync health tracking
  dashboard_ws.rs           # Real-time dashboard WebSocket
  rate_limit/
    mod.rs                  # Rate limit orchestrator
    proactive.rs            # Predict rate limits before rejection
    rejection.rs            # React to rate limit rejections
    error_type.rs           # RateLimitError types
  metrics/
    mod.rs                  # PrometheusMetrics aggregator
    atomic.rs               # Lock-free metric updates
    dashboard.rs            # Signal snapshots, diagnostics
    fields.rs               # All tracked metric fields
    output.rs               # Prometheus text format
    summary.rs              # Aggregated statistics
    updates.rs              # Event-driven metric recording

messages/
  all_mids.rs           # Mid-price message processing
  l2_book.rs            # L2 order book message processing
  trades.rs             # Trade message processing
  processors.rs         # Message processor traits
  context.rs            # Message context (timestamps, sequence)

core/
  components.rs         # Core component initialization
  state.rs              # CoreState — shared mutable state

fills/
  consumer.rs           # Fill event consumption
  dedup.rs              # Fill deduplication
  processor.rs          # Fill processing pipeline

execution/
  fill_tracker.rs       # Fill tracking and reporting
  order_lifecycle.rs    # Order state machine (pending -> open -> filled/cancelled)
```

---

## Event Loop Architecture

### Startup Sequence

```
1. sync_open_orders()    — fetch all orders from exchange
2. cancel_all()          — clean slate
3. sync_position()       — reconcile local vs exchange position
4. sync_limits()         — fetch exchange-enforced limits
5. enter main loop
```

### Main Loop

```
loop {
    1. Receive WS message (L2 book, trades, fills, user events)
    2. Dispatch to handler (handlers.rs)
    3. Accumulate events (event_accumulator.rs)
    4. If enough events accumulated -> update quotes (quote_engine.rs)
    5. Periodic: recovery check, safety audit, metrics flush
}
```

### Event Accumulator

Prevents excessive quote updates. Triggers when:
- L2 book changed significantly
- Trade occurred
- Fill received
- Timer elapsed (max staleness)

---

## Quote Generation Pipeline

`quote_engine.rs` drives each quote cycle:

```
1. Warmup check     — enough data for reliable estimates?
2. Estimator check  — kappa, sigma, regime all valid?
3. Risk check       — risk system allows quoting?
4. Strategy compute — GLFT formula -> optimal bid/ask
5. Ladder generate  — multi-level quote ladder
6. Reconcile        — diff against current orders
7. Execute          — place/cancel/modify minimum set
```

### Order Reconciliation (`reconcile.rs`)

Minimizes exchange API calls by diffing desired vs current:

- **Keep**: same price and size -> no action
- **Modify**: same price, different size -> modify
- **Cancel+Place**: different price -> cancel old, place new
- **Cancel**: existing not in new set -> cancel
- **Place**: new not in existing set -> place

---

## WebSocket Execution

`ws_executor.rs`:
- Primary: WS POST for lowest latency
- Fallback: REST API when WS degraded
- Automatic retry with exponential backoff

### Order Lifecycle (`execution/order_lifecycle.rs`)

```
Pending -> Open -> Filled | Cancelled | Expired
                -> PartialFill -> Filled | Cancelled
```

---

## Rate Limiting

Three-tier defense:

### 1. Proactive (`rate_limit/proactive.rs`)
- Sliding window rate tracking
- Predicts when limit will be hit
- Throttles before rejection

### 2. Reactive (`rate_limit/rejection.rs`)
- Handles `RateLimitError` from exchange
- Exponential backoff on rejection

### 3. Budget-based
- Token bucket for statistical impulse control
- Distributes order budget across cycles

---

## Reconnection & Recovery

### Feed Reconnection (`infra/reconnection.rs`)

```
Connected -> Disconnected -> Reconnecting (with backoff) -> Connected
```

- Exponential backoff with jitter
- Separate tracking per feed (Hyperliquid L2, trades, Binance)
- Connection supervisor monitors all feeds

### Recovery (`orchestrator/recovery.rs`)

After reconnection:
1. Safety sync: verify position matches exchange
2. State refresh: re-fetch margins, limits, open orders
3. Stuck orders: detect and cancel non-responsive orders
4. Orphan cleanup: reconcile local tracking vs exchange

---

## Data Quality

`data_quality.rs` validates incoming market data:
- Timestamp freshness
- Price sanity (no negative, no >10% jumps without explanation)
- Book consistency (bid < ask, levels sorted)
- Sequence number tracking (detect gaps)

Validation failures trigger data staleness warnings in risk system.

---

## Metrics

### Architecture

```
Event -> AtomicMetric (lock-free) -> MetricsSummary -> PrometheusOutput
```

### Key Metrics

| Category | Metrics |
|----------|---------|
| Fills | fill_count, fill_rate, fill_latency_us, adverse_pct |
| Spreads | quoted_spread_bps, effective_spread_bps, spread_factor |
| Orders | orders_placed, orders_cancelled, orders_modified, rejection_rate |
| Latency | quote_cycle_us, ws_roundtrip_us, order_ack_us |
| Risk | risk_severity, spread_multiplier, position_utilization |
| Data | book_age_ms, trade_age_ms, binance_age_ms |

---

## Arena Allocator

`arena.rs` — per-quote-cycle allocation:
- Pre-allocates memory for hot path
- Avoids heap allocation during quote cycle
- Reset between cycles (near-zero cost)
- Capacity constants in `capacity.rs`

---

## Common Debugging

### "Orders not sending"
1. Rate limiter — throttled?
2. WS executor — connection alive?
3. Risk system — quotes gated?
4. Order lifecycle — stuck in Pending?
5. Margin — enough for new orders?

### "Stale quotes / not updating"
1. Event accumulator — triggering updates?
2. Data quality — feed data fresh?
3. Connection supervisor — all feeds connected?
4. Warmup — estimator has enough data?

### "Position mismatch"
1. Safety auditor — `SafetyAuditor::audit()`
2. Orphan orders — fills from unknown orders?
3. Fill dedup — double-counted fills?
4. Reconciliation — local vs exchange position

### "High latency"
1. `quote_cycle_us` metric
2. Arena allocator — capacity sufficient?
3. Blocking in event loop — no sync I/O!
4. `ws_roundtrip_us` metric

---

## Key Rules

1. **Never block the event loop** — all I/O must be async
2. **Arena allocator on hot path** — no heap allocation in quote cycle
3. **Rate limiting is defense** — never rely on exchange not rejecting
4. **Reconciliation over creation** — minimize order operations
5. **Data quality gates quotes** — stale data = no quotes
6. **Recovery is automatic** — system must self-heal from disconnections

---

## Adding a New Message Handler

1. Define message type in `messages/`
2. Implement `MessageProcessor` trait
3. Add dispatch case in `handlers.rs`
4. Register event type in `event_accumulator.rs`
5. Update metrics in `metrics/updates.rs`
6. Add data quality checks in `data_quality.rs`

---

## Supporting Files

| File | Description |
|------|-------------|
| [references/debugging-playbook.md](references/debugging-playbook.md) | Expanded debugging playbooks for "orders not sending", "stale quotes", "position mismatch", and "high latency" -- full diagnostic chains with file references, specific checks, and resolution steps |
