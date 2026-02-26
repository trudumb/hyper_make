# Infrastructure Debugging Playbook

Expanded diagnostic procedures for the four most common operational failures.
Each scenario walks through a full diagnostic chain with specific file references,
log patterns to look for, and resolution steps.

---

## 1. "Orders Not Sending"

Full diagnostic chain from rate limiter to margin.

### Step 1: Rate Limiter State

**Files**: `infra/rate_limit/proactive.rs`, `infra/rate_limit/rejection.rs`

- Check if proactive rate limiter is throttling:
  - `ProactiveRateLimiter` uses a sliding window to predict when limits will be hit
  - If throttled, you will see the proactive limiter suppressing order calls
- Check if reactive backoff is active:
  - `RejectionRateLimiter` enters exponential backoff after exchange rejection
  - Look for `RateLimitError` events in logs
  - Backoff is per-side (bid/ask tracked separately)
- Check the rate limit monitor: `risk/monitors/rate_limit.rs`
  - `RateLimitMonitor` may be returning `RiskAction::ReduceOnly` or `PullQuotes`
- **Resolution**: If backoff is active, wait for it to decay. If proactive limiter is
  incorrectly predicting limits, check `infra/execution_budget.rs` token bucket config.
  If API quota is permanently low (e.g., 7% on hyna DEX), see `orchestrator/reconcile.rs`
  quota tiers: <5% conservation, <20% minimal, >=20% full.

### Step 2: WS Executor Connection

**Files**: `infra/ws_executor.rs`, `infra/connection_supervisor.rs`, `infra/reconnection.rs`

- Check `WsPostConfig` state:
  - `enabled: true` and `timeout: 500ms` are defaults (`ws_executor.rs`)
  - If WS POST fails, REST fallback should activate automatically
- Check `ConnectionSupervisor` health:
  - `SupervisorConfig::market_data_stale_threshold` = 10s default
  - `SupervisorEvent::Critical` means connection has been stale too long
  - `stale_count_threshold` requires 2+ consecutive stale readings
- Check `ConnectionHealthMonitor` state machine (`reconnection.rs`):
  - States: `Connected -> Disconnected -> Reconnecting -> Connected`
  - Exponential backoff with jitter on reconnect attempts
  - Separate tracking per feed (Hyperliquid L2, trades, Binance)
- **Resolution**: If WS is down, verify `fallback_to_rest: true`. If supervisor shows
  `ReconnectRecommended`, check network. If reconnection loops, increase backoff limits.

### Step 3: Risk System Gating

**Files**: `risk/aggregator.rs`, `risk/monitor.rs`, `orchestrator/quote_engine.rs`

- Check `AggregatedRisk` output:
  - `pull_quotes == true` means all quotes suppressed
  - `max_severity >= High` means at least one monitor triggered
  - Look at `kill_reasons` for kill switch activation
- Check individual monitors in order of priority:
  - `PriceVelocityMonitor` (priority 10): velocity > 5%/s pulls quotes
  - `CascadeMonitor` (priority 30): OI drop > threshold
  - `DataStalenessMonitor`: feed age > 5s triggers cautious mode
  - `LossMonitor`: cumulative loss > daily limit
  - `PositionMonitor`: inventory > max_inventory
- Check quote engine gating in `quote_engine.rs`:
  - Warmup check: enough data for reliable estimates?
  - Estimator check: kappa, sigma, regime all valid?
  - Risk check: risk system allows quoting?
- **Resolution**: Identify which monitor triggered. If kill switch, it requires manual
  reset. If data staleness, check feeds. If cascade, wait for OI to stabilize.

### Step 4: Order Lifecycle Stuck

**Files**: `execution/order_lifecycle.rs`, `orchestrator/order_ops.rs`

- Check `OrderLifecycleTracker::active_orders()`:
  - State machine: `Placed -> PartiallyFilled -> Filled | Cancelled | Expired | Rejected`
  - Orders stuck in `Placed` for > 5s may indicate exchange not acknowledging
  - Orphan detection: orders in local tracking but not on exchange
- Check `order_ops.rs`:
  - Place, cancel, modify abstractions
  - CLOID (client order ID) correlation for cancel+place sequences
  - `smart_reconcile` falls back from MODIFY to CANCEL+PLACE
- Check `SafetyAuditor` (`safety/auditor.rs`):
  - `stale_pending_removed`: orders that were pending too long
  - `stuck_cancels`: cancel requests that did not execute
  - `orphan_orders`: exchange orders not in local tracking
  - `is_synced`: whether local and exchange state match
- **Resolution**: Run safety audit. If orders are stuck in Pending, check WS executor.
  If orphan orders exist, reconcile. If CLOID mismatch, check dedup.

### Step 5: Margin Insufficient

**Files**: `infra/margin.rs`, `risk/limits.rs`

- Check `MarginTracker` state:
  - Available margin tracking for sizing decisions
  - `MarginMode::Cross` vs `MarginMode::Isolated` (HIP-3 requires Isolated)
  - Leverage limits from `AssetLeverageConfig` (exchange-derived, not hardcoded)
  - Tiered leverage: position-based leverage reduction at higher notional
- Check `RiskLimits` / `RiskChecker`:
  - Soft/hard position and order limits
  - Notional position limits
- **Resolution**: If margin insufficient, reduce order sizes. If leverage mismatch,
  verify `MarginMode` matches asset type. Check `sync_limits()` in event loop startup.

---

## 2. "Stale Quotes / Not Updating"

Full diagnostic chain from event triggers to warmup state.

### Step 1: Event Accumulator Triggers

**Files**: `orchestrator/event_accumulator.rs`, `events/quote_trigger.rs`

- Check `EventAccumulator` state:
  - Accumulates events and decides when to trigger reconciliation
  - Features: event buffering, priority handling, scope merging, debouncing
  - `AffectedTracker` tracks which sides/orders are affected
  - Records: price moves (both sides), fills (specific side+OID), side events
  - Converts accumulated events to `ReconcileScope` for efficient reconciliation
- Check `QuoteUpdateTrigger`:
  - L2 book changed significantly?
  - Trade occurred?
  - Fill received?
  - Timer elapsed (max staleness fallback)?
- Check debouncing:
  - Minimum interval between reconciliations configured in `EventDrivenConfig`
  - Too-aggressive debouncing can cause stale quotes
- **Resolution**: If events are accumulating but not triggering, check debounce config.
  If no events arriving at all, check feed connections (Step 3).

### Step 2: Data Quality Validation

**Files**: `infra/data_quality.rs`

- Check `DataQualityConfig`:
  - `max_data_age_ms`: 30,000ms default (30s before stale)
  - `max_sequence_gap`: 10 (gap detection)
  - `max_price_deviation_pct`: 0.20 (20% max deviation from mid)
  - `check_crossed_books`: true
- Check anomaly detection:
  - `AnomalyType::SequenceGap`: message loss detected
  - `AnomalyType::TimestampRegression`: timestamps going backward
  - `AnomalyType::StaleData`: data too old
  - `AnomalyType::CrossedBook`: best bid >= best ask
  - `AnomalyType::InvalidPrice` / `InvalidSize`: bad values
- Check `QuoteGateReason`:
  - `NoDataReceived`: no data for this asset yet
  - `StaleData { age_ms, threshold_ms }`: data exceeds staleness threshold
  - `CrossedBook { best_bid, best_ask }`: book is inverted
  - Data staleness gates quotes via `should_gate_quotes()` (15s threshold)
- **Resolution**: If `StaleData`, check feed. If `CrossedBook`, likely a feed issue
  or exchange problem. If `SequenceGap`, messages were lost -- check reconnection.

### Step 3: Connection Supervisor

**Files**: `infra/connection_supervisor.rs`, `infra/reconnection.rs`, `infra/sync_health.rs`

- Check `ConnectionSupervisor` stats:
  - `time_since_market_data`: should be < 10s
  - `time_since_user_event`: should be < 60s
  - `consecutive_stale_count`: increments on each stale check
  - `current_event`: Healthy / MarketDataStale / UserEventStale / ReconnectRecommended / Critical
  - `is_reconnecting`: whether reconnection is in progress
- Check per-feed reconnection state (`reconnection.rs`):
  - Separate state machines for Hyperliquid L2, trades, Binance
  - State: Connected / Disconnected / Reconnecting
  - Exponential backoff with jitter
- Check `SyncHealth` tracking:
  - Feed sync health across all data sources
- **Resolution**: If supervisor shows `MarketDataStale`, check if specific feed is
  disconnected. If all feeds are connected but data is stale, check exchange status.
  If `ReconnectRecommended`, system should self-heal. If `Critical`, may need restart.

### Step 4: Warmup Estimator State

**Files**: `orchestrator/quote_engine.rs`, `stochastic/beliefs.rs`

- Check warmup requirements in quote engine:
  - Enough L2 book updates received?
  - Enough trade observations for kappa estimation?
  - Sigma estimator has sufficient data?
  - Regime estimator has classified the market?
- Check `MarketBeliefs` state:
  - `FillIntensityPosterior::is_warmed_up()`: needs >= 10 fills AND > 60s total time
  - NIG posterior for drift: needs sufficient observations to narrow uncertainty
  - Overall confidence must exceed minimum threshold
- **Resolution**: If in warmup, quote engine will not produce quotes until sufficient
  data. This is by design. Check warmup estimator state in dashboard.
  IMPORTANT: Do NOT use "WaitToLearn" logic that blocks quoting -- this creates a
  cold-start deadlock (learned the hard way, see MEMORY.md).

### Step 5: Quote Engine Warmup Check

**Files**: `orchestrator/quote_engine.rs`

- The quote engine performs this pipeline each cycle:
  1. Warmup check -- enough data for reliable estimates?
  2. Estimator check -- kappa, sigma, regime all valid?
  3. Risk check -- risk system allows quoting?
  4. Strategy compute -- GLFT formula -> optimal bid/ask
  5. Ladder generate -- multi-level quote ladder
  6. Reconcile -- diff against current orders
  7. Execute -- place/cancel/modify minimum set
- Any step can short-circuit the pipeline
- Key gotcha: raw `kappa` from L2 book can be ~6400-7700, but `kappa_effective`
  (blended) is ~3250. Any code comparing kappa to thresholds must use kappa_effective.
- **Resolution**: Add logging at each pipeline step to identify where short-circuit
  occurs. Check dashboard for estimator readiness indicators.

---

## 3. "Position Mismatch"

Full diagnostic chain from safety audit to fill tracking.

### Step 1: Safety Auditor Reconciliation

**Files**: `safety/auditor.rs`

- Run `SafetyAuditor` to get `AuditResult`:
  - `orders_cleaned`: expired fill window orders cleaned up
  - `stale_pending_removed`: pending orders that timed out
  - `stuck_cancels`: cancel requests that did not execute
  - `orphan_orders`: exchange orders not in local tracking
  - `stale_local_removed`: local orders not on exchange
  - `is_synced`: overall sync status
  - `reduce_only_active` / `reduce_only_reason`: if reduce-only triggered
- Check `has_issues()`: returns true if any sync problems found
- **Resolution**: If not synced, trigger full reconciliation via `recovery.rs`.

### Step 2: Orphan Fills

**Files**: `infra/orphan_tracker.rs`, `fills/consumer.rs`

- Check `OrphanTracker`:
  - Prevents false orphan detection during normal order lifecycle
  - An "orphan fill" is a fill event for an order not in local tracking
  - This happens during cancel-fill races: order cancelled but filled first
- Check `FillConsumer` (`fills/consumer.rs`):
  - Fill event consumption pipeline
  - Processes fill events from exchange WebSocket
  - Must handle fills for orders that were being cancelled
- **Resolution**: If orphan fills are frequent, check cancel timing. The cancel-fill
  race is a known issue (see MEMORY.md "Known Issues"). Handled gracefully but causes
  position whipsaw.

### Step 3: Fill Dedup Issues

**Files**: `fills/dedup.rs`

- Check `FillDeduplicator`:
  - Uses fixed-size HashSet (capacity 10,000) with FIFO eviction
  - `is_duplicate(tid)`: returns true if trade ID already processed
  - `mark_processed(tid)`: returns false if already processed (duplicate)
  - Single source of truth for dedup (replaces scattered HashSets)
- Potential issues:
  - If capacity is too small and eviction happens, very old fills could be re-processed
  - If dedup is not checked before processing, double-counting occurs
  - Past bug: duplicate `on_trade` handler in handlers.rs double-counted trades for kappa
- **Resolution**: Verify dedup capacity is sufficient for fill volume. Ensure all fill
  processing paths go through the deduplicator.

### Step 4: Local vs Exchange State

**Files**: `orchestrator/recovery.rs`, `infra/reconciliation.rs`

- Check position reconciliation flow:
  - `recovery.rs`: After reconnection, runs safety sync:
    1. Verify position matches exchange
    2. Re-fetch margins, limits, open orders
    3. Detect and cancel non-responsive orders
    4. Orphan cleanup
  - `reconciliation.rs`: Event-driven position sync
- Compare local position tracking vs exchange API response:
  - Fetch current position from exchange
  - Diff against local `PositionTracker` state
  - Log discrepancy with full details
- **Resolution**: If mismatch found, trust exchange as ground truth. Reset local
  tracking to match exchange. Investigate root cause (likely orphan fills or dedup issue).

### Step 5: Fill Tracker

**Files**: `execution/fill_tracker.rs`

- Check `FillTracker` metrics:
  - `fill_rate`: fills per quotes placed
  - `adverse_selection_rate`: % of fills where price moved against within 1s
  - `queue_position`: where in the order book queue our fills occurred
  - `latency_ms`: fill latency p50/p99
- Check for missing `update_price_after()` calls:
  - Adverse selection calculation requires post-fill price update
  - If this is not called, AS metrics will be wrong
- Check `FillRecord` completeness:
  - `timestamp`, `side`, `fill_price`, `fill_size`, `queue_position`, `latency_ms`
  - `price_after_1s`: updated asynchronously via `PendingFillOutcome` queue
- **Resolution**: Verify `record_fill()` and `update_price_after()` are both being
  called. Check fill tracker window size (default 1000 fills).

---

## 4. "High Latency"

Full diagnostic chain from quote cycle to model computation.

### Step 1: quote_cycle_us Metric

**Files**: `infra/metrics/fields.rs`, `infra/metrics/updates.rs`

- Check `quote_cycle_us` in metrics:
  - Target: < 500us for a single quote cycle
  - Warning: > 1ms
  - Critical: > 5ms (orders will be stale by the time they hit exchange)
- Check `ws_roundtrip_us`:
  - WS POST roundtrip time
  - Target: < 200ms
  - If REST fallback active, expect 500ms+
- Check `order_ack_us`:
  - Time from order sent to exchange acknowledgement
  - Includes network latency + exchange processing
- Metric pipeline: `Event -> AtomicMetric (lock-free) -> MetricsSummary -> PrometheusOutput`
- **Resolution**: If quote_cycle_us is high, drill into steps 2-5. If ws_roundtrip_us
  is high, check network or WS connection quality.

### Step 2: Arena Allocator Capacity

**Files**: `infra/arena.rs`, `infra/capacity.rs`

- Check `QuoteCycleArena`:
  - Pre-allocates 8KB (DEFAULT_ARENA_CAPACITY)
  - Typical usage: ~5KB per quote cycle
  - Budget: 10 LadderLevels (640B) + 10 Quotes (160B) + 20 OrderSpecs (4KB) + overhead (3KB)
  - Uses bumpalo for arena allocation: ~10ns per alloc vs ~100ns for heap
  - `reset()` between cycles is near-zero cost
- Check if arena is overflowing:
  - If more than 8KB needed, bumpalo allocates additional chunks (slower)
  - Signs: allocation spikes in metrics, unexpected latency jitter
- Check `capacity.rs` constants:
  - Pre-allocation sizes for various data structures
  - `DATA_QUALITY_CHANNELS` and similar constants
- **Resolution**: If arena overflows regularly, increase DEFAULT_ARENA_CAPACITY.
  If many ladder levels needed, check if quote count is reasonable.

### Step 3: Blocking in Event Loop

**Files**: `orchestrator/event_loop.rs`, `orchestrator/handlers.rs`

- The event loop MUST be non-blocking:
  - All I/O must be async
  - No synchronous network calls in the hot path
  - No locks held during computation
- Common blocking sources:
  - Synchronous logging calls (use structured async logging via `infra/logging.rs`)
  - Lock contention on shared state (`core/state.rs` uses `CoreState`)
  - DNS resolution or HTTP calls in handler paths
  - Checkpoint persistence during quote cycle
- Check handler dispatch in `handlers.rs`:
  - L2 book, trades, fills, user events all dispatched here
  - Each handler must complete quickly
  - Analytics calls use `let _ =` pattern to never crash the trader
- **Resolution**: Profile the event loop. Look for any `.await` on slow operations
  in the hot path. Move slow operations to background tasks.

### Step 4: WS Roundtrip

**Files**: `infra/ws_executor.rs`, `infra/connection_supervisor.rs`

- Check WS POST performance:
  - Default timeout: 500ms (was 1s, reduced for latency)
  - If timeout hit, REST fallback adds ~200ms
  - WS POST architecture: `HyperliquidExecutor -> ExchangeClient -> ws_info_client -> ws_manager`
  - Pending posts use request/response correlation
- Check network conditions:
  - WSL2 adds ~1ms overhead for network calls
  - VPN/proxy adds latency
  - Exchange geographic distance matters
- Check supervisor roundtrip tracking:
  - `SupervisorStats` includes connection quality metrics
  - If `is_reconnecting == true`, roundtrip will be high during reconnect
- **Resolution**: If WS POST consistently slow, check if REST fallback is faster (unlikely).
  Consider `WsPostConfig::fast()` preset. Monitor connection quality over time.

### Step 5: Model Computation Time

**Files**: `stochastic/hjb_solver.rs`, `strategy/glft.rs`, `strategy/signal_integration.rs`

- Check HJB solver computation:
  - `optimal_quotes()`: should be < 100us
  - Involves: gamma blending, half-spread formula, inventory skew, predictive bias
  - Size distribution: iterates over depths (O(n) where n = number of levels)
- Check signal integration:
  - `signal_integration.rs` is the central hub
  - Multiple signal sources aggregated
  - Staleness checks for each signal
  - Additive spread adjustments capped at 20 bps (prevents 3.4x cascade)
- Check GLFT strategy:
  - `half_spread()`: O(1) -- just `(1/gamma) * ln(1 + gamma/kappa)`
  - `half_spread_with_drift()`: O(1) -- adds mu*T/2 term
  - Inventory skew: O(1)
  - Regime blending: O(regimes) -- 4 regimes
- **Resolution**: If model computation is slow, check if any estimator is doing
  expensive recalibration in the hot path (should be periodic, not per-cycle).
  HMM recalibration uses VecDeque rolling buffers (cap 2000).
  MI null distribution uses logarithmic schedule to avoid O(n*k*log(n)) cost.
