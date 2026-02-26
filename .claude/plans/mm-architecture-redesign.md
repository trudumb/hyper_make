# Market Maker Architecture Redesign: Eliminating Adverse Selection

## Context

This document is the authoritative reference for a team of agents implementing the architecture redesign. It diagnoses why the current system "sells bottoms and buys tops," specifies the target architecture, maps changes to files, and defines verification criteria.

Read this entire document before starting any implementation work.

---

## Part 1: Root Cause Diagnosis

The system exhibits systematic adverse selection — getting filled on the wrong side just before price moves against it. Seven interrelated failure modes cause this, ordered by severity.

### 1.1 Reactive Signals Applied as Predictions (CRITICAL)

**The chain:**
```
T=0ms:    Binance price moves
T=50-200ms: HL price follows
T=~200ms:   on_binance_price() → lag_analyzer caches signal
T=~250ms:   get_signals() reads cached last_lead_lag_signal
T=~300ms:   Quotes placed on HL
```

By T=300ms, HL momentum followers have already repriced the book. The lead-lag signal is stale on arrival.

**Where:** `signal_integration.rs:543-596` (input) vs `signal_integration.rs:758-786` (output). The `last_lead_lag_signal` at line 778 is from the *previous* `on_binance_price()` call — potentially one or more quote cycles stale.

**The math problem:** The HJB skew formula `skew = γσ²qT/2 + β_t/2` uses β_t (predictive drift) that should be *forward-looking*. Instead, β_t is populated from the NIG posterior mean, which learns from past observations. The predictive bias has no Binance-HL divergence term.

**Fix required:** Predictive lead-lag: fire on first statistical divergence (changepoint detection on Binance lead), not after MI exceeds threshold. Apply skew preemptively — small skew at first sign of divergence, max skew at confirmation.

### 1.2 No Queue Value Model (CRITICAL)

**The math:** The GLFT formula `δ* = (1/γ) ln(1 + γ/κ)` assumes all fills at depth δ are equivalent. A fill at front-of-queue in quiet markets has positive expected edge. A fill at back-of-queue during a sweep has catastrophic adverse selection. The system cannot distinguish these.

**What exists:** Fill tracker (`execution/fill_tracker.rs`) records queue position and adverse selection rate — but only post-hoc for analytics. Queue position estimator (`tracking/queue/`) estimates position from L2 deltas. Fill probability model (`simulation/prediction.rs`) uses `1 - exp(-κ exp(-δ/δ_char) × T)`. None of these feed into quoting decisions.

**What's missing:**
- `queue_value(side, price_level, state) -> expected_edge_bps`: Expected edge conditional on being filled at this level with current queue position
- `fill_prob(side, level, horizon, state) -> p_fill`: Fill probability conditional on queue rank, not just depth
- `toxicity_score(state) -> {Benign, Normal, Toxic}`: Fast regime classification that bypasses slow EM convergence

**Fix required:** Three microstructure models (section 2.4) that condition on queue state and feed directly into execution decisions.

### 1.3 Skew Direction Semantic Collision (MAJOR)

**The blend** at `signal_integration.rs:978-983`:
```rust
raw_skew = base_skew_bps          // lead-lag: +1 = bullish
    + cross_venue_skew_bps         // cross-venue: +1 = bullish
    + buy_pressure_skew_bps        // z-score based
    + inventory_skew_bps           // -position_ratio × sensitivity (opposes position)
    + signal_skew_bps              // (alpha - 0.5) × 2.0 × max_bps
```

When long and the market is rallying:
- `inventory_skew_bps` is negative (lean asks to reduce long) — **correct**
- `signal_skew_bps` is positive (lean bids because bullish) — **fights inventory skew**
- These cancel, leaving near-zero skew — **wrong**

The inventory skew fights the alpha signal instead of complementing it. Then clamped to 80% of half-spread (`signal_integration.rs:982`), further limiting directional expression.

**Fix required:** Separate inventory management from directional conviction. The execution state machine (section 2.5) handles this: when you're long and the market is rallying, the correct action is `Mode::Maker` with tightened asks (ride the trend), not symmetric quoting with zero net skew.

### 1.4 Informed Flow Model Learns Too Slowly (MAJOR)

The EM-based informed flow decomposition (`estimator/informed_flow.rs`) updates every ~50 trades. In a cascade, 50 trades happen in 1-3 seconds. During convergence:
- `p_informed` slowly climbs from baseline (~0.1) toward actual (~0.6+)
- `spread_multiplier()` follows a sigmoid: barely widens until `p_informed > 0.3`
- By the time spreads widen meaningfully, 10-20 adverse fills have already occurred

The pre-fill classifier (`adverse_selection/pre_fill_classifier.rs`) exists and predicts toxicity before fills, but its output is a `spread_multiplier` — it can only change width, not direction. It can't say "don't quote the bid" or "switch to TakerExit mode."

**Fix required:** Fast toxicity classifier (section 2.4.3) that maps OFI features to discrete regimes within 1-2 trades, not 50. Output feeds into mode selection (section 2.5), not just spread width.

### 1.5 Quote Gate Too Permissive (MODERATE)

`quote_gate.rs:151`: `quote_flat_without_edge: true` (default).

When flat with no directional edge and ambiguous toxicity, the system quotes both sides symmetrically. On Hyperliquid with worse latency than informed flow, this offers free options to faster participants. They pick off whichever side is about to be wrong.

**Fix required:** The execution state machine (section 2.5) replaces the binary quote gate with explicit modes. When flat and edgeless, `Mode::Flat` means no quoting. `Mode::Maker` requires a positive queue value signal on at least one side.

### 1.6 No Fill Survival / Queue-Reactive Model (MODERATE)

Fill probability at `simulation/prediction.rs` is:
```
P(fill in [0, T]) = 1 - exp(-κ × exp(-δ/δ_char) × T)
```

This is symmetric — it doesn't distinguish between fills caused by:
- Price moving through your level (toxic sweep)
- Noise trader hitting your resting order (benign spread capture)

**Fix required:** Fill survival model (section 2.4.2) conditioned on queue rank, depth ahead, imbalance, OFI, and volatility regime.

### 1.7 Control Layer Solves the Wrong Problem (MODERATE)

The HJB solver (`hjb_solver.rs:179-256`) solves for optimal spread assuming symmetric Poisson arrivals with known intensity κ. Extensions (regime gamma, predictive bias) don't change the fundamental assumption: you can choose your fill quality by choosing your depth.

On Hyperliquid with worse latency, you get filled when someone *wants* to fill you — which is exactly when the fill is worth less than you're paying for it.

**Fix required:** The control loop (section 2.5.2) conditions on who is filling you (toxicity regime) and what your queue position implies about fill quality, not just whether you get filled.

---

## Part 2: Target Architecture

### 2.1 System Topology

```
┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  ┌─────────────┐
│ Feed Handler │→│ Feature Engine│→│ Alpha Engine  │→│ Execution Engine │→│ Risk Daemon  │
│             │  │              │  │              │  │                 │  │ (independent)│
│ L2 book +   │  │ Rolling      │  │ Stat-arb /   │  │ State machine + │  │ Hard stops + │
│ trades via  │  │ features per │  │ predictive   │  │ queue value +   │  │ kill switch  │
│ WebSocket   │  │ symbol       │  │ models       │  │ fill survival   │  │              │
└─────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  └─────────────┘
```

Communication: lock-free ring buffers (`crossbeam`) per data channel. Single-threaded event loops pinned to cores. Flat structs with pre-allocated pools, zero memcpy between stages.

**This maps to the existing codebase as:**

| Proposed Stage | Current Location | Changes Needed |
|---|---|---|
| Feed Handler | `messages/`, `infra/` (WS handling) | Minor: add normalized event enum, monotonic timestamps |
| Feature Engine | `estimator/`, `tracking/`, partial `strategy/` | Major: extract rolling feature computation from signal_integration into standalone module |
| Alpha Engine | `strategy/signal_integration.rs`, `belief/` | Major: separate predictive alpha from execution skew |
| Execution Engine | `orchestrator/quote_engine.rs`, `control/`, `quoting/` | Major: add state machine, queue value, fill survival |
| Risk Daemon | `risk/`, `safety/` | Minor: already mostly independent |

### 2.2 Market Data & Order Book

**Current state:** Hyperliquid WS in `messages/`, Binance in `infra/`. Book maintained in `tracking/`. Events not normalized.

**Required changes:**

```rust
// New: src/market_maker/events/normalized.rs
/// Single normalized event enum for all inbound data
pub enum MarketEvent {
    Trade { side: Side, qty: f64, px: f64, local_ts_ns: u64, remote_seq: u64 },
    BookDelta { side: Side, level: u8, px: f64, delta_qty: f64, local_ts_ns: u64, remote_seq: u64 },
    FundingUpdate { rate_8h: f64, next_settlement_ts: u64 },
}
```

All timestamps use local monotonic clock + remote sequence number. Never trust wall-clock alone.

**Files to modify:**
- `src/market_maker/events/mod.rs` — add `normalized.rs` module
- `src/market_maker/messages/` — normalize inbound WS messages to `MarketEvent`
- `src/market_maker/orchestrator/handlers.rs` — consume `MarketEvent` instead of raw WS types

### 2.3 Feature / State Engine

**Current state:** Features are computed inside `signal_integration.rs` (a 1000+ line monolith) as part of the quoting path. Rolling state is scattered across: `estimator/`, `tracking/`, and fields in `SignalIntegrator`.

**Required changes:**

```rust
// New: src/market_maker/features/state_snapshot.rs
/// Compact state snapshot published every 100ms
/// Contains everything needed for queue value, fill survival, and toxicity models
/// Target size: <16kb per snapshot
pub struct StateSnapshot {
    pub timestamp_ns: u64,

    // L1/L2
    pub best_bid: f64,
    pub best_ask: f64,
    pub spread_bps: f64,
    pub depth_bids: [f64; 10],   // size at each of top 10 bid levels
    pub depth_asks: [f64; 10],   // size at each of top 10 ask levels

    // Imbalance
    pub book_imbalance: f64,     // (Σb - Σa) / (Σb + Σa) for top k levels

    // Order flow
    pub ofi_500ms: f64,          // Order flow imbalance, 500ms window
    pub ofi_2s: f64,             // Order flow imbalance, 2s window
    pub trade_sign_run_length: i32,
    pub vwap_direction: f64,     // Volume-weighted trade direction

    // Volatility
    pub sigma_ewma: f64,         // EWMA of |Δmid|
    pub sigma_realized_1m: f64,  // 1-minute realized vol

    // Regime
    pub regime: Regime,          // {Quiet, Normal, Volatile, Cascade}
    pub regime_probs: [f64; 4],  // Soft HMM weights

    // Our state
    pub inventory: f64,
    pub queue_rank_bid: f64,     // Estimated queue rank at best bid
    pub queue_rank_ask: f64,     // Estimated queue rank at best ask
}
```

**Files to create:**
- `src/market_maker/features/mod.rs` — feature engine module
- `src/market_maker/features/state_snapshot.rs` — StateSnapshot struct
- `src/market_maker/features/rolling.rs` — rolling feature computation (extracted from signal_integration.rs)
- `src/market_maker/features/ofi.rs` — order flow imbalance computation

**Files to modify:**
- `src/market_maker/strategy/signal_integration.rs` — extract feature computation, keep only signal aggregation
- `src/market_maker/mod.rs` — register features module

### 2.4 Microstructure Models

Three models that plug into execution. All must be <50µs per inference.

#### 2.4.1 Queue Value Model

Predicts expected edge per unit size given StateSnapshot + hypothetical queue rank.

```rust
// New: src/market_maker/models/queue_value.rs
pub struct QueueValueModel {
    // Lightweight model: gradient boosted trees or linear with interactions
    // Trained offline on recorded book + simulated passive orders
    // Label: realized edge = spread captured - (mid move after fill × side)
}

impl QueueValueModel {
    /// Expected edge in bps for joining at this level
    /// Negative = expected adverse selection exceeds spread capture
    pub fn queue_value(&self, side: Side, price_level: f64, state: &StateSnapshot) -> f64;
}
```

**Training labels (offline):**
For each hypothetical join at best bid/ask at time t, track:
- Whether fully filled before: price moved > X ticks against, or time > H
- Realized edge: spread captured – (mid move after fill × side)

**Features:** Everything in StateSnapshot + hypothetical queue rank.

**Latency budget:** <50µs per inference. Keep model tiny, load in memory.

**Files to create:**
- `src/market_maker/models/mod.rs` — models module
- `src/market_maker/models/queue_value.rs` — queue value model + heuristic baseline
- `src/market_maker/models/training_labels.rs` — label generation from recorded data

#### 2.4.2 Fill Survival Model

Predicts time-to-fill conditioned on queue state.

```rust
// New: src/market_maker/models/fill_survival.rs
pub struct FillSurvivalModel {
    // Survival model: exponential with state-dependent rate
    // Conditioned on: level, queue rank fraction, depth ahead, imbalance, OFI, vol, toxicity regime
}

impl FillSurvivalModel {
    /// Probability of fill within horizon_s seconds
    pub fn fill_prob(&self, side: Side, level: f64, horizon_s: f64, state: &StateSnapshot) -> f64;

    /// Pre-computed for standard horizons: 0.5s, 1s, 3s
    pub fn fill_probs_standard(&self, side: Side, level: f64, state: &StateSnapshot) -> [f64; 3];
}
```

**What this replaces:** The current `estimate_fill_probability()` in `simulation/prediction.rs` which uses `1 - exp(-κ exp(-δ/δ_char) × T)` — symmetric and not queue-aware.

**Files to create:**
- `src/market_maker/models/fill_survival.rs` — survival model + heuristic baseline

**Files to modify:**
- `src/market_maker/simulation/prediction.rs` — deprecate old fill prob, delegate to new model

#### 2.4.3 Toxicity / Regime Classifier

Fast regime classification that fires within 1-2 trades, not 50.

```rust
// New: src/market_maker/models/toxicity.rs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToxicityRegime {
    Benign,  // Normal spread capture expected
    Normal,  // Standard conditions
    Toxic,   // High adverse selection, widen or pull
}

pub struct ToxicityClassifier {
    // Logistic regression on OFI lagged values, trade intensity asymmetry
    // Much faster convergence than EM (updates every trade, not every 50)
}

impl ToxicityClassifier {
    /// Current toxicity regime
    pub fn toxicity_score(&self, state: &StateSnapshot) -> ToxicityRegime;

    /// Update on every trade (not every 50)
    pub fn on_trade(&mut self, trade: &Trade, state: &StateSnapshot);
}
```

**What this replaces:** The slow EM-based informed flow decomposition in `estimator/informed_flow.rs` that takes ~50 trades to converge. The pre-fill classifier in `adverse_selection/pre_fill_classifier.rs` is kept but its output now feeds into mode selection, not just spread width.

**Files to create:**
- `src/market_maker/models/toxicity.rs` — fast toxicity classifier

**Files to modify:**
- `src/market_maker/adverse_selection/pre_fill_classifier.rs` — output feeds mode selection
- `src/market_maker/strategy/signal_integration.rs` — use toxicity classifier instead of EM p_informed for gating

### 2.5 Execution Engine: State Machine & Control Logic

This is the core architectural change. Replace the single-mode "always compute spread and skew" approach with an explicit state machine per symbol.

#### 2.5.1 Per-Symbol State

```rust
// New: src/market_maker/execution/state_machine.rs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    /// No position, no edge signal. Don't quote.
    Flat,

    /// Active market making. Quote one or both sides based on queue value.
    Maker,

    /// Actively entering a position via aggressive limit or market order.
    TakerEntry { direction: Side, timeout_ms: u64 },

    /// Actively exiting a position via aggressive limit or market order.
    TakerExit { urgency: f64 },

    /// Reducing inventory. More aggressive levels, ignore queue value unless benign regime.
    InventoryReduce { urgency: f64 },
}

pub struct SymbolMMState {
    pub mode: ExecutionMode,
    pub current_quotes: Option<ActiveQuotes>,
    pub target_inventory: f64,      // From alpha engine
    pub max_inventory: f64,         // Risk constraint
    pub last_action_ts: u64,
    pub last_mid: f64,
    pub last_regime: ToxicityRegime,
}

pub struct ActiveQuotes {
    pub bid_level: f64,             // Ticks from mid
    pub ask_level: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub bid_order_id: Option<u64>,
    pub ask_order_id: Option<u64>,
    pub queue_rank_bid: f64,        // Estimated queue rank
    pub queue_rank_ask: f64,
}
```

#### 2.5.2 Control Loop (Per Snapshot)

```rust
// Pseudocode for the control loop — each step is a concrete function
fn on_state_snapshot(&mut self, s: &StateSnapshot) {
    // 1. Update regime
    let regime = self.toxicity_classifier.toxicity_score(s);

    // 2. Compute alpha (target inventory from directional model)
    let target_inventory = self.alpha_engine.target_inventory(s);

    // 3. Compute risk envelope
    let clamped_target = target_inventory.clamp(-self.max_inventory, self.max_inventory);
    let vol_scaled_max = self.max_inventory / (s.sigma_realized_1m / self.baseline_sigma).max(1.0);
    let safe_target = clamped_target.clamp(-vol_scaled_max, vol_scaled_max);

    // 4. Select mode
    let mode = self.select_mode(s, regime, safe_target);

    // 5. Execute mode logic
    let order_actions = match mode {
        ExecutionMode::Flat => self.handle_flat(),
        ExecutionMode::Maker => self.handle_maker(s, regime, safe_target),
        ExecutionMode::TakerEntry { direction, timeout_ms } =>
            self.handle_taker_entry(s, direction, timeout_ms),
        ExecutionMode::TakerExit { urgency } =>
            self.handle_taker_exit(s, urgency),
        ExecutionMode::InventoryReduce { urgency } =>
            self.handle_inventory_reduce(s, urgency),
    };

    // 6. Order diffing — minimal set of cancels/replaces/new orders
    self.apply_order_diff(order_actions);
}

fn select_mode(&self, s: &StateSnapshot, regime: ToxicityRegime, target: f64) -> ExecutionMode {
    let inv = s.inventory;
    let max = self.max_inventory;

    if inv.abs() > max {
        return ExecutionMode::InventoryReduce { urgency: 1.0 };
    }

    if regime == ToxicityRegime::Toxic {
        if inv.abs() > max * 0.1 {
            return ExecutionMode::InventoryReduce { urgency: 0.8 };
        }
        return ExecutionMode::Flat;
    }

    if (target - inv).abs() > max * 0.3 {
        // Large gap between target and actual — consider taking
        let direction = if target > inv { Side::Buy } else { Side::Sell };
        // Compare E[market_fill] vs E[limit_fill] using models
        let taker_ev = self.estimate_taker_ev(s, direction);
        let maker_ev = self.estimate_maker_ev(s, direction);
        if taker_ev > maker_ev {
            return ExecutionMode::TakerEntry { direction, timeout_ms: 500 };
        }
    }

    if inv.abs() < max * 0.05 && target.abs() < max * 0.1 {
        // Near flat, small target — check if making is profitable
        let bid_qv = self.queue_value.queue_value(Side::Buy, s.best_bid, s);
        let ask_qv = self.queue_value.queue_value(Side::Sell, s.best_ask, s);
        if bid_qv <= 0.0 && ask_qv <= 0.0 {
            return ExecutionMode::Flat; // No positive edge on either side
        }
    }

    ExecutionMode::Maker
}

fn handle_maker(&self, s: &StateSnapshot, regime: ToxicityRegime, target: f64) -> Vec<OrderAction> {
    let mut actions = Vec::new();

    for side in [Side::Buy, Side::Sell] {
        // Evaluate candidate levels: best, best±1 tick
        let candidates = self.candidate_levels(s, side);

        let mut best_level = None;
        let mut best_epnl = f64::NEG_INFINITY;

        for level in candidates {
            let qv = self.queue_value.queue_value(side, level, s);
            let p_fill = self.fill_survival.fill_prob(side, level, 1.0, s);

            let inv_after = s.inventory + if side == Side::Buy { 1.0 } else { -1.0 };
            let inv_penalty = self.lambda_inv * inv_after.abs() * s.sigma_ewma;

            let epnl = qv - inv_penalty;

            if p_fill > self.p_fill_min && epnl > best_epnl {
                best_epnl = epnl;
                best_level = Some(level);
            }
        }

        if let Some(level) = best_level {
            let size = self.compute_size(s, side, target, regime);
            actions.push(OrderAction::Quote { side, level, size });
        }
    }

    actions
}
```

**Files to create:**
- `src/market_maker/execution/state_machine.rs` — ExecutionMode, SymbolMMState, control loop
- `src/market_maker/execution/mode_handlers.rs` — per-mode handler logic (Maker, Flat, TakerEntry, etc.)
- `src/market_maker/execution/order_diff.rs` — minimal order diff computation

**Files to modify:**
- `src/market_maker/orchestrator/quote_engine.rs` — delegate to state machine instead of direct GLFT → ladder
- `src/market_maker/control/quote_gate.rs` — replaced by state machine mode selection (deprecate)
- `src/market_maker/execution/mod.rs` — register new modules

### 2.6 OMS and Risk Layer

**Current state:** Mostly correct. The risk daemon (`risk/`, `safety/`) is already independent. Kill switch works.

**Required changes:**
- Max outstanding passive volume per symbol (enforce in OMS, not just quote engine)
- Min time between aggressive orders (anti-self-harm rate limit)
- Risk daemon can send `KillSwitch` via shared channel independent of execution thread

**Files to modify:**
- `src/market_maker/execution/order_lifecycle.rs` — add passive volume tracking
- `src/market_maker/risk/monitors/` — add per-symbol notional caps
- `src/market_maker/safety/auditor.rs` — add watchdog for execution thread liveness

### 2.7 Backtesting & Policy Improvement

**Current state:** Fill simulator in `simulation/` but no order-book replay. No latency model.

**Required changes:**

```rust
// New: src/market_maker/simulation/replay.rs
pub struct LatencyModel {
    /// Distribution of decision-to-exchange latencies (ms)
    pub order_latency: LatencyDistribution,
    /// Distribution of cancel-to-effective latencies (ms)
    pub cancel_latency: LatencyDistribution,
}

pub struct ReplayEngine {
    /// Reuses feed-handler and execution-engine in replay mode
    /// Reads historical ticks and simulates fill/cancel outcomes
    pub latency_model: LatencyModel,
}
```

**Files to create:**
- `src/market_maker/simulation/replay.rs` — order book replay with latency model
- `src/market_maker/simulation/latency_model.rs` — empirical latency distributions

**Files to modify:**
- `src/market_maker/simulation/mod.rs` — register new modules

---

## Part 3: Implementation Phases

### Phase 0: Foundations (no behavior change)

**Goal:** Create the new module structure, event normalization, and feature engine without changing any quoting behavior.

| Task | Files | Agent |
|---|---|---|
| Create `features/` module with StateSnapshot | `features/mod.rs`, `features/state_snapshot.rs` | signals |
| Create `models/` module with trait definitions | `models/mod.rs`, `models/queue_value.rs`, `models/fill_survival.rs`, `models/toxicity.rs` | signals |
| Create normalized event types | `events/normalized.rs` | infra |
| Create execution state machine types | `execution/state_machine.rs` | strategy |
| Register all new modules in `mod.rs` | `src/market_maker/mod.rs` | lead |
| Add `#[cfg(test)]` unit tests for all new types | All new files | all agents |

**Verification:** `cargo clippy -- -D warnings && cargo test`

### Phase 1: Feature Engine (extract, don't rewrite)

**Goal:** Extract rolling feature computation from `signal_integration.rs` into `features/` module. Wire StateSnapshot production. No quoting behavior change yet.

| Task | Files | Agent |
|---|---|---|
| Extract OFI computation from signal_integration | `features/ofi.rs`, `strategy/signal_integration.rs` | signals |
| Extract imbalance computation | `features/rolling.rs` | signals |
| Wire StateSnapshot production in quote engine | `orchestrator/quote_engine.rs` | infra |
| Add snapshot logging for offline model training | `features/logging.rs` | analytics |

**Verification:** StateSnapshot populated every quote cycle. All existing tests pass. Log file contains snapshots for offline analysis.

### Phase 2: Toxicity Classifier (fast regime detection)

**Goal:** Replace slow EM convergence with fast logistic toxicity classifier. This is the highest-impact single change.

| Task | Files | Agent |
|---|---|---|
| Implement logistic toxicity classifier | `models/toxicity.rs` | signals |
| Wire classifier into signal_integration | `strategy/signal_integration.rs` | signals |
| Add spread widening override on Toxic regime | `orchestrator/quote_engine.rs` | strategy |
| Add calibration metrics (Brier score for toxic predictions) | `calibration/` | analytics |

**Verification:** Toxicity flips to `Toxic` within 1-3 trades of a cascade. Spreads widen within 1 trade of regime flip. Brier score < 0.25 on backtest data.

### Phase 3: Queue Value Heuristic (bootstrap before ML)

**Goal:** Implement queue value as a heuristic first (no ML), then swap in learned model later. The heuristic uses: spread − expected_AS(regime) − fee. If negative, don't quote that side.

| Task | Files | Agent |
|---|---|---|
| Implement heuristic queue value | `models/queue_value.rs` | signals |
| Implement heuristic fill survival | `models/fill_survival.rs` | signals |
| Wire into quote engine as advisory signal | `orchestrator/quote_engine.rs` | strategy |
| Log queue value predictions + outcomes for training | `models/training_labels.rs` | analytics |

**Verification:** Queue value correctly negative during cascades. Positive during quiet periods. Prediction/outcome pairs logged for future ML training.

### Phase 4: Execution State Machine (the core change)

**Goal:** Replace single-mode quoting with explicit state machine. This is the most complex phase.

| Task | Files | Agent |
|---|---|---|
| Implement state machine core | `execution/state_machine.rs` | strategy |
| Implement Flat mode handler | `execution/mode_handlers.rs` | strategy |
| Implement Maker mode handler with queue value | `execution/mode_handlers.rs` | strategy |
| Implement InventoryReduce mode handler | `execution/mode_handlers.rs` | strategy |
| Implement order diff computation | `execution/order_diff.rs` | strategy |
| Wire state machine into quote engine | `orchestrator/quote_engine.rs` | infra (plan mode) |
| Deprecate quote gate (keep as fallback) | `control/quote_gate.rs` | strategy |
| Add mode transition logging | `tracking/` | analytics |

**Verification:**
- Mode transitions correct on simulated data (cascade → InventoryReduce, quiet → Maker)
- `Mode::Flat` fires when no edge on either side
- `Mode::InventoryReduce` fires immediately when inventory > max
- All existing risk limits still enforced (defense-in-depth)
- `cargo test` passes
- `cargo clippy -- -D warnings` clean

### Phase 5: Predictive Lead-Lag (latency compensation)

**Goal:** Fire lead-lag signal on first divergence, not after MI confirmation.

| Task | Files | Agent |
|---|---|---|
| Add changepoint detection on Binance-HL spread | `strategy/signal_integration.rs` | signals |
| Reduce MI threshold for preemptive skew | `strategy/signal_integration.rs` | signals |
| Scale skew by divergence magnitude, not binary | `strategy/signal_integration.rs` | signals |
| Wire into state machine's target_inventory | `execution/state_machine.rs` | strategy |

**Verification:** Lead-lag skew appears 50-100ms earlier than current. Adverse selection rate on lead-lag fills decreases.

### Phase 6: Replay & Policy Improvement

**Goal:** Order book replay engine for offline policy tuning.

| Task | Files | Agent |
|---|---|---|
| Implement replay engine | `simulation/replay.rs` | infra |
| Implement latency model | `simulation/latency_model.rs` | infra |
| Wire state machine into replay | `simulation/` | strategy |
| Add policy iteration logging | `analytics/` | analytics |

**Verification:** Replay produces same state transitions as live for recorded data. PnL attribution matches between replay and live.

---

## Part 4: Agent Assignments & File Ownership

### signals agent
Owns: `features/`, `models/`, `estimator/`, `adverse_selection/`, `calibration/`
- All feature computation and model implementation
- Toxicity classifier, queue value model, fill survival model
- OFI, imbalance, rolling feature extraction

### strategy agent
Owns: `strategy/`, `quoting/`, `execution/state_machine.rs`, `execution/mode_handlers.rs`, `execution/order_diff.rs`, `control/`
- State machine design and mode handler logic
- Signal integration changes (direction, not feature extraction)
- `signal_integration.rs` is strategy-only for skew/direction changes

### infra agent (plan mode required for orchestrator/)
Owns: `orchestrator/`, `infra/`, `messages/`, `events/`
- Event normalization
- Quote engine wiring (delegating to state machine)
- WebSocket handling, rate limiting

### analytics agent
Owns: `analytics/`, `tracking/`, `simulation/`, `calibration/`
- Prediction logging, label generation
- Mode transition logging
- Calibration metrics (Brier score, IR)
- Replay engine integration

### risk agent (plan mode required)
Owns: `risk/`, `safety/`
- Passive volume caps
- Watchdog for execution thread
- Defense-in-depth position checks

### lead (you)
Owns: `src/market_maker/mod.rs`, `Cargo.toml`, `.claude/`, `config/`
- Module registration
- Coordination between agents
- Plan approval for orchestrator/ and risk/ changes

---

## Part 5: File Inventory

### New Files to Create

```
src/market_maker/features/mod.rs
src/market_maker/features/state_snapshot.rs
src/market_maker/features/rolling.rs
src/market_maker/features/ofi.rs
src/market_maker/features/logging.rs

src/market_maker/models/mod.rs
src/market_maker/models/queue_value.rs
src/market_maker/models/fill_survival.rs
src/market_maker/models/toxicity.rs
src/market_maker/models/training_labels.rs

src/market_maker/execution/state_machine.rs
src/market_maker/execution/mode_handlers.rs
src/market_maker/execution/order_diff.rs

src/market_maker/events/normalized.rs

src/market_maker/simulation/replay.rs
src/market_maker/simulation/latency_model.rs
```

### Existing Files to Modify

```
src/market_maker/mod.rs                          — register features/, models/ modules
src/market_maker/strategy/signal_integration.rs  — extract features, use toxicity classifier
src/market_maker/orchestrator/quote_engine.rs    — delegate to state machine
src/market_maker/control/quote_gate.rs           — deprecate (keep as fallback)
src/market_maker/simulation/prediction.rs        — delegate to fill survival model
src/market_maker/execution/mod.rs                — register state machine
src/market_maker/events/mod.rs                   — register normalized events
src/market_maker/adverse_selection/pre_fill_classifier.rs — output feeds mode selection
src/market_maker/execution/order_lifecycle.rs    — add passive volume tracking
src/market_maker/simulation/mod.rs               — register replay
```

### Files NOT to Touch (keep as-is)

```
src/market_maker/stochastic/hjb_solver.rs  — correct math, just needs better inputs
src/market_maker/adaptive/blended_kappa.rs — works well, feeds into models
src/market_maker/risk/                     — mostly correct, minor additions only
src/market_maker/safety/                   — don't change safety-critical code unnecessarily
src/market_maker/fills/                    — fill processing works, just add new consumers
src/market_maker/checkpoint/               — add new fields with #[serde(default)]
```

---

## Part 6: Key Invariants (All Agents Must Enforce)

1. **kappa > 0.0** in all formula paths — GLFT blows up at zero
2. **gamma > 0.0** — risk aversion must be positive
3. **ask_price > bid_price** — spread invariant in all code paths
4. **inventory.abs() <= max_inventory** — hard limit, never violated
5. **No hardcoded parameters** — all values regime-dependent or configurable
6. **Units in variable names** — `spread_bps` not `spread`, `time_s` not `time`
7. **`#[serde(default)]`** on all checkpoint fields for backward compatibility
8. **Measurement before modeling** — log prediction/outcome pairs before building models
9. **Queue value consulted before every passive quote** — never quote with negative expected edge
10. **Toxicity regime consulted before every mode transition** — never enter Maker during Toxic

---

## Part 7: Success Criteria

| Metric | Current | Target | How to Measure |
|---|---|---|---|
| Adverse selection rate per fill | ~40-60% of fills | < 25% of fills | markout analysis at 5s |
| Mode = Flat during cascades | Never (always quoting) | >90% of cascade ticks | mode transition log |
| Queue value negative → no quote | Never checked | 100% enforcement | quote gate audit |
| Toxicity detection latency | ~50 trades (3-5s) | < 3 trades (< 0.5s) | regime flip timestamp vs cascade start |
| Lead-lag signal latency | ~250-350ms after Binance move | < 150ms | signal timestamp analysis |
| Fill quality distribution | Bimodal (good + catastrophic) | Unimodal (filtered catastrophic) | fill edge histogram |
| Quoting uptime in non-toxic regimes | ~12% (emergency pulls) | > 80% | time in Mode::Maker / time in non-Toxic |
| Inventory exceedance | 3.95x max observed | Never >1.0x | position tracking |

---

## Part 8: What This Document Replaces

This supersedes and extends:
- `.claude/plans/principled-architecture-redesign.md` (Feb 12 audit-based redesign)
- The L0-L5 layered pipeline remains correct; this document adds the execution state machine, microstructure models, and feature engine on top

The principled-architecture-redesign.md addressed the **internal** problems (tautological AS, floor binding, zero skew, emergency pull paralysis). This document addresses the **external** problem: the system is slower than informed flow on Hyperliquid and must compensate with better prediction, not just better spread math.

Both documents should be read together. Where they conflict, this document takes precedence.
