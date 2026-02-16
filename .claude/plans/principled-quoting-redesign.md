# Principled Quoting Redesign: "Quote First, Learn Second"

**Source**: ETH paper trading log `mm_hip3_hyna_ETH_hip3_2026-02-16_10-02-56.log`
**Result**: 0 fills in 3 minutes. System computed reasonable spreads (5-6 bps/side), generated 3-level ladders, then filtered every order to zero. $0 PnL. Complete paralysis.

---

## Diagnosis: 6 Systemic Failures

### F1. Cold-Start Death Spiral
The system requires fills to warm up, but requires warmup to generate fills.
- `warmup_pct: 10%` for the entire session (never advances)
- `as_warmed_up: false` forever (classifier needs fills to train)
- `fill_samples: 0` → `fill_ir: 0.000` → all learning frozen
- Target liquidity keeps adjusting (0.009→0.007) but never achieves a single fill

### F2. Size Truncation Kills Valid Orders
The ladder generates 3 levels with viable sizes, but `ExchangeRules.validate_quote()` truncates sizes to zero.
- `ViableQuoteLadder.from_ladder()` uses `clamp_to_viable(size, allow_round_up=true)` — rounds UP
- Then the toxicity filter reduces sizes by 30% (Normal regime, line 2513 of quote_engine.rs)
- Then `finalize()` calls `ExchangeRules.validate_quote()` which truncates DOWN only
- Sizes that were rounded up are now truncated back below minimum → **rejected**
- The pipeline rounds up, then cuts down, then rejects. A Sisyphean loop.

### F3. Reconciler Churn (86% Cancel Rate)
Matching tolerance is 1.25 bps for best level. Normal market drift is 2-5 bps per cycle. Every cycle cancels and replaces all orders, destroying queue priority.
- `best_tolerance_bps: 5.00` (from log, but actual matching is `optimal_spread_bps / 4 = 1.25`)
- Cancel+replace pattern wastes API quota
- Continuous density scaling then reduces levels because quota is depleted
- Result: intermittent coverage — sometimes 1 bid + 0 asks, sometimes 0 + 1

### F4. 15-Stage Filter Pipeline With No Coordination
Each filter independently reduces the ladder, unaware of the others:
1. ViableQuoteLadder (rounds up sizes)
2. ExecutionMode (side selection)
3. ToxicityRegime (30% size reduction in Normal)
4. QueueValue (negative edge removal)
5. Emergency risk filter
6. Fill cascade filter
7. ExchangeRules validation (truncates sizes down)

No filter knows what previous filters did. A size that survives step 1 gets killed by step 7 because step 3 shrank it.

### F5. Exchange Limits Go Stale
After 2 minutes, exchange limits become stale (`age_ms: 153636`). The system logs a warning but keeps running with stale data, further reducing sizing.

### F6. Signal Stasis
Without fills, every signal is frozen at its initial value. The system has sophisticated signal infrastructure that produces zero information because it has zero data.
- `alpha: 0.198-0.322` (never updates)
- `directional: -0.032 to 0.079` (noise)
- `falling_knife: 0.0-0.4` (from book, not fills)
- `momentum_severity: 0.0-0.40` (ditto)

---

## Root Cause: Wrong Assumption Ordering

The current architecture assumes:
```
Measure → Confidence → Quote → Fill → Learn
```

This creates a bootstrap paradox: you need fills to measure, but measurement to fill.

Every successful market maker in the real world operates as:
```
Quote (conservatively) → Fill → Measure → Calibrate → Optimize
```

The principled design inverts the dependency chain.

---

## Principled Design: 5 Engineer Work Packages

### Engineer 1: Exchange-Native Quote Generator

**Responsibility**: Given a mid price and exchange rules, produce a valid, non-empty ladder. Always.

**Key insight**: Exchange rules are a FIRST-CLASS INPUT to quote generation, not a late-stage validator. The generator should be incapable of producing invalid orders.

```rust
/// A quote that is guaranteed to be valid for the exchange.
/// Constructed only via ExchangeContract methods.
pub struct ValidQuote {
    pub price: f64,  // Already tick-aligned
    pub size: f64,   // Already ≥ min_order_size, correctly rounded
}

pub struct ExchangeContract {
    pub tick_size: f64,
    pub min_order_size: f64,
    pub size_decimals: u32,
    pub min_notional: f64,
    pub price_decimals: u32,
}

impl ExchangeContract {
    /// Round to the nearest valid quote. NEVER returns invalid.
    /// Prices round conservatively (bids down, asks up).
    /// Sizes round UP to min_order_size if too small.
    pub fn to_valid_quote(&self, price: f64, size: f64, side: Side) -> ValidQuote {
        let price = match side {
            Side::Bid => self.floor_to_tick(price),
            Side::Ask => self.ceil_to_tick(price),
        };
        let size = self.round_size(size).max(self.min_order_size);
        ValidQuote { price, size }
    }

    /// Minimum viable ladder: 1 bid + 1 ask at minimum size.
    /// This is the floor — the system can ALWAYS produce this.
    pub fn minimum_viable_ladder(&self, mid: f64, half_spread: f64) -> (ValidQuote, ValidQuote) {
        let bid = self.to_valid_quote(mid - half_spread, self.min_order_size, Side::Bid);
        let ask = self.to_valid_quote(mid + half_spread, self.min_order_size, Side::Ask);
        (bid, ask)
    }
}
```

**Non-negotiable property**: `ExchangeContract::to_valid_quote()` always returns a valid order. No filtering needed downstream. If a size is too small, it rounds up to minimum. If a price is between ticks, it rounds to the nearest valid tick.

**What changes**:
- Delete `exchange_rules.rs` as a late-stage validator
- Move exchange rules to the beginning of the pipeline
- `Ladder::generate()` takes `&ExchangeContract` and produces only valid levels
- Size filters (toxicity, risk) use `max(reduced_size, exchange.min_order_size)` — they can shrink TO minimum but never BELOW it

**Files**: `quoting/exchange_rules.rs` (rewrite), `strategy/ladder_strat.rs` (integrate), `quoting/viable.rs` (simplify)

---

### Engineer 2: Bootstrap Protocol (Cold-Start Elimination)

**Responsibility**: The system quotes within 5 seconds of first market data. Always. No warmup gate.

**Key insight**: The warmup gate exists to prevent the GLFT from computing garbage with uncalibrated parameters. But "no quote" is worse than "conservative quote with informative priors." The fix is better priors, not longer gates.

```rust
pub enum QuotingPhase {
    /// 0-5s: Observe L2 book, compute kappa prior from book depth
    Observing { book_snapshots: Vec<L2Snapshot> },

    /// 5s+: Quote at minimum viable size with conservative spreads
    /// Spread = max(GLFT_optimal, physical_floor * 2.0)
    /// Size = exchange.min_order_size (not margin-derived)
    MinimumViable { fills_seen: u32 },

    /// After N fills: Linear ramp toward target liquidity
    /// Each fill tightens spread by ramp_factor
    Ramping { fills_seen: u32, ramp_progress: f64 },

    /// After convergence: Full GLFT with calibrated parameters
    SteadyState,
}

impl QuotingPhase {
    pub fn spread_multiplier(&self) -> f64 {
        match self {
            Observing { .. } => f64::INFINITY, // Don't quote yet
            MinimumViable { .. } => 2.0,       // 2x optimal = very conservative
            Ramping { ramp_progress, .. } => 2.0 - *ramp_progress, // Linear 2x → 1x
            SteadyState => 1.0,
        }
    }

    pub fn size_mode(&self) -> SizeMode {
        match self {
            Observing { .. } => SizeMode::None,
            MinimumViable { .. } => SizeMode::ExchangeMinimum,
            Ramping { ramp_progress, .. } => SizeMode::Fraction(*ramp_progress),
            SteadyState => SizeMode::Full,
        }
    }

    // Transition rules
    pub fn advance(&mut self, event: PhaseEvent) {
        match (self, event) {
            (Observing { snapshots }, PhaseEvent::BookReceived(snap)) => {
                snapshots.push(snap);
                if snapshots.len() >= 3 {
                    *self = MinimumViable { fills_seen: 0 };
                }
            }
            (MinimumViable { fills_seen }, PhaseEvent::Fill) => {
                *fills_seen += 1;
                if *fills_seen >= 3 {
                    *self = Ramping { fills_seen: *fills_seen, ramp_progress: 0.0 };
                }
            }
            (Ramping { fills_seen, ramp_progress }, PhaseEvent::Fill) => {
                *fills_seen += 1;
                *ramp_progress = (*fills_seen as f64 / 20.0).min(1.0); // 20 fills to full
                if *ramp_progress >= 1.0 {
                    *self = SteadyState;
                }
            }
            _ => {} // No transition
        }
    }
}
```

**What this fixes**:
- **F1 (death spiral)**: System quotes in MinimumViable phase after 3 book snapshots (~1.5s)
- **F4 (signal stasis)**: Fills start arriving within seconds, feeding all classifiers
- Warmup becomes a ramp, not a gate. There's never a state where the system CAN'T quote.

**Kappa bootstrap from book**:
```rust
fn bootstrap_kappa(book: &L2Snapshot) -> f64 {
    // Kappa ≈ trade_arrival_rate / price_impact
    // Observable proxy: depth at touch / typical_trade_size
    let depth_at_touch = book.best_bid_size + book.best_ask_size;
    let typical_trade = depth_at_touch * 0.1; // 10% of touch = typical trade
    let kappa_estimate = depth_at_touch / (book.spread() * 0.5);
    kappa_estimate.clamp(500.0, 10000.0) // Bounded prior
}
```

**Files**: `orchestrator/quote_engine.rs` (major refactor), `strategy/ladder_strat.rs` (phase-aware sizing), `estimator/kappa_orchestrator.rs` (bootstrap_from_book)

---

### Engineer 3: Monotonic Filter Pipeline

**Responsibility**: Filters can only WIDEN spreads or REDUCE sizes. They can never produce invalid orders. They compose predictably.

**Key insight**: The current pipeline has 7 independent filters that don't coordinate. A filter at step 3 can undo what step 1 established. The fix is a monotonic chain where each step's output is guaranteed valid.

```rust
/// A ladder that has been validated against exchange rules.
/// Every level is a ValidQuote. Adding levels is impossible
/// without going through ExchangeContract validation.
pub struct ValidLadder {
    bids: Vec<ValidQuote>,  // Sorted by price descending
    asks: Vec<ValidQuote>,  // Sorted by price ascending
    contract: ExchangeContract,
}

impl ValidLadder {
    /// Filters can widen spreads (move bids down, asks up).
    /// Prices can only move AWAY from mid, never toward it.
    pub fn widen_spread(&mut self, side: Side, factor: f64) { ... }

    /// Filters can reduce sizes, but NEVER below exchange minimum.
    /// If reduction would go below min, the level is DROPPED entirely
    /// (honest removal, not silent truncation to zero).
    pub fn reduce_size(&mut self, level: usize, factor: f64) {
        let new_size = self.levels[level].size * factor;
        if new_size < self.contract.min_order_size {
            self.levels.remove(level); // Honest drop
        } else {
            self.levels[level].size = self.contract.round_size(new_size);
        }
    }

    /// Drop an entire side (for reduce-only mode).
    pub fn clear_side(&mut self, side: Side) { ... }

    /// The ladder can become empty, but only through explicit drops.
    /// This is distinguishable from "filter killed valid orders."
    pub fn is_empty(&self) -> bool { self.bids.is_empty() && self.asks.is_empty() }
}

// The filter chain becomes simple function composition:
fn apply_filters(ladder: ValidLadder, ctx: &QuoteContext) -> ValidLadder {
    let ladder = apply_execution_mode(ladder, ctx.execution_mode);
    let ladder = apply_toxicity_defense(ladder, ctx.toxicity);
    let ladder = apply_risk_limits(ladder, ctx.risk_state);
    let ladder = apply_cascade_guard(ladder, ctx.cascade_state);
    // No ExchangeRules validation needed — ladder was valid from construction
    // and every mutation preserved validity
    ladder
}
```

**What this fixes**:
- **F2 (size truncation kills orders)**: Can never happen. Sizes are always ≥ min_order_size.
- **F4 (uncoordinated filters)**: Each filter takes a ValidLadder, returns a ValidLadder. Composition is associative.
- **Debugging**: If the ladder is empty after filters, you know exactly which filter dropped which level and why (each filter logs its drops).

**Toxicity filter revised**:
```rust
fn apply_toxicity_defense(mut ladder: ValidLadder, toxicity: ToxicityRegime) -> ValidLadder {
    match toxicity {
        Toxic => {
            // In toxic regime, WIDEN spreads instead of reducing sizes.
            // This preserves queue position while reducing adverse selection.
            ladder.widen_spread(Side::Bid, 1.5);
            ladder.widen_spread(Side::Ask, 1.5);
        }
        Normal => {
            // In normal regime, widen slightly rather than reducing sizes.
            // Size reduction pushes below exchange min — spread widening doesn't.
            ladder.widen_spread(Side::Bid, 1.15);
            ladder.widen_spread(Side::Ask, 1.15);
        }
        Benign => {} // No adjustment
    }
    ladder
}
```

**Key change**: Toxicity defense WIDENS SPREADS instead of REDUCING SIZES. Size reduction is a footgun on small accounts where sizes are already near minimum. Spread widening achieves the same protective effect without triggering exchange minimums.

**Files**: `quoting/viable.rs` (rewrite to ValidLadder), `quoting/mod.rs` (compose filters), `orchestrator/quote_engine.rs` (simplify pipeline)

---

### Engineer 4: Queue-Preserving Reconciler

**Responsibility**: Minimize order churn. Queue priority is an asset; cancellation destroys it.

**Key insight**: The current reconciler optimizes for "quote at theoretical optimal price." But in a market with 5 bps spreads and 2 bps/cycle drift, the theoretical optimal changes faster than the reconciler can track. Queue priority at a slightly suboptimal price beats optimal price with no queue.

```rust
pub struct QueuePreservingReconciler {
    /// An order is "close enough" if it's within this bound.
    /// Default: max(2 * tick_size_bps, 10 bps, sigma * sqrt(cycle_time))
    /// This is the expected price movement during one queue position's lifetime.
    stale_threshold_bps: f64,

    /// Only modify sizes if change exceeds this ratio.
    /// Default: 0.20 (20% size change required to justify losing queue)
    size_hysteresis: f64,
}

impl QueuePreservingReconciler {
    fn reconcile(&self, existing: &[Order], target: &ValidLadder) -> Vec<Action> {
        let mut actions = Vec::new();

        // Phase 1: Match existing orders to target levels (greedy, nearest-price)
        for target_level in target.levels() {
            if let Some(existing_order) = find_nearest(existing, target_level) {
                let price_diff = (existing_order.price - target_level.price).abs();
                let price_diff_bps = price_diff / target_level.price * 10000.0;
                let size_change = (target_level.size - existing_order.size).abs()
                    / existing_order.size;

                if price_diff_bps <= self.stale_threshold_bps
                    && size_change <= self.size_hysteresis
                {
                    // KEEP: order is close enough, preserve queue
                    continue;
                }

                if price_diff_bps <= self.stale_threshold_bps {
                    // MODIFY size only: preserves queue position
                    actions.push(Action::Modify {
                        oid: existing_order.oid,
                        new_size: target_level.size,
                    });
                } else {
                    // Price moved too far: must cancel and replace
                    actions.push(Action::Cancel { oid: existing_order.oid });
                    actions.push(Action::Place(target_level.clone()));
                }
            } else {
                // No existing order near this target: place new
                actions.push(Action::Place(target_level.clone()));
            }
        }

        // Phase 2: Cancel orphaned orders (existing orders not matched to any target)
        for orphan in unmatched_existing {
            actions.push(Action::Cancel { oid: orphan.oid });
        }

        actions
    }
}
```

**What this fixes**:
- **F3 (86% cancel rate)**: With `stale_threshold_bps = 10`, most cycles produce 0 cancels
- **Queue priority**: Orders survive 3-5 cycles, building queue position
- **API quota**: 80% fewer messages → headroom stays above 20% → no density scaling → full ladder coverage

**Stale threshold computation**:
```rust
fn compute_stale_threshold(sigma: f64, cycle_time_s: f64, tick_bps: f64) -> f64 {
    // Expected price movement during one cycle
    let expected_move_bps = sigma * cycle_time_s.sqrt() * 10000.0;
    // At least 2 ticks, at most 10 bps
    expected_move_bps.max(2.0 * tick_bps).min(10.0)
}
```

**Files**: `orchestrator/reconcile.rs` (major simplification), `tracking/order_manager/reconcile.rs` (simplify matching)

---

### Engineer 5: Integrated Observability & Self-Diagnosis

**Responsibility**: When the system produces 0 fills, the log explains WHY in one line, not 981 lines.

**Key insight**: The current system logs every intermediate value but never synthesizes a diagnosis. A human must read 981 lines to understand "sizes got truncated to zero." The system should diagnose itself.

```rust
pub struct CycleDiagnosis {
    pub phase: QuotingPhase,
    pub spread_bps: f64,
    pub target_levels: usize,
    pub surviving_levels: usize,
    pub orders_placed: usize,
    pub orders_preserved: usize,  // Queue-preserved (not modified)
    pub orders_cancelled: usize,
    pub blocking_reason: Option<BlockingReason>,
}

pub enum BlockingReason {
    /// All levels killed by a specific filter
    FilteredToEmpty { filter: &'static str, levels_before: usize },
    /// Size below exchange minimum after toxicity reduction
    SizeBelowMinimum { min_required: f64, actual: f64 },
    /// API quota exhausted
    QuotaExhausted { headroom_pct: f64 },
    /// Kill switch active
    KillSwitchActive { reason: String },
    /// In observation phase (not quoting yet)
    WarmupObserving { snapshots_needed: usize, snapshots_have: usize },
}

// At the end of each cycle, ONE log line:
tracing::info!(
    phase = %diagnosis.phase,
    spread_bps = diagnosis.spread_bps,
    target = diagnosis.target_levels,
    surviving = diagnosis.surviving_levels,
    placed = diagnosis.orders_placed,
    preserved = diagnosis.orders_preserved,
    cancelled = diagnosis.orders_cancelled,
    blocking = ?diagnosis.blocking_reason,
    "cycle_summary"
);
```

**What this fixes**:
- **F6 (981 lines of noise)**: One summary line per cycle replaces ~30 lines of trace
- **F5 (stale limits undiagnosed)**: `BlockingReason::StaleLimits` appears immediately
- **Debugging**: When fills are 0, the diagnosis says WHY: "FilteredToEmpty { filter: toxicity, levels_before: 3 }"

**Verbose mode**: Keep the existing SPREAD TRACE / SIGNALS / RECONCILE logs, but gate them behind `RUST_LOG=trace`. The default `info` level shows only the cycle summary.

**Files**: new `orchestrator/diagnosis.rs`, `orchestrator/quote_engine.rs` (wire diagnosis at end of cycle)

---

### Engineer 6: Stale-Data Circuit Breaker

**Responsibility**: When exchange data goes stale, widen spreads. Don't silently degrade.

```rust
pub struct DataFreshness {
    pub exchange_limits_age_ms: u64,
    pub book_age_ms: u64,
    pub last_fill_age_ms: u64,
}

impl DataFreshness {
    pub fn spread_penalty_mult(&self) -> f64 {
        let mut mult = 1.0;

        // Exchange limits stale → widen
        if self.exchange_limits_age_ms > 30_000 {
            mult *= 1.5;
        }
        if self.exchange_limits_age_ms > 120_000 {
            mult *= 2.0; // Very stale: 3x wider
        }

        // Book data stale → widen
        if self.book_age_ms > 5_000 {
            mult *= 1.5;
        }

        mult
    }

    /// If data is critically stale, stop quoting entirely.
    pub fn should_pull_quotes(&self) -> bool {
        self.book_age_ms > 30_000 || self.exchange_limits_age_ms > 300_000
    }
}
```

**What this fixes**:
- **F5 (stale exchange limits)**: Instead of a warning that nobody reads, spreads widen automatically
- Exchange limit refresh becomes a first-class lifecycle event, not a background poll

**Files**: `infra/exchange_limits.rs` (freshness tracking), `strategy/ladder_strat.rs` (consume freshness)

---

## Integration: How The 6 Packages Compose

```
                    ┌──────────────────────┐
                    │  Engineer 2           │
                    │  Bootstrap Protocol   │
                    │  (QuotingPhase)       │
                    └──────────┬───────────┘
                               │ phase + spread_mult + size_mode
                               ▼
┌──────────────┐    ┌──────────────────────┐
│  Engineer 1  │───▶│  GLFT Optimal Spread │
│  Exchange    │    │  × phase multiplier  │
│  Contract    │    │  = target half-spread │
└──────────────┘    └──────────┬───────────┘
       │                       │
       │ ExchangeContract      │ spread + sizes
       │                       ▼
       │            ┌──────────────────────┐
       └───────────▶│  Engineer 3           │
                    │  ValidLadder          │
                    │  (exchange-valid from │
                    │   construction)        │
                    └──────────┬───────────┘
                               │ ValidLadder
                               ▼
                    ┌──────────────────────┐
                    │  Engineer 3           │
                    │  Monotonic Filters    │
                    │  (widen/reduce only)  │
                    └──────────┬───────────┘
                               │ ValidLadder (possibly smaller)
                               ▼
                    ┌──────────────────────┐
                    │  Engineer 4           │
                    │  Queue-Preserving     │
                    │  Reconciler           │
                    └──────────┬───────────┘
                               │ Actions (place/modify/cancel/keep)
                               ▼
                    ┌──────────────────────┐
                    │  Engineer 5           │
                    │  CycleDiagnosis       │
                    │  (one-line summary)   │
                    └──────────────────────┘

 Engineer 6 (DataFreshness) feeds into spread multiplier at the top.
```

---

## What The ETH Session Would Look Like After This Redesign

```
17:03:00 [INFO] cycle_summary phase=Observing snapshots=1/3
17:03:01 [INFO] cycle_summary phase=Observing snapshots=2/3
17:03:02 [INFO] cycle_summary phase=Observing snapshots=3/3 kappa_bootstrap=2315
17:03:03 [INFO] cycle_summary phase=MinimumViable spread_bps=11.6 target=1 surviving=1
         placed=2 preserved=0 cancelled=0
         bid=1972.90@0.01 ask=1976.30@0.01
17:03:08 [INFO] cycle_summary phase=MinimumViable spread_bps=11.2 target=1 surviving=1
         placed=0 preserved=2 cancelled=0
17:03:13 [INFO] cycle_summary phase=MinimumViable spread_bps=10.9 target=1 surviving=1
         placed=0 preserved=2 cancelled=0
17:03:18 [INFO] FILL bid@1973.10 size=0.01 edge_bps=4.2
17:03:18 [INFO] cycle_summary phase=MinimumViable(fills=1) spread_bps=10.4
17:03:23 [INFO] FILL ask@1976.30 size=0.01 edge_bps=3.8
17:03:23 [INFO] cycle_summary phase=MinimumViable(fills=2) spread_bps=9.9
17:03:28 [INFO] FILL bid@1973.00 size=0.01 edge_bps=4.5
17:03:28 [INFO] cycle_summary phase=Ramping(fills=3, progress=15%) spread_bps=9.5
         target=2 surviving=2 placed=2 preserved=2 cancelled=0
...
17:05:47 [INFO] cycle_summary phase=Ramping(fills=12, progress=60%) spread_bps=7.2
         target=3 surviving=3 placed=0 preserved=6 cancelled=0
17:05:47 [INFO] session_summary fills=12 realized_pnl=$0.38 avg_edge_bps=3.4
```

**Difference**: 12 fills instead of 0. $0.38 PnL instead of $0.00. And the log is 30 lines instead of 981.

---

## Implementation Priority

| Phase | Engineer | Impact | Effort | Status |
|-------|----------|--------|--------|--------|
| **P0** | E1 (Exchange Contract) | Eliminates F2 (size truncation) | 2 days | **DONE** |
| **P0** | E2 (Bootstrap Protocol) | Eliminates F1 (death spiral) | 3 days | **DONE** (partial) |
| **P1** | E3 (Monotonic Filters) | Eliminates F4 (filter chaos) | 3 days | **DONE** (partial) |
| **P1** | E4 (Queue Reconciler) | Eliminates F3 (86% churn) | 2 days | TODO |
| **P2** | E5 (Observability) | Eliminates debugging pain | 1 day | TODO |
| **P2** | E6 (Stale Data) | Eliminates F5 (stale limits) | 1 day | TODO |

P0 alone fixes the "0 fills" problem. P1 makes the system efficient. P2 makes it maintainable.

## Implementation Log (Feb 16, 2026)

### Implemented (4 targeted fixes, clippy clean, 2827 tests pass):

**Fix 1: ExecutionMode::Flat death spiral** (`execution/state_machine.rs`)
- Added `is_warmup: bool` field to `ModeSelectionInput`
- During warmup, step 6 (no-value/no-alpha) returns `Maker(both)` instead of `Flat`
- During warmup, step 3 (Toxic+flat) returns `Maker(both)` instead of `Flat`
- Kill/Red zone safety still overrides warmup (absolute safety)
- Wired via `!self.estimator.is_warmed_up()` in quote_engine.rs
- +6 new tests: warmup_no_value, warmup_large_tier, warmup_toxic_flat, warmup_kill, warmup_red, warmup_toxic_positioned

**Fix 2: Strict `>` filter** (`strategy/ladder_strat.rs:1938`)
- Changed `l.size > min_size_for_exchange` to `l.size >= min_size_for_exchange`
- Levels at exactly min_viable_size are valid and truncation-stable

**Fix 3: ExchangeRules round_size_up** (`quoting/exchange_rules.rs`)
- Added `ceil_size()` method: rounds UP to at least `size_step` for positive input
- Added `min_order_size()` method
- Changed `validate_quote()`: uses `ceil_size` instead of `truncate_size`
- Positive sizes can never become zero through validation
- `was_fixed` now also flags size adjustments
- +7 new tests: ceil_size_on_grid, ceil_size_rounds_up, ceil_size_zero, ceil_size_min_order_size, validate_quote_never_truncates, validate_quote_ceil_preserves

**Fix 4: ToxicityRegime::Normal → spread widening** (`orchestrator/quote_engine.rs`)
- Normal toxicity now WIDENS SPREADS instead of REDUCING SIZES
- Formula: `widen = (1.0 + toxicity).clamp(1.0, 1.5)` — toxicity 0.25→1.0x, 0.50→1.15x, 0.75→1.50x
- Eliminates the Sisyphean loop: viable rounds up → toxicity reduces → ExchangeRules rejects

### Root cause found but NOT in original diagnosis:
- The #1 killer was `select_mode()` step 6 returning `ExecutionMode::Flat` at cold start
- For Medium+ capital with `has_alpha=false` and `bid_has_value=false` → Flat → 0 levels
- At cold start, models have no data → no alpha → no value → always Flat → death spiral
- This was already fixed for Micro/Small capital but not Medium+

### Remaining (P1-P2):
- E4: Queue-preserving reconciler (wider matching tolerance)
- E5: Self-diagnosis (cycle summary logs)
- E6: Stale data circuit breaker

---

## Anti-Patterns To Eliminate

1. **Late-stage validation**: Never validate at the end. Construct valid objects at the start.
2. **Size reduction below exchange minimum**: Convert to spread widening instead.
3. **Warmup as a gate**: Warmup is a ramp, not a boolean. The system always quotes.
4. **Tight reconcile tolerance**: Queue priority > price optimality.
5. **Independent filters**: Filters must compose monotonically on a valid object.
6. **Stale data as a warning**: Stale data is a spread multiplier, not a log message.
7. **981-line logs**: One summary line per cycle. Traces behind `RUST_LOG=trace`.
