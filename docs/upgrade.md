 Stable Order Lifecycle Architecture

 Context

 Problem: 5 minutes of live ETH trading on hyna DEX produced 60 orders placed, 58 cancelled, 0 fills. Every ~6-second cycle: cancel both resting orders, place 2 new ones. 40 API
 calls/min for zero economic value. On a 7% quota budget, this is catastrophic.

 Metrics progression (from log):

 ┌────────┬────────┬───────────┬────────┐
 │ Uptime │ Placed │ Cancelled │ Filled │
 ├────────┼────────┼───────────┼────────┤
 │ 60s    │ 0      │ 0         │ 0      │
 ├────────┼────────┼───────────┼────────┤
 │ 120s   │ 8      │ 6         │ 0      │
 ├────────┼────────┼───────────┼────────┤
 │ 180s   │ 26     │ 24        │ 0      │
 ├────────┼────────┼───────────┼────────┤
 │ 240s   │ 42     │ 42        │ 0      │
 ├────────┼────────┼───────────┼────────┤
 │ 300s   │ 60     │ 58        │ 0      │
 └────────┴────────┴───────────┴────────┘

 Root Causes (4 interlocking failures):

 1. Kappa jitter — recomputed from 4 sources/cycle (robust, adaptive, legacy, regime@60%). Oscillates 1169→1176→1194→1169. Through GLFT δ=(1/γ)ln(1+γ/κ), produces 1-6 bps price shifts
  per cycle.
 2. Price grid disabled — PriceGrid exists in quoting/price_grid.rs with full snap logic, tests, and wiring points... but enabled: false by default. Sub-tick floating-point
 differences treated as new prices.
 3. Modify rate-limited to uselessness — full modify infra exists (10 files), but can_modify() enforces 2000ms debounce. When gated, actions silently dropped. Next cycle: drift grew →
  CANCEL+PLACE (costs MORE than modify would have). Classic timeout inversion.
 4. Tight latch threshold — floors at 1-2 bps, catching nearly every cycle given kappa noise floor.

 Key insight: The system already has most building blocks (ParameterSmoother with 19 tests, PriceGrid wired in 6 files, modify_bulk_orders in executor, priority_based_matching with
 queue awareness). They're just disabled, unwired, or misconfigured.

 ---
 Architecture: 6 Layers, 6 Engineers

 Core principle: Orders are assets with queue position value. Every cancel destroys value. The system should minimize order mutations while maintaining price accuracy.

 [Parameter Sources] → L1:Smoothing → L2:PriceGrid → [Ladder]
                                                         ↓
 [Exchange] ← L3:ModifyFirst ← L4:ToleranceBands ← [Reconciler]
                     ↑                                    ↑
                L5:QuotaBudget                    L6:ChurnMetrics


 ---
 Layer 1: Parameter Smoothing — Wire Existing ParameterSmoother

 Engineer: signals agent
 Effort: ~2h (wiring only, implementation complete)
 Risk: Zero (config-gated, enabled: false default)

 What exists: src/market_maker/strategy/params/smoothing.rs
 - ParameterSmoother struct with EMA + deadband + regime-change fast-track
 - 19 passing unit tests
 - smooth(&mut MarketParams, regime_changed: bool) ready to call
 - NOT referenced outside its own file — needs wiring

 Changes:
 1. src/market_maker/mod.rs — add parameter_smoother: ParameterSmoother field to MarketMaker struct, initialize with ParameterSmoother::new(SmootherConfig::default())
 2. src/market_maker/orchestrator/quote_engine.rs — call self.parameter_smoother.smooth(&mut market_params, regime_changed) immediately after ParameterAggregator::build() and before
 any downstream consumption. Single call site ensures all consumers see smoothed values.
 3. Config: SmootherConfig already has #[serde(default)] — just needs threading via MarketMakerConfig

 Recommended defaults: kappa_alpha: 0.10, kappa_deadband_pct: 0.05 (5% deadband — kappa must change by 5% to propagate). Kappa oscillation 1169→1176 (0.6% change) is suppressed. Only
 genuine shifts (>5%) pass through.

 Impact: Eliminates ~80% of price drift at the source. All downstream layers benefit.

 ---
 Layer 2: Price Grid — Enable Existing PriceGrid

 Engineer: strategy agent
 Effort: ~3h (enable + enhance directional snapping)
 Risk: Low (config-gated, 11 existing tests)

 What exists: src/market_maker/quoting/price_grid.rs
 - PriceGrid::for_current_state(), snap(), same_point()
 - Sigma-adaptive spacing, quota-pressure scaling
 - Already wired in quote_engine.rs:2406, reconcile.rs:1315, priority_based_matching:697
 - enabled: false by default

 Changes:
 1. src/market_maker/quoting/price_grid.rs — add snap_bid() (floors to grid) and snap_ask() (ceils to grid) to prevent bid/ask collapsing to same grid point near mid
 2. src/market_maker/orchestrator/quote_engine.rs:2406-2423 — use directional snap: grid.snap_bid() for bids, grid.snap_ask() for asks
 3. Enable via config: PriceGridConfig { enabled: true, min_tick_multiple: 2.0, sigma_divisor: 4.0, pressure_scaling: 2.0 }
 4. Add grid epoch counter — when grid hasn't shifted, reconciler can fast-path skip

 Grid math for ETH@$2000: tick=0.01 (0.5 bps), sigma_1m~5 bps → grid_step = max(1.0 bps, 5/4=1.25 bps) = 1.25 bps = $0.25. Kappa jitter producing <1 bps price shifts is fully
 absorbed.

 Impact: Quantizes remaining drift into discrete steps. same_point() in priority_based_matching preserves orders without any API call.

 ---
 Layer 3: Modify-First Order Lifecycle

 Engineer: infra agent (plan mode)
 Effort: ~3h
 Risk: Medium (changes modify debounce behavior)

 What exists:
 - HyperliquidExecutor::modify_bulk_orders() — fully implemented with OID remapping
 - LadderAction::Modify — generated by priority_based_matching()
 - ProactiveRateTracker::can_modify() — 2000ms debounce gate

 Changes:
 1. src/market_maker/infra/rate_limit/proactive.rs — reduce min_modify_interval_ms from 2000 to 500. Still prevents intra-cycle double-modify, but doesn't block next cycle.
 2. src/market_maker/orchestrator/reconcile.rs:1571-1576 — when modify is debounced, latch the orders (keep resting) instead of silently dropping. The drift is within modify range
 (tighter than cancel threshold), so the order is "close enough":
 // WAS: silently drop, causing cancel+place next cycle
 // NOW: latch (keep resting), retry modify next cycle
 debug!("Modify debounced: latching orders (drift within modify range)");
 3. src/market_maker/tracking/order_manager/reconcile.rs — in priority_based_matching(), generate LadderAction::Modify for price changes within max_modify_price_bps (currently only
 used for size-down). Price-modify loses queue on Hyperliquid but saves 1 API call vs cancel+place.

 Impact: When action IS needed, uses 1 API call instead of 2. Eliminates the timeout inversion trap.

 ---
 Layer 4: Reconciler Tolerance Bands

 Engineer: strategy agent or lead (owns tracking/order_manager/reconcile.rs)
 Effort: ~3h
 Risk: Medium (affects reconciliation decisions)

 What exists:
 - DynamicReconcileConfig with gamma/kappa/sigma-aware threshold computation
 - priority_based_matching() with multi-tier tolerance per level priority
 - latch_threshold_bps in DynamicReconcileConfig

 Changes:
 1. Queue-position-weighted latch: In priority_based_matching() latch check, widen tolerance for orders with good queue position (low depth_ahead). Front-of-queue orders get +50%
 latch bonus. Rationale: their queue value is highest, so the cost of replacing them is highest.
 2. Raise latch floor: Change normal-headroom latch clamp from (2.0, 10.0) to (3.0, 10.0). The 2 bps floor is below the noise floor even after smoothing.
 3. Remove dead function: Delete standalone latch_threshold_bps() at line 22-40 of orchestrator/reconcile.rs — superseded by DynamicReconcileConfig.latch_threshold_bps, creates
 confusion.
 4. Outer level relaxation: Ensure outer levels (priority > 0) have 2-3x the latch of best level. They are less fill-critical and churning them wastes quota.

 Impact: Widens the "no action needed" zone. With smoothing + grid + wider latch, most cycles produce zero actions.

 ---
 Layer 5: Quota-Aware Decision Making

 Engineer: infra agent
 Effort: ~3h
 Risk: Medium-high (changes budget allocation)

 What exists:
 - BudgetPacer with OperationPriority — exists but not used in reconcile filtering
 - Cycle interval throttling wired in quote_engine.rs
 - CumulativeQuotaState with exponential backoff

 Changes:
 1. Quota-scaled tolerance: Before priority_based_matching(), multiply latch by a quota factor: 1.0x at >=30% headroom, ramping to 3.0x at <10%. Low quota → wider tolerance → fewer
 mutations → less quota consumed (virtuous cycle instead of vicious cycle).
 2. Priority-filtered execution: After actions are partitioned, filter by OperationPriority. When quota is tight, skip LowValue (small price drifts), execute only HighValue (coverage
 gaps) and Emergency (risk-driven).
 3. Per-operation budget allocation: 40% modify, 40% place, 20% cancel. Modifies get priority because they're most quota-efficient.

 Impact: Prevents the death spiral where low quota → more expensive operations → even lower quota.

 ---
 Layer 6: Observability and Churn Diagnostics

 Engineer: analytics agent
 Effort: ~2h
 Risk: Zero (read-only metrics)

 Changes:
 1. Churn-specific counters: record_modify_debounced(), record_latch_preserved(), record_grid_preserved(), record_smoother_suppression() — tells you WHERE in the 6-layer stack churn
 was prevented.
 2. Rolling ChurnTracker: Window of 100 cycles tracking cancel/fill/modify/place/skip ratios. Key metrics:
   - cancel_fill_ratio — target <2.0 (currently: infinity)
   - api_calls_per_fill — target <10 (currently: infinity)
   - skip_rate — target >80% (currently: 0%)
 3. Cycle summary log: End-of-reconcile log line with cancels/modifies/places/skips/latched counts.
 4. Churn spiral alert: When cancel_fill_ratio > 10 for 50+ cycles, log warning and optionally boost smoothing deadband.

 ---
 Implementation Sequence

 Deploy in dependency order (each layer enables the next):

 ┌───────┬───────────────────┬───────────────┬────────────────────────────────────────────────────┬─────────────────────────┐
 │ Phase │       Layer       │     Agent     │                    Key File(s)                     │        Gated By         │
 ├───────┼───────────────────┼───────────────┼────────────────────────────────────────────────────┼─────────────────────────┤
 │ 1     │ L1: Smoothing     │ signals       │ params/smoothing.rs, quote_engine.rs, mod.rs       │ SmootherConfig.enabled  │
 ├───────┼───────────────────┼───────────────┼────────────────────────────────────────────────────┼─────────────────────────┤
 │ 2     │ L6: Observability │ analytics     │ infra/metrics/, new ChurnTracker                   │ Always on               │
 ├───────┼───────────────────┼───────────────┼────────────────────────────────────────────────────┼─────────────────────────┤
 │ 3     │ L2: Price Grid    │ strategy      │ quoting/price_grid.rs, quote_engine.rs             │ PriceGridConfig.enabled │
 ├───────┼───────────────────┼───────────────┼────────────────────────────────────────────────────┼─────────────────────────┤
 │ 4     │ L4: Tolerance     │ strategy/lead │ tracking/order_manager/reconcile.rs                │ Latch floor config      │
 ├───────┼───────────────────┼───────────────┼────────────────────────────────────────────────────┼─────────────────────────┤
 │ 5     │ L3: Modify-First  │ infra         │ rate_limit/proactive.rs, orchestrator/reconcile.rs │ min_modify_interval_ms  │
 ├───────┼───────────────────┼───────────────┼────────────────────────────────────────────────────┼─────────────────────────┤
 │ 6     │ L5: Quota-Aware   │ infra         │ orchestrator/reconcile.rs, rate_limit/proactive.rs │ Headroom thresholds     │
 └───────┴───────────────────┴───────────────┴────────────────────────────────────────────────────┴─────────────────────────┘

 Note: Phases 1-2 can be done in parallel. Phases 3-4 can be done in parallel. Phases 5-6 depend on earlier phases.

 ---
 Expected Impact

 ┌──────────────────────────┬──────────┬───────────────────────────┐
 │          Metric          │  Before  │           After           │
 ├──────────────────────────┼──────────┼───────────────────────────┤
 │ Cancels per cycle        │ 2        │ 0 (most cycles)           │
 ├──────────────────────────┼──────────┼───────────────────────────┤
 │ API calls per min        │ 40       │ 2-5                       │
 ├──────────────────────────┼──────────┼───────────────────────────┤
 │ Modify vs cancel ratio   │ 0%       │ >80% (when action needed) │
 ├──────────────────────────┼──────────┼───────────────────────────┤
 │ Queue position preserved │ 0%       │ >80%                      │
 ├──────────────────────────┼──────────┼───────────────────────────┤
 │ Cancel/fill ratio        │ ∞ (58/0) │ <2.0                      │
 ├──────────────────────────┼──────────┼───────────────────────────┤
 │ Cycles with zero actions │ 0%       │ >80%                      │
 └──────────────────────────┴──────────┴───────────────────────────┘

 ---
 Verification

 1. Unit tests: Each layer has existing or new tests (smoother: 19, grid: 11+, reconcile: extensive)
 2. Clippy: cargo clippy -- -D warnings after all changes
 3. Full suite: cargo test — must not regress existing 2,827 passing tests
 4. Paper trading: Run with all layers enabled, compare churn metrics to this baseline log
 5. Live canary: Deploy with SmootherConfig.enabled: true and PriceGridConfig.enabled: true first. Monitor cancel/fill ratio. Then progressively enable remaining layers.

 ---
 Critical Files Summary

 ┌──────────────────────────────────────────────────────┬────────────────┬──────────────────────────────────────────────────────────┐
 │                         File                         │     Layers     │                         Changes                          │
 ├──────────────────────────────────────────────────────┼────────────────┼──────────────────────────────────────────────────────────┤
 │ src/market_maker/strategy/params/smoothing.rs        │ L1             │ Already complete — just wire                             │
 ├──────────────────────────────────────────────────────┼────────────────┼──────────────────────────────────────────────────────────┤
 │ src/market_maker/mod.rs                              │ L1             │ Add parameter_smoother field                             │
 ├──────────────────────────────────────────────────────┼────────────────┼──────────────────────────────────────────────────────────┤
 │ src/market_maker/orchestrator/quote_engine.rs        │ L1, L2         │ Call smoother, directional grid snap                     │
 ├──────────────────────────────────────────────────────┼────────────────┼──────────────────────────────────────────────────────────┤
 │ src/market_maker/quoting/price_grid.rs               │ L2             │ Add snap_bid/snap_ask, epoch                             │
 ├──────────────────────────────────────────────────────┼────────────────┼──────────────────────────────────────────────────────────┤
 │ src/market_maker/infra/rate_limit/proactive.rs       │ L3, L5         │ Reduce debounce, budget allocation                       │
 ├──────────────────────────────────────────────────────┼────────────────┼──────────────────────────────────────────────────────────┤
 │ src/market_maker/orchestrator/reconcile.rs           │ L3, L4, L5, L6 │ Latch on debounce, quota-scaled tolerance, churn summary │
 ├──────────────────────────────────────────────────────┼────────────────┼──────────────────────────────────────────────────────────┤
 │ src/market_maker/tracking/order_manager/reconcile.rs │ L3, L4         │ Modify for price changes, queue-weighted latch           │
 ├──────────────────────────────────────────────────────┼────────────────┼──────────────────────────────────────────────────────────┤
 │ src/market_maker/infra/metrics/                      │ L6             │ Churn counters                                           │
 └──────────────────────────────────────────────────────┴────────────────┴──────────────────────────────────────────────────────────┘