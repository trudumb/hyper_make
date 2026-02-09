# Infra Agent Memory

## Paper Trader Architecture
- `SimulationState` in `src/bin/paper_trader.rs` (~3100 lines) holds all market making state
- `adaptive_spreads` is a direct field on `SimulationState` (not nested in stochastic like live system)
- Live system: `self.stochastic.adaptive_spreads` (handlers.rs:479)
- Paper system: `self.adaptive_spreads` (paper_trader.rs:1299)
- `on_simulated_fill()` is the paper trader's equivalent of the live fill handler
- `check_adverse_selection_outcomes()` resolves AS after 5s delay using mid price change

## Wiring Gaps Found (2026-02-08)
- `adaptive_spreads.on_fill_simple()` was NOT called from paper trader -- root cause of warmup stuck at 10%
- Fix: Added at paper_trader.rs:1291-1305, mirrors handlers.rs:456-484 exactly
- `reduce_only` handling was already present at paper_trader.rs:2229-2243 (no fix needed)

## Risk System in Paper Trader
- `build_paper_risk_aggregator()` at paper_trader.rs:1567 creates relaxed risk limits
- `PositionMonitor` produces `RiskAction::ReduceOnly` when value_utilization > 1.0
- Risk checks happen before quote generation in the main event loop
- Order: kill check -> pull_quotes -> reduce_only -> pull_buys/pull_sells

## Key API Signatures
- `AdaptiveSpreadCalculator::on_fill_simple(as_realized, fill_distance, pnl, book_kappa)` -- all f64, distances as fractions (not bps)
- `as_realized = (mid - fill_price) * direction / fill_price` where direction = +1 buy, -1 sell
- `depth_from_mid = |fill_price - mid| / mid`
- `fill_pnl = -as_realized` (negative AS = profit)

## BBO Crossing Validation (2026-02-08)
- Production bug: bid at 32.992 crossed BBO ask at 32.992 → "Post only would have immediately matched" rejection
- One side rejected, other placed → one-sided directional exposure (opposite of market making)
- Root cause: no pre-placement BBO validation; mid shift between quote calc and placement
- Fix: 3-layer defense-in-depth BBO validation:
  1. `reconcile_ladder_smart()` — validates ALL quotes against cached BBO, skips BOTH sides if any would cross
  2. `place_bulk_ladder_orders()` — per-order filter against BBO (catches modify fallback paths)
  3. `place_new_order()` — single-order mode BBO guard
- Added `cached_best_bid`, `cached_best_ask`, `last_l2_update_time` fields to MarketMaker struct
- Updated `handle_l2_book()` to populate BBO cache on each L2 update
- Book staleness gate: skip quote cycle if L2 data > 5 seconds old
- Staleness buffer: widen validation margin by 1 tick/second when book age > 2 seconds
- Tick proxy: `mid_price * 0.0001` (1 bps) — conservative for most assets
- Also filters modify specs that would cross BBO before sending to exchange

## Learning Loops Wired in Live (2026-02-09)
- `estimator.on_own_fill()` now called in handlers.rs fill loop (was missing)
- `PendingFillOutcome` struct added to fills/mod.rs for 5s AS markout
- VecDeque field on InfraComponents (core/components.rs) -- avoids editing lead-owned mod.rs
- `check_pending_fill_outcomes()` method in handlers.rs, called from handle_all_mids
- calibration_controller.record_fill() and signal_integrator.on_fill() were already wired
- Key: when adding state to MarketMaker but can't edit mod.rs, put it on a component bundle (InfraComponents, Tier1, etc.)

## Serena Tool Notes
- `find_symbol` requires `name_path_pattern` not `name_path` parameter
- Paper trader methods not always found by Serena's symbol tools -- use Grep as fallback
