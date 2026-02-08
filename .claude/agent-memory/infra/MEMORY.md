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

## Serena Tool Notes
- `find_symbol` requires `name_path_pattern` not `name_path` parameter
- Paper trader methods not always found by Serena's symbol tools -- use Grep as fallback
