# Infra Agent Memory

## Current State
- Dashboard data pipeline: 9 gaps closed (Feb 27)
- Per-asset staleness reconnection for paper mode (Feb 27)
- All learning loops wired in live (estimator.on_own_fill, AS markout, calibration)

## Key Architecture
- Paper: `SimulationState` in `paper_trader.rs`, `adaptive_spreads` is direct field
- Live: `self.stochastic.adaptive_spreads` in handlers.rs
- BBO validation: 3-layer defense (reconcile, bulk_ladder, single_order)
- Quota-aware cycle intervals: >=30% → 1s, 10-30% → 3s, 5-10% → 5s, <5% → 10s

## Active Gotchas
- When adding state to MarketMaker but can't edit mod.rs, put on InfraComponents/Tier1
- `AdaptiveSpreadCalculator::on_fill_simple()` expects fractions, not bps
- Serena `find_symbol` misses paper_trader methods — use Grep as fallback
- `CancelResult::AlreadyFilled` must NOT remove from tracking (fill notification arrives separately)
