# Session 2026-02-16: Bootstrap from Book Wiring

## What Changed
Wired CalibrationCoordinator + MarketProfile into the MarketMaker runtime, closing the "Bootstrap from Book" paper trading infrastructure gap.

## Files Modified (9 files)
1. **src/market_maker/mod.rs** — Added `market_profile` and `calibration_coordinator` fields to MarketMaker struct; initialized in constructor; checkpoint save now saves actual state (was `default()`); restore now restores coordinator state
2. **src/market_maker/orchestrator/handlers.rs** — handle_l2_book feeds L2 to MarketProfile; seeds coordinator when `bootstrap_from_book && !seeded && profile.initialized`; check_pending_fill_outcomes feeds markout AS to coordinator
3. **src/market_maker/orchestrator/quote_engine.rs** — Added `calibration_coordinator: &self.calibration_coordinator` to ParameterSources
4. **src/market_maker/strategy/params/aggregator.rs** — Added `calibration_coordinator` field to ParameterSources; build() populates coordinator_kappa/uncertainty/use fields
5. **src/market_maker/strategy/market_params.rs** — Added 3 fields: coordinator_kappa, coordinator_uncertainty_premium_bps, use_coordinator_kappa
6. **src/market_maker/strategy/ladder_strat.rs** — Kappa priority chain: Robust > Adaptive > **Coordinator** > Legacy; coordinator uncertainty premium routes through warmup_addon_bps
7. **src/market_maker/calibration/gate.rs** — Added 2 edge case tests
8. **src/market_maker/estimator/calibration_coordinator.rs** — Added 2 regression tests (E2E convergence, checkpoint round-trip)
9. **src/market_maker/learning/rl_agent.rs** — Updated baseline field comment documenting QL agent deprecation

## Key Design Decisions
- **Feed L2 in handle_l2_book**: Coordinator seeded once from MarketProfile when `bootstrap_from_book` policy flag is true (Micro/Small tiers)
- **Feed fills in check_pending_fill_outcomes** (NOT handle_user_fills): Uses markout-based `was_adverse` to avoid AS tautology bug
- **Kappa priority**: Coordinator sits between Adaptive and Legacy — only used when robust/adaptive haven't converged
- **Uncertainty premium**: Routes through warmup_addon_bps in SpreadComposition, decays to 0 as fills accumulate
- **Checkpoint**: CalibrationCoordinatorCheckpoint is a type alias for CalibrationCoordinator itself — just clone for save/restore

## Validation
- Clippy clean (0 warnings)
- 2,896 tests pass, 7 failures all pre-existing (drawdown, inventory_skew, spreads_monotonically_tighten)
- 6 new tests all pass: E2E convergence, checkpoint round-trip, kappa priority chain, robust override, gate edge cases (marginal 3/5, insufficient short session)

## Architecture Notes
- `compute_spread_composition` is diagnostic only (used for decomposition visibility) — uses basic robust/legacy kappa path
- `generate_ladder` is where coordinator kappa is actually selected (the full priority chain)
- Coordinator kappa has built-in 0.5x warmup factor at 0 fills, converges to 1.0x — this naturally widens spreads 2x during cold start
- effective_kappa().max(10.0) hard floor prevents GLFT blowup
