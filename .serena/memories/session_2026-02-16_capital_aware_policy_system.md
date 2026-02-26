# Session 2026-02-16: Capital-Aware Policy System + AS Tautology Fix

## What Was Done
Implemented 6-phase "Capital-Aware Market Making: Principled Redesign" plan to fix zero-fill problem
discovered in log `mm_hip3_hyna_ETH_hip3_2026-02-15_20-17-10.log` (888 lines, 3+ min, zero fills).

### Root Causes Fixed
1. **L3 Death Spiral (PRIMARY)**: Stochastic controller used tautologically-negative edge (-0.41 bps)
   to ramp trust 0.20→0.96, killing quoting entirely after 60 seconds.
2. **Reconciler Quota Churn (SECONDARY)**: sqrt(headroom/0.20) scaling at 8.2% headroom truncated
   3 levels to 2, burning 12+ API calls/cycle.
3. **Warmup Death Spiral (TERTIARY)**: No fills → no calibration → 10% warmup forever → wide spreads.

### 6 Phases Implemented

| Phase | Files Changed | What |
|-------|--------------|------|
| 1. Policy Foundation | capacity.rs, config/mod.rs, market_params.rs, aggregator.rs, mod.rs | `CapitalAwarePolicy` struct with tier-specific params (Micro/Small/Medium/Large), wired into `MarketParams` and `MarketMaker` |
| 2. AS Tautology Fix | fills/processor.rs, adverse_selection/estimator.rs | Use `fill.mid_at_placement` instead of `state.latest_mid` in `record_fill()` — fixes structurally negative edge |
| 3. Strategy Layer | ladder_strat.rs | Policy-aware level cap (`max_levels_per_side`), skip entropy optimization for Micro/Small |
| 4. Batch Reconciler | quote_engine.rs | Route to batch reconcile (2 API calls) when `policy.use_batch_reconcile`, with price drift threshold |
| 5. Controller & Quote Gate | quote_engine.rs, quote_gate.rs, reconcile.rs | L3 trust capped at `policy.max_l3_trust_uncalibrated` until fills >= `min_fills_for_trust_ramp`; quota density scaling disableable |
| 6. Integration & Warmup | quote_engine.rs, glft.rs, ladder_strat.rs, capacity.rs, risk/mod.rs | Policy-driven warmup bootstrap (0.40 floor for bootstrap_from_book tiers), fill-based progress from policy target, `warmup_floor_bps` replaces hardcoded 3.0 bps, serde support, EmergencyInput params struct |

### Key Design Decisions
- `CapitalAwarePolicy` is the SINGLE place where tier→behavior mapping lives
- Policy threaded via `MarketParams.capital_policy` (strategy layer) and `MarketMaker.capital_policy` (reconciler)
- `#[serde(default)]` on `CapitalAwarePolicy` for checkpoint backward compatibility
- `EmergencyInput` params struct replaces 8-arg `EmergencyDecision::evaluate()` (clippy too_many_arguments)
- Test helper `eval()` wraps struct construction for test convenience

### CapitalAwarePolicy Tier Values
| Field | Micro | Small | Medium | Large |
|-------|-------|-------|--------|-------|
| max_levels_per_side | 2 | 3 | 5 | 8 |
| skip_entropy_optimization | true | true | false | false |
| use_batch_reconcile | true | true | false | false |
| price_drift_threshold_bps | 3.0 | 2.0 | 1.5 | 1.0 |
| max_l3_trust_uncalibrated | 0.30 | 0.50 | 0.80 | 0.95 |
| always_quote_minimum | true | true | false | false |
| min_fills_for_trust_ramp | 10 | 20 | 30 | 50 |
| warmup_fill_target | 5 | 10 | 25 | 50 |
| bootstrap_from_book | true | true | false | false |
| warmup_floor_bps | 3.0 | 4.0 | 6.0 | 8.0 |
| quota_min_headroom_for_full | 0.05 | 0.10 | 0.15 | 0.20 |
| quota_density_scaling | false | false | true | true |

## Validation
- **Clippy**: Clean (0 warnings)
- **Tests**: 2795 pass, 7 fail (4 pre-existing drawdown+calibration, 3 pre-existing inventory_skew from prior uncommitted work)
- **No new test failures introduced** (was 2759 pass / 4 fail before changes, now 2795 pass / 7 fail)
- Pre-existing inventory_skew test failures due to `inventory_skew_bps = 0.0` hardcoded in signal_integration.rs:1072

## Files Modified (16 total)
- config/capacity.rs, config/mod.rs
- strategy/market_params.rs, strategy/params/aggregator.rs
- strategy/ladder_strat.rs, strategy/glft.rs, strategy/regime_state.rs
- fills/processor.rs, adverse_selection/estimator.rs
- orchestrator/quote_engine.rs, orchestrator/reconcile.rs
- control/quote_gate.rs
- risk/mod.rs
- mod.rs

## Still Open
- **AS tautology partial**: mid_at_placement fix is in fills/processor.rs but EdgeSnapshot integration not yet verified
- **InformedFlow tightening** (P0-6): not addressed this session
- **inventory_skew_bps = 0.0**: hardcoded in signal_integration.rs:1072, 3 tests expect non-zero
- **bootstrap_from_book**: policy field exists but actual L2 book kappa estimation not implemented
- Plan file: `.claude/plans/steady-spinning-peach.md`
