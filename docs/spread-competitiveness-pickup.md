# Spread Competitiveness: Pickup Notes (2026-03-03)

## Problem Statement
System quotes too wide on HIP-3 HYPE. Target: competitive with 3-8 bps total market spread. Observed: 12-18 bps total. User mandate: model-driven, no hardcoded overrides, self-calibrating.

## What Was Done This Session

### Fixes Applied (compiled, tested, in release binary)
1. **AS → kappa reduction (Glosten-Milgrom)** — `ladder_strat.rs` ~L869. Replaces old `solve_min_gamma()` death spiral. `alpha = AS / (AS + spread + fee)`, `kappa_eff = kappa * (1 - alpha)`, cap alpha=0.5.
2. **Regime risk premium double-count removed** — `ladder_strat.rs` ~L651. `total_risk_premium_bps` already includes `regime_risk_premium_bps`; was added twice.
3. **AS floor HWM ratchet removed** — `quote_engine.rs` ~L1137. Permanent ratchet replaced with direct estimator value.
4. **Funding rate wired** — `quote_engine.rs` ~L851. `set_funding_rate_8h()` was never called.
5. **HIP-3 profile floor 7.5→2.0 bps** — `spread_profile.rs`.
6. **sigma_baseline 0.00015→0.00035** — `risk_config.rs`. Realistic HYPE vol prevents excess_volatility feature from inflating gamma.
7. **gamma_min 0.20→0.10** — `risk_config.rs`.
8. **Lead-lag skew cap 3→15 bps** — `signal_integration.rs` (both HIP-3 and liquid).
9. **Binance active requotes** — `handlers.rs` + `event_loop.rs`. Requote when divergence > 2bps.
10. **Stripped Micro tier gamma boost** — `risk_model.rs` `compute_gamma_with_policy()`. Removed 1.2× hardcoded multiplier. Model decides.
11. **Lowered beta_edge_uncertainty 1.5→0.5** — `risk_model.rs`. Cold-start gamma inflation from 2.12× to 1.28×.
12. **Lowered beta_calibration 0.8→0.3** — `risk_model.rs`.
13. **Cold-start edge_uncertainty 0.5→0.3** — `market_params.rs`.
14. **Cold-start calibration_deficit 0.25→0.0** — `market_params.rs`.
15. **Gamma calibrator min_samples 100→20** — `gamma_calibrator.rs`. Starts influencing after 20 fills.
16. **Gamma calibrator apply interval 50→10 fills** — `handlers.rs` ~L569.

### Files Modified
- `src/market_maker/strategy/ladder_strat.rs` — AS→kappa, regime double-count, E[PnL] filter
- `src/market_maker/strategy/risk_model.rs` — stripped Micro boost, lowered betas, updated tests
- `src/market_maker/strategy/risk_config.rs` — gamma_min, sigma_baseline
- `src/market_maker/strategy/market_params.rs` — cold-start defaults
- `src/market_maker/strategy/signal_integration.rs` — lead-lag skew caps
- `src/market_maker/config/spread_profile.rs` — HIP-3 floor
- `src/market_maker/orchestrator/quote_engine.rs` — funding rate wiring, AS floor HWM
- `src/market_maker/orchestrator/handlers.rs` — Binance requote, gamma calibrator interval
- `src/market_maker/orchestrator/event_loop.rs` — Binance requote wiring
- `src/market_maker/adaptive/gamma_calibrator.rs` — min_samples

### Test Status
- All 438 targeted tests pass (risk_model, gamma, calibration)
- clippy clean, fmt clean
- Release binary built at 17:29

## Unsolved: The 3× Gamma Mystery

**Observed**: gamma_market=0.387 in paper trading logs.
**Calculated from features**: gamma≈0.127 (all known features accounted for).
**Discrepancy**: 3.05× unexplained.

### Possible causes (investigate next session):
1. **model_uncertainty feature** — `kappa_ci_width/3.0`. During warmup at 61%, CI width might be huge (e.g., 9.0 → model_uncertainty=1.0 → beta_uncertainty=0.2 → contribution=0.2). But that's only 1.22×, not 3×.
2. **Hidden gamma multiplier outside CalibratedRiskModel** — the log says "gamma includes book_depth + warmup scaling" but I stripped those from `compute_gamma_with_policy()`. Check if there's ANOTHER code path applying them.
3. **conviction_gamma_mult** — log shows `conviction_shift=0.000` which suggests mult=1.0. Verify.
4. **gamma_base mismatch** — verify the HIP-3 config actually sets `log_gamma_base = ln(0.15)`. Might be using Default (0.30) instead of hip3() constructor.
5. **The risk_model instance might not be the hip3 one** — check initialization path in `market_maker.rs` / `quote_engine` to ensure HIP-3 CalibratedRiskModel is used.

### How to debug:
Add INFO-level feature decomposition to `compute_gamma()` in `risk_model.rs` (temporarily). Log all 16 `beta × feature` products, raw_sum, regulated_sum, log_gamma_base, final gamma. Then run paper trading and trace the exact inflation path.

## User's Key Directive (Unfinished)

> "why doesnt the system start up, ingest data for a minute or however long it takes to calibrate then switch live?"

The system HAS a warmup phase (estimator waits for sigma/kappa convergence). But:
- Micro tier forces 10s timeout — too short
- After warmup, edge_uncertainty starts at 0.3 regardless of estimation quality
- Gamma calibrator needs 20 fills (was 100) to start adjusting betas

**Proper fix**: observation-only phase (60-120s), then calibrated quoting. After warmup completes, set edge_uncertainty based on actual estimation convergence (e.g., sigma CV < 0.1 → edge_uncertainty=0.15).

## Paper Trading Results (Pre-Mystery Fix)

### Run 1 (sigma_baseline fix only): ~6 min
- 3 fills, avg_edge_bps=2.8, gross_pnl_bps=8.3 — **POSITIVE EDGE**
- gamma=0.367→0.367, kappa=2052, sigma=0.000281
- Spreads: 2.9-6.7 bps in cycle_summary

### Run 2 (all fixes): ~3 min
- 1 fill, avg_edge_bps=2.6
- gamma=0.418-0.463, kappa=1675-2086, sigma=0.000294
- Spreads: 4.1-7.0 bps in cycle_summary

### Key Victory
- gamma went from 100 (death spiral) → 0.37-0.46 (manageable)
- Fills have POSITIVE edge (2.6-2.8 bps avg)
- But gamma is still 3× higher than it should be → fix the mystery multiplier

## Gamma Calibrator Status
**Already wired** (contrary to plan claim):
- `quote_engine.rs:3289` — caches features/gamma each quote cycle
- `handlers.rs:1325` — copies cache to PendingFillOutcome at fill time
- `handlers.rs:558-579` — feeds (features, gamma, realized_edge) to calibrator at markout
- Every 10 fills (was 50), applies learned betas to risk model
- Blends smoothly: at N fills, blend = N/20 (was N/100)

## Invariants
- `kappa > 0.0` always (GLFT formula)
- `gamma > 0.0` always
- `ask_price > bid_price`
- Kill switch never auto-recovers from financial triggers
- All checkpoint fields: `#[serde(default)]`
- No binary side-clearing
