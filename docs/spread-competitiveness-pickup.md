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

---

# Experience Replay & Offline Learning: Pickup Notes (2026-03-07)

## What's DONE (all 5 stages implemented, compiles, clippy+fmt clean, 34 tests pass)

### Stage 1: Experience Logging ✅
- `ExperienceRecord` extended with 6 optional fields (`drift_penalty`, `bandit_multiplier`, `vol_ratio`, `mdp_state_idx`, `bandit_arm_idx`, `inventory_risk_at_fill`)
- `from_markout()` constructor + `MarkoutExperienceParams` struct
- `PendingFillOutcome` extended with fill-time snapshots (MDP state, bandit arm, inv risk, vol ratio)
- Fill-time snapshot in `handlers.rs` ~L1425, markout-time logging ~L665
- Config: `enable_experience_logging`, `experience_dir`, in `stochastic.rs`
- Paper validated: 12+ fills/10min, valid SARSA records in `logs/experience/*.jsonl`

### Stage 2: Replay Buffer ✅
- `src/market_maker/learning/replay_buffer.rs` — `ReplayBuffer`, `ReplayStatistics`
- Load from dir, push with capacity eviction, deterministic LCG sampling, statistics
- 7 unit tests

### Stage 3: Fitted Q-Iteration ✅
- `src/market_maker/learning/fqi.rs` — `FittedQIterator`, `FQIConfig`, `FQIResult`, `FQICheckpoint`
- Double Q-learning, 45 states × 8 actions = 360 cells
- `FQIPolicyRecommendation` with arm→delta_bps mapping
- Checkpoint persistence via `FQICheckpoint` (to/from `FQIResult`)
- 7 unit tests

### Stage 4: Policy Feedback ✅
- `quote_engine.rs` ~L2913: FQI blend into `rl_spread_delta_bps`
- Gated by `fqi_blend_weight > 0.0`, `fqi_result.is_some()`, `is_warmed_up()`
- Config: `fqi_blend_weight` (default 0.0 = disabled), `fqi_min_fills`, `fqi_refit_interval`

### Stage 5: Counterfactual Analysis ✅
- `src/market_maker/learning/counterfactual.rs` — per-fill regret, aggregate report
- 4 unit tests

### Remaining Items (implemented but NOT yet validated in paper)

1. **Checkpoint save/restore for FQI** ✅ DONE
   - `assemble_checkpoint_bundle()` now saves `fqi_result` → `fqi_q_table`
   - `restore_from_bundle()` now restores `fqi_q_table` → `fqi_result`

2. **Periodic FQI refit** ✅ DONE (code exists, BUG blocks paper validation)
   - `handlers.rs` PHASE 7: increments `fqi_fills_since_refit`, triggers refit at interval
   - Loads from `experience_dir`, fits FQI, stores result

3. **Counterfactual JSONL on shutdown** ✅ DONE (code exists, blocked by same bug)
   - `recovery.rs` shutdown: if `fqi_result` exists, runs counterfactual analysis, writes JSON

4. **`with_experience_logging()` syncs `experience_dir`** ✅ DONE
   - Previously `with_experience_logging("logs/experience")` didn't update `stochastic_config.experience_dir`, so FQI refit would look in wrong dir

## CRITICAL BUG: TOML `[stochastic]` not wired into binary

**Root cause found**: `src/bin/market_maker.rs` line 1652:
```rust
let stochastic_config = StochasticConfig {
    enable_position_ramp: false,
    enable_performance_gating: false,
    ..Default::default()  // ← IGNORES ALL TOML [stochastic] VALUES
};
```

The `AppConfig` struct (line 413) has `pub stochastic: StochasticConfig` and the TOML parses it correctly. But when building `MmConfig`, the binary constructs a fresh `StochasticConfig` with defaults instead of using `config.stochastic`.

**Fix needed** (5 lines):
```rust
let stochastic_config = StochasticConfig {
    enable_position_ramp: false,
    enable_performance_gating: false,
    ..config.stochastic  // ← Use TOML values as base
};
```

This affects ALL `[stochastic]` TOML fields, not just FQI. The paper binary has the same issue at line ~2778.

**After fixing**: `enable_experience_logging = true` and `fqi_refit_interval = 1` from TOML will take effect, the FQI refit will fire, counterfactual report will be written on shutdown.

## TOML Config (current, in `market_maker_live.toml`)
```toml
[stochastic]
enable_experience_logging = true
fqi_refit_interval = 1    # Set low for validation, raise to 200 after
fqi_min_fills = 3          # Set low for validation, raise to 100 after
```

## Paper Trading Validation Results (2026-03-07)

### Run 1 (10 min, before FQI refit interval lowered)
- 12 fills, valid experience records in `logs/experience/`
- Experience logging works end-to-end
- FQI refit did NOT trigger (interval=200, only 12 fills)

### Run 2-3 (5-10 min, with low refit interval)
- FQI refit did NOT trigger — **because `enable_experience_logging` defaults to false**
- Bug identified: TOML `[stochastic]` section completely ignored by binary

## Next Steps (in order)

1. **Fix the TOML→StochasticConfig wiring** in `src/bin/market_maker.rs`:
   - Line 1652: change `..Default::default()` → `..config.stochastic`
   - Line ~2778 (paper mode): same fix
   - This unblocks ALL stochastic TOML config, not just FQI

2. **Paper validate FQI pipeline**:
   - Run 10 min with `fqi_refit_interval = 5`, `fqi_min_fills = 5`
   - Expect: "FQI refit triggered" + "FQI refit completed" in logs
   - Expect: `data/analytics/counterfactual_report.json` written on shutdown
   - Expect: checkpoint contains `fqi_q_table` with non-empty Q-values

3. **Restore production thresholds**:
   - `fqi_refit_interval = 200`
   - `fqi_min_fills = 100`
   - `fqi_blend_weight = 0.0` (keep disabled until counterfactual shows improvement)

4. **Remove diagnostic logs**:
   - `handlers.rs`: revert "FQI refit triggered" info log to debug
   - `handlers.rs`: revert "FQI refit skipped" back to debug

5. **Commit** all changes with conventional commit format

## Files Modified This Session (2026-03-07)

| File | Change |
|------|--------|
| `src/market_maker/mod.rs` | FQI checkpoint save (assemble) + restore, `with_experience_logging` syncs `experience_dir`, removed `#[allow(dead_code)]` on `fqi_fills_since_refit` |
| `src/market_maker/orchestrator/handlers.rs` | PHASE 7: periodic FQI refit logic after markout |
| `src/market_maker/orchestrator/recovery.rs` | Counterfactual report on shutdown |
| `market_maker_live.toml` | Added `fqi_refit_interval`, `fqi_min_fills` |

## Key Code Locations

| What | Where |
|------|-------|
| FQI refit trigger | `handlers.rs` ~L697 (PHASE 7 in `check_pending_fill_outcomes`) |
| Counterfactual on shutdown | `recovery.rs` ~L33 (in `shutdown()`) |
| Checkpoint save FQI | `mod.rs` ~L1278 (`assemble_checkpoint_bundle`) |
| Checkpoint restore FQI | `mod.rs` ~L1366 (`restore_from_bundle`) |
| Experience dir sync | `mod.rs` ~L1031 (`with_experience_logging`) |
| TOML stochastic wiring BUG | `src/bin/market_maker.rs` L1652 |
| Paper mode stochastic BUG | `src/bin/market_maker.rs` ~L2778 |
