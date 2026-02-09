# Session: HIP-3 Live Profitability Fixes & Parallel Validation
**Date**: 2026-02-09
**Duration**: ~2 hours across 2 context windows

## Objective
Fix HIP-3 HYPE market maker profitability: 57 bps spread, -1.5 bps edge → target ~15 bps spread, +3 bps edge.

## 5 Fixes Implemented (all validated)

### Fix 1: Cold-Start Staleness Distinction
- **File**: `src/market_maker/strategy/signal_integration.rs:949-970`
- **Problem**: Cold-start signals (0 observations) treated as stale → permanent 2.0x multiplier
- **Fix**: Only count signal as stale if `observation_count > 0` (had data, lost it)
- **Result**: Staleness 2.0x → 1.0x on cold start (1.5x when genuine staleness at min 12)

### Fix 2: Ladder Fee Correction
- **File**: `src/market_maker/quoting/ladder/mod.rs:133-137`
- **Problem**: `fees_bps: 3.5` double-counted AS (already modeled by DepthDecayAS)
- **Fix**: Changed to `fees_bps: 1.5` (actual Hyperliquid maker fee)
- **Also fixed**: Stale comment at `ladder_strat.rs:334`

### Fix 3: Ladder Level Deduplication
- **File**: `src/market_maker/quoting/ladder/generator.rs`
- **Problem**: Sub-$0.001 price differences collapse after rounding → 7/8 levels identical
- **Fix**: `dedup_merge_levels()` after sort, merges same-price levels

### Fix 4: Log-Additive Gamma (CalibratedRiskModel)
- **File**: `src/market_maker/strategy/risk_model.rs` + `src/bin/market_maker.rs`
- **Problem**: 7+ multiplicative gamma scalars → 1.2^7 = 3.6x explosion
- **Fix**: `risk_model_blend: 1.0` → pure log-additive CalibratedRiskModel for HIP-3
- **Result**: gamma stable at 0.067 (paper) / 0.312-0.480 (live, higher due to position)

### Fix 5: Volatility-Scaled AS Threshold
- **Files**: `src/market_maker/orchestrator/handlers.rs`, `src/bin/paper_trader.rs`
- **Problem**: Hardcoded 1 bps threshold → BTC 5s random walk ~3.2 bps → 100% adverse rate
- **Fix**: `threshold = max(1.0, 2 × sigma_bps × √markout_s)` → 2-sigma noise filter
- **Result**: AS rate ~25% (was ~100%)

## Parallel Validation Results

### Paper Trader (HYPE) — SUCCESS
| Metric | Value |
|--------|-------|
| Fills | 16 in 28 min (~34/hr) |
| Edge | **+2.79 bps** (significant at 95%) |
| Sharpe | **927** annualized |
| Total spread | ~14.6 bps |
| Gamma | 0.067 (stable) |
| Kappa | 5,275 (graduated warmup) |
| Win rate | 93.8% |

### Live MM (hyna HYPE HIP-3) — QUOTA-CONSTRAINED
| Metric | Value |
|--------|-------|
| Fills | 7 in 27 min |
| Edge | -1.5 bps |
| PnL | +$0.26 |
| Max drawdown | 0.5% |
| API headroom | **7% permanent** |
| Position | -1.57 at shutdown |
| Kappa | 3,671 (graduated, own_fills=11) |

### Root Cause of Live Underperformance
**Exchange API rate limit (7% headroom)**, NOT code bugs. This forces:
1. Inventory-forcing mode (one-sided quoting)
2. Position whipsaw: 0.26 → -0.23 → 1.94 → -1.57
3. Adverse fills from forced direction changes

## Key Discoveries

### Staleness Genuine After 12 Minutes
Live MM had genuine 1.5x staleness from minute 12 onward — some signal went stale during the run.
Need to identify which signal and whether it's expected for HIP-3.

### Cancel-Fill Race Condition
Order oid=316954325257 was cancelled but got filled first → position jumped from 1.06 to 1.94.
System handled it gracefully ("already filled when cancelled") but it amplified position whipsaw.

### Quota-Aware Recovery Logic
`reconcile.rs:894-974` has three tiers:
- headroom < 5%: conservation mode with backoff
- headroom < 20%: minimal levels (2 per side)
- headroom >= 20%: full replenishment

`quote_gate.rs:895-974` inventory-forcing triggers at headroom < 10%.
Epsilon probe requires headroom > 30% (never triggers at 7%).

## Iteration Priorities (Not Yet Implemented)
1. Investigate hyna DEX 7% quota — account tier or DEX-specific limit?
2. Passive quoting mode for low-quota DEXes (reduce frequency, keep two-sided)
3. Seed live warmup from paper checkpoints (kappa 5275 vs priors)
4. Two-sided minimum — always quote both sides to avoid position whipsaw
5. Close residual position (-1.57 HYPE short)

## Architecture Insight
- `signal_integration.rs` central hub for all signal/staleness logic
- `CalibratedRiskModel` (risk_model.rs) is the safe gamma path — log-additive with bounded coefficients
- `risk_model_blend: 1.0` = pure calibrated, `0.0` = pure legacy multiplicative
- Live learning loops now fully wired (handlers.rs parity with paper_trader.rs)

## Files Modified This Session
- `src/market_maker/strategy/signal_integration.rs` — cold-start staleness guards
- `src/market_maker/quoting/ladder/mod.rs` — fees_bps 3.5 → 1.5
- `src/market_maker/quoting/ladder/generator.rs` — dedup_merge_levels()
- `src/market_maker/strategy/risk_model.rs` — RiskModelConfig::hip3()
- `src/bin/market_maker.rs` — wire hip3 risk model config
- `src/market_maker/orchestrator/handlers.rs` — vol-scaled AS threshold
- `src/bin/paper_trader.rs` — vol-scaled AS threshold
- `src/market_maker/strategy/ladder_strat.rs:334` — stale comment fix

## Test Count
2,131+ tests, clippy clean.
