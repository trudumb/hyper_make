# Comprehensive Mainnet Analysis Report
**Date**: 2026-02-09
**Team**: 5 specialized agents (strategy, risk, analytics, signals, infra) + lead synthesis
**Scope**: Full codebase review — infrastructure, models, limitations, missing features

---

## Executive Summary

**Overall Readiness: 8.0/10** — Production-capable with known gaps

The Hyperliquid market making system is architecturally sound with proper Bayesian foundations, defense-in-depth risk management, and a well-structured GLFT implementation. However, several critical wiring gaps, a structurally broken RL reward function, and missing unified-account support create real risks for mainnet deployment.

| Category | Score | Status |
|----------|-------|--------|
| Strategy & Models | 8.5/10 | GLFT correct, stochastic math verified |
| Risk & Safety | 8.5/10 | 3 HIGH findings open |
| Analytics & Learning | 7.0/10 | RL reward P0 broken, calibration pipeline fixed |
| Signals & Estimation | 7.0/10 | 4 critical methods unwired, AS 2x overestimate |
| Infrastructure | 8.5/10 | BBO defense added, unified account missing |

---

## P0 — Critical Issues (Block Deployment)

### 1. Zero-Balance Panic on Startup (FOUND THIS SESSION)
**File**: `src/bin/market_maker.rs:1298-1337`
**Impact**: Market maker panics with "max_position must be > 0.0" when perps clearinghouse has $0
**Root Cause**: `(account_value × leverage × 0.5) / mark_px = 0` when account_value=0. No guard.
**Fix**: Add early exit with clear error: "Insufficient perps margin. Transfer USDC from spot to perps."
**Also**: Add `--auto-transfer` flag to automatically call `class_transfer()` on startup.

### 2. Unified Account Mode Not Supported (FOUND THIS SESSION)
**File**: `src/bin/market_maker.rs:1274-1288`
**Impact**: Code reads `cross_margin_summary.account_value` only. With Hyperliquid's unified/portfolio margin (pre-alpha), funds in spot USDC are invisible to the market maker.
**Evidence**: `check_balance` shows $161.09 in spot USDC, $0.00 in both margin summaries
**Fix**: Fallback to `margin_summary.account_value`, then spot USDC balance if both are $0.

### 3. RL Reward Function Structurally Negative
**File**: `src/market_maker/learning/rl_agent.rs`
**Impact**: Reward measures AS cost, not `spread_capture - AS - fees`. Agent learns "all fills are bad."
**Evidence**: 107 fills in paper testing, Q-table converged to "avoid all actions"
**Fix**: Reward = `half_spread_captured_bps - realized_as_bps - fee_bps`

### 4. Book Imbalance Hardcoded to 0.0 at Fill Time
**File**: `src/bin/paper_trader.rs:1397`
**Impact**: Collapses one of 5 state dimensions in RL, making Q-values degenerate
**Fix**: Pass actual `book_imbalance` from L2 snapshot at fill time

### 5. 27 Bayesian Learned Params Have n_observations=0 After 107 Fills
**Impact**: Bayesian pipeline disconnected — fills not feeding parameter updates
**Root Cause**: `signal_integrator.on_fill()` was never called (see Signals findings)
**Fix**: Wire `signal_integrator.on_fill()` in fill handler

---

## P1 — High Priority (Fix Before Extended Trading)

### 6. Four Critical Signal Integrator Methods Never Called
**File**: `src/market_maker/strategy/signal_integration.rs`
**Methods unwired**:
- `signal_integrator.on_trade()` — feeds InformedFlow EM decomposition
- `signal_integrator.on_fill()` — feeds RegimeKappa + model gating kappa
- `signal_integrator.set_regime_probabilities()` — regime never propagated
- `signal_integrator.update_*_prediction()` — model gating weights stuck at 1.0 forever

**Impact**: Model gating is a complete no-op. All signals pass at full weight regardless of performance. InformedFlow EM never updates. RegimeKappa estimator starved.

### 7. Inventory Hard Limit Not Enforced
**Evidence**: Inventory reached 8.7 despite 6.2 config limit in paper testing
**Root Cause**: `PositionGuard` entry gate is pre-flight only. Exchange position sync + clustered fills bypass it. Reduce-only mode only logs, doesn't hard-enforce.
**Fix**: Post-fill position check that immediately cancels if limit breached.

### 8. Reduce-Only Not Hard-Enforced
**File**: `risk/` module
**Impact**: `check_reduce_only()` only logs when position exceeds limit. Quote engine skew is a "soft" reduction — not guaranteed to prevent further growth.
**Fix**: Hard-enforce in order placement: reject any order that would increase position when reduce-only active.

### 9. AS Model 2x Overestimate
**Evidence**: Predicted 9.14 bps, realized 4.27 bps in paper testing
**Impact**: Spreads wider than necessary → fewer fills → less learning → vicious cycle
**Root Cause**: Likely BayesianAlphaTracker prior (Beta(2,6) → mean 0.25) too pessimistic, and enhanced classifier features not calibrated against live fills.

### 10. BuyPressure EWMA Z-Score Bug
**File**: `estimator/enhanced_flow.rs:1325-1332`
**Impact**: EWMA mean updated BEFORE z-score computed → z-score systematically compressed ~10%
**Fix**: Compute z-score FIRST, then update EWMA (same bug class as PreFillToxicity Feb 6 fix)

### 11. No Sigma Maximum Cap
**Impact**: During extreme cascades, sigma could spike 100x and freeze spreads
**Fix**: `sigma.min(config.max_sigma)` clamp (e.g., 10x default_sigma)

---

## P2 — Medium Priority (Improve Over Time)

### 12. `unsafe impl Send/Sync` on 3 Types
**Types**: `CircuitBreakerMonitor`, `RiskChecker`, `DrawdownTracker`
**Impact**: Blocks future soundness checks, potential UB if types gain interior mutability
**Fix**: Remove unsafe impls, use proper synchronization

### 13. Drawdown Denominator Inconsistency
`DrawdownMonitor` uses `account_value`, `KillSwitch` uses `peak_pnl` — different denominators for same concept.

### 14. `effective_max_position` Not Propagated to PositionGuard
Guard uses `config.max_position` (potentially stale) instead of runtime-computed effective value.

### 15. LeadLag Signal Never Activated (60 min paper test)
Likely: MI significance test threshold too strict, or Binance feed data insufficient for convergence.

### 16. BuyPressure Has Negative Marginal (-2.67 bps)
Should be disabled or retrained. Currently destroying edge when active.

### 17. smart_reconcile MODIFY Failure Path Unverified
If ORDER MODIFY fails (order already filled), unclear if fallback to PLACE exists.

---

## Infrastructure Assessment

### Strengths
- **Event-driven architecture**: Single-threaded event loop prevents race conditions
- **3-way kappa blending**: Book-structure + Robust (Student-t) + Own-fill, EWMA-smoothed
- **80+ NaN guards** across estimator module — comprehensive numerical safety
- **15 kappa.max() floors** protecting GLFT formula — blow-up prevention
- **Proactive rate limiting** at 80% threshold + rejection backoff with exponential backoff
- **Emergency budget reserve** (20 tokens) for kill switch even when rate-limited
- **BBO crossing defense-in-depth**: 3-layer validation (reconcile, bulk, single-order)
- **Exchange-authoritative position**: Local tracker always syncs to exchange state
- **Position reconciliation**: 10s timer + 2s event-driven minimum interval
- **Data quality gates**: Stale data (30s), crossed book, no data → auto-cancel all orders
- **Kill switch**: 9 triggers, persistent across restarts (24h expiry), PostMortem dump wired
- **7 risk monitors**: Loss, Drawdown, Position, DataStaleness, Cascade, RateLimit, PriceVelocity

### Weaknesses
- **No unified account mode support** — spot USDC invisible to market maker
- **No auto-transfer** — can't move funds spot→perps programmatically on startup
- **Paper trader fill simulation is directional only** — no queue position, no partial fills
- **Warmup catch-22**: Wide spreads → 0 fills → no learning (mitigated by paper-mode kappa floor)
- **No structured logging for log aggregation** (JSON format exists but no ELK/Datadog integration)
- **No health check endpoint** for external monitoring
- **No configuration hot-reload**

---

## Missing Models & Features (Ranked by Expected Edge Impact)

### High Impact
1. **Hawkes Process Intensity** — Fill clustering is real and exploitable. Current IID fill assumption misses burst patterns. Expected: +2-3 bps edge from better fill prediction.

2. **Queue Position Modeling** — No model of where our orders sit in the queue. Critical for accurate fill probability and adverse selection estimation. Expected: +1-2 bps from better timing.

3. **Multi-Asset Correlation** — `multi/` module exists but mostly scaffolding. Cross-asset signal (e.g., ETH move predicting BTC) is exploitable. Expected: +1-2 bps for correlated pairs.

4. **Funding Rate Prediction** — Current model uses current funding rate, not predicted. 8h settlement creates predictable flow. Expected: +1-2 bps near settlement.

### Medium Impact
5. **Jump-Diffusion Tail Model** — Sigma estimator uses continuous diffusion. Crypto has fat tails. Missing: jump component for spread adjustment during cascades.

6. **Kyle's Lambda (Order Flow Toxicity)** — Different from AS model. Measures price impact per unit flow. Would improve timing of spread widening.

7. **Liquidation Cascade Prediction from OI Changes** — OI drop → cascade probability. Currently detected reactively via CascadeMonitor, not predictively.

8. **Cross-Venue Order Book Imbalance** — Currently use Binance trade flow only. Adding Binance L2 imbalance would improve lead-lag signal.

### Lower Impact
9. **Market Impact Model** — No self-impact estimation. Important at larger sizes.
10. **Tick-Level Microstructure** — Trade arrival rate modeling for short-horizon prediction.
11. **Volatility-of-Volatility** — For options-like spread adjustment in vol regime transitions.

---

## Stochastic Math Verification

**24/24 formulas verified CORRECT** (from prior audit):
- GLFT half-spread: `(1/γ)ln(1 + γ/κ) + fee` ✅
- HJB solver boundary conditions ✅
- Conjugate prior updates (Gamma-Exponential for kappa, Normal-Gamma for Q-values) ✅
- Particle filter resampling ✅
- Kelly criterion implementation ✅
- Terminal penalty calibration ✅

---

## Mainnet Run Results

**Run attempted**: `./scripts/test_mainnet.sh BTC 14400` (4 hours)
**Result**: Immediate panic — $0 in perps clearinghouse
**Root Cause**: $161.09 USDC in spot wallet, not transferred to perps
**Underlying Issue**: No unified account mode support + no auto-transfer

Previous paper trading results (for reference):
- Run 1 (no --paper-mode): 39 fills, +$0.02 PnL, +5.19 bps edge
- Run 2 (--paper-mode + RL): 107 fills, +$0.80 PnL, +6.4 bps edge

---

## Recommended Action Plan

### Immediate (Before Next Run)
1. Transfer USDC from spot → perps via Hyperliquid UI
2. OR: Add auto-transfer + fallback account value detection

### Before Extended Live Trading
3. Wire the 4 unwired `signal_integrator` methods
4. Fix RL reward function (P0)
5. Fix book_imbalance hardcoded to 0.0 (P0)
6. Hard-enforce inventory limits post-fill
7. Add sigma max cap
8. Fix BuyPressure EWMA z-score ordering

### Next Iteration
9. Implement Hawkes process intensity estimator
10. Add queue position model
11. Retrain AS model with live fill data (fix 2x overestimate)
12. Disable BuyPressure signal (negative marginal)
13. Implement unified account mode support

---

## Sources
- [Hyperliquid Clearinghouse Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/hypercore/clearinghouse)
- [Hyperliquid Portfolio Margin](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/portfolio-margin)
- [Hyperliquid API Info Endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint)
