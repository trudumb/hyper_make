# Code Reviewer Memory

## Phase 6 Parameter Calibration Audit (2026-02-09)

**Verdict**: PASS with CONCERNS — Go/No-Go: **GO** (with monitoring)

Full report: [audit-report-2026-02-09.md](audit-report-2026-02-09.md)

### Executive Findings
1. **Kappa Estimation** ✅ PASS — Three-way confidence-weighted blending (book, robust Student-t, own-fill) with proper Gamma-Exponential conjugacy. Floor applied at all GLFT formula sites (15 instances of `kappa.max(1.0)`). Warmup uses observation count not confidence (prevents false graduation). EWMA smoothing (α=0.9) reduces variance.
2. **Sigma Adaptation** ⚠️ CONCERN — Particle filter lags on discontinuous regime changes. No visible max sigma cap (could freeze spreads during cascades). Warmup timeout fallback is safe.
3. **Data Staleness** ✅ PASS — `should_gate_quotes()` checks no-data/stale/crossed. Active defense cancels existing orders when gated. 30s threshold. Book kappa confidence decays naturally when L2 stale.
4. **Position Reconciliation** ⚠️ CONCERN — Event-driven (10s timer + 2s min interval). Exchange is authoritative. Potential 200ms race window during high fill rates. smart_reconcile MODIFY fallback path not traced.
5. **Rate Limits** ✅ PASS — Proactive tracking at 80% threshold. IP: 1200/min, Address: $1/fill + 10K buffer. Rejection backoff: 3 rejections → 5s exponential. Emergency reserve (20 tokens) protects kill switch.
6. **Numerical Stability** ✅ PASS — 80+ NaN guards. Empty book triggers gate. Known u64 overflow fixed (saturating_sub).

### Most Likely Failure Mode
**Kappa stagnation** during low-fill illiquid markets → spreads too tight/wide for regime → adverse selection → reactive widening after damage done. Mitigated by robust orchestrator but still vulnerable.

### Critical Recommendations
- Add `sigma.min(config.max_sigma)` cap (e.g., 10x default)
- Trace smart_reconcile MODIFY→PLACE fallback
- Add cascade-mode sigma override on OI drop
- Monitor kappa confidence (<50% after 1hr = alert)

## Phase 6 Stochastic Modeling Audit (2026-02-09 earlier)

### Key Findings
- Normal-Gamma conjugate updates in rl_agent.rs are mathematically correct (Murphy 2007)
- sample_uniform() uses unsafe static mutable LCG -- data race risk, poor quality for Thompson sampling
- import_q_table_as_prior() REPLACES live Q-values (n=0), destroying real-world learning on hot-reload
- SimToRealConfig (min_real_fills, action_bound_sigma, auto_disable) is defined but NOT wired into select_action_idx()
- BOCD warmup gating uses hybrid entropy + hard cap -- sound design
- Legacy changepoint_detected() ignores regime-aware thresholds (uses config.threshold instead)
- HJB solver correctly floors kappa at 1.0 and gamma at 0.01 to prevent blow-ups
- TD target fed into Bayesian posterior update is a known approximation (BDQN), not strictly correct but practical

### Patterns to Watch
- "Config struct defined but not enforced" -- SimToRealConfig is an example. Always trace from config definition to actual usage in hot paths.
- "Replace vs blend" -- any import/reload that uses `=` instead of blending with existing state deserves scrutiny
- EWMA update-before-compute bug pattern (seen in BuyPressure, PreFillToxicity) -- always check ordering

### Files Reviewed
- `/src/market_maker/learning/rl_agent.rs` -- Q-learning, Thompson sampling, sim-to-real
- `/src/market_maker/control/changepoint.rs` -- BOCD with regime-aware thresholds
- `/src/market_maker/stochastic/hjb_solver.rs` -- HJB optimal quotes
- `/src/market_maker/stochastic/beliefs.rs` -- MarketBeliefs, regime_blend
- `/src/market_maker/control/mod.rs` -- assess_learning_trust
- `/src/market_maker/mod.rs` -- hot-reload mechanism

## Phase 8 RL Strategy Audit (2026-02-09)

### CRITICAL: RL Agent is Observe-Only (Not Controlling Live Quotes)

**Root Cause**: RL agent generates `gamma_multiplier` and `omega_multiplier` via ParameterAction (lines 1807-1808 in rl_agent.rs), which are populated into `MarketParams.rl_gamma_multiplier` and `rl_omega_multiplier` (quote_engine.rs:1669-1672), but **NO DOWNSTREAM CODE CONSUMES THESE FIELDS**. The GLFT strategy's `effective_gamma()` method (glft.rs:137-220) never reads `rl_gamma_multiplier` or `rl_omega_multiplier`. The only consumption is legacy BPS-delta mode (rl_spread_delta_bps, rl_bid_skew_bps, rl_ask_skew_bps), which is a simple additive adjustment, not parameter-based control.

**Evidence**:
- grep shows rl_gamma_multiplier/rl_omega_multiplier are ONLY initialized to 1.0 (market_params.rs:927-928, aggregator.rs:291-292)
- effective_gamma() applies calibration_gamma_mult and tail_risk_multiplier but never checks rl_gamma_multiplier
- Half-spread calculation (glft.rs:643-644) uses raw gamma, no RL scaling
- Position manager skew (not shown in excerpt) would need to consume rl_omega_multiplier but doesn't

**Impact**: Agent is learning Q-values from fills (handlers.rs:632-640 shows update loop) but NOT driving decisions. Exploration is occurring, rewards are computed, but the policy outputs are ignored for parameter-based actions. This means the RL agent is effectively in perpetual "observation mode" even when rl_enabled=true and past min_real_fills threshold.

**Fix Required**: Wire rl_gamma_multiplier and rl_omega_multiplier into:
1. GLFT gamma calculation (after calibration_gamma_mult, before tail_risk_multiplier)
2. Position manager skew calculation (scale base skew by rl_omega_multiplier)

### Sim-to-Real Transfer Bug

**Issue**: `import_q_table_as_prior()` (rl_agent.rs:1571-1587) calls `BayesianQValue::with_discounted_prior()` which sets `n=0` (line 1124), **erasing real-world observation counts**. On hot-reload from paper trading, this REPLACES all live Q-values instead of blending.

**Expected**: Live Q(s,a) with n=50 observations should BLEND with paper Q(s,a) weighted at 30%, preserving live experience.

**Actual**: Live Q(s,a) is replaced by paper Q(s,a) with n=0, so next live fill restarts learning from scratch.

**Fix**: Change import_q_table_as_prior to only update states where live_values[i].count() == 0 (cold start), or use a weighted blend that preserves n.

### MDP State Completeness

**PASS**: MDP state includes adverse selection (line 1650), regime (vol_ratio line 1649), inventory, OBI, Hawkes excitation. This is sufficient for learning from adversarial fills.

**CONCERN**: Reward function (rl_agent.rs:964-996) includes inventory penalty and edge component, but edge is computed as `depth_bps - fee_bps` (handlers.rs:612) WITHOUT subtracting realized AS. This means adverse fills still show positive reward if depth > fee, even when AS wipes out the edge.

**Fix**: handlers.rs:612 should be `depth_bps - realized_as_bps - fee_bps` to match the reward docstring claim that "realized_edge_bps should be spread_capture - AS_cost - fees".

### RL Safety Mechanisms

**CONCERN**: SimToRealConfig defines `min_real_fills`, `action_bound_sigma`, `auto_disable_after_fills` (rl_agent.rs:1228-1251) but **none of these are checked in select_action_idx()** (lines 1320-1379). The action bounding and auto-disable logic exists in quote_engine.rs:1681-1716, but that's just for the legacy BPS-delta mode, not parameter actions.

**Fix**: Add guards in select_action_idx() or in quote_engine before applying gamma/omega multipliers to enforce sim-to-real safety.

### Files Reviewed
- `/src/market_maker/learning/rl_agent.rs` -- RL agent implementation
- `/src/market_maker/orchestrator/quote_engine.rs` -- RL integration point
- `/src/market_maker/orchestrator/handlers.rs` -- RL reward computation and updates
- `/src/market_maker/strategy/glft.rs` -- Quote generation (where RL should apply but doesn't)
- `/src/market_maker/strategy/market_params.rs` -- MarketParams field definitions
