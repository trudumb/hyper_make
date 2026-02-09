# Code Reviewer Memory

## Phase 6 Stochastic Modeling Audit (2026-02-09)

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
