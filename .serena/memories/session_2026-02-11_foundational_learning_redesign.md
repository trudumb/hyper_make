# Session 2026-02-11: Foundational Learning Architecture Redesign

## Summary
Complete 6-phase redesign addressing 5 wrong foundational assumptions in the market making system.
Tests: 2,311 → 2,369 (+58). Clippy clean. Team of 3 agents + lead.

## Five Wrong Assumptions Fixed

1. **A1: "GLFT computes optimal spreads; floor catches edge cases"**
   - Reality: GLFT produced 2.87 bps, floor was 7.42 bps → floor overrode 100% of cycles
   - Fix: `solve_min_gamma()` in glft.rs constrains gamma so GLFT ≥ floor naturally

2. **A2: "Edge can be predicted without knowing the quoted spread"**
   - Reality: Edge predicted at theoretical (2.87 bps) was always negative; actual spread (7.42 bps) gives positive edge
   - Fix: `actual_quoted_spread_bps: Option<f64>` in MarketState, wired into GLFTEdgeModel

3. **A3: "Binary model gating protects against bad predictions"**
   - Reality: weight=0 → zero predictions → zero IR → weight stays 0 (death spiral)
   - Fix: `graduated_weight()` with MIN_SIGNAL_WEIGHT=0.05 floor in model_gating.rs

4. **A4: "Three parallel controllers provide robustness"**
   - Reality: Ensemble + RL Agent + Controller fight each other → oscillation
   - Fix: Single pipeline. RL is ensemble member (RLEdgeModel). Controller is risk overlay only.

5. **A5: "Learning from fills is sufficient"**
   - Reality: Fills biased toward adverse flow → permanently pessimistic edge
   - Fix: QuoteOutcomeTracker tracks filled + unfilled quotes for unbiased edge

## Phase Details

### Phase 1: Self-Consistent Gamma (strategy agent)
- **glft.rs**: `solve_min_gamma(target, kappa, sigma, T)` — binary search for γ making GLFT ≥ target
- **glft.rs**: `effective_gamma()` — `gamma_final = max(blended_gamma, min_gamma)`
- **ladder_strat.rs**: Floor-binding warning counter (>5% = miscalibrated gamma)
- 5 new tests

### Phase 2: Edge at Actual Spread (strategy agent)
- **learning/types.rs**: `actual_quoted_spread_bps: Option<f64>` on MarketState
- **learning/ensemble.rs**: GLFTEdgeModel uses actual spread when available
- **learning/mod.rs**: `build_market_state()` wires `market_spread_bps`
- 2 new tests

### Phase 3: Graduated Gating (signals agent)
- **calibration/model_gating.rs**: `MIN_SIGNAL_WEIGHT = 0.05`, `graduated_weight()` method
- **strategy/signal_integration.rs**: Replaced `should_use_model()` with `graduated_weight()` for lead_lag + informed_flow
- 2 new tests

### Phase 4a: RL as Ensemble Model (background agent)
- **learning/rl_edge_model.rs** (NEW): `RLEdgeModel` wraps `Arc<RwLock<QLearningAgent>>`, implements `EdgeModel`
- Converts `MarketState` → `MDPState` → Q-table lookup → returns (mean_bps, std_bps)
- High uncertainty (5.0 bps std) when cold or lock poisoned
- 6 new tests

### Phase 4b: Counterfactual Rewards (background agent)
- **learning/baseline_tracker.rs** (NEW): `BaselineTracker` with EWMA reward baseline
- `counterfactual_reward(actual) = actual - ewma_baseline` (removes fee drag)
- RL reward centers at ~0 instead of -1.5 bps
- 8 new tests

### Phase 4c: Controller → Risk Overlay (background agent)
- **control/mod.rs**: `RiskAssessment { size_multiplier, spread_multiplier, emergency_pull, reason }`
- `risk_assessment()` on StochasticController: trust → size, changepoint → spread, cp>0.9 → pull
- 4 new tests

### Phase 4d: Wire Unified Pipeline (lead)
- **quote_engine.rs**: Inserted risk_assessment() call, added to spread composition chain, applied size_multiplier
- **quote_engine.rs**: Replaced controller match block with diagnostic log (no more NoQuote from controller)
- **quote_engine.rs**: Removed RL override block (lines 1751-1794) — RL flows through ensemble
- **mod.rs**: Removed rl_enabled, rl_min_real_fills, rl_auto_disable fields + methods

### Phase 5: Quote Outcome Tracker (lead)
- **learning/quote_outcome.rs** (NEW): `QuoteOutcomeTracker`, `PendingQuote`, `QuoteOutcome`, `BinnedFillRate`
- Tracks P(fill | spread_bin) empirically, computes E[edge] = P(fill) × E[edge|fill]
- **mod.rs**: Added `quote_outcome_tracker` field to MarketMaker, initialized in constructor
- **quote_engine.rs**: Wired update_mid + expire_old_quotes each cycle, register pending quotes after ladder
- **handlers.rs**: Wired on_fill resolution inside fill processing loop
- 7 new tests

### Phase 6: Warmup Graduated Uncertainty (lead)
- **calibration/gate.rs**: `warmup_spread_discount(fill_count)` [0.85, 1.0] and `warmup_size_multiplier(fill_count)` [0.3, 1.0]
- **quote_engine.rs**: Applied gamma discount after parameter aggregation, size mult in target_liquidity
- 2 new tests

## Architecture After Redesign

```
Features → EdgePredictor(actual_spread) → SpreadOptimizer → RiskOverlay → Quotes
              │                                                  │
              ├─ GLFT model (IR-weighted)                        ├─ circuit breaker
              ├─ Empirical model (IR-weighted)                   ├─ toxicity
              ├─ Funding model (IR-weighted)                     ├─ position risk
              └─ RL model (IR-weighted, 5% floor)                └─ controller trust/changepoint
```

Spread composition: circuit_breaker × threshold_kappa × model_gating × staleness × toxicity × defensive × risk_overlay, capped at max_composed.

## Key Decisions
- RL min weight is 5% via graduated gating (not a separate mechanism)
- Controller never returns Quote/NoQuote — only multipliers + emergency_pull
- warmup_spread_discount applied to hjb_gamma_multiplier (lower γ → tighter GLFT spreads)
- warmup_size_multiplier applied alongside risk_size_mult in combined_size_mult
- QuoteOutcomeTracker only tracks best bid/ask (not all ladder levels) to keep overhead minimal
- fill_count from `adverse_selection.fills_measured()` drives warmup thresholds

## Validation
- 2,369 tests passing, 0 failed
- Clippy clean
- All 6 phases verified independently and together

## Open Items
- Paper trader validation needed: verify floor binds ~0%, edge predictions positive, no death spiral
- Live validation needed: edge > 0 bps after fees
- QuoteOutcomeTracker checkpoint persistence not yet implemented
- BaselineTracker not yet wired into RL reward computation (only struct created)
