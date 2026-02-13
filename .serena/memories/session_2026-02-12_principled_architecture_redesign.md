# Session 2026-02-12: Principled Architecture Redesign

## Summary
Analyzed the Feb 12 19:37 mainnet log (22K lines, 103 fills, ~1hr) alongside the strategy redesign doc 
and Feb 12 audit findings. Used 4-agent team for parallel deep codebase analysis. Produced unified 
architecture redesign treating 5 strategy principles as foundational assumptions.

## Deliverable
`.claude/plans/principled-architecture-redesign.md` — 400+ line design document

## New Findings from Feb 12 19:37 Log
- effective_max_position = 53-58 HYPE from margin vs config = 3.24 (STILL broken after Feb 10 "fixes")
- Floor binding every cycle: GLFT 3.04-5.58 bps, floor 5.51-10.01 bps
- Untracked fill of 3.14 HYPE (cancel-fill race → position jumped to -5.77)
- Edge prediction: always ~-0.7 bps predicted, realized -7 to +9 bps (no predictive power)
- directional signal computed but only affects margin split, never actual price skew
- 3 fill cascades, position swung -5.87 to +4.99
- Sharpe(all) = 439 (unreliable due to AS tautology)

## Architecture: 6 Layers (Foundation Up)
0. **Measurement Substrate**: OrderLifecycle stores mid_at_placement; EdgeSnapshot deferred to 5s markout
1. **Inventory Governor**: config.max_position is HARD ceiling; 4 zones (Green/Yellow/Red/Kill)
2. **Regime State Machine**: HMM + kappa_orchestrator → RegimeParams consumed by everything
3. **Spread Engine**: solve_min_gamma(fee + AS + risk) — NO separate floor, buffer, or cap
4. **Asymmetric Pricer**: inventory_skew (always active) + signal_skew (direction [-1,1])
5. **Risk Overlay**: graduated widening (never cancel-all except kill switch)

## Key Design Decisions
- Margin-derived capacity used ONLY for notional sizing, NEVER for position limits
- No separate floor concept — floor emerges from fee + measured_AS + risk_premium
- Skew scales with half_spread (proportional to risk, not absolute)
- Emergency pull → 2.5x widen (not cancel-all), reduce-only quotes ALWAYS preserved
- Signals must produce direction [-1,1], not just spread_adjustment_bps
- EdgeSnapshot created ONLY after 5s markout (not at fill time)

## Migration: 6 Phases
1. Measurement fix (unblocks everything)
2. Inventory Governor (safety critical)
3. Regime Engine (unlocks spread + skew)
4. Spread Engine (eliminates floor)
5. Asymmetric Pricer (eliminates zero skew)
6. Risk Overlay simplification

## What Gets Deleted
- adaptive/learned_floor.rs (replaced by measured components in RegimeParams)
- 7-deep spread multiplier chain (replaced by regime + overlay)
- effective_max_position from margin (replaced by config hard cap)
- conditional_as_buffer_bps = 2.0 (replaced by measured AS)
- Emergency pull cancel-all (replaced by graduated widening)
- InformedFlow tightening (harmful, -0.23 bps)
- Immediate AS computation at fill time (replaced by deferred markout)

## Team Used
4 parallel Explore agents: pipeline-tracer, risk-analyst, spread-regime-analyst, skew-signals-analyst
All completed analysis of their domains before synthesis.
