# Session: Production Pipeline Wiring (Feb 15, 2026)

## What Changed
Wired 5 standalone components into the production quote pipeline (`quote_engine.rs:update_quotes()`).
All components existed as dead code from previous session — this session activated them.

## Components Wired

| Component | Where | Behavioral Change |
|-----------|-------|-------------------|
| **ToxicityRegime** | After line 633 (pre_fill_classifier block) | Replaces hardcoded 0.75/0.50 toxicity thresholds |
| **FeatureSnapshot** | After line 711 (ParameterAggregator::build()) | Provides compact state for downstream models |
| **ExecutionMode** | After line 2285 (kill switch ladder depth) | Replaces QuoteGate OnlyBids/OnlyAsks side-masking |
| **QueueValueHeuristic** | After toxicity defense, before Risk Emergency | Filters levels with negative expected edge |
| **CUSUM divergence** | In ExecutionMode's `has_alpha` field | Additional alpha source beyond lead-lag MI |

## Files Modified
- `src/market_maker/orchestrator/quote_engine.rs` — All 5 wirings (6 insertion/replacement points)
- `src/market_maker/mod.rs` — Removed `last_toxicity_cancel_ms` field + `#[allow(dead_code)]` from queue_value_heuristic
- `src/market_maker/orchestrator/handlers.rs` — Queue value outcome feedback in `check_pending_fill_outcomes()`

## Key Decisions
1. **ToxicityRegime replaces hardcoded thresholds** — regime-aware with built-in hysteresis, no cooldown timer needed
2. **ExecutionMode replaces QuoteGate side-masking** — both ladder mode AND single-quote fallback updated
3. **QuoteGate KEPT for spread-width decisions** — WidenSpreads/NoQuote spread multipliers still active (lines 2208+)
4. **`last_toxicity_cancel_ms` removed** — ToxicityRegime's hysteresis handles cancel storm prevention natively
5. **Queue value feedback uses Normal toxicity default** — fill-time toxicity not tracked yet (conservative)
6. **CUSUM extends has_alpha** — `signals.lead_lag_actionable || signal_integrator.has_cusum_divergence()`

## What Was NOT Changed
- QuoteGate spread-widening logic (WidenSpreads, NoQuote → 2x spread) — unchanged, orthogonal
- Risk Emergency filter — unchanged, runs after all new filters
- Fill Cascade tracker — unchanged
- ReplayEngine — offline-only, no production wiring needed
- CusumDetector in signal_integration.rs — already wired, verified only

## Verification
- `cargo clippy -- -D warnings`: clean
- `cargo test`: 2,676 pass / 4 pre-existing failures (drawdown + calibration_coordinator)
- No regressions

## Architecture: Quote Pipeline Filter Order (post-wiring)
1. QuoteGate spread-widening (WidenSpreads/NoQuote → market_params.spread_widening_mult)
2. Ladder generation (calculate_ladder)
3. **ExecutionMode** side selection (Flat/Maker/InventoryReduce)
4. **ToxicityRegime** defense (Toxic/Normal/Benign size adjustments)
5. **QueueValue** per-level filtering (negative edge removal)
6. Risk Emergency filter (position-reducing quotes survive)
7. Fill Cascade filter (same-side fill runaway prevention)

## Open Items
- Queue value feedback at fill time uses `ToxicityRegime::Normal` default — should store toxicity at quote time
- `feature_snapshot.queue_rank_bid/ask` hardcoded to 0.0 in FeatureSnapshot (Phase 3 TODO in snapshot.rs)
- Single-quote fallback mode has ExecutionMode wiring but no toxicity regime or queue value filtering
