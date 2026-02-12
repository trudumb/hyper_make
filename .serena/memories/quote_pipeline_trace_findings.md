# Quote Pipeline Trace - Raw Findings

## File Layout Overview
- **handlers.rs**: WebSocket message dispatcher (AllMids, Trades, L2Book, UserFills, OrderUpdates)
- **quote_engine.rs**: 1,800+ lines - THE critical path containing the entire composition chain
- **signal_integration.rs**: 6 signals combined (lead-lag, informed flow, regime kappa, model gating, VPIN, buy pressure)
- **ladder_strat.rs**: Multi-level GLFT quoting with Bayesian fill probabilities
- **glft.rs**: Half-spread formula and gamma calculation
- **kill_switch.rs**: 8 monitors for emergency shutdown
- **pre_fill_classifier.rs**: 6-signal Bayesian blend for pre-fill toxicity
- **reconcile.rs**: Final order placement with position/reduce-only checks

## Quote Engine Flow (quote_engine.rs)
1. Data quality gate (line 32)
2. Warmup timeout check (line 62)
3. Circuit breaker checks (line 112)
4. Risk limit checks (line 146)
5. Belief system update (line 159)
6. Parameter aggregation from multiple sources (line 512)
7. Warmup graduated uncertainty (line 584)
8. BOCPD regime detection (line 600)
9. Belief skewness spread adjustment (line 657)
10. Position continuation model (line 708)
11. HMM regime probabilities (line 819)
12. Lead-lag signal integration (line 858)
13. Cross-venue beliefs (line 924)
14. Risk overlay assessment (line 980)
15. Spread multiplier composition (line 993-1110)
16. Size multiplier composition (line 1112)
17. Measured latency (line 1129)
18. Proactive position management (line 1179-1257)
19. Target liquidity computation (line 1262-1308)
20. Ladder generation (line 1500+)
21. Reconciliation (line 1500+)

## Spread Multiplier Chain (quote_engine.rs line 993-1110)
1. Base: 1.0 (from circuit_breaker_action)
2. × threshold_kappa_mult (line 1000) - momentum regime detection
3. × model_gating_mult (line 1012) - low model confidence
4. × staleness_mult (line 1022) - stale signal defense
5. × toxicity_mult (line 1055) - proactive toxicity from VPIN/informed/trend
6. × defensive_mult (line 1074) - fill-based supplementary defense
7. × risk_overlay.spread_multiplier (line 1086) - controller changepoint/trust
8. Capped by max_composed (line 1097)
9. Applied to market_params.spread_widening_mult (line 1101)

## Risk Features Integration
- circuit_breaker: line 112-143
- drawdown: line 154-157
- risk position check: line 146-151
- HJB controller momentum: line 424-429
- stochastic constraints: line 1142
- calibration progress: line 552-556

## Key Override/Cap Points
1. Data quality gate → cancel all orders (line 32)
2. Warmup timeout → forced start (line 62)
3. Circuit breaker → pause/cancel/widen (line 112)
4. Position hard limit → hard breach (line 148)
5. Drawdown emergency → pause (line 154)
6. OI cap → skip quotes (line 252)
7. Rate limit → skip requote (line 262)
8. Warmup size mult → reduces position (line 584-598)
9. Risk size mult → reduces position (line 1113)
10. Drawdown size mult → reduces position (line 1117)
11. Spread composition cap → min(total, max_composed) (line 1097)
12. Effective liquidity caps → min(derived, config, position, exchange) (line 1279)
