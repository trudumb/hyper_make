Principled Adverse Selection Architecture Redesign                                                                                                                                     
                                                                                                                                                                                      
 Context

 After a 3h19m mainnet session (Feb 14, 158 fills, -$0.82 PnL), AS remains the primary profitability blocker. 36 adverse fills (22% rate), many at 0.16 bps spread (minimum tick —
 being picked off at BBO), edge frequently negative (-1.66, -2.44 bps). The system has extensive AS infrastructure built across 10+ modules, but critical wiring is broken: 1 of 6
 classifier signals is dead, bias correction is never called, the InformedFlow model actively harms edge by -0.23 bps, and the toxicity threshold is set so high (0.5) that
 average-and-below toxicity produces zero spread response.

 Goal: Complete the existing AS wiring and fix thresholds so the infrastructure actually works. No new modules needed — the architecture is sound, the plumbing is broken.

 ---
 Changes Ordered by PnL Impact

 Phase 1: Wire Dead Signals + Fix Thresholds (~30 min)

 Fix 1: Wire trend signal to PreFillASClassifier [+0.5 to +1.0 bps]

 The trend signal has weight 0.30 (highest of all 6 signals) but update_trend() is never called in the live pipeline. trend_momentum_bps stays 0.0 permanently. This is the single
 highest-impact fix — the log shows "buys get adversely selected frequently when price drops after fill" which is exactly what Kyle 1985 drift conditioning detects.

 - File: src/market_maker/orchestrator/quote_engine.rs
 - Location: ~line 567, after self.tier1.pre_fill_classifier.update_funding(...) in the Phase 3 block
 - What: Add self.tier1.pre_fill_classifier.update_trend(trend_signal.long_momentum_bps) using the trend_signal already computed at line 458
 - Guard: Only call when trend_signal.is_warmed_up to avoid noise
 - Existing code: trend_signal is self.estimator.trend_signal(position_value) — already computed, contains long_momentum_bps (5-min EWMA)
 - Effect: -30bps downtrend → bid trend_signal = 1.0 → contributes 0.30 to weighted toxicity → bid spread_multiplier ~1.4x

 Fix 2: Lower toxicity→spread threshold from 0.5 to 0.3 [+0.3 to +0.5 bps]

 Current spread_multiplier() only widens above toxicity 0.5 (average). The warmup prior at 0.35 produces ZERO widening. Below-average toxicity is completely ignored.

 - File: src/market_maker/adverse_selection/pre_fill_classifier.rs
 - Location: Line ~1024 in spread_multiplier()
 - Current: let excess = (corrected_toxicity - 0.5).max(0.0) * 2.0
 - Change: Add configurable toxicity_floor (default 0.3) to PreFillClassifierConfig, then:
 let excess = (corrected_toxicity - self.config.toxicity_floor).max(0.0)
     / (1.0 - self.config.toxicity_floor);
 - Effect: Warmup prior 0.35 → multiplier ~1.14x. Toxicity 0.5 → multiplier ~1.57x. Proportional response across the full range.
 - Test update: test_as_bias_correction_applied_to_spread_multiplier and test_warmup_prior_produces_mild_spread_protection will need adjusted expectations

 Fix 3: Wire AS bias correction [+0.2 to +0.3 bps]

 observe_as_outcome() exists, is tested (3 passing tests), computes EWMA of (predicted - realized) AS, and spread_multiplier() already applies self.bias_correction() at line 1020. But
  observe_as_outcome() is never called from the orchestrator.

 - File: src/market_maker/orchestrator/handlers.rs
 - Location: Inside check_pending_fill_outcomes(), ~line 308 (after self.stochastic.signal_integrator.update_as_prediction(...))
 - What: Add:
 self.tier1.pre_fill_classifier.observe_as_outcome(
     predicted_as_bps_val, markout_as_bps
 );
 - Variables: predicted_as_bps_val (line 288/329) and markout_as_bps (line 322-327) already in scope
 - Effect: Closes the prediction→correction loop. If classifier overpredicts by 5 bps, bias correction reduces effective toxicity by ~0.25

 Fix 4: Disable InformedFlow tightening [+0.23 bps]

 InformedFlow's asymmetric_multiplier can go below 1.0 on one side, actively tightening spreads. Marginal value: -0.23 bps. The informed_flow_spread_mult at quote_engine.rs:1070 only
 checks > 1.01 (widening), but the asymmetric path bypasses this.

 - File: src/market_maker/strategy/signal_integration.rs
 - Location: Where signals.informed_flow_spread_mult is set
 - Change: Clamp to minimum 1.0: signals.informed_flow_spread_mult = mult.max(1.0)
 - Alternative: Set informed_flow_adjustment.min_tighten_mult = 1.0 and also clamp the bid_mult/ask_mult asymmetric components to >= 1.0 in model_gating.rs

 Phase 2: Active Defense Mechanisms (~45 min)

 Fix 5: Cancel-on-toxicity for resting orders [+0.5 to +1.0 bps]

 Log shows fills at spread_bps: 0.16 — BBO orders being picked off. When toxicity spikes, resting orders should be pulled before adverse fills. Currently cancellation only happens in
 the periodic quote cycle (too slow) or cascade tracker (too late, needs 3+ fills).

 - File: src/market_maker/orchestrator/quote_engine.rs (in update_quotes())
 - Location: Early in the quote cycle, before ladder computation
 - What: Check pre-fill toxicity, and if above emergency threshold (0.75), mark that side's orders for immediate cancellation by zeroing the ladder on that side
 - Implementation:
 const TOXICITY_CANCEL_THRESHOLD: f64 = 0.75;
 let bid_tox = market_params.pre_fill_toxicity_bid;
 let ask_tox = market_params.pre_fill_toxicity_ask;
 // Set a flag or directly zero the target for that side in the ladder computation
 - Key constraint: Keep this in the existing quote cycle flow (not a separate handler) to avoid async complexity and rate limit issues. The quote cycle already runs on every L2
 update.
 - Cooldown: Add a last_toxicity_cancel_time field, minimum 5s between toxicity-triggered cancels to avoid cancel storms

 Fix 6: Per-side size reduction from toxicity [+0.2 to +0.3 bps]

 When toxicity is moderate (0.5-0.75), don't cancel but reduce quote sizes on the toxic side.

 - File: src/market_maker/strategy/ladder_strat.rs
 - Location: After the pre-fill spread multiplier application (~line 1091), before final ladder generation
 - What: Apply a size multiplier per-side based on toxicity:
 if toxicity > 0.5: size_mult = 1.0 - (toxicity - 0.5) * 1.0  → [0.5, 1.0]
 clamp to [0.3, 1.0]
 - Integration: Add pre_fill_size_mult_bid and pre_fill_size_mult_ask to MarketParams (populated in ParameterAggregator), or compute inline from existing pre_fill_toxicity_bid/ask
 fields

 Fix 7: Wire EnhancedASClassifier output [+0.1 to +0.2 bps]

 The EnhancedASClassifier has 10 microstructure features (intensity z-score, price impact, run length, volume imbalance, spread widening, book velocity, arrival speed, size z-score,
 size/direction concentration). Its record_outcome() is already called in handlers.rs:267. But output is never consumed.

 - File: src/market_maker/orchestrator/quote_engine.rs
 - Location: Phase 3 block, after pre_fill_classifier updates (~line 568)
 - What: Call set_blended_toxicity() on pre_fill_classifier with enhanced classifier output
 - Verify: Check EnhancedASClassifier API — it may not have predict_toxicity(is_bid) in the same shape. Adapt as needed.

 ---
 Files Modified (Summary)

 ┌──────────────────────────────────────────┬────────────┬─────────────────────────────────────────────────────────┐
 │                   File                   │   Fixes    │                         Changes                         │
 ├──────────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
 │ orchestrator/quote_engine.rs             │ F1, F5, F7 │ Wire trend signal, toxicity-cancel flag, enhanced blend │
 ├──────────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
 │ adverse_selection/pre_fill_classifier.rs │ F2         │ Lower threshold, add toxicity_floor config field        │
 ├──────────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
 │ orchestrator/handlers.rs                 │ F3         │ Wire observe_as_outcome()                               │
 ├──────────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
 │ strategy/signal_integration.rs           │ F4         │ Clamp informed_flow_spread_mult >= 1.0                  │
 ├──────────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
 │ strategy/ladder_strat.rs                 │ F6         │ Per-side toxicity size reduction                        │
 ├──────────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
 │ strategy/market_params.rs                │ F6         │ Optional: add pre_fill_size_mult_bid/ask                │
 ├──────────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────┤
 │ calibration/model_gating.rs              │ F4         │ Clamp asymmetric multipliers >= 1.0                     │
 └──────────────────────────────────────────┴────────────┴─────────────────────────────────────────────────────────┘

 ---
 Verification

 1. cargo clippy -- -D warnings — must pass clean
 2. cargo test — all tests pass (update threshold-dependent tests for F2)
 3. Key tests to check after F2:
   - test_as_bias_correction_applied_to_spread_multiplier
   - test_warmup_prior_produces_mild_spread_protection
   - test_dynamic_as_buffer_scales_with_warmup
 4. After deployment: monitor these log fields:
   - as_warmed_up should eventually become true
   - trend_momentum_bps should be non-zero (confirms F1)
   - spread_multiplier should show values > 1.0 during toxic flow (confirms F2)
   - bias_correction_bps should appear in periodic logs (confirms F3)
   - Fill rate should decrease on toxic side (confirms F5/F6)
   - Edge should improve from current ~-1.5 bps toward positive

 Implementation Order

 Execute sequentially (file ownership prevents parallelism for most changes):

 1. F1 (trend signal) + F3 (bias correction) + F4 (InformedFlow clamp) — independent, can be done together
 2. F2 (threshold) — then run tests, fix any broken threshold expectations
 3. cargo clippy && cargo test
 4. F5 (cancel-on-toxicity) + F6 (size reduction)
 5. F7 (enhanced classifier blend)
 6. Final cargo clippy && cargo test

 Expected total impact: +2.0 to +3.5 bps, swinging current -1.5 bps edge to approximately +0.5 to +2.0 bps.