Fix Saw-Tooth Inventory: Zero-Crossing Guard + Skew Revival + Kappa Fix                                                                                                                                    
                                                                                                                                                                                                            
 Context

 22-minute live session on HIP-3 (hyna:HYPE) revealed the system is a symmetric pinball machine. Position see-sawed through zero 3 times, ending -3.03 short in a +14.8 bps uptrend, with $-0.01 total P&L. 
  Seven concerns identified, distilled to four confirmed root causes with code-level evidence.

 Intended outcome: Position-aware quoting where (1) sweeps cannot push through zero, (2) inventory skew is non-zero, and (3) kappa doesn't self-tighten from near-zero-distance fills.

 ---
 Root Cause Summary

 ┌─────┬───────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────┬──────────────────────────────┐
 │  #  │                                Root Cause                                 │                           Evidence                            │            Impact            │
 ├─────┼───────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┤
 │ RC1 │ PositionGuard.update_position() never called in production                │ grep position_guard.update_position = 0 matches outside tests │ skew_bps=0.00 entire session │
 ├─────┼───────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┤
 │ RC2 │ Wrong-side liquidity = `max_pos -                                         │ pos                                                           │ ` with no cap                │
 ├─────┼───────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┤
 │ RC3 │ Kappa fills use fill_price for both args → distance=0 → kappa inflates 2× │ handlers.rs:1189 fill_price, fill_price                       │ Spread compressed 6→3.4 bps  │
 ├─────┼───────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┼──────────────────────────────┤
 │ RC4 │ Drift shrunk to zero by James-Stein (P_MIN=2.0 >> drift²=0.0006)          │ shrunken_drift_rate_per_sec() returns 0.0 for slow trends     │ PPIP drift term = 0          │
 └─────┴───────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────┴──────────────────────────────┘

 ---
 Fix A: Wire Position Guard (1 line each in 2 files)

 Problem: position_guard.rs:144 has update_position(&mut self, position: f64) which sets position and recomputes last_skew_bps. Never called in production.

 Effect when wired: At pos=-3.33/max=18, direct_skew = (-3.33/18) × 15.0 = -2.78 bps. This feeds into lead_lag_signal_bps at quote_engine.rs:1625, which shifts bid/ask asymmetry.

 Changes

 src/market_maker/orchestrator/quote_engine.rs (~line 1620):
 // BEFORE:
 let position_guard_skew_bps = self.safety.position_guard.inventory_skew_bps();

 // AFTER:
 self.safety.position_guard.update_position(position_now);
 let position_guard_skew_bps = self.safety.position_guard.inventory_skew_bps();

 src/market_maker/orchestrator/handlers.rs (after line 1192, in fill handler):
 // Keep position guard fresh between quote cycles
 self.safety.position_guard.update_position(self.position.position());

 Tests

 - Existing position_guard tests validate compute_inventory_skew_bps() math
 - Add integration test: after wiring, pos_guard_skew in ESTIMATOR_DIAGNOSTICS log should be non-zero when position is non-zero

 ---
 Fix B: Zero-Crossing Guard — Cap Wrong-Side Liquidity

 Problem: At pos=-3.33/max=18, the ladder offers 14.67 contracts of asks (wrong side). A sweep eats 3.33 reducing bids + pushes 1-2 contracts into opposite direction. Governor Green zone (< 50%) has NO   
 sizing restriction.

 Fix: After per-side capacity computation in ladder_strat.rs:1469, cap accumulating-side capacity at |position|. When flat (pos ≈ 0), both sides uncapped for initial position opening.

 Changes

 src/market_maker/strategy/ladder_strat.rs (insert after line 1469, before exchange limits):
 // === ZERO-CROSSING GUARD ===
 // Cap accumulating-side liquidity at |position| so sweeps cannot push
 // position through zero. When flat, both sides uncapped.
 let (local_available_bids, local_available_asks) = {
     let abs_pos = position.abs();
     let min_order_size = config.min_order_size.unwrap_or(0.01);
     if abs_pos > min_order_size {
         if position > 0.0 {
             // Long: asks are reducing (uncapped), bids are accumulating (cap)
             (local_available_bids.min(abs_pos), local_available_asks)
         } else {
             // Short: bids are reducing (uncapped), asks are accumulating (cap)
             (local_available_bids, local_available_asks.min(abs_pos))
         }
     } else {
         // Near-flat: both sides uncapped for initial position opening
         (local_available_bids, local_available_asks)
     }
 };

 Note: The min_order_size guard ensures the system can open initial positions from flat. Once |position| > min_order, the cap engages.

 Effect at pos=-3.33: available_for_asks drops from 14.67 → 3.33. A sweep of all 5 ask levels can at most bring position to ~0 (flat), not -3.33 → +1.67.

 Edge Cases

 - Position = 0: both sides uncapped (correct — system needs to open positions)
 - Position = -0.01: asks capped at 0.01 (below min_order → uncapped via guard)
 - Position = -3.33: asks capped at 3.33, bids uncapped at 21.33 (correct asymmetry)
 - After reducing fill brings pos from -3.33 to -2.33: cap auto-adjusts to 2.33

 Tests

 1. pos=-3.33, max=18: available_for_asks = 3.33 (was 14.67)
 2. pos=+5.0, max=18: available_for_bids = 5.0 (was 23.0)
 3. pos=0.0: both sides = max_position (uncapped)
 4. pos=-0.005 (below min_order): both sides uncapped

 ---
 Fix C: Fix Kappa Self-Tightening — Use mid_at_placement

 Problem: handlers.rs:1189 passes fill_price as both placement_price and fill_price to on_own_fill(). Since maker fills execute at the limit price, distance = |fill_price - fill_price| / fill_price = 0,  
 floored to 0.1 bps. After 16 fills, kappa inflates from 1500 → 3000+, halving GLFT half-spread.

 Fix: mid_at_placement is already computed at handlers.rs:1143 in the same scope. Pass it as the reference price.

 Changes

 src/market_maker/orchestrator/handlers.rs (lines 1186-1192):
 // BEFORE:
 let fill_size: f64 = fill.sz.parse().unwrap_or(0.0);
 self.estimator.on_own_fill(
     fill.time,  // timestamp_ms
     fill_price, // placement_price (best approximation)
     fill_price, // fill_price
     fill_size, is_buy,
 );

 // AFTER:
 let fill_size: f64 = fill.sz.parse().unwrap_or(0.0);
 // Use mid_at_placement as reference for kappa distance.
 // GLFT kappa measures fill rate decay as distance FROM MID.
 // fill_price for both args gave distance ≈ 0, inflating kappa 2×.
 self.estimator.on_own_fill(
     fill.time,         // timestamp_ms
     mid_at_placement,  // reference: mid when order was placed
     fill_price,        // where the order actually filled
     fill_size, is_buy,
 );

 Effect: With bid at $26.50 and mid=$26.55: distance = |26.50 - 26.55| / 26.55 = 18.8 bps. Kappa stabilizes at ~500-1500 instead of inflating to 3000+. GLFT half-spread stays in the 5-10 bps range.       

 Risk

 - Very low. mid_at_placement is already validated (> 0.0 check at line 1147) with self.latest_mid fallback.
 - The on_own_fill chain (parameter_estimator.rs:565 → kappa_orchestrator.rs:590 → kappa.rs:170) handles all edge cases.

 ---
 Fix D: Unshrunk Drift for PPIP Skew

 Problem: drift_estimator.rs:54 P_MIN=2.0. James-Stein shrinkage zeros drift unless |drift| > √P_MIN = 1.41 bps/sec. A +14.8 bps / 20 min trend = 0.012 bps/sec — 100× below threshold. The shrunken drift  
 feeds PPIP via market_params.drift_rate_per_sec, killing the drift_cost term entirely.

 Fix: Add a raw (unshrunk) drift field for PPIP. Keep shrunken drift for GLFT half-spread asymmetry (where phantom skew is dangerous).

 Changes

 src/market_maker/strategy/market_params.rs (near line 779, after drift_rate_per_sec):
 /// Raw (unshrunk) drift rate for PPIP skew (fractional/sec).
 /// Unlike drift_rate_per_sec, not zeroed by James-Stein shrinkage.
 /// PPIP's own Bayesian uncertainty (ambiguity/timing terms) handles noise filtering.
 pub drift_rate_per_sec_raw: f64,
 Default to 0.0 in the Default impl.

 src/market_maker/strategy/params/aggregator.rs (near line 779, in build()):
 drift_rate_per_sec_raw: 0.0, // Set by quote_engine from drift estimator

 src/market_maker/orchestrator/quote_engine.rs (near line 962, after drift_rate_per_sec assignment):
 market_params.drift_rate_per_sec_raw = self.drift_estimator.drift_rate_per_sec();

 src/market_maker/strategy/glft.rs (line 1510, inline PPIP):
 // BEFORE:
 let drift_mean = market_params.drift_rate_per_sec;

 // AFTER:
 // Use unshrunk drift for PPIP — PPIP's ambiguity/timing terms handle noise.
 // Shrunken drift (drift_rate_per_sec) zeros slow trends below 1.4 bps/sec.
 let drift_mean = market_params.drift_rate_per_sec_raw;

 Also in PosteriorPredictiveSkew::update_from_params() (glft.rs:108):
 // BEFORE:
 self.drift_mean = params.drift_rate_per_sec;

 // AFTER:
 self.drift_mean = params.drift_rate_per_sec_raw;

 Effect: A 0.025 bps/sec trend produces drift_cost = 0.025 × 60 = 1.5 bps of PPIP skew (was 0.0). Combined with Fix A (position guard 2.78 bps), total skew reaches ~4.3 bps at pos=-3.33 — meaningful      
 directional pressure.

 ---
 Implementation Order

 ┌───────┬──────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
 │ Phase │                      Fixes                       │                      Files Modified                       │                        Rationale                         │
 ├───────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
 │ 1     │ A + C (same file: handlers.rs + quote_engine.rs) │ handlers.rs, quote_engine.rs, position_guard.rs           │ Highest impact, lowest risk, partially overlapping files │
 ├───────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
 │ 2     │ D (drift field)                                  │ market_params.rs, aggregator.rs, quote_engine.rs, glft.rs │ Independent of Phase 1, adds 1 field                     │
 ├───────┼──────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
 │ 3     │ B (zero-crossing guard)                          │ ladder_strat.rs                                           │ Most impactful sizing change, test independently         │
 └───────┴──────────────────────────────────────────────────┴───────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘

 Phases 1+2 can run in parallel (different files). Phase 3 should land separately.

 ---
 Verification

 cargo fmt --all -- --check
 cargo clippy --lib -- -D warnings
 cargo test --lib position_guard       # Fix A: skew is non-zero
 cargo test --lib kappa                # Fix C: distance != 0
 cargo test --lib ppip                 # Fix D: drift feeds through
 cargo test --lib cascade_widens       # Regression: cascade still works
 cargo test --lib ladder               # Fix B: sizing caps
 cargo nextest run                     # Full suite

 Manual validation (next paper trading session)

 1. skew_bps in ESTIMATOR_DIAGNOSTICS should be non-zero when position is non-zero
 2. pos_guard_skew should show ~2-3 bps at moderate utilization
 3. Kappa should NOT inflate beyond 2× prior after 15 fills
 4. Spread should NOT compress below 4.5 bps during the session
 5. Zero-crossings should be 0-1 (was 3 in the baseline session)
 6. Wrong-side liquidity should approximately equal |position|, not 5× position