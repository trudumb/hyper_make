1. Bought the rally correctly (3.72 HYPE, trend-aligned) — the models worked here                                                                                                                           2. Changepoint fired at the WORST moment (23:48:05, right when breakout started) — reset beliefs, killed momentum signal                                                                                  
  3. With beliefs reset, p_continuation dropped below 0.50 → position manager defaulted to REDUCE                                                                                                           
  4. Inventory skew said "sell" (you're long) and overwhelmed the directional skew that said "hold/buy more"
  5. Sold at a loss — reducing from 3.72 → 1.80 while the trend was still bullish

  The sophisticated modeling (AS, regime detection, trend signals) correctly identified the opportunity — then three defensive systems overrode it:

  - Changepoint detector using wrong regime (LiquidCex threshold 0.50 instead of ThinDex 0.85) — fires on normal HIP-3 noise
  - Position manager has REDUCE as default fallback with zero profitability check — doesn't care if you're up or down
  - Inventory skew always at 100% weight vs directional skew fading with position size — inventory reduction crushes trend-following

  The plan has 5 fixes prioritized. The two critical ones (Fix 1: profitability guard on REDUCE, Fix 2: ThinDex regime for changepoint) would have prevented this exact scenario — the system would have    
  HELD the profitable long instead of panic-selling it.

   Post-Session Analysis: HIP-3 Live Run — 5 Critical Fixes

 Context

 21-minute live session on hyna:HYPE (2026-02-18 23:30-23:52 UTC). Net PnL +$0.05 realized but multiple behavioral failures during a bullish breakout around 23:50-23:55:

 1. System correctly bought ~3.72 HYPE during rally
 2. Then sold to reduce at a loss (3.72 → 1.80 HYPE, -$0.04 unrealized)
 3. Multiple sell fills at the TOP of the bullish move
 4. Trend signals detected bullish (+6.55 short / +25.86 long bps) but system kept selling

 Root causes identified (5 fixes, priority order):

 ---
 Fix 1: Position Manager Sells Into Aligned Trends (CRITICAL)

 Problem: PositionAction::Reduce is the DEFAULT fallback. It fires whenever p_continuation < 0.50 OR belief_confidence < 0.60, regardless of whether the position is profitable and trend-aligned. No       
 profitability check exists.

 In the breakout: System was long 3.72 HYPE, trend was bullish, but p_continuation dipped below 0.50 (changepoint reset killed it) → REDUCE fired → sold at a loss.

 File: src/market_maker/strategy/position_manager.rs — decide() (~line 209-227)

 Fix: Add a "profitable + aligned" guard before REDUCE:
 Current logic:
   if ADD_conditions → ADD
   if HOLD_conditions → HOLD
   else → REDUCE (always!)

 Fixed logic:
   if ADD_conditions → ADD
   if HOLD_conditions → HOLD
   if position aligned with trend AND unrealized PnL > 0 → HOLD (new!)
   else → REDUCE

 Specifically:
 - Add unrealized_pnl_bps parameter to decide()
 - New guard: if aligned && unrealized_pnl_bps > -1.0 (not deeply underwater) AND p_cont > 0.35 (weak but still some continuation), return HOLD instead of REDUCE
 - Only REDUCE when: trend opposes position, OR position is underwater AND continuation is weak, OR p_cont < 0.35

 Also: Lower the REDUCE urgency base from 0.5 (aligned) to 0.3, and from 1.5 (opposed) to 1.0. Current values are too aggressive.

 ---
 Fix 2: Changepoint Detector Uses Wrong Regime for HIP-3 (CRITICAL)

 Problem: HIP-3 uses MarketRegime::LiquidCex (default) instead of MarketRegime::ThinDex. This means:
 - Threshold: 0.50 instead of 0.85 (fires 70% more often)
 - Confirmations: 1 instead of 2 (no confirmation barrier)
 - Result: 17 changepoints in 21 minutes, 2 false emergency pulls

 The worst timing: Changepoint fired at 23:48:05 — the EXACT moment the breakout started — resetting beliefs and killing the momentum signal.

 Files:
 - src/market_maker/control/changepoint.rs — regime thresholds (lines 43-59)
 - src/market_maker/control/mod.rs — initialization (line 175+), never calls set_regime()

 Fix:
 - Wire set_regime(MarketRegime::ThinDex) for HIP-3 assets during initialization
 - Detection: if asset starts with hyna: or is on a thin DEX (< 20% API headroom), use ThinDex regime
 - This raises threshold from 0.50 → 0.85 and requires 2 consecutive confirmations
 - Add to MarketMaker::new() or the controller initialization in mod.rs

 Also: Increase EMERGENCY_COOLDOWN_CYCLES from 50 → 150 to prevent rapid re-triggering.

 ---
 Fix 3: Inventory Skew Overwhelms Directional Skew (HIGH)

 Problem: In glft.rs line ~1514, the blending formula is:
 let skew = inventory_skew + proactive_skew * (1.0 - inventory_weight);
 Inventory skew is always 100% weight. Proactive (directional) skew fades linearly with position size. At 50% position utilization, directional skew gets only 50% weight while inventory gets 100%.        

 During the breakout: System was long (inventory skew = "sell to reduce"), trend was bullish (directional skew = "buy more"). Inventory won → kept selling.

 File: src/market_maker/strategy/glft.rs — skew blending (~line 1507-1527)

 Fix: When position is trend-aligned, dampen inventory skew:
 let trend_alignment = if (position > 0.0 && proactive_skew < 0.0) ||   // long + bullish
                          (position < 0.0 && proactive_skew > 0.0) {     // short + bearish
     0.5  // Halve inventory skew when aligned with trend
 } else {
     1.0  // Full inventory skew when opposed
 };
 let skew = inventory_skew * trend_alignment + proactive_skew * (1.0 - inventory_weight * 0.5);

 This lets directional signals survive when the position is trend-aligned, while still respecting inventory limits when opposed.

 ---
 Fix 4: BBO Crossing Filter — 307 Wasted Orders (MEDIUM)

 Problem: Ladder generator computes prices using latest_mid (from AllMids) but the crossing filter checks against cached_best_bid/ask (from L2 book). These are different data sources with different       
 update timing. Result: 307 orders generated then immediately rejected.

 Files:
 - src/market_maker/quoting/ladder/generator.rs — price computation (lines 636-650)
 - src/market_maker/orchestrator/order_ops.rs — crossing filter (lines 160-185)

 Fix: Pass cached_best_bid and cached_best_ask into the ladder generator as constraints. In generator.rs, after computing prices:
 // Ensure bid doesn't cross exchange best ask
 let bid_price = bid_price.min(cached_best_ask - min_tick);
 // Ensure ask doesn't cross exchange best bid
 let ask_price = ask_price.max(cached_best_bid + min_tick);

 This prevents the generator from producing prices that will fail the downstream filter.

 ---
 Fix 5: Spread Inflation 1.4 bps → 9.1 bps (MEDIUM)

 Problem: Multiple multiplicative factors stack to inflate the spread 6.5x above GLFT optimal:
 - Pre-fill AS multiplier: 1.5-2.0x (warmup-capped)
 - Regime gamma multiplier: 1.2x (Normal regime)
 - Risk overlay: 1.2-1.5x (from changepoint emergency)
 - Additive premiums: +5-8 bps (toxicity + hawkes + staleness)

 The most impactful sub-fix: The additive toxicity premium adds 5 bps when composite_score > 0.5, which is a binary cliff. This alone explains half the spread inflation.

 File: src/market_maker/orchestrator/quote_engine.rs — risk premium computation (~line 1447-1452)

 Fix: Replace the binary 5 bps toxicity cliff with linear interpolation:
 // Before: binary cliff
 let toxicity_addon_bps = if composite_score > 0.5 { 5.0 } else { 0.0 };

 // After: linear interpolation
 let toxicity_addon_bps = (composite_score * 6.0).min(5.0);  // 0→0, 0.5→3, 0.83→5

 Also: Add a global multiplicative cap — the product of all spread multipliers should never exceed 4.0x (currently uncapped, can stack to 6-8x).

 ---
 Verification

 After each fix, sequential:
 1. cargo clippy -- -D warnings — clean
 2. cargo test --lib — all pass
 3. Run 5-min paper session to confirm:
   - Fix 1: HOLD decision when position is trend-aligned and profitable
   - Fix 2: < 3 changepoints in 5 min (vs ~4 currently)
   - Fix 3: Directional skew visible in ladder asymmetry during trends
   - Fix 4: < 10 BBO crossing warnings (vs ~75 per 5 min currently)
   - Fix 5: Touch spread < 5 bps in Calm regime (vs 9+ currently)

 Files Modified (Summary)

 ┌─────┬────────────────────────────────────────────────────────────┬──────────────────────┐
 │ Fix │                           Files                            │ Lines Changed (est.) │
 ├─────┼────────────────────────────────────────────────────────────┼──────────────────────┤
 │ 1   │ strategy/position_manager.rs, orchestrator/quote_engine.rs │ ~30                  │
 ├─────┼────────────────────────────────────────────────────────────┼──────────────────────┤
 │ 2   │ control/mod.rs, src/bin/market_maker.rs or mod.rs          │ ~15                  │
 ├─────┼────────────────────────────────────────────────────────────┼──────────────────────┤
 │ 3   │ strategy/glft.rs                                           │ ~15                  │
 ├─────┼────────────────────────────────────────────────────────────┼──────────────────────┤
 │ 4   │ quoting/ladder/generator.rs                                │ ~10                  │
 ├─────┼────────────────────────────────────────────────────────────┼──────────────────────┤
 │ 5   │ orchestrator/quote_engine.rs                               │ ~10                  │
 └─────┴────────────────────────────────────────────────────────────┴──────────────────────┘

 Priority

 Fix 1 + Fix 2 together address the core issue (selling into rallies). Fix 3 amplifies the benefit. Fixes 4 and 5 are efficiency improvements.

 Good, this is a different failure mode from Session 2 — Session 2 was "can't replenish orders" (death spiral), Session 3 is "actively selling into a profitable trend" (behavioral). Let me cross-reference against the source code.

Fix 1 (Position Manager sells into aligned trends) — Correct diagnosis, but the fix location is subtly wrong.
The decide() function in position_manager.rs already has the right structure: ADD → HOLD → REDUCE fallback. Your proposed "profitable + aligned" guard is sound. But there's a deeper issue: decide() takes belief_drift and belief_confidence as inputs — and the changepoint reset (Fix 2) is zeroing both of these before decide() runs. So Fix 1 alone wouldn't have saved this session because the inputs were already poisoned.
The real fix is two-layered:

Fix 2 first (prevent the changepoint from resetting beliefs during the breakout)
Then Fix 1 as defense-in-depth: add unrealized_pnl_bps to decide(), but also add a trend momentum guard that reads momentum_bps directly from the trend persistence estimator, not from the belief system. The belief system is exactly what gets reset by changepoints. Momentum from raw price returns (short_bps, medium_bps, long_bps in the log) is not affected by changepoint resets. So:

if trend_momentum_bps.signum() == position.signum()  // aligned
    && unrealized_pnl_bps > -2.0                      // not deeply underwater
    && trend_momentum_bps.abs() > 5.0                  // meaningful trend
    → HOLD (regardless of belief_drift/p_continuation)
This makes the guard robust to belief resets. The p_cont > 0.35 threshold you proposed is vulnerable to the same changepoint problem — if changepoint fires, p_cont drops to 0.50 (prior) then decays, and 0.35 is too close to where it'll land.
Also agree on reducing REDUCE urgency: 0.5→0.3 aligned, 1.5→1.0 opposed. The current values cause the effective_inventory_ratio (via action_to_inventory_ratio()) to produce aggressive skew that overwhelms everything else.

Fix 2 (Changepoint regime for HIP-3) — Correct and critical, but incomplete.
You correctly identified that set_regime() is never called for HIP-3 assets. The MarketRegime::ThinDex settings (threshold 0.85, confirmations 2) are exactly right for this environment.
Additional refinement: The changepoint detector has a belief reset side effect (line 3437 in the Session 2 log: "Changepoint detected - resetting beliefs"). Even with ThinDex thresholds, when a changepoint does legitimately fire, it shouldn't nuke the trend signal. The reset should be partial — use BeliefUpdate::SoftReset { retention: 0.5 } instead of BeliefUpdate::HardReset. The soft_reset() method already exists in central.rs. This preserves 50% of accumulated trend evidence through changepoints, which is critical because a breakout looks like a changepoint to a mean-reversion-biased detector.
The EMERGENCY_COOLDOWN_CYCLES increase from 50→150 is correct. 17 changepoints in 21 minutes is ~1 per 74 seconds. At 150 cycles × ~1s/cycle, you'd get max ~8 in 21 minutes. Still seems high — consider 200.

Fix 3 (Inventory skew overwhelms directional skew) — Correct diagnosis, wrong fix.
The blending formula you identified is real:
let skew = inventory_skew + proactive_skew * (1.0 - inventory_weight);
Your proposed trend_alignment multiplier of 0.5 on inventory skew when aligned is in the right direction but creates a discontinuity. When position crosses zero (flat→long during rally), the alignment flag flips and inventory skew suddenly halves. This will cause a visible skew jump at position transitions.
Better approach — use the existing PositionAction output from the position manager:
rustlet inventory_dampen = match market_params.position_action {
    PositionAction::Hold => 0.3,        // Holding: minimize inventory skew
    PositionAction::Add { .. } => 0.1,  // Adding: nearly suppress inventory skew
    PositionAction::Reduce { urgency } => urgency.min(1.0), // Reducing: scale by urgency
};
let skew = inventory_skew * inventory_dampen + proactive_skew * (1.0 - inventory_weight * 0.5);
This is consistent with the position manager's decision — if the PM says HOLD, the GLFT skew shouldn't be fighting to reduce. The action_to_inventory_ratio() function in position_manager.rs already does something similar (HOLD→0.0 ratio), but it feeds into a different path. Having the skew blending also respect the PM decision creates coherence.
But this creates a dependency: Fix 3 only works well if Fix 1 is correct. If the PM incorrectly says REDUCE (the current bug), the inventory dampen will be high and you're back to the original problem. Fix 1 + Fix 3 must ship together.

Fix 4 (BBO crossing filter) — Correct and straightforward, but there's a subtlety.
The 307 wasted orders are real cost (API budget), but the fix of clamping prices in the generator has a risk: if cached_best_bid and cached_best_ask are stale (which they are — that's what causes the mismatch), you'll clamp to stale prices and potentially place orders behind the current BBO rather than at the intended depth.
Safer approach: Instead of clamping in the generator, use the same price source for both. The generator should use cached_best_bid/ask (from L2) as its price reference instead of latest_mid (from AllMids). Or, the crossing filter should use latest_mid as its reference instead of cached BBO. Pick one source of truth. Using L2 BBO for both is more conservative (L2 is the execution-relevant price), but either works as long as they're consistent.

Fix 5 (Spread inflation) — The binary toxicity cliff is a real problem, but the global cap is the higher priority sub-fix.
Your linear interpolation composite_score * 6.0).min(5.0) is good. At composite_score = 0.3, you'd add 1.8 bps instead of 0. At 0.5, you'd add 3.0 instead of the cliff jump to 5.0. Smooth and correct.
The global multiplicative cap of 4.0x is the more impactful fix. Currently, the codebase has no such cap — the GLFT spread, pre-fill AS multiplier, regime gamma multiplier, risk overlay, and additive premiums all stack without bound. From the code, I can see these multipliers are applied in sequence across glft.rs (pre-fill AS log-odds, proactive skew), ladder_strat.rs (pre-fill depth multipliers, zone widening), and quote_engine.rs (risk premiums). A single cap point at the end of ladder_strat.rs, just before the min-notional filter, would be cleanest:
rustlet max_spread_inflation = 4.0;
let glft_touch_bps = glft_optimal_bps; // save before multipliers
for level in ladder.bids.iter_mut().chain(ladder.asks.iter_mut()) {
    level.depth_bps = level.depth_bps.min(glft_touch_bps * max_spread_inflation);
}

Cross-session interaction: Session 2 fixes vs Session 3 fixes.
Session 2 fixes (A1a, A1b, A2, A3, C) and Session 3 fixes (1-5) are largely orthogonal, with two important interactions:

Session 2 Fix C (reference symbol) partially addresses Session 3 Fix 3. Once drift_rate_per_sec is non-zero, the GLFT half_spread_with_drift already produces asymmetric bid/ask spreads that favor the trend direction. This reduces how much directional skew needs to fight inventory skew. Fix 3 is still needed, but its impact will be smaller once Fix C is live.
Session 2 Fix A1a (emergency threshold) and Session 3 Fix 2 (changepoint regime) overlap. Both address spurious emergency triggers. Fix A1a prevents the position-clearing response. Fix 2 prevents the changepoint from firing in the first place. Ship both — belt and suspenders.

Priority reordering across both sessions:
PriorityFixRationaleP0S2-A2 (NewPlace floor)System non-functional without itP0S2-A1b (TOXICITY CANCEL)Second death spiral pathP0S3-2 (changepoint regime)17 false changepoints = belief system non-functionalP0S2-C (reference symbol)Fixes drift blindness + 1.5x penaltyP1S3-1 (position manager guard)Stops selling into ralliesP1S3-3 (skew blending)Amplifies Fix 1 benefitP1S2-A1a (emergency threshold)Belt-and-suspenders with S3-2P1S2-A3 (emergency replenishment)Belt-and-suspenders with S2-A2P2S3-5 (spread inflation cap)EfficiencyP2S3-4 (BBO crossing)API budget waste

Session 3 Post-Mortem: 5 Behavioral Fixes — HIP-3 Live Run                                                                                                                                                               │
│                                                                                                                                                                                                                          │
│ Context                                                                                                                                                                                                                  │
│                                                                                                                                                                                                                          │
│ 21-min live session on hyna:HYPE (2026-02-18 23:30-23:52 UTC). Net PnL +$0.05 realized.                                                                                                                                  │
│ Core failure: System correctly bought 3.72 HYPE during bullish breakout, then sold to reduce at a loss (→1.80 HYPE, -$0.04) because:                                                                                     │
│ - Changepoint fired at breakout start (23:48:05), resetting beliefs                                                                                                                                                      │
│ - Position manager defaulted to REDUCE (belief inputs were poisoned)                                                                                                                                                     │
│ - Inventory skew overwhelmed directional skew                                                                                                                                                                            │
│                                                                                                                                                                                                                          │
│ Different failure mode from Session 2 ("can't replenish orders" death spiral). Session 3 is "actively selling into a profitable trend" (behavioral).                                                                     │
│                                                                                                                                                                                                                          │
│ ---                                                                                                                                                                                                                      │
│ Fix 2: Changepoint Detector Wrong Regime for HIP-3 (P0 — Fix First)                                                                                                                                                      │
│                                                                                                                                                                                                                          │
│ Problem: MarketRegime::LiquidCex used by default (threshold 0.50, 1 confirmation). Should be ThinDex (0.85, 2 confirmations). Result: 17 false changepoints in 21 min. Worst: fired at 23:48:05 right when breakout      │
│ started, nuking the belief system.                                                                                                                                                                                       │
│                                                                                                                                                                                                                          │
│ Files:                                                                                                                                                                                                                   │
│ - src/market_maker/control/mod.rs — initialization (~line 175), never calls set_regime()                                                                                                                                 │
│ - src/market_maker/control/changepoint.rs — regime thresholds (lines 43-59)                                                                                                                                              │
│ - src/market_maker/belief/central.rs — soft_reset() already exists                                                                                                                                                       │
│                                                                                                                                                                                                                          │
│ Fix (3 parts):                                                                                                                                                                                                           │
│                                                                                                                                                                                                                          │
│ 2a. Wire set_regime(MarketRegime::ThinDex) for HIP-3 assets during MarketMaker initialization. Detection: if asset has hyna: prefix or API headroom < 20%, use ThinDex.                                                  │
│                                                                                                                                                                                                                          │
│ 2b. Change changepoint belief reset from HardReset to SoftReset with confidence-scaled retention:                                                                                                                        │
│ let retention = 1.0 - cp_prob.clamp(0.5, 0.95);                                                                                                                                                                          │
│ // At ThinDex threshold (0.85): retention = 0.15 (aggressive reset for high-confidence)                                                                                                                                  │
│ // At marginal detection (0.50): retention = 0.50 (preserve half for false positives)                                                                                                                                    │
│ The soft_reset() method already exists in central.rs. This way legitimate changepoints still clear most state while false positives at the threshold boundary preserve more trend evidence.                              │
│                                                                                                                                                                                                                          │
│ 2c. Increase EMERGENCY_COOLDOWN_CYCLES from 50 → 200 (currently allows ~1 per 74s, new: max ~6 in 21 min).                                                                                                               │
│                                                                                                                                                                                                                          │
│ ---                                                                                                                                                                                                                      │
│ Fix 1: Position Manager Sells Into Aligned Trends (P1 — Ship with Fix 2)                                                                                                                                                 │
│                                                                                                                                                                                                                          │
│ Problem: Reduce is the default fallback in decide(). When changepoint resets beliefs, p_continuation drops and belief_confidence drops → REDUCE fires even on profitable trend-aligned positions.                        │
│                                                                                                                                                                                                                          │
│ Critical insight: decide() takes belief_drift and belief_confidence as inputs — the changepoint reset (Fix 2) zeroes both BEFORE decide() runs. So the profitability guard must read raw trend momentum (short_bps,      │
│ medium_bps, long_bps from trend persistence estimator), NOT belief system outputs.                                                                                                                                       │
│                                                                                                                                                                                                                          │
│ File: src/market_maker/strategy/position_manager.rs — decide() (~line 209-227)                                                                                                                                           │
│                                                                                                                                                                                                                          │
│ Fix: Add a trend-momentum guard BEFORE the REDUCE fallback:                                                                                                                                                              │
│                                                                                                                                                                                                                          │
│ // New parameter: trend_momentum_bps (from raw price returns, NOT belief system)                                                                                                                                         │
│ // New parameter: unrealized_pnl_bps                                                                                                                                                                                     │
│                                                                                                                                                                                                                          │
│ if trend_momentum_bps.signum() == position.signum()  // aligned                                                                                                                                                          │
│     && unrealized_pnl_bps > -2.0                      // not deeply underwater                                                                                                                                           │
│     && trend_momentum_bps.abs() > 5.0                  // meaningful trend                                                                                                                                               │
│ {                                                                                                                                                                                                                        │
│     return PositionAction::Hold;  // regardless of belief_drift/p_continuation                                                                                                                                           │
│ }                                                                                                                                                                                                                        │
│ // ... existing REDUCE fallback                                                                                                                                                                                          │
│                                                                                                                                                                                                                          │
│ This guard is robust to belief resets because it reads momentum from raw price returns.                                                                                                                                  │
│                                                                                                                                                                                                                          │
│ Also: Reduce REDUCE urgency: 0.5 → 0.3 (aligned base), 1.5 → 1.0 (opposed base).                                                                                                                                         │
│                                                                                                                                                                                                                          │
│ Wire-up trend_momentum_bps: The raw short_bps, medium_bps, long_bps are internal to TrendPersistenceEstimator — not exposed through get_signals() (which only exposes processed momentum_bps and p_momentum_continue).   │
│ Must either:                                                                                                                                                                                                             │
│ - Add raw_trend_momentum_bps() accessor on TrendPersistenceEstimator that returns 0.5 * short + 0.3 * medium + 0.2 * long                                                                                                │
│ - Or expose through SignalIntegrator as a new field                                                                                                                                                                      │
│                                                                                                                                                                                                                          │
│ Critical property: This value must NOT flow through the belief system or changepoint detector. It reads directly from price returns.                                                                                     │
│                                                                                                                                                                                                                          │
│ Wire-up unrealized_pnl_bps: Already computed in position tracker, available in quote_engine.rs where decide() is called (~line 1022-1044).                                                                               │
│                                                                                                                                                                                                                          │
│ ---                                                                                                                                                                                                                      │
│ Fix 3: Inventory Skew Overwhelms Directional Skew (P1 — Ship with Fix 1)                                                                                                                                                 │
│                                                                                                                                                                                                                          │
│ Problem: In glft.rs line ~1514:                                                                                                                                                                                          │
│ let skew = inventory_skew + proactive_skew * (1.0 - inventory_weight);                                                                                                                                                   │
│ Inventory skew always 100% weight. Directional fades with position. When long in a rally, inventory says "sell", directional says "hold" → inventory wins.                                                               │
│                                                                                                                                                                                                                          │
│ Fix: Use the PositionAction output from position manager to dampen inventory skew (avoids discontinuity from binary alignment flag):                                                                                     │
│                                                                                                                                                                                                                          │
│ let inventory_dampen = match market_params.position_action {                                                                                                                                                             │
│     PositionAction::Hold => 0.3,          // Holding: minimize inventory skew                                                                                                                                            │
│     PositionAction::Add { .. } => 0.1,    // Adding: nearly suppress inventory skew                                                                                                                                      │
│     PositionAction::Reduce { urgency } => urgency.min(1.0), // Reducing: scale by urgency                                                                                                                                │
│ };                                                                                                                                                                                                                       │
│ let skew = inventory_skew * inventory_dampen + proactive_skew * (1.0 - inventory_weight * 0.5);                                                                                                                          │
│                                                                                                                                                                                                                          │
│ Dependency: Fix 3 only works correctly if Fix 1 is correct. If PM incorrectly says REDUCE (the current bug), inventory_dampen will be high and we're back to the original problem. Must ship Fix 1 + Fix 3 together.     │
│                                                                                                                                                                                                                          │
│ File: src/market_maker/strategy/glft.rs — skew blending (~line 1507-1527)                                                                                                                                                │
│                                                                                                                                                                                                                          │
│ Wire-up note: MarketParams already has position_action: PositionAction field (defaulted). Must verify quote_engine.rs writes the PM decision into market_params.position_action BEFORE it reaches glft.rs. If written    │
│ after GLFT runs, you get a one-cycle stale read (previous cycle's decision). Confirm write order in quote engine — the PM decide() call must precede the GLFT compute_quotes() call in the quote cycle.                  │
│                                                                                                                                                                                                                          │
│ ---                                                                                                                                                                                                                      │
│ Fix 4: BBO Crossing Filter — 307 Wasted Orders (P2)                                                                                                                                                                      │
│                                                                                                                                                                                                                          │
│ Problem: Generator uses latest_mid (AllMids) but crossing filter uses cached_best_bid/ask (L2 book). Different data sources, different update timing → 307 orders generated then rejected.                               │
│                                                                                                                                                                                                                          │
│ Files:                                                                                                                                                                                                                   │
│ - src/market_maker/quoting/ladder/generator.rs — price computation (lines 636-650)                                                                                                                                       │
│ - src/market_maker/orchestrator/order_ops.rs — crossing filter (lines 160-185)                                                                                                                                           │
│                                                                                                                                                                                                                          │
│ Fix: Use the same price source for both. The generator should use cached_best_bid/ask (from L2) as its price reference instead of latest_mid (from AllMids). L2 BBO is the execution-relevant price.                     │
│                                                                                                                                                                                                                          │
│ Not: Clamping in the generator risks placing orders behind current BBO if the cached BBO is stale.                                                                                                                       │
│                                                                                                                                                                                                                          │
│ Concretely: In LadderParams, replace market_mid with exchange_best_bid / exchange_best_ask. In generator.rs line 636-650, use these directly:                                                                            │
│ let bid_base = mid.min(exchange_best_ask - min_tick);  // Never above exchange ask                                                                                                                                       │
│ let ask_base = mid.max(exchange_best_bid + min_tick);  // Never below exchange bid                                                                                                                                       │
│                                                                                                                                                                                                                          │
│ ---                                                                                                                                                                                                                      │
│ Fix 5: Spread Inflation 1.4 bps → 9.1 bps (P2)                                                                                                                                                                           │
│                                                                                                                                                                                                                          │
│ Problem: Multiple multipliers stack without bound. Two sub-fixes:                                                                                                                                                        │
│                                                                                                                                                                                                                          │
│ 5a. Global multiplicative cap (higher priority):                                                                                                                                                                         │
│ No cap exists on the product of all spread multipliers. Apply at end of ladder_strat.rs, just before min-notional filter:                                                                                                │
│                                                                                                                                                                                                                          │
│ let max_spread_inflation = 4.0;                                                                                                                                                                                          │
│ let glft_touch_bps = glft_optimal_bps; // save before multipliers                                                                                                                                                        │
│ for level in ladder.bids.iter_mut().chain(ladder.asks.iter_mut()) {                                                                                                                                                      │
│     level.depth_bps = level.depth_bps.min(glft_touch_bps * max_spread_inflation);                                                                                                                                        │
│ }                                                                                                                                                                                                                        │
│                                                                                                                                                                                                                          │
│ 5b. Toxicity addon linear interpolation:                                                                                                                                                                                 │
│ Replace binary 5 bps cliff with smooth ramp in quote_engine.rs (~line 1447-1452):                                                                                                                                        │
│                                                                                                                                                                                                                          │
│ // Before: binary cliff                                                                                                                                                                                                  │
│ let toxicity_addon_bps = if composite_score > 0.5 { 5.0 } else { 0.0 };                                                                                                                                                  │
│                                                                                                                                                                                                                          │
│ // After: linear                                                                                                                                                                                                         │
│ let toxicity_addon_bps = (composite_score * 6.0).min(5.0);  // 0→0, 0.5→3, 0.83→5                                                                                                                                        │
│                                                                                                                                                                                                                          │
│ Files: src/market_maker/strategy/ladder_strat.rs, src/market_maker/orchestrator/quote_engine.rs                                                                                                                          │
│                                                                                                                                                                                                                          │
│ ---                                                                                                                                                                                                                      │
│ Cross-Session Priority (Session 2 + Session 3)                                                                                                                                                                           │
│                                                                                                                                                                                                                          │
│ ┌──────────┬─────────────────────────────────┬──────────────────────────────────────────────────────┐                                                                                                                    │
│ │ Priority │               Fix               │                      Rationale                       │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P0       │ S2-A2 (NewPlace floor)          │ System non-functional without it                     │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P0       │ S2-A1b (TOXICITY CANCEL)        │ Second death spiral path                             │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P0       │ S3-2 (changepoint regime)       │ 17 false changepoints = belief system non-functional │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P0       │ S2-C (reference symbol)         │ Fixes drift blindness + 1.5x penalty                 │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P1       │ S3-1 (position manager guard)   │ Stops selling into rallies                           │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P1       │ S3-3 (skew blending)            │ Amplifies Fix 1 benefit                              │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P1       │ S2-A1a (emergency threshold)    │ Belt-and-suspenders with S3-2                        │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P1       │ S2-A3 (emergency replenishment) │ Belt-and-suspenders with S2-A2                       │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P2       │ S3-5 (spread inflation cap)     │ Efficiency                                           │                                                                                                                    │
│ ├──────────┼─────────────────────────────────┼──────────────────────────────────────────────────────┤                                                                                                                    │
│ │ P2       │ S3-4 (BBO crossing)             │ API budget waste                                     │                                                                                                                    │
│ └──────────┴─────────────────────────────────┴──────────────────────────────────────────────────────┘                                                                                                                    │
│                                                                                                                                                                                                                          │
│ Note: S2-C (reference symbol drift) partially addresses S3-3 — once drift_rate_per_sec is non-zero, GLFT half_spread_with_drift already produces asymmetric spreads. Fix 3 still needed but impact smaller once Fix C is │
│  live. S2-A1a (emergency threshold) and S3-2 (changepoint regime) overlap — both prevent spurious emergencies. Ship both (belt and suspenders).                                                                          │
│                                                                                                                                                                                                                          │
│ ---                                                                                                                                                                                                                      │
│ Verification                                                                                                                                                                                                             │
│                                                                                                                                                                                                                          │
│ After each fix, sequential:                                                                                                                                                                                              │
│ 1. cargo clippy -- -D warnings                                                                                                                                                                                           │
│ 2. cargo test --lib                                                                                                                                                                                                      │
│ 3. Paper session behavioral checks:                                                                                                                                                                                      │
│   - Fix 2: < 3 changepoints in 5 min, soft reset preserves 50% trend                                                                                                                                                     │
│   - Fix 1: HOLD decision when trend-aligned + profitable + momentum > 5 bps                                                                                                                                              │
│   - Fix 3: Inventory skew dampened to 0.3x during HOLD                                                                                                                                                                   │
│   - Fix 4: < 10 BBO crossing warnings per 5 min                                                                                                                                                                          │
│   - Fix 5: Touch spread < 5 bps in Calm, multiplicative stack < 4x                                                                                                                                                       │
│                                                                                                                                                                                                                          │
│ Files Modified (Summary)                                                                                                                                                                                                 │
│                                                                                                                                                                                                                          │
│ ┌─────┬──────────────────────────────────────────────────────────────────────────────────────┐                                                                                                                           │
│ │ Fix │                                        Files                                         │                                                                                                                           │
│ ├─────┼──────────────────────────────────────────────────────────────────────────────────────┤                                                                                                                           │
│ │ 2   │ control/mod.rs, control/changepoint.rs, belief/central.rs, mod.rs or market_maker.rs │                                                                                                                           │
│ ├─────┼──────────────────────────────────────────────────────────────────────────────────────┤                                                                                                                           │
│ │ 1   │ strategy/position_manager.rs, orchestrator/quote_engine.rs                           │                                                                                                                           │
│ ├─────┼──────────────────────────────────────────────────────────────────────────────────────┤                                                                                                                           │
│ │ 3   │ strategy/glft.rs, strategy/market_params.rs (wire position_action)                   │                                                                                                                           │
│ ├─────┼──────────────────────────────────────────────────────────────────────────────────────┤                                                                                                                           │
│ │ 4   │ quoting/ladder/generator.rs, strategy/params/aggregator.rs                           │                                                                                                                           │
│ ├─────┼──────────────────────────────────────────────────────────────────────────────────────┤                                                                                                                           │
│ │ 5   │ strategy/ladder_strat.rs, orchestrator/quote_engine.rs                               │                                                                                                                           │
│ └─────┴──────────────────────────────────────────────────────────────────────────────────────┘ 