The plan is written. Here's the summary of what I found and what the plan proposes:                                                                                                   
                                                                                                                                                                                        
  The Diagnosis: Your system has one design flaw expressed 5 ways — no guaranteed-presence invariant. Every pipeline layer can produce zero quotes, and when they compose (softmax      
  concentration + headroom-scaled API costs + directional margin suppression + warmup gating + cascading spread transforms), zero-quote probability hits 94%. The system correctly      
  determines it can't make money at the spreads it computes, so it doesn't trade. When it does trade, it's one-sided (only sells), so it gets run over.                                 
                                                                                                                                                                                      
  The Fix Architecture: Two-phase quoting. Phase A (guaranteed) always produces 1 bid + 1 ask with a simple formula. Phase B (optimal) is the existing pipeline. If Phase B fails, Phase
   A's quotes are used. Defense = wider spreads, never = don't quote.

  6 Workstreams, 3 phases:
  - Day 1 (parallel): Flat API costs (scorer), uniform small-capital allocation (risk_budget), warmup bootstrap (quote_engine)
  - Day 2: Guaranteed quote floor (ladder_strat), reconciler always-execute (reconcile)
  - Day 3: Spread pipeline simplification (remove kappa cap, l2_reservation_shift, spread_compensation_mult)

  Net impact: ~80 lines REMOVED. The system gets simpler.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Ready to code?
                                                                                                                                                                                        
 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Principled Quoting Overhaul: Always-On Market Presence                                                                                                                               

 Context

 Trigger: 4-hour production run on HYPE (Feb 18, mm_hip3_hyna_HYPE_hip3_2026-02-18_01-54-29.log) produced:
 - 0 fills for the first 30+ minutes, 21 fills total over 4 hours
 - 6 wins / 15 losses (71% losers), 31.99 bps adverse selection
 - PnL: -$2.37, final position 1.76 HYPE long
 - bid_levels: 0 on hundreds of cycles (system only sells)
 - Reconciler took 0 actions on 94% of cycles (total_value_bps: 0.00, actions: 0)
 - as_warmed_up: false after 4 hours

 Root cause: Not 5 independent bugs — one design flaw expressed 5 ways: the system has no guaranteed-presence invariant. Every layer can produce zero quotes. When layers compose,
 zero-quote probability approaches certainty for small capital.

 Intended outcome: A market maker that ALWAYS has at least 1 bid + 1 ask resting. Defensive mechanisms modulate WHERE and HOW MUCH to quote, never WHETHER to quote. First fill within
 5 minutes. Warmup complete within 15 minutes.

 ---
 Architecture: Two-Phase Quoting

 Phase A (Guaranteed): Always produce 1 bid + 1 ask
   - Simple formula: half_spread = max(fee + tick, GLFT_optimal) + inventory_skew
   - Size = min_viable_size (smallest exchange-legal order)
   - Cannot be filtered, scored, or budget-gated away
   - Only kill switch can disable

 Phase B (Optimal): Existing GLFT/tick-grid/softmax pipeline
   - Generates 0..N additional levels per side
   - If Phase B produces results → they REPLACE Phase A for that side
   - If Phase B fails on a side → Phase A's quote is used

 ---
 5 Root Causes & Their Fixes

 #: RC1
 Root Cause: Softmax concentration → 0 levels for one side
 File: risk_budget.rs:122
 Fix: Uniform allocation bypass for small capital (<=4 min-viable orders)
 ────────────────────────────────────────
 #: RC2
 Root Cause: Reconciler API cost at 8% headroom = 75 bps → rejects everything
 File: scorer.rs:127, budget_allocator.rs:118
 Fix: Flat API costs (remove headroom multiplier), relax negative-value filter
 ────────────────────────────────────────
 #: RC3
 Root Cause: Directional signals suppress bid margin to 0
 File: ladder_strat.rs:1567-1595
 Fix: Inventory-only margin split, directional expressed through skew only
 ────────────────────────────────────────
 #: RC4
 Root Cause: Warmup requires fills, fills require quoting, quoting gated by warmup
 File: quote_engine.rs, ladder_strat.rs:1057
 Fix: L2-based warmup, 50% progress floor after first data
 ────────────────────────────────────────
 #: RC5
 Root Cause: 8+ cascading spread transformations → complexity death spiral
 File: ladder_strat.rs
 Fix: Remove kappa cap, l2_reservation_shift, spread_compensation_mult

 ---
 6 Workstreams (Parallel Where Possible)

 WS1: Guaranteed Quote Floor (ladder_strat.rs)

 Owner: Engineer A (strategy agent)
 Phase: 2 (after WS3 merges)

 Changes:
 1. New method generate_guaranteed_quotes() — add after generate_fallback_concentrated_ladder (line 636):
   - Computes half_spread = max(fee_bps + tick_bps, GLFT_optimal_bps)
   - Applies inventory_skew = inventory_ratio * half_spread * 0.5
   - bid_depth = half_spread + skew, ask_depth = half_spread - skew
   - Size = SizeQuantum::compute(...).min_viable_size
   - Returns (Option<LadderLevel>, Option<LadderLevel>)
   - Kill zone check: omit accumulating side only in Red/Kill zone
 2. Integrate at end of generate_ladder() — after size cap/dedup (~line 2250):
 if ladder.bids.is_empty() || ladder.asks.is_empty() {
     let (g_bid, g_ask) = self.generate_guaranteed_quotes(...);
     if ladder.bids.is_empty() { if let Some(b) = g_bid { ladder.bids.push(b); } }
     if ladder.asks.is_empty() { if let Some(a) = g_ask { ladder.asks.push(a); } }
 }
 3. Simplify margin split (lines 1496-1592): Replace 4-component stochastic-weighted split with inventory-only:
 let inv_factor = (inventory_ratio * 0.3).clamp(-0.20, 0.20);
 let ask_margin_weight = (0.5 + inv_factor).clamp(0.30, 0.70);
 3. Removes ~90 lines of momentum/urgency/directional/knife margin split code. Those signals already affect skew (signal_integration.rs); expressing them in margin split too causes
 one side to get zero budget.

 Verification:
 - generate_ladder() with $156 capital, flow_imbalance=-0.5 → at least 1 bid + 1 ask
 - Guaranteed spread between fee + tick and 3 * GLFT_optimal
 - All existing tests pass

 ---
 WS2: Flat API Costs in Scorer (scorer.rs)

 Owner: Engineer B (analytics agent)
 Phase: 1 (independent, start immediately)

 Changes:
 1. Replace dynamic_api_cost_bps() (line 127-131) with flat costs:
 fn api_cost_bps(action: ActionType) -> f64 {
     match action {
         ActionType::Latch => 0.0,
         ActionType::ModifySize => 0.5,
         ActionType::ModifyPrice => 1.0,
         ActionType::CancelPlace => 2.0,
         ActionType::NewPlace => 0.5,
         ActionType::StaleCancel => 0.5,
     }
 }
 1. Remove headroom parameter from function signature and all call sites in score_all().
 2. Relax negative-value filter in budget_allocator.rs:118:
 budgeted.retain(|u| u.value_bps > -2.0);
 2. Allows slightly negative-value actions through — guaranteed quotes always deserve execution. The -2 bps floor prevents truly destructive actions.

 Verification:
 - NewPlace at 5.5 bps depth, Normal toxicity → value_bps > 0 (was -75 bps at 8% headroom)
 - api_cost_bps(CancelPlace) == 2.0 regardless of headroom
 - Existing scorer tests adapted for new costs

 ---
 WS3: Uniform Allocation for Small Capital (risk_budget.rs)

 Owner: Engineer C (analytics agent)
 Phase: 1 (independent, start immediately)

 Changes:
 1. Add small-capital bypass at top of allocate_risk_budget() (after line 147):
 let max_possible_levels = (total_capacity / budget.min_viable_size).floor() as usize;
 if max_possible_levels <= 4 && max_possible_levels >= 1 {
     let usable_levels = levels.len().min(max_possible_levels);
     return allocate_uniform(&levels[..usable_levels], budget, total_capacity);
 }
 1. New helper allocate_uniform() distributes total_capacity equally across levels, uses ceil_size for sizes near min_viable to avoid truncation below threshold.
 2. Use ceil_size for small sizes in uniform path to prevent 0.33 HYPE → 0 when min is 0.34.

 Verification:
 - total_capacity=0.68, min_viable=0.34, 2 levels → exactly 2 allocations (was 0 or 1)
 - total_capacity=8.0, 5 levels → uses softmax (unchanged behavior)
 - Total allocated never exceeds position_capacity
 - All 16 existing risk_budget tests pass

 ---
 WS4: Reconciler Always Executes (reconcile.rs)

 Owner: Engineer D (infra agent)
 Phase: 2 (after WS2 merges)

 Changes:
 1. Remove headroom < 10% early return (reconcile.rs line 2048):
 Delete the headroom < 0.10 gate that prevents ALL reconciliation. Rate limiting is already handled by budget calculation (fewer calls) and continuous density scaling (fewer levels).
 Keep the headroom < 0.01 emergency gate.
 2. Guaranteed placement bypass — after scoring, before budget allocation:
 // If no resting bids/asks, mark first NewPlace as guaranteed (bypass budget filter)
 if num_current_bids == 0 {
     if let Some(first_bid) = bid_scored.iter_mut().find(|s| s.action == ActionType::NewPlace) {
         first_bid.value_bps = first_bid.value_bps.max(50.0);
     }
 }
 // Same for asks
 2. A market maker with 0 resting orders must always place. The 50.0 bps override ensures it passes any filter.

 Verification:
 - 0 current orders + headroom=5% → places at least 1 bid + 1 ask
 - Existing orders matching targets → latches correctly
 - Budget exhaustion only suppresses non-guaranteed actions

 ---
 WS5: Warmup Bootstrap (quote_engine.rs)

 Owner: Engineer E (signals agent)
 Phase: 1 (independent, start immediately)

 Changes:
 1. L2-based AS warmup — wherever market_params.as_warmed_up is set (in params/aggregator.rs:259):
 as_warmed_up: sources.adverse_selection.is_warmed_up()
     || sources.cycle_count >= 50   // 50 L2 observations enough for basic defense
 2. Warmup progress floor at 50% after first L2 data arrives:
 let warmup_progress = if first_data_received {
     computed_warmup_progress.max(0.50)
 } else {
     computed_warmup_progress
 };
 2. Currently warmup_progress stalls at 40% without fills (kappa component = 0). With 50% floor, the system reaches operational state faster.
 3. Remove zero-AS fallback in ladder_strat.rs:1057-1061:
 Change from if as_warmed_up { AS_value } else { 0.0 } to always apply AS (capped by warmup_as_cap). Zero AS = zero defense = worse than capped defense.
 Note: This line is in ladder_strat.rs — coordinate with WS1 engineer.

 Verification:
 - After 50 cycles with 0 fills → as_warmed_up = true
 - Warmup progress >= 50% after first L2 observation
 - AS adjustment applied during warmup, capped at warmup_as_cap

 ---
 WS6: Spread Pipeline Simplification (ladder_strat.rs, signal_integration.rs)

 Owner: Engineer F (strategy agent)
 Phase: 3 (AFTER WS1 merges — guaranteed quotes provide safety net)

 Changes:
 1. Remove kappa-cap logic (ladder_strat.rs lines 1308-1336):
 kappa_spread_bps = 2.0 / kappa * 10000 caps spreads AFTER GLFT already used kappa. Circular and removes gamma contribution. Delete ~28 lines + SPREAD TRACE logging.
 2. Remove l2_reservation_shift (ladder_strat.rs lines 1076-1101):
 Mid-price adjustment from mu / (gamma * sigma^2) double-counts directional info already in skew (signal_integration.rs). Delete ~25 lines.
 3. Remove spread_compensation_mult (ladder_strat.rs lines 1963-1983):
 Small-capital spread widening is redundant — GLFT already widens when gamma is higher (small capital = higher risk aversion). Delete ~20 lines.
 4. Bound combined_skew_bps in signal_integration.rs:
 combined_skew_bps = combined_skew_bps.clamp(-15.0, 15.0);
 4. Prevents skew from pushing one side to cross mid, which causes one-sided quoting.

 Verification:
 - SPREAD TRACE phases reduced from 6 to 4
 - combined_skew_bps bounded in [-15, 15]
 - All existing tests pass

 ---
 Integration Order

 Day 1 (parallel):  WS2 (scorer)    WS3 (risk_budget)    WS5 (quote_engine)
                       ↓                  ↓                     ↓
 Day 2 (sequential): WS4 (reconcile,    WS1 (ladder_strat,
                     needs WS2)          needs WS3)
                                            ↓
 Day 3 (sequential):                    WS6 (ladder_strat + signal_integration,
                                        needs WS1)

 File ownership (no conflicts within phases):

 ┌──────────────────────────────────┬───────┬───────┬───────┬───────┬───────┬───────┐
 │               File               │  WS1  │  WS2  │  WS3  │  WS4  │  WS5  │  WS6  │
 ├──────────────────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
 │ strategy/ladder_strat.rs         │ Day 2 │       │       │       │       │ Day 3 │
 ├──────────────────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
 │ tracking/order_manager/scorer.rs │       │ Day 1 │       │       │       │       │
 ├──────────────────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
 │ quoting/ladder/risk_budget.rs    │       │       │ Day 1 │       │       │       │
 ├──────────────────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
 │ orchestrator/reconcile.rs        │       │       │       │ Day 2 │       │       │
 ├──────────────────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
 │ orchestrator/budget_allocator.rs │       │       │       │ Day 2 │       │       │
 ├──────────────────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
 │ orchestrator/quote_engine.rs     │       │       │       │       │ Day 1 │       │
 ├──────────────────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
 │ strategy/signal_integration.rs   │       │       │       │       │       │ Day 3 │
 ├──────────────────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┤
 │ strategy/params/aggregator.rs    │       │       │       │       │ Day 1 │       │
 └──────────────────────────────────┴───────┴───────┴───────┴───────┴───────┴───────┘

 ---
 Expected Outcomes

 ┌───────────────────────┬──────────────────────────┬───────────────────────────┐
 │        Metric         │          Before          │           After           │
 ├───────────────────────┼──────────────────────────┼───────────────────────────┤
 │ Cycles with 0 actions │ 94% (2053/2176)          │ < 10%                     │
 ├───────────────────────┼──────────────────────────┼───────────────────────────┤
 │ Time to first fill    │ 30+ min (never in paper) │ 1-5 min                   │
 ├───────────────────────┼──────────────────────────┼───────────────────────────┤
 │ AS warmup completion  │ Never (4 hours)          │ ~15 min                   │
 ├───────────────────────┼──────────────────────────┼───────────────────────────┤
 │ Bid levels = 0        │ Hundreds of cycles       │ 0 (guaranteed floor)      │
 ├───────────────────────┼──────────────────────────┼───────────────────────────┤
 │ Win/loss ratio        │ 29% (6/21)               │ > 50% (symmetric quoting) │
 ├───────────────────────┼──────────────────────────┼───────────────────────────┤
 │ Adverse selection     │ 31.99 bps                │ < 5 bps (calibrated AS)   │
 └───────────────────────┴──────────────────────────┴───────────────────────────┘

 ---
 Verification Plan

 1. Unit tests: Each WS adds tests for its specific changes
 2. Clippy: cargo clippy -- -D warnings after each WS merge
 3. Full test suite: cargo test after all WS merge
 4. Paper trading validation: 30-min paper run on HYPE, verify:
   - Every cycle has >= 1 bid + 1 ask
   - First fill within 5 minutes
   - Warmup reaches 100% within 15 minutes
   - PnL positive or near-zero (not -$2.37)
   - Adverse selection < 10 bps
 5. Regression: Compare spread traces before/after for identical market data

 ---
 Net Code Impact

 - Removed: ~160 lines (kappa cap, l2_reservation_shift, spread_compensation_mult, 4-component margin split, headroom multiplier, headroom early returns)
 - Added: ~80 lines (generate_guaranteed_quotes, uniform allocation bypass, warmup bootstrap, skew bound)
 - Net: ~80 lines removed. The system gets SIMPLER, not more complex.