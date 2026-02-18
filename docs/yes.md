 Architecture Upgrade: Jump-Competitive Market Making

 Context

 The system has strong theoretical foundations (GLFT, Bayesian kappa, graduated risk) but operational bugs prevent it from working at all — 94% of cycles produce zero actions, warmup deadlocks, one-sided quoting, and    
 API quota death spirals. Meanwhile, significant infrastructure is already built but unwired: CrossAssetSignals (BTC lead-lag, funding divergence, OI-vol), ParameterSmoother (19 tests), PriceGrid (15 tests), multi-asset 
  allocator (8 tests), JointDynamics, FundingRateEstimator. This plan consolidates the three pending design docs (yay.md, plan.md, upgrade.md) into a phased execution roadmap that first makes the system work, then makes 
  it compete.

 Current state: -1.5 bps edge, 94% zero-action cycles, 32 bps adverse selection
 Target state: +8 to +18 bps edge, <10% zero-action cycles, <5 bps AS

 ---
 Sprint 1: "Make It Work" (Week 1-2) — ~8 tasks, +5 to +8 bps

 Everything from yay.md. The system ALWAYS has >= 1 bid + 1 ask resting.

 1.1 Flat API Costs in Scorer

 - File: tracking/order_manager/scorer.rs
 - What: Replace dynamic_api_cost_bps() (lines 120-131) with action.base_cost_bps() directly. Remove headroom param from function signature and all call sites in score_all().
 - Why: 1/headroom multiplier at 8% headroom = 75 bps/action. Makes ALL actions negative EV.
 - Impact: +1 bps
 - Risk: Low — budget allocator still limits total API calls

 1.2 Budget Allocator Relaxation

 - File: orchestrator/budget_allocator.rs
 - What: Line 118: change value_bps > 0.0 to value_bps > -2.0
 - Why: Guaranteed quotes may have slightly negative EV due to API costs but a MM with 0 resting orders MUST place
 - Impact: +0.5 bps

 1.3 Uniform Allocation for Small Capital

 - File: quoting/ladder/risk_budget.rs
 - What: After line 125, add small-capital bypass: when floor(total_capacity / min_viable_size) <= 4, use existing allocate_risk_budget_uniform() (line 248) instead of softmax. Use ceil_size for sizes near min_viable.   
 - Why: Softmax concentration gives 0 levels to one side at $156 capital
 - Impact: +0.5 bps

 1.4 AS Warmup Bootstrap

 - File: strategy/params/aggregator.rs line 259
 - What: Add || sources.cycle_count >= 50 to as_warmed_up computation. Add warmup_progress floor at 0.50 after first L2 data.
 - Why: Breaks AS↔fill↔quoting deadlock
 - Impact: +1 bps

 1.5 Guaranteed Quote Floor

 - File: strategy/ladder_strat.rs
 - What: New generate_guaranteed_quotes() method: half_spread = max(fee_bps + tick_bps, GLFT_optimal), size = min_viable, inventory skew. Integrate at end of generate_ladder() after size cap/dedup.
 - Why: THE critical fix. Transforms 94% zero-action cycles to 100% market presence.
 - Depends on: 1.3
 - Impact: +3 bps

 1.6 Reconciler Always-Execute

 - File: orchestrator/reconcile.rs
 - What: Remove headroom < 0.10 early returns (lines 729, 2048). Keep 1% hard block. Add guaranteed placement bypass: if 0 resting bids, boost first NewPlace to value_bps.max(50.0).
 - Depends on: 1.1
 - Impact: +1 bps

 1.7 Spread Pipeline Simplification

 - File: strategy/ladder_strat.rs
 - What: Remove 3 redundant transforms:
   a. Kappa cap (lines 1309-1336) — circular with GLFT, ~28 lines
   b. l2_reservation_shift (lines 1076-1101) — double-counts skew, ~25 lines
   c. Multi-component margin split (line 1590) — replace with inv_factor = (inventory_ratio * 0.3).clamp(-0.20, 0.20), ~90 lines removed
 - Depends on: 1.5 (safety net)
 - Impact: +0.5 bps (eliminates one-sided quoting)
 - Net: ~160 lines REMOVED

 1.8 Cleanup kappa_spread_bps

 - File: strategy/market_params.rs line 808, params/aggregator.rs
 - What: Remove kappa_spread_bps: Option<f64> field and its computation site

 Sprint 1 Verification

 cargo clippy -- -D warnings
 cargo test
 Paper trading 30 min on HYPE: every cycle >= 1 bid + 1 ask, first fill < 5 min, warmup 100% < 15 min, PnL near-zero, AS < 10 bps.

 ---
 Sprint 2: "Make It Defend" (Week 3-4) — 4 tasks, +2 to +3.5 bps

 Everything from plan.md. Complete the AS infrastructure so the system doesn't get picked off.

 2.1 InformedFlow Tightening Clamp

 - File: strategy/signal_integration.rs
 - What: Clamp informed_flow_spread_mult >= 1.0. Also clamp bid_mult/ask_mult asymmetric components in calibration/model_gating.rs.
 - Why: InformedFlow can tighten spreads below 1.0x, causing -0.23 bps
 - Impact: +0.23 bps

 2.2 Cancel-on-Toxicity

 - File: orchestrator/quote_engine.rs
 - What: Early in update_quotes(), check pre_fill_toxicity_bid/ask. If > 0.75, zero target ladder for that side (reconciler cancels). Add last_toxicity_cancel_time with 5s cooldown.
 - Why: Prevents fills at 0.16 bps spread (minimum tick) during toxic flow
 - Impact: +0.5 to +1.0 bps

 2.3 Per-Side Size Reduction from Toxicity

 - File: strategy/ladder_strat.rs
 - What: After pre-fill spread multiplier (line ~1091), apply per-side size mult: if toxicity > 0.5: size_mult = (1.0 - (toxicity - 0.5)).clamp(0.3, 1.0). Add pre_fill_size_mult_bid/ask to MarketParams.
 - Impact: +0.2 to +0.3 bps

 2.4 EnhancedASClassifier Blend

 - File: orchestrator/quote_engine.rs Phase 3 block
 - What: Call set_blended_toxicity() on pre_fill_classifier with EnhancedASClassifier output. record_outcome() already wired at handlers.rs:267.
 - Impact: +0.1 to +0.2 bps

 ---
 Sprint 3: "Make It Stable" (Week 5-6) — 4 tasks, +1 to +2 bps

 Everything from upgrade.md. Eliminate order churn, preserve queue position.

 3.1 Queue-Position-Weighted Latch

 - File: tracking/order_manager/reconcile.rs
 - What: In priority_based_matching(), widen latch for orders with good queue position (low depth_ahead). Front-of-queue: +50% latch bonus.
 - Impact: +0.3 bps

 3.2 Raise Latch Floor + Dead Code Cleanup

 - File: tracking/order_manager/reconcile.rs, orchestrator/reconcile.rs
 - What: Change latch clamp from (2.0, 10.0) to (3.0, 10.0). Delete dead latch_threshold_bps() at orchestrator/reconcile.rs:22-40.
 - Impact: +0.2 bps

 3.3 Quota-Scaled Tolerance

 - File: orchestrator/reconcile.rs
 - What: Before priority_based_matching(), multiply latch by quota factor: 1.0x at >=30%, ramp to 3.0x at <10%. Creates virtuous cycle (less quota used → wider tolerance → even less used).
 - Impact: +0.5 bps

 3.4 Priority-Filtered Execution

 - File: orchestrator/reconcile.rs
 - What: After scoring, classify by OperationPriority. At headroom <20%, skip LowValue actions; execute only HighValue and Emergency. BudgetPacer with OperationPriority exists but unused.
 - Impact: +0.3 bps

 ---
 Sprint 4: "Make It Smart" (Week 7-10) — 4 tasks, +2 to +4 bps

 Wire existing but unused alpha signals. This is where you start competing.

 4.1 Wire CrossAssetSignals to Pipeline

 - Files: learning/cross_asset.rs (COMPLETE, unwired), mod.rs, handlers.rs, signal_integration.rs
 - What: Add cross_asset_signals: CrossAssetSignals to MarketMaker. Wire update_btc_return() from Binance feed, update_funding() + update_oi() from exchange data. Consume aggregate_signal() and vol_multiplier() in       
 get_signals().
 - Existing: CrossAssetSignals::for_altcoin(), LeadLagModel, FundingDivergenceModel, OIVolModel — all complete with tests
 - Why: BTC leads HYPE by 50-500ms. This is a measured, genuine edge.
 - Impact: +1 to +2 bps

 4.2 Funding Rate Signal for Quoting

 - Files: process_models/funding.rs (COMPLETE), signal_integration.rs
 - What: Wire basis_velocity(), arbitrage_opportunity(), premium_alpha(), drift_adjustment() into get_signals(). Skew toward convergence, widen near settlement.
 - Existing: FundingRateEstimator with warmup, confidence, clamping — all built
 - Why: 8h funding cycle creates predictable flow patterns
 - Impact: +0.5 to +1 bps

 4.3 OI-Based Cascade Prediction

 - Files: learning/cross_asset.rs (OIVolModel), quote_engine.rs
 - What: Track OI from exchange WebSocket. When vol_multiplier() > 1.5, add as spread widening input in toxicity pipeline.
 - Why: OI drop signals liquidation cascade — widen before the wave arrives
 - Impact: +0.5 to +1 bps

 4.4 Side-Specific AS in GLFT

 - File: strategy/glft.rs
 - What: Compute separate bid_half_spread and ask_half_spread using existing kappa_bid() and kappa_ask() from estimator/mod.rs, incorporating side-specific AS from realized_as_buy()/sell().
 - Why: Asymmetric AS common during trending markets
 - Impact: +0.5 bps

 ---
 Sprint 5: "Make It Compete" (Week 11-14) — 3 tasks, +1 to +3 bps

 5.1 Multi-Asset Market Making

 - Files: multi/allocator.rs (COMPLETE, 8 tests), multi/margin_pool.rs, multi/batch.rs, src/bin/market_maker.rs
 - What: Wire AssetAllocator to run multiple MarketMaker instances with shared account and quota. Start with 2-3 uncorrelated assets.
 - Risk: High — start paper-only

 5.2 Continuous Kappa Blending

 - File: estimator/calibration_coordinator.rs
 - What: Replace 3-phase system with continuous: fill_weight = min(fill_count / 100, 0.8). Eliminates discrete parameter jumps.
 - Impact: +0.5 bps

 5.3 Inventory-Aware Kelly Sizing

 - File: learning/mod.rs
 - What: Wire kelly_tracker.kelly_recommendation() to position limits: effective_max = base_max * kelly_fraction.clamp(0.1, 1.0). Gate by kelly_warmed_up().
 - Impact: +0.5 bps

 ---
 Sprint 6: "Make It Robust" (Week 15-20) — 4 tasks, +0.5 to +1 bps

 6.1 Shadow Trading / Offline Policy Evaluation

 6.2 Automatic Model Staleness Detection

 6.3 Cancel-Race AS Tracking (wire existing adverse_selection/cancel_race.rs)

 6.4 L3-Like Queue Estimation (wire existing tracking/queue/ + QueuePositionEstimator)

 ---
 Cumulative Impact

 ┌────────────────────┬───────┬──────────────┬──────────────┬──────────────────────────┐
 │       Sprint       │ Weeks │ Expected bps │  Cumulative  │        Key Metric        │
 ├────────────────────┼───────┼──────────────┼──────────────┼──────────────────────────┤
 │ 1: Make It Work    │ 1-2   │ +5 to +8     │ +3.5 to +6.5 │ 94% zero-action → <10%   │
 ├────────────────────┼───────┼──────────────┼──────────────┼──────────────────────────┤
 │ 2: Make It Defend  │ 3-4   │ +2 to +3.5   │ +5.5 to +10  │ AS 32 bps → <5 bps       │
 ├────────────────────┼───────┼──────────────┼──────────────┼──────────────────────────┤
 │ 3: Make It Stable  │ 5-6   │ +1 to +2     │ +6.5 to +12  │ Cancel/fill ∞ → <2       │
 ├────────────────────┼───────┼──────────────┼──────────────┼──────────────────────────┤
 │ 4: Make It Smart   │ 7-10  │ +2 to +4     │ +8.5 to +16  │ Cross-asset signals live │
 ├────────────────────┼───────┼──────────────┼──────────────┼──────────────────────────┤
 │ 5: Make It Compete │ 11-14 │ +1 to +3     │ +9.5 to +19  │ Multi-asset portfolio    │
 ├────────────────────┼───────┼──────────────┼──────────────┼──────────────────────────┤
 │ 6: Make It Robust  │ 15-20 │ +0.5 to +1   │ +10 to +20   │ Model decay detection    │
 └────────────────────┴───────┴──────────────┴──────────────┴──────────────────────────┘

 ---
 Architecture Principles (Non-Negotiable)

 1. GLFT theory: All spreads via delta = (1/gamma) * ln(1 + gamma/kappa) + fee
 2. Bayesian estimation: Conjugate priors with proper count-based updates
 3. Graduated risk: Green/Yellow/Red/Kill with hysteresis, never binary
 4. Additive risk premiums: No multiplicative cascading
 5. Defense first: Guaranteed quotes are WIDE quotes, not aggressive ones
 6. Measurement before modeling: Every signal through MI gating + calibration

 Execution: All 6 Sprints

 Implement all 6 sprints sequentially. Each sprint: implement all tasks → clippy → test → move to next.

 Sprint execution order: 1 → 2 → 3 → 4 → 5 → 6 (dependency chain respected)

 Parallelism within sprints: Use agent teams where file ownership is non-overlapping. One cargo command at a time (resource constraint).

 After each sprint:
 cargo clippy -- -D warnings   # Must pass clean
 cargo test                     # All tests pass

 Total scope: ~27 tasks across 6 sprints. ~500 lines removed, ~800 lines added. Net: system gets simpler and more capable.