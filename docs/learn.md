Paper-to-Live EV Gap: Analysis & Improvement Hypotheses

  Based on deep analysis from 4 specialized agents across your stochastic control, execution, signal integration, and infrastructure modules, here's the full picture.

  ---
  The Gap: +2.79 bps (paper) vs -1.5 bps (live) = 4.29 bps leakage
  ┌────────────────────────────────┬─────────────┬───────────────────────────────────────────────────────────────┐
  │             Source             │ Est. Impact │                          Root Cause                           │
  ├────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────┤
  │ Queue position simulation bias │ ~2.0 bps    │ Paper: 60% penetration + age bonus. Live: back of book        │
  ├────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────┤
  │ AS under-widening              │ ~1.0 bps    │ Paper has vol-scaled AS threshold; live ran static weights    │
  ├────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────┤
  │ Rate limit staleness           │ ~0.8 bps    │ 7% quota → 5-15s stale quotes during moves                    │
  ├────────────────────────────────┼─────────────┼───────────────────────────────────────────────────────────────┤
  │ Cold-start warmup              │ ~0.5 bps    │ Paper inherits calibrated AS model; live starts from defaults │
  └────────────────────────────────┴─────────────┴───────────────────────────────────────────────────────────────┘
  ---
  I. Stochastic Control Improvements

  Hypothesis 1: The Time Horizon T Is Miscalibrated

  Your HJB controller uses session_duration_secs = 86400 (24h) and the GLFT uses T = 1/λ (inverse arrival rate). But perpetual futures have no natural session — the real horizon is the 8-hour funding cycle.

  Conjecture: Setting T = time_to_next_funding_settlement would:
  - Create natural urgency near settlement (inventory matters for funding P&L)
  - Remove the arbitrary 24h session that makes terminal penalty meaningless
  - Allow γ × σ² × q × T skew term to scale with actual risk horizon

  Iteration: Replace fixed session with rolling 8h windows aligned to funding timestamps. The terminal penalty becomes penalty = expected_spread_to_cross + funding_cost_of_carrying.

  Hypothesis 2: Value Function Basis Is Too Sparse for Quota States

  Your value function approximation uses 15 basis functions but none encode rate limit state. The controller literally cannot reason about quota scarcity.

  Conjecture: Adding basis functions for:
  - rate_limit_headroom (linear + quadratic)
  - headroom × position (cross-term: quota matters MORE when you have inventory)
  - headroom × sigma (quota matters MORE in volatile markets)

  ...would let TD(0) learn that "quoting with 7% headroom and 1.5 HYPE position has negative expected value — don't."

  Iteration: Add 3 quota basis functions to value.rs, retrain with paper episodes that simulate rate limit pressure.

  Hypothesis 3: The Reward Function Is Not Grounded in Realized PnL

  Your controller uses arbitrary reward coefficients (-0.05, -0.01, 0.1) for dump/wait/build actions. These aren't derived from actual spread capture minus fees minus AS.

  Conjecture: Replace synthetic rewards with:
  R(s, a) = realized_spread_capture - realized_as_bps - fee_bps - funding_carry_cost
  This is what your analytics module already computes via EdgeSnapshot. Wire it as the true reward signal.

  Iteration: After each fill, compute realized_edge_bps and use it as the TD target instead of synthetic expected_value.

  ---
  II. Game-Theoretic Improvements

  Hypothesis 4: You're Playing Against Yourself (Inventory Forcing Death Spiral)

  At 7% quota, the system enters inventory-forcing mode — one-sided quoting to reduce position. But this creates a self-defeating game:

  1. Short -1.57 HYPE → quote only bids
  2. Get filled → now less short or long → switch to asks
  3. Get filled → short again → repeat
  4. Each whipsaw pays spread twice AND burns quota

  This is a dominated strategy. In game-theoretic terms, you're playing "always defect" when the optimal strategy is "conditional cooperation with the market."

  Conjecture: Replace inventory-forcing with patience + wider spreads:
  - Instead of one-sided forcing, quote both sides at 2-3x normal width
  - Accept lower fill rate but preserve two-sided market
  - Position mean-reverts naturally through balanced flow
  - Quota consumption drops because wider spreads need fewer updates

  Iteration: In quote_gate.rs:939-982, replace inventory_forcing_decision() with wide_two_sided_decision() that multiplies spreads by 1.0 / headroom_pct.sqrt() instead of going one-sided.

  Hypothesis 5: Competitor Modeling Is Unused

  You have competitor_snipe_prob, competitor_spread_factor, and competitor_count in MarketParams but these are observation-only. No strategic response.

  Conjecture (Stackelberg Leader-Follower): If you're the only MM on HIP-3 (likely given its illiquidity), you're a monopolist LP. Your optimal spread isn't GLFT's competitive-market formula — it's:
  δ_monopolist = δ_GLFT + (1/η) × market_share
  where η is taker price elasticity. On illiquid tokens, takers are inelastic (they need to trade), so you can quote wider than GLFT suggests.

  Iteration: Estimate η from fill rate vs spread width regression. If fills don't decay much when you widen by 5 bps, you're under-pricing your liquidity.

  Hypothesis 6: Adverse Selection Is a Signaling Game

  Your AS classifier treats all fills as "toxic or not." But informed traders don't always trade — they only trade when your quotes are mispriced. This is a Bayesian signaling game:

  - When you quote tight, only informed traders fill you (adverse selection)
  - When you quote wide, only noise traders fill you (safe but low volume)
  - The optimal signal-dependent spread is:
  δ*(s) = δ_GLFT + (1/γ) × ln(P(informed | s) / P(noise | s))

  Your pre_fill_classifier already computes P(informed | microstructure). But you're applying it as a multiplier (1.0-3.0x) instead of the log-odds correction above.

  Iteration: In glft.rs spread computation, replace multiplicative AS adjustment with log-odds additive term from the classifier posterior.

  ---
  III. Microstructure & Execution Improvements

  Hypothesis 7: Queue Position Is the Largest Unmodeled Factor

  Paper assumes 60% queue penetration. Live has zero queue visibility. This is ~2 bps of leakage.

  Conjecture: You can estimate queue position from L2 snapshots:
  queue_position_frac ≈ size_ahead_of_my_price / total_size_at_my_level
  Since Hyperliquid L2 snapshots show aggregate size at each level, and you know your own order size, you can back out approximate position.

  Iteration:
  1. Track size_at_my_level from L2 when order is placed
  2. Track size_at_my_level over time — if it grows, you're further back
  3. Adjust fill probability: P(fill) = P(price_touch) × (1 - queue_frac)^α
  4. Use this in paper trader to make paper more realistic (close the gap from the paper side)

  Hypothesis 8: Cancel-Fill Race Creates Hidden Adverse Selection

  You noted "order cancelled but filled first → position spike." This isn't a bug — it's selection bias. The fills you receive during cancellation are the most toxic fills because:
  - You cancel when you detect adverse movement
  - But the fill arrived because an informed trader was ahead of your cancel
  - These fills have the worst markout by construction

  Conjecture: Model cancel-fill race as:
  P(fill_before_cancel | adverse_signal) > P(fill_before_cancel | no_signal)
  The AS cost of cancel-race fills is 2-3x normal fill AS. You should pre-compensate by:
  1. Reducing size on levels most likely to race-fill (nearest to mid)
  2. Adding cancel_latency_bps to the spread floor

  Iteration: Track race_fill_as_bps separately from normal_fill_as_bps. If ratio > 2x, add the excess to your spread floor.

  Hypothesis 9: Multiplicative Spread Multipliers Cause Cascade Widening

  Your signal integration compounds multipliers: informed × gating × cross_venue × staleness. Three independent 1.5x adjustments → 3.4x total. This is exponentially conservative.

  Conjecture: Switch to additive log-space (same pattern as your calibrated gamma):
  log(total_mult) = Σ log(mult_i)   // Already multiplicative
  // Replace with:
  total_adjustment_bps = Σ adjustment_i_bps   // Additive, bounded
  δ_final = δ_GLFT + total_adjustment_bps

  Iteration: In signal_integration.rs, convert multiplicative spread_mult to additive bps adjustments with a hard cap (e.g., max +20 bps total adjustment). This prevents the 4x widening cascade that makes you
  uncompetitive.

  Hypothesis 10: Quota-Aware Spread Optimization (Shadow Pricing)

  Your shadow price at 7% is ~50 bps but it's a binary veto (quote/don't quote). The optimal solution from queuing theory is a continuous shadow price that widens spreads proportionally:

  δ*(q, headroom) = δ_GLFT + λ_shadow / headroom

  where λ_shadow is the marginal value of one API request. When headroom is high, shadow price ≈ 0. When low, it smoothly widens spreads to reduce update frequency.

  Conjecture: This transforms the rate limit problem from a hard constraint (inventory forcing, backoff) to a soft cost (wider spreads, fewer but better quotes).

  Iteration: In quote_gate.rs, replace the tier system with:
  1. Compute λ_shadow = base_shadow / headroom_pct.max(0.01)
  2. Add to GLFT spread: δ_final = δ_GLFT + λ_shadow_bps
  3. Reduce ladder density: levels = max(1, (5.0 * headroom_pct) as usize)
  4. Never go one-sided — always maintain two-sided market

  ---
  IV. Priority-Ranked Iteration Plan
  ┌──────────┬────────────────────────────────────────────────┬──────────────────────────────────────┬────────────┐
  │ Priority │                   Hypothesis                   │           Expected Impact            │ Complexity │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P0       │ H4: Kill inventory-forcing, use wide two-sided │ +1.5 bps (eliminates whipsaw)        │ Low        │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P0       │ H10: Continuous shadow pricing                 │ +0.8 bps (smooth quota degradation)  │ Medium     │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P1       │ H7: Queue position estimation from L2          │ +1.0 bps (better paper calibration)  │ Medium     │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P1       │ H9: Additive spread adjustments                │ +0.5 bps (prevents cascade widening) │ Low        │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P1       │ H1: Funding-cycle time horizon                 │ +0.3 bps (better terminal behavior)  │ Medium     │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P2       │ H3: Realized PnL as TD reward                  │ +0.2 bps (better controller)         │ Medium     │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P2       │ H6: Log-odds AS integration                    │ +0.3 bps (principled AS pricing)     │ Medium     │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P2       │ H2: Quota basis functions in value fn          │ +0.2 bps (quota-aware decisions)     │ Low        │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P3       │ H5: Monopolist LP pricing                      │ +0.5 bps (market power extraction)   │ High       │
  ├──────────┼────────────────────────────────────────────────┼──────────────────────────────────────┼────────────┤
  │ P3       │ H8: Cancel-race AS tracking                    │ +0.2 bps (hidden AS detection)       │ Medium     │
  └──────────┴────────────────────────────────────────────────┴──────────────────────────────────────┴────────────┘
  The P0 items alone (H4 + H10) should recover ~2.3 bps — enough to flip live EV from -1.5 bps to positive. Want me to implement any of these?







  Here's the full picture of what happened.
                                                                                                                                                                                   
  The Session: 89 Minutes, -$5.02, Kill Switch              
                                                                                                                                                                                   
  HYPE dropped 188 bps during our session. We accumulated 12.81 HYPE long (4x our configured max of 3.24) in the first 7 minutes — 13 buys, 1 sell. Then the changepoint detector  
  went haywire, pulled ALL quotes for 78 of 89 minutes (89% of the session). We sat there with a massive long position, unable to reduce, while the market bled us dry. Kill switch
   fired at -$5.02.

  The 6 Bugs That Killed Us

  1. AS Measurement is Tautological — Both "adverse selection" and "spread depth" are computed from self.latest_mid at the same instant. They're algebraically identical. So
  realized_edge = 0 - 1.5 = -1.5 bps for 97% of fills. Every learning signal downstream (RL rewards, edge predictions, calibration) is trained on pure noise. We literally cannot
  measure our own edge.

  2. ZERO Directional Skew — Across 1,964 quote cycles, combined_skew_bps ≈ 0 always. No signal produces skew. The HL-native flow fallback has imbalance_30s: 0.0 hardcoded and
  avg_buy/sell_size: 0.0 not tracked. We quote perfectly symmetrically during sweeps. 24 buys vs 5 sells = textbook adverse selection from one-sided flow we never leaned away
  from.

  3. RegimeDetection is Connected But Output is Dead — Has live kappa data ranging 500-4203 (8.4x variation!) but spread_adjustment_bps = 0.0 always. The kappa_effective value is
  computed, stored, logged... and never consumed by the spread calculator. The signal sees regime changes and screams into the void.

  4. Emergency Pull Paralysis — 903 emergency pulls. The "safety" mechanism caused the loss. We accumulated 10+ HYPE long BEFORE the overlay kicked in, then it locked us out for
  78 minutes straight. We couldn't reduce position because the overlay cancelled ALL quotes (including the sells we desperately needed). Two brief 3-minute windows opened — and we
   bought MORE in both.

  5. Position Limit Not Enforced — Config says max_position: 3.24. We hit 12.81. Same effective_max_position bug from Feb 10 — derived from margin (~55 HYPE) instead of user
  config.

  6. InformedFlow is Actively Harmful — The one signal that actually does something has marginal value of -0.23 bps. It tightens spreads when p_informed < 0.05, which reduces our
  safety buffer right before we get run over.

  What's Missing in Feature Engineering

  - No directional intelligence whatsoever (no Binance for HYPE, no working flow-based skew)
  - Regime kappa data exists but is ignored (wiring bug)
  - No multi-timescale order flow (30s EWMA hardcoded to 0)
  - No OI delta or book depth velocity (both have TODO comments with = 0.0)
  - QuoteOutcomeTracker accumulates data but never feeds it back to the spread optimizer
  - Only 2 of 6 signals (InformedFlow, VPIN) produce any spread adjustment — and one is harmful

  Priority Fix List

  The full analysis is saved to .claude/plans/live-session-audit-findings.md. Top priorities:
  ┌──────┬────────────────────────────────────────────────┬────────────────────────────────────────────────────┐
  │  #   │                      Fix                       │                        Why                         │
  ├──────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ P0-1 │ Store mid_at_placement, fix EdgeSnapshot       │ Unlocks all learning — prerequisite for everything │
  ├──────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ P0-2 │ Emergency pull: let reduce-only quotes survive │ Prevents the exact loss pattern we just saw        │
  ├──────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ P0-3 │ Enforce config.max_position as hard cap        │ We hit 4x our limit                                │
  ├──────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ P0-4 │ Wire regime kappa to spread calculator         │ 8.4x kappa variation is being ignored              │
  ├──────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ P0-5 │ Fix HL flow feature vec (30s, sizes = 0)       │ Enables directional skew                           │
  ├──────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ P0-6 │ Disable InformedFlow tightening                │ One-line fix, +0.23 bps                            │
  └──────┴────────────────────────────────────────────────┴────────────────────────────────────────────────────┘
  Want me to start implementing these fixes?

  if you were designing systems to address your concerns, examine the current system and design a principled solution that would have emerged if the change was a foundational assumption from the start using a team of 5-6 engineers

  1. QUEUE-AWARE OVERRIDE: Orders successfully latched by matching are immediately cancelled by the queue-aware enhancement. See oid 4: matched at 0.0 bps, 2.1% size diff → latched →
  then QUEUE-AWARE OVERRIDE: forcing ask refresh (overriding latch!)
  2. Size volatility: Target sizes swing 20-100% between cycles (edge predictions fluctuate), exceeding the 30% latch_size_fraction.

  Modifies ARE happening now (17 cycles with modifies). The stability floor and Case 2b are working. But queue-aware override is the remaining churn driver
  