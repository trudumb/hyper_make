# RFC Addendum: Spread Extraction Extensions

**Status**: Draft  
**Depends on**: RFC Unified Adverse Selection Framework (Components 1-8)  
**Date**: 2026-02-19

---

## Motivation

The base RFC optimizes the COST side of market making — preventing adverse selection from destroying PnL. But the profit equation has two sides:

```
PnL = Σ (spread_earned × volume) - Σ (adverse_selection) - Σ (inventory_cost) - Σ (carry_cost)
          ↑ THIS IS UNOPTIMIZED            ↑ RFC Components 1-8 optimize these
```

The base RFC treats `mid(t)` as the fair value anchor. Every quote is placed relative to the exchange mid. The spread is computed from a single GLFT formula. Every fill is treated identically. The quote cycle runs on a fixed timer regardless of information arrival.

Each of these is leaving money on the table. This addendum introduces six extensions that extract more spread by improving the QUALITY of every fill through better fair value estimation, fill-conditional strategy, and information-aware quote lifecycle management.

The extensions are ordered by expected impact per unit of implementation complexity.

---

## Extension 1: Micro-Price Fair Value Estimator

### The Problem with Exchange Mid

The exchange mid is `(best_bid + best_ask) / 2`. On a tick-constrained market with 5 bps minimum tick width (like HYPE), this is always exactly 2.5 bps from either side. It contains zero sub-tick information.

But sub-tick information exists. When there is 10× more size on the bid than the ask, the next trade is overwhelmingly likely to be buyer-initiated at the ask. Fair value is closer to the ask than to the bid. The exchange mid ignores this completely.

Every basis point of fair value accuracy compounds directly into PnL. If your fair value is 0.8 bps more accurate than the exchange mid, every fill you take is 0.8 bps better priced. On a system doing 200+ fills per day, that is 160 bps of cumulative edge PER DAY from fair value alone.

### The Micro-Price

Build fair value in layers, each adding incremental accuracy:

**Layer 0: Exchange Mid (baseline)**
```
p₀ = (best_bid + best_ask) / 2
```

No sub-tick information. This is what everyone uses.

**Layer 1: Volume-Weighted Mid (Stoikov micro-price)**
```
p₁ = best_bid × (V_ask / (V_bid + V_ask)) + best_ask × (V_bid / (V_bid + V_ask))

where:
  V_bid = total visible size at best bid
  V_ask = total visible size at best ask
```

When V_bid >> V_ask, p₁ shifts toward the ask. Intuition: the thick bid is a wall that absorbs selling, so the next move is more likely upward. This gives sub-tick resolution even on a discrete price grid.

Empirically, this predicts the next mid-price change with R² ≈ 0.02-0.05 on crypto perps. Small, but pure edge.

**Layer 2: Multi-Depth Weighted**
```
p₂ = Σ_{d=1}^{D} w(d) × p₁(d)

where:
  p₁(d)  = volume-weighted mid at depth d
  w(d)   = fill_probability_weight(d) / Σ w    (normalized)
  fill_probability_weight(d) = κ · exp(-κ · depth_bps(d) / σ)
```

Deeper book levels contribute with exponentially decaying weight. A massive bid 50 bps below mid matters much less than a small bid at touch. The weights use the same fill probability model as the GLFT, ensuring consistency.

**Layer 3: Trade-Flow Augmented**
```
p₃ = p₂ + λ_trade × OFI × σ

where:
  OFI  = order flow imbalance (buy volume - sell volume) over trailing window
  λ_trade = regression coefficient from historical OFI → next-mid-change
  σ    = current volatility
```

Recent trade flow moves fair value. If the last 10 seconds saw $50k of buyer-initiated flow vs $10k seller-initiated, fair value is shifted upward. OFI is computed from the existing TradeFlowTracker — the infrastructure already exists.

**Layer 4: Drift-Augmented (integration with base RFC)**
```
p₄ = p₃ + μ̂ × τ_micro

where:
  μ̂       = Kalman/IMM drift estimate (Component 1)
  τ_micro = micro-price forecast horizon (1-3 seconds)
```

The drift estimate from the base RFC pushes the micro-price in the direction the market is trending. τ_micro is SHORT — this is a 1-3 second forecast, not the full τ_eff used for the reservation price. The micro-price says "where will mid be in 2 seconds?", the reservation price says "what is my indifference price given my risk?"

**Layer 5: Cross-Venue Informed**
```
p₅ = p₄ + β_cross × Δp_reference

where:
  Δp_reference = price change on reference venue (BTC on Binance, HYPE on CEX)
                  that hasn't yet propagated to our venue
  β_cross      = regression coefficient (our asset's response to reference moves)
```

When BTC drops 10 bps on Binance and HYPE hasn't moved yet on Hyperliquid, p₅ shifts downward immediately. This is the cross-asset lead signal from Component 7, but applied to fair value estimation rather than drift. The difference: drift affects the RESERVATION PRICE (and thus spread asymmetry), while the micro-price shift affects the ANCHOR (and thus the absolute level of all quotes).

### Integration with Base RFC

The reservation price equation changes from:

```
r(t, q) = mid(t) + drift_shift - inventory_penalty - funding_carry
```

to:

```
r(t, q) = p*(t) + drift_shift - inventory_penalty - funding_carry

where p*(t) = micro_price (layers 1-5)
```

The drift_shift term remains because it serves a different purpose: the micro-price is a SHORT-HORIZON fair value estimate (where will mid be in 1-3 seconds), while the drift_shift is a MEDIUM-HORIZON risk adjustment (what is the expected cost of holding inventory for the next 30-60 seconds).

### Measurable Edge

After building the micro-price, validate by computing:
```
MSE_micro = E[(mid(t+5s) - p*(t))²]
MSE_mid   = E[(mid(t+5s) - mid(t))²]

improvement = 1 - MSE_micro / MSE_mid
```

Each 1% improvement in MSE translates to approximately 0.2-0.5 bps of additional edge per fill. The micro-price should show 5-15% MSE improvement on crypto perps, yielding 1-7 bps of additional edge per fill.

---

## Extension 2: Markout Decomposition Engine

### What Markout Is

After every fill at time t, price p, side s (buy=+1, sell=-1):

```
markout(τ) = s × (mid(t + τ) - p)    for τ ∈ {1s, 5s, 10s, 30s, 60s, 300s}
```

Positive markout means the fill was profitable (we bought below where the market went, or sold above). Negative markout means adverse selection (we bought above where the market went).

### The Markout Curve

A single markout number at one horizon conflates multiple effects. The CURVE across horizons reveals the fill's true nature:

```
Shape A: [+2.5, +2.0, +1.5, +1.0, +0.5]  — Decaying Positive
  → Clean MM profit. We captured spread and it's slowly eroding.
  → Our spread was correctly calibrated for this market state.
  
Shape B: [+3.0, +1.0, -1.0, -4.0, -8.0]  — Positive Then Collapse
  → We looked good short-term, got destroyed long-term.
  → Slow information leak — our spread caught the transient but missed the trend.
  → Kalman drift should have been more aggressive.

Shape C: [-3.0, -5.0, -7.0, -9.0, -11.0]  — Monotonic Adverse
  → Pure toxic fill. Adversely selected from the first millisecond.
  → We should not have been quoting at this depth/size/state.
  → E[PnL] was wrong — the adverse move was underestimated.

Shape D: [-6.0, -4.0, -1.0, +1.0, +2.0]  — Dip Then Recovery
  → Temporary adverse impact that mean-reverts. Liquidation overshoot.
  → If we hold through the drawdown, this fill is profitable.
  → PATIENCE is the optimal strategy. Do NOT aggressively unwind.

Shape E: [+1.0, +1.0, +1.0, +1.0, +1.0]  — Flat Positive
  → We captured edge that persists. Likely a counterparty mispricing.
  → Our fair value estimate was better than the market's.
  → Micro-price working correctly.
```

### Decomposition

For each fill, decompose markout into additive components:

```
markout(τ) = realized_half_spread + temporary_impact(τ) + permanent_impact + noise(τ)

where:
  realized_half_spread = s × (mid(t) - p)         // edge at fill time (always positive if spread > 0)
  temporary_impact(τ)  = s × (mid(t+τ) - mid(t+∞))  // transient price move that reverts
  permanent_impact     = s × (mid(t+∞) - mid(t))   // irreversible adverse selection
  mid(t+∞)            ≈ mid(t + 300s)              // use 5-minute horizon as "permanent"
```

Track these components as rolling statistics. Report:

```
                      Mean    Std     Win%
Realized half-spread: +2.8    0.5     98%
Temporary impact:     -0.4    3.2     42%
Permanent impact:     -1.6    4.1     35%
Net markout (60s):    +0.8    5.0     56%
```

This tells you:
- Your SPREAD is correctly calibrated (mean half-spread of 2.8 bps)
- TEMPORARY impact is slightly negative (not a problem — it mean-reverts)
- PERMANENT impact is -1.6 bps (this is the adverse selection your features DIDN'T catch)
- Net markout is positive (system is profitable) but the 35% permanent-impact win rate means 65% of fills are adversely selected. Room to improve.

### Real-Time Fill Classification

Don't wait for the full 300s markout. Classify fills in real-time using the FIRST 1-5 seconds:

```
// Feature vector at time of fill
x = [
    fill_size / avg_fill_size,        // size anomaly
    book_imbalance_at_fill,            // book state
    hawkes_intensity_at_fill,          // clustering
    imm_crisis_probability,            // regime
    time_since_last_fill,              // fill clustering
    fill_depth_bps,                    // how deep the fill was
    trade_imbalance_1s_before_fill,    // preceding trade flow
    micro_price_displacement,          // was our quote stale?
]

// Fast logistic classifier
P(toxic) = sigmoid(β · x)

// Train on historical markout labels:
//   toxic = 1 if markout_60s < -2 bps
//   benign = 0 otherwise
```

The classifier uses features AVAILABLE AT FILL TIME (not future information). Train on historical fills where you have the full markout curve. The classifier's job is to predict the SHAPE of the markout curve from the initial conditions.

### Fill-Conditional Response

The classification triggers different post-fill strategies:

```
if P(toxic) > 0.7:                              // HIGH TOXICITY
    // Emergency: cancel remaining orders on filled side
    cancel_side(filled_side)
    // Shift reservation price by expected adverse move
    reservation_adjustment = -E[permanent_impact | toxic] × direction
    // Quote aggressively on reducing side (wider discount to unwind fast)
    reducing_spread_mult = 0.7                   // tighter reducing quotes
    // Update Kalman with strong bearish/bullish observation
    drift_estimator.update_emergency(direction, confidence=0.8)

elif P(toxic) < 0.3:                            // LOW TOXICITY
    // Expected mean reversion — hold position
    unwind_urgency = LOW
    // Potentially tighten reducing-side quotes to capture reversion
    reducing_spread_mult = 1.0                   // normal reducing quotes
    // The fill is expected to become more profitable as transient impact reverts
    // PATIENCE: don't aggressively unwind

elif 0.3 < P(toxic) < 0.5 and Shape ≈ D:       // LIQUIDATION PATTERN
    // Detected overshoot with expected recovery
    // Hold through drawdown, but set a stop-loss at permanent_impact_95th_percentile
    unwind_urgency = VERY_LOW
    max_hold_time = 120s                         // wait up to 2 min for reversion
    stop_loss_bps = permanent_impact_95th        // cut if exceeds this

else:                                            // UNCERTAIN
    // Normal operation, let the base RFC handle it
    unwind_urgency = NORMAL
```

This is the fill-conditional response the current system completely lacks. Every fill currently triggers the same behavior — update position, continue quoting. The extension makes the system react differently to DIFFERENT TYPES of fills.

### Feedback Loop to Spread Calibration

The markout engine creates a feedback loop:

```
1. Compute rolling average markout by state bin
2. States where avg_markout < target_markout → WIDEN spread in that state
3. States where avg_markout > target_markout → TIGHTEN spread in that state
4. Target markout = minimum_acceptable_edge (e.g., +0.5 bps at 60s)
```

The target is not zero (break-even). A market maker should have consistently positive markout — the question is HOW positive. Too positive means spreads are too wide and volume is being left on the table. Too close to zero means adverse selection is eating almost all the spread.

The optimal target depends on risk aversion, but a principled starting point is:
```
target_markout(60s) = γ_base × σ × sqrt(τ_holding)
```

This is the risk premium — you should earn AT LEAST the risk cost of holding inventory.

---

## Extension 3: Quote Staleness Model

### Why Staleness Matters

Your quotes are computed at time t₀. They are valid at t₀. At t₀ + ε, new information has arrived (an L2 update, a trade, a cross-venue move) and your quotes are no longer optimal. The longer a quote sits, the more likely it is mispriced — and a mispriced resting order is a free option for informed traders.

On a 5-6 second quote cycle, your quotes are stale for approximately 80-90% of their lifetime. Every fill that occurs in the last 3 seconds of a cycle has a higher probability of being adversely selected than a fill in the first second.

This is not hypothetical. Compute the empirical markout by time-since-last-update:

```
Expected result:
  Fills at t₀ + 0-1s: markout ≈ +2.0 bps (fresh quotes, good fills)
  Fills at t₀ + 1-2s: markout ≈ +1.0 bps
  Fills at t₀ + 2-3s: markout ≈ +0.2 bps
  Fills at t₀ + 3-4s: markout ≈ -0.5 bps (stale quotes, adverse fills)
  Fills at t₀ + 4-5s: markout ≈ -1.5 bps
```

If this gradient is steep, you're losing significant edge to staleness. The fix is not simply "update faster" (there are rate limits and latency constraints). The fix is to MODEL staleness and act on it.

### Staleness as a Continuous Variable

```
freshness(t) = exp(-Σ information_events(t₀, t))

where each event contributes:
  L2 book change (touch):     w_l2_touch × |Δsize| / avg_size
  L2 book change (deep):      w_l2_deep × |Δsize| / avg_size
  Trade (same venue):         w_trade × trade_size / avg_trade_size
  Cross-venue price change:   w_cross × |Δp_cross| / σ
  Funding rate change:        w_funding × |Δf| / σ_f
  
freshness ∈ (0, 1]: 1 = perfectly fresh, → 0 = completely stale
```

The weights w_* are calibrated from markout data: events that predict worse markout get higher weights. This is a supervised learning problem:
```
Target: markout_10s for fills at different freshness levels
Features: information events between quote placement and fill
Learn: which events cause the most staleness
```

### Staleness-Adjusted E[PnL]

Fold staleness into the per-level expected PnL computation from the base RFC:

```
E[PnL](δ, side, freshness) = freshness × E[PnL_fresh](δ, side) 
                            + (1 - freshness) × E[PnL_stale](δ, side)

where:
  E[PnL_fresh] = the base RFC E[PnL] (assumes quotes are correctly priced)
  E[PnL_stale] = -E[adverse_move_from_stale_info]    (negative by definition)
```

When freshness is high (just updated), E[PnL] is dominated by the RFC formula. When freshness is low (old quotes, lots of new info), E[PnL] shifts negative. The quote's optimal size naturally tapers to zero as it becomes stale.

This eliminates a whole class of adverse selection without widening spreads: you're not quoting wider, you're quoting LESS when your information is stale. When you DO quote, your quotes are well-calibrated.

### Event-Driven Reprice Trigger

Instead of repricing on a fixed cycle, trigger repricing when the information state warrants it:

```
if freshness < freshness_floor:
    trigger_immediate_reprice()

// Dynamic freshness floor based on regime:
freshness_floor = 0.3  (quiet regime)
freshness_floor = 0.6  (trending regime — tighter freshness required)
freshness_floor = 0.8  (crisis regime — almost continuous repricing)
```

In quiet markets, let quotes age — there's minimal staleness cost. In crisis, reprice aggressively on every significant information event. The IMM regime probabilities from Phase 1b drive the freshness floor.

But even without event-driven repricing (if rate limits prevent it), the staleness-adjusted E[PnL] provides protection: stale quotes have their effective size reduced to zero, so even if you CAN'T cancel them, you've already reduced their exposure.

### Asymmetric Staleness

Staleness is not symmetric between sides. When the market moves down:

```
Bids: freshness DROPS (our bids are too high relative to new fair value)
Asks: freshness STAYS HIGH or INCREASES (our asks are correctly priced or better)
```

A system that tracks per-side staleness can:
1. Cancel only the stale side (bids) while keeping the fresh side (asks)
2. Reduce only bid sizes while maintaining ask sizes
3. Prioritize repricing the stale side over the fresh side

This asymmetric response is FASTER than repricing both sides: you cut your exposure in half in one operation instead of needing to recompute everything.

---

## Extension 4: Counterparty Flow Classification

### The Insight

Not all fills are created equal. A fill from a liquidation engine has DIFFERENT information content than a fill from an arbitrage bot, which is different from a fill from retail flow. The current system treats them identically.

On Hyperliquid specifically:
- **Liquidation fills**: detectable from OI changes coinciding with fills, or from vault liquidation events. Extremely toxic short-term (the liquidation cascades), but price often overshoots by 20-50% and mean-reverts within 60-120 seconds. Shape D markout.
- **Arb fills**: detectable from cross-venue correlation (fills arrive within milliseconds of price moves on other venues). Permanently informative — no mean-reversion. Shape C markout. Avoid completely.
- **Other MM fills**: detectable from fill pattern (regular sizes, symmetric, time-of-day dependent). Low toxicity — these are inventory management trades from competitors. Shape A markout.
- **Retail/noise fills**: detectable from size distribution (small, irregular) and timing (often during social media-driven spikes). Very low toxicity, slight positive signal (retail tends to buy at the top). Shape A or E markout.
- **Whale accumulation**: detectable from sustained one-sided flow with iceberg-like patterns (many fills of similar size at regular intervals). Permanently informative at longer horizons but doesn't cause immediate adverse moves. Shape B markout — positive short-term, adverse long-term.

### Bayesian Flow Type Posterior

Maintain a real-time posterior over flow types:

```
Flow state: θ ∈ {Liquidation, Arb, OtherMM, Retail, Whale}

Prior: P(θ) updated from historical frequency (e.g., Liquidation: 5%, Arb: 15%, OtherMM: 30%, Retail: 40%, Whale: 10%)

Observations per fill:
  x₁ = fill_size / median_fill_size                    (arb and whale use characteristic sizes)
  x₂ = time_since_last_same_side_fill                  (arb is bursty, retail is sporadic)
  x₃ = cross_venue_correlation_at_fill_time             (arb: high, others: low)
  x₄ = OI_change_coinciding_with_fill                   (liquidation: OI drops)
  x₅ = recent_fill_count_same_side / recent_fill_count  (one-sided = whale or liquidation)
  x₆ = fill_size_coefficient_of_variation_recent         (whale: low CV, retail: high CV)
  x₇ = bid_ask_sweep_count_recent                        (liquidation: high sweep count)

Likelihood: P(x | θ) modeled as class-conditional distributions learned from labeled historical fills

Posterior: P(θ | x₁:ₙ) ∝ P(x₁:ₙ | θ) × P(θ)   (updated after each fill)
```

The posterior doesn't need to be certain. Even a noisy classification (60% confidence) is valuable because the COSTS of different flow types are dramatically different.

### Flow-Conditional Quoting

The flow type posterior modifies both the reservation price and the E[PnL] computation:

```
// Expected adverse move is a MIXTURE over flow types
E[adverse_move] = Σ_θ  P(θ) × E[adverse_move | θ]

where:
  E[adverse_move | Liquidation] = large_short_term BUT negative_long_term (mean-reverts)
  E[adverse_move | Arb]         = permanent, proportional to cross-venue displacement
  E[adverse_move | OtherMM]     = near_zero
  E[adverse_move | Retail]      = slightly_negative (contrarian signal)
  E[adverse_move | Whale]       = moderate_positive, slow-acting

// Expected holding time is also flow-conditional
E[τ_holding | θ] = {
    Liquidation: 90s      (hold through reversion)
    Arb:         5s       (unwind immediately)
    OtherMM:     20s      (normal)
    Retail:      30s      (patient)
    Whale:       10s      (unwind quickly — permanent information)
}
```

The flow-conditional E[adverse_move] feeds directly into the E[PnL] per level from the base RFC. When P(Arb) is high, E[adverse_move] is large and permanent → E[PnL] goes negative at all depths → quotes pull to zero. When P(Retail) is high, E[adverse_move] is near zero → E[PnL] is maximally positive → quotes tighten to capture volume.

### Spread Extraction from Flow Classification

The edge: you quote TIGHTER than competitors when retail flow is dominant (capturing volume they miss), and WIDER when informed flow is dominant (avoiding fills they take). Over time, your fill mix is systematically less toxic.

Quantitative estimate: if 40% of fills are retail with -0.5 bps adverse selection and 15% are arb with -8 bps adverse selection, and you can avoid 50% of arb fills through wider quotes while capturing 20% more retail through tighter quotes:

```
Current: 0.40 × (-0.5) + 0.15 × (-8.0) + ... = -1.4 bps average AS
Improved: 0.48 × (-0.5) + 0.075 × (-8.0) + ... = -0.84 bps average AS
Delta: +0.56 bps per fill
```

At 200 fills/day, that is 112 bps/day of additional edge from fill quality alone.

---

## Extension 5: Inventory Turnover / Holding Time Optimization

### The Problem

The base RFC treats τ_holding as exogenous — an estimated time to unwind one unit. But τ_holding is endogenous: it depends on how aggressively you quote the reducing side, which is a control variable.

The tradeoff:

```
Aggressive unwind (tight reducing quotes):
  + Low holding cost (short τ_holding)
  + Low adverse selection risk (position exits quickly)
  - Low spread earned on reducing side (tight quotes = less edge per unwind fill)
  - Higher market impact (aggressive pricing competes with your own interest)

Patient unwind (wide reducing quotes):
  + High spread earned per unwind fill
  + Natural mean-reversion may do the work for free
  - High holding cost (long τ_holding)
  - High adverse selection risk (trend may continue)
  - Funding cost accumulates
```

This is a Bellman equation. The optimal unwind speed depends on:
- Drift estimate μ̂ and regime (if trend is against you, unwind fast)
- Funding rate f̂ (if carry is expensive, unwind fast)
- Volatility σ (if vol is high, the option value of waiting is higher)
- Markout curve shape (if fills show Shape D — reversion — wait)
- Inventory level (if near capacity, unwind fast regardless)

### The Bellman Formulation

State: (q, μ̂, P(regime), σ, f̂)
Action: reducing_spread_adjustment ∈ [-3, +3] bps (tighten or widen reducing quotes)
Reward: spread_earned - holding_cost - adverse_selection_cost

```
V(q, state) = max_a { 
    E[PnL(a, state)] + β × E[V(q', state') | q, a, state]
}

where:
  q'      = q - fill_size (if reducing fill occurs with probability P_fill(a))
  state'  = updated state after one time step
  β       = discount factor (≈ 0.999 for 5s cycles)
  
  E[PnL(a, state)] = P_fill(a) × (reducing_spread + a) - |q| × (γ(q)σ²Δt + f̂Δt)
```

Solving the full Bellman is computationally expensive. Approximate with a lookup table over discretized states:

```
Bins:
  q:       [-4, -2, -1, 0, 1, 2, 4]  (7 inventory levels)
  regime:  [Quiet, Trending, Crisis]    (3 regimes)
  funding: [Paying, Neutral, Receiving] (3 funding states)
  
Total: 7 × 3 × 3 = 63 state bins
Per-bin: optimal reducing_spread_adjustment ∈ [-3, +3] bps

Table solved offline via value iteration, refreshed daily.
```

In crisis regime with high funding and large inventory → table says: unwind aggressively (-3 bps, i.e., tighten reducing quotes by 3 bps below GLFT optimal).

In quiet regime with negative funding and moderate inventory → table says: be patient (+2 bps, i.e., widen reducing quotes above GLFT optimal and wait for mean reversion or favorable unwind).

### Integration with Base RFC

The Bellman output is a single number: `reducing_spread_adjustment_bps`. This modifies the E[PnL] filter on the reducing side:

```
// In E[PnL] computation for reducing-side quotes:
effective_depth = glft_depth + reducing_spread_adjustment

// Negative adjustment = tighter quotes = faster unwind = less edge per fill
// Positive adjustment = wider quotes = slower unwind = more edge per fill
```

The base RFC handles WHICH SIDE to reduce on (via drift and inventory penalty). This extension optimizes HOW AGGRESSIVELY to reduce.

---

## Extension 6: Adaptive Quote Cycle and Level Spacing

### Adaptive Cycle Timing

The current system runs a fixed 5-6 second quote cycle. This is a blunt instrument:

In quiet markets: 5 seconds is fine. Low information arrival. Quotes stay fresh.
In volatile markets: 5 seconds is an eternity. By 2 seconds, quotes are stale. By 4 seconds, every fill is adverse.

The quote staleness model (Extension 3) already tells you when quotes are stale. Use this to DRIVE the cycle timing:

```
next_cycle_time = base_cycle_time × freshness_scaling(regime)

where:
  freshness_scaling(Quiet)    = 1.5   (extend to 7-8 seconds — save rate limit budget)
  freshness_scaling(Trending) = 0.7   (compress to 3-4 seconds)
  freshness_scaling(Crisis)   = 0.3   (compress to 1.5-2 seconds — reprice rapidly)
```

In crisis, you spend your rate limit budget on rapid repricing when it matters most. In quiet markets, you conserve it. Over a session, the total number of quote updates may be the same, but they're concentrated where they extract the most value.

Additionally, within a cycle, trigger IMMEDIATE repricing on specific events:

```
Event-driven reprice triggers:
  1. Cross-venue price move > 2σ in < 1 second
  2. Touch-level size drops below 20% of previous cycle's level
  3. Fill on OUR book at any level
  4. IMM regime transition probability > 0.3
  5. Freshness drops below floor for current regime
```

These are not additional scheduled cycles — they are interrupt-driven responses to information events. The system handles the event, reprices the affected side, and returns to the normal cycle.

### Adaptive Level Spacing

The base RFC uses geometric spacing between quote levels. But optimal spacing depends on the BOOK SHAPE:

```
Geometric spacing: levels at d, d×r, d×r², d×r³, ...
  → Equal spacing in log-price. Ignores the actual book topology.

Adaptive spacing: levels positioned at LOCAL book gaps where your fill probability is highest.
```

**The Gap Exploitation Algorithm:**

```
fn compute_adaptive_levels(book: &L2Book, num_levels: usize, side: Side) -> Vec<f64> {
    let depths = book.price_levels(side, max_depth_bps=100);
    
    // Compute "crowdedness" at each depth: how much OTHER size exists
    let crowd = depths.iter().map(|d| book.size_within(d - ε, d + ε, side)).collect();
    
    // Compute "gap score" at each depth: inverse of crowdedness
    // High gap score = few other orders = high marginal fill probability for us
    let gap_score = crowd.iter().map(|c| 1.0 / (c + floor)).collect();
    
    // Weight gap score by fill probability (deeper = less likely to fill)
    let weighted = gap_score.iter().zip(depths.iter())
        .map(|(g, d)| g * fill_probability(*d))
        .collect();
    
    // Select top-N depths by weighted gap score
    let mut best_depths: Vec<f64> = weighted.top_n(num_levels);
    best_depths.sort();
    best_depths
}
```

The intuition: if there are 50 other orders at 5 bps depth but zero orders at 7 bps depth, your marginal fill probability at 7 bps is MUCH higher than at 5 bps (you'd be alone in the queue at 7 bps, vs last-in-queue at 5 bps). The extra 2 bps of depth costs you 2 bps of edge per fill, but the dramatically higher fill probability more than compensates.

This requires L2 book data, which the system already ingests. The computation is O(n log n) in the number of price levels — well within the 1ms hot-tier latency budget.

**Queue Position Awareness:**

On price-time priority books, your effective fill probability depends on queue position:

```
P(fill | depth, queue_position) = P(price_reached) × P(our_turn | price_reached)

P(our_turn | price_reached) ≈ 1 - (size_ahead_of_us / total_size_at_level)
```

If you're last in a queue of 500 contracts at the best bid, P(our_turn) is near zero even if the bid is likely to trade. You'd earn more by placing at the next level where you'd be first in queue.

The system should track its own queue position at each level (inferable from order placement time and L2 updates) and factor this into E[PnL]:

```
E[PnL_queue_adjusted](level) = P(fill | queue_pos) × edge - (1 - P(fill)) × opportunity_cost

opportunity_cost = E[PnL] at the best alternative level we COULD be quoting at
```

If being 500th in queue at level 1 gives E[PnL] of 0.01 bps, but being 1st in queue at level 2 gives E[PnL] of 0.8 bps, the capital is better deployed at level 2.

---

## Extension Interactions and Composition

These six extensions are not independent. They create compounding interactions:

```
Micro-price (Ext 1) → better fair value anchor
  → better markout measurement (Ext 2) because markout is relative to fair value
    → better fill classification because markout shapes are cleaner
      → better flow-conditional strategy (Ext 4) because classification is more accurate
        → better holding time optimization (Ext 5) because AS estimates are flow-conditional

Staleness model (Ext 3) → knows when quotes are stale
  → adaptive cycle timing (Ext 6) reprices when staleness is high
    → fills occur on FRESHER quotes → better markout → better calibration
    
Flow classification (Ext 4) → knows WHO is trading
  → informs staleness model (Ext 3) because arb flow creates staleness faster than retail
    → informs holding time (Ext 5) because optimal patience depends on counterparty
```

The extensions should be implemented in order (1 → 2 → 3 → 4 → 5 → 6) because each subsequent extension benefits from the ones before it. But Extensions 1-3 can be implemented before the base RFC is complete — they are additive improvements that work with the existing system.

---

## Measurable Outcomes Per Extension

| Extension | Metric | Expected Improvement | Measurement |
|---|---|---|---|
| 1. Micro-price | Mid prediction MSE | -10 to -15% | Compare MSE of p* vs exchange mid for 5s-ahead prediction |
| 2. Markout decomp | Net markout at 60s | +0.5 to +1.5 bps per fill | Before/after average markout with fill-conditional strategy |
| 3. Staleness | Adverse fill rate | -20 to -30% | Fraction of fills with negative 10s markout |
| 4. Flow classification | Average AS cost | -0.5 to -1.0 bps per fill | Weighted average permanent impact across fills |
| 5. Holding time | Inventory half-life | -15 to -25% | Time to reduce position by 50% after accumulation |
| 6. Adaptive spacing | Fill rate per level | +10 to +30% | Number of fills per quote-second at comparable depths |

Compound effect: if each extension contributes 0.5-1.0 bps of edge per fill, the full stack adds 3-6 bps per fill. At 200 fills/day on a $30 contract, that is $18-36/day of additional PnL — meaningful relative to current daily PnL of ~$5-15.

---

## Implementation Priority

```
HIGHEST IMPACT / LOWEST COMPLEXITY:
  Extension 1 (Micro-price)    — Uses existing book data. Pure math. Immediate edge.
  Extension 2 (Markout engine) — Logging infrastructure. Enables everything else.

HIGH IMPACT / MEDIUM COMPLEXITY:
  Extension 3 (Staleness)      — Requires per-event tracking. High ROI.
  Extension 4 (Flow classify)  — Requires labeled data. Builds over time.

MEDIUM IMPACT / HIGH COMPLEXITY:
  Extension 5 (Holding time)   — Bellman solver. Requires Extension 4.
  Extension 6 (Adaptive)       — Queue estimation. Requires L2 book analysis.
```

Extension 2 (markout) is the enabling infrastructure. Without it, you cannot validate any other extension. Build the markout engine FIRST, even before the micro-price, because it tells you whether any change you make is actually improving fill quality.
