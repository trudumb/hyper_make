# RFC Addendum: Proprietary Spread Extraction Algorithms

**Status**: Draft  
**Depends on**: RFC Unified Adverse Selection Framework (Components 1-8), IMM Drift Estimator (Phase 1b)  
**Date**: 2026-02-19

---

## Why the Base RFC Is Necessary but Insufficient

The base RFC solves a defensive problem: stop accumulating toxic inventory. It does this well. But it treats the spread itself as a given — the GLFT computes "optimal" spread from κ and γ, and every quote earns that spread minus adverse selection.

This is the same framework every market maker uses. The GLFT is public. The Kalman filter is public. The only differentiation is parameter calibration, and parameter calibration converges — everyone using the same model on the same data gets the same parameters.

Real edge comes from modeling things competitors DO NOT MODEL. On thick, anonymous venues (Binance BTC-USDT, CME ES), there's little exploitable structure — the market is efficient and the competition is fierce. On HIP-3 thin perps on Hyperliquid, there is enormous exploitable structure that exists because of specific properties of this venue:

1. **Full transparency**: every order, cancel, and modification is visible on-chain
2. **Thin books**: 3-5 active participants, your flow is 30-60% of depth
3. **Identifiable participants**: on thin books, individual actors are distinguishable by behavioral fingerprint
4. **Mechanism-driven flow**: hourly funding settlements create predictable, exploitable microstructure
5. **Observable liquidation thresholds**: margin requirements are public, OI is visible, liquidation cascades are computable in advance
6. **Wide latency spread**: 50ms to 30+ seconds between fastest and slowest participants — information propagates in observable waves

Each property creates a specific algorithmic opportunity that DOES NOT EXIST on traditional venues.

---

## Algorithm 1: Liquidation Frontier Mapping

### The Mechanism

On Hyperliquid, every leveraged position has a liquidation price determined by entry price, leverage, and maintenance margin. When the mark price crosses this threshold, the position is forcibly closed by the liquidation engine. These forced closures are market orders — they hit existing bids (for long liquidations) or asks (for short liquidations) at whatever price is available.

On thin HIP-3 books, a single liquidation can represent 20-50% of total book depth. The liquidation itself moves the price, potentially triggering MORE liquidations. This is the cascade mechanism.

The critical insight: **liquidation prices are not random. They are computable from observable data.**

### The Liquidation Density Function

Build a probability distribution over liquidation trigger prices:

```
ρ_long(p) = expected long liquidation volume at price p
ρ_short(p) = expected short liquidation volume at price p
```

**Inputs (all observable on Hyperliquid):**

```
1. Aggregate open interest by direction (long OI, short OI)
2. Mark price history (infer entry price distribution from OI changes)
3. Funding rate history (infer average leverage from funding participation)
4. Maintenance margin requirements (public, per-asset)
```

**Construction:**

When OI increases at time t at price p(t), new positions were opened near p(t). Store these as entries in a running distribution:

```
entry_distribution: Vec<(entry_price, size, direction, timestamp)>

// When long OI increases by ΔOI at mark price p:
entry_distribution.push((p, ΔOI, Long, t))

// When long OI decreases by ΔOI (voluntary close or liquidation):
// Remove oldest entries first (FIFO approximation) or proportionally
entry_distribution.remove_proportional(ΔOI, Long)
```

For each surviving entry, compute liquidation price:

```
// For a long position entered at p_entry with leverage L:
p_liq_long = p_entry × (1 - 1/L + maintenance_margin_rate)

// For a short position:
p_liq_short = p_entry × (1 + 1/L - maintenance_margin_rate)
```

Leverage L is not directly observable per-position. Estimate from:
- Average leverage implied by funding rate magnitude (high funding → high avg leverage)
- Cross-sectional distribution from historical liquidation events (backtest which leverage assumptions best predict actual liquidation cascades)
- Conservative assumption: use the LOWEST common leverage (3-5x) to compute the NEAREST liquidation frontier, then higher leverages for the deeper frontier

The density function is:

```
ρ_long(p) = Σ_{entries where p_liq(entry) ≈ p} entry.size
```

Discretize to 1-bps bins. Update every quote cycle from OI feed.

### Quoting Strategy Around the Frontier

When ρ_long(p) is large at some price level p_cluster below current price:

```
Current price: 30.00
Liquidation cluster: 29.50 (-167 bps), estimated 50 contracts

Scenarios:
  A) Price stays above 29.50 → cluster never triggers → normal MM operation
  B) Price drops to 29.50 → 50 contracts of forced selling hit the book
     → Price drops through the cluster, potentially to 29.20-29.30
     → Mean-reversion after cascade clears (selling pressure gone)
     → Price recovers to 29.40-29.50 within 60-120 seconds
```

**Pre-cascade positioning (price approaching cluster):**

```
// As price approaches within 2× the cluster's expected impact distance:
approach_distance_bps = (current_price - cluster_price) / current_price × 10000
cluster_impact_bps = estimated_impact(cluster_size, book_depth_at_cluster)

if approach_distance_bps < 2 × cluster_impact_bps:
    // Pull bids from the cascade zone
    // Reason: fills in this zone are maximally toxic short-term
    for level in bid_levels:
        if level.price >= cluster_price - cluster_impact_bps:
            level.size = 0  // don't provide liquidity INTO the cascade
    
    // Place bids BELOW the cascade zone
    // Reason: post-cascade mean-reversion means these fills are Shape D 
    // (short-term adverse, long-term profitable)
    overshoot_bid_price = cluster_price - cluster_impact_bps × 1.5
    overshoot_bid_size = max_size  // absorb the overshoot
    
    // Place asks IN the cascade zone
    // Reason: during the cascade, panicked sellers will pay maximum premium
    cascade_ask_price = cluster_price + 5bps  // just above the liquidation trigger
    cascade_ask_size = available_size
```

**During cascade (liquidations actively firing):**

```
// Detected by: rapid OI decrease + price approaching cluster + high sell flow
if is_cascade_active(oi_change_rate, price, cluster_price):
    // Bids only at overshoot levels (below cascade)
    // Asks aggressive (close to current mid, profit from panic)
    // DO NOT place bids within the cascade — they will be swept instantly
    
    // Key signal: cascade is ENDING when OI decrease rate slows
    cascade_exhaustion = d(ΔOI)/dt  // second derivative of OI
    if cascade_exhaustion > 0:  // rate of liquidation is DECREASING
        // Begin placing bids closer to current price
        // The worst is over; captures the recovery
```

**Post-cascade (recovery):**

```
// Cascade complete. We hold long inventory from overshoot bids.
// Flow type: liquidation → Shape D markout → HOLD for reversion
unwind_urgency = VERY_LOW
max_hold_time = 120s
// Wait for recovery. Place asks at pre-cascade price levels.
```

### Why This Is Proprietary

This algorithm CANNOT exist on traditional venues because:
- Liquidation thresholds are private on Binance/FTX/Bybit
- OI by direction isn't available at sufficient granularity on CME
- Book depth on major venues is too deep for a single liquidation to cascade

On HIP-3, all three conditions are met: liquidations are mechanism-driven, OI is observable, and books are thin enough that liquidation volume is material.

### Expected Edge

Historically, liquidation cascades on thin perps overshoot by 30-80 bps and recover 40-60% within 120 seconds. If you capture 50 bps of overshoot on a 2-contract fill, that's $0.30 per cascade event. With 2-3 cascade events per day on thin HIP-3 assets, that's $0.60-0.90/day — significant relative to current daily PnL.

---

## Algorithm 2: Information Propagation Function

### The Physics of Price Discovery

When an exogenous event occurs (BTC moves on Binance, macro news, whale trade on a CEX), the information propagates through the order book in waves. On Binance BTC-USDT, this propagation happens in microseconds — too fast to exploit. On HIP-3 thin perps, it happens over SECONDS TO MINUTES. The waves are individually observable.

Model the information state of the market as a function of time since the event:

```
I(t) ∈ [0, 1]

I(0)  = 0     // no information incorporated immediately after event
I(∞)  = 1     // fully incorporated at long horizon
I(t)  = 1 - exp(-Σ_k λ_k × max(0, t - τ_k))

where:
  k = participant class (arb_bot, other_mm, algo_trader, retail)
  λ_k = information incorporation RATE per class
  τ_k = reaction DELAY per class (latency floor)
```

**Empirical estimation:** for each cross-venue price move (BTC/ETH moves on Binance that exceed 3σ):

```
1. Mark event time t_event
2. Track HYPE book mid, order flow, and fills from t_event to t_event + 300s
3. Compute: I_hat(t) = |Δmid_hype(t)| / |Δmid_hype(300s)|
   (what fraction of the total move has occurred by time t?)
4. Fit I_hat(t) to the multi-class propagation model
5. Extract λ_k and τ_k per participant class
```

Expected calibration on HIP-3:

```
Class           τ (reaction delay)    λ (incorporation rate)
Arb bots        50-200ms              10-50/sec (fast)
Other MMs       200ms-2s              1-5/sec (moderate)
Algo traders    1-5s                  0.5-2/sec
Retail          5-30s                 0.1-0.5/sec
```

### How This Creates Edge

**Fill toxicity as a function of I(t):**

When a fill arrives at your book, its toxicity depends on WHERE in the propagation curve it falls:

```
fill_toxicity(t_fill) ≈ 1 - I(t_fill - t_last_event)
```

A fill at I=0.05 (very early, 95% of information not yet incorporated) is filled by someone who KNOWS. Extremely toxic. A fill at I=0.90 (late, most information incorporated) is filled by someone LATE. Low toxicity — they're paying the new correct price.

**Spread as a function of I(t):**

In the immediate aftermath of an event (I low), quote wide — fills are maximally toxic. As I rises toward 1, tighten — fills become safe.

```
spread_multiplier(t) = 1 + spread_alpha × (1 - I(t))²

where spread_alpha ≈ 2-5 (calibrate from markout data segmented by I(t))
```

This is NOT the same as "widen when vol is high." Vol may not spike at all if the event is a gradual BTC drift. But the propagation function STILL rises from 0 to 1 — there's still a period where HYPE hasn't caught up to BTC. The widening is driven by INFORMATION STATE, not volatility.

**Temporal arbitrage:**

When I(t) < 0.3 (early propagation), you have superior information if your cross-venue feed is faster than most participants'. You can:

1. SHIFT your reservation price in the direction of the event BEFORE other MMs have repriced
2. Your "old" quotes on the CORRECT side (if BTC dropped: your asks) become VALUABLE because they're priced at the old, higher level
3. Your "old" quotes on the WRONG side (bids) need to be cancelled IMMEDIATELY

The propagation function tells you HOW MUCH TIME YOU HAVE to cancel the wrong side. If τ_other_mm ≈ 1 second and your latency is 200ms, you have 800ms to cancel before the OTHER MM's repricing makes your stale quotes the best available price to hit.

```
cancel_urgency = max(0, τ_next_fast_participant - our_latency) / τ_next_fast_participant
// cancel_urgency ∈ [0, 1]: 1 = we have plenty of time, 0 = no time, they're already repricing
```

### The Propagation Function as a Regime Indicator

The SHAPE of I(t) varies by event type:

```
Sharp event (flash crash, liquidation cascade):
  I(t) → step function. I jumps from 0 to 0.8 in < 1 second.
  → Everyone processes simultaneously. No temporal arb. WIDEN immediately.

Gradual event (BTC slow drift, narrative shift):
  I(t) → slow sigmoid. I takes 30-60 seconds to reach 0.8.
  → Large temporal arb window. EXPLOIT the delay.

Information conflict (contradictory signals):
  I(t) → non-monotonic. I rises then FALLS as market re-evaluates.
  → Initial reaction was wrong. Fade the move. Mean reversion opportunity.
```

The second derivative of I(t) tells you whether the market is STILL incorporating information (I'' < 0, concave, slowing down) or ACCELERATING its incorporation (I'' > 0, convex, speeding up). Acceleration means more participants are reacting, which means the move has more conviction.

---

## Algorithm 3: Adversarial Participant Fingerprinting

### Why This Works on HIP-3

On Binance BTC perps, there are thousands of active participants and the L2 book aggregates them anonymously. On HIP-3 thin perps, there are 3-5 active market makers and a handful of regular traders. L2 book changes of 0.5 contracts at a specific price are likely from ONE entity. The book is sparse enough to decompose.

### Fingerprinting via L2 Book Diffs

Every L2 book update is a diff: size added or removed at a specific price. Track these diffs as a time series:

```
BookEvent {
    timestamp_ms: u64,
    side: BidOrAsk,
    price: f64,
    size_delta: f64,     // positive = added, negative = removed
    resulting_size: f64,  // total size at this level after the change
}
```

Cluster events into "participant sessions" using behavioral features:

```
Feature vector per event cluster:
  f1: characteristic_sizes       // histogram of |size_delta| values
  f2: depth_profile              // which depths (bps from mid) are quoted
  f3: update_frequency           // inter-event time distribution
  f4: symmetry                   // ratio of bid events to ask events
  f5: response_latency           // time from trade/book event to this participant's reaction
  f6: cancel_pattern             // does participant cancel-and-replace or modify in place?
  f7: time_of_day_activity       // activity distribution across hours
  f8: size_scaling_with_vol      // does size change when vol changes? how?
```

**Clustering algorithm:** online DBSCAN or streaming Gaussian mixture on the feature vectors. With only 3-5 active MMs, even a simple k-means with k=5 will separate them.

**Validation:** when a participant is fingerprinted, their FUTURE events should cluster consistently. If cluster assignment is stable over hours/days, the fingerprint is real.

### What You Learn From Each Participant

**A. Their quote cycle timing:**
```
Participant A: updates every 4.2 ± 0.3 seconds
Participant B: updates every 8.0 ± 1.0 seconds  
Participant C: event-driven (no fixed cycle, responds to fills/book changes)

Your cycle: 5-6 seconds

Exploitation:
- Participant B's quotes are maximally stale at t ≈ 7.5s into their cycle
- If their cycle phase is trackable (it is, from the timestamps of their events),
  you know WHEN their quotes are stale
- Tighten your quotes during their stale window to capture fills they'd otherwise get
- Widen back when they reprice (their fresh quotes are competitive)
```

**B. Their spread function:**
```
Participant A: always quotes at 5/10/15 bps
Participant B: quotes at 3/7/12 bps in quiet, widens to 8/15/25 bps when vol spikes

Your spread: adaptive via GLFT

Exploitation:
- When Participant B widens (you see their book change), they're signaling
  that they perceive higher risk. This IS information — their model
  is telling them something. Feed this as a drift observation:
  
  z_competitor_widen = -direction × (new_spread_B - old_spread_B) / old_spread_B
  R_competitor = σ_competitor²
  
  A 50% spread widening by the most sophisticated competitor is a strong signal.
  Stronger than most microstructure features because it reflects THEIR processed
  view of all THEIR features.
```

**C. Their inventory management:**
```
If Participant A's bid sizes shrink while ask sizes grow, they're accumulating
long and trying to unwind. This tells you:
  1. They've been buying (likely informed? or just unlucky?)
  2. Their model says they should unwind (bearish on forward drift?)
  3. Their asks will be more aggressive (competition for unwinding)

If you're ALSO long, their aggressive asks compete with yours for unwind fills.
If you're FLAT, their aggressive asks are cheap inventory for you.
```

**D. Withdrawal detection (most valuable):**

```
Normal state: Participant A quotes 6 bid + 6 ask levels, total 15 contracts per side
Withdrawal:   Participant A drops to 2 bid + 2 ask levels, total 3 contracts per side

This is a LEADING INDICATOR. They pulled liquidity because their model says 
conditions are dangerous. This happens BEFORE the adverse move, not after.

Signal:
  participant_liquidity_withdrawal = Σ_participants (normal_depth - current_depth) / normal_depth
  
  When this exceeds 0.5 (participants have pulled >50% of typical liquidity),
  the book is fragile. Any flow will cause outsized price impact.
  
  Response:
  - Widen spreads (less competition anyway since others pulled)
  - Reduce sizes (book is thin, impact is large)
  - Increase IMM crisis probability directly:
    w_crisis += withdrawal_signal × coupling_strength
```

### Competitive Response Modeling

Once you've fingerprinted competitors, model their RESPONSE FUNCTION:

```
For each competitor c, estimate:
  Δquote_c(t+dt) = f(book_state(t), fill_event(t), price_change(t))

This is a lightweight regression:
  Δspread_c = β₀ + β₁×vol_change + β₂×book_imbalance_change + β₃×fill_on_c_side + ε
  Δdepth_c  = β₀ + β₁×position_proxy + β₂×spread_change + ε

Calibrate β per participant from historical L2 data.
```

With the response function, you can PREDICT their next quote before they make it. This is a 1-5 second forecast window — enough to position your own quotes to either:
- Undercut them when they're about to widen (capture fills before they re-enter)
- Step back when they're about to tighten (avoid being picked off by their fresh quotes)
- Place at gaps you predict they'll leave

---

## Algorithm 4: Endogenous Impact and Large-Player Optimal Control

### Why GLFT Is Wrong on Thin Books

The GLFT assumes the market maker is infinitesimal — your trades don't affect mid. The optimal quote sizes are computed independently per level. Each fill is evaluated as if it doesn't change the book.

On HIP-3, you might be 40-60% of the bid-side depth. When you place a 1.0 contract bid, you've just increased total bid depth by 30%. When that bid fills, you've removed 30% of bid depth. Mid moves.

**Quantifying the distortion:**

```
Standard GLFT assumes: Δmid(fill) = 0
Reality on thin book:  Δmid(fill) ≈ α × (fill_size / total_depth_same_side)^β × σ

Calibrate α, β from YOUR historical fills:
  For each fill at time t:
    impact = mid(t + 1s) - mid(t)  (signed)
    x = fill_size / book_depth_at_fill_time
  Regress: |impact| ~ α × x^β × σ
```

Expected values on HIP-3 thin perps: α ≈ 0.3-0.8, β ≈ 0.5-1.0.

For a 1.0 contract fill on a book with 3.0 total depth: impact ≈ 0.5 × (1.0/3.0)^0.7 × σ ≈ 0.23σ. At σ = 50 bps, that is ~12 bps of self-impact PER FILL. This is 2-3x larger than your spread. Ignoring it is catastrophic.

### Impact-Aware E[PnL]

Replace the base RFC's E[PnL]:

```
// Base RFC (assumes zero impact):
E[PnL](δ, s) = P(fill) × [edge - AS_cost - carry]

// Impact-aware:
E[PnL](δ, s) = P(fill) × [edge - AS_cost - carry - SELF_IMPACT(s) - ROUND_TRIP_IMPACT(s)]

where:
  SELF_IMPACT(s)       = α × (s / book_depth)^β × σ    // our fill moves mid against us
  ROUND_TRIP_IMPACT(s) = α × (s / projected_reducing_depth)^β × σ  // unwinding moves mid against us too

  projected_reducing_depth = book_depth_opposite_side - our_size_opposite_side
                           // depth available for unwinding, EXCLUDING our own quotes
```

The ROUND_TRIP_IMPACT is the insight most models miss. When you buy 1.0 contract, you must eventually sell it. The selling side's depth — excluding your own asks (which you can't fill against yourself) — determines the impact of unwinding. If the opposite side has 4.0 total depth but 2.0 of that is YOUR asks, the effective unwind depth is only 2.0. Your unwind impact is twice what it appears.

### Inter-Level Coupling

On thick books, fills at level 1 don't affect the economics of level 2. On thin books, they do:

```
If your bid at level 1 fills (1.0 contract), then:
  - Book depth at level 1: reduced by 1.0
  - Mid: shifted up by impact(1.0)
  - Your bid at level 2: now CLOSER to mid than planned (because mid moved toward it)
  - Level 2's fill probability: INCREASED
  - Level 2's edge: DECREASED (it's closer to the new mid)
  - Level 2's impact: INCREASED (book is thinner after level 1 fill)
```

The cascade effect: level 1 fill makes level 2 MORE LIKELY to fill but LESS PROFITABLE per fill. If level 2 fills, it makes level 3 even more likely and even less profitable.

**Backward induction solution:**

```
// Solve from deepest level inward
for level in (num_levels-1)..=0:
    // Conditional on all deeper levels having filled:
    cumulative_fills = Σ_{j > level} s_j
    post_fill_book_depth = initial_depth - cumulative_fills - self_impact_on_depth(cumulative_fills)
    post_fill_mid = initial_mid + cumulative_impact(cumulative_fills)
    
    // Effective edge at this level accounts for the moved mid
    effective_edge = abs(level_price - post_fill_mid)
    
    // Impact of THIS level's fill (on a thinner book)
    this_impact = impact_function(s_level, post_fill_book_depth)
    
    // E[PnL] for this level conditional on deeper fills
    conditional_epnl = P(fill_level | deeper_fills) × (effective_edge - AS - this_impact - carry)
    
    // Optimal size is smaller at shallower levels because the impact cost is higher
    // (the book has been thinned by deeper fills)
    s_level = argmax { s × conditional_epnl(s) - risk_cost(s) }
```

This naturally produces a CONCAVE size profile: largest at the deepest level (where impact is smallest), smallest at touch (where impact is largest). The standard geometric size profile is wrong on thin books.

### Your Quotes AS the Book

A second-order effect: by placing or removing quotes, you change the book topology that other participants observe and react to.

```
Scenario: you place a 2.0 contract bid at 5 bps depth
  - Other participants see thicker bid side → shift their fair value estimate UP
  - They tighten their asks (thinking the bid is "real" support)
  - Their tighter asks benefit you (cheaper to unwind longs)
  - You've improved your unwind cost by placing a bid

Conversely: you remove your bid
  - Book thins → other participants widen
  - Less competition for your asks
```

This is Stackelberg-type strategic interaction. You're the first mover (place quotes), others respond. The optimal quote placement ACCOUNTS for their expected response:

```
E[PnL_strategic](s, δ) = E[PnL_myopic](s, δ) + E[Δ_competitor_quotes(s, δ)] × value_of_tighter_opposite

where:
  E[Δ_competitor_quotes] = how competitors adjust when they see our quote placement
  value_of_tighter_opposite = how much their adjustment benefits our inventory management
```

This is hard to estimate precisely but can be approximated: large bids cause other MMs to tighten asks by ε bps on average. If you're long and want to unwind, strategically placing BIDS (which you don't expect to fill) that cause others to tighten their asks improves your unwind price.

---

## Algorithm 5: Funding Settlement Microstructure Exploitation

### The Structural Clock

Hyperliquid funding settles hourly. This is a DETERMINISTIC clock signal that creates PREDICTABLE flow:

```
Time relative to settlement:
  t = -600s to -300s:  Traders begin adjusting positions to manage funding exposure
  t = -300s to -60s:   Flow intensifies. Payers (wrong side) close positions.
  t = -60s to 0s:      Maximum urgency. Last-second closes.
  t = 0s:              Settlement. Funding exchanged.
  t = 0s to +120s:     Reversion. Traders re-enter positions at "reset" prices.
  t = +120s to +600s:  Normal trading resumes.
```

The flow direction is KNOWN from the funding rate:
- Positive funding (longs pay shorts): pre-settlement selling, post-settlement buying
- Negative funding (shorts pay longs): pre-settlement buying, post-settlement selling

This is not a statistical pattern. It is a mechanism consequence. As long as funding exists with hourly settlements, this flow exists.

### Settlement-Aware Quote Schedule

```
fn settlement_quote_adjustment(
    time_to_settlement_s: f64,
    funding_rate: f64,
    funding_zscore: f64,
) -> SettlementAdjustment {
    let payment_direction = if funding_rate > 0.0 { Sell } else { Buy };
    let intensity = settlement_flow_intensity(time_to_settlement_s);
    let magnitude = funding_zscore.abs().min(3.0) / 3.0; // normalized [0,1]
    
    // Pre-settlement: widen quotes on the side getting hit, tighten on the other
    if time_to_settlement_s > 0.0 && time_to_settlement_s < 600.0 {
        // Sellers are desperate (positive funding) — charge premium on bids
        // Our bids: WIDEN (sellers pay more to close)
        // Our asks: TIGHTEN (we want to be short going into settlement)
        let widen_bps = intensity × magnitude × max_widen_bps;
        let tighten_bps = intensity × magnitude × max_tighten_bps;
        
        match payment_direction {
            Sell => SettlementAdjustment {
                bid_spread_add_bps: widen_bps,    // bids wider: sellers pay more
                ask_spread_add_bps: -tighten_bps, // asks tighter: we accumulate short
                target_inventory_bias: -1.0,       // we want to be short at settlement
            },
            Buy => SettlementAdjustment {
                bid_spread_add_bps: -tighten_bps,
                ask_spread_add_bps: widen_bps,
                target_inventory_bias: 1.0,        // we want to be long at settlement
            },
        }
    }
    // Post-settlement: reversion window
    else if time_to_settlement_s < 0.0 && time_to_settlement_s > -180.0 {
        // Flow reverses. Traders re-enter positions.
        // Our inventory from pre-settlement is now on the RIGHT side of the reversion.
        // Unwind patiently — we're being paid by the reversion flow.
        SettlementAdjustment {
            bid_spread_add_bps: 0.0,
            ask_spread_add_bps: 0.0,
            target_inventory_bias: 0.0,  // unwind toward flat
            unwind_patience: HIGH,
        }
    }
    else {
        SettlementAdjustment::neutral()
    }
}

fn settlement_flow_intensity(time_to_settlement_s: f64) -> f64 {
    // Empirically calibrated intensity curve
    // Peaks at settlement, ramps from ~600s before
    if time_to_settlement_s <= 0.0 { return 0.0; }
    let x = time_to_settlement_s / 600.0; // normalized [0, 1], 1 = far from settlement
    (1.0 - x).powi(2) // quadratic ramp: slow start, accelerating toward settlement
}
```

### Directional Positioning via Funding Prediction

Go further: predict the funding rate itself, not just react to it.

```
Next-hour funding rate is a function of:
  1. Current premium/discount (perp price vs index)
  2. Current OI and its rate of change
  3. Recent funding history (momentum)
  4. Time-of-day effects (Asian hours vs US hours have different avg funding)

f̂_next = β₀ + β₁×premium + β₂×ΔOI + β₃×f_current + β₄×tod_effect
```

If you predict f̂_next is strongly positive (longs will pay) AND current price is near a liquidation cluster (Algorithm 1), the combined signal is:

```
Expected sequence:
  1. High funding → longs start closing → selling pressure
  2. Price drops toward liquidation cluster
  3. Cascade triggers → price overshoots
  4. Post-cascade + post-settlement: double reversion
  
Optimal response:
  - Be SHORT entering the settlement window
  - Hold short through the cascade
  - Cover (buy) at the overshoot bottom
  - Net profit: funding received + cascade spread + reversion capture
```

This combines Algorithms 1 and 5 into a single strategy that exploits both the liquidation mechanism and the funding mechanism simultaneously.

---

## Algorithm 6: Volatility Surface Extraction (MM as Short Options Portfolio)

### Reframing the Market Maker

Every resting quote is an option you've written:

```
Resting bid at price B, size s:
  = Short put with strike B, notional s
  = You've committed to buy s units at B if the market drops there
  = "Premium" = spread from mid to B
  = "Payoff" = adverse selection if filled

Resting ask at price A, size s:
  = Short call with strike A, notional s
  
Full ladder (bids + asks at multiple levels):
  = Short straddle/strangle portfolio
  = Multiple strikes (levels), multiple sizes
```

The "premium" you collect is the spread. The "realized volatility" is the adverse selection. This is not a metaphor — it is a mathematically precise equivalence.

### Computing Your Implied Volatility

The Black-Scholes value of a short option at depth δ bps with size s and time-to-reprice τ:

```
option_value = s × σ_implied × sqrt(τ / (2π))    // ATM approximation for short straddle

where τ = time until next reprice (your quote cycle length, ~5s)
```

The spread you charge (in bps) IS the implied volatility:

```
σ_implied = spread_bps / (C × sqrt(τ_cycle))

where C = sqrt(2/π) × normalization_factor ≈ 0.80 for ATM straddle

For spread = 5 bps, τ_cycle = 5s:
  σ_implied = 5 / (0.80 × sqrt(5/31536000)) ≈ 5 / (0.80 × 0.000398) ≈ 15,700 bps annual

This seems large, but it's correct: a 5 bps spread over a 5-second window implies ENORMOUS annualized vol because the window is tiny.
```

The meaningful comparison is σ_implied vs σ_realized over the SAME window:

```
σ_realized_5s = realized volatility measured over 5-second windows

vol_edge = σ_implied - σ_realized_5s
```

When vol_edge > 0: you're charging MORE than realized vol. You're extracting vol premium. Good.
When vol_edge < 0: you're charging LESS than realized vol. Every fill is a losing option trade. Widen.

### Dynamic Spread from Vol Edge

Instead of the GLFT's static formula, compute spread directly from the options framework:

```
min_spread_bps = C × σ_realized_5s × sqrt(τ_cycle)    // break-even spread
target_spread_bps = min_spread + vol_premium_target

where vol_premium_target = target edge per trade in bps
```

This adapts FASTER than the GLFT to vol changes because σ_realized_5s updates every 5 seconds (one window's worth of data). The GLFT uses σ estimated over 300s, which lags.

### The Greeks of Your Quote Portfolio

Compute the Greeks of your outstanding quote portfolio:

```
Delta = Σ (bid_sizes) - Σ (ask_sizes)    // net directional exposure
        + Σ (bid_fill_prob × bid_sizes) - Σ (ask_fill_prob × ask_sizes)  // expected fills
        
Gamma = Σ (d(fill_prob)/dp × sizes)     // how fast delta changes with price
        ≈ proportional to total size at touch (highest gamma at touch)

Theta = Σ (spread_per_level × fill_prob × size) / τ_cycle   // time decay = spread income per second

Vega  = d(portfolio_value) / d(σ)
        // positive: benefits from vol increase (wider spreads more than compensate for AS)
        // negative: harmed by vol increase (AS increases faster than spread income)
```

**The key insight: you want positive Theta (earning spread) and controlled Gamma (limited adverse selection exposure).**

The GLFT optimizes Theta naively (maximize spread earned). But it doesn't control Gamma. On thin books, Gamma is concentrated at touch, meaning a small price move causes large adverse selection on your touch-level orders.

**Gamma management strategy:**

```
// Current: large size at touch, smaller size deeper
// Problem: maximum gamma (adverse selection exposure) at the most dangerous level

// Better: reduce size at touch, increase size at depth
// This has the SAME total exposure but lower gamma
// Fewer fills at touch (where they're most toxic) and more at depth (where they're safer)

// Compute gamma-optimal size profile:
for level in levels:
    fill_prob = kappa × exp(-kappa × depth / sigma)
    gamma_contribution = d(fill_prob) / d(price) × size
    
    // Constrain: total_gamma < gamma_budget
    // Maximize: total_theta (spread × fill_prob × size) subject to gamma constraint
    
    // This is a convex optimization: maximize Θ subject to Γ ≤ Γ_max
    // Solution: Lagrange multiplier λ_gamma
    // s*(level) = Θ(level) / (risk_cost + λ_gamma × Γ(level))
```

The gamma-optimal size profile is FLATTER than the standard profile: less concentrated at touch, more distributed across depth. This earns slightly less spread per fill (deeper fills have less edge) but has dramatically lower adverse selection (because Gamma is lower).

### Regime-Conditional Vol Extraction

The vol_edge varies by IMM regime:

```
Quiet regime:
  σ_realized ≈ low and stable
  σ_implied (from spread) ≈ fixed at GLFT level
  vol_edge ≈ positive (GLFT spread overestimates vol in quiet markets)
  Strategy: TIGHTEN spreads to capture more volume at still-positive vol edge

Trending regime:
  σ_realized ≈ elevated but predictable (drift-dominated)
  vol_edge ≈ variable
  Strategy: SKEW is more important than level. The drift makes one side high-vol and the other low-vol.
    - Accumulating side: σ_realized is HIGH (drift + diffusion). Widen.
    - Reducing side: σ_realized is LOWER (drift works in your favor). Tighten.

Crisis regime:
  σ_realized ≈ spiking, potentially 5-10× quiet level
  vol_edge ≈ NEGATIVE (your spread hasn't caught up to realized vol)
  Strategy: WIDEN aggressively until vol_edge is positive again. 
    Pull quotes entirely if vol_edge < -X (you're giving away free options).
```

The vol framework tells you EXACTLY when to pull quotes: when your spread implies a lower volatility than what's being realized. This is a principled substitute for the killed quote gate — instead of "pull at 50% inventory," it's "pull when you're short vol at negative edge."

---

## Algorithm 7: Growth-Optimal Sizing (Non-Ergodic Kelly)

### Why γ Is Arbitrary

The GLFT's risk aversion parameter γ controls everything: spread width, size, inventory penalty. But γ has no physical interpretation. It's calibrated to "feel right" — low enough to quote tight, high enough to not blow up.

The Kelly criterion provides a PRINCIPLED sizing framework where the size comes from the growth rate of capital, not an arbitrary aversion parameter.

### The Non-Ergodic Problem

Standard Kelly assumes the time average equals the ensemble average. For a market maker with finite capital taking correlated fills, this is wrong.

Consider: your 6 bid levels can ALL fill in a cascade. That's 6 correlated losing trades in seconds. The ensemble average (expectation over many independent fills) is positive. The time-average (what actually happens to your ONE account) includes the possibility that 6 fills in 5 seconds wipes out a week of spread income.

The time-average growth rate for correlated fills:

```
g = E[ln(W(t+1) / W(t))]
  ≈ Σ_i P(fill_i) × (edge_i / W) - (1/2W²) × Σ_{i,j} Cov(fill_i, fill_j) × s_i × s_j

where:
  W = current capital
  edge_i = expected PnL of fill at level i
  Cov(fill_i, fill_j) = covariance between fill outcomes at levels i and j
  s_i = size at level i
```

The COVARIANCE term is what most models ignore. When fills are independent, Cov(i,j) = 0 for i≠j and the standard Kelly applies. When fills are correlated (cascade), the covariance matrix is dense and the quadratic penalty is MUCH larger.

### Fill Correlation Structure

On HIP-3 with cascade risk, the fill correlation matrix has structure:

```
         Level 1  Level 2  Level 3  Level 4  Level 5  Level 6
Level 1 [  1.0    0.7      0.5      0.3      0.2      0.1  ]
Level 2 [  0.7    1.0      0.7      0.5      0.3      0.2  ]
Level 3 [  0.5    0.7      1.0      0.7      0.5      0.3  ]
...

// Adjacent levels are highly correlated (if level 1 fills, level 2 probably fills too)
// Distant levels are weakly correlated
// In crisis: ALL correlations approach 1.0 (full cascade)
```

Estimate the correlation matrix from historical fill data:

```
For each pair of levels (i, j):
  ρ(i,j) = P(fill_j within 10s | fill_i) / P(fill_j unconditional)

In quiet regime: ρ(1,2) ≈ 0.3 (level 2 fill is 30% more likely given level 1 fill)
In crisis: ρ(1,2) ≈ 0.9 (cascade — fills are near-certain to chain)
```

### Kelly-Optimal Sizing

The optimal total exposure that maximizes growth rate:

```
S* = (edge_vector)ᵀ × Σ⁻¹ × ones / (ones^T × Σ⁻¹ × ones)
   = (edge per fill) / (portfolio variance per unit size)

where:
  edge_vector = [edge_1, edge_2, ..., edge_N]
  Σ = covariance matrix of fill outcomes (includes correlation + impact)
  S* = optimal total notional exposure across all levels
```

In quiet regime with low correlation: S* is large (diversification across levels provides safety)
In crisis with high correlation: S* collapses (no diversification benefit — all fills are the same bet)

### Fractional Kelly via Regime

Apply a regime-dependent Kelly fraction:

```
f(regime) = {
    Quiet:    0.8-1.0  // near-full Kelly, fills are independent
    Trending: 0.5-0.7  // fills correlated along trend, reduce
    Crisis:   0.2-0.4  // fills maximally correlated, aggressive reduction
}

actual_total_exposure = f(regime) × S*_kelly
```

Distribute across levels using the growth-optimal profile:

```
s_level_i = actual_total_exposure × w_i

where w_i = optimal weight from quadratic optimization:
  max Σ w_i × edge_i  subject to  w^T Σ w ≤ target_variance
```

### Replacing γ

This REPLACES the continuous γ(q) from the base RFC as the sizing mechanism:

```
// Old (base RFC): γ(q) controls spread and size. Arbitrary.
// New: Kelly sizing controls total exposure. Principled.

// The effective γ IS the Kelly sizing constraint:
γ_effective(regime, correlation) = W / S*_kelly(regime, correlation)

// This γ is DERIVED from growth rate optimization, not set by hand.
// It automatically:
//   - Increases (wider spread, smaller size) in crisis (correlation high)
//   - Decreases (tighter spread, larger size) in quiet (correlation low)
//   - Depends on capital W (larger account → more exposure)
//   - Depends on edge per fill (higher edge → more exposure)
```

---

## Composition: How the Seven Algorithms Interact

```
                Liquidation Frontier (1)
                        │
                        ▼
              Predictable flow zones
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
    Funding Clock (5)  Propagation (2)  Participant Model (3)
            │           │               │
            ▼           ▼               ▼
    Settlement-aware   Fill toxicity   Competitor signal
    inventory bias     by I(t)         extraction
            │           │               │
            └─────┬─────┘───────────────┘
                  ▼
          Impact-Aware E[PnL] (4)
                  │
                  ▼
          Vol Surface Extraction (6)
          γ_eff from Kelly Sizing (7)
                  │
                  ▼
            OPTIMAL QUOTES
          (price, size, timing per level)
```

The interaction chain:
1. **Liquidation frontier** identifies WHERE dangerous flow will come from
2. **Funding clock** identifies WHEN the flow will arrive
3. **Propagation function** identifies HOW FAST information incorporates after an event
4. **Participant model** identifies WHO is trading and what their behavior reveals
5. **Impact model** correctly prices the cost of being a large player on a thin book
6. **Vol surface** ensures you're always charging MORE than realized vol (positive edge)
7. **Kelly sizing** ensures your total exposure is growth-optimal given correlation and regime

No single algorithm is sufficient. The combination creates a system that exploits the specific structure of HIP-3 thin perps in ways that don't exist on traditional venues.

---

## Implementation Priority

```
HIGHEST STRUCTURAL EDGE:
  Algorithm 1 (Liquidation Frontier)  — Mechanism-driven, deterministic, unique to Hyperliquid
  Algorithm 5 (Funding Settlement)    — Clock signal, structural, reliable

HIGHEST INFORMATIONAL EDGE:
  Algorithm 2 (Propagation Function)  — Causal model most competitors don't have
  Algorithm 3 (Participant Fingerprint) — Only possible on thin books

HIGHEST PRICING EDGE:
  Algorithm 4 (Endogenous Impact)     — Corrects a fundamental GLFT assumption
  Algorithm 6 (Vol Surface)           — Principled spread calibration
  Algorithm 7 (Kelly Sizing)          — Principled position sizing

RECOMMENDED ORDER:
  Phase 1: Algorithms 4 + 6 (impact model + vol surface)
    → Fixes the biggest pricing errors in the current system
    → Immediate PnL improvement from correct sizing on thin books
    
  Phase 2: Algorithms 1 + 5 (liquidation frontier + funding clock)
    → Adds structural edge that compounds with pricing improvements
    → Requires OI feed (Algorithm 1) and funding prediction model (Algorithm 5)
    
  Phase 3: Algorithms 2 + 3 (propagation + fingerprinting)
    → Informational edge that requires observation data to accumulate
    → Start data collection in Phase 1, deploy models in Phase 3
    
  Phase 4: Algorithm 7 (Kelly sizing)
    → Requires fill correlation data from Phases 1-3
    → Replace γ after the other algorithms are providing calibrated inputs
```
