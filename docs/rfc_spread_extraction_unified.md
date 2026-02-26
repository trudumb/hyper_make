# RFC: Spread Extraction Architecture

**Status**: Draft  
**Depends on**: RFC Unified Adverse Selection Framework (Components 1-8), IMM Drift Estimator (Phase 1b)  
**Date**: 2026-02-19

---

## Architecture

The base RFC solves the defensive problem: don't accumulate toxic inventory. It wires signals into the reservation price so the system retreats before damage is done. That's necessary. It's not sufficient.

The profit equation:

```
PnL = Σ(spread_earned × fill_quality × volume) - Σ(adverse_selection) - Σ(carry) - Σ(impact)
       ↑ unoptimized                              ↑ base RFC
```

This document optimizes the revenue side. It does so through two layers:

**Layer A — Foundational.** Six structural corrections to assumptions the GLFT makes that are wrong on any market. These are venue-agnostic, first-principles improvements to fair value estimation, fill pricing, spread calibration, parameter adaptation, position sizing, and inventory management.

**Layer B — Specialization.** Market-structure-specific algorithms that plug into the Layer A interfaces when the venue's properties support them. These activate on thin books, transparent order flow, mechanism-driven flow patterns, or wide latency spreads. They degrade gracefully — when the market is too thick or too fast, the specializations contribute nothing and the foundational layer runs standalone.

The layers are not independent. Each Layer B specialization maps to a specific Layer A component as an additional observation, a refined parameter estimate, or a structural correction. The interface between layers is always one of three types:

```
Type 1: OBSERVATION    — Specialization provides (z, R) to a filter or estimator
Type 2: PARAMETER      — Specialization refines a parameter used by a foundation component
Type 3: STATE          — Specialization provides a state variable to a control problem
```

---

## Foundation 1: Latent Fair Value Filter

### The Problem

The GLFT takes exchange mid as fair value. The base RFC improves this to `reservation = mid + drift_shift - inventory_penalty - funding_carry`, but mid is still the anchor.

Mid is not fair value. It is a noisy, discrete, manipulable observation of fair value. The true fair value V(t) is a continuous latent variable that no one observes directly. Everything observable — the book, trade flow, cross-venue prices, your own fills — is a noisy function of V(t). The exchange mid is one such observation, and not the best one.

When V(t) sits between two ticks, mid snaps to the NBBO midpoint and stays there until a tick boundary is crossed. V(t) may drift through a 4 bps range with mid showing zero movement. Every system anchored to mid is blind to sub-tick dynamics.

### The Model

State vector:

```
x = [V, μ, log(σ)]

V      = latent fair value (price level)
μ      = instantaneous drift (rate of V change)
log(σ) = log-volatility (stochastic vol, ensures σ > 0)
```

Dynamics:

```
dV      = μ dt + σ dW₁
dμ      = -θ_μ · μ dt + σ_μ dW₂                          (OU mean-reversion on drift)
d(logσ) = -θ_σ · (logσ - logσ̄) dt + σ_σ dW₃              (OU mean-reversion on log-vol)
```

V drifts with rate μ and diffuses with intensity σ. Both μ and σ are themselves stochastic processes. The system nests the base RFC's Kalman drift estimator as the special case where V is observed perfectly (V = mid) and σ is constant.

### Observations

Every observation is a noisy function of the latent state. The filter's job is to fuse them:

```
Observation 1: Exchange mid
  z_mid = V + ε_mid
  R_mid = (tick_size / 2)² + spoofing_noise²
  // Coarse observation, at least half-tick discretization noise

Observation 2: Book imbalance (multi-depth)
  z_bim = α_bim × (V - mid) + ε_bim
  // BIM predicts sub-tick displacement of V from mid
  // When V > mid, bids heavier than asks → positive BIM
  // α_bim calibrated from: regress BIM on subsequent mid-changes

Observation 3: Trade flow
  z_ofi = α_ofi × μ + ε_ofi
  // Net buy flow observes μ (drift), not V directly

Observation 4: Cross-venue price
  z_cross = V + basis + ε_cross
  // Observes V with basis offset and venue-specific noise

Observation 5: Own fills (Glosten-Milgrom — see Foundation 2)
  z_fill = V + adversarial_shift × side
  // Asymmetric observation — a fill reveals V is beyond your quote

Observation 6: Feature pipeline (base RFC Components 7-8)
  z_feature_k = α_k × μ + ε_k    for each feature k
  // Each feature from the base RFC is an observation of drift μ
  // The IMM/UKF replaces the scalar Kalman as the recipient
```

### Filter Implementation

The system is nonlinear (stochastic vol, adversarial fill model). Two options:

**UKF with IMM wrapper (recommended):** Run 3 UKFs (Quiet/Trending/Crisis from Phase 1b), each with different process noise parameters. The IMM mode probabilities weight the outputs. Gives particle-filter flexibility with Kalman stability.

**Particle filter:** N=200-500 particles. Handles arbitrary nonlinearity and multi-modal posteriors (regime switches). Higher variance, higher computational cost. Use if UKF proves insufficient.

### What Changes in the Base RFC

```
// Old:
reservation = mid + drift_shift(μ̂_kalman) - inventory_penalty - funding_carry

// New:
reservation = V̂ + drift_shift(μ̂) - inventory_penalty - funding_carry

// V̂ and μ̂ come from the SAME filter (both are state variables).
// drift_shift may still apply if μ̂ is used for risk adjustment over a
// longer horizon than the filter's update rate.
```

### Specialization 1A: Participant Signal Extraction

**Activation condition:** Book is thin enough to decompose L2 changes into individual participants (typically < 10 active quoters at any depth).

**Type:** OBSERVATION into the latent value filter.

On thin books, L2 changes of characteristic size at characteristic depths are attributable to specific participants. When a fingerprinted participant widens their spread, it is an observation about V:

```
Observation 7: Competitor spread change
  z_competitor = -direction × Δspread_c / σ
  R_competitor = σ²_competitor / confidence(c)

  // Participant c widening by 5 bps is bearish information if c is sophisticated.
  // confidence(c) = historical accuracy of c's spread changes as a predictor.
```

When a participant WITHDRAWS liquidity (drops from 6 levels to 2), this is a high-R observation with extreme z — equivalent to a large but uncertain signal about regime change:

```
Observation 8: Liquidity withdrawal
  withdrawal_score = Σ_c (normal_depth_c - current_depth_c) / normal_depth_c
  z_withdrawal = -withdrawal_score × scale
  R_withdrawal = σ²_withdrawal / (n_withdrawing_participants)
  
  // Multiple participants withdrawing simultaneously: R shrinks (higher confidence)
  // Single participant: R large (could be idiosyncratic)
```

These observations flow into the same filter as everything else. On thick books where participants aren't identifiable, these observations don't exist — the filter runs on Observations 1-6 alone with no degradation.

**Fingerprinting method:**

Each participant has a behavioral signature extracted from L2 book diff events:

```
Features per event cluster:
  characteristic_sizes:    histogram of |size_delta| values
  depth_profile:           which depths (bps from mid) are quoted
  update_frequency:        inter-event time distribution
  symmetry:                bid/ask event ratio
  response_latency:        time from trade/book event to participant's reaction
  cancel_pattern:          cancel-and-replace vs modify-in-place
```

Online clustering (streaming GMM or DBSCAN) with k=3-10 clusters. Stable assignment over hours validates the fingerprint. The clustering runs passively, collecting data from first boot. No minimum data requirement — the filter's R for competitor observations starts large (low confidence) and shrinks as the fingerprint stabilizes.

**Competitive response modeling:**

Once fingerprinted, model each competitor's response function:

```
Δspread_c(t+dt) = β₀ + β₁ × vol_change + β₂ × book_imbalance_change + β₃ × fill_event + ε
```

Predict their next quote before they make it. 1-5 second forecast window — enough to position your own quotes to either undercut them when they're about to widen (capture fills before they re-enter) or step back when they're about to tighten (avoid being picked off by their fresh quotes).

### Specialization 1B: Information Propagation Function

**Activation condition:** Cross-venue latency spread between fastest and slowest participants is > 500ms (i.e., information propagation is observable on human timescales).

**Type:** PARAMETER — refines the observation noise R for all observations as a function of information state.

When an exogenous event occurs (BTC moves on Binance, macro news), information propagates through the book in waves. On thick fast markets this completes in microseconds. On thin or slow markets it takes seconds to minutes, and the waves are individually observable.

Model the information state:

```
I(t) ∈ [0, 1]

I(0)  = 0         // no information incorporated immediately after event
I(∞)  = 1         // fully incorporated at long horizon
I(t)  = 1 - exp(-Σ_k λ_k × max(0, t - τ_k))

where:
  k = participant class (arb_bot, other_mm, algo_trader, retail)
  λ_k = information incorporation RATE per class
  τ_k = reaction DELAY per class (latency floor)
```

Calibrate from cross-venue events: for each reference-venue price move exceeding 3σ, measure what fraction of the total local-venue move has occurred by time t.

**How I(t) modifies the filter:**

When I(t) is low (information just arrived, hasn't propagated), ALL observations are noisier — they reflect the old state, not the new state. The propagation function scales observation noise:

```
R_effective(obs, t) = R_base(obs) / max(I(t), I_floor)

// When I(t) ≈ 0.1 (early propagation), R is 10× larger → observations less trusted
// When I(t) ≈ 0.9 (nearly absorbed), R is close to base → observations trusted
// I_floor prevents division by zero (e.g., 0.05)
```

This is principled: when the market hasn't yet incorporated new information, the observations you're seeing (mid, BIM, trade flow) are stale relative to the true state. The filter should trust them less.

**How I(t) modifies spread:**

The adversarial learning wrapper (Foundation 4) sees fill quality degrade when I(t) is low. But the propagation function provides a CAUSAL mechanism for direct spread adjustment:

```
spread_multiplier(t) = 1 + α_prop × (1 - I(t))²

// When I(t) low: widen. Fills are maximally toxic (counterparty has information you don't).
// When I(t) high: tighten. Information absorbed, fills are safe.
```

This is NOT the same as "widen when vol is high." Vol may not spike at all during a slow BTC drift. But I(t) still rises from 0 to 1 — there's still a period where the local venue hasn't caught up to the reference. The widening is driven by INFORMATION STATE, not volatility.

**Temporal exploitation window:**

When I(t) < 0.3, if your cross-venue feed is faster than most local participants, you have a window to:
1. Shift reservation price in the event direction before others reprice
2. Cancel the stale side (your bids if BTC dropped) before competitors fill them
3. Keep the fresh side (your asks) which are now correctly priced or better

The propagation function tells you how much time you have:

```
cancel_urgency = max(0, τ_next_fast_participant - our_latency) / τ_next_fast_participant
```

On thick fast markets, τ_next_fast_participant ≈ your latency → cancel_urgency ≈ 0 → no exploitable window. On thin slow markets, τ_next_fast_participant >> your latency → large window. The algorithm degrades gracefully.

---

## Foundation 2: Glosten-Milgrom Adversarial Fill Pricing

### The Problem

The GLFT treats fills as exogenous Poisson arrivals. A fill is independent of fair value — it just "happens" with rate κ. After a fill, the system updates position and continues.

This is wrong. A fill is the most informative event the market maker observes. When someone fills your ask, they are revealing that they believe V > ask. The fill is adversarial evidence about V. The GLFT captures this indirectly through a fixed "adverse selection" parameter, but that parameter doesn't update in real-time based on WHICH fill occurred, WHO filled you, or WHAT conditions held at fill time.

### The Model

The market contains two counterparty types:
- **Informed:** knows V with precision σ_informed. Trades only when V differs from your quote enough to overcome transaction costs. Fraction π of all flow.
- **Uninformed:** trades for exogenous reasons. Fraction (1-π).

Fill likelihood:

```
P(fill at ask A | V, informed) = Φ((V - A) / σ_informed)
P(fill at ask A | V, uninformed) = κ_uninformed × dt
P(fill at ask A | V) = π × Φ((V - A) / σ_informed) + (1-π) × κ_uninformed × dt
```

Bayesian update after fill at ask A:

```
P(V | fill at A) ∝ P(fill at A | V) × P(V)

// For Gaussian prior P(V) ~ N(V̂, P_V):
// Posterior V̂ shifts UPWARD (fill is evidence V > A)
// Magnitude depends on π (informed fraction) and P_V (uncertainty)

V̂_new = V̂_old + K_gm × (A + expected_informed_edge - V̂_old)
P_V_new = (1 - K_gm) × P_V_old

// Symmetric for bid fill: V̂ shifts downward
```

**Critical distinction from the base RFC's drift update:**

The base RFC's Kalman updates μ̂ (drift). The GM update updates V̂ (level). Both happen on every fill. They compound:

```
Fill at bid → V̂ shifts DOWN (fair value just dropped — level update)
           → μ̂ shifts DOWN (drift is more bearish — derivative update)
           → Reservation price drops by BOTH effects
```

The current system only does the drift update, and weakly. The GM update is the missing half.

**Setting spreads from the GM model:**

The Glosten-Milgrom zero-profit spread:

```
ask_gm = E[V | fill at ask] = V̂ + π × E[V - ask | V > ask, informed]
bid_gm = E[V | fill at bid] = V̂ - π × E[bid - V | V < bid, informed]

spread_gm = ask_gm - bid_gm = 2 × π × conditional_adverse_selection
```

This is the break-even spread. Below this, every fill loses money in expectation. It serves as a FLOOR — never quote tighter than spread_gm.

**Estimating π (informed fraction):**

```
For each historical fill, compute markout at τ = 30s:
  toxic if markout < -threshold
  benign otherwise

π̂(t) = EWMA of (fraction of fills classified as toxic)
```

π̂ feeds the GM update gain. High π̂ (recent fills have been toxic) → larger V̂ shifts per fill → wider GM floor. Low π̂ → smaller shifts → tighter GM floor.

### Specialization 2A: Flow-Type-Conditional Informed Fraction

**Activation condition:** Fills are classifiable by flow type (from OI data, cross-venue correlation, fill patterns, or liquidation detection).

**Type:** PARAMETER — replaces scalar π̂ with a per-flow-type π̂(θ).

On venues where flow types are distinguishable, maintain per-type informed fractions:

```
Flow types: θ ∈ {Liquidation, Arb, OtherMM, Retail, Whale}

π̂(Liquidation) ≈ 0.9   // extremely toxic short-term (but Shape D — reverts)
π̂(Arb)         ≈ 0.95  // precisely informative, permanent
π̂(OtherMM)     ≈ 0.3   // sometimes informed, usually inventory management
π̂(Retail)      ≈ 0.1   // mostly noise, slight contrarian signal
π̂(Whale)       ≈ 0.7   // informed at longer horizons, gradual
```

The GM update gain K_gm becomes flow-conditional:

```
K_gm(fill) = f(π̂(θ_fill), P_V, σ_informed(θ_fill))

// After arb fill: large K_gm → large V̂ shift (this fill is maximally informative)
// After retail fill: small K_gm → small V̂ shift (this fill is mostly noise)
```

**Bayesian flow-type posterior (real-time):**

```
Observations per fill:
  x₁ = fill_size / median_fill_size
  x₂ = time_since_last_same_side_fill
  x₃ = cross_venue_correlation_at_fill_time
  x₄ = OI_change_coinciding_with_fill
  x₅ = recent_fill_count_same_side / total
  x₆ = fill_size_CV_recent
  x₇ = sweep_count_recent

P(θ | x₁:ₙ) ∝ P(x₁:ₙ | θ) × P(θ)
```

The classifier doesn't need to be certain. Even 60% confidence that a fill is arb vs retail produces meaningfully different GM updates, which produce meaningfully different V̂ trajectories, which produce meaningfully different spreads.

On venues where flow types aren't distinguishable (anonymous, thick), the scalar π̂ applies uniformly. No degradation.

### Specialization 2B: Propagation-Conditioned Toxicity

**Activation condition:** Information propagation function I(t) is estimable (see Specialization 1B).

**Type:** PARAMETER — refines π̂ as a function of I(t) at fill time.

A fill at I(t) = 0.05 (very early, 95% of information not yet incorporated) is filled by someone who has information you don't. A fill at I(t) = 0.90 (late, market has absorbed the event) is filled by someone slow — they're paying the new correct price.

```
π̂_effective(fill) = π̂_base × (1 + α_prop × (1 - I(t_fill)))

// Early fills (low I): π̂ inflated → larger GM update → more defensive
// Late fills (high I): π̂ near base → normal GM update
```

This separation explains why the same fill SIZE and SIDE can have dramatically different toxicity depending on WHEN in the information cycle it arrives.

### Specialization 2C: Fill-Conditional Post-Fill Response

**Activation condition:** Markout data available for fill classification.

**Type:** STATE — provides real-time fill classification for immediate tactical response.

Don't wait for the full markout. Classify fills in real-time from features available AT FILL TIME:

```
Feature vector at fill:
  fill_size / avg_size, book_imbalance, hawkes_intensity, imm_crisis_prob,
  time_since_last_fill, fill_depth_bps, trade_imbalance_1s_before, I(t_fill)

Logistic classifier → P(toxic)
Trained on historical fills labeled by 60s markout
```

The classification triggers different post-fill strategies:

```
P(toxic) > 0.7:  EMERGENCY
  Cancel remaining orders on filled side
  Update drift estimator with strong directional signal
  Aggressively reprice reducing side (tighter, to unwind fast)

P(toxic) < 0.3:  BENIGN
  Hold position — expected mean reversion
  Normal quoting continues
  Potentially tighten reducing side to capture reversion spread

0.3-0.5 + markout Shape D (dip then recovery):  LIQUIDATION PATTERN
  Hold through drawdown
  Set time-based stop (cut if no reversion within 120s)
  Wait for overshoot to recover

Uncertain:  DEFAULT
  Let base RFC handle it (reservation price + E[PnL] filter)
```

---

## Foundation 3: Spread as Written Option Premium

### The Problem

The GLFT derives spread from `(1/γ) × ln(1 + γ/κ)` plus adjustments. This formula has no relationship to the economic object you create by posting a resting order.

### What a Resting Order Is

A resting bid at price B, size s, time-to-reprice τ is a short European put:

```
Strike: K = B     Notional: N = s     Expiry: T = τ     Spot: S = V̂

Premium collected = V̂ - B (half-spread from fair value to bid)
Exercise occurs when counterparty fills (V drops below B)
Payoff to counterparty = max(0, V_true - B)
```

Resting ask = short call. Full ladder = short straddle/strangle portfolio at multiple strikes.

### The Profitability Condition

The Greeks of the quote portfolio:

```
Theta = Σ (spread_per_level × fill_prob × size) / τ_cycle
  // Positive. This IS your spread income. Revenue per unit time.

Gamma = Σ (-Γ_option_i × s_i)
  // Negative (short straddle). Measures adverse selection SENSITIVITY.
  // Highest at touch (ATM options have max gamma).
  // This is what HURTS you when price moves.

Instantaneous PnL:
  E[pnl/dt] = Theta - 0.5 × |Gamma| × σ²

Profitable when: Theta > 0.5 × |Gamma| × σ²
  ⟺ spread income > adverse selection exposure × realized variance
```

This is the single most important inequality in market making. Everything in the base RFC — drift adjustment, inventory penalty, flow classification — exists to keep this inequality positive. The options framework makes it EXPLICIT and MONITORABLE.

### Three Spread Floors

The system runs three concurrent spread floors. The binding constraint varies by regime:

```
1. GM floor (Foundation 2):  break-even against adversarial fills
   Dominates when: informed fraction is high (toxic flow regime)

2. GLFT risk premium (base RFC):  compensation for inventory carrying risk
   Dominates when: inventory is large and drift is uncertain

3. Option premium floor:  compensation for vol exposure
   min_half_spread ≥ σ × sqrt(τ_cycle / (2π))
   Dominates when: realized vol is high (vol spike, crisis regime)

Actual spread = max(gm_floor, glft_premium, option_floor) + online_correction
```

### Gamma-Constrained Size Allocation

The GLFT allocates size per level independently. The options framework couples them through portfolio Gamma:

```
max   Σ_i Θ_i(δ_i, s_i)              // maximize spread income
s.t.  Σ_i |Γ_i(δ_i, s_i)| ≤ Γ*      // control adverse selection exposure
      s_i ≥ 0

Solution via Lagrangian:
  s*_i ∝ Θ_i / |Γ_i|                  // size proportional to Theta/Gamma ratio

Levels with high Theta/Gamma ratio get more size.
Touch has high Theta AND high Gamma — the ratio determines allocation.
Deep levels have low Theta but very low Gamma — often high ratio.
```

In high-vol regimes: Gamma at touch is very high → touch size gets cut relative to depth.
In low-vol regimes: Gamma is low everywhere → size is more evenly distributed.

This produces a size profile structurally different from the GLFT's level-independent computation.

### Vol Edge Monitor

Compute the implied vol of your quote portfolio vs realized vol:

```
σ_implied = spread_bps / (C × sqrt(τ_cycle))    // what vol your spread prices in
σ_realized = realized vol over same τ_cycle window
vol_edge = σ_implied - σ_realized

vol_edge > 0: extracting vol premium. Maintain or tighten.
vol_edge < 0: giving away vol premium. Widen or pull.
```

When vol_edge < 0, the `Θ - 0.5|Γ|σ²` condition is violated. This is a principled replacement for the quote gate — instead of "pull at 50% inventory," it's "pull when you're writing options at negative edge."

### Specialization 3A: Endogenous Impact Correction

**Activation condition:** Your quote sizes are a material fraction of book depth (> 10% of same-side depth at any quoted level).

**Type:** PARAMETER — modifies the option payoff model to include self-impact.

On thick books, your fill doesn't move mid. The option payoff is purely from the counterparty's information. On thin books, your fill DOES move mid. The option payoff includes your own impact:

```
// Standard option payoff (thick book):
payoff = |V_true - strike|

// Impact-adjusted option payoff (thin book):
payoff = |V_true - strike| + impact(fill_size, book_depth)

where:
  impact(s, D) = α × (s / D)^β × σ
  α ≈ 0.3-0.8, β ≈ 0.5-1.0 (calibrate from own historical fills)
```

The impact term increases the effective premium you need to charge. The option floor becomes:

```
min_half_spread ≥ σ × sqrt(τ / (2π)) + α × (s / D)^β × σ
                   ↑ vol exposure         ↑ self-impact cost
```

**Round-trip impact:**

When you buy, you must eventually sell. The selling side's depth — EXCLUDING your own asks — determines unwind impact:

```
unwind_depth = opposite_side_depth - your_opposite_quotes
round_trip_impact = impact(s, book_depth) + impact(s, unwind_depth)
```

On thick books: impact terms are negligible (s/D ≈ 0). The correction vanishes.
On thin books: impact terms dominate. The correction can exceed the base spread.

**Inter-level coupling:**

A fill at level 1 thins the book, increasing impact at level 2. Level 2's fill probability rises (closer to mid after impact) but its expected PnL falls (higher impact cost on thinner book). Solve via backward induction:

```
for level in (deepest)..=(touch):
    cumulative_fills = Σ_{deeper levels} s_j
    thinned_depth = initial_depth - cumulative_fills
    this_impact = impact(s_level, thinned_depth)
    effective_edge = level_price_distance_from_moved_mid
    E[PnL_level] = P(fill) × (effective_edge - AS - this_impact)
    s_level = optimize(E[PnL_level], risk_cost)
```

Produces a concave size profile: largest at depth, smallest at touch. The GLFT's profile is wrong when you're a material fraction of the book.

---

## Foundation 4: Adversarial Online Adaptation

### The Problem

Every model parameter (κ, σ, γ, all the feature α's) is estimated from historical data. Historical data is non-stationary. Parameters start degrading immediately after calibration. But the deeper problem is the stationarity assumption itself: other participants observe your behavior and adapt to exploit it. A fixed strategy is exploitable. The market is not a distribution — it's an adversary.

### The Model: Multiplicative Weights

Discretize the action space as CORRECTIONS to the GLFT baseline:

```
spread_adj ∈ {-3, -2, -1, 0, +1, +2, +3, +5, +8} bps
size_scale ∈ {0.3, 0.5, 0.7, 1.0, 1.3, 1.5}

Total: 54 actions (adjustment pairs on top of GLFT base)
```

Maintain a weight per action. Update weights using counterfactual PnL:

```
w_a(0) = 1/54

After each quote cycle t:
  for each action a:
    pnl_a(t) = counterfactual_pnl(a, fills(t), price_path(t))
  
  for each action a:
    w_a(t+1) = w_a(t) × exp(η × pnl_a(t))
  
  w = w / sum(w)
  
  // Choose: weighted average
  spread_adj_t = Σ_a w_a × a.spread_adj
  size_scale_t = Σ_a w_a × a.size_scale
```

### Computing Counterfactual PnL

For each action a = (spread_adj, size_scale):

```
hypothetical_bid = actual_bid + spread_adj / 2
hypothetical_ask = actual_ask - spread_adj / 2
hypothetical_size = actual_size × size_scale

For each actual fill:
  Would this fill have occurred at the hypothetical price?
  If fill.price >= hypothetical_ask → yes (buy fill captured)
  If fill.price <= hypothetical_bid → yes (sell fill captured)
  Compute edge relative to V̂ at fill time, scale by hypothetical_size
  
Subtract inventory cost for hypothetical position
```

Counterfactual PnL is conservative for wider spreads (miss fills) and optimistic for tighter spreads (capture more). This asymmetry is correct — it naturally biases toward slightly wider spreads as a safety margin.

### Guarantee

```
Regret(T) ≤ O(sqrt(T × ln(54)))

// After T=1000 cycles (~90 min), the adaptive strategy is within ~50 cycles' PnL
// of the BEST FIXED spread/size over the entire session.
// No fixed strategy can guarantee this.
```

The learner converges to the optimal correction for whatever the market is currently doing. Regime changes, competitor adaptation, microstructure shifts — it handles all of them automatically without recalibration.

### Hierarchical Learning Rate

The learning rate η controls adaptation speed vs stability. Use a meta-learner:

```
η ∈ {0.01, 0.03, 0.1, 0.3}
Run 4 parallel MW instances. Weight by their own cumulative PnL.
Fast learners dominate during regime changes. Slow learners dominate during stability.
```

### Specialization 4A: Participant-Informed Counterfactuals

**Activation condition:** Participant fingerprinting active (Specialization 1A).

**Type:** PARAMETER — improves counterfactual PnL estimates.

Standard counterfactual asks "would this fill have occurred at this hypothetical price?" When you know WHO filled you, the answer is sharper:

```
// Standard: would the fill have occurred?
P(fill | hypothetical_ask, market) = f(price_distance, κ, σ)

// Participant-informed: would THIS PARTICIPANT have filled?
P(fill | hypothetical_ask, participant_c) = f(price_distance, κ_c, response_model_c)

// Participant C always fills within 2 bps of mid but never fills at 5+ bps
// A hypothetical that widens from 3 to 5 bps: standard model says "fill less likely"
// Participant model says "fill probability drops to near zero" — much sharper estimate
```

Sharper counterfactuals → faster MW convergence → faster adaptation to current conditions.

### Specialization 4B: Funding Clock in Reward Function

**Activation condition:** Periodic funding settlement with predictable flow.

**Type:** STATE — adds a periodic component to the reward function.

The MW reward is PnL per cycle. In the pre-settlement window, the reward should account for the KNOWN upcoming flow:

```
reward(t) = fill_pnl(t) + settlement_positioning_value(t)

settlement_positioning_value = {
  If funding > 0 and we're net short approaching settlement:
    + expected_funding_income - expected_settlement_flow_cost
  If funding > 0 and we're net long approaching settlement:
    - expected_funding_payment - expected_adverse_flow_damage
}
```

This teaches the MW algorithm that actions which build short inventory in the 10 minutes before a positive-funding settlement have higher reward than the fills alone would suggest. The algorithm learns to tighten asks (accumulate short) pre-settlement and widen bids (avoid longs) without explicit programming.

The settlement clock is general to any venue with periodic funding. On perpetual futures (Hyperliquid, Binance, dYdX), settlement is hourly or 8-hourly. On venues without funding, this specialization contributes nothing — the periodic term is zero.

---

## Foundation 5: Growth-Optimal Sizing

### The Problem

The GLFT's risk aversion γ controls spread and size. It has no physical interpretation. The base RFC improves this with γ(q) = γ_base × (1 + β(q/q_max)²), but γ_base is still arbitrary.

### The Model: Non-Ergodic Kelly

**Single-fill Kelly:**

```
For a fill with edge e and adverse selection variance σ²_AS:
  g(s) ≈ s × e/W - s² × σ²_AS / (2W²)    (growth rate)
  s* = e × W / σ²_AS                        (Kelly optimal)
```

**Portfolio Kelly (multiple simultaneous quotes):**

All quotes can fill in a cascade. The growth rate of the portfolio accounts for correlated outcomes:

```
g(s₁,...,s_N) ≈ Σ_i s_i × e_i / W - (1/2W²) × s^T × Σ × s

where Σ_ij = ρ(i,j) × σ_AS_i × σ_AS_j    (fill outcome covariance matrix)
```

**Estimating the fill correlation matrix:**

```
ρ(i,j) = P(fill_j within τ_window | fill_i) / P(fill_j unconditional)

Quiet regime:  ρ(adjacent_levels) ≈ 0.2-0.4
Crisis regime: ρ(adjacent_levels) ≈ 0.8-0.95 (cascade)
```

The IMM regime probabilities condition the matrix:

```
Σ_effective = P(Quiet) × Σ_quiet + P(Trending) × Σ_trending + P(Crisis) × Σ_crisis
```

**Growth-optimal total exposure:**

```
S* = e_vec^T × Σ⁻¹ × 1 / (1^T × Σ⁻¹ × 1)    (multivariate Kelly)
```

Quiet (low ρ, diagonal Σ): S* is large — diversification across levels.
Crisis (high ρ, dense Σ): S* collapses — all fills are the same bet.

**Fractional Kelly:**

```
f(regime) = {
    Quiet:    0.7 - 0.9
    Trending: 0.4 - 0.6
    Crisis:   0.15 - 0.3
}

actual_exposure = f(regime) × S*_kelly
```

**The derived γ:**

```
γ_effective = W / S*_kelly

// This γ:
//   - Increases when capital shrinks (drawdown protection)
//   - Increases when fill correlation is high (cascade protection)
//   - Decreases when edge increases (more aggressive when profitable)
//   - All AUTOMATIC, derived from growth rate optimization
```

### Specialization 5A: Liquidation-Frontier-Augmented Correlation

**Activation condition:** Liquidation density function ρ(p) is estimable (OI data + margin requirements available).

**Type:** PARAMETER — provides state-dependent jump risk in the correlation matrix.

When price is near a liquidation cluster, fill correlation jumps because a cascade is imminent:

```
// Base correlation matrix Σ from historical fill data

// Augmented with liquidation proximity:
proximity = max(0, 1 - |current_price - nearest_cluster_price| / (σ × cluster_impact_distance))
Σ_augmented = Σ_base + proximity × (Σ_crisis - Σ_base)

// When approaching a liquidation cluster: correlation → crisis levels
// When far from any cluster: correlation → base levels
```

**The liquidation density function:**

```
ρ_long(p) = expected long liquidation volume triggered at price p

Constructed from:
  1. OI changes tracked by price level (when OI increases at mark price p, entries logged near p)
  2. Leverage inferred from funding rate magnitude and historical liquidation events
  3. Liquidation price computed per entry: p_liq = p_entry × (1 - 1/L + maintenance_margin)
  4. Aggregate into density function, discretized to 1-bps bins
  5. Update every cycle from OI feed
```

When ρ(p) is large at some price near current price, the Kelly sizing automatically reduces (higher correlation → smaller S*) without needing a separate liquidation-specific mechanism. The general framework absorbs the venue-specific information.

**Quoting around the frontier:**

When proximity > 0.5 (approaching a cluster), the Kelly-reduced sizing combines with a quoting shift from the reservation price:

```
// Reservation price incorporates expected cascade:
// V̂ filter's drift estimate μ̂ already incorporates the gravitational pull of the cluster
// (from the propagation of fills as price approaches)
// The reduced Kelly sizing handles the rest

// Additionally: place recovery bids BELOW the cascade zone
// These have POSITIVE expected markout (Shape D — liquidation overshoot reverts)
// The fill correlation at these depths is DIFFERENT from the cascade zone:
//   In cascade zone: fills are correlated with further selling (ρ ≈ 1)
//   Below cascade: fills are anticorrelated with further selling (reversion)
//   Σ for below-cascade levels has NEGATIVE off-diagonal entries vs cascade levels
//   → Kelly INCREASES exposure at these depths. Principled bottom-fishing.
```

This is the critical insight: the correlation matrix isn't uniformly positive. Recovery fills after a cascade are NEGATIVELY correlated with cascade fills. The Kelly framework naturally INCREASES exposure to recovery levels precisely when it DECREASES exposure to cascade levels. No special case required — the math does it.

---

## Foundation 6: Optimal Unwind Policy

### The Problem

After accumulating inventory, the GLFT uses the same spread formula for reducing as for accumulating. The speed of unwinding is implicitly determined by κ and spread — it's not a decision variable. But unwind speed is a strategic choice:

- Aggressive (tight reducing quotes): fast, low carry, less edge per unwind fill, self-impact pushes mid against you
- Patient (wide reducing quotes): slow, high carry, more edge per fill, market may revert in your favor

### The Model: Bellman Equation

State:

```
s = (q, μ̂, regime, f̂, markout_shape)

q             ∈ {-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4}
regime        ∈ {Quiet, Trending, Crisis}
f̂             ∈ {Paying, Neutral, Receiving}
markout_shape ∈ {Reversion, Flat, Continuation}

Total states: 11 × 3 × 3 × 3 = 297
```

Action:

```
a ∈ {-4, -3, -2, -1, 0, +1, +2, +3, +4} bps adjustment to reducing-side spread
```

Reward:

```
R(s, a) = P_fill(a) × (δ_base + a)            // spread earned on reducing fills
         - |q| × γ(q) × σ² × dt               // inventory risk cost
         - |q| × f̂ × dt                        // funding carry
         - |q| × μ̂ × sign(q) × dt              // drift cost
```

Bellman:

```
V(s) = max_a { R(s, a) + β × Σ_{s'} P(s'|s,a) × V(s') }
```

Value iteration offline. 297 × 9 = 2,673 state-action pairs. Converges in seconds. Refresh daily.

Output: lookup table (q, regime, funding, markout_shape) → optimal_reducing_spread_adjustment.

```
Example entries:
  (q=3, Crisis, Paying, Continuation)  → -4 bps (unwind aggressively)
  (q=3, Quiet, Receiving, Reversion)   → +3 bps (patient, market reverts, carry is free)
  (q=1, Trending, Neutral, Flat)       → -1 bps (moderate urgency)
  (q=0.5, Quiet, Neutral, any)         → +1 bps (no urgency, earn more per fill)
```

### Specialization 6A: Funding Settlement in Transition Model

**Activation condition:** Periodic funding settlement exists.

**Type:** STATE — adds a time-of-hour dimension to the Bellman state.

With hourly funding, the transition model gains periodic structure:

```
// Expanded state:
s = (q, μ̂, regime, f̂, markout_shape, time_to_settlement)

time_to_settlement ∈ {>10min, 5-10min, 2-5min, 0-2min, post_0-3min}

Total states: 297 × 5 = 1,485 (still trivial to solve)
```

The transition model captures settlement-driven flow:

```
P(q' | q, a, time_to_settlement < 5min, funding > 0):
  // Higher probability of bid fills (sellers closing longs before settlement)
  // Higher probability of ask fills in post-settlement window (re-entry)
  
P(regime' | regime, time_to_settlement < 2min, funding_zscore > 2):
  // Higher transition probability to Crisis (settlement cascades)
```

The Bellman policy now includes settlement-aware behavior:

```
(q=0, 2-5min before settlement, funding=Paying) → a = -2 bps on ASKS
  // Accumulate short position. You'll receive funding AND the post-settlement reversion.

(q=-2, post_0-3min, funding=Paying) → a = +2 bps on BIDS
  // Hold short position. Reversion flow will let you cover at better prices.
```

This isn't a heuristic — it's the OPTIMAL policy given the periodic flow structure. The Bellman solver discovers the settlement strategy automatically from the transition model.

---

## Markout Engine (Cross-Cutting Infrastructure)

Foundations 2, 4, 5, and 6 all depend on fill quality measurement. The markout engine is the shared infrastructure.

### Definition

For each fill at time t, price p, side s:

```
markout(τ) = s × (mid(t + τ) - p)    for τ ∈ {1s, 5s, 10s, 30s, 60s, 300s}
```

### The Markout Curve

The curve across horizons reveals the fill's nature:

```
Shape A: [+2.5, +2.0, +1.5, +1.0, +0.5]  — Decaying Positive
  Clean MM profit. Spread correctly calibrated.

Shape B: [+3.0, +1.0, -1.0, -4.0, -8.0]  — Positive Then Collapse
  Short-term edge, long-term adverse. Slow information leak.

Shape C: [-3.0, -5.0, -7.0, -9.0, -11.0] — Monotonic Adverse
  Pure toxic. Should not have been quoting at this depth/state.

Shape D: [-6.0, -4.0, -1.0, +1.0, +2.0]  — Dip Then Recovery
  Liquidation overshoot. Hold through drawdown = profitable.

Shape E: [+1.0, +1.0, +1.0, +1.0, +1.0]  — Flat Positive
  Persistent edge. Fair value estimate better than market's.
```

### Decomposition

```
markout(τ) = realized_half_spread + temporary_impact(τ) + permanent_impact + noise(τ)

realized_half_spread = s × (mid(t) - p)
permanent_impact     = s × (mid(t + 300s) - mid(t))
temporary_impact(τ)  = markout(τ) - realized_half_spread - permanent_impact
```

Rolling statistics:

```
                      Mean    Std     Win%
Realized half-spread: +2.8    0.5     98%
Temporary impact:     -0.4    3.2     42%
Permanent impact:     -1.6    4.1     35%
Net markout (60s):    +0.8    5.0     56%
```

### Who Consumes It

```
Foundation 2 (GM):     markout labels → π̂ estimation, fill-type-conditional π̂
Foundation 4 (MW):     counterfactual PnL computation
Foundation 5 (Kelly):  fill correlation matrix Σ (which fills cluster?)
Foundation 6 (Bellman): markout_shape state variable, reward calibration
Specialization 2C:    real-time fill classification for post-fill response
```

Build the markout engine FIRST. Without it, Foundations 2, 4, 5, and 6 have no calibration signal.

---

## Unified Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MARKET DATA FEED                            │
│    L2 book, trades, cross-venue, OI, funding, fills               │
└───────┬─────────────────┬──────────────────┬───────────────────────┘
        │                 │                  │
        ▼                 ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────────────────────┐
│  FOUNDATION 1 │  │SPECIALIZATION│  │      SPECIALIZATION          │
│  Latent Value │◄─┤ 1A: Participant  │  │ 1B: Information Propagation  │
│  Filter (UKF  │  │   Fingerprint│  │     Function I(t)            │
│   + IMM)      │  │   Signals    │  └──────────┬───────────────────┘
│               │  └──────────────┘             │
│  State: V̂,μ̂,σ̂│◄──────────────────────────────┘ (scales obs noise R)
└───────┬───────┘
        │ V̂, μ̂, σ̂, P(regime)
        ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐
│  FOUNDATION 2 │◄─┤SPEC 2A: Flow │◄─┤SPEC 2B: I(t)-│
│  Glosten-     │  │ Type π̂(θ)   │  │ Conditioned  │
│  Milgrom Fill │  └──────────────┘  │ Toxicity     │
│  Pricing      │                    └──────────────┘
│               │  ┌──────────────┐
│  Output: V̂    │◄─┤SPEC 2C: Fill │
│  updates, π̂,  │  │ Conditional  │
│  GM floor     │  │ Response     │
└───────┬───────┘  └──────────────┘
        │ GM floor, π̂
        ▼
┌───────────────┐  ┌──────────────┐
│  FOUNDATION 3 │◄─┤SPEC 3A:      │
│  Options      │  │ Endogenous   │
│  Framework    │  │ Impact       │
│               │  │ Correction   │
│  Θ, Γ, Vega  │  └──────────────┘
│  Option floor │
│  Gamma sizing │
└───────┬───────┘
        │ three spread floors, Gamma-constrained sizes
        ▼
┌───────────────┐  ┌──────────────┐
│  FOUNDATION 4 │◄─┤SPEC 4A:      │
│  Adversarial  │  │ Participant  │
│  Online       │  │ Counterfact. │
│  Learning     │  ├──────────────┤
│               │  │SPEC 4B:      │
│  MW on        │  │ Funding Clock│
│  spread × size│  │ in Reward    │
└───────┬───────┘  └──────────────┘
        │ online spread/size correction
        ▼
┌───────────────┐  ┌──────────────┐
│  FOUNDATION 5 │◄─┤SPEC 5A:      │
│  Kelly Sizing │  │ Liquidation  │
│               │  │ Frontier in  │
│  S*, f(regime)│  │ Correlation  │
│  γ_effective  │  └──────────────┘
└───────┬───────┘
        │ total exposure, per-level sizes
        ▼
┌───────────────┐  ┌──────────────┐
│  FOUNDATION 6 │◄─┤SPEC 6A:      │
│  Bellman      │  │ Settlement   │
│  Unwind       │  │ Clock in     │
│               │  │ Transitions  │
│  Reducing-side│  └──────────────┘
│  adjustment   │
└───────┬───────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        QUOTE GENERATION                            │
│                                                                    │
│  reservation = V̂ + drift_shift(μ̂) - inv_penalty(γ_eff) - carry   │
│  spread = max(gm_floor, glft_premium, option_floor) + mw_correction│
│  size = min(kelly_size, gamma_constrained, epnl_positive)          │
│  reducing_adj = bellman_lookup(q, regime, funding, markout_shape)  │
│                                                                    │
│  Per level: if Θ_level - 0.5|Γ_level|σ² > 0 → quote              │
│             else → don't quote (replaces quote gate)               │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MARKOUT ENGINE (feedback)                      │
│                                                                    │
│  Per fill: markout curve at {1s, 5s, 10s, 30s, 60s, 300s}        │
│  Decompose: half_spread + temp_impact + permanent_impact           │
│  Classify: Shape A/B/C/D/E                                        │
│  Feed back to: F2 (π̂), F4 (counterfactual), F5 (Σ), F6 (shape)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Specialization Activation Matrix

Each specialization has an activation condition. When the condition is not met, the specialization contributes nothing and the foundation runs standalone:

```
Spec   Name                     Activation Condition                  Interface
─────  ────────────────────     ──────────────────────────────────    ─────────
1A     Participant Fingerprint  < 10 active quoters at any depth     OBSERVATION
1B     Information Propagation  Cross-venue latency spread > 500ms   PARAMETER (R scaling)
2A     Flow-Type π̂             Flow types distinguishable (OI, etc)  PARAMETER (π̂ per type)
2B     Propagation Toxicity    I(t) estimable (Spec 1B active)       PARAMETER (π̂ scaling)
2C     Fill-Conditional        Markout data available (>100 fills)   STATE (post-fill action)
3A     Endogenous Impact       Own quotes > 10% of same-side depth   PARAMETER (impact cost)
4A     Participant Counterfact  Fingerprinting active (Spec 1A)      PARAMETER (sharper CF)
4B     Funding Clock Reward    Periodic funding settlement exists    STATE (periodic reward)
5A     Liquidation Frontier    OI + margin data available            PARAMETER (correlation Σ)
6A     Settlement Transitions  Periodic funding exists               STATE (time-to-settle)
```

On a thick, anonymous, no-funding venue (e.g., equity options): no specializations activate. All six foundations run standalone. They are still correct — just without the additional edge from market structure exploitation.

On Hyperliquid HIP-3 thin perps: most or all specializations activate. Each one plugs into the corresponding foundation through a clean interface, providing additional observations, refined parameters, or structural state variables.

On something in between (Binance HYPE-USDT perp): some specializations activate (4B, 5A, 6A from funding mechanism; possibly 3A if you're a large fraction of depth; 2A if liquidations are detectable). The architecture handles the partial activation naturally.

---

## Implementation Order

### Phase 0: Markout Engine
Build first. Without it, Foundations 2, 4, 5, 6 have no calibration signal. Log every fill with: timestamp, side, price, size, mid at fill, then mid at t+{1,5,10,30,60,300}s. Compute decomposition. Classify shape. Store.

### Phase 1: Foundation 1 (Latent Value Filter) + Foundation 3 (Options Framework)
These are independent and can be built in parallel.

**F1** replaces "mid" as the anchor with a proper Bayesian estimate. Subsumes the Kalman drift estimator from the base RFC — V̂ and μ̂ come from the same filter. Every subsequent improvement compounds on better V̂.

**F3** provides a principled spread floor (never undercharge for vol exposure), principled size allocation (Gamma-constrained), and the profitability monitor Θ - 0.5|Γ|σ² which replaces the quote gate.

### Phase 2: Foundation 2 (GM Fill Pricing) + Foundation 4 (Adversarial Learning)
Both require Phase 0 (markout data) but are independent of each other.

**F2** adds Bayesian fill updates to V̂ (the GM update) and produces the adversarial spread floor. π̂ calibrated from markout labels.

**F4** wraps the spread computation with online adaptation. Counterfactual PnL requires markout data. Converges within ~30 minutes of live trading per session.

### Phase 3: Foundation 5 (Kelly Sizing) + Foundation 6 (Bellman Unwind)
Both require fill data from Phases 0-2.

**F5** requires fill correlation matrix across regimes (from IMM + fill clustering data). Replaces arbitrary γ_base.

**F6** requires markout shape classification. 297-state Bellman is trivial to solve once state variables are available.

### Phase 4: Specializations (activate as data and conditions permit)

```
Immediate (data collection from boot):
  1A: Participant fingerprinting — start collecting L2 diff data, cluster passively
  3A: Endogenous impact — start logging own fills vs book depth, calibrate α, β
  
After ~1 day of data:
  2C: Fill-conditional response — train classifier on markout labels
  5A: Liquidation frontier — requires OI feed wiring + entry tracking
  4B, 6A: Funding clock — requires settlement timestamp tracking

After ~1 week of data:
  1B: Information propagation — requires cross-venue event detection + I(t) calibration
  2A: Flow-type classification — requires labeled fills across types
  2B: Propagation-conditioned toxicity — requires both I(t) and flow labels
  4A: Participant counterfactuals — requires fingerprint stability validation
```

---

## Measurable Outcomes

Each foundation has a specific metric. The specializations improve the same metric when active:

```
Foundation  Metric                     Baseline     Target
─────────  ────────────────────────    ────────     ──────
F1         V̂ prediction MSE (5s)      exchange mid  -10 to -20% MSE
F2         Mean permanent impact/fill  -1.6 bps     -0.8 to -1.2 bps (better fill pricing)
F3         Fraction of Θ-Γσ² < 0      unmeasured    < 5% of quote-seconds
F4         Regret vs best fixed        unmeasured    < 100 bps cumulative per session
F5         Max drawdown / total PnL    unmeasured    < 3.0 (Kelly smooths equity curve)
F6         Inventory half-life         unmeasured    -15 to -30% (faster when optimal)
Markout    Net 60s markout             +0.5-1.0 bps +2.0-3.5 bps (compound of F1-F6)
```

The markout improvement is the ultimate metric. Each foundation contributes 0.3-0.8 bps. They compound because better V̂ feeds better fill classification feeds better spread calibration feeds better sizing feeds better unwinding. The feedback loop through the markout engine means the system gets MORE PROFITABLE as it runs longer — posteriors tighten, π̂ improves, MW converges, correlation matrices sharpen, Bellman tables calibrate.
