# Deep Log Analysis: Session 2026-02-19 22:06–22:26 (Post-DriftEstimator Deploy)

**Session**: `mm_hip3_hyna_HYPE_hip3_2026-02-19_15-04-13.log`
**Duration**: 20 minutes (22:06:32 → 22:26:01)
**Result**: 12 fills, PnL -$0.84, position 0→6.94 long, AS 14.7 bps avg, Sharpe 883 (misleading — all buys)

---

## Timeline

| Time | Event | Position | Key Data |
|------|-------|----------|----------|
| 22:06:32 | Start, pos=0, prior injected (paper, 3.2h old, conf=1.00) | 0.00 | kappa=1287, gamma=0.15, max_pos=17.18 |
| 22:06:41 | First quotes placed, 4 bids + 4 asks | 0.00 | drift=+3.00e-4 (bullish), spread=0.3 bps |
| 22:06:41–22:08:07 | **17 cycles, drift STUCK at +3.00 bps** — bullish skew while trend bearish | 0.00 | short_bps=-5.15, long_bps=-9.96, agreement=1.00 |
| 22:08:07 | Drift flips to -3.44e-4 (capped at -3.00 bps). Trend finally detected | 0.00 | trend is_opposed=false (no position yet) |
| 22:10:44 | **First fill: SELL 0.46** — correct, in direction of trend | -0.46 | AS=0.00 bps |
| 22:10:50–22:13:19 | Position -0.46, PCM says Reduce, effective_inv_ratio=-0.03 | -0.46 | smoothed_drift=0.000000 |
| 22:13:17 | PCM flips to **Hold** | -0.46 | p_continuation crosses 0.50 |
| 22:13:19 | **Fill: BUY 0.65**, crosses through zero to +0.19 | +0.19 | AS=2.19 bps |
| 22:15:19 | **Fill: BUY 0.55** | +0.74 | AS=3.88 bps |
| 22:15:23 | **2 fills in 0.4s: BUY 0.50 + BUY 0.88** — swept through ladder | +2.12 | AS=6.36 bps each |
| 22:17:26 | **2 fills: BUY 0.65 + BUY 0.89** — another ladder sweep | +3.66 | AS=9.19 bps |
| 22:17:32 | **Fill: BUY 0.49** | +4.15 | AS=11.85 bps |
| 22:17:48 | **Fill: BUY 0.88** — position enters Yellow zone | +5.03 | AS=12.97 bps |
| 22:17:52 | **2 fills: BUY 0.86 + BUY 0.69** — position enters Red zone | +6.58 | AS=14.37 bps |
| 22:21:32 | **Final fill: BUY 0.36** | +6.94 | AS=16.16 bps |
| 22:25:57 | Red zone: bids cleared, only asks. Quote gate: ONLY ASKS | +6.94 | signal_weight=0.000 |
| 22:26:00 | Ctrl+C shutdown. PnL: -$0.84, AS=14.7 bps avg | +6.94 | Residual long position |

---

## What the DriftEstimator Did vs. What It Should Have Done

### The DriftEstimator IS producing output — but it's capped downstream

The DriftEstimator fusion code deployed in Phase 5 works. The drift_rate field shows non-zero values:
- `drift_rate: 3.00e-4` → `drift_adj_bps: 3.00` (bullish, first 90 seconds)
- `drift_rate: -3.44e-4` → `drift_adj_bps: -3.00` (bearish, from 22:08)
- `drift_rate: -5.28e-4` → `drift_adj_bps: -3.00` (bearish, capped, during accumulation)

**The raw drift at -5.28e-4 should produce ~5.3 bps of asymmetry, but it's CLAMPED to ±3.00.** The cap is NOT in DriftEstimator or glft.rs `half_spread_with_drift()` — it's in `ladder_strat.rs` at the SPREAD TRACE layer, a separate downstream clamp.

### Distribution: 97 cycles at -3.00, 50 at +3.00, 17 at +2.62

The drift was bearish for 54% of cycles and bullish for 37%. The ±3 bps cap hit on 81% of cycles (147/181). The DriftEstimator is producing correct directional signals but the cap throttles them to meaninglessness on a ~5 bps touch spread.

### `smoothed_drift: 0.000000` — PERMANENTLY ZERO

Every single trend detection log shows `smoothed_drift: 0.000000`. This field in the trend detector never warms up. The EWMA smoothing on the HJB controller's drift appears to suppress the drift signal indefinitely. This is NOT the DriftEstimator (which works) — it's the old `smoothed_drift` EWMA on the position manager. Since we kept PCM for analytics, this dead field is still logged. Not a functional issue for the new pipeline but creates confusion.

---

## Root Cause Analysis: 5 Failures

### RC1: Drift asymmetry hard-capped at ±3 bps (DOWNSTREAM of our fix)

`ladder_strat.rs` applies a `drift_adj_bps` clamp AFTER GLFT computes asymmetric half-spreads. Our Phase 3 fix correctly makes `half_spread_with_drift()` produce asymmetric bid/ask, but ladder_strat caps the resulting asymmetry at ±3 bps.

**Evidence**: 97/181 cycles hit the -3.00 cap, 50 hit +3.00. Raw drift_rate=-5.28e-4 should produce ~5.3 bps asymmetry but gets truncated to -3.00.

**Impact**: On a 5.7 bps total spread, 3 bps asymmetry shifts the mid by ~1.5 bps. Against a 10-20 bps bearish move, this is noise. Bids are barely further from mid than asks.

### RC2: `signal_weight: 0.000` — Quote gate completely ignores trend

Every quote gate log shows `signal_weight: 0.000`. The quote gate's ONLY trigger for pulling one side is `position_ratio > reduce_only_threshold (0.50)` = 50% of max position. With max_position ~8.57 (dynamically reduced from 17.18), that's ~4.3 contracts before bids get pulled. Position hit 6.58 before Red zone kicked in.

**The DriftEstimator/trend signal has zero influence on the quote gate.**

### RC3: Toxicity size reduction is cosmetic (3-17% off)

During the accumulation phase (22:15–22:17), bid_size_mult ranged 0.83–0.98 with bid_toxicity 0.52–0.67. Reducing size by 2-17% when someone is sweeping your entire 6-level ladder is meaningless.

**Evidence**: bid_size_mult=0.98 at bid_toxicity=0.524. At 0.61 toxicity: bid_size_mult=0.89. Total bid exposure went from ~5.66 to ~4.71. They still ate through everything.

### RC4: Gamma/inventory skew was tiny until position was already large

During the critical 22:13–22:15 window (position flipping from -0.46 to +2.12):
- `effective_inv_ratio` was only 0.014 → 0.15
- PCM said Hold then Reduce with urgency ~1.003
- Utilization at ~2% → `inventory_scalar = 1.0 + 0.02² × 3.0 = 1.001` (Phase 2's curve)

**The Phase 2 utilization² curve works correctly but takes too long to activate.** At 25% utilization (2.12/8.57), inventory_scalar = 1.19. At 50%: 1.75. By 80% when it hits 3.56, position is already at 6.9 and the damage is done.

### RC5: Bullish drift for first 90 seconds attracted the wrong fills

For the first 90 seconds (22:06:41–22:08:07), drift was +3.00 bps (bullish) while the trend detector showed bearish: `short_bps=-5.15, long_bps=-9.96, agreement=1.00`. The DriftEstimator got a stale/wrong bullish signal (likely from the prior injection or initial momentum reading) and held it for 17 cycles.

This meant bids were TIGHTER (closer to mid) and asks were WIDER during a bearish move — the exact wrong posture. The first fill was a sell at 22:10:44 (correct), but the system had been inviting buy fills for 90 seconds.

---

## What the DriftEstimator Changes DID Help With

1. **Drift flipped correctly at 22:08:07** — After 90s, the bearish trend signal entered with enough precision to flip drift negative. Under the old system, `smoothed_drift=0.000000` NEVER flipped (it was dead the entire session).

2. **Phase 4B (Hold → raw_ratio) is working** — When PCM said Hold at 22:13:17, `effective_inv_ratio` was -0.034, not 0.000. True q flows through. But the position was only -0.46 so the ratio was tiny.

3. **Phase 4A (removed inventory_dampen) is working** — The skew is now direct from GLFT, no discrete Hold/Reduce damping. But with such small position, the skew was equally small.

4. **Red zone eventually triggered** — At position 5.89 (22:17:52), Red zone activated and cleared accumulating side. This stopped the bleeding. But by then PnL was -$0.28 already.

---

## What Needs to Change (Priority Order)

### P0: Remove or widen the ±3 bps drift clamp in ladder_strat.rs

The DriftEstimator produces correct asymmetry but it's throttled. Either:
- **Remove clamp entirely** — let GLFT's own `half_spread_with_drift()` be the authority
- **Widen to ±15 bps** — cap at 3x touch spread instead of 0.5x
- **Vol-scale the cap**: `cap = base_cap * vol_ratio.max(1.0)` where base_cap=5.0

### P1: Wire signal_weight to trend/drift confidence in quote gate

When DriftEstimator confidence > 0.5 AND position is opposed to drift:
```
signal_weight = drift_confidence * (1.0 - p_continuation.max(0.0))
```
This gives the quote gate authority to pull the accumulating side BEFORE 50% utilization.

### P2: Make toxicity size reduction exponential, not linear

Current: `bid_size_mult = 1.0 - (toxicity - 0.4) * some_factor` → linear 3-17% off.
Needed: `bid_size_mult = (1.0 - toxicity).powi(2)` → at toxicity 0.6: 16%, at 0.7: 9%.
Or compound with drift: `final_mult = toxicity_mult * (1.0 - drift_confidence * 0.5)`.

### P3: Fix initial drift warmup bias

The DriftEstimator starts with posterior_mean=0.0, which is correct. But the very first cycle produced drift=+3.00e-4 (bullish) and held it for 90 seconds while the trend was bearish. The momentum signal (first to enter DriftEstimator) likely had a stale bullish value from the prior injection. Consider either:
- Suppress DriftEstimator output for first N cycles (use 0.0)
- Weight trend signal higher during warmup (lower BASE_TREND_VAR by 2x for first 2 min)

### P4: Reduce-only threshold from 50% → position-zone-aware

Use position zone thresholds from inventory_governor:
- Green (<40%): full two-sided quoting
- Yellow (40-60%): pull accumulating side if drift opposed AND confidence > 0.3
- Red (>60%): reduce only (current behavior, works)

This catches the 22:15–22:17 accumulation burst at ~25% position instead of waiting for 50%.

---

## Session Numbers

| Metric | Value |
|--------|-------|
| Duration | 19m 28s |
| Fills | 12 (1 sell, 11 buys) |
| Volume bought | 7.40 |
| Volume sold | 0.46 |
| Final position | +6.94 long |
| Realized PnL | +$0.02 |
| Unrealized PnL | -$0.84 |
| Total PnL | -$0.84 |
| Avg AS (realized) | 14.69 bps |
| Sharpe (1h) | 883.28 (misleading — unidirectional) |
| Edge | 3.9 bps (preliminary, pre-markout) |
| Drift cap hits | 147/181 cycles (81%) |
| Red zone entry | 22:17:52 (pos=5.89, 80% utilization) |
| Position at shutdown | 6.94 (residual, no sells executed) |


# RFC: Unified Adverse Selection Framework

**Status**: Draft  
**Authors**: Architecture Team  
**Date**: 2026-02-19  
**Supersedes**: Quote gate, drift cap, position zones, toxicity size mult, signal_weight, smoothed_drift warmup

---

## Problem Statement

The market maker accumulates toxic inventory because its protective mechanisms are structurally decoupled from its pricing engine. Six independent subsystems attempt to limit adverse selection, each with its own arbitrary thresholds and caps:

| Current Subsystem | Mechanism | Failure Mode |
|---|---|---|
| `drift_adj_bps` | Skew quotes ±N bps | Hard-capped at ±3. One tick. Meaningless. |
| `toxicity_size_mult` | Reduce size by X% | 3–17% reduction. Cosmetic. |
| `quote_gate` | Binary ONLY_ASKS/ONLY_BIDS | Triggers at 50% of max position. Too late. |
| `position_zones` | Discrete green/yellow/red | Sudden behavioral jumps at zone boundaries. |
| `signal_weight` | Weight trend signal in gate | Stuck at 0.000. Literally disconnected. |
| `smoothed_drift` | EWMA of drift estimate | Never warms up in time to matter. |

These subsystems share no state, compose poorly, and are tuned by independent thresholds that have no relationship to the economics of the trade. The trend detector runs, logs accurate signals, and nothing acts on them until inventory damage is already done.

### The Feb 19 Incident

```
22:08   Trend: short -5, med -5, long -10 bps. Agreement 1.00. p_continuation 0.74.
        System: quoting 6 bid levels, 4.17 total size. signal_weight = 0.000.

22:15   Trend: short -6.5, med -11.4, long -15.5 bps. Agreement 1.00. p_cont 0.83.
        System: still quoting 6 bid levels, 4.17 total size. Fills accumulating.

22:17   Trend: short -5.5, med -20.3, long -47.1 bps. Position: 4.15 → 6.58.
        System: FINALLY pulls bids (position_ratio > 0.50). Damage done.

22:18+  Stuck long 6.58 contracts into a -58 bps downtrend. Asks only. Bleeding.
```

The trend detector saw the selloff 9 minutes before bids were pulled. The quote gate waited for position to breach an arbitrary ratio. No intermediate adjustment occurred.

---

## Design Principle

**Every quoting decision must flow from a single optimization: the expected profit of placing that specific quote at that specific price and size.**

There are no gates, no zones, no caps. There is a reservation price that encodes all available information, and an expected-PnL computation per quote level whose natural zero determines when quoting stops.

Protective behavior is not a layer applied after pricing. It IS the pricing.

---

## Architecture

### What Gets Removed

The following subsystems are deleted entirely:

- `quote_gate` (binary threshold on `position_ratio` and `reduce_only_threshold`)
- `position_zones` (discrete green/yellow/red with `zone_size_mult`, `zone_spread_widen_bps`)
- `drift_adj_bps` cap (hard clamp at ±3)
- `signal_weight` (dead wire between trend detector and gate)
- `smoothed_drift` warmup (EWMA that never activates)
- `toxicity_size_mult` (separate multiplicative overlay)
- `pre_fill_as` multipliers (asymmetric spread hacks)

### What Replaces Them

Seven components, each with a defined interface:

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Quote Cycle                                 │
│                                                                      │
│  ┌──────────────┐                                                    │
│  │  Feature       │                                                   │
│  │  Pipeline       │──┐                                               │
│  │  (Component 7)  │  │                                               │
│  └──────────────┘  │                                                 │
│                     │                                                 │
│  ┌──────────────┐  │   ┌───────────────────┐                         │
│  │  Drift        │◀─┘──▶│  Reservation       │                        │
│  │  Estimator    │      │  Price             │                        │
│  │  (Kalman)     │      │  r(t,q)            │                        │
│  └──────┬───────┘      └────────┬──────────┘                        │
│         │                       │                                     │
│  ┌──────▼───────┐      ┌──────▼──────────┐                          │
│  │  Directional  │      │  Per-Level       │                          │
│  │  Flow         │─────▶│  Expected PnL    │──▶ Quotes               │
│  │  Model        │      │  & Optimal Size  │                          │
│  └──────────────┘      └──────┬──────────┘                          │
│                               │                                       │
│  ┌──────────────┐      ┌─────▼────────┐                              │
│  │  Continuous   │─────▶│  Funding      │                             │
│  │  γ(q)         │      │  Carry Cost   │                             │
│  └──────────────┘      │  (Component 6) │                             │
│                         └──────────────┘                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Bayesian Drift Estimator

**Owner**: Engineer 1  
**Replaces**: `drift_adj_bps` cap, `smoothed_drift` warmup, `signal_weight`

### Model

The drift estimator maintains a posterior distribution over the instantaneous price drift μ using a Kalman filter.

**State vector:**

```
x = [μ]    (scalar: drift rate in bps/sec)
```

**Transition model:**

```
μ(t+dt) = μ(t) · exp(-κ_μ · dt) + ε_μ

where:
  κ_μ  = mean-reversion rate of drift (units: 1/sec)
  ε_μ  ~ N(0, σ_μ² · dt)
  σ_μ  = process noise (how fast drift can change)
```

The mean-reversion ensures the drift estimate decays toward zero in the absence of new evidence. This replaces the warmup mechanism — there is no warmup. The prior starts at μ=0 with variance σ_μ²/2κ_μ (the stationary variance), and updates begin immediately.

**Observation model (price returns):**

```
Δp(t) = μ(t) · dt + σ_p · dW

Observation:  z = Δp / dt   (return rate)
Noise:        R = σ_p² / dt
```

**Observation model (fill events):**

Each fill on the bid side is an observation that the true price is more likely to be falling. Each fill on the ask side is an observation that it is rising. The signal strength depends on the fill's distance from mid:

```
z_fill = -sign(side) · (mid - fill_price) / σ_p
R_fill = σ_fill²   (fill observation noise, tuned by backtest)
```

A bid fill at 5 bps below mid in a low-vol environment is a stronger bearish signal than one at 1 bp below mid in a high-vol environment. This is not a threshold — it is a likelihood ratio.

**Kalman update (standard form):**

```
Predict:
  μ̂⁻  = μ̂ · exp(-κ_μ · dt)
  P⁻   = P · exp(-2κ_μ · dt) + σ_μ² · dt

Update (on each observation z with noise R):
  K  = P⁻ / (P⁻ + R)
  μ̂  = μ̂⁻ + K · (z - μ̂⁻)
  P  = (1 - K) · P⁻
```

**What this provides:**

- `μ̂`: point estimate of drift (bps/sec). Feeds directly into reservation price.
- `P`: posterior variance. When P is large (uncertain), μ̂ stays near 0 and quoting is symmetric. When P is small (confident), μ̂ moves and quoting skews. No arbitrary confidence thresholds.

### Parameters

| Parameter | Physical Meaning | Estimation Method |
|---|---|---|
| `κ_μ` | How fast drift regimes decay. ~0.01–0.05/sec (20–100s half-life) | Autocorrelation of returns at multiple lags |
| `σ_μ` | How volatile drift is. Controls how fast estimator can react. | Calibrate to observed regime change speed |
| `σ_fill` | How noisy fills are as drift signals. | Backtest: regress post-fill returns on fill signals |

These are not arbitrary thresholds. They are properties of the market that can be measured from historical data and updated online.

### Integration with Existing Trend Detector

The multi-timeframe trend detector (short/medium/long bps, agreement, p_continuation) is not deleted. It becomes an additional observation source for the Kalman filter:

```
z_trend = -trend_magnitude · agreement · p_continuation
R_trend = σ_trend² / (agreement²)   (higher agreement = lower noise)
```

When all timeframes agree on a bearish trend (agreement=1.0, p_continuation=0.83), this produces a strong, low-noise observation that rapidly shifts μ̂ negative. No `signal_weight` needed — the Kalman gain handles weighting automatically based on the relative confidence of this observation versus others.

---

## Component 2: Directional Flow Model

**Owner**: Engineer 2  
**Replaces**: `toxicity_size_mult`, `bid_toxicity`/`ask_toxicity` separate tracking

### Model

The current system estimates a single κ (order arrival intensity). This is insufficient because it conflates informed and uninformed flow.

Split into directional arrival rates:

```
κ_buy(t)  = arrival rate of buyers (lifting asks)
κ_sell(t) = arrival rate of sellers (hitting bids)
```

**Estimation:** exponentially weighted count of fills per unit time, by side:

```
κ_sell(t) = Σ exp(-λ · (t - t_i)) · 1[fill_i = bid]  /  Σ exp(-λ · (t - t_i))
κ_buy(t)  = Σ exp(-λ · (t - t_i)) · 1[fill_i = ask]  /  Σ exp(-λ · (t - t_i))
```

**Derived quantities:**

```
κ_total = κ_buy + κ_sell                     (total arrival intensity, replaces single κ)
flow_imbalance = (κ_sell - κ_buy) / κ_total  (range: -1 to +1)
informed_fraction = |flow_imbalance|          (VPIN-like measure)
```

### How This Feeds the System

1. **`flow_imbalance` feeds the drift estimator** as an additional observation:
   ```
   z_flow = -flow_imbalance · scale_factor
   R_flow = σ_flow² / κ_total   (more fills = more confident signal)
   ```

2. **`κ_buy` and `κ_sell` feed the GLFT directly** as side-specific arrival rates. When κ_sell >> κ_buy, the optimal bid depth is wider because adverse selection cost on bids is higher. This is not a multiplier on size — it is a direct input to the optimal spread computation.

3. **`informed_fraction` feeds the volatility estimate** (see Component 4): when informed traders are active, effective volatility is higher because price is about to move.

---

## Component 3: GLFT with Drift-Augmented Reservation Price

**Owner**: Engineer 3  
**Replaces**: `drift_adj_bps` (capped), `pre_fill_as` multipliers, the entire spread asymmetry mechanism

### The Reservation Price

The reservation price is the indifference price at which the market maker would be willing to trade, given current inventory and information.

```
r(t, q) = mid(t) + drift_shift(t) - inventory_penalty(t, q) - funding_carry(t, q)

where:
  drift_shift      = μ̂ · τ_eff / (γ(q) · σ²)
  inventory_penalty = q · γ(q) · σ² · τ_remaining
  funding_carry     = q · f̂(t) · τ_funding
  
  μ̂            = Kalman filter drift estimate (Component 1)
  τ_eff        = effective horizon for drift signal (seconds)
  γ(q)         = continuous risk aversion (Component 4)
  σ            = current volatility estimate
  q            = inventory in contracts
  τ_remaining  = time horizon for inventory penalty
  f̂(t)         = expected funding rate (Component 6)
  τ_funding    = expected holding time until next funding settlement
```

The `funding_carry` term is critical on perpetual futures. When funding is positive (longs pay shorts) and we are accumulating a long position, the reservation price drops — we demand more edge to compensate for the carry cost. When funding is negative (shorts pay longs), holding longs is subsidized and the reservation price rises slightly, allowing tighter bids. This is not a separate adjustment — it is a term in the same equation as drift and inventory.

**The drift_shift term is the critical change.** In the current system, drift shifts the mid by at most ±3 bps. In this system, there is no cap. If μ̂ is -20 bps/sec and the estimator is confident, the reservation price can shift by 15+ bps below mid. This is correct — it means fair value really is below the observed mid, and the market hasn't caught up yet.

### Optimal Quotes (per level)

For each ladder level i with depth d_i from the reservation price:

```
Bid depth:   δ_b(i) = (1/γ) · ln(1 + γ/κ_sell) + d_i
Ask depth:   δ_a(i) = (1/γ) · ln(1 + γ/κ_buy)  + d_i

Bid price:   p_b(i) = r - δ_b(i)
Ask price:   p_a(i) = r + δ_a(i)
```

Note: κ_sell appears in the bid depth (not κ_buy) because the adverse selection cost of a bid fill depends on the intensity of informed selling. When κ_sell is high, the `ln(1 + γ/κ_sell)` term is smaller, meaning the base spread from the GLFT formula alone tightens — but the drift_shift in the reservation price more than compensates by pulling r itself lower.

When the drift_shift is large and negative:
- All bids drop well below mid. The closest bid may be 15 bps below mid instead of 5.
- All asks drop too — but only by the drift_shift. They end up closer to mid, inviting fills that would reduce long inventory.
- This is the natural skew. No asymmetric multiplier needed.

### Per-Level Expected PnL (replaces quote gate)

For each bid level i, compute:

```
E[PnL_bid(i)] = P(fill_i) · [edge_i - E[adverse_move | fill_i] - E[carry_cost | fill_i]]

where:
  P(fill_i)     = κ_sell · exp(-κ_sell · δ_b(i) / σ) · dt
  edge_i        = mid - p_b(i)  (how far below mid we're buying)
  E[adverse_move | fill_i] = μ̂ · τ_holding + σ · sqrt(τ_holding) · AS_ratio
  E[carry_cost | fill_i]   = f̂ · τ_holding    (expected funding paid while holding)

  AS_ratio  = informed_fraction · E[informed_move] / σ
  τ_holding = expected time to unwind 1 unit (estimated from κ_buy)
  f̂         = expected funding rate per contract per second (Component 6)
```

The carry cost term means that in high-funding environments, E[PnL] goes negative sooner. If funding is +5 bps/hr and expected hold time is 10 minutes, that is ~0.8 bps of carry cost per fill — which matters when your edge per fill is only 2–3 bps. The system naturally widens bids (or stops bidding entirely) when funding makes it unprofitable to hold the resulting inventory, without any funding-specific threshold.

**When E[PnL_bid(i)] ≤ 0, the optimal size at that level is zero.**

This is not a gate. It is the natural optimum. The system does not "decide to pull bids" — the profitability of bidding is simply negative, so the optimal bid size at that level is zero. As conditions change, profitability may become positive again, and bids reappear.

The transition is smooth. As E[PnL] approaches zero from above, optimal size tapers continuously. As E[PnL] crosses zero, size reaches zero. There is no threshold, no discrete zone boundary, and no binary switch.

### Optimal Size Per Level

For levels where E[PnL] > 0:

```
s_i = argmax_s { s · E[PnL(i, s)] - risk_cost(s, q) }

where:
  risk_cost(s, q) = γ(q + s) · σ² · τ · s²    (marginal risk of adding s to inventory q)
```

This is a convex optimization in s, has a unique maximum, and naturally decreases as inventory grows (because γ(q) increases with q — see Component 4).

---

## Component 4: Continuous Inventory Penalty

**Owner**: Engineer 4  
**Replaces**: `position_zones` (green/yellow/red), `zone_size_mult`, `zone_spread_widen_bps`, `reduce_only_threshold`

### The Problem with Zones

Discrete zones create discontinuities. At 49% of max position (yellow zone), the system behaves one way. At 51% (red zone), it suddenly changes behavior. This is both exploitable and fragile — small fluctuations near a boundary cause oscillation.

### Continuous γ(q)

Replace with a smooth, convex risk aversion function:

```
γ(q) = γ_base · (1 + β · (q / q_max)²)
```

| q / q_max | γ(q) / γ_base (β=4) | Effect |
|---|---|---|
| 0.00 | 1.00 | Baseline risk aversion |
| 0.25 | 1.25 | Spreads widen 25%, sizes reduce slightly |
| 0.50 | 2.00 | Spreads double, sizes ~halve |
| 0.75 | 3.25 | Aggressive skew, thin quoting on accumulating side |
| 1.00 | 5.00 | Very wide spreads, near-zero size on accumulating side |

This feeds into every part of the system simultaneously:

1. **Reservation price**: higher γ → larger inventory penalty → reservation price further from mid on the accumulating side → wider spreads on that side.
2. **Optimal depth**: higher γ → `(1/γ)·ln(1+γ/κ)` changes shape → wider base spread.
3. **Optimal size**: higher γ → `risk_cost(s, q)` is larger → smaller optimal sizes.
4. **E[PnL] threshold**: higher γ → the risk cost term dominates sooner → E[PnL] crosses zero at wider depths.

All four effects compound smoothly. There is no zone boundary, no discrete behavioral change, and no separate multiplier table.

### Parameters

| Parameter | Physical Meaning | Estimation Method |
|---|---|---|
| `γ_base` | Risk aversion at zero inventory. Controls base spread. | Existing GLFT calibration (currently 0.15) |
| `β` | How much risk aversion amplifies with inventory. | Backtest: optimize for Sharpe over historical fills |
| `q_max` | Capacity limit. Where γ(q) becomes prohibitive. | Capital and margin constraints (currently ~8.5 units) |

The shape of γ(q) can be upgraded (exponential, power law, etc.) but the quadratic form is analytically tractable and sufficient.

---

## Component 5: Online Parameter Estimation

**Owner**: Engineer 5  
**Replaces**: Fixed `kappa_prior`, fixed `sigma_prior`, hardcoded warmup windows

### Principle

Every parameter in the system should have a principled default, be estimable from data, and update online. No parameter should require manual tuning per-asset or per-session.

### What Gets Estimated Online

| Parameter | Method | Window |
|---|---|---|
| σ (volatility) | Exponentially weighted realized vol | Rolling, ~300s (already exists) |
| κ_buy, κ_sell | Exponentially weighted fill counts by side | Rolling, ~120s |
| κ_μ (drift reversion) | Autocorrelation of returns | Calibrated offline, checked weekly |
| σ_μ (drift process noise) | Residual variance of drift innovations | Updated every ~60s |
| informed_fraction | From directional flow model | Continuous |

### Cold Start

At session start, all estimates use priors:

```
μ̂     = 0          (no drift assumed)
P      = σ_μ²/2κ_μ  (stationary variance — wide prior)
κ_sell = κ_prior     (current kappa estimate)
κ_buy  = κ_prior
σ      = σ_prior     (from asset metadata or recent history)
```

The Kalman filter naturally handles cold start. High posterior variance P means the Kalman gain K is large, so the first few observations move the estimate quickly. As P shrinks (more observations), the estimate stabilizes. There is no warmup period, no minimum trade count, and no warmup decay function.

---

## Component 6: Funding Carry Cost

**Owner**: Engineer 6 (shared with Integration)

### Why Funding Belongs in the Core Equation

On perpetual futures, funding is not a peripheral concern — it is a continuous cost or revenue stream that directly affects the profitability of holding inventory. The current system ignores it entirely. A market maker that accumulates 6.58 long contracts during a period of positive funding is paying to hold a losing position.

Funding also carries information. Extreme funding rates reveal crowding. High positive funding (longs paying shorts) means the market is net long — which means the sell pressure that is hitting our bids may be a liquidation cascade or a crowded-trade unwind. This is not just a cost to account for; it is a signal about the probability distribution of future price moves.

### Funding Rate Estimation

Hyperliquid publishes the current funding rate and the predicted next funding rate. The estimator maintains a short-term forecast:

```
f̂(t) = w_current · f_current + w_predicted · f_predicted + w_ewma · f_ewma

where:
  f_current    = current published funding rate (annualized or per-interval)
  f_predicted  = exchange's predicted next funding rate
  f_ewma       = exponentially weighted moving average of realized funding
  w_*          = weights (sum to 1), with w_predicted dominant near settlement
```

The forecast is converted to a per-second rate for use in the reservation price and E[PnL] equations.

### Funding as a Drift Signal

Extreme funding rates are informative about future price direction. The relationship is mean-reverting: high positive funding predicts eventual downward pressure (as longs capitulate or get liquidated), and high negative funding predicts upward pressure.

This enters the drift estimator as an additional observation:

```
z_funding = -sign(f̂) · funding_zscore · scale_factor

where:
  funding_zscore = (f̂ - f_mean) / f_std   (how extreme current funding is)
  f_mean, f_std  = rolling mean and std of funding rate (~24h window)
  scale_factor   = calibrated from historical funding → return regression

R_funding = σ_funding² / min(|funding_zscore|, 1)
            (extreme funding = higher confidence signal)
```

When funding is at +3σ (extremely positive), the drift estimator receives a strong bearish observation. This shifts the reservation price down, widening bids, before the liquidation cascade even begins. When funding is near its mean, R_funding is large and the observation has minimal impact.

### Carry Cost in Holding Time Estimation

The expected carry cost depends on how long we expect to hold inventory. This is already estimated as τ_holding in the E[PnL] computation. The carry cost term is:

```
E[carry_cost(q)] = |q| · f̂_per_second · τ_holding · sign_penalty

where:
  sign_penalty = {
    1.0   if sign(q) = sign(f̂)   (we're on the paying side)
    -1.0  if sign(q) ≠ sign(f̂)   (we're on the receiving side)
  }
```

When we are long and funding is positive (we pay), carry cost is positive — it subtracts from E[PnL]. When we are short and funding is positive (we receive), carry cost is negative — it adds to E[PnL], making short positions more attractive to hold.

### Interaction with Continuous γ(q)

Funding carry cost compounds with the inventory penalty. At high inventory AND high funding, the effective cost of adding one more unit is:

```
marginal_cost(q) = γ(q)·σ²·τ + f̂·τ_holding
```

Both terms grow with q (γ(q) directly, and τ_holding indirectly because it takes longer to unwind a larger position). This creates the correct convex penalty structure: the last contract added is much more expensive than the first, through both risk AND carry channels.

### Parameters

| Parameter | Physical Meaning | Source |
|---|---|---|
| f_current, f_predicted | Current/next funding rate | Exchange API (directly observable) |
| f_mean, f_std | Historical funding statistics | Rolling 24h window |
| scale_factor | Funding → drift signal strength | Offline regression: funding_zscore vs next-hour return |
| σ_funding | Observation noise for funding signal | Residual variance of funding → return regression |

These are all either directly observable or estimable from historical data. No arbitrary thresholds.

---

## Component 7: Feature Engineering Pipeline

**Owner**: Engineer 7 (dedicated)

### Principle

The drift estimator (Component 1) accepts observations through a uniform interface: a value z and a noise variance R. Any signal that is informative about future price direction can be added as an observation source without modifying the estimator itself.

The feature pipeline is the collection of signal extractors that feed this interface. Each feature has a physical interpretation, a measurable signal-to-noise ratio, and an offline-validated relationship to future returns. Features that do not survive out-of-sample validation are not included.

### Feature Categories

#### Category A: Order Book Microstructure

These features are derived from the L2 order book snapshot, available every quote cycle.

**A1. Book Imbalance (BIM)**

```
BIM(d) = (V_bid(d) - V_ask(d)) / (V_bid(d) + V_ask(d))

where:
  V_bid(d) = total bid volume within d bps of mid
  V_ask(d) = total ask volume within d bps of mid
  d        = depth parameter (typically 10, 25, 50 bps)
```

BIM is a well-studied predictor of short-term price direction. When bids outweigh asks at a given depth, the price is more likely to tick up. The signal is strongest at shallow depths (10 bps) and decays at deeper levels.

**Observation interface:**
```
z_bim = BIM(10) · α_bim     (α_bim calibrated offline)
R_bim = σ_bim² / book_depth_usd   (deeper books = more reliable signal)
```

**A2. Book Pressure Gradient**

BIM at a single depth is a snapshot. The gradient across depths reveals whether support/resistance is concentrated or distributed:

```
BPG = BIM(10) - BIM(50)

BPG > 0: support is concentrated near the touch (fragile — can be swept quickly)
BPG < 0: support is distributed deep (resilient — harder to move the price)
```

When BPG is strongly positive and we are long, the apparent bid support is fragile. This is an additional bearish observation:

```
z_bpg = -BPG · sign(q) · α_bpg    (when long, fragile bid support is bearish)
R_bpg = σ_bpg²
```

**A3. Book Acceleration (ΔBook)**

The rate of change of book imbalance is more informative than the level. If bids are being pulled faster than asks, the book is deteriorating even if BIM is still positive:

```
ΔBIM = (BIM(t) - BIM(t - Δt)) / Δt

where Δt = 1-3 quote cycles
```

A rapidly deteriorating bid side (ΔBIM << 0) precedes price drops. Market makers pulling their bids is informed behavior.

```
z_dbook = ΔBIM · α_dbook
R_dbook = σ_dbook² / max(1, n_observations)
```

#### Category B: Trade Flow Microstructure

These features are derived from the fill stream and trade tape.

**B1. Sweep Detection**

A sweep is a sequence of fills that walks through multiple price levels on one side within a short window. Sweeps indicate aggressive informed flow — someone is willing to pay increasing slippage to build or exit a position.

```
sweep_score = Σ (size_i · levels_crossed_i) for fills within τ_sweep window

where:
  levels_crossed = number of price levels consumed by this fill
  τ_sweep        = detection window (~1-3 seconds)
  only fills on one side contribute (bid sweeps vs ask sweeps tracked separately)
```

A bid-side sweep (aggressive selling through our bids) is a strong bearish observation:

```
z_sweep = -sweep_score_bid + sweep_score_ask
R_sweep = σ_sweep² / sweep_score_total   (larger sweeps = more confident)
```

**B2. Trade Arrival Clustering (Hawkes Intensity)**

Informed traders tend to cluster their trades. A burst of sells followed by a pause is a different signal than a steady trickle of sells at the same average rate. The Hawkes process models self-exciting arrival intensity:

```
λ(t) = λ_base + Σ α · exp(-β · (t - t_i))

where:
  t_i    = timestamps of recent fills
  α      = excitation parameter (how much each fill increases future arrival rate)
  β      = decay parameter (how fast the excitation fades)
  λ_base = baseline arrival rate
```

When λ(t) >> λ_base on the bid side, we are in a burst of selling. The excess intensity λ(t) - λ_base is the self-exciting component — it measures how much the current selling is feeding on itself.

```
z_hawkes = -(λ_sell(t) - λ_base_sell) + (λ_buy(t) - λ_base_buy)
R_hawkes = σ_hawkes² / (λ_sell(t) + λ_buy(t))
```

This naturally captures liquidation cascades: each liquidation triggers more liquidations, creating a self-exciting sell arrival pattern.

**B3. Fill Size Distribution Shift**

Informed traders tend to trade in sizes different from the baseline. A sudden appearance of large fills on the bid side (someone dumping) or a shift from many small fills to fewer large fills indicates a different type of counterparty.

```
size_zscore_bid = (mean_fill_size_bid_recent - mean_fill_size_bid_baseline) / std_fill_size

z_fillsize = -size_zscore_bid · α_fillsize
R_fillsize = σ_fillsize² / n_recent_fills
```

#### Category C: Cross-Venue and Derivative Signals

**C1. Spot-Perp Basis**

The basis between spot HYPE and the perpetual contract encodes aggregate positioning. When the perp trades at a premium to spot (positive basis), the market is net long via leverage. When the basis compresses or inverts, longs are unwinding.

```
basis(t) = (perp_mid - spot_mid) / spot_mid · 10000   (in bps)
Δbasis   = basis(t) - basis(t - Δt)
```

A rapidly compressing basis (Δbasis << 0) when we are long is a strong bearish signal — it indicates leveraged longs are exiting.

```
z_basis = Δbasis · α_basis
R_basis = σ_basis²
```

For HIP-3 assets where a direct spot feed may not exist, the basis can be approximated using the funding rate integral or by referencing the CEX spot price for HYPE.

**C2. Cross-Asset Correlation Signal**

HYPE's price is correlated with BTC and the broader crypto market. A move in BTC that hasn't yet propagated to HYPE is informative:

```
expected_hype_return = β_hype_btc · btc_return_recent
actual_hype_return   = hype_return_recent
residual             = actual_hype_return - expected_hype_return
lead_signal          = β_hype_btc · btc_return_very_recent - hype_return_very_recent

where:
  β_hype_btc         = rolling regression coefficient (HYPE return ~ BTC return)
  btc_return_recent  = BTC return over ~30s window
  hype_return_recent = HYPE return over same window
  very_recent        = ~5s window (captures lead-lag)
```

When BTC drops and HYPE hasn't followed yet, `lead_signal` is negative — HYPE is about to catch down. This is a leading indicator that arrives before any fills happen on our book.

```
z_lead = lead_signal · α_lead
R_lead = σ_lead² / β_confidence   (β_confidence from regression R²)
```

**C3. Open Interest Rate of Change**

Rising open interest alongside falling price indicates new short positions being opened — bearish continuation. Falling OI alongside falling price indicates long liquidations — potentially approaching a floor.

```
ΔOI = (OI(t) - OI(t - Δt)) / OI(t - Δt)

z_oi = ΔOI · sign(price_return) · α_oi
       (rising OI + falling price = bearish; falling OI + falling price = less bearish)
R_oi = σ_oi²
```

#### Category D: Volatility Regime Features

**D1. Realized Volatility Term Structure**

Compare short-term realized vol to longer-term:

```
vol_ratio = σ_realized_30s / σ_realized_300s
```

When vol_ratio >> 1, we are in a volatility expansion (recent moves are larger than the baseline). This doesn't indicate direction, but it inflates σ in the reservation price and E[PnL] equations, naturally widening spreads and reducing sizes.

```
σ_effective = σ_baseline · max(1.0, vol_ratio^ν)

where ν ∈ (0, 1) controls how much short-term vol shocks affect quoting.
      ν = 0.5 means sqrt-scaling: 4x vol spike → 2x spread widening.
```

This is not a separate "regime detection" overlay. It feeds directly into σ, which propagates through every equation.

**D2. Return Distribution Tail Detection**

Kurtosis of recent returns detects fat-tail regimes where adverse moves are larger than Gaussian models predict:

```
kurt = E[(r - μ)^4] / σ^4 - 3   (excess kurtosis, 0 for Gaussian)
```

When kurt > 0 (fat tails), the E[adverse_move] term in the PnL equation should use a fatter-tailed distribution. Practically, this scales the AS_ratio:

```
AS_ratio_adjusted = AS_ratio · (1 + κ_tail · max(0, kurt))
```

This is a principled correction: when the return distribution has fat tails, the expected adverse move conditional on being filled is larger than the Gaussian estimate.

### Feature Composition Rules

**Independence:** Features feed the drift estimator independently. The Kalman filter handles correlation implicitly through its update mechanics — correlated features that agree produce smaller posterior variance (higher confidence), while features that disagree produce larger posterior variance (appropriate caution).

**Validation gate:** A feature is included only if it satisfies:
1. Out-of-sample predictive R² > 0.01 for 10-second-ahead returns (low bar, but nonzero)
2. Signal is not redundant with existing features (incremental R² > 0.003)
3. Signal survives transaction cost adjustment (predicted edge > fee + spread cost)

Features that fail validation are excluded entirely — not down-weighted, not capped, excluded. This is the one place where a hard cutoff is justified: including noise features dilutes the Kalman filter's posterior precision.

**Latency tiers:** Features are grouped by computational cost and update frequency:

| Tier | Update Frequency | Examples |
|---|---|---|
| Hot (< 1ms) | Every quote cycle (~5-6s) | BIM, ΔBIM, fill flow, sweep detection |
| Warm (< 10ms) | Every 2-3 cycles | Hawkes intensity, basis, vol ratio |
| Cold (< 100ms) | Every 5-10 cycles | Cross-asset regression, OI, funding forecast |

Hot-tier features feed the drift estimator at full frequency. Warm and cold features use their most recent value between updates. The Kalman filter handles asynchronous observations natively — it simply runs a predict step between observation updates.

### Feature Parameter Estimation

All feature parameters (α_*, σ_*, scale factors) are estimated offline from historical data using the same methodology:

1. Compute raw feature values on historical tick data
2. Regress 10-second-ahead return on feature value
3. α = regression coefficient, σ = residual standard deviation
4. Out-of-sample validation on held-out period (walk-forward)
5. Parameters are frozen for live use and re-estimated weekly

This is standard supervised learning applied to microstructure features. The Kalman filter is the "model" that combines them; the feature pipeline is the "feature engineering" step that a team would iterate on.

### What This Looks Like in Practice

On Feb 19, with the full feature pipeline active:

```
22:06  All features near baseline. μ̂ ≈ 0. Normal quoting.

22:07  BTC drops 15 bps. HYPE hasn't moved yet.
       z_lead = -0.08 (bearish lead signal). μ̂ shifts slightly negative.
       Bids pull back ~2 bps. No fills yet.

22:08  HYPE starts following BTC down. Price returns confirm the lead signal.
       z_bim = -0.3 (bid side thinning). z_dbook = -0.15 (book deteriorating).
       z_lead + z_bim + z_price all agree. Posterior narrows. μ̂ → -8 bps/sec.
       Bids pull back ~10 bps. First fill would need to be 10+ bps below mid.

22:09  Funding at +2σ above mean. z_funding = -0.1 (moderate bearish carry signal).
       Hawkes intensity spiking on bid side: z_hawkes = -0.2.
       μ̂ → -14 bps/sec. E[PnL_bid] < 0. Bids at zero size.
       All of this happens BEFORE any significant fills on our book.

22:10+ We never accumulate more than ~0.5 contracts. The feature pipeline
       gave the drift estimator 4-5 independent bearish signals before position
       built up. The Feb 19 incident doesn't happen.
```

The difference between the base RFC (Components 1-5) and the full system (Components 1-8) is the difference between reacting to fills and anticipating them. The base system waits for bid fills to inform the drift estimator. The feature pipeline sees the BTC move, the book deterioration, the funding extreme, and the Hawkes clustering — all of which precede the fills. The drift estimator shifts μ̂ before a single contract trades.

---

## Component 8: Integration and Composition

**Owner**: Engineer 6 (shared with Funding)

### The Quote Cycle (Revised)

```
fn quote_cycle():
    // 1. Update drift estimator with latest price observation
    drift.predict(dt)
    drift.update_price(latest_return, dt)
    
    // 2. If any fills since last cycle, update with fill observations
    for fill in new_fills:
        drift.update_fill(fill.side, fill.price, mid)
        flow.record_fill(fill.side, fill.timestamp)
    
    // 3. If trend detector has new reading, update with trend observation  
    if trend.updated:
        drift.update_trend(trend.magnitude, trend.agreement, trend.p_continuation)
    
    // 4. Feature pipeline: compute and feed all active features
    //    Hot tier (every cycle):
    let bim = book.imbalance(10)
    let dbim = book.imbalance_delta(10, last_bim, dt)
    let sweep = flow.sweep_score(tau_sweep)
    let hawkes = flow.hawkes_intensity()
    drift.update_observation(bim.z, bim.R)
    drift.update_observation(dbim.z, dbim.R)
    drift.update_observation(sweep.z, sweep.R)
    drift.update_observation(hawkes.z, hawkes.R)
    
    //    Warm tier (every 2-3 cycles):
    if warm_cycle:
        let basis = venues.spot_perp_basis()
        let vol_ratio = vol.term_structure_ratio()
        drift.update_observation(basis.z, basis.R)
        sigma_effective = sigma_baseline * vol_ratio.scaling()
    
    //    Cold tier (every 5-10 cycles):
    if cold_cycle:
        let lead = venues.cross_asset_lead(btc_return, hype_return)
        let oi = exchange.oi_rate_of_change()
        let funding_signal = funding.zscore_observation()
        drift.update_observation(lead.z, lead.R)
        drift.update_observation(oi.z, oi.R)
        drift.update_observation(funding_signal.z, funding_signal.R)
    
    // 5. Read current state
    mu_hat = drift.mean()              // drift estimate
    P      = drift.variance()          // drift uncertainty
    sigma  = sigma_effective           // vol with term structure adjustment
    q      = current_inventory
    kappa_sell = flow.kappa_sell()
    kappa_buy  = flow.kappa_buy()
    gamma_q = gamma_base * (1.0 + beta * (q / q_max).powi(2))
    f_hat   = funding.forecast()       // expected funding rate (per second)
    
    // 6. Compute reservation price (THE critical line)
    drift_shift  = mu_hat * tau_eff / (gamma_q * sigma * sigma)
    inv_penalty  = q * gamma_q * sigma * sigma * tau_remaining
    fund_carry   = q * f_hat * tau_funding
    reservation  = mid + drift_shift - inv_penalty - fund_carry
    
    // 7. Compute optimal quotes per level
    for level in 0..num_levels:
        // Bid
        base_depth_bid = (1.0 / gamma_q) * (1.0 + gamma_q / kappa_sell).ln()
        bid_price = reservation - base_depth_bid - level_spacing(level)
        bid_epnl = expected_pnl_bid(bid_price, mid, mu_hat, sigma, 
                                     kappa_sell, gamma_q, q, f_hat)
        bid_size = if bid_epnl > 0.0 { 
            optimal_size_bid(bid_epnl, gamma_q, sigma, q) 
        } else { 0.0 }
        
        // Ask  
        base_depth_ask = (1.0 / gamma_q) * (1.0 + gamma_q / kappa_buy).ln()
        ask_price = reservation + base_depth_ask + level_spacing(level)
        ask_epnl = expected_pnl_ask(ask_price, mid, mu_hat, sigma, 
                                     kappa_buy, gamma_q, q, f_hat)
        ask_size = if ask_epnl > 0.0 { 
            optimal_size_ask(ask_epnl, gamma_q, sigma, q) 
        } else { 0.0 }
        
        emit_quote(level, bid_price, bid_size, ask_price, ask_size)
```

### What the Feb 19 Incident Looks Like Under This System

```
22:06  μ̂ ≈ 0, P large (wide prior). γ(0) = γ_base. Funding moderate.
       Normal quoting. 6 bid + 6 ask. Similar to current system.

22:08  Price observations: -5 to -10 bps returns.
       BTC dropped 15 bps 30s ago — lead signal z_lead = -0.08 (bearish).
       Book imbalance shifting: z_bim = -0.2 (bid side thinning).
       Kalman update: μ̂ → -6 bps/sec, P narrowing from multiple sources.
       drift_shift ≈ -7 bps. Bids pull back 7 bps from where they'd otherwise be.
       Still quoting bid levels but all shifted down. No fills on our book yet.

22:09  Funding at +2σ. z_funding = -0.1 (bearish carry signal).
       Hawkes intensity spiking on bid side: z_hawkes = -0.15.
       Book pressure gradient: support is shallow (BPG > 0). z_bpg = -0.1.
       Four independent bearish signals converging. Posterior narrows sharply.
       Kalman: μ̂ → -12 bps/sec.
       drift_shift ≈ -14 bps. E[PnL_bid] approaching 0 at touch.
       funding_carry term adds ~1 bps additional penalty per contract.

22:10  More bearish observations. First bid fills arrive on other MMs' books.
       Kalman update from fills: μ̂ → -16 bps/sec, P very narrow.
       flow_imbalance → -0.6 (more selling than buying).
       drift_shift ≈ -18 bps. Touch bid is 18+ bps below mid.
       E[PnL_bid] < 0 at all reasonable depths (drift cost + carry cost > edge).
       Bid sizes → 0 across all levels.
       Asks shift down toward mid (inviting unwind fills).
       q ≈ 0.3 (bought almost nothing because bids retreated before fills arrived).

22:12  Continued selling. Trend detector fires: agreement 1.0, p_cont 0.83.
       Trend observation confirms what features already showed.
       μ̂ holds at -16 bps/sec with very tight posterior.
       q ≈ 0.3. No additional accumulation.

22:15  Position: ~0.3 (not 2.12). Bids have been at zero size for 5 minutes.
       Asks are aggressive. Waiting for a bounce to unwind.

22:17  Position: ~0.3 (not 6.58). The selloff sweeps through other MMs, not us.
       Our maximum adverse exposure: ~$9 instead of ~$190.
```

The key difference: in the current system, protection is reactive (wait for inventory to build, then act). In this system, protection is predictive (drift estimate shifts reservation price before fills even happen).

---

## What Doesn't Change

- The GLFT mathematical framework (it was the right choice; the issue is what feeds it)
- The ladder strategy structure (multi-level quoting with geometric spacing)
- The tick grid discretization
- The reconciler / order management layer
- The kill switch (still needed as a last-resort safety net)
- The WebSocket execution infrastructure
- The margin / leverage calculations

---

## Migration Path

### Phase 1: Drift Estimator (Components 1 + 3)

Replace `drift_adj_bps` cap and `smoothed_drift` with the Kalman filter. Wire μ̂ into the reservation price. Remove the ±3 bps clamp. This alone would have prevented most of the Feb 19 damage.

**Test**: replay Feb 19 log. Verify that bids retreat before position exceeds 2.0.

### Phase 2: Directional Flow (Component 2)

Replace single κ with κ_buy/κ_sell. Remove `toxicity_size_mult`. Wire directional kappas into GLFT depths.

**Test**: on historical fills, verify that bid-side κ spikes precede adverse price moves.

### Phase 3: Continuous γ(q) (Component 4)

Replace position zones with smooth γ(q). Remove `zone_size_mult`, `zone_spread_widen_bps`, `reduce_only_threshold`.

**Test**: verify no discontinuities in spread or size as inventory varies from 0 to max.

### Phase 4: Expected PnL Gate (Component 3, full)

Replace the binary quote gate with per-level E[PnL] computation. Remove `quote_gate`, `signal_weight`.

**Test**: verify that E[PnL] < 0 naturally occurs before position reaches current reduce_only_threshold.

### Phase 5: Online Estimation (Component 5)

Replace fixed priors with online estimation. Remove hardcoded `kappa_prior`, `sigma_prior`.

**Test**: verify parameter convergence on live data within 60s of session start.

### Phase 6: Funding Integration (Component 6)

Add funding carry cost to reservation price and E[PnL]. Add funding rate as a drift estimator observation source.

**Test**: on historical periods with extreme funding, verify that quoting behavior adjusts — wider bids when funding is high positive and we're long, tighter bids when funding subsidizes our position. Verify that funding-zscore observations improve drift estimator accuracy on funding settlement windows.

### Phase 7: Feature Pipeline — Book Microstructure (Component 7, partial)

Add hot-tier features: BIM, ΔBIM, book pressure gradient, sweep detection. These are the lowest-latency, highest-value features. Wire into drift estimator.

**Test**: compute incremental R² of each feature for 10-second-ahead returns. Verify each feature passes the validation gate (R² > 0.01, incremental R² > 0.003). On Feb 19 replay, verify that book features shift μ̂ before fills arrive.

### Phase 8: Feature Pipeline — Trade Flow and Cross-Venue (Component 7, full)

Add warm and cold tier features: Hawkes intensity, fill size distribution, spot-perp basis, cross-asset lead signal, OI rate of change. Add volatility regime features (vol term structure, kurtosis).

**Test**: full walk-forward backtest over 30+ days of historical data. Measure: (a) Sharpe improvement vs Phase 5 baseline, (b) max drawdown reduction, (c) inventory half-life improvement. Features that do not improve (a) by at least 0.1 Sharpe are removed.

---

## Appendix A: Parameter Sensitivity

The system has tunable parameters organized by component. All have physical interpretations and measurable values.

### Core Parameters (Components 1-5)

| Parameter | Range | Sensitivity | If too high | If too low |
|---|---|---|---|---|
| κ_μ (drift reversion) | 0.005–0.05 | Medium | Drift estimate reverts too fast, misses trends | Drift estimate sticks, slow to correct after reversals |
| σ_μ (drift noise) | 0.001–0.01 | High | Estimator overreacts to noise | Estimator too sluggish to catch real regime changes |
| σ_fill (fill noise) | 0.5–5.0 | Low | Fills don't inform drift much | Individual fills cause large drift jumps |
| γ_base | 0.10–0.30 | Medium | Wider spreads, fewer fills, lower volume | Tighter spreads, more adverse selection exposure |
| β (inventory curvature) | 2–8 | Medium | Cuts off quoting too aggressively at moderate inventory | Insufficient protection at high inventory |
| τ_eff (drift horizon) | 10–60s | Medium | Drift signal amplified too much | Drift signal doesn't shift reservation price enough |

### Funding Parameters (Component 6)

| Parameter | Range | Sensitivity | If too high | If too low |
|---|---|---|---|---|
| scale_factor (funding→drift) | 0.01–0.10 | Medium | Overreacts to funding extremes, pulls quotes unnecessarily | Misses funding-driven regime changes (liquidation cascades) |
| σ_funding (observation noise) | 0.5–3.0 | Low | Funding barely informs drift | Normal funding fluctuations cause drift jumps |
| τ_funding (carry horizon) | 300–3600s | Medium | Overestimates carry cost, quotes too wide | Underestimates carry cost, holds losing positions too cheaply |

### Feature Parameters (Component 7)

Feature parameters (α_*, σ_*) are estimated per-feature via offline regression and are not manually tuned. The meta-parameters that control the pipeline are:

| Parameter | Range | Sensitivity | If too high | If too low |
|---|---|---|---|---|
| R²_threshold (inclusion gate) | 0.005–0.02 | Medium | Excludes marginal but useful features | Includes noise features that dilute signal |
| ν (vol scaling exponent) | 0.3–0.7 | Medium | Vol spikes cause excessive spread widening | Underreacts to vol regime changes |
| κ_tail (kurtosis scaling) | 0.1–0.5 | Low | Overstates adverse move in fat-tail regimes | Underestimates tail risk |
| τ_sweep (sweep window) | 1–5s | Low | Catches more sweeps but noisier | Misses slower sweeps |
| λ (Hawkes decay) | 0.1–1.0/s | Medium | Excitation fades too fast, misses sustained pressure | Intensity stays elevated too long after pressure ends |

### Calibration Approach

Phase 1 (offline): Compute κ_μ and σ_μ from historical price series using maximum likelihood on the OU process. Compute σ_fill by regressing post-fill price moves on fill signals. Set γ_base from existing calibration. Set β by grid search over historical P&L with different curvatures. Compute funding regression (funding_zscore vs next-hour return) to set scale_factor and σ_funding. Compute all feature α and σ values via walk-forward regression on historical tick data.

Phase 2 (online): Core parameters (σ, κ_buy, κ_sell, σ_μ, informed_fraction) update online via their respective estimators. Funding statistics (f_mean, f_std) update on a rolling 24h window. Feature parameters (α_*, σ_*) are frozen for live use and re-estimated weekly via automated pipeline. β, τ_eff, and the feature validation gate are reviewed weekly.

---

## Appendix B: Why Not Thresholds

A threshold-based system has a fundamental structural problem: the threshold value encodes a point estimate of where behavior should change, with no uncertainty. When the market shifts faster than the threshold allows, or slower, the threshold is wrong.

A Bayesian system encodes a full posterior. When uncertainty is high, the system is naturally conservative (wide spreads, symmetric quoting). When uncertainty is low, it acts decisively. The transition between these regimes is continuous and driven by evidence, not by a hardcoded number.

The practical consequence: a threshold needs to be "right" for all market conditions. A Bayesian estimator adapts to whatever conditions it observes, because its behavior is derived from the posterior, not from a point estimate.