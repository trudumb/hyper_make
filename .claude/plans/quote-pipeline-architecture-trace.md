# Quote Pipeline Architecture Trace

## Complete Data Flow: Market Data → Exchange Orders

Traced from `update_quotes()` in `quote_engine.rs` through all downstream decision points.

---

## Stage 1: Gate Checks (quote_engine.rs:29-277)

**Purpose**: Block quoting when conditions are unsafe.

| Gate | Line | Action |
|------|------|--------|
| Data quality | 33 | Cancel all orders if data stale/crossed |
| Warmup | 63-110 | Wait for estimator OR timeout (force complete with Bayesian priors) |
| Circuit breaker | 113-144 | PauseTrading / CancelAll / WidenSpreads |
| Risk position check | 147-152 | Hard limit on position notional |
| Drawdown | 155-158 | Emergency pause |
| HIP-3 OI cap | 253-260 | Skip if OI cap reached |
| Rate limit | 263-277 | Minimum requote interval |
| Recovery | 318-323 | IOC recovery skips normal quoting |

**Redesign Principle Impact**:
- **Inventory Governor**: No position check uses `config.max_position` here. Risk position check uses `position_notional` which is sensible but disconnected from the later effective_max calculation.

---

## Stage 2: Belief & Calibration Updates (quote_engine.rs:160-315)

**Purpose**: Update Bayesian beliefs, AS calibration, quote outcome tracking.

- Price returns → centralized belief system (line 189)
- Calibration controller → fill-hungry gamma multiplier (line 224)
- Quote outcome tracker → expire old quotes (line 237)
- Belief snapshot taken for entire cycle (line 243)

**No decision points** — pure state updates.

---

## Stage 3: Position Limit Computation (quote_engine.rs:348-399)

**Purpose**: Compute `effective_max_position` from multiple sources.

**Critical Path**:
```
margin_quoting_capacity = (available_margin × max_leverage) / latest_mid
dynamic_max_position = max_position_value / latest_mid   (from kill switch)

pre_effective_max_position = min(margin_quoting_capacity, dynamic_max_position)
    → if max_position_user_specified: min(above, config.max_position)
    → else: above as-is
```

**BUG #5 — Position Limit Bypass**: When `max_position_user_specified = false` (which is the case when the user provides `--max-position` via CLI but the flag isn't set), the `config.max_position` is NEVER applied. The margin-based capacity can be 10-100x larger.

**Redesign Principle Impact**:
- **Inventory Governor**: `config.max_position` should ALWAYS be the ceiling, regardless of how it was specified. The `max_position_user_specified` flag is a design smell — the user's intent should be respected unconditionally.

---

## Stage 4: Signal Integration (signal_integration.rs:696-970)

**Purpose**: Aggregate cross-venue, flow, regime, and model-gating signals into spread/skew adjustments.

**Key outputs**:
- `combined_skew_bps` — directional skew for asymmetric quoting
- `total_spread_mult` — additive spread widening from signals
- `kappa_effective` — regime-conditioned fill intensity

### Skew Path:
```
base_skew_bps = lead_lag_skew_bps × skew_direction   (if lead_lag_actionable)
                0.0                                     (if not actionable)

cross_venue_skew_bps = skew × confidence × (max_skew/2) × gate  (if cross_venue_valid)
                       0.0                                        (if not valid)

buy_pressure_skew_bps = excess × bps_per_z × z.signum()  (if warmed up + z > threshold)
                        0.0                                 (if not)

combined_skew_bps = base + cross_venue + buy_pressure
```

**BUG #2 — Zero Skew**: For HYPE (no Binance feed), `lead_lag_actionable = false`, `cross_venue_valid = false`. The HL-native fallback (line 896-908) uses `imbalance_5s` and `imbalance_30s`, but **BUG #5 from audit**: these are hardcoded to 0 because the HL flow feature vector is never populated with actual data (`set_hl_flow_features` is called but `imbalance_30s` comes in as 0).

**HL-Native Fallback**:
```
if !lead_lag_actionable && !cross_venue_valid && combined_skew == 0:
    flow_dir = imbalance_5s × 0.6 + imbalance_30s × 0.4    // Both 0!
    combined_skew = (flow_dir × cap).clamp(-cap, cap)        // = 0
```

Result: `combined_skew_bps = 0` always on HYPE.

### Spread Multiplier Path:
```
informed_excess = informed_flow_spread_mult - 1.0
gating_excess = gating_spread_mult - 1.0
cross_venue_excess = cross_venue_spread_mult - 1.0
total_excess = (sum).clamp(-0.1, max_excess)
total_spread_mult = 1.0 + total_excess
```

This is additive (good) — prevents multiplicative explosion.

### Regime Kappa Path:
```
kappa_effective = regime_kappa.kappa_effective()   // Computed
current_regime = regime_kappa.current_regime()      // Computed
```

**BUG #3 — Dead Kappa**: `kappa_effective` is stored in `signals.kappa_effective` and later passed to `market_params.regime_kappa`, but the CONSUMPTION path in `generate_ladder` only uses it with 60% blend weight. The original path through `signal_integration.rs` computed `regime_kappa_spread_mult` in the `SignalContributionRecord` but this was purely diagnostic — it was never applied to spread_multiplier or total_spread_mult.

**Redesign Principle Impact**:
- **Asymmetric Pricing**: Skew should be the DEFAULT, not gated by `lead_lag_actionable`. Every quote should have directional skew from inventory, flow, and momentum.
- **Regime State Machine**: `kappa_effective` is computed but partially consumed — 60% blend in ladder is good but only happened after Feb 11 fix (line 688-699 in ladder_strat.rs).

---

## Stage 5: Market Params Assembly (quote_engine.rs:520-586)

**Purpose**: Build `MarketParams` from `ParameterSources` via `ParameterAggregator::build()`.

Key fields populated:
- `sigma`, `kappa`, `microprice`, `flow_imbalance`, `momentum_bps`
- `lead_lag_signal_bps` — set to 0.0 initially, wired in Stage 6
- `effective_max_position` — from margin, kill switch, or config fallback
- `predicted_alpha` — P(informed|fill) from AS estimator
- `regime_kappa` — set to None initially, wired in Stage 7

---

## Stage 6: Signal Integration into MarketParams (quote_engine.rs:899-1025)

**Purpose**: Wire `IntegratedSignals` into `MarketParams`.

**Skew wiring**:
```
if lead_lag_actionable:
    market_params.lead_lag_signal_bps = combined_skew_bps + position_guard_skew_bps
else:
    // momentum fallback (if lag model warmed up and lag < 0)
    market_params.lead_lag_signal_bps = momentum_bps × confidence × 0.5
    // always add position guard
    market_params.lead_lag_signal_bps += position_guard_skew_bps
```

**Position guard**: `position_guard_skew_bps` is always additive — this is the ONE source of inventory-based skew.

**Redesign Principle Impact**:
- **Asymmetric Pricing**: `position_guard_skew_bps` is the ONLY unconditional skew source. But it's a linear inventory skew, not the proper GLFT q×γ×σ² skew. Proper GLFT skew should be computed IN the ladder strategy, not as an external additive adjustment.

---

## Stage 7: Spread Multiplier Composition (quote_engine.rs:1026-1180)

**Purpose**: Compute final `spread_multiplier` from all risk sources.

```
spread_multiplier = 1.0                              // base
  × circuit_breaker_mult                              // if WidenSpreads
  × threshold_kappa_mult                              // momentum regime
  × 1.0 (regime_kappa_mult hardcoded to 1.0)          // DEAD — no longer applied
  × model_gating_mult                                 // low model confidence
  × staleness_mult                                    // stale signals
  × toxicity_mult                                     // proactive toxicity
  × defensive_mult                                    // fill-based defense
  × risk_overlay.spread_multiplier                    // changepoint/trust
  → capped at max_composed_spread_mult
```

Then: `market_params.spread_widening_mult *= spread_multiplier`

**BUG #6 — InformedFlow Tightening**: `informed_flow_spread_mult` goes through the additive path in `total_spread_mult` (Stage 4) but is NOT part of the multiplicative chain here. However, it has a separate effect through `gating_adj_bps` — and audit found it TIGHTENS spreads (marginal value -0.23 bps). This is because when `p_informed` is low, `informed_flow_spread_mult < 1.0`, which makes `informed_excess < 0`, pulling the additive spread down.

**Redesign Principle Impact**:
- **Spread as Risk Buffer**: The multiplicative chain has 7+ independent multipliers. In practice, MOST are 1.0 in normal conditions, but during stress they compound. The additive approach in Stage 4 is better but only covers 3 signals. Should consolidate into ONE risk-adjusted spread computation.

---

## Stage 8: Ladder Generation (ladder_strat.rs:573-1170)

**Purpose**: Compute actual bid/ask prices and sizes.

### 8a. Gamma Computation (608-654)
```
if adaptive_mode:
    gamma = adaptive_gamma × tail_risk × calibration × bayesian
else:
    gamma = legacy_gamma × bayesian   // legacy includes calibration
```

### 8b. Kappa Selection (656-725)
```
PRIORITY: kappa_robust > adaptive_kappa > legacy(kappa × (1-alpha))

THEN regime blending:
    kappa = 0.4 × selected_kappa + 0.6 × regime_kappa

THEN AS feedback:
    if alpha > 0.3 && warmed_up && not_legacy:
        kappa *= (1.0 - 0.5 × alpha)
```

### 8c. GLFT Spread Computation (884-934)
```
dynamic_depths = depth_generator.compute_depths(gamma, kappa, sigma)
    → GLFT formula: δ = (1/γ) × ln(1 + γ/κ)  per level

THEN floor clamp:
    for each depth: if depth < effective_floor_bps: depth = effective_floor_bps
```

**BUG #3 DETAIL — Floor Binding**: In the live session, the GLFT optimal was ~2.87 bps but the floor was ~8.0 bps. Floor overrode 100% of cycles. The Feb 11 fix added `solve_min_gamma()` to constrain γ so GLFT >= floor, but this only works in the adaptive path. The diagnostic at line 946-955 tracks floor-binding frequency.

### 8d. Spread Widening (market_params.spread_widening_mult)
**IMPORTANT**: `spread_widening_mult` is applied through `depth_generator` depth computation (via `compute_depths_with_dynamic_bounds` which uses sigma and kappa) BUT the multiplicative chain from Stage 7 goes into `market_params.spread_widening_mult` which... is NOT consumed inside `generate_ladder` at all!

Wait — let me verify. The `spread_widening_mult` is set on `market_params` at line 1170 of quote_engine.rs, AFTER the `ParameterAggregator::build()` call. But `generate_ladder` receives `&market_params`. Let me check if it's used.

Actually, looking more carefully: `market_params.spread_widening_mult` is used inside `market_params.effective_spread_floor()` (the static path) and may flow through `compute_stochastic_constraints()`. The dynamic depths are computed from gamma and kappa — the widening mult would need to flow through sigma scaling or gamma scaling to actually widen the GLFT spread. The comment at line 1010-1023 says "all risk factors now flow through gamma" — but `spread_widening_mult` is NOT flowing through gamma in the current code. It's applied to `effective_floor_bps` in the static path only.

This is a subtle but important observation: the 7-multiplier chain from Stage 7 may not actually widen the GLFT spread in adaptive mode.

### 8e. Margin-Split Sizing (1048-1170)
```
usable_margin = available_margin × 0.70
ask_margin_weight = 0.5 + weighted_signals
    where signals = inventory(50%) + momentum(25%) + urgency(10%) + directional(15%)
available_for_asks = usable_margin × ask_weight
available_for_bids = usable_margin × (1 - ask_weight)
```

This is the ONLY place where margin allocation asymmetry happens based on signals. It's good but disconnected from the GLFT skew.

---

## Stage 9: Post-Ladder Filtering (quote_engine.rs:2119-2380)

**Purpose**: Filter generated quotes for safety.

Order of filters (each can clear quotes):

1. **Quote Gate** (2148-2176): One-sided filtering (OnlyBids/OnlyAsks)
2. **Risk Level Emergency** (2183-2201): Clear position-increasing side
3. **Fill Cascade** (2207-2256): 3+ same-side fills → widen, 5+ → suppress
4. **Graduated Emergency Pull** (2258-2327): Graduated urgency (0-1.0)
5. **Reduce-Only** (2329-2380): Position/margin/liquidation limits

### Emergency Pull (Graduated):
```
pull_urgency = f(is_opposed, inventory_frac, momentum_abs_bps)

if urgency > 0.7:  clear increasing side completely
if urgency > 0.4:  keep only innermost level
if urgency > 0.0:  halve sizes on increasing side
```

**BUG #4 — Emergency Paralysis**: The old binary emergency pull (from `StochasticController.risk_assessment()`) fires at `cp_prob > 0.95` with 50-cycle cooldown. During the live session, it fired 903 times, locking out quoting for 78/89 minutes. But this is the OLD path — the graduated pull (compute_pull_urgency) is the NEW path. The issue was:
- The controller's `risk_assessment()` returned `RiskLevel::Emergency` (line 1032) which went through the SEPARATE filter at line 2183-2201
- This cleared ALL position-increasing quotes
- Since the bot was long and only asks survived, it could only sell
- But if no asks were generated (min_notional filtering), it was paralyzed

The graduated pull is better but ALSO only affects the increasing side. If you're long and need to sell (reduce), these filters don't help — they ONLY clear bids. The REDUCE path comes from the reduce-only filter (Stage 9.5) which allows asks through.

The PARALYSIS occurs when:
1. Emergency pull clears bids (reducing side for short, or increasing side for long)
2. Reduce-only clears bids (increasing for long)
3. But ask generation fails (insufficient margin/min_notional)
4. Result: NO quotes on either side → position stays, losses continue

---

## Stage 10: Reconciliation (reconcile.rs)

**Purpose**: Submit filtered quotes to exchange.

Two paths:
- `reconcile_ladder_smart`: Uses ORDER MODIFY for queue preservation
- `reconcile_ladder_side`: Legacy all-or-nothing

### `place_bulk_ladder_orders` Position Checks (reconcile.rs:1810-1846):
```
max_pos = if max_position_user_specified:
    min(effective_max_position, config.max_position)
else:
    effective_max_position

// Hard limit: block if position >= max_pos AND increasing
// Reduce-only: block if position >= 0.95 × max_pos AND increasing
```

This is defense-in-depth — same `max_position_user_specified` guard.

---

## Root Cause Analysis: 6 Audit Bugs

### Bug 1: AS Tautological
**Where**: `handlers.rs:857-862` — `mid_at_fill: self.latest_mid`
**Root cause**: When a fill happens, `mid_at_fill` is recorded as `self.latest_mid` at fill processing time. The 5-second markout at `handlers.rs:250-283` computes:
```
mid_change_bps = (latest_mid_now - mid_at_fill) / mid_at_fill × 10000
```
The issue was `mid_at_fill = latest_mid` at the SAME INSTANT, so short-term AS ≈ 0.

**Fix applied** (partially): Line 798-802 now looks up `tracked.mid_at_placement` from the order tracker. However, fallback is still `self.latest_mid` for untracked orders. And `predicted_as_bps` at line 288/329 uses `self.estimator.total_as_bps()` which is the estimator's model, NOT the placement-based measurement — so the PREDICTION side is still using a potentially stale value.

### Bug 2: Zero Skew
**Where**: `signal_integration.rs:844-848` and `896-908`
**Root cause**: `lead_lag_actionable = false` (no Binance), `cross_venue_valid = false`, `buy_pressure` not warmed up. HL-native fallback uses `imbalance_5s/30s` which are 0.0 because `set_hl_flow_features` receives a vector with zero values for `imbalance_30s`.

### Bug 3: Regime Kappa Dead
**Where**: `quote_engine.rs:1065-1078` — `regime_kappa_mult = 1.0` hardcoded
**Root cause**: The regime kappa was previously multiplied into spread, but this was removed. Instead it now flows through `market_params.regime_kappa` → `ladder_strat.rs:688-699` where it's blended at 60% weight. So it IS consumed now, but the 8.4x variation (kappa 500-4200 across regimes) gets dampened to at most 1.5x via the 60% blend because the base kappa is also regime-adjacent.

### Bug 4: Emergency Paralysis
**Where**: `control/mod.rs:530-605` and `quote_engine.rs:2183-2201`
**Root cause**: `risk_assessment()` returns `Emergency` based on `cp_prob > 0.95`. The filter at line 2183 clears position-increasing side. If the bot is long, bids are cleared. If ask generation then fails (min_notional or margin), NO quotes at all → paralysis. The 50-cycle cooldown (line 579) was added but during high-cp periods it re-triggers constantly.

### Bug 5: Position Limit Bypass
**Where**: `quote_engine.rs:374-394` and `ladder_strat.rs:591`
**Root cause**: `effective_max_position` derives from margin (potentially 10-100x config), and `config.max_position` is only applied when `max_position_user_specified = true`. In `generate_ladder`, `quoting_capacity()` (line 590) explicitly ignores `max_position` — "user's max_position is ONLY used for reduce-only filter, not quoting capacity."

### Bug 6: InformedFlow Harmful
**Where**: `signal_integration.rs:762-775`
**Root cause**: When `p_informed` is low, `spread_multiplier(effective_p)` returns < 1.0, making `informed_excess < 0`, which TIGHTENS the additive spread. This is intentional (tighten when safe) but audit found the tightening is net-harmful because the p_informed estimate itself is unreliable.

---

## Architectural Coupling Analysis

### What's Entangled That Shouldn't Be

1. **Position limits computed in 4+ places**:
   - `quote_engine.rs:374-394` (pre_effective_max_position)
   - `quote_engine.rs:1237-1250` (effective_max_position from MarketParams)
   - `quote_engine.rs:1331-1335` (proactive adjustments)
   - `ladder_strat.rs:590-595` (quoting_capacity override)
   - `reconcile.rs:1810-1846` (defense-in-depth)
   - `quoting/filter.rs:489-639` (reduce-only)

   Each has different logic, different guards, different defaults. No single source of truth.

2. **Spread widening has two disjoint paths**:
   - Multiplicative chain → `market_params.spread_widening_mult` (7 multipliers)
   - GLFT gamma/kappa → `dynamic_depths` in generate_ladder
   - These DON'T interact — the multiplicative chain from Stage 7 goes into `spread_widening_mult` which is only consumed in the STATIC floor path, not the GLFT depth computation.

3. **Skew computed in 3 separate places**:
   - `signal_integration.rs:844-890` → `combined_skew_bps` → `lead_lag_signal_bps`
   - `position_guard` → `position_guard_skew_bps` (additive in quote_engine)
   - `ladder_strat.rs:729-733` → `inventory_ratio` for GLFT q-term
   - These are all additive but semantically different. Position guard duplicates what GLFT's inventory ratio already does.

4. **Regime information flows through 3+ independent channels**:
   - HMM probabilities → `regime_probs` in MarketParams
   - `RegimeState` → `regime_kappa` in MarketParams
   - `ThresholdKappa` → `threshold_kappa_mult` in spread multiplier
   - `BOCPD` → kappa discount (line 644-658)
   - These all try to widen spreads in volatile regimes but don't coordinate.

5. **Emergency response has multiple competing mechanisms**:
   - Circuit breaker (PauseTrading, CancelAll, WidenSpreads)
   - Risk overlay (Emergency, Elevated, Normal from controller)
   - Graduated pull (compute_pull_urgency)
   - Fill cascade tracker
   - Reduce-only filter
   - Kill switch (separate, in handlers)

   These can conflict: e.g., reduce-only allows position-reducing quotes, but emergency pull clears the OTHER side, and if the remaining side fails min_notional → paralysis.

---

## Decision Points Where Redesign Principles Would Change Flow

### Principle 1: Spread as Risk Buffer
**Current**: Floor overrides GLFT 100% of time in paper/early live. Multiplicative chain disconnected from GLFT.
**Change needed**:
- Make gamma the SOLE spread control variable
- `solve_min_gamma()` should be the PRIMARY path, not a fix
- All risk signals (toxicity, staleness, model confidence) → gamma adjustment
- Remove parallel multiplicative spread chain

### Principle 2: Asymmetric Pricing
**Current**: Skew is 0 unless Binance feed active or HL flow features populated.
**Change needed**:
- Position guard skew should be integral to GLFT (q-term), not additive
- Flow imbalance → skew even without cross-venue
- Fix HL flow feature vector so `imbalance_30s != 0`
- Default to asymmetric quotes based on inventory alone

### Principle 3: Inventory Governor
**Current**: Position limits computed in 4+ places with different logic. `config.max_position` only enforced conditionally.
**Change needed**:
- Single `InventoryGovernor` struct, computed once, passed everywhere
- `config.max_position` is ALWAYS the ceiling
- Quoting capacity, reduce-only threshold, kill switch limit ALL derive from one source
- Defense-in-depth checks all use the same source

### Principle 4: Hedging
**Current**: No hedging mechanism. Residual risk accumulates until reduce-only or kill switch.
**Change needed**:
- After each fill, compute residual risk (position × expected vol)
- If residual > threshold, hedge via aggressive limit order or IOC
- Reduce-only should be the LAST resort, not the primary position control

### Principle 5: Regime State Machine
**Current**: Regime information flows through 4 independent channels that don't coordinate.
**Change needed**:
- Single `RegimeState` enum (Calm, Normal, Volatile, Extreme)
- ALL parameters (gamma, kappa, floor, skew sensitivity, max_position) switch based on regime
- One source of truth consumed by all downstream logic
- Transitions logged and auditable
