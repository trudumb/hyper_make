# Principled Architecture Redesign

## Premise

This document redesigns the market maker as if the five strategy principles from `docs/market-maker-strategy-redesign.md` were **foundational assumptions**, not afterthoughts. It draws on evidence from two live mainnet sessions (Feb 12), a 6-bug systemic audit, algebraic proof of the AS tautology, and deep analysis of the current codebase by 4 parallel agents.

The current system has these five principles implemented as **overlays** on a fundamentally symmetric, floor-dominated, margin-derived core. Each overlay fights the others. The redesign makes them **the system itself**.

---

## Evidence Base

### Live Session Feb 12 19:37 (this log, 22K lines, ~1hr, 103 fills)
- `effective_max_position` = 53-58 HYPE from margin vs `config.max_position` = 3.24 HYPE
- Floor binding every cycle: GLFT optimal 3.04-5.58 bps, floor 5.51-10.01 bps
- `directional` signal computed but never produces skew (only affects margin split)
- Untracked fill of 3.14 HYPE (cancel-fill race → position jumped to -5.77)
- Edge prediction: always ~-0.7 bps predicted, realized -7 to +9 bps (no predictive power)
- 3 fill cascades triggered, position swung -5.87 to +4.99 during session
- Sharpe(all) = 439 reported (unreliable — AS tautological)

### Live Session Feb 12 03:16 (earlier, 89min, 29 fills, -$5.02)
- Position reached 12.81 HYPE (3.95x configured max)
- Risk overlay pulled ALL quotes for 78 of 89 minutes
- 903 emergency pulls, 35-minute dead zone with position stuck
- Kill switch triggered at -$5.02 daily loss

### 6 Systemic Bugs (audit)
1. **AS measurement tautological** — `as_realized` and `depth_from_mid` both use `latest_mid`
2. **Zero directional skew** — no signal produces `skew_adjustment_bps`
3. **RegimeDetection dead output** — kappa 500-4203 (8.4x) computed but never consumed
4. **Emergency pull paralysis** — cancels ALL quotes including reduce-only
5. **Position limit not enforced** — `effective_max_position` from margin >> config
6. **InformedFlow harmful** — tightens spreads, marginal value -0.23 bps

### Root Causes (from agent analysis)
- **Position**: `ladder_strat.rs` computes quoting capacity from `margin_available * leverage / price`. The `user_max_position` only gates reduce-only mode, not position-increasing orders. `place_bulk_ladder_orders` cumulative size check uses margin-derived limits.
- **Spread**: `generate_ladder` computes `effective_floor_bps = AS_floor + conditional_AS_buffer`, then clamps GLFT to floor. Even with `solve_min_gamma`, the AS buffer (2 bps) pushes floor above GLFT output.
- **Skew**: `signal_integration.rs` `compute_signals()` returns `SignalContributions` with only `spread_adjustment_bps`. The skew computation path exists in `ladder_strat.rs` via `directional` signal but it only affects `margin_for_bids`/`margin_for_asks` split, not quote prices.
- **Regime**: `estimator/volatility/regime.rs` produces `VolatilityRegime` (Low/Normal/High/Extreme) and kappa estimates. `quote_engine.rs` reads regime for logging but spread composition chain uses `kappa_robust` from `kappa_orchestrator.rs`, ignoring regime kappa entirely.
- **Emergency pull**: `control/mod.rs` `risk_assessment()` sets `emergency_pull = true` when changepoint > 0.95. `quote_engine.rs` `apply_emergency_pull()` clears both sides unconditionally with `compute_pull_urgency()` determining size reduction.

---

## The Architecture

### Design Principle

Every component has exactly one job. No component fights another. The pipeline flows in one direction with no backtracking. Position safety is structural, not advisory.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEASUREMENT SUBSTRATE                         │
│  OrderLifecycle: mid_at_placement → fill → markout → edge       │
│  (Foundation — nothing learns until this is calibrated)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 L0: INVENTORY GOVERNOR                           │
│  config.max_position is HARD CEILING. Three zones:              │
│  GREEN (<50%) → full quoting                                    │
│  YELLOW (50-80%) → bias toward reducing + wider increasing      │
│  RED (>80%) → reduce-only, both sides                           │
│  KILL (>100%) → cancel all, alert                               │
│  ⚡ First check. Exits immediately if KILL zone.                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ zone, max_new_exposure
┌──────────────────────────▼──────────────────────────────────────┐
│                 L1: REGIME STATE MACHINE                         │
│  Input: σ (GARCH), κ (kappa_orchestrator), OI_delta, book_depth │
│  Output: RegimeParams {                                         │
│    kappa, sigma, gamma_base,                                    │
│    as_expected_bps, risk_premium_bps,                           │
│    skew_sensitivity, max_position_fraction                      │
│  }                                                              │
│  Transitions have hysteresis (EMA smoothed, not per-tick)       │
│  Three states: QUIET, NORMAL, VOLATILE (+ CRISIS override)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ RegimeParams
┌──────────────────────────▼──────────────────────────────────────┐
│                 L2: SPREAD ENGINE                                │
│  half_spread = (1/γ)·ln(1 + γ/κ) + fee                        │
│                                                                  │
│  γ = solve_min_gamma(target_floor, κ, σ, T)                    │
│  target_floor = fee + AS_expected + risk_premium                │
│  All three components from RegimeParams (measured, not magic)   │
│                                                                  │
│  NO separate floor. NO AS buffer. NO kappa cap.                 │
│  The math produces the right spread because γ is self-consistent│
│  with the regime-specific floor target.                         │
│                                                                  │
│  Invariant: half_spread ≥ fee + AS_expected + risk_premium      │
│  (by construction, not by clamping)                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ half_spread_bps
┌──────────────────────────▼──────────────────────────────────────┐
│                 L3: ASYMMETRIC PRICER                            │
│                                                                  │
│  mid_effective = mid + inventory_skew + signal_skew             │
│                                                                  │
│  inventory_skew:                                                │
│    = -(position / max_position) × skew_sensitivity × half_spread│
│    Always active. If position ≠ 0, skew ≠ 0. Period.           │
│    Scales with half_spread so skew is proportional to risk.     │
│                                                                  │
│  signal_skew:                                                   │
│    Each signal produces normalized output in [-1, 1]:           │
│    - Order flow imbalance (measured from trade tape)            │
│    - Momentum (multi-timeframe MA crossover)                    │
│    - Regime trend component (HMM state + drift)                 │
│    - Funding proximity (predictable near settlement)            │
│    Combined: Σ(signal_i × graduated_weight_i) × half_spread    │
│                                                                  │
│  bid = mid_effective - half_spread                              │
│  ask = mid_effective + half_spread                              │
│  (Spread is symmetric around skewed mid)                        │
│                                                                  │
│  Neutral (skew ≈ 0) is a DERIVED state when:                   │
│  position ≈ 0 AND signals ≈ 0 AND regime = QUIET               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ bid, ask (per level)
┌──────────────────────────▼──────────────────────────────────────┐
│                 L4: RISK OVERLAY                                 │
│                                                                  │
│  Graduated (never binary):                                      │
│  - Changepoint detection → spread_mult [1.0, 3.0]              │
│  - Learning trust → size_mult [0.3, 1.0]                       │
│  - Kill switch monitors → HARD stop (separate from graduated)   │
│                                                                  │
│  CRITICAL INVARIANT:                                            │
│  Risk overlay NEVER cancels reduce-only quotes.                 │
│  "Emergency" = widen + reduce size, NOT cancel-all.             │
│  Cancel-all is ONLY for kill switch trigger.                    │
│                                                                  │
│  After overlay:                                                 │
│  bid_final = mid_effective - half_spread × spread_mult          │
│  ask_final = mid_effective + half_spread × spread_mult          │
│  sizes scaled by size_mult                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ final quotes
┌──────────────────────────▼──────────────────────────────────────┐
│                 L5: LADDER GENERATOR + RECONCILER                │
│                                                                  │
│  Pure function: (quotes, zone, liquidity) → Ladder              │
│  No spread computation. No skew computation. No regime logic.   │
│  Just mechanical level generation + order reconciliation.       │
│                                                                  │
│  Position gate in reconciler (defense-in-depth):                │
│  place_bulk_ladder_orders checks config.max_position            │
│  as HARD limit, independent of governor.                        │
│                                                                  │
│  Untracked fill → immediate full reconciliation                 │
│  Cancel-fill race → tombstone tracking with assertion           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer Details

### Measurement Substrate

**The single most important change.** Nothing else works without this.

```rust
/// Stored when order is PLACED (not filled)
struct OrderLifecycle {
    oid: u64,
    side: Side,
    price: f64,
    size: f64,
    mid_at_placement: f64,       // ← THE KEY FIELD
    spread_at_placement_bps: f64, // actual half-spread from our ladder
    placed_at: Instant,
    regime_at_placement: RegimeState,
}

/// Computed when fill ARRIVES
struct FillMeasurement {
    fill_price: f64,
    mid_at_fill: f64,           // latest_mid when fill logged
    spread_captured_bps: f64,    // |fill_price - mid_at_placement| / mid_at_placement × 10000
    // NOT: |fill_price - latest_mid| (which is tautological)
}

/// Computed 5 seconds AFTER fill (deferred)
struct MarkoutMeasurement {
    mid_at_markout: f64,
    as_bps: f64,                // (mid_at_markout - mid_at_fill) × direction × 10000
    realized_edge_bps: f64,      // spread_captured - as_bps - fee_bps
}

/// EdgeSnapshot created ONLY after markout completes
struct EdgeSnapshot {
    spread_captured_bps: f64,    // from FillMeasurement
    as_realized_bps: f64,        // from MarkoutMeasurement (5s markout)
    realized_edge_bps: f64,      // spread_captured - as_realized - fee
    regime_at_placement: RegimeState,
    // NO immediate AS computation. NO tautological depth_from_mid.
}
```

**What this eliminates:**
- The tautological `as_realized ≈ depth_from_mid` (both using `latest_mid`)
- The dual AS paths (immediate + 5s markout feeding different consumers)
- The `predicted_spread_bps == realized_spread_bps` identity
- Edge prediction training on noise

**What this enables:**
- Actual measurement of spread capture vs adverse selection
- RL reward that's informative (edge ≈ 2-5 bps, not ≈ -1.5 bps noise)
- Calibration metrics that mean something
- Regime-conditional performance attribution

### L0: Inventory Governor

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
enum PositionZone {
    Green,    // < 50% of max — full quoting
    Yellow,   // 50-80% — bias toward reducing, wider on increasing side
    Red,      // > 80% — reduce-only both sides
    Kill,     // > 100% — cancel all, kill switch
}

struct InventoryGovernor {
    max_position: f64,  // FROM CONFIG. NEVER from margin. NEVER overridden.
}

impl InventoryGovernor {
    /// FIRST call in every quote cycle. Returns zone + constraints.
    fn assess(&self, position: f64) -> PositionAssessment {
        let ratio = position.abs() / self.max_position;
        let zone = match ratio {
            r if r < 0.5 => PositionZone::Green,
            r if r < 0.8 => PositionZone::Yellow,
            r if r <= 1.0 => PositionZone::Red,
            _ => PositionZone::Kill,
        };

        let max_new_exposure = match zone {
            PositionZone::Green => self.max_position - position.abs(),
            PositionZone::Yellow => (self.max_position - position.abs()) * 0.5,
            PositionZone::Red => 0.0,  // reduce-only
            PositionZone::Kill => 0.0,
        };

        PositionAssessment { zone, max_new_exposure, position_ratio: ratio }
    }
}
```

**What this eliminates:**
- `effective_max_position` derived from margin (53-58 HYPE vs 3.24 config)
- Reduce-only bypass via margin-derived limits
- Scattered position checks in 4+ different files
- The Feb 10 and Feb 12 position limit violations

**Critical design choice:** Margin-derived capacity is used ONLY for notional sizing (how much USD per order). It is NEVER used for position limits. `config.max_position` is the ceiling, full stop.

### L1: Regime State Machine

```rust
#[derive(Debug, Clone, Copy)]
enum MarketRegime {
    Quiet,     // σ < baseline × 0.7, kappa high, OI stable
    Normal,    // baseline conditions
    Volatile,  // σ > baseline × 1.3 OR kappa < baseline × 0.5
    Crisis,    // σ > baseline × 3.0 OR OI drops > 5% in 1min
}

struct RegimeParams {
    kappa: f64,              // fill intensity (from kappa_orchestrator)
    sigma: f64,              // realized volatility (from GARCH/EWMA)
    gamma_base: f64,         // risk aversion base (regime-specific)
    as_expected_bps: f64,    // expected adverse selection (from markout EWMA)
    risk_premium_bps: f64,   // additional buffer (regime-specific)
    skew_sensitivity: f64,   // how aggressively to skew (0.5 in Quiet, 1.5 in Crisis)
    max_position_frac: f64,  // governor can tighten (0.3 in Crisis)
    spread_mult_floor: f64,  // minimum spread multiplier (1.0 in Quiet, 2.0 in Crisis)
}

impl RegimeEngine {
    fn current_params(&self) -> RegimeParams {
        // Uses EMA-smoothed regime probabilities, not per-tick switching
        // Hysteresis: must stay in new regime for 30s before full transition
        // Blends between regimes during transition periods
    }
}
```

**What this eliminates:**
- `kappa_effective` computed but never consumed (Bug 3)
- Separate `adaptive/learned_floor.rs` (floor emerges from `fee + as_expected + risk_premium`)
- Hardcoded AS buffer (`conditional_as_buffer_bps = 2.0`)
- The 7-deep spread multiplier chain (replaced by regime-specific base params)

**What this enables:**
- Kappa variation (8.4x observed) directly drives spread computation
- Regime-specific risk aversion (tight in Quiet, wide in Crisis)
- Governor tightening in Crisis (max_position_frac = 0.3 → effective max drops)
- Per-regime performance attribution

### L2: Spread Engine

```rust
impl SpreadEngine {
    fn compute(&self, params: &RegimeParams, fee_bps: f64) -> f64 {
        let target_floor_bps = fee_bps + params.as_expected_bps + params.risk_premium_bps;

        // γ that makes GLFT(γ, κ, σ) ≥ target_floor
        let gamma = solve_min_gamma(
            target_floor_bps / 10_000.0,
            params.kappa,
            params.sigma,
            self.holding_time,
        );

        // GLFT half-spread (always ≥ target_floor by construction)
        let half_spread = (1.0 / gamma) * (1.0 + gamma / params.kappa).ln() + fee_bps / 10_000.0;

        half_spread * 10_000.0 // return bps
    }
}
```

**Why there is no floor override:**

The current system computes GLFT optimal (3 bps), then clamps to floor (8 bps). This makes GLFT decorative.

In the new design, `solve_min_gamma` finds γ such that GLFT **produces** the floor naturally. If AS is 3 bps and risk premium is 2 bps, and fee is 1.5 bps, then target_floor = 6.5 bps. γ is chosen so GLFT ≥ 6.5 bps **by the math**, not by clamping.

**Why the AS buffer disappears:**

The current `conditional_as_buffer_bps = 2.0` is a magic number that inflates the floor. In the new design, `as_expected_bps` comes from the Measurement Substrate — it's the actual measured AS from 5-second markouts, EMA-smoothed, regime-conditional. If AS is actually 2 bps, `as_expected_bps = 2.0`. No extra buffer needed because:
1. The measurement is correct (not tautological)
2. The risk_premium covers uncertainty
3. γ absorbs remaining risk

### L3: Asymmetric Pricer

```rust
impl AsymmetricPricer {
    fn compute_skew(&self,
        position: f64,
        max_position: f64,
        signals: &CombinedSignals,
        regime: &RegimeParams,
        half_spread_bps: f64,
    ) -> f64 {
        // Inventory skew: ALWAYS active when position ≠ 0
        let inventory_ratio = position / max_position; // [-1, 1]
        let inventory_skew = -inventory_ratio * regime.skew_sensitivity * half_spread_bps;

        // Signal skew: sum of normalized signals × weights × half_spread
        let signal_skew = signals.combined_direction * half_spread_bps * 0.3;
        // combined_direction in [-1, 1], 30% of half_spread max influence

        // Total skew applied to mid
        let total_skew_bps = inventory_skew + signal_skew;

        // Clamp: skew cannot exceed half_spread (prevents crossing)
        total_skew_bps.clamp(-half_spread_bps * 0.8, half_spread_bps * 0.8)
    }
}

/// Each signal provides a normalized direction
struct CombinedSignals {
    /// [-1, 1] where -1 = bearish, +1 = bullish
    combined_direction: f64,
    /// Confidence [0, 1] — scales the signal's contribution
    confidence: f64,
}

impl SignalAggregator {
    fn aggregate(&self) -> CombinedSignals {
        let mut direction = 0.0;
        let mut total_weight = 0.0;

        // Order flow imbalance (from trade tape, NOT hardcoded to 0)
        if let Some(flow) = self.order_flow_imbalance() {
            let w = self.flow_weight * graduated_weight(self.flow_ir);
            direction += flow * w;  // flow in [-1, 1]
            total_weight += w;
        }

        // Momentum (multi-timeframe)
        if let Some(mom) = self.momentum_signal() {
            let w = self.momentum_weight * graduated_weight(self.momentum_ir);
            direction += mom * w;  // mom in [-1, 1]
            total_weight += w;
        }

        // Regime trend (from HMM state)
        if let Some(trend) = self.regime_trend() {
            let w = self.regime_weight * graduated_weight(self.regime_ir);
            direction += trend * w;
            total_weight += w;
        }

        let normalized = if total_weight > 0.0 { direction / total_weight } else { 0.0 };

        CombinedSignals {
            combined_direction: normalized.clamp(-1.0, 1.0),
            confidence: total_weight.min(1.0),
        }
    }
}
```

**What this eliminates:**
- `combined_skew_bps = 0.0` for 99.8% of cycles (Bug 2)
- The `directional` field that affects margin split but not prices
- HL-native skew fallback with hardcoded `imbalance_30s: 0.0`
- InformedFlow tightening (harmful signal, Bug 6)

**Key design choice:** Inventory skew scales with `half_spread`. This means:
- In Quiet regime (tight spread): small skew in absolute terms, but large relative to spread
- In Crisis regime (wide spread): large absolute skew, naturally aggressive about reducing position
- This is the "spread as risk buffer" principle: spread carries the risk information, and skew leverages it

### L4: Risk Overlay

```rust
impl RiskOverlay {
    fn assess(&self) -> RiskModifiers {
        // Changepoint detection → graduated spread multiplier
        let cp_prob = self.changepoint_detector.probability();
        let spread_mult = match cp_prob {
            p if p < 0.5 => 1.0,
            p if p < 0.8 => 1.0 + (p - 0.5) * 2.0, // 1.0 → 1.6
            p if p < 0.95 => 1.6 + (p - 0.8) * 4.0, // 1.6 → 2.2
            _ => 2.5, // max 2.5x, NEVER cancel
        };

        // Learning trust → size multiplier
        let trust = self.learning_trust;
        let size_mult = 0.3 + 0.7 * trust; // [0.3, 1.0]

        RiskModifiers { spread_mult, size_mult }
    }
}

// CRITICAL: Emergency pull semantics
// Old: cancel_all_orders() — kills reduce-only quotes, causes paralysis
// New: widen_all_orders(3.0x) — preserves all quotes, just makes them wider
// Cancel-all is ONLY for kill switch (true emergency, not just changepoint)
```

**What this eliminates:**
- Emergency pull paralysis (Bug 4) — 78/89 min locked out
- 903 emergency pulls that prevented position reduction
- Binary cancel-all behavior (replaced by graduated widening)
- Overlapping authority between controller, circuit breaker, and kill switch

### L5: Ladder Generator + Reconciler

```rust
impl LadderGenerator {
    /// Pure function. No spread computation. No regime logic.
    fn generate(
        &self,
        mid_effective: f64,  // already skewed
        half_spread: f64,    // already regime-adjusted
        risk_mods: &RiskModifiers,
        zone: PositionZone,
        liquidity: f64,
    ) -> Ladder {
        let adjusted_spread = half_spread * risk_mods.spread_mult;
        let adjusted_size = liquidity * risk_mods.size_mult;

        match zone {
            PositionZone::Kill => Ladder::empty(),
            PositionZone::Red => {
                // Only reduce-only quotes
                self.generate_reduce_only(mid_effective, adjusted_spread, adjusted_size)
            }
            _ => {
                self.generate_full(mid_effective, adjusted_spread, adjusted_size, zone)
            }
        }
    }
}

impl Reconciler {
    fn place_orders(&mut self, ladder: &Ladder, position: f64, max_position: f64) {
        // Defense-in-depth: HARD position check independent of governor
        for order in &ladder.orders {
            let new_position = position + order.signed_size();
            if new_position.abs() > max_position {
                warn!("Reconciler blocked order exceeding config.max_position");
                continue; // Skip, don't crash
            }
            self.place(order);
        }
    }
}
```

---

## What Gets Deleted

| Current Module/Concept | Replacement | Why |
|---|---|---|
| `adaptive/learned_floor.rs` | RegimeParams.as_expected_bps + risk_premium | Floor emerges from measured components, not learned as separate parameter |
| Spread multiplier chain (7 multipliers) | RegimeParams + RiskModifiers | Two clear sources of spread adjustment, not seven conflicting ones |
| `effective_max_position` from margin | `config.max_position` always | Margin determines notional sizing, never position limits |
| `conditional_as_buffer_bps = 2.0` | RegimeParams.as_expected_bps (measured) | Buffer was compensating for tautological AS measurement |
| Emergency pull cancel-all | RiskOverlay graduated widening | Cancel-all caused the exact losses it was trying to prevent |
| InformedFlow tightening | Removed entirely | Harmful signal, marginal value -0.23 bps |
| Immediate AS computation (line 714) | Deferred to 5s markout | Immediate computation is tautological |
| `predicted_spread_bps = depth_bps` | `spread_at_placement_bps` from OrderLifecycle | No more identity between predicted and realized |
| Margin split via directional signal | Actual price skew via AsymmetricPricer | Skew should move prices, not just margin allocation |

---

## What Gets Kept

| Module | Status |
|---|---|
| `glft.rs` + `solve_min_gamma` | Keep — core math is correct, just needs correct inputs |
| `kappa_orchestrator.rs` | Keep — robust kappa estimation works well |
| `estimator/volatility/regime.rs` | Keep — HMM detection works, output just needs wiring |
| `pre_fill_classifier.rs` | Keep — correctly uses 5s markout, feeds useful toxicity signal |
| `kill_switch.rs` | Keep — hard limits are correct, just needs config.max as ceiling |
| `quoting/ladder/generator.rs` | Keep — entropy optimizer, geometric spacing work |
| `fills/processor.rs` | Keep — tombstone tracking, dedup work |
| Checkpoint system | Keep — add OrderLifecycle persistence |

---

## The Pipeline (Execution Order)

```
1. Market data arrives (L2 book, trades, AllMids)
   │
2. Estimators update (kappa, sigma, regime probabilities)
   │
3. InventoryGovernor.assess(position) → zone, max_new_exposure
   │  ↳ If KILL → cancel all, trigger kill switch, RETURN
   │
4. RegimeEngine.current_params() → RegimeParams
   │  ↳ Blends HMM probabilities, kappa from orchestrator, sigma from GARCH
   │  ↳ Hysteresis-smoothed (no per-tick switching)
   │
5. SpreadEngine.compute(RegimeParams) → half_spread_bps
   │  ↳ solve_min_gamma ensures GLFT ≥ (fee + AS + risk)
   │  ↳ No floor. No buffer. No cap. Math produces right answer.
   │
6. SignalAggregator.aggregate() → CombinedSignals
   │  ↳ Each signal: direction [-1,1] × graduated_weight
   │  ↳ Flow imbalance from trade tape (not hardcoded 0)
   │
7. AsymmetricPricer.compute_skew(position, signals, regime, half_spread)
   │  → inventory_skew + signal_skew → skewed_mid
   │  ↳ Inventory skew ALWAYS active when position ≠ 0
   │
8. RiskOverlay.assess() → spread_mult, size_mult
   │  ↳ Graduated (never binary cancel-all)
   │  ↳ Reduce-only quotes ALWAYS preserved
   │
9. LadderGenerator.generate(skewed_mid, half_spread × spread_mult, zone)
   │  → Ladder with bid/ask levels
   │
10. Reconciler.place_orders(ladder, position, config.max_position)
    │  ↳ Defense-in-depth: hard position check per order
    │  ↳ OrderLifecycle created: stores mid_at_placement
    │
11. Fill arrives → FillMeasurement (immediate)
    │  ↳ spread_captured = |fill_price - mid_at_placement| (NOT latest_mid)
    │
12. 5 seconds later → MarkoutMeasurement (deferred)
    │  ↳ as_realized = mid_change since fill
    │  ↳ EdgeSnapshot created NOW (not at fill time)
    │  ↳ Fed to: RL reward, calibration, PnL attribution, spread optimizer
```

---

## Migration Strategy

### Phase 1: Measurement Fix (unblocks everything)
1. Add `mid_at_placement` to order tracker
2. Defer EdgeSnapshot to 5s markout
3. Fix `spread_captured_bps` to use placement mid
4. **Validate**: edge distribution shifts from ≈0 to ≈2-5 bps

### Phase 2: Inventory Governor (safety critical)
1. Create `InventoryGovernor` struct
2. Replace all `effective_max_position` with `config.max_position`
3. Add zone computation + reduce-only enforcement
4. Wire as FIRST check in `update_quotes()`
5. Defense-in-depth in reconciler
6. **Validate**: position never exceeds config in paper trading

### Phase 3: Regime Engine (unlocks spread + skew)
1. Create `RegimeEngine` wrapping existing HMM + kappa orchestrator
2. Produce `RegimeParams` consumed by spread + pricer
3. Wire kappa_effective into spread computation
4. **Validate**: spread varies with regime, not fixed to floor

### Phase 4: Spread Engine (eliminates floor)
1. Move spread computation to `SpreadEngine`
2. Use `RegimeParams.as_expected_bps` (from measurement substrate)
3. Let `solve_min_gamma` produce self-consistent γ
4. Delete learned_floor, AS buffer, kappa cap
5. **Validate**: floor binding rate drops from 100% to <5%

### Phase 5: Asymmetric Pricer (eliminates zero skew)
1. Create `AsymmetricPricer`
2. Inventory skew always active
3. Signals produce direction [-1,1] not just spread_bps
4. Delete margin-split-only directional logic
5. **Validate**: skew ≠ 0 whenever position ≠ 0

### Phase 6: Risk Overlay Simplification
1. Replace emergency pull cancel-all with graduated widening
2. Preserve reduce-only quotes in all risk states
3. Consolidate multiplier chain to regime + overlay only
4. **Validate**: quoting uptime > 80% (was 12%)

---

## Success Criteria

| Metric | Current | Target | Why |
|---|---|---|---|
| Floor binding rate | 100% | < 5% | GLFT produces correct spread by construction |
| Skew ≠ 0 when position ≠ 0 | 0.2% | 100% | Inventory skew always active |
| Position / max_position | 3.95x | ≤ 1.0x | Governor + defense-in-depth |
| Quoting uptime | 12% | > 80% | No cancel-all except kill switch |
| Edge prediction R² | ~0 | > 0.1 | Measurement not tautological |
| AS measurement accuracy | Tautological | Within 1 bps of markout | mid_at_placement tracked |
| Fill imbalance in sweeps | 4.8:1 buy/sell | < 2:1 | Skew leans away from flow |
| Regime kappa consumed | No | Yes | RegimeEngine → SpreadEngine |
| Signals producing skew | 0 of 6 | 3+ of 6 | Direction output required |

---

## Appendix: Current vs Redesign Decision Points

| Decision | Current | Redesign |
|---|---|---|
| "What spread?" | GLFT → floor clamp → AS buffer → kappa cap | solve_min_gamma(fee + AS + risk) → done |
| "What skew?" | 0 (directional → margin only) | inventory_skew + signal_skew (always) |
| "Can I quote?" | margin-derived limit → reduce-only guess | Governor zone check → hard ceiling |
| "Am I in danger?" | 7 multipliers × changepoint → cancel-all | Regime params + graduated overlay |
| "Was that fill good?" | AS ≈ depth (tautological) | spread_captured - markout_AS - fee |
| "What regime?" | Detected, ignored | Detected → RegimeParams → everything |
| "Emergency?" | Cancel all quotes for 78 min | Widen 2.5x, keep reduce-only |
