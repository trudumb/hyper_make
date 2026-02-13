# Foundational Redesign: Principled Architecture from Strategic Assumptions

## Context

The Feb 12 HYPE mainnet session lost $5.02 in 89 minutes, exposing 6 systemic bugs that share a common root cause: **the system was built bottom-up as patches on patches, with no strategic principle serving as an architectural foundation**. The current 22-step quote pipeline chains 8 multiplicative spread adjustments, has dead signal paths, tautological measurement, and safety mechanisms that cause harm.

The strategy redesign doc (`docs/market-maker-strategy-redesign.md`) identifies 5 strategic principles. This plan treats each as a **foundational assumption** — redesigning the affected subsystems as they would have been built if that principle was primary from day one.

## Evidence from Live Session (Feb 12 19:37 log)

| Symptom | Root Cause | Current Code Path |
|---------|-----------|-------------------|
| `kappa=18,500` → 2 bps spreads → picked off | Regime kappa (500-4203) computed but never consumed | `signal_integration.rs` computes, `quote_engine.rs` ignores |
| `reduce_only_threshold=0.00` → one-sided at pos=0.23 | PnL tracker returns 0, any position triggers reduce-only | `quote_gate.rs:1835-1841` regime inference broken |
| `smoothed_drift=0.000000` → zero skew | NOT a bug — EWMA averaging on thin DEX genuinely produces ~0 | `controller.rs:273/276` IS updated, input is small |
| `effective_max_position=54` on $100 capital | margin($170) × leverage(10x) / price($31) = 54.8 | `market_params.rs:1165` margin dominates, config ignored |
| `edge_bps ≈ -1.5` always | AS measured with latest_mid at fill time (tautological) | `handlers.rs:804-806` instant AS, markout better but feeds wrong consumers |
| `bid_levels=0, ask_levels=1-3` for 78/89 min | Emergency pull cancels ALL quotes, no reduce-only exemption | `control/mod.rs` emergency_pull has no cooldown enforcement |

### Corrected Understanding (Plan Agent Findings)

Previous audit contained 2 errors corrected during planning:

1. **"HL flow features hardcoded to 0"** — WRONG. `set_hl_flow_features()` IS called at `handlers.rs:617` with real data from `trade_flow_tracker` (EWMA imbalance at 30s/5m) and Hawkes (1s/5s proxy). Values are small because HYPE DEX trades are sparse (~10/min), not because they're hardcoded.

2. **"drift_ewma never updated"** — WRONG. `update_momentum_signals()` IS called at `quote_engine.rs:432` and updates `drift_ewma` at `controller.rs:273` (OU path) and `276-277` (legacy EWMA path). Value is ~0 because thin-DEX momentum genuinely averages near zero.

3. **Root cause of zero inventory skew** (NEW finding): `position_guard.rs:307` computes `q = position / max_position`. With `effective_max_position=54` and position=0.23: `q=0.004 → skew=0.06 bps`. With correct `config.max_position=3.24`: `q=0.071 → skew=1.86 bps`. At position=2.99 (peak in log): wrong denominator gives 0.83 bps, correct gives 13.8 bps. **Bug 5 (effective_max) causes Bug 2 (zero skew) — 16x amplification.**

---

## Architecture: 5 Sequential Phases

Each phase is self-contained, builds on the previous, and can be implemented by a single agent.

```
Phase 1: Measurement    — Fix AS tautology so we can measure truth
Phase 2: Regime         — Regime as state machine driving all params
Phase 3: Spread         — Single regime-parameterized spread formula
Phase 4: Skew           — Asymmetry as default, inventory as primary
Phase 5: Inventory      — Position state as system governor (fixes skew too)
```

---

## Phase 1: Measurement Infrastructure (signals agent)

**Principle**: You cannot optimize what you cannot measure. Fix edge measurement FIRST so all subsequent phases can validate their impact.

### Problem
- `handlers.rs:804-806`: AS uses `self.latest_mid` at fill time → AS ≈ depth → edge = -1.5 bps always
- Pre-fill classifier fed tautological outcomes → never learns
- All model IR ≈ 0 → graduated weights meaningless
- InformedFlow spread adjustment harmful (-0.23 bps)

### Changes

**`src/market_maker/orchestrator/handlers.rs`**
1. Edge measurement at fill time (lines ~784-911):
   - `mid_at_placement` already tracked in `TrackedOrder` ✓
   - Change instant `as_realized` computation: use `mid_at_placement` instead of `self.latest_mid`
   - `as_realized = (mid_at_placement - fill_price) * direction / fill_price` (distance from placement mid, not current mid)
   - This makes instant AS = 0 when fill exactly at predicted depth (correct neutral case)

2. Pre-fill classifier feedback (lines ~300-360):
   - `was_adverse` must use `mid_at_fill` vs `mid_at_markout`, NOT `latest_mid` vs `fill_price`
   - Already stored: `PendingFillOutcome.mid_at_fill` ✓
   - At 5s markout: `was_adverse = (mid_at_markout - mid_at_fill) * direction > threshold`

3. InformedFlow: disable spread tightening
   - In `signal_integration.rs` or `model_gating.rs`: set `min_tighten_mult = 1.0` (one-line fix)

**`src/market_maker/analytics/edge_metrics.rs`**
- EdgeSnapshot: ensure `realized_as_bps` populated from markout (5s), not instant
- `gross_edge_bps = spread_bps - markout_as_bps` (pre-fee, from markout)

### Validation
- `cargo test` — all 2,418+ tests pass
- `cargo clippy -- -D warnings` clean
- Edge distribution should center around 0 ± 3 bps (not locked at -1.5)

### Files Modified
- `src/market_maker/orchestrator/handlers.rs` (~10 lines changed)
- `src/market_maker/strategy/signal_integration.rs` (~1 line)
- `src/market_maker/analytics/edge_metrics.rs` (verify, likely no change)

---

## Phase 2: Regime as State Machine (signals agent)

**Principle**: Every parameter must be regime-dependent. A single regime detection step runs FIRST, and ALL downstream logic reads from the active regime.

### Problem
- `signals.kappa_effective` computed (range 500-4203, 8.4x variation) but NEVER consumed
- `spread_adjustment_bps: 0.0` hardcoded in attribution logger
- 8-multiplier spread chain makes regime adaptation unpredictable
- Regime detection exists (HMM in estimator, BOCPD in quote_engine) but outputs are dead

### Design

Create a `RegimeState` that is the SINGLE source of truth for all regime-dependent parameters:

```rust
// In strategy/regime_state.rs (NEW file, ~80 lines)

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketRegime {
    Calm,      // Low vol, mean-reverting
    Normal,    // Typical conditions
    Volatile,  // Elevated vol, trending possible
    Extreme,   // Cascade/liquidation regime
}

#[derive(Debug, Clone)]
pub struct RegimeParams {
    pub kappa: f64,                    // Regime-conditioned kappa
    pub spread_floor_bps: f64,        // Min spread for this regime
    pub skew_gain: f64,               // How aggressively to skew (0.5-2.0)
    pub max_position_fraction: f64,   // Of config.max_position (0.3-1.0)
    pub emergency_cp_threshold: f64,  // Changepoint prob for emergency pull
    pub reduce_only_fraction: f64,    // Position fraction triggering reduce-only
    pub size_multiplier: f64,         // Quote size scaling (0.3-1.0)
}

pub struct RegimeState {
    pub regime: MarketRegime,
    pub params: RegimeParams,
    pub confidence: f64,               // HMM posterior probability
    pub transition_cooldown_cycles: u32, // Hysteresis
}
```

### Changes

**`src/market_maker/strategy/regime_state.rs`** (NEW)
- `RegimeState::update(vol_regime, hmm_probs, bocpd_cp, kappa_effective)` → updates regime + params
- Hysteresis: require 5 consecutive cycles in new regime before transition
- Default params per regime from existing constants (kappa priors: 3000/2000/1000/500)

**`src/market_maker/orchestrator/quote_engine.rs`**
- Add `regime_state: RegimeState` field to MarketMaker (or QuoteEngine state)
- Call `regime_state.update()` EARLY in update_quotes(), after belief/estimator updates (~line 217)
- Replace `threshold_kappa_mult` computation (lines ~1000) with `regime_state.params`
- Replace hardcoded `spread_adjustment_bps: 0.0` with regime-derived value
- Wire `regime_state.params.kappa` into `market_params.kappa` (so GLFT uses regime kappa)

**`src/market_maker/strategy/signal_integration.rs`**
- `kappa_effective` already computed in `get_signals()` → now consumed via RegimeState
- Remove dead `regime_kappa_spread_mult` diagnostic-only computation

**`src/market_maker/strategy/ladder_strat.rs`**
- In `generate_ladder()` kappa selection (lines 656-724):
  - Priority: regime_kappa > kappa_robust > adaptive_kappa > legacy
  - This makes regime the PRIMARY driver of spread width

### What Gets DELETED
- `threshold_kappa_mult` computation in quote_engine.rs spread chain
- `spread_adjustment_bps: 0.0` hardcoded in live.rs
- Redundant regime_kappa_mult `> 1.01` gate (regime now always applies)

### Validation
- Regime transitions logged with kappa/spread changes
- In Calm: kappa ~3000, spreads ~4-5 bps
- In Extreme: kappa ~500, spreads ~15-20 bps
- Test: verify regime params produce expected GLFT spreads

### Files Modified
- `src/market_maker/strategy/regime_state.rs` (NEW, ~120 lines)
- `src/market_maker/strategy/mod.rs` (add module)
- `src/market_maker/orchestrator/quote_engine.rs` (~30 lines changed)
- `src/market_maker/strategy/signal_integration.rs` (~5 lines)
- `src/market_maker/strategy/ladder_strat.rs` (~10 lines)
- `src/market_maker/analytics/live.rs` (~2 lines, remove hardcoded 0.0)

---

## Phase 3: Spread as Primary Risk Buffer (strategy agent)

**Principle**: Spread is computed from ONE regime-parameterized formula. The 8-multiplier chain is replaced by a single composition with clear, bounded terms.

### Problem
- Current: `base × threshold_kappa × model_gating × staleness × toxicity × defensive × risk_overlay`, capped
- Each multiplier added as a patch, interaction effects unpredictable
- Three 1.5x adjustments compound to 3.375x (multiplicative) instead of 2.5x (intended additive)
- Spread floor overrode GLFT 100% before solve_min_gamma fix

### Design

Replace multiplicative chain with a single spread computation:

```rust
// Spread = GLFT_half_spread(regime_kappa, sigma, gamma) + risk_premium

fn compute_spread_bps(
    regime: &RegimeParams,
    sigma: f64,
    gamma: f64,       // from solve_min_gamma (Phase 2 of Feb 11 redesign)
    fee_bps: f64,
    risk_premium: &RiskPremium,
) -> f64 {
    // Core GLFT spread (already correct formula)
    let kappa = regime.kappa.max(1.0); // kappa > 0 invariant
    let half_spread = (1.0 / gamma) * (1.0 + gamma / kappa).ln();
    let half_spread_bps = half_spread * 10_000.0;

    // Risk premium: ADDITIVE, not multiplicative
    // Each component bounded independently, total capped
    let total_premium_bps = risk_premium.total_bps(); // sum of bounded terms

    // Final: GLFT + premium + fees, floored by regime minimum
    (half_spread_bps + total_premium_bps + fee_bps)
        .max(regime.spread_floor_bps)
}

struct RiskPremium {
    staleness_bps: f64,      // 0-5 bps, from data freshness
    toxicity_bps: f64,       // 0-10 bps, from VPIN/pre-fill classifier
    changepoint_bps: f64,    // 0-15 bps, from BOCPD probability
}
```

### Changes

**`src/market_maker/orchestrator/quote_engine.rs`**
- Lines ~993-1110 (spread multiplier composition): REPLACE entire block
- New function: `compute_risk_premium() -> RiskPremium` with 3 bounded additive terms
- Remove: `threshold_kappa_mult`, `model_gating_mult`, `staleness_mult`, `toxicity_mult`, `defensive_mult`
- Keep: `risk_overlay.spread_multiplier` (from controller, but make it additive too)
- The GLFT formula in glft.rs stays unchanged — only how we compose the final spread changes

**`src/market_maker/strategy/glft.rs`**
- No changes needed — `half_spread()` formula is correct
- `solve_min_gamma()` from Feb 11 redesign stays

**`src/market_maker/strategy/ladder_strat.rs`**
- `generate_ladder()` receives regime-parameterized kappa (from Phase 2)
- Remove informed flow alpha reduction (lines 701-724) — replaced by regime params
- Remove separate regime blending (lines 683-698) — regime kappa is already blended

### What Gets DELETED
- ~80 lines of multiplicative spread composition in quote_engine.rs
- `spread_widening_mult` field on MarketParams (replaced by additive premium)
- `defensive_mult` computation
- `toxicity_mult` computation (replaced by bounded toxicity_bps)
- InformedFlow kappa reduction in ladder_strat.rs

### Validation
- Spread in Calm regime: 4-6 bps (kappa ~3000, low premium)
- Spread in Extreme regime: 15-25 bps (kappa ~500, high premium)
- No multiplicative explosion: max premium capped at ~30 bps total
- Test: GLFT formula produces same half-spread for same kappa/sigma/gamma

### Files Modified
- `src/market_maker/orchestrator/quote_engine.rs` (~80 lines replaced)
- `src/market_maker/strategy/ladder_strat.rs` (~30 lines deleted/simplified)
- `src/market_maker/strategy/market_params.rs` (~5 lines, simplify fields)

---

## Phase 4: Asymmetry as Core Pricing Principle (strategy agent)

**Principle**: Symmetric quoting is the special case. The system ALWAYS computes directional skew from all available signals. Skew = 0 only when signals genuinely cancel.

### Problem (Corrected Diagnosis)

The zero-skew problem has THREE interacting causes, not the single "hardcoded to 0" previously believed:

1. **Inventory skew denominator wrong** (root cause): `position_guard.rs:307` computes `q = position / max_position`, but `max_position = effective_max_position = 54` instead of `config.max_position = 3.24`. This makes inventory skew **16x too small**:
   - At position=0.23: `0.23/54 → 0.06 bps` vs `0.23/3.24 → 1.86 bps`
   - At position=2.99 (peak): `2.99/54 → 0.83 bps` vs `2.99/3.24 → 13.8 bps`

2. **Directional flow signals near-zero on thin DEX**: `signal_integration.rs:896-908` HL-native fallback fires correctly, but `imbalance_5s` (Hawkes proxy) and `imbalance_30s` (EWMA) produce small values from ~10 trades/min on HYPE DEX → `combined_skew_bps ≈ 0.03 bps` (negligible)

3. **Two separate skew paths never combined properly**:
   - Path A (`signal_integration.rs:890`): `combined_skew_bps` — flow/directional signals only
   - Path B (`quote_engine.rs:879`): `position_guard_skew_bps` — inventory only
   - They're added at line 883/924, but Path A ≈ 0 and Path B ≈ 0 (wrong denominator)

### Design

Replace the 4-tier fallback cascade with additive skew from ALL sources simultaneously:

```rust
// All skew components computed SIMULTANEOUSLY, not as fallbacks
struct SkewComponents {
    inventory_bps: f64,     // Position-proportional, ALWAYS non-zero if position != 0
    flow_bps: f64,          // From HL trade flow (5s + 30s imbalance)
    momentum_bps: f64,      // From smoothed drift (already computed)
    cross_venue_bps: f64,   // From Binance lead-lag (when available)
}

impl SkewComponents {
    fn total(&self, regime: &RegimeParams) -> f64 {
        let raw = self.inventory_bps
            + self.flow_bps
            + self.momentum_bps
            + self.cross_venue_bps;
        // Scale by regime gain, cap at half-spread
        (raw * regime.skew_gain).clamp(-MAX_SKEW_BPS, MAX_SKEW_BPS)
    }
}
```

### Changes

**`src/market_maker/strategy/signal_integration.rs`**
- Replace 4-tier fallback cascade (lines 843-923) with additive composition:
  - Flow skew: `latest_hl_flow.imbalance_5s * 0.6 + imbalance_30s * 0.4` (always computed, not gated)
  - Momentum skew: from `smoothed_drift` (already non-zero when trends exist)
  - Cross-venue skew: when available (unchanged logic, additive not exclusive)
  - Buy pressure: unchanged, additive
- Remove the `if !lead_lag_actionable && !cross_venue_valid` gate — flow component always contributes
- Remove `FLOW_URGENCY_THRESHOLD` separate logic (merged into flow component with continuous scaling)

**`src/market_maker/orchestrator/quote_engine.rs`**
- Lines 881-924: Simplify the lead_lag_actionable branching
  - Inventory skew (from position_guard) + directional skew (from signal_integration) are ALWAYS added
  - No more conditional paths based on whether Binance is available
  - `market_params.lead_lag_signal_bps = combined_skew_bps + position_guard_skew_bps` (unconditional)

**`src/market_maker/risk/position_guard.rs`**
- No changes needed here — the fix is in Phase 5 (`effective_max_position` capped by config.max_position)
- Once Phase 5 fixes the denominator, inventory skew automatically becomes 16x larger

### What Gets DELETED
- The entire 4-tier fallback cascade in signal_integration.rs (lines 843-923)
- `HL_IMBALANCE_SHORT_WEIGHT`, `HL_IMBALANCE_MED_WEIGHT` constants (merged into additive weights)
- `FLOW_URGENCY_THRESHOLD` separate block (merged into continuous scaling)
- The `if lead_lag_actionable { ... } else { ... }` branching in quote_engine.rs (replaced by unconditional addition)

### Validation
- At position=0: skew from flow + momentum only
- At 50% position (after Phase 5 fix): inventory skew ~7.5 bps, dominates
- At 80% position: inventory skew ~12 bps, strongly reduces accumulation
- `combined_skew_bps != 0` for >80% of cycles (vs 0.2% today)
- No Binance: flow + inventory still produce meaningful skew (vs zero today)

### Files Modified
- `src/market_maker/strategy/signal_integration.rs` (~60 lines replaced)
- `src/market_maker/orchestrator/quote_engine.rs` (~30 lines simplified)

---

## Phase 5: Inventory as System Governor (risk agent, plan mode)

**Principle**: Position state drives ALL quoting decisions. `config.max_position` is THE hard ceiling. Safety mechanisms NEVER prevent position reduction.

### Problem
- `effective_max_position = 54` on $100 capital (margin-derived >> config) — **cascading effect: makes inventory skew 16x too small (Phase 4)**
- `reduce_only_threshold = 0.00` (any position triggers one-sided)
- Emergency pull cancels ALL quotes including reducing ones
- Kill switch position runaway uses margin-based limit, not config.max_position
- 78/89 min locked out by risk overlay → position stuck → loss
- `max_position_user_specified` gate at `quote_engine.rs:390` only activates if config flag set

### Design

Single `InventoryGovernor` enforcing position constraints at every decision point:

```rust
struct InventoryGovernor {
    max_position: f64,           // ALWAYS = config.max_position (THE ceiling)
    position: f64,               // Current position
    regime: MarketRegime,        // From Phase 2
}

impl InventoryGovernor {
    /// Position utilization: 0.0 = flat, 1.0 = at limit
    fn utilization(&self) -> f64 {
        (self.position.abs() / self.max_position).min(1.0)
    }

    /// Whether reducing quotes are needed (utilization > regime threshold)
    fn needs_reduction(&self) -> bool {
        self.utilization() > self.regime_params().reduce_only_fraction
    }

    /// CRITICAL: reducing quotes ALWAYS allowed, even during emergency
    fn filter_quotes(&self, bids: Vec<Quote>, asks: Vec<Quote>) -> (Vec<Quote>, Vec<Quote>) {
        let is_long = self.position > 0.0;
        if self.needs_reduction() {
            if is_long {
                (vec![], asks)  // Only sell to reduce long
            } else {
                (bids, vec![])  // Only buy to reduce short
            }
        } else {
            (bids, asks)
        }
    }
}
```

### Changes

**`src/market_maker/strategy/market_params.rs`**
- `effective_max_position()` (~line 390 in quote_engine.rs): ALWAYS cap by config.max_position
  - Change: `let pre_effective_max_position = pre_effective_max_position.min(self.config.max_position);`
  - Remove the `max_position_user_specified` gate — config is ALWAYS the ceiling
  - This fixes Bug 5 (position limit bypass) AND Bug 2 (zero skew, via position_guard denominator)

**`src/market_maker/control/mod.rs`** (risk overlay)
- `risk_assessment()`: when `emergency_pull = true`:
  - NEVER cancel ALL quotes
  - Instead: cancel INCREASING-side only, KEEP REDUCING-side
  - New field: `emergency_reduce_only: bool` (replaces binary `emergency_pull`)
- Enforce cooldown: check `last_emergency_pull_cycle` before triggering again
  - If within cooldown (50 cycles), skip the pull

**`src/market_maker/control/quote_gate.rs`**
- `reduce_only_threshold`: derive from regime params (Phase 2), NOT from PnL tracker
  - Calm: 0.7, Normal: 0.5, Volatile: 0.3, Extreme: 0.2
  - This fixes the `threshold = 0.00` bug

**`src/market_maker/orchestrator/quote_engine.rs`**
- After all spread/skew computation, apply inventory governor filter
- Emergency pull path: only cancel increasing-side orders
- Remove separate `proactive_position_management` block (lines ~1179-1257) — merged into governor

**`src/market_maker/orchestrator/reconcile.rs`**
- `place_bulk_ladder_orders()`: hard check `position.abs() < config.max_position` before ANY order placement
- Reduce-only enforcement: if `position.abs() > reduce_only_threshold * max_position`, reject increasing orders

**`src/market_maker/risk/kill_switch.rs`**
- `check_position_runaway()`: use `config.max_position * 2.0` as hard limit (not margin-derived)
- This means runaway triggers at 2x user's configured max, not 2x margin (which could be 100x config)

### Cascading Benefits (from fixing effective_max_position)
- **Position guard inventory skew**: denominator changes from 54 → 3.24, making skew 16x larger
- **Reduce-only triggering**: utilization measured correctly (0.23/3.24 = 7% vs 0.23/54 = 0.4%)
- **Kill switch**: runaway at 6.48 (2×3.24) instead of 109 (2×54.4)
- **Quote gate**: `urgency` and `reduce_only_threshold` measured against correct denominator

### What Gets DELETED
- `proactive_position_management` block in quote_engine.rs (~80 lines)
- `max_position_user_specified` gate in quote_engine.rs (~line 390)
- Emergency pull binary (replaced by emergency_reduce_only)

### Validation
- At position=0.23 (log scenario): bids+asks both active (not one-sided)
- At position=2.5 (75% of 3.24 max): increasing side reduced, not eliminated
- At position=3.0 (93% of max): reduce-only mode, sells only (if long)
- Emergency pull with position=3.0: sells still posted (reducing)
- Kill switch runaway: triggers at 6.48 (2×3.24), not 109 (2×54.4)
- Inventory skew at position=2.0: ~9.3 bps (vs 0.56 bps with old denominator)
- `cargo test` — all tests pass

### Files Modified
- `src/market_maker/orchestrator/quote_engine.rs` (~line 390, remove gate + ~40 lines remove proactive block)
- `src/market_maker/control/mod.rs` (~20 lines)
- `src/market_maker/control/quote_gate.rs` (~15 lines)
- `src/market_maker/orchestrator/reconcile.rs` (~10 lines)
- `src/market_maker/risk/kill_switch.rs` (~5 lines)

---

## Implementation: Sequential Team

Each phase runs as a separate agent, sequentially, building on the previous phase's output.

| Phase | Agent Type | Approx Scope | Dependencies |
|-------|-----------|-------------|--------------|
| 1. Measurement | signals | ~15 lines changed, 3 files | None |
| 2. Regime | signals | ~120 lines new + 40 changed, 6 files | Phase 1 |
| 3. Spread | strategy | ~80 lines replaced, 3 files | Phase 2 |
| 4. Skew | strategy | ~90 lines replaced, 2 files | Phase 2, 3 |
| 5. Inventory | risk (plan mode) | ~100 lines changed, 5 files | Phase 2 |

After each phase: `cargo test && cargo clippy -- -D warnings`

After all phases: full integration test — verify the log scenario (position=0.23) no longer produces one-sided quoting with zero skew.

## Verification Checklist

- [ ] Edge measurement no longer tautological (AS ≠ depth always)
- [ ] Regime transitions produce visible kappa/spread changes in logs
- [ ] Spread in Calm: 4-6 bps, Extreme: 15-25 bps (from GLFT, not multiplier chain)
- [ ] combined_skew_bps != 0 when position or flow is non-zero
- [ ] config.max_position = effective_max_position ceiling ALWAYS
- [ ] reduce_only_threshold > 0 (regime-derived, not 0.00)
- [ ] Emergency pull preserves reducing quotes
- [ ] Kill switch runaway uses config-based limit
- [ ] Inventory skew at 50% utilization: ~7.5 bps (not 0.06 bps)
- [ ] All 2,418+ tests pass, clippy clean
