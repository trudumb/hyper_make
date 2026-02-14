# 10-Phase Architecture Fix — Exact Code Locations & Snippets

## Overview
This document captures the EXACT current code for all 10 phases of the principled architecture redesign, with line numbers, struct fields, and imports needed for modifications.

---

## PHASE 1: Hardcoded tick_size_bps Constants

### File 1: `src/market_maker/orchestrator/quote_engine.rs` (line 607)
**Current code:**
```rust
tick_size_bps: 10.0, // TODO: Get from asset metadata
```

**Status:** Hardcoded to 10.0 bps — needs dynamic lookup from AssetRuntimeConfig.price_multiplier

**What to change:**
- Replace hardcoded 10.0 with: `runtime_config.price_multiplier` (which is 100_000.0 for perps = 10 bps at $1.00)
- Compute as: `1.0 / runtime_config.price_multiplier * 10_000.0`

**Related:**
- **Line 2817**: `MOMENTUM_ENTRY_BPS: f64 = 10.0` — also hardcoded, should scale with tick_size

---

## PHASE 2: tick_size_bps in MarketParams Default

### File 2: `src/market_maker/strategy/market_params.rs` (line 1067)
**Current code:**
```rust
tick_size_bps: 10.0,          // Default 10 bps tick
```

**Status:** Default is correct but static. Should be set dynamically in orchestrator from AssetRuntimeConfig.

**Struct definition (lines 505-508):**
```rust
/// Asset tick size in basis points.
/// Spread floor must be >= tick_size_bps (can't quote finer than tick).
/// Set from asset metadata (e.g., BTC tick = 0.1 → 10 bps at $100k).
pub tick_size_bps: f64,
```

**Where it's used:**
- **Line 1567** in `effective_spread_floor()`: `let tick_floor = self.tick_size_bps / 10_000.0;`
- **Line 921** in ladder_strat.rs: `let tick_floor_bps = market_params.tick_size_bps;`

---

## PHASE 3: Emergency Pull Constants

### File 3: `src/market_maker/orchestrator/quote_engine.rs` (line 2817)
**Current code:**
```rust
const MOMENTUM_ENTRY_BPS: f64 = 10.0;

...

if !is_opposed || inventory_frac < INVENTORY_ENTRY_THRESHOLD || momentum_abs_bps < MOMENTUM_ENTRY_BPS {
```

**Status:** Hardcoded momentum threshold. Should scale with tick_size_bps.

**Need to make dynamic:**
1. Pull MOMENTUM_ENTRY_BPS out of function
2. Make it: `tick_size_bps.max(10.0)` — at least 10 bps but scales with precision

---

## PHASE 4: Cost-Basis-Aware Ladder Clamping

### File 4: `src/market_maker/strategy/ladder_strat.rs` (lines 921-926)
**Current code:**
```rust
let fee_bps = self.risk_config.maker_fee_rate * 10_000.0;
let tick_floor_bps = market_params.tick_size_bps;
let latency_floor_bps = market_params.latency_spread_floor * 10_000.0;
let effective_floor_bps = fee_bps
    .max(tick_floor_bps)
    .max(latency_floor_bps)
    .max(self.risk_config.min_spread_floor * 10_000.0);
```

**Status:** Correct structure but missing cost-basis clamping (unrealized_pnl_bps integration).

**What's needed around line 1850-1949 (cost-basis zone):**
- Add clamp to prevent underwater positions from widening spreads
- Formula: `if unrealized_pnl_bps < -50.0 { cap spread at 2× fee } else { normal }`

---

## PHASE 5: AS Measurement Tautology Fix

### File 5a: `src/market_maker/learning/mod.rs` (lines 290-305)
**Current measurement code:**
```rust
pub fn measure_outcome(&self, pred: &PendingPrediction) -> TradingOutcome {
    // Calculate realized adverse selection
    // AS = price_move_against_fill / fill_price
    let fill_price = pred.fill.price;
    let current_mid = self.last_mid;

    let price_move = if pred.fill.is_buy {
        // For buys, AS is positive if price went down (we bought high)
        (fill_price - current_mid) / fill_price * 10000.0
    } else {
        // For sells, AS is positive if price went up (we sold low)
        (current_mid - fill_price) / fill_price * 10000.0
    };

    let realized_as_bps = price_move.max(0.0);
```

**Status:** TAUTOLOGICAL — compares fill price to CURRENT mid, not mid AT PLACEMENT.

**Root cause:** Both `quoted_spread_bps` and `realized_as_bps` use `self.latest_mid` as reference point, making them perfectly correlated.

### File 5b: `src/market_maker/orchestrator/handlers.rs` (line 838-862)
**Fixed code ALREADY IN PLACE:**
```rust
let mid_at_placement = self.orders.get_order(fill.oid)
    .and_then(|tracked| {
        if tracked.mid_at_placement > 0.0 { Some(tracked.mid_at_placement) } else { None }
    })
    .unwrap_or(self.latest_mid); // Fallback for untracked orders

// Quoted spread: distance from placement mid to fill price (in bps)
let quoted_spread_bps = if mid_at_placement > 0.0 {
    ((fill_price - mid_at_placement).abs() / mid_at_placement) * 10_000.0
} else {
    0.0
};

// Compute instant AS using mid at order placement time.
// Previously used self.latest_mid (fill-time mid), which made AS ≈ depth
// (tautological since both reference the same mid). Using mid_at_placement
// measures how much the fill price differs from where we THOUGHT mid was.
let direction = if is_buy { 1.0 } else { -1.0 };
let as_realized = (mid_at_placement - fill_price) * direction / fill_price;

// Compute depth from placement mid (used by bandit reward + competitor model)
let depth_from_mid = if mid_at_placement > 0.0 {
    (fill_price - mid_at_placement).abs() / mid_at_placement
} else {
    (fill_price - self.latest_mid).abs() / self.latest_mid
};
```

**PendingFillOutcome struct (lines 57-70):**
```rust
#[derive(Debug, Clone)]
pub struct PendingFillOutcome {
    /// Fill timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
    /// Fill price
    pub fill_price: f64,
    /// Whether this was a buy fill (our bid was filled)
    pub is_buy: bool,
    /// Mid price at the time of fill
    pub mid_at_fill: f64,
    /// Mid price when the order was originally placed (from TrackedOrder)
    pub mid_at_placement: f64,
    /// Quoted spread in bps: |fill_price - mid_at_placement| / mid_at_placement * 10000
    pub quoted_spread_bps: f64,
}
```

**Status:** FIXED in handlers.rs. Now need to verify all learning modules consume `mid_at_placement` instead of recomputing with `self.latest_mid`.

---

## PHASE 6: HL-Native Skew Fallback

### File 6: `src/market_maker/strategy/signal_integration.rs` (lines 989-1001)
**Current code:**
```rust
if !signals.lead_lag_actionable
    && !signals.cross_venue_valid
    && signals.combined_skew_bps.abs() < 0.01
{
    // Blend short-term (5s) and medium-term (30s) imbalance for stability
    const HL_IMBALANCE_SHORT_WEIGHT: f64 = 0.6;
    const HL_IMBALANCE_MED_WEIGHT: f64 = 0.4;
    const HL_NATIVE_SKEW_FRACTION: f64 = 0.3;
    let flow_dir = self.latest_hl_flow.imbalance_5s * HL_IMBALANCE_SHORT_WEIGHT
        + self.latest_hl_flow.imbalance_30s * HL_IMBALANCE_MED_WEIGHT;
    let fallback_cap = self.config.max_lead_lag_skew_bps * HL_NATIVE_SKEW_FRACTION;
    signals.combined_skew_bps =
        (flow_dir * fallback_cap).clamp(-fallback_cap, fallback_cap);
}
```

**Status:** ALREADY IMPLEMENTED. Zero-skew fallback working.

**Issue found in Feb 12 audit:** Fallback didn't activate due to hardcoded flow features being zero. Now fixed by reading HL flow properly.

---

## PHASE 7: QUOTE_LATCH_THRESHOLD_BPS

### File 7: `src/market_maker/orchestrator/reconcile.rs` (line 26)
**Current code:**
```rust
const QUOTE_LATCH_THRESHOLD_BPS: f64 = 2.5;
```

**Status:** Static 2.5 bps. Should scale with tick_size_bps or be configurable.

**Usage (line 603):**
```rust
if price_diff_bps <= QUOTE_LATCH_THRESHOLD_BPS {
```

**What to change:**
- Make it: `tick_size_bps * 0.25` (latch within 1/4 of tick size to reduce churn)

---

## PHASE 8: Warmup Progress & Calculation

### File 8: `src/market_maker/adaptive/calculator.rs` (lines 403-421)
**Current code:**
```rust
pub fn warmup_progress(&self) -> f64 {
    // Adapt thresholds based on observed market activity
    let fill_rate = self.observed_fill_rate();
    let (floor_target, kappa_target): (usize, usize) = if fill_rate < 0.005 {
        (10, 5) // Low activity: relax thresholds
    } else {
        (20, 10) // Normal activity: standard thresholds
    };

    // Components contribute to overall progress
    let floor_progress =
        (self.floor.observation_count().min(floor_target) as f64) / floor_target as f64;
    let kappa_progress =
        (self.kappa.own_fill_count().min(kappa_target) as f64) / kappa_target as f64;
    // Gamma uses standardizers which need ~20 observations
    let gamma_progress = if self.gamma.is_warmed_up() { 1.0 } else { 0.5 };

    // Weighted average (floor and kappa most important)
    floor_progress * 0.4 + kappa_progress * 0.4 + gamma_progress * 0.2
}
```

**Status:** CORRECT formula. Formula is: `0.4 × floor_progress + 0.4 × kappa_progress + 0.2 × gamma_progress`.

**Uncertainty factor (lines 424-435):**
```rust
pub fn warmup_uncertainty_factor(&self) -> f64 {
    let progress = self.warmup_progress();
    // Start with 10% wider spreads, decay to 0% as we warm up
    // Reduced from 20% since priors are now well-calibrated
    1.0 + (1.0 - progress) * 0.1
}
```

---

## PHASE 9: Regime Gamma Multiplier

### File 9: `src/market_maker/strategy/regime_state.rs` (lines 24-30)
**Current code:**
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ControllerObjective {
    /// Tight spread, high skew gain, maximize spread capture.
    #[default]
    MeanRevert,
    /// Wide spread, minimize inventory, defensive posture.
    TrendingToxic,
}
```

**Status:** Controller objective enumeration. Maps to gamma multiplier values.

**Implied gamma multipliers (from market_params.rs lines 765-767):**
```rust
/// Regime gamma multiplier — routes regime risk through gamma instead of floor.
/// Calm=1.0, Normal=1.2, Volatile=2.0, Extreme=3.0.
pub regime_gamma_multiplier: f64,
```

**Expected discrete values:**
- Calm regime (Low volatility): gamma_mult = 1.0
- Normal regime: gamma_mult = 1.2
- Volatile regime (High volatility): gamma_mult = 2.0
- Extreme regime: gamma_mult = 3.0

**Where set in ladder_strat.rs (around line 637-660):**
- Ladder priority: `kappa_robust > adaptive_kappa > legacy kappa`
- Gamma blending: `kappa * regime_blend_weight (60%)`
- Dynamic cap: `2/kappa * 10_000` replaces hardcoded 3.0 bps

---

## PHASE 10: Kill Switch min_peak_for_drawdown

### File 10a: `src/market_maker/risk/kill_switch.rs` (line 71)
**Current code:**
```rust
/// Minimum peak PnL (USD) before drawdown check activates.
///
/// Drawdown from peak is only statistically meaningful when the peak
/// represents a significant sample of fills, not spread noise.
/// Below this threshold, `check_daily_loss()` provides the safety net.
///
/// Default: $1.00 (roughly 40 fills at 5 bps capture on $50 notional).
/// Production: `max(1.0, max_position_value * 0.02)`.
pub min_peak_for_drawdown: f64,
```

**Default value (line 99):**
```rust
min_peak_for_drawdown: 1.0,
```

**Production value (lines 137-139):**
```rust
// With max_position_usd=$1000: min_peak=$20 (~800 fills to activate).
// check_daily_loss ($50) still protects against catastrophic loss.
min_peak_for_drawdown: 1.0_f64.max(max_position_usd * 0.02),
```

**Status:** CORRECT logic. Scales with max_position_value as 2%.

---

## PHASE 11: Checkpoint serde(default) Fields

### File 11: `src/market_maker/checkpoint/types.rs`

**CheckpointBundle struct (lines 59-108):**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointBundle {
    pub metadata: CheckpointMetadata,
    pub learned_params: LearnedParameters,
    pub pre_fill: PreFillCheckpoint,
    pub enhanced: EnhancedCheckpoint,
    pub vol_filter: VolFilterCheckpoint,
    pub regime_hmm: RegimeHMMCheckpoint,
    pub informed_flow: InformedFlowCheckpoint,
    pub fill_rate: FillRateCheckpoint,
    pub kappa_own: KappaCheckpoint,
    pub kappa_bid: KappaCheckpoint,
    pub kappa_ask: KappaCheckpoint,
    pub momentum: MomentumCheckpoint,
    /// Kelly win/loss tracker state for position sizing persistence
    #[serde(default)]
    pub kelly_tracker: KellyTrackerCheckpoint,
    /// Model ensemble weights for prediction persistence
    #[serde(default)]
    pub ensemble_weights: EnsembleWeightsCheckpoint,
    /// Contextual bandit spread optimizer state (replaces RL MDP)
    #[serde(default)]
    pub spread_bandit: SpreadBanditCheckpoint,
    /// Baseline tracker EWMA for counterfactual reward centering
    #[serde(default)]
    pub baseline_tracker: BaselineTrackerCheckpoint,
    /// Quote outcome tracker fill rate bins
    #[serde(default)]
    pub quote_outcomes: QuoteOutcomeCheckpoint,
    /// Kill switch state for persistence across restarts
    #[serde(default)]
    pub kill_switch: KillSwitchCheckpoint,
    /// Calibration readiness assessment, stamped at save time
    #[serde(default)]
    pub readiness: PriorReadiness,
    /// Calibration coordinator state for L2-derived kappa blending
    #[serde(default)]
    pub calibration_coordinator: CalibrationCoordinatorCheckpoint,
}
```

**Status:** Core fields (metadata, learned_params, pre_fill, etc.) DON'T have serde(default). Newer fields DO (kelly_tracker, ensemble_weights, spread_bandit, baseline_tracker, quote_outcomes, kill_switch, readiness, calibration_coordinator).

**What to check:**
- Ensure ALL newly-added checkpoint fields have `#[serde(default)]`
- When adding new fields to CheckpointBundle: MUST add `#[serde(default)]` for backward compatibility
- PreFillCheckpoint has selective defaults (lines 142-179): `imbalance_ewma_mean`, `imbalance_ewma_var`, etc.

---

## SUMMARY TABLE: Exact Code Locations

| Phase | File | Lines | Field/Const | Current Value | Status |
|-------|------|-------|-------------|---------------|--------|
| 1 | quote_engine.rs | 607 | tick_size_bps | 10.0 (hardcoded) | NEEDS: Dynamic from price_multiplier |
| 1 | quote_engine.rs | 2817 | MOMENTUM_ENTRY_BPS | 10.0 | NEEDS: Scale with tick_size |
| 2 | market_params.rs | 1067 | tick_size_bps default | 10.0 | OK (set dynamically in orchestrator) |
| 3 | quote_engine.rs | 2817-2821 | Emergency pull logic | Hardcoded BPS | NEEDS: Scale with tick_size |
| 4 | ladder_strat.rs | 921-926 | Floor calculation | No cost-basis clamp | NEEDS: Add unrealized_pnl check |
| 5a | learning/mod.rs | 290-305 | AS measurement | Uses current_mid (TAUTOLOGICAL) | NEEDS: Use mid_at_placement |
| 5b | handlers.rs | 838-862 | mid_at_placement lookup | Uses TrackedOrder.mid | IMPLEMENTED |
| 6 | signal_integration.rs | 989-1001 | HL fallback skew | Uses flow imbalance | IMPLEMENTED |
| 7 | reconcile.rs | 26 | QUOTE_LATCH_THRESHOLD_BPS | 2.5 (static) | NEEDS: Scale with tick_size |
| 8 | adaptive/calculator.rs | 403-421 | warmup_progress() | 0.4+0.4+0.2 formula | CORRECT |
| 9 | regime_state.rs | 24-30 | ControllerObjective enum | MeanRevert / TrendingToxic | CORRECT |
| 10 | risk/kill_switch.rs | 71, 99, 139 | min_peak_for_drawdown | 1.0 (default), 2% × max_pos | CORRECT |
| 11 | checkpoint/types.rs | 85-107 | serde(default) fields | 8 new fields have it | CHECK: All new fields must have it |

---

## Key Imports Needed

For Phase 1-2 (tick_size_bps integration):
```rust
use crate::market_maker::config::AssetRuntimeConfig;
```

For Phase 4 (cost-basis clamping):
```rust
use crate::market_maker::strategy::market_params::MarketParams;
```

For Phase 5 (AS tautology fix):
```rust
// Already in handlers.rs:
use crate::market_maker::orchestrator::order_ops::TrackedOrder;
use crate::market_maker::fills::PendingFillOutcome;
```

For Phase 11 (checkpoint serialization):
```rust
use serde::{Deserialize, Serialize};
```

---

## Next Steps After Code Review

1. **Phase 1-2**: Set `market_params.tick_size_bps` from `runtime_config.price_multiplier` in orchestrator initialization
2. **Phase 3**: Make MOMENTUM_ENTRY_BPS dynamic: `const_threshold.max(tick_size_bps)`
3. **Phase 4**: Add cost-basis clamp in ladder_strat around line 1850-1949
4. **Phase 5**: Audit all AS learning modules to use `mid_at_placement` from PendingFillOutcome
5. **Phase 7**: Scale QUOTE_LATCH_THRESHOLD_BPS by tick_size_bps
6. **Phase 11**: Run `cargo clippy -- -D warnings` to catch any serde issues

All other phases (6, 8, 9, 10) are already correctly implemented.
