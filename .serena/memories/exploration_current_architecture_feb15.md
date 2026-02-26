# Current Architecture Exploration — Feb 15

## File Overview & Key Structures

### 1. auto_derive.rs — Parameter Derivation
**Purpose**: Auto-derive all trading parameters from single capital input + exchange metadata.

**Key Structures**:
```rust
pub struct ExchangeContext {
    pub mark_px: f64,                  // Current mark price
    pub account_value: f64,            // Equity
    pub available_margin: f64,         // Available margin
    pub max_leverage: f64,             // Max leverage
    pub fee_bps: f64,                  // 1.5 for Hyperliquid
    pub sz_decimals: u32,              // Order size rounding
    pub min_notional: f64,             // 10.0 for Hyperliquid
}

pub struct DerivedParams {
    pub max_position: f64,             // Hard ceiling (contracts)
    pub target_liquidity: f64,         // Per-side liquidity
    pub risk_aversion: f64,            // gamma for GLFT
    pub max_bps_diff: u16,             // Requote threshold
    pub viable: bool,                  // Can trade at all?
    pub diagnostic: Option<String>,    // Explanation if not viable
}

pub fn auto_derive(
    capital_usd: f64,
    spread_profile: SpreadProfile,
    ctx: &ExchangeContext,
) -> DerivedParams
```

**Max Position Computation**:
- `max_from_capital = capital_usd / mark_px`
- `max_from_margin = (available_margin * max_leverage * 0.5) / mark_px` (50% safety factor)
- `max_position = min(max_from_capital, max_from_margin).max(0.0)`
- Viability check: `max_position >= min_order` (min_order = 10 USD * 1.15 / mark_px)

---

### 2. core.rs — MarketMakerConfig
**Purpose**: Runtime configuration for the trading system.

**Key Fields**:
```rust
pub struct MarketMakerConfig {
    pub asset: Arc<str>,
    pub target_liquidity: f64,
    pub risk_aversion: f64,            // gamma
    pub max_bps_diff: u16,             // Requote threshold
    
    // Position limits (CRITICAL)
    pub max_position: f64,             // Hard ceiling (contracts)
    pub max_position_usd: f64,         // USD equivalent (source of truth when set)
    pub max_position_user_specified: bool,  // Was this explicitly set? (v. margin-derived?)
    
    pub decimals: u32,
    pub sz_decimals: u32,
    pub multi_asset: bool,
    pub stochastic: StochasticConfig,
    pub smart_reconcile: bool,
    pub reconcile: ReconcileConfig,
    pub runtime: AssetRuntimeConfig,   // Pre-computed at startup
    pub initial_isolated_margin: f64,
    pub dex: Option<String>,
    pub collateral: CollateralInfo,
    pub impulse_control: ImpulseControlConfig,
    pub spread_profile: SpreadProfile,
    pub fee_bps: f64,                  // 1.5 bps maker fee
}
```

**Key Invariants**:
- `risk_aversion > 0.0` (GLFT formula diverges at zero)
- `max_position > 0.0`
- `target_liquidity > 0.0`
- `fee_bps` in [0.0, 100.0)

---

### 3. inventory_governor.rs — InventoryGovernor
**Purpose**: Enforces `config.max_position` as a HARD ceiling via position zones.

**Position Zones**:
```
Green:  [0%, 60%) → Full two-sided quoting, spread_mult = 1.0
Yellow: [60%, 80%) → Bias toward reducing, spread_mult = 1.0 + (ratio - 0.6) / 0.2 (ramps to 2.0)
Red:    [80%, 100%) → Reduce-only both sides, spread_mult = 2.0
Kill:   [100%+) → Cancel all, spread_mult = 3.0
```

**Key Structures**:
```rust
pub enum PositionZone {
    Green,  // < 60%
    Yellow, // 60-80%
    Red,    // 80-100%
    Kill,   // >= 100%
}

pub struct PositionAssessment {
    pub zone: PositionZone,
    pub max_new_exposure: f64,         // For Yellow/Red zones
    pub position_ratio: f64,           // |position| / max_position
    pub position: f64,                 // Signed position
    pub is_long: bool,
    pub increasing_side_spread_mult: f64,  // Ramp: 1.0 (Green) → 2.0 (Red) → 3.0 (Kill)
}

pub struct InventoryGovernor {
    max_position: f64,                 // Hard ceiling from config
}

impl InventoryGovernor {
    pub fn assess(&self, position: f64) -> PositionAssessment
    pub fn would_exceed(&self, current_position: f64, order_size: f64, is_buy: bool) -> bool
    pub fn is_reducing(&self, position: f64, is_buy: bool) -> bool
    pub fn zone_adjusted_max(&self, regime_volatility_fraction: f64) -> f64
        // Extreme regime (fraction=1.0) shrinks max to 30% of config max
}
```

---

### 4. signal_integration.rs — SignalIntegrator
**Purpose**: Unified interface for all model signals (lead-lag, informed flow, regime kappa, model gating).

**Key Structures**:
```rust
pub struct SignalIntegrator {
    // Analyzers (Phase 1-4 inputs)
    lag_analyzer: LagAnalyzer,
    informed_flow: InformedFlowEstimator,
    regime_kappa: RegimeKappaEstimator,
    model_gating: ModelGating,
    binance_flow: BinanceFlowAnalyzer,
    cross_venue: CrossVenueAnalyzer,
    
    // State
    latest_binance_mid: f64,
    latest_hl_mid: f64,
    last_lead_lag_signal: LeadLagSignal,
    last_flow_decomp: FlowDecomposition,
    last_cross_venue_features: CrossVenueFeatures,
    latest_hl_flow: FlowFeatureVec,
    position: f64,
    max_position: f64,
    
    // CRITICAL: Cross-venue signal availability tracking
    has_had_cross_venue: Cell<bool>,    // Ever had valid cross-venue signal?
    logged_no_signal_warning: Cell<bool>,
    
    // CUSUM preemptive detection (Phase 5)
    cusum_detector: CusumDetector,
    cusum_divergence_bps: f64,
}

pub fn disable_binance_signals(&mut self) {
    // Called when Binance connection lost
    // Clears staleness and sets leading signal actionable=false
}

pub fn set_skew_context(&mut self, position: f64, max_position: f64, signal_alpha: f64) {
    // Sets inventory_skew + alpha_skew, clamped to 80% half-spread
}
```

**Signal Availability (`has_alpha`)** in quote_engine.rs:
```rust
has_alpha: signals.lead_lag_actionable  // From LeadLagSignal
```

**No-Signal Safety Logic**:
- When `has_had_cross_venue.get() = false` (no valid cross-venue feed ever):
  - Quote cap is 30% of max_position (safety mode)
  - Log warning every 10 cycles
- Once any valid signal received, `has_had_cross_venue = true` (sticky)

---

### 5. state_machine.rs — ExecutionMode & Mode Selection
**Purpose**: Pure state machine for mapping inventory/toxicity → execution decisions.

**Key Structures**:
```rust
pub enum ExecutionMode {
    Flat,                          // Cancel all quotes
    Maker { bid: bool, ask: bool }, // GLFT ladder, optionally one-sided
    InventoryReduce { urgency: f64 }, // Aggressive reducing [0, 1]
}

pub struct ModeSelectionInput {
    pub position_zone: PositionZone,
    pub toxicity_regime: ToxicityRegime,
    pub bid_has_value: bool,      // From QueueValueHeuristic
    pub ask_has_value: bool,      // From QueueValueHeuristic
    pub has_alpha: bool,           // lead_lag_actionable
    pub position: f64,
}

pub fn select_mode(input: &ModeSelectionInput) -> ExecutionMode {
    // Priority order:
    // 1. Kill zone → Flat
    // 2. Red zone → InventoryReduce(urgency=1.0)
    // 3. Toxic + flat → Flat
    // 4. Toxic + positioned → InventoryReduce(urgency=0.7)
    // 5. Yellow zone → Maker(reducing side only)
    // 6. No value, no alpha → Flat
    // 7. Default → Maker(sides with value or alpha)
}
```

**Rule 6 (Flat mode when no signals)**: Line 122
```rust
if !input.bid_has_value && !input.ask_has_value && !input.has_alpha {
    return ExecutionMode::Flat;
}
```

---

### 6. queue_value.rs — QueueValueHeuristic
**Purpose**: Level-by-level quote filtering via expected value estimation.

**Key Structures**:
```rust
pub struct QueueValueHeuristic {
    prediction_bias_bps: f64,  // EWMA of prediction errors
    outcome_count: usize,
    alpha: f64,                // EWMA smoothing (0.05)
}

const MAKER_FEE_BPS: f64 = 1.5;
const AS_COST_BENIGN_BPS: f64 = 1.0;
const AS_COST_NORMAL_BPS: f64 = 3.0;
const AS_COST_TOXIC_BPS: f64 = 8.0;
const QUEUE_PENALTY_MAX: f64 = 0.30;

impl QueueValueHeuristic {
    pub fn queue_value(
        &self,
        depth_bps: f64,
        toxicity: ToxicityRegime,
        queue_rank: f64,
    ) -> f64 {
        // depth_bps - expected_as(toxicity) - queue_penalty - 1.5 (fee)
        let expected_as = match toxicity { ... };
        let queue_penalty = depth_bps * 0.30 * queue_rank;
        depth_bps - expected_as - queue_penalty - 1.5 - prediction_bias_bps
    }
    
    pub fn should_quote(&self, depth_bps: f64, toxicity: ToxicityRegime, queue_rank: f64) -> bool {
        self.queue_value(...) > 0.0
    }
}
```

---

### 7. ladder_strat.rs — LadderStrategy
**Purpose**: Multi-level GLFT quoting with Bayesian fill probabilities.

**Key Structures**:
```rust
pub struct LadderStrategy {
    pub risk_config: RiskConfig,
    pub ladder_config: LadderConfig,
    depth_generator: DynamicDepthGenerator,
    fill_model: BayesianFillModel,      // Learns fill probabilities
    pub risk_model: CalibratedRiskModel,
    pub risk_model_config: RiskModelConfig,
    pub kelly_sizer: KellySizer,
}

const MAX_SINGLE_ORDER_FRACTION: f64 = 0.25;  // 25% of effective_max per order
```

**Concentration Fallback Logic**:
- When all levels fail min_notional check → collapse to fewer levels
- Size-capped: single order ≤ 25% of total_size
- Prevents max-position fills on single order

**Key Methods**:
```rust
pub fn generate_concentrated_quotes(
    &self,
    levels: usize,                     // num_levels from config
    total_size: f64,
    effective_max_position: f64,
    is_buy: bool,
) -> Vec<Quote>
```

---

### 8. quote_engine.rs — Quote Generation Main Loop
**Purpose**: Central orchestrator for all quote cycle logic.

**Position Safety Architecture (Lines 144-196)**:

```rust
// PHASE 3: INVENTORY GOVERNOR (First structural check)
let position_assessment = self.inventory_governor.assess(self.position.position());
// → PositionZone (Green/Yellow/Red/Kill)

// If Kill zone → cancel all quotes, return

// PHASE 7: Graduated risk overlay multipliers
let mut risk_overlay_mult = 1.0;
let mut risk_size_reduction = 1.0;
let mut risk_reduce_only = false;

// Yellow zone: widen increasing side, cap exposure
if position_assessment.zone == PositionZone::Yellow {
    risk_overlay_mult *= position_assessment.increasing_side_spread_mult;
}

// Red zone: reduce-only mode
if position_assessment.zone == PositionZone::Red {
    risk_reduce_only = true;
    risk_overlay_mult *= position_assessment.increasing_side_spread_mult;
}

// Circuit breaker & risk checks → additional multiplicative widening
// Cap risk_overlay_mult at 3.0 (single overlay)
risk_overlay_mult = risk_overlay_mult.min(3.0);
```

**Effective Max Position Calculation (Lines 470-497)**:
```rust
// config.max_position is PRIMARY ceiling, margin is solvency floor ONLY
let margin_quoting_capacity = (available_margin * max_leverage / mid) if valid else 0.0;
let pre_effective_max_position = if margin_quoting_capacity > 1e-9 {
    margin_quoting_capacity.min(self.config.max_position)
} else {
    self.config.max_position
};
```

**Mode Selection (Line 2303 in quote_engine context)**:
```rust
let mode = select_mode(&ModeSelectionInput {
    position_zone: position_assessment.zone,
    toxicity_regime: ...,
    bid_has_value: queue_values.bid_has_value,
    ask_has_value: queue_values.ask_has_value,
    has_alpha: signals.lead_lag_actionable,  // Phase 1: cross-venue signal availability
    position: self.position.position(),
});
```

**No-Signal Handling (Signal Integration context)**:
- If `has_had_cross_venue = false` (no valid Binance feed ever):
  - Quote 30% position cap (safety mode)
  - Log warning
- One-sided quoting enabled via ExecutionMode when needed

---

## Key Architecture Decisions

### Position Management
1. **InventoryGovernor** is FIRST check (before any other logic)
2. **config.max_position** is HARD ceiling (never overridden by margin)
3. Margin only acts as solvency floor (can lower effective limit)
4. Zone-based: Green (1.0x) → Yellow (1.0-2.0x ramp) → Red (2.0x) → Kill (3.0x)

### Signal Quality & Safety
1. **has_had_cross_venue**: Cell<bool> tracks if Binance signal ever valid
2. **disable_binance_signals()**: Called on connection loss, clears staleness
3. **No-signal safety**: 30% position cap when signal unavailable
4. **Rule 6**: Flat mode when no queue value + no alpha

### Concentration & Sizing
1. **MAX_SINGLE_ORDER_FRACTION = 0.25**: Prevents max-position fills
2. **Concentration fallback**: Collapses to fewer levels if min_notional blocking
3. **Capital-limited levels**: Compute max viable levels from margin

### Regime-Dependent Parameters
1. **Kappa**: Ladder strategy uses robust_kappa when available (60% blend for HIP-3)
2. **Gamma**: Base γ from config, blended with calibrated risk model
3. **AS buffer**: Dynamic `raw_buffer * (1 - warmup_fraction)`
4. **Spread floor**: 3.0 bps (HL fee) + 1.17×3bps risk ≈ 8.01 bps (paper: 2.5 bps)

---

## Files to Modify (High-Level)

1. **auto_derive.rs**: Add `max_position_usd` output parameter
2. **core.rs**: Already has `max_position_usd` + `max_position_user_specified` fields
3. **inventory_governor.rs**: No changes needed (already enforces hard ceiling)
4. **signal_integration.rs**: Already has `has_had_cross_venue` + `disable_binance_signals()`
5. **state_machine.rs**: `select_mode()` already implements Rule 6 (line 122)
6. **queue_value.rs**: No changes needed
7. **ladder_strat.rs**: May need concentration mode adjustments
8. **quote_engine.rs**: Integrate no-signal safety + concentration mode parameters

---

## Open Questions for Implementation

1. **Concentration mode trigger**: What % position cap when no cross-venue signal?
   - Current: 30% (hardcoded in no-signal safety)
   - Should be configurable?

2. **Ladder level reduction**: When margin-limited, should we:
   - Scale `num_levels` proportionally? (current behavior)
   - Scale `target_liquidity` per-level?
   - Both?

3. **Concentration fallback sizing**: Should MAX_SINGLE_ORDER_FRACTION be regime-dependent?
   - Current: 25% (fixed)
   - Quiet markets: 50%?
   - Cascade: 10%?

4. **Spot loss carry-through**: How should we handle accumulated spot losses across restarts?
   - Reset every session?
   - Track in checkpoint?
   - Use in kill-switch threshold adjustment?
