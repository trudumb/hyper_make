# Principled Systems Redesign: 6-Engineer Implementation Plan

## Context

Session 5 analysis (1h33m, 41 fills, +$0.75, Sharpe 449) revealed 5 systemic issues that trace to missing foundational assumptions. This plan designs the systems that would have existed if these concerns were first-class from day one.

**Findings:**
1. **Signal Tautology**: All 6 signals derive from HL volume imbalance (no Binance). Edge attribution divides PnL equally → tautological 6.0/-6.0 bps.
2. **Static Gamma**: `gamma_mult=1.20` constant across all regimes. **Root cause: single-line wiring bug** — quote_engine.rs line 1158 reads discrete regime params (rarely transitions) instead of blended (continuously varies).
3. **Position Swings**: -4.42 to +3.40 (7.82 range) on $100 capital. No capital-proportional limits.
4. **Order Waste**: 31% of generated orders filtered. 8% of cycles produce zero orders. BBO crossing = 46% of filtering.
5. **Magic Numbers**: ~80 hardcoded constants with no statistical justification.

---

## Engineer 1: Risk — Capital-Proportional Position & Kill Switch

**Owner**: `risk` agent
**Files**: `risk/inventory_governor.rs`, `risk/kill_switch.rs`, `risk/state.rs`, `orchestrator/quote_engine.rs`

### 1a. Capital-Proportional Position Limits (P1)

New `CapitalProportionalConfig` struct in `risk/inventory_governor.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CapitalProportionalConfig {
    pub position_budget_pct: f64,     // 0.50 — max notional as % of capital
    pub baseline_daily_vol: f64,      // 0.02 — baseline vol for scaling
    pub vol_sensitivity: f64,         // 0.5 — how fast limits tighten with vol
    pub min_position_usd: f64,        // 10.0 — floor (exchange minimum)
    pub enabled: bool,                // true
}
```

Formula: `max_position = (capital × budget_pct × vol_adj) / mark_price`
- vol_adj = `1 / (1 + sensitivity × max(0, realized_vol/baseline - 1))`
- $100 capital, HYPE $30.90, normal vol: max ≈ 0.83 contracts
- $100, high vol: max ≈ 0.32 contracts (prevents -4.42 swings)

Wire in `quote_engine.rs:510-516`:
```rust
let capital_max = capital_proportional_config.max_position_contracts(
    margin_state.account_value, market_params.sigma_effective, self.latest_mid,
);
let pre_effective_max_position = margin_quoting_capacity
    .min(self.config.max_position)
    .min(capital_max);  // NEW ceiling
```

### 1b. Position Swing Governor (P2)

Track 5-minute rolling position high/low watermarks in `RiskState`:
```rust
pub position_swing_5m: f64,  // (max - min) / max_position over 5m window
```

Risk response: `swing > 1.0` → widen spreads, `> 1.5` → reduce-only, `> 2.0` → pull quotes.

### 1c. Capital-Proportional Kill Switch (P2)

New `KillSwitchConfig::from_capital(capital_usd, max_position_usd)`:
- `max_daily_loss`: capital × 5% (not fixed $50)
- `max_absolute_drawdown`: capital × 2%
- `max_position_value`: from capital-proportional config
- `min_peak_for_drawdown`: capital × 0.5%

For $100: loss=$5, drawdown=$2. For $10K: loss=$500, drawdown=$200.

---

## Engineer 2: Strategy — Regime-Adaptive Gamma

**Owner**: `strategy` agent
**Files**: `orchestrator/quote_engine.rs`, `strategy/regime_state.rs`, `strategy/risk_model.rs`, `strategy/market_params.rs`

### 2a. Fix Gamma Wiring Bug (P0 — deploy immediately)

**Single-line fix** in `quote_engine.rs` line 1158:
```rust
// BEFORE (broken — reads discrete params, rarely transitions):
market_params.regime_gamma_multiplier = rp.gamma_multiplier;

// AFTER (reads blended, continuously varies):
market_params.regime_gamma_multiplier = self.stochastic.regime_state.effective_gamma_multiplier();
```

**Root cause**: `rp` = `self.stochastic.regime_state.params` = discrete regime params. These only update when `ConvictionState::should_transition()` returns true (requires conviction margin > 0.20 AND KL divergence > 0.3 nats AND regime-dependent dwell time). All three conditions are rarely met simultaneously → label stays `Normal` (gamma_mult=1.20) nearly always.

Meanwhile `effective_gamma_multiplier()` (regime_state.rs:773) returns the EWMA-smoothed belief-weighted average — this continuously varies but is **never read by the quote engine**.

### 2b. Widen Regime Gamma Range (P0, same PR)

In `regime_state.rs`, update `MarketRegime::default_params()`:

| Regime | Current γ_mult | Proposed γ_mult | Rationale |
|--------|---------------|----------------|-----------|
| Calm | 1.0 | 0.7 | Tighter spreads → more fills → more edge |
| Normal | 1.2 | 1.0 | Baseline = neutral multiplier |
| Volatile | 2.0 | 3.0 | Material widening |
| Extreme | 3.0 | 6.0 | Aggressive cascade defense |

Also in `effective_gamma_multiplier()`: remove `.max(1.0)` floor → `.max(0.5)` (allow Calm to tighten below 1.0).

Update `GAMMA_MULTIPLIERS` const: `[0.7, 1.0, 3.0, 6.0]`.

### 2c. Add regime_severity Feature to CalibratedRiskModel (P1)

New field in `RiskFeatures`: `pub regime_severity: f64` (Calm=0, Normal=0.33, Volatile=0.67, Extreme=1.0).
New coefficient in `CalibratedRiskModel`: `pub beta_regime: f64` (default 2.0).

Impact: Even with all other features cold/default:
- Calm: gamma = exp(ln(0.15) + 2.0×0.0) = 0.15
- Normal: gamma = exp(ln(0.15) + 2.0×0.33) = 0.29
- Volatile: gamma = exp(ln(0.15) + 2.0×0.67) = 0.57
- Extreme: gamma = exp(ln(0.15) + 2.0×1.0) = 1.11

Guarantees 7.4x gamma variation from Calm to Extreme through calibrated model alone.

Wire `regime_severity` via `MarketParams` from blended beliefs in `quote_engine.rs`.

### 2d. Increase CalibratedRiskModel Beta Sensitivity (P1, with 2c)

| Beta | Current | Proposed | Effect at feature=1.0 |
|------|---------|----------|-----------------------|
| beta_volatility | 1.0 | 1.5 | 4.5x gamma |
| beta_toxicity | 0.5 | 0.8 | 2.2x gamma |
| beta_inventory | 0.3 | 0.5 | 1.65x gamma |
| beta_hawkes | 0.4 | 0.6 | 1.8x gamma |
| beta_regime | — | 2.0 | 7.4x gamma (NEW) |
| gamma_min | 0.05 | 0.03 | Tighter Calm floor |
| gamma_max | 5.0 | 8.0 | Wider Extreme ceiling |

---

## Engineer 3: Signals — Independent Data Sources

**Owner**: `signals` agent
**Files**: `estimator/book_shape_signal.rs` (NEW), `estimator/oi_signal.rs` (NEW), `strategy/signal_integration.rs`, `orchestrator/handlers.rs`

### 3a. BookShapeSignal — L2 Order Book (P1)

New file `estimator/book_shape_signal.rs`. Uses `HlOrderBook` data source (independent of trade flow).

Extracts:
- **Depth asymmetry**: bid_depth / ask_depth at top 5 levels → directional signal
- **Kappa asymmetry**: log(kappa_bid / kappa_ask) — reuses `BookKappaEstimator` regression
- **Depth momentum**: rate of change of asymmetry

Wire from `handlers.rs:handle_l2_book()` → `signal_integrator.on_l2_book(bids, asks, mid)`.

**Independence**: L2 book shape reflects resting (passive) flow, causally prior to aggressive trade flow that all current signals measure.

### 3b. FundingAlphaSignal — Funding Rate Mean-Reversion (P1)

No new file — infrastructure exists in `process_models/funding.rs`. Fields `funding_basis_velocity`, `funding_premium_alpha`, `funding_skew_bps` exist in `IntegratedSignals` but are NOT wired into `combined_skew_bps`.

Fix: In `signal_integration.rs:get_signals()`, add funding-derived skew:
```rust
let funding_skew_bps = if self.config.use_funding_alpha {
    let excess = funding_rate - funding_ewma;
    (-excess * 10_000.0 * 0.1).clamp(-max_funding_skew_bps, max_funding_skew_bps)
} else { 0.0 };
```

**Independence**: Funding rate = aggregate market positioning (minutes-to-hours timescale), completely different from trade flow (milliseconds).

### 3c. OiMomentumSignal — Open Interest (P2)

New file `estimator/oi_signal.rs`. Classic CME interpretation:
- OI up + price up = new longs (bullish conviction)
- OI down + price down = liquidation cascade → widen spreads

Lower priority — requires verifying OI data is flowing through WS handlers.

### 3d. Graceful Degradation (P1)

In `SignalIntegratorConfig::hip3()`:
```rust
use_lead_lag: false,     // No Binance → honest about it
use_cross_venue: false,  // No Binance → honest about it
use_book_shape: true,    // NEW
use_funding_alpha: true, // NEW
```

Gate HL-native fallback removal behind `disable_hl_native_fallback: bool` (default: false). Enable after paper-validating new signals.

---

## Engineer 4: Analytics — Honest Attribution

**Owner**: `analytics` agent
**Files**: `analytics/signal_independence.rs` (NEW), `analytics/orthogonal_attribution.rs` (NEW), `analytics/live.rs`, `analytics/attribution.rs`

### 4a. SignalIndependenceTracker (P1)

New file `analytics/signal_independence.rs`.

Tracks pairwise Pearson correlation between signals over a rolling 500-cycle window. Computes **effective independent signal count** via participation ratio:
```
effective_n = n² / Σᵢⱼ C²ᵢⱼ
```
Where C is the correlation matrix. When all signals identical: effective_n=1. When all independent: effective_n=n. O(n²), no eigensolver needed.

Also tracks `is_degenerate(a, b) -> bool` when |ρ| > 0.8.

### 4b. Information Scarcity Premium (P1)

In `quote_engine.rs`, when `effective_count < nominal_count`:
```rust
let info_deficit = 1.0 - (effective_count / nominal_count).sqrt();
let scarcity_premium_bps = info_deficit * 2.0;  // max 2 bps
```

Fewer independent signals → wider spreads (honest about uncertainty).

### 4c. Shapley R² Attribution (P2)

New file `analytics/orthogonal_attribution.rs`.

Online covariance tracking (Welford's algorithm) + exact Shapley decomposition of R². For n≤9 signals: 2⁹=512 coalitions, each ≤9×9 matrix inversion. Total: ~5ms, run every 200 cycles.

Correctly handles correlated signals: two perfectly correlated signals each get 50% of shared R², not 100% each.

---

## Engineer 5: Infra — GenerationEnvelope

**Owner**: `infra` agent
**Files**: `quoting/envelope.rs` (NEW), `orchestrator/quote_engine.rs`, `strategy/ladder_strat.rs`, `quoting/ladder/generator.rs`, `orchestrator/reconcile.rs`

### 5a. GenerationEnvelope Struct (P1)

New file `quoting/envelope.rs`:
```rust
pub struct GenerationEnvelope {
    // Price bounds
    pub bid_price_ceiling: f64,  // exchange_ask - 2*tick - staleness
    pub ask_price_floor: f64,    // exchange_bid + 2*tick + staleness
    // Capacity (per side)
    pub bid_total_capacity: f64, // min(position_cap, exchange_cap, margin_cap)
    pub ask_total_capacity: f64,
    // Level count
    pub max_bid_levels: usize,   // min(configured, capacity/min_size, quota)
    pub max_ask_levels: usize,
    // Side permissions (from ExecutionMode)
    pub bid_allowed: bool,
    pub ask_allowed: bool,
    // Exchange precision
    pub tick_size: f64,
    pub min_order_size: f64,
    pub min_notional: f64,
}
```

Computed ONCE per cycle. Unifies ALL constraints into a single source of truth.

### 5b. Wire Envelope → Generator (P1)

1. Move `select_mode()` BEFORE `calculate_ladder()` in quote_engine.rs (safe — all inputs already available)
2. Add early exit: `if envelope.is_empty() { return Ok(()); }` → eliminates 8% zero-order cycles
3. Pass envelope to `ladder_strat.rs:generate_ladder()` → generator uses envelope bounds by construction
4. `build_raw_ladder()` uses envelope price bounds instead of computing its own BBO margins

### 5c. Convert Filters to Monitoring Assertions (P2)

Keep all downstream filters in `reconcile.rs` but add violation counters:
- `envelope_bbo_violation_count`
- `envelope_capacity_violation_count`

Target metrics: generate-to-place ratio > 90% (up from 69%).

**Expected elimination**: 813 "Reducing ladder levels" warnings → 0. 91 zero-order cycles → ~5. BBO crossing filtering → <1%.

---

## Engineer 6: Config — Magic Number Extraction

**Owner**: Lead
**Files**: `config/risk.rs`, `config/stochastic.rs`, `quoting/ladder/mod.rs`, `strategy/market_params.rs`, `strategy/params/aggregator.rs`, + consumer files

### 6a. New Config Structs (P2)

6 new structs, all `#[serde(default)]`:
- `RegimeUtilizationConfig` — margin util curve (0.85→0.50 by regime)
- `TrendSignalConfig` — momentum thresholds, blend weights
- `RiskPremiumConfig` — regime premiums, funding scaling, BOCPD gates
- `ToxicityCancelConfig` — cancel threshold, cooldown, addon curve
- `HawkesScalingConfig` — baseline, sensitivity, max ratio
- `CalibrationGateConfig` — observation thresholds, confidence caps

### 6b. Extend Existing Configs (P2)

- `LadderConfig` +7 fields: quota scaling, geometric ratio, regime multipliers, inflation cap
- `DynamicRiskConfig` +3 fields: no-fill penalty bounds

### 6c. Replace ~80 Magic Numbers (P2)

~70 replacements across `ladder_strat.rs`, `quote_engine.rs`, `glft.rs`, `position_manager.rs`. All defaults match current values → zero behavioral change.

---

## Implementation Priority

| Priority | Phase | Engineer | Risk | Impact |
|----------|-------|----------|------|--------|
| **P0** | 2a: Fix gamma wiring | Strategy | Zero (1 line) | Unlocks regime-adaptive risk |
| **P0** | 2b: Widen gamma range | Strategy | Low | Meaningful spread variation |
| **P1** | 1a: Capital-proportional limits | Risk | Low | Prevents -4.42 swings on $100 |
| **P1** | 2c+2d: regime_severity + betas | Strategy | Low | Calibrated model works properly |
| **P1** | 3a+3b: BookShape + FundingAlpha | Signals | Medium | Independent signals for HIP-3 |
| **P1** | 3d: Graceful degradation | Signals | Medium | Honest about missing data |
| **P1** | 4a+4b: Independence tracker | Analytics | Low | Detects tautology at runtime |
| **P1** | 5a+5b: GenerationEnvelope | Infra | Medium | 31% waste → <5% |
| **P2** | 1b+1c: Swing governor + kill switch | Risk | Low | Defense-in-depth |
| **P2** | 3c: OI momentum | Signals | Low | Additional independent signal |
| **P2** | 4c: Shapley attribution | Analytics | Low | Causal attribution |
| **P2** | 5c: Filter monitoring | Infra | None | Observability |
| **P2** | 6a-c: Magic number extraction | Lead | Low | All params tunable |

---

## Verification

After each engineer's changes (sequential `cargo` commands):
1. `cargo clippy -- -D warnings` — clean
2. `cargo test` — all ~3011 tests pass
3. Paper session behavioral checks per phase

**Key behavioral targets:**
- gamma_mult varies between 0.7-6.0 across cycles (currently stuck at 1.20)
- Max position < 2 contracts on $100 capital (currently -4.42)
- Signal marginal values differentiated (currently all identical 6.0 bps)
- Generate-to-place ratio > 90% (currently 69%)
- Zero "Reducing ladder levels" warnings
- Effective independent signal count reported in logs

---

## New Files Created

| File | Engineer | Lines (est.) |
|------|----------|-------------|
| `quoting/envelope.rs` | Infra | ~150 |
| `analytics/signal_independence.rs` | Analytics | ~200 |
| `analytics/orthogonal_attribution.rs` | Analytics | ~250 |
| `estimator/book_shape_signal.rs` | Signals | ~150 |
| `estimator/oi_signal.rs` | Signals | ~120 |

## Key Existing Files Modified

| File | Engineers | Changes |
|------|----------|---------|
| `orchestrator/quote_engine.rs` | Strategy, Infra, Analytics, Risk | Gamma wiring fix, envelope, early-exit, scarcity premium, capital limits |
| `strategy/regime_state.rs` | Strategy | Widen gamma range, update GAMMA_MULTIPLIERS |
| `strategy/risk_model.rs` | Strategy | Add regime_severity, increase betas |
| `strategy/signal_integration.rs` | Signals | New signals, graceful degradation, HIP-3 config |
| `strategy/ladder_strat.rs` | Infra, Config | Envelope reads, magic number extraction |
| `risk/inventory_governor.rs` | Risk | CapitalProportionalConfig |
| `risk/kill_switch.rs` | Risk | from_capital() constructor |
| `analytics/live.rs` | Analytics | Wire independence tracker + attributor |
| `orchestrator/handlers.rs` | Signals | Wire BookShape from L2, OI data |
| `config/risk.rs` | Config | New config structs |
| `config/stochastic.rs` | Config | New config structs |
