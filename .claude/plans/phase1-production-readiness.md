# Phase 1: Production Readiness — Team Plan

## Executive Summary

**Current state**: Paper trader generates 19 fills in 218s at 15.48 bps total spread, net PnL -$0.83. Competitive BTC perp top-of-book is 1-3 bps. We're 5-8x too wide.

**Goal**: Fix measurement, tighten spreads, activate signals, harden paper trader, tune parameters — then collect 7+ days of data to validate edge before deploying real capital.

**Team**: 5 agents working in parallel on distinct file sets (no conflicts).

**Success criteria**: Paper Sharpe > 2.0 over 7-day run, Brier score < 0.20, avg spread < 6 bps total.

---

## Agent 1: Calibration Pipeline Engineer

### Mission
Fix the broken calibration pipeline so we can measure model accuracy. Without this, every other optimization is blind guessing.

### Problem
`CalibrationAnalyzer.add_record()` is **never called** despite 19 fills. Predictions exist (`PredictionLogger`), outcomes exist (`OutcomeTracker`), but they are never joined.

### Files Owned
- `src/market_maker/simulation/prediction.rs` — add method to retrieve completed records
- `src/market_maker/simulation/outcome.rs` — add cycle-to-outcome matching
- `src/market_maker/simulation/calibration.rs` — verify add_record() works correctly
- `src/bin/paper_trader.rs` — wire the prediction→outcome→calibration pipeline (lines 1487-1494, 1798, 2080, 2330-2340)

### Detailed Steps

#### Step 1: Capture cycle_id from prediction logging
**Location**: `paper_trader.rs:2080`
```rust
// CURRENT (cycle_id discarded):
prediction_logger.log_prediction(market_snapshot, predictions);

// FIX: capture cycle_id
let cycle_id = prediction_logger.log_prediction(market_snapshot, predictions);
```

#### Step 2: Add public accessor for pending records in PredictionLogger
**Location**: `prediction.rs` — add after line 565
```rust
/// Retrieve a completed record by cycle_id (after outcomes attached)
pub fn get_record(&self, cycle_id: u64) -> Option<PredictionRecord> {
    // Read from pending_records if still in memory
    self.pending_records.lock().unwrap().get(&cycle_id).cloned()
}
```

#### Step 3: Build ObservedOutcomes from fill data
**Location**: `paper_trader.rs` — in the fill processing section (~line 1798)
- After `outcome_tracker.on_fill()`, construct `ObservedOutcomes` from fill data
- Call `prediction_logger.attach_outcomes(cycle_id, outcomes)`

#### Step 4: Feed completed records to CalibrationAnalyzer
**Location**: `paper_trader.rs` — add a periodic check (every 10s or every 100 quotes)
- Retrieve records that have outcomes attached
- Call `calibration_analyzer.add_record(&record)` for each

#### Step 5: Verify report output
- `generate_report()` should now show non-zero Brier scores
- `n_fill_predictions` and `n_as_predictions` should be > 0

### Architecture Notes
- `PredictionRecord.outcomes` is `Option<ObservedOutcomes>` — only records with `Some(...)` are processed by `add_record()`
- `PredictionLogger` stores pending records in `Arc<Mutex<HashMap<u64, PredictionRecord>>>` (line 470)
- `OutcomeTracker` has `create_cycle_attribution()` (line 284) but it's never called — may be useful
- Markout horizons: 100ms, 1s, 10s — the 1s horizon is used for AS calculation

### Checkpoints

| # | Checkpoint | Verification | Done? |
|---|-----------|-------------|-------|
| 1 | cycle_id captured from log_prediction() | grep "let cycle_id" in paper_trader.rs | [ ] |
| 2 | PredictionLogger exposes completed records | Unit test: log→attach→get returns Some | [ ] |
| 3 | ObservedOutcomes constructed from fills | Log: `[CALIBRATION] record added cycle={} fills={}` | [ ] |
| 4 | CalibrationAnalyzer receives records | `n_fill_predictions > 0` in report | [ ] |
| 5 | Brier score is non-zero | `brier_score > 0.0` in calibration_report.json | [ ] |
| 6 | `cargo clippy -- -D warnings` clean | Zero warnings | [ ] |
| 7 | `cargo test` passes | 1925+ tests pass | [ ] |

### Success Markers
- **Minimum**: Brier score > 0.0 with n_samples > 10 after a 120s run
- **Target**: Brier score < 0.25, Information Ratio > 0.3, Conditional Calibration curves non-empty
- **Stretch**: Per-model Brier breakdown (fill model vs AS model)

---

## Agent 2: Spread Construction Tightener

### Mission
Close the gap between GLFT optimal spread and actual quoted spread. Current: 15.48 bps total. Target: < 6 bps total.

### Problem
The GLFT formula produces `optimal_spread_bps=9.0` per side (18 bps total), but we need 2-3 bps per side for competitive quoting. The 15.48 bps actual spread comes from the GLFT being too wide (gamma too high, kappa too low for live) plus 6+ bps of additional widening from floor clamping, geometric spacing, and skew.

### Files Owned
- `src/market_maker/quoting/ladder/depth_generator.rs` — DynamicDepthConfig defaults, GLFT formula, geometric spacing
- `src/market_maker/quoting/ladder/generator.rs` — skew application, RL adjustments
- `src/market_maker/strategy/ladder_strat.rs` — floor computation, depth clamping, entropy optimizer (lines 754-850, 928-976)

### Detailed Steps

#### Step 1: Audit the 11-phase spread pipeline
Trace every bps of widening. The full chain (with line numbers):

| Phase | Location | Mechanism | Current Effect |
|-------|----------|-----------|----------------|
| 1 | `depth_generator.rs:243-255` | GLFT: (1/γ)ln(1+γ/κ)+fee | 9.0 bps per side |
| 2 | `depth_generator.rs:289-294` | `min_spread_floor_bps=4.0` clamp | +0 (GLFT > 4.0) |
| 3 | `depth_generator.rs:297-300` | Hard cap (disabled) | +0 |
| 4 | `depth_generator.rs:567-576` | Geometric ratio 1.5: L0=9, L1=13.5, L2=20.25... | Widens outer levels |
| 5 | `ladder_strat.rs:754-770` | Adaptive floor (Bayesian AS) | +0 to +3 bps |
| 6 | `ladder_strat.rs:777-788` | Conditional AS buffer | +0 to +1.5 bps |
| 7 | `ladder_strat.rs:821-831` | Clamp all depths >= effective_floor | +0 to +4 bps |
| 8 | `ladder_strat.rs:836-850` | Kappa spread cap (if configured) | -0 to -1 bps |
| 9 | `generator.rs:782-866` | Inventory skew (γσ²τ + drift + funding) | +0.5 to +2 bps |
| 10 | `generator.rs:977-1033` | RL policy (clamped ±5 bps) | +0 to +2 bps |
| 11 | `paper_trader.rs:2023-2025` | Measurement: (best_ask - best_bid) / mid × 10000 | Total spread |

#### Step 2: Add per-phase diagnostic logging
**Location**: `ladder_strat.rs` — after each phase, log the current bid/ask depth at touch:
```
[SPREAD TRACE] phase=floor optimal=9.0 after_floor=9.0
[SPREAD TRACE] phase=as_buffer after_buffer=10.5
[SPREAD TRACE] phase=clamp after_clamp=10.5
[SPREAD TRACE] phase=skew after_skew=11.2/10.8
[SPREAD TRACE] phase=rl after_rl=12.0/11.5
[SPREAD TRACE] phase=final total=23.5
```

#### Step 3: Lower min_spread_floor_bps for paper mode
**Location**: `depth_generator.rs:183` — default is 4.0 bps
- Paper mode: lower to 1.0 bps (GLFT should determine the floor, not a hardcoded value)
- Pass this via DynamicDepthConfig constructor or paper-mode flag

#### Step 4: Disable RL adjustments during warmup
**Location**: `generator.rs:977-1033` — RL confidence should be 0 during warmup
- If `warmup_pct < 50%`, set `effective_confidence = 0.0`

#### Step 5: Reduce geometric_ratio for tighter ladders
**Location**: `depth_generator.rs:176` — default is 1.5
- Consider 1.2 for paper mode: L0=9, L1=10.8, L2=13.0 (vs 13.5, 20.25)
- This clusters more size near the touch where fills happen

#### Step 6: Add spread_at_touch logging alongside avg_spread
**Location**: `paper_trader.rs:2023-2025` — also log the GLFT optimal for comparison
```
[SPREAD] total=15.48 optimal_per_side=9.00 floor=2.50 touch_bid=7.5 touch_ask=8.0
```

### Checkpoints

| # | Checkpoint | Verification | Done? |
|---|-----------|-------------|-------|
| 1 | Per-phase spread trace logging added | grep `SPREAD TRACE` in output | [ ] |
| 2 | Identify which phase adds most bps | Analysis doc with bps attribution | [ ] |
| 3 | min_spread_floor_bps lowered for paper | Config change verified in logs | [ ] |
| 4 | RL disabled during warmup | `rl_confidence=0.0` in warmup logs | [ ] |
| 5 | Geometric ratio tightened | Outer level depths closer to touch | [ ] |
| 6 | Avg spread reduced | < 10 bps total in 120s run | [ ] |
| 7 | `cargo clippy -- -D warnings` clean | Zero warnings | [ ] |
| 8 | `cargo test` passes | 1925+ tests pass | [ ] |

### Success Markers
- **Minimum**: Avg spread < 12 bps total (down from 15.48)
- **Target**: Avg spread < 8 bps total (within 2 bps of 2×GLFT optimal)
- **Stretch**: Avg spread < 6 bps total (competitive for BTC perps)

### Key Insight
The spread measurement (15.48 bps) is **total** (bid+ask). GLFT optimal of 9 bps is **per-side**. So 2×9=18 bps expected, meaning our 15.48 is actually tighter than 2×GLFT. The real issue is that GLFT optimal itself is too wide — which is Agent 5's domain (gamma/kappa tuning). Agent 2 should focus on eliminating unnecessary widening beyond GLFT optimal.

---

## Agent 3: Signal Activation & Wiring

### Mission
Activate disconnected signals that are measured but not feeding into quoting decisions. Close open feedback loops.

### Files Owned
- `src/market_maker/estimator/regime_hmm.rs` — Observation struct, new_full() constructor
- `src/market_maker/estimator/fill_rate_model.rs` — wire into quote engine
- `src/market_maker/adverse_selection/estimator.rs` — close AS feedback loop
- `src/market_maker/strategy/signal_integration.rs` — signal routing

### Detailed Steps

#### Step 1: Wire OI/liquidation leading indicators to HMM
**Problem**: `Observation::new()` (regime_hmm.rs:91-100) hardcodes:
- `oi_level: 1.0` (neutral, never updated)
- `oi_velocity: 0.0` (zero, never updated)
- `liquidation_pressure: 0.0` (zero, never updated)

**Fix location**: `paper_trader.rs:782` — change from `Observation::new()` to `Observation::new_full()` (regime_hmm.rs:101-119)

**Data sources needed**:
- `oi_level`: Normalize OI from Hyperliquid API (already tracked in exchange module)
- `oi_velocity`: Rate of change over 1-minute window
- `liquidation_pressure`: OI drop > 5% + extreme funding rate

**If OI data not available in paper mode**: Use 0.0 defaults but wire the code path so it works when data is present.

#### Step 2: Wire falling_knife/rising_knife into LadderStrategy
**Problem**: Momentum signals computed (parameter_estimator.rs:1143-1149) and passed to MarketParams (aggregator.rs:230-231) but **only used in GLFTStrategy** (glft.rs:1037-1068), **not LadderStrategy**.

**Fix location**: `ladder_strat.rs` — after line 964 (margin allocation):
```rust
// Mirror glft.rs:1051-1062 logic:
// If position opposes momentum + severity > 0.5, amplify inventory skew
let momentum_severity = market_params.falling_knife_score
    .max(market_params.rising_knife_score);
let momentum_direction = if market_params.falling_knife_score > market_params.rising_knife_score {
    -1.0 // market falling
} else {
    1.0  // market rising
};
let position_opposes = (inventory_ratio > 0.0 && momentum_direction < 0.0)
    || (inventory_ratio < 0.0 && momentum_direction > 0.0);

if position_opposes && momentum_severity > 0.5 {
    // Amplify inventory reduction urgency
    urgency_score = (urgency_score + momentum_severity / 3.0).min(1.0);
}
```

#### Step 3: Close AS feedback loop
**Problem**: `predicted_alpha` (estimator.rs:408-423) measures "how informed is flow NOW" but is not used to adjust kappa or spread dynamically.

**Current usage**: Only `as_spread_adjustment` is used (a static buffer). The rich `predicted_alpha` signal is wasted.

**Fix location**: `ladder_strat.rs` — near kappa selection (lines 637-660):
```rust
// Reduce effective kappa when informed flow detected
// (lower kappa = wider spreads = more defensive)
let alpha = market_params.predicted_alpha.unwrap_or(0.0);
if alpha > 0.3 && market_params.as_warmed_up {
    effective_kappa *= (1.0 - 0.5 * alpha); // At alpha=0.6, kappa halved
}
```

#### Step 4: Add diagnostic logging for each signal's contribution
For every quote cycle, log:
```
[SIGNALS] alpha={:.3} falling_knife={:.1} rising_knife={:.1} oi_velocity={:.4}
          momentum_opposes={} urgency={:.3} kappa_adj={:.0}
```

### Checkpoints

| # | Checkpoint | Verification | Done? |
|---|-----------|-------------|-------|
| 1 | HMM observation uses new_full() | grep `oi_level` in HMM logs (non-default) | [ ] |
| 2 | falling_knife wired to LadderStrategy | grep `momentum_severity` in ladder logs | [ ] |
| 3 | predicted_alpha modulates kappa | kappa changes when alpha > 0.3 | [ ] |
| 4 | Signal diagnostic logging added | grep `[SIGNALS]` in output | [ ] |
| 5 | Paper run shows signal values changing | Non-zero values for new signals | [ ] |
| 6 | `cargo clippy -- -D warnings` clean | Zero warnings | [ ] |
| 7 | `cargo test` passes | 1925+ tests pass | [ ] |

### Success Markers
- **Minimum**: All signals produce non-zero values in logs
- **Target**: At least one signal demonstrably changes quote behavior (narrower/wider in response to conditions)
- **Stretch**: Per-signal PnL attribution shows net positive contribution

### Important: Do NOT touch
- `src/market_maker/strategy/ladder_strat.rs` lines 754-850 (Agent 2's domain)
- `src/bin/paper_trader.rs` calibration wiring (Agent 1's domain)
- Kill switch / risk config (Agent 4's domain)

---

## Agent 4: Paper Trader Realism & Hardening

### Mission
Make paper trading results predictive of live performance. Current paper mode is too permissive on fills (0.9 touch probability) but too strict on losses (5% drawdown kill). Fix both.

### Files Owned
- `src/bin/paper_trader.rs` — risk config (lines 1412-1425), fill sim config (lines 1469-1480), warmup overrides
- `src/market_maker/risk/kill_switch.rs` — add paper-mode relaxation
- `src/market_maker/adaptive/calculator.rs` — warmup formula adjustments

### Detailed Steps

#### Step 1: Relax kill switch for paper-mode learning
**Location**: `paper_trader.rs:1412-1425`

Current thresholds:
- `max_daily_loss: 500.0` → **Change to 2000.0**
- `max_drawdown: 0.05` → **Change to 0.15** (15%)
- `cascade_kill_threshold: 0.95` → **Change to 2.0**

Rationale: 5% drawdown killed the last run at 218s. For learning, we need to survive adverse periods and see how the system recovers. Live mode keeps the conservative limits.

#### Step 2: Tune fill simulation for realism
**Location**: `paper_trader.rs:1469-1474`

Current paper-mode:
- `touch_fill_probability: 0.9` → **Change to 0.5**
- `queue_position_factor: 0.9` → **Change to 0.6**

Rationale: 90% fill probability is unrealistic — our quotes are at the back of the book (15 bps from mid when top-of-book is 1-3 bps). More conservative fills mean the system learns from realistic conditions, not fantasy.

#### Step 3: Add per-fill PnL attribution logging
**Location**: `paper_trader.rs` — in on_simulated_fill() handler

For each fill, log:
```
[FILL PNL] side=Buy price=68100.00 size=0.01 spread_capture=0.45
           mid_at_fill=68095.50 depth_bps=6.6 inventory_after=-0.02
```

Also track running totals and log every 30s:
```
[PNL SUMMARY] fills=15 spread=+$12.30 as=-$8.40 inv=-$1.20 fee=-$2.10 net=+$0.60
              win_rate=53% avg_fill_quality=1.8bps
```

#### Step 4: Add warmup progression logging
**Location**: `paper_trader.rs` — in periodic diagnostics section

Log warmup components:
```
[WARMUP] progress=35% floor_prog=40%(8/20 fills) kappa_prog=50%(5/10 fills) gamma=warm
         uncertainty_factor=1.065 adaptive_floor=3.2bps
```

#### Step 5: Two-sided fill balance tracking
**Location**: `paper_trader.rs` — add counters

Track and log:
```
[FILL BALANCE] total=19 buys=3 sells=16 imbalance=68%
               avg_buy_depth=5.2bps avg_sell_depth=7.8bps
```

If imbalance > 70%, log a warning — this indicates the system is accumulating directional risk.

#### Step 6: Add session duration protection
**Location**: `paper_trader.rs` — main loop

If kill switch triggers, log the reason and detailed state, but allow configurable restart/continue behavior for learning mode:
```
[KILL SWITCH] reason="Drawdown 10.7% > 5.0%" position=-0.38 unrealized_pnl=-$650
              Continuing in observation-only mode (no new quotes)
```

### Checkpoints

| # | Checkpoint | Verification | Done? |
|---|-----------|-------------|-------|
| 1 | Kill switch limits relaxed for paper | 300s run completes without kill | [ ] |
| 2 | Fill probability reduced to 0.5 | Fewer fills but more realistic | [ ] |
| 3 | Per-fill PnL logging added | grep `[FILL PNL]` in output | [ ] |
| 4 | Warmup progression logging added | grep `[WARMUP]` shows progress | [ ] |
| 5 | Fill balance tracking added | grep `[FILL BALANCE]` shows buy/sell counts | [ ] |
| 6 | 300s run completes successfully | Full duration, no early kills | [ ] |
| 7 | `cargo clippy -- -D warnings` clean | Zero warnings | [ ] |
| 8 | `cargo test` passes | 1925+ tests pass | [ ] |

### Success Markers
- **Minimum**: 300s run completes without kill switch, fills on both sides
- **Target**: 600s run with warmup progressing past 30%, two-sided fills
- **Stretch**: 1800s (30 min) run showing warmup > 50% and adaptive spread tightening

---

## Agent 5: Kappa & Gamma Regime Tuning

### Mission
Find the parameter space that produces competitive spreads (3-6 bps total). Current GLFT gives 9 bps/side — we need 2-3 bps/side.

### Files Owned
- `src/market_maker/estimator/kappa_orchestrator.rs` — confidence gating, paper→live transition
- `src/market_maker/estimator/regime_kappa.rs` — per-regime kappa integration
- `src/market_maker/adaptive/calculator.rs` — warmup formula
- `src/market_maker/adaptive/config.rs` — AdaptiveBayesianConfig

### Detailed Steps

#### Step 1: GLFT sensitivity analysis
Compute the parameter space for competitive spreads.

**GLFT formula**: δ* = (1/γ) × ln(1 + γ/κ) + 0.00015

**Target**: δ* = 2.0 bps (0.0002) per side → 4 bps total spread

| κ | Required γ for δ*=2 bps | Required γ for δ*=3 bps |
|-------|-------------------------|-------------------------|
| 8000 | 0.04 | 0.07 |
| 10000 | 0.05 | 0.09 |
| 15000 | 0.07 | 0.14 |
| 20000 | 0.09 | 0.18 |

**Current**: γ=0.24, κ=8000 → δ*=9.0 bps. Need γ ≈ 0.04-0.07 for competitive spreads.

#### Step 2: Implement gamma regime scaling
**Location**: `ladder_strat.rs` — where gamma is set (early in generate_ladder)

Current: γ=0.24 (static, set in paper_trader.rs:432 as `gamma * 0.5`)

Add regime-dependent scaling:
```rust
let gamma_base = 0.07;  // Competitive base (down from 0.24)
let gamma = match current_regime {
    Regime::Low => gamma_base * 0.7,     // Tighter in calm markets
    Regime::Normal => gamma_base,
    Regime::High => gamma_base * 2.0,    // Wider in vol
    Regime::Extreme => gamma_base * 5.0, // Pull back hard in cascades
};
```

#### Step 3: Implement confidence-gated kappa transition
**Location**: `paper_trader.rs:845` — replace hardcoded floor

```rust
if self.paper_mode {
    let kappa_conf = self.estimator.kappa_orchestrator().confidence();
    let paper_kappa_floor = if kappa_conf < 0.75 {
        8000.0  // During warmup: force competitive kappa
    } else {
        // Transition: blend floor with learned value
        let learned = market_params.kappa_robust;
        (8000.0 * (1.0 - kappa_conf) + learned * kappa_conf).max(2000.0)
    };
    // ... apply floor as before
}
```

#### Step 4: Wire regime kappa into ladder strategy
**Problem**: RegimeKappaEstimator exists (regime_kappa.rs) with per-regime priors [3000, 2000, 1000, 500] but is **not used** in ladder_strat.rs kappa selection (lines 637-660).

**Fix**: In ladder_strat.rs kappa selection, add regime kappa as a blending source:
```rust
// After line 660, before using effective_kappa:
if let Some(regime_kappa) = market_params.regime_kappa {
    // Blend: 70% current selection + 30% regime-specific
    effective_kappa = 0.7 * effective_kappa + 0.3 * regime_kappa;
}
```

#### Step 5: Paper-mode gamma override
**Location**: `paper_trader.rs:432`

Current: `let effective_gamma = if paper_mode { gamma * 0.5 } else { gamma };`

Change to: `let effective_gamma = if paper_mode { 0.07 } else { gamma };`

This forces competitive gamma in paper mode while keeping live mode configurable.

### Checkpoints

| # | Checkpoint | Verification | Done? |
|---|-----------|-------------|-------|
| 1 | GLFT sensitivity table computed | Doc with κ/γ → spread mapping | [ ] |
| 2 | Gamma lowered for paper mode | `gamma=0.07` in logs | [ ] |
| 3 | optimal_spread_bps < 4.0 | Log shows tighter GLFT optimal | [ ] |
| 4 | Confidence-gated kappa transition | kappa_floor reduces as confidence > 0.75 | [ ] |
| 5 | Regime kappa blended in | Different kappa values across regimes | [ ] |
| 6 | 120s run with tight spreads | Avg spread < 6 bps total | [ ] |
| 7 | `cargo clippy -- -D warnings` clean | Zero warnings | [ ] |
| 8 | `cargo test` passes | 1925+ tests pass | [ ] |

### Success Markers
- **Minimum**: optimal_spread_bps < 5.0 per side (down from 9.0)
- **Target**: Avg spread < 6 bps total with 20+ fills in 300s
- **Stretch**: Spreads adapt by regime (tighter in calm, wider in vol)

### Key Math Reference
```
GLFT: δ* = (1/γ) × ln(1 + γ/κ) + fee

When γ << κ: δ* ≈ 1/κ + fee
  → κ=8000: δ* ≈ 1.25 + 1.5 = 2.75 bps
  → κ=20000: δ* ≈ 0.50 + 1.5 = 2.00 bps

When γ/κ ≈ 1: δ* ≈ (1/γ) × 0.693 + fee
  → γ=0.07, κ=8000: δ* = 14.3 × 0.00000875 × 10000 + 1.5 ≈ 2.75 bps
```

---

## Cross-Agent Coordination

### File Ownership (No Conflicts)

| Agent | Owned Files | Do Not Touch |
|-------|-------------|-------------|
| 1 (Calibration) | prediction.rs, outcome.rs, calibration.rs | ladder_strat.rs, depth_generator.rs |
| 2 (Spreads) | depth_generator.rs, generator.rs, ladder_strat.rs (lines 754-850) | prediction.rs, kappa_orchestrator.rs |
| 3 (Signals) | regime_hmm.rs, fill_rate_model.rs, as estimator, signal_integration.rs | depth_generator.rs, kill_switch.rs |
| 4 (Realism) | paper_trader.rs (risk config, fill config, logging), kill_switch.rs | prediction.rs, kappa_orchestrator.rs |
| 5 (Kappa/Gamma) | kappa_orchestrator.rs, regime_kappa.rs, adaptive/calculator.rs, adaptive/config.rs | depth_generator.rs, kill_switch.rs |

### Shared File: paper_trader.rs
Multiple agents touch paper_trader.rs but at different line ranges:
- **Agent 1**: Lines 1487-1494 (creation), 1798 (fill), 2080 (prediction), 2330 (report)
- **Agent 2**: Lines 2023-2025 (spread measurement)
- **Agent 3**: Line 782 (HMM observation)
- **Agent 4**: Lines 1412-1425 (risk config), 1469-1480 (fill config), logging additions
- **Agent 5**: Lines 432 (gamma), 845-856 (kappa floor)

**Rule**: Each agent only touches their designated line ranges. If ranges overlap, the later-numbered agent defers.

### Shared File: ladder_strat.rs
- **Agent 2**: Lines 754-850 (floor, clamping, caps)
- **Agent 3**: Lines 637-660 (kappa selection), 928-976 (momentum/inventory)
- **Agent 5**: Lines 637-660 (regime kappa blending)

**Conflict**: Agent 3 and Agent 5 both touch kappa selection (637-660).
**Resolution**: Agent 5 owns the kappa selection block. Agent 3 adds alpha-based adjustment AFTER Agent 5's regime blending.

---

## Integration Test Protocol

After all 5 agents complete:

### Step 1: Merge and build
```bash
cargo clippy -- -D warnings  # Must be clean
cargo test                    # Must pass 1925+ tests
```

### Step 2: Short validation run (120s)
```bash
cargo run --release --bin paper_trader -- --paper-mode --duration 120
```

**Expected**:
- `optimal_spread_bps < 5.0` (Agent 5: gamma lowered)
- `avg_spread < 8.0 bps` total (Agent 2: widening removed)
- Fills on both sides (Agent 4: balanced sim)
- `[CALIBRATION]` records appearing (Agent 1)
- `[SIGNALS]` values non-zero (Agent 3)
- No kill switch trigger (Agent 4: limits relaxed)

### Step 3: Medium validation run (600s)
```bash
cargo run --release --bin paper_trader -- --paper-mode --duration 600 --report
```

**Expected**:
- 50+ fills across 600s
- Brier score > 0.0 in calibration_report.json
- Warmup progressing past 20%
- Net PnL trending toward breakeven or positive

### Step 4: Long validation run (1800s)
```bash
cargo run --release --bin paper_trader -- --paper-mode --duration 1800 --report
```

**Expected**:
- 200+ fills
- Warmup > 50%
- Adaptive spread tightening visible in logs
- Brier score stabilizing

---

## Go/No-Go Criteria for Phase 2

| Metric | Phase 1 Target | Phase 2 Requirement | Production Threshold |
|--------|---------------|--------------------|--------------------|
| Avg Spread (total) | < 8 bps | < 6 bps | < 4 bps |
| Fills per 300s | > 30 | > 100 | N/A (live) |
| Brier Score | > 0.0 | < 0.20 | < 0.15 |
| Net PnL (300s) | > -$5 | > $0 | > $0 |
| Warmup (600s) | > 20% | > 50% | 100% |
| Kill switch triggers | 0 in 600s | 0 in 1800s | N/A |
| Paper Sharpe (7d) | N/A | > 2.0 | > 1.5 |
| Two-sided fill ratio | > 30/70 | > 40/60 | > 45/55 |
