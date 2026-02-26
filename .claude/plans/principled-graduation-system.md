# Principled Paper-to-Live Graduation System

## Diagnosis: Why Paper Doesn't Graduate to Live

### The 6 Structural Gaps

After auditing all four subsystems (paper trading, risk/safety, calibration/validation, execution/exchange), the answer is clear: **the system treats paper→live as a binary switch, when it should be a continuous confidence spectrum**.

#### Gap 1: Fill Simulation Fidelity
Paper uses `P(fill) = 0.3 × price_factor × size_factor × queue_factor`. Live has cancel-fill races, order rejections, latency variance (100-300ms), and queue priority loss. Paper systematically overestimates fill rate and underestimates adverse selection. Any paper Sharpe/edge metric is biased upward. **There is no measurement of this bias.**

#### Gap 2: Exchange Interaction Layer Untested
Paper uses 3 WS channels, instant order confirmation, no rejections, no quota limits. Live uses 9 channels, has quota death spirals (7% on hyna), connection drops, and margin-based position limits. Every live incident (Feb 9 kill switch failure, Feb 10 reduce-only bypass, Feb 12 emergency paralysis, Feb 16 cold-start death spiral) was an exchange-interaction bug that paper **cannot surface**.

#### Gap 3: Metrics Observed But Not Acted Upon
- Brier score: computed, never gates
- Signal health (MI): tracked, never queried by production code
- Edge bias: detected via EWMA, never auto-corrected
- Regime emission params: never update online
- CalibrationGate: runs once at startup, never again

The system computes 50+ metrics but only a handful actually modify trading behavior.

#### Gap 4: Binary Mode Switch
Exactly two modes exist: paper ($0 at risk) or live (full exposure). No shadow mode, no graduated sizing, no "trade at 10% while validating." `CapitalAwarePolicy` has tiers but they're static config, not dynamically adjusted.

#### Gap 5: No Sim-to-Real Gap Measurement
When paper predicts +3 bps edge and live realizes +0.5 bps, no system tracks this discrepancy. Edge validation exists per-fill but not as a paper→live correlation metric. Without measuring the gap, you can't close it.

#### Gap 6: Safety Systems Discovered Bugs Only In Live
Kill switch was non-functional, position limits had 10-100x bypass, emergency pull caused paralysis, cold-start produced death spiral. Each was discovered only under real money pressure. Paper's simplified execution never exercises these code paths.

### The Fundamental Insight

**Paper tests the MATH** (GLFT spreads, signal integration, ladder generation). It does **not** test the **ENGINEERING** (order lifecycle, state reconciliation, exchange interaction, failure recovery). And the engineering is where every live bug has been.

---

## Principled Design: Trading as a Continuous Confidence Spectrum

### Core Principle

> "Confidence is not a gate — it's a throttle."

Instead of paper → live, the system maintains a **TradingConfidence** that continuously maps to exposure:

```
Confidence    Exposure Level       Size Multiplier    Behavior
─────────────────────────────────────────────────────────────
0.00 - 0.10   Shadow               0%                Compute quotes, don't place
0.10 - 0.30   MicroLive            5-10%             Minimal size, learning fills
0.30 - 0.50   Ramping              10-50%            Size proportional to confidence
0.50 - 0.70   Cautious             50-80%            Normal but conservative
0.70 - 0.90   Full                 80-100%           Standard operations
0.90 - 1.00   Optimized            100%+             Can tighten spreads
```

**Asymmetric transitions**: Demotion is instant (one bad event halves exposure). Promotion is slow (requires sustained good metrics over configurable windows, minimum dwell time at each level).

### The Central Type

```rust
pub struct TradingConfidence {
    // Component scores (0.0 to 1.0)
    model_confidence: f64,      // From continuous calibration
    execution_confidence: f64,  // From exchange health + sim-to-real gap
    safety_confidence: f64,     // From drawdown + risk state

    // Derived
    overall: f64,               // Weighted blend
    exposure_level: ExposureLevel,
    size_multiplier: f64,

    // Hysteresis
    last_promotion: Instant,
    last_demotion: Instant,
    min_dwell_at_current: Duration,

    // History for promotion criteria
    fill_count_at_level: u64,
    positive_edge_streak: u64,
    time_at_level: Duration,
}
```

This lives at the CENTER of the architecture:
- **Consulted by**: QuoteEngine (size), ReconcileLoop (position limits), RiskOverlay (spread multiplier)
- **Updated by**: ContinuousCalibrationGate, SimToRealTracker, SafetyMonitor, ExchangeHealthMonitor

---

## 6 Subsystems (6 Engineers)

### Engineer 1: Shadow Trading Engine (Sim-to-Real Bridge)

**Purpose**: Quantify the gap between paper predictions and live reality.

**Design**: Run the full quoting pipeline in parallel — one path computes paper fills, one path observes real fills. Compare continuously.

**Key Metric**: `sim_to_real_gap = EWMA(|paper_edge - live_edge|)`

**What exists today**: `PaperEnvironment` + `FillSimulator` + `AnalyticsLogger`

**What's missing**:
- Parallel shadow mode (paper alongside live, not instead of)
- Sim-to-real correlation tracker
- Fill model calibration (use live fills to update paper fill simulator parameters)
- Paper fill probability validation (predicted P(fill) vs observed fill rate)

**New modules**:
```
src/market_maker/shadow/
├── mod.rs              — ShadowEngine: parallel paper evaluation
├── correlation.rs      — SimToRealTracker: gap measurement
├── fill_calibrator.rs  — Online FillSim parameter updates from live fills
└── paper_validator.rs  — P(fill) predicted vs observed
```

**Integration**:
- Runs every quote cycle alongside real quoting
- Produces `sim_to_real_gap` → feeds `execution_confidence`
- When gap > threshold, flags demotion signal to TradingConfidence

---

### Engineer 2: Graduated Exposure Controller (Capital Allocation)

**Purpose**: Map confidence → position limits → size scaling, with automatic promotion/demotion.

**Design**: Replaces static `CapitalAwarePolicy` tiers with dynamic exposure levels.

**What exists today**: `CapitalAwarePolicy` (static tiers), `warmup_progress` (one-directional 0→1.0), `adaptive_warmup_progress` in `MarketParams`

**What's missing**:
- Dynamic tier promotion/demotion
- Bi-directional confidence (can go DOWN, not just up)
- Promotion criteria (sustained positive edge, low Brier, low sim-to-real gap)
- Demotion triggers (safety events, model degradation, gap widening)
- Minimum dwell times per level

**New modules**:
```
src/market_maker/graduation/
├── mod.rs              — TradingConfidence struct + computation
├── exposure.rs         — ExposureController: level transitions
├── promotion.rs        — PromotionCriteria: evidence-based advancement
├── demotion.rs         — DemotionTriggers: fast automatic reduction
└── history.rs          — LevelHistory: audit trail of transitions
```

**Promotion Criteria** (ALL must hold for `min_dwell_duration`):
```rust
pub struct PromotionCriteria {
    min_fills_at_level: u64,        // e.g., 50 fills
    min_dwell_duration: Duration,   // e.g., 30 minutes
    max_brier_score: f64,           // e.g., 0.25 (better than uninformed)
    min_information_ratio: f64,     // e.g., 0.5
    max_sim_to_real_gap_bps: f64,   // e.g., 2.0 bps
    max_drawdown_pct: f64,          // e.g., 1.0%
    min_positive_edge_pct: f64,     // e.g., 60% of fills had positive edge
}
```

**Demotion Triggers** (ANY one triggers immediate demotion):
```rust
pub enum DemotionTrigger {
    DrawdownExceeded(f64),          // > threshold → drop 2 levels
    SafetyEscalation(SafetyLevel),  // Any Level 2+ → drop to MicroLive
    ModelDegradation(String),       // IR dropped below 0.3 → drop 1 level
    SimToRealDiverged(f64),         // Gap > 5 bps → drop 1 level
    ExchangeHealthDegraded,         // Reconnections, quota issues → drop 1
    ManualOverride,                 // Operator intervention
}
```

---

### Engineer 3: Continuous Model Validator (Calibration Gate That Never Closes)

**Purpose**: Transform the one-time CalibrationGate into a continuous "should I still be trading at this level?" assessment.

**What exists today**: `CalibrationGate` (runs once), `ModelGatingSystem` (graduated weights), `BrierScoreTracker`, `InformationRatioTracker`, `SignalHealthMonitor`

**What's missing**:
- Continuous gate evaluation (every N fills, not just startup)
- Brier score → model weight feedback (not just computation)
- Signal MI → automatic retirement (below 30% baseline for 20 periods)
- Regime emission param updates (online learning)
- Model confidence → exposure feedback (low confidence → lower size)

**New modules**:
```
src/market_maker/validation/
├── mod.rs                  — ContinuousValidator: orchestrates all checks
├── model_scorecard.rs      — Per-model live performance tracking
├── signal_retirement.rs    — Automatic signal retirement on MI decay
├── regime_updater.rs       — Online HMM emission parameter updates
└── confidence_aggregator.rs — model_confidence computation
```

**Model Scorecard** (evaluated every 20 fills):
```rust
pub struct ModelScorecard {
    pub model_name: String,
    pub brier_score: f64,           // Rolling 100-fill window
    pub information_ratio: f64,     // Rolling
    pub mutual_information: f64,    // vs baseline
    pub edge_bias_bps: f64,         // EWMA of predicted - realized
    pub calibration_slope: f64,     // Regression of predicted vs realized
    pub last_evaluated: Instant,
    pub verdict: ModelVerdict,      // Healthy / Degraded / Retired
}
```

**Verdicts**:
- **Healthy**: Brier < 0.25, IR > 0.5, MI > 50% baseline → full weight
- **Degraded**: Any metric borderline → graduated_weight applies (5-50%)
- **Retired**: Brier > 0.35 OR IR < 0.3 OR MI < 30% for 20 periods → weight = 0, signal disabled

**Integration**: `model_confidence = mean(healthy_model_count / total_model_count, mean_model_scores)`

---

### Engineer 4: Exchange Hardening Layer (Realistic Failure Testing)

**Purpose**: Close the gap between paper's perfect exchange and live's messy reality.

**What exists today**: `PaperEnvironment`, `FillSimulator`, `ConnectionSupervisor`, `ProactiveRateLimiter`

**What's missing**:
- Enhanced paper mode (realistic latency distribution, rejection rates, quota modeling)
- Chaos injection framework (drop messages, inject latency, simulate 429s)
- Dry-run mode (send + immediate cancel — tests connectivity without real exposure)
- Order timeout detection (placed > 5s with no confirmation → stale)
- Quota reservation (always keep 5% headroom for emergency cancels)

**New modules**:
```
src/market_maker/simulation/
├── enhanced_sim.rs     — EnhancedFillSimulator with realistic failure modes
├── chaos.rs            — ChaosInjector: random failure injection
├── latency_model.rs    — Realistic latency distribution (log-normal + spikes)
└── quota_sim.rs        — Quota consumption simulation

src/market_maker/execution/
├── timeout_detector.rs — Order timeout detection + stale marking
├── quota_reserve.rs    — Quota reservation for emergency cancels
└── dry_run.rs          — DryRunExecutor: place + immediate cancel
```

**Enhanced Paper Simulation**:
```rust
pub struct EnhancedSimConfig {
    // Latency
    pub mean_latency_ms: f64,           // 150ms (log-normal)
    pub latency_std_ms: f64,            // 50ms
    pub spike_probability: f64,          // 1% chance of 500ms+ spike

    // Rejections
    pub rejection_rate: f64,             // 2% of orders rejected
    pub margin_cancel_rate: f64,         // 0.5% margin-cancelled

    // Quota
    pub simulate_quota: bool,            // Model IP + address limits
    pub quota_429_probability: f64,      // 0.1% random 429

    // Connection
    pub disconnect_probability: f64,     // 0.01% per second
    pub disconnect_duration_ms: (u64, u64), // 500ms to 10s
}
```

**Execution Levels** (replaces binary paper/live):
```
Level 0: Pure paper (current FillSimulator)
Level 1: Enhanced paper (realistic failures)
Level 2: Dry-run (real exchange, place + cancel)
Level 3: Micro-live (real orders, minimal size)
Level 4: Full live
```

---

### Engineer 5: Automated Safety Escalation (Graduated Defense)

**Purpose**: Replace the binary kill switch with graduated response levels.

**What exists today**: `KillSwitch` (Level 4 only), `CircuitBreaker` (defined, not integrated), `DrawdownTracker` (graduated but separate), `ExecutionMode` (Flat/Maker/InventoryReduce)

**What's missing**:
- Unified escalation system with levels 1-3
- Minimum dwell times (prevent oscillation)
- Automatic de-escalation (sustained good metrics)
- Risk budget (PaR — position at risk)
- Safety events → TradingConfidence feedback

**New modules**:
```
src/market_maker/safety/
├── escalation.rs       — SafetyEscalator: unified multi-level system
├── levels.rs           — SafetyLevel definitions + transition rules
├── risk_budget.rs      — Position-at-Risk (PaR) monitoring
└── deescalation.rs     — Evidence-based recovery criteria
```

**Safety Levels**:
```rust
pub enum SafetyLevel {
    /// Normal operations. All systems green.
    Green,

    /// Caution: widen spreads 1.5x, reduce new order size 50%.
    /// Entry: drawdown > 0.5%, model degradation, exchange reconnection
    /// Min dwell: 30 seconds
    Yellow,

    /// Alert: single-side only (reduce inventory), size 25%.
    /// Entry: drawdown > 1%, 2+ models degraded, position > 70% limit
    /// Min dwell: 60 seconds
    Orange,

    /// Danger: cancel all quotes, reduce-only.
    /// Entry: drawdown > 2%, cascade detected, exchange health critical
    /// Min dwell: 5 minutes
    Red,

    /// Emergency: hard shutdown, persist state, postmortem dump.
    /// Entry: drawdown > 5%, position runaway, stale data > threshold
    /// This is the current KillSwitch — preserved as last resort.
    Kill,
}
```

**Transition Rules**:
```rust
pub struct EscalationRules {
    // Escalation: instant, no minimum dwell required
    pub escalate_instantly: bool,  // true — any trigger → immediate

    // De-escalation: slow, evidence-based
    pub deescalation_criteria: DeescalationCriteria,
    pub min_dwell_per_level: HashMap<SafetyLevel, Duration>,

    // Asymmetry: can skip levels going UP but not DOWN
    pub allow_skip_escalation: bool,   // true — Yellow → Red ok
    pub allow_skip_deescalation: bool,  // false — Red → Yellow → Green only
}
```

**Integration**:
- SafetyEscalator replaces standalone KillSwitch + CircuitBreaker + DrawdownTracker
- Feeds `safety_confidence` to TradingConfidence
- On escalation to Orange+: triggers DemotionTrigger in ExposureController

---

### Engineer 6: Decision Audit & Observability (Why Did We Do That?)

**Purpose**: Every quote cycle produces a structured decision record enabling replay, A/B testing, real-time monitoring, and automated postmortem.

**What exists today**: `AnalyticsLogger` (4 JSONL streams), `EdgeSnapshot`, `FillSignalSnapshot`, `PostMortemDump`

**What's missing**:
- Unified decision record (one struct per cycle, not 4 separate logs)
- Real-time confidence dashboard
- Decision replay ("what would have happened with params X?")
- Automated postmortem trigger (on safety escalation)
- A/B variant tracking (not just paper vs live, but param variants)

**New modules**:
```
src/market_maker/audit/
├── mod.rs              — DecisionAuditLog: unified per-cycle record
├── record.rs           — CycleDecisionRecord struct
├── replay.rs           — DecisionReplay: offline "what if" analysis
├── postmortem.rs       — AutomatedPostmortem: triggered by escalation
└── dashboard.rs        — ConfidenceDashboard: real-time metrics endpoint
```

**Cycle Decision Record**:
```rust
pub struct CycleDecisionRecord {
    pub timestamp: u64,
    pub cycle_number: u64,

    // Input state
    pub market_state: MarketStateSnapshot,
    pub model_scores: Vec<ModelScorecard>,
    pub confidence: TradingConfidence,
    pub safety_level: SafetyLevel,

    // Decision chain
    pub raw_spread_bps: f64,        // GLFT output
    pub signal_adjustments: Vec<SignalAdjustment>,
    pub risk_overlay: RiskOverlayEffect,
    pub exposure_scaling: f64,      // From confidence
    pub final_spread_bps: f64,      // What was actually quoted

    // Execution
    pub orders_placed: Vec<OrderAction>,
    pub orders_modified: Vec<OrderAction>,
    pub orders_cancelled: Vec<OrderAction>,
    pub orders_latched: usize,

    // Outcome (filled async)
    pub fills_this_cycle: Vec<FillRecord>,
    pub pnl_delta_bps: f64,
    pub position_after: f64,
}
```

---

## Implementation Roadmap

### Phase 0: Foundation (Week 1)
- Define `TradingConfidence` type in `src/market_maker/graduation/mod.rs`
- Thread it through QuoteEngine, ReconcileLoop, RiskOverlay
- Wire existing `warmup_progress` as initial `model_confidence` source
- **No behavior change** — just plumbing

### Phase 1: Safety Escalation (Week 2) — Engineer 5
- Implement `SafetyEscalator` with 5 levels
- Wrap existing KillSwitch as Level 4 (Kill)
- Wire DrawdownTracker thresholds to Yellow/Orange/Red
- Wire CircuitBreaker triggers to escalation
- Add min dwell times and asymmetric transitions
- **Behavior change**: Graduated response instead of binary kill

### Phase 2: Graduated Exposure (Week 3) — Engineer 2
- Implement `ExposureController` with promotion/demotion
- Replace static CapitalAwarePolicy tiers with dynamic levels
- Define promotion criteria (fills, edge, Brier, IR)
- Define demotion triggers (drawdown, safety events, model degradation)
- **Behavior change**: Can now start at MicroLive and ramp up

### Phase 3: Continuous Validation (Week 4) — Engineer 3
- Implement `ContinuousValidator` running every 20 fills
- Wire Brier score → model gating (not just computation)
- Wire signal MI → automatic retirement
- Feed `model_confidence` into TradingConfidence
- **Behavior change**: Models that degrade reduce exposure

### Phase 4: Decision Audit (Week 5) — Engineer 6
- Implement `CycleDecisionRecord` per quote cycle
- Unified logging (replaces 4 separate JSONL streams)
- Automated postmortem on safety escalation
- Confidence dashboard endpoint
- **Behavior change**: Full observability

### Phase 5: Shadow Trading (Week 6) — Engineer 1
- Implement parallel shadow mode
- Sim-to-real gap tracker
- Fill model calibration from live fills
- Feed `execution_confidence` into TradingConfidence
- **Behavior change**: Can measure paper prediction quality

### Phase 6: Exchange Hardening (Week 7+) — Engineer 4
- Enhanced paper simulator with realistic failures
- Chaos injection framework
- Order timeout detection
- Quota reservation
- **Behavior change**: Paper catches more bugs before live

---

## What This Enables

With this system, the paper→live graduation looks like:

```
Day 0: Start in Shadow mode (confidence=0)
  ↓ Paper models calibrate, CalibrationGate passes
  ↓ Confidence rises to 0.15

Day 0 + 30min: Auto-promote to MicroLive (5% size)
  ↓ Real fills start arriving
  ↓ Sim-to-real gap measured
  ↓ 50 fills with >60% positive edge
  ↓ Brier < 0.25, IR > 0.5
  ↓ Confidence rises to 0.35

Day 0 + 2h: Auto-promote to Ramping (25% size)
  ↓ Sustained performance, gap < 2 bps
  ↓ No safety escalations
  ↓ Confidence rises to 0.55

Day 0 + 6h: Auto-promote to Cautious (60% size)
  ↓ Full day of positive PnL
  ↓ All models Healthy
  ↓ Confidence rises to 0.75

Day 1: Full mode (100% size)
  ↓ Continuous monitoring
  ↓ Any degradation → instant demotion
  ↓ Recovery requires re-earning confidence
```

**The key insight**: No human needs to decide "we're ready to go live." The system continuously proves itself, and exposure tracks that proof. And if something goes wrong, exposure drops automatically — long before the kill switch fires.

---

## Files Modified (High-Level)

| Module | Changes |
|--------|---------|
| `graduation/` (NEW) | TradingConfidence, ExposureController, promotion/demotion |
| `safety/escalation.rs` (NEW) | SafetyEscalator with 5 levels |
| `validation/` (NEW) | ContinuousValidator, ModelScorecard |
| `shadow/` (NEW) | ShadowEngine, SimToRealTracker |
| `audit/` (NEW) | CycleDecisionRecord, automated postmortem |
| `simulation/enhanced_sim.rs` (NEW) | Realistic failure modes |
| `orchestrator/quote_engine.rs` | Thread TradingConfidence, apply size_multiplier |
| `orchestrator/reconcile.rs` | Apply exposure-based position limits |
| `risk/mod.rs` | SafetyEscalator replaces standalone monitors |
| `config/capacity.rs` | Dynamic tiers replace static CapitalAwarePolicy |
| `calibration/gate.rs` | Continuous evaluation, not one-time |
| `calibration/model_gating.rs` | Brier thresholds → automatic action |
