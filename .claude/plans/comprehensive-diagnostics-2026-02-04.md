# Comprehensive Market Maker Diagnostics Report
## Session: HYPE on hyna (HIP-3) | 2026-02-04

---

## Executive Summary

Your market maker ran for 53 minutes with **-$1.76 PnL** and **32 fills**. The system is fundamentally sound but has **5 critical issues** that prevent proper model validation and competitive edge:

| Issue | Severity | Impact | Fix Effort |
|-------|----------|--------|------------|
| Calibration infrastructure disconnected | **CRITICAL** | Cannot measure if models work | 2-3 days |
| Edge predictions 150x off from reality | **HIGH** | False confidence in edge | 1-2 days |
| Adverse selection not pre-emptively gated | **HIGH** | ~70 bps lost to toxic flow | 1 day |
| Pre-fill signals out of sync | **MEDIUM** | Stale toxicity predictions | 0.5 days |
| Quota warnings misunderstood | **LOW** | Operator confusion only | Documentation |

**Bottom Line**: The system quotes and executes, but you're flying blind on whether your models add value. Fix calibration first.

---

## Part 1: Calibration Infrastructure (CRITICAL)

### The Problem

Every 5-minute calibration report shows:
```
fill_ir=0.000, fill_samples=0, fill_brier=0.0000
as_ir=0.000, as_samples=0
lag_ir=0.000, lag_mi_decay=0.00000/day
```

**Zero samples collected despite 32 fills.** The entire calibration infrastructure exists but is disconnected from the trading pipeline.

### Root Cause Analysis

The `ModelCalibrationOrchestrator` has three calibrated model wrappers:
- `CalibratedFillModel` - tracks fill probability predictions
- `CalibratedAdverseSelection` - tracks AS predictions
- `CalibratedLagAnalyzer` - tracks lead-lag signal predictions

Each wrapper implements the pattern:
1. `predict()` → returns (value, prediction_id)
2. `record_outcome()` → links outcome to prediction_id
3. `information_ratio()` → computes IR from linked pairs

**BUT**: The production code never calls these methods. Instead:
- Fill probabilities computed via `BayesianFillModel.fill_probability()` directly
- AS computed via `AdverseSelectionEstimator` directly
- Outcomes recorded to 7+ components but NOT to `ModelCalibrationOrchestrator`

### What's Missing (Code Locations)

| Gap | File | What Should Happen |
|-----|------|-------------------|
| No prediction IDs in orders | `tracking/order_manager/types.rs` | Add `prediction_id: Option<u64>` to TrackedOrder |
| No predict() calls | `orchestrator/quote_engine.rs` | Call `model_calibration.fill_model.predict()` at quote gen |
| No calibration in FillState | `fills/processor.rs:513-575` | Add `model_calibration: &mut ModelCalibrationOrchestrator` |
| No record_outcome() calls | `fills/processor.rs:823-1056` | Call `record_outcome()` after existing analytics |
| No FillState wiring | `orchestrator/handlers.rs:329` | Wire `model_calibration` field |

### Implementation Plan

#### Step 1: Add Prediction IDs to Order Metadata

```rust
// In tracking/order_manager/types.rs (TrackedOrder struct)
pub struct TrackedOrder {
    pub oid: u64,
    pub cloid: Option<String>,
    // ADD THESE:
    pub fill_prediction_id: Option<u64>,
    pub as_prediction_id: Option<u64>,
    pub lag_prediction_id: Option<u64>,
    pub prediction_timestamp_ms: Option<u64>,
    // ... existing fields
}
```

#### Step 2: Make Predictions at Quote Generation

```rust
// In orchestrator/quote_engine.rs, when generating each ladder level:
let (fill_prob, fill_pred_id) = self.components.stochastic
    .model_calibration.fill_model.predict(depth_bps);

let (as_prob, as_pred_id) = self.components.stochastic
    .model_calibration.as_model.predict();

// Store in pending order metadata
pending_order.fill_prediction_id = Some(fill_pred_id);
pending_order.as_prediction_id = Some(as_pred_id);
```

#### Step 3: Add ModelCalibrationOrchestrator to FillState

```rust
// In fills/processor.rs (FillState struct)
pub struct FillState<'a> {
    pub position: &'a mut PositionTracker,
    pub orders: &'a mut OrderManager,
    pub adverse_selection: &'a mut AdverseSelectionEstimator,
    // ADD THIS:
    pub model_calibration: &'a mut ModelCalibrationOrchestrator,
    // ... existing fields
}
```

#### Step 4: Record Outcomes in Fill Processor

```rust
// In fills/processor.rs, after line 901 (after stochastic_controller.on_fill)

// === Calibration Outcome Recording ===
if let Some(order) = state.orders.get_order(fill.oid) {
    // Record fill probability outcome
    if let Some(fill_pred_id) = order.fill_prediction_id {
        state.model_calibration.fill_model.record_outcome(fill_pred_id, true);
    }

    // Record adverse selection outcome
    if let Some(as_pred_id) = order.as_prediction_id {
        let realized_as_bps = state.adverse_selection.realized_as_bps();
        state.model_calibration.as_model.record_outcome(as_pred_id, realized_as_bps);
    }
}
```

#### Step 5: Wire FillState in Handlers

```rust
// In orchestrator/handlers.rs (handle_user_fills, around line 329)
let mut fill_state = fills::FillState {
    position: &mut self.position,
    orders: &mut self.orders,
    adverse_selection: &mut self.tier1.adverse_selection,
    // ADD THIS:
    model_calibration: &mut self.stochastic.model_calibration,
    // ... existing fields
};
```

### Verification

After implementation, verify:
1. `fill_samples > 0` in calibration logs
2. `fill_ir` converges to some value (hopefully > 1.0)
3. Degradation warnings appear if IR < 1.0

---

## Part 2: Edge Prediction System (HIGH)

### The Problem

EdgeScatter logs show massive prediction errors:

| Predicted (bps) | Realized (bps) | Error |
|-----------------|----------------|-------|
| 0.11 | -14.88 | 135x |
| 0.21 | -15.14 | 72x |
| 0.07 | -7.21 | 103x |
| 0.09 | +6.36 | 71x |

**Predictions hover at 0.1 bps while realized swings ±15 bps.** Uncertainty (4 bps) is also far too tight.

### Root Cause Analysis

The edge prediction system has fundamental architectural flaws:

1. **GLFT predicts optimal QUOTE WIDTH, not realized edge**
   ```
   δ* = (1/γ) × ln(1 + γ/κ)  // This is spread to QUOTE, not profit to EXPECT
   ```

2. **AS prediction cannot forecast toxic flow arrival**
   - Predicts ~1-2 bps AS using static parameters
   - Reality: AS is 5-15 bps in volatile regimes, 15-30 bps in cascades
   - No mechanism to predict WHEN toxic flow arrives

3. **Empirical model learns on sparse, non-stationary data**
   - Only 9 bins (3×3 vol×toxicity)
   - Cold starts return (0.0 bps, 5.0 bps std)
   - Market structure changes weekly

4. **Funding model adds noise at wrong horizon**
   - Funding settles every 8 hours
   - Edge measured at 1-second horizon
   - These are orthogonal

5. **Uncertainty bounds 20-100x too tight**
   - Ensemble std: 1-2 bps
   - Realized variance: ±15 bps

### Recommended Solution

**Stop trying to predict 1-second realized edge.** Instead:

#### Option A: Track Fill Profitability by Context (Recommended)

```rust
// Track edge by (regime, position_bucket, ladder_level)
struct FillProfitabilityTracker {
    // Key: (regime_id, position_decile, depth_bucket)
    // Value: EWMA of realized edge
    buckets: HashMap<(u8, u8, u8), f64>,
}

// Use for parameter tuning, not edge gating
impl FillProfitabilityTracker {
    fn should_widen_spread(&self, regime: u8, position: f64, depth: f64) -> bool {
        let bucket_edge = self.get_bucket_edge(regime, position, depth);
        bucket_edge < -2.0  // Widen if losing > 2 bps consistently
    }
}
```

#### Option B: Use Edge Predictions Only for Reservation Shift

Keep the ensemble but limit its scope:
- Only use for quote price adjustment (reservation shift)
- Do NOT use for quoting/sizing decisions
- Remove edge-based gating entirely

#### Option C: Bayesian Edge with Proper Uncertainty

Replace point estimates with full posterior:
```rust
// Instead of: edge_mean = 0.1, edge_std = 4.0
// Use: P(edge > 0) with properly calibrated uncertainty

let edge_posterior = NormalInverseGamma::posterior(
    prior_mean: 0.0,
    prior_var: 100.0,  // Wide prior: ±10 bps
    observations: recent_realized_edges,
);

let p_positive = edge_posterior.cdf(0.0);
// Use p_positive for sizing, not point estimate
```

### Implementation Priority

1. **Immediate**: Widen uncertainty bounds 10x (trivial change)
2. **Week 1**: Implement fill profitability tracker by context
3. **Week 2**: Remove edge prediction from quoting decisions
4. **Week 3**: Use context-based tracker for parameter tuning

---

## Part 3: Adverse Selection System (HIGH)

### Current State

The AS system has **good architecture but critical gaps**:

#### What's Working ✓
- 5-signal pre-fill classifier (orderbook, flow, regime, changepoint, funding)
- Asymmetric spread widening (bid/ask treated separately)
- Post-fill AS measurement at 6 horizons
- DepthDecayAS calibration (AS₀, δ_char)

#### What's Broken ✗

| Issue | Location | Impact |
|-------|----------|--------|
| Signals out of sync | L2 vs quote_engine timing | 50-100ms stale |
| Trade flow loses magnitude | `quote_engine.rs:384-388` | Saturates quickly |
| No quote gating | `glft.rs:670-694` | Widens but still quotes |
| Signal weights hardcoded | `pre_fill_classifier.rs` | Not calibrated |
| No feedback loop | N/A | Predictions not validated |

### Critical Fix: Add Quote Gating

Currently the system WIDENS spreads when toxicity is high but STILL QUOTES. This means you still get filled on toxic flow, just at slightly better prices.

```rust
// In glft.rs, around line 670 (BEFORE spread widening)
// ADD THIS GATE:

const MAX_TOXICITY_TO_QUOTE: f64 = 0.6;  // ~1.8x multiplier threshold

// Gate toxic side entirely
if pre_fill_mult_bid > 2.0 {
    // Don't quote bids at all - toxic for buying
    return QuoteResult {
        bid: None,  // Skip bid
        ask: Some(ask_quote),
        reason: "bid_toxicity_gate",
    };
}

if pre_fill_mult_ask > 2.0 {
    // Don't quote asks at all - toxic for selling
    return QuoteResult {
        bid: Some(bid_quote),
        ask: None,  // Skip ask
        reason: "ask_toxicity_gate",
    };
}
```

### Critical Fix: Sync Signal Updates

Move orderbook update from L2 handler to quote engine:

```rust
// REMOVE from handlers.rs (L2 handler):
// self.tier1.pre_fill_classifier.update_orderbook(bid_depth, ask_depth);

// ADD to quote_engine.rs (at start of quote cycle):
let (bid_depth, ask_depth) = self.get_current_book_depth();
self.tier1.pre_fill_classifier.update_orderbook(bid_depth, ask_depth);
// Now all signals sampled at same timestamp
```

### Critical Fix: Preserve Trade Flow Magnitude

```rust
// In quote_engine.rs, replace lines 384-388:

// CURRENT (loses magnitude):
let flow_imb = self.estimator.flow_imbalance();  // [-1, 1]
let buy_volume = (1.0 + flow_imb) / 2.0;

// REPLACE WITH:
let (buy_vol, sell_vol) = self.estimator.recent_volumes();  // Actual volumes
let total_vol = buy_vol + sell_vol;
let flow_imb = if total_vol > 0.0 {
    (buy_vol - sell_vol) / total_vol
} else {
    0.0
};
// Scale threshold by total volume
let volume_scaled_threshold = self.config.flow_threshold * (1.0 + total_vol.ln().max(0.0));
```

### Feedback Loop: Validate Predictions

Add correlation tracking between predicted toxicity and realized AS:

```rust
// In processor.rs, after recording fill:
let predicted_toxicity = fill_snapshot.pre_fill_toxicity;
let realized_as = state.adverse_selection.realized_as_bps();

// Log for offline analysis
info!(
    target: "as_calibration",
    predicted_toxicity,
    realized_as,
    side = fill.side_str(),
    "[AS_Calibration] predicted vs realized"
);

// Online correlation tracking
state.as_calibration_tracker.record(predicted_toxicity, realized_as);
```

---

## Part 4: Quote-to-Fill Lifecycle (Implementation Guide)

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Quote Generation                                       │
│ File: orchestrator/quote_engine.rs                              │
│ ─────────────────────────────────────────────────────────────── │
│ 1. Check circuit breakers, risk limits                          │
│ 2. Update belief system (HMM, BOCPD, etc.)                      │
│ 3. Build MarketParams from estimators                           │
│ 4. → HERE: Call model_calibration.predict() for each model      │
│ 5. → HERE: Store prediction_ids in quote metadata               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ PHASE 2: Ladder Generation                                      │
│ File: strategy/ladder_strat.rs                                  │
│ ─────────────────────────────────────────────────────────────── │
│ 1. Compute gamma (risk aversion) from market conditions         │
│ 2. Select kappa (fill intensity) from estimators                │
│ 3. Generate GLFT-optimal depths                                 │
│ 4. Compute fill probability per level (BayesianFillModel)       │
│ 5. Entropy-optimize size allocation                             │
│ 6. Apply inventory skew                                         │
│ OUTPUT: Ladder { bids[], asks[] } with prediction_ids attached  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ PHASE 3: Order Placement                                        │
│ File: orchestrator/reconcile.rs, order_ops.rs                   │
│ ─────────────────────────────────────────────────────────────── │
│ 1. Compare target ladder to current orders                      │
│ 2. Cancel stale orders, place new orders                        │
│ 3. Create TrackedOrder with prediction_ids                      │
│ 4. Store in OrderManager                                        │
│ KEY: prediction_ids travel with order through cloid matching    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ PHASE 4: Fill Event                                             │
│ File: fills/mod.rs                                              │
│ ─────────────────────────────────────────────────────────────── │
│ WebSocket delivers UserFills                                    │
│ Create FillEvent with:                                          │
│   - tid, oid, cloid (for matching)                              │
│   - price, size, side                                           │
│   - mid_at_fill (for spread calculation)                        │
│ Match cloid → TrackedOrder → recover prediction_ids             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ PHASE 5: Fill Processing                                        │
│ File: fills/processor.rs                                        │
│ ─────────────────────────────────────────────────────────────── │
│ 1. Deduplication (by tid)                                       │
│ 2. Capture FillSignalSnapshot (all market state)                │
│ 3. Update: Position, Orders, AS, Queue, PnL, Estimators         │
│ 4. → HERE: Call model_calibration.record_outcome()              │
│ 5. Bayesian learning (fill model update)                        │
│ 6. Schedule markout measurements (500ms, 2s, 10s)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ PHASE 6: Outcome Resolution                                     │
│ File: tracking/calibration_wiring.rs                            │
│ ─────────────────────────────────────────────────────────────── │
│ PredictionOutcomeStore links prediction → outcome:              │
│   - fill_pred_id → filled=true                                  │
│   - as_pred_id → realized_as_bps                                │
│   - lag_pred_id → signal_led=true/false                         │
│ Compute Information Ratio from linked pairs                     │
│ Surface degradation if IR < 1.0                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Attachment Points

| Phase | File:Line | What to Add |
|-------|-----------|-------------|
| Quote Gen | `quote_engine.rs:~200` | `fill_model.predict(depth_bps)` |
| Order Create | `order_ops.rs:~145` | Store prediction_ids in TrackedOrder |
| Fill Match | `fills/mod.rs:~100` | Recover prediction_ids via cloid |
| Outcome | `processor.rs:~901` | `record_outcome(pred_id, result)` |

---

## Part 5: Quota Management (LOW - Documentation Only)

### Understanding "Quota Exhausted with Position"

This warning is **working as designed**, not a bug:

```
Quota exhausted with position - entering inventory-forcing mode
position=0.14-0.75, headroom_pct=8.8-9.0%, reduce_side=asks
```

**What's happening:**
1. Rate limit headroom drops below 10%
2. System has non-zero position (≥1% of max)
3. System enters inventory-forcing mode
4. Only quotes the side that reduces position

**Why it happens:**
- Quota depletes from API calls (~1-2/sec)
- Position accumulates from fills (~1-2/min)
- These converge within 30-60 minutes
- Small positions (0.14-0.75) are NORMAL market-making

**This is correct behavior:**
- Preserves remaining quota for risk reduction
- Prioritizes closing out position over new trades
- Prevents death spiral of quota exhaustion + stuck position

### If You Want to Reduce These Warnings

1. **Batch more aggressively** - fewer API calls per quote cycle
2. **Increase position threshold** - `position_threshold = 0.05 * max_position` instead of 0.01
3. **Widen headroom trigger** - activate at 15% instead of 10%

But note: these warnings are informational, not errors.

---

## Part 6: Implementation Priority & Timeline

### Phase 1: Critical Fixes (Week 1)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Wire calibration predict() | 4h | CRITICAL | quote_engine.rs |
| Wire calibration record_outcome() | 4h | CRITICAL | processor.rs, handlers.rs |
| Add AS quote gating | 2h | HIGH | glft.rs |
| Sync pre-fill signals | 2h | MEDIUM | handlers.rs, quote_engine.rs |

### Phase 2: Edge Prediction Overhaul (Week 2)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Widen uncertainty bounds 10x | 1h | HIGH | ensemble.rs |
| Implement fill profitability tracker | 8h | HIGH | New file |
| Remove edge-based quoting gates | 2h | MEDIUM | decision.rs |

### Phase 3: AS Refinement (Week 3)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Preserve trade flow magnitude | 2h | MEDIUM | quote_engine.rs |
| Add AS prediction validation | 4h | MEDIUM | processor.rs |
| Online weight learning | 8h | LOW | pre_fill_classifier.rs |

### Phase 4: Monitoring & Validation (Week 4)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Dashboard calibration panel | 4h | MEDIUM | dashboard.rs |
| Automated degradation alerts | 2h | MEDIUM | handlers.rs |
| Weekly calibration report | 4h | LOW | New script |

---

## Part 7: Success Criteria

After implementing these fixes, you should see:

### Calibration Metrics (Within 1 Day)
- `fill_samples > 0` (not stuck at 0)
- `fill_ir` converging to some value
- `as_samples > 0`
- Degradation warnings if IR < 1.0

### Edge Prediction (Within 1 Week)
- Uncertainty bounds ~40-50 bps (not 4 bps)
- Or: edge predictions removed from quoting decisions
- Fill profitability tracker populating

### Adverse Selection (Within 2 Weeks)
- Quote gating reducing toxic fills by 20-30%
- Realized AS < 1.5 bps (down from 2.18 bps)
- Pre-fill predictions correlating with outcomes

### Overall PnL (Within 1 Month)
- Positive daily PnL (currently -$1.76)
- Clear attribution: spread capture - AS - fees
- Model health dashboard showing all green

---

## Appendix: Code Snippets Ready for Copy-Paste

### A. TrackedOrder with Prediction IDs

```rust
// Add to tracking/order_manager/types.rs

#[derive(Debug, Clone)]
pub struct TrackedOrder {
    pub oid: u64,
    pub cloid: Option<String>,
    pub asset: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub state: OrderState,
    pub created_at: Instant,

    // Calibration prediction IDs
    pub fill_prediction_id: Option<u64>,
    pub as_prediction_id: Option<u64>,
    pub lag_prediction_id: Option<u64>,
    pub predicted_fill_prob: Option<f64>,
    pub predicted_as_bps: Option<f64>,
}
```

### B. Prediction Calls at Quote Generation

```rust
// Add to orchestrator/quote_engine.rs, in quote generation loop

// For each ladder level:
let depth_bps = level.depth_bps;

// Make calibrated predictions
let (fill_prob, fill_pred_id) = self.components.stochastic
    .model_calibration
    .fill_model
    .predict(depth_bps);

let (as_prob, as_pred_id) = self.components.stochastic
    .model_calibration
    .as_model
    .predict();

// Store in level metadata for later attachment to order
level.fill_prediction_id = Some(fill_pred_id);
level.as_prediction_id = Some(as_pred_id);
level.predicted_fill_prob = Some(fill_prob);
level.predicted_as_bps = Some(as_prob);
```

### C. FillState with ModelCalibrationOrchestrator

```rust
// Modify fills/processor.rs

pub struct FillState<'a> {
    pub position: &'a mut PositionTracker,
    pub orders: &'a mut OrderManager,
    pub adverse_selection: &'a mut AdverseSelectionEstimator,
    pub pnl_tracker: &'a mut PnLTracker,
    pub estimator: &'a mut ParameterEstimator,
    pub stochastic_controller: &'a mut StochasticController,
    pub learning: &'a mut LearningModule,
    // ADD THIS:
    pub model_calibration: &'a mut ModelCalibrationOrchestrator,
}
```

### D. Record Outcomes in Fill Processor

```rust
// Add to fills/processor.rs, after line 901

// === Calibration Outcome Recording ===
if let Some(order) = state.orders.get_order(fill.oid) {
    // Fill probability outcome (filled = true since we're processing a fill)
    if let Some(pred_id) = order.fill_prediction_id {
        state.model_calibration.fill_model.record_outcome(pred_id, true);
    }

    // Adverse selection outcome
    if let Some(pred_id) = order.as_prediction_id {
        let realized_as = state.adverse_selection.realized_as_bps();
        state.model_calibration.as_model.record_outcome(pred_id, realized_as);
    }

    // Lag outcome (if we have the signal)
    if let Some(pred_id) = order.lag_prediction_id {
        let signal_led = fill_snapshot.binance_led_hl;
        state.model_calibration.lag_model.record_outcome(pred_id, signal_led);
    }
}

tracing::debug!(
    fill_pred_id = ?order.fill_prediction_id,
    as_pred_id = ?order.as_prediction_id,
    "Calibration outcomes recorded"
);
```

### E. AS Quote Gating

```rust
// Add to strategy/glft.rs, around line 670 (before spread widening)

const MAX_TOXICITY_MULTIPLIER: f64 = 2.0;

// Gate extremely toxic sides - don't quote at all
if params.pre_fill_mult_bid > MAX_TOXICITY_MULTIPLIER {
    info!(
        toxicity_mult = params.pre_fill_mult_bid,
        "Gating bid quotes due to high toxicity"
    );
    half_spread_bid = f64::INFINITY;  // Effectively skip bid
}

if params.pre_fill_mult_ask > MAX_TOXICITY_MULTIPLIER {
    info!(
        toxicity_mult = params.pre_fill_mult_ask,
        "Gating ask quotes due to high toxicity"
    );
    half_spread_ask = f64::INFINITY;  // Effectively skip ask
}
```

---

## Summary

Your market maker has solid foundations but is missing the measurement infrastructure needed to validate and improve models. The immediate priority is:

1. **Wire calibration** - This is blocking everything else
2. **Gate toxic flow** - Stop the bleeding from AS
3. **Fix edge predictions** - Stop false confidence

Once calibration is working, you'll have data to make informed decisions about model improvements. Without it, you're optimizing blind.

**Expected improvement from these fixes: +20-50 bps daily edge capture.**
