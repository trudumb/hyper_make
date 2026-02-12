# Fill Handling & Edge Measurement System - Architecture Exploration

## Overview
Deep architectural analysis of how fills are processed, adverse selection is computed, and edge measurements flow through the system for RL training and model calibration.

---

## 1. FILL HANDLING PATH (`handlers.rs`)

### Entry Point: `handle_user_fills()` → Lines 700-810+

**File**: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/orchestrator/handlers.rs`

#### 1.1 Fill Arrival & AS Computation (Lines 700-730)
```rust
// Line 700-710: For each fill in batch:
let fill_price: f64 = fill.px.parse().unwrap_or(0.0);
let is_buy = fill.side == "B" || fill.side.to_lowercase() == "buy";

// Line 709-714: Realized AS computation
let direction = if is_buy { 1.0 } else { -1.0 };
let as_realized = (self.latest_mid - fill_price) * direction / fill_price;

// Line 718: Depth from mid (uses latest_mid as proxy for mid_at_placement)
let depth_from_mid = (fill_price - self.latest_mid).abs() / self.latest_mid;
```

**Key Issue Identified**: 
- `depth_from_mid` uses `self.latest_mid` instead of actual `mid_at_placement`
- This is a LIMITATION: We don't currently track the mid price when order was placed
- See Section 2 for order tracking architecture

#### 1.2 Kappa Learning from Own Fills (Lines 735-745)
```rust
// Line 739-745: estimator learns own fill intensity
self.estimator.on_own_fill(
    fill.time,           // timestamp_ms
    fill_price,          // placement_price (best approximation - WRONG!)
    fill_price,          // fill_price
    fill_size,
    is_buy,
);
```

**Design Note**: Uses `fill_price` as both placement and fill price (conservative, biases toward zero depth).

#### 1.3 Pending Fill Outcome Queue (Lines 747-757)
```rust
// Line 750-757: Queue for 5-second markout
self.infra.pending_fill_outcomes.push_back(
    crate::market_maker::fills::PendingFillOutcome {
        timestamp_ms: fill.time,
        fill_price,
        is_buy,
        mid_at_fill: self.latest_mid,  // ← Captured at fill time
    },
);
```

**Data Structure**:
```rust
pub struct PendingFillOutcome {
    pub timestamp_ms: u64,
    pub fill_price: f64,
    pub is_buy: bool,
    pub mid_at_fill: f64,  // ← Critical: mid at fill arrival
}
```

Location: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/fills/mod.rs:57-66`

#### 1.4 ModelGating Updates (Lines 759-776)
```rust
// Line 762-767: Signal integrator on_fill() for kappa IR tracking
self.stochastic.signal_integrator.on_fill(
    fill.time,
    fill_price,
    fill_size,
    self.latest_mid,
);

// Line 772-776: Update AS prediction outcome
let predicted_as_prob = self.stochastic.theoretical_edge.bayesian_adverse();
let was_adverse = as_realized * 10000.0 > 3.0;  // 3 bps threshold
self.stochastic.signal_integrator.update_as_prediction(predicted_as_prob, was_adverse);
```

#### 1.5 EDGE TRACKING BLOCK (Lines 778-809) ⭐ CRITICAL

```rust
// Line 779-798: EdgeSnapshot creation
const MAKER_FEE_BPS: f64 = 1.5;
let depth_bps = depth_from_mid * 10_000.0;
let as_realized_bps = as_realized * 10_000.0;
let predicted_as_bps = self.estimator.total_as_bps();  // ← Uses AS decomposition

let snap = EdgeSnapshot {
    timestamp_ns: fill.time * 1_000_000,
    predicted_spread_bps: depth_bps,      // ← Spread capture (order depth)
    realized_spread_bps: depth_bps,       // ← Same as predicted (no ex-post adjustment)
    predicted_as_bps,                     // ← From estimator (~0-5 bps)
    realized_as_bps: as_realized_bps,     // ← Actual mid move post-fill
    fee_bps: MAKER_FEE_BPS,
    predicted_edge_bps: depth_bps - predicted_as_bps - MAKER_FEE_BPS,
    realized_edge_bps: depth_bps - as_realized_bps - MAKER_FEE_BPS,
    gross_edge_bps: depth_bps - as_realized_bps,  // Pre-fee edge
};

// Line 799-803: Edge recording
self.tier2.edge_tracker.add_snapshot(snap.clone());
let fill_pnl_bps = snap.realized_edge_bps;
self.live_analytics.record_fill(fill_pnl_bps, Some(&snap));

// Line 808: Phase 5 - QuoteOutcomeTracker resolution
let _ = self.quote_outcome_tracker.on_fill(is_buy, snap.realized_edge_bps);
```

---

## 2. ADVERSE SELECTION (AS) MEASUREMENT

### 2.1 Real-Time AS (at fill): Lines 709-714
Immediate AS computed using `latest_mid` vs `fill_price`:
```
as_realized = (mid_after - fill_price) × direction / fill_price
```
- Positive AS = loss (price moved against us)
- Captured in `PendingFillOutcome.mid_at_fill` for 5s markout

### 2.2 Five-Second Markout: Lines 223-314

**File**: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/orchestrator/handlers.rs:223-314`

```rust
fn check_pending_fill_outcomes(&mut self, now_ms: u64) {
    const OUTCOME_DELAY_MS: u64 = 5_000;
    const ADVERSE_NOISE_MULT: f64 = 2.0;
    const MIN_ADVERSE_THRESHOLD_BPS: f64 = 1.0;

    let sigma_bps = self.estimator.sigma() * 10_000.0;
    let adverse_threshold_bps = 
        (ADVERSE_NOISE_MULT * sigma_bps * MARKOUT_SECONDS.sqrt()).max(MIN_ADVERSE_THRESHOLD_BPS);

    // Line 236-255: Check all pending outcomes
    while let Some(front) = self.infra.pending_fill_outcomes.front() {
        if now_ms.saturating_sub(front.timestamp_ms) < OUTCOME_DELAY_MS {
            break;
        }

        let pending = self.infra.pending_fill_outcomes.pop_front().unwrap();
        
        // Compute 5s markout
        let mid_change_bps = 
            ((self.latest_mid - pending.mid_at_fill) / pending.mid_at_fill) * 10_000.0;
        let was_adverse = if pending.is_buy {
            mid_change_bps < -adverse_threshold_bps  // Mid dropped -> adverse for buy
        } else {
            mid_change_bps > adverse_threshold_bps   // Mid rose -> adverse for sell
        };

        // Line 259-270: Feed outcome to classifiers
        self.tier1.pre_fill_classifier.record_outcome(
            pending.is_buy,
            was_adverse,
            Some(magnitude_bps),
        );

        // Line 272-295: Feed to Bayesian parameter learner
        let as_realized = (self.latest_mid - pending.fill_price) * direction / pending.fill_price;
        let as_realized_bps = as_realized * 10_000.0;
        let predicted_as_bps_val = self.estimator.total_as_bps();

        let outcome = FillOutcome {
            was_informed: was_adverse,
            realized_as_bps: as_realized_bps,
            fill_distance_bps,
            realized_pnl: fill_pnl,
            predicted_as_bps: Some(predicted_as_bps_val),
        };
        self.stochastic.learned_params.observe_fill(&outcome);
    }
}
```

### 2.3 AS Decomposition (`total_as_bps()`)

**File**: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/estimator/as_decomposition.rs:457-479`

```rust
/// Total AS (immediate horizon, typically 1s)
pub fn total_as_bps(&self) -> f64 {
    self.as_at_horizon(self.config.immediate_horizon_idx)  // Uses 1s horizon
}

/// Permanent AS (long horizon, typically 5min)
pub fn permanent_as_bps(&self) -> f64 {
    self.as_at_horizon(self.config.permanent_horizon_idx)
}

/// Temporary AS (immediate - permanent)
pub fn temporary_as_bps(&self) -> f64 {
    let immediate = self.total_as_bps();
    let permanent = self.permanent_as_bps();
    (immediate - permanent).max(0.0)
}
```

The decomposition breaks AS into:
- **Total** (~1s horizon): immediate adverse selection
- **Permanent**: long-term component (reflects informed flow)
- **Temporary**: noise/liquidity component
- **Timing**: residual from execution delay

---

## 3. EDGESNAPSHOT STRUCT & CONSUMERS

**File**: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/analytics/edge_metrics.rs:1-27`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSnapshot {
    pub timestamp_ns: u64,
    pub predicted_spread_bps: f64,        // Order depth (from placement)
    pub realized_spread_bps: f64,         // Actual spread captured
    pub predicted_as_bps: f64,            // Pre-fill AS estimate
    pub realized_as_bps: f64,             // Post-fill AS (from 5s markout)
    pub fee_bps: f64,                     // Maker fee 1.5 bps
    pub predicted_edge_bps: f64,          // spread - AS - fees (predicted)
    pub realized_edge_bps: f64,           // spread - AS - fees (realized)
    pub gross_edge_bps: f64,              // spread - AS (pre-fee)
}
```

### Consumers of EdgeSnapshot:

1. **EdgeTracker** (lines 34-198 of edge_metrics.rs)
   - Tracks mean realized/predicted edge
   - Computes t-test for positive edge significance
   - Provides `last_realized_edge_bps()` for RL reward signal
   - Usage: `snap.realized_edge_bps` → TD reward for Quote actions

2. **LiveAnalytics** (line 803 in handlers.rs)
   - Records fill for Sharpe tracking
   - `record_fill(fill_pnl_bps, Some(&snap))`

3. **QuoteOutcomeTracker** (line 808 in handlers.rs)
   - Resolves pending quotes as filled
   - `on_fill(is_buy, snap.realized_edge_bps)` 
   - Tracks P(fill | spread) empirically

4. **RLEdgeModel** (indirectly)
   - Edge metrics inform confidence thresholding
   - Model gating uses edge health for weight adjustment

---

## 4. QUOTEOUTCOMETRACKER (Phase 5 Addition)

**File**: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/learning/quote_outcome.rs`

Purpose: Unbiased edge estimation accounting for survivorship bias.

### Key Components:

```rust
/// Pending quote awaiting outcome
pub struct PendingQuote {
    pub timestamp_ms: u64,
    pub half_spread_bps: f64,
    pub is_bid: bool,
    pub state: CompactMarketState,  // Market state at quote time
}

/// Outcome tracking
pub enum QuoteOutcome {
    Filled { edge_bps: f64, state, spread_bps },
    Expired { state, spread_bps, subsequent_move_bps },
}

/// Main tracker
pub struct QuoteOutcomeTracker {
    pending_quotes: VecDeque<PendingQuote>,
    outcome_log: VecDeque<QuoteOutcome>,
    fill_rate: BinnedFillRate,  // P(fill | spread_bin)
    last_mid: f64,
}
```

### Usage in Handlers:

```rust
// Line 808: Register fill outcome
let _ = self.quote_outcome_tracker.on_fill(is_buy, snap.realized_edge_bps);

// Expiry is called elsewhere (quote_engine likely)
quote_outcome_tracker.expire_old_quotes(now_ms);
```

### Output Statistics:

```rust
pub struct OutcomeStats {
    pub n_total: usize,              // Total quotes
    pub n_filled: usize,             // Fills
    pub n_expired: usize,            // Unfilled
    pub fill_rate: f64,              // P(fill)
    pub mean_edge_given_fill: f64,   // E[edge | fill]
    pub expected_edge: f64,          // P(fill) × E[edge | fill] ← Unbiased!
}
```

---

## 5. ORDER TRACKING & PLACEMENT MID

### 5.1 TrackedOrder Structure

**File**: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/tracking/order_manager/types.rs:202-235`

```rust
pub struct TrackedOrder {
    pub oid: u64,
    pub cloid: Option<String>,          // ← Client Order ID for deterministic tracking
    pub side: Side,
    pub price: f64,                     // ← Placement price (ORDER DEPTH = |placement - mid_at_placement|)
    pub size: f64,
    pub filled: f64,
    pub state: OrderState,
    pub placed_at: Instant,
    pub state_changed_at: Instant,
    pub last_fill_at: Option<Instant>,
    pub fill_tids: SmallVec<[u64; 4]>,
    pub fill_prediction_id: Option<u64>,
    pub as_prediction_id: Option<u64>,
    pub depth_bps: Option<f64>,         // ← Cached depth at placement time!
}
```

**Gap Identified**: 
- `depth_bps` is cached but `mid_at_placement` is NOT stored
- Cannot reconstruct placement mid = placement_price ± depth_bps

### 5.2 Fill Event with CLOID Support

**File**: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/fills/mod.rs:68-149`

```rust
pub struct FillEvent {
    pub tid: u64,
    pub oid: u64,
    pub cloid: Option<String>,                    // ← For deterministic matching
    pub size: f64,
    pub price: f64,
    pub is_buy: bool,
    pub mid_at_fill: f64,                         // ← Captured!
    pub placement_price: Option<f64>,             // ← When tracked
    pub timestamp: Instant,
    pub asset: String,
}

pub fn depth_bps(&self) -> f64 {
    let reference_price = self.placement_price.unwrap_or(self.price);
    if self.mid_at_fill > 0.0 {
        ((reference_price - self.mid_at_fill).abs() / self.mid_at_fill) * 10_000.0
    } else {
        0.0
    }
}
```

---

## 6. HARDCODED HL-NATIVE FLOW FEATURES

### Location: handlers.rs Lines 554-559

```rust
let flow_features = HL_FlowFeatures {
    // ...
    imbalance_30s: 0.0,         // ← HARDCODED (not tracked on HL)
    intensity: self.tier2.hawkes.intensity_ratio(),
    avg_buy_size: 0.0,          // ← HARDCODED (no separate tracking)
    avg_sell_size: 0.0,         // ← HARDCODED
    size_ratio: 0.0,            // ← HARDCODED
};
```

### Why Hardcoded:

Hyperliquid provides:
- Real-time L2 book (bid/ask depth) → Used for `kappa` estimation
- Trade stream → Used for `intensity`, `vpin`
- NO rolling volume windows per side → Cannot compute `avg_buy_size`, `avg_sell_size`

### Where Used:

**File**: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/strategy/signal_integration.rs:890-901`

```rust
// Lines 894-900: Fallback HL-native flow blending
let flow_dir = self.latest_hl_flow.imbalance_5s * HL_IMBALANCE_SHORT_WEIGHT
             + self.latest_hl_flow.imbalance_30s * HL_IMBALANCE_MED_WEIGHT;
let fallback_cap = self.config.max_lead_lag_skew_bps * HL_NATIVE_SKEW_FRACTION;
signals.combined_skew_bps = (flow_dir * fallback_cap).clamp(-fallback_cap, fallback_cap);
```

Since `imbalance_30s` is always 0.0, this effectively uses only `imbalance_5s` (hawkes flow).

---

## 7. INFORMED FLOW SIGNAL INTEGRATION

### Model Gating Weighting

**File**: `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/strategy/signal_integration.rs`

```rust
// Lines 912-913: Per-signal gating
informed_flow_gating_weight: signals.informed_flow_gating_weight,
informed_flow_spread_mult: signals.informed_flow_spread_mult,
```

The `informed_flow_spread_mult` is computed from:
1. Predicted `p_informed` (probability flow is informed)
2. InformedFlowAdjustment config
3. ModelGating weight (caps at 5% floor internally via `graduated_weight()`)

---

## 8. PREDICTED_AS_BPS COMPUTATION

### Two Paths:

**Path 1: Fast-path in handlers.rs (Line 787)**
```rust
let predicted_as_bps = self.estimator.total_as_bps();
```
- Calls `ASDecomposition::total_as_bps()` → immediate horizon (1s)
- Returns ~0-5 bps estimate

**Path 2: Calibration markout (Line 283)**
```rust
let predicted_as_bps_val = self.estimator.total_as_bps();
```
- Same estimator, sampled at 5s markout
- Used for Bayesian parameter learning

### Components:

1. **ASDecomposition** (5 estimators, one per horizon)
   - Tracks permanent, temporary, timing components
   - Uses rolling window of realized AS measurements
   - Updates via `FillOutcome` from 5s markout

2. **Pre-Fill Classifier** (Bayesian toxicity model)
   - Input: book imbalance, flow, volatility
   - Output: P(informed), used as prior
   - Calibrated against 5s markout outcomes

---

## 9. RL REWARD SIGNAL CHAIN

### Flow from EdgeSnapshot → RL Q-Update:

1. **EdgeSnapshot.realized_edge_bps** (handlers.rs:808)
   ```
   realized_edge = spread_captured - AS - fees
   ```

2. **EdgeTracker.last_realized_edge_bps()** 
   → Used as TD reward for Quote action
   → File: `src/market_maker/learning/rl_edge_model.rs` (not shown but referenced)

3. **Factors in reward**:
   - Positive = profitable quote
   - Negative = losing quote
   - Corrected for fees (makerfee 1.5 bps)
   - Corrected for adverse selection post-fill

4. **Quote vs NoQuote decision**:
   - Quote if E[edge | state, action] > threshold
   - Threshold gated by model confidence + risk assessment

---

## 10. KEY GAPS & DESIGN IMPLICATIONS

| Gap | Current | Impact | Redesign Opportunity |
|-----|---------|--------|----------------------|
| **Mid at placement** | Uses `latest_mid` (proxy) | Underestimates depth when mid has moved | Track `mid_at_placement` per order |
| **Placement mid in EdgeSnapshot** | Not captured | Cannot refine edge estimate ex-post | Add `mid_at_placement` field |
| **HL-native 30s imbalance** | Hardcoded 0.0 | No signal degradation when Binance unavailable | Could track manually from trades |
| **Size imbalance** | Hardcoded 0.0 | Cannot skew on size asymmetry | Compute EWMA buy/sell sizes from stream |
| **Quote outcome matching** | Last-match-same-side | Race condition with partial fills | Use CLOID for deterministic matching |

---

## 11. SUMMARY: DATA FLOW DIAGRAM

```
Order Placement
    |
    v
Track: TrackedOrder {cloid, price, placed_at, depth_bps}
    |
    +---> WebSocket: UserFills message
         |
         v
    Fill Arrival (Line 700):
    - Compute: as_realized = (latest_mid - fill_price) / fill_price
    - Compute: depth_from_mid = |fill_price - latest_mid| / latest_mid  [USES PROXY MID]
    - Queue PendingFillOutcome {mid_at_fill, fill_price, is_buy}
         |
         +---> Immediate: EdgeSnapshot {depth_bps, predicted_as_bps, realized_as_bps (instant)}
         |     |
         |     +---> EdgeTracker.add_snapshot()
         |     +---> live_analytics.record_fill()
         |     +---> QuoteOutcomeTracker.on_fill(realized_edge_bps) ← RL Reward
         |
         v
    5 Second Markout (Line 223-314):
    - Compute: realized_as = (latest_mid - mid_at_fill) vs threshold
    - Feed outcome to: pre_fill_classifier, enhanced_classifier
    - Update: learned_params.observe_fill()
    - Calibrate: AS estimator, kappa estimator
         |
         v
    Model Gating:
    - Update IR weight for: lead_lag, informed_flow, regime_kappa
    - Adjust next quote spreads based on model confidence
```

---

## 12. FILES INVOLVED

Core files for redesign:
- `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/orchestrator/handlers.rs` (700-810)
- `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/analytics/edge_metrics.rs`
- `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/learning/quote_outcome.rs`
- `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/fills/mod.rs`
- `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/tracking/order_manager/types.rs`
- `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/estimator/as_decomposition.rs`
- `/mnt/c/Users/17808/Desktop/hyper_make/src/market_maker/strategy/signal_integration.rs`

