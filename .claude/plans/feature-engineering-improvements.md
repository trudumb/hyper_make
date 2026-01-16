# Feature Engineering Improvements Plan

## Overview

Enhance the market maker's feature engineering based on IBM best practices and identified gaps in the current 7-layer feature pipeline.

## Current State Summary

**Strengths:**
- 7-layer feature pipeline (volume clock → volatility → flow → liquidity → regime → fair value → adaptive)
- Bayesian estimation with Gamma conjugacy for kappa
- Online Welford standardization per-signal
- Multi-scale volatility (5/20/100 tick half-lives)
- Regime-conditional calibration tracking

**Critical Gaps:**
1. No Mutual Information (MI) estimation - can't quantify feature value in bits
2. No feature interaction terms - only additive gamma weights
3. No cross-feature correlation matrix
4. No temporal/seasonal features (time-of-day, funding settlement proximity)
5. No lag analysis for cross-exchange signals
6. No automatic signal decay tracking

---

## Proposed Improvements

### Phase 1: Mutual Information Infrastructure (Foundation)

**Files to create/modify:**
- `src/market_maker/estimator/mutual_info.rs` (new)
- `src/market_maker/estimator/mod.rs` (add module)

**Implementation:**
```rust
// k-NN MI estimator using Kraskov algorithm
pub struct MutualInfoEstimator {
    k: usize,  // neighbors (typically 3-5)
    // Uses digamma function: ψ(k) - ψ(n) + d*ln(n)
}

pub fn estimate_mi(x: &[f64], y: &[f64], k: usize) -> f64;
pub fn estimate_conditional_mi(x: &[f64], y: &[f64], z: &[f64], k: usize) -> f64;
```

**Signals to evaluate:**
- Book imbalance → price direction (expect ~0.02-0.05 bits)
- Trade flow → fill probability (expect ~0.03-0.08 bits)
- Momentum → adverse selection (expect ~0.01-0.03 bits)

---

### Phase 2: Feature Interaction Terms

**Files to modify:**
- `src/market_maker/adaptive/shrinkage_gamma.rs`
- `src/market_maker/adaptive/config.rs`

**Current:** Additive weights only
```rust
weighted_sum = Σ wᵢ × zᵢ  // z = standardized signal
```

**Proposed:** Add interaction terms
```rust
pub enum GammaSignal {
    // Existing
    VolatilityRatio,
    JumpRatio,
    InventoryUtilization,
    HawkesIntensity,
    // NEW: Interactions
    VolatilityXMomentum,      // vol × |momentum| - wider in volatile + trending
    RegimeXInventory,          // regime_score × inventory_util - more risk-averse when exposed in cascade
    JumpXFlow,                 // jump_ratio × |flow_imbalance| - jumps + directional flow = toxic
}
```

**Rationale:** Non-linear effects are real:
- High volatility alone is fine; high volatility + momentum is dangerous
- Large inventory alone is manageable; large inventory + regime shift is not

---

### Phase 3: Temporal Feature Engineering

**Files to create/modify:**
- `src/market_maker/estimator/temporal.rs` (new)
- `src/market_maker/strategy/params/mod.rs`

**Features to add:**

1. **Time-of-day encoding** (cyclic):
```rust
pub struct TimeFeatures {
    hour_sin: f64,  // sin(2π × hour/24)
    hour_cos: f64,  // cos(2π × hour/24)
    funding_proximity: f64,  // 0-1, how close to 8h settlement
}
```

2. **Multi-horizon momentum**:
```rust
pub struct MultiScaleMomentum {
    momentum_1s: f64,
    momentum_10s: f64,
    momentum_60s: f64,
    momentum_300s: f64,  // NEW: 5-min scale
    trend_agreement: f64,  // correlation across scales
}
```

3. **Funding rate dynamics**:
```rust
pub struct FundingFeatures {
    funding_rate_8h: f64,        // current rate
    funding_rate_delta: f64,     // change from last period
    funding_imbalance: f64,      // predicted flow direction
}
```

---

### Phase 4: Cross-Feature Correlation Tracking

**Files to modify:**
- `src/market_maker/estimator/covariance.rs` (expand existing)
- `src/market_maker/adaptive/mod.rs`

**Current:** Only (κ, σ) covariance tracked

**Proposed:** Full feature correlation matrix
```rust
pub struct FeatureCorrelationTracker {
    // Track 10x10 correlation matrix
    features: Vec<&'static str>,  // ["kappa", "sigma", "momentum", "flow", ...]
    correlation_matrix: Vec<Vec<f64>>,
    ewma_cov: Vec<Vec<f64>>,

    // Detect multicollinearity
    pub fn condition_number(&self) -> f64;
    pub fn variance_inflation_factors(&self) -> Vec<f64>;
}
```

**Use cases:**
- Detect when signals become redundant (correlation > 0.8)
- Weight decorrelated signals higher
- Alert when new regime causes correlation breakdown

---

### Phase 5: Lag Analysis for Cross-Exchange Signals

**Files to create:**
- `src/market_maker/estimator/lag_analysis.rs` (new)

**Implementation:**
```rust
pub struct LagAnalyzer {
    candidate_lags_ms: Vec<i64>,  // [-500, -200, -100, 0, 100, 200, 500]
    signal_buffer: RingBuffer<(i64, f64)>,  // (timestamp, signal_value)
    target_buffer: RingBuffer<(i64, f64)>,  // (timestamp, target_value)
}

impl LagAnalyzer {
    // Find lag that maximizes MI
    pub fn optimal_lag(&self) -> (i64, f64);  // (lag_ms, mi_at_lag)

    // Cross-correlation function
    pub fn ccf(&self, max_lag_ms: i64) -> Vec<(i64, f64)>;
}
```

**Application:** Binance → Hyperliquid lead-lag timing
- Currently assumed 50-500ms, but not measured
- Optimal lag varies by volatility regime
- Track decay as arbitrageurs close gap

---

### Phase 6: Signal Decay Monitoring

**Files to create:**
- `src/market_maker/tracking/signal_decay.rs` (new)

**Implementation:**
```rust
pub struct SignalDecayTracker {
    signal_name: String,
    daily_mi: RingBuffer<(Date, f64)>,  // Historical MI values
}

impl SignalDecayTracker {
    pub fn trend(&self) -> f64;  // Slope of MI over time
    pub fn half_life_days(&self) -> Option<f64>;  // Time to 50% MI decay
    pub fn is_stale(&self, threshold_days: f64) -> bool;
}

pub struct SignalDecayReport {
    signals: Vec<SignalDecayTracker>,
    pub fn generate_report(&self) -> String;  // Weekly signal health
}
```

**Alert thresholds:**
- Half-life < 30 days → Warning
- Half-life < 7 days → Critical (remove signal)
- MI dropped > 50% in past week → Investigate

---

### Phase 7: Automated Feature Validation

**Files to modify:**
- `src/market_maker/estimator/mod.rs`
- `src/market_maker/infra/data_quality.rs`

**Add feature validation layer:**
```rust
pub struct FeatureValidator {
    bounds: HashMap<String, (f64, f64)>,  // (min, max) per feature
    staleness_threshold_ms: u64,
}

impl FeatureValidator {
    pub fn validate(&self, name: &str, value: f64, timestamp: i64) -> FeatureStatus;
}

pub enum FeatureStatus {
    Valid,
    OutOfBounds { value: f64, bounds: (f64, f64) },
    Stale { age_ms: u64 },
    NaN,
}
```

**Auto-learned bounds:**
- Track rolling 99.9th percentile
- Flag values outside 3σ from historical mean
- Graceful degradation: use fallback value, widen spreads

---

## Verification Plan

1. **Unit tests for MI estimator:**
   - Test with known distributions (Gaussian → MI = -0.5 log(1-ρ²))
   - Verify convergence with sample size

2. **Backtest feature interactions:**
   - Compare IR with/without interaction terms
   - Target: IR improvement > 5%

3. **Live A/B test temporal features:**
   - Paper trade with time-of-day features enabled
   - Measure fill rate around funding settlements

4. **Signal decay baseline:**
   - Compute current MI for all signals
   - Establish decay rate baselines

---

## File Summary

| File | Action | Priority |
|------|--------|----------|
| `src/market_maker/estimator/mutual_info.rs` | Create | P0 |
| `src/market_maker/estimator/temporal.rs` | Create | P1 |
| `src/market_maker/estimator/lag_analysis.rs` | Create | P1 |
| `src/market_maker/tracking/signal_decay.rs` | Create | P2 |
| `src/market_maker/adaptive/shrinkage_gamma.rs` | Modify | P1 |
| `src/market_maker/estimator/covariance.rs` | Expand | P2 |
| `src/market_maker/infra/data_quality.rs` | Expand | P2 |

---

## User Decisions

- **MI estimation:** k-NN Kraskov algorithm (more accurate, worth the compute cost)
- **Feature interactions:** All three (Vol×Momentum, Regime×Inventory, Jump×Flow)
- **Cross-exchange:** Design interface now, Binance feed integration planned for later

---

## Implementation Order

1. **Phase 1: MI Infrastructure** - Foundation for measuring all other improvements
2. **Phase 2: Feature Interactions** - Immediate gamma improvement
3. **Phase 3: Temporal Features** - Funding settlement edge
4. **Phase 4: Correlation Tracking** - Detect redundant signals
5. **Phase 5: Lag Analysis** - Interface ready for Binance integration
6. **Phase 6: Signal Decay** - Long-term alpha monitoring
7. **Phase 7: Feature Validation** - Production safety
