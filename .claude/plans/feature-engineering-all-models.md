# Feature Engineering Plan: All 10 Prediction Models

**Date**: 2026-02-05
**Objective**: Transform all models from IR ≈ 0 to IR > 1.0 through systematic feature engineering

## Current State (from 1-hour validation run)

| Model | IR | Resolution | Concentration | Root Cause |
|-------|-----|------------|---------------|------------|
| InformedFlow | 0.02 | 0.0036 | 54% | Raw features not discriminative |
| PreFillToxicity | 0.01 | 0.0017 | 82% | Sigmoid saturation |
| EnhancedTox | 0.04 | 0.0099 | 31% | Best baseline, needs scaling |
| RegimeHighVol | 0.00 | 0.0000 | 88% | HMM thresholds miscalibrated |
| Momentum | 0.04 | 0.0089 | 26% | Single-scale, needs multi-timeframe |
| BuyPressure | 0.02 | 0.0037 | 37% | Simple ratio, needs flow clustering |
| LeadLag | 0.00 | 0.0006 | 5800% | Unstable lag, no causality gate |
| CV_Agreement | 0.00 | 0.0000 | 100% | conf=0.00 always, gating too strict |
| CV_Toxicity | 0.00 | 0.0000 | 99% | Same as above |
| CV_Direction | 0.00 | 0.0000 | 100% | Same as above |

---

## Phase 1: Foundation Fix (AdaptiveStandardizer)

**Problem**: All models suffer from features that cluster around the base rate after sigmoid transformation.

**Solution**: Implement `AdaptiveStandardizer` that z-score normalizes features before sigmoid:

```rust
pub struct AdaptiveStandardizer {
    running_mean: f64,
    running_var: f64,
    n: usize,
    alpha: f64,  // EMA decay for online updates
}

impl AdaptiveStandardizer {
    pub fn standardize(&mut self, raw: f64) -> f64 {
        // Update running statistics
        self.n += 1;
        let delta = raw - self.running_mean;
        self.running_mean += self.alpha * delta;
        self.running_var = (1.0 - self.alpha) * self.running_var + self.alpha * delta * delta;

        // Z-score normalize
        let std = self.running_var.sqrt().max(1e-6);
        let z = (raw - self.running_mean) / std;

        // Clamp to prevent extreme values
        z.clamp(-3.0, 3.0)
    }
}
```

**Files to modify**:
- `src/market_maker/adaptive/standardizer.rs` - Enhance existing implementation
- `src/market_maker/adverse_selection/pre_fill_classifier.rs` - Use standardizer
- `src/market_maker/adverse_selection/enhanced_classifier.rs` - Use standardizer

---

## Phase 2: Model-Specific Feature Engineering

### 2.1 InformedFlow (IR: 0.02 → target 1.2+)

**Current features**:
- `size` - trade size
- `inter_arrival_ms` - time between trades
- `price_impact_bps` - realized price impact
- `book_imbalance` - bid/ask depth ratio

**New features to add**:

```rust
pub struct EnhancedTradeFeatures {
    // Existing
    pub size: f64,
    pub inter_arrival_ms: u64,
    pub price_impact_bps: f64,
    pub book_imbalance: f64,

    // NEW: Clustering detection
    pub trade_cluster_intensity: f64,    // Trades per second in last 5s
    pub size_surprise: f64,              // z-score of size vs recent distribution

    // NEW: Cross-side aggression
    pub aggression_ratio: f64,           // Taker volume / Maker volume
    pub sweep_depth: f64,                // How many levels the trade swept

    // NEW: Temporal patterns
    pub funding_proximity: f64,          // Time to funding (8h cycle)
    pub session_time_feature: f64,       // Asian/Euro/US session encoding

    // NEW: Information asymmetry
    pub spread_normalized_size: f64,     // Size / (spread_bps * typical_size)
    pub kyle_lambda: f64,                // Price impact per unit volume
}
```

**Implementation**:
```rust
// Trade cluster intensity: Poisson rate of arrivals
let arrivals_5s = trades.iter().filter(|t| t.ts > now - 5000).count();
let cluster_intensity = arrivals_5s as f64 / 5.0;

// Size surprise: How unusual is this trade?
let size_zscore = standardizer.standardize(trade.size);

// Kyle's lambda: Realized price impact regression
let kyle_lambda = price_moves.iter().zip(volumes.iter())
    .map(|(dp, v)| dp.abs() / v.max(1e-6))
    .sum::<f64>() / n as f64;
```

**Files**: `src/market_maker/estimator/informed_flow.rs`

---

### 2.2 PreFillToxicity (IR: 0.01 → target 1.2+)

**Current problem**: 82% concentration - sigmoid saturation

**Fix**: Replace linear combination with learned weights + standardization

```rust
pub struct PreFillFeatures {
    // Standardized inputs (z-scores, not raw)
    pub book_imbalance_z: f64,
    pub flow_imbalance_z: f64,
    pub spread_z: f64,
    pub volatility_z: f64,
    pub cluster_intensity_z: f64,

    // NEW: Interaction terms
    pub imbalance_x_volatility: f64,     // High vol + imbalance = toxic
    pub spread_x_flow: f64,              // Wide spread + one-sided flow = toxic

    // NEW: Regime-conditional
    pub regime_weight: f64,              // Scale by regime uncertainty
}
```

**Key change**: The learned weights [0.32, 0.27, 0.23, 0.09, 0.09] are static. Make them regime-dependent:

```rust
fn get_regime_weights(&self, regime: usize) -> [f64; 5] {
    match regime {
        0 => [0.4, 0.3, 0.15, 0.1, 0.05],  // Calm: book imbalance matters more
        1 => [0.32, 0.27, 0.23, 0.09, 0.09], // Normal: balanced
        2 => [0.2, 0.2, 0.35, 0.15, 0.1],   // Volatile: spread/vol matter more
        3 => [0.1, 0.4, 0.2, 0.2, 0.1],     // Cascade: flow dominates
        _ => [0.2, 0.2, 0.2, 0.2, 0.2],     // Uniform fallback
    }
}
```

**Files**: `src/market_maker/adverse_selection/pre_fill_classifier.rs`

---

### 2.3 EnhancedToxicity (IR: 0.04 → target 1.5+)

**Current state**: Best performer. Features: run_length, vol_imbal, spread_widen, intensity

**Enhancements**:

```rust
pub struct MicrostructureFeatures {
    // Existing (standardized)
    pub run_length_z: f64,
    pub volume_imbalance_z: f64,
    pub spread_widening_z: f64,
    pub trade_intensity_z: f64,

    // NEW: Order flow toxicity (VPIN-inspired)
    pub volume_synchronized_prob: f64,   // P(informed) from volume buckets

    // NEW: Book dynamics
    pub bid_pull_rate: f64,              // Rate of bid cancellations
    pub ask_pull_rate: f64,              // Rate of ask cancellations
    pub queue_position_decay: f64,       // How fast our queue position degrades

    // NEW: Trade classification
    pub lee_ready_sign: f64,             // Lee-Ready trade classification
    pub bulk_volume_classification: f64, // Bulk volume classification

    // NEW: Entropy features
    pub trade_direction_entropy: f64,    // Low entropy = one-sided, toxic
    pub size_distribution_entropy: f64,  // Low entropy = informed (consistent sizes)
}
```

**Files**: `src/market_maker/adverse_selection/enhanced_classifier.rs`, `src/market_maker/adverse_selection/microstructure_features.rs`

---

### 2.4 RegimeHighVol (IR: 0.00 → target 1.2+)

**Current problem**: Never detects Calm (n=0) or Volatile (n=0). Everything is Normal or Cascade.

**Root cause**: HMM emission thresholds are wrong for BTC.

**Fix 1**: Recalibrate emission parameters from historical data:

```rust
impl RegimeHMM {
    pub fn calibrate_from_data(&mut self, observations: &[Observation]) {
        // Compute empirical quantiles for sigma
        let sigmas: Vec<f64> = observations.iter().map(|o| o.sigma).collect();
        sigmas.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Set thresholds at 25th, 50th, 75th percentiles
        let p25 = sigmas[sigmas.len() / 4];
        let p50 = sigmas[sigmas.len() / 2];
        let p75 = sigmas[3 * sigmas.len() / 4];

        self.emission_params = [
            EmissionParams { sigma_mean: p25 * 0.5, sigma_std: p25 * 0.3, ... }, // Calm
            EmissionParams { sigma_mean: p50, sigma_std: (p75 - p25) * 0.5, ... }, // Normal
            EmissionParams { sigma_mean: p75, sigma_std: p75 * 0.3, ... }, // Volatile
            EmissionParams { sigma_mean: p75 * 2.0, sigma_std: p75 * 0.5, ... }, // Cascade
        ];
    }
}
```

**Fix 2**: Add more discriminative features to observation:

```rust
pub struct RegimeObservation {
    pub sigma: f64,           // Volatility (existing)
    pub spread: f64,          // Spread (existing)
    pub flow_imbalance: f64,  // Flow imbalance (existing)

    // NEW
    pub oi_change_rate: f64,       // Open interest change (cascade indicator)
    pub liquidation_intensity: f64, // Liquidation rate (cascade indicator)
    pub funding_rate: f64,          // Funding rate (regime indicator)
    pub volume_ratio: f64,          // Current vol / 24h average vol
}
```

**Files**: `src/market_maker/estimator/regime_hmm.rs`

---

### 2.5 Momentum (IR: 0.04 → target 1.3+)

**Current**: Single-scale momentum over last N trades.

**Enhancement**: Multi-scale momentum with decay weighting:

```rust
pub struct MultiScaleMomentum {
    // Short-term (5-10 trades, ~1-2 seconds)
    pub momentum_short: f64,
    pub momentum_short_confidence: f64,

    // Medium-term (50-100 trades, ~10-20 seconds)
    pub momentum_medium: f64,
    pub momentum_medium_confidence: f64,

    // Long-term (500+ trades, ~1-2 minutes)
    pub momentum_long: f64,
    pub momentum_long_confidence: f64,

    // Cross-scale agreement
    pub scale_agreement: f64,  // Do all scales agree on direction?

    // Momentum quality
    pub trend_strength: f64,   // ADX-like measure
    pub momentum_persistence: f64,  // Autocorrelation of returns
}

impl MultiScaleMomentum {
    pub fn combined_signal(&self) -> f64 {
        // Agreement bonus: if all scales agree, boost confidence
        let agreement_bonus = if self.scale_agreement > 0.8 { 1.3 } else { 1.0 };

        // Weighted combination with recency preference
        let raw = 0.5 * self.momentum_short
                + 0.35 * self.momentum_medium
                + 0.15 * self.momentum_long;

        raw * agreement_bonus * self.trend_strength
    }
}
```

**Files**: `src/market_maker/estimator/momentum.rs`

---

### 2.6 BuyPressure (IR: 0.02 → target 1.2+)

**Current**: Simple buy ratio over last N trades.

**Enhancement**: Trade clustering and smart aggregation:

```rust
pub struct BuyPressureFeatures {
    // Volume-weighted buy ratio (not count-weighted)
    pub volume_weighted_buy_ratio: f64,

    // Time-decay weighted
    pub ema_buy_ratio: f64,

    // Cluster detection
    pub buy_cluster_count: usize,   // Consecutive buy sequences
    pub sell_cluster_count: usize,
    pub max_cluster_size: usize,    // Longest run

    // Size-conditional
    pub large_trade_buy_ratio: f64, // Buy ratio for trades > 2x median
    pub small_trade_buy_ratio: f64, // Buy ratio for trades < 0.5x median

    // Predictive feature: large trades often precede direction
    pub large_trade_direction_lead: f64,
}
```

**Files**: `src/bin/prediction_validator.rs` (ObservationBuffers)

---

### 2.7 LeadLag (IR: 0.00 → target 1.5+)

**Current problem**:
- Lag oscillates between -500ms and +500ms
- Concentration 5800% (all predictions in one bin)
- conf=0.00 gates it off

**Fix 1**: Stability requirement before using signal:

```rust
pub struct LeadLagGate {
    lag_history: VecDeque<i64>,  // Last N lag estimates
    stability_window: usize,     // How many to check
    max_variance_ms: f64,        // Max allowed variance
}

impl LeadLagGate {
    pub fn is_stable(&self) -> bool {
        if self.lag_history.len() < self.stability_window {
            return false;
        }

        let recent: Vec<_> = self.lag_history.iter().rev().take(self.stability_window).collect();
        let mean = recent.iter().map(|&&x| x as f64).sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|&&x| (x as f64 - mean).powi(2)).sum::<f64>() / recent.len() as f64;

        // Require: variance < threshold AND mean > 0 (causal)
        variance.sqrt() < self.max_variance_ms && mean > 50.0
    }
}
```

**Fix 2**: Use MI as confidence, not just lag:

```rust
pub fn lead_lag_probability(&self) -> f64 {
    if !self.gate.is_stable() {
        return 0.5;  // No signal
    }

    let lag = self.current_lag_ms;
    let mi = self.mutual_information;

    // Confidence based on MI strength
    let mi_confidence = (mi / 2.0).min(1.0);  // MI > 2.0 = full confidence

    // Direction from price diff
    let direction = if self.binance_mid > self.hl_mid { 1.0 } else { -1.0 };

    // Scale probability by confidence
    0.5 + direction * mi_confidence * 0.4  // Range [0.1, 0.9]
}
```

**Files**: `src/market_maker/estimator/lag_analysis.rs`, `src/market_maker/signal_integration.rs`

---

### 2.8-2.10 CrossVenue Features (IR: 0.00 → target 1.2+)

**Current problem**: `conf=0.00` always. Features never activate.

**Root cause** (from code inspection needed): The confidence calculation is gated too strictly.

**Fix**: Lower activation thresholds and add feature scaling:

```rust
impl SignalIntegrator {
    pub fn get_cross_venue_confidence(&self) -> f64 {
        // Current: requires many conditions, often returns 0.0
        // Fix: Use soft thresholds with decay

        let binance_staleness = (now - self.last_binance_ts) as f64 / 1000.0;
        let hl_staleness = (now - self.last_hl_ts) as f64 / 1000.0;

        // Exponential decay instead of hard cutoff
        let freshness = (-binance_staleness / 5.0).exp() * (-hl_staleness / 5.0).exp();

        // Sample count confidence (soft threshold)
        let sample_conf = (self.binance_samples as f64 / 100.0).min(1.0);

        // Combined confidence
        freshness * sample_conf * self.mi_confidence
    }
}
```

**For CV_Agreement**:
```rust
pub fn cross_venue_agreement(&self) -> f64 {
    let bn_direction = (self.binance_momentum > 0.0) as i8 * 2 - 1;  // -1 or +1
    let hl_direction = (self.hl_momentum > 0.0) as i8 * 2 - 1;

    // Agreement strength based on magnitude alignment
    let bn_strength = self.binance_momentum.abs();
    let hl_strength = self.hl_momentum.abs();

    if bn_direction == hl_direction {
        // Same direction: confidence = min strength
        bn_strength.min(hl_strength) * bn_direction as f64
    } else {
        // Opposite direction: disagreement signal
        -(bn_strength + hl_strength) / 2.0
    }
}
```

**Files**: `src/market_maker/estimator/cross_venue.rs`, `src/market_maker/signal_integration.rs`

---

## Phase 3: Validation Protocol

After implementing each phase, re-run prediction_validator:

```bash
cargo run --release --bin prediction_validator -- \
    --asset BTC \
    --duration 1h \
    --report-interval 60 \
    --regime-breakdown \
    --warmup-samples 200 \
    --network mainnet
```

**Success criteria per model**:
- IR > 1.0 (model adds value over base rate)
- Concentration < 50% (predictions are discriminative)
- Resolution > 0.01 (outcomes vary with predictions)

---

## Implementation Order

1. **AdaptiveStandardizer** (fixes all models partially)
2. **RegimeHMM recalibration** (unlocks Calm/Volatile detection)
3. **LeadLag stability gate** (makes signal usable)
4. **CrossVenue ungating** (activates 3 dead features)
5. **EnhancedTox features** (highest current IR, best ROI)
6. **InformedFlow features** (core toxicity detection)
7. **Momentum multi-scale** (already decent, enhance further)
8. **BuyPressure clustering** (moderate improvement expected)
9. **PreFillToxicity regime weights** (depends on HMM fix)

---

## Files to Modify

| Priority | File | Changes |
|----------|------|---------|
| P0 | `src/market_maker/adaptive/standardizer.rs` | Implement AdaptiveStandardizer |
| P0 | `src/market_maker/estimator/regime_hmm.rs` | Recalibrate thresholds |
| P1 | `src/market_maker/estimator/lag_analysis.rs` | Add stability gate |
| P1 | `src/market_maker/signal_integration.rs` | Ungate cross-venue, fix confidence |
| P2 | `src/market_maker/adverse_selection/enhanced_classifier.rs` | Add entropy features |
| P2 | `src/market_maker/estimator/informed_flow.rs` | Add clustering, Kyle lambda |
| P2 | `src/market_maker/estimator/momentum.rs` | Multi-scale momentum |
| P3 | `src/market_maker/adverse_selection/pre_fill_classifier.rs` | Regime-conditional weights |
| P3 | `src/bin/prediction_validator.rs` | Enhanced BuyPressure features |

---

## Expected Outcomes

After full implementation:

| Model | Current IR | Target IR | Key Change |
|-------|------------|-----------|------------|
| InformedFlow | 0.02 | 1.2+ | Clustering + Kyle lambda |
| PreFillToxicity | 0.01 | 1.2+ | Standardization + regime weights |
| EnhancedTox | 0.04 | 1.5+ | Entropy features + VPIN |
| RegimeHighVol | 0.00 | 1.2+ | Threshold recalibration |
| Momentum | 0.04 | 1.3+ | Multi-scale agreement |
| BuyPressure | 0.02 | 1.2+ | Volume-weighted + clustering |
| LeadLag | 0.00 | 1.5+ | Stability gate + MI confidence |
| CV_Agreement | 0.00 | 1.2+ | Soft thresholds |
| CV_Toxicity | 0.00 | 1.2+ | Ungating + scaling |
| CV_Direction | 0.00 | 1.2+ | Momentum-based direction |
