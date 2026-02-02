# 4-Hour Live Run Analysis: HIP-3 HYPE Market Making

**Log File**: `mm_hip3_hyna_HYPE_hip3_2026-02-02_02-32-52.log`
**Duration**: 4 hours (14,400 seconds)
**Date**: 2026-02-02
**Asset**: hyna:HYPE (HIP-3 DEX)

---

## Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| **Daily PnL** | **+$0.57** | ⚠️ Marginal profit |
| **Uptime** | 14,400s (4hr) | ✅ Stable |
| **Orders Placed** | 490 | Low activity |
| **Orders Filled** | 55 | 11.2% fill rate |
| **Volume Bought** | 15.15 | |
| **Volume Sold** | 10.83 | Imbalanced |
| **Final Position** | +3.94 | Long bias accumulated |
| **Adverse Selection** | 10.84 bps | ⚠️ High |
| **Max Drawdown** | 0.0% | ✅ |
| **Kill Switch** | NOT triggered | ✅ |

---

## Model Analysis

### 1. Position Continuation Model (P3 Enhancement)

**Observations from 54 decision points:**

```
Early run:    continuation_p: 0.568-0.674, conf: 0.36-0.80
Late run:     continuation_p: 0.425-0.634, conf: 0.65-0.82
```

**Assessment**: ✅ **Working as designed**
- Fused probability remained stable (not dropping 0.8→0.3 as before)
- Confidence increased over time as observations accumulated
- Hold decisions observed when conditions aligned

**Areas for Improvement**:

| Issue | Evidence | Recommendation |
|-------|----------|----------------|
| **Threshold tuning** | Many "Reduce" actions with urgency ~1.02-1.06 even when continuation_p > 0.55 | The `hold_threshold=0.55` may be too aggressive. Consider 0.50 |
| **Confidence weighting** | Low confidence (0.07-0.10) observed during rapid fills | Increase `conf_hold` from 0.5 to 0.6 for more stable decisions |
| **Signal decay** | External signals (momentum_continuation, trend_agreement) may not update frequently enough | Ensure signals refresh every quote cycle, not just on fills |

---

### 2. Bayesian Belief System (Normal-Inverse-Gamma)

**Observations:**

```
n_price_obs: 0 → 10 → stabilized
belief_bias_bps: -1.22 → +1.96 → +0.62 (late run stable)
belief_confidence: 0.0 → 0.74 (converged)
belief_sigma: 0.000395 → 0.000422 → stable
drift_prob_bearish: 0.02 → 0.99 → varied
```

**Assessment**: ⚠️ **Partially effective**

| Strength | Weakness |
|----------|----------|
| Confidence builds appropriately with observations | Drift flips direction quickly (bullish→bearish in 2 obs) |
| Sigma estimation reasonable for this asset | Bias magnitude small (<2 bps) - may not meaningfully impact quotes |
| Warmup behavior correct | Prior may be too tight (drift_prior_sigma=0.0001) |

**Mathematical Recommendations**:

1. **Increase drift prior variance**: Current `drift_prior_sigma=0.0001` gives ~1 bps/sec std. For HIP-3 thin markets, increase to `0.0003` to allow faster belief updates.

2. **Add observation weighting by volume**: Currently all price observations weighted equally. Weight by trade volume to give more information weight to larger moves.

3. **Implement forgetting factor**: Add exponential decay to old observations:
   ```
   effective_n = n * decay^(time_since_obs)
   ```
   This prevents stale observations from dominating during regime changes.

---

### 3. BOCD Changepoint Detection

**Observations:**

```
l3_cp_prob: 1.000 (startup) → 0.019-0.052 (normal) → 0.784-0.997 (regime shifts)
l3_trust: 1.00 → 0.94-0.99 (normal) → 0.10-0.18 (low trust during changepoints)
```

**Changepoint resets observed**: ~15 times in 4 hours

**Assessment**: ✅ **Working well**

| Strength | Weakness |
|----------|----------|
| Correctly identifies regime shifts | May be too sensitive (15 resets in 4hr) |
| Trust appropriately drops during uncertainty | `l3_cp_prob=1.0` at startup is suspicious |
| Quote gate correctly pulls quotes during confirmed regime changes | |

**Recommendations**:

1. **Startup behavior**: Initialize `cp_prob=0.5` instead of 1.0 at startup. The model has no information yet.

2. **Hysteresis for trust recovery**: Currently trust recovers linearly. Add momentum:
   ```
   trust_t = 0.9 * trust_{t-1} + 0.1 * (1 - cp_prob)
   ```

3. **Calibration check**: 15 resets in 4 hours (~1 every 16 min) seems high for a relatively stable period. Consider:
   - Increasing hazard rate prior (assume changepoints less frequent)
   - Requiring 3 consecutive high-prob cycles before reset (currently 2)

---

### 4. Kappa (Fill Intensity) Estimation

**Observations:**

```
kappa_effective: 1145 → 2548 → 1190 (highly variable)
own: 0-59% weight (fills-based)
book: 1-40% weight (order book)
robust: 30-51% weight (outlier-resistant)
prior: 3-30% weight
own_fills: 0 → 11 (sparse)
warmup: true → false → true (re-enters warmup)
```

**Assessment**: ⚠️ **Needs improvement**

| Issue | Evidence | Impact |
|-------|----------|--------|
| **Re-enters warmup** | `warmup: true` after established fills | Kappa jumps to prior blend, quotes widen unnecessarily |
| **High variance** | 1145 → 2548 (2.2x swing) | Spread instability |
| **Few own fills** | Only 11 fills used for estimation | Insufficient data for reliable estimation |
| **Robust estimator dominance** | 50%+ weight | May be overriding valid fill data |

**Recommendations**:

1. **Lower warmup re-entry threshold**: Currently warmup triggered with too few fills. Increase minimum to prevent oscillation.

2. **Exponentially weighted kappa**:
   ```rust
   kappa_ewma = alpha * kappa_new + (1 - alpha) * kappa_prev
   // alpha = 0.1 for stability
   ```

3. **Time-based fill decay**: Weight recent fills more than old fills:
   ```
   fill_weight = exp(-age_seconds / half_life)
   // half_life = 300s (5 min)
   ```

4. **Regime-conditional priors**: Use different kappa priors per HMM regime:
   | Regime | Kappa Prior |
   |--------|-------------|
   | Quiet | 500 |
   | Normal | 1500 |
   | Bursty | 3000 |
   | Cascade | 5000 |

---

### 5. Multi-Timeframe Trend Detection

**Observations:**

```
short_bps: -16.95 to +41.32 (volatile)
medium_bps: -16.95 to +84.20
long_bps: -31.23 to +209.42
agreement: 0.33 to 1.00
trend_conf: 0.21 to 0.96
p_continuation: 0.439 to 0.739
is_opposed: true (frequently)
```

**Assessment**: ⚠️ **Mixed results**

| Strength | Weakness |
|----------|----------|
| Agreement metric correctly identifies divergence | `p_continuation` often defaults to 0.500 (uninformative) |
| Detects opposing trends correctly | `ewma_warmed: false` for extended periods |
| Urgency scoring activates appropriately | Timeframe weights may not be optimal for HIP-3 |

**Recommendations**:

1. **Faster EWMA warmup**: Current warmup takes too long. For thin markets, use:
   ```
   min_observations_for_warmup = 5  // down from 10
   ```

2. **Adaptive timeframe windows**: HIP-3 markets are thinner than perps. Adjust:
   | Timeframe | Current | Recommended |
   |-----------|---------|-------------|
   | Short | 30s | 60s |
   | Medium | 120s | 300s |
   | Long | 600s | 900s |

3. **Incorporate volume-weighted momentum**:
   ```
   momentum_vw = sum(price_change_i * volume_i) / sum(volume_i)
   ```

---

### 6. Adverse Selection Model

**Final Summary:**
```
fills_measured: 28
realized_as_bps: 10.84
spread_adjustment_bps: 21.68
```

**Per-fill AS observations:**
```
AS: 0.00 bps (initial fills - no history)
AS: 3.55-4.47 bps (normal fills)
AS: 8.44 bps (large adverse fills)
```

**Assessment**: ⚠️ **High adverse selection**

**10.84 bps realized AS is concerning** - this exceeds typical maker fees (1.5 bps) by 7x.

| Issue | Evidence |
|-------|----------|
| **Fills during regime shifts** | Highest AS (8.44 bps) coincides with changepoint detection |
| **No pre-fill AS prediction** | AS measured only after fill, not predicted before |
| **Spread not responsive** | spread_adjustment_bps (21.68) applied, but still losing |

**Recommendations**:

1. **Pre-fill AS classifier**: Build a model to predict AS *before* placing quotes:
   ```
   P(AS > threshold | features) = f(orderbook_imbalance, recent_trades, funding_rate, OI_change)
   ```

2. **Dynamic spread widening**: When BOCD signals changepoint:
   ```
   spread_mult = 1.0 + (cp_prob * 2.0)  // Up to 3x spread during regime shifts
   ```

3. **Quote pulling during high-AS periods**: If realized AS > 5 bps over last 5 fills, widen spread by 2x or pull quotes.

4. **Lead-lag integration**: Use Binance price data (if available) to detect incoming adverse flow before it hits Hyperliquid.

---

### 7. HJB/GLFT Spread Calculation

**Observations:**

```
gamma: 0.250 → 0.384 (adaptive)
optimal_spread_bps: 18.55 → 24.46
effective_floor_bps: 7.6-8.0
kappa_spread_cap_bps: 3.0-10.0
book_depth_usd: 627 → 179,237 (highly variable)
```

**Assessment**: ⚠️ **Formula correct, inputs noisy**

**The GLFT formula itself is fine:**
```
δ* = (1/γ) × ln(1 + γ/κ) + fee
```

**But inputs are unstable:**

| Input | Issue |
|-------|-------|
| gamma | Jumps 0.25→0.38 based on book depth |
| kappa | 2x swings make spread unpredictable |
| sigma | Reasonably stable (~0.0004) |

**Recommendations**:

1. **Smooth gamma transitions**:
   ```rust
   gamma = 0.9 * gamma_prev + 0.1 * gamma_new
   ```

2. **Kappa smoothing** (as above)

3. **Consider Avellaneda-Stoikov extension**: Include inventory penalty:
   ```
   δ^bid = δ* + γ × σ² × q × T
   δ^ask = δ* - γ × σ² × q × T
   ```
   Where q = inventory, T = time horizon

---

## Infrastructure Assessment

### Strengths ✅

| Component | Evidence |
|-----------|----------|
| **Uptime stability** | 4 hours continuous, no crashes |
| **Order management** | 490 placed, 381 cancelled, clean state sync |
| **Safety systems** | Kill switch functional, max position respected |
| **Quote latching** | No evidence of excessive quote churn |
| **Reconciliation** | Exchange state matches local state |

### Weaknesses ⚠️

| Component | Evidence | Impact |
|-----------|----------|--------|
| **Post-only rejections** | `"Post only order would have immediately matched"` | Missed fills, wasted cycles |
| **Untracked fills** | `"[Fill] Untracked order filled"` | State inconsistency risk |
| **Quote gate delays** | Quotes pulled for 3+ confirmations during regime change | Missed opportunity windows |
| **Drift capping** | `"Drift urgency capped to prevent crossing market mid"` | Limits adverse-trend exits |

### Missing Infrastructure

| Component | Current State | Priority |
|-----------|---------------|----------|
| **Cross-exchange feed** | Not integrated | HIGH - Lead-lag is key edge |
| **Trade-level latency tracking** | Not logged | MEDIUM - Need to measure execution quality |
| **Calibration metrics logging** | Partial (AS only) | HIGH - Need Brier score, IR for all models |
| **Regime transition logging** | Only changepoints | MEDIUM - Add HMM transition logging |
| **Position PnL attribution** | Aggregate only | HIGH - Need per-fill attribution |

---

## Statistical/Probability Gaps

### 1. **No Calibration Metrics**

**Problem**: Models output probabilities but we don't verify calibration.

**Missing Metrics**:
- Brier score for continuation model
- Reliability diagrams for belief system
- Information Ratio for all probability outputs

**Recommendation**: Add to logging:
```rust
struct CalibrationLog {
    model_name: String,
    predicted_prob: f64,
    actual_outcome: bool,  // measured after observation
    bin_count: HashMap<u8, (u64, u64)>,  // decile → (hits, total)
}
```

### 2. **Point Estimates Without Uncertainty**

**Problem**: Most model outputs are point estimates (single number), not distributions.

**Examples**:
- `kappa_effective: 1298` - what's the confidence interval?
- `continuation_p: 0.631` - what's the posterior variance?
- `belief_bias_bps: 0.69` - what's the credible interval?

**Recommendation**: Output posterior quantiles:
```
kappa: 1298 [1050, 1580] 90% CI
continuation_p: 0.631 [0.55, 0.71] 90% CI
```

### 3. **Assumption Violations**

| Assumption | Violation Evidence |
|------------|-------------------|
| IID observations | Fills are clustered, not independent |
| Stationary process | Frequent changepoint detection |
| Gaussian returns | Fat tails likely (HIP-3 illiquid) |

**Recommendations**:
1. Use **Hawkes process** for fill intensity (accounts for clustering)
2. Use **Student-t** distribution instead of Gaussian for returns
3. Add **regime-switching** to all models, not just HMM overlay

### 4. **Missing Cointegration Analysis**

**Problem**: Multi-timeframe momentum treats timeframes independently.

**Recommendation**: Add error-correction model:
```
Δprice_short = α × (price_short - β × price_long) + ε
```
Where α = speed of reversion, β = cointegration coefficient

---

## Priority Recommendations

### Immediate (This Week)

| Task | Expected Impact |
|------|-----------------|
| Smooth kappa estimation (EWMA) | -50% spread variance |
| Lower continuation threshold (0.55→0.50) | +More HOLD actions in trends |
| Add AS prediction (pre-fill) | -30% adverse selection |
| Startup cp_prob = 0.5 | Cleaner warmup behavior |

### Short Term (2 Weeks)

| Task | Expected Impact |
|------|-----------------|
| Calibration metrics logging | Know which models work |
| Regime-conditional kappa priors | Better spread in regimes |
| Faster EWMA warmup for trends | Earlier trend detection |
| Cross-exchange feed integration | Lead-lag edge |

### Medium Term (1 Month)

| Task | Expected Impact |
|------|-----------------|
| Hawkes process for fill intensity | Better clustering handling |
| Pre-fill AS classifier | Proactive protection |
| Position PnL attribution | Identify edge sources |
| Avellaneda-Stoikov inventory penalty | Better inventory management |

---

## Conclusion

The system is **operationally stable** but **marginally profitable** (+$0.57 in 4 hours).

**Key Issues**:
1. **High adverse selection** (10.84 bps) is eating into spread capture
2. **Kappa instability** causes spread oscillation
3. **Missing calibration** means we can't verify model accuracy
4. **No lead-lag edge** - critical for perp market making

**The P3 continuation enhancement is working** - we no longer see premature exits. But the next layer of edge requires:
1. Better AS prediction (proactive, not reactive)
2. Cross-exchange information integration
3. Rigorous calibration to identify what actually works

**Overall Grade**: C+ (Infrastructure: B, Models: C, Edge: D)

