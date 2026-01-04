# Bayesian Parameter Updates: Data Type Analysis

**Author**: Systems Engineering Analysis
**Date**: 2026-01-04
**Scope**: `src/market_maker/estimator/` module
**Status**: Critical review of statistical foundations

---

## Executive Summary

The market maker's parameter estimation pipeline uses Bayesian methods for adaptive parameter learning. However, **several update mechanisms are misspecified relative to the actual data-generating processes (DGPs)**. This document provides a rigorous analysis of where the statistical assumptions break down and the resulting impact on market making performance.

### Key Findings

| Issue | Severity | Impact |
|-------|----------|--------|
| Volume-weighted kappa breaks conjugacy | **High** | Overstates confidence, biases posterior |
| Heterogeneous data sources conflated | **High** | Blends incompatible parameters |
| Stochastic vol assumes wrong observation model | **Medium** | Biases mean-reversion speed |
| Jump detection uses hard threshold | **Medium** | Misses soft/gradual jumps |
| Heavy-tail detection is reactive | **Medium** | Underestimates tail risk |
| Microprice regression has endogeneity | **Low** | Potentially biased coefficients |

---

## 1. Kappa Estimation: Broken Conjugacy

### 1.1 The Claimed Model

From `kappa.rs:17-27`, the documentation states:

```
In GLFT, κ is the fill rate decay parameter in λ(δ) = A × exp(-κδ).
When modeling fill distances as exponential with rate κ:

- Likelihood: L(δ₁...δₙ | κ) = κⁿ exp(-κ Σδᵢ)
- Gamma prior: π(κ | α₀, β₀) ∝ κ^(α₀-1) exp(-β₀ κ)
- Posterior: π(κ | data) = Gamma(α₀ + n, β₀ + Σδᵢ)
```

This is textbook Gamma-Exponential conjugacy. The posterior parameters should be:
- `α_posterior = α₀ + n` (prior shape + **count** of observations)
- `β_posterior = β₀ + Σδᵢ` (prior rate + **sum** of distances)

### 1.2 What the Code Actually Does

From `kappa.rs:236-246`:

```rust
// Posterior parameters with volume weighting
let posterior_alpha = self.prior_alpha + self.sum_volume;           // ❌ WRONG
let posterior_beta = self.prior_beta + self.sum_volume_weighted_distance;  // ❌ WRONG
```

The code uses:
- `α_posterior = α₀ + Σvᵢ` (sum of volumes as pseudo-count)
- `β_posterior = β₀ + Σ(vᵢ × δᵢ)` (volume-weighted distances)

### 1.3 Why This Is Statistically Invalid

**The conjugacy relationship requires the likelihood to factor correctly.**

For n independent observations δ₁, δ₂, ..., δₙ from Exp(κ):
```
L(δ | κ) = ∏ᵢ κ exp(-κδᵢ) = κⁿ exp(-κ Σδᵢ)
```

The volume-weighted version would require:
```
L_weighted(δ, v | κ) = κ^(Σvᵢ) exp(-κ Σvᵢδᵢ)
```

But this is NOT the likelihood for volume-weighted observations. A single large trade (v=10) at distance δ=5bp is **not statistically equivalent** to 10 independent small trades at 5bp each. The large trade is:
- One realization from the same distribution
- Subject to one instance of market conditions
- Potentially from one informed trader

### 1.4 Consequences

| Effect | Description |
|--------|-------------|
| **Inflated confidence** | Large trades artificially boost effective sample size |
| **Biased posterior** | Mean shifts toward distances of high-volume trades |
| **Violated GLFT assumptions** | Spread formula depends on correct κ |

### 1.5 Correct Approaches

**Option A: Use observation counts (not volumes)**
```rust
let posterior_alpha = self.prior_alpha + self.observations.len() as f64;
let posterior_beta = self.prior_beta + self.sum_distance;  // unweighted
```

**Option B: Derive correct weighted likelihood**
If volume-weighting is desired, model each observation as:
```
δᵢ | vᵢ, κ ~ Exponential(κ)  // distance independent of volume
vᵢ ~ some_distribution       // volume has its own model
```
Then the joint likelihood factors differently and requires a different conjugate update.

**Option C: Use importance sampling**
Treat volume as importance weights and use weighted particle filtering instead of conjugate updates.

---

## 2. Heterogeneous Data Sources

### 2.1 Data Streams in the System

The `ParameterEstimator` receives data from multiple sources with fundamentally different data-generating processes:

#### Stream 1: Market-Wide Trades (`on_trade`)
```rust
// parameter_estimator.rs:198-200
self.market_kappa.on_trade(timestamp_ms, price, size, self.current_mid);
```

- **DGP**: Aggregation of ALL market participants
- **Includes**: Informed traders, noise traders, arbitrageurs, other MMs
- **Measures**: Where trades execute relative to mid (market-wide)
- **Selection**: No conditioning on our strategy

#### Stream 2: Our Order Fills (`on_own_fill`)
```rust
// parameter_estimator.rs:305-306
self.own_kappa.record_fill_distance(timestamp_ms, placement_price, fill_price, fill_size);
```

- **DGP**: Conditional on our order placement
- **Includes**: Only trades where WE are counterparty
- **Measures**: How far price moved before hitting OUR orders
- **Selection**: Conditioned on our quotes existing at that level

#### Stream 3: L2 Book Updates (`on_l2_book`)
```rust
// parameter_estimator.rs:336-352
self.book_structure.update(bids, asks, mid);
self.microprice_estimator.on_book_update(...);
```

- **DGP**: Snapshots of order book state
- **Includes**: All resting liquidity
- **Measures**: Current market structure
- **Timing**: Discrete snapshots (not continuous)

### 2.2 The Blending Problem

The code blends different κ estimates:

```rust
// parameter_estimator.rs:422-439
pub fn kappa(&self) -> f64 {
    let own_conf = self.own_kappa.confidence();
    let own = self.own_kappa.posterior_mean();
    let market = self.market_kappa.posterior_mean();

    // Linear blending based on confidence
    own_conf * own + (1.0 - own_conf) * market
}
```

**Statistical Problem**: These are estimating DIFFERENT parameters:
- `κ_own` = fill rate decay for OUR orders
- `κ_market` = fill rate decay for ALL market trades

Linear blending assumes they're noisy estimates of the same underlying quantity. They're not.

### 2.3 Correct Approach: Hierarchical Model

```
κ_market ~ Gamma(α₀, β₀)                    # Market-wide prior
κ_own | κ_market ~ Gamma(f(κ_market), g(κ_market))  # Our κ depends on market
```

Then use the market data to inform the prior for our own fill rate, not blend the posteriors directly.

### 2.4 Information Content Differs

| Data Source | Information per Observation | Arrival Rate | Relevance to Our PnL |
|-------------|---------------------------|--------------|---------------------|
| Market trades | Low (includes noise) | High | Indirect |
| Our fills | High (direct measurement) | Low | Direct |
| L2 book | Medium (state snapshot) | High | Indirect |

The current system treats all observations equally after confidence weighting, ignoring these differences.

---

## 3. Stochastic Volatility: Wrong Observation Model

### 3.1 The Assumed Model

From `volatility.rs:677-682`:
```rust
/// dσ² = κ(θ - σ²)dt + ξσ² dZ  with Corr(dW, dZ) = ρ
```

This is a Heston-style OU process for variance.

### 3.2 The Calibration Assumption

From `volatility.rs:182-203`:
```rust
fn kappa(&self) -> Option<f64> {
    // ...
    // For OU process: autocorr ≈ exp(-κ × Δt)
    // κ ≈ -ln(autocorr) / Δt
    //
    // Assumes Δt ≈ 1 second between observations.  // ❌ WRONG
```

### 3.3 Why This Is Wrong

**Volume ticks are NOT equally spaced in time.**

The volume clock produces observations when cumulative volume exceeds a threshold. In practice:
- During high activity: ticks every 100ms
- During low activity: ticks every 10+ seconds
- During news: clustered bursts

The autocorrelation formula `ρ = exp(-κΔt)` is only valid when Δt is known and constant.

### 3.4 Additional Issues

#### Missing Observation Noise
The variance V_t is not observed directly—it's estimated from returns:
```
V̂_t = r_t² (realized variance proxy)
V̂_t = V_true + ε_measurement
```

The measurement error ε biases κ upward because:
- Noise looks like fast mean-reversion
- Two consecutive noisy estimates appear less correlated than true values

#### Discrete Sampling
The continuous OU process observed at discrete times has transition density:
```
V_{t+Δ} | V_t ~ Normal(θ + (V_t - θ)e^{-κΔ}, ξ²(1-e^{-2κΔ})/(2κ))
```

The calibration should use this exact discrete likelihood, not the continuous autocorrelation approximation.

### 3.5 Impact

| Effect | Direction | Magnitude |
|--------|-----------|-----------|
| κ_vol bias | Upward | Moderate (10-30%) |
| θ_vol accuracy | Slightly low | Small |
| ξ_vol accuracy | Variable | Depends on regime |

---

## 4. Jump Detection: Binary Classification

### 4.1 Current Implementation

From `jump.rs:88-91`:
```rust
pub fn is_jump(&self, log_return: f64, sigma_clean: f64) -> bool {
    let threshold = self.config.jump_threshold_sigmas * sigma_clean;
    log_return.abs() > threshold  // Hard binary decision
}
```

### 4.2 Problems with Hard Thresholds

#### Uncertainty in σ is Ignored
If σ_clean has 20% estimation error:
- True 3σ event could appear as 2.4σ or 3.6σ
- Threshold crossing is probabilistic, not deterministic

#### No Soft Classification
Returns of 2.9σ and 3.1σ are treated completely differently despite being nearly identical. This creates:
- Unstable behavior near threshold
- Loss of information (probability vs binary)

#### Independence Assumption
Each return is evaluated independently, but jumps cluster (Hawkes process):
- P(jump at t | jump at t-1) > P(jump at t | no jump at t-1)

### 4.3 Proper Bayesian Approach

Model price as jump-diffusion:
```
dP = σ dW + J dN
J ~ N(μ_j, σ_j²)
N ~ Poisson(λ)
```

For each return r_t, compute posterior probability:
```
P(jump | r_t) = P(r_t | jump) × P(jump) / P(r_t)

where:
P(r_t | jump) = N(r_t; μ_j, σ_j²)
P(r_t | no jump) = N(r_t; 0, σ²)
P(jump) = λΔt  (for small Δt)
```

This gives a continuous [0,1] probability instead of binary classification.

---

## 5. Heavy-Tail Detection: Reactive Not Preventive

### 5.1 Current Implementation

From `kappa.rs:204-234`:
```rust
// Track heavy-tail detection
if self.cv > 1.2 {
    self.heavy_tail_count += 1;
}
// ... later ...
if self.is_heavy_tailed {
    let multiplier = (2.0 - self.cv).clamp(0.5, 1.0);
    self.kappa_posterior_mean * multiplier
}
```

### 5.2 The Problem

This is a **post-hoc correction** to a posterior computed under the wrong model. The sequence is:
1. Assume Exponential distribution
2. Compute Gamma posterior
3. Detect CV > 1.2 (heavy tail)
4. Apply ad-hoc multiplier to posterior mean

### 5.3 Why This Is Suboptimal

The Exponential assumption affects:
- How each observation updates the posterior
- The posterior variance (uncertainty)
- The interpretation of confidence scores

By the time CV > 1.2 is detected, the posterior has already been corrupted by many updates under the wrong likelihood.

### 5.4 Correct Approach: Mixture or Robust Prior

**Option A: Mixture Model**
```
δ ~ π × Exponential(κ) + (1-π) × Pareto(α, δ_min)
```
Infer (κ, π, α) jointly.

**Option B: Student-t Likelihood**
```
δ ~ t_ν(1/κ)  // Student-t with ν degrees of freedom
```
Heavier tails built in; ν controls tail weight.

**Option C: Gamma-Gamma Hierarchical**
```
κ ~ Gamma(α₀, β₀)
δ | κ, σ ~ Gamma(shape=1, rate=κ/σ)  // Allows overdispersion
```

---

## 6. Microprice Regression: Endogeneity

### 6.1 The Model

From `microprice.rs:47-52`:
```rust
/// E[r_{t+Δ}] = β_book × book_imbalance + β_flow × flow_imbalance
```

### 6.2 Endogeneity Problem

Our market making creates a feedback loop:

```
Our quotes → Book depth → book_imbalance signal
                              ↓
                      Our next quotes ← microprice estimate
```

OLS assumes E[ε | X] = 0, but if X (book_imbalance) is influenced by our past actions, which depend on past residuals, this assumption fails.

### 6.3 Severity Assessment

This issue is **lower severity** because:
- We are a small fraction of total book depth
- Ridge regularization shrinks coefficients toward zero
- Coefficients are clamped to ±10bps

However, in thin markets where we provide significant depth, bias could be material.

### 6.4 Mitigation

Use lagged instruments:
```rust
// Use book_imbalance from 500ms ago, not current
let instrument = lagged_book_imbalance;
```

Or explicitly model our contribution:
```rust
let external_imbalance = book_imbalance - our_contribution;
```

---

## 7. Summary: Data Type to Model Mapping

| Data Type | Assumed Distribution | Actual DGP | Mismatch Severity |
|-----------|---------------------|------------|-------------------|
| Fill distances | Exponential(κ) | Heavy-tailed, clustered | **High** |
| Volume weights | Pseudo-observations | Correlated with distance | **High** |
| Variance obs | OU, fixed Δt | OU, irregular Δt + noise | **Medium** |
| Jump indicator | Binary threshold | Mixture, clustered | **Medium** |
| Book imbalance | Exogenous | Partially endogenous | **Low** |

---

## 8. Recommendations

### 8.1 High Priority

1. **Fix Kappa Conjugacy**
   - Use observation counts, not volume sums
   - Or derive correct weighted likelihood

2. **Hierarchical κ Model**
   - Model κ_own and κ_market jointly
   - Use market data to inform prior, not blend posteriors

### 8.2 Medium Priority

3. **Track Δt for Volatility**
   - Store inter-observation times
   - Use exact discrete OU likelihood

4. **Soft Jump Classification**
   - Output P(jump) ∈ [0,1]
   - Weight downstream calculations by probability

5. **Heavy-Tail Prior**
   - Use mixture or Student-t from the start
   - Remove post-hoc CV adjustment

### 8.3 Lower Priority

6. **Microprice Instruments**
   - Use lagged signals
   - Or model our own book contribution

---

## 9. Appendix: Mathematical Details

### A.1 Correct Gamma-Exponential Update

For n iid observations δ₁, ..., δₙ from Exp(κ):

**Prior**: κ ~ Gamma(α₀, β₀)
- Mean: α₀/β₀
- Variance: α₀/β₀²

**Likelihood**:
```
L(δ | κ) = ∏ᵢ κ exp(-κδᵢ) = κⁿ exp(-κ Σδᵢ)
```

**Posterior**: κ | δ ~ Gamma(α₀ + n, β₀ + Σδᵢ)
- Mean: (α₀ + n) / (β₀ + Σδᵢ)
- Variance: (α₀ + n) / (β₀ + Σδᵢ)²

### A.2 Discrete OU Transition Density

For OU process dV = κ(θ - V)dt + ξ√V dW observed at times t₁ < t₂:

Let Δ = t₂ - t₁. Then:
```
V_{t₂} | V_{t₁} ~ Normal(μ_Δ, σ²_Δ)

where:
μ_Δ = θ + (V_{t₁} - θ) exp(-κΔ)
σ²_Δ = (ξ²/2κ)(1 - exp(-2κΔ))
```

### A.3 Jump-Diffusion Posterior

For return r observed from jump-diffusion:
```
P(J=1 | r) = [λ × N(r; μ_j, σ_j²)] / [λ × N(r; μ_j, σ_j²) + (1-λ) × N(r; 0, σ²)]
```

This is a simple application of Bayes' rule with mixture likelihood.

---

## 10. References

1. Guéant, O., Lehalle, C.A., Fernandez-Tapia, J. (2012). "Optimal Portfolio Liquidation with Limit Orders"
2. Barndorff-Nielsen, O.E., Shephard, N. (2004). "Power and Bipower Variation"
3. Aït-Sahalia, Y., Jacod, J. (2009). "Testing for Jumps in a Discretely Observed Process"
4. Gelman, A. et al. (2013). "Bayesian Data Analysis, 3rd Edition"
