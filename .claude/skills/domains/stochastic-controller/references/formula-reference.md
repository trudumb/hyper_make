# Stochastic Controller Formula Reference

Consolidated math reference for all formulas used in the stochastic control system.
For each formula: equation, variable definitions, implementation location, and typical parameter ranges.

---

## 1. GLFT Optimal Half-Spread

The core Avellaneda-Stoikov formula from HJB first-order condition.

### Base Formula

```
delta* = (1/gamma) * ln(1 + gamma/kappa)
```

| Variable | Definition | Units | Typical Range |
|----------|-----------|-------|---------------|
| `delta*` | Optimal half-spread | fractional (multiply by 10000 for bps) | 2-50 bps |
| `gamma` | Risk aversion parameter | dimensionless | 0.01-2.5 (quiet: 0.035, cascade: 1.5) |
| `kappa` | Fill intensity (arrival rate) | fills/time | 50-50000 (quiet: ~200, liquid: ~6000) |

**Implementation**: `stochastic/hjb_solver.rs:HJBSolver::optimal_quotes()` line 196
Also in: `strategy/glft.rs:GLFTStrategy::half_spread()`

**Notes**:
- `kappa` MUST be > 0.0 or the formula blows up (ln of infinity)
- Floor kappa at 1.0 in implementation: `beliefs.expected_kappa.max(1.0)`
- After computing, add maker fee (1.5 bps on Hyperliquid) and clamp to [min, max]
- Lower kappa (fewer fills) => wider spread; higher kappa => tighter spread

### Drift Extension

```
delta_bid = base + mu * T / 2
delta_ask = base - mu * T / 2
```

| Variable | Definition | Units | Typical Range |
|----------|-----------|-------|---------------|
| `mu` | Price drift rate | fractional per second | -0.001 to +0.001 |
| `T` | Time horizon | seconds | 60s (1 min default) |

**Implementation**: `strategy/glft.rs:GLFTStrategy::half_spread_with_drift()` line 571

**Notes**:
- Positive drift (price rising) => widen bids, tighten asks
- Negative drift (price falling) => tighten bids, widen asks
- Floor at maker fee rate to avoid negative-EV quotes
- mu comes from NIG posterior mean `E[mu | data]` (see formula 3)

---

## 2. Inventory Skew

Optimal quote skew from HJB solution, combining inventory penalty and predictive drift.

### Formula

```
skew = gamma * sigma^2 * q * T / 2 + beta_t / 2
       |___ inventory component ___|   |_ drift _|
```

| Variable | Definition | Units | Typical Range |
|----------|-----------|-------|---------------|
| `sigma` | Volatility (posterior mean) | fractional per sqrt(s) | 0.0001-0.01 |
| `q` | Current inventory (normalized: position/max_position) | dimensionless | -1.0 to +1.0 |
| `T` | Time horizon | seconds | 60s |
| `beta_t` | Predictive bias = `E[mu | data]` | fractional per second | -0.001 to +0.001 |

**Implementation**: `stochastic/hjb_solver.rs:HJBSolver::optimal_quotes()` lines 205-218

**Notes**:
- Positive skew = shift quotes down (sell more aggressively)
- Negative skew = shift quotes up (buy more aggressively)
- Long position (q > 0) gives positive inventory skew (want to sell)
- `predictive_bias_scale` default is 0.5 (the 1/2 factor)
- The skew is in bps after multiplying by 10000

---

## 3. NIG Conjugate Updates (Drift and Volatility)

Normal-Inverse-Gamma posterior for jointly estimating drift `mu` and volatility `sigma^2`.

### Prior

```
sigma^2 ~ InverseGamma(alpha_0, beta_0)
mu | sigma^2 ~ Normal(m_0, sigma^2 / k_0)
```

### Single Observation Update

After observing return `x`:

```
k_n     = k_0 + 1
m_n     = (k_0 * m_0 + x) / k_n
alpha_n = alpha_0 + 0.5
beta_n  = beta_0 + 0.5 * k_0 * (x - m_0)^2 / k_n
```

### Batch Update

After observing `n` returns with sample mean `x_bar` and sum of squares `SS`:

```
k_n     = k_0 + n
m_n     = (k_0 * m_0 + n * x_bar) / k_n
alpha_n = alpha_0 + n/2
beta_n  = beta_0 + 0.5 * SS + 0.5 * k_0 * n * (x_bar - m_0)^2 / k_n
```

### Posterior Quantities

```
E[mu | data]      = m_n                           (posterior mean drift)
Var[mu | data]    = beta_n / (k_n * (alpha_n - 1))  (for alpha_n > 1)
E[sigma^2 | data] = beta_n / (alpha_n - 1)          (for alpha_n > 1)
```

| Parameter | Default Prior | Meaning |
|-----------|--------------|---------|
| `m_0` | 0.0 | Zero-centered drift prior |
| `k_0` | 1.0 | Pseudo-sample size (1 prior observation) |
| `alpha_0` | 2.0 | Shape (alpha > 1 gives finite variance) |
| `beta_0` | 0.0001 | Scale (prior variance ~ 0.0001) |

**Implementation**: `stochastic/conjugate.rs:NormalInverseGamma`
- `update()` for single observation (line 98)
- `update_batch()` for batch (line 113)
- `posterior_mean()` for `E[mu | data]` (line 155)
- `posterior_sigma_sq()` for `E[sigma^2 | data]` (line 178)
- `decay()` for non-stationarity: blends toward prior with retention factor

**Notes**:
- The posterior mean `m_n` IS the predictive drift signal -- not a heuristic
- When beliefs shift to negative mu (capitulation), beta_t < 0 automatically
- Weighted update via `update_weighted(x, weight)` for time-weighted observations
- `soft_reset(retention)` keeps a fraction of learned information

---

## 4. Gamma Conjugate Updates (Fill Intensity)

Gamma posterior for fill intensity `kappa`, with depth-dependent exposure.

### Prior and Model

```
kappa ~ Gamma(alpha, beta)
fills ~ Poisson(kappa * exp(-gamma_depth * delta) * dt)
```

### Update

After observing `n_fills` in time `dt` at depth `delta`:

```
effective_exposure = dt * exp(-gamma_depth * delta / 10000)
alpha_n = alpha + n_fills
beta_n  = beta + effective_exposure
```

### Posterior Quantities

```
E[kappa | data] = alpha / beta           (posterior mean)
Var[kappa]      = alpha / beta^2         (posterior variance)
CV              = 1 / sqrt(alpha)        (coefficient of variation)
lambda(delta)   = E[kappa] * exp(-gamma_depth * delta / 10000)  (intensity at depth)
```

| Parameter | Default Prior | Meaning |
|-----------|--------------|---------|
| `alpha_0` | 20.0 | Shape (moderate confidence) |
| `beta_0` | 0.1 | Rate (prior mean = 200 fills/time) |
| `gamma_depth` | 0.5 | Depth sensitivity (lambda halves every ~1.4 bps) |

**Implementation**: `stochastic/conjugate.rs:FillIntensityPosterior`
- `update(n_fills, dt, depth_bps)` (line 325)
- `posterior_mean()` (line 334)
- `intensity_at_depth(depth_bps)` (line 351)
- `is_warmed_up()`: requires >= 10 fills AND > 60s total time

**Notes**:
- Depth-dependent exposure adjusts for the fact that deeper quotes fill less often
- The fill probability at depth: `P(fill) = 1 - exp(-lambda * dt)` (line 282 in hjb_solver.rs)
- `decay(factor)` blends alpha/beta toward prior for non-stationarity

---

## 5. BOCD Run-Length Update

Bayesian Online Change Point Detection (Adams and MacKay, 2007).

### Run-Length Distribution

```
P(r_t = 0 | x_{1:t}) = sum_r [ P(r_{t-1} = r) * H * pi(x_t | r) ]    (changepoint)
P(r_t = r+1 | x_{1:t}) = P(r_{t-1} = r) * (1 - H) * pi(x_t | r)      (growth)
```

Then normalize: `P(r_t | x_{1:t}) = P(r_t) / sum(P(r_t))`

| Variable | Definition | Typical Value |
|----------|-----------|---------------|
| `H` | Hazard rate (P(changepoint per step)) | 1/250 (expected run = 250 obs) |
| `pi(x_t \| r)` | Predictive probability under run length `r` | Student-t from Normal-Gamma |
| `r_t` | Run length at time `t` | 0 to max_run_length (500) |

### Changepoint Detection

```
cp_prob(k) = sum_{r=0}^{k-1} P(r_t = r | data)    (prob of CP in last k obs)
```

Detection thresholds (regime-dependent):
- ThinDex (HIP-3): threshold = 0.85, requires 2 confirmations
- LiquidCex: threshold = 0.50, requires 1 confirmation
- Cascade: threshold = 0.30, requires 1 confirmation

**Implementation**: `control/changepoint.rs:ChangepointDetector`
- `update(observation)` (line 365)
- `changepoint_probability(k)` (line 436)
- `detect_with_confirmation()` returns `None | Pending(n) | Confirmed` (line 482)
- `should_reset_beliefs()`: P(CP in last 10) > 0.7

**Notes**:
- Initialization uses spread distribution (50% on r=0..5, 50% on r=5..20) to avoid
  false positives at startup (P0 FIX)
- Warmup: requires min 10 observations + entropy below threshold (or 50 obs hard cap)
- `run_length_entropy()` measures uncertainty about regime age

---

## 6. Value Function

Linear value function approximation with terminal condition.

### Terminal Condition

```
V(T, x, q, S) = x + q * S - penalty * q^2
```

| Variable | Definition |
|----------|-----------|
| `x` | Cash wealth |
| `q` | Inventory (position) |
| `S` | Mid-price |
| `penalty` | Terminal inventory penalty |

### Basis Function Approximation

```
V(s) = w^T * phi(s) = sum_i w_i * phi_i(s)
```

21 basis features (indexed 0-20):

| Index | Feature | Formula | Purpose |
|-------|---------|---------|---------|
| 0 | Constant | 1.0 | Bias term |
| 1 | Wealth | x / 1000 | Linear wealth effect |
| 2 | Wealth^2 | (x/1000)^2 | Risk aversion |
| 3 | Position | q | Linear inventory |
| 4 | Position^2 | q^2 | Quadratic inventory cost |
| 5 | Abs Position | \|q\| | Symmetric inventory penalty |
| 6 | Time | t | Time progression |
| 7 | Time urgency | sqrt(1-t) | Urgency near terminal |
| 8 | Terminal | sigmoid((t-0.95)*20) | Smooth terminal indicator |
| 9 | Position x Time | q * t | Inventory costlier near end |
| 10 | Position x AS | q * E[AS] / 10 | Inventory x toxicity |
| 11 | Expected Edge | E[edge] / 10 | Edge estimate |
| 12 | Edge Uncertainty | std(edge) / 10 | Exploration value |
| 13 | Confidence | confidence | Overall belief quality |
| 14 | Regime Entropy | H(regime) | Regime uncertainty |
| 15 | Headroom | h | Rate limit headroom |
| 16 | Headroom^2 | h^2 | Nonlinear quota value |
| 17 | Headroom x \|Pos\| | h * \|q\| | Quota matters more w/ inventory |
| 18 | Drift x Position | drift * q * 100 | Penalizes opposing drift |
| 19 | OU Uncertainty | min(ou_unc, 1.0) | Drift estimate confidence |
| 20 | Drift x Time | drift * (1-t) * 100 | Drift matters w/ time left |

### TD(0) Weight Update

```
td_error = r + gamma * V(s') - V(s)
w <- w + alpha * td_error * phi(s)
```

**Implementation**: `control/value.rs:ValueFunction`
- `compute_basis(state)` (line 87)
- `value(state)` (line 75)
- `update_td(transition)` (line 148)
- Config: 21 basis functions, gamma=0.99, lr=0.01 with 0.9999 decay

---

## 7. OU Drift Process

Ornstein-Uhlenbeck mean-reverting drift model with threshold gating.

### Process

```
dD_t = theta * (mu - D_t) * dt + sigma_D * dW_t
```

| Variable | Definition | Default | Regime Variants |
|----------|-----------|---------|-----------------|
| `theta` | Mean reversion rate | 0.5 | Normal: 0.5, Trending: 0.2, Cascade: 1.0 |
| `mu` | Long-term mean | 0.0 | Neutral drift |
| `sigma_D` | Drift volatility | 0.001 | 1 bps/sec typical |

### Threshold Gating (reconciliation filter)

Only reconcile if innovation exceeds threshold:

```
|observed - predicted| > k * sigma * sqrt(dt)
```

| Parameter | Default | Regime Variants |
|-----------|---------|-----------------|
| `reconcile_k` | 2.0 | Normal: 2.0, Trending: 1.5, Cascade: 3.0 |

Filters ~60-80% of noise-induced updates while responding to genuine regime changes.

### Regime Adaptation

Via `adapt_to_regime(regime)`:
- **Normal**: theta=0.5, reconcile_k=2.0 (balanced)
- **Trending**: theta=0.2, reconcile_k=1.5 (trust trends longer, slower reversion)
- **Cascade**: theta=1.0, reconcile_k=3.0 (fast mean reversion, high noise filter)

**Implementation**: `process_models/hjb/ou_drift.rs:OUDriftEstimator`
- `OUDriftConfig` (line 29)
- `adapt_to_regime()` (line 353)
- Update produces `OUUpdateResult` with drift, predicted, innovation, threshold
- Half-life = ln(2)/theta (0.5 => ~1.4s half-life)

**Notes**:
- Variance floor: `min_variance = 1e-12` prevents degenerate behavior
- Variance cap: `max_variance = 0.01` prevents runaway estimation
- The OU model replaces simple EWMA smoothing for more principled drift tracking
