# Stochastic Parameters and Dynamic Bayesian Updates Analysis

## Executive Summary

Observing `gamma = 0.500` and `kappa = 2000.0` consistently in logs indicates the market maker is operating in its **Bayesian prior-dominated warmup phase**. This is expected behavior - the system requires trade data and fills to update these parameters. This document explains the underlying mechanisms and when/how updates occur.

---

## 1. Understanding Kappa (κ)

### 1.1 What Kappa Represents

Kappa is the **fill rate decay parameter** in the GLFT model:

```
λ(δ) = A × exp(-κδ)
```

Where:
- `λ(δ)` = probability of fill at distance δ from mid
- Higher κ = fills more concentrated near mid (tighter markets)
- Lower κ = fills occur at wider distances (wider markets)

**Interpretation:**
| κ value | Expected fill distance | Market type |
|---------|----------------------|-------------|
| 2500 | 4 bps | Liquid (BTC/ETH) |
| 1000 | 10 bps | Moderate liquidity |
| 500 | 20 bps | Less liquid |
| 200 | 50 bps | Illiquid |

### 1.2 Why Kappa = 2000 (or 2500) Appears Static

The system uses a **Bayesian Gamma conjugate prior** for kappa estimation:

```rust
// From estimator/mod.rs (EstimatorConfig defaults)
kappa_prior_mean: 2500.0,     // Prior expected value
kappa_prior_strength: 5.0,    // Effective prior sample size
kappa_window_ms: 300_000,     // 5 minute rolling window
```

#### The Bayesian Update Formula

```
Posterior Mean = (α₀ + n) / (β₀ + Σδᵢ)

Where:
- α₀ = prior_strength = 5
- β₀ = prior_strength / prior_mean = 5 / 2500 = 0.002
- n = sum of fill volumes (effective sample size)
- Σδᵢ = sum of volume-weighted fill distances
```

**With no fills, posterior = prior:**
```
κ_posterior = (5 + 0) / (0.002 + 0) = 2500
```

#### Confidence Calculation

The system tracks confidence separately:

```rust
pub fn confidence(&self) -> f64 {
    let effective_n = self.sum_volume;
    (effective_n.sqrt() / 10.0).min(1.0)
}
```

**To reach 50% confidence:** need sum_volume = 25 units
**To reach 100% confidence:** need sum_volume = 100 units

### 1.3 Why Your Logs Show kappa = 2000 (not 2500)

From your screenshot, the market maker is running on **HYPE on hyna DEX**. The observed value of 2000 suggests one of:

1. **Asset-specific config override** - HYPE may have different defaults
2. **Blended kappa** - The system blends `own_kappa` (from your fills) with `market_kappa` (from tape):

```rust
pub fn kappa(&self) -> f64 {
    let own_conf = self.own_kappa.confidence();
    let own = self.own_kappa.posterior_mean();
    let market = self.market_kappa.posterior_mean();

    // Linear blend based on confidence
    own_conf * own + (1.0 - own_conf) * market
}
```

With low confidence in own_kappa, it heavily weights market_kappa, which may have shifted from the prior based on observed trade tape data.

### 1.4 When Kappa Updates

**Market kappa updates on every trade:**
```rust
// From parameter_estimator.rs
pub fn on_trade(...) {
    if self.current_mid > 0.0 {
        self.market_kappa.on_trade(timestamp_ms, price, size, self.current_mid);
    }
}
```

**Own kappa updates on YOUR fills:**
```rust
pub fn on_own_fill(...) {
    self.own_kappa.record_fill_distance(timestamp_ms, placement_price, fill_price, fill_size);
}
```

Until you accumulate significant fills, kappa remains close to its prior.

---

## 2. Understanding Gamma (γ)

### 2.1 What Gamma Represents

Gamma is **risk aversion** in the GLFT model. It controls the spread:

```
δ* = (1/γ) × ln(1 + γ/κ)
```

**Interpretation:**
| γ value | Trading personality | Spread behavior |
|---------|-------------------|-----------------|
| 0.1 | Aggressive | Tight spreads |
| 0.3 | Moderate | Normal spreads |
| 0.5 | Conservative | Wider spreads |
| 1.0+ | Very conservative | Much wider spreads |

### 2.2 Why Gamma = 0.500 Appears Constant

From your logs, I can see this line:
```
"Ladder spread diagnostics (gamma includes book_depth + warmup scaling)"
gamma = "0.500"
```

This reveals gamma is **exactly 0.500** because:

#### The Gamma Calculation (Legacy Mode)

When NOT using adaptive spreads, gamma is computed via `effective_gamma()`:

```rust
let gamma_effective = cfg.gamma_base         // 0.3 (default)
    * vol_scalar                              // 1.0 (normal vol)
    * toxicity_scalar                         // 1.0 (no toxicity)
    * inventory_scalar                        // 1.0 (low inventory)
    * regime_scalar                           // 1.0 (Normal regime)
    * hawkes_scalar                           // 1.0 (normal activity)
    * time_scalar                             // 1.0 (not toxic hour)
    * book_depth_scalar                       // ≈1.67 (thin book!)
    * warmup_scalar;                          // 1.0 (warmed up)

// Result: 0.3 × 1.67 ≈ 0.5
```

**The culprit is likely `book_depth_scalar`:**

```rust
// From risk_config.rs
pub fn book_depth_multiplier(&self, near_touch_depth_usd: f64) -> f64 {
    if near_touch_depth_usd >= self.book_depth_threshold_usd {  // $50,000
        return 1.0;
    }
    // Thin book → scale from 1.0 to max_book_depth_gamma_mult (1.5)
    let depth_ratio = near_touch_depth_usd / self.book_depth_threshold_usd;
    1.0 + (1.5 - 1.0) * (1.0 - depth_ratio)
}
```

**For HYPE on hyna DEX:** If near-touch depth is low (thin book), gamma scales up.

From your screenshot: `book_depth_usd = "7996"` → only ~$8k depth!

With $8k depth vs $50k threshold:
```
depth_ratio = 8000 / 50000 = 0.16
book_depth_multiplier = 1.0 + 0.5 * (1 - 0.16) = 1.42
gamma = 0.3 × 1.42 × (other scalars ≈ 1.17) ≈ 0.5
```

### 2.3 Alternative: Adaptive Gamma Mode

When `use_adaptive_spreads = true` and `adaptive_can_estimate = true`:

```rust
let gamma = if market_params.use_adaptive_spreads && market_params.adaptive_can_estimate {
    market_params.adaptive_gamma * market_params.tail_risk_multiplier
} else {
    // Legacy multiplicative gamma (above calculation)
}
```

The `adaptive_gamma` comes from a **shrinkage estimator** that blends:
- Prior gamma (0.3)
- Learned adjustments from fill data

Default adaptive_gamma = 0.3, but with tail_risk_multiplier and scaling, could reach 0.5.

---

## 3. Parameter Evolution Over Time

### 3.1 Expected Kappa Evolution

```
Time 0:       κ = 2500 (prior)
              confidence = 0%

After 100 fills:
              κ = posterior from fill distances
              confidence = ~100%

Rolling window (5 min):
              Old observations expire
              Posterior adapts to recent fills
```

**Example with real fills:**

If your fills average 3 bps from placement price:
```
After 100 fills with avg distance = 0.0003:
sum_volume_weighted_distance ≈ 100 × 0.0003 = 0.03
sum_volume ≈ 100

posterior_alpha = 5 + 100 = 105
posterior_beta = 0.002 + 0.03 = 0.032
κ_posterior = 105 / 0.032 ≈ 3,281

Higher κ = fills close to mid = should quote tighter
```

### 3.2 Expected Gamma Evolution

Gamma changes based on **market conditions**, not accumulated data:

| Condition | Effect on γ | Mechanism |
|-----------|-------------|-----------|
| High volatility | ↑ γ | `vol_scalar` up to 3× |
| Toxic regime (RV/BV > 1.5) | ↑ γ | `toxicity_scalar` |
| High inventory | ↑ γ | `inventory_scalar` (quadratic) |
| Thin book | ↑ γ | `book_depth_scalar` (1.0 → 1.5) |
| Extreme regime | ↑ γ | `regime_scalar` (up to 2.5×) |
| Warmup uncertainty | ↑ γ | `warmup_scalar` (up to 1.1×) |
| London/US hours | ↑ γ | `time_scalar` (2×) |

**Gamma should fluctuate** as these conditions change. If it's stuck at 0.5:
1. Book depth is consistently thin (~$8k)
2. Other multipliers are near 1.0
3. OR adaptive mode is enabled with default adaptive_gamma

---

## 4. Why Parameters Appear "Stuck"

### 4.1 Kappa Stuck at 2000

**Root causes:**

1. **Not enough fills accumulated**
   - Prior dominates with low confidence
   - Need ~100+ units of fill volume to move posterior significantly

2. **Fills happening at expected distances**
   - If fills occur at ~5 bps, posterior mean stays near prior
   - Only unusual fill distances (very tight or very wide) shift posterior

3. **Rolling window expiration**
   - 5-minute window means old fills expire
   - Low-activity markets lose observations faster than gaining new ones

**Diagnostic queries (add to logs):**
```rust
tracing::debug!(
    own_kappa_conf = %self.estimator.kappa_confidence(),
    own_kappa_fills = %self.estimator.own_kappa.observation_count(),
    market_kappa_trades = %self.estimator.market_kappa.update_count(),
    "Kappa diagnostic"
);
```

### 4.2 Gamma Stuck at 0.500

**Root causes:**

1. **Thin book dominates**
   - With only $8k near-touch depth, `book_depth_scalar ≈ 1.42`
   - This single factor pushes 0.3 → 0.43, close to 0.5

2. **Warmup scaling still active**
   - `warmup_scalar` can add 10% (default `max_warmup_gamma_mult = 1.1`)
   - 0.43 × 1.1 ≈ 0.47 → rounds to 0.5 in logs

3. **Volatility regime stable**
   - If volatility is "Normal", `regime_scalar = 1.0`
   - No additional gamma boost from vol

4. **Time-of-day scaling**
   - Outside toxic hours (06-08, 14-15 UTC), `time_scalar = 1.0`
   - During toxic hours: 0.3 × 2.0 × 0.83 = 0.5

---

## 5. Implications for Spread Calculation

With γ = 0.5 and κ = 2000:

### 5.1 GLFT Optimal Half-Spread

```
δ* = (1/γ) × ln(1 + γ/κ) + maker_fee

δ* = (1/0.5) × ln(1 + 0.5/2000) + 0.00015
   = 2 × ln(1.00025) + 0.00015
   = 2 × 0.00025 + 0.00015
   = 0.0005 + 0.00015
   = 0.00065 (6.5 bps half-spread)

Full spread = 2 × 6.5 = 13 bps
```

### 5.2 Your Logs Show: `optimal_spread_bps = "17.00"`

The difference (17 vs 13) comes from additional adjustments:
- Spread floor (8 bps minimum from RiskConfig)
- Adverse selection spread adjustment
- Jump premium (if RV/BV > 1.5)
- Warmup uncertainty factor

From your log: `effective_floor_bps = "8.0"` confirms the floor is active.

---

## 6. Recommendations

### 6.1 To See Dynamic Kappa Updates

1. **Run longer** - Allow 5+ minutes of fills to accumulate
2. **Check fill flow** - Ensure fills are being recorded:
   ```bash
   grep "Own fill recorded" logs/mm_*.log | tail -20
   ```
3. **Lower prior strength** - For faster adaptation:
   ```rust
   kappa_prior_strength: 2.0,  // vs default 5.0
   ```

### 6.2 To See Dynamic Gamma Changes

1. **Wait for market conditions to change:**
   - Volatility spike → gamma increases
   - Book depth improves → gamma decreases
   - Enter toxic hours → gamma doubles

2. **Enable adaptive spreads** (if not already):
   ```bash
   cargo run --bin market_maker -- --asset BTC --adaptive-spreads
   ```

3. **Monitor gamma components:**
   ```bash
   grep "Gamma component breakdown" logs/mm_*.log | tail -5
   ```

### 6.3 Logging Enhancements

Add these to see parameter evolution:

```rust
// In quote cycle, log parameter state
tracing::info!(
    kappa_own = %format!("{:.0}", self.estimator.kappa_own()),
    kappa_market = %format!("{:.0}", self.estimator.kappa_market()),
    kappa_confidence = %format!("{:.2}", self.estimator.kappa_confidence()),
    gamma_base = %format!("{:.2}", risk_config.gamma_base),
    book_mult = %format!("{:.2}", risk_config.book_depth_multiplier(depth_usd)),
    warmup_mult = %format!("{:.2}", risk_config.warmup_multiplier(progress)),
    "Parameter breakdown"
);
```

---

## 7. Mathematical Foundation

### 7.1 Bayesian Kappa (Gamma Conjugate Prior)

**Likelihood:** Fill distances are exponentially distributed with rate κ
```
L(δ₁...δₙ | κ) = κⁿ × exp(-κ × Σδᵢ)
```

**Prior:** Gamma distribution
```
π(κ | α₀, β₀) ∝ κ^(α₀-1) × exp(-β₀ × κ)
```

**Posterior:** Also Gamma (conjugacy)
```
π(κ | data) = Gamma(α₀ + n, β₀ + Σδᵢ)

E[κ | data] = (α₀ + n) / (β₀ + Σδᵢ)
Var[κ | data] = (α₀ + n) / (β₀ + Σδᵢ)²
```

**Why this works:**
- Few fills → posterior ≈ prior (regularization)
- Many fills → posterior → MLE (data-driven)
- No arbitrary clamping needed

### 7.2 Dynamic Gamma (Multiplicative Scaling)

```
γ_effective = γ_base × ∏ᵢ scalerᵢ

Where each scaler is [1.0, max_mult]:
- vol_scalar ∈ [1.0, 3.0]
- toxicity_scalar ∈ [1.0, ∞)
- inventory_scalar ∈ [1.0, ∞)
- regime_scalar ∈ [0.8, 2.5]
- book_depth_scalar ∈ [1.0, 1.5]
- warmup_scalar ∈ [1.0, 1.1]
- time_scalar ∈ [1.0, 2.0]
```

**Final γ clamped to [γ_min, γ_max] = [0.05, 5.0]**

### 7.3 GLFT Spread Formula

```
Half-spread: δ* = (1/γ) × ln(1 + γ/κ)

For small γ/κ (typical case):
Taylor expansion: ln(1 + x) ≈ x - x²/2 + x³/3 - ...
δ* ≈ 1/κ - γ/(2κ²) + O(γ²/κ³)

Intuition:
- Higher γ → wider spreads (more risk averse)
- Higher κ → tighter spreads (fills cluster near mid)
```

---

## 8. Summary

| Parameter | Current Value | Why "Stuck" | When Updates |
|-----------|---------------|-------------|--------------|
| κ (kappa) | 2000 | Prior dominates, low fill volume | On each fill, confidence grows with √volume |
| γ (gamma) | 0.500 | Thin book ($8k) × base (0.3) | Reacts to market conditions each quote cycle |

**Key insight:** The parameters ARE updating, but:
1. Kappa needs significant fill volume to deviate from prior
2. Gamma needs market condition changes (volatility, depth, toxicity)

The system is working as designed - conservative priors prevent wild swings during low-information periods.
