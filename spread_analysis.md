# Comprehensive Spread Calculation Analysis

## Executive Summary

The market maker is producing wider-than-expected spreads (~9.2 bps) despite low volatility (~1.21 bps). This analysis traces the issue through the entire pipeline from estimation to quote generation.

---

## 1. Observed Symptoms (from logs)

| Parameter | Value | Expected | Status |
|-----------|-------|----------|--------|
| Spread | ~9.2 bps ($81 on $87,888 BTC) | ~2-5 bps | ❌ Too wide |
| gamma_spread | 10000.00 | Dynamic | ❌ Clamped at max |
| gamma_skew | 10000.00 | Dynamic | ❌ Clamped at max |
| is_toxic | true | false | ⚠️ Possibly correct |
| jump_ratio | 1.61 | ~1.0 | ✓ Above 1.5 threshold |
| sigma_effective | 0.000121 (1.21 bps) | - | ✓ Normal |
| flow_adj | ~7 bps | ~0.5-1 bps | ⚠️ Seems high |

---

## 2. Root Cause Analysis

### 2.1 Primary Issue: Gamma Clamping at Maximum

**The Formula:**
```rust
gamma = target_half_spread * kappa / (max_position * sigma²)
```

**With your observed values:**
- `target_half_spread` = 5 bps = 0.0005 (assuming `half_spread_bps: 5`)
- `kappa` ≈ 50-100 (from L2 regression)
- `max_position` ≈ 0.1-1.0 BTC (typical config)
- `sigma` = 0.000121 (very low!)

**Calculation:**
```
sigma² = 0.000121² = 1.46e-8

gamma = 0.0005 * 100 / (0.1 * 1.46e-8)
      = 0.05 / 1.46e-9
      = 34,246,575  ← EXPLOSION!

Clamped to: max_gamma = 10000
```

**Impact:** When gamma is clamped, the half-spread formula loses its intended behavior:
```rust
half_spread = (1/gamma) * ln(1 + gamma/kappa)
            = (1/10000) * ln(1 + 10000/100)
            = 0.0001 * ln(101)
            = 0.0001 * 4.615
            = 0.000462 = 4.62 bps
```

This becomes the *base* spread before all adjustments.

---

### 2.2 Toxic Regime Detection

**Current Configuration:**
```rust
jump_ratio_threshold: 1.5  // LOWERED from 3.0
```

**Your logs show:** `jump_ratio: 1.61`

**Conclusion:** The toxic regime detection is **working as intended**. The threshold was deliberately lowered to 1.5 to detect toxic flow earlier.

**Toxicity Multiplier:**
```rust
factor = (1.61 / 2.0).clamp(1.0, 2.5) = 0.805 → clamped to 1.0
```

At `jump_ratio = 1.61`, the toxicity multiplier is 1.0x (no widening yet). The multiplier only starts affecting spread when `jump_ratio > 2.0`.

---

### 2.3 Flow Adjustment Breakdown

**The formula:**
```rust
flow_adjustment = flow_imbalance * half_spread * 0.2
```

**Expected range:**
- `flow_imbalance` ∈ [-1, 1]
- `half_spread` ≈ 4.6 bps (from gamma clamp)
- Maximum `flow_adjustment` = 1.0 * 4.6 * 0.2 = **0.92 bps**

**Your observation:** ~7 bps flow adjustment

**Possible explanations:**
1. The 7 bps includes ALL adjustments (toxicity + skew + falling knife + flow)
2. The `flow_imbalance` calculation may be returning values outside [-1, 1] (check `.clamp()` calls)
3. Confusion in log interpretation

---

### 2.4 Complete Spread Decomposition

Based on the code, the final spread is:

```
bid_delta = (half_spread + skew + bid_protection - flow_adjustment) × toxicity_multiplier
ask_delta = (half_spread - skew + ask_protection - flow_adjustment) × toxicity_multiplier

total_spread = bid_delta + ask_delta
             ≈ 2 × half_spread × toxicity_multiplier + protections
```

**With observed values:**
```
half_spread ≈ 4.6 bps (from gamma clamp)
toxicity_multiplier = 1.0
base_spread = 2 × 4.6 = 9.2 bps  ← MATCHES YOUR OBSERVATION!
```

---

## 3. The Volatility Estimation Pipeline

### 3.1 Volume Clock Sampling

```
Raw Trades → Volume Buckets (0.01 BTC each) → VWAP Returns
```

**Current config:**
```rust
initial_bucket_volume: 0.01,  // Very small buckets
min_bucket_volume: 0.001,     // Even smaller allowed
```

**Potential issue:** Small volume buckets may capture microstructure noise, making BV/RV estimates noisy.

### 3.2 Bipower Variation (σ estimation)

```rust
// RV: EWMA of r²
rv = α × r² + (1-α) × rv

// BV: EWMA of (π/2)|rₜ||rₜ₋₁|
bv = α × (π/2)|rₜ||rₜ₋₁| + (1-α) × bv

// σ_clean = √BV (robust to jumps)
// σ_total = √RV (includes jumps)
```

**Your σ = 0.000121 is very low** because:
1. The market is genuinely quiet
2. VWAP pre-averaging is filtering out most microstructure noise
3. BV specifically excludes jump contributions

### 3.3 Sigma Effective Blending

```rust
fn sigma_effective(&self) -> f64 {
    let clean = self.sigma_clean();
    let total = self.sigma_total();
    let jump_ratio = self.jump_ratio_fast();

    // At ratio=1: pure clean
    // At ratio=3: 67% total
    let jump_weight = 1.0 - (1.0 / jump_ratio.max(1.0));
    let jump_weight = jump_weight.clamp(0.0, 0.85);

    (1.0 - jump_weight) * clean + jump_weight * total
}
```

**With `jump_ratio = 1.61`:**
```
jump_weight = 1.0 - (1.0 / 1.61) = 1.0 - 0.621 = 0.379

sigma_effective = 0.621 × sigma_clean + 0.379 × sigma_total
```

---

## 4. Kappa Estimation Analysis

**Method:** Weighted linear regression on L2 book depths

```rust
// Model: ln(size) = ln(A) - κ × distance
// Slope = -κ

// In practice, slope is often POSITIVE (liquidity increases with distance)
// Code defaults to κ = 50 when this happens
```

**If kappa is too low:** gamma explodes → clamping
**If kappa is too high:** spread becomes too tight

---

## 5. Recommended Fixes

### 5.1 Fix Gamma Clamping (High Priority)

**Option A: Increase max_gamma**
```rust
// In GLFTStrategy::new()
max_gamma: 100000.0,  // Was 10000.0
```

**Option B: Add sigma floor for gamma calculation**
```rust
fn derive_gamma(&self, target_half_spread: f64, kappa: f64, sigma: f64, max_position: f64) -> f64 {
    // Ensure sigma is at least 1 bps for gamma calculation
    let sigma_floor = 0.0001;  // 1 bps minimum
    let sigma_adjusted = sigma.max(sigma_floor);
    let sigma_sq = sigma_adjusted.powi(2);
    // ... rest of calculation
}
```

**Option C: Use sigma_total instead of sigma_clean for gamma**
```rust
// In calculate_quotes()
let sigma_for_spread = market_params.sigma.max(market_params.sigma_total);
```

### 5.2 Add Gamma Clamp Logging

```rust
fn derive_gamma(&self, ...) -> f64 {
    let raw_gamma = target_half_spread * kappa / (max_pos * sigma_sq);
    let clamped = raw_gamma.clamp(self.min_gamma, self.max_gamma);
    
    if raw_gamma >= self.max_gamma {
        warn!(
            raw_gamma = %format!("{:.0}", raw_gamma),
            clamped = %format!("{:.0}", clamped),
            sigma = %format!("{:.6}", sigma),
            "Gamma clamped at maximum - spread may be distorted"
        );
    }
    
    clamped
}
```

### 5.3 Review Toxic Threshold

The 1.5 threshold is intentionally low for early detection. However, if it's causing too many false positives:

```rust
// In EstimatorConfig
jump_ratio_threshold: 2.0,  // More conservative
```

### 5.4 Verify Flow Imbalance Bounds

The code already clamps to [-1, 1]:
```rust
pub(super) fn imbalance(&self) -> f64 {
    self.ewma_imbalance.clamp(-1.0, 1.0)
}
```

But verify the EWMA update doesn't accumulate beyond bounds.

---

## 6. Diagnostic Recommendations

### 6.1 Enhanced Logging

Add these debug fields to track the issue:

```rust
debug!(
    // Existing fields...
    gamma_raw = %format!("{:.0}", raw_gamma),
    gamma_clamped = (raw_gamma >= self.max_gamma),
    sigma_used = %format!("{:.6}", sigma_for_spread),
    spread_contribution_base = %format!("{:.2}", half_spread * 10000.0),
    spread_contribution_toxic = %format!("{:.2}", (toxicity_multiplier - 1.0) * half_spread * 10000.0),
    spread_contribution_flow = %format!("{:.2}", flow_adjustment * 10000.0),
    "Spread breakdown in bps"
);
```

### 6.2 Unit Test for Observed Conditions

```rust
#[test]
fn test_observed_conditions_from_logs() {
    let strategy = GLFTStrategy::new(0.01);
    let config = QuoteConfig {
        mid_price: 87888.0,
        half_spread_bps: 5,
        decimals: 2,
        sz_decimals: 4,
        min_notional: 10.0,
        max_position: 0.1,  // Adjust based on your config
    };
    
    let params = MarketParams {
        sigma: 0.000121,        // From your logs
        sigma_total: 0.000195,  // Estimated from jump_ratio
        sigma_effective: 0.000121,
        kappa: 100.0,
        arrival_intensity: 1.0,
        is_toxic_regime: true,
        jump_ratio: 1.61,
        momentum_bps: 0.0,
        flow_imbalance: 0.0,
        falling_knife_score: 0.0,
        rising_knife_score: 0.0,
    };
    
    let (bid, ask) = strategy.calculate_quotes(&config, 0.0, 0.1, 0.05, &params);
    
    let spread_bps = (ask.unwrap().price - bid.unwrap().price) / config.mid_price * 10000.0;
    
    println!("Spread: {:.2} bps", spread_bps);
    
    // With current code, expect ~9 bps
    // With fixes, should be closer to 2-5 bps
}
```

---

## 7. Summary

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Wide spread | Gamma clamping at 10000 | Increase max_gamma or add sigma floor |
| Gamma explosion | Very low sigma (0.000121) | Use sigma floor for gamma derivation |
| Toxic detection | Threshold lowered to 1.5 | Intended behavior, but can raise to 2.0 |
| ~7 bps flow_adj | Likely includes all adjustments | Clarify logging |

**Primary Fix:** Address the gamma clamping by either raising `max_gamma` or implementing a sigma floor in `derive_gamma()`. This is the single change that will have the biggest impact on spread normalization.
