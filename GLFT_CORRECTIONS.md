# GLFT Strategy Mathematical Corrections

## Summary of Issues Found

Your original implementation had three fundamental mathematical errors:

1. **Wrong half-spread formula** (swapped variables)
2. **Gamma derived from target spread** (inverts the model)
3. **Skew divided by kappa** (wrong formula, missing time horizon)

---

## Error #1: Wrong Half-Spread Formula

### Original Code
```rust
fn half_spread(&self, gamma: f64, kappa: f64) -> f64 {
    let ratio = kappa / gamma;  // ← κ/γ
    (1.0 / kappa) * (1.0 + ratio).ln()  // ← (1/κ)
}
// Result: δ = (1/κ) × ln(1 + κ/γ)
```

### Correct GLFT Formula
```rust
fn optimal_half_spread(&self, gamma: f64, kappa: f64) -> f64 {
    let ratio = gamma / kappa;  // ← γ/κ (FIXED)
    (1.0 / gamma) * (1.0 + ratio).ln()  // ← (1/γ) (FIXED)
}
// Result: δ = (1/γ) × ln(1 + γ/κ)
```

### Numerical Example
With γ = 0.5, κ = 100:

| Formula | Calculation | Result |
|---------|------------|--------|
| **Correct**: (1/γ) × ln(1 + γ/κ) | (1/0.5) × ln(1 + 0.5/100) = 2 × ln(1.005) | **0.998%** |
| **Wrong**: (1/κ) × ln(1 + κ/γ) | (1/100) × ln(1 + 100/0.5) = 0.01 × ln(201) | **5.30%** |

The formulas give completely different results!

---

## Error #2: Deriving Gamma from Target Spread

### Original Approach (BROKEN)
```rust
// "I want 10bp spread, derive gamma to achieve it"
let target_half_spread = config.half_spread_bps as f64 / 10000.0;  // 0.001
let gamma = self.derive_gamma(target_half_spread, kappa, ...);
let half_spread = self.half_spread(gamma, kappa);  // ≈ target (always!)
```

**Problem**: This makes volatility, market conditions, and kappa irrelevant. You're forcing a fixed spread regardless of risk.

With κ = 100 and target δ = 0.0001 (1bp):
```
γ = κ / (e^(κ×δ) - 1) = 100 / (e^0.01 - 1) = 100 / 0.01005 ≈ 9950
```

Gamma becomes enormous (~10,000), which then breaks the skew formula.

### Correct Approach
```rust
// γ is YOUR risk aversion - a personality parameter
// The market (κ, σ) dictates the spread, not you
let gamma = self.risk_aversion;  // Fixed parameter, e.g., 0.5
let half_spread = self.optimal_half_spread(gamma, kappa);  // Market-driven
```

Now when κ drops (thin book), spread naturally widens. When κ rises (deep book), spread tightens. **The model responds to market conditions**.

---

## Error #3: Wrong Skew Formula

### Original Code (THREE bugs)
```rust
fn inventory_skew(&self, inventory_ratio: f64, sigma: f64, gamma: f64, kappa: f64) -> f64 {
    inventory_ratio * gamma * sigma.powi(2) / kappa  // WRONG!
}
```

**Bugs**:
1. Divides by κ (kappa doesn't belong in skew formula)
2. Uses `inventory_ratio` instead of actual position `q`
3. Missing time horizon `T`

### Correct GLFT Formula
```rust
fn reservation_price_offset(&self, position: f64, gamma: f64, sigma: f64, holding_time: f64) -> f64 {
    // r - s = -q × γ × σ² × T
    -position * gamma * sigma.powi(2) * holding_time
}
```

**Where T comes from**:
```rust
let holding_time = 1.0 / arrival_intensity;  // T = 1/λ
```

### Numerical Impact

With position q = 0.01 BTC, γ = 0.5, σ = 0.0002 (per-second), λ = 0.5 fills/sec:

| Formula | Calculation | Skew |
|---------|-------------|------|
| **Original** (κ=100): q×γ×σ²/κ | 0.01 × 0.5 × 4×10⁻⁸ / 100 | **2×10⁻¹² = 0.0 bps** |
| **Correct** (T=2s): -q×γ×σ²×T | -0.01 × 0.5 × 4×10⁻⁸ × 2 | **-4×10⁻¹⁰ = 0.04 bps** |

The correct formula gives 200× larger skew! And it actually scales with market conditions.

---

## Why the Original System Failed

### The Vicious Cycle

1. You set `half_spread_bps = 10` (10bp target)
2. System derives γ ≈ 10,000 to hit that spread
3. Skew formula uses this meaningless γ: `0.5 × 10000 × (0.0002)² / 100`
4. Skew = 2×10⁻⁸ = **0.0002 bps** (essentially zero)
5. No inventory management → adverse selection destroys you

### Why Your Spread Was Always ~2bp

Even though you set 10bp target, the swapped formula + derived gamma created a fixed output:
- With γ ≈ 10000 and κ ≈ 100:
- Wrong formula: (1/κ) × ln(1 + κ/γ) = (1/100) × ln(1 + 100/10000)
- = 0.01 × ln(1.01) ≈ 0.01 × 0.01 = **0.0001 = 1bp**

No matter what happened in the market!

---

## The Corrected Model in Action

### Example Scenario
- BTC at $87,000
- Position: Long 0.01 BTC
- κ = 100 (moderate book depth)
- σ = 0.0002 per-second (20bp/sec volatility)
- λ = 0.5 fills/sec → T = 2 seconds
- γ = 0.5 (moderate risk aversion)

### Step-by-Step Calculation

**1. Optimal Half-Spread**
```
δ = (1/γ) × ln(1 + γ/κ)
δ = (1/0.5) × ln(1 + 0.5/100)
δ = 2 × ln(1.005)
δ = 2 × 0.00499
δ = 0.00998 ≈ 100 bps (1%)
```

**2. Reservation Price Offset**
```
skew = -q × γ × σ² × T
skew = -0.01 × 0.5 × (0.0002)² × 2
skew = -0.01 × 0.5 × 4×10⁻⁸ × 2
skew = -4×10⁻¹⁰
```

In price terms: $87,000 × (-4×10⁻¹⁰) = -$0.000035 (negligible for small position)

**3. Final Quotes**
```
reservation = $87,000 - $0.000035 ≈ $87,000
bid = reservation - δ×mid = $87,000 - 0.00998×$87,000 = $86,131
ask = reservation + δ×mid = $87,000 + 0.00998×$87,000 = $87,869

Spread = $1,738 = 2% (200 bps total, 100 bps half-spread)
```

### When Market Gets Thin (κ drops to 20)
```
δ = (1/0.5) × ln(1 + 0.5/20)
δ = 2 × ln(1.025)
δ = 2 × 0.0247
δ = 0.0494 ≈ 494 bps (5%)
```

**The spread automatically widened 5× because liquidity dried up!**

This is the behavior you want - the model responds to market conditions.

---

## Config Changes Required

### Old Config (Broken)
```toml
half_spread_bps = 10  # Forces fixed spread, breaks model
```

### New Config (Correct)
```toml
risk_aversion = 0.5  # Your risk personality (γ)
# Remove half_spread_bps - the market determines spread now
```

### Risk Aversion Guidelines

| γ Value | Style | Expected Spread (κ=100) | Inventory Skew |
|---------|-------|------------------------|----------------|
| 0.1 | Very Aggressive | ~20bp | Minimal |
| 0.3 | Aggressive | ~60bp | Light |
| 0.5 | Moderate | ~100bp | Medium |
| 1.0 | Conservative | ~200bp | Heavy |
| 2.0 | Very Conservative | ~350bp | Aggressive |

---

## Summary

| Issue | Original | Corrected |
|-------|----------|-----------|
| Half-spread formula | (1/κ)×ln(1+κ/γ) | (1/γ)×ln(1+γ/κ) |
| γ source | Derived from target spread | Fixed risk parameter |
| Skew formula | q×γ×σ²/κ | -q×γ×σ²×T |
| Time horizon T | Missing | T = 1/λ from arrival intensity |
| Market responsiveness | None (fixed output) | Spreads widen/tighten with κ |
| Inventory management | Broken (skew ≈ 0) | Working (meaningful skew) |
