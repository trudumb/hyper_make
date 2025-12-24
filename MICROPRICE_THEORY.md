# Theoretically Grounded Signal Integration

## The Problem with Magic Numbers

The original approach had:
```rust
let flow_adjustment = flow_imbalance * half_spread * 0.2;  // WHY 0.2?
let imbalance_skew = book_imbalance * half_spread * 0.3;   // WHY 0.3?
```

These violate the principle of deriving everything from market observables.

---

## Three Theoretically Sound Approaches

### Approach 1: Microprice (Recommended)

**Theory**: The "true" fair price incorporates information from order book state.

**Key Insight**: Book imbalance and flow imbalance **predict short-term returns**. 
If we can estimate *how much* they predict returns, we can compute a better 
fair price to quote around.

**Math**:
```
E[r_{t+Δ} | book_imbalance] = β_book × book_imbalance
E[r_{t+Δ} | flow_imbalance] = β_flow × flow_imbalance

microprice = mid × (1 + β_book × book_imbalance + β_flow × flow_imbalance)
```

**Key Advantage**: The β coefficients are **estimated from your own data**, 
not hardcoded. They adapt to:
- The specific asset you're trading
- Current market regime  
- Time of day effects

**Implementation**:
```rust
// In ParameterEstimator, add:
microprice: MicropriceEstimator,

// On each book update:
self.microprice.update(timestamp, mid, book_imbalance, flow_imbalance);

// In strategy, use microprice instead of mid:
let fair_price = self.estimator.microprice();
let (bid, ask) = glft_quotes(fair_price, gamma, kappa, ...);
```

**Why This Preserves GLFT Optimality**:
- GLFT is optimal for quoting around a given "fair price"
- We're just using a *better* estimate of fair price
- The spread/skew formulas remain mathematically correct

---

### Approach 2: Asymmetric Fill Rates

**Theory**: Book imbalance affects which side fills faster.

If there's buying pressure (positive imbalance):
- Asks fill faster (aggressive buyers lifting offers)
- Bids fill slower (fewer sellers hitting bids)

**Math**: Instead of symmetric κ, use:
```
κ_bid = κ_base × (1 - α × imbalance)  // Buying pressure → lower bid κ → wider bid
κ_ask = κ_base × (1 + α × imbalance)  // Buying pressure → higher ask κ → tighter ask
```

**The α coefficient** should be estimated from actual fill rate data:
- Track fill rates on each side
- Regress against imbalance at time of order placement
- Estimate α that best predicts asymmetric fills

**Advantage**: Stays entirely within GLFT framework. Asymmetry emerges
from the optimal spread formula: δ = (1/γ) × ln(1 + γ/κ)

---

### Approach 3: Direct Return Prediction → Skew

**Theory**: If we can predict E[r], we should shade our quotes accordingly.

**Math**:
```
predicted_return = β_book × book_imbalance + β_flow × flow_imbalance
skew_adjustment = γ × predicted_return × T
```

This is similar to inventory skew: we're accounting for expected P&L 
from predicted price movement.

**Advantage**: Integrates naturally with existing skew calculation.

---

## Comparison

| Approach | Pros | Cons |
|----------|------|------|
| Microprice | Clean separation, preserves GLFT | Requires coefficient estimation |
| Asymmetric κ | Stays in GLFT framework | Need fill-rate tracking |
| Return prediction → skew | Simple to implement | Ad-hoc addition to formula |

**Recommendation**: Start with **Microprice**. It's the cleanest theoretically
and the estimation is straightforward.

---

## Implementation Details

### Coefficient Estimation

Use rolling online regression with:
- **Window**: 1-5 minutes (enough data, but adapts to regime changes)
- **Forward horizon**: 100-500ms (typical fill time)
- **Minimum observations**: 30-50 before using estimates

### Handling Cold Start

Before enough observations:
```rust
if !microprice_estimator.is_warmed_up() {
    // Fall back to raw mid
    let fair_price = mid;
} else {
    let fair_price = microprice_estimator.microprice();
}
```

### Diagnostic Logging

Track and log:
- `beta_book`, `beta_flow` (should be small, ~0.001-0.01)
- `R²` for each regression (how predictive are the signals?)
- `microprice_adjustment_bps` (should be small, ~1-5 bps typically)

If β estimates are huge or R² is near 1.0, something is wrong.

---

## Expected Results

For typical crypto markets:
- `beta_book` ≈ 0.001-0.01 (1-10 bps per unit imbalance)
- `beta_flow` ≈ 0.0005-0.005 (similar magnitude)
- `R²` ≈ 0.01-0.10 (signals are predictive but noisy)

These small coefficients explain why hardcoded 0.2-0.3 were *way too aggressive*.
The actual predictive power is much lower.

---

## Integration with Existing System

```rust
// In estimator.rs
pub struct ParameterEstimator {
    // ... existing fields ...
    microprice: MicropriceEstimator,
}

impl ParameterEstimator {
    pub fn microprice(&self) -> f64 {
        if self.microprice.is_warmed_up() {
            self.microprice.microprice()
        } else {
            self.current_mid
        }
    }
    
    pub fn beta_book(&self) -> f64 { self.microprice.beta_book() }
    pub fn beta_flow(&self) -> f64 { self.microprice.beta_flow() }
}

// In strategy.rs - the clean version
impl QuotingStrategy for GLFTStrategy {
    fn calculate_quotes(...) -> (Option<Quote>, Option<Quote>) {
        // Use microprice instead of raw mid
        let fair_price = market_params.microprice;
        
        // Standard GLFT - NO ad-hoc adjustments
        let half_spread = (1.0/gamma) * (1.0 + gamma/kappa).ln();
        let skew = inventory_ratio * gamma * sigma.powi(2) * T;
        
        let bid = fair_price * (1.0 - half_spread - skew);
        let ask = fair_price * (1.0 + half_spread - skew);
        
        // That's it! No magic multipliers.
    }
}
```

---

## Why This Matters

1. **Interpretability**: Coefficients have meaning (bps of return per unit signal)
2. **Adaptability**: Estimates update as market changes
3. **Testability**: Can validate R² and coefficient stability
4. **Principled**: Derived from data, not guessed
5. **Maintains optimality**: GLFT framework intact
