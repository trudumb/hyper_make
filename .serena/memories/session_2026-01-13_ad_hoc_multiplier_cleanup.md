# Session: 2026-01-13 Ad-Hoc Multiplier Cleanup

## Summary
Systematic cleanup of ad-hoc multipliers in GLFT strategy to trust first-principles mathematics.

## Problem Statement
The GLFT model is self-adaptive (κ handles market tightness, γ flows through formula), but ad-hoc multipliers were overriding this mathematically principled behavior:
1. **Redundant scaling**: Volatility scaled twice (vol_scalar + regime_scalar)
2. **Contradictory logic**: Tight spreads from κ were manually widened by regime multipliers
3. **Arbitrary thresholds**: Hawkes activity used hardcoded 0.8/0.9 percentile cutoffs

## Changes Made

### 1. Removed `spread_regime_mult` (glft.rs:454-466)
**Before:**
```rust
let spread_regime_mult = match market_params.spread_regime {
    SpreadRegime::VeryTight => 1.3,
    SpreadRegime::Tight => 1.1,
    SpreadRegime::Normal => 1.0,
    SpreadRegime::Wide => 0.95,
    SpreadRegime::VeryWide => 0.9,
};
half_spread_bid *= spread_regime_mult;
```

**Why removed:** This directly contradicted GLFT's κ-based adaptation. When κ indicates tight spreads are appropriate (deep book), manually widening via a multiplier undermined the mathematical model.

### 2. Removed `regime_scalar` (glft.rs:109-114)
**Before:**
```rust
let regime_scalar = match market_params.volatility_regime {
    VolatilityRegime::Low => 0.8,
    VolatilityRegime::Normal => 1.0,
    VolatilityRegime::High => 1.5,
    VolatilityRegime::Extreme => 2.5,
};
```

**Why removed:** Redundant with `vol_scalar` which already handles volatility scaling via the continuous relationship: `1 + weight × (σ/σ_baseline - 1)`. The discrete regime buckets with arbitrary multipliers caused double-scaling.

### 3. Refactored `hawkes_scalar` to Continuous Function
**Before:**
```rust
let hawkes_scalar = if market_params.hawkes_activity_percentile > 0.9 {
    1.5
} else if market_params.hawkes_activity_percentile > 0.8 {
    1.2
} else {
    1.0
};
```

**After:**
```rust
let hawkes_baseline = 0.5;  // Median as baseline
let hawkes_sensitivity = 2.0;
let hawkes_scalar = 1.0 + hawkes_sensitivity * (percentile - hawkes_baseline).max(0.0);
```

**Why changed:** Continuous relationship is mathematically justified: when λ_current >> λ_baseline, informed flow probability increases. Arbitrary thresholds (0.8, 0.9) lacked derivation.

### 4. Documented Deep Constants
Added derivation comments for:
- `0.00005` (Hawkes skew scaling): ~0.5 bps per 0.1 imbalance at 100% activity
- `0.01` (Funding skew scaling): Conservative funding cost incorporation

## Files Modified

| File | Changes |
|------|---------|
| `src/market_maker/strategy/glft.rs` | Removed spread_regime_mult, regime_scalar; refactored hawkes_scalar; documented constants |

## Impact Analysis

**Gamma Factors (8 → 7):**
Before: gamma_base × vol_scalar × toxicity_scalar × inventory_scalar × **regime_scalar** × hawkes_scalar × time_scalar × book_depth_scalar × warmup_scalar × calibration_scalar

After: gamma_base × vol_scalar × toxicity_scalar × inventory_scalar × hawkes_scalar × time_scalar × book_depth_scalar × warmup_scalar × calibration_scalar

**Spread Adjustments (removed 1):**
Before: half_spread × **spread_regime_mult** (after GLFT calculation)
After: Trusts GLFT model to handle market tightness via κ

## Verification
- ✅ `cargo build`: Compiles successfully
- ✅ `cargo test`: 823 tests pass
- Removed unused imports: `VolatilityRegime`, `SpreadRegime`

## Priority 2: Derive from Data (COMPLETED)

### 1. Documented `toxic_hour_gamma_multiplier: 2.0` (risk_config.rs:84-90)
**Added derivation comment:**
```rust
/// DERIVATION (Dec 2025 trade analysis, 2,038 trades):
/// - Toxic hours (6,7,14 UTC): avg adverse selection = -14 bps
/// - Maker fee: 1.5 bps
/// - Normal spread floor: 8 bps
/// - Required to break even: toxic_mult = (|toxic_edge| + fees) / normal_spread
///                                      = (14 + 1.5) / 8 = 1.94 ≈ 2.0
pub toxic_hour_gamma_multiplier: f64,
```

**Status:** ✅ Already justified from empirical data - documented the derivation.

### 2. Documented `max_book_depth_gamma_mult: 1.5` (risk_config.rs:113-119)
**Added derivation comment:**
```rust
/// DERIVATION (Exit slippage model):
/// - Reference depth: $50k (can exit typical $10k position at near-touch)
/// - At $10k depth: only 20% executable at touch, 80% walks book
/// - AS decay model: AS(δ) = 3 × exp(-δ/8) bps
/// - Expected slippage increase when depth drops 5×: ~50%
/// - gamma_adj = 1 + slippage_increase = 1.5
pub max_book_depth_gamma_mult: f64,
```

**Status:** ✅ Justified from AS decay model.

### 3. Implemented Dynamic Warmup Multiplier (CI-based)

**New function in risk_config.rs:208-231:**
```rust
/// Warmup multiplier derived from Bayesian posterior uncertainty.
///
/// DERIVATION (Information-theoretic):
/// - Wider credible interval → more parameter uncertainty → higher risk
/// - CI width normalized by mean gives dimensionless uncertainty measure
/// - Linear scaling: gamma_mult = 1 + ci_width / scaling_factor
/// - scaling_factor=10 chosen so CI width of 1.0 → mult of 1.1
///
/// At warmup start: ci_width ≈ 1.0 → mult ≈ 1.1
/// After convergence: ci_width ≈ 0.3 → mult ≈ 1.03
pub fn warmup_multiplier_from_ci(&self, kappa_ci_width: f64) -> f64 {
    if !self.enable_warmup_gamma_scaling {
        return 1.0;
    }
    let scaling_factor = 10.0;
    (1.0 + kappa_ci_width / scaling_factor).clamp(1.0, self.max_warmup_gamma_mult)
}
```

**New field in market_params.rs:**
```rust
/// Kappa credible interval width (normalized by mean).
/// ci_width = (ci_95_upper - ci_95_lower) / kappa_mean
pub kappa_ci_width: f64,  // Default: 1.0 (high uncertainty)
```

**Wired in glft.rs (~line 139):**
```rust
let warmup_scalar = if market_params.kappa_ci_width > 0.0 {
    cfg.warmup_multiplier_from_ci(market_params.kappa_ci_width)
} else {
    cfg.warmup_multiplier(market_params.adaptive_warmup_progress)
};
```

**Populated in aggregator.rs (~line 665):**
```rust
kappa_ci_width: {
    let (ci_lower, ci_upper) = est.hierarchical_kappa_ci_95();
    let kappa_mean = est.kappa();
    if ci_upper > ci_lower && kappa_mean > 0.0 {
        (ci_upper - ci_lower) / kappa_mean
    } else {
        1.0
    }
},
```

**Status:** ✅ Dynamic CI-based scaling replaces static 1.1 multiplier.

## Files Modified (Priority 2)

| File | Changes |
|------|---------|
| `src/market_maker/strategy/risk_config.rs` | Documented toxic_hour and book_depth derivations; added `warmup_multiplier_from_ci()` |
| `src/market_maker/strategy/market_params.rs` | Added `kappa_ci_width: f64` field |
| `src/market_maker/strategy/glft.rs` | Wired CI-based warmup_scalar |
| `src/market_maker/strategy/params/aggregator.rs` | Computed `kappa_ci_width` from Bayesian posterior |

## Remaining Items for Future Work

### Priority 3: Document Deep Constants
1. `0.5 × leverage` (margin sizing) - market_params.rs:1181
2. `50 bps` jump premium cap - evaluate removal
3. `0.0001` funding threshold

## Key Principle
**Target pattern:**
```rust
// If conditions warrant wider spreads, adjust γ with mathematical justification
let gamma_adjusted = gamma_base * principled_scaling_factor(uncertainty, conditions);
// Let GLFT do its job
let optimal_spread = glft_formula(gamma_adjusted, kappa);
// Trust the math
```

**Avoid anti-pattern:**
```rust
// Math says X, but we don't trust it, so multiply by Y
let actual_spread = optimal_spread * arbitrary_multiplier;
```
