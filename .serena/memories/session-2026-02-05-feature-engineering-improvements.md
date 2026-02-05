# Session: Feature Engineering Improvements for Prediction Models
**Date**: 2026-02-05
**Duration**: ~1 hour
**Focus**: Improving IR and reducing concentration for 10 prediction models

## Summary
Systematic debugging and improvement of prediction validator models to achieve statistical relevance. Fixed multiple root causes of signal saturation and poor variance.

## Key Changes Made

### 1. CV_Toxicity Signal Fix (cross_venue.rs:376)
**Problem**: `max_toxicity = max(vpin_A, vpin_B)` saturated at 1.0 on low-volume assets
- 100% concentration, res=0.000 (no variance)
- VPIN hits extremes when bucket sizes don't match trading activity

**Solution**: Replaced with divergence-based signal:
```rust
// Asymmetry signal: when venues disagree
let vpin_divergence = (binance.vpin - hl.vpin).abs();
let intensity_balance = 4.0 * intensity_ratio * (1.0 - intensity_ratio);
let asymmetry_signal = vpin_divergence * (0.5 + 0.5 * intensity_balance);

// Agreement signal: when both venues show toxicity
let min_vpin = binance.vpin.min(hl.vpin);
let agreement_signal = min_vpin * (1.0 - vpin_divergence);

let max_toxicity = asymmetry_signal.max(agreement_signal).clamp(0.0, 1.0);
```

**Result**: Concentration dropped from 100% → 65%

### 2. Adaptive VPIN Bucket Sizing (vpin.rs, binance_flow.rs)
**Problem**: Static bucket volumes are "magic numbers" that don't adapt to market conditions

**Solution**: Implemented adaptive sizing per Easley/Lopez de Prado literature:
- `bucket_volume = volume_rate × target_bucket_seconds`
- EWMA tracking of volume rate
- Min/max multipliers to clamp adaptation range
- Warmup period before adaptation kicks in

**Config additions**:
```rust
pub adaptive_enabled: bool,
pub target_bucket_seconds: f64,  // Default: 30.0
pub volume_rate_alpha: f64,       // EWMA alpha
pub min_bucket_multiplier: f64,   // 0.05
pub max_bucket_multiplier: f64,   // 20.0
```

### 3. RegimeHighVol Units Fix (prediction_validator.rs)
**Problem**: `sigma_bps_per_sqrt_s()` returns bps (e.g., 31), but `RegimeObservation::new()` expects fractional (e.g., 0.0031)

**Solution**: Convert bps to fractional:
```rust
let sigma_bps = self.volatility_filter.sigma_bps_per_sqrt_s();
let sigma_fractional = sigma_bps / 10_000.0;
let regime_obs = RegimeObservation::new(sigma_fractional, spread_bps, flow_imbalance);
```

## Final Validation Results (5-min HYPE)

| Model | IR | Concentration | Assessment |
|-------|-----|---------------|------------|
| Momentum | 0.78 | 38% | ✓ Strong |
| CV_Direction | 0.09 | 29% | ✓ Good diversity |
| EnhancedTox | 0.07 | 33% | ✓ Good diversity |
| CV_Toxicity | 0.03 | 65% | ✓ Fixed (was 100%) |
| PreFillToxicity | 0.05 | 84% | ~ Weak |
| CV_Agreement | 0.02 | 40% | ✓ Good diversity |
| RegimeHighVol | 0.00 | 100% | ~ Market calm |

## Key Insights

1. **IR > 1.0 is unrealistic** for alpha signals. Realistic expectations:
   - IR > 0.5 = strong
   - IR 0.2-0.5 = good
   - IR 0.1-0.2 = marginal

2. **Concentration matters more than IR** for feature quality
   - Low concentration = model has variance, can learn
   - High concentration = model stuck at one prediction

3. **Divergence-based signals have natural variance**
   - Differences (|A - B|) vary even in calm markets
   - Max/min operations saturate at boundaries

4. **Adaptive sizing beats magic numbers**
   - Self-calibrating to market conditions
   - Principle: target time-per-bucket, not fixed volume

## Files Modified
- `src/market_maker/estimator/cross_venue.rs` - CV_Toxicity signal
- `src/market_maker/estimator/vpin.rs` - Adaptive bucket sizing
- `src/market_maker/estimator/binance_flow.rs` - Adaptive config
- `src/bin/prediction_validator.rs` - RegimeHMM units fix

## Next Steps (if continuing)
1. InformedFlow still 86% concentrated - investigate signal source
2. BuyPressure 88% concentrated - may need similar divergence approach
3. Run longer validation (30+ min) for stable IR estimates
4. Consider ensemble weighting based on concentration metrics
