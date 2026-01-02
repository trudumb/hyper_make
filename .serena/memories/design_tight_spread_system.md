# Design: Tight Spread Quoting System

**Status**: Design Complete
**Date**: 2026-01-01
**Location**: `docs/design/tight_spread_system.md`

## Overview

System to enable 3-5 bps spreads during calm markets while maintaining 8 bps during adverse conditions.

## New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| MarketRegimeDetector | `estimator/regime.rs` | Classify Calm/Normal/Volatile/Cascade |
| ToxicityPredictor | `estimator/toxicity.rs` | Forecast P(adverse selection) |
| DynamicSpreadFloor | `strategy/dynamic_floor.rs` | Regime-based floor calculation |
| TightSpreadGate | `strategy/tight_gate.rs` | Binary can-quote-tight decision |

## Key Interfaces

### MarketRegime
```rust
enum MarketRegime { Calm, Normal, Volatile, Cascade }
// Calm → 3 bps, Normal → 5 bps, Volatile → 8 bps, Cascade → pull
```

### TightSpreadConfig (Default Values)
```rust
enabled: false,              // Must opt-in
calm_floor_bps: 3.0,
normal_floor_bps: 5.0,
volatile_floor_bps: 8.0,
sigma_calm_ratio: 0.7,       // σ < 0.7×baseline = calm
sigma_volatile_ratio: 2.0,   // σ > 2×baseline = volatile
jump_ratio_threshold: 1.5,
toxicity_tight_threshold: 0.1,
toxic_hours: [6, 7, 14],
max_inventory_for_tight: 0.3,
```

## Algorithms

### Regime Detection
- Cascade: cascade_severity > 0.5
- Calm: vol_ratio < 0.7 AND jump_ratio < 1.2 AND |book_imb| < 0.3
- Volatile: vol_ratio > 2.0 OR jump_ratio > 1.5 OR |book_imb| > 0.6
- Normal: else

### Toxicity Scoring
- book_imbalance: 0-0.25
- flow_imbalance: 0-0.25
- toxic_hour: 0.20 (if hour in [6,7,14])
- large_trades: 0-0.20
- volatility: 0-0.10
- Total: 0.0 (safe) to 1.0 (very toxic)

### Tight Quote Gate
```
can_quote_tight = ALL(
  config.enabled,
  regime == Calm,
  regime_confidence > 0.7,
  toxicity < 0.1,
  toxicity_confidence > 0.6,
  hour NOT IN toxic_hours,
  |inventory| < 30% max
)
```

## Integration Points

1. **ParameterEstimator**: Add regime + toxicity detection
2. **MarketParams**: Add regime, toxicity, dynamic_floor fields
3. **LadderStrategy**: Use dynamic_floor instead of static
4. **DynamicDepthGenerator**: Apply regime gamma multiplier

## Metrics

- mm_market_regime{regime="..."}
- mm_toxicity_score
- mm_dynamic_floor_bps
- mm_tight_quote_active
- mm_fills_by_regime{regime="..."}
- mm_adverse_selection_by_regime{regime="..."}

## Rollout

1. Metrics Only (1 week): Log regime/toxicity, don't use
2. Shadow Mode (1 week): Calculate dynamic floor, log only
3. Gradual (2 weeks): Enable with calm_floor=5 bps
4. Full: Lower calm_floor to 3 bps

## Risk Mitigation

- Default to Normal on low confidence
- Kill switch: disable if AS in calm > 3 bps
- Alert on regime detection staleness
