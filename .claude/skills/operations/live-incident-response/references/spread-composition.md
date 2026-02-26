# Spread Composition Chain

The final spread multiplier is the product of up to seven independent multiplicative factors,
applied at `src/market_maker/orchestrator/quote_engine.rs` lines 992-1107.

## The Chain

```
spread_multiplier = circuit_breaker
                  * threshold_kappa
                  * model_gating
                  * staleness
                  * toxicity
                  * defensive
                  * risk_overlay
                  (capped at max_composed_spread_mult, default 10.0x)
```

## Factor Details

| # | Factor | Variable | Source | Lines | Range |
|---|--------|----------|--------|-------|-------|
| 1 | Circuit breaker | `breaker_action` match | `CircuitBreakerAction::WidenSpreads { multiplier }` from `risk/circuit_breaker.rs:47-64` | 992-996 | 1.0x or action-specific (OI=cancel, funding=1.5x, spread=2.0x, fill=1.5x) |
| 2 | Threshold kappa | `threshold_kappa_mult` | `self.stochastic.threshold_kappa.regime().spread_multiplier()` | 998-1009 | 1.0x (normal) to ~2.0x (momentum regime) |
| 3 | Model gating | `model_gating_mult` | `signal_integrator.model_gating_spread_multiplier()` at `signal_integration.rs:978` | 1011-1019 | 1.0x (well-calibrated) to 2.0x (uncalibrated) |
| 4 | Staleness | `staleness_mult` | `signal_integrator.staleness_spread_multiplier()` at `signal_integration.rs:999` | 1021-1029 | 1.0x (fresh), 1.5x (1 stale signal), 2.0x (2+ stale signals) |
| 5 | Toxicity | `toxicity_mult` | `self.tier2.toxicity.evaluate(&toxicity_input).spread_multiplier` from `analytics/market_toxicity.rs` | 1031-1071 | 1.0x (calm) to ~3.0x (high VPIN + informed flow) |
| 6 | Defensive | `defensive_mult` | `self.tier2.edge_tracker.max_defensive_multiplier()` at `analytics/edge_metrics.rs:192` | 1073-1083 | 1.0x (positive gross edge) to 5.0x (mean gross edge <= -3 bps) |
| 7 | Risk overlay | `risk_overlay.spread_multiplier` | `control/mod.rs:775-784` `RiskAssessment` struct | 1085-1093 | 1.0x (normal) to 3.0x (elevated risk from controller) |

## Global Cap

**Line 1096-1097**: `spread_multiplier = spread_multiplier.min(max_composed)` where
`max_composed` comes from `self.tier2.toxicity.config().max_composed_spread_mult`.
Default value: **10.0x** (set at `analytics/market_toxicity.rs:98,121`).

## Application

**Line 1100-1101**: `market_params.spread_widening_mult *= spread_multiplier`

The composed multiplier is applied to `spread_widening_mult` on the `MarketParams` struct,
which the GLFT strategy uses to widen the base optimal spread.

## Logging

**Lines 1102-1107**: All components are logged together:
```rust
toxicity = format!("{:.2}x", toxicity_mult),
defensive = format!("{:.2}x", defensive_mult),
staleness = format!("{:.2}x", staleness_mult),
model_gating = format!("{:.2}x", model_gating_mult),
total = format!("{:.2}x", spread_multiplier),
```

Note: circuit_breaker, threshold_kappa, and risk_overlay are logged separately at their
respective application points.

## Compounding Example

A realistic scenario where multiple factors are mildly elevated:

| Factor | Value | Running Product |
|--------|-------|----------------|
| circuit_breaker | 1.0x | 1.0x |
| threshold_kappa | 1.3x | 1.3x |
| model_gating | 1.4x | 1.82x |
| staleness | 1.5x | 2.73x |
| toxicity | 1.2x | 3.28x |
| defensive | 1.0x | 3.28x |
| risk_overlay | 1.0x | 3.28x |

No single factor exceeds 1.5x, but the product is 3.28x -- too wide for fills on
most HIP-3 assets. This is the multiplicative cascade problem.
