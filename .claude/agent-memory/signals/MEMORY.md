# Signals Agent Memory

## Current Open Issues
- `signal_integrator.on_trade()` — feeds InformedFlow EM, wiring status unknown post-redesign
- `signal_integrator.on_fill()` — feeds RegimeKappa + model gating kappa, verify wired
- Model gating `update_*` calls may still be unwired → weights stay at 1.0 (gating = no-op)

## Key File Locations
- `signal_integration.rs` — central signal hub (STRATEGY TEAM OWNS)
- `estimator/enhanced_flow.rs` — BuyPressureTracker, EnhancedFlowEstimator
- `estimator/informed_flow.rs` — InformedFlowEstimator (EM decomposition)
- `estimator/regime_kappa.rs` — RegimeKappaEstimator (4-regime Bayesian)
- `estimator/lag_analysis.rs` — LagAnalyzer (MI-based lead-lag)
- `calibration/model_gating.rs` — ModelGating (IR-based per-signal gating)
- `adverse_selection/pre_fill_classifier.rs` — PreFillASClassifier
- `adverse_selection/enhanced_classifier.rs` — EnhancedASClassifier

## Active Gotchas
- EWMA update-before-compute bug class: always compute z BEFORE EWMA update
- Binance symbol auto-resolution: HL-native assets (HYPE, PURR) return None → feed disabled
- `CrossVenueFeatures::default()` = neutral (0.0 confidence, 1.0 spread mult)
