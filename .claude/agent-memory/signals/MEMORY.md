# Signals Agent Memory

## Signal Wiring Audit (2026-02-07)

### Critical Unwired Methods
- `signal_integrator.on_trade()` (feeds InformedFlow EM) - NEVER called anywhere
- `signal_integrator.on_fill()` (feeds RegimeKappa + model gating kappa) - NEVER called
- `signal_integrator.set_regime_probabilities()` / `set_regime()` - NEVER called
- `signal_integrator.update_as_prediction()` / `update_informed_prediction()` / `update_lead_lag_prediction()` - NEVER called (model gating stays at initial weights=1.0)

### BuyPressure EWMA Z-Score Bug
`enhanced_flow.rs:1325-1332` - EWMA mean updated BEFORE z-score computed (same bug class as PreFillToxicity Feb 6 fix). Z-score systematically compressed ~10%.

### Key File Locations
- `signal_integration.rs` - Central hub, all signal composition (STRATEGY TEAM OWNS)
- `estimator/enhanced_flow.rs` - BuyPressureTracker (line 1241+), EnhancedFlowEstimator
- `estimator/informed_flow.rs` - InformedFlowEstimator (EM decomposition)
- `estimator/regime_kappa.rs` - RegimeKappaEstimator (4-regime Bayesian kappa)
- `estimator/lag_analysis.rs` - LagAnalyzer (MI-based lead-lag)
- `calibration/model_gating.rs` - ModelGating (IR-based per-signal gating)
- `adverse_selection/pre_fill_classifier.rs` - PreFillASClassifier (z-scored toxicity)
- `adverse_selection/enhanced_classifier.rs` - EnhancedASClassifier (microstructure-based)

### Model Gating Behavior When Unfed
- `cached_weights` starts at `ModelWeights::full()` (all 1.0)
- `maybe_update_weights()` only triggered by `update_*` calls
- Since those are never called, weights stay at 1.0 forever
- Result: gating is a no-op, all signals pass through at full weight

### Verified Healthy
- PreFillToxicity z-score fix (Feb 6) - correctly computes z BEFORE EWMA update
- LeadLag MI significance test - correctly gates on null distribution
- CrossVenue flow analysis - properly wired via `on_binance_trade()`
- VPIN blending - correctly integrated with EM toxicity

### Cross-Venue Signal / Binance Feed Fix (2026-02-08)
- **Bug**: `--binance-symbol` defaulted to `btcusdt` for ALL assets. Trading HYPE fed BTC noise into lead-lag + cross-venue.
- **Fix**: Added `resolve_binance_symbol()` in `infra/binance_feed.rs` — maps assets to Binance Futures symbols.
- Known assets (BTC, ETH, SOL, etc.) auto-resolve; HL-native (HYPE, PURR) return None → feed disabled.
- Explicit `--binance-symbol` override still available for custom mappings.
- Graceful degradation: `CrossVenueFeatures::default()` = neutral (0.0 confidence, 1.0 spread mult), `LeadLagSignal::default()` = not actionable.
- Changed CLI args in `market_maker.rs`, `paper_trader.rs`, `prediction_validator.rs` from `String` to `Option<String>`.
- 5 new tests, all 16 existing binance_feed tests pass.
