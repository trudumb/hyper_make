# Session: 2026-02-06 — Paper Trader Learning Loops Wired

## Context
Deep audit of paper_trader revealed that all feedback loops were broken. The system was running blind with default priors — no learning from fills, no adverse selection tracking, no calibration progression.

## Critical Findings (Infrastructure Audit)

### Working Components (3/10)
- **Volatility (bipower)**: Self-calibrates correctly
- **Regime HMM**: Auto-calibrates at 500 obs, beliefs updating
- **VPIN**: Bucket rolling active

### Broken Components (7/10)
| Component | Root Cause |
|-----------|-----------|
| Kappa (own fills) | `estimator.on_own_fill()` NEVER called in `on_simulated_fill()` |
| Pre-Fill Classifier | `record_outcome()` NEVER called — `learning_samples=0` |
| Calibration Progress | Hardcoded to `0.0` in `build_market_params()` |
| Gamma adaptation | Constant `0.480` — never adjusts to regime |
| Optimal spread | Stuck at floor `16.02 bps` — never adapts |
| Lead-lag | No Binance feed connected in paper trader |
| Informed flow / Cross-venue | No training data, producing zeros |

## Fixes Implemented (all in `src/bin/paper_trader.rs`)

### Fix 1: Kappa from Own Fills
- `estimator.on_own_fill()` now called in `on_simulated_fill()`
- Feeds: timestamp_ms, fill_price, fill_size, is_buy
- Populates: own_kappa, directional kappa, hierarchical_kappa, kappa_orchestrator, AS decomposition, fill_rate_model, edge_surface

### Fix 2: Adverse Selection Outcome Feedback
- Added `PendingFillOutcome` struct and deque to `SimulationState`
- In `on_simulated_fill()`: queue fill with mid_at_fill timestamp
- In `update_mid()`: call `check_adverse_selection_outcomes()`
- After 5s markout: compute `was_adverse` (mid moved against us >1bps)
- Call `pre_fill_classifier.record_outcome(is_bid, was_adverse, Some(magnitude_bps))`

### Fix 3: CalibrationController Wired
- Added `CalibrationController` field to `SimulationState`
- Config: `enabled=true, target_fill_rate=60/hr, min_gamma_mult=0.3`
- `record_fill()` called on each simulated fill
- `update_calibration_status(as_fills, kappa_conf)` called every quote cycle
- `build_market_params()` reads real values instead of hardcoded zeros

### Diagnostic Logging Added
- `[LEARNING]` line in periodic signal health: fills, kappa_conf, as_samples, cal_progress, cal_gamma
- `[FILL FEEDBACK]` line on each fill: oid, side, price, depth_bps, total_fills, kappa_conf
- `[AS OUTCOME]` line on adverse fills: side, magnitude, mid comparison

## Validation Results
- `cal_progress=52%` (was 0%) — from kappa_confidence=0.71 alone
- `cal_gamma=0.67` (was 1.0) — tighter quotes during warmup
- `as_samples=1` (was 0) — classifier receiving outcome data
- `kappa_effective` dynamically changing (2694→1875) from robust estimation
- Checkpoint persistence: `pre_fill_samples=1 kappa_own_obs=1`

## Earlier Fixes (Same Session)
1. **Fill simulator queue model**: Added `ignore_book_depth: bool` to `FillSimulatorConfig`. Paper trader sets `true` → flat queue model instead of real L2 depth.
2. **Order churn**: `price_threshold_bps` changed 2.0→5.0. HYPE at $33 has 3bps/tick, old threshold caused cancel every tick.
3. **Margin bug** (from prior session): `MarginAwareSizer::update_state()` was never called.

## Import Path
`CalibrationController` re-exported via `market_maker::estimator::mod.rs` → `market_maker::mod.rs` (`pub use estimator::*`).
Import as: `use hyperliquid_rust_sdk::market_maker::{CalibrationController, CalibrationControllerConfig};`

## Remaining Work
- **Fill rate still low for HYPE in quiet markets** — learning loops wired but need fills to activate
- **Lead-lag signal**: Needs Binance feed connected in paper_trader
- **Informed flow / Cross-venue**: Need training data sources
- **Competitive viability analysis**: Fees vs edge assessment pending
- **Longer validation run**: Need sustained run with fills to confirm kappa updates from own_fills
