# Learning Feedback Loops — Detailed Reference

## Loop 1: Kappa from Own Fills

**Trigger**: `on_own_fill(timestamp, placement_price, fill_price)` in `estimator/mod.rs`
**Updates**: Hawkes process intensity parameters (baseline lambda, self-excitation alpha, decay beta)
**Purpose**: Estimates order book depth decay rate (kappa) from observed fill frequency
**Integration**: `handlers.rs:~1051` — called on every fill event

## Loop 2: AS Markout Queue

**Trigger**: Fill event → pending fill outcome added to queue
**Updates**: `pending_fill_outcomes` VecDeque in handlers
**Purpose**: Tracks each fill for future markout evaluation (5s horizon)
**Integration**: `handlers.rs` — fill pushed to pending queue immediately

## Loop 3: AS Outcome Feedback

**Trigger**: Markout timer expires (5s after fill)
**Updates**: `pre_fill_classifier.record_outcome(is_buy, was_adverse, magnitude_bps)`
**Purpose**: Labels fills as adverse/non-adverse based on subsequent price movement
**Integration**: `handlers.rs:303-307` — resolved when markout price available

## Loop 4: Calibration Progress

**Trigger**: Periodic timer (every N quote cycles)
**Updates**: `calibration_controller` — Brier scores, IR, conditional calibration per model
**Purpose**: Tracks whether models are improving, degrading, or drifting
**Integration**: `handlers.rs` — periodic update cycle

## Loop 5: Sigma Update

**Trigger**: Trade events, L2 book updates
**Updates**: Realized volatility estimator (EWMA of squared returns)
**Purpose**: Feeds sigma into GLFT formula (`0.5 * gamma * sigma^2 * T`)
**Integration**: `handlers.rs` — on trade and book events

## Loop 6: Regime Update

**Trigger**: Trade events, L2 book updates
**Updates**: HMM belief state (4 states: Calm, Normal, Volatile, Extreme)
**Purpose**: Determines regime-dependent parameters (gamma mult, risk premium)
**Integration**: `handlers.rs` — on trade and book events

## Loop 7: Quote Outcome Tracking

**Trigger**: Fill event OR 30s expiry timeout
**Updates**: `QuoteOutcomeTracker` — BinnedFillRate with Beta posteriors per spread bin
**Purpose**: Unbiased fill rate estimation across all spread levels (includes unfilled quotes)
**Integration**: `learning/quote_outcome.rs` — `on_fill()` and `expire_old_quotes()`

**Detail**: 8 fine bins [0,2), [2,4), ..., [20,inf) + 4 coarse bins for hierarchical shrinkage.
Shrinkage weight `w = 5.0 / (1.0 + n_fine / 50.0)` decays as fine bin accumulates data.

## Loop 8: Spread Bandit Update

**Trigger**: Fill event with realized edge
**Updates**: `SpreadBandit` — Normal-Gamma posterior for the (context, arm) cell that produced the fill
**Purpose**: Learns which spread multiplier (0.85-1.40) maximizes edge in each market context
**Integration**: `handlers.rs:1205-1214` — `update_from_pending(baseline_adjusted_reward)`

**Detail**: Reward = `realized_edge_bps - ewma_baseline`. Baseline subtraction centers rewards
around zero so the bandit can distinguish good arms from bad (without subtraction, all arms
return ~-1.5 bps fees, making learning impossible).

## Loop 9: Ensemble Weight Update

**Trigger**: Fill event (same as Loop 8)
**Updates**: `AdaptiveEnsemble` — IR and Brier Score for the "SpreadBandit" model entry
**Purpose**: Tracks whether the bandit is adding value vs other edge models
**Integration**: `handlers.rs:1210-1214` — `update_performance("SpreadBandit", edge, brier, ts)`

**Detail**: Softmax over IR with temperature parameter. Low temperature → exploit best model.
High temperature → explore uniformly. Water-filling ensures all models maintain minimum weight.

## Interaction Between Loops

```
Loops 5+6 (sigma, regime) → inform SpreadBandit context selection
Loop 8 (bandit) → produces spread → determines Loop 7 outcomes
Loop 7 (outcomes) → fills optimal_spread_bps() → diagnostic input
Loop 3 (AS feedback) → informs kappa_bid/kappa_ask asymmetry
Loop 9 (ensemble) → weights bandit contribution to final spread
Loop 1 (kappa) → updates GLFT formula input → affects all spread levels
Loop 4 (calibration) → may trigger model gating → affects risk premium
```
