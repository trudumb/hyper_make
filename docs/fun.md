# Principled Foundations: What This System Would Look Like If We Got It Right From the Start

## Context

12 minutes of live mainnet trading on HYPE validator perps with $100 capital. 11 fills, position swung between -0.77 and +5.14 HYPE. The system technically functioned -- quotes were placed, fills occurred, position moved -- but every subsystem revealed a gap between its design assumptions and the reality of small-capital, low-fill-rate live trading.

This document examines each issue not as a bug to patch, but as a missing foundational assumption. For each concern: what does the current code actually do, why does it fail, and what would the system look like if the correct assumption had been there from the start.

---

## Priority Map

| # | Issue | Impact | Effort | Priority |
|---|-------|--------|--------|----------|
| 8 | Floor binding kills GLFT adaptiveness | Fatal -- GLFT output is irrelevant | Medium | P0 |
| 6 | Position sizing chain too restrictive | Fatal -- empty ladders at $100 | Medium | P0 |
| 2 | Regime HMM too trigger-happy | Fatal -- Extreme at 200 obs kills quoting | Medium | P0 |
| 9 | Ladder generation fragile at small size | Fatal -- all levels below min_notional | Low | P0 |
| 10 | Pricing precision errors | Directly loses fills | Low | P0 |
| 7 | StochasticConfig TOML is dead | Wastes operator time, false confidence | Low | P1 |
| 1 | AS model never warms up | Runs unprotected for first hour | High | P1 |
| 3 | Fill processing invisible | Can't diagnose live issues | Low | P1 |
| 4 | Sharpe meaningless at N<50 | Misleading operator metrics | Low | P2 |
| 5 | Realized PnL not visible | Can't determine profitability | Medium | P2 |

---

## Issue 1: Adverse Selection Model Never Warms Up In Time

### Current State

The AS system has two components:

1. **AdverseSelectionEstimator** (`adverse_selection/mod.rs`): Needs observed fills to compute markout-based AS estimates. The `fills_measured()` count drives calibration_controller gating.

2. **PreFillASClassifier** (`adverse_selection/pre_fill_classifier.rs`): A 6-feature classifier (imbalance, flow, regime, funding, changepoint, trend) that predicts toxicity before fills happen. It has a warmup prior (lines 707-716):

```rust
if self.update_count < self.config.warmup_prior_min_updates {
    let prior_weight = 1.0 - (self.update_count as f64 / self.config.warmup_prior_min_updates as f64);
    data_weight * classifier_tox + prior_weight * self.config.warmup_prior_toxicity
}
```

The warmup_prior_toxicity provides mild protection, but the main AS estimator (`as_warmed_up` in MarketParams) stays false until enough fills accumulate. With HYPE getting maybe 5-10 fills per hour, this means AS is effectively disabled for the entire first trading hour of every session.

### Why It Fails

The foundational assumption is wrong: AS warmup is gated on *own fills*, but own fills are the scarcest signal in the system. A market maker on an illiquid token might get 5 fills/hour. Waiting for 50 fills means 10 hours of unprotected quoting.

### Principled Solution: Market-Data-Driven AS From First Tick

If AS protection were a foundational assumption, it would never depend on own fills for the initial estimate.

**Architecture**: Two-tier AS with immediate coverage.

```
Tier 1 (Immediate, from market data):
  - Trade flow toxicity from public trades (VPIN, trade imbalance)
  - Book imbalance asymmetry (bid depth vs ask depth)
  - Cross-venue price divergence (Binance leads)
  - Funding rate extremes (predictable flow direction)

Tier 2 (Calibrated, from own fills):
  - Markout-based AS at 5s horizon (current system)
  - Depth-dependent AS decay curve (DepthDecayAS)
  - Fill-conditional posterior updates
```

**Concrete changes**:

- **`adverse_selection/mod.rs`**: Add `market_data_as_estimate(flow_imbalance, book_imbalance, vpin, funding_rate) -> f64` that returns a conservative AS estimate from public data alone. No fills required.
- **`strategy/market_params.rs`**: Change `as_warmed_up: bool` to `as_confidence: f64` on a [0, 1] scale. At 0.0, use market-data-only estimate. At 1.0, use full markout-calibrated model. Linear blend in between.
- **`strategy/ladder_strat.rs`**: Replace the binary `if market_params.as_warmed_up` check with `as_confidence`-weighted blend between prior and calibrated AS.
- **`orchestrator/quote_engine.rs`**: Wire `market_toxicity_scorer` (already exists in `analytics/market_toxicity.rs`) as the Tier 1 AS input from session start.

**Key insight**: The system already computes VPIN, trade flow imbalance, and book imbalance for other purposes. The only missing piece is routing these into the AS estimate during warmup, instead of treating AS as binary warmed/not-warmed.

---

## Issue 2: Regime Classifier (HMM) Is Too Trigger-Happy

### Current State

`estimator/regime_hmm.rs` implements a 4-state HMM (Low/Normal/High/Extreme) with auto-calibration:

```rust
calibration_buffer_size: 200,   // line 438
initial_calibration_done: false, // line 439
```

At 200 observations, `calibrate_from_observations()` fires (line 562-582). It computes percentile-based emission thresholds from whatever data it has seen. The problem: 200 observations of a quiet market produces a very narrow distribution. The 95th percentile of a quiet window might be what's actually *normal* volatility. So when normal trading resumes, the HMM classifies it as Extreme.

The Extreme regime sets `max_position_fraction: 0.3` (regime_state.rs line 90). At $100 capital with config max_position ~3.14 HYPE, this drops effective_max_position to 0.94 HYPE. The multiplicative chain in quote_engine.rs (line 1268-1282) then applies:

```rust
let new_effective = margin_effective
    .min(self.config.max_position)
    * market_params.max_position_fraction.clamp(0.1, 1.0);
```

0.94 HYPE at $25/HYPE = $23.50 total budget. Split across 10 ladder levels = $2.35/level. Min notional is $10. All levels fail. Empty ladder.

### Why It Fails

200 observations is about 3 minutes of data on a token with ~1 trade/second. The HMM calibrates its entire worldview from 3 minutes of one market condition and then treats everything else as extreme. The auto-calibration is doing exactly what it was told to do -- the problem is that 200 observations is far too few to establish a credible distribution.

### Principled Solution: Conservative Priors That Resist Premature Calibration

If "everything is regime-dependent" were truly foundational, the regime classifier would be designed to be *harmless until proven helpful*.

**Architecture**: Prior-dominated regime with graduated authority.

```
Phase 1 (0-1000 obs): Prior-only regime
  - Use with_baseline_volatility() from session start (already exists, line 483)
  - Regime probabilities reflect prior, not data
  - max_position_fraction always >= 0.8 (no tightening from uncertain classifier)

Phase 2 (1000-5000 obs): Blended regime
  - Auto-calibration fires at 1000 (not 200)
  - Regime tightening capped: max_position_fraction >= 0.5
  - Log regime transition reasons at INFO level for diagnostics

Phase 3 (5000+ obs): Full authority
  - Current behavior: regime controls max_position_fraction fully
  - Recalibration interval: 2000 (current 500 is fine for this phase)
```

**Concrete changes**:

- **`estimator/regime_hmm.rs`**:
  - Change `calibration_buffer_size` from 200 to 1000.
  - Add `authority: f64` field that starts at 0.0 and grows with `min(1.0, observation_count / 5000.0)`.
  - `calibrate_from_observations()` should require at least 500 observations (not 100, line 823).
  - Add `fn damped_max_position_fraction(&self) -> f64` that returns `1.0 - authority * (1.0 - raw_fraction)`. At 200 obs, authority ~0.04 so Extreme only reduces by ~3% instead of 70%.

- **`strategy/regime_state.rs`**: Add `RegimeParams::damped(authority: f64) -> RegimeParams` that interpolates between Calm (safe defaults) and the regime's actual parameters based on authority.

- **`orchestrator/quote_engine.rs`**: Replace direct `market_params.max_position_fraction` usage (line 1271) with `regime_hmm.damped_max_position_fraction()`.

- **Startup**: Call `hmm.with_baseline_volatility(config_sigma, config_spread_bps)` at initialization so the HMM starts with asset-appropriate priors. The code for this exists (line 483-516) but is not called in the main initialization path.

**Key insight**: The current HMM has `with_baseline_volatility()` which sets `initial_calibration_done = true` and skips auto-calibration entirely. This should be the default path for every asset, with auto-calibration being a refinement, not an override.

---

## Issue 3: Fill Processing Is Invisible

### Current State

Fill processing in `messages/user_fills.rs` logs at `debug!` level (line 143):

```rust
debug!(
    asset = %ctx.asset,
    processed = result.fills_processed,
    skipped = result.fills_skipped,
    "UserFills processed"
);
```

The `FillProcessor::process()` in `fills/processor.rs` only logs at `info!` level for edge cases (immediate fill dedup, cancel-fill race). Normal resting-order fills that update position are processed silently at debug level. The position tracker updates internally but emits no log.

### Why It Fails

In production, operators run at `info` log level. They see quote placements, risk assessments, spread diagnostics -- but never "bought 0.3 HYPE at $25.12, position now 1.7 HYPE, realized PnL +$0.08". This makes live monitoring impossible and debugging requires log-level changes that produce enormous output.

### Principled Solution: Fills Are First-Class Events

If fills were treated as the most important events in the system (they are -- they're the only events that move real money), every fill would produce a structured INFO log.

**Architecture**: Single canonical fill log line.

```
[FILL] BOUGHT 0.30 HYPE @ $25.12 | mid=$25.15 | depth=1.2bps | pos: 1.4 -> 1.7 | rpnl=$0.08 | upnl=$0.23 | edge=3.1bps | session_fills=7
```

**Concrete changes**:

- **`fills/processor.rs`**: After position update succeeds (around line 773), add:

```rust
info!(
    "[FILL] {} {:.4} {} @ {:.4} | mid={:.4} | depth={:.1}bps | pos: {:.4} -> {:.4} | rpnl=${:.2} | edge={:.1}bps | session_fills={}",
    if fill.is_buy { "BOUGHT" } else { "SOLD" },
    fill.size,
    state.asset,
    fill.price,
    fill.mid_at_fill,
    depth_bps,
    old_position,
    state.position.position(),
    realized_pnl,
    edge_bps,
    self.fill_count,
);
```

- **`fills/processor.rs`**: Track `old_position` before `state.position.process_fill()` call. Compute `realized_pnl` from PnL tracker delta. Compute `edge_bps = (fill.price - fill.mid_at_fill) / fill.mid_at_fill * 10000.0` with sign adjustment for side.

- **`messages/user_fills.rs`**: Change the summary log from `debug!` to `info!` and add fill count and position summary.

This is ~20 lines of code. The information is already available in the FillState bundle; it just needs to be logged.

---

## Issue 4: Sharpe Ratio Is Meaningless at N<50

### Current State

`analytics/sharpe.rs` has `has_sufficient_data()` (line 129-131):

```rust
pub fn has_sufficient_data(&self) -> bool {
    self.returns.len() >= 30
}
```

The live analytics summary in `analytics/live.rs` appends " (insufficient data)" when this is false (line 162-166). But the Sharpe number itself is still computed and logged prominently:

```
Sharpe(all)=789.23 (insufficient data)
```

An operator sees "789" and feels good, missing the caveat.

### Why It Fails

Sharpe ratio is `mean / std * sqrt(annualization)`. With 11 fills, all in the same direction during a trending micro-moment, the std is tiny and the mean is positive. The resulting Sharpe is astronomical but meaningless. The "(insufficient data)" disclaimer is too subtle.

### Principled Solution: Replace Premature Sharpe With Fill-Count-Appropriate Metrics

**Architecture**: Tiered metrics by sample size.

```
N < 10:   "Too few fills for any statistics"
          Show: fill_count, gross_pnl, avg_fill_edge_bps
N < 50:   "Preliminary (wide confidence interval)"
          Show: avg_edge_bps, win_rate, avg_win/avg_loss, 90% CI on Sharpe
N >= 50:  "Reliable"
          Show: Full Sharpe with confidence interval (already computed)
```

**Concrete changes**:

- **`analytics/live.rs`**: In `maybe_log_summary()`, replace the single Sharpe log with tiered output:

```rust
if sharpe_summary.count < 10 {
    info!(
        "[ANALYTICS] Fills={} | Gross PnL={:.2} bps | Avg edge={:.1} bps | (too few fills for Sharpe)",
        sharpe_summary.count, gross_pnl_bps, mean_edge_bps,
    );
} else if sharpe_summary.count < 50 {
    info!(
        "[ANALYTICS] Fills={} | Sharpe={:.1} [CI: {:.1}, {:.1}] (PRELIMINARY) | Win rate={:.0}% | Edge={:.1} bps",
        sharpe_summary.count, sharpe_all, sharpe_lo, sharpe_hi, win_rate_pct, mean_edge_bps,
    );
} else {
    // Current full output
}
```

- **`analytics/sharpe.rs`**: Add `fn sharpe_or_none(&self) -> Option<f64>` that returns `None` below 30 samples instead of a number. Callers can pattern-match to handle the no-data case.

---

## Issue 5: Not Profitable (Or Can't Tell)

### Current State

The system tracks:
- **Theoretical edge** (markout-based): `EdgeTracker` in tier2 computes mean edge from 5-second markouts after fill placement. Shows "9.0 bps" which is the markout edge, not realized profit.
- **PnL tracker** (`tracking/pnl_tracker.rs`): Tracks realized and unrealized PnL internally but doesn't log it prominently.
- **Sharpe tracker**: Based on fill returns, but the return computation uses `fill_pnl_bps` which may or may not include fees.

What's missing: A clear periodic log line showing `realized_pnl`, `unrealized_pnl`, `total_pnl`, `fees_paid`, `net_edge_after_fees`.

### Principled Solution: PnL Is The Dashboard

**Concrete changes**:

- **`orchestrator/quote_engine.rs`**: In the periodic diagnostics section, add a PnL summary:

```rust
info!(
    "[PnL] realized=${:.4} unrealized=${:.4} total=${:.4} fees=${:.4} fills={} | net_edge={:.1}bps",
    realized_pnl, unrealized_pnl, total_pnl, total_fees, fill_count, net_edge_bps,
);
```

- **`tracking/pnl_tracker.rs`**: Expose `total_fees()` and `net_pnl()` methods. Track cumulative fees from fill events (fee data is available in UserFills but not currently extracted).

- **`analytics/live.rs`**: Add `pnl_summary()` that returns realized, unrealized, total, and fees.

---

## Issue 6: Position Sizing Chain Is Overly Restrictive For Small Capital

### Current State

The effective_max_position computation in quote_engine.rs (lines 1265-1282) is multiplicative:

```
effective_max_position = min(margin_derived, config.max_position)
                       * max_position_fraction      // regime: 0.3 for Extreme
```

Then in the ladder (ladder_strat.rs line 1131-1162):

```
MAX_MARGIN_UTILIZATION = 0.70    // Use 70% of available margin
```

Then per-level:

```
total_size / num_levels           // Split across 10 levels
```

Then min_notional filter (generator.rs line 601-614):

```
if bid_price * size >= min_notional  // $10 minimum
```

With $100 capital at 3x leverage: $300 buying power. 70% utilization = $210. Split across bids and asks = $105/side. Regime Extreme at 30% = $31.50/side. Across 10 levels = $3.15/level. Min notional is $10. **All levels fail.**

Even in Normal regime (80%): $300 * 0.7 * 0.5 * 0.8 / 10 = $8.40/level. Still below $10. Every level fails.

### Why It Fails

The system was designed for $10K+ capital where these multipliers are comfortable. At $100, the multiplicative chain compounds to make quoting impossible. The concentration fallback (generator.rs lines 620-660) catches the case where all levels fail min_notional by creating a single level at max(25% of total, min_notional), but this was only recently added and doesn't address the root cause.

### Principled Solution: Capital-Aware Ladder Sizing

If small capital were a foundational assumption, the system would work backwards from min_notional to determine the number of levels, not forward from a fixed level count.

**Architecture**: Adaptive level count.

```
1. Compute total_budget_per_side = margin * utilization * regime_fraction / 2
2. max_viable_levels = floor(total_budget_per_side / min_notional)
3. actual_levels = min(config.num_levels, max(1, max_viable_levels))
4. per_level_size = total_budget_per_side / actual_levels
```

**Concrete changes**:

- **`strategy/ladder_strat.rs`**: In `calculate_ladder()`, before calling `Ladder::generate()`, compute viable level count:

```rust
let budget_per_side = effective_liquidity * mid_price / 2.0;
let max_viable_levels = (budget_per_side / MIN_ORDER_NOTIONAL).floor() as usize;
let actual_levels = self.ladder_config.num_levels.min(max_viable_levels.max(1));

if actual_levels < self.ladder_config.num_levels {
    info!(
        budget_per_side = %format!("{:.2}", budget_per_side),
        min_notional = MIN_ORDER_NOTIONAL,
        config_levels = self.ladder_config.num_levels,
        actual_levels = actual_levels,
        "Capital-constrained: reducing ladder levels"
    );
}
```

- **`quoting/ladder/generator.rs`**: In `build_raw_ladder()`, change the concentration fallback to be the *default* behavior when total_size is small, not an error recovery path. If `total_size * mid_price < min_notional * 2`, skip the multi-level approach entirely and produce 1-2 levels at touch.

- **`orchestrator/quote_engine.rs`**: Add a `small_capital_mode` flag that activates when `account_value < 500`. In this mode:
  - `num_levels` auto-reduces to fit within budget
  - `MAX_MARGIN_UTILIZATION` increases to 0.85 (still safe with kill switch)
  - Regime max_position_fraction floor is 0.6 (never below 60% of max, since the max is already tiny)

- **`strategy/regime_state.rs`**: Add a `RegimeParams::for_small_capital()` variant that uses less aggressive tightening. Extreme would use `max_position_fraction: 0.6` instead of 0.3.

**Key insight**: The current system treats position limits as fractions of a large number. When the base is small, fractions-of-fractions converge to zero. The fix is to set absolute floors: `effective_max_position >= min_notional * 2 / mid_price` always, regardless of regime.

---

## Issue 7: StochasticConfig TOML Is Completely Dead

### Current State

`config/stochastic.rs` defines `StochasticConfig` with ~40 parameters (kelly_fraction, entropy settings, HJB controller, etc.). In `src/bin/market_maker.rs`, the AppConfig struct (line 397-410):

```rust
pub struct AppConfig {
    pub network: NetworkConfig,
    pub trading: TradingConfig,
    pub strategy: StrategyConfig,
    pub logging: LoggingConfig,
    pub monitoring: MonitoringAppConfig,
    pub kill_switch: KillSwitchAppConfig,
}
```

There is no `stochastic: StochasticConfig` field. The StochasticConfig is constructed directly in code (line 1621-1638):

```rust
let stochastic_config = {
    let mut cfg = StochasticConfig::default();
    // ... CLI overrides only ...
    cfg
};
```

Users can write a `[stochastic]` section in their TOML config file. It parses without error (serde ignores unknown sections by default) and has zero effect. Every parameter uses `StochasticConfig::default()`.

### Why It Fails

This is a configuration honesty problem. Users believe they're controlling the system when they edit TOML values. The system silently ignores their changes. This is particularly dangerous for risk parameters like `kelly_fraction`, `tight_quoting_max_toxicity`, or `hjb_terminal_penalty`.

### Principled Solution: Single Source of Truth for Configuration

**Option A (simple)**: Add `stochastic` to AppConfig and wire it through.

```rust
pub struct AppConfig {
    // ... existing fields ...
    #[serde(default)]
    pub stochastic: StochasticAppConfig,
}
```

Where `StochasticAppConfig` mirrors the TOML-friendly subset of `StochasticConfig` and converts via `From`.

**Option B (thorough)**: Consolidate all configuration into a single `AppConfig` with nested sections, each deriving Deserialize, so the TOML file is the authoritative source. CLI args override TOML values. No parameter exists only in code defaults.

**Concrete changes**:

- **`src/bin/market_maker.rs`**: Add `stochastic: StochasticAppConfig` to AppConfig. Implement `From<StochasticAppConfig> for StochasticConfig`. In the config loading code, replace `StochasticConfig::default()` with `config.stochastic.into()`.

- **Validation**: Add a startup log that prints all non-default StochasticConfig values at INFO level so users can verify their TOML took effect:

```rust
if stochastic_config != StochasticConfig::default() {
    info!("StochasticConfig overrides: kelly_fraction={}, ...", ...);
}
```

---

## Issue 8: Floor Binding Is Constant (GLFT Output Is Irrelevant)

### Current State

In `strategy/ladder_strat.rs`, the spread computation goes:

```
1. GLFT optimal: delta = (1/gamma) * ln(1 + gamma/kappa) => 3.7-4.8 bps
2. Regime floor: Volatile=10bps, Extreme=20bps
3. effective_floor = max(regime_floor, tick_floor, latency_floor, min_spread_floor)
4. Final = max(glft_optimal, effective_floor)
```

The effective_floor is 10-12.5 bps (regime floor + safety margins). GLFT produces 3.7-4.8 bps. The floor always wins. The warning fires every cycle (line 995-1000):

```rust
"Floor binding -- gamma may be miscalibrated (should be rare after self-consistent gamma)"
```

This makes the entire GLFT computation -- gamma calibration, kappa estimation, sigma measurement, risk model blending -- irrelevant. The system always quotes at the static floor.

### Why It Fails

The floor was designed as a safety net for rare transient conditions. But the regime classifier (Issue 2) says we're in Volatile/Extreme from startup, and the Volatile floor is 10 bps. GLFT with reasonable kappa (~2000-8000) and gamma (~0.1-0.5) produces 3-5 bps. The floor is 2-3x the optimal spread. This means:

1. The system is always quoting wider than optimal, missing fills
2. The spread never adapts to market conditions (it's always the floor)
3. All the estimation machinery (kappa, sigma, gamma calibration) is dead weight

### Principled Solution: GLFT Output IS the Spread; Floors Are Physical Constraints Only

If "GLFT is the spread engine" were foundational, floors would only exist for physical constraints that GLFT can't model: tick size, latency risk, and exchange fees. Everything else (AS risk, regime risk, cascade risk) would be encoded as inputs to GLFT via gamma and kappa.

**Architecture**:

```
Spread = GLFT(gamma_effective, kappa_effective, sigma)
       where gamma_effective includes all risk premia
       and kappa_effective includes AS discount

Floor = max(tick_size, latency_floor, maker_fee)    // Physical constraints ONLY
       typically 1.5-3 bps on Hyperliquid

Risk premia -> gamma, NOT floor:
  - Regime risk -> gamma * (1 + regime_premium)
  - AS risk -> kappa * (1 - as_discount)
  - Cascade risk -> gamma * (1 + hawkes_premium)
```

**Concrete changes**:

- **`strategy/regime_state.rs`**: Remove `spread_floor_bps` from RegimeParams entirely. Replace with `gamma_multiplier` and `kappa_discount`. Calm: gamma_mult=1.0, kappa_discount=0.0. Extreme: gamma_mult=3.0, kappa_discount=0.5.

- **`strategy/ladder_strat.rs`** (around line 914):
  - Change `effective_floor_bps` to ONLY include physical constraints: `max(tick_floor, latency_floor, fee_floor)`. This should be ~1.5-3 bps, not 10-20 bps.
  - Remove the regime_floor_bps from effective_floor computation.
  - Wire regime risk through gamma instead: `gamma *= regime_params.gamma_multiplier`.

- **`risk_config.min_spread_floor`**: Reduce default from current value to match the physical floor (~1.5 bps for maker fee on Hyperliquid). This is the absolute minimum where trading is break-even on fees alone.

- **`strategy/ladder_strat.rs`**: Change the floor-binding warning to fire only when GLFT < physical_floor (which should genuinely be rare and indicates gamma is way too low), not when GLFT < the inflated regime floor.

**Expected result**: GLFT produces 3-5 bps in normal conditions. Regime risk widens this to 6-10 bps via gamma. Extreme conditions widen to 15-25 bps via gamma. But the spread *adapts* -- it's not a static clamp. Fills improve because the spread matches market conditions rather than a worst-case floor.

---

## Issue 9: Ladder Generation Fragile At Small Size

### Current State

`quoting/ladder/generator.rs` in `build_raw_ladder()` (lines 584-615):

```rust
for (&depth_bps, &size) in depths.iter().zip(sizes.iter()) {
    // ...
    if bid_price * size >= min_notional {
        bids.push(level);
    }
}
```

When ALL levels fail min_notional, the concentration fallback (lines 620-660) creates a single level. But this fallback:
1. Uses `MAX_SINGLE_ORDER_FRACTION` (25% of total) which may still be below min_notional
2. Has a secondary `min_size_for_notional` calculation that does work, but only for the first depth level
3. Was recently added and not battle-tested for the asymmetric path

### Why It Fails

The generate function distributes `total_size` across `num_levels` levels uniformly before considering min_notional. With small total_size, every level ends up below the threshold. The fallback catches this but it's an error-recovery path, not a deliberate design.

### Principled Solution: Min-Notional-Aware Allocation From The Start

**Architecture**: The ladder generator should guarantee at least one viable level before attempting multi-level distribution.

**Concrete changes**:

- **`quoting/ladder/generator.rs`**: At the top of `generate()` and `generate_asymmetric()`, compute the budget feasibility:

```rust
let total_notional = params.total_size * params.mid_price;
let min_viable_levels = 1;
let max_viable_levels = (total_notional / (min_notional * 1.05)).floor() as usize;

if max_viable_levels == 0 {
    // Can't even meet min_notional with entire budget -- return empty ladder
    // (caller should log and handle)
    return Ladder::empty();
}

let actual_levels = config.num_levels.min(max_viable_levels).max(min_viable_levels);
```

Then use `actual_levels` instead of `config.num_levels` for the depth/size allocation. This eliminates the need for the concentration fallback entirely because every level is guaranteed to be viable.

- **`strategy/ladder_strat.rs`**: Add a pre-check that logs clearly when capital constrains the ladder:

```rust
if max_viable_levels < config.num_levels {
    info!(
        "[LADDER] Capital-constrained: {} viable levels (config: {}), total_notional=${:.2}",
        max_viable_levels, config.num_levels, total_notional,
    );
}
```

---

## Issue 10: Pricing Precision Errors

### Current State

Two types of errors observed:
1. "Price must be divisible by tick size" -- the price rounding in `round_to_significant_and_decimal()` doesn't always produce tick-aligned prices.
2. "Post only order would have immediately matched" -- a limit order was placed at or beyond the current BBO, meaning it would cross the spread.

The tick size is hardcoded as a TODO in quote_engine.rs (line 607):

```rust
tick_size_bps: 10.0, // TODO: Get from asset metadata
```

The anti-crossing logic uses `bid_base = mid.min(market_mid)` and `ask_base = mid.max(market_mid)` (generator.rs lines 581-582) which should prevent crossing, but doesn't account for:
- The BBO moving between quote generation and order submission
- Rounding pushing a price across the spread boundary
- Microprice being between bid and ask where rounding goes wrong

### Why It Fails

Tick alignment and BBO crossing are physical exchange constraints. The system treats them as rounding concerns but they're ordering constraints: `bid_price < best_ask` and `ask_price > best_bid` and `price % tick_size == 0`.

### Principled Solution: Exchange Constraints As First-Class Types

**Concrete changes**:

- **`quoting/ladder/generator.rs`**: Add a final validation pass after all rounding:

```rust
fn validate_exchange_constraints(
    ladder: &mut Ladder,
    tick_size: f64,
    best_bid: f64,
    best_ask: f64,
) {
    // 1. Snap all prices to tick grid
    for level in ladder.bids.iter_mut() {
        level.price = (level.price / tick_size).floor() * tick_size;
    }
    for level in ladder.asks.iter_mut() {
        level.price = (level.price / tick_size).ceil() * tick_size;
    }

    // 2. Ensure no bid >= best_ask (would immediately match)
    ladder.bids.retain(|l| l.price < best_ask);

    // 3. Ensure no ask <= best_bid
    ladder.asks.retain(|l| l.price > best_bid);
}
```

- **Asset metadata**: The tick size must come from exchange metadata (asset info response), not a hardcoded constant. Fetch it at startup and store in config.

- **`orchestrator/quote_engine.rs`**: Replace the hardcoded `tick_size_bps: 10.0` with the actual tick size from the exchange asset info, which is already available from the `AssetRuntimeConfig`.

---

## Summary: What Changes First

If I had to implement these in order of impact on making the next live session profitable:

1. **Issue 8 (Floor binding)** -- Remove regime floors from the spread floor. Let GLFT produce the spread. Route regime risk through gamma. This alone should bring spreads from 10-12.5 bps down to 4-6 bps, dramatically improving fill rate.

2. **Issue 6 (Position sizing)** -- Add capital-aware level count. At $100, use 2-3 levels instead of 10. Each level is then $15-30, comfortably above min_notional.

3. **Issue 2 (Regime HMM)** -- Increase calibration_buffer_size to 1000 and add authority damping. This prevents the "Extreme at 200 obs" problem that cascades into Issues 6 and 9.

4. **Issue 10 (Pricing)** -- Add tick alignment and BBO crossing validation. This directly prevents order rejections.

5. **Issue 9 (Ladder fragility)** -- Pre-compute viable level count. This is partially addressed by #2 but needs the generator-level fix for robustness.

6. **Issue 7 (Dead TOML)** -- Wire StochasticConfig to AppConfig. Quick win, high trust-building value.

7. **Issues 1, 3, 4, 5** -- AS warmup, fill logging, Sharpe gating, PnL visibility. These are all important for operational quality but don't directly block profitability.

The first three changes together address the root cause: the system was designed for large-capital liquid markets and breaks structurally at small capital on illiquid tokens. Making it work at $100 on HYPE is not about adding features -- it's about removing assumptions that only hold at scale.
