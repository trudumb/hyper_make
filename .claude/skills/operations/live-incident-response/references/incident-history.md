# Incident History

Known past incidents from the Hyperliquid market maker, documented from session memories
and MEMORY.md. Use this as a reference when diagnosing new incidents -- similar patterns
tend to recur.

## Incident Table

| Date | Symptom | Root Cause | Fix | Key File |
|------|---------|-----------|-----|----------|
| Feb 9 | Stuck in NoQuote after 1 fill | Drawdown denominator bug: `summary()` divides by `peak_pnl` (tiny after 1 fill) showing 1000%+ drawdown. Actual kill switch check in `state.rs:drawdown()` uses `account_value` but was not yet wired. | Fixed `RiskState::drawdown()` to use `account_value` denominator (state.rs:253-261). Added `min_peak_for_drawdown` guard. | `risk/state.rs`, `risk/kill_switch.rs` |
| Feb 9 | Permanent 2x staleness on HIP-3 | `use_lead_lag` and `use_cross_venue` config flags were true but HIP-3 has no Binance listing. Signals never warm up, `staleness_spread_multiplier()` permanently returns 2.0x. | Added `disable_binance_signals()` method to `SignalIntegrator`. Called when no Binance symbol found for asset. | `strategy/signal_integration.rs:1040` |
| Feb 9 | 9 of 10 learning loops missing in live | Paper trader had all learning loops wired but live `handlers.rs` was missing: own-fill kappa, pending fill outcomes, AS classifier feedback, and full analytics bundle. | Wired all 9 missing loops in `handlers.rs`: `estimator.on_own_fill()`, `PendingFillOutcome` queue, `pre_fill_classifier.record_outcome()`, `LiveAnalytics` bundle. | `orchestrator/handlers.rs` |
| Feb 9 | RL disabled by default | `rl_enabled` field defaulted to `false` in config `mod.rs`. RL ensemble member never participated. | Changed default to `true`. Removed `--enable-rl` flag. | `config/mod.rs` |
| Feb 10 | Duplicate on_trade double-counting kappa | Two `on_trade` calls in `handlers.rs` (line ~437-440) plus canonical one in `messages/trades.rs:85`. Double kappa counting inflated raw kappa estimates. | Removed duplicate `on_trade` call from `handlers.rs`. | `orchestrator/handlers.rs` |
| Feb 10 | Kappa cycle skip -- raw kappa > 5000 | Feature compared raw kappa (~6400-7700 from L2) against 5000 threshold. Should have used `kappa_effective` (~3250, blended). Skipped entire quote cycles. | Removed the feature entirely (redundant with proactive rate limiter). | `orchestrator/quote_engine.rs` |
| Feb 10 | WaitToLearn cold-start deadlock | L3 readiness level required learning to complete before quoting. But learning requires fills, fills require quoting. System stuck in "wait to learn" forever. | Changed to fall-through: quote anyway even when not fully learned. You can only learn by quoting. | `orchestrator/quote_engine.rs` |
| Feb 10 | Cold-start staleness 2.0x | `staleness_spread_multiplier()` was penalizing signals with `observation_count == 0`. These are cold-start (never had data), not stale (had data then lost it). | Added `observation_counts() != (0, 0)` guard and `was_ever_warmed_up()` checks. Only penalize staleness if signal was previously warmed. | `strategy/signal_integration.rs:999-1033` |

## Pattern Taxonomy

These incident patterns repeat. When diagnosing a new incident, check which pattern it matches.

### Pattern 1: Cold-Start Deadlock
**Signature**: System stuck at startup, no quoting ever begins.
**Root cause**: A precondition for quoting can only be satisfied BY quoting.
**Examples**: WaitToLearn deadlock, cold-start staleness.
**Defense**: Any gate that blocks quoting must have a timeout or fallthrough.

### Pattern 2: Missing Feed Permanent Penalty
**Signature**: A spread multiplier is permanently elevated (1.5-2.0x), never recovers.
**Root cause**: A signal is enabled in config but its data source doesn't exist for this asset.
**Examples**: Binance staleness on HIP-3, cross-venue flow on DEX-only tokens.
**Defense**: Auto-detect missing feeds at startup; disable dependent signals.

### Pattern 3: Raw vs Processed Value Comparison
**Signature**: A threshold check fires when it shouldn't, or doesn't fire when it should.
**Root cause**: Code compares a raw value against a threshold designed for a processed value.
**Examples**: Raw kappa (~6400) vs threshold meant for kappa_effective (~3250); peak_pnl
denominator vs account_value denominator.
**Defense**: Name variables to indicate whether they are raw or processed. Add comments
next to threshold comparisons noting which variant is expected.

### Pattern 4: Paper-Live Divergence
**Signature**: Feature works in paper trading but fails or is absent in live.
**Root cause**: Paper trader and live MM have parallel but not identical wiring.
**Examples**: 9/10 missing learning loops, RL disabled by default.
**Defense**: Audit paper vs live wiring periodically. Shared trait implementations help.

### Pattern 5: Multiplicative Compounding
**Signature**: Spreads are 3-5x wider than any single factor would justify.
**Root cause**: Multiple independent spread multipliers are multiplied together.
**Examples**: staleness 1.5x * model_gating 1.5x * toxicity 1.5x = 3.375x.
**Defense**: Log all multiplier components. Set global cap (currently 10.0x). Consider
additive composition for factors that should be independent.

## Operational Context from Incidents

### Feb 9: First Live Deployment (HIP-3)

The first live deployment on HIP-3 (hyna DEX token) hit three blockers simultaneously:
drawdown false trigger, permanent staleness, and missing learning loops. The compound
effect was that the MM placed 1 fill, triggered the kill switch from a meaningless
drawdown percentage, and could never recover because the manual reset would just re-trigger
due to the permanent staleness widening.

**Resolution time**: ~4 hours to diagnose all three root causes.
**Paper PnL at the time**: +2.79 bps edge, Sharpe 927. Live was -1.5 bps due to 7% API quota.

### Feb 10: Live Quoting Fix

After the Feb 9 fixes, the MM was deployed again but still not quoting. Three additional
bugs were found: duplicate kappa counting, kappa threshold skip, and WaitToLearn deadlock.
These were all independent but combined to prevent any quotes from being placed.

**Resolution time**: ~2 hours. After fixes, 27 order placements in 120s, 3W/1L Kelly.
**Position movement**: -0.77 to +5.14 HYPE in the first session.

### Feb 10: Spread Cascade Discovery

During the drift audit, the team discovered that multiplicative spread compounding was
the #1 hidden EV drain. Individual factors of 1.3-1.5x compounded to 3.4x total spread.
The additive cap of 20 bps was added to signal integration, and the global multiplicative
cap of 10.0x was added to the toxicity config.

**Impact**: P0 fixes alone estimated to recover ~2.3 bps of the 4.29 bps paper-to-live gap.

## Lessons Learned

1. **EWMA update-before-compute**: This was a recurring bug pattern in early development.
   Always update the EWMA state BEFORE reading the smoothed value, not after.

2. **You can only learn by quoting**: Never block quoting to "wait and learn" unless you
   have an alternative data source. Cold-start deadlocks are silent killers.

3. **Raw kappa vs effective kappa**: `estimator.kappa()` returns raw L2 kappa (~6400-7700).
   The blended `kappa_effective` (~3250) is what should be used for threshold comparisons.

4. **Drawdown from tiny peaks is meaningless**: A $0.02 peak from one fill makes any loss
   look like 1000%+ drawdown. Use `min_peak_for_drawdown` guard and `account_value` denominator.

5. **Defense first**: When uncertain, widen spreads. Missing a trade costs basis points.
   Getting run over in a cascade costs the account.

## Key File Map

Quick reference for all risk/quoting infrastructure files relevant to incident triage.

| Component | Path | Key Lines |
|-----------|------|-----------|
| Kill switch | `risk/kill_switch.rs` | is_triggered:443, reset:461, check:471, summary:866 |
| Kill reasons | `risk/kill_switch.rs` | KillReason enum:282-307 |
| Risk state | `risk/state.rs` | drawdown():253-262, RiskState struct:12-101 |
| Risk aggregator | `risk/aggregator.rs` | max_severity:170-171 |
| Circuit breaker | `risk/circuit_breaker.rs` | actions:47-64, most_severe:358 |
| Loss monitor | `risk/monitors/loss.rs` | evaluate:34 |
| Drawdown monitor | `risk/monitors/drawdown.rs` | evaluate:44, min_peak:53 |
| Position monitor | `risk/monitors/position.rs` | evaluate:42, hard limit 2x:45 |
| Position velocity | `risk/monitors/position_velocity.rs` | evaluate:63, thresholds:57-59 |
| Data staleness | `risk/monitors/data_staleness.rs` | evaluate:49, grace:67-80 |
| Cascade monitor | `risk/monitors/cascade.rs` | evaluate:47, pull:0.8, kill:5.0 |
| Price velocity | `risk/monitors/price_velocity.rs` | evaluate:39, pull:5%/s, kill:15%/s |
| Rate limit monitor | `risk/monitors/rate_limit.rs` | evaluate:39, kill at 3 errors |
| Data quality gate | `infra/data_quality.rs` | should_gate_quotes:287, reasons:72-79 |
| Connection health | `infra/reconnection.rs` | states:48-59, stale check:144 |
| Connection super. | `infra/connection_supervisor.rs` | events:50-61, stale:10s |
| Spread chain | `orchestrator/quote_engine.rs` | multiplier composition:992-1107 |
| Quote gate | `control/quote_gate.rs` | QuoteDecision:41, NoQuoteReason:78 |
| Signal staleness | `strategy/signal_integration.rs` | staleness_mult:999, disable_binance:1040 |
| Model gating mult | `strategy/signal_integration.rs` | model_gating_spread_mult:978 |
| Edge defensive | `analytics/edge_metrics.rs` | max_defensive_multiplier:192 |
| Toxicity | `analytics/market_toxicity.rs` | max_composed:64,98 (default 10.0x) |
| Rate limit infra | `infra/rate_limit/mod.rs` | module overview |
| Shadow spread | `control/quote_gate.rs` | continuous_shadow_spread_bps:885 |
| Quota ladder | `control/quote_gate.rs` | continuous_ladder_levels:901 |
| Risk overlay | `control/mod.rs` | RiskAssessment struct:775-784 |
