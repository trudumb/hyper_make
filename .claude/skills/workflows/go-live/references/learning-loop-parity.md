# Learning Loop Parity: Paper vs Live

Reference table for verifying all learning feedback loops are correctly wired in live mode.

## Architecture Note

Since the unified architecture redesign, both paper and live trading use the **same** `MarketMaker<S, Env>` generic struct. Paper uses `PaperEnvironment` and live uses `LiveEnvironment`, but **all handlers are shared** via `handlers.rs`. This means:

- `handle_user_fills()` (handlers.rs:571) is called by BOTH paper and live
- The `FillProcessor` (fills/processor.rs:720) is called by BOTH paper and live
- Additional learning loops in handlers.rs:700-805 run for BOTH paper and live
- The quote engine (quote_engine.rs) runs identically in both modes

The primary parity risk is now in **binary setup** (`src/bin/market_maker.rs`), where paper mode (`run_paper_mode()`, line 2467) and live mode share config but may diverge on:
- Feature flags (e.g., `disable_binance_signals`)
- Config defaults (e.g., `StochasticConfig::default()` vs loaded config)
- Prior injection (live loads paper checkpoint, paper starts cold)

## Learning Loop Table

| # | Loop | What It Does | File:Line (shared handler) | Secondary Location | Verify Command |
|---|------|-------------|---------------------------|-------------------|----------------|
| 1 | **Kappa from own fills** | Feeds fill distance to kappa estimator for fill intensity learning | `handlers.rs:738` (`estimator.on_own_fill()`) | `fills/processor.rs:906` (same call in FillProcessor) | `grep -n "on_own_fill" src/market_maker/orchestrator/handlers.rs` |
| 2 | **AS markout queue** | Records fill for 5-second markout to detect adverse selection | `handlers.rs:749-756` (`pending_fill_outcomes.push_back(PendingFillOutcome{...})`) | Drained in `handlers.rs:241` (`check_pending_fill_outcomes`) | `grep -n "PendingFillOutcome" src/market_maker/orchestrator/handlers.rs` |
| 3 | **Pre-fill classifier outcome** | Feeds markout result (adverse/not) to Bayesian AS classifier | `handlers.rs:259` (`pre_fill_classifier.record_outcome()`) | Also: `fills/processor.rs:998` (in FillProcessor markout path) | `grep -n "pre_fill_classifier.record_outcome" src/market_maker/orchestrator/handlers.rs` |
| 4 | **AS prediction update** | Updates model gating's AS calibration with predicted vs actual | `handlers.rs:301` and `handlers.rs:773-775` (`signal_integrator.update_as_prediction()`) | Two call sites: one in markout drain, one in fill handler | `grep -n "update_as_prediction" src/market_maker/orchestrator/handlers.rs` |
| 5 | **LiveAnalytics record_fill** | Updates Sharpe tracker and signal PnL attribution per fill | `handlers.rs:799` (`live_analytics.record_fill()`) | Defined in `analytics/live.rs:94` | `grep -n "live_analytics.record_fill" src/market_maker/orchestrator/handlers.rs` |
| 6 | **LiveAnalytics record_quote_cycle** | Records signal contributions each quote cycle for attribution | `quote_engine.rs:868` (`live_analytics.record_quote_cycle(&signals)`) | Defined in `analytics/live.rs:125` | `grep -n "record_quote_cycle" src/market_maker/orchestrator/quote_engine.rs` |
| 7 | **LiveAnalytics maybe_log_summary** | Periodic summary log (Sharpe, attribution) every 30 seconds | `handlers.rs:1791` (`live_analytics.maybe_log_summary()`) | Defined in `analytics/live.rs:155` | `grep -n "maybe_log_summary" src/market_maker/orchestrator/handlers.rs` |
| 8 | **Quote outcome tracker** | Registers pending quotes, resolves on fill/expiry for unbiased edge | Register: `quote_engine.rs:2305,2314` / Fill: `handlers.rs:804` / Update: `quote_engine.rs:235-236` | Defined in `learning/quote_outcome.rs:191` (`QuoteOutcomeTracker`) | `grep -n "quote_outcome_tracker" src/market_maker/orchestrator/quote_engine.rs` |
| 9 | **RL baseline tracker** | EWMA baseline for counterfactual reward computation | `learning/rl_agent.rs:1402` (`baseline: BaselineTracker`) | Defined in `learning/baseline_tracker.rs:15` | `grep -n "BaselineTracker" src/market_maker/learning/rl_agent.rs` |

## How to Verify All Loops Are Wired

Run this combined grep to confirm all 9 loops appear in the shared handler path:

```bash
echo "=== Loop 1: Kappa from own fills ==="
grep -n "estimator.on_own_fill" src/market_maker/orchestrator/handlers.rs

echo "=== Loop 2: AS markout queue ==="
grep -n "pending_fill_outcomes.push_back" src/market_maker/orchestrator/handlers.rs

echo "=== Loop 3: Pre-fill classifier outcome ==="
grep -n "pre_fill_classifier.record_outcome" src/market_maker/orchestrator/handlers.rs

echo "=== Loop 4: AS prediction update ==="
grep -n "update_as_prediction" src/market_maker/orchestrator/handlers.rs

echo "=== Loop 5: LiveAnalytics record_fill ==="
grep -n "live_analytics.record_fill" src/market_maker/orchestrator/handlers.rs

echo "=== Loop 6: LiveAnalytics record_quote_cycle ==="
grep -n "record_quote_cycle" src/market_maker/orchestrator/quote_engine.rs

echo "=== Loop 7: LiveAnalytics maybe_log_summary ==="
grep -n "maybe_log_summary" src/market_maker/orchestrator/handlers.rs

echo "=== Loop 8: Quote outcome tracker ==="
grep -n "quote_outcome_tracker" src/market_maker/orchestrator/quote_engine.rs
grep -n "quote_outcome_tracker" src/market_maker/orchestrator/handlers.rs

echo "=== Loop 9: RL baseline tracker ==="
grep -n "BaselineTracker" src/market_maker/learning/rl_agent.rs
```

Each grep must return at least one match. If any loop is missing, the live trader will silently degrade -- it will still trade but with stale parameters.

## Known Past Regressions

1. **Duplicate on_trade** (fixed): `handlers.rs` had a duplicate `on_trade` call that double-counted market trades for kappa estimation. Already handled in `messages/trades.rs:85`.
2. **9/10 learning features missing from live** (fixed 2026-02-09): Paper had all learning loops wired but live `handlers.rs` was missing most of them. The unified architecture now shares all handlers.
3. **Cold-start staleness** (fixed): `signal_integration.rs:1006` -- without the `observation_count > 0` guard, cold-start signals triggered permanent 2x staleness widening.
