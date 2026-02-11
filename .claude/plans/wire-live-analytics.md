# Wire LiveAnalytics Call Sites

## Objective
Wire 4 LiveAnalytics call sites in the live event loop to enable runtime Sharpe tracking, signal attribution, and periodic logging.

## Files to Modify

### 1. `src/market_maker/orchestrator/handlers.rs` — 3 call sites

**Call site 1: After each fill (in handle_user_fills per-fill loop)**
- Location: In the `for fill in &user_fills.data.fills` block, after the existing edge tracking section
- Code: `self.live_analytics.record_fill(fill_pnl_bps, Some(&snap));`
- `fill_pnl_bps` is already computed as `realized_edge_bps` in the RL section (depth_bps - as_bps - fee_bps)
- The `EdgeSnapshot` is already constructed — reuse it
- Move the EdgeSnapshot construction slightly earlier so it's in scope, or compute fill_pnl_bps from existing variables

**Call site 3: In periodic_component_update()**
- Location: At the end of `periodic_component_update()`
- Code: `self.live_analytics.maybe_log_summary(mean_edge);`
- `mean_edge` from `self.tier2.edge_tracker.mean_realized_edge_bps()`

### 2. `src/market_maker/orchestrator/quote_engine.rs` — 1 call site

**Call site 2: After get_signals() in update_quotes()**
- Location: Right after `let signals = self.stochastic.signal_integrator.get_signals();`
- Code: `self.live_analytics.record_quote_cycle(&signals);`

### 3. `src/market_maker/orchestrator/recovery.rs` — 1 call site

**Call site 4: On shutdown**
- Location: Near the beginning of `shutdown()`, before order cancellation
- Code: `self.live_analytics.flush();`

## Verification
- `cargo clippy --lib -- -D warnings`
- `cargo test --lib orchestrator`
- `cargo test --lib analytics`
