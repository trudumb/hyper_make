# Fix: Live Market Maker Killed in 108 Seconds While Paper Trader Works Fine

## Context

The paper trader ran BTC for 45 min, got 20 fills, +$0.16 PnL, Sharpe 840. Then you ran `./scripts/test_hip3.sh HYPE hyna` and the live MM was killed after **108 seconds** by the drawdown kill switch (177% > 10%). Here's exactly what happened:

### The Failure Sequence (from `mm_hip3_hyna_HYPE_hip3_2026-02-08_20-16-30.log`)

1. **03:16:37** - Started `hyna:HYPE` at $32.95. Position: 0. Max position: 1.517589 contracts ($50 USD limit)
2. **03:16:37 to 03:18:19** - **103 seconds of no quoting** (warmup + waiting for data)
3. **03:18:19** - First quote cycle. Ladder collapsed to **single bid 1.51@32.992** and **single ask 1.51@33.006**. Each order = 100% of max position.
4. **03:18:20** - Bid **rejected**: "Post only would have immediately matched, bbo was 32.977@32.992" (stale mid price caused bid to cross BBO)
5. **03:18:20** - Ask placed at 33.006, 1.51 contracts. **Only sell-side live.**
6. **03:18:23** - Ask **FILLED**: sold 1.51 HYPE. Position: **-1.51** (100% of max, 316% of effective_max_position)
7. **03:18:24** - Spread capture: +$0.02. Then price moved against. Unrealized loss: -$0.0168
8. **03:18:24** - Drawdown: `(0.02 - (-0.0168)) / 0.02 = 184%` > 10% threshold. **KILL SWITCH.**

### Five Root Causes

| # | Bug | Severity | Impact |
|---|-----|----------|--------|
| 1 | **Single order = 100% of max position** | P0 | One fill maxes out position, zero ability to recover |
| 2 | **Drawdown fires on $0.02 peak PnL** | P0 | Any micro-move triggers 100%+ drawdown from tiny peak |
| 3 | **Binance feed hardcoded to `btcusdt`** when trading HYPE | P1 | Cross-venue signals are noise, not signal |
| 4 | **103s first-quote delay** | P1 | Lost 2 minutes of trading time on a 3200s session |
| 5 | **Stale mid price** causes bid to cross BBO | P1 | Half the order book rejected, creating one-sided exposure |

### Why Paper Trader Didn't Hit This

- BTC at $70k with $200 limit = 0.00284 BTC per order (tiny). HYPE at $33 with $50 limit = 1.51 contracts (max position).
- Paper-mode kappa floor (8000) = tighter spreads. Live used kappa=1393 (wider, more aggressive sizes).
- Paper fills are simulated (no "post only would match" rejection). Live got rejected on bid side.
- Paper accumulated 20 fills gradually. Live got ONE fill for 100% of max position.

---

## Plan

### Fix 1: Per-Order Size Cap (P0)
**File:** `src/market_maker/strategy/ladder_strat.rs`

The "Bid/Ask concentration fallback" (lines ~62-63 in live log) collapses all 25 levels into a single order at tightest depth. The size is uncapped.

**Change:** In the concentration fallback code path, cap per-order size at `max(min_meaningful_size, 0.25 * effective_max_position_for_risk)`. No single resting order should exceed 25% of the user's risk-based max position.

Also: in the general ladder generation, apply a per-level cap of `effective_max_position * 0.33 / num_active_levels.max(1)` so even multi-level ladders can't accumulate to max in one sweep.

### Fix 2: Drawdown Kill Switch Minimum Peak Threshold (P0)
**File:** `src/market_maker/risk/kill_switch.rs`

`check_drawdown()` (line 554) only guards `peak_pnl <= 0.0`. But with peak=$0.02, a $0.04 swing = 300% drawdown.

**Change:** Add a minimum absolute peak threshold. If `peak_pnl < min_peak_for_drawdown` (default: $1.00 or 2% of max_position_value), skip drawdown check. The absolute daily loss check (`check_daily_loss`) still protects against real losses.

```rust
fn check_drawdown(&self, state: &KillSwitchState, config: &KillSwitchConfig) -> Option<KillReason> {
    // Need meaningful peak before drawdown is useful
    let min_peak = config.max_position_value * 0.02; // 2% of max position value
    if state.peak_pnl <= min_peak.max(1.0) {
        return None;
    }
    // ... existing logic
}
```

### Fix 3: Dynamic Binance Feed Symbol (P1)
**File:** `src/market_maker/mod.rs` (around Binance feed initialization, line ~28 in log: `"symbol":"btcusdt"`)

The Binance WebSocket always subscribes to `btcusdt`, even when trading HYPE. For HYPE, the cross-venue signal is meaningless and adds noise.

**Change:**
- If the asset is BTC or ETH (or has a Binance equivalent), use the correct symbol
- If the asset has no Binance equivalent (like HYPE), **disable** the Binance feed entirely
- Add a `binance_symbol` config option for explicit override

### Fix 4: Reduce First-Quote Delay (P1)
**File:** `src/market_maker/orchestrator/event_loop.rs`

103 seconds elapsed before the first quote cycle. The warmup phase waits for trade data, but on low-liquidity HIP-3 DEX assets, trades may be infrequent.

**Change:** Add a maximum warmup timeout of 30 seconds. After 30s without sufficient data, start quoting with prior parameters (kappa=1500, sigma=prior). The warmup can continue refining while quotes are live. This is already partially handled by `warmup_pct` logic but the initial quote trigger needs a time-based fallback.

### Fix 5: Stale Mid Price Guard Before Order Placement (P1)
**File:** `src/market_maker/orchestrator/reconcile.rs`

The bid at 32.992 crossed the actual BBO ask at 32.992. The mid price used for quote generation was stale relative to the real BBO.

**Change:** Before placing orders, validate that:
- `bid_price < best_ask` and `ask_price > best_bid` from the most recent L2 book
- If the L2 book is older than 5 seconds, widen spreads by an additional buffer
- Log a warning and skip the order (don't place) if it would cross

---

## Files to Modify

1. `src/market_maker/strategy/ladder_strat.rs` — per-order size cap in concentration fallback
2. `src/market_maker/risk/kill_switch.rs` — min peak threshold for drawdown check
3. `src/market_maker/mod.rs` — dynamic Binance symbol selection
4. `src/market_maker/orchestrator/event_loop.rs` — warmup timeout fallback
5. `src/market_maker/orchestrator/reconcile.rs` — pre-placement BBO validation

## Verification

1. `cargo clippy -- -D warnings` — clean build
2. `cargo test` — all tests pass
3. Existing kill_switch tests in `src/market_maker/risk/kill_switch.rs` updated for min-peak logic
4. Manual verification: the user runs `./scripts/test_hip3.sh HYPE hyna 300 hip3` and confirms:
   - Orders are sized at max 25% of position limit per order
   - Drawdown doesn't fire until meaningful PnL peak exists
   - No `btcusdt` Binance subscription in logs when trading HYPE
   - First quote within 30 seconds of startup
   - No "post only would have immediately matched" rejections
