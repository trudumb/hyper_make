---
description: Domain-specific pitfalls and common mistakes
globs:
  - "src/market_maker/**"
---

# Domain Gotchas

Hard-won lessons from production. Each of these has caused real money loss or significant debugging time.

## Formula Safety

- `kappa > 0.0` always — GLFT formula has `ln(1 + gamma/kappa)`, blows up at zero
- `gamma > 0.0` always — risk aversion must be positive
- `ask_price > bid_price` — spread invariant must hold in all code paths

## Hyperliquid-Specific

- **Maker fee is 1.5 bps** — always include in spread calculation
- **Binance leads Hyperliquid by 50-500ms** — this lead-lag is exploitable but decaying
- **Funding rate settlement** creates predictable flow patterns — don't ignore near settlement
- **OI drops signal liquidation cascades** — widen immediately

## Multiplicative Compounding

The most common spread bug: multiple independent factors each add "a little" defense, but `1.5 x 1.5 x 1.5 = 3.375x`. The post-2026-02-20 architecture uses additive `total_risk_premium_bps` to prevent this.

## Binary Side-Clearing

Binary side-clearing (pulling all quotes on one side) is ALWAYS wrong. Three code paths used to do it:
1. ExecutionMode state machine (removed)
2. Emergency filter (fixed to route through graduated widening)
3. TOXICITY CANCEL (deleted)

Route through sigma/AS multipliers instead.

## Checkpoint Persistence

- Kill switch state persists across restarts
- Must clear checkpoint `kill_switch.triggered=false` manually to restart after kill switch
- All new checkpoint fields must use `#[serde(default)]` for backward compat
