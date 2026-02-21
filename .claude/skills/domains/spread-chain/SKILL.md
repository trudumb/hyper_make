---
name: spread-chain
description: Documents the additive spread composition pipeline from GLFT optimal through to final bid/ask prices. Use when debugging wide spreads, investigating spread component contributions, tuning defensive behavior, or understanding why quotes are wider than expected. Critical for incident triage.
requires:
  - measurement-infrastructure
user-invocable: false
---

# Spread Chain Skill

## Architecture (Post 2026-02-20 Unified Stochastic Refactor)

The spread pipeline is **purely additive**. The old multiplicative chain (`base x factor1 x factor2 x factor3`) was deleted because it caused death spirals: `1.5 x 1.5 x 1.5 = 3.375x` blowups when independent factors compounded.

## The Pipeline

### Stage 1: GLFT Optimal Half-Spread

Formula: `delta* = (1/gamma) * ln(1 + gamma/kappa) + 0.5 * gamma * sigma^2 * T + maker_fee`

- `gamma` = risk aversion (regime-scaled via `regime_gamma_multiplier`: Calm=1.0, Normal=1.2, Volatile=2.0, Extreme=3.0)
- `kappa` = order book depth decay (from Hawkes-based estimator)
- `sigma` = realized volatility (per-second)
- `T` = holding time horizon (seconds)
- `maker_fee` = 1.5 bps (constant on Hyperliquid)

Computed in `glft.rs:405-430` via `solve_min_gamma()`.

### Stage 2: Additive Risk Premium (`total_risk_premium_bps`)

All risk adjustments accumulate into a single additive field on `MarketParams`. Contributors:

| Source | File:Line | Description |
|--------|-----------|-------------|
| Regime risk | `market_params.rs:818` | HMM-quantized: Calm=0, Normal=2, Volatile=5, Extreme=10 bps |
| Hawkes addon | `quote_engine.rs:~1786` | Fill cascade intensity premium |
| Toxicity addon | `quote_engine.rs:~1787` | Informed flow + microstructure hazards |
| Cross-venue | `quote_engine.rs:1628` | Binance-HL disagreement: `(cv_mult - 1.0) * 5.0` bps |
| Belief skewness | `quote_engine.rs:1187` | Tail risk from fat-tail detection |
| Staleness penalty | `quote_engine.rs:~2760` | Data freshness: `(stale_mult - 1.0) * 5.0` bps |
| Signal sources | `quote_engine.rs:1762` | OI vol, funding settlement, cancel-race AS |

### Stage 3: SpreadComposition (Final Assembly)

Defined in `market_params.rs:20-95`:

```
total_half_spread = max(
    GLFT_half_spread + risk_premium + quota_addon + warmup_addon,
    fee_bps                    // Safety floor: never tighter than fee
)

bid_half_spread = total_half_spread + cascade_bid_addon
ask_half_spread = total_half_spread + cascade_ask_addon
```

Components:
- `glft_half_spread_bps` — from Stage 1
- `risk_premium_bps` — from Stage 2 (`total_risk_premium_bps`)
- `quota_addon_bps` — API rate limit headroom pressure
- `warmup_addon_bps` — confidence buffer during cold-start
- `cascade_bid/ask_addon_bps` — asymmetric cascade widening

### Stage 4: Asymmetric Pricing

In `ladder_strat.rs:1046-1047`:
- `kappa_bid != kappa_ask` when informed flow is detected
- Each side gets its own GLFT formula with independent drift adjustment
- AS deduplication: `as_net = max(0, as_raw - vol_floor)` — vol component already priced in GLFT

### Stage 5: Physical Floor

In `ladder_strat.rs:974-989`:
```
effective_floor = max(fee_bps, tick_size_bps, latency_floor_bps, min_spread_floor, option_floor_bps)
```

This is the absolute minimum — purely physical constraints. No risk premium in the floor.

## Key Design Decisions

1. **No multiplicative chains** — all additive. `1.5 + 1.5 + 1.5 = 4.5 bps` not `1.5 x 1.5 x 1.5 = 3.375x`
2. **Gamma routes continuous risk** — regime tightening via `regime_gamma_multiplier` (1.0-3.0x), not floor clamping
3. **solve_min_gamma is diagnostic** — the actual floor is physical constraints, not binary search
4. **AS deduplication** — subtract vol floor from AS before adding residual (avoids double-counting)
5. **Always quote** — no binary circuit breakers returning (None, None). Graduated widening instead.

## Debugging Wide Spreads

1. **Which component is elevated?** Check `EstimatorDiagnostics` log (printed every 10 cycles in `quote_engine.rs`)
2. **If regime risk is high**: Is the regime classifier correct? Check HMM beliefs in diagnostics
3. **If staleness premium**: Is a signal source actually stale, or cold-start? Check `was_ever_warmed_up()`
4. **If cross-venue**: Is Binance feed connected? Check latency EWMA
5. **If signal sources**: Check OI vol, funding proximity, cancel-race AS individually
6. **If warmup addon**: How far through warmup? Check fill count vs warmup threshold
7. **If cascade addon**: Is there active cascade detection? Check OI drop rate

## Incident History

- **Feb 12 (V1)**: Multiplicative quota (3.34x) + bandit (2x) = 6.7x blowup. Led to additive redesign.
- **Feb 18 (Post-deployment)**: Emergency clearing at 32% position + TOXICITY CANCEL re-firing + scorer valued NewPlace at -3 bps = local_bids=0 forever. Led to deletion of binary side-clearing.
- **Feb 20 (Unified refactor)**: Deleted `staleness_addon_bid/ask_bps`, `flow_toxicity_addon_bid/ask_bps`, `sigma_cascade_mult`. Replaced with additive `total_risk_premium_bps` pipeline.

## Key File Map

| Component | File | Key Lines |
|-----------|------|-----------|
| SpreadComposition struct | `strategy/market_params.rs` | 20-95 |
| MarketParams fields | `strategy/market_params.rs` | 106-167, 818, 827 |
| GLFT formula | `strategy/glft.rs` | 405-430 |
| Risk premium accumulation | `orchestrator/quote_engine.rs` | 1760-1795 |
| Cross-venue premium | `orchestrator/quote_engine.rs` | 1628 |
| Staleness penalty | `orchestrator/quote_engine.rs` | ~2760 |
| Physical floor | `strategy/ladder_strat.rs` | 974-989 |
| Asymmetric bid/ask | `strategy/ladder_strat.rs` | 1046-1047 |
| AS dedup | `strategy/ladder_strat.rs` | 1052-1075 |
| Diagnostics log | `orchestrator/quote_engine.rs` | EstimatorDiagnostics |
