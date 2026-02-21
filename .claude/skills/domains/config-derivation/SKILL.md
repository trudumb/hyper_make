---
name: config-derivation
description: Documents auto_derive.rs first-principles parameter derivation from capital and exchange metadata. Use when onboarding new assets, debugging parameter mismatches, understanding why gamma/max_position/target_liquidity have their values, or adding new derived parameters.
requires:
  - measurement-infrastructure
user-invocable: false
---

# Config Derivation Skill

## Purpose

All trading parameters can be auto-derived from a single input (`capital_usd`) plus exchange context (mark price, margin, leverage, fees). This eliminates arbitrary config numbers and ensures parameters are always self-consistent.

## Key Principle

**One sizing input**: `capital_usd` is the ONLY human-specified parameter. Everything else derives from it plus exchange metadata.

## Core Function

`auto_derive(capital_usd, spread_profile, ctx) -> DerivedParams`

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `capital_usd` | `f64` | Capital to deploy (the ONE sizing input) |
| `spread_profile` | `SpreadProfile` | Market type: Default, Hip3, Aggressive |
| `ctx` | `ExchangeContext` | mark_px, account_value, available_margin, max_leverage, fee_bps, sz_decimals, min_notional |

### Outputs (DerivedParams)

| Field | Description | Derivation |
|-------|-------------|------------|
| `max_position` | Maximum position in contracts | `capital_usd * safety_factor * max_leverage / mark_px` |
| `target_liquidity` | Liquidity per side in contracts | Fraction of max_position |
| `risk_aversion` | Gamma for GLFT formula | Derived from spread profile + capital tier |
| `max_bps_diff` | Requote threshold (bps) | Profile-dependent |
| `viable` | Whether trading is possible | `max_position >= min_order_size` |
| `capital_profile` | Tier + ladder depth info | From position limits + exchange minimums |

### Capital Tiers

| Tier | Viable Levels/Side | Behavior |
|------|-------------------|----------|
| Micro | 1-2 | Minimum ladder, widest spreads |
| Small | 3-5 | Reduced ladder, wider spreads compensate |
| Medium | 6-15 | Standard operation |
| Large | 16+ | Full ladder capacity |

### Key Constraints

- **Safety factor**: 50% margin buffer for adverse moves
- **Min viable position**: `min_notional * 1.15 / mark_px` — the minimum that produces a valid exchange order
- **Staleness**: Profile recomputes if mark price moves >10%
- **All tiers flow through same GLFT pipeline** — `capital_limited_levels` in `generate_ladder()` naturally constrains

## Spread Profiles

Defined in `config/spread_profile.rs`:
- **Default**: Standard market making parameters
- **Hip3**: Tighter spreads for HIP-3 assets (lower liquidity)
- **Aggressive**: Tighter quotes, higher fill rate, more inventory risk

## When to Use

- **Onboarding new asset**: `auto_derive()` produces all parameters from capital + exchange context
- **Debugging parameters**: If gamma/position/liquidity seem wrong, check `ExchangeContext` inputs
- **Adding derived params**: Follow the pattern — derive from capital + context, never hardcode
- **Capital changes**: Profile auto-recomputes; staleness detection triggers recomputation

## Key File Map

| File | Purpose |
|------|---------|
| `config/auto_derive.rs` | `auto_derive()` function, `DerivedParams`, `CapitalProfile`, `ExchangeContext` |
| `config/spread_profile.rs` | `SpreadProfile` enum and profile-specific parameters |
| `config/capacity.rs` | `CapacityConfig` for ladder depth constraints |
| `config/core.rs` | Top-level config that calls `auto_derive()` |
| `orchestrator/quote_engine.rs` | Consumes `DerivedParams` for quote generation |
