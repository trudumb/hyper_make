---
description: Rust coding conventions for the market making codebase
globs:
  - "src/**/*.rs"
---

# Coding Conventions

## Type Safety

- Use strong newtypes for domain concepts: `Bps(f64)`, `Price(f64)`, `Size(f64)`
- Use `const` for magic numbers, never inline
- `#[serde(default)]` on all checkpoint fields for backward compatibility

## Unit Documentation

Document units in variable names — unit mismatches cause real money loss:
- `spread_bps` not `spread` — bps vs fraction
- `time_to_settlement_s` not `time_to_settlement` — seconds vs milliseconds
- `funding_rate_8h` not `funding_rate` — 8h vs annualized
- `sigma_realized_1m` — timeframe in the name

## Parameter Discipline

- No hardcoded parameters — all values must be regime-dependent or configurable
- `kappa > 0.0` invariant in all formula paths — GLFT blows up at zero
- `gamma > 0.0` — risk aversion must be positive
- For `too_many_arguments`: prefer parameter structs over `#[allow]` suppressions

## Code Patterns

- Prefer `validated()` builder pattern for structs with public fields that need clamping
- `let _ =` pattern for analytics/logging calls that must never crash the trader
- Arena allocator on hot path — no heap allocation during quote cycle
