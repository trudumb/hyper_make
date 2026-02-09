---
description: Safety-critical rules for risk, safety, orchestrator, and binary modules
globs:
  - "src/market_maker/risk/**"
  - "src/market_maker/safety/**"
  - "src/market_maker/orchestrator/**"
  - "src/bin/**"
---

# Safety-Critical Code Rules

These modules directly control position limits and emergency shutdowns. Bugs here cause immediate financial loss.

## Inventory Invariants

- `inventory.abs() <= max_inventory` — hard limit, never violated
- `ask_price > bid_price` — spread invariant must hold in all code paths
- Kill switch state persists across restarts

## Risk Aggregation

- `RiskAggregator` takes **maximum severity** across monitors — one Critical overrides five Normals
- New monitors should default to `RiskSeverity::Normal` when uncertain
- All thresholds must be configurable, not hardcoded

## Defense First

- When uncertain, widen spreads — missing a trade is cheap; getting run over is not
- Push back on aggressive parameter changes from other agents
- Circuit breakers handle market-condition triggers (OI drops, funding extremes)
- Safety auditor runs periodic reconciliation between local and exchange state

## Review Requirements

Changes to these modules require extra scrutiny:
- All severity transitions must be tested
- Default to safer option in ambiguous cases
- Position limits enforced in all code paths
- Kill switch handles all emergency scenarios
- Clear reason strings for all risk assessments
- Plan approval required for orchestrator/ changes
