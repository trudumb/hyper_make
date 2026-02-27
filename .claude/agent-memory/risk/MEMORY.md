# Risk Agent Memory

## Current State
- 9 kill switch triggers, checkpoint persistence with 24h expiry
- InventoryGovernor: hard position ceiling, zone classification (Green/Yellow/Red/Kill)
- PriceVelocityMonitor: pull 5%/s, kill 15%/s (priority 10)
- PositionVelocityMonitor: warn 0.50, pull 1.00, kill 2.00 (priority 12)
- Direction hysteresis: per-side gamma penalty after zero-crossing

## Open Issues
- Inventory breach: entry gate is pre-flight only, exchange sync + clustered fills bypass it
- Reduce-only not hard-enforced: `check_reduce_only()` only logs, skew is soft
- Drawdown denominator inconsistency: DrawdownMonitor uses account_value, KillSwitch uses peak_pnl
- `effective_max_position` not propagated to PositionGuard (guard uses config.max_position)

## Active Gotchas
- `update_position()` triggers liquidation detection — tests that call it without recording fills first will fail
- New tests must set `last_fill_time` to expired instant for trigger tests
- `RiskAggregator` takes MAXIMUM severity — one Critical overrides five Normals
- DrawdownMonitor: was 100x too sensitive before removing `/ 100.0` (Feb 7 fix)
