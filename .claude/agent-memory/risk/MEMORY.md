# Risk Agent Memory

## Risk System Validation (2026-02-07)

Comprehensive audit of all risk systems for production readiness.

### Findings

1. **Kill switch 7 triggers**: All wired and tested. Production preset correct.
2. **Position guard entry gate**: FIXED. Now wired in `place_new_order()` (order_ops.rs).
   `PositionGuard` added to `SafetyComponents`. Hard entry gate rejects orders that exceed
   95% of max_position (worst-case). Stateless check uses live position, no stale data.
3. **RiskAggregator rate limits**: Fully wired. `RateLimitMonitor` reads from `RiskState`.
4. **Cascade false positives**: `cascade_active` flag (3x baseline) is more sensitive than
   `cascade_severity` (normalized to 5x). Not a bug, design choice. RiskAggregator only acts at
   severity > 0.4 which is correct.
5. **DrawdownMonitor threshold BUG**: FIXED. Removed `/ 100.0` from
   `DrawdownMonitor::new(config.max_drawdown)`. Was 100x too sensitive (0.02% vs 2%).
6. **Reentry manager**: Well-implemented with production preset.
7. **Kill switch persistence**: FIXED (2026-02-09). Now in CheckpointBundle with 24h expiry.

### Key Architecture Notes

- `SafetyComponents` (core/components.rs:155): kill_switch, risk_aggregator, fill_processor,
  risk_checker, drawdown_tracker, signal_store, position_guard.
- `build_risk_aggregator()` (mod.rs): LossMonitor, DrawdownMonitor, PositionMonitor,
  DataStalenessMonitor, CascadeMonitor(0.8, 0.95), RateLimitMonitor, PriceVelocityMonitor
- `cascade_severity()` in liquidation.rs:436: `((ratio - 1.0) / (threshold - 1.0)).clamp(0,1)`
- `cascade_active` flag: intensity > 3x baseline (separate from severity normalization)
- Kill switch has dual protection: KillSwitch::check() AND RiskAggregator monitors
- CheckpointBundle saves model parameters + kill switch state (since 2026-02-09)

### Flash Crash Protection (2026-02-09)

8. **PriceVelocityMonitor**: NEW monitor in `risk/monitors/price_velocity.rs`. Tracks
   `price_velocity_1s` (abs(delta_mid/mid)/elapsed_s) computed in `handle_all_mids()`.
   Pull quotes at 5%/s (configurable via `price_velocity_threshold`), kill at 15%/s (3x).
   Priority 10 (highest among monitors). Field added to RiskState + KillSwitchConfig.
   Velocity tracked via `last_mid_for_velocity` and `last_mid_velocity_time` on MarketMaker.

9. **Liquidation self-detection**: Added to kill_switch.rs. Tracks `last_fill_time` (updated
   via `record_own_fill()` called after fill processing in handlers.rs). On `update_position()`,
   if position delta > 20% of max_position AND no fill in last 5s, triggers kill switch with
   `LiquidationDetected` reason. Startup safety: skips until first fill recorded (avoids false
   positive from exchange position sync). Config: `liquidation_position_jump_fraction` (0.20),
   `liquidation_fill_timeout_s` (5).

**Gotcha**: `update_position()` now triggers liquidation detection, so existing tests that call
it without recording fills first may fail. Fixed `test_summary` by the startup guard (no trigger
before first fill). New tests must set `last_fill_time` to expired instant for trigger tests.

### PositionVelocityMonitor (2026-02-09)

10. **PositionVelocityMonitor**: NEW monitor in `risk/monitors/position_velocity.rs`. Reads
    `position_velocity_1m` from RiskState (abs position change per minute, normalized by max_position).
    Three thresholds: warn (0.50, widen 1.5-3x), pull (1.00), kill (2.00). Priority 12.
    Config: `position_velocity_threshold` on KillSwitchConfig (default 0.50).
    Use `PositionVelocityMonitor::from_base_threshold(config.position_velocity_threshold)`.

11. **unsafe impl Send/Sync FIXED**: Replaced with safe `const _: fn() = || { ... }` pattern
    on DrawdownTracker, CircuitBreakerMonitor, and RiskChecker.

### InventoryGovernor (2026-02-12)

12. **InventoryGovernor**: New module in `risk/inventory_governor.rs`. Enforces `config.max_position`
    as HARD ceiling — margin-derived limits can only LOWER it, never raise above config.
    Zone classification: Green (<50%), Yellow (50-80%), Red (80-100%), Kill (>100%).
    Methods: `new()`, `assess()`, `max_position()`, `would_exceed()`, `is_reducing()`.
    19 unit tests. Wired into: reconcile.rs (place_bulk_ladder_orders), order_ops.rs (place_new_order),
    ladder_strat.rs (effective_max_position cap), glft.rs (same), quote_engine.rs (unconditional cap),
    build_risk_state (unconditional cap). MarketMaker struct field initialized from config.max_position.

    **Key fix**: `max_position_user_specified` conditional was REMOVED from caps — config.max_position
    is ALWAYS the ceiling. Previously, when `max_position_user_specified=false`, margin-derived limits
    (e.g., 55 HYPE) could override user's config (3.24 HYPE), causing 3.95x overshoot.

### Full Audit (2026-02-09)

**Overall: 8.5/10 production-ready.** 3 HIGH, 3 MEDIUM, 5 LOW, 3 INFORMATIONAL findings.

**HIGH findings (status):**
- Inventory breach: entry gate is pre-flight only, exchange sync + clustered fills bypass it
- Reduce-only not hard-enforced: `check_reduce_only()` only logs, quote engine skew is soft
- ~~No position velocity monitor~~ FIXED (2026-02-09): PositionVelocityMonitor added

**MEDIUM findings (status):**
- ~~`unsafe impl Send/Sync`~~ FIXED (2026-02-09): replaced with safe compile-time assertions
- Drawdown denominator inconsistency: DrawdownMonitor uses account_value, KillSwitch uses peak_pnl
- `effective_max_position` not propagated to PositionGuard (guard uses config.max_position)

**All PASS items:** 7 monitors correct, action escalation chain complete, kill switch 9 triggers verified, checkpoint persistence with 24h expiry, PostMortemDump wired on kill, Alerter thread-safe with dedup, circuit breakers covering 5 conditions, reentry manager with multi-kill escalation.
