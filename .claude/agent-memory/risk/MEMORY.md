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
