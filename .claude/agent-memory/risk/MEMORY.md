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
7. **Kill switch persistence**: NOT in CheckpointBundle. Kill switch state lost on restart.

### Key Architecture Notes

- `SafetyComponents` (core/components.rs:155): kill_switch, risk_aggregator, fill_processor,
  risk_checker, drawdown_tracker, signal_store, position_guard.
- `build_risk_aggregator()` (mod.rs:703): LossMonitor, DrawdownMonitor, PositionMonitor,
  DataStalenessMonitor, CascadeMonitor(0.8, 0.95), RateLimitMonitor
- `cascade_severity()` in liquidation.rs:436: `((ratio - 1.0) / (threshold - 1.0)).clamp(0,1)`
- `cascade_active` flag: intensity > 3x baseline (separate from severity normalization)
- Kill switch has dual protection: KillSwitch::check() AND RiskAggregator monitors
- CheckpointBundle only saves model parameters, not safety state
