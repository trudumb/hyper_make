# Session: Stochastic Constraints Implementation
**Date**: 2026-01-02
**Focus**: First-principles stochastic constraints for tight spread quoting

## Summary

Implemented comprehensive stochastic constraints based on `tight_spread_first_principles.md` analysis. These constraints ensure tight spreads are only used when market conditions support profitable market making.

## Key Implementation

### New Configuration Parameters (`StochasticConfig`)

```rust
// Latency-based spread floor
use_latency_spread_floor: bool,       // Enable σ × √(2×τ_update) floor
quote_update_latency_ms: f64,         // Default: 50ms

// Book depth constraints
use_book_depth_constraint: bool,      // Enable depth checking
min_book_depth_usd: f64,              // Default: $50k minimum
tight_spread_book_depth_usd: f64,     // Default: $200k for tight spreads

// Conditional tight quoting
use_conditional_tight_quoting: bool,  // Enable all prerequisites
tight_quoting_max_inventory: f64,     // Default: 0.3 (30%)
tight_quoting_max_toxicity: f64,      // Default: 0.1 (jump_ratio-1)
tight_quoting_excluded_hours: Vec<u8>,// Default: [7, 14] UTC
```

### New MarketParams Fields

```rust
tick_size_bps: f64,              // Tick size constraint
latency_spread_floor: f64,       // σ × √(2×τ_update/1000)
near_touch_depth_usd: f64,       // Book depth near touch
tight_quoting_allowed: bool,     // All conditions met?
tight_quoting_block_reason: Option<String>,
stochastic_spread_multiplier: f64, // 1.5x when tight blocked
```

### Core Logic: `compute_stochastic_constraints()`

Evaluates 5 conditions for tight quoting:
1. **Volatility Regime**: Must be Low or Normal (not High/Extreme)
2. **Toxicity**: jump_ratio - 1 < threshold (default 0.1)
3. **Inventory**: |position|/max_position < threshold (default 0.3)
4. **Time of Day**: Not during excluded hours (default: 7, 14 UTC)
5. **Book Depth**: near_touch_depth_usd >= min_book_depth_usd

When ANY condition fails → `stochastic_spread_multiplier = 1.5` (widen spreads 50%)

### Effective Spread Floor

```rust
pub fn effective_spread_floor(&self, risk_config_floor: f64) -> f64 {
    let tick_floor = self.tick_size_bps / 10_000.0;
    risk_config_floor
        .max(tick_floor)
        .max(self.latency_spread_floor)
}
```

Returns maximum of:
- Static `min_spread_floor` from RiskConfig
- Tick size (can't quote finer than tick)
- Latency floor: σ × √(2×τ_update) (update delay cost)

## Files Modified

1. `src/market_maker/mod.rs` - Added chrono::Timelike, calls compute_stochastic_constraints()
2. `src/market_maker/config.rs` - Added StochasticConfig parameters
3. `src/market_maker/strategy/market_params.rs` - Added fields and compute method
4. `src/market_maker/strategy/params.rs` - Added StochasticConstraintParams
5. `src/market_maker/strategy/glft.rs` - Uses effective_spread_floor and multiplier
6. `src/market_maker/strategy/ladder_strat.rs` - Applies floor and multiplier to depths
7. `src/market_maker/estimator/parameter_estimator.rs` - Added near_touch_depth_usd()
8. `src/market_maker/estimator/kappa.rs` - Added near_touch_depth() to BookStructureEstimator

## Integration Points

### GLFT Strategy
```rust
// Uses effective floor instead of static floor
let effective_floor = market_params.effective_spread_floor(self.risk_config.min_spread_floor);
half_spread_bid = half_spread_bid.max(effective_floor);

// Applies stochastic multiplier when tight quoting blocked
if market_params.stochastic_spread_multiplier > 1.0 {
    half_spread_bid *= market_params.stochastic_spread_multiplier;
}
```

### Ladder Strategy
```rust
// Apply stochastic floor to dynamic depths
for depth in dynamic_depths.bid.iter_mut() {
    if *depth < effective_floor_bps {
        *depth = effective_floor_bps;
    }
}

// Apply multiplier when tight quoting blocked
if market_params.stochastic_spread_multiplier > 1.0 {
    for depth in dynamic_depths.bid.iter_mut() {
        *depth *= market_params.stochastic_spread_multiplier;
    }
}
```

## Related Sessions
- `session_2026-01-01_spread_analysis` - Trade history analysis revealing toxic hours
- `tight_spread_first_principles` - Theory document analyzed for implementation
- `session_2026-01-01_ws_order_state_integration` - Previous session foundation

## Testing
- All 458 tests pass
- Clippy clean (no warnings)
- Build successful

## Next Steps
- Monitor production behavior during excluded hours (7, 14 UTC)
- Tune `tight_quoting_max_toxicity` based on observed AS
- Consider adding more excluded hours based on trade history
- Get actual tick_size_bps from asset metadata (currently hardcoded to 10bps)
