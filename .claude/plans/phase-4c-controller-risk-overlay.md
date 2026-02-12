# Phase 4c: Controller -> Risk Overlay

## Objective

Transform StochasticController from a decision-maker into a risk overlay by adding a `risk_assessment()` method that returns continuous risk multipliers. The existing `act()` method is preserved for Phase 4d migration.

## Files Modified

1. **`src/market_maker/control/mod.rs`** (owned by risk domain)
   - Add `RiskAssessment` struct (alongside existing `ControllerResult`, `SystemHealthReport`)
   - Add `risk_assessment()` method to `impl StochasticController[0]` block
   - Add re-export of `RiskAssessment`
   - Add 3 new tests

## Implementation Details

### RiskAssessment struct (after ControllerResult, ~line 713)

```rust
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub size_multiplier: f64,      // [0.1, 1.0]
    pub spread_multiplier: f64,    // [1.0, 3.0]
    pub emergency_pull: bool,
    pub reason: String,
}
```

With `Default` impl returning neutral values (1.0, 1.0, false, "normal").

### risk_assessment() method (in impl StochasticController[0], after learning_trust())

Uses existing fields:
- `self.learning_trust` (already computed in `act()`) -> size_multiplier
- `self.changepoint.changepoint_probability(5)` -> spread_multiplier + emergency
- Handles cold start (cycle_count == 0 or changepoint not warmed up) -> default

### Tests

1. `test_risk_assessment_default_cold` - Fresh controller returns neutral
2. `test_risk_assessment_high_changepoint` - Feed sudden shift, verify spread widening + emergency
3. `test_risk_assessment_low_trust` - Degraded health reduces size

## Verification

```
cargo test --lib -- control::tests
cargo clippy -- -D warnings
```

## Risk

LOW - purely additive new method and struct, no existing code modified. `act()` untouched.
