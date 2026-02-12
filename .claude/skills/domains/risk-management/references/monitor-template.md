# Risk Monitor Implementation Template

Complete guide for implementing a new `RiskMonitor` in the Hyperliquid market maker.
Includes trait skeleton, registration, testing requirements, common patterns, and checklist.

---

## 1. Trait Implementation Skeleton

All monitors implement the `RiskMonitor` trait from `risk/monitor.rs`.
Monitors are stateless -- all state comes from `RiskState`.

```rust
//! My new risk monitor.

use crate::market_maker::risk::{RiskAction, RiskAssessment, RiskMonitor, RiskState};

/// Monitors [describe what this monitors].
///
/// [Explain the risk scenario and why this monitor exists.]
pub struct MyNewMonitor {
    /// Threshold for widening spreads
    widen_threshold: f64,
    /// Threshold for pulling quotes
    pull_threshold: f64,
    /// Threshold for kill switch
    kill_threshold: f64,
}

impl MyNewMonitor {
    /// Create a new monitor with specified thresholds.
    ///
    /// # Arguments
    ///
    /// * `pull_threshold` - Value at which to pull quotes
    /// * `kill_threshold` - Value at which to trigger kill switch
    pub fn new(pull_threshold: f64, kill_threshold: f64) -> Self {
        Self {
            widen_threshold: pull_threshold * 0.5,
            pull_threshold,
            kill_threshold,
        }
    }
}

impl Default for MyNewMonitor {
    fn default() -> Self {
        // Provide sensible defaults -- never leave thresholds unset
        Self::new(/* pull */ 0.8, /* kill */ 5.0)
    }
}

impl RiskMonitor for MyNewMonitor {
    fn evaluate(&self, state: &RiskState) -> RiskAssessment {
        let metric = state.some_relevant_field; // Read from RiskState only

        // Kill switch at extreme values
        if metric > self.kill_threshold {
            return RiskAssessment::critical(
                self.name(),
                format!(
                    "Extreme condition: metric {:.2} > {:.2}",
                    metric, self.kill_threshold
                ),
            )
            .with_metric(metric)
            .with_threshold(self.kill_threshold);
        }

        // Pull quotes at high values
        if metric > self.pull_threshold {
            return RiskAssessment::high(
                self.name(),
                RiskAction::PullQuotes,
                format!(
                    "Dangerous condition: metric {:.2} > {:.2}",
                    metric, self.pull_threshold
                ),
            )
            .with_metric(metric)
            .with_threshold(self.pull_threshold);
        }

        // Widen spreads at moderate values
        if metric > self.widen_threshold {
            let excess = metric - self.widen_threshold;
            let factor = 1.0 + excess * 0.5; // +50% spread per unit above threshold
            return RiskAssessment::high(
                self.name(),
                RiskAction::WidenSpreads(factor),
                format!(
                    "Elevated risk: metric {metric:.2}, widening {factor:.1}x"
                ),
            )
            .with_metric(metric)
            .with_threshold(self.widen_threshold);
        }

        // Default: no risk detected
        RiskAssessment::ok(self.name())
    }

    fn name(&self) -> &'static str {
        "MyNewMonitor"
    }

    fn priority(&self) -> u32 {
        // 0-10: Critical monitors (loss, position, price velocity)
        // 10-50: High priority (cascade, flash crash)
        // 50-100: Standard (drawdown, data staleness)
        // 100+: Informational
        50
    }

    // Optional: disable the monitor dynamically
    // fn is_enabled(&self) -> bool { true }
}
```

---

## 2. Where to Register the Monitor

**File**: `risk/aggregator.rs`

Monitors are registered with `RiskAggregator` using the builder pattern or `add_monitor()`:

```rust
// Builder pattern (at construction time)
let aggregator = RiskAggregator::new()
    .with_monitor(Box::new(LossMonitor::new(500.0)))
    .with_monitor(Box::new(PositionMonitor::new(10000.0)))
    .with_monitor(Box::new(MyNewMonitor::default()));  // Add yours

// Mutable (after construction)
aggregator.add_monitor(Box::new(MyNewMonitor::new(0.8, 5.0)));
```

The aggregator automatically sorts monitors by priority (lower priority number = evaluated first).

**Registration location**: Find where the `RiskAggregator` is constructed in
`core/components.rs` or the initialization code for your deployment mode.
Add your monitor there.

---

## 3. Testing Requirements

**All severity transitions must be tested.** This is non-negotiable for safety code.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::risk::RiskSeverity;

    #[test]
    fn test_calm_returns_ok() {
        let monitor = MyNewMonitor::default();
        let state = RiskState {
            some_relevant_field: 0.1,  // Well below threshold
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::None);
        assert!(!assessment.is_actionable());
    }

    #[test]
    fn test_moderate_widens_spreads() {
        let monitor = MyNewMonitor::default();
        let state = RiskState {
            some_relevant_field: 0.6,  // Above widen, below pull
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        if let RiskAction::WidenSpreads(factor) = assessment.action {
            assert!(factor > 1.0);
        } else {
            panic!("Expected WidenSpreads action");
        }
    }

    #[test]
    fn test_high_pulls_quotes() {
        let monitor = MyNewMonitor::default();
        let state = RiskState {
            some_relevant_field: 1.0,  // Above pull threshold
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::High);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
    }

    #[test]
    fn test_extreme_triggers_kill() {
        let monitor = MyNewMonitor::default();
        let state = RiskState {
            some_relevant_field: 6.0,  // Above kill threshold
            ..Default::default()
        };

        let assessment = monitor.evaluate(&state);
        assert_eq!(assessment.severity, RiskSeverity::Critical);
        assert!(assessment.should_kill());
    }

    #[test]
    fn test_boundary_values() {
        let monitor = MyNewMonitor::new(0.8, 5.0);
        // Test exactly at threshold
        let state = RiskState {
            some_relevant_field: 0.8,
            ..Default::default()
        };
        let assessment = monitor.evaluate(&state);
        // Verify behavior at exact boundary (> vs >=)
        assert!(assessment.is_actionable());
    }

    #[test]
    fn test_custom_thresholds() {
        let monitor = MyNewMonitor::new(0.5, 2.0);
        // Verify custom thresholds work
        let state = RiskState {
            some_relevant_field: 0.6,
            ..Default::default()
        };
        let assessment = monitor.evaluate(&state);
        assert!(matches!(assessment.action, RiskAction::PullQuotes));
    }
}
```

---

## 4. Common Patterns

### Threshold Escalation (3-tier)

The standard pattern used by `CascadeMonitor`, `PriceVelocityMonitor`, etc.:

```
Metric below widen_threshold  -> RiskAssessment::ok()        (Normal)
Metric above widen_threshold  -> RiskAction::WidenSpreads(f) (High)
Metric above pull_threshold   -> RiskAction::PullQuotes      (High)
Metric above kill_threshold   -> RiskAction::Kill(reason)    (Critical)
```

### Hysteresis

Prevent oscillation between states. Use separate thresholds for entering and exiting:

```rust
// Enter cautious mode at 0.8, exit at 0.6
let enter_threshold = 0.8;
let exit_threshold = 0.6;  // 20% lower to prevent oscillation
```

The graduated emergency pull system (from Drift Audit) uses 3 tiers with hysteresis.

### Cooldown Timers

After triggering an action, require a minimum time before downgrading:

```rust
// If you need stateful cooldown, track it outside the monitor
// (monitors are stateless, so cooldown state lives in RiskState or externally)
```

### Available RiskAction Variants

From `risk/monitor.rs`:

| Action | Severity | Spread Mult | Size Mult | Use When |
|--------|----------|-------------|-----------|----------|
| `None` | None | 1.0 | 1.0 | All clear |
| `Warn(msg)` | Low | 1.0 | 1.0 | Log only |
| `WidenSpreads(f)` | Medium | f | 0.8 | Elevated risk |
| `SkewAway(bps)` | Medium | 1.2 | 0.8 | Directional risk |
| `ReduceOnly` | High | 1.5 | 0.5 | Over position limit |
| `PullSide(buys)` | High | 1.5 | 0.5 | One-sided risk |
| `PullQuotes` | High | INF | 0.0 | Dangerous conditions |
| `Kill(reason)` | Critical | INF | 0.0 | Emergency shutdown |

### RiskAssessment Convenience Constructors

```rust
RiskAssessment::ok("MonitorName")                          // No risk
RiskAssessment::warn("MonitorName", "description")         // Medium severity
RiskAssessment::high("MonitorName", action, "description") // High severity
RiskAssessment::critical("MonitorName", "reason")          // Kill switch

// Builder methods for metrics tracking:
.with_metric(value)      // Numeric metric value
.with_threshold(thresh)  // Threshold that was exceeded
```

---

## 5. Implementation Checklist

Before merging a new risk monitor, verify:

- [ ] **Clear reason strings**: Every non-ok assessment has a human-readable description
      that includes the metric value and threshold (e.g., "velocity 0.08/s > 0.05/s")
- [ ] **Configurable thresholds**: No hardcoded magic numbers -- all thresholds are
      constructor parameters with sensible defaults
- [ ] **Default to Normal**: When uncertain or when data is missing, return
      `RiskAssessment::ok()` -- defense-first means widening elsewhere, not false-alarming
- [ ] **Never panic**: Monitor evaluation must never panic. Use `.unwrap_or()`,
      `.clamp()`, and `.max()` defensively on all numeric operations
- [ ] **All severity transitions tested**: Tests cover: ok, widen, pull, kill, and
      boundary values for each threshold
- [ ] **Priority set correctly**: Critical monitors (loss, position) = 0-10,
      market-condition monitors = 10-50, informational = 100+
- [ ] **RiskState only**: Read from `RiskState` fields only -- do not add new data
      sources to the evaluation hot path
- [ ] **Metric and threshold attached**: Use `.with_metric()` and `.with_threshold()`
      so the aggregator can track and log trigger details
- [ ] **Send + Sync**: The `RiskMonitor` trait requires `Send + Sync` -- no interior
      mutability without synchronization
- [ ] **Registered in aggregator**: Monitor is added to `RiskAggregator` in the
      appropriate initialization path
