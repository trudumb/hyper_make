# Session: Statistical Impulse Control Implementation Plan

**Date**: 2026-01-04
**Status**: Plan Complete ✅ | Implementation Pending ⏳

## Problem Statement

User identified that the market maker is **exhausting API rate limits** while achieving suboptimal fills. The core insight:

> "The cost of a Cancel/Replace is high (API limits), but the marginal benefit of moving a quote by 1 tick is often near zero."

**Current Architecture**: Continuous control pattern - updates on every market tick regardless of fill probability impact.

**Goal**: Reduce API churn by 50-70% while maintaining fill rate through first-principles fill probability filtering.

## Key Discovery: Dormant Infrastructure (80% Complete)

During exploration, found that queue-aware reconciliation infrastructure **already exists but is never called**:

### `src/market_maker/tracking/queue/tracker.rs`
```rust
// IMPLEMENTED but DORMANT - never called from reconciliation
pub fn fill_probability(&self, oid: u64, horizon_seconds: f64) -> Option<f64> {
    let p_touch = self.probability_touch(oid, horizon_seconds)?;
    let p_exec = self.probability_execute_given_touch(oid, horizon_seconds)?;
    Some(p_touch * p_exec)
}

pub fn should_refresh(&self, oid: u64, horizon_seconds: f64) -> bool {
    // Implemented but never called
}
```

### `src/market_maker/tracking/order_manager/reconcile.rs`
```rust
pub struct ReconcileConfig {
    // ... existing fields ...
    pub use_queue_aware: bool,        // EXISTS - UNUSED
    pub queue_horizon_seconds: f64,   // EXISTS - UNUSED
}
```

### `src/market_maker/infra/rate_limit.rs`
- `ProactiveRateLimitTracker` already has `record_fill_volume()` hook for token earning

## Solution Design

### Phase 1: Token Budget System (NEW)

Create `src/market_maker/infra/execution_budget.rs` (~150 lines):

```rust
pub struct ExecutionBudgetConfig {
    pub initial_tokens: f64,        // 100.0
    pub max_tokens: f64,            // 500.0
    pub emergency_reserve: f64,     // 20.0
    pub reward_per_1000_usd: f64,   // 0.5 tokens per $1000 filled
    pub min_tokens_per_fill: f64,   // 0.5 (for HIP-3 small notional)
    pub cost_per_action: 1.0,
    pub bulk_discount_factor: 0.8,
}

pub struct ExecutionBudget {
    available_tokens: f64,
    total_earned: f64,
    total_spent: f64,
    skipped_count: u64,
    emergency_uses: u64,
}
```

**Token Earning Formula** (based on Hyperliquid rate limit docs):
```rust
fn on_fill(&mut self, notional_usd: f64) {
    let volume_reward = (notional_usd / 1000.0) * self.config.reward_per_1000_usd;
    let reward = volume_reward.max(self.config.min_tokens_per_fill); // Floor for HIP-3
    self.available_tokens = (self.available_tokens + reward).min(self.config.max_tokens);
}
```

### Phase 2: Impulse Filter (NEW)

Create `src/market_maker/tracking/order_manager/impulse_filter.rs` (~120 lines):

```rust
pub struct ImpulseFilterConfig {
    pub improvement_threshold: f64,     // 0.10 (10% Δλ required)
    pub fill_horizon_seconds: f64,      // 1.0 second
    pub queue_lock_threshold: f64,      // 0.30 (lock if P(fill) > 30%)
    pub queue_lock_override_bps: f64,   // 25.0 (override if price moved >25bps)
}

pub enum ImpulseDecision { Update, Skip, Locked }

impl ImpulseFilter {
    pub fn evaluate(
        &self,
        queue_tracker: &QueuePositionTracker,
        oid: u64,
        current_price: f64,
        new_price: f64,
        price_diff_bps: f64,
        kappa: f64,
        sigma: f64,
    ) -> ImpulseDecision {
        // 1. Get current P(fill) from queue tracker
        let p_fill_current = queue_tracker.fill_probability(oid, horizon);

        // 2. If high P(fill), LOCK unless price moved significantly
        if p_fill_current > queue_lock_threshold && price_diff_bps < override_bps {
            return ImpulseDecision::Locked;
        }

        // 3. Calculate Δλ = (λ_new - λ_current) / λ_current
        let delta_lambda = (p_fill_new - p_fill_current) / p_fill_current;

        // 4. Only update if improvement exceeds threshold
        if delta_lambda > improvement_threshold {
            ImpulseDecision::Update
        } else {
            ImpulseDecision::Skip
        }
    }
}
```

### Phase 3: Integration (Wire Dormant Code)

**Modify `reconcile_side_smart()` in `src/market_maker/tracking/order_manager/reconcile.rs`**:

```rust
// After matching, BEFORE decision logic:
if config.use_impulse_filter {
    if let (Some(qt), Some((kappa, sigma))) = (queue_tracker, market_params) {
        let impulse = ImpulseFilter::new(config.impulse_config.clone());
        match impulse.evaluate(qt, m.order.oid, m.order.price, target_level.price, ...) {
            ImpulseDecision::Skip => continue,    // Skip this update
            ImpulseDecision::Locked => continue,  // Don't disturb high P(fill) order
            ImpulseDecision::Update => {}         // Proceed with normal logic
        }
    }
}
```

**Wire in `src/market_maker/mod.rs`**:
- Pass `queue_tracker` to `reconcile_side_smart()`
- Check execution budget before reconciliation
- Spend budget after execution
- Earn budget on fills

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `src/market_maker/infra/execution_budget.rs` | CREATE | ~150 |
| `src/market_maker/tracking/order_manager/impulse_filter.rs` | CREATE | ~120 |
| `src/market_maker/tracking/order_manager/reconcile.rs` | MODIFY | +30 |
| `src/market_maker/tracking/order_manager/mod.rs` | MODIFY | +2 |
| `src/market_maker/infra/mod.rs` | MODIFY | +1 |
| `src/market_maker/core/components.rs` | MODIFY | +5 |
| `src/market_maker/config.rs` | MODIFY | +25 |
| `src/market_maker/mod.rs` | MODIFY | +40 |
| `src/market_maker/infra/metrics.rs` | MODIFY | +30 |
| `src/bin/market_maker.rs` | MODIFY | +15 |

**Total: ~420 lines of code**

## User Preferences

- **Δλ Threshold**: 10% (more responsive to market changes)
- **Queue Lock**: 30% P(fill) threshold
- **Min tokens per fill**: 0.5 (for HIP-3 small notional markets)

## Default Configuration

```rust
ImpulseControlConfig {
    enabled: false,  // Disabled until validated
    budget: ExecutionBudgetConfig {
        initial_tokens: 100.0,
        max_tokens: 500.0,
        emergency_reserve: 20.0,
        reward_per_1000_usd: 0.5,
        min_tokens_per_fill: 0.5,
        cost_per_action: 1.0,
        bulk_discount_factor: 0.8,
    },
    filter: ImpulseFilterConfig {
        improvement_threshold: 0.10,
        fill_horizon_seconds: 1.0,
        queue_lock_threshold: 0.30,
        queue_lock_override_bps: 25.0,
    },
}
```

## Prometheus Metrics to Add

```
mm_impulse_budget_available    # Current tokens
mm_impulse_budget_utilization  # spent / (earned + initial)
mm_impulse_filter_blocked      # Actions blocked by Δλ filter
mm_impulse_queue_locked        # Actions blocked by queue lock
mm_impulse_budget_skipped      # Full cycles skipped due to budget
```

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| API calls/min | ~200+ | ~60-80 |
| Actions skipped | 0 | 50-70% |
| Fill rate | X | X (maintained) |
| Queue lock saves | 0 | 10-20% of high-prob orders |

## Rollback Strategy

| Component | Rollback |
|-----------|----------|
| Token Budget | Set `initial_tokens: f64::MAX` |
| Impulse Filter | Set `use_impulse_filter: false` |
| Queue Lock | Set `queue_lock_threshold: 1.0` |
| All | Set `impulse_control.enabled: false` |

## Plan File Location

Full implementation plan: `/home/jcritch22/.claude/plans/zippy-beaming-knuth.md`

## Implementation Order

1. Phase 1: Token Budget (isolated, no breaking changes)
2. Phase 2: Impulse Filter (isolated, no breaking changes)
3. Phase 3: Integration (wires Phase 1+2 + dormant code)
4. Phase 4: Config & CLI
5. Phase 5: Metrics

## Related Sessions

- `session_2026-01-04_calibration_fill_rate_controller.md` - Fill-hungry calibration (complements impulse control)
- `session_2026-01-04_queue_position_rate_limit_optimization.md` - Queue tracking implementation
- `session_2026-01-02_rate_limit_and_bulk_cancel.md` - Rate limit tracking infrastructure

## Hyperliquid Rate Limit Reference (from docs)

- **Per Address**: 1 request per $1 USDC traded (cumulative)
- **Initial Buffer**: 10,000 requests
- **IP Limit**: 1200 weight/minute
- **Implication**: Small HIP-3 fills ($13.50) earn almost nothing at volume-based rate → need `min_tokens_per_fill` floor
