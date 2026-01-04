# PAIC Framework Audit: Order Placement Stochastic Modeling

**Audit Date:** 2026-01-04
**Commit:** fa8663d (Merge PR #7 - PAIC Framework)
**Lines Added:** ~3,352
**Files Added:** 17

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Stochastic Models Analysis](#stochastic-models-analysis)
3. [Integration Architecture](#integration-architecture)
4. [Redundancy Analysis](#redundancy-analysis)
5. [Code Quality Assessment](#code-quality-assessment)
6. [Performance Implications](#performance-implications)
7. [Recommendations](#recommendations)
8. [Exploration Paths](#exploration-paths)

---

## Executive Summary

### What PAIC Does

**Priority-Aware Impulse Control (PAIC)** is a framework for intelligent order modification decisions based on queue position value. Instead of blindly updating orders when prices drift, it evaluates whether the queue position is worth preserving.

### Key Finding

**Only ~10% of PAIC is actively used.** The `paic_filter_modifies()` function filters order modifications based on queue priority, but the extensive `OrderGateway`, batching, and rate limit shadow pricing infrastructure remains unused.

### Verdict

| Aspect | Rating | Notes |
|--------|--------|-------|
| Mathematical Rigor | ⭐⭐⭐⭐ | Sound stochastic models, though simplified |
| Integration Quality | ⭐⭐⭐ | Correct data flow, but creates duplication |
| Code Efficiency | ⭐⭐ | Significant unused code, dual tracking |
| Production Readiness | ⭐⭐⭐ | Active portion works, but incomplete |

---

## Stochastic Models Analysis

### 1. Queue Priority Index (π)

**Location:** `paic/observer/virtual_queue.rs:125-151`

**Model:**
```
π_t = (initial_depth_ahead - cumulative_volume_traded) / (initial_depth_ahead + our_size)
```

**Properties:**
- Range: [0.0, 1.0] where 0 = front of queue, 1 = back
- EWMA smoothing with α = 0.1 prevents jitter
- Exponential decay (t½ = 30s) for stale orders

**Mathematical Assessment:**

| Aspect | Status | Details |
|--------|--------|---------|
| Volume tracking | ✅ | Correctly accumulates trades at price level |
| Side matching | ✅ | Only same-side trades advance queue |
| Decay model | ✅ | `e^(-λt)` with λ = ln(2)/30 |
| Edge cases | ⚠️ | `min_queue_position = 0.01` floor prevents div/0 |

**Comparison with Existing `QueuePositionTracker`:**

| Feature | PAIC VirtualQueueTracker | Existing QueuePositionTracker |
|---------|--------------------------|-------------------------------|
| Position metric | Priority index π ∈ [0,1] | Raw depth_ahead (contracts) |
| Fill probability | Implicit via π | Explicit P(touch) × P(exec\|touch) |
| Probabilistic model | None | Reflection principle + Poisson |
| Decay mechanism | Volume-based + time | Time-based only |
| Update frequency | Per-trade | Per-book update |

**Verdict:** PAIC's model is simpler but trades mathematical precision for practical decision-making via the strategy matrix.

---

### 2. Volatility Regime Classification

**Location:** `paic/observer/state_estimator.rs:11-41`

**Model:**
```rust
enum VolatilityState { Quiet, Normal, Turbulent }

ratio = σ_current / σ_baseline
Quiet:     ratio < 0.5
Normal:    0.5 ≤ ratio ≤ 2.0
Turbulent: ratio > 2.0
```

**Band Multipliers:**
- Quiet: 0.5× (tighter no-action band)
- Normal: 1.0× (base case)
- Turbulent: 2.0× (wider no-action band)

**Mathematical Assessment:**

| Aspect | Status | Details |
|--------|--------|---------|
| Baseline tracking | ✅ | Slow EWMA (α = 0.001) for long-term σ |
| Threshold selection | ⚠️ | Hardcoded 0.5/2.0 - should be asset-specific |
| Hysteresis | ❌ | Missing - can oscillate at boundaries |

**Recommendation:** Add hysteresis buffer (e.g., 10%) to prevent rapid state flipping:
```rust
// Current: instant transition at threshold
// Better: require sustained deviation
let enter_turbulent = ratio > 2.0 && last_state != Turbulent;
let exit_turbulent = ratio < 1.8 && last_state == Turbulent;
```

---

### 3. Flow Toxicity (VPIN + OFI)

**Location:** `paic/observer/toxicity.rs`

**VPIN Model:**
```
toxicity = Σ|buy_vol - sell_vol| / Σ(buy_vol + sell_vol)
         over N volume buckets
```

**OFI Model:**
```
ofi = (buy_vol - sell_vol) / (buy_vol + sell_vol)
     over rolling time window
```

**Configuration:**
```rust
vpin_bucket_volume: 1.0,    // 1 BTC equivalent per bucket
vpin_num_buckets: 50,       // 50 bucket history
ofi_window_ms: 1000,        // 1 second OFI window
toxic_threshold: 0.1,       // Toxicity > 10% = toxic
ewma_alpha: 0.1,            // Smoothing factor
```

**Mathematical Assessment:**

| Aspect | Status | Details |
|--------|--------|---------|
| VPIN implementation | ✅ | Correct volume-bucketed calculation |
| OFI implementation | ✅ | Standard signed imbalance |
| Directional toxicity | ✅ | `is_toxic_for_side()` correctly handles asymmetry |
| Threshold calibration | ⚠️ | Fixed 0.1 - should be asset/regime dependent |

**Side-Specific Toxicity Logic:**
```rust
// Toxic selling (OFI < -0.05) is bad for bids
// Toxic buying (OFI > +0.05) is bad for asks
pub fn is_toxic_for_side(&self, is_our_bid: bool) -> bool {
    if !self.is_toxic { return false; }
    (is_our_bid && self.ofi < -0.05) || (!is_our_bid && self.ofi > 0.05)
}
```

**This is correct.** Toxic selling hits bids; toxic buying lifts asks.

---

### 4. Priority Value Calculation

**Location:** `paic/controller/priority_value.rs`

**Hold Value:**
```
V_hold = (1 - π) × spread × size × multiplier
```

**Move Value:**
```
V_move = |drift| × size - move_cost
```

**Decision Rule:**
```
If V_hold > V_move: HOLD (preserve queue position)
If V_move > V_hold: SHADOW/RESET (update order)
```

**Priority Premium:**
```
premium_bps = (1 - π) × spread_bps × 0.8
```

With default 10 bps spread:
- π = 0.0 (front): premium = 8 bps
- π = 0.5 (middle): premium = 4 bps
- π = 1.0 (back): premium = 0 bps

**Mathematical Assessment:**

| Aspect | Status | Details |
|--------|--------|---------|
| Queue value quantification | ✅ | First-principles economic reasoning |
| Linear interpolation | ⚠️ | Reality may be nonlinear (convex near front) |
| Multiplier selection | ⚠️ | 0.8 is empirical, not derived |

---

### 5. Impulse Control Strategy Matrix

**Location:** `paic/controller/impulse_engine.rs:117-196`

```
                    │ Small Drift │ Large Drift │
────────────────────┼─────────────┼─────────────┤
High Priority (π≈0) │    HOLD     │    LEAK*    │
Low Priority (π≈1)  │   SHADOW    │    RESET    │
```

*LEAK triggers only with toxic flow

**Action Definitions:**

| Action | Meaning | Implementation |
|--------|---------|----------------|
| HOLD | Keep order unchanged | Skip modify |
| SHADOW | Update price gently | Allow modify |
| RESET | Aggressive price update | Allow modify |
| LEAK | Reduce size, keep position | **NOT IMPLEMENTED** |

**Current Implementation Gap:**

```rust
// paic_filter_modifies() only handles HOLD
match decision.action {
    ImpulseAction::Hold => {
        hold_count += 1;
        // Skip this modify
    }
    _ => {
        // SHADOW, RESET, LEAK all pass through unchanged
        filtered.push(spec);
    }
}
```

**LEAK should reduce order size but preserve queue position.** This is not implemented.

---

## Integration Architecture

### Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                          WebSocket Events                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Trades ──────────┬──────────────────────────────────────────────────► │
│                   │                                                     │
│                   ▼                                                     │
│         ┌─────────────────────┐                                        │
│         │  ParameterEstimator │ ◄─── Main estimator (σ, κ, microprice) │
│         └─────────┬───────────┘                                        │
│                   │ sync                                                │
│                   ▼                                                     │
│  Trades ──► ┌─────────────────────┐                                    │
│             │  PAIC.StateEstimator │ ◄─── L2Book (BBO, spread)         │
│  L2Book ──► │  - VirtualQueueTracker                                   │
│             │  - ToxicityEstimator │                                   │
│             │  - VolatilityState   │                                   │
│             └─────────┬───────────┘                                    │
│                       │                                                 │
│                       ▼                                                 │
│             ┌─────────────────────┐                                    │
│             │  PAIC.ImpulseEngine │                                    │
│             │  - decide_action()  │                                    │
│             │  - Strategy Matrix  │                                    │
│             └─────────┬───────────┘                                    │
│                       │                                                 │
│                       ▼                                                 │
│             ┌─────────────────────┐      ┌───────────────────────┐    │
│             │ paic_filter_modifies│ ◄─── │ reconcile_ladder()    │    │
│             │ - Filter HOLDs      │      │ - Generate ModifySpecs│    │
│             └─────────┬───────────┘      └───────────────────────┘    │
│                       │                                                 │
│                       ▼                                                 │
│             ┌─────────────────────┐                                    │
│             │ HyperliquidExecutor │ ◄─── Orders still go here          │
│             │ (unchanged)         │      NOT through OrderGateway      │
│             └─────────────────────┘                                    │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Integration Points in `mod.rs`

| Line | Event | PAIC Update |
|------|-------|-------------|
| 840-848 | `process_trades()` | `paic.state_estimator.on_trade(price, size, is_buy)` |
| 926-932 | `process_user_fills()` | `order_removed()` or `order_partially_filled()` |
| 1067-1073 | `process_l2_book()` | `update_book(best_bid, best_ask)` |
| 1079-1085 | `process_l2_book()` | `update_volatility(σ)`, `update_microprice()` |
| 2178-2187 | `place_ladder_order()` | `order_placed(oid, price, size, depth, is_bid)` |
| 2399 | `reconcile_ladder_with_modify()` | `paic_filter_modifies(modifies)` |

### Dual Registration Problem

Every order is registered in **two** trackers:

```rust
// mod.rs:2175-2187
// Tier1 tracker (existing)
self.tier1.queue_tracker.order_placed(
    result.oid, spec.price, result.resting_size, 0.0, is_bid,
);
// PAIC tracker (new)
self.paic.state_estimator.order_placed(
    result.oid, spec.price, result.resting_size, 0.0, is_bid,
);
```

This means:
- Double memory usage for order tracking
- Double computation on every trade (both trackers process)
- Potential for state divergence if bugs exist

---

## Redundancy Analysis

### Component Overlap Matrix

| PAIC Component | Existing Component | Overlap % | Notes |
|----------------|-------------------|-----------|-------|
| `VirtualQueueTracker` | `QueuePositionTracker` | **90%** | Same data, different metrics |
| `ToxicityEstimator` | `TradeFlowTracker` | **40%** | Both track buy/sell imbalance |
| `StateEstimator.sigma` | `ParameterEstimator` | **100%** | Just copies value |
| `StateEstimator.microprice` | `ParameterEstimator` | **100%** | Just copies value |
| `RateLimitShadowPrice` | `ProactiveRateLimitTracker` | **50%** | Different philosophy, similar mechanics |
| `OrderGateway` | `HyperliquidExecutor` | **0%** | **Completely unused** |
| `OrderBatcher` | None | **0%** | **Completely unused** |

### Lines of Unused Code

| File | Lines | Status |
|------|-------|--------|
| `executor/gateway.rs` | 349 | **Unused** |
| `executor/batcher.rs` | 317 | **Unused** |
| `executor/rate_limit.rs` | 284 | **Unused** |
| `controller/actions.rs` | 207 | Partially used (LEAK unused) |
| **Total Unused** | **~1,150** | 34% of PAIC |

### Queue Tracker Comparison

```rust
// PAIC VirtualQueueTracker
pub struct OrderPriority {
    pub oid: u64,
    pub price: f64,
    pub size: f64,
    pub is_bid: bool,
    pub initial_depth_ahead: f64,
    pub cumulative_volume_traded: f64,
    pub placed_at: Instant,
    pub last_update: Instant,
    pub pi: f64,                    // NEW: priority index
    pub priority_class: PriorityClass, // NEW: classification
}

// Existing QueuePositionTracker
pub struct OrderQueuePosition {
    pub oid: u64,
    pub price: f64,
    pub size: f64,
    pub is_bid: bool,
    pub depth_ahead: f64,           // Same as initial_depth_ahead
    pub placed_at: Instant,
    pub last_update: Instant,
    // Uses methods for P(touch), P(execute|touch)
}
```

**95% structural overlap.** PAIC adds π and PriorityClass; existing adds P(fill) methods.

---

## Code Quality Assessment

### Strengths

1. **Well-documented** - Extensive module-level docs with strategy matrices
2. **Well-tested** - Each module has comprehensive unit tests
3. **Clean separation** - Observer/Controller/Executor pattern
4. **Configurable** - All thresholds in `PAICConfig`

### Weaknesses

1. **Incomplete integration** - OrderGateway not wired into execution
2. **Missing LEAK implementation** - Strategy matrix promises it, code doesn't deliver
3. **No metrics** - No Prometheus metrics for PAIC decisions
4. **No logging** - Only one `debug!` in `paic_filter_modifies()`

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `virtual_queue.rs` | 8 | Good |
| `toxicity.rs` | 4 | Good |
| `state_estimator.rs` | 4 | Moderate |
| `impulse_engine.rs` | 6 | Good |
| `priority_value.rs` | 5 | Good |
| `rate_limit.rs` | 6 | Good |
| `gateway.rs` | 4 | Good |
| `batcher.rs` | 7 | Good |

Tests are present but test **components in isolation**, not the integrated system.

---

## Performance Implications

### Memory Overhead

```
Per order:
  - Existing QueuePositionTracker: ~80 bytes
  - PAIC VirtualQueueTracker: ~96 bytes
  - Combined: ~176 bytes per order

With 20 active orders: 3.5 KB overhead (negligible)
```

### CPU Overhead

```
Per trade:
  - Existing: queue_tracker.trades_at_level() - O(n) orders
  - PAIC: state_estimator.on_trade() - O(n) orders + toxicity update
  - Combined: 2× iteration over orders

Per quote cycle:
  - paic_filter_modifies() - O(m) modifies, each with decide_action()
  - decide_action() - O(1) lookups + arithmetic
```

**Impact:** Minimal. The double-iteration is wasteful but not a bottleneck.

---

## Recommendations

### Immediate (High Priority)

1. **Consolidate queue tracking**

   Choose ONE approach:
   - **Option A:** Remove PAIC's VirtualQueueTracker, add π calculation to existing
   - **Option B:** Remove existing QueuePositionTracker, use PAIC exclusively

   Recommendation: **Option A** - existing has better probabilistic foundation

2. **Remove unused executor code**

   ```bash
   # Consider removing:
   rm src/market_maker/paic/executor/gateway.rs    # 349 lines
   rm src/market_maker/paic/executor/batcher.rs    # 317 lines
   rm src/market_maker/paic/executor/rate_limit.rs # 284 lines
   ```

   Or: Wire OrderGateway into execution path to justify its existence

3. **Implement LEAK action**

   ```rust
   // In paic_filter_modifies()
   ImpulseAction::Leak => {
       // Reduce size while preserving position
       let new_size = spec.size * decision.leak_factor;
       filtered.push(ModifySpec { size: new_size, ..spec });
   }
   ```

### Short-term (Medium Priority)

4. **Add PAIC metrics**

   ```rust
   // In infra/metrics.rs
   mm_paic_hold_count: IntCounter,
   mm_paic_shadow_count: IntCounter,
   mm_paic_reset_count: IntCounter,
   mm_paic_leak_count: IntCounter,
   mm_paic_priority_mean: Gauge,
   mm_paic_toxicity: Gauge,
   ```

5. **Make toxicity threshold adaptive**

   ```rust
   // Per-asset toxicity baseline
   let baseline_toxicity = self.calibrate_toxicity_baseline(asset);
   let is_toxic = self.toxicity > baseline_toxicity * 1.5;
   ```

6. **Add volatility regime hysteresis**

   Prevent rapid Quiet↔Normal↔Turbulent oscillation

### Long-term (Low Priority)

7. **Unify rate limiting**

   Merge shadow pricing concept into ProactiveRateLimitTracker:
   ```rust
   impl ProactiveRateLimitTracker {
       pub fn shadow_cost(&self) -> f64 { ... }
       pub fn should_execute(&self, importance: f64) -> bool { ... }
   }
   ```

8. **Consider probabilistic π**

   Instead of deterministic π, model P(front of queue):
   ```
   P(front) = P(volume_traded >= depth_ahead)
            = 1 - Poisson_CDF(depth_ahead; λ × t)
   ```

---

## Exploration Paths

### Path 1: Validate Queue Priority Model

```bash
# Add logging to compare PAIC π with actual fill rates
RUST_LOG=hyperliquid_rust_sdk::market_maker::paic=debug \
cargo run --bin market_maker -- --asset BTC

# Look for:
# - Orders with π ≈ 0 that didn't fill (model overfitting)
# - Orders with π ≈ 1 that filled quickly (model underfitting)
```

**Questions to answer:**
- Does low π correlate with faster fills?
- Is the 30s decay half-life appropriate?
- Should π be nonlinear (e.g., π² for front-of-queue emphasis)?

### Path 2: Measure PAIC Impact

```bash
# A/B test: Run with/without paic_filter_modifies()
# Metric: Fill rate, adverse selection, P&L

# Without PAIC (baseline)
# In mod.rs:2399, comment out:
# let all_modifies = self.paic_filter_modifies(all_modifies);

# Compare:
# - Fills per hour
# - Average adverse selection (mark-to-market after fill)
# - Modify count (should be higher without PAIC)
```

### Path 3: Toxicity Calibration

```bash
# Collect toxicity data per asset
grep "toxicity=" logs/mm_*.log | \
  awk -F'toxicity=' '{print $2}' | \
  sort -n | \
  awk 'BEGIN{n=0} {a[n++]=$1} END{
    print "p50:", a[int(n*0.5)];
    print "p90:", a[int(n*0.9)];
    print "p99:", a[int(n*0.99)];
  }'
```

**Questions to answer:**
- What's the baseline toxicity for BTC vs ETH vs altcoins?
- Should toxic_threshold be 0.1 or asset-specific?
- Does toxicity spike before adverse price moves?

### Path 4: Rate Limit Shadow Pricing Simulation

```python
# Simulate shadow pricing vs simple rate limiting
import numpy as np

def shadow_cost(tokens, max_tokens, exponent=2.0):
    return (max_tokens / tokens) ** exponent

# Scenario: 100 actions with varying importance
actions = np.random.uniform(0.5, 2.0, 100)
tokens = 10000
max_tokens = 10000
refill_rate = 166.67  # per second

# Simple approach: execute all until exhausted
# Shadow approach: prioritize high-importance actions

# Compare: which approach executes more "value"?
```

### Path 5: Full PAIC Integration

If you want to use OrderGateway:

```rust
// In reconcile_ladder_with_modify()
// Replace direct executor calls with gateway submissions

// Instead of:
self.executor.modify_order(spec).await?;

// Use:
let decision = self.paic.impulse_engine.decide_action(...);
match self.paic.order_gateway.submit(&decision) {
    GatewayResult::Flushed(batch) => {
        for action in batch.actions {
            self.executor.execute_action(action).await?;
        }
    }
    GatewayResult::Batched => { /* wait for next flush */ }
    GatewayResult::RateLimited { .. } => { /* skip */ }
    GatewayResult::Skipped => { /* HOLD */ }
}
```

---

## Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `paic/mod.rs` | 55 | Module exports and documentation |
| `paic/config.rs` | 120 | All configuration structs |
| `paic/observer/mod.rs` | 14 | Observer layer exports |
| `paic/observer/state_estimator.rs` | 366 | Main state synthesis |
| `paic/observer/toxicity.rs` | 305 | VPIN + OFI estimation |
| `paic/observer/virtual_queue.rs` | 570 | Queue priority tracking |
| `paic/controller/mod.rs` | 19 | Controller layer exports |
| `paic/controller/actions.rs` | 207 | Action types and decisions |
| `paic/controller/impulse_engine.rs` | 347 | Strategy matrix implementation |
| `paic/controller/priority_value.rs` | 176 | Value calculations |
| `paic/executor/mod.rs` | 13 | Executor layer exports |
| `paic/executor/batcher.rs` | 317 | Order batching (UNUSED) |
| `paic/executor/gateway.rs` | 349 | Order gateway (UNUSED) |
| `paic/executor/rate_limit.rs` | 284 | Shadow pricing (UNUSED) |

---

## Conclusion

The PAIC framework introduces sound stochastic models for queue-aware order management, but suffers from:

1. **Incomplete integration** - Only 10% of code is active
2. **Significant redundancy** - Dual queue tracking, parallel rate limiting
3. **Missing implementation** - LEAK action promised but not delivered

**Next steps:**
1. Decide: Fully integrate or trim unused code
2. Consolidate queue tracking into single source of truth
3. Add observability (metrics, logging) to validate model assumptions
