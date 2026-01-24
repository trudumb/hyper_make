# Infrastructure Hardening Plan
## CI/CD, Orchestrator Tests, Health Endpoints, Unwrap Reduction

**Created:** 2026-01-24
**Scope:** Address critical infrastructure gaps from analysis report (excluding Dockerfile)

---

## Overview

This plan addresses 4 critical gaps:
1. **No CI/CD Pipeline** - Add GitHub Actions
2. **Orchestrator Module Has 0 Tests** - Add 25+ integration tests
3. **No Health Endpoints** - Add `/healthz` and `/readyz`
4. **Critical unwrap() Calls** - Fix 3 highest-risk locations

---

## Phase 1: GitHub Actions CI/CD (1 day)

### Files to Create

**`.github/workflows/ci.yml`**
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2

      - name: Format check
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --all-targets -- -D warnings

      - name: Build
        run: cargo build --release

      - name: Test
        run: cargo test --lib
```

### Verification
```bash
# Test locally before push
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib
```

---

## Phase 2: Health Endpoints (0.5 day)

### File to Modify

**`src/bin/market_maker.rs`** - Add health routes to existing Axum app

### Implementation

Add alongside existing `/metrics` endpoint (~line 1620):

```rust
// Health check state
struct HealthState {
    metrics: Arc<RwLock<PrometheusMetrics>>,
    last_quote_time: Arc<AtomicU64>,
}

// Liveness probe - process is alive
async fn healthz() -> StatusCode {
    StatusCode::OK
}

// Readiness probe - ready to serve traffic
async fn readyz(State(state): State<Arc<HealthState>>) -> StatusCode {
    let last_quote_ms = state.last_quote_time.load(Ordering::Relaxed);
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // Ready if quoted within last 30 seconds
    if now_ms - last_quote_ms < 30_000 {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

// Update router creation
let app = Router::new()
    .route("/metrics", get(metrics_handler))
    .route("/api/dashboard", get(dashboard_handler))
    .route("/healthz", get(healthz))           // ADD
    .route("/readyz", get(readyz))             // ADD
    .layer(CorsLayer::permissive());
```

### Verification
```bash
# Start market maker, then:
curl http://localhost:9090/healthz   # Should return 200
curl http://localhost:9090/readyz    # Should return 200 or 503
```

---

## Phase 3: Fix Critical unwrap() Calls (0.5 day)

### Priority 1: `src/market_maker/orchestrator/reconcile.rs:510,512`

**Problem:** NaN prices cause panic during order sort
**Risk:** CRITICAL - Hot path, crashes trading

```rust
// BEFORE (lines 510-512)
if side == Side::Buy {
    sorted_current.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
} else {
    sorted_current.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
}

// AFTER - Handle NaN safely
if side == Side::Buy {
    sorted_current.sort_by(|a, b| {
        b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal)
    });
} else {
    sorted_current.sort_by(|a, b| {
        a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal)
    });
}
```

### Priority 2: Add NaN validation upstream

In `reconcile.rs`, add validation before sorting:

```rust
// Filter out any orders with NaN prices (should never happen, but defensive)
sorted_current.retain(|order| {
    if order.price.is_nan() {
        tracing::error!(oid = ?order.oid, "Order has NaN price, removing from reconciliation");
        false
    } else {
        true
    }
});
```

### Verification
```bash
cargo test --lib reconcile
cargo clippy -- -D warnings
```

---

## Phase 4: Orchestrator Integration Tests (2 days)

### File to Create

**`src/market_maker/orchestrator/tests.rs`** (~400 lines)

### Test Categories

#### 4.1 Quote Engine Tests (8 tests)
```rust
#[cfg(test)]
mod quote_engine_tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_pause_trading_blocks_quotes() { }

    #[test]
    fn test_circuit_breaker_cancel_all_cancels_existing() { }

    #[test]
    fn test_circuit_breaker_widen_spreads_multiplier() { }

    #[test]
    fn test_drawdown_pause_blocks_new_quotes() { }

    #[test]
    fn test_warmup_incomplete_skips_quoting() { }

    #[test]
    fn test_risk_limit_exceeded_reduces_size() { }

    #[test]
    fn test_oi_cap_blocks_new_orders() { }

    #[test]
    fn test_rate_limit_throttle_skips_cycle() { }
}
```

#### 4.2 Reconciliation Tests (6 tests)
```rust
#[cfg(test)]
mod reconcile_tests {
    #[test]
    fn test_reconcile_cancel_stale_orders() { }

    #[test]
    fn test_reconcile_place_new_orders() { }

    #[test]
    fn test_reconcile_modify_price_update() { }

    #[test]
    fn test_min_notional_filters_small_orders() { }

    #[test]
    fn test_ladder_action_ordering() { }

    #[test]
    fn test_nan_price_handling() { }  // Validates fix from Phase 3
}
```

#### 4.3 Order Operations Tests (5 tests)
```rust
#[cfg(test)]
mod order_ops_tests {
    #[test]
    fn test_bulk_cancel_updates_state() { }

    #[test]
    fn test_cancel_retry_on_failure() { }

    #[test]
    fn test_place_order_notional_validation() { }

    #[test]
    fn test_order_state_transition_active_to_cancelled() { }

    #[test]
    fn test_bayesian_learning_on_cancel() { }
}
```

#### 4.4 Safety Sync Tests (4 tests)
```rust
#[cfg(test)]
mod safety_sync_tests {
    #[test]
    fn test_orphan_detection_with_grace_period() { }

    #[test]
    fn test_stale_pending_cleanup() { }

    #[test]
    fn test_snapshot_freshness_validation() { }

    #[test]
    fn test_position_drift_detection() { }
}
```

#### 4.5 Shutdown Tests (2 tests)
```rust
#[cfg(test)]
mod shutdown_tests {
    #[test]
    fn test_graceful_shutdown_cancels_orders() { }

    #[test]
    fn test_shutdown_logs_final_state() { }
}
```

### File to Modify

**`src/market_maker/orchestrator/mod.rs`** - Add test module

```rust
#[cfg(test)]
mod tests;
```

### Verification
```bash
cargo test --lib orchestrator::tests -- --nocapture
cargo test --lib 2>&1 | tail -5  # Check total count increases by ~25
```

---

## Implementation Order

| Day | Phase | Deliverable | Tests Added |
|-----|-------|-------------|-------------|
| 1 AM | Phase 1 | GitHub Actions CI | 0 |
| 1 PM | Phase 2 | Health endpoints | 0 |
| 2 AM | Phase 3 | Fix unwrap() calls | 1 |
| 2-3 | Phase 4 | Orchestrator tests | 25 |

---

## Success Criteria

1. **CI/CD:** `cargo fmt`, `clippy`, `test` all pass in GitHub Actions
2. **Health:** `/healthz` returns 200, `/readyz` returns 200/503 appropriately
3. **Unwrap:** `reconcile.rs` handles NaN prices without panic
4. **Tests:** Orchestrator module has 25+ tests, all passing

---

## Files Summary

| Action | File |
|--------|------|
| CREATE | `.github/workflows/ci.yml` |
| MODIFY | `src/bin/market_maker.rs` (health endpoints) |
| MODIFY | `src/market_maker/orchestrator/reconcile.rs` (NaN fix) |
| CREATE | `src/market_maker/orchestrator/tests.rs` |
| MODIFY | `src/market_maker/orchestrator/mod.rs` (add tests module) |

---

## Verification Commands

```bash
# After all changes:
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib 2>&1 | tail -5
curl http://localhost:9090/healthz
curl http://localhost:9090/readyz
```
