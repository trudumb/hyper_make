# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Design Principles](#design-principles)
3. [Development Workflow](#development-workflow)
4. [Code Conventions](#code-conventions)
5. [Testing Requirements](#testing-requirements)
6. [Session Memory Workflow](#session-memory-workflow)
7. [Architecture Overview](#architecture-overview)
8. [Module Reference](#module-reference)
9. [Common Pitfalls](#common-pitfalls)
10. [Debugging Guide](#debugging-guide)

---

## Quick Reference

### Build Commands

```bash
cargo build                    # Compile the project
cargo fmt -- --check           # Format checking
cargo clippy -- -D warnings    # Lint with warnings-as-errors
cargo test                     # Run test suite
./ci.sh                        # Full CI pipeline (build, fmt, clippy, test)
```

### Run Market Maker

```bash
# Testnet (development)
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug \
cargo run --bin market_maker -- --asset BTC

# Mainnet (production)
RUST_LOG=hyperliquid_rust_sdk::market_maker=info \
cargo run --bin market_maker -- --network mainnet --asset BTC

# HIP-3 DEX
cargo run --bin market_maker -- --network mainnet --asset BTC --dex hyna

# With timestamped logs (recommended for analysis)
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S) && \
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug \
cargo run --bin market_maker -- \
  --asset BTC \
  --log-file logs/mm_testnet_BTC_${TIMESTAMP}.log
```

### Key Files for Common Tasks

| Task | Files |
|------|-------|
| Change quoting logic | `strategy/glft.rs`, `strategy/ladder_strat.rs` |
| Modify spread calculation | `quoting/ladder/depth_generator.rs` |
| Adjust risk parameters | `strategy/risk_config.rs`, `config.rs` |
| Add new estimator | `estimator/` directory |
| Change order execution | `infra/executor.rs` |
| Modify kill switch | `risk/kill_switch.rs`, `risk/monitors/` |
| Update Prometheus metrics | `infra/metrics.rs` |

### CLI Quick Reference

```bash
cargo run --bin market_maker -- --help              # All options
cargo run --bin market_maker -- generate-config     # Sample config
cargo run --bin market_maker -- --list-dexs         # Available DEXs
cargo run --bin market_maker -- --dry-run           # Validate without trading
```

---

## Design Principles

### 1. First-Principles Mathematics

**All decisions derive from stochastic control theory, not ad-hoc heuristics.**

The GLFT (Guéant-Lehalle-Fernandez-Tapia) model is the foundation:
```
δ* = (1/γ) × ln(1 + γ/κ)     # Optimal half-spread
skew = (q/Q_max) × γ × σ² × T  # Inventory skew
```

When adding features, ask: "What is the mathematical justification?"

**Examples of first-principles thinking:**
- Spread floor = σ × √(2×τ_update) + fees (latency constraint)
- Toxic hour scaling derived from empirical E[Δp|fill] analysis
- Kelly sizing from actual win rate and edge measurements

### 2. Data-Driven Adaptation

**Parameters are estimated live from market data, not hardcoded.**

| Parameter | Source | Update Frequency |
|-----------|--------|------------------|
| σ (volatility) | Bipower variation on volume-bucketed returns | Every volume tick |
| κ (kappa) | L2 book depth decay regression | Every L2 update |
| microprice | Ridge regression on book/flow imbalances | Every trade |
| γ_effective | Dynamic scaling from vol/toxicity/inventory | Every quote cycle |

**When to hardcode vs estimate:**
- Hardcode: Physical constraints (tick size, latency, fees)
- Estimate: Market state (volatility, order flow, regime)

### 3. Defense in Depth

**Multiple independent safety layers prevent catastrophic losses.**

```
Layer 1: Pre-trade checks (margin, reduce-only)
Layer 2: Position monitors (size, value limits)
Layer 3: P&L monitors (loss, drawdown limits)
Layer 4: Market monitors (cascade, staleness)
Layer 5: Kill switch (emergency shutdown)
```

Never remove a safety layer without adding an equivalent or better one.

### 4. Modular Architecture

**Components are isolated, testable, and replaceable.**

The `MarketMaker<S: QuotingStrategy, E: OrderExecutor>` pattern enables:
- Swapping strategies without changing infrastructure
- Testing with mock executors
- Isolated unit tests for each component

### 5. Conditional Complexity

**Quote tight only when conditions are safe.**

```
Can_Quote_Tight =
    (Regime == Calm) AND
    (Toxicity < 0.1) AND
    (Hour NOT IN toxic_hours) AND
    (|Inventory| < 0.3 × max_position)
```

The 8 bps floor is correct for general operation. Tighter spreads require ALL conditions met.

---

## Development Workflow

### Standard Development Cycle

```
1. Read existing code before modifying
2. Understand the mathematical model behind the component
3. Write tests FIRST (unit tests for pure functions)
4. Implement with proper error handling
5. Run full CI: cargo fmt && cargo clippy && cargo test
6. Test with actual market data (testnet or log analysis)
7. Create session memory documenting changes
```

### Before Making Changes

**Always do these steps:**

1. **Read the target file(s)** - Never propose changes blind
2. **Search for usages** - `grep -r "FunctionName"` before renaming
3. **Check tests** - `cargo test` should pass before AND after
4. **Review related components** - Changes often ripple

### Log Analysis Workflow

After running the market maker:

```bash
# Quick error check
grep -c "ERROR" logs/mm_*.log
grep -c "WARN" logs/mm_*.log

# Analyze with Claude
# Provide log file and request: sc:analyze
```

The `sc:analyze` slash command provides:
- Behavior summary
- Issues categorized by severity
- Recommended fixes with code locations

### Slash Commands Available

| Command | Purpose |
|---------|---------|
| `/mm-gaps` | Track production gap implementation progress |

### Creating Session Memories

After significant work, create a memory in `.serena/memories/`:

```markdown
# Session: {YYYY-MM-DD} {Short Description}

## Summary
{1-2 sentence summary}

## Changes Made
{List of changes with file locations}

## Files Modified
{Table of files and changes}

## Verification
{Test results, verification steps}

## Next Steps
{Future work, follow-up tasks}
```

Naming convention: `session_{YYYY-MM-DD}_{short_description}.md`

---

## Code Conventions

### Rust Style

```rust
// Use explicit types for clarity in financial calculations
let spread_bps: f64 = 8.0;
let position: f64 = tracker.position();

// Prefer early returns for guards
if !estimator.is_warmed_up() {
    return None;
}

// Use descriptive constants
const MIN_SPREAD_FLOOR_BPS: f64 = 8.0;
const MAKER_FEE_BPS: f64 = 1.5;

// Log with structured fields
debug!(
    spread_bps = %spread,
    position = %pos,
    "Calculated quotes"
);
```

### File Organization

```
src/market_maker/
├── mod.rs              # Main orchestrator (~1,500 lines)
├── config.rs           # All configuration structs
├── core/               # Component bundles (Tier1, Tier2, etc.)
├── estimator/          # Parameter estimation pipeline
├── strategy/           # Quoting strategies
├── quoting/            # Quote generation (ladder, filters)
├── risk/               # Risk management
├── tracking/           # Order and position state
├── infra/              # Infrastructure (executor, metrics)
├── process_models/     # Stochastic processes (hawkes, etc.)
├── adverse_selection/  # Fill quality measurement
├── safety/             # Exchange reconciliation
├── fills/              # Fill processing pipeline
└── messages/           # WebSocket message handlers
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Structs | PascalCase | `ParameterEstimator` |
| Traits | PascalCase + verb | `QuotingStrategy`, `OrderExecutor` |
| Functions | snake_case | `calculate_quotes()` |
| Constants | SCREAMING_SNAKE | `MIN_SPREAD_FLOOR_BPS` |
| Config fields | snake_case | `gamma_base`, `min_spread_floor` |
| Metrics | mm_snake_case | `mm_spread_bps`, `mm_position` |

### Error Handling

```rust
// Use Result for recoverable errors
fn fetch_metadata(&self) -> Result<Meta, Error> {
    // ...
}

// Use Option for optional values
fn get_fill_price(&self, oid: u64) -> Option<f64> {
    // ...
}

// Log errors with context
if let Err(e) = exchange.place_order(order).await {
    error!(error = %e, order_id = %oid, "Failed to place order");
}
```

---

## Testing Requirements

### Before Every Commit

```bash
cargo fmt       # Format code
cargo clippy -- -D warnings  # Zero warnings allowed
cargo test      # All tests must pass
```

### Test Organization

```rust
// Unit tests in the same file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_calculation() {
        let spread = calculate_spread(0.3, 100.0);
        assert!((spread - 0.008).abs() < 0.001);
    }
}
```

### What to Test

| Component | Test Focus |
|-----------|------------|
| Estimators | Warmup logic, edge cases (zero vol, extreme values) |
| Strategies | Quote generation, inventory skew, edge cases |
| Risk | Kill switch triggers, threshold boundaries |
| Tracking | State transitions, fill deduplication |

### Integration Testing

Run against testnet with small sizes:
```bash
cargo run --bin market_maker -- \
  --asset BTC \
  --target-liquidity 0.001 \
  --log-file logs/test_session.log
```

Analyze results:
```bash
grep "Fill processed" logs/test_session.log
grep "Quote cycle" logs/test_session.log | tail -20
```

---

## Session Memory Workflow

### Why Session Memories Matter

Session memories in `.serena/memories/` provide:
- Persistent context across Claude sessions
- Decision rationale preservation
- Implementation checkpoint recovery
- Debugging history for regressions

### Key Memories to Reference

| Memory | Purpose |
|--------|---------|
| `project_architecture_overview.md` | System architecture reference |
| `tight_spread_first_principles.md` | Mathematical foundations |
| `session_checkpoint_profitability_fixes.md` | Key profitability improvements |
| `design_bayesian_estimator_v2.md` | V2 estimator design |

### Creating Effective Memories

**Good memory:**
```markdown
# Session: 2026-01-04 Rate Limit Fix

## Summary
Reduced API calls by 60% through bulk cancel optimization.

## Root Cause
Individual cancel calls per order exhausted rate limit in 50 minutes.

## Solution
Implemented `cancel_bulk_orders()` in OrderExecutor trait.

## Files Modified
- `infra/executor.rs:145-180` - Added bulk cancel method
- `mod.rs:890-920` - Integration

## Metrics Impact
- Cancel API calls per requote: 10 → 1
- Rate limit buffer exhaustion: 50 min → 250 min

## Verification
✅ cargo test (572 passed)
✅ 5-minute live test, no rate limit warnings
```

**Bad memory:**
```markdown
# Fixed stuff

Made some changes to the rate limiting.
```

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MarketMaker<S, E>                           │
│                     (Orchestrator / Event Loop)                     │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Strategy   │  │   Executor   │  │   Estimator  │               │
│  │  (GLFT/      │  │ (Hyperliquid │  │ (σ, κ, μ)    │               │
│  │   Ladder)    │  │   Exchange)  │  │              │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Component Bundles                         │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │  Tier1: AdverseSelection, QueueTracker, LiquidationDetector │   │
│  │  Tier2: Hawkes, Funding, Spread, PnL                        │   │
│  │  Safety: KillSwitch, RiskAggregator, FillProcessor          │   │
│  │  Infra: Margin, Prometheus, ConnectionHealth, DataQuality   │   │
│  │  Stochastic: HJBController, DynamicRisk                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Hyperliquid Exchange                            │
│  WebSocket: AllMids, L2Book, Trades, UserFills                      │
│  REST: Orders, Cancels, Account State                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Trades → VolumeBucket → VWAP → BipowerVariation → σ, RV/BV
L2Book → WeightedKappa → κ
       → BookImbalance ──┐
Trades → FlowImbalance ──┴→ MicropriceEstimator → microprice

microprice + σ + κ → GLFTStrategy → Quote(bid, ask)
Quote → LadderGenerator → Ladder[5 levels × 2 sides]
Ladder → Reconciler → BulkOrder/BulkCancel → Exchange
```

### Event Loop

```
loop {
    select! {
        AllMids(mid) => update_microprice(mid),
        Trades(trade) => {
            update_volatility(trade);
            update_flow_imbalance(trade);
        },
        L2Book(book) => {
            update_kappa(book);
            update_book_imbalance(book);
        },
        UserFills(fill) => {
            update_position(fill);
            update_adverse_selection(fill);
        },
        timer(100ms) => {
            if should_quote() {
                calculate_ladder();
                reconcile_orders();
            }
            check_kill_switch();
        }
    }
}
```

### GLFT Strategy

```
microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)
δ = (1/γ) × ln(1 + γ/κ)                    # Optimal half-spread
skew = (q/Q_max) × γ × σ² × T              # Inventory skew
bid = microprice × (1 - δ - skew)
ask = microprice × (1 + δ - skew)
```

**Parameters:**
- γ (gamma): Risk aversion (0.1 aggressive → 1.0 conservative)
- σ (sigma): Volatility from bipower variation
- κ (kappa): Order flow intensity from book depth
- T: Holding time horizon (1/λ from trade rate)

### Ladder Quoting

```
Level 0: depth = min_depth_bps (5 bps default)
Level 1: depth = level_0 × spacing_factor
...
Level N: depth = max_depth_bps (50 bps default)

Size allocation (geometric decay):
size[i] = base_size × decay_factor^i
where decay_factor = 0.53 (default)
```

---

## Module Reference

### Core Modules (~24K lines, 74 files)

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `strategy/` | Quote pricing logic | `GLFTStrategy`, `LadderStrategy`, `RiskConfig` |
| `estimator/` | Live parameter estimation | `ParameterEstimator`, `BipowerVariation`, `Microprice` |
| `tracking/` | Order & position state | `OrderManager`, `PositionTracker`, `QueueTracker` |
| `risk/` | Risk monitoring | `KillSwitch`, `RiskAggregator`, `RiskMonitor` |
| `quoting/` | Quote generation | `LadderGenerator`, `Optimizer` |
| `infra/` | Infrastructure | `HyperliquidExecutor`, `PrometheusMetrics` |
| `process_models/` | Stochastic processes | `Hawkes`, `Funding`, `Liquidation` |
| `adverse_selection/` | Fill quality | `AdverseSelectionEstimator` |

### Estimator Pipeline

| Component | Input | Output | Update |
|-----------|-------|--------|--------|
| `VolumeClock` | Trades | Volume ticks | Per trade |
| `BipowerVariation` | VWAP returns | σ, RV/BV | Per volume tick |
| `KappaEstimator` | L2 book | κ | Per L2 update |
| `MicropriceEstimator` | Mid, imbalances | microprice, β | Per trade |
| `SoftJumpClassifier` | σ, returns | P(jump) | Per trade |
| `HierarchicalKappa` | Market κ, fills | κ_own | Per fill |

### Risk Monitors

| Monitor | Trigger | Action |
|---------|---------|--------|
| `LossMonitor` | Daily loss > $500 | Kill switch |
| `DrawdownMonitor` | Drawdown > 5% | Kill switch |
| `PositionMonitor` | Value > $10K | Kill switch |
| `CascadeMonitor` | Severity > 0.95 | Kill switch |
| `StalenessMonitor` | Data > 5s old | Quote cancel |

---

## Common Pitfalls

### 1. Spread Cap Override

**Problem:** GLFT calculates optimal 8-9 bps, but spread gets capped.

**Symptom:** Logs show `optimal_spread_bps: 8.5` but actual spread is 4-5 bps.

**Fix:** Check `market_spread_cap_multiple` in `depth_generator.rs`. Should be 0.0 (disabled).

### 2. Immediate Fill Double-Counting

**Problem:** API returns fill on place, WebSocket also delivers fill.

**Symptom:** Position jumps 2x expected on fills.

**Fix:** `FillProcessor` tracks `immediate_fill_amounts` by OID. WebSocket fills deduct from this.

### 3. Order State Confusion

**Problem:** Orders tracked as `Resting` but already filled on exchange.

**Symptom:** Orphaned orders, reconciliation warnings.

**States:**
- `Resting`: Actively on book (safe to cancel)
- `PartialFilled`: Partially filled, rest on book
- `FilledImmediately`: Full fill on place (terminal)
- `CancelPending`: Cancel in flight (don't re-cancel)

### 4. Warmup Bypass

**Problem:** Quoting starts before estimators stabilize.

**Symptom:** Wild spreads, large losses in first 30 seconds.

**Check:** `estimator.is_warmed_up()` requires:
- 20 volume ticks
- 10 L2 updates
- 50 microprice observations

### 5. Rate Limit Exhaustion

**Problem:** Individual cancel calls exhaust 10K/min limit.

**Symptom:** `429 Too Many Requests` errors after ~50 minutes.

**Fix:** Use `cancel_bulk_orders()` for ladder reconciliation.

### 6. HIP-3 Asset Confusion

**Problem:** Asset not found on HIP-3 DEX.

**Fix:** Asset names auto-prefix with DEX: `--asset BTC --dex hyna` → uses `hyna:BTC`.

### 7. Reduce-Only Mode Stuck

**Problem:** All orders cancelled, position won't reduce.

**Causes:**
1. Liquidation proximity (`buffer_ratio < 0.5`)
2. Margin utilization > 80%
3. Position > `max_position` (if set)

**Debug:**
```bash
grep "Reduce-only" logs/mm_*.log
# Check trigger reason in log
```

### 8. Time Zone Issues

**Problem:** Toxic hour scaling not activating.

**Fix:** Hours in `RiskConfig` are UTC. Check server timezone.

---

## Debugging Guide

### Checking Market Data

```bash
# Verify data is flowing
grep "AllMids" logs/mm_*.log | tail -5
grep "L2Book" logs/mm_*.log | tail -5
grep "Trades processed" logs/mm_*.log | tail -5
```

### Checking Estimator State

```bash
# Parameter estimation
grep "sigma=" logs/mm_*.log | tail -5
grep "kappa=" logs/mm_*.log | tail -5
grep "microprice=" logs/mm_*.log | tail -5

# Warmup status
grep -i "warm" logs/mm_*.log
```

### Checking Quote Generation

```bash
# Ladder calculation
grep "Calculated ladder" logs/mm_*.log | tail -10

# Spread diagnostics
grep "spread_bps" logs/mm_*.log | tail -10
```

### Checking Risk State

```bash
# Kill switch status
grep -i "kill" logs/mm_*.log

# Reduce-only triggers
grep -i "reduce" logs/mm_*.log

# Cascade detection
grep -i "cascade" logs/mm_*.log
```

### Prometheus Metrics

```bash
# Check metrics endpoint
curl -s localhost:9090/metrics | grep mm_

# Key metrics to monitor
curl -s localhost:9090/metrics | grep -E "(mm_position|mm_spread_bps|mm_daily_pnl)"
```

### Common Log Patterns

| Pattern | Meaning |
|---------|---------|
| `Quote cycle completed` | Normal operation |
| `Warming up` | Estimator collecting data |
| `Reduce-only mode activated` | Risk limit hit |
| `Kill switch condition` | Emergency threshold approaching |
| `KILL SWITCH TRIGGERED` | Emergency shutdown |
| `Cascade detected` | Liquidation cascade |
| `Data staleness exceeded` | WebSocket may be disconnected |

---

## Hyperliquid Constraints

- **Minimum order notional:** $10 USD
- **Price precision:** 5 significant figures, max `6 - sz_decimals` decimals
- **Size precision:** Truncate to `sz_decimals` (from asset metadata)
- **Rate limits:** ~10,000 requests/minute (IP-based)
- **WebSocket:** Max 100 subscriptions per connection

---

## Further Reading

- [WORKFLOW.md](./WORKFLOW.md) - Operational procedures and SOP
- [Hyperliquid Docs](https://hyperliquid.gitbook.io/) - Exchange API reference
- [GLFT Paper](https://arxiv.org/abs/1105.3115) - Optimal market making theory
- `.serena/memories/` - Session memories and design documents

---

## Empirical Verification

**"Real validation through actual tests with data."**

When exploring complex behaviors, API interactions, or verifying fixes, do not rely solely on unit tests or theoretical code analysis. Create dedicated diagnostic binaries to validate hypotheses against the live environment (Testnet).

**Workflow:**
1. **Identify the Hypothesis**: Defined clearly (e.g., "Are orders actually Post-Only?").
2. **Create Diagnostic Binary**: Create a focused binary in `src/bin/` (e.g., `src/bin/verify_alo.rs`) that:
   - Connects to the exchange (Testnet).
   - Performs specific read/write operations.
   - Outputs raw data confirming or disproving the hypothesis.
3. **Execute**: Run against the live network.
4. **Analyze**: Check if we "see what we are suspecting".
5. **Troubleshoot**: If results differ from expectations, investigate based on the *actual data*.
