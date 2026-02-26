# Project Index: hyperliquid_rust_sdk (Market Maker)

Generated: 2026-01-10

---

## Project Overview

**Type**: Rust SDK + Professional Market Making System
**Platform**: Hyperliquid DEX (Perpetuals + HIP-3)
**Mathematical Foundation**: GLFT (Guéant-Lehalle-Fernandez-Tapia) Stochastic Control
**Codebase Size**: ~175 Rust files, ~57K lines, 693 tests

---

## Project Structure

```
src/
├── lib.rs                    # Library root, public exports
├── bin/
│   └── market_maker.rs       # Main CLI entry point
├── exchange/                 # Exchange API (orders, cancels, accounts)
├── info/                     # Info client (metadata, positions)
├── types/                    # Shared types (book, orders, trades)
├── ws/                       # WebSocket subscriptions
└── market_maker/            # Core market making system (125 files)
    ├── mod.rs               # Orchestrator (5,296 lines)
    ├── config.rs            # Configuration types
    ├── adaptive/            # Adaptive learning (fill controller, gamma)
    ├── adverse_selection/   # Fill quality measurement
    ├── estimator/           # Parameter estimation (σ, κ, microprice)
    ├── fills/               # Fill processing pipeline
    ├── infra/               # Infrastructure (executor, metrics, margin)
    ├── messages/            # WebSocket message handlers
    ├── multi/               # Multi-asset support
    ├── process_models/      # Stochastic processes (Hawkes, HJB)
    ├── quoting/             # Quote/ladder generation
    ├── risk/                # Risk management & kill switch
    ├── safety/              # Exchange reconciliation
    ├── strategy/            # Quoting strategies (GLFT, Ladder)
    └── tracking/            # Order/position state management
```

---

## Entry Points

| Entry | Path | Purpose |
|-------|------|---------|
| **CLI Binary** | `src/bin/market_maker.rs` | Production market maker |
| **Debug WS** | `src/bin/debug_ws.rs` | WebSocket diagnostics |
| **Balance Check** | `src/bin/check_balance.rs` | Account balance query |
| **ALO Check** | `src/bin/check_alo.rs` | Post-only order verification |
| **Entropy Test** | `src/bin/test_entropy.rs` | Entropy optimizer tests |
| **Reconcile Test** | `src/bin/test_reconcile.rs` | Order reconciliation tests |

---

## Core Modules

### Strategy (`strategy/`)
- **`QuotingStrategy`** trait: Interface for all strategies
- **`GLFTStrategy`**: GLFT optimal market making
- **`LadderStrategy`**: Multi-level ladder quoting
- **`InventoryAwareStrategy`**: Inventory-skewed quotes
- **`SymmetricStrategy`**: Simple symmetric quotes

### Estimator (`estimator/`)
- **`ParameterEstimator`**: Orchestrates all estimation
- **`volatility`**: Bipower variation σ estimation
- **`kappa`/`book_kappa`/`hierarchical_kappa`**: κ estimation
- **`microprice`**: Microprice regression
- **`momentum`**: Trend detection
- **`soft_jump`**: Jump probability classification

### Risk (`risk/`)
- **`KillSwitch`**: Emergency shutdown system
- **`RiskAggregator`**: Combines all risk monitors
- **Monitors**: Loss, Drawdown, Position, Cascade, DataStaleness

### Quoting (`quoting/`)
- **`LadderGenerator`**: Multi-level quote generation
- **`EntropyOptimizer`**: Entropy-based size allocation
- **`filter`**: Quote size filtering

### Infra (`infra/`)
- **`HyperliquidExecutor`**: Exchange order execution
- **`PrometheusMetrics`**: Metrics collection
- **`MarginAwareSizer`**: Position sizing with margin
- **`RecoveryManager`**: State recovery
- **`RateLimiter`**: API rate limiting

### Tracking (`tracking/`)
- **`OrderManager`**: Active order state
- **`PositionTracker`**: Position and P&L
- **`QueuePositionTracker`**: Queue priority tracking
- **`WsOrderStateManager`**: WebSocket-based order state

### Process Models (`process_models/`)
- **`HJBController`**: Hamilton-Jacobi-Bellman controller
- **`HawkesOrderFlowEstimator`**: Hawkes process intensity
- **`FundingRateEstimator`**: Funding rate tracking
- **`LiquidationCascadeDetector`**: Cascade detection

---

## Configuration

| File | Purpose |
|------|---------|
| `Cargo.toml` | Project dependencies |
| `market_maker.toml` | Default MM configuration |
| `rust-toolchain.toml` | Rust version pinning |

### Key Config Structs
- `MarketMakerConfig`: Main configuration
- `RiskConfig`: Risk parameters (gamma, max position)
- `LadderConfig`: Ladder quoting settings
- `EstimatorConfig`: Parameter estimation settings
- `KillSwitchConfig`: Kill switch thresholds

---

## Key Commands

```bash
# Build & Test
cargo build                    # Compile
cargo test                     # Run 693 tests
./ci.sh                        # Full CI (build, fmt, clippy, test)

# Run Market Maker
cargo run --bin market_maker -- --asset BTC                    # Testnet
cargo run --bin market_maker -- --network mainnet --asset BTC  # Mainnet
cargo run --bin market_maker -- --asset BTC --dex hyna         # HIP-3
cargo run --bin market_maker -- --help                         # All options

# Diagnostics
cargo run --bin check_balance
cargo run --bin debug_ws
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `alloy` | Ethereum primitives, signing |
| `tokio` | Async runtime |
| `tokio-tungstenite` | WebSocket client |
| `reqwest` | HTTP client |
| `serde`/`serde_json` | Serialization |
| `tracing` | Structured logging |
| `axum` | Prometheus HTTP server |
| `clap` | CLI argument parsing |
| `rand`/`rand_distr` | RNG for allocation |

---

## Quick Reference

### Mathematical Foundation (GLFT)
```
δ* = (1/γ) × ln(1 + γ/κ)      # Optimal half-spread
skew = (q/Q_max) × γ × σ² × T   # Inventory skew
```

### Design Principles
1. **First-Principles Math**: All decisions from stochastic control
2. **Data-Driven**: Parameters estimated live, not hardcoded
3. **Defense in Depth**: 5 independent safety layers
4. **Modular**: Pluggable strategies and executors

### Key Files for Common Tasks
| Task | Files |
|------|-------|
| Change quoting | `strategy/glft.rs`, `strategy/ladder_strat.rs` |
| Modify spread | `quoting/ladder/depth_generator.rs` |
| Adjust risk | `strategy/risk_config.rs`, `config.rs` |
| Kill switch | `risk/kill_switch.rs`, `risk/monitors/` |
| Execution | `infra/executor.rs` |

---

## Session Memories

Key memories in `.serena/memories/`:
- `project_architecture_overview.md` - System architecture
- `tight_spread_first_principles.md` - Mathematical foundations
- `design_bayesian_estimator_v2.md` - V2 estimator design
- `session_checkpoint_profitability_fixes.md` - Key fixes

---

## Test Coverage

- **Unit Tests**: 693 passed
- **Integration Tests**: Via testnet binaries
- **Test Files**: Located within modules (`#[cfg(test)]`)

---

*This index was auto-generated. See CLAUDE.md for detailed development guidelines.*
