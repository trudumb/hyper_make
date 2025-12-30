# Market Maker Workflow Guide

This document provides a comprehensive overview of the Hyperliquid market maker's workflow, architecture, intent, and operational commands.

## Table of Contents

1. [Project Intent](#project-intent)
2. [Architecture Overview](#architecture-overview)
3. [Core Workflow](#core-workflow)
4. [Getting Started](#getting-started)
5. [Configuration](#configuration)
6. [Running the Market Maker](#running-the-market-maker)
7. [Monitoring & Observability](#monitoring--observability)
8. [Risk Management](#risk-management)
9. [Development Workflow](#development-workflow)
10. [Module Reference](#module-reference)

---

## Project Intent

### What This System Does

This is a **production-grade automated market maker** for the Hyperliquid decentralized exchange. It provides continuous two-sided liquidity by placing bid and ask orders around a fair price, earning the spread while managing inventory risk.

### Core Philosophy

1. **First-Principles Mathematics**: All decisions derive from stochastic control theory (GLFT model), not ad-hoc heuristics
2. **Data-Driven Adaptation**: Parameters (volatility, order flow, adverse selection) are estimated live from market data
3. **Defense in Depth**: Multiple layers of risk controls prevent catastrophic losses
4. **Modular Architecture**: Components are isolated, testable, and replaceable

### Key Capabilities

- **Optimal Quote Placement**: Guéant-Lehalle-Fernandez-Tapia (GLFT) model for spread and inventory skew
- **Multi-Level Ladder Quoting**: 5+ price levels per side for deeper liquidity provision
- **Live Parameter Estimation**: σ (volatility), κ (book depth), microprice from real-time data
- **Adverse Selection Measurement**: Ground truth E[Δp | fill] tracking
- **Liquidation Cascade Detection**: Hawkes process for tail risk identification
- **Kill Switch Protection**: Automatic shutdown on loss/position/staleness limits

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MarketMaker<S, E>                           │
│                     (Orchestrator / Event Loop)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Strategy   │  │   Executor   │  │   Estimator  │               │
│  │  (GLFT/      │  │ (Hyperliquid │  │ (σ, κ, μ)    │               │
│  │   Ladder)    │  │   Exchange)  │  │              │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Component Bundles                         │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  Tier1: AdverseSelection, QueueTracker, LiquidationDetector │    │
│  │  Tier2: Hawkes, Funding, Spread, PnL                        │    │
│  │  Safety: KillSwitch, RiskAggregator, FillProcessor          │    │
│  │  Infra: Margin, Prometheus, ConnectionHealth, DataQuality   │    │
│  │  Stochastic: HJBController, DynamicRisk                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Hyperliquid Exchange                             │
│  WebSocket: AllMids, L2Book, Trades, UserFills                      │
│  REST: Orders, Cancels, Account State                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Organization (~24K lines, 74 files)

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `strategy/` | Quote pricing logic | GLFTStrategy, LadderStrategy, RiskConfig |
| `estimator/` | Live parameter estimation | VolumeClock, BipowerVariation, Microprice |
| `tracking/` | Order & position state | OrderManager, PositionTracker, QueueTracker |
| `adverse_selection/` | Fill quality measurement | ASEstimator, DepthDecayAS |
| `process_models/` | Stochastic processes | Hawkes, Funding, Liquidation, Spread, HJB |
| `risk/` | Risk monitoring | KillSwitch, RiskAggregator, Monitors |
| `quoting/` | Quote generation | LadderGenerator, Optimizer, Filters |
| `infra/` | Infrastructure | Margin, Prometheus, Reconnection, DataQuality |
| `safety/` | Exchange reconciliation | SafetyAuditor |
| `core/` | Component bundles | Tier1, Tier2, Safety, Infra, Stochastic |

---

## Core Workflow

### Event Loop (Simplified)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Event Loop                           │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   AllMids     │      │    Trades     │      │  UserFills    │
│  (mid price)  │      │  (volatility) │      │   (fills)     │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Update        │      │ Update        │      │ Update        │
│ Microprice    │      │ σ, κ, regime  │      │ Position      │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Calculate Quotes    │
                    │   (Strategy + Params) │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Reconcile Ladder    │
                    │   (Cancel/Place)      │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Check Kill Switch   │
                    └───────────────────────┘
```

### Quote Calculation Flow

1. **Microprice Estimation**: Fair price from book/flow imbalance signals
   ```
   microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)
   ```

2. **GLFT Optimal Spread**: Market-driven half-spread
   ```
   δ = (1/γ) × ln(1 + γ/κ)
   ```

3. **Inventory Skew**: Position-dependent price adjustment
   ```
   skew = (q/Q_max) × γ × σ² × T
   ```

4. **Final Quotes**:
   ```
   bid = microprice × (1 - δ - skew)
   ask = microprice × (1 + δ - skew)
   ```

---

## Getting Started

### Prerequisites

- Rust 1.70+
- Hyperliquid account with API access
- Private key for signing transactions

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd hyper_make

# Build the project
cargo build --release

# Run tests to verify
cargo test
```

### Environment Setup

```bash
# Create .env file (or export directly)
echo "HYPERLIQUID_PRIVATE_KEY=your_private_key_here" > .env
```

---

## Configuration

### Generate Sample Config

```bash
cargo run --bin market_maker -- generate-config
```

This creates `market_maker.toml` with all available options:

```toml
# market_maker.toml - Hyperliquid Market Maker Configuration

[network]
base_url = "testnet"  # mainnet, testnet, localhost
# private_key = "..."  # Prefer env var HYPERLIQUID_PRIVATE_KEY

[trading]
asset = "ETH"
target_liquidity = 0.25          # Size per side
risk_aversion = 0.5              # γ: 0.1 (aggressive) to 2.0 (conservative)
max_bps_diff = 2                 # Max deviation before requoting
max_absolute_position_size = 0.5 # Position limit
leverage = 20                    # Optional: uses max if not set

[strategy]
strategy_type = "ladder"         # symmetric, inventory_aware, glft, ladder
estimation_window_secs = 300     # 5-minute rolling window
min_trades = 50                  # Warmup threshold
warmup_decay_secs = 300          # Adaptive warmup decay

[strategy.ladder_config]
num_levels = 5
min_depth_bps = 5
max_depth_bps = 50
geometric_spacing = true

[logging]
level = "info"                   # trace, debug, info, warn, error
format = "pretty"                # pretty, json, compact

[monitoring]
metrics_port = 9090
enable_http_metrics = true
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk_aversion` | 0.5 | γ - controls spread width and inventory skew |
| `target_liquidity` | 0.25 | Order size per side |
| `max_absolute_position_size` | 0.5 | Position limit (triggers reduce-only mode) |
| `max_bps_diff` | 2 | Price deviation threshold for requoting |
| `min_trades` | 50 | Warmup trades before quoting begins |

---

## Running the Market Maker

### Basic Commands

```bash
# Run with defaults (reads market_maker.toml)
cargo run --bin market_maker

# Run with specific asset
cargo run --bin market_maker -- --asset BTC

# Run with debug logging
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug cargo run --bin market_maker

# Run with debug logging and log file
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug cargo run --bin market_maker -- \
  --asset BTC --log-file mm.log

# Validate configuration without running
cargo run --bin market_maker -- validate-config

# Show all CLI options
cargo run --bin market_maker -- --help
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--config <path>` | Config file path (default: market_maker.toml) |
| `--asset <symbol>` | Override asset (e.g., BTC, ETH) |
| `--target-liquidity <f64>` | Override liquidity per side |
| `--risk-aversion <f64>` | Override gamma parameter |
| `--max-position <f64>` | Override position limit |
| `--leverage <u32>` | Override leverage |
| `--network <network>` | mainnet, testnet, localhost |
| `--log-level <level>` | trace, debug, info, warn, error |
| `--log-file <path>` | Log to file (also logs to stdout) |
| `--metrics-port <port>` | Prometheus metrics port (0 to disable) |

### Production Run Example

```bash
# Full production command with all options
RUST_LOG=hyperliquid_rust_sdk::market_maker=info \
cargo run --release --bin market_maker -- \
  --config production.toml \
  --asset ETH \
  --network mainnet \
  --log-file /var/log/market_maker.log \
  --metrics-port 9090
```

---

## Monitoring & Observability

### Prometheus Metrics

Access at `http://localhost:9090/metrics`:

```
# Position
mm_position{asset="ETH"} 0.125
mm_max_position{asset="ETH"} 0.5
mm_inventory_utilization{asset="ETH"} 0.25

# P&L
mm_daily_pnl{asset="ETH"} 15.50
mm_realized_pnl{asset="ETH"} 12.30
mm_unrealized_pnl{asset="ETH"} 3.20
mm_drawdown_pct{asset="ETH"} 0.02

# Market
mm_mid_price{asset="ETH"} 3500.50
mm_spread_bps{asset="ETH"} 8.5
mm_sigma{asset="ETH"} 0.00015
mm_kappa{asset="ETH"} 85.2
mm_jump_ratio{asset="ETH"} 1.2

# Risk
mm_kill_switch_triggered{asset="ETH"} 0
mm_cascade_severity{asset="ETH"} 0.15
mm_adverse_selection_bps{asset="ETH"} 0.8
mm_tail_risk_multiplier{asset="ETH"} 1.0
```

### Log Analysis

```bash
# Watch for fills
tail -f mm.log | grep "Fill processed"

# Watch for quote updates
tail -f mm.log | grep "Calculated ladder"

# Watch for risk events
tail -f mm.log | grep -E "(Kill switch|cascade|adverse selection)"

# JSON log parsing
cat mm.log | jq 'select(.target == "hyperliquid_rust_sdk::market_maker")'
```

### Key Log Events

| Event | Level | Meaning |
|-------|-------|---------|
| `Market maker started` | INFO | System initialized successfully |
| `Warming up parameter estimator` | INFO | Collecting initial data |
| `Parameter estimation warmed up` | INFO | Ready to quote |
| `Fill processed` | INFO | Order filled, position updated |
| `Kill switch condition detected` | WARN | Risk limit approaching |
| `KILL SWITCH TRIGGERED` | ERROR | Emergency shutdown initiated |

---

## Risk Management

### Kill Switch Conditions

The market maker automatically shuts down when:

| Condition | Default | Description |
|-----------|---------|-------------|
| Max Daily Loss | $500 | Cumulative session loss |
| Max Drawdown | 5% | Peak-to-trough P&L |
| Max Position Value | $10,000 | |position| × mid_price |
| Stale Data | 5 seconds | No market data received |
| Cascade Severity | 0.95 | Liquidation cascade detected |
| Rate Limit Errors | 10 | Exchange rate limiting |

### Graceful Shutdown

When triggered (or on Ctrl+C):

1. Stop accepting new trades
2. Cancel all resting orders with retry
3. Log final position and P&L
4. Log adverse selection summary
5. Exit cleanly

### Reduce-Only Mode

Automatically activated when:
- Position exceeds `max_position`
- Position value exceeds `max_position_value`

In reduce-only mode:
- Only orders that reduce position are placed
- Quotes on the exposed side are cancelled

---

## Development Workflow

### Build Commands

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Format code
cargo fmt

# Lint with warnings as errors
cargo clippy -- -D warnings

# Run all tests
cargo test

# Run specific test
cargo test test_glft_zero_inventory

# Run full CI pipeline
./ci.sh
```

### Code Organization

```
src/
├── bin/
│   └── market_maker.rs     # CLI entry point
├── market_maker/
│   ├── mod.rs              # MarketMaker orchestrator
│   ├── config.rs           # Configuration types
│   ├── strategy/           # Quoting strategies
│   │   ├── glft.rs         # GLFT optimal spread
│   │   ├── ladder_strat.rs # Multi-level ladder
│   │   └── risk_config.rs  # Dynamic gamma
│   ├── estimator/          # Parameter estimation
│   │   ├── volume.rs       # Volume clock
│   │   ├── volatility.rs   # Bipower variation
│   │   └── microprice.rs   # Fair price learning
│   ├── tracking/           # State management
│   │   ├── order_manager/  # Order lifecycle
│   │   ├── position.rs     # Position tracking
│   │   └── queue/          # Queue position
│   └── ...
└── lib.rs                  # Library exports
```

### Adding a New Strategy

```rust
// In strategy/my_strategy.rs
pub struct MyStrategy {
    // Custom parameters
}

impl QuotingStrategy for MyStrategy {
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        // Your quoting logic here
        // market_params contains: microprice, sigma, kappa, etc.
    }

    fn name(&self) -> &'static str {
        "MyStrategy"
    }
}
```

### Testing

```bash
# Run unit tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific module tests
cargo test market_maker::strategy::tests

# Run integration tests (requires testnet)
cargo test --test integration -- --ignored
```

---

## Module Reference

### Strategy Module (`strategy/`)

| File | Purpose |
|------|---------|
| `glft.rs` | GLFT optimal market making |
| `ladder_strat.rs` | Multi-level ladder with GLFT |
| `simple.rs` | SymmetricStrategy, InventoryAwareStrategy |
| `risk_config.rs` | RiskConfig for dynamic gamma |
| `market_params.rs` | MarketParams aggregation |

### Estimator Module (`estimator/`)

| File | Purpose |
|------|---------|
| `volume.rs` | Volume clock for time normalization |
| `volatility.rs` | Bipower variation σ estimation |
| `jump.rs` | Jump detection via RV/BV ratio |
| `kappa.rs` | Book depth decay estimation |
| `microprice.rs` | Fair price from signal regression |
| `kalman.rs` | Kalman filter price smoothing |

### Tracking Module (`tracking/`)

| File | Purpose |
|------|---------|
| `order_manager/` | Order lifecycle state machine |
| `position.rs` | Position tracking with fill dedup |
| `queue/` | Queue position and fill probability |
| `pnl.rs` | P&L tracking with attribution |

### Process Models (`process_models/`)

| File | Purpose |
|------|---------|
| `hawkes.rs` | Self-exciting order flow |
| `funding.rs` | Funding rate prediction |
| `liquidation.rs` | Cascade detection |
| `spread.rs` | Spread regime tracking |
| `hjb_control.rs` | HJB optimal inventory |

### Risk Module (`risk/`)

| File | Purpose |
|------|---------|
| `kill_switch.rs` | Emergency shutdown |
| `aggregator.rs` | Multi-monitor risk eval |
| `monitors/` | Individual risk monitors |
| `state.rs` | Unified risk state |

---

## Quick Reference Card

```bash
# === BUILD ===
cargo build                     # Dev build
cargo build --release           # Release build
cargo fmt && cargo clippy       # Format + lint

# === TEST ===
cargo test                      # All tests
cargo test strategy             # Strategy tests

# === RUN ===
# Generate config
cargo run --bin market_maker -- generate-config

# Validate config
cargo run --bin market_maker -- validate-config

# Run testnet
cargo run --bin market_maker -- --asset ETH --network testnet

# Run mainnet (production)
RUST_LOG=info cargo run --release --bin market_maker -- \
  --network mainnet --log-file mm.log

# Debug mode
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug cargo run --bin market_maker

# === MONITOR ===
# Prometheus metrics
curl localhost:9090/metrics

# Watch logs
tail -f mm.log | grep -E "(Fill|Quote|Kill)"
```

---

## Further Reading

- [CLAUDE.md](./CLAUDE.md) - Detailed architecture and implementation notes
- [Hyperliquid Docs](https://hyperliquid.gitbook.io/) - Exchange API reference
- [GLFT Paper](https://arxiv.org/abs/1105.3115) - Optimal market making theory
