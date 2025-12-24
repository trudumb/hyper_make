# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build                    # Compile the project
cargo fmt -- --check           # Format checking
cargo clippy -- -D warnings    # Lint with warnings-as-errors
cargo test                     # Run test suite
./ci.sh                        # Run full CI pipeline (build, fmt, clippy, test)
```

Run examples with:
```bash
cargo run --example <example_name>  # e.g., cargo run --example order_and_cancel
```

Run market maker:
```bash
cargo run --bin market_maker -- --asset BTC              # Run with defaults
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug cargo run --bin market_maker -- --asset BTC --log-file mm.log   #runwithdebug
cargo run --bin market_maker -- --help                   # Show CLI options
cargo run --bin market_maker -- generate-config          # Generate sample config
```

## Architecture

This is a Rust SDK for the Hyperliquid DEX API. It provides trading, market data, and WebSocket streaming capabilities.

### Core Modules

**`exchange/`** - Trading operations client
- `ExchangeClient`: Main trading interface (wallet, HTTP, metadata, coin-to-asset mapping)
- `actions.rs`: Trade actions (UsdSend, UpdateLeverage, BulkOrder, BulkCancel, ApproveAgent, etc.)
- `order.rs`, `cancel.rs`, `modify.rs`: Order/cancel/modify request types
- Uses builder pattern for complex orders

**`info/`** - Market data queries
- `InfoClient`: Queries market metadata, user state, orders, fees, funding history

**`ws/`** - WebSocket real-time data
- `WsManager`: Manages connections, subscriptions, message routing
- `Subscription` enum: AllMids, L2Book, Trades, OrderUpdates, UserEvents, Candle, etc.

**`signature/`** - EIP-712 cryptographic signing
- `sign_l1_action()`, `sign_typed_data()`: Transaction signing for Ethereum compatibility

**`market_maker/`** - Modular market making system
- `mod.rs`: `MarketMaker<S, E>` orchestrator - coordinates strategy, orders, position, execution
- `config.rs`: `MarketMakerConfig`, `QuoteConfig`, `Quote`, `MarketMakerMetricsRecorder` trait
- `strategy.rs`: `QuotingStrategy` trait + `SymmetricStrategy` + `InventoryAwareStrategy` + `GLFTStrategy`
- `estimator.rs`: `ParameterEstimator` - econometric pipeline with volume clock, bipower variation, regime detection, microprice estimation
- `order_manager.rs`: `OrderManager`, `TrackedOrder`, `Side` - tracks resting orders by oid
- `position.rs`: `PositionTracker` - position state with fill deduplication by tid
- `executor.rs`: `OrderExecutor` trait + `HyperliquidExecutor` - abstracts order placement/cancellation

### Supporting Files

- `consts.rs`: API URLs (mainnet/testnet/localhost), constants (EPSILON, INF_BPS)
- `errors.rs`: Error enum (client/server/parsing/validation errors)
- `helpers.rs`: Utility functions (nonce, float formatting, BaseUrl enum)
- `meta.rs`: Asset metadata types (Meta, AssetMeta, AssetContext)
- `req.rs`: HTTP client wrapper with error parsing

### Examples

33 API examples in `examples/` (run with `cargo run --example <name>`):
- **Exchange**: `order_and_cancel`, `leverage`, `market_order_and_cancel`, `order_and_cancel_cloid`
- **Spot**: `spot_order`, `spot_transfer`
- **Transfers**: `usdc_transfer`, `vault_transfer`, `class_transfer`, `bridge_withdraw`
- **Account**: `agent`, `approve_builder_fee`, `claim_rewards`, `set_referrer`
- **WebSocket**: `ws_all_mids`, `ws_l2_book`, `ws_trades`, `ws_orders`, `ws_candles`, `ws_bbo`, etc.
- **Advanced**: `info` (comprehensive API reference), `using_big_blocks`

### Tools

Production tools in `src/bin/` (run with `cargo run --bin <name>`):
- **market_maker**: Automated market making tool with CLI configuration, multiple strategies, and structured logging

### Key Patterns

- **Wallet auth**: Uses `PrivateKeySigner` from Alloy for signing
- **Async-first**: Built on tokio with async WebSocket and HTTP
- **MessagePack**: Uses `rmp-serde` for binary protocol efficiency
- **Builder pattern**: Complex orders built through builder structs

### Market Maker Architecture

The market maker uses a modular, trait-based design for extensibility:

```
MarketMaker<S: QuotingStrategy, E: OrderExecutor>
    ├── QuotingStrategy (trait)     # Pluggable pricing logic
    │   ├── SymmetricStrategy       # Equal spread both sides
    │   ├── InventoryAwareStrategy  # Position-based skew
    │   └── GLFTStrategy            # Optimal MM (Guéant-Lehalle-Fernandez-Tapia)
    ├── OrderExecutor (trait)       # Abstracted execution (testable)
    │   └── HyperliquidExecutor     # Real exchange client
    ├── ParameterEstimator          # Live σ, κ, τ estimation from market data
    ├── OrderManager                # Tracks resting orders
    └── PositionTracker             # Fill dedup + position state
```

**GLFT Strategy (default):**

Uses stochastic control theory for optimal market making with data-driven fair price:
- **Microprice**: Fair price learned from book/flow imbalance signals (replaces ad-hoc adjustments)
- **σ (sigma)**: Jump-robust volatility from bipower variation (√BV)
- **κ (kappa)**: Order flow intensity from fill-rate estimation
- **τ (tau)**: Time horizon estimated from trade rate (faster markets → smaller τ)
- **γ (gamma)**: Dynamic risk aversion via `RiskConfig`, scales with volatility/toxicity/inventory

Formulas:
```
microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)  # Data-driven fair price
δ = (1/γ) × ln(1 + γ/κ)                    # Optimal half-spread (GLFT)
skew = (q/Q_max) × γ × σ² × T              # Inventory skew (T = 1/λ holding time)
bid = microprice × (1 - δ - skew)          # Quote around fair price
ask = microprice × (1 + δ - skew)
```

**RiskConfig (Dynamic Gamma):**

`RiskConfig` controls how γ adapts to market conditions:
- `gamma_base`: Base risk aversion (0.1 aggressive → 1.0 conservative)
- `sigma_baseline`: Reference volatility for scaling (default 0.0002 = 2bp/sec)
- `volatility_weight`: How much high vol increases γ (0.0-1.0)
- `max_volatility_multiplier`: Cap on vol-driven γ increase
- `toxicity_threshold`: Jump ratio above which γ scales (default 1.5)
- `toxicity_sensitivity`: γ increase per unit of jump_ratio
- `inventory_threshold`: Utilization % before inventory scaling (default 0.5)
- `inventory_sensitivity`: Quadratic scaling near position limits
- `gamma_min`, `gamma_max`: Bounds for effective gamma
- `min_spread_floor`: Minimum spread (default 1bp)
- `max_holding_time`: Cap on T to prevent skew explosion (default 120s)

**Parameter Estimation (Econometric Pipeline):**

The `ParameterEstimator` provides HFT-grade market parameter estimation:

1. **Volume Clock**: Adaptive volume-based sampling (1% of 5-min rolling volume)
   - Normalizes by economic activity instead of wall-clock time
   - Bucket threshold adapts to market conditions

2. **VWAP Pre-Averaging**: Returns calculated on bucket VWAPs
   - Filters bid-ask bounce noise from raw trade prices

3. **Bipower Variation**: Jump-robust volatility estimation
   - RV (Realized Variance): EWMA of r²
   - BV (Bipower Variation): EWMA of (π/2) × |r_t| × |r_{t-1}|
   - σ = √BV (clean diffusion component)

4. **Regime Detection**: Jump ratio (RV/BV) identifies toxic flow
   - Normal: ratio ≈ 1.0
   - Toxic: ratio > 3.0 (jumps dominate)

5. **Weighted Kappa**: L2 book depth decay estimation
   - Truncates orders > 1% from mid
   - Proximity-weighted regression on first 15 levels
   - κ = negative slope of ln(cumulative_depth) vs distance

Data flow:
```
Trades(px, sz, time) → VolumeBucket → VWAP → BipowerVariation → σ, RV/BV
L2Book(bids, asks)   → WeightedKappa → κ
                     → BookImbalance ──┐
Trades(side, sz)     → FlowImbalance ──┴→ MicropriceEstimator → microprice, β_book, β_flow
```

Warmup: 20 volume ticks + 10 L2 updates + 50 microprice observations before quoting begins.

6. **Microprice Estimation**: Data-driven fair price from signal prediction
   - Rolling online 2-variable regression (book_imbalance, flow_imbalance → returns)
   - Window: 60 seconds, forward horizon: 300ms
   - Learns β_book, β_flow coefficients (~0.001-0.01 typical)
   - microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)
   - Replaces hardcoded adjustments with adaptive, market-derived coefficients

**Adding a new strategy:**
```rust
pub struct MyStrategy { /* params */ }

impl QuotingStrategy for MyStrategy {
    fn calculate_quotes(&self, config: &QuoteConfig, position: f64,
                        max_position: f64, target_liquidity: f64,
                        market_params: &MarketParams)
        -> (Option<Quote>, Option<Quote>) {
        // MarketParams contains:
        //   microprice: f64        - Data-driven fair price (quote around this)
        //   sigma: f64             - √BV (jump-robust volatility)
        //   sigma_effective: f64   - σ blended with jump component
        //   kappa: f64             - order flow intensity
        //   arrival_intensity: f64 - volume ticks per second
        //   is_toxic_regime: bool  - RV/BV > threshold
        //   jump_ratio: f64        - RV/BV ratio
        //   beta_book: f64         - Learned book imbalance coefficient
        //   beta_flow: f64         - Learned flow imbalance coefficient
        // Return (bid, ask) quotes
    }
    fn name(&self) -> &'static str { "MyStrategy" }
}
```

**Hyperliquid constraints:**
- Minimum order notional: $10 USD
- Price precision: 5 significant figures, max `6 - sz_decimals` decimals (perps)
- Size precision: truncate to `sz_decimals` (from asset metadata)
