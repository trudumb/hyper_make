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

**`market_maker/`** - Modular market making system (~24K lines, 74 files)

**Core:**
- `mod.rs`: `MarketMaker<S, E>` orchestrator (1,500 lines)
- `config.rs`: `MarketMakerConfig`, `QuoteConfig`, `Quote`
- `core/`: Component bundles (Tier1, Tier2, Safety, Infra, Stochastic)

**Estimator Pipeline** (`estimator/` - 10 files):
- `parameter_estimator.rs`: Main orchestrator with `MarketEstimator` trait
- `volatility.rs`: Bipower variation, regime detection, stochastic vol
- `kappa.rs`: Bayesian order flow intensity estimation
- `microprice.rs`: Ridge regression fair price from signals
- `volume.rs`: Volume clock, bucket accumulation
- `momentum.rs`: Trade flow tracking, falling knife detection
- `kalman.rs`: Noise filtering
- `jump.rs`: Poisson jump process estimation

**Strategies** (`strategy/` - 6 files):
- `mod.rs`: `QuotingStrategy` trait definition
- `glft.rs`: GLFT optimal control strategy
- `ladder_strat.rs`: Multi-level ladder quoting
- `simple.rs`: `SymmetricStrategy`, `InventoryAwareStrategy`
- `risk_config.rs`: Dynamic gamma configuration
- `market_params.rs`, `params.rs`: Parameter aggregation

**Quoting** (`quoting/`):
- `ladder/`: Pure computation module
  - `generator.rs`: Depth spacing, fill intensity, spread capture
  - `optimizer.rs`: Constrained variational allocation
  - `mod.rs`: `Ladder`, `LadderLevel`, `LadderConfig` types
- `filter.rs`: Reduce-only mode enforcement

**Risk Management** (`risk/`):
- `kill_switch.rs`: Emergency shutdown triggers
- `aggregator.rs`: Unified `RiskAggregator` evaluation
- `state.rs`: `RiskState` snapshot
- `monitor.rs`: `RiskMonitor` trait
- `monitors/`: Loss, Drawdown, Position, Cascade, DataStaleness, RateLimit

**Process Models** (`process_models/`):
- `hawkes.rs`: Self-exciting order flow intensity
- `liquidation.rs`: Cascade detection and tail risk
- `funding.rs`: Funding rate estimation
- `spread.rs`: Bid-ask spread dynamics
- `hjb_control.rs`: Hamilton-Jacobi-Bellman inventory control

**Infrastructure** (`infra/`):
- `executor.rs`: `OrderExecutor` trait + `HyperliquidExecutor`
- `margin.rs`: `MarginAwareSizer` for pre-flight checks
- `metrics.rs`: Prometheus metrics (846 lines)
- `reconnection.rs`: Connection health monitoring
- `data_quality.rs`: Market data validation

**Tracking** (`tracking/`):
- `order_manager.rs`: `OrderManager`, `TrackedOrder`, `OrderState`
- `position.rs`: `PositionTracker` with fill deduplication
- `pnl.rs`: P&L attribution and tracking
- `queue.rs`: Queue position estimation

**Adverse Selection** (`adverse_selection/` - 3 files):
- `estimator.rs`: E[Δp|fill] measurement, EWMA tracking
- `depth_decay.rs`: AS(δ) = AS₀ × exp(-δ/δ_char) model
- `mod.rs`: `AdverseSelectionConfig`

**Fill Processing** (`fills/`):
- `processor.rs`: `FillProcessor` orchestration
- `consumer.rs`: `FillConsumer` trait for extensibility
- `dedup.rs`: Centralized deduplication by trade ID

**Message Handlers** (`messages/`):
- Focused handlers: `all_mids.rs`, `trades.rs`, `l2_book.rs`, `user_fills.rs`
- `context.rs`: Shared `MessageContext`
- `processors.rs`: Processing utilities

**Safety** (`safety/`):
- `auditor.rs`: `SafetyAuditor` for state reconciliation

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

The market maker uses a modular, trait-based design with component bundles:

```
MarketMaker<S: QuotingStrategy, E: OrderExecutor>
    │
    ├── Core Fields
    │   ├── config: MarketMakerConfig
    │   ├── strategy: S                    # Pluggable pricing logic
    │   ├── executor: E                    # Abstracted execution
    │   ├── estimator: ParameterEstimator  # Live σ, κ, τ estimation
    │   ├── orders: OrderManager           # Tracks resting orders
    │   └── position: PositionTracker      # Fill dedup + position state
    │
    ├── tier1: Tier1Components             # Production resilience
    │   ├── adverse_selection              # E[Δp|fill] measurement
    │   ├── depth_decay_as                 # Depth-dependent AS model
    │   ├── queue_tracker                  # Queue position estimation
    │   └── liquidation_detector           # Cascade detection
    │
    ├── tier2: Tier2Components             # Process models
    │   ├── hawkes                         # Self-exciting order flow
    │   ├── funding                        # Funding rate tracking
    │   ├── spread_tracker                 # Spread dynamics
    │   └── pnl_tracker                    # P&L attribution
    │
    ├── safety: SafetyComponents           # Risk management
    │   ├── kill_switch                    # Emergency shutdown
    │   ├── risk_aggregator                # Unified risk evaluation
    │   └── fill_processor                 # Centralized fill handling
    │
    ├── infra: InfraComponents             # Infrastructure
    │   ├── margin_sizer                   # Pre-flight margin checks
    │   ├── prometheus                     # Metrics endpoint
    │   ├── connection_health              # WebSocket health
    │   └── data_quality                   # Data validation
    │
    └── stochastic: StochasticComponents   # Optimal control
        ├── hjb_controller                 # HJB inventory control
        ├── stochastic_config              # Stochastic parameters
        └── dynamic_risk_config            # Dynamic γ scaling
```

**Strategies:**
```
QuotingStrategy (trait)
    ├── SymmetricStrategy       # Equal spread both sides
    ├── InventoryAwareStrategy  # Position-based skew
    ├── GLFTStrategy            # Optimal MM (Guéant-Lehalle-Fernandez-Tapia)
    └── LadderStrategy          # Multi-level ladder quoting

OrderExecutor (trait)
    └── HyperliquidExecutor     # Real exchange client
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

### Tier 1 Features (Production-Critical)

The market maker includes advanced features beyond basic quoting:

**Adverse Selection Measurement** (`adverse_selection/`):
- Ground truth measurement: E[Δp | fill] at 1-second horizon
- Tracks realized adverse selection separately for buy/sell fills (EWMA)
- Calculates predicted α (probability of informed flow)
- Provides spread adjustment recommendations
- Fill deduplication by trade ID prevents double-counting

```
Key metrics:
- realized_as_bps: Actual adverse selection experienced (0.5-2 bps typical)
- spread_adjustment: Recommended spread widening based on AS
- alpha: Probability of trading against informed flow
```

**Liquidation Cascade Detection** (`process_models/liquidation.rs`):
- Size-weighted Hawkes process models self-exciting liquidation events
- Cascade severity scoring (0.0 = calm, 1.0+ = extreme)
- Quote-pulling circuit breaker when severity exceeds threshold
- Tail risk multiplier scales γ from 1.0 to 5.0 during cascades
- Tracks cascade direction (long/short/both liquidations)

```
Configuration:
- cascade_pull_threshold: Severity level to pull all quotes (default 0.8)
- tail_risk_multiplier: γ scaling factor during cascades (1.0-5.0)
- cascade_decay: Half-life of cascade intensity decay
```

**Queue Position Tracking** (`tracking/queue.rs`):
- Estimates depth-ahead for resting orders
- Fill probability calculation: P(fill) = P(touch) × P(execute|touch)
- Queue decay modeling (20% per-second default)
- Refresh decision logic (cancel/replace vs hold)
- Order age tracking for stale order detection

```
Key outputs:
- queue_position: Estimated contracts ahead of our order
- fill_probability: Likelihood of execution before cancellation
- should_refresh: Boolean signal for order refresh
```

### Ladder Quoting System

Multi-level ladder quoting for deeper liquidity provision:

**Structure:**
- Default 5 levels per side (configurable via `ladder_levels`)
- Geometric size decay across levels (e.g., 0.022 → 0.01165 → 0.00557 → 0.00256 → 0.00116)
- Price spacing increases with distance from mid
- Inventory-aware sizing reduces exposed side

**Bulk Order Placement:**
- All ladder levels placed in single API call (`BulkOrder`)
- Bulk cancellation for efficient order management
- Level-by-level margin constraint checking
- Reconciliation detects changed levels, only updates necessary orders

**Ladder Reconciliation Logic:**
```
1. Calculate new ladder from strategy
2. Compare against current resting orders (by price/size)
3. Cancel orders at levels that changed
4. Place new orders at updated levels
5. Track all order IDs for fill association
```

**Size Allocation Formula:**
```
size[i] = base_size × decay_factor^i
where:
  base_size = target_liquidity / sum(decay_factor^i for i in 0..levels)
  decay_factor = 0.53 (default, yields ~50% size reduction per level)
```

### Kill Switch System

Production safety mechanisms to prevent catastrophic losses:

**Position Value Limit:**
- Maximum USD value of position (default $10,000)
- Triggers graceful shutdown when exceeded
- Formula: `position_value = |position| × mid_price`

**Drawdown Limits:**
- Maximum intraday drawdown percentage (default 5%)
- Tracks peak P&L and current P&L
- Formula: `drawdown_pct = (peak_pnl - current_pnl) / account_value`

**Data Staleness:**
- Maximum age of market data before quoting stops (default 5 seconds)
- Separate thresholds for L2 book, trades, and user events
- Stale data triggers quote cancellation, not full shutdown

**Cascade Severity:**
- When liquidation cascade severity exceeds threshold, quotes pulled
- Does not trigger full shutdown, resumes when cascade subsides

**Graceful Shutdown Protocol:**
```
1. Kill switch condition detected
2. Log ERROR with reason and metrics
3. Cancel all resting orders (bulk cancel)
4. Confirm all cancellations received
5. Log final state (position, P&L, adverse selection summary)
6. Exit process
```

### Prometheus Metrics

Full observability via Prometheus metrics endpoint:

**Position Metrics:**
- `mm_position`: Current position in contracts
- `mm_max_position`: Maximum allowed position
- `mm_inventory_utilization`: position / max_position ratio

**P&L Metrics:**
- `mm_daily_pnl`: Realized + unrealized P&L for session
- `mm_peak_pnl`: Highest P&L reached during session
- `mm_drawdown_pct`: Current drawdown from peak
- `mm_realized_pnl`: Sum of closed trade P&L
- `mm_unrealized_pnl`: Mark-to-market of open position

**Order Metrics:**
- `mm_orders_placed`: Total orders placed
- `mm_orders_filled`: Total orders filled
- `mm_orders_cancelled`: Total orders cancelled
- `mm_fill_volume_buy`: Total buy volume filled
- `mm_fill_volume_sell`: Total sell volume filled

**Market Metrics:**
- `mm_mid_price`: Current mid price
- `mm_spread_bps`: Current bid-ask spread in basis points
- `mm_sigma`: Current volatility estimate (σ)
- `mm_jump_ratio`: Current RV/BV ratio
- `mm_kappa`: Current order flow intensity (κ)

**Estimator Metrics:**
- `mm_microprice_deviation_bps`: Microprice vs mid deviation
- `mm_book_imbalance`: Current L2 book imbalance (-1 to 1)
- `mm_flow_imbalance`: Current trade flow imbalance (-1 to 1)
- `mm_beta_book`: Learned book imbalance coefficient
- `mm_beta_flow`: Learned flow imbalance coefficient

**Risk Metrics:**
- `mm_kill_switch_triggered`: Boolean (0/1) for kill switch status
- `mm_cascade_severity`: Current liquidation cascade severity
- `mm_adverse_selection_bps`: Running adverse selection estimate
- `mm_tail_risk_multiplier`: Current gamma scaling from tail risk

**Infrastructure Metrics:**
- `mm_data_staleness_secs`: Age of most recent market data
- `mm_quote_cycle_latency_ms`: Time to complete quote cycle
- `mm_volatility_regime`: Current regime (0=Normal, 1=High, 2=Extreme)

### Extended RiskConfig Parameters

Additional parameters beyond basic gamma scaling:

**Cascade/Tail Risk:**
- `cascade_pull_threshold`: Cascade severity to pull all quotes (default 0.8)
- `tail_risk_base`: Base tail risk multiplier (default 1.0)
- `tail_risk_sensitivity`: How fast multiplier increases with cascade severity
- `max_tail_risk_multiplier`: Cap on tail risk gamma scaling (default 5.0)

**Hawkes Order Flow:**
- `hawkes_decay`: Decay rate for self-exciting intensity (default 0.1)
- `hawkes_activity_weight`: Weight of Hawkes activity in gamma scaling
- `hawkes_baseline`: Baseline intensity for Hawkes process

**Funding Rate:**
- `funding_rate_weight`: Impact of funding on inventory target
- `funding_rate_horizon`: Lookahead for funding cost estimation (default 1 hour)

**Spread Regime:**
- `spread_regime_tight_mult`: Gamma multiplier in tight spread regime (default 0.8)
- `spread_regime_wide_mult`: Gamma multiplier in wide spread regime (default 1.5)
- `spread_percentile_window`: Rolling window for spread percentile (default 300s)

### Production Infrastructure

**Margin-Aware Sizer** (`infra/margin.rs`):
- Pre-flight margin checks before order placement
- Reduces order size if margin insufficient
- Respects leverage constraints from account settings
- Prevents order rejections from margin failures

**Connection Health Monitoring:**
- Tracks last update time for each data feed
- Detects WebSocket disconnections via staleness
- Triggers reconnection or quote cancellation
- Logged every 60 seconds in safety sync

**Safety State Sync:**
- Periodic reconciliation with exchange state (every 60s)
- Detects orphaned orders (on exchange but not tracked locally)
- Cancels stale orders (tracked locally but gone from exchange)
- Logs sync status and any discrepancies

**Reduce-Only Mode:**
- Activated when position exceeds max_position
- Cancels all orders that would increase position
- Only allows orders that reduce exposure
- Logged as WARN when triggered
