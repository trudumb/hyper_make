# Market Maker Architecture Overview

**Project**: hyper_make (Hyperliquid Rust SDK with Market Maker)
**Last Updated**: 2026-01-01

## Core Components

### Strategy Layer (`src/market_maker/strategy/`)

- **glft.rs**: GLFT optimal market making strategy
  - Formula: δ = (1/γ) × ln(1 + γ/κ) + maker_fee
  - Dynamic gamma scaling (vol, toxicity, inventory, regime)
  - Asymmetric bid/ask kappa support
  
- **ladder_strat.rs**: Multi-level ladder quoting
  - Uses GLFT for optimal depths
  - Constrained optimizer for size allocation
  - Kelly-Stochastic allocation option
  
- **risk_config.rs**: Dynamic risk parameters
  - gamma_base, min_spread_floor, maker_fee_rate
  - Time-of-day scaling for toxic hours
  - Flow sensitivity for skew adjustment

### Quoting Layer (`src/market_maker/quoting/ladder/`)

- **depth_generator.rs**: GLFT-optimal depth computation
  - DynamicDepthConfig with floors/caps
  - Geometric or linear spacing
  - Market spread cap option
  
- **generator.rs**: Ladder generation logic
- **optimizer.rs**: Constrained variational optimizer

### Estimator Pipeline (`src/market_maker/estimator/`)

- **parameter_estimator.rs**: Main orchestrator
- **volatility.rs**: Bipower variation, regime detection
- **kappa.rs**: Order flow intensity estimation
- **microprice.rs**: Fair price from signals
- **momentum.rs**: Trade flow, falling knife detection

### Risk Management (`src/market_maker/risk/`)

- **kill_switch.rs**: Emergency shutdown
- **aggregator.rs**: Unified risk evaluation
- **monitors/**: Loss, Drawdown, Position, Cascade, DataStaleness

## Key Configuration Files

| File | Purpose |
|------|---------|
| `config.rs` | MarketMakerConfig, QuoteConfig, StochasticConfig |
| `risk_config.rs` | RiskConfig with spread floors and gamma settings |
| `ladder/mod.rs` | LadderConfig with fees and level settings |

## CLI Parameters

```bash
cargo run --bin market_maker -- \
  --asset BTC \
  --target-liquidity 0.25 \
  --risk-aversion 0.5 \    # gamma_base
  --max-bps-diff 2 \
  --max-position 0.5
```

## Order State Management (`src/market_maker/tracking/`)

### Order States
- `Resting`: Order resting on book (active)
- `PartialFilled`: Partially filled, rest still resting (active)
- `FilledImmediately`: Full immediate fill from API (terminal)
- `CancelPending`: Cancel in flight
- `CancelConfirmed`: Cancel acknowledged
- `FilledDuringCancel`: Filled while cancel pending
- `Filled`: Fully filled via WS
- `Cancelled`: Successfully cancelled

### Key Functions
- `is_active()`: Returns true for `Resting | PartialFilled` only
- `is_terminal()`: Includes `FilledImmediately`, `Filled`, `Cancelled`, etc.

### Immediate Fill Deduplication
```rust
// FillProcessor tracks amount per OID, not just presence
immediate_fill_amounts: HashMap<u64, f64>

// When WS fill arrives:
let skip = remaining.min(fill.size);
let update = fill.size - skip;
position.process_fill(update, is_buy);
```

## Prometheus Metrics (port 9090)

- `mm_spread_bps`: Current spread
- `mm_position`: Current position
- `mm_daily_pnl`: Session P&L
- `mm_adverse_selection_bps`: Realized AS
- `mm_sigma`: Volatility estimate
- `mm_kappa`: Order flow intensity

## Important Constants

- Default gamma_base: 0.3
- Default min_spread_floor: 8 bps
- Default maker_fee_rate: 1.5 bps
- Default ladder levels: 5
- Default fees_bps (spread capture): 3.5 bps
