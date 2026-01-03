# Market Maker Architecture Overview

**Project**: hyper_make (Hyperliquid Rust SDK with Market Maker)
**Last Updated**: 2026-01-02

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

## Stochastic Constraints (Added 2026-01-02)

The market maker includes first-principles stochastic constraints:
- **Tick size floor**: Spreads can't be finer than tick size
- **Latency floor**: σ × √(2×τ_update) accounts for quote update delay
- **Conditional tight quoting**: 5 prerequisites (vol regime, toxicity, inventory, time, book depth)
- **Book depth threshold**: Minimum depth required for tight spreads

Key parameters in `StochasticConfig`:
- `quote_update_latency_ms`: 50ms default
- `tight_quoting_max_inventory`: 0.3 (30%)
- `tight_quoting_max_toxicity`: 0.1
- `tight_quoting_excluded_hours`: [7, 14] UTC

See `session_2026-01-02_stochastic_constraints` for implementation details.

## HIP-3 Multi-DEX Support

### Auto-Prefix Feature (2026-01-02)
Asset names are automatically prefixed with DEX name when `--dex` is specified:
- `--asset BTC --dex hyna` → uses `hyna:BTC`
- Location: `src/bin/market_maker.rs:649-669`

### Core Implementation (Added 2026-01-02)

The market maker supports trading on HIP-3 builder-deployed perpetuals:

### New Types
- `PerpDex`: DEX metadata (name, full_name, deployer, oracle_updater)
- `PerpDexLimits`: Per-DEX OI caps and transfer limits
- `DexAssetMap`: Per-DEX asset index mapping

### CLI Flags
```bash
--dex <name>           # HIP-3 DEX name (e.g., "hyena", "felix")
--list-dexs            # List available DEXs and exit
--initial-isolated-margin <USD>  # Isolated margin for HIP-3 (default: 1000)
--force-isolated       # Force isolated margin mode
```

### API Patterns
```rust
// DEX-specific metadata
info_client.meta_for_dex(Some("hyena")).await?;
info_client.perp_dexs().await?;
info_client.perp_dex_limits("hyena").await?;

// WebSocket with DEX context
Subscription::AllMids { dex: Some("hyena".to_string()) }
Subscription::L2Book { coin: "BTC".to_string(), dex: Some("hyena".to_string()) }
```

### Backward Compatibility
- No `--dex` flag = validator perps (existing behavior)
- `meta()` unchanged (calls `meta_for_dex(None)`)
- Subscription variants work with `dex: None`

See `session_2026-01-02_hip3_dex_support_complete` for full implementation details.

## Important Constants

- Default gamma_base: 0.3
- Default min_spread_floor: 8 bps
- Default maker_fee_rate: 1.5 bps
- Default ladder levels: 5
- Default fees_bps (spread capture): 3.5 bps
