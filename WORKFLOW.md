# Workflow Guide

Unified binary for paper and live market making on Hyperliquid perpetual futures.

## Architecture

A single `market_maker` binary serves both paper and live trading via the `TradingEnvironment` trait:

```
market_maker
  ├── paper         → PaperEnvironment (simulated fills, no real orders)
  ├── run           → LiveEnvironment  (auto-calibrates if needed)
  ├── generate-config
  ├── validate-config
  └── status
```

The same `MarketMaker` core, strategy, and risk logic runs in both modes. Only the execution boundary differs — paper synthesizes fills from market trades via `FillSimulator`, live sends real orders via `HyperliquidExecutor`.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `market_maker generate-config` | Output sample TOML config |
| `market_maker validate-config` | Validate config without running |
| `market_maker paper --duration 60` | Paper trade for 60 seconds |
| `market_maker run` | Live market making (auto-calibrates if no prior) |
| `market_maker run --dry-run` | Connect and validate, no orders |
| `market_maker run --force` | Live with cold-start (skip calibration) |
| `market_maker status` | Show account position and balances |

## Auto-Calibration Pipeline

By default, `market_maker run` checks for a calibrated prior before going live:

```
market_maker run --asset HYPE --dex hyena
        │
        ▼
┌─────────────────┐   calibrated + fresh   ┌────────────┐
│  Load & Assess  │ ─────────────────────▶ │    Live    │
│  Paper Prior    │                         │  Trading   │
└────────┬────────┘                         └────────────┘
         │ missing / stale / insufficient
         ▼
┌─────────────────┐   duration elapsed     ┌─────────────┐  passes  ┌────────────┐
│  Auto Paper     │ ──────────────────▶    │  Assess &   │ ───────▶ │    Live    │
│  (30 min)       │                        │  Stamp      │          │  Trading   │
└─────────────────┘                        └──────┬──────┘          └────────────┘
                                                  │ fails
                                                  ▼
                                           Error + diagnostic
                                           (or --force override)
```

### Calibration verdicts

Each saved checkpoint is stamped with a `PriorReadiness` assessment:

| Verdict | Criteria | Behavior |
|---------|----------|----------|
| **Ready** | Session >= 30 min AND all 5 estimators converged AND >= 20 kelly fills | Full confidence live trading |
| **Marginal** | Session >= 30 min AND >= 3/5 estimators converged | Live with defensive spreads (wider, smaller size) |
| **Insufficient** | Otherwise | Blocked from live (auto-paper or error) |

The 5 core estimators: volatility (>= 200 obs), kappa (>= 50 obs), adverse selection (>= 100 samples), regime HMM (>= 200 obs), fill rate (>= 200 obs).

### Calibration flags

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-calibration` | false | Run live with whatever priors exist (no gate) |
| `--force` | false | Cold-start live with no prior (implies skip) |
| `--calibration-duration <SECS>` | 1800 | Auto-paper duration in seconds |
| `--no-auto-paper` | false | Refuse to start if no calibrated prior exists |

### Prior staleness

Priors older than 4 hours are considered stale and trigger auto-paper. The prior path is auto-resolved from `data/checkpoints/paper/{ASSET}/prior.json`, or set explicitly via `--paper-checkpoint`.

## Paper Trading

Paper mode subscribes to real market data but simulates fills locally. No orders are placed on the exchange.

### Basic usage

```bash
# Build
cargo build --bin market_maker

# 60-second paper session on HYPE via hyena DEX
./target/debug/market_maker paper --asset HYPE --dex hyena --duration 60

# Indefinite paper session (Ctrl+C to stop)
./target/debug/market_maker paper --asset BTC --duration 0

# With verbose logging
RUST_LOG=debug ./target/debug/market_maker paper --asset HYPE --dex hyena --duration 300
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--asset <ASSET>` | (required via config) | Asset to trade (BTC, HYPE, etc.) |
| `--dex <NAME>` | — | HIP-3 DEX name (hyena, felix, etc.) |
| `--duration <SECS>` | 0 | Session length; 0 = indefinite |
| `--spread-profile <P>` | `default` | `default` (40-50 bps), `hip3` (15-25 bps), `aggressive` (10-20 bps) |

### Checkpoints

Paper sessions save checkpoints to `data/checkpoints/paper/{ASSET}/`:

```bash
ls data/checkpoints/paper/HYPE/
# prior.json          — Q-table, kappa estimates, confidence, readiness stamp
```

Each saved checkpoint includes a `PriorReadiness` stamp with the calibration verdict and estimator observation counts. These stamps allow the `run` command to make instant go/no-go decisions.

### Script wrapper

The `scripts/paper_trading.sh` script adds logging, dashboard, and reporting on top:

```bash
./scripts/paper_trading.sh BTC 300              # 5-min session
./scripts/paper_trading.sh BTC 3600 --report    # 1-hour with calibration report
./scripts/paper_trading.sh BTC 300 --dashboard  # With live dashboard at localhost:3000
./scripts/paper_trading.sh BTC 300 --capture    # Dashboard + screenshot capture
```

## Prior Transfer (Paper to Live)

Prior transfer is now automatic. When you run `market_maker run`, it:

1. Checks `data/checkpoints/paper/{ASSET}/prior.json` for an existing prior
2. Assesses calibration readiness from the stamp
3. If Ready/Marginal: injects the prior and starts live
4. If Missing/Stale/Insufficient: auto-papers for 30 min (configurable), then re-assesses

### Manual prior transfer (still supported)

```bash
# 1. Run paper session to build priors
./target/debug/market_maker paper --asset HYPE --dex hyena --duration 3600

# 2. Verify checkpoint exists
ls data/checkpoints/paper/HYPE/prior.json

# 3. Live run with explicit paper checkpoint
./target/release/market_maker run --paper-checkpoint data/checkpoints/paper/HYPE
```

The `--paper-checkpoint` flag seeds the Q-table with paper-learned state-action values. Only cold states (visit count = 0) are seeded; live-learned states are preserved.

## Live Trading

### Dry run (recommended first step)

Connects to the exchange, validates config, subscribes to data feeds, but places no orders:

```bash
./target/release/market_maker run --dry-run
```

Confirms: config valid, WS connection OK, market data flowing, account state accessible.

### Full run

```bash
# Release build for production
cargo build --release

# Default: auto-calibrates if no prior, then goes live
./target/release/market_maker run --asset HYPE --dex hyena

# Force cold-start (skip calibration gate)
./target/release/market_maker run --asset HYPE --dex hyena --force

# Custom calibration duration (10 min instead of 30)
./target/release/market_maker run --asset HYPE --dex hyena --calibration-duration 600

# Refuse to auto-paper (error if no calibrated prior)
./target/release/market_maker run --asset HYPE --dex hyena --no-auto-paper
```

### Key safety flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | false | No orders placed |
| `--max-position <N>` | config | Hard position cap (contracts) |
| `--max-position-usd <N>` | config | Hard position cap (USD) |
| `--max-drawdown <F>` | 0.05 | Kill switch at 5% drawdown |
| `--max-daily-loss <USD>` | config | Daily loss kill switch |
| `--leverage <N>` | config | Leverage multiplier |

## Troubleshooting

### WSL2 linker OOM

`cargo test` may fail linking examples on WSL2 with limited memory. Use:

```bash
cargo test --lib    # Skip example binaries
```

### Missing checkpoints

Without `--force` or `--skip-calibration`, a missing prior triggers auto-paper calibration. With `--force`, the MM starts with default priors (cold start).

### WS disconnection

The connection supervisor auto-reconnects with exponential backoff. If data goes stale for >15 seconds, the quote gate pauses quoting until fresh data arrives.

### HIP-3 API quota

HIP-3 tokens may have low API rate limit headroom (~7%). The quote gate widens spreads via `1/sqrt(headroom)` when headroom drops, and switches to inventory-forcing mode below 10%. Check `--list-dexs` for available DEXs and their constraints.
