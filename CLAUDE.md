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

Run tools with:
```bash
cargo run --bin market_maker -- --help  # Market maker tool with CLI
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

**`market_maker.rs`** - Market making utilities and order state tracking

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
