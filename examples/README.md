# Hyperliquid SDK Examples

This directory contains examples demonstrating how to use the Hyperliquid Rust SDK.

## Running Examples

```bash
cargo run --example <example_name>
```

## Categories

### Exchange Operations

| Example | Description |
|---------|-------------|
| `order_and_cancel` | Place a limit order and cancel it |
| `order_and_cancel_cloid` | Place order with client order ID (UUID) |
| `order_and_schedule_cancel` | Place order with scheduled cancel time |
| `order_with_builder_and_cancel` | Place order with builder fee |
| `market_order_and_cancel` | Place market order and close position |
| `market_order_with_builder_and_cancel` | Market order with builder fee |
| `leverage` | Update leverage and isolated margin settings |

### Spot Trading

| Example | Description |
|---------|-------------|
| `spot_order` | Place a spot trading order |

### Transfers

| Example | Description |
|---------|-------------|
| `usdc_transfer` | Transfer USDC to another address |
| `spot_transfer` | Transfer spot tokens to another address |
| `vault_transfer` | Deposit/withdraw from a vault |
| `class_transfer` | Transfer between spot and perp classes |
| `bridge_withdraw` | Withdraw funds via bridge |

### Account Management

| Example | Description |
|---------|-------------|
| `agent` | Create an agent wallet (restricted trading) |
| `approve_builder_fee` | Approve a builder fee structure |
| `claim_rewards` | Claim available rewards |
| `set_referrer` | Set a referral code |

### Info Queries

| Example | Description |
|---------|-------------|
| `info` | Comprehensive API reference (20+ query types) |

### WebSocket Subscriptions

All WebSocket examples subscribe for 30 seconds then unsubscribe.

| Example | Description |
|---------|-------------|
| `ws_all_mids` | Subscribe to all mid prices |
| `ws_l2_book` | Subscribe to L2 order book |
| `ws_trades` | Subscribe to trades |
| `ws_orders` | Subscribe to order updates |
| `ws_candles` | Subscribe to candlestick data |
| `ws_bbo` | Subscribe to best bid/offer |
| `ws_spot_price` | Subscribe to spot prices |
| `ws_user_events` | Subscribe to user events |
| `ws_user_fundings` | Subscribe to funding payments |
| `ws_user_non_funding_ledger_updates` | Subscribe to non-funding ledger updates |
| `ws_notification` | Subscribe to notifications |
| `ws_active_asset_ctx` | Subscribe to active asset context |
| `ws_active_asset_data` | Subscribe to active asset data |
| `ws_web_data2` | Subscribe to web data |

### Advanced

| Example | Description |
|---------|-------------|
| `using_big_blocks` | Enable/disable big blocks feature |

## Note

All examples use a test private key for demonstration purposes. **Never use these keys with real funds.**
