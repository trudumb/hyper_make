# Session: HIP-3 Multi-DEX Support Implementation

**Date:** 2026-01-02
**Status:** Complete
**Related:** session_2026-01-02_hip3_mainnet_support

## Summary

Completed comprehensive HIP-3 multi-DEX support implementation, allowing users to trade on builder-deployed perpetuals (Hyena, Felix, etc.) by specifying `--dex` flag.

## Changes Implemented

### Phase 1: DEX Metadata Types (src/meta.rs)
- Added `PerpDex` struct: name, full_name, deployer, oracle_updater
- Added `PerpDexLimits` struct: OI caps, transfer limits

### Phase 2: InfoClient DEX Support (src/info/info_client.rs)
- Added `dex: Option<String>` to `InfoRequest` variants (Meta, MetaAndAssetCtxs, AllMids)
- New variants: `PerpDexs`, `PerpDexLimits`
- New methods: `meta_for_dex()`, `perp_dexs()`, `perp_dex_limits()`, `all_mids_for_dex()`

### Phase 3: WebSocket Subscriptions (src/ws/ws_manager.rs)
- Added `dex: Option<String>` to: AllMids, Candle, L2Book, Trades, Bbo
- Updated `get_identifier()` for subscription matching

### Phase 4: DexAssetMap (src/exchange/exchange_client.rs)
- New `DexAssetMap` struct for per-DEX asset index mapping
- Methods: `build()`, `asset_index()`, `has_coin()`, `display_name()`

### Phase 5: Config Update (src/market_maker/config.rs)
- Added `dex: Option<String>` field to `MarketMakerConfig`

### Phase 6: CLI Flags (src/bin/market_maker.rs)
- `--dex <name>`: Specify HIP-3 DEX (e.g., "hyena", "felix")
- `--list-dexs`: List available HIP-3 DEXs and exit
- Updated metadata query to use `meta_for_dex()`

### Phase 7: MarketMaker Integration (src/market_maker/mod.rs)
- WebSocket subscriptions include DEX context
- AllMids, Trades, L2Book all pass `dex: self.config.dex.clone()`

### Example Fixes
- Updated 5 examples to include `dex: None`: ws_candles, ws_l2_book, ws_trades, ws_all_mids, ws_bbo

## Usage Examples

```bash
# List available DEXs
cargo run --bin market_maker -- --list-dexs

# Trade validator BTC (default, backward compatible)
cargo run --bin market_maker -- --asset BTC

# Trade Hyena's BTC perp
cargo run --bin market_maker -- --asset BTC --dex hyena

# Trade Felix's BTC with custom margin
cargo run --bin market_maker -- --asset BTC --dex felix --initial-isolated-margin 2000
```

## API Patterns

```rust
// InfoClient DEX-specific queries
let meta = info_client.meta_for_dex(Some("hyena")).await?;
let dexs = info_client.perp_dexs().await?;
let limits = info_client.perp_dex_limits("hyena").await?;

// WebSocket with DEX
Subscription::AllMids { dex: Some("hyena".to_string()) }
Subscription::L2Book { coin: "BTC".to_string(), dex: Some("hyena".to_string()) }
```

## Test Results
- All 524 tests pass
- Build succeeds without errors

## Additional Work: WORKFLOW.md SOP Update

Transformed WORKFLOW.md into comprehensive Standard Operating Procedure (SOP):
- Pre-flight checklists (first run, each session, production)
- Complete CLI options reference including HIP-3 flags
- Dedicated HIP-3 DEX Operations section
- Troubleshooting guide with problem/solution format
- Emergency procedures for kill switch, orphaned orders
- Maintenance procedures (daily/weekly operations)
- Quick reference card for essential commands

## Files Modified

1. `src/meta.rs` - PerpDex, PerpDexLimits types
2. `src/info/info_client.rs` - DEX-aware InfoRequest and methods
3. `src/ws/ws_manager.rs` - DEX-aware Subscription enum
4. `src/exchange/exchange_client.rs` - DexAssetMap struct
5. `src/market_maker/config.rs` - dex field
6. `src/bin/market_maker.rs` - CLI flags
7. `src/market_maker/mod.rs` - WebSocket subscriptions
8. `src/lib.rs` - Export new types
9. `examples/ws_*.rs` (5 files) - Add dex: None
10. `WORKFLOW.md` - Comprehensive SOP overhaul

## Backward Compatibility

- `meta()` unchanged (calls `meta_for_dex(None)`)
- `Subscription::AllMids` without dex works as before
- No `--dex` flag = validator perps (existing behavior)
