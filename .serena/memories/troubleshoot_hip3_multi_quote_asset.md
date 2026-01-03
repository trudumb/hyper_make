# Troubleshooting: HIP-3 Multi-Quote Asset Support
**Date**: 2026-01-02
**Status**: Implementation Complete ✓

## Problem Statement
HIP-3 DEXs can use stablecoins other than USDC as margin/collateral. Current codebase assumes USDC.

## API Discovery

### perpDexs → meta → collateralToken
```bash
# Get DEX list
curl -s https://api.hyperliquid.xyz/info -d '{"type": "perpDexs"}'

# Get collateral token for specific DEX
curl -s https://api.hyperliquid.xyz/info -d '{"type": "meta", "dex": "hyna"}'
# Returns: {"universe": [...], "collateralToken": 235}
```

### Collateral Token Mapping
| DEX | Token Index | Token Name |
|-----|-------------|------------|
| xyz | 0 | USDC |
| flx | 360 | USDH |
| vntl | 360 | USDH |
| hyna | 235 | USDE |

### Token Resolution
```bash
curl -s https://api.hyperliquid.xyz/info -d '{"type": "spotMetaAndAssetCtxs"}'
# Returns tokens array with index → name mapping
```

## Codebase USDC Hardcoding Locations

### API Response Fields (Cannot Change)
- `src/types/ledger.rs`: `usdc` field in AccountDelta, etc.
- `src/types/user.rs`: `usdc` field in SpotUserState

### Hardcoded Constants (Need Attention)
- `src/types/trades.rs:126,147`: Test fixtures use `"USDC"`
- `src/exchange/exchange_client.rs:532`: `"USDC"` string
- `src/exchange/transfers.rs`: `usdc_transfer()` method

### Comments/Documentation
- `src/market_maker/infra/rate_limit.rs:261`: Mentions USDC in rate limit comment

## What Works Without Changes
- P&L tracking (uses position × price)
- Margin calculations (uses ratios)
- Strategy logic (quote-agnostic)
- `fee_token` already parsed from API

## Required Changes

### Phase 1: Metadata (Must Have)
1. Add `collateral_token: Option<u32>` to `Meta` struct in `src/meta.rs`
2. Add token info lookup function
3. Store in MarketMakerConfig at startup

### Phase 2: Runtime (Should Have)
1. Validate fee_token matches expected collateral
2. Add quote asset to Prometheus metrics labels
3. Log quote asset in startup banner

### Phase 3: Transfers (Nice to Have)
1. Support spot_transfer for HIP-3 collateral
2. Query collateral token balance

## Key Type Changes

### src/meta.rs - Meta struct
```rust
#[derive(Deserialize, Debug, Clone)]
pub struct Meta {
    pub universe: Vec<AssetMeta>,
    #[serde(default)]
    pub collateral_token: Option<u32>,  // NEW: HIP-3 collateral index
}
```

### New: CollateralInfo
```rust
pub struct CollateralInfo {
    pub token_index: u32,
    pub name: String,
    pub sz_decimals: u8,
    pub wei_decimals: u8,
}
```

## Implementation Summary

### Files Modified
1. **src/meta.rs**
   - Added `collateral_token: Option<u32>` to `Meta` struct
   - Created `CollateralInfo` struct with methods:
     - `usdc()` - default USDC collateral
     - `from_token_index()` - resolve token from spot metadata
     - `balance_from_spot()` - extract balance from spot balances
     - `available_balance_from_spot()` - get withdrawable balance
     - `transfer_token()` - token symbol for transfers
     - `format_amount()` - format amount with token precision

2. **src/market_maker/config.rs**
   - Added `collateral: CollateralInfo` field to `MarketMakerConfig`

3. **src/market_maker/messages/context.rs**
   - Added `expected_collateral: Arc<str>` to `MessageContext`

4. **src/market_maker/messages/user_fills.rs**
   - Added fee_token validation warning

5. **src/market_maker/infra/metrics.rs**
   - Updated `to_prometheus_text()` to include quote asset label

6. **src/bin/market_maker.rs**
   - Added collateral resolution at startup
   - Updated startup banner to display quote asset
   - Added quote asset to Prometheus endpoint

7. **src/lib.rs**
   - Exported `CollateralInfo`

### Tests Added
- 19 meta tests pass including collateral balance extraction

## Cross-Reference
- `session_2026-01-02_hip3_dex_support_complete`: Full HIP-3 support
- `session_2026-01-02_hip3_auto_prefix`: Asset name auto-prefix
