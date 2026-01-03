# Session: HIP-3 Asset Index Fix

**Date:** 2026-01-03
**Status:** Complete
**Issue:** HIP-3 DEX orders rejected with "Insufficient margin to place order. asset=0"

## Problem Summary

Orders placed on HIP-3 DEXs (e.g., `hyna:BTC`) were being rejected because the wrong asset index was being sent to the API.

### Root Cause

The `ExchangeClient` was using raw indices from the DEX meta (0, 1, 2...) instead of the correct HIP-3 formula:

```
asset_index = 100000 + (perp_dex_index × 10000) + index_in_meta
```

**Example:**
- `hyna:BTC` with `perp_dex_index=1` and `index_in_meta=0`
- **Wrong:** `asset=0` (sent to validator BTC perp)
- **Correct:** `asset=110000` (sent to hyna DEX BTC)

### Evidence from Logs

```
"Bulk order 0 rejected: asset=hyna:BTC ... error=Insufficient margin to place order. asset=0"
```

The `asset=0` in the API response indicated orders were being routed to validator perps instead of the HIP-3 DEX.

## Solution Implemented

### 1. New Constructor: `ExchangeClient::new_for_dex()`

Location: `src/exchange/exchange_client.rs:230-295`

```rust
pub async fn new_for_dex(
    client: Option<Client>,
    wallet: PrivateKeySigner,
    base_url: Option<BaseUrl>,
    meta: Option<Meta>,
    vault_address: Option<Address>,
    dex: Option<&str>,  // NEW: DEX name for HIP-3
) -> Result<ExchangeClient>
```

Key logic:
```rust
if let Some(dex_name) = dex {
    // Look up DEX index from perp_dexs() API
    let perp_dexs = info.perp_dexs().await?;
    let dex_index = perp_dexs
        .iter()
        .position(|d| d.as_ref().map(|d| d.name.as_str()) == Some(dex_name))
        .ok_or(Error::AssetNotFound)?;

    // Apply HIP-3 formula
    let base_offset = 100000 + (dex_index as u32 * 10000);
    for (asset_ind, asset) in meta.universe.iter().enumerate() {
        let full_index = base_offset + asset_ind as u32;
        coin_to_asset.insert(asset.name.clone(), full_index);
    }
}
```

### 2. Updated Market Maker Binary

Location: `src/bin/market_maker.rs:842-851`

```rust
let exchange_client = ExchangeClient::new_for_dex(
    None,
    wallet.clone(),
    Some(base_url),
    Some(meta.clone()),
    None,
    dex.as_deref(),  // Pass DEX name for HIP-3 support
)
```

### 3. Backward Compatibility

The existing `new()` method now delegates to `new_for_dex(None)`, preserving all existing behavior for validator perps.

## Asset Index Ranges (Reference)

| Asset Type | Formula | Example |
|------------|---------|---------|
| Validator perps | `index` | BTC = 0 |
| Spot assets | `10000 + index` | PURR/USDC = 10000 |
| HIP-3 DEXs | `100000 + (dex_idx × 10000) + index` | hyna:BTC = 110000 |

## Testing

```bash
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug cargo run --bin market_maker -- \
  --network mainnet \
  --asset BTC \
  --dex hyna \
  --log-file logs/mm_hyna_BTC_test.log
```

Verify logs show correct asset index (e.g., 110000 for hyna:BTC).

## Files Modified

1. `src/exchange/exchange_client.rs` - Added `new_for_dex()` with HIP-3 formula
2. `src/bin/market_maker.rs` - Updated to use `new_for_dex()`

## Related Documentation

- [Hyperliquid Asset IDs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/asset-ids)
- Session memory: `session_2026-01-02_hip3_dex_support_complete`
