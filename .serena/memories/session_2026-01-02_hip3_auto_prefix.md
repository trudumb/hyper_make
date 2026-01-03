# Session: HIP-3 Auto-Prefix Asset Names
**Date**: 2026-01-02
**Status**: Complete

## Summary
Implemented auto-prefix logic for HIP-3 DEX asset names to improve user experience.

## Problem Solved
When using HIP-3 DEXs, asset names in the API are prefixed with the DEX name:
- Validator perps: `"BTC"`, `"ETH"`, `"SOL"`
- HIP-3 DEX (hyna): `"hyna:BTC"`, `"hyna:ETH"`, `"hyna:SOL"`

**Before**: Users had to manually prefix asset names:
```bash
cargo run --bin market_maker -- --asset hyna:BTC --dex hyna
```

**After**: Auto-prefix handles it automatically:
```bash
cargo run --bin market_maker -- --asset BTC --dex hyna
# Automatically uses "hyna:BTC"
```

## Implementation Details

### Location
`src/bin/market_maker.rs:649-669`

### Code Added
```rust
// Auto-prefix asset with DEX name for HIP-3 DEXs
// API returns prefixed names like "hyna:BTC" for HIP-3 assets
let asset = if let Some(ref dex_name) = dex {
    if asset.contains(':') {
        // Already prefixed (e.g., "hyna:BTC"), use as-is
        asset
    } else {
        // Auto-prefix: "BTC" + "hyna" → "hyna:BTC"
        let prefixed = format!("{}:{}", dex_name, asset);
        info!(
            original = %asset,
            prefixed = %prefixed,
            dex = %dex_name,
            "Auto-prefixed asset name for HIP-3 DEX"
        );
        prefixed
    }
} else {
    // No DEX specified, use plain asset name (validator perps)
    asset
};
```

### Behavior Matrix
| `--asset` | `--dex` | Result | Explanation |
|-----------|---------|--------|-------------|
| `BTC` | (none) | `"BTC"` | Validator perps, no prefix |
| `BTC` | `hyna` | `"hyna:BTC"` | Auto-prefixed |
| `hyna:BTC` | `hyna` | `"hyna:BTC"` | Already prefixed, no change |
| `ETH` | `flx` | `"flx:ETH"` | Auto-prefixed |

## Key Discoveries

### HIP-3 DEX Naming
- Mainnet DEX is `hyna` (not `hyena`)
- Testnet and mainnet have completely different DEXs
- API endpoint: `/info` with `{"type": "perpDexs"}` lists available DEXs

### Asset Name Format
- Validator perps: Plain names (`BTC`, `ETH`)
- HIP-3 DEXs: Prefixed names (`hyna:BTC`, `hyna:ETH`)
- Colon (`:`) separates DEX name from asset symbol

## Testing
All three scenarios verified:
1. ✅ `--asset BTC --dex hyna` → logs "Auto-prefixed" and uses `hyna:BTC`
2. ✅ `--asset hyna:BTC --dex hyna` → no log, uses `hyna:BTC`
3. ✅ `--asset BTC` (no dex) → uses plain `BTC`

## Related Sessions
- `session_2026-01-02_hip3_dex_support_complete`: Full HIP-3 multi-DEX implementation
- `session_2026-01-02_hip3_mainnet_support`: Initial HIP-3 planning

## Uncommitted Changes
Part of larger HIP-3 implementation work. Changes ready for commit.
