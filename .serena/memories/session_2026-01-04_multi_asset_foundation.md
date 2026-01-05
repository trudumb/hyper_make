# Session: 2026-01-04 Multi-Asset 1000-Order Foundation

## Summary
Implemented foundation layer for multi-asset market making to utilize the 1000-order limit for improved capital efficiency. Created core infrastructure: AssetId, AssetAllocator, BatchAccumulator, SharedMarginPool.

## Problem Statement
- Current system: 10 orders per asset (5 levels × 2 sides)
- Hyperliquid allows: 1000 orders (default) + 1 per 5M USDC volume
- Opportunity: Multi-asset quoting with volatility-weighted allocation

## Architecture
```
Hub-and-Spoke Pattern:
┌─────────────────────────────────────────────────────────────┐
│              MultiAssetCoordinator (Hub) [Future]           │
│  - Shared margin pool    - Asset allocator                  │
│  - Batch accumulator     - Cross-asset risk                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Worker   │  │ Worker   │  │ Worker   │  │ Worker   │    │
│  │ BTC      │  │ ETH      │  │ SOL      │  │ HYPE     │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `src/market_maker/infra/capacity.rs` | Modified | Expanded constants (ORDER_MANAGER_CAPACITY: 20→1000) |
| `src/market_maker/config.rs` | Modified | Added MultiAssetConfig, AssetSpec structs |
| `src/market_maker/tracking/asset_id.rs` | Created | AssetId type with FNV-1a hashing |
| `src/market_maker/tracking/mod.rs` | Modified | Export AssetId |
| `src/market_maker/multi/mod.rs` | Created | Module exports |
| `src/market_maker/multi/allocator.rs` | Created | Volatility-weighted order allocation |
| `src/market_maker/multi/batch.rs` | Created | Batch accumulator for order operations |
| `src/market_maker/multi/margin_pool.rs` | Created | Shared margin tracking |
| `src/market_maker/mod.rs` | Modified | Export multi module |

## Key Components

### 1. AssetId (`tracking/asset_id.rs`)
- FNV-1a hash of "dex:symbol" or "validator:symbol"
- Compact (u64), hashable, Copy
- Used as key in all multi-asset HashMaps

### 2. AssetAllocator (`multi/allocator.rs`)
Volatility-weighted order budget allocation:
```
weight_i = 1/σ_i  (inverse volatility)
Normalize: weight_i /= Σ weights
Cap: weight_i = min(weight_i, 0.30)  // Concentration limit
Allocate: min 5 levels to all, distribute rest by weight
Scale: High vol regime → more levels for defense
```

### 3. BatchAccumulator (`multi/batch.rs`)
Accumulates operations per quote cycle:
- Execution order: cancels → modifies → places
- Groups by asset for efficient API calls
- Tracks success/failure for fallback handling

### 4. SharedMarginPool (`multi/margin_pool.rs`)
Cross-asset capital management:
- Total utilization limit (80%)
- Per-asset concentration limit (30%)
- Reduce-only trigger at 80% utilization
- Quoting capacity calculation per asset

### 5. MultiAssetConfig (`config.rs`)
Configuration for multi-asset mode:
```rust
pub struct MultiAssetConfig {
    pub assets: Vec<AssetSpec>,
    pub total_order_limit: usize,     // 1000 default
    pub min_levels_per_asset: usize,  // 5 minimum
    pub max_levels_per_asset: usize,  // 25 maximum
    pub rebalance_interval_secs: u64, // 300 (5 min)
    pub max_concentration_pct: f64,   // 0.30 (30%)
}
```

## Capacity Constants Updated
```rust
ORDER_MANAGER_CAPACITY: 1000  // from 20
BULK_ORDER_CAPACITY: 200      // from 10
LADDER_LEVEL_INLINE_CAPACITY: 25  // from 8
DEPTH_INLINE_CAPACITY: 25     // from 8
MAX_MULTI_ASSETS: 20          // new
DEFAULT_ORDER_LIMIT: 1000     // new
```

## Verification
- `cargo build` - Success
- All new code has unit tests
- Backward compatibility preserved (single-asset mode unchanged)

## Future Work (Phase 2+)
1. **AssetWorker**: Extract per-asset logic from MarketMaker
2. **MultiAssetCoordinator**: Central orchestrator with event routing
3. **CLI Integration**: `--multi-asset BTC,ETH,SOL` flag
4. **Cross-asset risk**: Unified kill switch triggers
5. **Correlation tracking**: Diversification-adjusted limits

## Usage (Future)
```bash
# Multi-asset mode (future)
cargo run --bin market_maker -- \
  --multi-asset BTC,ETH,SOL,AVAX,DOGE \
  --total-order-limit 500 \
  --min-levels 5 \
  --max-levels 15
```

## Session Metrics
- Duration: ~45 minutes
- Files created: 5
- Files modified: 4
- Build status: Success
- Tests: Multi module tests pass (existing test failures in kappa_orchestrator unrelated)
