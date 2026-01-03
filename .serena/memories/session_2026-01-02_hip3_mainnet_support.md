# HIP-3 Mainnet Support Implementation

## Session Date: 2026-01-02

## Summary
Implemented comprehensive HIP-3 (builder-deployed perpetuals) support following the plan at `PLAN_HIP3_MAINNET_SUPPORT.md`. Key design principle: **ZERO HOT-PATH OVERHEAD** - all HIP-3 detection is pre-computed at startup.

## Implementation Phases Completed

### Phase 0: AssetRuntimeConfig (Pre-computed Configuration)
**File:** `src/market_maker/config.rs`
- `AssetRuntimeConfig` struct with pre-computed fields:
  - `is_cross: bool` - margin mode (false for HIP-3)
  - `is_hip3: bool` - builder-deployed status
  - `oi_cap_usd: f64` - OI cap (f64::MAX if none)
  - `deployer: Option<Arc<str>>` - builder address
  - `sz_multiplier: f64` - for size truncation
- Inline helpers: `remaining_oi_capacity()`, `truncate_size()`

### Phase 1: AssetMeta HIP-3 Fields
**File:** `src/meta.rs`
- Extended `AssetMeta` with: `deployer`, `dex_id`, `oi_cap_usd`, `is_builder_deployed`
- Added `AssetType` enum: `ValidatorPerp`, `BuilderPerp`, `Spot`
- Detection methods: `is_hip3()`, `allows_cross_margin()`, `asset_type()`
- HIP-3 detected via: `only_isolated`, `margin_mode: "noCross"/"strictIsolated"`, or `is_builder_deployed`

### Phase 2: MarginMode Infrastructure
**File:** `src/market_maker/infra/margin.rs`
- `MarginMode` enum with `Cross` and `Isolated` variants
- `MarginMode::from_asset_meta()` constructor
- Updated `MarginConfig` with `margin_mode` and `initial_isolated_margin`

### Phase 3 & 4: Exchange Client & Market Maker Integration
**File:** `src/bin/market_maker.rs`
- Runtime config created from asset metadata at startup
- Leverage call uses `runtime_config.is_cross` (not hardcoded `true`)
- `MmConfig` includes `runtime: AssetRuntimeConfig` and `initial_isolated_margin`
- Logs HIP-3 status at startup with deployer, OI cap info

### Phase 5: Risk Management Adjustments
**File:** `src/market_maker/risk/kill_switch.rs`
- `KillSwitchConfig::for_margin_mode()` method
- Isolated margin gets tighter limits:
  - 20% tighter drawdown (0.8x multiplier)
  - 30% tighter position value (0.7x multiplier)

### Phase 6-8: Builder Fee Tracking
**File:** `src/types/trades.rs`
- `TradeInfo` extended with `builder_fee: Option<String>` (serde default)
- Helper methods: `total_fee()`, `builder_fee_amount()`, `protocol_fee()`, `is_hip3_fill()`
- Fee calculation: `protocol_fee = total_fee - builder_fee`

### Phase 9: OI Cap Enforcement
**File:** `src/market_maker/mod.rs`
- Pre-flight OI check in `update_quotes()` after warmup check
- Uses pre-computed `remaining_oi_capacity(current_notional)`
- Skips quotes with warning log when OI cap reached

### Phase 10: CLI Enhancements
**File:** `src/bin/market_maker.rs`
- `--initial-isolated-margin` flag (default $1000)
- `--force-isolated` flag to override cross margin on validator perps

### Phase 11: Tests
- 9 new tests for `AssetRuntimeConfig` in `src/market_maker/config.rs`
- 4 new tests for `TradeInfo` fee methods in `src/types/trades.rs`
- Total: 524 tests passing

## Key Design Decisions

### Zero Hot-Path Overhead
All HIP-3 detection happens ONCE at startup via `AssetRuntimeConfig::from_asset_meta()`. Hot paths only read pre-computed booleans/floats - no string comparisons or Option unwraps.

### Margin Mode Detection
```rust
pub fn is_hip3(&self) -> bool {
    self.is_builder_deployed.unwrap_or(false)
        || self.only_isolated.unwrap_or(false)
        || self.margin_mode.as_deref() == Some("noCross")
        || self.margin_mode.as_deref() == Some("strictIsolated")
}
```

### OI Cap Fast Path
```rust
#[inline(always)]
pub fn remaining_oi_capacity(&self, current_oi: f64) -> f64 {
    (self.oi_cap_usd - current_oi).max(0.0)
}
// Fast path: oi_cap_usd == f64::MAX for unlimited
```

## Usage

```bash
# HIP-3 asset (auto-detected)
cargo run --bin market_maker -- --asset MEMECOIN

# With custom isolated margin
cargo run --bin market_maker -- --asset MEMECOIN --initial-isolated-margin 500.0

# Force isolated mode on validator perp (testing)
cargo run --bin market_maker -- --asset BTC --force-isolated
```

## Files Modified
- `src/meta.rs` - AssetMeta, AssetType
- `src/market_maker/config.rs` - AssetRuntimeConfig, MarketMakerConfig
- `src/market_maker/infra/margin.rs` - MarginMode
- `src/market_maker/risk/kill_switch.rs` - for_margin_mode()
- `src/market_maker/mod.rs` - OI cap enforcement
- `src/types/trades.rs` - TradeInfo builder_fee
- `src/bin/market_maker.rs` - CLI flags, runtime config integration
- `src/lib.rs` - AssetRuntimeConfig export
