# HIP-3 Mainnet Support Implementation Plan (v2 - Latency Optimized)

## Executive Summary

This plan adds support for HIP-3 (builder-deployed perpetuals) to the Hyperliquid Rust SDK and market maker. HIP-3 assets have specific constraints that differ from validator-operated perps, primarily **isolated-only margin mode**.

**Key Design Principle: ZERO HOT-PATH OVERHEAD**

All HIP-3 detection and margin mode logic is pre-computed at startup. The quote cycle and message handlers have no additional branches, string comparisons, or Option unwraps compared to current implementation.

## HIP-3 Key Constraints (from Official Docs)

| Constraint | Value |
|-----------|-------|
| **Margin Mode** | Isolated-only (cross margin NOT supported) |
| **Fee Structure** | Deployer configurable 0-300% share (0-100% growth mode) |
| **Fee in API** | `fee` field in fills includes `builderFee` (optional if non-zero) |
| **OI Caps** | Notional limits set by deployer |
| **Leverage** | Configurable by deployer, may differ from mainnet defaults |
| **DEX Abstraction** | Automatic collateral routing from validator perps balance |
| **Slashing Risk** | 500k HYPE stake, burned if malicious |

**Sources:**
- [HIP-3: Builder-deployed perpetuals](https://hyperliquid.gitbook.io/hyperliquid-docs/hyperliquid-improvement-proposals-hips/hip-3-builder-deployed-perpetuals)
- [HIP-3 Deployer Actions](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/hip-3-deployer-actions)
- [Fees Documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees)
- [maxBuilderFee API](https://docs.chainstack.com/reference/hyperliquid-info-max-builder-fee)

---

## CRITICAL: Latency Considerations

### Hot Path Analysis

The following paths are latency-critical (< 1ms target):
1. **Quote cycle** (`update_quotes()`) - runs every price tick
2. **Message handlers** (`handle_all_mids`, `handle_trades`, `handle_l2_book`)
3. **Order placement** (`place_bulk_orders`)
4. **Fill processing** (`handle_user_fills`)

### Design Principles

1. **Pre-compute everything at startup** - margin mode, OI caps, fee multipliers
2. **Use `#[repr(u8)]` enums** - single-byte comparison, not string matching
3. **Store pre-computed `is_cross: bool`** - avoid enum matching in hot paths
4. **No `Option::unwrap()` in hot paths** - use pre-resolved values
5. **Inline critical methods** - `#[inline(always)]` for margin checks

---

## Phase 0: Pre-Computed Asset Configuration (LATENCY CRITICAL)

### 0.1 Unified Asset Runtime Config

**File:** `src/market_maker/config.rs`

Create a **pre-computed** runtime configuration that resolves all Option types and string comparisons at startup:

```rust
/// Pre-computed asset configuration for zero-overhead hot paths.
///
/// All HIP-3 detection is done ONCE at startup. Quote cycle uses only
/// primitive fields (bool, f64) with no Option unwraps or string comparisons.
///
/// NOTE: Fees are NOT pre-computed. HIP-3 builder fees vary per deployer
/// (0-300% share) and are included in the `fee` field of each fill.
/// See Phase 8 for fee handling.
#[derive(Debug, Clone)]
pub struct AssetRuntimeConfig {
    // === Pre-computed margin fields (NO OPTIONS) ===
    /// Whether to use cross margin (pre-computed from AssetMeta)
    /// HOT PATH: Used directly in margin calculations
    pub is_cross: bool,

    /// Open interest cap in USD (f64::MAX if no cap)
    /// HOT PATH: Pre-flight check before order placement
    pub oi_cap_usd: f64,

    /// Pre-computed sz_decimals as f64 power for truncation
    /// HOT PATH: Avoids powi() call in size formatting
    pub sz_multiplier: f64,

    /// Pre-computed price decimals multiplier
    pub price_multiplier: f64,

    // === Cold path metadata (startup only) ===
    /// Asset name (Arc for cheap cloning)
    pub asset: Arc<str>,

    /// Maximum leverage (from API)
    pub max_leverage: f64,

    /// Whether this is a HIP-3 builder-deployed asset
    pub is_hip3: bool,

    /// Deployer address (for logging/display only)
    pub deployer: Option<Arc<str>>,
}

impl AssetRuntimeConfig {
    /// Build from API metadata - called ONCE at startup.
    pub fn from_asset_meta(meta: &AssetMeta) -> Self {
        // Resolve HIP-3 status ONCE
        let is_hip3 = meta.is_builder_deployed.unwrap_or(false)
            || meta.only_isolated.unwrap_or(false)
            || meta.margin_mode.as_deref() == Some("noCross")
            || meta.margin_mode.as_deref() == Some("strictIsolated");

        Self {
            // Pre-compute for hot path
            is_cross: !is_hip3,
            oi_cap_usd: meta.oi_cap_usd.unwrap_or(f64::MAX),
            sz_multiplier: 10_f64.powi(meta.sz_decimals as i32),
            price_multiplier: 10_f64.powi(5), // 5 sig figs for perps

            // Cold path storage
            asset: Arc::from(meta.name.as_str()),
            max_leverage: meta.max_leverage as f64,
            is_hip3,
            deployer: meta.deployer.as_ref().map(|d| Arc::from(d.as_str())),
        }
    }

    /// Fast size truncation (hot path).
    #[inline(always)]
    pub fn truncate_size(&self, size: f64) -> f64 {
        (size * self.sz_multiplier).trunc() / self.sz_multiplier
    }

    /// Check OI cap (hot path) - returns max additional notional allowed.
    #[inline(always)]
    pub fn remaining_oi_capacity(&self, current_oi: f64) -> f64 {
        (self.oi_cap_usd - current_oi).max(0.0)
    }
}
```

### 0.2 Update MarketMakerConfig

**File:** `src/market_maker/config.rs`

```rust
#[derive(Debug, Clone)]
pub struct MarketMakerConfig {
    // ... existing fields ...

    /// Pre-computed runtime config (resolved at startup)
    /// All hot-path reads go through this.
    pub runtime: AssetRuntimeConfig,
}
```

### 0.3 Startup Resolution Pattern

**File:** `src/bin/market_maker.rs`

```rust
// At startup, resolve everything ONCE:
let asset_meta = meta.universe.iter()
    .find(|a| a.name == asset)
    .ok_or_else(|| anyhow!("Asset {} not found", asset))?;

let runtime_config = AssetRuntimeConfig::from_asset_meta(asset_meta);

info!(
    asset = %runtime_config.asset,
    is_cross = runtime_config.is_cross,
    is_hip3 = runtime_config.is_hip3,
    oi_cap = %if runtime_config.oi_cap_usd == f64::MAX {
        "unlimited".to_string()
    } else {
        format!("${:.0}", runtime_config.oi_cap_usd)
    },
    "Asset runtime config resolved (fees from fill data)"
);

// Set leverage with pre-computed is_cross
exchange_client
    .update_leverage(leverage, &asset, runtime_config.is_cross, None)
    .await?;
```

---

## Phase 1: Asset Metadata Enhancement

### 1.1 Extend `AssetMeta` Structure

**File:** `src/meta.rs`

Add new fields to capture HIP-3-specific metadata from API:

```rust
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AssetMeta {
    pub name: String,
    pub sz_decimals: u32,
    pub max_leverage: usize,

    // Existing margin fields
    #[serde(default)]
    pub only_isolated: Option<bool>,
    #[serde(default)]
    pub margin_mode: Option<String>,
    #[serde(default)]
    pub is_delisted: Option<bool>,

    // NEW: HIP-3 specific fields (parsed from API, used only at startup)
    #[serde(default)]
    pub deployer: Option<String>,
    #[serde(default)]
    pub dex_id: Option<u32>,
    #[serde(default)]
    pub oi_cap_usd: Option<f64>,
    #[serde(default)]
    pub is_builder_deployed: Option<bool>,
}
```

### 1.2 Add Compact Asset Type Enum

**File:** `src/meta.rs`

```rust
/// Asset deployment type - compact representation for startup logic.
/// Uses repr(u8) for minimal memory footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AssetType {
    ValidatorPerp = 0,
    BuilderPerp = 1,  // HIP-3
    Spot = 2,
}

impl AssetMeta {
    /// Determine asset type - called ONCE at startup.
    /// Do NOT call in hot paths - use AssetRuntimeConfig.is_hip3 instead.
    pub fn asset_type(&self) -> AssetType {
        if self.is_builder_deployed.unwrap_or(false)
           || self.only_isolated.unwrap_or(false)
           || self.margin_mode.as_deref() == Some("noCross")
           || self.margin_mode.as_deref() == Some("strictIsolated") {
            AssetType::BuilderPerp
        } else {
            AssetType::ValidatorPerp
        }
    }
}
```

---

## Phase 2: Margin Mode Infrastructure

### 2.1 Add `MarginMode` Enum

**File:** `src/types/margin.rs` (new file)

```rust
/// Margin mode for a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarginMode {
    /// Cross margin - shares margin across all positions
    #[default]
    Cross,
    /// Isolated margin - each position has dedicated margin
    Isolated,
}

impl MarginMode {
    /// Get `is_cross` flag for API calls.
    pub fn is_cross(&self) -> bool {
        matches!(self, MarginMode::Cross)
    }

    /// Create from asset metadata.
    pub fn from_asset_meta(meta: &AssetMeta) -> Self {
        if meta.allows_cross_margin() {
            MarginMode::Cross
        } else {
            MarginMode::Isolated
        }
    }
}
```

### 2.2 Add Margin Mode to `MarginConfig`

**File:** `src/market_maker/infra/margin.rs`

```rust
#[derive(Debug, Clone)]
pub struct MarginConfig {
    // ... existing fields ...

    /// Margin mode for this asset (derived from metadata)
    pub margin_mode: MarginMode,

    /// Initial isolated margin to allocate (USD)
    /// Only used when margin_mode == Isolated
    pub initial_isolated_margin: f64,
}

impl MarginConfig {
    /// Create from asset metadata with proper margin mode.
    pub fn from_asset_meta(asset_meta: &AssetMeta) -> Self {
        let leverage_config = AssetLeverageConfig::from_asset_meta(asset_meta);
        let margin_mode = MarginMode::from_asset_meta(asset_meta);

        Self {
            max_leverage: leverage_config.max_leverage,
            leverage_config: Some(leverage_config),
            margin_buffer: 0.8,
            max_notional_position: 100_000.0,
            max_order_notional: 10_000.0,
            refresh_interval: Duration::from_secs(10),
            margin_mode,
            initial_isolated_margin: 1000.0, // $1000 default
        }
    }
}
```

### 2.3 Extend `MarginState` for Dual Tracking

**File:** `src/market_maker/infra/margin.rs`

```rust
/// Margin state supporting both cross and isolated modes.
#[derive(Debug, Clone, Default)]
pub struct MarginState {
    // Cross margin state
    pub cross_account_value: f64,
    pub cross_margin_used: f64,
    pub cross_available_margin: f64,

    // Isolated margin state (for the specific asset)
    pub isolated_margin: f64,
    pub isolated_position_value: f64,
    pub isolated_liquidation_price: Option<f64>,

    // Effective values (based on mode)
    pub account_value: f64,
    pub margin_used: f64,
    pub total_notional: f64,
    pub available_margin: f64,
    pub current_leverage: f64,
    pub last_updated: Option<Instant>,

    /// Current margin mode
    pub margin_mode: MarginMode,
}

impl MarginState {
    /// Create from user state response with mode awareness.
    pub fn from_user_state(
        response: &UserStateResponse,
        asset_position: Option<&AssetPosition>,
        margin_mode: MarginMode,
    ) -> Self {
        let cross_value = response.cross_margin_summary.account_value
            .parse::<f64>().unwrap_or(0.0);
        let cross_used = response.cross_margin_summary.total_margin_used
            .parse::<f64>().unwrap_or(0.0);

        let isolated_value = response.margin_summary.account_value
            .parse::<f64>().unwrap_or(0.0);

        // Select effective values based on mode
        let (account_value, margin_used, total_notional) = match margin_mode {
            MarginMode::Cross => {
                let ntl = response.cross_margin_summary.total_ntl_pos
                    .parse::<f64>().unwrap_or(0.0);
                (cross_value, cross_used, ntl)
            }
            MarginMode::Isolated => {
                // For isolated, get position-specific values
                let (ntl, margin) = asset_position
                    .map(|p| {
                        let pos_val = p.position.position_value
                            .parse::<f64>().unwrap_or(0.0);
                        let margin = p.position.margin_used
                            .parse::<f64>().unwrap_or(0.0);
                        (pos_val, margin)
                    })
                    .unwrap_or((0.0, 0.0));
                (isolated_value, margin, ntl)
            }
        };

        let available = (account_value - margin_used).max(0.0);
        let leverage = if account_value > 0.0 { total_notional / account_value } else { 0.0 };

        Self {
            cross_account_value: cross_value,
            cross_margin_used: cross_used,
            cross_available_margin: (cross_value - cross_used).max(0.0),
            isolated_margin: isolated_value,
            isolated_position_value: total_notional,
            isolated_liquidation_price: None, // Fetch separately if needed
            account_value,
            margin_used,
            total_notional,
            available_margin: available,
            current_leverage: leverage,
            last_updated: Some(Instant::now()),
            margin_mode,
        }
    }
}
```

---

## Phase 3: Exchange Client Updates

### 3.1 Add Margin Mode Parameter to Leverage Update

**File:** `src/exchange/accounts.rs`

The existing `update_leverage` already supports `is_cross` parameter. No changes needed, but we need to use it correctly.

### 3.2 Add Isolated Margin Management

**File:** `src/exchange/accounts.rs`

Existing `update_isolated_margin` method is sufficient. Add helper:

```rust
impl ExchangeClient {
    /// Set up margin mode for an asset based on its metadata.
    ///
    /// For HIP-3 assets, this:
    /// 1. Sets leverage in isolated mode
    /// 2. Allocates initial isolated margin
    pub async fn setup_margin_for_asset(
        &self,
        asset_meta: &AssetMeta,
        leverage: u32,
        initial_margin_usd: Option<f64>,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<()> {
        let is_cross = asset_meta.allows_cross_margin();

        // Set leverage with correct mode
        self.update_leverage(leverage, &asset_meta.name, is_cross, wallet).await?;

        // For isolated assets, allocate initial margin
        if !is_cross {
            if let Some(margin) = initial_margin_usd {
                self.update_isolated_margin(margin, &asset_meta.name, wallet).await?;
            }
        }

        Ok(())
    }
}
```

### 3.3 Add HIP-3 Info Requests

**File:** `src/info/info_client.rs`

```rust
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "camelCase")]
pub enum InfoRequest {
    // ... existing variants ...

    /// Get user's HIP-3 DEX abstraction state
    #[serde(rename = "userDexAbstraction")]
    UserDexAbstraction {
        user: Address,
    },

    /// Get builder fee approval status
    #[serde(rename = "maxBuilderFee")]
    MaxBuilderFee {
        user: Address,
        builder: Address,
    },
}

impl InfoClient {
    /// Get user's HIP-3 DEX positions and state.
    pub async fn user_dex_abstraction(&self, address: Address) -> Result<UserDexAbstractionResponse> {
        let input = InfoRequest::UserDexAbstraction { user: address };
        self.send_info_request(input).await
    }

    /// Get approved builder fee for a specific builder.
    pub async fn max_builder_fee(&self, user: Address, builder: Address) -> Result<MaxBuilderFeeResponse> {
        let input = InfoRequest::MaxBuilderFee { user, builder };
        self.send_info_request(input).await
    }
}
```

---

## Phase 4: Market Maker Integration

### 4.1 Add Margin Mode to MarketMakerConfig

**File:** `src/market_maker/config.rs`

```rust
#[derive(Debug, Clone)]
pub struct MarketMakerConfig {
    // ... existing fields ...

    /// Margin mode for the asset (auto-detected from metadata)
    pub margin_mode: MarginMode,

    /// Initial isolated margin allocation (USD)
    /// Used when margin_mode == Isolated
    pub initial_isolated_margin: f64,

    /// Whether this is a HIP-3 builder-deployed asset
    pub is_hip3_asset: bool,
}
```

### 4.2 Update Market Maker Binary

**File:** `src/bin/market_maker.rs`

```rust
// In the leverage setup section (around line 720):

// Detect margin mode from asset metadata
let asset_meta = meta.universe.iter()
    .find(|a| a.name == asset)
    .ok_or_else(|| anyhow!("Asset {} not found in metadata", asset))?;

let margin_mode = MarginMode::from_asset_meta(asset_meta);
let is_cross = margin_mode.is_cross();

info!(
    asset = %asset,
    margin_mode = ?margin_mode,
    is_hip3 = asset_meta.is_hip3(),
    only_isolated = ?asset_meta.only_isolated,
    "Detected margin mode for asset"
);

// Set leverage with correct margin mode
match exchange_client
    .update_leverage(leverage, &asset, is_cross, None)
    .await
{
    Ok(response) => {
        info!(
            leverage = leverage,
            is_cross = is_cross,
            response = ?response,
            "Leverage set successfully"
        );
    }
    Err(e) => {
        error!(error = %e, "Failed to set leverage");
        return Err(e.into());
    }
}

// For isolated margin assets, allocate initial margin
if !is_cross {
    let initial_margin = config.initial_isolated_margin.unwrap_or(1000.0);
    info!(
        margin = initial_margin,
        asset = %asset,
        "Allocating initial isolated margin"
    );

    exchange_client
        .update_isolated_margin(initial_margin, &asset, None)
        .await?;
}
```

### 4.3 Update Margin Sizer Usage

**File:** `src/market_maker/mod.rs`

```rust
// In the margin state update section:

fn update_margin_state(&mut self, user_state: &UserStateResponse) {
    // Find our asset's position
    let asset_position = user_state.asset_positions.iter()
        .find(|p| p.position.coin == self.config.asset.as_ref());

    // Update with correct margin mode
    let new_state = MarginState::from_user_state(
        user_state,
        asset_position,
        self.config.margin_mode,
    );

    self.infra.margin_sizer.update_from_state(new_state);
}
```

---

## Phase 5: Risk Management Adjustments

### 5.1 Isolated Margin Kill Switch

**File:** `src/market_maker/risk/kill_switch.rs`

```rust
impl KillSwitchConfig {
    /// Create config appropriate for margin mode.
    pub fn for_margin_mode(margin_mode: MarginMode, base_config: Self) -> Self {
        match margin_mode {
            MarginMode::Cross => base_config,
            MarginMode::Isolated => {
                // Isolated margin is more dangerous - tighter limits
                Self {
                    // Track isolated margin specifically
                    max_drawdown_pct: base_config.max_drawdown_pct * 0.8, // 20% tighter
                    // Isolated positions can liquidate faster
                    max_position_value: base_config.max_position_value * 0.7,
                    ..base_config
                }
            }
        }
    }
}
```

### 5.2 Add Liquidation Price Monitoring

**File:** `src/market_maker/risk/monitors/liquidation.rs` (new file)

```rust
/// Monitor liquidation price for isolated margin positions.
pub struct LiquidationPriceMonitor {
    /// Minimum distance from mid to liquidation (as fraction)
    min_distance_fraction: f64,
    /// Current liquidation price (if known)
    liquidation_price: Option<f64>,
    /// Position side (true = long)
    is_long: Option<bool>,
}

impl LiquidationPriceMonitor {
    pub fn new(min_distance_fraction: f64) -> Self {
        Self {
            min_distance_fraction,
            liquidation_price: None,
            is_long: None,
        }
    }

    pub fn update(&mut self, liquidation_price: f64, is_long: bool) {
        self.liquidation_price = Some(liquidation_price);
        self.is_long = Some(is_long);
    }

    /// Check if mid price is dangerously close to liquidation.
    pub fn check_distance(&self, mid_price: f64) -> RiskCheckResult {
        match (self.liquidation_price, self.is_long) {
            (Some(liq_px), Some(is_long)) => {
                let distance = if is_long {
                    (mid_price - liq_px) / mid_price
                } else {
                    (liq_px - mid_price) / mid_price
                };

                if distance < self.min_distance_fraction {
                    RiskCheckResult::Critical(format!(
                        "Liquidation price {:.2} is {:.1}% from mid {:.2}",
                        liq_px, distance * 100.0, mid_price
                    ))
                } else if distance < self.min_distance_fraction * 2.0 {
                    RiskCheckResult::Warning(format!(
                        "Approaching liquidation: {:.1}% distance",
                        distance * 100.0
                    ))
                } else {
                    RiskCheckResult::Ok
                }
            }
            _ => RiskCheckResult::Ok,
        }
    }
}
```

---

## Phase 6: Testing & Validation

### 6.1 Add HIP-3 Detection Tests

**File:** `src/meta.rs` (in tests module)

```rust
#[test]
fn test_hip3_detection_only_isolated() {
    let meta = AssetMeta {
        name: "MEMECOIN".to_string(),
        sz_decimals: 0,
        max_leverage: 3,
        only_isolated: Some(true),
        margin_mode: None,
        is_delisted: None,
        deployer: None,
        dex_id: None,
        oi_cap_usd: None,
        is_builder_deployed: None,
    };

    assert!(meta.is_hip3());
    assert!(!meta.allows_cross_margin());
    assert_eq!(meta.asset_type(), AssetType::BuilderPerp);
}

#[test]
fn test_hip3_detection_margin_mode() {
    let meta = AssetMeta {
        name: "EXOTIC".to_string(),
        sz_decimals: 2,
        max_leverage: 5,
        only_isolated: None,
        margin_mode: Some("noCross".to_string()),
        is_delisted: None,
        deployer: Some("0x1234...".to_string()),
        dex_id: Some(1),
        oi_cap_usd: Some(10_000_000.0),
        is_builder_deployed: Some(true),
    };

    assert!(meta.is_hip3());
    assert!(!meta.allows_cross_margin());
}

#[test]
fn test_validator_perp_allows_cross() {
    let meta = AssetMeta {
        name: "BTC".to_string(),
        sz_decimals: 5,
        max_leverage: 50,
        only_isolated: None,
        margin_mode: None,
        is_delisted: None,
        deployer: None,
        dex_id: None,
        oi_cap_usd: None,
        is_builder_deployed: None,
    };

    assert!(!meta.is_hip3());
    assert!(meta.allows_cross_margin());
    assert_eq!(meta.asset_type(), AssetType::ValidatorPerp);
}
```

### 6.2 Add Margin Mode Tests

**File:** `src/market_maker/infra/margin.rs` (in tests module)

```rust
#[test]
fn test_margin_config_from_hip3_asset() {
    let meta = AssetMeta {
        name: "HIP3COIN".to_string(),
        sz_decimals: 2,
        max_leverage: 10,
        only_isolated: Some(true),
        margin_mode: None,
        is_delisted: None,
        deployer: Some("0xbuilder".to_string()),
        dex_id: Some(5),
        oi_cap_usd: Some(5_000_000.0),
        is_builder_deployed: Some(true),
    };

    let config = MarginConfig::from_asset_meta(&meta);

    assert_eq!(config.margin_mode, MarginMode::Isolated);
    assert_eq!(config.max_leverage, 10.0);
    assert!(config.leverage_config.as_ref().unwrap().isolated_only);
}

#[test]
fn test_margin_state_isolated_mode() {
    // Simulate isolated margin state
    let state = MarginState {
        margin_mode: MarginMode::Isolated,
        isolated_margin: 1000.0,
        isolated_position_value: 5000.0,
        account_value: 1000.0,
        margin_used: 500.0,
        available_margin: 500.0,
        current_leverage: 5.0,
        ..Default::default()
    };

    assert_eq!(state.margin_mode, MarginMode::Isolated);
    assert_eq!(state.current_leverage, 5.0);
}
```

---

## Phase 7: DEX Abstraction & Collateral Routing (NEW)

### 7.1 Enable DEX Abstraction at Startup

**File:** `src/exchange/actions.rs`

```rust
/// Enable HIP-3 DEX abstraction for automatic collateral routing.
/// When enabled, actions on HIP-3 perps automatically transfer collateral
/// from validator-operated USDC perps balance.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct UserDexAbstraction {
    pub enable: bool,
}
```

**File:** `src/exchange/accounts.rs`

```rust
impl ExchangeClient {
    /// Enable HIP-3 DEX abstraction for automatic collateral routing.
    /// Call ONCE at startup before trading HIP-3 assets.
    pub async fn enable_dex_abstraction(
        &self,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let timestamp = next_nonce();

        let action = Actions::UserDexAbstraction(UserDexAbstraction { enable: true });
        let connection_id = action.hash(timestamp, self.vault_address)?;
        let action = serde_json::to_value(&action)?;
        let is_mainnet = self.http_client.is_mainnet();
        let signature = sign_l1_action(wallet, connection_id, is_mainnet)?;

        self.post(action, signature, timestamp).await
    }
}
```

### 7.2 Startup Sequence for HIP-3 Assets

**File:** `src/bin/market_maker.rs`

```rust
// At startup, BEFORE setting leverage:
if runtime_config.is_hip3 {
    info!("Enabling DEX abstraction for HIP-3 asset {}", asset);
    exchange_client.enable_dex_abstraction(None).await?;
}

// Then set leverage with correct margin mode
exchange_client
    .update_leverage(leverage, &asset, runtime_config.is_cross, None)
    .await?;

// For isolated assets, allocate initial margin
if !runtime_config.is_cross {
    exchange_client
        .update_isolated_margin(initial_margin, &asset, None)
        .await?;
}
```

---

## Phase 8: Builder Fee Tracking (CORRECTED)

### Key Insight: Fees Are Already in Fill Data

**The `fee` field in fills already includes builder fees.** No multiplier needed.

Per [API docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint):
- `fee`: Total fee (inclusive of `builderFee`)
- `builderFee`: Optional, only present if non-zero

HIP-3 deployers can set fee share from 0-300% (0-100% in growth mode). The actual fee varies per deployer and is reflected in the fill data.

### 8.1 Extend TradeInfo with Builder Fee

**File:** `src/types/trades.rs`

```rust
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TradeInfo {
    pub coin: String,
    pub side: String,
    pub px: String,
    pub sz: String,
    pub time: u64,
    pub hash: String,
    pub start_position: String,
    pub dir: String,
    pub closed_pnl: String,
    pub oid: u64,
    pub cloid: Option<String>,
    pub crossed: bool,
    pub fee: String,           // Total fee (includes builder fee)
    pub fee_token: String,
    pub tid: u64,

    // NEW: HIP-3 builder fee tracking
    /// Builder fee component (optional, only present if non-zero)
    #[serde(default)]
    pub builder_fee: Option<String>,
}

impl TradeInfo {
    /// Get total fee as f64.
    #[inline(always)]
    pub fn total_fee(&self) -> f64 {
        self.fee.parse().unwrap_or(0.0)
    }

    /// Get builder fee component as f64 (0.0 if not present).
    #[inline(always)]
    pub fn builder_fee_amount(&self) -> f64 {
        self.builder_fee
            .as_ref()
            .and_then(|f| f.parse().ok())
            .unwrap_or(0.0)
    }

    /// Get protocol fee component (total - builder).
    #[inline(always)]
    pub fn protocol_fee(&self) -> f64 {
        self.total_fee() - self.builder_fee_amount()
    }
}
```

### 8.2 Update AssetRuntimeConfig (Remove fee_multiplier)

**File:** `src/market_maker/config.rs`

```rust
pub struct AssetRuntimeConfig {
    // REMOVED: fee_multiplier: f64 - fees come from fill data

    // === Pre-computed margin fields (NO OPTIONS) ===
    pub is_cross: bool,
    pub oi_cap_usd: f64,
    pub sz_multiplier: f64,
    pub price_multiplier: f64,

    // === Cold path metadata ===
    pub asset: Arc<str>,
    pub max_leverage: f64,
    pub is_hip3: bool,
    pub deployer: Option<Arc<str>>,
}
```

### 8.3 P&L Tracker Uses Actual Fee

**File:** `src/market_maker/tracking/pnl.rs`

```rust
impl PnLTracker {
    /// Record fill using actual fee from exchange.
    /// The fee already includes builder fee for HIP-3 assets.
    #[inline(always)]
    pub fn record_fill(&mut self, fill: &TradeInfo) {
        let fee = fill.total_fee();  // Use actual fee, not multiplied
        let builder_fee = fill.builder_fee_amount();

        // Track fees separately for analytics
        self.total_fees += fee;
        self.total_builder_fees += builder_fee;
        self.total_protocol_fees += fill.protocol_fee();

        // ... rest of P&L calculation
    }
}
```

### 8.4 Query Fee Schedule at Startup (Optional, for Logging)

**File:** `src/info/info_client.rs`

Add query for user fee schedule to display expected rates:

```rust
impl InfoClient {
    /// Query user's fee schedule (maker/taker rates, discounts).
    pub async fn user_fees(&self, address: Address) -> Result<UserFeesResponse> {
        let input = InfoRequest::UserFees { user: address };
        self.send_info_request(input).await
    }
}
```

At startup, log fee information:

```rust
// Log user fee schedule for awareness
let fee_schedule = info_client.user_fees(user_address).await?;
info!(
    maker_rate = %fee_schedule.fee_schedule.add,
    taker_rate = %fee_schedule.fee_schedule.cross,
    referral_discount = %fee_schedule.active_referral_discount,
    is_hip3 = runtime_config.is_hip3,
    "Fee schedule loaded (HIP-3 builder fees applied at trade time)"
);
```

### 8.5 Prometheus Metrics for Builder Fees

**File:** `src/market_maker/infra/metrics.rs`

```rust
// Add metrics for fee breakdown
mm_total_fees: Gauge,
mm_builder_fees: Gauge,
mm_protocol_fees: Gauge,
mm_builder_fee_ratio: Gauge,  // builder_fees / total_fees
```

---

## Phase 9: OI Cap Enforcement (NEW)

### 9.1 Pre-Flight OI Check in Order Placement

**File:** `src/market_maker/mod.rs`

```rust
/// Check if order would exceed OI cap.
/// Called before placing orders - HOT PATH, must be fast.
#[inline(always)]
fn check_oi_capacity(&self, order_notional: f64) -> bool {
    // Fast path: no cap
    if self.config.runtime.oi_cap_usd == f64::MAX {
        return true;
    }

    // Check remaining capacity
    let current_notional = self.position.position().abs() * self.latest_mid;
    current_notional + order_notional <= self.config.runtime.oi_cap_usd
}
```

### 9.2 Integrate into Quote Cycle

```rust
async fn update_quotes(&mut self) -> Result<()> {
    // ... existing warmup check ...

    // OI cap pre-flight (fast path for unlimited)
    let max_order_notional = self.config.runtime.remaining_oi_capacity(
        self.position.position().abs() * self.latest_mid
    );

    if max_order_notional < MIN_ORDER_NOTIONAL {
        warn!(oi_cap = self.config.runtime.oi_cap_usd, "OI cap reached, skipping quotes");
        return Ok(());
    }

    // ... rest of quote logic with constrained max_order_notional ...
}
```

---

## Phase 10: CLI Enhancements

### 10.1 Add Margin Mode CLI Flag

**File:** `src/bin/market_maker.rs`

```rust
#[derive(Parser)]
struct Cli {
    // ... existing fields ...

    /// Initial isolated margin allocation in USD
    #[arg(long, default_value = "1000.0")]
    initial_isolated_margin: f64,

    /// Force isolated margin mode even for cross-capable assets
    #[arg(long)]
    force_isolated: bool,

    /// Skip DEX abstraction enable (for debugging)
    #[arg(long)]
    skip_dex_abstraction: bool,
}
```

---

## Implementation Order (Revised)

| Phase | Priority | Description | Files |
|-------|----------|-------------|-------|
| **0. Runtime Config** | P0 | Pre-computed asset config | `config.rs` |
| **1. Metadata** | P0 | Parse HIP-3 fields from API | `meta.rs` |
| **2. Margin Infra** | P0 | MarginMode enum, dual tracking | `margin.rs` |
| **3. Exchange Client** | P1 | DEX abstraction, leverage setup | `accounts.rs` |
| **4. Market Maker** | P1 | Use runtime config, OI checks | `mod.rs`, `bin/market_maker.rs` |
| **5. Builder Fee Parsing** | P1 | Parse `builderFee` from fills | `trades.rs`, `pnl.rs` |
| **6. Risk Mgmt** | P2 | Tighter limits for isolated | `kill_switch.rs` |
| **7. Testing** | P1 | Unit + integration tests | Various |

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Incorrect margin mode detection | Order rejections | Validate against live API, log discrepancies |
| Insufficient isolated margin | Liquidation | Monitor liq price, auto top-up if close |
| OI cap exceeded | Order rejection | Pre-flight check in quote cycle |
| DEX abstraction not enabled | Collateral routing fails | Enable at startup, verify state |
| Missing `builderFee` field | Inaccurate fee breakdown | Field is optional, gracefully handle None |

---

## Latency Benchmark Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| HIP-3 detection | 0 ns | Pre-computed at startup |
| Margin mode check | 0 ns | Single bool field access |
| OI cap check | <10 ns | f64 comparison only |
| Fee parsing | ~20 ns | String parse only on fills (not hot path) |

---

## Validation Checklist

- [ ] Cross margin assets still work as before
- [ ] HIP-3 assets correctly use isolated mode
- [ ] Leverage is set with correct `is_cross` flag
- [ ] DEX abstraction enabled for HIP-3 assets
- [ ] Initial isolated margin is allocated
- [ ] OI cap is enforced before order placement
- [ ] `builderFee` parsed from fills when present
- [ ] P&L tracking uses actual fee from fill data (not multiplied)
- [ ] Fee breakdown metrics exported (total/builder/protocol)
- [ ] Kill switch triggers appropriately for isolated positions
- [ ] Liquidation price is monitored for isolated positions
- [ ] CLI displays margin mode in status output
- [ ] All existing tests pass
- [ ] New tests cover HIP-3 detection and margin mode logic
- [ ] **Zero additional latency in quote cycle** (benchmark verified)
