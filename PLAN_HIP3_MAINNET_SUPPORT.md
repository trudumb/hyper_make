# HIP-3 Mainnet Support Implementation Plan

## Executive Summary

This plan adds support for HIP-3 (builder-deployed perpetuals) to the Hyperliquid Rust SDK and market maker. HIP-3 assets have specific constraints that differ from validator-operated perps, primarily **isolated-only margin mode**.

## HIP-3 Key Constraints (from Official Docs)

| Constraint | Value |
|-----------|-------|
| **Margin Mode** | Isolated-only (cross margin NOT supported) |
| **Fee Structure** | 2x normal fees (deployer gets 50%) |
| **OI Caps** | Notional limits set by deployer |
| **Leverage** | Configurable by deployer, may differ from mainnet defaults |
| **Slashing Risk** | 500k HYPE stake, burned if malicious |

**Sources:**
- [HIP-3: Builder-deployed perpetuals](https://hyperliquid.gitbook.io/hyperliquid-docs/hyperliquid-improvement-proposals-hips/hip-3-builder-deployed-perpetuals)
- [FalconX HIP-3 Analysis](https://www.falconx.io/newsroom/the-transformational-potential-of-hyperliquids-hip-3)

---

## Phase 1: Asset Metadata Enhancement

### 1.1 Extend `AssetMeta` Structure

**File:** `src/meta.rs`

Add new fields to capture HIP-3-specific metadata:

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

    // NEW: HIP-3 specific fields
    /// Deployer address (present for HIP-3 assets)
    #[serde(default)]
    pub deployer: Option<String>,
    /// DEX identifier for builder-deployed perps
    #[serde(default)]
    pub dex_id: Option<u32>,
    /// Open interest cap in USD notional
    #[serde(default)]
    pub oi_cap_usd: Option<f64>,
    /// Whether this is a HIP-3 (builder-deployed) asset
    #[serde(default)]
    pub is_builder_deployed: Option<bool>,
}
```

### 1.2 Add Asset Type Enum

**File:** `src/meta.rs`

```rust
/// Asset deployment type for margin mode determination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetType {
    /// Validator-operated perp (supports cross margin)
    ValidatorPerp,
    /// Builder-deployed HIP-3 perp (isolated-only)
    BuilderPerp,
    /// Spot asset (no margin)
    Spot,
}

impl AssetMeta {
    /// Determine asset type from metadata.
    pub fn asset_type(&self) -> AssetType {
        if self.is_builder_deployed.unwrap_or(false)
           || self.only_isolated.unwrap_or(false)
           || self.margin_mode.as_deref() == Some("noCross") {
            AssetType::BuilderPerp
        } else {
            AssetType::ValidatorPerp
        }
    }

    /// Check if cross margin is allowed.
    pub fn allows_cross_margin(&self) -> bool {
        matches!(self.asset_type(), AssetType::ValidatorPerp)
    }

    /// Check if this is a HIP-3 asset.
    pub fn is_hip3(&self) -> bool {
        matches!(self.asset_type(), AssetType::BuilderPerp)
    }
}
```

### 1.3 Extend `AssetLeverageConfig`

**File:** `src/meta.rs`

```rust
impl AssetLeverageConfig {
    /// Create leverage config from asset metadata with HIP-3 awareness.
    pub fn from_asset_meta(meta: &AssetMeta) -> Self {
        Self {
            asset: meta.name.clone(),
            max_leverage: meta.max_leverage as f64,
            isolated_only: meta.only_isolated.unwrap_or(false)
                          || meta.is_builder_deployed.unwrap_or(false)
                          || meta.margin_mode.as_deref() == Some("noCross"),
            tiers: vec![],
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

## Phase 7: CLI Enhancements

### 7.1 Add Margin Mode CLI Flag

**File:** `src/bin/market_maker.rs`

```rust
#[derive(Parser)]
struct Cli {
    // ... existing fields ...

    /// Override margin mode (auto-detected by default)
    #[arg(long)]
    margin_mode: Option<String>,

    /// Initial isolated margin allocation in USD
    #[arg(long, default_value = "1000.0")]
    initial_isolated_margin: f64,

    /// Force isolated margin mode even for cross-capable assets
    #[arg(long)]
    force_isolated: bool,
}
```

### 7.2 Add HIP-3 Asset Info Command

```rust
#[derive(Subcommand)]
enum Commands {
    // ... existing commands ...

    /// Show HIP-3 asset information
    Hip3Info {
        /// Asset to query
        #[arg(long)]
        asset: String,
    },
}
```

---

## Implementation Order

| Phase | Priority | Estimated Files Changed |
|-------|----------|------------------------|
| 1. Asset Metadata | P0 | `src/meta.rs` |
| 2. Margin Infrastructure | P0 | `src/types/margin.rs`, `src/market_maker/infra/margin.rs` |
| 3. Exchange Client | P1 | `src/exchange/accounts.rs`, `src/info/info_client.rs` |
| 4. Market Maker | P1 | `src/bin/market_maker.rs`, `src/market_maker/mod.rs`, `src/market_maker/config.rs` |
| 5. Risk Management | P2 | `src/market_maker/risk/` |
| 6. Testing | P1 | Various test modules |
| 7. CLI Enhancements | P3 | `src/bin/market_maker.rs` |

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Incorrect margin mode detection | Order rejections | Validate against live API response |
| Insufficient isolated margin | Liquidation | Monitor liquidation price, add margin buffer |
| Fee assumptions (2x) | P&L miscalculation | Fetch actual fee from exchange |
| OI cap exceeded | Order rejection | Track OI and reject oversized orders |

---

## Validation Checklist

- [ ] Cross margin assets still work as before
- [ ] HIP-3 assets correctly use isolated mode
- [ ] Leverage is set with correct `is_cross` flag
- [ ] Initial isolated margin is allocated
- [ ] Kill switch triggers appropriately for isolated positions
- [ ] Liquidation price is monitored for isolated positions
- [ ] CLI displays margin mode in status output
- [ ] All existing tests pass
- [ ] New tests cover HIP-3 detection and margin mode logic
