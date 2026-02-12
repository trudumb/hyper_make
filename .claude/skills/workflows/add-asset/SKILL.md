---
name: add-asset
description: End-to-end workflow for onboarding a new asset to the market maker. Use when adding BTC, ETH, SOL, or any Hyperliquid perpetual. Covers viability analysis, config generation via auto_derive, spread profile selection, paper validation, and live deployment.
disable-model-invocation: true
context: fork
agent: general-purpose
argument-hint: "[asset] [capital_usd]"
allowed-tools: Read, Grep, Glob, Bash
---

# Add Asset Workflow

End-to-end process for onboarding a new Hyperliquid perpetual asset to the market maker. This workflow covers viability analysis, configuration generation, paper validation, and live deployment. Follow each step in order.

**Arguments**: `[asset]` is the symbol (e.g., BTC, ETH, SOL, HYPE) and `[capital_usd]` is the capital to allocate in USD.

## Key Files

```
src/market_maker/config/auto_derive.rs          # First-principles parameter derivation
src/market_maker/config/spread_profile.rs        # SpreadProfile enum (Default/Hip3/Aggressive)
src/market_maker/config/core.rs                  # MarketMakerConfig struct
src/market_maker/config/multi_asset.rs           # MultiAssetConfig, AssetSpec
src/market_maker/config/runtime.rs               # AssetRuntimeConfig (HIP-3 detection)
src/market_maker/config/risk.rs                  # DynamicRiskConfig
src/market_maker/multi/allocator.rs              # Volatility-weighted asset allocator
src/market_maker/multi/margin_pool.rs            # Shared margin pool
src/market_maker/infra/binance_feed.rs           # resolve_binance_symbol() — Binance pair mapping
src/market_maker/strategy/signal_integration.rs  # Lead-lag and cross-venue signal flags
src/bin/market_maker.rs                          # CLI flags and startup wiring
```

---

## Step 1: Viability Analysis

Before configuring anything, determine whether the asset is viable for market making.

### Minimum Requirements

| Criteria | Minimum | Ideal | Red Flag |
|----------|---------|-------|----------|
| Daily volume | >$1M | >$10M | <$500K |
| Typical spread | >5 bps | >10 bps | <3 bps (too competitive) |
| Max leverage | >=3x | >=10x | 1x (no leverage) |
| Binance pair | Preferred | Required for lead-lag | None (HIP-3 only) |

### Information Gathering

1. **Check Binance pair availability**:
   - Look up the asset in `src/market_maker/infra/binance_feed.rs`, function `resolve_binance_symbol()`
   - Known Binance-mapped assets: BTC, ETH, SOL, DOGE, AVAX, LINK, ARB, OP, SUI, APT, NEAR, ATOM, DOT, XRP, ADA, WIF, PEPE, BONK, and more
   - Hyperliquid-native tokens (HYPE, PURR, JEFF) return `None` -- no Binance equivalent

2. **Check Hyperliquid exchange info**:
   ```bash
   # List available assets and their metadata
   cargo run --release --bin market_maker -- --asset <ASSET> --dry-run
   # List HIP-3 DEXs
   cargo run --release --bin market_maker -- --list-dexs
   ```

3. **Determine asset type**:
   - **Validator perp**: Standard asset on Hyperliquid (BTC, ETH, SOL, etc.). Cross-margin supported, high leverage.
   - **HIP-3 builder-deployed**: DEX-specific token (HYPE on hyna, etc.). Isolated margin only, lower leverage, may have OI cap.
   - Check `AssetRuntimeConfig::from_asset_meta()` in `config/runtime.rs` -- HIP-3 detection happens at startup.

### Red Flags (Do Not Proceed)

- Volume < $500K/day: fills too infrequent for learning loops to converge
- No Binance pair AND low volume: both signals and fills will be poor
- Leverage 1x: cannot size positions meaningfully with limited capital
- Asset recently delisted or about to be delisted
- OI cap extremely low relative to intended capital deployment

### Decision

- If all minimum criteria are met, proceed to Step 2.
- If the asset is borderline, consider starting with paper trading only (Step 4) before committing capital.
- If red flags are present, do not onboard. Document the reason for future reference.

---

## Step 2: Config Generation

Configuration is driven by the `auto_derive()` function in `src/market_maker/config/auto_derive.rs`. It takes a single sizing input (`capital_usd`), a `SpreadProfile`, and `ExchangeContext` to produce self-consistent parameters.

### SpreadProfile Selection

Choose the profile based on asset characteristics. Defined in `src/market_maker/config/spread_profile.rs`:

| Profile | Target Spread | Gamma (risk aversion) | Sizing Fraction | Kappa Prior | Use For |
|---------|--------------|----------------------|-----------------|-------------|---------|
| `Default` | 40-50 bps | 0.3 | 20% of max_pos | 500-2500 | BTC, ETH, SOL, standard perps |
| `Hip3` | 15-25 bps | 0.15 | 30% of max_pos | 1500 | HYPE, PURR, HIP-3 DEX tokens |
| `Aggressive` | 10-20 bps | 0.10 | 40% of max_pos | 2000 | High-confidence assets (EXPERIMENTAL) |

**How to choose**:
- Standard validator perps with Binance pair: `Default`
- HIP-3 builder-deployed tokens: `Hip3`
- Only use `Aggressive` for assets where you have strong alpha conviction and want tight spreads

### Auto-Derive Pipeline

The `auto_derive()` function computes:

1. **max_position**: `min(capital_usd / mark_px, available_margin * max_leverage * 0.5 / mark_px)`
   - The 0.5 safety factor leaves 50% margin buffer for adverse moves
   - Both capital and margin constraints are enforced

2. **target_liquidity**: `max(max_position * sizing_fraction, min_order * 5)`
   - Ensures at least 5 levels of minimum order size
   - Capped at max_position

3. **risk_aversion (gamma)**: From SpreadProfile (0.3 / 0.15 / 0.10)
   - Controls spread width via GLFT: delta = (1/gamma) * ln(1 + gamma/kappa)
   - Lower gamma = tighter spreads = more fills but more adverse selection risk

4. **max_bps_diff**: `(fee_bps * 2.0).clamp(3, 15)`
   - Cold-start requoting threshold; runtime `DynamicReconcileConfig` refines from sigma

5. **Viability check**: `max_position >= min_order`
   - min_order = (min_notional * 1.15) / mark_px
   - If not viable, the diagnostic explains the minimum capital needed

### Example: Deriving Config for HYPE at $200

```
ExchangeContext:
  mark_px: 20.0, available_margin: 200.0, max_leverage: 5.0, fee_bps: 1.5

auto_derive(200.0, SpreadProfile::Hip3, &ctx):
  max_from_capital = 200 / 20 = 10.0 contracts
  max_from_margin  = 200 * 5 * 0.5 / 20 = 25.0 contracts
  max_position     = min(10.0, 25.0) = 10.0
  min_order        = (10.0 * 1.15) / 20 = 0.575
  viable           = 10.0 >= 0.575 = true
  risk_aversion    = 0.15 (Hip3)
  target_liquidity = max(10.0 * 0.30, 0.575 * 5) = max(3.0, 2.875) = 3.0
  max_bps_diff     = (1.5 * 2.0).clamp(3, 15) = 3
```

### CLI Flags

```bash
cargo run --release --bin market_maker -- \
  --asset <ASSET> \
  --capital-usd <AMOUNT> \
  --spread-profile <default|hip3|aggressive> \
  --dex <DEX_NAME>               # Only for HIP-3 assets (e.g., "hyna") \
  --initial-isolated-margin 1000 # Only for HIP-3 (default $1000) \
  --force-isolated               # Force isolated margin on validator perps
```

---

## Step 3: Key Config Decisions

After auto-derive produces base parameters, several critical decisions remain.

### Binance Signal Flags

**THIS IS THE MOST COMMON MISCONFIGURATION BUG.**

In `src/market_maker/strategy/signal_integration.rs`, two flags control Binance-dependent signals:

- `use_lead_lag: bool` -- Binance-to-Hyperliquid price lead-lag signal
- `use_cross_venue: bool` -- Binance order flow signal

**Rules**:
- If `resolve_binance_symbol(asset)` returns `Some(...)`: set both to `true`
- If `resolve_binance_symbol(asset)` returns `None`: **BOTH MUST BE `false`**

**What happens if you get this wrong**: The system enables staleness detection for signals that can never arrive. The `SignalIntegrator` applies a permanent 2x staleness multiplier to spreads because the Binance feed will never produce data. This was the root cause of HIP-3 live getting stuck in NoQuote after 1 fill (see MEMORY.md, 2026-02-09).

**Fix if misconfigured**: Call `disable_binance_signals()` on the `SignalIntegrator` (wired in `src/market_maker/mod.rs:624`). The startup code in `market_maker.rs` does this automatically when no Binance symbol is found.

### Kill Switch Thresholds

Scale with capital allocation. The kill switch monitors drawdown as a percentage of `account_value` (NOT peak PnL -- that was a bug, fixed in `state.rs`).

| Capital | Daily Drawdown Limit | Reasoning |
|---------|---------------------|-----------|
| <$500 | 10% ($50) | Small account, preserve capital |
| $500-$5K | 5% | Standard risk management |
| $5K-$50K | 3% | Significant capital at risk |
| >$50K | 2% | Conservative, protect large account |

The kill switch state persists across restarts via `KillSwitchCheckpoint` (24h expiry).

### HIP-3 Specific Configuration

If the asset is a HIP-3 builder-deployed token, additional considerations apply:

1. **Lower volume = slower learning**: Kappa and AS classifiers need fills to learn. HIP-3 tokens may take 48h+ to reach meaningful calibration.

2. **Wider natural spreads = more edge opportunity**: HIP-3 tokens often have 15-30 bps spreads, providing more room for market making profit.

3. **API quota may be limited**: The hyna DEX has been observed with only 7% API headroom, triggering inventory-forcing mode. Monitor `quota_headroom` in logs.

4. **Risk model blend**: Set `risk_model_blend = 1.0` for HIP-3 tokens. This uses the calibrated risk model (log-additive gamma) instead of the data-driven model, preventing gamma explosion in thin markets.

5. **Isolated margin only**: HIP-3 assets use isolated margin (`is_cross = false`). The `initial_isolated_margin` flag controls the initial allocation (default $1000).

6. **OI cap**: HIP-3 tokens may have open interest caps. Check `AssetRuntimeConfig.oi_cap_usd` (unlimited for validator perps, finite for HIP-3).

7. **Collateral**: HIP-3 DEXs may use different stablecoins (USDE, USDH, USDC). This is resolved at startup from spot metadata.

8. **Monopolist pricing**: `RiskConfig::hip3()` enables `use_monopolist_pricing` for tokens where the MM may be the sole liquidity provider. This adds a markup based on taker elasticity.

### RL Configuration

- `rl_enabled`: Should be `true` for all assets
- The RL model (`RLEdgeModel`) participates as an ensemble member with a minimum 5% floor weight
- For new assets, Q-values start cold and only seed from prior states (states with `n=0`)
- Do NOT import Q-table from a different asset -- the state space is asset-specific

### Session Position Ramp

Configured in `AssetRuntimeConfig` via `SessionPositionRamp`:
- Default: 30-minute sqrt ramp, starting at 10% capacity
- For HIP-3 tokens: consider extending to 60 minutes due to slower calibration
- The ramp prevents full position exposure before models have calibrated

---

## Step 4: Paper Validation

Before deploying live capital, validate the configuration in paper trading mode.

**References the `/paper-trading` workflow for detailed setup instructions.**

### Setup

```bash
# Build paper trader
cargo build --release --bin paper_trader

# Run paper trader (user executes manually)
RUST_LOG=info ./target/release/paper_trader \
  --asset <ASSET> \
  --capital-usd <AMOUNT> \
  --spread-profile <PROFILE> \
  --dex <DEX_NAME>  # if HIP-3
```

### Duration

| Asset Type | Minimum Duration | Recommended |
|------------|-----------------|-------------|
| Major (BTC, ETH) | 24h | 48h |
| Mid-cap with Binance | 24h | 48h |
| HIP-3 token | 48h | 72h |
| Low volume (<$5M/day) | 72h | 1 week |

### Validation Criteria

All criteria must be met before proceeding to live deployment:

| Metric | Minimum | Target | Fail |
|--------|---------|--------|------|
| Edge (bps) | > 0 bps | > 1.5 bps | < 0 bps |
| Fill rate | > 10 fills/hr | > 30 fills/hr | < 5 fills/hr |
| Adverse selection rate | < 40% | < 25% | > 50% |
| Sharpe ratio | > 0 | > 1.0 | < 0 |
| Kill switch triggers | 0 | 0 | Any |
| Spread vs market | < 2x typical | ~1x typical | > 3x typical |
| Calibration progress | > 50% | > 80% | < 20% |

### Three Critical Feedback Loops

Verify these are all working within the first 30 minutes of paper trading:

1. **Kappa from own fills**: `estimator.on_own_fill()` called on simulated fills. Check: `kappa_confidence` increasing from 0.

2. **AS outcome feedback**: `PendingFillOutcome` queue with 5s markout. Check: `pre_fill_classifier` outcome counts > 0.

3. **Calibration controller**: `CalibrationController` tracking `(as_progress + kappa_progress) / 2`. Check: `calibration_progress` > 0.

If any loop is broken, the system will not learn and paper results are meaningless. See `/paper-trading` workflow for debugging.

### Interpreting Paper Results

- **Positive PnL in quiet markets**: Strategy captures spread. Good baseline.
- **Negative PnL in volatile markets**: Expected, but magnitude matters. Should be small relative to spread capture.
- **High AS rate (>40%)**: Pre-fill toxicity classifier needs tuning, or asset has inherently high information asymmetry.
- **Low fill rate**: Kappa estimate may be too low (spreads too wide), or market is too thin.
- **Spread much wider than market**: Check if staleness multiplier is active (Binance signals misconfigured?).

---

## Step 5: Go Live

Transition from paper to live deployment.

### Pre-Live Checklist

- [ ] Paper trading ran for minimum duration (Step 4)
- [ ] All validation criteria met
- [ ] Kill switch thresholds configured appropriately for capital size
- [ ] Binance signal flags correct (`use_lead_lag`, `use_cross_venue`)
- [ ] Spread profile matches asset type
- [ ] For HIP-3: DEX name, isolated margin, collateral verified
- [ ] Account has sufficient funds deposited on Hyperliquid
- [ ] For HIP-3: collateral token deposited (USDE/USDH/USDC as appropriate)

### Capital Ramp Strategy

Do NOT deploy full target capital on day one. Use a graduated ramp:

| Phase | Capital | Duration | Criteria to Advance |
|-------|---------|----------|-------------------|
| Phase 1 | 25% of target | Days 1-3 | Edge > 0, no kill switch, fills flowing |
| Phase 2 | 50% of target | Days 4-7 | Edge > 1 bps, Sharpe > 0, AS < 35% |
| Phase 3 | 75% of target | Week 2 | Consistent positive PnL, cal > 60% |
| Phase 4 | 100% of target | Week 3+ | Stable metrics across regimes |

### Deployment Command

```bash
# Live deployment (user executes manually)
RUST_LOG=info ./target/release/market_maker \
  --asset <ASSET> \
  --capital-usd <PHASE_1_AMOUNT> \
  --spread-profile <PROFILE> \
  --dex <DEX_NAME>  # if HIP-3
```

### Paper-to-Live Gap

Expect a paper-to-live gap of 2-5 bps. Sources of leakage:
- **Queue position bias (~2 bps)**: Paper trader assumes favorable queue position
- **AS under-widening (~1 bps)**: Real adverse selection is harder to detect
- **Rate limit staleness (~0.8 bps)**: API quota constraints cause stale quotes
- **Warmup defaults (~0.5 bps)**: Cold-start models are less accurate live

The gap should narrow over the first week as models calibrate to live conditions.

---

## Step 6: Post-Onboarding Monitoring

### Day 1: Continuous Monitoring

- Monitor the dashboard continuously for the first trading session
- Verify all learning loops are active (kappa updates, AS outcomes, calibration progress)
- Watch for kill switch activations -- any trigger means something is misconfigured
- Check spread is within 2x of market typical
- Verify position stays within max_position bounds
- For HIP-3: monitor API quota headroom (should stay >10%)

### Week 1: Daily Review

- Review calibration metrics daily:
  - Brier Score < 0.25 (predictions are calibrated)
  - Information Ratio > 1.0 (signals add value)
  - Edge > 0 bps (positive expected value)
- Adjust gamma/kill switch thresholds based on actual performance
- If edge is consistently negative: widen spreads (increase gamma), check AS classifier
- If fill rate is too low: check kappa estimate, may need tighter spreads
- Scale capital per the ramp schedule (Step 5)

### Month 1: Scaling Decision

- If metrics are consistently positive across different market regimes:
  - Scale to full target capital
  - Consider adding to multi-asset pool (Step 7)
  - Monitor signal MI for decay (edge decays over time)
- If metrics are mixed:
  - Hold at current capital level
  - Investigate regime-specific failures (see `/debug-pnl` workflow)
  - Consider parameter adjustments before scaling
- If metrics are consistently negative:
  - Reduce capital or remove the asset
  - Document what went wrong for future reference
  - Check if market conditions changed (volume dried up, new competitors)

---

## Step 7: Multi-Asset Considerations

When adding an asset to an existing multi-asset portfolio, additional coordination is required.

### Capital Allocation

The `AssetAllocator` in `src/market_maker/multi/allocator.rs` uses inverse-volatility weighting:

- Lower volatility assets get more order levels (higher capital efficiency)
- Concentration cap prevents any single asset from consuming >30% of total orders
- Minimum 5 levels per asset guaranteed
- Total order limit: 1000 (default), up to 5000 with volume

### Configuration

```rust
// In MultiAssetConfig (src/market_maker/config/multi_asset.rs)
MultiAssetConfig {
    assets: vec![
        AssetSpec::new("BTC", None),                           // Validator perp
        AssetSpec::new("ETH", None),                           // Validator perp
        AssetSpec::hip3("HYPE", "hyna"),                       // HIP-3 with DEX
        AssetSpec::new("SOL", None).with_spread_profile(SpreadProfile::Default),
    ],
    total_order_limit: 1000,
    min_levels_per_asset: 5,
    max_levels_per_asset: 25,
    rebalance_interval_secs: 300,  // 5 minutes
    max_concentration_pct: 0.30,   // 30% cap per asset
    default_spread_profile: SpreadProfile::Default,
}
```

### Shared Margin Pool

The `SharedMarginPool` in `src/market_maker/multi/margin_pool.rs` enforces:

- **Total utilization cap**: 80% of account value (leaves buffer for adverse moves)
- **Per-asset concentration**: No single asset > 30% of total capital
- **Reduce-only mode**: Triggered at 80% utilization
- **Quoting capacity**: Computed per-asset respecting both global and per-asset limits

### Cross-Asset Risk

When adding a correlated asset (e.g., adding ETH when already trading BTC):
- Net directional exposure is higher than individual positions suggest
- Kill switch should consider portfolio-level drawdown, not just per-asset
- High-vol regimes affect all correlated assets simultaneously -- may need wider spreads across the board

When adding an uncorrelated asset (e.g., adding HYPE when trading BTC):
- Diversification benefit -- portfolio risk lower than sum of parts
- Independent kill switch thresholds per asset are appropriate
- API quota is shared -- more assets = less headroom per asset

### Adding to Existing Portfolio

1. Start the new asset in paper trading while existing assets continue live
2. Verify the new asset does not degrade existing assets (check for API quota pressure)
3. Deploy live at 25% capital, monitor portfolio-level metrics
4. Let the allocator rebalance naturally (5-minute interval)
5. Monitor concentration -- new asset should not crowd out existing profitable ones

---

## Quick Reference: Asset Onboarding Decision Tree

```
Is the asset on Hyperliquid?
  No  -> Cannot onboard
  Yes -> Is it a HIP-3 token?
    Yes -> Use SpreadProfile::Hip3
           Set --dex <name>
           Set use_lead_lag=false, use_cross_venue=false
           Need collateral in correct token (USDE/USDH/USDC)
    No  -> Does it have a Binance pair? (check resolve_binance_symbol())
      Yes -> Use SpreadProfile::Default
             Set use_lead_lag=true, use_cross_venue=true
      No  -> Use SpreadProfile::Default
             Set use_lead_lag=false, use_cross_venue=false
             (Limited alpha without lead-lag signal)

Is volume > $1M/day?
  No  -> Do not onboard (fills too infrequent)
  Yes -> Is capital >= minimum viable? (check auto_derive viability)
    No  -> Increase capital or pick different asset
    Yes -> Proceed to paper validation (Step 4)
```

## Checklist

- [ ] Viability analysis complete (volume, spread, leverage, Binance pair)
- [ ] SpreadProfile selected (Default / Hip3 / Aggressive)
- [ ] auto_derive produces viable config (max_position >= min_order)
- [ ] Binance signal flags correct (lead-lag, cross-venue)
- [ ] Kill switch thresholds scaled to capital
- [ ] HIP-3 specifics handled (if applicable): DEX, margin, collateral, risk_model_blend
- [ ] Paper trading completed for minimum duration
- [ ] All validation criteria met (edge, fill rate, AS rate, Sharpe)
- [ ] All three feedback loops verified (kappa, AS, calibration)
- [ ] Live deployment at 25% capital
- [ ] Capital ramp schedule planned (25% -> 50% -> 75% -> 100%)
- [ ] Day 1 continuous monitoring planned
- [ ] Week 1 daily review scheduled
- [ ] Multi-asset coordination configured (if adding to portfolio)

See also: `references/asset-characteristics.md` for per-asset trading characteristics.
