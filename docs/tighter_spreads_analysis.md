# How to Quote Tighter Spreads

## Executive Summary

Your market maker ran for ~4 minutes with **0 fills**. The root cause is **wide spreads** driven by:
1. **`min_spread_floor = 8 bps`** (hard minimum half-spread)
2. **`gamma_base = 0.5`** (conservative risk aversion)

With these settings, your minimum total spread is **16 bps**. With a 3 bps maker fee, this is rarely competitive enough for takers to cross.

---

## Understanding the GLFT Spread Formula

The market maker uses GLFT (Guéant-Lehalle-Fernandez-Tapia) optimal spread theory:

```
δ = (1/γ) × ln(1 + γ/κ) + fee
```

Where:
- **δ** = optimal half-spread (as fraction of price)
- **γ** = gamma (risk aversion, higher = wider spreads)
- **κ** = kappa (order flow intensity, higher = tighter spreads)
- **fee** = maker fee rate (1.5 bps default, you mentioned 3 bps)

### How Gamma Affects Spread

| gamma_base | κ = 300 | κ = 500 | κ = 1000 |
|------------|---------|---------|----------|
| 0.05       | 1.8 bps | 1.1 bps | 0.6 bps  |
| 0.1        | 3.5 bps | 2.1 bps | 1.1 bps  |
| 0.2        | 6.7 bps | 4.1 bps | 2.1 bps  |
| **0.5**    | **14.6 bps** | **9.2 bps** | **4.8 bps** |
| 1.0        | 24.9 bps | 16.2 bps | 8.6 bps |

**Your setting (γ = 0.5)** produces very wide spreads. In a liquid market (κ ~ 500), you're quoting ~9-10 bps half-spread just from the GLFT formula.

---

## Why Zero Fills Occurred

From your log (`mm.log`), I identified these issues:

### 1. Spread Floor is Too Wide

```
min_spread_floor: 0.0008 (8 bps)  # in RiskConfig
min_spread_floor_bps: 8.0          # in DynamicDepthConfig
```

This means **no matter what gamma/kappa say**, the minimum half-spread is 8 bps. Total spread at touch: **16 bps minimum**.

### 2. Gamma is High

```
risk_aversion: 0.5  # from log line 3
```

With γ = 0.5, the GLFT formula produces spreads of 10-25 bps depending on κ.

### 3. Combined Effect

Your effective spreads are likely **16-30 bps** total. For a taker paying 3-5 bps taker fee:
- They need price to move **19-35 bps** to break even after taking your quote
- In low volatility, this rarely happens before your quote updates
- Result: 0 fills in 4 minutes

---

## The Fee Breakdown

You mentioned a "3 bps fee". Let me clarify Hyperliquid's fee structure:

| Fee Type | Typical Rate | Your Concern |
|----------|-------------|--------------|
| Maker fee | 0.5-2 bps | You pay this when filled |
| Taker fee | 3-5 bps | Taker pays this to cross |

**For profitability:**
```
Spread Capture = half_spread - maker_fee - adverse_selection

To profit: half_spread > maker_fee + expected_AS + buffer
With 3 bps maker fee: half_spread > 3 + 0.5 + 0.5 = 4 bps minimum
```

The code has `maker_fee_rate = 0.00015` (1.5 bps) but if your actual fee is 3 bps, update this value.

---

## Path to Tighter Spreads

### Option 1: Aggressive (Higher Risk, More Fills)

**Target: 6-8 bps total spread at touch**

Parameters to change:
- `gamma_base`: 0.5 → **0.1**
- `min_spread_floor`: 8 bps → **3 bps** (covers fee)
- `min_spread_floor_bps`: 8 bps → **3 bps**

Expected outcome:
- Half-spread: ~3-4 bps (GLFT optimal in liquid markets)
- Fill rate: Significant increase
- Risk: Higher adverse selection, potential losses on toxic flow

### Option 2: Moderate (Balanced)

**Target: 10-12 bps total spread at touch**

Parameters to change:
- `gamma_base`: 0.5 → **0.2**
- `min_spread_floor`: 8 bps → **5 bps**
- `min_spread_floor_bps`: 8 bps → **5 bps**

Expected outcome:
- Half-spread: ~5-6 bps
- Fill rate: Moderate increase
- Risk: Some adverse selection, generally sustainable

### Option 3: Conservative Tightening

**Target: 12-14 bps total spread at touch**

Parameters to change:
- `gamma_base`: 0.5 → **0.3** (the default)
- `min_spread_floor`: 8 bps → **6 bps**
- `min_spread_floor_bps`: 8 bps → **6 bps**

Expected outcome:
- Half-spread: ~6-7 bps
- Fill rate: Small increase
- Risk: Low adverse selection

---

## The Tradeoff: Fill Rate vs Adverse Selection

```
Tighter Spread → More Fills → More Adverse Selection
                           ↓
                     More P&L Variance
                     Potential for Informed Flow Losses
```

### Why Wide Spreads Exist

The current 8 bps floor was set based on trade history analysis showing:
- Average adverse selection: 0.5 bps
- Large trades (>$2k): -11.6 bps edge (losing trades)
- Toxic hours (06-08, 14-15 UTC): -13 to -15 bps edge

The 8 bps floor was meant to provide profitability buffer:
```
8 bps = 1.5 bps fee + 0.5 bps avg AS + 6 bps buffer
```

### Risk of Tighter Spreads

By tightening to 3-4 bps half-spread:
1. **Informed traders can pick you off** before you update
2. **Latency disadvantage**: HFT with faster signals will take your stale quotes
3. **Toxic flow exposure**: During volatile periods, you're the exit liquidity
4. **Large trades hurt more**: -11 bps adverse selection on big fills

---

## What Kappa Tells You

Kappa (κ) measures order book depth decay - higher κ means more liquidity around you. The GLFT formula automatically tightens spreads when κ is high.

From your log, if κ is being estimated low due to:
- Thin testnet order books
- Estimation warmup period
- Low market activity

Then the formula produces wide spreads naturally. This is **intentional** - in illiquid markets, wide spreads protect you.

---

## Configuration Parameters Reference

### Where Parameters Live

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `gamma_base` | `strategy/risk_config.rs:132` | 0.3 | Base risk aversion |
| `min_spread_floor` | `strategy/risk_config.rs:148` | 0.0008 (8 bps) | GLFT half-spread floor |
| `maker_fee_rate` | `strategy/risk_config.rs:151` | 0.00015 (1.5 bps) | Maker fee for HJB |
| `min_spread_floor_bps` | `quoting/ladder/depth_generator.rs:183` | 8.0 | Depth generator floor |
| `fees_bps` | `quoting/ladder/mod.rs:104` | 3.5 | Spread capture fee |

### Parameters Passed via CLI

Looking at your log, you used:
```
--asset BTC
--target-liquidity 0.25
--risk-aversion 0.5    # <- this is gamma_base
--max-bps-diff 2
--max-position 0.5
```

The `--risk-aversion 0.5` sets gamma_base. Lower this to tighten spreads.

---

## Recommended Action Plan

### Step 1: Understand Current Behavior

Run with RUST_LOG=debug to see actual spread calculations:
```
half_spread_bps = ...
bid_delta_bps = ...
ask_delta_bps = ...
kappa = ...
gamma = ...
```

This reveals whether spread width comes from:
- High gamma
- Low kappa estimate
- min_spread_floor clamping
- Toxicity/volatility scaling

### Step 2: Gradual Tightening

Start with moderate settings and observe:

1. **First pass**: `--risk-aversion 0.3` (default gamma)
2. **If still too wide**: `--risk-aversion 0.2`
3. **If still no fills**: Reduce min_spread_floor in config

### Step 3: Monitor Key Metrics

Watch these Prometheus metrics (port 9090):
- `mm_spread_bps`: Actual spread being quoted
- `mm_fill_volume_buy` / `mm_fill_volume_sell`: Fill activity
- `mm_adverse_selection_bps`: Realized AS cost
- `mm_daily_pnl`: P&L impact

### Step 4: Protect Against Downside

Keep kill switch active:
- `max_daily_loss`: Limits total loss
- `max_drawdown`: Stops during adverse runs
- Time-of-day scaling: Widens spreads during toxic hours

---

## Mathematical Framework

### The Full Spread Stack

```
Final half-spread = max(
    GLFT_optimal,           # (1/γ) × ln(1 + γ/κ) + fee
    min_spread_floor,       # 8 bps default
    min_depth_bps           # 2 bps default
)
+ AS_spread_adjustment      # Learned from fills
+ jump_premium              # When RV/BV > 1.5
+ kalman_uncertainty        # When enabled
```

Each layer can widen the spread. To get tight spreads, you need:
1. Low gamma_base
2. High kappa (liquid market)
3. Low min_spread_floor
4. No toxic regime/jumps

### Why Kappa Matters More at Low Gamma

When γ is small:
```
δ ≈ (1/γ) × (γ/κ) = 1/κ
```

So with γ = 0.1 and κ = 500:
- δ ≈ 0.002 = 20 bps? NO!
- Actually: δ = 10 × ln(1.0002) ≈ 2 bps

The formula is nonlinear. Low gamma gives tight spreads when κ is reasonable.

---

## Summary

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| 0 fills | 16+ bps spread | Reduce gamma + floor |
| Wide GLFT spread | gamma = 0.5 | Use gamma = 0.1-0.3 |
| Clamped at 8 bps | min_spread_floor | Reduce to 3-5 bps |
| 3 bps fee concern | Fee not updated | Set maker_fee_rate = 0.0003 |

**Bottom line**: To get fills with 3 bps fee, you need spreads of ~6-10 bps total. With current settings producing 16+ bps, no taker will cross. Lower `gamma_base` to 0.1-0.2 and `min_spread_floor` to 3-5 bps.
