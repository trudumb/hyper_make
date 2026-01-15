# Heatmap-Quoting Overlay Analysis: Finding Hidden Alpha

## The Core Idea

Overlay our quoting behavior on an order book heatmap to answer: **Where are we quoting relative to where liquidity exists, appears, and disappears?**

Standard market making literature treats the order book as a static snapshot. The heatmap reveals it as a living organism with migration patterns, territorial behavior, and predator-prey dynamics. Our quotes are participants in this ecosystem—are we predator or prey?

---

## What the Heatmap Reveals That Snapshots Don't

### 1. Liquidity Migration Patterns

The horizontal bands in the heatmap aren't random. They represent:
- **Anchoring behavior**: Traders cluster quotes at round numbers, previous highs/lows, VWAP levels
- **Defensive repositioning**: When price approaches, liquidity retreats (the "ghost" effect)
- **Accumulation zones**: Where liquidity persistently rebuilds after being consumed

**Alpha opportunity**: If liquidity consistently migrates in predictable patterns relative to price, we can position quotes to be *where liquidity is going*, not where it currently is.

### 2. The Liquidity "Shadow"

Notice how liquidity often appears asymmetrically—more below price during uptrends, more above during downtrends. This is the market's collective positioning revealing directional bias.

**Question for our system**: Are our quotes respecting or ignoring this shadow? If we're quoting symmetrically into an asymmetric book, we're providing information about our lack of directional view.

### 3. Disappearing Walls = Information Events

When a large liquidity cluster suddenly vanishes (horizontal line disappears), one of two things happened:
1. It got filled (aggressive flow absorbed it)
2. It got pulled (the quoter saw something and retreated)

Both are information. (2) is especially valuable—someone with capital at risk decided the risk/reward changed.

**Alpha opportunity**: If we can detect wall-pulling faster than our quote update cycle, we're exposed. If we can detect it *before* it happens (via cross-exchange signals or order flow), we have edge.

---

## Overlay Analysis: Questions to Answer

### A. Quote Positioning Relative to Walls

Visualizing our quotes against the heatmap reveals:

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Quotes consistently inside walls | Protected from adverse selection, but poor queue position | Acceptable if fill rate is adequate |
| Quotes at same level as walls | Competing with large size, likely back of queue | Problematic—getting filled means wall pulled |
| Quotes outside walls | First to get hit, maximum adverse selection | Only acceptable if edge on direction |
| Quotes "chase" wall movements | Reactive, providing information | Need to lead, not follow |

### B. Fill Location Analysis

Map where our fills occur on the heatmap:

**Hypothesis**: Fills that occur when our quote is the *last remaining liquidity* at a level (wall just pulled) are toxic. Fills that occur when we're *shielded* by remaining liquidity are benign.

This is a refinement of basic adverse selection—not just "did price move against us" but "what was the microstructure context of the fill?"

### C. Quote Persistence During Liquidity Events

When the heatmap shows a "flush" (rapid liquidity disappearance across multiple levels):

1. How long did our quotes remain?
2. Did we get filled during the flush?
3. What was the P&L of those fills?

**Hypothesis**: Quotes that survive flushes longer than the median quote have higher adverse selection. We should be *faster* at pulling during flushes, not slower.

---

## Novel Alpha Sources from This Analysis

### 1. "Liquidity Regime" Classification

The heatmap implicitly shows different liquidity regimes:

- **Thick book**: Dense horizontal bands, liquidity at many levels
- **Thin book**: Sparse bands, large gaps between liquidity
- **Migrating book**: Liquidity bands shifting directionally
- **Collapsing book**: Liquidity disappearing from multiple levels (cascade precursor)

Current approach: Use HMM on price/volatility features.
**Enhancement**: Use heatmap features directly—liquidity entropy, band density, migration velocity.

These are *leading* indicators vs price-based features which are lagging.

### 2. "Relative Queue Position" Inference

We can't observe our actual queue position. But we can infer it:

If total visible liquidity at our quote level is X, and our size is Y:
- If X is large, we're likely mid-to-back of queue
- If X is small (close to Y), we're likely front of queue

**Heatmap enhancement**: Track how X evolves over time at our quote level. If X is shrinking while price is stable, others are pulling—we're moving up in queue OR about to be exposed.

### 3. "Spoofing Detection" for Quote Protection

Much heatmap liquidity is fake (will pull before being hit). Characteristics:
- Appears/disappears rapidly (high flicker rate)
- Placed at psychologically significant levels
- Correlated with aggressive flow on opposite side

**Alpha opportunity**: If we can classify visible liquidity as "real" vs "fake," we can:
- Ignore fake walls when positioning
- Treat fake wall *disappearance* as noise, not signal
- Avoid being the patsy when a spoofer triggers our quote update

### 4. "Footprint Minimization"

If our quotes are a significant fraction of visible liquidity at a level, we're providing information to:
- Other market makers (who can now estimate our inventory)
- Informed traders (who know where liquidity exists)

**Heatmap overlay reveals**: How "visible" are we? Are we a distinct horizontal line, or noise in a thick band?

**Trade-off**: Being visible means better queue position but higher information leakage. Optimal strategy may be to match quote size to ambient liquidity.

### 5. "Cross-Exchange Heatmap Arbitrage"

If we overlay Binance heatmap and Hyperliquid heatmap:

**Hypothesis**: Liquidity patterns on Binance lead Hyperliquid by the same lag as price. Wall formation on Binance predicts wall formation on Hyperliquid.

**Alpha opportunity**: Position quotes on Hyperliquid based on where liquidity is *forming* on Binance, not where it currently exists on Hyperliquid.

This is lead-lag applied to microstructure, not just price.

---

## Implementation Path

### Phase 1: Visualization Infrastructure

Build tooling to:
1. Reconstruct historical heatmaps from order book snapshots
2. Overlay our quote history on the heatmap
3. Mark fills with outcome labels (profitable/adverse)

**Output**: Visual debugging tool for microstructure analysis.

### Phase 2: Feature Extraction

Derive quantitative features from heatmap:
- Liquidity asymmetry ratio (above vs below mid)
- Band density (levels with >X size within Y bps)
- Migration velocity (how fast is the liquidity centroid moving)
- Flicker rate (quote appearance/disappearance frequency by level)
- Our relative visibility (our size / total size at level)

### Phase 3: Conditional Analysis

Segment all metrics by heatmap features:
- Fill adversity conditioned on liquidity regime
- Optimal spread conditioned on band density
- Quote update latency requirements conditioned on flicker rate

### Phase 4: Predictive Models

Use heatmap features as inputs to:
- Adverse selection classifier (augment with "was wall present at fill?")
- Fill intensity model (condition kappa on local liquidity density)
- Regime HMM (add liquidity features to observation model)

---

## Key Research Questions

1. **Is wall-relative position predictive of fill toxicity?**
   - Measure: Adverse selection of fills inside vs outside walls
   - Expectation: Inside-wall fills are less toxic

2. **Does liquidity migration velocity predict short-term returns?**
   - Measure: Correlation between liquidity centroid movement and 1-minute returns
   - Expectation: Liquidity leads price (informed liquidity providers)

3. **Can we detect cascade onset from heatmap features before price moves?**
   - Measure: Lead time of "liquidity collapse" signal vs price drop
   - Expectation: 100-500ms lead time if signal exists

4. **Is cross-exchange heatmap correlation exploitable?**
   - Measure: Predictive power of Binance liquidity features for Hyperliquid fills
   - Expectation: Some signal, decaying as market matures

5. **What is the information cost of our visibility?**
   - Measure: Adverse selection when we're >X% of level vs <X%
   - Expectation: Higher visibility = higher adverse selection (others front-run us)

---

## Why This Isn't in Public Literature

Academic market microstructure focuses on:
- Theoretical models (Kyle, Glosten-Milgrom) with stylized order flow
- Empirical studies on executed trades (TAQ data)
- Optimal execution (minimizing impact for a given order)

What's missing:
- **Order book dynamics over time** (heatmap view is proprietary data)
- **Market maker perspective** (academics study takers, not makers)
- **Cross-venue microstructure** (Binance→Hyperliquid channel is crypto-specific)
- **Real-time feature engineering** (academics analyze post-hoc, we need real-time)

The heatmap is a **practitioner's tool**, not an academic one. The alpha is in the details of implementation and the specific features that matter for *our* venue.

---

## Part 2: Complete Visualization Infrastructure

### Current State Assessment

**What we have:**
- Single-file React dashboard (`mm-dashboard-fixed.html`) with Recharts
- Rust WebSocket server pushing real-time updates
- Dashboard aggregator maintaining rolling state
- Basic visualizations: PnL, regime probabilities, order book snapshot, spread distribution

**What's missing for serious analysis:**
- No historical replay capability (can only see "now")
- No quote overlay on order book heatmap
- No cross-exchange visualization
- No fill forensics (why did this fill lose money?)
- No time-synchronized multi-panel analysis
- No annotation/labeling for research
- Limited interactivity (no zoom, pan, selection)

### The Vision: Three-Mode Dashboard

The visualization system needs to support three distinct use cases:

```
┌─────────────────────────────────────────────────────────────┐
│                    VISUALIZATION MODES                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│   LIVE MODE     │  REPLAY MODE    │    RESEARCH MODE        │
│                 │                 │                         │
│ Real-time       │ Historical      │ Multi-day analysis      │
│ monitoring      │ debugging       │ and hypothesis testing  │
│                 │                 │                         │
│ "What's         │ "What happened  │ "Is this pattern        │
│ happening now?" │ at 14:32:05?"   │ statistically real?"    │
└─────────────────┴─────────────────┴─────────────────────────┘
```

---

### Mode 1: Live Dashboard (Enhanced)

**Purpose:** Real-time situational awareness during trading.

**Current capabilities:** Basic metrics, regime display, PnL tracking.

**Needed enhancements:**

#### 1.1 Heatmap with Quote Overlay (Core Feature)

```
┌────────────────────────────────────────────────────────────────┐
│  ETH/USDT Order Book Heatmap - Hyperliquid        [Live] [5m] │
├────────────────────────────────────────────────────────────────┤
│ $1590 ┤░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓░░░░░░░░░░░░░░░░░░░░░│
│       │                              ▲ Wall detected           │
│ $1585 ┤░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│       │         ╭─────── Mid price (white line)                │
│ $1580 ┤░░░░░░░░░░░░░░░░░█████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│       │                 ▲ Our ask quote (green marker)         │
│ $1575 ┤░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│       │         ╰─────── Our bid quote (red marker)            │
│ $1570 ┤░░░░░░░░░░▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│       │          ▲ Support zone (persistent liquidity)         │
├───────┼────────────────────────────────────────────────────────┤
│       │ -5m        -4m        -3m        -2m        -1m    now │
└───────┴────────────────────────────────────────────────────────┘
  Legend: ░ Low liquidity  ▓ Medium  █ High  ● Fill event
```

**Key elements:**
- Order book depth as color intensity (log scale for visibility)
- Mid price as continuous white line
- Our bid/ask quotes as colored markers tracking with time
- Fill events as circles (green = profitable, red = adverse)
- Wall detection annotations (auto-detected large resting orders)
- Liquidity regime indicator (thick/thin/migrating/collapsing)

#### 1.2 Cross-Exchange Panel

```
┌─────────────────────────────────────────────────────────────┐
│  Lead-Lag Monitor                              [Binance→HL] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Binance Mid ────────────────────────────────╮              │
│                                               │ 127ms lag   │
│  HL Mid      ─────────────────────────────────╯             │
│                                                             │
│  Current lag: 127ms  |  5min avg: 142ms  |  Decay: -2.1%/wk │
│                                                             │
│  [Binance Book Imbalance]  ████████░░  +0.23 (bid heavy)    │
│  [HL Book Imbalance]       ██████░░░░  +0.15 (bid heavy)    │
│                                                             │
│  Signal: Binance imbalance leads by ~89ms                   │
└─────────────────────────────────────────────────────────────┘
```

#### 1.3 Fill Forensics Panel

Real-time annotation of every fill with microstructure context:

```
┌─────────────────────────────────────────────────────────────┐
│  Recent Fills                                    [Last 50]  │
├─────────────────────────────────────────────────────────────┤
│  14:32:05.127  BUY  0.5 ETH @ 1574.25                       │
│    ├─ P&L: -$1.23 (adverse)                                 │
│    ├─ Wall context: Inside wall (wall at 1574.00, 12 ETH)   │
│    ├─ Book state: Thin (density 0.23)                       │
│    ├─ Regime: Trending (p=0.67)                             │
│    ├─ Binance delta: +$0.45 in prior 100ms                  │
│    └─ Classification: Informed flow (high confidence)       │
│                                                             │
│  14:32:04.892  SELL 0.3 ETH @ 1575.10                       │
│    ├─ P&L: +$0.87 (favorable)                               │
│    ├─ Wall context: Outside wall (nearest at 1576.00)       │
│    ├─ Book state: Thick (density 0.71)                      │
│    └─ Classification: Noise flow (medium confidence)        │
└─────────────────────────────────────────────────────────────┘
```

#### 1.4 Regime & Model Health

```
┌─────────────────────────────────────────────────────────────┐
│  Model Health Monitor                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Regime Probabilities (stacked area, last 30 min)           │
│  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░ │
│  Quiet ██  Trending ██  Volatile ██  Cascade ██             │
│                                                             │
│  Calibration (rolling 1-hour):                              │
│    Fill Probability    Brier: 0.18  IR: 1.34  [OK]          │
│    Adverse Selection   Brier: 0.21  IR: 1.12  [OK]          │
│    Regime HMM          Brier: 0.15  IR: 1.56  [OK]          │
│                                                             │
│  Alerts:                                                    │
│    ⚠ Lead-lag R² dropped 15% vs 7-day avg                   │
│    ⚠ Kappa estimate uncertainty high (σ/μ > 0.5)            │
└─────────────────────────────────────────────────────────────┘
```

---

### Mode 2: Replay Dashboard

**Purpose:** Post-hoc debugging of specific incidents.

**Use case:** "We lost $500 between 14:30 and 14:35. What happened?"

#### 2.1 Time-Synchronized Multi-Panel View

All panels scrub together with a single timeline control:

```
┌─────────────────────────────────────────────────────────────┐
│  Timeline: [|◄] [◄◄] [►||] [►►] [►|]   14:32:05.127        │
│  ══════════════════════●══════════════════════════════════  │
│            14:30      14:32      14:34      14:36           │
└─────────────────────────────────────────────────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Heatmap     │ │  Cross-Exch   │ │   PnL Curve   │
│   + Quotes    │ │  Comparison   │ │   + Fills     │
│               │ │               │ │               │
│  [synced to   │ │  [synced to   │ │  [synced to   │
│   timeline]   │ │   timeline]   │ │   timeline]   │
└───────────────┘ └───────────────┘ └───────────────┘
```

#### 2.2 Event Markers & Annotations

```
Timeline events (clickable to jump):
  ▼ 14:30:12  Regime shift: Quiet → Trending
  ▼ 14:31:45  Large wall appeared at 1572.00 (25 ETH)
  ▼ 14:32:01  Wall pulled (1572.00)
  ▼ 14:32:05  Fill: adverse selection detected
  ▼ 14:32:08  Cascade onset (OI drop > 2%)
  ▼ 14:33:15  Regime shift: Trending → Cascade
  ▼ 14:34:22  Spread widened to 15 bps (defensive)
```

#### 2.3 Counterfactual Analysis

"What if we had used different parameters?"

```
┌─────────────────────────────────────────────────────────────┐
│  Counterfactual: Spread Sensitivity                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Actual spread: 5 bps          P&L: -$523                   │
│  Counterfactual: 8 bps         P&L: -$312  (saved $211)     │
│  Counterfactual: 10 bps        P&L: -$189  (saved $334)     │
│  Counterfactual: 3 bps         P&L: -$891  (lost $368)      │
│                                                             │
│  [Chart showing fill probability vs spread at this regime]  │
│                                                             │
│  Insight: Spread was too tight for cascade regime.          │
│  Recommendation: Gamma floor of 8 bps in cascade regime.    │
└─────────────────────────────────────────────────────────────┘
```

---

### Mode 3: Research Dashboard

**Purpose:** Multi-day statistical analysis and hypothesis testing.

**Use case:** "Is wall-relative position actually predictive of adverse selection?"

#### 3.1 Cohort Analysis

```
┌─────────────────────────────────────────────────────────────┐
│  Cohort: Fills by Wall-Relative Position                    │
│  Date range: 2026-01-01 to 2026-01-14  |  N = 12,847 fills  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Position           N       Adverse%    Avg P&L    p-value  │
│  ─────────────────────────────────────────────────────────  │
│  Inside wall      4,231      18.2%      +$0.34     baseline │
│  At wall level    2,156      31.7%      -$0.12     <0.001   │
│  Outside wall     3,892      42.3%      -$0.67     <0.001   │
│  No wall nearby   2,568      28.9%      +$0.08     0.023    │
│                                                             │
│  [Box plot showing P&L distribution by cohort]              │
│                                                             │
│  Conclusion: Wall-relative position is highly predictive.   │
│  Effect size: 24.1% adverse selection reduction when        │
│  inside wall vs outside.                                    │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2 Feature Exploration

```
┌─────────────────────────────────────────────────────────────┐
│  Feature: Liquidity Migration Velocity                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Scatter: migration_velocity vs 1min_return]               │
│                                                             │
│  Correlation: 0.23 (p < 0.001)                              │
│  Mutual Information: 0.089 bits                             │
│  Lead time: ~200ms (velocity leads returns)                 │
│                                                             │
│  [Histogram: migration_velocity distribution]               │
│  [Time series: feature value over 14 days]                  │
│                                                             │
│  Regime breakdown:                                          │
│    Quiet:    corr = 0.08  (not predictive)                  │
│    Trending: corr = 0.41  (highly predictive)               │
│    Volatile: corr = 0.19  (moderately predictive)           │
│    Cascade:  corr = 0.52  (highly predictive)               │
└─────────────────────────────────────────────────────────────┘
```

#### 3.3 A/B Test Results

```
┌─────────────────────────────────────────────────────────────┐
│  A/B Test: Wall-Aware Quote Positioning                     │
│  Duration: 7 days  |  Traffic split: 50/50                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    Control        Treatment      Delta      │
│  ─────────────────────────────────────────────────────────  │
│  Total P&L         $1,234         $1,567        +27.0%     │
│  Adverse Select%   31.2%          24.8%         -6.4pp     │
│  Fill Rate         0.42           0.38          -9.5%      │
│  Sharpe Ratio      1.23           1.67          +0.44      │
│                                                             │
│  Statistical significance: p = 0.003                        │
│  Recommendation: DEPLOY to production                       │
└─────────────────────────────────────────────────────────────┘
```

---

### Data Architecture

#### Storage Requirements

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CAPTURE (Real-time)                                        │
│  ├─ Order book snapshots (100ms interval)                   │
│  │   └─ ~50 levels each side, both exchanges                │
│  │   └─ ~100KB/sec compressed                               │
│  ├─ Our quotes (every update)                               │
│  │   └─ bid/ask price, size, timestamp                      │
│  ├─ Fills (every event)                                     │
│  │   └─ Full context: book state, regime, signals           │
│  ├─ Model outputs (every quote cycle)                       │
│  │   └─ Predictions, confidence, feature values             │
│  └─ Regime state (every transition + 1min snapshots)        │
│                                                             │
│  STORAGE (Time-series optimized)                            │
│  ├─ Hot: Last 24 hours in memory (for live + recent replay) │
│  ├─ Warm: Last 30 days in local SSD (for replay)            │
│  └─ Cold: Full history in S3/GCS (for research)             │
│                                                             │
│  ESTIMATED VOLUME                                           │
│  ├─ Raw: ~8 GB/day                                          │
│  ├─ Compressed: ~1.5 GB/day                                 │
│  └─ 30-day warm storage: ~45 GB                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Data Schema: Fill Event (Enriched)

```rust
struct EnrichedFill {
    // Core fill data
    timestamp_us: i64,
    side: Side,
    price: f64,
    size: f64,

    // Outcome (filled in after mark-to-market window)
    pnl_1s: Option<f64>,
    pnl_10s: Option<f64>,
    pnl_60s: Option<f64>,
    adverse_selection: bool,

    // Microstructure context at fill time
    book_snapshot: BookContext,
    wall_context: WallContext,
    regime_state: RegimeState,

    // Cross-exchange context
    binance_mid: f64,
    binance_delta_100ms: f64,
    lead_lag_estimate_ms: f64,

    // Model state at fill time
    fill_prob_prediction: f64,
    adverse_prob_prediction: f64,
    kappa_estimate: f64,
    gamma_used: f64,
}

struct BookContext {
    mid_price: f64,
    spread_bps: f64,
    bid_depth_10bps: f64,
    ask_depth_10bps: f64,
    liquidity_asymmetry: f64,
    band_density: f64,
    migration_velocity: f64,
}

struct WallContext {
    nearest_wall_distance_bps: f64,
    nearest_wall_size: f64,
    position_relative_to_wall: WallPosition,  // Inside, At, Outside, None
    wall_age_ms: f64,  // How long has this wall existed?
    wall_flicker_rate: f64,  // Is it stable or flickering?
}
```

---

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │  Hyperliquid│     │   Binance   │     │   Market    │      │
│   │   WebSocket │     │  WebSocket  │     │   Maker     │      │
│   └──────┬──────┘     └──────┬──────┘     │   Core      │      │
│          │                   │            └──────┬──────┘      │
│          ▼                   ▼                   │              │
│   ┌─────────────────────────────────────────────┴────────┐     │
│   │              DATA CAPTURE LAYER                       │     │
│   │  • Book snapshots (both exchanges)                    │     │
│   │  • Quote events (ours)                                │     │
│   │  • Fill events (enriched)                             │     │
│   │  • Model outputs                                      │     │
│   └──────────────────────────┬───────────────────────────┘     │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐             │
│          ▼                   ▼                   ▼             │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │  In-Memory  │     │   TimeSeries│     │   Object    │      │
│   │  Ring Buffer│     │   DB (QuestDB│     │   Store     │      │
│   │  (24h hot)  │     │   or similar)│     │   (S3/GCS)  │      │
│   └──────┬──────┘     └──────┬──────┘     └─────────────┘      │
│          │                   │                                  │
│          └───────────────────┴───────────────────┐             │
│                                                  ▼             │
│   ┌─────────────────────────────────────────────────────┐      │
│   │              VISUALIZATION BACKEND                   │      │
│   │  • WebSocket server (live updates)                   │      │
│   │  • REST API (historical queries)                     │      │
│   │  • Aggregation engine (research queries)             │      │
│   └──────────────────────────┬──────────────────────────┘      │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────┐      │
│   │              FRONTEND (React + D3/Canvas)            │      │
│   │                                                      │      │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │      │
│   │  │  Live   │  │ Replay  │  │Research │             │      │
│   │  │  Mode   │  │  Mode   │  │  Mode   │             │      │
│   │  └─────────┘  └─────────┘  └─────────┘             │      │
│   │                                                      │      │
│   └─────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Frontend Technology Decision

**Current:** Single-file React with Recharts (CDN-loaded)

**Limitation:** Recharts can't efficiently render 50k+ data points for heatmaps or handle canvas-based high-performance rendering.

**Options:**

| Approach | Pros | Cons |
|----------|------|------|
| **Stay with Recharts** | Simple, working, no build | Can't do heatmaps efficiently |
| **Add D3.js** | Powerful, flexible | Steep learning curve, verbose |
| **Use Lightweight Charts (TradingView)** | Purpose-built for trading | Less flexible for custom viz |
| **Canvas + custom rendering** | Maximum performance | More work, harder to maintain |
| **WebGL (regl/three.js)** | Handles millions of points | Overkill for our scale |

**Recommendation:** Hybrid approach
- Keep Recharts for standard charts (PnL, regime, simple time series)
- Add **Lightweight Charts** for price/quote visualization (built for this)
- Add **custom Canvas component** for order book heatmap (specific need)

```
┌─────────────────────────────────────────────────────────────┐
│  Technology Stack (Proposed)                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Build System:      Vite (fast, modern, easy migration)     │
│  Framework:         React 18 (keep current, upgrade)        │
│  State:             Zustand (simple) or Jotai (atomic)      │
│  Standard Charts:   Recharts (keep)                         │
│  Trading Charts:    Lightweight Charts (TradingView OSS)    │
│  Heatmap:           Custom Canvas component                 │
│  Styling:           Tailwind (keep)                         │
│  Real-time:         WebSocket (keep) + React Query          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Implementation Roadmap

#### Phase 1: Data Capture Foundation

**Goal:** Capture everything needed for offline analysis.

- Add book snapshot capture (100ms, both exchanges)
- Enrich fill events with full microstructure context
- Store to local files (Parquet or similar)
- Build replay data loader

**Validation:** Can reconstruct any historical 5-minute window.

#### Phase 2: Heatmap Visualization

**Goal:** Core heatmap with quote overlay working.

- Build Canvas-based heatmap component
- Implement quote overlay rendering
- Add fill event markers
- Wire up to live WebSocket feed

**Validation:** Heatmap matches reference image style, quotes visible.

#### Phase 3: Replay Mode

**Goal:** Time-travel debugging capability.

- Build timeline scrubber component
- Implement time-synchronized multi-panel
- Add event markers and annotations
- Load historical data on demand

**Validation:** Can replay any incident from last 30 days.

#### Phase 4: Cross-Exchange View

**Goal:** Binance-Hyperliquid comparison.

- Side-by-side or overlaid book visualization
- Lead-lag real-time indicator
- Imbalance comparison
- Predictive signal display

**Validation:** Can visually confirm lead-lag relationship.

#### Phase 5: Research Mode

**Goal:** Statistical analysis without leaving the UI.

- Cohort builder (filter fills by any dimension)
- Aggregation queries (group by, statistics)
- Visualization of distributions, correlations
- Export to notebook for deeper analysis

**Validation:** Can answer "is X predictive of Y" questions in UI.

---

### API Endpoints (Proposed)

```
LIVE MODE
  GET  /api/dashboard           Current state snapshot
  WS   /ws/dashboard            Real-time updates

REPLAY MODE
  GET  /api/replay/range        Available data range
  GET  /api/replay/snapshot     State at specific timestamp
  GET  /api/replay/window       Data for time window
  GET  /api/replay/events       Events in time range

RESEARCH MODE
  POST /api/research/cohort     Build cohort with filters
  POST /api/research/aggregate  Aggregate metrics for cohort
  GET  /api/research/features   Available features for analysis
  POST /api/research/correlate  Correlation between features
```

---

### Success Metrics for Infrastructure

| Metric | Target | Rationale |
|--------|--------|-----------|
| Live latency (data → render) | < 100ms | Real-time feel |
| Replay load time (5-min window) | < 2s | Usable for debugging |
| Heatmap render (10k points) | 60 fps | Smooth interaction |
| Storage cost | < $50/month | Sustainable |
| Time to answer "why did we lose money" | < 5 min | Debugging efficiency |

---

## Next Steps

1. Build heatmap visualization with quote overlay
2. Instrument fill events with microstructure context
3. Run exploratory analysis on 2 weeks of data
4. Identify highest-signal features for model integration
5. A/B test quote positioning strategies informed by findings

---

## Appendix: Heatmap Feature Definitions

```
liquidity_asymmetry = (sum(size[price > mid]) - sum(size[price < mid]))
                     / (sum(size[price > mid]) + sum(size[price < mid]))

band_density = count(levels with size > threshold) / total_levels

migration_velocity = d/dt(sum(price[i] * size[i]) / sum(size[i]))  # Liquidity-weighted price movement

flicker_rate[level] = count(appear/disappear events) / time_window

relative_visibility = our_size[level] / total_size[level]

wall_distance = min(|our_quote - wall_price|) for walls > size_threshold
```
