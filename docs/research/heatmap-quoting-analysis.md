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

---

## Part 3: Complete Model Infrastructure

Before improving any model, you need to know exactly how wrong your current models are and in what ways. This is the foundation everything else builds on.

### 3.1 Measurement Infrastructure

#### The Prediction Log Schema

Every quote cycle, your system makes implicit predictions. These need to be recorded with enough granularity to diagnose failures.

```rust
struct PredictionRecord {
    // Timing
    timestamp_ns: u64,
    quote_cycle_id: u64,

    // Market state at prediction time
    market_state: MarketStateSnapshot,

    // Model outputs (what we predicted)
    predictions: ModelPredictions,

    // What actually happened (filled in async)
    outcomes: Option<ObservedOutcomes>,
}

struct MarketStateSnapshot {
    // L2 book state
    bid_levels: Vec<(f64, f64)>,  // (price, size) for top N levels
    ask_levels: Vec<(f64, f64)>,

    // Derived quantities your models use
    microprice: f64,
    microprice_std: f64,  // uncertainty on microprice

    // Kappa inputs
    kappa_book: f64,
    kappa_robust: f64,
    kappa_own: f64,
    kappa_final: f64,

    // Volatility
    sigma_bipower: f64,
    sigma_realized_1m: f64,

    // Gamma inputs
    gamma_base: f64,
    gamma_effective: f64,

    // External state
    funding_rate: f64,
    time_to_funding_settlement_s: f64,
    open_interest: f64,
    open_interest_delta_1m: f64,

    // Cross-exchange (if available)
    binance_mid: Option<f64>,
    binance_hl_spread: Option<f64>,

    // Your position
    inventory: f64,
    inventory_age_s: f64,
}

struct ModelPredictions {
    // For each quote level you're placing
    levels: Vec<LevelPrediction>,

    // Aggregate predictions
    expected_fill_rate_1s: f64,
    expected_fill_rate_10s: f64,
    expected_adverse_selection_bps: f64,
    regime_probabilities: HashMap<Regime, f64>,
}

struct LevelPrediction {
    side: Side,
    price: f64,
    size: f64,
    depth_from_mid_bps: f64,

    // Fill probability predictions at different horizons
    p_fill_100ms: f64,
    p_fill_1s: f64,
    p_fill_10s: f64,

    // Conditional predictions
    p_adverse_given_fill: f64,  // P(price moves against us | we get filled)
    expected_pnl_given_fill: f64,

    // Queue position estimate
    estimated_queue_position: f64,
    estimated_queue_total: f64,
}

struct ObservedOutcomes {
    // Fill outcomes
    fills: Vec<FillOutcome>,

    // Price evolution
    price_1s_later: f64,
    price_10s_later: f64,
    price_60s_later: f64,

    // Did we get adversely selected?
    adverse_selection_realized_bps: f64,
}

struct FillOutcome {
    level_index: usize,
    fill_timestamp_ns: u64,
    fill_price: f64,
    fill_size: f64,

    // Post-fill price evolution
    mark_price_at_fill: f64,
    mark_price_100ms_later: f64,
    mark_price_1s_later: f64,
    mark_price_10s_later: f64,
}
```

#### Calibration Analysis Pipeline

**Step 1: Probability Calibration Curves**

For each prediction type (fill probability at different horizons), bin predictions and compare to realized frequencies:

```
Algorithm: BuildCalibrationCurve
Input: predictions[], outcomes[], num_bins=20

1. Sort (prediction, outcome) pairs by prediction value
2. Divide into num_bins equal-sized buckets
3. For each bucket:
   - mean_predicted = average of predictions in bucket
   - realized_frequency = fraction where outcome=True
   - count = number of samples in bucket
4. Return [(mean_predicted, realized_frequency, count), ...]
```

**Interpretation:**
- Perfect calibration: points lie on y=x diagonal
- Overconfident: curve below diagonal (you predict 70%, reality is 50%)
- Underconfident: curve above diagonal

**Step 2: Brier Score Decomposition**

The Brier score is the mean squared error of probability predictions:

```
BS = (1/N) Σ (pᵢ - oᵢ)²
```

Where pᵢ is predicted probability and oᵢ ∈ {0,1} is outcome.

Decompose into three components:

```
BS = Reliability - Resolution + Uncertainty

Reliability = (1/N) Σₖ nₖ(p̄ₖ - ōₖ)²
  - Measures calibration quality
  - Lower is better
  - If your 70% predictions hit 70%, this is 0

Resolution = (1/N) Σₖ nₖ(ōₖ - ō)²
  - Measures discrimination ability
  - Higher is better
  - Are your high predictions different from low predictions?

Uncertainty = ō(1 - ō)
  - Base rate variance
  - Not controllable, just the inherent difficulty
```

**Step 3: Information Ratio**

For each model output, compute how much information it provides:

```
Information Ratio = Resolution / Uncertainty

IR > 1.0: Model predictions carry useful information
IR ≈ 1.0: Model is roughly as good as predicting base rate
IR < 1.0: Model is adding noise
```

Track IR over time. If IR degrades, your model is becoming stale.

**Step 4: Conditional Calibration Analysis**

Overall calibration can hide regime-dependent failures. Slice your calibration analysis by:

```
Conditioning Variables:
├── Volatility regime (σ quartiles)
├── Funding rate regime (positive/negative/extreme)
├── Time of day (funding settlement windows)
├── Inventory state (long/flat/short)
├── Recent fill rate (active/quiet)
├── Book imbalance (bid-heavy/balanced/ask-heavy)
└── Cross-exchange state (HL leading/lagging/synced)
```

#### Outcome Attribution Pipeline

When you lose money, you need to know why:

```rust
struct CycleAttribution {
    cycle_id: u64,
    gross_pnl: f64,

    // Decomposition
    spread_capture: f64,      // Revenue from bid-ask spread
    adverse_selection: f64,   // Loss from fills before adverse moves
    inventory_cost: f64,      // Cost of holding inventory
    fee_cost: f64,            // Exchange fees

    // Model accuracy
    fill_prediction_error: f64,
    adverse_selection_prediction_error: f64,
    volatility_prediction_error: f64,
}
```

**Daily Attribution Report:**

```
=== PnL Attribution: 2026-01-13 ===

Gross PnL:              -$127.45
├── Spread Capture:     +$284.30  (working correctly)
├── Adverse Selection:  -$312.80  (THIS IS THE PROBLEM)
├── Inventory Cost:     -$45.20   (acceptable)
└── Fees:               -$53.75   (fixed cost)

Model Accuracy:
├── Fill Prediction:    Brier=0.18, IR=1.34  (good)
├── Adverse Selection:  Brier=0.31, IR=0.89  (NEEDS WORK)
└── Volatility:         RMSE=0.000012        (acceptable)

Regime Breakdown:
├── Quiet (68% of time):    +$89.20
├── Active (28% of time):   -$45.60
└── Extreme (4% of time):   -$171.05  (INVESTIGATE)
```

This tells you exactly where to focus: adverse selection prediction during extreme regimes.

---

### 3.2 Information Source Audit

Before building models, systematically measure what signals contain predictive information.

#### Mutual Information Estimation

For each candidate signal X and target variable Y, estimate mutual information:

```
I(X; Y) = H(Y) - H(Y|X)
```

Where H is entropy. This measures how many bits of information X provides about Y.

**Continuous Variables: k-NN Estimator (Kraskov et al.)**

```rust
fn estimate_mutual_information(
    x: &[f64],  // Signal values
    y: &[f64],  // Target values
    k: usize    // Number of neighbors (typically 3-10)
) -> f64 {
    let n = x.len();

    // Build k-d trees for joint and marginals
    // Find k-th nearest neighbor distance in joint space
    // Count points within eps in marginals
    // MI ≈ digamma(k) + digamma(n) - digamma(n_x) - digamma(n_y)

    // ... implementation details
}
```

#### Signal Catalog

```rust
struct SignalCatalog {
    // Book-derived
    microprice_imbalance: f64,      // (bid_size - ask_size) / (bid_size + ask_size)
    book_pressure: f64,             // Integrated depth asymmetry
    spread_bps: f64,
    depth_at_1bps: f64,
    depth_at_5bps: f64,

    // Trade-derived
    trade_imbalance_1s: f64,        // Net signed volume last 1s
    trade_imbalance_10s: f64,
    trade_arrival_rate: f64,        // Trades per second
    avg_trade_size: f64,
    large_trade_indicator: bool,    // Trade > 2σ from mean

    // Hyperliquid-specific
    funding_rate: f64,
    funding_rate_change_1h: f64,
    time_to_funding_settlement: f64,
    open_interest: f64,
    open_interest_change_1m: f64,
    open_interest_change_10m: f64,

    // Cross-exchange
    binance_hl_spread: f64,
    binance_lead_indicator: f64,    // Recent price change on Binance
    binance_volume_ratio: f64,      // Binance volume / HL volume

    // Derived/Composite
    funding_x_imbalance: f64,       // funding_rate * trade_imbalance
    oi_momentum: f64,               // OI change acceleration
}
```

#### Signal Audit Report Format

```
=== Signal Audit Report: 2026-01-13 ===

Target: PriceDirection1s

Signal                      MI (bits)  Corr    Opt Lag   Regime Var
─────────────────────────────────────────────────────────────────────
binance_lead_indicator      0.089      0.31    -150ms    High (2.3x in volatile)
trade_imbalance_1s          0.067      0.24    0ms       Medium
microprice_imbalance        0.045      0.19    0ms       Low
funding_x_imbalance         0.041      0.15    0ms       High (3.1x near settlement)
open_interest_change_1m     0.023      0.08    0ms       Low

Target: AdverseSelectionOnNextFill

Signal                      MI (bits)  Corr    Opt Lag   Regime Var
─────────────────────────────────────────────────────────────────────
trade_arrival_rate          0.134      0.42    0ms       High (4.2x in cascade)
large_trade_indicator       0.098      0.38    -50ms     Medium
binance_hl_spread           0.076      0.29    -100ms    High

ACTIONABLE INSIGHTS:
1. binance_lead_indicator is your highest-value unused signal for direction
2. trade_arrival_rate strongly predicts adverse selection - use for dynamic kappa
3. funding_x_imbalance has 3x higher MI near settlement - regime-condition this
```

---

### 3.3 Proprietary Fill Intensity Model (Hawkes Process)

Standard Hawkes process for trade arrivals:

```
λ(t) = μ + ∫₀ᵗ α·e^(-β(t-s)) dN(s)
```

Where:
- μ = baseline intensity
- α = excitation from each event
- β = decay rate
- N(s) = counting process (number of trades by time s)

#### Exchange-Specific Extensions

**Extension 1: State-Dependent Baseline**

```
μ(t) = μ₀ · exp(w_F · F(t) + w_OI · ΔOI(t) + w_τ · τ(t))
```

Where:
- F(t) = funding rate
- ΔOI(t) = OI change rate
- τ(t) = time to funding settlement (cyclical feature)

**Extension 2: Trade-Type-Dependent Excitation**

```rust
fn compute_excitation(trade: &Trade, our_side: Side) -> f64 {
    let base_alpha = 0.3;

    // Size effect: larger trades excite more
    let size_mult = (trade.size / MEDIAN_TRADE_SIZE).sqrt().min(3.0);

    // Side effect: trades on our side are more relevant for our fills
    let side_mult = if trade.side == our_side { 1.5 } else { 0.8 };

    // Aggressor effect: market orders excite more than limit fills
    let aggressor_mult = if trade.is_aggressor { 1.2 } else { 1.0 };

    base_alpha * size_mult * side_mult * aggressor_mult
}
```

**Extension 3: Queue-Position-Dependent Kernel**

```rust
fn adaptive_kernel(
    time_since_trade: f64,
    queue_change_since_trade: f64,
    beta: f64
) -> f64 {
    // Standard temporal decay
    let temporal = (-beta * time_since_trade).exp();

    // Queue consumption effect
    let queue_mult = 1.0 + 0.5 * (queue_change_since_trade / TYPICAL_QUEUE_SIZE).min(1.0);

    temporal * queue_mult
}
```

#### Full Model Specification

```rust
struct HyperliquidFillIntensityModel {
    // Baseline parameters
    mu_0: f64,
    w_funding: f64,
    w_oi_change: f64,
    w_time_to_settlement: f64,

    // Excitation parameters
    alpha_base: f64,
    alpha_size_power: f64,
    alpha_same_side_mult: f64,
    alpha_aggressor_mult: f64,

    // Decay parameters
    beta_time: f64,
    beta_queue_sensitivity: f64,

    // Regime-switching
    regime_multipliers: HashMap<Regime, f64>,
}

impl HyperliquidFillIntensityModel {
    fn intensity_at(
        &self,
        t: f64,
        recent_trades: &[Trade],
        queue_position: f64,
        market_state: &MarketState
    ) -> f64 {
        // State-dependent baseline
        let funding_effect = self.w_funding * market_state.funding_rate;
        let oi_effect = self.w_oi_change * market_state.oi_change_rate;
        let settlement_effect = self.w_time_to_settlement
            * (market_state.time_to_settlement / 8.0 * TAU).sin();

        let mu_t = self.mu_0 * (funding_effect + oi_effect + settlement_effect).exp();

        // Excitation from recent trades
        let excitation = self.compute_excitation_sum(t, recent_trades, queue_position);

        // Regime adjustment
        let regime_mult = self.regime_multipliers
            .get(&market_state.regime)
            .copied()
            .unwrap_or(1.0);

        (mu_t + excitation) * regime_mult
    }
}
```

#### Converting Intensity to Kappa

```rust
fn intensity_to_kappa(
    fill_intensity_model: &HyperliquidFillIntensityModel,
    market_state: &MarketState,
    reference_depth_bps: f64
) -> f64 {
    // Kappa represents: additional fills per unit of spread tightening
    // κ = ∂(fill_rate)/∂(depth)

    let eps = 0.1;  // 0.1 bps perturbation

    let fill_rate_at_depth = fill_intensity_model.expected_fills_in_window(
        0.0, 1.0, reference_depth_bps, market_state
    );

    let fill_rate_tighter = fill_intensity_model.expected_fills_in_window(
        0.0, 1.0, reference_depth_bps - eps, market_state
    );

    let kappa = (fill_rate_tighter - fill_rate_at_depth) / (eps / 10000.0);

    kappa.max(100.0)  // Floor to prevent division issues in GLFT
}
```

---

### 3.4 Adverse Selection Decomposition

#### Mixture Model for Trade Classification

Every trade comes from one of several latent types:

```
Types: {noise, informed, liquidation, arbitrage}

P(type | observable_features) = softmax(W · φ(features))
```

**Feature Engineering:**

```rust
struct TradeFeatures {
    // Size features
    size_zscore: f64,
    size_quantile: f64,

    // Timing features
    time_since_last_trade_ms: f64,
    trades_in_last_1s: u32,
    trades_in_last_10s: u32,

    // Aggression features
    is_aggressor: bool,
    crossed_spread_bps: f64,

    // Directional features
    signed_volume_imbalance_1s: f64,
    signed_volume_imbalance_10s: f64,

    // Funding interaction
    funding_rate: f64,
    trade_aligns_with_funding: bool,

    // Cross-exchange
    binance_price_change_100ms: f64,
    binance_hl_spread_at_trade: f64,

    // Book state
    book_imbalance_at_trade: f64,
    depth_consumed_pct: f64,

    // Hyperliquid-specific
    oi_change_1m_before: f64,
    near_liquidation_price: bool,
}
```

**Labeling Strategy (for training):**

```rust
enum TradeLabel {
    Informed,      // Price moved >X bps in trade direction within 10s
    Noise,         // Price stayed flat or reversed
    Liquidation,   // Part of a liquidation cascade (detect via OI drop)
    Arbitrage,     // Cross-exchange spread closed immediately after
}

fn label_trade(trade: &Trade, future_prices: &[f64], context: &MarketContext) -> TradeLabel {
    // Check for liquidation cascade
    if context.oi_dropped_significantly && context.funding_extreme {
        return TradeLabel::Liquidation;
    }

    // Check for arbitrage
    if context.cross_exchange_spread_closed_within_500ms {
        return TradeLabel::Arbitrage;
    }

    // Informed vs noise based on ex-post price move
    let price_10s_later = future_prices[100];
    let signed_move = compute_signed_move(trade, price_10s_later);

    if signed_move > 5.0 {  // Moved >5 bps in trade direction
        TradeLabel::Informed
    } else {
        TradeLabel::Noise
    }
}
```

#### Real-Time Integration

```rust
struct AdverseSelectionAdjuster {
    classifier: TradeClassifier,
    informed_intensity: ExponentialMovingAverage,
    kappa_discount_per_informed_pct: f64,
    spread_premium_per_informed_pct: f64,
}

impl AdverseSelectionAdjuster {
    fn on_trade(&mut self, trade: &Trade, features: &TradeFeatures) {
        let informed_prob = self.classifier.informed_probability(features);
        self.informed_intensity.update(informed_prob);
    }

    fn get_kappa_adjustment(&self) -> f64 {
        let informed_pct = self.informed_intensity.value() * 100.0;
        let adjustment = 1.0 - informed_pct * self.kappa_discount_per_informed_pct;
        adjustment.max(0.3)  // Don't reduce kappa by more than 70%
    }

    fn get_spread_adjustment_bps(&self) -> f64 {
        let informed_pct = self.informed_intensity.value() * 100.0;
        informed_pct * self.spread_premium_per_informed_pct
    }
}
```

#### Liquidation Detection Subsystem

```rust
struct LiquidationDetector {
    oi_history: RingBuffer<(u64, f64)>,
    funding_history: RingBuffer<(u64, f64)>,
    cascade_threshold_oi_drop_pct: f64,
    liquidation_probability: f64,
}

impl LiquidationDetector {
    fn update(&mut self, current_oi: f64, current_funding: f64, timestamp: u64) {
        self.oi_history.push((timestamp, current_oi));
        self.funding_history.push((timestamp, current_funding));

        let oi_1m_ago = self.oi_history.get_at_time(timestamp - 60_000);
        let oi_change_pct = (current_oi - oi_1m_ago) / oi_1m_ago * 100.0;
        let funding_percentile = self.funding_history.percentile_rank(current_funding);

        self.liquidation_probability = self.compute_liquidation_probability(
            oi_change_pct,
            funding_percentile,
            current_funding
        );
    }

    fn is_cascade_active(&self) -> bool {
        self.liquidation_probability > 0.5
    }
}
```

---

### 3.5 Regime Detection System (HMM)

#### Hidden Markov Model Specification

**States:**

```rust
enum MarketRegime {
    Quiet,          // Low volatility, balanced flow, normal fill rates
    Trending,       // Directional momentum, elevated adverse selection
    Volatile,       // High volatility, wide spreads, uncertain direction
    Cascade,        // Liquidation cascade, extreme toxicity
}
```

**Emission Model:**

```rust
struct RegimeEmissionModel {
    volatility_mean: f64,
    volatility_std: f64,
    trade_intensity_mean: f64,
    trade_intensity_std: f64,
    imbalance_mean: f64,
    imbalance_std: f64,
    adverse_selection_mean: f64,
    adverse_selection_std: f64,
}

struct HMMParams {
    transition_probs: [[f64; 4]; 4],
    emissions: [RegimeEmissionModel; 4],
    initial_probs: [f64; 4],
}
```

**Initialization:**

```rust
fn initialize_hmm_params() -> HMMParams {
    HMMParams {
        transition_probs: [
            // From Quiet: mostly stays quiet
            [0.95, 0.03, 0.019, 0.001],
            // From Trending: can stay or revert
            [0.10, 0.85, 0.04, 0.01],
            // From Volatile: can calm down or escalate
            [0.15, 0.10, 0.70, 0.05],
            // From Cascade: usually short, reverts to volatile
            [0.05, 0.05, 0.60, 0.30],
        ],
        // ... emission parameters
    }
}
```

#### Online Filtering (Forward Algorithm)

```rust
struct OnlineHMMFilter {
    params: HMMParams,
    belief: [f64; 4],  // P(regime | observations so far)
    observation_buffer: RingBuffer<ObservationVector>,
}

struct ObservationVector {
    volatility: f64,
    trade_intensity: f64,
    imbalance: f64,
    adverse_selection: f64,
}

impl OnlineHMMFilter {
    fn update(&mut self, obs: &ObservationVector) {
        // Prediction step: apply transition matrix
        let mut predicted = [0.0; 4];
        for j in 0..4 {
            for i in 0..4 {
                predicted[j] += self.params.transition_probs[i][j] * self.belief[i];
            }
        }

        // Update step: multiply by observation likelihood
        let mut updated = [0.0; 4];
        let mut normalizer = 0.0;

        for i in 0..4 {
            let likelihood = self.observation_likelihood(obs, i);
            updated[i] = predicted[i] * likelihood;
            normalizer += updated[i];
        }

        // Normalize
        for i in 0..4 {
            self.belief[i] = updated[i] / normalizer;
        }
    }

    fn most_likely_regime(&self) -> MarketRegime {
        let max_idx = self.belief.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;

        match max_idx {
            0 => MarketRegime::Quiet,
            1 => MarketRegime::Trending,
            2 => MarketRegime::Volatile,
            3 => MarketRegime::Cascade,
            _ => unreachable!(),
        }
    }
}
```

#### Regime-Dependent Parameters

```rust
struct RegimeSpecificParams {
    gamma: f64,
    kappa_multiplier: f64,
    spread_floor_bps: f64,
    max_inventory: f64,
}

fn get_regime_params(regime: MarketRegime) -> RegimeSpecificParams {
    match regime {
        MarketRegime::Quiet => RegimeSpecificParams {
            gamma: 0.3,
            kappa_multiplier: 1.0,
            spread_floor_bps: 5.0,
            max_inventory: 1.0,
        },
        MarketRegime::Trending => RegimeSpecificParams {
            gamma: 0.5,
            kappa_multiplier: 0.7,
            spread_floor_bps: 10.0,
            max_inventory: 0.5,
        },
        MarketRegime::Volatile => RegimeSpecificParams {
            gamma: 0.8,
            kappa_multiplier: 1.5,
            spread_floor_bps: 15.0,
            max_inventory: 0.3,
        },
        MarketRegime::Cascade => RegimeSpecificParams {
            gamma: 2.0,
            kappa_multiplier: 5.0,
            spread_floor_bps: 50.0,
            max_inventory: 0.1,
        },
    }
}

fn blend_params_by_belief(hmm_filter: &OnlineHMMFilter) -> RegimeSpecificParams {
    let probs = hmm_filter.regime_probabilities();

    let mut blended = RegimeSpecificParams::default();

    for (regime, prob) in probs {
        let params = get_regime_params(regime);
        blended.gamma += prob * params.gamma;
        blended.kappa_multiplier += prob * params.kappa_multiplier;
        blended.spread_floor_bps += prob * params.spread_floor_bps;
        blended.max_inventory += prob * params.max_inventory;
    }

    blended
}
```

---

### 3.6 Cross-Exchange Lead-Lag Model

This is potentially the highest-value proprietary signal.

#### Lead-Lag Estimation

**Observation:** Binance BTC perp leads Hyperliquid BTC perp by 50-500ms depending on conditions.

```rust
struct LeadLagEstimator {
    binance_changes: RingBuffer<(u64, f64)>,
    hl_changes: RingBuffer<(u64, f64)>,

    lag_estimate_ms: f64,
    lag_estimate_std: f64,
    beta_estimate: f64,  // HL_change ≈ β × Binance_change(t - lag)
    beta_estimate_std: f64,

    r_squared: f64,
    sample_count: usize,
}

impl LeadLagEstimator {
    fn estimate_lag(&mut self) {
        // Grid search over candidate lags
        let candidate_lags_ms: Vec<f64> = (-100..=500).step_by(10)
            .map(|x| x as f64).collect();

        let mut best_lag = 0.0;
        let mut best_r2 = -1.0;
        let mut best_beta = 0.0;

        for &lag_ms in &candidate_lags_ms {
            let (beta, r2) = self.compute_regression_at_lag(lag_ms);
            if r2 > best_r2 {
                best_r2 = r2;
                best_lag = lag_ms;
                best_beta = beta;
            }
        }

        // Update estimates with smoothing
        let alpha = 0.1;
        self.lag_estimate_ms = alpha * best_lag + (1.0 - alpha) * self.lag_estimate_ms;
        self.beta_estimate = alpha * best_beta + (1.0 - alpha) * self.beta_estimate;
        self.r_squared = alpha * best_r2 + (1.0 - alpha) * self.r_squared;
    }
}
```

#### Regime-Conditioned Lead-Lag

The lead-lag relationship changes with volatility:

```rust
struct RegimeConditionedLeadLag {
    estimators: HashMap<VolatilityRegime, LeadLagEstimator>,
    current_regime: VolatilityRegime,
}

impl RegimeConditionedLeadLag {
    fn predict_hl_move(&self, recent_binance_change: f64, time_since_change_ms: f64) -> f64 {
        let (lag_ms, beta) = self.get_current_estimate();

        if time_since_change_ms < lag_ms {
            let completion_fraction = time_since_change_ms / lag_ms;
            let remaining_move = beta * recent_binance_change * (1.0 - completion_fraction);
            remaining_move
        } else {
            0.0
        }
    }
}
```

#### Integration into Quote Generation

```rust
fn compute_adjusted_microprice(
    local_microprice: f64,
    lead_lag_model: &RegimeConditionedLeadLag,
    recent_binance_move: f64,
    time_since_binance_move_ms: f64
) -> f64 {
    let predicted_hl_move = lead_lag_model.predict_hl_move(
        recent_binance_move,
        time_since_binance_move_ms
    );

    local_microprice * (1.0 + predicted_hl_move)
}

fn compute_directional_skew(
    lead_lag_model: &RegimeConditionedLeadLag,
    recent_binance_move: f64,
    time_since_binance_move_ms: f64,
    base_skew: f64
) -> f64 {
    let (lag_ms, _beta) = lead_lag_model.get_current_estimate();

    if time_since_binance_move_ms < lag_ms {
        let expected_direction = recent_binance_move.signum();
        let confidence = lead_lag_model.current_r_squared();

        // Add skew in direction of expected move
        base_skew + expected_direction * confidence * 2.0  // 2 bps at full confidence
    } else {
        base_skew
    }
}
```

---

### 3.7 Integration Architecture

```rust
struct EnhancedQuoteEngine {
    // Existing components
    kappa_orchestrator: KappaOrchestrator,
    volatility_estimator: BipowerVariation,
    microprice_estimator: MicropriceEstimator,

    // New components
    prediction_logger: PredictionLogger,
    fill_intensity_model: HyperliquidFillIntensityModel,
    trade_classifier: TradeClassifier,
    adverse_selection_adjuster: AdverseSelectionAdjuster,
    hmm_filter: OnlineHMMFilter,
    liquidation_detector: LiquidationDetector,
    lead_lag_model: RegimeConditionedLeadLag,

    // Calibration tracking
    calibration_metrics: CalibrationMetrics,
}

impl EnhancedQuoteEngine {
    fn generate_quotes(&mut self, market_data: &MarketData) -> QuoteSet {
        // 1. Update all models with new data
        self.update_models(market_data);

        // 2. Log predictions for calibration
        let predictions = self.generate_predictions(market_data);
        self.prediction_logger.log(&predictions, market_data);

        // 3. Get regime-blended parameters
        let regime_params = blend_params_by_belief(&self.hmm_filter);

        // 4. Check for liquidation cascade
        if self.liquidation_detector.is_cascade_active() {
            return self.generate_defensive_quotes(market_data);
        }

        // 5. Compute adjusted microprice using lead-lag
        let adjusted_microprice = compute_adjusted_microprice(
            self.microprice_estimator.get_microprice(),
            &self.lead_lag_model,
            market_data.recent_binance_move,
            market_data.time_since_binance_move_ms
        );

        // 6. Compute fill-intensity-based kappa
        let intensity_kappa = intensity_to_kappa(
            &self.fill_intensity_model,
            &market_data.market_state,
            10.0
        );

        // 7. Apply adverse selection adjustment
        let adjusted_kappa = intensity_kappa
            * self.adverse_selection_adjuster.get_kappa_adjustment()
            * regime_params.kappa_multiplier;

        // 8. Compute GLFT optimal spread
        let gamma = regime_params.gamma;
        let glft_half_spread = (1.0 / gamma) * (1.0 + gamma / adjusted_kappa).ln()
            + MAKER_FEE;

        // 9. Apply floor and uncertainty premium
        let spread_floor = regime_params.spread_floor_bps / 10000.0;
        let uncertainty_mult = self.compute_uncertainty_multiplier();
        let final_half_spread = glft_half_spread.max(spread_floor) * uncertainty_mult;

        // 10. Compute inventory skew
        let inventory_skew = self.compute_inventory_skew(market_data);

        // 11. Add lead-lag directional skew
        let directional_skew = compute_directional_skew(
            &self.lead_lag_model,
            market_data.recent_binance_move,
            market_data.time_since_binance_move_ms,
            0.0
        ) / 10000.0;

        // 12. Generate final quotes
        let bid_depth = final_half_spread + inventory_skew - directional_skew;
        let ask_depth = final_half_spread - inventory_skew + directional_skew;

        QuoteSet {
            bid_price: adjusted_microprice * (1.0 - bid_depth),
            ask_price: adjusted_microprice * (1.0 + ask_depth),
            bid_size: self.compute_bid_size(regime_params.max_inventory),
            ask_size: self.compute_ask_size(regime_params.max_inventory),

            regime: self.hmm_filter.most_likely_regime(),
            kappa_used: adjusted_kappa,
            gamma_used: gamma,
            spread_bps: (bid_depth + ask_depth) * 10000.0,
        }
    }
}
```

---

### 3.8 Implementation Priority

Based on expected edge per engineering effort:

| Phase | Focus | Rationale |
|-------|-------|-----------|
| 1 | Measurement Infrastructure | Foundational - everything else depends on measuring properly |
| 2 | Signal Audit | Identify highest-value unused signals before building |
| 3 | Lead-Lag Model | Highest expected edge for moderate complexity |
| 4 | Regime Detection | Replace warmup multiplier with proper Bayesian belief |
| 5 | Adverse Selection Classifier | Requires labeled data from earlier phases |
| 6 | Fill Intensity Model | Most complex, requires queue position inference |

Each component should be validated against the measurement infrastructure before moving to the next. If a component doesn't improve calibration metrics, investigate why before building more complexity.
