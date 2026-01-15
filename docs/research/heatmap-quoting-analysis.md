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
