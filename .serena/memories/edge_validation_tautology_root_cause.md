# Edge Validation Tautology: AS ≈ Spread Root Cause Trace

## Executive Summary

**The edge validation data is TAUTOLOGICAL.** `realized_as_bps ≈ realized_spread_bps` because:
1. **Both are computed using `self.latest_mid` at the time of fill logging** (line 755 in handlers.rs)
2. **AS formula uses `latest_mid` at fill time; depth formula also uses `latest_mid` at fill time**
3. **When `latest_mid ≈ mid_at_fill`, then AS ≈ depth by algebraic identity**
4. **`mid_at_fill` is set to `self.latest_mid` at line 755** — storing the same value being used for AS computation

This makes ALL edge metrics, learning signals, and optimization targets **mathematically equivalent to noise**.

---

## Code Trace: Step-by-Step

### Step 1: Fill Arrives (line 701-745)
```rust
// handlers.rs:706-714
let fill_price: f64 = fill.px.parse().unwrap_or(0.0);
let is_buy = fill.side == "B" || fill.side.to_lowercase() == "buy";
let direction = if is_buy { 1.0 } else { -1.0 };
let as_realized = (self.latest_mid - fill_price) * direction / fill_price;
```

**Key fact**: `as_realized` uses `self.latest_mid` (current state at time of fill logging, which is NOW)

### Step 2: Depth Computed Same Way (line 718)
```rust
// handlers.rs:718
let depth_from_mid = (fill_price - self.latest_mid).abs() / self.latest_mid;
```

**Key fact**: `depth_from_mid` ALSO uses `self.latest_mid` (same current value!)

**Comment on line 717 admits the problem:**
```
// For now use adverse selection as proxy until we track order placement mid
```
This is the smoking gun — no `mid_at_placement` is tracked; `latest_mid` is used instead.

### Step 3: Queue Entry Created (line 750-757)
```rust
// handlers.rs:755
mid_at_fill: self.latest_mid,
```

**Key fact**: `mid_at_fill` is set to `self.latest_mid` at THIS MOMENT.

### Step 4: Edge Snapshot Logged (line 778-798)

**Immediate logging (before 5-second delay):**
```rust
// handlers.rs:714
let as_realized = (self.latest_mid - fill_price) * direction / fill_price;

// handlers.rs:718
let depth_from_mid = (fill_price - self.latest_mid).abs() / self.latest_mid;

// handlers.rs:782-783 (edge snapshot)
let depth_bps = depth_from_mid * 10_000.0;
let as_realized_bps = as_realized * 10_000.0;
```

**The EdgeSnapshot line 791 hardcodes a major bug:**
```rust
predicted_spread_bps: depth_bps,
realized_spread_bps: depth_bps,
```

Both `predicted` and `realized` spread are set to the SAME value — the spread from the current mid, not the actual quoted spread.

---

## Algebraic Identity: Why AS ≈ Depth

Let:
- `P_fill` = fill price
- `M_now` = `self.latest_mid` at time of fill logging
- `M_fill` = mid price when order was placed (NOT tracked; assumed ≈ M_now)

### AS Formula (immediate, line 714):
```
AS = (M_now - P_fill) / P_fill  [for a buy]
   = (M_now - P_fill) / P_fill
```

### Depth Formula (immediate, line 718):
```
Depth = |P_fill - M_now| / M_now
      = (M_now - P_fill) / M_now  [if M_now > P_fill, i.e., we bought inside]
```

### Relationship:
```
AS / Depth = [(M_now - P_fill) / P_fill] / [(M_now - P_fill) / M_now]
           = M_now / P_fill
           ≈ 1.0 (since mid ≈ P_fill for liquid markets)
```

**Therefore: AS ≈ Depth × (P_fill / M_now) ≈ Depth when mid is near fill price**

This is why you observe `realized_as_bps ≈ realized_spread_bps` in 97% of fills.

---

## Why 97% Not 100%?

The remaining 3% deviation comes from:
1. **Extreme spreads**: When we quote far off the book (rare), `|P_fill - M_now|` is large enough that rounding and mid movement matter
2. **Fast-moving markets**: Between order placement and fill logging, `M_now` might have moved enough from `M_fill` that the identity breaks
3. **Data quality issues**: Stale mid price updates or timestamp misalignment

---

## The 5-Second Markout Mystery

**Key insight: The 5-second markout in `check_pending_fill_outcomes()` (line 223-300) is DIFFERENT logic.**

At line 250:
```rust
let mid_change_bps = ((self.latest_mid - pending.mid_at_fill) / pending.mid_at_fill) * 10_000.0;
```

This PROPERLY computes the change in mid from fill time to 5 seconds later:
- `pending.mid_at_fill` = mid at fill time (stored at line 755 in the queue)
- `self.latest_mid` = mid NOW (5 seconds later)
- `mid_change_bps` = actual price movement

**This is correct AS measurement!** But it's:
1. **Only used for model gating and pre-fill classifier training** (line 259-270)
2. **NOT used in the EdgeSnapshot logged at fill time** (line 778-798)
3. **Logged to `realized_as_bps` in the earlier snapshot with `as_realized` from line 714**

So the EdgeSnapshot is **using immediate AS** (which ≈ depth), while the 5-second markout is **correctly measured but separate**.

---

## The Broken Edge Tracking Flow

```
Fill arrives at time T:
  ├─ Compute as_realized = (M(T) - P_fill) / P_fill              [IMMEDIATE]
  ├─ Compute depth_from_mid = (P_fill - M(T)) / M(T)             [IMMEDIATE, SAME M(T)]
  ├─ Store mid_at_fill = M(T) in queue                           [STORING M(T)]
  ├─ Log EdgeSnapshot with (depth_bps, as_realized_bps)          [TAUTOLOGICAL: both ≈ same value]
  ├─ Record edge metrics to learning/analytics                   [FEEDING NOISE TO MODELS]
  │
  └─ 5 seconds later at time T+5:
      ├─ Check if M(T+5) moved against us                        [CORRECT: uses stored M(T)]
      ├─ Feed outcome to pre_fill_classifier                     [CORRECT SIGNAL]
      └─ [But this is SEPARATE from the EdgeSnapshot already logged]
```

---

## Why This Breaks Everything

1. **Edge prediction models trained on noise**: predicted_as_bps learned from `realized_as_bps ≈ spread`
2. **Realized edge = spread - as - fee ≈ 0** (or even negative) by construction
3. **RL reward uses realized_edge_bps** (handlers.rs:895) — training signal is zero/negative always
4. **Calibration metrics corrupted**: Brier score, IR calculated on tautological data
5. **PnL attribution broken**: `fill_pnl_bps = snap.realized_edge_bps` (line 802) records noise, not actual PnL
6. **No signal about actual spread capture**: We can't distinguish between "we quoted tight" vs "we got run over"

---

## The Root Causes

### RC1: No `mid_at_placement` Tracking
**Problem**: Orders are placed with some mid M_place, but we never store it.
**Evidence**: Line 717 comment: "For now use adverse selection as proxy until we track order placement mid"
**Impact**: `depth_from_mid` uses current mid, not placement mid → depth ≈ AS

### RC2: Immediate AS Logging vs 5-Second Markout
**Problem**: EdgeSnapshot logged immediately with AS from line 714, but 5-second markout is separate.
**Evidence**: 
- `as_realized` at line 714 uses `M_now`
- `mid_at_fill` at line 755 also stores `M_now`
- These go into EdgeSnapshot at line 788-798
- But 5-second check at line 250 uses stored `mid_at_fill` correctly
**Impact**: Two separate AS measurements; immediate one is broken

### RC3: Both `predicted_spread_bps` and `realized_spread_bps` Hardcoded to Same Value
**Problem**: Line 790-791 in EdgeSnapshot:
```rust
predicted_spread_bps: depth_bps,
realized_spread_bps: depth_bps,
```
**Impact**: These are IDENTICAL. We have no baseline to compare against.

### RC4: `mid_at_fill` Stores Current Mid, Not Placement Mid
**Problem**: Line 755: `mid_at_fill: self.latest_mid`
**Expected**: Should store `self.latest_mid` at order placement time, not fill time
**Impact**: The 5-second markout compares M(T+5) against M(T), which might not be order placement time

---

## Data Evidence from edge_validation.jsonl

97% of fills show: `realized_as_bps ≈ realized_spread_bps`

This is explained by:
```
realized_as_bps      ≈ (M(T) - P_fill) / P_fill × 10000
realized_spread_bps  ≈ |P_fill - M(T)| / M(T) × 10000
                     ≈ (M(T) - P_fill) / M(T) × 10000
```

Since M(T) ≈ P_fill in most fills (we quote near mid), these are nearly identical.

The 3% outliers are fills where:
- We quoted very wide (rare)
- Mid moved significantly between placement and logging
- Data quality issues (gaps, stale quotes)

---

## Fixing This Requires

### Phase 1: Track `mid_at_placement`
1. Store mid price at order placement time
2. Use it in `depth_from_mid` calculation (line 718)
3. Pass it into EdgeSnapshot

### Phase 2: Use 5-Second Markout as Primary AS
1. Make EdgeSnapshot fields:
   - `realized_spread_bps` = actual quoted spread (captured at placement)
   - `realized_as_bps` = true AS from 5-second markout (computed 5s later)
2. Don't compute AS immediately at fill time
3. Delay EdgeSnapshot logging until markout completes

### Phase 3: Remove Tautology
1. Separate "spread capture" (distance from mid at placement) from "adverse selection" (mid movement after fill)
2. Metrics become:
   - **Edge** = spread_at_placement - AS_at_markout - fees
   - Not: spread_at_now - AS_at_now - fees (which ≈ 0)

---

## Next Steps

1. **Do NOT use edge_validation.jsonl data** until this is fixed
2. **RL rewards are currently training on noise** (realized_edge_bps ≈ 0)
3. **Calibration metrics are corrupted** (all tautological)
4. **Paper trading edge metrics are unreliable**
5. **The Feb 11 redesign's edge/AS measurements are fundamentally broken**

This explains why:
- AS-based floor optimization doesn't improve
- RL agent can't learn meaningful reward signal
- Edge prediction models collapse to zero
- Paper trading Sharpe looks good but realizes poorly on live (different market structure)

