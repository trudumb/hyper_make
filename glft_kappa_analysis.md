# GLFT Market Making: Theoretical Foundations and Implementation Corrections

## Executive Summary

The current implementation produces uncompetitive spreads (180-280 bps) because **κ is being estimated from the wrong data source**. The GLFT formula is mathematically correct, but κ represents fill rate decay (from trade execution data), not liquidity decay (from L2 book shape).

| Current State | Correct State |
|---------------|---------------|
| κ from L2 book: ~50-100 | κ from trade distribution: ~2000-5000 |
| Half-spread: 140-150 bps | Half-spread: 3-10 bps |
| Total spread: 280+ bps | Total spread: 6-20 bps |
| No fills | Competitive fills |

---

## Part 1: The GLFT Model from First Principles

### The Market Maker's Problem

You're a market maker quoting bid price $p^b$ and ask price $p^a$ around mid price $s$. You face a fundamental tradeoff:

- **Quote tight** → Get filled often, but eat adverse selection when prices move against you
- **Quote wide** → Better prices when filled, but rarely get filled

The goal: find the optimal spread that maximizes expected utility of terminal wealth, accounting for inventory risk from holding positions.

### The Fill Rate Model (Where κ Lives)

When you post a limit order at distance δ from mid, market orders arrive and potentially fill you. The GLFT model assumes fill rate follows an exponential decay:

```
λ(δ) = A × exp(-κ × δ)
```

Where:
- **λ(δ)** = rate at which you get filled when quoting at distance δ
- **A** = baseline arrival intensity of marketable orders  
- **κ** = sensitivity of fill rate to quote distance
- **δ** = distance from mid (as fraction, e.g., 0.0005 = 5 bps)

**Interpretation of κ:**
- **High κ (e.g., 2000-5000)**: Fill rate drops sharply with distance. You must quote tight to get filled. This is typical of liquid markets where trades cluster near mid.
- **Low κ (e.g., 50-200)**: Fill rate decays slowly. You can quote wide and still get filled. This would imply trades execute across a wide range of prices (rare in liquid markets).

### The Inventory Risk Model

While waiting for fills, you hold inventory $q$ exposed to price moves with volatility σ. Your **reservation price** (indifference mid) shifts away from the market mid:

```
r = s - q × γ × σ² × T
```

Where:
- **s** = market mid price
- **q** = current inventory (signed: positive = long, negative = short)
- **γ** = risk aversion parameter (higher = more conservative)
- **σ²** = price variance per unit time
- **T** = time horizon (or 1/λ for infinite horizon formulation)

This creates the **inventory skew**: when long, your reservation price drops (you want to sell), so you quote a tighter ask and wider bid.

### The Optimal Spread Solution

Solving the Hamilton-Jacobi-Bellman equation for expected utility maximization yields the optimal half-spread:

```
δ* = (1/γ) × ln(1 + γ/κ)
```

This elegant formula encodes the core tradeoff:

| Parameter | Effect on Spread | Intuition |
|-----------|------------------|-----------|
| Higher γ | Wider spread | More risk averse → demand more edge |
| Higher κ | Tighter spread | Fills decay fast → must quote close to mid |
| Lower κ | Wider spread | Fills available anywhere → can be selective |

---

## Part 2: Numerical Examples

### Example A: Liquid Market (Correct κ)

On a liquid BTC market, most trades execute within 2-5 bps of mid. This gives us:

```
Average trade distance = 3 bps = 0.0003
κ ≈ 1/0.0003 ≈ 3333
```

With γ = 0.5, κ = 3333:
```
δ* = (1/0.5) × ln(1 + 0.5/3333)
   = 2 × ln(1.00015)
   = 2 × 0.00015
   ≈ 0.0003 = 3 bps half-spread
```

**Result: 6 bps total spread — Competitive!**

### Example B: Current Implementation (Wrong κ)

The L2 book regression gives κ ≈ 70:

```
With γ = 0.57, κ = 70:
δ* = (1/0.57) × ln(1 + 0.57/70)
   = 1.75 × ln(1.0081)
   = 1.75 × 0.0081
   ≈ 0.0142 = 142 bps half-spread
```

**Result: 284 bps total spread — Uncompetitive, no fills**

### The Magnitude of the Error

The κ estimate is approximately **50x too low**, causing spreads to be **25-50x too wide**.

| κ Value | γ = 0.5 Half-Spread | Total Spread |
|---------|---------------------|--------------|
| 50 | 198 bps | 396 bps |
| 100 | 99 bps | 198 bps |
| 500 | 20 bps | 40 bps |
| 1000 | 10 bps | 20 bps |
| 2000 | 5 bps | 10 bps |
| 5000 | 2 bps | 4 bps |

---

## Part 3: The Estimation Error

### What the Current Code Measures

```rust
// L2 book regression: fits ln(size) ~ -κ × distance
// Model: L(δ) = A × exp(-κ × δ)
// Measures: "How does STANDING LIQUIDITY decay with distance?"
```

The code attempts to fit exponential decay to order book depth:
- Collects (distance, size) pairs from L2 book
- Fits weighted linear regression: ln(size) = a - κ × distance
- Extracts κ = -slope

**Problem**: In real order books, passive liquidity often **increases** with distance from mid:
- Competitive MMs crowd the BBO with small size
- Patient limit orders stack up further from mid
- Result: positive or flat slope → κ estimate is meaningless

When slope is positive (common), the code clamps to κ = 50, which then dominates the EWMA.

### What κ Actually Represents

```rust
// Fill rate model: λ(δ) = A × exp(-κ × δ)  
// Measures: "How does FILL PROBABILITY decay with distance?"
```

This is fundamentally different! It comes from analyzing **where trades actually execute**, not where orders sit:

- Trades on liquid markets cluster tightly around mid
- Very few trades execute 10+ bps from mid
- The distribution of trade distances tells us κ

### Visual Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    TWO DIFFERENT DISTRIBUTIONS                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  L2 BOOK SHAPE                      TRADE EXECUTION DISTANCE    │
│  ══════════════                     ════════════════════════    │
│                                                                 │
│  Size                               Trade Count                 │
│   │                                  │                          │
│   │    ████                          │██                        │
│   │   ██████                         │████                      │
│   │  ████████                        │██████                    │
│   │ ██████████                       │████████                  │
│   │████████████                      │██████████                │
│   └──────────────► Distance          └──────────────► Distance  │
│      from mid                           from mid                │
│                                                                 │
│  Liquidity INCREASES                 Fills CONCENTRATED         │
│  with distance (often)               near mid (always)          │
│                                                                 │
│  Slope: positive or flat             Slope: steep negative      │
│  Implied κ: ~50-100                  Implied κ: ~2000-5000      │
│                                                                 │
│  ❌ WRONG for GLFT                   ✓ CORRECT for GLFT        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Correct κ Estimation

### Theory: κ from Trade Distribution

For the exponential fill rate model λ(δ) = A × exp(-κδ), the mean of the distribution is:

```
E[δ] = 1/κ
```

Therefore:
```
κ = 1/E[δ] = 1/(volume-weighted average trade distance from mid)
```

### Implementation

```rust
/// Estimate κ from where trades actually execute relative to mid.
/// This is the FILL RATE decay parameter the GLFT formula needs.
#[derive(Debug)]
struct FillRateKappaEstimator {
    /// Rolling buffer of (distance, volume, timestamp)
    observations: VecDeque<(f64, f64, u64)>,
    window_ms: u64,
    
    /// Running sums for O(1) updates
    volume_weighted_distance: f64,
    total_volume: f64,
    
    /// EWMA smoothed kappa
    kappa: f64,
    alpha: f64,
    
    /// Minimum volume before estimates are valid
    min_volume: f64,
}

impl FillRateKappaEstimator {
    fn new(window_ms: u64, half_life_volume: f64, default_kappa: f64, min_volume: f64) -> Self {
        Self {
            observations: VecDeque::with_capacity(1000),
            window_ms,
            volume_weighted_distance: 0.0,
            total_volume: 0.0,
            kappa: default_kappa,
            alpha: (2.0_f64.ln() / half_life_volume).clamp(0.001, 0.5),
            min_volume,
        }
    }
    
    /// Process a trade execution
    fn on_trade(&mut self, timestamp_ms: u64, price: f64, size: f64, mid: f64) {
        if mid <= 0.0 || size <= 0.0 || price <= 0.0 {
            return;
        }
        
        // Distance from mid at time of trade (always positive)
        let distance = ((price - mid) / mid).abs();
        
        // Add observation
        self.observations.push_back((distance, size, timestamp_ms));
        self.volume_weighted_distance += distance * size;
        self.total_volume += size;
        
        // Expire old observations
        self.expire_old(timestamp_ms);
        
        // Update κ estimate
        self.update_kappa();
    }
    
    fn expire_old(&mut self, now: u64) {
        let cutoff = now.saturating_sub(self.window_ms);
        while let Some((dist, size, ts)) = self.observations.front() {
            if *ts < cutoff {
                self.volume_weighted_distance -= dist * size;
                self.total_volume -= size;
                self.observations.pop_front();
            } else {
                break;
            }
        }
        
        // Prevent numerical drift
        if self.observations.is_empty() {
            self.volume_weighted_distance = 0.0;
            self.total_volume = 0.0;
        }
    }
    
    fn update_kappa(&mut self) {
        if self.total_volume < self.min_volume {
            return;  // Not enough data
        }
        
        // Volume-weighted average distance
        let avg_distance = self.volume_weighted_distance / self.total_volume;
        
        // For exponential distribution: E[δ] = 1/κ
        // Therefore: κ = 1/E[δ]
        if avg_distance > 1e-8 {
            let kappa_instant = 1.0 / avg_distance;
            
            // Sanity bounds (100 = 100bps avg distance, 50000 = 0.2bps avg distance)
            let kappa_bounded = kappa_instant.clamp(100.0, 50000.0);
            
            self.kappa = self.alpha * kappa_bounded + (1.0 - self.alpha) * self.kappa;
        }
    }
    
    fn kappa(&self) -> f64 {
        self.kappa
    }
    
    fn is_valid(&self) -> bool {
        self.total_volume >= self.min_volume
    }
    
    fn avg_trade_distance_bps(&self) -> f64 {
        if self.total_volume < 1e-9 {
            return 0.0;
        }
        (self.volume_weighted_distance / self.total_volume) * 10000.0
    }
}
```

### Expected Results

| Market Condition | Avg Trade Distance | κ | GLFT Half-Spread (γ=0.5) |
|------------------|-------------------|---|--------------------------|
| Very liquid | ~2 bps | ~5000 | ~2 bps |
| Liquid | ~3 bps | ~3333 | ~3 bps |
| Normal | ~5 bps | ~2000 | ~5 bps |
| Moderate | ~10 bps | ~1000 | ~10 bps |
| Thin | ~25 bps | ~400 | ~25 bps |

The formula naturally produces competitive spreads when fed correct inputs.

---

## Part 5: Proper Use of L2 Order Book Data

The L2 book is valuable, but for different purposes than κ estimation.

### 1. Book Imbalance → Directional Skew

```rust
/// Asymmetric depth signals directional pressure
/// Positive = more bids = buying pressure = lean ask tighter
fn book_imbalance(bids: &[(f64, f64)], asks: &[(f64, f64)], levels: usize) -> f64 {
    let bid_depth: f64 = bids.iter().take(levels).map(|(_, sz)| sz).sum();
    let ask_depth: f64 = asks.iter().take(levels).map(|(_, sz)| sz).sum();
    
    let total = bid_depth + ask_depth;
    if total < 1e-9 { return 0.0; }
    
    (bid_depth - ask_depth) / total  // Range: [-1, 1]
}

// Usage: adjust skew, not spread width
let imbalance = book_imbalance(&bids, &asks, 5);
let imbalance_skew = imbalance * max_imbalance_skew_bps * 0.0001;
```

### 2. Near-Touch Liquidity → Gamma Scaling

```rust
/// Thin book near touch = higher adverse selection risk = scale γ up
fn liquidity_within_bps(
    bids: &[(f64, f64)], 
    asks: &[(f64, f64)], 
    mid: f64, 
    max_bps: f64
) -> f64 {
    let max_dist = max_bps * 0.0001;
    
    let bid_liq: f64 = bids.iter()
        .take_while(|(px, _)| (mid - px) / mid <= max_dist)
        .map(|(_, sz)| sz)
        .sum();
    
    let ask_liq: f64 = asks.iter()
        .take_while(|(px, _)| (px - mid) / mid <= max_dist)
        .map(|(_, sz)| sz)
        .sum();
    
    bid_liq + ask_liq
}

// Usage: scale gamma when book is thin
fn gamma_liquidity_multiplier(near_liq: f64, reference_liq: f64) -> f64 {
    if near_liq >= reference_liq {
        1.0
    } else {
        let ratio = reference_liq / near_liq.max(0.001);
        ratio.sqrt().clamp(1.0, 3.0)  // Cap at 3x
    }
}
```

### 3. Queue Depth → Fill Time Estimation

```rust
/// How much size ahead of us at BBO?
/// Deep queue = lower fill probability = may need tighter quotes
fn queue_at_touch(bids: &[(f64, f64)], asks: &[(f64, f64)]) -> (f64, f64) {
    let bid_queue = bids.first().map(|(_, sz)| *sz).unwrap_or(0.0);
    let ask_queue = asks.first().map(|(_, sz)| *sz).unwrap_or(0.0);
    (bid_queue, ask_queue)
}
```

### 4. Market Spread → Sanity Check

```rust
/// Current market spread as competitive reference
fn market_spread_bps(bids: &[(f64, f64)], asks: &[(f64, f64)]) -> Option<f64> {
    let best_bid = bids.first()?.0;
    let best_ask = asks.first()?.0;
    let mid = (best_bid + best_ask) / 2.0;
    Some((best_ask - best_bid) / mid * 10000.0)
}
```

### Summary: L2 Book Usage

| L2 Data | Use For | Do NOT Use For |
|---------|---------|----------------|
| Bid/ask imbalance | Directional skew adjustment | Spread width |
| Near-touch liquidity | γ scaling (thin book risk) | κ estimation |
| Queue depth | Fill time estimates | Spread calculation |
| Market spread | Sanity check / reference | Our spread |

---

## Part 6: Revised System Architecture

```
═══════════════════════════════════════════════════════════════════════════════
                           PARAMETER ESTIMATION PIPELINE
═══════════════════════════════════════════════════════════════════════════════

TRADE TAPE ──────────────────────────────────────────────────────────────────┐
    │                                                                         │
    ├──► Volume Clock                                                         │
    │       └──► VWAP per bucket                                             │
    │             └──► Log returns                                           │
    │                   └──► Bipower Variation ──────────────────► σ (vol)   │
    │                                                                         │
    ├──► Trade Price vs Mid                                                   │
    │       └──► Distance distribution                                       │
    │             └──► Volume-weighted mean                                  │
    │                   └──► κ = 1/E[distance] ──────────────────► κ (fill)  │
    │                                                                         │
    ├──► Volume Bucket Timestamps                                             │
    │       └──► Inter-arrival times                                         │
    │             └──► EWMA of rate ─────────────────────────────► λ (arrvl) │
    │                                                                         │
    └──► Aggressor Side (if available)                                        │
            └──► Buy vs Sell volume                                          │
                  └──► Flow imbalance ───────────────────► flow_skew (dir)   │
                                                                              │
═══════════════════════════════════════════════════════════════════════════════
                                GLFT FORMULA
═══════════════════════════════════════════════════════════════════════════════
                                                                              
    INPUTS:                                                                   
    ├── γ (risk aversion, possibly scaled by regime/inventory)               
    ├── κ (fill rate decay from trade distribution)                          
    ├── σ (volatility from bipower variation)                                
    ├── λ (arrival intensity) → T = 1/λ                                      
    └── q (current inventory)                                                
                                                                              
    CALCULATIONS:                                                             
    ├── Half-Spread:    δ* = (1/γ) × ln(1 + γ/κ)                             
    └── Inventory Skew: Δ = (q/Q_max) × γ × σ² × T                           
                                                                              
═══════════════════════════════════════════════════════════════════════════════
                                     │
                                     ▼
L2 ORDER BOOK ───────────────────────────────────────────────────────────────┐
    │                                                                         │
    ├──► Book Imbalance (bid vs ask depth)                                   │
    │       └──► Directional lean ───────────────► imbalance_skew            │
    │                                                                         │
    ├──► Near-Touch Liquidity (depth within X bps)                           │
    │       └──► Thin book detector ─────────────► γ multiplier              │
    │                                                                         │
    ├──► Queue Depth at BBO                                                   │
    │       └──► Fill time estimate ─────────────► (informational)           │
    │                                                                         │
    └──► Current Market Spread                                                │
            └──► Sanity check ───────────────────► (monitoring)              │
                                                                              │
═══════════════════════════════════════════════════════════════════════════════
                               FINAL QUOTES
═══════════════════════════════════════════════════════════════════════════════

    bid = mid - δ* - inventory_skew - imbalance_skew - flow_skew
    ask = mid + δ* - inventory_skew + imbalance_skew + flow_skew

═══════════════════════════════════════════════════════════════════════════════
```

---

## Part 7: Implementation Checklist

### Remove/Replace

- [ ] Remove `WeightedKappaEstimator` (L2 book regression)
- [ ] Remove kappa estimation from `on_l2_book()`
- [ ] Remove hardcoded κ = 50 fallback for positive slopes

### Add

- [ ] Add `FillRateKappaEstimator` (trade distance distribution)
- [ ] Integrate κ update into `on_trade()` flow
- [ ] Add `avg_trade_distance_bps()` for monitoring
- [ ] Add validation that κ is in reasonable range (100-50000)

### Modify

- [ ] Update `ParameterEstimator` to use trade-based κ
- [ ] Update logging to show avg_trade_distance alongside κ
- [ ] Update config to have sensible κ defaults (~2000 for liquid markets)

### Preserve

- [ ] Keep L2 book subscription (needed for imbalance, liquidity checks)
- [ ] Keep book imbalance calculation (use for skew)
- [ ] Keep near-touch liquidity calculation (use for γ scaling)

---

## Part 8: Validation

### Expected Log Output After Fix

```
{"message":"Kappa from trade distribution",
 "trades_in_window": 847,
 "avg_distance_bps": 3.2,
 "kappa": 3125,
 "kappa_ewma": 2890}

{"message":"GLFT spread components",
 "gamma": 0.57,
 "kappa": 2890,
 "half_spread_bps": 3.5,
 "inventory_skew_bps": 0.2,
 "imbalance_skew_bps": 0.8}

{"message":"Final quotes",
 "bid": 87523.0,
 "ask": 87530.0,
 "spread_bps": 8.0}
```

### Competitive Benchmarks

| Metric | Before Fix | After Fix | Target |
|--------|------------|-----------|--------|
| κ estimate | 50-100 | 2000-5000 | Market-derived |
| Half-spread | 140-150 bps | 3-8 bps | Competitive |
| Total spread | 280-300 bps | 6-16 bps | < 20 bps |
| Fill rate | ~0 | Normal | Regular fills |

---

## Conclusion

The GLFT formula is correct. The implementation error was using L2 book shape to estimate κ, which measures something completely different from fill rate decay. By estimating κ from the distribution of trade execution distances, the formula will naturally produce competitive spreads without any hardcoded floors or overrides.

The key insight: **κ answers "how quickly does my fill probability decay as I quote wider?" — and that question can only be answered by observing where trades actually execute.**
