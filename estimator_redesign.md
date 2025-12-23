# Estimator Redesign: Dual-Sigma & Flow Imbalance Architecture

## Executive Summary

The current estimator has a fundamental flaw: it uses Bipower Variation (BV) for volatility, which **by design filters out jumps**. This creates a paradox—jumps cause adverse selection, but the estimator ignores them.

**Result**: σ = 0.00018 during a crash, skew ≈ 0, quotes don't react.

## Current Architecture Problems

### Problem 1: Single Sigma for Dual Purposes

The GLFT model uses σ in two places with different requirements:

| Formula | Needs | Current Implementation |
|---------|-------|----------------------|
| Half-spread: `ψ = (1/γ) × ln(1 + γ/κ)` | Clean σ (continuous risk) | ✓ BV is correct |
| Inventory skew: `skew = (q/Q) × γ × σ² / κ` | Total σ (includes jumps) | ✗ BV filters out jumps |

**Solution**: Provide `sigma_clean` (BV) and `sigma_total` (RV) separately.

### Problem 2: No Directional Flow Detection

RV/BV ratio detects jump *magnitude* but not *direction*. A +3% then -3% move has the same ratio as a -6% crash.

**Solution**: Add signed momentum/flow imbalance metrics.

### Problem 3: Volume Clock Latency

VWAP buckets complete every few seconds. During a crash, you're quoting based on stale information.

**Solution**: Add high-frequency (100ms) trade-level metrics alongside volume clock.

### Problem 4: EWMA Half-Life Too Slow

With half_life = 50 ticks at 0.5 ticks/sec = 100 seconds of memory. Crashes happen in milliseconds.

**Solution**: Multi-timescale estimators with fast/slow components.

---

## Redesigned Architecture

### 1. Dual Volatility Signals

```rust
pub struct VolatilityEstimates {
    /// Clean volatility (√BV) - for base spread pricing
    /// Robust to jumps, measures continuous diffusion
    pub sigma_clean: f64,
    
    /// Total volatility (√RV) - for inventory risk/skew
    /// Includes jumps, measures actual price variance
    pub sigma_total: f64,
    
    /// Jump intensity: RV/BV ratio (1.0 = normal, >2.0 = jumps)
    pub jump_ratio: f64,
    
    /// Effective sigma for skew: blend based on regime
    /// In normal: uses sigma_clean. In toxic: uses sigma_total
    pub sigma_effective: f64,
}

impl VolatilityEstimates {
    pub fn compute_effective(&mut self) {
        // Blend: use more of sigma_total when jump_ratio is elevated
        // At ratio=1: weight=0 (pure clean)
        // At ratio=3: weight=0.67 (mostly total)
        // At ratio=5: weight=0.8 (almost pure total)
        let jump_weight = 1.0 - (1.0 / self.jump_ratio);
        let jump_weight = jump_weight.clamp(0.0, 0.9);
        
        self.sigma_effective = (1.0 - jump_weight) * self.sigma_clean 
                             + jump_weight * self.sigma_total;
    }
}
```

### 2. Signed Flow Imbalance Detector

Tracks directional momentum, not just magnitude:

```rust
/// Tracks signed cumulative returns over multiple windows
pub struct FlowImbalanceDetector {
    /// Recent signed returns: (timestamp_ms, log_return)
    returns: VecDeque<(u64, f64)>,
    
    /// Window durations in ms
    windows_ms: Vec<u64>,  // e.g., [100, 500, 2000]
}

impl FlowImbalanceDetector {
    /// Compute signed momentum for each window
    /// Returns: Vec of (window_ms, signed_momentum_bps)
    pub fn momentum(&self, now_ms: u64) -> Vec<(u64, f64)> {
        self.windows_ms.iter().map(|&window| {
            let cutoff = now_ms.saturating_sub(window);
            let sum: f64 = self.returns.iter()
                .filter(|(t, _)| *t >= cutoff)
                .map(|(_, r)| r)
                .sum();
            (window, sum * 10_000.0)  // Convert to bps
        }).collect()
    }
    
    /// Is the market in a falling knife pattern?
    /// Returns severity: 0.0 = normal, 1.0 = severe downward momentum
    pub fn falling_knife_score(&self, now_ms: u64) -> f64 {
        let momenta = self.momentum(now_ms);
        
        // Check if ALL windows show negative momentum (consistent selling)
        let all_negative = momenta.iter().all(|(_, m)| *m < 0.0);
        if !all_negative { return 0.0; }
        
        // Score based on magnitude of short-term momentum
        let short_term = momenta.iter()
            .find(|(w, _)| *w <= 500)
            .map(|(_, m)| m.abs())
            .unwrap_or(0.0);
        
        // 20 bps in 500ms = score 1.0
        (short_term / 20.0).clamp(0.0, 2.0)
    }
}
```

### 3. Trade-Level Imbalance (Tick-by-Tick)

Track buy vs sell aggression without waiting for volume buckets:

```rust
/// High-frequency order flow imbalance from trade tapes
pub struct TradeFlowTracker {
    /// Recent trades: (timestamp_ms, signed_volume)
    /// Positive = buy aggressor, Negative = sell aggressor
    trades: VecDeque<(u64, f64)>,
    
    /// Rolling window for imbalance calculation
    window_ms: u64,  // e.g., 1000ms
    
    /// EWMA of imbalance for smoothing
    ewma_imbalance: f64,
    alpha: f64,
}

impl TradeFlowTracker {
    /// Add a trade. is_buy_aggressor from trade tape.
    pub fn on_trade(&mut self, timestamp_ms: u64, size: f64, is_buy_aggressor: bool) {
        let signed = if is_buy_aggressor { size } else { -size };
        self.trades.push_back((timestamp_ms, signed));
        
        // Expire old trades
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self.trades.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
            self.trades.pop_front();
        }
        
        // Update EWMA
        let instant_imbalance = self.compute_imbalance();
        self.ewma_imbalance = self.alpha * instant_imbalance 
                            + (1.0 - self.alpha) * self.ewma_imbalance;
    }
    
    /// Compute current imbalance: (buy_vol - sell_vol) / total_vol
    /// Range: [-1, 1] where -1 = all selling, +1 = all buying
    fn compute_imbalance(&self) -> f64 {
        let (buy_vol, sell_vol) = self.trades.iter().fold((0.0, 0.0), |(b, s), (_, v)| {
            if *v > 0.0 { (b + v, s) } else { (b, s - v) }
        });
        let total = buy_vol + sell_vol;
        if total < 1e-9 { return 0.0; }
        (buy_vol - sell_vol) / total
    }
    
    /// Get smoothed imbalance
    pub fn imbalance(&self) -> f64 {
        self.ewma_imbalance
    }
    
    /// Is there toxic selling? (strong sell imbalance)
    pub fn is_toxic_selling(&self) -> bool {
        self.ewma_imbalance < -0.3  // >65% of volume is sell aggressor
    }
}
```

### 4. Multi-Timescale Variance (Fast + Slow)

Instead of one EWMA, track multiple timescales:

```rust
/// Multi-timescale variance estimator
pub struct MultiScaleVariance {
    /// Fast variance (reacts in ~5 ticks / ~2 seconds)
    fast: EwmaVariance,
    
    /// Medium variance (reacts in ~20 ticks / ~10 seconds)
    medium: EwmaVariance,
    
    /// Slow variance (reacts in ~100 ticks / ~60 seconds)
    slow: EwmaVariance,
}

struct EwmaVariance {
    half_life_ticks: f64,
    alpha: f64,
    rv: f64,  // Realized variance (includes jumps)
    bv: f64,  // Bipower variation (excludes jumps)
    last_abs_return: Option<f64>,
}

impl MultiScaleVariance {
    pub fn new() -> Self {
        Self {
            fast: EwmaVariance::new(5.0),    // ~2 seconds
            medium: EwmaVariance::new(20.0), // ~10 seconds
            slow: EwmaVariance::new(100.0),  // ~60 seconds
        }
    }
    
    pub fn on_return(&mut self, log_return: f64) {
        self.fast.update(log_return);
        self.medium.update(log_return);
        self.slow.update(log_return);
    }
    
    /// Composite sigma: uses fast when elevated, blends to slow otherwise
    pub fn adaptive_sigma(&self) -> f64 {
        let fast_sigma = self.fast.rv.sqrt();
        let slow_sigma = self.slow.rv.sqrt();
        
        // If fast >> slow, market is accelerating - use fast
        // If fast ≈ slow, market is stable - use slow for stability
        let ratio = fast_sigma / slow_sigma.max(1e-9);
        
        if ratio > 2.0 {
            // Acceleration detected - weight toward fast
            let weight = ((ratio - 1.0) / 2.0).clamp(0.0, 0.8);
            weight * fast_sigma + (1.0 - weight) * slow_sigma
        } else {
            // Stable - use slow for less noise
            0.3 * fast_sigma + 0.7 * slow_sigma
        }
    }
    
    /// Jump ratio at each timescale
    pub fn jump_ratios(&self) -> (f64, f64, f64) {
        (self.fast.jump_ratio(), self.medium.jump_ratio(), self.slow.jump_ratio())
    }
}
```

---

## Revised MarketParams Structure

```rust
pub struct MarketParams {
    // === Volatility (all per-second, NOT annualized) ===
    
    /// Clean volatility (√BV) - use for base spread
    pub sigma_clean: f64,
    
    /// Total volatility (√RV) - use for skew/risk
    pub sigma_total: f64,
    
    /// Effective sigma for skew (blended based on regime)
    pub sigma_effective: f64,
    
    // === Order Book ===
    
    /// Order book depth decay constant
    pub kappa: f64,
    
    /// Order arrival intensity (volume ticks/sec)
    pub arrival_intensity: f64,
    
    // === Regime Detection ===
    
    /// RV/BV jump ratio (1.0 = normal, >2.0 = jumps)
    pub jump_ratio: f64,
    
    /// Is market in toxic regime?
    pub is_toxic_regime: bool,
    
    // === NEW: Directional Flow ===
    
    /// Signed momentum (bps over last 500ms)
    /// Negative = market falling
    pub momentum_bps: f64,
    
    /// Order flow imbalance [-1, 1]
    /// Negative = sell pressure dominant
    pub flow_imbalance: f64,
    
    /// Falling knife score [0, 2]
    /// >1.0 = dangerous downward momentum
    pub falling_knife_score: f64,
}
```

---

## Revised Strategy Integration

### GLFTStrategy Changes

```rust
impl QuotingStrategy for GLFTStrategy {
    fn calculate_quotes(&self, ...) -> (Option<Quote>, Option<Quote>) {
        // 1. Use sigma_clean for base half-spread (continuous pricing)
        let half_spread = self.half_spread(gamma, kappa);
        
        // 2. Use sigma_effective for skew (includes jump risk when elevated)
        let sigma_for_skew = market_params.sigma_effective;
        let gamma_for_skew = self.derive_gamma(target_half_spread, kappa, sigma_for_skew, max_position);
        let base_skew = self.inventory_skew(inventory_ratio, sigma_for_skew, gamma_for_skew, kappa);
        
        // 3. Additional directional adjustment
        let directional_adjustment = self.compute_directional_adjustment(market_params, half_spread);
        
        // 4. Falling knife protection
        if market_params.falling_knife_score > 1.0 {
            // Severe downward momentum - protect the bid
            let protection = market_params.falling_knife_score * half_spread;
            bid_delta += protection;  // Push bid further away
            
            // If we're already long, this is an emergency
            if inventory_ratio > 0.0 {
                // Consider pulling bid entirely
                // Or: bid_delta *= 2.0;
            }
        }
        
        // ... rest of calculation
    }
    
    fn compute_directional_adjustment(&self, params: &MarketParams, half_spread: f64) -> f64 {
        // Flow imbalance adjustment:
        // If sell pressure (-0.5 imbalance), shift both quotes down slightly
        // This anticipates continued selling pressure
        let flow_adj = params.flow_imbalance * half_spread * 0.3;
        
        // Momentum adjustment:
        // If momentum is -30 bps over 500ms, anticipate further decline
        let momentum_adj = (params.momentum_bps / 100.0) * half_spread;
        
        flow_adj + momentum_adj
    }
}
```

---

## Concrete Fix for Your 88018 Crash

With the redesigned system, here's what would have happened:

**Before the fill (19:44:21)**:
- `sigma_total` = 0.0012 (120 bps/sec - includes the crash)
- `sigma_clean` = 0.00018 (18 bps/sec - filtered)
- `sigma_effective` = 0.0009 (blended due to elevated jump_ratio)
- `momentum_bps` = -40 (falling 40 bps in 500ms)
- `flow_imbalance` = -0.6 (heavy selling)
- `falling_knife_score` = 1.5 (severe)

**Quote calculation**:
```
γ_for_skew = 0.00105 × 100 / (1.0 × 0.0009²) = 130 (not clamped!)
skew = 0.23 × 130 × 0.0009² / 100 = 0.000024 = 2.4 bps (meaningful!)

falling_knife_protection = 1.5 × 5.25 bps = 7.9 bps additional bid distance

Total bid distance from mid: 5.25 + 2.4 + 7.9 = 15.5 bps = $13.65
Bid would be at: 88057.5 - 13.65 = 88043.85

The fill at 88018 would NOT have happened because your bid would be at 88044,
not 88018. The aggressive seller would have walked the book past you.
```

---

## Implementation Priority

1. **Dual Sigma** (sigma_clean + sigma_total): Core fix, 1-2 hours
2. **Momentum Detector**: Add 500ms signed return tracking, 1 hour  
3. **Trade Flow Imbalance**: Requires is_buy_aggressor from trade tape, 2 hours
4. **Multi-Timescale Variance**: Refactor existing EWMA, 2 hours
5. **Strategy Integration**: Wire new params into GLFT, 1 hour

---

## Testing the Fix

### Unit Test: Crash Scenario

```rust
#[test]
fn test_falling_knife_protection() {
    let mut estimator = ParameterEstimator::new(config);
    
    // Simulate crash: VWAP drops 88053 → 88029 → 88018
    let vwaps = [88053.0, 88029.0, 88018.0];
    for (i, vwap) in vwaps.iter().enumerate() {
        estimator.on_bucket(i as u64 * 200, *vwap, 0.01);
    }
    
    let params = estimator.market_params();
    
    // sigma_total should be elevated (includes crash)
    assert!(params.sigma_total > 0.0005, "sigma_total should reflect crash");
    
    // sigma_clean should be lower (filters crash)
    assert!(params.sigma_clean < params.sigma_total);
    
    // Momentum should be negative
    assert!(params.momentum_bps < -20.0, "Should detect negative momentum");
    
    // Falling knife should be triggered
    assert!(params.falling_knife_score > 1.0, "Should detect falling knife");
}
```

---

## Summary

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Tiny σ during crash | BV filters jumps | Dual sigma: use RV for skew |
| No skew reaction | σ² → 0 | sigma_effective blends in RV during jumps |
| Quotes on wrong side | No directional info | Momentum + flow imbalance |
| Stale quotes | Volume clock latency | Trade-level tracking + multi-timescale |
| Slow adaptation | 50-tick half-life | Fast/slow EWMA blend |
