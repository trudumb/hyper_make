# First Principles: Tight Spread Quoting

## The Fundamental Equation

```
E[P&L per trade] = spread_captured - adverse_selection - fees - inventory_cost
```

For tight spreads to work, you need:
1. HIGH fill rate (volume compensates for lower per-trade profit)
2. PREDICTABLE adverse selection (know when NOT to quote tight)
3. FAST quote updates (escape before toxic flow arrives)
4. NEUTRAL inventory (minimize holding risk)

---

## Core Stochastic Models Required

### 1. Price Process

**Jump-Diffusion (Merton)**
```
dS = μSdt + σSdW + J×dN(λ)
```
Captures sudden moves with Poisson jumps.

**Hawkes Process (Self-Exciting)**
```
λ(t) = λ₀ + Σ α×exp(-β(t-tᵢ))
```
Critical for understanding order flow clustering and cascades.

### 2. Volatility Estimation

**Bipower Variation (Jump-Robust)**
```
BV = (π/2) × Σ|r_t| × |r_{t-1}|
σ_clean = √BV  (diffusion component only)
jump_ratio = RV/BV  (>1.5 = toxic regime)
```

### 3. Adverse Selection

**Glosten-Milgrom**
```
Spread = 2 × α × E[|v - p| | trade]
```
Where α = probability trade is informed.

**Depth-Dependent AS**
```
AS(δ) = AS₀ × exp(-δ/δ_char)
```
Deeper quotes have less adverse selection.

### 4. Fill Probability

**First-Passage Time**
```
P(fill | δ, τ) = 2×Φ(-δ/(σ×√τ))
```
Probability Brownian motion reaches depth δ within time τ.

---

## GLFT Optimal Spread

```
δ* = (1/γ) × ln(1 + γ/κ) + fee
```

Parameters:
- γ = risk aversion (lower = tighter)
- κ = order flow intensity (higher = tighter)
- σ = volatility (higher = wider)

**Key Insight**: GLFT assumes constant parameters. Expert implementations make them DYNAMIC.

---

## Regime-Based Spread Selection

### Market Regimes
| Regime | σ Condition | jump_ratio | Min Spread |
|--------|-------------|------------|------------|
| Calm | σ < σ_baseline | < 1.2 | 3 bps |
| Normal | σ ≈ σ_baseline | 1.2-1.5 | 5 bps |
| Volatile | σ > 2×σ_baseline | 1.5-3.0 | 8 bps |
| Cascade | N/A | > 3.0 | ∞ (pull) |

### Conditional Tight Quoting
```
Can_Quote_Tight = 
    (Regime == Calm) AND
    (Toxicity_Predicted < 0.1) AND
    (Hour NOT IN [6, 7, 14]) AND
    (Update_Latency < 50ms) AND
    (|Inventory| < 0.3 × max_position)
```

---

## Predictive Alpha Model

### Features for Toxicity Prediction
1. Book imbalance: (bid_qty - ask_qty) / total
2. Trade flow imbalance: net direction of last N trades
3. Volatility ratio: σ_current / σ_baseline
4. Time of day: one-hot for known toxic hours
5. Recent large trades: size > 2× average
6. Cross-asset signals: BTC movement for altcoins

### Model Output
```
P(adverse_selection > threshold | features)
```

If P > 0.3: WIDEN spreads
If P < 0.1: TIGHTEN spreads

---

## Speed Requirements

**Safe Tight Spread Formula**
```
δ_min = σ × √(2×τ_update) + fee + buffer
```

With τ_update = 50ms, σ = 1 bp/sec, fee = 3 bps:
```
δ_min = 0.0001 × √0.1 + 0.0003 ≈ 0.03 + 3 = 3.3 bps
```

**Implication**: To quote 3 bps, you need ~50ms update latency.

---

## Implementation Gaps in hyper_make

### Gap 1: Static Spread Floor
- Current: Fixed 8 bps in all conditions
- Target: 3-8 bps based on regime

### Gap 2: Reactive AS Measurement
- Current: Measures AS AFTER fills
- Target: PREDICT AS BEFORE quoting

### Gap 3: Quote Update Speed
- Current: Full reconciliation cycle
- Target: Fast path for price-only updates

### Gap 4: No Cross-Asset Signals
- Current: Single-asset
- Target: Use BTC as leading indicator

---

## Kelly-Optimal Sizing for Tight Spreads

```
f* = (edge - AS) / variance
```

Tight spread → small edge → smaller optimal size

This is why Kelly-Stochastic allocation reduces size at shallow depths:
the edge is thin, so bet small.

---

## Summary

Expert market makers don't quote tight ALL the time. They quote tight when:

1. **Regime is Calm**: Low vol, no jumps
2. **Toxicity is Predictably Low**: Model says safe
3. **They're Fast**: Can escape before picked off
4. **Inventory is Neutral**: No directional risk

The 8 bps floor is CORRECT for general operation. Tight spreads require CONDITIONAL activation based on real-time regime and toxicity assessment.
