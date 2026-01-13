# Session: 2026-01-13 Spread Calculation Workflow Analysis

## Summary
Traced the complete spread calculation workflow from raw market data to final quote prices, creating a state diagram showing exactly why spreads are ~27 bps on mainnet BTC.

## Key Discoveries

### GLFT Formula Implementation
```
δ = (1/γ) × ln(1 + γ/κ) + f_maker
```
- Location: `quoting/ladder/depth_generator.rs:243-256`
- With γ=0.195, κ=4750: produces ~3.6 bps per side
- For large κ >> γ: formula approximates 1/κ (independent of gamma)

### Spread Calculation Chain (Observed: 26.85 bps)
```
GLFT → Floor → L2 Multiplier → Final
3.6 bps → 8.0 bps → ×1.69 → 13.5 bps/side (27 bps total)
```

### Gamma Calculation Path
- **File**: `strategy/ladder_strat.rs:345-372`
- gamma_base = 0.3 (config)
- Scaled by: calibration_scalar, tail_risk_mult, warmup factors
- Final: 0.195 (includes book_depth + warmup scaling)
- **Key insight**: Lower gamma → wider spreads in GLFT

### Kappa Selection Priority
- **File**: `strategy/ladder_strat.rs:374-399`
1. ROBUST (V3): if `use_kappa_robust` - **SELECTED**
2. ADAPTIVE: if adaptive mode
3. LEGACY: book_kappa × (1 - predicted_alpha)

### Kappa Orchestrator Blending
- **File**: `estimator/kappa_orchestrator.rs`
- Formula: own×0% + book×40% + robust×30% + prior×30%
- With own=2500, book=2500, robust=10000, prior=2500 → 4750

### L2 Uncertainty Multiplier
- **File**: `strategy/ladder_strat.rs:517-533`
- Bayesian uncertainty premium during warmup
- Formula: `spread_mult = 1.0 + uncertainty_scaling × (σ_μ / baseline)`
- At 10% warmup: ~1.69× multiplier

## State Diagram Summary
```
Market Data → Estimators → Gamma/Kappa → GLFT → Floor → L2 Mult → Quotes
   L2/Trades    σ,κ,μ      0.195/4750   3.6bp   8bp     ×1.69   13.5bp
```

## Files Modified
None - research/documentation task only

## Verification
```bash
grep "Ladder spread diagnostics" logs/mm_mainnet_BTC_*.log | tail -5
grep "Kappa orchestrator breakdown" logs/mm_mainnet_BTC_*.log | tail -5
```

## Key Insight
The 27 bps spread during early warmup (10%) is intentional:
- GLFT says 3.6 bps (very tight due to high kappa)
- Floor ensures profitability (8 bps minimum)
- L2 uncertainty protects during model calibration (×1.69)
- As warmup → 100%, spreads tighten automatically
