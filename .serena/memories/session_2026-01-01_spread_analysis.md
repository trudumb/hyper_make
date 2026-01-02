# Session: Spread Tightening Analysis

**Date**: 2026-01-01
**Duration**: ~15 minutes
**Topic**: Diagnosing why market maker has zero fills due to wide spreads

## Context

User ran market maker on BTC testnet with:
- `gamma_base = 0.5` (risk aversion)
- `min_depth_bps = 2`
- 4 minutes runtime → **0 fills**
- Fee reported as 3 bps

## Key Findings

### Root Cause: Spreads Too Wide

1. **min_spread_floor = 8 bps** (in RiskConfig, line 148)
   - Hard minimum half-spread
   - Total spread at touch: minimum 16 bps

2. **gamma_base = 0.5** (user setting)
   - Default is 0.3
   - GLFT formula: δ = (1/γ) × ln(1 + γ/κ) + fee
   - With γ=0.5, κ=500: ~9 bps half-spread

3. **Combined effect**: 16-20 bps total spread
   - Takers with 3-5 bps fee need 20+ bps price move
   - In low volatility: rarely fills

### Code Locations

| Parameter | Location | Default |
|-----------|----------|---------|
| `gamma_base` | `src/market_maker/strategy/risk_config.rs:132` | 0.3 |
| `min_spread_floor` | `src/market_maker/strategy/risk_config.rs:148` | 0.0008 (8 bps) |
| `maker_fee_rate` | `src/market_maker/strategy/risk_config.rs:151` | 0.00015 (1.5 bps) |
| `min_spread_floor_bps` | `src/market_maker/quoting/ladder/depth_generator.rs:183` | 8.0 |
| `fees_bps` | `src/market_maker/quoting/ladder/mod.rs:104` | 3.5 |

### GLFT Spread Sensitivity

| gamma_base | κ=300 | κ=500 | κ=1000 |
|------------|-------|-------|--------|
| 0.05 | 1.8 bps | 1.1 bps | 0.6 bps |
| 0.1 | 3.5 bps | 2.1 bps | 1.1 bps |
| 0.2 | 6.7 bps | 4.1 bps | 2.1 bps |
| 0.5 | 14.6 bps | 9.2 bps | 4.8 bps |

## Recommendations

### For Tighter Spreads

| Approach | gamma_base | min_spread_floor | Expected Spread |
|----------|------------|------------------|-----------------|
| Aggressive | 0.1 | 3 bps | 6-8 bps |
| Moderate | 0.2 | 5 bps | 10-12 bps |
| Conservative | 0.3 | 6 bps | 12-14 bps |

### Tradeoffs

- Tighter spread → More fills → More adverse selection risk
- 8 bps floor was set based on trade history analysis:
  - Large trades showed -11.6 bps adverse selection
  - Toxic hours (06-08, 14-15 UTC): -13 to -15 bps edge

## Deliverable

Created comprehensive analysis document:
`/home/jcritch22/projects/hyper_make/docs/tighter_spreads_analysis.md`

Covers:
- GLFT formula breakdown
- Parameter locations
- Risk tradeoffs
- Step-by-step action plan
- Prometheus metrics to monitor

## Technical Insights

1. **Fee configuration mismatch**: User says 3 bps, code has 1.5 bps maker fee
2. **Multiple spread floors**: Both RiskConfig and DynamicDepthConfig have 8 bps floor
3. **Gamma scaling**: Many factors multiply gamma (vol, toxicity, inventory, regime, hawkes, time-of-day)
4. **Log analysis**: Shows "Low buy capacity" warnings due to position limits, not spread issues

## Next Steps for User

1. Lower `--risk-aversion` CLI arg to 0.1-0.2
2. Modify `min_spread_floor` in config files to 3-5 bps
3. Update `maker_fee_rate` if actual fee is 3 bps (set to 0.0003)
4. Monitor with RUST_LOG=debug to see spread calculations
