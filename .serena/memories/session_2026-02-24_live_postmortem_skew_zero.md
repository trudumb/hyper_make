# Session: Feb 24 Live Post-Mortem — skew_bps = 0.00

## What Happened
- 25-min HYPE session on hyna DEX, $100 capital, 10x leverage
- 57 fills, ended -8.65 short while HYPE trended +38 bps up
- Total PnL: -$1.60 (realized +$0.31, unrealized -$1.69)
- Max drawdown: 494.1% relative. Kill switch did NOT trigger. Manual Ctrl+C.

## Root Causes (ranked)

### RC1: `skew_bps = 0.00` entire session
- `inventory_skew_bps = 0.0` hardcoded in signal_integration.rs:1072
- Zero GLFT reservation price shift — quotes symmetric regardless of inventory
- At -8.65 short in +38 bps uptrend, bids and asks equally distant from mid
- **This is THE fix** — GLFT formula produces `gamma * q * sigma^2 * T` skew but it's zeroed out

### RC2: Drift asymmetry negligible
- Kalman drift_adj_bps: -1.7 bps. Actual trend: +38 bps. Response: 4%.
- Drift uncertainty ~3.5 kept the signal damped
- Even the direction was sometimes wrong (negative drift while trend was up)

### RC3: Cascade sweeps twice in 4 minutes
- 13:29:53: 7 sells in <1s, position 2.04 → -4.55
- 13:34:02: 6 sells in <1s, position -3.67 → -7.54
- Fill cascade detector fired (WIDEN/SUPPRESS) but only for 15-30s
- Need sustained pullback after cascades, not just brief widening

### RC4: Size reduction ≠ price skew
- `Inventory pressure: size_mult=0.53` on asks — reduces new short exposure
- But doesn't help close existing short — bids not made more aggressive
- At 45% of max position, the system should be VERY asymmetric on price

## Key Data Points
- ESTIMATOR_DIAGNOSTICS: `skew_bps: 0.00` and `pos_guard_skew: 0.00` every single cycle
- Trend detection: `is_opposed: true` from 13:27 onward (2.5 min in), never recovered
- Touch spreads at end: bid=1.50 bps (floor!), ask=4.17 bps
- `continuation_p: 0.678` with `urgency_score: 3.3` — system KNEW it was opposed but couldn't act

## Fix Priority
1. Wire inventory skew in signal_integration.rs (unzero the skew_bps)
2. Amplify drift_adj_bps when is_opposed=true (20-50% of trend, not 4%)
3. Sustained cascade pullback (2-5 min, not 15-30s)
4. Position-proportional spread widening on the wrong side

## Validation
- Detailed log analysis in `.claude/plans/melodic-splashing-journal.md`
- Log file: `logs/mm_hip3_hyna_HYPE_hip3_2026-02-24_06-24-01.log`
