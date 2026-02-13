Mainnet Validation Report — HYPE 1h Session (2026-02-13)                                                                                                                      
                                                                                                                                                                                                                              Duration: 20:10:57 → 21:08+ UTC (~57 min active)                                                                                                                                                                          
  Log: 17,254 lines | Fills: 21 | Errors: 47                                                                                                                                                                                                                                                                                                                                                                                                              Final Metrics                                                                                                                                                                                                                                                                                                                                                                                                                                           ┌────────────────────┬────────────────────────────────┐                                                                                                                                                                     │       Metric       │             Value              │                                                                                                                                                                   
  ├────────────────────┼────────────────────────────────┤
  │ Realized PnL       │ -$0.14                         │
  ├────────────────────┼────────────────────────────────┤
  │ Unrealized PnL     │ -$0.06                         │
  ├────────────────────┼────────────────────────────────┤
  │ Total PnL          │ -$0.25                         │
  ├────────────────────┼────────────────────────────────┤
  │ Final Position     │ -1.52 HYPE                     │
  ├────────────────────┼────────────────────────────────┤
  │ Avg AS per fill    │ 4.65 bps                       │
  ├────────────────────┼────────────────────────────────┤
  │ Sharpe (1h)        │ 342.96 (PRELIMINARY, 21 fills) │
  ├────────────────────┼────────────────────────────────┤
  │ Edge               │ 10.3 bps                       │
  ├────────────────────┼────────────────────────────────┤
  │ Avg total spread   │ 31.7 bps                       │
  ├────────────────────┼────────────────────────────────┤
  │ Avg optimal spread │ 13.14 bps                      │
  └────────────────────┴────────────────────────────────┘

  Feature Validation Scorecard

  ┌─────┬──────────────────────┬─────────┬──────────────────────────────────────────────────────────────────────────────────────┐
  │  #  │       Feature        │ Status  │                                       Evidence                                       │
  ├─────┼──────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 1   │ Gamma routing        │ WORKING │ 252 cycles at 3.00x, 73 at 2.00x, 159 at 1.20x (regime-dependent)                    │
  ├─────┼──────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 2   │ Cost-basis clamp     │ WORKING │ 412 clamp events, entry/breakeven tracked, urgency < 0.5 prevents selling below cost │
  ├─────┼──────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 3   │ HMM stabilization    │ PARTIAL │ First regime = Calm (good), but 45 transitions in 57 min = extreme oscillation       │
  ├─────┼──────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 4   │ Capital-aware sizing │ WORKING │ Concentration fallback, per-level cap at 25% of risk max                             │
  ├─────┼──────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 5   │ Tiered Sharpe        │ WORKING │ "PRELIMINARY" at 21 fills (correct tier)                                             │
  ├─────┼──────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ 6   │ PnL logging          │ WORKING │ Periodic [PnL] session summary every 35s with realized + unrealized                  │
  └─────┴──────────────────────┴─────────┴──────────────────────────────────────────────────────────────────────────────────────┘

  3 Critical Issues Found

  1. Regime Oscillation (45 transitions in 57 min)
  - After initial 10-min Calm period (good — stabilization buffer working), regime oscillates rapidly between Volatile/Extreme/Normal
  - Extreme regime dominates (252/484 = 52% of cycles at gamma_mult=3.00)
  - This forces wider spreads than necessary most of the time
  - Root cause: HMM is too sensitive to HYPE's microstructure. HYPE has thin liquidity → kappa naturally drops to ~500-1000 → HMM classifies as Extreme → kappa recovers → Normal → repeat
  - Fix needed: Regime stickiness / minimum dwell time (e.g., 2-5 min before transition allowed)

  2. Invalid Price Errors (43 occurrences, 20:21 - 20:28)
  - All "Order has invalid price" — concentrated in a 7-minute window during Extreme regime
  - The cost-basis breakeven clamp produces prices the exchange rejects
  - 1 additional "invalid size" error (0.205 → probably below HYPE's size tick)
  - Errors stopped after 20:28 because price moved enough that breakeven no longer triggers
  - Fix needed: Round breakeven prices to exchange tick grid, not just sig figs

  3. GLFT Calibration Gap
  - Optimal spread avg = 13.14 bps, but actual total spread avg = 31.7 bps
  - GLFT produces spreads too narrow for HYPE's actual volatility
  - sigma too low (~0.0001 vs real ~0.0014), kappa too high (~7000 vs effective ~1000-2000)
  - The gamma_mult partially compensates (3x in Extreme), but the base is wrong
  - Fix needed: Sigma estimation recalibration, kappa estimation anchoring to HYPE's actual fill rate

  Fill Analysis

  - 21 fills: 11 buys, 10 sells (balanced)
  - Position oscillated between -1.52 and +1.80 (bounded by inventory governor)
  - 3 initial buys burst at 20:13:03 (likely queued fills)
  - Later fills show healthy 2-7 min spacing
  - AS ranged from 0 bps (first 3, no markout data yet) to 6.83 bps
  - Average AS = 4.65 bps vs edge of 10.3 bps → net positive edge of ~5.6 bps (healthy)

  Emergency Pulls: 138 events

  - Frequently triggered due to regime oscillation → emergency pull → wider bid/ask gap
  - Working correctly but too frequent (should be rare, not 28% of cycles)