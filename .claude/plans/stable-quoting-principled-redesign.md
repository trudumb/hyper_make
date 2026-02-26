# Stable Quoting: Principled Redesign

## Diagnosis (from live ETH trading on hyna DEX)

**Symptom**: 99:1 cancel/fill ratio. Every cycle: cancel both orders, place 2 new ones. Zero fills in 7+ minutes. ChurnTracker correctly alerting at cycle 50.

**Root cause chain**:
1. Kappa cap formula `(2/kappa * 10000).max(1.5)` produces 2.9 bps/side with kappa=2457
2. At sigma=0.000161, ETH mid moves ~1.75 bps per 5s cycle (empirical from 15 consecutive cycles)
3. After 2-3 cycles, accumulated drift (3.5-5.25 bps) exceeds latch threshold (3.0-5.0 bps)
4. Result: cancel+place every 2-3 cycles, sometimes every cycle
5. Zero modifies because modify path only handles "same price, size decrease" — never fires
6. Smoother disabled, PriceGrid disabled — L1 and L2 layers not active

**The fundamental violation**: The spread (2.9 bps/side) is NARROWER than the per-cycle noise floor (1.75 bps). This guarantees churn because orders can never rest long enough for tolerance bands to help.

## Design Principle

**Stability Floor**: `half_spread >= 2 × expected_mid_movement_per_cycle`

This ensures orders survive at least 2 cycles without action, giving tolerance bands and latching room to work.

## Fixes (ordered by impact)

### Fix 1: Stability-Aware Kappa Cap (HIGH IMPACT)
`ladder_strat.rs`: Replace `.max(1.5)` kappa cap floor with sigma-derived stability floor.

```
stability_floor_bps = max(5.0, 2.0 × sigma × √cycle_dt × 10000)
kappa_cap_effective = max(kappa_cap_bps, stability_floor_bps)
```

For ETH: `2.0 × 0.000161 × √5 × 10000 = 7.2 bps` → half-spread ≥ 7.2 bps → total ≥ 14.4 bps.
Within hip3 target (15-25 bps). Eliminates ~80% of churn at source.

### Fix 2: Price-Change Modify Path (MEDIUM IMPACT)
`tracking/order_manager/reconcile.rs`: Add Case 2b between latch and cancel+place:
- When order is matched within tolerance but price drift exceeds latch
- Use MODIFY to change price (and optionally size)
- Saves 1 API call vs cancel+place (50% reduction per action)
- Loses queue at old price level, but gains queue at new level immediately

### Fix 3: Enable Parameter Smoother (LOW-MEDIUM IMPACT)
Wire `SmootherConfig { enabled: true }` for sim/paper mode.
Kappa EWMA with 5% deadband suppresses ~80% of parameter jitter.
Already built and tested (26 tests). Zero risk to enable in sim.

### Fix 4: Enable PriceGrid in Sim (LOW IMPACT)
Wire `PriceGridConfig { enabled: true }` for sim/paper mode.
Directional snap (bids floor, asks ceil) absorbs sub-tick oscillations.
Already built and tested. Zero risk in sim.

### Fix 5: Hard Kappa Cap Floor (SAFETY NET)
Change `.max(1.5)` to `.max(5.0)` as absolute floor regardless of sigma.
Prevents spread-narrower-than-fee scenarios on any asset.

## Expected Impact

| Metric | Before | After Fix 1 | After All |
|--------|--------|-------------|-----------|
| Half-spread | 2.9 bps | 7.2 bps | 7.2 bps |
| Drift per cycle | 1.75 bps | 1.75 bps | 0.35 bps (smoothed) |
| Cycles before action | 1-2 | 3-4 | 8-15 |
| API calls/min | ~12 | ~4 | ~1-2 |
| Cancel/fill ratio | 99:0 | <5:1 | <2:1 |

## Deferred Work (next session)
- Zone-based reconciler (replace target-matching with zone validation)
- Action prioritizer with per-cycle budget
- ChurnTracker → stability floor feedback loop
- Adaptive cycle interval (slow down when stable, speed up on regime change)
