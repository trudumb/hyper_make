# Level 1 Architectural Refactor: Centralized BeliefState

**Goal**: Replace fragmented belief modules with single source of truth
**Status**: Phase 1 Complete (2026-02-02)
**Remaining Effort**: 1-3 weeks

---

## Progress Summary

### Phase 1: Create belief/ module with structs ✅ COMPLETE

**Files Created:**
```
src/market_maker/belief/
├── mod.rs              # Module exports + Regime enum
├── central.rs          # CentralBeliefState struct (main orchestrator)
├── messages.rs         # BeliefUpdate enum for all update types
├── publisher.rs        # BeliefPublisher trait for estimators
└── snapshot.rs         # BeliefSnapshot and component belief structs
```

**Key Components Implemented:**
- `CentralBeliefState` - Thread-safe orchestrator with RwLock
- `CentralBeliefConfig` - Configuration (hip3, liquid presets)
- `BeliefSnapshot` - Point-in-time read-only view
- `BeliefUpdate` - Enum for all update types
- `BeliefPublisher` trait + `PublisherHandle` helper
- Sub-structs: DriftVolatilityBeliefs, KappaBeliefs, ContinuationBeliefs, RegimeBeliefs, ChangepointBeliefs, EdgeBeliefs, CalibrationState

**Tests:** 54 tests passing (`cargo test belief --lib`)

---

## Remaining Phases

### Phase 2: Add CentralBeliefState to MarketMaker ⬜ TODO

**Files to modify:**
- `src/market_maker/mod.rs` - Add `central_beliefs: CentralBeliefState` field
- `src/market_maker/core/mod.rs` - Add to component bundles

**Tasks:**
1. Add `central_beliefs` field to `MarketMaker<S, E>` struct
2. Initialize in `MarketMaker::new()` with appropriate config
3. Wire up channel for async updates (optional)

### Phase 3: Dual-write (update both old + new) ⬜ TODO

**Files to modify:**
- `src/market_maker/orchestrator/quote_engine.rs`
- `src/market_maker/orchestrator/handlers.rs`

**Tasks:**
1. On price updates, also call `central_beliefs.update(BeliefUpdate::PriceReturn {...})`
2. On fills, also call `central_beliefs.update(BeliefUpdate::OwnFill {...})`
3. On market trades, also call `central_beliefs.update(BeliefUpdate::MarketTrade {...})`
4. Forward regime updates from HMM
5. Forward changepoint observations from BOCD

### Phase 4: Migrate QuoteEngine reads ⬜ TODO

**Files to modify:**
- `src/market_maker/orchestrator/quote_engine.rs`

**Tasks:**
1. Replace scattered reads like:
   ```rust
   // OLD
   let bias = self.stochastic.beliefs_builder.beliefs().predictive_bias();
   let kappa = self.estimator.kappa();

   // NEW
   let beliefs = self.central_beliefs.snapshot();
   let bias = beliefs.drift_vol.expected_drift;
   let kappa = beliefs.kappa.kappa_effective;
   ```
2. Verify behavior unchanged via tests

### Phase 5: Migrate ParameterAggregator ⬜ TODO

**Files to modify:**
- `src/market_maker/strategy/params/aggregator.rs`

**Tasks:**
1. Add `build_from_beliefs()` method that takes `&BeliefSnapshot`
2. Map snapshot fields to MarketParams fields
3. Deprecate scattered field reads

### Phase 6: Migrate QuoteGate ⬜ TODO

**Files to modify:**
- `src/market_maker/control/quote_gate.rs`

**Tasks:**
1. Update `QuoteGateInput` to include `beliefs: BeliefSnapshot`
2. Replace 40+ scattered field reads with snapshot access
3. Simplify threshold logic using belief confidences

### Phase 7: Remove dual-write ⬜ TODO (HIGH RISK)

**Tasks:**
1. Remove old belief update paths
2. Remove old read paths
3. Run full test suite
4. Paper trade for stability verification

### Phase 8: Cleanup deprecated code ⬜ TODO (HIGH RISK)

**Files to potentially deprecate/remove:**
- Parts of `src/market_maker/control/belief.rs` (Layer 3 BeliefState)
- Duplicate logic in `src/market_maker/stochastic/beliefs.rs`

**Tasks:**
1. Mark deprecated code with `#[deprecated]`
2. Remove after verification period
3. Update documentation

---

## Verification Commands

```bash
# After Phase 4 (QuoteEngine migration)
cargo build --lib
cargo test belief --lib

# Run 2-hour test
./scripts/test_hip3.sh HYPE hyna 7200 hip3

LOG=$(ls -t logs/mm_hip3_*.log | head -1)

# Check calibration metrics (should now be populated)
grep "calibration" $LOG | tail -20

# Check fill IR (target: >1.0)
grep "fill_ir" $LOG | tail -5

# Check AS IR (target: >1.0)
grep "as_ir" $LOG | tail -5

# Verify single-source reads
grep "BeliefSnapshot" $LOG | head -10
```

---

## Success Criteria

- [ ] All consumers read from CentralBeliefState (no scattered reads)
- [ ] Calibration metrics logged for every fill prediction
- [ ] IR > 1.0 for fill predictions after 100+ samples
- [ ] No threshold spaghetti in QuoteEngine/QuoteGate
- [ ] Tests pass for all belief components

---

## Architecture Reference

### CentralBeliefState Usage

```rust
// Initialize
let config = CentralBeliefConfig::hip3();
let central_beliefs = CentralBeliefState::new(config);

// Publishers send updates
central_beliefs.update(BeliefUpdate::PriceReturn {
    return_frac: 0.001,
    dt_secs: 1.0,
    timestamp_ms: now_ms(),
});

// Consumers read snapshots
let beliefs = central_beliefs.snapshot();
let kappa = beliefs.kappa.kappa_effective;
let regime = beliefs.regime.current;
let p_cont = beliefs.continuation.p_fused;
```

### BeliefSnapshot Structure

```rust
pub struct BeliefSnapshot {
    pub drift_vol: DriftVolatilityBeliefs,    // E[μ], E[σ], P(bearish)
    pub kappa: KappaBeliefs,                   // EWMA-smoothed, components
    pub continuation: ContinuationBeliefs,     // P(cont) fused
    pub regime: RegimeBeliefs,                 // [Quiet,Normal,Bursty,Cascade]
    pub changepoint: ChangepointBeliefs,       // P(cp), run_length
    pub edge: EdgeBeliefs,                     // Expected edge, P(positive)
    pub calibration: CalibrationState,         // Brier, IR metrics
    pub stats: BeliefStats,                    // n_obs, timestamps
}
```
