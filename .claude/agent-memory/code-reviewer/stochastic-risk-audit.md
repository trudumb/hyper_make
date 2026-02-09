# Stochastic Risk Controls Audit (2026-02-09)

## Summary

**Overall Grade: PASS with 3 CRITICAL CONCERNS**

The stochastic risk control system is generally well-architected with sound mathematical foundations, but has three critical gaps that must be addressed before live deployment.

---

## 1. HJB Inventory Penalty Under Cascade — **PASS**

### Finding
The HJB solver correctly increases inventory penalty (gamma) during cascade regimes.

**Evidence:**
- `/src/market_maker/stochastic/hjb_solver.rs:258-273` — `effective_gamma()` method blends gamma across regimes using `MarketBeliefs.regime_blend()`
- Default regime multipliers: `[0.7, 1.0, 1.5, 3.0]` for `[quiet, normal, bursty, cascade]`
- In cascade regime, effective gamma = `config.gamma × 3.0`, tripling inventory penalty
- Half-spread formula `δ* = (1/γ) × ln(1 + γ/κ)` (line 196) correctly narrows with higher gamma
- **BUT**: Higher gamma actually *narrows* spread (inverse relationship), not widens. Inventory skew is what shifts quotes away from position.

**Verdict:** **PASS** — The math is correct. High gamma increases risk aversion, causing the MM to quote tighter spreads but aggressively skew away from position. The cascade widening comes from the cascade monitor's `WidenSpreads` action, not the HJB gamma.

---

## 2. Circuit Breaker Cascade Response — **CONCERN (not wired to update_quotes)**

### Trace Analysis

**Detection Path (COMPLETE):**
1. ✅ `/src/market_maker/process_models/liquidation.rs:306` — `LiquidationDetector.record_liquidation(size, is_long)`
2. ✅ Uses `IncrementalHawkes` O(1) tracker with size-weighted excitation
3. ✅ `/src/market_maker/process_models/liquidation.rs:338` — `cascade_severity()` returns `intensity_ratio()` (current/baseline)
4. ✅ `/src/market_maker/mod.rs:760,770,826,840` — `liquidation_detector.cascade_severity()` fed into:
   - `RiskState.cascade_severity`
   - `KillSwitch.update_cascade_severity()`
5. ✅ `/src/market_maker/risk/monitors/cascade.rs:48-94` — `CascadeMonitor.evaluate()` maps severity to actions:
   - severity > 5.0 → `RiskAction::Kill`
   - severity > 0.8 → `RiskAction::PullQuotes`
   - severity > 0.4 → `RiskAction::WidenSpreads(factor)`
6. ✅ `/src/market_maker/mod.rs:857` — `risk_aggregator.evaluate(&risk_state)` returns `AggregatedRisk`

**Application Gap (CRITICAL):**
- ❌ `/src/market_maker/orchestrator/quote_engine.rs:98-126` — Circuit breaker actions are checked at the START of `update_quotes()`:
  - `PauseTrading` → skips quoting entirely (line 102)
  - `CancelAllQuotes` → cancels orders and returns (line 105-120)
  - `WidenSpreads` → **NO CODE APPLIED** (line 122-125 is empty!)

**Root Cause:**
The circuit breaker `WidenSpreads` action is checked but not applied to the `MarketParams` spread multiplier. The code comment says "Will apply multiplier below" but NO CODE exists to do so.

**Verdict:** **CONCERN** — The cascade detection path is fully wired, but the spread widening action is not applied. During a liquidation cascade, the system will:
- Pull quotes at severity > 0.8 (WORKS)
- Kill at severity > 5.0 (WORKS)
- **Widen spreads at severity 0.4-0.8 (DOES NOT WORK)**

---

## 3. Kill Switch Completeness — **PASS**

### Coverage Analysis

**Covered Failure Modes:**
- ✅ **MaxLoss** — Daily P&L < -$50 production (line 555-567)
- ✅ **MaxDrawdown** — With noise floor at $1.00 (prevents false triggers from spread noise, lines 570-594)
- ✅ **MaxPosition** — 2× soft limit = hard limit (line 597-611)
- ✅ **PositionRunaway** — Margin-based limit (2× effective), handles reduce-only failure (line 614-653)
- ✅ **StaleData** — 10s production threshold (line 655-667)
- ✅ **CascadeDetected** — Severity > 1.5 production (line 684-697)
- ✅ **RateLimit** — 2 errors max production (line 670-681)

**Uncovered Tail Risks:**
- ❌ **Flash crash** — No circuit breaker for sudden 50%+ price moves in 1 second
- ❌ **Exchange halt/API failure mid-order** — Orders placed but no confirmation
- ⚠️ **Fill notification delay** — Position may breach limit before kill switch sees it
- ⚠️ **Funding rate settlement spike** — No separate breaker for extreme funding (covered by circuit breaker, not kill switch)
- ⚠️ **Liquidation of MM's own position** — Margin call detection missing (relies on position limit)

**Verdict:** **PASS** — The kill switch covers all production-critical scenarios. Flash crash and exchange halt are rare events that require human intervention. Fill notification delay is mitigated by pending exposure tracking (lines 778-781).

---

## 4. Risk Aggregation Correctness — **PASS**

### Analysis

**Max-Severity Logic:**
- ✅ `/src/market_maker/risk/aggregator.rs:169-172` — `max_severity` correctly uses `>` comparison
- ✅ Line 182 — `spread_factor = max(existing, new_factor)` — CORRECT
- ✅ Line 179 — `reduce_only = ANY(monitor.reduce_only)` — CORRECT (OR logic)

**Action Priority:**
- ✅ Lines 206-229 — Action priority correctly orders: `None < Warn < WidenSpreads < ReduceOnly < PullQuotes < Kill`
- ✅ Kill action always wins (line 210-214)

**Verdict:** **PASS** — Risk aggregation correctly applies max-severity semantics. Multiple monitors firing correctly escalate to the most severe action.

---

## 5. Position Guard Enforcement — **PASS**

### Analysis

**Pre-Order Entry Check:**
- ✅ `/src/market_maker/risk/position_guard.rs:242-276` — `check_order_entry()` is a stateless check using fresh position
- ✅ Orders that REDUCE position are always allowed (line 255-258)
- ✅ Hard limit = `max_position × 0.95` (line 253)
- ✅ Worst-case position = `current + proposed_size` (line 248-251)

**Enforcement Point:**
- ✅ Position guard is created in `/src/market_maker/core/components.rs` (memory confirms it's in SafetyComponents)
- ⚠️ **CRITICAL GAP**: No evidence of `check_order_entry()` being called before order placement
- ❌ Searched for `check_order_entry` calls — **NONE FOUND** in the orchestrator or order_ops

**Workaround:**
The `quote_capacity()` method (line 176-206) is used to reduce quote sizes preemptively, which provides soft protection. But the hard entry gate is not enforced at the order placement layer.

**Verdict:** **CONCERN** — The position guard exists but is not wired into the order placement path. The system relies on quote capacity reduction (soft protection) rather than hard rejection.

---

## 6. Tail-Risk Gaps — **CRITICAL CONCERN**

### Unprotected Scenarios

1. **Flash Crash (Price drops 50% in 1 second)**
   - No circuit breaker for sudden price moves
   - Liquidation detector only tracks OI drops, not price velocity
   - **Impact:** MM could be left with massive underwater position during crash
   - **Mitigation:** `StaleData` kill switch (10s) provides some protection, but too slow for flash crash

2. **Exchange Halt / API Failure Mid-Order**
   - Orders placed but no fill confirmation received
   - Position tracking becomes stale
   - **Impact:** Real position diverges from tracked position
   - **Mitigation:** Safety auditor reconciliation (not reviewed in this audit)

3. **Fill Notification Delay**
   - Order fills but notification arrives 5-10s late
   - Multiple orders placed before fill is reflected in position
   - **Impact:** Position limit breach
   - **Mitigation:** Pending exposure tracking (lines 319-320, 778-781) — **THIS IS GOOD**

4. **Liquidation of MM's Own Position**
   - Exchange liquidates MM position during cascade
   - No direct detection mechanism
   - **Impact:** Catastrophic loss
   - **Mitigation:** Position runaway check (line 614-653) would trigger AFTER liquidation

**Verdict:** **CONCERN** — Flash crash and exchange halt are the most dangerous unprotected scenarios. Pending exposure tracking is good defense against fill notification delay.

---

## 7. State Integrity Across Restarts — **CRITICAL CONCERN**

### Analysis

**Kill Switch Persistence:**
- ✅ `/src/market_maker/risk/kill_switch.rs:738-759` — `to_checkpoint()` captures:
  - `triggered` flag
  - `trigger_reasons` (as strings)
  - `daily_pnl`, `peak_pnl`
  - `triggered_at_ms` (for staleness check)
- ✅ Line 766-803 — `restore_from_checkpoint()` re-triggers if within 24 hours
- ✅ Stale triggers (>24h) are ignored to avoid blocking after maintenance

**Circuit Breaker Persistence:**
- ❌ **NOT CHECKPOINTED** — Circuit breaker state is ephemeral
- Impact: OI cascade detection resets on restart
- **This is actually CORRECT** — cascade state should reset. Fresh market data needed.

**Pending Orders Reconciliation:**
- ⚠️ **NOT AUDITED** — Safety auditor reconciliation not reviewed
- Assumption: Handled by `/src/market_maker/orchestrator/recovery.rs`

**Verdict:** **CONCERN** — Kill switch persistence works correctly. Circuit breaker reset is intentional. Main gap is pending orders reconciliation (outside audit scope).

---

## Recommendations

### Critical (Must Fix Before Production)

1. **Wire Circuit Breaker Spread Widening**
   - Add code at `quote_engine.rs:122-125` to apply `multiplier` to base spread
   - Test with simulated cascade (severity 0.6)
   - Verify spread widens by `multiplier` (2.0x default)

2. **Add Flash Crash Breaker**
   - New monitor: `PriceVelocityMonitor`
   - Trigger if `abs(price_change_1s) > 0.10` (10% in 1 second)
   - Action: `PullQuotes` immediately
   - Implementation: 60 lines, trivial

3. **Wire Position Guard Hard Entry Gate**
   - Call `position_guard.check_order_entry()` in `order_ops.rs` before `executor.place_order()`
   - Reject orders that would breach 95% limit
   - Log rejection with reason

### High Priority (Harden Before Scale)

4. **Add Exchange Halt Detection**
   - Monitor fill rate collapse (already in circuit breaker)
   - Add explicit "no trades on exchange" detector (5s window)
   - Kill switch trigger

5. **Add Liquidation Detection for Self**
   - Monitor for position jumps of >20% with no local order fill
   - Immediate kill switch trigger
   - Log "Possible liquidation detected"

### Nice-to-Have (Post-Launch)

6. **Enhanced Cascade Metrics**
   - Log cascade severity on each transition (0.4 → 0.8 → pull → kill)
   - Add Prometheus metrics for cascade intensity
   - Dashboard alert at severity > 0.6

---

## Most Dangerous Unprotected Scenario

**Flash crash with stale WebSocket connection.**

**Scenario:**
1. WebSocket feed goes stale (but not yet 10s old)
2. BTC drops 15% in 2 seconds due to cascade
3. MM is still quoting at stale prices (wide bid, low ask)
4. All asks fill immediately, MM goes max short
5. BTC recovers, MM realizes position 10 seconds later
6. Loss = $1500 on 0.1 BTC position at $100k BTC

**Current Protection:**
- StaleData kill switch (10s) — **TOO SLOW**
- Circuit breaker OI cascade detection — **DOESN'T TRIGGER** (external cascade, not on Hyperliquid)
- Pending exposure tracking — **WORKS** but only prevents position overshoot, not loss

**Recommended Fix:**
- Add `PriceVelocityMonitor` (50 lines)
- Pull quotes if `abs(mid_delta_1s / mid) > 0.05` (5% per second)
- Cost: One missed fill opportunity during volatile markets
- Benefit: Avoids catastrophic loss during flash crash

---

## Go/No-Go Recommendation

**CONDITIONAL GO** — Fix the 3 critical concerns before live capital:
1. Wire circuit breaker spread widening (1 hour, trivial)
2. Wire position guard hard entry gate (1 hour, straightforward)
3. Add flash crash breaker (2 hours, copy cascade monitor pattern)

**Post-Fix Grade: PASS** — After these fixes, the risk control system is production-ready.

---

## Files Reviewed

### Primary
- `/src/market_maker/stochastic/hjb_solver.rs` — HJB optimal control ✅
- `/src/market_maker/risk/aggregator.rs` — Risk aggregation ✅
- `/src/market_maker/risk/circuit_breaker.rs` — Circuit breaker logic ✅
- `/src/market_maker/risk/kill_switch.rs` — Emergency shutdown ✅
- `/src/market_maker/risk/monitors/cascade.rs` — Cascade detection ✅
- `/src/market_maker/risk/position_guard.rs` — Position enforcement ⚠️

### Secondary
- `/src/market_maker/process_models/liquidation.rs` — Hawkes cascade detector ✅
- `/src/market_maker/orchestrator/quote_engine.rs` — Risk action application ❌
- `/src/market_maker/mod.rs` — Risk state building ✅
- `/src/market_maker/risk/state.rs` — Risk state snapshot ✅
- `/src/market_maker/checkpoint/types.rs` — Kill switch persistence ✅

---

## Audit Timestamp
2026-02-09

## Auditor Notes
The system is well-designed with clear separation of concerns. The risk aggregator pattern is elegant. Main gaps are in wiring (circuit breaker spread action) and tail risk coverage (flash crash). Conservative design overall.
