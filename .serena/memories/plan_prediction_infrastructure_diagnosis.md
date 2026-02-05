# Plan: Prediction Infrastructure Diagnosis & Fix
**Source**: `.claude/plans/greedy-dancing-zebra.md`
**Created**: 2026-02-04

## Executive Summary
15+ critical issues found across three categories:
1. **Model defects** - Models can't learn or output near-constant values
2. **Validation flaws** - prediction_validator measures wrong things
3. **Integration gaps** - Models don't receive proper signals

**Bottom line**: Both models AND measurement are broken. Fix measurement first.

---

## Category 1: Model Defects

### 1.1 InformedFlowEstimator - CAN'T LEARN
**File**: `src/market_maker/estimator/informed_flow.rs`
- `price_impact_bps` always = 0.0 → can't distinguish informed from noise
- Hardcoded priors (5% informed, 85% noise) → 500+ trades to adapt
- Falls back to prior on likelihood underflow → returns constant
- Learning rate λ=0.999 → ~500 trades to halve old observations

### 1.2 PreFillASClassifier - NON-LEARNING
**File**: `src/market_maker/adverse_selection/pre_fill_classifier.rs`
- Fixed weights (0.30, 0.25, 0.25, 0.10, 0.10) → never adapts
- Symmetric bid/ask assumption → wrong for asymmetric microstructure
- Staleness detection exists but unused

### 1.3 RegimeHMM - LAGGING FEATURES
**File**: `src/market_maker/estimator/regime_hmm.rs`
- Only 2 features: vol + spread → can't predict cascades
- Vol computed over 10-60s window → 100-500ms late detection
- Spread is circular (we set it) → feedback loop
- Missing OI, liquidation indicators

---

## Category 2: Validation Infrastructure Flaws

### 2.1 Survivorship Bias (CRITICAL)
**File**: `src/bin/prediction_validator.rs`
- Predictions only recorded on synthetic fills
- Never testing adverse selection avoidance scenarios
- Selection bias inflates apparent model quality

### 2.2 Threshold Too Tight (2 bps)
- 2 bps adverse threshold below noise floor
- BTC spread: 3-10 bps typically
- Everything looks "adverse" → base rate 70-90%

### 2.3 Trade Prices Instead of Mids
- Reference/outcome prices are trade prices, not mids
- Bid/ask bounce causes systematic bias

### 2.4 No Warmup Period
- Models start with defaults, predictions immediate
- First 50 predictions are random + priors

### 2.5 IR Calculation Issues
**File**: `src/market_maker/calibration/information_ratio.rs`
- Default base_rate=0.5 when no data
- 10 bins with <1000 samples → high variance

---

## Category 3: Integration Gaps

### 3.1 Feature Timing Mismatch
- sigma() from 1-10s vol, spread_bps() current, flow_imbalance() 1-5s
- All fed as simultaneous but 2-5s apart

### 3.2 Calibration Not Wired
- Production calibration exists but not connected
- Predictions never stored with IDs
- Outcomes never matched → IR undefined (0/0 → 0)

---

## Prioritized Fix Plan

### Phase 1: MEASURE FIRST (1-2 days)
1.1: Fix prediction_validator thresholds (2→5-10 bps, use mids, 100-sample warmup)
1.2: Fix survivorship bias (record on ALL book updates)
1.3: Verify IR computation (debug logging, check bin distribution)

### Phase 2: FIX CRITICAL MODEL BUGS (2-3 days)
2.1: Fix InformedFlowEstimator price_impact (use EWMA of realized impacts)
2.2: Add learning to PreFillASClassifier (online linear regression for weights)
2.3: Add cascade signals to RegimeHMM (OI level/velocity, liquidation indicator)

### Phase 3: FIX INTEGRATION (1-2 days)
3.1: Synchronize feature sampling (single timestamp for observation bundle)
3.2: Wire production calibration (predict() before orders, record_outcome() after fills)

---

## Files to Modify
| File | Changes |
|------|---------|
| `src/bin/prediction_validator.rs` | Thresholds, warmup, survivorship bias, mid prices |
| `src/market_maker/estimator/informed_flow.rs` | price_impact computation, faster learning |
| `src/market_maker/adverse_selection/pre_fill_classifier.rs` | Online weight learning |
| `src/market_maker/estimator/regime_hmm.rs` | OI/liquidation features |
| `src/market_maker/calibration/information_ratio.rs` | Default base_rate, min sample warning |

---

## Verification
After Phase 1:
```bash
cargo run --release --bin prediction_validator -- --asset BTC --duration 1h --adverse-threshold-bps 8.0
```

Expected:
- Base rate 30-50% (not 70-90%)
- IR computable (not 0.000)
- Different models show different IR

## Key Insight
Fix measurement first → then you'll know which models to fix vs. remove.
