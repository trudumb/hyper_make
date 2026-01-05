# Session: 2026-01-05 Kappa Warmup Fix

## Summary

Fixed critical bug where kappa collapsed from 500 → 280 during warmup on HIP-3 HYPE, causing 71 bps spreads instead of ~47 bps.

## Root Cause

In `kappa_orchestrator.rs`, the warmup logic only disabled the robust estimator (market trades), but still allowed book_kappa to contribute:

1. `own_kappa.confidence()` returned 93% from tight prior (NOT from actual data)
2. `book_kappa.confidence()` became non-zero after 5 L2 updates (returning R²)
3. Book kappa regressed to ~280 on sparse HIP-3 books
4. Confidence-weighted blend: 93% own + high% book → dragged kappa to ~280-300

## Fix

Changed `kappa_effective()` to return ONLY prior during warmup (no own fills):

```rust
// During warmup: ONLY use prior - disable own, book, and robust
if !has_own_fills {
    return self.config.prior_kappa;
}
```

Also updated `component_breakdown()` to reflect same logic for consistent logging.

## Files Modified

| File | Change |
|------|--------|
| `src/market_maker/estimator/kappa_orchestrator.rs:180-192` | Early return to prior during warmup |
| `src/market_maker/estimator/kappa_orchestrator.rs:240-250` | Updated component_breakdown for warmup |

## Results

| Metric | Before | After |
|--------|--------|-------|
| kappa_robust | 282-307 (unstable) | 500 (stable) |
| total_spread_bps | 71.3 | 46.9-47.3 |
| Spread reduction | - | 34% |

## Key Insight

During warmup (no own fills), we MUST NOT trust:
- `own_kappa.confidence()` - inflated by tight prior, not actual data
- `book_kappa.confidence()` - R² can be high on sparse books with low κ

The only safe signal is `own_kappa.observation_count() >= 5` - actual fills.

## Verification

```
22/22 kappa_robust values = 500 (100% stable)
Spreads: 46.9-47.3 bps (down from 71 bps)
```
