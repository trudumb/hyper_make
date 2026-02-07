# Analytics Agent Memory

## Calibration Pipeline Fix (2026-02-07)

**Root cause**: Two bugs causing 0 fill prediction samples despite 5 fills and 7026 prediction records.

### Bug 1 (Rust - prediction.rs): attach_outcomes drained records prematurely
`PredictionLogger::attach_outcomes()` called `pending.remove(&cycle_id)` and wrote the record to disk immediately.
When `drain_completed_records()` was called later (every 10s in paper_trader), the records were already gone from `pending_records`.
Result: `calibration_analyzer.add_record()` was never called with any completed records.

**Fix**: Changed `attach_outcomes()` to use `pending.get_mut()` instead of `pending.remove()`. The record stays in `pending_records`
with `outcomes = Some(...)` until `drain_completed_records()` finds it, writes to disk, and returns it for calibration analysis.

### Bug 2 (Python - calibration_report.py): Field name mismatch with Rust serialization
The Python script expected flat fields `fill_probability` and `was_filled` on each prediction record.
The actual Rust `PredictionRecord` serializes as nested JSON:
- `predictions.levels[i].p_fill_1s` (fill probability per level)
- `outcomes.fills[j].level_index` (which levels were filled)
- `predictions.expected_adverse_selection_bps` / `outcomes.adverse_selection_realized_bps`

**Fix**: Rewrote Python extraction to traverse the nested structure, matching each `LevelPrediction` against fill outcomes by `level_index`.

### Files modified
- `src/market_maker/simulation/prediction.rs` - `attach_outcomes` method
- `scripts/analysis/calibration_report.py` - prediction extraction logic (lines 323-387)

### Key pattern to remember
When a logger has "pending" + "drain" semantics, intermediate operations should NOT remove from pending.
Only the drain operation should remove and write.
