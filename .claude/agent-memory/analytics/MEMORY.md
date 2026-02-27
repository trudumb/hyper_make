# Analytics Agent Memory

## Current State
- Calibration pipeline working: prediction logger drain semantics fixed (Feb 7)
- JSONL logging active for fill predictions and calibration records

## Active Gotchas
- Logger "pending" + "drain" semantics: intermediate ops should NOT remove from pending, only drain removes
- Python calibration_report.py expects nested JSON from Rust PredictionRecord (not flat fields)
- `predictions.levels[i].p_fill_1s` and `outcomes.fills[j].level_index` for matching
