# Signal Diagnostic Infrastructure - Live Test Results

## Session: 2026-02-03

### Test Overview
- **Duration**: ~6 minutes live trading on HIP-3 DEX
- **Asset**: hyna:HYPE
- **Fills**: 16 total
- **Export File**: `fills_with_signals.json`

### Infrastructure Status
✅ FillSignalSnapshot struct capturing 27 signal fields
✅ Markout tracking at 500ms, 2s, 10s horizons
✅ JSON export auto-triggered at 5-minute intervals
✅ CLI flag `--signal-export-path` wired to MarketMaker

### Critical Finding: Pre-Fill Toxicity Classifier Blind Spot

**The pre-fill toxicity signal failed to detect 6 of 7 toxic fills (86% miss rate).**

| Fill Side | Toxicity Signal | CP Prob | Actual Markout |
|-----------|-----------------|---------|----------------|
| Bid | 0-2% | 1-2% | -50 bps |
| Bid | 0-2% | 1-2% | -51 bps |
| Bid | 0% | 1-2% | -50 bps |
| Bid | 0% | 1% | -50 bps |
| Bid | 0% | 1% | -50 bps |
| Bid | 0% | 2% | -23 bps |
| Ask | **67%** | **100%** | -12 bps |

**Pattern**: Bid-side toxicity is completely undetected. Ask-side works.

### Signal Calibration Metrics
- Mean markout: -16.57 bps (losing money on average)
- Min markout: -50.88 bps
- Max markout: +8.76 bps
- Toxic fills (< -10 bps): 7 of 16 (44%)

### Files Modified (Signal Diagnostic Infrastructure)
1. `src/market_maker/fills/processor.rs` - FillSignalSnapshot, PendingMarkout, FillSignalStore
2. `src/market_maker/fills/mod.rs` - Export new types
3. `src/market_maker/core/components.rs` - Added signal_store to SafetyComponents
4. `src/market_maker/mod.rs` - Added cached_market_params, with_signal_export_path()
5. `src/market_maker/orchestrator/handlers.rs` - Wire signal capture and markout updates
6. `src/market_maker/orchestrator/quote_engine.rs` - Cache market_params
7. `src/bin/market_maker.rs` - Added --signal-export-path CLI argument
8. `scripts/analysis/signal_calibration.py` - Analysis script (needs numpy)
9. `scripts/analysis/toxic_fill_audit.py` - Toxic fill audit script

### Next Steps
1. Investigate bid/ask asymmetry in Pre-Fill AS Classifier
2. Add directional features (bid vs ask) to classifier
3. Run longer test (30-60 min) for statistical significance
4. Retrain classifier with fill signal data

### Technical Notes
- Export triggers: 100 fills OR 5 minutes elapsed
- Markout windows: 500ms, 2s, 10s from fill timestamp
- All signals from MarketParams captured at fill time
- Position and mid price tracked for markout calculation
