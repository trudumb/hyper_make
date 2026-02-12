# Live Session Audit: HYPE Mainnet Feb 12 03:16-04:45 UTC

## Session Outcome
- **Duration**: 89 minutes, **PnL**: -$5.02 (kill switch triggered)
- **Fills**: 29 total (24 buys / 5 sells = 4.8:1 imbalance)
- **Final position**: 12.81 HYPE long (3.95x configured max of 3.24)
- **Market**: HYPE dropped 188 bps ($30.88 → $30.30) during session
- **Quoting uptime**: ~11 minutes of 89 (12%) — risk overlay disabled quoting for 78 min

## 6 Systemic Bugs Found

### BUG 1: AS Measurement Tautology (P0 — ALL metrics broken)
**Location**: `handlers.rs:714-718`
- `as_realized` and `depth_from_mid` both use `self.latest_mid` at same instant
- Algebraically: AS ≈ depth → realized_edge = 0 - fees = -1.50 bps for 97% of fills
- No `mid_at_placement` tracked (acknowledged TODO in code comment line 717)
- The 5-second markout (lines 223-300) correctly measures AS but feeds ONLY pre_fill_classifier, NOT EdgeSnapshot
- **Impact**: RL reward = noise, calibration = noise, PnL attribution = noise, edge prediction = noise

### BUG 2: Zero Directional Skew (P0 — sitting duck in sweeps)
- combined_skew_bps = 0.0 for 99.8% of 1,964 cycles
- No signal produces skew_adjustment_bps
- HL-native skew fallback has: `imbalance_30s: 0.0` (hardcoded), `avg_buy/sell_size: 0.0` (not tracked)
- Position-based inventory skew exists but maxes at 0.42 bps (negligible)
- **Impact**: We quote symmetrically during cascades. 24 buys / 5 sells = perfect adverse selection

### BUG 3: RegimeDetection Wired But Output Dead (P0)
- Has rich kappa data: range 500-4203 across session (8.4x variation!)
- `signals.kappa_effective` is computed but NEVER consumed in quote_engine.rs
- `spread_adjustment_bps: 0.0` hardcoded in live attribution logger
- **Impact**: Regime changes detected but ignored. Spreads don't widen when they should.

### BUG 4: Emergency Pull Paralysis (P0 — killed 89% of session)
- 903 emergency pulls total: 853 from `extreme_changepoint_0.99/1.00`
- Risk overlay pulled ALL quotes for 78 of 89 minutes
- System accumulated 10+ HYPE long BEFORE overlay kicked in, then couldn't reduce
- Position sat at 9-13 HYPE while market dropped 188 bps → -$5 → kill switch
- Two brief 3-min windows allowed fills: ALL were more buys (no sells filled)
- **Impact**: The "safety" mechanism caused the loss by preventing position reduction

### BUG 5: Position Limit Not Enforced (P0 — repeat of Feb 10 bug)
- Configured `max_position_contracts: 3.24`, actual position reached 12.81 (3.95x)
- `effective_max_position` derived from margin (~55 HYPE) vs user config (3.24)
- Same root cause as Feb 10 reduce-only bypass — our fixes may not be fully deployed
- **Impact**: Position 4x beyond configured risk limit

### BUG 6: InformedFlow Signal is Harmful (P1)
- Marginal value: -0.23 bps (negative — tightens spreads before getting run over)
- Tighten threshold: `p_informed < 0.05 → spreads × 0.9` — reduces buffer when uncertain
- **Impact**: Active signal making us worse

## Signal Status

| Signal | Active % | Contribution | Status |
|--------|----------|-------------|--------|
| InformedFlow | 76.9% | spread [-10, +41] bps, skew: 0 | HARMFUL (marginal -0.23 bps) |
| VPIN | 99.3% | spread [-33, +49] bps, skew: 0 | Working (spread only) |
| RegimeDetection | 100% | spread: 0, skew: 0 | DATA EXISTS, OUTPUT DEAD |
| LeadLag | 0.0% | none | DEAD (no Binance for HYPE) |
| CrossVenue | 1.0% | negligible | DEAD |
| BuyPressure | 0.2% | negligible | DEAD |

## Session Timeline

| Time | Event |
|------|-------|
| 03:16:40 | Session start, $100 capital, 10x leverage |
| 03:16:48 | First quotes placed (8s after start) |
| 03:17:07 | First fills: 2 buys simultaneously |
| 03:17-03:23 | 13 buys, 1 sell → position = 10.44 (6 minutes!) |
| 03:19:11 | First changepoint detected |
| 03:23:52 | First emergency BID pull (position already 10.44) |
| 03:26:44 | Risk overlay locks: `extreme_changepoint_0.99` — ALL quotes cancelled |
| 03:35-03:38 | Brief 3-min window: 4 buys, 2 sells |
| 03:38:47 | Risk overlay locks again. Position = 9.45 |
| 03:38-04:13 | **35 MINUTE DEAD ZONE** — no fills, position stuck, market dropping |
| 04:13-04:16 | Brief 3-min window: 6 buys, 1 sell → position = 12.81 |
| 04:16-04:45 | Risk overlay locked. Market continues dropping. |
| 04:45:26 | Kill switch: daily loss $5.02 > $5.00 limit |
| 04:45:27 | Shutdown. 12.81 HYPE long left on books. |

**Kelly tracker**: 9W / 20L = 31% win rate (degraded from 57% early)

## Additional Issues
- **14 "invalid size" order rejections**: fractional sizes HL rejects
- **940 SafetySync mismatches**: exchange vs local order count persistent disagreement
- **128 floor binding warnings**: GLFT overridden by floor in 13% of cycles
- **Warmup never completed**: stuck at 90% (fills stopped → kappa calibration stalled)
- **risk_model_blend: 0.0**: pure legacy multiplicative gamma, not calibrated model

## Prioritized Fix List

| # | Fix | Impact | Effort | Files |
|---|-----|--------|--------|-------|
| P0-1 | AS tautology: store mid_at_placement | Unlocks ALL learning | M | handlers.rs, order tracking |
| P0-2 | Emergency pull: ensure reduce-only quotes survive | Prevents paralysis loss | S | control/mod.rs, quote_engine.rs |
| P0-3 | Position limit: enforce config.max_position as hard cap | Safety critical | S | Verify Feb 10 fixes deployed |
| P0-4 | Regime kappa: wire signals.kappa_effective to spread | +0.5-1 bps | S | quote_engine.rs |
| P0-5 | HL-native skew: fix flow feature vec (30s=0, sizes=0) | +0.3-0.5 bps | S | handlers.rs |
| P0-6 | InformedFlow: disable tightening (min_tighten_mult=1.0) | +0.23 bps | Trivial | model_gating.rs |
| P1-1 | QuoteOutcomeTracker → spread optimizer feedback | +0.3 bps | M | quote_outcome.rs, quote_engine.rs |
| P1-2 | Multi-timescale order flow imbalance | +0.2 bps | M | new struct, handlers.rs |
| P1-3 | OI delta + book depth velocity (hardcoded 0.0 TODOs) | Safety | M | quote_engine.rs |
| P2-1 | Verify funding proximity wiring | +0.1 bps | S | temporal.rs → spread path |

**Total addressable**: ~3.4 bps. Current: -1.5 bps (pure fee drag, but unmeasurable due to Bug 1).
