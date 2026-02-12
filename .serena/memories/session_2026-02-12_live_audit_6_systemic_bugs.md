# Session 2026-02-12: Live Mainnet Audit — 6 Systemic Bugs Found

## Session Summary
- **What**: Deep analysis of 4h HYPE mainnet test (Feb 12 03:16-04:45 UTC) using 4-agent team
- **Outcome**: Found 6 systemic bugs explaining -$5.02 loss and 0.8% win rate
- **Data analyzed**: 246 fills (edge_validation.jsonl), 1964 signal cycles, 9943 log lines
- **Also fixed**: AS default=1500 bug (category mismatch) and emergency pull overactivity (threshold+cooldown)

## The 6 Bugs (Priority Order)

### BUG 1: AS Measurement Tautological (handlers.rs:714-718)
- `as_realized` and `depth_from_mid` BOTH use `self.latest_mid` at same instant
- Algebraically: AS ≈ depth → realized_edge = 0 - fees = -1.50 bps for 97% of fills
- No `mid_at_placement` stored (TODO comment on line 717 acknowledges this)
- 5-second markout (lines 223-300) correctly measures AS but feeds ONLY pre_fill_classifier, NOT EdgeSnapshot
- **Impact**: ALL downstream learning (RL, edge prediction, calibration) trained on noise

### BUG 2: Zero Directional Skew
- combined_skew_bps ≈ 0 for 99.8% of 1964 cycles
- HL-native skew fallback: `imbalance_30s: 0.0` hardcoded, `avg_buy/sell_size: 0.0` not tracked
- Session: 24 buys / 5 sells — never leaned away from flow
- **Impact**: Symmetric quoting during one-sided sweeps = textbook adverse selection

### BUG 3: RegimeDetection Wired But Dead
- Live kappa data ranges 500-4203 (8.4x variation)
- `signals.kappa_effective` computed but NEVER consumed in quote_engine.rs after line 865
- `spread_adjustment_bps: 0.0` hardcoded in live.rs:291
- **Impact**: Regime changes detected and ignored

### BUG 4: Emergency Pull Paralysis (78 of 89 minutes)
- 903 emergency pulls, risk overlay pulled ALL quotes for 89% of session
- System accumulated 10+ HYPE long BEFORE overlay activated, then couldn't reduce
- 35-minute dead zone with position stuck at 9.45 while market dropped 188 bps
- Fixed this session: threshold 0.9→0.95, 50-cycle cooldown added
- **Still need**: reduce-only quotes must survive emergency pull

### BUG 5: Position Limit Not Enforced (REPEAT of Feb 10)
- Config max_position=3.24, actual=12.81 (3.95x limit)
- effective_max_position from margin (~55 HYPE) vs user config (3.24)
- Feb 10 fixes may not be fully deployed in this build

### BUG 6: InformedFlow Signal Harmful
- Marginal value: -0.23 bps (negative)
- Tightens spreads when p_informed < 0.05 (min_tighten_mult=0.9)
- Fix: set min_tighten_mult=1.0 (disable tightening)

## Signal Status from Live Data
| Signal | Active% | Contribution | Status |
|--------|---------|-------------|--------|
| InformedFlow | 76.9% | spread [-10,+41] bps | HARMFUL (-0.23 bps marginal) |
| VPIN | 99.3% | spread [-33,+49] bps | Working (spread only) |
| RegimeDetection | 100% | spread: 0, skew: 0 | DATA EXISTS, OUTPUT DEAD |
| LeadLag | 0.0% | none | DEAD (no Binance for HYPE) |
| CrossVenue | 1.0% | negligible | DEAD |
| BuyPressure | 0.2% | negligible | DEAD |

## Code Changes This Session
1. `handlers.rs`: predicted_as_bps now uses `self.estimator.total_as_bps()` (real AS from EWMA) instead of `bayesian_adverse() * 10_000` (probability mismatch giving 1500 bps)
2. `handlers.rs`: Added `MarketEstimator` trait import
3. `control/mod.rs`: Emergency pull threshold 0.9→0.95, 50-cycle cooldown, `last_emergency_pull_cycle` field
4. Tests: 2369 passed, clippy clean

## Priority Fix List (from feature gap analysis)
| # | Fix | Impact | Effort |
|---|-----|--------|--------|
| P0-1 | Store mid_at_placement, fix EdgeSnapshot | Unlocks ALL learning | M |
| P0-2 | Reduce-only quotes survive emergency pull | Prevents paralysis loss | S |
| P0-3 | Enforce config.max_position hard cap | Safety critical | S |
| P0-4 | Wire regime kappa to spread calculator | +0.5-1 bps | S |
| P0-5 | Fix HL flow feature vec (30s, sizes hardcoded 0) | Enables skew | S |
| P0-6 | Disable InformedFlow tightening | +0.23 bps | Trivial |
| P1-1 | QuoteOutcomeTracker → spread optimizer feedback | +0.3 bps | M |
| P1-2 | Multi-timescale order flow imbalance | +0.2 bps | M |
| P1-3 | OI delta + book depth velocity (TODO=0.0) | Safety | M |

## Key Metrics
- **Session PnL**: -$5.02 (kill switch at daily loss limit)
- **Win rate**: 0.8% (2/246 in edge_validation) / 31% Kelly (9W/20L in log)
- **Quoting uptime**: 12% (11 of 89 minutes)
- **Position peak**: 12.81 HYPE (3.95x configured max)
- **Realized AS**: 2.34 bps mean (from markout, not tautological metric)
- **Fill clustering**: 51% of inter-fill intervals < 10s

## Deliverables
- `.claude/plans/live-session-audit-findings.md` — Full findings document
- `.serena/memories/edge_validation_tautology_root_cause.md` — AS tautology proof (from Explore agent)
