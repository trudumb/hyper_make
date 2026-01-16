# CLAUDE.md - Hyperliquid Market Making System

## Project Overview

This is a quantitative market making system for Hyperliquid perpetual futures, built in Rust. The system implements Guéant-Lehalle-Fernandez-Tapia (GLFT) optimal market making with proprietary extensions for:

- Exchange-specific fill intensity modeling (Hawkes processes)
- Adverse selection prediction and avoidance
- Cross-exchange lead-lag exploitation (Binance → Hyperliquid)
- Regime detection and adaptive parameter tuning

**Primary Goal:** Capture bid-ask spread while minimizing adverse selection losses.

**Key Insight:** Edge comes from superior parameter estimation, not from the GLFT formula itself (which is public). The competitive advantage is in:
1. Better kappa (fill intensity) estimation using exchange-specific features
2. Faster adverse selection detection
3. Cross-exchange information advantages
4. Regime-aware parameter adaptation

---

## Skill System

This project uses a skill-based architecture for Claude CLI. Skills provide domain-specific guidance for building and maintaining the trading system.

### Skill Hierarchy

```
foundation/          # Must read first for any model work
├── measurement-infrastructure/   # Prediction logging
├── calibration-analysis/         # Model validation
└── signal-audit/                 # Feature selection

models/              # Individual predictive components
├── fill-intensity-hawkes/        # Kappa estimation
├── adverse-selection-classifier/ # Toxic flow prediction
├── regime-detection-hmm/         # Market state tracking
└── lead-lag-estimator/           # Cross-exchange signal

integration/         # Putting it together
└── quote-engine/                 # Full pipeline

operations/          # Production concerns
└── daily-calibration-report/     # Health monitoring
```

### Skill Dependencies

```
measurement-infrastructure (ALWAYS READ FIRST)
         │
         ├──→ calibration-analysis
         │
         └──→ signal-audit
                   │
                   ├──→ fill-intensity-hawkes
                   │
                   ├──→ adverse-selection-classifier
                   │
                   ├──→ regime-detection-hmm
                   │
                   └──→ lead-lag-estimator
                              │
                              └──→ quote-engine
                                        │
                                        └──→ daily-calibration-report
```

### When to Read Which Skills

| Task | Skills to Read (in order) |
|------|---------------------------|
| Any model work | measurement-infrastructure FIRST, always |
| Debug PnL issue | calibration-analysis → identify weak component → that model's skill |
| Add new signal | signal-audit → measurement-infrastructure |
| Improve fill prediction | measurement-infrastructure → signal-audit → fill-intensity-hawkes |
| System losing money in cascades | adverse-selection-classifier, regime-detection-hmm |
| Cross-exchange edge decaying | lead-lag-estimator |
| Wire up new component | quote-engine |
| Production monitoring | daily-calibration-report |

---

## Development Principles

### 1. Measurement Before Modeling

**Never build a model without first measuring what you're trying to predict.**

Before implementing any predictive component:
1. Define the prediction target precisely
2. Set up logging for predictions and outcomes
3. Establish baseline metrics (base rate, naive predictor)
4. Only then build the model
5. Validate against calibration metrics

### 2. Calibration is Ground Truth

A model is only as good as its calibration metrics say it is. Key metrics:

- **Brier Score**: Mean squared error of probability predictions
- **Information Ratio**: Resolution / Uncertainty (>1.0 means model adds value)
- **Conditional Calibration**: Metrics sliced by regime, volatility, etc.

If IR < 1.0, the model is adding noise. Remove it.

### 3. Edge Decays

All alpha decays. Monitor for:
- Signal MI dropping week-over-week
- Lead-lag R² declining (arbitrageurs closing the gap)
- Regime parameters drifting

Build staleness detection into every model.

### 4. Regime Awareness

Never use a single parameter value. Everything is regime-dependent:
- Kappa (fill rate) varies 10x between quiet and cascade
- Optimal gamma (risk aversion) varies 5x
- Spread floors vary 10x

Use the HMM belief state to blend parameters, not hard switches.

### 5. Defense First

When uncertain, widen spreads. The cost of missing a trade is small. The cost of getting run over in a liquidation cascade is large.

---

## Key Domain Concepts

### GLFT Formula

The optimal half-spread from Guéant-Lehalle-Fernandez-Tapia:

```
δ* = (1/γ) * ln(1 + γ/κ) + fee
```

Where:
- γ = risk aversion parameter (higher = wider spreads)
- κ = fill intensity (fills per unit spread)
- fee = maker fee (1.5 bps on Hyperliquid)

**The formula is trivial. The edge is in estimating γ and κ correctly.**

### Hyperliquid-Specific Features

1. **Funding Rate**: Paid every 8 hours. Creates predictable flow patterns around settlement.
2. **Open Interest**: On-chain, real-time. OI drops signal liquidation cascades.
3. **Liquidation Cascades**: Forced selling creates toxic flow. Detect via OI drop + extreme funding.
4. **Vault Competition**: Other market makers (HLP vault, etc.) compete for queue position.

### Cross-Exchange Dynamics

Binance BTC perp is the most liquid venue. Price discovery happens there first.
- Binance leads Hyperliquid by 50-500ms depending on volatility
- This lead-lag is exploitable but decaying as arbitrageurs compete

---

## Coding Conventions

### Rust Style

```rust
// Use strong types for domain concepts
struct Bps(f64);
struct Price(f64);
struct Size(f64);

// Document units in variable names
let spread_bps: f64 = 5.0;
let time_to_settlement_s: f64 = 3600.0;
let funding_rate_8h: f64 = 0.0001;

// Use const for magic numbers
const MAKER_FEE_BPS: f64 = 1.5;
const CASCADE_OI_DROP_THRESHOLD: f64 = -0.02;
```

### Critical Invariants

```rust
assert!(ask_price > bid_price);           // Spread positive
assert!(kappa > 0.0);                     // Or GLFT blows up
assert!(inventory.abs() <= max_inventory); // Risk limits
assert!(bid_price < microprice);          // Correct side of mid
assert!(ask_price > microprice);
```

---

## Implementation Priority

Build components in this order (each depends on previous):

1. **Week 1-2**: measurement-infrastructure, calibration-analysis
2. **Week 3-4**: signal-audit
3. **Week 5-6**: lead-lag-estimator (highest edge/effort)
4. **Week 7-8**: regime-detection-hmm
5. **Week 9-10**: adverse-selection-classifier
6. **Week 11-12**: fill-intensity-hawkes
7. **Week 13**: quote-engine integration, daily-calibration-report

---

## Anti-Patterns

1. **Trusting Backtest Results**: Always paper trade 1+ week before live
2. **Overfitting**: Need 100+ samples per parameter minimum
3. **Ignoring Regime Shifts**: Always track metrics by regime
4. **Point Estimates Without Uncertainty**: Use distributions, widen when uncertain
5. **Synchronous Model Updates**: Never retrain in the hot path

---

## Notes for Claude

1. **Always check calibration metrics** before and after changes
2. **The foundation skills are not optional** - read measurement-infrastructure first
3. **Prefer explicit over clever** - bugs cost real money
4. **Units matter** - bps vs fraction, seconds vs milliseconds, 8h vs annualized
5. **Regime matters** - single parameter values are almost always wrong
6. **Defense wins** - when in doubt, widen spreads
7. **Manual Execution Only**: The user wants to run all code manually. Do NOT execute binaries or scripts (e.g. `cargo run`, `./scripts/...`). Provide copy-pasteable blocks for the user to execute.

---

## Project Plans

**Always save implementation plans to `.claude/plans/` inside the project directory.**

```
.claude/
└── plans/
    └── <descriptive-name>.md
```

When entering plan mode or creating implementation plans:
1. Create the plan file at `.claude/plans/<descriptive-name>.md`
2. Use kebab-case naming (e.g., `feature-engineering-improvements.md`)
3. Include: objectives, phases, files to modify, verification steps
4. Reference plan files in Serena session memories for continuity

This keeps plans version-controlled with the project and accessible across sessions.