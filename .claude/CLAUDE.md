# CLAUDE.md - Hyperliquid Market Making System

Rust-based quantitative market making for Hyperliquid perpetual futures using GLFT optimal market making with proprietary extensions. See @README.md for full project overview.

---

## Project Layout

```
src/market_maker/
  adaptive/        — blended kappa, learned floor, shrinkage gamma
  adverse_selection/ — pre-fill classifier, microstructure features
  analytics/       — PnL attribution, edge metrics, Sharpe, persistence
  belief/          — Bayesian belief tracking
  calibration/     — Brier score, IR, model gating, prediction logging
  checkpoint/      — checkpoint persistence
  config/          — runtime configuration
  control/         — stochastic controller, value function, actions
  core/            — component initialization, shared state
  edge/            — AB testing, signal health
  estimator/       — kappa, sigma, regime, flow, mutual info estimators
  events/          — event types
  execution/       — order lifecycle, fill tracker
  fills/           — fill consumption, deduplication
  infra/           — connection supervisor, rate limiting, metrics
  latent/          — latent state models
  learning/        — adaptive ensemble, confidence, execution learning
  messages/        — WS message types and processors
  monitoring/      — alerter, dashboard, postmortem
  multi/           — multi-asset coordination
  orchestrator/    — event loop, handlers, quote engine, reconciliation
  process_models/  — spread dynamics, funding, liquidation, HJB
  quoting/         — quote filtering, ladder generation, entropy
  risk/            — monitors, circuit breaker, kill switch, position guard
  safety/          — auditor
  simulation/      — fill simulator, Monte Carlo
  stochastic/      — conjugate priors, HJB solver, continuation
  strategy/        — GLFT strategy, signal integration, position manager
  tracking/        — calibration, order, PnL, position, signal decay
```

---

## Build & Verification

```bash
cargo build                      # Dev build
cargo build --release            # Release build (incremental, fast)
cargo build --profile release-prod  # Production build (LTO, slow)

cargo test                       # Run all tests
cargo test <test_name>           # Prefer single tests for speed
cargo clippy -- -D warnings      # Lint — treat warnings as errors

# Binaries (user runs these manually, never run them yourself UNLESS explicitly asked)
# market_maker (paper mode: `market_maker paper`), parameter_estimator, health_dashboard, calibration_report
```

IMPORTANT: After any code change, run `cargo clippy -- -D warnings` and fix all warnings before considering the task done.

---

## Skill System

Domain knowledge lives in `.claude/skills/`, not here. Skills are loaded on demand — read them when relevant to the task. Workflow skills (`/debug-pnl`, `/add-signal`, `/paper-trading`) run as forked subagents.

| Task | Skills / Workflows |
|------|-------------------|
| Any model work | `measurement-infrastructure` FIRST, always |
| Debug PnL issue | `/debug-pnl` workflow, or: `calibration-analysis` -> component skill |
| Add new signal | `/add-signal` workflow, or: `signal-audit` -> `measurement-infrastructure` |
| Improve fill prediction | `fill-intensity-hawkes` -> `signal-audit` |
| Losing money in cascades | `adverse-selection-classifier` + `regime-detection-hmm` + `risk-management` |
| Cross-exchange edge decay | `lead-lag-estimator` |
| Wire up component | `quote-engine` + `infrastructure-ops` |
| Risk/safety changes | `risk-management` |
| Controller/decision logic | `stochastic-controller` |
| Paper trading issues | `/paper-trading` workflow |

---

## Core Rules

IMPORTANT — these are non-negotiable:

1. **Measurement before modeling** — never build a model without first defining the prediction target, setting up logging, and establishing baseline metrics
2. **Calibration is ground truth** — track Brier Score, Information Ratio, and Conditional Calibration for every model
3. **Everything is regime-dependent** — never use a single parameter value; kappa varies 10x, gamma varies 5x between quiet and cascade
4. **Defense first** — when uncertain, widen spreads. Missing a trade is cheap; getting run over in a cascade is not
5. **Manual execution only** — Do not run binaries or scripts (`cargo run`, `./scripts/...`) unless the user explicitly asks you to. By default, provide copy-pasteable blocks for the user to execute
6. **Prefer explicit over clever** — bugs cost real money
7. **Edge decays** — build staleness detection into every model; monitor signal MI, lead-lag R^2, regime drift

---

## Workflow

- **Verify your work**: Run tests and clippy after changes. If you can't verify it, flag it to the user
- **Plans**: Save to `.claude/plans/<descriptive-name>.md` (kebab-case). Include objectives, phases, files to modify, verification steps
- **Compaction**: When compacting, preserve the full list of modified files, test commands used, and any calibration metrics discussed
- **Subagents**: Use for codebase exploration and investigation to keep main context clean

## Memory Management

Keep memory files continuously updated throughout every session — don't wait until the end.

- **Auto memory** (`~/.claude/projects/.../memory/MEMORY.md`): Update after every significant milestone — commit, fix validated, architecture decision made, bug discovered, or live run completed. Keep the timeline table and "Current State" section current.
- **Serena memories** (`.serena/memories/`): Write a session memory at the end of each session with: what changed, why, validation results, and open issues. Name format: `session_YYYY-MM-DD_<topic>`.
- **What to record immediately**: bug root causes, parameter values that worked/failed, architectural decisions with rationale, test results, live run metrics, anything that cost time to figure out.
- **Keep it honest**: Record actual commit hashes, real metrics, and ground truth — not aspirational numbers. If something is broken, say so.

---

## Common Gotchas

- `kappa` must be > 0.0 or GLFT formula blows up
- Maker fee is 1.5 bps on Hyperliquid — always include in spread calculation
- Binance leads Hyperliquid by 50-500ms — this lead-lag is exploitable but decaying
- Funding rate settlement creates predictable flow patterns — don't ignore near settlement
- OI drops signal liquidation cascades — widen immediately

---

See also: @.claude/rules/ for git workflow, code style, agent teams, and safety rules.
