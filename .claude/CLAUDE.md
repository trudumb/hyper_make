# CLAUDE.md - Hyperliquid Market Making System

Rust-based quantitative market making for Hyperliquid perpetual futures using GLFT optimal market making with proprietary extensions. See @README.md for full project overview.

---

## Build & Verification

```bash
# Build
cargo build                      # Dev build
cargo build --release            # Release build (incremental, fast)
cargo build --profile release-prod  # Production build (LTO, slow)

# Test & lint — ALWAYS run after making changes
cargo test                       # Run all tests
cargo test <test_name>           # Prefer single tests for speed
cargo clippy -- -D warnings      # Lint — treat warnings as errors

# Binaries (user runs these manually, never run them yourself UNLESS explicitly asked)
# market_maker, parameter_estimator, paper_trader, health_dashboard, calibration_report
```

IMPORTANT: After any code change, run `cargo clippy -- -D warnings` and fix all warnings before considering the task done.

---

## Coding Conventions

- Use strong newtypes for domain concepts: `Bps(f64)`, `Price(f64)`, `Size(f64)`
- Document units in variable names: `spread_bps`, `time_to_settlement_s`, `funding_rate_8h`
- Use `const` for magic numbers, never inline
- `bps` vs `fraction`, `seconds` vs `milliseconds`, `8h` vs `annualized` — unit mismatches cause real money loss

---

## Skill System

Domain knowledge lives in `.claude/skills/`, not here. Skills are loaded on demand — read them when relevant to the task.

| Task | Read these skills (in order) |
|------|------------------------------|
| Any model work | `measurement-infrastructure` FIRST, always |
| Debug PnL issue | `calibration-analysis` → weak component's skill |
| Add new signal | `signal-audit` → `measurement-infrastructure` |
| Improve fill prediction | `measurement-infrastructure` → `signal-audit` → `fill-intensity-hawkes` |
| Losing money in cascades | `adverse-selection-classifier`, `regime-detection-hmm` |
| Cross-exchange edge decay | `lead-lag-estimator` |
| Wire up new component | `quote-engine` |

---

## Core Rules

IMPORTANT — these are non-negotiable:

1. **Measurement before modeling** — never build a model without first defining the prediction target, setting up logging, and establishing baseline metrics
2. **Calibration is ground truth** — track Brier Score, Information Ratio, and Conditional Calibration for every model
3. **Everything is regime-dependent** — never use a single parameter value; kappa varies 10x, gamma varies 5x between quiet and cascade
4. **Defense first** — when uncertain, widen spreads. Missing a trade is cheap; getting run over in a cascade is not
5. **Manual execution only** — Do not run binaries or scripts (`cargo run`, `./scripts/...`) unless the user explicitly asks you to. By default, provide copy-pasteable blocks for the user to execute
6. **Prefer explicit over clever** — bugs cost real money
7. **Edge decays** — build staleness detection into every model; monitor signal MI, lead-lag R², regime drift

---

## Git & GitHub Workflow

IMPORTANT: Use `gh` CLI for ALL GitHub interactions. Never use raw API calls or web fetch.

### Commit Conventions

This project uses **conventional commits**. Write in imperative mood, explain "why" not "what":

```
feat(model): Add Hawkes process intensity estimator
fix(quote-engine): Correct spread floor in cascade regime
refactor(signals): Extract lead-lag into standalone module
perf(ws): Reduce Binance feed latency with zero-copy parsing
test(calibration): Add Brier score regression tests
docs(skills): Update measurement-infrastructure skill
chore(deps): Bump tokio to 1.35
```

Scopes: `model`, `quote-engine`, `signals`, `exchange`, `risk`, `calibration`, `ws`, `config`, `deps`, `skills`

### Branch Strategy

- **No direct commits to `main`** — all changes go through PRs
- Branch naming: `feat/<name>`, `fix/<name>`, `refactor/<name>`, `perf/<name>`
- Merge only after tests pass and clippy is clean

### GitHub CLI Commands

```bash
# Branches & PRs
gh pr create --title "feat(model): Add regime HMM" --body "## Summary\n- ..."
gh pr view <number>                    # View PR details
gh pr diff <number>                    # View PR diff
gh pr checks <number>                  # Check CI status
gh pr merge <number> --squash          # Squash merge after approval
gh pr list --state open                # List open PRs

# Code review
gh pr review <number> --approve
gh pr review <number> --request-changes --body "..."
gh pr comment <number> --body "..."

# Issues
gh issue create --title "..." --body "..." --label "bug"
gh issue list --label "model" --state open
gh issue view <number>
gh issue close <number> --reason completed

# CI/CD & repo
gh run list                            # List recent workflow runs
gh run view <run-id>                   # View run details
gh run watch <run-id>                  # Watch run in progress
gh release list                        # List releases
```

### PR Workflow

1. Create feature branch: `git checkout -b feat/<name>`
2. Implement changes, run `cargo test && cargo clippy -- -D warnings`
3. Commit with conventional format (atomic commits — one logical change per commit)
4. Push and create PR: `gh pr create --title "..." --body "..."`
5. PR body must include: **Summary** (what & why), **Test plan** (how to verify), **Risk** (what could go wrong for a trading system)
6. Wait for CI checks: `gh pr checks <number>`
7. After review approval, squash merge: `gh pr merge <number> --squash`

### Code Review via `gh`

When reviewing PRs, use `gh` to fetch all context:
```bash
gh pr view <number> --json title,body,author,files,reviews
gh pr diff <number>
gh pr checks <number>
```

Review checklist for this project:
- Calibration metrics validated before/after
- No hardcoded parameters (must be regime-dependent)
- Units documented in variable names
- Risk limits respected (`inventory.abs() <= max_inventory`)
- Spread invariants maintained (`ask > bid`, both correct side of microprice)

---

## Workflow

- **Verify your work**: Run tests and clippy after changes. If you can't verify it, flag it to the user
- **Plans**: Save to `.claude/plans/<descriptive-name>.md` (kebab-case). Include objectives, phases, files to modify, verification steps
- **Compaction**: When compacting, preserve the full list of modified files, test commands used, and any calibration metrics discussed
- **Subagents**: Use for codebase exploration and investigation to keep main context clean

---

## Agent Teams

Teams are enabled. Key rules:

1. **File ownership** — each teammate owns distinct files; no two teammates edit the same file
2. **Plan approval required** for changes to: `src/quote_engine/`, `src/risk/`, `src/exchange/`
3. **Teammates must NOT run binaries** — they produce code; user executes manually
4. **Defense-first applies to teams** — push back on aggressive parameter changes

---

## Common Gotchas

- `kappa` must be > 0.0 or GLFT formula blows up
- Maker fee is 1.5 bps on Hyperliquid — always include in spread calculation
- Binance leads Hyperliquid by 50-500ms — this lead-lag is exploitable but decaying
- Funding rate settlement creates predictable flow patterns — don't ignore near settlement
- OI drops signal liquidation cascades — widen immediately
