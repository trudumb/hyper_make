# CLAUDE.md - Hyperliquid Market Making System

Rust-based quantitative market making for Hyperliquid perpetual futures using GLFT optimal market making. See `README.md` for overview.

## Core Rules

1. **Measurement before modeling** — define prediction target + logging + baseline before building any model
2. **Calibration is ground truth** — Brier Score, IR, Conditional Calibration for every model
3. **Everything is regime-dependent** — kappa varies 10x, gamma varies 5x between quiet and cascade
4. **Defense first** — when uncertain, widen spreads. Missing a trade is cheap; cascades are not
5. **Manual execution only** — never run trading binaries unless user explicitly asks (hook-enforced)
6. **Prefer explicit over clever** — bugs cost real money
7. **Edge decays** — build staleness detection into every model; monitor signal MI, lead-lag R^2, regime drift

## Build & Verification

```bash
cargo fmt --all -- --check              # 1. Check formatting first
cargo clippy --all-targets -- -D warnings  # 2. Lint — treat warnings as errors
cargo nextest run                       # 3. Run tests (isolated processes, OOM-safe)
```

**Resource constraint**: One cargo command at a time (hook-enforced mutex). Sequential only — concurrent builds crash the machine.

**BANNED: Full-suite `cargo test` / `cargo nextest run`** — Do NOT run the full test suite unless the user explicitly asks for it. The full suite is slow and resource-intensive. Instead, run only the specific test(s) relevant to your change:
```bash
cargo nextest run -E 'test(specific_test_name)'   # Single test
cargo nextest run -p specific_package              # Single crate
cargo test --lib -p specific_package               # Lib tests only for one crate
```

IMPORTANT: After any code change, run `cargo fmt --all -- --check` then `cargo clippy --all-targets -- -D warnings` and fix all issues before considering the task done.

## Structure

- **Domain knowledge**: `.claude/skills/` — loaded automatically with agents
- **File ownership**: `.claude/ownership.json` — hook-enforced agent boundaries
- **Agents**: `.claude/agents/` — signals, strategy, infra, analytics, risk, code-reviewer, lead
- **Workflows**: `/debug-pnl`, `/go-live`, `/paper-trading`, `/add-signal`, `/add-asset`
- **Rules**: `.claude/rules/` — git workflow, code style, agent teams, safety, domain gotchas, memory workflow
- **Hooks**: `.claude/hooks/` — cargo mutex, binary blocker, file ownership, protected paths, lint reminder

See `@.claude/rules/` for detailed conventions.
