---
description: Agent team coordination rules and file ownership
globs:
---

# Agent Teams

Agent teams enabled via `.claude/settings.json`. Custom agents in `.claude/agents/` encode domain ownership, preloaded skills, review checklists, and coordination rules. Use teams for parallel work on independent modules; use subagents for sequential tasks or same-file edits.

## Agents

`signals`, `strategy`, `infra` (plan mode), `risk` (plan mode), `analytics`, `code-reviewer` (read-only)

## Compositions

- **full-feature** (5: all agents) — large cross-cutting features
- **model-improvement** (3: signals, strategy, analytics) — estimation and calibration work
- **bug-investigation** (3+: hypothesis testing) — parallel root cause analysis
- **code-review** (1: code-reviewer) — fast automated review

## Key Rules

- File ownership is exclusive — no two teammates edit the same file
- `signal_integration.rs` is strategy-only; `mod.rs` re-exports are lead-only
- Plan approval required for `orchestrator/`, `risk/`, `safety/`, `exchange/`, `src/bin/`
- All teammates run clippy, no hardcoded params, `#[serde(default)]` on checkpoint fields
- Teammates must NOT run binaries — user executes manually

## File Ownership

See `.claude/ownership.json` for the machine-readable ownership manifest (enforced by hooks).

The lead owns: `src/market_maker/mod.rs`, `config/`, `belief/`, `multi/`, `latent/`, `src/bin/`, `src/exchange/`, `src/ws/`, `src/info/`, `src/lib.rs`, `.claude/`, `Cargo.toml`
