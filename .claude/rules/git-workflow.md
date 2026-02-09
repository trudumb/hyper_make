---
description: Git & GitHub workflow conventions for this project
globs:
---

# Git & GitHub Workflow

IMPORTANT: Use `gh` CLI for ALL GitHub interactions. Never use raw API calls or web fetch.

## Commit Conventions

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

## Branch Strategy

- **No direct commits to `main`** — all changes go through PRs
- Branch naming: `feat/<name>`, `fix/<name>`, `refactor/<name>`, `perf/<name>`
- Merge only after tests pass and clippy is clean

## PR Workflow

1. Create feature branch: `git checkout -b feat/<name>`
2. Implement changes, run `cargo test && cargo clippy -- -D warnings`
3. Commit with conventional format (atomic commits — one logical change per commit)
4. Push and create PR: `gh pr create --title "..." --body "..."`
5. PR body must include: **Summary** (what & why), **Test plan** (how to verify), **Risk** (what could go wrong for a trading system)
6. Wait for CI checks: `gh pr checks <number>`
7. After review approval, squash merge: `gh pr merge <number> --squash`

## Code Review Checklist

When reviewing PRs, use `gh` to fetch all context:
```bash
gh pr view <number> --json title,body,author,files,reviews
gh pr diff <number>
gh pr checks <number>
```

Review checklist:
- Calibration metrics validated before/after
- No hardcoded parameters (must be regime-dependent)
- Units documented in variable names
- Risk limits respected (`inventory.abs() <= max_inventory`)
- Spread invariants maintained (`ask > bid`, both correct side of microprice)
