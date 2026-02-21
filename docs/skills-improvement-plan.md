# Skills & Domain Knowledge: Audit + Improvement Plan

## Current Inventory

### What Exists (16 skills, 5 workflows)

```
foundation/ (3)
  measurement-infrastructure     # Prediction logging, outcome tracking
  calibration-analysis           # Brier, IR, conditional calibration, PnL attribution
  signal-audit                   # Mutual information, signal quality, decay tracking

models/ (4)
  fill-intensity-hawkes          # Kappa estimation, Hawkes process
  adverse-selection-classifier   # Trade classification, 26-feature MLP
  regime-detection-hmm           # 4-state HMM, Baum-Welch, belief blending
  lead-lag-estimator             # Binance→HL cross-exchange signal

domains/ (4)
  risk-management                # Layered monitors, circuit breakers, kill switch
  infrastructure-ops             # Event loop, WS, rate limiting, reconnection
  checkpoint-management          # Persistence, prior transfer, CalibrationGate
  stochastic-controller          # Bayesian beliefs, HJB solver, changepoint, VOI

integration/ (1)
  quote-engine                   # Full pipeline: market data → GLFT → final quotes

operations/ (2)
  daily-calibration-report       # Automated health check, alerting
  live-incident-response         # Incident taxonomy, triage, key file map

workflows/ (5)
  debug-pnl                      # PnL diagnostic routing
  go-live                        # Paper→live transition, parity audit
  add-signal                     # New signal end-to-end
  add-asset                      # New asset onboarding
  paper-trading                  # Paper trader setup and debugging
```

### Agent Skill Preloads (Current)

| Agent | Preloads | Missing (per routing table) |
|-------|----------|----------------------------|
| signals | measurement-infrastructure, signal-audit, calibration-analysis | **fill-intensity-hawkes, adverse-selection-classifier, regime-detection-hmm, lead-lag-estimator** |
| strategy | quote-engine, stochastic-controller | **regime-detection-hmm, risk-management, lead-lag-estimator** |
| infra | infrastructure-ops, quote-engine | **checkpoint-management** |
| analytics | calibration-analysis, daily-calibration-report | **measurement-infrastructure, signal-audit** |
| risk | *(none)* | **risk-management, live-incident-response** |
| code-reviewer | *(none — read-only)* | — |

---

## Problem Analysis

### P1: Missing Skills (5 gaps)

These are referenced in CLAUDE.md, agents, or plans but don't exist:

| Skill | Where Referenced | Impact |
|-------|-----------------|--------|
| **learning-pipeline** | CLAUDE.md routing table ("future work") | RL agent, adaptive ensemble, confidence tracking, baseline tracker — agent reads raw source instead |
| **spread-chain** | live-incident-response (7 multiplicative factors), multiple incident post-mortems | The spread composition pipeline (staleness, model gating, toxicity, edge, etc.) is the #1 source of production bugs and has no skill |
| **multi-asset** | add-asset workflow, config/multi_asset.rs | Allocator, margin pool, cross-asset correlation, OI caps — scattered across workflow steps |
| **config-derivation** | add-asset workflow, go-live workflow | `auto_derive.rs` first-principles parameter derivation is critical but undocumented as a skill |
| **fill-processing** | go-live (FillProcessor), live-incident-response | Unified fill pipeline, deduplication, outcome tracking — involved in most incidents |

### P2: Agent Preload Gaps

The routing table in CLAUDE.md says "losing money in cascades → adverse-selection-classifier + regime-detection-hmm + risk-management" but signals agent only preloads 3 of 7 skills it needs. When the model hits a cascade debugging task, it has to discover and load skills mid-session rather than starting with them. This wastes turns and context.

**Quantified**: signals agent is missing 4 model skills. strategy is missing 3. risk preloads *zero* skills despite owning `risk/`, `safety/`, `monitoring/`.

### P3: Structural Issues

**3a. Implementation files are aspirational, not verified.**
Every model skill has an `implementation.md` with full Rust structs and functions. But these are *reference implementations*, not copies of actual code. After months of development and incidents, actual code has diverged. Example: `fill-intensity-hawkes/implementation.md` shows `HyperliquidFillIntensityModel` with specific fields — actual `estimator/kappa.rs` has evolved through multiple incident fixes.

**3b. No dependency declarations in frontmatter.**
Skills reference dependencies in prose ("Requires: measurement-infrastructure") but there's no machine-readable way to declare this. When an agent preloads `fill-intensity-hawkes` it doesn't automatically get `measurement-infrastructure`.

**3c. Content duplication across skills.**
The GLFT formula `delta* = (1/gamma) * ln(1 + gamma/kappa)` appears in: quote-engine, stochastic-controller, strategy agent prompt, and fill-intensity-hawkes. When the formula context changes, 4 files need updating.

**3d. Skills don't track codebase anchors.**
Skills reference files by path but not line numbers, and there's no "last verified" date. After a refactor, the paths may be stale. The live-incident-response skill *does* have line references (e.g., `kill_switch.rs:443`) — this should be the standard, not the exception.

**3e. No tiered loading strategy.**
All skills are equal weight. But `measurement-infrastructure` should load before ANY model work (Core Rule #1), while `daily-calibration-report` is only needed for operational review. There's no way to express "always load before X" or "only load when explicitly invoked."

### P4: Content Quality Issues

**4a. Skills are too long for preloading.**
`quote-engine/SKILL.md` + `implementation.md` is ~800 lines. When preloaded into an agent's context, this consumes significant budget. Skills need a "preload summary" (key decisions, invariants, file map) vs "full reference" (implementation details) split.

**4b. Skills don't encode incident learnings.**
The live-incident-response skill has an excellent incident history. But the individual model skills don't reference the incidents that changed them. Example: the "kappa cycle skip" incident (Feb 10) should be a warning in fill-intensity-hawkes: "Never compare raw kappa against thresholds — use kappa_effective."

**4c. Workflow skills don't have verification scripts.**
As identified in the architecture redesign, workflows use prose gates ("STOP HERE if...") instead of executable verification. The `/go-live` workflow has bash verification snippets inline but no standalone scripts that could be referenced by hooks.

---

## Improvement Plan

### Tier 1: High Impact / Low Effort

#### 1.1 Fix agent preloads (30 min)

Update each agent's `skills:` frontmatter to include ALL skills from the routing table:

```yaml
# signals.md
skills:
  - measurement-infrastructure
  - signal-audit
  - calibration-analysis
  - fill-intensity-hawkes
  - adverse-selection-classifier
  - regime-detection-hmm
  - lead-lag-estimator

# strategy.md
skills:
  - quote-engine
  - stochastic-controller
  - regime-detection-hmm
  - risk-management
  - lead-lag-estimator

# infra.md
skills:
  - infrastructure-ops
  - quote-engine
  - checkpoint-management

# analytics.md
skills:
  - calibration-analysis
  - daily-calibration-report
  - measurement-infrastructure
  - signal-audit

# risk.md (currently preloads NOTHING)
skills:
  - risk-management
  - live-incident-response
```

**After this change, remove the routing table from CLAUDE.md** — it becomes redundant.

#### 1.2 Add `requires` to skill frontmatter (20 min)

Add dependency declarations so skills can express load ordering:

```yaml
---
name: fill-intensity-hawkes
description: ...
requires:
  - measurement-infrastructure
  - signal-audit
---
```

This doesn't change runtime behavior today, but it documents the DAG explicitly and enables future automated loading.

#### 1.3 Add incident cross-references to model skills (1 hr)

Add a `## Known Issues from Production` section to each model skill referencing relevant incidents:

**fill-intensity-hawkes** add:
```markdown
## Known Issues from Production

- **Feb 10: Kappa cycle skip** — Raw kappa (~6400-7700 from L2) was compared against
  a 5000 threshold. Should have used `kappa_effective` (~3250, blended). Feature was
  removed as redundant. See live-incident-response/incident-history.md.
- **Feb 10: Duplicate on_trade double-counting** — Two `on_trade` calls in handlers.rs
  plus canonical one in messages/trades.rs caused 2x kappa inflation. Removed duplicate.
```

**adverse-selection-classifier** add:
```markdown
## Known Issues from Production

- **Mainnet analysis: 2x AS overestimate** — AS classifier was double-counting in
  certain flow conditions, leading to spreads wider than necessary.
```

**regime-detection-hmm** add:
```markdown
## Known Issues from Production

- **Feb 10: Cold-start staleness 2.0x** — `staleness_spread_multiplier()` penalized
  signals with `observation_count == 0` (cold-start) same as signals that lost data
  (actually stale). Added `was_ever_warmed_up()` guard.
```

**risk-management** add:
```markdown
## Known Issues from Production

- **Feb 9: Drawdown denominator bug** — `summary()` divided by `peak_pnl` (tiny after 1
  fill) showing 1000%+ drawdown. Fixed to use `account_value` denominator with
  `min_peak_for_drawdown` guard. See risk/state.rs:253-262.
```

### Tier 2: High Impact / Medium Effort

#### 2.1 Create `spread-chain` skill (NEW — highest-value missing skill)

This is the single most impactful missing skill. The spread composition pipeline is the #1 source of production incidents (3 of 10 documented incidents involve spread multipliers).

```
.claude/skills/domains/spread-chain/
├── SKILL.md
└── implementation.md
```

**SKILL.md content outline:**

```markdown
---
name: spread-chain
description: The 7 multiplicative factors that compose the final spread from GLFT
  optimal through to exchange order price. Use when debugging wide spreads, investigating
  spread multiplier compounding, tuning defensive behavior, or understanding why
  quotes are wider than expected. Critical for incident triage.
requires:
  - regime-detection-hmm
  - risk-management
---

# Spread Chain Skill

## The Pipeline

Raw GLFT spread passes through 7 multiplicative stages:

1. **GLFT optimal** — `(1/gamma) * ln(1 + gamma/kappa) + fee`
2. **Regime floor** — `max(spread, regime_floor_bps)`
3. **Staleness multiplier** — from signal_integration.rs:999
4. **Model gating multiplier** — from signal_integration.rs:978
5. **Edge defensive multiplier** — from analytics/edge_metrics.rs:192
6. **Toxicity multiplier** — from analytics/market_toxicity.rs:64,98
7. **Risk overlay** — from circuit_breaker.rs + cascade monitor

## Critical Invariant

Multipliers compound multiplicatively:
  `1.5 × 1.5 × 1.5 = 3.375x`

Global cap: 10.0x (market_toxicity.rs:64)

## Incident Pattern: Multiplicative Compounding

This is the most common spread bug. Multiple independent factors each add "a little"
defense, but the product makes spreads 3-5x wider than any single factor justifies.

**Defense**: Log all multiplier components. Set global cap. Consider additive
composition for factors that should be independent.

## Key File Map

| Component | File | Key Lines |
|-----------|------|-----------|
| GLFT base | orchestrator/quote_engine.rs | ~line 450 |
| Staleness mult | strategy/signal_integration.rs | 999-1033 |
| Model gating mult | strategy/signal_integration.rs | 978 |
| Edge defensive | analytics/edge_metrics.rs | 192 |
| Toxicity | analytics/market_toxicity.rs | 64, 98 |
| Risk overlay | control/mod.rs | 775-784 |
| Global cap | analytics/market_toxicity.rs | 64 (10.0x) |
| Spread composition | orchestrator/quote_engine.rs | 992-1107 |

## Debugging Wide Spreads

1. Check each multiplier independently — which one is elevated?
2. If staleness: is a signal source actually stale, or is it cold-start?
3. If model gating: which model has low IR?
4. If toxicity: is it justified (real cascade) or stuck (stale state)?
5. If multiple: are they independent, or double-counting the same signal?
```

#### 2.2 Create `learning-pipeline` skill (NEW)

```
.claude/skills/domains/learning-pipeline/
├── SKILL.md
└── references/
    └── feedback-loops.md
```

Covers:
- The 9 learning feedback loops (currently documented only in go-live workflow)
- RL agent: Q-table, reward function, cold-start seeding, ensemble participation
- Adaptive ensemble: model weights, Dirichlet updates
- Confidence tracking: observation counts, warmup progression
- Baseline tracker: counterfactual reward for RL
- Quote outcome tracker: unbiased edge measurement

This skill fills the "future work" gap in CLAUDE.md and gives the analytics agent proper domain knowledge for the learning system.

#### 2.3 Create `config-derivation` skill (NEW)

Covers `auto_derive.rs` — the first-principles parameter derivation that computes gamma, kappa, max_position, spread profile from market characteristics. Currently referenced by add-asset and go-live workflows but never documented as standalone knowledge.

#### 2.4 Split skills into preload summaries + full references

For each skill, create a two-tier structure:

```
models/fill-intensity-hawkes/
├── SKILL.md              # Preload summary (~50 lines): key decisions, invariants,
│                         # API surface, file map, known issues
└── references/
    ├── implementation.md  # Full reference: all Rust code, derivations
    ├── estimation.md      # Batch + online estimation details
    └── incidents.md       # Production incidents involving this component
```

The SKILL.md becomes the "context-efficient" version that agents preload. The references/ directory is read on-demand when doing deep work on that component.

**Restructure priority** (by preload frequency):
1. quote-engine (preloaded by signals + strategy + infra = 3 agents)
2. measurement-infrastructure (preloaded by signals + analytics = 2 agents, also Core Rule #1)
3. regime-detection-hmm (preloaded by signals + strategy = 2 agents)
4. risk-management (preloaded by strategy + risk = 2 agents)

### Tier 3: Medium Impact / Higher Effort

#### 3.1 Add codebase anchors with verification

Add `## File Map` sections to every skill with `file:line` references, plus a verification script:

```bash
#!/bin/bash
# .claude/scripts/verify-skill-anchors.sh
# Checks that file:line references in skills still exist

errors=0
while IFS=: read -r file line pattern; do
  if ! sed -n "${line}p" "$file" | grep -q "$pattern"; then
    echo "STALE: $file:$line should contain '$pattern'"
    ((errors++))
  fi
done < .claude/skills/.anchors

echo "$errors stale anchors found"
exit $errors
```

The `.anchors` file is generated from skill content and checked periodically. This catches drift after refactors.

#### 3.2 Create `fill-processing` skill (NEW)

Covers the unified fill pipeline: FillProcessor, deduplication, pending fill outcomes, markout tracking. Multiple incidents (9 of 10 missing learning loops, duplicate on_trade) stem from this pipeline.

#### 3.3 Create `multi-asset` skill (NEW)

Extracts multi-asset knowledge currently buried in add-asset workflow: AssetAllocator, inverse-volatility weighting, margin pool, cross-asset position limits, OI caps.

#### 3.4 Add `stale-if` conditions to skills

Each skill gets a structured "this skill needs updating if..." section:

```yaml
---
name: risk-management
stale-if:
  - "New monitor added to risk/monitors/ not documented here"
  - "RiskSeverity enum changed"
  - "Kill switch reasons modified"
  - "Circuit breaker thresholds changed"
last-verified: "2026-02-15"
---
```

This gives any engineer a checklist for whether a skill is current.

#### 3.5 Workflow verification scripts

For each workflow, extract the inline bash verification into standalone scripts:

```
workflows/go-live/
├── SKILL.md
└── verify/
    ├── phase1_paper_thresholds.sh    # Parses logs, checks edge > 0, fills > 10/hr, etc.
    ├── phase2_parity_audit.sh        # Checks all 9 learning loops are wired
    ├── phase3_config_validation.sh   # Validates config consistency
    └── phase4_monitoring_checklist.sh # Checks dashboard, alerts, metrics
```

Each script outputs structured JSON: `{"phase": 1, "checks": [...], "passed": true}`.
Workflows reference these scripts as required steps. Future hooks can gate progression.

---

## Dependency Graph (After Improvements)

```
                    measurement-infrastructure  ←── Core Rule #1: always load first
                    /           |           \
                   /            |            \
          signal-audit   calibration-analysis  (all model work)
              |                 |
    ┌─────────┼─────────┬──────┼──────────────────────┐
    ▼         ▼         ▼      ▼                       ▼
fill-intensity  adverse-sel  regime-hmm  lead-lag    spread-chain (NEW)
    hawkes      classifier   detection   estimator
    |           |            |           |              |
    └─────┬─────┘            |           |              |
          ▼                  ▼           ▼              ▼
      quote-engine  ←── stochastic-controller ←── risk-management
          |                                            |
          ▼                                            ▼
   infrastructure-ops                          live-incident-response
          |
          ▼
   checkpoint-management ←── learning-pipeline (NEW)
                                    |
                              config-derivation (NEW)
                                    |
                              fill-processing (NEW)
                                    |
                              multi-asset (NEW)

Workflows (invoke full pipelines):
  /debug-pnl      → calibration-analysis → route to component skill
  /add-signal      → signal-audit → measurement-infrastructure → component
  /paper-trading   → checkpoint-management → learning-pipeline
  /go-live         → learning-pipeline → live-incident-response
  /add-asset       → config-derivation → multi-asset → /paper-trading

Operations (scheduled/manual):
  daily-calibration-report → calibration-analysis
  spread-chain             → (incident triage)
```

## Migration Order

| Week | Action | Files Changed | Impact |
|------|--------|---------------|--------|
| 1 | Fix agent preloads + add `requires` to frontmatter | 5 agent .md files, 16 skill SKILL.md files | Agents start with right knowledge |
| 1 | Add incident cross-references to model skills | 4 SKILL.md files | Production learnings don't get lost |
| 1 | Remove routing table from CLAUDE.md | CLAUDE.md | ~20 lines of context budget recovered |
| 2 | Create spread-chain skill | New skill directory | #1 incident source gets documented |
| 2 | Create learning-pipeline skill | New skill directory | Fills "future work" gap |
| 3 | Split top-4 skills into preload + reference | 4 skill directories restructured | Context budget savings per session |
| 3 | Create config-derivation skill | New skill directory | Critical undocumented knowledge |
| 4 | Add codebase anchors + verification script | All skills, new script | Drift detection |
| 4 | Create fill-processing skill | New skill directory | Incident pattern coverage |
| 5 | Create multi-asset skill | New skill directory | Operational knowledge |
| 5 | Add workflow verification scripts | 5 verify/ directories | Phase gates become executable |

## Context Budget Impact

| Change | Lines Saved/Added | Net per Session |
|--------|-------------------|-----------------|
| Remove routing table from CLAUDE.md | -20 lines | -20 |
| Agent preloads (right skills loaded, wrong ones not) | ±0 but *correct* context | Quality improvement |
| Split skills to preload summaries | -600 lines (across top-4 skills) | -150 per agent avg |
| New skills (5) | +250 lines (summaries only) | +50 per relevant session |
| Incident cross-refs | +40 lines | +10 per model skill |
| **Net** | | **~-110 lines per session** |

The net effect is *less* context consumed with *more relevant* knowledge loaded, because agents preload focused summaries instead of discovering and reading full skills mid-session.
