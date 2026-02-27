---
description: Memory and plan management workflow
globs:
---

# Memory Management

## Memory Layers

Three layers, each with a distinct role:

1. **Auto memory MEMORY.md** (`~/.claude/projects/-home-trudumb-hyper-make/memory/MEMORY.md`)
   - Loaded into EVERY session automatically — keep under 200 lines
   - Contains: current state, topic file index, key patterns (top 12), recent changes
   - Update after: commits, architecture decisions, production runs, significant milestones

2. **Auto memory topic files** (`~/.claude/projects/-home-trudumb-hyper-make/memory/*.md`)
   - Read on demand when relevant to current task — can be longer
   - `architecture.md` — module relationships, GLFT flow, key APIs, wiring patterns
   - `debugging.md` — bug root causes, fix patterns, production incidents
   - `live-trading.md` — production results, parameter values, incidents, postmortems
   - `agent-work-log.md` — what each agent team has built/fixed, with dates
   - Update when: new bugs found, production runs completed, architecture changes, agent work done

3. **Agent memory** (`.claude/agent-memory/{agent}/MEMORY.md`)
   - Lean (~20-30 lines): current open issues, pending work, domain-specific gotchas
   - NOT for historical records (those go in auto memory topic files)
   - Update when: issues resolved, new domain-specific gotchas discovered

4. **Serena memories** (`.serena/memories/`)
   - Archived session logs — read only when investigating historical incidents
   - No longer the primary storage for session knowledge
   - New session insights go into auto memory topic files instead

## What Goes Where

| Content Type | Destination |
|-------------|-------------|
| Key pattern / lesson learned | MEMORY.md "Key Patterns" section |
| Bug root cause + fix | `debugging.md` |
| Architecture decision | `architecture.md` |
| Production run results | `live-trading.md` |
| Agent implementation work | `agent-work-log.md` |
| Agent-specific open issue | `.claude/agent-memory/{agent}/MEMORY.md` |
| Recent significant change | MEMORY.md "Recent Changes" table |
| Current project phase | MEMORY.md "Current State" section |

## Update Discipline

- **Record immediately**: bug root causes, parameter values that worked/failed, architectural decisions with rationale, test results, live run metrics
- **Keep it honest**: actual commit hashes, real metrics, ground truth — not aspirational numbers
- Plans and memories are **living documents** — update continuously as work progresses

## Plan Workflow

- Save plans to `.claude/plans/<descriptive-name>.md` (kebab-case)
- Include objectives, phases, files to modify, verification steps
- **Update the plan as you work** — mark phases complete, note deviations, add discoveries
- Plans are not write-once; they evolve with the implementation

## Compaction

When compacting, preserve the full list of modified files, test commands used, and any calibration metrics discussed.

## Subagents

Use for codebase exploration and investigation to keep main context clean.
