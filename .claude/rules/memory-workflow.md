---
description: Memory and plan management workflow
globs:
---

# Memory Management

Plans and memories are **living documents** — update them continuously as work progresses, not just at creation or session end. They should always reflect the current state of the work.

- **Auto memory** (`~/.claude/projects/.../memory/MEMORY.md`): Update after every significant milestone — commit, fix validated, architecture decision made, bug discovered, or live run completed. Keep the timeline table and "Current State" section current.
- **Serena memories** (`.serena/memories/`): Write a session memory at the end of each session with: what changed, why, validation results, and open issues. Name format: `session_YYYY-MM-DD_<topic>`.
- **Plans ↔ Memories in sync**: When a plan changes direction, update the memory to explain why. When a memory captures a discovery, update the plan to reflect it. Neither should go stale while the other advances.
- **What to record immediately**: bug root causes, parameter values that worked/failed, architectural decisions with rationale, test results, live run metrics, anything that cost time to figure out.
- **Keep it honest**: Record actual commit hashes, real metrics, and ground truth — not aspirational numbers. If something is broken, say so.

## Plan Workflow

- Save plans to `.claude/plans/<descriptive-name>.md` (kebab-case)
- Include objectives, phases, files to modify, verification steps
- **Update the plan as you work** — mark phases complete, note deviations, add discoveries
- Plans are not write-once; they evolve with the implementation

## Compaction

When compacting, preserve the full list of modified files, test commands used, and any calibration metrics discussed.

## Subagents

Use for codebase exploration and investigation to keep main context clean.
