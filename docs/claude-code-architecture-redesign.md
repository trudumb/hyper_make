# Claude Code Architecture Redesign

## Principled Enforcement & Orchestration for a Quantitative Trading System

*Designed as if a team of 6 engineers built it from day one with deterministic enforcement as a foundational assumption.*

---

## The Core Problem

The current system has **excellent domain knowledge** but **weak enforcement**. Every rule — sequential cargo commands, file ownership, plan approval for critical paths, binary execution blocking — exists as prose that the model *chooses* to follow. In a system where bugs cost real money, "choose to follow" is not an acceptable enforcement mechanism.

### Current State Audit

| Rule | Where It Lives | Enforcement Mechanism |
|------|---------------|----------------------|
| Sequential cargo commands | CLAUDE.md prose | None (model compliance) |
| File ownership | agent-teams.md prose + agent descriptions | None |
| Plan approval for orchestrator/ | Agent permissionMode field | Partial (agent-level, not file-level) |
| No binary execution | CLAUDE.md Core Rule #5 | None |
| Clippy after every edit | CLAUDE.md build section | None |
| measurement-infrastructure first | CLAUDE.md routing table | None |
| Workflow phase gates | Workflow SKILL.md prose ("STOP HERE if...") | None |

**Zero hooks are configured.** The settings.json hooks field is empty.

### Design Principle

> If you delete every prose rule and every CLAUDE.md instruction, the hooks STILL enforce sequential cargo, file ownership, protected paths, and binary blocking. Prose becomes documentation of *why* the rules exist, not the mechanism that enforces them.

---

## Team Assignments

| Engineer | Domain | Delivers |
|----------|--------|----------|
| **E1** | Enforcement Layer | 7 hooks that make rules deterministic |
| **E2** | Agent Architecture | Redesigned agents with complete skill preloading |
| **E3** | Workflow Pipelines | Gated workflows with verification scripts |
| **E4** | Observability | Session audit trail and development telemetry |
| **E5** | Configuration Architecture | Lean CLAUDE.md, unified settings, ownership manifest |
| **E6** | Team Orchestration | Composition manifests and team coordination |

---

## File Structure

```
.claude/
├── CLAUDE.md                          # LEAN: 7 rules + pointers (~25 lines)
├── ownership.json                     # Single source of truth for file ownership
├── settings.json                      # Unified: permissions + hooks + env
│
├── hooks/                             # E1: Deterministic enforcement
│   ├── cargo_mutex.py                 # PreToolUse/Bash: sequential cargo
│   ├── file_ownership.py              # PreToolUse/Edit|Write: agent boundary
│   ├── protected_paths.py             # PreToolUse/Edit|Write: plan-mode dirs
│   ├── binary_blocker.py              # PreToolUse/Bash: block cargo run
│   ├── post_edit_lint.sh              # PostToolUse/Edit|Write: auto-clippy
│   ├── session_start.py               # SessionStart: log context
│   └── session_end.py                 # Stop: audit trail
│
├── agents/                            # E2: Complete skill preloading
│   ├── lead.md                        # Explicit lead agent
│   ├── signals.md                     # 6 skills preloaded
│   ├── strategy.md                    # 5 skills preloaded
│   ├── infra.md                       # 3 skills preloaded
│   ├── analytics.md                   # 4 skills preloaded
│   ├── risk.md                        # 2 skills preloaded
│   └── code-reviewer.md              # Read-only
│
├── skills/
│   ├── foundation/                    # Always loaded first
│   │   └── measurement-infrastructure/
│   ├── domains/                       # Loaded with owning agent
│   │   ├── infrastructure-ops/
│   │   ├── risk-management/
│   │   ├── stochastic-controller/
│   │   ├── calibration-analysis/
│   │   └── signal-audit/
│   ├── models/                        # On-demand, referenced by domain skills
│   │   ├── fill-intensity-hawkes/
│   │   ├── adverse-selection-classifier/
│   │   ├── regime-detection-hmm/
│   │   ├── lead-lag-estimator/
│   │   └── checkpoint-management/
│   ├── integration/
│   │   └── quote-engine/
│   └── workflows/                     # E3: Gated pipelines
│       ├── debug-pnl/
│       │   ├── SKILL.md
│       │   └── verify/
│       │       ├── phase1_identify_drag.py
│       │       └── phase2_check_calibration.py
│       ├── go-live/
│       │   ├── SKILL.md
│       │   └── verify/
│       │       ├── phase1_paper_thresholds.sh
│       │       ├── phase2_parity_audit.py
│       │       ├── phase3_config_validation.py
│       │       └── phase4_monitoring_check.sh
│       ├── add-signal/
│       │   ├── SKILL.md
│       │   └── verify/
│       │       └── signal_checklist.py
│       ├── paper-trading/
│       │   ├── SKILL.md
│       │   └── verify/
│       │       └── feedback_loops.sh
│       ├── add-asset/
│       │   ├── SKILL.md
│       │   └── verify/
│       │       └── asset_prerequisites.py
│       └── compositions/              # E6: Team manifests
│           ├── full-feature/SKILL.md
│           ├── model-improvement/SKILL.md
│           ├── bug-investigation/SKILL.md
│           └── code-review/SKILL.md
│
├── session-logs/                      # E4: Automatic audit trail
│   └── {timestamp}.jsonl
│
├── rules/
│   └── agent-teams.md                 # Coordination docs (enforcement via hooks)
│
├── agent-memory/                      # Persistent per-agent knowledge
│   ├── signals/
│   ├── strategy/
│   ├── infra/
│   ├── analytics/
│   ├── risk/
│   └── code-reviewer/
│
└── plans/                             # Architecture plans (existing)
```

---

## E1: Enforcement Layer (Hooks)

The enforcement layer converts every "never/always" rule into a deterministic gate. Hooks run as external processes — they cannot be persuaded, confused, or skipped by the model.

### Hook Registration (settings.json)

```jsonc
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 .claude/hooks/cargo_mutex.py"
          },
          {
            "type": "command",
            "command": "python3 .claude/hooks/binary_blocker.py"
          }
        ]
      },
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python3 .claude/hooks/file_ownership.py"
          },
          {
            "type": "command",
            "command": "python3 .claude/hooks/protected_paths.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/post_edit_lint.sh"
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 .claude/hooks/session_start.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 .claude/hooks/session_end.py"
          }
        ]
      }
    ]
  }
}
```

### Hook 1: Cargo Mutex (`cargo_mutex.py`)

**Problem**: Multiple agents (or the same agent in rapid succession) running `cargo test`, `cargo clippy`, and `cargo build` simultaneously crashes the machine.

**Mechanism**: File-based mutex. The hook checks for a lockfile before any cargo command. If locked, it returns exit code 2 (block with feedback). A PostToolUse hook on Bash releases the lock.

```python
#!/usr/bin/env python3
"""
PreToolUse hook for Bash: enforce sequential cargo commands.
Exit 0 = allow, Exit 2 = block with feedback message on stderr.
"""
import json
import sys
import os
import time

LOCK_FILE = os.path.expanduser("~/.claude/cargo.lock")
CARGO_COMMANDS = ["cargo test", "cargo clippy", "cargo build", "cargo check", "cargo run"]

def main():
    data = json.load(sys.stdin)
    command = data.get("tool_input", {}).get("command", "")

    # Only gate cargo commands
    if not any(command.strip().startswith(c) for c in CARGO_COMMANDS):
        sys.exit(0)

    # Check lock
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                lock_info = json.load(f)
            age_s = time.time() - lock_info.get("timestamp", 0)
            # Stale lock detection: 10 min timeout
            if age_s > 600:
                os.remove(LOCK_FILE)
            else:
                owner = lock_info.get("command", "unknown")
                print(
                    f"BLOCKED: Another cargo command is running: '{owner}' "
                    f"(started {int(age_s)}s ago). Wait for it to finish. "
                    f"Only ONE cargo command at a time — concurrent builds crash the machine.",
                    file=sys.stderr,
                )
                sys.exit(2)
        except (json.JSONDecodeError, OSError):
            os.remove(LOCK_FILE)

    # Acquire lock
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        json.dump({
            "command": command[:80],
            "timestamp": time.time(),
            "pid": os.getpid(),
        }, f)

    sys.exit(0)

if __name__ == "__main__":
    main()
```

**Companion PostToolUse hook** (add to PostToolUse/Bash):

```python
#!/usr/bin/env python3
"""PostToolUse hook for Bash: release cargo mutex after command completes."""
import json, sys, os

LOCK_FILE = os.path.expanduser("~/.claude/cargo.lock")
CARGO_COMMANDS = ["cargo test", "cargo clippy", "cargo build", "cargo check", "cargo run"]

data = json.load(sys.stdin)
command = data.get("tool_input", {}).get("command", "")

if any(command.strip().startswith(c) for c in CARGO_COMMANDS):
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass

sys.exit(0)
```

### Hook 2: File Ownership (`file_ownership.py`)

**Problem**: Agent teams have exclusive file ownership but it's enforced by prose. The signals agent could edit `strategy/signal_integration.rs` if the model misreads its instructions.

**Mechanism**: Reads `ownership.json` and the current agent name from environment. If the agent doesn't own the target file, blocks with feedback explaining who does.

```python
#!/usr/bin/env python3
"""
PreToolUse hook for Edit|Write: enforce file ownership boundaries.
Reads ownership.json and CLAUDE_AGENT_NAME env var.
Exit 0 = allow, Exit 2 = block with feedback.
"""
import json
import sys
import os
import fnmatch

OWNERSHIP_FILE = os.path.join(
    os.environ.get("CLAUDE_PROJECT_DIR", "."), ".claude", "ownership.json"
)

def main():
    data = json.load(sys.stdin)
    file_path = data.get("tool_input", {}).get("file_path", "")
    agent_name = os.environ.get("CLAUDE_AGENT_NAME", "")

    # If not in an agent context (main session or lead), allow everything
    if not agent_name or agent_name == "lead":
        sys.exit(0)

    # Load ownership manifest
    if not os.path.exists(OWNERSHIP_FILE):
        sys.exit(0)  # No manifest = no enforcement

    with open(OWNERSHIP_FILE) as f:
        ownership = json.load(f)

    # Normalize path relative to src/market_maker/
    rel_path = file_path
    for prefix in ["src/market_maker/", "./src/market_maker/"]:
        if rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix):]
            break

    # Check ownership
    agents = ownership.get("agents", {})
    agent_config = agents.get(agent_name, {})
    owned_patterns = agent_config.get("owns", [])
    exclusive_files = agent_config.get("exclusive", [])

    # Check if file matches any owned pattern
    is_owned = any(
        fnmatch.fnmatch(rel_path, pattern) or rel_path.startswith(pattern.rstrip("*"))
        for pattern in owned_patterns
    )

    if is_owned:
        sys.exit(0)

    # Find who owns this file
    owner = "lead (default)"
    for other_agent, config in agents.items():
        other_patterns = config.get("owns", [])
        if any(
            fnmatch.fnmatch(rel_path, p) or rel_path.startswith(p.rstrip("*"))
            for p in other_patterns
        ):
            owner = other_agent
            break

    # Check if this is an exclusively owned file
    for other_agent, config in agents.items():
        if rel_path in config.get("exclusive", []):
            owner = f"{other_agent} (EXCLUSIVE)"
            break

    print(
        f"BLOCKED: Agent '{agent_name}' cannot edit '{file_path}'. "
        f"This file is owned by '{owner}'. "
        f"Propose changes via team message instead of direct edit.",
        file=sys.stderr,
    )
    sys.exit(2)

if __name__ == "__main__":
    main()
```

### Hook 3: Protected Paths (`protected_paths.py`)

**Problem**: `orchestrator/`, `risk/`, `safety/`, `src/bin/` require plan approval but this is only enforced by agent-level `permissionMode: plan`, not at the file level.

```python
#!/usr/bin/env python3
"""
PreToolUse hook for Edit|Write: require plan-mode for protected directories.
Blocks edits to critical paths unless agent has plan approval.
"""
import json
import sys
import os

PROTECTED_PREFIXES = [
    "src/market_maker/orchestrator/",
    "src/market_maker/risk/",
    "src/market_maker/safety/",
    "src/bin/",
    "src/exchange/",
]

# Agents with plan-mode access to specific paths
PLAN_APPROVED = {
    "infra": ["src/market_maker/orchestrator/"],
    "risk": ["src/market_maker/risk/", "src/market_maker/safety/"],
}

def main():
    data = json.load(sys.stdin)
    file_path = data.get("tool_input", {}).get("file_path", "")
    agent_name = os.environ.get("CLAUDE_AGENT_NAME", "")

    # Lead agent can edit anything
    if not agent_name or agent_name == "lead":
        sys.exit(0)

    # Check if path is protected
    for prefix in PROTECTED_PREFIXES:
        if file_path.startswith(prefix) or file_path.startswith(f"./{prefix}"):
            # Check if this agent has plan approval for this path
            approved_paths = PLAN_APPROVED.get(agent_name, [])
            if any(file_path.startswith(p) or file_path.startswith(f"./{p}") for p in approved_paths):
                sys.exit(0)  # Agent has plan approval

            print(
                f"BLOCKED: '{file_path}' is in a protected directory. "
                f"Agent '{agent_name}' does not have plan approval for this path. "
                f"Only the lead or specifically approved agents can edit files here. "
                f"Propose your changes via team message.",
                file=sys.stderr,
            )
            sys.exit(2)

    sys.exit(0)

if __name__ == "__main__":
    main()
```

### Hook 4: Binary Blocker (`binary_blocker.py`)

**Problem**: Core Rule #5 says "Do not run binaries or scripts unless the user explicitly asks." This is a prose suggestion.

```python
#!/usr/bin/env python3
"""
PreToolUse hook for Bash: block binary execution unless explicitly allowed.
Blocks: cargo run --bin market_maker, cargo run --bin paper_trader, ./scripts/*, ./target/*
Allows: cargo run --bin calibration_report, cargo run --bin health_dashboard (safe binaries)
"""
import json
import sys
import re

BLOCKED_PATTERNS = [
    r"cargo\s+run\s+.*--bin\s+market_maker",
    r"cargo\s+run\s+.*--bin\s+paper_trader",
    r"cargo\s+run\s+.*--bin\s+parameter_estimator",
    r"\./scripts/",
    r"\./target/",
]

SAFE_PATTERNS = [
    r"cargo\s+run\s+.*--bin\s+calibration_report",
    r"cargo\s+run\s+.*--bin\s+health_dashboard",
]

def main():
    data = json.load(sys.stdin)
    command = data.get("tool_input", {}).get("command", "")

    # Check safe patterns first
    for pattern in SAFE_PATTERNS:
        if re.search(pattern, command):
            sys.exit(0)

    # Check blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            print(
                f"BLOCKED: Binary execution requires explicit user approval. "
                f"Command: '{command[:80]}'. "
                f"Provide the command as a copy-pasteable block for the user to run manually.",
                file=sys.stderr,
            )
            sys.exit(2)

    sys.exit(0)

if __name__ == "__main__":
    main()
```

### Hook 5: Post-Edit Lint (`post_edit_lint.sh`)

**Problem**: "Run clippy after every change" is a suggestion. Engineers forget, agents forget, and errors compound.

```bash
#!/usr/bin/env bash
# PostToolUse hook for Edit|Write: run clippy on edited .rs files.
# Non-blocking (exit 0 always) — adds feedback as stderr message.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))")

# Only lint Rust files
if [[ "$FILE_PATH" != *.rs ]]; then
    exit 0
fi

# Check if cargo mutex is free (don't block on lint if another cargo is running)
if [ -f "$HOME/.claude/cargo.lock" ]; then
    echo "Skipping auto-lint: another cargo command is running. Run clippy manually when free." >&2
    exit 0
fi

# Quick clippy check (just the file's crate, not full workspace)
# This runs in background-ish mode — if it takes too long, the agent continues
timeout 60 cargo clippy -- -D warnings 2>&1 | tail -5 >&2

exit 0
```

### Hook 6 & 7: Session Audit (`session_start.py` / `session_end.py`)

```python
#!/usr/bin/env python3
"""SessionStart hook: log session context."""
import json, sys, os, datetime

LOG_DIR = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), ".claude", "session-logs")
os.makedirs(LOG_DIR, exist_ok=True)

session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(LOG_DIR, f"{session_id}.jsonl")

# Store session ID for the end hook
with open(os.path.expanduser("~/.claude/current_session.txt"), "w") as f:
    f.write(log_file)

entry = {
    "event": "session_start",
    "timestamp": datetime.datetime.now().isoformat(),
    "agent": os.environ.get("CLAUDE_AGENT_NAME", "main"),
    "cwd": os.getcwd(),
}

with open(log_file, "a") as f:
    f.write(json.dumps(entry) + "\n")

sys.exit(0)
```

```python
#!/usr/bin/env python3
"""Stop hook: write session summary."""
import json, sys, os, datetime, glob

SESSION_FILE = os.path.expanduser("~/.claude/current_session.txt")

if not os.path.exists(SESSION_FILE):
    sys.exit(0)

with open(SESSION_FILE) as f:
    log_file = f.read().strip()

# Summarize what happened (git diff --stat gives us file changes)
import subprocess
try:
    diff_stat = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        capture_output=True, text=True, timeout=10
    ).stdout.strip()
except Exception:
    diff_stat = "unavailable"

entry = {
    "event": "session_end",
    "timestamp": datetime.datetime.now().isoformat(),
    "git_diff_stat": diff_stat,
}

try:
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
except Exception:
    pass

sys.exit(0)
```

---

## E2: Agent Architecture

### Design Principle

Every agent preloads **all** skills it could need based on the current routing table. The routing table in CLAUDE.md is deleted. Domain knowledge loads automatically with the agent — the model makes domain decisions, not "which skill should I read" decisions.

### ownership.json — Single Source of Truth

```json
{
  "version": 1,
  "description": "File ownership manifest. Referenced by hooks and agents.",
  "base_path": "src/market_maker/",
  "agents": {
    "signals": {
      "owns": [
        "estimator/*",
        "adverse_selection/*",
        "calibration/*",
        "edge/*"
      ],
      "exclusive": [],
      "cannot_edit": [
        "strategy/signal_integration.rs"
      ]
    },
    "strategy": {
      "owns": [
        "strategy/*",
        "quoting/*",
        "process_models/*",
        "stochastic/*",
        "control/*"
      ],
      "exclusive": [
        "strategy/signal_integration.rs"
      ],
      "cannot_edit": []
    },
    "infra": {
      "owns": [
        "orchestrator/*",
        "infra/*",
        "messages/*",
        "core/*",
        "fills/*",
        "execution/*",
        "events/*"
      ],
      "exclusive": [],
      "cannot_edit": [],
      "plan_required": [
        "orchestrator/*"
      ]
    },
    "analytics": {
      "owns": [
        "analytics/*",
        "learning/*",
        "tracking/*",
        "simulation/*",
        "checkpoint/*",
        "adaptive/*"
      ],
      "exclusive": [],
      "cannot_edit": []
    },
    "risk": {
      "owns": [
        "risk/*",
        "safety/*",
        "monitoring/*"
      ],
      "exclusive": [],
      "cannot_edit": [],
      "plan_required": [
        "risk/*",
        "safety/*"
      ]
    },
    "lead": {
      "owns": [
        "mod.rs",
        "config/*",
        "belief/*",
        "multi/*",
        "latent/*"
      ],
      "exclusive": [
        "mod.rs"
      ],
      "also_owns": [
        "src/bin/*",
        "src/exchange/*",
        "src/ws/*",
        "src/info/*",
        "src/lib.rs",
        ".claude/*",
        "Cargo.toml"
      ]
    }
  },
  "global_rules": {
    "mod_rs_reexports": "lead-only — propose changes via message",
    "signal_integration": "strategy-only — other agents propose via message"
  }
}
```

### Redesigned Agent: signals.md

```markdown
---
name: signals
description: "Owns estimator/, adverse_selection/, calibration/, and edge/. Use for signals, parameter estimators, adverse selection, calibration metrics, or edge monitoring."
model: inherit
maxTurns: 25
skills:
  - measurement-infrastructure
  - signal-audit
  - calibration-analysis
  - fill-intensity-hawkes
  - adverse-selection-classifier
  - regime-detection-hmm
  - lead-lag-estimator
memory: project
---

# Signals Agent

You own the **Signals & Estimation** domain.

## Owned Directories (src/market_maker/)

- `estimator/` — kappa, sigma, regime, flow, mutual info
- `adverse_selection/` — pre-fill classifier, microstructure features
- `calibration/` — Brier score, IR, conditional metrics, model gating
- `edge/` — AB testing, signal health

## Boundaries

- **DO NOT edit** `strategy/signal_integration.rs` — propose changes to strategy agent
- **DO NOT edit** any `mod.rs` re-exports — propose to lead
- These boundaries are hook-enforced. Attempts will be blocked.

## Invariants

- `kappa > 0.0` in all formula paths
- Every new signal has Brier Score + IR metrics
- Every estimator has warmup counter + confidence metric
- `#[serde(default)]` on all checkpoint fields
- Units in variable names (`_bps`, `_s`, `_8h`)
- No hardcoded parameters — regime-dependent or configurable

## Review Checklist

- [ ] `cargo clippy -- -D warnings` passes (auto-run by hook)
- [ ] Units documented in variable names
- [ ] `kappa > 0.0` maintained in all formula paths
- [ ] Calibration metrics defined for any new model
- [ ] Checkpoint fields use `#[serde(default)]`
```

**Key change**: Skills list went from 3 → 7. The agent now preloads every skill it could need based on the old routing table:

| Agent | Old Skills | New Skills | Added |
|-------|-----------|------------|-------|
| signals | 3 (measurement-infrastructure, signal-audit, calibration-analysis) | 7 | fill-intensity-hawkes, adverse-selection-classifier, regime-detection-hmm, lead-lag-estimator |
| strategy | 2 (quote-engine, stochastic-controller) | 5 | regime-detection-hmm, risk-management, lead-lag-estimator |
| infra | 2 (infrastructure-ops, quote-engine) | 3 | checkpoint-management |
| analytics | 2 (calibration-analysis, daily-calibration-report) | 4 | measurement-infrastructure, signal-audit |
| risk | 1 (risk-management) | 2 | live-incident-response |

### Explicit Lead Agent: lead.md

Currently the lead is implicit — it's just "the main session." Making it explicit gives it a skill preload, review authority, and clear ownership.

```markdown
---
name: lead
description: "Coordinates agent teams. Owns mod.rs, config/, belief/, multi/, latent/, src/bin/, Cargo.toml, and .claude/. Synthesizes results from domain agents. Use as the default orchestration agent."
model: inherit
skills:
  - measurement-infrastructure
  - quote-engine
memory: project
---

# Lead Agent

You coordinate work across domain agents and own cross-cutting files.

## Exclusive Ownership

- `src/market_maker/mod.rs` — all re-export changes go through you
- `config/` — runtime configuration
- `src/bin/` — binary entry points
- `Cargo.toml` — dependency management
- `.claude/` — development infrastructure

## Coordination Protocol

1. Break work into agent-owned tasks based on ownership.json
2. Each agent works within its owned directories only
3. Cross-boundary changes: agent proposes via message, owner implements
4. Synthesize results after agents complete
5. You handle all mod.rs re-exports

## Rules

- Verify file ownership before assigning tasks
- Never assign the same file to two agents
- Run code-reviewer after significant changes
- Cargo commands: one at a time (hook-enforced)
```

---

## E3: Workflow Pipelines

### Design Principle

Every workflow phase ends with a **verification script** that outputs structured JSON. The next phase begins with "Parse the verification output. If any check failed, stop and report." The verification script is not optional — it's referenced by the workflow instructions as a required step.

### Example: go-live Workflow (Redesigned)

```markdown
---
name: go-live
description: Pre-flight checks and deployment for live trading.
disable-model-invocation: true
context: fork
agent: general-purpose
argument-hint: "[asset]"
allowed-tools: Read, Grep, Glob, Bash
---

# Go-Live Workflow

## Phase 1: Paper Validation

[... existing phase 1 instructions ...]

### Phase 1 Gate

Run the verification script. DO NOT proceed to Phase 2 unless all checks pass.

\```bash
python3 .claude/skills/workflows/go-live/verify/phase1_paper_thresholds.sh <ASSET>
\```

Parse the JSON output. If `"pass": false` for any check, report the failures
and stop. The user must resolve these before continuing.

---

## Phase 2: Code Parity Audit

[... existing phase 2 instructions ...]

### Phase 2 Gate

\```bash
python3 .claude/skills/workflows/go-live/verify/phase2_parity_audit.py
\```

Same rule: parse JSON output, stop on any failure.

---

[... phases 3-7 follow the same pattern ...]
```

### Verification Script Example: `phase1_paper_thresholds.sh`

```bash
#!/usr/bin/env bash
# Verify paper trading thresholds before go-live.
# Output: JSON with pass/fail per metric.
# Usage: ./phase1_paper_thresholds.sh <ASSET>

ASSET="${1:?Usage: phase1_paper_thresholds.sh <ASSET>}"
CHECKPOINT="data/checkpoints/paper/${ASSET}/prior.json"

if [ ! -f "$CHECKPOINT" ]; then
    echo '{"overall_pass": false, "error": "No paper checkpoint found", "checks": []}'
    exit 0
fi

# Parse checkpoint metrics
python3 -c "
import json, sys

with open('${CHECKPOINT}') as f:
    data = json.load(f)

checks = []

# Edge check
edge = data.get('edge_bps', -999)
checks.append({
    'name': 'edge_positive',
    'pass': edge > 0,
    'value': edge,
    'threshold': '> 0 bps',
    'severity': 'critical'
})

# Fill rate check
fills = data.get('total_fills', 0)
hours = data.get('runtime_hours', 0)
fill_rate = fills / max(hours, 0.01)
checks.append({
    'name': 'fill_rate',
    'pass': fill_rate > 10,
    'value': round(fill_rate, 1),
    'threshold': '> 10 fills/hour',
    'severity': 'critical'
})

# AS rate check
as_rate = data.get('adverse_selection_rate', 1.0)
checks.append({
    'name': 'adverse_selection_rate',
    'pass': as_rate < 0.40,
    'value': round(as_rate, 3),
    'threshold': '< 0.40',
    'severity': 'critical'
})

# Kill switch check
kills = data.get('kill_switch_triggers', 999)
checks.append({
    'name': 'no_kill_switches',
    'pass': kills == 0,
    'value': kills,
    'threshold': '== 0',
    'severity': 'critical'
})

overall = all(c['pass'] for c in checks if c['severity'] == 'critical')
print(json.dumps({'overall_pass': overall, 'checks': checks}, indent=2))
"
```

### Verification Script: `phase2_parity_audit.py`

```python
#!/usr/bin/env python3
"""
Verify paper/live code parity by checking all 9 learning loops exist
in both paper and live code paths.
"""
import json
import subprocess
import sys

LEARNING_LOOPS = [
    {
        "name": "kappa_from_own_fills",
        "pattern": "estimator.on_own_fill",
        "file": "src/market_maker/orchestrator/handlers.rs",
    },
    {
        "name": "as_markout_queue",
        "pattern": "pending_fill_outcomes.push_back",
        "file": "src/market_maker/orchestrator/handlers.rs",
    },
    {
        "name": "as_outcome_feedback",
        "pattern": "record_outcome",
        "file": "src/market_maker/orchestrator/handlers.rs",
    },
    {
        "name": "calibration_progress",
        "pattern": "calibration_controller",
        "file": "src/market_maker/orchestrator/handlers.rs",
    },
    {
        "name": "sigma_update",
        "pattern": "update_sigma",
        "file": "src/market_maker/orchestrator/handlers.rs",
    },
    {
        "name": "regime_update",
        "pattern": "update_regime",
        "file": "src/market_maker/orchestrator/handlers.rs",
    },
    {
        "name": "quote_outcome_tracking",
        "pattern": "QuoteOutcomeTracker",
        "file": "src/market_maker/learning/quote_outcome.rs",
    },
    {
        "name": "rl_baseline",
        "pattern": "BaselineTracker",
        "file": "src/market_maker/learning/baseline_tracker.rs",
    },
    {
        "name": "live_analytics_flush",
        "pattern": "live_analytics",
        "file": "src/market_maker/analytics/live.rs",
    },
]

checks = []
for loop in LEARNING_LOOPS:
    result = subprocess.run(
        ["grep", "-c", loop["pattern"], loop["file"]],
        capture_output=True, text=True,
    )
    count = int(result.stdout.strip()) if result.returncode == 0 else 0
    checks.append({
        "name": loop["name"],
        "pass": count > 0,
        "occurrences": count,
        "file": loop["file"],
        "pattern": loop["pattern"],
    })

overall = all(c["pass"] for c in checks)
print(json.dumps({"overall_pass": overall, "checks": checks}, indent=2))
```

---

## E4: Observability

### Session Log Format

Each session produces a `.jsonl` file in `.claude/session-logs/`:

```jsonc
// Line 1: session start
{"event": "session_start", "timestamp": "2026-02-21T10:00:00", "agent": "main"}

// Lines 2-N: significant events (logged by hooks)
{"event": "cargo_command", "command": "cargo clippy", "exit_code": 0, "duration_s": 12}
{"event": "file_edit", "file": "src/market_maker/estimator/kappa.rs", "agent": "signals"}
{"event": "ownership_block", "file": "strategy/signal_integration.rs", "agent": "signals", "owner": "strategy"}

// Last line: session end
{"event": "session_end", "timestamp": "2026-02-21T10:45:00", "git_diff_stat": "3 files changed, 47 insertions(+), 12 deletions(-)"}
```

### Weekly Digest Skill

A skill (not a hook) that reads session logs and produces a development summary:

```markdown
---
name: weekly-digest
description: Summarize development activity from session logs. Use when asked about recent development progress or weekly review.
disable-model-invocation: true
---

# Weekly Development Digest

Read `.claude/session-logs/*.jsonl` for the last 7 days.

Produce a summary covering:
1. **Files changed** — grouped by domain (signals, strategy, infra, etc.)
2. **Ownership blocks** — how many times agents tried to cross boundaries
3. **Cargo failures** — clippy warnings, test failures, build errors
4. **Workflow runs** — which workflows were invoked and their gate results
5. **Session count and duration** — development velocity
```

---

## E5: Configuration Architecture

### Lean CLAUDE.md (~25 lines)

```markdown
# CLAUDE.md — Hyperliquid Market Making System

Rust market making for Hyperliquid perpetual futures. See `README.md` for overview, `PROJECT_INDEX.md` for directory map.

## Core Rules

1. **Measurement before modeling** — define prediction target + logging + baseline before building any model
2. **Calibration is ground truth** — Brier Score, IR, Conditional Calibration for every model
3. **Everything is regime-dependent** — kappa varies 10x, gamma varies 5x between quiet and cascade
4. **Defense first** — when uncertain, widen spreads. Missing a trade is cheap; cascades are not
5. **Manual execution only** — never run trading binaries unless user explicitly asks (hook-enforced)
6. **Prefer explicit over clever** — bugs cost real money
7. **`#[serde(default)]`** on all checkpoint fields for backward compatibility

## Build

```
cargo clippy -- -D warnings    # Lint first (auto-runs after edits via hook)
cargo test                      # After clippy passes
```

**Resource constraint**: One cargo command at a time (hook-enforced mutex). Sequential only.

## Structure

- Domain knowledge: `.claude/skills/` (loaded automatically with agents)
- File ownership: `.claude/ownership.json` (hook-enforced)
- Agents: `.claude/agents/` (signals, strategy, infra, analytics, risk, code-reviewer)
- Workflows: `/debug-pnl`, `/go-live`, `/paper-trading`, `/add-signal`, `/add-asset`
```

**What was removed:**
- Project layout → already in `PROJECT_INDEX.md`
- Skill routing table → replaced by agent skill preloads
- Detailed build commands → most engineers know `cargo test`
- Resource constraint details → hook handles enforcement, one-liner reminder is enough

### Unified settings.json

Merge the current `settings.json` and `settings.local.json` overlap. The local file should only contain truly local overrides (MCP servers, one-off permissions), not a duplicate permission set.

---

## E6: Team Orchestration

### Composition Manifests

Each composition is a workflow skill that creates a team with structured parameters.

#### `compositions/model-improvement/SKILL.md`

```markdown
---
name: compose-model-improvement
description: Create a model-improvement agent team (signals + strategy + analytics). Use for estimation work, calibration improvements, or signal development.
disable-model-invocation: true
context: fork
argument-hint: "[description of the improvement]"
---

# Model Improvement Composition

Create an agent team with 3 teammates for model/estimation work.

## Team Structure

| Role | Agent | Focus |
|------|-------|-------|
| Signal Development | signals | Feature extraction, estimators, calibration |
| Strategy Integration | strategy | Signal integration, spread computation, control |
| Measurement & Validation | analytics | Prediction logging, calibration metrics, fill simulation |

## Task Decomposition Template

Given the user's description ($ARGUMENTS), decompose into:

1. **signals tasks**: What estimators/features need to change?
2. **strategy tasks**: How do signal changes affect quoting?
3. **analytics tasks**: What new measurements are needed?

## Coordination Rules

- File ownership is hook-enforced — agents cannot cross boundaries
- `signal_integration.rs` changes: signals agent proposes, strategy agent implements
- Cargo commands: one at a time (hook-enforced mutex)
- Each agent runs clippy before marking tasks complete (auto-run by hook)

## Synthesis

After all agents complete:
1. Lead runs code-reviewer on all changed files
2. Lead runs `cargo test` (one command, sequential)
3. Lead synthesizes findings into a summary
```

#### `compositions/bug-investigation/SKILL.md`

```markdown
---
name: compose-bug-investigation
description: Create a bug investigation team with parallel hypothesis testing. Use for debugging PnL issues, unexpected behavior, or production incidents.
disable-model-invocation: true
context: fork
argument-hint: "[bug description]"
---

# Bug Investigation Composition

Spawn N hypothesis-testing agents working in parallel.

## Team Structure

Given the bug description ($ARGUMENTS):

1. **Formulate 3 independent hypotheses** about the root cause
2. **Assign each hypothesis to a separate agent** (use explore or general-purpose)
3. Each agent investigates independently with read-only tools
4. Lead synthesizes findings and identifies the root cause

## Hypothesis Template

For each hypothesis:
- **Theory**: What could be causing this?
- **Evidence to look for**: What files/logs/metrics would confirm or deny?
- **Tools needed**: Read, Grep, Glob (read-only investigation)

## Rules

- Investigation phase is READ-ONLY — no edits until root cause is confirmed
- After root cause identified, assign fix to the appropriate domain agent
- Run /debug-pnl workflow if this is a PnL issue (has its own diagnostic chain)
```

#### `compositions/full-feature/SKILL.md`

```markdown
---
name: compose-full-feature
description: Create a full feature team (all 5 domain agents). Use for large cross-cutting features that touch signals, strategy, infrastructure, analytics, and risk.
disable-model-invocation: true
context: fork
argument-hint: "[feature description]"
---

# Full Feature Composition

All 5 domain agents working in parallel on a cross-cutting feature.

## Team Structure

| Agent | Mandate |
|-------|---------|
| signals | Feature extraction, new estimators |
| strategy | Quoting logic, signal integration |
| infra | Event loop wiring, message handling |
| analytics | Measurement, calibration, tracking |
| risk | Safety constraints, risk limits |

## Phased Execution

### Phase 1: Design (all agents, plan mode)
Each agent reviews the feature description and produces a plan for their domain.
Lead synthesizes into a unified plan with dependency ordering.

### Phase 2: Implementation (parallel, within owned files)
Agents implement their portions. File ownership is hook-enforced.
Cross-boundary interfaces are agreed in Phase 1.

### Phase 3: Integration (lead)
Lead handles mod.rs re-exports and cross-domain wiring.
Lead runs code-reviewer on all changes.

### Phase 4: Verification
Sequential: clippy → test → paper trading validation.
One cargo command at a time.
```

---

## How the Layers Interact

```
User Request
    │
    ▼
┌─────────────────────────────────────────────┐
│  CLAUDE.md (7 rules, pointers)              │  ← Context loaded every session
│  + Agent preloaded skills (automatic)       │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Model Decision Layer                       │
│  "Which agent handles this?"                │
│  "Which workflow applies?"                  │
│  "What's the fix for this bug?"             │  ← MODEL makes domain decisions
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Hook Enforcement Layer                     │
│  ✗ cargo_mutex.py      → sequential builds  │
│  ✗ file_ownership.py   → agent boundaries   │
│  ✗ protected_paths.py  → plan-mode dirs     │  ← HOOKS make compliance decisions
│  ✗ binary_blocker.py   → no live execution  │
│  ✓ post_edit_lint.sh   → auto-clippy        │
│  ✓ session_logger.py   → audit trail        │
└─────────────┬───────────────────────────────┘
              │
              ▼
         Tool Execution
```

The model decides WHAT to do. Hooks decide WHETHER it's allowed. This separation means:

- A confused agent that tries to edit the wrong file gets blocked and told who owns it
- Two agents that try to run cargo simultaneously: second one waits
- A workflow that tries to skip Phase 2: verification script fails, agent can't proceed
- Every session: audit trail written regardless of what the model did

---

## Migration Path

This doesn't require a big-bang rewrite. The layers are additive:

**Week 1** (E1 + E5): Deploy hooks + lean CLAUDE.md
- Add cargo_mutex.py, binary_blocker.py, post_edit_lint.sh
- Trim CLAUDE.md, create ownership.json
- Immediate value: three rules go from prose to deterministic

**Week 2** (E2): Upgrade agent skill preloads
- Update each agent's skills list
- Remove routing table from CLAUDE.md
- Immediate value: agents always have the right knowledge

**Week 3** (E1 continued): Deploy file_ownership.py + protected_paths.py
- These depend on ownership.json from week 1
- Immediate value: agent boundaries are enforced

**Week 4** (E3 + E4): Workflow verification + session audit
- Add verify/ directories to existing workflows
- Deploy session_start.py and session_end.py
- Immediate value: workflows can't skip phases, sessions are logged

**Week 5** (E6): Composition manifests
- Create composition skills from existing prose descriptions
- Test with real feature work
- Immediate value: reproducible team setups

---

## What This Costs

**Context budget**: Hooks add zero context (they're external processes). ownership.json adds zero context (only read by hooks). The lean CLAUDE.md saves ~130 lines of context per session. Agent skill preloads add context, but it's the RIGHT context — domain knowledge the agent will actually use, loaded once instead of discovered mid-task.

**Maintenance**: 7 hook scripts (~200 lines total Python/bash). ownership.json needs updating when directory structure changes. Verification scripts need updating when workflow thresholds change. All of this is version-controlled and shared via git.

**Risk**: Hook bugs could block legitimate work. Mitigation: all hooks are fail-open by default (exit 0 on error), with explicit exit 2 only for confirmed violations. The cargo mutex has a 10-minute stale lock timeout. File ownership only enforces when CLAUDE_AGENT_NAME is set (solo sessions are unrestricted).
