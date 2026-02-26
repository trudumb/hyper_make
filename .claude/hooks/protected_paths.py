#!/usr/bin/env python3
"""
PreToolUse hook for Edit|Write: require plan-mode for protected directories.
Blocks edits to critical paths unless agent has plan approval.
Exit 0 = allow, Exit 2 = block with feedback.
All exceptions fail-open (exit 0).
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

PLAN_APPROVED = {
    "infra": ["src/market_maker/orchestrator/"],
    "risk": ["src/market_maker/risk/", "src/market_maker/safety/"],
}

SESSIONS_DIR = os.path.expanduser("~/.claude/sessions")


def get_agent_name(data):
    """Get agent name from session file."""
    session_id = data.get("session_id", "")
    if session_id:
        agent_file = os.path.join(SESSIONS_DIR, f"{session_id}.agent")
        try:
            with open(agent_file) as f:
                return f.read().strip()
        except (OSError, FileNotFoundError):
            pass
    return ""


def main():
    try:
        data = json.load(sys.stdin)
        file_path = data.get("tool_input", {}).get("file_path", "")
        agent_name = get_agent_name(data)

        if not agent_name or agent_name == "lead":
            sys.exit(0)

        norm_path = file_path
        if norm_path.startswith("./"):
            norm_path = norm_path[2:]

        for prefix in PROTECTED_PREFIXES:
            if norm_path.startswith(prefix):
                approved_paths = PLAN_APPROVED.get(agent_name, [])
                if any(norm_path.startswith(p) for p in approved_paths):
                    sys.exit(0)

                print(
                    f"BLOCKED: '{file_path}' is in a protected directory ({prefix}). "
                    f"Agent '{agent_name}' does not have plan approval for this path. "
                    f"Only the lead or specifically approved agents can edit files here. "
                    f"Propose your changes via team message.",
                    file=sys.stderr,
                )
                sys.exit(2)

        sys.exit(0)

    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
