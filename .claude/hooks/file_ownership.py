#!/usr/bin/env python3
"""
PreToolUse hook for Edit|Write: enforce agent file boundaries.
Reads ownership.json and agent name from session file.
Exit 0 = allow, Exit 2 = block with feedback.
All exceptions fail-open (exit 0).

Agent name detection fallback chain:
1. ~/.claude/sessions/{session_id}.agent (written by session_start.py)
2. No agent name found -> allow all (solo session, fail-open)
"""
import json
import sys
import os
import fnmatch

OWNERSHIP_FILE = os.path.join(os.getcwd(), ".claude", "ownership.json")
SESSIONS_DIR = os.path.expanduser("~/.claude/sessions")


def get_agent_name(data):
    """Get agent name from session file written by session_start.py."""
    session_id = data.get("session_id", "")
    if session_id:
        agent_file = os.path.join(SESSIONS_DIR, f"{session_id}.agent")
        try:
            with open(agent_file) as f:
                return f.read().strip()
        except (OSError, FileNotFoundError):
            pass
    return ""


def path_matches(rel_path, pattern):
    """Check if a relative path matches an ownership pattern."""
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    prefix = pattern.rstrip("*").rstrip("/")
    if prefix and (rel_path.startswith(prefix + "/") or rel_path == prefix):
        return True
    return False


def main():
    try:
        data = json.load(sys.stdin)
        file_path = data.get("tool_input", {}).get("file_path", "")
        agent_name = get_agent_name(data)

        # No agent context (solo session or lead) = allow everything
        if not agent_name or agent_name == "lead":
            sys.exit(0)

        # Load ownership manifest
        if not os.path.exists(OWNERSHIP_FILE):
            sys.exit(0)

        with open(OWNERSHIP_FILE) as f:
            ownership = json.load(f)

        base_path = ownership.get("base_path", "src/market_maker/")

        # Normalize file_path to be relative to base_path
        rel_path = file_path
        for prefix in [base_path, "./" + base_path, os.path.join(os.getcwd(), base_path)]:
            if rel_path.startswith(prefix):
                rel_path = rel_path[len(prefix):]
                break
        else:
            # File is not under base_path -- allow (non-market_maker files)
            sys.exit(0)

        agents = ownership.get("agents", {})

        # Check cannot_edit list
        agent_config = agents.get(agent_name, {})
        cannot_edit = agent_config.get("cannot_edit", [])
        for blocked in cannot_edit:
            if rel_path == blocked or fnmatch.fnmatch(rel_path, blocked):
                exclusive_owner = "another agent"
                for other_name, other_cfg in agents.items():
                    if rel_path in other_cfg.get("exclusive", []):
                        exclusive_owner = f"{other_name} (EXCLUSIVE)"
                        break
                print(
                    f"BLOCKED: Agent '{agent_name}' cannot edit '{file_path}'. "
                    f"This file is owned by {exclusive_owner}. "
                    f"Propose changes via team message instead.",
                    file=sys.stderr,
                )
                sys.exit(2)

        # Check if this agent owns the file
        owned_patterns = agent_config.get("owns", [])
        if any(path_matches(rel_path, p) for p in owned_patterns):
            sys.exit(0)

        # Find who owns it
        owner = "lead (default)"
        for other_name, other_cfg in agents.items():
            if other_name == agent_name:
                continue
            if any(path_matches(rel_path, p) for p in other_cfg.get("owns", [])):
                owner = other_name
                break
            if rel_path in other_cfg.get("exclusive", []):
                owner = f"{other_name} (EXCLUSIVE)"
                break

        print(
            f"BLOCKED: Agent '{agent_name}' cannot edit '{file_path}'. "
            f"This file is owned by '{owner}'. "
            f"Propose changes via team message instead of direct edit.",
            file=sys.stderr,
        )
        sys.exit(2)

    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
