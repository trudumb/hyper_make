#!/usr/bin/env python3
"""
SessionStart hook: log session context and capture agent name.
Writes agent name to ~/.claude/sessions/{session_id}.agent for use
by ownership hooks (CLAUDE_AGENT_NAME is not set by Claude Code).
Always exits 0 (fail-open).
"""
import json
import sys
import os
import datetime

LOG_DIR = os.path.join(os.getcwd(), ".claude", "session-logs")
SESSIONS_DIR = os.path.expanduser("~/.claude/sessions")

try:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    data = json.load(sys.stdin)
    session_id = data.get("session_id", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    agent_name = data.get("agent_name", "") or data.get("agent", {}).get("name", "")

    # Write agent name for ownership hooks
    if agent_name:
        agent_file = os.path.join(SESSIONS_DIR, f"{session_id}.agent")
        with open(agent_file, "w") as f:
            f.write(agent_name)

    # Write session log entry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(LOG_DIR, f"{timestamp}.jsonl")

    # Store log file path for session_end hook
    with open(os.path.expanduser("~/.claude/current_session.txt"), "w") as f:
        f.write(log_file)

    entry = {
        "event": "session_start",
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "agent": agent_name or "main",
        "cwd": os.getcwd(),
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

except Exception:
    pass  # Fail-open

sys.exit(0)
