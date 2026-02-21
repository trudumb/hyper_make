#!/usr/bin/env python3
"""
Stop hook: write session summary with git diff stat.
Always exits 0 (fail-open).
"""
import json
import sys
import os
import datetime
import subprocess

SESSION_FILE = os.path.expanduser("~/.claude/current_session.txt")

try:
    if not os.path.exists(SESSION_FILE):
        sys.exit(0)

    with open(SESSION_FILE) as f:
        log_file = f.read().strip()

    # Get git diff stat for session summary
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

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

except Exception:
    pass  # Fail-open

sys.exit(0)
