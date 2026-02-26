#!/usr/bin/env python3
"""
PostToolUse hook for Bash: release cargo mutex after command completes.
Always exits 0 (post-hooks are advisory).
"""
import json
import sys
import os

LOCK_FILE = os.path.expanduser("~/.claude/cargo.lock")
CARGO_COMMANDS = ["cargo test", "cargo clippy", "cargo build", "cargo check", "cargo run"]

try:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.exit(0)

    data = json.loads(raw)
    command = data.get("tool_input", {}).get("command", "")

    if any(command.strip().startswith(c) for c in CARGO_COMMANDS):
        try:
            os.remove(LOCK_FILE)
        except (FileNotFoundError, OSError):
            pass
except Exception as e:
    # Log unexpected errors to stderr for debugging, but never block
    print(f"cargo_mutex_release: unexpected error: {e}", file=sys.stderr)

sys.exit(0)
