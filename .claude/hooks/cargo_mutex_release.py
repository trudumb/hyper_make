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
    data = json.load(sys.stdin)
    command = data.get("tool_input", {}).get("command", "")

    if any(command.strip().startswith(c) for c in CARGO_COMMANDS):
        try:
            os.remove(LOCK_FILE)
        except FileNotFoundError:
            pass
except Exception:
    pass

sys.exit(0)
