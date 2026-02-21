#!/usr/bin/env python3
"""
PreToolUse hook for Bash: enforce sequential cargo commands.
File-based mutex prevents concurrent cargo builds that crash the machine.
Exit 0 = allow, Exit 2 = block with feedback message on stderr.
All exceptions fail-open (exit 0).
"""
import json
import sys
import os
import time

LOCK_FILE = os.path.expanduser("~/.claude/cargo.lock")
CARGO_COMMANDS = ["cargo test", "cargo clippy", "cargo build", "cargo check", "cargo run"]
STALE_TIMEOUT_S = 600  # 10 min


def main():
    try:
        data = json.load(sys.stdin)
        command = data.get("tool_input", {}).get("command", "")

        # Only gate cargo commands
        if not any(command.strip().startswith(c) for c in CARGO_COMMANDS):
            sys.exit(0)

        # Check existing lock
        if os.path.exists(LOCK_FILE):
            try:
                with open(LOCK_FILE) as f:
                    lock_info = json.load(f)
                age_s = time.time() - lock_info.get("timestamp", 0)
                if age_s > STALE_TIMEOUT_S:
                    os.remove(LOCK_FILE)
                else:
                    owner = lock_info.get("command", "unknown")
                    print(
                        f"BLOCKED: Another cargo command is running: '{owner}' "
                        f"(started {int(age_s)}s ago). Wait for it to finish. "
                        f"Only ONE cargo command at a time -- concurrent builds crash the machine.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
            except (json.JSONDecodeError, OSError):
                try:
                    os.remove(LOCK_FILE)
                except OSError:
                    pass

        # Acquire lock
        os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
        with open(LOCK_FILE, "w") as f:
            json.dump({
                "command": command[:80],
                "timestamp": time.time(),
                "pid": os.getpid(),
            }, f)

        sys.exit(0)

    except Exception:
        # Fail-open: never block on hook errors
        sys.exit(0)


if __name__ == "__main__":
    main()
