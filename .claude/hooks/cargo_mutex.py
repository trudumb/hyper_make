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
STALE_TIMEOUT_S = 300  # 5 min (reduced from 10 — PID check handles most cases faster)


def _is_pid_alive(pid):
    """Check if a process is still running. Returns True if alive, False if dead."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another user — treat as alive (safe default)
        return True
    except OSError:
        return False


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

                session_pid = lock_info.get("session_pid", lock_info.get("pid", 0))

                # Check if owning session is still alive
                if session_pid > 0 and not _is_pid_alive(session_pid):
                    # Session is dead — lock is orphaned, remove and proceed
                    try:
                        os.remove(LOCK_FILE)
                    except OSError:
                        pass
                else:
                    # Session alive (or no PID) — check staleness as backup
                    age_s = time.time() - lock_info.get("timestamp", 0)
                    if age_s > STALE_TIMEOUT_S:
                        try:
                            os.remove(LOCK_FILE)
                        except OSError:
                            pass
                    else:
                        owner = lock_info.get("command", "unknown")
                        print(
                            f"BLOCKED: Another cargo command is running: '{owner}' "
                            f"(started {int(age_s)}s ago, session_pid={session_pid}). "
                            f"Wait for it to finish. "
                            f"Only ONE cargo command at a time -- concurrent builds crash the machine.",
                            file=sys.stderr,
                        )
                        sys.exit(2)
            except (json.JSONDecodeError, OSError):
                try:
                    os.remove(LOCK_FILE)
                except OSError:
                    pass

        # Acquire lock — store parent (session) PID for liveness checks
        os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
        with open(LOCK_FILE, "w") as f:
            json.dump({
                "command": command[:80],
                "timestamp": time.time(),
                "session_pid": os.getppid(),
            }, f)

        sys.exit(0)

    except Exception:
        # Fail-open: never block on hook errors
        sys.exit(0)


if __name__ == "__main__":
    main()
