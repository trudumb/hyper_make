#!/usr/bin/env python3
"""
PreToolUse hook for Bash: block binary execution unless explicitly allowed.
Trading binaries require user to run manually (Core Rule #5).
Exit 0 = allow, Exit 2 = block with feedback.
All exceptions fail-open (exit 0).
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
    try:
        data = json.load(sys.stdin)
        command = data.get("tool_input", {}).get("command", "")

        # Check safe patterns first (allow list takes priority)
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

    except Exception:
        # Fail-open
        sys.exit(0)


if __name__ == "__main__":
    main()
