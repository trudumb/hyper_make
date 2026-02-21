#!/usr/bin/env bash
# PostToolUse hook for Edit|Write: advisory clippy reminder for .rs files.
# Does NOT auto-run clippy (10 edits = 10 runs at 30-60s each is too expensive).
# Instead: prints a reminder to stderr if clippy has not run recently.
# Non-blocking (exit 0 always).

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)

# Only care about Rust files
if [[ "$FILE_PATH" != *.rs ]]; then
    exit 0
fi

# Check if clippy has run in the last 5 minutes
LAST_CLIPPY="$HOME/.claude/last_clippy_ts"

if [ -f "$LAST_CLIPPY" ]; then
    last_ts=$(cat "$LAST_CLIPPY" 2>/dev/null || echo "0")
    now_ts=$(date +%s)
    age=$(( now_ts - last_ts ))
    if [ "$age" -lt 300 ]; then
        exit 0
    fi
fi

echo "Reminder: Run 'cargo clippy -- -D warnings' before completing your task. Edited: $FILE_PATH" >&2
exit 0
