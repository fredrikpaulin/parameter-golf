#!/bin/bash
# Long training run with auto-restart on OOM crash.
# Checkpoints save every 200K steps. On crash, this script
# restarts training and it resumes from the last checkpoint.
#
# Usage: TIME_BUDGET=43200 bash agent-data/experiments/diffusion/run_long.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

MAX_RESTARTS=5
attempt=0

while [ $attempt -lt $MAX_RESTARTS ]; do
    attempt=$((attempt + 1))
    echo "=== Training attempt $attempt / $MAX_RESTARTS ==="

    python3 -u agent-data/experiments/diffusion/train.py 2>&1 | tee run.log
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "=== Training completed successfully ==="
        exit 0
    fi

    # Check if checkpoint exists (meaning we can resume)
    if [ -f "$SCRIPT_DIR/checkpoints/state.json" ]; then
        echo "=== Crashed (exit $exit_code), checkpoint found — restarting in 5s ==="
        sleep 5
    else
        echo "=== Crashed (exit $exit_code), no checkpoint — cannot resume ==="
        exit $exit_code
    fi
done

echo "=== Exhausted $MAX_RESTARTS restart attempts ==="
exit 1
