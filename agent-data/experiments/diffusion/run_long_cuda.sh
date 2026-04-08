#!/bin/bash
# Long CUDA training with auto-restart on crash.
# Usage: TIME_BUDGET=86400 bash run_long_cuda.sh
# Override batch size: BATCH_TOKENS=65536 TIME_BUDGET=600 bash run_long_cuda.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MAX_RESTARTS=10
attempt=0

while [ $attempt -lt $MAX_RESTARTS ]; do
    attempt=$((attempt + 1))
    echo "=== Training attempt $attempt / $MAX_RESTARTS ==="

    python3 -u train_cuda.py 2>&1 | tee run_cuda.log
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "=== Training completed successfully ==="
        exit 0
    fi

    if [ -f "$SCRIPT_DIR/checkpoints_cuda/checkpoint.pt" ]; then
        echo "=== Crashed (exit $exit_code), checkpoint found — restarting in 5s ==="
        sleep 5
    else
        echo "=== Crashed (exit $exit_code), no checkpoint — cannot resume ==="
        exit $exit_code
    fi
done

echo "=== Exhausted $MAX_RESTARTS restart attempts ==="
exit 1
