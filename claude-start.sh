#!/usr/bin/env bash
# Launch Claude Code autonomous experiment loop for diffusion
# Usage: ./claude-start.sh [optional-tag]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-$(date +%b%d | tr '[:upper:]' '[:lower:]')-diff}"

cd "$SCRIPT_DIR"

claude "Read program.md and kick off the autonomous experiment loop. Use run tag: $TAG" \
  --allowedTools "Bash(*)" "Read" "Edit" "Write" "Grep" "Glob" \
