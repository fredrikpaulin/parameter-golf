#!/bin/bash
# Start Claude Code for autonomous parameter-golf experimentation
# Pre-allows common tools so the agent can run unattended

cd "$(dirname "$0")"

claude "Read CLAUDE.md and program.md, then follow the instructions. Begin the setup phase." \
  --allowedTools "Bash(*)" \
  --allowedTools "Edit" \
  --allowedTools "Write" \
  --allowedTools "Read"
