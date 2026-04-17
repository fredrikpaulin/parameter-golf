#!/bin/bash
# H100 proxy run on 2xRTX 5080 — mirrors the 10-min 8xH100 recipe.
#
# Why this exists:
#   The parameter-golf challenge scores a 10-min run on 8xH100. Before spinning
#   up a pod, we want a local standard experiment on hightower that PREDICTS
#   what that H100 run will produce. A naive wall-time scale doesn't work —
#   effective batch size dominates the optimization trajectory, and our usual
#   5080 recipe (~32K eff batch, many steps) lives in a different regime than
#   the H100 recipe (~524K eff batch, fewer steps).
#
# What this does:
#   - Matches the 8xH100 effective batch exactly: 524,288 tokens/step
#     (H100: 65,536 tok/gpu × 8 gpus; here: 16,384 tok/gpu × 2 gpus × 16 accum)
#   - Runs ~8400s so total tokens processed is comparable to 10 min on 8xH100
#     (empirical throughput ratio ≈ 14x: 2.51M vs 0.18M tok/s)
#   - Uses the current best architecture (SP8192, 6L/768d, 48M params)
#
# What a result here tells you:
#   The BPB a 10-min 8xH100 run of the same recipe would land near. If proxy
#   BPB is X, expect the real H100 run to give X ± small noise. This is
#   different from our 6h/24h "quality ceiling" runs, which benefit from
#   more gradient steps at small batch and will read LOWER than the H100 run.
#
# Expected runtime: ~2h 20min wall-clock on 2xRTX 5080.
# Expected step count: ~2500-2900 optimizer steps (vs 2874 for 8xH100-muon3).
#
# Usage:
#   bash run_h100_proxy.sh                       # default: SP8192, 8400s
#   VOCAB_SIZE=1024 bash run_h100_proxy.sh       # rerun SP1024 comparison
#   TIME_BUDGET=600 bash run_h100_proxy.sh       # quick smoke test
#
# Prereqs:
#   - train_cuda.py must support VOCAB_SIZE env var (hightower copy does)
#   - data/datasets/fineweb10B_sp8192/ shards present
#   - data/tokenizers/fineweb_8192_bpe.model present

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export TIME_BUDGET=${TIME_BUDGET:-8400}
export VOCAB_SIZE=${VOCAB_SIZE:-8192}
export BATCH_TOKENS=${BATCH_TOKENS:-16384}
export GRAD_ACCUM=${GRAD_ACCUM:-16}

EFF_BATCH=$((BATCH_TOKENS * 2 * GRAD_ACCUM))

echo "=== H100 proxy run ==="
echo "  TIME_BUDGET   = ${TIME_BUDGET}s"
echo "  VOCAB_SIZE    = ${VOCAB_SIZE}"
echo "  BATCH_TOKENS  = ${BATCH_TOKENS} per GPU"
echo "  GRAD_ACCUM    = ${GRAD_ACCUM}"
echo "  WORLD_SIZE    = 2 (2x RTX 5080)"
echo "  eff batch     = ${EFF_BATCH} tokens/step (target: 524288)"
echo "======================"

torchrun --standalone --nproc_per_node=2 train_cuda.py 2>&1 | tee run_h100_proxy.log
