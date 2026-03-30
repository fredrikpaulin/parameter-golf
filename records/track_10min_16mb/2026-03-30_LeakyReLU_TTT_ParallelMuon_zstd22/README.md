# LeakyReLU² + Legal TTT + Parallel Muon + zstd-22

**val_bpb: TBD** (pending 8xH100 runs) | 8×H100 SXM

## Changes from SOTA (1.1194 BPB)

Fork of `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` with:

1. **zstd-22 compression** (was LZMA preset 6) — saves ~400KB on artifact, giving headroom for bigger model or safety margin under 16MB limit
2. **BigramHash 3072** (was 1536) — SOTA ablation showed -0.0009 BPB from 2048→3072

## Recommended Run Command

```bash
pip install zstandard  # required for zstd-22 compression

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Experiments to Sweep

All tunable via env vars, no code changes needed:

| Experiment | Change | Expected |
|-----------|--------|----------|
| BigramHash 4096 | `BIGRAM_VOCAB_SIZE=4096` | -0.0005 more? |
| TTT 5 epochs | `TTT_EPOCHS=5` | -0.0005 BPB |
| TTT higher LR | `TTT_LR=0.003` | +/- 0.001 BPB |
| Warmdown 4000 | `WARMDOWN_ITERS=4000` | +/- 0.0005 BPB |

## Base

- LeakyReLU² activation, Legal Score-First TTT, Parameter Banking + Parallel Muon
- PR #414 stack, PR #399 optimizer, PR #461 TTT, PR #493/#518 activation
