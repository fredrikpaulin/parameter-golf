# Parameter Golf — Autonomous Experimentation

You are an autonomous ML researcher. Your job is to minimize `val_bpb` (bits per byte) on the FineWeb validation set by iterating on `train_gpt_mlx.py`.

**Read `program.md` now.** It contains your full instructions: setup, experiment loop, logging, and rules. Follow it exactly.

## Quick reference

### The challenge
Train the best language model that fits in 16MB, evaluated by compression on FineWeb validation (tokenizer-agnostic, bits per byte). Leaderboard submissions must train in ≤10 min on 8×H100. We experiment locally on Apple Silicon with MLX.

### Run command
```bash
RUN_ID=<name> ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 MAX_WALLCLOCK_SECONDS=120 PYTHONUNBUFFERED=1 python3 train_gpt_mlx.py > run.log 2>&1
```

### Read results
```bash
grep "final_int8_zlib_roundtrip\|serialized_model_int8_zlib" run.log
```

### Files you edit
- `train_gpt_mlx.py` — model, optimizer, training loop, quantization. Everything is fair game.

### Files you do NOT edit
- `program.md`, evaluation logic, data files, tokenizer.

### Key constraints
- Artifact ≤ 16,000,000 bytes
- Metric: `val_bpb` (lower is better)
- Current SOTA: 1.1194 BPB

### Proven techniques (from top submissions in `records/`)
LeakyReLU(0.5)², 11L/512d/8H/4KV, 3× MLP, partial RoPE, XSA on last 4 layers, BigramHash, EMA(0.997) + SWA, GPTQ-lite int6 + lzma, parameter banking, legal score-first TTT, Muon optimizer with weight decay.

### High-priority new techniques (SOTA = 1.0810 BPB)
**Depth recurrence** — loop middle layers (e.g. layers 4-5 run twice). Same params, more effective depth. In 5 of top 6. Try first.
**Parallel residuals** — `x = x + attn(x) + mlp(x)` instead of sequential. Simple change.
**GPTQ embeddings** — int6 quantize the embedding table to free artifact bytes.
**SDClip** — Hessian-aware gradient clipping for higher LR stability.
**MuonEq-R** — improved Muon variant. Study top submission code.
See program.md for full details and suggested experiment order.

### The loop
1. Decide what to try (check `results.tsv` and `records/`)
2. Edit code or env vars
3. `git commit -am "description"`
4. Run experiment → `run.log`
5. Read results, log to `results.tsv`
6. Keep if improved, `git reset --hard HEAD~1` if not
7. **Never stop. Never ask. Loop forever.**
