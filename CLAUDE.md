# Parameter Golf — Diffusion Experiment

## Active Experiment: Discrete Diffusion LM (MLX)

Working directory: `agent-data/experiments/diffusion/`

### Quick Start

```bash
cd /path/to/parameter-golf
TIME_BUDGET=120 python3 -u agent-data/experiments/diffusion/train.py 2>&1 | tee run.log
```

### Experiment Loop (Autonomous)

Follow the loop in `agent-data/experiments/diffusion/program.md`:

1. Read current `train.py` and `results.tsv` to understand state
2. Pick one change from the ideas list below (or devise your own based on results so far)
3. Edit `train.py` — make ONE change at a time so we can attribute improvements
4. Run: `TIME_BUDGET=120 python3 -u agent-data/experiments/diffusion/train.py 2>&1 | tee run.log`
5. Extract `val_bpb` from the output
6. Append result to `results.tsv` with format: `tag\tval_bpb\tstatus\tdescription`
7. If BPB improved: keep the change, commit with `git add agent-data/experiments/diffusion/ && git commit -m "diffusion: <description>"`
8. If BPB worsened: revert `train.py` to previous version
9. Go to step 1

### Current Status

Best BPB: **2.288** (6L/384d, RoPE, fixed 50% masking, Muon optimizer, 300s training)

**KEY BREAKTHROUGH**: RoPE (rotary position embeddings) was the critical missing piece. Without positional info in attention, the model couldn't learn to use context — it learned a global prior instead (CE ~5.9 at ALL noise levels). With RoPE, BPB dropped from 3.45 → 2.42 in a single change.

**Current findings**:
- Fixed 50% masking >> variable-t training (the model can't handle attention at high masking rates during training)
- Muon optimizer helps slightly over AdamW
- No time embedding needed (mask pattern implicitly encodes noise level)
- Loss still dropping at 300s — more training time helps
- Target: AR baseline is 1.11 BPB

### Known Issues

- MLX optimizer API: must use dict-based `optimizer.apply_gradients(grads_flat, params_flat)` then `model.update()`. The model-based call is broken.
- Variable-t training causes attention collapse even with RoPE — the model learns to ignore attention because high-t training is noisy
- Fixed masking rate mismatches ELBO eval but transfers well to low-t; poorly to high-t

### Ideas to Try (Priority Order)

1. **[DONE - no effect] Zero skip / remove skip** — skip was removed, model works without it
2. **[DONE - marginal] Higher LR / different schedules / ELBO steps** — all gave ~3.48 BPB
3. **[DONE - BREAKTHROUGH] RoPE** — unlocked context usage, BPB 3.45→2.42
4. **[DONE - best] Muon optimizer + fixed 50% masking** — current best config
5. **More training time** — 600s run in progress, loss still dropping
6. **Scale model** — try 4L/512d (~11.5M params) for wider attention
7. **Self-conditioning** — double effective depth at eval time
8. **Multi-rate masking per batch** — each sequence gets different mask rate
9. **Increase seq_len to 1024** — match AR baseline context length
10. **Weight tying with RoPE** — save params, might help at this scale

### Architecture

- Absorbing-state MDLM: forward process masks tokens, model denoises
- Bidirectional transformer (no causal mask)
- Skip connection: identity signal at unmasked positions, zeroed at masked (after fix)
- Cosine noise schedule
- ELBO-based BPB evaluation
- 1024-token BPE vocabulary (sentencepiece)

### Constraints

- 16MB artifact limit (code + compressed model weights)
- BPB evaluated on FineWeb validation set
- Must use sentencepiece tokenizer at `data/tokenizers/fineweb_1024_bpe.model`
- Training data at `data/datasets/fineweb10B_sp1024/`
- Target: beat AR baseline of 1.11 BPB (ambitious — start by getting below 3.0)

### Don'ts

- Don't modify data loading paths or tokenizer
- Don't install packages beyond what's available (mlx, numpy, sentencepiece)
- Don't change the ELBO evaluation methodology (it's correct per MDLM paper)
- Don't make multiple changes per run — isolate variables
