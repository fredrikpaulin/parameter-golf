# Diffusion LM Experiment — Session Context

## What This Is

Absorbing-state MDLM (Masked Diffusion Language Model) trained on FineWeb with MLX on Apple Silicon. Goal: minimize validation BPB (bits per byte). AR baseline is 1.11 BPB.

## Current Best: 1.9206 BPB

Achieved with: 6L/384d, RoPE, QK-norm, variable-t [0.1-0.6], ELBO importance weighting, stratified t sampling (10 strata), Muon optimizer, 3600s training, 30174 steps. Loss was still dropping at end of training.

## Architecture

- Bidirectional transformer (no causal mask), 6L/384d, 6 heads, MLP 3x (relu²), seq 512, ~9.8M params
- 1024-token BPE vocab (sentencepiece), +1 mask token
- RoPE in attention
- QK-norm: rms_norm on Q and K per-head after RoPE, before dot product
- t-conditioning: linear projection of scalar t added to input embeddings
- Logit softcap (30.0)
- Cosine noise schedule, t sampled from [0.1, 0.6] with stratified sampling (10 strata cycling)
- ELBO importance weighting: loss *= dalpha/mask_prob (makes loss an unbiased ELBO estimate)

## Optimizer

Muon (Newton-Schulz orthogonalization for 2D weight matrices, Adam for embeddings/scalars). AdamW cannot train this model — attention never learns. Muon LR=0.02, Embed LR=0.03, Scalar LR=0.02, momentum=0.95, grad_clip=1.0, warmup=50 steps, warmdown=15%.

## Key Breakthroughs (chronological)

1. **RoPE** — BPB 3.45→2.42. Without positional info, model learned unigram prior (CE ~5.9 at all t).
2. **ELBO importance weighting + variable-t** — BPB 2.42→2.18. Loss multiplied by dalpha/mask_prob prevents high-t gradient domination.
3. **Tighter t range [0.1-0.6]** — BPB 2.18→2.08. Avoids near-random high-t regime.
4. **QK-norm** — BPB 2.08→1.92. Eliminates gnorm spikes, stabilizes long training.

## Per-t Diagnostics (best run, 30174 steps)

| t | CE | Notes |
|---|---|---|
| 0.05 | 0.86 | Below AR baseline (1.11) |
| 0.10 | 1.12 | At AR baseline |
| 0.20 | 1.38 | |
| 0.30 | 1.85 | |
| 0.50 | 3.08 | |
| 0.70 | 4.82 | |
| 0.90 | 5.88 | |

## What Failed

- **Extended t range [0.05-0.6]** — ELBO weight dalpha/mask_prob ≈ 23 at t=0.05, too much gradient variance.
- **EMA weights** — With rapid improvement, EMA decay 0.999 averages final weights with much worse early weights. Catastrophic at short training.
- **Self-conditioning** — Double forward cost, model not trained for it at eval. BPB exploded.
- **AdamW** — Cannot train attention at all (BPB stays 3.47). Muon is non-negotiable.
- **AdaLN-Zero** — Too many extra params for no gain.
- **SwiGLU** — Marginal gain not worth 25% more params.
- **Depth recurrence 3L×2** — Competitive at short training, but unique 6L wins with more steps.
- **Weight tying** — Hurts performance.
- **Larger batch** — Fewer steps hurt more than larger batch helps.
- **seq_len=1024** — Slower per step, fewer total steps at same time budget. Revisit with longer training.

## Experiment Protocol

1. Make ONE change at a time
2. Test at 120s first (`TIME_BUDGET=120`)
3. Compare 120s BPB to baseline (~2.46 at 120s with QK-norm)
4. If promising, run full 3600s (`TIME_BUDGET=3600`)
5. Log to results.tsv: `tag\tval_bpb\tstatus\tdescription`
6. If better: keep change, commit with `git add -f agent-data/experiments/diffusion/ && git commit`
7. If worse: revert train.py

## Run Command

```bash
cd ~/parameter-golf
TIME_BUDGET=120 python3 -u agent-data/experiments/diffusion/train.py 2>&1 | tee run.log
```

For long runs, use `caffeinate` to prevent thermal throttling:
```bash
caffeinate -i python3 -u agent-data/experiments/diffusion/train.py
```

## Current Experiment: Learnable Attention Temperature

Added learnable per-head log-scale parameter `attn_log_scale` initialized to `log(1/sqrt(64)) ≈ -2.08`. With QK-norm making Q,K unit-RMS, the fixed `1/sqrt(hd)` scaling may be suboptimal — heads could benefit from learning their own sharpness. Parameter trained via Adam (scalar param).

120s baseline to beat: ~2.4558 (qknorm-120s).

## Remaining Ideas (Priority)

1. **Learnable attention temperature** ← currently testing
2. **Scale model** (8L/512d or similar) — now that QK-norm works, more capacity could help
3. **More training time** (7200s) — loss was still dropping at 30K steps
4. **seq_len=1024** — revisit with longer training budget
5. **Different noise schedule parameters** — cosine s parameter tuning
6. **Wider MLP** (4x instead of 3x) — more FFN capacity
7. **More heads** (8 or 12 instead of 6) — finer-grained attention patterns

## Known Issues

- MLX optimizer: must use dict-based `optimizer.apply_gradients(grads_flat, params_flat)` then `model.update()`. Model-based call is broken.
- Machine thermal throttling on long runs — use `caffeinate -i` wrapper.
- git add needs `-f` flag (agent-data is in .gitignore).

## File Locations

- Training script: `agent-data/experiments/diffusion/train.py`
- Results log: `agent-data/experiments/diffusion/results.tsv`
- Experiment plan: `agent-data/experiments/diffusion/program.md`
- Project instructions: `CLAUDE.md`
- Tokenizer: `data/tokenizers/fineweb_1024_bpe.model`
- Training data: `data/datasets/fineweb10B_sp1024/`

## Full Results History

55 experiments run. See `results.tsv` for complete log. Key milestones:
- Baseline: 3.49 BPB (plain transformer, no RoPE, AdamW)
- After RoPE: 2.42
- After ELBO weighting + variable-t: 2.18
- After tighter t range + long training: 2.08
- After QK-norm + full training: **1.92** ← current best
