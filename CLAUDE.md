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

Best BPB: **1.8013** (6L/384d, RoPE, QK-norm, GeLU, variable-t [0.1-0.5] + ELBO weighting, stratified t sampling (8 strata), Muon optimizer, 21600s training, 201K steps)

**KEY BREAKTHROUGHS** (in order of impact):
1. **RoPE** — Without positional info in attention, the model learned a global unigram prior (CE ~5.9 at ALL noise levels). With RoPE, BPB dropped 3.45 → 2.42 in a single change.
2. **ELBO importance weighting** — Loss multiplied by `dalpha/mask_prob` downweights high-t steps where gradients are noisy. This made variable-t training work (previously caused attention collapse).
3. **Variable-t training [0.05-0.75]** — With ELBO weighting, variable-t outperforms fixed 50% masking. Tight range avoids the near-random high-t regime.
4. **QK-norm** — rms_norm on Q,K before attention dot product. Eliminates gnorm spikes, stabilizes training. BPB 2.0758→1.9206 (full 3600s run). Model now beats 2.0 barrier.

**Current findings**:
- Muon optimizer is ESSENTIAL — AdamW cannot train attention (BPB stays 3.47)
- t-conditioning via linear projection helps model adapt to noise level
- No time embedding needed beyond simple linear t_embed
- Loss still dropping at 3600s/31K steps — more training time keeps helping
- Logit softcap helps stability for long training (removing it hurt 600s runs)
- Depth recurrence (3L×2) is competitive for short runs but unique layers win with more training
- Stratified t sampling (10 strata) smooths loss/gnorm curves, marginal BPB gain (2.0812→2.0758)
- QK-norm (rms_norm on Q,K before attention) stabilizes training, eliminates gnorm spikes, BPB 2.0758→1.9206
- Loss still dropping at 30K steps (final smoothed loss ~10.2) — more training time will continue improving
- At t=0.05 the model achieves CE=0.86, which is BELOW the AR baseline of 1.11
- Target: AR baseline is 1.11 BPB

**Per-t diagnostics (21600s GeLU run, 201K steps)**:
- t=0.05: CE=0.38
- t=0.10: CE=0.75
- t=0.20: CE=1.07 ← below AR baseline (1.11)
- t=0.30: CE=1.42
- t=0.50: CE=2.90
- t=0.70: CE=4.72
- t=0.90: CE=6.21

### Known Issues

- MLX optimizer API: must use dict-based `optimizer.apply_gradients(grads_flat, params_flat)` then `model.update()`. The model-based call is broken.
- Gradient norm variance from ELBO weighting — dalpha/mask_prob creates spikes near t extremes. Addressed by stratified t sampling (10 strata cycling) and tightening t range to [0.1-0.6].
- Self-conditioning was tried and failed — too expensive (double fwd cost) and model wasn't trained for it at eval time.

### Ideas to Try (Priority Order)

1. **[DONE - no effect] Zero skip / remove skip** — skip removed, model works without it
2. **[DONE - marginal] Higher LR / different schedules / ELBO steps** — all gave ~3.48 BPB
3. **[DONE - BREAKTHROUGH] RoPE** — unlocked context usage, BPB 3.45→2.42
4. **[DONE - BREAKTHROUGH] ELBO importance weighting + variable-t** — made variable-t work, BPB 2.42→2.18
5. **[DONE - best] Muon optimizer** — essential for training attention
6. **[DONE - hurt] Self-conditioning** — double fwd cost, model not trained for it
7. **[DONE - no gain] AdaLN-Zero** — too many params for no gain
8. **[DONE - marginal] Depth recurrence 3L×2** — competitive short runs, worse long runs
9. **[DONE - marginal] SwiGLU activation** — marginal gain (2.1775 vs 2.1787) not worth 25% more params
10. **[DONE - BEST] Even tighter t range [0.1-0.6]** — combined with longer training, BPB 2.18→2.08
11. **[DONE - marginal] Variance reduction** — stratified t sampling (10 strata), smoother training, BPB 2.0812→2.0758
12. **[DONE - BEST] More training time** — 2400s→2.14, 3600s→2.08, loss still dropping
13. **Scale model** — now that training works, try larger models with longer training
14. **Increase seq_len to 1024** — was tried at 300s and hurt (fewer steps), revisit with longer training
15. **[DONE - BEST] QK-norm** — rms_norm on Q,K before attention, BPB 2.0758→1.9206 (full 3600s, 30174 steps)
16. **[DONE - hurt] Extended t range [0.05-0.6]** — low-t ELBO weights too high, hurt training
17. **[DONE - hurt] EMA weights** — EMA lags behind rapidly improving weights at short training

### Architecture

- Absorbing-state MDLM: forward process masks tokens, model denoises
- Bidirectional transformer (no causal mask)
- RoPE (rotary position embeddings) in attention
- QK-norm (rms_norm on Q and K per-head before dot product)
- t-conditioning: linear projection of scalar t added to input embeddings
- Variable-t training with ELBO importance weighting (dalpha/mask_prob)
- Cosine noise schedule, t sampled from [0.1, 0.5] with stratified sampling (8 strata)
- Muon optimizer (Newton-Schulz for 2D matrices, Adam for embeddings/scalars)
- Logit softcap (30.0)
- ELBO-based BPB evaluation (64 steps)
- 1024-token BPE vocabulary (sentencepiece)
- 6L/384d, 6 heads, MLP 3x, seq 512, ~9.8M params

### Constraints

- 16MB artifact limit (code + compressed model weights)
- BPB evaluated on FineWeb validation set
- Must use sentencepiece tokenizer at `data/tokenizers/fineweb_1024_bpe.model`
- Training data at `data/datasets/fineweb10B_sp1024/`
- Target: beat AR baseline of 1.11 BPB (broke 2.0 barrier — now targeting below 1.5)

### Don'ts

- Don't modify data loading paths or tokenizer
- Don't install packages beyond what's available (mlx, numpy, sentencepiece)
- Don't change the ELBO evaluation methodology (it's correct per MDLM paper)
- Don't make multiple changes per run — isolate variables
- Don't use AdamW — Muon is essential for training attention
