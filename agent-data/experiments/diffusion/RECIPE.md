# Discrete Diffusion LM — Best Recipe & Experiment Log

## Best Result

**1.7344 BPB** after 24 hours of training (809K steps) on Apple Silicon (MLX). The autoregressive baseline on the same data/tokenizer is 1.11 BPB. At low noise levels the model already beats this baseline — t=0.20 achieves CE=0.96, t=0.30 achieves CE=1.09 — but the ELBO integral over all noise levels pulls the aggregate score up.

## Architecture

The model is a 9.6M-parameter bidirectional transformer trained as an absorbing-state masked diffusion language model (MDLM). Tokens are masked according to a cosine noise schedule, and the model learns to predict the original tokens at masked positions. Evaluation uses the ELBO bound with 64-step midpoint quadrature.

**Model**: 6 layers, 384 hidden dim, 6 attention heads (64d per head), MLP expansion 3x, sequence length 512. 1024-token BPE vocabulary (sentencepiece on FineWeb).

**Key components** (each was validated by ablation):

- **RoPE** in attention. Without positional information the model learns a global unigram prior regardless of noise level (CE ~5.9 everywhere). RoPE was the single largest improvement: BPB 3.45 → 2.42.
- **QK-norm** (RMS normalization on Q and K per-head before the dot product). Eliminates gradient norm spikes and stabilizes long training. BPB 2.08 → 1.92 at 3600s.
- **GeLU activation** in the MLP (replacing squared ReLU). Smoother gradients, lower loss plateau. BPB 1.92 → 1.91 at 3600s, with the gap widening at longer training.
- **Logit softcap** at 30.0 (tanh clamping). Marginal at short budgets but prevents divergence over long runs. Removing it at 600s was slightly better but at 3600s+ the model becomes unstable.
- **t-conditioning**: a single linear projection from scalar t to the model dimension, added to the input embeddings. Minimal overhead, helps the model adapt its predictions to the noise level.

**Optimizer**: Muon (Newton-Schulz orthogonalization, 5 iterations) for all 2D weight matrices, Adam (betas 0.9/0.95) for embeddings and scalar parameters. Muon LR 0.02, embedding LR 0.03, scalar LR 0.02. 50-step linear warmup, flat schedule for 85% of training, linear warmdown over the final 15%.

**Training loss**: variable-t with ELBO importance weighting. At each step a noise level t is sampled from [0.1, 0.5] using stratified sampling (8 strata of width 0.05, cycled deterministically). Tokens are masked at rate alpha(t) from the cosine schedule. The cross-entropy loss on masked positions is multiplied by dalpha/dt / mask_prob, making it an unbiased estimate of the ELBO. Gradient clipping at 1.0.

## Breakthrough Timeline

The project started at 3.49 BPB and reached 1.73 over roughly 75 experiments. The breakthroughs came in clusters.

**Phase 1 — Discovering the model can't use context (BPB ~3.48)**. The initial model with no positional encoding learned a bag-of-tokens unigram prior. Sinusoidal embeddings, AdaLN-Zero conditioning, weight tying, larger models — nothing moved the needle beyond noise. The model literally could not attend to position.

**Phase 2 — RoPE unlocks context (3.48 → 2.42)**. Adding rotary position embeddings was transformative. The model immediately started using local context, and BPB dropped by a full point. This also revealed that Muon was essential: AdamW could not train the attention mechanism at all (BPB stuck at 3.47 with or without RoPE).

**Phase 3 — Variable-t and ELBO weighting (2.42 → 2.08)**. Fixed 50% masking worked well initially but couldn't match a proper variable-t objective. The key insight was that naive variable-t training fails because high-noise steps generate huge, noisy gradients. ELBO importance weighting (loss × dalpha/mask_prob) solved this by making the loss an unbiased ELBO estimate while naturally downweighting the noisy regime. Combined with narrowing the t range from [0.05, 0.95] to [0.1, 0.6] and then [0.1, 0.5], this pushed BPB below 2.1.

**Phase 4 — Stability improvements (2.08 → 1.92)**. QK-norm was the standout here. RMS-normalizing Q and K before the attention dot product eliminates periodic gradient norm spikes that had been limiting learning rate and long-run stability. Stratified t sampling (dividing the t range into equal strata and cycling through them) further smoothed training.

**Phase 5 — Scaling training time (1.92 → 1.73)**. With the recipe locked in, longer training kept improving: 1 hour → 1.91, 6 hours → 1.80, 12 hours → 1.76, 24 hours → 1.73. The loss is still dropping at 809K steps. An MLX metal memory limit (~499K allocations) crashes training at ~248K steps, worked around with periodic checkpointing and an auto-restart wrapper.

## Per-t Diagnostics (24-hour run, 809K steps)

| Noise level | CE at masked positions | vs AR baseline (1.11) |
|---|---|---|
| t = 0.05 | 0.56 | 2.0x better |
| t = 0.10 | 0.67 | 1.7x better |
| t = 0.20 | 0.96 | below baseline |
| t = 0.30 | 1.09 | below baseline |
| t = 0.50 | 2.71 | above baseline |
| t = 0.70 | 4.75 | above baseline |
| t = 0.90 | 6.09 | above baseline |

The model excels at low noise (few masked tokens, strong context available) and struggles at high noise (most tokens masked, approaching random). The ELBO integrates across all levels, so the high-t regime — where denoising is nearly impossible — still drags the aggregate score above the AR baseline.

## Rejected Experiments

### Didn't help at all

**AdamW optimizer** — Tried twice, with and without frozen RoPE. AdamW cannot train bidirectional attention with Muon-initialized weights. BPB stays at 3.47 regardless of learning rate. The Newton-Schulz orthogonalization in Muon appears essential for learning attention patterns in this architecture.

**AdaLN-Zero** (adaptive layer norm with zero-initialized scale/shift per layer, conditioned on t). Added 5M parameters for zero BPB improvement. The simple linear t-conditioning works just as well at a fraction of the cost.

**Self-conditioning** (feeding the model's own previous prediction as auxiliary input). Doubled the forward pass cost. At training time the extra expense meant fewer steps per second. At eval time the model hadn't been trained to use its own predictions, so the signal was meaningless. Even with self-conditioning during both training and eval, the step-count penalty dominated.

**Weight tying** (shared embedding and output projection). Hurt performance at every training budget tested. The input embedding needs to represent masked tokens; the output head only predicts over the real vocabulary. Tying them creates a conflict.

**EMA weights** for evaluation. With rapid early learning, EMA (decay 0.999) lags far behind the live weights. By the time EMA catches up, the run is over. Might help at very long training budgets but we never found a regime where it was worth it.

**Weight decay** (0.01 on matrix parameters). Pure regularization fighting against useful learning. BPB went from 1.92 to 1.95 at 3600s. The model is undertrained, not overfit.

**Learnable attention temperature** (per-head log-scale parameter on attention scores). Hurt across all t values. QK-norm already controls attention magnitude; adding a learnable scale on top introduces instability.

### Helped marginally (not worth the complexity)

**SwiGLU activation** (gated linear unit, 12.3M params vs 9.8M). BPB 2.178 vs 2.179 — statistically identical while using 25% more parameters. GeLU without gating is simpler and equally effective.

**Depth recurrence** (3 unique layers looped twice, 5.2M params). Competitive at short budgets due to faster per-step speed, but unique layers always win given sufficient training time. At 600s, 6 unique layers beat 3×2 by 0.02 BPB.

**Stratified t sampling** (dividing [0.1, 0.5] into uniform strata). Produced visibly smoother loss and gradient norm curves, but BPB improvement was only 2.0812 → 2.0758. Kept in the recipe because it costs nothing and makes training more predictable.

**Cosine noise schedule parameter** (s=0.03 vs default 0.008). BPB 1.9038 vs 1.9067. Within noise. The cosine schedule is robust to this parameter.

### Fundamentally flawed approaches

**Curriculum training** (fixed 50% masking for 70% of training, then switch to variable-t ELBO). Loss spiked from 3 to 12 at the switch point. The model trained exclusively at t=0.5 has never seen low-t inputs. When variable-t kicks in, the ELBO importance weights amplify terrible predictions at t=0.1-0.2 (where dalpha/mask_prob is very large). Same failure mode at both 70/30 and earlier 70/30 splits.

**Frequency-informed masking** (masking rare tokens more often using sqrt(1/freq) weights). Training used biased per-token mask probabilities but ELBO evaluation uses uniform masking. This train/eval distribution mismatch degraded performance at every noise level. BPB 2.13 vs 1.91 baseline.

**Hadamard rotation before QK-norm** (Walsh-Hadamard transform to spread outlier energy across dimensions). The butterfly operations are expensive in MLX — 23% fewer steps per time budget. The outlier flattening provided no benefit over plain QK-norm, and the step-count penalty dominated.

**Extended t range [0.05, 0.6]** (adding a low-noise stratum below 0.1). The ELBO importance weight dalpha/mask_prob becomes extremely large near t=0 because mask_prob approaches zero while dalpha stays finite. These near-zero-noise steps produce enormous loss values that destabilize training.

**Narrower t range [0.1, 0.4]**. The model over-specialized on mid-range noise levels. t=0.05 CE regressed from 0.77 to 1.25 because the model never saw noise levels below 0.1 during training. The sweet spot is [0.1, 0.5] — broad enough to generalize, narrow enough to avoid the high-noise wasteland.

**Larger models at fixed compute budget**. Both 8L/384d (12.6M) and 6L/512d (16.8M) were tested at various budgets. At 120s they get 35% fewer steps and lose. At 3600s the 512d model still can't overcome its per-step speed penalty (22K vs 34K steps). The 6L/384d architecture hits a sweet spot of capacity vs throughput for this data and compute regime.

**Larger batch size** (16K tokens). Only 871 steps at 300s vs 2900+ at 4096 tokens. Fewer optimization steps always loses in this compute-limited regime. The 4096-token batch is small enough to maintain high step throughput while large enough for stable gradients.

## Open Questions

- The loss is still dropping at 809K steps. How far does more training time take us? Extrapolating the log-linear trend: 48 hours might reach ~1.68, 1 week might approach 1.5.
- Can the model beat the AR baseline (1.11 BPB) in aggregate? The per-t diagnostics suggest the low-noise regime is already there. The gap comes from t > 0.3 where denoising is inherently hard.
- Would a larger model (512d+) win given a 24-hour+ training budget? At 3600s the step-count penalty dominates, but at 86400s the model gets 800K+ steps either way.
- Is there a better loss weighting that focuses more on the productive t range without creating the instabilities we saw with extended/narrowed ranges?
