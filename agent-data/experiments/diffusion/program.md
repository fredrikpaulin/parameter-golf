# Text Diffusion Experiment

Autonomous experiment loop for masked discrete diffusion language modeling.

## Context

We're competing in the OpenAI Parameter Golf challenge: train the best LM that fits in 16MB, evaluated by BPB on FineWeb validation. This experiment explores whether discrete diffusion can outperform autoregressive modeling at small scales.

The key advantages of diffusion here:
1. **Bidirectional context** — every position sees all other positions (no causal mask). Strictly more information per parameter than AR.
2. **Implicit depth recurrence** — each denoising step reuses the same weights. 10 steps with a 10-layer model = 100 effective layers.
3. **Test-time compute scaling** — more denoising steps at eval = tighter ELBO = lower BPB. Free compute scaling.

We use the MDLM (Masked Diffusion Language Model) approach: forward process randomly masks tokens, model predicts original tokens at masked positions.

## Setup

1. **Run tag**: e.g. `mar29-diff`. Branch `experiment/diffusion/<tag>`.
2. **Create branch**: `git checkout -b experiment/diffusion/<tag>`
3. **Read**: this program.md, train.py, ../../README.md, ../../train_gpt_mlx.py
4. **Verify data**.
5. **Initialize results.tsv**.
6. **Go**.

## What You CAN Do

- Modify `train.py` — noise schedule, denoising steps, model architecture, loss weighting, ELBO computation, optimizer, etc.

## What You CANNOT Do

- Modify data loading or eval code outside train.py
- Install extra packages
- Modify this program.md

## Critical: BPB Evaluation

This is the hardest part. Diffusion models define log-likelihoods via an ELBO:

```
log p(x) >= E_q [ sum_t log p(x_0 | x_t, t) * weight(t) ]
```

For MDLM, this simplifies to a weighted sum of cross-entropy losses at each noise level. Use many ELBO samples (500-1000) for a tight bound. The ELBO is a lower bound, so our BPB is pessimistic — but it must be correct.

**The BPB calculation must be tokenizer-agnostic (bits per byte, not bits per token).** Count actual UTF-8 bytes per token using the sentencepiece tokenizer, same as the AR baseline.

## Key Research Questions (priority order)

1. Can MDLM-style diffusion match the AR baseline BPB at same param count?
2. What noise schedule works best? (cosine, linear, learned)
3. How many ELBO samples needed for a tight bound?
4. Does test-time compute scaling work? (more denoise steps = lower BPB?)
5. Can a hybrid AR+diffusion model beat pure AR?

## Important: Start Simple

The diffusion approach is complex. Start with the simplest possible version:
- Uniform masking schedule
- Fixed mask ratio per step (not continuous time)
- Simple cross-entropy loss on masked positions
- Few ELBO steps at eval (100 to start)

Get this working and measuring BPB correctly first. Then iterate.

## First Run

Run `train.py` as-is to get baseline val_bpb with the MDLM architecture.

## Experiment Loop

LOOP FOREVER:

1. Git state check
2. Modify `train.py`
3. `git commit`
4. Run: `python3 train.py > run.log 2>&1`
5. Extract: `grep "^val_bpb:" run.log`
6. Crash? → fix or skip
7. Log to results.tsv
8. Improved → keep. Worse → revert.

**NEVER STOP**.

## Logging

Tab-separated `results.tsv`:
```
commit	val_bpb	status	description
```

## Ideas to Try

- Baseline: 10-layer bidirectional transformer, MDLM training, 100-step ELBO eval
- Noise schedule: cosine (proven in image diffusion), linear, sqrt
- Loss weighting: uniform, importance-weighted (focus on low-noise steps)
- ELBO steps at eval: 100, 500, 1000, 2000
- Continuous-time diffusion (SEDD-style exact log-likelihood)
- Self-conditioning: feed the model's previous prediction as extra input
- Classifier-free guidance at eval (unconditional + conditional denoising)
- Absorbing state diffusion (D3PM): tokens transition to [MASK] state
- Hybrid: small AR model provides initial predictions, diffusion refines
- Bidirectional attention gives better gradients — try larger learning rates
- The model sees all positions — might not need positional embeddings at all?
- Iterative refinement: at eval, run multiple rounds of denoise+remask
