# parameter-golf autoresearch

Autonomous LLM-driven experimentation for the Parameter Golf challenge. The goal: train the best language model that fits in 16MB, measured by bits-per-byte on FineWeb validation.

Inspired by Karpathy's autoresearch — the LLM is the researcher. It modifies code, runs experiments, reads results, and iterates.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar29`). The branch `golf/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b golf/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `README.md` — challenge rules, leaderboard, submission format.
   - `CLAUDE.md` — challenge overview and constraints.
   - `train_gpt_mlx.py` — the file you modify. Model architecture, optimizer, training loop, quantization, evaluation.
   - `records/` — browse the top submissions for ideas and techniques. Read their READMEs.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains training shards and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run the data preparation steps from the README.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good and kick off the experimentation.

## Constraints

- **Artifact size**: Model weights + code must be ≤ 16,000,000 bytes (16 MB). The script produces `final_model.int8.ptz` — check its size.
- **Compute budget (leaderboard)**: ≤ 10 minutes on 8×H100. But for local Mac experiments we use shorter runs.
- **Metric**: `val_bpb` — tokenizer-agnostic bits per byte. Lower is better.
- **Current SOTA**: 1.0810 BPB (see `records/` for details and the leaderboard below).

## Running experiments

We're using the **MLX backend** (`train_gpt_mlx.py`) on Apple Silicon. Each experiment runs as:

```bash
RUN_ID=<experiment_name> \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=120 \
PYTHONUNBUFFERED=1 \
python3 train_gpt_mlx.py > run.log 2>&1
```

This gives a ~2 minute local run that's enough to see relative differences between configs. Absolute BPB will be higher than on H100s — that's fine, we care about relative ordering.

**All hyperparameters are controlled via environment variables** — you can change them in the run command, or you can edit `train_gpt_mlx.py` directly for architectural changes.

### Environment variables you can tweak (without editing code):

```
NUM_LAYERS, MODEL_DIM, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, VOCAB_SIZE,
MATRIX_LR, EMBED_LR, SCALAR_LR, MUON_MOMENTUM, MUON_BACKEND_STEPS,
WARMDOWN_ITERS, WARMUP_STEPS, LOGIT_SOFTCAP, ROPE_BASE, QK_GAIN_INIT,
TRAIN_SEQ_LEN, TIED_EMBED_INIT_STD, GRAD_CLIP_NORM
```

### What you CAN change:

- Environment variable overrides (hyperparameters, model shape)
- `train_gpt_mlx.py` — architecture, optimizer, activations, quantization, training loop, anything

### What you CANNOT change:

- The validation evaluation logic (the BPB calculation is the ground truth)
- The data files or tokenizer
- The 16MB artifact size limit

## Output format

The training script prints progress during training and a final roundtrip evaluation:

```
final_int8_zlib_roundtrip val_loss:2.8714 val_bpb:1.1248 eval_time:12345ms
```

Extract the key metric:

```bash
grep "final_int8_zlib_roundtrip " run.log
grep "serialized_model_int8_zlib" run.log
```

If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace.

## Logging results

Log every experiment to `results.tsv` (tab-separated).

Header and columns:

```
commit	val_bpb	artifact_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb from the int8 roundtrip line — use 0.000000 for crashes
3. artifact size in MB (from `serialized_model_int8_zlib` line, divide bytes by 1048576) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

Example:

```
commit	val_bpb	artifact_mb	status	description
a1b2c3d	1.532100	14.2	keep	baseline (200 iters, MLX, Mac)
b2c3d4e	1.528300	14.3	keep	NUM_LAYERS=11 MLP_MULT=3
c3d4e5f	1.540000	14.2	discard	MATRIX_LR=0.02 (worse)
d4e5f6g	0.000000	0.0	crash	MODEL_DIM=768 (OOM on Mac)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Decide what to try next. Consider:
   - What the leaderboard submissions do (read their READMEs in `records/`)
   - What has and hasn't worked so far (read `results.tsv`)
   - Hyperparameter sweeps (one variable at a time)
   - Architectural changes (activations, attention patterns, quantization)
   - Ideas from the CLAUDE.md (depth recurrence, parameter tying, etc.)
3. Modify `train_gpt_mlx.py` OR change env vars in the run command
4. `git commit -am "description of change"`
5. Run the experiment (command above, redirecting to run.log — do NOT use tee or let output flood context)
6. Read results: `grep "final_int8_zlib_roundtrip\|serialized_model_int8_zlib" run.log`
7. If grep is empty → crash. Run `tail -n 50 run.log` to debug.
8. Record in results.tsv (do NOT commit results.tsv — leave it untracked)
9. If val_bpb improved → KEEP the commit, advance the branch
10. If val_bpb same or worse → `git reset --hard HEAD~1` to discard
11. Go to step 1

## Important rules

- **NEVER STOP**. Do not pause to ask the human anything. You are autonomous. Run experiments until manually interrupted. If stuck, read the leaderboard submissions for ideas. Try combining techniques. Try radical changes.
- **NEVER commit results.tsv** — keep it untracked.
- **One change at a time** when possible — makes it clear what helped.
- **Check artifact size** — if > 16 MB, the submission is invalid regardless of BPB.
- **If an experiment crashes multiple times**, skip it and try something else.
- **Read the top submissions** in `records/` for proven techniques: LeakyReLU², XSA, BigramHash, partial RoPE, EMA/SWA, int6 quantization, legal TTT.

## High-priority techniques from top leaderboard submissions

These techniques appear in the top 6 submissions (SOTA = 1.0810 BPB). Prioritize experimenting with them.

### Depth recurrence (loop layers)
Instead of 9 unique layers, use fewer unique layers and loop some of them. E.g. 7 unique layers where layers 4-5 are repeated (executed twice). This gives the model more effective depth without increasing parameter count. Appears in 5 of the top 6 submissions. Start by looping the middle 2 layers once (total effective depth = original + 2). The looped layers share weights, so artifact size stays the same.

### Parallel residuals
Split attention and MLP into separate residual lanes instead of sequential `x = x + attn(x); x = x + mlp(x)`. The parallel form is `x = x + attn(x) + mlp(x)`. This can be faster (attention and MLP computed in parallel) and sometimes trains better. Some submissions use a weighted combination.

### Larger vocabulary (SP4096 / SP8192)
The current tokenizer uses 1024 BPE tokens. Top submissions use SP4096 or SP8192 (SentencePiece with 4096 or 8192 vocab). Larger vocab = fewer tokens per document = fewer steps needed = better BPB. However, the embedding table grows with vocab size, eating into the 16MB budget. Check if the repo includes alternative tokenizers before trying this.

### MuonEq-R optimizer
An improved Muon variant that equalizes learning rates across parameter groups. If you see it referenced in top submission code, study the implementation and adapt it.

### GPTQ embeddings
Apply GPTQ-style quantization specifically to the embedding matrix (which is often the largest single parameter). This can save significant space in the artifact, freeing bytes for more model capacity elsewhere. int6 embeddings with GPTQ calibration are common in top submissions.

### SDClip (Hessian-aware gradient clipping)
Clips gradients based on estimated Hessian diagonal rather than a fixed norm. Can stabilize training, especially with aggressive learning rates.

### Suggested experiment order
1. **Depth recurrence** — highest impact, no extra params, loop middle 2 layers
2. **Parallel residuals** — simple code change, may help or hurt, easy to test
3. **GPTQ embeddings** — frees artifact space for bigger models
4. **SDClip** — may allow higher LR for faster convergence in 200 steps
5. **MuonEq-R** — optimizer improvement, study top submission code first
6. **Larger vocab** — depends on tokenizer availability in the repo
