# Parameter Golf Agent Loop Architecture

An autonomous experiment system for iterating toward lower BPB on the Parameter Golf challenge. The core idea: treat each training run as a function from config → score, and build a loop that proposes configs, runs them, parses results, and decides what to try next.

## Constraints Recap

- **Artifact**: ≤ 16 MB (model weights + code)
- **Compute**: ≤ 10 min on 8×H100 SXM
- **Metric**: bits-per-byte on FineWeb validation (tokenizer-agnostic)
- **Current SOTA**: 1.1194 BPB (LeakyReLU² + Legal TTT + Parallel Muon)

## System Overview

```
┌─────────────────────────────────────────────────┐
│                  ORCHESTRATOR                    │
│  (outer loop — runs on CPU, drives everything)  │
├─────────────────────────────────────────────────┤
│                                                 │
│   ┌──────────┐   ┌──────────┐   ┌───────────┐  │
│   │ PROPOSER │──▸│ EXECUTOR │──▸│  ANALYZER  │  │
│   └──────────┘   └──────────┘   └───────────┘  │
│        ▲                              │         │
│        └──────────────────────────────┘         │
│                 feedback loop                   │
│                                                 │
│   ┌──────────────────────────────────────────┐  │
│   │              RESULT STORE                │  │
│   │  (JSON log of every run: config → score) │  │
│   └──────────────────────────────────────────┘  │
│                                                 │
│   ┌──────────────────────────────────────────┐  │
│   │            STRATEGY BANK                 │  │
│   │  (ranked list of untried ideas + priors) │  │
│   └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Components

### 1. Orchestrator (`agent/orchestrator.js`)

The top-level loop. Keeps running until a time or iteration budget is exhausted. Each cycle:

```
while (budget_remaining()) {
  const proposal = proposer.next(result_store, strategy_bank)
  const run_result = executor.run(proposal)
  const analysis = analyzer.evaluate(run_result, result_store)
  result_store.append({ proposal, run_result, analysis })
  strategy_bank.update(analysis)
}
```

State is persisted to disk as plain JSON after every cycle so crashes are recoverable. No database, no framework — just files.

### 2. Proposer (`agent/proposer.js`)

Decides what to try next. Operates in three modes, cycling based on context:

**Mode A — Hyperparameter sweep.** Takes a base config and varies one or two env vars at a time. Uses the result store to pick the most promising axis. Simple grid or random search within bounds.

**Mode B — Ablation.** When a run improves the score, the proposer generates ablation configs that remove or isolate each change to measure individual contribution.

**Mode C — Architecture mutation.** Proposes structural changes to the training script itself: swapping activations, adding/removing layers, changing quantization scheme, toggling techniques like EMA/SWA/TTT. These are expressed as patches (diff hunks) applied to a base script.

The proposer maintains a priority queue ranked by expected improvement. Priority is informed by:
- How much headroom a technique showed in leaderboard submissions
- Whether an axis has been explored yet
- Diminishing returns on axes already swept

```json
{
  "type": "hyperparam_sweep",
  "base": "train_gpt.py",
  "changes": {
    "NUM_LAYERS": 11,
    "MLP_MULT": 3,
    "MATRIX_LR": 0.035
  },
  "rationale": "11L+3xMLP is the foundation of top-5 submissions, matrix_lr 0.035 interpolates between baseline 0.04 and best 0.03"
}
```

### 3. Executor (`agent/executor.js`)

Takes a proposal and actually runs training. Responsibilities:

- Writes a modified `train_gpt.py` if the proposal includes patches
- Sets env vars from the proposal config
- Launches the training run via `torchrun` (or local single-GPU for fast iteration)
- Enforces a per-run wallclock timeout (configurable, default 10 min)
- Captures stdout/stderr to a log file
- On completion, extracts the final artifact size and round-trip BPB from the log

For fast local iteration (no 8×H100 cluster), the executor supports a **reduced mode**: shorter training (60s wallclock), smaller batch, single GPU. Scores won't match leaderboard but relative ordering between configs is preserved — good enough for ranking proposals.

```json
{
  "schema": "run_result",
  "proposal_id": "sweep_017",
  "seed": 1337,
  "val_bpb": 1.1248,
  "val_loss": 2.8714,
  "artifact_bytes": 15876510,
  "wallclock_s": 598.2,
  "steps": 7182,
  "ms_per_step": 83.4,
  "log_file": "runs/sweep_017_seed1337.log"
}
```

### 4. Analyzer (`agent/analyzer.js`)

Parses run results and produces actionable signals:

- **Delta tracking**: compares new BPB against the current best and the proposal's baseline
- **Artifact budget check**: flags if artifact > 16 MB (run is invalid)
- **Wallclock check**: flags if training exceeded 600s
- **Regression detection**: alerts if a change worsened BPB by more than noise (> 0.002)
- **Improvement significance**: requires ≥ 0.005 improvement across 3 seeds to flag as real progress (mirrors leaderboard rules)
- **Gradient signal**: for sweeps, estimates the partial derivative of BPB w.r.t. each swept variable

Output is a structured verdict:

```json
{
  "proposal_id": "sweep_017",
  "verdict": "improvement",
  "delta_bpb": -0.0032,
  "significant": false,
  "needs_more_seeds": true,
  "artifact_ok": true,
  "wallclock_ok": true,
  "next_suggestions": [
    "run seeds 42 and 2025 to confirm",
    "try MATRIX_LR in [0.03, 0.032] — gradient points lower"
  ]
}
```

### 5. Result Store (`agent/store.json`)

Flat JSON array of every run. No indexing needed at this scale (hundreds of runs, not millions). Queryable by simple filters.

Schema per entry:

```json
{
  "id": "sweep_017_seed1337",
  "timestamp": "2026-03-29T14:22:00Z",
  "proposal": { "...proposal object..." },
  "result": { "...run_result object..." },
  "analysis": { "...analyzer verdict..." }
}
```

### 6. Strategy Bank (`agent/strategies.json`)

A ranked list of ideas to try, seeded from analysis of the 26 existing leaderboard submissions. Each strategy has a status and a prior estimate of its impact.

```json
[
  {
    "id": "leaky_relu_sweep",
    "description": "Sweep LeakyReLU negative_slope in [0.3, 0.7]",
    "category": "activation",
    "estimated_impact": -0.003,
    "status": "untried",
    "source": "top submission uses 0.5, may not be optimal"
  },
  {
    "id": "depth_recurrence",
    "description": "Share weights across blocks 3-6 (loop 2x), freeing params for wider MLP",
    "category": "architecture",
    "estimated_impact": -0.005,
    "status": "untried",
    "source": "CLAUDE.md mentions depth recurrence as unexplored direction"
  }
]
```

## Search Dimensions

The agent explores these axes, roughly ordered by expected payoff and ease of testing:

### Tier 1 — Low-risk, high-signal (sweep first)

| Axis | Range | Granularity |
|------|-------|-------------|
| `NUM_LAYERS` | 9–13 | 1 |
| `MLP_MULT` | 2–4 | 1 |
| `MODEL_DIM` | 512, 640, 768 | — |
| `MATRIX_LR` | 0.02–0.06 | 0.005 |
| `EMBED_LR` | 0.3–0.9 | 0.1 |
| `MUON_MOMENTUM` | 0.90–0.97 | 0.01 |
| `WARMDOWN_ITERS` | 800–2000 | 200 |
| LeakyReLU slope | 0.3–0.7 | 0.1 |

### Tier 2 — Structural changes (need script patches)

- Quantization scheme: int8 vs int6 vs int5 vs ternary (tradeoff: more params at lower precision vs fewer at higher)
- EMA decay and SWA interval tuning
- Partial RoPE dimension count
- XSA layer range and head count
- BigramHash vocab expansion size
- U-Net skip connections vs flat

### Tier 3 — Speculative / high-effort

- Depth recurrence (weight sharing across blocks)
- Low-rank factored layers (LoRA-style training)
- Mixture of experts with shared experts
- Novel tokenizer (larger BPE vocab to reduce sequence length)
- Test-time training protocol tuning (chunk size, LR, epochs)
- Knowledge distillation from a larger teacher model

## Execution Strategy

### Phase 1: Establish a local baseline (1–2 hours)

Run the current baseline `train_gpt.py` with default params on available hardware. Even a single GPU with 60s cap gives a relative reference point. Record it in the store.

### Phase 2: Sweep Tier 1 hyperparams (4–8 hours)

Each sweep run takes ~2 min on a single GPU in reduced mode. With ~50 configs across Tier 1, this is tractable. The analyzer identifies the best config per axis, then the proposer combines top settings into a merged config.

### Phase 3: Ablation and confirmation (2–4 hours)

Run the merged config 3 seeds at full 10-min budget (or reduced-mode equivalent). Ablate each change to verify it contributes independently.

### Phase 4: Tier 2 structural experiments (days)

Each structural change requires modifying the training script. The proposer generates diffs; the executor applies them. One change at a time, measured against the Phase 3 baseline.

### Phase 5: Tier 3 speculative bets (ongoing)

Longer-shot ideas explored when Tier 1+2 headroom is exhausted.

## File Layout

```
agent/
  orchestrator.js      — main loop
  proposer.js          — generates experiment proposals
  executor.js          — runs training, captures results
  analyzer.js          — evaluates runs, produces verdicts
  store.json           — persistent log of all runs
  strategies.json      — ranked idea bank
  config_schema.json   — JSON Schema for proposals and results
  patches/             — saved script diffs for structural experiments
  runs/                — per-run logs and artifacts
```

## Config Schema (JSON Schema)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "id": { "type": "string" },
    "type": { "enum": ["hyperparam_sweep", "ablation", "architecture_mutation"] },
    "base_script": { "type": "string", "default": "train_gpt.py" },
    "env_overrides": {
      "type": "object",
      "additionalProperties": { "type": ["string", "number"] }
    },
    "patch_file": { "type": ["string", "null"], "default": null },
    "seeds": {
      "type": "array",
      "items": { "type": "integer" },
      "default": [1337]
    },
    "wallclock_cap_s": { "type": "number", "default": 600 },
    "rationale": { "type": "string" }
  },
  "required": ["id", "type"]
}
```

## Run Result Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "proposal_id": { "type": "string" },
    "seed": { "type": "integer" },
    "val_bpb": { "type": "number" },
    "val_loss": { "type": "number" },
    "artifact_bytes": { "type": "integer" },
    "wallclock_s": { "type": "number" },
    "steps": { "type": "integer" },
    "ms_per_step": { "type": "number" },
    "log_file": { "type": "string" },
    "exit_code": { "type": "integer" }
  },
  "required": ["proposal_id", "seed", "val_bpb", "artifact_bytes", "wallclock_s"]
}
```

## Key Design Decisions

**Why not Bayesian optimization?** With ~50 dimensions and expensive evaluations, BO's surrogate model struggles. A structured sweep + ablation approach is more interpretable and the proposer can use domain knowledge from the leaderboard to prioritize. If you want to add BO later, the result store has the right shape for it — just fit a GP on the (config, bpb) pairs.

**Why JSON files instead of a database?** At the scale of this challenge (hundreds of runs, not millions), flat JSON is simpler to debug, version-control, and inspect. The store is append-only with occasional rewrites for compaction.

**Why patches for architecture changes?** The training script is ~1100 lines. Maintaining N forked copies is a nightmare. Patches keep a single base script and express each experiment as a minimal diff. The executor applies the patch to a temp copy, runs it, and discards the copy.

**Why single-GPU reduced mode?** Full 8×H100 runs cost ~$3/run at spot pricing. For hyperparameter sweeps where we need relative ordering (not absolute BPB), a 60s single-GPU run gives the same ranking at 1/50th the cost. Full runs are reserved for confirmation and submission.

## Getting Started

```bash
# Seed the strategy bank from leaderboard analysis
bun agent/orchestrator.js --init

# Run the agent loop (will keep going until stopped)
bun agent/orchestrator.js --budget-hours 8 --gpu-mode reduced

# Run a specific proposal manually
bun agent/executor.js --proposal sweep_017 --seed 1337

# Show best results so far
bun agent/analyzer.js --report
```

## Interaction with Existing Codebase

The agent does not modify the baseline `train_gpt.py` or any files in `records/`. It operates entirely within the `agent/` directory. When a config beats SOTA across 3 seeds on full hardware, the orchestrator generates a submission-ready package (README, submission.json, training script, logs) in `agent/submissions/` for manual review before copying to `records/`.
