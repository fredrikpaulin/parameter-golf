#!/usr/bin/env python3
"""
Parameter Golf — Discrete Diffusion LM (MLX)
Absorbing-state diffusion with cross-entropy training + SEDD insights.
Key ideas from SEDD paper: log-linear noise schedule, proper ELBO weighting.
Bidirectional transformer backbone — no causal mask.
"""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# CONFIG
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16
TIME_BUDGET = float(os.environ.get("TIME_BUDGET", 300.0))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024")
TOKENIZER_PATH = str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model")

VOCAB_SIZE = 1024
MASK_TOKEN_ID = VOCAB_SIZE
NUM_LAYERS = 6  # Unique layers
NUM_LOOPS = 1   # Loop through layers this many times (effective depth = NUM_LAYERS * NUM_LOOPS)
MODEL_DIM = 384
NUM_HEADS = 6
MLP_MULT = 3
SEQ_LEN = 512
LOGIT_SOFTCAP = 30.0

# Noise schedule: "loglinear" (SEDD-style) or "cosine" (MDLM-style)
NOISE_SCHEDULE = "cosine"
SIGMA_MIN = 1e-4
SIGMA_MAX = 20.0
EVAL_ELBO_STEPS = 64

# Training
TRAIN_BATCH_TOKENS = 4096
WARMUP_STEPS = 50
WARMDOWN_FRAC = 0.15
MAX_ITERATIONS = 500_000
VAL_BATCH_TOKENS = 4096
MATRIX_LR = 0.02
SCALAR_LR = 0.02
EMBED_LR = 0.03
MUON_MOMENTUM = 0.95
MUON_STEPS = 5
GRAD_CLIP = 1.0
CHECKPOINT_EVERY = 100_000  # Save checkpoint every N steps to survive OOM at ~248K
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# ==============================================================================
# CHECKPOINTING
# ==============================================================================

def save_checkpoint(model, split_opt, step, total_training_time, smooth_loss, train_loader):
    """Save model weights, optimizer momentum, and training state."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    # Model weights
    params = dict(tree_flatten(model.parameters()))
    npz_params = {k: np.array(v) for k, v in params.items()}
    np.savez(str(CHECKPOINT_DIR / "model.npz"), **npz_params)
    # Muon momentum buffers
    npz_bufs = {k: np.array(v) for k, v in split_opt.muon_bufs.items()}
    np.savez(str(CHECKPOINT_DIR / "muon_bufs.npz"), **npz_bufs)
    # Training state
    state = {
        "step": step,
        "total_training_time": total_training_time,
        "smooth_loss": smooth_loss,
        "data_file_idx": train_loader.file_idx,
        "data_pos": train_loader.pos,
    }
    with open(CHECKPOINT_DIR / "state.json", "w") as f:
        json.dump(state, f)
    print(f"  [checkpoint saved at step {step}, time={total_training_time:.0f}s]")


def load_checkpoint(model, split_opt, train_loader):
    """Load checkpoint if it exists. Returns (step, total_training_time, smooth_loss) or None."""
    state_path = CHECKPOINT_DIR / "state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        state = json.load(f)
    # Model weights
    data = np.load(str(CHECKPOINT_DIR / "model.npz"))
    params = {k: mx.array(data[k]) for k in data.files}
    model.update(tree_unflatten(list(params.items())))
    mx.eval(model.parameters())
    # Muon momentum buffers
    buf_data = np.load(str(CHECKPOINT_DIR / "muon_bufs.npz"))
    for k in buf_data.files:
        if k in split_opt.muon_bufs:
            split_opt.muon_bufs[k] = mx.array(buf_data[k])
    mx.eval(*split_opt.muon_bufs.values())
    # Restore data loader position
    train_loader.file_idx = state["data_file_idx"]
    train_loader.tokens = load_data_shard(train_loader.files[train_loader.file_idx])
    train_loader.pos = state["data_pos"]
    print(f"  [checkpoint loaded: step={state['step']}, time={state['total_training_time']:.0f}s]")
    return state["step"], state["total_training_time"], state["smooth_loss"]


def clear_checkpoint():
    """Remove checkpoint after successful completion."""
    if CHECKPOINT_DIR.exists():
        for f in CHECKPOINT_DIR.iterdir():
            f.unlink()
        CHECKPOINT_DIR.rmdir()
        print("  [checkpoint cleared]")


# ==============================================================================
# HELPERS
# ==============================================================================

def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def hadamard_transform(x):
    """Fast Walsh-Hadamard transform on the last dimension.
    Spreads energy evenly across dimensions, flattening outliers.
    x shape: (..., d) where d must be a power of 2.
    Returns: rotated x, normalized by 1/sqrt(d)."""
    d = x.shape[-1]
    orig_shape = x.shape
    # Flatten leading dims: (*, d)
    x = x.reshape(-1, d)
    n = x.shape[0]
    h = 1
    while h < d:
        # Reshape to (n, d/(2h), 2, h) for butterfly
        x = x.reshape(n, d // (2 * h), 2, h)
        a = x[:, :, 0, :] + x[:, :, 1, :]
        b = x[:, :, 0, :] - x[:, :, 1, :]
        x = mx.concatenate([a[:, :, None, :], b[:, :, None, :]], axis=2)
        x = x.reshape(n, d)
        h *= 2
    return (x * (1.0 / math.sqrt(d))).reshape(orig_shape)


def zeropower_newtonschulz5(g, steps=5, eps=1e-7):
    """Orthogonalize a 2D matrix with Newton-Schulz iteration (Muon optimizer)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


class SplitOptimizer:
    """Muon for 2D matrices, Adam for scalars/embeddings."""
    def __init__(self, model):
        params = dict(tree_flatten(model.parameters()))
        self.matrix_keys = [k for k, p in params.items() if p.ndim == 2 and 'embed' not in k and 'time' not in k]
        self.embed_keys = [k for k, p in params.items() if 'embed' in k and p.ndim == 2]
        self.scalar_keys = [k for k, p in params.items() if k not in self.matrix_keys and k not in self.embed_keys]
        # Muon momentum buffers
        self.muon_bufs = {k: mx.zeros_like(params[k]) for k in self.matrix_keys}
        # Adam for scalars
        self.adam_scalar = optim.Adam(learning_rate=SCALAR_LR, betas=[0.9, 0.95])
        self.adam_embed = optim.Adam(learning_rate=EMBED_LR, betas=[0.9, 0.95])

    def step(self, model, grads_tree, step_num, lr_mul):
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        # Muon for matrix params
        lr = MATRIX_LR * lr_mul
        for k in self.matrix_keys:
            if k not in grads:
                continue
            g = grads[k]
            buf = MUON_MOMENTUM * self.muon_bufs[k] + g
            self.muon_bufs[k] = buf
            g_eff = g + MUON_MOMENTUM * buf
            g_ortho = zeropower_newtonschulz5(g_eff, MUON_STEPS)
            scale = math.sqrt(max(1.0, float(params[k].shape[0]) / float(params[k].shape[1])))
            updated[k] = params[k] - lr * (g_ortho * scale).astype(params[k].dtype)
        # Adam for embeddings
        self.adam_embed.learning_rate = EMBED_LR * lr_mul
        embed_g = {k: grads[k] for k in self.embed_keys if k in grads}
        embed_p = {k: params[k] for k in self.embed_keys if k in grads}
        if embed_g:
            embed_upd = self.adam_embed.apply_gradients(embed_g, embed_p)
            updated.update(embed_upd)
        # Adam for scalars
        self.adam_scalar.learning_rate = SCALAR_LR * lr_mul
        scalar_g = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_p = {k: params[k] for k in self.scalar_keys if k in grads}
        if scalar_g:
            scalar_upd = self.adam_scalar.apply_gradients(scalar_g, scalar_p)
            updated.update(scalar_upd)
        model.update(tree_unflatten(list(updated.items())))
        # Evaluate momentum buffers to prevent MLX computation graph accumulation
        mx.eval(*self.muon_bufs.values())


def load_data_shard(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size < 2 or int(header[0]) != 20240520:
        raise ValueError(f"Bad shard header: {path}")
    ntok = int(header[2])
    return np.fromfile(path, dtype="<u2", offset=256 * 4, count=ntok)


class DataLoader:
    def __init__(self, pattern, batch_tokens, seq_len):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No shards matching {pattern}")
        self.batch_tokens = batch_tokens
        self.seq_len = seq_len
        self.batch_size = batch_tokens // seq_len
        self.file_idx = 0
        self.pos = 0
        self.tokens = load_data_shard(self.files[0])

    def reset(self):
        self.file_idx = 0
        self.pos = 0
        self.tokens = load_data_shard(self.files[0])

    def next_batch(self):
        needed = self.batch_size * self.seq_len + 1
        while self.pos + needed > len(self.tokens):
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.tokens = load_data_shard(self.files[self.file_idx])
            self.pos = 0
        chunk = self.tokens[self.pos:self.pos + needed].astype(np.int32)
        self.pos += self.batch_size * self.seq_len
        buf = mx.array(chunk)
        x = buf[:self.batch_size * self.seq_len].reshape(self.batch_size, self.seq_len)
        return x


def build_bytes_per_token(sp):
    vocab_size = sp.vocab_size()
    bpt = np.ones(vocab_size + 1, dtype=np.int32)
    for tok_id in range(vocab_size):
        if sp.is_control(tok_id) or sp.is_unknown(tok_id) or sp.is_unused(tok_id):
            continue
        if sp.is_byte(tok_id):
            bpt[tok_id] = 1
            continue
        piece = sp.id_to_piece(tok_id)
        if piece.startswith("▁"):
            piece = " " + piece[1:]
        bpt[tok_id] = max(len(piece.encode("utf-8")), 1)
    return bpt


# ==============================================================================
# NOISE SCHEDULE
# ==============================================================================
# Two options:
# 1) Log-linear (SEDD): sigma_bar(t) = sigma_min * (sigma_max/sigma_min)^t
#    alpha(t) = 1 - exp(-sigma_bar(t))
#    sigma_rate(t) = sigma_bar(t) * log(sigma_max/sigma_min)  [nearly constant-ish]
#
# 2) Cosine (MDLM): alpha(t) = 1 - cos²((t+s)/(1+s) * π/2) / cos²(s/(1+s) * π/2)
# ==============================================================================

_LOG_RATIO = math.log(SIGMA_MAX / SIGMA_MIN)

def get_sigma_bar(t):
    """Cumulative noise sigma_bar(t)."""
    if NOISE_SCHEDULE == "loglinear":
        return SIGMA_MIN * math.exp(_LOG_RATIO * t)
    else:  # cosine
        s = 0.008
        f_t = math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        alpha = max(1e-7, min(1.0 - 1e-7, 1.0 - f_t / f_0))
        return -math.log(1.0 - alpha)

def get_mask_prob(t):
    """Probability that a token is masked at time t."""
    return 1.0 - math.exp(-get_sigma_bar(t))

def get_sigma_rate(t, dt=1e-5):
    """Forward rate d(sigma_bar)/dt."""
    if NOISE_SCHEDULE == "loglinear":
        return get_sigma_bar(t) * _LOG_RATIO
    else:
        return (get_sigma_bar(min(t + dt, 1.0)) - get_sigma_bar(max(t - dt, 0.0))) / (2 * dt)

def get_dalpha_dt(t, dt=1e-5):
    """d(alpha)/dt for ELBO weighting."""
    a_hi = get_mask_prob(min(t + dt, 1.0))
    a_lo = get_mask_prob(max(t - dt, 0.0))
    return (a_hi - a_lo) / (2 * dt)


# ==============================================================================
# MODEL — Bidirectional Transformer
# ==============================================================================

def _build_rope_freqs(seq_len, head_dim, base=10000.0):
    """Precompute RoPE cos/sin for seq_len positions."""
    freqs = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    t = mx.arange(seq_len).astype(mx.float32)
    angles = t[:, None] * freqs[None, :]  # (T, head_dim/2)
    cos_vals = mx.cos(angles)  # (T, hd/2)
    sin_vals = mx.sin(angles)  # (T, hd/2)
    return cos_vals, sin_vals


def _apply_rope(x, cos_vals, sin_vals):
    """Apply rotary embeddings to x of shape (B, heads, T, hd)."""
    T = x.shape[2]
    cos_v = cos_vals[:T]  # (T, hd/2)
    sin_v = sin_vals[:T]  # (T, hd/2)
    # Split x into two halves
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices
    # Rotate
    o1 = x1 * cos_v - x2 * sin_v
    o2 = x1 * sin_v + x2 * cos_v
    # Interleave back
    return mx.concatenate([o1[..., None], o2[..., None]], axis=-1).reshape(x.shape)


class BidirectionalAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, rope_cos, rope_sin):
        B, T, _ = x.shape
        hd = self.head_dim
        q = self.c_q(x).reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)

        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        # QK-norm: normalize Q and K per-head for stable, direction-only attention
        q = rms_norm(q)
        k = rms_norm(k)

        attn = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(hd))
        attn = mx.softmax(attn, axis=-1).astype(v.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(out)


class DiffusionBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult):
        super().__init__()
        self.attn = BidirectionalAttention(dim, n_heads)
        self.fc = nn.Linear(dim, dim * mlp_mult, bias=False)
        self.proj = nn.Linear(dim * mlp_mult, dim, bias=False)

    def __call__(self, x, rope_cos, rope_sin):
        x = x + self.attn(rms_norm(x), rope_cos, rope_sin)
        h = self.fc(rms_norm(x))
        h = nn.gelu(h)
        x = x + self.proj(h)
        return x


class DiffusionLM(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, mlp_mult, seq_len):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, dim)  # +1 for [MASK]
        self.blocks = [DiffusionBlock(dim, n_heads, mlp_mult) for _ in range(n_layers)]
        self.out_head = nn.Linear(dim, vocab_size, bias=True)
        self.logit_softcap = LOGIT_SOFTCAP
        # RoPE precomputed
        self._rope_cos, self._rope_sin = _build_rope_freqs(seq_len, dim // n_heads)
        # t conditioning: scalar t → dim
        self.t_embed = nn.Linear(1, dim, bias=True)

    def __call__(self, x_noised, t_value, mask=None):
        B, T = x_noised.shape
        h = self.embed(x_noised).astype(COMPUTE_DTYPE)

        # Condition on noise level
        t_scalar = mx.array([[t_value]]).astype(COMPUTE_DTYPE)
        t_bias = self.t_embed(t_scalar)  # (1, 1, dim)
        h = h + t_bias

        rope_cos = self._rope_cos[:T].astype(COMPUTE_DTYPE)
        rope_sin = self._rope_sin[:T].astype(COMPUTE_DTYPE)

        for _ in range(NUM_LOOPS):
            for block in self.blocks:
                h = block(h, rope_cos, rope_sin)
        h = rms_norm(h)

        logits = self.out_head(h)
        if self.logit_softcap > 0:
            logits = self.logit_softcap * mx.tanh(logits / self.logit_softcap)
        return logits


_CURRICULUM_PROGRESS = 0.0  # Set by training loop
_GLOBAL_STEP = 0  # For stratified t sampling

# ==============================================================================
# TRAINING LOSS — Variable t, masked-only CE
# ==============================================================================

_N_STRATA = 8  # Divide [0.1, 0.5] into 8 strata of width 0.05

def diffusion_loss(model, tokens):
    B, T = tokens.shape

    # Stratified t sampling: cycle through strata for even coverage,
    # sample uniformly within each stratum. Reduces gradient variance
    # from ELBO weighting without any bias or extra compute.
    stratum = _GLOBAL_STEP % _N_STRATA
    t_lo = 0.1 + stratum * 0.05
    t_hi = t_lo + 0.05
    t_val = float(mx.random.uniform(low=t_lo, high=t_hi, shape=()))
    mask_prob = max(get_mask_prob(t_val), 0.01)
    dalpha = get_dalpha_dt(t_val)

    mask = mx.random.uniform(shape=(B, T)) < mask_prob
    masked_tokens = mx.where(mask, MASK_TOKEN_ID, tokens)

    logits = model(masked_tokens, t_val, mask=mask)
    logits_flat = logits.reshape(-1, VOCAB_SIZE).astype(mx.float32)
    targets_flat = tokens.reshape(-1)

    # CE at masked positions, weighted by ELBO importance (dalpha / mask_prob)
    # This makes the loss an unbiased estimate of the ELBO across t values
    per_tok = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
    mask_flat = mask.reshape(-1).astype(mx.float32)
    avg_ce = mx.sum(per_tok * mask_flat) / mx.maximum(mx.sum(mask_flat), mx.array(1.0))
    loss = avg_ce * (dalpha / mask_prob)
    return loss


# ==============================================================================
# ELBO EVALUATION
# ==============================================================================
# Absorbing-state ELBO (MDLM Eq. 12):
#   -log p(x) <= integral_0^1 (alpha'(t)/alpha(t)) * E[ sum_{i:masked} -log p(x_i|x_t,t) ] dt
# Using midpoint quadrature and simplifying:
#   ELBO_per_token = sum_k delta_alpha_k * avg_CE_masked(t_k)
# ==============================================================================

def compute_elbo_bpb(model, val_loader, bytes_per_token, num_steps, num_batches):
    total_weighted_nll = 0.0
    total_bytes = 0
    n_sequences = 0

    for batch_idx in range(num_batches):
        tokens = val_loader.next_batch()
        B, T = tokens.shape
        n_sequences += B

        token_np = np.array(tokens).reshape(-1)
        total_bytes += int(bytes_per_token[token_np].sum())

        batch_nll = 0.0
        for step_idx in range(num_steps):
            t_mid = (step_idx + 0.5) / num_steps

            t_lo = step_idx / num_steps
            t_hi = (step_idx + 1) / num_steps
            alpha_lo = get_mask_prob(t_lo)
            alpha_hi = get_mask_prob(t_hi)
            delta_alpha = abs(alpha_hi - alpha_lo)
            if delta_alpha < 1e-8:
                continue

            mask_prob = max(get_mask_prob(t_mid), 0.005)
            mask = mx.random.uniform(shape=(B, T)) < mask_prob
            masked_tokens = mx.where(mask, MASK_TOKEN_ID, tokens)
            logits = model(masked_tokens, t_mid, mask=mask)

            logits_flat = logits.reshape(-1, VOCAB_SIZE).astype(mx.float32)
            targets_flat = tokens.reshape(-1)
            per_token_nll = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
            mask_flat = mask.reshape(-1).astype(mx.float32)
            mx.eval(per_token_nll, mask_flat)

            # Avg CE at masked positions only (correct ELBO for absorbing state)
            n_masked = float(mx.sum(mask_flat))
            if n_masked > 0:
                avg_masked_nll = float(mx.sum(per_token_nll * mask_flat)) / n_masked
            else:
                avg_masked_nll = 0.0

            # ELBO per-token contribution: delta_alpha * avg_masked_NLL
            batch_nll += avg_masked_nll * delta_alpha

        # batch_nll is now per-token ELBO for this batch's sequences
        # Accumulate as total ELBO * num_tokens_in_batch
        total_weighted_nll += batch_nll * B * T

        if (batch_idx + 1) % 4 == 0:
            print(f"  ELBO eval {batch_idx+1}/{num_batches}")

    total_tokens = n_sequences * T
    avg_nll_nats = total_weighted_nll / total_tokens
    bpb = avg_nll_nats / math.log(2) * (total_tokens / total_bytes)
    return bpb


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    t_start = time.time()
    mx.random.seed(42)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    bytes_per_token = build_bytes_per_token(sp)

    batch_size = TRAIN_BATCH_TOKENS // SEQ_LEN
    print(f"DiffusionLM: {NUM_LAYERS}L dim={MODEL_DIM} heads={NUM_HEADS} "
          f"mlp={MLP_MULT}x seq={SEQ_LEN} batch={batch_size}")
    print(f"Schedule: {NOISE_SCHEDULE} | ELBO steps: {EVAL_ELBO_STEPS} | Muon LR: {MATRIX_LR}")
    if NOISE_SCHEDULE == "loglinear":
        print(f"sigma_min={SIGMA_MIN} sigma_max={SIGMA_MAX}")
    print(f"Time budget: {TIME_BUDGET}s")

    # Print schedule diagnostics
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tv = max(min(t_val, 0.999), 0.001)
        print(f"  t={t_val:.2f}: mask_prob={get_mask_prob(tv):.4f} "
              f"sigma_bar={get_sigma_bar(tv):.4f} "
              f"dalpha_dt={get_dalpha_dt(tv):.4f}")

    model = DiffusionLM(VOCAB_SIZE, MODEL_DIM, NUM_LAYERS, NUM_HEADS, MLP_MULT, SEQ_LEN)

    flat = tree_flatten(model.parameters())
    num_params = sum(p.size for _, p in flat)
    print(f"Parameters: {num_params:,}")

    train_loader = DataLoader(f"{DATA_PATH}/fineweb_train_*.bin", TRAIN_BATCH_TOKENS, SEQ_LEN)
    val_loader = DataLoader(f"{DATA_PATH}/fineweb_val_*.bin", VAL_BATCH_TOKENS, SEQ_LEN)

    split_opt = SplitOptimizer(model)
    loss_and_grad = nn.value_and_grad(model, diffusion_loss)

    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0

    # Resume from checkpoint if available
    ckpt = load_checkpoint(model, split_opt, train_loader)
    if ckpt is not None:
        step, total_training_time, smooth_loss = ckpt

    while step < MAX_ITERATIONS:
        t0 = time.time()

        tokens = train_loader.next_batch()
        loss_val, grads = loss_and_grad(model, tokens)
        mx.eval(loss_val, grads)

        # Gradient clipping
        grad_pairs = tree_flatten(grads)
        grad_norm_sq = sum(float(mx.sum(g * g)) for _, g in grad_pairs)
        grad_norm = grad_norm_sq ** 0.5
        if grad_norm > GRAD_CLIP:
            scale = GRAD_CLIP / grad_norm
            grads = tree_unflatten([(k, g * scale) for k, g in grad_pairs])

        # LR schedule with warmup + cosine warmdown
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        global _CURRICULUM_PROGRESS, _GLOBAL_STEP
        _CURRICULUM_PROGRESS = progress
        _GLOBAL_STEP = step
        if progress > (1.0 - WARMDOWN_FRAC):
            lr_mul = (1.0 - progress) / WARMDOWN_FRAC
        elif step < WARMUP_STEPS:
            lr_mul = (step + 1) / WARMUP_STEPS
        else:
            lr_mul = 1.0
        lr_mul = max(lr_mul, 0.01)

        split_opt.step(model, grads, step, lr_mul)
        mx.eval(model.parameters())

        dt = time.time() - t0
        if step > 5:
            total_training_time += dt

        lv = float(loss_val)
        smooth_loss = 0.9 * smooth_loss + 0.1 * lv
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        remaining = max(0, TIME_BUDGET - total_training_time)

        if step % 20 == 0:
            tok_s = TRAIN_BATCH_TOKENS / max(dt, 1e-6)
            print(f"step {step:05d} ({100*progress:.1f}%) | loss: {debiased:.4f} | "
                  f"gnorm: {grad_norm:.4f} | dt: {dt*1000:.0f}ms | tok/s: {tok_s:,.0f} | "
                  f"remaining: {remaining:.0f}s")

        step += 1
        if step % 5000 == 0:
            gc.collect()
        if step % CHECKPOINT_EVERY == 0 and step > 0:
            save_checkpoint(model, split_opt, step, total_training_time, smooth_loss, train_loader)
        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    # Diagnostic: CE at specific t values
    print("\n--- Per-t diagnostics ---")
    val_loader.reset()
    diag_tokens = val_loader.next_batch()
    per_t_ces = {}
    for t_diag in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        mp = get_mask_prob(t_diag)
        dmask = mx.random.uniform(shape=diag_tokens.shape) < mp
        masked = mx.where(dmask, MASK_TOKEN_ID, diag_tokens)
        dlogits = model(masked, t_diag, mask=dmask)
        dlogits_f = dlogits.reshape(-1, VOCAB_SIZE).astype(mx.float32)
        dtargets = diag_tokens.reshape(-1)
        dce = nn.losses.cross_entropy(dlogits_f, dtargets, reduction="none")
        dmask_f = dmask.reshape(-1).astype(mx.float32)
        mx.eval(dce, dmask_f)
        n_m = float(mx.sum(dmask_f))
        avg_ce = float(mx.sum(dce * dmask_f)) / n_m if n_m > 0 else 0.0
        per_t_ces[t_diag] = avg_ce
        print(f"  t={t_diag:.2f}: mask_prob={mp:.3f}, n_masked={int(n_m)}, avg_CE_masked={avg_ce:.4f}")

    print(f"\nELBO eval ({EVAL_ELBO_STEPS} levels x 8 batches)...")
    val_loader.reset()
    val_bpb = compute_elbo_bpb(model, val_loader, bytes_per_token,
                                num_steps=EVAL_ELBO_STEPS, num_batches=8)

    clear_checkpoint()  # Clean up after successful completion
    t_end = time.time()
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"model_dim:        {MODEL_DIM}")
    print(f"num_layers:       {NUM_LAYERS}")
    print(f"noise_schedule:   {NOISE_SCHEDULE}")
    print(f"elbo_steps:       {EVAL_ELBO_STEPS}")

    # === COPY-PASTE SUMMARY (share this block with Claude) ===
    print("\n" + "=" * 50)
    print("RUN SUMMARY — copy everything below this line")
    print("=" * 50)
    print(f"val_bpb: {val_bpb:.6f}")
    print(f"steps: {step} | time: {total_training_time:.0f}s | params: {num_params/1e6:.1f}M")
    per_t_str = " | ".join(f"t{t}={ce:.2f}" for t, ce in sorted(per_t_ces.items()))
    print(f"per-t: {per_t_str}")
    print(f"final_smooth_loss: {debiased:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
