#!/usr/bin/env python3
"""
Parameter Golf — Discrete Diffusion LM (PyTorch/CUDA, DDP)
Best recipe port for multi-GPU H100 training.

Usage:
    # Single GPU
    TIME_BUDGET=600 python3 -u train_cuda.py

    # Multi-GPU via torchrun
    TIME_BUDGET=600 torchrun --standalone --nproc_per_node=8 train_cuda.py

    # Override batch tokens per GPU
    BATCH_TOKENS=32768 TIME_BUDGET=600 torchrun --standalone --nproc_per_node=8 train_cuda.py

Data layout (relative to repo root):
    data/tokenizers/fineweb_1024_bpe.model
    data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
    data/datasets/fineweb10B_sp1024/fineweb_val_*.bin
"""
from __future__ import annotations

import contextlib
import gc
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ==============================================================================
# CONFIG
# ==============================================================================

# DDP setup — works for both single-GPU and torchrun
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DDP = WORLD_SIZE > 1
IS_MASTER = RANK == 0

DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

TIME_BUDGET = float(os.environ.get("TIME_BUDGET", 600.0))

VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 1024))
MASK_TOKEN_ID = VOCAB_SIZE

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_VOCAB_SUFFIX = {1024: "sp1024", 4096: "sp4096", 8192: "sp8192"}
_DATA_DIR = _VOCAB_SUFFIX.get(VOCAB_SIZE, f"sp{VOCAB_SIZE}")
DATA_PATH = os.environ.get("DATA_PATH", str(REPO_ROOT / "data" / "datasets" / f"fineweb10B_{_DATA_DIR}"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", str(REPO_ROOT / "data" / "tokenizers" / f"fineweb_{VOCAB_SIZE}_bpe.model"))
NUM_LAYERS = 6
MODEL_DIM = 768
NUM_HEADS = 12
MLP_MULT = 3
SEQ_LEN = 512
LOGIT_SOFTCAP = 30.0

T_MIN = 0.1
T_MAX = 0.6
NOISE_SCHEDULE = "cosine"
SIGMA_MIN = 1e-4
SIGMA_MAX = 20.0
EVAL_ELBO_STEPS = 64

# Per-GPU batch size. Effective batch = TRAIN_BATCH_TOKENS * WORLD_SIZE
TRAIN_BATCH_TOKENS = int(os.environ.get("BATCH_TOKENS", 24576))
WARMUP_STEPS = 50
WARMDOWN_FRAC = 0.15
MAX_ITERATIONS = 1_000_000
VAL_BATCH_TOKENS = 4096
MATRIX_LR = 0.02
SCALAR_LR = 0.02
EMBED_LR = 0.03
MUON_MOMENTUM = 0.95
MUON_STEPS = 3
# Newton-Muon (arXiv 2604.01472): right-precondition grad by (ZZᵀ)⁻¹ before NS.
# Z = per-layer input activation matrix. K = EMA(ZZᵀ/N) tracked per Muon layer,
# inverse refreshed every NMUON_REFRESH steps with trace-scaled Tikhonov ridge.
NMUON_ENABLED = int(os.environ.get("NMUON", 1)) != 0
NMUON_BETA = 0.9        # EMA decay for K
NMUON_GAMMA = 0.2       # trace-scaled ridge coefficient
NMUON_REFRESH = 16      # refresh K⁻¹ every N steps
NMUON_K_INIT = 1e-3     # K init scale: K0 = NMUON_K_INIT · I
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 1))
SC_CADENCE = 3  # 1-in-N training steps do a detached teacher fwd for mask-gated sc
CHECKPOINT_EVERY = 50_000
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints_cuda"

def log(msg):
    """Print only on master rank."""
    if IS_MASTER:
        print(msg, flush=True)

# ==============================================================================
# NOISE SCHEDULE
# ==============================================================================

_LOG_RATIO = math.log(SIGMA_MAX / SIGMA_MIN)

def get_sigma_bar(t):
    if NOISE_SCHEDULE == "loglinear":
        return SIGMA_MIN * math.exp(_LOG_RATIO * t)
    else:
        s = 0.008
        f_t = math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        alpha = max(1e-7, min(1.0 - 1e-7, 1.0 - f_t / f_0))
        return -math.log(1.0 - alpha)

def get_mask_prob(t):
    return 1.0 - math.exp(-get_sigma_bar(t))

def get_dalpha_dt(t, dt=1e-5):
    a_hi = get_mask_prob(min(t + dt, 1.0))
    a_lo = get_mask_prob(max(t - dt, 0.0))
    return (a_hi - a_lo) / (2 * dt)

def alpha_to_t(alpha, s=0.008):
    """Invert the cosine schedule: given alpha in [0,1], return t in [0,1]."""
    f0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
    val = (1.0 - alpha) * f0
    val = max(0.0, min(1.0, val))
    arg = math.acos(math.sqrt(val))
    t = (1 + s) * (2 / math.pi) * arg - s
    return max(0.0, min(1.0, t))

# ==============================================================================
# HELPERS
# ==============================================================================

def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def load_data_shard(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size < 2 or int(header[0]) != 20240520:
        raise ValueError(f"Bad shard header: {path}")
    ntok = int(header[2])
    return np.fromfile(path, dtype="<u2", offset=256 * 4, count=ntok)


def _find_coprime(n, start=7919):
    """Find a number coprime to n, starting from start."""
    from math import gcd
    c = start
    while gcd(c, n) != 1:
        c += 1
    return c

class DataLoader:
    """Coprime-stride data loader: jumps through the shard by a large coprime
    stride so consecutive batches come from different documents/regions.
    Reduces correlation between nearby minibatches."""
    def __init__(self, pattern, batch_tokens, seq_len, rank=0, world_size=1):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No shards matching {pattern}")
        self.batch_tokens = batch_tokens
        self.seq_len = seq_len
        self.batch_size = batch_tokens // seq_len
        self.rank = rank
        self.world_size = world_size
        self.file_idx = rank % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self._init_stride()
        self.pos = 0

    def _init_stride(self):
        """Compute coprime stride for current shard."""
        chunk_size = self.batch_size * self.seq_len
        n_chunks = max(1, len(self.tokens) // chunk_size)
        self.stride = _find_coprime(n_chunks) if n_chunks > 1 else 1
        self.n_chunks = n_chunks
        self.chunk_idx = 0

    def reset(self):
        self.file_idx = self.rank % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self._init_stride()
        self.pos = 0
        self.chunk_idx = 0

    def next_batch(self):
        chunk_size = self.batch_size * self.seq_len
        # Advance to next shard if we've visited all chunks
        if self.chunk_idx >= self.n_chunks:
            self.file_idx = (self.file_idx + self.world_size) % len(self.files)
            self.tokens = load_data_shard(self.files[self.file_idx])
            self._init_stride()
            self.chunk_idx = 0
        # Coprime stride through the shard
        offset = (self.chunk_idx * self.stride % self.n_chunks) * chunk_size
        # Fallback if near end of shard
        if offset + chunk_size > len(self.tokens):
            offset = 0
        chunk = self.tokens[offset:offset + chunk_size].astype(np.int32)
        self.chunk_idx += 1
        self.pos = offset + chunk_size
        buf = torch.from_numpy(chunk).to(DEVICE, dtype=torch.long)
        return buf.reshape(self.batch_size, self.seq_len)


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
# MODEL — Bidirectional Transformer
# ==============================================================================

def _build_rope_freqs(seq_len, head_dim, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    angles = t[:, None] * freqs[None, :]
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x, cos_vals, sin_vals):
    """x: (B, heads, T, hd)"""
    T = x.shape[2]
    cos_v = cos_vals[:T].to(x.device)
    sin_v = sin_vals[:T].to(x.device)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos_v - x2 * sin_v
    o2 = x1 * sin_v + x2 * cos_v
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class BidirectionalAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, _ = x.shape
        hd = self.head_dim
        q = self.c_q(x).reshape(B, T, self.n_heads, hd).permute(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, self.n_heads, hd).permute(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, self.n_heads, hd).permute(0, 2, 1, 3)

        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        # QK-norm
        q = rms_norm(q)
        k = rms_norm(k)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        attn = F.softmax(attn, dim=-1).to(v.dtype)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(out)


class DiffusionBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult):
        super().__init__()
        self.attn = BidirectionalAttention(dim, n_heads)
        self.fc = nn.Linear(dim, dim * mlp_mult, bias=False)
        self.proj = nn.Linear(dim * mlp_mult, dim, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(rms_norm(x), rope_cos, rope_sin)
        h = self.fc(rms_norm(x))
        h = F.gelu(h)
        x = x + self.proj(h)
        return x


class DiffusionLM(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, mlp_mult, seq_len):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, dim)
        self.blocks = nn.ModuleList(
            [DiffusionBlock(dim, n_heads, mlp_mult) for _ in range(n_layers)]
        )
        self.out_head = nn.Linear(dim, vocab_size, bias=True)
        self.logit_softcap = LOGIT_SOFTCAP
        rope_cos, rope_sin = _build_rope_freqs(seq_len, dim // n_heads)
        self.register_buffer("_rope_cos", rope_cos)
        self.register_buffer("_rope_sin", rope_sin)
        self.t_embed = nn.Linear(1, dim, bias=True)

    def forward(self, x_noised, t_value, mask=None, sc_emb=None):
        B, T = x_noised.shape
        h = self.embed(x_noised)
        t_scalar = torch.tensor([[t_value]], device=x_noised.device, dtype=h.dtype)
        t_bias = self.t_embed(t_scalar)
        h = h + t_bias
        if sc_emb is not None:
            # Mask-gated self-conditioning (v2): sc contributes only at masked
            # positions. v1 pollution at low-t unmasked positions killed the
            # signal; gating removes that while preserving high-t help.
            assert mask is not None, "sc_emb requires mask to gate"
            h = h + sc_emb.to(h.dtype) * mask.unsqueeze(-1).to(h.dtype)

        rope_cos = self._rope_cos[:T]
        rope_sin = self._rope_sin[:T]

        for block in self.blocks:
            h = block(h, rope_cos, rope_sin)
        h = rms_norm(h)

        logits = self.out_head(h)
        if self.logit_softcap > 0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits

# ==============================================================================
# MUON OPTIMIZER
# ==============================================================================

def zeropower_newtonschulz5(g, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (torch.sqrt(torch.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class SplitOptimizer:
    """Muon for 2D weight matrices, Adam for embeddings/scalars.

    Optionally applies Newton-Muon right-preconditioning (arXiv 2604.01472)
    to the Muon gradient: g ← g · K⁻¹ where K is the EMA of input-activation
    second moment ZZᵀ/N per layer.
    """
    def __init__(self, model):
        self.matrix_params = {}
        self.embed_params = {}
        self.scalar_params = {}

        for name, p in model.named_parameters():
            if p.ndim == 2 and "embed" not in name and "t_embed" not in name:
                self.matrix_params[name] = p
            elif "embed" in name and p.ndim == 2:
                self.embed_params[name] = p
            else:
                self.scalar_params[name] = p

        self.muon_bufs = {k: torch.zeros_like(p) for k, p in self.matrix_params.items()}

        self.adam_embed = torch.optim.Adam(
            self.embed_params.values() if self.embed_params else [torch.zeros(1)],
            lr=EMBED_LR, betas=(0.9, 0.95),
        )
        self.adam_scalar = torch.optim.Adam(
            self.scalar_params.values() if self.scalar_params else [torch.zeros(1)],
            lr=SCALAR_LR, betas=(0.9, 0.95),
        )

        # Newton-Muon preconditioner state.
        # For layers with input dim > MODEL_DIM divisible by MODEL_DIM (MLP
        # contraction), we use block-diagonal K: split input into MODEL_DIM
        # chunks and track a separate K per chunk.
        self.nmuon_enabled = NMUON_ENABLED
        self.muon_modules = {}   # param_name → nn.Linear (module whose weight is that param)
        self.ztz_ema = {}        # param_name → list[tensor(n,n)], fp32 EMA
        self.ztz_inv = {}        # param_name → list[tensor(n,n)], fp32 cached inverse
        self.batch_ztz = {}      # param_name → list[tensor(n,n)], per-step accumulator
        self.batch_tokens = {}   # param_name → int (per-rank tokens this step)
        self._record_ztz = True
        self._hook_handles = []

        if self.nmuon_enabled:
            all_modules = dict(model.named_modules())
            for pname, p in self.matrix_params.items():
                mod_name = pname.rsplit(".", 1)[0]
                mod = all_modules.get(mod_name)
                if not isinstance(mod, nn.Linear):
                    continue
                self.muon_modules[pname] = mod
                d_in = mod.weight.shape[1]
                is_block = (d_in > MODEL_DIM and d_in % MODEL_DIM == 0)
                # Pre-allocate batch_ztz buffers once; reuse across steps to
                # avoid allocator churn (fresh 101 MB/step of fp32 fragmenting
                # the arena with sc v2's 805 MB transient).
                dev = p.device
                if is_block:
                    n_blocks = d_in // MODEL_DIM
                    self.batch_ztz[pname] = [
                        torch.zeros(MODEL_DIM, MODEL_DIM, device=dev, dtype=torch.float32)
                        for _ in range(n_blocks)
                    ]
                else:
                    self.batch_ztz[pname] = [
                        torch.zeros(d_in, d_in, device=dev, dtype=torch.float32)
                    ]
                self.batch_tokens[pname] = 0
                h = mod.register_forward_pre_hook(self._make_hook(pname, is_block))
                self._hook_handles.append(h)

        log(f"  Muon keys: {len(self.matrix_params)}, Embed keys: {len(self.embed_params)}, Scalar keys: {len(self.scalar_params)}")
        if self.nmuon_enabled:
            log(f"  Newton-Muon ON: β={NMUON_BETA}, γ={NMUON_GAMMA}, refresh={NMUON_REFRESH}, hooked {len(self.muon_modules)} modules")

    def _make_hook(self, pname, is_block):
        """Forward pre-hook factory. Accumulates XᵀX into self.batch_ztz[pname]
        during student forward only (guarded by is_grad_enabled to skip
        teacher sc fwd and eval). Matmul runs inside torch.no_grad() on a
        detached X — otherwise the XᵀX node would stay in the autograd graph
        and retain all upstream activations, causing OOM on DDP backward."""
        def hook(module, inputs):
            if not self._record_ztz or not torch.is_grad_enabled():
                return
            X = inputs[0]
            if X.dim() > 2:
                X = X.reshape(-1, X.shape[-1])
            N = X.shape[0]
            with torch.no_grad():
                Xd = X.detach()
                if is_block:
                    d = MODEL_DIM
                    n_blocks = Xd.shape[1] // d
                    ztz_list = [(Xd[:, i*d:(i+1)*d].t() @ Xd[:, i*d:(i+1)*d]).float()
                                for i in range(n_blocks)]
                else:
                    ztz_list = [(Xd.t() @ Xd).float()]
            # Buffers pre-allocated in __init__; always in-place add to avoid
            # allocator churn that fragments CUDA memory over long runs.
            for i, z in enumerate(ztz_list):
                self.batch_ztz[pname][i].add_(z)
            self.batch_tokens[pname] += N
        return hook

    def update_preconditioner(self, step_idx):
        """Call after all backward passes for this step, before the Muon step.

        1. Allreduce ZᵀZ across DDP ranks (so every rank applies the same K).
        2. Update per-layer EMA: K ← β·K + (1-β) · (ZᵀZ / N_total).
        3. Refresh cached K⁻¹ every NMUON_REFRESH steps (Cholesky-based).
        """
        if not self.nmuon_enabled or not self.batch_ztz:
            return

        # DDP sync: sum ZᵀZ and token counts across ranks, then compute the
        # global per-rank-averaged ZᵀZ/N.
        if IS_DDP:
            for pname, ztz_list in self.batch_ztz.items():
                for z in ztz_list:
                    dist.all_reduce(z, op=dist.ReduceOp.SUM)
            tok_tensor = torch.tensor(
                [self.batch_tokens[pname] for pname in self.batch_ztz],
                device=DEVICE, dtype=torch.float64,
            )
            dist.all_reduce(tok_tensor, op=dist.ReduceOp.SUM)
            tokens_global = {pname: float(tok_tensor[i].item())
                             for i, pname in enumerate(self.batch_ztz)}
        else:
            tokens_global = {p: float(n) for p, n in self.batch_tokens.items()}

        for pname, ztz_list in self.batch_ztz.items():
            n_tok = max(tokens_global[pname], 1.0)
            if pname not in self.ztz_ema:
                # Init K at NMUON_K_INIT · I per block
                self.ztz_ema[pname] = [
                    NMUON_K_INIT * torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
                    for z in ztz_list
                ]
            for i, z in enumerate(ztz_list):
                avg = z / n_tok
                self.ztz_ema[pname][i].mul_(NMUON_BETA).add_(avg, alpha=(1.0 - NMUON_BETA))

        # Zero buffers in-place — do NOT .clear() the dict; the tensors are
        # pre-allocated once in __init__ and reused every step to prevent
        # allocator fragmentation.
        for ztz_list in self.batch_ztz.values():
            for z in ztz_list:
                z.zero_()
        for pname in self.batch_tokens:
            self.batch_tokens[pname] = 0

        # Refresh inverse every NMUON_REFRESH steps (and at the very first call)
        if step_idx % NMUON_REFRESH == 0 or not self.ztz_inv:
            for pname, K_list in self.ztz_ema.items():
                invs = []
                for K in K_list:
                    n = K.shape[0]
                    ridge = NMUON_GAMMA * (K.diagonal().sum() / n)
                    K_reg = K + ridge * torch.eye(n, device=K.device, dtype=K.dtype)
                    try:
                        L = torch.linalg.cholesky(K_reg)
                        K_inv = torch.cholesky_inverse(L)
                    except Exception:
                        # Fallback to identity (skip preconditioning this step for this block)
                        K_inv = torch.eye(n, device=K.device, dtype=K.dtype)
                    invs.append(K_inv)
                self.ztz_inv[pname] = invs

    def precondition(self, pname, g):
        """Right-multiply g by cached K⁻¹ for this layer. No-op if K⁻¹ not
        yet computed for this layer (first step of training)."""
        if not self.nmuon_enabled or pname not in self.ztz_inv:
            return g
        K_invs = self.ztz_inv[pname]
        if len(K_invs) == 1:
            return g @ K_invs[0]
        # Block-diagonal: split g's input-dim columns into MODEL_DIM chunks
        chunk = K_invs[0].shape[0]
        parts = g.split(chunk, dim=1)
        return torch.cat([p @ K_inv for p, K_inv in zip(parts, K_invs)], dim=1)

    def zero_grad(self):
        for p in self.matrix_params.values():
            if p.grad is not None:
                p.grad.zero_()
        self.adam_embed.zero_grad()
        self.adam_scalar.zero_grad()

    def state_dict(self):
        return {
            "muon_bufs": {k: v.cpu() for k, v in self.muon_bufs.items()},
            "adam_embed": self.adam_embed.state_dict(),
            "adam_scalar": self.adam_scalar.state_dict(),
            # Newton-Muon: persist EMA (but not cached inverse; rebuilt on refresh)
            "ztz_ema": {k: [z.cpu() for z in v] for k, v in self.ztz_ema.items()},
        }

    def load_state_dict(self, sd):
        for k, v in sd["muon_bufs"].items():
            if k in self.muon_bufs:
                self.muon_bufs[k] = v.to(DEVICE)
        self.adam_embed.load_state_dict(sd["adam_embed"])
        self.adam_scalar.load_state_dict(sd["adam_scalar"])
        # Newton-Muon EMA restore (optional — checkpoints from pre-NMUon runs
        # won't have this key, and will restart EMA from scratch next step)
        for k, v in sd.get("ztz_ema", {}).items():
            if k in self.matrix_params:
                self.ztz_ema[k] = [z.to(DEVICE) for z in v]

# ==============================================================================
# CHECKPOINTING
# ==============================================================================

def save_checkpoint(model, opt, step, total_training_time, smooth_loss, train_loader):
    if not IS_MASTER:
        return
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "total_training_time": total_training_time,
        "smooth_loss": smooth_loss,
        "data_file_idx": train_loader.file_idx,
        "data_pos": train_loader.pos,
    }, str(CHECKPOINT_DIR / "checkpoint.pt"))
    log(f"  [checkpoint saved at step {step}, time={total_training_time:.0f}s]")


def load_checkpoint(model, opt, train_loader):
    ckpt_path = CHECKPOINT_DIR / "checkpoint.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    # Skip checkpoint if model architecture changed (e.g. different dim or seq_len)
    ckpt_embed = ckpt["model"].get("embed.weight", torch.empty(0,0))
    ckpt_dim = ckpt_embed.shape[-1] if ckpt_embed.ndim == 2 else 0
    ckpt_vocab = ckpt_embed.shape[0] if ckpt_embed.ndim == 2 else 0
    ckpt_seq = ckpt["model"].get("_rope_cos", torch.empty(0,0)).shape[0]
    if ckpt_dim != MODEL_DIM or ckpt_seq != SEQ_LEN or ckpt_vocab != VOCAB_SIZE + 1:
        log(f"  [checkpoint skipped: dim={ckpt_dim}/seq={ckpt_seq}/vocab={ckpt_vocab} vs {MODEL_DIM}/{SEQ_LEN}/{VOCAB_SIZE+1}, training from scratch]")
        return None
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    train_loader.file_idx = ckpt["data_file_idx"]
    train_loader.tokens = load_data_shard(train_loader.files[train_loader.file_idx])
    train_loader.pos = ckpt["data_pos"]
    log(f"  [checkpoint loaded: step={ckpt['step']}, time={ckpt['total_training_time']:.0f}s]")
    return ckpt["step"], ckpt["total_training_time"], ckpt["smooth_loss"]

# ==============================================================================
# TRAINING LOSS
# ==============================================================================

_N_STRATA = 8
_GLOBAL_STEP = 0
_ALPHA_MIN = get_mask_prob(T_MIN)
_ALPHA_MAX = get_mask_prob(T_MAX)

def diffusion_loss(model, tokens):
    # Alpha-uniform stratified sampling: q(t) ∝ dalpha/dt, so the ELBO
    # importance weight collapses from (dalpha/mask_prob) to (1/mask_prob).
    # This matches training density to where BPB mass actually lives.
    global _GLOBAL_STEP
    B, T = tokens.shape

    stratum = _GLOBAL_STEP % _N_STRATA
    stride = (_ALPHA_MAX - _ALPHA_MIN) / _N_STRATA
    a_lo = _ALPHA_MIN + stratum * stride
    a_hi = a_lo + stride
    alpha_val = float(torch.empty(1).uniform_(a_lo, a_hi).item())
    t_val = alpha_to_t(alpha_val)
    mask_prob = max(alpha_val, 0.01)

    mask = torch.rand(B, T, device=tokens.device) < mask_prob
    masked_tokens = torch.where(mask, MASK_TOKEN_ID, tokens)

    # Mask-gated partial self-conditioning (v2): 1-in-SC_CADENCE steps run a
    # detached teacher fwd and feed sc_emb into the student, gated by mask.
    use_sc = (_GLOBAL_STEP % SC_CADENCE == 0)
    sc_emb = None
    if use_sc:
        with torch.no_grad():
            t_logits = model(masked_tokens, t_val, mask=mask)
            # Chunked bf16 softmax @ embed — avoids a (B, T, V) fp32 allocation
            # that OOMs at SP16384. bf16 is fine: sc_emb is a weak input
            # perturbation, not a gradient path.
            embed_ct = model.embed.weight[:VOCAB_SIZE].to(t_logits.dtype)
            sc_emb = torch.empty(B, T, MODEL_DIM, device=t_logits.device, dtype=t_logits.dtype)
            for i in range(0, T, 128):
                j = min(i + 128, T)
                sc_emb[:, i:j, :] = F.softmax(t_logits[:, i:j, :], dim=-1) @ embed_ct
            del t_logits

    logits = model(masked_tokens, t_val, mask=mask, sc_emb=sc_emb)
    logits_flat = logits.reshape(-1, VOCAB_SIZE).float()
    targets_flat = tokens.reshape(-1)

    per_tok = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    mask_flat = mask.reshape(-1).float()
    avg_ce = (per_tok * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
    return avg_ce / mask_prob

# ==============================================================================
# ELBO EVALUATION
# ==============================================================================

@torch.no_grad()
def compute_elbo_bpb(model, val_loader, bytes_per_token, num_steps, num_batches):
    total_weighted_nll = 0.0
    total_bytes = 0
    n_sequences = 0

    for batch_idx in range(num_batches):
        tokens = val_loader.next_batch()
        B, T = tokens.shape
        n_sequences += B

        token_np = tokens.cpu().numpy().reshape(-1)
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
            mask = torch.rand(B, T, device=tokens.device) < mask_prob
            masked_tokens = torch.where(mask, MASK_TOKEN_ID, tokens)
            # Always-2-pass at eval: teacher → sc_emb → student.
            # Chunked bf16 softmax @ embed (matches training path).
            t_logits = model(masked_tokens, t_mid, mask=mask)
            embed_ct = model.embed.weight[:VOCAB_SIZE].to(t_logits.dtype)
            sc_emb = torch.empty(B, T, MODEL_DIM, device=t_logits.device, dtype=t_logits.dtype)
            for i in range(0, T, 128):
                j = min(i + 128, T)
                sc_emb[:, i:j, :] = F.softmax(t_logits[:, i:j, :], dim=-1) @ embed_ct
            del t_logits
            logits = model(masked_tokens, t_mid, mask=mask, sc_emb=sc_emb)

            logits_flat = logits.reshape(-1, VOCAB_SIZE).float()
            targets_flat = tokens.reshape(-1)
            per_token_nll = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            mask_flat = mask.reshape(-1).float()

            n_masked = mask_flat.sum().item()
            if n_masked > 0:
                avg_masked_nll = (per_token_nll * mask_flat).sum().item() / n_masked
            else:
                avg_masked_nll = 0.0

            batch_nll += avg_masked_nll * delta_alpha

        total_weighted_nll += batch_nll * B * T

        if (batch_idx + 1) % 4 == 0:
            log(f"  ELBO eval {batch_idx + 1}/{num_batches}")

    total_tokens = n_sequences * T
    avg_nll_nats = total_weighted_nll / total_tokens
    bpb = avg_nll_nats / math.log(2) * (total_tokens / total_bytes)
    return bpb

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    t_start = time.time()

    # DDP init
    if IS_DDP:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    log(f"Device: {DEVICE}, dtype: {COMPUTE_DTYPE}, world_size: {WORLD_SIZE}")
    if IS_MASTER and DEVICE.startswith("cuda"):
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            log(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f}GB)")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    bytes_per_token = build_bytes_per_token(sp)

    batch_size = TRAIN_BATCH_TOKENS // SEQ_LEN
    eff_batch = TRAIN_BATCH_TOKENS * WORLD_SIZE * GRAD_ACCUM
    log(f"DiffusionLM: {NUM_LAYERS}L dim={MODEL_DIM} heads={NUM_HEADS} "
        f"mlp={MLP_MULT}x seq={SEQ_LEN}")
    log(f"Batch: {batch_size}/gpu x {WORLD_SIZE} GPUs = {eff_batch} tokens/step")
    log(f"Schedule: {NOISE_SCHEDULE} | ELBO steps: {EVAL_ELBO_STEPS} | Muon LR: {MATRIX_LR}")
    log(f"Time budget: {TIME_BUDGET}s")

    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tv = max(min(t_val, 0.999), 0.001)
        log(f"  t={t_val:.2f}: mask_prob={get_mask_prob(tv):.4f} dalpha_dt={get_dalpha_dt(tv):.4f}")

    raw_model = DiffusionLM(VOCAB_SIZE, MODEL_DIM, NUM_LAYERS, NUM_HEADS, MLP_MULT, SEQ_LEN)
    raw_model = raw_model.to(DEVICE)

    num_params = sum(p.numel() for p in raw_model.parameters())
    log(f"Parameters: {num_params:,}")

    # DDP data loaders — each rank reads different data
    train_loader = DataLoader(f"{DATA_PATH}/fineweb_train_*.bin", TRAIN_BATCH_TOKENS, SEQ_LEN,
                              rank=RANK, world_size=WORLD_SIZE)
    val_loader = DataLoader(f"{DATA_PATH}/fineweb_val_*.bin", VAL_BATCH_TOKENS, SEQ_LEN)

    # Optimizer on raw model (before DDP wrapping)
    split_opt = SplitOptimizer(raw_model)

    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0

    ckpt = load_checkpoint(raw_model, split_opt, train_loader)
    if ckpt is not None:
        step, total_training_time, smooth_loss = ckpt

    # Wrap with DDP after checkpoint load
    if IS_DDP:
        model = torch.nn.parallel.DistributedDataParallel(raw_model, device_ids=[LOCAL_RANK])
    else:
        model = raw_model

    global _GLOBAL_STEP

    while step < MAX_ITERATIONS:
        t0 = time.time()

        split_opt.zero_grad()
        _GLOBAL_STEP = step

        for accum_idx in range(GRAD_ACCUM):
            tokens = train_loader.next_batch()

            # All ranks must use the same t_val for DDP gradient sync
            if IS_DDP:
                # Alpha-uniform stratified sampling (see diffusion_loss for rationale)
                stratum = (step * GRAD_ACCUM + accum_idx) % _N_STRATA
                stride = (_ALPHA_MAX - _ALPHA_MIN) / _N_STRATA
                a_lo = _ALPHA_MIN + stratum * stride
                a_hi = a_lo + stride
                a_tensor = torch.empty(1, device=DEVICE)
                if IS_MASTER:
                    a_tensor.uniform_(a_lo, a_hi)
                dist.broadcast(a_tensor, src=0)
                alpha_val = float(a_tensor.item())
                t_val = alpha_to_t(alpha_val)
                mask_prob = max(alpha_val, 0.01)

                B, T = tokens.shape
                mask = torch.rand(B, T, device=tokens.device) < mask_prob
                masked_tokens = torch.where(mask, MASK_TOKEN_ID, tokens)

                # Mask-gated partial self-conditioning (v2): 1-in-SC_CADENCE
                # steps do a detached teacher fwd on raw_model (un-DDP-wrapped),
                # project softmax(logits) through embed.weight to get sc_emb,
                # then the student fwd gates sc_emb by mask so only masked
                # positions see the sc perturbation.
                use_sc = ((step * GRAD_ACCUM + accum_idx) % SC_CADENCE == 0)
                sc_emb = None
                if use_sc:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                            t_logits = raw_model(masked_tokens, t_val, mask=mask)
                        # Chunked bf16 softmax @ embed — avoids a (B, T, V) fp32
                        # allocation that OOMs the 16GB 5080 at SP16384. bf16 is
                        # fine here: sc_emb is a weak input perturbation, not a
                        # gradient path.
                        embed_ct = raw_model.embed.weight[:VOCAB_SIZE].to(t_logits.dtype)
                        sc_emb = torch.empty(B, T, MODEL_DIM, device=t_logits.device, dtype=t_logits.dtype)
                        for i in range(0, T, 128):
                            j = min(i + 128, T)
                            sc_emb[:, i:j, :] = F.softmax(t_logits[:, i:j, :], dim=-1) @ embed_ct
                        del t_logits

                # Skip DDP allreduce on intermediate accum steps
                ctx = model.no_sync if (IS_DDP and accum_idx < GRAD_ACCUM - 1) else contextlib.nullcontext
                with ctx():
                    with torch.amp.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                        logits = model(masked_tokens, t_val, mask=mask, sc_emb=sc_emb)
                        logits_flat = logits.reshape(-1, VOCAB_SIZE).float()
                        targets_flat = tokens.reshape(-1)
                        per_tok = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                        mask_flat = mask.reshape(-1).float()
                        avg_ce = (per_tok * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
                        loss_val = avg_ce / mask_prob / GRAD_ACCUM
                    loss_val.backward()
            else:
                with torch.amp.autocast(device_type="cuda", dtype=COMPUTE_DTYPE, enabled=(DEVICE.startswith("cuda"))):
                    loss_val = diffusion_loss(model, tokens) / GRAD_ACCUM
                loss_val.backward()

        # Gradient clipping
        all_params = list(raw_model.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP).item()

        # LR schedule
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        if progress > (1.0 - WARMDOWN_FRAC):
            lr_mul = (1.0 - progress) / WARMDOWN_FRAC
        elif step < WARMUP_STEPS:
            lr_mul = (step + 1) / WARMUP_STEPS
        else:
            lr_mul = 1.0
        lr_mul = max(lr_mul, 0.01)

        # Adam steps
        for pg in split_opt.adam_embed.param_groups:
            pg["lr"] = EMBED_LR * lr_mul
        split_opt.adam_embed.step()

        for pg in split_opt.adam_scalar.param_groups:
            pg["lr"] = SCALAR_LR * lr_mul
        split_opt.adam_scalar.step()

        # Newton-Muon: update per-layer K = EMA(ZZᵀ/N), refresh K⁻¹ on cadence.
        # No-op if NMUON_ENABLED=False.
        split_opt.update_preconditioner(step)

        # Muon step (manual, operates on raw_model params which DDP keeps in sync)
        with torch.no_grad():
            lr = MATRIX_LR * lr_mul
            for name, p in split_opt.matrix_params.items():
                if p.grad is None:
                    continue
                g = p.grad.float()
                # Newton-Muon right-preconditioning: g ← g · K⁻¹ (paper Alg 1:
                # "K⁻¹ applied to raw layer gradient, before momentum and the
                # rest of the Muon pipeline").
                g = split_opt.precondition(name, g)
                buf = MUON_MOMENTUM * split_opt.muon_bufs[name] + g
                split_opt.muon_bufs[name] = buf
                g_eff = g + MUON_MOMENTUM * buf
                g_ortho = zeropower_newtonschulz5(g_eff, MUON_STEPS)
                scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
                p.add_(g_ortho * scale, alpha=-lr)

        dt = time.time() - t0
        if step > 5:
            total_training_time += dt

        lv = loss_val.item()
        smooth_loss = 0.9 * smooth_loss + 0.1 * lv
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        remaining = max(0, TIME_BUDGET - total_training_time)

        if step % 20 == 0:
            tok_s = eff_batch / max(dt, 1e-6)
            mem = f" | mem: {torch.cuda.memory_allocated() / 1e9:.1f}GB" if DEVICE.startswith("cuda") else ""
            log(f"step {step:05d} ({100 * progress:.1f}%) | loss: {debiased:.4f} | "
                f"gnorm: {grad_norm:.4f} | dt: {dt * 1000:.0f}ms | tok/s: {tok_s:,.0f} | "
                f"remaining: {remaining:.0f}s{mem}")

        step += 1
        if step % 5000 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        if step % CHECKPOINT_EVERY == 0 and step > 0:
            save_checkpoint(raw_model, split_opt, step, total_training_time, smooth_loss, train_loader)
        # Synchronized stop: rank 0 decides, broadcasts to all ranks
        if IS_DDP:
            stop = torch.tensor([1 if (step > 5 and total_training_time >= TIME_BUDGET) else 0],
                                device=DEVICE, dtype=torch.int32)
            dist.broadcast(stop, src=0)
            if stop.item():
                break
        else:
            if step > 5 and total_training_time >= TIME_BUDGET:
                break

    # Eval on master only
    if IS_MASTER:
        raw_model.eval()
        print("\n--- Per-t diagnostics ---")
        val_loader.reset()
        diag_tokens = val_loader.next_batch()
        per_t_ces = {}
        with torch.no_grad():
            for t_diag in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                mp = get_mask_prob(t_diag)
                dmask = torch.rand(diag_tokens.shape, device=diag_tokens.device) < mp
                masked = torch.where(dmask, MASK_TOKEN_ID, diag_tokens)
                # Always-2-pass at eval: teacher → sc_emb → student.
                # Chunked bf16 softmax @ embed (matches training path).
                t_logits = raw_model(masked, t_diag, mask=dmask)
                dB, dT = masked.shape
                embed_ct = raw_model.embed.weight[:VOCAB_SIZE].to(t_logits.dtype)
                sc_emb = torch.empty(dB, dT, MODEL_DIM, device=t_logits.device, dtype=t_logits.dtype)
                for i in range(0, dT, 128):
                    j = min(i + 128, dT)
                    sc_emb[:, i:j, :] = F.softmax(t_logits[:, i:j, :], dim=-1) @ embed_ct
                del t_logits
                dlogits = raw_model(masked, t_diag, mask=dmask, sc_emb=sc_emb)
                dlogits_f = dlogits.reshape(-1, VOCAB_SIZE).float()
                dtargets = diag_tokens.reshape(-1)
                dce = F.cross_entropy(dlogits_f, dtargets, reduction="none")
                dmask_f = dmask.reshape(-1).float()
                n_m = dmask_f.sum().item()
                avg_ce = (dce * dmask_f).sum().item() / n_m if n_m > 0 else 0.0
                per_t_ces[t_diag] = avg_ce
                print(f"  t={t_diag:.2f}: mask_prob={mp:.3f}, n_masked={int(n_m)}, avg_CE_masked={avg_ce:.4f}")

        print(f"\nELBO eval ({EVAL_ELBO_STEPS} levels x 8 batches)...")
        val_loader.reset()
        val_bpb = compute_elbo_bpb(raw_model, val_loader, bytes_per_token,
                                    num_steps=EVAL_ELBO_STEPS, num_batches=8)

        save_checkpoint(raw_model, split_opt, step, total_training_time, smooth_loss, train_loader)
        t_end = time.time()
        print("---")
        print(f"val_bpb:          {val_bpb:.6f}")
        print(f"training_seconds: {total_training_time:.1f}")
        print(f"total_seconds:    {t_end - t_start:.1f}")
        print(f"num_steps:        {step}")
        print(f"num_params_M:     {num_params / 1e6:.1f}")
        print(f"model_dim:        {MODEL_DIM}")
        print(f"num_layers:       {NUM_LAYERS}")
        print(f"world_size:       {WORLD_SIZE}")

        # === COPY-PASTE SUMMARY ===
        print("\n" + "=" * 50)
        print("RUN SUMMARY — copy everything below this line")
        print("=" * 50)
        print(f"val_bpb: {val_bpb:.6f}")
        print(f"steps: {step} | time: {total_training_time:.0f}s | params: {num_params / 1e6:.1f}M | gpus: {WORLD_SIZE}")
        per_t_str = " | ".join(f"t{t}={ce:.2f}" for t, ce in sorted(per_t_ces.items()))
        print(f"per-t: {per_t_str}")
        print(f"final_smooth_loss: {debiased:.4f}")
        print("=" * 50)

    if IS_DDP:
        dist.barrier()  # wait for master to finish eval before destroying
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
