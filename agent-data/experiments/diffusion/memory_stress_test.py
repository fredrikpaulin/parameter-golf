#!/usr/bin/env python3
"""
Memory stress test v3 — real model dimensions, tiny batch.
Uses 6L/384d (same weight shapes as train.py) but batch=1, seq=32 for speed.
"""
import gc
import math
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# Real model dimensions — same allocation patterns as train.py
DIM = 384
HEADS = 6
LAYERS = 6
MLP_MULT = 3
VOCAB = 1024
# Tiny batch/seq for speed
SEQ = 32
BATCH = 1

MUON_MOMENTUM = 0.95
MUON_STEPS = 5
GRAD_CLIP = 1.0

TARGET = 300_000
REPORT = 2000


def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if tr: x = x.T
    return x.astype(g.dtype)


def _build_rope_freqs(seq_len, head_dim, base=10000.0):
    freqs = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    t = mx.arange(seq_len).astype(mx.float32)
    angles = t[:, None] * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


def _apply_rope(x, cos_vals, sin_vals):
    T = x.shape[2]
    cos_v, sin_v = cos_vals[:T], sin_vals[:T]
    x1, x2 = x[..., ::2], x[..., 1::2]
    o1 = x1 * cos_v - x2 * sin_v
    o2 = x1 * sin_v + x2 * cos_v
    return mx.concatenate([o1[..., None], o2[..., None]], axis=-1).reshape(x.shape)


class BidirectionalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = DIM // HEADS
        self.c_q = nn.Linear(DIM, DIM, bias=False)
        self.c_k = nn.Linear(DIM, DIM, bias=False)
        self.c_v = nn.Linear(DIM, DIM, bias=False)
        self.c_proj = nn.Linear(DIM, DIM, bias=False)

    def __call__(self, x, rope_cos, rope_sin):
        B, T, _ = x.shape
        hd = self.head_dim
        q = self.c_q(x).reshape(B, T, HEADS, hd).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, HEADS, hd).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, HEADS, hd).transpose(0, 2, 1, 3)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        q = rms_norm(q)
        k = rms_norm(k)
        attn = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(hd))
        attn = mx.softmax(attn, axis=-1).astype(v.dtype)
        return self.c_proj((attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = BidirectionalAttention()
        self.fc = nn.Linear(DIM, DIM * MLP_MULT, bias=False)
        self.proj = nn.Linear(DIM * MLP_MULT, DIM, bias=False)

    def __call__(self, x, rc, rs):
        x = x + self.attn(rms_norm(x), rc, rs)
        x = x + self.proj(nn.gelu(self.fc(rms_norm(x))))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB + 1, DIM)
        self.blocks = [Block() for _ in range(LAYERS)]
        self.head = nn.Linear(DIM, VOCAB, bias=True)
        self.t_embed = nn.Linear(1, DIM, bias=True)
        self._rc, self._rs = _build_rope_freqs(SEQ, DIM // HEADS)

    def __call__(self, x, t):
        h = self.embed(x).astype(mx.bfloat16)
        h = h + self.t_embed(mx.array([[t]]).astype(mx.bfloat16))
        rc = self._rc[:x.shape[1]].astype(mx.bfloat16)
        rs = self._rs[:x.shape[1]].astype(mx.bfloat16)
        for b in self.blocks:
            h = b(h, rc, rs)
        logits = self.head(rms_norm(h))
        return 30.0 * mx.tanh(logits / 30.0)


class SplitOpt:
    def __init__(self, model):
        params = dict(tree_flatten(model.parameters()))
        self.mat_keys = [k for k, p in params.items() if p.ndim == 2 and 'embed' not in k and 'time' not in k]
        self.emb_keys = [k for k, p in params.items() if 'embed' in k and p.ndim == 2]
        self.scl_keys = [k for k, p in params.items() if k not in self.mat_keys and k not in self.emb_keys]
        self.muon_bufs = {k: mx.zeros_like(params[k]) for k in self.mat_keys}
        self.adam_s = optim.Adam(learning_rate=0.02, betas=[0.9, 0.95])
        self.adam_e = optim.Adam(learning_rate=0.03, betas=[0.9, 0.95])
        print(f"  Muon keys: {len(self.mat_keys)}, Embed keys: {len(self.emb_keys)}, Scalar keys: {len(self.scl_keys)}")

    def step(self, model, grads_tree):
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        for k in self.mat_keys:
            if k not in grads: continue
            g = grads[k]
            buf = MUON_MOMENTUM * self.muon_bufs[k] + g
            self.muon_bufs[k] = buf
            g_eff = g + MUON_MOMENTUM * buf
            g_ortho = zeropower_newtonschulz5(g_eff, MUON_STEPS)
            scale = math.sqrt(max(1.0, params[k].shape[0] / params[k].shape[1]))
            updated[k] = params[k] - 0.02 * (g_ortho * scale).astype(params[k].dtype)
        eg = {k: grads[k] for k in self.emb_keys if k in grads}
        ep = {k: params[k] for k in self.emb_keys if k in grads}
        if eg: updated.update(self.adam_e.apply_gradients(eg, ep))
        sg = {k: grads[k] for k in self.scl_keys if k in grads}
        sp = {k: params[k] for k in self.scl_keys if k in grads}
        if sg: updated.update(self.adam_s.apply_gradients(sg, sp))
        model.update(tree_unflatten(list(updated.items())))
        mx.eval(*self.muon_bufs.values())


def loss_fn(model, tokens):
    mask = mx.random.uniform(shape=tokens.shape) < 0.3
    masked = mx.where(mask, VOCAB, tokens)
    logits = model(masked, 0.3)
    ce = nn.losses.cross_entropy(logits.reshape(-1, VOCAB).astype(mx.float32), tokens.reshape(-1), reduction="none")
    mf = mask.reshape(-1).astype(mx.float32)
    return mx.sum(ce * mf) / mx.maximum(mx.sum(mf), mx.array(1.0)) * 2.0


def mem_stats():
    parts = []
    for name, fn in [('active', 'get_active_memory'), ('peak', 'get_peak_memory'), ('cache', 'get_cache_memory')]:
        if hasattr(mx, fn):
            parts.append(f"{name}={getattr(mx, fn)() / 1e9:.3f}GB")
        elif hasattr(mx, 'metal') and hasattr(mx.metal, fn):
            parts.append(f"{name}={getattr(mx.metal, fn)() / 1e9:.3f}GB")
    return " ".join(parts) if parts else "no stats"


def main():
    print(f"Stress test v3: {TARGET} steps, REAL dims ({LAYERS}L/{DIM}d), tiny batch ({BATCH}x{SEQ})")
    print(f"Crash expected ~248K if leak is size-dependent\n")

    mx.random.seed(42)
    model = Model()
    n = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Params: {n:,}")

    opt = SplitOpt(model)
    lag = nn.value_and_grad(model, loss_fn)
    tokens = mx.random.randint(0, VOCAB, shape=(BATCH, SEQ))
    mx.eval(tokens)

    t0 = time.time()
    for step in range(TARGET):
        lv, grads = lag(model, tokens)
        mx.eval(lv, grads)

        gp = tree_flatten(grads)
        gn = sum(float(mx.sum(g * g)) for _, g in gp) ** 0.5
        if gn > GRAD_CLIP:
            s = GRAD_CLIP / gn
            grads = tree_unflatten([(k, g * s) for k, g in gp])

        opt.step(model, grads)
        mx.eval(model.parameters())

        if step % 5000 == 0:
            gc.collect()
            if hasattr(mx, 'clear_cache'):
                mx.clear_cache()
            elif hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()

        if step % REPORT == 0:
            el = time.time() - t0
            sps = (step + 1) / max(el, 1e-6)
            eta = (TARGET - step) / max(sps, 1) / 60
            print(f"step {step:>7d}/{TARGET} | {sps:.0f} steps/s | ETA {eta:.1f}min | {mem_stats()}")

    print(f"\n{'='*50}")
    print(f"PASSED {TARGET} steps — no crash!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
