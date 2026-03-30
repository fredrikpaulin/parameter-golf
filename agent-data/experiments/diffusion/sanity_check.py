"""Sanity check: can MLX train ANYTHING on this data?
Minimal model: just embedding + weight-tied output. No transformer."""
import glob, math, os, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024")
VOCAB_SIZE = 1024
SEQ_LEN = 256
BATCH = 16

def load_data_shard(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    ntok = int(header[2])
    return np.fromfile(path, dtype="<u2", offset=256*4, count=ntok)

files = sorted(glob.glob(f"{DATA_PATH}/fineweb_train_*.bin"))
tokens_all = load_data_shard(files[0])

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, 128)
        # Small init like reference GPT (std=0.005)
        self.embed.weight = mx.random.normal(self.embed.weight.shape) * 0.005
    def __call__(self, x):
        h = self.embed(x)  # (B, T, 128)
        logits = h @ self.embed.weight.T  # (B, T, V) weight-tied
        return logits

model = TinyModel()
optimizer = optim.Adam(learning_rate=1e-3)

def loss_fn(model, tokens):
    logits = model(tokens)
    return nn.losses.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE).astype(mx.float32),
        tokens.reshape(-1),
        reduction="mean"
    )

loss_and_grad = nn.value_and_grad(model, loss_fn)

pos = 0
for step in range(200):
    chunk = tokens_all[pos:pos+BATCH*SEQ_LEN].astype(np.int32)
    pos += BATCH*SEQ_LEN
    batch = mx.array(chunk).reshape(BATCH, SEQ_LEN)
    
    loss_val, grads = loss_and_grad(model, batch)
    mx.eval(loss_val, grads)
    grads_flat = dict(tree_flatten(grads))
    params_flat = dict(tree_flatten(model.parameters()))
    updated = optimizer.apply_gradients(grads_flat, params_flat)
    model.update(tree_unflatten(list(updated.items())))
    mx.eval(model.parameters())
    
    if step % 10 == 0:
        print(f"step {step:03d} | loss: {float(loss_val):.4f}")

print("\nIf loss decreased, optimizer works. If flat, something is broken with MLX.")
