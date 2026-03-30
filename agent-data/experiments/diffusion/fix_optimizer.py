"""Verify: dict-based apply_gradients works, model-based doesn't."""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 5, bias=False)
    def __call__(self, x):
        return self.linear(x)

x = mx.ones((1, 16))
target = mx.array([2])

def model_loss(model, x):
    logits = model(x)
    return nn.losses.cross_entropy(logits, target, reduction="mean")

# === Test A: model-based (BROKEN) ===
print("=== Test A: optimizer.apply_gradients(grads, model) ===")
model_a = SimpleModel()
opt_a = optim.Adam(learning_rate=1e-2)
lag_a = nn.value_and_grad(model_a, model_loss)

for step in range(50):
    loss_val, grads = lag_a(model_a, x)
    mx.eval(loss_val, grads)
    opt_a.apply_gradients(grads, model_a)
    mx.eval(model_a.parameters())
    if step % 10 == 0:
        print(f"  step {step}: loss={float(loss_val):.4f}")

# === Test B: dict-based (CORRECT) ===
print("\n=== Test B: dict-based apply_gradients ===")
model_b = SimpleModel()
opt_b = optim.Adam(learning_rate=1e-2)
lag_b = nn.value_and_grad(model_b, model_loss)

for step in range(50):
    loss_val, grads = lag_b(model_b, x)
    mx.eval(loss_val, grads)
    grads_flat = dict(tree_flatten(grads))
    params_flat = dict(tree_flatten(model_b.parameters()))
    updated = opt_b.apply_gradients(grads_flat, params_flat)
    model_b.update(tree_unflatten(list(updated.items())))
    mx.eval(model_b.parameters())
    if step % 10 == 0:
        print(f"  step {step}: loss={float(loss_val):.4f}")

print("\nTest A should be FLAT, Test B should DECREASE.")
