"""Debug: WHY is apply_gradients not updating?"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
    def __call__(self, x):
        return self.linear(x)

model = SimpleModel()

x = mx.ones((1, 4))
target = mx.array([1])

def loss_fn(model, x):
    logits = model(x)
    return nn.losses.cross_entropy(logits, target, reduction="mean")

loss_and_grad = nn.value_and_grad(model, loss_fn)
loss_val, grads = loss_and_grad(model, x)
mx.eval(loss_val, grads)

print("=== Parameter tree ===")
for k, v in tree_flatten(model.parameters()):
    print(f"  {k}: shape={v.shape} dtype={v.dtype}")

print("\n=== Gradient tree ===")
for k, v in tree_flatten(grads):
    print(f"  {k}: shape={v.shape} dtype={v.dtype} norm={float(mx.sum(v*v)**0.5):.6f}")

print("\n=== Param before update ===")
p_before = float(model.linear.weight[0, 0])
print(f"  linear.weight[0,0] = {p_before:.6f}")

# Try apply_gradients
optimizer = optim.Adam(learning_rate=0.01)
optimizer.apply_gradients(grads, model)
mx.eval(model.parameters())

print("\n=== Param after apply_gradients ===")
p_after = float(model.linear.weight[0, 0])
print(f"  linear.weight[0,0] = {p_after:.6f}")
print(f"  Changed: {p_before != p_after}")

# If that didn't work, try manually
if p_before == p_after:
    print("\n=== apply_gradients is broken! Trying manual approach... ===")

    # Check MLX version
    import mlx
    print(f"  MLX version: {mlx.__version__}")

    # Try model.update approach
    new_params = optim.Adam(learning_rate=0.01).apply_gradients(grads, model)
    print(f"  apply_gradients returned: {type(new_params)}")

    # Try passing parameters dict instead of model
    optimizer2 = optim.Adam(learning_rate=0.01)
    params = model.trainable_parameters()
    print(f"\n  trainable_parameters type: {type(params)}")
    for k, v in tree_flatten(params):
        print(f"    {k}: {v.shape}")

    # Manual SGD to verify gradients are real
    print("\n=== Manual SGD ===")
    grads_flat = tree_flatten(grads)
    for k, g in grads_flat:
        print(f"  grad {k}: max={float(mx.max(mx.abs(g))):.6f}")

    # Actually apply manually
    params_list = tree_flatten(model.parameters())
    grads_list = tree_flatten(grads)
    new = []
    for (pk, pv), (gk, gv) in zip(params_list, grads_list):
        new.append((pk, pv - 0.01 * gv))
    model.load_weights(new)
    mx.eval(model.parameters())
    p_manual = float(model.linear.weight[0, 0])
    print(f"\n  After manual SGD: linear.weight[0,0] = {p_manual:.6f}")
    print(f"  Changed: {p_before != p_manual}")

    # Now test if model actually produces different output
    new_loss = loss_fn(model, x)
    mx.eval(new_loss)
    print(f"  Loss before: {float(loss_val):.6f}, after manual SGD: {float(new_loss):.6f}")
