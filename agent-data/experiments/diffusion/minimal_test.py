"""Rock-bottom MLX test: does gradient descent work AT ALL?"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Test 1: Raw manual gradient descent (no nn.Module, no optimizer)
print("=== Test 1: Manual gradient descent on raw array ===")
W = mx.random.normal((5,)) * 0.01  # 5-dim logits
target = 2  # predict class 2

for step in range(50):
    def loss_fn(w):
        return nn.losses.cross_entropy(w.reshape(1, -1), mx.array([target]), reduction="mean")

    loss_val = loss_fn(W)
    grad = mx.grad(loss_fn)(W)
    mx.eval(loss_val, grad)
    W = W - 0.1 * grad
    mx.eval(W)

    if step % 10 == 0:
        print(f"  step {step}: loss={float(loss_val):.4f} W[target]={float(W[target]):.4f}")

print()

# Test 2: nn.Module + optimizer (like our training loop)
print("=== Test 2: nn.Module + Adam optimizer ===")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 5, bias=False)
    def __call__(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = optim.Adam(learning_rate=1e-2)

# Fixed input and target
x = mx.ones((1, 16))
target = mx.array([2])

def model_loss(model, x):
    logits = model(x)
    return nn.losses.cross_entropy(logits, target, reduction="mean")

loss_and_grad = nn.value_and_grad(model, model_loss)

for step in range(50):
    loss_val, grads = loss_and_grad(model, x)
    mx.eval(loss_val, grads)
    optimizer.apply_gradients(grads, model)
    mx.eval(model.parameters())

    if step % 10 == 0:
        print(f"  step {step}: loss={float(loss_val):.4f}")

print()

# Test 3: nn.Embedding + weight-tied output (our exact setup)
print("=== Test 3: Embedding + weight-tied output ===")

class EmbedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)  # small vocab for test
    def __call__(self, x):
        h = self.embed(x)
        return h @ self.embed.weight.T

model2 = EmbedModel()
optimizer2 = optim.Adam(learning_rate=1e-2)

tokens = mx.array([[0, 1, 2, 3, 4, 5, 6, 7]])  # (1, 8)

def embed_loss(model, tokens):
    logits = model(tokens)
    return nn.losses.cross_entropy(
        logits.reshape(-1, 32),
        tokens.reshape(-1),
        reduction="mean"
    )

loss_and_grad2 = nn.value_and_grad(model2, embed_loss)

for step in range(100):
    loss_val, grads = loss_and_grad2(model2, tokens)
    mx.eval(loss_val, grads)
    optimizer2.apply_gradients(grads, model2)
    mx.eval(model2.parameters())

    if step % 20 == 0:
        # Also check if params are actually changing
        p_val = float(model2.embed.weight[0, 0])
        print(f"  step {step}: loss={float(loss_val):.4f} embed[0,0]={p_val:.6f}")

print()

# Test 4: Embedding lookup + SEPARATE head (no weight tying)
print("=== Test 4: Embedding + separate head (no tying) ===")

class SepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.head = nn.Linear(16, 32, bias=False)
    def __call__(self, x):
        return self.head(self.embed(x))

model3 = SepModel()
optimizer3 = optim.Adam(learning_rate=1e-2)
loss_and_grad3 = nn.value_and_grad(model3, embed_loss)

for step in range(100):
    loss_val, grads = loss_and_grad3(model3, tokens)
    mx.eval(loss_val, grads)
    optimizer3.apply_gradients(grads, model3)
    mx.eval(model3.parameters())

    if step % 20 == 0:
        p_val = float(model3.head.weight[0, 0])
        print(f"  step {step}: loss={float(loss_val):.4f} head[0,0]={p_val:.6f}")
