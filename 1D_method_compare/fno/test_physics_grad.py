"""
Quick test to verify physics loss is differentiable
"""
import torch
import numpy as np
from physics_loss_torch import PhysicsLossPyTorch
from fem1d import FEM1DBar

print("="*80)
print("Testing Physics Loss Differentiability")
print("="*80)

# Setup
L = 1.0
A = 1.0
n_node = 101
batch_size = 4

# Create loss
physics_loss = PhysicsLossPyTorch(L, A, n_node)
fem_numpy = FEM1DBar(L, A, n_node)

# Generate test data
torch.manual_seed(42)
E_pred = torch.rand(batch_size, n_node) * 0.5 + 0.8
E_pred.requires_grad = True
F_batch = torch.rand(batch_size) * 0.1

# Generate corresponding u using numpy FEM
u_obs_list = []
for i in range(batch_size):
    E_np = E_pred[i].detach().numpy()
    F_np = F_batch[i].item()
    u_np = fem_numpy.forward_solve(E_np, F_np)
    u_obs_list.append(u_np)
u_obs = torch.tensor(np.array(u_obs_list), dtype=torch.float32)

print(f"\n1. Forward pass...")
print(f"   E_pred shape: {E_pred.shape}, requires_grad: {E_pred.requires_grad}")
print(f"   u_obs shape: {u_obs.shape}")
print(f"   F_batch shape: {F_batch.shape}")

loss = physics_loss(E_pred, u_obs, F_batch)
print(f"   Physics loss: {loss.item():.6e}")

print(f"\n2. Backward pass...")
loss.backward()

print(f"   ✓ Gradient computed!")
print(f"   Gradient shape: {E_pred.grad.shape}")
print(f"   Gradient norm: {E_pred.grad.norm().item():.6e}")
print(f"   Gradient range: [{E_pred.grad.min():.6e}, {E_pred.grad.max():.6e}]")

if E_pred.grad.norm() > 1e-10:
    print(f"\n✓✓✓ SUCCESS: Physics loss is differentiable!")
    print(f"   Gradients are non-zero and can backpropagate.")
else:
    print(f"\n✗✗✗ WARNING: Gradients are near zero!")

print("="*80)
