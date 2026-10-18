import os
import sys
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from asm.asm_log import AdjointStateMethod1DBar
from utils.data_generation import *

print("="*80)
print("Gradient Verification Test for ASM")
print("="*80)

# Parameters
nel = 50  # Use smaller mesh for faster verification
L = 1.0
A = 1.0
P = 1e-2
E0 = 1.0
alpha = 1e-8  # Tikhonov regularization
beta = 0.0    # Boundary force penalty
noise_level = 0.01
distribution_type = 'linear'

# Create ASM solver
asm = AdjointStateMethod1DBar(nel, L, A, P, alpha=alpha, beta=beta)
coord = asm.coord

# Generate true E and data
E_true = generate_E_true(distribution_type, coord, L, E0)
u_exact = generate_displacement_data(asm, E_true)
# u_noise = generate_noise_data(u_exact, noise_level, seed=42)

# Set measured data for gradient verification
asm.u_noise = u_exact

# Test point: random m values
print(f"\nTest Configuration:")
print(f"  Number of elements: {nel}")
print(f"  Number of nodes: {asm.nNod}")
print(f"  Regularization alpha: {alpha}")
print(f"  Boundary penalty beta: {beta}")
print(f"  Distribution type: {distribution_type}")

# Generate random m for testing
np.random.seed(123)
m_test = np.random.randn(asm.nNod) * 0.2 + 0.5  # Random around 0.5
E_min = 1e-4

print(f"\nTest point m statistics:")
print(f"  Mean: {np.mean(m_test):.4f}")
print(f"  Std:  {np.std(m_test):.4f}")
print(f"  Min:  {np.min(m_test):.4f}")
print(f"  Max:  {np.max(m_test):.4f}")

# Verify gradient
results = asm.verify_gradient(m_test, E_min=E_min, eps=1e-7, num_samples=10)

# Plot comparison if errors are large
if results['max_rel_error'] > 1e-4:
    print("Warning: Large gradient errors detected!")
    print("This suggests an error in the analytical gradient implementation.")
else:
    print("\n Gradient appears correct!")

print("\n" + "="*80)
print("Testing with different alpha values...")
print("="*80)

for alpha_test in [0.0, 1e-10, 1e-8, 1e-6]:
    print(f"\nalpha = {alpha_test:.0e}")
    asm_test = AdjointStateMethod1DBar(nel, L, A, P, alpha=alpha_test, beta=0.0)
    asm_test.u_noise = u_exact

    E_test = E_min + np.exp(m_test)
    u_test = asm_test.forward_solve(E_test)
    lambda_test = asm_test.adjoint_solve(E_test, u_test, u_exact)

    # Analytical
    grad_analytical = asm_test.compute_gradient(u_test, lambda_test, m_test, E_min)

    # Finite difference (only a few components for speed)
    grad_fd_samples = []
    indices_sample = [0, asm_test.nNod//2, asm_test.nNod-1]
    eps = 1e-7

    for i in indices_sample:
        m_plus = m_test.copy()
        m_plus[i] += eps
        E_plus = E_min + np.exp(m_plus)
        J_plus = asm_test.objective_function(E_plus, u_exact)

        m_minus = m_test.copy()
        m_minus[i] -= eps
        E_minus = E_min + np.exp(m_minus)
        J_minus = asm_test.objective_function(E_minus, u_exact)

        grad_fd_i = (J_plus - J_minus) / (2 * eps)
        rel_error = abs(grad_analytical[i] - grad_fd_i) / (abs(grad_fd_i) + 1e-12)

        print(f"  Node {i:3d}: Analytical={grad_analytical[i]:12.6e}, "
              f"FD={grad_fd_i:12.6e}, RelErr={rel_error:.2e}")

print("\n" + "="*80)
print("Gradient verification complete!")
print("="*80)
