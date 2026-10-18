import os
import sys
import matplotlib.pyplot as plt
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from asm.asm_log import *

# Parameters
nel = 200
L = 1.0
A = 1.0
P = 1e-2
E0 = 1.0
alpha = 1e-15  # Tikhonov regularization parameter
beta = 0.0   # Boundary force penalty parameter
amp = P / A / E0

# Create ASM solver
asm = AdjointStateMethod1DBar(nel, L, A, P, alpha=alpha, beta=beta)
coord = asm.coord

# True E distribution at nodes (choose one)
distribution_type = 'heaviside'
if distribution_type == 'sinusoidal':
    k = 5
    a = 2 * np.pi * k / L
    E_true = E0 * (2 + np.sin(a * coord))
    # u_exact = 2 * (amp / a / np.sqrt(3)) * (np.arctan((2 * np.tan(a * coord / 2) + 1) / np.sqrt(3)) - np.pi / 6 + np.pi * (k * coord / L + 1 / 2))
    u_exact = 2 * (amp / a / np.sqrt(3)) * (np.arctan((2 * np.tan(a * coord / 2) + 1) / np.sqrt(3)) - np.pi / 6)
elif distribution_type == 'linear':
    E_true = E0 * (1 + coord / L)
    u_exact = amp * L * np.log(1 + coord / L)
elif distribution_type == 'quadratic':
    E_true = E0 * (1 + (coord / L)**2)
    u_exact = amp * L * np.arctan(coord /L)
elif distribution_type == 'heaviside':
    # ASME JAM 2024, Eq. (28): 4-step (Heaviside) modulus distribution
    r_coord = coord / L
    E_true = np.empty_like(coord, dtype=float)
    E_true[r_coord < 0.25] = 2.00 * E0
    E_true[(r_coord >= 0.25) & (r_coord < 0.50)] = 1.75 * E0
    E_true[(r_coord >= 0.50) & (r_coord < 0.75)] = 1.50 * E0
    E_true[r_coord >= 0.75] = 1.25 * E0
else:
    raise ValueError("distribution_type must be 'sinusoidal', 'linear', 'quadratic', or 'heaviside'.")
# Generate noise measured data
u_measured, _ = generate_noise_disp(nel, L, A, P, E_true, noise_level=0.01)
strain, _ = asm.compute_strain_stress(u_measured, E_true)
# Initial guess (at nodes)
# E = E_min + exp(m)
m_init = np.ones(asm.nNod) * E0
E_min = 1e-4
# E_init = E_min + np.exp(m)

# Optimize using unified interface
# Choose method: 'L-BFGS-B', 'trust-constr' or 'CG'
m_opt, history = asm.optimize(
    u_measured,
    m_init,
    E_min,
    method='L-BFGS-B',  # Change method here
    max_iter=1000,
    tol=1e-6
)

# Compute final displacement
E_opt = E_min + np.exp(m_opt)
u_opt = asm.forward_solve(E_opt)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# E distribution at nodes
axes[0, 0].plot(coord, E_true, 'b-o', label='True E', linewidth=2, markersize=4)
axes[0, 0].plot(coord, E_opt, 'r--', label='Identified E', linewidth=2)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('E')
axes[0, 0].set_title('Young\'s Modulus Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Displacement
# axes[0, 1].plot(coord, u_exact, 'b-o', label='Exact u', linewidth=2, markersize=4)
axes[0, 1].plot(coord, u_measured, 'go', label='Measured u', markersize=4, alpha=0.6)
axes[0, 1].plot(coord, u_opt, 'r--', label='Identified u', linewidth=2)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('u')
axes[0, 1].set_title('Displacement')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Objective function history
axes[1, 0].semilogy(history['J'], 'b-', linewidth=2)
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Objective Function J')
axes[1, 0].set_title('Convergence: Objective Function')
axes[1, 0].grid(True)

# Gradient norm history
axes[1, 1].semilogy(history['grad_norm'], 'r-', linewidth=2)
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Gradient Norm')
axes[1, 1].set_title('Convergence: Gradient Norm')
axes[1, 1].grid(True)

plt.tight_layout()

# Save figure
filename = f'ASM_{distribution_type}_{nel}'
filepath_png = os.path.join(script_dir, f'{filename}.png')
filepath_tif = os.path.join(script_dir, f'{filename}.tif')
# plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
# plt.savefig(filepath_tif, dpi=300, bbox_inches='tight')
plt.show()

# Print error
E_error = np.linalg.norm(E_opt - E_true) / np.linalg.norm(E_true)
u_error = np.linalg.norm(u_opt - u_exact) / np.linalg.norm(u_exact)
print(f"\nRelative error in E: {E_error:.6e}")
print(f"Relative error in u: {u_error:.6e}")
