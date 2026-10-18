import os
import matplotlib.pyplot as plt
from asm.asm_log import AdjointStateMethod1DBar, generate_noise_disp
import numpy as np
from utils import dic_postprocess, data_generation


# Parameters
L = 1.0
A = 1.0
P = 1e-2
alpha = 1e-15  # Tikhonov regularization parameter
beta = 0.0   # Boundary force penalty parameter

nel = 200
nstrain = 20
noise_level = 0.00
seed = 4
# True E distribution at nodes (choose one)
distribution_type = 'heaviside'
E0 = 1.0
amp = P / A / E0
# Create ASM solver
asm = AdjointStateMethod1DBar(nel, L, A, P, alpha=alpha, beta=beta)
coord = asm.coord
E_true, strain_exact, strain_noise = data_generation(distribution_type, coord, nel, nstrain, P, A, E0, L, noise_level, seed)
# Generate synthetic measured data
u_exact, u_noise = generate_noise_disp(nel, L, A, P, E_true, noise_level)

# Initial guess (at nodes)
## E = E_min + exp(m)
m_init = np.ones(asm.nNod) * E0
E_min = 1e-4
# E_init = E_min + np.exp(m)

# Optimize using unified interface
# Choose method: 'L-BFGS-B', 'trust-constr' or 'CG'
m_opt, history = asm.optimize(
    u_noise,
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
axes[0, 1].plot(coord, u_noise, 'go', label='Measured u', markersize=4, alpha=0.6)
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
filepath_png = f'{filename}.png'
filepath_tif = f'{filename}.tif'
# plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
# plt.savefig(filepath_tif, dpi=300, bbox_inches='tight')
plt.show()

# Print error
E_error = np.linalg.norm(E_opt - E_true) / np.linalg.norm(E_true)
u_error = np.linalg.norm(u_opt - u_exact) / np.linalg.norm(u_exact)
print(f"\nRelative error in E: {E_error:.6e}")
print(f"Relative error in u: {u_error:.6e}")