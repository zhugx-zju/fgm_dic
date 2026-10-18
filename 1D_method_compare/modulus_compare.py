import os
import matplotlib.pyplot as plt
from asm.asm_modulus import AdjointStateMethod1DBar
import numpy as np
from utils.data_generation import *
from utils.dim_solve import supg, bubnov

# Geometry Parameters
L = 1.0
A = 1.0
P = 1e-2

# Material Parameters
distribution_type = 'linear'  ## sinusoidal heaviside linear quadratic
E0 = 1.0

# Inversion Parameters
alpha = 1e-10  # Tikhonov regularization parameter
beta = 0.0   # Boundary force penalty parameter
nel = 40
nstrain = 2
noise_level = 0.0  # Percentage
seed = 40
tol = 1e-12

# True E distribution at nodes
# Create ASM solver
asm = AdjointStateMethod1DBar(nel, L, A, P, alpha=alpha, beta=beta)
coord = asm.coord

# Generate simulation data
E_true = generate_E_true(distribution_type, coord, L, E0)
strain_exact = generate_strain_data(distribution_type, coord, nel, nstrain, P, A, E0, L)
u_exact = generate_displacement_data(asm, E_true)

# Generate noisy simulation data
u_noise = generate_noise_data(u_exact, noise_level, seed)
strain_noise = generate_noise_data(strain_exact, noise_level, seed)

# BUBNOV
E_bubnov = bubnov(nstrain, nel, coord, strain_noise, P, L, A)
# SUPG
E_supg = supg(nstrain, nel, coord, strain_noise, P, L, A)
# ASM
# Initial guess (at nodes)
E_init = np.ones(asm.nNod) * E0 * 1.5
E_min = 1e-4
# Optimize using unified interface
# Choose method: 'L-BFGS-B', 'trust-constr' or 'CG'
E_opt, history = asm.optimize(
    u_noise,
    E_init,
    E_min,
    method='L-BFGS-B',  # Change method here
    max_iter=1000,
    tol=tol
)
# Compute final displacement
u_opt = asm.forward_solve(E_opt)

# Compute strain from displacement (for strain plot)
strain_coord = np.linspace(0, L, (nstrain - 1) * nel + 1)

# Calculate relative errors
E_error_bubnov = np.abs(E_bubnov - E_true) / E_true
E_error_supg = np.abs(E_supg - E_true) / E_true
E_error_asm = np.abs(E_opt - E_true) / E_true

# Print errors
print(f"\nRelative error in E (Bubnov): {np.linalg.norm(E_bubnov - E_true) / np.linalg.norm(E_true):.6e}")
print(f"Relative error in E (SUPG): {np.linalg.norm(E_supg - E_true) / np.linalg.norm(E_true):.6e}")
print(f"Relative error in E (ASM): {np.linalg.norm(E_opt - E_true) / np.linalg.norm(E_true):.6e}")

# Figure 1: Three methods comparison + True E
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ax1.plot(coord, E_true, 'k-', label='True E', linewidth=2.5)
ax1.plot(coord, E_bubnov, 'b--', label='Bubnov-Galerkin', linewidth=2)
ax1.plot(coord, E_supg, 'g--', label='SUPG', linewidth=2)
ax1.plot(coord, E_opt, 'r--', label='ASM', linewidth=2)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('E', fontsize=12)
ax1.set_title('Young\'s Modulus Distribution Comparison', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# Figure 2: Relative errors
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
ax2.plot(coord, E_error_bubnov, 'b-', label='Bubnov-Galerkin', linewidth=2)
ax2.plot(coord, E_error_supg, 'g-', label='SUPG', linewidth=2)
ax2.plot(coord, E_error_asm, 'r-', label='ASM', linewidth=2)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Relative Error |E - E_true| / E_true', fontsize=12)
ax2.set_title('Relative Error Comparison', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# Figure 3: Displacement and Strain Data Comparison
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

# Displacement comparison
axes3[0].plot(coord, u_exact, 'k-', label='True', linewidth=2.5)
axes3[0].plot(coord, u_noise, 'gs', label='Noisy', markersize=4, markerfacecolor='none', markeredgewidth=1.5)
axes3[0].plot(coord, u_opt, 'r--', label='ASM', linewidth=2)
axes3[0].set_xlabel('x', fontsize=12)
axes3[0].set_ylabel('Displacement u', fontsize=12)
axes3[0].set_title('Displacement Comparison', fontsize=13)
axes3[0].legend(fontsize=11, loc='best')
axes3[0].grid(True, alpha=0.3)

# Strain comparison
axes3[1].plot(strain_coord, strain_exact, 'k-o', label='True', linewidth=2, markersize=6, markerfacecolor='k')
axes3[1].plot(strain_coord, strain_noise, 'gs', label='Noisy', markersize=7, markerfacecolor='none', markeredgewidth=1.5)
axes3[1].set_xlabel('x', fontsize=12)
axes3[1].set_ylabel('Strain ε', fontsize=12)
axes3[1].set_title('Strain Data Comparison', fontsize=13)
axes3[1].legend(fontsize=11, loc='best')
axes3[1].grid(True, alpha=0.3)

plt.tight_layout()

# Figure 4: ASM Convergence History
fig4, axes4 = plt.subplots(1, 3, figsize=(14, 5))

# Objective function iteration
axes4[0].semilogy(history['J'], 'b-', linewidth=2.5)
axes4[0].set_xlabel('Iteration', fontsize=12)
axes4[0].set_ylabel('Objective Function J', fontsize=12)
axes4[0].set_title('Objective Function History', fontsize=13)
axes4[0].grid(True, alpha=0.3)

# Gradient evolution
axes4[1].semilogy(history['grad_norm'], 'r-', linewidth=2.5)
axes4[1].set_xlabel('Iteration', fontsize=12)
axes4[1].set_ylabel('Gradient Norm', fontsize=12)
axes4[1].set_title('Gradient Norm History', fontsize=13)
axes4[1].grid(True, alpha=0.3)

# Row 2, Col 2: Relative change in E
axes4[2].semilogy(history['E_rel_change'], 'g-', linewidth=2)
axes4[2].set_xlabel('Iteration', fontsize=11)
axes4[2].set_ylabel('||ΔE|| / ||E||', fontsize=11)
axes4[2].set_title('Relative Change in E', fontsize=12)
axes4[2].grid(True, alpha=0.3)

plt.tight_layout()

# Save figures
filename_base = f'Compare_{distribution_type}_{nel}'
# fig1.savefig(f'{filename_base}_methods.png', dpi=300, bbox_inches='tight')
# fig1.savefig(f'{filename_base}_methods.tif', dpi=300, bbox_inches='tight')
# fig2.savefig(f'{filename_base}_errors.png', dpi=300, bbox_inches='tight')
# fig2.savefig(f'{filename_base}_errors.tif', dpi=300, bbox_inches='tight')
# fig3.savefig(f'{filename_base}_data_comparison.png', dpi=300, bbox_inches='tight')
# fig3.savefig(f'{filename_base}_data_comparison.tif', dpi=300, bbox_inches='tight')
# fig4.savefig(f'{filename_base}_convergence.png', dpi=300, bbox_inches='tight')
# fig4.savefig(f'{filename_base}_convergence.tif', dpi=300, bbox_inches='tight')

plt.show()