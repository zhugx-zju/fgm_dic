import os
import matplotlib.pyplot as plt
from asm.asm_strain import AdjointStateMethod1DBar  # Changed to strain-based ASM
import numpy as np
from utils.data_generation import *
from utils.dim_solve import supg, bubnov

# Geometry Parameters
L = 1.0
A = 1.0
P = 1e-2

# Material Parameters
distribution_type = 'sinusoidal'  ## sinusoidal heaviside linear quadratic
E0 = 1.0

# Inversion Parameters
beta = 0.0   # Boundary force penalty parameter
nel = 40
nstrain = 4  # Number of strain measurement points
noise_level = 5.0  # Percentage
seed = 42
tol = 1e-12

# L-curve parameters
alpha_min = 1e-14
alpha_max = 1e-10
n_alpha = 20  # Number of alpha values to test

# True E distribution at nodes
# Create ASM solver (temporarily with alpha=0)
asm_temp = AdjointStateMethod1DBar(nel, L, A, P, alpha=0.0, beta=beta)
coord = asm_temp.coord

# Generate simulation data
E_true = generate_E_true(distribution_type, coord, L, E0)
strain_exact = generate_strain_data(distribution_type, coord, nel, nstrain, P, A, E0, L)
u_exact = generate_displacement_data(asm_temp, E_true)

# Generate noisy simulation data
u_noise = generate_noise_data(u_exact, noise_level, seed)
strain_noise = generate_noise_data(strain_exact, noise_level, seed)

print("="*80)
print("L-curve Analysis for ASM Regularization Parameter Selection")
print("="*80)
print(f"Distribution: {distribution_type}")
print(f"Noise level: {noise_level}%")
print(f"Number of elements: {nel}")
print(f"Number of strain measurements: {nstrain}")
print(f"Testing alpha range: [{alpha_min:.2e}, {alpha_max:.2e}]")
print("="*80)

# L-curve analysis
print("\nPerforming L-curve analysis...")
alpha_values = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
residual_norms = []
regularization_norms = []
E_solutions = []

E_init = np.ones(asm_temp.nNod) * E0 * 1.5
E_min = 1e-4

for i, alpha in enumerate(alpha_values):
    print(f"\n[{i+1}/{n_alpha}] Testing alpha = {alpha:.6e}")

    # Create ASM solver with current alpha
    asm = AdjointStateMethod1DBar(nel, L, A, P, alpha=alpha, beta=beta)

    # Optimize
    E_opt, history = asm.optimize(
        strain_noise,
        E_init,
        E_min,
        method='L-BFGS-B',
        max_iter=1000,
        tol=tol
    )

    E_solutions.append(E_opt)

    # Compute residual norm (data misfit)
    residual_norm = asm.compute_strain_fitting(E_opt, strain_noise)
    residual_norms.append(residual_norm)

    # Compute regularization norm (smoothness of E)
    reg_norm = asm.compute_tikhonov_term(E_opt)
    regularization_norms.append(reg_norm)

    print(f"  Residual norm: {residual_norm:.6e}, Regularization norm: {reg_norm:.6e}")

# Find L-curve corner using maximum curvature
x = np.log10(np.array(residual_norms))
y = np.log10(np.array(regularization_norms))

# Normalize to [0,1] range for numerical stability
residual_normalized = (x - np.mean(x)) / (np.std(x) + 1e-15)
reg_normalized = (y - np.mean(y)) / (np.std(y) + 1e-15)

# Method 1: Menger curvature (more robust)
curvature = np.zeros(n_alpha)
for i in range(1, n_alpha - 1):
    # Three points on the L-curve
    p1 = np.array([residual_normalized[i-1], reg_normalized[i-1]])
    p2 = np.array([residual_normalized[i], reg_normalized[i]])
    p3 = np.array([residual_normalized[i+1], reg_normalized[i+1]])

    # Menger curvature: κ = 4*Area / (|p1-p2|*|p2-p3|*|p3-p1|)
    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    d12 = np.linalg.norm(p2 - p1) + 1e-15
    d23 = np.linalg.norm(p3 - p2) + 1e-15
    d31 = np.linalg.norm(p1 - p3) + 1e-15
    curvature[i] = 4 * area / (d12 * d23 * d31)

print("\nCurvature values:")
for i in range(n_alpha):
    print(f"  α={alpha_values[i]:.2e}: curvature={curvature[i]:.6e}")

# Find index of maximum curvature
optimal_idx = np.argmax(curvature[1:-1]) + 1  # Exclude endpoints
alpha_optimal = alpha_values[optimal_idx]
E_optimal = E_solutions[optimal_idx]

print("\n" + "="*80)
print(f"Optimal alpha selected by L-curve: {alpha_optimal:.6e}")
print(f"  Residual norm: {residual_norms[optimal_idx]:.6e}")
print(f"  Regularization norm: {regularization_norms[optimal_idx]:.6e}")
print("="*80)

# BUBNOV
print("\nRunning Bubnov-Galerkin...")
E_bubnov = bubnov(nstrain, nel, coord, strain_noise, P, L, A)

# SUPG
print("Running SUPG...")
E_supg = supg(nstrain, nel, coord, strain_noise, P, L, A)

# ASM - Use optimal alpha from L-curve
print(f"\nRunning ASM with optimal alpha = {alpha_optimal:.6e}...")
E_opt = E_optimal
asm_final = AdjointStateMethod1DBar(nel, L, A, P, alpha=alpha_optimal, beta=beta)
u_opt = asm_final.forward_solve(E_opt)

# Compute strain from displacement (for strain plot)
strain_coord = np.linspace(0, L, (nstrain - 1) * nel + 1)

# Calculate relative errors (using E_opt from L-curve)
E_error_bubnov = np.abs(E_bubnov - E_true) / E_true
E_error_supg = np.abs(E_supg - E_true) / E_true
E_error_asm = np.abs(E_opt - E_true) / E_true

# Print errors
print(f"\nRelative error in E (Bubnov): {np.linalg.norm(E_bubnov - E_true) / np.linalg.norm(E_true):.6e}")
print(f"Relative error in E (SUPG): {np.linalg.norm(E_supg - E_true) / np.linalg.norm(E_true):.6e}")
print(f"Relative error in E (ASM): {np.linalg.norm(E_opt - E_true) / np.linalg.norm(E_true):.6e}")

# Figure 0: L-curve
fig0, ax0 = plt.subplots(1, 1, figsize=(10, 8))
ax0.loglog(residual_norms, regularization_norms, 'b-o', linewidth=2, markersize=8, label='L-curve')
ax0.loglog(residual_norms[optimal_idx], regularization_norms[optimal_idx], 'r*',
           markersize=20, label=f'Optimal α = {alpha_optimal:.2e}')

# Annotate some alpha values
for i in [0, optimal_idx, n_alpha-1]:
    ax0.annotate(f'α={alpha_values[i]:.1e}',
                xy=(residual_norms[i], regularization_norms[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax0.set_xlabel('Residual Norm ||ε_computed - ε_measured||', fontsize=13)
ax0.set_ylabel('Regularization Norm ||dE/dx||', fontsize=13)
ax0.set_title('L-curve for Regularization Parameter Selection', fontsize=14)
ax0.legend(fontsize=12, loc='best')
ax0.grid(True, which='both', alpha=0.3)
plt.tight_layout()

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

# Figure 4: ASM Convergence History for optimal alpha
# Need to get the history for optimal alpha
print(f"\nRe-running ASM with optimal alpha to get convergence history...")
asm_final = AdjointStateMethod1DBar(nel, L, A, P, alpha=alpha_optimal, beta=beta)
E_opt_final, history = asm_final.optimize(
    strain_noise,
    E_init,
    E_min,
    method='L-BFGS-B',
    max_iter=1000,
    tol=tol
)

# Update E_opt and u_opt with final results
E_opt = E_opt_final
u_opt = asm_final.forward_solve(E_opt)
E_error_asm = np.abs(E_opt - E_true) / E_true

print(f"\nFinal relative error in E (ASM): {np.linalg.norm(E_opt - E_true) / np.linalg.norm(E_true):.6e}")

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
# fig0.savefig(f'{filename_base}_Lcurve.png', dpi=300, bbox_inches='tight')
# fig0.savefig(f'{filename_base}_Lcurve.tif', dpi=300, bbox_inches='tight')
# fig1.savefig(f'{filename_base}_methods.png', dpi=300, bbox_inches='tight')
# fig1.savefig(f'{filename_base}_methods.tif', dpi=300, bbox_inches='tight')
# fig2.savefig(f'{filename_base}_errors.png', dpi=300, bbox_inches='tight')
# fig2.savefig(f'{filename_base}_errors.tif', dpi=300, bbox_inches='tight')
# fig3.savefig(f'{filename_base}_data_comparison.png', dpi=300, bbox_inches='tight')
# fig3.savefig(f'{filename_base}_data_comparison.tif', dpi=300, bbox_inches='tight')
# fig4.savefig(f'{filename_base}_convergence.png', dpi=300, bbox_inches='tight')
# fig4.savefig(f'{filename_base}_convergence.tif', dpi=300, bbox_inches='tight')

plt.show()