import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from asm.asm_modulus import AdjointStateMethod1DBar
from utils.data_generation import *
from utils.dim_solve import supg, bubnov

print("="*80)
print("ASM vs DIM Methods Comparison Test")
print("="*80)

# Parameters
L = 1.0
A = 1.0
P = 1e-2
E0 = 1.0
nel = 200
nstrain = 20

# Test different scenarios
scenarios = [
    {'noise': 0.0, 'alpha': 0.0, 'name': 'No noise, No reg'},
    {'noise': 0.01, 'alpha': 0.0, 'name': '1% noise, No reg'},
    {'noise': 0.05, 'alpha': 0.0, 'name': '5% noise, No reg'},
    {'noise': 0.01, 'alpha': 1e-10, 'name': '1% noise, alpha=1e-10'},
    {'noise': 0.01, 'alpha': 1e-8, 'name': '1% noise, alpha=1e-8'},
]

distribution_types = ['linear', 'quadratic', 'sinusoidal']

results = {}

for dist_type in distribution_types:
    print(f"\n{'='*80}")
    print(f"Testing: {dist_type} distribution")
    print(f"{'='*80}")

    results[dist_type] = {}

    # Create solver
    asm = AdjointStateMethod1DBar(nel, L, A, P)
    coord = asm.coord

    # True E
    E_true = generate_E_true(dist_type, coord, L, E0)

    # Generate strain data for DIM
    strain_exact = generate_strain_data(dist_type, coord, nel, nstrain, P, A, E0, L)

    for scenario in scenarios:
        noise_level = scenario['noise']
        alpha = scenario['alpha']
        name = scenario['name']

        print(f"\n{'-'*60}")
        print(f"Scenario: {name}")
        print(f"{'-'*60}")

        # Generate noisy data
        u_exact = generate_displacement_data(asm, E_true)
        u_noise = generate_noise_data(u_exact, noise_level, seed=42)
        strain_noise = generate_noise_data(strain_exact, noise_level, seed=42)

        # DIM methods (use strain)
        E_bubnov = bubnov(nstrain, nel, coord, strain_noise, P, L, A)
        E_supg = supg(nstrain, nel, coord, strain_noise, P, L, A)

        # ASM (use displacement)
        asm_solver = AdjointStateMethod1DBar(nel, L, A, P, alpha=alpha)
        E_init = np.ones(asm_solver.nNod) * E0 * 1.5
        E_min = 1e-4

        print(f"Running ASM optimization...")
        E_asm, history = asm_solver.optimize(
            u_noise,
            E_init,
            E_min=E_min,
            method='L-BFGS-B',
            max_iter=500,
            tol=1e-10
        )

        # Compute errors
        error_bubnov = np.linalg.norm(E_bubnov - E_true) / np.linalg.norm(E_true)
        error_supg = np.linalg.norm(E_supg - E_true) / np.linalg.norm(E_true)
        error_asm = np.linalg.norm(E_asm - E_true) / np.linalg.norm(E_true)

        print(f"\nRelative Errors:")
        print(f"  Bubnov-Galerkin: {error_bubnov:.6e}")
        print(f"  SUPG:            {error_supg:.6e}")
        print(f"  ASM:             {error_asm:.6e}")

        # Store results
        results[dist_type][name] = {
            'E_true': E_true,
            'E_bubnov': E_bubnov,
            'E_supg': E_supg,
            'E_asm': E_asm,
            'error_bubnov': error_bubnov,
            'error_supg': error_supg,
            'error_asm': error_asm,
            'history': history
        }

        # Quick analysis
        if error_asm < error_bubnov and error_asm < error_supg:
            print("  ✓ ASM is BEST!")
        elif error_asm < 1.5 * min(error_bubnov, error_supg):
            print("  ≈ ASM is comparable")
        else:
            print("  X ASM is worse")
            # Diagnose why
            print(f"\n  Diagnosis:")
            print(f"    - Converged: {asm_solver.converged}")
            print(f"    - Final J: {history['J'][-1]:.6e}")
            print(f"    - Iterations: {len(history['J'])}")
            print(f"    - Final ||grad||: {history['grad_norm'][-1]:.6e}")
            print(f"    - Data mismatch at convergence:")
            u_asm = asm_solver.forward_solve(E_asm)
            data_error = np.linalg.norm(u_asm - u_noise) / np.linalg.norm(u_noise)
            print(f"      ||u_asm - u_noise|| / ||u_noise|| = {data_error:.6e}")

print("\n" + "="*80)
print("Summary of Results")
print("="*80)

for dist_type in distribution_types:
    print(f"\n{dist_type.upper()} Distribution:")
    print("-" * 60)
    print(f"{'Scenario':<30} {'Bubnov':<12} {'SUPG':<12} {'ASM':<12} {'Winner':<10}")
    print("-" * 60)

    for name, res in results[dist_type].items():
        errors = [res['error_bubnov'], res['error_supg'], res['error_asm']]
        methods = ['Bubnov', 'SUPG', 'ASM']
        winner = methods[np.argmin(errors)]

        print(f"{name:<30} {res['error_bubnov']:<12.4e} {res['error_supg']:<12.4e} "
              f"{res['error_asm']:<12.4e} {winner:<10}")

print("\n" + "="*80)
print("Key Questions to Answer:")
print("="*80)
print("1. Is ASM worse ONLY with regularization (alpha > 0)?")
print("2. Is ASM worse ONLY with high noise?")
print("3. Is the initialization E_init = 1.5*E0 too far from truth?")
print("4. Does ASM need more iterations to converge?")
print("\nNext steps:")
print("- If ASM is worse WITHOUT regularization → problem in forward/adjoint/gradient")
print("- If ASM is worse ONLY WITH regularization → regularization parameter tuning")
print("- If ASM is worse for ALL cases → may need better initialization or constraints")
