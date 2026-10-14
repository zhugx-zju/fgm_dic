"""Controlled comparison of two regularizers on synthetic exponential E(x)."""

import argparse
import json
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from fgm_asm.fem_forward import fem_assemble, forward_solver
from fgm_asm.inverse_solver import lbfgs_inverse_solver_scipy
from fgm_asm.material import MaterialInfo
from fgm_asm.mesh import MeshInfo, setup_boundary_conditions
from fgm_asm.utils import add_noise_to_displacement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nel-x', type=int, default=55)
    parser.add_argument('--nel-y', type=int, default=10)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--e-right', type=float, default=2.0,
                        help='True E at x=110 mm in MPa; E at x=0 is 1 MPa')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=1e-6)
    parser.add_argument('--max-iter', type=int, default=2000)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if (args.nel_x < 2 or args.nel_y < 2 or args.noise < 0 or args.gamma < 0
            or not np.isfinite(args.e_right) or args.e_right <= 0):
        parser.error('mesh dimensions must be >=2; noise/gamma >=0; e-right >0')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output-dir must be new or empty to preserve prior experiment results')

    forward_config = config.get_forward_config()
    inverse_config = config.get_inverse_config()
    mesh = MeshInfo(forward_config.geo_l, forward_config.geo_h, args.nel_x, args.nel_y)
    truth = np.exp(np.log(args.e_right) * mesh.X / mesh.geo_l)
    material = MaterialInfo(nu=forward_config.nu)
    material.update(truth)
    bc = setup_boundary_conditions(
        mesh, mesh.geo_l, mesh.geo_h, forward_config.f_tot, forward_config.thickness,
    )
    clean_displacement = forward_solver(fem_assemble(mesh, material, bc))
    measured_displacement = add_noise_to_displacement(clean_displacement, args.noise, args.seed)
    mesh.assemble_mass_matrix()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    estimates = {}
    for mode in ('gradient', 'affine_detrended'):
        with (args.output_dir / f'{mode}.log').open('w', encoding='utf-8') as log:
            with redirect_stdout(log):
                result = lbfgs_inverse_solver_scipy(
                    mesh, bc, measured_displacement, gamma=args.gamma,
                    max_iter=args.max_iter, ftol=inverse_config.ftol,
                    gtol=inverse_config.gtol, nu=forward_config.nu,
                    regularization_mode=mode,
                )
        estimate = result['E_final']
        estimates[mode] = estimate
        error = 100 * np.abs(estimate - truth) / truth
        x = mesh.X
        estimate_field = estimate.reshape(mesh.nods_y, mesh.nods_x)
        transverse_deviation = estimate_field - estimate_field.mean(axis=0)
        rows[mode] = {
            'converged': bool(result['converged']),
            'iterations': int(result['n_iterations']),
            'message': result['message'],
            'full_mae_pct': float(error.mean()),
            'left_mae_pct': float(error[x <= 10].mean()),
            'interior_mae_pct': float(error[(x >= 20) & (x <= 90)].mean()),
            'right_mae_pct': float(error[x >= 100].mean()),
            'max_error_pct': float(error.max()),
            'transverse_rms_pct': float(
                100 * np.sqrt(np.mean(transverse_deviation**2)) / truth.mean()
            ),
            'final_cost': float(result['final_cost']),
            'data_cost': float(result['cost_tar_history'][-1]),
            'regularization_cost': float(result['cost_reg_history'][-1]),
        }
    metadata = {
        'truth': (f'E(x)=exp(log({args.e_right})*x/{mesh.geo_l} mm) MPa; '
                  f'E(0)=1 MPa; E({mesh.geo_l})={args.e_right} MPa'),
        'inverse': 'independent nodal E(x,y), common initial E=0.501 MPa',
        'mesh': [args.nel_x, args.nel_y],
        'force_N_each_end': forward_config.f_tot,
        'thickness_mm': forward_config.thickness,
        'nu': forward_config.nu,
        'noise_fraction': args.noise,
        'seed': args.seed,
        'gamma': args.gamma,
        'max_iter': args.max_iter,
        'ftol': inverse_config.ftol,
        'gtol': inverse_config.gtol,
        'results': rows,
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    np.savez_compressed(
        args.output_dir / 'fields.npz', truth=truth, x=mesh.X, y=mesh.Y,
        u_measured=measured_displacement, **estimates,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.8), constrained_layout=True)
    def reshape(values):
        return values.reshape(mesh.nods_y, mesh.nods_x)
    modulus_min = min(float(truth.min()), *(float(v.min()) for v in estimates.values()))
    modulus_max = max(float(truth.max()), *(float(v.max()) for v in estimates.values()))
    error_max = max(float((100 * np.abs(v - truth) / truth).max()) for v in estimates.values())
    for column, (mode, estimate) in enumerate(estimates.items()):
        color_modulus = axes[0, column].pcolormesh(
            mesh.plot_x, mesh.plot_y, reshape(estimate), cmap='viridis',
            shading='auto', vmin=modulus_min, vmax=modulus_max,
        )
        color_error = axes[1, column].pcolormesh(
            mesh.plot_x, mesh.plot_y, reshape(100 * np.abs(estimate - truth) / truth),
            cmap='magma', shading='auto', vmin=0, vmax=error_max,
        )
        axes[0, column].set_title(mode)
        axes[1, column].set_title(f"Absolute relative error; MAE {rows[mode]['full_mae_pct']:.3g}%")
        for row in range(2):
            axes[row, column].set_aspect('equal')
            axes[row, column].set(xlabel='x (mm)', ylabel='y (mm)')
    fig.colorbar(color_modulus, ax=axes[0, :], label='E (MPa)', shrink=0.8)
    fig.colorbar(color_error, ax=axes[1, :], label='Absolute relative error (%)', shrink=0.8)
    for extension in ('png', 'pdf'):
        fig.savefig(args.output_dir / f'field_comparison.{extension}', dpi=300)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
