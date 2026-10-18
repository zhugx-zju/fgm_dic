"""Fair synthetic comparison: independent repeat displacement selects gamma.

Run from the repository root: python -m script.benchmark_tgv_regularization ...
True E is used only for final scoring/plots, never for selecting parameters.
Existing raw runs are resumed only if the complete protocol matches exactly.
"""

import argparse
import json
import platform
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

import config
from fgm_asm.fem_forward import fem_assemble, forward_solver
from fgm_asm.inverse_solver import lbfgs_inverse_solver_scipy
from fgm_asm.material import MaterialInfo
from fgm_asm.mesh import MeshInfo, setup_boundary_conditions
from fgm_asm.utils import add_noise_to_displacement


METHODS = ('gradient', 'affine_detrended', 'tgv')
LABELS = dict(gradient='Gradient Tikhonov', affine_detrended='Affine detrended', tgv='Huber TGV2')
COLORS = dict(gradient='#0072B2', affine_detrended='#D55E00', tgv='#009E73')


def truth_field(mesh, case):
    t = mesh.X / mesh.geo_l
    if case == 'linear':
        return 1 + t
    if case == 'exponential':
        return np.exp(np.log(2) * t)
    if case == 'layered':
        # A prescribed seven-zone challenge on the rectangle, not dogbone geometry.
        values = np.array([1., 1.4, 1.2, 1.8, 1.5, 2., 1.7])
        return values[np.minimum((7*t).astype(int), 6)]
    raise ValueError(case)


def mass_cost(mesh, residual):
    return float(0.5 * residual @ (mesh.M @ residual))


def metrics(mesh, estimate, truth):
    err = 100*np.abs(estimate-truth)/truth
    field = estimate.reshape(mesh.nods_y, mesh.nods_x)
    return dict(full_mae_pct=float(err.mean()), max_error_pct=float(err.max()),
                left_mae_pct=float(err[mesh.X <= 10].mean()),
                right_mae_pct=float(err[mesh.X >= 100].mean()),
                transverse_rms_pct=float(100*np.sqrt(np.mean((field-field.mean(axis=0))**2))/truth.mean()))


def figures(root, mesh, case, truth, results, seeds):
    # Show first predeclared test seed for every case; no selection by appearance.
    estimates = {mode: results[(mode, seeds[0])]['E_final'] for mode in METHODS}
    extent = (0, mesh.geo_l, 0, mesh.geo_h)
    fields = {'Truth': truth, **{LABELS[m]: e for m, e in estimates.items()}}
    vmin = min(e.min() for e in fields.values())
    vmax = max(e.max() for e in fields.values())
    emax = max(np.max(100*np.abs(e-truth)/truth) for e in estimates.values())
    fig, axes = plt.subplots(2, 4, figsize=(14, 4.9), constrained_layout=True)
    for col, (label, e) in enumerate(fields.items()):
        cm = axes[0, col].imshow(e.reshape(mesh.nods_y, mesh.nods_x), origin='lower',
                                 extent=extent, aspect='equal', vmin=vmin, vmax=vmax,
                                 interpolation='nearest', cmap='viridis')
        axes[0, col].set_title(label)
        ce = axes[1, col].imshow((100*np.abs(e-truth)/truth).reshape(mesh.nods_y, mesh.nods_x),
                                 origin='lower', extent=extent, aspect='equal', vmin=0, vmax=emax,
                                 interpolation='nearest', cmap='magma')
        axes[1, col].set_title(f'MAE = {np.mean(100*np.abs(e-truth)/truth):.3f}%')
        for ax in axes[:, col]:
            ax.set(xlabel='x (mm)', ylabel='y (mm)')
    fig.colorbar(cm, ax=axes[0, :], label='E (MPa)', shrink=.7)
    fig.colorbar(ce, ax=axes[1, :], label='Absolute relative error (%)', shrink=.7)
    fig.suptitle(f'{case}; test seed {seeds[0]}; identical data and independent nodal E(x,y)')
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'{case}_fields.{ext}', dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    x = mesh.plot_x[0]
    axes[0].plot(x, truth.reshape(mesh.nods_y, -1).mean(axis=0), 'k--', label='Truth')
    for mode in METHODS:
        profiles = np.array([results[(mode, seed)]['E_final'].reshape(mesh.nods_y, -1).mean(axis=0)
                             for seed in seeds])
        axes[0].plot(x, profiles.mean(axis=0), label=LABELS[mode], color=COLORS[mode])
        axes[0].fill_between(x, profiles.min(axis=0), profiles.max(axis=0), color=COLORS[mode], alpha=.15)
        errors = [metrics(mesh, results[(mode, seed)]['E_final'], truth)['full_mae_pct'] for seed in seeds]
        axes[1].plot(seeds, errors, 'o-', label=LABELS[mode], color=COLORS[mode])
    axes[0].set(xlabel='x (mm)', ylabel='Mean over y: E (MPa)', title='Range across test noise seeds')
    axes[1].set(xlabel='Test noise seed', ylabel='Full field MAE (%)', title='Independent test repetitions')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'{case}_profiles.{ext}', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nel-x', type=int, default=55)
    parser.add_argument('--nel-y', type=int, default=10)
    parser.add_argument('--noise', type=float, default=.01)
    parser.add_argument('--gammas', type=float, nargs='+', default=[1e-7, 1e-6, 1e-5])
    parser.add_argument('--cases', nargs='+', choices=['linear', 'exponential', 'layered'],
                        default=['linear', 'exponential', 'layered'])
    parser.add_argument('--test-seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--max-iter', type=int, default=2000)
    parser.add_argument('--max-fun', type=int, default=2400)
    parser.add_argument('--phase', choices=['all', 'calibrate', 'evaluate', 'plot'], default='all')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.nel_x < 2 or args.nel_y < 2 or args.noise < 0 or any(g < 0 for g in args.gammas):
        parser.error('mesh dimensions >=2, noise and gamma >=0 required')
    if set(args.test_seeds) & {40, 1040}:
        parser.error('Test seeds must differ from calibration seed 40 and repeat seed 1040')
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    fc = config.get_forward_config()
    protocol = dict(mesh=[args.nel_x, args.nel_y], geometry_mm=[fc.geo_l, fc.geo_h],
                    thickness_mm=fc.thickness, force_N_each_end=fc.f_tot, nu=fc.nu,
                    noise=args.noise, gammas=args.gammas, cases=args.cases,
                    methods=METHODS, calibration_seeds=[40, 1040], test_seeds=args.test_seeds,
                    max_iter=args.max_iter, max_fun=args.max_fun, ftol=1e-30, gtol=1e-12,
                    initial_modulus_MPa=.501,
                    tgv=dict(length_scale=20., modulus_scale=1., ratio=2., delta=.02),
                    selection='minimum independent repeat mass-weighted displacement error; no E truth',
                    python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    platform=platform.platform(),
                    base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    # Convert tuples to JSON lists before comparison.
    protocol = json.loads(json.dumps(protocol))
    manifest = root/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        parser.error('Existing protocol differs; use another output directory')
    manifest.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    mesh = MeshInfo(fc.geo_l, fc.geo_h, args.nel_x, args.nel_y)
    mesh.assemble_mass_matrix()
    bc = setup_boundary_conditions(mesh, fc.geo_l, fc.geo_h, fc.f_tot, fc.thickness)
    summary = {}
    for case in args.cases:
        folder = root/case
        folder.mkdir(exist_ok=True)
        truth = truth_field(mesh, case)
        material = MaterialInfo(nu=fc.nu)
        material.update(truth)
        clean = forward_solver(fem_assemble(mesh, material, bc))
        validation = add_noise_to_displacement(clean, args.noise, 1040)

        def run(mode, gamma, seed, label):
            path = folder/f'{label}_{mode}.npz'
            details_path = path.with_suffix('.json')
            if path.exists() and details_path.exists():
                return dict(np.load(path)), json.loads(details_path.read_text())
            measured = add_noise_to_displacement(clean, args.noise, seed)
            start = perf_counter()
            with path.with_suffix('.log').open('w', encoding='utf-8') as log, redirect_stdout(log):
                result = lbfgs_inverse_solver_scipy(
                    mesh, bc, measured, gamma=gamma, max_iter=args.max_iter, max_fun=args.max_fun,
                    ftol=1e-30, gtol=1e-12, nu=fc.nu, regularization_mode=mode,
                    tgv_options=protocol['tgv'])
            arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
            arrays.update(truth=truth, x=mesh.X, y=mesh.Y, u_measured=measured)
            details = dict(gamma=gamma, seed=seed, mode=mode, seconds=perf_counter()-start,
                           converged=result['converged'], message=result['message'],
                           iterations=result['n_iterations'], evaluations=result['n_evaluations'],
                           final_cost=result['final_cost'],
                           joint_gradient_norm=float(np.linalg.norm(result.get('joint_gradient', result['final_gradient']))),
                           validation_displacement_cost=mass_cost(mesh, result['U_final']-validation))
            np.savez_compressed(path, **arrays)
            details_path.write_text(json.dumps(details, indent=2), encoding='utf-8')
            print(f'{case} {label} {mode}: {details}', flush=True)
            return arrays, details

        selection_path = folder/'selection.json'
        if args.phase in ('all', 'calibrate'):
            selection = {}
            for mode in METHODS:
                candidates = []
                for index, gamma in enumerate(args.gammas):
                    _, details = run(mode, gamma, 40, f'calibration_{index}')
                    candidates.append(details)
                best = min(candidates, key=lambda d: d['validation_displacement_cost'])
                selection[mode] = dict(gamma=best['gamma'], candidates=candidates)
            selection_path.write_text(json.dumps(selection, indent=2), encoding='utf-8')
        if args.phase == 'calibrate':
            continue
        selection = json.loads(selection_path.read_text())
        results = {}
        rows = {}
        for mode in METHODS:
            rows[mode] = dict(selected_gamma=selection[mode]['gamma'], repetitions=[])
            for seed in args.test_seeds:
                if args.phase == 'plot':
                    path = folder/f'test_{seed}_{mode}.npz'
                    arrays, details = dict(np.load(path)), json.loads(path.with_suffix('.json').read_text())
                else:
                    arrays, details = run(mode, selection[mode]['gamma'], seed, f'test_{seed}')
                results[(mode, seed)] = arrays
                rows[mode]['repetitions'].append({**details, **metrics(mesh, arrays['E_final'], truth)})
            for metric in ('full_mae_pct', 'left_mae_pct', 'right_mae_pct', 'transverse_rms_pct', 'seconds'):
                values = [r[metric] for r in rows[mode]['repetitions']]
                rows[mode][metric] = dict(mean=float(np.mean(values)), min=float(np.min(values)), max=float(np.max(values)))
        summary[case] = rows
        figures(root, mesh, case, truth, results, args.test_seeds)
        (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'Completed {case}', flush=True)


if __name__ == '__main__':
    main()
