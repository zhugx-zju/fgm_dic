"""Isolated exact-TV verification, with no changes to the default DIC workflow.

Run an isolated scan: python -m lib.exact_tv_study scan
The truth is used for generation/evaluation only, never for initialization,
stopping, ADMM penalty adaptation, or L-curve selection.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.sparse import save_npz

from fgm_asm.exact_tv import (ExactTVOptions, FEDataOracle, QuadratureGradient,
                              solve_exact_tv)
from fgm_asm.mesh import MeshInfo
from fgm_asm.research import make_observation_case, solve_forward
from fgm_asm.l_curve import select_l_curve_corner
from lib.dic_case_workflow import homogeneous_dic_initial


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'results/dic_full/exact_tv_20261021'


def save_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert),
                    encoding='utf-8')


def truth_function(kind, length=30., height=8.):
    if kind == 'uniform':
        return lambda x, y: 1800.0 + np.zeros_like(x)
    if kind == 'smooth':
        return lambda x, y: 1800.0*(1+.2*np.cos(2*np.pi*x/length)+.1*np.cos(np.pi*y/height))
    if kind == 'step':
        return lambda x, y: np.where(x+.6*y < .45*length, 2400., 1400.)
    raise ValueError(kind)


def dataset(kind, nx=12, ny=4, seed=20261021, noise_mm=0.0002):
    """Two-dimensional truth on a 4x finer forward mesh, separate observations."""
    length, height, thickness, nu = 30.0, 8.0, 3.2, .26
    mesh, truth_mesh = MeshInfo(length, height, nx, ny), MeshInfo(length, height, 4*nx, 4*ny)
    truth = truth_function(kind, length, height)
    force = 119.77617870327234 * height/20.0
    u, _ = solve_forward(truth_mesh, force, thickness, nu, truth)
    weights = np.ones((truth_mesh.nods_y, truth_mesh.nods_x))
    weights[[0, -1], :] *= .5
    weights[:, [0, -1]] *= .5
    weights = weights.ravel() * truth_mesh.el * truth_mesh.eh
    rng = np.random.default_rng(seed)
    observed = u.reshape(-1, 2) + rng.normal(0, noise_mm, (truth_mesh.n_nod, 2))
    case = make_observation_case(mesh, '180N_synthetic', force, thickness,
                                 truth_mesh.coord, observed, weights)
    initial, provenance = homogeneous_dic_initial(mesh, [case], thickness, nu)
    return mesh, case, initial, truth(mesh.X, mesh.Y), dict(
        kind=kind, nx=nx, ny=ny, truth_nx=4*nx, truth_ny=4*ny, seed=seed,
        noise_std_mm=noise_mm, nu=nu, length_mm=length, height_mm=height,
        thickness_mm=thickness, specimen_width_mm=20., roi_end_force_N=force,
        initialization=provenance, independent_y_parameters=True,
        known_interfaces_in_inverse=False, truth_used_for_selection=False,
        noise_is_measured_DIC=False)


def solve_h1(mesh, case, initial, gamma, budget=12000):
    """Same data oracle/initial/budget; original gamma/2 integral |grad E|^2."""
    op, oracle = QuadratureGradient(mesh), FEDataOracle(mesh, [case], .26)

    def objective(m):
        e = np.exp(m)
        f, g = oracle(e)
        ge = op.apply(e)
        return f+.5*gamma*float(op.area @ np.sum(ge**2, axis=1)), (
            g+gamma*op.adjoint(ge))*e

    result = minimize(objective, np.log(initial), jac=True, method='L-BFGS-B', options={
        'maxiter': 2000, 'maxfun': budget-1, 'ftol': 0., 'gtol': 1e-9,
        'maxls': 40, 'maxcor': 20})
    value, grad = objective(result.x)
    e = np.exp(result.x)
    reg = float(op.area @ np.sum(op.apply(e)**2, axis=1))
    return dict(E=e, status=str(result.message), converged=bool(np.max(np.abs(grad)) <= 1e-9),
                evaluations=oracle.evaluations, state=oracle.last_state,
                objective=float(value), regularization=reg,
                stationarity=float(np.max(np.abs(grad))))


def field_metrics(mesh, case, field, kind):
    """Common area-weighted evaluation on the finer observation lattice."""
    obs = case.observation
    predicted = np.asarray(obs.H[::2, ::2] @ field).ravel()
    # Observation coordinates are explicit in the controlled synthetic lattice.
    xx, yy = np.meshgrid(np.linspace(0, mesh.geo_l, 4*mesh.nel_x+1),
                         np.linspace(0, mesh.geo_h, 4*mesh.nel_y+1))
    truth = truth_function(kind)(xx.ravel(), yy.ravel())
    weights = obs.weight_vector[::2]
    relative = np.sqrt(weights @ (predicted-truth)**2 / (weights @ truth**2))
    return dict(area_relative_rms=float(relative),
                field_min_MPa=float(field.min()), field_max_MPa=float(field.max()),
                field_std_MPa=float(np.std(field)))


def save_dataset(directory, kind, case, initial, metadata):
    """Freeze observations and H, so plotting needs no forward solve."""
    obs = case.observation
    np.savez_compressed(directory/f'{kind}_observations.npz',
        coordinates=np.column_stack((obs.x_meas, obs.y_meas)),
        observed=obs.u_measured, weights=obs.weight_vector, initial=initial)
    save_npz(directory/f'{kind}_H.npz', obs.H)
    save_json(directory/f'{kind}_data_metadata.json', metadata)


def evaluation_rules():
    return dict(declared_before_any_L_curve_selection=True,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_seeds=[20261023, 20261024], gamma_frozen_from_seed=20261022,
        required_selected_status_for_both_methods_and_both_fields=True,
        required_KKT_convergence_for_all_evaluation_runs=True,
        step_TV_error_max_ratio_to_H1_each_seed=.8,
        smooth_TV_error_max_ratio_to_H1_each_seed=1.2,
        uniform_false_structure_std_max_MPa=9.,
        uniform_gamma_source='same method gamma selected on step; never fitted to uniform truth',
        actual_DIC_if_gate_fails=False,
        thresholds_are_screening_rules_not_scientific_universal_constants=True)


def scan(output):
    """50-point selection pilot; no actual DIC until this gate has passed."""
    output.mkdir(parents=True, exist_ok=False)
    opts = ExactTVOptions()
    protocol = dict(stage='selection_pilot', inverse_mesh=[12, 4], truth_mesh=[48, 16],
        identification_seed=20261022, evaluation_seeds=[20261023, 20261024],
        noise_std_mm=.0002, options=asdict(opts), gamma_count=50,
        gamma_ranges=dict(exact_tv=[1e-12, 1e-6], h1=[1e-14, 1e-7]),
        objectives=dict(exact_tv='0.5*r.T@W_area@r + gamma*integral(norm(grad(E)))',
                        h1='0.5*r.T@W_area@r + 0.5*gamma*integral(norm(grad(E))**2)'),
        l_curve_ordinate=dict(exact_tv='TV(E)', h1='sqrt(integral(norm(grad(E))**2))'),
        same_initial_and_observation=True, independent_y_parameters=True,
        true_interface_is_oblique=True, truth_used_for_selection=False,
        gate='Both smooth and step must have resolved L-curve corners supported by KKT-qualified solves; otherwise stop before DIC.',
        source_sha256={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in ('fgm_asm/exact_tv.py','lib/exact_tv_study.py')},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    save_json(output/'protocol.json', protocol)
    save_json(output/'evaluation_gate.json', evaluation_rules())
    selections = []
    for kind in ('smooth', 'step'):
        mesh, case, initial, truth, meta = dataset(kind, seed=20261022)
        save_dataset(output, kind, case, initial, meta)
        for method, endpoints in protocol['gamma_ranges'].items():
            directory = output/f'{kind}_{method}'
            directory.mkdir()
            gamma = np.geomspace(*endpoints, 50)
            rows, fields = [], []
            for index, weight in enumerate(gamma):
                start = time.perf_counter()
                if method == 'exact_tv':
                    result = solve_exact_tv(mesh, [case], initial, weight, options=opts)
                    reg = QuadratureGradient(mesh).value(result['E'])
                    eta = reg
                    save_json(directory/f'history_{index:02d}.json', result['history'])
                    # Numerically constant fields are outside a log-TV curve.
                    norm_resolved = QuadratureGradient(mesh).rms(
                        QuadratureGradient(mesh).apply(result['E'])) > opts.primal_atol*10
                else:
                    result = solve_h1(mesh, case, initial, weight)
                    reg = result['regularization']
                    eta = np.sqrt(reg)
                    norm_resolved = True
                row = dict(index=index, gamma=float(weight), method=method, kind=kind,
                    status=result['status'], converged=result['converged'],
                    norm_resolved=bool(norm_resolved), evaluations=result['evaluations'],
                    elapsed_s=time.perf_counter()-start,
                    rho=float(np.sqrt(2*result['state']['cost_tar'])), eta=float(eta),
                    regularization=float(reg))
                if method == 'exact_tv':
                    row['certificate'] = result['history'][-1]
                else:
                    row['stationarity'] = result['stationarity']
                rows.append(row)
                fields.append(result['E'])
                save_json(directory/'scan.json', rows)
                np.savez_compressed(directory/'fields.npz', E=np.asarray(fields),
                                     truth=truth, initial=initial, coordinates=mesh.coord)
                print(kind, method, index, result['status'], 'eval', result['evaluations'], flush=True)
            selection = select_l_curve_corner(gamma, [r['rho'] for r in rows],
                [r['eta'] for r in rows], [r['converged'] and r['norm_resolved'] for r in rows])
            save_json(directory/'selection.json', selection)
            selected = dict(kind=kind, method=method, status=selection['status'],
                            selected_gamma=selection['selected_gamma'])
            # Evaluate truth only after selection is complete and frozen.
            if selection['selected_gamma'] is not None:
                index = selection['optimal_index']
                selected['metrics'] = field_metrics(mesh, case, fields[index], kind)
                selected['evaluations'] = rows[index]['evaluations']
                selected['eligible_count'] = int(np.sum(selection['eligible']))
            selections.append(selected)
            save_json(output/'selections.json', selections)
            print('SELECTION', selected, flush=True)
    gate = dict(selection_gate_passed=all(s['status'] == 'selected' for s in selections),
                actual_DIC_run=False, reason='Selection pilot only; physical DIC not used.')
    save_json(output/'gate.json', gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['scan'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or DEFAULT_OUTPUT/'selection_pilot'
    scan(destination.resolve())
