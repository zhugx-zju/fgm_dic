"""Fixed-gamma real-DIC mesh diagnostic, separate from a new L-curve scan.

Reuse the existing FE, observation, adjoint and certified exact-TV solver.
Neither calibration nor held-out residuals choose the gamma or initialization.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from fgm_asm.exact_tv import FEDataOracle, QuadratureGradient
from fgm_asm.load_case import evaluate_observation
from fgm_asm.research import solve_forward
from .dic_case_workflow import (load_prepared, _metrics, _optimize, _regional_comparison,
                                _solver_summary, _tv_arrays, _save_tv_evidence)
from .dic_exact_tv_report import _audit_field, _profile
from .experiment_data import file_sha256
from .research_reporting import save_run


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prolong_field(coarse, fine, values):
    """Exact Q4 prolongation on a rectangular nested grid, without layer ties."""
    interpolate = RegularGridInterpolator((coarse.plot_y[:, 0], coarse.plot_x[0]),
        np.asarray(values).reshape(coarse.nods_y, coarse.nods_x), bounds_error=True)
    return interpolate(fine.coord[:, ::-1])


def prolong_dual(coarse, fine, dual, gamma):
    """Interpolate at Gauss points; constant edge extension and ball projection."""
    x = np.einsum('en,gn->ge', coarse.x_ele, coarse.gauss_N).ravel()
    y = np.einsum('en,gn->ge', coarse.y_ele, coarse.gauss_N).ravel()
    xs, ys = np.unique(x), np.unique(y)
    if len(xs)*len(ys) != len(x):
        raise ValueError('Expected a tensor-product Gauss grid')
    grid = np.empty((len(ys), len(xs), 2))
    grid[np.searchsorted(ys, y), np.searchsorted(xs, x)] = dual
    fx = np.einsum('en,gn->ge', fine.x_ele, fine.gauss_N).ravel()
    fy = np.einsum('en,gn->ge', fine.y_ele, fine.gauss_N).ravel()
    xy = np.column_stack([np.clip(fy, ys[0], ys[-1]), np.clip(fx, xs[0], xs[-1])])
    result = RegularGridInterpolator((ys, xs), grid)(xy)
    return result*np.minimum(1., gamma/np.maximum(np.linalg.norm(result, axis=1), 1e-300))[:, None]


def _context(root, coarse_root):
    cfg, mesh, cases, meta, provenance = load_prepared(root/'01_prepared')
    old_cfg, old_mesh, old_cases, old_meta, _ = load_prepared(coarse_root/'01_prepared')
    if cfg.REGULARIZATION_MODE != 'exact_tv' or old_cfg.REGULARIZATION_MODE != 'exact_tv':
        raise ValueError('Both meshes must use exact TV')
    for key in ('GEO_L', 'GEO_H', 'THICKNESS', 'PRIMARY_NU', 'FIT_CASES', 'PREDICT_CASES',
                'REGION_BOUNDARIES_X', 'DISPLACEMENT_COMPONENT_WEIGHTS', 'EXACT_TV_OPTIONS', 'MAX_FUN'):
        if getattr(cfg, key) != getattr(old_cfg, key):
            raise ValueError(f'Mesh comparison differs in {key}')
    if (mesh.nel_x, mesh.nel_y) != (2*old_mesh.nel_x, 2*old_mesh.nel_y):
        raise ValueError('This diagnostic requires nested refinement by two in both directions')
    if cfg.EXACT_TV_PRIMAL_STEP != 4*old_cfg.EXACT_TV_PRIMAL_STEP:
        raise ValueError('Expected the declared quarter-area numerical step scaling')
    if meta['force_pairing'] != old_meta['force_pairing'] or meta['calibration'] != old_meta['calibration']:
        raise ValueError('Forces or independent calibration differ')
    with np.load(root/'01_prepared/fields.npz') as new, np.load(coarse_root/'01_prepared/fields.npz') as old:
        if new.files != old.files:
            raise ValueError('Prepared observation arrays differ')
        for key in new.files:
            np.testing.assert_array_equal(new[key], old[key])
    inverse = _read(coarse_root/'03_inverse/run_metadata.json')
    if inverse['status'] != 'computed' or not any(
            s['nu'] == cfg.PRIMARY_NU and s['converged'] for s in inverse['solvers']):
        raise ValueError('Need a certified coarse primary solution')
    source = coarse_root/'03_inverse'/f'nu_{cfg.PRIMARY_NU:.2f}'/'fields.npz'
    with np.load(source) as saved:
        old_field, old_dual = saved['E_inverse'].copy(), saved['tv_dual'].copy()
    return cfg, mesh, cases, meta, provenance, old_mesh, old_field, old_dual, inverse['selected_gamma'], source


def preflight(root, coarse_root):
    cfg, mesh, cases, meta, provenance, old_mesh, old_field, old_dual, gamma, source = _context(root, coarse_root)
    field = prolong_field(old_mesh, mesh, old_field)
    dual = prolong_dual(old_mesh, mesh, old_dual, gamma)
    output = root/'mesh_diagnostic'
    protocol = dict(workflow='dic_tv_fixed_gamma_mesh_diagnostic', status='preflight',
        coarse_root=str(coarse_root.resolve()), gamma=gamma, gamma_selection='coarse 180N L-curve; NOT reselected on refined mesh',
        nu=cfg.PRIMARY_NU, coarse_elements=[old_mesh.nel_x, old_mesh.nel_y],
        refined_elements=[mesh.nel_x, mesh.nel_y], coarse_parameters=old_mesh.n_nod, refined_parameters=mesh.n_nod,
        initialization='exact nested Q4 prolongation of coarse certified E; interpolated/projected Gauss dual',
        uses_calibration_in_objective_or_initialization=False, uses_held_out_for_selection=False,
        objective='0.5*r.T@W_area@r + gamma*integral(sqrt(E_x^2+E_y^2))',
        displacement_and_modulus_mesh_both_refined=True, all_y_parameters_independent=True,
        solver_step_note='primal step x4 compensates quarter element area; objective and KKT tolerances unchanged',
        coarse_fields_sha256=file_sha256(source), **provenance)
    if output.exists():
        recorded = _read(output/'run_metadata.json')
        if (output/'solution').exists() or any(recorded.get(k) != v for k, v in protocol.items()):
            raise ValueError('Preflight retry must use the same untouched protocol')
        with np.load(output/'fields.npz') as saved:
            np.testing.assert_array_equal(saved['initial_E'], field)
            np.testing.assert_array_equal(saved['initial_dual'], dual)
    else:
        save_run(output, cfg, protocol, {'initial_E': field, 'initial_dual': dual})
    fit = [case for case in cases if case.name in cfg.FIT_CASES]
    oracle = FEDataOracle(mesh, fit, cfg.PRIMARY_NU)
    start = time.perf_counter()
    value, gradient = oracle(field)
    seconds = time.perf_counter()-start
    log_field = np.log(field)
    directions = [np.cos(np.pi*mesh.X/mesh.geo_l), np.sin(np.pi*mesh.X/mesh.geo_l)+.2*np.cos(np.pi*mesh.Y/mesh.geo_h)]
    errors = []
    for direction in directions:
        direction /= np.linalg.norm(direction)
        step = 1e-3
        finite = (oracle(np.exp(log_field+step*direction))[0]-oracle(np.exp(log_field-step*direction))[0])/(2*step)
        exact = (gradient*field) @ direction
        errors.append(float(abs(finite-exact)/max(abs(finite), abs(exact), 1e-20)))
    if max(errors) > 1e-4:
        raise AssertionError(f'Refined data gradient failed finite differences: {errors}')
    rng = np.random.default_rng(1237)
    op = QuadratureGradient(mesh)
    x, p = rng.normal(size=mesh.n_nod), rng.normal(size=dual.shape)
    transpose_error = abs(float(np.sum(op.area[:, None]*op.apply(x)*p)-x@op.adjoint(p)))
    assert transpose_error < 1e-9
    affine = 1500+3*old_mesh.X+7*old_mesh.Y
    np.testing.assert_allclose(prolong_field(old_mesh, mesh, affine), 1500+3*mesh.X+7*mesh.Y, atol=1e-10, rtol=0)
    nodal_u = np.column_stack([.1+.003*mesh.X-.004*mesh.Y, .2+.002*mesh.X+.001*mesh.Y]).ravel()
    observation = fit[0].observation
    xy = np.column_stack([observation.x_meas, observation.y_meas])
    # Match the existing H operator's roundoff convention at the upper edges.
    # This is a test expectation, not a change to DIC coordinates or H.
    h_xy = np.clip(xy, [0., 0.], [mesh.geo_l, mesh.geo_h])
    edge_tolerance = 1e-9*max(mesh.geo_l, mesh.geo_h, 1.)+1e-12
    for axis, bound in enumerate((mesh.geo_l, mesh.geo_h)):
        h_xy[np.abs(h_xy[:, axis]-bound) <= edge_tolerance, axis] = bound
    exact_u = np.column_stack([.1+.003*h_xy[:, 0]-.004*h_xy[:, 1], .2+.002*h_xy[:, 0]+.001*h_xy[:, 1]]).ravel()
    h_error = float(np.max(np.abs(observation.H@nodal_u-exact_u)))
    assert h_error < 1e-12
    uniform_u, _ = solve_forward(mesh, fit[0].bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, np.full(mesh.n_nod, 2000.))
    strain = fit[0].bc_info['tensile_end_force']/(mesh.geo_h*cfg.THICKNESS*2000.)
    exact_u = np.column_stack([strain*mesh.X, -cfg.PRIMARY_NU*strain*mesh.Y]).ravel()
    rigid = np.zeros((2*mesh.n_nod, 3))
    rigid[0::2, 0], rigid[1::2, 1] = 1., 1.
    rigid[0::2, 2], rigid[1::2, 2] = -mesh.Y, mesh.X
    delta = uniform_u-exact_u
    residual = delta-rigid@np.linalg.lstsq(rigid, delta, rcond=None)[0]
    analytic_error = float(np.max(np.abs(residual)))
    assert analytic_error < 1e-9
    checks = dict(observation_arrays_bitwise_identical=True, observation_affine_error_mm=h_error,
        existing_H_boundary_roundoff_snap_mm=float(np.max(np.abs(h_xy-xy))),
        uniform_uniaxial_analytic_max_error_mm=analytic_error, prolongation_affine_exact=True,
        gradient_fd_relative_errors=errors, gradient_transpose_absolute_error=transpose_error,
        oracle_seconds=seconds, initial_data_cost=value,
        coarse_TV=QuadratureGradient(old_mesh).value(old_field), prolonged_TV=op.value(field),
        verification_source_sha256=file_sha256(Path(__file__)))
    (output/'preflight.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


def solve(root):
    output = root/'mesh_diagnostic'
    protocol = _read(output/'run_metadata.json')
    if not (output/'preflight.json').exists():
        raise ValueError('Complete preflight first')
    coarse = Path(protocol['coarse_root'])
    cfg, mesh, cases, meta, provenance, old_mesh, old_field, _, gamma, source = _context(root, coarse)
    if file_sha256(source) != protocol['coarse_fields_sha256'] or any(protocol[k] != v for k, v in provenance.items()):
        raise ValueError('Frozen source changed')
    destination = output/'solution'
    if destination.exists():
        raise FileExistsError('Preserve the existing diagnostic solution')
    with np.load(output/'fields.npz') as saved:
        initial, dual = saved['initial_E'], saved['initial_dual']
    fit = [case for case in cases if case.name in cfg.FIT_CASES]
    start = time.perf_counter()
    result = _optimize(cfg, mesh, fit, gamma, cfg.PRIMARY_NU, initial, dual)
    elapsed = time.perf_counter()-start
    summary = dict(workflow=protocol['workflow'], status='computed' if result['converged'] else 'not_converged',
        solver=_solver_summary(result), gamma=gamma, nu=cfg.PRIMARY_NU, elapsed_seconds=elapsed,
        gamma_reselected=False, **provenance)
    save_run(destination, cfg, summary, {'E_inverse': result['E_final'], **_tv_arrays(result),
        'final_gradient': result['final_gradient'], 'cost_history': result['cost_history']})
    _save_tv_evidence(destination, result)
    audit = _audit_field(mesh, fit, cfg, QuadratureGradient(mesh), destination, gamma, cfg.PRIMARY_NU, result['converged'])
    assert audit['certificate_reproduced']
    (destination/'independent_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    rows, regions = [], []
    for current_mesh, current_cases, field, name in ((old_mesh, load_prepared(coarse/'01_prepared')[2], old_field, 'coarse'),
                                                    (mesh, cases, result['E_final'], 'refined')):
        for case in current_cases:
            u, _ = solve_forward(current_mesh, case.bc_info['tensile_end_force'], cfg.THICKNESS, cfg.PRIMARY_NU, field)
            rows.append(dict(mesh=name, case=case.name, role='fit' if case.name in cfg.FIT_CASES else 'prediction',
                             **_metrics(evaluate_observation(case, u))))
        frame = _regional_comparison(current_mesh, field, pd.DataFrame(meta['calibration']), cfg.REGION_BOUNDARIES_X)
        frame.insert(0, 'mesh', name)
        regions.append(frame)
    pd.DataFrame(rows).to_csv(output/'displacement_comparison.csv', index=False)
    pd.concat(regions).to_csv(output/'regional_comparison.csv', index=False)
    print(json.dumps(summary, ensure_ascii=False, default=str, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'solve'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--coarse', type=Path)
    args = parser.parse_args()
    if args.action == 'preflight':
        if args.coarse is None:
            parser.error('--coarse is required for preflight')
        preflight(args.root, args.coarse)
    else:
        solve(args.root)
