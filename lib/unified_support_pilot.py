"""Prepare or solve one controlled material-support/solver pilot method.

All methods use the same fixed cut quadrature and original log-coordinate
stationarity check. No gamma selection is performed with three pilot points.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from fgm_asm.fixed_region_material import FixedRegionMaterialMap
from fgm_asm.local_region_material import LocalRegionMaterialMap
from fgm_asm.material_preconditioning import MeanFluctuationMetric, solve_projected_lbfgs
from fgm_asm.integration import rectangular_quadrature
from fgm_asm.mesh import MeshInfo
from fgm_asm.unified_inverse import UnifiedObjective, solve_unified
from .config_utils import configuration_fingerprint, load_python_configuration
from .research_reporting import save_run
from . import unified_coarse_lcurve as shared
from .unified_gradient_layered_study import read_json, write_json, write_csv

ROOT = Path(__file__).resolve().parents[1]


def hashes():
    extra = ('fgm_asm/local_region_material.py', 'fgm_asm/material_preconditioning.py',
             'lib/unified_support_pilot.py')
    return dict(shared.hashes(), **{name: shared.digest(ROOT/name) for name in extra})


def prepare(cfg):
    if Path(cfg.CASE_OUTPUT).exists():
        raise FileExistsError('Pilot exists; change RUN_NAME instead of overwriting')
    requested = configuration_fingerprint(cfg)
    source = Path(cfg.REFERENCE_RUN)
    meta = read_json(source/'01_prepared/run_metadata.json')
    old = meta['configuration']
    for name in ('GEO_L', 'GEO_H', 'THICKNESS', 'BOUNDARIES', 'BEARING_WIDTH', 'FORCE', 'CALIBRATION'):
        setattr(cfg, name, old[name])
    initial = read_json(source/'03_inverse/380N/initialization.json')
    cfg.UNIFORM_INITIAL_MODULUS = initial['uniform_modulus_MPa']
    with np.load(source/'01_prepared/fields.npz') as saved:
        arrays = {name: saved[name] for name in ('xy', 'data', 'areas')}
    with np.load(source/'03_inverse/380N/initialization.npz') as saved:
        arrays['initial_region_log_coefficients'] = saved['theta']
    save_run(Path(cfg.CASE_OUTPUT)/'01_prepared', cfg,
        dict(status='prepared', requested_configuration_sha256=requested, kernel_sha256=hashes(),
             source_metadata_sha256=shared.digest(source/'01_prepared/run_metadata.json'),
             source_data_sha256=shared.digest(source/'01_prepared/fields.npz'),
             source_initialization_sha256=shared.digest(source/'03_inverse/380N/initialization.npz'),
             initialization='Same saved 380N data-only region-constant coefficients for every method',
             initialization_first_order_passed=initial['first_order_passed'],
             force_is_assumed=True, source_force_pairing=meta['force_pairing'],
             scope='Fixed-interface support and solver pilot, not selected gamma or validated E'), arrays)
    out = Path(cfg.CASE_OUTPUT)
    write_json(out/'01_prepared/inputs.json', dict(fields_sha256=shared.digest(out/'01_prepared/fields.npz')))
    print('保存位置：', out, flush=True)


def load(config):
    cfg = load_python_configuration(config)
    root = Path(cfg.CASE_OUTPUT)
    meta = read_json(root/'01_prepared/run_metadata.json')
    if configuration_fingerprint(cfg) != meta['requested_configuration_sha256'] or hashes() != meta['kernel_sha256']:
        raise ValueError('Pilot configuration/kernel differs from frozen preparation')
    if shared.digest(root/'01_prepared/fields.npz') != read_json(root/'01_prepared/inputs.json')['fields_sha256']:
        raise ValueError('Pilot observation/initial-state identity mismatch')
    return root, SimpleNamespace(**meta['configuration'])


def compact_quadrature(m):
    """Remove globally zero-weight padding; keep every nonzero integration point."""
    keep = np.any(m.gauss_abs_det_j > 0, axis=1)
    for name in ('gauss_N', 'gauss_dNdx', 'gauss_dNdy', 'gauss_w', 'gauss_det_j',
                 'gauss_abs_det_j', 'B_gauss', 'gauss_mass_local'):
        setattr(m, name, getattr(m, name)[keep])
    m.quadrature_metadata = dict(m.quadrature_metadata,
         unpadded_points_per_element=int(keep.sum()), zero_weight_padding_only=True)
    return m


def build(cfg, data, method):
    m = MeshInfo(cfg.GEO_L, cfg.GEO_H, *cfg.ELEMENTS)
    m = compact_quadrature(rectangular_quadrature(m, cfg.QUADRATURE_ORDER,
                            cfg.QUADRATURE_SUBDIVISIONS, tuple(cfg.BOUNDARIES)))
    full = FixedRegionMaterialMap(m, n_regions=cfg.N_REGIONS,
               interfaces=np.array(cfg.BOUNDARIES)[:, None], epsilon=cfg.EPSILON_MM,
               sharp=method != 'smooth_full')
    local = method in ('sharp_local', 'local_projected', 'local_metric')
    parameter_map = LocalRegionMaterialMap(full) if local else full
    full_theta = np.repeat(data['initial_region_log_coefficients'], m.n_nod)
    theta = full_theta[parameter_map.active_indices] if local else full_theta
    case = shared.case_from_arrays(m, cfg, data['xy'], data['data'], data['areas'], '380N')
    return parameter_map, case, theta


def mean_metric(parameter_map, case, theta, cfg):
    """Data-only GN curvature of physical regional-constant coefficient changes."""
    start = time.perf_counter()
    objective = UnifiedObjective(parameter_map, case, nu=cfg.NU, gamma=0)
    jacobian = []
    h = cfg.MEAN_FD_LOG_STEP
    for segment in parameter_map.slices:
        if np.ptp(theta[segment]) > 1e-12:
            raise ValueError('Coarse metric setup requires regional-constant initial coefficients')
        plus, minus = theta.copy(), theta.copy()
        plus[segment] += h; minus[segment] -= h
        a = objective.evaluate(plus)['prediction']
        b = objective.evaluate(minus)['prediction']
        jacobian.append((a-b)/(2*h*np.exp(theta[segment][0])))
    J = np.column_stack(jacobian)
    H = J.T @ (case.observation.weight_vector[:, None]*J)
    metric = MeanFluctuationMetric(parameter_map, H)
    evidence = dict(setup_forward_evaluations=objective.evaluations,
        setup_adjoint_evaluations=objective.evaluations,
        elapsed_seconds=time.perf_counter()-start,
        data_mean_hessian_eigenvalues=metric.eigenvalues.tolist(),
        data_mean_hessian_condition=float(metric.eigenvalues[-1]/metric.eigenvalues[0]),
        auxiliary_row_sum_correction_max=metric.row_sum_correction_max,
        mean_definition='Physical area-weighted regional coefficient-field mean; fluctuations have zero mean',
        objective_changed=False, original_log_box_and_projected_gradient_retained=True)
    return metric, H, evidence


def solve_method(root, cfg, method):
    method_start = time.perf_counter()
    out = root/method
    out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():
        raise FileExistsError('Completed method exists; inspect or replot saved fields')
    with np.load(root/'01_prepared/fields.npz') as saved:
        data = {name: saved[name] for name in saved.files}
    parameter_map, case, theta0 = build(cfg, data, method)
    metric, setup = None, dict(setup_forward_evaluations=0, elapsed_seconds=0.)
    if method == 'local_metric':
        metric, H, setup = mean_metric(parameter_map, case, theta0, cfg)
        write_json(out/'metric_setup.json', setup)
        np.savez_compressed(out/'metric_setup.npz', mean_hessian=H)
    records = []
    for index, gamma in enumerate(cfg.GAMMAS):
        folder = out/f'gamma_{index:02d}'
        if (folder/'summary.json').exists():
            records.append(read_json(folder/'summary.json')); continue
        folder.mkdir(exist_ok=True)
        print(f'{method} {index+1}/3 gamma={gamma:.12g}, n={parameter_map.n_material}', flush=True)
        progress = lambda r: print(f'{method}/{index}: FE={r["evaluation"]}, J={r["value"]:.9g}, g={r["gradient_inf"]:.3g}', flush=True)
        if method in ('local_projected', 'local_metric'):
            state, evidence, history = solve_projected_lbfgs(parameter_map, case, theta0,
                gamma=gamma, nu=cfg.NU, modulus_bounds=cfg.MODULUS_BOUNDS_MPA,
                max_iterations=cfg.MAX_ITERATIONS, max_evaluations=cfg.MAX_EVALUATIONS,
                gtol=cfg.GTOL, metric=metric, progress=progress)
        else:
            state, evidence, history = solve_unified(parameter_map, case, cfg.UNIFORM_INITIAL_MODULUS,
                initial_parameters=theta0, gamma=gamma, nu=cfg.NU,
                modulus_bounds=cfg.MODULUS_BOUNDS_MPA, max_iterations=cfg.MAX_ITERATIONS,
                max_evaluations=cfg.MAX_EVALUATIONS, ftol=cfg.FTOL, gtol=cfg.GTOL, progress=progress)
        np.savez_compressed(folder/'fields.npz', **{name: state[name] for name in
            ('theta', 'U', 'prediction', 'residual', 'gradient')})
        write_csv(folder/'history.csv', history)
        residual = state['residual'].reshape(-1, 2)
        row = dict(method=method, index=index, gamma=gamma, parameters=parameter_map.n_material,
            residual_norm=float(np.sqrt(2*state['data_cost'])),
            regularization_norm=float(np.sqrt(state['regularization'])),
            data_cost=state['data_cost'], regularization=state['regularization'],
            component_rmse_um=(1000*np.sqrt(np.sum(data['areas'][:, None]*residual**2, axis=0)/data['areas'].sum())).tolist(),
            selected_gamma=False, **evidence)
        write_json(folder/'summary.json', row)
        records.append(row)
    write_json(out/'summary.json', dict(method=method, records=records, setup=setup,
          total_solve_forward_evaluations=sum(r['forward_evaluations'] for r in records),
          elapsed_seconds_total=time.perf_counter()-method_start,
          kernel_sha256=hashes(), scope='Three fixed pilot gamma values; no parameter selection'))
    print('Completed', method, [(r['first_order_passed'], r['forward_evaluations'], r['projected_gradient_inf']) for r in records], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'solve'))
    parser.add_argument('--config', default='unified_support_pilot_config')
    parser.add_argument('--method', choices=('smooth_full', 'sharp_full', 'sharp_local', 'local_projected', 'local_metric'))
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(load_python_configuration(args.config)); return
    if args.method is None:
        parser.error('--method is required')
    root, cfg = load(args.config)
    solve_method(root, cfg, args.method)


if __name__ == '__main__':
    main()
