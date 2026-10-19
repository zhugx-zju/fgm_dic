"""Check load proportionality and conditional inverse sensitivity after selection."""

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fgm_asm.research import compare_displacement, partition_modulus
from lib.dic_noise import _decompose
from lib.dic_revision_study import (assumed_registration, audit_inputs, make_cases, state_metrics)
from lib.experiment_data import file_sha256, load_calibration
from lib.research_reporting import (plot_displacement_comparison, save_run, save_x_profile)
from lib.research_workflows import _mesh, _regional_comparison, _solver_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dic_revision_config')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Sensitivity output must be new')
    cfg = importlib.import_module(args.config)
    metadata_path = args.source / 'inverse_primary' / 'run_metadata.json'
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    for key, digest in meta['source_sha256'].items():
        if key.replace('\\', '/').startswith('fgm_asm/') and file_sha256(cfg.PROJECT_ROOT / key) != digest:
            raise ValueError('FEM source differs from selected inference')
    gamma = meta['selected_gamma']
    all_platforms, forces, _, _ = audit_inputs(cfg)
    primary = all_platforms['_3']
    a, b = assumed_registration(primary['180N'].coordinates, cfg.GEO_L, cfg.GEO_H)
    xy = primary['180N'].coordinates @ a.T + b
    if not np.array_equal(primary['180N'].grid_ids, primary['280N'].grid_ids):
        raise ValueError('Load proportionality requires matched point IDs')
    ratio = forces['280N']['mean_corrected_force_N'] / forces['180N']['mean_corrected_force_N']
    first, second = [primary[name].displacement @ a.T for name in ('180N', '280N')]
    difference = second - ratio * first
    cleaned = _decompose(xy, np.stack([difference, -difference]), np.array([0., 1.]))['cleaned'][0]
    linearity = {'force_ratio': ratio, 'rmse_components_mm': np.sqrt(np.mean(cleaned**2, axis=0)),
                 'rms_both_components_mm': float(np.sqrt(np.mean(cleaned**2))),
                 'p95_components_mm': np.quantile(np.abs(cleaned), .95, axis=0),
                 'interpretation': '280N DIC minus scaled 180N DIC after three rigid modes; independent of FEM modulus and assumed zone order.'}
    affine = []
    design = np.column_stack([np.ones(len(xy)), xy - xy.mean(axis=0)])
    for name, u in [('180N', first), ('280N', second)]:
        coefficients = np.linalg.lstsq(design, u, rcond=None)[0]
        gradient = coefficients[1:].T
        affine.append({'case': name, 'global_axial_strain': gradient[0, 0],
                       'global_transverse_strain': gradient[1, 1],
                       'apparent_contraction_ratio': -gradient[1, 1] / gradient[0, 0],
                       'global_shear_strain': .5 * (gradient[0, 1] + gradient[1, 0])})
    calibration = load_calibration(cfg.CALIBRATION_CSV, cfg.CALIBRATION_METHOD, cfg.REGION_INFILL_PERCENT)
    reference = partition_modulus(calibration.mean_E_MPa, cfg.REGION_BOUNDARIES_X)
    save_run(args.output, cfg, {'workflow': 'conditional_dic_sensitivity',
                              'coordinates_verified': False, 'selected_gamma_fixed': gamma,
                              'source_metadata_sha256': file_sha256(metadata_path),
                              'load_proportionality': linearity, 'affine_diagnostics': affine},
             {'coordinates': xy, 'load_proportionality_residual': cleaned})
    pd.DataFrame(affine).to_csv(args.output / 'global_affine_diagnostics.csv', index=False)
    plot_displacement_comparison(args.output, 'model_independent_load_proportionality', xy, second,
                                 {'prediction': (second-cleaned).ravel()})
    revision_changes = []
    for revision in ('_2', '_3'):
        old, new = all_platforms['']['180N'], all_platforms[revision]['180N']
        old_frame = pd.DataFrame(np.column_stack([old.coordinates, old.displacement]),
                                 index=pd.MultiIndex.from_arrays(old.grid_ids.T))
        new_frame = pd.DataFrame(np.column_stack([new.coordinates, new.displacement]),
                                 index=pd.MultiIndex.from_arrays(new.grid_ids.T))
        common = old_frame.index.intersection(new_frame.index)
        q, p = old_frame.loc[common].to_numpy(), new_frame.loc[common].to_numpy()
        coord_fit = np.linalg.lstsq(np.column_stack([np.ones(len(q)), q[:, :2]]), p[:, :2], rcond=None)[0]
        residual = p[:, :2] - np.column_stack([np.ones(len(q)), q[:, :2]]) @ coord_fit
        revision_changes.append({'revision': revision, 'common_ids': len(common),
                                  'coordinate_fit_new_from_old': coord_fit.tolist(),
                                  'coordinate_fit_max_residual_mm': float(np.max(np.abs(residual))),
                                  'displacement_lsq_scale': float(np.sum(p[:, 2:]*q[:, 2:]) / np.sum(q[:, 2:]**2))})
    (args.output / 'revision_scaling.json').write_text(json.dumps(revision_changes, indent=2), encoding='utf-8')
    rows = []
    # A free 2D field can be reflected exactly with the data. This check makes
    # clear why inverse residuals cannot establish the physical infill direction.
    with np.load(args.source / 'inverse_primary' / 'fields.npz') as source_fields:
        primary_field = source_fields['E_inverse'].copy()
    mesh = _mesh(cfg)
    reversed_field = primary_field.reshape(mesh.nods_y, mesh.nods_x)[:, ::-1].ravel()
    reverse_a, reverse_b = assumed_registration(primary['180N'].coordinates, cfg.GEO_L, cfg.GEO_H, -1)
    reverse_cases = make_cases(cfg, mesh, primary, forces, reverse_a, reverse_b)
    reverse_states = compare_displacement(mesh, reversed_field, reverse_cases, cfg.NU)
    reverse_output = args.output / 'reverse_direction'
    save_run(reverse_output, cfg, {'coordinates_verified': False, 'variant': 'reverse_direction',
                                  'method': 'Exact reflection of the saved free 2D field; no optimization',
                                  'matrix': reverse_a, 'offset_mm': reverse_b},
             {'coordinates': mesh.coord, 'E_inverse': reversed_field})
    _regional_comparison(mesh, reversed_field, calibration, cfg.REGION_BOUNDARIES_X).to_csv(
        reverse_output / 'regional_modulus_comparison.csv', index=False)
    save_x_profile(reverse_output, mesh, reversed_field, reference)
    for state in reverse_states:
        rows.append(dict(variant='reverse_direction', case=state['name'], converged=True,
                         iterations=0, gamma=gamma, nu=cfg.NU, initial_MPa=cfg.E_REF,
                         parameters=mesh.n_nod, **state_metrics(state)))
    variants = [('shift_minus4', '_3', -4., 1700., .3, 35, 10),
                ('shift_plus4', '_3', 4., 1700., .3, 35, 10),
                ('revision2', '_2', 0., 1700., .3, 35, 10),
                ('initial1200', '_3', 0., 1200., .3, 35, 10),
                ('poisson045', '_3', 0., 1700., .45, 35, 10),
                ('coarse_mesh', '_3', 0., 1700., .3, 28, 8)]
    # Keep the regularization normalization at 1700 MPa even when initialization
    # changes; otherwise the initial-value check would also change the penalty.
    from fgm_asm import lbfgs_inverse_solver_scipy
    for label, revision, shift, initial, nu, nx, ny in variants:
        run_cfg = SimpleNamespace(**{k: v for k, v in vars(cfg).items() if k.isupper()}, __file__=cfg.__file__)
        run_cfg.GAMMA, run_cfg.NU, run_cfg.NEL_X, run_cfg.NEL_Y = gamma, nu, nx, ny
        run_cfg.INITIAL_MODULUS_MPA = initial
        matrix, offset = assumed_registration(primary['180N'].coordinates, cfg.GEO_L, cfg.GEO_H, shift_x=shift)
        mesh = _mesh(run_cfg)
        cases = make_cases(run_cfg, mesh, all_platforms[revision], forces, matrix, offset)
        result = lbfgs_inverse_solver_scipy(
            mesh, observation_cases=cases[:1], E_init=np.full(mesh.n_nod, initial),
            gamma=gamma, modulus_scale=cfg.E_REF, max_iter=cfg.MAX_ITER,
            max_fun=cfg.MAX_FUN, ftol=cfg.FTOL, gtol=cfg.GTOL, nu=nu,
            regularization_mode=cfg.REGULARIZATION_MODE, tgv_options=cfg.TGV_OPTIONS)
        states = compare_displacement(mesh, result['E_final'], cases, nu)
        output = args.output / label
        save_run(output, run_cfg, {'coordinates_verified': False, 'variant': label,
                                  'matrix': matrix, 'offset_mm': offset,
                                  'solver': _solver_summary(result), 'revision': revision},
                 {'coordinates': mesh.coord, 'E_inverse': result['E_final'],
                  'cost_history': result['cost_history'], 'gradient_norms': result['gradient_norms']})
        regions = _regional_comparison(mesh, result['E_final'], calibration, cfg.REGION_BOUNDARIES_X)
        regions.to_csv(output / 'regional_modulus_comparison.csv', index=False)
        save_x_profile(output, mesh, result['E_final'], reference)
        for case, state in zip(cases, states):
            rows.append(dict(variant=label, case=case.name, converged=result['converged'],
                             iterations=result['n_iterations'], gamma=gamma, nu=nu,
                             initial_MPa=initial, parameters=mesh.n_nod, **state_metrics(state)))
        pd.DataFrame(rows).to_csv(args.output / 'inverse_sensitivity.csv', index=False)


if __name__ == '__main__':
    main()
