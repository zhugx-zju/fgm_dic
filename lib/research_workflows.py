"""Orchestration for synthetic checks, calibrated forward comparison and DIC inversion."""

from pathlib import Path

import numpy as np
import pandas as pd

from fgm_asm import MeshInfo, lbfgs_inverse_solver_scipy
from fgm_asm.observation import build_observation_operator
from fgm_asm.research import (
    compare_displacement, make_observation_case, partition_modulus, solve_forward,
)
from .experiment_data import (
    file_sha256, load_calibration, load_dic_platform, load_platform_force,
    transform_dic_platform,
)
from .research_reporting import plot_displacement_comparison, plot_modulus_comparison, save_run


def _mesh(cfg, forward=False):
    return MeshInfo(cfg.GEO_L, cfg.GEO_H,
                    cfg.FORWARD_NEL_X if forward else cfg.NEL_X,
                    cfg.FORWARD_NEL_Y if forward else cfg.NEL_Y)


def _inverse(cfg, mesh, cases):
    return lbfgs_inverse_solver_scipy(
        mesh, observation_cases=cases, E_init=np.full(mesh.n_nod, cfg.E_REF),
        gamma=cfg.GAMMA, modulus_scale=cfg.E_REF, max_iter=cfg.MAX_ITER,
        max_fun=cfg.MAX_FUN, ftol=cfg.FTOL, gtol=cfg.GTOL, nu=cfg.NU,
        regularization_mode=cfg.REGULARIZATION_MODE, tgv_options=cfg.TGV_OPTIONS,
    )


def _solver_summary(result):
    return {key: result[key] for key in ('converged', 'message', 'n_iterations',
                                        'n_evaluations', 'final_cost', 'regularization_mode')}


def _comparison_summary(states):
    return [{k: s[k] for k in ('name', 'cost_tar', 'weighted_rms',
                               'rmse_components_mm', 'rigid_coefficients', 'rigid_origin')} for s in states]


def run_synthetic(cfg, output):
    """Generate on a separate mesh; train on one load, predict the other."""
    if Path(output).exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    mesh, fine = _mesh(cfg), _mesh(cfg, forward=True)
    if cfg.TRUTH == 'uniform':
        truth = lambda x, y: np.full_like(x, cfg.E_REF)
    elif cfg.TRUTH == 'smooth_2d':
        truth = lambda x, y: cfg.E_REF * (
            1 + 0.5 * x / cfg.GEO_L + 0.05 * np.sin(np.pi * x / cfg.GEO_L) *
            np.cos(2 * np.pi * y / cfg.GEO_H))
    elif cfg.TRUTH == 'seven_regions':
        truth = partition_modulus(cfg.E_REF * np.array([1., 1.2, 1.1, 1.6, 1.5, 1.8, 2.]))
    else:
        raise ValueError(f'Unknown synthetic TRUTH: {cfg.TRUTH}')
    if len(cfg.FORCES_N) < 2:
        raise ValueError('Synthetic workflow needs training and held-out prediction loads')
    # Point coordinates are independent of both FE grids and cover valid interior.
    point_rng = np.random.default_rng(cfg.SEED)
    xy = point_rng.uniform([0, 0], [cfg.GEO_L, cfg.GEO_H], size=(cfg.N_POINTS, 2))
    cases = []
    for index, force in enumerate(cfg.FORCES_N):
        u, _ = solve_forward(fine, force, cfg.THICKNESS, cfg.NU, truth)
        measured = (build_observation_operator(fine, xy[:, 0], xy[:, 1]) @ u).reshape(-1, 2)
        rigid = np.column_stack([-xy[:, 1], xy[:, 0]]) * cfg.RIGID_ROTATION_RAD
        noise = np.random.default_rng(cfg.SEED + 1000 + index).normal(
            scale=cfg.NOISE_STD_MM, size=measured.shape,
        )
        measured = measured + rigid + np.asarray(cfg.RIGID_TRANSLATION_MM) + noise
        case = make_observation_case(
            mesh, f'load_{index}', force, cfg.THICKNESS, xy, measured,
            np.full(cfg.N_POINTS, cfg.GEO_L * cfg.GEO_H / cfg.N_POINTS),
            metadata={'source': 'synthetic', 'seed': cfg.SEED + 1000 + index,
                      'displacement_reference': 'unloaded'},
        )
        cases.append(case)
    result = _inverse(cfg, mesh, cases[:1])
    states = compare_displacement(mesh, result['E_final'], cases, cfg.NU)
    reference = truth(mesh.X, mesh.Y)
    summary = {'workflow': 'synthetic', 'solver': _solver_summary(result),
               'fit_cases': [cases[0].name], 'predict_cases': [c.name for c in cases[1:]],
               'modulus_mae_percent': float(100 * np.mean(np.abs(result['E_final'] / reference - 1))),
               'comparison': _comparison_summary(states),
               'scope': 'Synthetic consistency check; no experimental feasibility claim.'}
    arrays = {'coordinates': mesh.coord, 'E_reference': reference,
              'E_inverse': result['E_final'], 'observation_coordinates': xy,
              'cost_history': result['cost_history']}
    for case, state in zip(cases, states):
        arrays[f'{case.name}_observed'] = case.observation.u_measured
        arrays[f'{case.name}_predicted'] = state['prediction']
        arrays[f'{case.name}_residual'] = state['residual']
    save_run(output, cfg, summary, arrays)
    for case, state in zip(cases, states):
        plot_displacement_comparison(output, case.name, xy,
                                     case.observation.u_measured.reshape(-1, 2), state)
    plot_modulus_comparison(output, mesh, reference, result['E_final'])
    return summary


def prepare_experiment(cfg, load_fields=True):
    platforms, records = {}, {}
    for name, setting in cfg.PLATFORMS.items():
        if load_fields:
            platforms[name] = load_dic_platform(setting['folder'], *setting['stages'])
        _, records[name] = load_platform_force(cfg.FORCE_PAIRING_CSV, setting['setting_N'])
        if (records[name]['stage_first'], records[name]['stage_last']) != setting['stages']:
            raise ValueError(f'{name}: force window does not match selected DIC frames')
    calibration = load_calibration(cfg.CALIBRATION_CSV, cfg.CALIBRATION_METHOD)
    summary = {'workflow': 'experiment', 'platforms': {k: v.metadata for k, v in platforms.items()},
               'force_pairing': records, 'calibration': calibration.to_dict(orient='records'),
               'calibration_source': {'path': str(cfg.CALIBRATION_CSV),
                                      'sha256': file_sha256(cfg.CALIBRATION_CSV)},
               'force_source': {'path': str(cfg.FORCE_PAIRING_CSV),
                                'sha256': file_sha256(cfg.FORCE_PAIRING_CSV)},
               'coordinates_verified': cfg.DIC_COORDINATES_VERIFIED}
    return platforms, calibration, records, summary


def _experiment_cases(cfg, mesh, platforms, records):
    cases = {}
    for name, platform in platforms.items():
        xy, uv, std = transform_dic_platform(
            platform, cfg.DIC_TO_FEM_MATRIX, cfg.DIC_TO_FEM_OFFSET,
            verified=cfg.DIC_COORDINATES_VERIFIED,
        )
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] <= cfg.GEO_L) &
                 (xy[:, 1] >= 0) & (xy[:, 1] <= cfg.GEO_H))
        if valid.sum() < 4 or np.any(np.ptp(xy[valid], axis=0) == 0):
            raise ValueError(f'{name}: fewer than four spatially separated points in physical ROI')
        if not np.isfinite(cfg.NOISE_FLOOR_MM) or cfg.NOISE_FLOOR_MM <= 0:
            raise ValueError('NOISE_FLOOR_MM must be positive and finite')
        # Grid has uniform spacing in its checked similarity transform. Cell
        # areas come from the measured spacing, not from normalizing the ROI.
        distances = []
        for axis in (0, 1):
            order = np.lexsort((platform.grid_ids[:, 1-axis], platform.grid_ids[:, axis]))
            ids, points = platform.grid_ids[order], xy[order]
            delta = np.diff(ids, axis=0)
            adjacent = (delta[:, axis] == 0) & (np.abs(delta[:, 1-axis]) == 1)
            if not adjacent.any():
                raise ValueError('Cannot infer DIC grid spacing for area weights')
            distances.append(np.median(np.linalg.norm(np.diff(points, axis=0)[adjacent], axis=1)))
        area = distances[0] * distances[1]
        metadata = dict(platform.metadata, force_pairing=records[name],
                        points_inside_roi=int(valid.sum()), points_outside_roi=int((~valid).sum()))
        sigma = np.maximum(std[valid], cfg.NOISE_FLOOR_MM)[:, None]
        cases[name] = make_observation_case(
            mesh, name, records[name]['mean_corrected_force_N'], cfg.THICKNESS,
            xy[valid], uv[valid], np.full(valid.sum(), area), sigma, metadata,
        )
    return cases


def _regional_comparison(mesh, field, calibration, boundaries):
    values = (field[mesh.ele_nods_id - 1] @ mesh.gauss_N.T).T
    x = np.einsum('en,gn->ge', mesh.x_ele, mesh.gauss_N)
    weights = mesh.gauss_abs_det_j * mesh.gauss_w[:, None]
    groups = np.searchsorted(boundaries, x, side='right')
    rows = []
    for index, record in enumerate(calibration.to_dict(orient='records')):
        mask = groups == index
        if not mask.any():
            raise ValueError('Inversion mesh does not resolve all seven calibration regions')
        average = float(np.average(values[mask], weights=weights[mask]))
        spread = float(np.sqrt(np.average((values[mask] - average)**2, weights=weights[mask])))
        rows.append(dict(record, inverse_mean_E_MPa=average, inverse_spatial_sd_MPa=spread,
                         difference_percent=100 * (average / record['mean_E_MPa'] - 1)))
    return pd.DataFrame(rows)


def run_experiment(cfg, output, mode='inspect'):
    """Inspect needs no registration; quantitative modes require verified coordinates."""
    if Path(output).exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if mode not in {'inspect', 'model', 'forward', 'inverse'}:
        raise ValueError(f'Unknown experiment mode {mode}')
    platforms, calibration, records, summary = prepare_experiment(cfg, load_fields=(mode != 'model'))
    summary['mode'] = mode
    if mode == 'model':
        # Independent calibrated forward prediction is available before DIC
        # registration. It is not yet a quantitative comparison with DIC.
        mesh = _mesh(cfg, forward=True)
        reference = partition_modulus(calibration['mean_E_MPa'].to_numpy(), cfg.REGION_BOUNDARIES_X)
        arrays = {'coordinates': mesh.coord, 'E_calibrated': reference(mesh.X, mesh.Y)}
        for name, record in records.items():
            arrays[f'{name}_U'], _ = solve_forward(mesh, record['mean_corrected_force_N'],
                                                  cfg.THICKNESS, cfg.NU, reference)
        summary['status'] = 'calibrated_model_prediction_only'
        summary['scope'] = 'Independent calibrated rectangle model; DIC comparison pending registration.'
        save_run(output, cfg, summary, arrays)
        return summary
    if mode == 'inspect':
        summary['status'] = ('ready_for_registration_check' if not cfg.DIC_COORDINATES_VERIFIED
                             else 'registered_input_inspection')
        arrays = {}
        for name, platform in platforms.items():
            arrays[f'{name}_coordinates'] = platform.coordinates
            arrays[f'{name}_displacement_mean'] = platform.displacement
            arrays[f'{name}_temporal_std'] = platform.temporal_std
        save_run(output, cfg, summary, arrays)
        return summary

    mesh = _mesh(cfg, forward=(mode == 'forward'))
    cases = _experiment_cases(cfg, mesh, platforms, records)
    reference = partition_modulus(calibration['mean_E_MPa'].to_numpy(), cfg.REGION_BOUNDARIES_X)
    arrays = {'coordinates': mesh.coord, 'E_calibrated': reference(mesh.X, mesh.Y)}
    calibrated_mesh = mesh if mode == 'forward' else _mesh(cfg, forward=True)
    calibrated_cases = (cases if mode == 'forward' else
                        _experiment_cases(cfg, calibrated_mesh, platforms, records))
    calibrated = compare_displacement(calibrated_mesh, reference,
                                      list(calibrated_cases.values()), cfg.NU)
    arrays['calibrated_coordinates'] = calibrated_mesh.coord
    arrays['E_calibrated_forward'] = reference(calibrated_mesh.X, calibrated_mesh.Y)
    summary['calibrated_forward_comparison'] = _comparison_summary(calibrated)
    states, prefix = calibrated, 'calibrated'
    if mode == 'inverse':
        if set(cfg.FIT_CASES) & set(cfg.PREDICT_CASES):
            raise ValueError('FIT_CASES and PREDICT_CASES must be disjoint')
        result = _inverse(cfg, mesh, [cases[k] for k in cfg.FIT_CASES])
        states = compare_displacement(mesh, result['E_final'], list(cases.values()), cfg.NU)
        summary['solver'] = _solver_summary(result)
        summary['inverse_comparison'] = _comparison_summary(states)
        summary['fit_cases'], summary['predict_cases'] = cfg.FIT_CASES, cfg.PREDICT_CASES
        arrays.update(E_inverse=result['E_final'], cost_history=result['cost_history'])
        prefix = 'inverse'
    for case, calibrated_state, state in zip(cases.values(), calibrated, states):
        arrays[f'{case.name}_coordinates'] = np.column_stack([case.observation.x_meas,
                                                             case.observation.y_meas])
        arrays[f'{case.name}_observed'] = case.observation.u_measured
        arrays[f'{case.name}_calibrated_prediction'] = calibrated_state['prediction']
        arrays[f'{case.name}_calibrated_U'] = calibrated_state['U']
        arrays[f'{case.name}_{prefix}_prediction'] = state['prediction']
        arrays[f'{case.name}_{prefix}_residual'] = state['residual']
        arrays[f'{case.name}_{prefix}_U'] = state['U']
    summary['status'] = ('not_converged' if mode == 'inverse' and not result['converged']
                         else 'computed')
    summary['scope'] = ('Rectangle two-end uniform traction model; Poisson ratio, calibration '
                        'transfer and full-specimen boundary equivalence remain to be validated.')
    summary['observation_metadata'] = {name: c.metadata for name, c in cases.items()}
    save_run(output, cfg, summary, arrays)
    for case, calibrated_state, state in zip(cases.values(), calibrated, states):
        xy = arrays[f'{case.name}_coordinates']
        uv = case.observation.u_measured.reshape(-1, 2)
        plot_displacement_comparison(output, f'{case.name}_calibrated', xy, uv, calibrated_state)
        if mode == 'inverse':
            plot_displacement_comparison(output, f'{case.name}_inverse', xy, uv, state)
    if mode == 'inverse':
        _regional_comparison(mesh, result['E_final'], calibration, cfg.REGION_BOUNDARIES_X).to_csv(
            Path(output) / 'regional_modulus_comparison.csv', index=False, encoding='utf-8-sig',
        )
        plot_modulus_comparison(output, mesh, arrays['E_calibrated'], result['E_final'])
    return summary
