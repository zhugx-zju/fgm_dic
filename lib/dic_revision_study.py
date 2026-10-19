"""Revised DIC audit and conditional inference using the shared mechanics kernel.

This module never enables formal DIC_COORDINATES_VERIFIED. Missing landmarks
are represented by explicit alternative registrations, without scale fitting.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fgm_asm.research import compare_displacement, make_observation_case, partition_modulus
from .dic_noise import analyze_dic_series
from .experiment_data import (check_platform_reference, dic_roi_mask, load_calibration,
                              load_dic_frame_series, load_dic_platform, load_platform_force)
from .research_reporting import (plot_displacement_comparison, plot_modulus_comparison,
                                 save_run, save_x_profile)
from .dic_revision_reporting import save_conditional_comparison
from .research_workflows import _inverse, _mesh, _regional_comparison, _solver_summary


def assumed_registration(reference_xy, length, width, direction=1, shift_x=0.):
    """An assumption for sensitivity calculations, never a verified transform.

    Keep the mm unit scale exactly one. Centering is explicitly assumed rather
    than inferred as a physical correspondence. The same reference must be used
    for all platforms and revisions in a comparison.
    """
    if direction not in (-1, 1):
        raise ValueError('direction must be +1 or -1')
    reference_xy = np.asarray(reference_xy, dtype=float)
    if reference_xy.ndim != 2 or reference_xy.shape[1] != 2 or not np.isfinite(reference_xy).all():
        raise ValueError('Expected finite reference coordinates (points,2)')
    matrix = np.array([[0., float(direction)], [1., 0.]])
    center = (reference_xy.min(axis=0) + reference_xy.max(axis=0)) / 2
    offset = np.array([length / 2 + shift_x, width / 2]) - matrix @ center
    return matrix, offset


def block_split(xy, block_mm=6., buffer_mm=1.):
    """Spatial validation tiles with a guard band in the training observations.

    Uses coordinates only, never modulus truth or the held-out 280N response.
    Validation is every fourth diagonal tile; Euclidean distance to the entire
    validation set excludes nearby training points to reduce spatial leakage.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype=float)
    if block_mm <= 0 or buffer_mm < 0:
        raise ValueError('Positive blocks and a nonnegative buffer are required')
    tiles = np.floor(xy / block_mm).astype(int)
    validation = (tiles[:, 0] + 2 * tiles[:, 1]) % 4 == 1
    if not validation.any() or validation.all():
        raise ValueError('Spatial split needs both training and validation points')
    distance = cKDTree(xy[validation]).query(xy)[0]
    train = (~validation) & (distance > buffer_mm)
    if train.sum() < 4:
        raise ValueError('Too few training points after spatial buffering')
    return train, validation


def grid_cell_area(platform):
    """Measured reference grid spacing; no normalization to specimen area."""
    distances = []
    for axis in (0, 1):
        order = np.lexsort((platform.grid_ids[:, 1-axis], platform.grid_ids[:, axis]))
        ids, points = platform.grid_ids[order], platform.coordinates[order]
        delta = np.diff(ids, axis=0)
        adjacent = (delta[:, axis] == 0) & (np.abs(delta[:, 1-axis]) == 1)
        if not adjacent.any():
            raise ValueError('Cannot determine physical grid cell area')
        distances.append(np.median(np.linalg.norm(np.diff(points, axis=0)[adjacent], axis=1)))
    return float(np.prod(distances))


def make_cases(cfg, mesh, platforms, forces, matrix, offset, subset=None):
    """Explicit conditional observations; no formal registration flag is set."""
    cases = []
    for name, platform in platforms.items():
        xy = platform.coordinates @ matrix.T + offset
        uv = platform.displacement @ matrix.T
        valid = dic_roi_mask(xy, cfg.GEO_L, cfg.GEO_H)
        if subset is not None:
            valid &= subset[name]
        if valid.sum() < 4:
            raise ValueError(f'{name}: insufficient observations in conditional ROI')
        area = grid_cell_area(platform)
        case = make_observation_case(
            mesh, name, forces[name]['mean_corrected_force_N'], cfg.THICKNESS,
            xy[valid], uv[valid], np.full(valid.sum(), area), cfg.WORKING_SIGMA_MM,
            metadata={'registration_status': cfg.REGISTRATION_STATUS,
                      'coordinates_verified': False, 'assumption': cfg.REGISTRATION_ASSUMPTION,
                      'matrix': matrix.tolist(), 'offset_mm': offset.tolist(),
                      'cell_area_mm2': area, 'points_inside': int(valid.sum()),
                      'force_pairing': forces[name],
                      'weighting': 'equal measured cell areas / constant working sigma squared'},
        )
        cases.append(case)
    return cases


def state_metrics(state):
    """Physical residual summaries; no pass threshold or confidence level."""
    error = state['residual'].reshape(-1, 2)
    return {'weighted_rms_mm': float(state['weighted_rms']),
            'rmse_ux_mm': float(np.sqrt(np.mean(error[:, 0]**2))),
            'rmse_uy_mm': float(np.sqrt(np.mean(error[:, 1]**2))),
            'p95_ux_mm': float(np.quantile(np.abs(error[:, 0]), .95)),
            'p95_uy_mm': float(np.quantile(np.abs(error[:, 1]), .95))}


def audit_inputs(cfg):
    platforms, forces, rows, noise = {}, {}, [], {}
    for revision in ('', '_2', '_3'):
        platforms[revision] = {}
        for name, stages in cfg.STAGES.items():
            folder = cfg.DATA_ROOT / (name + revision)
            platform = load_dic_platform(folder, *stages)
            platforms[revision][name] = platform
            if name not in forces:
                _, forces[name] = load_platform_force(cfg.FORCE_PAIRING_CSV, int(name[:-1]))
                if tuple(stages) != (forces[name]['stage_first'], forces[name]['stage_last']):
                    raise ValueError('DIC frame selection does not match paired force window')
            span = np.ptp(platform.coordinates, axis=0)
            rows.append({'folder': folder.name, 'selected_frames': platform.metadata['frame_count'],
                         'available_frames': len(list(folder.glob('stage_*.csv'))),
                         'points': len(platform.coordinates), 'width_X_mm': span[0],
                         'length_Y_mm': span[1], 'cell_area_mm2': grid_cell_area(platform),
                         'raw_temporal_sd_DX_mm': np.median(platform.temporal_std[:, 0]),
                         'raw_temporal_sd_DY_mm': np.median(platform.temporal_std[:, 1])})
            if revision == cfg.PRIMARY_REVISION:
                noise[name], _ = analyze_dic_series(load_dic_frame_series(folder, *stages),
                                                    cfg.FRAME_RATE_HZ)
                noise[name]['unit'] = 'mm'
        check_platform_reference(platforms[revision])
    return platforms, forces, pd.DataFrame(rows), noise


def run_study(cfg, output, smoke=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    platforms, forces, audit, noise = audit_inputs(cfg)
    primary = platforms[cfg.PRIMARY_REVISION]
    matrix, offset = assumed_registration(primary[cfg.FIT_CASE].coordinates, cfg.GEO_L, cfg.GEO_H)
    calibration = load_calibration(cfg.CALIBRATION_CSV, cfg.CALIBRATION_METHOD, cfg.REGION_INFILL_PERCENT)
    reference = partition_modulus(calibration.mean_E_MPa, cfg.REGION_BOUNDARIES_X)
    summary = {'workflow': 'revised_dic_conditional_study', 'coordinates_verified': False,
               'registration_status': cfg.REGISTRATION_STATUS,
               'registration_assumption': cfg.REGISTRATION_ASSUMPTION,
               'matrix': matrix, 'offset_mm': offset,
               'scope': 'Exploratory under explicit registration assumptions; not formal physical validation.',
               'force_pairing': forces, 'noise': noise,
               'platforms': {rev or 'original': {name: p.metadata for name, p in fields.items()}
                             for rev, fields in platforms.items()}}
    save_run(output, cfg, summary)
    audit.to_csv(output / 'input_audit.csv', index=False)
    fine, mesh = _mesh(cfg, forward=True), _mesh(cfg)
    fine_cases = make_cases(cfg, fine, primary, forces, matrix, offset)
    calibrated = compare_displacement(fine, reference, fine_cases, cfg.NU)
    forward_arrays = {'coordinates': fine.coord, 'E_calibrated': reference(fine.X, fine.Y)}
    for case, state in zip(fine_cases, calibrated):
        xy = np.column_stack([case.observation.x_meas, case.observation.y_meas])
        forward_arrays[f'{case.name}_xy'] = xy
        forward_arrays[f'{case.name}_observed'] = case.observation.u_measured
        forward_arrays[f'{case.name}_predicted'] = state['prediction']
        forward_arrays[f'{case.name}_U'] = state['U']
        plot_displacement_comparison(output, case.name + '_independent', xy,
                                     case.observation.u_measured.reshape(-1, 2), state)
    save_run(output / 'independent_forward', cfg,
             dict(summary, comparison=[dict(case=s['name'], **state_metrics(s)) for s in calibrated]),
             forward_arrays)
    # Inspect assumed placement and direction without choosing them by fit quality.
    forward_sensitivity = []
    for revision, direction, shift in (('_3', 1, -4.), ('_3', 1, 0.), ('_3', 1, 4.),
                                        ('_3', -1, 0.), ('_2', 1, 0.)):
        a, b = assumed_registration(primary[cfg.FIT_CASE].coordinates, cfg.GEO_L, cfg.GEO_H,
                                    direction, shift)
        cases = make_cases(cfg, fine, platforms[revision], forces, a, b)
        for state in compare_displacement(fine, reference, cases, cfg.NU):
            forward_sensitivity.append(dict(revision=revision, direction=direction, shift_x_mm=shift,
                                            case=state['name'], **state_metrics(state)))
    pd.DataFrame(forward_sensitivity).to_csv(output / 'forward_registration_sensitivity.csv', index=False)
    xy = primary[cfg.FIT_CASE].coordinates @ matrix.T + offset
    train, validation = block_split(xy, cfg.VALIDATION_BLOCK_MM, cfg.VALIDATION_BUFFER_MM)
    fit_platform = {cfg.FIT_CASE: primary[cfg.FIT_CASE]}
    train_cases = make_cases(cfg, mesh, fit_platform, forces, matrix, offset, {cfg.FIT_CASE: train})
    validation_cases = make_cases(cfg, mesh, fit_platform, forces, matrix, offset,
                                 {cfg.FIT_CASE: validation})
    records = []
    candidates = cfg.GAMMA_CANDIDATES[:1] if smoke else cfg.GAMMA_CANDIDATES
    for gamma in candidates:
        run_cfg = SimpleNamespace(**{k: v for k, v in vars(cfg).items() if k.isupper()},
                                  __file__=cfg.__file__)
        run_cfg.GAMMA = gamma
        result = _inverse(run_cfg, mesh, train_cases)
        score = compare_displacement(mesh, result['E_final'], validation_cases, cfg.NU)[0]
        record = dict(gamma=gamma, converged=result['converged'],
                      iterations=result['n_iterations'], train_points=int(train.sum()),
                      validation_points=int(validation.sum()), **state_metrics(score))
        records.append(record)
        save_run(output / f'validation_gamma_{gamma:g}', run_cfg,
                 dict(summary, solver=_solver_summary(result), validation=record),
                 {'E_inverse': result['E_final'], 'cost_history': result['cost_history'],
                  'train_mask': train, 'validation_mask': validation})
        pd.DataFrame(records).to_csv(output / 'gamma_validation.csv', index=False)
    eligible = [r for r in records if r['converged']]
    if not eligible:
        raise RuntimeError('No validation candidate converged; inspect saved diagnostics before full inference')
    chosen = min(eligible, key=lambda r: r['weighted_rms_mm'])['gamma']
    final_cfg = SimpleNamespace(**{k: v for k, v in vars(cfg).items() if k.isupper()},
                                __file__=cfg.__file__)
    final_cfg.GAMMA = chosen
    full_cases = make_cases(cfg, mesh, primary, forces, matrix, offset)
    result = _inverse(final_cfg, mesh, [c for c in full_cases if c.name == cfg.FIT_CASE])
    states = compare_displacement(mesh, result['E_final'], full_cases, cfg.NU)
    arrays = {'coordinates': mesh.coord, 'E_inverse': result['E_final'],
              'E_reference': reference(mesh.X, mesh.Y), 'cost_history': result['cost_history'],
              'cost_tar_history': result['cost_tar_history'], 'cost_reg_history': result['cost_reg_history'],
              'gradient_norms': result['gradient_norms'], 'final_gradient': result['final_gradient']}
    for case, state, baseline in zip(full_cases, states, calibrated):
        arrays[f'{case.name}_coordinates'] = np.column_stack([case.observation.x_meas, case.observation.y_meas])
        arrays[f'{case.name}_observed'] = case.observation.u_measured
        arrays[f'{case.name}_inverse_prediction'] = state['prediction']
        arrays[f'{case.name}_calibrated_prediction'] = baseline['prediction']
        arrays[f'{case.name}_inverse_U'] = state['U']
    final_summary = dict(summary, selected_gamma=chosen,
                         selection_rule='Minimum buffered spatial validation RMS on 180N only among converged candidates',
                         fit_cases=[cfg.FIT_CASE], predict_cases=[cfg.PREDICT_CASE],
                         solver=_solver_summary(result), inverse_parameter_count=mesh.n_nod,
                         inverse_dimension='independent nodal E(x,y)',
                         comparison=[dict(case=s['name'], **state_metrics(s)) for s in states])
    final_output = output / 'inverse_primary'
    save_run(final_output, final_cfg, final_summary, arrays)
    save_conditional_comparison(final_output, full_cases, calibrated, states, [cfg.FIT_CASE])
    _regional_comparison(mesh, result['E_final'], calibration, cfg.REGION_BOUNDARIES_X).to_csv(
        final_output / 'regional_modulus_comparison.csv', index=False)
    plot_modulus_comparison(final_output, mesh, arrays['E_reference'], result['E_final'])
    save_x_profile(final_output, mesh, result['E_final'], reference)
    return final_summary
